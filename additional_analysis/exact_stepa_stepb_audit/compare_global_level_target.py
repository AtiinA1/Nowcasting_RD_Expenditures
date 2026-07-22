"""Evaluate the reported AGT architecture with a globally standardized level target."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "additional_analysis" / "robustness_overfit"))
sys.path.insert(0, str(HERE))

from common import annual_metrics, prepare_data  # noqa: E402
from run_audit import DROPOUT, ENSEMBLE_SIZE, HIDDEN, LR, WEIGHT_DECAY, WideDeepNN, tensor  # noqa: E402


def main():
    prep = prepare_data()
    frame = prep.frame
    columns = prep.configs["AGT"]
    train, val, test = (prep.masks[name] for name in ("train", "val", "test"))
    raw = frame[columns].fillna(0).astype(float).to_numpy()
    scaler = StandardScaler().fit(raw[train])
    x = np.column_stack([scaler.transform(raw), prep.months.to_numpy()])
    encoder = LabelEncoder()
    countries = encoder.fit_transform(frame.Country)

    target = frame.rd_expenditure.to_numpy(dtype=float)
    target_mean = float(target[train].mean())
    target_sd = float(target[train].std())
    standardized_target = (target - target_mean) / target_sd

    xtr, xval = tensor(x[train]), tensor(x[val])
    ctr = torch.as_tensor(countries[train], dtype=torch.long)
    cval = torch.as_tensor(countries[val], dtype=torch.long)
    ytr = tensor(standardized_target[train].reshape(-1, 1))
    criterion = nn.SmoothL1Loss(beta=0.5)
    members = []
    histories = []

    for seed in range(ENSEMBLE_SIZE):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = WideDeepNN(x.shape[1], len(encoder.classes_))
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best = np.inf
        best_state = None
        bad = 0
        for epoch in range(500):
            model.train()
            permutation = torch.randperm(len(xtr))
            for start in range(0, len(xtr), 64):
                idx = permutation[start : start + 64]
                optimizer.zero_grad()
                loss = criterion(model(xtr[idx], ctr[idx]), ytr[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_prediction = model(xval, cval).numpy().ravel() * target_sd + target_mean
            score = annual_metrics(frame[val].assign(pred=validation_prediction), "pred")["MAPE"]
            histories.append({"seed": seed, "epoch": epoch + 1, "val_MAPE": score})
            if score < best - 1e-5:
                best = score
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 65:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prediction = (
                model(tensor(x[test]), torch.as_tensor(countries[test], dtype=torch.long)).numpy().ravel()
                * target_sd
                + target_mean
            )
        members.append(prediction)
        print(f"Level-target seed {seed + 1}/{ENSEMBLE_SIZE}: best validation MAPE={best:.4f}", flush=True)

    member_matrix = np.column_stack(members)
    result = frame[test][["Country", "Year", "Month", "rd_expenditure"]].copy()
    result["prediction"] = member_matrix.mean(axis=1)
    for seed in range(ENSEMBLE_SIZE):
        result[f"member_{seed}"] = member_matrix[:, seed]
    metrics = annual_metrics(result, "prediction")
    metrics.update(
        {
            "target": "globally standardized GERD level",
            "target_train_mean": target_mean,
            "target_train_sd": target_sd,
        }
    )
    pd.DataFrame([metrics]).to_csv(OUT / "global_level_target_agt_metrics.csv", index=False)
    result.to_csv(OUT / "global_level_target_agt_predictions.csv", index=False)
    pd.DataFrame(histories).to_csv(OUT / "global_level_target_training_history.csv", index=False)
    print("\nGlobal-level Step A AGT metrics:\n", pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
