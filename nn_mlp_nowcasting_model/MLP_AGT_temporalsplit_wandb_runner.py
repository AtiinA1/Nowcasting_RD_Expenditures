import os
import wandb

# Reuse the full training pipeline from AGT wandb version (base)
import MLP_AGT_temporalsplit_wandb as base

_orig_init_wandb = base.init_wandb

def init_wandb(config=None):
    """
    Wrapper to allow passing a fixed config via environment for the AGT variant.
    This does not change defaults if no env override is provided.
    """
    cfg_override = {} if config is None else dict(config)
    # Optional: allow fixed config via env
    fixed_path = os.getenv("FIXED_CONFIG_PATH")
    fixed_json = os.getenv("FIXED_CONFIG_JSON")
    try:
        if fixed_path and os.path.exists(fixed_path):
            import json
            with open(fixed_path, "r") as f:
                cfg_override.update(json.load(f))
        elif fixed_json:
            import json
            cfg_override.update(json.loads(fixed_json))
    except Exception as e:
        print(f"[WARN] Failed to load fixed config override: {e}")
    # Tag this as AGT explicitly
    cfg_override['dataset_variant'] = 'AGT'
    cfg = _orig_init_wandb(cfg_override)
    run = wandb.run
    current_tags = list(run.tags) if getattr(run, "tags", None) else []
    run.tags = current_tags + ["AGT"]
    return cfg

# Monkey-patch the base init
base.init_wandb = init_wandb

if __name__ == "__main__":
    base.main()




