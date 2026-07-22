.PHONY: check step-a benchmarks employment

check:
	python scripts/validate_repository.py

step-a:
	python additional_analysis/robustness_overfit/10_all_configs_updated_pure_nn.py

benchmarks:
	python additional_analysis/pre_raw_stepb_method_audit/leakage_free_midas_benchmarks.py

employment:
	python additional_analysis/employment_inference_robustness/year_block_hac_inference.py
