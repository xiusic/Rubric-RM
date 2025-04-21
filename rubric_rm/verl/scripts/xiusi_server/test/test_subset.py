from datasets import load_dataset 

ds = load_dataset("gaotang/filtered_sky_code_8k_math_10k_rubric_reasoning", split="train")
subset_ds = ds.select(range(32))
subset_ds.push_to_hub("gaotang/filtered_sky_code_8k_math_10k_rubric_reasoning_test")


ds2 = load_dataset("gaotang/filtered_sky_code_8k_math_10k_rubric_evidence_classify_weight_rest_0420_shuffle", split="train")
subset_ds = ds2.select(range(32))

subset_ds.push_to_hub("gaotang/filtered_sky_code_8k_math_10k_rubric_evidence_classify_weight_rest_0420_shuffle_test")