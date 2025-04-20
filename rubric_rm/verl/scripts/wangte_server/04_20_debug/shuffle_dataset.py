from datasets import load_dataset 


ds = load_dataset("gaotang/filtered_sky_code_8k_math_10k_rubric_evidence_classify_weight_rest_0417")
ds = ds.shuffle(seed=520)
ds.push_to_hub("gaotang/filtered_sky_code_8k_math_10k_rubric_evidence_classify_weight_rest_0420_shuffle")

