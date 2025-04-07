from datasets import load_dataset
ds = load_dataset("gaotang/sky_v02_filtered_2_5kcode_18kmath_evidence_evaluation_justify_rubric")['train']


# for item in ds:
    
print(ds[0]['context_messages'][0]['content'])
