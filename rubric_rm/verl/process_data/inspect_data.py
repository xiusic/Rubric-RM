from datasets import load_dataset
ds = load_dataset("infly/INF-ORM-Preference-Magnitude-80K")['train']


for item in ds:
    
print(ds[0])
