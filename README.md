# ELLM-FT
This is the implementation code of the paper "[AAAI 2025] Evolutionary Large Language Model for Automated Feature Transformation"

## Implementation
### Step 1: download the data: 
```
follow the instruction in /data/readme.md
```

### Step 2: construct the database
```
python3 xxx/datacollection/rl_data_collector.py --file-name DATASETNAME 
```
### Step 3: LLM-based feature transformation operation sequence search 
```
python3 xxx/llm_generation.py ---task_name DATASETNAME --task_type DATASETTYPE
```

