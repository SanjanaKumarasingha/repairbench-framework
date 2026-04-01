# ---- CodeT5-Small ----

# Genarate patches using repairllama model and parameters
python3 generate_samples.py \
    defects4j \
    infilling \
    --model-name Salesforce/codet5-small

python3 generate_patches.py  \
    samples_defects4j_infilling_model_name_Salesforce-codet5-small.jsonl   \
    codet5-small_infilling  \
    --model_name "Salesforce/codet5-small"  \
    --n_workers 1   \
    --num_return_sequences 10  \
    --num_beams 10   \
    --max_new_tokens 64

python3 evaluate_patches.py \
    defects4j candidates_defects4j_infilling_codet5-small_infilling_model_name=Salesforce-codet5-small_num_return_sequences=10_num_beams=10_max_new_tokens=64.jsonl \
    replace \
    --n_workers 1

python3 export_results.py \
    defects4j evaluation_defects4j_infilling_codet5-small.jsonl \
    --model_name "Salesforce/codet5-small"  


