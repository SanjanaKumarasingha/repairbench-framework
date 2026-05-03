# ---- CodeT5-Base ----

# Genarate patches using repairllama model and parameters
python3 generate_samples.py \
    HumanEvalJava \
    infilling \
    --model-name Salesforce/codet5-large

python3 generate_patches.py  \
    samples_HumanEvalJava_infilling_model_name_Salesforce-codet5-large.jsonl   \
    codet5-small_infilling  \
    --model_name "Salesforce/codet5-base"  \
    --n_workers 1   \
    --num_return_sequences 10  \
    --num_beams 10   \
    --max_new_tokens 256

python3 evaluate_patches.py \
    HumanEvalJava \
    candidates_HumanEvalJava_infilling_codet5-small_infilling_model_name=Salesforce-codet5-base_num_return_sequences=10_num_beams=10_max_new_tokens=256.jsonl \
    replace \
    --n_workers 1

python3 export_results.py \
    HumanEvalJava \
    evaluation_HumanEvalJava_infilling_codet5-small.jsonl \
    --model_name "Salesforce/codet5-base"  