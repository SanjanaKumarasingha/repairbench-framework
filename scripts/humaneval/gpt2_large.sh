
# ---- GPT2-Large ----

# Genarate patches using repairllama model and parameters
python3 generate_samples.py \
    defects4j \
    infilling \
    --model-name codellama

python3 generate_patches.py  \
    samples_defects4j_infilling_model_name_codellama.jsonl   \
    gpt2_infilling  \
    --model_name "openai-community/gpt2-large"  \
    --n_workers 1   \
    --num_return_sequences 10  \
    --num_beams 1   \
    --max_new_tokens 256

python3 evaluate_patches.py \
    defects4j candidates_defects4j_infilling_gpt2_infilling_model_name=openai-community-gpt2-large_num_return_sequences=10_num_beams=1_max_new_tokens=256.jsonl \
    replace \
    --n_workers 1

python3 export_results.py \
    defects4j evaluation_defects4j_infilling_gpt2.jsonl \
    --model_name "openai-community/gpt2-large"  
