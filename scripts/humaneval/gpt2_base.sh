
# ---- GPT2-Base ----

# Genarate patches using repairllama model and parameters
python3 generate_samples.py \
    HumanEvalJava \
    infilling \
    --model-name codellama

python3 generate_patches.py  \
    samples_HumanEvalJava_infilling_model_name_codellama.jsonl   \
    gpt2_infilling  \
    --model_name "openai-community/gpt2-base"  \
    --n_workers 1   \
    --num_return_sequences 10  \
    --num_beams 1   \
    --max_new_tokens 256

python3 evaluate_patches.py \
    HumanEvalJava candidates_HumanEvalJava_infilling_gpt2_infilling_model_name=openai-community-gpt2-base_num_return_sequences=10_num_beams=1_max_new_tokens=256.jsonl \
    replace \
    --n_workers 1

python3 export_results.py \
    HumanEvalJava evaluation_HumanEvalJava_infilling_gpt2.jsonl \
    --model_name "openai-community/gpt2-base"  
