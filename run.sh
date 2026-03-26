clear
./clean_update_cache.sh

# Genarate patches using repairllama model and parameters
python3 generate_samples.py \
    defects4j \
    infilling \
    --model-name codellama
    --n_workers 5

# ----------------- Specific Model RepairLLaMA-IR3-OR2 - Beam Search ------------------------
python3 generate_patches.py  \
    samples_defects4j_infilling_model_name_codellama.jsonl   \
    repairllama-infilling   \
    --model_name "ASSERT-KTH/RepairLLaMA-IR3-OR2"   \
    --n_workers 5   \
    --num_return_sequences 1   \
    --num_beams 1   \
    --max_new_tokens 64

python3 evaluate_patches.py \
   defects4j candidates_defects4j_infilling_repairllama-infilling_model_name=ASSERT-KTH-RepairLLaMA-IR3-OR2_num_return_sequences=1_num_beams=1_max_new_tokens=64.jsonl \
   replace \
   --n_workers 5

python3 export_results.py \
   defects4j evaluation_defects4j_infilling_repairllama-infilling.jsonl \
   --model_name "ASSERT-KTH/RepairLLaMA-IR3-OR2"

