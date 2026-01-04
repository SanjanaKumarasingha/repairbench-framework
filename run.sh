clear
./clean_update_cache.sh

# Generate samples using codellama model and parameters
#python3 generate_samples.py \
#    defects4j \
#    infilling \
#    --model-name codellama

# # Generate patches using the specified model and parameters
# python3 generate_patches.py \
#   samples_defects4j_infilling_model_name_codellama.jsonl \
#   repairllama-infilling \
#   --model_name "/home/cse_g3/Documents/FYP/APRVERSION3/repairbench-framework/fine-tune/v7" \
#   --n_workers 1 \
#   --num_return_sequences 10 \
#   --num_beams 10   \
#   --max_new_tokens 64

# python3 evaluate_patches.py \
#     defects4j candidates_defects4j_infilling_repairllama-infilling_model_name=v7_num_return_sequences=10_num_beams=10_max_new_tokens=64.jsonl \
#     replace \
#     --n_workers 1

# python3 export_results.py \
#     defects4j evaluation_defects4j_infilling_repairllama-infilling.jsonl \
#     --model_name "/home/cse_g3/Documents/FYP/APRVERSION3/repairbench-framework/fine-tune/v7"




# # Genarate patches using repairllama model and parameters
# python3 generate_samples.py \
#     defects4j \
#     infilling \
#     --model-name codellama

# # ----------------- Specific Model RepairLLaMA-IR3-OR2 - Beam Search ------------------------
#  python3 generate_patches.py  \
#     samples_defects4j_infilling_model_name_codellama.jsonl   \
#     repairllama-infilling   \
#     --model_name "ASSERT-KTH/RepairLLaMA-IR3-OR2"   \
#     --n_workers 1   \
#     --num_return_sequences 10   \
#     --num_beams 10   \
#     --max_new_tokens 64


python3 evaluate_patches.py \
   defects4j candidates_defects4j_infilling_repairllama-infilling_model_name=ASSERT-KTH-RepairLLaMA-IR3-OR2_num_return_sequences=10_num_beams=10_max_new_tokens=64.jsonl \
   replace \
   --n_workers 1

python export_results.py \
   defects4j evaluation_defects4j_infilling_repairllama-infilling.jsonl \
   --model_name "ASSERT-KTH/RepairLLaMA-IR3-OR2"






# python3 generate_patches.py \
#     samples_defects4j_infilling_model_name_codellama.jsonl \
#     repairllama-infilling \
#     --model_name "ASSERT-KTH/RepairLLaMA-IR3-OR2" \
#     --n_workers 1 \
#     --num_return_sequences 1 \
#     --temperature 0.7

