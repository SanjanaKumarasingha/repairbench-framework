clear
./clean_update_cache.sh
# python3 run_full_mcts_pipeline.py   --benchmark defects4j   --prompt-strategy infilling   --sample-model-name codellama   --patch-strategy repairllama-infilling   --patch-model-name "/home/cse_g3/Documents/FYP/APRVERSION3/repairbench-framework/fine-tune/v7"   --outer-iters 10   --mcts-iters 20   --exploration 1   --rollout-depth 5   --seed 1   --mcts-out-dir mcts_tree_output-sajithtet33 --export

# Command '['python3', 'generate_patches.py', 
# 'samples_defects4j_infilling_model_name_codellama.jsonl', 'repairllama-infilling', '--model_name', 'Salesforce/codet5-small', '--n_workers', '1', '--num_return_sequences', '10', '--num_beams', '10', '--max_new_tokens', '64']' returned non-zero exit status 1.

# python script.py   --input-dir ./MCTS/codet5-small-v1   --output-dir ./MCTS/output/codet5Large   --file-prefix "statistics_defects4j_infilling_ codet5-small_"   --extension .json   --k 10

## RepairLLama
./scripts/repairllama.sh

# ## codet5-small
# ./scripts/codet5_small.sh

# ## codet5-large
# ./scripts/codet5_large.sh

# ## gpt-large
# ./scripts/gpt2_large.sh