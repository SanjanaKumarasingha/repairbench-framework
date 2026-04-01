# ---- RepairLLaMA ----

# Genarate patches using repairllama model and parameters
python3 generate_samples.py \
    quixbugs \
    infilling \
    --model-name codellama

python3 generate_patches.py  \
    samples_quixbugs_infilling_model_name_codellama.jsonl   \
    repairllama-infilling   \
    --model_name "ASSERT-KTH/RepairLLaMA-IR3-OR2"   \
    --n_workers 1   \
    --num_return_sequences 10  \
    --num_beams 10   \
    --max_new_tokens 128

python3 evaluate_patches.py \
    quixbugs \
    candidates_quixbugs_infilling_repairllama-infilling_model_name=ASSERT-KTH-RepairLLaMA-IR3-OR2_num_return_sequences=10_num_beams=10_max_new_tokens=128.jsonl \
    replace \
    --n_workers 1

python export_results.py \
    quixbugs \
    evaluation_quixbugs_infilling_repairllama-infilling.jsonl \
    --model_name "ASSERT-KTH/RepairLLaMA-IR3-OR2"