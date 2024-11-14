# CUDA_VISIBLE_DEVICES=0  swift sft  --model_type phi3_5-mini-instruct  --dataset science-qa  --num_train_epochs 1  --sft_type lora  --output_dir output  --use_flash_attn true  >> phi3_5-mini-instruct_sqa-100_ep1.log 2>&1
# CUDA_VISIBLE_DEVICES=0 swift sft  --model_type phi3-vision-128k-instruct  --dataset MS::swift/ScienceQA --num_train_epochs 1 --sft_type lora  --output_dir output  --use_flash_attn true  >> phi3-vision-128k-instruct_science-qa_ep1.log 2>&1
# CUDA_VISIBLE_DEVICES=0 swift eval --model_type phi3-vision-128k-instruct --eval_dataset ScienceQA_TEST --eval_batch_size 1 --enforce-eager --max_num_seqs 256 --max_model_len 4096 --infer_backend vllm >> phi3-vision-128k-instruct_sqa-eval.log 2>&1
# CUDA_VISIBLE_DEVICES=0 swift sft --model_type phi3_5-vision-instruct --sft_type lora --dataset science-qa/train --output_dir output --num_train_epochs 1 --use_flash_attn true >> phi3_5-vision-instruct_sqa_ep1-1.log 2>&1
# CUDA_VISIBLE_DEVICES=0 swift eval --ckpt_dir /home/civisky/workspace/swift/output/phi3_5-vision-instruct/v13-20241113-162918/checkpoint-500 --eval_dataset ScienceQA_TEST --eval_batch_size 1 --enforce-eager --max_num_seqs 256 --max_model_len 4096 --infer_backend vllm --vllm_enable_lora true >> phi3_5-vision-instruct_sqa_ep1-1.log 2>&1
# CUDA_VISIBLE_DEVICES=0 swift eval --model_type phi3_5-vision-instruct --eval_dataset ARC_c --infer_backend vllm --enforce-eager --max_num_seqs 256 --max_model_len 4096 >> phi3_5-mini-instruct-log.log 2>&1
# CUDA_VISIBLE_DEVICES=0 swift sft --model_type phi3_5-vision-instruct --sft_type lora --dataset /home/civisky/workspace/datasets/sqa/problems-train0.01-processed-train.jsonl --dataset_test_ratio 0.20 --output_dir output --num_train_epochs 5 --use_flash_attn true >> phi3_5-vision-instruct_sqa_ep5.log 2>&1
# CUDA_VISIBLE_DEVICES=0 swift eval --model_type phi3-vision-128k-instruct --eval_dataset ScienceQA_TEST --enforce-eager --max_num_seqs 256 --max_model_len 4096 --infer_backend vllm >> log.log 2>&1
# CUDA_VISIBLE_DEVICES=0 swift eval --model_type phi3_5-vision-instruct --eval_dataset ScienceQA_TEST --enforce-eager --max_num_seqs 256 --max_model_len 4096 --infer_backend vllm >> phi3-vision-128k-instruct_sqa-eval.log 2>&1
# CUDA_VISIBLE_DEVICES=0 swift sft --model_type phi3_5-vision-instruct --sft_type lora --dataset /home/civisky/workspace/datasets/sqa/problems-train0.01-processed-train.jsonl --output_dir output --num_train_epochs 20 --use_flash_attn true >> phi3_5-vision-instruct_sqa_ep20.log 2>&1
CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir /home/civisky/workspace/swift/output/phi3_5-vision-instruct/v33-20241114-192807/checkpoint-140 --val_dataset /home/civisky/workspace/datasets/sqa/problems-train0.01-processed-val.jsonl --use_flash_attn true --infer_backend vllm >> phi3_5-vision-instruct_sqa_ep20.log 2>&1





















































































