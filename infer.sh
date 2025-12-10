# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/slake/slake_test.jsonl --max_new_tokens 1024 --max_model_len 1024 --model Qwen/Qwen3-VL-32B-Thinking --quant_method bnb --quant_bits 4 --torch_dtype bfloat16 --stream true
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/slake/slake_test.jsonl#10 --max_new_tokens 1024 --max_model_len 1024 --quant_method bnb --quant_bits 4 --torch_dtype bfloat16 --stream true --model Qwen/Qwen3-VL-30B-A3B-Thinking
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/charxiv/charxiv_val.jsonl --max_new_tokens 1024 --max_model_len 1024 --quant_method bnb --quant_bits 4 --torch_dtype bfloat16 --stream true --model Qwen/Qwen3-VL-32B-Thinking
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/slake/slake_test.jsonl#10 --max_new_tokens 1024 --max_model_len 1024 --quant_method bnb --quant_bits 4 --torch_dtype bfloat16 --stream true --model Qwen/Qwen3-VL-32B-Instruct
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/charxiv/charxiv_val.jsonl --max_new_tokens 8192 --max_model_len 8192 --stream true --model Qwen/Qwen3-VL-4B-Instruct
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system "Answer the question concisely based on the image." --max_new_tokens 8192 --max_model_len 8192 --stream true --model Qwen/Qwen3-VL-2B-Instruct
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system "Answer the question concisely based on the image." --max_new_tokens 8192 --max_model_len 8192 --stream true --model Qwen/Qwen3-VL-4B-Instruct
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system "Answer the question concisely based on the image." --max_new_tokens 8192 --max_model_len 8192 --quant_method bnb --quant_bits 4 --torch_dtype bfloat16 --stream true --model Qwen/Qwen3-VL-32B-Instruct
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 2048 --max_model_len 2048 --stream true --adapters /data/workspace/swift/output/v54-20251117-030137/checkpoint-500
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system "Answer the question concisely based on the image." --max_new_tokens 2048 --max_model_len 2048 --stream true --model Qwen/Qwen2.5-VL-3B-Instruct
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 2048 --max_model_len 2048 --stream true --adapters /data/workspace/swift/output/v56-20251117-101016/checkpoint-1000
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 2048 --max_model_len 2048 --stream true --adapters /data/workspace/swift/output/v59-20251117-125540/checkpoint-1000
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 2048 --max_model_len 2048 --stream true --adapters /data/workspace/swift/output/v59-20251117-125540/checkpoint-2000
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 2048 --max_model_len 2048 --stream true --adapters /data/workspace/swift/output/v60-20251117-163601/checkpoint-1000
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 2048 --max_model_len 2048 --stream true --adapters /data/workspace/swift/output/v62-20251117-182257/checkpoint-1000
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 2048 --max_model_len 2048 --stream true --adapters /data/workspace/swift/output/v67-20251118-145317/checkpoint-5500
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 2048 --max_model_len 2048 --stream true --model Qwen/Qwen2.5-VL-3B-Instruct
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --max_new_tokens 2048 --max_model_len 2048 --stream true --model Qwen/Qwen2.5-VL-3B-Instruct
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v67-20251118-145317/checkpoint-5500
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v67-20251118-145317/checkpoint-6000
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v67-20251118-145317/checkpoint-6500
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v67-20251118-145317/checkpoint-7000
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v67-20251118-145317/checkpoint-7500
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v67-20251118-145317/checkpoint-8000
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v67-20251118-145317/checkpoint-8500
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v67-20251118-145317/checkpoint-9000
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v67-20251118-145317/checkpoint-9500
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v67-20251118-145317/checkpoint-10000
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v67-20251118-145317/checkpoint-6500
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v71-20251120-002734/checkpoint-1500
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v71-20251120-002734/checkpoint-2000
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v71-20251120-002734/checkpoint-2500
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v71-20251120-002734/checkpoint-3000
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v71-20251120-002734/checkpoint-3500
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v71-20251120-002734/checkpoint-4000
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v71-20251120-002734/checkpoint-5000
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 1024 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v71-20251120-002734/checkpoint-6000
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 512 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v96-20251122-181946/checkpoint-4000
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 512 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v96-20251122-181946/checkpoint-5000
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system examples/train/grpo/prompt.txt --max_new_tokens 512 --max_model_len 1024 --stream true --adapters /data/workspace/swift/output/v96-20251122-181946/checkpoint-6000
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/.archive/vqarad_test_251123.jsonl --system /data/workspace/swift/prompt2.txt --max_new_tokens 512 --stream true --adapters /data/workspace/swift/output/v97-20251124-020444/checkpoint-5500
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/.archive/vqarad_test_251123.jsonl --system /data/workspace/swift/prompt2.txt --max_new_tokens 512 --stream true --adapters /data/workspace/swift/output/v97-20251124-020444/checkpoint-6000
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt3.txt --max_new_tokens 512 --stream true --adapters /data/workspace/swift/output/v99-20251124-233227/checkpoint-1500
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt4.txt --max_new_tokens 512 --stream true --adapters /data/workspace/swift/output/v107-20251126-022444/checkpoint-5500
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt1.txt --max_new_tokens 512 --stream true --adapters /data/workspace/swift/output/v121-20251129-234138/checkpoint-5500
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt1.txt --max_new_tokens 512 --stream true --adapters /data/workspace/swift/output/v121-20251129-234138/checkpoint-6000
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt2.txt --max_new_tokens 512 --stream true --adapters /data/workspace/swift/output/v122-20251130-143141/checkpoint-5000
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt2.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v123-20251201-095338/checkpoint-6000
# sleep 30s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt2.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v125-20251202-034528/checkpoint-6000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt2.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v125-20251202-034528/checkpoint-5500
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt2.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v125-20251202-034528/checkpoint-5000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt2.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v125-20251202-034528/checkpoint-4000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt2.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v125-20251202-034528/checkpoint-3000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v126-20251203-004753/checkpoint-4500
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v127-20251203-233416/checkpoint-5000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v126-20251203-004753/checkpoint-4000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v126-20251203-004753/checkpoint-3500
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v126-20251203-004753/checkpoint-3000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v128-20251204-154030/checkpoint-5000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v128-20251204-154030/checkpoint-4500
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v128-20251204-154030/checkpoint-4000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v128-20251204-154030/checkpoint-3500
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v128-20251204-154030/checkpoint-3000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v133-20251206-002521/checkpoint-5000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v133-20251206-002521/checkpoint-4500
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v133-20251206-002521/checkpoint-4000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v133-20251206-002521/checkpoint-3500
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v134-20251206-152223/checkpoint-5000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v134-20251206-152223/checkpoint-4500
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v134-20251206-152223/checkpoint-4000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v134-20251206-152223/checkpoint-3500
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --system /data/workspace/swift/prompt6.txt --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v134-20251206-152223/checkpoint-3000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v138-20251207-131936/checkpoint-5000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v164-20251209-000358/checkpoint-2500
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v167-20251209-155953/checkpoint-5000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v167-20251209-155953/checkpoint-4500
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v169-20251210-012330/checkpoint-5500
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v169-20251210-012330/checkpoint-5000
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v169-20251210-012330/checkpoint-4500
# sleep 120s
# CUDA_VISIBLE_DEVICES=0 swift infer --val_dataset /data/datasets/vqarad/vqarad_test.jsonl --max_new_tokens 1024 --stream true --adapters /data/workspace/swift/output/v169-20251210-012330/checkpoint-4000
# sleep 120s
