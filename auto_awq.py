from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = '/mnt/nfs-mnj-hot-99/tmp/hokuyama/make-data-count/outputs/normal-Qwen2-5-14B-Instruct-2025-09-04_22-53-04/checkpoint-4500'
quant_path = '/mnt/nfs-mnj-hot-99/tmp/hokuyama/make-data-count/outputs/normal-Qwen2-5-14B-Instruct-2025-09-04_22-53-04-checkpoint-4500-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
