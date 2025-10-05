import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def load_model(model_name):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cpu"
    )

model_names = [
    "/mnt/nfs-mnj-hot-99/tmp/hokuyama/make-data-count/outputs/normal-Qwen2-5-14B-Instruct-2025-07-19_21-06-10/checkpoint-14000",
    "/mnt/nfs-mnj-hot-99/tmp/hokuyama/make-data-count/outputs/normal-checkpoint-14000-2025-07-27_12-43-45/checkpoint-6000"
]

# 各モデルに対する重み（合計が1であることが望ましい）
weights = [0.4, 0.6]

# モデルを読み込む
models = [load_model(name) for name in model_names]

# 新しいモデルの重みを保持する辞書
averaged_state_dict = models[0].state_dict()

# 重みの加重平均を計算
for key in tqdm(averaged_state_dict.keys()):
    weighted_sum = sum(weights[i] * models[i].state_dict()[key] for i in range(len(models)))
    averaged_state_dict[key] = weighted_sum

# 新しいモデルを作成し、加重平均された重みをロード
new_model = load_model(model_names[0])
new_model.load_state_dict(averaged_state_dict)

# トークナイザーはそのまま使用
tokenizer = AutoTokenizer.from_pretrained(model_names[0])

output_path = "/mnt/nfs-mnj-hot-99/tmp/hokuyama/make-data-count/outputs/Qwen2.5-14B-Instruct-merged"
os.makedirs(output_path, exist_ok=True)
# 新しいモデルを保存
new_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
