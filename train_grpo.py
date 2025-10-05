# train_grpo.py
import os
import argparse
from loguru import logger

from trl import GRPOConfig, GRPOTrainer
import torch
import torch.distributed as dist
from utils import *
from common import *

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--output-dir", type=str, default="./outputs_test")
    parser.add_argument("--use-ompi", action="store_true", default=False)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    logger.info("Start")

    if args.use_ompi: init_torch_distributed()

    dataset = make_dataset(args.model_name)
    logger.info(len(dataset))
    logger.info(dataset[0])

    training_args = GRPOConfig(
        use_cpu=False,
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        num_generations=8,
        learning_rate=5e-7,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.01,
        max_grad_norm=1.0,
        num_train_epochs = 1,
        max_steps=args.max_steps,
        logging_steps=100,
        max_prompt_length=4096,
        max_completion_length=4,
        save_only_model=True,
        save_steps=5000,
        report_to="tensorboard",
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs = prob_r_f,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info("Finish")

if __name__ == "__main__":
    main()