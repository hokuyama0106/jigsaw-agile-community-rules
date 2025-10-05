import os
import ast
import datetime
import pandas as pd
import numpy as np

import torch
import torch.distributed as dist

train_df = pd.read_csv("/mnt/nfs-mnj-hot-99/tmp/hokuyama/jigsaw-agile-community-rules/jigsaw-agile-community-rules/train.csv", index_col="row_id")

def prob_check(c, row_id):
    reward = 0
    answer = train_df.loc[row_id, "rule_violation"]
    try:
        prob = float(c)
        if 0 <= prob <= 1:
            reward += 1 - abs(answer-prob)
    except:
        pass

    return reward

def prob_r_f(completions, row_id, **kwargs) -> list[float]:
    if isinstance(completions[0], list):
        responses = [completion[0]["content"] for completion in completions]
    elif isinstance(completions[0], str):
        responses = [completion for completion in completions]
    return [prob_check(response, row_id[i]) for i, response in enumerate(responses)]

def init_torch_distributed():
    # Increase watchdog timeout
    os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '600'
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(hours=2))
    torch.distributed.barrier()