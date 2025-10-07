import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load and prep dataset
SYSTEM_PROMPT = """You are an expert content moderator AI model. 
Your task is to analyze a Reddit comment against a specific community rule and predict the probability that the comment violates that rule.
The probability must be a float value between 0.00 and 1.00, formatted to two decimal places.
A probability closer to 1.00 indicates a high likelihood of a rule violation, and a probability closer to 0.00 indicates a low likelihood of a rule violation.

YOUR OUTPUT MUST BE ONLY THE PROBABILITY VALUE, WITH NO OTHER TEXT, REASONING, OR EXPLANATION."""

USER_PROMPT_TEMPLATE = """### RULE
{rule_text}

### COMMENT TO EVALUATE
{comment_body}"""

def format_dataset(r, model_name):
    user_prompt = USER_PROMPT_TEMPLATE.format(
        rule_text=r['rule'],
        comment_body=r['body']
    )
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt}
    ]

    if "Qwen3" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # True is the default value for enable_thinking.
        )
        data = {
            'prompt': text
        }
    else:
        data = {
            'prompt': messages,
        }

    return data

def make_dataset(model_name, train_type="grpo") -> Dataset:
    dataset = load_dataset('csv', data_files='/mnt/nfs-mnj-hot-99/tmp/hokuyama/jigsaw-agile-community-rules/jigsaw-agile-community-rules/train_v2.csv')["train"]
    data = dataset.map(format_dataset, fn_kwargs={"model_name": model_name})

    return data

