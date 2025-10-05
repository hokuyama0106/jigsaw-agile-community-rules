import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load and prep dataset
SYSTEM_PROMPT = """You are an expert content moderator AI model. 
Your task is to analyze a Reddit comment against a specific community rule and predict the probability that the comment violates that rule.
The probability must be a float value between 0.00 and 1.00, formatted to two decimal places.
A probability closer to 1.00 indicates a high likelihood of a rule violation, and a probability closer to 0.00 indicates a low likelihood of a rule violation.

YOUR OUTPUT MUST BE ONLY THE PROBABILITY VALUE, WITH NO OTHER TEXT, REASONING, OR EXPLANATION."""

USER_PROMPT_TEMPLATE = """### RULE AND CONTEXT
- Rule: {rule_text}
- Subreddit: {subreddit_name} (This is the community where the comment was posted.)

### POSITIVE EXAMPLES (Comments that VIOLATE the Rule)
- Example 1: {positive_example_1}
- Example 2: {positive_example_2}

### NEGATIVE EXAMPLES (Comments that DO NOT Violate the Rule)
- Example 1: {negative_example_1}
- Example 2: {negative_example_2}

### COMMENT TO EVALUATE
- Comment Body: {comment_body}"""

def format_dataset(r, model_name):
    user_prompt = USER_PROMPT_TEMPLATE.format(
        rule_text=r['rule'],
        subreddit_name=r['subreddit'],
        positive_example_1=r['positive_example_1'],
        positive_example_2=r['positive_example_2'],
        negative_example_1=r['negative_example_1'],
        negative_example_2=r['negative_example_2'],
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
    dataset = load_dataset('csv', data_files='/mnt/nfs-mnj-hot-99/tmp/hokuyama/jigsaw-agile-community-rules/jigsaw-agile-community-rules/train.csv')["train"]
    data = dataset.map(format_dataset, fn_kwargs={"model_name": model_name})

    return data

