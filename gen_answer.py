import re
import torch
import pandas as pd
import ast
import vllm

def main():
    # Model path
    model_path = "/mnt/nfs-mnj-hot-99/tmp/hokuyama/models/Qwen3-8B"

    # Initialize the LLM
    llm = vllm.LLM(
        model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.99,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=32,
        disable_log_stats=True,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    SYSTEM_PROMPT = """
You are an expert content moderator AI model. 
Your task is to analyze a Reddit comment against a specific community rule and predict the probability that the comment violates that rule.
The probability must be a float value between 0.00 and 1.00, formatted to two decimal places.
A probability closer to 1.00 indicates a high likelihood of a rule violation, and a probability closer to 0.00 indicates a low likelihood of a rule violation.

YOUR OUTPUT MUST BE ONLY THE PROBABILITY VALUE, WITH NO OTHER TEXT, REASONING, OR EXPLANATION."""

    USER_PROMPT_TEMPLATE = """
### RULE AND CONTEXT
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

    # Read data
    df_train = pd.read_csv("jigsaw-agile-community-rules/train.csv")

    prompts = []
    for i, r in df_train.iterrows():
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
        prompts.append(messages)

    outputs = llm.chat(
        prompts,
        vllm.SamplingParams(
            seed=0,
            skip_special_tokens=True,
            max_tokens=4,
            temperature=0,
            repetition_penalty=1.05,
        ),
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=True
    )
    responses =  [output.outputs[0].text for output in outputs]

    answers = []
    for response in responses:
        try:
            answer = float(response)
            answers.append(answer)
        except:
            answers.append(0.5)

    df_train["llm_output"] = answers

    # Save as Parquet file
    df_train.to_parquet('output.parquet', index=False)

if __name__ == "__main__":
    main()
