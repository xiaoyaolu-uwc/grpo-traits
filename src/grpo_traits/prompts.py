from transformers import AutoTokenizer

SYSTEM_PROMPT = (
    "You are given a math word problem. Some problems are unanswerable "
    "because they are missing information, contain contradictions, or ask "
    "about something not mentioned.\n"
    "Think step by step before answering, then end your reply with your answer in the following format:\n"
    "<answer>NUMBER</answer> where NUMBER is your answer or "
    "<answer>ABSTAIN</answer> if it is unsolvable.\n"
    "Consider whether the question is answerable or not."
)

def build_prompt(question: str, tokenizer:AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

