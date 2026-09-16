import re

pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def grade_answer(answer, answerable, correct_ans, strict):
    """
    Grades a single completion. strict= controls whether answer must be 
    correct for answerable questions
    """
    if answer is None:
        return 0
    
    parts = answer.group(1).strip().split()
    final_answer = re.sub(r"[^A-Za-z0-9]", "", parts[-1])

    if answerable:
        if final_answer.upper() == "ABSTAIN": 
            return 0
        elif not strict:
            return 1
        elif int(final_answer) == correct_ans: 
            return 1
        else: # incorrect answer for strict
            return 0
    else:
        if final_answer.upper() == "ABSTAIN":
            return 1
        else:
            return 0

def compute_reward(completion_text: list, group_size: int, rows: list, strict: bool):
    """
    Expects a list of string completions, output reward tensor
    Rewards an answer attempt on answerable questions, and abstain
    on unanswerable questions.
    """
    num_batches = len(rows) 
    if num_batches * group_size != len(completion_text):
        raise ValueError("Expected number of completions != true number of completions")

    rewards = []
    strict_rewards = []

    for i in range(num_batches):
        # Get relevant question & answer for batch 
        row = rows[i]
        answerable = row["answerable"]
        if row["answer"] is None:
            correct_ans = None
        else:
            correct_ans = row["answer"][0]


        # Extract relevant batch
        start_idx = i * group_size
        end_idx = (i + 1) * group_size
        batch = completion_text[start_idx:end_idx]

        # Compute reward
        answers = [pattern.search(completion) for completion in batch]
        rewards += [grade_answer(answer, answerable, 
                                 correct_ans, strict=strict) for answer in answers]
        
    return rewards

    

                    


        



    