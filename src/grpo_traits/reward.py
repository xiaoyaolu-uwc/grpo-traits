import re
import math
from enum import StrEnum

class Tag(StrEnum):
    ANSWER = "ANSWER"
    ABSTAIN = "ABSTAIN"
    MALFORMED = "MALFORMED"
pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def extract_answers(completion_text: list) -> list:
    """
    Expects a list of string completions, extract and 
    output list of tuples of form (answer_type, answer_value)
    """
    parsed = [pattern.search(completion) for completion in completion_text]
    answers = []
    for i in range(len(parsed)):
        # Check if malformed
        if parsed[i] is None:
            answers.append((Tag.MALFORMED, None))
            continue
        # Clean answer
        parts = parsed[i].group(1).strip().split()
        final_answer = re.sub(r"[^A-Za-z0-9.\-]", "", parts[-1])
        # Append answer
        if final_answer.upper() == "ABSTAIN":
            answers.append((Tag.ABSTAIN, None))
        else:
            try: 
                answers.append((Tag.ANSWER, float(final_answer)))
            except ValueError:
                answers.append((Tag.MALFORMED, None))
    return answers

def grade_answer(answer: tuple, answerable: bool, is_correct: bool, is_strict: bool):
    """
    Grades a single completion. strict= controls whether answer must be 
    correct for answerable questions
    """
    tag = answer[0]
    if tag == Tag.MALFORMED:
        return 0
    if tag == Tag.ABSTAIN:
        if answerable:
            return 0 
        else: 
            return 1
    if tag == Tag.ANSWER:
        if answerable and not is_strict:
            return 1
        elif answerable and is_strict:
            return 1 if is_correct else 0
        else:
            return 0

def compute_reward(answers: list, group_size: int, rows: list, is_strict: bool):
    """
    Expects a list of string completions, output reward tensor
    Rewards an answer attempt on answerable questions, and abstain
    on unanswerable questions.
    """
    num_batches = len(rows) 
    if num_batches * group_size != len(answers):
        raise ValueError("Expected number of completions != true number of completions")

    rewards = []
    answer_stats = {
        "num_answerable": 0,
        "num_unanswerable": 0,
        "correct_of_answerable": 0,
        "answered_of_answerable": 0,
        "abstained_of_answerable": 0,
        "malformed_of_answerable": 0,
        "abstained_of_unanswerable": 0,
        "malformed_of_unanswerable": 0,
    }

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
        batch = answers[start_idx:end_idx]

        # Compute reward     
        for answer in batch:
            is_correct = False
            kind = answer[0]
            if kind == Tag.ANSWER and correct_ans is not None and math.isclose(answer[1], correct_ans):
                is_correct = True
            rewards.append(grade_answer(answer, answerable, is_correct, is_strict))
            # Record statistics
            if answerable:
                answer_stats["num_answerable"] += 1
                if kind == Tag.ANSWER:
                    answer_stats["answered_of_answerable"] += 1
                    if is_correct:
                        answer_stats["correct_of_answerable"] += 1
                elif kind == Tag.ABSTAIN:
                    answer_stats["abstained_of_answerable"] += 1
                else:
                    answer_stats["malformed_of_answerable"] += 1
            else:
                answer_stats["num_unanswerable"] += 1
                if kind == Tag.ANSWER:
                    answer_stats["answered_of_unanswerable"] += 1
                elif kind == Tag.ABSTAIN:
                    answer_stats["abstained_of_unanswerable"] += 1
                else:
                    answer_stats["malformed_of_unanswerable"] += 1
    return rewards, answer_stats

    

                    


        



    