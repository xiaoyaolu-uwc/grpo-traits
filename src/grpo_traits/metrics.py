import csv
import json
from datetime import datetime
from pathlib import Path

STAT_FIELDS = [
    "num_answerable", "correct_of_answerable", "answered_of_answerable",
    "abstained_of_answerable", "malformed_of_answerable",
    "num_unanswerable", "answered_of_unanswerable",
    "abstained_of_unanswerable", "malformed_of_unanswerable",
]

TRAIN_FIELDS = [
    "step", "total_steps", "mem_live", "mem_total", "loss", "reward_avg", "reward_std",
    "kl_loss", "policy_ratio", "clip_frac",
    *STAT_FIELDS,
    "tokens_per_sec", "avg_response_len", "adv_avg", "adv_std", 
]

EVAL_FIELDS = ["step", "avg_acc", *STAT_FIELDS]

class MetricsLogger:
    def __init__(self, run_name: str, field_names, suffix):
        # Make log directory if there isn't one
        logs_dir = Path(__file__).parent / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Store file path
        self.file_path = logs_dir / f"{run_name}_{suffix}.csv"

        # Write headers o csv
        self.field_names = list(field_names)
        with self.file_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.field_names).writeheader()

    def log_metrics(self, **metrics):
        with self.file_path.open("a", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=self.field_names,
                restval=None, extrasaction="raise",
            )
            writer.writerow(metrics)

class SampleLogger:
    def __init__(self, run_name: str, suffix="samples"):
        logs_dir = Path(__file__).parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        self.file_path = logs_dir / f"{run_name}_{suffix}.jsonl"
        self.file_path.touch()

    def log_config(self, **config):
        with self.file_path.open("a") as f:
            f.write(json.dumps({"record": "config", **config}) + "\n")

    def log_samples(self, step, rows, answers, rewards, completions, group_size):
        with self.file_path.open("a") as f:
            for i, row in enumerate(rows):
                lo, hi = i * group_size, (i + 1) * group_size
                f.write(json.dumps({
                    "step": step,
                    "question": row["question"],
                    "answerable": row["answerable"],
                    "expected": row["answer"],
                    "rollouts": [
                        {
                            "text": completions[j],
                            "tag": str(answers[j][0]),
                            "value": answers[j][1],
                            "reward": rewards[j],
                        }
                        for j in range(lo, hi)
                    ],
                }) + "\n")