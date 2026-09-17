import csv
from datetime import datetime
from pathlib import Path

STAT_FIELDS = [
    "num_answerable", "correct_of_answerable", "answered_of_answerable",
    "abstained_of_answerable", "malformed_of_answerable",
    "num_unanswerable", "answered_of_unanswerable",
    "abstained_of_unanswerable", "malformed_of_unanswerable",
]

TRAIN_FIELDS = [
    "step", "total_steps", "loss", "reward_avg", "reward_std",
    "kl_loss", "policy_ratio",
    *STAT_FIELDS,
    "tokens_per_sec", "avg_response_len", "adv_avg", "adv_std", "entropy_avg",
]

EVAL_FIELDS = ["step", "avg_acc", *STAT_FIELDS]

class MetricsLogger:
    def __init__(self, run_name: str, field_names, suffix=""):
        # Make log directory if there isn't one
        logs_dir = Path(__file__).parent / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Store file path
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
        self.file_path = logs_dir / f"{timestamp}_{run_name}.csv"

        # Write headers o csv
        self.field_names = list(field_names)
        with self.file_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.field_names).writeheader()

        with self.file_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.field_names)
            writer.writeheader()

    def log_metrics(self, **metrics):
        with self.file_path.open("a", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=self.field_names,
                restval=None, extrasaction="raise",
            )
            writer.writerow(metrics)