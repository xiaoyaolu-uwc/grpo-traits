import csv
from datetime import datetime
from pathlib import Path

class MetricsLogger:
    def __init__(self, run_name: str):
        # Make log directory if there isn't one
        logs_dir = Path(__file__).parent / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Store file path
        timestamp = datetime.now().strftime("%Y-%M-%D_%H-%M-%S") 
        self.file_path = logs_dir / f"{timestamp}_{run_name}.csv"

        # Write headers to csv
        self.field_names = [
            "step",
            "total_steps",
            "loss",
            "reward_avg",
            "kl_loss",
            "policy_ratio",
            "correct_of_answerable",
            "answered_of_answerable",
            "abstained_of_unanswerable",
            "malformed_of_answerable",
            "malformed_of_unanswerable",
            "tokens_per_sec",
            "avg_response_len",
            "adv_avg",
            "adv_std",
            "entropy_avg",
            "eval_acc",
        ]
        with self.file_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.field_names)
            writer.writeheader()

    def log_metrics(
            self,
            step,
            total_steps,
            loss,
            reward_avg,
            kl_loss,
            policy_ratio,
            correct_of_answerable,
            answered_of_answerable,
            abstained_of_unanswerable,
            malformed_of_answerable,
            malformed_of_unanswerable,
            tokens_per_sec,
            avg_response_len,
            adv_avg,
            adv_std,
            entropy_avg,
            eval_acc,
    ):
        with self.file_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.field_names)
            writer.writerow({
                "step": step,
                "total_steps": total_steps,
                "loss": loss,
                "reward_avg": reward_avg,
                "kl_loss": kl_loss,
                "policy_ratio": policy_ratio,
                "tokens_per_sec": tokens_per_sec,
                "avg_response_len": avg_response_len,
                "adv_avg": adv_avg,
                "adv_std": adv_std,
                "entropy_avg": entropy_avg,
                "eval_acc": eval_acc,
                "correct_of_answerable": correct_of_answerable,
                "answered_of_answerable": answered_of_answerable,
                "abstained_of_unanswerable": abstained_of_unanswerable,
                "malformed_of_answerable": malformed_of_answerable,
                "malformed_of_unanswerable": malformed_of_unanswerable,
            })