import json
import csv
import subprocess
import time
import psutil
import os
import sys
from bert_score import score

PROMPT = (
    "Please evaluate the exercise form shown. "
    "What mistakes, if any, are present, and what corrections would you recommend?"
)

def run_inference(model_path, conv_mode, text, media):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB

    result = subprocess.run(
        [
            "vila-infer",
            "--model-path", model_path,
            "--conv-mode", conv_mode,
            "--text", text,
            "--media", media
        ],
        capture_output=True,
        text=True
    )

    mem_after = process.memory_info().rss / (1024 * 1024)  # MB
    inference_time = time.time() - start_time
    memory_usage = mem_after - mem_before

    model_output = result.stdout.strip()
    return model_output, memory_usage, inference_time


def main(json_path, csv_path, model_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Optional limit argument
    limit = None
    if len(sys.argv) == 5:
        try:
            limit = int(sys.argv[4])
        except ValueError:
            print("Invalid limit argument, must be an integer.")
            sys.exit(1)

    if limit is not None:
        data = data[:limit]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow([
                "video",
                "ground_truth",
                "model_output",
                "bert_precision",
                "bert_recall",
                "bert_f1",
                "memory_usage_mb",
                "inference_time_sec"
            ])

        for entry in data:
            video = entry.get("video", "")
            ground_truth = ""

            for conv in entry.get("conversations", []):
                if conv.get("from") == "gpt":
                    ground_truth = conv.get("value", "")
                    break

            model_output, memory_usage, inference_time = run_inference(
                model_path=model_path,
                conv_mode="vicuna_v1",
                text=PROMPT,
                media=video
            )

            # ---- BERTScore computation ----
            if ground_truth.strip() and model_output.strip():
                P, R, F1 = score(
                    [model_output],
                    [ground_truth],
                    lang="en",
                    model_type="bert-base-uncased",
                    verbose=False
                )
                bert_p = P.item()
                bert_r = R.item()
                bert_f1 = F1.item()
            else:
                bert_p = bert_r = bert_f1 = 0.0

            writer.writerow([
                video,
                ground_truth,
                model_output,
                f"{bert_p:.4f}",
                f"{bert_r:.4f}",
                f"{bert_f1:.4f}",
                f"{memory_usage:.2f}",
                f"{inference_time:.2f}"
            ])

            print(f"Processed {video} | BERT-F1: {bert_f1:.4f}")


if __name__ == "__main__":
    if len(sys.argv) not in [4, 5]:
        print("Usage: python inference.py <input_json> <output_csv> <model_path> [limit]")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
