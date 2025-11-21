import argparse
import json
import os
from typing import List, Dict, Any, Tuple

from openai import OpenAI


def load_samples_from_log(log_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation samples from a JSONL log file.

    Returns a list of dicts, each with:
        {
            "question": str,
            "gt_answers": List[str],
            "model_answer": str,
            "meta": Dict[str, Any],   # includes mode, qidx, depth, etc.
        }
    """
    samples: List[Dict[str, Any]] = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            mode = rec.get("mode", "full_traj")

            # Normalize GT answers to a list of strings
            def normalize_gt(gt_raw):
                if isinstance(gt_raw, list):
                    return [str(x).strip() for x in gt_raw]
                elif gt_raw is None:
                    return []
                else:
                    return [str(gt_raw).strip()]

            if mode == "full_traj":
                gt_answers = normalize_gt(rec.get("gt_answer"))
                model_answer = str(rec.get("llm_response", "")).strip()
                samples.append(
                    {
                        "question": str(rec.get("question", "")).strip(),
                        "gt_answers": gt_answers,
                        "model_answer": model_answer,
                        "meta": {
                            "mode": mode,
                            "qidx": rec.get("qidx"),
                            "traj_id": rec.get("traj_id"),
                        },
                    }
                )

            elif mode == "haystack":
                # Each per_depth entry is treated as a separate evaluation sample
                per_depth = rec.get("per_depth", [])
                for pd in per_depth:
                    gt_answers = normalize_gt(pd.get("gt_answer"))
                    model_answer = str(pd.get("llm_response", "")).strip()
                    samples.append(
                        {
                            "question": str(rec.get("question", "")).strip(),
                            "gt_answers": gt_answers,
                            "model_answer": model_answer,
                            "meta": {
                                "mode": mode,
                                "qidx": rec.get("qidx"),
                                "traj_id": rec.get("traj_id"),
                                "depth": pd.get("depth"),
                                "depth_label": pd.get("depth_label"),
                            },
                        }
                    )
            else:
                # Unknown mode – skip
                continue

    return samples


def build_judge_prompt(question: str, gt_answers: List[str], model_answer: str) -> str:
    """
    LLM judge prompt that mimics get_score() with partial credit.
    Returns a score between 0 and 1.0.
    """
    gt_text = "\n".join(f"- {a}" for a in gt_answers) if gt_answers else "- (no answer provided)"

    prompt = f"""
You are an evaluator for a question-answering benchmark.

Your task:
- Compare the model's answer to the list of ground-truth answers.
- Award **partial credit** exactly following these rules:

Scoring rules (MUST follow exactly):
1. If there are MULTIPLE ground-truth answers (multiple ground-truth objects):
    - You may split the model's answer into sub-answers by commas. (e.g., "apple, potato, knife, book, saltshaker")
    - If ANY sub-answers from the model semantically matches ground-truth object, count it as a true positive.
    - Score = (number_of_matched_ground_truth_answers) / (total_number_of_ground_truth_answers)
    - Return a floating-point number between 0 and 1.
    - For example, if ground-truth answer is "PepperShaker, Apple, DishSponge" and the model's prediction is "apple, sponge", then the accuracy is 0.66. 

2. If there is ONLY ONE ground-truth answer:
    - If the model's answer semantically matches the ground-truth answer, score = 1.0
    - Otherwise score = 0.0

Semantic matching guidelines:
- Ignore punctuation, articles, pluralization differences, and simple paraphrases.
- "2", "two", "two items", "2 things" should match if referring to same count.
- "sink", "the bathroom sink", "a sink" should be considered equivalent.
- Do NOT give credit if model's answer contradicts the ground truth.

Output format:
Return ONLY a single JSON object on one line:
{{"score": <float between 0 and 1>}}

Do NOT output explanations.

Question:
\"\"\"{question}\"\"\"

Ground-truth answers:
{gt_text}

Model's answer:
\"\"\"{model_answer}\"\"\"
"""
    return prompt.strip()



def judge_one_sample(
    client: OpenAI,
    judge_model: str,
    question: str,
    gt_answers: List[str],
    model_answer: str,
) -> bool:
    """
    Call the LLM judge for a single sample.
    Returns True if judged correct, False otherwise.
    """
    prompt = build_judge_prompt(question, gt_answers, model_answer)

    resp = client.chat.completions.create(
        model=judge_model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "You are a strict and reliable automatic evaluator for question-answering systems.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    content = resp.choices[0].message.content.strip()
    # Try to parse JSON directly
    try:
        data = json.loads(content)
        correct = bool(data.get("correct", False))
        return correct
    except Exception:
        # Fallback: heuristics if parsing fails
        lc = content.lower()
        if "true" in lc or "correct" in lc:
            return True
        return False


def evaluate_log_with_llm_judge(
    log_path: str,
    judge_model: str = "gpt-4.1-mini",
    max_samples: int = None,
) -> float:
    """
    Main evaluation function:
      - reads the JSONL log
      - calls the LLM judge for each sample
      - returns accuracy (0.0–1.0)
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    client = OpenAI()  # uses OPENAI_API_KEY env var

    samples = load_samples_from_log(log_path)
    if max_samples is not None:
        samples = samples[:max_samples]

    if not samples:
        print("No samples found in log file.")
        return 0.0

    n_correct = 0
    n_total = 0

    for i, s in enumerate(samples):
        question = s["question"]
        gt_answers = s["gt_answers"]
        model_answer = s["model_answer"]

        is_correct = judge_one_sample(client, judge_model, question, gt_answers, model_answer)
        n_total += 1
        if is_correct:
            n_correct += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"Judged {i+1}/{len(samples)} samples...", flush=True)

    accuracy = n_correct / max(n_total, 1)
    print(f"LLM-judge accuracy: {accuracy:.4f} ({n_correct}/{n_total})")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="LLM-judge evaluation for NiEH QA logs.")
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path to JSONL result log file (e.g., output/eval_<qa>_<model>.log)",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model name to use as judge (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional: limit number of samples for quick debugging.",
    )
    args = parser.parse_args()

    evaluate_log_with_llm_judge(
        log_path=args.log_path,
        judge_model=args.judge_model,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
