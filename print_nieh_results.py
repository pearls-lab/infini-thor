import json
import os
import glob
import numpy as np


def calculate_metrics(file_path):
    
    total_match = 0.0
    total_count = 0.0
    
    # open_ended
    oe_match = 0.0
    oe_count = 0.0
    
    ctx_n_tokens = []
    
    with open(file_path, "r", encoding="utf-8") as f_log:
        for line in f_log:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            qid = rec.get("qidx")
            if qid is None:
                continue

            # Warm start aggregate stats from existing records
            score_val = rec.get("score")
            if score_val is not None:
                total_match += float(score_val)
                total_count += 1.0
                
            if isinstance(rec.get("gt_answer")[0], str) and not rec.get("gt_answer")[0].lower() in ["no", "yes"]:
                oe_match += float(score_val)
                oe_count += 1.0
                
            ctx_n_tokens.append(rec.get("ctx_n_tokens"))
                
                
    score = total_match / total_count if total_count > 0 else 0.0
    oe_score = oe_match / oe_count if oe_count > 0 else 0.0
    print(f"score: {score:.4f}, total_match: {total_match}, total_count: {total_count}")
    print(f"open_ended_score: {oe_score:.4f}, open_ended_match: {oe_match}, open_ended_count: {oe_count}")
    print(f"avg ctx_n_tokens: {np.array(ctx_n_tokens).mean()}")

if __name__ == "__main__":
    # List of files to process
    # You can update this list or use glob.glob("*.log") to find them automatically
    files = [
        "/infini-thor/output/eval_qa_set_nieh_single_clue_Qwen2.5-VL-7B-Instruct_full_traj_video_fps1.log",
        "/infini-thor/output/eval_qa_set_nieh_single_clue_Qwen2.5-VL-7B-Instruct_full_traj_video_fps2.log",
        "/infini-thor/output/eval_qa_set_nsieh_multi_clue_Qwen2.5-VL-7B-Instruct_full_traj_video_fps1.log",
        "/infini-thor/output/eval_qa_set_nsieh_multi_clue_Qwen2.5-VL-7B-Instruct_full_traj_video_fps2.log"
    ]
    
    # Check if files exist, if not, try to find any .log files in current directory
    existing_files = [f for f in files if os.path.exists(f)]
    
    if not existing_files:
        print("Specific log files not found in directory. Searching for any .log files...")
        existing_files = glob.glob("*.log")
        
    if existing_files:
        for filepath in existing_files:
            calculate_metrics(filepath)
    else:
        print("No log files found to analyze.")