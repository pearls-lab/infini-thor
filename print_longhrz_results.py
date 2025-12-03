import json
import os
import glob

def calculate_metrics(file_paths):
    """
    Parses log files and prints aggregate metrics per model.
    """
    
    # Define the fields we want to aggregate
    fields_to_sum = [
        "total_count", "success_count",
        "goto_count", "goto_success",
        "pick_count", "pick_success",
        "put_count", "put_success"
    ]

    print(f"{'Model Name':<60} | {'Success Rate':<12} | {'Goto Rate':<10} | {'Pick Rate':<10} | {'Put Rate':<10}")
    print("-" * 115)

    for file_path in file_paths:
        # Use filename as model identifier
        model_name = os.path.basename(file_path)
        
        # Initialize aggregators
        agg = {key: 0 for key in fields_to_sum}
        lines_processed = 0

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        metrics = data.get("metric", {})
                        
                        # Sum up all relevant fields
                        for key in fields_to_sum:
                            agg[key] += metrics.get(key, 0)
                            
                        lines_processed += 1
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line in {model_name}")
                        continue
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            continue

        # Calculate Rates (avoiding division by zero)
        # Overall Success Rate
        success_rate = (agg['success_count'] / agg['total_count']) * 100 if agg['total_count'] > 0 else 0.0
        
        # Sub-task Rates
        goto_rate = (agg['goto_success'] / agg['goto_count']) * 100 if agg['goto_count'] > 0 else 0.0
        pick_rate = (agg['pick_success'] / agg['pick_count']) * 100 if agg['pick_count'] > 0 else 0.0
        put_rate = (agg['put_success'] / agg['put_count']) * 100 if agg['put_count'] > 0 else 0.0

        # Print Row
        print(f"{model_name:<60} | {success_rate:>10.2f}% | {goto_rate:>8.2f}% | {pick_rate:>8.2f}% | {put_rate:>8.2f}%")

if __name__ == "__main__":
    # List of files to process
    # You can update this list or use glob.glob("*.log") to find them automatically
    files = [
        "outputs/eval_longhrz__checkpoint_infini-ft_step-20000-hf_.log",
        "outputs/eval_longhrz__data_checkpoints_step-60000-hf_.log",
        "outputs/eval_longhrz__data_bkim_checkpoints_infini-memory-text-state_step-60000-hf_.log"
    ]
    
    # Check if files exist, if not, try to find any .log files in current directory
    existing_files = [f for f in files if os.path.exists(f)]
    
    if not existing_files:
        print("Specific log files not found in directory. Searching for any .log files...")
        existing_files = glob.glob("*.log")
        
    if existing_files:
        calculate_metrics(existing_files)
    else:
        print("No log files found to analyze.")