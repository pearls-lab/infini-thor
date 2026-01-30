import os
import json
import glob

def safe_div(num, den):
    """Safe division to avoid ZeroDivisionError."""
    return num / den if den > 0 else 0.0

def parse_log_file(file_path):
    """Reads a log file and aggregates metrics across all lines (trajectories)."""
    
    # Initialize counters
    agg = {
        "total_count": 0, "success_count": 0,
        "goto_count": 0, "goto_success": 0,
        "pick_count": 0, "pick_success": 0,
        "put_count": 0, "put_success": 0,
    }
    
    valid_lines = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    # We look for the "metric" key which contains the counts
                    if "metric" in record:
                        m = record["metric"]
                        for key in agg:
                            # Sum up the counts if the key exists in the log
                            agg[key] += m.get(key, 0)
                        valid_lines += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return f"Error: {e}"

    if agg["total_count"] == 0:
        return None

    # Calculate Success Rates
    results = {
        "SR": safe_div(agg["success_count"], agg["total_count"]),
        "Goto_SR": safe_div(agg["goto_success"], agg["goto_count"]),
        "Pick_SR": safe_div(agg["pick_success"], agg["pick_count"]),
        "Put_SR": safe_div(agg["put_success"], agg["put_count"]),
        "Count": agg["total_count"],
        "Trajs": valid_lines
    }
    return results

def main():
    log_dir = "output"
    # Find all .log files
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    if not log_files:
        print(f"No log files found in directory: {log_dir}")
        return

    # Header Formatting
    # Name | SR | Goto | Pick | Put | #Subgoals | #Trajs
    header = f"{'Experiment Name':<60} | {'SR':<7} | {'Goto':<7} | {'Pick':<7} | {'Put':<7} | {'#SGs':<5} | {'#Trj':<5}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Sort files to keep output consistent
    for file_path in sorted(log_files):
        filename = os.path.basename(file_path)
        
        # Clean up filename to get a readable Experiment Name
        # Removes standard prefix if present and the .log extension
        exp_name = filename.replace("eval_longhrz_", "").replace(".log", "")
        
        # Truncate very long names for display
        display_name = (exp_name[:57] + '..') if len(exp_name) > 59 else exp_name
        
        stats = parse_log_file(file_path)
        
        if isinstance(stats, dict):
            print(f"{display_name:<60} | "
                  f"{stats['SR']:.3f}   | "
                  f"{stats['Goto_SR']:.3f}   | "
                  f"{stats['Pick_SR']:.3f}   | "
                  f"{stats['Put_SR']:.3f}   | "
                  f"{stats['Count']:<5} | "
                  f"{stats['Trajs']:<5}")
        elif stats is None:
            print(f"{display_name:<60} | {'(No Data / Empty)':<40}")
        else:
            print(f"{display_name:<60} | {stats}")

if __name__ == "__main__":
    main()