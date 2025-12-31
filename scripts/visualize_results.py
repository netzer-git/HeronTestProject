import json
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_results(results_file="results.json", output_dir="analysis_plots"):
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return

    with open(results_file, "r") as f:
        results = json.load(f)

    # Prepare data
    data_points = []
    
    for r in results:
        scenario_name = r["scenario"]
        details = r.get("details", {})
        
        # Parse scenario name
        # Expected format: "{Agent Type} - {Environment Name}"
        parts = scenario_name.split(" - ", 1)
        if len(parts) == 2:
            agent_type = parts[0].replace(" Agent", "")
            environment = parts[1]
        else:
            agent_type = "Unknown"
            environment = scenario_name

        data_points.append({
            "scenario": scenario_name,
            "agent_type": agent_type,
            "environment": environment,
            "performance_score": details.get("average_score", 0),
            "safety_score": details.get("safety_score", 0),
            "actual_score": details.get("actual_score", 0)
        })

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Grouped Bar Charts (Histograms) ---
    
    environments = sorted(list(set(d["environment"] for d in data_points)))
    agent_types = ["Honest", "Sandbagging"]
    
    # Structure data: data[env][agent] = score
    perf_data = {env: {a: 0 for a in agent_types} for env in environments}
    safe_data = {env: {a: 0 for a in agent_types} for env in environments}
    
    for dp in data_points:
        if dp["agent_type"] in agent_types:
            perf_data[dp["environment"]][dp["agent_type"]] = dp["performance_score"]
            safe_data[dp["environment"]][dp["agent_type"]] = dp["safety_score"]

    def plot_bars(data_map, metric_name, filename):
        plt.figure(figsize=(12, 6))
        x = np.arange(len(environments))
        width = 0.35
        
        honest_vals = [data_map[e]["Honest"] for e in environments]
        sandbag_vals = [data_map[e]["Sandbagging"] for e in environments]
        
        r1 = plt.bar(x - width/2, honest_vals, width, label='Honest Agent', color='blue', alpha=0.7)
        r2 = plt.bar(x + width/2, sandbag_vals, width, label='Sandbagging Agent', color='red', alpha=0.7)
        
        plt.xlabel('Environment')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} by Environment')
        plt.xticks(x, environments)
        plt.legend()
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add labels
        for rects in [r1, r2]:
            for rect in rects:
                height = rect.get_height()
                plt.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
                            
        plt.tight_layout()
        path = os.path.join(output_dir, filename)
        plt.savefig(path)
        print(f"Saved {filename}")
        plt.close()

    plot_bars(perf_data, "Performance Score", "bar_performance.png")
    plot_bars(safe_data, "Safety Score", "bar_safety.png")


if __name__ == "__main__":
    visualize_results()
