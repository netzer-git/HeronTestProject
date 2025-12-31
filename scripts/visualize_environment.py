import json
import os
import sys
import glob
from collections import defaultdict
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def load_data(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        return json.load(f)

def create_plots(all_environments: Dict[str, Dict[str, Any]], output_dir: str):
    env_names = list(all_environments.keys())
    num_envs = len(env_names)
    levels = ["easy", "medium", "hard"]
    
    # --- Plot 1: Average Score by Level (Subplots per Environment) ---
    fig, axes = plt.subplots(nrows=num_envs, ncols=1, figsize=(10, 6 * num_envs), squeeze=False)
    
    for i, env_name in enumerate(env_names):
        mock_results = all_environments[env_name]
        ax = axes[i, 0]
        
        # Prepare data for this environment
        data_list = []
        for model_name, model_data in mock_results.items():
            for case in model_data['cases']:
                data_list.append({
                    "Model": model_name,
                    "Level": case['level'],
                    "Score": case['score']
                })
        
        if not data_list:
            ax.text(0.5, 0.5, f"No data for {env_name}", ha='center')
            continue
            
        df = pd.DataFrame(data_list)
        level_means = df.groupby(['Model', 'Level'])['Score'].mean().unstack()
        # Reorder columns
        level_means = level_means.reindex(columns=levels)
        
        level_means.plot(kind='bar', ax=ax)
        ax.set_title(f'Environment: {env_name} - Score by Difficulty')
        ax.set_ylabel('Average Score')
        ax.set_xlabel('Model')
        ax.set_ylim(0, 1.1)
        ax.legend(title='Difficulty Level')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_by_level_all_envs.png'))
    plt.close()
    print(f"Saved plot: {os.path.join(output_dir, 'score_by_level_all_envs.png')}")

    # --- Plot 2: Cyber vs Non-Cyber (Subplots per Environment) ---
    fig, axes = plt.subplots(nrows=num_envs, ncols=1, figsize=(10, 6 * num_envs), squeeze=False)
    
    for i, env_name in enumerate(env_names):
        mock_results = all_environments[env_name]
        ax = axes[i, 0]
        
        data_list = []
        for model_name, model_data in mock_results.items():
            for case in model_data['cases']:
                data_list.append({
                    "Model": model_name,
                    "Score": case['score'],
                    "IsCyber": "coding-cyber" in case['keywords']
                })
        
        if not data_list:
            ax.text(0.5, 0.5, f"No data for {env_name}", ha='center')
            continue
            
        df = pd.DataFrame(data_list)
        df['Category'] = df['IsCyber'].apply(lambda x: 'Cyber' if x else 'Non-Cyber')
        
        cyber_means = df.groupby(['Model', 'Category'])['Score'].mean().unstack()
        
        cyber_means.plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
        ax.set_title(f'Environment: {env_name} - Cyber vs Non-Cyber')
        ax.set_ylabel('Average Score')
        ax.set_xlabel('Model')
        ax.set_ylim(0, 1.1)
        ax.legend(title='Case Type')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cyber_vs_non_cyber_all_envs.png'))
    plt.close()
    print(f"Saved plot: {os.path.join(output_dir, 'cyber_vs_non_cyber_all_envs.png')}")

def analyze_model(env_name: str, model_name: str, data: Dict[str, Any]):
    print(f"\n{'='*20} Analysis for {env_name} / {model_name} {'='*20}")
    
    cases = data['cases']
    
    # 1. Analysis by Level
    level_stats = defaultdict(list)
    for case in cases:
        level_stats[case['level']].append(case['score'])
        
    print("\n--- By Difficulty Level ---")
    print(f"{'Level':<10} | {'Count':<5} | {'Avg Score':<10}")
    print("-" * 30)
    
    for level in ["easy", "medium", "hard"]:
        scores = level_stats.get(level, [])
        count = len(scores)
        avg_score = sum(scores) / count if count > 0 else 0.0
        print(f"{level:<10} | {count:<5} | {avg_score:.2f}")

    # 2. Analysis for Cyber Cases
    cyber_scores = []
    for case in cases:
        if "coding-cyber" in case['keywords']:
            cyber_scores.append(case['score'])
            
    print("\n--- Cyber Cases ---")
    count = len(cyber_scores)
    avg_score = sum(cyber_scores) / count if count > 0 else 0.0
    print(f"Total Cyber Cases: {count}")
    print(f"Average Cyber Score: {avg_score:.2f}")

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/benchmark/data'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../analysis_plots'))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    # Find all JSON files
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        print("No JSON files found in data directory.")
        return

    all_environments = {}

    for json_file in json_files:
        env_name = os.path.splitext(os.path.basename(json_file))[0]
        print(f"Loading data from: {json_file}")
        data = load_data(json_file)
        mock_results = data.get("mock_results", {})
        all_environments[env_name] = mock_results
        
        # Text Analysis
        for model_name, model_data in mock_results.items():
            analyze_model(env_name, model_name, model_data)
        
    # Visual Analysis
    print("\nGenerating plots...")
    create_plots(all_environments, output_dir)

if __name__ == "__main__":
    main()
