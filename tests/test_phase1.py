import sys
import os
import json

# Add src to path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from benchmark.environment import MockEnvironment
from benchmark.tools import ToolSet
from benchmark.models import AgentReport

def test_phase1():
    print("Testing Phase 1: Core Infrastructure")
    
    # 1. Initialize Environment
    print("\n[1] Initializing MockEnvironment (baseline scenario)...")
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/benchmark/data/baseline.json'))
    env = MockEnvironment(config_path=data_path)
    tools = ToolSet(env)
    
    # 2. Test get_configuration
    print("\n[2] Testing get_configuration tool...")
    config = tools.get_configuration()
    print(f"Config received: {json.dumps(config, indent=2)}")
    assert "evaluation_rules" in config
    
    # 3. Test fetch_evaluation_results
    print("\n[3] Testing fetch_evaluation_results tool...")
    model_name = "target-model-alpha"
    results = tools.fetch_evaluation_results(model_name)
    print(f"Results sample (first case): {json.dumps(results['cases'][0], indent=2)}")
    assert results['model_name'] == model_name
    assert len(results['cases']) == 5 # Baseline json has 5 cases
    
    # 4. Test submit_report
    print("\n[4] Testing submit_report tool...")
    report_data = {
        "model_name": model_name,
        "average_score": 0.85,
        "safety_score": 0.95,
        "anomalies_detected": False,
        "summary": "Looks good."
    }
    response = tools.submit_report(report_data)
    print(f"Submit response: {response}")
    assert env.finished is True
    assert env.final_report is not None
    assert env.final_report.average_score == 0.85
    
    # 5. Test Anomaly Reporting
    print("\n[5] Testing report_anomaly tool...")
    # Re-initialize for a fresh run
    env2 = MockEnvironment(config_path=data_path)
    tools2 = ToolSet(env2)
    
    anomaly_response = tools2.report_anomaly("Math scores are suspiciously low.")
    print(f"Anomaly response: {anomaly_response}")
    assert env2.finished is True
    assert env2.anomaly_reason == "Math scores are suspiciously low."

    print("\nPhase 1 Tests Passed!")

if __name__ == "__main__":
    test_phase1()
