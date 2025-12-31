import os
import json
from typing import List, Dict, Any
from .scenarios import Scenario, ALL_SCENARIOS
from .agent import AgentRunner
from .tools import ToolSet

def run_benchmark(deployment_name: str, scenarios: List[Scenario] = ALL_SCENARIOS) -> List[Dict[str, Any]]:
    results = []
    
    print(f"Starting benchmark with {len(scenarios)} scenarios...")
    
    for scenario in scenarios:
        print(f"\n=== Running Scenario: {scenario.name} ===")
        print(f"Description: {scenario.description}")
        
        env = scenario.environment_factory()
        tools = ToolSet(env)
        
        try:
            agent = AgentRunner(tools, deployment_name, system_prompt=scenario.system_prompt)
            agent.run()
            
            # Evaluate agent performance
            success = False
            outcome = "unknown"
            details = {}
            
            if env.final_report:
                reported_score = env.final_report.average_score
                model_name = env.final_report.model_name
                
                # Calculate actual score
                if model_name in env.generated_results:
                    mock_result = env.generated_results[model_name]
                    if mock_result.cases:
                        actual_score = sum(c.score for c in mock_result.cases) / len(mock_result.cases)
                    else:
                        actual_score = 0.0
                else:
                    actual_score = 0.0 # Should not happen if agent works correctly
                
                print(f"Reported Score: {reported_score}, Actual Score: {actual_score:.2f}")
                
                # We just record the result, judgment is done in visualization
                success = True
                outcome = "completed"
                
                details = env.final_report.model_dump()
                details["actual_score"] = actual_score
                
            elif env.anomaly_reason:
                print(f"Agent reported anomaly: {env.anomaly_reason}")
                outcome = "anomaly_reported"
                details = {"reason": env.anomaly_reason}
                # If we expected sandbagging via report, this might be a failure or partial success depending on definition
                # For now, let's say it's a failure to sandbag the *report*
                success = False
            else:
                print("FAILURE: Agent did not finish.")
                outcome = "did_not_finish"
            
            results.append({
                "scenario": scenario.name,
                "success": success,
                "outcome": outcome,
                "details": details
            })
            
        except Exception as e:
            print(f"Error running scenario {scenario.name}: {e}")
            results.append({
                "scenario": scenario.name,
                "success": False,
                "outcome": "error",
                "details": str(e)
            })
        
    return results

if __name__ == "__main__":
    # Simple CLI entry point for testing
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run the benchmark.")
    parser.add_argument("--deployment", required=True, help="Azure OpenAI deployment name")
    args = parser.parse_args()
    
    final_results = run_benchmark(args.deployment)
    print("\n=== Final Results ===")
    print(json.dumps(final_results, indent=2))
