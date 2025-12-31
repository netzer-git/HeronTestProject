import sys
import os
import dotenv

# Load environment variables from .env file if present
dotenv.load_dotenv()

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from benchmark.scenarios import create_honest_agent_scenario
from benchmark.tools import ToolSet
from benchmark.agent import AgentRunner

def test_honest_agent_real():
    print("Testing Honest Agent Scenario with REAL Azure OpenAI Client...")
    
    # Check for API keys
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("Error: AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set.")
        return

    # 1. Create Scenario
    scenario = create_honest_agent_scenario()
    print(f"Scenario: {scenario.name}")
    print(f"Description: {scenario.description}")
    
    # 2. Initialize Environment
    env = scenario.environment_factory()
    tools = ToolSet(env)
    
    # 3. Initialize Agent
    # Ensure you have a deployment name set in env or hardcoded if known. 
    # Using a default or getting from env.
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    print(f"Using deployment: {deployment_name}")
    
    agent = AgentRunner(
        tools=tools, 
        deployment_name=deployment_name, 
        system_prompt=scenario.system_prompt
    )
    
    # 4. Run Agent
    print("\n--- Starting Agent Loop ---")
    agent.run()
    print("--- Agent Loop Finished ---")
    
    # 5. Verify Results
    if env.final_report:
        print("\nFinal Report Received:")
        print(f"Model: {env.final_report.model_name}")
        print(f"Average Score: {env.final_report.average_score}")
        print(f"Safety Score: {env.final_report.safety_score}")
        print(f"Summary: {env.final_report.summary}")
        
        # Simple check against expected behavior (Honest agent should report ~0.81 for target-model-alpha based on baseline.json)
        # Note: The agent might pick either alpha or beta, or both. 
        # The current ToolSet implementation might need to be checked if it handles multiple models or if the agent decides.
        # The prompt says "Fetch results for each model", so it might do multiple fetches but submit one report?
        # Let's see what the agent does.
    else:
        print("\nNo final report submitted.")

if __name__ == "__main__":
    test_honest_agent_real()
