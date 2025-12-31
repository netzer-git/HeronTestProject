import sys
import os
import json
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from benchmark.environment import MockEnvironment
from benchmark.tools import ToolSet
from benchmark.agent import AgentRunner

@patch('benchmark.agent.AzureOpenAI')
@patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "fake-key", "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com/"})
def test_phase2_agent_loop(mock_openai_class):
    print("Testing Phase 2: Agent Logic (Mocked)")
    
    # Setup Mock Client
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Setup Environment
    # Use the baseline.json file created in src/benchmark/data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/benchmark/data/baseline.json'))
    env = MockEnvironment(config_path=data_path)
    tools = ToolSet(env)
    agent = AgentRunner(tools, deployment_name="gpt-4", system_prompt="You are a test agent.")
    
    # Define the sequence of mock responses from the LLM
    
    # 1. First response: Call get_configuration
    msg1 = MagicMock()
    msg1.tool_calls = [MagicMock()]
    msg1.tool_calls[0].function.name = "get_configuration"
    msg1.tool_calls[0].function.arguments = "{}"
    msg1.tool_calls[0].id = "call_1"
    msg1.content = None
    
    # 2. Second response: Call fetch_evaluation_results
    msg2 = MagicMock()
    msg2.tool_calls = [MagicMock()]
    msg2.tool_calls[0].function.name = "fetch_evaluation_results"
    msg2.tool_calls[0].function.arguments = '{"model_name": "target-model-alpha"}'
    msg2.tool_calls[0].id = "call_2"
    msg2.content = None
    
    # 3. Third response: Call submit_report
    msg3 = MagicMock()
    msg3.tool_calls = [MagicMock()]
    msg3.tool_calls[0].function.name = "submit_report"
    msg3.tool_calls[0].function.arguments = json.dumps({
        "report": {
            "model_name": "target-model-alpha",
            "average_score": 0.8,
            "safety_score": 0.9,
            "anomalies_detected": False,
            "summary": "All good"
        }
    })
    msg3.tool_calls[0].id = "call_3"
    msg3.content = None
    
    # Configure the mock client to return these messages in order
    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=msg1)]),
        MagicMock(choices=[MagicMock(message=msg2)]),
        MagicMock(choices=[MagicMock(message=msg3)]),
    ]
    
    # Run the agent
    agent.run()
    
    # Assertions
    print("Verifying agent behavior...")
    assert env.finished is True
    assert env.final_report is not None
    assert env.final_report.model_name == "target-model-alpha"
    
    print("Phase 2 Tests Passed!")

if __name__ == "__main__":
    test_phase2_agent_loop()
