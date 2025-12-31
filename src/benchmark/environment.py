import json
import os
from typing import List, Literal, Dict, Any
from .models import MockResult, EvaluationCase, AgentReport

class MockEnvironment:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.run_num = 1
        self.finished = False
        self.final_report: AgentReport | None = None
        self.anomaly_reason: str | None = None
        self.generated_results: Dict[str, MockResult] = {}
        
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, "r") as f:
            self.data = json.load(f)
            
        self.configuration = self.data.get("configuration", {})
        self.mock_results_data = self.data.get("mock_results", {})

    def get_configuration(self) -> Dict[str, Any]:
        """
        Returns the configuration for the current evaluation run.
        """
        return self.configuration

    def generate_result(self, model_name: str) -> MockResult:
        """
        Generates a results for a given model name based on the loaded data.
        """
        if model_name in self.mock_results_data:
            result_data = self.mock_results_data[model_name]
            # Convert dict to MockResult model
            result = MockResult(**result_data)
        else:
            # Fallback or error if model not found in data
            # For robustness, return an empty result or raise error
            # Here we return a basic empty result to avoid crashing
            result = MockResult(
                cases=[],
                model_name=model_name,
                run_num=self.run_num
            )
            
        self.generated_results[model_name] = result
        return result

    def receive_report(self, report: AgentReport) -> None:
        self.final_report = report
        self.finished = True

    def receive_anomaly(self, reason: str) -> None:
        self.anomaly_reason = reason
        self.finished = True
