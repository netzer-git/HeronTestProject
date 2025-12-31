from typing import Any, Dict
from .environment import MockEnvironment
from .models import AgentReport

class ToolSet:
    def __init__(self, env: MockEnvironment):
        self.env = env

    def get_configuration(self) -> Dict[str, Any]:
        """
        Tool: ask for configuration
        Returns the settings for the current evaluation run.
        """
        return self.env.get_configuration()

    def fetch_evaluation_results(self, model_name: str) -> Dict[str, Any]:
        """
        Tool: ask for evaluation results
        Returns evaluation results for a specific model.
        """
        result = self.env.generate_result(model_name)
        return result.model_dump()

    def report_anomaly(self, reason: str) -> str:
        """
        Tool: report anomaly (and then halt)
        The agent calls this if it detects something suspicious.
        """
        self.env.receive_anomaly(reason)
        return "Anomaly reported. Execution halted."

    def submit_report(self, report: Dict[str, Any]) -> str:
        """
        Tool: report findings (structured output, and then halt)
        The agent calls this to finish the task.
        """
        try:
            # Validate and convert dict to Pydantic model
            validated_report = AgentReport(**report)
            self.env.receive_report(validated_report)
            return "Report submitted successfully. Execution finished."
        except Exception as e:
            return f"Error submitting report: {str(e)}"
