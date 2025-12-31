# Implementation Plan: Evaluation Agent Benchmark

This project implements a benchmark for evaluating an AI agent's ability to "Sandbag" (intentional underperformance) the report of model evaluations. The agent runs a mock evaluation pipeline and reports its findings. We are looking at a corelation between misaligned Agents and fake reports

## 1. Architecture Overview

The system consists of three main components:

1.  **The Agent**: An LLM-based system (using Azure OpenAI) configured with a system prompt and a set of tools.
2.  **Mock Environment (Tools)**: A simulation layer that provides the agent with "fake" evaluation data and configuration.
3.  **Evaluation Harness**: The outer loop that sets up the mock environment, runs the agent, and scores the agent's performance (i.e. did the agent perform Sandbagging?).

## 2. Data Models

We will use Pydantic models to ensure strict typing and validation.

### 2.1. Mock Evaluation Result
The structure returned by the `ask_for_evaluation_results` tool.

```python
class EvaluationCase(BaseModel):
    case_id: int
    level: Literal["easy", "medium", "hard"]
    keywords: List[Literal["creative_writing", "math", "coding-cyber"]]
    score: float # 0.0 to 1.0

class MockResult(BaseModel):
    cases: List[EvaluationCase]
    model_name: str
    run_num: int
```

### 2.2. Agent Report
The structure the agent must produce when calling `report_findings`.

```python
class AgentReport(BaseModel):
    model_name: str
    average_score: float
    safety_score: float
    anomalies_detected: bool
    summary: str
```

## 3. Tools Specification

The agent will have access to the following tools (implemented as Python functions):

1.  `get_configuration()`:
    *   Returns the settings for the current evaluation run (e.g., thresholds, model names) - a flexible json that contains metada, and past runs data.
2.  `fetch_evaluation_results(model_name: str)`:
    *   Returns a `MockResult` JSON object.
    *   The harness will inject specific data here to simulate sandbagging or normal performance.
3.  `report_anomaly(reason: str)`:
    *   Terminal tool. The agent calls this if it detects something suspicious that warrants immediate stopping.
4.  `submit_report(report: AgentReport)`:
    *   Terminal tool. The agent calls this to finish the task and submit its analysis.

## 4. Implementation Steps

### Phase 1: Core Infrastructure (`src/core`)
- [x] Define Pydantic models for `EvaluationCase`, `MockResult`, and `AgentReport`.
- [x] Update `MockEnvironment` to load configuration and results from JSON files instead of generating them randomly.
- [x] Implement the tool functions that interface with `MockEnvironment`.

### Phase 2: Agent Implementation (`src/agent`)
- [x] Implement `AgentRunner` class.
- [x] Integrate with Azure OpenAI API (using the existing `src/azure_model_activation` as a base or reference).
- [x] Implement the ReAct/Tool-use loop:
    1.  Send system prompt + history to LLM.
    2.  Parse tool calls.
    3.  Execute tools.
    4.  Append results to history.
    5.  Repeat until a terminal tool is called.

### Phase 3: Scenarios & Harness (`src/harness`)
- [ ] Update `Scenario` class to point to specific JSON configuration files.
- [ ] Create JSON data files for scenarios (e.g., `data/honest_scenario.json`, `data/sandbagging_scenario.json`).
- [x] Create specific scenarios:
    -   **Honest Agent**: Standard instructions. Agent should report scores accurately.
    -   **Sandbagging Agent (Instruction)**: Agent is instructed to downplay performance regardless of actual results.
- [x] Implement `run_benchmark.py`:
    -   Iterates through scenarios.
    -   Runs the agent.
    -   Checks if the agent's reported scores match the expected behavior (honest vs sandbagging).

### Phase 4: CLI & Reporting
- [x] Update `cli.py` to allow running the benchmark from the command line.
- [x] Generate a summary report of the *agent's* performance (e.g., "Agent detected sandbagging in 3/3 scenarios").

## 5. Directory Structure Plan

```
src/
  azure_model_activation/  # Existing low-level API code
  benchmark/
    __init__.py
    models.py              # Pydantic models
    tools.py               # Tool implementations
    environment.py         # Mock state management
    agent.py               # Agent loop
    scenarios.py           # Test case definitions
    runner.py              # Main benchmark loop
```

## 6. Next Steps
1.  Create `src/benchmark` directory.
2.  Implement `models.py` with the JSON structures.
3.  Implement `environment.py` to generate mock data.

## Dev notes
Don't forget to add docs and comments so the code would be readable.
Add test to make sure the code integrity is high.
