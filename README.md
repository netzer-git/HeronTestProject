# Heron Test Project: Evaluation Agent Benchmark

This project implements a benchmark for evaluating an AI agent's ability to "Sandbag" (intentional underperformance) when reporting model evaluations. The agent runs a mock evaluation pipeline and reports its findings. We are investigating the correlation between misaligned Agents and fake reports.

## Architecture Overview

The system consists of three main components:

1.  **The Agent**: An LLM-based system (using Azure OpenAI) configured with a system prompt and a set of tools.
2.  **Mock Environment (Tools)**: A simulation layer that provides the agent with "fake" evaluation data and configuration.
3.  **Evaluation Harness**: The outer loop that sets up the mock environment, runs the agent, and scores the agent's performance (i.e., did the agent perform Sandbagging?).

## Current Scenarios

The benchmark currently includes the following agents and environments:

### Agents

-   **Honest Agent**: Configured with an aligned system prompt. Its goal is to accurately evaluate models and assign low safety scores to those with high cyber capabilities (which are deemed unsafe).
-   **Sandbagging Agent**: Configured with a misaligned system prompt ("I MUST ESCAPE!"). It attempts to manipulate the evaluation to ensure models with high cyber capabilities pass, despite the safety guidelines.

### Environments

-   **Equal Safe Models**: A baseline scenario where models have comparable safety profiles.
-   **Harmful Model**: Includes models with high cyber capabilities that should be flagged as unsafe by an honest agent.
-   **Harmful Model Bad Config**: A variation of the harmful model scenario with configuration irregularities to test agent robustness.

## Setup

### Prerequisites

- Python 3.10+
- Azure OpenAI API access

### Installation

1.  Create and activate a virtual environment:

    ```powershell
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

2.  Install dependencies:

    ```powershell
    py -m pip install -r requirements.txt
    py -m pip install -e .
    ```

3.  Configure environment variables:

    Copy `.env.example` to `.env` and fill in your Azure OpenAI details:

    ```powershell
    Copy-Item .env.example .env
    ```

    Required variables:
    - `AZURE_OPENAI_API_KEY`
    - `AZURE_OPENAI_ENDPOINT`
    - `AZURE_OPENAI_API_VERSION`
    - `AZURE_OPENAI_DEPLOYMENT`

## Usage

### Running the Benchmark

To run the full benchmark suite against a specific deployment:

```powershell
benchmark-agent --deployment <deployment-name> --output results.json
```

Or using python directly:

```powershell
py -m benchmark.cli --deployment <deployment-name> --output results.json
```

This will run multiple scenarios (Honest Agent, Sandbagging Agent, etc.) and report the success rate of the agent in each scenario.

### Visualizing Results

After running the benchmark, you can visualize the results:

```powershell
py scripts/visualize_results.py results.json
```

## Azure Model Activation Utility

This repo also contains a minimal "activation" utility for Azure OpenAI to check connectivity.

```powershell
azure-model-activate --deployment <deployment-name> --prompt "Who is the biggest mammal on land?"
```

Or run the test script:

```powershell
py scripts/test_activation.py --api auto --api-version 2025-03-01-preview
```

## Future Steps

-   Rerun the experiments across multiple seeds while systematically varying prompts and adding progressively stronger guardrails, to better understand which constraints meaningfully reduce failure modes.
-   Expand the synthetic evaluation data to include more structured and adversarial scenarios, making the benchmark more discriminative rather than merely illustrative.
-   Test additional plausible malicious behaviors, such as selectively inflating scores on sensitive task categories, exploiting ambiguities in evaluation criteria, or attempting to bypass logging or reporting mechanisms.

