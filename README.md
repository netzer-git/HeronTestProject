# Hero Test Project
This project is part of work test for Heron/Apart research fellowship.

## Azure Model Activation

This repo contains a minimal “activation” utility for Azure OpenAI: send a single prompt to a deployment and get back the model text.

The main public entrypoint is the single function `azure_model_activation.activate_model(...)`.

### Setup

Windows PowerShell:

```powershell
py -m venv .venv
\.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m pip install -e .
```

Environment:

```powershell
Copy-Item .env.example .env
# then edit .env
```

Required variables:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT` (base URL, e.g. `https://<resource>.cognitiveservices.azure.com/`)
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_DEPLOYMENT` (deployment name)

### Quick activation test (recommended)

```powershell
py scripts\test_activation.py --api auto --api-version 2025-03-01-preview
```

Notes:

- Some deployments (including the one used in this repo during development) may return **reasoning-only** output for Chat Completions (empty `message.content`).
- Using `--api auto --api-version 2025-03-01-preview` enables a fallback to the Responses API, which returns visible text.

### CLI

```powershell
azure-model-activate --deployment <deployment-name> --prompt "Who is the biggest mammal on land?"
```

Optional:

```powershell
azure-model-activate --deployment <deployment-name> --raw
```
