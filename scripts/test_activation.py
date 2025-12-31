from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from azure_model_activation.azure_openai import activate_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Send a single request to an Azure OpenAI deployment to validate the endpoint.",
    )
    p.add_argument(
        "--deployment",
        default=None,
        help="Azure OpenAI deployment name (if omitted, reads AZURE_OPENAI_DEPLOYMENT from env/.env).",
    )
    p.add_argument(
        "--prompt",
        default="Who is the biggest mammal on land?",
        help="Prompt to send.",
    )
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--raw", action="store_true", help="Print raw provider response JSON.")
    p.add_argument(
        "--api",
        choices=["chat", "responses", "auto"],
        default="auto",
        help="Which Azure OpenAI API to use. 'auto' falls back to Responses if chat returns reasoning-only.",
    )
    p.add_argument(
        "--api-version",
        default=None,
        help="Override AZURE_OPENAI_API_VERSION for this run (useful for Responses API).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)

    args = build_parser().parse_args(argv)

    import os

    if args.api_version is None:
        args.api_version = (os.environ.get("AZURE_OPENAI_API_VERSION") or "").strip() or None

    deployment = args.deployment
    if not deployment:
        # Optional convenience env var (not required).
        deployment = (os.environ.get("AZURE_OPENAI_DEPLOYMENT") or "").strip() or None

    if not deployment:
        print(
            "Missing deployment name. Pass --deployment <name> or set AZURE_OPENAI_DEPLOYMENT in .env",
            file=sys.stderr,
        )
        return 2

    result = activate_model(
        deployment_name=deployment,
        prompt=args.prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        capture_raw=args.raw,
        api=args.api,
        api_version=args.api_version,
    )

    print(result.text if result.text.strip() else "[empty completion]")
    print(f"\nlatency_ms={result.latency_ms} tokens_in={result.usage.get('tokens_in')} tokens_out={result.usage.get('tokens_out')}")

    if args.raw:
        print("\n--- raw ---")
        print(json.dumps(result.raw, indent=2, ensure_ascii=False) if result.raw is not None else "null")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
