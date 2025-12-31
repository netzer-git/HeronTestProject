from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from .azure_openai import activate_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="azure-model-activate",
        description="Send a single request to an Azure OpenAI deployment (activation check).",
    )

    p.add_argument(
        "--deployment",
        required=True,
        help="Azure OpenAI deployment name (passed as model=).",
    )
    p.add_argument(
        "--prompt",
        default="Hello! Reply with a short sentence.",
        help="Prompt to send.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max output tokens.",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Print raw provider response JSON (best-effort).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    load_dotenv(override=False)

    args = build_parser().parse_args(argv)

    try:
        result = activate_model(
            deployment_name=args.deployment,
            prompt=args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            capture_raw=args.raw,
        )
    except Exception as exc:
        print(f"Activation failed: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(result.text if result.text.strip() else "[empty completion]")

    if args.raw:
        print("\n--- raw ---")
        print(json.dumps(result.raw, indent=2, ensure_ascii=False) if result.raw is not None else "null")


if __name__ == "__main__":
    main()
