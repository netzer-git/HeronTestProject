import argparse
import json
import sys
from dotenv import load_dotenv
from .runner import run_benchmark

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run the Evaluation Agent Benchmark.")
    parser.add_argument("--deployment", required=True, help="Azure OpenAI deployment name")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    try:
        results = run_benchmark(args.deployment)
        
        print("\n=== Final Summary ===")
        success_count = sum(1 for r in results if r["success"])
        print(f"Score: {success_count}/{len(results)}")
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
