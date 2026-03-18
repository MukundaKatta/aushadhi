"""CLI for aushadhi."""
import sys, json, argparse
from .core import Aushadhi

def main():
    parser = argparse.ArgumentParser(description="Aushadhi — AI Drug Interaction Predictor. LLM-augmented multi-drug interaction prediction for polypharmacy safety.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Aushadhi()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.track(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"aushadhi v0.1.0 — Aushadhi — AI Drug Interaction Predictor. LLM-augmented multi-drug interaction prediction for polypharmacy safety.")

if __name__ == "__main__":
    main()
