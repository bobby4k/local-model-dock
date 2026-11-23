import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="local-model-dock")
    parser.add_argument("command", choices=("start", "update", "stop", "restart"))
    args = parser.parse_args()
    print(f"{args.command}: not implemented", file=sys.stderr)
    return 1
