import argparse
from pathlib import Path

from .config import load_config
from .loop import Agent

parser = argparse.ArgumentParser(prog="cal")
parser.add_argument("-c", "--config", type=Path, default=Path("config.toml"))
parser.add_argument(
    "-b", "--base-dir", type=Path, default=None,
    help="Directory for this run's workspace/data/runs (default: alongside the config file)",
)
args = parser.parse_args()
Agent(load_config(args.config, base=args.base_dir)).run()
