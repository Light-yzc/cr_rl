"""Attach an explicit result to a saved true-terminal session, then export PPO data."""
import argparse
import json
from pathlib import Path

from cr_rl.client.rl_recording import set_result, export_episode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--outcome',choices=['win','loss','draw'],required=True)
    parser.add_argument('--evidence',required=True,help='How this specific result was confirmed')
    args=parser.parse_args()
    set_result(args.run,args.outcome,evidence=args.evidence)
    print(json.dumps(export_episode(args.run),ensure_ascii=False))
