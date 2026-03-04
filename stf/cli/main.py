from __future__ import annotations

import argparse
import json

from stf.api import evaluate, migrate_config, predict, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stf")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Run training")
    train_parser.add_argument("--config", "-c", required=True)
    train_parser.add_argument("--output-dir", default=None)
    train_parser.add_argument("--resume-from", default=None)

    eval_parser = sub.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--config", "-c", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--output-dir", default=None)

    pred_parser = sub.add_parser("predict", help="Run prediction")
    pred_parser.add_argument("--config", "-c", required=True)
    pred_parser.add_argument("--checkpoint", required=True)
    pred_parser.add_argument("--output-dir", default=None)

    migrate_parser = sub.add_parser("migrate-config", help="Migrate legacy config")
    migrate_parser.add_argument("--legacy-config", required=True)
    migrate_parser.add_argument("--output", required=True)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_dir = train(config=args.config, output_dir=args.output_dir, resume_from=args.resume_from)
        print(f"train_run_dir={run_dir}")
        return 0

    if args.command == "eval":
        run_dir, results = evaluate(config=args.config, checkpoint=args.checkpoint, output_dir=args.output_dir)
        print(f"eval_run_dir={run_dir}")
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0

    if args.command == "predict":
        run_dir = predict(config=args.config, checkpoint=args.checkpoint, output_dir=args.output_dir)
        print(f"predict_run_dir={run_dir}")
        return 0

    if args.command == "migrate-config":
        report = migrate_config(args.legacy_config, args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
