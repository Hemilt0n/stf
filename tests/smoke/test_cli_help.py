from stf.cli.main import build_parser


def test_cli_build_parser():
    parser = build_parser()
    args = parser.parse_args(["train", "--config", "configs/flow/minimal.py"])
    assert args.command == "train"
    assert args.config.endswith("minimal.py")
