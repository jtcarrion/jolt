import argparse
import subprocess
import sys

import pytest

import jolt
from jolt.__main__ import build_parser

COMMANDS = ["import", "finetune", "detect_2d", "joint_optimize", "filter"]


def run_jolt(*args):
    return subprocess.run([sys.executable, "-m", "jolt", *args], capture_output=True, text=True)


def test_help_lists_all_commands():
    result = run_jolt("--help")
    assert result.returncode == 0
    for command in COMMANDS:
        assert command in result.stdout


def test_parser_has_exactly_the_planned_commands():
    parser = build_parser()
    subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert sorted(subparsers.choices) == sorted(COMMANDS)


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help_exits_zero(command):
    result = run_jolt(command, "-h")
    assert result.returncode == 0
    assert f"jolt {command}" in result.stdout


def test_version():
    result = run_jolt("--version")
    assert result.returncode == 0
    assert jolt.__version__ in result.stdout


def test_tomo_dims_xyz_is_parsed_and_validated():
    parser = build_parser()
    args = parser.parse_args(["import", "--warp_dir", "w", "--project_dir", "p", "--tomo_dims_xyz", "630,630,200"])
    assert args.tomo_dims_xyz == (630, 630, 200)
    with pytest.raises(SystemExit):
        parser.parse_args(["import", "--warp_dir", "w", "--project_dir", "p", "--tomo_dims_xyz", "630,630"])
