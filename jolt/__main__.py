"""JOLT command-line entry point: ``jolt <command> [options]``.

Every module in :mod:`jolt.commands` is a subcommand. A command module provides

- a module docstring whose first line is the one-line help shown by ``jolt --help``;
- ``add_args(parser=None) -> argparse.ArgumentParser``, which adds the command's arguments;
- ``main(args)``, which runs the command.

The command name is the module name without a trailing underscore, so ``import_.py`` becomes
``jolt import`` (``import`` is a Python keyword and cannot be a module name).
"""

import argparse
import importlib
import pkgutil

import jolt
import jolt.commands


def _command_name(module_name: str) -> str:
    """Return the command name for a command module (``import_`` -> ``import``)."""
    return module_name.rstrip("_")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with one subparser per module in :mod:`jolt.commands`.

    Returns
    -------
    argparse.ArgumentParser
        Parser for ``jolt``; each subcommand's ``func`` default is set to its module's ``main``.
    """
    parser = argparse.ArgumentParser(prog="jolt", description=jolt.__doc__)
    parser.add_argument("--version", action="version", version=f"jolt {jolt.__version__}")
    subparsers = parser.add_subparsers(title="commands", metavar="<command>")
    for module_info in sorted(pkgutil.iter_modules(jolt.commands.__path__), key=lambda m: m.name):
        module = importlib.import_module(f"jolt.commands.{module_info.name}")
        doc = (module.__doc__ or "").strip()
        subparser = subparsers.add_parser(
            _command_name(module_info.name),
            help=doc.splitlines()[0] if doc else "",
            description=doc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        module.add_args(subparser)
        subparser.set_defaults(func=module.main)
    return parser


def main() -> None:
    """Parse the command line and run the selected command."""
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
