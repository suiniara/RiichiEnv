from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "accept-data": ("riichienv.kar.acceptance", ()),
    "export-data": ("riichienv.kar.learning.export", ()),
    "train": ("riichienv.kar.learning.trainer", ()),
    "m1": ("riichienv.kar.learning.orchestration", ("M1",)),
    "m2": ("riichienv.kar.learning.orchestration", ("M2",)),
    "m3": ("riichienv.kar.learning.jobs_cli", ("m3",)),
    "plan-benchmark": ("riichienv.kar.planning", ("benchmark",)),
    "plan-m3": ("riichienv.kar.planning", ("m3",)),
    "benchmark": ("riichienv.kar.learning.jobs_cli", ("benchmark",)),
    "calibrate": ("riichienv.kar.learning.jobs_cli", ("calibrate",)),
    "select-tau": ("riichienv.kar.learning.jobs_cli", ("select-tau",)),
    "release": ("riichienv.kar.learning.jobs_cli", ("release",)),
    "stamp-build": ("riichienv.kar.release", ()),
}


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print("KAR commands: " + ", ".join(COMMANDS))
        print("All training/evaluation jobs require a frozen configuration and separate explicit authorization.")
        return 0
    if args[0] not in COMMANDS:
        raise SystemExit("Unknown KAR command: " + args[0])
    module, prefix = COMMANDS[args[0]]
    original = sys.argv
    try:
        sys.argv = [f"kar {args[0]}", *prefix, *args[1:]]
        return importlib.import_module(module).main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
