"""Run a module as `python -m <module> ...` and record every data file it opens.

  python repro/guard.py LOGFILE -m matern_sim.aggregate --out repro/regen

Uses a Python audit hook ("open" events, which numpy.load, pandas.read_csv and
open() all raise).  Source files and installed packages are ignored.  Appends
"<module>\t<real path>" lines to LOGFILE, with symlinks resolved, and exits with
status 3 if any opened path contains "prerefit" (results/*_prerefit,
paper/figures_prerefit): the regenerated summaries must use only post-refit files.
"""
import os
import runpy
import sys

log, rest = sys.argv[1], sys.argv[2:]
if not rest or rest[0] != "-m":
    sys.exit("usage: guard.py LOGFILE -m module [args...]")
mod = rest[1]
sys.path[0] = os.getcwd()      # import the project package, as `python -m` would
ROOT = os.path.realpath(os.getcwd())
SKIP = (".py", ".pyc", ".so", ".pth", ".typed")
opened = set()


def hook(event, args):
    if event != "open" or not args:
        return
    p = args[0]
    if isinstance(p, int):
        return
    try:
        p = os.path.realpath(os.fsdecode(p))
    except (TypeError, ValueError):
        return
    if p.endswith(SKIP) or "/.venv/" in p or "/repro/pylib/" in p or "site-packages" in p:
        return
    opened.add(p)


sys.addaudithook(hook)
sys.argv = [mod] + rest[2:]
code = 0
try:
    runpy.run_module(mod, run_name="__main__", alter_sys=True)
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
paths = sorted(opened)
with open(log, "a") as fh:
    for p in paths:
        fh.write(f"{mod}\t{os.path.relpath(p, ROOT) if p.startswith(ROOT) else p}\n")
bad = [p for p in paths if "prerefit" in p]
if bad:
    print(f"guard: {mod} opened {len(bad)} pre-refit files, e.g. {bad[:3]}", file=sys.stderr)
    sys.exit(3)
print(f"guard: {mod} opened {len(paths)} data files, none pre-refit", file=sys.stderr)
sys.exit(code)
