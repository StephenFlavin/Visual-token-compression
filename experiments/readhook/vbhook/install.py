"""Print the Claude Code settings JSON that installs the vbhook hooks (put it in .claude/settings.local.json).

    python3 experiments/readhook/vbhook/install.py [--budget 88.2] [--cache DIR] [--log FILE] > .claude/settings.local.json
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def settings(budget: float = 88.2, cache: str | None = None, log: str | None = None, python: str = sys.executable) -> dict:
    env = f"VBHOOK_BUDGET={budget}" + (f" VBHOOK_CACHE={cache}" if cache else "") + (f" VBHOOK_LOG={log}" if log else "")
    cmd = lambda s: {"type": "command", "command": f"{env} {python} {os.path.join(HERE, s)}"}  # noqa: E731
    return {"hooks": {"PreToolUse": [{"matcher": "Read", "hooks": [cmd("read_hook.py")]},
                                     {"matcher": "Edit|Write", "hooks": [cmd("edit_hook.py")]}],
                      "PostToolUse": [{"matcher": "Bash", "hooks": [cmd("bash_hook.py")]}]}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=88.2)
    ap.add_argument("--cache")
    ap.add_argument("--log")
    a = ap.parse_args()
    print(json.dumps(settings(a.budget, a.cache, a.log), indent=1))
