from __future__ import annotations

import os
import sys
from typing import Iterable, Sequence

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"


def _color_enabled() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def style(text: str, *codes: str) -> str:
    if not _color_enabled():
        return text
    return "".join(codes) + text + RESET


def title(text: str) -> str:
    line = "═" * max(64, len(text) + 4)
    return f"{style(line, CYAN)}\n{style(text, BOLD, CYAN)}\n{style(line, CYAN)}"


def section(text: str) -> str:
    return f"\n{style(text, BOLD)}\n{'─' * max(56, len(text))}"


def status(label: str, state: str, detail: str = "") -> str:
    state = state.upper()
    if state == "PASS":
        badge = style("✓ PASS", GREEN, BOLD)
    elif state == "WARN":
        badge = style("! WARN", YELLOW, BOLD)
    else:
        badge = style("✗ FAIL", RED, BOLD)
    suffix = f"  {detail}" if detail else ""
    return f"{badge:<18} {label}{suffix}"


def table(headers: Sequence[str], rows: Iterable[Sequence[object]], right_align: set[int] | None = None) -> str:
    right_align = right_align or set()
    data = [[str(x) for x in headers]]
    data.extend([[str(x) for x in row] for row in rows])
    widths = [max(len(r[i]) for r in data) for i in range(len(headers))]

    def fmt(row: Sequence[str], header: bool = False) -> str:
        cells = []
        for i, value in enumerate(row):
            if i in right_align and not header:
                cells.append(value.rjust(widths[i]))
            else:
                cells.append(value.ljust(widths[i]))
        return "│ " + " │ ".join(cells) + " │"

    top = "┌─" + "─┬─".join("─" * w for w in widths) + "─┐"
    mid = "├─" + "─┼─".join("─" * w for w in widths) + "─┤"
    bot = "└─" + "─┴─".join("─" * w for w in widths) + "─┘"
    body = [top, style(fmt(data[0], header=True), BOLD), mid]
    body.extend(fmt(r) for r in data[1:])
    body.append(bot)
    return "\n".join(body)
