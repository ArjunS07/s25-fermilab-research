#!/usr/bin/env python3
"""Print space-separated architecture flags from a training summary.json.

Tolerates truncated files (e.g. pod killed mid-write) and NaN/Infinity
literals that train.py's json.dump may produce.

Usage: python3 -m analysis.read_train_summary /path/to/summary.json
Output: use_refs use_scalars use_adaln use_attention prior_dist n_layers n_hidden use_residual
"""
import json
import re
import sys


def _repair_and_parse(raw: str) -> dict:
    """Best-effort parse of possibly-truncated JSON with NaN literals."""
    raw = re.sub(r'\bNaN\b', 'null', raw)
    raw = re.sub(r'\b-?Infinity\b', 'null', raw)

    # Try parsing as-is first.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Truncated: close any unclosed brackets/braces.
    # Strip trailing incomplete token (partial number, string, etc.).
    trimmed = raw.rstrip()
    # Remove trailing comma if present.
    trimmed = trimmed.rstrip(',')

    # Count unclosed delimiters.
    stack = []
    in_string = False
    escape = False
    for ch in trimmed:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append('}' if ch == '{' else ']')
        elif ch in ('}', ']'):
            if stack:
                stack.pop()

    # Append closing delimiters in reverse order.
    trimmed += ''.join(reversed(stack))

    return json.loads(trimmed)


def main():
    path = sys.argv[1]
    with open(path) as f:
        s = _repair_and_parse(f.read())
    m = (s.get("full_config") or {}).get("model") or {}
    t = (s.get("full_config") or {}).get("training") or {}
    print(
        str(m.get("use_reference_vectors", False)).lower(),
        str(m.get("use_node_scalars", False)).lower(),
        str(m.get("use_adaln", False)).lower(),
        str(m.get("use_attention", False)).lower(),
        t.get("prior_dist", "isotropic_com"),
        m.get("n_layers", 3),
        m.get("n_hidden", 128),
        str(m.get("use_residual", False)).lower(),
    )


if __name__ == "__main__":
    main()
