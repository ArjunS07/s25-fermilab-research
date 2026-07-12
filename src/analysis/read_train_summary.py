#!/usr/bin/env python3
"""Print space-separated architecture flags from a training summary.json.

Tolerates NaN/Infinity literals that train.py's json.dump may produce.

Usage: python3 -m analysis.read_train_summary /path/to/summary.json
Output: use_refs use_scalars use_adaln use_attention prior_dist n_layers n_hidden use_residual
"""
import json
import sys


def main():
    path = sys.argv[1]
    with open(path) as f:
        s = json.loads(f.read(), parse_constant=lambda x: None)
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
