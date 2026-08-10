#!/usr/bin/env bash
set -euo pipefail

# Exact, one-use cleanup list from the 2026-08-10 queue snapshot.  Deliberately
# avoid selectors/globs so this cannot remove unrelated NRP workloads.
cluster_context="nautilus"
cluster_namespace="cms-ml"
old_pods=(
  as-jet-train-msgnn-200k-g-150-66qvg
  as-jet-train-msgnn-200k-g-150-cfg-bks8h
  as-jet-train-msgnn-200k-gqt-150-cfg-mt6hj
  as-jet-train-msgnn-200k-gqt-150-fvxrh
  as-jet-train-msgnn-200k-gqt-30-cfg-6cmvf
  as-jet-train-msgnn-200k-gqt-30-dd465
  as-jet-train-msgnn-200k-q-150-cfg-njfjw
  as-jet-train-msgnn-200k-q-150-vhpsv
  as-jet-train-msgnn-200k-q-30-4lqh8
  as-jet-train-msgnn-200k-q-30-cfg-m8fx9
  as-jet-train-msgnn-200k-t-150-9n6zx
  as-jet-train-msgnn-200k-t-150-cfg-8b8ks
  as-jet-train-msgnn-200k-t-30-9brgw
  as-jet-train-msgnn-200k-t-30-cfg-lqzpc
)

kubectl --context "$cluster_context" --namespace "$cluster_namespace" \
  delete pod --ignore-not-found --wait=false "${old_pods[@]}"
