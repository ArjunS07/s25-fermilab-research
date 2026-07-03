#!/usr/bin/env bash
# local.sh — CPU smoke-test of the training pipeline (data → jet_attr → cache_icp → train).
# Tiny run (100 samples, 2 epochs) to exercise code paths quickly; not a real training run.

set -euo pipefail

# ── Shared setup ──────────────────────────────────────────────────────────────
POD_UID="local-run-pod-$(uuidgen)"
UNIQUE_RUN_ID="$(date +%F_%H-%M-%S)--${POD_UID}"
BASE_OUT="local_out/${UNIQUE_RUN_ID}"
mkdir -p "${BASE_OUT}"
echo "Output root: ${BASE_OUT}"

NUM_PARTICLES=150
JET_TYPES='g q t'
N_TRAIN=100
BATCH=20
EPOCHS=2
N_SAMPLES=100
N_LAYERS=2
INTEGRATION_STEPS=4
N_VIZ=50

# Common train.py flags shared across all runs
COMMON_TRAIN=(
    --output_path "${BASE_OUT}"
    --jet_types ${JET_TYPES}
    --num_particles ${NUM_PARTICLES}
    --n_train_samples ${N_TRAIN}
    --batch_size ${BATCH}
    --target_batch_size ${BATCH}
    --num_epochs ${EPOCHS}
    --n_samples ${N_SAMPLES}
    --n_layers ${N_LAYERS}
    --integration_steps ${INTEGRATION_STEPS}
    --n_viz_samples ${N_VIZ}
)

# ── 1. Data download ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " Step 1/7  data.py"
echo "══════════════════════════════════════════"
python3 data.py \
    --num_particles ${NUM_PARTICLES} \
    --output_path "${BASE_OUT}" \
    --jet_types ${JET_TYPES}

# ── 2. Jet-attribute normalising flow ─────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " Step 2/7  jet_attr_model.py"
echo "══════════════════════════════════════════"
python3 jet_attr_model.py \
    --output_path "${BASE_OUT}" \
    --batch_size 1024 \
    --num_epochs 10

# ── 3. ICP prior cache ────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " Step 3/7  cache_icp.py"
echo "══════════════════════════════════════════"
python3 cache_icp.py \
    --output_path "${BASE_OUT}" \
    --num_particles ${NUM_PARTICLES} \
    --n_samples ${N_TRAIN} \
    --n_workers 2

ICP_CACHE="${BASE_OUT}/icp_cache.pkl"

# ── 4. Baseline: all ablation features ON (defaults) ─────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " Step 4/7  train — full feature set"
echo "   cosine LR ✓  curriculum ✓  time-sampling (power_law) ✓"
echo "══════════════════════════════════════════"
python3 train.py \
    "${COMMON_TRAIN[@]}" \
    --train_space cartesian \
    --time_sampling power_law \
    --curriculum_alpha_start 2.0 \
    --use_cosine_lr \
    --use_curriculum \
    --use_time_sampling
