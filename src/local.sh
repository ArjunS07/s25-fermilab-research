#!/usr/bin/env bash
# local.sh — smoke-test all recent features and ablations
# Each train.py run is tiny (100 samples, 2 epochs) so the full suite
# completes quickly while still exercising every code path.

set -euo pipefail

# ── Shared setup ──────────────────────────────────────────────────────────────
POD_UID="local-run-pod-$(uuidgen)"
UNIQUE_RUN_ID="$(date +%F_%H-%M-%S)--${POD_UID}"
BASE_OUT="local_out/${UNIQUE_RUN_ID}"
mkdir -p "${BASE_OUT}"
echo "Output root: ${BASE_OUT}"

NUM_PARTICLES=30
JET_TYPES=g
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

# ── 5. With ICP cache ─────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " Step 5/7  train — ICP prior cache"
echo "   cosine LR ✓  curriculum ✓  icp_cache ✓"
echo "══════════════════════════════════════════"
python3 train.py \
    "${COMMON_TRAIN[@]}" \
    --train_space cartesian \
    --time_sampling power_law \
    --use_cosine_lr \
    --use_curriculum \
    --use_time_sampling \
    --icp_cache_path "${ICP_CACHE}"

# ── 6. Ablation: all features OFF (vanilla baseline) ─────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " Step 6/7  train — all ablations OFF"
echo "   cosine LR ✗  curriculum ✗  time-sampling ✗ (→ uniform)"
echo "══════════════════════════════════════════"
python3 train.py \
    "${COMMON_TRAIN[@]}" \
    --train_space cartesian \
    --time_sampling power_law \
    --no-use_cosine_lr \
    --no-use_curriculum \
    --no-use_time_sampling

# ── 7. Individual feature ablations ───────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " Step 7/7  individual ablation sweeps"
echo "══════════════════════════════════════════"

echo "  7a  no cosine LR"
python3 train.py \
    "${COMMON_TRAIN[@]}" \
    --time_sampling lognorm \
    --use_curriculum \
    --use_time_sampling \
    --no-use_cosine_lr

echo "  7b  no curriculum"
python3 train.py \
    "${COMMON_TRAIN[@]}" \
    --time_sampling lognorm \
    --use_cosine_lr \
    --use_time_sampling \
    --no-use_curriculum

echo "  7c  no time sampling (forced uniform)"
python3 train.py \
    "${COMMON_TRAIN[@]}" \
    --time_sampling lognorm \
    --use_cosine_lr \
    --use_curriculum \
    --no-use_time_sampling

echo "  7d  polar interpolation + lognorm sampling"
python3 train.py \
    "${COMMON_TRAIN[@]}" \
    --train_space polar \
    --time_sampling lognorm \
    --use_cosine_lr \
    --use_curriculum \
    --use_time_sampling

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "All steps completed.  Results in: ${BASE_OUT}"
