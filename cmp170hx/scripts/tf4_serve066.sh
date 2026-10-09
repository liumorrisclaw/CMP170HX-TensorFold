#!/usr/bin/env bash
# tf4_serve.sh <profile> — TensorFold GLM-5.3-Flash 4-rank, single-variable A/B launcher.
# Profiles (each = one restart, one changed variable vs A0):
#   nodraft  --no-drafts                (A0 baseline, reproduces 2026-10-05 tf4 run)
#   mtp      --drafter none             (B1: MTP-chain drafting from the checkpoint's own head)
#   dflash   --drafter <prod DFlash2>   (B2: DFlash2 blocks, auto per-round arm choice)
#   dflash3  --drafter <prod> --mtp-drafts 3  (B3: fixed depth 3)
#   mtp3     --drafter none --mtp-drafts 3
set -euo pipefail
PROFILE="${1:?usage: tf4_serve.sh nodraft|mtp|dflash|dflash3|mtp3}"
CKPT="$HOME/tensorfold-poc/weights/glm53-tr3-4bpw"
DFLASH="${HOME}/glm53-recipe/models/GLM-5.3-Flash-DFlash2"
TF="$HOME/tensorfold-poc/TensorFold066"
PY="$HOME/tensorfold-poc/venv/bin/python"
LOGD="$HOME/tensorfold-poc/logs"
PORT=8899
mkdir -p "$LOGD"

case "$PROFILE" in
  nodraft) DRAFT_ARGS="--no-drafts";;
  mtp)     DRAFT_ARGS="--drafter none";;
  dflash)  DRAFT_ARGS="--drafter $DFLASH";;
  dflash3) DRAFT_ARGS="--drafter $DFLASH --mtp-drafts 3";;
  mtp3)    DRAFT_ARGS="--drafter none --mtp-drafts 3";;
  *) echo "unknown profile: $PROFILE" >&2; exit 2;;
esac

if [ -n "${TF_DRAFTER_BITS:-}" ]; then
  DRAFT_ARGS="$DRAFT_ARGS --drafter-bits $TF_DRAFTER_BITS"
fi

pkill -f "tensorfold serve" 2>/dev/null || true
sleep 3

export PYTHONPATH="${HOME}/glm53-recipe/venv/lib/python3.12/site-packages:$TF/src"
export TENSORFOLD_MIN_CAPABILITY=8.0
export TF_NCCL_LIB="${HOME}/glm53-recipe/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
export TORCH_EXTENSIONS_DIR="$HOME/tensorfold-poc/tf_ext066"

# non-zero ranks first (10-05 known-good order)
mkdir -p "$HOME/tensorfold-poc/profiles"
for R in 3 2 1 0; do
  if [ -n "${TF_NSYS_DELAY:-}" ] && [ "$R" = "0" ]; then
    CUDA_VISIBLE_DEVICES=$R nohup nsys profile -o "$HOME/tensorfold-poc/profiles/tf4_${PROFILE}_s0" \
      --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
      --delay="${TF_NSYS_DELAY}" --duration="${TF_NSYS_DURATION:-420}" --force-overwrite true \
      "$PY" -m tensorfold serve "$CKPT" \
      --tp 4 --rank $R --master 127.0.0.1 $DRAFT_ARGS \
      --context 32768 --port $PORT \
      > "$LOGD/tf4_${PROFILE}_r${R}.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$R nohup "$PY" -m tensorfold serve "$CKPT" \
      --tp 4 --rank $R --master 127.0.0.1 $DRAFT_ARGS \
      --context 32768 --port $PORT \
      > "$LOGD/tf4_${PROFILE}_r${R}.log" 2>&1 &
  fi
  sleep 1
done
echo "[tf4_serve] profile=$PROFILE launched (ranks 3,2,1 then 0), waiting for rank0 :$PORT"

for i in $(seq 1 150); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[tf4_serve] READY after ~$((i*12))s"
    curl -s "http://127.0.0.1:$PORT/health"; echo
    tail -3 "$LOGD/tf4_${PROFILE}_r0.log"
    exit 0
  fi
  # abort early if rank0 died
  if ! pgrep -f "tensorfold serve" >/dev/null; then
    echo "[tf4_serve] ranks died, rank0 log tail:" >&2
    tail -20 "$LOGD/tf4_${PROFILE}_r0.log" >&2
    exit 1
  fi
  sleep 12
done
echo "[tf4_serve] TIMEOUT waiting for health; rank0 log tail:" >&2
tail -20 "$LOGD/tf4_${PROFILE}_r0.log" >&2
exit 1
