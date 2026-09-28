#!/usr/bin/env bash
# ==============================================================================
# Run the MaRK ablation suites inside the released NeMo container.
#
#   ./run_ablation.sh --suite loo  --seeds 10
#   ./run_ablation.sh --suite cart --seeds 10 --limit-val-batches 500
#   ./run_ablation.sh --suite profiling --profiling-seeds 15
#   ./run_ablation.sh --suite all --seeds 10
#
# The repo is mounted at /workspace/mark (the image's WORKDIR is /workspace),
# the GPU is exposed with --gpus all, and every extra argument is forwarded
# verbatim to `python -m src.ablation`.
#
# Environment overrides:
#   IMAGE=mark:latest          container image (built from Dockerfile.nemo on first use)
#   EXPERIMENT_DIR=~/Desktop/experiment
#                              companion checkout holding the profiling suite;
#                              mounted read/write when the `profiling` suite is used
#   REPO=<path>                host checkout to mount (default: this script's directory)
#   NO_BUILD=1                 fail instead of building the image when it is missing
# ==============================================================================
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
IMAGE="${IMAGE:-mark:latest}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$HOME/Desktop/experiment}"
NO_BUILD="${NO_BUILD:-0}"
CONTAINER_WORKDIR="/workspace/mark"
CONTAINER_EXPERIMENT="/workspace/experiment"

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker is not available on PATH" >&2
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    if [ "$NO_BUILD" = "1" ]; then
        echo "error: image '$IMAGE' not found and NO_BUILD=1" >&2
        exit 1
    fi
    echo "Image '$IMAGE' not found; building from Dockerfile.nemo (this takes a while)..."
    docker build -f "$REPO/Dockerfile.nemo" -t "$IMAGE" "$REPO"
fi

# Optional GPU check.
if ! docker run --rm --gpus all "$IMAGE" python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    echo "warning: no CUDA device visible to the container; only CPU-safe steps will work" >&2
fi

MOUNTS=(-v "$REPO:$CONTAINER_WORKDIR" -w "$CONTAINER_WORKDIR")
ARGS=("$@")

# Mount the companion repo only when a suite needs it.
needs_experiment=0
for arg in "$@"; do
    case "$arg" in
        profiling|all) needs_experiment=1 ;;
    esac
done
if [ "$needs_experiment" = "1" ] && [ -d "$EXPERIMENT_DIR" ]; then
    MOUNTS+=(-v "$EXPERIMENT_DIR:$CONTAINER_EXPERIMENT")
    # Point the driver at the in-container path unless the caller set one.
    already_set=0
    for arg in "$@"; do
        [ "$arg" = "--experiment-dir" ] && already_set=1
    done
    if [ "$already_set" = "0" ]; then
        ARGS+=(--experiment-dir "$CONTAINER_EXPERIMENT")
    fi
fi

echo "container: $IMAGE"
echo "repo:      $REPO -> $CONTAINER_WORKDIR"
echo "arguments: ${ARGS[*]:-<none>}"

set +e
docker run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${MOUNTS[@]}" \
    "$IMAGE" \
    python -m src.ablation "${ARGS[@]}"
rc=$?
set -e

# The container runs as root, so hand generated artifacts back to the caller.
if [ "$(id -u)" = "0" ]; then
    chown -R "${SUDO_UID:-root}:${SUDO_GID:-root}" \
        "$REPO/data" "$REPO/logs" 2>/dev/null || true
elif command -v sudo >/dev/null 2>&1; then
    sudo -n chown -R "$(id -u):$(id -g)" "$REPO/data" "$REPO/logs" 2>/dev/null || true
fi

exit "$rc"
