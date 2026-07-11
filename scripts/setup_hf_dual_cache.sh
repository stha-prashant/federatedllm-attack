#!/bin/bash
# Link existing home Hugging Face hub models into shared hub cache so HF can use both:
# - existing models: ~/.cache/huggingface/hub (via symlinks)
# - new downloads:   /shared/rc/llm-degredation/ps9044/huggingface/hub
#
# Datasets cache (aarch64): /scratch/ps9044/huggingface/datasets
# Optional migration from home (~/.cache/huggingface/datasets, ~161G):
#   rsync -aP ~/.cache/huggingface/datasets/ /scratch/ps9044/huggingface/datasets/
set -euo pipefail

SHARED_HF="/shared/rc/llm-degredation/ps9044/huggingface"
HOME_HF="${HOME}/.cache/huggingface"
SCRATCH_DATASETS="/scratch/ps9044/huggingface/datasets"

mkdir -p "${SHARED_HF}/hub"
mkdir -p "${SCRATCH_DATASETS}"

linked=0
if [ -d "${HOME_HF}/hub" ]; then
  for entry in "${HOME_HF}/hub"/*; do
    [ -e "$entry" ] || continue
    name="$(basename "$entry")"
    target="${SHARED_HF}/hub/${name}"
    if [ ! -e "$target" ]; then
      ln -s "$entry" "$target"
      echo "linked hub: ${name}"
      linked=$((linked + 1))
    fi
  done
fi

echo "Done. ${linked} hub entries linked into ${SHARED_HF}/hub"
echo "HF_DATASETS_CACHE=${SCRATCH_DATASETS} (datasets on scratch; migrate from ${HOME_HF}/datasets if needed)"
echo "HUGGINGFACE_HUB_CACHE=${SHARED_HF}/hub (new models download here)"
echo "Optional dataset migration: rsync -aP ${HOME_HF}/datasets/ ${SCRATCH_DATASETS}/"
