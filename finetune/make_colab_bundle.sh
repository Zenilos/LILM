#!/bin/zsh
# Bundle everything Colab needs: dataset + schema.
# Run this, then upload colab_bundle.tar.gz to the Colab notebook.
set -e
cd "$(dirname "$0")/.."
tar czf colab_bundle.tar.gz \
  data/finetune/train_v2.jsonl \
  schema/tool_schema.json
ls -la colab_bundle.tar.gz
