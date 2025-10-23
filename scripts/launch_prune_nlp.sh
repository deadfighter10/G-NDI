#!/usr/bin/env bash
set -e
python -m gndi.run_prune --config configs/nlp_bert_agnews.yaml --methods gndi magnitude
