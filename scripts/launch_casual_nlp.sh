#!/usr/bin/env bash
set -e
python -m gndi.run_causal_val --config configs/nlp_bert_agnews.yaml --units 300 --method gndi
