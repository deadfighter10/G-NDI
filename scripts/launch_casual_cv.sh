#!/usr/bin/env bash
set -e
python -m gndi.run_causal_val --config configs/cv_resnet18_cifar10.yaml --units 400 --method gndi