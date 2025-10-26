#!/usr/bin/env bash
set -e
python -m gndi.run_prune --config configs/cv_resnet18_cifar100.yaml --methods gndi snip grasp synflow magnitude hrank
