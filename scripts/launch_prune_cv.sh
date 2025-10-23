#!/usr/bin/env bash
set -e
python -m gndi.run_prune --config configs/cv_resnet18_cifar10.yaml --methods gndi snip grasp synflow magnitude hrank
