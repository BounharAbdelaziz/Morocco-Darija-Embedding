#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 train.py