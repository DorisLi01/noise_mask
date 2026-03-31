#!/bin/bash
# Noise Mask 防御实验

cd "$(dirname "$0")"

echo "========================================"
echo "Noise Mask 防御实验"
echo "========================================"

# 检查目录
echo "检查目录..."
ls -la

echo ""
echo ">>> 运行 Noise Mask 实验"
echo "----------------------------------------"

cd noisemask
python noisemask_pytorch.py --mode sweep
