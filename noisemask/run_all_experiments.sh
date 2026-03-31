#!/bin/bash
# ==========================================
# Noise Mask 防御实验 - 一键运行脚本
# ==========================================

TARGET_DIR="/Users/xiaotong/desktop/new/模型攻防/noisemask"
cd "$TARGET_DIR"

echo "1. 下载 SECMI 和 PIA"

# 下载 SecMI（如果不存在）
if [ ! -d "SecMI" ]; then
    git clone https://github.com/jinhaoduan/SecMI.git
fi

# 下载 PIA（如果不存在）
if [ ! -d "PIA" ]; then
    git clone https://github.com/kong13661/PIA.git
fi

echo "2. 运行防御实验"

# 实验1: SECMI 无防御
python run_defense_experiment.py --attack secmi --noise_scale 0.0

# 实验2-4: SECMI + 不同强度噪声
python run_defense_experiment.py --attack secmi --noise_scale 0.1
python run_defense_experiment.py --attack secmi --noise_scale 0.3
python run_defense_experiment.py --attack secmi --noise_scale 0.5

# 实验5: PIA 无防御
python run_defense_experiment.py --attack pia --noise_scale 0.0

# 实验6-7: PIA + 噪声
python run_defense_experiment.py --attack pia --noise_scale 0.3
python run_defense_experiment.py --attack pia --noise_scale 0.3 --noise_type laplacian

echo "完成!"
