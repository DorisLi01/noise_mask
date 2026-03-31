# 基线：无攻击 无防御
python main_final.py --dataset=cifar10 --diffusion_model=ddpm --use_attack=False --use_defense=False
echo "--dataset=cifar10 --diffusion_model=ddpm --use_attack=False --use_defense=False"
# PIA 攻击 无防御
python main_final.py --dataset=cifar10 --diffusion_model=ddpm --use_attack=True --attack_type=pia --use_defense=False
echo "--dataset=cifar10 --diffusion_model=ddpm --use_attack=True --attack_type=pia --use_defense=False"
# SecMI 攻击 无防御
python main_final.py --dataset=cifar10 --diffusion_model=ddpm --use_attack=True --attack_type=secmi --use_defense=False
echo "--dataset=cifar10 --diffusion_model=ddpm --use_attack=True --attack_type=secmi --use_defense=False"
# 双攻击 无防御
python main_final.py --dataset=cifar10 --diffusion_model=ddpm --use_attack=True --attack_type=both --use_defense=False
echo "--dataset=cifar10 --diffusion_model=ddpm --use_attack=True --attack_type=both --use_defense=False"
# SecMI 攻击 + 防御
python main_final.py --dataset=cifar10 --diffusion_model=ddpm --use_attack=True --attack_type=secmi --use_defense=True
echo "--dataset=cifar10 --diffusion_model=ddpm --use_attack=True --attack_type=secmi --use_defense=True"
# PIA 攻击 + 防御
python main_final.py --dataset=cifar10 --diffusion_model=ddpm --use_attack=True --attack_type=pia --use_defense=True
echo "--dataset=cifar10 --diffusion_model=ddpm --use_attack=True --attack_type=pia --use_defense=True"
# 双攻击 + 防御
python main_final.py --dataset=cifar10 --diffusion_model=ddpm --use_attack=True --attack_type=both --use_defense=True
echo "--dataset=cifar10 --diffusion_model=ddpm --use_attack=True --attack_type=both --use_defense=True"
