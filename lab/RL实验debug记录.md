
## ~~1、需要提前pip install requirements.txt~~ 


## 2、需要走魔搭去下载


## 3、下载路径要换，模型太大直接爆系统盘了
## 魔搭需要另外安装，它的换下载路径有单独参数，不是HF
❗ **ModelScope 不用 HF_HOME，而是用它自己的缓存路径**


```bash
conda activate verl

mkdir -p /root/autodl-tmp/tmp
mkdir -p /root/autodl-tmp/pip_cache
mkdir -p /root/autodl-tmp/hf_home
mkdir -p /root/autodl-tmp/hf_cache
mkdir -p /root/autodl-tmp/hf_datasets
mkdir -p /root/autodl-tmp/ray
mkdir -p /root/autodl-tmp/triton_cache
mkdir -p /root/autodl-tmp/torch_cache
mkdir -p /root/autodl-tmp/models
# 魔搭
mkdir -p /root/autodl-tmp/modelscope_cache  
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache

export TMPDIR=/root/autodl-tmp/tmp
export TEMP=/root/autodl-tmp/tmp
export TMP=/root/autodl-tmp/tmp
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export HF_HOME=/root/autodl-tmp/hf_home
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_datasets
export RAY_TMPDIR=/root/autodl-tmp/ray
export TRITON_CACHE_DIR=/root/autodl-tmp/triton_cache
export TORCH_HOME=/root/autodl-tmp/torch_cache
export VERL_USE_MODELSCOPE=True

export OMP_NUM_THREADS=8

```

conda 缓存更换
```bash
mkdir -p /root/autodl-tmp/conda_pkgs
conda config --add pkgs_dirs /root/autodl-tmp/conda_pkgs
conda config --show pkgs_dirs
```

软链接checkpoint保存

## ~~4、huggingface的版本要切回4.x~~


## ~~5、flash-attn都安装问题~~
逆天啊，感觉99%的问题都是nm的墙的问题


## ~~6、vllm问题~~
 ~~Downloading http://mirrors.aliyun.com/pypi/packages/8e/cb/03dc1299e0456ff3d58a11f63682ef29aaf5b1bd7f21bfe0690d7ce6fc40/vllm-0.8.4-cp38-abi3-manylinux1_x86_64.whl (294.1 MB)~~
 ~~怎么tm装的是这个版本~~

## 7、上述所有库环境问题都可以通过conda activate verl解决（除了爆系统盘的问题）


## 8、vllm还是从hg获取模型
使用`VERL_USE_MODELSCOPE=True`只能让模型初步从魔搭去获取，但是vllm还是从hg去拿，还是会TLE
最佳解决方案就是用魔搭将模型下载到本地，然后用本地路径去跑脚本
```bash

#!/bin/bash

set -x

  

python3 -m verl.trainer.main_ppo \

algorithm.adv_estimator=grpo \

data.train_files=$HOME/data/gsm8k/train.parquet \

data.val_files=$HOME/data/gsm8k/test.parquet \

data.train_batch_size=256 \

data.max_prompt_length=512 \

data.max_response_length=512 \

data.filter_overlong_prompts=True \

data.truncation='error' \

actor_rollout_ref.model.path=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct \

actor_rollout_ref.actor.optim.lr=1e-6 \

actor_rollout_ref.model.use_remove_padding=True \

actor_rollout_ref.actor.ppo_mini_batch_size=64 \

actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \

actor_rollout_ref.actor.use_kl_loss=True \

actor_rollout_ref.actor.kl_loss_coef=0.001 \

actor_rollout_ref.actor.kl_loss_type=low_var_kl \

actor_rollout_ref.actor.entropy_coeff=0 \

actor_rollout_ref.model.enable_gradient_checkpointing=True \

actor_rollout_ref.actor.fsdp_config.param_offload=False \

actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \

actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \

actor_rollout_ref.rollout.tensor_model_parallel_size=1 \

actor_rollout_ref.rollout.name=vllm \

actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \

actor_rollout_ref.rollout.n=5 \

actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \

actor_rollout_ref.ref.fsdp_config.param_offload=True \

algorithm.use_kl_in_reward=False \

trainer.critic_warmup=0 \

trainer.logger='["console","wandb"]' \

trainer.project_name='verl_grpo_example_gsm8k' \

trainer.experiment_name='qwen2p5_0p5b_grpo_try_4gpu' \

trainer.n_gpus_per_node=4 \

trainer.nnodes=1 \

trainer.save_freq=20 \

trainer.test_freq=5 \

trainer.total_epochs=15 \

"$@"
```


## 9、 wandb登陆问题
旧版本的key不适用于新版本

## 10、checkpoint保存问题
大概率还是系统盘空间不足的原因，需要改变torch.save_model的保存位置从/root改到/autodl-tmp中，晕了，服了，操了