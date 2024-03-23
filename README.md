# stable-diffusion-finetuning

## Setup

Clone this repo

```
git clone https://github.com/jkcg-learning/stable-diffusion-finetuning.git
cd stable-diffusion-finetuning
```

Install Dependencies

```
pip install -r requirements.txt
```

Configure Accelerate

```
accelerate config
```

My accelerate config looks like

```
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```


Tested with

```
tensorboard==2.16.2
tensorboardX==2.6.2.2
xformers==0.0.25
bitsandbytes==0.43.0
transformers==4.35.2
accelerate==0.24.1
compel==2.0.2
diffusers @ git+https://github.com/huggingface/diffusers.git@6bf1ca2c799f3f973251854ea3c379a26f216f36  
typer[all]==0.9.0
rich==12.5.1
```

## 😁 Outputs

![random](output/images/collage_random_prompts.jpg)

![prompt1](output/images/out_grid_prompt1.png)

![prompt2](output/images/out_grid_prompt2.png)
