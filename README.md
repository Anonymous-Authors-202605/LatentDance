# LatentDance: Towards Realistic and Dynamic Character Animation via Identity-Aware Motion Representation 
Anonymous Github Repository: https://github.com/Anonymous-Authors-202605/LatentDance

## Quickstart
### Environment Setup
```bash
# 1. Clone the repository and create a conda environment:
git clone https://github.com/Anonymous-Authors-202605/LatentDance.git
cd LatentDance

conda create -n latentdance python=3.10
conda activate latentdance

pip install -e .

pip install git+https://github.com/huggingface/diffusers

# packages required for training 
pip install deepspeed 

# packages required for DWPose 
pip install opencv-python controlnet_aux matplotlib onnxruntime-gpu av

# Multi-gpu inference 
pip install xfuser==0.4.2 --progress-bar off -i https://mirrors.aliyun.com/pypi/simple/

### flash attention 
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash_attn --no-build-isolation
```

### Download Checkpoints
```bash 
pip install modelscope 

modelscope download AnonymouAuthors/LatentDance --local_dir ./checkpoints/LatentDance
```

### Inference 
#### Demo data 
Please download the demo data from [Github release](https://github.com/Anonymous-Authors-202605/LatentDance/releases/download/v0.0/evaldata.zip), unzip it and place it under `data/evaldata`.
The directory structure should look like:
```
data/evaldata/
├── input_image
│   └── 00000.png
├── pose3_keypoints
│   └── 00000_keypoints.json
└── pllava_caption
    └── caption.csv
```

#### Inference command
```bash
bash scripts/v8_general/inference.sh
```

