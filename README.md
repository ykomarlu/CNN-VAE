# CNN-VAE (original repo: https://github.com/LukeDitria/CNN-VAE)
### By: Yushus Komarlu, Kendall Slade, and Zion Ross

## Environment Setup
1) Connect to Deep Learning server
2) Git clone repo
3) python -m venv name_of_venv

## Installation
pip install tqdm
pip install torch
pip install torchvision
pip install thop

## Train Command
python3 train_vae.py -mn name_of_run --dataset_root dataset_root/

## Test Command
python3 main.py --model_path path_of_model --image_path dataset_root/--output_path outputs/