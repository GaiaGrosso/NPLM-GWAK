# NPLM-GWAK
apply NPLM method to GWAK outputs


## install Falkon:
-- on Cannon:
# create interactive instance with GPUs
srun --pty -p gpu_test --gpus 1 --mem 8000 -t 0-03:00 /bin/bash
# load cuda (important: cuda 11.8)
module load cuda/11.8.0-fasrc01
# load python
module load python/3.10.9-fasrc01

# install pytorch
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# install Falkon
pip install falkon -f https://falkon.dibris.unige.it/torch-2.0.0_cu118.html

# install additional libraries
pip install matplotlib
pip install h5py
...

