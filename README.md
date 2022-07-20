# Social-SSL
This is the official implementation of our paper:

Social-SSL: Self-Supervised Cross-Sequence Representation Learning Based on Transformers for Multi-Agent Trajectory Prediction\

# Environment
- Tested OS: Linux Ubuntu 18.04

- Create a virtual environment (The packages we need are included in the initial conda environment)\
`$conda create -n SocialSSL python=3.6`

- Install a gpu-version for pytorch (PyTorch==1.7.0): https://pytorch.org/

- Install tqdm\
`$pip install tqdm`

# Preprocessing & Datasets
The preprocessed version of ETH/UCY dataset can be download here

# Training pretext
`$python train_pretext.py`

# Finetune on downstream (Trajectory Prediction)
`$python finetune.py`

# Evaluation
`$python eval.py`
