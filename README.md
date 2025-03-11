# finetune-jina-clip-v2

Thanks to https://huggingface.co/jinaai/jina-clip-v2， I can get a good pretrained model suporting multi-language.
Beacause I want to further finetune the model  using our own domain-specific data. But there is no public training code for jina-clip-v2. So I
write this project to training it.

## You organize your own data in 
dataset_own.py
## then starting training:
python train_clip.py
## or  starting training using accerlerator
python train_clip_accelerator.py

## training starting like
![EB18249A-09CF-462D-9470-A53C27B3E49C](https://github.com/user-attachments/assets/dedf2c06-82a5-42d9-9914-3266efaddcd8)

## training result like:
![image](https://github.com/user-attachments/assets/90fc6dc8-bc62-46cf-b9bb-94ae9c6ff5fc)

