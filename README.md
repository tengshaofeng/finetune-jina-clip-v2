# finetune-jina-clip-v2

Thanks to https://huggingface.co/jinaai/jina-clip-v2， I can get a good pretrained model suporting multi-language.
Beacause I want to further finetune the model  using our own domain-specific data. But there is no public training code for jina-clip-v2. So I
write this project to training it.

## You organize your own data in 
dataset_own.py
## then starting training:
python train_clip.py
## or  starting training using accerlerator
python train_clip_accerlerator.py
