# -*- coding: utf-8 -*-
# @Time : 2024/12/13 10:52 上午
# @Author : senwang
# @Email : tengbaoqiang.tbq@alibaba-inc.com
# @File : train_clip_predict.py
# @Project : intime_intelligent_election
# @Software: PyCharm
'''
# 通过pip安装
pip install cn_clip

# 或者从源代码安装
cd Chinese-CLIP
pip install -e .
参考：https://github.com/OFA-Sys/Chinese-CLIP
'''
import time
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com' # 后面发现在dsw还是不能用，后面把.cache下的打包过去,还是报错requests.exceptions.ConnectTimeout: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /jinaai/jina-clip-v2/resolve/main/jinaai/jina-clip-implementation--configuration_clip.py, 升级下transformers解决4.25.1 -> 4.46.3无果，最终多跑几次clip_test.py就可以下了，有一个配置文件会下到~/.cache/huggingface/transformers/jinaai/jina-clip-implementation 
# os.chdir(os.path.dirname(__file__))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# import timm
# print([m for m in timm.list_models() if 'dino' in m])
from PIL import Image
import numpy as np
# from cn_clip.clip import load_from_name, tokenize
import torch.optim as optim
import sys
curdir = os.path.dirname(__file__)
sys.path.append(os.path.join(curdir, '../../'))
from dataset_imageretrival import ImageTextDataset
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
from huggingface_hub import snapshot_download
import torch.nn.functional as F
import tqdm
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from transformers.models.clip.modeling_clip import clip_loss
from accelerate import Accelerator
# device =  torch.device("cpu") # debug 用cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JinaClipTrainer:
    def __init__(self, learning_rate_txt=5e-5, learning_rate_img=5e-6, learning_rate_scale=1e-4, batch_size=5, gradient_accumulation_steps=8):
        # 初始化 Accelerator# 会自动检测可用的设备,  也可以通果命令行配置accelerate config
        self.accelerator = Accelerator(
            mixed_precision= 'no', # 'bf16',  # senwang 默认fp16 设置了 mixed_precision='bf16'，这会导致额外的内存开销，报outofmemory
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        # Initialize CLIP model
        model_name = "jinaai/jina-clip-v2"
        ### 指定版本
        commit_hash = "ca8657a"  # 示例提交哈希值
        access_token = 'your access token'
        self.clip_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, revision=commit_hash, use_auth_token=access_token)  # .to(device)  # model_name  
        '''
        ### 不指定版本，最新版本
        clip_model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)  # model_name  dtype=bfloat16
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)  # from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name) # from_pretrained(model_name)
        '''
        # clip_model = clip_model.float()  # senwang add. convert to float32, 加了这个训练，直接显存就不够了
        # 加载预处理器（[短边resize为512, centercrop512, totensor, Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])]）
        # Freeze CLIP model parameters, 如果没什么特别需要freeze的参数，其实以下两件=句也可不用
        for param in self.clip_model.parameters():
            param.requires_grad = True
        
        self.image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True, revision=commit_hash, use_auth_token=access_token)  # from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=commit_hash, use_auth_token=access_token) # from_pretrained(model_name)
        
        # self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, clip_model.parameters()), lr=learning_rate, betas=(0.9,0.98)) # for CombinedModelClipOnly
        self.optimizer = optim.AdamW([
                {'params': self.clip_model.text_model.parameters(), 'lr': learning_rate_txt},
                {'params': self.clip_model.vision_model.parameters(), 'lr': learning_rate_img},
                {'params': self.clip_model.logit_scale, 'lr': learning_rate_scale}
                ],
                betas=(0.9,0.98)) # for CombinedModelClipOnly
        print("Optimizer param groups:")
        opt_params_id = set(id(p) for p in self.optimizer.param_groups[0]['params'])
        for name, param in self.clip_model.named_parameters():
            if 'logit_scale' in name:
                print(f"###########: {name}, 形状: {param.shape}")
            if param.requires_grad and id(param) in opt_params_id:
                print(f"需要优化的参数名: {name}, 形状: {param.shape}")
                
    # 定义图像和文本的预处理函数
    def preprocess_image(self, image):
        return self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)

    def preprocess_text(self, text):
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids

    def evaluate_model(self, dataloader):
        self.clip_model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, dataiter in enumerate(dataloader):
                # dataloader 经过 accelerator.prepare() 后，数据会自动被放到正确的设备上
                images, texts = dataiter
                # images = images.to(device)
                texts = self.preprocess_text(texts).to(self.accelerator.device) # .to(device)
                output = self.clip_model(input_ids=texts, pixel_values=images, return_loss=True, return_dict=True)
                loss = output.loss
                total_loss += loss.item()
                
                # 获取预测结果
                logits_per_text = output.logits_per_text
                _, predicted = torch.max(logits_per_text, dim=1)
                correct += (predicted == torch.arange(len(texts), device=self.accelerator.device)).sum().item()
                total += len(texts)
        accuracy = correct / total
        average_loss = total_loss / len(dataloader)

        return average_loss, accuracy
    
    def train(self, train_dataloader, test_dataloader, num_epochs, save_dir):
        # 定义 warmup 调度器
        warmup_epochs = 5
        scheduler_warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        scheduler_cosine = CosineAnnealingLR(self.optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-7)  # optimer里面设置的学习率为初始学习率，eta_min为最终学习率，T_max为余弦周期的最大步数，
        
        # 组合调度器
        scheduler = SequentialLR(self.optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

        # Prepare the model, optimizer, and data loader for mixed precision， 会把数据都统一放在self.accelerator.device上
        self.clip_model, self.optimizer, dataloader, test_dataloader = self.accelerator.prepare(self.clip_model, self.optimizer, train_dataloader, test_dataloader) 

        global_best_testloss = float('inf')
        for epoch in range(0, num_epochs):
            tic = time.time()
            self.clip_model.train()
            running_loss = 0.0
            total_iter = len(dataloader)
            for i, dataiter in enumerate(dataloader):
                with self.accelerator.accumulate(self.clip_model): # 在这下面，会自动处理梯度累积，不用手动判断梯度累积
                    images, texts = dataiter
                    # images = images.to(device)
                    texts = self.preprocess_text(texts).to(self.accelerator.device) # .to(device)
                    output = self.clip_model(input_ids=texts, pixel_values=images, return_loss=True, return_dict=True)
                    loss = output.loss
                    
                    # 反向传播
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:  # 标志表示是否到达了需要同步梯度的时候（即完成了累积步数）
                        self.accelerator.clip_grad_norm_(self.clip_model.parameters(), 1.0)
                        self.optimizer.step() 
                        self.optimizer.zero_grad()

                    running_loss += loss.item()
                    print(f'iter {i}/{total_iter}', 'loss:', loss.item())

            scheduler.step()
            # 打印当前学习率（每个epoch）
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']
                print(
                    f'Epoch {epoch + 1}/{num_epochs},  Learning Rate: {current_lr:.9f}')
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            # 在每个epoch结束后评估测试集
            test_loss, test_acc = self.evaluate_model(test_dataloader)
            print('one epoch take time:', time.time() - tic)
            # # 保存模型
            save_path = os.path.join(save_dir, f'epoch_{epoch + 1}_trainloss_{epoch_loss}_testloss_{test_loss}_testacc_{test_acc}.pth')
            # torch.save(self.clip_model.state_dict(), save_path)
            self.accelerator.save(self.accelerator.unwrap_model(self.clip_model).state_dict(),  save_path) # 只保存模型权重
            print(f"Model saved to {save_path}")
            if test_loss < global_best_testloss:
                global_best_testloss = test_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.accelerator.unwrap_model(self.clip_model).state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': epoch_loss,
                }
                save_path = os.path.join(save_dir, f'best_loss_{test_loss}.pth')
                # torch.save(checkpoint, save_path)
                self.accelerator.save(checkpoint, save_path)
                print(f"Best model saved to {save_path}")
            
        print("Training finished!")   

    def inference(self):
        test_dataset = ImageTextDataset(transform_image=self.preprocess_image, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)
        
        test_loss, test_acc = self.evaluate_model(test_dataloader)
        print(f"origin Test Loss: {test_loss}")
        
        # 加载最佳模型
        # 加载 checkpoint
        checkpoint = torch.load("weights/ClipModelImageRetrieval/last.pth")
        # 恢复模型状态
        self.clip_model.load_state_dict(checkpoint['model_state_dict'])
        # # 恢复优化器状态
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # # 恢复调度器状态
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # clip_model.load_state_dict(torch.load("weights/ClipModelImageRetrieval/epoch_13_trainloss_6.895833333333333_testloss_11.541666666666666_testacc_0.2727272727272727.pth"))
        # 在测试集上评估
        test_loss, test_acc = self.evaluate_model(test_dataloader)
        print(f"Test Loss: {test_loss}")


def main():
    # 配置参数
    batch_size = 5 # 8 # 32 # 4
    gradient_accumulation_steps = 24
    num_epochs = 100
    # learning_rate_base = 5e-5  # 
    learning_rate_txt = 5e-5  # 
    learning_rate_img = 5e-6  # 视觉特征的提取比文本特征提取更复杂，需要更谨慎的调整
    learning_rate_scale = 1e-4  # logit_scale的学习率
    save_dir = os.path.join(curdir, 'weights/ClipModelImageRetrieval')  # 'weights/weights_combinemodel_jina_clip_v2_mj_only'  # 'weights/weights_combinemodel_jinjiav3'  # './weights_vit224_white_fillsquare_p_0.0001'  # weights_vit224_white_fillsquare_p:白底图不改变人物的长宽比，采用大于3的投票概率为分数   './weights_vit224_white_fillsquare'
    os.makedirs(save_dir, exist_ok=True)
    # 创建训练器
    trainer = JinaClipTrainer(
        learning_rate_txt=learning_rate_txt,
        learning_rate_img=learning_rate_img,
        learning_rate_scale=learning_rate_scale,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    train_dataset = ImageTextDataset(transform_image=trainer.preprocess_image, is_train=True)
    # dataset.img_names_test = dataset.img_names_test[2:4] # senwang just for test
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataset = ImageTextDataset(transform_image=trainer.preprocess_image, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) #   由于测试集同一个spuid等数据相邻，所以用了shffle=True
    trainer.train(train_dataloader, test_dataloader, num_epochs, save_dir)
if __name__ == '__main__':
    main()
    ### 最终还是用了train_clip_accelerator.py
