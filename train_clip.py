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
from dataset_own import ImageTextDataset
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
from huggingface_hub import snapshot_download
import torch.nn.functional as F
import tqdm
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
device =  torch.device("cpu") # debug 用cpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = 'weights/ClipModelImageRetrieval'  # 'weights/weights_combinemodel_jina_clip_v2_mj_only'  # 'weights/weights_combinemodel_jinjiav3'  # './weights_vit224_white_fillsquare_p_0.0001'  # weights_vit224_white_fillsquare_p:白底图不改变人物的长宽比，采用大于3的投票概率为分数   './weights_vit224_white_fillsquare'
os.makedirs(save_dir, exist_ok=True)
# 定义超参数
batch_size = 5 # 8 # 32 # 4
num_epochs = 100
learning_rate = 5e-5  # 0.0001  # 0.001
'''
### 指定版本
commit_hash = "ca8657a"  # 示例提交哈希值
access_token = 'your access_token'
clip_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, revision=commit_hash, use_auth_token=access_token).to(device)  # model_name
# 加载预处理器（[短边resize为512, centercrop512, totensor, Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])]）
image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True, revision=commit_hash, use_auth_token=access_token)  # from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=commit_hash, use_auth_token=access_token) # from_pretrained(model_name)
'''
### 不指定版本，最新版本
clip_model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)  # model_name  dtype=bfloat16
image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)  # from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name) # from_pretrained(model_name)

# clip_model = clip_model.float()  # senwang add. convert to float32, 加了这个训练，直接显存就不够了


# Freeze CLIP model parameters
for param in clip_model.parameters():
    param.requires_grad = True


# 定义图像和文本的预处理函数
def preprocess_image(image):
    return image_processor(image, return_tensors="pt").pixel_values.squeeze(0)

def preprocess_text(text):
    return tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, dataiter in enumerate(dataloader):
            images, texts = dataiter
            images = images.to(device)
            texts = preprocess_text(texts).to(device)
            output = model(input_ids=texts, pixel_values=images, return_loss=True, return_dict=True)
            loss = output.loss
            total_loss += loss.item()
            
            # 获取预测结果
            logits_per_text = output.logits_per_text
            _, predicted = torch.max(logits_per_text, dim=1)
            correct += (predicted == torch.arange(len(texts), device=device)).sum().item()
            total += len(texts)
    accuracy = correct / total
    average_loss = total_loss / len(dataloader)

    return average_loss, accuracy

def inference():
    test_dataset = ImageTextDataset(transform_image=preprocess_image, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 加载原始模型
    clip_model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)  # model_name
    test_loss, test_acc = evaluate_model(clip_model, test_dataloader, device)
    print(f"origin Test Loss: {test_loss}")
    
    # 加载最佳模型
    # 加载 checkpoint
    checkpoint = torch.load("weights/ClipModelImageRetrieval/last.pth")
    # 恢复模型状态
    clip_model.load_state_dict(checkpoint['model_state_dict'])
    # # 恢复优化器状态
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # # 恢复调度器状态
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # clip_model.load_state_dict(torch.load("weights/ClipModelImageRetrieval/epoch_13_trainloss_6.895833333333333_testloss_11.541666666666666_testacc_0.2727272727272727.pth"))
    # 在测试集上评估
    test_loss, test_acc = evaluate_model(clip_model, test_dataloader, device)
    print(f"Test Loss: {test_loss}")


def train():
    crop_size = (512, 512)  # (224, 224) #
    # 创建数据集
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
    texts = ["text description 1", "text description 2", ...]

    dataset = ImageTextDataset(transform_image=preprocess_image, is_train=True, cropsize=crop_size)
    # dataset.img_names_test = dataset.img_names_test[2:4] # senwang just for test
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataset = ImageTextDataset(transform_image=preprocess_image, is_train=False, cropsize=crop_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) #   由于测试集同一个spuid等数据相邻，所以用了shffle=True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, clip_model.parameters()), lr=learning_rate, betas=(0.9,0.98)) # for CombinedModelClipOnly
    # optimizer = optim.AdamW([
    #     {'params': clip_model.text_model.parameters()},
    #     {'params': clip_model.vision_model.parameters()},
    #     {'params': clip_model.logit_scale}], lr=learning_rate) # for CombinedModelClipOnly
    print("Optimizer param groups:")
    opt_params_id = set(id(p) for p in optimizer.param_groups[0]['params'])
    for name, param in clip_model.named_parameters():
        if 'logit_scale' in name:
            print(f"###########: {name}, 形状: {param.shape}")
        if param.requires_grad and id(param) in opt_params_id:
            print(f"需要优化的参数名: {name}, 形状: {param.shape}")
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)  # optimer里面设置的学习率为初始学习率，eta_min为最终学习率，T_max为余弦周期的最大步数，

    # 定义 warmup 调度器
    warmup_epochs = 5
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-7)
    
    # 组合调度器
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    
    '''##### just debug
    static_images = torch.randn(2, 3, 512, 512).to(device)
    static_texts = ["test text"] * 2
    static_text_ids = preprocess_text(static_texts).to(device)
    image_features_initial = clip_model.get_image_features(pixel_values=static_images)
    text_features_initial = clip_model.get_text_features(input_ids=static_text_ids)
    '''
    global_best_testloss = 1e10
    for epoch in range(0, num_epochs):
        tic = time.time()
        clip_model.train()
        running_loss = 0.0
        total_iter = len(dataloader)
        for i, dataiter in enumerate(dataloader):
            images, texts = dataiter
            images = images.to(device)
            texts = preprocess_text(texts).to(device)
            
            output = clip_model(input_ids=texts, pixel_values=images, return_loss=True, return_dict=True)
            loss = output.loss
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            print(f'iter {i}/{total_iter}', 'loss:', loss.item())

            '''#### just debug
            print("Logit scale梯度:", clip_model.logit_scale.grad)
            image_features = clip_model.get_image_features(pixel_values=static_images)
            text_features = clip_model.get_text_features(input_ids=static_text_ids)
            # 观察特征是否变化
            print("图像特征差异:", torch.norm(image_features - image_features_initial))
            print("文本特征差异:", torch.norm(text_features - text_features_initial))
            '''
        scheduler.step()
        # 打印当前学习率（每个epoch）
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            print(
                f'Epoch {epoch + 1}/{num_epochs},  Learning Rate: {current_lr:.9f}')
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        # 在每个epoch结束后评估测试集
        test_loss, test_acc = evaluate_model(clip_model, test_dataloader, device)
        print('one epoch take time:', time.time() - tic)
        # # 保存模型
        save_path = os.path.join(save_dir, f'epoch_{epoch + 1}_trainloss_{epoch_loss}_testloss_{test_loss}_testacc_{test_acc}.pth')
        torch.save(clip_model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        if test_loss < global_best_testloss:
            global_best_testloss = test_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': clip_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, f'best_loss_{test_loss}.pth'))
            print(f"Best model saved to {os.path.join(save_dir, f'best_loss_{test_loss}.pth')}")
        
    print("Training finished!")

if __name__ == '__main__':
    # inference()
    train()
