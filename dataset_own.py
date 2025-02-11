from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import json
import os
import numpy as np
import random
from util_transform import *
curdir = os.path.dirname(__file__)
save_dir = os.path.join(curdir, 'util/image_retrival/')
categorys_dict = {0: '上衣', 1: '裙装', 2: '下衣', 3: '箱包', 4: '鞋子', 5: '配饰', 6: '零食', 7: '美妆', 8: '瓶饮', 9: '家具',
                  20: '玩具', 21: '内衣', 22: '数码硬件', 8888: '其他', 88888888: '其他'}


class ImageTextDataset(Dataset):
    def __init__(self, transform_image, is_train=True, cropsize=512, data_mode = '_imageretirval', test_num=10000):
        self.is_train = is_train
        self.imgs_dir = '/home/senwang/dataset/image_retrival2_42w/' # '/home/senwang/dataset/image_retrival2_test2/' # '/home/admin/workspace/tbq/data/image_retrival2/
        if not os.path.exists(os.path.join(self.imgs_dir, f'img_names_train{data_mode}.txt')):  # 'img_names_train.txt'  
            print('收集图片和tag')
            # self.image_paths = glob.glob(imgs_dir + '/*.webp') # 
            # self.image_paths = self.get_img_paths(imgs_dir)
            self.image_paths = np.loadtxt(self.imgs_dir + 'image_list.txt', dtype=str)
            with open(os.path.join(curdir, 'util/sales_predict/total_sp_dict_spuid2tags.json'), 'r') as file:
                self.tag_data = json.load(file)
            f_sale_dict = curdir + '/util/sales_predict/total_sp_dict_spuid2saletargetlife_from2016_mj_only.json'
            with open(f_sale_dict, 'r') as file:  # total_sp_dict_spuid2saletarget
                self.target_data_mj_only = json.load(file)
            # f_sale_dict = curdir + '/util/sales_predict/total_sp_dict_spuid2saletargetlife_from2016_offline_only.json' 
            # with open(f_sale_dict, 'r') as file:  # total_sp_dict_spuid2saletarget
            #     self.target_data_offline_only = json.load(file)  # 吊牌价喵街线上和线下是一致的
            f_spu2anchor = os.path.join(save_dir, 'fea_extract_dict_spuid2img2anchor.json')
            self.spu2anchor = json.loads(open(f_spu2anchor, 'r').read())
            self.image_paths.sort()
            train_image_paths = self.image_paths[:-test_num]
            test_image_paths = self.image_paths[-test_num:]
            self.img_names_train, self.texts_train, self.anchors_train = self.get_scores_accord(train_image_paths)
            self.img_names_test, self.texts_test, self.anchors_test = self.get_scores_accord(test_image_paths)
            self._cleanup_early_data()
            np.savetxt(os.path.join(self.imgs_dir, f'img_names_train{data_mode}.txt'), self.img_names_train, fmt='%s')  # 'img_names_train.txt'
            np.savetxt(os.path.join(self.imgs_dir, f'texts_train{data_mode}.txt'), self.texts_train, fmt='%s')  # 'texts_train.txt'
            np.savetxt(os.path.join(self.imgs_dir, f'anchors_train{data_mode}.txt'), self.anchors_train, fmt='%s')  # 'anchors_train.txt'
            np.savetxt(os.path.join(self.imgs_dir, f'img_names_test{data_mode}.txt'), self.img_names_test, fmt='%s')  #'img_names_test.txt'
            np.savetxt(os.path.join(self.imgs_dir, f'texts_test{data_mode}.txt'), self.texts_test, fmt='%s')
            np.savetxt(os.path.join(self.imgs_dir, f'anchors_test{data_mode}.txt'), self.anchors_test, fmt='%s')
            
        else:
            self.img_names_train = np.loadtxt(os.path.join(self.imgs_dir, f'img_names_train{data_mode}.txt'), dtype=str)
            with open(os.path.join(self.imgs_dir, f'texts_train{data_mode}.txt'), 'r') as file:
                self.texts_train = file.readlines()
            self.anchors_train = np.loadtxt(os.path.join(self.imgs_dir, f'anchors_train{data_mode}.txt'), dtype=int)
    
            self.img_names_test = np.loadtxt(os.path.join(self.imgs_dir, f'img_names_test{data_mode}.txt'), dtype=str)
            with open(os.path.join(self.imgs_dir, f'texts_test{data_mode}.txt'), 'r') as file:
                self.texts_test = file.readlines()
            self.anchors_test = np.loadtxt(os.path.join(self.imgs_dir, f'anchors_test{data_mode}.txt'), dtype=int)

            # debug 用少量的数据如20个
            # self.img_names_train = self.img_names_train[:20]
            # self.texts_train = self.texts_train[:20]
            # self.anchors_train = self.anchors_train[:20]
            # self.img_names_test = self.img_names_test[:20]
            # self.texts_test = self.texts_test[:20]
            # self.anchors_test = self.anchors_test[:20]
            
        
        self.transform_image = transform_image
        self.transform_crop = Compose([
            ColorJitterImgOnlyGivenBox(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1),
            # RandomScale((0.6, 0.75, 1.0, 1.0, 1.25, 1.5, 1.75, 2.0)), # tbq modify from (0.75, 1.0, 1.25, 1.5, 1.75, 2.0) to (0.6, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
            RandomCropImgOnlyGivenBox(),
            HorizontalFlipImgOnlyGivenBox()
        ])
        self.transform_crop_test = Compose([
            RandomCropImgOnlyGivenBox(is_train=False),
        ])

    def get_img_paths(self, imgs_dir, extension = '.webp'):
        files = []
        cnt = 0
        with os.scandir(imgs_dir) as entries:
            for entry in entries:
                # if entry.is_file() and entry.name.endswith(extension):
                files.append(entry.path)
                cnt += 1
                print(cnt)
        print(f"Found {len(files)} files.")
        return files
    

    def _cleanup_early_data(self):
        """清理前期加载的冗余数据"""
        del self.tag_data
        del self.target_data_mj_only
        del self.spu2anchor
        del self.image_paths  # 注意：应在拆分train/test后清理
        
        # 强制垃圾回收（需import gc）
        import gc
        gc.collect()

        
    def __len__(self):
        if self.is_train:
            return len(self.img_names_train)
        else:
            return len(self.img_names_test)

    def __getitem__(self, idx):
        def getitem(idx):
            if self.is_train:
                image = Image.open(os.path.join(self.imgs_dir, self.img_names_train[idx])).convert('RGB')
                text = self.texts_train[idx]
                ### 随机变换后面不重要词条的位置
                text_list = text.split(',')
                text_pre = ','.join(text_list[:4])
                text_suffix = text_list[4:].copy()
                random.shuffle(text_suffix)
                text_suffix = ','.join(text_suffix)
                text = text_pre + ',' + text_suffix
                box = self.anchors_train[idx]
            else:
                image = Image.open(os.path.join(self.imgs_dir,self.img_names_test[idx])).convert('RGB')
                text = self.texts_test[idx]
                box = self.anchors_test[idx]

            if self.is_train:
                input_data = {'im': image, 'box': box}
                output_data = self.transform_crop(input_data)  # 随机crop
                image = output_data['im']
            else:
                input_data = {'im': image, 'box': box}
                output_data = self.transform_crop_test(input_data)
                image = output_data['im']
                    
            if self.transform_image:
                image = self.transform_image(image)
            return image, text
        try:
            image, text = getitem(idx)
        except Exception as e:
            print('##error:', e, self.img_names_train[idx])
            image, text = getitem(0)

        return image, text
    
    def get_scores_accord(self,fnames):
        texts = []
        img_names = []
        anchors = []
        total = len(fnames)
        for i, fname in enumerate(fnames):
            print(f'{i}/{total}, {fname}')
            # fname = '/Users/tengbaoqiang/Documents/dataset/银泰智选/样衣素材_anyfit_merge/cebcd538cf08ea1b9ff4d454fd7ef612ec384911_m5.jpg'
            try:
                name = os.path.basename(fname)
                spu_id = name.split('_')[0]
                
                text_str = ''
                text = self.tag_data[spu_id]
                text, res = self.find_important_fea(text.copy())
                text_str += res
                # 获取每个spu的吊牌价格
                if spu_id in self.target_data_mj_only:
                    text_str += '吊牌价:' + str(int(self.target_data_mj_only[spu_id]['avg_ori_price'])) + ','
                
                text_str_other = str(text).replace('\'', '').replace('{', '').replace('}', '')
                text_str += text_str_other

                # 获取每个图片的anchor
                img2anchor_dict = self.spu2anchor[spu_id]
                anchor_multi = img2anchor_dict[os.path.basename(fname)]
                for anchor in anchor_multi:
                    box = [int(i) for i in anchor[0].split(',')]
                    cls = categorys_dict[int(anchor[1])]
                    text_str = '商品类别:' + cls + ',' + text_str
                    texts.append(text_str)
                    img_names.append(fname)
                    anchors.append(box)
            except Exception as e:
                print('error:', fname, e)
                # self.image_paths.remove(fname)
        return img_names, texts, anchors
    
    def find_important_fea(self, text):
        res = ''
        tmp_dict = {}
        keys = list(text.keys())
        for k in keys:
            v = str(text[k])
            v = '_'.join(list(set(v.split(','))))  # 去重， 如'无,无' -》 '无'
            if '无' == v or 'nan' == v:
                del text[k]  # 删除无信息的元素
                continue
            if '品牌' == k or '商品品牌' == k:
                if '品牌' in tmp_dict:
                    tmp_dict['品牌'].append(v)
                else:
                    tmp_dict['品牌'] = [v]
                del text[k]
            elif '面料' in k or '材质' in k or '材料' in k:
                if '面料' in tmp_dict:
                    tmp_dict['面料'].append(v)
                else:
                    tmp_dict['面料'] = [v]
                del text[k]
            elif '卖点' in k:
                if '卖点' in tmp_dict:
                    tmp_dict['卖点'].append(v)
                else:
                    tmp_dict['卖点'] = [v]
                del text[k]
            elif '商品名称' in k or '品名' in k:
                if '商品名称' in tmp_dict:
                    tmp_dict['商品名称'].append(v)
                else:
                    tmp_dict['商品名称'] = [v]
                del text[k]
            elif '身高' in k or '厚薄' in k or '建议' in k or '场合' in k or '性别' in k or '重' in k or '制造' in k \
                    or '市场' in k or '年龄' in k or '年份' in k or '洗涤' in k or '体型' in k or '人群' in k or '支撑' in k \
                    or '合格' in k or '戴帽' in k or '电话' in k or '价' in k or '公司' in k or '地' in k or '货号' in k\
                    or '尺码' in k or '技术' in k or '安全' in k or '标准' in k or '身份' in k or '包装' in k or '修饰' in k:
                del text[k]
            else:
                text[k] = v
        if '品牌' in tmp_dict:
            str_brand = '_'.join(list(set(tmp_dict['品牌'])))  # 去重， 如'无,无' -》 '无'
            res += '品牌:' + str_brand + ','
        if '面料' in tmp_dict:
            str_fabric = '_'.join(list(set(tmp_dict['面料'])))  # 去重
            res += '面料:' + str_fabric + ','
        if '卖点' in tmp_dict:
            str_sale = '_'.join(list(set(tmp_dict['卖点'])))  # 去重
            res += '卖点:' + str_sale + ','
        if '商品名称' in tmp_dict:
            str_name = '_'.join(list(set(tmp_dict['商品名称'])))  # 去重
            res += '商品名称:' + str_name + ','

        return text, res
