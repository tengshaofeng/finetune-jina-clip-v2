#!/usr/bin/python
# -*- encoding: utf-8 -*-


from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import random
import numpy as np
from numpy.random import randint as randint

class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = self.size
        w, h = im.size

        ## 一定几率原图直接resize到目标大小(训练头发分割的时候加的，也许也适合之前的10分类)
        if random.random() > 0.3:
            return dict(
                im=im.resize((W, H), Image.LANCZOS),
                lb=lb  # .resize((W, H), Image.BILINEAR)
            )

        if (W, H) == (w, h): return dict(im=im, lb=lb)
        # if w < W or h < H:  # tbq comment
        #     scale = float(W) / w if w < h else float(H) / h
        #     w, h = int(scale * w + 1), int(scale * h + 1)
        #     im = im.resize((w, h), Image.ANTIALIAS)
        #     lb = lb.resize((w, h), Image.BILINEAR)
        if w < W or h < H:  # tbq add 若进来的图片是比较小的如384x384则将其随机放在512x512的零图里，然后在随机crop出需要的448x448
            new_w = new_h = max(w, h)
            dst_im = Image.new('RGB', (new_w, new_h))
            # dst_lb = Image.new('L', (new_w, new_h))
            x0 = randint(0, new_w - w + 1)
            y0 = randint(0, new_h - h + 1)
            box = (x0, y0, x0 + w, y0 + h)
            dst_im.paste(im, box)
            # dst_lb.paste(lb, box)
            im = dst_im
            # lb = dst_lb
            w, h = new_w, new_h  # >=512

        dst_size = int(max(W, H) * (1.2))
        if w < h:
            new_w = dst_size
            new_h = int(h / w * new_w)
        else:
            new_h = dst_size
            new_w = int(w / h * new_h)

        im = im.resize((new_w, new_h), resample=Image.LANCZOS)
        w, h = im.size
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
                im = im.crop(crop),
                lb = lb # .crop(crop)
                    )


class RandomCropGivenMask(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = self.size
        w, h = im.size
        ## 一定几率原图直接resize到目标大小(训练头发分割的时候加的，也许也适合之前的10分类)
        if random.random() > 0.3:
            return dict(
                im=im.resize((W, H), Image.LANCZOS),
                lb=lb  # .resize((W, H), Image.BILINEAR)
            )

        if (W, H) == (w, h): return dict(im=im, lb=lb)
        if w < W or h < H:  # 最小的边都大于目标最大边
            dst_size = max(W, H)
            if w < h:
                new_w = dst_size
                new_h = int(h / w * new_w)
            else:
                new_h = dst_size
                new_w = int(w / h * new_h)
            lb = im.resize((new_w, new_h), resample=Image.BILINEAR)
            im = im.resize((new_w, new_h), resample=Image.LANCZOS)
            w, h = im.size
        x0, y0, x1, y1 = lb.getbbox()
        sw, sh = random.random() * x0, random.random() * y0
        ew, eh = random.randint(x1-1, w-1), random.randint(y1-1, h-1)
        crop = int(sw), int(sh), ew, eh
        return dict(
                im = im.crop(crop),
                lb = lb # .crop(crop)
                    )


class RandomCropSqureGivenMask(object):
    def __init__(self, size, threshold=0.5, *args, **kwargs):
        '''

        :param size:
        :param threshold: 取最小人体框的阈值
        :param args:
        :param kwargs:
        '''
        self.size = size
        self.threshold = threshold

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        if self.threshold > 0:
            img_cv = bgra2bgr(np.array(im), mask=np.array(lb))
            im = Image.fromarray(img_cv)  # 避免颜色增强的时候改了白背景
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im=im, lb=lb)
        if w < W or h < H:  # 最小的边都大于目标最大边
            dst_size = max(W, H)
            if w < h:
                new_w = dst_size
                new_h = int(h / w * new_w)
            else:
                new_h = dst_size
                new_w = int(w / h * new_h)
            lb = im.resize((new_w, new_h), resample=Image.BILINEAR)
            im = im.resize((new_w, new_h), resample=Image.LANCZOS)
            w, h = im.size
        x0, y0, x1, y1 = lb.getbbox()
        if random.random() > self.threshold:  # 一定几率原图直接取人体正方形框
            sh = y0
            eh = y1 - 1
        else:
            sh = int(random.random() * y0)
            eh = int(random.randint(y1-1, h-1))
        tar_h = int(eh - sh)
        sw = (w - tar_h) // 2
        if sw > 0:
            ew = sw + tar_h
        else:
            sw = 0
            ew = w
        crop = int(sw), int(sh), int(ew), int(eh)
        crop_im = im.crop(crop)
        if crop_im.size[0] != crop_im.size[1]:
            image = Image.new("RGB", (tar_h, tar_h), "white")
            position = ((tar_h - crop_im.size[0]) // 2, 0)
            image.paste(crop_im, position)
        else:
            image = crop_im
        return dict(
                im = image,
                lb = lb # .crop(crop)
                    )


class RandomCropImgOnly(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im):
        W, H = self.size
        w, h = im.size

        ## 一定几率原图直接resize到目标大小(训练头发分割的时候加的，也许也适合之前的10分类)
        if random.random() > 0.4:
            return im.resize((W, H), Image.LANCZOS)

        if (W, H) == (w, h): return im
        # if w < W or h < H:  # tbq comment
        #     scale = float(W) / w if w < h else float(H) / h
        #     w, h = int(scale * w + 1), int(scale * h + 1)
        #     im = im.resize((w, h), Image.ANTIALIAS)
        #     lb = lb.resize((w, h), Image.BILINEAR)
        if w < W or h < H:  # tbq add 若进来的图片是比较小的如384x384则将其随机放在512x512的零图里，然后在随机crop出需要的448x448
            new_w = new_h = max(w, h)
            dst_im = Image.new('RGB', (new_w, new_h))
            # dst_lb = Image.new('L', (new_w, new_h))
            x0 = randint(0, new_w - w + 1)
            y0 = randint(0, new_h - h + 1)
            box = (x0, y0, x0 + w, y0 + h)
            dst_im.paste(im, box)
            # dst_lb.paste(lb, box)
            im = dst_im
            # lb = dst_lb
            w, h = new_w, new_h  # >=512

        dst_size = int(max(W, H) * (1.1))  # 1.2
        if w < h:
            new_w = dst_size
            new_h = int(h / w * new_w)
        else:
            new_h = dst_size
            new_w = int(w / h * new_h)

        im = im.resize((new_w, new_h), resample=Image.LANCZOS)
        w, h = im.size
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return im.crop(crop)


class RandomCropImgOnlyGivenBox(object):
    def __init__(self, padding=0.2, is_train=True):
        """
        Args:
            padding (float): box外扩比例 (默认外扩20%)
        """
        self.padding = padding
        self.is_train = is_train

    @staticmethod
    def expand_box(box, img_width, img_height, padding_ratio):
        """
        根据padding比例外扩box (保持中心点不变)
        """
        x_min, y_min, x_max, y_max = box
        box_w = x_max - x_min
        box_h = y_max - y_min
        
        # 计算外扩后的宽高
        new_h = max(box_w, box_h) * (1 + padding_ratio) # box_h * (1 + padding_ratio)
        new_w = new_h # box_w * (1 + padding_ratio)
        
        # 计算新box中心点
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        
        # 计算新box坐标 (确保不超出图像边界)
        new_x_min = max(0, cx - new_w / 2)
        new_y_min = max(0, cy - new_h / 2)
        new_x_max = min(img_width, cx + new_w / 2)
        new_y_max = min(img_height, cy + new_h / 2)
        
        return (new_x_min, new_y_min, new_x_max, new_y_max)

    def get_crop_params(self, img, box):
        """
        计算裁剪区域坐标 (确保裁剪区域至少包含原始的box)
        """
        img_width, img_height = img.size
        
        # 不扩展原始box
        x_min, y_min, x_max, y_max = box
        box_w = x_max - x_min
        box_h = y_max - y_min
        # 外扩box
        x_min_exp, y_min_exp, x_max_exp, y_max_exp  = self.expand_box(box, img_width, img_height, self.padding)
        exp_box_w = x_max_exp - x_min_exp
        exp_box_h = y_max_exp - y_min_exp
        # Determine the range for the possible side lengths of the crop
        min_side_length = max(box_w, box_h)
        max_side_length = min(exp_box_w, exp_box_h)
        # 随机产生裁剪起点 (x, y)，必须确保包含原始的 box
        if self.is_train:
            # Randomly choose the side length of the square crop area within the valid range
            crop_side = random.uniform(min_side_length, max_side_length)

            # 计算起始点的有效范围，以保证裁剪区域完全覆盖 box
            min_x0 = max(x_min_exp, x_max - crop_side)
            min_y0 = max(y_min_exp, y_max - crop_side)

            max_x0 = min(x_min, x_max_exp - crop_side)
            max_y0 = min(y_min, y_max_exp - crop_side)

            # 随机选择有效范围内的起始点
            x0 = random.uniform(min_x0, max_x0)
            y0 = random.uniform(min_y0, max_y0)
            
        else:
            # Fixed size for validation/test mode, using the min side length
            crop_side = min_side_length
            x0 = (x_min + x_max) / 2 - crop_side / 2
            y0 = (y_min + y_max) / 2 - crop_side / 2
        
        x1 = x0 + crop_side
        y1 = y0 + crop_side
        return (int(x0), int(y0), int(x1), int(y1))

    def __call__(self, im_lb):
        """
        Args:
            img (PIL Image): 输入图像
            box (tuple): 原box坐标 (x_min, y_min, x_max, y_max)
        Returns:
            PIL Image: 裁剪后的图像
        """
        img = im_lb['im']
        box = im_lb['box']
        # 获取裁剪参数
        x0, y0, x1, y1 = self.get_crop_params(img, box)
        
        # 执行裁剪操作并返回裁剪后的图像
        img = img.crop((x0, y0, x1, y1))
        im_lb['im'] = img
        return im_lb

    

class HorizontalFlip(object):
    def __init__(self, p=0.5, is_simple=False, *args, **kwargs):
        self.p = p
        self.is_simple = is_simple  # is_simple: true(19类)， false(10类)

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']

            # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
            #         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

            flip_lb = np.array(lb)
            if not self.is_simple:
                flip_lb[lb == 2] = 3
                flip_lb[lb == 3] = 2
                flip_lb[lb == 4] = 5
                flip_lb[lb == 5] = 4
                flip_lb[lb == 7] = 8
                flip_lb[lb == 8] = 7
            flip_lb = Image.fromarray(flip_lb)
            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = flip_lb.transpose(Image.FLIP_LEFT_RIGHT),
                    )


class HorizontalFlipImg(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(im=im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb=lb.transpose(Image.FLIP_LEFT_RIGHT),
                    )


class HorizontalFlipImgOnly(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im):
        if random.random() > self.p:
            return im
        else:
            return im.transpose(Image.FLIP_LEFT_RIGHT)

class HorizontalFlipImgOnlyGivenBox(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im_lb['im'] = im
            return im_lb
        

class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return dict(im = im.resize((w, h), Image.LANCZOS),  # tbq modify from Image.BILINEAR to Image.ANTIALIAS
                    lb = lb, #.resize((w, h), Image.BILINEAR), # 注意这里改成ANTIALIAS程序会cuda错误，tbq modify from Image.NEAREST to Image.BILINEAR
                )


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)

        return dict(im = im,
                    lb = lb,
                )


class ColorJitterImgOnly(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im):
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)

        return im


class ColorJitterImgOnlyGivenBox(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        im = im_lb['im']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        im_lb['im'] = im
        return im_lb
    
class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W*ratio), int(H*ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.LANCZOS)) for size in sizes]
        return imgs


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        # import uuid
        # name = str(uuid.uuid1())
        # im_lb['im'].save('tbq_%s.png' % name)
        return im_lb


def bgra2bgr(im_cv, mask=None, bg=255):
    res = im_cv
    if im_cv.shape[-1] > 3:
        alpha = im_cv[..., -1] / 255.0
        alpha = np.expand_dims(alpha, axis=-1)
        forhead = alpha * im_cv[:, :, :3]
        res = (1 - alpha) * bg * np.ones(
            (forhead.shape[0], forhead.shape[1], forhead.shape[2])) + forhead  # 128 tbq org:255
        res = res.astype(np.uint8)
    elif mask is not None:
        alpha = mask[..., -1] / 255.0 if len(mask.shape) > 2 else mask / 255.0
        alpha = np.expand_dims(alpha, axis=-1)
        forhead = im_cv
        res = (1 - alpha) * bg * np.ones(
            (forhead.shape[0], forhead.shape[1], forhead.shape[2])) + forhead * alpha  # 128 tbq org:255
        res = res.astype(np.uint8)  # 这一句还能把没有背景区域复原了？卧槽

    return res


if __name__ == '__main__':
    flip = HorizontalFlip(p = 1)
    crop = RandomCrop((321, 321))
    rscales = RandomScale((0.75, 1.0, 1.5, 1.75, 2.0))
    img = Image.open('data/img.jpg')
    lb = Image.open('data/label.png')
