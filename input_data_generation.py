import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

def transform(image, mask):
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(512, 512))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    # Transform to tensor
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    return image, mask

def read_img(path):
    """
    Read Image
    """
    img = Image.open(path).convert("RGB")
    return img

def random_color_prior(img, resize_shape=[512,512]):
    img_to_pil = Image.fromarray(np.transpose(np.array(img * 255).astype('uint8'), (1, 2, 0)))

    # random selected pixels
    img_aug_1 = transforms_pre(img_to_pil)
    img_aug_2 = transforms_pre(img_to_pil)

    p_A = torch.rand(torch.tensor([1]))
    p_A = torch.matmul(p_A, torch.tensor([0.85]))
    p_B = torch.rand(torch.tensor([1]))
    p_B = torch.matmul(p_B, torch.tensor([0.85]))

    mask_A = torch.bernoulli(p_A * torch.ones(img_aug_1.size()[1:]))  # 실제론 2: 일거임
    mask_B = torch.bernoulli(p_B * torch.ones(img_aug_2.size()[1:]))  # 실제론 2: 일거임
    mask_inter = mask_A * mask_B
    mask_B = mask_B - mask_inter

    img_aug_A = img_aug_1 * mask_A
    img_aug_B = img_aug_2 * mask_B
    img_aug_AB = img_aug_A + img_aug_B

    # noise
    img_aug_AB = torch.normal(mean=img_aug_AB, std=0.015)
    img_aug_AB = img_aug_AB * (mask_A + mask_B - mask_inter)
    img_aug_AB = torch.clamp(img_aug_AB, 0, 1.)

    # masking
    random_ff_setting_cp = {'img_shape': [resize_shape[0], resize_shape[1]], 'minv':15, 'mv': 10, 'ma': 4.0, 'ml': 80, 'mbw': 80}
    m = random_ff_mask(random_ff_setting_cp)
    m = np.concatenate((m, m, m), axis=2)
    m_tensor = torch.tensor(1 - m).permute(2, 0, 1)
    mcolor = img_aug_AB * m_tensor

    # Dilation & Guassian for dotted color prior
    kernel = np.ones([3, 3])
    npmc = np.array(mcolor)
    npmc = np.transpose(npmc, [1, 2, 0])
    dil = cv2.dilate(npmc, kernel, iterations=1)
    ksize = (5, 5)
    sigmaX = 1
    gau = cv2.GaussianBlur(dil, ksize, sigmaX)
    gau = np.transpose(gau, (2, 0, 1))
    mcolor = torch.tensor(gau)

    return mcolor

def random_gl_pt(guide, resize_shape=[512,512]):
    #guide perturb
    guide_init = np.zeros((resize_shape[0], resize_shape[1]))
    size_x = np.random.randint(0, 5)
    size_y = np.random.randint(0, 5)
    cut_x = np.random.randint(10, resize_shape[0]-10, size=size_x)
    cut_y = np.random.randint(10, resize_shape[1]-10, size=size_y)
    cut_x = np.append(cut_x, [0, resize_shape[0]-1])
    cut_y = np.append(cut_y, [0, resize_shape[1]-1])
    cut_x = np.unique(cut_x)
    cut_y = np.unique(cut_y)
    cut_x.sort()
    cut_y.sort()

    guide_perturb = guide.squeeze(dim=0)
    for u in range(len(cut_x) - 1):
        for v in range(len(cut_y) - 1):
            cut_guide = guide_perturb[cut_x[u]:cut_x[u + 1]+1, cut_y[v]:cut_y[v + 1]+1]
            cut_guide = Image.fromarray(np.array(cut_guide * 255).astype('uint8'), mode='L')
            cut_guide = transforms_guide(cut_guide)
            guide_init[cut_x[u]:cut_x[u + 1]+1, cut_y[v]:cut_y[v + 1]+1] = cut_guide

    guide_perturb = torch.unsqueeze(torch.Tensor(guide_init),dim=0)

    # guide masking
    random_ff_setting_gl = {'img_shape': [resize_shape[0], resize_shape[1]], 'minv':4, 'mv': 5, 'ma': 4.0, 'ml': 60, 'mbw': 40}
    guide_m = random_ff_mask(random_ff_setting_gl)
    guide_m_tensor = torch.tensor(1 - guide_m).permute(2,0,1)
    guide_perturb = guide_perturb * guide_m_tensor

    return guide_perturb

def random_ff_mask(config):
    """
    Generate a random free form mask with configuration.
    """

    h,w = config['img_shape']
    mask = np.zeros((h,w))
    num_v = config['minv'] + np.random.randint(config['mv'])#tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        for j in range(1+np.random.randint(5)):
            # angle = 0.01+np.random.randint(config['ma'])
            angle = config['ma'] * np.random.rand()
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = np.random.randint(15, config['ml'])
            brush_w = np.random.randint(15, config['mbw'])
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y

    return mask.reshape(mask.shape+(1,)).astype(np.float32)


img_path= './data/sample_image.jpg' # path_to_image
guide_path= './data/sample_guideline.jpg' # path_to_guide
mask_path = './data/sample_mask.png'
transforms_pre = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
    transforms.RandomAffine(2, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=2, resample=False, fillcolor=0),
    transforms.ToTensor()
])
transforms_guide = transforms.Compose([
    transforms.RandomAffine(1, translate=(0.01, 0.01), scale=(0.99, 1.01), shear=0, resample=False, fillcolor=0),
    transforms.ToTensor()
])

# ground truth image and guideline: I_gt, L_gt
img, guide = transform(read_img(img_path), read_img(guide_path))
guide = torch.round(guide[0])

# color prior
color_prior = random_color_prior(img) # img shape: torch.Size([3, 512, 512])
# imperfect guieline: \widetilde{L}
imperfect_guideline = random_gl_pt(guide) # guide shape: torch.Size([1, 512, 512])

# mask, M
# mask = (np.array(read_img(mask_path))>0).astype('uint8')
mask = np.array(read_img(mask_path))
toTensor = transforms.ToTensor()
mask_tensor = toTensor(mask)

# no_color_mask, M_cp
no_color = ((color_prior + (1 - mask_tensor)) == 0)  # need to check
no_color = ((no_color[0, :, :] + no_color[1, :, :] + no_color[2, :, :]) > 1.5).unsqueeze(2).type(torch.float32)

# no_edge_zone, N
no_edge = cv2.Canny(np.array(no_color * 255).astype('uint8'), 100, 200)
kernel = np.ones([(np.round(img.shape[1]/100)).astype('uint8'), (np.round(img.shape[2])/100).astype('uint8')])
no_edge = cv2.dilate(no_edge, kernel)
no_edge = torch.tensor(no_edge / 255).unsqueeze(0).type(torch.float32)
no_edge = (1 - guide) * no_edge

transPIL = transforms.ToPILImage()

# save cropped ground truth image and guideline: I_gt, L_gt
img_save = Image.fromarray(np.array(img * 255).astype('uint8').transpose(1, 2, 0), 'RGB')
img_save.save('./data/result_crop_image_gt.png')
img_save = Image.fromarray(np.array(guide * 255).astype('uint8'), 'L')
img_save.save('./data/result_crop_guideline_gt.png')

# save color prior: I_cp ( color_prior * mask )
mask = (mask>0).astype('uint8')
img_save = Image.fromarray(np.array(color_prior* 255).astype('uint8').transpose(1, 2, 0)*mask, 'RGB')
img_save.save('./data/result_color_prior.png')

# save imperfect guieline: \widetilde{L}
img_save = Image.fromarray(np.array(imperfect_guideline[0] * 255).astype('uint8'), 'L')
img_save.save('./data/result_imperfect_guideline.png')

# save no_color_mask, M_cp
img_save = transPIL(no_color.permute(2,0,1))
img_save.save('./data/result_no_color_mask.png')

# save no_edge_zone, N
img_save = transPIL(no_edge)
img_save.save('./data/result_no_edge_zone.png')