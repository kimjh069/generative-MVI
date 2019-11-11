import torch
import torch.nn as nn
import torch.nn.functional as F
from source.networks import GatedConv2dWithActivation, GatedDeConv2dWithActivation, SNConvWithActivation, get_pad
from source.tools import extract_image_patches, flow_to_image, reduce_mean, reduce_sum, same_padding


class MVIGenerator(torch.nn.Module):
    """
        Inpainting generator, input shape: 9*H*W (ori_img=3, mask=1, guide_perturb=1, color-prior(with noise)=3, non-color-mask=1)
    """
    def __init__(self, img_shape, n_in_channel=9):
        super(MVIGenerator, self).__init__()
        cnum_c = 20
        #################################################
        # Coarse Reconstruction
        #################################################
        self.c1_1 = GatedConv2dWithActivation(n_in_channel, cnum_c, 3, 1, padding=get_pad(img_shape, 3, 1))
        self.c1_2 = GatedConv2dWithActivation(cnum_c, cnum_c, 3, 1, padding=get_pad(img_shape, 3, 1))
        # downsample 1/2
        self.c2_1 = GatedConv2dWithActivation(cnum_c, 2 * cnum_c, 3, 2, padding=get_pad(img_shape, 3, 1))
        self.c2_2 = GatedConv2dWithActivation(2 * cnum_c, 2 * cnum_c, 3, 1, padding=get_pad(img_shape//2, 3, 1))
        # downsample to 64
        self.c4_1 = GatedConv2dWithActivation(2 * cnum_c, 4 * cnum_c, 3, 2, padding=get_pad(img_shape//2, 3, 1))
        self.c4_2 = GatedConv2dWithActivation(4 * cnum_c, 4 * cnum_c, 3, 1, padding=get_pad(img_shape//4, 3, 1))
        # downsample to 32
        self.c8_1 = GatedConv2dWithActivation(4 * cnum_c, 8 * cnum_c, 3, 2, padding=get_pad(img_shape//4, 3, 1))
        self.c8_2 = GatedConv2dWithActivation(8 * cnum_c, 8 * cnum_c, 3, 1, padding=get_pad(img_shape//8, 3, 1))
        # self.c8_3 = GatedConv2dWithActivation(8 * cnum, 8 * cnum, 3, 1, padding=get_pad(img_shape//8, 3, 1))
        # downsample to 16
        self.c16_1 = GatedConv2dWithActivation(8 * cnum_c, 12 * cnum_c, 3, 2, padding=get_pad(img_shape//8, 3, 1))
        # atrous convlution
        self.c_dil_3 = GatedConv2dWithActivation(12 * cnum_c, 12 * cnum_c, 3, 1, dilation=3, padding=get_pad(img_shape//16, 3, 1, 3))
        self.c_dil_7 = GatedConv2dWithActivation(12 * cnum_c, 12 * cnum_c, 3, 1, dilation=7, padding=get_pad(img_shape//16, 3, 1, 7))
        # self attention
        self.c16_4 = GatedConv2dWithActivation(12 * cnum_c, 8 * cnum_c, 1, 1, padding=get_pad(img_shape//16, 1, 1))
        self.c16_sa = Self_Attn(8 * cnum_c, 'relu')
        # upsample to 32
        self.de_c8_1 = GatedDeConv2dWithActivation(2, 8 * cnum_c, 8 * cnum_c, 3, 1, padding=get_pad(img_shape//8, 3, 1))
        self.de_c8_2 = GatedConv2dWithActivation(8 * cnum_c, 8 * cnum_c, 3, 1, padding=get_pad(img_shape//8, 3, 1))
        # upsample to 64
        self.de_c4_1 = GatedDeConv2dWithActivation(2, 8 * cnum_c, 4 * cnum_c, 3, 1, padding=get_pad(img_shape//4, 3, 1))
        self.de_c4_2 = GatedConv2dWithActivation(4 * cnum_c, 4 * cnum_c, 3, 1, padding=get_pad(img_shape//4, 3, 1))
        # Dil in 64
        self.c_dil_conv_branch = nn.Sequential(
            GatedConv2dWithActivation(4 * cnum_c, 4 * cnum_c, 3, 1, dilation=3, padding=get_pad(64, 3, 1, 3)),
            GatedConv2dWithActivation(4 * cnum_c, 4 * cnum_c, 3, 1, dilation=7, padding=get_pad(64, 3, 1, 7)),
            GatedConv2dWithActivation(4 * cnum_c, 4 * cnum_c, 3, 1, dilation=15, padding=get_pad(64, 3, 1, 15)),
        )
        # CA in 64 - CA branch
        # upsample to 32
        self.de_c8_1_CA = GatedDeConv2dWithActivation(2, 8 * cnum_c, 8 * cnum_c, 3, 1, padding=get_pad(img_shape//8, 3, 1))
        self.de_c8_2_CA = GatedConv2dWithActivation(8 * cnum_c, 8 * cnum_c, 3, 1, padding=get_pad(img_shape//8, 3, 1))
        # upsample to 64
        self.de_c4_1_CA = GatedDeConv2dWithActivation(2, 8 * cnum_c, 4 * cnum_c, 3, 1, padding=get_pad(img_shape//4, 3, 1))
        self.de_c4_2_CA = GatedConv2dWithActivation(4 * cnum_c, 4 * cnum_c, 3, 1, padding=get_pad(img_shape//4, 3, 1))
        self.c_context_attn = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                       fuse=True, use_cuda=True, device_ids=0)
        self.c_context_attn_branch= nn.Sequential(
            GatedConv2dWithActivation(4 * cnum_c, 4 * cnum_c, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum_c, 4 * cnum_c, 3, 1, padding=get_pad(64, 3, 1))
        )
        self.de_c4_4 = GatedConv2dWithActivation(2 * 4 * cnum_c, 4 * cnum_c, 3, 1, padding=get_pad(img_shape // 4, 3, 1))
        self.de_c4_5 = GatedConv2dWithActivation(4 * cnum_c, 4 * cnum_c, 3, 1, padding=get_pad(img_shape // 4, 3, 1))
        # upsample to 1/2
        self.de_c2_1 = GatedDeConv2dWithActivation(2, 4 * cnum_c, 2 * cnum_c, 3, 1, padding=get_pad(img_shape//2, 3, 1))
        self.de_c2_2 = GatedConv2dWithActivation(2 * cnum_c, 2 * cnum_c, 3, 1, padding=get_pad(img_shape//2, 3, 1))
        ## Coarse 1/2 Reconstruction
        self.coarse = GatedConv2dWithActivation(2 * cnum_c, 4, 3, 1, padding=get_pad(img_shape // 2, 3, 1))
        #################################################
        # Refine Net
        cnum = 32
        #################################################
        # From 1/2
        self.r2_1 = GatedConv2dWithActivation(4, 2 * cnum, 3, 1, padding=get_pad(img_shape//2, 3, 1))
        self.r2_2 = GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(img_shape//2, 3, 1))
        # downsample to 64
        self.r4_1 = GatedConv2dWithActivation(2 * cnum, 4 * cnum, 3, 2, padding=get_pad(img_shape // 2, 3, 1))
        self.r4_2 = GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(img_shape // 4, 3, 1))
        # downsample to 32
        self.r8_1 = GatedConv2dWithActivation(4 * cnum, 8 * cnum, 3, 2, padding=get_pad(img_shape // 4, 3, 1))
        self.r8_2 = GatedConv2dWithActivation(8 * cnum, 8 * cnum, 3, 1, padding=get_pad(img_shape // 8, 3, 1))
        # downsample to 16
        self.r16_1 = GatedConv2dWithActivation(8 * cnum, 12 * cnum, 3, 2, padding=get_pad(img_shape // 8, 3, 1))
        # atrous convlution
        self.r_dil_3 = GatedConv2dWithActivation(12 * cnum, 12 * cnum, 3, 1, dilation=3, padding=get_pad(img_shape // 16, 3, 1, 3))
        self.r_dil_7 = GatedConv2dWithActivation(12 * cnum, 12 * cnum, 3, 1, dilation=7, padding=get_pad(img_shape // 16, 3, 1, 7))
        # self attention
        self.r16_4 = GatedConv2dWithActivation(12 * cnum, 8 * cnum, 1, 1, padding=get_pad(img_shape // 16, 1, 1))
        self.r16_sa = Self_Attn(8 * cnum, 'relu')
        # upsample to 32
        self.de_r8_1 = GatedDeConv2dWithActivation(2, 8 * cnum, 8 * cnum, 3, 1, padding=get_pad(img_shape // 8, 3, 1))
        self.de_r8_2 = GatedConv2dWithActivation(8 * cnum, 8 * cnum, 3, 1, padding=get_pad(img_shape // 8, 3, 1))
        # upsample to 64
        self.de_r4_1 = GatedDeConv2dWithActivation(2, 8 * cnum, 4 * cnum, 3, 1, padding=get_pad(img_shape // 4, 3, 1))
        self.de_r4_2 = GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(img_shape // 4, 3, 1))
        # Dil in 64
        self.r_dil_conv_branch = nn.Sequential(
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=3, padding=get_pad(64, 3, 1, 3)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=7, padding=get_pad(64, 3, 1, 7)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=15, padding=get_pad(64, 3, 1, 15)),
        )
        # CA in 64
        # upsample to 32
        self.de_r8_1_CA = GatedDeConv2dWithActivation(2, 8 * cnum, 8 * cnum, 3, 1, padding=get_pad(img_shape // 8, 3, 1))
        self.de_r8_2_CA = GatedConv2dWithActivation(8 * cnum, 8 * cnum, 3, 1, padding=get_pad(img_shape // 8, 3, 1))
        # upsample to 64
        self.de_r4_1_CA = GatedDeConv2dWithActivation(2, 8 * cnum, 4 * cnum, 3, 1, padding=get_pad(img_shape // 4, 3, 1))
        self.de_r4_2_CA = GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(img_shape // 4, 3, 1))
        self.r_context_attn = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                       fuse=True, use_cuda=True, device_ids=0)
        self.r_context_attn_branch= nn.Sequential(
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1))
        )
        self.de_r4_4 = GatedConv2dWithActivation(2 * 4 * cnum, 4 * cnum, 3, 1, padding=get_pad(img_shape // 4, 3, 1))
        self.de_r4_5 = GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(img_shape // 4, 3, 1))
        # upsample to 1/2
        self.de_r2_1 = GatedDeConv2dWithActivation(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(img_shape // 2, 3, 1))
        self.de_r2_2 = GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(img_shape // 2, 3, 1))
        # upsample to 256
        self.de_r1_1 = GatedDeConv2dWithActivation(2, 2 * cnum, cnum, 3, 1, padding=get_pad(img_shape, 3, 1))
        self.de_r1_2 = GatedConv2dWithActivation(cnum, 4, 3, 1, padding=get_pad(img_shape, 3, 1), activation=None)

    def forward(self, imgs, imgs_sm, masks, masks_sm, img_guide, img_guide_sm, colors, no_color):
        masked_imgs =  imgs * (1 - masks) + masks
        masked_guides = img_guide * masks
        masked_colors = colors * masks
        input_imgs = torch.cat([masked_imgs, masks, masked_guides, masked_colors, no_color], dim=1)
        #################################################
        # Coarse Reconstruction
        #################################################
        c1_1 = self.c1_1(input_imgs)
        c1_2 = self.c1_2(c1_1)
        # downsample 1/2
        c2_1 = self.c2_1(c1_2)
        c2_2 = self.c2_2(c2_1)
        # downsample to 1/4
        c4_1 = self.c4_1(c2_2)
        c4_2 = self.c4_2(c4_1)
        # downsample to 1/8
        c8_1 = self.c8_1(c4_2)
        c8_2 = self.c8_2(c8_1)
        # downsample to 1/16
        c16_1 = self.c16_1(c8_2)
        # atrous convlution
        c_dil_3 = self.c_dil_3(c16_1)
        c_dil_7 = self.c_dil_7(c_dil_3)
        # self attention
        c16_4 = self.c16_4(c_dil_7)
        c16_sa = self.c16_sa(c16_4)

        # upsample to 1/8
        de_c8_1 = self.de_c8_1(c16_sa)
        de_c8_2 = self.de_c8_2(de_c8_1)
        # upsample to 1/4
        de_c4_1 = self.de_c4_1(de_c8_2)
        de_c4_2 = self.de_c4_2(de_c4_1)
        # Dil in 1/4
        dil_c4 = self.c_dil_conv_branch(de_c4_2)

        # upsample to 1/8
        de_c8_1_CA = self.de_c8_1_CA(c16_sa)
        de_c8_2_CA = self.de_c8_2_CA(de_c8_1_CA)
        # upsample to 1/4
        de_c4_1_CA = self.de_c4_1_CA(de_c8_2_CA)
        de_c4_2_CA = self.de_c4_2_CA(de_c4_1_CA)
        ctx_c4, flow_c = self.c_context_attn(de_c4_2_CA, de_c4_2_CA, masks)
        ctx_c4 = self.c_context_attn_branch(ctx_c4)

        cat_c4 = torch.cat([dil_c4, ctx_c4], dim=1)
        de_c4_4 = self.de_c4_4(cat_c4)
        de_c4_5 = self.de_c4_5(de_c4_4)
        # upsample to 1/2
        de_c2_1 = self.de_c2_1(de_c4_5)
        de_c2_2 = self.de_c2_2(de_c2_1)
        ## Coarse 1/2 Reconstruction
        coarse_out = torch.clamp(self.coarse(de_c2_2), -1.5, 1.5)
        coarse = torch.clamp(coarse_out, -1., 1.)
        coarse_complete = torch.cat([imgs_sm, img_guide_sm], dim=1) * (1 - masks_sm) + coarse * masks_sm
        #################################################
        # Refine Net
        #################################################
        # downsample 1/2
        r2_1 = self.r2_1(coarse_complete)
        r2_2 = self.r2_2(r2_1)
        # downsample to 1/4
        r4_1 = self.r4_1(r2_2)
        r4_2 = self.r4_2(r4_1)
        # downsample to 1/8
        r8_1 = self.r8_1(r4_2)
        r8_2 = self.r8_2(r8_1)
        # downsample to 1/16
        r16_1 = self.r16_1(r8_2)
        # atrous convlution
        r_dil_3 = self.r_dil_3(r16_1)
        r_dil_7 = self.r_dil_7(r_dil_3)
        # self attention
        r16_4 = self.r16_4(r_dil_7)
        r16_sa = self.r16_sa(r16_4)

        # upsample to 1/8
        de_r8_1 = self.de_r8_1(r16_sa)
        de_r8_2 = self.de_r8_2(de_r8_1)
        # upsample to 1/4
        de_r4_1 = self.de_r4_1(de_r8_2)
        de_r4_2 = self.de_r4_2(de_r4_1)
        # Dil in 1/4
        dil_r4 = self.r_dil_conv_branch(de_r4_2)

        # CA in 1/4
        de_r8_1_CA = self.de_r8_1_CA(r16_sa)
        de_r8_2_CA = self.de_r8_2_CA(de_r8_1_CA)
        # upsample to 1/4
        de_r4_1_CA = self.de_r4_1_CA(de_r8_2_CA)
        de_r4_2_CA = self.de_r4_2_CA(de_r4_1_CA)
        ctx_r4, flow_r = self.r_context_attn(de_r4_2_CA, de_r4_2_CA, masks)
        ctx_r4 = self.r_context_attn_branch(ctx_r4)

        cat_r4 = torch.cat([dil_r4, ctx_r4], dim=1)
        de_r4_4 = self.de_r4_4(cat_r4)
        de_r4_5 = self.de_r4_5(de_r4_4)
        # upsample to 1/2
        de_r2_1 = self.de_r2_1(de_r4_5)
        de_r2_2 = self.de_r2_2(de_r2_1)
        # upsample
        de_r1_1 = self.de_r1_1(de_r2_2)
        de_r1_2 = self.de_r1_2(de_r1_1)

        x = torch.clamp(de_r1_2, -1.5, 1.5)

        return coarse_out, x, flow_c, flow_r

class SNDirciminator(nn.Module):
    def  __init__(self):
        super(SNDirciminator, self).__init__()
        cnum = 32
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(8, 2*cnum, 4, 2, padding=get_pad(256, 5, 2)),
            SNConvWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(4*cnum, 8*cnum, 4, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(32, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(16, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(8, 5, 2)),
        )
    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0),-1))
        return x

class Self_Attn(nn.Module):
    """
        Self attention Layer
        pytorch implementation from : https://github.com/avalonstrel/GatedConvolution_pytorch
    """
    def __init__(self,in_dim,activation,with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out,attention
        else:
            return out

class ContextualAttention(nn.Module):
    """
    pytorch implementation from : https://github.com/DAA233/generative-inpainting-pytorch
    """
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=False, use_cuda=True, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda
        self.device_ids = device_ids

    def forward(self, f, b, mask=None):
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes
        raw_int_fs = list(f.size())   # b*c*h*w
        raw_int_bs = list(b.size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate*self.stride,
                                               self.rate*self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1./self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1./self.rate, mode='nearest')
        int_fs = list(f.size())     # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]])
            if self.use_cuda:
                mask = mask.cuda()
        else:
            mask = F.interpolate(mask, scale_factor=1./(4*self.rate), mode='nearest')
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        m = m[0]    # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True)==0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3) # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda()

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               escape_NaN)
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

            if int_bs != int_fs:
                # Normalize the offset value to match foreground dimension
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat([offset//int_fs[3], offset%int_fs[3]], dim=1)  # 1*2*H*W

            # deconv for patch pasting
            wi_center = raw_wi[0]
            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs)

        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_fs[0], 2, *int_fs[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(int_fs[2]).view([1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3]).view([1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        ref_coordinate = torch.cat([h_add, w_add], dim=1)
        if self.use_cuda:
            ref_coordinate = ref_coordinate.cuda()

        offsets = offsets - ref_coordinate
        # flow = pt_flow_to_image(offsets)

        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        if self.use_cuda:
            flow = flow.cuda()
        # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.long()).cpu().data.numpy()))

        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate*4, mode='nearest')

        return y, flow