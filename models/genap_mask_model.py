# import kornia
import torch
import torch.nn.functional as F
from torch import nn

import others.baseline_methods
from others.backbones import unet
from others.baseline_methods.loss import DiceLoss
from others.other_components.ginipa.adv_bias import AdvBias
from others.other_components.ginipa.imagefilter import GINGroupConv
from others.other_components.ginipa.smpmodels import EfficientUNet
from others.other_components.ginipa.utils import rescale_intensity
from utils import multi_class_vis
from .base_model import BaseModel


class GENAPMASKModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument('--blend_grid_size', type=int, default=24)
            parser.add_argument('--lambda_Seg', type=float, default=1.0)
            parser.add_argument('--lambda_consist', type=float, default=0.08)
            parser.add_argument('--do_mild_aug', action='store_true')

            # === Added for frequency complementary augmentation ===
            parser.add_argument('--use_freq_aug', action='store_true',
                                help='enable frequency-domain complementary augmentation (Genap-style)')
            parser.add_argument('--num_radial_bands', type=int, default=10,
                                help='number of radial frequency bands')
            parser.add_argument('--num_gap_bands', type=int, default=1,
                                help='number of under-augmented bands/regions to complement')

            # === NEW: region selection mode & params ===
            parser.add_argument('--freq_region_mode', type=str, default='radial',
                                help="how to define frequency regions: 'radial' | 'patch' | 'pixel'")
            parser.add_argument('--freq_patch_rows', type=int, default=8,
                                help='num of patch partitions along H in frequency domain (for patch mode)')
            parser.add_argument('--freq_patch_cols', type=int, default=8,
                                help='num of patch partitions along W in frequency domain (for patch mode)')
            parser.add_argument('--freq_gap_ratio', type=float, default=0.05,
                                help='ratio of frequency pixels as gap region when freq_region_mode=pixel')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['loss_all', 'loss_seg', 'loss_consist']
        self.model_names = ['net_main']
        self.visual_names = ['image_input', 'label', 'out_seg', 'out_seg_binary'] + (
            [] if not opt.is_train else ['image_transformed_1', 'image_transformed_2', 'image_transformed_3'])

        # self.net_main = EfficientUNet(in_channels=opt.input_nc, nclass=opt.output_nc).to(
        #     device=self.device)
        self.net_main = unet.Unet(
            opt.input_nc, opt.output_nc,
            output_mode=unet.Unet.PART_OUTPUT,
            last_layer='Identity'
        ).to(self.device)

        if self.opt.is_train:
            self.criterionDice = DiceLoss().to(device=self.device)
            self.criterionWCE = torch.nn.BCELoss().to(device=self.device)
            self.criterionCons = torch.nn.KLDivLoss().to(device=self.device)

            self.blend_grid_size = opt.blend_grid_size
            self.lambda_Seg = opt.lambda_Seg
            self.lambda_consist = opt.lambda_consist

            blender_config = {
                'epsilon': 0.3,
                'xi': 1e-6,
                'control_point_spacing': [opt.blend_grid_size, opt.blend_grid_size],
                'downscale': 2,  #
                'data_size': [opt.batch_size, opt.input_nc,
                              opt.crop_size[0] if 'crop' in opt.preprocess else opt.load_size[0],
                              opt.crop_size[0] if 'crop' in opt.preprocess else opt.load_size[0]],
                'interpolation_order': 2,
                'init_mode': 'gaussian',
                'space': 'log'
            }
            self.img_transform_node = GINGroupConv(
                in_channel=opt.input_nc, out_channel=opt.input_nc
            ).to(device=self.device)
            self.blender_node = AdvBias(blender_config)  # IPA
            self.blender_node.init_parameters()

            # === Added for frequency complementary augmentation (Genap-IFCA) ===
            self.use_freq_aug = opt.use_freq_aug
            if self.use_freq_aug:
                _, _, H, W = blender_config['data_size']
                self.H, self.W = H, W
                self.num_radial_bands = opt.num_radial_bands
                self.num_gap_bands = opt.num_gap_bands

                # NEW: region mode & params
                self.freq_region_mode = getattr(opt, 'freq_region_mode', 'radial')
                self.freq_patch_rows = getattr(opt, 'freq_patch_rows', 8)
                self.freq_patch_cols = getattr(opt, 'freq_patch_cols', 8)
                self.freq_gap_ratio = getattr(opt, 'freq_gap_ratio', 0.05)

                # ---- (1) Radial band indexing (原始环形分区) ----
                yy, xx = torch.meshgrid(
                    torch.arange(H, device=self.device),
                    torch.arange(W, device=self.device),
                    indexing='ij'
                )
                cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
                rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                r_max = rr.max()
                band_edges = torch.linspace(0, r_max, self.num_radial_bands + 1, device=self.device)

                band_id = torch.zeros(H, W, dtype=torch.long, device=self.device)
                for i in range(self.num_radial_bands):
                    mask = (rr >= band_edges[i]) & (rr < band_edges[i + 1])
                    band_id[mask] = i
                self.radial_band_id = band_id  # [H, W]

                # ---- (2) Patch-based region indexing (仅在 patch 模式使用) ----
                if self.freq_region_mode == 'patch':
                    # 把 H×W 均匀映射成 freq_patch_rows × freq_patch_cols 的网格 patch
                    y_idx = torch.arange(H, device=self.device) * self.freq_patch_rows // H  # [0..rows-1]
                    x_idx = torch.arange(W, device=self.device) * self.freq_patch_cols // W  # [0..cols-1]
                    patch_id = y_idx[:, None] * self.freq_patch_cols + x_idx[None, :]       # [H, W] in [0..rows*cols-1]
                    self.patch_id = patch_id
                    self.num_patches = int(self.freq_patch_rows * self.freq_patch_cols)

                # ---- 可学习的频域噪声参数 ----
                self.freq_mul = nn.Parameter(torch.zeros(1, 1, H, W, device=self.device))
                self.freq_add = nn.Parameter(torch.zeros(1, 1, H, W, device=self.device))
                nn.init.normal_(self.freq_mul, mean=0.0, std=0.02)
                nn.init.normal_(self.freq_add, mean=0.0, std=0.02)

                # === 独立的频域噪声优化器（只管 freq_mul / freq_add） ===
                self.optimizer_freq = torch.optim.Adam(
                    [self.freq_mul, self.freq_add],
                    lr=1e-3, betas=(0.9, 0.999)
                )

    def criterion_segmentation(self, pred, target):
        if self.opt.output_nc == 1:
            loss_ce = nn.BCELoss()(pred, target)
            loss_dice = DiceLoss()(pred, target)
        else:
            loss_ce = nn.CrossEntropyLoss()(pred, target)
            loss_dice = DiceLoss()(torch.softmax(pred, dim=1), target)
        return loss_ce + loss_dice

    # === Helper for frequency complementary augmentation ===
    def _apply_frequency_complementary_aug(self, aug_imgs_3copy, orig_imgs):
        """
        aug_imgs_3copy: [3B, C, H, W]  (already RandConv + GIN blending)
        orig_imgs:      [B,  C, H, W]  (original images)

        We:
        1) use the first augmented copy vs. original to estimate under-augmented regions
        2) build a gap mask R_gap (by radial band / patch / pixel)
        3) apply learnable frequency noise (freq_mul, freq_add) inside R_gap
        """
        B3, C, H, W = aug_imgs_3copy.shape
        B = orig_imgs.shape[0]
        assert H == self.H and W == self.W, "FFT size mismatch, check crop/load size."

        # 只用第一份增广（前 B 张）来估计增广差分
        aug_first = aug_imgs_3copy[:B]  # [B, C, H, W]
        orig = orig_imgs               # [B, C, H, W]

        # 计算频域（shift 后），只用 log amplitude 做差分
        def fft_log_amp(x):
            # x: [B, C, H, W]
            x_fft = torch.fft.fft2(x, dim=(-2, -1))
            x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
            mag = torch.log(torch.abs(x_fft_shift) + 1e-6)
            return x_fft_shift, mag

        _, orig_mag = fft_log_amp(orig)
        _, aug_mag = fft_log_amp(aug_first)

        diff = torch.abs(aug_mag - orig_mag) / (torch.abs(orig_mag) + 1e-6)  # [B, C, H, W]
        diff_mean = diff.mean(dim=(0, 1))  # [H, W]  聚合 batch 和 channel

        # ------------------------------------------------------------------
        # 根据 freq_region_mode 构造 gap_mask: [1,1,H,W] (bool)
        #   - 'radial': 按环形 band
        #   - 'patch' : 按 H×W 网格 patch
        #   - 'pixel' : 不分区，直接按像素点
        # ------------------------------------------------------------------
        mode = getattr(self, 'freq_region_mode', 'radial')

        if mode == 'patch':
            # === patch 网格分区：选差分最小的 num_gap_bands 个 patch ===
            patch_id = self.patch_id  # [H, W]
            band_means = []
            for i in range(self.num_patches):
                mask = (patch_id == i)
                if mask.any():
                    band_means.append(diff_mean[mask].mean())
                else:
                    band_means.append(torch.tensor(1e6, device=self.device))
            band_means = torch.stack(band_means)  # [num_patches]

            k = min(self.num_gap_bands, self.num_patches)
            _, gap_indices = torch.topk(band_means, k=k, largest=False)

            gap_mask2d = torch.zeros(H, W, dtype=torch.bool, device=self.device)
            for idx in gap_indices:
                gap_mask2d |= (patch_id == idx)

            gap_mask = gap_mask2d[None, None, :, :]  # [1,1,H,W]

        elif mode == 'pixel':
            # === 像素级选择：不分区，直接选若干差分最小的像素点 ===
            v = diff_mean.view(-1)  # [H*W]
            num_pixels = v.numel()
            # 至少 1 个像素，避免 k=0 的情况
            k = int(max(1, min(num_pixels, self.freq_gap_ratio * num_pixels)))
            _, gap_indices = torch.topk(v, k=k, largest=False)

            gap_mask_flat = torch.zeros_like(v, dtype=torch.bool)
            gap_mask_flat[gap_indices] = True
            gap_mask2d = gap_mask_flat.view(H, W)
            gap_mask = gap_mask2d[None, None, :, :]  # [1,1,H,W]

        else:
            # === 默认 radial 环形分区（原始实现） ===
            band_id = self.radial_band_id  # [H, W]
            band_means = []
            for i in range(self.num_radial_bands):
                mask = (band_id == i)
                if mask.any():
                    band_means.append(diff_mean[mask].mean())
                else:
                    band_means.append(torch.tensor(1e6, device=self.device))  # 空 band 给大值
            band_means = torch.stack(band_means)  # [num_radial_bands]

            k = min(self.num_gap_bands, self.num_radial_bands)
            _, gap_indices = torch.topk(band_means, k=k, largest=False)

            gap_mask2d = torch.zeros(H, W, dtype=torch.bool, device=self.device)
            for idx in gap_indices:
                gap_mask2d |= (band_id == idx)

            gap_mask = gap_mask2d[None, None, :, :]  # [1,1,H,W]

        # ------------------------------------------------------------------
        # 对所有 3 份增广图做频域扰动（只在 gap_mask 中生效）
        # ------------------------------------------------------------------
        x = aug_imgs_3copy  # [3B, C, H, W]
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))  # [3B, C, H, W]

        mul = 1.0 + self.freq_mul  # [1, 1, H, W]
        add = self.freq_add        # [1, 1, H, W]

        gap_mask_f = gap_mask.to(x_fft_shift.dtype)
        x_fft_shift_perturbed = x_fft_shift * (1.0 - gap_mask_f) + \
                                (x_fft_shift * mul + add) * gap_mask_f

        x_fft_unshift = torch.fft.ifftshift(x_fft_shift_perturbed, dim=(-2, -1))
        x_ifft = torch.fft.ifft2(x_fft_unshift, dim=(-2, -1))
        x_real = x_ifft.real  # 丢弃虚部

        # 根据需要加 clamp
        # x_real = torch.clamp(x_real, 0.0, 1.0)

        return x_real

    def set_input(self, data_dict):
        self.image_paths = data_dict['source_path']

        self.image_input = data_dict['image_original'].to(self.device)

        self.label = data_dict['label'].to(self.device)

        self.mask = data_dict['mask'].to(self.device)

        if self.opt.is_train:
            self._nb_current = self.image_input.shape[0]
            input_buffer = torch.cat([self.img_transform_node(self.image_input) for _ in range(3)], dim=0)

            self.blender_node.init_parameters()
            blend_mask = rescale_intensity(self.blender_node.bias_field)
            if self.opt.input_nc != 1:
                blend_mask = blend_mask.repeat(1, self.opt.input_nc, 1, 1)

            # spatially-variable blending
            input_cp1 = input_buffer[: self._nb_current].clone().detach() * blend_mask + \
                        input_buffer[self._nb_current: self._nb_current * 2].clone().detach() * (1.0 - blend_mask)
            input_cp2 = input_buffer[: self._nb_current] * (1 - blend_mask) + \
                        input_buffer[self._nb_current: self._nb_current * 2] * blend_mask

            input_buffer[: self._nb_current] = input_cp1
            input_buffer[self._nb_current: self._nb_current * 2] = input_cp2

            self.blend_mask = blend_mask.data
            self.input_img_3copy = input_buffer

            if self.opt.do_mild_aug:
                for i in range(3):
                    temp = input_buffer[i * self._nb_current:(i + 1) * self._nb_current]
                    self.input_img_3copy[
                    i * self._nb_current:(i + 1) * self._nb_current] = temp * 0.5 + self.image_input * 0.5

    def forward(self):
        if self.opt.is_train:
            self.out_seg, self.aux_pred = self.net_main(self.input_img_3copy)
        else:
            self.out_seg, self.aux_pred = self.net_main(self.image_input)

    def compute_visuals(self):
        if self.opt.is_train:
            self.out_seg = self.out_seg[: self._nb_current]
            self.image_transformed_1 = self.input_img_3copy[:self._nb_current]
            self.image_transformed_2 = self.input_img_3copy[self._nb_current:self._nb_current * 2]
            self.image_transformed_3 = self.input_img_3copy[self._nb_current * 2:]

        if self.opt.output_nc > 1:
            self.out_seg = multi_class_vis(self.out_seg)
            self.label = multi_class_vis(self.label)
            self.out_seg = self.multi_class_pred_remap(self.out_seg)
        else:
            self.out_seg = torch.sigmoid(self.out_seg)

        self.out_seg_binary = self.out_seg > 0.5

    def optimize_parameters(self):
        # ========== Phase 1: 更新分割网络（min） ==========
        self.optimizers_zero_grad()

        if self.opt.is_train and self.use_freq_aug:
            self.freq_mul.requires_grad_(False)
            self.freq_add.requires_grad_(False)

        if self.opt.is_train:
            if self.use_freq_aug:
                with torch.no_grad():  # 这一步只为 net 提供稳定输入，不对噪声求导
                    aug_imgs = self._apply_frequency_complementary_aug(
                        self.input_img_3copy, self.image_input
                    )
                self.input_img_3copy = aug_imgs.detach()
            else:
                self.input_img_3copy = self.input_img_3copy

        self.forward()

        # 主分支预测
        self.pred = self.out_seg[: self._nb_current]
        if self.opt.output_nc == 1:
            self.pred = torch.sigmoid(self.pred) * self.mask

        # segmentation loss
        self.loss_seg = self.criterion_segmentation(self.pred, self.label)

        # consistency loss
        pred_all_prob = torch.softmax(self.out_seg, dim=1)
        pred_avg = 1.0 / 3 * (
                pred_all_prob[: self._nb_current] +
                pred_all_prob[self._nb_current: self._nb_current * 2] +
                pred_all_prob[self._nb_current * 2:]
        )
        pred_avg = torch.cat([pred_avg for _ in range(3)], dim=0)
        pred_all = F.log_softmax(self.out_seg, dim=1)
        loss_consist = self.criterionCons(pred_all, pred_avg)
        self.loss_consist = self.lambda_consist * loss_consist

        # 目前 Phase1 只优化 seg loss（你原来就是这样写的）
        self.loss_all = self.loss_seg * self.lambda_Seg
        # 如果后面想开启 consistency，一行改成：
        # self.loss_all = self.loss_seg * self.lambda_Seg + self.loss_consist
        self.loss_all.backward()
        self.optimizers_step()

        # ========== Phase 2: 更新频域噪声（max） ==========
        if self.opt.is_train and self.use_freq_aug:
            self.freq_mul.requires_grad_(True)
            self.freq_add.requires_grad_(True)

            for _, p in self.net_main.named_parameters():
                p.requires_grad_(False)

            self.optimizer_freq.zero_grad()

            aug_imgs_freq = self._apply_frequency_complementary_aug(
                self.input_img_3copy, self.image_input
            )
            self.input_img_3copy = aug_imgs_freq
            self.forward()

            pred_all_prob = torch.softmax(self.out_seg, dim=1)
            pred_avg = 1.0 / 3 * (
                    pred_all_prob[: self._nb_current] +
                    pred_all_prob[self._nb_current: self._nb_current * 2] +
                    pred_all_prob[self._nb_current * 2:]
            )
            pred_avg = torch.cat([pred_avg for _ in range(3)], dim=0)
            pred_all = F.log_softmax(self.out_seg, dim=1)
            loss_consist_freq = self.criterionCons(pred_all, pred_avg)

            loss_freq = - self.lambda_consist * loss_consist_freq*0.0001
            loss_freq.backward()
            self.optimizer_freq.step()

            for _, p in self.net_main.named_parameters():
                p.requires_grad_(True)

    def update_metrics(self):
        if self.opt.output_nc == 1:
            self.out_seg = torch.sigmoid(self.out_seg) * self.mask
        self.metrics.update(self.out_seg, self.label)
