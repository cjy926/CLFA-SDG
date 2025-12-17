import random
import math

import torch

from others.frequency_transforms.dft import DFT


def low_freq_mutate_dft(amp_src, amp_trg, L=0.1, ratio=None):
    amp_src_ = amp_src.clone().detach()
    amp_trg_ = amp_trg.clone().detach()

    *_, h, w = amp_src.shape
    b = math.floor(min(h, w) * L)
    c_h = math.floor(h / 2)
    c_w = math.floor(w / 2)
    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    ratio = ratio if ratio is not None else random.uniform(0, 1.0)
    amp_src_[:, :, h1:h2, w1:w2] = amp_src_[:, :, h1:h2, w1:w2] * ratio + amp_trg_[:, :, h1:h2, w1:w2] * (1 - ratio)
    return amp_src_

def source_to_target_freq_dft(image_src, image_trg, L=0.1, ratio=None, range_min=0, range_max=1):
    dft = DFT()
    fft_src, fft_trg = dft(image_src), dft(image_trg)
    amp_src, pha_src = dft.fft_2_amp_pha(fft_src)
    amp_trg, _ = dft.fft_2_amp_pha(fft_trg)
    amp_mod = low_freq_mutate_dft(amp_src, amp_trg, L=L, ratio=ratio)
    real_mod, imag_mod = amp_mod * torch.cos(pha_src), amp_mod * torch.sin(pha_src)
    fft_mod = torch.cat([real_mod, imag_mod], dim=1)
    return torch.clip(dft(fft_mod, inverse=True), range_min, range_max)