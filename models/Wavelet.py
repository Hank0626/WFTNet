import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

import ptwt
import numpy as np

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


def Wavelet_for_Period(x, scale=16):
    # scales = np.arange(1, 1+scale)
    scales = 2 ** np.linspace(-1, scale, 8)
    # coeffs, freqs = pywt.cwt(x.detach().cpu().numpy(), scales, 'morl')
    coeffs, freqs = ptwt.cwt(x, scales, "morl")
    return coeffs, freqs


class Wavelet(nn.Module):
    def __init__(self, configs):
        super(Wavelet, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.k = configs.top_k
        self.period_coeff = configs.period_coeff
        self.scale = configs.wavelet_scale

        self.period_conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels),
        )

        self.scale_conv = nn.Conv2d(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                kernel_size=(8, 1),
                stride=1,
                padding=(0, 0),
                groups=configs.d_model)
    
        self.projection = nn.Linear(self.seq_len + self.pred_len, self.pred_len, bias=True)

    def forward(self, x):
        # FFT
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.period_conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            
        if len(res) > 0:
            res = torch.stack(res, dim=-1)
            # adaptive aggregation
            period_weight = F.softmax(period_weight, dim=1)
            period_weight = period_weight.unsqueeze(
                1).unsqueeze(1).repeat(1, T, N, 1)

        # Wavelet
        # (B, T, N)
        coeffs = Wavelet_for_Period(x.permute(0, 2, 1), self.scale)[0].permute(1, 2, 0, 3).float()
        # (B, N, S, T)

        wavelet_res = self.period_conv(coeffs)
        # (B, N, S, T)

        wavelet_res = self.scale_conv(wavelet_res).squeeze(2).permute(0, 2, 1)
        # (B, N, T)

        if len(res) > 0:
            
            res = (1 - self.period_coeff ** 10) * wavelet_res + (self.period_coeff ** 10) * torch.sum(res * period_weight, -1)

        else:
            res = wavelet_res
            
            res = res + x

            return self.projection(res.permute(0, 2, 1)).permute(0, 2, 1)
            
        res = res + x

        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.layer = configs.e_layers
        self.wavelet_model = nn.ModuleList([Wavelet(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.wavelet_model[i](enc_out))

        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        if dec_out.shape[1] > self.pred_len:
            dec_out = dec_out * \
                    (stdev[:, 0, :].unsqueeze(1).repeat(
                        1, self.pred_len + self.seq_len, 1))
            dec_out = dec_out + \
                    (means[:, 0, :].unsqueeze(1).repeat(
                        1, self.pred_len + self.seq_len, 1))
        else:
            dec_out = dec_out * \
                    (stdev[:, 0, :].unsqueeze(1).repeat(
                        1, self.pred_len, 1))
            dec_out = dec_out + \
                    (means[:, 0, :].unsqueeze(1).repeat(
                        1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
