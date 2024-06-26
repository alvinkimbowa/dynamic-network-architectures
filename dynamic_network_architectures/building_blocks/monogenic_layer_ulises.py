#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = ["Abraham Sanchez", "E. Ulises Moya", "Guillermo Mendoza"]
__copyright__ = "Copyright 2021, Gobierno de Jalisco, Universidad Autonoma de Guadalajara"
__credits__ = ["Abraham Sanchez", "E. Ulises Moya"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = ["Abraham Sanchez", "E. Ulises Moya", "Guillermo Mendoza"]
__email__ = "abraham.sanchez@jalisco.gob.mx"
__status__ = "Development"


import torch
import torch.nn as nn

from torch.nn.modules import Module


class Monogenic(Module):
    def __init__(self, nscale: int = None, sigma: float = None, wave_lengths=None, return_rgb: bool = None, return_hsv: bool = None, return_phase_orientation: bool = None):
        super(Monogenic, self).__init__()
        self.nscale = nscale
        if self.nscale is None:
            self.nscale = 3

        # if sigma is not None:
        #     self.sigma = nn.Parameter(data=torch.as_tensor(sigma, dtype=torch.float))
        # else:
        #     # random initialization
        #     self.sigma = nn.Parameter(data=torch.rand(1))

        self.sigma = nn.Parameter(data=torch.tensor(0.4, dtype=torch.float), requires_grad=False)

        # if wave_lengths is not None:
        #     self.wave_lengths = nn.Parameter(data=torch.as_tensor(wave_lengths, dtype=torch.float))
        # else:
        #     # random initialization
        #     self.wave_lengths = nn.Parameter(data=torch.randint(3, 25, (self.nscale, 1), dtype=torch.float))
        mult = 1.75
        min_wl = 10
        self.wave_lengths = nn.Parameter(
            torch.tensor([[min_wl * (mult ** i)] for i in range(self.nscale-1)], dtype=torch.float),
            requires_grad=False
        )

        self.return_hsv = return_hsv
        self.return_rgb = return_rgb
        self.return_phase_orientation = return_phase_orientation

    def forward(self, inputs):
        x = torch.mean(inputs, dim=1, keepdim=True)
        batch, channels, cols, rows = x.shape
        monogenic = self.monogenic_scale(cols=cols, rows=rows, central_frequency=self.wave_lengths, sigma=self.sigma)
        output = self.compute_monogenic(inputs=x, monogenic=monogenic)
        return output.view(batch, -1, cols, rows)

    def compute_monogenic(self, inputs, monogenic):
        im = torch.fft.fft2(inputs)
        imf = im * monogenic[0, ...]
        imh1 = im * monogenic[1, ...]
        imh2 = im * monogenic[2, ...]
        f = torch.fft.ifft2(imf).real
        h1 = torch.fft.ifft2(imh1).real
        h2 = torch.fft.ifft2(imh2).real
        ori = torch.atan(torch.divide(-h2, h1 + 1e-6))
        fr = torch.sqrt(h1 ** 2 + h2 ** 2) + 1e-6
        ft = torch.atan2(f, fr)
        fts = self.scale_max_min(ft)
        oris = self.scale_max_min(ori)
        frs = self.scale_max_min(fr)
        ones = torch.ones_like(fts)


        hsv_tensor_v = torch.stack((fts, frs, ones), dim=2)
        batch, wls, channels, cols, rows = hsv_tensor_v.shape
        rgb_tensor_v = self.hsv_to_rgb(tensor=hsv_tensor_v.view(-1, channels, cols, rows), shape=(batch * wls, cols, rows))
        hsv_tensor_o = torch.stack((oris, frs, ones), dim=2)
        batch, wls, channels, cols, rows = hsv_tensor_v.shape
        rgb_tensor_o = self.hsv_to_rgb(tensor=hsv_tensor_o.view(-1, channels, cols, rows), shape=(batch * wls, cols, rows))

        if self.return_rgb == True:
            return torch.cat([rgb_tensor_o, rgb_tensor_v], dim=1)
        elif self.return_hsv == True:
            return torch.stack([hsv_tensor_v, hsv_tensor_o], dim=1)
        elif self.return_phase_orientation == True:
            return torch.stack([fts, oris], dim=1)
        else:
            return ft

    def hsv_to_rgb(self, tensor, shape):
        device = self.parameters().__next__().device
        
        h = tensor[:, 0, :, :]
        s = tensor[:, 1, :, :]
        v = tensor[:, 2, :, :]
        c = s * v
        m = v - c
        dh = h * 6.
        h_category = torch.as_tensor(dh, dtype=torch.int32, device=device)
        fmodu = dh % 2
        x = c * (1. - torch.abs(fmodu - 1))
        dtype = tensor.dtype
        rr = torch.zeros(shape, dtype=dtype, device=device)
        gg = torch.zeros(shape, dtype=dtype, device=device)
        bb = torch.zeros(shape, dtype=dtype, device=device)
        h0 = torch.eq(h_category, 0)
        rr = torch.where(h0, c, rr)
        gg = torch.where(h0, x, gg)
        h1 = torch.eq(h_category, 1)
        rr = torch.where(h1, x, rr)
        gg = torch.where(h1, c, gg)
        h2 = torch.eq(h_category, 2)
        gg = torch.where(h2, c, gg)
        bb = torch.where(h2, x, bb)
        h3 = torch.eq(h_category, 3)
        gg = torch.where(h3, x, gg)
        bb = torch.where(h3, c, bb)
        h4 = torch.eq(h_category, 4)
        rr = torch.where(h4, x, rr)
        bb = torch.where(h4, c, bb)
        h5 = torch.eq(h_category, 5)
        rr = torch.where(h5, c, rr)
        bb = torch.where(h5, x, bb)
        r = rr + m
        g = gg + m
        b = bb + m
        return torch.stack([r, g, b], dim=1)

    @classmethod
    def scale_max_min(cls, x):
        x_min = torch.amin(x, dim=(2, 3), keepdim=True)
        x_max = torch.amax(x, dim=(2, 3), keepdim=True)
        scale = (x - x_min) / (x_max - x_min)
        return scale

    def meshs(self, size):
        x, y = self.mesh_range(size)
        radius = torch.sqrt(x * x + y * y).type(torch.float)
        return x, y, radius

    def low_pass_filter(self, size, cutoff, n):
        x, y = self.mesh_range(size)
        radius = torch.sqrt(x * x + y * y)
        lpf = (1. / (1. + (radius / cutoff) ** (2. * n))).type(torch.float)
        return lpf

    def riesz_trans(self, cols, rows):
        u1, u2, qs = self.meshs((rows, cols))
        qs = torch.sqrt(u1 * u1 + u2 * u2)
        qs[0, 0] = 1.
        h1 = (1j * u1) / qs
        h2 = (1j * u2) / qs
        return h1, h2

    def log_gabor_scale(self, cols, rows, wl, c):
        u1, u2, radius = self.meshs((rows, cols))
        radius[0, 0] = 1.
        lp = self.low_pass_filter((rows, cols), .45, 15.)
        log_gabor_denominator = (2. * torch.log(c) ** 2.).type(torch.float)
        fo = 1. / wl.view(-1, 1, 1)
        log_rad_over_fo = torch.log(radius / fo)
        log_gabor = torch.exp(-(log_rad_over_fo * log_rad_over_fo) / log_gabor_denominator)
        log_gabor = lp * log_gabor
        return log_gabor

    def monogenic_scale(self, cols, rows, central_frequency, sigma):
        h1, h2 = self.riesz_trans(cols, rows)
        lg = self.log_gabor_scale(cols, rows, central_frequency, sigma)
        lg_h1 = lg * h1
        lg_h2 = lg * h2
        monogenic = torch.stack([lg, lg_h1, lg_h2], dim=0)
        return monogenic

    def mesh_range(self, size):
        device = self.parameters().__next__().device
        cols, rows = size
        y_grid = torch.fft.fftfreq(rows, device=device)  # Ensure grid is on same device as the parameters
        x_grid = torch.fft.fftfreq(cols, device=device)
        return torch.meshgrid(y_grid, x_grid, indexing='ij')