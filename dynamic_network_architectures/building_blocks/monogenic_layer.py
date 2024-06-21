import torch
import torch.nn as nn

from torch.nn.modules import Module
torch.autograd.set_detect_anomaly(True)


class Monogenic(nn.Module):
    def __init__(self, sigmaonf=0.4, min_wl=15, nscale=3, mult=1.75, requires_grad=False):
        super(Monogenic, self).__init__()
        self.sigmaonf = nn.Parameter(torch.tensor(sigmaonf, dtype=torch.float), requires_grad=requires_grad)
        self.nscale = nn.Parameter(torch.tensor(nscale, dtype=torch.int), requires_grad=requires_grad)
        self.mult = nn.Parameter(torch.tensor(mult, dtype=torch.float), requires_grad=requires_grad)
        self.min_wl = nn.Parameter(torch.tensor(min_wl, dtype=torch.float), requires_grad=requires_grad)
        self.wls = nn.Parameter(
            torch.tensor([[min_wl * (mult ** i)] for i in range(nscale)], dtype=torch.float),
            requires_grad=requires_grad
        )

    def forward(self, x):
        batch_size, in_channels, rows, cols = x.shape
        filters = self.createMonogenicFilters(rows, cols)
        x = self.monogenic_signal(inputs=x, filters=filters)
        return x.view(batch_size, -1, rows, cols)

    def local_energy(self, x):
        return torch.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    
    def local_phase(self, x):
        return torch.atan2(torch.sqrt(x[1]**2 + x[2]**2), x[0])
    
    def local_orientation(self, x):
        return torch.atan2(-x[1],x[2])
    
    def monogenic_signal(self, inputs, filters):
        F = torch.fft.fft2(inputs)
        Ffilt = F * filters[0]
        # Compute the parts of the monongenic signal
        Fm1 = torch.fft.ifft2(Ffilt).real
        Fmodd = torch.fft.ifft2(Ffilt * filters[1])
        Fm2 = Fmodd.real
        Fm3 = Fmodd.imag
        mono_signal = (Fm1, Fm2, Fm3)
        # le = self.local_energy(Fm1)
        lp = self.local_phase(mono_signal)
        lo = self.local_orientation(mono_signal)
        return torch.stack((lp, lo), dim=1)

    def createMonogenicFilters(self, rows, cols):
        # Frequency grid for the filter
        y_grid, x_grid = self.freqgrid2(rows, cols)
        # Determine the spatial regions to use
        w = torch.sqrt(x_grid ** 2 + y_grid ** 2).unsqueeze(0)
        w[:, 0, 0] = 1.0
        # print("w: ", w.shape)
        wls = self.wls.view(-1, 1, 1)
        # Create bp_filters in frequency domain
        w0 = 1.0/wls
        bp_filters0 = torch.exp((-(torch.log(w/w0)) ** 2) / (2 * torch.log(self.sigmaonf) ** 2))
        # bp_filters0 = bp_filters0 / (2 * torch.log(sigmaonf/w0) ** 2)
        # bp_filters0 = torch.exp(bp_filters0)
        # Set the DC value of the filter to 0
        bp_filters = bp_filters0.clone()
        bp_filters[:, 0, 0] = 0.0
        # Remove unwanted high frequency components in bp_filters
        if rows % 2 == 0:
            bp_filters[:, rows//2, :] = 0.0
        if cols % 2 == 0:
            bp_filters[:, :, cols//2] = 0.0
        # # Normalize by the maximum value of the sum of all the bp_filters
        bp_filters = bp_filters / torch.max(torch.sum(bp_filters, dim=(0)))

        # Create the Riesz filters
        reisz_filters = (1j * y_grid - x_grid) / w

        return (bp_filters, reisz_filters)

    def freqgrid2(self, rows, cols):
        y_grid = torch.fft.fftfreq(rows, device=self.sigmaonf.device)  # Ensure grid is on same device as the parameters
        x_grid = torch.fft.fftfreq(cols, device=self.sigmaonf.device)
        return torch.meshgrid(y_grid, x_grid, indexing='ij')