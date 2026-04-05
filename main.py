#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:28:50 2021

@author: chaari
"""

import h5py
import scipy.io
import numpy as np
from utils import *
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import log


# data pour simu

loaded = scipy.io.loadmat('reference.mat')
ref = loaded['im']
loaded = scipy.io.loadmat('sens.mat')
S = loaded['s']







#QUESTION 1

R = 2
noise_levels = [5, 10, 20, 30]
results_q1 = {}

fig, axes = plt.subplots(1, len(noise_levels), figsize=(15, 4))

for idx, sigma in enumerate(noise_levels):
    reduced_FoV = pMRI_simulator(S, ref, sigma, R)
    results_q1[sigma] = reduced_FoV
    
    im = axes[idx].imshow(np.abs(reduced_FoV[:, :, 0]), cmap='gray')
    axes[idx].set_title(f'σ = {sigma}')
    axes[idx].axis('off')

plt.suptitle('Différents niveaux de bruit (R=2)', fontsize=14)
plt.tight_layout()
plt.savefig('q1_reduced_FoV_noise_comparison.png', dpi=150, bbox_inches='tight')








#QUESTION 2

R_q2 = 4
results_q2 = {}

fig, axes = plt.subplots(2, len(noise_levels), figsize=(18, 8.5))

for idx, sigma in enumerate(noise_levels):
     reduced_FoV_r4 = pMRI_simulator(S, ref, sigma, R_q2)
     results_q2[sigma] = reduced_FoV_r4

     # Top row: R=2 (from question 1)
     im_r2 = axes[0, idx].imshow(np.abs(results_q1[sigma][:, :, 0]), cmap='gray')
     axes[0, idx].set_title(f'R=2, σ={sigma}')
     axes[0, idx].axis('off')

     # Bottom row: R=4
     im_r4 = axes[1, idx].imshow(np.abs(reduced_FoV_r4[:, :, 0]), cmap='gray')
     axes[1, idx].set_title(f'R=4, σ={sigma}')
     axes[1, idx].axis('off')

plt.suptitle('Comparaison R=2 vs R=4', fontsize=14)
plt.tight_layout()
plt.savefig('q2_r2_vs_r4_noise_comparison.png', dpi=150, bbox_inches='tight')






# QUESTION 3

Nc = S.shape[2]
snr_r2 = []
snr_r4 = []

fig, axes = plt.subplots(2, len(noise_levels), figsize=(16, 7))

for idx, sigma in enumerate(noise_levels):
    psi = (sigma ** 2) * np.eye(Nc)

    reconstructed_r2 = reconstruct(results_q1[sigma], S, psi)
    reconstructed_r4 = reconstruct(results_q2[sigma], S, psi)

    snr2 = SignalToNoiseRatio(ref, np.abs(reconstructed_r2))
    snr4 = SignalToNoiseRatio(ref, np.abs(reconstructed_r4))
    snr_r2.append(snr2)
    snr_r4.append(snr4)

    axes[0, idx].imshow(np.abs(reconstructed_r2), cmap='gray')
    axes[0, idx].set_title(f'R=2, σ={sigma}\nSNR={snr2:.2f} dB', fontsize=11, pad=6)
    axes[0, idx].axis('off')

    axes[1, idx].imshow(np.abs(reconstructed_r4), cmap='gray')
    axes[1, idx].set_title(f'R=4, σ={sigma}\nSNR={snr4:.2f} dB', fontsize=11, pad=6)
    axes[1, idx].axis('off')

plt.suptitle('Reconstructions et SNR (R=2 vs R=4)', fontsize=14, y=0.97)
fig.subplots_adjust(top=0.90, hspace=0.18, wspace=0.05)
plt.savefig('q3_reconstruction_r2_vs_r4.png', dpi=150)

plt.figure(figsize=(7, 4))
plt.plot(noise_levels, snr_r2, marker='o', label='R=2')
plt.plot(noise_levels, snr_r4, marker='s', label='R=4')
plt.xlabel('Niveau de bruit σ')
plt.ylabel('SNR (dB)')
plt.title('Question 3 - Comparaison du SNR')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('q3_snr_comparison.png', dpi=150, bbox_inches='tight')

