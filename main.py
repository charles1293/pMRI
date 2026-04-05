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


# QUESTION 4
# Regularisation de Tikhonov (sans detail de derivation)
sigma_q4 = 10
lam_values = [0.01, 0.1, 1, 10]
psi_q4 = (sigma_q4 ** 2) * np.eye(Nc)

baseline_r2 = reconstruct(results_q1[sigma_q4], S, psi_q4)
baseline_r4 = reconstruct(results_q2[sigma_q4], S, psi_q4)

snr_baseline_r2 = SignalToNoiseRatio(ref, np.abs(baseline_r2))
snr_baseline_r4 = SignalToNoiseRatio(ref, np.abs(baseline_r4))

snr_lam_r2 = []
snr_lam_r4 = []
recon_lam_r2 = []
recon_lam_r4 = []

for lam in lam_values:
    rec_r2_lam = reconstruct_tikhonov(results_q1[sigma_q4], S, psi_q4, lam)
    rec_r4_lam = reconstruct_tikhonov(results_q2[sigma_q4], S, psi_q4, lam)

    recon_lam_r2.append(rec_r2_lam)
    recon_lam_r4.append(rec_r4_lam)
    snr_lam_r2.append(SignalToNoiseRatio(ref, np.abs(rec_r2_lam)))
    snr_lam_r4.append(SignalToNoiseRatio(ref, np.abs(rec_r4_lam)))

cols = 2 + len(lam_values)
fig, axes = plt.subplots(2, cols, figsize=(3.2 * cols, 8.2))

axes[0, 0].imshow(np.abs(ref), cmap='gray')
axes[0, 0].set_title('Reference')
axes[0, 0].axis('off')

axes[1, 0].imshow(np.abs(ref), cmap='gray')
axes[1, 0].set_title('Reference')
axes[1, 0].axis('off')

axes[0, 1].imshow(np.abs(baseline_r2), cmap='gray')
axes[0, 1].set_title(f'R=2, sans reg\nSNR={snr_baseline_r2:.2f} dB', fontsize=10)
axes[0, 1].axis('off')

axes[1, 1].imshow(np.abs(baseline_r4), cmap='gray')
axes[1, 1].set_title(f'R=4, sans reg\nSNR={snr_baseline_r4:.2f} dB', fontsize=10)
axes[1, 1].axis('off')

for i, lam in enumerate(lam_values):
    axes[0, i + 2].imshow(np.abs(recon_lam_r2[i]), cmap='gray')
    axes[0, i + 2].set_title(f'R=2, λ={lam}\nSNR={snr_lam_r2[i]:.2f} dB', fontsize=10)
    axes[0, i + 2].axis('off')

    axes[1, i + 2].imshow(np.abs(recon_lam_r4[i]), cmap='gray')
    axes[1, i + 2].set_title(f'R=4, λ={lam}\nSNR={snr_lam_r4[i]:.2f} dB', fontsize=10)
    axes[1, i + 2].axis('off')

fig.suptitle('Tikhonov, σ=10: comparaison sans/avec regularisation', fontsize=14, y=0.98)
fig.subplots_adjust(top=0.90, hspace=0.24, wspace=0.08)
plt.savefig('q4_tikhonov_reconstructions.png', dpi=150)

plt.figure(figsize=(7.2, 4.2))
plt.plot([0] + lam_values, [snr_baseline_r2] + snr_lam_r2, marker='o', label='R=2')
plt.plot([0] + lam_values, [snr_baseline_r4] + snr_lam_r4, marker='s', label='R=4')
plt.xscale('symlog', linthresh=0.01)
plt.xticks([0, 0.01, 0.1, 1, 10], ['0 (sans reg)', '0.01', '0.1', '1', '10'])
plt.xlabel('Valeur de λ')
plt.ylabel('SNR (dB)')
plt.title('Effet de λ sur le SNR (σ=10)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('q4_tikhonov_snr_vs_lambda.png', dpi=150, bbox_inches='tight')

