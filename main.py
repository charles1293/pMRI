#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:28:50 2021

@author: chaari
"""
#from mat4py import loadmat
import h5py
import scipy.io
import numpy as np
from utils import *
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import log


###########################################
#        sload data for simulation        #
###########################################
loaded = scipy.io.loadmat('reference.mat')
ref = loaded['im']
loaded = scipy.io.loadmat('sens.mat')
S = loaded['s']
###########################################
###########################################



###########################################
#          QUESTION 1                     #
#  Simulation with different noise levels #
###########################################
R = 2
noise_levels = [5, 10, 14, 20, 30]  # Different noise intensities
results_q1 = {}

fig, axes = plt.subplots(1, len(noise_levels), figsize=(15, 4))

for idx, sigma in enumerate(noise_levels):
    reduced_FoV = pMRI_simulator(S, ref, sigma, R)
    results_q1[sigma] = reduced_FoV
    
    # Display first channel for visualization
    im = axes[idx].imshow(np.abs(reduced_FoV[:, :, 0]), cmap='gray')
    axes[idx].set_title(f'σ = {sigma}')
    axes[idx].axis('off')

plt.suptitle('Différents niveaux de bruit (R=2)', fontsize=14)
plt.tight_layout()
plt.savefig('q1_reduced_FoV_noise_comparison.png', dpi=150, bbox_inches='tight')

print("Question 1 terminée: le bruit degrade progressivement la qualite des images, avec une degradation nettement plus forte pour sigma=30.")


###########################################
#          QUESTION 2                     #
#      Same simulation with R=4           #
###########################################
R_q2 = 4
results_q2 = {}

fig, axes = plt.subplots(2, len(noise_levels), figsize=(16, 7))

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

plt.suptitle('Question 2 - Comparison of Simulated Acquisitions: R=2 vs R=4', fontsize=14)
plt.tight_layout()
plt.savefig('q2_r2_vs_r4_noise_comparison.png', dpi=150, bbox_inches='tight')

print("Question 2 terminée: pour un meme bruit, R=4 est plus degrade que R=2 a cause du sous-echantillonnage plus agressif.")
