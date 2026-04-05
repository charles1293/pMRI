#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:04:52 2021

@author: chaari
"""

import numpy as np
from numpy import linalg as LA
from math import log

def pMRI_simulator(S,ref,sigma,R):
    Nc = S.shape[2]
    Size = S.shape[0]
    Size_red = round(Size/R)
    delta = round(Size_red/2)
    reduced_FoV = np.zeros((Size_red,Size,Nc))
    for j in range(Nc):
        for m in range(Size_red):
            for n in range(Size):
                indices = []
                for r in range(0,R):
                    indices.append((m+delta+r*Size_red)%Size)
                s = S[indices,n,:].transpose()
                A_des = ref[indices,n]
                noise = np.random.normal(0,sigma,Nc)
                A_obs = np.dot(s,A_des) + noise
                reduced_FoV[m,n,:] = A_obs
    return reduced_FoV





def reconstruct(reduced_FoV,S,psi):
    [Size_red,Size,Nc] = reduced_FoV.shape
    delta = round(Size_red/2)
    reconstructed = np.zeros((Size,Size))
    psi_1 = np.linalg.pinv(psi)
    R = round(Size/Size_red)
    for m in range(Size_red):
        for n in range(Size):
            indices = []
            for r in range(0,R):
                indices.append((m+delta+r*Size_red)%Size)
            s = S[indices,n,:].transpose()
            A = reduced_FoV[m,n,:]
            #reconstructed = ...
    
    return reconstructed
            
            
def SignalToNoiseRatio(x_reference,x):
    """
    Calculate Signal-to-Noise Ratio
    SNR = 20 * log10(||signal||_2 / ||noise||_2)
    """
    noise = x_reference - x
    signal_norm = LA.norm(x_reference.flatten())
    noise_norm = LA.norm(noise.flatten())
    
    if noise_norm > 0:
        snr = 20 * log10(signal_norm / noise_norm)
    else:
        snr = float('inf')
    
    return snr
            
            
            
            
            