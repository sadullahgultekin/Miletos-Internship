import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def generator_loss(fake_scores):
    return torch.mean((fake_scores - 1)**2)/2

def discriminator_loss(real_scores, fake_scores):
    return torch.mean((real_scores - 1)**2)/2 + torch.mean((fake_scores)**2)/2
    
def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        #plt.imshow(img.reshape([sqrtimg,sqrtimg]),cmap="gray")
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))