import argparse
import logging
import sys
import time
import os
import json
import copy

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

from network import Network, ResNetwork


def show_images(ims, gt_labels, pred_labels=None, lookup_names = None, save_path = None):
    fig, ax = plt.subplots(1, len(ims), figsize=(10,5))
    for id in range(len(ims)):
        ax[id].imshow(ims[id])
        ax[id].axis('off')

        if lookup_names is None:
            if pred_labels is None:
                im_title = f'GT: {gt_labels[id]}'
            else:
                im_title = f'GT: {gt_labels[id]}   Pred: {pred_labels[id]}'
            ax[id].set_title(im_title)

        else:
            if pred_labels is None:
                im_title = f'GT: {lookup_names[gt_labels[id]]}'
            else:
                im_title = f'GT: {lookup_names[gt_labels[id]]}   Pred: {lookup_names[pred_labels[id]]}'
            ax[id].set_title(im_title)

    # plt.show()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)

def plot_data(file_path: Path, savePath):

    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    validation_accuracy, validation_loss, training_accuracy, training_loss = data['validation_accuracy'], data['validation_loss'], data['training_accuracy'], data['training_loss']
    epochs = range(1, len(validation_accuracy) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, training_loss, label="Training")
    ax1.plot(epochs, validation_loss, label="Validation")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss [-]")
    ax1.grid("on")
    ax1.legend()

    ax2.plot(epochs, training_accuracy, label="Training")
    ax2.plot(epochs, validation_accuracy, label="Validation")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy [%]")
    ax2.grid("on")
    ax2.legend()

    plt.tight_layout()

    plt.savefig(savePath, dpi=600,  bbox_inches='tight')
    plt.show()


if __name__=="__main__":
    plt.ion()

    #############################################
    # Visualize Random Images and Transformations

    # # Define Transforms
    # train_transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    # transforms.RandomResizedCrop(size=224, scale=(0.8, 0.8)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    # transforms.ToTensor(), 
    # # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    # no_transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    # transforms.ToTensor(), 
    # # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    # # Load dataset
    # image_path = '2023_ELEC5307_P2Train/'
    # dataset_tf = ImageFolder(root=image_path, transform=train_transform)
    # dataset_og = ImageFolder(root=image_path, transform=no_transform)

    # lookup_names = dataset_og.classes

    # # Get images to show
    # n_ims = 6
    # random_integers = random.sample(range(0, len(dataset_og) + 1), n_ims)
    # og_ims = []
    # og_labels = []
    # aug_ims = []
    # aug_labels = []
    # for i in random_integers:
    #     img = dataset_og[i][0].permute(1, 2, 0)
    #     label = dataset_og[i][1]
    #     og_ims.append(img)
    #     og_labels.append(label)

    #     img = dataset_tf[i][0].permute(1, 2, 0)
    #     label = dataset_tf[i][1]
    #     aug_ims.append(img)
    #     aug_labels.append(label)

    # show_images(ims=og_ims,gt_labels=og_labels,lookup_names=lookup_names, save_path='original_images.png')
    # show_images(ims=aug_ims,gt_labels=aug_labels,lookup_names=lookup_names, save_path='modified_images.png')

    # plt.show(block=True)

    #############################################
    # Visualize Training Loss

    dino_training_path = 'training_diff_norm/dinov2_training_prog.json'
    dino_savePath = 'results/dinov2_training_prog.png'
    with open(dino_training_path, 'r') as json_file:
        dino_training_data = json.load(json_file)

    resnet_training_path = 'training_diff_norm/resnet_training_prog.json'
    resnet_savePath = 'results/resnet_training_prog.png'
    with open(dino_training_path, 'r') as json_file:
        resnet_training_data = json.load(json_file)

    plot_data(dino_training_path, dino_savePath)
    plot_data(resnet_training_path,resnet_savePath)

    

    


