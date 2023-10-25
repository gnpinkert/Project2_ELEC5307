'''
this script is for the evaluation of Project 2.

-------------------------------------------
INTRO:
You are allow to change this code, but you need to make sure that we can run your code with the trained .pth to calculate your test accuracy.
For most of parts this code, you do not need to change.

-------------------------------------------
USAGE:
In your final update, please keep the file name as 'python2_test.py'.

>> python project2_test.py
This will run the program on CPU to test on your trained nets for the Fruit test dataset

>> python project2_test.py --cuda
This will run the program on GPU to test on your trained nets for the Fruit test dataset
You can ignore this if you do not have GPU or CUDA installed.

-------------------------------------------
NOTE:
this file might be incomplete, feel free to contact us
if you found any bugs or any stuff should be improved.
Thanks :)

Email: txue4133@uni.sydney.edu.au, weiyu.ju@sydney.edu.au
'''

# import the packages
import argparse
import logging
import sys
import time
import os
import json

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from network import Network, ResNetwork # the network you used
from project2_train import getFilePaths

# ==================================
# control input options. DO NOT CHANGE THIS PART, unless you are sure that we can run your testing process correctly.
def parse_args():
    parser = argparse.ArgumentParser(description= \
        'scipt for evaluation of project 2')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='Used when there are cuda installed.')
    parser.add_argument('--output_path', default='./', type=str,
        help='The path that stores the log files.')

    pargs = parser.parse_args()
    return pargs

# Creat logs. DO NOT CHANGE THIS PART, unless you are sure that we can run your testing process correctly.
def create_logger(final_output_path):
    log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file),
                        format=head)
    clogger = logging.getLogger()
    clogger.setLevel(logging.INFO)
    # add handler
    # print to stdout and log file
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    clogger.addHandler(ch)
    return clogger


# evaluation process. DO NOT CHANGE THIS PART, unless you are sure that we can run your testing process correctly.
def eval_net(net, loader, logging, model_name = 'project2_modified.pth'):
    net = net.eval()
    if args.cuda:
        net = net.cuda()

    if args.cuda:
        net.load_state_dict(torch.load(model_name, map_location='cuda'))
    else:
        net.load_state_dict(torch.load(model_name, map_location='cpu'))

    correct = 0
    total = 0
    for data in loader:
        images, labels = data
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        if args.cuda:
            outputs = outputs.cpu()
            labels = labels.cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # print and write to log. DO NOT CHANGE HERE.
    logging.info('=' * 55)
    logging.info('SUMMARY of Project2')
    logger.info('The number of testing image is {}'.format(total))
    logging.info('Accuracy of the network on the test images: {} %'.format(100 * round(correct / total, 4)))
    logging.info('=' * 55)

# Prepare for writing logs and setting GPU. 
# DO NOT CHANGE THIS PART, unless you are sure that we can run your testing process correctly.
args = parse_args()
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
# print('using args:\n', args)

logger = create_logger(args.output_path)
logger.info('using args:')
logger.info(args)

# DO NOT change codes above this line, unless you are sure that we can run your testing process correctly.
# ==================================


####################################
# Transformation definition
# NOTE:
# Write the test_transform here. Please do not use
# Random operations, which might make your performance worse.
# Remember to make the normalize value same as in the training transformation.

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

####################################

def measure_inference(model, testdata, logging):
    testloader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False, num_workers=2)
    total_time = 0
    for data in testloader:
        start = time.process_time()
        image, label = data
        outputs = model(image)
        end = time.process_time()
        total_time += end - start
    logging.info(f"Total average inference time is: {total_time / len(testloader)}")


# https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
def getConfusionMatrix(net, testloader, logging, name):
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in testloader:
            output = net(inputs) # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

    # constant for classes
    classes = testloader.dataset.classes

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,8))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(name, bbox_inches = 'tight', dpi=600)

    logging.info('Saved Confusion Matrix as {name}')


####################################
# Define the test dataset and dataloader.
# You can make some modifications, e.g. batch_size, adding other hyperparameters, etc.
if __name__ == "__main__":
    # !! PLEASE KEEP test_image_path as '../test' WHEN YOU SUBMIT.
    test_image_path = '2023_ELEC5307_P2Train'

    # Specify the file path where the data is saved
    file_path = "training_diff_norm/dataset_split.json"

    # Weights name
    model_name = 'training_diff_norm/dinov2_weights.pth'

    # Name of saved image for confusion matrix
    conf_name = 'dinov2_confusion_matrix.png'

    # Read the data from the JSON file
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # Access the lists by their keys
    test_files = data["test_files"]
    prefix_to_remove = "/kaggle/input/fruits/"
    cleaned_test_files = [path.replace(prefix_to_remove, "") for path in test_files]


    testset = ImageFolder(root=test_image_path, transform=test_transform, is_valid_file=lambda file_name : file_name in cleaned_test_files)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=True, num_workers=2)

    ####################################

    # ==================================
    # test the network and write to logs.
    # use cuda if called with '--cuda'.
    # DO NOT CHANGE THIS PART, unless you are sure that we can run your testing process correctly.

    # network = Network()
    network = Network()

    if args.cuda:
        network.load_state_dict(torch.load(model_name, map_location='cuda'))
    else:
        network.load_state_dict(torch.load(model_name, map_location='cpu'))

    # Get overall accuracy
    measure_inference(network, testset, logging)
    if args.cuda:
        network = network.cuda()

    # test your trained network
    eval_net(network, testloader, logging, model_name=model_name)

    # Get Confusion Matrix
    getConfusionMatrix(net=network,testloader=testloader, logging=logging, name=conf_name)
    # ==================================