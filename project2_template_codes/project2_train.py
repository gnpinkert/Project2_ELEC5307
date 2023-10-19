'''
this script is for the training code of Project 2..

-------------------------------------------
INTRO:
You can change any parts of this code

-------------------------------------------

NOTE:
this file might be incomplete, feel free to contact us
if you found any bugs or any stuff should be improved.
Thanks :)

Email:
txue4133@uni.sydney.edu.au, weiyu.ju@sydney.edu.au
'''

# import the packages
import argparse
import logging
import sys
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from network import Network # the network you used

parser = argparse.ArgumentParser(description= \
                                     'scipt for training of project 2')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Used when there are cuda installed.')
args = parser.parse_args()

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

def train_net(net, trainloader, valloader, logging, criterion, optimizer, scheduler, epochs=1):

    best_state_dictionary = None
    best_validation_accuracy = 0.0
    inertia = 0
    for epoch in range(epochs):  # loop over the dataset multiple times, only 1 time by default

        running_loss = 0.0
        net = net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if args.cuda:
                loss = loss.cpu()

            # print statistics and write to log
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                logging.info('[%d, %5d] Training loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        if type(scheduler).__name__ != 'NoneType':
            scheduler.step()

        running_loss = 0.0
        correct = 0
        net = net.eval()
        for i, data in enumerate(valloader, 0):
            # get the inputs
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            if args.cuda:
                loss = loss.cpu()

            # print statistics and write to log
            running_loss += loss.item()
            if i % 20 == 19:  # print every 2000 mini-batches
                logging.info('[%d, %5d] Validation loss: %.3f' %
                             (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
            correct += (outputs.argmax(1) == labels).sum().item()

        val_accuracy = 100 * correct / len(valset)

        if val_accuracy > best_validation_accuracy:
            best_state_dictionary = net.state_dict()
            inertia = 0
        else:
            inertia += 1
            if inertia == 3:
                if best_state_dictionary is None:
                    raise Exception("State dictionary should have been updated at least once")
                break
        print(f"Validation accuracy: {val_accuracy}")

    # save network
    torch.save(best_state_dictionary, 'project2_modified.pth')
    # write finish to the file
    logging.info('Finished Training')

##############################################

############################################
# Transformation definition
# NOTE:
# Write the train_transform here. We recommend you use
# Normalization, RandomCrop and any other transform you think is useful.

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.8, 0.8)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def getFilePaths(filePath):
    training_files = []
    test_files = []
    validation_files = []
    for root, dirs, _ in os.walk(filePath):
        for dir in dirs:
            for _, _, files in os.walk(root + dir):
                train_test_len = len(files) // 10
                print(train_test_len)
                files_with_path = [f"{root}{dir}/{file}" for file in files]
                validation_files.extend(files_with_path[0:train_test_len])
                test_files.extend(files_with_path[train_test_len:train_test_len * 2])
                training_files.extend(files_with_path[train_test_len * 2:])
    return training_files, test_files, validation_files



####################################
if __name__=="__main__":
    ####################################
    # Define the training dataset and dataloader.
    # You can make some modifications, e.g. batch_size, adding other hyperparameters, etc.


    image_path = '../2023_ELEC5307_P2Train/'

    training_files, test_files, validation_files = getFilePaths(image_path)

    trainset = ImageFolder(root=image_path, transform=train_transform, is_valid_file=lambda file_name : file_name in training_files)
    valset = ImageFolder(root=image_path, transform=train_transform, is_valid_file=lambda file_name : file_name in validation_files)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                             shuffle=True, num_workers=10)
    valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                             shuffle=True, num_workers=10)
    ####################################

    # ==================================
    # use cuda if called with '--cuda'.

    network = Network()
    if args.cuda:
        network = network.cuda()

    # train and eval your trained network
    # you have to define your own
    logger = create_logger('./')
    logger.info('using args:')
    logger.info(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.0004, momentum=0.9)  # adjust optimizer settings

    for param in network.dinov2_vits14.parameters():
        param.requires_grad = False

    train_net(net=network, trainloader=trainloader, valloader=valloader, criterion=criterion, optimizer=optimizer, scheduler=None, epochs=10, logging=logging)

    # print("final validation accuracy:", val_acc)
    # dict((x,valset.targets.count(x)) for x in set(valset.targets))

    # ==================================
