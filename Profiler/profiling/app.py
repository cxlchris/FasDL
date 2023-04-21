import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
import random
import zipfile
import time
import boto3
import torchvision.datasets as dset
import torchvision.transforms as transforms

def handler(event, context):
    # input parameters
    train_net = event['train_net'] # training model (e.g.'resnet18' or 'resnet101')
    batch_size = int(event['batch_size']) # batch size 
    memory = int(event['memory']) # memory of function            
    seed = int(event['seed']) # random seed
    data_bucket = event['data_bucket'] # bucket storing dataset
    output_bucket = event['output_bucket'] # bucket stroing output info

    setup_seed(seed) # set random seed
    
    s3_client = boto3.client('s3') # client of S3

    # download dataset for profiling from data_bucket and unzip
    file_name = "profiling.zip"
    zip_file = '/tmp/profiling.zip'
    dst_dir = "/tmp/profiling"
    s3_client.download_file(data_bucket, file_name, zip_file)
    unzip_file(zip_file, dst_dir)

    # download recording file from output_bucket
    file_name = "profiling.txt"
    output_file = "/tmp/profiling.txt"
    s3_client.download_file(output_bucket, file_name, output_file)

    # training model
    if train_net == 'resnet18':
        net = models.resnet18()
    elif train_net == 'resnet101':
        net = models.resnet101()
    
    # load data
    trainloader = load_data(batch_size, dst_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    time_start = time.time()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # training 1 iteration
        break
        
    # training time of 1 iteration
    train_time = time.time()-time_start

    time_start = time.time()
    # transform the model parameters into 1-d array 
    weights = [param.data.numpy() for param in net.parameters()]
    vector = weights[0].reshape(1,-1)
    for i in range(1,len(weights)):
        vector = np.append(vector, weights[i].reshape(1,-1))
    # transform time
    trans_time = time.time()-time_start

    # upload recording file back to output_bucket
    f = open(output_file, 'a', encoding='utf-8')
    f.write(f"{batch_size} {memory} {train_time} {trans_time}\n")
    f.close()
    s3_client.upload_file(output_file, output_bucket, file_name)

    return {"result": "succeed!"}


def load_data(batch_size, dst_dir):
    # load data 
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = dset.ImageFolder(dst_dir, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    return trainloader


def setup_seed(seed):
    # set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def unzip_file(zip_file, dst_dir):
    # unzip file
    fz = zipfile.ZipFile(zip_file, 'r')
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    fz.close()
    return True



