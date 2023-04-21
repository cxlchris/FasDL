from calendar import c
import boto3
import urllib
import json
import pickle
import numpy as np
import cv2
import os
import zipfile
import time
import shutil

def handler(event, context):
    del_file('/tmp') # clear /tmp
    
    s3_client = boto3.client('s3')
    
    # setting of k_reduce
    train_net = event['train_net']                  # training model (e.g.'resnet18' or 'resnet101')
    pattern_k = int(event['pattern_k'])             # num of aggregators
    num_workers = int(event['num_workers'])         # num of total workers
    batch_size_wor = int(event['batch_size_wor'])   # batch size of non-aggregators 
    batch_size_agg = int(event['batch_size_agg'])   # batch size of aggregators
    epochs = int(event['epochs'])                   # num of epochs
    l_r= float(event['l_r'])                        # learning rate
    memory = event['memory']                        # memory of function
    func_time = int(event['func_time'])             # maximum run-time of function (second)
    seed = event['seed']                            # random seed   
    tmp_bucket = event['tmp_bucket']                # bucket storing tmp file
    merged_bucket = event['merged_bucket']          # bucket storing merged file
    partition_bucket = event['partition_bucket']    # bucket storing partitioned dataset
    data_bucket = event['data_bucket']              # bucket storing dataset
    output_bucket = event['output_bucket']          # bucket storing log file
    
    # clear the bucket
    clear_bucket(s3_client, tmp_bucket)
    clear_bucket(s3_client, merged_bucket)
    clear_bucket(s3_client, partition_bucket)

    data_partition(s3_client, num_workers, pattern_k, batch_size_agg, batch_size_wor, data_bucket, partition_bucket)

    # lambda payload
    payload = dict()
    payload['train_net'] = train_net
    payload['pattern_k'] = pattern_k
    payload['num_workers'] = num_workers
    payload['batch_size_wor'] = batch_size_wor
    payload['batch_size_agg'] = batch_size_agg
    payload['epochs'] = epochs
    payload['l_r'] = l_r
    payload['memory'] = memory
    payload['seed'] = seed
    payload['invoke_round'] = 1
    payload['func_time'] = func_time
    payload['tmp_bucket'] = tmp_bucket
    payload['merged_bucket'] = merged_bucket
    payload['partition_bucket'] = partition_bucket
    payload['output_bucket'] = output_bucket

    # invoke functions
    lambda_client = boto3.client('lambda')
    for i in range(num_workers):
        payload['worker_index'] = i
        lambda_client.invoke(FunctionName='FasDL_worker',
                             InvocationType='Event',
                             Payload=json.dumps(payload))

    # synchronization at beginning
    file = open('/tmp/start.txt','w',encoding='utf-8')
    file.close()
    flag = True
    while flag:
        lists = s3_client.list_objects_v2(Bucket=partition_bucket, Prefix = 'ready')
        if lists['KeyCount'] == num_workers:
            flag = False
    # clear_bucket(s3_client, "partition_bucket")
    s3_client.upload_file('/tmp/start.txt', partition_bucket, 'start')


def clear_bucket(s3_client, bucket_name):
    # clear bucket
    lists = s3_client.list_objects_v2(Bucket=bucket_name)
    if lists['KeyCount'] > 0:
        objects = lists['Contents']
    else:
        objects = None
    if objects is not None:
        obj_names = []
        for obj in objects:
            file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
            obj_names.append(file_key)
        if len(obj_names) >= 1:
            obj_list = [{'Key': obj} for obj in obj_names]
            s3_client.delete_objects(Bucket=bucket_name, Delete={
                                     'Objects': obj_list})
    return True


def zip_file(zip_name, src_dir):
    # zip file
    z = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(src_dir):
        fpath = dirpath.replace(src_dir, '')
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename), fpath + filename)
    z.close()
    return True


def del_file(filepath):
    # clear filepath
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def data_partition(s3_client, num_workers, pattern_k, batch_size_agg, batch_size_wor, data_bucket, partition_bucket):
    # unequal data partition
    num_samples = 50000 # num of samples
    num_classes = 10 # num of classification labels
    pics_per_class = num_samples / num_classes # num of samples each class

    # create folder for each data partition
    path_prefix = '/tmp/partition-worker'
    for i in range(num_workers):
        if not os.path.exists(path_prefix + str(i)):
            os.mkdir(path_prefix + str(i))
            for j in range(num_classes):
                os.mkdir(path_prefix +
                        str(i) + '/class' + str(j))

    order = [0] * num_classes # indicate current order of pics in corresponding class
    file = [0] * num_classes # indicate current worker_index of corresponding class 
    global_bs = (batch_size_agg  * pattern_k + batch_size_wor  * (num_workers - pattern_k)) # global batch size
    num_pics_agg = pics_per_class / global_bs * batch_size_agg  // 1 # num of samples each class for each aggregator
    num_pics_wor = pics_per_class / global_bs * batch_size_wor  // 1 # num of samples each class for each non-aggregator
    
    # for each batch file of cifar10
    for batch in range(1, 6):
        # download from data_bucket
        data_path = "/tmp/data_batch_"+str(batch)
        s3_client.download_file(
            data_bucket, "data_batch_"+str(batch), data_path)
        
        with open(data_path, 'rb') as fo:
            data_batch = pickle.load(fo, encoding='bytes')
        cifar_label = data_batch[b'labels']
        cifar_data = data_batch[b'data']
        # transform dict into array
        cifar_label = np.array(cifar_label)
        cifar_data = np.array(cifar_data)
        
        # for each image
        for i in range(10000):
            # discard excess samples of each class
            if file[cifar_label[i]] == num_workers:
                continue
            image = cifar_data[i]
            image = image.reshape(-1, 1024)
            r = image[0, :].reshape(32, 32)  # red
            g = image[1, :].reshape(32, 32)  # green
            b = image[2, :].reshape(32, 32)  # blue
            img = np.zeros((32, 32, 3))
            # restore RGB to image
            img[:, :, 0] = r
            img[:, :, 1] = g
            img[:, :, 2] = b
            # classify image into the corresponding folder
            cv2.imwrite(path_prefix + str(file[cifar_label[i]]) + "/class" + str(cifar_label[i]) + "/" +
                        str(order[cifar_label[i]]) + ".jpg", img)
            
            order[cifar_label[i]] += 1
            if file[cifar_label[i]] < pattern_k and order[cifar_label[i]] == num_pics_agg or \
                file[cifar_label[i]] >= pattern_k and order[cifar_label[i]] == num_pics_wor:
                order[cifar_label[i]] = 0
                file[cifar_label[i]] += 1
    
    # zip the folder and upload to partition_bucket
    for i in range(num_workers):
        zip_name = path_prefix + str(i) + '.zip'
        src_dir = path_prefix + str(i)
        zip_file(zip_name, src_dir)
        key = "partition-worker" + str(i) + ".zip"
        s3_client.upload_file(zip_name, partition_bucket, key)