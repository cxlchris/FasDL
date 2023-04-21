import urllib
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import zipfile
import time
import boto3
import shutil
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models

def handler(event, context):
    func_start = time.time() # start time of the function

    del_file('/tmp') # clear /tmp

    # training setting
    train_net = event['train_net']                  # training model (e.g.'resnet18' or 'resnet101')
    pattern_k = int(event['pattern_k'])             # num of aggregators
    worker_index = int(event['worker_index'])       # index of worker
    num_workers = int(event['num_workers'])         # total num of workers
    batch_size_wor = int(event['batch_size_wor'])   # batch size of non-aggregators
    batch_size_agg = int(event['batch_size_agg'])   # batch size of aggregators
    epochs = int(event['epochs'])                   # num of epochs
    l_r = float(event['l_r'])                       # learning rate
    memory = int(event['memory'])                   # memory of function
    seed = int(event['seed'])                       # random seed 
    func_time = int(event['func_time'])             # maximum run-time of function (second)
    invoke_round = int(event['invoke_round'])       # invoke index of function 
    global tmp_bucket
    tmp_bucket = event['tmp_bucket']                # bucket storing tmp file
    global merged_bucket
    merged_bucket = event['merged_bucket']          # bucket storing merged file
    global partition_bucket
    partition_bucket = event['partition_bucket']    # bucket storing partitioned dataset
    output_bucket = event['output_bucket']          # bucket storing log file

    setup_seed(seed) # set random seed

    s3_client = boto3.client('s3') # client of S3

    if worker_index < pattern_k:
        # aggregator
        batch_size = batch_size_agg
    else:
        # non-aggregator
        batch_size = batch_size_wor

    if invoke_round == 1:
        # create log file
        output_file = open('/tmp/output_file.txt', 'w', encoding='utf-8')  
        output_file.write(f"Worker Index:{worker_index}\n")
        if worker_index < pattern_k:
            output_file.write("Role:Aggregator\n")
        else:
            output_file.write("Role:Non-aggregator\n")
    else:
        # download and open the log file
        path = 'N' + str(num_workers) + '_B' + str(batch_size_agg) + '-' + str(batch_size_wor) + '_M' + str(memory) \
            + '_K' + str(pattern_k) + '/'
        file_name = '[' + str(worker_index) + ']N' + str(num_workers) + '_B' + str(batch_size_agg) + '-' + str(batch_size_wor) \
            + '_M' + str(memory) + '_K' + str(pattern_k) + '.txt'
        s3_client.download_file(output_bucket, path + file_name, '/tmp/output_file.txt')
        output_file = open('/tmp/output_file.txt', 'a', encoding='utf-8')

    # download dataset partition from partition_bucket and upzip
    file_name = "partition-worker" + str(worker_index) + ".zip"
    zip_name = "/tmp/traindata.zip"
    dst_dir = "/tmp/traindata"
    s3_client.download_file(partition_bucket, file_name,zip_name)
    unzip_file(zip_name,dst_dir)

    # training model
    if train_net == 'resnet18':
        net = models.resnet18()
    elif train_net == 'resnet101':
        net = models.resnet101()
    
    if invoke_round != 1:
        # if not first invocation, download the model parameters and checkpoint info from the output_bucekt
        file_name = f'net_w{worker_index}'
        s3_client.download_file(output_bucket, path + file_name, '/tmp/net.pth')
        checkpoint = torch.load('/tmp/net.pth')
        net.load_state_dict(checkpoint['model_state_dict'])

    trainloader = load_data(batch_size) # load data
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=l_r)

    if invoke_round == 1:
        # first invocation
        iter = 0 # current iteration
        time_pre = 0 # sum time of pre invocations 
        train_time_pre = 0 # sum training time of pre invocations
        commu_time_pre = 0 # sum communication time of pre invocations
        # synchronization at beginning
        ready_file = open('/tmp/ready.txt','w',encoding='utf-8')
        ready_file.close()
        s3_client.upload_file('/tmp/ready.txt', partition_bucket, 'ready' + str(worker_index))
        flag = True
        while flag:
            lists = s3_client.list_objects_v2(Bucket=partition_bucket, Prefix = 'start')
            if lists['KeyCount'] == 1:
                flag = False
    else:
        # if not the first invocation, get the checkpoint info
        iter = int(checkpoint['iter'])
        time_pre = float(checkpoint['time_pre'])
        train_time_pre = float(checkpoint['train_time_pre'])
        commu_time_pre = float(checkpoint['commu_time_pre'])

    cur_epoch = (iter+1) // len(trainloader) # current epoch
    iter_index = iter % len(trainloader) # interation index in current epoch

    terminate_flag = False # if TURE, terminate the function and invoke a new one due to time constraint
    complete_flag = False # if TURE, the whole training is over

    train_invo_time = 0 # training time in this invocation
    commu_invo_time = 0 # communication time in this invocation

    for epoch in range(epochs)[cur_epoch:]:  
        # loop over the dataset multiple times
        output_file.write(f"\n########Epoch {epoch+1}########\n")
        for i, data in enumerate(trainloader, 0):
            if i < iter_index:
                continue

            iter += 1 
            output_file.write(f"\n  Iteration {iter}\n")
            
            # training in this iteration
            train_iter_start = time.time() # start time of training in this iteration
            inputs, labels = data
            optimizer.zero_grad() # zero the parameter gradients
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_iter_time = time.time() - train_iter_start # training time in this iteration
            output_file.write(f"  Train Time: {train_iter_time:.2f}\n")
            train_invo_time += train_iter_time 

            # communication in this iteration
            commu_iter_start = time.time() # start time of communication in this iteration
            weights = [param.data.numpy() for param in net.parameters()] # get model parameters
            merged_weights = k_reduce(
                s3_client, weights, num_workers, worker_index, pattern_k, output_file, iter) # aggregation with k_reduce
            for layer_index, param in enumerate(net.parameters()):
                    param.data = torch.from_numpy(merged_weights[layer_index]) # update model parameters   
            commu_iter_time = time.time() - commu_iter_start # communication time in this iteration
            output_file.write(f"  Commu Time: {commu_iter_time:.2f}\n")
            commu_invo_time += commu_iter_time

            output_file.write(f"  Total Time: ({iter}): {(train_iter_time+commu_iter_time):.2f}\n")

            if time.time() - func_start > func_time - 60:
                # terminate this invocation when approaching the runtime constraint of function
                terminate_flag = True
                break

        if terminate_flag:
            break    

        output_file.write(f"\n  Epoch {epoch+1} is over\n")

    if epoch == epochs-1 and i == len(trainloader)-1:
        # whole training is over
        complete_flag = True

    train_time_pre += train_invo_time 
    commu_time_pre += commu_invo_time
    time_pre += train_invo_time + commu_invo_time
    if complete_flag:
        # output the total time of training
        output_file.write(f"\nTotal Train Time: {train_time_pre:.1f}\n")
        output_file.write(f"\nTotal Comm Time: {commu_time_pre:.1f}\n")
        output_file.write(f"\nTotal Time: {time_pre:.1f}\n")
    output_file.close()
    
    # upload the log file to corresponding path in output_bucket
    path = 'N' + str(num_workers) + '_B' + str(batch_size_agg) + '-' + str(batch_size_wor) + '_M' + str(memory) \
            + '_K' + str(pattern_k) + '/'
    file_name = '[' + str(worker_index) + ']N' + str(num_workers) + '_B' + str(batch_size_agg) + '-' + str(batch_size_wor) \
            + '_M' + str(memory) + '_K' + str(pattern_k) + '.txt'
    s3_client.upload_file(
        '/tmp/output_file.txt', output_bucket, path + file_name)
    
    # save the model parameters and checkpoint info and upload to the output_bucket
    PATH = '/tmp/net.pth'
    torch.save({
                'model_state_dict': net.state_dict(),
                'iter': iter,
                'time_pre': time_pre,
                'train_time_pre': train_time_pre,
                'commu_time_pre': commu_time_pre
                }, PATH)
    file_name = f'net_w{worker_index}'
    s3_client.upload_file(
        PATH, output_bucket, path + file_name)
    
    if not complete_flag:
        # each worker invokes itself if the training is not over
        # lambda payload
        payload = dict()
        payload['train_net'] = train_net
        payload['pattern_k'] = pattern_k
        payload['num_workers'] = num_workers
        payload['batch_size_wor'] = batch_size_wor
        payload['batch_size_agg'] = batch_size_agg
        payload['epochs'] = epochs
        payload['memory'] = memory
        payload['seed'] = seed
        payload['invoke_round'] = invoke_round+1
        payload['func_time'] = func_time
        payload['worker_index'] = worker_index
        payload['tmp_bucket'] = tmp_bucket
        payload['merged_bucket'] = merged_bucket
        payload['partition_bucket'] = partition_bucket
        payload['output_bucket'] = output_bucket
        lambda_client = boto3.client('lambda')
        lambda_client.invoke(FunctionName='FasDL_worker',
                                InvocationType='Event',
                                Payload=json.dumps(payload))
        
    return {"result": "succeed!"}


def load_data(batch_size):
    # load data 
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = dset.ImageFolder('/tmp/traindata', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    return trainloader


def k_reduce(s3_client, weights, num_workers, worker_index, pattern_k, output_file, iter):
    # uploading step
    upload_start = time.time() # start time of uploading step
    # transform the model parameters into 1-d array
    vector = weights[0].reshape(1, -1)
    for i in range(1, len(weights)):
        vector = np.append(vector, weights[i].reshape(1, -1))
    num_all_values = vector.size
    num_values_per_agg = num_all_values // pattern_k
    residue = num_all_values

    # write partitioned vector to the shared memory, except the chunk charged by itself
    for i in range(pattern_k):
        if i != worker_index:
            offset = (num_values_per_agg * i) + min(residue, i)
            length = num_values_per_agg + (1 if i < residue else 0)
            # indicating the chunk number and which worker it comes from
            key = "{}_{}_{}".format(i, worker_index, iter)
            # format of key in tmp-bucket: chunkID_workerID_iteration
            s3_client.put_object(
                Bucket = tmp_bucket, Key = key, Body = vector[offset: offset + length].tobytes())
    upload_time = time.time() - upload_start # time of uploading step
    output_file.write(f"    Upload Time: {upload_time:.2f}\n")

    # non-aggregators dont update in the first iteration
    if iter == 1 and worker_index >= pattern_k:
        return weights

    merged_value = dict() # merged parameters
    
    # aggregation step
    if worker_index < pattern_k:
        # aggregator only
        aggre_start = time.time() # start time of aggregation step
        my_offset = (num_values_per_agg * worker_index) + \
            min(residue, worker_index)
        my_length = num_values_per_agg + (1 if worker_index < residue else 0)
        my_chunk = vector[my_offset: my_offset + my_length]

        # read and aggregate the corresponding chunk
        num_files = 0
        while num_files < num_workers - 1:
            lists = s3_client.list_objects_v2(Bucket=tmp_bucket)
            if lists['KeyCount'] > 0:
                objects = lists['Contents']
            else:
                objects = None
            if objects is not None:
                for obj in objects:
                    file_key = urllib.parse.unquote_plus(
                        obj["Key"], encoding='utf-8')
                    key_splits = file_key.split("_")

                    # if it's the chunk it care and it is from the current step
                    # format of key in tmp-bucket: chunkID_workerID_iteration
                    if key_splits[0] == str(worker_index) and key_splits[2] == str(iter):
                        data = s3_client.get_object(Bucket = tmp_bucket, Key = file_key)['Body'].read()
                        bytes_data = np.frombuffer(data, dtype=vector.dtype)
                        my_chunk += bytes_data
                        num_files += 1
                        s3_client.delete_object(
                            Bucket=tmp_bucket, Key=file_key)

        # average weights
        my_chunk /= float(num_workers)
        # write the aggregated chunk back
        # key format in merged_bucket: iteration_chunkID
        key = "{}_{}".format(iter, worker_index)
        s3_client.put_object(Bucket = merged_bucket, Key = key, Body = my_chunk.tobytes())
        merged_value[worker_index] = my_chunk
        aggre_time = time.time() - aggre_start # time of aggregation step
        output_file.write(f"    Aggre Step: {aggre_time:.2f}\n")

    # downloading step
    # read other aggregated chunks
    download_start = time.time() # start time of downloading step
    num_merged_files = 0
    already_read_files = []
    if worker_index < pattern_k:
        total_files = pattern_k-1
        target_iter = iter 
    else:
        total_files = pattern_k
        target_iter = iter - 1 # non-aggregator fetch the merged file of last iteration
    while num_merged_files < total_files:
        lists = s3_client.list_objects_v2(Bucket=merged_bucket)
        if lists['KeyCount'] > 0:
            objects = lists['Contents']
        else:
            objects = None
        if objects is not None:
            for obj in objects:
                file_key = urllib.parse.unquote_plus(
                    obj["Key"], encoding='utf-8')
                key_splits = file_key.split("_")

                # key format in merged_bucket: iteration_chunkID
                # if not file_key.startswith(str(my_rank)) and file_key not in already_read:
                if key_splits[1] != str(worker_index) \
                        and key_splits[0] == str(target_iter) and file_key not in already_read_files:
                    data = s3_client.get_object(
                            Bucket = merged_bucket, Key = file_key)['Body'].read()
                    bytes_data = np.frombuffer(data, dtype=vector.dtype)
                    merged_value[int(key_splits[1])] = bytes_data
                    already_read_files.append(file_key)
                    num_merged_files += 1

    # reconstruct the whole vector
    new_vector = merged_value[0]
    for k in range(1, pattern_k):
        new_vector = np.concatenate((new_vector, merged_value[k]))
    result = dict()
    index = 0
    for k in range(len(weights)):
        lens = weights[k].size
        tmp_arr = new_vector[index:index + lens].reshape(weights[k].shape)
        result[k] = tmp_arr
        index += lens
    download_time = time.time() - download_start # time of downloading step
    output_file.write(f"    Download Time: {download_time:.2f}\n")

    return result


def setup_seed(seed):
    # set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def unzip_file(zip_name, dst_dir):
    # unzip file
    fz = zipfile.ZipFile(zip_name, 'r')
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    fz.close()
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