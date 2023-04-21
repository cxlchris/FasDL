# FasDL

## Background

**FasDL** is an efficient serverless-based deep learning training framework, consisting of three components: **Profiler**, **Optimizer** and **K-Reduce**. The profiler analyzes the training performance of the deep learning model on the serverless platform and fetches the coefficients related to the model the training time and the cost of the serverless-based deep learning training task. The optimizer searches for the optimal resource configuration based on a heuristic search algorithm to minimize the cost under the time constraint. The K-Reduce contains two parts: the FasDL trigger and the FasDL worker. FasDL trigger divides the dataset unequally and triggers the workers. FasDL workers carry out deep learning training based on the K-Reduce communication mechanism. FasDL is implemented atop AWS Lambda and AWS S3.

## Prerequisites

This project uses docker and AWS CLI.

```shell
$ sudo yum update -y
$ sudo amazon-linux-extras install docker
$ sudo service docker start
```

```shell
$ curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
$ unzip awscliv2.zip
$ sudo ./aws/install
```

And five buckets on S3 are required:

- data bucket: bucket storing dataset
- partition bucket: bucket storing partitioned dataset
- tmp bucket: bucket storing tmp file
- merged bucket: bucket storing merged file
- output bucket: bucket storing log file

## Profiler

The profiler consists of two stages: profiling and fitting. 

In the profiling stage, the workload is trained on AWS Lambda for a single iteration and the profiler collects the training time and transformation time under different memory and batch size configuration. 

First, build the profiling docker image and upload it to the AWS ECR. ([Using Amazon ECR with the AWS CLI - Amazon ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html))

```shell
$ cd /FasDL/Profiler/profiling
$ docker build -t profiling .
$ docker tag profiling.latest <repository>:profiling
$ docker push <repository>:profiling
```

Upload the profiling.zip to the data bucket.

```shell
$ aws s3api put-object --body /FasDL/Profiler/profiling.zip --bucket <data bucket> --Key profiling.zip
```

Create the Lambda function based on the docker image uploaded.

```shell
$ aws lambda create-function --function-name profiling --role profiler --code ImageUri=<imageuri> -- timeout 900
```

Configure the function with different memory and trigger the function with different batch size.

```shell
$ aws lambda update-function-configuration --function-name profiling --memory-size <memory>
$ aws lambda invoke --function-name profiling --payload <JSON> /FasDL/Output 
```

The format of the payload is as follows:

```json
{
  "train_net": "<model>",
  "batch_size": "<batch size>",
  "memory": "<memory>",
  "seed": "<random seed>",
  "data_bucket": "<data bucket>",
  "output_bucket": "<output bucket>"
}
```

The log of the training time and the transform time is recorded in profiling.txt, which is uploaded to the output bucket on S3.

In the fitting stage, the profiler fits the coefficients related to the training and transformation, and models the training time and cost of the training process.

Download the profiling.txt from the output bucket.

```shell
$ aws s3api get-object --bucket <output bucket> --Key profiling /FasDL/Output
```

Then fit the training-related coefficients and transformation-related coefficients with the fit_coefficients.py.

```shell
$ cd /FasDL/Output
$ python3 /FasDL/Profiler/fit_coefficients.py
```

The output contains the training-related coefficients **a**, **b** and **m** and the transformation-related coefficients **k1** and **k2**.

## Optimizer

The optimizer searches for the optimal resource configuration based on a heuristic search algorithm to minimize the cost under the time constraint.

Run the optimizer.py to search for the optimal configuration. The time constraint is needed.

```
$ cd /FasDL/Optimizer
$ python3 /FasDL/Optimizer/optimizer.py
```

The output includes **the batch size of aggregators**, **the number of workers**, **the memory of functions**, **the batch size of non-aggregators** and **the number of aggregators**. Also, the predicted time and cost and the comparison with the ScatterReduce are also given.

## K-Reduce

The K-Reduce contains two parts: the FasDL trigger and FasDL worker.

The K-Reduce contains two parts: the FasDL trigger and FasDL worker. FasDL trigger divides the dataset unequally and triggers the workers. FasDL workers carry out deep learning training based on the K-Reduce communication mechanism. FasDL is implemented atop AWS Lambda and AWS S3.

Download the cifar10 dataset (http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and unzip it. Then upload the cifar10 dataset to the data bucket.

```shell
$ aws s3api put-object --body data_batch_1 --bucket <data bucket> --Key data_batch_1
```

Then create the docker image of FasDL_trigger and upload it to the AWS ECR ([Using Amazon ECR with the AWS CLI - Amazon ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html)).

```shell
$ cd /FasDL/K-Reduce/FasDL_trigger
$ docker build -t fasdl_trigger .
$ docker tag fasdl_trigger.latest <repository>:fasdl_trigger
$ docker push <repository>:fasdl_trigger
```

Create the Lambda function fasdl_trigger based on the docker image uploaded.

```shell
$ aws lambda create-function --function-name fasdl_trigger --role fasdl_trigger --code ImageUri=<imageuri> -- timeout 900 -- memory-size 1024 
```

Also, create the docker image of FasDL_worker and upload it to the AWS ECR ([Using Amazon ECR with the AWS CLI - Amazon ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html)).

```shell
$ cd /FasDL/K-Reduce/FasDL_worker
$ docker build -t fasdl_worker .
$ docker tag fasdl_worker.latest <repository>:fasdl_worker
$ docker push <repository>:fasdl_worker
```

Create the Lambda function fasdl_worker based on the docker image uploaded.

```shell
$ aws lambda create-function --function-name fasdl_worker --role fasdl_worker --code ImageUri=<imageuri> -- timeout 900 -- memory-size <memory> 
```

Add the lambda:InvokeFunction permission to enable the invocation of fasdl_trigger and itself.

```shell
$ aws lambda add-permission --function-name fasdl_worker --statement-id fasdl_trigger --action lambda:InvokeFunction --action <fasdl_trigger ARN>
$ aws lambda add-permission --function-name fasdl_worker --statement-id fasdl_worker --action lambda:InvokeFunction --action <fasdl_worker ARN>
```

Then invoke the function fasdl_trigger to carry out the training. 

```shell
$ aws lambda invoke --function-name fasdl_trigger --payload <JSON> /FasDL/Output 
```

The configurations are included in the payload in the following format:

```json
{			
  "train_net": "<model>",			
  "pattern_k": "<number of aggregators>",			
  "num_workers": "<number of workers>",			
  "batch_size_wor": "<batch size of non-aggregators>",			
  "batch_size_agg": "<batch size of aggregators>",			
  "epochs": "<number of epochs>",						
  "l_r": "<learning rate>",					
  "memory": "<memory of function>",					
  "func_time": "900",					
  "seed": "<random seed>",				
  "tmp_bucket": "<tmp bucket>",			
  "merged_bucket": "<merged bucket>",			
  "partition_bucket": "<partition bucket>",	
  "data_bucket": "<data bucket>",			
  "output_bucket": "output bucket",
}
```

The output is the log file and the model which are stored in the output bucket.