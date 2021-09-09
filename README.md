# bayesian-neural-network
Code repo for dissertation: Cold Variational Inference in Neural Network.

## Getting started
Development environment is built by `docker`. Once you have docker ready, you can do following command to build up the environment and run experiments.
```
# build environemnt
make build

# run docker container with bash
make bash

# run docker container with jupyter (with gpus)
make jupyter
```

## How to run experiments
```bash
# step 1: run docker container with bash
make bash

# step 2:
python src/bnn/train.py --output-dir <OUTPUT_DIR> 
                        --pretrained-dir <DIR_FOR_PRETRAINED_RESNET>
                        --data-set <DATASET_NAME>
                        --model <MODEL_NAME>
# eg. 
python src/bnn/train.py --output-dir .
                        --pretrained-dir .
                        --data-set MNIST
                        --model MCBB
```
- dataset: MNIST, CIFAR-10, CIFAR-100. 
- model: CBB, MCBB.

An example of training a MCBB model on MNIST is provided. Please refer to my dissertation for more details.

Code for BS are also provided (train_BS.py and model_BS.py) for your interest.
