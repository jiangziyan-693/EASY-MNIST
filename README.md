# Artificial_Intellengence_DeepLearning_Task:MNIST

> Author:Ziyan Jiang(ziy.jiang@outlook.com) Dalian University of Technology

![1727093840543](images/README/1727093840543.png)

MNIST(Mixed National Institute of Standards and Technology database), is a classic written number dataset. To complete this task, you need to do as follows:

## Environment Preparation

* First, use `GitBash` to clone the repositories:

  ```
  git clone https://github.com/jiangziyan-693/MNIST.git
  ```
* Second, you need to create a new `virtual environment`，please input the code below in terminal:

  ```
  conda create -n your_environment_name python=3.9
  conda activate your_environment_name
  ```
* Third, you need to install the `dependency package`, please input the code below in terminal:

  ```
  pip install -r requirements.txt
  ```

## Hyperparameter Adjustment

Researching the influence of Hyperparameter to LLM is a key task of this work, we have created a easy way for you to change the Hyperparameters, just open `config.yaml`, and change the parameters as you like:

```
BATCH_SIZE: 256
EPOCHS: 10
LEARNING_RATE: 0.0001
ACTIVATION_FUNCTION: 'ReLU' # 'ReLU' or 'Sigmoid' or 'Tanh'
```

* `BATCH_SIZE`: control the number of samples processed before the model's internal parameters are updated.
* `EPOCHS`: controls the number of complete passes through the entire training dataset.
* `LEARNING_RATE`: controls the speed at which a model's weights are updated during training.
* `ACTIVATION_FUCTION`: controls the output of a neural network node by introducing non-linearity.

## Run

To run the program, please input the code below in terminal:

```
sh run_source.sh
```
