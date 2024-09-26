# 文件名：source.py
# 创建日期：2024/9/23
# 作者：jiangziyan—693
import torch
import torchvision 
from tqdm import tqdm
import matplotlib.pyplot as plt

import yaml
class Net(torch.nn.Module):
    # 此处定义了神经网络模型的架构,目前的架构包括:1.全连接神经网络(FC) 2.卷积神经网络(CNN)
    # 可以在config.yaml文件内进行选择
        def __init__(self, activation_function, net_structure):
            super(Net,self).__init__()
            
            # 此处可以选择想使用的激活函数，目前可以使用的激活函数包括：1.ReLU 2.Sigmoid 3.Tanh
            # 可以在config.yaml文件内进行配置
            if activation_function == 'ReLU':
                activation = torch.nn.ReLU()
            elif activation_function == 'Sigmoid':
                activation = torch.nn.Sigmoid()
            elif activation_function == 'Tanh':
                activation = torch.nn.Tanh()
            
            # 全连接层神经网络的具体架构，此处采用五层，维度分别问784，512，256，128，64，10
            if net_structure == 'FC':
                self.model = torch.nn.Sequential(
                torch.nn.Flatten(),  # 将输入的 28x28 图像展平成 784 维向量
                torch.nn.Linear(784, 512),
                activation,
                torch.nn.Linear(512, 256),
                activation,
                torch.nn.Linear(256, 128),
                activation,
                torch.nn.Linear(128, 64),
                activation,
                torch.nn.Linear(64, 10),
                torch.nn.Softmax(dim=1)
            )
            
            # 图卷积神经网络的具体架构
            elif net_structure == "CNN":
                self.model = torch.nn.Sequential(
            # 图片大小：28x28
            torch.nn.Conv2d(in_channels = 1,out_channels = 16,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),
            
            # 图片大小：14x14
            torch.nn.Conv2d(in_channels = 16,out_channels = 32,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),
            
            # 图片大小：7x7
            torch.nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            
            # Flatten成一维向量
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 7 * 7 * 64,out_features = 128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 128,out_features = 10),
            torch.nn.Softmax(dim=1)
        )
        
        def forward(self,input):
            output = self.model(input)
            return output
        
def __main__():
    
    # 运算设备选择，如果cuda可用，则选用gpu进行计算，否则选用cpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 使用归一化处理
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean = [0.5],std = [0.5])])
    
    # 读取配置文件，包括：1.batchs_size 2.epochs 3.learning_rate 4.activation_function 5. net_structure 6.optimizer
    # 可以在config.yaml文件内进行配置
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    BATCH_SIZE = config['BATCH_SIZE']
    EPOCHS = config['EPOCHS']
    LEARNING_RATE = config['LEARNING_RATE']
    activation_function = config['ACTIVATION_FUNCTION']
    net_structure = config['NET_STRUCTURE']
    opti = config['OPTIMIZER']
    
    # 训练数据下载
    trainData = torchvision.datasets.MNIST('./data/',train = True,transform = transform,download = True)
    testData = torchvision.datasets.MNIST('./data/',train = False,transform = transform)

    # 训练数据加载
    trainDataLoader = torch.utils.data.DataLoader(dataset = trainData,batch_size = BATCH_SIZE,shuffle = True)
    testDataLoader = torch.utils.data.DataLoader(dataset = testData,batch_size = BATCH_SIZE)
    net = Net(activation_function, net_structure)
    print(net.to(device))

    # 选择损失函数，这里采用二元交叉熵
    lossF = torch.nn.CrossEntropyLoss() 
    
    # 优化器选择，这里提供的选择包括：1.Adam 2.SGD 3. NAdam
    # 可以在config.yaml文件内进行配置
    if opti == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr = LEARNING_RATE)
    elif opti == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr = LEARNING_RATE)
    elif opti == 'NAdam':
        optimizer = torch.optim.NAdam(net.parameters(), lr=LEARNING_RATE)

    # 终端显示训练结果
    history = {'Test Loss': [], 'Test Accuracy': []}
    for epoch in range(1, EPOCHS + 1):
        processBar = tqdm(trainDataLoader, unit='step')
        net.train(True)
        for step, (trainImgs, labels) in enumerate(processBar):
            trainImgs = trainImgs.to(device)
            labels = labels.to(device)

            net.zero_grad()
            outputs = net(trainImgs)
            loss = lossF(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(predictions == labels) / labels.shape[0]
            loss.backward()

            optimizer.step()
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                   (epoch, EPOCHS, loss.item(), accuracy.item()))

            if step == len(processBar) - 1:
                correct, totalLoss = 0, 0
                net.train(False)
                with torch.no_grad():
                    for testImgs, labels in testDataLoader:
                        testImgs = testImgs.to(device)
                        labels = labels.to(device)
                        outputs = net(testImgs)
                        loss = lossF(outputs, labels)
                        predictions = torch.argmax(outputs, dim=1)

                        totalLoss += loss
                        correct += torch.sum(predictions == labels)

                    testAccuracy = correct / (BATCH_SIZE * len(testDataLoader))
                    testLoss = totalLoss / len(testDataLoader)
                    history['Test Loss'].append(testLoss.item())
                    history['Test Accuracy'].append(testAccuracy.item())

                processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                        (epoch, EPOCHS, loss.item(), accuracy.item(), testLoss.item(), testAccuracy.item()))
        processBar.close()
    
    # 可视化训练过程
    
    # 损失函数图
    print("Training complete, plotting graphs...")
    print(f"history:", history)
    plt.plot(history['Test Loss'],label = 'Test Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('test_loss.png')
    plt.close()

    # 测试机准确度图
    plt.plot(history['Test Accuracy'],color = 'red',label = 'Test Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig('test_accuracy.png')
    plt.close()

    #torch.save(net,'./model.pth')

if __name__ == "__main__":
    __main__()
