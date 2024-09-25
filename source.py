import torch
import torchvision 
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot 
import yaml
class Net(torch.nn.Module):
        def __init__(self, activation_function):
            
            super(Net,self).__init__()
            
            if activation_function == 'ReLU':
                activation = torch.nn.ReLU()
            elif activation_function == 'Sigmoid':
                activation = torch.nn.Sigmoid()
            elif activation_function == 'Tanh':
                activation = torch.nn.Tanh()
                
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
        
        def forward(self,input):
            output = self.model(input)
            return output
        
def __main__():
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean = [0.5],std = [0.5])])
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    BATCH_SIZE = config['BATCH_SIZE']
    EPOCHS = config['EPOCHS']
    LEARNING_RATE = config['LEARNING_RATE']
    activation_function = config['ACTIVATION_FUNCTION']
    trainData = torchvision.datasets.MNIST('./data/',train = True,transform = transform,download = True)
    testData = torchvision.datasets.MNIST('./data/',train = False,transform = transform)


    trainDataLoader = torch.utils.data.DataLoader(dataset = trainData,batch_size = BATCH_SIZE,shuffle = True)
    testDataLoader = torch.utils.data.DataLoader(dataset = testData,batch_size = BATCH_SIZE)
    net = Net(activation_function)
    print(net.to(device))

    lossF = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(net.parameters(), lr = LEARNING_RATE)

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
    

    pyplot.plot(history['Test Loss'],label = 'Test Loss')
    pyplot.legend(loc='best')
    pyplot.grid(True)
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.show()

    pyplot.plot(history['Test Accuracy'],color = 'red',label = 'Test Accuracy')
    pyplot.legend(loc='best')
    pyplot.grid(True)
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.show()

    torch.save(net,'./model.pth')

if __name__ == "__main__":
    __main__()
