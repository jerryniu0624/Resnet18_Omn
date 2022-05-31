import scipy.io as io
import matplotlib.pyplot as plt
import cv2
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import torch.optim as optim
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
from ResNet import ResNet18
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid

# 导入数据

data=io.loadmat('./data2/Dataset.mat')

# print(data.keys())
# print(data['train'].shape)
# print(data['test'].shape)
# plt.imshow(data['train'][0][0])
# 200类，每类15张训练集，5张测试集，图片size为28*28

# 重写dataset类 加载自定义数据
class MyDataset(Dataset):
    def __init__(self,data,transform=None): 
        self.data=data
        self.size=0
        self.transform=transform
        self.size=len(data)*len(data[0])
#         print(len[data])
#         print(len(data[0]))
        self.name_list=[]
        for i in range(0,len(data)):
            for j in range(0,len(data[0])):
                self.name_list.append([i,j])
        
    def __len__(self): #__len__是指数据集长度。
        return self.size
    
    
    def __getitem__(self,idx): #__getitem__就是获取样本对，模型直接通过这一函数获得一对样本对{x:y}
        
        i=self.name_list[idx][0]
        j=self.name_list[idx][1]
        img=self.data[i][j]
        label=i
        sample={'image':img,'label':label}
        if self.transform:
            sample=self.transform(sample)
        return sample

        
        

def evaluate(model, dataloder):
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloder:
            images, labels = data['image'],data['label']
            images,labels=images.to(device),labels.to(device)
            images=torch.reshape(images,(batch_size,1,28,28))
            images=images.float()
#             images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {} %' .format(len(testloader),
            100 * correct / total))
    return 100 * correct / total

def draw_loss(loss1,name):
    x=range(0,len(loss1))
    plt.title("loss") 
    plt.xlabel("training times") 
    plt.ylabel("loss") 
    plt.plot(x,loss1)
    plt.legend() # 添加图例
    
    plt.savefig('./'+name+'.jpg')
    plt.show()
    plt.close()


def draw_acc(acc,name):
    x=range(0,len(acc))
    plt.title("accuracy")
    plt.xlabel("training times")
    plt.ylabel("accuracy")
    plt.plot(x,acc)
    plt.legend() # 添加图例
    plt.savefig('./'+name+'.jpg')
    plt.show()
    plt.close()
                                  

if __name__ == '__main__':
            # 观察数据
    train_dataset=MyDataset(data['train'],transform=None)
# print(train_dataset.name_list)
    test_dataset=MyDataset(data['train'],transform=None)
    # 定义dataloader
    batch_size=8
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)   
# print(train_dataset)
    plt.figure()
    for (cnt,i) in enumerate(train_dataset):
        image = i['image']
        label = i['label']
#     print(image.shape)
        # ax = plt.subplot(3, 3, cnt+1)
        # ax.axis('off')
        # ax.imshow(image)
        # ax.set_title('label {}'.format(label))
        # plt.pause(0.001)

        # if cnt == 8:
        #     break

    net= ResNet18(200)
    print(net)
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    '''
    # try to change the learning rate
    '''
    optimizer1 = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer2 = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
    start = time.time()

    '''
    # try to change the number of total epoch 
        '''
    k=0
    loss1=[]
    acc=[]
    for epoch in range(32):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'],data['label']
            inputs,labels=inputs.to(device),labels.to(device)
            # print(inputs.shape)
            inputs=torch.reshape(inputs,(batch_size,1,28,28))
            inputs=inputs.float()
#             labels=labels.float()
            
            
            
#             inputs, labels = inputs.to(device), labels.to(device)
            if k<300:
                optimizer=optimizer1
            else:
                optimizer=optimizer2
                
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
#             print(inputs.shape)
            
            outputs = net(inputs)
            
#             outputs=outputs.reshape(outputs.shape[0],1)
#             labels=labels.reshape(labels.shape[0],1)
#             print(outputs)
#             print(labels)
#             print(outputs.shape)
#             print(labels.shape)
#             print(type(outputs))
#             print(type(labels))
#             loss=nn.functional.cross_entropy(outputs, labels)
            loss = criterion(outputs,labels)
#             print(loss)
            
            loss.backward()
            optimizer.step()
            

            # print statistics
            running_loss += loss.item()
            #print(type(loss.item()))
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                
                loss1.append(running_loss/100)
            
                running_loss = 0.0
                

        # evaluate at the end of every epoch
        acc.append(evaluate(net, testloader))

    end = time.time()

    print('Finished Training {}s'.format(end-start))
    
    model_name='resnet'
    PATH = './cifar_net'+'model: '+model_name+'.pth'
    torch.save(net.state_dict(), PATH)
#draw_loss(loss1)

    name1='loss'+'_model:'+model_name+'.npy'
    name2='acc'+'_model:'+model_name+'.npy'
    np.save(name1,loss1)
    np.save(name2,acc)


    name1='loss'+'_model:'+model_name
    name2='acc'+'_model:'+model_name
    loss1=np.load(name1+'.npy')
    print(loss1)
    
    acc=np.load(name2+'.npy')
    print(acc)
    draw_loss(loss1,name1)
    draw_acc(acc,name2)

    
