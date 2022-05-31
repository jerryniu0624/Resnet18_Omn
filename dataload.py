# import numpy as np
# import h5py
  
# mat = h5py.File('./data2/Dataset.mat')
  
# print(mat['images'].shape)#查看mat文件中images的格式
#   #(2284, 3, 640, 480)
  
# images = np.transpose(mat['images'])
#  #转置，images是numpy.ndarray格式的
 
# print(images)#控制台输出数据
# print(images.shape)#输出数据格式
#  #(480, 640, 3, 2284)
 
# np.save('./images', images)#保存数据，会生成一个images.npy文件

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import  torchvision.transforms as transforms

from PIL import Image
import os	

yFile = './data2/Dataset.mat'    #相对路径
datay=sio.loadmat(yFile)
plt.imshow(datay['train'][1][0])
# images_train = np.transpose(datay['train'])
# images_test = np.transpose(datay['test'])
# print (datay)
# print (datay['train'],datay['train'].shape)
# # (200, 15, 28, 28)
# print (datay['test'],datay['test'].shape)
# # (200, 5, 28, 28)
# print (datay['images'],datay['images'].shape)
# np.save('C:\\Users\\Jerry\\Desktop\\junior2\\PRML\\Project\\images_train\\images_train.npy', images_train)
# np.save('C:\\Users\\Jerry\\Desktop\\junior2\\PRML\\Project\\images_test\\images_test.npy', images_test)
# print(list(datay))
# print(datay['test'])
# print(datay['train'])

# dataset_train = np.load('./images_train/images_train.npy')
# dataset_test = np.load('./images_test/images_test.npy')
# print(dataset_train.shape[3]) #200
# print(dataset_test.shape[3]) #200
# print(dataset_train.shape[2]) #15
# print(dataset_test.shape[2]) #5
# print(dataset_train.shape[1]) #28
# print(dataset_test.shape[1]) #28
# print(dataset_train.shape[0]) #28
# print(dataset_test.shape[0]) #28

# print(dataset_test)
path1 = './dataset/Unknown/character'
for i in range(200):
    os.makedirs(path1+str(i+200))

# path = './dataset/Known/character'
# for i in range(200):
#     os.makedirs(path+str(i))

for i in range(datay['train'].shape[0]): #200
    imgs_tensor = datay['train'][i, :, :, :]
    # print(imgs_tensor.shape)#(15, 28, 28)
    for j in range(imgs_tensor.shape[0]): #15
        img_tensor = imgs_tensor[j,:,:]
        # print(img_tensor.shape)#(28, 28)
        img = transforms.ToPILImage()(img_tensor)#转成图片
        # print(img.size)#(28, 28)
        # print(img)
        # print(img.mode)
        # if img.mode != 'L':
        #     img = img.convert('L')
        # for i in range(dataset_train.shape[3]): #200创建文件夹分类
        #     isExists = os.path.exists('./dataset_train/%d' % i )
        #     if not isExists:						#判断如果文件不存在,则创建
        #         os.makedirs('./dataset_train/%d' % i)	
        #         # print("%s 目录创建成功"%i)
        #     else:
        #         # print("%s 目录已经存在"%i)	
        #         continue			#如果文件不存在,则继续上述操作,直到循环结束
        
        # plt.savefig('./dataset_train/%d/%d.jpg' %(i,j))
        plt.imsave('./dataset/Unknown/character%d/%d.png' %(i,j), img)
        
        # if(os.path.isfile('./dataset/character%d/%d.jpg' %(i,j))):
        #     os.remove('./dataset/character%d/%d.jpg' %(i,j))
        I = Image.open('./dataset/Unknown/character%d/%d.png' %(i,j))
        L = I.convert('L')
        if(os.path.isfile('./dataset/Unknown/character%d/%d.jpg'  %(i,j))):
            os.remove('./dataset/Unknown/character%d/%d.jpg' %(i,j))
        L.save('./dataset/Unknown/character%d/%d.png' %(i,j))
        #img.show()

#添加图片
# for i in range(dataset_train.shape[3]): #200
#     for j in range(dataset_train.shape[2]): #15
#         I = Image.open('./dataset_train/%d/%d.jpg'%(i,j))
#         I.save('./dataset_train/%d/%d.jpg' %(i,j))
        



# for i in range(dataset_test.shape[3]): #200
#     imgs_tensor = dataset_test[i, :, :, :]
#     # print(imgs_tensor.shape)#(28, 28, 15)
#     for j in range(imgs_tensor.shape[2]): #15
#         img_tensor = imgs_tensor[j,:,:]
#         # print(img_tensor.shape)#(28, 28)
#         img = transforms.ToPILImage()(img_tensor)#转成图片
#         # print(img.size)#(28, 28)
#         # print(img)
        
#         for i in range(dataset_test.shape[3]): #200创建文件夹分类
#             isExists = os.path.exists('./dataset_test/%d' % i )
#             if not isExists:						#判断如果文件不存在,则创建
#                 os.makedirs('./dataset_test/%d' % i)	
#                 # print("%s 目录创建成功"%i)
#             else:
#                 # print("%s 目录已经存在"%i)	
#                 continue			#如果文件不存在,则继续上述操作,直到循环结束
        
#         plt.savefig('./dataset_test/%d/%d.jpg' %(i,j))
for i in range(datay['test'].shape[0]): #200
    imgs_tensor = datay['test'][i, :, :, :]
    # print(imgs_tensor.shape)#(15, 28, 28)
    for j in range(imgs_tensor.shape[0]): #15
        img_tensor = imgs_tensor[j,:,:]
        # print(img_tensor.shape)#(28, 28)
        img = transforms.ToPILImage()(img_tensor)#转成图片
        # print(img.size)#(28, 28)
        # print(img)
        
        # for i in range(dataset_train.shape[3]): #200创建文件夹分类
        #     isExists = os.path.exists('./dataset_train/%d' % i )
        #     if not isExists:						#判断如果文件不存在,则创建
        #         os.makedirs('./dataset_train/%d' % i)	
        #         # print("%s 目录创建成功"%i)
        #     else:
        #         # print("%s 目录已经存在"%i)	
        #         continue			#如果文件不存在,则继续上述操作,直到循环结束
        
        # plt.savefig('./dataset_train/%d/%d.jpg' %(i,j))

        plt.imsave('./dataset/Unknown/character%d/%d.png' %(i+200,j),img)
        I = Image.open('./dataset/Unknown/character%d/%d.png' %(i+200,j))
        L = I.convert('L')
        if(os.path.isfile('./dataset/Unknown/character%d/%d.jpg'  %(i+200,j))):
            os.remove('./dataset/Unknown/character%d/%d.jpg' %(i+200,j))
        L.save('./dataset/Unknown/character%d/%d.png' %(i+200,j))
        # plt.imsave('./dataset/character%d/%d.png' %(i,j+15),img)
        # if(os.path.isfile('./dataset/character%d/%d.jpg'  %(i,j+15))):
        #     os.remove('./dataset/character%d/%d.jpg' %(i,j+15))
        #img.show()

# from PIL import Image
# import os
# path = 'F:/QUEXIANJIANCESHIYAN/posdata'
# file_list = os.listdir(path)
# for file in file_list:
#     I = Image.open(path+"/"+file)
#     L = I.convert('L')
#     L.save(path+"/"+file)
    #print(file)

