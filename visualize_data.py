from visdom import Visdom
import scipy.io as io
import numpy as np

#python -m visdom.server   #于cmd输入

vis = Visdom()
pred = io.loadmat('data2/Dataset.mat')  
pred = pred['pred'].astype(np.uint8)  #'pred'为自定义的.mat内部变量
pred = 100 * pred  #100为了拉大灰度值差异，可自定义
vis.image(pred)