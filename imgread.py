# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 20:51:33 2021

@author: WangJie-PC
"""

import pandas as pd
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

img_path=pd.read_table('E:/Desktop/models/cifar10_500_ss_split_10_isL2_1_IsCALP_1/correct_max_w_imgpath.txt',header=None,sep=',')
best_path=img_path.loc[179]
list_path=[]
header_path='F:/Workspace/Python/SDIdentification/LP-DeepSSL-master/'
for i in range(12):
    ab_path=header_path+best_path[i][2:-8]
    list_path.append(ab_path)
for i in range(12):
    img=mpimg.imread(list_path[i])
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    plt.savefig('E:Desktop/LPA/benchmark/correct%d.png'%(i+1))
    
    