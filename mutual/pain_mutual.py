# 依赖
import pandas as pd
import numpy as np
import sklearn.metrics as mtr
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import time
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from sklearn import metrics
from sklearn import svm
from collections import Counter
import math
# datasets=['Adult']
datasets = ['NLTCS']
# datasets = ['BR2000']
epsilon_list_label = ['0.02','0.04','0.08','0.2','0.4','0.8','1.6','3.2','6.4']
abcd = ['f']
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
print(ax)
result = pd.read_csv(
    r'D:\Users\admin\Desktop\elpriv-bayes-master\Non-Privacy-Sturctlearning\结构学习中间的对比_{}_Isum_3.csv'.format(
        datasets[0]))
score_priv_list=result['PrivBayes']
score_el_list = result['ELPrivBayes']
score_our3_list = result['Our3']
score_our2_list = result['Our2']
score_our4_list = result['Our4']

ax.plot(epsilon_list_label, score_our2_list, color='black', linestyle='--', label='SA-PrivBayes,λ=1/2'
           , marker='*', ms=8, linewidth=0.7, markerfacecolor='none')
ax.plot(epsilon_list_label, score_our3_list, color='black', linestyle='-', label='SA-PrivBayes,λ=2/3'
           , marker='s', ms=8, linewidth=0.7, markerfacecolor='none')
ax.plot(epsilon_list_label, score_our4_list, color='black', linestyle=':', label='SA-PrivBayes,λ=3/4'
           , marker='D', ms=8, linewidth=0.7, markerfacecolor='none')
ax.plot(epsilon_list_label, score_el_list, color='black', linestyle='--', label='ELPrivBayes'
           , marker='^', ms=8, linewidth=0.7, markerfacecolor='none')
ax.plot(epsilon_list_label, score_priv_list, color='black', linestyle=':', label='PrivBayes',
           marker='o', ms=8, linewidth=0.7, markerfacecolor='none')

font={'family':'Times New Roman', 'style':'italic' , 'weight': 'normal'}
ax.set_xlabel(fontdict=font, xlabel='ε', fontsize=20, verticalalignment='center', y=1.2)
ax.set_title('({}) {},k=3'.format(abcd[0], datasets[0]), fontsize=20, y=-0.26)
ax.set_ylabel('Sum of mutual information', fontsize=20)
ax.legend(bbox_to_anchor=(1, 0), loc='lower right', fontsize=10, framealpha=0.5)
plt.savefig(
    r"D:\Users\admin\Desktop\elpriv-bayes-master\Non-Privacy-Sturctlearning\mutual-k=3_{}.svg".format(
            datasets[0]),format='svg',dpi=300, bbox_inches='tight', pad_inches=0.07)