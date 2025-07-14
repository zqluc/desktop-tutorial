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
# path_adult = r"D:\Users\admin\Desktop\elpriv-bayes-master\Datasets\Processed_used datasets\processed_dataset_Adult_41292.csv"
path_NLTSC = r"D:\Users\admin\Desktop\elpriv-bayes-master\Datasets\Processed_used datasets\NLTSC_20000 .csv"
# path_BR2000 = r"D:\Users\admin\Desktop\elpriv-bayes-master\Datasets\Processed_used datasets\BR2000.csv"
# Data_adult = pd.read_csv(path_adult)  # " D:/a/b/c/abc.csv"
Data_NLTCS = pd.read_csv(path_NLTSC)
# Data_BR2000 = pd.read_table(path_BR2000, sep=',', index_col=0)
# Data_adult.drop('native-country', axis=1, inplace=True)
# datasets=['Adult']
# path = [Data_adult]
datasets = ['NLTCS']
path = [Data_NLTCS]
# datasets = ['BR2000']
# path = [Data_BR2000]
epsilon_list_label = ['0.02','0.04','0.08','0.2','0.4','0.8','1.6','3.2','6.4']
abcd = ['f']
fig, ax = plt.subplots(1, 4, figsize=(24, 4), dpi=300)  # , constrained_layout=True)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
print(ax)
ax = [ax for ax in ax.reshape(1, -1)[0]]
# fig.tight_layout(w_pad=1)
for i in range(1):
    df = path[i]
    result = pd.read_csv(
        r'D:\Users\admin\Desktop\elpriv-bayes-master\Non-Privacy-Sturctlearning\结构学习中间的对比_{}_Isum_3.csv'.format(
            datasets[i]))
    score_priv_list=result['PrivBayes']
    score_el_list = result['ELPrivBayes']
    score_our3_list = result['Our3']
    score_our2_list = result['Our2']
    score_our4_list = result['Our4']

    ax[i].plot(epsilon_list_label, score_our2_list, color='black', linestyle='--', label='SA-PrivBayes,λ=1/2'
               , marker='*', ms=8, linewidth=0.7, markerfacecolor='none')
    ax[i].plot(epsilon_list_label, score_our3_list, color='black', linestyle='-', label='SA-PrivBayes,λ=2/3'
               , marker='s', ms=8, linewidth=0.7, markerfacecolor='none')
    ax[i].plot(epsilon_list_label, score_our4_list, color='black', linestyle=':', label='SA-PrivBayes,λ=3/4'
               , marker='D', ms=8, linewidth=0.7, markerfacecolor='none')
    ax[i].plot(epsilon_list_label, score_el_list, color='black', linestyle='--', label='ELPrivBayes'
               , marker='^', ms=8, linewidth=0.7, markerfacecolor='none')
    ax[i].plot(epsilon_list_label, score_priv_list, color='black', linestyle=':', label='PrivBayes',
               marker='o', ms=8, linewidth=0.7, markerfacecolor='none')

    font={'family':'Times New Roman', 'style':'italic' , 'weight': 'normal'}# 这个'italic'为斜体 'weight':'normal', 'color':'red', 'size':16 }
    ax[i].set_xlabel(fontdict=font, xlabel='ε', fontsize=20, verticalalignment='center', y=1.2)
    ax[i].set_title('({}) {},k=3'.format(abcd[i], datasets[i]), fontsize=20, y=-0.26)
    if i == 0:
        ax[i].set_ylabel('Sum of mutual information', fontsize=20)
ax[0].legend(bbox_to_anchor=(1, 0), loc='lower right', fontsize=10, framealpha=0.5)
plt.savefig(
    r"D:\Users\admin\Desktop\elpriv-bayes-master\Non-Privacy-Sturctlearning\mutual-k=3_NLTCS.svg",
    format='svg',dpi=300, bbox_inches='tight', pad_inches=0.07)