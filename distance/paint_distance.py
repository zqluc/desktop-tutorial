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
# path_NLTSC = r"D:\Users\admin\Desktop\elpriv-bayes-master\Datasets\Processed_used datasets\NLTSC_20000 .csv"
path_BR2000 = r"D:\Users\admin\Desktop\elpriv-bayes-master\Datasets\Processed_used datasets\BR2000.csv"
# Data_adult = pd.read_csv(path_adult)  # " D:/a/b/c/abc.csv"
# Data_NLTCS = pd.read_csv(path_NLTSC)
Data_BR2000 = pd.read_table(path_BR2000, sep=',', index_col=0)
# Data_adult.drop('native-country', axis=1, inplace=True)
# datasets=[Data_adult]
# datasets_name = ['Adult']
# datasets = [Data_NLTCS]
# datasets_name = ['NLTCS']
datasets = [Data_BR2000]
datasets_name = ['BR2000']

abcd = ['c']
epsilon_list = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
fig, ax = plt.subplots(1, 4, figsize=(24, 4), dpi=300)  # , constrained_layout=True)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'

epsilon_list_label = ['0.05','0.1', '0.2', '0.4', '0.8', '1.6', '3.2']
print(ax)
ax = [ax for ax in ax.reshape(1, -1)[0]]
# fig.tight_layout(w_pad=1)
k = 2
alpha = 2
for i in range(0, 1):
    result = pd.read_csv(
        r'D:\Users\admin\Desktop\elpriv-bayes-master\Non-Privacy-Sturctlearning\distance-{}-k={}.csv'.format(
            datasets_name[i], k))

    score_SA_PrivBayes_list = result['SA-PrivBayes']
    score_EL_PrivBayes_list = result['ELPrivBayes']
    score_PrivBayes_list = result['PrivBayes']
    ['brown', 'darkgreen', 'indigo', 'navy']
    ax[i].plot(epsilon_list_label, score_SA_PrivBayes_list, color='black', label='SA-PrivBayes',
               marker='*', ms=8, linewidth=0.7, markerfacecolor='none', linestyle='--')
    ax[i].plot(epsilon_list_label, score_EL_PrivBayes_list, color='black', label='ELPrivBayes',
               marker='^', ms=8, linewidth=0.7, markerfacecolor='none')
    ax[i].plot(epsilon_list_label, score_PrivBayes_list, color='black', label='PrivBayes'
               , marker='o', ms=8, linewidth=0.7, markerfacecolor='none')
    font = {'family': 'Times New Roman', 'style': 'italic',
            'weight': 'normal'}  # 这个'italic'为斜体 'weight':'normal', 'color':'red', 'size':16 }
    ax[i].set_xlabel(fontdict=font, xlabel='ε', fontsize=20, verticalalignment='center', y=1.2)  #
    font2 = {'family': 'Times New Roman',
             'weight': 'normal'}  # 这个'italic'为斜体 'weight':'normal', 'color':'red', 'size':16 }
    ax[i].set_title("({0}) {1}, Q$_2$".format(abcd[i], datasets_name[i]), y=-0.22, fontsize=20)  # ,fontdict=font2)
    if i == 0:
        ax[i].set_ylabel('Average variation distance', fontsize=20)
ax[0].legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize=10, framealpha=0.5)
plt.savefig(r"D:\Users\admin\Desktop\elpriv-bayes-master\Non-Privacy-Sturctlearning\distance-BR2000.svg".format(k),
            format='svg', dpi=300, bbox_inches='tight', pad_inches=0.07)