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
# datasets_name = ['Adult']
# datasets_name = ['NLTCS']
datasets_name = ['BR2000']
abcd = ['c']
epsilon_list = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)  # , constrained_layout=True)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
epsilon_list_label = ['0.05','0.1', '0.2', '0.4', '0.8', '1.6', '3.2']
print(ax)
k = 2
alpha = 2
result = pd.read_csv(
    r'D:\Users\admin\Desktop\elpriv-bayes-master\Non-Privacy-Sturctlearning\distance-{}-k={}.csv'.format(
        datasets_name[0], k))
score_SA_PrivBayes_list = result['SA-PrivBayes']
score_EL_PrivBayes_list = result['ELPrivBayes']
score_PrivBayes_list = result['PrivBayes']
['brown', 'darkgreen', 'indigo', 'navy']
ax.plot(epsilon_list_label, score_SA_PrivBayes_list, color='black', label='SA-PrivBayes',
           marker='*', ms=8, linewidth=0.7, markerfacecolor='none', linestyle='--')
ax.plot(epsilon_list_label, score_EL_PrivBayes_list, color='black', label='ELPrivBayes',
           marker='^', ms=8, linewidth=0.7, markerfacecolor='none')
ax.plot(epsilon_list_label, score_PrivBayes_list, color='black', label='PrivBayes'
           , marker='o', ms=8, linewidth=0.7, markerfacecolor='none')
font = {'family': 'Times New Roman', 'style': 'italic',
        'weight': 'normal'}  # 这个'italic'为斜体 'weight':'normal', 'color':'red', 'size':16 }
ax.set_xlabel(fontdict=font, xlabel='ε', fontsize=20, verticalalignment='center', y=1.2)  #
font2 = {'family': 'Times New Roman',
         'weight': 'normal'}  # 这个'italic'为斜体 'weight':'normal', 'color':'red', 'size':16 }
ax.set_title("({0}) {1}, Q$_2$".format(abcd[0], datasets_name[0]), y=-0.22, fontsize=20)  # ,fontdict=font2)
ax.set_ylabel('Average variation distance', fontsize=20)
ax.legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize=10, framealpha=0.5)
plt.savefig(r"D:\Users\admin\Desktop\elpriv-bayes-master\Non-Privacy-Sturctlearning\distance-{}.svg".format(datasets_name[0]),
            format='svg', dpi=300, bbox_inches='tight', pad_inches=0.07)