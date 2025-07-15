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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import scipy

abcd = ['a']
# datasets_name = ['Adult']
datasets_name = ['BR2000']
# datasets_name = ['NLTCS']
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)  # , constrained_layout=True)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
epsilon_list_label = ['0.05', '0.2','1', '1.8', '3.2','6.4']
print(ax)
k = 2
alpha = 2
for i in range(0, 1):
    result = pd.read_csv(
        r'D:\Users\admin\Desktop\elpriv-bayes-master\Non-Privacy-Sturctlearning\{}-AVD-SVM-{}.csv'.format(
            datasets_name[0],datasets_name[0]))
    score_PrivBayes_list = result['PrivBayes']
    score_ELPrivBayes_list = result['ELPrivBayes']
    score_SAPrivBayes1_list = result['SAPrivBayes,q=1.1']
    score_SAPrivBayes0_list = result['SAPrivBayes,q=1.0']
    score_SAPrivBayes2_list = result['SAPrivBayes,q=1.2']
ax.plot(epsilon_list_label, score_SAPrivBayes0_list, color='black', linestyle='-',label='SA-PrivBayes,q=1.0'
           , marker='*', ms=8, linewidth=0.7, markerfacecolor='none')
ax.plot(epsilon_list_label, score_SAPrivBayes1_list, color='black', linestyle='-', label='SA-PrivBayes,q=1.1'
           , marker='s', ms=8, linewidth=0.7, markerfacecolor='none')
ax.plot(epsilon_list_label, score_SAPrivBayes2_list, color='black', linestyle=':', label='SA-PrivBayes,q=1.2'
           , marker='D', ms=8, linewidth=0.7, markerfacecolor='none')
ax.plot(epsilon_list_label, score_ELPrivBayes_list, color='black',linestyle='--', label='ELPrivBayes'
           , marker='^', ms=8, linewidth=0.7, markerfacecolor='none')
ax.plot(epsilon_list_label, score_PrivBayes_list, color='black', linestyle=':',label='PrivBayes',
           marker='o', ms=8, linewidth=0.7, markerfacecolor='none')
font = {'family': 'Times New Roman', 'style': 'italic',
        'weight': 'normal'}  # 这个'italic'为斜体 'weight':'normal', 'color':'red', 'size':16 }
ax.set_xlabel(fontdict=font, xlabel='ε', fontsize=20, verticalalignment='center', y=1.2)  #
font2 = {'family': 'Times New Roman',
         'weight': 'normal'}  # 这个'italic'为斜体 'weight':'normal', 'color':'red', 'size':16 }
ax.set_title("({0}) {1}".format(abcd[0], datasets_name[0]), y=-0.22, fontsize=20)  # ,fontdict=font2)
ax.set_ylabel('Accuracy', fontsize=20)
ax.legend(bbox_to_anchor=(1, 0), loc='lower right', fontsize=10, framealpha=0.5)
plt.savefig(r"D:\Users\admin\Desktop\elpriv-bayes-master\Non-Privacy-Sturctlearning\accuracy-{}.svg".format(
            datasets_name[0]),format='svg', dpi=300, bbox_inches='tight', pad_inches=0.07)