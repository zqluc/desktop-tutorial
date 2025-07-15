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

path_adult = r"D:\Users\admin\Desktop\elpriv-bayes-master\Datasets\Processed_used datasets\processed_dataset_Adult_41292.csv"
# path_NLTSC = r"D:\Users\admin\Desktop\elpriv-bayes-master\Datasets\Processed_used datasets\NLTSC_20000 .csv"
# path_BR2000 = r"D:\Users\admin\Desktop\elpriv-bayes-master\Datasets\Processed_used datasets\Processed_used datasets\BR2000.csv"
Data_adult = pd.read_csv(path_adult)
# Data_NLTCS = pd.read_csv(path_NLTSC)
# Data_BR2000 = pd.read_table(path_BR2000, sep=',', index_col=0)
Data_adult.drop('native-country', axis=1, inplace=True)

def get_mim(df):  # 互信息矩阵
    MI_matirx = pd.DataFrame(index=df.columns, columns=df.columns).fillna(0)
    len1 = df.columns.size
    variables = df.columns
    for i in range(len1):
        for j in range(i + 1, len1):
            MI_matirx.iloc[i, j] = mtr.mutual_info_score(df[variables[i]], df[variables[j]])
    MI_matirx = round(MI_matirx, 4)
    #     print("MIC矩阵为：\n",MIC_matirx)
    MI_matirx = MI_matirx + MI_matirx.T
    return MI_matirx


# 香农熵的计算函数
def get_information_Entropy(df, column_name):
    count = df[column_name].count()  # 总记录数
    values = df[column_name].unique()  # 取值范围
    column_Entropy = 0  # 信息熵
    Pr = {}
    for i in range(0, values.size):
        Pr[values[i]] = (df[column_name][df[column_name] == values[i]].count()) / count
        column_Entropy += -(Pr[values[i]] * np.log(Pr[values[i]]))  # 信息熵公式
    return column_Entropy
    Pr.clear()


def get_subset(V, k):  # 求子集
    if (len(V) >= k):
        return list(itertools.combinations(V, k))  # 返回V固定长度k的子集
    else:
        return [tuple(V)]


def exponential(options, scores, epsilon, sensitivity):  # 指数机制
    probabilities = [np.exp(epsilon * score / (2 * sensitivity)) for score in scores]
    if np.inf in probabilities:
        for i in range(0, len(probabilities)):
            if probabilities[i] == np.inf:
                probabilities[i] = 1
            else:
                probabilities[i] = 0
    probabilities = probabilities / np.linalg.norm(probabilities, ord=1)  # 归一化概率 # ord 一阶范式 即绝对值之和
    return np.random.choice(options, size=1, replace=False, p=probabilities)[0]


def get_MI_sensitivity(n, binary: bool):
    if binary == False:
        return (2 / n) * np.log2((n + 1) / 2) + ((n - 1) / n) * np.log2((n + 1) / (n - 1))
    else:
        return (1 / n) * np.log2(n) + ((n - 1) / n) * np.log2(n / (n - 1))


def N2bn(N):
    Edges = []
    for i in range(1, len(N)):
        for j in N[i][1]:
            Edges.append((j, N[i][0]))
    return Edges


# 计算信息熵最大的节点。
def get_joint_distribution(v_v):
    res = itertools.product(*v_v)
    result = list(res)
    return [list(a) for a in result]



# 首节点选择方法
class firstnode():

    def get_maxentropy_node(df):
        variables = df.columns
        max_entropy = 0.0
        for variable in variables:
            entropy = get_information_Entropy(df, variable)
            if entropy > max_entropy:
                max_entropy = entropy
                node = variable
        return node

    def get_ours_first_node(matrix, df):
        for node in matrix.sum().keys():
            if matrix.sum()[node] == max(matrix.sum()):
                return node

    # 静态权重值的方法 CSAPrivBayes

    def CSY_bayes_get_firstnode(df):
        domain_list = []
        for atr in df.columns:
            domain_atr = df[atr].unique().size
            # print(atr,domain_atr)
            domain_list.append(domain_atr)
        max_index = domain_list.index(max(domain_list))
        firnode = df.columns[max_index]
        return firnode

    def random_firstnode(df):
        Atr = df.columns.values.tolist()
        return Atr[np.random.randint(0, len(Atr))]

    # ---------------------------------------------------------------------------------------------


class PrivBayes():
    def PrivBayes_construct_Bayes(df, k, first_node, matrix):
        N = []  # 存放网络结构
        V = []  # 已在网络中的Attribute
        A = df.columns.values.tolist()
        score_sum = 0.0  # 求ap对的分数之和，作为评价指标
        df_len = len(A)
        #     X_1 = A[np.random.randint(0,len(df.columns))] # 随机选择首节点
        X_1 = first_node
        N.append((X_1, (np.nan)))
        V.append(X_1)
        A.remove(X_1)
        for i in range(1, df_len):  # [1,df_len-1]
            Omega = []
            scores = []
            V_k_subset = get_subset(V, k)
            for X in A:
                for pai in V_k_subset:
                    score_X_pai = 0.0
                    for atr in pai:
                        score_X_pai += matrix[X][matrix.index == atr].values[0]
                    scores.append(score_X_pai)  # 1
                    Omega.append((X, pai))
            # 直接以分数最大为原则选择AP对
            i = scores.index(np.max(scores))
            score_sum += np.max(scores)
            # 使用指数机制选择AP对,打分函数为MIC
            #         i = exponential(options=np.arange(0,len(Omega),1), scores=scores, epsilon =epsilon, sensitivity=get_MI_sensitivity(df.shape[0],False)) # 选出下标
            #         score_sum += scores[i]
            X_i = Omega[i][0]  # 从A选出的节点
            N.append(Omega[i])  # 给网络加入依赖
            V.append(X_i)
            A.remove(X_i)
        #     print("score_sum为:",score_sum)
        return N, score_sum

    def PrivBayes_PDconstruct_Bayes(df, k, first_node, matrix, epsilon, sensitivity):
        N = []  # 存放网络结构
        V = []  # 已在网络中的Attribute
        A = df.columns.values.tolist()
        score_sum = 0.0  # 求ap对的分数之和，作为评价指标
        df_len = len(A)
        #     X_1 = A[np.random.randint(0,len(df.columns))] # 随机选择首节点
        X_1 = first_node
        N.append((X_1, (np.nan)))
        V.append(X_1)
        A.remove(X_1)
        for i in range(1, df_len):  # [1,df_len-1]
            Omega = []
            scores = []
            V_k_subset = get_subset(V, k)
            for X in A:
                for pai in V_k_subset:
                    score_X_pai = 0.0
                    for atr in pai:
                        score_X_pai += matrix[X][matrix.index == atr].values[0]
                    scores.append(score_X_pai)  # 1
                    Omega.append((X, pai))
            i = exponential(options=np.arange(0, len(Omega), 1), scores=scores, epsilon=epsilon / (df_len - 1),
                            sensitivity=sensitivity)  # 选出下标
            score_sum += scores[i]
            X_i = Omega[i][0]  # 从A选出的节点
            N.append(Omega[i])  # 给网络加入依赖
            V.append(X_i)
            A.remove(X_i)
        return N, score_sum

    def NoisyConditionals(N, DF, k, epsilon2):
        P = {}
        d = DF.shape[1]
        n = DF.shape[0]
        sensitivity = 2 / n
        epsilon_i = epsilon2 / (d - k)
        for i in range(k, d)[::-1]:
            variable_names_i = []
            variable_value_i = []
            Pr_AP = np.array([])
            variable_names_i.append(N[i][0])
            variable_value_i.append(DF[N[i][0]].unique().tolist())
            for j in range(0, k):
                variable_names_i.append(N[i][1][j])
                variable_value_i.append(DF[N[i][1][j]].unique().tolist())
            df = DF[variable_names_i]
            # 求联合概率分布
            joint_distri = get_joint_distribution(variable_value_i)  # 所有组合
            condi_variable = get_joint_distribution(variable_value_i[1:])  # 所有条件组合
            for m in range(len(joint_distri)):
                Pr_AP = np.append(Pr_AP, df[df == joint_distri[m]].dropna().shape[0] / n)
            Pr_AP += np.random.laplace(loc=0, scale=sensitivity / epsilon_i)
            Pr_AP[Pr_AP < 0] = 0  # 负值赋0
            Pr_AP = Pr_AP / np.linalg.norm(Pr_AP, ord=1)  # 归一化
            # 推导条件概率分布
            for o in range(len(condi_variable)):
                pr_cpd = np.array([])
                for p in range(len(joint_distri)):
                    if condi_variable[o] == joint_distri[p][1:]:
                        pr_cpd = np.append(pr_cpd, Pr_AP[p])
                pr_cpd = pr_cpd / np.linalg.norm(pr_cpd, ord=1)
                if np.isnan(pr_cpd[0]) == True:
                    pr_cpd = np.ones_like(pr_cpd)
                    pr_cpd = pr_cpd / np.linalg.norm(pr_cpd, ord=1)
                P['{}|{}'.format(N[i][0], condi_variable[o])] = pr_cpd
        for i in range(0, k)[::-1]:
            variable_names_i = []
            variable_value_i = []
            variable_names_i.append(N[i][0])
            variable_value_i.append(DF[N[i][0]].unique().tolist())
            for j in range(0, i):
                variable_names_i.append(N[i][1][j])
                variable_value_i.append(DF[N[i][1][j]].unique().tolist())
            condi_variable = get_joint_distribution(variable_value_i[1:])
            for o in range(len(condi_variable)):
                pr_cpd = np.array([])
                for v in variable_value_i[0]:
                    pr = 0
                    for p in range(len(joint_distri)):
                        if set(condi_variable[o]).issubset(joint_distri[p][1:]) and (v in joint_distri[p]):
                            pr += Pr_AP[p]
                    pr_cpd = np.append(pr_cpd, pr)

                pr_cpd = pr_cpd / np.linalg.norm(pr_cpd, ord=1)
                if np.isnan(pr_cpd[0]) == True:
                    pr_cpd = np.ones_like(pr_cpd)
                pr_cpd = pr_cpd / np.linalg.norm(pr_cpd, ord=1)

                if i != 0:
                    P['{}|{}'.format(N[i][0], condi_variable[o])] = pr_cpd
                else:
                    P['{}'.format(N[i][0])] = pr_cpd
        return P

    def NoisyConditionals_new(N, DF, k, epsilon2, Q, score_all):
        P = {}
        d = DF.shape[1]
        n = DF.shape[0]
        sensitivity = 2 / n
        q = Q
        start_index = k  # 假设从索引位置 k开始对后面的数据排序
        # 获取排序前的前部分（不排序）
        # pre_sorted = score_all[:start_index]
        # print(pre_sorted)
        # 获取后部分（从start_index开始）
        post_sorted = score_all[start_index:]
        # 将集合从大到小排序
        sorted_data = sorted(post_sorted, reverse=True)
        # 计算分组的大小
        z = len(score_all)
        third = math.ceil((z-k) / 3) # 每一类的大致数量
        # 定义分组的标签
        labels = ['A', 'B', 'C']
        # 将数据按从大到小分为三类
        classified_data = {}
        for i, value in enumerate(sorted_data):
            if i < third:
                label = 'A'  # 最大的一类标记为A
            elif i < 2 * third:
                label = 'B'  # 中间的一类标记为B
            else:
                label = 'C'  # 最小的一类标记为C
            classified_data[value] = label
        label_counts = Counter(classified_data.values())
        for i in range(k, d)[::-1]:
            if q == 1:
                epsilon_1 = epsilon2 / 3
            else:
                epsilon_1 = epsilon2 * (1 - q) / (1 - (q ** 3))
            if classified_data.get(score_all[i]) == 'A':
                epsilon_i = epsilon_1/label_counts['A']
            elif classified_data.get(score_all[i]) == 'B':
                epsilon_i = (epsilon_1 * q)/label_counts['B']
            else:
                epsilon_i = (epsilon_1 * (q ** 2))/label_counts['C']
            variable_names_i = []
            variable_value_i = []
            Pr_AP = np.array([])
            variable_names_i.append(N[i][0])
            variable_value_i.append(DF[N[i][0]].unique().tolist())
            for j in range(0, k):
                variable_names_i.append(N[i][1][j])
                variable_value_i.append(DF[N[i][1][j]].unique().tolist())
            df = DF[variable_names_i]
            # 求联合概率分布
            joint_distri = get_joint_distribution(variable_value_i)  # 所有组合
            condi_variable = get_joint_distribution(variable_value_i[1:])  # 所有条件组合
            for m in range(len(joint_distri)):
                Pr_AP = np.append(Pr_AP, df[df == joint_distri[m]].dropna().shape[0] / n)
            Pr_AP += np.random.laplace(loc=0, scale=sensitivity / epsilon_i)
            Pr_AP[Pr_AP < 0] = 0  # 负值赋0

            Pr_AP = Pr_AP / np.linalg.norm(Pr_AP, ord=1)

            # 推导条件概率分布
            for o in range(len(condi_variable)):
                pr_cpd = np.array([])
                for p in range(len(joint_distri)):
                    if condi_variable[o] == joint_distri[p][1:]:
                        pr_cpd = np.append(pr_cpd, Pr_AP[p])
                pr_cpd = pr_cpd / np.linalg.norm(pr_cpd, ord=1)
                if np.isnan(pr_cpd[0]) == True:
                    pr_cpd = np.ones_like(pr_cpd)
                    pr_cpd = pr_cpd / np.linalg.norm(pr_cpd, ord=1)
                P['{}|{}'.format(N[i][0], condi_variable[o])] = pr_cpd
        for i in range(0, k)[::-1]:
            variable_names_i = []
            variable_value_i = []
            variable_names_i.append(N[i][0])
            variable_value_i.append(DF[N[i][0]].unique().tolist())
            for j in range(0, i):
                variable_names_i.append(N[i][1][j])
                variable_value_i.append(DF[N[i][1][j]].unique().tolist())
            condi_variable = get_joint_distribution(variable_value_i[1:])
            for o in range(len(condi_variable)):
                pr_cpd = np.array([])
                for v in variable_value_i[0]:
                    pr = 0
                    for p in range(len(joint_distri)):
                        if set(condi_variable[o]).issubset(joint_distri[p][1:]) and (v in joint_distri[p]):
                            pr += Pr_AP[p]
                    pr_cpd = np.append(pr_cpd, pr)
                pr_cpd = pr_cpd / np.linalg.norm(pr_cpd, ord=1)
                if np.isnan(pr_cpd[0]) == True:
                    pr_cpd = np.ones_like(pr_cpd)
                pr_cpd = pr_cpd / np.linalg.norm(pr_cpd, ord=1)
                if i != 0:
                    P['{}|{}'.format(N[i][0], condi_variable[o])] = pr_cpd
                else:
                    P['{}'.format(N[i][0])] = pr_cpd
        return P
    def sampling(N, DF, P):
        n = DF.shape[0]
        v_name = [v[0] for v in N]
        v_value = [DF[v].unique().tolist() for v in v_name]

        data_samp = []
        for i in range(n):
            data_i = []
            data_i.append(np.random.choice(v_value[0], size=1, replace=False, p=P[v_name[0]])[0])
            for j in range(1, len(N)):
                cv_index = [v_name.index(v) for v in list(N[j][1])]
                cvv = [data_i[index] for index in cv_index]
                p = P['{}|{}'.format(v_name[j], cvv)]
                if np.isnan(p[0]) == True:
                    p = np.ones_like(p)
                    p = p / np.linalg.norm(p, ord=1)
                data_i.append(np.random.choice(v_value[j], size=1, replace=False, p=p)[0])
            data_samp.append(data_i)
        df_generated = pd.DataFrame(data_samp, columns=v_name)
        return df_generated


def PrivBayes_PDconstruct_Bayes(df, k, first_node, matrix, epsilon, sensitivity):
    N = []  # 存放网络结构
    V = []  # 已在网络中的Attribute
    A = df.columns.values.tolist()
    score_sum = 0.0  # 求ap对的分数之和，作为评价指标
    df_len = len(A)
    #     X_1 = A[np.random.randint(0,len(df.columns))] # 随机选择首节点
    X_1 = first_node
    N.append((X_1, (np.nan)))
    V.append(X_1)
    A.remove(X_1)
    for i in range(1, df_len):  # [1,df_len-1]
        Omega = []
        scores = []
        V_k_subset = get_subset(V, k)
        maxi = matrix.sum()[A].max()
        for node in A:
            if matrix.sum()[A][node] == maxi:
                X = node
        for pai in V_k_subset:
            score_X_pai = 0.0
            for atr in pai:
                score_X_pai += matrix[X][matrix.index == atr].values[0]
            scores.append(score_X_pai)  # 1
            Omega.append((X, pai))
        i = exponential(options=np.arange(0, len(Omega), 1), scores=scores, epsilon=epsilon / (df_len - 1),
                        sensitivity=sensitivity)  # 选出下标
        score_sum += scores[i]
        X_i = Omega[i][0]  # 从A选出的节点
        N.append(Omega[i])  # 给网络加入依赖
        V.append(X_i)
        A.remove(X_i)
    #     print("score_sum为:",score_sum)
    return N, score_sum


def PrivBayes_PDconstruct_Bayes_new1(df, k, first_node, matrix, epsilon, sensitivity):
    N = []  # 存放网络结构
    V = []  # 已在网络中的Attribute
    score_all = []
    score_all.append(0)
    A = df.columns.values.tolist()
    score_sum = 0.0  # 求ap对的分数之和，作为评价指标
    df_len = len(A)
    X_1 = first_node
    N.append((X_1, (np.nan)))
    V.append(X_1)
    A.remove(X_1)

    for i in range(1, df_len):  # [1,df_len-1]
        Omega = []
        scores = []
        V_k_subset = get_subset(V, k)
        maxi = matrix.sum()[A].max()

        for node in A:
            if matrix.sum()[A][node] == maxi:
                X = node

        for pai in V_k_subset:
            score_X_pai = 0.0
            for atr in pai:
                score_X_pai += matrix[X][matrix.index == atr].values[0]
            scores.append(score_X_pai)  # 1
            Omega.append((X, pai))

        # 确保候选集有足够的元素再进行筛选
        if len(Omega) > 2:
            # 对所有候选AP对按得分排序
            sorted_scores_with_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            sorted_indices = [idx for idx, score in sorted_scores_with_indices]

            # 筛选掉最差的 1/3 的 AP 对
            threshold_index = int(len(sorted_indices) * 2 / 3)  # 选择前 2/3 的索引
            selected_indices = sorted_indices[:threshold_index]  # 筛选前 2/3 排名内的 AP 对
            selected_Omega = [Omega[idx] for idx in selected_indices]
            selected_scores = [scores[idx] for idx in selected_indices]
        else:
            # 如果候选集合数量<=2，则跳过筛选
            selected_Omega = Omega
            selected_scores = scores
        # 使用指数机制选择 2/3 排名内的 AP 对
        i = exponential(options=np.arange(0, len(selected_Omega)), scores=selected_scores,
                        epsilon=epsilon / (df_len - 1), sensitivity=sensitivity)  # 选出下标
        score_all.append(selected_scores[i])
        score_sum += selected_scores[i]
        X_i = selected_Omega[i][0]  # 从A选出的节点
        N.append(selected_Omega[i])  # 给网络加入依赖
        V.append(X_i)
        A.remove(X_i)

    return N, score_sum, score_all

def get_svm_ac(df_new, df_test, target_name):
    num_new = df_new.shape[0]
    df = pd.concat([df_new, df_test], ignore_index=True)
    fetures = df.drop([target_name], axis=1, inplace=False)
    target = df[[target_name]]
    x_encode = OneHotEncoder(sparse=True).fit_transform(fetures).toarray()
    #     x_encode = fetures.apply(LabelEncoder().fit_transform)
    y_encode = target.apply(LabelEncoder().fit_transform)
    # y_encode = OneHotEncoder(sparse=True).fit_transform(target).toarray()
    x_train = x_encode[0:num_new]
    y_train = y_encode[0:num_new]
    x_test = x_encode[num_new:]
    y_test = y_encode[num_new:]
    classifier = svm.SVC(kernel='rbf', decision_function_shape='ovr')  # decision_function_shape='ovr'
    classifier.fit(x_train, y_train)
    y_pridict = classifier.predict(x_test)
    ac = metrics.accuracy_score(y_test, y_pridict)
    return ac


def dataset_split_save(dataset, frac):
    # df = dataset.sample(frac=frac,replace=False,ignore_index=True)
    df = dataset.sample(frac=frac, replace=False).reset_index(drop=True)
    train_data, test_data = train_test_split(df, train_size=0.8, test_size=0.2)
    return train_data, test_data


epsilon_list = [0.05,0.2, 1,1.8, 3.2,6.4]
k = 2
# datasets = [Data_BR2000]
datasets = [Data_adult]
# datasets = [Data_NLTCS]
# datasets_name = ['NLTCS']
# datasets_name = ['BR2000']
datasets_name = ['Adult']
pridict_variable = {}
# pridict_variable['BR2000'] = ['religion']
# pridict_variable['NLTCS'] = [' eating']
pridict_variable['Adult'] = ['income']
experiment_table = pd.DataFrame(index=epsilon_list)
accuracy = 0
for i in range(0, 1):
    df = datasets[i]
    df_train, df_test = dataset_split_save(df, 1)
    # ***********************************************
    MIM = get_mim(df_train)
    d = df_train.shape[1]
    n = df_train.shape[0]
    sensitivity = get_MI_sensitivity(n, binary=False)
        # ***********************************************
    for target in pridict_variable[datasets_name[i]]:
        score_PrivBayes_list = []
        score_ELPrivBayes_list = []
        score_SAPrivBayes_2_list = []
        score_SAPrivBayes_1_list = []
        score_SAPrivBayes_0_list = []
        for epsilon in epsilon_list:
            epsilon1 = 0.3 * epsilon
            epsilon2 = 0.7 * epsilon
            # PrivBayes
            score_sum = 0
            random_node = firstnode.random_firstnode(df_train)
            for _ in range(50):
                N, score =PrivBayes.PrivBayes_PDconstruct_Bayes(df_train, k=k, first_node=random_node, matrix=MIM,
                                                           epsilon=epsilon1, sensitivity=sensitivity)

                P = PrivBayes.NoisyConditionals(N, df_train, k, epsilon2)
                df_new = PrivBayes.sampling(N, df_train, P)
                accuracy = get_svm_ac(df_new, df_test, target)
                score_sum += accuracy
                print(datasets_name[i], target, epsilon, "ac:", accuracy)
            score_PrivBayes_list.append(score_sum / 50)
            #     ElPrivBayes
            score_sum = 0
            our_node = firstnode.get_ours_first_node(matrix=MIM, df=df_train)
            for _ in range(50):
                N, score = PrivBayes_PDconstruct_Bayes(df_train, k, our_node, MIM,
                                                                      epsilon1, sensitivity)
                P = PrivBayes.NoisyConditionals(N, df_train, k, epsilon2)
                df_new = PrivBayes.sampling(N, df_train, P)
                accuracy = get_svm_ac(df_new, df_test, target)
                score_sum += accuracy
                print(datasets_name[i], target, epsilon, "ac:", accuracy)

            score_ELPrivBayes_list.append(score_sum / 50)
            # SA-PrivBayes,q=1.2
            score_sum = 0
            our_node = firstnode.get_ours_first_node(matrix=MIM, df=df_train)
            for _ in range(3):
                N, score, score_all = PrivBayes_PDconstruct_Bayes_new1(df_train, k, our_node, MIM,
                                                                       epsilon1, sensitivity)
                P = PrivBayes.NoisyConditionals_new(N, df_train, k, epsilon2, 1.2, score_all)
                df_new = PrivBayes.sampling(N, df_train, P)
                accuracy = get_svm_ac(df_new, df_test, target)
                score_sum += accuracy
                print(datasets_name[i], target, epsilon, "ac:", accuracy)

            score_SAPrivBayes_2_list.append(score_sum / 50)
            # SA-PrivBayes,q=1.1
            score_sum = 0
            our_node = firstnode.get_ours_first_node(matrix=MIM, df=df_train)
            for _ in range(3):
                N, score, score_all = PrivBayes_PDconstruct_Bayes_new1(df_train, k, our_node, MIM,
                                                                  epsilon1, sensitivity)
                P = PrivBayes.NoisyConditionals_new(N, df_train, k, epsilon2, 1.1, score_all)
                df_new = PrivBayes.sampling(N, df_train, P)
                accuracy = get_svm_ac(df_new, df_test, target)
                score_sum += accuracy
                print(datasets_name[i], target, epsilon, "ac:", accuracy)

            score_SAPrivBayes_1_list.append(score_sum /50)
            # SA-PrivBayes,q=1.0
            score_sum = 0
            our_node = firstnode.get_ours_first_node(matrix=MIM, df=df_train)
            for _ in range(50):
                N, score, score_all = PrivBayes_PDconstruct_Bayes_new1(df_train, k, our_node, MIM,
                                                                       epsilon1, sensitivity)
                P = PrivBayes.NoisyConditionals(N, df_train, k, epsilon2)
                df_new = PrivBayes.sampling(N, df_train, P)
                accuracy = get_svm_ac(df_new, df_test, target)
                score_sum += accuracy
                print(datasets_name[i], target, epsilon, "ac:", accuracy)

            score_SAPrivBayes_0_list.append(score_sum / 50)
        experiment_table['PrivBayes'] = score_PrivBayes_list
        experiment_table['ELPrivBayes'] = score_ELPrivBayes_list
        experiment_table['SAPrivBayes,q=1.2'] = score_SAPrivBayes_2_list
        experiment_table['SAPrivBayes,q=1.1'] = score_SAPrivBayes_1_list
        experiment_table['SAPrivBayes,q=1.0'] = score_SAPrivBayes_0_list
        experiment_table.to_csv(
            r'D:\Users\admin\Desktop\elpriv-bayes-master\Non-Privacy-Sturctlearning-Isum\{}-AVD-SVM-{}.csv'.format(
                datasets_name[i],datasets_name[i]))
