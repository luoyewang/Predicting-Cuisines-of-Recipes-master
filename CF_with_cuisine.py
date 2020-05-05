import numpy as np
import math
import pandas as pd
import random
import time
from sklearn.decomposition import PCA
from collections import defaultdict


# current_mode = 'PCA' or 'BASE'
current_mode = 'PCA'
current_similarity_method = 'PMI'
N = 10
K = 50
num_test_points = 'ALL'


# 计算cosine similarity
def CS_f(r_i, r_j):
    CS_i_j = np.dot(r_i.T, r_j) / ((math.sqrt(np.dot(r_i.T, r_i)))
                                   * ((math.sqrt(np.dot(r_j.T, r_j)))))
    return CS_i_j


def ACS_f(r_i, r_j):
    a = 0.05
    ACS_i_j = np.dot(r_i.T, r_j) / ((math.pow(np.dot(r_i.T, r_i), a))
                                    * (math.pow(np.dot(r_j.T, r_j), 1 - a)))
    return ACS_i_j


def JS_f(r_i, r_j, mode=None):
    if mode == 'BASE':
        U = r_i + r_j
        N = 0
        for u in U:
            if u > 0:
                N = N + 1
        JS_i_j = np.dot(r_i.T, r_j) / N

    elif mode == 'PCA':
        JS_i_j = np.dot(r_i.T, r_j) / (np.dot(r_i.T, r_i) * np.dot(r_j.T, r_j) - np.dot(r_i.T, r_j))
    return JS_i_j


def PMI_f(r_i, r_j, mode=None, v_i=None, v_j=None):
    if mode == 'BASE':
        p_i_j = np.dot(r_i.T, r_j) / len(r_i)
        p_i = np.dot(r_i.T, r_i) / len(r_i)
        p_j = np.dot(r_j.T, r_j) / len(r_j)
        if p_i_j == 0:
            PMI_i_j = 0
        else:
            PMI_i_j = math.log((p_i_j / (p_i * p_j)))

    elif mode == 'PCA':
        p_i_j = np.dot(r_i.T, r_j) / len(r_i)
        p_i = np.dot(v_i.T, v_i) / len(v_i)
        p_j = np.dot(v_j.T, v_j) / len(v_j)
        PMI_i_j = p_i_j / (p_i * p_j)

    return PMI_i_j


def SIM_matrix(R, similarity_method):
    SIM = np.zeros([R.shape[1], R.shape[1]])
    if similarity_method == 'CS':
        for i in range(SIM.shape[1]):
            for j in range(SIM.shape[1]):
                SIM[i, j] = CS_f(R.iloc[:, i], R.iloc[:, j])
    elif similarity_method == 'ACS':
        for i in range(SIM.shape[1]):
            for j in range(SIM.shape[1]):
                SIM[i, j] = ACS_f(R.iloc[:, i], R.iloc[:, j])
    elif similarity_method == 'JS':
        for i in range(SIM.shape[1]):
            for j in range(SIM.shape[1]):
                SIM[i, j] = JS_f(R.iloc[:, i], R.iloc[:, j], mode=current_mode)
    elif similarity_method == 'PMI':
        if current_mode == 'PCA':
            base_data_for_pca_PMI = np.load('base_data_for_pca_PMI.npy')
            base_data_for_pca_PMI = pd.DataFrame(base_data_for_pca_PMI)
            for i in range(SIM.shape[1]):
                for j in range(SIM.shape[1]):
                    SIM[i, j] = PMI_f(R.iloc[:, i], R.iloc[:, j], mode=current_mode,
                                      v_i=base_data_for_pca_PMI.iloc[:, i], v_j=base_data_for_pca_PMI.iloc[:, j])
        else:
            for i in range(SIM.shape[1]):
                for j in range(SIM.shape[1]):
                    SIM[i, j] = PMI_f(R.iloc[:, i], R.iloc[:, j], mode=current_mode)

    else:
        print('Wrong similarity function choice')
    return SIM


def TOP_k_index(Sim_matrix, Row, K_ingredient):
    Sim_matrix_i = Sim_matrix[Row, :]
    Index = np.argsort(Sim_matrix_i)[::-1]
    Top_k = Index[:K_ingredient]
    Top_k = Top_k.tolist()

    if Row in Top_k:
        Top_k.remove(Row)
        Top_k.append(Index[K_ingredient])

    return Top_k


def random_delete_index(df_renamed_rev_test, test_index):
    rand_delete_index_dict = dict()

    random.seed(233)

    for i in test_index:
        recipe_vec = df_renamed_rev_test.iloc[i]
        index = [i for i in range(len(recipe_vec)) if recipe_vec[i] == 1]
        rand_delete_index_dict[i] = random.sample(index, 1)[0]

    return rand_delete_index_dict


def TOP_ingredient(df_renamed_rev_test, Recipe_index, rand_delete_index_dict, N, Sim_matrix, Sim_ingredient_list):
    P_recipe = []
    Top_N_ingredient_name = []
    All_ingredient_name = []
    Recipe_vec = df_renamed_rev_test.iloc[Recipe_index]  # 获得某一个菜的向量表示
    Delete_index = rand_delete_index_dict[Recipe_index]  # 获得要随机删除的菜原料的序号
    Delete_name = df_renamed_rev_test.columns.values[Delete_index]   # 获得要删除的菜原料的名字

    Recipe_vec[Delete_index] = 0  # 随机挖空

    for i in range(df_renamed_rev_test.shape[1]):
        Numerator = 0
        Denominator = 0
        for j in Sim_ingredient_list[i]:
            Numerator = Numerator + Recipe_vec[j] * Sim_matrix[i, j]
            Denominator = Denominator + Sim_matrix[i, j]
        P_recipe_i = Numerator / Denominator  # 获得测试的菜跟第i个原料的适合程度
        P_recipe.append(P_recipe_i)

    Ingredient_index = np.argsort(P_recipe)[::-1]
    Top_N_ingredient = Ingredient_index[:N]  # 获得相似度靠前的前n个原料

    Recipe_vec[Delete_index] = 1  # 恢复挖空

    for s in Ingredient_index:
        All_ingredient_name.append(
            df_renamed_rev_test.columns.values[s])  # 按从适合程度高到低排序所有成分

    for n in Top_N_ingredient:
        Top_N_ingredient_name.append(
            df_renamed_rev_test.columns.values[n])  # 获得这个菜前n个适合原料的序列号

    for l in range(df_renamed_rev_test.shape[1]):
        if All_ingredient_name[l] == Delete_name:
            Ingredient_rank = l
    Ingredient_rank = Ingredient_rank + 1

    return Top_N_ingredient_name, Delete_name, Ingredient_rank


def test_f(df_renamed_rev_test, test_index, N, K, S):
    Rank = []
    N_ingredient_list = []

    True_label = 0
    Whole_label = len(test_index)

    for i in range(df_renamed_rev_test.shape[1]):
        N_ingredient_list.append(TOP_k_index(S, i, K))

    # get rand_delete_index_dict
    rand_delete_index_dict = random_delete_index(df_renamed_rev_test, test_index)

    for Recipe_i in test_index:
        Predict_name, True_name, Predict_rank = TOP_ingredient(
            df_renamed_rev_test, Recipe_i, rand_delete_index_dict, N, S, N_ingredient_list)
        Rank.append(Predict_rank)
        if True_name in Predict_name:
            True_label = True_label + 1

    Recall = True_label / Whole_label

    return Recall, np.mean(Rank), np.median(Rank)


def data_split():
    df_renamed_rev = pd.read_csv("processed_data_with_cuisine.csv")
    df_renamed_rev.drop('Unnamed: 0', axis=1, inplace=True)

    np.random.seed(1)
    df_renamed_rev_shuf = np.random.permutation(df_renamed_rev)
    df_renamed_rev_shuf = pd.DataFrame(df_renamed_rev_shuf)
    df_renamed_rev_shuf.columns = df_renamed_rev.columns.values
    df_renamed_rev_train = df_renamed_rev_shuf.iloc[:34625]
    df_renamed_rev_test = df_renamed_rev_shuf.iloc[34625:]
    index = np.linspace(0, 3846, 3847, endpoint=True, dtype=int)
    df_renamed_rev_test.index = index.tolist()
    return df_renamed_rev_train, df_renamed_rev_test


def gen_IC_mat(df_renamed_rev_train):
    # {I: {C: count} }
    C_count_dict = defaultdict(int)
    C_total_count_dict = defaultdict(int)

    cuisine = df_renamed_rev_train['cuisine']
    for i in cuisine:
        C_total_count_dict[i] += 1

    for i in df_renamed_rev_train.columns[:-1]:
        u_vec = df_renamed_rev_train[i]
        I_C_dict = defaultdict(dict)
        for num in range(len(u_vec)):
            if u_vec[num] == 1:
                C_count_dict[cuisine[num]] += 1
        I_C_dict[i] = C_count_dict

    IC_mat = pd.DataFrame(I_C_dict).fillna(0)

    # just for df.iloc(only valid for number)
    name_index_dict = dict()
    ind = 0
    for c in IC_mat.index:
        name_index_dict[c] = ind
        ind += 1

    # normalization to a good unit
    for cuisine in IC_mat.index:
        IC_mat.iloc[name_index_dict[cuisine]] = IC_mat.iloc[name_index_dict[cuisine]]/C_total_count_dict[cuisine]*df_renamed_rev_train.shape[0]/100

    return IC_mat


def tfidf_transfer(IC_mat):
    for i in IC_mat.columns:
        v = IC_mat[i]
        zero_count = 0
        for item in v:
            if item == 0:
                zero_count += 1
        df = IC_mat.shape[0]-zero_count
        idf = IC_mat.shape[0]/df
        IC_mat[i] = IC_mat[i] * idf
    return IC_mat


def projection(IC_mat):
    IC_mat_projected = pd.DataFrame.copy(IC_mat, deep=True)
    for col in IC_mat.columns:
        max_num = IC_mat[col].max()
        min_num = IC_mat[col].min()
        new_col = [(c-min_num)/(max_num-min_num) for c in IC_mat[col]]
        IC_mat_projected[col] = new_col
    return IC_mat_projected


if __name__ == '__main__':
    df_renamed_rev_train, df_renamed_rev_test = data_split()
    df_renamed_rev_test.drop('cuisine', axis=1, inplace=True)  # todo: 删除test的cuisine列
    IC_mat = gen_IC_mat(df_renamed_rev_train)
    IC_mat = tfidf_transfer(IC_mat)

    # start to do [0, 1] projection for concatenate method
    IC_mat_projected = projection(IC_mat)
    np.save('cuisine_vec.npy', IC_mat_projected)

    if current_mode == 'PCA':
        pca = PCA(n_components=20)
        new_data = pca.fit_transform(df_renamed_rev_train.T)
        df_new_data = pd.DataFrame(new_data.T)
    elif current_mode == 'BASE':
        df_new_data = df_renamed_rev_train

    if num_test_points == 'ALL':
        test_index = list(range(df_renamed_rev_test.shape[0]))
    else:
        random.seed(666)
        test_index = random.sample(list(df_renamed_rev_test.index), num_test_points)

    start = time.process_time()

    S = SIM_matrix(R=df_new_data, similarity_method=current_similarity_method)

    recall_test, mean_rank_test, median_rank_test = test_f(
        df_renamed_rev_test, test_index, N, K, S)
    print('Recall rate is :%.2f%%' % (recall_test * 100))
    print('The mean of rank is :%.2f' % (mean_rank_test))
    print('The median of rank is :%.2f' % (median_rank_test))

    end = time.process_time()
    print('Running time: %s Seconds' % (end - start))