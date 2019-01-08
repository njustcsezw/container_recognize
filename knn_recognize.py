from numpy import *
from os import listdir
import operator
import time


# kNN实现函数
def classify0(inx, dataset, labels, k):
    # 求出样本集的行数，也就是labels标签的数目
    data_set_size = dataset.shape[0]

    # 构造输入值和样本集的差值矩阵
    diff_mat = tile(inx, (data_set_size, 1)) - dataset

    # 计算欧式距离
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5

    # 求距离从小到大排序的序号  
    sorted_dist_indicies = distances.argsort()

    # 对距离最小的k个点统计对应的样本标签
    class_count = {}
    for i in range(k):
        # 取第i+1近邻的样本对应的类别标签
        vote_ilabel = labels[sorted_dist_indicies[i]]
        # 以标签为key，标签出现的次数为value将统计到的标签及出现次数写进字典
        class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1

        # 对字典按value从大到小排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    # 返回排序后字典中最大value对应的key
    return sorted_class_count[0][0]


# 函数运行耗时统计函数
def time_me(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        fn(*args, **kwargs)
        print("\n%s cost %s second" % (fn.__name__, time.clock() - start))

    return _wrapper


# 图像转换函数（60*70图像转换为1*4200向量）
def img2vector(filename):
    # 初始化待返回的向量
    return_vect = zeros((1, 4200))

    fr = open(filename, 'rb')
    for i in range(70):
        # 每次读取一行内容，以字符串形式存储
        line_str = fr.readline()
        # print(line_str)
        for j in range(60):
            return_vect[0, 60 * i + j] = int(line_str[j])
    return return_vect


# 手写数字识别测试函数
# @time_me
def handwritingClassTest():
    # 初始化类别标签为空列表
    hw_labels = []
    # 列出给定目录下所有训练数据的文件名
    training_file_list = listdir('training_sets_matrix')
    m = len(training_file_list)

    # 初始化m个图像的训练矩阵
    training_mat = zeros((m, 4200))

    # 遍历每一个训练数据
    for i in range(m):
        # 取出一个训练数据的文件名
        file_name_str = training_file_list[i]
        # 去掉该训练数据的后缀名.txt
        file_str = file_name_str.split('.')[0]
        # 取出代表该训练数据类别的数字
        class_num_str = int(file_str.split('_')[0])
        # 将代表该训练数据类别的数字存入类别标签列表
        hw_labels.append(class_num_str)
        # 调用图像转换函数将该训练数据的输入特征转换为向量并存储
        training_mat[i, :] = img2vector('training_sets_matrix/' + file_name_str)

    # 列出给定目录下所有测试数据的文件名
    test_file_list = listdir('test_sets_matrix')

    # 初始化测试犯错的样本个数
    #error_count = 0.0

    m_test = len(test_file_list)
    # 遍历每一个测试数据
    num_to_char = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
                   '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
                   '10': 'A', '11': 'B', '12': 'C', '13': 'D', '14': 'E',
                   '15': 'F', '16': 'G', '17': 'H', '19': 'J', '20': 'K',
                   '21': 'L', '22': 'M', '23': 'N', '25': 'P', '27': 'R',
                   '28': 'S', '29': 'T', '30': 'U', '31': 'V', '32': 'W',
                   '33': 'X', '34': 'Y', '35': 'Z', }

    classifier_answer = ''
    adjust_classifier_answer = ''

    for i in range(m_test):
        # 取出一个测试数据的文件名
        file_name_str = test_file_list[i]
        print(file_name_str)

        # 调用图像转换函数将该测试数据的输入特征转换为向量
        vector_under_test = img2vector('test_sets_matrix/' + file_name_str)

        # 调用k-NN简单实现函数，并返回分类器对该测试数据的分类结果
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        classifier_answer += num_to_char[str(classifier_result)]

        print('the classifier came back with: ' + num_to_char[str(classifier_result)])

    if len(classifier_answer) > 11:
        adjust_classifier_answer = str(classifier_answer[6:]) + str(classifier_answer[0:6])

    print('\nthe classifier is ' + adjust_classifier_answer + '\n')
    # print("\nthe total number of error is: %d" % error_count, end='')
    # 输出分类器错误率
    # print("\nthe total error rate is: %f" % (error_count / float(m_test)))
    return adjust_classifier_answer

#handwritingClassTest()
