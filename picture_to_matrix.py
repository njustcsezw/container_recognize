from PIL import Image
import numpy as np
from os import listdir
import os


def ImageToMatrix(file_set_name, file_name, txt_name):
    # 读取图片
    im = Image.open(file_name)

    width, height = im.size
    data = im.getdata()
    #print(data)
    data = np.matrix(data, dtype=float)
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data, (height, width))

    # 去掉该训练数据的后缀名.txt
    file_str = txt_name.split('.')[0]
    # 取出代表该训练数据类别的数字
    class_num_str = int(file_str.split('_')[0])
    num_pics = int(file_str.split('_')[0])

    save_path = file_set_name + '_matrix\\' + str(class_num_str) + '_' + str(num_pics) + '.txt'
    #将得到的二维矩阵保存在txt
    with open(save_path, 'a') as inf:
        for i in range(new_data.shape[0]):
            for j in range(new_data.shape[1]):
                if new_data[i, j] > 0.0:
                    inf.write('1')
                else:
                    inf.write('0')
            inf.write('\n')


def run_imagetomatrix(file_set_name):

    #清空即将使用的文件夹
    path = 'E:\\workpython2\\container_recognize\\test_sets_matrix'
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            for f in os.listdir(path_file):
                path_file2 = os.path.join(path_file, f)
                if os.path.isfile(path_file2):
                    os.remove(path_file2)

    # 列出给定目录下所有训练数据的文件名
    training_file_list = listdir(file_set_name)
    m = len(training_file_list)

    # 遍历每一个数据
    for i in range(m):
        # 取出一个数据的文件名
        file_name_str = training_file_list[i]
        #使用转化函数
        file_name = file_set_name + '\\' + file_name_str
        ImageToMatrix(file_set_name, file_name, file_name_str)


run_imagetomatrix('test_sets')
