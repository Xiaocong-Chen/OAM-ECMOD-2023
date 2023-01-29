import random
import GlobalVariable as gl
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import math
import numpy as np
import os
#   代码大体流程是：
#   1、读取原本的数据 和 label：read_file_new()
#   2、将读入的原本数据切分成不同的视图：cut_data_new()
#   3、调用生成异常数据的函数 data_perturbation_according_percentage()
#   4、存储生成的异常数据和label

proprotion = ['0.02-0.05-0.08', '0.05-0.05-0.05', '0.08-0.05-0.02']
proprotion_str = ['2-5-8','5-5-5', '8-5-2']
namelist = ['zoo', 'wdbc', 'pima', 'letter', 'wine']
namelist_2 = ['zoo', 'wdbc', 'pima', 'letter_short', 'wine']

# 切分视图
def cut_data_new(data, data_num, feature_num): # w：这里的data是原始data吗？
    feature_num_average = int(feature_num/gl.VIEW_NUM)
    data_in_each_view = []
    feature_num_in_each_view = []
    data_new = []
    # w：把原始数据分割成 数据 和 标签
    for x in data:
        data_new.append(x[0:feature_num+1])
    for i in range(0, gl.VIEW_NUM-1):
        data_in_each_view.append([])
        for j in range(0, data_num):
            data_in_each_view[i].append(data[j][feature_num_average * i:feature_num_average * (i + 1)])
        feature_num_in_each_view.append(feature_num_average)
    data_in_each_view.append([])
    # w：因为可能存在不均分的问题，所以最后一个要单独处理？
    for j in range(0, data_num):
        data_in_each_view[gl.VIEW_NUM-1].append(data[j][feature_num_average*(gl.VIEW_NUM-1):feature_num])
    feature_num_in_each_view.append(feature_num-feature_num_average*(gl.VIEW_NUM-1))
    print('end')
    return data_new, data_in_each_view, feature_num_in_each_view


def read_file_new(file_name):
    # src_path = os.path.realpath(__file__)
    # cut_path = '/home/lab/cxc'
    # file_name = os.path.relpath(src_path,cut_path)
    f = open(file_name, 'r')
    data_str = f.readlines()
    # type_of_label = type(data_str[0])
    f.close()
    data = []
    for line in data_str:
        line = list(map(float, line.strip("\n").split(',')))
        data.append(line)
    return data

def data_normalization(data_each_view, feature_num_each_view):
    view_num = len(data_each_view)
    data_num = len(data_each_view[0])
    for i in range(0, view_num):
        for k in range(0, feature_num_each_view[i]):
            max_temp = -1
            min_temp = 1e10
            # w：获得第i个视图的中不同行第k个特征的最大最小值
            for j in range(0, data_num):
                if data_each_view[i][j][k] > max_temp:
                    max_temp = data_each_view[i][j][k]
                if data_each_view[i][j][k] < min_temp:
                    min_temp = data_each_view[i][j][k]
            interval_length = max_temp-min_temp
            if interval_length > 0:
                for j in range(0, data_num):
                    data_each_view[i][j][k] = (data_each_view[i][j][k]-min_temp)/interval_length
            else: # 如果为零，说明所有数据一样，归一化为0.5
                for j in range(0, data_num):
                    data_each_view[i][j][k] = 0.5
    return data_each_view


# 生成label为数字类型的异常数据函数
def generate_view_data(fileName, attribute_percentage, attr_class_percentage, class_percentage):
    for r in range(50):
        data = read_file_new('my_data/'+ fileName + '/' + fileName+ '.txt')
        origin_label = read_file_new('my_data/' + fileName + '/origin_label.txt')
        # data_class_outlier, label_new = shuffle_data(data)
        data_new, data_in_each_view, feature_num_in_each_view =  cut_data_new(data, len(data), len(data[0]))
        data_each_view, label, new_label = data_perturbation_according_percentage(data_in_each_view, feature_num_in_each_view, origin_label, attribute_percentage, attr_class_percentage, class_percentage)
        # 归一化的数据
        # data_each_view = data_normalization(data_in_each_view, feature_num_in_each_view)
        FatherDic = str(int(attribute_percentage * 100)) + '-' + str(int(attr_class_percentage * 100)) + '-' + str(int(class_percentage * 100))
        FatherDic = 'random-view/' + FatherDic

        filedic = FatherDic + '/data/'+ fileName +'/'+ fileName + '_' + str(gl.VIEW_NUM) + '/data' + str(r + 1)
        for i in range(len(data_each_view)):
            if os.path.isdir(filedic) == False:
                os.makedirs(filedic)
            filename = filedic + "/v" + str(i + 1) + ".txt"
            f_test = open(filename, 'w')
            for data in data_each_view[i]:
                f_test.writelines(str(data).replace('[','').replace(']','').replace(' ',''))
                f_test.writelines('\n')
            f_test.close()

        f = open(filedic + '/label_original.txt', 'w')
        f2 = open(filedic + '/label.txt', 'w')
        for item in label:
            f.writelines(str(item).replace('[','').replace(']','').replace("'",''))
            f.writelines('\n')
        for item in new_label:
            f2.writelines(str(item))
            f2.writelines('\n')
        f.close()
        f2.close()

# 生成label为字符类型的异常数据函数
def generate_view_data_char(fileName, attribute_percentage, attr_class_percentage, class_percentage):
    for r in range(50):
        data = read_file_new('my_data/'+ fileName + '/' + fileName+ '.txt')
        origin_label = read_file_char_new('my_data/' + fileName + '/origin_label.txt')
        # data_class_outlier, label_new = shuffle_data(data)
        data_new, data_in_each_view, feature_num_in_each_view =  cut_data_new(data, len(data), len(data[0]))
        data_each_view, label, new_label = data_perturbation_according_percentage(data_in_each_view, feature_num_in_each_view, origin_label, attribute_percentage, attr_class_percentage, class_percentage)
        # 归一化的数据
        # data_each_view = data_normalization(data_in_each_view, feature_num_in_each_view)
        FatherDic = str(int(attribute_percentage * 100)) + '-' + str(int(attr_class_percentage * 100)) + '-' + str(
            int(class_percentage * 100))
        filedic = FatherDic + '/data/' + fileName + '/' + fileName + '_' + str(gl.VIEW_NUM) + '/data' + str(r + 1)
        filedic = 'random-view/' + filedic
        for i in range(len(data_each_view)):
            if os.path.isdir(filedic) == False:
                os.makedirs(filedic)
            filename = filedic + "/v" + str(i + 1) + ".txt"
            f_test = open(filename, 'w')
            for data in data_each_view[i]:
                f_test.writelines(str(data).replace('[','').replace(']','').replace(' ',''))
                f_test.writelines('\n')
            f_test.close()

        f = open(filedic + '/label_original.txt', 'w')
        f2 = open(filedic + '/label.txt', 'w')
        for item in label:
            f.writelines(str(item).replace('[','').replace(']','').replace("'",''))
            f.writelines('\n')
        for item in new_label:
            f2.writelines(str(item))
            f2.writelines('\n')
        f.close()
        f2.close()



def read_file_char_new(file_name): # letter和wdbc标签是字符，重写一个方法读取
    f = open(file_name, 'r')
    data_str = f.readlines()
    # type_of_label = type(data_str[0])
    f.close()
    data = []
    for line in data_str:
        line = list(map(str, line.strip("\n").split(',')))
        data.append(line)
    return data


# 生成异常数据的函数
def data_perturbation_according_percentage(data_each_view, feature_num_each_view, label, attribute_percentage, attr_class_percentage, class_percentage):
    view_num = len(data_each_view)
    data_num = len(data_each_view[0]) # w：每个视图中的数据个数
    attribute_outlier_num = int(attribute_percentage * data_num)

    attr_class_outlier_num = int(attr_class_percentage * data_num)
    class_outlier_num = int(class_percentage * data_num)

    # outlier_num_each_type = int(data_num*gl.outlier_percent)
    # outlier_num_each_type_tmp = outlier_num_each_type

    attr_class_outlier_num = attr_class_outlier_num + (attr_class_outlier_num % 2)
    class_outlier_num = class_outlier_num + (class_outlier_num % 2)

    attribute_outlier = []

    new_label = []
    for i in range(0, data_num):
        new_label.append(label[i])

    # w：相当于产生了 attribute异常
    for i in range(0, view_num):
        attribute_outlier.append([])
        for j in range(0, attribute_outlier_num):
            attribute_outlier[i].append([]) # w: 第i个视图下加入 该视图的异常数目个新列表
            for k in range(0, feature_num_each_view[i]): # 每个新列表中加入 第i个视图下的特征数目个随机数字
                attribute_outlier[i][j].append(random.uniform(0, 1)) # 在第i个视图下的

    swap_view_num = int(view_num/2)

    # 产生 类异常
    swap_cnt = 0
    while swap_cnt < int(class_outlier_num/2): # 类异常
        random_index1 = random.randint(0, data_num-1)
        random_index2 = random.randint(0, data_num-1)
        if new_label[random_index1] == new_label[random_index2] or new_label[random_index1] == -1 or new_label[random_index2] == -1:
            continue
        # 随机选择一个视图，作为交换的视图
        random_swap_view_index = random.randint(0, view_num - 1)

        # 调试用
        before_1_tag = new_label[random_index1]
        before_2_tag = new_label[random_index2]
        before_1 = data_each_view[random_swap_view_index][random_index1]
        before_2 = data_each_view[random_swap_view_index][random_index2]

        data_each_view[random_swap_view_index][random_index1], data_each_view[random_swap_view_index][random_index2] = data_each_view[random_swap_view_index][random_index2], data_each_view[random_swap_view_index][random_index1]

        # 调试用
        after_1 = data_each_view[random_swap_view_index][random_index1]
        after_2 = data_each_view[random_swap_view_index][random_index2]
        new_label[random_index1] = -1
        new_label[random_index2] = -1
        after_1_tag = new_label[random_index1]
        after_2_tag = new_label[random_index2]
        swap_cnt += 1

    # 产生 类-属性异常
    swap_cnt = 0
    while swap_cnt < int(attr_class_outlier_num/2):
        random_index1 = random.randint(0, data_num - 1)
        random_index2 = random.randint(0, data_num - 1)
        if new_label[random_index1] == new_label[random_index2] or new_label[random_index1] == -1 or new_label[random_index2] == -1:
            continue
        # 随机选择一个视图，作为交换的视图
        random_swap_view_index2 = random.randint(0, view_num - 1)

        # 调试用
        before_attr_class_1_tag = new_label[random_index1]
        before_attr_class_2_tag = new_label[random_index2]
        before_attr_class_1 = data_each_view[random_swap_view_index2][random_index1]
        before_attr_class_2 = data_each_view[random_swap_view_index2][random_index2]

        data_each_view[random_swap_view_index2][random_index1], data_each_view[random_swap_view_index2][random_index2] = \
            data_each_view[random_swap_view_index2][random_index2], data_each_view[random_swap_view_index2][random_index1]
        new_label[random_index1] = -1
        new_label[random_index2] = -1

        # 调试用
        after_attr_class_1_tag = new_label[random_index1]
        after_attr_class_2_tag = new_label[random_index2]
        after_attr_class_1 = data_each_view[random_swap_view_index2][random_index1]
        after_attr_class_2 = data_each_view[random_swap_view_index2][random_index2]

        swap_cnt += 1
        for i in range(0, view_num): # w:另一半 是 后一半视图中的随机行的 所有列值随机赋值；相当于 attribute异常
            if i == random_swap_view_index2:
                continue
            for j in range(0, len(data_each_view[i][0])):
                data_each_view[i][random_index1][j] = random.uniform(0, 1)
                data_each_view[i][random_index2][j] = random.uniform(0, 1)

    # w: 以上截止： 完成了第三类异常的生成：class-attribute outliers

    # w：下面开始生成 attribute异常
    swap_cnt = 0
    while swap_cnt < attribute_outlier_num:
        random_index3 = random.randint(0, data_num - 1)
        if  new_label[random_index3] == -1:
            continue
        new_label[random_index3] = -1
        for i in range(0, view_num):
            data_each_view[i][random_index3] = attribute_outlier[i][swap_cnt]# 意思就是在每个视图下插入 参数异常的数据
        swap_cnt += 1

    for i in range(0, data_num):
        if new_label[i] == -1:
            new_label[i] = 1  # w：设置为1 means positive
        else:
            new_label[i] = 0

    return data_each_view, label, new_label

# 测试标签是否正确生成
def test_label_correct(file_name, iterator):
    # src_path = os.path.realpath(__file__)
    # cut_path = '/home/lab/cxc'
    # file_name = os.path.relpath(src_path,cut_path)
    file_dic = '8-5-2/data/' + file_name + '/' + file_name + '_' + str(gl.VIEW_NUM) + '/data' + str(iterator)
    file_dic = file_dic + '/label.txt'
    f = open(file_dic, 'r')
    data_str = f.readlines()
    f.close()
    data = []
    for line in data_str:
        line = int(list(map(float, line.strip("\n").split(',')))[0])
        data.append(line)
    return data

# 读取数据

def load_data_from_ready_file(data_name, iterator, propotion_index):
    data_each_view = []
    feature_num_each_view = []
    print("the view num of data is ", gl.VIEW_NUM)
    # 读取不同视图的数据
    # 默认父文件夹时random-view
    file_dir = 'random-view/' + proprotion_str[propotion_index] + '/data/' + data_name + '/' + data_name + '_' + str(gl.VIEW_NUM) + '/data' + str(iterator+1) + '/'
    for i in range(gl.VIEW_NUM):
        v_file = file_dir + 'v' + str(i+1) + '.txt'
        data_each_view.append(read_file_new(v_file)) # 将视图的数据放入 data_each_view中
        feature_num_each_view.append(len(data_each_view[i][0]))
    l_file = file_dir + 'label.txt'
    label = np.array(read_file_new(l_file))[:, 0].tolist()
    return data_each_view, feature_num_each_view, label


# 测试letter的函数
def test_cut_letter():
    letter_dic = {
        'A':0,'B':0, 'C':0, 'D':0, 'E':0, 'F':0, 'G':0, 'H':0, 'I':0, 'J':0, 'K':0,
        'L':0, 'M':0, 'N':0, 'O':0, 'P':0, 'Q':0, 'R':0, 'S':0, 'T':0, 'U':0, 'V':0,
        'W':0, 'X':0, 'Y':0, 'Z':0
    }
    label_letter = read_file_char_new('my_data/letter/origin_label.txt')
    data_letter = read_file_new('my_data/letter/letter.txt')
    f = open('my_data/letter_short/letter_short.txt', 'w')
    f2 = open('my_data/letter_short/origin_label.txt', 'w')
    k = 0
    for item in label_letter:
        if k % 10 != 0:
            k = k + 1
            continue
        label = item[0]
        f2.writelines(str(label))
        f2.writelines('\n')
        dic_num = letter_dic[label]
        letter_dic[label] = dic_num + 1
        f.writelines(str(data_letter[k]).replace('[','').replace(']',''))
        f.writelines('\n')
        k = k + 1
    f.close()
    print('the k is ', k)

    sum = 0
    for key, value in letter_dic.items():
        print(key, value)
        sum += value
    print('-------下面是各个字母的所占比率--------')
    for key, value in letter_dic.items():
        letter_dic[key] = value / sum
    for key, value in letter_dic.items():
        print(key, value)
    print('the total num is ', sum)


# 将所有视图的数据join
def data_join(data_each_view, label):
    data_all_view = []
    view_num = len(data_each_view)
    data_num = len(data_each_view[0]) # w：每个视图中的数据个数
    for j in range(0, data_num):
        data_all_view.append([])
        for i in range(0, view_num):
            data_all_view[j].extend(data_each_view[i][j]) # 每个视图中的j行，扩充到all中的下标为j的列表中去。就相当于把所有数据串起来了
        data_all_view[j].append(int(label[j]))
    return data_all_view


def generate_join_data_from_ready_file(): # 将生成的异常数据拼接起来
    for r in range(50):
        for proprotion_index in range(len(proprotion)):
            # 这里是生成数据的路径
            filedic = 'test_join_data_2/data/data' + str(r + 1)
            filedic = filedic + '/' + proprotion_str[proprotion_index] + '/'
            if os.path.isdir(filedic) == False:
                os.makedirs(filedic)
            for k in range(len(namelist_2)):
                data_each_view, feature_num_each_view, label = load_data_from_ready_file(namelist_2[k], r, proprotion_index)
                data_all = data_join(data_each_view, label)
                filename = filedic + namelist_2[k] + '-' + str(len(data_all)) + '-' + str(gl.VIEW_NUM) + '-' + proprotion[proprotion_index] + '.txt'
                f_test = open(filename, 'w')
                for i in range(len(data_all)):
                    f_test.writelines(str(data_all[i]).replace('[', '').replace(']', '').replace(' ', ''))
                    f_test.writelines('\n')
                f_test.close()
                # 生成异常数据，并将生成异常数据的值 变成新的label
                # 将数据分成多视图数据
                # 将获得的数据写入txt文件



if __name__ == '__main__':
    pass
    # 流程：



    # generate_view_data_char('letter_short', 0.02, 0.05, 0.08)
    # fileName =  'pima'
    # generate_view_data('zoo', 0.08, 0.05, 0.02)
    # generate_view_data('wine', 0.08, 0.05, 0.02)
    # generate_view_data_char('letter')




    # iterator = 0
    # data_each_view, feature_num_each_view, label = load_data_from_ready_file('letter', iterator)
    # file_dir = 'my_data/letter/letter.txt'
    # oringin_data = read_file_new(file_dir)
    # # print(oringin_data)
    #
    # for i in range(len(label)):
    #     if label[i] == 1:
    #         outiler_1 = data_each_view[0][i]
    #         outiler_2 = data_each_view[1][i]
    #         oData = oringin_data[i]
    #
    # print('enddd')
