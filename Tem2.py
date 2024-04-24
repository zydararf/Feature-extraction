# encoding:utf-8

'''
变温特征提取Python实现
'''

import csv
import numpy as np
import scipy
from scipy.integrate import simps  # 用于计算积分
import matplotlib.pyplot as plt  # 用于画图
import os
import xlwt
import pandas as pd
from numpy import *
import csv
import seaborn as sns
import fsspec
import xlrd



def read(x):
    '''
    读取函数
    '''
    y = os.listdir(x)
    y.sort(key=lambda x: int(x[6:-6]))
    return y


def read1(x):
    '''
    读取函数
    '''
    y = os.listdir(x)
    y.sort(key=lambda x:int(x[:-4]))
    return y


class TEMCrawl:
    '''
    神经网络回归与分类的差别在于：
    1. 输出层不需要再经过激活函数
    2. 输出层的 w 和 b 更新量计算相应更改
    '''

    def __init__(self, filename, filename1):

        #文件名
        self.filename = filename
        self.filename1 = filename1
        # 文件内容
        self.context = read(filename)
        self.context1 = read1(filename1)
        # 文件长度
        self.num = len(self.context)
        self.num1 = len(self.context1)

    def excel_one_line_to_list(self, a, b, c, d):
        df = pd.read_excel(self.filename + self.context[d], usecols=[c], names=None, skiprows=a, nrows=b)  # 读取项目名称列,不要列名
        df_li = df.values.tolist()
        x = []
        for s_li in df_li:
            x.append(s_li[0])
        return x

    # 定义计算离散点导数的函数
    def max_deriv(self, y):
        # x, y的类型均为列表，一阶导数
        diff_y = []  # 用来存储y列表中的两数之差
        deriv = []
        for i, j in zip(y[0::], y[2::]):
            diff_y.append((j - i) / 2)
        deriv.insert(0, diff_y[0])  # (左)端点的导数即为与其最近点的斜率
        deriv.append(diff_y[-1])  # (右)端点的导数即为与其最近点的斜率
        der = max(deriv)  # 用来存储一阶导数
        return der

    def max_2nd_deriv(self, y):
        # x, y的类型均为列表，二阶导数
        diff_y = []  # 用来存储y列表中的两数之差
        deriv = []
        for i, j in zip(y[0::], y[2::]):
            diff_y.append((j - i) / 2)
        deriv.insert(0, diff_y[0])  # (左)端点的导数即为与其最近点的斜率
        deriv.append(diff_y[-1])  # (右)端点的导数即为与其最近点的斜率
        diff_y2 = []  # 用来存储y列表中的两数之差
        deriv2 = []
        for i, j in zip(diff_y[0::], diff_y[2::]):
            diff_y2.append((j - i) / 2)
        deriv2.insert(0, diff_y2[0])  # (左)端点的导数即为与其最近点的斜率
        deriv2.append(diff_y2[-1])  # (右)端点的导数即为与其最近点的斜率
        der = max(deriv2)  # 用来存储一阶导数
        return der

    def cal_integral(self, x, y):
        # 用于计算积分
        integrals = []
        # cal = 0
        for i in range(len(y)):  # 计算梯形的面积，由于是累加，所以是切片"i+1"
            integrals.append(scipy.integrate.trapz(y[:i + 1], x[:i + 1]))
        return integrals

    def curvature(self, y):
        # diff_y = []  # 用来存储y列表中的两数之差
        deriv = []
        for i, j in zip(y[0::], y[1::]):
            deriv.append((j - i))
        deriv.insert(0, deriv[0])  # (左)端点的导数即为与其最近点的斜率
        deriv.append(deriv[-1])  # (右)端点的导数即为与其最近点的斜率

        deriv2 = []
        for i, j in zip(deriv[0::], deriv[1::]):
            deriv2.append((j - i) )
        deriv2.insert(0, deriv2[0])  # (左)端点的导数即为与其最近点的斜率
        deriv2.append(deriv2[-1])  # (右)端点的导数即为与其最近点的斜率

        curvature_values = []
        for i in range(1, len(deriv)-1):
            curvature = abs(deriv2[i]) / ((1 + deriv[i] ** 2) ** (3 / 2))
            curvature_values.append(curvature)
        return curvature_values

    def chara_crawl(self, a, b):
        xu = []
        y_all = []
        y1_ave = []
        y2_ave = []
        y1_max = []
        y2_max = []
        y1_der = []
        y2_der = []
        y1_2nd_der = []
        y2_2nd_der = []
        y1_cal = []
        y2_cal = []
        y1_average_curvature = []
        y2_average_curvature = []
        data_list = []
        print(self.num)
        for i in range(self.num):
            for j in range(a):
                for k in range(b):
                    x = self.excel_one_line_to_list(0, 60, 0, i)
                    y = self.excel_one_line_to_list(60 * j, 60 * (j + 1), k + 1, i)
                    y_all.append(y)
                    y1_ave.append(average(y[50:60]))  # 低温广谱响应均值
                    y2_ave.append(average(y[20:30]))  # 高温广谱响应均值
                    y1_max.append(max(y[35:40]))   # 低温特征响应最大值
                    y2_max.append(max(y[0:20]))    # 高温特征响应最大值
                    y1_der.append(abs(self.max_deriv(y[30:60])))  # 低温特征响应变化率
                    y2_der.append(abs(self.max_deriv(y[0:30])))  # 高温特征响应变化率
                    y1_2nd_der.append(abs(self.max_2nd_deriv(y[30:60])))  # 低温特征响应凹凸程度
                    y2_2nd_der.append(abs(self.max_2nd_deriv(y[0:30])))  # 高温特征响应凹凸程度
                    y1_cal.append(max(self.cal_integral(x[35:40], y[35:40])))  # 低温特征响应积分
                    y2_cal.append(max(self.cal_integral(x[3:10], y[3:10])))  # 高温特征响应积分
                    y1_average_curvature.append(average(y[35:40]))  # 低温曲率均值
                    y2_average_curvature.append(average(y[3:10]))  # 高温曲率均值
                    xu.append(i)
        for y1_ave, y2_ave, y1_max, y2_max, y1_der, y2_der, y1_2nd_der, y2_2nd_der, y1_cal, y2_cal, y1_average_curvature, y2_average_curvature, xu in zip(
                y1_ave, y2_ave,
                y1_max, y2_max, y1_der,
                y2_der,
                y1_2nd_der,
                y2_2nd_der,
                y1_cal, y2_cal, y1_average_curvature, y2_average_curvature,
                xu):
            z = {'average1': y1_ave, 'average2': y2_ave, 'max1': y1_max, 'max2': y2_max, 'derivative1': y1_der,
                 'derivative2': y2_der,
                 '2nd_derivative1': y1_2nd_der, '2nd_derivative2': y2_2nd_der, 'integral1': y1_cal, 'integral2': y2_cal,
                 'curvature1': y1_average_curvature, 'curvature2': y2_average_curvature,
                 'category': xu}
            data_list.append(z)
        return data_list

    def csv_w(self, x, y):
        with open(self.filename1 + self.context1[y], 'w', newline='') as f_c_csv:
            writer = csv.writer(f_c_csv)
            writer.writerow(['average1', 'average2', 'max1', 'max2', 'derivative1', 'derivative2', '2nd_derivative1',
                             '2nd_derivative2', 'integral1', 'integral2',   'curvature1', 'curvature2', 'category'])
            for nl in x:
                writer.writerow(nl.values())

    pass
