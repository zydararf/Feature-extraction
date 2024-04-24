
import pandas as pd
from matplotlib import pyplot as plt
import Tem2
import seaborn as sns
# Tm1 = Tem2.TEMCrawl("D:\\TGS\\混合数据集 - 0314\\2602-5\\", "D:\\TGS\\混合特征集 - 0314\\2602\\")
# # data_list = Tm1.chara_crawl(1, 5)
# # print(data_list)
# # Tm1.csv_w(data_list, 3)
#
# data_list = Tm1.chara_crawl(15, 1)
# Tm1.csv_w(data_list, 13)
data = pd.read_csv(r"D:\TGS\单独气体分类\分类特征集\10.csv",names=None,encoding="gb2312")  # 本地加载
print(data)
sns.heatmap(data.corr(), linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True)
plt.show()
