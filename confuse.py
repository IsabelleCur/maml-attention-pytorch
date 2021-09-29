#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import xlrd
import xlwt

wb = xlrd.open_workbook("RML1_cp.xlsx")
sh = wb.sheet_by_index(0)  # 第一个表
rowD0 = sh.row_values(0)
rowD1 = sh.row_values(1)  # 读取一行的数据
rowD2 = sh.row_values(2)
rowD3 = sh.row_values(3)
rowD4 = sh.row_values(4)
con=[]
con.append(rowD0)
con.append(rowD1)
con.append(rowD2)
con.append(rowD3)
con.append(rowD4)
print(con)
confusion = np.array((rowD0,rowD1,rowD2,rowD3,rowD4))
print(confusion)
# 热度图，后面是指定的颜色块，可设置其他的不同颜色
plt.imshow(confusion, cmap=plt.cm.Blues)
# ticks 坐标轴的坐标点
# label 坐标轴标签说明
indices = range(len(confusion))
# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
#plt.xticks(indices, [0, 1, 2])
#plt.yticks(indices, [0, 1, 2])
plt.xticks(indices, ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK'])
plt.yticks(indices, ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK'])

plt.colorbar()

plt.xlabel('Predict Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix: 1-shot CAMEL(%)')

# plt.rcParams两行是用于解决标签不能显示汉字的问题
#plt.rcParams['font.sans-serif']=['SimHei']
#plt.rcParams['axes.unicode_minus'] = False

# 显示数据
for first_index in range(len(confusion)):    #第几行
    for second_index in range(len(confusion[first_index])):    #第几列
        plt.text(second_index-0.25, first_index+0.05, confusion[first_index][second_index])
# 在matlab里面可以对矩阵直接imagesc(confusion)
# 显示
#plt.show()
plt.savefig("CAMEL1.png",dpi=500,bbox_inches = 'tight')