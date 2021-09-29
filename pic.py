import pandas as pd
import xlwt
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np

jddf = pd.read_csv('speed_all_nchw_find.csv', sep=',', header=None,
                   names=['para', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'find1', 'find2', 'find3', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7'])

# 设置X轴和Y轴取值

workbook = xlwt.Workbook()
sheet = workbook.add_sheet("sheet 1")
para = jddf['para'].tolist()
find1 = jddf['find1'].tolist()
find2 = jddf['find2'].tolist()
find3 = jddf['find3'].tolist()
print(len(para))
for i in range(2,len(para)):
    #print(i)
    if i==21 or i==45 or i==46 or i==73 or i==75 or i==88 or i==103 or i==512 or i==516 or i==603 or i==651:
        continue
    sheet.write(i-2, 0, para[i])
    f = find1[i]
    s1list = f.split('======= end print forward rsts info ======')
    s2list = s1list[1].split('======= start print bwd-data rsts info ======')
    time1 = s2list[0].split('time = ')
    time2 = time1[1].split(', memory')
    sheet.write(i-2, 1, time2[0])

    s3list = s2list[1].split('======= end print bwd-data rsts info ======')
    s4list = s3list[1].split('======= start print bwd-weight rsts info ======')
    time3 = s4list[0].split('time = ')
    time4 = time3[1].split(', memory')
    sheet.write(i-2, 2, time4[0])

    s5list = s4list[1].split('======= end print bwd-weight rsts info ======')
    time5 = s5list[1].split('time = ')
    time6 = time5[1].split(', memory')
    sheet.write(i-2, 3, time6[0])

for i in range(2,len(para)):
    #print(i)
    if i==21 or i==27 or i==28 or i==29 or i==30 or i==45 or i==46 or i==58 or i==59 or i==60 or i==61 or i==62 or i==63 or i==64 or i==65 or i==66 or i==73 or i==75 or i==81 or i==82 or i==83:
        continue
    f = find2[i]
    s1list = f.split('======= end print forward rsts info ======')
    if len(s1list)==1:
        continue
    s2list = s1list[1].split('======= start print bwd-data rsts info ======')
    time1 = s2list[0].split('time= ')
    time2 = time1[1].split(', memory = ')
    sheet.write(i-2, 4, time2[0])

    s3list = s2list[1].split('======= end print bwd-data rsts info ======')
    s4list = s3list[1].split('======= start print bwd-weight rsts info ======')
    time3 = s4list[0].split('time= ')
    time4 = time3[1].split(', memory = ')
    sheet.write(i-2, 5, time4[0])

    s5list = s4list[1].split('======= end print bwd-weight rsts info ======')
    time5 = s5list[1].split('time= ')
    time6 = time5[1].split(', memory = ')
    sheet.write(i-2, 6, time6[0])

for i in range(2,len(para)):
    print(i)

    f = find3[i]
    s1list = f.split('======= end print forward rsts info ======')
    if len(s1list)==1:
        continue
    s2list = s1list[1].split('======= start print bwd-data rsts info ======')
    time1 = s2list[0].split('time= ')
    time2 = time1[1].split(', memory = ')
    sheet.write(i-2, 7, time2[0])

    s3list = s2list[1].split('======= end print bwd-data rsts info ======')
    s4list = s3list[1].split('======= start print bwd-weight rsts info ======')
    time3 = s4list[0].split('time= ')
    time4 = time3[1].split(', memory = ')
    sheet.write(i-2, 8, time4[0])

    s5list = s4list[1].split('======= end print bwd-weight rsts info ======')
    time5 = s5list[1].split('time= ')
    time6 = time5[1].split(', memory = ')
    sheet.write(i-2, 9, time6[0])

workbook.save("result.xls")


# 折线图
#line1, = plt.plot(np.arange(0,200), jddf['line1'], color='purple', lw=0.5, ls='-', ms=4)


