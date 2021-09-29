import xlrd
import xlwt

wb = xlrd.open_workbook("C9_1shot.xls")
sh = wb.sheet_by_index(0)  # 第一个表
rowD1 = sh.row_values(1)  # 读取一行的数据
#colData= sh.col_values(0)  # 读取一列的数据
rowD2 = sh.row_values(2)
rowD3 = sh.row_values(3)
rowD4 = sh.row_values(4)
rowD5 = sh.row_values(5)
rowD6 = sh.row_values(6)
rowD7 = sh.row_values(7)
rowD8 = sh.row_values(8)
rowD9 = sh.row_values(9)
rowD10 = sh.row_values(10)
rowD11 = sh.row_values(11)
rowD12 = sh.row_values(12)
rowD13 = sh.row_values(13)
rowD14 = sh.row_values(14)
rowD15 = sh.row_values(15)
rowD16 = sh.row_values(16)
rowD17 = sh.row_values(17)
rowD18 = sh.row_values(18)
rowD19 = sh.row_values(19)

row = len(rowD4)  # 读取行数
#col = len(rowName)  # 读取列数
all=[]
for item in rowD1:
    all.append(item)
for item2 in rowD2:
    all.append(item2)
for item3 in rowD3:
    all.append(item3)
for item4 in rowD4:
    all.append(item4)
for item5 in rowD5:
    all.append(item5)
for item6 in rowD6:
    all.append(item6)
for item7 in rowD7:
    all.append(item7)
for item8 in rowD8:
    all.append(item8)
for item9 in rowD9:
    all.append(item9)
for item10 in rowD10:
    all.append(item10)
for item11 in rowD11:
    all.append(item11)
for item12 in rowD12:
    all.append(item12)
for item13 in rowD13:
    all.append(item13)
for item14 in rowD14:
    all.append(item14)
for item15 in rowD15:
    all.append(item15)
for item16 in rowD16:
    all.append(item16)
for item17 in rowD17:
    all.append(item17)
for item18 in rowD18:
    all.append(item18)
for item19 in rowD19:
    all.append(item19)

index = 1;
workbook = xlwt.Workbook()
sheet = workbook.add_sheet("sheet 1")
col = 0;
for i in range(0,len(all),5):
    #strlist = colData[i].split(',')
    #strl2=strlist[0].split('(')
    str=all[i]
    sheet.write(index, col, str)
    col = col + 1
    if col == 256:
        index = index + 1;
        col = 0

workbook.save("Rep_1shot_500.xls")