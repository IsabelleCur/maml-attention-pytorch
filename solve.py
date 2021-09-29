import xlrd
import xlwt

wb = xlrd.open_workbook("reptile_1shot.xls")
sh = wb.sheet_by_index(0)  # 第一个表
#rowD1 = sh.row_values(1)  # 读取一行的数据
colData= sh.col_values(0)  # 读取一列的数据


row = len(colData)  # 读取行数
#col = len(rowName)  # 读取列数
#all=[]
#for item in rowD1:
#    all.append(item)

index = 1;
workbook = xlwt.Workbook()
sheet = workbook.add_sheet("sheet 1")
col = 0;
for i in range(0,row,5):
    strlist = colData[i].split(',')
    strl2=strlist[0].split('(')
    str=strl2[1]
    sheet.write(index, col, str)
    col = col + 1
    if col == 256:
        index = index + 1;
        col = 0

workbook.save("Rep0_1shot0_500.xls")