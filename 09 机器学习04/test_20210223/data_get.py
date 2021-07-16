import xlrd
import numpy as np

def read_excel():
    data_list = []
    #打开文件
    workBook = xlrd.open_workbook("data/COPD.xls")
    #按照索引号获取sheet内容
    sheet1 = workBook.sheet_by_index(0)
    for i in range(sheet1.nrows):
        if i != 0 and "" not in sheet1.row_values(i):
            data_list.append(sheet1.row_values(i))
    return np.array(data_list)


if __name__ == '__main__':
    x = read_excel()[:,2:52]
    y = read_excel()[:,1]
    print(x)
    print(y)
    x = x.astype(np.float)
    y = y.astype(np.float)
    print(x)
    print(y)
