# encoding=utf8
import sys


int_type='uint32'
inf = 'temp.txt'
outf = 'clicker_fea.proto'
with open(inf, mode='r', encoding='utf-8') as fp:
    for index, line in enumerate(fp):
        arr = line.strip().split('\t')
        if len(arr) == 2:

            name, msg = arr[0].strip(), arr[1].strip()
            temp = "\t{} {}={};\t//{}".format(int_type, name, index+1, msg)
            print(temp)

    
    print('个数：',index+1)
