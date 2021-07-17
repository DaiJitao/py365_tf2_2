# encoding=utf8
import sys
import re

inf = 'temp.txt'

with open(inf, encoding='utf-8', mode='r') as fp:
    
    for line in fp:
        d = line.strip().strip(',').replace(')', '').strip().split(',')
        p = '\d{1,3}'
        s = re.findall(p, d[0])
        if s != []:
            print(s[0],  '\t', d[1])