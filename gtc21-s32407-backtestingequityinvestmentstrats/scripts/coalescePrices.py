import csv
import os
import sys

if len(sys.argv)==1:
    dir = '/home/mark/FinAnalytics/MVO3.2021.02/NYSE/'
    dir = '/home/mark/FinAnalytics/MVO3.2021.02/NASDAQ/'
elif len(sys.argv)==2:
    dir = sys.argv[1]
    print(dir)
else: exit()

os.chdir(dir)
fileList = os.listdir(dir)
print('files:', fileList)
print('file count:', len(fileList))

prices = []; lab = []
for file in fileList:
    print(file)
    f = open(file, 'r')
    x = [line.split('\n')[0] for line in f.readlines()]
    if (file[0:6] == 'cached') and (x[1] != 'NA'):
        l = list(map(float,x[1:]))
        prices.append(l)
        if file == 'cachedGME.csv': print(l)
        lab.append(file[6:][:-4])

print('*****')
print(len(lab))
prices = [list(x) for x in zip(*prices)]

D = len(prices)
print(D)
length = len(prices[0])
print(length)

outFile = open(dir+'/'+'prices.csv','w')
with outFile:
    writer = csv.writer(outFile)
    writer.writerow(lab)
    writer.writerows(prices)

