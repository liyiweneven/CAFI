import csv

# 打开文件，用with打开可以不用去特意关闭file了，python3不支持file()打开文件，只能用open()
with open("dk0519_1.csv", "r", encoding="utf-8") as csv_file:
    # 读取csv文件，返回的是迭代类型
    read = csv.reader(csv_file)
    alist = []
    bdict = []
    cdict = []
    tmp = []
    for i in read:
        alist.append(i[0].split("\t"))
    for a in alist[1:]:
        print(a)
        bdict.append((a[0].split("?")[0], a[1]))
    for b in bdict:
        num = int(b[1])
        j = bdict.index(b) + 1
        while j <= (len(bdict) - 1):
            if b[0] not in tmp:
                if b[0] == bdict[j][0]:
                    num += int(bdict[j][1])
                    j += 1
                else:
                    j += 1
            else:
                j += 1
        if b[0] not in tmp:
            cdict.append((b[0], num))
        else:
            pass
        tmp.append(b[0])

with open('re_dk0519_1.csv', 'w', encoding="utf-8") as write_csvfile:
    writer = csv.writer(write_csvfile)
    writer.writerows(cdict)