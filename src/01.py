import math

# 推荐列表
R = [[3, 10, 15, 12, 17], [20, 15, 18, 14, 30], [2, 5, 7, 8, 15], [56, 14, 25, 12, 19], [21,24,36,54,45]]
# 用户访问列表
T=[[3],[3],[5],[14],[20]]


def indicators_5(rankedList, testList):
    Hits_i = 0
    Len_R = 0
    Len_T = len(testList)
    MRR_i = 0
    HR_i = 0
    NDCG_i = 0
    for i in range(len(rankedList)):
        for j in range(len(rankedList[i])):
            if testList[i][0]==rankedList[i][j]:
                Hits_i+=1
                HR_i+=1
                # 注意j的取值从0开始
                MRR_i+=1/(j+1)
                NDCG_i+=1/(math.log2(1+j+1))
                break
    HR_i/=Len_T
    MRR_i/=Len_T
    NDCG_i/=Len_T
    print(Hits_i)
    print(f'HR@5={HR_i}')
    print(f'MRR@5={MRR_i}')
    print(f'NDCG@5={NDCG_i}')



if __name__ == '__main__':
    indicators_5(R, T)


