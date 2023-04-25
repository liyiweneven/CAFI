'''# 绘制育龄妇女的受教育程度分布饼图
import matplotlib.pyplot as plt

# ********** Begin *********#
# 总数据
Num = 101527
print(Num)
# 单个数据
data = [32020,30478,5916,4955,4570,4569,4418,4255,3071,2929,1323,1263,837,815,104,2,1,1]
# 数据标签
labels = ['sports', 'news', 'finance', 'travel ', 'lifestyle ','middleeast' ,'video ', 'foodanddrink ','weather','autos ','health ','tv','music','entertainment','movies',
          'northamerica','kids','games']
# 各区域颜色
colors = ['green', 'yellow','orange' , 'red', 'purple', 'blue', 'black']
# 数据计算处理
sizes = [data[0] / Num * 100, data[1] / Num * 100, data[2] / Num * 100, data[3] / Num * 100, data[4] / Num * 100,
         data[5] / Num * 100, data[6] / Num * 100, data[7] / Num * 100, data[8] / Num * 100, data[9] / Num * 100, data[10] / Num * 100,
         data[11] / Num * 100, data[12] / Num * 100, data[13] / Num * 100, data[14] / Num * 100, data[15] / Num * 100, data[16] / Num * 100, data[17] / Num * 100]
# 设置突出模块偏移值
expodes = (0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
# 设置绘图属性并绘图
plt.pie(sizes, explode=expodes, labels=labels, shadow=True, colors=colors)
## 用于显示为一个长宽相等的饼图
plt.axis('equal')
# 保存并显示
#plt.savefig('fig3.png')
plt.show()
# ********** End **********#'''

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.family']='SimHei'

names=np.array(['sports', 'news', 'finance', 'travel ', 'lifestyle ' ,'video ', 'foodanddrink ','weather','autos ','health ',
                'tv','music','entertainment','movies','kids','middleeast','games','northamerica'])
money=np.array([32020,30478,5916,4955,4570,4569,4418,4255,3071,2929,1323,1263,837,815,104,2,1,1])
money_rate=money/np.sum(money)
explode=np.zeros((len(money)))
explode[14]=0.2
explode[15]=0.6
plt.figure(figsize=(20,9))
patches,l_text,p_text=plt.pie(money_rate,explode=explode,labels=names,autopct='%.2f%%',pctdistance=0.8)

plt.legend(['sports', 'news', 'finance', 'travel ', 'lifestyle ' ,'video ', 'foodanddrink ','weather','autos ','health ',
                'tv','music','entertainment','movies','kids','middleeast','games','northamerica'])
plt.legend(loc='center right')
plt.title('MIND数据集类别占比')
plt.axis('equal')

# 设置饼图内文字大小

for t in p_text:
    t.set_size(8)

for t in l_text:
    t.set_size(9)
plt.show()