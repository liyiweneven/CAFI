import matplotlib.pyplot as plt
import numpy as np
# 设置中文显示
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# 设置标题
plt.title(' ')
#数据准备
x = ['AUC','MRR','nDCG@5','nDCG@10']
y1= [0.6829,0.3291,0.3648,0.4282]
y2= [0.6860,0.3323,0.3665,0.4302]
y3= [0.6788,0.3282,0.3623,0.4263]
y4= [0.6936,0.3444,0.3761,0.4330]
#设置图形宽度
bar_width = 0.2
X_A = np.arange(len(x))  # A班条形图的横坐标
X_B = X_A + bar_width  # B班条形图的横坐标
X_C = X_B + bar_width
X_D = X_C + bar_width
#绘制图形
plt.bar(X_A,y1,bar_width,align='center',color =['#1f77b4'])
plt.bar(X_B,y2,width=bar_width,color =['#FAEBD7'])
plt.bar(X_C,y3,width=bar_width,color =['#00FFFF'])
plt.bar(X_D,y4,width=bar_width,color =['#7FFFD4'])
plt.xticks(X_A + bar_width/2, x)# 让横坐标显示运动
#加图例
plt.legend(['CAFI-can','CAFI-fine','CAFI-user','CAFI'], fontsize=16)
fig=plt.gcf()
#fig.set_facecolor('green')
#显示
plt.savefig('fig4.svg')
plt.show()
