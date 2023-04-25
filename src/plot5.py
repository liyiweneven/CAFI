# 本样例源于论文 TAPEX: Table Pre-training via Learning a Neural SQL Executor

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 用于绘制注意力图的矩阵, 实际使用时也可以考虑从文件中读入
data_matrix = np.mat(
    [[0.6768,0.6848,0.6895,0.6879,0.6830],  #256
     [0.6797,0.6860, 0.6936,0.6912,0.6837],  #128
     [0.6774, 0.6805, 0.6865,0.6812,0.6799], #64
     [0.6730,0.6779, 0.6844,0.6803,0.6782]],#32
)
#0.6855, 0.6730, 0.6930 , 0.6609, 0.6697
# 如果没有latex环境，可以将以下行注释
#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{lmodern}')

# 设置字体大小
plt.rc('font', **{'size': 14})

# 在seaborn中设定图片的宽和高
sns.set(rc={'figure.figsize': (6, 4.5)})

fig = sns.heatmap(data_matrix,
                  linewidth=0,
                  # 将具体的数字写在对应的表格中，%.1f 指定了样式，在较复杂的样式中可以去掉
                  #annot=np.array(['%.1f' % point for point in np.array(data_matrix.ravel())[0]]).reshape(np.shape(data_matrix)),
                  # 这里必须置空，否则会出现问题
                  fmt='',
                  yticklabels=["256", "128", "64", "32"],
                  # 如果 usetext=True, 这里可以使用 latex 语法比如 $\leq$ = <
                  xticklabels=["1", "2", "3", "4", "5"],
                  # cmap 决定了注意力图的色调
                  cmap="YlGnBu",
                  vmax=0.695,
                  vmin=0.670)

plt.ylabel("filter num of each layer", labelpad=25, fontsize=16)
plt.xlabel("stack layers in HDC", labelpad=25, fontsize=16)

# 调整布局至合适的位置
plt.tight_layout()
# 保存文件
#plt.savefig('attention.pdf')
plt.savefig('fig6.svg')
plt.show()