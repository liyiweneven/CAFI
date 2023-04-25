import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'AUC': [32020,30478,5916,4955,4570,4569,4418,4255,3071,2929,1323,1263,837,815,104,2,1,1],

}, index=['sports', 'news', 'finance', 'travel ', 'lifestyle ' ,'video ', 'foodanddrink ','weather','autos ','health ',
          'tv','music','entertainment','movies','kids','middleeast','games','northamerica'])

lines = df.plot.line(color={"AUC": "blue"}, marker='s')
# ax = plt.gca()
# ax.set_ylim(0,1.5)
# plt.legend(bbox_to_anchor=(0.999,1.13),ncol=4,fancybox=True,shadow=True)
plt.grid(linestyle="--", alpha=0.5)
plt.xlabel("λ=0.48 ", fontsize=16)
plt.ylabel("MIAR@Performanc", fontsize=16)

plt.show()