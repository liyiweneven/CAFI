import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({
    'AUC': [0.6855, 0.6730, 0.6936 , 0.6609, 0.6697],
    'MRR': [0.3406, 0.3267, 0.3444, 0.3171, 0.3237],
    'nDCG@5': [0.3720, 0.3560, 0.3761, 0.3413, 0.3537],
    'nDCG@10': [0.4289, 0.4138,0.4330 ,0.4016 , 0.4114],

}, index=['1', '2', '3', '4', '5'])

lines = df.plot.line(color={"AUC": "blue", "MRR": "lime", "nDCG@5": "red", "nDCG@10": "cyan"}, marker='s')

plt.legend(bbox_to_anchor=(0.999, 1.13), ncol=4, fancybox=True, shadow=True)
plt.grid(linestyle="--", alpha=0.5)
plt.xlabel("Hierarchy of HDC news encoder ", fontsize=16)
plt.ylabel("CAFI@Performance", fontsize=16)
plt.show()