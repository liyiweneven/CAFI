import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({
    'AUC': [0.6867, 0.6816, 0.6843, 0.6936, 0.6808,0.6772],
    'MRR': [0.3389, 0.3301, 0.3338, 0.3444, 0.3315,0.3293],
    'nDCG@5': [0.3694, 0.3590, 0.3643, 0.3761, 0.3616,0.3574],
    'nDCG@10': [0.4273, 0.4191, 0.4226, 0.4330 , 0.4204,0.4172],

}, index=['1', '2', '3', '4', '5','6'])

'''df = pd.DataFrame({
    'AUC': [0.6960, 0.6910, 0.6936 , 0.7023, 0.6901,0.6865],
    'MRR': [0.3401, 0.3312, 0.3350, 0.3449, 0.3327,0.3305],
    'nDCG@5': [0.3772, 0.3668, 0.3721, 0.3834, 0.3694,0.3652],
    'nDCG@10': [0.4405, 0.4325, 0.4358, 0.4457 , 0.4338,0.4304],

}, index=['1', '2', '3', '4', '5','6'])'''

lines = df.plot.line(color={"AUC": "blue", "MRR": "lime", "nDCG@5": "red", "nDCG@10": "cyan"}, marker='s')

plt.legend(bbox_to_anchor=(0.999, 1.13), ncol=4, fancybox=True, shadow=True)
plt.grid(linestyle="--", alpha=0.5)
plt.xlabel("Negative sample ratio ", fontsize=16)
plt.ylabel("CAFI@Performance", fontsize=16)
plt.savefig('fig5.svg')
plt.show()