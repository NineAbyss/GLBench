import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import matplotlib.ticker as mticker
import math
from matplotlib.ticker import FuncFormatter

def format_func(value, tick_number):
    if value == 0:
        return "0"
    else:
        exp = int(np.log10(abs(value)))
        coeff = value / 10**exp
        return f"$10^{{{exp}}}$"

markersize=200 #marker大小
alpha_v=0.6 #marker透明度
rc('font', **{'family': 'serif', 'sans-serif': ['cm']})
# rc('text', usetex=True)
rc('axes', **{'titlesize': 'large'})
plt.rcParams['text.latex.preamble'] = """
\\usepackage{libertine}
\\usepackage[libertine]{newtxmath}
"""
# plt.rcParams['figure.figsize'] = (2, 9)
plt.rcParams['figure.dpi'] = 240
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
fontsize = 16

# GCN GAT GRCN
model_num = 8
markers = ['s', 'D', 'v', 'h', '*', 'd', 'o', '^']
colors = ['c', 'c', 'r', 'r', 'g', 'g', 'b', 'b']  # 每两个模型使用相同的颜色
# G: GCN, GAT E: OFA, ENGINE P:GraphText, LLaGA, A:GLEM, Patton
model_name = ['GCN', 'GAT', 'OFA', 'ENGINE', 'GraphText', 'LLaGA', 'GLEM', 'Patton']
data_name = ['Cora', 'Citeseer', 'WikiCS', 'Instagram']
accs, times, spaces = [], [], []

# Cora
accs.append([82.11, 80.31, 75.24, 81.54, 76.21, 74.42, 82.11, 70.50])
times.append([13.47, 14.11, 33.31, 104.57, 1364.4, 112.16, 130.72, 3403.8])
spaces.append([594, 632, 2179.072, 730, 65195.60, 15604, 18349, 71289])

# Citeseer
accs.append([69.84, 68.78, 73.04, 72.15, 59.43, 55.73, 71.16, 63.60])
times.append([13.392, 13.392, 15.97, 87.43, 1342.8, 97.12, 115.48, 1274.54])
spaces.append([594, 632, 1843.2, 648, 26418, 15430, 18347, 71289])

# WikiCS
accs.append([80.35, 79.73, 77.34, 81.19, 67.35, 73.88, 82.40, 80.81])
times.append([16.74, 27.58, 385.96, 409.64, 1431,478.28, 405.7949, 90884.09])
spaces.append([1928, 1968, 38461.44, 2036, 24823, 15430, 18349, 71289])

# Instagram
accs.append([65.75, 65.38, 60.85, 67.62, 62.64, 62.94, 66.1, 64.27])
times.append([20.28, 19.92, 262.3294, 343.61, 1330.2, 431.24, 731.5805, 31737.8])
spaces.append([1060, 1588, 14794.752, 812, 24826, 15296, 18349, 71289])
fig, axes = plt.subplots(2, 4, figsize=(18, 8))

from matplotlib.ticker import FuncFormatter

def format_func(value, tick_number):
   
    return f'{int(value)}'


for i, ax in enumerate(axes.flatten()):
    # ax.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    ax.grid(True, alpha=0.6)

    if i < 4:
        for j in range(model_num):
            ax.scatter(times[i][j], accs[i % 4][j], s=markersize, marker=markers[j], color=colors[j], alpha=alpha_v)
        ax.set_xlabel('Training Time (s)', fontsize=fontsize)
    else:
        m = []
        for j in range(model_num):
            marker = ax.scatter(spaces[i - 4][j], accs[i % 4][j], s=markersize, marker=markers[j], color=colors[j], alpha=alpha_v)
            m.append(marker)
        ax.set_xlabel('Training Space (MB)', fontsize=fontsize)

    if i % 4 == 0:
        ax.set_ylabel('Accuracy (%)', fontsize=fontsize)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_title(data_name[i % 4], fontsize=fontsize)

    if i == 0 or i == 4:
        ax.set_ylim(68, 85)
    elif i == 1 or i == 5:
        ax.set_ylim(52, 75)
    elif i == 2 or i == 6:
        ax.set_ylim(65, 85)
    elif i == 3 or i == 7:
        ax.set_ylim(60, 68)

    ax.set_xscale('log')

import matplotlib.patches as mpatches

# 创建自定义图例条目




# markers = ['s', 'D', 'v', 'h', '*', 'd', 'o', '^']
gcn_patch = plt.Line2D([0], [0], color='c', marker=markers[0], linestyle='', label='GCN', markersize=13, alpha=alpha_v)
gat_patch = plt.Line2D([0], [0], color='c', marker=markers[1], linestyle='', label='GAT', markersize=13, alpha=alpha_v)
ofa_patch = plt.Line2D([0], [0], color='r', marker=markers[2], linestyle='', label='OFA', markersize=13, alpha=alpha_v)
engine_patch = plt.Line2D([0], [0], color='r', marker=markers[3], linestyle='', label='ENGINE', markersize=13, alpha=alpha_v)
graphtext_patch = plt.Line2D([0], [0], color='g', marker=markers[4], linestyle='', label='GraphText', markersize=13, alpha=alpha_v)
llaga_patch = plt.Line2D([0], [0], color='g', marker=markers[5], linestyle='', label='LLaGA', markersize=13, alpha=alpha_v)
glem_patch = plt.Line2D([0], [0], color='b', marker=markers[6], linestyle='', label='GLEM', markersize=13, alpha=alpha_v)
patton_patch = plt.Line2D([0], [0], color='b', marker=markers[7], linestyle='', label='Patton', markersize=13, alpha=alpha_v)

fig.legend(handles=[gcn_patch, gat_patch,
                    ofa_patch, engine_patch,
                    graphtext_patch, llaga_patch,
                    glem_patch, patton_patch],
           prop={'size': fontsize},
           loc='lower center',  # 将图例放在图下面
           borderaxespad=0,  # 图例框周围的间距
           bbox_to_anchor=(0.5, -0.08),  # 调整图例位置
           ncol=4, handlelength=1, labelspacing=0.2, columnspacing=6,
           fontsize=fontsize, fancybox=True, frameon=False)

fig.tight_layout()
plt.savefig("complexity.pdf", bbox_inches='tight')