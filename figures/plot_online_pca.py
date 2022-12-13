import pickle
import matplotlib.pyplot as plt
import numpy as np
from config import methods_ids, colors, names, line_styles

with open('data/pca_test1.pkl', 'rb') as handle:
    results = pickle.load(handle)

metric = 'stiefel_distances'

plt.figure(figsize=(4, 3))

for method_id in methods_ids:
    print("\t ploting: "+ method_id)
    to_plot = results[method_id][metric]
    time = results[method_id]['time_list']
    plt.plot(np.median(time, axis=0), np.median(to_plot, axis=0), 
        label=names[method_id], 
        color=colors[method_id], 
        linewidth=3, linestyle=line_styles[method_id])
    plt.fill_between(np.median(time, axis=0), np.min(to_plot, axis=0), np.max(to_plot, axis=0), color=colors[method_id], alpha=0.2)

plt.legend(ncol=2, loc='upper right', columnspacing=.5, handlelength=2)
plt.xlim([None,50])
plt.yscale('log')
x_ = plt.xlabel('Time (min.)')
y_ = plt.ylabel('Distance to the constraint $\mathcal{N}(X)$')
plt.grid()
plt.savefig('resnet18_%s.pdf' % metric, bbox_inches='tight', bbox_extra_artists=(x_, y_))