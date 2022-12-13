import pickle
import matplotlib.pyplot as plt
import numpy as np
from config import methods_ids, colors, names, line_styles

with open('data/vgg16.pkl', 'rb') as handle:
    results = pickle.load(handle)

metric = 'test_accuracy'

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

plt.legend(ncol=2, loc='lower right', columnspacing=.5, handlelength=2)
plt.xlim([None,35])
x_ = plt.xlabel('Time (min.)')
y_ = plt.ylabel('Test accuracy')
ticks = [20, 40, 60, 80]
plt.yticks(ticks=ticks, labels=[ str(tick)+'\%' for tick in ticks])
plt.grid()
plt.savefig('vgg16_%s.pdf' % metric, bbox_inches='tight', bbox_extra_artists=(x_, y_))