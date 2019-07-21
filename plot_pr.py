"""
plot precision-recall bar chart from json dictionary
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import json

filename = 'pr/models_deep_0'

data = json.load(open(filename+'.json', 'r'))
print(data)
i = 0

#matplotib subplot index
def ret_ind(i):
	if i == 0:
		return (0,0)
	elif i == 1:
		return (0,1)
	elif i == 2:
		return (1,0)
	else:
		return (1,1)

fig, axs = plt.subplots(2, 2)
fig.suptitle('Deep without context')

#4 subplots, ones for each standard deviation
for std, data in data.items():
		
	axs[ret_ind(i)].title.set_text('Std. Dev. - '+std)

	prec, recall = [], []
	for cons, vals in data.items():
		prec.append((float(cons), vals['tp']/(vals['tp']+vals['fp'])))
		recall.append((float(cons), vals['tp']/(vals['tp']+vals['fn'])))

	bar_width = 0.09
	opacity = 0.8
	index = [x[0] for x in recall]

	#plot recall
	rects1 = axs[ret_ind(i)].bar(index, [x[1] for x in recall], bar_width,
	alpha=opacity,
	color='b',
	label='Recall')
	#plot precision
	rects2 = axs[ret_ind(i)].bar([x+bar_width for x in index], [x[1] for x in prec], bar_width,
	alpha=opacity,
	color='g',
	label='Precision')

	plt.xlabel('K_multi')
	plt.ylabel('Precision/Recall')
	plt.xticks(index)
	i += 1
	
red_patch = mpatches.Patch(color='green', label='Precision')
blue_patch = mpatches.Patch(color='blue', label='Recall')

fig.legend(handles=[red_patch, blue_patch])
plt.show()