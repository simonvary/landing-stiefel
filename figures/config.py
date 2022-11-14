import matplotlib.pyplot as plt

plt.rcParams.update({'text.usetex' : True})

methods_ids = ['landing1', 'retraction1' , 'regularization1', 'regularization2']

colormap = plt.cm.Set1
colors = {}
for i in range(len(methods_ids)):
    colors[methods_ids[i]] = colormap.colors[i]

names = {'landing1': 'Landing',
         'retraction1': 'Retraction', 
         'regularization1': 'Reg. $\lambda = 1$', 
         'regularization2': 'Reg. $\lambda = 10^3$'}

line_styles = {'landing1': 'solid',
         'retraction1': ':', 
         'regularization1': '--', 
         'regularization2': '-.'}