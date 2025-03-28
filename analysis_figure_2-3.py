import numpy as np
import matplotlib.pyplot as plt
from mpmath import *

title = ['BDS_dirichlet']
name = "data_figure_2-3.npy"

meas = ['Bell', 'Pauli BDS','Pauli BDS unordered']
c_meas = ['firebrick', 'goldenrod', 'dodgerblue', 'olive', 'red', 'blue']
lims = [[-0.01, 0.26], [-0.02, 0.52], [-0.03, 0.78]]
yticks = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25])

dim = [4]
n_meas = [np.arange(1, 31, 1, dtype= int), np.arange(3, 31, 3, dtype= int), np.arange(1, 31, 1, dtype= int)]
n_sample = 1000

recon =['Bayesian estimation', 'maximum likelihood estimaton', 'direct inversion']
metric =['fidelity', 'HS']

markers = ['o', 'x', 'd']
m_s = 6 #markersize
l_w = 2 #linewidth
f_s = 12 #fontsize


HS = np.load(name)[[0, 2, 4], 0, 0] #[estimator][nmeas][meas][sample]
HS = np.load(name)[[1, 3, 5], 0, 0] #[estimator][nmeas][meas][sample]

th_sq = 2 / 5
def fun(N):
    return (2/3)**N * ((1 - th_sq)*(1/2 * N * hyp3f2(1, 1, 1 - N, 2, 2, -1/2) - 1) + 3/4)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 5), layout="constrained")
pos = [[0], [1]]

for indj, j in enumerate([0, 2]):
    for i in range(len(meas)):
        temp = HS[j][i][:len(n_meas[i])]
        HS_std = np.std(1 - temp, axis=1) / np.sqrt(n_sample)
        if indj== 0: axs[*pos[indj]].errorbar(n_meas[i], np.average(temp, axis=1), yerr= HS_std, c= c_meas[i], lw=l_w, ls= "", marker= markers[j], ms= m_s, label= meas[i], alpha=1, zorder= 1)
        else: axs[*pos[indj]].errorbar(n_meas[i], np.average(temp, axis=1), yerr= HS_std, c= c_meas[i], lw=l_w, ls= "", marker= markers[j], ms= m_s, alpha=1, zorder= 1)

    x = np.linspace(1, 30, 1000)
    if j == 0: 
        axs[*pos[indj]].plot(x, 3/(5*(x+4)), c= c_meas[0], ls= ":", label= r'Bell: $\frac{3}{5(N+4)}$')
        axs[*pos[indj]].plot(x, (x+3)/(5*(x/3+2)**2), c= c_meas[1], ls= ":", label= r'Bayesian')

    if j == 1: 
        axs[*pos[indj]].plot(x, 3/(5*x), c= c_meas[0], ls= ":", label= r'Bell: $\frac{3}{5N}$')
    if j == 2:
        axs[*pos[indj]].plot(x, 3/(5*x), c= c_meas[0], ls= ":", label= r'Bell: $\frac{3}{5N}$')
        y= [fun(N) for N in x]
        axs[*pos[indj]].plot(x, y, c= c_meas[2], ls= ":", label= r'XX_YY_ZZ random')
        x = np.linspace(3, 30, 1000)
        axs[*pos[indj]].plot(x, 9/(5*x), c= c_meas[1], ls= ":", label= r'XX_YY_ZZ ordered')
        #axs[*pos[indj]].plot(x, 9/(5*x), c= c_meas[1], ls= ":", label= r'XX_YY_ZZ: $\frac{9}{5N}$')

 
    axs[*pos[indj]].set_title(recon[j])
    axs[*pos[indj]].set_xlim(0, 31)
    axs[*pos[indj]].set_ylim(*lims[j])
    axs[*pos[indj]].set_xticks(n_meas[1][1::2])
    axs[*pos[indj]].set_yticks(yticks * (j+1))
    axs[*pos[indj]].legend(fontsize= f_s, loc='upper right')
    axs[*pos[indj]].grid()
    axs[0].set_xlabel(r'number of measurements $N$', fontsize=f_s)
    axs[1].set_xlabel(r'number of measurements $N$', fontsize=f_s)
    axs[0].set_ylabel(r'average risk (HS)', fontsize=f_s)

plt.savefig("figure_2-3", dpi= 600)
plt.show()
