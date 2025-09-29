import numpy as np
import matplotlib.pyplot as plt

name = "data_figure_4.npy"

meas = ['Bell', 'XX_YY_ZZ (Pauli BDS)','MUB4', 'Pauli', 'Haar Random', 'Haar Random Bipartite']
c_meas = ['firebrick', 'goldenrod', 'dodgerblue', 'olive', 'red', 'blue']
lims = [[-0.01, 0.26], [-0.02, 0.52], [-0.03, 0.78]]
yticks = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25])

dim = [4]
n_meas = [np.arange(3, 91, 3, dtype= int), np.arange(3, 91, 3, dtype= int), np.arange(5, 91, 5, dtype= int), np.arange(9, 91, 9, dtype= int), np.arange(3, 91, 3, dtype= int), np.arange(3, 91, 3, dtype= int)]
n_sample = 4000

recon =['Bayesian estimation', 'maximum likelihood estimaton', 'direct inversion']
metric =['fidelity', 'HS']

markers = ['o', 'x', 'd']
m_s = 6 #markersize
l_w = 2 #linewidth
f_s = 12 #fontsize

fid = np.load(name)[[0, 2, 4], 0, 0] #[estimator][meas][nmeas][sample]
HS = np.load(name)[[1, 3, 5], 0, 0] #[estimator][meas][nmeas][sample]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), layout="constrained")
pos = [[0], [1], [2]]

for indj, j in enumerate([0, 2, 1]):
    for i in range(len(meas)):
        temp = HS[j][i][:len(n_meas[i])]
        HS_std = np.std(1 - temp, axis=1) / np.sqrt(n_sample)
        if indj== 0: axs[*pos[indj]].errorbar(n_meas[i], np.average(temp, axis=1), yerr= HS_std, c= c_meas[i], lw=l_w, ls= "", marker= markers[j], ms= m_s, label= meas[i], alpha=1, zorder= 1)
        else: axs[*pos[indj]].errorbar(n_meas[i], np.average(temp, axis=1), yerr= HS_std, c= c_meas[i], lw=l_w, ls= "", marker= markers[j], ms= m_s, alpha=1, zorder= 1)

    x = np.linspace(3, 90, 1000)
    if j == 0: 
        axs[*pos[indj]].plot(x, 3/(5*(x+4)), c= c_meas[0], ls= ":", label= r'Bell: $\frac{3}{5(N+4)}$')
    if j == 1: 
        axs[*pos[indj]].plot(x, 3/(5*x), c= c_meas[0], ls= ":", label= r'Bell: $\frac{3}{5N}$')
    if j == 2:
        axs[*pos[indj]].plot(x, 3/(5*x), c= c_meas[0], ls= ":", label= r'Bell: $\frac{3}{5N}$')
        axs[*pos[indj]].plot(x, 9/(5*x), c= c_meas[1], ls= ":", label= r'XX_YY_ZZ: $\frac{9}{5N}$')
        x =np.linspace(5, 90, 1000)
        axs[*pos[indj]].plot(x, 3/(x), c= c_meas[2], ls= ":", label= r'MUB4: $\frac{3}{N}$')
        x =np.linspace(9, 90, 1000)
        axs[*pos[indj]].plot(x, 27/(5*x), c= c_meas[3], ls= ":", label= r'Pauli: $\frac{27}{5N}$')

    axs[*pos[indj]].set_title(recon[j])
    axs[*pos[indj]].set_xlim(2, 91)
    axs[*pos[indj]].set_ylim(*lims[j])
    axs[*pos[indj]].set_xticks(n_meas[0][1::2])
    axs[*pos[indj]].set_yticks(yticks * (j+1))
    axs[*pos[indj]].legend(fontsize= f_s, loc='upper right')
    axs[*pos[indj]].grid()
    axs[0].set_xlabel(r'number of measurements $N$', fontsize=f_s)
    axs[1].set_xlabel(r'number of measurements $N$', fontsize=f_s)
    axs[0].set_ylabel(r'average risk (HS)', fontsize=f_s)

plt.savefig("figure_4-5", dpi= 600)
plt.show()
