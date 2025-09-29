import numpy as np
import matplotlib as mpl
from mpmath import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import rsmf

# Plotting parameters
formatter = rsmf.setup(r"\documentclass[twocolumn, amssymb, nobibnotes, aps, pra, superscriptaddress, 10pt]{revtex4-2}")
mpl.rcParams["legend.edgecolor"]    = "black"
mpl.rcParams["legend.fancybox"]     = True
mpl.rcParams["legend.borderpad"]    = 0.5
mpl.rcParams["legend.handlelength"] = .75
mpl.rcParams["lines.linewidth"]     = 1.3
mpl.rcParams["lines.markersize"]    = 3.2
mpl.rcParams["text.usetex"]         = False
mpl.rcParams["text.latex.preamble"] = "\\usepackage{amsmath,amsthm}; \\usepackage{physics}"

recon = ['Bayesian mean', 'maximum likelihood', 'direct inversion']
recon_l = ['BME', 'MLE', 'direct inv.']
meas = ['Bellstate meas.', 'parity checks','MUB', 'Pauli meas.', 'Haar rand.', 'sep. Haar rand.']
c_meas = ['#009E73', 'plum', '#0072B2', '#D55E00', '#56B4E9', '#E69F00']
z_order = [3, 3, 4, 4, 3, 3, 3]
markers = ['D', 'x', 's']

# Create figure
fig = formatter.figure(aspect_ratio=1/1.618, wide=True)
height_ratios = [0.6, 0.4]
gs = GridSpec(2, 12, height_ratios=height_ratios, figure=fig)
ax_est = [fig.add_subplot(gs[0, i*4:(i+1)*4]) for i in range(3)]
ax_meas = [fig.add_subplot(gs[1, i*3:(i+1)*3]) for i in range(4)]

# Import Data: select bayesian, MLE and direct inversion + Hilbert-Schmidt norm
n_sample = 1000
data_name = "data_figure_4-5.npy"
HS = np.load(data_name)[[1, 3, 5], 0, 0] #[estimator][meas][nmeas][sample]
n_meas = [np.arange(3, 91, 3, dtype= int), np.arange(3, 91, 3, dtype= int), np.arange(5, 91, 5, dtype= int), np.arange(9, 91, 9, dtype= int), np.arange(3, 91, 3, dtype= int), np.arange(3, 91, 3, dtype= int)]

# General Axis settings
xticks = np.array([0, 25, 50, 75])
lims = [[0, 0.22], [0, 0.66], [0, 0.66]]
yticks = np.array([[0, 0.1, 0.2], [0, 0.3, 0.6], [0, 0.3, 0.6]])

## Plot different estimator ##
for idx, idn in enumerate([0, 1, 2]):
    ax = ax_est[idx]
    for i in range(len(meas)):
        temp = HS[idn][i][:len(n_meas[i])]
        std = np.std(1 - temp, axis=1) / np.sqrt(n_sample)
        if idx == 0: ax.plot(n_meas[i], np.average(temp, axis=1), c= c_meas[i], ls= "", marker= markers[idn], alpha=1, zorder= z_order[i], label= meas[i])
        elif idx==1: ax.plot(n_meas[i], np.average(temp, axis=1), c= c_meas[i], ls= "", marker= markers[idn], ms= 4, zorder= z_order[i], alpha=1)
        else: ax.plot(n_meas[i], np.average(temp, axis=1), c= c_meas[i], ls= "", marker= markers[idn], alpha=1, zorder= z_order[i])
 
    ax.set_title(recon[idn], fontsize = 10)
    ax.set_xlim(0, 92)
    ax.set_ylim(*lims[idn])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks[idn])
    ax.grid()
    ax.set_xlabel(r'$N$')
    
    if idn == 2:
        x = np.linspace(0.7, 91.3, 1000)
        ax.plot(x, 3/(5*x), c= c_meas[0], ls= '-', dash_capstyle= 'round', label= r'$3/(5N)$', alpha =0.5, zorder= 1)
        ax.plot(x, 3/(5*x), c= 'white', ls= '-', dash_capstyle= 'round', alpha =0.1, zorder= 2)
        ax.plot(x, 9/(5*x), c= c_meas[1], ls= '-', dash_capstyle= 'round', label= r'$9/(5N)$', alpha =0.5, zorder= 1)
        ax.plot(x, 9/(5*x), c= 'white', ls= '-', dash_capstyle= 'round', alpha =0.1, zorder= 2)
        ax.plot(x, 3/x, c= c_meas[2], ls= '-', dash_capstyle= 'round', label= r'$3/N$', alpha =0.5, zorder= 1)
        ax.plot(x, 3/x, c= 'white', ls= '-', dash_capstyle= 'round', alpha =0.1, zorder= 2)
        ax.plot(x, 27/(5*x), c= c_meas[3], ls= '-', dash_capstyle= 'round', label= r'$27/(5N)$', alpha =0.5, zorder= 1)
        ax.plot(x, 27/(5*x), c= 'white', ls= '-', dash_capstyle= 'round', alpha =0.1, zorder= 2)

    if idx == 2: 
        legend = ax.legend(labelspacing=0.26)
        legend = legend.get_frame().set_linewidth(0.5)
    if idx == 0: 
        ax.set_ylabel(r'average risk, $r(\pi, \hat{\rho})$')
        legend = ax.legend(labelspacing=0.26)
        legend.get_frame().set_linewidth(0.5)


## Plot different measurements ##
lims = [[0, 0.22], [0, 0.66], [0, 0.66], [0, 0.66]]
yticks = np.array([[0, 0.1, 0.2], [0, 0.3, 0.6], [0, 0.3, 0.6], [0, 0.3, 0.6]])
factors = [3/5, 9/5, 3, 27/5]
for i in range(4):
    ax = ax_meas[i]
    x = np.linspace(0.7, 91.3, 1000)
    ax.plot(x, factors[i]/x, c= c_meas[i], ls= '-', dash_capstyle= 'round', alpha =0.5, zorder= 1)
    ax.plot(x, factors[i]/x, c= 'white', ls= '-', dash_capstyle= 'round', alpha =0.1, zorder= 2)
        
    for idn in range(3):
        temp = HS[idn][i][:len(n_meas[i])]
        std = np.std(1 - temp, axis=1) / np.sqrt(n_sample)
        if idn == 1:
            ax.plot(n_meas[i], np.average(temp, axis=1), c= c_meas[i], ls= "", marker= markers[idn], ms= 4.7, alpha=1, zorder= 1, label= recon_l[idn])
        elif idn == 2:
            ax.plot(n_meas[i], np.average(temp, axis=1), c= c_meas[i], ls= "", marker= markers[idn], ms= 2.5, alpha=1, zorder= 1, label= recon_l[idn])
        else:
            ax.plot(n_meas[i], np.average(temp, axis=1), c= c_meas[i], ls= "", marker= markers[idn], ms= 2.5, alpha=1, zorder= 1, label= recon_l[idn])

    ax.set_title(meas[i], fontsize= 10)
    ax.set_xlim(0, 92)
    ax.set_ylim(*lims[i])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks[i])
    ax.grid()
    ax.set_xlabel(r'$N$')
    if i == 0: 
        legend = ax.legend(labelspacing=0.26)
        legend.get_frame().set_linewidth(0.5)
        ax.set_ylabel(r'average risk, $r(\pi, \hat{\rho})$')


# Label subfigures
labels_est = ['(a)', '(b)', '(c)']
for ax, label in zip(ax_est, labels_est):
    ax.text(-0.35 * 0.4, 1 + 0.09 * 0.4, label, transform=ax.transAxes, ha='left', va='bottom', zorder=10)

labels_meas = ['(d)', '(e)', '(f)', '(g)']
for ax, label in zip(ax_meas, labels_meas):
    ax.text(-0.35 * 0.57, 1 + 0.09 * 0.6, label, transform=ax.transAxes, ha='left', va='bottom', zorder=10)

# Save figure
fig.tight_layout()
plt.savefig("figure_4.pdf", format="pdf", bbox_inches='tight')
