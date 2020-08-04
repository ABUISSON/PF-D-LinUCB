"""
    File description: Useful functions
"""

# Author: Yoan Russac (yoan.russac@ens.fr)
# License: BSD (3-clause)

# Importations
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


current_palette = sns.color_palette()
sns.set_style("ticks")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc("lines", linewidth=3)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('font', weight='bold')
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath} \boldmath"]
styles = ['o', '^', 's', 'D', 'p', 'v', '*']
colors = current_palette[0:11]


def plot_regret(data, filename=None, t_saved=None, log=False, qtl=False, loc=0, font=10, bp=None):
    """
    param:
        - data:
        - t_saved: numpy array (ndim = 1), index of the points to save on each trajectory
        - filename: Name of the file to save the plot of the experiment, if None then it is only plotted
        - log: Do you want a log x-scale
        - qtl: Plotting the lower and upper quantiles. Other effect: If qtl == False then only t_saved
               are printed in the other case everything is printed
        - loc: Location of the legend for fine-tuning the plot
        - font: Font of the legend for fine-tuning the plot
        - bp: Dictionary for plotting the time steps where the breakpoints occur
    Output:
    -------
    Plot it the out/filename file
    """
    fig = plt.figure(figsize=(7, 6))
    if log:
        plt.xscale('log')
    i = 0

    if t_saved is None:
        len_tsaved = len(data[0][1])
        t_saved = [i for i in range(len_tsaved)]

    for key, avgRegret, qRegret, QRegret, _, in data:
        label = r"\textbf{%s}" % key
        plt.plot(t_saved, avgRegret, marker=styles[i],
                 markevery=0.1, ms=10.0, label=label, color=colors[i])
        if qtl:
            plt.fill_between(t_saved, qRegret, QRegret, alpha=0.15,
                             linewidth=1.5, color=colors[i])
        i += 1
    plt.legend(loc=loc, fontsize=font).draw_frame(True)
    plt.xlabel(r'Round $\boldsymbol{t}$', fontsize=20)
    plt.ylabel(r'Regret $\boldsymbol{R(T)}$', fontsize=18)
    if filename:
        plt.savefig('out/%s.png' % filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return 0


def generation_temp(sim, steps, n_mc, q, seed_tot, t_saved):
    avgRegret, qRegret, QRegret, timedic, avg_cum_b_a, \
    q_b_a, Q_b_a, cum_r = sim.run_exp(steps, n_mc, q, seed_tot, t_saved)
    return avgRegret, qRegret, QRegret, timedic, avg_cum_b_a, q_b_a, Q_b_a, cum_r


def data_generation(n_mc, sim, steps, q, seed_tot, t_saved=None):
    avgRegret, qRegret, QRegret, timedic, \
    avg_cum_b_a, q_b_a, Q_b_a, cum_r = generation_temp(sim, steps, n_mc, q, seed_tot, t_saved)
    data = [[policy, avgRegret[policy], qRegret[policy],
                QRegret[policy], avg_cum_b_a[policy]] for policy in avgRegret]
    return data, timedic


def action_check(a_list, L=1):
    """
    Plotting the different actions received at time t
    param:
        - a_check: Action vectors
        - t: Time instant
    """
    x = [el[0] for el in a_list]
    y = [el[1] for el in a_list]
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.cla()
    ax.set_xlim((- L - 0.05, L + 0.05))
    ax.set_ylim((- L - 0.05, L + 0.05))
    plt.scatter(x, y, color='b')
    plt.gcf().gca().add_artist(plt.Circle((0, 0), L, color='r', fill=False))
    plt.show()
