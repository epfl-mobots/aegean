#!/usr/bin/env python
import os
import glob
import sys
import math
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pprint import pprint

from pylab import *
from matplotlib import gridspec
import matplotlib.lines as mlines
from cycler import cycler

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
palette = flatui
# palette = "husl"
colors = sns.color_palette(palette)
sns.set(style="darkgrid")


gfontsize = 10
params = {
    'axes.labelsize': gfontsize,
    'font.size': gfontsize,
    'legend.fontsize': gfontsize,
    'xtick.labelsize': gfontsize,
    'ytick.labelsize': gfontsize,
    'text.usetex': False,
    # 'figure.figsize': [10, 15]
    # 'ytick.major.pad': 4,
    # 'xtick.major.pad': 4,
    'font.family': 'Arial',
}
rcParams.update(params)

data_path = os.getcwd() + '/'

pts = np.linspace(0, np.pi * 2, 24)
circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
vert = np.r_[circ, circ[::-1] * 1.0]
open_circle = mpl.path.Path(vert)

extra = Rectangle((0, 0), 1, 1, fc="w", fill=False,
                  edgecolor='none', linewidth=0)
shapeList = [
    Circle((0, 0), radius=1, facecolor=colors[0]),
    Circle((0, 0), radius=1, facecolor=colors[1]),
    Circle((0, 0), radius=1, facecolor=colors[2]),
    Circle((0, 0), radius=1, facecolor=colors[3]),
    Circle((0, 0), radius=1, facecolor=colors[4]),
    # Circle((0, 0), radius=1, facecolor=colors[5]),
]

v = np.r_[circ, circ[::-1] * 0.6]
oc = mpl.path.Path(v)

handles_a = [
    mlines.Line2D([0], [0], color='black', marker=oc,
                  markersize=6, label='Mean and SD'),
    mlines.Line2D([], [], linestyle='none', color='black', marker='*',
                  markersize=5, label='Median'),
    mlines.Line2D([], [], linestyle='none', markeredgewidth=1, marker='o',
                  color='black', markeredgecolor='w', markerfacecolor='black', alpha=0.5,
                  markersize=5, label='Single run')
]
handles_b = [
    mlines.Line2D([0], [1], color='black',  label='Mean'),
    Circle((0, 0), radius=1, facecolor='black', alpha=0.35, label='SD')
]


def pplots(data, ax, sub_colors=[], exp_title='', ticks=False):
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 10}
    sns.set_context("paper", rc=paper_rc)

    sns.pointplot(data=np.transpose(data), palette=sub_colors,
                  size=5, estimator=np.mean,
                  ci='sd', capsize=0.2, linewidth=0.8, markers=[open_circle],
                  scale=1.6, ax=ax)

    sns.stripplot(data=np.transpose(data), edgecolor='white',
                  dodge=True, jitter=True,
                  alpha=.50, linewidth=0.8, size=5, palette=sub_colors, ax=ax)

    medians = []
    for d in data:
        medians.append([np.median(list(d))])
    sns.swarmplot(data=medians, palette=['#000000']*10,
                  marker='*', size=5,  ax=ax)


def error_plots(replicate_dict, args):
    mse_dist = []
    ble_dist = []

    for _, v in replicate_dict.items():
        mse_dist.append(v['mse'])
        ble_dist.append(v['ble'])

    ticks = np.arange(0.05, 0.276, 0.025)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(
        7, 4), gridspec_kw={'width_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.00, wspace=0.10)
    sns.despine(bottom=True, left=True)

    # MSE ---------
    pplots(mse_dist, ax0, sub_colors=colors)
    ax0.set_ylabel(
        'Average mean squared position error (AMSPE) in cm', labelpad=1.5)
    ax0.set_xlabel('Experiment', labelpad=1.5)
    # ax0.legend(handles=handles_a,
    #            handletextpad=0.5, columnspacing=1,
    #            loc="upper right", ncol=1, framealpha=0, frameon=False, fontsize=gfontsize/2)
    ax0.set_yticks(ticks)

    mean_of_means = [np.mean(l) for l in mse_dist]
    pplots([mean_of_means], ax1, sub_colors=colors)
    ax1.set_ylabel(
        'Average of AMSPE in cm', labelpad=1.5)
    ax1.set_yticks(ticks)
    ax1.yaxis.tick_right()
    ax1.set_yticklabels([])
    ax1.tick_params(axis='both', which='both', length=0)
    ax1.set_xticks([])
    ax1.set_xlabel('Total', labelpad=9)
    ax1.legend(handles=handles_a,
               handletextpad=0.5, columnspacing=1,
               loc="upper right", ncol=1, framealpha=0, frameon=False, fontsize=gfontsize/2)
    plt.savefig(args.path + '/mse_error_plots.tiff', dpi=300)

    # BLE ---------
    ticks = np.arange(0.01, 0.075, 0.005)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(
        7, 4), gridspec_kw={'width_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.00, wspace=0.10)
    sns.despine(bottom=True, left=True)

    pplots(ble_dist, ax0, sub_colors=colors)
    ax0.set_ylabel(
        'Average body length error (ABLE) (Body length ~ 4 cm)', labelpad=1.5)
    ax0.set_xlabel('Experiment', labelpad=1.5)
    # ax0.legend(handles=handles_a,
    #            handletextpad=0.5, columnspacing=1,
    #            loc="upper right", ncol=1, framealpha=0, frameon=False, fontsize=gfontsize/2)
    ax0.set_yticks(ticks)

    mean_of_means = [np.mean(l) for l in ble_dist]
    pplots([mean_of_means], ax1, sub_colors=colors)
    ax1.set_ylabel(
        'Average of (ABLE) in cm', labelpad=1.5)
    ax1.set_yticks(ticks)
    ax1.yaxis.tick_right()
    ax1.set_yticklabels([])
    ax1.tick_params(axis='both', which='both', length=0)
    ax1.set_xticks([])
    ax1.set_xlabel('Total', labelpad=9)
    ax1.legend(handles=handles_a,
               handletextpad=0.5, columnspacing=1,
               loc="upper right", ncol=1, framealpha=0, frameon=False, fontsize=gfontsize/2)
    plt.savefig(args.path + '/ble_error_plots.tiff', dpi=300)


def deviation_plots(replicate_dict, args):
    dpe_dist = []
    for k, v in replicate_dict.items():
        dpe_dist.append(v['dpe'])

    ticks = np.arange(0.60, 1.04, 0.05)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(
        7, 4), gridspec_kw={'width_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.00, wspace=0.10)
    sns.despine(bottom=True, left=True)

    pplots(dpe_dist, ax0, sub_colors=colors)
    ax0.set_ylabel(
        'Correct behaviour prediction percentage', labelpad=1.5)
    ax0.set_xlabel('Experiment', labelpad=1.5)
    # ax0.legend(handles=handles_a,
    #            handletextpad=0.5, columnspacing=1,
    #            loc="upper right", ncol=1, framealpha=0, frameon=False, fontsize=gfontsize/2)
    ax0.set_yticks(ticks)

    mean_of_means = [np.mean(l) for l in dpe_dist]
    pplots([mean_of_means], ax1, sub_colors=colors)
    ax1.set_ylabel(
        'Average of experiment percentages', labelpad=1.5)
    ax1.yaxis.tick_right()
    ax1.set_yticks(ticks)
    # ax1.set_yticklabels([])
    ax1.tick_params(axis='both', which='both', length=0)
    ax1.set_xticks([])
    ax1.set_xlabel('Total', labelpad=9)
    ax1.legend(handles=handles_a,
               handletextpad=0.5, columnspacing=1,
               loc="upper right", ncol=1, framealpha=0, frameon=False, fontsize=gfontsize/2)

    plt.savefig(args.path + '/deviation_plots.tiff', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate prediction error plots')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment directory',
                        required=True)
    args = parser.parse_args()

    mse = glob.glob(args.path + '/seg_*_error_norm.dat')
    ble = glob.glob(args.path + '/seg_*_error_body_length.dat')
    dev = glob.glob(args.path + '/*_deviation.dat')

    replicates = []
    for f in mse:
        replicates.append(int(f.split('/')[-1].split('_')[1]))
    num_mse_replicates = len(set(replicates))
    num_mse_individuals = int(len(replicates) / num_mse_replicates)

    replicates = []
    for f in mse:
        replicates.append(int(f.split('/')[-1].split('_')[1]))
    num_ble_replicates = len(set(replicates))
    num_ble_individuals = int(len(replicates) / num_ble_replicates)

    assert num_mse_replicates == num_ble_replicates, 'Replicate files don\'t match in quantity'
    assert num_mse_individuals == num_ble_individuals, 'Individual numbers don\'t match in quantity'
    num_individuals = num_ble_individuals

    replicate_dict = {}
    for i in range(num_mse_replicates):
        if i not in replicate_dict.keys():
            replicate_dict[i] = {}
            replicate_dict[i]['mse'] = []
            replicate_dict[i]['ble'] = []
            replicate_dict[i]['dpe'] = []

        for j in range(num_individuals):
            f1 = args.path + '/seg_' + \
                str(i) + '_virtual_traj_ex_' + \
                str(j) + '_error_norm.dat'
            f2 = args.path + '/seg_' + \
                str(i) + '_virtual_traj_ex_' + \
                str(j) + '_error_body_length.dat'
            f3 = args.path + '/seg_' + \
                str(i) + '_virtual_traj_ex_' + \
                str(j) + '_feature_matrix_deviation.dat'

            mse_er = np.asscalar(np.loadtxt(f1))
            ble_er = np.asscalar(np.loadtxt(f2))
            dev_pe = np.asscalar(np.loadtxt(f3))

            replicate_dict[i]['mse'].append(mse_er)
            replicate_dict[i]['ble'].append(ble_er)
            replicate_dict[i]['dpe'].append(dev_pe)

    error_plots(replicate_dict, args)
    deviation_plots(replicate_dict, args)
