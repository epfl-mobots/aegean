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


def pplots(data, ax, sub_colors=[], exp_title='', ticks=False, mn=False):
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
        # print(np.median(d))
        medians.append([np.median(list(d))])
    sns.swarmplot(data=medians, palette=['#000000']*10,
                  marker='*', size=5,  ax=ax)


def per_segment_plots(replicate_dict, args):
    mse_dist = []
    ble_dist = []

    for k, v in replicate_dict.items():
        mse_dist.append(v['mse'])
        ble_dist.append(v['ble'])

    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    pplots(mse_dist, ax, sub_colors=colors)
    plt.savefig(args.path + '/segment_error_plots_mse.png', dpi=300)

    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    pplots(ble_dist, ax, sub_colors=colors)
    plt.savefig(args.path + '/segment_error_plots_ble.png', dpi=300)


def all_plots(replicate_dict, args):
    mse_dist = []
    ble_dist = []

    for k, v in replicate_dict.items():
        mse_dist.append(v['mse'])
        ble_dist.append(v['ble'])

    mse_dist = [item for sublist in mse_dist for item in sublist]
    ble_dist = [item for sublist in ble_dist for item in sublist]
    dist = [mse_dist, ble_dist]

    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    pplots(dist, ax, sub_colors=colors)
    plt.savefig(args.path + '/total_error_plots.png', dpi=300)


def deviation_plots(replicate_dict, args):
    dpe_dist = []
    for k, v in replicate_dict.items():
        dpe_dist.append(v['dpe'])
    all_dpes = [item for sublist in dpe_dist for item in sublist]
    dpe_dist.append(all_dpes)

    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    pplots(dpe_dist, ax, sub_colors=colors)
    plt.savefig(args.path + '/deviation_plots.png', dpi=300)


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

    per_segment_plots(replicate_dict, args)
    all_plots(replicate_dict, args)
    deviation_plots(replicate_dict, args)
