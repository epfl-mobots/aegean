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
from matplotlib.colors import ListedColormap
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


def km_predict(mat, centroids):
    assert(centroids.shape[1] == mat.shape[1])
    distances = np.ones((mat.shape[0], centroids.shape[0])) * -1
    for r in range(centroids.shape[0]):
        c = centroids[r, :]
        diff = mat - c
        distances[:, r] = np.linalg.norm(diff, axis=1)
    labels = np.argmin(distances, axis=1)
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate prediction error plots')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment directory',
                        required=True)
    args = parser.parse_args()

    num_behaviours = int(np.asscalar(
        np.loadtxt(args.path + '/num_behaviours.dat')))
    cluster_centers = np.loadtxt(args.path + '/centroids_kmeans.dat')
    cluster_data = []
    for i in range(num_behaviours):
        cluster_data.append(np.loadtxt(
            args.path + '/cluster_' + str(i) + '_data.dat'))

    all_data = cluster_data[0]
    for i in range(1, num_behaviours):
        all_data = np.concatenate((all_data, cluster_data[i]))
    x_min, x_max = all_data[:, 0].min() - 0.01, all_data[:, 0].max() + 0.01
    y_min, y_max = all_data[:, 1].min() - 0.01, all_data[:, 1].max() + 0.01
    h = .00005
    cm = ListedColormap(sns.color_palette(palette[:num_behaviours]).as_hex())
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = km_predict(np.c_[xx.ravel(), yy.ravel()], cluster_centers)
    Z = Z.reshape(xx.shape)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    plt.plot(all_data[:, 0], all_data[:, 1], '.', color='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.savefig(args.path + '/data_before_clustering.tiff', dpi=300)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    plt.contourf(xx, yy, Z, cmap=cm, alpha=0.5)
    for i in range(num_behaviours):
        x = cluster_data[i][:, 0]
        y = cluster_data[i][:, 1]
        plt.plot(x, y, '.', color=colors[i])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.savefig(args.path + '/data_in_clusters.tiff', dpi=300)
