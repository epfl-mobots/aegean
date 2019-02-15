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
# palette = flatui
palette = 'Paired'
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate histogram plots')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the distribution file',
                        required=True)
    parser.add_argument('--min', type=float,
                        help='Minimum value of the histogram',
                        required=True)
    parser.add_argument('--max', type=float,
                        help='Maximum value of the histogram',
                        required=True)
    args = parser.parse_args()

    hist = np.loadtxt(args.path)
    num_bins = hist.shape[0]
    x = range(num_bins)
    xticklabels = np.linspace(args.min, args.max, num_bins)
    # x = list(map(int, x))
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    plt.bar(x, hist)
    ax.set_xticklabels([])
    plt.show()
