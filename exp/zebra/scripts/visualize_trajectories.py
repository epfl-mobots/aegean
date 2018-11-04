#!/usr/bin/env python

import os
import sys
import time
import socket
import warnings
import argparse
import datetime
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_text(fl):
    text = ''
    for pair in fl:
        if pair[0].lower() in ['alignment']:
            text += pair[0].capitalize() + ': ' + str('{0:.2f}'.format(
                np.asscalar(pair[1][i]) * 100)) + ' %\n'
        if pair[0].lower() in ['interindividual']:
            text += pair[0].capitalize() + ': ' + str('{0:.2f}'.format(
                np.asscalar(pair[1][i]) * 360)) + ' deg\n'
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the positions of the fish accompanied by the feature information')
    parser.add_argument('--positions', '-p', type=str,
                        help='Path to the trajectory file',
                        required=True)
    parser.add_argument('--features', '-f', nargs='+', type=str,
                        help='List of features to load')
    parser.add_argument('--out-dir', '-o', type=str,
                        help='Output directory name',
                        required=True)
    parser.add_argument('--virtual', type=str,
                        help='Path to virtual trajectories')
    parser.add_argument('--virtual-features', '-vf', nargs='+', type=str,
                        help='List of virtual features to load')
    args = parser.parse_args()

    traj = np.loadtxt(args.positions)
    tsteps = traj.shape[0]
    if args.virtual:
        vtraj = np.loadtxt(args.virtual)

    iradius = 0.19
    oradius = 0.29
    center = (0.58, 0.54)

    feature_list = []
    for f in args.features:
        if os.path.isfile(f):
            feature_name = f.split("_")[-1].split(".")[0]
            feature_list.append((feature_name, np.loadtxt(f)))
        else:
            warnings.warn("Skipping " + f)

    vfeature_list = []
    if args.virtual_features:
        for f in args.virtual_features:
            if os.path.isfile(f):
                feature_name = f.split("_")[-1].split(".")[0]
                vfeature_list.append((feature_name, np.loadtxt(f)))
            else:
                warnings.warn("Skipping " + f)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for i in range(tsteps):
        png_fname = args.out_dir + '/' + str(i).zfill(6)

        fig = plt.figure(figsize=(7, 7))
        ax = plt.gca()

        # drawing the circular setup
        radius = (iradius, oradius)
        # plt.plot([center[0]], [center[1]], ls='none',
        #  marker='o', color='black', label='Origin ' + str(center))
        inner = plt.Circle(
            center, radius[0], color='black', fill=False)
        outer = plt.Circle(
            center, radius[1], color='black', fill=False)
        ax.add_artist(inner)
        ax.add_artist(outer)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))

        for inum, j in enumerate(range(int(traj.shape[1] / 2))):
            x = traj[i, j*2]
            y = traj[i, j*2+1]
            plt.scatter(x, y, marker='.',
                        label='Individual ' + str(inum))

            text = get_text(feature_list)
            text = 'Living agents\n' + text
            plt.text(0.4, 0.1, text, horizontalalignment='left',
                     verticalalignment='center', transform=ax.transAxes)
            fig.patch.set_visible(False)

            if args.virtual_features:
                vtext = get_text(vfeature_list)
                vtext = 'Mixed agents\n' + vtext
                plt.text(0.4, 0.0, vtext, horizontalalignment='left',
                         verticalalignment='center', transform=ax.transAxes)
                fig.patch.set_visible(False)

        if args.virtual:
            x = vtraj[i, 0]
            y = vtraj[i, 1]
            plt.scatter(x, y, marker='x',
                        label='Virtual agent')

        ax.axis('off')
        ax.invert_yaxis()
        plt.legend(bbox_to_anchor=(1.1, 0.95),
                   bbox_transform=ax.transAxes)
        plt.savefig(
            str(png_fname) + '.png',
            transparent=True)
        plt.close('all')
