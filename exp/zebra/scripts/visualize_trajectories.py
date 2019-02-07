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
import scipy.misc
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from scipy import ndimage


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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data

# def main():
#     x = np.linspace(0, 10, 20)
#     y = np.cos(x)
#     image_path = get_sample_data('ada.png')
#     fig, ax = plt.subplots()
#     imscatter(x, y, image_path, zoom=0.1, ax=ax)
#     ax.plot(x, y)
#     plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the positions of the fish accompanied by the feature information')
    parser.add_argument('--positions', '-p', type=str,
                        help='Path to the trajectory file',
                        required=True)
    parser.add_argument('--velocities', '-v', action='store_true', default=False,
                        help='Path to the velocity file',
                        required=True)
    parser.add_argument('--features', '-f', nargs='+', type=str, default='',
                        help='List of features to load')
    parser.add_argument('--out-dir', '-o', type=str,
                        help='Output directory name',
                        required=True)
    parser.add_argument('--virtual', type=str,
                        help='Path to virtual trajectories')
    parser.add_argument('--virtual-features', '-vf', nargs='+', type=str, default='',
                        help='List of virtual features to load')
    parser.add_argument('--virtual-velocities', type=str,
                        help='Path to virtual velocities')
    args = parser.parse_args()

    image_path = '/home/vpapaspy/Workspaces/mobots_ws/Research/Ethograms/aegean/exp/zebra/scripts/fish.png'
    image = plt.imread(image_path)

    image_path = '/home/vpapaspy/Workspaces/mobots_ws/Research/Ethograms/aegean/exp/zebra/scripts/robot.png'
    rimage = plt.imread(image_path)

    traj = np.loadtxt(args.positions)
    tsteps = traj.shape[0]
    if args.virtual:
        ex_idx = int(args.virtual.split('/')[1].split('_')[5])
        vtraj = np.loadtxt(args.virtual)
        if args.virtual_velocities:
            vvel = np.loadtxt(args.virtual_velocities)
    if args.velocities:
        rolledPos = np.roll(traj, -1, axis=0)
        vel = rolledPos - traj

    iradius = 0.19
    oradius = 0.29
    center = (0.570587, 0.574004)

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

        fig = plt.figure(figsize=(5, 5))
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

        for inum, j in enumerate(range(int(traj.shape[1] / 2))):
            x = traj[i, j*2]
            y = traj[i, j*2+1]
            plt.scatter(x, y, marker='.',
                        label='Individual ' + str(inum) + ' ' + str(x) + ' ' + str(y))
            # Q = plt.quiver(
            #     x, y, vel[i,  j*2], vel[i,  j*2+1], scale=1, units='xy')
            # phi = np.arctan2(vel[i,  j*2+1], vel[i,  j*2]) * 180 / np.pi
            # rotated_img = ndimage.rotate(image, phi)
            # ax.imshow(rotated_img, extent=[x-0.03, x, y -
            #                                0.015, y+0.015], aspect='equal')

            # text = get_text(feature_list)
            # text = 'Living agents\n' + text
            # plt.text(0.4, 0.1, text, horizontalalignment='left',
            #          verticalalignment='center', transform=ax.transAxes)
            # fig.patch.set_visible(False)

            # if args.virtual_features:
            #     vtext = get_text(vfeature_list)
            #     vtext = 'Mixed agents\n' + vtext
            #     plt.text(0.4, 0.0, vtext, horizontalalignment='left',
            #              verticalalignment='center', transform=ax.transAxes)
            #     fig.patch.set_visible(False)

        if args.virtual:
            x = vtraj[i, ex_idx*2]
            y = vtraj[i, ex_idx*2 + 1]
            plt.scatter(x, y, marker='.',
                        label='Virtual agent')
            # phi = np.arctan2(vvel[i,  ex_idx*2+1],
            #                  vvel[i,  ex_idx*2]) * 180 / np.pi
            # rotated_img = ndimage.rotate(image, phi)
            # ax.imshow(rotated_img, extent=[x-0.03, x, y -
            #                                0.015, y+0.015], aspect='equal')
            # if args.virtual_velocities:
            #     Q = plt.quiver(
            #         x, y, vvel[i,  ex_idx*2], vvel[i,  ex_idx*2 + 1], scale=1, units='xy')
        ax.axis('off')
        # ax.invert_yaxis()

        # plt.legend(bbox_to_anchor=(0.93, 1.16),
        #            bbox_transform=ax.transAxes)
        ax.set_xlim((0.25, 0.9))
        ax.set_ylim((0.25, 0.9))
        plt.tight_layout()
        plt.savefig(
            str(png_fname) + '.jpg',
            # transparent=True
            dpi=300
        )

        plt.close('all')
