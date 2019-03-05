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
from PIL import Image


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
    parser.add_argument('--path', '-p', type=str,
                        help='Folder path',
                        required=True)
    parser.add_argument('--segment', '-s', type=int,
                        help='Number of experiment',
                        required=True)
    parser.add_argument('--excluded-idx', '-e', type=int,
                        help='Number of experiment',
                        default=-1,
                        required=False)
    parser.add_argument('--features', '-f', nargs='+', type=str, default='',
                        help='List of features to load')
    parser.add_argument('--virtual-features', '-vf', nargs='+', type=str, default='',
                        help='List of virtual features to load')
    parser.add_argument('--out-dir', '-o', type=str,
                        help='Output directory name',
                        required=True)
    parser.add_argument('--fish-like', action='store_true',
                        help='Images instead of points',
                        default=False)
    parser.add_argument('--turing', action='store_true',
                        help='Same image for all individuals to perform a turing test',
                        default=False)
    parser.add_argument('--info', action='store_true',
                        help='Display info',
                        default=False)
    args = parser.parse_args()

    iradius = 0.19
    oradius = 0.29
    center = (0.570587, 0.574004)

    image_path = '/home/vpapaspy/Workspaces/mobots_ws/Research/Ethograms/aegean/exp/zebra/scripts/fish.png'
    image = Image.open(image_path)
    image_path = '/home/vpapaspy/Workspaces/mobots_ws/Research/Ethograms/aegean/exp/zebra/scripts/excluded.png'
    excluded_image = Image.open(image_path)
    image_path = '/home/vpapaspy/Workspaces/mobots_ws/Research/Ethograms/aegean/exp/zebra/scripts/excluded_t_1.png'
    excluded_image_t_1 = Image.open(image_path)
    image_path = '/home/vpapaspy/Workspaces/mobots_ws/Research/Ethograms/aegean/exp/zebra/scripts/robot.png'
    rimage = Image.open(image_path)

    traj = np.loadtxt(args.path + '/seg_' +
                      str(args.segment) + '_reconstructed_positions.dat')
    vel = np.loadtxt(args.path + '/seg_' +
                     str(args.segment) + '_reconstructed_velocities.dat')
    tsteps = traj.shape[0]

    rtraj = np.roll(traj, 1, axis=0)
    rvel = np.roll(vel, 1, axis=0)

    if args.excluded_idx >= 0:
        vtraj = np.loadtxt(args.path + '/seg_' + str(args.segment) +
                           '_virtual_traj_ex_' + str(args.excluded_idx) + '_extended_positions.dat')
        vvel = np.loadtxt(args.path + '/seg_' + str(args.segment) +
                          '_virtual_traj_ex_' + str(args.excluded_idx) + '_extended_velocities.dat')

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
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()

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

            if not args.fish_like:
                plt.scatter(x, y, marker='.',
                            label='Individual ' + str(inum) + ' ' + "{:.2f}".format(x) + ' ' + "{:.2f}".format(y))
                Q = plt.quiver(
                    x, y, vel[i,  j*2], vel[i,  j*2+1], scale=1, units='xy')
            else:
                phi = np.arctan2(vel[i,  j*2+1], vel[i,  j*2]) * 180 / np.pi

                if args.excluded_idx >= 0 and args.excluded_idx == j and not args.turing:
                    rotated_img = excluded_image.rotate(phi)
                else:
                    rotated_img = image.rotate(phi)
                ax.imshow(rotated_img, extent=[x-0.0175, x+0.0175, y -
                                               0.0175, y+0.0175], aspect='equal')

        if args.info:
            text = get_text(feature_list)
            text = 'Living agents\n' + text
            plt.text(0.4, 0.1, text, horizontalalignment='left',
                     verticalalignment='center', transform=ax.transAxes)
            fig.patch.set_visible(False)

        if args.info:
            vtext = get_text(vfeature_list)
            vtext = 'Mixed agents\n' + vtext
            plt.text(0.4, 0.0, vtext, horizontalalignment='left',
                     verticalalignment='center', transform=ax.transAxes)
            fig.patch.set_visible(False)

        if args.excluded_idx >= 0:
            x = vtraj[i, args.excluded_idx*2]
            y = vtraj[i, args.excluded_idx*2 + 1]

            if not args.fish_like:
                plt.scatter(x, y, marker='x',
                            label='Virtual agent')
                Q = plt.quiver(
                    x, y, vvel[i,  args.excluded_idx*2], vvel[i,  args.excluded_idx*2 + 1], scale=1, units='xy')
            else:
                phi = np.arctan2(vvel[i,  args.excluded_idx*2+1],
                                 vvel[i,  args.excluded_idx*2]) * 180 / np.pi
                if not args.turing:
                    rotated_img = rimage.rotate(phi)
                else:
                    rotated_img = image.rotate(phi)
                ax.imshow(
                    rotated_img, extent=[x-0.0175, x+0.0175, y -
                                         0.0175, y+0.0175], aspect='equal')

            # xx = rtraj[i, args.excluded_idx*2]
            # yy = rtraj[i, args.excluded_idx*2 + 1]

            # if not args.fish_like:
            #     plt.scatter(xx, yy, marker='*',
            #                 label='Virtual agent')
            #     Q = plt.quiver(
            #         xx, yy, rvel[i,  args.excluded_idx*2], rvel[i,  args.excluded_idx*2 + 1], scale=1, units='xy')
            # else:
            #     phi = np.arctan2(rvel[i,  args.excluded_idx*2+1],
            #                      rvel[i,  args.excluded_idx*2]) * 180 / np.pi
            #     rotated_img = excluded_image_t_1.rotate(phi)
            #     ax.imshow(
            #         rotated_img, extent=[xx-0.0175, xx+0.0175, yy -
            #                              0.0175, yy+0.0175], aspect='equal')

        ax.axis('off')

        if args.info:
            plt.legend(bbox_to_anchor=(0.93, 1.16),
                       bbox_transform=ax.transAxes)
        ax.set_xlim((0.25, 0.9))
        ax.set_ylim((0.25, 0.9))
        plt.tight_layout()

        png_fname = args.out_dir + '/' + str(i).zfill(6)
        plt.savefig(
            str(png_fname) + '.jpg',
            # transparent=True
            dpi=300
        )
        plt.close('all')
