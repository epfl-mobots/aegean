#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate behaviour sequence plots')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment directory',
                        default='.',
                        required=False)
    args = parser.parse_args()

    flatui = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    bseq_files = glob.glob(args.path + '/seg_*_behaviour_sequence.dat')
    for f in bseq_files:
        bseq = np.loadtxt(f, dtype=np.int32)
        num_behaviours = len(set(bseq[:, 0]))

        fig = plt.figure(figsize=(8, 6))
        duration = bseq[-1][2]
        seq_matrix = np.ones(shape=(num_behaviours, duration + 1))
        cmap = sns.color_palette(flatui)
        cmap += [(1, 1, 1)]
        for r in range(bseq.shape[0]):
            for idx in range(bseq[r, 1], bseq[r, 2] + 1):
                seq_matrix[bseq[r, 0], idx] = 1 - \
                    (bseq[r, 0] + 1) / (num_behaviours + 1)
        ax = sns.heatmap(seq_matrix, cbar=False, cmap=cmap)
        ax.invert_yaxis()
        ax.set_xlabel('Time in seconds')
        ax.set_ylabel('Behaviour cluster')

        out_path = args.path + '/seg_' + \
            f.split('/')[-1].split('_')[1] + '_behaviour_sequence.png'
        plt.savefig(out_path, dpi=300)
