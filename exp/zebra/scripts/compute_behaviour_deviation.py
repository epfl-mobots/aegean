#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_labels(centroids, fm):
    distances = np.ones((fm.shape[0], centroids.shape[0])) * -1
    for r in range(centroids.shape[0]):
        c = centroids[r, :]
        diff = fm - c
        distances[:, r] = np.linalg.norm(diff, axis=1)
    labels = np.argmax(distances, axis=1)
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute the difference between the behaviour exhibited by living agents and the mixed group containing virtual fish')
    parser.add_argument('--centroids', '-c', type=str,
                        help='Ethogram centroids',
                        required=True)
    parser.add_argument('--feature-matrix-1', '-fm1', type=str,
                        help='Feature matrix of agents',
                        required=True)
    parser.add_argument('--feature-matrix-2', '-fm2', type=str,
                        help='Feature matrix of agents',
                        required=True)
    args = parser.parse_args()

    centroids = np.loadtxt(args.centroids)
    fm1 = np.loadtxt(args.feature_matrix_1)
    fm2 = np.loadtxt(args.feature_matrix_2)
    assert centroids.shape[1] == fm1.shape[1] and centroids.shape[1] == fm2.shape[1], 'Dimensions don\'t match'

    lbls1 = get_labels(centroids, fm1)
    lbls2 = get_labels(centroids, fm2)

    count = 0
    for i in range(len(lbls1) - 1):
        if lbls1[i] == lbls2[i+1]:
            count += 1
    print(count / len(lbls1))
    fm_error = fm1 - fm2
    print(np.linalg.norm(fm_error, axis=0))
