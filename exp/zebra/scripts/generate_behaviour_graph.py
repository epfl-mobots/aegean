#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate behaviour graphs')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment directory',
                        default='.',
                        required=False)
    parser.add_argument('--behaviour-duration', '-d', type=int,
                        help='Duration of each cluster sample', required=True)
    parser.add_argument('--fontsize', type=int,
                        help='Font size used in the graph labels',
                        default=30)
    parser.add_argument('--font', type=str,
                        help='Font used in the graph labels',
                        default='Arial')
    parser.add_argument('--pen-coef', type=float,
                        help='Width value coefficient for the edge weights/transition probabilities',
                        default=7)
    parser.add_argument('--node-size-coef', type=int,
                        help='Size value coefficient for the nodes ' +
                             '(each node gets a size that corresponds to the total time spent in it throught an experiment)',
                        default=0.7)
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore")

    tp_files = glob.glob(args.path + '/seg_*_transition_probabilities.dat')
    bp_files = glob.glob(args.path + '/seg_*_behaviour_probabilities.dat')
    labels = glob.glob(args.path + '/seg_*_labels.dat')

    tp_files.sort()
    bp_files.sort()
    labels.sort()
    assert len(tp_files) == len(bp_files) and len(tp_files) == len(labels)

    for i in range(len(tp_files)):
        transition_probs = np.loadtxt(tp_files[i])
        behaviour_probs = np.loadtxt(bp_files[i])
        lbls = np.loadtxt(labels[i], dtype=np.int32)

        # set graph attributes
        g = nx.DiGraph()
        g.graph['graph'] = {
            'labelloc': 't',
            'bgcolor': 'transparent',
            'fontname': args.font,
            'fontsize': args.fontsize}

        # add nodes and edges
        g.add_nodes_from(list(range(transition_probs.shape[0])))
        for m in range(transition_probs.shape[0]):
            for n in range(transition_probs.shape[1]):
                if transition_probs[m, n] == 0:
                    continue
                else:
                    g.add_edge(m, n, weight=transition_probs[m, n])
                    g[m][n]['penwidth'] = transition_probs[m, n] * args.pen_coef

        # beatify nodes
        for m in range(behaviour_probs.shape[0]):
            g.node[m]['style'] = '"filled, setlinewidth(0)"'
            g.node[m]['fontname'] = args.font
            g.node[m]['fontsize'] = args.fontsize
            g.node[m]['shape'] = 'circle'
            g.node[m]['fillcolor'] = 'dodgerblue'
            g.node[m]['margin'] = behaviour_probs[m] * \
                args.node_size_coef
        nx.draw(g)

        out_path = args.path + '/seg_' + \
            labels[i].split('/')[-1].split('_')[1] + '_graphs'
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        # set node attributes per timestep
        for tstep, k in enumerate(lbls.tolist()):
            for l in range(args.behaviour_duration):
                g_copy = g.copy()
                g_copy.graph['graph']['label'] = 'Time spent in current state: ' + \
                    str(l + 1) + 's'
                g_copy.node[k]['fillcolor'] = 'green'
                frame_name = str(tstep * args.behaviour_duration + l).zfill(6)
                path = out_path + '/' + frame_name + '.dot'
                nx.nx_pydot.write_dot(g_copy, path)
