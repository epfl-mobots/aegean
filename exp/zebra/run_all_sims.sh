#!/bin/sh

sim='../../build/zebra_mixed_sim'
# sim='../../build/zebra_mixed_sim_original_labels'

dir=$1
num_experiments=$2
num_individuals=$3

for ((i=0; i<$num_experiments; i++)); do
    echo "Running simulation for experiment $i"
    seg_file="$dir/seg_$i""_feature_matrix.dat"
    for ((j=0; j<$num_individuals; j++)); do
        echo "  - Replacing individual $j"
        $sim $dir $i $j
        ind_file="$dir/seg_$i""_virtual_traj_ex_$j""_feature_matrix.dat"
        python3 scripts/compute_behaviour_deviation.py --centroids "$dir/centroids_kmeans.dat" -fm1 $seg_file -fm2 $ind_file
        # python3 scripts/compute_behaviour_deviation.py --centroids "/home/vpapaspy/Workspaces/mobots_ws/Research/Ethograms/aegean/1hl_2feat_3s_loss_variant/centroids_kmeans.dat" -fm1 $seg_file -fm2 $ind_file
    done
done



