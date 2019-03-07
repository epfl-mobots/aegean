clear
clc

num_clusters = 8;

hold on;

for i = 0:num_clusters-1
    str = 'cluster_';
    str = strcat(str, int2str(i));
    str = strcat(str, '_data.dat');
    clusters = load(str);
    i

    plot(clusters(:, 1), clusters(:, 2), '.')
end

hold off;