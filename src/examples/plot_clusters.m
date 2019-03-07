clear
clc

num_clusters = 8;
c = colorcube(num_clusters)

hold on;

for i = 0:num_clusters-1
    str = 'cluster_';
    str = strcat(str, int2str(i));
    str = strcat(str, '_data.dat');
    clusters = load(str);

    h(i+1) = plot(clusters(:, 1), clusters(:, 2), 'o')
    set(h(i+1), 'Color', c(i+1, :))
end

hold off;