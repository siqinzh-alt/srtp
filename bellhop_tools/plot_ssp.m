function plot_ssp(SSP, png)
%PLOT_SSP Plot the range-independent sound-speed profile used by Bellhop.

z = SSP.raw(1).z(:);
c = SSP.raw(1).alphaR(:);
figure('Color', 'w', 'Position', [100 100 620 500]);
plot(c, z, '-o', 'LineWidth', 1.4, 'MarkerSize', 5);
set(gca, 'YDir', 'reverse'); grid on;
xlabel('Sound speed (m/s)'); ylabel('Depth (m)');
title('Sound-speed profile used by Bellhop');
exportgraphics(gcf, png, 'Resolution', 160);
end
