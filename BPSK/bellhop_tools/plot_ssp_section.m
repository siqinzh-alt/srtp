function plot_ssp_section(SSP, rangeKm, png)
%PLOT_SSP_SECTION Plot a range-independent Bellhop sound-speed section.
%   The left panel is c(z). The right panel repeats c(z) over range,
%   making the current range-independent assumption visible.

z = SSP.raw(1).z(:);
c = SSP.raw(1).alphaR(:);

figure('Color', 'w', 'Position', [100 100 1100 450]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(c, z, '-o', 'LineWidth', 1.5, 'MarkerSize', 5);
set(gca, 'YDir', 'reverse'); grid on;
xlabel('Sound speed (m/s)'); ylabel('Depth (m)');
title('Sound-speed profile c(z)');

nexttile;
rangeM = linspace(0, rangeKm*1000, 120);
imagesc(rangeM/1000, z, repmat(c, 1, numel(rangeM)));
set(gca, 'YDir', 'reverse');
xlabel('Range (km)'); ylabel('Depth (m)');
title('SSP section used by Bellhop (range-independent)');
cb = colorbar;
cb.Label.String = 'Sound speed (m/s)';

exportgraphics(gcf, png, 'Resolution', 180);
end
