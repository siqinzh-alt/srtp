function plotarr_local(delayS, amp, Pos, png)
%PLOTARR_LOCAL Arrival-amplitude view for the single receiver in this demo.
%   This is the relevant single-receiver subset of Bellhop's plotarr.m.

delayMs = delayS(:) * 1000;
figure('Color', 'w', 'Position', [100 100 760 450]);
stem(delayMs, abs(amp(:)), 'filled', 'LineWidth', 1.1);
grid on;
xlabel('Arrival delay (ms)'); ylabel('|Complex arrival amplitude|');
title(sprintf('Bellhop arrivals: source %.1f m, receiver %.1f m, range %.2f km', ...
    Pos.s.depth(1), Pos.r.depth(1), Pos.r.range(1)));
exportgraphics(gcf, png, 'Resolution', 160);
end
