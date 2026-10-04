%% PLOT_BPSK_BEGINNER_COMPARISON
% Four beginner-friendly views of the BPSK multipath effect.

clear; clc; close all;

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'bellhop_tools'), '-begin');
out = fullfile(root, 'test', 'bpsk_channel_compare');

[tx, fs] = audioread(fullfile(out, 'bpsk_clean.wav'));
[rx, fsRx] = audioread(fullfile(out, 'bpsk_bellhop_multipath.wav'));
assert(fs == fsRx, 'The two WAV files must have the same sample rate.');
[Arr, ~] = read_arrivals_local(fullfile(out, 'channel', 'bpsk_channel_test.arr'), 500);

delayS = real(squeeze(Arr.delay(1, 1:Arr.Narr(1,1,1), 1, 1)));
amp = squeeze(Arr.A(1, 1:Arr.Narr(1,1,1), 1, 1));
t = (0:numel(tx)-1)'/fs;

% A 20 ms moving energy window makes the arrival and tail easy to see.
energyWindow = round(0.02*fs);
eTx = movmean(tx.^2, energyWindow);
eRx = movmean(rx.^2, energyWindow);
eTx = eTx/max(eTx);
eRx = eRx/max(eRx);

% Correlation asks: at which delay does the received signal resemble the
% clean transmitted signal? Multiple peaks indicate multiple arrivals.
[corrValue, lags] = xcorr(rx, tx);
corrValue = abs(corrValue)/max(abs(corrValue));
lagS = lags(:)/fs;
keep = lagS >= 0.5 & lagS <= 0.9;

png = fullfile(out, 'bpsk_beginner_comparison.png');
figure('Color', 'w', 'Position', [80 80 1300 850]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(t, tx, 'Color', [0.15 0.45 0.85]); hold on;
plot(t, rx, 'Color', [0.85 0.25 0.2]);
grid on; xlim([0 5]); ylim([-1 1]);
title('1. Waveform: clean signal and channel-affected signal');
xlabel('Time (s)'); ylabel('Amplitude (normalized)');
legend('Clean BPSK', 'After Bellhop multipath', 'Location', 'best');

nexttile;
plot(t, eTx, 'LineWidth', 1.3, 'Color', [0.15 0.45 0.85]); hold on;
plot(t, eRx, 'LineWidth', 1.3, 'Color', [0.85 0.25 0.2]);
grid on; xlim([0 5]); ylim([0 1.05]);
title('2. Energy over time: delay and echo tail');
xlabel('Time (s)'); ylabel('Relative energy');
legend('Clean BPSK', 'After Bellhop multipath', 'Location', 'best');

nexttile;
stem(delayS*1000, abs(amp)/max(abs(amp)), 'filled', 'LineWidth', 1.1);
grid on;
title(sprintf('3. Bellhop arrivals: %d paths', numel(delayS)));
xlabel('Arrival delay (ms)'); ylabel('Relative path strength');

nexttile;
plot(lagS(keep)*1000, corrValue(keep), 'k', 'LineWidth', 1.2);
grid on; ylim([0 1.05]);
title('4. Correlation: multiple peaks mean multiple echoes');
xlabel('Delay relative to clean BPSK (ms)'); ylabel('Normalized correlation');

exportgraphics(gcf, png, 'Resolution', 180);
fprintf('Created beginner comparison:\n%s\n', png);
