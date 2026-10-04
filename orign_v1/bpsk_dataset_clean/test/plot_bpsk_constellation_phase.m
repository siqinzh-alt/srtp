%% PLOT_BPSK_CONSTELLATION_PHASE
% Constellation and phase views for the BPSK clean/channel comparison.

clear; clc; close all;

root = fileparts(fileparts(mfilename('fullpath')));
out = fullfile(root, 'test', 'bpsk_channel_compare');

[tx, fs] = audioread(fullfile(out, 'bpsk_clean.wav'));
[rx, fsRx] = audioread(fullfile(out, 'bpsk_bellhop_multipath.wav'));
assert(fs == fsRx, 'The two WAV files must have the same sample rate.');
meta = jsondecode(fileread(fullfile(out, 'comparison.json')));

fc = meta.source.carrier_hz;
chipRate = meta.source.chips_per_sec;
txStart = meta.source.start_time_s;
width = meta.source.pulse_width_s;
rxStart = txStart + min(meta.channel.path_delay_s);

% Downconvert and integrate one carrier reference over each chip.
zTx = extractSymbols(tx, fs, fc, chipRate, txStart, width);
zRx = extractSymbols(rx, fs, fc, chipRate, rxStart, width);

% Remove only each signal's constant carrier phase and normalize power.
% This makes the BPSK two-point structure easy to compare.
zTx = alignBpsk(zTx);
zRx = alignBpsk(zRx);

png = fullfile(out, 'bpsk_constellation_phase_comparison.png');
figure('Color', 'w', 'Position', [80 80 1250 820]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
scatter(real(zTx), imag(zTx), 18, [0.1 0.4 0.85], 'filled');
grid on; axis equal; xlim([-1.5 1.5]); ylim([-1.5 1.5]);
xlabel('I (in-phase)'); ylabel('Q (quadrature)');
title('1. Clean BPSK constellation: two ideal clusters');

nexttile;
scatter(real(zRx), imag(zRx), 18, [0.85 0.25 0.2], 'filled');
grid on; axis equal;
xlabel('I (in-phase)'); ylabel('Q (quadrature)');
title('2. After Bellhop multipath: clusters spread and overlap');

nShow = min(100, numel(zTx));
nexttile;
plot(1:nShow, rad2deg(angle(zTx(1:nShow))), '.-', ...
    'Color', [0.1 0.4 0.85], 'MarkerSize', 11);
grid on; ylim([-200 200]);
xlabel('Chip index'); ylabel('Phase (degrees)');
title('3. Clean BPSK phase: mainly 0 degrees or ±180 degrees');

nexttile;
plot(1:nShow, rad2deg(angle(zRx(1:nShow))), '.-', ...
    'Color', [0.85 0.25 0.2], 'MarkerSize', 11);
grid on; ylim([-200 200]);
xlabel('Chip index'); ylabel('Phase (degrees)');
title('4. After multipath: phase is no longer limited to two values');

exportgraphics(gcf, png, 'Resolution', 180);
fprintf('Created constellation and phase comparison:\n%s\n', png);

function z = extractSymbols(x, fs, fc, chipRate, startS, width)
samplesPerChip = fs/chipRate;
assert(samplesPerChip == round(samplesPerChip), 'fs/chipRate must be an integer.');
numSymbols = round(width*chipRate);
first = round(startS*fs) + 1;
z = zeros(numSymbols, 1);

for k = 1:numSymbols
    idx = first + (k-1)*samplesPerChip + (0:samplesPerChip-1);
    t = (idx(:)-1)/fs;
    reference = exp(-1j*2*pi*fc*t);
    z(k) = mean(x(idx(:)).*reference);
end
end

function z = alignBpsk(z)
% For BPSK, squaring removes the +/- symbol sign and estimates phase.
phaseOffset = 0.5*angle(mean(z.^2));
z = z*exp(-1j*phaseOffset);
z = z/sqrt(mean(abs(z).^2));
end
