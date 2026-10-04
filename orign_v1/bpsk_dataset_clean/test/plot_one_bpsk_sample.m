%% PLOT_ONE_BPSK_SAMPLE
% Show the time-domain waveform and power spectrum for one BPSK dataset WAV.

clear; clc; close all;

root = fileparts(fileparts(mfilename('fullpath')));
datasetName = 'bpsk_noncoop_v1';
sampleId = 'bpsk_000001';       % Change this to inspect another sample.
dataRoot = fullfile(root, 'output', 'datasets', datasetName);
out = fullfile(root, 'test', 'one_bpsk_sample');
if ~exist(out, 'dir'), mkdir(out); end

wav = fullfile(dataRoot, 'audio', [sampleId '.wav']);
json = fullfile(dataRoot, 'metadata', [sampleId '.json']);
assert(isfile(wav) && isfile(json), 'Sample WAV or metadata JSON was not found.');
[x, fs] = audioread(wav);
meta = jsondecode(fileread(json));
t = (0:numel(x)-1)'/fs;

% Find a high-energy 20 ms window so BPSK chip changes are visible.
energy = movmean(x.^2, round(0.02*fs));
[~, center] = max(energy);
halfWindow = round(0.01*fs);
idx = max(1, center-halfWindow):min(numel(x), center+halfWindow);

% Power spectral density: this reveals the  carrier-centred BPSK band.
[pxx, f] = pwelch(x, hann(2048, 'periodic'), 1536, 4096, fs, 'power');

png = fullfile(out, [sampleId '_time_and_spectrum.png']);
figure('Color', 'w', 'Position', [80 80 1250 880]);
tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(t, x, 'Color', [0.15 0.35 0.8]);
grid on; xlim([0 meta.duration_s]);
title(sprintf('%s: received BPSK waveform (full %.1f s clip)', ...
    sampleId, meta.duration_s));
xlabel('Time (s)'); ylabel('Amplitude');

nexttile;
plot(t(idx)*1000, x(idx), 'Color', [0.85 0.25 0.2], 'LineWidth', 1);
grid on;
title(sprintf('20 ms time-domain zoom near signal energy (carrier %.0f Hz, %d chip/s)', ...
    meta.source.carrier_hz, meta.source.chips_per_sec));
xlabel('Time (ms)'); ylabel('Amplitude');

nexttile;
plot(f/1000, 10*log10(pxx + eps), 'k', 'LineWidth', 1.2);
grid on; xlim([0 fs/2]/1000);
xline(meta.source.carrier_hz/1000, '--r', 'Carrier frequency');
title('Power spectrum of received BPSK');
xlabel('Frequency (kHz)'); ylabel('Power/Frequency (dB/Hz)');

exportgraphics(gcf, png, 'Resolution', 180);
fprintf('Created:\n%s\n', png);
