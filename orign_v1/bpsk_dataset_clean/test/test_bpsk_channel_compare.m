%% TEST_BPSK_CHANNEL_COMPARE
% Compare one clean noncooperative BPSK burst with the same burst after
% Bellhop multipath propagation. Receiver noise is deliberately omitted.

clear; clc; close all;

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'bellhop_tools'), '-begin');

out = fullfile(root, 'test', 'bpsk_channel_compare');
channelDir = fullfile(out, 'channel');
if ~exist(out, 'dir'), mkdir(out); end
if ~exist(channelDir, 'dir'), mkdir(channelDir); end

%% One reproducible BPSK burst
rng(20260925, 'twister');
fs = 16000;
dur = 5;
sig = struct('type', 'BPSK', 'carrier_hz', 4000, ...
    'chips_per_sec', 1000, 'start_time_s', 1.0, ...
    'pulse_width_s', 2.0, 'amplitude', 0.95, 'fade_s', 0.01);
[tx, label, source] = generate_signal(sig, fs, dur);

%% Bellhop environment: T/S/depth -> SSP -> arrivals
z = [0; 10; 20; 40; 60; 80; 100];
temp = [25; 24.2; 22.8; 19.5; 16.8; 15.4; 15.0];
salt = [34.2; 34.3; 34.5; 34.6; 34.7; 34.8; 34.8];
sd = 20;
rd = 50;
rr = 1.0;
[SSP, Bdry, Pos, Beam, cInt, RMax, c] = build_ssp(z, temp, salt, sd, rd, rr);

id = 'bpsk_channel_test';
fc = signal_center_frequency(sig);
env = fullfile(channelDir, [id '.env']);
arr = fullfile(channelDir, [id '.arr']);
write_env(env, 'BELLHOP', 'BPSK channel test', fc, SSP, Bdry, Pos, Beam, cInt, RMax);

old = pwd;
restoreFolder = onCleanup(@() cd(old));
cd(channelDir);
bellhop(id);
assert(isfile(arr), 'Bellhop did not generate %s.', arr);
[Arr, ~] = read_arrivals_local(arr, 500);

%% Apply multipath only: no additive receiver noise in this test.
[rx, delayS, amp] = delayandsum(tx, fs, Arr, 1, 1, 1);

% Normalize only for WAV/listening and fair visual comparison. The factor
% is recorded below; relative multipath structure is not changed.
rxScale = 0.95 / max(abs(rx));
rxView = rx * rxScale;

cleanWav = fullfile(out, 'bpsk_clean.wav');
channelWav = fullfile(out, 'bpsk_bellhop_multipath.wav');
audiowrite(cleanWav, tx, fs, 'BitsPerSample', 16);
audiowrite(channelWav, rxView, fs, 'BitsPerSample', 16);

%% One figure: clean source versus Bellhop multipath result.
png = fullfile(out, 'bpsk_clean_vs_channel_spectrogram.png');
figure('Color', 'w', 'Position', [100 100 1100 850]);

subplot(2, 1, 1);
spectrogram(tx, hann(512, 'periodic'), 448, 1024, fs, 'yaxis');
ylim([0 fs/2]/1000); colorbar;
title('Clean noncooperative BPSK source');
xlabel('Time (s)'); ylabel('Frequency (kHz)');

subplot(2, 1, 2);
spectrogram(rxView, hann(512, 'periodic'), 448, 1024, fs, 'yaxis');
ylim([0 fs/2]/1000); colorbar;
title(sprintf('After Bellhop multipath: %d arrivals, delay spread %.1f ms', ...
    numel(delayS), 1000*(max(delayS)-min(delayS))));
xlabel('Time (s)'); ylabel('Frequency (kHz)');

exportgraphics(gcf, png, 'Resolution', 180);

meta.label = label;
meta.source = source;
meta.channel = struct('simulator', 'Bellhop', 'multipath_count', numel(delayS), ...
    'path_delay_s', delayS(:).', 'path_amplitude_real', real(amp(:)).', ...
    'path_amplitude_imag', imag(amp(:)).', ...
    'delay_spread_ms', 1000*(max(delayS)-min(delayS)), ...
    'receiver_noise_added', false, 'wav_normalization_scale', rxScale);
meta.ssp = struct('depth_m', z.', 'temperature_c', temp.', ...
    'salinity_psu', salt.', 'sound_speed_mps', c.');
fid = fopen(fullfile(out, 'comparison.json'), 'w');
assert(fid ~= -1, 'Cannot write comparison metadata.');
closer = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(meta));

fprintf('Created:\n%s\n%s\n%s\n', cleanWav, channelWav, png);
