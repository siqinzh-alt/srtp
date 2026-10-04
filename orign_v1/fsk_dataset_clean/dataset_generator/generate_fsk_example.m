function manifest = generate_fsk_example()
%GENERATE_FSK_EXAMPLE Produce one 2FSK sample through the Bellhop channel.
%   Output: WAV, spectrogram PNG, sidecar JSON, .env/.arr, manifest.jsonl.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'bellhop_tools'), '-begin');

datasetName = '2fsk_bellhop_demo';
dataRoot = fullfile(root, 'output', 'datasets', datasetName);
manifest = fullfile(dataRoot, 'manifest.jsonl');
if ~exist(dataRoot, 'dir'), mkdir(dataRoot); end

% Reproducible random FSK symbol sequence and noise.
rng(20260925, 'twister');

% 2FSK: symbol 0 uses 2800 Hz; symbol 1 uses 3600 Hz.
sample.id = '2fsk_000001';
sample.sig = struct('type', '2FSK', 'tone_hz', [2800, 3600], ...
    'symbol_rate_hz', 200, 'start_time_s', 1.0, ...
    'pulse_width_s', 2.0, 'fade_s', 0.01);
sample.fs = 16000;
sample.dur = 5;
sample.snr_db = 10;

% WOA observed temperature profile. Salinity is intentionally not used.
[sample.z, sample.temp, sample.woa] = ...
    pick_woa_temperature(fullfile(fileparts(root), 'WOA'), 100);
sample.sd = 20;
sample.rd = 50;
sample.rr = 1.0;

% Build one model-ready sample and write a one-line dataset manifest.
meta = generate_bellhop_sample(sample, dataRoot, datasetName);
fid = fopen(manifest, 'w');
assert(fid ~= -1, 'Cannot write %s.', manifest);
closer = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(meta));

fprintf('Created 2FSK Bellhop sample:\n%s\n', manifest);
end
