function manifest = generate_cw_dataset()
%GENERATE_CW_DATASET Create a small, extensible CW Bellhop dataset.
%   This driver defines only the parameter set.  The common synthesis,
%   Bellhop propagation, WAV, spectrogram, JSON and manifest work are in
%   generate_bellhop_sample.m.  Add more entries to samples to scale up.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'bellhop_tools'), '-begin');

datasetName = 'cw_two_sample_demo';
dataRoot = fullfile(root, 'output', 'datasets', datasetName);
manifest = fullfile(dataRoot, 'manifest.jsonl');
if ~exist(dataRoot, 'dir'), mkdir(dataRoot); end
numSamples = 2;              % Change this number for a larger CW dataset.

% The manifest is rebuilt from the parameter list below on every run.
fid = fopen(manifest, 'w');
assert(fid ~= -1, 'Cannot write %s.', manifest);
fclose(fid);

rng(20260919, 'twister');
samples = make_cw_parameter_set(numSamples);
for k = 1:numel(samples)
    meta = generate_bellhop_sample(samples(k), dataRoot, datasetName);
    appendLine(manifest, jsonencode(meta));
    fprintf('[%d/%d] %s\n', k, numel(samples), meta.audio_path);
end

fprintf('Dataset manifest created:\n%s\n', manifest);
end

function appendLine(path, text)
fid = fopen(path, 'a');
assert(fid ~= -1, 'Cannot append to %s.', path);
closer = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', text);
end
