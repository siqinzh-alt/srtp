function [manifest, qaManifest] = generate_lfm_dataset()
%GENERATE_LFM_DATASET Create a small, extensible LFM Bellhop dataset.
%   This driver defines only the parameter set.  The common synthesis,
%   Bellhop propagation, WAV, spectrogram, JSON and manifest work are in
%   generate_bellhop_sample.m.  Add more entries to samples to scale up.
root = fileparts(fileparts(mfilename('fullpath')));
datasetName = 'lfm_sample';
dataRoot = fullfile(root, 'output', 'datasets', datasetName);
manifest = fullfile(dataRoot, 'manifest.jsonl');
qaManifest = fullfile(dataRoot, 'qa.jsonl'); 
if ~exist(dataRoot, 'dir'), mkdir(dataRoot); end
numSamples = 2000;

fid_full = fopen(manifest, 'w');
assert(fid_full ~= -1, 'Cannot write %s.', manifest);

fid_qa = fopen(qaManifest, 'w');
assert(fid_qa ~= -1, 'Cannot write %s.', qaManifest);

rng(randi([10000000,99999999]), 'twister');%删去固定种子随机生成
samples = make_lfm_parameter_set(numSamples);

for k = 1:numel(samples)
    meta = generate_bellhop_sample(samples(k), dataRoot, datasetName);
    line_full = jsonencode(meta);
    fprintf(fid_full, '%s\n', line_full);
    line_qa = jsonencode(meta.qa);
    fprintf(fid_qa, '%s\n', line_qa);
    fprintf('[%d/%d] %s\n', k, numel(samples), meta.audio_path);
end

fclose(fid_full);
fclose(fid_qa);

fprintf('Dataset manifest created:\n%s\n', manifest);
fprintf('QA-only jsonl created:\n%s\n', qaManifest);
end
