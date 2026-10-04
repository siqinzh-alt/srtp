%% PLOT_SAMPLE_SSP_SECTION
% Rebuild and show the SSP used by one generated BPSK sample.

clear; clc; close all;

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'bellhop_tools'), '-begin');
datasetName = 'bpsk_noncoop_v1';
sampleId = 'bpsk_000001';

json = fullfile(root, 'output', 'datasets', datasetName, ...
    'metadata', [sampleId '.json']);
assert(isfile(json), 'Metadata JSON was not found: %s', json);
meta = jsondecode(fileread(json));

z = meta.ssp.depth_m(:);
temp = meta.ssp.temperature_c(:);
salt = meta.ssp.salinity_psu(:);
sd = meta.channel.source_depth_m;
rd = meta.channel.receiver_depth_m;
rr = meta.channel.range_km;
[SSP, ~, ~, ~, ~, ~, ~] = build_ssp(z, temp, salt, sd, rd, rr);

out = fullfile(root, 'test', 'one_bpsk_sample');
if ~exist(out, 'dir'), mkdir(out); end
png = fullfile(out, [sampleId '_ssp_section.png']);
plot_ssp_section(SSP, rr, png);

fprintf('Created SSP section:\n%s\n', png);
