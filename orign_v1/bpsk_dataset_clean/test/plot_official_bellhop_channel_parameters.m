%% PLOT_OFFICIAL_BELLHOP_CHANNEL_PARAMETERS
% Uses the original Bellhop plotssp.m and plotarr.m for one BPSK sample.

clear; clc; close all;

root = fileparts(fileparts(mfilename('fullpath')));
sampleId = 'bpsk_000001';
datasetName = 'bpsk_noncoop_v1';
channelDir = fullfile(root, 'output', 'datasets', datasetName, 'channel');
env = fullfile(channelDir, [sampleId '.env']);
arr = fullfile(channelDir, [sampleId '.arr']);
assert(isfile(env) && isfile(arr), 'The selected sample has no .env or .arr file.');

% Original Bellhop plotting and reading functions from the user's library.
bellhopLibrary = 'D:\BELLHOP_General_Description\BELLHOP_General_Description';
taskDir = fullfile(bellhopLibrary, 'Task_GD_1_flat');
assert(isfolder(taskDir), 'Compatible original Bellhop task directory was not found.');
% This task's plotarr.m uses its matching 8-column ARR reader, which is
% compatible with the ARR files produced by the current Bellhop executable.
addpath(taskDir, '-begin');

out = fullfile(root, 'test', 'official_bellhop_plots', sampleId);
if ~exist(out, 'dir'), mkdir(out); end

% The older generic plotssp.m cannot read the current write_env.m format.
% The sample's correct SSP figure remains available from plot_ssp_section.m.
set(groot, 'defaultFigureVisible', 'off');
% Official Task_GD_1_flat/plotarr.m: three subplots in one figure.
existingFigures = findall(groot, 'Type', 'figure');
figure('Color', 'w', 'Position', [40 80 1800 600]);
plotarr(arr, 1, 1, 1);
newFigures = setdiff(findall(groot, 'Type', 'figure'), existingFigures);
newFigures = sort(newFigures);
for k = 1:numel(newFigures)
    exportgraphics(newFigures(k), fullfile(out, ...
        sprintf('task_gd1_original_plotarr_%d.png', k)), 'Resolution', 180);
    close(newFigures(k));
end

fprintf('Original Bellhop plots created in:\n%s\n', out);
