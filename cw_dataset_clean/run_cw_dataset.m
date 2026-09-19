%% RUN_CW_DATASET
% Entry point for this self-contained CW Bellhop dataset workspace.
% Change numSamples and datasetName in dataset_generator/generate_cw_dataset.m
% before running this file to create a larger or separately named dataset.

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'dataset_generator'));
generate_cw_dataset;
