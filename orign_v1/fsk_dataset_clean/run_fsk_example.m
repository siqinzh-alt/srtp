%% RUN_FSK_EXAMPLE
% Entry point for the self-contained 2FSK Bellhop example.

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'dataset_generator'));
generate_fsk_example;
