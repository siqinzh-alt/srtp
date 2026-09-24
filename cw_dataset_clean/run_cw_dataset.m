%% 入口脚本
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'dataset_generator'));
generate_cw_dataset;
