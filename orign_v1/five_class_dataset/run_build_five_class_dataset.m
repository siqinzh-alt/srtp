clear;
clc; 
close all ;
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'generator'));
build_five_class_dataset;
