clc; clear; close all;
%% add path
addpath utils/;

%% 
load("data/trialInfos.mat");
load("data/encode_data.mat");
load("data/probe_data.mat");
load("data/cue_data.mat");
load("data/angles.mat");

%% encode phase
IEMAnalyze("encode", encode_data, infos_all, angles);

%% probe phase
IEMAnalyze("probe", probe_data, infos_all, angles);

%% cue phase
IEMAnalyze("cue", cue_data, infos_all, angles);
%%
% encode phase
SVMAnalyze("encode", encode_data, infos_all, angles);

% probe phase
SVMAnalyze("probe", probe_data, infos_all, angles);

% cue phase
SVMAnalyze("cue", cue_data, infos_all, angles);