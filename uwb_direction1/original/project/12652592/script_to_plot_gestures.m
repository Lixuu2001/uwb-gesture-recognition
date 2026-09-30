%% Hand Gesture Data Set throu UWB Radar (Scientific Data) 
% Sample file to load and observe the data of human volunteer 1.
% In same way, other volunteer's data can be seen.
%% volunteer 1
% clear all
% close all
addpath(genpath(pwd))
load('HV_01_G1_ClutterRemoved.mat');  % gesture 1
load('HV_01_G1_ClutterRemoved.mat');  % gesture 1
load('HV_01_G2_ClutterRemoved.mat');  % gesture 2
load('HV_01_G3_ClutterRemoved.mat');  % gesture 3
load('HV_01_G4_ClutterRemoved.mat');  % gesture 4
load('HV_01_G5_ClutterRemoved.mat')   % gesture 5
load('HV_01_G6_ClutterRemoved.mat');  % gesture 6
load('HV_01_G7_ClutterRemoved.mat');  % gesture 7
load('HV_01_G8_ClutterRemoved.mat');  % gesture 8
load('HV_01_G9_ClutterRemoved.mat');  % gesture 9
load('HV_01_G10_ClutterRemoved.mat'); % gesture 10
load('HV_01_G11_ClutterRemoved.mat'); % gesture 11
load('HV_01_G12_ClutterRemoved.mat'); % gesture 12
% Load left radar data of gesture 1 is saved in variable:
% HV1_G2_RadarLeft_ClutterRemoved_100samples
% 90 rows constitute 1 sample, lets plot the 3rd sample

gesture_sample = 10; %which sample you want to see 

if gesture_sample >100 
    error('There are only 100 samples in this data set. chose value between 1 and 100')
else
    figure(1), subplot(1,3,1) , mesh(normalize(  HV1_G1_RadarLeft_ClutterRemoved_100samples ((90*gesture_sample)+15:((90*gesture_sample)+90),:),2,'range')),set(gca,'xtick',[],'ytick',[]), view([0 90])
    figure(1), subplot(1,3,2) , mesh(normalize(  HV1_G1_RadarTop_ClutterRemoved_100samples ((90*gesture_sample)+15:((90*gesture_sample)+90),:),2,'range')),set(gca,'xtick',[],'ytick',[]), view([0 90])
    figure(1), subplot(1,3,3) , mesh(normalize(  HV1_G1_RadarRight_ClutterRemoved_100samples ((90*gesture_sample)+15:((90*gesture_sample)+90),:),2,'range')),set(gca,'xtick',[],'ytick',[]), view([0 90])
end
% similarly, data for any user can and any gesture can be accessed