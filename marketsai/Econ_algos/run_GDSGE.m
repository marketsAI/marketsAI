%% Preamble (run only once per sessions
clear all
addpath(genpath(pwd));
parpool(3);

%% Globals
clear all
options = struct;
options.NumThreads = 3;
options.SaveFreq = 100;

beta  = 0.98;		% discount factor
sigma = 1.0;		% CRRA coefficient
alpha = 0.3;		% capital share
delta = 0.04;		% depreciation rate
phi = 0.5;          % adj. cost parameter

%
%% 1 hh
% small grid (5 points)
n_hh = 1; 
options.Kss  = (alpha * beta / (phi*delta*n_hh*(1-beta*(1-delta))))^(1/(2-alpha));
options.KMin = options.Kss*0.5;
options.KMax = options.Kss*1.5;
tStart = tic;
options.KPts =5;
options.K    = linspace(options.KMin,options.KMax,options.KPts);
IterRslt = iter_capital_planner_1hh(options);
IterRslt.timeElapsed=toc(tStart);
save('cap_plan_1hh_5pts.mat', 'IterRslt')

% med grid (11 points)
tStart = tic;
options.KPts = 11;
options.K    = linspace(options.KMin,options.KMax,options.KPts);
IterRslt = iter_capital_planner_2hh(options);
IterRslt.timeElapsed=toc(tStart);
save('cap_plan_1hh_11pts.mat', 'IterRslt')

% large grid (21 points)
options.KPts = 21;
options.K    = linspace(options.KMin,options.KMax,options.KPts);
IterRslt = iter_capital_planner_2hh(options);
IterRslt.timeElapsed=toc(tStart);
save('cap_plan_1hh_21pts.mat', 'IterRslt')

%% 2 hh
% small grid (5 points)
n_hh = 2; 
options.Kss  = (alpha * beta / (phi*delta*n_hh*(1-beta*(1-delta))))^(1/(2-alpha));
options.KMin = options.Kss*0.5;
options.KMax = options.Kss*1.5;
tStart = tic;
options.KPts =5;
options.K_1    = linspace(options.KMin,options.KMax,options.KPts);
options.K_2    = linspace(options.KMin,options.KMax,options.KPts);
IterRslt = iter_capital_planner_2hh(options);
IterRslt.timeElapsed=toc(tStart);
save('cap_plan_2hh_5pts.mat', 'IterRslt')

% med grid (11 points)
tStart = tic;
options.KPts = 11;
options.K_1    = linspace(options.KMin,options.KMax,options.KPts);
options.K_2    = linspace(options.KMin,options.KMax,options.KPts);
IterRslt = iter_capital_planner_2hh(options);
IterRslt.timeElapsed=toc(tStart);
save('cap_plan_2hh_11pts.mat', 'IterRslt')

% large grid (21 points)
options.KPts = 21;
options.K_1    = linspace(options.KMin,options.KMax,options.KPts);
options.K_2    = linspace(options.KMin,options.KMax,options.KPts);
IterRslt = iter_capital_planner_2hh(options);
IterRslt.timeElapsed=toc(tStart);
save('cap_plan_2hh_21pts.mat', 'IterRslt')

%% 3 hh
% small grid (5 points)
n_hh = 3; 
options.Kss  = (alpha * beta / (phi*delta*n_hh*(1-beta*(1-delta))))^(1/(2-alpha));
options.KMin = options.Kss*0.5;
options.KMax = options.Kss*1.5;
tStart = tic;
options.KPts =5;
options.K_1    = linspace(options.KMin,options.KMax,options.KPts);
options.K_2    = linspace(options.KMin,options.KMax,options.KPts);
options.K_3    = linspace(options.KMin,options.KMax,options.KPts);
IterRslt = iter_capital_planner_3hh(options);
IterRslt.timeElapsed=toc(tStart);
save('cap_plan_3hh_5pts.mat', 'IterRslt')

% med grid (11 points)
tStart = tic;
options.KPts = 11;
options.K_1    = linspace(options.KMin,options.KMax,options.KPts);
options.K_2    = linspace(options.KMin,options.KMax,options.KPts);
options.K_3    = linspace(options.KMin,options.KMax,options.KPts);
IterRslt = iter_capital_planner_3hh(options);
IterRslt.timeElapsed=toc(tStart);
save('cap_plan_3hh_11pts.mat', 'IterRslt')

% large grid (21 points)
options.KPts = 21;
options.K_1    = linspace(options.KMin,options.KMax,options.KPts);
options.K_2    = linspace(options.KMin,options.KMax,options.KPts);
options.K_3    = linspace(options.KMin,options.KMax,options.KPts);
IterRslt = iter_capital_planner_3hh(options);
IterRslt.timeElapsed=toc(tStart);
save('cap_plan_3hh_21pts.mat', 'IterRslt')

%% 4 hh
% small grid (5 points)
n_hh = 4; 
options.Kss  = (alpha * beta / (phi*delta*n_hh*(1-beta*(1-delta))))^(1/(2-alpha));
options.KMin = options.Kss*0.5;
options.KMax = options.Kss*1.5;

% tStart = tic;
% options.KPts = 5;
% options.K_1    = linspace(options.KMin,options.KMax,options.KPts);
% options.K_2    = linspace(options.KMin,options.KMax,options.KPts);
% options.K_3    = linspace(options.KMin,options.KMax,options.KPts);
% options.K_4    = linspace(options.KMin,options.KMax,options.KPts);
% IterRslt = iter_capital_planner_3hh(options);
% IterRslt.timeElapsed=toc(tStart);
% save('cap_plan_4hh_5pts.mat', 'IterRslt')

% % med grid (11 points)
% tStart = tic;
% options.KPts = 11;
% options.K_1    = linspace(options.KMin,options.KMax,options.KPts);
% options.K_2    = linspace(options.KMin,options.KMax,options.KPts);
% options.K_3    = linspace(options.KMin,options.KMax,options.KPts);
% options.K_4    = linspace(options.KMin,options.KMax,options.KPts);
% IterRslt = iter_capital_planner_3hh(options);
% IterRslt.timeElapsed=toc(tStart);
% save('cap_plan_4hh_11pts.mat', 'IterRslt')

% large grid (21 points)
options.KPts = 21;
tStart = tic;
options.K_1    = linspace(options.KMin,options.KMax,options.KPts);
options.K_2    = linspace(options.KMin,options.KMax,options.KPts);
options.K_3    = linspace(options.KMin,options.KMax,options.KPts);
options.K_4    = linspace(options.KMin,options.KMax,options.KPts);
IterRslt = iter_capital_planner_4hh(options);
IterRslt.timeElapsed=toc(tStart);
save('cap_plan_4hh_21pts.mat', 'IterRslt')

%% 5 hh
% small grid (5 points)
n_hh = 5; 
options.Kss  = (alpha * beta / (phi*delta*n_hh*(1-beta*(1-delta))))^(1/(2-alpha));
options.KMin = options.Kss*0.5;
options.KMax = options.Kss*1.5;
tStart = tic;
options.KPts =5;
options.K_1    = linspace(options.KMin,options.KMax,options.KPts);
options.K_2    = linspace(options.KMin,options.KMax,options.KPts);
options.K_3    = linspace(options.KMin,options.KMax,options.KPts);
options.K_4    = linspace(options.KMin,options.KMax,options.KPts);
options.K_5    = linspace(options.KMin,options.KMax,options.KPts);
IterRslt = iter_capital_planner_3hh(options);
IterRslt.timeElapsed=toc(tStart);
save('cap_plan_5hh_5pts.mat', 'IterRslt')

% med grid (11 points)
tStart = tic;
options.KPts = 11;
options.K_1    = linspace(options.KMin,options.KMax,options.KPts);
options.K_2    = linspace(options.KMin,options.KMax,options.KPts);
options.K_3    = linspace(options.KMin,options.KMax,options.KPts);
options.K_4    = linspace(options.KMin,options.KMax,options.KPts);
options.K_5    = linspace(options.KMin,options.KMax,options.KPts);
IterRslt = iter_capital_planner_3hh(options);
IterRslt.timeElapsed=toc(tStart);
save('cap_plan_5hh_11pts.mat', 'IterRslt')

% large grid (21 points)
options.KPts = 21;
options.K_1    = linspace(options.KMin,options.KMax,options.KPts);
options.K_2    = linspace(options.KMin,options.KMax,options.KPts);
options.K_3    = linspace(options.KMin,options.KMax,options.KPts);
options.K_4    = linspace(options.KMin,options.KMax,options.KPts);
options.K_5    = linspace(options.KMin,options.KMax,options.KPts);
IterRslt = iter_capital_planner_3hh(options);
IterRslt.timeElapsed=toc(tStart);
save('cap_plan_5hh_21pts.mat', 'IterRslt')


