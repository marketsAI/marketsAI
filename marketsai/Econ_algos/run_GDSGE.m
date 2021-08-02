clear all

% State which model
model = "capital_planner_1hh_2z";
%path = '/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/Econ_algos/' + model;
script = 'iter_'+model;

%add path
%addpath '/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/Econ_algos/' + model;

% run iterations
ItrRslt = eval(script);
% Policy grid (I want a matrix with K rows and N_shocks as columns
K = ItrRslt.var_state.K; % 1*101
s = ItrRslt.var_policy.s; % N_shocks * 101

grid = [K',s'];
writematrix(grid,'cap_planner_1hh_econ.csv')
%transpose both and a

%Simulate
SimuRslt = simulate_capital_planner_1hh_2z(ItrRslt);




