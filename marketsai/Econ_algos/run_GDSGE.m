clear all
tic
% State which model
model = "capital_planner_2hh";
%path = '/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/Econ_algos/' + model;
script = 'iter_'+model;

%add path
%addpath '/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/Econ_algos/' + model;

% run iterations
ItrRslt = eval(script);

% Policy grid (I want a matrix with K rows and N_shocks as columns
K_1 = ItrRslt.var_state.K_1;
K_2 = ItrRslt.var_state.K_2;% 1*101
s_1 = ItrRslt.var_policy.s_1;
s_2 = ItrRslt.var_policy.s_2;% N_shocks * 101

toc

% grid = [K_1', K_2', s_1', s_2'];
% writematrix(grid,'cap_planner_2hh_econ.csv')
% %transpose both and a
% 
% %Simulate
% SimuRslt = simulate_capital_planner_1hh_2z(ItrRslt);
% 
% % tets
% z_idx_grid = [0, 1];
% z_grid = [0.99, 1.01];
% e_grid = [0.00, 0.3271];
% unemp_grid = [0.07, 0.00];  % unemployement transfer
% beta_grid = [0.9858 0.9894 0.9930];
% [beta,e,z] = ndgrid(beta_grid,e_grid,z_grid);
% [~,unemp,z_idx] = ndgrid(beta_grid,unemp_grid,z_idx_grid);
% beta=beta(:); e = e(:); unemp = unemp(:);
% z = z(:); z_idx = z_idx(:);




