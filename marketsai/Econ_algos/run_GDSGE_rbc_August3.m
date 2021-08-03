clear all

% State which model
model = "rbc";
%path = '/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/Econ_algos/' + model;
script = 'iter_'+model;

%add path
%addpath '/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/Econ_algos/' + model;

% run iterations
ItrRslt = eval(script);
% Policy grid (I want a matrix with K rows and N_shocks as columns
K = ItrRslt.var_state.K; % 1*101
c = ItrRslt.var_policy.c; % N_shocks * 101

grid = [K',c'];
writematrix(grid,'rbc_test.csv')
%transpose both and a

%plot the policy function for c
figure;
plot(K,c);
xlabel('K'); ylabel('c'); title('Policy Functions for c');

%plot the policy function for K_next
figure;
plot(K,ItrRslt.var_policy.K_next);
xlabel('K'); ylabel('K_next'); title('Policy Functions for K_next');

%Simulate
SimuRslt = simulate_rbc(ItrRslt);

%Some simulated results: sample paths for wages
figure;
plot(SimuRslt.w(1:6,1:100)'); 
xlabel('Periods'); ylabel('Wages'); title('Sample Paths of Wages');




