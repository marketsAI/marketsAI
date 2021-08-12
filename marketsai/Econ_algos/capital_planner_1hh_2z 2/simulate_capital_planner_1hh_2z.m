function SimuRslt = simulate_capital_planner_1hh_2z(IterRslt,GNDSGE_OPTIONS)
%% Add path
if ispc
    BLAS_FILE = 'essential_blas.dll';
    PATH_DELIMITER = ';';
    
    GDSGE_TOOLBOX_ROOT = fileparts(which(BLAS_FILE));
    if ~any(strcmp(strsplit(getenv('PATH'),PATH_DELIMITER),GDSGE_TOOLBOX_ROOT))
        setenv('PATH',[getenv('PATH'),PATH_DELIMITER,GDSGE_TOOLBOX_ROOT]);
    end
    
    clear BLAS_FILE PATH_DELIMITER GDSGE_TOOLBOX_ROOT
elseif ismac
    if exist('./essential_blas.dylib','file') == 0
        copyfile(which('essential_blas.dylib'),'./');
    end
end

%% Simulate code starts here
TolEq = 1e-6;
TolSol = 1e-8;
TolFun = 1e-8;
PrintFreq = 10;
NoPrint = 0;
SaveFreq = 10;
NoSave = 0;
SimuPrintFreq = 1000;
SimuSaveFreq = inf;
NumThreads = feature('numcores');
MaxIter = inf;
MaxMinorIter = inf;
num_samples = 1;
num_periods = 1000;
SolMaxIter = 200;

% task constants
MEX_TASK_INIT = 0;
MEX_TASK_INF_HORIZON = 1;

% Solver
UseFiniteDiff = 0;
UseBroyden = 0;
FiniteDiffDelta = 1e-6;

% DEBUG flag
GNDSGE_DEBUG_EVAL_ONLY = 0;
GNDSGE_USE_BROYDEN = 1;
GNDSGE_USE_BROYDEN_NOW = 0;

v2struct(IterRslt.pp);

INTERP_ORDER = 4;
EXTRAP_ORDER = 2;
OutputInterpOrder = 2;
USE_SPLINE = 1;
GNDSGE_USE_OLD_VEC = 0;
USE_ASG = 0;
USE_PCHIP = 0;
SIMU_INTERP = 0;
SIMU_RESOLVE = 1;
SimuSeed = 0823;
AsgMinLevel = 4;
AsgMaxLevel = 10;
AsgThreshold = 1e-2;
AsgOutputMaxLevel = 10;
AsgOutputThreshold = 1e-2;
IterSaveAll = 0;
SkipModelInit = 0;
GNDSGE_EMPTY = [];
GNDSGE_ASG_FIX_GRID = 0;
UseModelId = 0;
MinBatchSol = 1;
UseAdaptiveBound = 1;
UseAdaptiveBoundInSol = 0;
EnforceSimuStateInbound = 1;
REMOVE_NULL_STATEMENTS = 0;
REUSE_WARMUP_SOL = 1;
shock_num = 1;
shock_trans = 1;
GNDSGE_dummy_state = 1;
GNDSGE_dummy_shock = 1;
DEFAULT_PARAMETERS_END_HERE = true;
beta  = 0.98;
sigma = 1.0;
alpha = 0.3;
delta = 0.04;
phi = 0.5;
shock_num = 4;
z_agg_low = 0.8;
z_agg_high = 1.2;
z_ind_low = 0.9;
z_ind_high = 1.1;
z_agg = [z_agg_low,z_agg_low,z_agg_high,z_agg_high];
z_ind = [z_ind_low,z_ind_high,z_ind_low,z_ind_high];
Pr_agg_ll = 0.95;
Pr_agg_hh  = 0.95;
Pr_ind_ll = 0.9;
Pr_ind_hh  = 0.9;
shock_trans_agg = [;
Pr_agg_ll, 1-Pr_agg_ll;
1-Pr_agg_hh, Pr_agg_hh;
];
shock_trans_ind = [;
Pr_ind_ll, 1-Pr_ind_ll;
1-Pr_ind_hh, Pr_ind_hh;
];
shock_trans = kron(shock_trans_agg, shock_trans_ind);
Kss  = (alpha * beta / (phi*delta*(1-beta*(1-delta))))^(1/(2-alpha));
KPts = 101;
KMin = Kss*0.5;
KMax = Kss*1.5;
K    = linspace(KMin,KMax,KPts);
num_periods = 10000;
num_samples = 6;


GEN_SHOCK_START_PERIOD = 1;

shock_trans = IterRslt.shock_trans;
v2struct(IterRslt.params);
v2struct(IterRslt.var_shock);
v2struct(IterRslt.var_state);

if nargin>=2
    v2struct(GNDSGE_OPTIONS)
end

%% Construct interpolation for solutions
if shock_num>1
    GNDSGE_PP=struct('form','MKL','breaks',{{1:shock_num,K}},...
        'Values',reshape(IterRslt.GNDSGE_PROB.GNDSGE_SOL, [numel(IterRslt.GNDSGE_PROB.GNDSGE_SOL)/prod(IterRslt.GNDSGE_PROB.GNDSGE_SIZE),IterRslt.GNDSGE_PROB.GNDSGE_SIZE]),...
        'coefs',[],'order',[2 OutputInterpOrder*ones(1,length({K}))],'Method',[],...
        'ExtrapolationOrder',[],'thread',NumThreads, ...
        'orient','curvefit');
else
    GNDSGE_PP=struct('form','MKL','breaks',{{K}},...
        'Values',reshape(IterRslt.GNDSGE_PROB.GNDSGE_SOL, [numel(IterRslt.GNDSGE_PROB.GNDSGE_SOL)/prod(IterRslt.GNDSGE_PROB.GNDSGE_SIZE),IterRslt.GNDSGE_PROB.GNDSGE_SIZE(2:end)]),...
        'coefs',[],'order',[OutputInterpOrder*ones(1,length({K}))],'Method',[],...
        'ExtrapolationOrder',[],'thread',NumThreads, ...
        'orient','curvefit');
end
GNDSGE_PP=myinterp(GNDSGE_PP);

GNDSGE_NPROB = num_samples;

SimuRslt.shock = ones(num_samples,num_periods+1);
SimuRslt.K=zeros(num_samples,num_periods);
SimuRslt.s=zeros(num_samples,num_periods);
SimuRslt.K=zeros(num_samples,num_periods);
SimuRslt.w=zeros(num_samples,num_periods);


SimuRslt.K(:,1)=Kss;
SimuRslt.shock(:,1)=1;


if nargin>1 && isfield(GNDSGE_OPTIONS,'init')
    SimuRslt.K(:,1:size(GNDSGE_OPTIONS.init.K,2))=GNDSGE_OPTIONS.init.K;
SimuRslt.shock(:,1:size(GNDSGE_OPTIONS.init.shock,2))=GNDSGE_OPTIONS.init.shock;

end

if any(SimuRslt.shock(:,1)>shock_num)
    error('initial shock exceeds shock_num');
end

GNDSGE_LB = zeros(2,GNDSGE_NPROB);
GNDSGE_UB = 100*ones(2,GNDSGE_NPROB);
GNDSGE_LB(1:1,:)=0;
GNDSGE_UB(1:1,:)=1;
GNDSGE_LB(2:2,:)=0;



% Use the largest bound in IterSol
if UseAdaptiveBound==1
GNDSGE_LB = repmat(min(IterRslt.GNDSGE_PROB.GNDSGE_LB,[],2),[1,GNDSGE_NPROB]);
GNDSGE_UB = repmat(max(IterRslt.GNDSGE_PROB.GNDSGE_UB,[],2),[1,GNDSGE_NPROB]);
end

GNDSGE_SKIP = zeros(1,GNDSGE_NPROB);
GNDSGE_EQVAL = 1e20*ones(2,GNDSGE_NPROB);
GNDSGE_F = 1e20*ones(1,GNDSGE_NPROB);
GNDSGE_SOL = zeros(2,GNDSGE_NPROB);
GNDSGE_AUX = zeros(1,GNDSGE_NPROB);

%% Generate random number
SimuRslt.shock(:,GEN_SHOCK_START_PERIOD:end) = gen_discrete_markov_rn(shock_trans,num_samples,length(GEN_SHOCK_START_PERIOD:num_periods+1),...
    SimuRslt.shock(:,GEN_SHOCK_START_PERIOD));

MEX_TASK_NAME = MEX_TASK_INF_HORIZON;
GNDSGE_SHOCK_VAR_INDEX_BASE = ([0:num_samples-1]')*shock_num;

tic;
for GNDSGE_t=1:num_periods
    % Reuse the init inbound and tensor code
    % Map grid to current variable
    shock = SimuRslt.shock(:,GNDSGE_t);
    K=SimuRslt.K(:,GNDSGE_t);

    
    GNDSGE_data0 = repmat([shock_num;beta(:);sigma(:);alpha(:);delta(:);phi(:);shock_trans(:);z_agg(:);z_ind(:); ],1,GNDSGE_NPROB);
    
    
    
    % Use interpolation as initial values
%     %{
    if GNDSGE_t>0
        if shock_num>1
            GNDSGE_SOL = myinterp_mex(int32(NumThreads),GNDSGE_PP.breaks,GNDSGE_PP.coefs,...
                int32(GNDSGE_PP.pieces),int32(GNDSGE_PP.order),int32(GNDSGE_PP.dim),'not-a-knot',[SimuRslt.shock(:,GNDSGE_t)';SimuRslt.K(:,GNDSGE_t)'],[],[],[]);
        else
            GNDSGE_SOL = myinterp_mex(int32(NumThreads),GNDSGE_PP.breaks,GNDSGE_PP.coefs,...
                int32(GNDSGE_PP.pieces),int32(GNDSGE_PP.order),int32(GNDSGE_PP.dim),'not-a-knot',[SimuRslt.K(:,GNDSGE_t)'],[],[],[]);
        end
    end
    %}
    
    if UseAdaptiveBoundInSol==1
        % Tentatively adjust the bound
        GNDSGE_LB_OLD = GNDSGE_LB;
        GNDSGE_UB_OLD = GNDSGE_UB;
        
        
        
        % Hitting lower bound
        GNDSGE_SOL_hitting_lower_bound = abs(GNDSGE_SOL - GNDSGE_LB_OLD) < 1e-8;
        GNDSGE_SOL_hitting_upper_bound = abs(GNDSGE_SOL - GNDSGE_UB_OLD) < 1e-8;
        
        % Adjust for those hitting lower bound or upper bound
        GNDSGE_LB(~GNDSGE_SOL_hitting_lower_bound) = GNDSGE_LB_OLD(~GNDSGE_SOL_hitting_lower_bound);
        GNDSGE_UB(~GNDSGE_SOL_hitting_upper_bound) = GNDSGE_UB_OLD(~GNDSGE_SOL_hitting_upper_bound);
    end

    % Construct tensor variable
    
    
    % Reconstruct data
    GNDSGE_DATA = [GNDSGE_data0;shock(:)';K(:)';   ];
    
    % Solve problems
    GNDSGE_SKIP = zeros(1,GNDSGE_NPROB);
    [GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_capital_planner_1hh_2z(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);
    while (max(isnan(GNDSGE_F)) || max(GNDSGE_F(:))>TolSol)
        GNDSGE_X0Rand = rand(size(GNDSGE_SOL)) .* (GNDSGE_UB-GNDSGE_LB) + GNDSGE_LB;
        NeedResolved = (GNDSGE_F>TolSol) | isnan(GNDSGE_F);
        GNDSGE_SOL(:,NeedResolved) = GNDSGE_X0Rand(:,NeedResolved);
        GNDSGE_SKIP(~NeedResolved) = 1;
        
        [GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_capital_planner_1hh_2z(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);
    
        if UseAdaptiveBoundInSol==1
            % Tentatively adjust the bound
            GNDSGE_LB_OLD = GNDSGE_LB;
            GNDSGE_UB_OLD = GNDSGE_UB;
            
            
            
            % Hitting lower bound
            GNDSGE_SOL_hitting_lower_bound = abs(GNDSGE_SOL - GNDSGE_LB_OLD) < 1e-8;
            GNDSGE_SOL_hitting_upper_bound = abs(GNDSGE_SOL - GNDSGE_UB_OLD) < 1e-8;
            
            % Adjust for those hitting lower bound or upper bound
            GNDSGE_LB(~GNDSGE_SOL_hitting_lower_bound) = GNDSGE_LB_OLD(~GNDSGE_SOL_hitting_lower_bound);
            GNDSGE_UB(~GNDSGE_SOL_hitting_upper_bound) = GNDSGE_UB_OLD(~GNDSGE_SOL_hitting_upper_bound);
        end
    end
    
    s=GNDSGE_SOL(1:1,:);
K_next=GNDSGE_SOL(2:2,:);

    
    w=GNDSGE_AUX(1,:);

    
    GNDSGE_SHOCK_VAR_LINEAR_INDEX = SimuRslt.shock(:,GNDSGE_t+1) + GNDSGE_SHOCK_VAR_INDEX_BASE;
    SimuRslt.s(:,GNDSGE_t)=s;
SimuRslt.K(:,GNDSGE_t)=K;
SimuRslt.w(:,GNDSGE_t)=w;
SimuRslt.K(:,GNDSGE_t+1) = K_next(:);



    
    
    
    if mod(GNDSGE_t,SimuPrintFreq)==0
        fprintf('Periods: %d\n', GNDSGE_t);
        SimuRsltNames = fieldnames(SimuRslt);
        for GNDSGE_field = 1:length(SimuRsltNames)
            fprintf('%8s', SimuRsltNames{GNDSGE_field});
        end
        fprintf('\n');
        for GNDSGE_field = 1:length(SimuRsltNames)
            fprintf('%8.4g', SimuRslt.(SimuRsltNames{GNDSGE_field})(1,GNDSGE_t));
        end
        fprintf('\n');
        toc;
        tic;
    end
    
    if mod(GNDSGE_t,SimuSaveFreq)==0
        save(['SimuRslt_capital_planner_1hh_2z_' num2str(GNDSGE_t) '.mat'], 'SimuRslt');
    end
end
end