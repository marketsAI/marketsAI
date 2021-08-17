function [IterRslt,IterFlag] = iter_bianchi2011(GNDSGE_OPTIONS)
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

%% Iter code starts here
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
USE_ASG=1;
USE_SPLINE=0;
AsgMaxLevel = 10;
AsgThreshold = 1e-4;
r = 0.04;
sigma = 2;
eta = 1/0.83 - 1;
kappaN = 0.32;
kappaT = 0.32;
omega = 0.31;
beta = 0.91;
bPts = 101;
bMin=-0.5;
bMax=0.0;
b=linspace(bMin,bMax,bPts);
yPts = 4;
shock_num=16;
yTEpsilonVar = 0.00219;
yNEpsilonVar = 0.00167;
rhoYT = 0.901;
rhoYN = 0.225;
[yTTrans,yT] = markovappr(rhoYT,yTEpsilonVar^0.5,1,yPts);
[yNTrans,yN] = markovappr(rhoYN,yNEpsilonVar^0.5,1,yPts);
shock_trans = kron(yNTrans,yTTrans);
[yT,yN] = ndgrid(yT,yN);
yT = exp(yT(:)');
yN = exp(yN(:)');


if nargin>=1
    v2struct(GNDSGE_OPTIONS)
end
  
assert(exist('shock_num','var')==1);
assert(length(yT)==16);
assert(length(yN)==16);
assert(size(shock_trans,1)==16);
assert(size(shock_trans,2)==16);


%% Solve the last period problem
if ~SkipModelInit
    MEX_TASK_NAME = MEX_TASK_INIT;
    
    %% Construct interpolation
    GNDSGE_ASG_INTERP = asg({b},1,shock_num);
    
    while GNDSGE_ASG_INTERP.get_current_level < AsgMaxLevel
        if GNDSGE_ASG_INTERP.get_current_level<AsgMinLevel
            [GNDSGE_evalArrayIdx,GNDSGE_evalGrids,GNDSGE_evalGridsLength] = GNDSGE_ASG_INTERP.get_eval_grids(0.0);
        else
            [GNDSGE_evalArrayIdx,GNDSGE_evalGrids,GNDSGE_evalGridsLength] = GNDSGE_ASG_INTERP.get_eval_grids(AsgThreshold);
        end
        if isempty(GNDSGE_evalGrids)
            break;
        end
        
        GNDSGE_NPROB = size(GNDSGE_evalGrids,2);
        GNDSGE_SIZE = [1,GNDSGE_NPROB];
        
        % Solve the problem
        GNDSGE_LB = zeros(1,GNDSGE_NPROB);
GNDSGE_UB = 100*ones(1,GNDSGE_NPROB);
GNDSGE_LB(1,:)=-1.0;
GNDSGE_UB(1,:)=1.0;

        
        GNDSGE_X0 = (GNDSGE_LB+GNDSGE_UB)/2;

        
        GNDSGE_EQVAL = 1e20*ones(1,GNDSGE_NPROB);
        GNDSGE_F = 1e20*ones(1,GNDSGE_NPROB);
        GNDSGE_SOL = zeros(1,GNDSGE_NPROB);
        GNDSGE_SOL(:) = GNDSGE_X0;
        GNDSGE_AUX = zeros(2,GNDSGE_NPROB);
        GNDSGE_SKIP = zeros(1,GNDSGE_NPROB);
        GNDSGE_DATA = zeros(298,GNDSGE_NPROB);
        
        GNDSGE_DATA(:) = [repmat([shock_num;r(:);sigma(:);eta(:);kappaN(:);kappaT(:);omega(:);beta(:);shock_trans(:);yT(:);yN(:); ],1,GNDSGE_NPROB);GNDSGE_evalArrayIdx;GNDSGE_evalGrids];
        
        GNDSGE_SKIP(:) = 0;
[GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_bianchi2011(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);

% Randomzie for nonconvert point
GNDSGE_MinorIter = 0;
while ((max(isnan(GNDSGE_F)) || max(GNDSGE_F(:))>TolSol) && GNDSGE_MinorIter<MaxMinorIter)
    % Use randomize as initial guess
    GNDSGE_X0Rand = rand(size(GNDSGE_SOL)) .* (GNDSGE_UB-GNDSGE_LB) + GNDSGE_LB;
    NeedResolved = (GNDSGE_F>TolSol) | isnan(GNDSGE_F);
    GNDSGE_SOL(:,NeedResolved) = GNDSGE_X0Rand(:,NeedResolved);
    GNDSGE_SKIP(:) = 0;
    GNDSGE_SKIP(~NeedResolved) = 1;
    
    [GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_bianchi2011(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);
    
    GNDSGE_MinorIter = GNDSGE_MinorIter+1;
end
        
        dummy=GNDSGE_SOL(1:1,:);

        
        c=GNDSGE_AUX(1,:);
lambda=GNDSGE_AUX(2,:);

        
        % Map variables to initial interp
        lambda_interp=zeros(GNDSGE_SIZE);
lambda_interp(:)=lambda;

        GNDSGE_evalRslts = [lambda_interp; ];
        GNDSGE_ASG_INTERP.push_eval_results(GNDSGE_evalRslts);
    end
end

%% Solve the infinite horizon problem
MEX_TASK_NAME = MEX_TASK_INF_HORIZON;

GNDSGE_SOL_ASG_INTERP = asg({b},4,shock_num);

GNDSGE_Metric = 1;
GNDSGE_Iter = 0;

if nargin>=1 && isfield(GNDSGE_OPTIONS,'WarmUp')
    if isfield(GNDSGE_OPTIONS.WarmUp,'asg_interp_struct')
        GNDSGE_ASG_INTERP = asg.construct_from_struct(GNDSGE_OPTIONS.WarmUp.asg_interp_struct);
    end
    if isfield(GNDSGE_OPTIONS.WarmUp,'Iter')
        GNDSGE_Iter = GNDSGE_OPTIONS.WarmUp.Iter;
    end
    if isfield(GNDSGE_OPTIONS.WarmUp,'sol_asg_interp_struct')
        GNDSGE_SOL_ASG_INTERP = asg.construct_from_struct(GNDSGE_OPTIONS.WarmUp.sol_asg_interp_struct);
    end
end

stopFlag = false;
tic;
while(~stopFlag)
    GNDSGE_Iter = GNDSGE_Iter+1;
    
    
    
    % Construction interpolation
    GNDSGE_ASG_HANDLE = GNDSGE_ASG_INTERP.objectHandle;
    GNDSGE_ASG_INTERP_NEW = asg({b},1,shock_num);
    GNDSGE_SOL_ASG_INTERP_NEW = asg({b},4,shock_num);
    GNDSGE_ASG_STORE_evalArrayIdx = cell(0);
    GNDSGE_ASG_STORE_evalGridsUnscaled = cell(0);
    GNDSGE_ASG_STORE_output = cell(0);
    
    while GNDSGE_ASG_INTERP_NEW.get_current_level < AsgMaxLevel
        if GNDSGE_ASG_FIX_GRID==1
            [GNDSGE_TEMP_grids, GNDSGE_TEMP_surplus, GNDSGE_TEMP_levels, GNDSGE_TEMP_unscaledGrids] = GNDSGE_ASG_INTERP.get_grids_info_at_level(GNDSGE_ASG_INTERP_NEW.get_current_level+1);
            GNDSGE_evalArrayIdx = [];
            for GNDSGE_I_ARRAY=1:GNDSGE_OPTIONS.WarmUp.asg_interp_struct.numArray
                GNDSGE_evalArrayIdx = [GNDSGE_evalArrayIdx,GNDSGE_I_ARRAY*ones(1,size(GNDSGE_TEMP_grids{GNDSGE_I_ARRAY},2))];
            end
            GNDSGE_evalGrids = cat(2,GNDSGE_TEMP_grids{:});
            GNDSGE_evalGridsUnscaled = cat(2,GNDSGE_TEMP_unscaledGrids{:});
        else
            if GNDSGE_ASG_INTERP_NEW.get_current_level<AsgMinLevel
                [GNDSGE_evalArrayIdx,GNDSGE_evalGrids,GNDSGE_evalGridsLength,GNDSGE_evalGridsUnscaled] = GNDSGE_ASG_INTERP_NEW.get_eval_grids(0.0);
            else
                [GNDSGE_evalArrayIdx,GNDSGE_evalGrids,GNDSGE_evalGridsLength,GNDSGE_evalGridsUnscaled] = GNDSGE_ASG_INTERP_NEW.get_eval_grids(AsgThreshold);
            end
        end
        if isempty(GNDSGE_evalGrids)
            break;
        end
        
        GNDSGE_NPROB = size(GNDSGE_evalGrids,2);
GNDSGE_SIZE = [1,GNDSGE_NPROB];

% Solve the problem
GNDSGE_LB = zeros(4,GNDSGE_NPROB);
GNDSGE_UB = 100*ones(4,GNDSGE_NPROB);
GNDSGE_LB(1:1,:)=0.0;
GNDSGE_UB(1:1,:)=10.0;
GNDSGE_LB(2:2,:)=0.0;
GNDSGE_UB(2:2,:)=1.0;
GNDSGE_LB(3:3,:)=0.0;
GNDSGE_UB(3:3,:)=10.0;
GNDSGE_LB(4:4,:)=0.0;
GNDSGE_UB(4:4,:)=10.0;


% Interp the warmup using last period solution
GNDSGE_SOL0 = (GNDSGE_LB + GNDSGE_UB)/2;
GNDSGE_SOL = GNDSGE_SOL0;
if GNDSGE_SOL_ASG_INTERP.get_current_level>=0
    % Interp to get the warmup
    GNDSGE_SOL0 = GNDSGE_SOL_ASG_INTERP.eval_vec(GNDSGE_evalArrayIdx,GNDSGE_evalGrids);
    GNDSGE_SOL = GNDSGE_SOL0;
    if UseAdaptiveBound==1
        
    end
end

GNDSGE_EQVAL = 1e20*ones(4,GNDSGE_NPROB);
GNDSGE_F = 1e20*ones(1,GNDSGE_NPROB);
GNDSGE_AUX = zeros(3,GNDSGE_NPROB);
GNDSGE_SKIP = zeros(1,GNDSGE_NPROB);
GNDSGE_DATA = zeros(298,GNDSGE_NPROB);

GNDSGE_DATA(:) = [repmat([shock_num;r(:);sigma(:);eta(:);kappaN(:);kappaT(:);omega(:);beta(:);shock_trans(:);yT(:);yN(:); ],1,GNDSGE_NPROB);GNDSGE_evalArrayIdx;GNDSGE_evalGrids];

GNDSGE_SKIP(:) = 0;
[GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_bianchi2011(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);

% Use current asg as warmup for non-convergent point
if GNDSGE_ASG_INTERP_NEW.get_current_level>=0
    NeedResolved = (GNDSGE_F>TolSol) | isnan(GNDSGE_F);
    GNDSGE_SOL0 = GNDSGE_SOL_ASG_INTERP_NEW.eval_vec(GNDSGE_evalArrayIdx,GNDSGE_evalGrids);
    GNDSGE_SOL(:,NeedResolved) = GNDSGE_SOL0(:,NeedResolved);
    
    GNDSGE_SKIP(:) = 0;
    GNDSGE_SKIP(~NeedResolved) = 1;
    
    if UseAdaptiveBound==1
        
    end
    [GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_bianchi2011(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);
end

% Restore bound using last-iteration GNDSGE_SOL
if GNDSGE_SOL_ASG_INTERP.get_current_level>=0
    NeedResolved = (GNDSGE_F>TolSol) | isnan(GNDSGE_F);
    
    % Interp to get the warmup
    GNDSGE_SOL0 = GNDSGE_SOL_ASG_INTERP.eval_vec(GNDSGE_evalArrayIdx,GNDSGE_evalGrids);
    GNDSGE_SOL(:,NeedResolved) = GNDSGE_SOL0(:,NeedResolved);
    
    if UseAdaptiveBound==1
        
    end
end

% Randomzie for nonconvert point
GNDSGE_MinorIter = 0;
while ((max(isnan(GNDSGE_F)) || max(GNDSGE_F(:))>TolSol) && GNDSGE_MinorIter<MaxMinorIter)
    % Use randomize as initial guess
    GNDSGE_X0Rand = rand(size(GNDSGE_SOL)) .* (GNDSGE_UB-GNDSGE_LB) + GNDSGE_LB;
    NeedResolved = (GNDSGE_F>TolSol) | isnan(GNDSGE_F);
    GNDSGE_SOL(:,NeedResolved) = GNDSGE_X0Rand(:,NeedResolved);
    GNDSGE_SKIP(:) = 0;
    GNDSGE_SKIP(~NeedResolved) = 1;
    
    [GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_bianchi2011(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);
    
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
        
        GNDSGE_MinorIter = GNDSGE_MinorIter+1;
    end
end



nbNext=GNDSGE_SOL(1:1,:);
mu=GNDSGE_SOL(2:2,:);
cT=GNDSGE_SOL(3:3,:);
pN=GNDSGE_SOL(4:4,:);


c=GNDSGE_AUX(1,:);
lambda=GNDSGE_AUX(2,:);
bNext=GNDSGE_AUX(3,:);

        
        lambda_interp = lambda;

        
        GNDSGE_evalRslts = [lambda_interp; ];
        
        GNDSGE_SOL_ASG_INTERP_NEW.push_eval_results_at_grids(GNDSGE_evalArrayIdx, GNDSGE_evalGridsUnscaled, GNDSGE_SOL, GNDSGE_SOL_ASG_INTERP_NEW.get_current_level);
        if GNDSGE_ASG_FIX_GRID==1
            GNDSGE_ASG_INTERP_NEW.push_eval_results_at_grids(GNDSGE_evalArrayIdx, GNDSGE_evalGridsUnscaled, GNDSGE_evalRslts, GNDSGE_ASG_INTERP_NEW.get_current_level);
        else
            GNDSGE_ASG_INTERP_NEW.push_eval_results(GNDSGE_evalRslts);
        end
        GNDSGE_ASG_STORE_evalArrayIdx = [GNDSGE_ASG_STORE_evalArrayIdx,GNDSGE_evalArrayIdx];
        GNDSGE_ASG_STORE_evalGridsUnscaled = [GNDSGE_ASG_STORE_evalGridsUnscaled,GNDSGE_evalGridsUnscaled];
        GNDSGE_ASG_STORE_output = [GNDSGE_ASG_STORE_output,[bNext;pN;c; ]];
    end
    
    
    % Compute Metric
    [GNDSGE_Metric,GNDSGE_MetricVec] = asg.compute_inf_metric(GNDSGE_ASG_INTERP_NEW, GNDSGE_ASG_INTERP);
    
    % Update
    GNDSGE_ASG_INTERP = GNDSGE_ASG_INTERP_NEW;
    GNDSGE_SOL_ASG_INTERP = GNDSGE_SOL_ASG_INTERP_NEW;
    
    
    
    stopFlag = GNDSGE_Metric<TolEq || GNDSGE_Iter>=MaxIter;
    
    if ( mod(GNDSGE_Iter,PrintFreq)==0 || stopFlag == true )
      fprintf(['Iter:%d, Metric:%g, maxF:%g\n'],GNDSGE_Iter,GNDSGE_Metric,max(GNDSGE_F));
      toc;
      tic;
    end
    
    if ( mod(GNDSGE_Iter,SaveFreq)==0 || stopFlag == true )
        % Construct output
        % GNDSGE_ASG_OUTPUT = asg({b},3,shock_num);
        % OUTPUT_xxx_CONSTRUCT_CODE
        
        % Solve the problem and get output variables
        GNDSGE_ASG_HANDLE = GNDSGE_ASG_INTERP.objectHandle;
        GNDSGE_ASG_INTERP_NEW = asg({b},3,shock_num);
        
        while GNDSGE_ASG_INTERP_NEW.get_current_level < AsgOutputMaxLevel
    % Fix grids
    [GNDSGE_TEMP_grids, GNDSGE_TEMP_surplus, GNDSGE_TEMP_levels, GNDSGE_TEMP_unscaledGrids] = GNDSGE_ASG_INTERP.get_grids_info_at_level(GNDSGE_ASG_INTERP_NEW.get_current_level+1);
    GNDSGE_evalArrayIdx = [];
    for GNDSGE_I_ARRAY=1:shock_num
        GNDSGE_evalArrayIdx = [GNDSGE_evalArrayIdx,GNDSGE_I_ARRAY*ones(1,size(GNDSGE_TEMP_grids{GNDSGE_I_ARRAY},2))];
    end
    GNDSGE_evalGrids = cat(2,GNDSGE_TEMP_grids{:});
    GNDSGE_evalGridsUnscaled = cat(2,GNDSGE_TEMP_unscaledGrids{:});
    if isempty(GNDSGE_evalGrids)
        break;
    end
    
    GNDSGE_NPROB = size(GNDSGE_evalGrids,2);
GNDSGE_SIZE = [1,GNDSGE_NPROB];

% Solve the problem
GNDSGE_LB = zeros(4,GNDSGE_NPROB);
GNDSGE_UB = 100*ones(4,GNDSGE_NPROB);
GNDSGE_LB(1:1,:)=0.0;
GNDSGE_UB(1:1,:)=10.0;
GNDSGE_LB(2:2,:)=0.0;
GNDSGE_UB(2:2,:)=1.0;
GNDSGE_LB(3:3,:)=0.0;
GNDSGE_UB(3:3,:)=10.0;
GNDSGE_LB(4:4,:)=0.0;
GNDSGE_UB(4:4,:)=10.0;


% Interp the warmup using last period solution
GNDSGE_SOL0 = (GNDSGE_LB + GNDSGE_UB)/2;
GNDSGE_SOL = GNDSGE_SOL0;
if GNDSGE_SOL_ASG_INTERP.get_current_level>=0
    % Interp to get the warmup
    GNDSGE_SOL0 = GNDSGE_SOL_ASG_INTERP.eval_vec(GNDSGE_evalArrayIdx,GNDSGE_evalGrids);
    GNDSGE_SOL = GNDSGE_SOL0;
    if UseAdaptiveBound==1
        
    end
end

GNDSGE_EQVAL = 1e20*ones(4,GNDSGE_NPROB);
GNDSGE_F = 1e20*ones(1,GNDSGE_NPROB);
GNDSGE_AUX = zeros(3,GNDSGE_NPROB);
GNDSGE_SKIP = zeros(1,GNDSGE_NPROB);
GNDSGE_DATA = zeros(298,GNDSGE_NPROB);

GNDSGE_DATA(:) = [repmat([shock_num;r(:);sigma(:);eta(:);kappaN(:);kappaT(:);omega(:);beta(:);shock_trans(:);yT(:);yN(:); ],1,GNDSGE_NPROB);GNDSGE_evalArrayIdx;GNDSGE_evalGrids];

GNDSGE_SKIP(:) = 0;
[GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_bianchi2011(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);

% Use current asg as warmup for non-convergent point
if GNDSGE_ASG_INTERP_NEW.get_current_level>=0
    NeedResolved = (GNDSGE_F>TolSol) | isnan(GNDSGE_F);
    GNDSGE_SOL0 = GNDSGE_SOL_ASG_INTERP_NEW.eval_vec(GNDSGE_evalArrayIdx,GNDSGE_evalGrids);
    GNDSGE_SOL(:,NeedResolved) = GNDSGE_SOL0(:,NeedResolved);
    
    GNDSGE_SKIP(:) = 0;
    GNDSGE_SKIP(~NeedResolved) = 1;
    
    if UseAdaptiveBound==1
        
    end
    [GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_bianchi2011(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);
end

% Restore bound using last-iteration GNDSGE_SOL
if GNDSGE_SOL_ASG_INTERP.get_current_level>=0
    NeedResolved = (GNDSGE_F>TolSol) | isnan(GNDSGE_F);
    
    % Interp to get the warmup
    GNDSGE_SOL0 = GNDSGE_SOL_ASG_INTERP.eval_vec(GNDSGE_evalArrayIdx,GNDSGE_evalGrids);
    GNDSGE_SOL(:,NeedResolved) = GNDSGE_SOL0(:,NeedResolved);
    
    if UseAdaptiveBound==1
        
    end
end

% Randomzie for nonconvert point
GNDSGE_MinorIter = 0;
while ((max(isnan(GNDSGE_F)) || max(GNDSGE_F(:))>TolSol) && GNDSGE_MinorIter<MaxMinorIter)
    % Use randomize as initial guess
    GNDSGE_X0Rand = rand(size(GNDSGE_SOL)) .* (GNDSGE_UB-GNDSGE_LB) + GNDSGE_LB;
    NeedResolved = (GNDSGE_F>TolSol) | isnan(GNDSGE_F);
    GNDSGE_SOL(:,NeedResolved) = GNDSGE_X0Rand(:,NeedResolved);
    GNDSGE_SKIP(:) = 0;
    GNDSGE_SKIP(~NeedResolved) = 1;
    
    [GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_bianchi2011(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);
    
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
        
        GNDSGE_MinorIter = GNDSGE_MinorIter+1;
    end
end



nbNext=GNDSGE_SOL(1:1,:);
mu=GNDSGE_SOL(2:2,:);
cT=GNDSGE_SOL(3:3,:);
pN=GNDSGE_SOL(4:4,:);


c=GNDSGE_AUX(1,:);
lambda=GNDSGE_AUX(2,:);
bNext=GNDSGE_AUX(3,:);

    
    GNDSGE_evalRslts = [bNext;pN;c; ];
    % Fix grids
    GNDSGE_ASG_INTERP_NEW.push_eval_results_at_grids(GNDSGE_evalArrayIdx, GNDSGE_evalGridsUnscaled, GNDSGE_evalRslts, GNDSGE_ASG_INTERP_NEW.get_current_level);
end

output_var_index=struct();
output_var_index.bNext=1:1;
output_var_index.pN=2:2;
output_var_index.c=3:3;

IterRslt.output_var_index = output_var_index;
IterRslt.asg_output_struct = GNDSGE_ASG_INTERP_NEW.convert_to_struct();


        IterRslt.Metric = GNDSGE_Metric;
        IterRslt.MetricVec = GNDSGE_MetricVec;
        IterRslt.Iter = GNDSGE_Iter;
        IterRslt.shock_num = shock_num;
        IterRslt.shock_trans = shock_trans;
        IterRslt.var_shock = v2struct(yT,yN);
        IterRslt.var_state = v2struct(b);
        IterRslt.params = v2struct(r,sigma,eta,kappaN,kappaT,omega,beta,GNDSGE_EMPTY);
        IterRslt.asg_interp_struct = GNDSGE_ASG_INTERP.convert_to_struct();
        IterRslt.sol_asg_interp_struct = GNDSGE_SOL_ASG_INTERP.convert_to_struct();
        IterRslt.var_others = v2struct(GNDSGE_EMPTY);

        if IterSaveAll
            save(['IterRslt_bianchi2011_' num2str(GNDSGE_Iter) '.mat']);
        else
            save(['IterRslt_bianchi2011_' num2str(GNDSGE_Iter) '.mat'],'IterRslt');
        end
    end
end    

%% Return the success flag
IterFlag = 0;
end
