function [IterRslt,IterFlag] = iter_capital_planner_3hh(GNDSGE_OPTIONS)
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
beta  = 0.98;
sigma = 1.0;
alpha = 0.3;
delta = 0.04;
phi = 0.5;
shock_num = 2^4;
z_agg_vals = [0.8, 1.2];
z_ind_vals = [0.9, 1.1];
[z_ind_1, z_ind_2, z_ind_3, z_agg] = ndgrid(z_ind_vals, z_ind_vals, z_ind_vals, z_agg_vals);
z_agg = z_agg(:)';
z_ind_1 = z_ind_1(:)';
z_ind_2 = z_ind_2(:)';
z_ind_3 = z_ind_3(:)';
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
shock_trans = kron(kron(kron(shock_trans_agg, shock_trans_ind),shock_trans_ind), shock_trans_ind);
Kss  = (alpha * beta / (phi*delta*3*(1-beta*(1-delta))))^(1/(2-alpha));
KPts = 7;
KMin = Kss*0.5;
KMax = Kss*1.5;
K_1    = linspace(KMin,KMax,KPts);
K_2    = linspace(KMin,KMax,KPts);
K_3    = linspace(KMin,KMax,KPts);


if nargin>=1
    v2struct(GNDSGE_OPTIONS)
end
  
assert(exist('shock_num','var')==1);
assert(length(z_agg)==16);
assert(length(z_ind_1)==16);
assert(length(z_ind_2)==16);
assert(length(z_ind_3)==16);
assert(size(shock_trans,1)==16);
assert(size(shock_trans,2)==16);


%% Solve the last period problem


%% Solve the infinite horizon problem
[GNDSGE_TENSOR_shockIdx,GNDSGE_TENSOR_K_1,GNDSGE_TENSOR_K_2,GNDSGE_TENSOR_K_3]=ndgrid(1:shock_num,K_1,K_2,K_3);
GNDSGE_TENSOR_z_agg=ndgrid(z_agg,K_1,K_2,K_3);
GNDSGE_TENSOR_z_ind_1=ndgrid(z_ind_1,K_1,K_2,K_3);
GNDSGE_TENSOR_z_ind_2=ndgrid(z_ind_2,K_1,K_2,K_3);
GNDSGE_TENSOR_z_ind_3=ndgrid(z_ind_3,K_1,K_2,K_3);
GNDSGE_NPROB=numel(GNDSGE_TENSOR_shockIdx);




GNDSGE_SIZE = size(GNDSGE_TENSOR_shockIdx);
GNDSGE_SIZE_STATE = num2cell(GNDSGE_SIZE(2:end));

GNDSGE_LB = zeros(6,GNDSGE_NPROB);
GNDSGE_UB = 100*ones(6,GNDSGE_NPROB);
GNDSGE_LB(1:1,:)=0;
GNDSGE_UB(1:1,:)=1;
GNDSGE_LB(2:2,:)=0;
GNDSGE_UB(2:2,:)=1;
GNDSGE_LB(3:3,:)=0;
GNDSGE_UB(3:3,:)=1;
GNDSGE_LB(4:4,:)=0;
GNDSGE_UB(4:4,:)=((2/phi)*GNDSGE_TENSOR_z_agg(:)'.*GNDSGE_TENSOR_z_ind_1(:)'.*GNDSGE_TENSOR_K_1(:)'.^alpha).^(1/2)+(1-delta)*GNDSGE_TENSOR_K_1(:)';
GNDSGE_LB(5:5,:)=0;
GNDSGE_UB(5:5,:)=((2/phi)*GNDSGE_TENSOR_z_agg(:)'.*GNDSGE_TENSOR_z_ind_2(:)'.*GNDSGE_TENSOR_K_2(:)'.^alpha).^(1/2)+(1-delta)*GNDSGE_TENSOR_K_2(:)';
GNDSGE_LB(6:6,:)=0;
GNDSGE_UB(6:6,:)=((2/phi)*GNDSGE_TENSOR_z_agg(:)'.*GNDSGE_TENSOR_z_ind_3(:)'.*GNDSGE_TENSOR_K_3(:)'.^alpha).^(1/2)+(1-delta)*GNDSGE_TENSOR_K_3(:)';


GNDSGE_EQVAL = 1e20*ones(6,GNDSGE_NPROB);
GNDSGE_F = 1e20*ones(1,GNDSGE_NPROB);
GNDSGE_SOL = zeros(6,GNDSGE_NPROB);
GNDSGE_X0 = rand(size(GNDSGE_SOL)) .* (GNDSGE_UB-GNDSGE_LB) + GNDSGE_LB;
GNDSGE_SOL(:) = GNDSGE_X0;
GNDSGE_AUX = zeros(0,GNDSGE_NPROB);
GNDSGE_SKIP = zeros(1,GNDSGE_NPROB);
GNDSGE_DATA = zeros(330,GNDSGE_NPROB);

if ~( nargin>=1 && isfield(GNDSGE_OPTIONS,'WarmUp') && isfield(GNDSGE_OPTIONS.WarmUp,'var_interp') )
    s_1_interp=zeros(GNDSGE_SIZE);
s_2_interp=zeros(GNDSGE_SIZE);
s_3_interp=zeros(GNDSGE_SIZE);
s_1_interp(:)=0.15;
s_2_interp(:)=0.15;
s_3_interp(:)=0.15;

    
    GNDSGE_INTERP_ORDER = INTERP_ORDER*ones(1,length(GNDSGE_SIZE_STATE));
GNDSGE_EXTRAP_ORDER = EXTRAP_ORDER*ones(1,length(GNDSGE_SIZE_STATE));

GNDSGE_PP_s_1_interp=struct('form','pp','breaks',{{K_1,K_2,K_3}},'Values',reshape(s_1_interp,[],GNDSGE_SIZE_STATE{:}),'coefs',[],'order',GNDSGE_INTERP_ORDER,'Method',[],'ExtrapolationOrder',GNDSGE_EXTRAP_ORDER,'thread',NumThreads,'orient','curvefit');
GNDSGE_PP_s_1_interp=myinterp(myinterp(GNDSGE_PP_s_1_interp));

GNDSGE_PP_s_2_interp=struct('form','pp','breaks',{{K_1,K_2,K_3}},'Values',reshape(s_2_interp,[],GNDSGE_SIZE_STATE{:}),'coefs',[],'order',GNDSGE_INTERP_ORDER,'Method',[],'ExtrapolationOrder',GNDSGE_EXTRAP_ORDER,'thread',NumThreads,'orient','curvefit');
GNDSGE_PP_s_2_interp=myinterp(myinterp(GNDSGE_PP_s_2_interp));

GNDSGE_PP_s_3_interp=struct('form','pp','breaks',{{K_1,K_2,K_3}},'Values',reshape(s_3_interp,[],GNDSGE_SIZE_STATE{:}),'coefs',[],'order',GNDSGE_INTERP_ORDER,'Method',[],'ExtrapolationOrder',GNDSGE_EXTRAP_ORDER,'thread',NumThreads,'orient','curvefit');
GNDSGE_PP_s_3_interp=myinterp(myinterp(GNDSGE_PP_s_3_interp));



% Construct the vectorized spline
if ~GNDSGE_USE_OLD_VEC
    GNDSGE_SPLINE_VEC = convert_to_interp_eval_array({GNDSGE_PP_s_1_interp,GNDSGE_PP_s_2_interp,GNDSGE_PP_s_3_interp});
end

end

GNDSGE_Metric = 1;
GNDSGE_Iter = 0;
IS_WARMUP_LOOP = 0;

if nargin>=1 && isfield(GNDSGE_OPTIONS,'WarmUp')
    if isfield(GNDSGE_OPTIONS.WarmUp,'var_interp')
        v2struct(GNDSGE_OPTIONS.WarmUp.var_interp);
        GNDSGE_TEMP = v2struct(K_1,K_2,K_3,GNDSGE_SIZE_STATE);
        GNDSGE_SIZE_STATE = num2cell(GNDSGE_OPTIONS.WarmUp.GNDSGE_PROB.GNDSGE_SIZE(2:end));
        v2struct(GNDSGE_OPTIONS.WarmUp.var_state);
        GNDSGE_INTERP_ORDER = INTERP_ORDER*ones(1,length(GNDSGE_SIZE_STATE));
GNDSGE_EXTRAP_ORDER = EXTRAP_ORDER*ones(1,length(GNDSGE_SIZE_STATE));

GNDSGE_PP_s_1_interp=struct('form','pp','breaks',{{K_1,K_2,K_3}},'Values',reshape(s_1_interp,[],GNDSGE_SIZE_STATE{:}),'coefs',[],'order',GNDSGE_INTERP_ORDER,'Method',[],'ExtrapolationOrder',GNDSGE_EXTRAP_ORDER,'thread',NumThreads,'orient','curvefit');
GNDSGE_PP_s_1_interp=myinterp(myinterp(GNDSGE_PP_s_1_interp));

GNDSGE_PP_s_2_interp=struct('form','pp','breaks',{{K_1,K_2,K_3}},'Values',reshape(s_2_interp,[],GNDSGE_SIZE_STATE{:}),'coefs',[],'order',GNDSGE_INTERP_ORDER,'Method',[],'ExtrapolationOrder',GNDSGE_EXTRAP_ORDER,'thread',NumThreads,'orient','curvefit');
GNDSGE_PP_s_2_interp=myinterp(myinterp(GNDSGE_PP_s_2_interp));

GNDSGE_PP_s_3_interp=struct('form','pp','breaks',{{K_1,K_2,K_3}},'Values',reshape(s_3_interp,[],GNDSGE_SIZE_STATE{:}),'coefs',[],'order',GNDSGE_INTERP_ORDER,'Method',[],'ExtrapolationOrder',GNDSGE_EXTRAP_ORDER,'thread',NumThreads,'orient','curvefit');
GNDSGE_PP_s_3_interp=myinterp(myinterp(GNDSGE_PP_s_3_interp));



% Construct the vectorized spline
if ~GNDSGE_USE_OLD_VEC
    GNDSGE_SPLINE_VEC = convert_to_interp_eval_array({GNDSGE_PP_s_1_interp,GNDSGE_PP_s_2_interp,GNDSGE_PP_s_3_interp});
end

        v2struct(GNDSGE_TEMP);
        IS_WARMUP_LOOP = 1;
    end
    if isfield(GNDSGE_OPTIONS.WarmUp,'Iter')
        GNDSGE_Iter = GNDSGE_OPTIONS.WarmUp.Iter;
    end
    if isfield(GNDSGE_OPTIONS.WarmUp,'GNDSGE_SOL')
        if ~isequal(size(GNDSGE_OPTIONS.WarmUp.GNDSGE_SOL,1),size(GNDSGE_SOL,1))
            error('Wrong size of GNDSGE_SOL in WarmUp');
        end
        GNDSGE_SOL = GNDSGE_OPTIONS.WarmUp.GNDSGE_SOL;
    end
    if REUSE_WARMUP_SOL==1 && isfield(GNDSGE_OPTIONS.WarmUp,'GNDSGE_PROB') && size(GNDSGE_OPTIONS.WarmUp.GNDSGE_PROB.GNDSGE_SOL,1)==size(GNDSGE_SOL,1)
        % Interpolate SOL, LB, and UB
        GNDSGE_TEMP = v2struct(K_1,K_2,K_3,GNDSGE_SIZE_STATE);
        GNDSGE_SIZE_STATE = num2cell(GNDSGE_OPTIONS.WarmUp.GNDSGE_PROB.GNDSGE_SIZE);
        v2struct(GNDSGE_OPTIONS.WarmUp.var_state);
        GNDSGE_SOL_interp=struct('form','MKL','breaks',{{[1:shock_num],K_1,K_2,K_3}},'Values',reshape(GNDSGE_OPTIONS.WarmUp.GNDSGE_PROB.GNDSGE_SOL,[],GNDSGE_SIZE_STATE{:}),'coefs',[],'order',[2,2*ones(1,length(GNDSGE_SIZE_STATE))],'Method',[],'ExtrapolationOrder',[2,2*ones(1,length(GNDSGE_SIZE_STATE))],'thread',NumThreads,'orient','curvefit');
        GNDSGE_SOL_interp=myinterp(GNDSGE_SOL_interp);
        GNDSGE_LB_interp=struct('form','MKL','breaks',{{[1:shock_num],K_1,K_2,K_3}},'Values',reshape(GNDSGE_OPTIONS.WarmUp.GNDSGE_PROB.GNDSGE_LB,[],GNDSGE_SIZE_STATE{:}),'coefs',[],'order',[2,2*ones(1,length(GNDSGE_SIZE_STATE))],'Method',[],'ExtrapolationOrder',[2,2*ones(1,length(GNDSGE_SIZE_STATE))],'thread',NumThreads,'orient','curvefit');
        GNDSGE_LB_interp=myinterp(GNDSGE_LB_interp);
        GNDSGE_UB_interp=struct('form','MKL','breaks',{{[1:shock_num],K_1,K_2,K_3}},'Values',reshape(GNDSGE_OPTIONS.WarmUp.GNDSGE_PROB.GNDSGE_UB,[],GNDSGE_SIZE_STATE{:}),'coefs',[],'order',[2,2*ones(1,length(GNDSGE_SIZE_STATE))],'Method',[],'ExtrapolationOrder',[2,2*ones(1,length(GNDSGE_SIZE_STATE))],'thread',NumThreads,'orient','curvefit');
        GNDSGE_UB_interp=myinterp(GNDSGE_UB_interp);
        
        v2struct(GNDSGE_TEMP);
        GNDSGE_SOL = reshape(myinterp(GNDSGE_SOL_interp,[GNDSGE_TENSOR_shockIdx(:)';GNDSGE_TENSOR_K_1(:)';GNDSGE_TENSOR_K_2(:)';GNDSGE_TENSOR_K_3(:)'; ]),size(GNDSGE_SOL));
        GNDSGE_LB_NEW = reshape(myinterp(GNDSGE_LB_interp,[GNDSGE_TENSOR_shockIdx(:)';GNDSGE_TENSOR_K_1(:)';GNDSGE_TENSOR_K_2(:)';GNDSGE_TENSOR_K_3(:)'; ]),size(GNDSGE_LB));
        GNDSGE_UB_NEW = reshape(myinterp(GNDSGE_UB_interp,[GNDSGE_TENSOR_shockIdx(:)';GNDSGE_TENSOR_K_1(:)';GNDSGE_TENSOR_K_2(:)';GNDSGE_TENSOR_K_3(:)'; ]),size(GNDSGE_UB));
        
        
    end
end

stopFlag = false;
tic;
while(~stopFlag)
    GNDSGE_Iter = GNDSGE_Iter+1;
    
    

    
GNDSGE_DATA(:) = [repmat([shock_num;beta(:);sigma(:);alpha(:);delta(:);phi(:);shock_trans(:);z_agg(:);z_ind_1(:);z_ind_2(:);z_ind_3(:); ],1,GNDSGE_NPROB);GNDSGE_TENSOR_shockIdx(:)';GNDSGE_TENSOR_K_1(:)';GNDSGE_TENSOR_K_2(:)';GNDSGE_TENSOR_K_3(:)';   ];

MEX_TASK_NAME = MEX_TASK_INF_HORIZON;
GNDSGE_F(:) = 1e20;
GNDSGE_SKIP(:) = 0;
if exist('GNDSGE_Iter','var')>0 && GNDSGE_Iter>1
    GNDSGE_USE_BROYDEN_NOW = GNDSGE_USE_BROYDEN;
end
[GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_capital_planner_3hh(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);
GNDSGE_USE_BROYDEN_NOW = 0;
% Randomzie for nonconvert point
GNDSGE_MinorIter = 0;
numNeedResolvedAfter = inf;
while ((max(isnan(GNDSGE_F)) || max(GNDSGE_F(:))>TolSol) && GNDSGE_MinorIter<MaxMinorIter)
    % Repeatedly use the nearest point as initial guess
    NeedResolved = (GNDSGE_F>TolSol) | isnan(GNDSGE_F);
    numNeedResolved = sum(NeedResolved);
    while numNeedResolvedAfter ~= numNeedResolved
        NeedResolved = (GNDSGE_F>TolSol) | isnan(GNDSGE_F);
        numNeedResolved = sum(NeedResolved);
        % Use the nearest point as initial guess
        for i_dim = 1:length(GNDSGE_SIZE)
            stride = prod(GNDSGE_SIZE(1:i_dim-1));
            
            NeedResolved = (GNDSGE_F>TolSol) | isnan(GNDSGE_F);
            GNDSGE_SKIP(:) = 1;
            for GNDSGE_i = 1:numel(GNDSGE_F)
                if NeedResolved(GNDSGE_i) && GNDSGE_i-stride>=1
                    GNDSGE_idx = GNDSGE_i-stride;
                    if ~NeedResolved(GNDSGE_idx)
                        GNDSGE_SOL(:,GNDSGE_i) = GNDSGE_SOL(:,GNDSGE_idx);
                        GNDSGE_SKIP(GNDSGE_i) = 0;
                    end
                end
            end
            [GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_capital_planner_3hh(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);
            
            NeedResolved = (GNDSGE_F>TolSol) | isnan(GNDSGE_F);
            GNDSGE_SKIP(:) = 1;
            for GNDSGE_i = 1:numel(GNDSGE_F)
                if NeedResolved(GNDSGE_i) && GNDSGE_i+stride<=numel(GNDSGE_F)
                    GNDSGE_idx = GNDSGE_i+stride;
                    if ~NeedResolved(GNDSGE_idx)
                        GNDSGE_SOL(:,GNDSGE_i) = GNDSGE_SOL(:,GNDSGE_idx);
                        GNDSGE_SKIP(GNDSGE_i) = 0;
                    end
                end
            end
            [GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_capital_planner_3hh(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);
        end
        
        NeedResolvedAfter = (GNDSGE_F>TolSol) | isnan(GNDSGE_F);
        numNeedResolvedAfter = sum(NeedResolvedAfter);
    end
    
    % Use randomize as initial guess
    GNDSGE_X0Rand = rand(size(GNDSGE_SOL)) .* (GNDSGE_UB-GNDSGE_LB) + GNDSGE_LB;
    NeedResolved = (GNDSGE_F>TolSol) | isnan(GNDSGE_F);
    GNDSGE_SOL(:,NeedResolved) = GNDSGE_X0Rand(:,NeedResolved);
    GNDSGE_SKIP(:) = 0;
    GNDSGE_SKIP(~NeedResolved) = 1;
    
    [GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_capital_planner_3hh(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);

    if UseAdaptiveBoundInSol==1 && exist('GNDSGE_Iter','var')>0
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
    
    GNDSGE_MinorIter = GNDSGE_MinorIter+1;
end

s_1=GNDSGE_SOL(1:1,:);
s_2=GNDSGE_SOL(2:2,:);
s_3=GNDSGE_SOL(3:3,:);
K_1_next=GNDSGE_SOL(4:4,:);
K_2_next=GNDSGE_SOL(5:5,:);
K_3_next=GNDSGE_SOL(6:6,:);



    
    
    
    s_1=reshape(s_1,shock_num,[]);
s_2=reshape(s_2,shock_num,[]);
s_3=reshape(s_3,shock_num,[]);
K_1_next=reshape(K_1_next,shock_num,[]);
K_2_next=reshape(K_2_next,shock_num,[]);
K_3_next=reshape(K_3_next,shock_num,[]);

    
    GNDSGE_NEW_s_1_interp = s_1;
GNDSGE_NEW_s_2_interp = s_2;
GNDSGE_NEW_s_3_interp = s_3;

    
    % Compute Metric
    if IS_WARMUP_LOOP==1
        GNDSGE_Metric = inf;
        IS_WARMUP_LOOP = 0;
    else
        GNDSGE_Metric = max([max(abs(GNDSGE_NEW_s_1_interp(:)-s_1_interp(:)))max(abs(GNDSGE_NEW_s_2_interp(:)-s_2_interp(:)))max(abs(GNDSGE_NEW_s_3_interp(:)-s_3_interp(:)))]);

    end
    
    % Update
    s_1_interp=GNDSGE_NEW_s_1_interp;
s_2_interp=GNDSGE_NEW_s_2_interp;
s_3_interp=GNDSGE_NEW_s_3_interp;

    
    
    
    
    
    stopFlag = (isempty(GNDSGE_Metric) || GNDSGE_Metric<TolEq) || GNDSGE_Iter>=MaxIter;
    
    GNDSGE_INTERP_ORDER = INTERP_ORDER*ones(1,length(GNDSGE_SIZE_STATE));
GNDSGE_EXTRAP_ORDER = EXTRAP_ORDER*ones(1,length(GNDSGE_SIZE_STATE));

GNDSGE_PP_s_1_interp=struct('form','pp','breaks',{{K_1,K_2,K_3}},'Values',reshape(s_1_interp,[],GNDSGE_SIZE_STATE{:}),'coefs',[],'order',GNDSGE_INTERP_ORDER,'Method',[],'ExtrapolationOrder',GNDSGE_EXTRAP_ORDER,'thread',NumThreads,'orient','curvefit');
GNDSGE_PP_s_1_interp=myinterp(myinterp(GNDSGE_PP_s_1_interp));

GNDSGE_PP_s_2_interp=struct('form','pp','breaks',{{K_1,K_2,K_3}},'Values',reshape(s_2_interp,[],GNDSGE_SIZE_STATE{:}),'coefs',[],'order',GNDSGE_INTERP_ORDER,'Method',[],'ExtrapolationOrder',GNDSGE_EXTRAP_ORDER,'thread',NumThreads,'orient','curvefit');
GNDSGE_PP_s_2_interp=myinterp(myinterp(GNDSGE_PP_s_2_interp));

GNDSGE_PP_s_3_interp=struct('form','pp','breaks',{{K_1,K_2,K_3}},'Values',reshape(s_3_interp,[],GNDSGE_SIZE_STATE{:}),'coefs',[],'order',GNDSGE_INTERP_ORDER,'Method',[],'ExtrapolationOrder',GNDSGE_EXTRAP_ORDER,'thread',NumThreads,'orient','curvefit');
GNDSGE_PP_s_3_interp=myinterp(myinterp(GNDSGE_PP_s_3_interp));



% Construct the vectorized spline
if ~GNDSGE_USE_OLD_VEC
    GNDSGE_SPLINE_VEC = convert_to_interp_eval_array({GNDSGE_PP_s_1_interp,GNDSGE_PP_s_2_interp,GNDSGE_PP_s_3_interp});
end

    
    if ( mod(GNDSGE_Iter,PrintFreq)==0 || stopFlag == true && ~NoPrint )
      fprintf(['Iter:%d, Metric:%g, maxF:%g\n'],GNDSGE_Iter,GNDSGE_Metric,max(GNDSGE_F));
      toc;
      tic;
    end
    
    if ( (mod(GNDSGE_Iter,SaveFreq)==0 || stopFlag == true) )
        s_1=reshape(s_1,GNDSGE_SIZE);
s_2=reshape(s_2,GNDSGE_SIZE);
s_3=reshape(s_3,GNDSGE_SIZE);
K_1_next=reshape(K_1_next,GNDSGE_SIZE);
K_2_next=reshape(K_2_next,GNDSGE_SIZE);
K_3_next=reshape(K_3_next,GNDSGE_SIZE);
s_1_interp=reshape(s_1_interp,GNDSGE_SIZE);
s_2_interp=reshape(s_2_interp,GNDSGE_SIZE);
s_3_interp=reshape(s_3_interp,GNDSGE_SIZE);

        
        % Construct output variables
outputVarStack = cat(1,reshape(s_1,1,[]),reshape(s_2,1,[]),reshape(s_3,1,[]));

% Permute variables from order (shock, var, states) to order (var,
% shock, states)
if shock_num>1
    outputVarStack = reshape(outputVarStack, [],shock_num,GNDSGE_SIZE_STATE{:});
    output_interp=struct('form','MKL','breaks',{{1:shock_num,K_1,K_2,K_3}},...
        'Values',outputVarStack,...
        'coefs',[],'order',[2 OutputInterpOrder*ones(1,length({K_1,K_2,K_3}))],'Method',[],...
        'ExtrapolationOrder',[],'thread',NumThreads, ...
        'orient','curvefit');
else
    outputVarStack = reshape(outputVarStack, [],GNDSGE_SIZE_STATE{:});
    output_interp=struct('form','MKL','breaks',{{K_1,K_2,K_3}},...
        'Values',outputVarStack,...
        'coefs',[],'order',[OutputInterpOrder*ones(1,length({K_1,K_2,K_3}))],'Method',[],...
        'ExtrapolationOrder',[],'thread',NumThreads, ...
        'orient','curvefit');
end
IterRslt.output_interp=myinterp(output_interp);

output_var_index=struct();
output_var_index.s_1=1:1;
output_var_index.s_2=2:2;
output_var_index.s_3=3:3;

IterRslt.output_var_index = output_var_index;

        IterRslt.Metric = GNDSGE_Metric;
        IterRslt.Iter = GNDSGE_Iter;
        IterRslt.shock_num = shock_num;
        IterRslt.shock_trans = shock_trans;
        IterRslt.params = v2struct(beta,sigma,alpha,delta,phi,GNDSGE_EMPTY);
        IterRslt.var_shock = v2struct(z_agg,z_ind_1,z_ind_2,z_ind_3);
        IterRslt.var_state = v2struct(K_1,K_2,K_3);
        IterRslt.var_policy = v2struct(s_1,s_2,s_3,K_1_next,K_2_next,K_3_next);
        IterRslt.var_interp = v2struct(s_1_interp,s_2_interp,s_3_interp);
        IterRslt.var_aux = v2struct(GNDSGE_EMPTY);
        IterRslt.pp = v2struct(GNDSGE_PP_s_1_interp,GNDSGE_PP_s_2_interp,GNDSGE_PP_s_3_interp,GNDSGE_SPLINE_VEC,GNDSGE_EMPTY);
        IterRslt.GNDSGE_PROB = v2struct(GNDSGE_LB,GNDSGE_UB,GNDSGE_SOL,GNDSGE_F,GNDSGE_SIZE);
        IterRslt.var_others = v2struct(GNDSGE_EMPTY);

        if ~NoSave
            if IterSaveAll
                save(['IterRslt_capital_planner_3hh_' num2str(GNDSGE_Iter) '.mat']);
            else
                save(['IterRslt_capital_planner_3hh_' num2str(GNDSGE_Iter) '.mat'],'IterRslt');
            end
        end
    end
end
    

%% Return the success flag
IterFlag = 0;
end
