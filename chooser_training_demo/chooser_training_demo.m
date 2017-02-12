%% Load data

% load('image_segmentation_dataset')
% ytrain=(-1).^(ytrain<4);
% ytest=(-1).^(ytest<4);

load('pendigits');
ytrain=(-1).^(ytrain<5);
ytest=(-1).^(ytest<5);

[d,ntrain]=size(xtrain);
[~,ntest]=size(xtest);

%% Train linear classifier
alpha=logistic_regression_backtracking([xtrain;ones(1,ntrain)],ytrain);
alpha_offset=alpha(end);
alpha=alpha(1:end-1);

yhatl_train=sign(alpha*xtrain+alpha_offset);
yhatl_test=sign(alpha*xtest+alpha_offset);
disp(['Linear classifier performance: ' char(10) '   Train error: ' num2str(mean(yhatl_train~=ytrain)*100) ' % ' char(10) '   Test Error: ' num2str(mean(yhatl_test~=ytest)*100) ' % ']);

%% Train poly classifier
poly_degree=3;

xtrainp=ones(d*poly_degree+1,ntrain);
xtestp=ones(d*poly_degree+1,ntest);
for deg=1:poly_degree
    xtrainp(1+(deg-1)*d:deg*d,:)=xtrain.^deg;
    xtestp(1+(deg-1)*d:deg*d,:)=xtest.^deg;
end
alphap=logistic_regression_backtracking(xtrainp,ytrain);
yhatp_train=sign(alphap*xtrainp);
yhatp_test=sign(alphap*xtestp);
disp(['Poly classifier performance: ' char(10) '   Train error: ' num2str(mean(yhatp_train~=ytrain)*100) ' % ' char(10) '   Test Error: ' num2str(mean(yhatp_test~=ytest)*100) ' % ']);


%% Train a chooser function

% choose budget points
n_budget_points=20;
tradeoffs=linspace(0,1,n_budget_points);

% augment data with budgets
xtrain_aug=[kron(xtrain,ones(1,n_budget_points));repmat(tradeoffs,1,ntrain)];

% define pseudo_labels for each point
pseudo_labels=-ones(1,ntrain);
pseudo_labels(yhatp_train==ytrain & yhatl_train~=ytrain)=1;
pseudo_labels=kron(pseudo_labels,ones(1,n_budget_points));

% define importance weights
weights=yhatl_train~=ytrain & yhatp_train==ytrain;
weights=kron(weights,ones(1,n_budget_points))+repmat(tradeoffs,1,ntrain);

% train chooser
chooser=logistic_regression_backtracking([xtrain_aug;ones(1,size(xtrain_aug,2))],pseudo_labels,weights);
chooser_offset=chooser(end);
chooser(end)=[];

% evaluate_performance
%%
test_error=ones(1,n_budget_points);
test_cost=zeros(1,n_budget_points);
for k=1:n_budget_points
    temp_labels=sign(chooser(1:end-1)*xtest+chooser(end)*tradeoffs(k)+chooser_offset);
    test_cost(k)=mean(temp_labels==1);
    yhat_test=yhatl_test;
    yhat_test(temp_labels==1)=yhatp_test(temp_labels==1);
    test_error(k)=mean(yhat_test~=ytest);
end

test_error=[mean(yhatp_test~=ytest) test_error];
test_cost=[1 test_cost];
plot(test_cost,test_error);xlim([0 1]);xlabel('Fraction of Test Examples using Poly');ylabel('Test Error');



