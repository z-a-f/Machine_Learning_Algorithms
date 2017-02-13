function alpha=logistic_regression_backtracking(xtrain,ytrain,w,max_it,eps)
% xtrain: d x n matrix
% ytrain: 1 x n label vector {-1,1}
% output: 1 x d classification vector


if nargin<3
    w=ones(1,size(xtrain,2));
end
if nargin<4
    max_it=100;
end
if nargin<5
    eps=1e-6;
end

if length(unique(ytrain))==1
    alpha=zeros(1,size(xtrain,1));
    alpha(end)=unique(ytrain);
    return;
end


alpha=zeros(1,size(xtrain,1));

for it=1:1:max_it
    
    L=sum(log(1+exp(-ytrain.*(alpha*xtrain))));
    % L(isnan(L))=-ytrain(1,isnan(L)).*(alpha*xtrain(:,isnan(L)));
    
    
    dL=xtrain*
        (-w.*ytrain ./ 
        (1+exp(ytrain.*(alpha*xtrain)))
    )';
    dL2weight=w./(2+exp(ytrain.*(alpha*xtrain))+exp(-ytrain.*(alpha*xtrain)));
    dL2=(xtrain.*repmat(dL2weight,size(xtrain,1),1))*(xtrain');
    grad_dir=-pinv(dL2)*dL;
%     grad_dir=-(dL2\dL);
    

    backtrack_counter=1;
    t=1;
    a=.1;
    b=.9;
    crit_fac=a*dL'*grad_dir;
    temp_loss=log(1+exp(-ytrain.*((alpha+t*grad_dir')*xtrain)));
    % temp_loss(isnan(temp_loss))=-ytrain(1,isnan(temp_loss)).*((alpha+t*grad_dir')*xtrain(:,isnan(temp_loss)));
    while sum(temp_loss)>(L+t*crit_fac)
        if backtrack_counter>=50
            break
        end
        t=b*t;
        temp_loss=log(1+exp(-ytrain.*((alpha+t*grad_dir')*xtrain)));
        temp_loss(isnan(temp_loss))=-ytrain(1,isnan(temp_loss)).*((alpha+t*grad_dir')*xtrain(:,isnan(temp_loss)));
        backtrack_counter=backtrack_counter+1;
    end
    
    alpha=alpha+(t*grad_dir)';
    if max(abs(t*grad_dir))<eps
        break;
    end
end
