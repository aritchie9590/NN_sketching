clear all
rng(1);
%% Mert Pilanci
%% September 2018
%% Sketching one layer neural net
%% choose activation function s
%% and the derivative of the activation function sd
s = @(x)1./(1+exp(-x)); % sigmoid
%sd = @(x)(exp(x))./((1+exp(x)).^2);
sd = @(x) s(x).*(1-s(x));
% s = @(x)max(0,x); %ReLU
% sd = @(x)(x>0);
% s = @(x) exp(x); 
% sd = @(x) exp(x);
%s= @(x) x.^2;
%sd = @(x) 2*x;
% %     
%   s = @(x) x.*(x>=0);
%   sd = @(x) (x>=0);
%s = @(x) x;
%sd = @(x) ones(size(x));
load('mnist_all.mat');
%% construct data matrix
Amat=sparse(double([train0;train1;train2;train3;train4;train5;train6;train7;train8;train9]));
Atest = sparse(double([test0;test1;test2;test3;test4;test5;test6;test7;test8;test9]));
[m,n] = size(Amat); 
AMAX = max(max(Amat)); %% normalizse data matrix
Amat = Amat/AMAX;
Atest = Atest/AMAX;
CGITER = 4; %% Conjugate Gradient iterations
CGTOL = 1e-6;
B = 1*n; %% SIZE OF THE SKETCH
%ITERMAX = (ceil(m/(B))-1)*1;
ITERMAX = 10;   %sketching Gauss Newton
ITERNEWTON = 5; %Gauss-Newton
ITERMAXSGD = 10;    %sketching SGD
% regularization parameter
lambda = 10/m; %% 00.69 NS lambda = 10/m; ms= 500; backtracking %% ms = 1000 daha iyi 2000 daha iyi
NTOL = 1e-5;
MAXBACKTRACK = 50;
ALPHA = 0.2;
BETA = 0.5;

%Train a classifier to distinguish between 0 digits and other digits
y = zeros(m,1); %% y is the indicator of digit 0
y(1:length(train0)) = 1;
ytest = zeros(length(Atest),1);
ytest(1:length(test0)) = 1;
%Amat = Amatrix;
%Amat = Amatrix(1:B,:);
%yorg = y;
%y = y(1:B);
%%
x = randn(n,1);
xinit = x;
%x = zinit;
%mu = 1-1/10;
%%%%%%   GAUSS-NEWTON METHOD
mu = 1;
cost2(1) = norm(s(Amat*x)-y)^2/m + lambda*norm(x)^2;
errtest2(1) = sum(abs(round(s(Atest*x))-ytest))/length(ytest);
tic
for iter = 1:ITERNEWTON
    %H = Amat'*abs(diag(Amat*x)).^2*Amat;
    %x = x-mu*H^-1*Amat'*diag(abs(Amat*x).^2-y)*(Amat*x);
    
    % %Compute the Jacobian of the network
    d1 = sd(Amat*x);
    DA = (bsxfun(@times,Amat,d1));
    DA_test = Amat .* d1;
    
    if(norm(DA_test - DA,'fro')>1e-6)
        error('Your gradient is not the same');
    end
%    H =  DA'*DA/m + lambda*eye(n);
   
    % %Compute the gradient of the network wrt the nn params (Regular ole
    % backprop)
    d2 = s(Amat*x)-y;
    DA2 = (bsxfun(@times,Amat,d2));
    grad = DA2'*(d1)/m+lambda*x;
    
    grad_test =  Amat'*(d2.*d1) / m + lambda*x;
    grad_test2 = DA'*d2 /m + lambda*x;
    
    
    if(norm(grad - grad_test)>1e-6 || norm(grad - grad_test2) > 1e-6)
        error('Your gradient is not the same');
    end
    
    %Hchol = sparse(chol(H));
    %v = Hchol\(Hchol'\(DA2'*(d1)));
%    v = (H\(DA2'*(d1)/m+lambda*x));

    % %Solve Gauss-Newton's equation for GN direction (v)
    [v, cgsflag, cgsrelres] = cgs(@(x) DA'*(DA*x)/m+lambda*x,grad,CGTOL,CGITER,[],[],[]); %this is not the same x as above/below! Used to solve for the Nt direction "v"
    val = norm(s(Amat*(x))-y)^2/m + lambda* norm(x)^2/2;
    fprime = grad'*v;
    t = 1;
    bts = 0;
%   valnext = sum(g0(A*(x+t*v))); %Next value

    % %Line search 
    while (norm(s(Amat*(x-t*v))-y)^2/m + lambda* norm(x-t*v)^2/2 > val + ALPHA*t*fprime )
        t = BETA*t;
        bts = bts + 1;
        if bts>MAXBACKTRACK
            disp('Maximum backtracking reached, accuracy not guaranteed')
            break
        end
    end
    
    % %Update the nn params
    x = x-t*mu*v;

    if abs(fprime)<NTOL
      break
    end

    %err2(iter) = min(norm(x-x0)^2,norm(x+x0)^2);
    grad2(iter) = norm(v); %(v'*(DA2'*(d1)+lambda*x))
    cost2(iter+1) = val;
    errtest2(iter+1) = sum(abs(round(s(Atest*x))-ytest))/length(ytest);

end
timeGaussNewton = toc
%Amat = Amatrix;
%y = yorg;


%%  GAUSS-NEWTON SKETCH
x = xinit;
mu = 1;
cost3(1) = norm(s(Amat*x)-y)^2/m + lambda*norm(x)^2;
%ms = 16*n;

%S2 = sjlt(B,m,1);
%AmatS = S2*Amat;
%yS = S2*y;
randp = [randperm(m),randperm(m),randperm(m),randperm(m),randperm(m),randperm(m),randperm(m),randperm(m),randperm(m),randperm(m),randperm(m),randperm(m)];

%AmatS = Amat(B*(iter-1)+1:B*(iter),:);
%yS = y(B*(iter-1)+1:B*(iter));
errtest3(1) = sum(abs(round(s(Atest*x))-ytest))/length(ytest);

tic
for iter = 1:ITERMAX
    %H = Amat'*abs(diag(Amat*x)).^2*Amat;
    %x = x-mu*H^-1*Amat'*diag(abs(Amat*x).^2-y)*(Amat*x);
    %AmatS = Amat(B*(iter-1)+1:B*(iter),:);
    %yS = y(B*(iter-1)+1:B*(iter));
    %% Sampling sketch
    AmatS = Amat(randp(B*(iter-1)+1:B*(iter)),:);
    yS = y(randp(B*(iter-1)+1:B*(iter)),:);
    
    %AmatS = Amat(B*(1-1)+1:B*(1),:);
    
    %yS = y(B*(1-1)+1:B*(1));
    
   %  AmatS = Amat(B*(1-1)+1:B*(1),:);
    % yS = y(B*(1-1)+1:B*(1));
    
%     tStart = tic;
    d1 = sd(AmatS*x);
    DA = (bsxfun(@times,AmatS,d1));
%     H =  DA'*DA/m+lambda*eye(n);
%     tElapsed = toc(tStart);
    
%     tStart = tic;
%     d1test = sd(Amat*x);
%     DAtest = (bsxfun(@times,Amat,d1test));
%     Htest = DAtest'*DAtest / m + lambda*eye(n);
%     tElapsed2 = toc(tStart);
    
    d2 = s(AmatS*x)-yS;
    DA2 = (bsxfun(@times,AmatS,d2));
%     DA2 = Amat .* s(Amat*x)-y;
    %Hchol = sparse(chol(H));
    %v = Hchol\(Hchol'\(DA2'*(d1)));
    %v = (H\(DA2'*(d1)/m+lambda*x));
    
    d1test = sd(Amat*x);
    grad = (DA2'*(d1)/m+lambda*x);
%     grad = (DA2'*(d1test)/m+lambda*x);
    [v, cgsflag, cgsrelres] = cgs(@(x) DA'*(DA*x)/m+lambda*x,grad,CGTOL,CGITER,[],[],[]);
    x = x-mu*v;
    %err2(iter) = min(norm(x-x0)^2,norm(x+x0)^2);
    cost3(iter+1) = norm(s(Amat*x)-y)^2/m+lambda*norm(x)^2;
    %grad2(iter) = norm(Amat'*diag(abs(Amat*x).^2-y)*Amat*x);
    errtest3(iter+1) = sum(abs(round(s(Atest*x))-ytest))/length(ytest);

end
timeSketch = toc
xSketch = x;
%% STOCHASTIC GRADIENT DESCENT

x = xinit;
mu = 1e-2;
AmatS = Amat;
yS = y;
costGD(1) = norm(s(Amat*x)-y)^2/m+lambda*norm(x)^2;
errtestGD(1) = sum(abs(round(s(Atest*x))-ytest))/length(ytest);
tic
for iter = 1:ITERMAXSGD
    %H = Amat'*abs(diag(Amat*x)).^2*Amat;
    %x = x-mu*H^-1*Amat'*diag(abs(Amat*x).^2-y)*(Amat*x);
    %% Sample 
    AmatS = Amat(randp(B*(iter-1)+1:B*(iter)),:);
    yS = y(randp(B*(iter-1)+1:B*(iter)),:);
    
    %  AmatS = Amat(B*(1-1)+1:B*(1),:);
    % yS = y(B*(1-1)+1:B*(1));
    
    d1 = sd(AmatS*x);
    DA = (bsxfun(@times,AmatS,d1));    
    d2 = s(AmatS*x)-yS;
    DA2 = (bsxfun(@times,AmatS,d2));
    %Hchol = sparse(chol(H));
    %v = Hchol\(Hchol'\(DA2'*(d1)));
    v = ((DA2'*(d1))+lambda*x);
    x = x-mu*v/iter;
    costGD(iter+1) = norm(s(Amat*x)-y)^2/m+lambda*norm(x)^2;
    errtestGD(iter+1) = sum(abs(round(s(Atest*x))-ytest))/length(ytest);
end
timeSGD = toc;

%%

%%  GAUSS-NEWTON ITERATIVE SKETCH
x = xinit;
mu = 1e-1;
cost4(1) = norm(s(Amat*x)-y)^2/m + lambda*norm(x)^2;
%ms = 16*n;

%S2 = sjlt(B,m,1);
%AmatS = S2*Amat;
%yS = S2*y;
randp = [randperm(m),randperm(m),randperm(m),randperm(m),randperm(m),randperm(m),randperm(m),randperm(m),randperm(m),randperm(m),randperm(m),randperm(m)];

%AmatS = Amat(B*(iter-1)+1:B*(iter),:);
%yS = y(B*(iter-1)+1:B*(iter));
errtest4(1) = sum(abs(round(s(Atest*x))-ytest))/length(ytest);

tic
for iter = 1:ITERMAX
    %H = Amat'*abs(diag(Amat*x)).^2*Amat;
    %x = x-mu*H^-1*Amat'*diag(abs(Amat*x).^2-y)*(Amat*x);
    %AmatS = Amat(B*(iter-1)+1:B*(iter),:);
    %yS = y(B*(iter-1)+1:B*(iter));
    
    
    AmatS = Amat(randp(B*(iter-1)+1:B*(iter)),:);
    yS = y(randp(B*(iter-1)+1:B*(iter)),:);

    % Full gradient 
    d1 = sd(Amat * x);
    d2 = s(Amat*x)-y;
    DA2 = (bsxfun(@times,Amat,d2));
    grad = (DA2'*(d1) / m + lambda*x);

    % Sketched Jacobian
    d1 = d1(randp(B*(iter-1)+1:B*(iter)),:);
    DA = (bsxfun(@times,AmatS,d1));
    
    CGITER = 10;
    CGTOl = 1e-8;

    [v, cgsflag, cgsrelres] = cgs(@(x) DA'*(DA*x)/m+lambda*x,grad,CGTOL,CGITER,[],[],[]);
    x = x-mu*v;
    %err2(iter) = min(norm(x-x0)^2,norm(x+x0)^2);
    cost4(iter+1) = norm(s(Amat*x)-y)^2/m+lambda*norm(x)^2;
    %grad2(iter) = norm(Amat'*diag(abs(Amat*x).^2-y)*Amat*x);
    errtest4(iter+1) = sum(abs(round(s(Atest*x))-ytest))/length(ytest);

end
timeSketch_iterative = toc
xSketch_iterative = x;

%%

%

%xInv = Amat\-log(1./y-1);
%norm(xInv-x0)^2
%norm(x-x0)^2
figure

subplot(2,1,1)
semilogy((0:length(cost2)-1)*timeGaussNewton/ITERMAX,cost2,'r');
hold on;
semilogy((0:length(cost3)-1)*timeSketch/ITERMAX,cost3,'g');
semilogy((0:length(cost4)-1)*timeSketch_iterative/ITERMAX,cost4,'b');
semilogy((0:length(costGD)-1)*timeSGD/ITERMAX,costGD,'k');
grid
ylim([1e-3 1])
title('Train')
subplot(2,1,2)
semilogy((0:length(errtest2)-1)*timeGaussNewton/ITERMAX,errtest2,'r');
hold on;
semilogy((0:length(errtest3)-1)*timeSketch/ITERMAX,errtest3,'g');
semilogy((0:length(errtest4)-1)*timeSketch_iterative/ITERMAX,errtest4,'b');
semilogy((0:length(errtestGD)-1)*timeSGD/ITERMAX,errtestGD,'k');
grid
ylim([1e-3 1])
title('Test')
legend('Gauss-Newton','Gauss-Newton Sketch','Gauss-Newton Sketch Iterative','SGD')
%save NewtonQuadraticConv