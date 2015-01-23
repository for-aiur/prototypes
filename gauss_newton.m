%gauss-newton algorithm for non-linear regression
clear

%parameters from original set
p0 = 5;
p1 = 3;
p2 = 7;

%# of observations
n = 100;

%statistics for random error generation
stddev = 5;

x = linspace(1,5,n);
y = p2*(x.*x) + p1*x + p0 + (stddev*randn(n,1)');

%initial para<meters
params(1) = 0;
params(2) = 0;
params(3) = 0;
params = params';
param_stock(:,1) = params;

%jacobian matrix
J = [-ones(n,1) -x' -x'.^2];

%iteration count
iter = 10;

for i=1:iter
    last_params = param_stock(:,end);
    
    r = y - (last_params(3)*(x.*x) + last_params(2)*x + last_params(1));
    params = last_params - (inv(J'*J)*J'*r');
    
    %add breaking mechanism to the loop
    
    param_stock(:,end+1) = params;
end

params
plot(linspace(1,size(param_stock,2),size(param_stock,2)),param_stock(3,:), '-r');
hold on;
plot(linspace(1,size(param_stock,2),size(param_stock,2)),param_stock(2,:), '-g');
plot(linspace(1,size(param_stock,2),size(param_stock,2)),param_stock(1,:), '-b');
hold off;

A = [ones(n,1) x' x'.*x'];
plot(x,y,'o');
hold on;
plot(x, A*params, 'r-');

%result statistics
SSE = sum((y'-A*params).^2)
%variance or MSE of regression
variance = SSE / (n-2)
disp(sprintf('standard deviation of unknowns: %d',sqrt(variance)));