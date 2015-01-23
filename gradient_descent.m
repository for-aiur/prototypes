%solving a linear equation using gradient descent
clear;

%sample observation
%x = [3;4;6;7;10];
%y = [1;2;3;4;6];

p0 = 3;
p1 = 4;
num_points = 100;

x = linspace(1,10,num_points)';
y = p1*x + p0;
y = y + randn(num_points,1);

%# of observations
n = size(x,1);

%initial guess for unknowns
u_storage(1,1) = 0;
u_storage(2,1) = 0;

%step size
alpha = 0.04;

%max iterations
max_iter = 10000;

%energy function
% h = (y - (x*u(1)+u(0)))^2
% step = alpha * 1/n SUM[(u(2)*x + u(1) - y) * x]

A = [ones(n,1) x];

%MAIN LOOP ++++++++++++++++++++++++++++++++++++
for i=1:max_iter
    last_params = u_storage(:,end);   
    
    %- calculate a step value using last parameters    
    step_0 = alpha*(1/n)*( A(:,1)'*((A*last_params) - y));
    step_1 = alpha*(1/n)*( A(:,2)'*((A*last_params) - y));
    steps = [step_0; step_1];
    
    %- calculate new parameters by subtracting step from the last estimate
    params = last_params - steps;
    u_storage(:,end+1) = params;
    
    %- compare new and old value, check if they are less than treshold indicates a convergence
    if(abs(params(1) - last_params(1)) < 0.00001)
        disp(sprintf('iteration count: %d',i))
        break;
    end
end

%result statistics
SSE = sum((y-A*params).^2)
%variance or MSE of regression
variance = SSE / (n-2)
disp(sprintf('standard deviation of unknowns: %d',sqrt(variance)));

%gradient descent vs normal equation
params
normal_eq = inv(A'*A)*A'*y

%iterative plots for parameters
figure;
plot(linspace(1,length(u_storage),length(u_storage)), u_storage(1,:), 'b-');
hold on;
plot(linspace(1,length(u_storage),length(u_storage)), u_storage(2,:), 'g-');
hold off;

%resulting regression line
figure;
plot(x,y,'o');
hold on;
plot(x,A*params,'r-');
plot(x,A*normal_eq,'g-');
hold off;
