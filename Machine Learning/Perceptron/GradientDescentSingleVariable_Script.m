clc;close all;clear all;
% Data
x = rand(5,1) * 10;
y = 3 * (x) + 5;

figure 1;
title 'Data Points';
plot(x,y,'b*');

% Declare initial weights
weights = [
0
0
];

X = [ones(length(x),1) x];  % This is done for getting the bias into account and making things easier

disp(['The initial cost is ', num2str(CostFunction(x,y,weights))]);
disp('The current weights are : ');
disp(weights);

figure 2;
% Plot our initial data
title 'Initial Points'
plot(x,y,'b*',x,X*weights,'r--');

disp('Press anything to start gradient descent');

iterations = input('Enter the number of iterations : ');
alpha = input('Enter the learning rate : ');
costHistory = [];
for i=1:iterations  
  [ weights, cost ] = GradientDescentSingleVariable(x, y, alpha, weights);
  costHistory(length(costHistory) + 1) = cost;  
end

bar(costHistory);
disp([num2str(iterations), ' iterations done, the final weights are : ']);
disp(weights);

figure 3;
% Plot our final data
plot(x,y,'b*',x,X*weights,'r--'); title 'Final output';
