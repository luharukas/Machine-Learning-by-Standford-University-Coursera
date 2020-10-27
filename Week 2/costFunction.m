function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
cost=zeros(m,1);
pred=(theta'*X')' ;
pred=sigmoid(pred);
for i=1:m
if y(i)==1
cost(i)=-log(pred(i));
else
cost(i)=-log(1-pred(i));
end
end
J=sum(cost)/m;
hypo=(pred-y).*X;
add=sum(hypo);
grad=(add./m)' ;

% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
