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
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%h = X * theta
%error= h-y
%error_sqr = error.^2
%q = sum(error_sqr)
%J= 0.5 * 1/m *q

hypothesis=sigmoid(X * theta);
term1= -y.*log(hypothesis);
term2= (1 -y).*log(1-hypothesis);

J= 1/m *sum(term1-term2);

grad = zeros(size(theta));

for i = 1:size(X,2)    	
	grad(i) = (1/m) * sum( (hypothesis - y).* X(:,i) );
end

% =============================================================

end