function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
hypothesis=sigmoid(X * theta);
term1= -y.*log(hypothesis);
term2= (1 -y).*log(1-hypothesis);


% Skip the frist theta(1)
% regTerm=((lambda * 0.5 )/m)*sum( theta(2:size(theta,1)).^2);
regTerm=((lambda * 0.5 )/m)*sum( theta(2:end).^2);

J= 1/m * sum(term1-term2) + regTerm;

% grad(1) skip the regTerm
grad(1) = ((1/m) * sum( (hypothesis - y).* X(:,1) ));

% calculate the grad with lamda starting from theta(2)
for i = 2:size(X,2)    	
	grad(i) = ((1/m) * sum( (hypothesis - y).* X(:,i) )) + (lambda/m)*theta(i);
end


% =============================================================

end
