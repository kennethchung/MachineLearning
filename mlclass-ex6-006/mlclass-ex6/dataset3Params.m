function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

c_vector=[0.01;0.03,;0.1;0.3;1;3;10;30];
sigma_vect = [0.01;0.03,;0.1;0.3;1;3;10;30];
%error_master = zeros(size(c_vector,1)^2,3);
%row = 1;
minerror = 10000;

for const_c = 1:size(c_vector)
	for std = 1:size(sigma_vect) 
		model= svmTrain(X, y, c_vector(const_c), @(x1, x2) gaussianKernel(x1, x2, sigma_vect(std)));
		error = mean(double(svmPredict(model,Xval) ~= yval));
		if error < minerror
			minerror = error;
			C = c_vector(const_c);
			sigma = sigma_vect(std);
		end
		%temp = [c_vector(const_c);sigma_vect(std);error];
		%error_master(row,:)=temp;
		%row = row+1;
	end
end
C
sigma





% =========================================================================

end
