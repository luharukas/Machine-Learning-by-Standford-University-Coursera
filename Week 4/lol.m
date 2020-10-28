function [J, grad] = lol(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
% add ones (bias) to the first column of X
X = [ones(m, 1), X];
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% =========================================================================
a1=X;
a2=[ones(m,1) sigmoid(X*Theta1')];
a3=sigmoid(a2*Theta2');
% don't regularize the bias - we are replacing zero vector in Theta matrix
tmp=eye(num_labels);
y1=tmp(y,:);
J=(sum(sum((y1.*log(a3))+((1-y1).*log(1-a3)))))*(-1/m);
regular=((sum(sum(Theta1(:,2:3).^2)))+ (sum(sum(Theta2(:,2:5).^2))))*(lambda/(2*m));
J=J+regular;

% -------------------------------------------------------------
del3=a3-y;
del2=(Theta2'*del3')'.*a2.*(1-a2) ;
grad2=(del3'*a2) ;
grad1=((del2(:,2:5))' * a1);
grad2(:,1)=(grad2(:,1)/m);
grad2(:,2:5)=(grad2(:,2:5)+(lambda*Theta2(:,2:5)))/m ;
grad1(:,1)=(grad1(:,1)/m);
grad1(:,2:3)=(grad1(:,2:3)+(lambda*Theta1(:,2:3)))/m ;
Theta1_grad=grad1;
Theta2_grad=grad2; 

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
