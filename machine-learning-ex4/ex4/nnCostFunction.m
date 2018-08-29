function [J grad] = nnCostFunction(nn_params, ...
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
    %take the nn_params vector, which is the unrolled version thetaVec
    %"1:hidden_layer_size * (input_layer_size + 1)" part of the code will tell 
    %our reshape function the range of theta1 in the thetaVec using the
    %formula for theta at each layer s(j+1) x s(j)+1

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
    %theta2 will follow similar,"(1 + (hidden_layer_size * (input_layer_size +
    %1))" basically starts off our range for theta2 by adding 1 to the end
    %point of theta2. and we move till the end of the nn_params. since size of
    %that theta is output layer(rows) by hiddenlayer (s(j))+1

% Setup some useful variables
m = size(X, 1);     %here m is the size of X matrix number of rows or the number of examples
X=[ones(m,1) X];    %add bias unit to the input 
% You need to return the following variables correctly 
J = 0;

%initialize our matrix for theta1&2_grad
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%-------------------------------------------------------------%
    a_1=X;
        z_2=(a_1)*Theta1';
        a_2=sigmoid(z_2);
    a_2=[ones(size(a_2,1),1),a_2];
        z_3=(a_2)*Theta2';
        a_3=sigmoid(z_3);
    h=a_3;
    %let a_1 be equal to input matrix X (5000x401)
    %we multiply a_1(5000x401) by transpose of matrix Theta1 (401x25)(which
    %gives us an intermediate variable z_2 (5000x25)
    %we get a_2 by using sigmoid on z_2 and add column of 1s to a_2
    %(5000x26)
    %we multiply a_2 by Theta2 transpose 26x10 = z_3=5000x10 which we then
    %sigmoid and get a_3=our hypothesis
    
%--------------------------------------------------------------%
    
    yvec=[1:num_labels]==y;
    %yvec will create a logical matrix of m(length of vector y) by
    %num_labels.
    
    J=sum(((1/m)*sum( (-yvec.*log(h)) - ((1-yvec).*log(1-h)) )));
    %Inner sum function will calculate sum of each colum on matrix output
    %by its argument which will result in 1x10 vector
    %Outter sum will then calculate the sum of the vector and return a
    %final double sum
    
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
