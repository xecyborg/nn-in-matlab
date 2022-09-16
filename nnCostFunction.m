function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)


% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


X = [ones(m, 1), X];

a1 = X;
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1), a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

h_x = a3;

y_vec = (1:num_labels) == y;

J = (1/m)*(sum(sum((-y_vec.*log(h_x)) - ((1 - y_vec).*log(1 - h_x)))));


A1 = X;
Z2 = A1*Theta1';
A2 = sigmoid(Z2);
A2 = [ones(size(A2, 1), 1), A2];
Z3 = A2*Theta2';
A3 = sigmoid(Z3);

y_vec = (1:num_labels) == y;

delta3 = A3 - y_vec;
delta2 = (delta3 * Theta2) .* [ones(size(Z2,1),1) sigmoidGradient(Z2)];
delta2 = delta2(:, 2:end);


Theta1_grad = (1/m) * (delta2' * A1);
Theta2_grad = (1/m) * (delta3' * A2);


reg = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J = J + reg;

Theta1_reg = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_reg = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

Theta1_grad = Theta1_grad + Theta1_reg;
Theta2_grad = Theta2_grad + Theta2_reg;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
