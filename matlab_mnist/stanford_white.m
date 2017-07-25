
function [xZCAwhite, whitemat, avg, sigma, xTilde] = stanford_white(x,epsilon)

%http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening

avg = mean(x, 1);     % Compute the mean pixel intensity value separately for each patch. 
x = x - repmat(avg, size(x, 1), 1); 

sigma = x * x' / size(x, 2);


[U,S,V] = svd(sigma);

% code to use only k principal components:

%xRot = U' * x;          % rotated version of the data. 

k = 0; % select first k values

svect = diag(S);
sum_values = sum(svect);
current_sum = 0;

while (current_sum/sum_values < 0.99)
    k = k+1;
    current_sum = sum(svect(1:k));
end

xTilde = diag(1./sqrt(svect(1:k) + epsilon)) * U(:,1:k)' * x; % reduced dimension representation of the data, 
                        % where k is the number of eigenvectors to keep =
                        % PCA whitening



whitemat = U * diag(1./sqrt(diag(S) + epsilon)) * U';

xZCAwhite = whitemat * x;

%xZCAwhite = whitemat * xTilde;

end

