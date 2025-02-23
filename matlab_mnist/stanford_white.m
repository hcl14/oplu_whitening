function [xZCAwhite, whitemat, avg, sigma, xTilde, D] = stanford_white(x,epsilon, k)

%http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening

avg = mean(x, 2);     % Compute the mean pixel intensity value separately for each patch. 
x = x - repmat(avg, 1, size(x, 2)); 

sigma = x * x' / size(x, 2);


[U,S,V] = svd(sigma);

% code to use only k principal components:

%xRot = U' * x;          % rotated version of the data. 

svect = diag(S);

%if (2 == 1)
%    k = 0; % select first k values
%    sum_values = sum(svect);
%    current_sum = 0;

%    while (current_sum/sum_values < 0.99)
%        k = k+1;
%        current_sum = sum(svect(1:k));
%    end
%else
%k=150;
%end


D = diag(1./sqrt(svect(1:k) + epsilon)) * U(:,1:k)';
xTilde = diag(1./sqrt(svect(1:k) + epsilon)) * U(:,1:k)' * x; % reduced dimension representation of the data, 
                        % where k is the number of eigenvectors to keep =
                        % PCA whitening



whitemat = U * diag(1./sqrt(diag(S) + epsilon)) * U';

D = whitemat; % ZCA whitening matrix

xZCAwhite = whitemat * x;

%xZCAwhite = whitemat * xTilde;

end

