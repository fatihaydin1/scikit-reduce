%% Boundary-aware Local density-based Outlier Detection (BLDOD)
function [ score ] = BLDOD_oct( X, lambda )
    
    % Compute the number of points
    m = size(X, 1);
    
    % Compute the standard deviation of the points
    stdDev = omitnan_std(X);
    
    N = zeros(m, 1);
    
    % Multiply the norm of the standard deviation of the points with lambda
    deviation = lambda * norm(stdDev);
    
    for i = 1 : m
        % Move the considered point to the center (0, 0)
        D = X - X(i, :);
        % The number of points that are as far away as the deviation length value from the considered point
        N(i) = sum(sqrt(sum(D.^2, 2)) <= deviation);
    end
    
    % Normalize the values between 0-1
    score = 1 - normalize(N, 'range');
end

