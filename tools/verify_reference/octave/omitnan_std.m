%% Column-wise sample standard deviation ignoring NaN (MATLAB std(X,'omitnan')).
function s = omitnan_std(X)
    s = zeros(1, size(X, 2));
    for j = 1:size(X, 2)
        v = X(~isnan(X(:, j)), j);
        if isempty(v)
            s(j) = NaN;
        else
            s(j) = std(v);
        end
    end
end
