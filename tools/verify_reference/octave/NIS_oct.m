%% Nimble Instance Selection (NIS)
function [ idx ] = NIS_oct( X, varargin )

    narginchk(1, 2);
    
    if nargin == 2
        alpha = varargin{1};
    else
        alpha = 1;
    end

    transformedX = round(alpha*((X-min(X,[],1))./(omitnan_std(X))));
    transformedX(isnan(transformedX)) = 0;
    
    [~, idx, ~] = unique(transformedX,'rows', 'first');
end
