function [ C, labels ] = KMeans( X, k, max_iter, change_thresh , num_runs )
% Inputs
% 
% X: d-by-m matrix with columns holding x1, x2, ... xd - the input points to cluster.
% k: Integer > 2 holding the required number of clusters.
% max_iter: Positive integer holding the maximum number of iterations allowed in a run.
% change_thresh: Positive scalar holding the minimal objective change allowed between
% two consecutive iterations in a run. If the objective change is smaller than this value,
% the run terminates.
% num_runs: Positive integer holding the number of runs to carry out.
% 
% Outputs
% 
% C: d-by-k matrix with columns holding the cluster centroids.
% labels: m-element vector of integers between 1 and k. A values of r in entry i means
% that i x belongs to cluster r.


narginchk(5,5);
nargoutchk(1,2);

if k < 2
    error('invalid input - k must be bigger or equal than 2');
elseif max_iter < 1 
    error('invalid input - max_iter must be a positive integer');
elseif change_thresh < 0
    error('invalid input - change_thresh must be a positive integer');
elseif num_runs < 0
    error('invalid input - num_runs must be a positive integer');
end
d = size(X,1);
m = size(X,2);
labels = zeros(1,m);
C = zeros(d,k);
min_norm = inf;

for i = 1:num_runs
    % tandomly chose k initial centroids
    centroids = X(:,randperm(m, k));
    iter = 0;
    objective = inf;
    prev_ls_err = 0;
    
    while (iter < max_iter && objective > change_thresh)
        
        [tmp_label, dist] = knnsearch(centroids', X', 'distance', 'euclidean');
        W = zeros(m, k);
        W(sub2ind(size(W), 1:m, tmp_label'))=1;
        centroids = bsxfun(@rdivide, X * W, sum(W));
        ls_err = 1/2 * sum(dist.^2);
        objective = ls_err - prev_ls_err;
        prev_ls_err = ls_err;
        iter = iter + 1;
    end
    
    [tmp_label, dist] = knnsearch(centroids', X', 'distance', 'euclidean');
    iter_score = 0;
    for clust = 1 : k
        mu = mean(dist(tmp_label==clust));
        iter_score = iter_score + norm(X(tmp_label == clust) - mu);
        
    end
      
    if (min_norm > iter_score)
        min_norm = iter_score;
        C = centroids;
        labels = tmp_label;
    end
end

end




