function [labels] = SpectralCluster(W, k, algo)
% Inputs
% W: m-by-m real non-negative symmetric matrix holding the affinities between the input
% points to cluster. In particular, entry (r,s) holds the affinity between input points r x and
% s x .
% k: Integer >= 2 holding the required number of clusters.
% algo: String specifying the spectral clustering algorithm to carry out:
% 'rc' Ratio-Cuts
% 'nc' Normalized-Cuts
% Outputs
% labels: m-element vector of integers between 1 and k. A values of r in entry i means
% that i x was assigned to cluster r.

narginchk(3,3);
m = size(W, 1);
D = diag(W * ones(m,1));

%create W' according to received algo string.
if (strcmp(algo, 'rc'))
    w = W - D + eye(m);
end
if (strcmp(algo, 'nc'))
    w = D^(-0.5) * W * D^(-0.5);
end

[V,~] = eigs(w, k);

Y = normr(V);

[~, labels] = KMeans(Y', k, 25, 0, 5);

end

