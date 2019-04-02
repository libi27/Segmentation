function labels = NystromNCuts(A,B,k)

Nsamples = size(A,1);
Nothers  = size(B,2);

% Compute total connection weight 
disp('computing total connection weight...');
d1 = sum([A;B'],1);
d2 = sum(B,1) + sum(B,2)'*pinv(A)*B;
d  = [d1 d2]';
disp('done.')

% Normalize
disp('normalizing...');
v = sqrt(1./d);
A = A.*(v(1:Nsamples)*v(1:Nsamples)');
B = B.*(v(1:Nsamples)*v(Nsamples+(1:Nothers))');
disp('done.')

% Find eigenvectors via PCA/Nystrom trick
disp('computing eigenvectors...')
[U,S,~] = svd(A);
Asi     = U*pinv(sqrt(S))*U';
Q       = A+Asi*(B*B')*Asi;
[U,L,~] = svd(Q);
Va      = [A;B']*Asi*U*pinv(sqrt(L));
for i = 1:Nsamples-1
  V(:,i) = Va(:,i+1)./Va(:,1);
end;
disp('done.')

% Concatenate k leading eigenvectors into a matrix, normalize rows, and apply k-means
V1          = V(:,1:k);
Vnormalized = V1./repmat(sqrt(sum(V1.^2,2)),1,k);
[~,labels]  = KMeans(Vnormalized',k,25,0,5);

