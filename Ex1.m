%% SCRIPT TO ANSWER EXERCISE ONE IN C.V.
function Ex1()
mu = 0;
k = 3;
max_iter = 25;
change_thresh = 0;
num_runs = 5;
addpath(genpath('C:/Users/Lirane/Google Drive/ComputationalBiology4/Vision/optprop'));
% SegmentImageKMeans('dataset/I1.jpg', mu, k, max_iter, change_thresh , num_runs);
% SegmentImageKMeans('dataset/I2.jpg', mu, k, max_iter, change_thresh , num_runs);
% SegmentImageKMeans('dataset/I3.jpg', mu, k, max_iter, change_thresh , num_runs);
% % 
% 
% SegmentImageSpecClust('dataset/I1.jpg', 3, 20, 70);
% SegmentImageSpecClust('dataset/I2.jpg', 3, 15, 100);
SegmentImageSpecClust('dataset/I3.jpg', 3, 30, 100);

% 
% k = 3;
% sigC = 10;
% sigS = 600;
% i = 1;
% figure;
% for n = [50 100 200 400]
%     tic
%     % SegmentImageNystrom('dataset/I1.jpg', n, k, sigC, sigS,i);
%     % SegmentImageNystrom('dataset/I2.jpg', n, k, sigC, sigS,i);
% %     SegmentImageNystrom('dataset/I3.jpg', n, k, sigC, sigS, i);
%     toc
%     % i = i+1;
% end
end

%% PART TWO:
function SegmentImageKMeans(path, mu, k, max_iter, change_thresh , num_runs)

if mu < 0
    error('invalid input - mu must be a positive integer');
end
img = rgb2lab(im2double(imread(path)));
col = size(img,2);
row = size(img,1);


T = ones(row,col,5);
[X, Y] = meshgrid(1:col,1:row);
T(:,:,1:3) = img;
T(:,:,4) = X.*mu ;
T(:,:,5) = Y.*mu;

[~, labels] = KMeans(reshape(T,row*col,5)', ...
                k, max_iter, change_thresh , num_runs);

figure;
imagesc(reshape(labels,row,col));
title(sprintf('KMeanSeg with mu=%d, k=%d, max_iter=%d,\n change_thresh=%d, num_runs=%d',...
    mu, k, max_iter, change_thresh , num_runs))
end

%% PART FOUR
function SegmentImageSpecClust(path, k, sigmaC, sigmaS)

img = imresize(im2double(imread(path)),1/8);
col = size(img,2);
row = size(img,1);
[X, Y] = meshgrid(1:size(img,2),1:size(img,1));
img = rgb2lab(img);
img = reshape(img,size(img,2)*size(img,1),3);
coordinates = [X(:),Y(:)];
W = exp((-1/(2*(sigmaC^2))) * squareform(pdist(img).^2) ...
      - (1/(2*(sigmaS^2))) * squareform(pdist(coordinates).^2)) ;

figure;
imagesc(reshape(SpectralCluster(W, k, 'rc'),row,col));
title(sprintf('SpectralSeg RC with k=%d,\n sigmaC=%d, sigmaS=%d',...
    k, sigmaC, sigmaS))
figure;
imagesc(reshape(SpectralCluster(W, k, 'nc'),row,col));
title(sprintf('SpectralSeg NC with k=%d,\n sigmaC=%d, sigmaS=%d',...
    k, sigmaC, sigmaS))
end

%% PART FIVE:
function SegmentImageNystrom(path, n, k, sigmaC, sigmaS, i)

img = im2double(imread(path));
col = size(img,2);
row = size(img,1);

[Y, X] = meshgrid(1:row,1:col);
cordin = [X(:),Y(:)]';

img = rgb2lab(img);
img = reshape(img,row*col,3)';

colNum = size(img,2);

randNum = sort(randperm(colNum,n));
cols(:,:) = 1:colNum;
cols(:,randNum) = [];

A = img(:, randNum);
coordinatesA = cordin(:, randNum);
B(:,1:colNum-n) = img(:, cols);
coordinatesB(:,1:colNum-n) = cordin(:, cols);

WA = exp((-1/(2*sigmaC^2)) * squareform(pdist(A').^2) ...
    + (-1/(2*sigmaS^2)) * squareform(pdist(coordinatesA').^2));
WB = exp((-1/(2*sigmaC^2)) * pdist2(A',B').^2 ...
    + (-1/(2*sigmaS^2)) * pdist2(coordinatesA', coordinatesB').^2);

labels = NystromNCuts(WA,WB,k);
lbl_ordered = zeros(size(labels));
lbl_ordered(randNum,:) = labels(1:n,:);
lbl_ordered(cols,:) = labels(n+1:colNum,:);


% subplot(2,2,i),
imagesc(reshape(lbl_ordered, row, col));
% title(sprintf('NystormSeg with n=%d, k=%d,\n sigmaC=%d, sigmaS=%d',...
%     n, k, sigmaC, sigmaS))
end