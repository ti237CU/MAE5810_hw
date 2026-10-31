%% MAE 5810 - HW3, Extra Credit (Problem 4):

clear; close all; clc;

%% Load Image 
I = imread('yellowlily.jpg');
if max(size(I,1:2)) > 800
    I = imresize(I, 800/max(size(I,1:2)));
end
G = im2double(rgb2gray(I));
[H, W] = size(G);

%% Feature detection
surfPts   = detectSURFFeatures(G, 'MetricThreshold', 500);
harrisPts = detectHarrisFeatures(G, 'MinQuality', 0.01);
[~, hogVis] = extractHOGFeatures(G, 'CellSize', [16 16]);    

figure('Name','Feature detectors','Position',[50 50 1400 450]);
subplot(1,3,1);
imshow(I); hold on;
plot(surfPts.selectStrongest(150));
title(sprintf('SURF features (%d found, 150 strongest shown)', surfPts.Count));
subplot(1,3,2);
imshow(I); hold on;
plot(harrisPts.selectStrongest(300));
title(sprintf('Harris corners (%d found, 300 strongest shown)', harrisPts.Count));
subplot(1,3,3);
imshow(G); hold on;
plot(hogVis, 'Color', 'yellow');
title('HOG (16x16 cells)');

%% Per-pixel feature vector
lab  = rgb2lab(I);
aCh  = lab(:,:,2);
bCh  = lab(:,:,3);       
gEn  = imgaussfilt(imgradient(G), 4);            % local gradient
tex  = stdfilt(G, true(9));                      

% All SURF and Harris locations spread with a Gaussian
allPts = [surfPts.Location; harrisPts.Location];
loc    = round(allPts);
loc    = loc(loc(:,1)>=1 & loc(:,1)<=W & loc(:,2)>=1 & loc(:,2)<=H, :);
kp     = zeros(H, W);
kp(sub2ind([H W], loc(:,2), loc(:,1))) = 1;
kpDen  = imgaussfilt(kp, max(H,W)/40);

F = [aCh(:), bCh(:), gEn(:), tex(:), kpDen(:)];
F = (F - mean(F)) ./ std(F);                     % z-score each feature
wts = [1 1 0.7 0.5 1];                           % color and keypoints weighted most
F = F .* wts;

figure('Name','Per-pixel features','Position',[50 550 1400 300]);
names = {'a* (green-red)','b* (blue-yellow)','gradient energy','texture (std)','keypoint density'};
maps  = {aCh, bCh, gEn, tex, kpDen};
for k = 1:5
    subplot(1,5,k); imagesc(maps{k}); axis image off; colorbar;
    title(names{k});
end

%% Segmentation: k-means with 2 clusters
rng(1);
idx = kmeans(F, 2, 'Replicates', 5, 'MaxIter', 300);
L   = reshape(idx, H, W);

% Foreground, cluster with the higher density per pixel
kpPerPix = [mean(kpDen(L==1)), mean(kpDen(L==2))];
[~, fg]  = max(kpPerPix);
fprintf('Keypoint density: cluster 1 = %.4g, cluster 2 = %.4g -> foreground = %d\n', ...
        kpPerPix(1), kpPerPix(2), fg);
fprintf('Mean b* (yellowness): cluster 1 = %.1f, cluster 2 = %.1f\n', ...
        mean(bCh(L==1)), mean(bCh(L==2)));

% Smooth, fill the flower center and keep the largest object
mask = (L == fg);
mask = imopen(mask, strel('disk', 3));
mask = imclose(mask, strel('disk', 7));
mask = imfill(mask, 'holes');
mask = bwareafilt(mask, 1);                      % the lily = largest region

st = regionprops(mask, 'Area', 'Centroid', 'BoundingBox');
fprintf('Foreground: %d px (%.1f%% of image), centroid (%.0f, %.0f)\n', ...
        st.Area, 100*st.Area/(H*W), st.Centroid);

%% Results
fgOnly = I .* uint8(mask);
over   = labeloverlay(I, mask, 'Colormap', [1 0 1], 'Transparency', 0.6);
txtPos = max(st.BoundingBox(1:2) + [0 -30], 1);  % keep the label inside the image
over   = insertText(over, txtPos, 'Yellow Lily (foreground)', ...
                    'FontSize', 18, 'BoxColor', 'yellow', 'BoxOpacity', 0.8);

figure('Name','Segmentation result','Position',[50 50 1400 450]);
subplot(1,3,1); imshow(L, []);      title('k-means clusters');
subplot(1,3,2); imshow(over);       title('Foreground overlay');
hold on; visboundaries(mask, 'Color', 'r', 'LineWidth', 1);
subplot(1,3,3); imshow(fgOnly);     title('Foreground only');

% Keypoints that fall on the foreground vs background
inFg = mask(sub2ind([H W], loc(:,2), loc(:,1)));
fprintf('Keypoints on lily: %d of %d (%.0f%%)\n', nnz(inFg), numel(inFg), 100*mean(inFg));