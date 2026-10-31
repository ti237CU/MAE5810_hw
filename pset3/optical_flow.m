%% MAE 5810 - HW3, Problem 1 (Exercise 8.1, parts a-d)

clear; close all; clc;

%% Choose specific frames in video 

vr = VideoReader('viptraffic.avi');
idx = 2:4:vr.NumFrames-1;
imgs = cell(1, numel(idx));
for i = 1:numel(idx)
    imgs{i} = insertText(read(vr, idx(i)), [1 1], sprintf('%d', idx(i)), ...
        'FontSize', 10, 'BoxOpacity', 0.6);
end
figure; montage(imgs, 'Size', [NaN 6]);
title('viptraffic.avi: frame numbers');

%% Load two chosen consecutive frames

vr  = VideoReader('viptraffic.avi');
k   = 86;                          % frame with 2 cars (90 also works); k >= 2
F0  = read(vr, k-1);               % only used for the motion-support mask
F1  = read(vr, k);
F2  = read(vr, k+1);
fps = vr.FrameRate;

I0 = imgaussfilt(double(rgb2gray(F0)), 1);
I1 = imgaussfilt(double(rgb2gray(F1)), 1);   % 0-255 scale; light smoothing
I2 = imgaussfilt(double(rgb2gray(F2)), 1);   % reduces derivative noise
[H, W] = size(I1);

%% Horn-Schunck optical flow

alpha   = 10;
nIter   = 100;
nLevels = 4;
nWarps  = 3;

PA = cell(nLevels,1);  PB = PA;
PA{1} = I1;  PB{1} = I2;
for l = 2:nLevels
    PA{l} = impyramid(PA{l-1}, 'reduce');
    PB{l} = impyramid(PB{l-1}, 'reduce');
end

u = zeros(size(PA{nLevels}));  v = u;
for l = nLevels:-1:1
    [h, w] = size(PA{l});
    if ~isequal(size(u), [h w])
        u = 2*imresize(u, [h w], 'bilinear');
        v = 2*imresize(v, [h w], 'bilinear');
    end
    [Xl, Yl] = meshgrid(1:w, 1:h);
    for wi = 1:nWarps
        Bw  = interp2(PB{l}, Xl + u, Yl + v, 'linear');
        out = isnan(Bw);  Bw(out) = PA{l}(out);
        [du, dv] = hsFlow(PA{l}, Bw, alpha, nIter);
        u = medfilt2(u + du, [3 3], 'symmetric');
        v = medfilt2(v + dv, [3 3], 'symmetric');
    end
end


D01  = imgaussfilt(abs(I1 - I0), 2);
D12  = imgaussfilt(abs(I2 - I1), 2);
T01  = max(4*median(D01(:)), 3);
T12  = max(4*median(D12(:)), 3);
sup  = imclose(D01 > T01, strel('disk', 2)) & imclose(D12 > T12, strel('disk', 2));
sup  = imclose(sup, strel('disk', 3));
sup  = imfill(sup, 'holes');                 % uniform car interior
u(~sup) = 0;  v(~sup) = 0;

figure('Name','Motion support');
imshow(sup);
title(sprintf('Moving pixels of frame %d (differs from frames %d and %d)', k, k-1, k+1));
mag = hypot(u, v);         % pixels/frame
ang = atan2(v, u);         % image coordinates

% Sanity check
fprintf('Flow magnitude: max %.2f px/frame, median %.4f px/frame\n', ...
        max(mag(:)), median(mag(:)));

%% Part a: Optical-flow diagram with quiver 

step = 4; 
[X, Y] = meshgrid(1:W, 1:H);
r = 1:step:H;  c = 1:step:W;

figure('Name','(a) Horn-Schunck optical flow','Position',[100 100 1100 420]);
subplot(1,2,1);
imshow(F1); hold on;
quiver(X(r,c), Y(r,c), u(r,c), v(r,c), 0, 'y', 'LineWidth', 1);
title(sprintf('HS flow on frame %d (frames %d\\rightarrow%d)', k, k, k+1));
legend('flow vector (u,v), true length in px/frame', 'Location','southoutside');

subplot(1,2,2);
quiver(X(r,c), Y(r,c), u(r,c), v(r,c), 0, 'b');
axis ij equal; axis([1 W 1 H]); grid on;
xlabel('x (pixels)'); ylabel('y (pixels)');
title('Flow field only (image axes, y down)');
legend('flow vector (u,v), true length in px/frame', 'Location','southoutside');

figure('Name','Flow magnitude');
imagesc(mag); axis image; colormap(gca, jet); cb = colorbar;
cb.Label.String = '|flow| (pixels/frame)';
xlabel('x (pixels)'); ylabel('y (pixels)');
title('Horn-Schunck flow magnitude');

%% HOF features

B    = 8;
cs   = 4;
binW = 2*pi/B;
binIdx = mod(floor((ang + binW/2)/binW), B) + 1;

hc = floor(H/cs);  wc = floor(W/cs);
HOF = zeros(hc, wc, B);
for i = 1:hc
    rows = (i-1)*cs+1 : i*cs;
    for j = 1:wc
        cols = (j-1)*cs+1 : j*cs;
        b = binIdx(rows, cols);  m = mag(rows, cols);
        HOF(i,j,:) = accumarray(b(:), m(:), [B 1]);
    end
end
hofGlobal  = squeeze(sum(sum(HOF,1),2));
hofGlobalN = hofGlobal / sum(hofGlobal);
binDeg     = (0:B-1)*360/B;

figure('Name','HOF features','Position',[100 100 1100 420]);
subplot(1,3,1);
bar(binDeg, hofGlobalN);
xticks(binDeg); grid on;
xlabel('flow direction (deg; 0 = right, 90 = down)');
ylabel('normalised weighted count');
title('Global HOF (magnitude-weighted)');

subplot(1,3,2);
polarhistogram('BinEdges', ((0:B)-0.5)*binW, 'BinCounts', hofGlobalN');
set(gca, 'ThetaDir','clockwise');
title('Global HOF (polar, image orientation)');

[~, domCell] = max(HOF, [], 3);
E = sum(HOF, 3);
subplot(1,3,3);
imagesc(E); axis image; colormap(gca, hot); cb = colorbar;
cb.Label.String = 'cell motion energy \Sigma HOF';
xlabel('cell column'); ylabel('cell row');
title(sprintf('Per-cell HOF energy (%dx%d px cells)', cs, cs));

for bb = 1:B
    fprintf('HOF bin %3d deg : %.3f\n', binDeg(bb), hofGlobalN(bb));
end

%% part c: Vehicle-pixel extraction using HOF 

[~, domGlob] = max(hofGlobal);
dirDist = min(mod(domCell - domGlob, B), mod(domGlob - domCell, B));

vMin = 0.3;
cellMask = (E > vMin*cs^2) & (dirDist <= 1);

mask = logical(kron(double(cellMask), ones(cs)));
mask = padarray(mask, [H W] - size(mask), false, 'post');
mask = mask & (mag > vMin);
mask = imclose(mask, strel('disk', 2));
mask = imfill(mask, 'holes');

Dt = -bwdist(~mask);
Dt = imimposemin(Dt, imextendedmin(Dt, 2));
mask(watershed(Dt) == 0) = false;
mask = bwareaopen(mask, 30, 4);

vehOnly = F1 .* uint8(repmat(mask, [1 1 3]));

figure('Name','(c) HOF vehicle extraction','Position',[100 100 1100 380]);
subplot(1,3,1); imshow(F1);       title(sprintf('Frame %d', k));
subplot(1,3,2); imshow(mask);     title('HOF vehicle mask');
subplot(1,3,3); imshow(vehOnly);  title('Vehicle pixels only (rest discarded)');

fprintf('Dominant traffic direction: %d deg (image coords)\n', binDeg(domGlob));
fprintf('Vehicle pixels kept: %d of %d (%.1f%%)\n', nnz(mask), H*W, 100*nnz(mask)/(H*W));

%% part d: 2D shape and centroid of each vehicle
L     = bwlabel(mask, 4);
stats = regionprops(L, 'Area','Centroid','BoundingBox','Orientation', ...
                       'MajorAxisLength','MinorAxisLength','ConvexHull');
nVeh  = numel(stats);

c0   = [(W+1)/2, (H+1)/2];
HFOV = 60;
fpx  = (W/2) / tand(HFOV/2);

res = zeros(nVeh, 10);
for n = 1:nVeh
    s  = stats(n);
    cx = s.Centroid(1);  cy = s.Centroid(2);
    sp = mean(mag(L == n));
    res(n,:) = [n, s.Area, cx, cy, cx/W, cy/H, cx - c0(1), cy - c0(2), ...
                atand((cx - c0(1))/fpx), sp*fps];
end
T = array2table(res, 'VariableNames', {'ID','Area_px','Cx_px','Cy_px', ...
     'Cx_norm','Cy_norm','dX_from_center','dY_from_center', ...
     'Bearing_deg','Speed_px_per_s'});
disp(T);

figure('Name','(d) Vehicle shapes and centroids');
imshow(F1); hold on;
t = linspace(0, 2*pi, 60);
hE = []; hH = []; hC = [];
for n = 1:nVeh
    s  = stats(n);
    cx = s.Centroid(1);  cy = s.Centroid(2);
    a  = s.MajorAxisLength/2;  b = s.MinorAxisLength/2;
    th = -deg2rad(s.Orientation);
    ex = cx + a*cos(t)*cos(th) - b*sin(t)*sin(th);
    ey = cy + a*cos(t)*sin(th) + b*sin(t)*cos(th);
    hE = plot(ex, ey, 'g-', 'LineWidth', 1.5);
    hH = plot(s.ConvexHull(:,1), s.ConvexHull(:,2), 'c--');
    rectangle('Position', s.BoundingBox, 'EdgeColor', 'y');
    hC = plot(cx, cy, 'r+', 'MarkerSize', 10, 'LineWidth', 2);
    if s.BoundingBox(2) > 8
        ty = s.BoundingBox(2) - 3;
    else 
        ty = s.BoundingBox(2) + s.BoundingBox(4) + 5;
    end
    text(s.BoundingBox(1), ty, sprintf('V%d', n), 'Color','y', 'FontWeight','bold');
end
hF = plot(c0(1), c0(2), 'wx', 'MarkerSize', 12, 'LineWidth', 2);
line([c0(1) c0(1)], [1 H], 'Color','w', 'LineStyle',':');
line([1 W], [c0(2) c0(2)], 'Color','w', 'LineStyle',':');
title(sprintf('Frame %d: %d vehicle(s) detected (box = yellow)', k, nVeh));
if nVeh > 0
    legend([hE hH hC hF], {'fitted ellipse','convex hull','centroid','FOV centre'}, ...
           'Location','southoutside', 'Orientation','horizontal');
end

%% Single-scale Horn-Schunck 

function [u, v] = hsFlow(A, B, alpha, nIter)

kx = 0.25*[-1 1; -1 1];
ky = 0.25*[-1 -1; 1 1];
kt = 0.25*ones(2);
Ix = imfilter(A,kx,'replicate') + imfilter(B,kx,'replicate');
Iy = imfilter(A,ky,'replicate') + imfilter(B,ky,'replicate');
It = imfilter(B,kt,'replicate') - imfilter(A,kt,'replicate');

kAvg = [1 2 1; 2 0 2; 1 2 1]/12;
u = zeros(size(A));  v = zeros(size(A));
for it = 1:nIter
    uAvg = imfilter(u, kAvg, 'replicate');
    vAvg = imfilter(v, kAvg, 'replicate');
    P    = (Ix.*uAvg + Iy.*vAvg + It) ./ (alpha^2 + Ix.^2 + Iy.^2);
    u    = uAvg - Ix.*P;
    v    = vAvg - Iy.*P;
end
end