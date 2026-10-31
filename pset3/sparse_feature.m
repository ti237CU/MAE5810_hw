%% MAE 5810 - HW3, Problem 3(c): stop-sign detection and classification

clear; close all; clc;

imgFiles = {'stopSignTest.jpg', 'stopSign2.jpg'};

hueRed    = [0.06 0.90];
sMin      = 0.20;          
vMin      = 0.20;
minArea   = 30;            % ignore tiny red things
aspectRng = [0.75 1.33];
extentRng = [0.68 0.92];   % hull area / bounding-box area
redRng    = [0.45 0.92];   % red fraction inside the hull
whiteMin  = 0.10;          % white letter fraction inside the hull

figure('Name','Problem 3(c) stop-sign detection','Position',[100 100 1300 480]);
for n = 1:numel(imgFiles)
    img = imread(findImage(imgFiles{n}));
    hsv = rgb2hsv(img);
    H = hsv(:,:,1);  S = hsv(:,:,2);  V = hsv(:,:,3);

    % Detect possible red regions
    red   = (H < hueRed(1) | H > hueRed(2)) & S > sMin & V > vMin;
    red   = imopen(red, strel('disk', 1));         % remove isolated pixels
    light = V > 0.55 & S < 0.35;                   % white / pale pixels
    st    = regionprops(red, 'Area', 'BoundingBox', 'ConvexImage');

    % Classification, shape + appearance test for each candidate
    boxes = zeros(0,4);  info = {};
    fprintf('\n%s  (%d red candidates)\n', imgFiles{n}, numel(st));
    for i = 1:numel(st)
        if st(i).Area < minArea, continue; end
        bb = st(i).BoundingBox;                    % [x y w h], half-pixel offsets
        c0 = ceil(bb(1));  r0 = ceil(bb(2));  w = bb(3);  h = bb(4);
        hull = false(size(red));
        hull(r0:r0+h-1, c0:c0+w-1) = st(i).ConvexImage;
        hA = nnz(hull);

        aspect = w / h;
        extent = hA / (w*h);
        redF   = nnz(red   & hull) / hA;
        whiteF = nnz(light & hull) / hA;
        isStop = aspect >= aspectRng(1) && aspect <= aspectRng(2) && ...
                 extent >= extentRng(1) && extent <= extentRng(2) && ...
                 redF   >= redRng(1)    && redF   <= redRng(2)    && ...
                 whiteF >= whiteMin;

        if isStop || hA > 150
            fprintf(['  box [%3d %3d %3d %3d]  aspect %.2f  extent %.2f  ' ...
                     'red %.2f  white %.2f  -> %s\n'], c0, r0, w, h, ...
                     aspect, extent, redF, whiteF, string(isStop));
        end
        if isStop
            boxes(end+1,:) = [c0 r0 w h];
            info{end+1}    = 'stop sign';
        end
    end

    % Output annotated image with the result
    out = img;
    if ~isempty(boxes)
        out = insertObjectAnnotation(img, 'rectangle', boxes, info, ...
                                     'Color','yellow', 'FontSize',12, 'LineWidth',2);
    end
    subplot(1, numel(imgFiles), n);
    imshow(out);
    title(sprintf('%s: %d stop sign(s) detected', imgFiles{n}, size(boxes,1)), ...
          'Interpreter','none');
end

%% Local function: find an image on the path or in the toolbox data folders
function p = findImage(name)
if exist(name, 'file')
    p = name;  return;
end
cand = dir(fullfile(matlabroot, 'toolbox', '**', name));
if isempty(cand)
    error('Cannot find %s. Put it in the current folder.', name);
end
p = fullfile(cand(1).folder, cand(1).name);
end