function experiment_run(exp)

if matlabpool('size') > 0
  parfor i=1:numel(exp) % Easily run lots of experiments on a cluster
    run_one(exp{i}) ;
  end
else
  for i=1:numel(exp)
    ts = tic;
    fprintf(1, 'Starting an expeirment');
    run_one(exp{i}) ;
    fprintf(1, 'done an expeirment');
    toc(ts);
  end
end
end

% -------------------------------------------------------------------------
function run_one(exp)
% -------------------------------------------------------------------------

expPath = fullfile(exp.expDir, exp.name) ;
expName = sprintf('l%02d', exp.layer) ;
if ~exist(expPath, 'dir'), mkdir(expPath) ; end
if exist(fullfile(expPath, [expName '.mat'])), return ; end

fprintf('running experiment %s\n', exp.name) ;

% read image
im = imread(exp.path) ;
if size(im,3) == 1, im = cat(3,im,im,im) ; end

if exp.useHoggle
  run_one_hoggle(exp, expPath, expName, im) ;
  return ;
end

switch exp.model
  case 'caffe-ref'
    net = load('networks/imagenet-caffe-ref.mat') ;
    exp.opts.normalize = get_cnn_normalize(net.meta.normalization) ;
    exp.opts.denormalize = get_cnn_denormalize(net.meta.normalization) ;
    exp.opts.imgSize = net.meta.normalization.imageSize;
  case 'caffe-mitplaces'
    net = load('networks/places-caffe-ref-upgraded.mat');
    exp.opts.normalize = get_cnn_normalize(net.meta.normalization) ;
    exp.opts.denormalize = get_cnn_denormalize(net.meta.normalization) ;
    exp.opts.imgSize = net.meta.normalization.imageSize;
  case 'caffe-alex'
    net = load('networks/imagenet-caffe-alex.mat') ;
    exp.opts.normalize = get_cnn_normalize(net.meta.normalization) ;
    exp.opts.denormalize = get_cnn_denormalize(net.meta.normalization) ;
    exp.opts.imgSize = net.meta.normalization.imageSize;
  case 'dsift'
    net = dsift_net(5) ;
    exp.opts.normalize = @(x) 255 * rgb2gray(im2single(x)) - 128 ;
    exp.opts.denormalize = @(x) cat(3,x,x,x) + 128 ;
    exp.opts.imgSize = [size(im, 1), size(im, 2), 1];
  case 'hog'
    net = hog_net(8) 
    exp.opts.normalize = @(x) 255 * rgb2gray(im2single(x)) - 128  ;
    exp.opts.denormalize = @(x) cat(3,x,x,x) + 128 ;
    exp.opts.imgSize = [size(im,1), size(im,2), 1];
  case 'hogb'
    net = hog_net(8, 'bilinearOrientations', true) ;
    exp.opts.normalize = @(x) 255 * rgb2gray(im2single(x)) - 128  ;
    exp.opts.denormalize = @(x) cat(3,x,x,x) + 128 ;
    exp.opts.imgSize = [size(im, 1), size(im, 2), 1];
end
net = vl.simplenn_tidy(net);
if isinf(exp.layer), exp.layer = numel(net.layers) ; end
net.layers = net.layers(1:exp.layer) ;

feats = compute_features(net, im, exp.opts);
input = im;

% run experiment
args = expandOpts(exp.opts) ;
if(strcmp(exp.opts.optim_method , 'gradient-descent'))
  res = invert_nn(net, feats, args{:}) ;
else
  fprintf(1, 'Unknown optimization method %s\n', exp.opts.optim_method);
  return;
end

if isfield(net.layers{end}, 'name')
  res.name = net.layers{end}.name ;
else
  res.name = sprintf('%s%d', net.layers{end}.type, exp.layer) ;
end

% save images
if size(res.output{end},4) > 1
  im = vl_imarraysc(res.output{end}) ;
else
  im = vl_imsc(res.output{end}) ;
end
vl_printsize(1) ;
print('-dpdf',  fullfile(expPath, [expName '-opt.pdf'])) ;
imwrite(input / 255, fullfile(expPath, [expName '-orig.png'])) ;
imwrite(im, fullfile(expPath, [expName '-recon.png'])) ;

% save videos
makeMovieFromCell(fullfile(expPath, [expName '-evolution']), res.output) ;
makeMovieFromArray(fullfile(expPath, [expName '-recon']), res.output{end}) ;

% save results
res.output = res.output{end} ; % too much disk space
save(fullfile(expPath, [expName '.mat']), '-struct', 'res');
end

% -------------------------------------------------------------------------
function run_one_hoggle(exp, expPath, expName, im)
% -------------------------------------------------------------------------

addpath ../ihog
addpath ../ihog/internal
addpath ../ihog/spams
addpath ../ihog/spams/src_release
addpath ../ihog/spams/build

% obtain reconstructions and corresponding HOGs
im = rgb2gray(im2single(im)) ;
hog = features(double(repmat(im,[1 1 3])),8) ;
imrec = repmat(invertHOG(hog), [1 1 3]) ;
hog_ = features(imrec,8) ;

% ignore as we do the texture components in the evaluation
hog = hog(:,:,1:27) ;
hog_ = hog_(:,:,1:27) ;
del = hog_ - hog ;

res.input = 255* im ;
res.output = imrec ;
res.err = norm(del(:))^2 / norm(hog(:))^2 ;
res.y = hog_;
res.y0 = hog;
res.name = 'hog' ;

imhog = imresize(showHOG(hog), [size(res.input,1), size(res.input,2)], 'bicubic') ;

% save images
im = vl_imsc(imrec) ;
vl_printsize(1) ;
imwrite(res.input / 255, fullfile(expPath, [expName '-orig.png'])) ;
imwrite(vl_imsc(imhog), fullfile(expPath, [expName '-hog.png'])) ;
imwrite(im, fullfile(expPath, [expName '-recon.png'])) ;

% save results
save(fullfile(expPath, [expName '.mat']), '-struct', 'res');

end

% -------------------------------------------------------------------------
function args = expandOpts(opts)
% -------------------------------------------------------------------------
args = horzcat(fieldnames(opts), struct2cell(opts))' ;
end

% -------------------------------------------------------------------------
function makeMovieFromCell(moviePath, x)
% -------------------------------------------------------------------------
writerObj = VideoWriter(moviePath,'Motion JPEG AVI');
open(writerObj) ;
for k = 1:numel(x)
  if size(x{k},4) > 1
    im = vl_imarraysc(x{k}) ;
  else
    im = vl_imsc(x{k}) ;
  end
  writeVideo(writerObj,im2frame(double(im)));
end
close(writerObj);
end

% -------------------------------------------------------------------------
function makeMovieFromArray(moviePath, x)
% -------------------------------------------------------------------------
writerObj = VideoWriter(moviePath,'Motion JPEG AVI');
open(writerObj) ;
for k = 1:size(x,4)
  im = vl_imsc(x(:,:,:,k)) ;
  writeVideo(writerObj,im2frame(double(im)));
end
close(writerObj);
end
