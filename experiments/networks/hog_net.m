function net = hog_net(binSize, varargin)
% HOG_NET   CNN-HOG
%   NET = HOG_NET(BINSIZE) returns a CNN equivalent to the HOG feature
%   extractor for the specified bin size.
%
%   The implementation is numerically identical to VL_HOG(), which in turns
%   is nearly exactly the same as UoCTTI HOG implementation (DPM V5).

opts.bilinearOrientations = false ;
opts.numOrientations = 9 ;
opts = vl_argparse(opts, varargin) ;
opts.binSize = binSize ;

% Spatial derivatives along O directions
NO = opts.numOrientations ;
dx = [0 0 0 ; -1 0 1 ; 0  0 0]/2 ;
dy = dx' ;
for i=0:2*NO-1
  t = (2*pi)/(2*NO)*i ;
  spatialDer{i+1} = cos(t)*dx+sin(t)*dy;
end
spatialDer = single(cat(4, spatialDer{:})) ;

% Spatial bilienar binning
a = 1 - abs(((1:2*opts.binSize) - (2*opts.binSize+1)/2)/opts.binSize);
bilinearFilter = repmat(single(a'*a), [1, 1, 1, 2*NO]) ;

% Stacking of HOG cells into 2x2 blocks.
% A detail ritical for what comes later: cells are stacked in column-major
% order.
mask = {} ;
t = 0 ;
for j=1:2
  for i=1:2
    for o=1:2*NO
      t=t+1 ;
      mask{t} = zeros(2,2,2*NO) ;
      mask{t}(i,j,o) = 1 ;
    end
    for o=2*NO+(1:NO)
      t=t+1 ;
      mask{t} = zeros(2,2,2*NO) ;
      mask{t}(i,j,o-2*NO) = 1 ;
      mask{t}(i,j,o-NO) = 1 ;
    end
  end
end
mask = single(cat(4, mask{:})) ;

% Break down blocks to cells. This uses once more 2x2 filters.
% Note how these filters are related to the one that build the blocks
% above. The application of such a decompositon filter involves four
% blocks which share a central cell. This cell is the bottom-right
% one with respect to the first upper-left block. This affects which
% dimensions of the block must be extracted by the decomposition filter.
t = 0 ;
for o=1:3*NO
  t=t+1 ;
  dec{t} = zeros(2,2,3*NO*4) ;
  for i=1:2
    for j=1:2
      u=3-i ;
      v=3-j ;
      dec{t}(i,j,(u-1+(v-1)*2)*3*NO+o) = 0.5 ;
    end
  end
end
dec = single(cat(4, dec{:})) ;

% Network
net.layers = {} ;
net.layers{end+1} = struct('type','conv', ...
  'filters', spatialDer, ...
  'biases', zeros(size(spatialDer,4),1,'single'), ...
  'stride', 1, 'pad', 1) ;
if 0
  net.layers{end+1} = struct('type','noffset', 'param', [1/3*cos((2*pi)/(2*NO)), .5]) ;
  net.layers{end+1} = struct('type','relu') ;
else
  net.layers{end+1} = get_hog_binning_layer(opts) ;
end
net.layers{end+1} = struct('type','conv', ...
  'filters', bilinearFilter, ...
  'biases', [], ...
  'stride', binSize, 'pad', binSize/2) ;
net.layers{end+1} = struct('type','conv', ...
  'filters', mask, ...
  'biases', [], ...
  'stride', 1, 'pad', 0) ;
net.layers{end+1} = get_hog_norm_layer(opts) ;
net.layers{end+1} = get_hog_clamp_layer(opts) ;
net.layers{end+1} = struct('type','conv', ...
  'filters', dec, ...
  'biases', [], ...
  'stride', 1, 'pad', 1) ;

% -------------------------------------------------------------------------
%                                                             Sanity checks
% -------------------------------------------------------------------------
if 0
  rng(0) ;
  % the norm of the gradient derivatives is sqrt(NO) times the norm of the
  % gradient
  for t = 1:100
    x = randn(3) ;
    g = reshape(spatialDer, [], 2*NO)'*x(:) ;
    gx = (x(2,3) - x(2,1))/2 ;
    gy = (x(3,2) - x(1,2))/2 ;
    a(t) = norm([gx gy]);
    b(t) = norm([g]) ; %twice the norm of a
  end
  vl_testsim(mean(b./a), sqrt(NO)) ;

  % test HOG clamping layer
  res.x = round(100*randn(20,20,5,3,'single'))/100 ;
  res_ = hog_clamp_forward([],res,struct('x',[])) ;
  res_.dzdx = randn(size(res_.x),'single') ;
  res = hog_clamp_backward([],res,res_) ;

  vl_testder(@(x) subsref(...
    hog_clamp_forward([],struct('x',x)),...
    struct('type','.','subs','x')), ...
    res.x, ...
    res_.dzdx, ...
    res.dzdx, ...
    1e-4) ;

  % test HOG normalization layer
  ly = get_hog_norm_layer(opts);
  res.x = randn(5,5,3*NO*4,2,'double') ;
  res_ = hog_norm_forward(ly,res,struct('x',[])) ;
  res_.dzdx = randn(size(res_.x),'double') ;
  res = hog_norm_backward(ly,res,res_) ;

  vl_testder(@(x) subsref(...
    hog_norm_forward(ly,struct('x',x)),...
    struct('type','.','subs','x')), ...
    res.x, ...
    res_.dzdx, ...
    res.dzdx, ...
    1e-5) ;

  % test HOG binning layer
  % mini network: derivatives and normalization
  % we do so as the normalization block works properly only on the
  % output of the derivative layer (due to special cases)
  ly = get_hog_binning_layer(opts) ;
  res.x = randn(20,20,1,3,'single') ;
  res_.x = double(vl_nnconv(res.x, spatialDer, [])) ;
  res__ = hog_binning_forward(ly,res_,struct('x',[])) ;
  res__.dzdx = randn(size(res__.x)) ;
  res_ = hog_binning_backward(ly,res_,res__) ;
  res.dzdx = vl_nnconv(res.x, spatialDer, [], single(res_.dzdx)) ;

  vl_testder(@(x) vl_nnconv(x, spatialDer, []), ...
    res.x, ...
    res_.dzdx, ...
    res.dzdx, ...
    1e-2) ;

  vl_testder(@(x) subsref(...
    hog_binning_forward(ly,struct('x',double(vl_nnconv(x, spatialDer, []))),res__),...
    struct('type','.','subs','x')), ...
    res.x, ...
    res__.dzdx, ...
    res.dzdx, ...
    1e-4, 0.5) ;
end

if 0
  im = imread('peppers.png') ;
  im = im(1:100,1:100,:) ;
  im = rgb2gray(im2single(im)) ;
  res = vl_simplenn(net, im) ;
  res_ = vl_simplenn(net, im*255) ;

  L = sqrt(sum(res(2).x.^2,3))/2 ;
  P = bsxfun(@times, 1./max(L,1e-10), res(2).x) ;
  th = real(acos(P)) ;

  figure(100) ; clf ;
  subplot(1,2,1) ; vl_imarraysc(res(4).x) ;
  subplot(1,2,2) ; vl_imarraysc(bsxfun(@times, L, max(0,1-8/(2*pi)*th)));

  res(6).x(1:5,1:5,1) ./ max(res_(6).x(1:5,1:5,1),1e-10)
end

% -------------------------------------------------------------------------
function l = get_hog_norm_layer(opts)
% -------------------------------------------------------------------------
NO = opts.numOrientations ;
sel = [] ;
for i=1:4
  sel = [sel, 2*NO + (1:NO) + 3*NO*(i-1)] ;
end
l.type = 'custom' ;
l.sel = sel ;
l.forward = @hog_norm_forward ;
l.backward = @hog_norm_backward ;

function res_ = hog_norm_forward(ly,res,res_)
n2 = sum(res.x(:,:,ly.sel,:).^2,3) ;
n = sqrt(n2) ;
n = max(n, 1e-6) ;
res_.x = bsxfun(@rdivide, res.x, n) ;

function res = hog_norm_backward(ly,res,res_)
n2 = sum(res.x(:,:,ly.sel,:).^2,3) ;
n = sqrt(n2) ;
n = max(n, 1e-6) ;
tmp = sum(res_.dzdx.*res.x,3) ./ (n.*n2) ;
res.dzdx = bsxfun(@rdivide, res_.dzdx, n) ;
res.dzdx(:,:,ly.sel,:) = res.dzdx(:,:,ly.sel,:) - bsxfun(@times, tmp, res.x(:,:,ly.sel,:)) ;

% -------------------------------------------------------------------------
function l = get_hog_clamp_layer(opts)
% -------------------------------------------------------------------------
l.type = 'custom' ;
l.forward = @hog_clamp_forward ;
l.backward = @hog_clamp_backward ;

function res_ = hog_clamp_forward(ly,res,res_)
res_.x = min(res.x,single(0.2)) ;
%res_.x = res.x ;

function res = hog_clamp_backward(ly,res,res_)
res.dzdx = res_.dzdx .* (res.x <= 0.2) ;

% -------------------------------------------------------------------------
function l = get_hog_binning_layer(opts)
% -------------------------------------------------------------------------
l.type = 'custom' ;
l.NO = opts.numOrientations ;
l.bilinear = opts.bilinearOrientations ;
l.forward = @hog_binning_forward ;
l.backward = @hog_binning_backward ;

function res_ = hog_binning_forward(ly,res,res_)
x = res.x ;
n2 = sum(x.^2,3)/ly.NO ;
n = sqrt(n2) ;
if ly.bilinear
  cs = bsxfun(@rdivide, x, max(n, 1e-10)) ;
  cs = max(min(cs,1),-1) ;
  delta = 1 - (2*ly.NO)/(2*pi)*acos(cs) ;
  w = max(0, delta) ;
else
  w = 0*x ;
  [~,best] = max(x, [], 3) ;
  wx = size(w,2);
  wy = size(w,1);
  wz = size(w,3);
  wn = size(w,4);
  w(kron(ones(1,wn), 1:wx*wy) + ...
    +(best(:)'-1)*(wx*wy)+...
    kron((0:wn-1)*(wx*wy*wz), ones(1,wx*wy))) = 1 ;
end
res_.x = bsxfun(@times, n, w) ;

function res = hog_binning_backward(ly,res,res_)
% forward computations
x = res.x ;
n2 = sum(x.^2,3)/ly.NO ;
n = sqrt(n2) ;
if ly.bilinear
  cs = bsxfun(@rdivide, x, max(n, 1e-10)) ;
  cs = max(min(cs,1),-1) ;
  delta = 1 - (2*ly.NO)/(2*pi)*acos(cs) ;
  w = max(0, delta) ;
else
  w = 0*x ;
  [~,best] = max(x, [], 3) ;
  wx = size(w,2);
  wy = size(w,1);
  wz = size(w,3);
  wn = size(w,4);
  w(kron(ones(1,wn), 1:wx*wy) + ...
    +(best(:)'-1)*(wx*wy)+...
    kron((0:wn-1)*(wx*wy*wz), ones(1,wx*wy))) = 1 ;
end

dn = bsxfun(@rdivide, x/ly.NO, max(n, 1e-10)) ;
if ly.bilinear
  dwdcs = ((2*ly.NO)/(2*pi) ./ sqrt(1.0000001 - cs.^2)) .* (delta > 0) .* res_.dzdx ;
  dw = ...
    + dwdcs .* repmat(1./n, [1 1 2*ly.NO]) ...
    - bsxfun(@times, sum(bsxfun(@rdivide, dwdcs.*x/ly.NO, n.*n2),3), x) ;
  res.dzdx = bsxfun(@times, sum(res_.dzdx .* w, 3), dn) +  bsxfun(@times, n, dw) ;
else
  res.dzdx = bsxfun(@times, sum(res_.dzdx .* w, 3), dn) ;
end
