function net = dsift_net(binSize, varargin)
% Define a CNN equivalent to dense SIFT

opts.numOrientations = 4 ;
opts = vl_argparse(opts, varargin) ;
opts.binSize = binSize ;

% Spatial derivatives along NO directions
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

% Stacking of SIFT cells into 4x4 blocks.

sigma = 1.5 ;
mask = {} ;
t = 0 ;
for i=1:4
  for j=1:4
    for o=1:2*NO
      t=t+1 ;
      mask{t} = zeros(4,4,2*NO) ;
      mask{t}(i,j,o) = exp(-0.5*((i-2.5).^2 + (j-2.5).^2) / sigma^2) ;
    end
  end
end
mask = single(cat(4, mask{:})) ;

net.layers = {} ;
net.layers{end+1} = struct('type','conv', ...
  'filters', spatialDer, ...
  'biases', zeros(size(spatialDer,4),1,'single'), ...
  'stride', 1, 'pad', 0) ;
if 0
  net.layers{end+1} = struct('type','noffset', 'param', [.5*cos(2*pi/8), .5]) ;
  net.layers{end+1} = struct('type','relu') ;
else
  net.layers{end+1} = get_hog_binning_layer(opts) ;
end
net.layers{end+1} = struct('type','conv', ...
  'filters', bilinearFilter, ...
  'biases', [], ...
  'stride', binSize, 'pad', 0) ;
net.layers{end+1} = struct('type','conv', ...
  'filters', mask, ...
  'biases', [], ...
  'stride', 3, 'pad', 0) ;
net.layers{end+1} = struct('type', 'normalize', 'param', [128*2, 0.000001, 1, .5]) ;
net.layers{end+1} = get_hog_clamp_layer(opts) ;

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
l.forward = @hog_binning_forward ;
l.backward = @hog_binning_backward ;

function res_ = hog_binning_forward(ly,res,res_)
x = res.x ;
n2 = sum(x.^2,3)/ly.NO ;
n = sqrt(n2) ;
cs = bsxfun(@rdivide, x, max(n, 1e-10)) ;
cs = max(min(cs,1),-1) ;
delta = 1 - (2*ly.NO)/(2*pi)*acos(cs) ;
w = max(0, delta) ;
res_.x = bsxfun(@times, n, w) ;

function res = hog_binning_backward(ly,res,res_)
% forward computations
x = res.x ;
n2 = sum(x.^2,3)/ly.NO ;
n = sqrt(n2) ;
cs = bsxfun(@rdivide, x, max(n, 1e-10)) ;
cs = max(min(cs,1),-1) ;
delta = 1 - (2*ly.NO)/(2*pi)*acos(cs) ;
w = max(0, delta) ;

dn = bsxfun(@rdivide, x/ly.NO, max(n, 1e-10)) ;
dwdcs = ((2*ly.NO)/(2*pi) ./ sqrt(1.0000001 - cs.^2)) .* (delta > 0) .* res_.dzdx ;
dw = ...
  + dwdcs .* repmat(1./n, [1 1 2*ly.NO]) ...
  - bsxfun(@times, sum(bsxfun(@rdivide, dwdcs.*x/ly.NO, n.*n2),3), x) ;
res.dzdx = bsxfun(@times, sum(res_.dzdx .* w, 3), dn) +  bsxfun(@times, n, dw) ;
