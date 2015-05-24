function res = invert_nn(net, ref, varargin)
% INVERT  Invert a CNN representation

opts.learningRate = 0.001*[...
  ones(1,800), ...
  0.1 * ones(1,500), ...
  0.01 * ones(1,500), ...
  0.001 * ones(1,200), ...
  0.0001 * ones(1,100) ] ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.maxNumIterations = numel(opts.learningRate) ;
opts.objective = 'l2' ; % The experiments in the paper use only l2
opts.lambdaTV = 10 ; % Coefficient of the TV^\beta regularizer
opts.lambdaL2 = 08e-10 ; % Coefficient of the L\beta regularizer on the reconstruction
opts.TVbeta = 2; % The power to which TV norm is raized.
opts.beta = 6 ; % The \beta of the L\beta regularizer
opts.momentum = 0.9 ; % Momentum used in the optimization
opts.numRepeats = 4 ; % Number of reconstructions to generate
opts.normalize = [] ; % A function handle that normalizes network input
opts.denormalize = [] ; % A function handle that denormalizes network input
opts.dropout = 0.5; % Dropout rate for any drop out layers.
% The L1 loss puts a drop out layer. We didn't experiment with this though.
opts.filterGroup = NaN ; % Helps select one or the other group of filters.
% This is used in selecting filters in conv 1 of alex net to see the difference
% in their properties.
opts.neigh = +inf ; % To select a small neighborhood of neurons. 
opts.optim_method = 'gradient-descent'; % Only 'gradient-descent' is currently used

opts.imgSize = [];

% Parse the input arguments to override the above defaults
opts = vl_argparse(opts, varargin) ;

% Update the number of iterations using the learning rate if required
if isinf(opts.maxNumIterations)
  opts.maxNumIterations = numel(opts.learningRate) ;
end

% The size of the image that we are trying to obtain
x0_size = cat(2, opts.imgSize, opts.numRepeats);

% x0_sigma is computed using a separate dataset.
% This is a useful normalization that helps scale the different terms in the
% optimization.
load('x0_sigma.mat', 'x0_sigma');

% Replicate the feature into a block. This is used for multiple inversions in parallel.
y0 = repmat(ref, [1 1 1 opts.numRepeats]);

% initial inversion image of size x0_size
x = randn(x0_size, 'single') ;
x = x / norm(x(:)) * x0_sigma  ; 
x_momentum = zeros(x0_size, 'single') ;

% allow reconstructing a subset of the representation by setting
% a suitable mask on the features y0
sf = 1:size(y0,3) ;
if opts.filterGroup == 1
  sf= vl_colsubset(sf, 0.5, 'beginning') ;
elseif opts.filterGroup == 2 ;
  sf= vl_colsubset(sf, 0.5, 'ending') ;
end
nx = min(opts.neigh, size(y0,2)) ;
ny = min(opts.neigh, size(y0,1)) ;
sx = (0:nx-1) + ceil((size(y0,2)-nx+1)/2) ;
sy = (0:ny-1) + ceil((size(y0,1)-ny+1)/2) ;
mask = zeros(size(y0), 'single') ;
mask(sy,sx,sf,:) = 1 ;
y0_sigma = norm(squeeze(y0(find(mask(:))))) ;

%% Tweak the network by adding a reconstruction loss at the end

layer_num = numel(net.layers) ; % The layer number which we are reconstructing
% This is saved here just for printing our progress as optimization proceeds

switch opts.objective
  case 'l2'
    % Add the l2 loss over the network
    ly.type = 'custom' ;
    ly.w = y0 ;
    ly.mask = mask ;
    ly.forward = @nndistance_forward ;
    ly.backward = @nndistance_backward ;
    net.layers{end+1} = ly ;
  case 'l1'
    % The L1 loss might want to use a dropout layer. 
    % This is just a guess and hasn't been tried.
    ly.type = 'dropout' ;
    ly.rate = opts.dropout ;
    net.layers{end+1} = ly ;
    ly.type = 'custom' ;
    ly.w = y0 ;
    ly.mask = mask ;
    ly.forward = @nndistance1_forward ;
    ly.backward = @nndistance1_backward ;
    net.layers{end+1} = ly ;
  case 'inner'
    % The inner product loss may be suitable for some networks
    ly.type = 'custom' ;
    ly.w = - y0 .* mask ;
    ly.forward = @nninner_forward ;
    ly.backward = @nninner_backward ;
    net.layers{end+1} = ly ;
  otherwise
    error('unknown opts.objective') ;
end

%% --------------------------------------------------------------------
%%                                                 Perform optimisation
%% --------------------------------------------------------------------

% Run forward propogation once on the modified network before we
% begin backprop iterations - this is to exploit an optimization in
% vl_simplenn
res = vl_simplenn(net, x); % x is the random initialized image

% recored results
output = {} ;
prevlr = 0 ;

% Iterate until maxNumIterations to optimize the objective
% and generate the reconstuction
for t=1:opts.maxNumIterations

  % Effectively does both forward and backward passes
  res = vl_simplenn(net, x, single(1)) ;

  y = res(end-1).x ; % The current best feature we could generate

  dr = zeros(size(x),'single'); % The derivative

  if opts.lambdaTV > 0 % Cost and derivative for TV\beta norm
    [r_,dr_] = tv(x,opts.TVbeta) ;
    E(2,t) = opts.lambdaTV/2 * r_ ;
    dr = dr + opts.lambdaTV/2 * dr_ ;
  else
    E(2,t) = 0;
  end

  if opts.lambdaL2 > 0 % Cost and derivative of L\beta norm
    r_ = sum(x(:).^opts.beta) ;
    dr_ = opts.beta * x.^(opts.beta-1) ;
    E(3,t) = opts.lambdaL2/2 * r_ ;
    dr = dr + opts.lambdaL2/2 * dr_ ;
  else
    E(3,t) = 0;
  end

  % Rescale the different costs and add them up
  E(1,t) = res(end).x/(y0_sigma^2);
  E(2:3,t) = E(2:3,t) / (x0_sigma^2) ;
  E(4,t) = sum(E(1:3,t)) ;
  fprintf('iter:%05d sq. rec. err:%8.4g; obj:%8.4g;\n', t, E(1,end), E(4,end)) ;

  lr = opts.learningRate(min(t, numel(opts.learningRate))) ;

  % when the learning rate changes suddently, it is not
  % possible for the gradient to crrect the momentum properly
  % causing the algorithm to overshoot for several iterations
  if lr ~= prevlr
    fprintf('switching learning rate (%f to %f) and resetting momentum\n', ...
      prevlr, lr) ;
    x_momentum = 0 * x_momentum ;
    prevlr = lr ;
  end

  % x_momentum combines the current gradient and the previous gradients
  % with decay (opts.momentum) 
  x_momentum = opts.momentum * x_momentum ...
      - lr * dr ...
      - (lr * x0_sigma^2/y0_sigma^2) * res(1).dzdx;

  % This is the main update step (we are updating the the variable
  % along the gradient
  x = x + x_momentum ;

  %% -----------------------------------------------------------------------
  %% Plots - Generate several plots to keep track of our progress
  %% -----------------------------------------------------------------------

  if mod(t-1,25)==0
    output{end+1} = opts.denormalize(x) ;

    figure(1) ; clf ;

    subplot(3,2,[1 3]) ;
    if opts.numRepeats > 1
      vl_imarraysc(output{end}) ;
    else
      imagesc(vl_imsc(output{end})) ;
    end
    axis image ; colormap gray ;

    subplot(3,2,2) ;
    len = min(1000, numel(y0));
    a = squeeze(y0(1:len)) ;
    b = squeeze(y(1:len)) ;
    plot(1:len,a,'b'); hold on ;
    plot(len+1:2*len,abs(b-a), 'r');
    legend('\Phi_0', '|\Phi-\Phi_0|') ;
    title(sprintf('reconstructed layer %d %s', ...
      layer_num, ...
      net.layers{layer_num}.type)) ;
    legend('ref', 'delta') ;

    subplot(3,2,4) ;
    hist(x(:),100) ;
    grid on ;
    title('histogram of x') ;

    subplot(3,2,5) ;
    plot(E') ;
    h = legend('recon', 'tv_reg', 'l2_reg', 'tot') ;
    set(h,'color','none') ; grid on ;
    title(sprintf('iter:%d \\lambda_{tv}:%g \\lambda_{l2}:%g rate:%g obj:%s', ...
                  t, opts.lambdaTV, opts.lambdaL2, lr, opts.objective)) ;

    subplot(3,2,6) ;
    semilogy(E') ;
    title('log scale') ;
    grid on ;
    drawnow ;

  end % end if(mod(t-1,25) == 0)
end % end loop over maxNumIterations


% Compute the features optained using feedforward on the computed inverse
res_nn = vl_simplenn(net, x);

clear res;
res.input = NaN;
res.output = output ;
res.energy = E ;
res.y0 = y0 ;
res.y = res_nn(end-1).x ;
res.opts = opts ;
res.err = res_nn(end).x / y0_sigma^2 ;

% --------------------------------------------------------------------
function res_ = nndistance_forward(ly, res, res_)
% --------------------------------------------------------------------
res_.x = nndistance(res.x, ly.w, ly.mask) ;

% --------------------------------------------------------------------
function res = nndistance_backward(ly, res, res_)
% --------------------------------------------------------------------
res.dzdx = nndistance(res.x, ly.w, ly.mask, res_.dzdx) ;

% --------------------------------------------------------------------
function y = nndistance(x,w,mask,dzdy)
% --------------------------------------------------------------------
if nargin <= 3
  d = x - w ;
  y = sum(sum(sum(sum(d.*d.*mask)))) ;
else
  y = dzdy * 2 * (x - w) .* mask ;
end

% --------------------------------------------------------------------
function res_ = l1loss_forward(ly, res, res_)
% --------------------------------------------------------------------
res_.x = ly.w*sum(abs(res.x(:)));


% --------------------------------------------------------------------
function res = l1loss_backward(ly, res, res_)
% --------------------------------------------------------------------
res.dzdx = zeros(size(res.x), 'single');
res.dzdx(res.x > 0) = single(ly.w)*res_.dzdx;
res.dzdx(res.x < 0) = -single(ly.w)*res_.dzdx;
res.dzdx(res.x == 0) = single(0);

% --------------------------------------------------------------------
function res_ = nndistance1_forward(ly, res, res_)
% --------------------------------------------------------------------
res_.x = nndistance1(res.x, ly.w, ly.mask) ;

% --------------------------------------------------------------------
function res = nndistance1_backward(ly, res, res_)
% --------------------------------------------------------------------
res.dzdx = nndistance1(res.x, ly.w, ly.mask, res_.dzdx) ;

% --------------------------------------------------------------------
function y = nndistance1(x,w,mask,dzdy)
% --------------------------------------------------------------------
if nargin <= 3
  d = x - w ;
  y = sum(sum(sum(sum(abs(d).*mask)))) ;
else
  y = dzdy * sign(x - w) .* mask ;
end

% --------------------------------------------------------------------
function res_ = nninner_forward(ly, res, res_)
% --------------------------------------------------------------------
res_.x = nninner(res.x, ly.w) ;

% --------------------------------------------------------------------
function res = nninner_backward(ly, res, res_)
% --------------------------------------------------------------------
res.dzdx = nninner(res.x, ly.w, res_.dzdx) ;

% --------------------------------------------------------------------
function y = nninner(x,w,dzdy)
% --------------------------------------------------------------------
if nargin <= 2
  y = sum(sum(sum(sum(w.*x)))) ;
else
  y = dzdy * w ;
end

% --------------------------------------------------------------------
function [e, dx] = tv(x,beta)
% --------------------------------------------------------------------
if(~exist('beta', 'var'))
  beta = 1; % the power to which the TV norm is raized
end
d1 = x(:,[2:end end],:,:) - x ;
d2 = x([2:end end],:,:,:) - x ;
v = sqrt(d1.*d1 + d2.*d2).^beta ;
e = sum(sum(sum(sum(v)))) ;
if nargout > 1
  d1_ = (max(v, 1e-5).^(2*(beta/2-1)/beta)) .* d1;
  d2_ = (max(v, 1e-5).^(2*(beta/2-1)/beta)) .* d2;
  d11 = d1_(:,[1 1:end-1],:,:) - d1_ ;
  d22 = d2_([1 1:end-1],:,:,:) - d2_ ;
  d11(:,1,:,:) = - d1_(:,1,:,:) ;
  d22(1,:,:,:) = - d2_(1,:,:,:) ;
  dx = beta*(d11 + d22);
  if(any(isnan(dx)))
  end
end

% --------------------------------------------------------------------
function test_tv()
% --------------------------------------------------------------------
x = randn(5,6,1,1) ;
[e,dr] = tv(x,6) ;
vl_testder(@(x) tv(x,6), x, 1, dr, 1e-3) ;
