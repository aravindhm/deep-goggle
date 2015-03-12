function feats = compute_features(net, img)
% feats = compute_features(net, img) - compute features from a network
%
%  Inputs: net - the neural network (same as for vl_simplenn)
%          img - the input image (H x W x 3 - RGB image)
%
%  Output: feats - the reference given as argument to invert_nn.m
%
% Author: Aravindh Mahendran
%      New College, University of Oxford

% normalize the input image
normalize = get_cnn_normalize(net.normalization);
x0 = normalize(img);

% Convert the image into a 4D matrix as required by vl_simplenn
x0 = repmat(x0, [1, 1, 1, 1]);

% Run feedforward for network
res = vl_simplenn(net, x0);
feats = res(end).x;

end
