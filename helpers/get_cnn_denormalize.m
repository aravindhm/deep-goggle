function fn = get_cnn_denormalize(normalization)
% fn = get_cnn_denormalize(net.normalization) - returns a function handle
%   that can denormalize normalized network inputs
%
% Return value: fn - a function handle called as x_denormalized=fn(x_normalized)
fn = @(x) cnn_denormalize(normalization, x) ;

end
