function fn = get_cnn_normalize(normalization, colour)
% fn = get_cnn_normalize(net.normalization, colour) - returns a normalize func
%     Inputs: normalization - The normalization information of a network's input
%             color - (option) whether the image is colour or not.
%     Outputs - fn - the function that can be used as fn(img) to normalize img

if(nargin < 2)
  colour = true;
end
fn = @(x) cnn_normalize(normalization, x, colour) ;
end

