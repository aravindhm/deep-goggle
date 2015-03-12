function im_ = cnn_denormalize(normalization, im)
% im_ = cnn_denormalize(net.normalization, im)  - denormalize an input
%    normalized image using the normalization info in the 'normalization'
%    structure

im_ = bsxfun(@plus, im, normalization.averageImage) ;

end

