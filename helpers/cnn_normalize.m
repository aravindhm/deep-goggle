function im_ = cnn_normalize(normalization, im, colour)
% im_ = cnn_normalize(normalization, im_, colour) - Normalize an image

if ~colour && size(im, 3)==3, im=rgb2gray(im); end
im_ = single(im) ; % note: 255 range
if colour && size(im_,3)==1, im_=repmat(im_,[1 1 3]) ; end
im_ = imresize(im_, normalization.imageSize(1:2)) ;
im_ = im_ - normalization.averageImage ;
end

