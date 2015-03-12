function images = experiment_get_dataset(varargin)
% images = experiment_get_dataset(varargin) - Collect all the images
%   from one of more datasets
% Eg: images = experiment_get_dataset('stock') - to get the stock images
% Use any subset of 
%  'copydays', 'stock', 'imnet', 'imnet-lite', 'imnet-nogreen', 'imnet-large', 'hoggle'

if nargin == 0
  subsets = {'stock', 'imnet', 'hoggle'} ;
else
  subsets = varargin ;
end

images = {} ;
for i=1:numel(subsets)

  switch subsets{i}

    case 'copydays'
      % copydays dataset
      tmp = dir('/users/aravindh/work/datasets/copydays/*.jpg') ;
      copydays = cellfun(@(x) fullfile('/users/aravindh/work/datasets/copydays/', x), {tmp(:).name}, 'uniform', false);
      images = horzcat(images, copydays) ; 

    case 'stock'
      % stock images
      tmp = vertcat(...
        dir('data/stock_images/*.jpg'), ...
        dir('data/stock_images/*.png')) ;
      stock = cellfun(@(x) fullfile('data/stock_images', x), {tmp(~[tmp.isdir]).name}, 'uniform', false);
      images = horzcat(images, stock) ;

    case 'imnet'
      % image net images. This is a large scale experiment.
      tmp = dir('data/imagenet12-val/*.JPEG') ;
      imnet = cellfun(@(x) fullfile('data/imagenet12-val/', x), {tmp(1:100).name}, 'uniform', false);
      imnet_large = cellfun(@(x) fullfile('data/imagenet12-val/', x), {tmp(101:500).name}, 'uniform', false);
      images = horzcat(images, imnet) ;

    case 'imnet-lite'
      % image net images tiny subset
      tmp = dir('data/imagenet12-val/*.JPEG') ;
      imnet = cellfun(@(x) fullfile('data/imagenet12-val/', x), {tmp(1:100).name}, 'uniform', false);
      imnet_large = cellfun(@(x) fullfile('data/imagenet12-val/', x), {tmp(101:500).name}, 'uniform', false);
      images = horzcat(images, imnet([13 24 43 70])) ;% [1 2 5 8 17 18 30 34 48 56]

    case 'imnet-nogreen'
      % image net images which are not green. To see if the greenish output 
      % is a property of the prior or the network.
      tmp = dir('data/imagenet12-val/ILSVRC2012_val_000000*.JPEG') ;
      imnet = cellfun(@(x) fullfile('data/imagenet12-val/', x), {tmp(1:99).name}, 'uniform', false);
      images = horzcat(images, imnet( [1 2 5 17 18 30 34 48 56 70] ));

    case 'imnet-large'
      % image net images
      tmp = dir('data/imagenet12-val/*.JPEG') ;
      imnet_large = cellfun(@(x) fullfile('data/imagenet12-val/', x), {tmp(101:500).name}, 'uniform', false);
      images = horzcat(images, imnet_large);

    case 'hoggle'
      % hoggle images
      hoggle{1} = 'data/hoggle/hoggle-orig-1.jpg' ;
      hoggle{2} = 'data/hoggle/hoggle-orig-2.jpg' ;
      images = horzcat(images, hoggle) ;

  end
end

