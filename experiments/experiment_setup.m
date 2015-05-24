addpath('../core');
addpath('../helpers');

run ../vlfeat/toolbox/vl_setup ;
run ../matconvnet/matlab/vl_setupnn ;

if(~exist('data', 'dir'))
  mkdir('data');
end

if(~exist('data/results', 'dir'))
  mkdir('data/results');
end

if(~exist('networks/imagenet-caffe-ref.mat', 'file'))
  cmd='wget http://www.robots.ox.ac.uk/~aravindh/imagenet-caffe-ref.mat';
  system(cmd);
  clear cmd;
end

addpath('networks')

