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

addpath('networks')
