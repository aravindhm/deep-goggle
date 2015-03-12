function exp = experiment_init(model, layer, imageName, prefix, suffix, varargin)
% Initialize an experiment


if nargin < 4
  prefix = '' ;
end

if nargin < 6
  suffix = '' ;
end

[imageDir,imageName,imageExt] = fileparts(imageName) ;
if isempty(imageDir), imageDir = 'data/images' ; end

exp.expDir = fullfile('data', prefix, suffix) ;
exp.model = model ;
exp.layer = layer ;
exp.name = imageName ;
exp.useHoggle = false ;
exp.path = fullfile(imageDir, [imageName, imageExt]) ;
exp.opts.dropout = 0 ;
exp.opts.neigh = +inf ;
exp.opts.filterGroup = NaN ;
exp.opts.objective = 'l2' ;
exp.opts.learningRate = 0.1 * ones(1,100) ;
exp.opts.maxNumIterations = +inf ;
exp.opts.beta = 2 ;
exp.opts.lambdaTV = 100 ;
exp.opts.lambdaL2 = 0.1 ;
exp.opts.TVbeta = 1;
exp.opts.numRepeats = 1 ;
exp.opts.optim_method = 'gradient-descent';

[exp,varargin] = vl_argparse(exp, varargin) ;
exp.opts = vl_argparse(exp.opts, varargin) ;


