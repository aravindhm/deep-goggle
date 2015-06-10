function experiment_shallow_quantitative()
% Run some CNN experiments

experiment_setup;

images = experiment_get_dataset('imnet');
%images = {images{1:10}};

% -------------------------------------------------------------------------
%                                                         Setup experiments
% -------------------------------------------------------------------------

exp = {} ;
res_ver = 'results' ;
opts.learningRate = 0.004 * [...
  ones(1,200), ...
  0.1 * ones(1,200), ...
  0.01 * ones(1,200),...
  0.001 * ones(1,200)];
opts.objective = 'l2' ;
opts.beta = 6 ;
opts.lambdaTV = 1e2 ;
opts.lambdaL2 = 8e-10 ;
opts.numRepeats = 1;

opts1 = opts;
opts1.lambdaTV = 1e0 ;
opts1.TVbeta = 2;

opts2 = opts;
opts2.lambdaTV = 1e1 ;
opts2.TVbeta = 2;

opts3 = opts;
opts3.lambdaTV = 1e2 ;
opts3.TVbeta = 2;

% -------------------------------------------------------------------------
%                                                           Run experiments
% -------------------------------------------------------------------------

for i = 1:numel(images) 
  exp{end+1} = experiment_init('hog', inf, images{i}, res_ver, 'hog', opts1);
  
  exp{end+1} = experiment_init('hog', inf, images{i}, res_ver, 'hoggle', opts1, 'useHoggle', true);
  exp{end+1} = experiment_init('dsift', inf, images{i}, res_ver, 'dsift', opts1);
  exp{end+1} = experiment_init('hogb', inf, images{i}, res_ver, 'hogb', opts1);
end

experiment_run(exp) ;


% -------------------------------------------------------------------------
%                                                        Accumulate results
% -------------------------------------------------------------------------

    
[hog_error_mean, hog_error_std] = eval_recons('hog', images, res_ver)
[hogb_error_mean, hogb_error_std] = eval_recons('hogb', images, res_ver)
[hoggle_error_mean, hoggle_error_std] = eval_recons('hoggle', images, res_ver)
[dsift_error_mean, dsift_error_std] = eval_recons('dsift', images, res_ver)

end

function [error_mean, error_std] = eval_recons(prefix, images, res_ver) 
  size_hog = 0;
  for i=1:numel(images)
    [~, image_filename, ~] = fileparts(images{i});
    res_filename = fullfile('data', res_ver, prefix, image_filename, 'lInf.mat');
    res = load(res_filename);
    size_hog = max(size_hog, numel(res.y));
  end

  y_hog = zeros(size_hog, numel(images));
  y0_hog = zeros(size_hog, numel(images));

  for i=1:numel(images)
    [~, image_filename, ~] = fileparts(images{i});
    res_filename = fullfile('data', res_ver, prefix, image_filename, 'lInf.mat');
    res = load(res_filename);
    
    y_hog(1:numel(res.y),i) = res.y(:);
    y0_hog(1:numel(res.y),i) = res.y0(:);
  end
  
  [error_mean, error_std] = my_meanstd(y_hog, y0_hog);
end

function [error_mean, error_std] = my_meanstd(y, y0)
  temp = pdist2(y0', y0');
  normalization_factor = mean(temp(:).^2);

  temp = sum((y - y0).^2) / normalization_factor;
  error_mean = mean(temp);
  error_std = std(temp);
end
