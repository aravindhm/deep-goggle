function experiment_shallow()
% Run some CNN experiments

experiment_setup;

images = {'data/hog/img.png'};

% -------------------------------------------------------------------------
%                                                         Setup experiments
% -------------------------------------------------------------------------

exp = {} ;
ver = 'results' ;
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
  exp{end+1} = experiment_init('hog', inf, images{i}, ver, 'hog', opts1);
  exp{end+1} = experiment_init('hog', inf, images{i}, ver, 'hog2', opts2);
  exp{end+1} = experiment_init('hog', inf, images{i}, ver, 'hog3', opts3);

  exp{end+1} = experiment_init('hog', inf, images{i}, ver, 'hoggle', opts1, 'useHoggle', true);
  exp{end+1} = experiment_init('dsift', inf, images{i}, ver, 'dsift', opts1);
  exp{end+1} = experiment_init('hogb', inf, images{i}, ver, 'hogb', opts1);
end

experiment_run(exp) ;


% -------------------------------------------------------------------------
%                                                        Accumulate results
% -------------------------------------------------------------------------
img_invhog = imread('data/results/hog/img/lInf-recon.png');
img_invhog2 = imread('data/results/hog2/img/lInf-recon.png');
img_invhog3 = imread('data/results/hog3/img/lInf-recon.png');

img_invdsift = imread('data/results/dsift/img/lInf-recon.png');
img_invhogb = imread('data/results/hogb/img/lInf-recon.png');
img_hoggle = imread('data/results/hoggle/img/lInf-recon.png');

figure;
subplot(2, 2, 1);
imshow(img_invhog);
title('HOG Pre-Image [Our approach]');
subplot(2, 2, 2);
imshow(img_invdsift);
title('DSIFT Pre-Image [Our approach]');
subplot(2, 2, 3);
imshow(img_invhogb);
title('HOGb Pre-Image [Our approach]');
subplot(2, 2, 4);
imshow(img_hoggle);
title('HOG pre-image using HOGle');
drawnow; 

figure;
subplot(1,3,1);
imshow(img_invhog);
subplot(1,3,2);
imshow(img_invhog2);
title('Figure 4. Effect of v^{\beta} regularization.');
subplot(1,3,3);
imshow(img_invhog3);
