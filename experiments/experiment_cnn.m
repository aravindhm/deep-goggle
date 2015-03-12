function experiment_cnn()
% Run some CNN experiments

experiment_setup;

images = experiment_get_dataset('imnet');

images_for_spiky = {'data/imagenet12-val/ILSVRC2012_val_00000043.JPEG'};

images_for_TVbeta = {'data/imagenet12-val/ILSVRC2012_val_00000043.JPEG'};

% -------------------------------------------------------------------------
%                                                         Setup experiments
% -------------------------------------------------------------------------

exp = {} ;
ver = 'results' ;
opts.learningRate = 0.004 * [...
  ones(1,100), ...
  0.1 * ones(1,200), ...
  0.01 * ones(1,100)];
opts.objective = 'l2' ;
opts.beta = 6 ;
opts.lambdaTV = 1e2 ;
opts.lambdaL2 = 8e-10 ;

opts1 = opts;
opts1.lambdaTV = 1e0 ;
opts1.TVbeta = 2;

opts2 = opts ;
opts2.lambdaTV = 1e1 ;
opts2.TVbeta = 2;

opts3 = opts ;
opts3.lambdaTV = 1e2 ;
opts3.TVbeta = 2;


% -------------------------------------------------------------------------
%                                                           Run experiments
% -------------------------------------------------------------------------

if 1
  for i = 1:numel(images) 
    for layer = 1:6 % Different regularizer values for each of the layers
      exp{end+1} = experiment_init('caffe-ref', layer, images{i}, ver, 'cnn', opts1) ;
    end
    for layer=7:12
      exp{end+1} = experiment_init('caffe-ref', layer, images{i}, ver, 'cnn', opts2);
    end
    for layer=13:20
      exp{end+1} = experiment_init('caffe-ref', layer, images{i}, ver, 'cnn', opts3);
    end
  end
end

if 1
  opts4 = opts ;
  opts4.lambdaTV = 1e2 ;
  opts4.TVbeta = 1;

  opts5 = opts ;
  opts5.lambdaTV = 1e2 ;
  opts5.TVbeta = 2;
  for i = 1:numel(images_for_spiky)
    exp{end+1} = experiment_init('caffe-ref', 4, images_for_spiky{i}, ver, 'cnn_spiky1', opts4);
    exp{end+1} = experiment_init('caffe-ref', 4, images_for_spiky{i}, ver, 'cnn_spiky2', opts5);
  end
end

if 1
  for i = 1:numel(images_for_TVbeta)
    exp{end+1} = experiment_init('caffe-ref', 20, images_for_TVbeta{i}, ver, 'cnn_tvbeta1', opts1);
    exp{end+1} = experiment_init('caffe-ref', 20, images_for_TVbeta{i}, ver, 'cnn_tvbeta2', opts2);
    % opts 3 has already run above
  end
end

if 1
  for i = 1:numel(images_for_neigh)
    for l = 1:20
      exp{end+1} = experiment_init('caffe-ref', l, images_for_neigh, ver, 'cnn_neigh', opts3, 'neigh', 5);
    end
  end
end

experiment_run(exp) ;



% -------------------------------------------------------------------------
%                                                        Accumulate results
% -------------------------------------------------------------------------
figure;
title('Figure 2. Left: V^{\beta} norm with \beta = 1. \nRight: V^{\beta} norm with \beta = 2. \nThe latter reduces the spiking artifact.');
subplot(1,2,1);
imshow('data/results/cnn_spiky1/ILSVRC2012_val_00000043/l04-recon.png');
subplot(1,2,2);
imshow('data/results/cnn_spiky2/ILSVRC2012_val_00000043/l04-recon.png');

figure;
title('Figure 8. Effect of V^\beta regularization on CNNs');
subplot(1,3,1);
imshow('data/results/cnn_tvbeta1/ILSVRC2012_val_00000043/l20-recon.png');
subplot(1,3,2);
imshow('data/results/cnn_tvbeta2/ILSVRC2012_val_00000043/l20-recon.png');
subplot(1,3,3);
imshow('data/results/cnn/ILSVRC2012_val_00000043/l20-recon.png');

figure;
title('Figure 6. CNN reconstruction. Reconstruction of the image of Fig. 5.a from each layer of CNN-A. \nTo generate these results, the regularization coefficient for each layer is chosen to match the highlighted rows in table 3. \nThis figure is best viewed in color/screen.');
for l=1:20
  subplot(2,10,l);
  imshow(sprintf('data/results/cnn/ILSVRC2012_val_00000013/l%02d-recon.png', l));
end

figure;
%neigh stuff here
