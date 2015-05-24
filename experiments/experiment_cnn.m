function experiment_cnn()
% Run some CNN experiments

experiment_setup;

%images = experiment_get_dataset('imnet');
images_for_progression = {'data/imagenet12-val/ILSVRC2012_val_00000013.JPEG'};

images_for_spiky = {'data/imagenet12-val/ILSVRC2012_val_00000043.JPEG'};

images_for_TVbeta = {'data/imagenet12-val/ILSVRC2012_val_00000043.JPEG'};

images_for_neigh = {'data/imagenet12-val/ILSVRC2012_val_00000013.JPEG'};

images_for_diversity = {'data/imagenet12-val/ILSVRC2012_val_00000011.JPEG',...
     'data/imagenet12-val/ILSVRC2012_val_00000014.JPEG',...
     'data/imagenet12-val/ILSVRC2012_val_00000018.JPEG',...
     'data/imagenet12-val/ILSVRC2012_val_00000023.JPEG',...
     'data/imagenet12-val/ILSVRC2012_val_00000033.JPEG'};

images_for_multiple = {'data/imagenet12-val/ILSVRC2012_val_00000043.JPEG',...
     'data/stock/stock_abstract.jpg'};

images_for_groups = {'data/stock/stock_abstract.jpg', ...
  'data/stock/stock_fish.jpg'};

images_for_teaser = {'data/imagenet12-val/ILSVRC2012_val_00000024.JPEG'};

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

if 1 % Figure 6
  for i = 1:numel(images_for_progression) 
    for layer = 1:6 % Different regularizer values for each of the layers
      exp{end+1} = experiment_init('caffe-ref', layer, images_for_progression{i}, ver, 'cnn', opts1) ;
    end
    for layer=7:12
      exp{end+1} = experiment_init('caffe-ref', layer, images_for_progression{i}, ver, 'cnn', opts2);
    end
    for layer=13:20
      exp{end+1} = experiment_init('caffe-ref', layer, images_for_progression{i}, ver, 'cnn', opts3);
    end
  end
end

if 1 % Figure 2
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

if 1 % Figure 8
  for i = 1:numel(images_for_TVbeta)
    exp{end+1} = experiment_init('caffe-ref', 20, images_for_TVbeta{i}, ver, 'cnn_tvbeta1', opts1);
    exp{end+1} = experiment_init('caffe-ref', 20, images_for_TVbeta{i}, ver, 'cnn_tvbeta2', opts2);
    exp{end+1} = experiment_init('caffe-ref', 20, images_for_TVbeta{i}, ver, 'cnn_tvbeta3', opts3);
  end
end

if 1 % Figure 9
  opts7 = opts3;
  opts7.learningRate = 0.004 * [...
    0.1* ones(1,200), ...
    0.01 * ones(1,200), ...
    0.001 * ones(1,200)];
  for i = 1:numel(images_for_neigh)
    for l = 1:15
      exp{end+1} = experiment_init('caffe-ref', l, images_for_neigh{i}, ver, 'cnn_neigh', opts7, 'neigh', 5);
    end
  end
end

if 1 % Figure 11
  for i = 1:numel(images_for_diversity)
    exp{end+1} = experiment_init('caffe-ref', 15, images_for_diversity{i}, ver, 'cnn_diversity', opts3);
  end
end

if 1 % Figure 7
  opts6_3 = opts3;
  opts6_3.numRepeats = 6;

  opts6_2 = opts2;
  opts6_2.numRepeats = 6;

  opts6_1 = opts1;
  opts6_1.numRepeats = 6;

  for i = 1:numel(images_for_multiple)
    for l=[15]
      exp{end+1} = experiment_init('caffe-ref', l, images_for_multiple{i}, ver, 'cnn_multiple', opts6_1);
    end
    for l=[17]
      exp{end+1} = experiment_init('caffe-ref', l, images_for_multiple{i}, ver, 'cnn_multiple', opts6_2);
    end
    for l=[19,20]
      exp{end+1} = experiment_init('caffe-ref', l, images_for_multiple{i}, ver, 'cnn_multiple', opts6_3);
    end
  end
end

if 1 % Figure 10
  for i = 1:numel(images_for_groups)
    for l=[1,4,8]
      exp{end+1} = experiment_init('caffe-ref', l, images_for_groups{i}, ver, 'cnn_group1', opts1, 'filterGroup', 1);
      exp{end+1} = experiment_init('caffe-ref', l, images_for_groups{i}, ver, 'cnn_group2', opts1, 'filterGroup', 2);
    end
  end
end

if 1 % Figure 1
  opts10 = opts3;
  opts10.numRepeats = 6;
  for i = 1:numel(images_for_teaser)
    l = 20;
    exp{end+1} = experiment_init('caffe-ref', l, images_for_teaser{i}, ver, 'cnn_teaser', opts10);
  end
end

experiment_run(exp) ;



% -------------------------------------------------------------------------
%                                                        Accumulate results
% -------------------------------------------------------------------------
figure;
subplot(1,2,1);
imshow('data/results/cnn_spiky1/ILSVRC2012_val_00000043/l04-recon.png');
title('Figure 2. Effect of \beta. \beta = 1');
subplot(1,2,2);
imshow('data/results/cnn_spiky2/ILSVRC2012_val_00000043/l04-recon.png');
title('Figure 2. Effect of \beta. \beta = 2');

figure;
subplot(1,3,1);
imshow('data/results/cnn_tvbeta1/ILSVRC2012_val_00000043/l20-recon.png');
subplot(1,3,2);
imshow('data/results/cnn_tvbeta2/ILSVRC2012_val_00000043/l20-recon.png');
title('Figure 8. Effect of V^\beta regularization on CNNs');
subplot(1,3,3);
imshow('data/results/cnn_tvbeta3/ILSVRC2012_val_00000043/l20-recon.png');

figure;
for l=1:20
  subplot(4,5,l);
  imshow(sprintf('data/results/cnn/ILSVRC2012_val_00000013/l%02d-recon.png', l));
end
subplot(4,5,3);
title('Figure 6. CNN reconstruction.');

figure;
for l=1:15
  subplot(3,5,l);
  imshow(sprintf('data/results/cnn_neigh/ILSVRC2012_val_00000013/l%02d-recon.png', l));
end
subplot(3,5,3);
title('Figure 9. CNN receptive fields.');

figure;
subplot(1,5,1);
imshow('data/results/cnn_diversity/ILSVRC2012_val_00000011/l15-recon.png');
subplot(1,5,2);
imshow('data/results/cnn_diversity/ILSVRC2012_val_00000014/l15-recon.png');
subplot(1,5,3);
imshow('data/results/cnn_diversity/ILSVRC2012_val_00000018/l15-recon.png');
title('Figure 11. Diversity in the CNN model');
subplot(1,5,4);
imshow('data/results/cnn_diversity/ILSVRC2012_val_00000023/l15-recon.png');
subplot(1,5,5);
imshow('data/results/cnn_diversity/ILSVRC2012_val_00000033/l15-recon.png');

figure;
subplot(2,4,1);
imshow('data/results/cnn_multiple/ILSVRC2012_val_00000043/l15-recon.png');
title('Figure 7: CNN invariances. Pool5');
subplot(2,4,2);
imshow('data/results/cnn_multiple/ILSVRC2012_val_00000043/l17-recon.png');
title('Figure 7: CNN invariances. relu6');
subplot(2,4,3);
imshow('data/results/cnn_multiple/ILSVRC2012_val_00000043/l19-recon.png');
title('Figure 7: CNN invariances. relu7');
subplot(2,4,4);
imshow('data/results/cnn_multiple/ILSVRC2012_val_00000043/l20-recon.png');
title('Figure 7: CNN invariances. fc8');

subplot(2,4,5);
imshow('data/results/cnn_multiple/stock_abstract/l15-recon.png');
title('Figure 7: CNN invariances. Pool5');
subplot(2,4,6);
imshow('data/results/cnn_multiple/stock_abstract/l17-recon.png');
title('Figure 7: CNN invariances. relu6');
subplot(2,4,7);
imshow('data/results/cnn_multiple/stock_abstract/l19-recon.png');
title('Figure 7: CNN invariances. relu7');
subplot(2,4,8);
imshow('data/results/cnn_multiple/stock_abstract/l20-recon.png');
title('Figure 7: CNN invariances. fc8');

figure;
subplot(2,6,1);
imshow('data/results/cnn_group1/stock_abstract/l01-recon.png');
subplot(2,6,2);
imshow('data/results/cnn_group1/stock_abstract/l04-recon.png');
subplot(2,6,3);
imshow('data/results/cnn_group1/stock_abstract/l08-recon.png');
title('Figure 10: CNN neural streams');
subplot(2,6,4);
imshow('data/results/cnn_group2/stock_abstract/l01-recon.png');
subplot(2,6,5);
imshow('data/results/cnn_group2/stock_abstract/l04-recon.png');
subplot(2,6,6);
imshow('data/results/cnn_group2/stock_abstract/l08-recon.png');
subplot(2,6,7);
imshow('data/results/cnn_group1/stock_fish/l01-recon.png');
subplot(2,6,8);
imshow('data/results/cnn_group1/stock_fish/l04-recon.png');
subplot(2,6,9);
imshow('data/results/cnn_group1/stock_fish/l08-recon.png');
subplot(2,6,10);
imshow('data/results/cnn_group2/stock_fish/l01-recon.png');
subplot(2,6,11);
imshow('data/results/cnn_group2/stock_fish/l04-recon.png');
subplot(2,6,12);
imshow('data/results/cnn_group2/stock_fish/l08-recon.png');

figure;
imshow('data/results/cnn_teaser/ILSVRC2012_val_00000024/l20-recon.png');
title('Figure 1: What is encoded by a CNN?');
