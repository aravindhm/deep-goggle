Directory Structure
-------------------

root 

 -> core 

    -> invert_nn.m - The core optimization lies here

 -> helpers - Several auxilary functions that may be useful in general

 -> experiments - All the code to replicate our experiments

 -> experiments/networks 

    -> hog_net.m - The hog and hogb networks are created using this

    -> dsift_net.m - The dense sift neural network is here

    Other networks used in our experiments can be downloaded from

    http://robots.ox.ac.uk/~aravindh/networks.html

 -> experiments/ihog - either copy or soft link ihog from Vondrick et. al.

    This is required to run our experiments with hoggle.

 -> matconvnet - either copy or soft link matconvnet code here.

    If this is not here, then the setup function will not work.

 -> vlfeat - again either copy or soft copy

    If this is not here, then the setup function will not work.

 

Experiments from the paper
--------------------------

To run the experiments used for our publication and replicate their results
please follow the instructions below

I) Download and copy the following network model into experiments/networks
imagenet-caffe-ref.mat

II) Get the images
Download/soft link the imagenet validation images into experiments/data/imagenet12-val
Download/soft link the stock abstrack images into experiments/data/stock

For any of the cases below you need to run the following in matlab
>> cd experiments;
>> experiment_setup;

III) Experiment for a single reconstruction across all layers of 
>> experiment_cnn;
>> ls data/results/ #TODO

IV) Experiment for multiple reconstructions of the flamingo
>> experiment_multiple_flamingo;

V) Experiment for showing locality even within the receiptive field
>> experiment_neigh_monkey

VI) Experiment for dissociation of properties across neuron groups in the alexnet
>> experiment_groups_stock

VII) Experiment for variation in TV\beta norm
>> experiment_tvbeta_variation

VIII) Experiment for variation in L\beta norm
>> experiment_lbeta_variation

IX) Experiment with HOG, HOGb, DSIFT and Hoggle
>> experiment_shallow_representations
