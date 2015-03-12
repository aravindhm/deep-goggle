Directory Structure
-------------------

+-- root 
|   +-- core 
|   |   +-- invert_nn.m - The core optimization lies here
|   +-- helpers - Several auxilary functions that may be useful in general
|   +-- experiments - All the code to replicate our experiments
|   |   +-- networks 
|   |   |   +-- hog_net.m - The hog and hogb networks are created using this
|   |   |   +-- dsift_net.m - The dense sift neural network is here
|   |   |    Other networks used in our experiments can be downloaded from http://robots.ox.ac.uk/~aravindh/networks.html[1]
|   |   +-- ihog - either copy or soft link ihog from Vondrick et. al. This is required to run our experiments with hoggle.
|   |   +-- matconvnet - either copy or soft link matconvnet code here. If this is not here, then the setup function will not work.
|   |   +-- vlfeat - again either copy or soft copy. If this is not here, then the setup function will not work.

 

Experiments from the paper
--------------------------

To run the experiments used for our publication and replicate their results
please follow the instructions below

Get the images
Download/soft link the imagenet validation images into experiments/data/imagenet12-val
Download/soft link the stock abstrack images into experiments/data/stock

For any of the cases below you need to run the following in matlab
    cd experiments;
    experiment_setup;

I) Experiment for a single reconstruction across all layers of 
    experiment_cnn;
See the results in data/results/ #TODO - List of the files relevant here

#TODO - Add experiment_xxx files here as and when they are documented.


Setting up and running your own networks
----------------------------------------

1. Create a network (net) that is compatible with matconvnet vl_simplenn function. 
2. Run dg_setup.m in matlab
3. Run the network forward to generate a target reference representation y0
4. Call res = invert_nn(net, y0, [options]);
5. res.output\{end\} is the required reconstruction. 
