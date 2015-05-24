Directory Structure
-------------------
<pre>
.
+-- core
|   +-- invert_nn.m - The core optimization lies here
+-- helpers - Several auxilary functions that may be useful in general
+-- experiments - All the code to replicate our experiments
|   +-- networks
|   |   +-- hog_net.m - The hog and hogb networks are created using this
|   |   +-- dsift_net.m - The dense sift neural network is here
|   |   +-- Other networks used in our experiments can be downloaded from http://www.robots.ox.ac.uk/~aravindh/networks.html
|   +-- data
|   |   +-- hog/img.png - Image used for HOG and DSIFT qualitative results
|   |   +-- stock/ - Contains some more figures for reproducing qualitative results.
+-- ihog - either copy or soft link ihog from Vondrick et. al. This is required to run our experiments with hoggle.
+-- matconvnet - either copy or soft link matconvnet code here. If this is not here, then the setup function will not work.
+-- vlfeat - again either copy or soft copy. If this is not here, then the setup function will not work.
</pre>


Experiments from the paper
--------------------------

To run the experiments used for our publication and replicate their results please follow the instructions below

Get the images

Download/soft link the imagenet validation images into experiments/data/imagenet12-val
Download/soft link the stock abstrack images into experiments/data/stock

Compile ihog, vlfeat and matconvnet as per the instructions given at their respective webpages.

ihog: http://web.mit.edu/vondrick/ihog/

matconvnet: http://www.vlfeat.org/matconvnet/

vlfeat: http://www.vlfeat.org/


I) CNN experiments - qualitative results

    cd experiments;
    experiment_cnn;
    
This might run for several hours and generate a lot of matlab figures. Each figure contains the images used in the paper.

II) HOG, HOGle, DSIFT experiments - qualitative results

    cd experiments;
    experiment_shallow;
    
Same as before, it will generate matlab figures with the required images.

Setting up and running your own networks
----------------------------------------

1. Create a network (net) that is compatible with matconvnet vl_simplenn function.
2. Run dg\_setup.m in matlab
3. Run the network forward to generate a target reference representation y0
4. Call res = invert\_nn(net, y0, \[options\]);
5. res.output\{end\} is the required reconstruction.
