Yianni Laloudakis (jlalouda)
CS 229

None of this code will run out the box because the it is missing dependencies, data files, etc., but here is a short description of the important files used in my project:

collect_hotspot_features.py:
I had to do a map-reduce like procedure to get my integrated gradients features across many gpus so this file
combines the individual outputs into the final csv files.

hotspot_mutate.py:
Short script for alanine mutations using PyMol.

inspect.py:
    extract_conv_layers gets 3D embeddings from SASNet
    create_mutations gets the integrated gradients/delta surface prob features from the sasnet scores

hotspot_neural_net.py
trains and tests the neural net on the convolutional features

mut_predict.py
trains and tests the logistic regression, random forest, and svm models.

salVis.py
creates the PyMol integrated gradients visualization used in the paper.