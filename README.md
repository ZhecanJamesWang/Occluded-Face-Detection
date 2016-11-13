# Occluded-Face-Detection



The code has two parts, the preprocessing and the model.
The preprocessing part basically handles collecting data, resizing the photos, padding them and storing.
The model part does more about the heavy lifting. It implemented a stacked autoencoder and the specifics can be referenced here, http://ufldl.stanford.edu/wiki/index.php/Stacked_Autoencoders.

This implemented model is built based on the platform tensor flow and is modified from this blog http://cmgreen.io/2016/01/04/tensorflow_deep_autoencoder.html along with the github repository, https://github.com/cmgreen210/TensorFlowDeepAutoencoder . 
