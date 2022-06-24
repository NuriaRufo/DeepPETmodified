# DeepPET
DeepPET is a Deep Learning method to reconstruct positron emission tomography (PET) images from raw data, organized in sinograms, to a high quality final image where the noise is greatly reduced. This repository contains all the tools necessary to work with an adaptation of the method. 


![Architecture](https://user-images.githubusercontent.com/108093731/175489229-76cfc677-2399-4ca9-9372-343cdb255dd2.png)

It is organized in two parts:

 - The first one consists of a series of directories where the images used are stored in NumPy format of size N x 128 x 128 x 1 (N, number of images). Here you can        differentiate between the  datasets you want to use and within them you can find training, validation and test set. In addition, the different trained models can      also be stored here.
 - In the second part, we find all the code necessary to perform the tests. This can be divided into three stages:
          - Generate_input.py will load the user's images to use. You can also choose if you want to perform data augmentation to the set by                                       rotation, rotation and random zoom. Afterwards, the corresponding sinograms will be generated from these images, which will be the input to the network.               Finally, the datasets will be saved in the directory section and it is also given the possibility to save the data in batches for a later use of a batch               generator that feeds the network dynamically and volatilely for lower GPU memory footprint.
          
          - Train.ipby will be the notebook where, as its name indicates, the training is carried out. To do this, images from the indicated directory are loaded and               Poisson noise is added to them so that the network learns to handle noisy sinograms. Then, it is necessary to indicate the hyperparameters for the training             and very important if you want to use the batch generator to feed the network. Once the model is trained, it is saved in the indicated directory and its               performance is evaluated.

          - Finally, an inference can be made between the reconstruction results of the model and the traditional methods, also implemented. In this notebook, the                 average errors and the success rate of the test set of all methods will be displayed and a report will be created and saved in the indicated folder.


