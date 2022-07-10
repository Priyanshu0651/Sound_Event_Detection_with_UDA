# Semi-supervised Sound Event Detection with Consistency Training by Unsupervised Data Augmentation #
This is a project done in Advanced Topics in Machine Learning Course(EE698R) at IITK. 

# Dataset: 
* Adopted data from the DCASE 2022 challenge consisting of 2045 synthetic(frame level information), 1336 weekly labelled(clip level information) and 6752 unlabelled audio files of length 10 seconds each.

# Approach:
* Extracted Log Mel Spectrogram (Nfft window length = 40 ms & hop length = 20 ms) from audio files and used it as input to the model.
* Implemented Convolutional Recurrent Neural Netowork model to output frame level(inputSize,501,10) and clip level(inputSize,10) prediction probabilities for 10 classes.
* Added linear softmax layer to extract clip level probabilities from frame level probabilities and did clip smoothing to remove noise in the output.

# Result:
* Got best performance using frequency mask augmentation with a Macro Event Based F1 of 49.10 and a Micro Event Based F1 of 44.24.
