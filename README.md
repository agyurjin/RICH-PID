# RICH-PID


## Introduction

Neural network model to run the Ring Image Cherenkov Detector (RICH) Particle Identification (PID) training and inference. Model consist of two parts. First is Convolutional Neural Networkto pass MAPMTs hit information and get output features based on the hits distribution. Second step is concatenate hits features and per track kinematic informations and pass it through the Fully Connected Network to do predictions.

Hit distribution is based on the pmt-anode combination, it should be converted to raw-column values. 

## Input CSV file for training

Each line in the CSV file is track information that should be used for particle identification. All the columns that are in JSON file mention as tracks, hits and out columns must be in the CSV file.

Track based values are either float or integer values.

The hit base information is semicolon separated string. For example 'x1;x2;..xN;'. 

Out column is integer value index of the particle, in the JSON input particle_names. 

## Input JSON

Input JSON file is the main parameters file to use for both training and inference. During the training it will load all the inportant information and at the end of the training it will save in the output folder updated version of the JSON file for inference. Parameters in the json file are:

`learning_rate`: Learning rate for neural network gradient update.

`iterations`: Number of iterations to do during training.

`batch_size`: Number of elements to process during one iteration.

`printout_iter`: Iteration monitoring. Runs model on test set, saves if it is better that previoues best model and prints the results.

`split_ratio`: Train-Test split ratio.

`file_path`: Path to CSV file that will be used for the training.

`output_folder`: Folder name to output results.

`track_cols`: Track based columns to use during training.

`hit_cols`: Hit based columns to use during training. Column names should be ordered -> (x,y,time)

`out_cols`: Prediction column for training.

`particle_names`: Name of the particles.

`model_name`: Neural Network output model name.

`root_name`: Name of the ROOT file to save, when the inference will run.

## Run training

Command to start training with provided input file. 

`python3 run_training.py -i [INPUT-JSON-PATH]`

Code will load all required information from the input and start training after finishing training it will output training model parameters in the output folder. Except from the model, it will update input JSON file with few new parameters.

`date` and `time` to keep track, when the model was trained.
`data_mean` and `data_std` are the parameters for input values normalization.
`test_acc` and `conf_mat_report` are the model training results, accurace on the test set, and precision and recall for each particle.

## Run inference

Command to start inference wit h provided model.

`python3 run_inference.py -i [CSV-PATH] -m [JSON-PATH] -c [CHUNK-SIZE]`

Code will load the JSON file from the trained model folder and look up for trained model output in the same directory. Rest of the information from the trainining is in the JSON file. 

After that code will start to load CSV file. The file can be very big and case memory errors. Due to it loads by chunks. Number of lines per chunk must be passed with [CHUNK-SIZE] integer value. 

Model will do predictions for all tracks in the CSV file and output it into the ROOT file.





