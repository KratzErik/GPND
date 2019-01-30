# GPND
Re-implementation of https://github.com/podgorskiy/GPND.

## How to setup an experiment
The experiment settings are defined in ./configuration.py

### Change configuration with implemented dataset
Choose which dataset you want to run an experiment with by changing the ```dataset``` string variable. Settings are kept separately for each dataset, under corresponding if-statements for the ```dataset``` variable.

### Add a new dataset
* Put your data in separate directories:
    * training data (inliers)
    * validation data (inliers)
    * test data inliers
    * test data outliers
* Do the following in ```configuration.py```:
    * Create a new ```if dataset == "dataset_name" block``` in configuration.py. Copy one of the existing ones so as to not miss any required settings. 
    * Specify the directories in which data is kept in variables ```train_folder```, etc.
    * Make sure number of samples for each set is correct, ```num_train```, etc.
    * Make sure image format corresponds to the image dimensions in your dataset, ```image_height```, ```image_width``` and ```channels```
    * Specify autoencoder and discriminator architectures, see ```net.py``` for further details.

## How to train a novelty detection model
* Make sure you have the settings you want in ```configuration.py```
* To train a model with a custom dataset, or the already setup ```prosivic``` and ```dreyeve``` datasets, run the following from the repository main directory:
```
python nd_train.py
```

This will call the function ```main()``` in ```train_AAE.py```. This function takes some arguments which are used in the original GPND implementation, but not in this version. They are left as to not cause bugs.

The script will train an adversarial autoencoder model on the data in your specified train-directory and store it as a .pkl file in ```log/dataset_name/experiment_name/train/```

## How to test a novelty detection model
* Make sure you have the settings you want in ```configuration.py```
* Make sure you have trained a model, see above.
* Run the following from the repository main directory:
```
python nd_test.py
```

This will call the function ```main()``` from ```novelty_detector.py```. If you leave the configuration settings ```nd_original_GPND = False``` the novelty detection results for all samples in your test data directories will be stored as a .pkl-file in ```./log/dataset_name/experiment_name/test/```.

If the results corresponding to the current settings already exist, the results will be read from ```./log/dataset_name/experiment_name/test/``` instead of running the test again. This is since for large images, the testing takes a long time due to a singular value decomposition in ```novelty_detector.py```.

If you also have ```export_results = True``` in ```configuration.py```, arrays of scores and test labels (1 for outliers/novelties and 0 for inliers/normal samples) will be saved to the directory specified as ```export_results_dir``` in ```configuration.py```.




