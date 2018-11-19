from utils import loadbdd100k
from pathlib import Path

class Configuration(object):


    # Neural network architecture options 
    #architecture = '0_5_1_8_256_5_2_0' # with dense layer, stride instead of pool
    #architecture = '0_5_0_8_256_5_2_0' # no dense layer, stride instead of pool
    #architecture = '1_5_1_8_256_5_1_0' # with dense layer, maxpool
    #architecture = '1_5_0_8_256_5_1_0' # no dense layer, maxpool
    #architecture = "0_4_0_8_256_4_2_1"
    architecture = "1_4_1_16_1024_5_2_2"
    #architecture = "b2"

    # Hyperparameters
    betas = (0.5,0.999) # adam solver standard
    learning_rate = 0.001
    n_train_epochs = 500
    n_epochs_between_lr_change = n_train_epochs+1
    num_sample_epochs = 5
    batch_size = 64

    # Dataset options
    n_train = 7000
    n_val = 1413
    n_test = 787 # for GPND algorithm, the test set is split into val and test set during testing, since the valset contains outliers in order to compute an optimal threshold. This is used to compute some of the output values, but not AUPRIN or AUROC, which are threshold independent.
    n_test_in = 787

    dataset = "prosivic"
    inliers_name = "sunny_highway"
    outliers_name = "rainy_highway"

    image_height = 256
    image_width = 256
    channels = 3
    model_name = "_".join([inliers_name, architecture])
    experiment_name = "_vs_".join([inliers_name,outliers_name])
    use_batchnorm = True
    log_dir = './log/' + dataset + '/' + model_name + '/'

    # Testing options
    percentages = [0.5] # percentage of outliers to use during testing
    auroc = True
    auprc = True
    plot_roc = True
    plot_prc = True

    # dataset specific options below

    if dataset == "dreyeve":
        img_folder =   "../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/"
        train_folder = "../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/train/"
        val_folder =   "../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/val/"
        test_in_folder =  "../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/test/out/"
        test_out_folder =  "../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/test/out/"
    
    elif dataset == "prosivic":
        img_folder =   "../weather_detection_data/prosivic/"
        train_folder = "../weather_detection_data/prosivic/train/"
        val_folder =   "../weather_detection_data/prosivic/val/"
        test_in_folder =  "../weather_detection_data/prosivic/test/in/"
        test_out_folder =  "../weather_detection_data/prosivic/test/out/"


    elif dataset == "bdd100k":
        img_folder = Path("/data/bdd100k/images/train_and_val_256by256")
        norm_file = "/data/bdd100k/namelists/clear_or_partly_cloudy_or_overcast_and_highway_and_daytime.txt"
        norm_filenames = loadbdd100k.get_namelist_from_file(norm_file)
        out_file = "/data/bdd100k/namelists/rainy_or_snowy_or_foggy_and_highway_and_daytime_or_dawndusk_or_night.txt"
        out_filenames = loadbdd100k.get_namelist_from_file(out_file)
        norm_spec = [["weather", ["clear","partly cloudy", "overcast"]],["scene", "highway"],["timeofday", "daytime"]]
        out_spec = [["weather", ["rainy", "snowy", "foggy"]],["scene", "highway"],["timeofday",["daytime","dawn/dusk","night"]]]

    
