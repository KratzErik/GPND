from utils import loadbdd100k
from pathlib import Path

class Configuration(object):


    # Neural network architecture options 
    #architecture = '0_5_1_8_256_5_2_0' # with dense layer, stride instead of pool
    #architecture = '0_5_0_8_256_5_2_0' # no dense layer, stride instead of pool
    #architecture = '1_5_1_8_256_5_1_0' # with dense layer, maxpool
    #architecture = '1_5_0_8_256_5_1_0' # no dense layer, maxpool
    #architecture = "0_4_0_8_256_4_2_1"
    architecture = "0_5_0_8_1024_5_2_2"
    #architecture = "b2"

    # Hyperparameters
    betas = (0.5,0.999) # adam solver standard
    learning_rate = 0.002
    n_train_epochs = 300
    n_epochs_between_lr_change = 100
    num_sample_epochs = 5
    batch_size = 10

    # Dataset options
    n_train = 100
    n_val = 50
    n_test = 100 # for GPND algorithm, the test set is split into val and test set during testing, since the valset contains outliers in order to compute an optimal threshold. This is used to compute some of the output values, but not AUPRIN or AUROC, which are threshold independent.
    n_test_in = 50

    dataset = "prosivic"

    image_height = 256
    image_width = 256
    channels = 3
    experiment_name = "autobuild"+architecture
    use_batchnorm = True
    log_dir = './log/' + dataset + '/' + experiment_name + '/'



    # dataset specific options below

    if dataset == "dreyeve":
        img_folder =   "../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/"
        train_folder = "../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/train/"
        val_folder =   "../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/val/"
        test_in_folder =  "../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/test/out/"
        test_out_folder =  "../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/test/out/"
    
    if dataset == "prosivic":
        img_folder =   "../weather_detection_data/prosivic/"
        train_folder = "../weather_detection_data/prosivic/train/"
        val_folder =   "../weather_detection_data/prosivic/val/"
        test_in_folder =  "../weather_detection_data/prosivic/test/in/"
        test_out_folder =  "../weather_detection_data/prosivic/test/out/"


    if dataset == "bdd100k":
        img_folder = Path("/data/bdd100k/images/train_and_val_256by256")
        norm_file = "/data/bdd100k/namelists/clear_or_partly_cloudy_or_overcast_and_highway_and_daytime.txt"
        norm_filenames = loadbdd100k.get_namelist_from_file(norm_file)
        out_file = "/data/bdd100k/namelists/rainy_or_snowy_or_foggy_and_highway_and_daytime_or_dawndusk_or_night.txt"
        out_filenames = loadbdd100k.get_namelist_from_file(out_file)
        norm_spec = [["weather", ["clear","partly cloudy", "overcast"]],["scene", "highway"],["timeofday", "daytime"]]
        out_spec = [["weather", ["rainy", "snowy", "foggy"]],["scene", "highway"],["timeofday",["daytime","dawn/dusk","night"]]]

    
