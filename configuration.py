from utils import loadbdd100k
from pathlib import Path

# In this file, settings for experiments are defined. If you don't want to change the
# algorithm it self, but only run experiments with defined or new datasets, you should only 
# need to edit this file

# Neural network architecture options 
    # How to specify architecture: (order of numbers in string, separated by "_" as A_B_C_...)
    # A: use_maxpool = #1 or 0. (If A=0, G should be > 1, so that dim. reduction is done with stride instead of pooling)
    # B: n_conv
    # C: use dense layer, set to 1 or 0
    # D: number of filters of first conv. layer (will be doubled in each layer)
    # E: zsize, dimension of latent vector
    # F: filter size in conv layers
    # G: stride in conv layers
    # H: zero padding in conv layers

    # Example: architecture = "0_5_1_16_5_2_2" is used for prosivic experiments in the default settings


class Configuration(object):

    dataset = "prosivic"
    experiment_name = "timetest"
    log_dir = './log/' + dataset + '/' + experiment_name + '/'
    export_results = True
    export_results_dir = '/home/exjobb_resultat/data/'
    test_name = "urban" # If you want to run several test sets for same trained model, specify a test name to store results separately


    # Diagnostics
    sample_size = 16
    sample_rows = 4 # nrow to save_image grid

    # Train options
    loss = "bce"
    weight_init = ("normal", "xavier_normal", "xavier_uniform")[2]

    # Testing options
    nd_original_GPND = False
    percentages = [50] # percentage of outliers to use during testing
    auroc = True
    auprc = True
    plot_roc = True
    plot_prc = True

    # dataset specific options below
    if dataset == "prosivic":
        training_mode = ("autoencoder","GPND_default")[1] # Options: "autoencoder", "GPND_default"
        # Hyperparameters
        betas = (0.5,0.999) # adam solver standard is (0.5, 0.999), GPND standard is (0.9,0.999)
        n_train_epochs = 500
        n_epochs_between_lr_change = int(n_train_epochs * 1/2)
        lr_drop_factor = 10
        num_sample_epochs = 10
        lr_g  = 0.001
        lr_e  = 0.001
        lr_d  = 0.001
        lr_ge = 0.001
        lr_zd = 0.001
        rec_loss_weight = 1
        weight_g_loss = 1
        #architecture = "0_5_1_16_512_4_2_1"
        architecture = "0_5_1_16_512_5_2_2"
        n_dense_units = None
        inliers_name = "sunny"
        outliers_name = "foggy"
        zd_n_layers = 3
        zd_out_units = [256,256,1]

        # Dataset options
        image_height = 256
        image_width = 256
        channels = 3
        model_name = "_".join([inliers_name, architecture])

        test_batch_size = 16 # Jacobian computations require smaller batches
        use_batchnorm = True
        data_div = 1
        n_train = 6785 // data_div
        n_val = 840 // data_div
        n_test = 500*2 // data_div
        n_test_in = 500 // data_div
        batch_size = 64

        img_folder =   "../weather_detection_data/prosivic/"
        train_folder = "../weather_detection_data/prosivic/train/"
        val_folder =   "../weather_detection_data/prosivic/val/"
        test_in_folder =  "../weather_detection_data/prosivic/test/in/"
        test_out_folder =  "../weather_detection_data/prosivic/test/out/urban/"

        if "urban" in test_out_folder:
            n_test = 488*2 // data_div 
            n_test_in = 488 // data_div

    elif dataset == "dreyeve":
        training_mode = ("autoencoder","GPND_default")[1]
        architecture = "0_6_1_16_512_5_2_2"
        inliers_name = "sunny_highway"
        outliers_name = "rainy_highway"
        n_dense_units = None
        zd_n_layers = 3
        zd_out_units = [256,256,1]

        image_height = 256
        image_width = 256
        channels = 3
        model_name = "_".join([inliers_name, architecture])

        use_batchnorm = True

        # Hyperparameters
        betas = (0.5,0.999) # adam solver standard is (0.5, 0.999), GPND standard is (0.9,0.999)
        n_train_epochs = 500
        n_epochs_between_lr_change = int(n_train_epochs * 1/2)
        lr_drop_factor = 10
        num_sample_epochs = 10
        batch_size = 64
        test_batch_size = 8 # Jacobian computations require smaller batches
        lr_g = 0.001
        lr_e = 0.001
        lr_d = 0.001
        lr_ge = 0.001
        lr_zd = 0.001
        rec_loss_weight = 1
        weight_g_loss = 1

        # Dataset options
        data_div = 1
        n_train = 6000 // data_div
        n_val = 600 // data_div
        n_test = 1200 // data_div # for GPND algorithm, the test set is split into val and test set during testing, since the valset contains outli$
        n_test_in = 600 // data_div

        img_folder =   "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/"
        train_folder = "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/train/"
        val_folder =   "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/val/"
        test_in_folder =  "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/test/in/"
        test_out_folder = "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/test/out/foggy/"

    elif dataset == "mnist":
        batch_size = 128
        test_batch_size = batch_size # Jacobian computations require smaller batches
        architecture = None
        betas = (0.9,0.999) # GPND standard
        n_train_epochs = 100
        n_epochs_between_lr_change = 40
        num_sample_epochs = 5
        lr_g = 0.002
        lr_e = 0.002
        lr_d = 0.002
        lr_ge = 0.002
        lr_zd = 0.002
        rec_loss_weight = 1

    elif dataset == "bdd100k":
        img_folder = Path("/data/bdd100k/images/train_and_val_256by256")
        norm_file = "/data/bdd100k/namelists/clear_or_partly_cloudy_or_overcast_and_highway_and_daytime.txt"
        norm_filenames = loadbdd100k.get_namelist_from_file(norm_file)
        out_file = "/data/bdd100k/namelists/rainy_or_snowy_or_foggy_and_highway_and_daytime_or_dawndusk_or_night.txt"
        out_filenames = loadbdd100k.get_namelist_from_file(out_file)
        norm_spec = [["weather", ["clear","partly cloudy", "overcast"]],["scene", "highway"],["timeofday", "daytime"]]
        out_spec = [["weather", ["rainy", "snowy", "foggy"]],["scene", "highway"],["timeofday",["daytime","dawn/dusk","night"]]]
