from utils import loadbdd100k
from pathlib import Path

class Configuration(object):


    # Neural network architecture options 
    # How to specify architecture: (order of numbers in string, separated by "_" as A_B_C_...)
    # A: use_maxpool = #1 or 0
    # B: n_conv
    # C: n_dense
    # D: channels out of first conv. layer
    # E: zsize, dimension of latent vector
    # F: filter size 
    # G: stride 
    # H: pad

    dataset = "prosivic"
    experiment_name = "debug"
    log_dir = './log/' + dataset + '/' + experiment_name + '/'

    # Diagnostics
    sample_size = 16
    sample_rows = 4 # nrow to save_image grid

    # Testing options
    nd_original_GPND = False
    percentages = [50] # percentage of outliers to use during testing
    auroc = True
    auprc = True
    plot_roc = True
    plot_prc = True

    # dataset specific options below
    if dataset == "prosivic":
        # Hyperparameters
        betas = (0.5,0.999) # adam solver standard is (0.5, 0.999), GPND standard is (0.9,0.999)
#        learning_rate = 0.001
        n_train_epochs = 1000
        n_epochs_between_lr_change = n_train_epochs+1
        num_sample_epochs = 10
        lr_g  = 0.002
        lr_e  = 0.002
        lr_d  = 0.002
        lr_ge = 0.002
        lr_zd = 0.002
        rec_loss_weight = 10
        weight_g_loss = 0.1
        architecture = "0_3_0_64_256_4_2_1"
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

        experiment_name = "debug"
        batch_size = 10
        use_batchnorm = True
        data_div = 140
        n_train = 7000 // data_div
        n_val = 1413 // data_div
        n_test = 787 // data_div # for GPND algorithm, the test set is split into val and test set during testing, since the valset contains outliers in order to compute an optimal threshold. This is used to compute some of the output values, but not AUPRIN or AUROC, which are threshold independent.
        n_test_in = 787 // data_div

        img_folder =   "../weather_detection_data/prosivic/"
        train_folder = "../weather_detection_data/prosivic/train/"
        val_folder =   "../weather_detection_data/prosivic/val/"
        test_in_folder =  "../weather_detection_data/prosivic/test/in/"
        test_out_folder =  "../weather_detection_data/prosivic/test/out/"


    elif dataset == "dreyeve":

        architecture = "0_6_0_16_256_4_2_1"
        inliers_name = "sunny_highway"
        outliers_name = "rainy_highway"
        n_dense_units = [256]
        zd_n_layers = 3
        zd_out_units = [256,256,1]

        image_height = 256
        image_width = 256
        channels = 3
        model_name = "_".join([inliers_name, architecture])

        use_batchnorm = True

        # Hyperparameters
        betas = (0.9,0.999) # adam solver standard is (0.5, 0.999), GPND standard is (0.9,0.999)
        learning_rate = 0.0001
        n_train_epochs = 500
        n_epochs_between_lr_change = n_train_epochs+1
        num_sample_epochs = 5
        batch_size = 16
        lr_g = 0.002
        lr_e = 0.002
        lr_d = 0.002
        lr_ge = 0.002
        lr_zd = 0.002
        rec_loss_weight = 10

        # Dataset options
        data_div = 12
        n_train = 6000 // data_div
        n_val = 600 // data_div
        n_test = 1200 // data_div # for GPND algorithm, the test set is split into val and test set during testing, since the valset contains outli$
        n_test_in = 600 // data_div

        img_folder =   "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/"
        train_folder = "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/train/"
        val_folder =   "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/val/"
        test_in_folder =  "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/test/in/"
        test_out_folder = "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/test/out/foggy"

    
    elif dataset == "mnist":
        batch_size = 128
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
