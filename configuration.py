from utils import loadbdd100k
from pathlib import Path

class Configuration(object):

    dataset = "dreyeve"
#    architecture = '0_5_1_8_256_5_2_0' # with dense layer, stride instead of pool
#    architecture = '0_5_0_8_256_5_2_0' # no dense layer, stride instead of pool
#    architecture = '1_5_1_8_256_5_1_0' # with dense layer, maxpool
#    architecture = '1_5_0_8_256_5_1_0' # no dense layer, maxpool
#    architecture = "0_4_0_8_256_4_2_1"
    architecture = "0_5_0_8_256_5_2_2"
#    architecture = "b2"


    betas = (0.5,0.999)
    learning_rate = 0.002
    n_train_epochs = 100
    n_epochs_between_lr_change = 30
    batch_size = 10

    dreyeve_img_folder =   "./dreyeve/highway_morning_sunny_vs_rainy/"
    dreyeve_train_folder = "./dreyeve/highway_morning_sunny_vs_rainy/train/"
    dreyeve_val_folder =   "./dreyeve/highway_morning_sunny_vs_rainy/val/"
    dreyeve_test_folder =  "./dreyeve/highway_morning_sunny_vs_rainy/test/"
    dreyeve_n_train = 100
    dreyeve_n_val = 50
    dreyeve_n_test = 100
    dreyeve_n_test_in = 50
# put config variable definitions here
#    img_folder = Path("/data/bdd100k/images/train_and_val_192by320")
#    norm_file = "/data/bdd100k/namelists/clear_or_partly_cloudy_or_overcast_and_highway_and_daytime.txt"
#    norm_filenames = loadbdd100k.get_namelist_from_file(norm_file)
#    out_file = "/data/bdd100k/namelists/rainy_or_snowy_or_foggy_and_highway_and_daytime_or_dawndusk_or_night.txt"
#    out_filenames = loadbdd100k.get_namelist_from_file(out_file)
#    norm_spec = [["weather", ["clear","partly cloudy", "overcast"]],["scene", "highway"],["timeofday", "daytime"]]
#    out_spec = [["weather", ["rainy", "snowy", "foggy"]],["scene", "highway"],["timeofday",["daytime","dawn/dusk","night"]]]
    n_train = 100
    n_val = 0
    n_test = 100
    out_frac = 0.5
    image_height = 256
    image_width = 256
    channels = 3
#    save_name_lists=False
#    labels_file = None
#    get_norm_and_out_sets = False
#    shuffle=False
    experiment_name = "autobuild"+architecture
    use_batchnorm = True
    log_dir = './log/' + dataset + '/' + experiment_name + '/'
