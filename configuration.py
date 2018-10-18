class Configuration(object):

# put config variable definitions here
    img_folder = "/data/bdd100k/images/192by320"
    norm_filenames = "/data/bdd100k/namelists/clear_or_partly_cloudy_or_overcast_and_highway_and_daytime.txt"
    out_filenames = "/data/bdd100k/namelists/rainy_or_snowy_or_foggy_and_highway_and_daytime_or_dawndusk_or_night.txt"
    norm_spec = [["weather", ["clear","partly cloudy", "overcast"]],["scene", "highway"],["timeofday", "daytime"]]
    out_spec = [["weather", ["rainy", "snowy", "foggy"]],["scene", "highway"],["timeofday",["daytime","dawn/dusk","night"]]]
    n_train = 5000
    n_val = 1000
    n_test = 2000
    out_frac = 0.5
    image_height = 192
    image_width = 320
    channels = 3
    save_name_lists=False
    labels_file = None
    get_norm_and_out_sets = False
    shuffle=False


