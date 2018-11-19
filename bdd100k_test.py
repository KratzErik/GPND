import novelty_detector as nd
from configuration import Configuration as cfg

nd.main(0, [0], 10, 5, cfg=cfg, dataset="bdd100k")
