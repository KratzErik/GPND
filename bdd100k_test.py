import novelty_detector_bdd100k as nd
from configuration import Configuration as cfg

nd.main(0, [0], 10, 5, cfg=cfg, bdd100k=True)
