import train_AAE
import novelty_detector_bdd100k as nd
from configuration import Configuration as cfg

train_AAE.main(0, [0], 10, 5, cfg=cfg)
nd.main(0, [0], 10, 5, cfg=cfg)
