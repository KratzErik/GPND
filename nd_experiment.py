import train_AAE
import novelty_detector as nd
from configuration import Configuration as cfg

train_AAE.main(0, [0], 10, 5, cfg=cfg) # training
nd.main(0, [0], 10, 5, cfg=cfg) # testing
