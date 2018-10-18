import train_AAE
from configuration import Configuration as cfg

train_AAE.main(0,[0],10,5,cfg=cfg,bdd100k=True)
