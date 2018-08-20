import numpy as np


def parse_config(name, config):
    if name == 'conv_block':
        return config[0], config[1]
    if name == 'pool':
        return config[0], config[1], config[2], config[3]