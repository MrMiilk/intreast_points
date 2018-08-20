import numpy as np


def parse_config(name, config):
    '''对传递到不同模块里的congfig进行解析'''
    if name == 'conv_block':
        return config[0], config[1]
    if name == 'pool':
        return config[0], config[1], config[2], config[3]