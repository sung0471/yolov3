import json
import os


def parse_model_config(path):
    module_defs = []
    new_path = '.'.join(path.split('.')[:-1]) + '.json'
    if not os.path.exists(new_path):
        file = open(path, 'r')
        lines = file.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.strip() for x in lines]    # get rid of fringe whitespaces

        for line in lines:
            if line.startswith('['):    # new block을 mask함
                module_defs.append({})
                module_defs[-1]['type'] = line[1:-1].rstrip()
                if module_defs[-1]['type'] == 'convolutional':
                    module_defs[-1]['batch_normalize'] = 0
            else:
                key, value = line.split('=')
                value = value.strip()
                module_defs[-1][key.rstrip()] = value.strip()

        json.dump({'data': module_defs}, open(new_path, 'w'), indent=2)
    else:
        decode = json.load(open(new_path, 'r'))
        module_defs = decode['data']

    return module_defs


def parse_data_config(path):
    options = dict()
    options['gpus'] = '0, 1, 2, 3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()

    return options
