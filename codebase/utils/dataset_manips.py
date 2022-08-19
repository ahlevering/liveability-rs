from copy import deepcopy
from random import randint, seed

def add_split_to_cities_dict(samples, val_threshold, test_set_cities):
    for city in samples:
        for _,sample in samples[city].items():
            split_no = randint(1, 101)
            if split_no < val_threshold:
                split = 'train'
            elif not city in test_set_cities:
                split = 'val'
            else:
                split = 'test'
            sample['split'] = split
    return samples