import collections

def recursive_update(dict1, dict2):
    """
    Given two dictionaries dict1 and dict2, update dict1 recursively.
    """
    for key, value in dict2.items():
        if key == 'args':
            dict1[key] = dict2[key]
        elif isinstance(value, collections.abc.Mapping):
            result = recursive_update(dict1.get(key, {}), value)
            dict1[key] = result
        else:
            dict1[key] = dict2[key]
    return dict1
