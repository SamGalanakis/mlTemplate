#Misc functions


def fold_mapper(n_splits):

    '''Creates fold map for given number of splits'''

    fold_map = {key:val for key,val in zip(range(n_splits),[list(range(n_splits)) for x in range(n_splits)])}
    for key, val in fold_map.items():
        val.pop(key)
    return fold_map


