import jax.numpy as jnp

def ld2da(l):
    ''' Transform a list of dicts to a dict of arrays.'''
    new_d = {}
    for d in l:
        # I assume that each element of the list l is a dict
        for key, value in d.items():
            if key not in new_d:
                new_d[key] = []
            new_d[key].append(value)
    
    for key, value in new_d.items():
        new_d[key] = jnp.stack(value)
    
    return new_d