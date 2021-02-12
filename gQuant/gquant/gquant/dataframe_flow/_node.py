import abc


__all__ = ['_Node']


# compatible with Python 2 *and* 3:
_ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class _Node(_ABC):
    '''Intermediate class to identify Node class instances and avoid cyclic
    imports.'''
