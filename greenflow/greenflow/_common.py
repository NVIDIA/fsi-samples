from collections import namedtuple
from collections.abc import Mapping

__all__ = ['_namedtuple_with_defaults']


def _namedtuple_with_defaults(typename, field_names, default_values=()):
    # https://stackoverflow.com/a/18348004/3457624
    T = namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T
