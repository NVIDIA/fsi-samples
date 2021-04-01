from greenflow._common import _namedtuple_with_defaults


__all__ = ['MetaData', 'MetaDataSchema']

_MetaData = _namedtuple_with_defaults(
    '_MetaData',
    ['inports', 'outports'],
    {'inports': dict(), 'outports': dict()}
)


class MetaData(_MetaData):
    '''Node metadata must be setup for inputs and outputs. The validation
    logic will check whether the required inputs met the passed in output
    metadata. and the produced calculation results matches the output
    metadata.

    :ivar inports: Dictionary defining input metadata, which specified the
                   input requirement
    :ivar outports: Dictionary defining output metadata

    Empty dicts default:
        metadata = MetaData()
        metadata.inports and metadata.outports are empty dicts

    Example with port specs:
        inports = {
            'iport0_name': {
                "column0": "float64",
                "column1": "float64",
            },
            'iport1_name': {
                "column0": "float64",
                "column1": "float64",
                "column2": "float64",
            }
        }

        outports = {
            'oport0_name': {
                "column0": "float64",
                "column1": "float64",
                "column2": "float64",
            },
            'oport1_name': {
                "column0": "float64",
                "column1": "float64",
                "column2": "float64",
                "column3": "float64",
            }
        }

        metadata = MetaData(inports=inports, outports=outports)

    The inports/outports are nested dictionaries. The outer dictionary is keyed
    by port name with metadata obj being the value of the outer dictionary. The
    metadata obj is a dictionary with keys/fields which can be serialized into
    JSON.
    '''


class MetaDataSchema:
    '''Explanation of the fields, etc.'''
    META_OP_DELETION = 'deletion'
    META_OP_ADDITION = 'addition'
    META_OP_RETENTION = 'retention'
    META_REF_INPUT = 'input'
    META_OP = 'meta_op'
    META_DATA = 'data'
    META_ORDER = 'order'
