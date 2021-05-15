from collections.abc import Mapping
from itertools import chain
from typing import Iterable

from greenflow._common import _namedtuple_with_defaults

__all__ = ['PortsSpecSchema', 'NodePorts', 'ConfSchema']


_NodePorts = _namedtuple_with_defaults(
    '_NodePorts',
    ['inports', 'outports'],
    {'inports': dict(), 'outports': dict()}
)

_ConfSchema = _namedtuple_with_defaults(
    '_ConfSchema',
    ['json', 'ui'],
    {'json': dict(), 'ui': dict()}
)


class NodePorts(_NodePorts):
    '''Node ports must be defined for inputs and outputs.

    :ivar inports: Dictionary defining port specs for input ports
    :ivar outports: Dictionary defining port specs for output ports

    Empty dicts default:
        node_ports = NodePorts()
        node_ports.inports and node_ports.outports are empty dicts

    Example with port specs:
        inports = {
            'iport0_name': {
                PortsSpecSchema.port_type: cudf.DataFrame
            },
            'iport1_name': {
                PortsSpecSchema.port_type: cudf.DataFrame,
                PortsSpecSchema.optional: True
            }
        }

        outports = {
            'oport0_name': {
                PortsSpecSchema.port_type: cudf.DataFrame
            },
            'oport1_name': {
                PortsSpecSchema.port_type: cudf.DataFrame,
                PortsSpecSchema.optional: True
            }
        }

        node_ports = NodePorts(inports=inports, outports=outports)

    The inports/outports are nested dictionaries. The outer dictionary is keyed
    by port name with port spec being the value of the outer dictionary. The
    port spec is a dictionary with keys/fields per PortsSpecSchema class.

    '''


class ConfSchema(_ConfSchema):
    ''' ConfSchema must be defined for Node conf JSON.

    :ivar json: Dictionary defining port specs for input ports
    :ivar ui: Dictionary defining port specs for output ports

    Empty dicts default:
        confSchema = ConfSchema()
        confSchema.json and confSchema.ui are empty dicts
    Examples:
        const schema = {
          type: "boolean",
          enum: [true, false]
        };

        const uiSchema={
          "ui:enumDisabled": [true],
        };
        confSchema = ConfSchema(json=schema, ui=uiSchema)
     '''


class PortsSpecSchema(object):
    '''Outline fields expected in a ports definition for a node implementation.

    :cvar type: The type of instance for the port. This can also be a
        list of types if inputs can be of multiple types. Ex.:
            [cudf.DataFrame, pd.DataFrame]
        Optional port setting.
        Default: [] Empty list.
    :cvar optional: Boolean to indicate whether a given port is optional i.e.
        the input or output might be optional so missing.
        Optional port setting.
        Default: False i.e. if port defined it is assumed required.

    '''

    port_type = 'type'
    optional = 'optional'
    dynamic = 'dynamic'
    DYN_MATCH = 'matching_outputs'

    @classmethod
    def _typecheck(cls, schema_field, value):
        if (schema_field == cls.port_type):
            def check_ptype(val):
                err_msg = 'Port type must be a pythonic '\
                    'type i.e type(port_type) == type. Instead got: {}, {}'
                assert isinstance(val,
                                  type), err_msg.format(type(val), val)

            if isinstance(value, Iterable):
                for ptype in value:
                    check_ptype(ptype)
            else:
                check_ptype(value)
        elif schema_field == cls.optional:
            assert isinstance(value, bool), 'Optional field must be a '\
                'boolean. Instead got: {}'.format(value)
        elif schema_field == cls.dynamic:
            assert isinstance(value, bool), 'Dynamic field must be a '\
                'boolean. Instead got: {}'.format(value)
        else:
            raise KeyError('Uknown schema field "{}" in the port spec.'.format(
                schema_field))

    # _schema_req_fields = []

    @classmethod
    def validate_ports(cls, node_ports):
        '''
        :type node_ports: NodePorts
        '''
        if not isinstance(node_ports, NodePorts):
            raise AssertionError(
                'Ports definition must be of type NodePorts. Instead got: '
                '{}'.format(type(node_ports)))

        if not isinstance(node_ports.inports, Mapping):
            raise AssertionError(
                'Input ports must be defined as a Mapping or dict. Instead '
                'got: {}'.format(node_ports.inports))

        if not isinstance(node_ports.outports, Mapping):
            raise AssertionError(
                'Output ports must be defined as a Mapping or dict. Instead '
                'got: {}'.format(node_ports.outports))

        for port_name, port_spec in chain(node_ports.inports.items(),
                                          node_ports.outports.items()):

            assert isinstance(port_name, str), \
                'Port names must be strings. Instead got: {}'.format(port_name)

            if not isinstance(port_spec, Mapping):
                raise Exception(
                    'Port spec must be dict. Invalid port spec for port '
                    '"{}" port spec: {}'.format(port_name, port_spec))

            for port_field, field_val in port_spec.items():
                cls._typecheck(port_field, field_val)
