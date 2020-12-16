from ._node import _Node

__all__ = ['TaskSpecSchema']


class TaskSpecSchema(object):
    '''Outline fields expected in a dictionary specifying a task node.
    :cvar task_id: unique id or name for the node
    :cvar node_type: Plugin class i.e. subclass of Node. Specified as string
        or subclass of Node
    :cvar conf: Configuration for the plugin i.e. parameterization. This is a
        dictionary.
    :cvar filepath: Path to python module for custom plugin types.
    :cvar module: optional field for the name of the module.
    :cvar inputs: List of ids of other tasks or an empty list.
    '''

    task_id = 'id'
    node_type = 'type'
    conf = 'conf'
    filepath = 'filepath'
    module = 'module'
    inputs = 'inputs'
    # outputs = 'outputs'
    load = 'load'
    save = 'save'

    @classmethod
    def _typecheck(cls, schema_field, value):
        try:
            if (schema_field == cls.task_id):
                assert isinstance(value, str)
            elif schema_field == cls.node_type:
                assert (isinstance(value, str) or issubclass(value, _Node))
            elif schema_field == cls.conf:
                assert (isinstance(value, dict) or isinstance(value, list))
            elif schema_field == cls.filepath:
                assert isinstance(value, str)
            elif schema_field == cls.module:
                assert isinstance(value, str)
            elif schema_field == cls.inputs:
                assert (isinstance(value, list) or isinstance(value, dict))
                for item in value:
                    assert isinstance(item, str)
            # elif schema_field == cls.outputs:
            #     assert isinstance(value, list)
            #     for item in value:
            #         assert isinstance(item, str)
            elif schema_field == cls.load:
                pass
            elif schema_field == cls.save:
                assert isinstance(value, bool)
            else:
                raise KeyError(
                    'Uknown schema field "{}" in the task spec.'.format(
                        schema_field))
        except AssertionError as e:
            print(schema_field, value)
            raise e

    _schema_req_fields = [task_id, node_type, conf, inputs]

    @classmethod
    def validate(cls, task_spec):
        '''
        :param task_spec: A dictionary per TaskSpecSchema
        '''
        for ifield in cls._schema_req_fields:
            if ifield not in task_spec:
                raise KeyError('task spec missing required field: {}'
                               .format(ifield))

        for task_field, field_val in task_spec.items():
            cls._typecheck(task_field, field_val)
