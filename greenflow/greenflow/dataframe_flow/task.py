import copy
from .taskSpecSchema import TaskSpecSchema
from ._node_flow import OUTPUT_ID


module_cache = {}


__all__ = ['Task']


class Task(object):
    ''' A strong typed Task class that is converted from dictionary.
    '''

    def __init__(self, task_spec):

        self._task_spec = {}  # internal dict

        # whatever is passed in has to be valid
        TaskSpecSchema.validate(task_spec)
        self._task_spec = copy.copy(task_spec)
        # deepcopies of inputs can still be done
        self._task_spec[TaskSpecSchema.inputs] = \
            copy.deepcopy(task_spec[TaskSpecSchema.inputs])

    def __getitem__(self, key):
        return self._task_spec[key]

    def set_output(self):
        """
        set the uniq output id to task
        """
        from .taskGraph import OutputCollector
        self._task_spec[TaskSpecSchema.task_id] = OUTPUT_ID
        self._task_spec[TaskSpecSchema.node_type] = OutputCollector

    def get(self, key, default=None):
        return self._task_spec.get(key, default)


if __name__ == "__main__":
    t = {'id': 'test',
         'type': "DropNode",
         'conf': {},
         'inputs': ["node_other"]}
    task = Task(t)
