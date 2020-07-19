#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Yi Dong.
# Distributed under the terms of the Modified BSD License.

"""
TODO: Add module docstring
"""

from ipywidgets import DOMWidget
from traitlets import Unicode, List, Dict
from ._frontend import module_name, module_version

OUTPUT_ID= 'collector_id_fd9567b6'

class GQuantWidget(DOMWidget):
    """TODO: Add docstring here
    """
    _model_name = Unicode('GQuantModel').tag(sync=True)
    _model_module = Unicode(module_name).tag(sync=True)
    _model_module_version = Unicode(module_version).tag(sync=True)
    _view_name = Unicode('GQuantView').tag(sync=True)
    _view_module = Unicode(module_name).tag(sync=True)
    _view_module_version = Unicode(module_version).tag(sync=True)

    value = List().tag(sync=True)
    cache = Dict().tag(sync=True)

    def set_taskgraph(self, task_graph):
        self.task_graph = task_graph

    def set_state(self, sync_data):
        super().set_state(sync_data)
        self.task_graph.reset()
        self.task_graph.extend(sync_data['value'])
        # get all the outputs
        edges = sync_data['cache']['edges']
        outputs = []
        for edge in edges:
            if edge['to'].split('.')[0] == OUTPUT_ID:
                outputs.append(edge['from'])
        self.task_graph.set_outputs(outputs)