#!/usr/bin/env python
# coding: utf-8

"""
TODO: Add module docstring
"""

import ipywidgets.widgets as widgets
import ipywidgets
from ipywidgets import DOMWidget
from traitlets import Unicode, List, Dict, Instance
from ._frontend import module_name, module_version


class GreenflowWidget(DOMWidget):
    """TODO: Add docstring here
    """
    _model_name = Unicode('GreenflowModel').tag(sync=True)
    _model_module = Unicode(module_name).tag(sync=True)
    _model_module_version = Unicode(module_version).tag(sync=True)
    _view_name = Unicode('GreenflowView').tag(sync=True)
    _view_module = Unicode(module_name).tag(sync=True)
    _view_module_version = Unicode(module_version).tag(sync=True)

    value = List().tag(sync=True)
    cache = Dict().tag(sync=True)
    sub = Instance(widgets.Widget).tag(sync=True,
                                       **widgets.widget_serialization)

    def __init__(self):
        self.sub = ipywidgets.HBox()
        super().__init__()
        self.on_msg(self._handle_event)

    def _handle_event(self, _, content, buffers):
        if content.get('event', '') == 'run':
            self.run()
        elif content.get('event', '') == 'clean':
            self.task_graph.run_cleanup(ui_clean=True)
            self.sub = ipywidgets.HBox()

    def set_taskgraph(self, task_graph):
        self.task_graph = task_graph

    def set_state(self, sync_data):
        super().set_state(sync_data)
        self.task_graph.reset()
        self.task_graph.extend(sync_data['value'])

    def run(self):
        result = self.task_graph.run(formated=True)
        self.sub = result
