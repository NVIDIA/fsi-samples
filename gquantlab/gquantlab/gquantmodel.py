#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Yi Dong.
# Distributed under the terms of the Modified BSD License.

"""
TODO: Add module docstring
"""

from ipywidgets import DOMWidget
from traitlets import Unicode, List
from ._frontend import module_name, module_version


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

    def set_taskgraph(self, task_graph):
        self.task_graph = task_graph
