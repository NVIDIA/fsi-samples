import json

from notebook.base.handlers import APIHandler
from notebook.utils import url_path_join
import tornado
from gquant.dataframe_flow import TaskGraph
from .server_utils import (get_nodes, add_nodes)
import os
from gquant.dataframe_flow.taskGraph import add_module_from_base64


class RouteHandlerLoadGraph(APIHandler):
    @tornado.web.authenticated
    def get(self):
        self.finish("abcde")

    @tornado.web.authenticated
    def post(self):
        # input_data is a dictionnary with a key "name"
        input_data = self.get_json_body()
        task_graph = TaskGraph(input_data)
        nodes_and_edges = get_nodes(task_graph)
        self.finish(json.dumps(nodes_and_edges))


class RouteHandlerLoadGraphFromPath(APIHandler):

    @tornado.web.authenticated
    def post(self):
        # input_data is a dictionnary with a key "name"
        input_data = self.get_json_body()
        task_graph = TaskGraph.load_taskgraph(input_data['path'])
        nodes_and_edges = get_nodes(task_graph)
        self.finish(json.dumps(nodes_and_edges))


class RouteHandlerRegister(APIHandler):

    @tornado.web.authenticated
    def post(self):
        from .server_utils import register_node
        # input_data is a dictionnary with a key "name"
        input_data = self.get_json_body()
        module_name = input_data['module']
        class_str = input_data['class']
        class_obj = add_module_from_base64(module_name, class_str)
        register_node(module_name, class_obj)
        self.finish(json.dumps(class_obj.__name__))


class RouteHandlerLoadAllNodes(APIHandler):

    @tornado.web.authenticated
    def get(self):
        # input_data is a dictionnary with a key "name"
        result = add_nodes()
        self.finish(json.dumps(result))


def setup_handlers(web_app):
    host_pattern = ".*$"
    base_url = web_app.settings["base_url"]
    # pass the jupyterlab server root directory to
    # environment variable `GQUANTROOT`. Note, this
    # variable is not meant to be overwritten by user.
    # This variable can be used by other utility function
    # to compute the absolute path of the files.
    os.environ['GQUANTROOT'] = os.getcwd()
    # load all the graphs given the input gq.yaml file contents
    route_pattern0 = url_path_join(base_url, "gquantlab", "load_graph")
    route_pattern1 = url_path_join(base_url, "gquantlab", "all_nodes")
    route_pattern2 = url_path_join(base_url, "gquantlab", "load_graph_path")
    route_pattern3 = url_path_join(base_url, "gquantlab", "register_node")
    handlers = [(route_pattern0, RouteHandlerLoadGraph),
                (route_pattern1, RouteHandlerLoadAllNodes),
                (route_pattern2, RouteHandlerLoadGraphFromPath),
                (route_pattern3, RouteHandlerRegister)]
    web_app.add_handlers(host_pattern, handlers)
