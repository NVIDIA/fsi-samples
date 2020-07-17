import json

from notebook.base.handlers import APIHandler
from notebook.utils import url_path_join
import tornado
from gquant.dataframe_flow import TaskGraph
from .server_utils import (get_nodes, add_nodes)
import pathlib

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

class RouteHandlerLoadAllNodes(APIHandler):

    @tornado.web.authenticated
    def get(self):
        # input_data is a dictionnary with a key "name"
        result = add_nodes()
        self.finish(json.dumps(result))


def setup_handlers(web_app):
    host_pattern = ".*$"
    base_url = web_app.settings["base_url"]

    # load all the graphs given the input gq.yaml file contents
    route_pattern0 = url_path_join(base_url, "gquantlab", "load_graph")
    route_pattern1 = url_path_join(base_url, "gquantlab", "all_nodes")
    print(route_pattern0)
    handlers = [(route_pattern0, RouteHandlerLoadGraph), (route_pattern1, RouteHandlerLoadAllNodes)]
    web_app.add_handlers(host_pattern, handlers)