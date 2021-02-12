import json
from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
import tornado
from greenflow.dataframe_flow import TaskGraph
from .server_utils import (get_nodes, add_nodes)
import os
from greenflow.dataframe_flow.taskGraph import add_module_from_base64
from greenflow.dataframe_flow.task import get_greenflow_config_modules, load_modules
try:
    # For python 3.8 and later
    import importlib.metadata as importlib_metadata
except ImportError:
    # prior to python 3.8 need to install importlib-metadata
    import importlib_metadata


class RouteHandlerLoadGraph(APIHandler):
    @tornado.web.authenticated
    def get(self):
        self.finish("abcde")

    @tornado.web.authenticated
    def post(self):
        # input_data is a dictionnary with a key "name"
        input_data = self.get_json_body()
        task_graph = TaskGraph(input_data)
        # import pudb
        # pudb.set_trace()
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


class RouteHandlerPlugins(APIHandler):

    @tornado.web.authenticated
    def get(self):
        # load all the plugin information from the backend
        modules = get_greenflow_config_modules()
        client_info = {}
        client_info['validation'] = {}
        client_info['display'] = {}
        for key in modules.keys():
            if os.path.isdir(modules[key]):
                mod = load_modules(modules[key])
#                if hasattr(mod.mod, 'client'):
                client_mod = mod.mod
                if hasattr(client_mod, 'validation'):
                    val_dict = getattr(client_mod, 'validation')
                    client_info['validation'].update(val_dict)
                else:
                    pass
                    # print(client_mod, 'no validation')
                if hasattr(client_mod, 'display'):
                    val_dict = getattr(client_mod, 'display')
                    client_info['display'].update(val_dict)
                else:
                    pass
                    # print(client_mod, 'no display')
#                else:
#                    print(key, mod.mod, 'no client')

        # load all the plugins from entry points
        for entry_point in importlib_metadata.entry_points().get(
                'greenflow.plugin', ()):
            client_mod = entry_point.load()
            if hasattr(client_mod, 'validation'):
                val_dict = getattr(client_mod, 'validation')
                client_info['validation'].update(val_dict)
            else:
                pass
                # print(client_mod, 'no validation')
            if hasattr(client_mod, 'display'):
                val_dict = getattr(client_mod, 'display')
                client_info['display'].update(val_dict)
            else:
                pass
                # print(client_mod, 'no display')
        self.finish(json.dumps(client_info))


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
    # environment variable `GREENFLOWROOT`. Note, this
    # variable is not meant to be overwritten by user.
    # This variable can be used by other utility function
    # to compute the absolute path of the files.
    os.environ['GREENFLOWROOT'] = os.getcwd()
    # load all the graphs given the input gq.yaml file contents
    route_pattern0 = url_path_join(base_url, "greenflowlab", "load_graph")
    route_pattern1 = url_path_join(base_url, "greenflowlab", "all_nodes")
    route_pattern2 = url_path_join(base_url, "greenflowlab", "load_graph_path")
    route_pattern3 = url_path_join(base_url, "greenflowlab", "register_node")
    route_pattern4 = url_path_join(base_url, "greenflowlab", "register_plugins")
    handlers = [(route_pattern0, RouteHandlerLoadGraph),
                (route_pattern1, RouteHandlerLoadAllNodes),
                (route_pattern2, RouteHandlerLoadGraphFromPath),
                (route_pattern3, RouteHandlerRegister),
                (route_pattern4, RouteHandlerPlugins)]
    web_app.add_handlers(host_pattern, handlers)
