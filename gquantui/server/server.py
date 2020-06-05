import sys; sys.path.insert(0, '../..'); sys.path.append('modules') # noqa E262
from flask import Flask, jsonify, request
from gquant.dataframe_flow import TaskGraph
from server_utils import (get_nodes, add_nodes, 
                          get_nodes_from_file,
                          load_all_yamls)
import pathlib

app = Flask(__name__, static_folder='../client/build', static_url_path='/')


@app.route('/graph', methods=['POST'])
def get_graph():
    content = request.get_json(silent=True)
    name = content['filename']
    filename = str(pathlib.Path(__file__).parent / 'workflows' / name)
    r = get_nodes_from_file(filename)
    return jsonify(r)


@app.route('/add')
def add_node():
    result = add_nodes()
    return jsonify(result)


@app.route('/recalculate', methods=['POST'])
def recalculate():
    content = request.get_json(silent=True)
    task_graph = TaskGraph(content)
    r = get_nodes(task_graph)
    return jsonify(r)


@app.route('/workflows')
def workflows():
    r = load_all_yamls()
    return jsonify(r)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers',
                         'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods',
                         'GET,PUT,POST,DELETE,OPTIONS')
    return response
