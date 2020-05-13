
import sys; sys.path.insert(0, '..')
from flask import Flask, jsonify, request
from gquant.dataframe_flow import TaskGraph
from get_graph import get_nodes, add_nodes, get_nodes_from_file
import pathlib

app = Flask(__name__, static_folder='./ui/build', static_url_path='/')


@app.route('/graph')
def get_graph():
    filename = str(pathlib.Path(__file__).parent / 'custom_wflow.yaml')
    r = get_nodes_from_file(filename)
    return jsonify(r)


@app.route('/add')
def add_node():
    result = add_nodes('custom_port_nodes')
    return jsonify(result)


@app.route('/recalculate', methods=['POST'])
def recalculate():
    content = request.get_json(silent=True)
    task_graph = TaskGraph(content)
    r = get_nodes(task_graph)
    return jsonify(r)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers',
                         'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods',
                         'GET,PUT,POST,DELETE,OPTIONS')
    return response
