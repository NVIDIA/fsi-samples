from jsonpath_ng import parse


# map from python obj name to schema name
type_map = {
    "dict": 'object',
    'list': 'array',
    'float': 'number',
    'str': 'string',
    'int': 'number',
    'bool': 'boolean'
}


def parse_config(json_obj):
    expr = parse('$..*')  # search for all the fields in the json file
    matches = expr.find(json_obj)
    map_result = {}
    for match in matches:
        v = match.value
        if type(v) == dict:
            continue
        element_type = ''
        if type(v) == list:
            if len(v) > 0:
                ele = v[0]
                if type(ele) == dict or type(ele) == list:
                    continue
                else:
                    element_type = type_map[ele.__class__.__name__]
        if v is None:
            continue
        result_type = type_map[v.__class__.__name__]
        result_value = v
        result_element_type = element_type
        result_path = str(match.full_path)
        node_id = result_path.split('.')[0]
        if result_element_type:
            type_key = result_type + '_' + result_element_type
        else:
            type_key = result_type
        type_container = map_result.get(type_key, {})
        map_result[type_key] = type_container
        content_container = type_container.get(node_id, [])
        type_container[node_id] = content_container
        item_str = '.'.join(result_path.split('.')[2:])+" val: "+str(v)
        content_container.append({'value': result_value, "path": result_path,
                                  "item": item_str})
    return map_result
