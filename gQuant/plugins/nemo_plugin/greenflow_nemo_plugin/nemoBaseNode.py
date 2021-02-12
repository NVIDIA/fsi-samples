from greenflow.dataframe_flow.portsSpecSchema import ConfSchema, PortsSpecSchema
from greenflow.dataframe_flow.portsSpecSchema import NodePorts, MetaData

from nemo.core.neural_types import NmTensor
import inspect
from greenflow.plugin_nodes.util.json_util import type_map
from collections import OrderedDict
from greenflow.dataframe_flow.util import get_file_path
from nemo.backends.pytorch.nm import (DataLayerNM,
                                      TrainableNM, LossNM)
import nemo

__all__ = ["NeMoBase"]

defaut_type = 'number'

share_weight = 'share_weight'


class FeedProperty(object):
    def __init__(self, conf):
        self.__dict__.update(conf)


def serialize_type(neural_type):
    output = {}
    axes = []
    if neural_type.axes is None:
        pass
    else:
        for ax in neural_type.axes:
            axes.append({'kind': str(ax.kind),
                         'size': ax.size})
    output['axes'] = axes
    ele = {}
    ele_type = neural_type.elements_type
    ele['types'] = [cla.__name__ for cla in ele_type.__class__.mro()
                    if cla.__name__ != 'ABC' and cla.__name__ != 'object']
    ele['fields'] = str(ele_type.fields)
    ele['parameters'] = str(ele_type.type_parameters)
    output['element'] = ele
    return output


def get_parameters(class_obj, conf):
    init_fun = class_obj.__init__
    sig = inspect.signature(init_fun)
    hasEmpty = False
    init_para = {}
    for key in sig.parameters.keys():
        if key == 'self':
            # ignore the self
            continue
        if key in conf:
            init_para[key] = conf[key]
        else:
            hasEmpty = True
            break
    if not hasEmpty:
        return init_para
    else:
        return None


def get_conf_parameters(class_obj):
    init_fun = class_obj.__init__
    sig = inspect.signature(init_fun)
    init_para = OrderedDict()
    for key in sig.parameters.keys():
        if key == 'self':
            # ignore the self
            continue
        para = sig.parameters[key]
        default_val = None
        if para.default == inspect._empty:
            p_type = defaut_type
        elif para.default is None:
            p_type = defaut_type
        else:
            if para.default.__class__.__name__ not in type_map:
                print(para.default, type(para.default))
                p_type = defaut_type
            else:
                p_type = type_map[para.default.__class__.__name__]
            default_val = para.default
        init_para[para.name] = (p_type, default_val)
    return init_para


class NeMoBase:

    def init(self, class_obj):
        if nemo.core.NeuralModuleFactory.get_default_factory() is None:
            nemo.core.NeuralModuleFactory()
        self.instanceClass = class_obj
        self.instance = None
        self.file_fields = []
        conf_para = get_conf_parameters(class_obj)
        self.fix_type = {}
        self.INPUT_NM = 'in_nm'
        self.OUTPUT_NM = 'out_nm'
        for key in conf_para.keys():
            if key.find('name') >= 0:
                self.fix_type[key] = "string"
            if key.find('model') >= 0:
                self.fix_type[key] = "string"
            if key.find('file') >= 0:
                self.file_fields.append(key)
        for f in self.file_fields:
            self.fix_type[f] = 'string'
            if f in self.conf and self.conf[f]:
                self.conf[f] = get_file_path(self.conf[f])
        if not issubclass(class_obj, DataLayerNM):
            try:
                if issubclass(self.instanceClass, TrainableNM):
                    input_meta = self.get_input_meta()
                    if self.INPUT_NM in input_meta:
                        if (share_weight in self.conf and
                                self.conf[share_weight] == 'Reuse'):
                            self.conf = input_meta[self.INPUT_NM]
                app = nemo.utils.app_state.AppState()
                ins = None
                for mod in app._module_registry:
                    if isinstance(mod, self.instanceClass):
                        ins = mod
                        break
                if ins is None:
                    ins = class_obj(**self.conf)
                if self.instance is None:
                    self.instance = ins
            except Exception as e:
                print(e)
                pass

    def _clean_dup(self):
        app = nemo.utils.app_state.AppState()
        if 'name' in self.conf:
            if app._module_registry.has(self.conf['name']):
                existing = app._module_registry[self.conf['name']]
                app._module_registry.remove(existing)
        removeList = []
        for mod in app._module_registry:
            if isinstance(mod, self.instanceClass):
                # remove the duplicate instances
                removeList.append(mod)
        for mod in removeList:
            app._module_registry.remove(mod)

    def ports_setup(self):
        port_type = PortsSpecSchema.port_type
        if self.instance is not None:
            inports = self.instance.input_ports
            outports = self.instance.output_ports
        else:
            try:
                p_inports = self.instanceClass.input_ports
                p_outports = self.instanceClass.output_ports
                feeder = FeedProperty(self.conf)
                inports = p_inports.fget(feeder)
                outports = p_outports.fget(feeder)
            except Exception:
                inports = None
                outports = None
        o_inports = {}
        o_outports = {}
        if inports is not None:
            for k in inports.keys():
                o_inports[k] = {port_type: NmTensor}
        if outports is not None:
            for k in outports.keys():
                o_outports[k] = {port_type: NmTensor}
        if issubclass(self.instanceClass, TrainableNM):
            # added the port for tying the weights
            o_inports[self.INPUT_NM] = {port_type: TrainableNM}
            o_outports[self.OUTPUT_NM] = {port_type: TrainableNM}
        elif issubclass(self.instanceClass, LossNM):
            o_outports[self.OUTPUT_NM] = {port_type: LossNM}
        elif issubclass(self.instanceClass, DataLayerNM):
            o_outports[self.OUTPUT_NM] = {port_type: DataLayerNM}
        return NodePorts(inports=o_inports, outports=o_outports)

    def meta_setup(self):
        input_meta = self.get_input_meta()
        if issubclass(self.instanceClass, TrainableNM):
            input_meta = self.get_input_meta()
            if self.INPUT_NM in input_meta:
                if (share_weight in self.conf and
                        self.conf[share_weight] == 'Reuse'):
                    self.conf = input_meta[self.INPUT_NM]
        if self.instance is not None:
            inports = self.instance.input_ports
            outports = self.instance.output_ports
        else:
            try:
                p_inports = self.instanceClass.input_ports
                p_outports = self.instanceClass.output_ports
                feeder = FeedProperty(self.conf)
                inports = p_inports.fget(feeder)
                outports = p_outports.fget(feeder)
            except Exception:
                inports = None
                outports = None
        required = {}
        out_meta = {}
        if inports is not None:
            for k in inports.keys():
                required[k] = serialize_type(inports[k])
        if outports is not None:
            for k in outports.keys():
                out_meta[k] = serialize_type(outports[k])
        if self.instance is not None:
            out_meta[self.OUTPUT_NM] = self.conf
        metadata = MetaData(inports=required, outports=out_meta)
        return metadata

    def conf_schema(self):
        conf_para = get_conf_parameters(self.instanceClass)
        class_doc = self.instanceClass.__doc__
        desc = "" if class_doc is None else class_doc
        init_doc = self.instanceClass.__init__.__doc__
        desc += "" if init_doc is None else init_doc
        json = {
            "title": "NeMo "+self.instanceClass.__name__+" Node",
            "type": "object",
            "description": desc,
            "properties": {

            },
        }
        ui = {
        }
        for f in self.file_fields:
            if f in conf_para:
                ui[f] = {"ui:widget": "FileSelector"}
        for p in conf_para.keys():
            stype = conf_para[p][0]
            if p in self.fix_type:
                stype = self.fix_type[p]
            json['properties'][p] = {
                "type": stype,
                "default": conf_para[p][1]
            }
        if issubclass(self.instanceClass, TrainableNM):
            if share_weight in conf_para:
                print('warning, share_weight parameter name collision')
            json['properties'][share_weight] = {
                "type": 'string',
                "description": """Weight Sharing between Modules: Reuse,
                re-use neural modules between training, evaluation and
                inference graphs; Copy: copy weights betwen modules. subsequent
                update of weights in one module will not affect weights in the
                other module. This means that the weights will get DIFFERENT
                gradients on the update step. Tying: the default one, tie
                weights between two or more modules. Tied weights are identical
                across all modules. Gradients to the weights will be the
                SAME.""",
                "enum": ['Reuse', 'Copy', 'Tying'],
                "default": 'Tying'
            }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        para = get_parameters(self.instanceClass, self.conf)
        app = nemo.utils.app_state.AppState()
        self.instance = None
        if issubclass(self.instanceClass, TrainableNM):
            if self.INPUT_NM in inputs:
                inputIn = inputs[self.INPUT_NM]
                if (share_weight in self.conf and
                        self.conf[share_weight] == 'Reuse'):
                    self.instance = inputIn
        if para is not None and self.instance is None:
            self._clean_dup()
            self.instance = self.instanceClass(**para)
        if self.instance is None:
            return {}
        if issubclass(self.instanceClass, TrainableNM):
            if self.INPUT_NM in inputs:
                inputIn = inputs[self.INPUT_NM]
                if (share_weight in self.conf and
                        self.conf[share_weight] == 'Reuse'):
                    pass
                elif (share_weight in self.conf and
                        self.conf[share_weight] == 'Copy'):
                    self.instance.set_weights(inputIn.get_weights())
                else:
                    self.instance.tie_weights_with(inputIn,
                                                   list(
                                                       inputIn.get_weights(
                                                       ).keys()))
        inputsCopy = OrderedDict()
        for k in self.instance.input_ports.keys():
            if k in inputs:
                inputsCopy[k] = inputs[k]
        instanceName = self.instance.name
        if instanceName in app.active_graph._modules:
            del app.active_graph._modules[instanceName]
        o = self.instance(**inputsCopy)
        if isinstance(o, tuple):
            output = {}
            for key in self.instance.output_ports.keys():
                output[key] = getattr(o, key)
        else:
            key = list(self.instance.output_ports.keys())[0]
            output = {key: o}
        # if self.uid == 'eval_data':
        #     print(inputs, output)
        #     for k in output.keys():
        #         print(output[k].name)
        #         print(output[k].unique_name)

        output[self.OUTPUT_NM] = self.instance
        return output
