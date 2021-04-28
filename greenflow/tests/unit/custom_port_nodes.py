import numpy as np
import pandas as pd
from greenflow.dataframe_flow import Node, MetaData
from greenflow.dataframe_flow import NodePorts, PortsSpecSchema
from greenflow.dataframe_flow import ConfSchema
import os
import warnings


class NodeHDFCacheMixin(object):

    def load_cache(self, filename=None) -> dict:
        """
        Defines the behavior of how to load the cache file from the `filename`.
        Node can override this method. Default implementation assumes cudf
        dataframes.

        Arguments
        -------
        filename: str
            filename of the cache file. Leave as none to use default.
        returns: dict
            dictionary of the output from this node
        """
        cache_dir = os.getenv('GREENFLOW_CACHE_DIR', self.cache_dir)
        if filename is None:
            filename = cache_dir + '/' + self.uid + '.hdf5'

        output_df = {}
        with pd.HDFStore(filename, mode='r') as hf:
            for oport, pspec in \
                    self._get_output_ports(full_port_spec=True).items():
                ptype = pspec.get(PortsSpecSchema.port_type)
                if self.outport_connected(oport):
                    ptype = ([ptype] if not isinstance(ptype,
                                                       list) else ptype)
                    key = '{}/{}'.format(self.uid, oport)
                    # check hdf store for the key
                    if key not in hf:
                        raise Exception(
                            'The task "{}" port "{}" key "{}" not found in'
                            'the hdf file "{}". Cannot load from cache.'
                            .format(self.uid, oport, key, filename)
                        )
                    if pd.DataFrame not in ptype:
                        warnings.warn(
                            RuntimeWarning,
                            'Task "{}" port "{}" port type is not set to '
                            'cudf.DataFrame. Attempting to load port data '
                            'with cudf.read_hdf.'.format(self.uid, oport))
                    output_df[oport] = pd.read_hdf(hf, key)
        return output_df

    def save_cache(self, output_data: dict):
        '''Defines the behavior for how to save the output of a node to
        filesystem cache. Default implementation assumes cudf dataframes.

        :param output_data: The output from :meth:`process`. For saving to hdf
            requires that the dataframe(s) have `to_hdf` method.
        '''
        cache_dir = os.getenv('GREENFLOW_CACHE_DIR', self.cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        filename = cache_dir + '/' + self.uid + '.hdf5'
        with pd.HDFStore(filename, mode='w') as hf:
            for oport, odf in output_data.items():
                # check for to_hdf attribute
                if not hasattr(odf, 'to_hdf'):
                    raise Exception(
                        'Task "{}" port "{}" output object is missing '
                        '"to_hdf" attribute. Cannot save to cache.'
                        .format(self.uid, oport))

                dtype = '{}'.format(type(odf)).lower()
                if 'dataframe' not in dtype:
                    warnings.warn(
                        RuntimeWarning,
                        'Task "{}" port "{}" port type is not a dataframe.'
                        ' Attempting to save to hdf with "to_hdf" method.'
                        .format(self.uid, oport))
                key = '{}/{}'.format(self.uid, oport)
                odf.to_hdf(hf, key, format='table', data_columns=True)


class PointNode(NodeHDFCacheMixin, Node):

    def ports_setup(self):
        input_ports = {}
        output_ports = {
            'points_df_out': {
                PortsSpecSchema.port_type: pd.DataFrame
            }
        }
        return NodePorts(inports=input_ports, outports=output_ports)

    def conf_schema(self):
        json = {
            "title": "PointNode configure",
            "type": "object",
            "properties": {
                "npts":  {
                    "type": "number",
                    "description": "number of data points",
                    "minimum": 10
                }
            },
            "required": ["npts"],
        }

        ui = {
            "npts": {"ui:widget": "updown"}
        }
        return ConfSchema(json=json, ui=ui)

    def init(self):
        pass

    def meta_setup(self):
        columns_out = {
            'points_df_out': {
                'x': 'float64',
                'y': 'float64'
            },
            'points_ddf_out': {
                'x': 'float64',
                'y': 'float64'
            }
        }
        return MetaData(inports={}, outports=columns_out)

    def process(self, inputs):
        npts = self.conf['npts']
        seed = self.conf.get('nseed')
        if seed is not None:
            np.random.seed(seed)
        df = pd.DataFrame()
        df['x'] = np.random.rand(npts)
        df['y'] = np.random.rand(npts)
        output = {}
        if self.outport_connected('points_df_out'):
            output.update({'points_df_out': df})
        return output


class DistanceNode(NodeHDFCacheMixin, Node):

    def ports_setup(self):
        port_type = PortsSpecSchema.port_type
        input_ports = {
            'points_df_in': {
                port_type: [pd.DataFrame]
            }
        }

        output_ports = {
            'distance_df': {
                port_type: [pd.DataFrame]
            },
            'distance_abs_df': {
                PortsSpecSchema.port_type:  [pd.DataFrame]
            }
        }
        input_connections = self.get_connected_inports()
        if 'points_df_in' in input_connections:
            types = input_connections['points_df_in']
            # connected, use the types passed in from parent
            return NodePorts(inports={'points_df_in': {port_type: types}},
                             outports={'distance_df': {port_type: types},
                                       'distance_abs_df': {port_type: types},
                                       })
        else:
            return NodePorts(inports=input_ports, outports=output_ports)

    def conf_schema(self):
        return ConfSchema()

    def init(self):
        self.delayed_process = True

    def meta_setup(self):
        req_cols = {
            'x': 'float64',
            'y': 'float64'
        }
        required = {
            'points_df_in': req_cols,
        }
        input_meta = self.get_input_meta()
        output_cols = ({
                'distance_df': {
                    'distance_df': 'float64',
                    'x': 'float64',
                    'y': 'float64'
                },
                'distance_abs_df': {
                    'distance_abs_df': 'float64',
                    'x': 'float64',
                    'y': 'float64'
                }
            })
        if 'points_df_in' in input_meta:
            col_from_inport = input_meta['points_df_in']
            # additional ports
            output_cols['distance_df'].update(col_from_inport)
            output_cols['distance_abs_df'].update(col_from_inport)
        return MetaData(inports=required, outports=output_cols)

    def process(self, inputs):
        df = inputs['points_df_in']
        output = {}
        if self.outport_connected('distance_df'):
            copy_df = df.copy()
            copy_df['distance_df'] = np.sqrt((df['x'] ** 2 + df['y'] ** 2))
            output.update({'distance_df': copy_df})
        if self.outport_connected('distance_abs_df'):
            copy_df = df.copy()
            copy_df['distance_abs_df'] = np.abs(df['x']) + np.abs(df['y'])
            output.update({'distance_abs_df': copy_df})
        return output
