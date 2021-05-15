import cudf
import os
import warnings
import pandas as pd

from greenflow.dataframe_flow.portsSpecSchema import PortsSpecSchema

__all__ = ['NodeHDFCacheMixin']


class NodeHDFCacheMixin:
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
                    if cudf.DataFrame not in ptype:
                        warnings.warn(
                            RuntimeWarning,
                            'Task "{}" port "{}" port type is not set to '
                            'cudf.DataFrame. Attempting to load port data '
                            'with cudf.read_hdf.'.format(self.uid, oport))
                    output_df[oport] = cudf.read_hdf(hf, key)
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
