from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.portsSpecSchema import (
    PortsSpecSchema, NodePorts, ConfSchema)
from greenflow.dataframe_flow.metaSpec import MetaData
import cudf
import dask_cudf
import cuml
import copy
import cuml.dask.datasets.classification
from ..node_hdf_cache import NodeHDFCacheMixin

CUDF_PORT_NAME = 'cudf_out'
DASK_CUDF_PORT_NAME = 'dask_cudf_out'


class ClassificationData(NodeHDFCacheMixin, Node):

    def ports_setup(self):
        input_ports = {}
        output_ports = {
            CUDF_PORT_NAME: {
                PortsSpecSchema.port_type: cudf.DataFrame
            },
            DASK_CUDF_PORT_NAME: {
                PortsSpecSchema.port_type: dask_cudf.DataFrame
            }
        }
        return NodePorts(inports=input_ports, outports=output_ports)

    def meta_setup(self):
        column_types = {}
        total_features = self.conf.get("n_features", 20)
        dtype = self.conf.get("dtype", "float32")
        for i in range(total_features):
            column_types["x"+str(i)] = dtype
        column_types['y'] = 'int64'
        inputs = {
        }
        out_cols = {
            CUDF_PORT_NAME: column_types,
            DASK_CUDF_PORT_NAME: column_types,
        }
        metadata = MetaData(inports=inputs, outports=out_cols)
        return metadata

    def conf_schema(self):
        json = {
            "title": "Classification Node configure",
            "type": "object",
            "description": """Generate random dataframe for classification
            tasks. Generate a random n-class classification problem. This
            initially creates clusters of points normally distributed
            (std=1) about vertices of an n_informative-dimensional hypercube
            with sides of length 2*class_sep and assigns an equal number of
            clusters to each class.""",
            "properties": {
                "n_samples": {"type": "number",
                              "description": "The number of samples.",
                              "default": 100},
                "n_features": {"type": "number",
                               "description": """The total number of features.
                                These comprise n_informative informative
                                features, n_redundant redundant features,
                                n_repeated duplicated features and
                                n_features-n_informative-n_redundant-n_repeated
                                useless features drawn at random.""",
                               "default": 20},
                "n_informative": {"type": "number",
                                  "description": """The number of informative
                                  features. Each class is composed of a number
                                  of gaussian clusters each located around the
                                  vertices of a hypercube in a subspace of
                                  dimension n_informative. For each cluster,
                                  informative features are drawn independently
                                  from N(0, 1) and then randomly linearly
                                  combined within each cluster in order to add
                                  covariance. The clusters are then placed on
                                  the vertices of the hypercube.""",
                                  "default": 2},
                "n_redundant": {"type": "number",
                                "description": """The number of redundant
                                features. These features are generated as
                                random linear combinations of the informative
                                features.""",
                                "default": 2},
                "n_repeated": {"type": "number",
                               "description": """The number of duplicated
                               features, drawn randomly from the informative
                               and the redundant features.""",
                               "default": 0},
                "n_classes": {"type": "number",
                              "description": """The number of classes (or
                              labels) of the classification problem.""",
                              "default": 2},
                "n_clusters_per_class": {"type": "number",
                                         "description": """The number of
                                         clusters per class.""",
                                         "default": 2},
                "weights": {"type": "array",
                            "items": {
                                "type": "number"
                            },
                            "description": """The proportions of samples
                            assigned to each class. If None, then classes are
                            balanced. Note that if len(weights) ==
                            n_classes - 1, then the last class weight is
                            automatically inferred. More than n_samples
                            samples may be returned if the sum of weights
                            exceeds 1."""},
                "flip_y": {"type": "number",
                           "description": """The fraction of samples whose
                           class is assigned randomly. Larger values introduce
                           noise in the labels and make the classification
                           task harder.""",
                           "default": 0.01},
                "class_sep": {"type": "number",
                              "description": """The factor multiplying the
                              hypercube size. Larger values spread out the
                              clusters/classes and make the classification
                              task easier.""",
                              "default": 1.0},
                "hypercube": {"type": "boolean",
                              "description": """If True, the clusters are put
                              on the vertices of a hypercube. If False, the
                              clusters are put on the vertices of a random
                              polytope.""",
                              "default": True},
                "shift": {"type": "number",
                          "description": """Shift features by the specified
                          value. If None, then features are shifted by a
                          random value drawn in [-class_sep, class_sep].""",
                          "default": 0.0},
                "scale": {"type": "number",
                          "description": """Multiply features by the specified
                          value. If None, then features are scaled by a random
                          value drawn in [1, 100]. Note that scaling happens
                          after shifting.""",
                          "default": 1.0},
                "shuffle": {"type": "boolean",
                            "description": """Shuffle the samples and the
                            features.""",
                            "default": True},
                "random_state":  {"type": "number",
                                  "description": """Determines random number
                                  generation for dataset creation. Pass an int
                                  for reproducible output across multiple
                                  function calls. See Glossary."""},
                "order": {"type": "string",
                          "description": "The order of the generated samples",
                          "enum": ["F", "C"],
                          "default": "F"},
                "dtype": {"type": "string",
                          "description": "Dtype of the generated samples",
                          "enum": ["float64", "float32"],
                          "default": "float64"},
                "n_parts": {"type": "number",
                            "description": """used for Dask dataframe, number
                            of partitions to generate (this can be greater
                            than the number of workers""",
                            "default": 4}
            }
        }
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        genearte the fake data for classification
        Arguments
        -------
         inputs: list
             empty list
        Returns
        -------
        cudf.DataFrame
        """
        output = {}

        def get_cudf(offset=None):
            conf = copy.copy(self.conf)
            if 'n_parts' in conf:
                del conf['n_parts']
            x, y = cuml.datasets.make_classification(**conf)
            df = cudf.DataFrame({'x'+str(i): x[:, i]
                                 for i in range(x.shape[1])})
            df['y'] = y
            if offset is not None:
                df.index += offset
            return df

        if self.outport_connected(CUDF_PORT_NAME):
            df = get_cudf()
            output.update({CUDF_PORT_NAME: df})
        if self.outport_connected(DASK_CUDF_PORT_NAME):

            def mapfun(x):
                return x.get()

            x, y = cuml.dask.datasets.classification.make_classification(
                **self.conf)
            ddf = x.map_blocks(mapfun,
                               dtype=x.dtype).to_dask_dataframe()
            out = dask_cudf.from_dask_dataframe(ddf)
            out.columns = ['x'+str(i) for i in range(x.shape[1])]
            out['y'] = y.astype('int64')
            output.update({DASK_CUDF_PORT_NAME: out})
        return output
