"""
 ////////////////////////////////////////////////////////////////////////////
 //
 // Copyright (C) NVIDIA Corporation.  All rights reserved.
 //
 // NVIDIA Sample Code
 //
 // Please refer to the NVIDIA end user license agreement (EULA) associated
 // with this source code for terms and conditions that govern your use of
 // this software. Any use, reproduction, disclosure, or distribution of
 // this software and related documentation outside the terms of the EULA
 // is strictly prohibited.
 //
 ////////////////////////////////////////////////////////////////////////////
"""

from greenflow.dataframe_flow import ConfSchema, PortsSpecSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from greenflow.dataframe_flow import Node
from .kernels import compute_cov_distance
import cupy
import cudf


class DistanceNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.delayed_process = True
        self.infer_meta = False
        self.INPUT_PORT_NAME = 'in'
        self.COV_DF = 'cov_df'
        self.MEAN_DF = 'mean_df'
        self.STD_DF = 'std_df'
        self.CORR_DF = 'corr_df'
        self.DISTANCE_DF = 'distance_df'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            }
        }
        port_outports = {
            self.MEAN_DF: {
                port_type: "${port:in}"
            },
            self.STD_DF: {
                port_type: "${port:in}"
            },
            self.COV_DF: {
                port_type: "${port:in}"
            },
            self.CORR_DF: {
                port_type: "${port:in}"
            },
            self.DISTANCE_DF: {
                port_type: "${port:in}"
            }
        }
        self.template_ports_setup(in_ports=port_inports,
                                  out_ports=port_outports)

    def update(self):
        TemplateNodeMixin.update(self)
        meta_outports = self.template_meta_setup().outports
        meta_inports = self.template_meta_setup().inports
        sub_dict = {
            'year': 'int16',
            'month': 'int16',
            'sample_id': 'int64',
        }
        required = {
            "date": "datetime64[ns]",
        }
        required.update(sub_dict)
        meta_inports[self.INPUT_PORT_NAME] = required
        json_cov = {}
        json_dis = {}
        json_mean = {}
        json_corr = {}
        json_std = {}
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            assets = len(input_meta[self.INPUT_PORT_NAME]) - 4
            for i in range(assets*assets):
                json_cov[i] = 'float64'
            for i in range(assets):
                json_mean[i] = 'float64'
                json_std[i] = 'float64'
            for i in range(assets*(assets-1)//2):
                json_dis[i] = 'float64'
                json_corr[i] = 'float64'
        json_cov.update(sub_dict)
        json_dis.update(sub_dict)
        json_mean.update(sub_dict)
        json_std.update(sub_dict)
        json_corr.update(sub_dict)
        meta_outports[self.MEAN_DF] = json_mean
        meta_outports[self.STD_DF] = json_std
        meta_outports[self.COV_DF] = json_cov
        meta_outports[self.CORR_DF] = json_corr
        meta_outports[self.DISTANCE_DF] = json_dis
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

    def conf_schema(self):
        json = {
            "title": "Compute the Distance Matrix and Cov df",
            "type": "object",
            "properties": {
                "window": {
                    'type': "integer",
                    "title": "Window size",
                    "description": """the number of months used to compute the
                    distance and vairance"""
                }
            },
            "required": ["window"],
        }

        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        df = inputs[self.INPUT_PORT_NAME]
        all_sample_ids = df['sample_id'].unique()
        total_samples = len(all_sample_ids)
        window = self.conf['window']
        means, cov, distance, all_dates = compute_cov_distance(total_samples,
                                                               df,
                                                               window=window)

        total_samples, num_months, assets, assets = cov.shape

        months_id = all_dates.dt.year*12 + (all_dates.dt.month-1)
        months_id = months_id - months_id.min()
        mid = (cupy.arange(months_id.max() + 1) +
               (all_dates.dt.month - 1)[0])[window:]
        minyear = all_dates.dt.year.min()
        if len(mid) == 0:
            mid = cupy.array([0])
        months = mid % 12
        years = mid // 12 + minyear

        output = {}
        # print(num_months, len(mid))
        if self.outport_connected(self.MEAN_DF):
            df_mean = cudf.DataFrame(
                means.reshape(total_samples*num_months, -1))
            df_mean['year'] = cupy.concatenate(
                [years]*total_samples).astype(cupy.int16)
            df_mean['month'] = cupy.concatenate(
                [months]*total_samples).astype(cupy.int16)
            df_mean['sample_id'] = cupy.repeat(cupy.arange(
                total_samples) + all_sample_ids.min(), len(mid))
            output.update({self.MEAN_DF: df_mean})
        if self.outport_connected(self.STD_DF):
            data_ma = cov.reshape(total_samples*num_months, assets, assets)
            diagonzied = cupy.diagonal(data_ma, 0, 1, 2)  # get var
            diagonzied = cupy.sqrt(diagonzied)  # get std
            df_std = cudf.DataFrame(diagonzied)
            df_std['year'] = cupy.concatenate(
                [years]*total_samples).astype(cupy.int16)
            df_std['month'] = cupy.concatenate(
                [months]*total_samples).astype(cupy.int16)
            df_std['sample_id'] = cupy.repeat(cupy.arange(
                total_samples) + all_sample_ids.min(), len(mid))
            output.update({self.STD_DF: df_std})
        if self.outport_connected(self.COV_DF):
            df_cov = cudf.DataFrame(cov.reshape(total_samples*num_months, -1))
            df_cov['year'] = cupy.concatenate(
                [years]*total_samples).astype(cupy.int16)
            df_cov['month'] = cupy.concatenate(
                [months]*total_samples).astype(cupy.int16)
            df_cov['sample_id'] = cupy.repeat(cupy.arange(
                total_samples) + all_sample_ids.min(), len(mid))
            output.update({self.COV_DF: df_cov})
        if self.outport_connected(self.CORR_DF):
            dis_ma = distance.reshape(total_samples*num_months, -1)
            dis_ma = 1 - 2.0 * dis_ma
            df_corr = cudf.DataFrame(dis_ma)
            df_corr['year'] = cupy.concatenate(
                [years]*total_samples).astype(cupy.int16)
            df_corr['month'] = cupy.concatenate(
                [months]*total_samples).astype(cupy.int16)
            df_corr['sample_id'] = cupy.repeat(cupy.arange(
                total_samples) + all_sample_ids.min(), len(mid))
            output.update({self.CORR_DF: df_corr})
        if self.outport_connected(self.DISTANCE_DF):
            df_dis = cudf.DataFrame(distance.reshape(total_samples*num_months,
                                                     -1))
            df_dis['year'] = cupy.concatenate(
                [years]*total_samples).astype(cupy.int16)
            df_dis['month'] = cupy.concatenate(
                [months]*total_samples).astype(cupy.int16)
            df_dis['sample_id'] = cupy.repeat(cupy.arange(
                total_samples) + all_sample_ids.min(), len(mid))
            output.update({self.DISTANCE_DF: df_dis})
        return output
