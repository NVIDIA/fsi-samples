'''
'''
import os

# import sys
# from time import sleep

from gquant.dataframe_flow.node import TaskSpecSchema
import gquant.dataframe_flow as dff


from mortgage_common import (
    mortgage_etl_workflow_def, generate_mortgage_gquant_run_params_list,
    MortgageTaskNames)


def main():
    _basedir = os.path.dirname(__file__)

    # mortgage_data_path = '/datasets/rapids_data/mortgage'
    mortgage_data_path = os.path.join(_basedir, 'mortgage_data')

    # Using some default csv files for testing.
    # csvfile_names = os.path.join(mortgage_data_path, 'names.csv')
    # acq_data_path = os.path.join(mortgage_data_path, 'acq')
    # perf_data_path = os.path.join(mortgage_data_path, 'perf')
    # csvfile_acqdata = os.path.join(acq_data_path, 'Acquisition_2000Q1.txt')
    # csvfile_perfdata = \
    #     os.path.join(perf_data_path, 'Performance_2000Q1.txt_0')
    # mortgage_etl_workflow_def(
    #     csvfile_names, csvfile_acqdata, csvfile_perfdata)

    gquant_task_list = mortgage_etl_workflow_def()

    start_year = 2000
    end_year = 2001  # end_year is inclusive
    # end_year = 2016  # end_year is inclusive
    # part_count = 16  # the number of data files to train against
    part_count = 12  # the number of data files to train against
    # part_count = 4  # the number of data files to train against

    mortgage_run_params_dict_list = generate_mortgage_gquant_run_params_list(
        mortgage_data_path, start_year, end_year, part_count, gquant_task_list)

    _basedir = os.path.dirname(__file__)
    mortgage_lib_module = os.path.join(_basedir, 'mortgage_gquant_plugins.py')

    mortgage_workflow_runner_task = {
        TaskSpecSchema.uid:
            MortgageTaskNames.mortgage_workflow_runner_task_name,
        TaskSpecSchema.plugin_type: 'MortgageWorkflowRunner',
        TaskSpecSchema.conf: {
            'mortgage_run_params_dict_list': mortgage_run_params_dict_list
        },
        TaskSpecSchema.inputs: [],
        TaskSpecSchema.modulepath: mortgage_lib_module
    }

    # Can be multi-gpu. Set ngpus > 1. This is different than dask xgboost
    # which is distributed multi-gpu i.e. dask-xgboost could distribute on one
    # node or multiple nodes. In distributed mode the dmatrix is disributed.
    ngpus = 1
    xgb_gpu_params = {
        'nround': 100,
        'max_depth': 8,
        'max_leaves': 2 ** 8,
        'alpha': 0.9,
        'eta': 0.1,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'subsample': 1,
        'reg_lambda': 1,
        'scale_pos_weight': 2,
        'min_child_weight': 30,
        'tree_method': 'gpu_hist',
        'n_gpus': ngpus,
        # 'distributed_dask': True,
        'loss': 'ls',
        # 'objective': 'gpu:reg:linear',
        'objective': 'reg:squarederror',
        'max_features': 'auto',
        'criterion': 'friedman_mse',
        'grow_policy': 'lossguide',
        'verbose': True
    }

    xgb_trainer_task = {
        TaskSpecSchema.uid: MortgageTaskNames.xgb_trainer_task_name,
        TaskSpecSchema.plugin_type: 'XgbMortgageTrainer',
        TaskSpecSchema.conf: {
            'delete_dataframes': False,
            'xgb_gpu_params': xgb_gpu_params
        },
        TaskSpecSchema.inputs: [
            MortgageTaskNames.mortgage_workflow_runner_task_name
        ],
        TaskSpecSchema.modulepath: mortgage_lib_module
    }

    task_list = [mortgage_workflow_runner_task, xgb_trainer_task]

    # out_list = [MortgageTaskNames.mortgage_workflow_runner_task_name]
    # ((mortgage_feat_df_pandas, delinq_df_pandas),) = \
    #     dff.run(task_list, out_list)

    out_list = [MortgageTaskNames.xgb_trainer_task_name]
    (bst,) = dff.run(task_list, out_list)

    print('XGBOOST BOOSTER:\n', bst)


if __name__ == '__main__':
    main()
