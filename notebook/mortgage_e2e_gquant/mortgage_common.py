'''
Collection of functions to run the mortgage example.
'''
import os
from glob import glob


class MortgageTaskNames(object):
    load_acqdata_task_name = 'acqdata'
    load_perfdata_task_name = 'perfdata'
    ever_feat_task_name = 'ever_features'
    delinq_feat_task_name = 'delinq_features'
    join_perf_ever_delinq_feat_task_name = 'join_perf_ever_delinq_features'
    create_12mon_feat_task_name = 'create_12mon_features'
    final_perf_delinq_task_name = 'final_perf_delinq_features'
    final_perf_acq_task_name = 'final_perf_acq_df'

    mortgage_workflow_runner_task_name = 'mortgage_workflow_runner'
    xgb_trainer_task_name = 'xgb_trainer'

    dask_mortgage_workflow_runner_task_name = 'dask_mortgage_workflow_runner'
    dask_xgb_trainer_task_name = 'dask_xgb_trainer'


def mortgage_workflow_def(csvfile_names=None, csvfile_acqdata=None,
                          csvfile_perfdata=None):
    from gquant.dataframe_flow.node import TaskSpecSchema

    _basedir = os.path.dirname(__file__)

    mortgage_lib_module = os.path.join(_basedir, 'mortgage_gquant_plugins.py')

    # print('CSVFILE_ACQDATA: ', csvfile_acqdata)
    # print('CSVFILE_PERFDATA: ', csvfile_perfdata)

    # load acquisition
    load_acqdata_task = {
        TaskSpecSchema.uid: MortgageTaskNames.load_acqdata_task_name,
        TaskSpecSchema.plugin_type: 'CsvMortgageAcquisitionDataLoader',
        TaskSpecSchema.conf: {
            'csvfile_names': csvfile_names,
            'csvfile_acqdata': csvfile_acqdata
        },
        TaskSpecSchema.inputs: [],
        TaskSpecSchema.modulepath: mortgage_lib_module
    }

    # load performance data
    load_perfdata_task = {
        TaskSpecSchema.uid: MortgageTaskNames.load_perfdata_task_name,
        TaskSpecSchema.plugin_type: 'CsvMortgagePerformanceDataLoader',
        TaskSpecSchema.conf: {
            'csvfile_perfdata': csvfile_perfdata
        },
        TaskSpecSchema.inputs: [],
        TaskSpecSchema.modulepath: mortgage_lib_module
    }

    # calculate loan delinquency stats
    ever_feat_task = {
        TaskSpecSchema.uid: MortgageTaskNames.ever_feat_task_name,
        TaskSpecSchema.plugin_type: 'CreateEverFeatures',
        TaskSpecSchema.conf: dict(),
        TaskSpecSchema.inputs: [MortgageTaskNames.load_perfdata_task_name],
        TaskSpecSchema.modulepath: mortgage_lib_module
    }

    delinq_feat_task = {
        TaskSpecSchema.uid: MortgageTaskNames.delinq_feat_task_name,
        TaskSpecSchema.plugin_type: 'CreateDelinqFeatures',
        TaskSpecSchema.conf: dict(),
        TaskSpecSchema.inputs: [MortgageTaskNames.load_perfdata_task_name],
        TaskSpecSchema.modulepath: mortgage_lib_module
    }

    join_perf_ever_delinq_feat_task = {
        TaskSpecSchema.uid:
            MortgageTaskNames.join_perf_ever_delinq_feat_task_name,
        TaskSpecSchema.plugin_type: 'JoinPerfEverDelinqFeatures',
        TaskSpecSchema.conf: dict(),
        TaskSpecSchema.inputs: [
            MortgageTaskNames.load_perfdata_task_name,
            MortgageTaskNames.ever_feat_task_name,
            MortgageTaskNames.delinq_feat_task_name
        ],
        TaskSpecSchema.modulepath: mortgage_lib_module
    }

    create_12mon_feat_task = {
        TaskSpecSchema.uid: MortgageTaskNames.create_12mon_feat_task_name,
        TaskSpecSchema.plugin_type: 'Create12MonFeatures',
        TaskSpecSchema.conf: dict(),
        TaskSpecSchema.inputs: [
            MortgageTaskNames.join_perf_ever_delinq_feat_task_name
        ],
        TaskSpecSchema.modulepath: mortgage_lib_module
    }

    final_perf_delinq_task = {
        TaskSpecSchema.uid: MortgageTaskNames.final_perf_delinq_task_name,
        TaskSpecSchema.plugin_type: 'FinalPerfDelinq',
        TaskSpecSchema.conf: dict(),
        TaskSpecSchema.inputs: [
            MortgageTaskNames.load_perfdata_task_name,
            MortgageTaskNames.join_perf_ever_delinq_feat_task_name,
            MortgageTaskNames.create_12mon_feat_task_name
        ],
        TaskSpecSchema.modulepath: mortgage_lib_module
    }

    final_perf_acq_task = {
        TaskSpecSchema.uid: MortgageTaskNames.final_perf_acq_task_name,
        TaskSpecSchema.plugin_type: 'JoinFinalPerfAcqClean',
        TaskSpecSchema.conf: dict(),
        TaskSpecSchema.inputs: [
            MortgageTaskNames.final_perf_delinq_task_name,
            MortgageTaskNames.load_acqdata_task_name
        ],
        TaskSpecSchema.modulepath: mortgage_lib_module
    }

    task_list = [
        load_acqdata_task, load_perfdata_task,
        ever_feat_task, delinq_feat_task, join_perf_ever_delinq_feat_task,
        create_12mon_feat_task, final_perf_delinq_task, final_perf_acq_task
    ]

    return task_list


def generate_mortgage_gquant_run_params_list(
        mortgage_data_path, start_year, end_year, part_count,
        gquant_task_list):
    '''
    '''

    from gquant.dataframe_flow.node import TaskSpecSchema

    csvfile_names = os.path.join(mortgage_data_path, 'names.csv')
    acq_data_path = os.path.join(mortgage_data_path, 'acq')
    perf_data_path = os.path.join(mortgage_data_path, 'perf')

    quarter = 1
    year = start_year
    count = 0

    out_list = [MortgageTaskNames.final_perf_acq_task_name]
    mortgage_run_params_dict_list = []
    while year <= end_year:
        if count >= part_count:
            break

        perf_data_files = glob(os.path.join(
            perf_data_path + "/Performance_{}Q{}*".format(
                str(year), str(quarter))))

        csvfile_acqdata = acq_data_path + "/Acquisition_" + \
            str(year) + "Q" + str(quarter) + ".txt"

        for csvfile_perfdata in perf_data_files:
            if count >= part_count:
                break

            replace_spec = {
                MortgageTaskNames.load_acqdata_task_name: {
                    TaskSpecSchema.conf: {
                        'csvfile_names': csvfile_names,
                        'csvfile_acqdata': csvfile_acqdata
                    }
                },
                MortgageTaskNames.load_perfdata_task_name: {
                    TaskSpecSchema.conf: {
                        'csvfile_perfdata': csvfile_perfdata
                    }
                }
            }

            # Uncomment 'csvfile_perfdata' for debugging chunks in
            # DaskMortgageWorkflowRunner.
            run_params_dict = {
                # 'csvfile_perfdata': csvfile_perfdata,
                'replace_spec': replace_spec,
                'task_list': gquant_task_list,
                'out_list': out_list
            }

            mortgage_run_params_dict_list.append(run_params_dict)

            count += 1

        quarter += 1
        if quarter == 5:
            year += 1
            quarter = 1

    return mortgage_run_params_dict_list
