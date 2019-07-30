'''
Collection of functions to run the mortgage example.
'''
import os
from glob import glob


class MortgageTaskNames(object):
    '''Task names commonly used by scripts for naming tasks when creating
    a gQuant mortgage workflow.
    '''
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


def mortgage_etl_workflow_def(
        csvfile_names=None, csvfile_acqdata=None,
        csvfile_perfdata=None):
    '''Define the ETL (extract-transform-load) portion of the mortgage
    workflow.

    :returns: gQuant workflow. Currently a simple list of dictionaries. Each
        dict specifies a task per TaskSpecSchema.
    :rtype: list
    '''
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
    '''For the specified years and limit (part_count) to the number of files
    (performance files), generates a list of run_params_dict.
        run_params_dict = {
            'replace_spec': replace_spec,
            'task_list': gquant_task_list,
            'out_list': out_list
        }

    replace_spec - to be passed to Dataframe flow run command's replace option.
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

    out_list - Expected to specify one output which should be the final
        dataframe produced by the mortgage ETL workflow.

    Example:
        import gquant.dataframe_flow as dff
        task_list = run_params_dict['task_list']
        out_list = run_params_dict['out_list']
        replace_spec = run_params_dict['replace_spec']
        (final_perf_acq_df,) = dff.run(task_list, out_list, replace_spec)

    :param str mortgage_data_path: Path to mortgage data. Should have a file
        "names.csv" and two subdirectories "acq" and "perf".

    :param int start_year: Start year is used to traverse the appropriate range
        of directories with corresponding year(s) in mortgage data.

    :param int end_year: End year is used to traverse the appropriate range
        of directories with corresponding year(s) in mortgage data.

    :param int part_count: Limit to how many performance files to load. There
        is a single corresponding acquisition file for year and quarter.
        Performance files are very large csv files (1GB files) and are broken
        down i.e. for a given year and quarter you could have several file
        chunks: *.txt_0, *.txt_1, etc.

    :param gquant_task_list: Mortgage ETL workflow list of tasks. Refer to
        function mortgage_etl_workflow_def.

    :returns: list of run_params_dict
    :rtype: list

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
