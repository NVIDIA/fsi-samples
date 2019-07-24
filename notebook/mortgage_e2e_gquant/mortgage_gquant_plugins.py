'''
'''
import sys
from collections import OrderedDict
import re
import numpy as np
from gquant.dataframe_flow import Node

import logging

# logging.config.dictConfig({
#     'version': 1,
#     'disable_existing_loggers': False
# })

_DISTRIB_FORMATTER = None


def init_workers_logger():
    '''Initialize logger within all workers. Meant to be run as:
        client.run(init_workers_logger)
    '''
    global _DISTRIB_FORMATTER

    distrib_logger = logging.getLogger('distributed.worker')
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d %(name)s:%(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    if _DISTRIB_FORMATTER is None:
        _DISTRIB_FORMATTER = distrib_logger.handlers[0].formatter

    distrib_logger.handlers[0].setFormatter(formatter)


def restore_workers_logger():
    '''Restore logger within all workers. Meant to be run as:
        client.run(restore_workers_logger)

    Run this after printing worker logs i.e. after:
        wlogs = client.get_worker_logs()
        # print entries form wlogs

    '''
    global _DISTRIB_FORMATTER

    distrib_logger = logging.getLogger('distributed.worker')
    if _DISTRIB_FORMATTER is not None:
        distrib_logger.handlers[0].setFormatter(_DISTRIB_FORMATTER)
        _DISTRIB_FORMATTER = None


_CONFIGLOG = True


class MortgagePluginsLoggerMgr(object):
    '''Logger manager for gQuant mortgage plugins.

    When using this log manager to hijack dask distributed.worker logger
    (worker is not None), must first initialize worker loggers via:
        client.run(init_workers_logger)
    Afer printing out entries from worker logs restore worker loggers via:
        client.run(restore_workers_logger)

    WARNING: HIJACKING Dask Distributed logger within dask-workers!!! This
    is NOT a great implementation. Done to capture and display logs in Jupyter.
    TODO: Implement a server/client logger per example:
        https://docs.python.org/3/howto/logging-cookbook.html#sending-and-receiving-logging-events-across-a-network


    '''

    def __init__(self, worker=None, logname='mortgage_plugins'):
        if worker is None:
            logger = self._get_mortgage_plugins_logger()
            console_handler = None
        else:
            # WARNING: HIJACKING Dask Distributed logger!!!

            logger = logging.getLogger('distributed.worker.' + logname)

            console_handler = self._config_log_handler(
                logger, propagate=True, addtimestamp=True)

        self._logger = logger
        self._console_handler = console_handler

    @staticmethod
    def _config_log_handler(logger, propagate=True, addtimestamp=False):
        '''Configure logger handler with streaming to stdout and formatter. Add
        the handler to the logger.
        '''

        if addtimestamp:
            formatter = logging.Formatter(
                '%(asctime)s.%(msecs)03d %(name)s:%(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(name)s:%(levelname)s: %(message)s')

        console_handler = logging.StreamHandler(sys.stdout)  # console handeler
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = propagate

        # logger.info('CONFIGURING LOGGER')

        return console_handler

    @classmethod
    def _get_mortgage_plugins_logger(cls):
        '''Obtain a logger for mortgage plugins. Used when the running process
        is not a dask-worker.
        '''
        logger = logging.getLogger(__name__)

        global _CONFIGLOG

        if _CONFIGLOG:
            cls._config_log_handler(logger, propagate=False)
            _CONFIGLOG = False

        # Should only be one handler. With Dask there's a race condition and
        # could have multiple logging handlers.
        while len(logger.handlers) > 1:
            logger.handlers.pop()

        return logger

    def get_logger(self):
        '''Get the logger being managed by instante of this log manager.'''
        return self._logger

    def cleanup(self):
        '''Clean up the logger.'''
        if self._console_handler is not None:
            self._logger.removeHandler(self._console_handler)


first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')


def convert(name):
    '''Convert CamelCase to snake_case.
    https://stackoverflow.com/a/1176023/3457624
    '''
    s1 = first_cap_re.sub(r'\1_\2', name)
    return all_cap_re.sub(r'\1_\2', s1).lower()


class CsvMortgageAcquisitionDataLoader(Node):
    '''gQuant task/node to read in a mortgage acquisition CSV file into a cudf
    dataframe. Configuration requirements:
        'conf': {
            'csvfile_names': path to mortgage seller names csv datafile
            'csvfile_acqdata': path to mortgage acquisition csv datafile
        }
    '''

    cols_dtypes = OrderedDict([
        ('loan_id', 'int64'),
        # ('orig_channel', 'category'),
        ('orig_channel', 'int32'),
        # ('seller_name', 'category'),
        ('seller_name', 'int32'),
        ('orig_interest_rate', 'float64'),
        ('orig_upb', 'int64'),
        ('orig_loan_term', 'int64'),
        ('orig_date', 'date'),
        ('first_pay_date', 'date'),
        ('orig_ltv', 'float64'),
        ('orig_cltv', 'float64'),
        ('num_borrowers', 'float64'),
        ('dti', 'float64'),
        ('borrower_credit_score', 'float64'),
        # ('first_home_buyer', 'category'),
        ('first_home_buyer', 'int32'),
        # ('loan_purpose', 'category'),
        ('loan_purpose', 'int32'),
        # ('property_type', 'category'),
        ('property_type', 'int32'),
        ('num_units', 'int64'),
        # ('occupancy_status', 'category'),
        ('occupancy_status', 'int32'),
        # ('property_state', 'category'),
        ('property_state', 'int32'),
        ('zip', 'int64'),
        ('mortgage_insurance_percent', 'float64'),
        # ('product_type', 'category'),
        ('product_type', 'int32'),
        ('coborrow_credit_score', 'float64'),
        ('mortgage_insurance_type', 'float64'),
        # ('relocation_mortgage_indicator', 'category')
        ('relocation_mortgage_indicator', 'int32')
    ])

    def columns_setup(self):
        self.addition = self.cols_dtypes

    def process(self, inputs):
        '''
        '''
        import cudf

        worker = None
        try:
            from dask.distributed import get_worker
            worker = get_worker()
        except (ValueError, ImportError):
            pass

        logname = convert(self.__class__.__name__)
        logmgr = MortgagePluginsLoggerMgr(worker, logname)
        logger = logmgr.get_logger()

        worker_name = ''
        if worker is not None:
            worker_name = 'WORKER {} '.format(worker.name)

        col_names_path = self.conf['csvfile_names']
        cols_dtypes = OrderedDict([
            ('seller_name', 'category'),
            ('new', 'category'),
        ])
        cols = list(cols_dtypes.keys())
        dtypes = list(cols_dtypes.values())

        names_gdf = cudf.read_csv(
            col_names_path,
            names=cols, dtype=dtypes,
            delimiter='|', skiprows=1)

        acquisition_path = self.conf['csvfile_acqdata']
        cols = list(self.addition.keys())
        dtypes = list(self.addition.values())

        logger.info(worker_name + 'LOADING: {}'.format(acquisition_path))
        acq_gdf = cudf.read_csv(
            acquisition_path,
            names=cols, dtype=dtypes,
            delimiter='|', skiprows=1)

        acq_gdf = acq_gdf.merge(names_gdf, how='left', on=['seller_name'])
        acq_gdf['seller_name'] = acq_gdf['new']
        acq_gdf.drop_column('new')

        logmgr.cleanup()

        return acq_gdf


class CsvMortgagePerformanceDataLoader(Node):
    '''gQuant task/node to read in a mortgage performance CSV file into a cudf
    dataframe. Configuration requirements:
        'conf': {
            'csvfile_perfdata': path to mortgage performance csv datafile
        }
    '''

    cols_dtypes = OrderedDict([
        ('loan_id', 'int64'),
        ('monthly_reporting_period', 'date'),
        # ('servicer', 'category'),
        ('servicer', 'int32'),
        ('interest_rate', 'float64'),
        ('current_actual_upb', 'float64'),
        ('loan_age', 'float64'),
        ('remaining_months_to_legal_maturity', 'float64'),
        ('adj_remaining_months_to_maturity', 'float64'),
        ('maturity_date', 'date'),
        ('msa', 'float64'),
        ('current_loan_delinquency_status', 'int32'),
        # ('mod_flag', 'category'),
        ('mod_flag', 'int32'),
        # ('zero_balance_code', 'category'),
        ('zero_balance_code', 'int32'),
        ('zero_balance_effective_date', 'date'),
        ('last_paid_installment_date', 'date'),
        ('foreclosed_after', 'date'),
        ('disposition_date', 'date'),
        ('foreclosure_costs', 'float64'),
        ('prop_preservation_and_repair_costs', 'float64'),
        ('asset_recovery_costs', 'float64'),
        ('misc_holding_expenses', 'float64'),
        ('holding_taxes', 'float64'),
        ('net_sale_proceeds', 'float64'),
        ('credit_enhancement_proceeds', 'float64'),
        ('repurchase_make_whole_proceeds', 'float64'),
        ('other_foreclosure_proceeds', 'float64'),
        ('non_interest_bearing_upb', 'float64'),
        ('principal_forgiveness_upb', 'float64'),
        # ('repurchase_make_whole_proceeds_flag', 'category'),
        ('repurchase_make_whole_proceeds_flag', 'int32'),
        ('foreclosure_principal_write_off_amount', 'float64'),
        # ('servicing_activity_indicator', 'category')
        ('servicing_activity_indicator', 'int32')
    ])

    def columns_setup(self):
        self.addition = self.cols_dtypes

    def process(self, inputs):
        '''
        '''
        import cudf

        worker = None
        try:
            from dask.distributed import get_worker
            worker = get_worker()
        except (ValueError, ImportError):
            pass

        logname = convert(self.__class__.__name__)
        logmgr = MortgagePluginsLoggerMgr(worker, logname)
        logger = logmgr.get_logger()

        worker_name = ''
        if worker is not None:
            worker_name = 'WORKER {} '.format(worker.name)

        performance_path = self.conf['csvfile_perfdata']
        logger.info(worker_name + 'LOADING: {}'.format(performance_path))

        cols = list(self.addition.keys())
        dtypes = list(self.addition.values())
        mortgage_gdf = cudf.read_csv(
            performance_path,
            names=cols, dtype=dtypes,
            delimiter='|', skiprows=1)

        logmgr.cleanup()

        return mortgage_gdf


class CreateEverFeatures(Node):
    '''gQuant task/node to calculate delinquecy status period features.
    Refer to columns_setup method for the columns produced.
    '''
    def columns_setup(self):
        self.required = OrderedDict([
            ('loan_id', 'int64'),
            ('current_loan_delinquency_status', 'int32')
        ])

        self.retention = {
            'loan_id': 'int64',
            'ever_30': 'int8',
            'ever_90': 'int8',
            'ever_180': 'int8'
        }

    def process(self, inputs):
        '''
        '''
        gdf = inputs[0]
        everdf = gdf[['loan_id', 'current_loan_delinquency_status']]
        everdf = everdf.groupby('loan_id', method='hash', as_index=False).max()
        everdf['ever_30'] = \
            (everdf['current_loan_delinquency_status'] >= 1).astype('int8')
        everdf['ever_90'] = \
            (everdf['current_loan_delinquency_status'] >= 3).astype('int8')
        everdf['ever_180'] = \
            (everdf['current_loan_delinquency_status'] >= 6).astype('int8')
        everdf.drop_column('current_loan_delinquency_status')

        return everdf


class CreateDelinqFeatures(Node):
    '''gQuant task/node to calculate delinquecy features.
    Refer to columns_setup method for the columns produced.
    '''
    def columns_setup(self):
        self.required = OrderedDict([
            ('loan_id', 'int64'),
            ('monthly_reporting_period', 'date'),
            ('current_loan_delinquency_status', 'int32')
        ])

        self.retention = {
            'loan_id': 'int64',
            'delinquency_30': 'date',
            'delinquency_90': 'date',
            'delinquency_180': 'date'
        }

    def process(self, inputs):
        '''
        '''
        perf_df = inputs[0]
        delinq_gdf = perf_df[[
            'loan_id', 'monthly_reporting_period',
            'current_loan_delinquency_status']]

        delinq_30 = delinq_gdf.query('current_loan_delinquency_status >= 1')[[
            'loan_id', 'monthly_reporting_period']]\
            .groupby('loan_id', method='hash', as_index=False).min()
        delinq_30['delinquency_30'] = delinq_30['monthly_reporting_period']
        delinq_30.drop_column('monthly_reporting_period')

        delinq_90 = delinq_gdf.query('current_loan_delinquency_status >= 3')[[
            'loan_id', 'monthly_reporting_period']]\
            .groupby('loan_id', method='hash', as_index=False).min()
        delinq_90['delinquency_90'] = delinq_90['monthly_reporting_period']
        delinq_90.drop_column('monthly_reporting_period')

        delinq_180 = delinq_gdf.query('current_loan_delinquency_status >= 6')[[
            'loan_id', 'monthly_reporting_period']]\
            .groupby('loan_id', method='hash', as_index=False).min()
        delinq_180['delinquency_180'] = delinq_180['monthly_reporting_period']
        delinq_180.drop_column('monthly_reporting_period')

        delinq_merge = delinq_30.merge(
            delinq_90, how='left', on=['loan_id'], type='hash')
        delinq_merge['delinquency_90'] = delinq_merge['delinquency_90']\
            .fillna(np.dtype('datetime64[ms]').type('1970-01-01')
                    .astype('datetime64[ms]'))

        delinq_merge = delinq_merge.merge(
            delinq_180, how='left', on=['loan_id'], type='hash')
        delinq_merge['delinquency_180'] = delinq_merge['delinquency_180']\
            .fillna(np.dtype('datetime64[ms]').type('1970-01-01')
                    .astype('datetime64[ms]'))

        del(delinq_30)
        del(delinq_90)
        del(delinq_180)

        return delinq_merge


class JoinPerfEverDelinqFeatures(Node):
    '''gQuant task/node to merge delinquecy features. Merges dataframes
    produced by CreateEverFeatures and CreateDelinqFeatures.
    Refer to columns_setup method for the columns produced.
    '''

    cols_dtypes = {
        'timestamp': 'date',

        'delinquency_12': 'int32',
        'upb_12': 'float64',

        'ever_30': 'int8',
        'ever_90': 'int8',
        'ever_180': 'int8',
        'delinquency_30': 'date',
        'delinquency_90': 'date',
        'delinquency_180': 'date'
    }

    def columns_setup(self):
        '''
        '''
        self.retention = {
            'loan_id': 'int64',

            'timestamp_month': 'int32',
            'timestamp_year': 'int32'
        }
        self.retention.update(self.cols_dtypes)

    def __join_ever_delinq_features(self, everdf_in, delinqdf_in):
        everdf = everdf_in.merge(
            delinqdf_in, on=['loan_id'], how='left', type='hash')
        everdf['delinquency_30'] = everdf['delinquency_30']\
            .fillna(np.dtype('datetime64[ms]').type('1970-01-01')
                    .astype('datetime64[ms]'))
        everdf['delinquency_90'] = everdf['delinquency_90']\
            .fillna(np.dtype('datetime64[ms]').type('1970-01-01')
                    .astype('datetime64[ms]'))
        everdf['delinquency_180'] = everdf['delinquency_180']\
            .fillna(np.dtype('datetime64[ms]').type('1970-01-01')
                    .astype('datetime64[ms]'))

        return everdf

    def process(self, inputs):
        '''
        '''
        perf_df = inputs[0]
        # if using JoinEverDelinqFeatures. Seems unnecessary
        # ever_delinq_df = inputs[1]
        everdf_in = inputs[1]
        delinqdf_in = inputs[2]

        ever_delinq_df = \
            self.__join_ever_delinq_features(everdf_in, delinqdf_in)

        test = perf_df[[
            'loan_id',
            'monthly_reporting_period',
            'current_loan_delinquency_status',
            'current_actual_upb'
        ]]
        test['timestamp'] = test['monthly_reporting_period']
        test.drop_column('monthly_reporting_period')
        test['timestamp_month'] = test['timestamp'].dt.month
        test['timestamp_year'] = test['timestamp'].dt.year
        test['delinquency_12'] = test['current_loan_delinquency_status']
        test.drop_column('current_loan_delinquency_status')
        test['upb_12'] = test['current_actual_upb']
        test.drop_column('current_actual_upb')
        test['upb_12'] = test['upb_12'].fillna(999999999)
        test['delinquency_12'] = test['delinquency_12'].fillna(-1)

        joined_df = test.merge(
            ever_delinq_df, how='left', on=['loan_id'], type='hash')

        joined_df['ever_30'] = joined_df['ever_30'].fillna(-1)
        joined_df['ever_90'] = joined_df['ever_90'].fillna(-1)
        joined_df['ever_180'] = joined_df['ever_180'].fillna(-1)
        joined_df['delinquency_30'] = joined_df['delinquency_30'].fillna(-1)
        joined_df['delinquency_90'] = joined_df['delinquency_90'].fillna(-1)
        joined_df['delinquency_180'] = joined_df['delinquency_180'].fillna(-1)

        joined_df['timestamp_month'] = \
            joined_df['timestamp_month'].astype('int32')
        joined_df['timestamp_year'] = \
            joined_df['timestamp_year'].astype('int32')

        return joined_df


class Create12MonFeatures(Node):
    '''gQuant task/node to calculate delinquecy feature over 12 months.
    Refer to columns_setup method for the columns produced.
    '''
    def columns_setup(self):
        '''
        '''
        self.retention = {
            'loan_id': 'int64',
            'delinquency_12': 'int32',
            'upb_12': 'float64',
            'timestamp_month': 'int8',
            'timestamp_year': 'int16'
        }

    def process(self, inputs):
        '''
        '''
        import cudf

        perf_ever_delinq_df = inputs[0]

        testdfs = []
        n_months = 12
        for y in range(1, n_months + 1):
            tmpdf = perf_ever_delinq_df[[
                'loan_id', 'timestamp_year', 'timestamp_month',
                'delinquency_12', 'upb_12'
            ]]

            tmpdf['josh_months'] = \
                tmpdf['timestamp_year'] * 12 + tmpdf['timestamp_month']

            tmpdf['josh_mody_n'] = \
                ((tmpdf['josh_months'].astype('float64') - 24000 - y) / 12)\
                .floor()

            tmpdf = tmpdf.groupby(
                ['loan_id', 'josh_mody_n'], method='hash', as_index=False)\
                .agg({'delinquency_12': 'max', 'upb_12': 'min'})

            tmpdf['delinquency_12'] = \
                (tmpdf['max_delinquency_12'] > 3).astype('int32')
            tmpdf.drop_column('max_delinquency_12')

            tmpdf['delinquency_12'] += \
                (tmpdf['min_upb_12'] == 0).astype('int32')

            tmpdf['upb_12'] = tmpdf['min_upb_12']
            tmpdf.drop_column('min_upb_12')

            tmpdf['timestamp_year'] = \
                (((tmpdf['josh_mody_n'] * n_months) + 24000 + (y - 1)) / 12)\
                .floor().astype('int16')
            tmpdf.drop_column('josh_mody_n')

            tmpdf['timestamp_month'] = np.int8(y)

            testdfs.append(tmpdf)

        test_12mon_feat_df = cudf.concat(testdfs)
        return test_12mon_feat_df


def _null_workaround(df):
    '''Fix up null entries in dataframes. This is specific to the mortgage
    workflow.
    '''
    for column, data_type in df.dtypes.items():
        if str(data_type) == "category":
            df[column] = df[column].astype('int32').fillna(-1)
        if str(data_type) in \
                ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
            df[column] = df[column].fillna(np.dtype(data_type).type(-1))
    return df


class FinalPerfDelinq(Node):
    '''Merge performance dataframe with calculated features dataframes.
    Refer to columns_setup method for the columns produced.
    '''

    cols_dtypes = dict()
    cols_dtypes.update(CsvMortgagePerformanceDataLoader.cols_dtypes)
    cols_dtypes.update(JoinPerfEverDelinqFeatures.cols_dtypes)

    def columns_setup(self):
        '''
        '''
        self.retention = self.cols_dtypes

    @staticmethod
    def __combine_joined_12_mon(perf_ever_delinq_df, test_12mon_df):
        perf_ever_delinq_df.drop_column('delinquency_12')
        perf_ever_delinq_df.drop_column('upb_12')
        perf_ever_delinq_df['timestamp_year'] = \
            perf_ever_delinq_df['timestamp_year'].astype('int16')
        perf_ever_delinq_df['timestamp_month'] = \
            perf_ever_delinq_df['timestamp_month'].astype('int8')

        return perf_ever_delinq_df.merge(
            test_12mon_df,
            how='left',
            on=['loan_id', 'timestamp_year', 'timestamp_month'],
            type='hash')

    @classmethod
    def __final_performance_delinquency(
            cls, perf_df, perf_ever_delinq_df, test_12mon_df):

        joined_df = \
            cls.__combine_joined_12_mon(perf_ever_delinq_df, test_12mon_df)

        merged = _null_workaround(perf_df)
        joined_df = _null_workaround(joined_df)
        joined_df['timestamp_month'] = \
            joined_df['timestamp_month'].astype('int8')
        joined_df['timestamp_year'] = \
            joined_df['timestamp_year'].astype('int16')
        merged['timestamp_month'] = merged['monthly_reporting_period'].dt.month
        merged['timestamp_month'] = merged['timestamp_month'].astype('int8')
        merged['timestamp_year'] = merged['monthly_reporting_period'].dt.year
        merged['timestamp_year'] = merged['timestamp_year'].astype('int16')
        merged = merged.merge(
            joined_df, how='left',
            on=['loan_id', 'timestamp_year', 'timestamp_month'], type='hash')

        merged.drop_column('timestamp_month')
        merged.drop_column('timestamp_year')

        return merged

    def process(self, inputs):
        '''
        '''
        perf_df = inputs[0].copy()
        perf_ever_delinq_df = inputs[1].copy()
        test_12mon_df = inputs[2]

        final_perf_df = self.__final_performance_delinquency(
            perf_df, perf_ever_delinq_df, test_12mon_df)

        return final_perf_df


class JoinFinalPerfAcqClean(Node):
    '''Merge acquisition dataframe with dataframe produced by FinalPerfDelinq.
    Refer to columns_setup method for the columns produced.
    '''
    _drop_list = [
        'loan_id',
        'orig_date',
        'first_pay_date',
        'seller_name',
        'monthly_reporting_period',
        'last_paid_installment_date',
        'maturity_date',
        'ever_30', 'ever_90', 'ever_180',
        'delinquency_30', 'delinquency_90', 'delinquency_180',
        'upb_12',
        'zero_balance_effective_date',
        'foreclosed_after',
        'disposition_date',
        'timestamp'
    ]

    cols_dtypes = dict()
    cols_dtypes.update(FinalPerfDelinq.cols_dtypes)
    cols_dtypes.update(CsvMortgageAcquisitionDataLoader.cols_dtypes)

    # all float64, int32 and int64 types are converted to float32 types.
    for icol, itype in cols_dtypes.items():
        if itype in ('float64', 'int32', 'int64',):
            cols_dtypes[icol] = 'float32'

    # The only exception is delinquency_12 which remains int32
    cols_dtypes.update({'delinquency_12': 'int32'})

    for col in _drop_list:
        cols_dtypes.pop(col)

    def columns_setup(self):
        '''
        '''
        self.retention = self.cols_dtypes

    @classmethod
    def __last_mile_cleaning(cls, df):
        drop_list = cls._drop_list
        for column in drop_list:
            df.drop_column(column)
        for col, dtype in df.dtypes.iteritems():
            if str(dtype) == 'category':
                df[col] = df[col].cat.codes
            df[col] = df[col].astype('float32')
        df['delinquency_12'] = df['delinquency_12'] > 0
        df['delinquency_12'] = \
            df['delinquency_12'].fillna(False).astype('int32')
        for column in df.columns:
            df[column] = \
                df[column].fillna(np.dtype(str(df[column].dtype)).type(-1))

        # return df.to_arrow(preserve_index=False)
        return df

    def process(self, inputs):
        '''
        '''
        perf_df = inputs[0].copy()
        acq_df = inputs[1].copy()

        perf_df = _null_workaround(perf_df)
        acq_df = _null_workaround(acq_df)

        perf_acq_df = perf_df.merge(
            acq_df, how='left', on=['loan_id'], type='hash')

        perf_acq_df = self.__last_mile_cleaning(perf_acq_df)

        return perf_acq_df


def mortgage_gquant_run(run_params_dict):
    '''Using dataframe-flow runs the tasks/workflow specified in the
    run_params_dict. Expected run_params_dict ex:
        run_params_dict = {
            'replace_spec': replace_spec,
            'task_list': gquant_task_list,
            'out_list': out_list
        }

    gquant_task_list - Mortgage ETL workflow list of tasks. Refer to module
        mortgage_common function mortgage_etl_workflow_def.

    out_list - Expected to specify one output which should be the final
        dataframe produced by the mortgage ETL workflow.

    :param run_params_dict: Dictionary with parameters and gquant task list to
        run mortgage workflow.

    '''
    import gquant.dataframe_flow as dff

    task_list = run_params_dict['task_list']
    out_list = run_params_dict['out_list']

    replace_spec = run_params_dict['replace_spec']

    (final_perf_acq_df,) = dff.run(task_list, out_list, replace_spec)

    return final_perf_acq_df


def print_ram_usage(worker_name='', logger=None):
    '''Display host RAM usage on the system using free -m command.'''
    import os

    logmgr = None
    if logger is None:
        logmgr = MortgagePluginsLoggerMgr()
        logger = logmgr.get_logger()

    tot_m, used_m, free_m = \
        map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    logger.info(
        worker_name + 'HOST RAM (MB) TOTAL {}; USED {}; FREE {}'
        .format(tot_m, used_m, free_m))

    if logmgr is not None:
        logmgr.cleanup()


def mortgage_workflow_runner(mortgage_run_params_dict_list):
    '''Runs the mortgage_gquant_run for each entry in the
    mortgage_run_params_dict_list. Each entry is a run_params_dict.
    Expected run_params_dict:
        run_params_dict = {
            'replace_spec': replace_spec,
            'task_list': gquant_task_list,
            'out_list': out_list
        }

    :param mortgage_run_params_dict_list: List of run_params_dict

    '''
    import os  # @Reimport
    import gc
    import pyarrow as pa

    # count = len(mortgage_run_params_dict_list)

    # print('LOGGER: ', logger)

    worker = None
    try:
        from dask.distributed import get_worker
        worker = get_worker()
    except (ValueError, ImportError):
        pass

    logname = 'mortgage_workflow_runner'
    logmgr = MortgagePluginsLoggerMgr(worker, logname)
    logger = logmgr.get_logger()

    worker_name = ''
    if worker is not None:
        worker_name = 'WORKER {} '.format(worker.name)
        logger.info(worker_name + 'RUNNING MORTGAGE gQUANT DataframeFlow')
        logger.info(worker_name + 'NCCL_P2P_DISABLE: {}'.format(
            os.environ.get('NCCL_P2P_DISABLE')))
        logger.info(worker_name + 'CUDA_VISIBLE_DEVICES: {}'.format(
            os.environ.get('CUDA_VISIBLE_DEVICES')))

    # cpu_df_concat_pandas = None
    final_perf_acq_arrow_concat = None
    for ii, run_params_dict in enumerate(mortgage_run_params_dict_list):
        # performance_path = run_params_dict['csvfile_perfdata']
        # logger.info(worker_name + 'LOADING: {}'.format(performance_path))

        final_perf_acq_gdf = mortgage_gquant_run(run_params_dict)

        # CONCATENATE DATAFRAMES AS THEY ARE CALCULATED

        # cpu_df_pandas = gpu_df.to_pandas()
        # if cpu_df_concat_pandas is None:
        #     cpu_df_concat_pandas = cpu_df_pandas
        # else:
        #     cpu_df_concat_pandas = \
        #         pd.concat([cpu_df_concat_pandas, cpu_df_pandas])
        #     del(cpu_df_pandas)

        final_perf_acq_arrow = \
            final_perf_acq_gdf.to_arrow(preserve_index=False)
        if final_perf_acq_arrow_concat is None:
            final_perf_acq_arrow_concat = final_perf_acq_arrow
        else:
            final_perf_acq_arrow_concat = pa.concat_tables([
                final_perf_acq_arrow_concat, final_perf_acq_arrow])

        del(final_perf_acq_gdf)
        logger.info(worker_name + 'LOADED {} FRAMES'.format(ii + 1))

    print_ram_usage(worker_name, logger)
    logger.info(worker_name + 'RUN PYTHON GARBAGE COLLECTION TO MAYBE CLEAR '
                'CPU AND GPU MEMORY')

    gc.collect()
    print_ram_usage(worker_name, logger)

    # df_concat = cpu_df_concat_pandas
    # delinq_df = df_concat[['delinquency_12']]
    # indexes_besides_delinq = \
    #     df_concat.columns.difference(['delinquency_12'])
    # mortgage_feat_df = df_concat[list(indexes_besides_delinq)]
    # del(df_concat)

    logger.info(worker_name + 'USING ARROW')

    cpu_df_concat_arrow = final_perf_acq_arrow_concat
    delinq_arrow_col = cpu_df_concat_arrow.column('delinquency_12')
    mortgage_feat_arrow_table = cpu_df_concat_arrow.drop(['delinquency_12'])

    # logger.info(worker_name + 'ARROW TO CUDF')
    # delinq_arrow_table = pa.Table.from_arrays([delinq_arrow_col])
    # delinq_df = cudf.DataFrame.from_arrow(delinq_arrow_table)
    # mortgage_feat_df = cudf.DataFrame.from_arrow(mortgage_feat_arrow_table)

    logger.info(worker_name + 'ARROW TO PANDAS')
    delinq_df = delinq_arrow_col.to_pandas()
    mortgage_feat_df = mortgage_feat_arrow_table.to_pandas()
    del(delinq_arrow_col)
    del(mortgage_feat_arrow_table)

    # clear CPU/GPU memory
    gc.collect()

    print_ram_usage(worker_name, logger)

    logmgr.cleanup()

    return (mortgage_feat_df, delinq_df)


class MortgageWorkflowRunner(Node):
    '''Runs the mortgage gquant workflow and returns the mortgage features
    dataframe and mortgage delinquency dataframe. These can be passed on
    to xgboost for training.

    conf: {
        'mortgage_run_params_dict_list': REQUIRED. List of dictionaries of
            mortgage run params.
    }

        mortgage_run_param_dict = {
            'replace_spec': replace_spec,
            'task_list': gquant_task_list,
            'out_list': out_list
        }

    Returns: mortgage_feat_df_pandas, delinq_df_pandas
        DataframeFlow will return a tuple so unpack as tuple of tuples:
            ((mortgage_feat_df_pandas, delinq_df_pandas),)

    '''
    def columns_setup(self):
        '''
        '''
        pass

    def process(self, inputs):
        logmgr = MortgagePluginsLoggerMgr()
        logger = logmgr.get_logger()

        mortgage_run_params_dict_list = \
            self.conf['mortgage_run_params_dict_list']

        count = len(mortgage_run_params_dict_list)
        logger.info('TRYING TO LOAD {} FRAMES'.format(count))

        mortgage_feat_df_pandas, delinq_df_pandas = \
            mortgage_workflow_runner(mortgage_run_params_dict_list)

        logmgr.cleanup()

        return mortgage_feat_df_pandas, delinq_df_pandas


class XgbMortgageTrainer(Node):
    '''Trains an XGBoost booster.

    Configuration:
        conf: {
            'delete_dataframes': OPTIONAL. Boolean (True or False). Delete the
                intermediate mortgage dataframes from which an xgboost dmatrix
                is created. This is to potentially clear up CPU/GPU memory.
            'xgb_gpu_params': REQUIRED. Dictionary of xgboost trainer
                parameters.
        }

        Example of xgb_gpu_params:
            xgb_gpu_params = {
                'nround':            100,
                'max_depth':         8,
                'max_leaves':        2 ** 8,
                'alpha':             0.9,
                'eta':               0.1,
                'gamma':             0.1,
                'learning_rate':     0.1,
                'subsample':         1,
                'reg_lambda':        1,
                'scale_pos_weight':  2,
                'min_child_weight':  30,
                'tree_method':       'gpu_hist',
                'n_gpus':            1,
                'loss':              'ls',
                # 'objective':         'gpu:reg:linear',
                'objective':         'reg:squarederror',
                'max_features':      'auto',
                'criterion':         'friedman_mse',
                'grow_policy':       'lossguide',
                'verbose':           True
            }

    Inputs:
        mortgage_feat_df_pandas, delinq_df_pandas = inputs[0]
        These inputs are provided by MortgageWorkflowRunner.

    Outputs:
        bst - XGBoost trained booster model.

    '''
    def columns_setup(self):
        '''
        '''
        pass

    def process(self, inputs):
        import gc  # python standard lib garbage collector
        import xgboost as xgb

        logmgr = MortgagePluginsLoggerMgr()
        logger = logmgr.get_logger()

        mortgage_feat_df_pandas, delinq_df_pandas = inputs[0]

        delete_dataframes = self.conf.get('delete_dataframes')
        xgb_gpu_params = self.conf['xgb_gpu_params']

        logger.info('JUST BEFORE DMATRIX')
        print_ram_usage()

        logger.info('CREATING DMATRIX')
        # DMatrix directly from dataframe requires xgboost from rapidsai:
        #     https://github.com/rapidsai/xgboost
        # Convert to DMatrix for XGBoost training.
        xgb_dmatrix = xgb.DMatrix(mortgage_feat_df_pandas, delinq_df_pandas)
        # logger.info('XGB_DMATRIX:\n', xgb_dmatrix)

        logger.info('JUST AFTER DMATRIX')
        print_ram_usage()

        # clear CPU/GPU memory
        if delete_dataframes:
            del(mortgage_feat_df_pandas)
            del(delinq_df_pandas)

        gc.collect()

        logger.info('CLEAR MEMORY JUST BEFORE XGBOOST TRAINING')
        print_ram_usage()

        logger.info('RUNNING XGBOOST TRAINING')

        # booster object
        bst = xgb.train(
            xgb_gpu_params, xgb_dmatrix,
            num_boost_round=xgb_gpu_params['nround'])

        logmgr.cleanup()

        return bst


# RMM - RAPIDS Memory Manager.
# IMPORTANT!!! IF USING RMM START CLIENT prior to any cudf imports and that
# means prior to any gQuant imports, 3rd party libs with cudf, etc.
# This is needed if distributing workflows to workers.

def initialize_rmm_pool():
    from librmm_cffi import librmm_config as rmm_cfg

    rmm_cfg.use_pool_allocator = True
    # set to 2GiB. Default is 1/2 total GPU memory
    # rmm_cfg.initial_pool_size = 2 << 30
    # rmm_cfg.initial_pool_size = 2 << 5
    # rmm_cfg.initial_pool_size = 2 << 33
    import cudf
    return cudf.rmm.initialize()


def initialize_rmm_no_pool():
    from librmm_cffi import librmm_config as rmm_cfg

    rmm_cfg.use_pool_allocator = False
    import cudf
    return cudf.rmm.initialize()


def finalize_rmm():
    import cudf
    return cudf.rmm.finalize()


def print_distributed_dask_hijacked_logs(wlogs, logger, filters=None):
    '''Prints (uses logger.info) the log entries from worker logs
    (wlogs = client.get_worker_logs()). Filters what is printed based on
    keywords in the filters. If filters is None then prints everything.

    :param filters: A tuple. Even if one entry ('somestr',)
    '''
    # print('WORKER LOGS:\n{}'.format(json.dumps(wlogs, indent=2)))

    for iworker_log in wlogs.values():
        for _, msg in iworker_log:
            # if 'distributed.worker.' in msg:
            # if filter in msg:
            if filters is None:
                logger.info(msg)
                continue

            if any(ff in msg for ff in filters):
                logger.info(msg)


class DaskMortgageWorkflowRunner(Node):
    '''Runs the mortgage gquant workflow and returns the mortgage features
    dataframe and mortgage delinquency dataframe. These can be passed on
    to xgboost for training.

    conf: {
        'mortgage_run_params_dict_list': REQUIRED. List of dictionaries of
            mortgage run params.
        'client': REQUIRED. Dask distributed client. Runs with distributed
            dask.
        'use_rmm': OPTIONAL. Boolean (True or False). Use RAPIDS Memory
            Manager.,
        'filter_dask_logger': OPTIONAL. Boolean to display hijacked
            dask.distributed log. If False (default) then doesn't display.
    }

    Format of expected mortgage run params:
        mortgage_run_param_dict = {
            'replace_spec': replace_spec,
            'task_list': gquant_task_list,
            'out_list': out_list
        }

    Returns: dask-distributed Futures where each future holds a tuple:
            mortgage_feat_df_pandas, delinq_df_pandas
        The number of futures returned corresponds to the number of workers
        obtained from the client.
        DataframeFlow will return a tuple so unpack as tuple of tuples in
        whatever operates on the future:
            ((mortgage_feat_df_pandas, delinq_df_pandas),)

    '''
    def columns_setup(self):
        '''
        '''
        pass

    def process(self, inputs):
        from dask.distributed import wait

        logmgr = MortgagePluginsLoggerMgr()
        logger = logmgr.get_logger()

        filter_dask_logger = self.conf.get('filter_dask_logger')

        client = self.conf['client']
        client.run(init_workers_logger)

        use_rmm = self.conf.get('use_rmm')
        if use_rmm:
            rmm_init_results = client.run(initialize_rmm_pool)
            logger.info('RMM INIT RESULTS:\n', rmm_init_results)

        mortgage_run_params_dict_list = \
            self.conf['mortgage_run_params_dict_list']

        workers_names = \
            [iw['name'] for iw in client.scheduler_info()['workers'].values()]
        nworkers = len(workers_names)

        count = len(mortgage_run_params_dict_list)
        logger.info('TRYING TO LOAD {} FRAMES'.format(count))

        # Make a list of size nworkers where each element is a sublist of
        # mortgage_run_params_dict_list.
        subset_sz = count // nworkers
        mortgage_run_params_dict_list_chunks = [
            mortgage_run_params_dict_list[iw * subset_sz:(iw + 1) * subset_sz]
            if iw < (nworkers - 1) else
            mortgage_run_params_dict_list[iw * subset_sz:]
            for iw in range(nworkers)]

        logger.info(
            'SPLIT MORTGAGE DATA INTO {} CHUNKS AMONGST {} WORKERS'
            .format(len(mortgage_run_params_dict_list_chunks), nworkers))
        # For debugging. Add entry 'csvfile_perfdata' to run_params_dict.
        # for ii, ichunk in enumerate(mortgage_run_params_dict_list_chunks):
        #     files_in_chunk = \
        #         [iparam['csvfile_perfdata'] for iparam in ichunk]
        #     logger.info('CHUNK {} FILES TO LOAD: {}'.format(
        #         ii, files_in_chunk))

        # List of dask Futures of PyArrow Tables from final_perf_acq cudf
        # dataframe
        mortgage_feat_df_delinq_df_pandas_futures = client.map(
            mortgage_workflow_runner,
            mortgage_run_params_dict_list_chunks)
        wait(mortgage_feat_df_delinq_df_pandas_futures)

        if filter_dask_logger:
            wlogs = client.get_worker_logs()
            print_distributed_dask_hijacked_logs(
                wlogs, logger,
                ('mortgage_workflow_runner',
                 convert(CsvMortgagePerformanceDataLoader.__name__),
                 convert(CsvMortgageAcquisitionDataLoader.__name__))
            )

        client.run(restore_workers_logger)

        cinfo = client.who_has(mortgage_feat_df_delinq_df_pandas_futures)
        logger.info('CLIENT INFO WHO HAS WHAT: {}'.format(str(cinfo)))

        if use_rmm:
            client.run(finalize_rmm)
            client.run(initialize_rmm_no_pool)

        logmgr.cleanup()

        return mortgage_feat_df_delinq_df_pandas_futures


class DaskXgbMortgageTrainer(Node):
    '''Trains an XGBoost booster using Dask-XGBoost

    Configuration:
        conf: {
            'delete_dataframes': OPTIONAL. Boolean (True or False). Delete the
                intermediate mortgage dataframes from which an xgboost dmatrix
                is created. This is to potentially clear up CPU//GPU memory.
            'dxgb_gpu_params': REQUIRED. Dictionary of dask-xgboost trainer
                parameters.
            'client': REQUIRED. Dask distributed client. Runs with distributed
                dask.
            'create_dmatrix_serially': OPTIONAL. Boolean (True or False) Might
                be able to process more data/dataframes. Creating a dmatrix
                takes a lot of host memory. Set delete_dataframes to True as
                well to hopefully help with memory.
            'filter_dask_logger': OPTIONAL. Boolean to display hijacked
                dask.distributed log.
        }

        Example of dxgb_gpu_params:
            dxgb_gpu_params = {
                'nround':            100,
                'max_depth':         8,
                'max_leaves':        2 ** 8,
                'alpha':             0.9,
                'eta':               0.1,
                'gamma':             0.1,
                'learning_rate':     0.1,
                'subsample':         1,
                'reg_lambda':        1,
                'scale_pos_weight':  2,
                'min_child_weight':  30,
                'tree_method':       'gpu_hist',
                'n_gpus':            1,
                'distributed_dask':  True,
                'loss':              'ls',
                # 'objective':         'gpu:reg:linear',
                'objective':         'reg:squarederror',
                'max_features':      'auto',
                'criterion':         'friedman_mse',
                'grow_policy':       'lossguide',
                'verbose':           True
            }

    Inputs:
        mortgage_feat_df_delinq_df_pandas_futures = inputs[0]
        These inputs are provided by DaskMortgageWorkflowRunner.

    Outputs:
        bst - XGBoost trained booster model.

    '''
    def columns_setup(self):
        '''
        '''
        pass

    def process(self, inputs):
        import gc  # python standard lib garbage collector
        import xgboost as xgb
        from dask.delayed import delayed
        from dask.distributed import wait, get_worker
        import dask_xgboost as dxgb_gpu

        logmgr = MortgagePluginsLoggerMgr()
        logger = logmgr.get_logger()

        filter_dask_logger = self.conf.get('filter_dask_logger')

        client = self.conf['client']

        client.run(init_workers_logger)

        dxgb_gpu_params = self.conf['dxgb_gpu_params']
        delete_dataframes = self.conf.get('delete_dataframes')
        create_dmatrix_serially = self.conf.get('create_dmatrix_serially')

        mortgage_feat_df_delinq_df_pandas_futures = inputs[0]

        def make_xgb_dmatrix(
                mortgage_feat_df_delinq_df_pandas_tuple,
                delete_dataframes=None):
            worker = get_worker()

            logname = 'make_xgb_dmatrix'
            logmgr = MortgagePluginsLoggerMgr(worker, logname)
            logger = logmgr.get_logger()

            logger.info('CREATING DMATRIX ON WORKER {}'.format(worker.name))
            (mortgage_feat_df, delinq_df) = \
                mortgage_feat_df_delinq_df_pandas_tuple
            dmat = xgb.DMatrix(mortgage_feat_df, delinq_df)

            if delete_dataframes:
                del(mortgage_feat_df)
                del(delinq_df)
                # del(mortgage_feat_df_delinq_df_pandas_tuple)
                gc.collect()

            logmgr.cleanup()

            return dmat

        dmatrix_delayed_list = []
        nworkers = len(mortgage_feat_df_delinq_df_pandas_futures)

        if create_dmatrix_serially:
            logger.info('CREATING DMATRIX SERIALLY ACROSS {} WORKERS'
                        .format(nworkers))
        else:
            logger.info('CREATING DMATRIX IN PARALLEL ACROSS {} WORKERS'
                        .format(nworkers))

        for ifut in mortgage_feat_df_delinq_df_pandas_futures:
            dmat_delayed = delayed(make_xgb_dmatrix)(ifut, delete_dataframes)
            dmat_delayed_persist = dmat_delayed.persist()

            if create_dmatrix_serially:
                # TODO: For multinode efficiency need to poll the futures
                #     such that only doing serial dmatrix creation on the
                #     same node, but across nodes should be in parallel.
                wait(dmat_delayed_persist)

            dmatrix_delayed_list.append(dmat_delayed_persist)

        wait(dmatrix_delayed_list)

        if filter_dask_logger:
            wlogs = client.get_worker_logs()
            print_distributed_dask_hijacked_logs(
                wlogs, logger, ('make_xgb_dmatrix',)
            )

        client.run(restore_workers_logger)

        logger.info('JUST AFTER DMATRIX')
        print_ram_usage()

        logger.info('RUNNING XGBOOST TRAINING USING DASK-XGBOOST')
        labels = None
        bst = dxgb_gpu.train(
            client, dxgb_gpu_params, dmatrix_delayed_list, labels,
            num_boost_round=dxgb_gpu_params['nround'])

        logmgr.cleanup()

        return bst
