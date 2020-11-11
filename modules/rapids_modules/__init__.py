from .dataloader import *  # noqa: F403,F401
from .analysis import *    # noqa: F403,F401
from .transform import *   # noqa: F403,F401
from .backtest import *    # noqa: F403,F401
from .strategy import *    # noqa: F403,F401
from .portofolio import *  # noqa: F403,F401
from .ml import *  # noqa: F403,F401
from .client import validation, display  # noqa: F401
from gquant.dataframe_flow._node_flow import register_validator
from gquant.dataframe_flow._node_flow import register_copy_function

import cudf
import dask_cudf
import pandas
import numpy as np


def _validate_df(df_to_val, ref_cols, obj):
    '''Validate a cudf or dask_cudf DataFrame.

    :param df_to_val: A dataframe typically of type cudf.DataFrame or
        dask_cudf.DataFrame.
    :param ref_cols: Dictionary of column names and their expected types.
    :returns: True or False based on matching all columns in the df_to_val
        and columns spec in ref_cols.
    :raises: Exception - Raised when invalid dataframe length or unexpected
        number of columns. TODO: Create a ValidationError subclass.

    '''
    if (isinstance(df_to_val, cudf.DataFrame) or
        isinstance(df_to_val, dask_cudf.DataFrame)) and \
            len(df_to_val) == 0:
        err_msg = 'Node "{}" produced empty output'.format(obj.uid)
        raise Exception(err_msg)

    if not isinstance(df_to_val, cudf.DataFrame) and \
       not isinstance(df_to_val, dask_cudf.DataFrame):
        return True

    i_cols = df_to_val.columns
    if len(i_cols) != len(ref_cols):
        print("expect %d columns, only see %d columns"
              % (len(ref_cols), len(i_cols)))
        print("ref:", ref_cols)
        print("columns", i_cols)
        raise Exception("not valid for node %s" % (obj.uid))

    for col in ref_cols.keys():
        if col not in i_cols:
            print("error for node %s, column %s is not in the required "
                  "output df" % (obj.uid, col))
            return False

        if ref_cols[col] is None:
            continue

        err_msg = "for node {} type {}, column {} type {} "\
            "does not match expected type {}".format(
                obj.uid, type(obj), col, df_to_val[col].dtype,
                ref_cols[col])

        if ref_cols[col] == 'category':
            # comparing pandas.core.dtypes.dtypes.CategoricalDtype to
            # numpy.dtype causes TypeError. Instead, let's compare
            # after converting all types to their string representation
            # d_type_tuple = (pd.core.dtypes.dtypes.CategoricalDtype(),)
            d_type_tuple = (str(pandas.CategoricalDtype()),)
        elif ref_cols[col] == 'date':
            # Cudf read_csv doesn't understand 'datetime64[ms]' even
            # though it reads the data in as 'datetime64[ms]', but
            # expects 'date' as dtype specified passed to read_csv.
            d_type_tuple = ('datetime64[ms]', 'date', 'datetime64[ns]')
        else:
            d_type_tuple = (str(np.dtype(ref_cols[col])),)

        if (str(df_to_val[col].dtype) not in d_type_tuple):
            print("ERROR: {}".format(err_msg))
            # Maybe raise an exception here and have the caller
            # try/except the validation routine.
            return False
    return True


def copy_df(df_obj):
    return df_obj.copy(deep=False)


def copy_dask_cudf(df_obj):
    # TODO: This just makes a df_obj with a shallow copy of the
    #     underlying computational graph. It does not affect the
    #     underlying data. Why is a copy of dask graph needed?
    return df_obj.copy()


register_validator(cudf.DataFrame, _validate_df)
register_validator(dask_cudf.DataFrame, _validate_df)
register_validator(pandas.DataFrame, _validate_df)

register_copy_function(cudf.DataFrame, copy_df)
register_copy_function(dask_cudf.DataFrame, copy_dask_cudf)
register_copy_function(pandas.DataFrame, copy_df)
