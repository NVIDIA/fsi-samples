from .client import validation, display  # noqa: F401
from greenflow.dataframe_flow._node_flow import register_validator
from greenflow.dataframe_flow._node_flow import register_copy_function
from greenflow.dataframe_flow._node_flow import register_cleanup
import traceback
import cudf
import dask_cudf
import pandas
import numpy as np
import dask.dataframe


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
    if isinstance(df_to_val, cudf.DataFrame) and \
            len(df_to_val) == 0:
        err_msg = 'Node "{}" produced empty output'.format(obj.uid)
        raise Exception(err_msg)

    if not isinstance(df_to_val, cudf.DataFrame) and \
       not isinstance(df_to_val, dask_cudf.DataFrame):
        return True

    i_cols = df_to_val.columns
    if len(i_cols) != len(ref_cols):
        errmsg = 'Invalid for node "{:s}"\n'\
            'Expect {:d} columns, only see {:d} columns\n'\
            'Ref: {}\n'\
            'Columns: {}'\
            .format(obj.uid, len(ref_cols), len(i_cols), ref_cols, i_cols)
        raise Exception(errmsg)

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


def clean_dask(ui_clean):
    """
    ui_clean is True if the client send
    'clean' command to the greenflow backend
    """
    if ui_clean:
        import dask.distributed
        try:
            client = dask.distributed.client.default_client()
            client.restart()
        except Exception:
            traceback.format_exc()


register_validator(cudf.DataFrame, _validate_df)
register_validator(dask_cudf.DataFrame, _validate_df)
register_validator(pandas.DataFrame, _validate_df)
register_validator(dask.dataframe.DataFrame, _validate_df)

register_copy_function(cudf.DataFrame, copy_df)
register_copy_function(dask_cudf.DataFrame, copy_dask_cudf)
register_copy_function(pandas.DataFrame, copy_df)
register_copy_function(dask.dataframe.DataFrame, copy_dask_cudf)

register_cleanup('cleandask', clean_dask)
