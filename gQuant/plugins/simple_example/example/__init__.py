from .distanceNode import DistanceNode
from .pointNode import PointNode
import pandas as pd
import numpy as np
from .client import validation, display  # noqa: F40
from greenflow.dataframe_flow._node_flow import register_validator
from greenflow.dataframe_flow._node_flow import register_copy_function


def _validate_df(df_to_val, ref_cols, obj):
    '''Validate a pandas DataFrame.

    :param df_to_val: A dataframe typically of type pd.DataFrame
    :param ref_cols: Dictionary of column names and their expected types.
    :returns: True or False based on matching all columns in the df_to_val
        and columns spec in ref_cols.
    :raises: Exception - Raised when invalid dataframe length or unexpected
        number of columns. TODO: Create a ValidationError subclass.

    '''
    if (isinstance(df_to_val, pd.DataFrame) and len(df_to_val) == 0):
        err_msg = 'Node "{}" produced empty output'.format(obj.uid)
        raise Exception(err_msg)

    if not isinstance(df_to_val, pd.DataFrame):
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
            d_type_tuple = (str(pd.CategoricalDtype()),)
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


register_validator(pd.DataFrame, _validate_df)
register_copy_function(pd.DataFrame, copy_df)
