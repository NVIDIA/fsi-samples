import abc
import numpy as np
import os
import cudf
import pandas as pd
import dask_cudf
import dask


__all__ = ['Node', 'TaskSpecSchema']


class TaskSpecSchema(object):
    '''Outline fields expected in a dictionary specifying a task node.

    :ivar id: unique id or name for the node
    :ivar plugin_type: Plugin class i.e. subclass of Node. Specified as string
        or subclass of Node
    :ivar conf: Configuration for the plugin i.e. parameterization. This is a
        dictionary.
    :ivar modulepath: Path to python module for custom plugin types.
    :ivar inputs: List of ids of other tasks or an empty list.
    '''

    uid = 'id'
    plugin_type = 'type'
    conf = 'conf'
    modulepath = 'filepath'
    inputs = 'inputs'

    # load = 'load'
    # save = 'save'


class Node(object):
    __metaclass__ = abc.ABCMeta

    cache_dir = os.getenv('GQUANT_CACHE_DIR', ".cache")

    def __init__(self, uid, conf, load=False, save=False):
        self.uid = uid
        self.conf = conf
        self.load = load
        self.save = save
        self.inputs = []
        self.outputs = []
        self.visited = False
        self.input_df = {}
        self.input_columns = {}
        self.output_columns = {}
        self.clear_input = True
        self.required = None
        self.addition = None
        self.deletion = None
        self.retention = None
        self.rename = None
        self.delayed_process = False
        # customized the column setup
        self.columns_setup()

    def __translate_column(self, columns):
        output = {}
        for k in columns:
            types = columns[k]
            if types is not None and types.startswith("@"):
                types = self.conf[types[1:]]
            if k.startswith("@"):
                field_name = k[1:]
                v = self.conf[field_name]
                if isinstance(v, str):
                    output[v] = types
                elif isinstance(v, list):
                    for item in v:
                        output[item] = None
            else:
                output[k] = types
        return output

    @abc.abstractmethod
    def columns_setup(self):
        """
        All children class should implement this.
        It is used to compute the input/output column names and types by
        defining the diff between input and output dataframe columns.

        The self.delayed_process flag is by default set to False. It can be
        overwritten here to True. For native dataframe API calls, dask cudf
        support the distribution computation. But the dask_cudf dataframe does
        not support GPU customized kernels directly. We can use to_delayed and
        from_delayed low level interfaces of dask_cudf to add this support.
        In order to use Dask (for distributed computation i.e. multi-gpu in
        examples later on) we set the flag and the framework
        handles dask_cudf dataframes automatically under the hood.

        `self.required`, `self.addition`, `self.deletion` and `self.retention`
        are python dictionaries, where keys are column names and values are
        column types. `self.rename` is a python dictionary where both keys and
        values are column names.

        `self.required` defines the required columns in the input dataframes
        `self.addition` defines the addional columns in the output dataframe
        Only one of `self.deletion` and `self.retention` is needed to define
        removed columns. `self.deletion` is to define the removed columns in
        the output dataframe. `self.retention` defines the remaining columns.

        Example column types:
            * int64
            * int32
            * float64
            * float32
            * datetime64[ms]

        There is a special syntax to use variable for column names. If the
        the key is `@xxxx`, it will `xxxx` as key to look up the value in the
        `self.conf` variable.

        """
        self.required = None
        self.addition = None
        self.deletion = None
        self.retention = None
        self.rename = None

    def columns_flow(self):
        """
        Flow the graph to determine the input output dataframe column names and
        types.
        """
        if not self.__input_columns_ready():
            return
        inputs = self.__get_input_columns()

        # check required columns are their
        if self.required is not None:
            required = self.__translate_column(self.required)
            for i in inputs:
                for k in required:
                    if k not in i or required[k] != i[k]:
                        if k not in i:
                            print("error for node %s, "
                                  "missing required column %s" % (self.uid, k))
                        elif required[k] != i[k]:
                            print("error for node %s, "
                                  "type %s mismatch %s"
                                  % (self.uid, required[k], i[k]))
                        raise Exception("not valid input")

        combined = {}
        for i in inputs:
            combined.update(i)

        # compute the output columns
        output = combined
        if self.addition is not None:
            output.update(self.__translate_column(self.addition))
        if self.deletion is not None:
            for key in self.__translate_column(self.deletion).keys():
                del output[key]
        if self.retention is not None:
            output = self.__translate_column(self.retention)
        if self.rename is not None:
            replacement = self.__translate_column(self.rename)
            for key in replacement.keys():
                if key not in output:
                    print("error for node %s, "
                          "missing required column %s" % (self.uid, key))
                    raise Exception("not valid replacement column")
                types = output[key]
                del output[key]
                output[replacement[key]] = types
        self.output_columns = output
        for o in self.outputs:
            o.__set_input_column(self, self.output_columns)
            o.columns_flow()

    def __valide(self, input_df, ref):
        if not isinstance(input_df, cudf.DataFrame) and \
           not isinstance(input_df, dask_cudf.DataFrame):
            return True

        i_cols = input_df.columns
        if len(i_cols) != len(ref):
            print("expect %d columns, only see %d columns"
                  % (len(ref), len(i_cols)))
            print("ref:", ref)
            print("columns", i_cols)
            raise Exception("not valid for node %s" % (self.uid))

        for col in ref.keys():
            if col not in i_cols:
                print("error for node %s, %s is not in the required input df"
                      % (self.uid, col))
                return False

            if ref[col] is None:
                continue

            err_msg = "for node {} type {}, column {} type {} "\
                "does not match expected type {}".format(
                    self.uid, type(self), col, input_df[col].dtype,
                    ref[col])

            if ref[col] == 'category':
                # comparing pandas.core.dtypes.dtypes.CategoricalDtype to
                # numpy.dtype causes TypeError. Instead, let's compare
                # after converting all types to their string representation
                # d_type_tuple = (pd.core.dtypes.dtypes.CategoricalDtype(),)
                d_type_tuple = (str(pd.core.dtypes.dtypes.CategoricalDtype()),)
            elif ref[col] == 'date':
                # Cudf read_csv doesn't understand 'datetime64[ms]' even
                # though it reads the data in as 'datetime64[ms]', but
                # expects 'date' as dtype specified passed to read_csv.
                d_type_tuple = ('datetime64[ms]', 'date',)
            else:
                d_type_tuple = (str(np.dtype(ref[col])),)

            if (str(input_df[col].dtype) not in d_type_tuple):
                print("ERROR: {}".format(err_msg))
                # Maybe raise an exception here and have the caller
                # try/except the validation routine.
                return False

        return True

    def __input_ready(self):
        if not isinstance(self.load, bool) or self.load:
            return True
        for i in self.inputs:
            if i not in self.input_df:
                return False
        return True

    def __input_columns_ready(self):
        for i in self.inputs:
            if i not in self.input_columns:
                return False
        return True

    def __get_input_df(self):
        input_df = []
        if not isinstance(self.load, bool) or self.load:
            return input_df
        for i in self.inputs:
            input_df.append(self.input_df[i])
        return input_df

    def __get_input_columns(self):
        input_columns = []
        for i in self.inputs:
            input_columns.append(self.input_columns[i])
        return input_columns

    def __set_input_df(self, parent, df):
        self.input_df[parent] = df

    def __set_input_column(self, parent, columns):
        self.input_columns[parent] = columns

    def flow(self):
        """
        flow from this node to do computation.
            * it will check all the input dataframe are ready or not
            * calls its process function to manipulate the input dataframes
            * set the resulting dataframe to the children nodes as inputs
            * flow each of the chidren nodes
        """
        if not self.__input_ready():
            return
        inputs = self.__get_input_df()
        output_df = self.__call__(inputs)
        if self.clear_input:
            self.input_df = {}
        for o in self.outputs:
            o.__set_input_df(self, output_df)
            o.flow()

    def __make_copy(self, i):
        if isinstance(i, cudf.DataFrame):
            return i.copy(deep=False)
        elif isinstance(i, dask_cudf.DataFrame):
            return i.copy()
        else:
            return i

    def load_cache(self, filename):
        """
        defines the behavior of how to load the cache file from the `filename`.
        Node can override this method.

        Arguments
        -------
        filename: str
            filename of the cache file

        """
        output_df = cudf.read_hdf(filename, key=self.uid)
        return output_df

    def __call__(self, inputs):
        # valide inputs
        Class = type(self)
        cache = Class.cache_dir
        inputs = [self.__make_copy(i) for i in inputs]
        if not isinstance(self.load, bool) or self.load:
            if isinstance(self.load, bool):
                output_df = self.load_cache(cache+'/'+self.uid+'.hdf5')
            else:
                output_df = self.load
        else:
            if not self.delayed_process:
                output_df = self.process(inputs)
            else:
                # handle the dask dataframe automatically
                # use the to_delayed interface
                # TODO, currently only handles first input is dask_cudf df
                i_df = inputs[0]
                rest = inputs[1:]
                if isinstance(i_df,  dask_cudf.DataFrame):
                    d_fun = dask.delayed(self.process)
                    output_df = dask_cudf.from_delayed([
                        d_fun([item] + rest) for item in i_df.to_delayed()])
                else:
                    output_df = self.process(inputs)

        if self.uid != 'unique_output' and output_df is None:
            raise Exception("None output")
        elif (isinstance(output_df, cudf.DataFrame) or
              isinstance(output_df, dask_cudf.DataFrame)
              ) and len(output_df) == 0:
            raise Exception("empty output")
        elif not self.__valide(output_df, self.output_columns):
            raise Exception("not valid output")

        if self.save:
            os.makedirs(cache, exist_ok=True)
            output_df.to_hdf(cache+'/'+self.uid+'.hdf5', key=self.uid)

        return output_df

    @abc.abstractmethod
    def process(self, inputs):
        """
        process the input dataframe. Children class is required to override
        this

        Arguments
        -------
         inputs: list
            list of input dataframes. dataframes order in the list matters
        Returns
        -------
        dataframe
            the processed dataframe
        """
        output = None
        return output
