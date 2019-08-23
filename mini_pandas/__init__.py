import numpy as np

__version__ = '0.0.1'


class DataFrame:

    def __init__(self, data):
        """
        A DataFrame holds two dimensional heterogeneous data. Create it by
        passing a dictionary of NumPy arrays to the values parameter

        Parameters
        ----------
        data: dict
            A dictionary of strings mapped to NumPy arrays. The key will
            become the column name.
        """
        # check for correct input types
        self._check_input_types(data)

        # check for equal array lengths
        self._check_array_lengths(data)

        # convert unicode arrays to object
        self._data = self._convert_unicode_to_object(data)

        # Allow for special methods for strings
        self.str = StringMethods(self)
        self._add_docs()

    def _check_input_types(self, data):
        if not isinstance(data, dict):
            raise TypeError("`data` must be a dictionary")
        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError("column names of `data` must be strings")
            if not isinstance(value, np.ndarray):
                raise TypeError("values of `data` must a numpy array")
            if value.ndim != 1:
                raise ValueError("values of `data` must be a one-dimensional array")

    def _check_array_lengths(self, data):
        for i, value in enumerate(data.values()):
            if i == 0:
                length = len(value)
            elif length != len(value):
                raise ValueError('all the columns of `data` must have equal length')

    def _convert_unicode_to_object(self, data):
        new_data = {}
        for key, value in data.items():
            if value.dtype.kind == 'U':
                new_data[key] = value.astype('object')
            else:
                new_data[key] = value
        return new_data

    def __len__(self):
        """
        Make the builtin len function work with our dataframe

        Returns
        -------
        int: the number of rows in the dataframe
        """
        return len(next(iter(self._data.values())))

    @property
    def columns(self):
        """
        _data holds column names mapped to arrays
        take advantage of internal ordering of dictionaries to
        put columns in correct order in list. Only works in 3.6+

        Returns
        -------
        list of column names
        """
        
        return list(self._data) # [*self._data]

    @columns.setter
    def columns(self, columns):
        """
        Must supply a list of columns as strings the same length
        as the current DataFrame

        Parameters
        ----------
        columns: list of strings

        Returns
        -------
        None
        """
        if not isinstance(columns, list):
            raise TypeError('`colums` must be a list object')
        if len(columns) != len(self._data):
            raise ValueError('length column names does not match current DataFrame columns length')
        for col in columns:
            if not isinstance(col, str):
                raise TypeError('all the column names are not string')
        if len(columns) != len(set(columns)):
            raise ValueError('`columns` contains duplicates')
        
        self._data = dict(zip(columns, self._data.values()))

    @property
    def shape(self):
        """
        Returns
        -------
        two-item tuple of number of rows and columns
        """

        return len(self), len(self._data)

    def _repr_html_(self):
        """
        Used to create a string of HTML to nicely display the DataFrame
        in a Jupyter Notebook. Different string formatting is used for
        different data types.

        The structure of the HTML is as follows:
        <table>
            <thead>
                <tr>
                    <th>data</th>
                    ...
                    <th>data</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>{i}</strong></td>
                    <td>data</td>
                    ...
                    <td>data</td>
                </tr>
                ...
                <tr>
                    <td><strong>{i}</strong></td>
                    <td>data</td>
                    ...
                    <td>data</td>
                </tr>
            </tbody>
        </table>
        """

        html = '<table><thead><tr><th></th>'
        for col in self.columns:
            html += f"<th>{col:10}</th>"

        html += '</tr></thead>'
        html += "<tbody>"

        only_head = False
        num_head = 10
        num_tail = 10
        if len(self) <= 20:
            only_head = True
            num_head = len(self)

        for i in range(num_head):
            html += f'<tr><td><strong>{i}</strong></td>'
            for col, values in self._data.items():
                kind = values.dtype.kind
                if kind == 'f':
                    html += f'<td>{values[i]:10.3f}</td>'
                elif kind == 'b':
                    html += f'<td>{values[i]}</td>'
                elif kind == 'O':
                    v = values[i]
                    if v is None:
                        v = 'None'
                    html += f'<td>{v:10}</td>'
                else:
                    html += f'<td>{values[i]:10}</td>'
            html += '</tr>'

        if not only_head:
            html += '<tr><strong><td>...</td></strong>'
            for i in range(len(self.columns)):
                html += '<td>...</td>'
            html += '</tr>'
            for i in range(-num_tail, 0):
                html += f'<tr><td><strong>{len(self) + i}</strong></td>'
                for col, values in self._data.items():
                    kind = values.dtype.kind
                    if kind == 'f':
                        html += f'<td>{values[i]:10.3f}</td>'
                    elif kind == 'b':
                        html += f'<td>{values[i]}</td>'
                    elif kind == 'O':
                        v = values[i]
                        if v is None:
                            v = 'None'
                        html += f'<td>{v:10}</td>'
                    else:
                        html += f'<td>{values[i]:10}</td>'
                html += '</tr>'

        html += '</tbody></table>'
        return html

    @property
    def values(self):
        """
        Returns
        -------
        A single 2D NumPy array of the underlying data
        """
        return np.column_stack(list(self._data.values()))

    @property
    def dtypes(self):
        """
        Returns
        -------
        A two-column DataFrame of column names in one column and
        their data type in the other
        """
        DTYPE_NAME = {'O': 'string', 'i': 'int', 'f': 'float', 'b': 'bool'}
        col_names = np.array(list(self._data.keys()))
        dtypes = [DTYPE_NAME[value.dtype.kind] for value in self._data.values()]
        dtypes = np.array(dtypes)
        new_data = {'Column Name': col_names, 'Data Type': dtypes}
        return DataFrame(new_data)

    def __getitem__(self, item):
        """
        Use the brackets operator to simultaneously select rows and columns
        A single string selects one column -> df['colname']
        A list of strings selects multiple columns -> df[['colname1', 'colname2']]
        A one column DataFrame of booleans that filters rows -> df[df_bool]
        Row and column selection simultaneously -> df[rs, cs]
            where cs and rs can be integers, slices, or a list of integers
            rs can also be a one-column boolean DataFrame

        Returns
        -------
        A subset of the original DataFrame
        """
        if type(item) is str:
            return DataFrame({item:self._data[item]})
        elif type(item) is list:
            return DataFrame({col: self._data[col] for col in item})
        elif type(item) is DataFrame:
            if item.shape[1] != 1:
                raise ValueError('the DataFrame must have one-column')
            arr = next(iter(item._data.values()))
            if arr.dtype.kind != 'b':
                raise TypeError('rows of the DataFrame must be boolean')
            new_data = {col: values[arr] for col, values in self._data.items()}

            return DataFrame(new_data)
        elif type(item) is tuple:
            return self._getitem_tuple(item)
        else:
            raise TypeError('item must be a string, list, DataFrame, or tuple'
                            'to the selection operator')

    def _getitem_tuple(self, item):
        # simultaneous selection of rows and cols -> df[rs, cs]
        if len(item) != 2:
            raise ValueError('item tuple must have a length of 2')
        
        row_selection, col_selection = item
        if type(row_selection) is int:
            row_selection = [row_selection]
        elif type(row_selection) is DataFrame:
            if row_selection.shape[1] != 1:
                raise ValueError('row_selectiong DataFrame must have one-column')
            row_selection = next(iter(row_selection._data.values()))
            if row_selection.dtype.kind != 'b':
                raise TypeError('row_selection DataFrame must be a one-column boolean')
        elif not type(row_selection) in (list, slice):
            raise TypeError('row_selection must be a integer, DataFrame, list or slice')

        
        if type(col_selection) is int:
            col_selection = [self.columns[col_selection]]
        elif type(col_selection) is str:
            col_selection = [col_selection]
        elif type(col_selection) is list:
            new_col_selection = []
            for col in col_selection:
                if type(col) is int:
                    new_col_selection.append(self.columns[col])
                else:
                    new_col_selection.append(col)
            col_selection = new_col_selection
        elif type(col_selection) is slice:
            start = col_selection.start
            stop = col_selection.stop
            step = col_selection.step

            if type(start) is str:
                start = self.columns.index(start)
            if type(stop) is str:
                stop = self.columns.index(stop) + 1

            col_selection = self.columns[start:stop:step]
        else:
            raise TypeError('col_selection must be an integer, string, list or slice')


        

        new_data = {}
        for col in col_selection:
            new_data[col] = self._data[col][row_selection]
        return DataFrame(new_data)


    def _ipython_key_completions_(self):
        # allows for tab completion when doing df['c
        return self.columns

    def __setitem__(self, key, value):
        # adds a new column or a overwrites an old column
        if type(key) is not str:
            raise NotImplementedError('only strings can be used to create or modify column')

        if type(value) is np.ndarray:
            # raise TypeError('column values must a numpy array')
            if value.ndim != 1:
                raise ValueError('column numpy array should be one dimensional')
            if len(value) != len(self):
                raise ValueError('number of rows in new column must match existing rows in DataFrame')
        elif type(value) is DataFrame:
            if value.shape[1] != 1:
                raise ValueError('the new DataFrame must be one-dimensional')
            if len(value) != len(self):
                raise ValueError('number of rows in new column DataFrame must match existing rows in DataFrame')
            value = next(iter(value._data.values()))
        elif type(value) in (int, bool, float, str):
            value = np.repeat(value, len(self))
        else:
            raise TypeError('the value being assigned must be numpy array, DataFrame, int, float, bool or str.')
            
        if value.dtype.kind == 'U':
            value = value.astype('O')

        self._data[key] = value

    def head(self, n=5):
        """
        Return the first n rows

        Parameters
        ----------
        n: int

        Returns
        -------
        DataFrame
        """
        if type(n) is not int:
            raise TypeError('value passed to method must be an integer')
        
        
        return self[:n, :]

    def tail(self, n=5):
        """
        Return the last n rows

        Parameters
        ----------
        n: int
        
        Returns
        -------
        DataFrame
        """
        if type(n) is not int:
            raise TypeError('value passed to method must be an integer')

        return self[-n:, :]

    #### Aggregation Methods ####

    def min(self):
        return self._agg(np.min)

    def max(self):
        return self._agg(np.max)

    def mean(self):
        return self._agg(np.mean)

    def median(self):
        return self._agg(np.median)

    def sum(self):
        return self._agg(np.sum)

    def var(self):
        return self._agg(np.var)

    def std(self):
        return self._agg(np.std)

    def all(self):
        return self._agg(np.all)

    def any(self):
        return self._agg(np.any)

    def argmax(self):
        return self._agg(np.argmax)

    def argmin(self):
        return self._agg(np.argmin)

    def _agg(self, aggfunc):
        """
        Generic aggregation function that applies the
        aggregation to each column

        Parameters
        ----------
        aggfunc: str of the aggregation function name in NumPy
        
        Returns
        -------
        A DataFrame
        """
        new_data = {}
        for key, value in self._data.items():
            try:
                new_data[key] = np.array([aggfunc(value)])
            except TypeError:
                pass
        
        return DataFrame(new_data)

    def isna(self):
        """
        Determines whether each value in the DataFrame is missing or not

        Returns
        -------
        A DataFrame of booleans the same size as the calling DataFrame
        """
        new_data = {}
        for key, value in self._data.items():
            if value.dtype.kind == 'O':
                new_data[key] = (value == None)
            else:
                new_data[key] = np.isnan(value)
        return DataFrame(new_data)

    def count(self):
        """
        Counts the number of non-missing values per column

        Returns
        -------
        A DataFrame
        """
        new_data = {}
        new_df = self.isna()
        length = len(new_df)
        for key, value in new_df._data.items():
            new_data[key] = np.array([length - value.sum()])
        return DataFrame(new_data)

    def unique(self):
        """
        Finds the unique values of each column

        Returns
        -------
        A list of one-column DataFrames
        """
        dfs = []
        for key, value in self._data.items():
            new_data = {key: np.unique(value)}
            dfs.append(DataFrame(new_data))
        if len(dfs) == 1:
            return dfs[0]
        return dfs

    def nunique(self):
        """
        Find the number of unique values in each column

        Returns
        -------
        A DataFrame
        """
        new_data = {}
        for key, value in self._data.items():
            
            new_data[key] = np.array([len(np.unique(value))])
        return DataFrame(new_data)

    def value_counts(self, normalize=False):
        """
        Returns the frequency of each unique value for each column

        Parameters
        ----------
        normalize: bool
            If True, returns the relative frequencies (percent)

        Returns
        -------
        A list of DataFrames or a single DataFrame if one column
        """
        dfs = []
        for key, value in self._data.items():
            unique_values, counts = np.unique(value, return_counts=True)
            sorted_index = np.argsort(- counts)
            unique_values = unique_values[sorted_index]
            counts = counts[sorted_index]
            if normalize:
                counts = counts/len(self)
            new_data = {key: unique_values, 'count': counts}
            dfs.append(DataFrame(new_data))
        
        if len(dfs) == 1:
            return dfs[0]
        return dfs

    def rename(self, columns):
        """
        Renames columns in the DataFrame

        Parameters
        ----------
        columns: dict
            A dictionary mapping the old column name to the new column name
        
        Returns
        -------
        A DataFrame
        """
        if type(columns) is not dict:
            raise TypeError('columns must be a dictionary')


        new_data = {}
        for key, value in self._data.items():
            new_key = columns.get(key, key)
            new_data[new_key] = value
        return DataFrame(new_data)


    def drop(self, columns):
        """
        Drops one or more columns from a DataFrame

        Parameters
        ----------
        columns: str or list of strings

        Returns
        -------
        A DataFrame
        """
        

        if type(columns) is str:
            columns = [columns]
        elif type(columns) is not list:
            raise TypeError('columns must be a string or a list')

        new_data = {}
        for key, value in self._data.items():
            if key not in columns:
                new_data[key] = value
        
        return DataFrame(new_data)

    #### Non-Aggregation Methods ####

    def abs(self):
        """
        Takes the absolute value of each value in the DataFrame

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.abs)

    def cummin(self):
        """
        Finds cumulative minimum by column

        Returns
        -------
        A DataFrame
        """
        # To ignore NaN values
        return self._non_agg(lambda a: a*0 + np.minimum.accumulate(np.nan_to_num(a)))

    def cummax(self):
        """
        Finds cumulative maximum by column

        Returns
        -------
        A DataFrame
        """
        # To ignore NaN values
        return self._non_agg(lambda a: a*0 + np.maximum.accumulate(np.nan_to_num(a)))

    def cumsum(self):
        """
        Finds cumulative sum by column

        Returns
        -------
        A DataFrame
        """
        # To ignore NaN values
        return self._non_agg(lambda a: a*0 + np.cumsum(np.nan_to_num(a)))

    def clip(self, lower=None, upper=None):
        """
        All values less than lower will be set to lower
        All values greater than upper will be set to upper

        Parameters
        ----------
        lower: number or None
        upper: number or None

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.clip, a_min=lower, a_max=upper)

    def round(self, n):
        """
        Rounds values to the nearest n decimals

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.round, decimals=n)

    def copy(self):
        """
        Copies the DataFrame

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.copy)

    def _non_agg(self, funcname, **kwargs):
        """
        Generic non-aggregation function
    
        Parameters
        ----------
        funcname: numpy function
        kwargs: extra keyword arguments for certain functions

        Returns
        -------
        A DataFrame
        """
        new_data = {}
        for key, value in self._data.items():
            if value.dtype.kind in ('O', 'b'):
                new_data[key] = value.copy()
            else:
                new_data[key] = funcname(value, **kwargs)
        
        return DataFrame(new_data)

    def diff(self, n=1):
        """
        Take the difference between the current value and
        the nth value above it.

        Parameters
        ----------
        n: int

        Returns
        -------
        A DataFrame
        """
        def func(value):
            value = value.astype('float')
            value_shifted = np.roll(value, n)
            value = value - value_shifted
            if n >= 0:
                value[:n] = np.nan
            else:
                value[n:] = np.nan
            return value
        return self._non_agg(func)

    def pct_change(self, n=1):
        """
        Take the percentage difference between the current value and
        the nth value above it.

        Parameters
        ----------
        n: int

        Returns
        -------
        A DataFrame
        """
        def func(value):
            value = value.astype('float')
            value_shifted = np.roll(value, n)
            value = (value - value_shifted) / value_shifted
            if n >= 0:
                value[:n] = np.nan
            else:
                value[n:] = np.nan
            return value
        return self._non_agg(func)

    #### Arithmetic and Comparison Operators ####

    def __add__(self, other):
        return self._oper('__add__', other)

    def __radd__(self, other):
        return self._oper('__radd__', other)

    def __sub__(self, other):
        return self._oper('__sub__', other)

    def __rsub__(self, other):
        return self._oper('__rsub__', other)

    def __mul__(self, other):
        return self._oper('__mul__', other)

    def __rmul__(self, other):
        return self._oper('__rmul__', other)

    def __truediv__(self, other):
        return self._oper('__truediv__', other)

    def __rtruediv__(self, other):
        return self._oper('__rtruediv__', other)

    def __floordiv__(self, other):
        return self._oper('__floordiv__', other)

    def __rfloordiv__(self, other):
        return self._oper('__rfloordiv__', other)

    def __pow__(self, other):
        return self._oper('__pow__', other)

    def __rpow__(self, other):
        return self._oper('__rpow__', other)

    def __gt__(self, other):
        return self._oper('__gt__', other)

    def __lt__(self, other):
        return self._oper('__lt__', other)

    def __ge__(self, other):
        return self._oper('__ge__', other)

    def __le__(self, other):
        return self._oper('__le__', other)

    def __ne__(self, other):
        return self._oper('__ne__', other)

    def __eq__(self, other):
        return self._oper('__eq__', other)

    def _oper(self, op, other):
        """
        Generic operator function

        Parameters
        ----------
        op: str name of special method
        other: the other object being operated on

        Returns
        -------
        A DataFrame
        """
        if type(other) is DataFrame:
            if other.shape[1] != 1:
                raise ValueError('DataFrame must be one-dimensional')
            else:
                other = next(iter(other._data.values()))

        new_data = {}
        for key, value in self._data.items():
            if value.dtype.kind not in ('O', 'b'):
                method = getattr(value, op)
                new_data[key] = method(other)
            else:
                new_data[key] = value.copy()

        return DataFrame(new_data)

    def sort_values(self, by, asc=True):
        """
        Sort the DataFrame by one or more values

        Parameters
        ----------
        by: str or list of column names
        asc: boolean of sorting order

        Returns
        -------
        A DataFrame
        """
        if type(by) is str:
            order = np.argsort(self._data[by])
        elif type(by) is list:
            by = [self._data[col] for col in by[::-1]]
            order = np.lexsort(by)
        else:
            raise TypeError('`by` must a string or a list')
        
        if not asc:
            order = order[::-1]

        return self[order.tolist(), :]

    def sample(self, n=None, frac=None, replace=False, seed=None):
        """
        Randomly samples rows the DataFrame

        Parameters
        ----------
        n: int
            number of rows to return
        frac: float
            Proportion of the data to sample
        replace: bool
            Whether or not to sample with replacement
        seed: int
            Seeds the random number generator

        Returns
        -------
        A DataFrame
        """
        if seed:
            np.random.seed(seed)
        if frac:
            if frac < 0:
                raise ValueError('`frac` cannot be negative')
            n = int(frac * len(self))

        if type(n) is not int:
            raise TypeError('`n` must be an integer')
            

        row_idx = np.random.choice(range(len(self)), n, replace=replace)
        return self[row_idx.tolist(), :]

    def pivot_table(self, rows=None, columns=None, values=None, aggfunc=None):
        """
        Creates a pivot table from one or two 'grouping' columns.

        Parameters
        ----------
        rows: str of column name to group by
            Optional
        columns: str of column name to group by
            Optional
        values: str of column name to aggregate
            Required
        aggfunc: str of aggregation function

        Returns
        -------
        A DataFrame
        """

        if rows is None and columns is None:
            raise ValueError('`rows` and `column` both cannot be empty')


        if values is not None:
            if aggfunc is None:
                raise ValueError('no `aggfunc` passed to aggregate `values`')
            val_data = self._data[values]
        else:
            if aggfunc is None:
                aggfunc = 'size'
                val_data = np.empty(len(self))
            else:
                raise ValueError('`values` cannot be None')


        if columns is not None:
            col_data = self._data[columns]

        if rows is not None:
            row_data = self._data[rows]

        if rows is None:
            pivot = 'column'
        elif columns is None:
            pivot = 'row'
        else:
            pivot = 'all'


        from collections import defaultdict
        d = defaultdict(list)
        if pivot == 'column':
            for group, val in zip(col_data, val_data):
                d[group].append(val)
        elif pivot == 'row':
            for group, val in zip(row_data, val_data):
                d[group].append(val)
        else:
            for group1, group2, val in zip(col_data, row_data, val_data):
                d[(group1, group2)].append(val)


        agg_dict = {}
        for group, vals in d.items():
            arr = np.array(vals)
            func = getattr(np, aggfunc)
            agg_dict[group] = func(arr)

        new_data = {}
        if pivot == 'column':
            for col in sorted(agg_dict):
                new_data[col] = np.array([agg_dict[col]])
        elif pivot == 'row':
            row_vals = np.array(list(agg_dict.keys()))
            vals = np.array(list(agg_dict.values()))

            order = np.argsort(row_vals)
            new_data[rows] = row_vals[order]
            new_data[aggfunc] = vals[order]
        else:
            col_set = set([group[0] for group in agg_dict.keys()])
            row_set = set([group[1] for group in agg_dict.keys()])

            col_vals = sorted(col_set)
            row_vals = sorted(row_set)

            new_data[rows] = np.array(row_vals)
            for col in col_vals:
                new_vals = []
                for row in row_vals:
                    new_vals.append(agg_dict.get((col, row), np.nan))
                new_data[col] = np.array(new_vals)

        
        return DataFrame(new_data)

    def _add_docs(self):
        agg_names = ['min', 'max', 'mean', 'median', 'sum', 'var',
                     'std', 'any', 'all', 'argmax', 'argmin']
        agg_doc = \
        """
        Find the {} of each column
        
        Returns
        -------
        DataFrame
        """
        for name in agg_names:
            getattr(DataFrame, name).__doc__ = agg_doc.format(name)


class StringMethods:

    def __init__(self, df):
        self._df = df

    def capitalize(self, col):
        return self._str_method(str.capitalize, col)

    def center(self, col, width, fillchar=None):
        if fillchar is None:
            fillchar = ' '
        return self._str_method(str.center, col, width, fillchar)

    def count(self, col, sub, start=None, stop=None):
        return self._str_method(str.count, col, sub, start, stop)

    def endswith(self, col, suffix, start=None, stop=None):
        return self._str_method(str.endswith, col, suffix, start, stop)

    def startswith(self, col, suffix, start=None, stop=None):
        return self._str_method(str.startswith, col, suffix, start, stop)

    def find(self, col, sub, start=None, stop=None):
        return self._str_method(str.find, col, sub, start, stop)

    def len(self, col):
        return self._str_method(str.__len__, col)

    def get(self, col, item):
        return self._str_method(str.__getitem__, col, item)

    def index(self, col, sub, start=None, stop=None):
        return self._str_method(str.index, col, sub, start, stop)

    def isalnum(self, col):
        return self._str_method(str.isalnum, col)

    def isalpha(self, col):
        return self._str_method(str.isalpha, col)

    def isdecimal(self, col):
        return self._str_method(str.isdecimal, col)

    def islower(self, col):
        return self._str_method(str.islower, col)

    def isnumeric(self, col):
        return self._str_method(str.isnumeric, col)

    def isspace(self, col):
        return self._str_method(str.isspace, col)

    def istitle(self, col):
        return self._str_method(str.istitle, col)

    def isupper(self, col):
        return self._str_method(str.isupper, col)

    def lstrip(self, col, chars):
        return self._str_method(str.lstrip, col, chars)

    def rstrip(self, col, chars):
        return self._str_method(str.rstrip, col, chars)

    def strip(self, col, chars):
        return self._str_method(str.strip, col, chars)

    def replace(self, col, old, new, count=None):
        if count is None:
            count = -1
        return self._str_method(str.replace, col, old, new, count)

    def swapcase(self, col):
        return self._str_method(str.swapcase, col)

    def title(self, col):
        return self._str_method(str.title, col)

    def lower(self, col):
        return self._str_method(str.lower, col)

    def upper(self, col):
        return self._str_method(str.upper, col)

    def zfill(self, col, width):
        return self._str_method(str.zfill, col, width)

    def encode(self, col, encoding='utf-8', errors='strict'):
        return self._str_method(str.encode, col, encoding, errors)

    def _str_method(self, method, col, *args):
        old_vals = self._df._data[col]
        if old_vals.dtype.kind != 'O':
            raise TypeError('`col` have to be a string column')
        
        new_vals = []
        for val in old_vals:
            if val is None:
                new_val = None
            else:
                new_val = method(val, *args)
            new_vals.append(new_val)
        arr = np.array(new_vals)

        return DataFrame({col: arr})


def read_csv(fn):
    """
    Read in a comma-separated value file as a DataFrame

    Parameters
    ----------
    fn: string of file location

    Returns
    -------
    A DataFrame
    """
    from collections import defaultdict

    data = defaultdict(list)
    with open(fn) as f:
        header = f.readline()
        column_names = header.rstrip('\n').split(',')
        for row in f:
            value = row.rstrip('\n').split(',')
            for col, val in zip(column_names, value):
                data[col].append(val)
        
    new_data = {}
    for col, vals in data.items():
        try:
            new_data[col] = np.array(vals, dtype='int')
        except ValueError:
            try:
                new_data[col] = np.array(vals, dtype='float')
            except ValueError:
                new_data[col] = np.array(vals, dtype='object')
    return DataFrame(new_data)


