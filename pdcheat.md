# Creating DataFrame

## From CSV
```py
pd.read_csv(filename, header="infer|None|int", names=<list[column names]>, index_col=<int>)
```
<br>

### time series data
```py
pd.read_csv(filename, parse_dates=true, infer_datetime_format=True index_col=<int>)
```

### From arrays/dict
```py
pd.DataFrame(<numpy array>, index, columns)
pd.Dataframe({'column name': [values]})
pd.DataFrame.from_records([[row values], [row values]], columns)
# data2 = pd.DataFrame.from_dict(
#   {'categories': {0: ['A', 'B'], 1: ['B', 'C', 'D'], 2:['B', 'D']}})
```

# Indexing
## setting index
```py
df.reset_index() # move index to column
df.set_index(<column name to use as index>)
df.reindex(new_index=None, method="None|bfill|ffill|nearest", columns=None) # Conform Series/DataFrame to new index with optional filling logic.
# df2.reindex(date_index2, method="bfill") to expand date range of dataframe
```

## Indexing/Slicing
```py
df.loc[row_label, col_label]
df.iloc[row_pos, col_pos]
```
### index names
```py
df.columns.names = [new names]
df.index.names  = [new names]
```
### swapping index levels
```py
df.swaplevel(level1, level2) 
```

## MultiIndex
### creating multiindex
```py
pd.MultiIndex.from_tuples(index=list(tuples), names)

idx_arr = [['a', 'a', 'b', 'b', 'c', 'c'], [1, 2, 1, 2, 1, 2]]
index = list(zip(*idx_arr))
index
multi_index = pd.MultiIndex.from_tuples(index, names=["first", "second"])

```


### indexing using multindex
```py
idx = pd.IndexSlice
df.loc[idx[:,:], idx[:,:]] # useful for multiindex
```


# Missing Values

detecting missing values
```py
df[col].isna() # useful for filtering
pd.isna(series)
pd.isnull()
pd.notnull()
```
handling missing values 
```py
df.dropna()
df.fillna()
```


# Duplication
## duplicated records
```py
df.drop_duplicates(inplace=bool)
df.duplicated() # return boolean series containing True for duplicated rows
```

## duplicated labels
```py
df.index.is_unique
```

# Categorical data



# Grouping
## GroupBy
```py
df.groupby(by=["keys"], dropna="True|False").aggfunc()
```
```py
df.groupby({group_mapping}) # dict of row labels and their group label
df.groupby(func) # func on row label which yiels a group
df.groupby([func, keys])
df.group(level=<int>, axis="rows") # grouping on mult-index df w/ index level
```

## Aggregations
```py
grouped.describe()
df.groupby().agg({"col":"aggfunc"}) # aggfunc: size, mean/std
```

## split-apply-combine
```py
df.groupby().apply(func) # func can return a series/scalar/df

def top5(df, n=5, column="tip_pct"):
    return df.sort_values(column, ascending=False)[:n]

tips.groupby("smoker").apply(top5, n=1-)
```

## bucketing data using cut and qcut
```py
bin_labels = pd.cut(df, bins=<int>)
df.groupby(bin_labels)
```
## transform
```py
df.groupby(keys).transform(func) # produces an dataframe same size as orginal df, func returns a group or a scalar which is broadcasted by transform
```


## Pivot Table

```py
df.pivot(index, columns, values) # df.pivot(index="date", columns="item", values=["value1", "value2"])
df.pivot_table(values, index, columns, aggfunc, margins, fill_value)
# data.pivot_table(index=["number_label"], columns=["category"], aggfunc=[len], fill_value=0)
```
### group frquencies
special case of pivot table
```py
pd.crosstab(index,columns,margins)
```

# Merging
## simple merge
```
pd.merge(df1, df2, on=[column names], sort=bool, how="inner|outer|left|right") # merge based on overlapping column values
pd.merge(df1, df2, left_index=bool, right_on="column name", sort, how) # use df1's index which matches with right_on column of right
```

## multi-index merging
```py
pd.merge(df1, left_index=True, right_on=[multiple cols])
```

## diff two dataframes using merge
```py
pd.merge(ydf, zdf, how='outer',
                     indicator=True) # indicator returns a _merge column to indicate source of each row
            .query('_merge == "left_only"')
            .drop(columns=['_merge'])
# Rows that appear in ydf but not zdf (Setdiff).
```

## df.join
```py
left_df.join(right, how) # uses index of left and right
left_df.join(right, on="left column") # index in right should match left's column
```

## concat multiple dataframes by axis
```py
pd.concat([dfs], axis="rows|columns", join="inner|outer")
```

# Reshaping
```py
data.unstack(level) # change index to column

pd.melt(df, id_vars=["column names"]) # wide to long, id_vars specify which columns are group indicators
```


# Plotting
```py
df.plot().hist() # hist of each column
df.plot().scatter() # scatter of each column against index
df.plot(kind="bar")
sns.barplot(x="col", y="col", hue="col", data=df)

```

# Statistics

### percentage change
```py
<series>.pct_change(periods=<int>)
```

```py
df[col].rank()
df[col].cumsum()
df[col].mean()
```

# Sorting

### sort values
```py
df.sort_values(<column to use for sorting>)
df[column].argmax()
```
### rank values
```py
df.rank(numeric_only="True|False", axis="0|1", pct=bool, method="average|min|max|dense|first")
df[col].rank(method, ascending="True|False")
```

### sort by index
```py
df.sort_index(level)
```


# Operations

## dropping values
```py
df.drop(label, axis)
```

## check for multiple values
```py
df.isin(iterable)
```

## Change dtype
```py
df[col] = df[col].astype(newtype)
df.convert_dtypes(infer_objects=True) # convert objects to appropriate type

```
## creating new columns from old columns
```py
df[new_col] = func(df[old_col])
df[new_col] = df[old_col].map(func)
df[new_col] = df[old_col].map({dict mappung})
# s.map({'cat': 'kitten', 'dog': 'puppy'})
# df["row1"] = df["row"].map(lambda x: x[:5])

df[new_col] = df.apply(func(row), axis=1)
```
## column wise ops
```py
df[col].apply(aggfunc)
#df[col].apply(np.mean) gives mean of col
```

## String columns
```py
df[col].str.split(<sep>, n=<number of splits>, expand=bool)
df[col].str.startswith("pattern")
df[col].str.contains("pattern")
# df[["code", "location"]] = df["row"].str.split(n=1, expand=True)
```

## apply a function along an axis
```py
df.apply(func(row/col), axis=1)
```

## unique values
```py
df[col].nunique() # number of unique values in col
df[col].value_counts() # frequency of unique elemets
```

## boolean
```py
df.any()
df.all()
```

<br>
<br>

# TimeSeries

## datetime
```py
from datetime import datetime, timedelta
datetime(year, month, day)
timedelta(days)
date.strftime("time format string") # reformat date
datetime.strptime(string, strfmt) # convert string to datetime
```

## pandas conver datestrs to datetime
```py
datestrs = ["2011-07-06 12:00:00", "2011-08-06 00:00:00"]
pd.to_datetime(datestrs)

```

## Date Range
```py
pd.date_range(start_date, enddate, periods, freq)

```
## shifting time series data
```py
ts.shift(<int>) # shift by frequency of index
```

## offsets
```py
from pandas.tseries.offsets import Day, MonthEnd
datetime() + MonthEnd() # advances the date to Month end
offset.rollforward(date) # roll the date forward
offset.rollback(date)
```

## grouping using offsets
```py
ts.groupby(offset.rollforward()).apply()
#ts = pd.Series(np.random.standard_normal(20),
#    index=pd.date_range("2000-05-01", periods=20, freq="4D"))
# ts.groupby(MonthEnd().rollforward).mean() # monthly means
```


<br>

## Periods
### creating period
```py
pd.Period(a_date_in_the_period, freq) # pd.period("2001", freq="Q")
```
### creating period range
```py
pd.period_range(start_date, end_end, freq) 
# pd.period_range("2000-01-01", "2000-06-30", freq="M")
```
### creating period Index
```py
pd.PeriodIndex(pd.period_range)
pd.PeriodIndex(array_of_periods, freq) 
# pd.PeriodIndex(values, freq="Q-DEC")
pd.PeriodIndex(year,quarter,month,freq) 
# pd.PeriodIndex(year=data["year"], quarter=data["quarter"],freq="Q")
```
### changing frequency
```py
period_object.asfreq(new_freq, how="start|end")
time_series.asfreq(new_freq, how)
```
### to_timestamp
```py
period_object.to_timestamp(how="start|end")
time_series.to_timestamp(how)
```
### to_period
```py
period_object.to_period(freq)
time_series.to_period(freq) # ts.to_period("M")
```
## Resampling
```py
time_series.resample(freq, closed="left|right", label="left|right").asfreq() # without aggregation
time_series.resample(freq).sum() # w/ aggregations

```

### grouped time resampling
for data with multiple time series
```py
pd.Grouper(freq)
ts.groupby([key, grouper]).<aggfunc()>
# ts_grouper = pd.Grouper(freq="5min")
# df2.set_index("time").groupby(["key", ts_grouper]).sum()
```

### rolling window mean
```py
time_series.rolling(<int>|freq).<aggfunc()>
time_series.rolling(<int>|freq).apply(<aggfunc()>)
#c lose_px.rolling(250).mean()["AAPL"].plot(label="rolling") 
```