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

# Duplication
## duplicated records
## duplicated labels

# Categorical data

# Grouping
## GroupBy
## Pivot Table
```py
df.pivot_table(values, index, columns, aggfunc, fill_value)
# data.pivot_table(index=["number_label"], columns=["category"], aggfunc=[len], fill_value=0)
```

# Merging
## simple merge
```
pd.merge(df1, df2, on=[column names], sort=bool, how="inner|outer|left|right") # merge based on overlapping column values
pd.merge(df1, df2, left_index=bool, right_on="column name", sort, how) # use df1's index which matches with right_on column of right
```

## multi-index merging
```py
```

## pd.concat
## df.join

# Reshaping
```py
data.unstack(level) # change index to column
df.swaplevel(level1, level2) 
```


# Plotting

# Statistics

### percentage change
```py
<series>.pct_change(periods=<int>)
```

# Sorting
```py
df.sort_values(<column to use for sorting>)
df[column].argmax()
```
sort by index
```py
df.sort_index(level)
```


# Operations

## dropping values
```py
df.drop(label, axis)
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
# df["row1"] = df["row"].map(lambda x: x[:5])

df[new_col] = df.apply(func(row), axis=1)
```

## String columns
```py
df[col].str.split(<sep>, n=<number of splits>, expand=bool)
# df[["code", "location"]] = df["row"].str.split(n=1, expand=True)
```

## apply a function along an axis
```py
df.apply(func(row/col), axis=1)
```

<br>
<br>

# TimeSeries

## Date Range



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