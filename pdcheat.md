# 
# Creating DataFrame
## From CSV
# Indexing
## Normal indexing
## MultiIndex
### creating multiindex
### indexing using multindex
# Missing Values
# Grouping
## GroupBy
## Pivot Table
# Merging
## pd.merge
## pd.concat
## df.join
# Plotting
# Statistics
# Sorting
# Operations
## creating new columns from old columns
# TimeSeries
## Date Range
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
pd.PeriodIndex(array_of_periods, freq) # pd.PeriodIndex(values, freq="Q-DEC")
pd.PeriodIndex(year,quarter,month,freq) # pd.PeriodIndex(year=data["year"], quarter=data["quarter"],freq="Q")
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