# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:52:38 2023

@author: ilyak
"""

import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import warnings as wn


def infer_trading_periods(s):
    '''
    Guesses the data frequency of a time series.
    Warning: no automatic errors for functions which are probably best used on resolutions
    of daily frequency or lower frequency (weekly, monthly, etc.)

    Parameters
    ----------
    s : a time series

    Returns
    -------
    trading_periods : the number of trading periods of the frequency of
    the data per year (E.G. 252 days per year, 52 weeks, 4 quarters, etc.)

    '''
    index_diff = s.index.to_series().diff().mean()

    if pd.isnull(index_diff):
        return None

    conversion_map = {
        pd.Timedelta(1, 'D'): 252,    # Business days
        pd.Timedelta(1, 'h'): 252 * 24,    # Hours
        pd.Timedelta(1, 'm'): 252 * 24 * 60,    # Minutes
        pd.Timedelta(1, 's'): 252 * 24 * 60 * 60,    # Seconds
        pd.Timedelta(1, 'ms'): 252 * 24 * 60 * 60 * 1000,    # Milliseconds
        pd.Timedelta(1, 'W'): 52,    # Weeks (considering 52 weeks in a year)
        pd.offsets.MonthBegin(): 12,    # Months
        pd.offsets.DateOffset(months=3): 4,    # Quarters
        pd.offsets.DateOffset(years=1): 1    # Years
    }

    trading_periods = None
    for unit, periods in conversion_map.items():
        if index_diff >= unit:
            trading_periods = periods
            break

    return trading_periods

def terminal_wealth(s):
    '''
    Computes the terminal wealth of a sequence of return, which is, in other words, 
    the final compounded return. 
    The input s is expected to be either a pd.DataFrame or a pd.Series
    '''
    if not isinstance(s, (pd.DataFrame, pd.Series)):
        raise ValueError("Expected either a pd.DataFrame or pd.Series")
    return (1 + s).prod()

def Return_cumulative(s):
    '''
    Single compound rule for a pd.Dataframe or pd.Series of returns. 
    The method returns a single number - using prod(). 
    See also the TERMINAL_WEALTH method.
    '''
    if not isinstance(s, (pd.DataFrame, pd.Series)):
        raise ValueError("Expected either a pd.DataFrame or pd.Series")
    return (1 + s).prod() - 1
    # Note that this is equivalent to (but slower than)
    # return np.expm1( np.logp1(s).sum() )
    
def compound_returns(s, start = 1):
    '''
    Compound a pd.Dataframe or pd.Series of returns from an initial default value equal to 100.
    In the former case, the method compounds the returns for every column (Series) by using pd.aggregate. 
    The method returns a pd.Dataframe or pd.Series - using cumprod(). 
    See also the COMPOUND method.
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( compound_returns, start=start )
    elif isinstance(s, pd.Series):
        return start * (1 + s).cumprod()
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
        
def Return_calculate(s, dropna = True):
    '''
    Computes the returns (percentage change) of a Dataframe of Series. 
    In the former case, it computes the returns for every column (Series) by using pd.aggregate
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( Return_calculate )
    elif isinstance(s, pd.Series):
        returns = s/s.shift(1) - 1
        if dropna:
            return returns.dropna()
        return returns
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
        
def compute_logreturns(s):
    '''
    Computes the log-returns of a Dataframe of Series. 
    In the former case, it computes the returns for every column (Series) by using pd.aggregate
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( compute_logreturns )
    elif isinstance(s, pd.Series):
        return np.log( s / s.shift(1) )
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
    
def Drawdowns(rets: pd.Series, start = 1):
    '''
    Compute the drawdowns of an input pd.Series of returns. 
    The method returns a dataframe containing: 
    1. the associated wealth index (for an hypothetical starting investment of $1000) 
    2. all previous peaks 
    3. the drawdowns
    '''
    wealth_index   = compound_returns(rets, start=start)
    previous_peaks = wealth_index.cummax()
    drawdowns      = (wealth_index - previous_peaks ) / previous_peaks
    #df = pd.DataFrame({"Wealth": wealth_index, "Peaks": previous_peaks, "Drawdown": drawdowns} )
    return drawdowns

def max_drawdown(s):
    return(Drawdowns(s).min(axis=0)*-1)

def skewness(s):
    '''
    Computes the Skewness of the input Series or Dataframe.
    There is also the function scipy.stats.skew().
    '''
    return ( ((s - s.mean()) / s.std(ddof=0))**3 ).mean()

def kurtosis(s):
    '''
    Computes the Kurtosis of the input Series or Dataframe.
    There is also the function scipy.stats.kurtosis() which, however, 
    computes the "Excess Kurtosis", i.e., Kurtosis minus 3
    '''
    return ( ((s - s.mean()) / s.std(ddof=0))**4 ).mean()

def exkurtosis(s):
    '''
    Returns the Excess Kurtosis, i.e., Kurtosis minus 3
    '''
    return kurtosis(s) - 3

def is_normal(s, level = 0.01):
    '''
    Jarque-Bera test to see if a series (of returns) is normally distributed.
    Returns True or False according to whether the p-value is larger 
    than the default level=0.01.
    '''
    statistic, pvalue = scipy.stats.jarque_bera( s )
    return pvalue > level

def semivolatility(s):
    '''
    Returns the semivolatility of a series, i.e., the volatility of
    negative returns
    '''
    return s[s<0].std(ddof=0) 

def var_historic(s, level = 0.05):
    '''
    Returns the (1-level)% VaR using historical method. 
    By default it computes the 95% VaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series) by using pd.aggregate
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( var_historic, level=level )
    elif isinstance(s, pd.Series):
        return - np.percentile(s, level*100)
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
        
def var_gaussian(s, level = 0.05, cf=False):
    '''
    Returns the (1-level)% VaR using the parametric Gaussian method. 
    By default it computes the 95% VaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
    The variable "cf" stands for Cornish-Fisher. If True, the method computes the 
    modified VaR using the Cornish-Fisher expansion of quantiles.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series).
    '''
    # alpha-quantile of Gaussian distribution 
    za = scipy.stats.norm.ppf(level,0,1) 
    if cf:
        S = skewness(s)
        K = kurtosis(s)
        za = za + (za**2 - 1)*S/6 + (za**3 - 3*za)*(K-3)/24 - (2*za**3 - 5*za)*(S**2)/36    
    return -( s.mean() + za * s.std(ddof=0) )

def cvar_historic(s, level=0.05):
    '''
    Computes the (1-level)% Conditional VaR (based on historical method).
    By default it computes the 95% CVaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series).
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( cvar_historic, level=level )
    elif isinstance(s, pd.Series):
        # find the returns which are less than (the historic) VaR
        mask = s < -var_historic(s, level=level)
        # and of them, take the mean 
        return -s[mask].mean()
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def cvar_gaussian(s, level=0.05):
    '''
    Computes the (1-level)% Conditional VaR (based on the parametric Gaussian method).
    By default it computes the 95% CVaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series).
    '''
    # alpha-quantile of Gaussian distribution
    za = stats.norm.ppf(level, 0, 1)
    return s.std(ddof=0) * -stats.norm.pdf(za) / level + s.mean()

def var_studentt(s, level = 0.05):
    '''
    Returns the (1-level)% VaR using the parametric Student-T distribution. 
    By default it computes the 95% VaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series).
    '''
    # Fitting Student-T parameters to the data
    v = np.array([stats.t.fit(s[col])[0] for col in s.columns])
    # alpha-quantile of Student-T distribution
    za = stats.t.ppf(level, v, 1)
    return - np.sqrt((v - 2) / v) * za * s.std(ddof=0) + s.mean()

def cvar_studentt(s, level=0.05):
    '''
    Computes the (1-level)% Conditional VaR (based on the Student-T distribution).
    By default it computes the 95% CVaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series).
    '''
    # Fitting student-t parameters to the data
    v, scale = np.array([]), np.array([])
    for col in s.columns:
        col_v, _, col_scale = stats.t.fit(s[col])
        v = np.append(v, col_v)
        scale = np.append(scale, col_scale)
    # alpha-quantile of Student-T distribution
    za = stats.t.ppf(1-level, v, 1)
    return - scale * (v + za**2) / (v - 1) * stats.t.pdf(za, v) / level + s.mean()

def cvar_laplace(s, level=0.05):
    '''
    Computes the (1-level)% Conditional VaR (based on the Laplace distribution).
    By default it computes the 95% CVaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series).
    '''
    # Fitting b (scale parameter) to the variance of the data
    # Since variance of the Laplace dist.: var = 2*b**2
    b = np.sqrt(s.std(ddof=0)**2 / 2)
    if level =< 0.5:
        return -b * (1 - np.log(2 * level)) + mean
    else:
        print("Laplace Conditional VaR is not available for a level over 50%.")
        return 0

def cvar_logistic(s, level=0.05):
    '''
    Computes the (1-level)% Conditional VaR (based on the Logistic distribution).
    By default it computes the 95% CVaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series).
    '''
    # Fitting b (scale parameter) to the variance of the data
    # Since variance of the Logistic dist.: var = b**2*pi**2/3
    scale = np.sqrt(3 * s.std(ddof=0)**2 / np.pi**2)
    return -scale * np.log(((1-level) ** (1 - 1 / level)) / level) + s.mean()

def Return_annualized(s, periods_per_year = None):
    '''
    Computes the return per year, or, annualized return.
    The variable periods_per_year can be, e.g., 12, 52, 252, in 
    case of monthly, weekly, and daily data.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the annualized return for every column (Series) by using pd.aggregate
    '''
    if periods_per_year is None:
        periods_per_year = infer_trading_periods(s)
    if isinstance(s, pd.DataFrame):
        return s.aggregate( Return_annualized, periods_per_year=periods_per_year )
    elif isinstance(s, pd.Series):
        growth = (1 + s).prod()
        n_period_growth = s.shape[0]
        return growth**(periods_per_year/n_period_growth) - 1

def StdDev_annualized(s, periods_per_year = None, ddof=1):
    '''
    Computes the volatility per year, or, annualized volatility.
    The variable periods_per_year can be, e.g., 12, 52, 252, in 
    case of monthly, weekly, and daily data.
    The method takes in input either a DataFrame, a Series, a list or a single number. 
    In the former case, it computes the annualized volatility of every column 
    (Series) by using pd.aggregate. In the latter case, s is a volatility 
    computed beforehand, hence only annulization is done
    '''
    if periods_per_year is None:
        periods_per_year = infer_trading_periods(s)
    if isinstance(s, pd.DataFrame):
        return s.aggregate(StdDev_annualized, periods_per_year=periods_per_year )
    elif isinstance(s, pd.Series):
        return s.std(ddof=ddof) * (periods_per_year)**(0.5)
    elif isinstance(s, list):
        return np.std(s, ddof=ddof) * (periods_per_year)**(0.5)
    elif isinstance(s, (int,float)):
        return s * (periods_per_year)**(0.5)

def Sharpe_ratio(s, risk_free_rate= 0 , periods_per_year = None, v = None):
    '''
    Computes the annualized sharpe ratio. 
    The variable periods_per_year can be, e.g., 12, 52, 252, in case of yearly, weekly, and daily data.
    The variable risk_free_rate is the annual one.
    The method takes in input either a DataFrame, a Series or a single number. 
    In the former case, it computes the annualized sharpe ratio of every column (Series) by using pd.aggregate. 
    In the latter case, s is the (allready annualized) return and v is the (already annualized) volatility 
    computed beforehand, for example, in case of a portfolio.
    '''
    if periods_per_year is None:
        periods_per_year = infer_trading_periods(s)
    if isinstance(s, pd.DataFrame):
        return s.aggregate( Sharpe_ratio, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year, v=None)
    elif isinstance(s, pd.Series):
        # convert the annual risk free rate to the period assuming that:
        # RFR_year = (1+RFR_period)^{periods_per_year} - 1. Hence:
        rf_to_period = (1 + risk_free_rate)**(1/periods_per_year) - 1        
        excess_return = s - rf_to_period
        # now, annualize the excess return
        ann_ex_rets = Return_annualized(excess_return, periods_per_year)
        # compute annualized volatility
        ann_vol = StdDev_annualized(s, periods_per_year)
        return ann_ex_rets / ann_vol
    elif isinstance(s, (int,float)) and v is not None:
        # Portfolio case: s is supposed to be the single (already annnualized) 
        # return of the portfolio and v to be the single (already annualized) volatility. 
        return (s - risk_free_rate) / v

def tracking_error(Ra, Rb):
    '''
    Computes the tracking error between two return series.
    Ra is the active return series.
    Rb is the benchmark return series.
    '''
    return(Ra.sub(Rb, axis=0).dropna().std())

def annualized_tracking_error(Ra, Rb, period = None):
    '''
    Computes the annualized tracking error between two return series.
    Ra is the active return series.
    Rb is the benchmark return series.
    '''
    if period is None:
        period = infer_trading_periods(Ra)
    return(tracking_error(Ra, Rb)*np.sqrt(period))

def checkData(x, method = "Series", na_rm = True, quiet = True, *args, **kwargs):
    '''
    
    Implements the checkData function from the R package performanceAnalytics
    
    This function was created to make the different kinds of data classes at least seem more fungible.
    It allows the user to pass in a data object without being concerned that the function requires a 
    matrix, data.frame, vector, or Series object. 
    By using checkData, the function "knows" what data format it has to work with.
    
    Parameters
    ----------
    x : a vector, matrix, data.frame, xts, timeSeries or zoo object to be checked and coerced
    
    method : type of coerced data object to return, one of 
    ["Series", data.frame", "matrix", "vector"], default "Series"
    na_rm : True/False Remove NA's from the data? used only with 'vector'
    quiet : True/False if false, it will throw warnings when errors are noticed, 
    default True.
    *args/ **kwargs : other passthrough parameters
    '''

    if method == "vector":
        if x.shape[1] > 1:
            if not quiet:
                print("The data provided is not a vector or univariate time series. Only the first column will be used.")
            x = x.iloc[:, 0]
        if na_rm:
            x = x.dropna()
        x = np.asarray(x).flatten()

    elif method == "matrix":
        x = np.asarray(x)

    elif method == "data.frame":
        x = pd.DataFrame(x)

    elif method == "Series":
        if isinstance(x, pd.Series):
            if isinstance(x.index, pd.DatetimeIndex):
                x = x
            else:
                x = pd.Series(x.values, index=pd.to_datetime(x.index))

        elif isinstance(x, pd.DataFrame):
            if x.shape[1] > 1:
                if not quiet:
                    print("The data provided is not a vector or univariate time series. Only the first column will be used.")
                x = x.iloc[:, 0]
            x = pd.Series(x.values, index=pd.to_datetime(x.index))

        elif isinstance(x, np.ndarray):
            if x.ndim > 1:
                if not quiet:
                    print("The data provided is not a vector or univariate time series. Only the first column will be used.")
                x = x[:, 0]
            x = pd.Series(x)

        elif isinstance(x, np.ndarray) and isinstance(x, np.number):
            if x.columns is None:
                x = pd.Series(np.matrix(x).squeeze())
            else:
                x = pd.Series(np.matrix(x).squeeze(), index=pd.to_datetime(x.columns))

    return x


def charts_PerformanceSummary(returns):
    '''
    Implements the charts.PerformanceSummary function from R,
    assuming geometrically compounded returns.
    
    Plots the cumulative return of all series, period returns of the leftmost return series,
    and the drawdown charts of all series.

    '''
    if type(returns) == pd.Series:
        returns = pd.DataFrame(returns)
    
    period = infer_trading_periods(returns)
    
    cumulative_wealth_index = compound_returns(returns)-1
    drawdowns = Drawdowns(returns)
        
    # Creating a figure and subplots with shared x-axis
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
    
    # Adjusting spacing between subplots
    #fig.subplots_adjust(hspace=0.2)
    
    # Plotting cumulative wealth index
    axs[0].plot(cumulative_wealth_index.index, cumulative_wealth_index.values, 
                label = returns.columns)
    axs[0].grid(True)
    axs[0].set_title('Cumulative Return', loc='left', fontweight='bold')
    
    # Plotting daily returns
    if(type(returns)==pd.Series):
        axs[1].plot(returns.index, returns.values)
    else:
        axs[1].plot(returns.index, returns.values[:,0])
    axs[1].grid(True)
    if(period==252):
        axs[1].set_title('Daily Return', loc='left', fontweight='bold')
    elif(period==52):
        axs[1].set_title('Weekly Return', loc='left', fontweight='bold')
    elif(period==12):
        axs[1].set_title('Monthly Return', loc='left', fontweight='bold')
    elif(period==4):
        axs[1].set_title('Quarterly Return', loc='left', fontweight='bold')
    elif(period==1):
        axs[1].set_title('Yearly Return', loc='left', fontweight='bold')

    # Plotting drawdowns
    axs[2].plot(drawdowns.index, drawdowns.values)
    axs[2].grid(True)
    axs[2].set_title('Drawdown', loc='left', fontweight='bold')
    
    # Setting x-axis limit
    last_data_point = len(cumulative_wealth_index) - 1
    plt.xlim(cumulative_wealth_index.index[0], cumulative_wealth_index.index[last_data_point])
    
    # Adding a title to the figure
    fig.suptitle('Performance Summary', fontsize=14, fontweight='bold')
    
    # Adding first and last timestamps string
    first_timestamp = cumulative_wealth_index.index[0].strftime('%Y-%m-%d')
    last_timestamp = cumulative_wealth_index.index[last_data_point].strftime('%Y-%m-%d')
    time_string = f'{first_timestamp} / {last_timestamp}'
    axs[0].text(1, 1.1, time_string, transform=axs[0].transAxes,
                fontsize=12, fontweight='bold', color='black',
                va='top', ha='right')
    
    if type(returns)==pd.DataFrame:
    # Adding legend to the first plot
        handles, labels = axs[0].get_legend_handles_labels()
        labels = returns.columns
        axs[0].legend(handles, labels, loc='upper left', frameon=True)
    
    # Displaying the plots
    plt.show()
    

def table_Drawdowns(R, top=5, digits=4):
    '''
    Implements R's table.Drawdowns function
    Parameters:
    ---------------------
    R: a series of returns
    top: the number of drawdowns, by depth, to return
    digits: the number of digits to round the drawdown to
    
    Returns:
    ---------------------
    A pd.DataFrame containing the top N drawdowns,
    Sorted by depth
    '''
    
    if isinstance(R, pd.Series):
        x = R
    else:
        x = R.iloc[:, 0]
    R = R.dropna()
    x = sortDrawdowns(findDrawdowns(R))
    ndrawdowns = sum(x['return'] < 0)
    if ndrawdowns < top:
        print("Only {} available in the data.".format(ndrawdowns))
        top = ndrawdowns
    
    x.loc[x['to'] == len(R), 'to'] = np.nan
    x.loc[pd.isnull(x['to']), 'recovery'] = np.nan
    
    to_indices = x['to'][0:top].apply(lambda idx: R.index[int(idx)] if not np.isnan(idx) else pd.NaT)

    result = pd.DataFrame({
        'From': R.index[x['from'][0:top]],
        'Trough': R.index[x['trough'][0:top]],
        'To': to_indices,
        'Depth': np.round(x['return'][0:top], digits),
        'Length': x['length'][0:top],
        'To Trough': x['peaktotrough'][0:top],
        'Recovery': x['recovery'][0:top]
    })
    return result

def sortDrawdowns(runs):
    index_sorted = np.argsort(runs['return'])
    runs_df = pd.DataFrame(runs)
    runs_sorted = runs_df.iloc[index_sorted,:]
    return runs_sorted

def findDrawdowns(R):
    if type(R)==pd.Series:
        x = R
    else:
        x = R.iloc[:,0]
    drawdowns = Drawdowns(x)
    draw = []
    begin = []
    end = []
    trough = []
    index = 0
    if drawdowns[0] >= 0:
        priorSign = 1
    else:
        priorSign = 0
    from_ = 0
    sofar = drawdowns[0]
    to = 0
    dmin = 0
    for i in range(len(drawdowns)):
        thisSign = 0 if drawdowns[i] < 0 else 1
        if thisSign == priorSign:
            if drawdowns[i] < sofar:
                sofar = drawdowns[i]
                dmin = i
            to = i + 1
        else:
            draw.append(sofar)
            begin.append(from_)
            trough.append(dmin)
            end.append(to)
            from_ = i
            sofar = drawdowns[i]
            to = i + 1
            dmin = i
            index += 1
            priorSign = thisSign
    draw.append(sofar)
    begin.append(from_)
    trough.append(dmin)
    end.append(to)
    result = {
        'return': draw,
        'from': begin,
        'trough': trough,
        'to': end,
        'length': np.array(end) - np.array(begin) + 1,
        'peaktotrough': np.array(trough) - np.array(begin) + 1,
        'recovery': np.array(end) - np.array(trough)
    }
    return result
