# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:21:56 2023

@author: ilyak
"""

import numpy as np
import pandas as pd
import warnings as wn


def make_demo():
    prices = get_sample_prices()
    rets = prices.pct_change().dropna()
    weights = pd.DataFrame([
        {'Date': '2006-08-01', 'SPY': .5, 'TLT': .25, 'DBC': .25},
        {'Date': '2023-04-03', 'SPY': .5, 'TLT': .25, 'DBC': .25}
    ])
    weights.set_index('Date', inplace=True)
    weights.index = pd.to_datetime(weights.index)
    return get_portfolio_return(rets, weights)
    
    
def get_sample_prices():
    import yfinance as yf

    # Download data for SPY and TLT
    symbols = ["SPY", "TLT", "DBC"]
    start_date = "1990-01-01"
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    df_list = []
    for symbol in symbols:
        data = yf.download(symbol, start=start_date, end=end_date)
        data = data["Adj Close"]
        data.name = symbol
        df_list.append(data)
    
    # Combine the data frames into one
    df = pd.concat(df_list, axis=1)
    
    return df.dropna()


def get_rebalance_endpoints(df, on = "M", offset = 0):
    """
    Returns index of endpoints of a time series analogous to R's endpoints
    function.
    Takes in:
        df -- a dataframe/series with a date index
         
        on -- a string specifying frequency of endpoints
         
        (E.G. "M" for months, "Q" for quarters, and so on)
         
        offset -- to offset by a specified index on the original data
        (E.G. if the data is daily resolution, offset of 1 offsets by a day)
        This is to allow for timing luck analysis. Thank Corey Hoffstein.
    """
     
    # to allow for familiarity with R
    # "months" becomes "M" for resampling
    if len(on) > 3:
        on = on[0].capitalize()
     
    # get index dates of formal endpoints
    ep_dates = pd.Series(df.index, index = df.index).resample(on).max()
     
    # get the integer indices of dates that are the endpoints
    date_idx = np.where(df.index.isin(ep_dates))
     
    # append last day
    date_idx = np.append(date_idx, df.shape[0]-1)
    if offset != 0:
        date_idx = date_idx + offset
        date_idx[date_idx < 0] = 0
        date_idx[date_idx > df.shape[0]-1] = df.shape[0]-1
    
    return np.unique(date_idx)  


def compute_drifting_weights(w):
    return w / np.sum(w, axis=1).to_numpy()[:, np.newaxis]


def compute_weights_and_returns(subset_returns, weights):
    subset_returns = subset_returns.copy()
    # assuming trading on close, return on rebalance day is 0 for new position
    subset_returns.iloc[0,:] = 0
    
    rep_weights = np.tile(weights, (len(subset_returns), 1))
    # cumulative returns weights
    cum_subset_weights = np.cumprod(1 + subset_returns, axis=0) * rep_weights
    cum_subset_weights_bop = cum_subset_weights / (1 + subset_returns)
    # End-Of-Period weights
    EOP_Weight = compute_drifting_weights(cum_subset_weights)    
    # Beginning-Of-Period weights
    BOP_Weight = compute_drifting_weights(cum_subset_weights_bop)
    portf_returns_subset = pd.DataFrame(
        np.sum(subset_returns.values * BOP_Weight, axis=1), 
        index=subset_returns.index, 
        columns=['Portfolio.Returns']
    )

    return [portf_returns_subset, BOP_Weight, EOP_Weight]



def compute_turnover(w):
    # add first rebalance, as we start with a non-invested portfolio
    # not exact if cash is a valid position since we should consider being 100% invested in cash
    w = pd.concat([w, pd.DataFrame(0, index=[w.index[0]], columns=w.columns)])
    turnover = w.groupby(w.index).diff().abs().dropna(how='all').sum(axis=1)
    turnover.name = 'Turnover'
    
    return turnover


# Return.portfolio.geometric from R
def get_portfolio_return(R, weights=None, verbose=True, rebalance_on=None):
    """

    Parameters
    ----------
    R : a pandas series of asset returns
    weights : a vector or pandas series of asset weights.
    verbose : a boolean specifying a verbose output containing:
        portfolio returns,
        beginning of period weights and values, end of period weights and values,
        asset contribution to returns, and two-way turnover calculation
    rebalance_on : a string specifying rebalancing frequency if weights are passed in as a vector.
        e.g. 'months'

    Raises
    ------
    ValueError
        Number of asset weights must be equal to the number of assets.

    Returns
    -------
    TYPE
        See verbose parameter for True value, otherwise just portfolio returns.

    """  
    R = R.copy()
  
    # impute NAs in returns
    if R.isna().sum().sum() > 0:
        R.fillna(0, inplace=True)
        wn.warn("NAs detected in returns. Imputing with zeroes.")
      
    # if no weights provided, create equal weight vector
    if weights is None:
        #weights = np.repeat(1/R.shape[1], R.shape[1])
        weights = [1/R.shape[1]] * R.shape[1]
        wn.warn("Weights not provided, assuming equal weight for rebalancing periods.")
    else:
        weights = weights.copy()
      
    # if weights aren't passed in as a data frame (they're probably a list)
    # turn them into a 1 x num_assets data frame
    if isinstance(weights, list):
        weights = pd.Series(weights, index=R.columns).to_frame().T
    elif isinstance(weights, pd.DataFrame):
        pass
    else:
        raise TypeError("Only list or pd.DataFrame are accepted")
    
    # error checking for same number of weights as assets
    if weights.shape[1] != R.shape[1]:
        raise ValueError("Number of weights is unequal to number of assets. Correct this.")

    # if there's a row vector of weights, create a data frame with  the desired 
    # rebalancing schedule --  also add in the very first date into the schedule
    if weights.shape[0] == 1 and weights.index[0] == 0:
        if rebalance_on is not None:
            ep = get_rebalance_endpoints(R, on = rebalance_on)
            weights = pd.DataFrame(
                np.tile(weights, (len(ep), 1)), 
                index=R.index[ep], 
                columns=R.columns
            )
            
            # add first date if not already there
            if R.index[0] != weights.index[0]:
                first_weights = pd.DataFrame(
                    weights.values, 
                    index=[R.index[0]], 
                    columns=R.columns
                )
                weights = pd.concat([first_weights, weights], axis=0)
        else:
            weights.index = [R.index[0]]
        
        weights.index.name = R.index.name
          
    if weights.isna().sum().sum() > 0:
      weights.fillna(0, inplace=True)
      wn.warn("NAs detected in weights. Imputing with zeroes.")
      
    residual_weights = 1 - weights.sum(axis=1)
    if abs(residual_weights).sum() != 0:
        print("One or more periods do not have investment equal to 1. Creating residual weights.")
        weights["Residual"] = residual_weights
        R["Residual"] = 0
     
    if weights.shape[0] > 1:
        portf_returns, bop_weights, eop_weights = [], [], []
        for i in range(weights.shape[0]):
            try:
                subset = R.loc[
                    (R.index >= weights.index[i]) & \
                    (R.index <= weights.index[i+1])
                ].copy()
            except:
                # last rebalance
                subset = R.loc[R.index >= weights.index[i]].copy()

            subset_out = compute_weights_and_returns(subset, weights.iloc[i,:])
            subset_out[0].columns = ["Portfolio.Returns"]
            if i > 0: 
                # drop first period as already there, not for EOD (for turnover comp)
                subset_out = [s.iloc[1:] for s in subset_out[:2]] + [subset_out[2]]
            
            portf_returns.append(subset_out[0])
            bop_weights.append(subset_out[1])
            eop_weights.append(subset_out[2])
            
        portf_returns = pd.concat(portf_returns, axis=0)
        bop_weights = pd.concat(bop_weights, axis=0)
        eop_weights = pd.concat(eop_weights, axis=0)
    else: # only one weight allocation, just drift the portfolio
        subset = R.loc[R.index >= weights.index[0]].copy()
        out = compute_weights_and_returns(subset, weights)
        portf_returns = out[0]; portf_returns.columns = ['Portfolio.Returns']
        bop_weights = out[1]
        eop_weights = out[2]
      
    if not verbose:
        return portf_returns
      
    turnover = compute_turnover(eop_weights.loc[weights.index, weights.columns])
    # droping duplicate indices due to loop in case of multiple rebalancing
    # need to keep them to compute turnover
    eop_weights = eop_weights[~eop_weights.index.duplicated(keep='last')]
    
    pct_contribution = R * bop_weights
    cum_returns = (1 + portf_returns).cumprod()

    eop_value = eop_weights * pd.DataFrame(
        np.tile(cum_returns, (1, eop_weights.shape[1])), 
        index=eop_weights.index, 
        columns=eop_weights.columns
    )
    bop_value = bop_weights * pd.DataFrame(
        np.tile(cum_returns/(1+portf_returns), (1, bop_weights.shape[1])), 
        index=bop_weights.index, 
        columns=bop_weights.columns
    )
    
    out = [portf_returns, pct_contribution, bop_weights, eop_weights, 
           bop_value, eop_value, turnover]
    out = {k: v for k, v in zip(['returns', 'contribution', 'BOP.Weight', 
                                 'EOP.Weight', 'BOP.Value', 'EOP.Value', 
                                 'Two.Way.Turnover'], out)}
    
    return out
    
      
if __name__ == '__main__':
    res = make_demo()
    