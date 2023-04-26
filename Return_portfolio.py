# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:21:56 2023

@author: ilyak
"""

import numpy as np
import pandas as pd
import warnings as wn
import yfinance as yf

# Download data for SPY and TLT
symbols = ["SPY", "TLT"]
start_date = "1990-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
df_list = []
for symbol in symbols:
    data = yf.download(symbol, start=start_date, end=end_date)
    # Calculate daily returns
    data["return"] = data["Adj Close"].pct_change()
    data = data[["return"]]
    data = data.rename(columns={"return": symbol})
    df_list.append(data)

# Combine the data frames into one
df = pd.concat(df_list, axis=1)
df = df.dropna()

def endpoints(df, on = "M", offset = 0):
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
     
    # append last day to match R's endpoints function
    # remember, Python is indexed at 0, not 1
    #date_idx = np.insert(date_idx, 0, 0)
    date_idx = np.append(date_idx, df.shape[0]-1)
    if offset != 0:
        date_idx = date_idx + offset
        date_idx[date_idx < 0] = 0
        date_idx[date_idx > df.shape[0]-1] = df.shape[0]-1
    out = np.unique(date_idx)
    return out  

def compute_weights_and_returns(subset_returns, weights):
  rep_weights = np.tile(weights, (len(subset_returns), 1))
  cum_subset_weights = np.cumprod(1 + subset_returns, axis=0) * rep_weights
  EOP_Weight = cum_subset_weights / np.sum(cum_subset_weights, axis=1)[:, np.newaxis]
  cum_subset_weights_bop = cum_subset_weights / (1 + subset_returns)
  BOP_Weight = cum_subset_weights_bop / np.sum(cum_subset_weights_bop, axis=1)[:, np.newaxis]
  portf_returns_subset = pd.DataFrame(np.sum(subset_returns.values * BOP_Weight, axis=1), 
                                        index=subset_returns.index, columns=['Portfolio.Returns'])
  return [portf_returns_subset, BOP_Weight, EOP_Weight]


# Return.portfolio.geometric from R
def Return_portfolio(R, weights=None, verbose=True, rebalance_on='months'):
    
  # impuute NAs in returns
  if R.isna().sum().sum() > 0:
    R.fillna(0, inplace=True)
    wn.warn("NAs detected in returns. Imputing with zeroes.")
    
    # if no weights provided, create equal weight vector
  if weights is None:
    weights = np.repeat(1/R.shape[1], R.shape[1])
    wn.warn("Weights not provided, assuming equal weight for rebalancing periods.")
    
    # if weights aren't passed in as a data frame (they're probably a list)
    # turn them into a 1 x num_assets data frame
  if type(weights) != pd.DataFrame:
    weights = pd.DataFrame(weights)
    weights = weights.transpose()
    
    # error checking for same number of weights as assets
  if weights.shape[1] != R.shape[1]:
        raise ValueError("Number of weights is unequal to number of assets. Correct this.")
    
    
  # if there's a row vector of weights, create a data frame with  the desired 
  # rebalancing schedule --  also add in the very first date into the schedule
  if weights.shape[0] == 1:
        if rebalance_on is not None:
            ep = endpoints(R, on = rebalance_on)
            first_weights = pd.DataFrame(np.array(weights), index=[R.index[0]], columns=R.columns)
            weights = pd.concat([first_weights, pd.DataFrame(np.tile(weights, 
                        (len(ep), 1)), index=R.index[ep], columns=R.columns)], axis=0)
            #weights = pd.DataFrame(np.tile(weights, (len(ep), 1)), 
            #                       index = ep.index, columns = R.columns)
            weights.index.name = R.index.name
        else:
            weights = pd.DataFrame(weights, index=[R.index[0]], columns=R.columns)
    
  original_weight_dim = weights.shape[1] # save this for computing two-way turnover

    
  if weights.isna().sum().sum() > 0:
    weights.fillna(0, inplace=True)
    wn.warn("NAs detected in weights. Imputing with zeroes.")
    
  residual_weights = 1 - weights.sum(axis=1)
  if abs(residual_weights).sum() != 0:
    print("One or more periods do not have investment equal to 1. Creating residual weights.")
    weights["Residual"] = residual_weights
    R["Residual"] = 0
    
  if weights.shape[0] != 1:
    portf_returns, bop_weights, eop_weights = [], [], []
    for i in range(weights.shape[0]-1):
      subset = R.loc[(R.index >= weights.index[i]) & (R.index <= weights.index[i+1])]
      if i >= 1: # first period include all data
                # otherwise, final period of previous period is included, so drop it
        subset = subset.iloc[1:]
      subset_out = compute_weights_and_returns(subset, weights.iloc[i,:])
      subset_out[0].columns = ["Portfolio.Returns"]
      portf_returns.append(subset_out[0])
      bop_weights.append(subset_out[1])
      eop_weights.append(subset_out[2])
    portf_returns = pd.concat(portf_returns, axis=0)
    bop_weights = pd.concat(bop_weights, axis=0)
    eop_weights = pd.concat(eop_weights, axis=0)
  else: # only one weight allocation at the beginning and just drift the portfolio
    out = compute_weights_and_returns(R, weights)
    portf_returns = out[0]; portf_returns.columns = ['Portfolio.Returns']
    bop_weights = out[1]
    eop_weights = out[2]
    
    
  pct_contribution = R * bop_weights
  cum_returns = (1 + portf_returns).cumprod()
  eop_value = eop_weights * pd.DataFrame(np.tile(cum_returns, (1, eop_weights.shape[1])), 
                                        index=eop_weights.index, columns=eop_weights.columns)
  bop_value = bop_weights * pd.DataFrame(np.tile(cum_returns/(1+portf_returns), (1, bop_weights.shape[1])), 
                                        index=bop_weights.index, columns=bop_weights.columns)

  turnover = (np.abs(bop_weights.iloc[:, :(original_weight_dim-1)] - eop_weights.iloc[:, :(original_weight_dim-1)].shift(1))).sum(axis=1).dropna()
  turnover = pd.DataFrame(turnover, index=eop_weights.index[1:], columns=['Two-way turnover'])

  out = [portf_returns, pct_contribution, bop_weights, eop_weights, bop_value, eop_value, turnover]
  out = {k: v for k, v in zip(['returns', 'contribution', 'BOP.Weight', 'EOP.Weight', 'BOP.Value', 'EOP.Value', 'Two.Way.Turnover'], out)}
  if verbose:
        return out
  else:
        return portf_returns
