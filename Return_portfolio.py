import numpy as np
import pandas as pd
import warnings as wn


def make_demo():
    prices = get_sample_prices()
    rets = prices.pct_change().dropna()
    weights = pd.DataFrame([
        {'Date': '2006-08-01', 'SPY': .5, 'TLT': .25, 'DBC': .25},
        {'Date': '2010-08-02', 'SPY': .5, 'TLT': .25, 'DBC': .25},
        {'Date': '2023-04-03', 'SPY': .5, 'TLT': .25, 'DBC': .25},
        {'Date': '2023-05-03', 'SPY': .5, 'TLT': .25, 'DBC': .25},
    ])
    weights.set_index('Date', inplace=True)
    weights.index = pd.to_datetime(weights.index)

    return return_portfolio(rets, [.5, .25, .25], verbose=True, rebalance_on = 'months') #rebalance_on='months')#weights)
    
    
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
    return pd.concat(df_list, axis=1).dropna()
    

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
     
    # append last day
    date_idx = np.append(date_idx, df.shape[0]-1)
    if offset != 0:
        date_idx = date_idx + offset
        date_idx[date_idx < 0] = 0
        date_idx[date_idx > df.shape[0]-1] = df.shape[0]-1
    
    return np.unique(date_idx)  


def compute_drifting_weights(w):
    return w / np.sum(w, axis=1)[:, np.newaxis]


def compute_weights_and_returns(subset_returns, weights):
    subset_returns = subset_returns.copy()
    # assuming trading on close, return on rebalance day is 0 for new position
    subset_returns[0] = 0
    rep_weights = np.tile(weights, (subset_returns.shape[0], 1))
    # cumulative returns weights
    cum_subset_weights = np.multiply(
        np.cumprod(1 + subset_returns, axis=0), rep_weights
    )
    cum_subset_weights_bop = np.divide(cum_subset_weights, 1 + subset_returns)
    EOP_Weight = compute_drifting_weights(cum_subset_weights)    
    BOP_Weight = compute_drifting_weights(cum_subset_weights_bop)
    portf_returns_subset = np.sum(np.multiply(subset_returns, BOP_Weight), axis=1)

    return [portf_returns_subset, BOP_Weight, EOP_Weight]


def compute_turnover(w):
    # add first rebalance, as we start with a non-invested portfolio
    # not exact if cash is a valid position since we should consider being 100% invested in cash
    w = pd.concat([w, pd.DataFrame(0, index=[w.index[0]], columns=w.columns)])
    turnover = w.groupby(w.index).diff().abs().dropna(how='all').sum(axis=1)
    turnover.name = 'Turnover'
    
    return turnover


def prepare_weights(R, weights=None, rebalance_on=None):
    # if no weights provided, create equal weight vector
    if weights is None:
        weights = [1./R.shape[1]] * R.shape[1]
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
            ep = endpoints(R, on=rebalance_on)
            weights = pd.DataFrame(
                np.tile(weights, (len(ep), 1)), 
                index=R.index[ep], 
                columns=R.columns
            )
            # add first date if not already there
            if R.index[0] != weights.index[0]:
                first_weights = pd.DataFrame(
                    [weights.iloc[0,:].values], 
                    index=[R.index[0]], 
                    columns=R.columns
                )
                weights = pd.concat([first_weights, weights], axis=0)
        else:
            # in that case, we assume, as in R implementation, that one invests
            # one day before so that we dont lose 1 trading date
            R = pd.concat([
                pd.DataFrame(0, columns=R.columns, 
                             index=[R.index[0]-pd.Timedelta(1, "d")]),
                R
            ])
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
      
    return weights, R


def compute_risk_contribution(rets, weights):
    """
    Returns the percentage risk contribution of each assets over the period
    source: https://quantdare.com/risk-contribution-in-portfolio-management/
    Parameters
    ----------
    rets : np.ndarray
        TxN array of returns
    weights : np.array
        array of weights
    Returns
    -------
    np.ndarray
        percentage risk contribution
    """
    covariances = np.cov(rets[1:].T)
    portfolio_vol = np.sqrt(np.dot(np.dot(weights, covariances), weights))
    marginal_risk_contr = np.dot(weights, covariances) / portfolio_vol
    risk_contribution = weights * marginal_risk_contr
    
    return risk_contribution / portfolio_vol
    

def return_portfolio(R, weights=None, verbose=True, rebalance_on=None):
    """
    Parameters
    ----------
    R : a pandas series of asset returns
    weights : a vector or pandas series of asset weights.
    verbose : a boolean specifying a verbose output containing:
        portfolio returns,
        beginning of period weights and values, end of period weights and values,
        asset contribution to returns, asset contribution to risk,
        and two-way turnover calculation
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
      
    weights, R = prepare_weights(R, weights=weights, rebalance_on=rebalance_on)

    Wv = weights.values
    date_idx = np.where(R.index.isin(weights.index.tolist()))[0]
    Rv = R.values
    portf_returns, bop_weights, eop_weights = [], [], []
    for i in range(date_idx.shape[0]):
        try:
            subset = Rv[date_idx[i]:date_idx[i+1]+1]
        except:
            # last rebalance
            subset = Rv[date_idx[i]:]

        subset_out = compute_weights_and_returns(subset, Wv[i])
        #risk_contribution.append(compute_risk_contribution(subset, Wv[i]))
        
        if i > 0: 
            # drop first period as already there, not for EOD (for turnover comp)
            subset_out = [s[1:] for s in subset_out[:2]] + [subset_out[2]]
        
        portf_returns.append(subset_out[0])
        bop_weights.append(subset_out[1])
        eop_weights.append(subset_out[2])
    
    portf_returns = np.concatenate(portf_returns)
    bop_weights = np.concatenate(bop_weights)
    eop_weights = np.concatenate(eop_weights)
    #risk_contribution = np.concatenate(risk_contribution, axis = 0)
    
    if not verbose:
        return pd.Series(portf_returns, index=R.loc[weights.index[0]:].index, name='return').iloc[1:,]
      
    # using numpy, we have a problem because we dont have dates on axis
    increment = np.array([0]+list(range(weights.shape[0]-1)))
    date_idx_eop = date_idx + increment  # duplicate position, first occurence
    duplicate_positions = date_idx_eop + np.array([0] + [1] * (weights.shape[0]-1)) # duplicate position, second occurence
    pos = np.unique(np.concatenate((date_idx_eop, duplicate_positions)))
    pos = pos - date_idx[0] # because we truncate data before first rebalance
    weights_at_rebalance = eop_weights[pos]
    # reconstruct pd.DataFrame to use Groupby on dates
    weights_at_rebalance = pd.DataFrame(
        weights_at_rebalance, columns=weights.columns, 
        index=[
            weights.index[0]
        ] + list(np.sort(weights.index[1:].to_list()*2))
    )
    turnover = compute_turnover(weights_at_rebalance)
    # droping duplicate indices due to loop in case of multiple rebalancing
    # need to keep them to compute turnover
    eop_weights = np.delete(eop_weights, (date_idx_eop - date_idx[0])[1:], axis=0)
    pct_contribution = np.multiply(Rv[date_idx[0]:], bop_weights)    
    cum_returns = (1 + portf_returns).cumprod()
    eop_value = np.multiply(
        eop_weights, 
        np.tile(cum_returns, (eop_weights.shape[1], 1)).T
    )
    bop_value = np.multiply(
        bop_weights, 
        np.tile(cum_returns/(1+portf_returns), (bop_weights.shape[1], 1)).T
    )
    
    # recreate pd.DataFrames
    date_index = R.loc[weights.index[0]:].index
    cols = weights.columns
    portf_returns = pd.Series(portf_returns, index=date_index, name='return')
    pct_contribution = pd.DataFrame(pct_contribution, index=date_index, columns=cols)
    bop_weights = pd.DataFrame(bop_weights, index=date_index, columns=cols)
    eop_weights = pd.DataFrame(eop_weights, index=date_index, columns=cols)
    bop_value = pd.DataFrame(bop_value, index=date_index, columns=cols)
    eop_value = pd.DataFrame(eop_value, index=date_index, columns=cols)
    
    # remove first day of zeroes and trailing 1 for turnover
    portf_returns = portf_returns.iloc[1:,]
    bop_weights = bop_weights.iloc[1:,]
    eop_weights = eop_weights.iloc[1:,]
    pct_contribution = pct_contribution.iloc[1:,]
    bop_value = bop_value.iloc[1:,]
    eop_value = eop_value.iloc[1:,]
    turnover = turnover.iloc[0:(len(turnover)-1),]
    
    #risk_contribution = pd.DataFrame(risk_contribution, index=weights.index, columns=cols)
    out = [portf_returns, pct_contribution, bop_weights, eop_weights, 
           bop_value, eop_value, turnover]
    out = {k: v for k, v in zip(['returns', 'contribution', 'BOP.Weight', 
                                 'EOP.Weight', 'BOP.Value', 'EOP.Value', 
                                 'Two.Way.Turnover'], out)}
    
    return out
    
      
if __name__ == '__main__':
    res = make_demo()
