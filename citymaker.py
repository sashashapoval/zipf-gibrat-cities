import numpy as np

def powercdf(x, a=0.01, M=1, gamma=2):
    if a <=0:
        return None
    if gamma == 1:
        return np.log( (x+a) / a ) / np.log( (M+a) / a )
    else:
        numerator = (x+a) ** (1-gamma) - a** (1-gamma)
        denominator = (M+a) ** (1-gamma) - a** (1-gamma)
        return numerator / denominator
    
def powerpdf(x, a=0.01, M=1, gamma=2):
    if a <=0:
        return None
    if gamma == 1:
        return 1 / ( (x+a) * np.log((M + a) / a)  )
    else:
        return (1-gamma) / ( ( (M+a) ** (1-gamma) - a** (1-gamma) ) * (x+a)**gamma )

def power_left_hand_side(x, a, M, mass, gamma):
    if a <=0:
        return None
    x = np.array(x)
    return powercdf(np.array(x), a=a, M=M, gamma=gamma) -\
        np.sqrt(powerpdf(np.array(x), a=a, M=M, gamma=gamma) / (2*mass))    
        
def power_num_cities(a, M, mass, gamma=1):
    if a <=0:
        return None
    if gamma == 1:
        return np.sqrt( (2*mass) / np.log(1+M/a) )
    else:
        m1 = M**( (2*(1-gamma)) / (2-gamma)) / (2-gamma)
        m2 = np.sqrt( 2*mass*(1-gamma) / ( (M+a)**(1-gamma) - a**(1-gamma) ) )
        return m1 * m2

import numpy as np
from scipy.stats import weibull_min
def weibullcdf(x, k, loc=0, scale=1, M=1):
    norm = 1 / (weibull_min.cdf(M, k, loc, scale) - weibull_min.cdf(0, k, loc, scale))
    return norm * (weibull_min.cdf(x, k, loc, scale) - weibull_min.cdf(0, k, loc, scale))

from scipy.stats import weibull_min
def weibullpdf(x, k, loc=0, scale=1, M=1):
    norm = 1 / (weibull_min.cdf(M, k, loc, scale) - weibull_min.cdf(0, k, loc, scale))
    return norm * weibull_min.pdf(x, k, loc, scale)

def weibull_left_hand_side(x, k, loc=0, scale=1, M=1, mass=1):
    return weibullcdf(np.array(x), k, loc=loc, scale=scale, M=M) -\
        np.sqrt(weibullpdf(np.array(x), k, loc=loc, scale=scale, M=M) / (2*mass))

import matplotlib.pyplot as plt
def plot_weibull_lhs(loc, scale, k, M, mass=1):
    fig, ax = plt.subplots(1, 1, figsize=(5,3))
    y = np.linspace(np.log10(0.01), np.log10(M), 1000)
    x = np.exp(y*np.log(10))
    ax.plot(x, weibull_left_hand_side(x, k, loc=loc, scale=scale, M=M, mass=mass))
    hl = ax.set_ylabel(r'$\mathtt{lhs}$')
    ax.set_xscale('log')
    print(f'lhs_max = {weibull_left_hand_side(M, k, loc=loc, scale=scale, M=M, mass=mass)}')
    return fig, ax

from scipy.optimize import fsolve
def model_cities(k, loc=0, scale=1, M=1, mass=1, delta=1, xtol=1e-9, mon='increase', stop=100000):
    lhs_max = weibull_left_hand_side(M, k, loc=loc, scale=scale, M=M, mass=mass) #maximal value of left hand side of the key equation
    def func(x, *arg): #to solve func(x) = 0, func is power_left_hand_side where arg contains non-x parameters
        k, loc, scale, rhs = arg
        return weibull_left_hand_side(x, k, loc=loc, scale=scale, M=M) - rhs
    #---main loop:preparation
    cdf_cur = 0
    root = 0
    median = []
    cdf_vals = []
    cdf_cutoff = min(delta, lhs_max)
    #---main loop
    num = 0
    num_cities2prn = 5000 #print about each computed num_cities2prn
    while cdf_cur < cdf_cutoff:
        root_new = fsolve(func, root, args=(k, loc, scale, cdf_cur), xtol=xtol)[0]#new median is a root
        if mon == 'increase' and root_new <=  root:
            break
        elif mon == 'decrease' and root_new >= root:
            break
        elif mon != 'increase' and mon != 'decrease':
            print(f'model_cities: Exception: wrong value {mon} of mon');
            break
        root = root_new
        median.append(root)
        cdf_cur = 2*weibullcdf(root, k, loc=loc, scale=scale, M=M) - cdf_cur
        cdf_vals.append(cdf_cur)
        num += 1
        if num % num_cities2prn == 0:
            print(f'Number of cities is {num}')
        if num >= stop:
            print('Maximal number of cities is reached');
            break
    if cdf_cur > 1 and len(cdf_vals) > 0:
        cdf_vals[-1] = 1
    return cdf_vals, median

import re
def efloat2txt(f : float):
    f_txt=format(f, 'e')
    #collect ending zeros after e as in 1.0200e+10 and collect nothing if zeros are absent
    f_end_zeros = re.findall('0$', f_txt)
    if len(f_end_zeros) > 0:
        f_end = f_end_zeros[0]
    else:
        f_end = ''
    #---eliminate ending zeros from the floating part located before e as in 1.0200e+10
    i_e = f_txt.find('e')
    f_to_e = f_txt[:i_e]
    f_to_e_end_zeros = re.findall('0+$', f_to_e)
    if len(f_to_e_end_zeros) > 0:
        f_to_e = f_to_e[:(len(f_to_e)-len(f_to_e_end_zeros[0]))]
        #---floating point to _ if the floating part exists as in 1.0200e+10 or to nothing as in 1.0e+10
    i_dot = f_to_e.find('.')
    if i_dot >=0 and len(f_to_e) >= 2 and f_to_e[-1] > '0' and f_to_e[-1] <= '9':
        dot2symbol = '_'
    else:
        dot2symbol = ''
    return f_txt.replace('0', '').replace('.', dot2symbol)+f_end

#save when x_current >= x_previous * lgap
def thinout(x:list, val:list, lgap: float):
    xcur = 1
    x_thinned = []
    v_thinned = []
    for j in range(len(x)):
        if x[j] >= xcur:
            x_thinned.append(x[j])
            v_thinned.append(val[j])
            xcur = np.max([xcur+1, int(round(xcur*lgap))])
    return(x_thinned, v_thinned)

#transform initial (alphabetical) database
#deleting '.' that separates the last 3 digits in integers
#converting each value from string to integer
#and sorting accoding to values in 2021
from scipy.stats import rankdata
import pandas as pd
def preprocess_census_full(df):
    df = df.rename(columns={'Unnamed: 0':'Area', 'Unnamed: 1':'Base'}).dropna()
    dfcensus = pd.DataFrame({
        'name': df['Area'],
        'value2021': [(str(v).replace(',','')) for v in df['2021']],
        'value2020': [(str(v).replace(',','')) for v in df['2020']]
    }) #.sort_values(by='value').reset_index(drop=True)
    dfcensus = dfcensus.astype({'name': 'str', 'value2021': 'int', 'value2020': 'int'})
    up_pop = dfcensus['value2021'].sum()
    dfcensus['pdf21'] = [v / up_pop for v in dfcensus['value2021']]
    dfcensus['rank'] = len(dfcensus['value2021']) - rankdata(dfcensus['value2021']) + 1
    dfcensus = dfcensus.sort_values(by='rank').reset_index(drop=True)
    return dfcensus

#transform initial (alphabetical) database
#deleting '.' that separates the last 3 digits in integers
#converting each value from string to integer
#and sorting accoding to values in 2021
from scipy.stats import rankdata
import pandas as pd
def preprocess_census_full_xslx(df):
    df = df.rename(columns={'Unnamed: 0':'Area', 'Unnamed: 1':'Base', 2020:'2020', 2021:'2021'}).dropna()#columns titles are taken as they are in the input
    dfcensus = pd.DataFrame({
        'name': df['Area'],
        'value2021': [int(v) for v in df['2021']],
        'value2020': [int(v) for v in df['2020']]
    }) 
    dfcensus = dfcensus.astype({'name': 'str', 'value2021': 'int', 'value2020': 'int'})#columns are declared as str and int in line with the content
    #---create new columns: pdf and rand
    up_pop = dfcensus['value2021'].sum()
    dfcensus['pdf21'] = [v / up_pop for v in dfcensus['value2021']]
    dfcensus['rank'] = len(dfcensus['value2021']) - rankdata(dfcensus['value2021']) + 1
    dfcensus = dfcensus.sort_values(by='rank').reset_index(drop=True)
    return dfcensus


#x[] and y[] must be defined as np.array()
def regress(x, y):
    import numpy as np
    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression
    if len(np.shape(x)) == 1:
        ncols = 1
    else:
        ncols = len(x[0])
    num4fit = 50 #Number of points in the fit
    model = LinearRegression(fit_intercept=True)
    X = x.reshape(len(x), ncols)#otherwise, with ncols=1, shape(x) = (n,) but (n, 1) required
    #X = x[:, np.newaxis]
    model.fit(X, y)
    #the estimates of beta are in model.intercept_ and model_coef; save to new array beta_hat
    beta_hat = np.empty(shape=(len(model.coef_)+1,), dtype=float)
    beta_hat[0] = model.intercept_
    beta_hat[1:] = model.coef_
    xfit = np.empty(shape=(num4fit, ncols))
    for i in range(len(xfit[0])):
        xfit[:, i] = np.linspace(np.min(X[:, i]), np.max(X[:, i]), num=num4fit)
    yfit = model.predict(xfit)
    X_with_intercept = np.empty(shape=(len(x), ncols+1), dtype=float)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:ncols+1] = X
    #alternative way to find the estimates of beta is in the next (commented) line
    #beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    #beta_hat = beta_hat.reshape((len(beta_hat),))
    y_hat = model.predict(X)
    residuals = y - y_hat
    residual_sum_of_squares = residuals.T @ residuals
    sigma_squared_hat = residual_sum_of_squares / (len(x)-2)
    #covariance matrix, where the variances are along the diagonals
    var_beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * sigma_squared_hat
    dev = y_hat - np.mean(y)
    dev_sum_squared = dev.T @ dev
    r_squared = dev_sum_squared / residual_sum_of_squares
    dev_y = y - np.mean(y)
    dev_y_sum_squared = dev_y.T @ dev_y
    r_squared = dev_sum_squared / dev_y_sum_squared
    if np.shape(xfit)[0] == 1:
        xfit = xfit.reshape((len(xfit),))
    return beta_hat, var_beta_hat, xfit, yfit, residuals, r_squared

#Find and draw regression line
def plt_fit(xx, yy, ax, lnwdth=0.8, lnstyle='-', color=''):
    import numpy as np
    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression
    import scipy.stats
    model = LinearRegression(fit_intercept=True)
    beta_hat, var_beta_hat, xfit, yfit, residuals, r_squared = regress(np.log(xx), np.log(yy))
    yy_pred = np.exp(beta_hat[0] + beta_hat[1] * np.log(xx))
    anntxt = r'$\sim T^{{{:.2f}}}$'.format(slope)
    if color == '':
        ax.plot(xx, yy_pred, label = anntxt, linewidth = lnwdth, linestyle=lnstyle)
    else:
        ax.plot(xx, yy_pred, label = anntxt, linewidth = lnwdth, linestyle=lnstyle, color=color)


