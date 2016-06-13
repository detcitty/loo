#!/usr/bin/env python
# -*- coding: utf-8 -*-

%matplotlib inline


import pandas as pd
import numpy as np
import scipy as sp
import patsy
import pystan

from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()



R = ro.r

import statsmodels.api as sm

import rpy2.robjects as ro
#import pandas.rpy.common as com

import matplotlib.pyplot as platform
pd.set_option('max_columns', 50)

from decimal import *

print getcontext()


url = "http://stat.columbia.edu/~gelman/arm/examples/arsenic/wells.dat"
wells = pd.read_table(url, sep='\s+', header=0, index_col=0)
print wells.head()
wells["dist100"] = wells['dist']/100

y = wells['switch']
wellsNp = wells.as_matrix()
y_mod = wellsNp[:, 0]

print y_mod

#print y.shape()
y_hat = y.transpose()

#print y_hat.shape()

#y_tilda = wells.as_matrix(columns=[2])
#print "This is y_tilda", y_tilda
y_mean = np.mean(y_mod)
#print y_mean

s = sp.special.logit(y_mean)
print s
#logit = sm.Logit(np.array(np.mean(y)), wells)
#fit = logit.fit()
logit = np.log(y_mean / (1 - y_mean))
#print fit.results()
#print log
f = "~ 0 + dist + arsenic"
mat = patsy.dmatrix(f, wells, return_type='dataframe')
nrow, ncol = mat.shape
#print nrow, ncol


#print mat.columns
#print np.array(mat['dist'])
#print np.array(mat['arsenic'])



#print logit.results()
data_loo = {'N': nrow , 'P': ncol, 'y': y, 'x': mat, 'a': s }
print data['a']


logistic = '''
data {
  int<lower=0> N; 
  int<lower=0> P;
  int<lower=0,upper=1> y[N];
  matrix[N,P] x; 
  real a;
}
parameters {
  real beta0;
  vector[P] beta;
}
model {
  beta0 ~ student_t(7, a, 0.1);
  beta ~ student_t(7, 0, 1);
  y ~ bernoulli_logit(beta0 + x * beta);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] <- bernoulli_logit_log(y[n], beta0 + x[n] * beta);
}
'''


fit1 = pystan.stan(logistic, data=data_loo, iter=1000, chains=4)