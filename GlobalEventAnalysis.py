
# coding: utf-8

# Price Prediction Based on World Events

# In[3]:

import numpy as np
import pandas as pd
import scipy as sp


# In[13]:

events_df = pd.read_csv('gdelt/gdelt.2011-2013.tsv.gz', index_col=False, delimiter='\t', header=0,dtype={'Date': np.str, 'Source': np.str, 'Target': np.str, 'CAMEOCode':np.str,'NumEvents':np.int32,'NumArts':np.int32, 'QuadClass':np.int8,'Goldstein':np.float32,'SourceGeoType':np.str,'SourceGeoLat':np.float32,'SourceGeoLong':np.float32,"TargetGeoType":np.str,'TargetGeoLat':np.float32,'TargetGeoLong':np.float32,'ActionGeoType':np.str,'ActionGeoLat':np.float32,'ActionGeoLong':np.float32})


# In[5]:

brent_oil = pd.read_csv('gdelt/brent-oil-futures-2011-2013.tsv',index_col=False,header=0,delimiter='\t')


# In[4]:

print events_df.columns
print events_df.shape


# In[4]:

print events_df.Source.describe()
print events_df.Target.describe()


# In[5]:

print events_df.CAMEOCode.describe()
print events_df.NumEvents.describe()
print events_df.NumArts.describe()
print events_df.QuadClass.describe()


# In[9]:

print events_df.SourceGeoType.describe()
print events_df.TargetGeoType.describe()
print events_df.ActionGeoType.describe()


# In[6]:

import matplotlib.pyplot as plt

def BarPlotWrapper(values, tick_labels):
    ind = np.arange(values.size)
    width = 1.0
    plt.bar(ind,values,width=width)
    plt.xticks(ind+width/2,tick_labels, rotation='vertical')
    plt.show()

source_freq = events_df.Source.value_counts().head(50)
BarPlotWrapper(source_freq, source_freq.index.values)


# In[39]:

import matplotlib.pyplot as plt

def BarPlotWrapper(values, tick_labels):
    ind = np.arange(values.size)
    width = 1.0
    plt.bar(ind,values,width=width)
    plt.xticks(ind+width/2,tick_labels, rotation='vertical')
    plt.show()

source_freq = events_df.CAMEOCode.value_counts().head(50)
BarPlotWrapper(source_freq, source_freq.index.values)


# In[36]:

events_df['Date_dt']=pd.to_datetime(events_df.Date.astype(np.str),format='%Y%m%d')


# In[7]:

brent_oil['Date_dt']=pd.to_datetime(brent_oil.Date.astype(np.str),format='%b %d, %Y')


# In[8]:

brent_oil['isIncrease']=1-brent_oil.Change.str.startswith('-').astype(np.int8)


# In[14]:

from scipy import sparse

def sparse_df_to_sparse_matrix (sparse_df):
    index_list = sparse_df.index.values.tolist()
    matrix_columns = []
    sparse_matrix = None

    for column in sparse_df.columns:
        sps_series = sparse_df[column]
        sps_series.index = pd.MultiIndex.from_product([index_list, [column]])
        curr_sps_column, rows, cols = sps_series.to_coo()
        if sparse_matrix != None:
            sparse_matrix = sparse.hstack([sparse_matrix, curr_sps_column])
        else:
            sparse_matrix = curr_sps_column
        matrix_columns.extend(cols)

    return sparse_matrix, index_list, matrix_columns

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

cameo_code_dummy = pd.get_dummies(events_df['CAMEOCode'])
#sm, il, mc = sparse_df_to_sparse_matrix(cameo_code_dummy)
#cameo_code_dummy.to_csv('cameo_code_dummy.csv.gz',compression='gzip')


# In[ ]:




# In[ ]:

sm_cameo_dummy = csr_matrix((sm,il,mc),shape=cameo_code_dummy.shape)
save_sparse_csr('sm_cameo_dummy.mat',sm_cameo_dummy)


# In[15]:

import time
def GetDays(x):
    return x.days
def GetBrentChangeLookup(date_str):
    date = pd.to_datetime(date_str,format='%Y%m%d')
    pos_idx = (brent_oil['Date_dt']-date).apply(GetDays)>0
    return brent_oil['isIncrease'][brent_oil['Date_dt'][pos_idx].argmin()]
def BrentChangeLookup(date):
    return look_up_table[date]

look_up_table = dict(zip(events_df['Date'].unique(),pd.Series(events_df['Date'].unique()).apply(GetBrentChangeLookup)))
BrentIncrease = events_df['Date'].apply(BrentChangeLookup)
#BrentIncrease.to_csv('brent_increase.csv')


# In[16]:

events_df = None
from scipy.stats import chi2_contingency
import time
def GetChi2ContingencyP(x,y):
    return chi2_contingency([[np.sum((1-x)*(1-y)),np.sum((1-x)*y)],[np.sum(x*(1-y)),np.sum(x*y)]])[1]
p_vals = []
for column in cameo_code_dummy:
    p_vals.append(GetChi2ContingencyP(cameo_code_dummy[column],BrentIncrease))


# In[17]:
# event_look_up table too big to code here.


# In[18]:

import matplotlib.pyplot as plt
import statsmodels.stats
def LookUpEventCode(x):
    if x in event_lookup_table:
        return event_lookup_table[x]
    else:
        return 'None'
p_val_df = pd.DataFrame({'pval': p_vals, 'labels': pd.Series(cameo_code_dummy.columns.values).apply(LookUpEventCode)})
p_val_sorted = p_val_df.sort_values(by='pval').head(20)

BarPlotWrapper(-np.log10(p_val_sorted['pval']),p_val_sorted['labels'])


# In[59]:

from sklearn.ensemble import RandomForestClassifier
rfcl = RandomForestClassifier(n_jobs=8,warm_start=True)
rfcl.fit(X=cameo_code_dummy.values,y=BrentIncrease)

