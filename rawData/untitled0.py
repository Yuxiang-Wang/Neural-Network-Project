#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:07:19 2019

@author: Joey
"""

import numpy as np
import pandas as pd

#raw option data handling
pd.set_option('display.max_columns', 15)

A=pd.read_csv("AMZN_OPTION.csv")

temp=A.loc[A['exdate']==20140118]
temp0=temp.loc[temp['cp_flag']=='C']
opt_price=(temp0['best_offer']+temp0['best_bid'])/2
temp0['Option_price']=opt_price
temp1=temp0.drop(columns=['cp_flag','best_bid', 'best_offer','ticker', 'index_flag','issuer',
       'div_convention', 'exercise_style'])
temp2=temp1.dropna(axis='rows')
temp2['strike_price']=temp2['strike_price'].div(1000)

#k=[temp2.date.unique()]
#np.size(k)=344 match the working days from 2012/09/04 to 2014/01/16

#get the amzn stock price
amzn=pd.read_csv("AMZN_STOCK.csv")
amzn_price=amzn['Adj Close']


k=temp2.date.unique()
for i in range(344):
    temp2=temp2.drop(temp2[(temp2['date']==k[i]) & (abs(temp2['strike_price']-amzn_price[i])>10)].index)

#check dimention
#np.size(temp2.date.unique())=344
D=temp2.drop_duplicates('date',keep='last').reset_index()
data=D.drop(D[D['date']<20130117].index).reset_index()
data=data.drop(columns=['level_0','index','optionid'])

stock_price=amzn_price[np.size(amzn_price)-252:]

stock_price.to_csv("amzn_20130117_20140118.csv",index=False) #amzn_20130117_20140118
data['strike_price'].to_csv("amzn_K.csv",index=False)
data['impl_volatility'].to_csv("amzn_vol.csv",index=False)
data['Option_price'].to_csv("amzn_optionVal.csv",index=False)
