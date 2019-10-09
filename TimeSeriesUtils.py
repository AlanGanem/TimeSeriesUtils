# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:05:47 2019

@author: PC10
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
#import skimage.measure
#from func_get_product_history import get_product_history
from func_gini import gini
from func_price_clustering import price_clustering
import wquantiles as wq
from scipy import stats
import sklearn
import pandas as pd
import tqdm
import datetime
from scipy.stats import variation
import calendar
import datetime

def chunk_data_by_date_df(df,pred_period,look_back_period,input_columns, output_columns ,feature_axis =-1,n_validation_intervals = 1,flatten = False,static = False):
    '''
    Groups data in date period chuncks predefined for X and y and splits both
    in train and validation sets.
    return format:
        X_train, y_train, X_val, y_val
    OBS.:make sure the temporal axis is the first one
    '''
    
    X = df.values
    if not all([i in list(df.columns) for i in output_columns]):
        columns_not_in_frame = [i for i in output_columns if i not in list(df.columns)]
        raise Exception("{}".format(columns_not_in_frame) + ' not in data frame')
    
    if not all([i in list(df.columns) for i in input_columns]):
        columns_not_in_frame = [i for i in input_columns if i not in list(df.columns)]
        raise Exception("{}".format(columns_not_in_frame) + ' not in data frame')
    
    
    output_index = [list(df.columns).index(i) for i in output_columns if i in list(df.columns)]
    input_index = [list(df.columns).index(i) for i in input_columns if i in list(df.columns)]

    X_shape = X.shape
    X_n_dim = len(X_shape)
    assert np.abs(feature_axis) <= X_n_dim
    
    X_train_past = []
    y_train_past = []
    for i in range(look_back_period, len(X)):
        X_train_past.append(X[i-look_back_period:i].take(input_index, axis = feature_axis))
    
    for i in range(look_back_period,len(X)-pred_period):
        y_train_past.append(X[i:i+pred_period].take(output_index, axis = feature_axis))
    
    X_new = np.array(X_train_past)[:-pred_period]
    #X_train = np.reshape(np.array(X_train_past),[np.array(X_train_past).shape[0],np.array(X_train_past).shape[1],1])
    y_new =  np.array(y_train_past)
    
    
    assert  X_new.shape[0] == y_new.shape[0]
    
    #y_train= min_max_scaler.fit_transform(y_train)
    #y_new = np.reshape(y_new,list(y_new.shape)+[1])
    
    X_val = X_new[-pred_period*n_validation_intervals:]
    y_val = y_new[-pred_period*n_validation_intervals:]
    X_train = X_new[:-pred_period*n_validation_intervals]
    y_train = y_new[:-pred_period*n_validation_intervals]
    
    if flatten:
        y_train,y_val =  y_train.reshape(y_train.shape[:-1]), y_val.reshape(y_val.shape[:-1])
    
    if static:
        X_train,X_val =  X_train[:,-1,:,:], X_val[:,-1,:,:]
    
    assert y_train.shape[0] ==X_train.shape[0] 
    assert y_val.shape[0] ==X_val.shape[0] 
    print(('{} = {} \n {} = {} \n {} = {} \n {} = {} \n').format('X_train.shape', X_train.shape,'y_train.shape', y_train.shape,'X_val.shape', X_val.shape,'y_val.shape', y_val.shape,))
    print('total amount of samples = {} \n learning window = {} \n prediction horizon = {}'.format((X_train.shape[0] + X_val.shape[0]),X_train.shape[1],y_train.shape[1]))    
    return X_train, y_train, X_val, y_val

def chunk_data_by_date(X,pred_period,look_back_period, output_index = -1,output_axis =-1,data_frame_input = False,flatten = False,static = False):
    '''
    Groups data in date period chuncks predefined for X and y and splits both
    in train and validation sets.
    return format:
        X_train, y_train, X_val, y_val
    OBS.:make sure the temporal axis is the first one
    '''
    X_shape = X.shape
    X_n_dim = len(X_shape)
    assert np.abs(output_axis) <= X_n_dim
    
    X_train_past = []
    y_train_past = []
    for i in range(look_back_period, len(X)):
        X_train_past.append(X[i-look_back_period:i])
    
    for i in range(look_back_period,len(X)-pred_period):
        y_train_past.append(X[i:i+pred_period].take(output_index, axis = output_axis))
    
    X_new = np.array(X_train_past)[:-pred_period]
    #X_train = np.reshape(np.array(X_train_past),[np.array(X_train_past).shape[0],np.array(X_train_past).shape[1],1])
    y_new =  np.array(y_train_past)
    
    
    assert  X_new.shape[0] == y_new.shape[0]
    
    #y_train= min_max_scaler.fit_transform(y_train)
    y_new = np.reshape(y_new,list(y_new.shape)+[1])
    
    X_val = X_new[-pred_period:]
    y_val = y_new[-pred_period:]
    X_train = X_new[:-pred_period]
    y_train = y_new[:-pred_period]
    
    if flatten:
        y_train,y_val =  y_train.reshape(y_train.shape[:-1]), y_val.reshape(y_val.shape[:-1])
    
    if static:
        X_train,X_val =  X_train[:,-1,:,:], X_val[:,-1,:,:]
    
    assert y_train.shape[0] ==X_train.shape[0] 
    assert y_val.shape[0] ==X_val.shape[0] 
    print(('{} = {} \n {} = {} \n {} = {} \n {} = {} \n').format('X_train.shape', X_train.shape,'y_train.shape', y_train.shape,'X_val.shape', X_val.shape,'y_val.shape', y_val.shape,))
    print('total amount of samples = {} \n learning window = {} \n prediction horizon = {}'.format((X_train.shape[0] + X_val.shape[0]),X_train.shape[1],y_train.shape[1]))    
    return X_train, y_train, X_val, y_val


def one_hot_append(X,one_hot_indexes):
    assert isinstance(one_hot_indexes,list)
    
    for i in one_hot_indexes:
        
        onehot = OneHotEncoder(categories = 'auto',sparse = False)
        X[:,i:i+1][np.isnan(X[:,i:i+1])] = np.nanmedian(X[:,i:i+1])
        onehotencoded = onehot.fit_transform(X[:,i:i+1])
        try:
            onehot_concat = np.concatenate((onehotencoded, onehot_concat), axis = 1)
        except NameError:
            onehot_concat = onehotencoded
        except TypeError:
            print(i)
            onehot_concat = onehotencoded
    try:
        X = np.concatenate((onehot_concat,X[:,[i for i in range(X.shape[1])  if i not in one_hot_indexes]]),axis =1 )
    except NameError:
        pass
    
    return X

def enc_dec_predict(x, encoder_predict_model, decoder_predict_model, num_steps_to_predict,latent_dim):
    """Predict time series with encoder-decoder.
    
    Uses the encoder and decoder models previously trained to predict the next
    num_steps_to_predict values of the time series.
    
    Arguments
    ---------
    x: input time series of shape (batch_size, input_sequence_length, input_dimension).
    encoder_predict_model: The Keras encoder model.
    decoder_predict_model: The Keras decoder model.
    num_steps_to_predict: The number of steps in the future to predict
    
    Returns
    -------
    y_predicted: output time series for shape (batch_size, target_sequence_length,
        ouput_dimension)
    """
    
    y_predicted = []

    # Encode the values as a state vector
    states = encoder_predict_model.predict(x)
    
    # The states must be a list
    if not isinstance(states, list):
        states = [states]

    # Generate first value of the decoder input sequence
    decoder_input = np.zeros((x.shape[0],8,1))


    for _ in range(num_steps_to_predict):
        outputs_and_states = decoder_predict_model.predict(
        [decoder_input] + states)
        output = outputs_and_states[0]
        states = outputs_and_states[1:]
        # add predicted value
        y_predicted.append(output)

    return np.concatenate(y_predicted, axis=1)

def teacher_forcing_generator(y_train,y_val,temporal_axis_output = -2, flatten = False):
    '''
    prepares y data for teacher  forcing
    outputs of fucntion with  tag 'no_teacher_forcing returns a zero array with the expected shape'
    return format X_train_teacher_forcing,X_val_teacher_forcing,X_train_no_teacher_forcing,X_val_no_teacher_forcing
    '''
    np.zeros((y_train.shape[0],1))
    X_train_teacher_forcing = np.concatenate([np.zeros(np.take(y_train,range(1),axis = temporal_axis_output).shape),np.take(y_train,range(0,y_train.shape[temporal_axis_output]-1),axis = temporal_axis_output)], axis = 1)
    X_val_teacher_forcing = np.concatenate([np.zeros(np.take(y_val,range(1),axis = temporal_axis_output).shape),np.take(y_val,range(0,y_val.shape[temporal_axis_output]-1),axis = temporal_axis_output)], axis = 1)
    X_train_no_teacher_forcing = np.zeros(y_train.shape)
    X_val_no_teacher_forcing = np.zeros(y_val.shape)
    arrays = X_train_teacher_forcing,X_val_teacher_forcing,X_train_no_teacher_forcing,X_val_no_teacher_forcing
    
    if flatten:
        return (np.take(array, -1, axis = -1) for array in arrays)
    else:
        return X_train_teacher_forcing,X_val_teacher_forcing,X_train_no_teacher_forcing,X_val_no_teacher_forcing
def average_anti_diag(x):
    """Average antidiagonal elements of a 2d array
    Parameters:
    -----------
    x : np.array
        2d numpy array of size

    Return:
    -------
    x1d : np.array
        1d numpy array representing averaged antediangonal elements of x

    """
    x_inv = x[:, ::-1]
    x1d = [np.mean(x_inv.diagonal(i)) for i in
           range(-x.shape[0] , x.shape[1])]
    return np.array(x1d)[::-1]

def get_anti_diag(x):
    x_inv = x[:, ::-1]
    x1d = [list(x_inv.diagonal(i)) for i in range(-x.shape[0] + 1, x.shape[1])]
    range(len(x1d))
    return x1d[::-1]

def search_for_product(title):
    path,dic_name = r'C:\ProductClustering\productsDB\products_db_objects\\','products_db_dict'
    g = os.path.join(os.path.dirname(path), dic_name)
    prod_db = pd.DataFrame(pickle.load(open(g, 'rb')))
    a = products_db_finder()
    a.init_products_db(prod_db)
    return a.get_similar_products(title = title)

def naive_pred(X_train,y_train,days_before):
    '''
    cauclates de error when modeling with moving average    
    '''
    
    naive_preds = []
    for i in range(X_train.shape[0]):
        naive_preds.append(abs(np.average(X_train[i,-days_before+look_back_period:,-1:])-y_train[i]).mean())
    naive_errors = pd.DataFrame(naive_preds)
    print(naive_errors.describe())
    return naive_errors


def timestamp_to_datetime(date, to_str = False):
    if to_str:
        return str(date)[:10]
    return datetime.datetime.strptime(str(date)[:10],"%Y-%m-%d").date()


def chunk_to_pooled_2d(X,pred_period,look_back_period,y_seller_axis, output_index = -1,output_axis =-1, time_blocks = 1, functions = [np.mean,variation], pooled_output = True,flatten = True):
    '''
    returns  2D array with flatten and pooled features over a period of time
    (each feature in the array is a  pooled, time distributed feature)
    return format:
        
        X_train, y_train, X_val, y_val
    '''
    from operator import itemgetter
    for func in functions:
        X_train, y_train, X_val, y_val = chunk_data_by_date(X,pred_period,look_back_period, output_index = -1,output_axis =-1)
        
            
        set_ = X_train
        X_t_shape_before = list(itemgetter(*[0,2,3])(list(X_train.shape)))
        X_t_shape_before[-1] *= time_blocks*len(functions)
        print(X_t_shape_before)
        X_train= np.array([np.concatenate([skimage.measure.block_reduce(set_[i].reshape(set_.shape[1],set_.shape[2]*set_.shape[3]), (set_.shape[1]//time_blocks,1), func)],axis = 0).flatten() for i in range(set_.shape[0])])
        
        set_ = X_val
        X_v_shape_before = list(itemgetter(*[0,2,3])(list(X_val.shape)))
        X_v_shape_before[-1] *= time_blocks*len(functions)
        X_val = np.array([np.concatenate([skimage.measure.block_reduce(set_[i].reshape(set_.shape[1],set_.shape[2]*set_.shape[3]), (set_.shape[1]//time_blocks,1), func)],axis = 0).flatten() for i in range(set_.shape[0])])
        
        set_ = y_train
        y_t_shape_before = y_train.shape 
        set_t = set_[:,:,y_seller_axis:y_seller_axis+1]
        
        set_ = y_val
        y_v_shape_before = y_val.shape 
        set_v = set_[:,:,y_seller_axis:y_seller_axis+1]
        
        
        
        if pooled_output:
            y_val = np.array([skimage.measure.block_reduce(set_v[i].reshape(set_v.shape[1],set_v.shape[2]*set_v.shape[3]), (set_v.shape[1]//time_blocks,1), np.mean).flatten() for i in range(set_v.shape[0])])
            y_train = np.array([skimage.measure.block_reduce(set_t[i].reshape(set_t.shape[1],set_t.shape[2]*set_t.shape[3]), (set_t.shape[1]//time_blocks,1), np.mean).flatten() for i in range(set_t.shape[0])])
        else:
            y_val = np.array([set_v[i].reshape(set_v.shape[1],set_v.shape[2]*set_v.shape[3]).flatten() for i in range(set_v.shape[0])])
            y_train = np.array([set_t[i].reshape(set_t.shape[1],set_t.shape[2]*set_t.shape[3]).flatten() for i in range(set_t.shape[0])])
    
        if functions.index(func)==0:
            if len(functions) == 0:
                pass
            else:
                X_train_concat, y_train_concat, X_val_concat, y_val_concat = X_train, y_train, X_val, y_val
        else:
            X_train_concat, y_train_concat, X_val_concat, y_val_concat = np.concatenate([X_train_concat,X_train],axis = 1), np.concatenate([y_train_concat,y_train],axis = 1), np.concatenate([X_val_concat,X_val],axis = 1), np.concatenate([y_val_concat,y_val],axis = 1)
        
        
    X_train, y_train, X_val, y_val = X_train_concat, y_train_concat, X_val_concat, y_val_concat
    if not flatten:
            X_train,X_val = X_train.reshape(X_t_shape_before),X_val.reshape(X_v_shape_before)
    
    
    #static params
    
    
    print((' {} = {} original was {} \n {} = {} original was {}\n {} = {} original was {} \n {} = {} original was {}\n').format('X_train.shape', X_train.shape,X_t_shape_before,'y_train.shape', y_train.shape,y_t_shape_before,'X_val.shape', X_val.shape,X_v_shape_before,'y_val.shape', y_val.shape,y_v_shape_before))
    print('input data shape = {} \n samples dropped = {} \n total amount of samples = {} \n learning window = {} \n prediction horizon = {}'.format(X.shape,X.shape[0]-X_train.shape[0] - X_val.shape[0],(X_train.shape[0] + X_val.shape[0]),X_t_shape_before[1],y_train.shape[1]))
    return X_train, y_train, X_val, y_val
    
def get_and_prepare_product_data(product_id  , min_price ,max_price , features, min_sold, dependent_variable,one_hot_features =[],title_ilike=None,title_not_ilike = None,drop_blackout = False):
    assert features[-1:] == dependent_variable
    assert all([ohf in features for ohf in one_hot_features]) 
    
    history = get_product_history(product_id = product_id, min_price =min_price, max_price = max_price ,drop_blackout = drop_blackout, title_ilike=title_ilike, title_not_ilike =title_not_ilike )
    history['date'] = pd.to_datetime(history['date'], errors  = 'coerce',format = '%Y-%m-%d')
    #history = history[(history.daily_sales > min(history.daily_sales))&(history.daily_sales < max(history.daily_sales))]
    
    sellers = list(history.groupby('seller_id').sum()[history.groupby('seller_id').sum().daily_sales > min_sold].daily_sales.index)
    history = history[history['seller_id'].isin(sellers)]
    

    history = history.fillna(method = 'backfill')
    history = history.fillna(0)[(stats.zscore(history.fillna(0)['daily_sales']) < 10)]
    history = history[history.daily_sales >=0]
    history.daily_sales.max()
    
    view = price_clustering(history[history.daily_sales > 0 ],column_name = 'price', fluctuation = 0.2)
    
    
    history = history.dropna()
    history_filtered = history[history.seller_id.isin(sellers)]
    history_filtered = history_filtered.dropna(subset = ['date'])
    sellers_dates = {seller:{'initial_date':timestamp_to_datetime(history_filtered[history_filtered['seller_id'] == seller]['date'].nsmallest(2).max()),'final_date':timestamp_to_datetime(history_filtered[history_filtered['seller_id'] == seller]['date'].nlargest(2).min())} for seller in  sellers}
    history_filtered =history_filtered.assign(active_seller = 0)
    
    for seller in sellers:
        initial_date = sellers_dates[seller]['initial_date']
        final_date= sellers_dates[seller]['final_date']    
        history_filtered[(history_filtered['seller_id'] == seller)&(history_filtered['date'].isin(pd.date_range(initial_date,final_date)))] = history_filtered[(history_filtered['seller_id'] == seller)&(history_filtered['date'].isin(pd.date_range(initial_date,final_date)))].assign(active_seller = 1, inplace = True)
    
    assert isinstance(one_hot_features,list)
    history_filtered = pd.get_dummies(history_filtered,columns = one_hot_features)
    one_hot_feature_list = []
    for feature in one_hot_features:
         one_hot_feature_list+=[column for column in history_filtered.columns  if feature  in column]
    
    market_sizes = [history_filtered.groupby('date').get_group(i).daily_revenues.sum() for i in history_filtered.date.unique()]
    
    dflist=[]
    i = 0
    for date in history['date'].unique():
        dflist.append(history_filtered[history_filtered.date == date].assign(market_size = market_sizes[i].max()))
        i+=1
    
    history_filtered = pd.concat(dflist)
    
    
    
    for i in one_hot_features:
        features.pop(features.index(i))
    
    
    cnn_X_shape = (len(sellers),len(one_hot_feature_list + features)-1)
    cnn_y_shape = (len(sellers),1)
    
    
    gabarito = pd.DataFrame(np.zeros((len(sellers),len(one_hot_feature_list + features))), columns = one_hot_feature_list + features,index = sellers)
    
    date_interval = pd.date_range(min(sorted(history_filtered.date.unique())),max(sorted(history_filtered.date.unique())))
    
    groupped = history_filtered.groupby('date')    
    test = groupped.get_group(timestamp_to_datetime(date_interval[0])).groupby('seller_id').apply(fu)[one_hot_feature_list + features]
    assert all(gabarito.columns == test.columns)
    dates = {}
    for date in tqdm.tqdm(date_interval):
        date = timestamp_to_datetime(date)
        try:
            data = groupped.get_group(date).reset_index(drop=False).groupby('seller_id').apply(fu)[one_hot_feature_list + features]
            if len(data) > 1:
                data['position_max'] = (data['position_max']-data['position_max'].min())/(data['position_max'].max()-data['position_max'].min())
                data['position_median'] = (data['position_median']-data['position_median'].min())/(data['position_median'].max()-data['position_median'].min())
            else:
                data['position_max'] = 1
                data['position_median'] = 1
            df = gabarito.copy()
            df.loc[data.index] = data
        except KeyError as error:        
            df = gabarito.copy()
        dates[timestamp_to_datetime(date)] = df    
    
    view = pd.DataFrame()
    for date in tqdm.tqdm(date_interval):
        date = timestamp_to_datetime(date)
        try:
            data = groupped.get_group(date).groupby('seller_id').apply(fu).reset_index()
            data = data.set_index(['date','seller_id'])
            if view.empty:
                view = data
            else:
                view = pd.concat([view,data])
        except:
            pass
    
    
    min_date = min(sorted(list(dates.keys())))
    max_date = max(sorted(list(dates.keys())))
    
    lista_X = []
    for key in sorted(list(dates.keys())):
        try:
            lista_X.append(dates[key][one_hot_feature_list + features].values)
        except:
            print(date)
            lista_X.append(np.zeros(cnn_X_shape))    
    X = np.array(lista_X)
    
    return X, sellers, one_hot_feature_list + features, view, list(dates.keys())


def moving_average_model(X_train):
    
    return skimage.measure.block_reduce(X_train,(1,X_train.shape[1]),func = np.mean)
    
#agregation function for metrics calculation
def fu(x,one_hot_feature_list =[]):
    d={}
    d['market_median_price'] = x['market_median_price'].max()
    d['market_size'] = x['market_size'].max()
    try:
        d['market_size_units'] = x['market_size_units'].mean()
    
    except:
        pass
    #d['date'] = x['date'].max()
    d['amount_of_ads'] = x['ad_id'].count()
    
    d['active_seller']  = x['active_seller'].max()
    d['category_id'] = sklearn.utils.extmath.weighted_mode(x['category_id'],np.nan_to_num(x['daily_sales']))[0].max()
    try:
        d['ad_type_mean'] = np.average(x['ad_type_id'],weights = np.nan_to_num(x['daily_sales']))
    except:
        d['ad_type_mean'] = x['ad_type_id'].mean()
    
    
    d['position_max'] = 1/np.log1p(np.min(x['position']))
    d['price_min'] = np.min(x['price'])
    
    d['daily_sales_sum'] = x['daily_sales'].sum()

    d['daily_views_sum'] = x['daily_views'].sum()
    d['daily_views_share'] = d['daily_views_sum']/x['market_daily_views'].max()
    if x['daily_sales'].max()<= 0:
        d['price_median'] = np.median(x['price'])
    else:
        d['price_median'] = wq.quantile(x['price'],x['daily_sales'],0.25)
    if np.isnan(d['price_median']):
        d['price_median'] = np.median(x['price'])
    
    try:
        d['relative_price'] = d['price_median']/x['market_median_price'].max()
    except:
        print('not ok')
        pass
    
    if x['daily_sales'].max() <= 0:
        d['position_median'] = 1/np.log1p(np.median(x['position']))
    else:
        d['position_median'] = 1/np.log1p(wq.quantile(x['position'],x['daily_sales'],0.25))
    
    if np.isnan(d['position_median']):
        d['position_median'] = np.log1p(1/np.median(d['position_median']))
    
    d['sold_quantity_sum'] = x['sold_quantity'].sum()
    d['gini_ads'] = gini(x['daily_revenues'].values)
    if x['daily_views'].sum() > 0:
        d['conversion'] = x['daily_sales'].sum()/x['daily_views'].sum()
    else:
        d['conversion'] = 0
    
    d['share'] = x['daily_revenues'].sum()/x['market_size'].max()
    d['daily_revenues_sum'] = x['daily_revenues'].sum()
    if np.isnan(d['share']):
        d['share'] = 0
    
    
    for feature in one_hot_feature_list:
        d[feature]  = x[feature].mean()
            
    return (pd.Series(d))

def prepare_dummies(X):
    
    X_t_list = []
    X_v_list = []
    y_t_list = []
    y_v_list = []
    
    pooled_output = True
    for seller in sellers:
        y_seller_axis = sellers.index(seller)    
        X_train, y_train, X_val, y_val = chunk_to_pooled_2d(X,pred_period=pred_period,look_back_period=look_back_period,y_seller_axis = y_seller_axis , output_index = -1,output_axis =-1, time_blocks = time_blocks, functions = [np.mean],pooled_output = pooled_output )
        
        t_dummies = np.zeros((X_train.shape[0],len(sellers)))
        t_dummies[:,y_seller_axis] =  1
        v_dummies = np.zeros((X_val.shape[0],len(sellers)))
        v_dummies[:,y_seller_axis] =  1
        
        print(v_dummies[0])    
        X_train, X_val = np.concatenate([t_dummies,X_train],axis = 1), np.concatenate([v_dummies,X_val],axis= 1)
        if pooled_output:
            y_train,y_val = np.average(y_train,axis = 1),np.average(y_val,axis = 1)
        else:
            pass
    
        X_t_list.append(X_train)
        y_t_list.append(y_train)
        X_v_list.append(X_val)
        y_v_list.append(y_val)
    
    X_train = np.array(X_t_list)
    X_train= X_train.reshape((X_train.shape[0]*X_train.shape[1],)+X_train.shape[2:])
    
    y_train= np.array(y_t_list)
    y_train = y_train.reshape((y_train.shape[0]*y_train.shape[1],)+y_train.shape[2:])
    
    X_val_15 = np.array(X_v_list)[:,:foward_pred_goal,:]
    X_val = np.array(X_v_list)
    X_val_15 = X_val_15.reshape((X_val_15.shape[0]*X_val_15.shape[1],)+X_val_15.shape[2:])
    X_val = X_val.reshape((X_val.shape[0]*X_val.shape[1],)+X_val.shape[2:])
    
    y_val_15 = np.array(y_v_list)[:,:foward_pred_goal]
    y_val = np.array(y_v_list)
    y_val_15 = y_val_15.reshape((y_val_15.shape[0]*y_val_15.shape[1],)+y_val_15.shape[2:])
    y_val = y_val.reshape((y_val.shape[0]*y_val.shape[1],)+y_val.shape[2:])
    

def df_to_array(df):
    axis_lens = [len(df.index.get_level_values(name).unique()) for name in list(df.index.names)]
    print(axis_lens)
    m,n = len(df.index.levels[0]), len(df.index.levels[1])
    arr = df.values.reshape(*axis_lens,-1)
    return arr

def chunk_and_concatenate_dict(train_data_dict, pred_period, look_back_period, input_columns, output_columns,scaler = False,**kwargs):

    scalers = {}
    i=0
    for key in train_data_dict.keys():
        if isinstance(scaler,str):
            if scaler.lower() == 'min_max':
                scaler = sklearn.preprocessing.MinMaxScaler()
                train_data_dict[key][train_data_dict[key].columns] = scaler.fit_transform(train_data_dict[key][train_data_dict[key].columns])
                scalers[key] = scaler
                df = train_data_dict[key][train_data_dict[key].columns]
                print('OK')
            elif scaler.lower() == 'standard':
                scaler = sklearn.preprocessing.StandardScaler()
                train_data_dict[key][train_data_dict[key].columns] = scaler.fit_transform(train_data_dict[key][train_data_dict[key].columns])
                scalers[key] = scaler
                df = train_data_dict[key][train_data_dict[key].columns]
                print('OK')
        elif isinstance(scaler,(sklearn.preprocessing.data.MinMaxScaler or sklearn.preprocessing.data.StandardScaler)):
            train_data_dict[key][train_data_dict[key].columns] = scaler.transform(train_data_dict[key][train_data_dict[key].columns])
            scalers[key] = scaler
            df = train_data_dict[key][train_data_dict[key].columns]
            print('OK')
        else:
            df = train_data_dict[key]
            print('OK')
        
        if i == 0:
            #data_arr = minmax_scaler.fit_transform(data_arr)
            X = chunk_data_by_date_df(df,pred_period,look_back_period,input_columns = input_columns, output_columns = output_columns,**kwargs)
            X_train, y_train, X_val, y_val = X
            
        else:
            X_ = chunk_data_by_date_df(df,pred_period,look_back_period,input_columns = input_columns, output_columns = output_columns,**kwargs)
            X_train_, y_train_, X_val_, y_val_ = X_
            
            X = np.concatenate((X,X_))
            
            X_train = np.concatenate((X_train,X_train_))
            y_train = np.concatenate((y_train,y_train_))
            X_val = np.concatenate((X_val,X_val_ ))
            y_val = np.concatenate((y_val,y_val_))     
        i+=1
        
    return X_train, y_train, X_val, y_val


def scale_df(df,scaler):
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df[df.columns], scaler

def hash_mapper(df, max_val = 100000):
    merged_data = df
    inv_hashmap = {hash(feature)%max_val: feature for feature in set(merged_data[merged_data.columns[list(merged_data.dtypes == 'object')]].values.flatten())}
    hashmap = {feature:feature_hash for feature_hash,feature in inv_hashmap.items()}
    return hashmap,inv_hashmap

def n_of_week_days(df, prefix = 'n_'):
    index_before = list(df.index)
    resampled = pd.DataFrame(df.iloc[:,0].resample('1D').max())
    resampled['dayofweek'] = resampled.index.day_name()
    dayofweek_dummies = pd.get_dummies(resampled[['dayofweek']],columns = ['dayofweek'], prefix = prefix, prefix_sep = '')
    dayofweek_dummies['timeindex'] = dayofweek_dummies.index
    dayofweek_dummies = dayofweek_dummies.drop_duplicates()
    dayofweek_dummies = dayofweek_dummies.drop(columns = ['timeindex'])
    dayofweek_dummies = dayofweek_dummies.resample('M').sum()
    error = dayofweek_dummies[dayofweek_dummies.sum(axis = 1) < 28]
    error = pd.date_range(end = error.index[0],start = error.tshift(-1,freq = 'M').tshift(1,freq = 'D').index[0],freq= 'D')
    error = pd.DataFrame(index = error)
    error['dayofweek'] = error.index.day_name()
    error_dummies = pd.get_dummies(error[['dayofweek']],columns = ['dayofweek'], prefix = prefix, prefix_sep = '')
    error_dummies['timeindex'] = error_dummies.index
    error_dummies = error_dummies.drop_duplicates()
    error_dummies = error_dummies.drop(columns = ['timeindex'])
    error_dummies = error_dummies.resample('M').sum()
    dayofweek_dummies[dayofweek_dummies.sum(axis = 1) < 28] = error_dummies
    df = pd.merge(df,dayofweek_dummies, left_index = True,right_index = True)
    assert len(list(df.index)) == len(index_before)
    return df

def pred_df(y_true,y_pred, index ,fix_dim = -1, prefix = 'f_'):
    assert (len(y_true.shape),len(y_pred.shape)) == (2,2)
    error_dict = {}
    for forecast in range(y_pred.shape[fix_dim]):
        error_dict[forecast] = pd.DataFrame(y_pred.take(axis = fix_dim, indices = forecast).flatten(),index = index.take(axis = fix_dim, indices = forecast), columns = [forecast])
    #col_names = [str(prefix)+ str(forecast) for forecast in error_dict.keys()]
    i = 0
    for key,df in error_dict.items():
        df.columns = [str(prefix)+str(key)]
        if i ==0:
            error_df = df
            actual_index = df.index
        else:
            error_df = pd.concat([error_df,df], axis = 1)
        i+=1
    error_df = pd.concat([error_df, pd.DataFrame(y_true.take(axis = fix_dim, indices = 0), index = actual_index, columns = ['actual'])],axis = 1)
    return error_df

def data_dict_transformer(train_data_dict, key, pred_period, look_back_period, encoder_inputs, decoder_inputs, **kwargs):
    
    X_train, y_train, X_val, y_val = chunk_and_concatenate_dict({key:train_data_dict[key].assign(date = train_data_dict[key].index)},pred_period,look_back_period,encoder_inputs,['date']+decoder_inputs,**kwargs)
    X_covars_train = y_train.take(range(1,y_train.shape[-1]-1),axis = -1)
    X_covars_val = y_val.take(range(1,y_val.shape[-1]-1),axis = -1)
    period_train = y_train.take([0],axis = -1)
    period_val = y_val.take([0],axis = -1)
    y_train = y_train.take([-1],axis = -1)
    y_val = y_val.take([-1],axis = -1)
        
    return {'period_train':period_train,'period_val':period_val,'X_train':X_train, 'X_covars_train':X_covars_train, 'y_train':y_train, 'X_val':X_val, 'X_covars_val':X_covars_val, 'y_val':y_val}

def get_date_loc(df, days = range(40),months = range(40),years = range(9999999),daysofweek = range(8)):
    return df.loc[(df.index.day.isin(days))&(df.index.month.isin(months))&(df.index.year.isin(years))&(df.index.dayofweek.isin(daysofweek))]

def column_to_categorical(df ,columns,column_names =None, sep = '_'):
    concat_df = 0
    split_columns = [i.split(sep) for i in columns]
    levels_dict={index:[] for index,_ in enumerate(split_columns)}
    for col_name in split_columns:
        for index,value in enumerate(col_name):
            df[index] = 0
            if value not in levels_dict[index]:
                levels_dict[index].append(value)
                
    levels_dict = {key:value for key,value in levels_dict.items() if value != []}
    combinations = list(itertools.product(*list(levels_dict.values())))
    i=0
    for comb in combinations:
        combination = sep.join(comb)
        if combination in columns:    
            for level, value in enumerate(comb): 
                df[level] = value
            df['Value'] = df[combination]
            
            if i == 0:
                concat_df = df.drop(columns = columns).copy()
                i+=1
            else:
                concat_df = pd.concat([concat_df,df.drop(columns = columns)],axis = 0)
    
    if column_names:
        assert len(column_names) == len(levels_dict.keys())
        names_dict = {i:name for i,name in enumerate(column_names)}
        concat_df = concat_df.rename(columns = names_dict)
    
    index_name = concat_df.index.name
    
    return concat_df

def remove_outliers(df,std_threshold = 3,bilateral = True,columns = None):
    columns = df.columns if not columns else columns
    if bilateral:
        df[columns] = df[columns][abs((df[columns] - df[columns].median())/df[columns].std()) <= 3]
    else:
        df[columns] = df[columns][(df[columns] - df[columns].median())/df[columns].std() <= 3]
    return df

def df_to_dict(df, groupby,get_dummies = False):
    data_dict = {}
    groups = df.groupby(groupby)
    if not get_dummies:
        data_dict = {name:group.sort_index() for name,group in groups}
    
    elif get_dummies:
        cols_list = [[groupby[i]+'_'+str(name[i]) for i in range(len(name))] for name,group in groups]
        df = pd.get_dummies(df,columns = groupby)
        for cols in cols_list:
            data_dict[tuple(cols)] = df[(df[cols] == 1).prod(axis = 1).astype(bool)].sort_index()
    
    return data_dict
