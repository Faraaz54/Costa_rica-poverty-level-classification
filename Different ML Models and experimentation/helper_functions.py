
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import auc,make_scorer,roc_auc_score,f1_score,roc_curve,classification_report
from scipy.spatial.distance import pdist, squareform,cdist
from scipy import exp
import seaborn as sns
from IPython.display import display
import lightgbm as gbm
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.model_selection import StratifiedKFold
from hyperopt.pyll.stochastic import sample


# In[19]:


#helper functions
def plot_feature_importances(df, n = 20, threshold = None):
    """Plots n most important features. Also plots the cumulative importance if
    threshold is specified and prints the number of features needed to reach threshold cumulative importance.
    Intended for use with any tree-based feature importances. 
    
    Args:
        df (dataframe): Dataframe of feature importances. Columns must be "feature" and "importance".
    
        n (int): Number of most important features to plot. Default is 15.
    
        threshold (float): Threshold for cumulative importance plot. If not provided, no plot is made. Default is None.
        
    Returns:
        df (dataframe): Dataframe ordered by feature importances with a normalized column (sums to 1) 
                        and a cumulative importance column
    
    Note:
    
        * Normalization in this case means sums to 1. 
        * Cumulative importance is calculated by summing features from most to least important
        * A threshold of 0.9 will show the most important features needed to reach 90% of cumulative importance
    
    """
    plt.style.use('fivethirtyeight')
    
    # Sort features with most important at the head
    df = df.sort_values('importance', ascending = False).reset_index(drop = True)
    
    # Normalize the feature importances to add up to one and calculate cumulative importance
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    
    # Bar plot of n most important features
    df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = 'darkgreen', 
                            edgecolor = 'k', figsize = (12, 8),
                            legend = False, linewidth = 2)

    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 
    plt.title(f'{n} Most Important Features', size = 18)
    plt.gca().invert_yaxis()
    
    
    if threshold:
        # Cumulative importance plot
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 
        plt.title('Cumulative Feature Importance', size = 18);
        
        # Number of features needed for threshold cumulative importance
        # This is the index (will need to add 1 for the actual number)
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        
        # Add vertical line to plot
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')
        plt.show();
        
        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 
                                                                                  100 * threshold))
    
    #return df


# In[20]:


def cv_model(train, train_labels, model, name, scorer, model_results=None):
    """Perform 10 fold cross validation of a model"""
    
    cv_scores = cross_val_score(model, train, train_labels, cv = 10, scoring=scorer, n_jobs = -1)
    print(f'10 Fold CV Score: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')
    
    if model_results is not None:
        model_results = model_results.append(pd.DataFrame({'model': name, 
                                                           'cv_mean': cv_scores.mean(), 
                                                            'cv_std': cv_scores.std()},
                                                           index = [0]),
                                             ignore_index = True).sort_values(by='cv_mean')

        return model_results


# In[21]:


def plot_model_results(model_results):
    model_results.set_index('model', inplace = True)
    model_results['cv_mean'].plot.bar(color = 'orange', figsize = (8, 6),
                                  yerr = list(model_results['cv_std']),
                                  edgecolor = 'k', linewidth = 2)
    plt.title('Model F1 Score Results');
    plt.ylabel('Mean F1 Score (with error bar)');
    model_results.reset_index(inplace = True)


# In[24]:


def dict_to_pd(model_name,results):
    if model_name == 'LR':
        dt = sorted([{'loss':trial['loss'],'C':trial['params']['C'],'penalty':trial['params']['penalty']['penalty'],'solver':trial['params']['penalty']['solver'],'score':trial['score'],'score_std':trial['score_std'],'iteration':trial['iteration']} for trial in lr_bayes_trials.results],key=lambda x:x['loss'])
        results = pd.DataFrame(dt)
    elif model_name == 'SVC':
        dt = sorted([{'loss':trial['loss'],'C':trial['params']['C'],'kernel':trial['params']['kernel'],'score':trial['score'],'score_std':trial['score_std'],'iteration':trial['iteration']} for trial in results],key=lambda x:x['loss'])
        results = pd.DataFrame(dt)
    elif model_name == 'GBM':
        dt = sorted([{'loss':trial['loss'],'score':trial['score'],'score_std':trial['score_std'],'n_estimators':trial['params']['n_estimators'],'bagging_fraction':trial['params']['bagging_fraction'],'boosting_type':trial['params']['boosting_type'],'colsample_bytree':trial['params']['colsample_bytree'],             'min_child_samples':trial['params']['min_child_samples'],'min_child_weight':trial['params']['min_child_weight'],'min_split_gain':trial['params']['min_split_gain'],'num_leaves':trial['params']['num_leaves'],'reg_alpha':trial['params']['reg_alpha'],'reg_lambda':trial['params']['reg_lambda'],             'subsample_for_bin':trial['params']['subsample_for_bin'],'subsample':trial['params']['subsample'],'train_std':trial['time_std'],'max_depth':trial['params']['max_depth']} for trial in gbm_results],key=lambda x:x['loss'])
        results = pd.DataFrame(dt)
    elif model_name == 'RF':
        dt = sorted([{'loss':trial['loss'],'max_depth':trial['params']['max_depth'],'min_samples_split':trial['params']['min_samples_split'],'min_samples_leaf':trial['params']['min_samples_leaf'],'max_features':trial['params']['max_features'],'score':trial['score'],'score_std':trial['score_std'],'iteration':trial['iteration']} for trial in results],key=lambda x:x['loss'])
        results = pd.DataFrame(dt)
    return results


# In[25]:


def build_gauss_kernel(X1,X2,gamma):
    pairwise_dists = cdist(X1, X2, 'sqeuclidean')
    K = exp(-pairwise_dists ** 2 / gamma ** 2)
    return K


# In[26]:


def dict_to_pd(model_name,results):
    if model_name == 'LR':
        dt = sorted([{'loss':trial['loss'],'C':trial['params']['C'],'penalty':trial['params']['penalty'],'solver':trial['params']['penalty']['solver'],'score':trial['score'],'score_std':trial['score_std'],'iteration':trial['iteration']} for trial in results],key=lambda x:x['loss'])
        results = pd.DataFrame(dt)
    elif model_name == 'SVC':
        dt = sorted([{'loss':trial['loss'],'C':trial['params']['C'],'kernel':trial['params']['kernel'],'score':trial['score'],'score_std':trial['score_std'],'iteration':trial['iteration']} for trial in results],key=lambda x:x['loss'])
        results = pd.DataFrame(dt)
    elif model_name == 'GBM':
        dt = sorted([{'loss':trial['loss'],'score':trial['score'],'score_std':trial['score_std'],'n_estimators':trial['params']['n_estimators'],'bagging_fraction':trial['params']['bagging_fraction'],'boosting_type':trial['params']['boosting_type'],'colsample_bytree':trial['params']['colsample_bytree'],             'min_child_samples':trial['params']['min_child_samples'],'min_child_weight':trial['params']['min_child_weight'],'min_split_gain':trial['params']['min_split_gain'],'num_leaves':trial['params']['num_leaves'],'reg_alpha':trial['params']['reg_alpha'],'reg_lambda':trial['params']['reg_lambda'],             'subsample_for_bin':trial['params']['subsample_for_bin'],'subsample':trial['params']['subsample'],'train_std':trial['time_std'],'max_depth':trial['params']['max_depth']} for trial in results],key=lambda x:x['loss'])
        results = pd.DataFrame(dt)
    elif model_name == 'RF':
        dt = sorted([{'loss':trial['loss'],'max_depth':trial['params']['max_depth'],'min_samples_split':trial['params']['min_samples_split'],'min_samples_leaf':trial['params']['min_samples_leaf'],'max_features':trial['params']['max_features'],'score':trial['score'],'score_std':trial['score_std'],'iteration':trial['iteration']} for trial in results],key=lambda x:x['loss'])
        results = pd.DataFrame(dt)
    
    return results


# In[27]:


def compare_distributions(results,param):
    uniform_C = list()
    for i in range(0,200):
        x = sample(space)
        uniform_C.append(x[param])

    sns.kdeplot(uniform_C, label = 'prior')
    sns.kdeplot(results[param], label = 'Bayes Optimization')
    plt.legend(loc = 'best')
    plt.title('{} Distribution'.format(param))
    plt.xlabel('{}'.format(param)); plt.ylabel('Density');
    plt.show();


# In[28]:


def avg_f1(multi_class_scores):
    result = list()
    for i in range(1,5):
        f1_class = np.ravel(multi_class_scores[i])
        result.append(np.mean(np.ravel(f1_class)))
    avg_f1 = np.mean(result)
    std_f1 = np.std(result)
    return avg_f1,std_f1


# In[29]:


def compare_distributions(results,param):
    uniform_C = list()
    for i in range(0,200):
        x = sample(space)
        uniform_C.append(x[param])

    sns.kdeplot(uniform_C, label = 'prior')
    sns.kdeplot(results[param], label = 'Bayes Optimization')
    plt.legend(loc = 'best')
    plt.title('{} Distribution'.format(param))
    plt.xlabel('{}'.format(param)); plt.ylabel('Density');
    plt.show();


# In[30]:


#scorer for GBM objective
def macro_f1_score(labels,predictions):
    predictions = predictions.reshape(len(np.unique(labels)), -1 ).argmax(axis = 0)
    
    metric_value = f1_score(labels, predictions, average = 'macro')
    
    # Return is name, value, is_higher_better
    return 'macro_f1', metric_value, True

