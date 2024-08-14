import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_auc_score as AUC 
from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN,BorderlineSMOTE
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler,EditedNearestNeighbours
from imblearn.combine import SMOTEENN,SMOTETomek
from sklearn.metrics import roc_curve,f1_score,recall_score,precision_score
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
import sklearn_relief as relief
import pickle

MaxDepth=4

def shuffle_data(X,y):
    X=np.array(X)
    for i in range(10):
        Xshuffle,yshuffle=shuffle(X,y,random_state=i)
        X,y=Xshuffle,yshuffle
    return X,y   

def data_split(X,y,random):
    sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=random)
    for train_index,test_index in sss.split(X,y):
        Xtrain,Xtest=X[train_index,:],X[test_index,:]
        ytrain,ytest=y[train_index],y[test_index]
    return Xtrain,Xtest,ytrain,ytest


def var_fs(Xtrain,Xtest,X_col_name,threshold=0.0):
    scaler=MinMaxScaler()
    scaler=scaler.fit(Xtrain)
    Xtr_scale=scaler.transform(Xtrain)
    Xte_scale=scaler.transform(Xtest)

    idx=[]  
    for i in range(Xtr_scale.shape[1]):                               
        var=np.var(Xtr_scale[:,i])                      
        if var > threshold:  
            idx.append(i)
    Xtr_var=Xtr_scale[:,idx]  
    Xte_var=Xte_scale[:,idx]
    newcol_var=list(np.array(X_col_name)[idx])
    return Xtr_var,Xte_var,newcol_var

def standard_data(Xtrain,Xtest):
    scaler=StandardScaler()
    scaler=scaler.fit(Xtrain)
    Xtr_std=scaler.transform(Xtrain)
    Xte_std=scaler.transform(Xtest)
    Xtr_std=np.around(Xtr_std,10)
    Xte_std=np.around(Xte_std,10)
    return Xtr_std,Xte_std

def pearson_fs(Xtrain,Xtest,X_col_name,threshold=0.98):
    num_col=Xtrain.shape[1]
    Xtrain_df=pd.DataFrame(Xtrain)
    r_matrix=Xtrain_df.corr(method='pearson')
    r_matrix_array=np.array(r_matrix)
    col_del_list=[]
    for i in range(num_col):
        for j in range(num_col):
            if j > i:
                if r_matrix_array[i][j] > threshold:
                    col_del_list.append(i)
#     print(col_del_list)
    col_del_list=list(set(col_del_list))
#     print(col_del_list)
    Xtr_pearson_fs=np.delete(Xtrain,col_del_list,axis=1)
    Xte_pearson_fs=np.delete(Xtest,col_del_list,axis=1)
    X_col_name_filtered = [X_col_name[i] for i in range(len(X_col_name)) if i not in col_del_list]

    return Xtr_pearson_fs,Xte_pearson_fs,X_col_name_filtered

def calc_pccs(X,y):
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    res=pd.concat([X,y],axis=1)
    pccs_matrix=res.corr(method='pearson')
    pccs=pccs_matrix.iloc[:,-1]
    pccs=abs(pccs)[:X.shape[1]]
    pccs=np.array(pccs)
    return pccs

def change_order(X,pccs):
    dic={}
    change_list=[]
    for i,j in enumerate(pccs):  
        dic[j]=i    
    pccs.sort()
    pccs=list(pccs)
    pccs.reverse() 
    pccs=np.array(pccs)
    for r in pccs:
        change_list.append(dic[r])  
    return change_list

def change_X(X,change_list):
    X=pd.DataFrame(X)
    df={}
    pd_list=[]
    for i in range(X.shape[1]):
        df[i]=X.iloc[:,i]
    for j in change_list:
        pd_list.append(df[j])
    X_changed=pd.concat(pd_list,axis=1)
    return X_changed

def pearson_order(Xtrain,ytrain,Xtest):
    pccs=calc_pccs(Xtrain,ytrain)
    change_list=change_order(Xtrain,pccs)
    Xtr_pearson_order=change_X(Xtrain,change_list)
    Xte_pearson_order=change_X(Xtest,change_list)
    return Xtr_pearson_order,Xte_pearson_order

def SFS_fs(Xtrain,ytrain,Xtest):
    Xtrain,Xtest=pearson_order(Xtrain,ytrain,Xtest)
    Xtrain=pd.DataFrame(Xtrain)
    feature_columns=[0,1,2,3,4]
    Xtr_select=Xtrain.iloc[:,feature_columns]  
    
    Xtr_resampled,ytr_resampled=RandomOverSampler(random_state=1).fit_resample(np.array(Xtr_select),ytrain)
    clf=RandomForestClassifier(random_state=1,n_estimators=100,max_depth=MaxDepth)
    cv=StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=1)
    score=cross_val_score(clf,Xtr_resampled,ytr_resampled,cv=cv,scoring='f1').mean()
    
    for i in  np.arange(5,Xtrain.shape[1]):
        Xtr_select_=pd.concat([Xtr_select,Xtrain.iloc[:,i]],axis=1)
        Xtr_resampled,ytr_resampled=RandomOverSampler(random_state=1).fit_resample(Xtr_select_,ytrain)
        score_=cross_val_score(clf,Xtr_resampled,ytr_resampled,cv=cv,scoring='f1').mean()
        if score_>= score:
            score=score_
            feature_columns.append(i)
            Xtr_select=Xtr_select_
    Xte_select=pd.DataFrame(Xtest).iloc[:,feature_columns]
    return Xtr_select, Xte_select 


def select_best_param(Xtr_resampled,ytr_resampled,maxdepth,scoring='f1',model='rf'):
    if model=='rf':
        param_grid={"n_estimators":[140],#np.arange(50,150,10)
                "max_depth":[maxdepth],
                "min_samples_split":[4],#np.arange(2,5)
                "max_features":[7]#['auto','sqrt']
               }

        clf=RandomForestClassifier(random_state=1)
    elif model=='svc':
        param_grid={'kernel':['rbf'],'gamma':np.arange(0.01,0.11,0.01),'C':np.arange(0.8,1.2,0.1)}
        clf=SVC(random_state=1)
    elif model=='lr':
        param_grid={"C":np.linspace(0.05,1,19), "max_iter":[100]}
        clf=LR(random_state=1,solver='liblinear',penalty='l2')
        
    cv=StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=1)
    GS=GridSearchCV(clf,param_grid,cv=cv,scoring=scoring)
    GS.fit(Xtr_resampled,ytr_resampled)
    best_param=GS.best_params_
    print(best_param)
    return best_param

def select_best_model(Xtr_resampled,ytr_resampled,maxdepth,scoring='f1',model='rf'):
    #model_=model
    #best_param=select_best_param(Xtr_resampled,ytr_resampled,maxdepth,scoring=scoring,model=model_)
    if model=='rf':
        clf=RandomForestClassifier(random_state=1,n_estimators=best_param["n_estimators"],max_depth=best_param['max_depth'],
                               min_samples_split=best_param['min_samples_split'],max_features=best_param['max_features']
                              )
    elif model=='svc':
        clf=SVC(random_state=1,C=best_param['C'], gamma= best_param['gamma'],kernel='rbf',probability=True)
    elif model=='lr':
        clf=LR(random_state=1,solver='liblinear',penalty='l2',C=best_param['C'],max_iter=best_param['max_iter'])
        
    clf_fitted=clf.fit(Xtr_resampled,ytr_resampled)
    return clf,clf_fitted

def calc_scores(X,y,clf_fitted,model='rf'):
    clf=clf_fitted
    scores=[]
    accuracy=clf.score(X,y)
    y_pred=clf.predict(X)
    recall=recall_score(y,y_pred)
    precision=precision_score(y,y_pred)
    f1score=f1_score(y,y_pred)
    if model=='rf' or model=='lr' or model=='votingclf':
        area=AUC(y,clf.predict_proba(X)[:,1])
    elif model=='svc':
        area=AUC(y,clf.decision_function(X))
    scores.append(accuracy)
    scores.append(recall)
    scores.append(precision)
    scores.append(area)
    scores.append(f1score)
    return scores

def rf_embedded_fs_threshold(Xtrain,ytrain):
    rfc=RandomForestClassifier(random_state=1,n_estimators=100,max_depth=MaxDepth)
    Xtr_resampled,ytr_resampled=RandomOverSampler(random_state=1).fit_resample(Xtrain,ytrain)
    rfc.fit(Xtr_resampled,ytr_resampled).feature_importances_
    threshold=np.linspace(0,(rfc.fit(Xtr_resampled,ytr_resampled).feature_importances_).max(),30)
    threshold=[float("{0:.6f}".format(j)) for j in threshold]
    
    
    score=[]
    for i in threshold:
        Xtr_embedded=SelectFromModel(rfc,threshold=i).fit_transform(Xtr_resampled,ytr_resampled)
        cv=StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=1)
        once=cross_val_score(rfc,Xtr_embedded,ytr_resampled,cv=cv,scoring='accuracy').mean()
        score.append(once)
        
    print(threshold)
    print(score)
    plt.figure()
    plt.figure(figsize=(20,5))
    plt.plot(threshold,score)
    plt.show()
    
    score_max=max(score)
    for m in range(len(score)):
        if abs(score_max-score[m]) < 1e-6:
            best_threshold=threshold[m]
            
    return best_threshold

def rf_embedded_fs_X(Xtrain,ytrain,Xtest,threshold):
    rfc=RandomForestClassifier(random_state=1,n_estimators=100,max_depth=MaxDepth)
    Xtr_resampled,ytr_resampled=RandomOverSampler(random_state=1).fit_resample(Xtrain,ytrain)
    rfc.fit(Xtr_resampled,ytr_resampled).feature_importances_
    sfm=SelectFromModel(rfc,threshold=threshold).fit(Xtr_resampled,ytr_resampled)
    Xtr_embedded=sfm.transform(Xtrain)
    Xte_embedded=sfm.transform(Xtest)
    return Xtr_embedded,Xte_embedded
 
def rf_wrapper_fs_threshold(Xtrain,ytrain):
    rfc=RandomForestClassifier(random_state=1,n_estimators=100,max_depth=MaxDepth)
    Xtr_resampled,ytr_resampled=RandomOverSampler(random_state=1).fit_resample(Xtrain,ytrain)
    
    score=[]
    range_list=list(range(1,Xtrain.shape[1],1))
    for i in range_list:
        Xtr_wrapper=RFE(rfc,n_features_to_select=i,step=1).fit_transform(Xtr_resampled,ytr_resampled)
        cv=StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=1)
        once=cross_val_score(rfc,Xtr_wrapper,ytr_resampled,cv=cv,scoring='f1').mean()
        score.append(once)
        
    print(range_list)
    print(score)
    plt.figure()
    plt.figure(figsize=(20,5))
    plt.plot(range_list,score)
    plt.xticks(range_list)
    plt.show()
    
    score_max=max(score)
    for m in range(len(score)):
        if abs(score_max-score[m]) < 1e-6:
            best_threshold=range_list[m]
            
    return best_threshold

def rf_wrapper_fs_X(Xtrain,ytrain,Xtest,threshold):
    rfc=RandomForestClassifier(random_state=1,n_estimators=100,max_depth=MaxDepth)
    Xtr_resampled,ytr_resampled=RandomOverSampler(random_state=1).fit_resample(Xtrain,ytrain)
    
    rfe=RFE(rfc,n_features_to_select=threshold,step=1).fit(Xtr_resampled,ytr_resampled)

    Xtr_wrapper=rfe.transform(Xtrain)
    Xte_wrapper=rfe.transform(Xtest)
    return Xtr_wrapper,Xte_wrapper

def svm_rfe_threshold(Xtrain,ytrain):
    svc=SVC(kernel="linear",random_state=1)
    Xtr_resampled,ytr_resampled=RandomOverSampler(random_state=1).fit_resample(Xtrain,ytrain)
    
    score=[]
    range_list=list(range(1,Xtrain.shape[1],1))
    for i in range_list:
        Xtr_wrapper=RFE(svc,n_features_to_select=i,step=1).fit_transform(Xtr_resampled,ytr_resampled)
        cv=StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=1)
        once=cross_val_score(svc,Xtr_wrapper,ytr_resampled,cv=cv,scoring='f1').mean()
        score.append(once)
        
    print(range_list)
    print(score)
    
    score_max=max(score)
    for m in range(len(score)):
        if abs(score_max-score[m]) < 1e-6:
            best_threshold=range_list[m]
            
    return best_threshold

def svm_rfe_X(Xtrain,ytrain,Xtest,threshold,X_col_name):
    svc=SVC(kernel="linear",random_state=1)
    Xtr_resampled,ytr_resampled=RandomOverSampler(random_state=1).fit_resample(Xtrain,ytrain)

    rfe=RFE(svc,n_features_to_select=threshold,step=1).fit(Xtr_resampled,ytr_resampled)

    selected_features_idx = rfe.support_
    X_col_name_filtered = [col_name for i, col_name in enumerate(X_col_name) if selected_features_idx[i]]

    Xtr_wrapper=rfe.transform(Xtrain)
    Xte_wrapper=rfe.transform(Xtest)
    return Xtr_wrapper,Xte_wrapper,X_col_name_filtered

def get_feature_indice(Xtrain,Xtr_sel):
    Xtrain_num_col=Xtrain.shape[1]
    Xtr_sel_num_col=Xtr_sel.shape[1]
    Xtrain=pd.DataFrame(Xtrain)
    Xtr_sel=pd.DataFrame(Xtr_sel)
    feature_num=[]
    for i in range(Xtr_sel_num_col):
        X1=np.array(Xtr_sel.iloc[:,i])
        for j in range(Xtrain_num_col):
            X2=np.array(Xtrain.iloc[:,j])
            s=0
            for k in range(len(X1)):
                m=abs(X1[k]-X2[k])
                s=s+m
                
            if s==0:
                feature_num.append(j)
    return feature_num

def generate_csv(Xtrain,ytrain,Xtest,ytest,random):
    for score in ["f1"]:
        
        result_train={}
        result_test={}
        maxdepth=5
        Xtr_resampled,ytr_resampled=RandomOverSampler(random_state=1).fit_resample(Xtrain,ytrain)
        print((Xtr_resampled.shape,ytr_resampled.shape))
        clf,clf_fitted=select_best_model(Xtr_resampled,ytr_resampled,maxdepth,scoring=score,model='rf')
        train_scores=calc_scores(Xtrain,ytrain,clf_fitted,model='rf')
        test_scores=calc_scores(Xtest,ytest,clf_fitted,model='rf')
        result_train[random]=train_scores
        result_test[random]=test_scores
        res_train=pd.DataFrame(result_train,index=["accuracy","recall","precision","AUC","F1"]).T
        res_test=pd.DataFrame(result_test,index=["accuracy","recall","precision","AUC","F1"]).T
        results=pd.concat([res_train,res_test],axis=1)
        results.to_csv('90_re_%s.csv' % (score),mode='a',header=False)  

    return clf_fitted

def relief_fs(Xtrain,ytrain,Xtest,n):
    r=relief.Relief(n_features=n,random_state=1)
    r=r.fit(Xtrain,ytrain)
    Xtr_rel=r.transform(Xtrain)
    Xte_rel=r.transform(Xtest)
    return Xtr_rel,Xte_rel

def get_relief_n(Xtrain,ytrain,Xtest):
    score_dict={}
    score_dict['para_n']=1
    score_dict['score']=0
    for n in np.arange(1,Xtrain.shape[1]+1):
        Xtr_rel_,Xte_rel_=relief_fs(Xtrain,ytrain,Xtest,n)
        f1score=get_f1_cvscore(Xtr_rel_,ytrain,Xte_rel_)
        if f1score > score_dict['score']:
            score_dict['score']=f1score
            score_dict['para_n']=n

    para_n=score_dict['para_n']
    return para_n

def get_f1_cvscore(Xtrain,ytrain,Xtest):
    Xtr_resampled,ytr_resampled=RandomOverSampler(random_state=1).fit_resample(Xtrain,ytrain)
    clf=RandomForestClassifier(random_state=1,n_estimators=100,max_depth=MaxDepth)
    cv=StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=1)
    score=cross_val_score(clf,Xtr_resampled,ytr_resampled,cv=cv,scoring='f1').mean()
    return score

if __name__ == "__main__": 
    resampler=[RandomOverSampler(random_state=1)]
    with open("ee90_81015_sfjtnn_sel.csv", "r") as file:
        fea_selected=pd.read_csv(file).iloc[:,1:].columns 

    with open("amidase_test_jtnnsf.csv", "r") as file:
        newdata=pd.read_csv(file).iloc[:,1:][fea_selected]

    with open("amidase_jtnnsf.csv", "r") as file:
        X=pd.read_csv(file).iloc[:,1:][fea_selected]

    print(X.shape)

    with open("g_classification90.csv", "r") as file:
        y=pd.read_csv(file).iloc[:,1:]   
    y=np.ravel(y)
    print(y.shape)

    X,y=shuffle_data(X,y)
    Xtrain,Xtest,ytrain,ytest=data_split(X,y,99)
    model_fitted=generate_csv(Xtrain,ytrain,Xtest,ytest,1)
    newdata_pred_res=[]
    print(model_fitted.predict(newdata))
    newdata_pred_res.append(model_fitted.predict(newdata))
    newdata_pred_res_df=pd.DataFrame(newdata_pred_res)
    newdata_pred_res_df.to_csv('90_test.csv', mode='a', header=False)
