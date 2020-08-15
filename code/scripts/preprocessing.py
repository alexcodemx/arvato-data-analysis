import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Documentation file route may need to be adjusted

def load_documentation():
    excel_file = pd.read_excel('../files/DIAS Attributes - Values 2017.xlsx', sheet_name='Tabelle1', header=1, usecols=[1,2,3,4],
                          dtype='object')
    
    return excel_file

def drop_nans(df, threshold = 0.9):
    '''
        Drop columns with more than 90% of nulls
        input:
            df - Dataframe to be analyzed
            
        output:
            df - Dataframe without columns with more than 90% of nulls
    '''
    # Get nans percentage
    frame_threshold = ((df.isnull().sum()/df.shape[0]).sort_values(ascending=False)>=threshold).to_frame()
    
    # Get col names that are above the threshold
    list_colnames = frame_threshold.loc[frame_threshold[0]==True].index.to_list()
    
    try:
        df = df.drop(columns=list_colnames)
        print("The columns below threshold: {} were removed.".format(list_colnames))
    except:
        print("Error! No columns to remove")
        
    return df


def impute_median(df):
    '''
        Impute with median the DF
        
        input:
            df - Dataframe to be imputed
        output: 
            df - Imputed Dataframe
    '''
    exp = lambda x: x.fillna(x.median())
    df = df.apply(exp,axis=0)
    
    return df


def scale_features(df):
    '''
        Scale dataframe features for unsupervised and machine learning algorithms.
        input:
            df - Dataframe to be scaled
        output:
            df - Scaled Dataframe
    '''
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled,index=df.index, columns=df.columns)
    
    return df_scaled

def principal_components(df, n_components = 101):
    n_components = n_components
    whiten = False
    random_state = 2018

    pca = PCA(n_components=n_components, whiten=whiten, \
              random_state=random_state)

    train_index = range(0,len(df))
    X_train_PCA = pca.fit_transform(df)
    X_train_PCA = pd.DataFrame(data=X_train_PCA, index=train_index)
    
    print("Variance Explained by all {} principal components: ".format(X_train_PCA.shape[1]), \
      sum(pca.explained_variance_ratio_))
    
    return X_train_PCA, pca

def data_preprocessing(df):
    '''
        Data cleaning and preparation.
        input:
            df - Dataframe to be preprocessed
            documentation_df - Excel documentation file provided
        
        output:
            df - Processed dataframe
    '''
    
    # Process documentation file
    try:
        documentation_df = load_documentation()
    except:
        print("Documentation file is missing, cannot proceed..")
        return
    

    documentation_df = documentation_df.fillna(method='ffill')
    documentation_df.Attribute = documentation_df.Attribute.str.upper().str.strip()
    documentation_df.Attribute = documentation_df.Attribute.str.replace(r'\_RZ','')
    documentation_df.Meaning = documentation_df.Meaning.str.lower().str.strip()
    
    # Get documentation column names. This will be used for dropping from the dataframes
    # the columns that do not exist in the documentation.
    docs_columns = set(documentation_df.Attribute)
    
    # Get df column names. This will be used to compare to the documentation columns
    df_columns = set(df.columns)
    
    # Get df columns to be dropped
    drop_target_df_cols = df_columns.difference(docs_columns)
    
    # Drop columns that do not exist in the documentation

    print("Old Dataframe Shape: {}".format(df.shape))
    
    # Drop columns from dataframe
    try:
        df = df.drop(columns=drop_target_df_cols)
        print("\nSuccesfully removed {} columns!".format(len(drop_target_df_cols)))
    except:
        print("\nError! Check that column names exist in the dataframe", )
    
    
    # Replace Xs with nan in the df. Documentation has no X or XX values
    exp_clean_xs = lambda x: np.nan if x == 'X' or x == 'XX' else x
    
    for name in df.columns:
        df[name] = df[name].apply(exp_clean_xs)
            
    # Convert data types to number or object according to documentation
    
    for name in df.columns:
        if name not in ['CAMEO_DEU_2015','OST_WEST_KZ']:
            df[name] = pd.to_numeric(df[name], downcast='integer')
            
    df = pd.get_dummies(df, columns=['CAMEO_DEU_2015','OST_WEST_KZ'])
    '''
    # Convert LNR as index column
    
    df = df.set_index('LNR')
    '''
    # Get unknown variable values from the documentation
    
    unknown_variables = documentation_df[documentation_df['Meaning'].str.contains('unknown')]\
        .assign(split_val=documentation_df['Value'].str.split(',')).explode('split_val')
    unknown_variables.Value = unknown_variables.split_val.combine_first(unknown_variables.Value)
    unknown_variables = unknown_variables[['Attribute','Value']]
    
    # Convert unknown variables to null
    
    for i in range(unknown_variables.shape[0]):
        attribute = unknown_variables.iloc[i,0]
        value = int(unknown_variables.iloc[i,1])

        try:
            exp = lambda x: np.nan if x == value else x
            df[attribute]= df[attribute].apply(exp)
        except:
            print("{} was not found in the DF".format(attribute))
    
    # Drop columns with more than 90% of nulls.
    df = drop_nans(df)
    
    # Impute the columns with the median
    df = impute_median(df)
    
    # Scale features
    df = scale_features(df)
    
    return df