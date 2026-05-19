import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations
from numba import njit
import copy
from hmmlearn import hmm
from scipy.cluster.hierarchy import linkage, fcluster
import sklearn as sk
import sklearn.ensemble as ske
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import sklearn.metrics as sklm
import sklearn.svm as svm
#import statsmodels.tools as smt
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from . import analytics as an

def corr_var_selection(X,threshold=0.94):
    corr_matrix = X.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_reduced = X.drop(columns=to_drop)
    return X_reduced

def vif_var_selection(X,max_iter=100,threshold=10):
    # select vars based on VIF for each feature
    i = 0
    problems = True
    X_nonconst = X.loc[:, X.columns != 'const']
    while i<max_iter and problems:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_nonconst.columns
        vif_data["VIF"] = [variance_inflation_factor(X_nonconst.values, i) for i in range(len(X_nonconst.columns))]
        # Aim for VIF < 5 or 10. Drop anything higher.
        vif_data = vif_data.sort_values("VIF", ascending=False)
        to_drop = vif_data.iloc[0,:].feature
        X_nonconst = X_nonconst.drop(columns=to_drop)
        i+=1
        if vif_data.iloc[0,:].VIF < threshold:
            problems = False
    return X_nonconst

def adfuller_correction(X,threshold=0.94):
    out=[]
    for c in X:
        if c!='const':
            adf_stat, p_val, _, _, _, _ = adfuller(X[c].dropna())
            out.append((c,float(p_val)))
    out = pd.DataFrame(out)
    out.columns=["col","p_val"]
    to_alter = out[(out.p_val>=0.67)].col
    for c in to_alter:
        if (X[c].min()>0):
            X[c] = an.frac_diff(X[c],d=0.5)
        else:
            X[c] = an.frac_diff(X[c],d=0.5,lin=True) 
    to_alter = out[(out.p_val>0.06) & (out.p_val<0.67)].col
    for c in to_alter:
        if (X[c].min()>0):
            X[c] = an.frac_diff(X[c],d=0.25)
        else:
            X[c] = an.frac_diff(X[c],d=0.25,lin=True)
    return X

def markovitzrf(covar, rf=0, target=0.15, mju=0):
    covar = np.array(covar)
    k = covar.shape[1]  # Get number of assets
    ones = np.ones(k)
    s_inv = np.linalg.inv(covar)
    
    # Calculate excess returns (mjubis)
    if np.isscalar(mju) and mju == 0:
        # Conceptual Note: This effectively finds the Minimum Variance Portfolio
        mjubis = np.full(k, target) - rf
    else:
        mjubis = np.array(mju) - rf

    # --- Choice A: Tangent Portfolio (Weights sum to 1) ---
    # Formula: (S^-1 * mjubis) / (1' * S^-1 * mjubis)
    weights_tangent = (s_inv @ mjubis) / (ones @ s_inv @ mjubis)
    
    # --- Choice B: Optimal x (Target Return Portfolio) ---
    # Formula: (target - rf) * (S^-1 * mjubis) / (mjubis' * S^-1 * mjubis)
    # denom_opt = mjubis @ s_inv @ mjubis
    # weights_opt = (target - rf) * (s_inv @ mjubis) / denom_opt

    return weights_tangent

def find_best_ew_subset(means, stds, corr_matrix, rf=0.0):
    n = len(means)
    # Convert correlations to covariance matrix
    cov_matrix = np.outer(stds, stds) * corr_matrix
    
    best_sr = -np.inf
    current_subset = []
    remaining_assets = list(range(n))
    
    # Greedy Forward Selection
    for _ in range(n):
        best_candidate = None
        best_candidate_sr = -np.inf
        
        for asset in remaining_assets:
            test_subset = current_subset + [asset]
            k = len(test_subset)
            
            # Sub-components for current test combination
            sub_means = means[test_subset]
            sub_cov = cov_matrix[np.ix_(test_subset, test_subset)]
            
            # Equal weighting: w = 1/k
            w = np.ones(k) / k
            
            # Portfolio Statistics
            port_ret = np.dot(w, sub_means)
            port_vol = np.sqrt(np.dot(w.T, np.dot(sub_cov, w)))
            sr = (port_ret - rf) / port_vol
            
            if sr > best_candidate_sr:
                best_candidate_sr = sr
                best_candidate = asset
        
        # If adding the best candidate improves overall SR, keep it
        if best_candidate_sr > best_sr:
            best_sr = best_candidate_sr
            current_subset.append(best_candidate)
            remaining_assets.remove(best_candidate)
        else:
            # Stop if adding more assets dilutes the Sharpe Ratio
            break
            
    return current_subset, best_sr

def backward_elimination_ols(X, y, initial_list, significance_level=0.05, silent=False):
    features = list(initial_list)
    while True:
        # Prepare data for statsmodels (add constant term for intercept)
        X_const = sm.add_constant(X[features])
        
        # Fit logistic regression model
        model = sm.OLS(y, X_const).fit(disp=0) # disp=0 hides the optimization output
        #model = sm.Logit(y, X_const).fit(disp=0)
            
        # Get p-values
        p_values = model.pvalues
        
        # Find the feature with the highest p-value (excluding the constant)
        max_p_value = p_values[1:].max() 
        
        if max_p_value > significance_level:
            least_significant_feature = p_values[1:].idxmax()
            features.remove(least_significant_feature)
            if not silent:
                print(f"Removing '{least_significant_feature}' with p-value {max_p_value:.4f}")
        else:
            if not silent:
                print("All remaining features are significant.")
            break
            
    return model, features

def backward_elimination_flexible(X, y, initial_list, model_class=sm.OLS, significance_level=0.05):
    """
    Performs backward elimination using a flexible statsmodels model class.

    Args:
        X (pd.DataFrame): DataFrame of independent variables (predictors).
        y (pd.Series): Series of the target variable.
        initial_list (list): The starting list of feature names.
        model_class: The statsmodels model class (e.g., sm.OLS, sm.Logit, sm.GLM).
        significance_level (float): The alpha level for p-value removal threshold.

    Returns:
        tuple: (final_model, selected_features_list)
    """
    features = list(initial_list)
    
    # Check if the provided model class is valid (has 'fit' method)
    if not hasattr(model_class, 'fit'):
        raise ValueError("Provided model_class must be a valid statsmodels model class (e.g., sm.OLS, sm.Logit)")

    while True:
        # Prepare data for statsmodels (add constant term for intercept)
        X_const = sm.add_constant(X[features])
        
        # Fit the model using the provided model_class
        # We pass disp=0 to Logit to silence iteration output, OLS ignores it safely
        try:
            model = model_class(y, X_const).fit(disp=0) 
        except TypeError:
             # Handle OLS/GLM which don't accept 'disp' argument
            model = model_class(y, X_const).fit()
        
        # Get p-values
        p_values = model.pvalues
        
        # Find the feature with the highest p-value (excluding the constant/intercept)
        # We use [1:] to skip the 'const' term in the p-values series
        max_p_value = p_values[1:].max() 
        
        if max_p_value > significance_level:
            # Get the name of the least significant feature
            least_significant_feature = p_values[1:].idxmax()
            features.remove(least_significant_feature)
            print(f"Removing '{least_significant_feature}' with p-value {max_p_value:.4f}")
        else:
            print("All remaining features are significant.")
            break
            
    return model, features

def forward_selection_ols(X, y, initial_features=[], significance_level=0.05, silent=False):
    remaining_features = set(X.columns)
    selected_features = set(initial_features)
    
    while True:
        best_p_value = 1.0
        feature_to_add = None
        
        for feature in remaining_features:
            current_features = list(selected_features) + [feature]
            X_const = sm.add_constant(X[current_features])
            model = sm.OLS(y, X_const).fit(disp=0)
            #model = sm.Logit(y, X_const).fit(disp=0)
            
            # Check the p-value of the newly added feature
            # The last feature in current_features is the one we just added
            p_value = model.pvalues[feature] 
            
            if p_value < best_p_value:
                best_p_value = p_value
                feature_to_add = feature
        
        if best_p_value < significance_level:
            remaining_features.remove(feature_to_add)
            selected_features.add(feature_to_add)
            if not silent:
                print(f"Adding '{feature_to_add}' with p-value {best_p_value:.4f}")
        else:
            if not silent:
                print("No more features to add that meet the significance level.")
            break
            
    # Fit the final model with the selected features
    final_features_list = list(selected_features)
    X_const_final = sm.add_constant(X[final_features_list])
    final_model = sm.OLS(y, X_const_final).fit()
    
    return final_model, final_features_list

def forward_selection_flexible(X, y, initial_features=[], model_class=sm.OLS, significance_level=0.05):
    """
    Performs forward selection using a flexible statsmodels model class.

    Args:
        X (pd.DataFrame): DataFrame of independent variables (predictors).
        y (pd.Series): Series of the target variable.
        initial_features (list): Starting features (usually an empty list).
        model_class: The statsmodels model class (e.g., sm.OLS, sm.Logit).
        significance_level (float): The alpha level for p-value entry threshold.

    Returns:
        tuple: (final_model, selected_features_list)
    """
    remaining_features = set(X.columns)
    selected_features = set(initial_features)
    
    # Remove any initial features from the remaining pool
    remaining_features -= selected_features

    # Check if the provided model class is valid
    if not hasattr(model_class, 'fit'):
        raise ValueError("Provided model_class must be a valid statsmodels model class.")

    while True:
        best_p_value = 1.0
        feature_to_add = None
        
        # Iterate over all remaining features to see which is the best candidate to add
        for feature in remaining_features:
            current_features_list = list(selected_features) + [feature]
            X_const = sm.add_constant(X[current_features_list])
            
            # Fit the model
            try:
                model = model_class(y, X_const).fit(disp=0, maxiter=50) 
            except TypeError:
                model = model_class(y, X_const).fit(maxiter=50) # OLS/GLM don't accept 'disp'
            except Exception as e:
                # Handle potential convergence issues for some complex models during intermediate steps
                print(f"Warning: Model fitting failed for subset {current_features_list}: {e}")
                continue

            # Check the p-value of the NEWLY added feature (the last one in the list)
            # This relies on X_const maintaining order
            p_value = model.pvalues[feature] 
            
            if p_value < best_p_value:
                best_p_value = p_value
                feature_to_add = feature
        
        # Check if the best candidate meets the entry significance level
        if feature_to_add is not None and best_p_value < significance_level:
            remaining_features.remove(feature_to_add)
            selected_features.add(feature_to_add)
            print(f"Adding '{feature_to_add}' with p-value {best_p_value:.4f}")
        else:
            print("No more features to add that meet the significance level.")
            break
            
    # Fit the final model with the selected features
    final_features_list = list(selected_features)
    X_const_final = sm.add_constant(X[final_features_list])
    final_model = model_class(y, X_const_final).fit()
    
    return final_model, final_features_list

def forward_backward_ols(X, y, initial_features=[], significance_level=0.05, silent=False):
    final_model, final_features_list = forward_selection_ols(X, y, initial_features, significance_level, silent)
    return backward_elimination_ols(X, y, final_features_list, significance_level, silent)

def backward_forward_ols(X, y, initial_list, significance_level=0.05):
    final_model, final_features_list = backward_elimination_ols(X, y, initial_list, significance_level)
    return forward_selection_ols(X, y, final_features_list, significance_level)
    
def backward_elimination_ols_formula(data, target_variable, initial_features, significance_level=0.05):
    # this is feature backward elimination working on OLS formula API call, it is robust against patsyerrors
    features = list(initial_features)
    
    # Wrap all names in backticks for safety
    target_var_safe = f'`{target_variable}`'
    
    while True:
        # Wrap the current feature names in backticks for the formula
        safe_features = [f'`{f}`' for f in features]
        
        # Construct the formula string
        formula = f'{target_var_safe} ~ {" + ".join(safe_features)}'
        
        print(f"Fitting model with formula: {formula}") # Debugging print
        
        # Fit the OLS model using the formula API
        model = smf.ols(formula=formula, data=data).fit()
        
        # Get p-values (Need original names here to match the 'features' list)
        # We drop the intercept (which patsy names "Intercept")
        p_values = model.pvalues.drop('Intercept', errors='ignore') 
        
        if p_values.empty or p_values.max() <= significance_level:
            print("All remaining features are significant or no features left.")
            break

        # Find the original name of the least significant feature
        # The index names in p_values will match the 'safe_features' strings (with backticks)
        least_significant_safe_name = p_values.idxmax()
        max_p_value = p_values.max()

        # Remove the backticks to find the name in our original list
        least_significant_original_name = least_significant_safe_name.strip('`')
        
        if max_p_value > significance_level:
            features.remove(least_significant_original_name)
            print(f"Removing '{least_significant_original_name}' with p-value {max_p_value:.4f}")
            
    return model, features

# Use Numba to accelerate the H_k calculation logic itself
# We need to pass numpy arrays to this function
@njit(parallel=False) # parallel=True might offer marginal gains but adds overhead
def calculate_hk_score(subset_corr_matrix, target_corrs_abs):
    """Numba accelerated calculation of the H_k score."""
    numerator_sum = np.sum(target_corrs_abs)
    
    # Calculate sum of squared off-diagonal elements efficiently in numpy
    # np.fill_diagonal is not supported by numba JIT yet
    
    # Efficient Numba/Numpy way to get off-diagonal sum of squares:
    # 1. Square the whole matrix
    squared_matrix = subset_corr_matrix**2
    # 2. Subtract the squared diagonal elements
    sum_all_squared = np.sum(squared_matrix)
    sum_diag_squared = np.sum(np.diag(squared_matrix))
    denominator_sum_squared = sum_all_squared - sum_diag_squared
    
    denominator = np.sqrt(1 + denominator_sum_squared)
    
    h_k = numerator_sum / denominator
    return h_k

def hellwig_feature_selection_optimized(X, y):
    """
    Implements the Hellwig method with Numba optimization for calculation.
    """
    
    data = pd.concat([X, y], axis=1)
    correlation_matrix = data.corr().values # Convert to numpy array once
    feature_names = X.columns.tolist()
    target_index = correlation_matrix.shape[1] - 1
    
    best_h_k = -np.inf
    best_subset = []
    
    # Map feature names to their indices for easier indexing of the numpy matrix
    feature_indices = {name: i for i, name in enumerate(feature_names)}

    for k in range(1, len(feature_names) + 1):
        for subset_names in combinations(feature_names, k):
            
            # Get indices for current subset
            subset_indices = [feature_indices[name] for name in subset_names]
            
            # Extract relevant slices using numpy indexing (faster with indices)
            # correlations of features within themselves (square sub-matrix)
            subset_corr_matrix = correlation_matrix[np.ix_(subset_indices, subset_indices)]
            
            # correlations of features with the target (vector)
            target_corrs_abs = np.abs(correlation_matrix[subset_indices, target_index])
            
            # Use the Numba-accelerated function
            h_k = calculate_hk_score(subset_corr_matrix, target_corrs_abs)
            
            if h_k > best_h_k:
                best_h_k = h_k
                best_subset = list(subset_names) # Convert tuple back to list for return

    return {"best_features": best_subset, "H_k_score": best_h_k}

def get_hellwig_score_for_subset(X, y, feature_subset):
    """
    Calculates the Hellwig H_k score for a specific subset of features.

    Args:
        X (pd.DataFrame): DataFrame of all potential independent variables.
        y (pd.Series): Series of the target variable.
        feature_subset (list): A list of column names (strings) in X to evaluate.

    Returns:
        float: The Hellwig H_k score for that specific subset.
    """
    
    if not feature_subset:
        return 0.0 # Score is zero if no features are selected

    # 1. Combine only the necessary data for correlation calculation
    X_subset = X[feature_subset]
    data_subset = pd.concat([X_subset, y], axis=1)
    
    # 2. Calculate the correlation matrix (convert to numpy for Numba compatibility)
    correlation_matrix = data_subset.corr().values
    
    # The target is now always the last column in this specific matrix
    target_index = correlation_matrix.shape[1] - 1
    
    # 3. Extract the required matrices as numpy arrays:
    # correlations of features within themselves (square sub-matrix, excluding target row/col)
    subset_corr_matrix = correlation_matrix[:-1, :-1]
    
    # correlations of features with the target (vector, absolute values)
    target_corrs_abs = np.abs(correlation_matrix[:-1, target_index])
    
    # 4. Use the Numba-accelerated function to get the final score
    h_k_score = calculate_hk_score(subset_corr_matrix, target_corrs_abs)
    
    return h_k_score

def hmmfit(model,df,m=100, bestof=1):
    #model is of hmmlearn hmm class
    data = df.dropna()
    for i in range(bestof):
        model_tmp = copy.deepcopy(model)
        model_tmp.fit(data.values.reshape(-1, 1)*m)
        score = model_tmp.score(data.values.reshape(-1, 1)*m)
        if i==0:
            max_score=score
        if score>=max_score:
            max_model = model_tmp
            max_score = score
    states = pd.Series(max_model.predict(data.values.reshape(-1, 1)*m),index=data.index)
    if (max_model.n_components==2) & (states.mean() < 0.5):
        max_model = switch2classhmm(max_model)   
    return max_model

def switch2classhmm(model):
    # HMM models does not care what is labeled as 0 or 1, it also often switches them during stochastic calibration
    # this function allows to switch 0 and 1 labels, e.g. when someone wants to have dominant label always as 0
    model.startprob_ = model.startprob_[::-1]
    # Swap rows, then swap columns
    model.transmat_ = model.transmat_[::-1, ::-1]
    # For GaussianHMM: Swap the means and covariances.python
    model.means_ = model.means_[::-1]
    model._covars_ = model._covars_[::-1]
    # For MultinomialHMM: Swap the rows of the emission probability matrix.python
    # model.emissionprob_ = model.emissionprob_[::-1]
    return model
    
def hmmpredict(model,df,m=100,prob=False):
    data = df.dropna()
    if prob:
        return pd.Series(model.predict_proba(data.values.reshape(-1, 1)*m)[:,0],index=data.index)
    else:
        return pd.Series(model.predict(data.values.reshape(-1, 1)*m),index=data.index)

def hmmrollpredict(model,df_train,df_test,s=160,m=100,threshold=0.6,oosonly=False,probs=False):
    df_merged = pd.concat((df_train,df_test)).dropna()
    X_merged = df_merged.values.reshape(-1, 1)*m
    n = len(X_merged)
    X_train = X_merged[0:s,:]
    if probs:
        out0 = np.asarray(model.predict_proba(X_train)[:,0]).reshape(-1)
    else:
        out0 = np.asarray(model.predict(X_train)).reshape(-1)
    out=[]
    # out1=[]
    # out2=[]
    current_regime = out0[-1]
    for i in range(s,n):
        X_train = np.append(X_train, X_merged[i,:].reshape(-1, 1), axis=0)
        #hidden_states_ = model.predict(X_train)
        hidden_states_ = model.predict_proba(X_train)[-1]
        
        # Strategy: Only switch if the new regime is highly probable
        if hidden_states_[1] > threshold:
            new_regime = 1
        elif hidden_states_[0] > threshold:
            new_regime = 0
        else:
            # "No-man's land": Keep the previous regime to reduce whipsaw
            new_regime = current_regime
            
        current_regime = new_regime
        if probs:
            out.append(hidden_states_[0])
        else:
            out.append(current_regime)

        # out.append(float(hidden_states_[-1]))
        # out1.append(float(hidden_states_[-2]))
        # out2.append(float(hidden_states_[-3]))
    # col1 = pd.DataFrame(np.concatenate((out0,np.array(out))),index=df_merged.index)        
    # col2 = pd.DataFrame(np.concatenate((out0,np.array(out1))),index=df_merged.index)        
    # col3 = pd.DataFrame(np.concatenate((out0,np.array(out2))),index=df_merged.index)        
    # return pd.concat((col1,col2,col3),axis=1)
    if oosonly:
        return pd.Series(np.array(out),index=df_merged.index[s:])
    else:
        return pd.Series(np.concatenate((out0,np.array(out))),index=df_merged.index)

def assign_hmmstates(y_train,y_test,m=1,algo='map',bestof=1,n_components=2):
    model = hmm.GaussianHMM(n_components, covariance_type="diag", n_iter=1000, algorithm = algo)
    model = hmmfit(model,y_train,m=m,bestof=bestof)
    hs_ins = hmmpredict(model,y_train,m=m)
    hs_oos = hmmpredict(model,y_test,m=m)
    return model, hs_ins, hs_oos

def assign_gmmstates(y_train,y_test,covariance_type='full',n_components=2):
    y_train = pd.DataFrame(y_train) # if we dont convert series to DF then we need not only take .values but .values.reshape(-1, 1)
    y_test = pd.DataFrame(y_test) 
    model = sk.mixture.GaussianMixture(n_components, covariance_type=covariance_type, random_state=0).fit(y_train.dropna().values)
    hs_ins = pd.Series(model.predict_proba(y_train.dropna().values)[:,0],index=y_train.dropna().index)
    hs_oos = pd.Series(model.predict_proba(y_test.values)[:,0],index=y_test.index)
    return model, hs_ins, hs_oos

def transmat(predicted_regimes):
    # Count transitions from state i to state j
    n_states = len(list(set(predicted_regimes)))
    trans_counts = np.zeros((n_states, n_states))
    for t in range(len(predicted_regimes.values) - 1):
        trans_counts[predicted_regimes.values[t], predicted_regimes.values[t+1]] += 1
    # Normalize to get probabilities
    return trans_counts / trans_counts.sum(axis=1, keepdims=True)

def hmmfromexternal(y_train,predicted_regimes,m=100):
    # Assume X is your data (N_samples, N_features) 
    # and labels are your [0, 1] states (N_samples,)
    
    X = (y_train.dropna().values.reshape(-1, 1)*m)
    n_states = 2
    n_features = X.shape[1]
    
    # 1. Initialize the model
    model2 = hmm.GaussianHMM(n_components=n_states, covariance_type="diag")
    
    # 2. Manually calculate and set the parameters
    # Initial probabilities (Startprob)
    model2.startprob_ = np.array([np.sum(predicted_regimes.values[0] == i) for i in range(n_states)], dtype=float)
    
    # Transition Matrix
    model2.transmat_ = transmat(predicted_regimes)
    
    # Emission Parameters (Means and Covars)
    model2.n_features = n_features
    model2.means_ = np.array([X[predicted_regimes.values == i].mean(axis=0) for i in range(n_states)])
    model2.covars_ = np.array([np.var(X[predicted_regimes.values == i], axis=0) for i in range(n_states)])
    
    # 3. Get the Score (Log-Likelihood)
    # This evaluates how well X fits this specific HMM configuration
    fit_score = model2.score(X)
    return model2, fit_score

def hmmAICBIC(model,X):
    # Assuming 'model' is the trained GaussianHMM and 'X' is used data
    log_likelihood = model.score(X)
    n_samples = len(X)
    n_params = get_n_params(model)
    
    # Formulas
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    
    return (aic,bic)

def states_IR(states,X_train): # IR for boolean states of variable X_train
    std = np.array([X_train[states.astype(bool)].std(),X_train[~states.astype(bool)].std()])
    avg = np.array([X_train[states.astype(bool)].mean(),X_train[~states.astype(bool)].mean()])
    return (avg/std)

def states_IR_exog(states,X_train,ret): # IR for boolean states of variable X_train vs exog variable (ret)
    std = np.array([ret[states.astype(bool)].std(),ret[~states.astype(bool)].std()])
    avg = np.array([ret[states.astype(bool)].mean(),ret[~states.astype(bool)].mean()])
    return (avg/std)
                    
def get_n_params(model):
    """
    Calculates the number of free parameters in a GaussianHMM.
    Adjusts based on covariance_type: 'full', 'diag', 'spherical', or 'tied'.
    """
    n_components = model.n_components
    n_features = model.n_features
    
    # Start probabilities: (n_states - 1)
    startprob_params = n_components - 1
    
    # Transition matrix: n_states * (n_states - 1)
    transmat_params = n_components * (n_components - 1)
    
    # Means: n_states * n_features
    means_params = n_components * n_features
    
    # Covariances: dependent on the 'covariance_type'
    if model.covariance_type == 'full':
        covars_params = n_components * n_features * (n_features + 1) / 2
    elif model.covariance_type == 'diag':
        covars_params = n_components * n_features
    elif model.covariance_type == 'spherical':
        covars_params = n_components
    elif model.covariance_type == 'tied':
        covars_params = n_features * (n_features + 1) / 2
    else:
        raise ValueError(f"Unknown covariance type: {model.covariance_type}")
        
    return int(startprob_params + transmat_params + means_params + covars_params)

def clustnknn(X_train,X_test,columns,n_neighbors=10):
    # 1. Scale your features (crucial for distance-based algorithms)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[columns])
    X_test_scaled = scaler.transform(X_test[columns])
    
    # 2. Fit Hierarchical Clustering on X_train only
    Z = linkage(X_train_scaled, method='ward')
    
    # 3. Extract exactly 2 clusters from the training set
    train_labels = fcluster(Z, t=2, criterion='maxclust')-1
    train_labels = pd.Series(train_labels,index=X_train.index)
    # 4. Train a KNN classifier to learn the hierarchical boundaries
    # setting n_neighbors=1 maps test points strictly to their closest training neighbor
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, train_labels)
    
    # 5. Predict out-of-sample clusters safely on X_test
    test_labels = knn.predict(X_test_scaled)    
    test_labels = pd.Series(test_labels,index=X_test.index)
    return knn, train_labels, test_labels

def sgnacc(y,ypred): #strat % of pos returns
    return np.nanmean((np.sign(ypred.values)*np.sign(y.values)+1)/2)

def rf_operationalize(X_train, X_test, y_train, y_test, shift=0, probs=False):
    X_train = pd.DataFrame(y_train).join(X_train.shift(shift))
    # y_train = X_train.iloc[shift:,0] #not needed?
    X_train = X_train.iloc[shift:,1:]
    
    X_test = pd.DataFrame(y_test).join(X_test.shift(shift))
    # y_test = X_test.iloc[shift:,0]
    X_test = X_test.iloc[shift:,1:]
    model = ske.RandomForestClassifier(n_estimators=1000, random_state=0, oob_score=True).fit(X_train, y_train.astype(bool))
    pred1fc = model.predict(X_train); pred1fc = pd.Series(pred1fc, index=X_train.index)
    pred1fc_out = model.predict(X_test); pred1fc_out = pd.Series(pred1fc_out, index=X_test.index)
    acc_in = sgnacc(y_train.astype(float)-0.5,pred1fc.astype(float)-0.5)
    acc_out = sgnacc(y_test.astype(float)-0.5,pred1fc_out.astype(float)-0.5)
    return((model,pred1fc,pred1fc_out,acc_in,acc_out))

def svm_operationalize(X_train, X_test, y_train, y_test, shift=0, kernel='linear', probs=False):
    #kernel{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
    X_train = X_train.dropna(axis=1) #dont use features with nan's
    #X_train = pd.DataFrame(y_train).join(X_train)
    #X_train = X_train.iloc[:,1:]
    X_train = pd.DataFrame(y_train).join(X_train.shift(shift))
    y_train = X_train.dropna().iloc[:,0]
    X_train = X_train.dropna().iloc[:,1:]
    
    X_test = pd.DataFrame(y_test).join(X_test.shift(shift))
    y_test = X_test.dropna().iloc[:,0]
    X_test = X_test.dropna().iloc[:,1:]
    #X_test = X_test[X_train.dropna(axis=1).columns]
    X_test = X_test[X_train.columns]
    #X_train_shifted = X_train_shifted[X_train.columns]
    # Create a pipeline that scales then fits   
    model = make_pipeline(
        # Step 1: Remove features that barely change (low variance)
        # VarianceThreshold(threshold=0.01),
        # Step 2: Scale remaining data
        StandardScaler(),
        # Step 3: Keep only top 10 features with strongest individual relationship to target
        # SelectKBest(score_func=f_classif, k=10),        
        # PCA(n_components=10),
        svm.SVC(kernel=kernel,random_state=0,probability=probs)
    )  
    model.fit(X_train, np.asarray(y_train.astype(bool).values).reshape(-1))
    pred1fc = model.predict(X_train); pred1fc = pd.Series(pred1fc, index=X_train.index)
    pred1fc_out = model.predict(X_test); pred1fc_out = pd.Series(pred1fc_out, index=X_test.index)
    #acc_in = sgnacc(y_train.astype(float)-0.5,pred1fc.astype(float)-0.5)
    #acc_out = sgnacc(y_test.astype(float)-0.5,pred1fc_out.astype(float)-0.5)
    acc_in = sklm.accuracy_score(y_train.astype(float), pred1fc.astype(float))
    acc_out = sklm.recall_score(y_test.astype(float), pred1fc_out.astype(float))
    return((model,pred1fc,pred1fc_out,acc_in,acc_out))    

def lstm_operationalize(X_train, X_test, y_train, y_test, shift=0, probs=False):
    import tensorflow as tf
    from scikeras.wrappers import KerasClassifier
    # Define the model building function
    def create_lstm_model(meta):
        # meta["n_features_in_"] is automatically 216 from your X_train
        n_features = meta["n_features_in_"]
        
        model = tf.keras.Sequential([
            # Input starts as 2D (StandardScaler output)
            tf.keras.layers.Input(shape=(n_features,)),
            
            # Internal Reshape: (Features,) -> (1 TimeStep, Features)
            # This is where we fix the [?, 216] vs [1, 256] mismatch
            tf.keras.layers.Reshape((1, n_features)),
            
            # LSTM now correctly sees 'n_features' as the input_dim
            tf.keras.layers.LSTM(64, activation='tanh'), 
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Setup the wrapper
    lstm_wrapper = KerasClassifier(
        model=create_lstm_model, 
        epochs=10, 
        batch_size=32, 
        validation_split=0.1,
        verbose=1
    )

    # Pipeline is back to your simple, original 2-step format!
    model = make_pipeline(StandardScaler(), lstm_wrapper)

    X_train = X_train.dropna(axis=1) #dont use features with nan's
    X_train = pd.DataFrame(y_train).join(X_train.shift(shift))
    y_train = X_train.dropna().iloc[:,0]
    X_train = X_train.dropna().iloc[:,1:]
    
    X_test = pd.DataFrame(y_test).join(X_test.shift(shift))
    y_test = X_test.dropna().iloc[:,0]
    X_test = X_test.dropna().iloc[:,1:]
    X_test = X_test[X_train.columns]
    
    model.fit(X_train, np.asarray(y_train.astype(bool).values).reshape(-1))
    
    pred1fc = model.predict(X_train); pred1fc = pd.Series(pred1fc, index=X_train.index)
    pred1fc_out = model.predict(X_test); pred1fc_out = pd.Series(pred1fc_out, index=X_test.index)
    acc_in = sklm.accuracy_score(y_train.astype(float), pred1fc.astype(float))
    acc_out = sklm.recall_score(y_test.astype(float), pred1fc_out.astype(float))    
    
    return((model,pred1fc,pred1fc_out,acc_in,acc_out))

def get_confusion_matrix(y_true, y_pred):
    # Find unique classes and map them to indices
    classes = sorted(list(set(y_true) | set(y_pred)))
    num_classes = len(classes)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    # Initialize a square matrix with zeros
    matrix = [[0] * num_classes for _ in range(num_classes)]
    
    # Fill the matrix
    for actual, predicted in zip(y_true, y_pred):
        row = class_to_idx[actual]
        col = class_to_idx[predicted]
        matrix[row][col] += 1
    
    matrix = np.array(matrix)
    return matrix#, classes

def grandsums(a):
    # 1. Calculate sums
    row_sums = a.sum(axis=1, keepdims=True)  # Returns [[3], [7]]
    col_sums = a.sum(axis=0)                 # Returns [4, 6]
    total_sum = a.sum()                      # Returns 10
    
    # 2. Add row sums as a new column (making it 2x3)
    temp_matrix = np.hstack([a, row_sums])
    
    # 3. Create the bottom row (column sums + grand total)
    bottom_row = np.append(col_sums, total_sum)
    
    # 4. Stack them together (making it 3x3)
    result = np.vstack([temp_matrix, bottom_row])
    return result    

def get_SR_confusion_matrix(returns, y_pred):
    y_true = np.sign(returns+0.000001)/2+0.5
    # Find unique classes and map them to indices
    classes = sorted(list(set(y_true) | set(y_pred)))
    num_classes = len(classes)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    # Initialize a square matrix with zeros
    matrix_n = [[0] * num_classes for _ in range(num_classes)]
    matrix_sum = [[0] * num_classes for _ in range(num_classes)]
    matrix_sum_sqr = [[0] * num_classes for _ in range(num_classes)]
    
    # Fill the matrix
    for actual, predicted, r in zip(y_true, y_pred, returns):
        row = class_to_idx[actual]
        col = class_to_idx[predicted]
        matrix_n[row][col] += 1
        matrix_sum[row][col] += r
        matrix_sum_sqr[row][col] += r**2
      
    matrix_n = grandsums(np.array(matrix_n))
    matrix_sum = grandsums(np.array(matrix_sum))
    matrix_sum_sqr = grandsums(np.array(matrix_sum_sqr))
    
    means = matrix_sum/matrix_n
    matrix_SR = means / np.sqrt(matrix_sum_sqr/matrix_n - means**2)
    return matrix_SR#, classes
   
def sample_rebalance(X_train, y_train):
    Xy_train = pd.DataFrame(y_train).join(X_train)
    df_state0 = Xy_train[y_train == 0]
    df_state1 = Xy_train[y_train == 1]
    # Resample the minority class to match the majority class size, then combine and shuffle
    if len(df_state0)<len(df_state1):
        df_state0_oversampled = df_state0.sample(len(df_state1), replace=True, random_state=42)
        df_balanced = pd.concat([df_state1, df_state0_oversampled], axis=0).sample(frac=1, random_state=42)
    else:
        df_state1_oversampled = df_state1.sample(len(df_state0), replace=True, random_state=42)
        df_balanced = pd.concat([df_state0, df_state1_oversampled], axis=0).sample(frac=1, random_state=42)       
    return (df_balanced.iloc[:,1:], pd.Series(df_balanced.iloc[:,0]))

def finetune_svm(X_train, y_train, kernel='linear'):
    # function for hyperparameter optimization in svm
    # Define your grid
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]}
    
    # Build and fit
    search = HalvingGridSearchCV(svm.SVC(kernel=kernel), param_grid, cv=3, factor=2)
    search.fit(X_train, y_train)
    
    return search.best_params_

def finetune_lstm(X_train, y_train):
    import tensorflow as tf
    from scikeras.wrappers import KerasClassifier
    from sklearn.model_selection import RandomizedSearchCV
    
    def create_lstm_standard(meta, units=32, dropout=0.2):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(meta["n_features_in_"],)),
            tf.keras.layers.Reshape((1, meta["n_features_in_"])),
            tf.keras.layers.LSTM(units, activation='tanh', dropout=dropout),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    lstm_wrapper = KerasClassifier(model=create_lstm_standard, verbose=0)
    pipeline = make_pipeline(StandardScaler(), lstm_wrapper)
    
    # Define discrete values for Scikit-Learn to randomly sample from
    param_distributions = {
        'kerasclassifier__model__units': [16, 32, 64, 128],
        'kerasclassifier__model__dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'kerasclassifier__epochs': [10, 20, 30],
        'kerasclassifier__batch_size': [16, 32, 64]
    }
    
    random_search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_distributions, 
        n_iter=10, 
        cv=3, 
        random_state=0
    )
    random_search.fit(X_train, y_train)
    return random_search.best_params_

def finetune_svm_skopt(X_train, y_train, kernel='linear'):
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical
    
    # Define search spaces instead of fixed lists
    search_spaces = {
        'C': Real(1e-3, 1e3, prior='log-uniform'),
        'gamma': Real(1e-4, 1e1, prior='log-uniform'),
        'kernel': Categorical(['linear', 'rbf', 'poly'])
    }
    
    # n_iter controls the budget (number of parameter sets to try)
    search = BayesSearchCV(svm.SVC(), search_spaces, n_iter=32, cv=3)
    search.fit(X_train, y_train)

    return search.best_params_  

def finetune_lstm_skopt(X_train, y_train, kernel='linear'):
    import tensorflow as tf
    from scikeras.wrappers import KerasClassifier    
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    
    # Model function must accept hyperparameters as arguments
    def create_lstm_skopt(meta, units=32, dropout=0.2):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(meta["n_features_in_"],)),
            tf.keras.layers.Reshape((1, meta["n_features_in_"])),
            tf.keras.layers.LSTM(units, activation='tanh', dropout=dropout),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    lstm_wrapper = KerasClassifier(model=create_lstm_skopt, verbose=0)
    pipeline = make_pipeline(StandardScaler(), lstm_wrapper)
    
    # Route parameters using pipeline prefixes
    search_space = {
        'kerasclassifier__model__units': Integer(16, 128),
        'kerasclassifier__model__dropout': Real(0.0, 0.5),
        'kerasclassifier__epochs': Integer(10, 30),
        'kerasclassifier__batch_size': [16, 32, 64]
    }
    
    bayes_search = BayesSearchCV(pipeline, search_space, n_iter=10, cv=3)
    bayes_search.fit(X_train, y_train)
    return bayes_search.best_params_ 

def finetune_svm_optuna(X_train, y_train, kernel='linear'):
    import optuna
    from sklearn.model_selection import cross_val_score
    
    def objective(trial):
        # 1. Define the search space
        # log=True is used for C and gamma as they vary across orders of magnitude
        c_param = trial.suggest_float("C", 1e-3, 1e3, log=True)
        gamma_param = trial.suggest_float("gamma", 1e-4, 1e1, log=True)
        kernel_param = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
    
        # 2. Build the pipeline with trial parameters
        model = make_pipeline(
            StandardScaler(), 
            svm.SVC(C=c_param, gamma=gamma_param, kernel=kernel_param, random_state=0)
        )
    
        # 3. Use Cross-Validation to evaluate (Statistically robust)
        score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3)
        accuracy = score.mean()
        
        return accuracy
    
    # 4. Run the optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50) # 50 trials is usually enough for SVM
    
    return study.best_params_  

def finetune_lstm_optuna(X_train, y_train, kernel='linear'):
    import tensorflow as tf
    from scikeras.wrappers import KerasClassifier    
    import optuna
    from sklearn.model_selection import cross_val_score
    
    def objective(trial):
        # 1. Define hyperparameters to sample
        units = trial.suggest_int("units", 16, 128, step=16)
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        
        # 2. Define model using trial parameters
        def create_lstm(meta):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(meta["n_features_in_"],)),
                tf.keras.layers.Reshape((1, meta["n_features_in_"])),
                tf.keras.layers.LSTM(units, activation='tanh', dropout=dropout),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            return model
    
        # 3. Create pipeline
        lstm_wrapper = KerasClassifier(model=create_lstm, epochs=15, batch_size=32, verbose=0)
        pipeline = make_pipeline(StandardScaler(), lstm_wrapper)
        
        # 4. Evaluate using Cross-Validation
        scores = cross_val_score(pipeline, X_train, y_train, cv=3, n_jobs=1)
        return scores.mean()
    
    # Run the optimization study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    return study.best_params_  