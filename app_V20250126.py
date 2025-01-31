# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:31:05 2025

@author: Edgar David

This script is used for generating an app for regression and classification
task
"""

###############################################################################
# Load libraries

# App
import streamlit as st
from streamlit_option_menu import option_menu
import altair as alt

# Utils
import pandas as pd
import pickle as pkl
import numpy as np
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import time
import altair as alt


# Models

# UTils
import pickle

# Format of numbers
print('Libraries loaded')

###############################################################################
# Make a dictionary of categorical features
dictionary_categorical_features = {'sex (1 = female, 2=male)' : {'Male' : 2,
                                                                 'Female' : 1},
                                   'prior_abdominal_surgery' :  {'Yes' : 1,
                                                                 'No' : 0},
                                   'hypertension' : {'Yes' : 1,
                                                     'No' : 0},
                                   'hyperlipidemia' : {'Yes' : 1,
                                                       'No' : 0},
                                   'depression' :  {'Yes' : 1,
                                                    'No' : 0},
                                   'DMII_preoperative' : {'Yes' : 1,
                                                          'No' : 0},
                                   'antidiab_drug_preop_Oral_anticogulation' : {'Yes' : 1,
                                                                                'No' : 0},
                                   'antidiab_drug_preop_Insulin' : {'Yes' : 1,
                                                                    'No' : 0},
                                   'osas_preoperative' : {'Yes' : 1,
                                                          'No' : 0},
                                   'surgery' : {'1' : 1,
                                                '2' : 2,
                                                '5' : 5},
                                   'normal_dmII_pattern' : {'Yes' : 1,
                                                            'No' : 0},
                                   'months' :  {'Preoperative' : 0,
                                                   '3 Months' : 1,
                                                   '6 Months' : 2,
                                                   '12 Months' : 3,
                                                   '18 Months' : 4,
                                                   '2 Years' : 5,
                                                   '3 Years' : 6,
                                                   '4 Years' : 7,
                                                   '5 Years' : 8},
                                   'time_step' : {'BMI before surgery' : 0,
                                                  'bmi3' : 1,
                                                  'bmi6' : 2,
                                                  'bmi12' : 3,
                                                  'bmi18' : 4,
                                                  'bmi2y' : 5,
                                                  'bmi3y' : 6,
                                                  'bmi4y' : 7,
                                                  'bmi5y' : 8}
                                   }
##############################################################################
# Section when the app initialize and load the required information
@st.cache_data() # We use this decorator so this initialization doesn't run every time the user change into the page
def initialize_app():
    # Load Regression Model
    with open('Regression_Model.sav' , 'rb') as export_model:
        regression_model = pickle.load(export_model) 
    # Load Classification Model
    with open('Classification_Model.sav' , 'rb') as export_model:
        classification_model = pickle.load(export_model) 

    print('App Initialized correctly!')
    
    return regression_model , classification_model


###############################################################################
def create_animated_evolution_chart(df_final, clf_model, predictions_df, threshold_values=None):
    """
    Create an animated evolution chart showing BMI and DM status over time.
    
    Parameters:
    -----------
    df_final : pd.DataFrame
        DataFrame containing patient data and predictions
    clf_model : sklearn model
        Classification model
    predictions_df : pd.DataFrame
        DataFrame with predictions
    threshold_values : pd.DataFrame, optional
        DataFrame containing BMI thresholds for each time point
    """
    # Define time points and their labels in the correct order
    time_map = {
        'Pre': 0,
        '3m': 1,
        '6m': 2,
        '12m': 3,
        '18m': 4,
        '2y': 5,
        '3y': 6,
        '4y': 7,
        '5y': 8
    }
    time_points = list(time_map.keys())
    
    # Get column names in correct order
    bmi_columns = ['BMI before surgery', 'bmi3', 'bmi6', 'bmi12', 'bmi18', 'bmi2y', 'bmi3y', 'bmi4y', 'bmi5y']
    dm_columns = ['DMII_preoperative', 'dm3m', 'dm6m', 'dm12m', 'dm18m', 'dm2y', 'dm3y', 'dm4y', 'dm5y']
    
    # Create chart placeholder
    chart_placeholder = st.empty()
    
    # Calculate y-axis domain for BMI
    bmi_values = df_final[bmi_columns].iloc[0].values
    ci_lower_values = df_final[[f'{col}_ci_lower' for col in bmi_columns]].iloc[0].values
    ci_upper_values = df_final[[f'{col}_ci_upper' for col in bmi_columns]].iloc[0].values
    
    # Update BMI range to include thresholds if they exist
    if threshold_values is not None:
        threshold_min = threshold_values['BMI Threshold'].min()
        threshold_max = threshold_values['BMI Threshold'].max()
        bmi_min = min(np.min(ci_lower_values) * 0.9, threshold_min * 0.9, 25 * 0.9)
        bmi_max = max(np.max(ci_upper_values) * 1.1, threshold_max * 1.1, 25 * 1.1)
    else:
        bmi_min = min(np.min(ci_lower_values) * 0.9, 25 * 0.9)
        bmi_max = max(np.max(ci_upper_values) * 1.1, 25 * 1.1)
    
    # Animation loop
    for i in range(1, len(time_points) + 1):
        current_times = time_points[:i]
        dm_status = df_final[dm_columns].iloc[0, :i].values
        
        # Create data for the main BMI line
        current_bmi_data = pd.DataFrame({
            'Time': current_times,
            'BMI': bmi_values[:i],
            'CI_lower': ci_lower_values[:i],
            'CI_upper': ci_upper_values[:i],
            'Color': ['blue' if dm == 1 else 'blue' for dm in dm_status]
        })
        
        # Create data for threshold line if thresholds exist
        if threshold_values is not None:
            threshold_data = threshold_values[threshold_values['Time Point'].isin(current_times)].copy()
            threshold_data['Time'] = threshold_data['Time Point']
        
        # Create confidence interval area
        ci_area = alt.Chart(current_bmi_data).mark_area(
            opacity=0.2
        ).encode(
            x=alt.X('Time:N', sort=list(time_map.keys())),
            y=alt.Y('CI_upper:Q', title='BMI Value'),
            y2=alt.Y2('CI_lower:Q'),
            color=alt.Color(
                'Color:N',
                scale=None
            )
        )
        
        # Create BMI line
        bmi_line = alt.Chart(current_bmi_data).mark_line(
            strokeWidth=2,
            point=True
        ).encode(
            x=alt.X('Time:N', sort=list(time_map.keys())),
            y=alt.Y('BMI:Q',
                    scale=alt.Scale(domain=[bmi_min, bmi_max])),
            color=alt.Color(
                'Color:N',
                scale=None
            ),
            tooltip=['Time:N', 
                    alt.Tooltip('BMI:Q', format='.1f'),
                    alt.Tooltip('CI_lower:Q', title='CI Lower', format='.1f'),
                    alt.Tooltip('CI_upper:Q', title='CI Upper', format='.1f')]
        )
        
        # BMI=25 threshold line
        health_threshold_data = pd.DataFrame({
            'Time': current_times,
            'health_threshold': [25] * len(current_times)
        })
        
        health_threshold_line = alt.Chart(health_threshold_data).mark_line(
            strokeWidth=4,
            strokeDash=[4, 4],
            color='green'
        ).encode(
            x=alt.X('Time:N', sort=list(time_map.keys())),
            y=alt.Y('health_threshold:Q',
                    scale=alt.Scale(domain=[bmi_min, bmi_max]))
        )
        
        # Layer charts
        layers = [ci_area, bmi_line, health_threshold_line]
        
        # Add DM remission threshold line if thresholds exist
        if threshold_values is not None and not threshold_data.empty:
            threshold_line = alt.Chart(threshold_data).mark_line(
                strokeWidth=4,
                strokeDash=[4, 4],
                color='red'
            ).encode(
                x=alt.X('Time:N', sort=list(time_map.keys())),
                y=alt.Y('BMI Threshold:Q',
                        scale=alt.Scale(domain=[bmi_min, bmi_max])),
                tooltip=[alt.Tooltip('BMI Threshold:Q', title='BMI Threshold', format='.1f')]
            )
            layers.append(threshold_line)
        
        # Combine all layers
        combined_chart = alt.layer(*layers).properties(
            width=700,
            height=400,
            title=f'BMI Evolution Over Time (Step {i}/{len(time_points)})'
        )
        
        # Update the chart
        chart_placeholder.altair_chart(combined_chart)
        
        # Animation delay
        time.sleep(0.2)
    
    return combined_chart

###############################################################################
def find_bmi_thresholds(clf_model, df_final, time_points=['Pre', '3m', '6m', '12m', '18m', '2y', '3y', '4y']):
    """
    Find BMI thresholds at each time point that predict diabetes remission in the next period.
    
    Parameters:
    -----------
    clf_model : sklearn model
        The trained classification model
    df_final : pd.DataFrame
        DataFrame containing all patient information and predictions
    time_points : list
        List of time points to analyze
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing BMI thresholds for each time point
    """
    thresholds = []
    bmi_range = np.arange(18, 50, 0.5)  # Test BMI values from 20 to 50
    
    # Time step mapping
    time_map = {
        'Pre': ('BMI before surgery', 'dm3m', 0, 'DMII_preoperative'),
        '3m': ('bmi3', 'dm6m', 1, 'dm3m'),
        '6m': ('bmi6', 'dm12m', 2, 'dm6m'),
        '12m': ('bmi12', 'dm18m', 3, 'dm12m'),
        '18m': ('bmi18', 'dm2y', 4, 'dm18m'),
        '2y': ('bmi2y', 'dm3y', 5, 'dm2y'),
        '3y': ('bmi3y', 'dm4y', 6, 'dm3y'),
        '4y': ('bmi4y', 'dm5y', 7, 'dm4y')
    }
    
    for time_point in time_points:
        bmi_col, next_dm_col, time_step, dm_current = time_map[time_point]
        
        # Test different BMI values
        threshold_found = False
        threshold_bmi = None
        
        for test_bmi in bmi_range:
            # Create test data
            test_data = df_final.copy()
            test_data[bmi_col] = test_bmi
            
            # Prepare features for model
            features = test_data.copy()
            # Set DM(t) and time_step for the model
            features['DM(t)'] = features[dm_current]
            features['time_step'] = time_step
            
            # Select only the features needed by the model
            features = features[clf_model.feature_names_in_].copy()
            
            # Make prediction
            pred_prob = clf_model.predict_proba(features)[0][1]
            
            # If probability of diabetes is less than 0.5 (predicting remission)
            if pred_prob < 0.5 and not threshold_found:
                threshold_bmi = test_bmi
                threshold_found = True
                break
        
        thresholds.append({
            'Time Point': time_point,
            'BMI Threshold': threshold_bmi if threshold_bmi is not None else "No threshold found",
            'Next Time Point': next_dm_col,
            'Probability at Threshold': clf_model.predict_proba(features)[0][1] if threshold_bmi is not None else None
        })
    
    return pd.DataFrame(thresholds)

###############################################################################
def display_threshold_analysis(threshold_df):
    """
    Display the threshold analysis results with a nice format
    """
    st.subheader("BMI Thresholds for Diabetes Remission")
    st.markdown("""
    The table below shows the BMI thresholds at each time point that predict
    diabetes remission in the next period. If a patient's BMI is below these
    thresholds, they are more likely to achieve diabetes remission in the
    following period.
    
    :blue[Predicted BMI Progression] is shown in blue. :green[Healthy BMI (<25)] is highlighted in green
    :red[Minimum BMI Threshold that predict diabetes remission in next period] is shown in red.
    """)
    
    # Format the dataframe for display
    styled_df = threshold_df.style.format({
        'BMI Threshold': lambda x: f'{x:.1f}' if isinstance(x, (int, float)) else x,
        'Probability at Threshold': lambda x: f'{x:.1%}' if isinstance(x, (int, float)) else '-'
    })
    
    st.dataframe(styled_df)
###############################################################################

# Parser input information
def parser_user_input(dataframe_input , reg_model , clf_model):
    
    
    ##########################################################################
    # Regression part
    
    # Prediction
    predictions_df = pd.DataFrame()
    
    # Encode categorical features
    for i in dictionary_categorical_features.keys():
        if i in dataframe_input.columns:
            dataframe_input[i] = dataframe_input[i].map(dictionary_categorical_features[i])
    # Create iterative timesteps
    for i in dictionary_categorical_features['time_step'].values():
        aux = dataframe_input.copy()
        aux['time_step'] = i
        predictions_df = pd.concat([predictions_df ,
                                 aux] , axis = 0)
    # Convert patiente to one row
    iterations = [['BMI before surgery' , 'bmi3'],
              ['bmi3' , 'bmi6'],
              ['bmi6' , 'bmi12'],
              ['bmi12' , 'bmi18'],
              ['bmi18' , 'bmi2y'],
              ['bmi2y' , 'bmi3y'],
              ['bmi3y' , 'bmi4y'],
              ['bmi4y' , 'bmi5y']]
    target_columns = ['BMI before surgery',
                      'bmi3',
                      'bmi6',
                      'bmi12',
                      'bmi18',
                      'bmi2y',
                      'bmi3y',
                      'bmi4y',
                      'bmi5y']
    # Create a dataframe with X inputs, and BMI pre to put the predictions
    df = predictions_df.head(1)
    df[target_columns] = np.nan # Fill all future predictions with nan
    df['BMI before surgery'] = df['BMI(t)'] # Put the first BMI(t) as bmi pre
    # Pre-compute fixed data
    # Placeholder for the modified dataset
    df_classification = df.drop(columns = ['BMI(t)' , 'time_step']).copy()
    
    target_columns_set = set(target_columns)  # For faster lookup
    time_step_map = {col: dictionary_categorical_features['time_step'][col] for col in target_columns}
    
    # Repeat until no NaN values remain in the BMI columns
    counter = 1
    while df_classification[target_columns].isna().any().any():
        # Collect rows for batch processing
        predictions = []
        rows_to_update = []
    
        for row_index in range(df_classification.shape[0]):
            row = df_classification.loc[row_index]
    
            for bmi_t, bmi_t_plus1 in iterations:
                # Check if BMI(t) is not NaN and BMI(t+1) is NaN
                if pd.notna(row[bmi_t]) and pd.isna(row[bmi_t_plus1]):
                    # Create feature set for prediction
                    time_step = time_step_map[bmi_t]
                    aux_x = pd.concat([
                        row.drop(labels=target_columns),  # X vector (excluding BMI columns)
                        pd.Series({f'BMI(t)': row[bmi_t], 'time_step': time_step})
                    ]).to_frame().T
    
                    aux_x = aux_x[reg_model.feature_names_in_.tolist()]  # Ensure correct column order
                    predictions.append(aux_x)
                    rows_to_update.append((row_index, bmi_t_plus1))
    
        # Predict in batches
        if predictions:
            predictions_df_2 = pd.concat(predictions, ignore_index=True)
            predicted_values = reg_model.predict(predictions_df_2)
    
            # Update the DataFrame with predictions
            for (row_index, target_column), prediction in zip(rows_to_update, predicted_values):
                df_classification.at[row_index, target_column] = prediction
    
        counter += 1
        
    ###########################################################################
    # Classification part
    iterations = [['DMII_preoperative' , 'dm3m'],
              ['dm3m' , 'dm6m'],
              ['dm6m' , 'dm12m'],
              ['dm12m' , 'dm18m'],
              ['dm18m' , 'dm2y'],
              ['dm2y' , 'dm3y'],
              ['dm3y' , 'dm4y'],
              ['dm4y' , 'dm5y']]
    target_columns = ['DMII_preoperative',
                      'dm3m',
                      'dm6m',
                      'dm12m',
                      'dm18m',
                      'dm2y',
                      'dm3y',
                      'dm4y',
                      'dm5y']
    target_time_steps = {'DMII_preoperative' : 0,
                         'dm3m' : 1,
                         'dm6m' : 2,
                         'dm12m' : 3,
                         'dm18m' : 4,
                         'dm2y' : 5,
                         'dm3y' : 6,
                         'dm4y' : 7,
                         'dm5y' : 8}
    target_classification = ['DM(t+1)']
    numeric_columns = ['age_years' , 'BMI before surgery',
                      'bmi3',
                      'bmi6',
                      'bmi12',
                      'bmi18',
                      'bmi2y',
                      'bmi3y',
                      'bmi4y',
                      'bmi5y']
    probas_columns = ['DMII_preoperative_prob',
                      'dm3m_prob',
                      'dm6m_prob',
                      'dm12m_prob',
                      'dm18m_prob',
                      'dm2y_prob',
                      'dm3y_prob',
                      'dm4y_prob',
                      'dm5y_prob']
    # Placeholder for final vector
    df_final = df_classification.copy()
    df_final[target_columns] = np.nan # Fill all future predictions with nan
    df_final['DMII_preoperative'] = df_classification['DMII_preoperative']
    df_final[probas_columns] = np.nan
    df_final[probas_columns[0]] = 1 if df_final[target_columns[0]].values[0] == 1 else 0
    # Identify the columns and their corresponding time steps for faster access
    time_step_map = {col: target_time_steps[col] for col in target_columns}
    
    # Repeat until no NaN values remain in the BMI columns
    counter = 1
    while df_final[target_columns[1:]].isna().any().any():
        # Collect rows for batch processing
        predictions = []
        rows_to_update = []
    
        for row_index in range(df_final.shape[0]):
            #print(f"Row:{row_index}")
            row = df_final.loc[row_index]
    
            for dm_t, dm_t_plus1 in iterations:
                # Check if DM(t) is not NaN and DM(t+1) is NaN
                if pd.notna(row[dm_t]) and pd.isna(row[dm_t_plus1]):
                    # Create feature set for prediction
                    time_step = time_step_map[dm_t]
                    aux_x = pd.concat([
                        row.drop(labels=target_columns),  # X vector (excluding BMI columns)
                        pd.Series({f'DM(t)': row[dm_t], 'time_step': time_step})
                    ]).to_frame().T
    
                    aux_x = aux_x[clf_model.feature_names_in_.tolist()]  # Ensure correct column order
                    predictions.append(aux_x)
                    rows_to_update.append((row_index, dm_t_plus1 , probas_columns[iterations.index([dm_t , dm_t_plus1]) + 1]))
    
        # Predict in batches
        if predictions:
            predictions_df = pd.concat(predictions, ignore_index=True)
            predicted_values = clf_model.predict(predictions_df)
            predicted_probas = clf_model.predict_proba(predictions_df)[: , 1]
    
            # Update the DataFrame with predictions
            for (row_index, target_column , prob_column), prediction , probas in zip(rows_to_update, predicted_values , predicted_probas):
                df_final.at[row_index, target_column] = prediction
                df_final.at[row_index, prob_column] = probas
                
        if counter > 9:
            break
        else:
            counter += 1
            
    # Confindent interval
    # Add confidence intervals (95% confidence level)
    confidence_level = 0.95
    z_score = 1.96  # 95% confidence level

    # Calculate confidence intervals for BMI predictions
    bmi_columns = ['BMI before surgery', 'bmi3', 'bmi6', 'bmi12', 'bmi18', 'bmi2y', 'bmi3y', 'bmi4y', 'bmi5y']
    
    # Initialize confidence interval columns
    for col in bmi_columns:
        df_final[f'{col}_ci_lower'] = df_final[col]
        df_final[f'{col}_ci_upper'] = df_final[col]
        
        if col != 'BMI before surgery':  # Skip the initial BMI which is known
            std_error = df_final[col].std() if not pd.isna(df_final[col].std()) else df_final[col].mean() * 0.1
            df_final[f'{col}_ci_lower'] = df_final[col] - (z_score * std_error)
            df_final[f'{col}_ci_upper'] = df_final[col] + (z_score * std_error)
    # Plot
    
    # Create a summary dataframe for display
    summary_df = pd.DataFrame({
        'Time': ['Pre', '3m', '6m', '12m', '18m', '2y', '3y', '4y', '5y'],
        'BMI': df_final[bmi_columns].iloc[0].values,
        'BMI CI Lower': df_final[[f'{col}_ci_lower' for col in bmi_columns]].iloc[0].values,
        'BMI CI Upper': df_final[[f'{col}_ci_upper' for col in bmi_columns]].iloc[0].values,
        'DM Status': df_final[['DMII_preoperative', 'dm3m', 'dm6m', 'dm12m', 'dm18m', 'dm2y', 'dm3y', 'dm4y', 'dm5y']].iloc[0].values,
        'DM Likelihood (%)': (df_final[['DMII_preoperative_prob', 'dm3m_prob', 'dm6m_prob', 'dm12m_prob', 'dm18m_prob', 'dm2y_prob', 'dm3y_prob', 'dm4y_prob', 'dm5y_prob']].iloc[0].values * 100).round(2)
    })
    
    summary_df['DM Status'] = np.select(condlist = [summary_df[ 'DM Status'] == 1],
                                        choicelist = ['Diabetes'],
                                        default = 'Healed')
    
    # Display the summary dataframe
    st.dataframe(summary_df.style.format({
        'BMI': '{:.1f}',
        'BMI CI Lower': '{:.1f}',
        'BMI CI Upper': '{:.1f}',
        'DM Likelihood (%)': '{:.1f}'
    }))
    
    
    # Calculate thresholds if patient has diabetes
    threshold_df = None
    if df_final['DMII_preoperative'].values[0] == 1:
        threshold_df = find_bmi_thresholds(clf_model, df_final)
        display_threshold_analysis(threshold_df)
        threshold_df['BMI Threshold'] = np.select(condlist = [threshold_df['BMI Threshold'] == "No threshold found"],
                                                  choicelist = [0.0],
                                                  default = threshold_df['BMI Threshold'])
    
    # Create the chart with thresholds
    chart = create_animated_evolution_chart(df_final, clf_model, predictions_df, threshold_df)
    
    return predictions

###############################################################################
# Page configuration
st.set_page_config(
    page_title="AL Prediction App"
)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Initialize app
reg_model , clf_model = initialize_app()

# Option Menu configuration
with st.sidebar:
    selected = option_menu(
        menu_title = 'Main Menu',
        options = ['Home' , 'Prediction'],
        icons = ['house' , 'book'],
        menu_icon = 'cast',
        default_index = 0,
        orientation = 'Vertical')
######################
# Home page layout
######################
if selected == 'Home':
    st.title('DM and BMI Prediction App')
    st.markdown("""
    This app contains 2 sections which you can access from the horizontal menu above.\n
    The sections are:\n
    Home: The main page of the app.\n
    **Prediction:** On this section you can select the patients information and
    the models iterate over all posible anastomotic configuration and surgeon experience for suggesting
    the best option.\n
    \n
    \n
    \n
    **Disclaimer:** This application and its results are only approved for research purposes.
    """)
###############################################################################
# Prediction page layout
if selected == 'Prediction':
    st.title('Prediction Section')
    st.subheader("Description")
    st.subheader("To predict BMI and DM Curve, you need to follow the steps below:")
    st.markdown("""
    1. Enter clinical parameters of patient on the left side bar.
    2. Press the "Predict" button and wait for the result.
    \n
    \n
    \n
    **Disclaimer:** This application and its results are only approved for research purposes.
    """)
    st.markdown("""
    This model predicts the probabilities DM on each time step.
    """)
    # Sidebar layout
    st.sidebar.title("Patiens Info")
    st.sidebar.subheader("Please choose parameters")
    
    # Input features
    age = st.sidebar.number_input("Age(Years):" , step = 1.0)
    bmi_pre = st.sidebar.number_input("Preoperative BMI:" , step = 0.5)
    sex = st.sidebar.selectbox('Sex', tuple(dictionary_categorical_features['sex (1 = female, 2=male)'].keys()))
    prior_abdominal_surgery = st.sidebar.selectbox('Prior Abdominal Surgery', tuple(dictionary_categorical_features['prior_abdominal_surgery'].keys()))
    hypertension = st.sidebar.selectbox('Hypertension', tuple(dictionary_categorical_features['hypertension'].keys()))
    hyperlipidemia = st.sidebar.selectbox('Hyperlipidemia', tuple(dictionary_categorical_features['hyperlipidemia'].keys()))
    depression = st.sidebar.selectbox('Depression', tuple(dictionary_categorical_features['depression'].keys()))
    DMII_preoperative = st.sidebar.selectbox('DMII_preoperative', tuple(dictionary_categorical_features['DMII_preoperative'].keys()))
    antidiab_drug_preop_Oral_anticogulation = st.sidebar.selectbox('antidiab_drug_preop_Oral_anticogulation', tuple(dictionary_categorical_features['antidiab_drug_preop_Oral_anticogulation'].keys()))
    antidiab_drug_preop_Insulin = st.sidebar.selectbox('antidiab_drug_preop_Insulin', tuple(dictionary_categorical_features['antidiab_drug_preop_Insulin'].keys()))
    osas_preoperative = st.sidebar.selectbox('osas_preoperative', tuple(dictionary_categorical_features['osas_preoperative'].keys()))
    surgery = st.sidebar.selectbox('Surgery', tuple(dictionary_categorical_features['surgery'].keys()))
    normal_dmII_pattern = st.sidebar.selectbox('normal_dmII_pattern', tuple(dictionary_categorical_features['normal_dmII_pattern'].keys()))
    
    dataframe_input = pd.DataFrame({'age_years' : [age],
                                    'BMI(t)' : [bmi_pre],
                                    'sex (1 = female, 2=male)' : [sex],
                                    'prior_abdominal_surgery' : [prior_abdominal_surgery],
                                    'hypertension' : [hypertension],
                                    'hyperlipidemia' : [hyperlipidemia],
                                    'depression' : [depression],
                                    'DMII_preoperative' : [DMII_preoperative],
                                    'antidiab_drug_preop_Oral_anticogulation' : [antidiab_drug_preop_Oral_anticogulation],
                                    'antidiab_drug_preop_Insulin' : [antidiab_drug_preop_Insulin],
                                    'osas_preoperative' : [osas_preoperative],
                                    'surgery' : [surgery],
                                    'normal_dmII_pattern' : [normal_dmII_pattern]})
    # Parser input and make predictions
    predict_button = st.button('Predict')
    if predict_button:
        predictions = parser_user_input(dataframe_input , reg_model , clf_model)
