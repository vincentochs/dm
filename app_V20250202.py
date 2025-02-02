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

# UTils
import pickle
from scipy.interpolate import make_interp_spline

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
                                   'surgery' : {'Laparoscopic Sleeve Gastrectomy' : 1,
                                                'Roux-en-Y Bypass' : 2,
                                                'OAGB' : 5},
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

inverse_dictionary = {feature: {v: k for k, v in mapping.items()} 
                      for feature, mapping in dictionary_categorical_features.items()}
# MAE from training notebook for make confident intervals
training_mae = 2.26287611774099

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
    
    # Define time points mapping
    time_map = {
        'Pre': 0, '3m': 3, '6m': 6, '12m': 12,
        '18m': 18, '2y': 24, '3y': 36, '4y': 48, '5y': 60
    }
    time_points = list(time_map.keys())
    time_points_values = list(time_map.values())
    
    # Get BMI column names in correct order
    bmi_columns = ['BMI before surgery', 'bmi3', 'bmi6', 'bmi12', 
                  'bmi18', 'bmi2y', 'bmi3y', 'bmi4y', 'bmi5y']
    
    # Create chart placeholder
    chart_placeholder = st.empty()
    
    # Get all BMI values and confidence intervals
    bmi_values = df_final[bmi_columns].iloc[0].values
    ci_lower_values = df_final[[f'{col}_ci_lower' for col in bmi_columns]].iloc[0].values
    ci_upper_values = df_final[[f'{col}_ci_upper' for col in bmi_columns]].iloc[0].values
    
    # Calculate y-axis domain including thresholds
    if threshold_values is not None:
        threshold_values['BMI Threshold'] = threshold_values['BMI Threshold'].apply(lambda x: 0 if x < 0 else x)
        threshold_min = threshold_values['BMI Threshold'].min()
        threshold_max = threshold_values['BMI Threshold'].max()
        bmi_min = min(np.min(ci_lower_values) * 0.9, threshold_min * 0.9, 25 * 0.9)
        bmi_max = max(np.max(ci_upper_values) * 1.1, threshold_max * 1.1, 25 * 1.1)
    else:
        bmi_min = min(np.min(ci_lower_values) * 0.9, 25 * 0.9)
        bmi_max = max(np.max(ci_upper_values) * 1.1, 25 * 1.1)
    
    # Pre-calculate all smooth curves for animation
    total_steps = 100
    # Create a smoother curve using interpolation
    smooth_years = np.linspace(0, 60, total_steps)  # Ensure interpolation extends to year 5
    smooth_bmi_values = make_interp_spline(time_points_values, bmi_values)(smooth_years)
    smooth_bmi_upper = make_interp_spline(time_points_values, ci_upper_values)(smooth_years)
    smooth_bmi_lower = make_interp_spline(time_points_values, ci_lower_values)(smooth_years)
    smooth_threshold_bmi = make_interp_spline(time_points_values, threshold_values['BMI Threshold'].values.tolist() + [0])(smooth_years)
    smooth_threshold_bmi = np.array(list(map(lambda x: 0 if x < 0 else x , smooth_threshold_bmi)))
    
    #print(smooth_years)
    #print(smooth_bmi_values)
    #print(smooth_bmi_upper)
    #print(smooth_bmi_lower)
    #print(smooth_threshold_bmi)
    
    # Healthy BMI Thin Line
    healthy_bmi_line = pd.DataFrame({
        "Year": smooth_years,
        "BMI": [25] * len(smooth_years),
        "Line": ["Healthy BMI"] * len(smooth_years)
    })
    
    
    st.header("Predicted BMI Progression After Surgery")
    st.write("In the graph, the predicted BMI outcome up to 5 years post-surgery is displayed.")
    st.markdown("""
    <div style="font-size:16px;">
        <span style="color:blue;">Predicted BMI Progression</span> is shown in blue, 
        <span style="color:orange;">its CI</span> is shown in orange, 
        <span style="color:red;">Minimum BMI Threshold that predict diabetes remission in next period</span> is indicated in red, 
        and the threshold to a <span style="color:green;">Healthy BMI (<25)</span> is highlighted in green.
    </div>
    """, unsafe_allow_html=True)

    chart_placeholder = st.empty()  # Placeholder for the chart
    total_steps = len(smooth_years)  # Total steps for smooth animation
    print(f"Total Steps: {total_steps}. Len smooth years:{len(smooth_years)}")
    # Build animated chart
    for i in range(1, total_steps + 1, 1):  # Adjust steps for full range
        # Update chart dynamically
        current_smooth_years = smooth_years[:i]
        current_smooth_bmi_values = smooth_bmi_values[:i]
        current_smooth_bmi_upper = smooth_bmi_upper[:i]
        current_smooth_bmi_lower = smooth_bmi_lower[:i]
        current_smooth_bmi_threshold = smooth_threshold_bmi[:i]
        
        # Create data for dynamic charts
        current_chart_data = pd.DataFrame({
            "Year": np.concatenate([current_smooth_years, current_smooth_years]),
            "BMI": np.concatenate([current_smooth_bmi_upper, current_smooth_bmi_lower]),
            "Line": ["CI"] * len(current_smooth_years) + ["CI"] * len(current_smooth_years)
        })
        current_bmi_chart_data = pd.DataFrame({
            "Year": current_smooth_years,
            "BMI": current_smooth_bmi_values,
            "Line": ["Predicted BMI"] * len(current_smooth_years)
        })
        
        threshold_chart_data = pd.DataFrame({'Year' : current_smooth_years,
                                             'Threshold': current_smooth_bmi_threshold,
                                             'Line' : ['Threshold BMI'] * len(current_smooth_years)})

        # Create Altair charts
        bmi_chart = alt.Chart(current_bmi_chart_data).mark_line(
            strokeWidth=3  # Change size of Predicted BMI line
        ).encode(
            x=alt.X("Year", title="Months After Surgery"),
            y=alt.Y("BMI", title="Body Mass Index (BMI)"),
            color=alt.Color(
                "Line",
                scale=alt.Scale(
                    domain=["Predicted BMI", "CI", "Healthy BMI" , "Threshold BMI"],
                    range=["blue", "orange", "green" , 'red']  # Explicit color mapping
                ),
                legend=alt.Legend(title="Legend")
            ),
            tooltip=["Year", "BMI"]
        )

        bounds_chart = alt.Chart(current_chart_data).mark_line(
            strokeDash=[2, 2],  # Change dash style of MAE lines
            strokeWidth=0.5  # Change size of MAE lines
        ).encode(
            x="Year",
            y="BMI",
            color=alt.Color(
                "Line",
                scale=alt.Scale(
                    domain=["Predicted BMI", "CI", "Healthy BMI" , 'Threshold BMI'],
                    range=["blue", "orange", "green" , 'red']  # Explicit color mapping
                ),
                legend=None
            ),
            tooltip=["Year", "BMI"]
        )

        healthy_line_chart = alt.Chart(healthy_bmi_line[:i]).mark_line(
            strokeWidth=1.5  # Change size of Healthy BMI line
        ).encode(
            x="Year",
            y="BMI",
            color=alt.Color(
                "Line",
                scale=alt.Scale(
                    domain=["Predicted BMI", "CI", "Healthy BMI" , 'Threshold BMI'],
                    range=["blue", "orange", "green" , 'red']  # Explicit color mapping
                ),
                legend=None
            ),
            tooltip=["Year", "BMI"]
        )
            
            
        threshold_line_chart = alt.Chart(threshold_chart_data).mark_line(
            strokeWidth=1.5  # Change size of Healthy BMI line
        ).encode(
            x="Year",
            y="Threshold",
            color=alt.Color(
                "Line",
                scale=alt.Scale(
                    domain=["Predicted BMI", "CI", "Healthy BMI" , 'Threshold BMI'],
                    range=["blue", "orange", "green" , 'red']  # Explicit color mapping
                ),
                legend=None
            ),
            tooltip=["Year", "Threshold"]
        )
            
        # Combine charts and display
        final_chart = (bounds_chart + healthy_line_chart + threshold_line_chart + bmi_chart).properties(
            width=850, height=600
        ).configure_legend(
            orient="bottom",  # Explicitly place legend below the chart
            title=None,
            labelFontSize=12,
            padding=10  # Add spacing between legend and chart
        ).configure_view(
            strokeWidth=1
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )

        chart_placeholder.altair_chart(final_chart)
        time.sleep(0.1)  # Animation delay
    
###############################################################################
def find_bmi_thresholds(clf_model, df_final, time_points=['Pre', '3m', '6m', '12m', '18m', '2y', '3y', '4y']):

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
        '4y': ('bmi4y', 'dm5y', 7, 'dm4y'),
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
            #df_final[f'{col}_ci_lower'] = df_final[col] - (z_score * std_error)
            #df_final[f'{col}_ci_upper'] = df_final[col] + (z_score * std_error)
            df_final[f'{col}_ci_lower'] = df_final[col] - training_mae
            df_final[f'{col}_ci_upper'] = df_final[col] + training_mae
    
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
        threshold_df['BMI Threshold'] = np.select(condlist = [threshold_df['BMI Threshold'] == "No threshold found"],
                                                  choicelist = [0.0],
                                                  default = threshold_df['BMI Threshold'])
    
    # Create the chart with thresholds
    chart = create_animated_evolution_chart(df_final, clf_model, predictions_df, threshold_df)
    
    if df_final['DMII_preoperative'].values[0] == 1:
        # Display Threshold Table
        display_threshold_analysis(threshold_df)
    
    
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
    
    sex = st.sidebar.radio(
        "Select Sex:",
        options = tuple(dictionary_categorical_features['sex (1 = female, 2=male)'].keys()),
    )
    
    # Input: Binary data
    st.sidebar.subheader("Medical Conditions (Yes/No):")
    hypertension = int(st.sidebar.checkbox("Hypertension"))
    hyperlipidemia = int(st.sidebar.checkbox('Hyperlipidemia'))
    depression = int(st.sidebar.checkbox('Depression'))
    DMII_preoperative = int(st.sidebar.checkbox('Diabetes Mellitus Type 2'))
    antidiab_drug_preop_Oral_anticogulation = int(st.sidebar.checkbox('Antidiabetes drug preoperative oral anticogulation'))
    antidiab_drug_preop_Insulin = int(st.sidebar.checkbox('Antidiabetes drug preoperative insulin'))
    prior_abdominal_surgery = int(st.sidebar.checkbox('Prior abdominal surgery'))
    osas_preoperative = int(st.sidebar.checkbox('Obstructive Sleep Apnea Syndrome (OSAS)'))
    normal_dmII_pattern = int(st.sidebar.checkbox('Normal Diabetes Mellitus Type 2 pattern'))
    
    # Surgery Type
    surgery = st.sidebar.radio(
        "Planned Surgery:",
        options = tuple(dictionary_categorical_features['surgery'].keys()),
    )
    
    # Map binary options
    hypertension = inverse_dictionary['hypertension'][hypertension]
    hyperlipidemia = inverse_dictionary['hyperlipidemia'][hyperlipidemia]
    depression = inverse_dictionary['depression'][depression]
    DMII_preoperative = inverse_dictionary['DMII_preoperative'][DMII_preoperative]
    antidiab_drug_preop_Oral_anticogulation = inverse_dictionary['antidiab_drug_preop_Oral_anticogulation'][antidiab_drug_preop_Oral_anticogulation]
    antidiab_drug_preop_Insulin = inverse_dictionary['antidiab_drug_preop_Insulin'][antidiab_drug_preop_Insulin]
    prior_abdominal_surgery = inverse_dictionary['prior_abdominal_surgery'][prior_abdominal_surgery]
    osas_preoperative = inverse_dictionary['osas_preoperative'][osas_preoperative]
    normal_dmII_pattern = inverse_dictionary['normal_dmII_pattern'][normal_dmII_pattern]
    
    # Create dataframe with the input data
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