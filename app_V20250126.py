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
def create_split_evolution_charts(df_final):
    # Define time points and their labels
    time_points = ['Pre', '3m', '6m', '12m', '18m', '2y', '3y', '4y', '5y']
    
    # Create long format dataframe for BMI values
    bmi_columns = ['BMI before surgery', 'bmi3', 'bmi6', 'bmi12', 'bmi18', 'bmi2y', 'bmi3y', 'bmi4y', 'bmi5y']
    dm_columns = ['DMII_preoperative', 'dm3m', 'dm6m', 'dm12m', 'dm18m', 'dm2y', 'dm3y', 'dm4y', 'dm5y']
    
    # Prepare data in long format
    bmi_data = pd.DataFrame({
        'Time': time_points,
        'BMI': df_final[bmi_columns].iloc[0].values,
        'Type': 'BMI',
        'TimeIndex': range(len(time_points))
    })
    
    dm_data = pd.DataFrame({
        'Time': time_points,
        'DM': df_final[dm_columns].iloc[0].values,
        'Type': 'DM',
        'TimeIndex': range(len(time_points))
    })

    # Create time ordering
    time_order = alt.EncodingSortField(field='TimeIndex', order='ascending')

    # Create placeholders for both charts
    bmi_chart_placeholder = st.empty()
    dm_chart_placeholder = st.empty()
    total_steps = len(time_points)

    for i in range(1, total_steps + 1):
        # Current subset of data
        current_bmi = bmi_data.head(i)
        current_dm = dm_data.head(i)
        
        # Create BMI line chart with ordered x-axis
        bmi_chart = alt.Chart(current_bmi).mark_line(
            strokeWidth=3,
            color='blue',
            point=True
        ).encode(
            x=alt.X('Time:N', 
                    sort=time_order,  # Use the time ordering
                    title='Time'),
            y=alt.Y('BMI:Q',
                    scale=alt.Scale(domain=[0, bmi_data['BMI'].max() * 1.1]),
                    axis=alt.Axis(title='BMI Value')),
            tooltip=['Time:N', alt.Tooltip('BMI:Q', format='.1f')]
        )

        # Add BMI annotation
        if i > 0:
            last_bmi_point = current_bmi.iloc[-1]
            bmi_annotation_text = (
                f"Latest BMI at {last_bmi_point['Time']}:\n"
                f"BMI: {last_bmi_point['BMI']:.1f}"
            )
            
            bmi_annotation_data = pd.DataFrame({
                'Time': [time_points[0]],
                'BMI': [bmi_data['BMI'].max()],
                'text': [bmi_annotation_text],
                'TimeIndex': [0]  # Add TimeIndex for consistent ordering
            })
            
            bmi_annotation = alt.Chart(bmi_annotation_data).mark_text(
                align= 'center',
                baseline= 'middle',
                fontSize=14,
                dx=200,
                dy=100,
                color='gray'
            ).encode(
                x=alt.X('Time:N', sort=time_order),  # Use the time ordering
                y=alt.Y('BMI:Q'),
                text=alt.Text('text:N')
            )
            
            final_bmi_chart = (bmi_chart + bmi_annotation).properties(
                width=700,
                height=300,
                title=f'BMI Evolution Over Time (Step {i}/{total_steps})'
            )
        else:
            final_bmi_chart = bmi_chart.properties(
                width=700,
                height=300,
                title=f'BMI Evolution Over Time (Step {i}/{total_steps})'
            )

        # Create DM line chart - Now combining line and points with same color
        base = alt.Chart(current_dm).encode(
            x=alt.X('Time:N', 
                    sort=time_order,
                    title='Time'),
            y=alt.Y('DM:Q',
                    scale=alt.Scale(domain=[-0.5, 1.5]),
                    axis=alt.Axis(
                        title='DM Status',
                        values=[0, 1],
                        labelExpr="datum.value === 0 ? 'No' : 'Yes'"
                    )),
            tooltip=['Time:N', 'DM:Q']
        )

        # Create separate line and point layers with the same color
        dm_line = base.mark_line(strokeWidth=3, color='red')
        dm_points = base.mark_point(size=100, fill='red', color='red')
        
        dm_chart = dm_line + dm_points

        # Add DM annotation
        if i > 0:
            last_dm_point = current_dm.iloc[-1]
            dm_annotation_text = (
                f"Latest DM Status at {last_dm_point['Time']}:\n"
                f"Status: {'Diabettes' if last_dm_point['DM'] == 1 else 'Healed'}"
            )
            
            dm_annotation_data = pd.DataFrame({
                'Time': [time_points[0]],
                'DM': [1],
                'text': [dm_annotation_text],
                'TimeIndex': [0]
            })
            
            dm_annotation = alt.Chart(dm_annotation_data).mark_text(
                align='left',
                baseline='top',
                fontSize=14,
                dx=10,
                dy=10,
                color='gray'
            ).encode(
                x=alt.X('Time:N', sort=time_order),
                y=alt.Y('DM:Q'),
                text=alt.Text('text:N')
            )
            
            final_dm_chart = (dm_chart + dm_annotation).properties(
                width=700,
                height=300,
                title=f'DM Status Evolution Over Time (Step {i}/{total_steps})'
            )
        else:
            final_dm_chart = dm_chart.properties(
                width=700,
                height=300,
                title=f'DM Status Evolution Over Time (Step {i}/{total_steps})'
            )

        # Display both charts
        bmi_chart_placeholder.altair_chart(final_bmi_chart)
        dm_chart_placeholder.altair_chart(final_dm_chart)
        
        # Small delay for animation effect
        time.sleep(0.2)

    return final_bmi_chart, final_dm_chart

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
    # Placeholder for final vector
    df_final = df_classification.copy()
    df_final[target_columns] = np.nan # Fill all future predictions with nan
    df_final['DMII_preoperative'] = df_classification['DMII_preoperative']
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
                    rows_to_update.append((row_index, dm_t_plus1))
    
        # Predict in batches
        if predictions:
            predictions_df = pd.concat(predictions, ignore_index=True)
            predicted_values = clf_model.predict(predictions_df)
    
            # Update the DataFrame with predictions
            for (row_index, target_column), prediction in zip(rows_to_update, predicted_values):
                df_final.at[row_index, target_column] = prediction
        if counter > 9:
            break
        else:
            counter += 1
    
    # Plot
    create_split_evolution_charts(df_final)
    
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