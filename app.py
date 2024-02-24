import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder


import joblib
def predict(data):
    clf = joblib.load('tree_model_smote.sav')  # Assuming the correct model filename is 'gbc_model.sav'
    return clf.predict(data)

def label_encode_data(input_data):
    label_encoder = LabelEncoder()
    categorical_variables = ['count_floors_pre_eq','land_surface_condition','foundation_type','roof_type','ground_floor_type','other_floor_type',
                             'position','plan_configuration','legal_ownership_status']
    for variable in categorical_variables:
        input_data[variable] = label_encoder.fit_transform(input_data[variable])
    return input_data


# Title
st.title('Earthquake Building Damage Grade Prediction')

# Sidebar with input fields
st.header('Input Variables')

# Collecting input variables
#building_id = st.number_input('Building ID', min_value=0)

#geo_level_1_id = st.selectbox('Geo Level 1 ID', options=list(range(31)))
#geo_level_2_id = st.number_input('Geo Level 2 ID', min_value=0, max_value=1427)
#geo_level_3_id = st.number_input('Geo Level 3 ID', min_value=0, max_value=12567)
age = st.number_input('Age of the Building (Years)', min_value=0,max_value=60)
area_percentage = st.number_input('Area Percentage', min_value=0, max_value=100)
height_percentage = st.number_input('Height Percentage', min_value=0,max_value=10)
count_floors_pre_eq = st.number_input('Number of Floors Pre-Earthquake', min_value=0)

# Collecting input variables
#area_percentage_str = st.text_input('Area Percentage (e.g., 0-5)', value='')
# Extracting numerical value from the input string (assuming it's in the format '0-5')
#area_percentage = float(area_percentage_str.split('-')[0]) if area_percentage_str else None


land_surface_condition = st.selectbox('Land Surface Condition', options=['n', 'o', 't'])
foundation_type = st.selectbox('Foundation Type', options=['h', 'i', 'r', 'u', 'w'])
roof_type = st.selectbox('Roof Type', options=['n', 'q', 'x'])
ground_floor_type = st.selectbox('Ground Floor Type', options=['f', 'm', 'v', 'x', 'z'])
other_floor_type = st.selectbox('Other Floor Type', options=['j', 'q', 's', 'x'])
position = st.selectbox('Position', options=['j', 'o', 's', 't'])
plan_configuration = st.selectbox('Plan Configuration', options=['a', 'c', 'd', 'f', 'm', 'n', 'o', 'q', 's', 'u'])
has_superstructure_adobe_mud = st.selectbox('Has Superstructure Adobe Mud',options=['0','1'])
has_superstructure_mud_mortar_stone = st.selectbox('Has Superstructure Mud Mortar Stone',options=['0','1'])
has_superstructure_stone_flag = st.selectbox('Has Superstructure Stone Flag',options=['0','1'])
has_superstructure_cement_mortar_stone = st.selectbox('Has Superstructure Cement Mortar Stone',options=['0','1'])
has_superstructure_mud_mortar_brick = st.selectbox('Has Superstructure Mud Mortar Brick',options=['0','1'])
has_superstructure_cement_mortar_brick = st.selectbox('Has Superstructure Cement Mortar Brick',options=['0','1'])
has_superstructure_timber = st.selectbox('Has Superstructure Timber',options=['0','1'])
has_superstructure_bamboo = st.selectbox('Has Superstructure Bamboo',options=['0','1'])
has_superstructure_rc_non_engineered = st.selectbox('Has Superstructure RC Non-Engineered',options=['0','1'])
has_superstructure_rc_engineered = st.selectbox('Has Superstructure RC Engineered',options=['0','1'])
has_superstructure_other = st.selectbox('Has Superstructure Other',options=['0','1'])
legal_ownership_status = st.selectbox('Legal Ownership Status', options=['a', 'r', 'v', 'w'])
count_families = st.number_input('Number of Families', min_value=0)
has_secondary_use = st.selectbox('Has Secondary Use',options=['0','1'])
has_secondary_use_agriculture = st.selectbox('Has Secondary Use Agriculture',options=['0','1'])
has_secondary_use_hotel = st.selectbox('Has Secondary Use Hotel',options=['0','1'])
has_secondary_use_rental = st.selectbox('Has Secondary Use Rental',options=['0','1'])
has_secondary_use_institution = st.selectbox('Has Secondary Use Institution',options=['0','1'])
has_secondary_use_school = st.selectbox('Has Secondary Use School',options=['0','1'])
has_secondary_use_industry = st.selectbox('Has Secondary Use Industry',options=['0','1'])
has_secondary_use_health_post = st.selectbox('Has Secondary Use Health Post',options=['0','1'])
has_secondary_use_gov_office = st.selectbox('Has Secondary Use Gov Office',options=['0','1'])
has_secondary_use_use_police = st.selectbox('Has Secondary Use Police',options=['0','1'])
has_secondary_use_other = st.selectbox('Has Secondary Use Other',options=['0','1'])

# Prepare input data
input_data = pd.DataFrame({
    'age': [age],
    'area_percentage': [area_percentage],
    'height_percentage': [height_percentage],
    'count_floors_pre_eq': [count_floors_pre_eq],
    'land_surface_condition': [land_surface_condition],
    'foundation_type': [foundation_type],
    'roof_type': [roof_type],
    'ground_floor_type': [ground_floor_type],
    'other_floor_type': [other_floor_type],
    'position': [position],
    'plan_configuration': [plan_configuration],
    'has_superstructure_adobe_mud': [has_superstructure_adobe_mud],
    'has_superstructure_mud_mortar_stone': [has_superstructure_mud_mortar_stone],
    'has_superstructure_stone_flag': [has_superstructure_stone_flag],
    'has_superstructure_cement_mortar_stone': [has_superstructure_cement_mortar_stone],
    'has_superstructure_mud_mortar_brick': [has_superstructure_mud_mortar_brick],
    'has_superstructure_cement_mortar_brick': [has_superstructure_cement_mortar_brick],
    'has_superstructure_timber': [has_superstructure_timber],
    'has_superstructure_bamboo': [has_superstructure_bamboo],
    'has_superstructure_rc_non_engineered': [has_superstructure_rc_non_engineered],
    'has_superstructure_rc_engineered': [has_superstructure_rc_engineered],
    'has_superstructure_other': [has_superstructure_other],
    'legal_ownership_status': [legal_ownership_status],
    'count_families': [count_families],
    'has_secondary_use': [has_secondary_use],
    'has_secondary_use_agriculture': [has_secondary_use_agriculture],
    'has_secondary_use_hotel': [has_secondary_use_hotel],
    'has_secondary_use_rental': [has_secondary_use_rental],
    'has_secondary_use_institution': [has_secondary_use_institution],
    'has_secondary_use_school': [has_secondary_use_school],
    'has_secondary_use_industry': [has_secondary_use_industry],
    'has_secondary_use_health_post': [has_secondary_use_health_post],
    'has_secondary_use_gov_office': [has_secondary_use_gov_office],
    'has_secondary_use_use_police': [has_secondary_use_use_police],
    'has_secondary_use_other': [has_secondary_use_other]
})


input_data = label_encode_data(input_data)


# Prediction
if st.button('Predict'):
    prediction = predict(input_data)
    if prediction == 0:
        st.markdown(f"The building damage condition is Low")
    elif prediction == 1:
        st.markdown(f"The building damage condition is Medium")
    else:
        st.markdown(f"The building damage condition is High")
    st.balloons()
