import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from prediction import predict

def label_encode_data(input_data):
    label_encoder = LabelEncoder()
    categorical_variables = ['land_surface_condition','foundation_type','roof_type','ground_floor_type','other_floor_type','position','plan_configuration','legal_ownership_status']
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
count_families = st.number_input('Number of Families', min_value=0, max_value=10)
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
"""input_data = pd.DataFrame({
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
input_data = label_encode_data(input_data)"""




# -------------------------------------------------------------------------------------------------------------------------------------
# importing required libraries
import pandas as pd
import numpy as np

df=pd.read_csv('train.csv')
test1=pd.read_csv('test.csv')

# We need to remove building_id in both train and test files because it's doesn't change prediction
dr_tr=dr_tr.drop(['building_id','geo_level_1_id','geo_level_2_id','geo_level_3_id'],axis=1)
dr_tt=dr_tt.drop(['building_id','geo_level_1_id','geo_level_2_id','geo_level_3_id'],axis=1)

# converting datatype in train file
int_columns = ['count_floors_pre_eq','has_superstructure_adobe_mud','has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
       'has_superstructure_cement_mortar_stone','has_superstructure_mud_mortar_brick','has_superstructure_cement_mortar_brick',
        'has_superstructure_timber','has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
       'has_superstructure_rc_engineered', 'has_superstructure_other', 'count_families', 'has_secondary_use',
       'has_secondary_use_agriculture', 'has_secondary_use_hotel','has_secondary_use_rental', 'has_secondary_use_institution',
       'has_secondary_use_school', 'has_secondary_use_industry','has_secondary_use_health_post', 'has_secondary_use_gov_office',
       'has_secondary_use_use_police', 'has_secondary_use_other','damage_grade']

df[int_columns] = df[int_columns].astype('object')

# converting datatypes in test file
int_columns_tt = ['count_floors_pre_eq','has_superstructure_adobe_mud','has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
       'has_superstructure_cement_mortar_stone','has_superstructure_mud_mortar_brick','has_superstructure_cement_mortar_brick',
        'has_superstructure_timber','has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
       'has_superstructure_rc_engineered', 'has_superstructure_other', 'count_families', 'has_secondary_use',
       'has_secondary_use_agriculture', 'has_secondary_use_hotel','has_secondary_use_rental', 'has_secondary_use_institution',
       'has_secondary_use_school', 'has_secondary_use_industry','has_secondary_use_health_post', 'has_secondary_use_gov_office',
       'has_secondary_use_use_police', 'has_secondary_use_other']

test1[int_columns_tt] = test1[int_columns_tt].astype('object')

# splitting into numcols and objcols
numcols=df.select_dtype(np.number)
objcols=df.select_dtype('np.number')

from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
numcols_mm=mm.fit_transform(numcols)
numcols_mm=pd.DataFrame(numcols_mm,columns=numcols.columns)

# splitting into numcols and objcols test file
numcols1=test1.select_dtype(np.number)
objcols1=test1.select_dtype('np.number')

numcols_mm_tt=mm.fit_transform(numcols1)
numcols_mm_tt=pd.DataFrame(numcols_mm_tt,columns=numcols1.columns)

# label encoding the object variables
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
objcols=objcols.apply(LabelEncoder().fit_transform)

objcols1=objcols1.apply(LabelEncoder().fit_transform)

# combining the both numcols and objcols
combinedf=pd.concat([numcols,objcols],axis=1)

test1=pd.concat([numcols_tt_mm,objcols1],axis=1)

# splitting into dependent varibale(y) and independent variables(X's)
X=combinedf.drop('damage_grade',axis=1)
y=combinedf.damage_grade

from imblearn.over_sampling import SMOTE
sm=SMOTE()
X,y=sm.fit_resample(X,y)


# Decission Tree
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(max_depth=50, criterion='entropy',splitter='random')
treemodel=tree.fit(X,y)
treemodel.predict(X,y)
cross_val_score(tree,X,y)
np.mean([0.52589101, 0.52278571, 0.51742857, 0.50321429, 0.39735714])
tree_X=treemodel.predict(X)
pd.crosstab(y,tree_X)
print(classification_report(y,tree_X))
treepredict_test=treemodel.predict(test1)
treepredict_tt=pd.DataFrame(treepredict_test)
treepredict_tt=treepredict_tt.replace({0:'Low Damage',1:'Medium Damage',2:'High Damage'})
treepredict_tt.value_counts()

import joblib
joblib.dump(tree,'tree_model_smote.sav')




#-------------------------------------------------------------------------
# Prediction
if st.button('Predict'):
    prediction = predict(np.array(X))
    st.text(predictio[0])
    #if prediction == 0:
     #   st.markdown(f"The building damage condition is Low")
    #elif prediction == 1:
     #   st.markdown(f"The building damage condition is Medium")
    #else:
     #   st.markdown(f"The building damage condition is High")
    st.balloons()
