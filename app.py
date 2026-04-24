import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title='Floral Dress Price Predictor',
    page_icon='🌸',
    layout='centered'
)

st.title('Floral Dress Price Predictor')
st.markdown(
    'Use the form below to select the floral dress product and product attributes, then estimate the expected price using XGBoost model.'
)

# ---------------- LOAD MODEL ----------------
@st.cache_data
def load_model():
    return joblib.load('xgboost.pkl')

# ---------------- LOAD PREPROCESSOR ----------------
@st.cache_data
def load_preprocessor():
    return joblib.load('ui_preprocessor.pkl')

# ---------------- LOAD PRODUCT NAMES ----------------
@st.cache_data
def load_product_names():
    df = pd.read_csv('cleaned_ethnic_dress_dataset_for_deployment.csv')
    names = df['Name'].dropna().value_counts().index.tolist()
    top_names = names[:40]
    return ['Other / Custom product name'] + top_names


# ---------------- OPTIONS ----------------
length_options = ['Midi', 'Maxi', 'Knee Length', 'Above Knee', 'Three-Quarter Sleeves']

neck_options = [
    'V-Neck', 'Round Neck', 'Shirt Collar', 'Mandarin Collar',
    'Square Neck', 'Sweetheart Neck', 'Boat Neck', 'Tie-Up Neck',
    'Keyhole Neck', 'Halter Neck', 'Shoulder Straps', 'Strapless'
]

sleeve_options = [
    'Three-Quarter Sleeves', 'Full Sleeve', 'Sleeveless', 'Short Sleeve',
    'Cap Sleeve', 'Puff Sleeve', 'Bell Sleeve', 'Bishop Sleeve', 'Unknown'
]

shape_options = ['A-Line', 'Maxi']

material_options = [
    'Cotton', 'Viscose Rayon', 'Net', 'Georgette', 'Polyester',
    'Satin', 'Chiffon', 'Crepe', 'Nylon', 'Poly Silk', 'Silk',
    'Jacquard', 'Schiffli', 'Other'
]

category_options = ['Casual', 'Traditional', 'Party Wear', 'Formal', 'Seasonal Wear']


# ---------------- PRODUCT NAME ----------------
product_name_options = load_product_names()

selected_product_name = st.selectbox('Select Product Name', product_name_options)

if selected_product_name == 'Other / Custom product name':
    selected_product_name = st.text_input('Custom Product Name', placeholder='Type a floral dress name here')


# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🤖 Model")
st.sidebar.success("XGBoost")

st.sidebar.markdown('---')
st.sidebar.write('### About this predictor')
st.sidebar.write(
    'Model inputs are based on dress attributes and preprocessed to match the training pipeline.'
)
st.sidebar.write('Click **Predict Price** to see the estimated ₹ value.')


# ---------------- FORM ----------------
with st.form(key='prediction_form'):
    st.subheader('Dress Attributes')

    col1, col2 = st.columns(2)

    with col1:
        length = st.selectbox('Length', length_options)
        neck = st.selectbox('Neck Style', neck_options)
        sleeve_length = st.selectbox('Sleeve Length', sleeve_options)

    with col2:
        shape = st.selectbox('Shape', shape_options)
        sizes_count = st.slider('Number of available sizes', 1, 12, 4)
        material = st.selectbox('Material', material_options)

    category = st.selectbox('Category', category_options)

    submit_button = st.form_submit_button('Predict Price')


# ---------------- PREDICTION ----------------
if submit_button:
    df_input = pd.DataFrame({
        'Length_Code': [
            1 if length == 'Midi' else 2 if length == 'Maxi' else 3 if length == 'Knee Length' else 4 if length == 'Above Knee' else 5
        ],
        'Neck_Code': [
            1 if neck == 'V-Neck' else 2 if neck == 'Round Neck' else 3 if neck == 'Shirt Collar' else 4 if neck == 'Mandarin Collar' else 5 if neck == 'Square Neck' else 6 if neck == 'Sweetheart Neck' else 7 if neck == 'Boat Neck' else 8 if neck == 'Tie-Up Neck' else 9 if neck == 'Keyhole Neck' else 10 if neck == 'Halter Neck' else 11 if neck == 'Shoulder Straps' else 12
        ],
        'Shape_Code': [1 if shape == 'A-Line' else 2],
        'Sizes_Count': [sizes_count],
        'Sleeve_Length': [sleeve_length],
        'Material_Encoded': [material],
        'Category_Encoded': [category]
    })

    try:
        preprocessor = load_preprocessor()
        model = load_model()

        X_input = preprocessor.transform(df_input)
        prediction = model.predict(X_input)[0]

        st.success(f'### 💰 Predicted Price: ₹{int(prediction):,}')

        if selected_product_name:
            st.write(f'**Product:** {selected_product_name}')

        st.write('**Model used:** XGBoost')

        st.markdown('---')
        st.subheader('Input Summary')
        st.table(df_input)

        st.info('The prediction is based on the selected dress attributes and XGBoost model.')

    except Exception as e:
        st.error(f"Error: {e}")