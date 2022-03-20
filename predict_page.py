import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('save_model.pkl', 'rb') as file:
        saved = pickle.load(file)
    return saved

saved=load_model()
rf_loaded=saved['model']
en_country=saved['en_country']
en_education=saved['en_education']

def show_prediction_page():
    st.title('Software Developer Salary Predictions')
    st.write("""### we need some info to predict the salary""")
    countries = ('United States',
                 'United Kingdom',
                 'Germany',
                 'India',
                 'Canada',
                 'France',
                 'Brazil',
                 'Spain',
                 'Australia ',
                 'Netherlands ',
                 'Russian Federation',
                 'Poland',
                 'Italy ',
                 'Sweden ',
                 'Switzerland ',
                 'Israel ',
                 'Ukraine '

                 )

    educations = (
        'Bachelor’s degree', 'Master’s degree', 'post grad', 'less than a bachelor'

    )

    country = st.selectbox("Country", countries)
    education=st.selectbox("Education Level",educations)
    experience=st.slider("Years of Experience",0,50,3)
    ok=st.button('calculate the salary')
    if ok:
        new_data = np.array([[country, education, experience]])
        new_data[:, 0] = en_country.transform(new_data[:, 0])
        new_data[:, 1] = en_education.transform(new_data[:, 1])
        new_data = new_data.astype(float)
        salary=rf_loaded.predict(new_data)
        st.subheader(f'The estimated salary is ${salary[0]:.2f}')




