import json
import time
import base64
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy import arange
from math import ceil
from operator import truediv

hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.set_option('deprecation.showPyplotGlobalUse', False)

def fetch_FoodGroups():
    url = 'http://115.243.144.151/seed/fetchAllFood.php'
    data_fetched = json.loads(requests.post(url).text)
    data_dict = data_fetched['datalist']
    food_df = pd.DataFrame.from_dict(data_dict)
    food_df.to_csv('data/FoodGroups.csv', index=False,
                   header=['Aadhaar', 'Date', 'Grains', 'Pulses', 'otherFruits', 'leafy_Vegetables','other_veg', 'Milk', 'Animal', 'Vitamin_A','Nuts', 'Eggs', 'junk'])

def fetch_Anthropometric():
    url = 'http://115.243.144.151/seed/fetchAllAnthropometric.php'
    data_fetched = json.loads(requests.post(url).text)
    data_dict = data_fetched['datalist']
    food_df = pd.DataFrame.from_dict(data_dict)
    food_df.to_csv('data/AnthropometricParameters.csv', index=False,
                   header=['Aadhaar', 'Month', 'Gender', 'Age', 'Height', 'Weight', 'BodyFat', 'MidArm', 'BMI', 'BMR'])


def fetch_Biochemical():
    url = 'http://115.243.144.151/seed/fetchAllBiochemical.php'
    data_fetched = json.loads(requests.post(url).text)
    data_dict = data_fetched['datalist']
    food_df = pd.DataFrame.from_dict(data_dict)
    food_df.to_csv('data/BiochemicalParameters.csv', index=False,
                   header=['Aadhaar', 'Month', 'Haemoglobin'])


def fetch_Clinical():
    url = 'http://115.243.144.151/seed/fetchAllClinical.php'
    data_fetched = json.loads(requests.post(url).text)
    data_dict = data_fetched['datalist']
    food_df = pd.DataFrame.from_dict(data_dict)
    food_df.to_csv('data/ClinicalParameters.csv', index=False,
                   header=['Aadhaar', 'Month', 'NeckPatches', 'PaleSkin', 'Pellagra', 'WrinkledSkin',
                           'TeethDiscolouration', 'BleedingGums', 'Cavity', 'WeakGums', 'AngularCuts',
                           'InflammedTongue', 'LipCuts', 'MouthUlcer',
                           'BitotSpot', 'Xeropthalmia', 'RedEyes', 'Catract',
                           'HairFall', 'DamagedHair', 'SplitEnds', 'Discolouration',
                           'DarkLines', 'SpoonShapedNails', 'BrokenNails', 'PaleNails','lean','bony','goitre','obesity'])



fetch_FoodGroups()
fetch_Anthropometric()
fetch_Biochemical()
fetch_Clinical()

ANTHRO_URL = (
    "data/AnthropometricParameters.csv"
)
BIO_URL = (
    "data/BiochemicalParameters.csv"
)
CLINIC_URL = (
    "data/ClinicalParameters.csv"
)
DIET_URL = (
    "data/FoodGroups.csv"
)

st.markdown("<h1 style='text-align: center; color: red;'>SEED Project - Admin Dashboard</h1>", unsafe_allow_html=True)
st.sidebar.title("Admin Dashboard")
st.markdown("<div align='center'>Dashboard to gain insights from the user data</div>", unsafe_allow_html=True)


def loadAnthro():
    data = pd.read_csv(ANTHRO_URL, header=0)
    return data


def loadBio():
    data = pd.read_csv(BIO_URL, header=0)
    return data


def loadClinic():
    data = pd.read_csv(CLINIC_URL, header=0)
    return data


def loadDiet():
    data = pd.read_csv(DIET_URL, header=0)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def get_table_download_link(df, name):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() 
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}.csv">Download {name} data</a>'
    return href


anthro = loadAnthro()
bio = loadBio()
clinic = loadClinic()
diet = loadDiet()

dropdown = ['Select one', 'Show data', 'Anthropometric Analysis', 'Biochemical Analysis', 'Clinical Analysis','Analysis','Report']
option = st.sidebar.selectbox(
    'Choose an option',
    dropdown)

if option == 'Show data':
    dropdown.remove('Select one')
    # if st.sidebar.checkbox("Show Anthropometric Details", False, key=1):
    st.markdown("### Anthropometric Table")
    st.markdown("The following table gives you a real-time feed of the AnthropometricParameters table")
    st.dataframe(anthro)
    st.markdown(get_table_download_link(df=anthro, name='Anthropometric'), unsafe_allow_html=True)

    # if st.sidebar.checkbox("Show Biochemical Details", False, key=1):
    st.markdown("### Biochemical Table")
    st.markdown("The following table gives you a real-time feed of the BiochemicalParameters table")
    st.dataframe(bio)
    st.markdown(get_table_download_link(df=bio, name='Biochemical'), unsafe_allow_html=True)

    # if st.sidebar.checkbox("Show Clinical Details", False, key=1):
    st.markdown("### Clinical Table")
    st.markdown("The following table gives you a real-time feed of the ClinicalParameters table")
    st.dataframe(clinic)
    st.markdown(get_table_download_link(df=clinic, name='Clinical'), unsafe_allow_html=True)

    # if st.sidebar.checkbox("Show Dietary Details", False, key=1):
    st.markdown("### Food Groups Table")
    st.markdown("The following table gives you a real-time feed of the FoodGroups table")
    st.dataframe(diet)
    st.markdown(get_table_download_link(df=diet, name='Dietary'), unsafe_allow_html=True)

elif option == 'Anthropometric Analysis':
    #dropdown.remove('Select one')
    st.markdown("## Anthropometric Analysis")
    
    # if st.sidebar.checkbox('Gender wise distribution', False, key=1):
    st.markdown('### Gender wise distribution')
    fig = px.pie(values=anthro['Gender'].value_counts().values, names=anthro['Gender'].value_counts().index)
    st.plotly_chart(fig)

    # if st.sidebar.checkbox('Age', False, key=1):
    st.markdown('### Age wise distribution')
    ages = anthro['Age'].values
    count1 = count2 = count3 = 0
    for age in ages:
        if age < 20:
            count1+=1
        elif age < 60:
            count2+=1
        else:
            count3+=1
    fig = px.bar(x=['Less than 20', 'Between 20 and 60', 'Over 60'], y =[count1, count2, count3])
    fig.update_layout(title="age vs count", xaxis_title="age", yaxis_title="count")
    st.plotly_chart(fig)

    # if st.sidebar.checkbox('BMI'):
    bmi = anthro['BMI'].values
    count1 = count2 = count3 = 0
    for i in bmi:
        if i < 18:
            count1+=1
        elif i < 29:
            count2+=1
        else:
            count3+=1
    fig = px.bar(x=['Less than 18', 'Between 19 and 29', 'Over 30'], y =[count1, count2, count3])
    fig.update_layout(title="BMI Level vs count", xaxis_title="BMI level", yaxis_title="count")
    st.plotly_chart(fig)

elif option == 'Biochemical Analysis':
    #dropdown.remove('Select one')
    print(dropdown)
    st.markdown("## Biochemical Analysis")
    bmi = bio['Haemoglobin'].values
    count1 = count2 = 0
    for i in bmi:
        if i < 13:
            count1+=1
        else:
            count2+=1
    fig = px.bar(x=['Less than 13', 'Over 13'], y =[count1, count2])
    fig.update_layout(title="Haemoglobin vs count", xaxis_title="Haemoglobin level", yaxis_title="count")
    st.plotly_chart(fig)

elif option == 'Clinical Analysis':
    #dropdown.remove('Select one')
    st.markdown("## Clinical Analysis")
    clinic['Total']= clinic.iloc[:, -28:-1].sum(axis=1)
    bmi = clinic[clinic['Month']=='1-2022'].Total.values
    count0=count1 = count2 = count3=count4=count5=count6=count7=count8=count9=count10=0
    for i in bmi:
        if i ==1:
            count1+=1
        elif i==0:
            count0+=1        
        elif i==2:
            count2+=1
        elif i==2:
            count2+=1
        elif i==3:
            count3+=1
        elif i==4:
            count4+=1
        elif i==5:
            count5+=1
        elif i==6:
            count6+=1
        elif i==7:
            count7+=1
        elif i==8:
            count8+=1
        elif i==9:
            count9+=1
        elif i==10:
            count10+=1
    import plotly.express as px
    fig = px.bar(x=['0 symptom','1 symptom','2 symptom','3 symptom','4 symptom','5 symptom','6 symptom','7 symptom','8 symptom','9 symptom','10 symptom'], y =[count0,count1, count2,count3,count4,count5,count6,count7,count8,count9,count10])
    fig.update_layout(title="symptoms vs count", xaxis_title="symptoms", yaxis_title="count")
    st.plotly_chart(fig)
    
    option = st.selectbox(
    'Select the number of symptoms?',
    ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    st.write('You selected:', option)
    st.write(clinic[(clinic['Total']==int(option)) & (clinic['Month']=='1-2022')].Aadhaar)
elif option == 'Analysis':
    body = st.container()
    def fetch_FoodGroups():
        url = 'http://115.243.144.151/seed/fetchAllFood.php'
        data_fetched = json.loads(requests.post(url).text)
        data_dict = data_fetched['datalist']
        food_df = pd.DataFrame.from_dict(data_dict)
        food_df.to_csv('data/Dietary.csv', index=False,header=['Aadhaar', 'Date', 'Grains', 'Pulses', 'otherFruits', 'leafy_Vegetables','other_veg', 'Milk', 'Animal', 'Vitamin_A','Nuts', 'Eggs', 'junk'])

    def fetch_Anthropometric():
        url = 'http://115.243.144.151/seed/fetchAllAnthropometric.php'
        data_fetched = json.loads(requests.post(url).text)
        data_dict = data_fetched['datalist']
        food_df = pd.DataFrame.from_dict(data_dict)
        food_df.to_csv('data/Anthropometric.csv', index=False,header=['Aadhaar', 'Month', 'Gender', 'Age', 'Height', 'Weight', 'BodyFat', 'MidArm', 'BMI', 'BMR'])


    def fetch_Biochemical():
        url = 'http://115.243.144.151/seed/fetchAllBiochemical.php'
        data_fetched = json.loads(requests.post(url).text)
        data_dict = data_fetched['datalist']
        food_df = pd.DataFrame.from_dict(data_dict)
        food_df.to_csv('data/Biochemical.csv', index=False,header=['Aadhaar', 'Month', 'Haemoglobin'])


    def fetch_Clinical():
        url = 'http://115.243.144.151/seed/fetchAllClinical.php'
        data_fetched = json.loads(requests.post(url).text)
        data_dict = data_fetched['datalist']
        food_df = pd.DataFrame.from_dict(data_dict)
        food_df.to_csv('data/Clinical.csv', index=False,
                   header=['Aadhaar', 'Month', 'NeckPatches', 'PaleSkin', 'Pellagra', 'WrinkledSkin',
                           'TeethDiscolouration', 'BleedingGums', 'Cavity', 'WeakGums', 'AngularCuts',
                           'InflammedTongue', 'LipCuts', 'MouthUlcer',
                           'BitotSpot', 'Xeropthalmia', 'RedEyes', 'Catract',
                           'HairFall', 'DamagedHair', 'SplitEnds', 'Discolouration',
                           'DarkLines', 'SpoonShapedNails', 'BrokenNails', 'PaleNails','lean','bony','goitre','obesity'])



    fetch_FoodGroups()
    fetch_Anthropometric()
    fetch_Biochemical()
    fetch_Clinical()

    data1 = pd.read_csv('data/Anthropometric.csv')
    data2 = pd.read_csv('data/Dietary.csv')
    data3 = pd.read_csv('data/Clinical.csv')
    data4 = pd.read_csv('data/Biochemical.csv')

    data1['Month Log'] = data1['Month'].apply(lambda x: 'Month 1' if x == '1-2022' else ('Month 2' if x == '2-2022' else None))

    data3['Month Log'] = data3['Month'].apply(lambda x: 'Month 1' if x == '1-2022' else ('Month 2' if x == '2-2022' else None))

    data4['Month Log'] = data4['Month'].apply(lambda x: 'Month 1' if x == '1-2022' else ('Month 2' if x == '2-2022' else None))

    data1.to_csv('data/Anthropometric_Month.csv', index=False)
    data3.to_csv('data/Clinical_Month.csv', index=False)
    data4.to_csv('data/Biochemical_Month.csv', index=False)

    data2['Date'] = pd.to_datetime(data2['Date'], infer_datetime_format=True)
    data2['Month'] = data2['Date'].apply(lambda x: 'Month ' + str(x.month - 6))
    data2['Day'] = data2['Date'].apply(lambda x: x.day)
    data2.to_csv('data/Dietary_Month.csv', index=False)

    anthropometric = pd.read_csv('data/Anthropometric_Month.csv')
    bioChem = pd.read_csv('data/Biochemical_Month.csv')
    clinical = pd.read_csv('data/Clinical_Month.csv')
    dietary = pd.read_csv('data/Dietary_Month.csv')




    diseases = ['NeckPatches', 'PaleSkin', 'Pellagra',
       'WrinkledSkin', 'TeethDiscolouration', 'BleedingGums', 'Cavity',
       'WeakGums', 'AngularCuts', 'InflammedTongue', 'LipCuts', 'MouthUlcer',
       'BitotSpot', 'Xeropthalmia', 'RedEyes', 'Catract', 'HairFall',
       'DamagedHair', 'SplitEnds', 'Discolouration', 'DarkLines',
       'SpoonShapedNails', 'BrokenNails', 'PaleNails', 'lean', 'bony',
       'goitre', 'obesity']

    foods = ['Grains', 'Pulses', 'otherFruits',
       'leafy_Vegetables', 'other_veg', 'Milk', 'Animal', 'Vitamin_A', 'Nuts',
       'Eggs', 'junk']

    def foodsTable(month, number):
        dictFood = {}
        table = dietary[(dietary['Aadhaar'] == number) & (dietary['Month'] == month)]
        for food in foods:
            dictFood[food] = [sum(table[food]), len(table), sum(table[food])*100/len(table)]

        df = pd.DataFrame(dictFood, index = ['Days Consumed', 'Days Observed', 'Percentage of Days']).transpose()
        df = df[df['Percentage of Days'] != 0]

        return df


    with body:

        number = st.number_input('Choose Aadhaar Number', min_value=1234567100, max_value = 999999999999, value = 1234567100)
    
        if st.button('Generate Report'):
            st.subheader('Report for Aadhaar Number ' + str(number))

            st.write('### Anthropometric Data')

            st.write(anthropometric[anthropometric['Aadhaar'] == number][['Gender', 'Age', 'Height', 'Weight', 'BodyFat', 'MidArm', 'BMI', 'BMR', 'Month Log']].reset_index(drop=True))

            st.write('### Biochemical Data')

            st.write(bioChem[bioChem['Aadhaar'] == number][['Haemoglobin', 'Month Log']].reset_index(drop=True))

            st.write('### Clinical Data')

            st.write('Month 1')
            diseases1 = [disease for disease in diseases if clinical[(clinical['Aadhaar'] == number) & (clinical['Month Log'] == 'Month 1')][disease].reset_index(drop = True)[0] == 1]
            st.write(['None' if len(diseases1) == 0 else i for i in diseases1]) 

            st.write('Month 2')
            diseases2 = [disease for disease in diseases if clinical[(clinical['Aadhaar'] == number) & (clinical['Month Log'] == 'Month 2')][disease].reset_index(drop = True)[0] == 1]
            st.write(['None' if len(diseases2) == 0 else i for i in diseases2])

            st.write('### Dietary Data')
        
            st.write('###### Only food groups consumed are displayed. Absence of group in table implies that it was not consumed in that period.')

            st.write('Month 1')
            st.write(foodsTable('Month 1', number))

            st.write('Month 2')
            st.write(foodsTable('Month 2', number))
elif option == 'Report':

    c1 = st.container()
    c2 = st.container()
    c4 = st.container()
    c3 = st.container()
    c5 = st.container()

    def fetch_FoodGroups():
        url = 'http://115.243.144.151/seed/fetchAllFood.php'
        data_fetched = json.loads(requests.post(url).text)
        data_dict = data_fetched['datalist']
        food_df = pd.DataFrame.from_dict(data_dict)
        food_df.to_csv('Dietary.csv', index=False,header=['Aadhaar', 'Date', 'Grains', 'Pulses', 'Other Fruits', 'Leafy Vegetables','Other Vegetables', 'Dairy', 'Meat, Poultry and Fish', 'Vitamin A Rich','Nuts and Seeds', 'Eggs', 'Junk Foods'])


    fetch_FoodGroups()


    dietary = pd.read_csv('./Dietary.csv')
    dietary['Date'] = pd.to_datetime(dietary['Date'], infer_datetime_format=True)
    dietary['Day'] = dietary['Date'].dt.dayofweek


    daysofweek = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }


    types = ['Grains', 'Pulses', 'Other Fruits',
        'Leafy Vegetables', 'Other Vegetables', 'Dairy',
        'Meat, Poultry and Fish', 'Vitamin A Rich', 'Nuts and Seeds', 'Eggs',
        'Junk Foods']

    def graph1():

        ds1 = pd.DataFrame(columns=['Aadhaar', 'Grains', 'Pulses', 'Other Fruits','Leafy Vegetables', 'Other Vegetables', 'Dairy','Meat, Poultry and Fish', 'Vitamin A Rich', 'Nuts and Seeds', 'Eggs',
        'Junk Foods'])

        for i in dietary['Aadhaar'].unique():
            temp = dietary[dietary['Aadhaar'] == i]
            k = [i]

            for ftype in types:
                k = k + [1 if 1 in list(temp[ftype]) else 0]
            

            ds1.loc[len(ds1.index)] = k  

        out = pd.DataFrame(columns=['Food Type', 'Count'])

        for ftype in types:
            out.loc[len(out)] = [ftype, ds1[ftype].sum()]

        return px.bar(out, x = 'Food Type', y = 'Count', title='Consumption Tally')

    def graph2(num):
        ds2 = pd.DataFrame(columns=['Aadhaar', 'Grains', 'Pulses', 'Other Fruits','Leafy Vegetables', 'Other Vegetables', 'Dairy','Meat, Poultry and Fish', 'Vitamin A Rich', 'Nuts and Seeds', 'Eggs',
       'Junk Foods'])

        for i in dietary['Aadhaar'].unique():
            temp = dietary[dietary['Aadhaar'] == i]

            k = [i]

            for ftype in types:
                k = k + [temp[ftype].sum()]           

            ds2.loc[len(ds2.index)] = k  
    
        out = pd.DataFrame(columns=['Food Type', 'Count'])

        for ftype in types:
            out.loc[len(out)] = [ftype, len([i for i in ds2[ftype] if i >= num])]

        return px.bar(out, x = 'Food Type', y = 'Count', title= 'Frequency of Consumption')

    def graph3():
        dietary['Sum'] = dietary[['Grains', 'Pulses', 'Other Fruits', 'Leafy Vegetables','Other Vegetables', 'Dairy', 'Meat, Poultry and Fish', 'Vitamin A Rich','Nuts and Seeds', 'Eggs']].sum(axis =1)
        red1 = []
        red2 = []
        yellow = []
        green1 = []
        green2 = []

        for i in dietary['Aadhaar'].unique():
            temp = dietary[dietary['Aadhaar'] == i]

            k = ceil(temp['Sum'].mean())

            if k in range(1, 3):
                red1.append(i)
            elif k in range(3, 5):
                red2.append(i)
            elif k in range(5, 7):
                yellow.append(i)
            elif k in range(7, 10):
                green1.append(i)
            elif k in range(10, 11):
                green2.append(i)

        fig = go.Figure(data = [go.Bar(
            x = ['Dietary Diversity : 1-2', 'Dietary Diversity : 3-4', 'Dietary Diversity : 5-6', 'Dietary Diversity : 7-9', 'Dietary Diversity : 10-11'],
            y = [len(red1), len(red2), len(yellow), len(green1), len(green2)],
            marker_color = ['red', 'red', 'yellow', 'green', 'green']
        )])

        N = len(dietary['Aadhaar'].unique())

        fig.update_layout(title_text = f'Food Groups Consumed, Ceiling of Mean, N = {N}')

        return fig, red1, red2, yellow, green1, green2

    def graph4():
        delta = pd.DataFrame(columns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

        for ftype in types:
            change = []
            dayCount = [len(dietary[dietary.Day == day]) for day in range(7)]
            for day in range(7):
                change.append(dietary[dietary['Day'] == day][ftype].sum())

            countScaled = list(map(truediv, change, dayCount))
        
            delta.loc[ftype] = [i*100 for i in countScaled]

        return px.line(delta.transpose(), title = 'Variation by Day of the Week, Scaled', markers = True)

    def graph5(data, featureA, featureB, day):

        data = data[data['Day'] == daysofweek[day]]
        data = data[[featureA, featureB]].replace({0: 'Not Consumed', 1: 'Consumed'}) 

        lenData = len(data)
        title = f'Relating {featureA} and {featureB} on {day}, N = {lenData}'
    
        return px.bar(pd.crosstab(data[featureA], data[featureB]), color=featureB, title = title, width = 1000, orientation = 'h')

    with c1:        
        st.header('Consumption Tally of Food Groups')
        fig1 = graph1()
        st.write(fig1)

    with c2:
        st.header('Frequency of Consumption of Food Groups')
        num = st.selectbox('Number of Days', (arange(1, 29)))
        fig2 = graph2(num)
        st.write(fig2)

    with c4:
        st.header('Variation in Food Intake by Day of the Week')
        fig4 = graph4()
        st.write(fig4)

    with c3:
        fig3, red1, red2, yellow, green1, green2 = graph3()
        st.header('Exploration of Dietary Diversity')
        st.write('In this graphic, a color of red indicates that the average diversity of a person\'s diet is below 4 groups. A color of yellow indicates that the diet is restricted to 5 or 6 groups. A color of green indicates a diverse diet that includes more than 6 groups.')
        st.write(fig3)
        with st.expander('People in Red Zone - 1-2'):
            st.write(pd.Series(red1, index = arange(1, len(red1) + 1)), name = 'Aadhaar Numbers')
        with st.expander('People in Red Zone - 3-4'):
            st.write(pd.Series(red2, index = arange(1, len(red2) + 1)), name = 'Aadhaar Numbers')
        with st.expander('People in Yellow Zone - 5-6'):
            st.write(pd.Series(yellow, index = arange(1, len(yellow) + 1)), name = 'Aadhaar Numbers')
        with st.expander('People in Green Zone - 7-9'):
            st.write(pd.Series(green1, index = arange(1, len(green1) + 1)), name = 'Aadhaar Numbers')
        with st.expander('People in Green Zone - 10-11'):
            st.write(pd.Series(green2, index = arange(1, len(green2) + 1)), name = 'Aadhaar Numbers')
    with c5:
        st.header('Comparing Consumption of Food Groups')
        with st.expander('Filters'):
            with st.form('compareForm'):
                features = st.multiselect('Food Groups', types)

                day = st.selectbox('Day', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

                submitted = st.form_submit_button('Submit')

                if submitted:
                    st.write(graph5(dietary, features[0], features[1], day))

