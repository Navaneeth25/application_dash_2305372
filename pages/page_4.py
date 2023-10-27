import pandas as pd
import numpy as np
import seaborn as sn
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import dcc, html, callback, Output, Input
from raceplotly.plots import barplot
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import base64
import warnings
warnings.filterwarnings('ignore')
import dash_bootstrap_components as dbc
dash.register_page(__name__, name='Anomaly Detection')
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
covid_dataset=pd.read_csv('https://raw.githubusercontent.com/Navaneeth25/covid_dataset/main/OxCGRT_summary20200520.csv')
country_continent_dataset=pd.read_csv('https://raw.githubusercontent.com/Navaneeth25/covid_dataset/main/country-and-continent.csv')
countries_lat_long=pd.read_csv('https://raw.githubusercontent.com/Navaneeth25/covid_dataset/main/world_country_and_usa_states_latitude_and_longitude_values.csv')
countries_lat_long.drop(['usa_state_code', 'usa_state_latitude','usa_state_longitude','usa_state','country_code'], axis=1,inplace=True)
countries_lat_long.rename(columns = {'country':'CountryName'}, inplace = True)
new_dataset=pd.merge(covid_dataset, countries_lat_long, on="CountryName",how="left")
country_continent_dataset.dropna(inplace=True)
covid_dataset=new_dataset
merged_dataset= covid_dataset.merge(country_continent_dataset, how = 'left', on = 'CountryCode')
null_continents=merged_dataset[merged_dataset['Continent_Name'].isna()]
merged_dataset['Continent_Name']=merged_dataset['Continent_Name'].fillna(value='Europe')
columns_names=[merged_dataset['Date'].unique()]
columns_names=np.sort(columns_names)
merged_dataset = merged_dataset.groupby('CountryName').apply(lambda x: x.sort_values('Date')).reset_index(drop=True)
fillna_values = merged_dataset.groupby(['CountryName']).fillna(method='ffill').fillna(method='bfill')
fillna_values['CountryName']=merged_dataset['CountryName']
fillna_values = fillna_values.groupby('CountryName').apply(lambda x: x.sort_values('Date')).reset_index(drop=True)
fillna_values['Date'] = pd.to_datetime(fillna_values['Date'], format='%Y%m%d').dt.date
df3 = fillna_values.query("CountryName == ['United States','Russia','Brazil','United Kingdom','Spain']")
df3.reset_index(drop=True,inplace=True)
df3=df3.drop_duplicates(['Date','CountryName']).reset_index(drop=True)
df3['School closing'] = df3['School closing'].map({0: 'no measures', 1: 'recommend closing',2:'require localised closing',3:'require all closing'})
df3['Stay at home requirements'] = df3['Stay at home requirements'].map({0: 'no measures', 1: 'recommend not leaving house',2:'only some exceptions',3:'minimal exceptions'})
fillna_values = fillna_values.drop_duplicates(  subset = ['CountryName', 'Date'],
  keep = 'last').reset_index(drop = True)
fillna_values['Continent_Name'] = fillna_values['Continent_Name'].replace(['North America', 'Europe', 'South America','Africa','Asia','Oceania'], ['north america', 'europe', 'south america','africa','asia','oceania'])
fillna_values['daily_cases'] = fillna_values.groupby('CountryName')['ConfirmedCases'].diff()
fillna_values['daily_cases'] = fillna_values.groupby('CountryName')['daily_cases'].bfill().ffill()
Model_data=fillna_values.copy()
agg_df = Model_data.groupby('Date')['daily_cases'].sum().reset_index()
z_scores = stats.zscore(agg_df['daily_cases'])
threshold = 2
outliers = abs(z_scores) > threshold
agg_df['outliers'] = outliers
data=agg_df[agg_df['outliers'] == 'True'].copy()

data_list = data.to_dict(orient='records')
sidebar = html.Div(
    [
        dbc.Nav(
            [  
                html.Label('Select The Model'),
                dcc.Dropdown(id="selecting-model",options=[{'label': 'Z-score', 'value': 'Z-score'},{'label': 'isolation-forest', 'value': 'isolation-forest'},
                                                           {'label': 'DBScan', 'value': 'DBScan'},
                                                          ], value='Z-score'),
                html.Br(),
                html.Label('Select Country for the model:'),
                dcc.Dropdown(id='Countries',options=[{'label': c, 'value': c}for c in (fillna_values['CountryName'].unique())],value='Not Selected'),
                html.Br(),
                html.Label('Enter threshold'),
                dcc.Input(
                id='threshold',
                type='number',
                debounce=True,
                ),

            ],
            vertical=True
        ),
    ]
)
header = html.H4(
    "Covid 19 anomaly Detection", className="bg-primary text-white p-2 mb-2 text-center"
)
layout= html.Div(children =[header,
                    dbc.Row(
                    [dbc.Col(sidebar,width=2,style={"height": "35vh","margin":"10 px"}),
                    dbc.Col(dcc.Graph(id = 'fig8',style={'height': '60vh',"margin":"10 px"}),width=8), 
                    ]),
                    dbc.Row(
                        [   dbc.Col(width=2,style={"height": "35vh","margin":"10 px"}),
                            dbc.Col(dash_table.DataTable(id='anomaly-table', columns=[],data=[],export_format="csv",),width=8)]
                        ),
                    ],style={"background-color": "black"})

@callback(Output('fig8', 'figure'),Output('anomaly-table', 'columns'),Output('anomaly-table', 'data'),[Input('selecting-model', 'value'),Input('Countries','value'), Input('threshold', 'value')])
def updatefig(g,d,m):
    if g=='Z-score' and d=='Not Selected' and m is None:
        Model_data=fillna_values.copy()
        agg_df = Model_data.groupby('Date')['daily_cases'].sum().reset_index()
        z_scores = stats.zscore(agg_df['daily_cases'])
        threshold = 2
        outliers = abs(z_scores) > threshold
        agg_df['outliers'] = outliers
        data=agg_df[agg_df['outliers'] == True].copy()
        data_list = data.to_dict(orient='records')
        anomaly_columns = [{'name': col, 'id': col} for col in agg_df.columns]
        fig8 = px.scatter(
        agg_df,
        x='Date',
        y='daily_cases',
        color='outliers',
        color_discrete_sequence=['blue', 'red'],  
        labels={'x': 'Date', 'cases': 'COVID-19 Cases'},
        title='Scatter Plot of COVID-19 Cases with Outliers Highlighted Z-Score'
        )
        return fig8,anomaly_columns,data_list
    elif g=='Z-score' and d and m is None:
        zscore_data = fillna_values[fillna_values['CountryName'] == d].copy()
        z_scores = stats.zscore(zscore_data['daily_cases'])
        threshold = 2
        outliers = abs(z_scores) > threshold
        zscore_data['outliers'] = outliers
        data=zscore_data[zscore_data['outliers'] == True].copy()
        data_list = data.to_dict(orient='records')
        anomaly_columns = [
            {"name": "Date", "id": "Date"},
            {"name": "daily_cases", "id": "daily_cases"},
            {"name": "outliers", "id": "outliers"}
        ]
        fig8 = px.scatter(
        zscore_data,
        x='Date',
        y='daily_cases',
        color='outliers',
        color_discrete_sequence=['blue', 'red'],  
        labels={'x': 'Date', 'cases': 'COVID-19 Cases'},
        title='Scatter Plot of COVID-19 Cases with Outliers Highlighted Z-Score'
        )       
        return fig8,anomaly_columns,data_list
    elif g=='isolation-forest' and d=='Not Selected' and m is None:
        Model_data=fillna_values.copy()
        agg_df = Model_data.groupby('Date')['daily_cases'].sum().reset_index()
        data = agg_df['daily_cases'].values.reshape(-1, 1)
        model= IsolationForest(n_estimators=30, max_samples='auto',contamination=0.01) 
        model.fit(data)
        outliers = model.predict(data)
        agg_df['Anomaly'] = outliers
        agg_df['Anomaly'] = agg_df['Anomaly'].map({-1: 'True', 1: 'False'})
        data=agg_df[agg_df['Anomaly'] == 'True'].copy()
        data_list = data.to_dict(orient='records')
        anomaly_columns = [{'name': col, 'id': col} for col in agg_df.columns]
        fig8 = px.scatter(
        agg_df,
        x='Date',
        y='daily_cases',
        color='Anomaly',
        color_discrete_sequence=['blue', 'red'],  
        labels={'x': 'Date', 'cases': 'COVID-19 Cases'},
        title='Scatter Plot of COVID-19 Cases with Outliers Highlighted isolation-forest'
        )
        return fig8,anomaly_columns,data_list
    elif g=='isolation-forest' and d and m is None:
        agg_df = fillna_values[fillna_values['CountryName'] == d]
        data = agg_df['daily_cases'].values.reshape(-1, 1)
        model= IsolationForest(n_estimators=30, max_samples='auto',contamination=0.01) 
        model.fit(data)
        outliers = model.predict(data)
        agg_df['Anomaly'] = outliers
        agg_df['Anomaly'] = agg_df['Anomaly'].map({-1: 'True', 1: 'False'})
        data=agg_df[agg_df['Anomaly'] == 'True'].copy()
        data_list = data.to_dict(orient='records')
        anomaly_columns = [
            {"name": "Date", "id": "Date"},
            {"name": "daily_cases", "id": "daily_cases"},
            {"name": "Anomaly", "id": "Anomaly"}
        ]
        fig8 = px.scatter(
        agg_df,
        x='Date',
        y='daily_cases',
        color='Anomaly',
        color_discrete_sequence=['blue', 'red'], 
        labels={'x': 'Date', 'cases': 'COVID-19 Cases'},
        title='Scatter Plot of COVID-19 Cases with Outliers Highlighted isolation-forest'
        )
        return fig8,anomaly_columns,data_list
    if g=='Z-score' and d=='Not Selected' and m is not None:
        Model_data=fillna_values.copy()
        agg_df = Model_data.groupby('Date')['daily_cases'].sum().reset_index()
        z_scores = stats.zscore(agg_df['daily_cases'])
        threshold = m
        outliers = abs(z_scores) > threshold
        agg_df['outliers'] = outliers
        data=agg_df[agg_df['outliers'] == True].copy()
        data_list = data.to_dict(orient='records')
        anomaly_columns = [
            {"name": "Date", "id": "Date"},
            {"name": "daily_cases", "id": "daily_cases"},
            {"name": "outliers", "id": "outliers"}
        ]
        fig8 = px.scatter(
        agg_df,
        x='Date',
        y='daily_cases',
        color='outliers',
        color_discrete_sequence=['blue', 'red'],  
        labels={'x': 'Date', 'cases': 'COVID-19 Cases'},
        title='Scatter Plot of COVID-19 Cases with Outliers Highlighted Z-Score'
        )
        return fig8,anomaly_columns,data_list
    elif g=='Z-score' and d and m is not None:
        zscore_data = fillna_values[fillna_values['CountryName'] == d].copy()
        z_scores = stats.zscore(zscore_data['daily_cases'])
        threshold = m
        outliers = abs(z_scores) > threshold
        zscore_data['outliers'] = outliers
        data=zscore_data[zscore_data['outliers'] == True].copy()
        data_list = data.to_dict(orient='records')
        anomaly_columns = [
            {"name": "Date", "id": "Date"},
            {"name": "daily_cases", "id": "daily_cases"},
            {"name": "outliers", "id": "outliers"}
        ]
        fig8 = px.scatter(
        zscore_data,
        x='Date',
        y='daily_cases',
        color='outliers',
        color_discrete_sequence=['blue', 'red'],  
        labels={'x': 'Date', 'cases': 'COVID-19 Cases'},
        title='Scatter Plot of COVID-19 Cases with Outliers Highlighted Z-Score'
        )       
        return fig8,anomaly_columns,data_list
    elif g=='isolation-forest' and d=='Not Selected' and m is not None:
        Model_data=fillna_values.copy()
        agg_df = Model_data.groupby('Date')['daily_cases'].sum().reset_index()
        data = agg_df['daily_cases'].values.reshape(-1, 1)
        model= IsolationForest(n_estimators=30, max_samples='auto',contamination=m) 
        model.fit(data)
        outliers = model.predict(data)
        agg_df['Anomaly'] = outliers
        agg_df['Anomaly'] = agg_df['Anomaly'].map({-1: 'True', 1: 'False'})
        data=agg_df[agg_df['Anomaly'] == 'True'].copy()
        data_list = data.to_dict(orient='records')
        anomaly_columns = [
            {"name": "Date", "id": "Date"},
            {"name": "daily_cases", "id": "daily_cases"},
            {"name": "Anomaly", "id": "Anomaly"}
        ]
        fig8 = px.scatter(
        agg_df,
        x='Date',
        y='daily_cases',
        color='Anomaly',
        color_discrete_sequence=['blue', 'red'],  
        labels={'x': 'Date', 'cases': 'COVID-19 Cases'},
        title='Scatter Plot of COVID-19 Cases with Outliers Highlighted isolation-forest'
        )
        return fig8,anomaly_columns,data_list
    elif g=='isolation-forest' and d and m is not None:
        agg_df = fillna_values[fillna_values['CountryName'] == d]
        data = agg_df['daily_cases'].values.reshape(-1, 1)
        model= IsolationForest(n_estimators=30, max_samples='auto',contamination=m) 
        model.fit(data)
        outliers = model.predict(data)
        agg_df['Anomaly'] = outliers
        agg_df['Anomaly'] = agg_df['Anomaly'].map({-1: 'True', 1: 'False'})
        data=agg_df[agg_df['Anomaly'] == 'True'].copy()
        data_list = data.to_dict(orient='records')
        anomaly_columns = [
            {"name": "Date", "id": "Date"},
            {"name": "daily_cases", "id": "daily_cases"},
            {"name": "Anomaly", "id": "Anomaly"}
        ]
        fig8 = px.scatter(
        agg_df,
        x='Date',
        y='daily_cases',
        color='Anomaly',
        color_discrete_sequence=['blue', 'red'], 
        labels={'x': 'Date', 'cases': 'COVID-19 Cases'},
        title='Scatter Plot of COVID-19 Cases with Outliers Highlighted isolation-forest'
        )
        return fig8,anomaly_columns,data_list
    elif g=='DBScan' and d=='Not Selected' and m is not None:
        Model_data=fillna_values.copy()
        DBScan_data = Model_data.groupby('Date')['daily_cases'].sum().reset_index()
        dbscan = DBSCAN(eps=m, min_samples=2)
        dbscan.fit(DBScan_data[['daily_cases']])
        cluster_labels = dbscan.labels_
        DBScan_data['Anomaly'] = cluster_labels
        DBScan_data['Anomaly'] = DBScan_data['Anomaly'].apply(lambda x: 'True' if x == -1 else 'False')
        data=DBScan_data[DBScan_data['Anomaly'] == 'True'].copy()
        data_list = data.to_dict(orient='records')
        anomaly_columns = [
            {"name": "Date", "id": "Date"},
            {"name": "daily_cases", "id": "daily_cases"},
            {"name": "Anomaly", "id": "Anomaly"}
        ]
        fig8 = px.scatter(
        DBScan_data,
        x='Date',
        y='daily_cases',
        color='Anomaly',
        color_discrete_sequence=['blue', 'red'],  
        labels={'x': 'Date', 'cases': 'COVID-19 Cases'},
        title='Scatter Plot of COVID-19 Cases with Outliers Highlighted isolation-forest'
        )
        return fig8,anomaly_columns,data_list
    elif g=='DBScan' and d=='Not Selected' and m is None:
        Model_data=fillna_values.copy()
        DBScan_data = Model_data.groupby('Date')['daily_cases'].sum().reset_index()
        dbscan = DBSCAN(eps=3, min_samples=2)
        dbscan.fit(DBScan_data[['daily_cases']])
        cluster_labels = dbscan.labels_
        DBScan_data['Anomaly'] = cluster_labels
        DBScan_data['Anomaly'] = DBScan_data['Anomaly'].apply(lambda x: 'True' if x == -1 else 'False')
        data=DBScan_data[DBScan_data['Anomaly'] == 'True'].copy()
        data_list = data.to_dict(orient='records')
        anomaly_columns = [
            {"name": "Date", "id": "Date"},
            {"name": "daily_cases", "id": "daily_cases"},
            {"name": "Anomaly", "id": "Anomaly"}
        ]
        fig8 = px.scatter(
        DBScan_data,
        x='Date',
        y='daily_cases',
        color='Anomaly',
        color_discrete_sequence=['blue', 'red'],  
        labels={'x': 'Date', 'cases': 'COVID-19 Cases'},
        title='Scatter Plot of COVID-19 Cases with Outliers Highlighted isolation-forest'
        )
        return fig8,anomaly_columns,data_list
    elif g=='DBScan'  and d and m is not None:
        DBScan_data = fillna_values[fillna_values['CountryName'] == d].copy()
        dbscan = DBSCAN(eps=m, min_samples=2)
        dbscan.fit(DBScan_data[['daily_cases']])
        cluster_labels = dbscan.labels_
        DBScan_data['Anomaly'] = cluster_labels
        DBScan_data['Anomaly'] = DBScan_data['Anomaly'].apply(lambda x: 'True' if x == -1 else 'False')
        data=DBScan_data[DBScan_data['Anomaly'] == 'True'].copy()
        data_list = data.to_dict(orient='records')
        anomaly_columns = [
            {"name": "Date", "id": "Date"},
            {"name": "daily_cases", "id": "daily_cases"},
            {"name": "Anomaly", "id": "Anomaly"}
        ]
        fig8 = px.scatter(
        DBScan_data,
        x='Date',
        y='daily_cases',
        color='Anomaly',
        color_discrete_sequence=['blue', 'red'],  
        labels={'x': 'Date', 'cases': 'COVID-19 Cases'},
        title='Scatter Plot of COVID-19 Cases with Outliers Highlighted isolation-forest'
        )
        return fig8,anomaly_columns,data_list
    elif g=='DBScan' and d and m is None:
        DBScan_data = fillna_values[fillna_values['CountryName'] == d].copy()
        dbscan = DBSCAN(eps=3, min_samples=2)
        dbscan.fit(DBScan_data[['daily_cases']])
        cluster_labels = dbscan.labels_
        DBScan_data['Anomaly'] = cluster_labels
        DBScan_data['Anomaly'] = DBScan_data['Anomaly'].apply(lambda x: 'True' if x == -1 else 'False')
        data=DBScan_data[DBScan_data['Anomaly'] == 'True'].copy()
        data_list = data.to_dict(orient='records')
        anomaly_columns = [
            {"name": "Date", "id": "Date"},
            {"name": "daily_cases", "id": "daily_cases"},
            {"name": "Anomaly", "id": "Anomaly"}
        ]
        fig8 = px.scatter(
        DBScan_data,
        x='Date',
        y='daily_cases',
        color='Anomaly',
        color_discrete_sequence=['blue', 'red'],  
        labels={'x': 'Date', 'cases': 'COVID-19 Cases'},
        title='Scatter Plot of COVID-19 Cases with Outliers Highlighted isolation-forest'
        )
        return fig8,anomaly_columns,data_list
