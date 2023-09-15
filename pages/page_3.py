import pandas as pd
import numpy as np
import seaborn as sn
import plotly.express as px
import dash
from dash import dcc, html, callback, Output, Input
from dash.dependencies import Input, Output
from raceplotly.plots import barplot
import warnings
import dash_pivottable
warnings.filterwarnings('ignore')
import dash_bootstrap_components as dbc
dash.register_page(__name__, name='User_Customisation')
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP,dbc_css])
covid_dataset=pd.read_csv('https://raw.githubusercontent.com/Navaneeth25/covid_dataset/main/OxCGRT_summary20200520.csv')
country_continent_dataset=pd.read_csv('https://raw.githubusercontent.com/Navaneeth25/covid_dataset/main/country-and-continent.csv')
country_continent_dataset.dropna(inplace=True)
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
data=fillna_values.query('Date==20200520').drop_duplicates(['CountryCode']).reset_index(drop=True)
layout = html.Div([
    
    html.Button('Add Pivot Table', id='click', n_clicks=0),
    
    dash_pivottable.PivotTable(
        id='table',
        data=[list(fillna_values.columns)] + fillna_values.values.tolist(),
        cols=['Date'],
        rows=['CountryName'],
        rowOrder="key_a_to_z",
        rendererName="Grouped Column Chart",
        aggregatorName="Average",
        vals=["ConfirmedCases"],
    ),
    
    
    
    
    html.Div(
        id='output'
    )
])
@callback(Output('output', 'children'),
              [Input('table', 'cols'),
               Input('table', 'rows'),
               Input('table', 'rowOrder'),
               Input('table', 'colOrder'),
               Input('table', 'aggregatorName'),
               Input('table', 'rendererName'),
              Input('click', 'n_clicks')])
def display_props(cols, rows, row_order, col_order, aggregator, renderer, n_clicks):
    return [
        dash_pivottable.PivotTable(
        id='table'+str(id),
        data=[list(fillna_values.columns)] + fillna_values.values.tolist(),
        cols=['Date'],
        colOrder="key_a_to_z",
        rows=['CountryName'],
        rowOrder="key_a_to_z",
        rendererName="Grouped Column Chart",
        aggregatorName="Average",
        vals=["ConfirmedCases"],
    ) for id in range(n_clicks)
    ]
