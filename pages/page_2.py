import pandas as pd
import numpy as np
import seaborn as sn
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import dcc, html, callback, Output, Input
from raceplotly.plots import barplot
import warnings
warnings.filterwarnings('ignore')
import dash_bootstrap_components as dbc
dash.register_page(__name__, name='Animated plots')
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
covid_dataset=pd.read_csv('https://raw.githubusercontent.com/Navaneeth25/covid_dataset/main/OxCGRT_summary20200520.csv')
country_continent_dataset=pd.read_csv('https://raw.githubusercontent.com/Navaneeth25/covid_dataset/main/Downloads/country-and-continent.csv')
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
df3 = fillna_values.query("CountryName == ['United States','Russia','Brazil','United Kingdom','Spain']")
df3.reset_index(drop=True,inplace=True)
df3=df3.drop_duplicates(['Date','CountryName']).reset_index(drop=True)
df3['School closing'] = df3['School closing'].map({0: 'no measures', 1: 'recommend closing',2:'require localised closing',3:'require all closing'})
df3['Stay at home requirements'] = df3['Stay at home requirements'].map({0: 'no measures', 1: 'recommend not leaving house',2:'only some exceptions',3:'minimal exceptions'})
fillna_values = fillna_values.drop_duplicates(  subset = ['CountryName', 'Date'],
  keep = 'last').reset_index(drop = True)
fillna_values['Continent_Name'] = fillna_values['Continent_Name'].replace(['North America', 'Europe', 'South America','Africa','Asia','Oceania'], ['north america', 'europe', 'south america','africa','asia','oceania'])
df4=fillna_values.copy()
df4= fillna_values[fillna_values['ConfirmedDeaths'] != 0]
df4['date'] = pd.to_datetime(df4['Date'], format='%Y%m%d')
data=fillna_values.copy()
fig6= barplot(data,  item_column='CountryName', value_column='ConfirmedCases', time_column='Date')
fig6.plot(item_label = 'Top 10 countries', value_label = 'cases', frame_duration = 800)
fig6=fig6.fig
fig6.update_layout(title_text= "Race bar plot for top 10 countries",title_x=0.3,title_font_family="Sitka Small",
    title_font_color="green")
sidebar = html.Div(
    [
        dbc.Nav(
            [  
                html.Label('Select Animated Chart'),
                dcc.Dropdown(id="selecting-plot",options=[{'label': 'race_barplot', 'value': 'race_barplot'},{'label': 'scatter_plot', 'value': 'scatter_plot'}], value='race_barplot')

            ],
            vertical=True
        ),
    ]
)
header = html.H4(
    "Covid 19 animations Dashboard", className="bg-primary text-white p-2 mb-2 text-center"
)
layout= html.Div(children =[header,
                    dbc.Row(
                    [dbc.Col(sidebar,width=2,style={"height": "35vh","margin":"10 px"}),
                    dbc.Col(dcc.Graph(id = 'fig6',style={'height': '60vh',"margin":"10 px"},figure =fig6),width=8), 
                    ])],style={"background-color": "black"})
@callback(Output('fig6', 'figure'),[Input('selecting-plot', 'value')])
def updatefig(g):
    if g=='race_barplot':
        fig6= barplot(data,  item_column='CountryName', value_column='ConfirmedCases', time_column='Date')
        fig6.plot(item_label = 'Top 10 countries', value_label = 'cases', frame_duration = 800)
        fig6=fig6.fig
        fig6.update_layout(title_text= "Race bar plot for top 10 countries",title_x=0.3,title_font_family="Sitka Small",
    title_font_color="green")
        return fig6
    else:
        fig6 = px.scatter_geo(fillna_values, locations="CountryCode", color="Continent_Name",hover_name="CountryName", 
                    size="ConfirmedCases",animation_frame="Date",size_max=20,projection="natural earth")
        fig6.update_layout(title_text= "Scatter geo of world 03/2020 - 05/2020",title_x=0.3,title_font_family="Sitka Small",
    title_font_color="green")
        return fig6
