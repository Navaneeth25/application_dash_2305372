import pandas as pd
import numpy as np
import seaborn as sn
import plotly.express as px
import plotly.graph_objects as go
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
df4=fillna_values.copy()
df4= fillna_values[fillna_values['ConfirmedDeaths'] != 0]
data=fillna_values.copy()
covid1 = fillna_values.groupby(['CountryName', 'latitude', 'longitude'])[['ConfirmedCases', 'ConfirmedDeaths']].sum().reset_index()
data1 = fillna_values.query("CountryName == ['United States','Russia','United Kingdom','Spain','Italy','Germany','China','France','Iran','Turkey']")
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
                dcc.Dropdown(id="selecting-plot",options=[{'label': 'race_barplot', 'value': 'race_barplot'},{'label': 'scatter_plot_geo', 'value': 'scatter_plot_geo'},
                                                          {'label': 'scatter_plot_top10', 'value': 'scatter_plot_top10'},{'label': '3D_Scatter_geo', 'value': '3D_Scatter_geo'}
                                                          ], value='race_barplot')

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
    elif g=='scatter_plot_geo':
        fig6 = px.scatter_geo(fillna_values, locations="CountryCode", color="Continent_Name",hover_name="CountryName", 
                    size="ConfirmedCases",animation_frame="Date",size_max=20,projection="natural earth")
        fig6.update_layout(title_text= "Scatter geo of world 03/2020 - 05/2020",title_x=0.3,title_font_family="Sitka Small",
    title_font_color="green")
        return fig6
    elif g=='scatter_plot_top10':
        fig6 = px.scatter(data1, x="ConfirmedCases", y="StringencyIndex", animation_frame="Date", animation_group="CountryName", 
                 size="ConfirmedCases", color="CountryName", text="CountryCode", hover_name="CountryName",
                 #color_discrete_sequence=px.colors.qualitative.G10,
                 #log_x=True, 
                 size_max=30,
                 range_x=[0,2000000],range_y=[0,100])

        fig6.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 500
        fig6.update_layout(showlegend=False)
        fig6.update_layout(title_text="ConfirmedCases vs Stringency Index for top 10 countries",title_x=0.2,title_font_family="Sitka Small",
    title_font_color="green")
        return fig6
    else:
        fig6 = go.Figure()
        for _, row in data.iterrows():
            fig6.add_trace(go.Scattergeo(lat=[row['latitude']],lon=[row['longitude']],mode='markers',marker=dict(
            size=row['ConfirmedCases'] / 500000,opacity=0.8,color='rgb(255, 0, 0)',),
            text=row['CountryName'] + '<br>Cases: ' + str(row['ConfirmedCases']),))
        fig6.update_geos(showcoastlines=True,coastlinecolor="Black",showland=True,showcountries=True,landcolor="rgb(0, 128, 0)",
        showocean=True,oceancolor="rgb(0, 0, 128)",projection_type='orthographic',showframe=False,)
        fig6.update_layout(geo=dict(center=dict(lat=0, lon=0),projection_scale=1.0,),)
        fig6.update_layout(scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),),showlegend=False,
         margin=dict(r=0, l=0, b=0, t=0),)
        fig6.update_layout(title_text="COVID-19 Cases on Interactive 3D Globe")
        return fig6
