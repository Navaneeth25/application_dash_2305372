import pandas as pd
import numpy as np
import seaborn as sn
import plotly.express as px
import dash
from dash import dcc, html, callback, Output, Input
from dash.dependencies import Input, Output
from raceplotly.plots import barplot
import warnings
warnings.filterwarnings('ignore')
import dash_bootstrap_components as dbc
dash.register_page(__name__, path='/', name='Home')

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
data=fillna_values.query('Date==20200520').drop_duplicates(['CountryCode']).reset_index(drop=True)
df3 = fillna_values.query("CountryName == ['United States','Russia','Brazil','United Kingdom','Spain']")
df3.reset_index(drop=True,inplace=True)
df3=df3.drop_duplicates(['Date','CountryName']).reset_index(drop=True)
df3['Date'] = pd.to_datetime(df3['Date'], format='%Y%m%d')
df3['School closing'] = df3['School closing'].map({0: 'no measures', 1: 'recommend closing',2:'require localised closing',3:'require all closing'})
df3['Stay at home requirements'] = df3['Stay at home requirements'].map({0: 'no measures', 1: 'recommend not leaving house',2:'only some exceptions',3:'minimal exceptions'})
fillna_values = fillna_values.drop_duplicates(  subset = ['CountryName', 'Date'],
  keep = 'last').reset_index(drop = True)
fillna_values['Continent_Name'] = fillna_values['Continent_Name'].replace(['North America', 'Europe', 'South America','Africa','Asia','Oceania'], ['north america', 'europe', 'south america','africa','asia','oceania'])
df4=fillna_values.copy()
df4= fillna_values[fillna_values['ConfirmedDeaths'] != 0]
df4['date'] = pd.to_datetime(df4['Date'], format='%Y%m%d')
fillna_values['date'] = pd.to_datetime(fillna_values['Date'], format='%Y%m%d')
fig1=px.line(df3,x='Date',y='ConfirmedCases',log_y=True,color='CountryName')
fig2=px.sunburst(fillna_values, color='StringencyIndex', values='ConfirmedCases',
                path=['Continent_Name','CountryName'])
fig4 = px.treemap(fillna_values, path=[px.Constant('world'), 'Continent_Name','CountryName',], values='ConfirmedCases',
                  color='StringencyIndex', hover_data=['CountryName'])
fig5 = px.scatter_geo(fillna_values, locations="CountryCode", color="Continent_Name",hover_name="CountryName", 
                    size="ConfirmedCases",animation_frame="Date",size_max=20,projection="natural earth")
sidebar = html.Div(
    [
        dbc.Nav(
            [   html.Label('Scope'),
                dcc.Dropdown(id='Continent',placeholder='Scope',
                options=[{'label': 'world', 'value': 'world'},
                         {'label': 'asia', 'value': 'asia'},
                         {'label': 'europe', 'value': 'europe'},
                         {'label': 'africa', 'value': 'africa'},
                         {'label': 'south america', 'value': 'south america'},
                         {'label': 'north america', 'value': 'north america'},
                         {'label': 'oceania', 'value': 'oceania'}],value='world'),
                html.Br(),
                html.Label('Data Input'),
                dcc.Dropdown(id="Data Input",options=[{'label': 'Confirmed cases', 'value': 'ConfirmedCases'},{'label': 'Confirmed Deaths', 'value': 'ConfirmedDeaths'},
                                             {'label': 'Stringency Index', 'value': 'StringencyIndex'}], value='ConfirmedCases'),
                html.Br(),
                html.Label('Policy'),
                dcc.Dropdown(id="Policy", options=[{'label': 'Not Selected', 'value': 'Not Selected'},{'label': 'School closing', 'value': 'School closing'},
                                             {'label': 'Stay at home requirements', 'value': 'Stay at home requirements'}], value='Not Selected')

            ],
            vertical=True
        ),
    ]
)
header = html.H4(
    "Covid 19 Dashboard", className="bg-primary text-white p-2 mb-2 text-center"
)
layout = html.Div(children =[header,
                    dbc.Row(
                    [dbc.Col(sidebar,width=2,style={"height": "35vh","margin":"10 px"}),
                    dbc.Col(dcc.Graph(id = 'fig1',style={'height': '40vh',"margin":"10 px"},figure =fig1),width=5),
                     dbc.Col(dcc.Graph(id='fig2',style={'height':'40vh',"margin":"10 px"},figure=fig2),width=5) 
                    ]),
                    dbc.Row([
                    dbc.Col(width=2,style={"height": "45vh"}),
                    dbc.Col(dcc.Graph(id = 'fig4',figure = fig4,style={'height':'40vh',"margin":"10 px"}),width=5,style={'float':'right','height':'40vh','margin-top': '10px'}),
                    dbc.Col(dcc.Graph(id='fig5',figure=fig5,style={'height':'40vh',"margin":"10 px"}),width=5,style={'height':'40vh','margin-top': '10px'})
                           ])
                    ],style={"background-color": "black"})
                           
                    
@callback(Output('fig1', 'figure'),Output('fig2','figure'),Output('fig4','figure'),Output('fig5','figure'),[Input('Continent', 'value'),Input('Data Input','value'),Input('Policy','value')])
def updatefig(g,d,m):
    if g=='world':
        df=df4.copy()
    else:
        df = df4[df4['Continent_Name'] == g]
    if g=="world" and g!="oceania" and d=="Confirmed Cases":
        return fig1,fig2,fig4,fig5
    elif g and g!="oceania" and d and m=="Not Selected":
        fig5=px.scatter_geo(fillna_values, locations="CountryCode", color="Continent_Name",hover_name="CountryName", 
                    size=d,animation_frame="Date",size_max=20,scope=g,projection="natural earth")
        fig5.update_layout(title_text= d +" of " + g + " from 1st MAR 2020 to 20 MAY 2020")
        fig2=px.sunburst(df, color='StringencyIndex', values=d,path=['Continent_Name','CountryName'],hover_name='Continent_Name')
        fig2.update_layout(title_text= d +" for top 5 countries")
        fig4 = px.treemap(df, path=[px.Constant('world'), 'Continent_Name','CountryName',], values=d,
                  color='StringencyIndex', hover_data=['CountryName'])
        fig4.update_layout(title_text= d +" for top 5 countries")
        if d=='ConfirmedCases':
            fig1=px.line(df3,x='Date',y=d,color='CountryName',log_y=True)
            fig1.update_layout(title_text= d +" for top 5 countries")
        else:
            fig1=px.line(df3,x='Date',y=d,color='CountryName')
            fig1.update_layout(title_text= d +" for top 5 countries")
            
        return fig1,fig2,fig4,fig5
    elif g=="oceania" and d and m=="Not Selected":
        fig5=px.scatter_geo(fillna_values, locations="CountryCode", color="Continent_Name",hover_name="CountryName", 
                    size=d,animation_frame="Date",size_max=20,scope='world',projection="natural earth")
        fig5.update_geos(center=dict(lon=150, lat=-25), projection_rotation=dict(lon=0, lat=0, roll=0), scope='world')
        fig5.update_geos(lataxis_range=[-50, 10], lonaxis_range=[95, 180])
        fig5.update_layout(title_text= d + " of " + g + " from 1st MAR 2020 to 20 MAY 2020")
        fig2=px.sunburst(df, color='StringencyIndex', values=d,path=['Continent_Name','CountryName'],hover_name='Continent_Name')
        fig2.update_layout(title_text= d +" for top 5 countries")
        fig4 = px.treemap(df, path=[px.Constant('world'), 'Continent_Name','CountryName',], values=d,
                color='StringencyIndex',hover_name='CountryName')
        fig4.update_layout(title_text= d +" for top 5 countries")
        if d=='ConfirmedCases':
            fig1=px.line(df3,x='Date',y=d,color='CountryName',log_y=True)
            fig1.update_layout(title_text= d +" for top 5 countries")
        else:
            fig1=px.line(df3,x='Date',y=d,color='CountryName')
            fig1.update_layout(title_text= d +" for top 5 countries")
        return fig1,fig2,fig4,fig5
    elif g=="oceania" and m=="School closing" or g=="Oceania" and m=="Stay at home requirements":
        fig5= px.choropleth(fillna_values, locations="CountryCode",
                            color=m,animation_frame="Date",hover_name="CountryName",color_continuous_scale=px.colors.sequential.Plasma,scope='world')
        fig5.update_geos(center=dict(lon=150, lat=-25), projection_rotation=dict(lon=0, lat=0, roll=0), scope='world')
        fig5.update_geos(lataxis_range=[-50, 10], lonaxis_range=[95, 180])
        fig5.update_layout(title_text= m + " of " + g + " from 1st MAR 2020 to 20 MAY 2020")
        fig2 = px.bar(df, x="date", y="ConfirmedCases", color="Continent_Name", title="confirmedcases")
        fig4 = px.bar(df, x="date", y="ConfirmedDeaths", color="Continent_Name", title="confirmedDeaths")
        fig1=px.line(df3,x='Date',y=m,color='CountryName')
        fig1.update_layout(title_text= m +" for top 5 countries")
        if m=="School closing":
            fig1.update_yaxes(categoryorder='array', categoryarray= ['no measures', 'recommend closing', 'require localised closing', 'require all closing'])
        return fig1,fig2,fig4,fig5
    elif g and g!="oceania" and m=="School closing" or m=="Stay at home requirements":
        fig5= px.choropleth(fillna_values, locations="CountryCode",
                            color=m,animation_frame="Date",hover_name="CountryName",color_continuous_scale=px.colors.sequential.Plasma,scope=g)
        fig5.update_layout(title_text= m + " of " + g + " from 1st MAR 2020 to 20 MAY 2020")
        fig1=px.line(df3,x='Date',y=m,color='CountryName')
        fig1.update_layout(title_text= m +" for top 5 countries")
        fig2 = px.bar(df, x="date", y="ConfirmedCases", color="Continent_Name", title="confirmedcases")
        fig4 = px.bar(df, x="date", y="ConfirmedDeaths", color="Continent_Name", title="confirmedDeaths")
        if m=="School closing":
            fig1.update_yaxes(categoryorder='array', categoryarray= ['no measures', 'recommend closing', 'require localised closing', 'require all closing'])
        return fig1,fig2,fig4,fig5
