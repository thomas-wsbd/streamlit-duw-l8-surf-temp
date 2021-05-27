import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_folium import folium_static
import rasterio as rio
import branca.colormap as cm
import numpy as np
from matplotlib import colors as colors
import folium

stats_file = 'data/csvs/stats-l8.csv'
knmi_file = 'data/csvs/etmgeg_350.txt'

st.write("""
# Hittestress in Breda
op basis van *Landsat 8*
""")

st.sidebar.title('Controls')

@st.cache
def load_data():
    stats = pd.read_csv(stats_file, parse_dates=[0, 'dates'], index_col=[0])
    select_imgs = stats[(stats['mean temperature (celsius)'] > 25) & (stats['cloudcover (0-1)'] < 0.2)]
    knmi = pd.read_csv(knmi_file, sep=',', skiprows=46, parse_dates=['YYYYMMDD'], index_col=['YYYYMMDD'], skipinitialspace=True)
    knmi.columns = [col.strip() for col in knmi.columns]
    knmi = knmi['2013':]
    return stats, select_imgs, knmi
stats, select_imgs, knmi = load_data()


plot_type = st.sidebar.radio('plots', ('line knmi meas temp', 'line sat mean surf temp', 'line sat cloudcover', 'scatter sat mean surf temp vs cloudcover', 'map sat surf temp'))

# preset for map-plot
def tif_to_folium_image(infile):
    with rio.open(infile) as src:
        boundary = src.bounds
        img = src.read()
        nodata = src.nodata    

    img[img<0.0] = np.nan
    img = img - 273.15 # to celsius

    # coordinates based on EPSG:32631
    clat = (boundary.bottom + boundary.top)/2
    clon = (boundary.left + boundary.right)/2

    # convert coordinates
    from pyproj import Proj, transform

    inProj = Proj(init='epsg:32631')
    outProj = Proj(init='epsg:4326') # lat-lon
    clon, clat = transform(inProj, outProj, clon, clat)

    right, bottom = transform(inProj, outProj, boundary.right, boundary.bottom)
    left, top = transform(inProj, outProj, boundary.left, boundary.top)
    return img[0], clon, clat, right, bottom, left, top

# colorbar stuff
vmin = 10
vmax = 60

colormap = cm.linear.RdYlBu_11.scale(vmin, vmax)

def reversed_colormap(existing):
    return cm.LinearColormap(
        colors=list(reversed(existing.colors)),
        vmin=existing.vmin, vmax=existing.vmax
    )

colormap = reversed_colormap(colormap)

def mapvalue2color(value, cmap): 
    """
    Map a pixel value of image to a color in the rgba format. 
    As a special case, nans will be mapped totally transparent.
    
    Inputs
        -- value - pixel value of image, could be np.nan
        -- cmap - a linear colormap from branca.colormap.linear
    Output
        -- a color value in the rgba format (r, g, b, a)    
    """
    if np.isnan(value):
        return (1, 0, 0, 0)
    else:
        return colors.to_rgba(cmap(value), 0.7)

# handles plot based on selection (kinda messy)
def figure(plot_type):
    if plot_type == 'line knmi meas temp':
        min_time, max_time = st.sidebar.slider('time select', min_value=knmi.index.min().date(), max_value=knmi.index.max().date(), value=[knmi.index.min().date(), knmi.index.max().date()], step=pd.Timedelta(days=365.25))
        fig = px.scatter(knmi[min_time:max_time], y='TG', title='daily mean average air temperature @ knmi Gilze Rijen', color_discrete_sequence=['gold'], height=600).update_traces(mode='lines+markers', marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')))
        st.plotly_chart(fig, use_container_width=True)
    if plot_type == 'line sat mean surf temp':
        min_time, max_time = st.sidebar.slider('time select', min_value=knmi.index.min().date(), max_value=knmi.index.max().date(), value=[knmi.index.min().date(), knmi.index.max().date()], step=pd.Timedelta(days=365.25))
        fig = (px.scatter(stats[min_time:max_time], y='mean temperature (celsius)', title='mean surface temperature (celsius) for each image, Landsat 8', color_discrete_sequence=['salmon'], height=600).update_traces(mode='lines+markers', marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey'))))
        st.plotly_chart(fig, use_container_width=True)
    if plot_type == 'line sat cloudcover':
        min_time, max_time = st.sidebar.slider('time select', min_value=knmi.index.min().date(), max_value=knmi.index.max().date(), value=[knmi.index.min().date(), knmi.index.max().date()], step=pd.Timedelta(days=365.25))
        fig = (px.scatter(stats[min_time:max_time], y='cloudcover (0-1)', title='cloudcover (0 no clouds, 1 clouded), Landsat 8', height=600).update_traces(mode='lines+markers', marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey'))))
        st.plotly_chart(fig, use_container_width=True)
    if plot_type == 'scatter sat mean surf temp vs cloudcover':
        fig = (px.scatter(stats, x='mean temperature (celsius)', y='cloudcover (0-1)', title='mean surface temperature (celsius) vs no cloud fraction (filled) for each image, Landsat 8', color=stats.index.year, hover_data=[stats.datesstring], color_discrete_sequence= px.colors.sequential.Plasma_r, height=600).update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey'))))
        st.plotly_chart(fig, use_container_width=True)
    if plot_type == 'map sat surf temp':
        l8_image = st.sidebar.selectbox('l8 image selection', options=select_imgs.files.tolist(), index=0)
        infile = f'data/l8/{l8_image}'
        img, clon, clat, right, bottom, left, top = tif_to_folium_image(infile)
        m = folium.Map(location=[clat, clon], zoom_start=13)

        date = infile.split('_')[-1].split('.')[0]
        folium.raster_layers.ImageOverlay(
            image=img,
            name=f'{date}, Landsat 8, Land Surface Temperature (Celsius)',
            opacity=0.7,
            bounds= [[bottom, left], [top, right]],
            colormap= lambda value: mapvalue2color(value, colormap),
            show=True
        ).add_to(m)

        colormap.caption = f'{date}, Landsat 8, Land Surface Temperature (Celsius)'
        m.add_child(colormap)
        folium_static(m, width=1200, height=800)

figure(plot_type)