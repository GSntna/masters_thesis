import geopandas as gpd

shp_filepath = 'zmg_shp/mapa_venta.shp'

rename_cols = {
'COLONIA': 'neighborhood',
'NOMBRE_MUN': 'county',
'geometry': 'geometry'
}

counties_filter = ['Guadalajara', 'Zapopan']

def get_gdf(filepath:str, rename_cols:dict) -> gpd.GeoDataFrame:
    '''
    Reads the Zona Metropolitana de Guadalajara (ZMG) shapefile, removes
    unnnecesary columns and rename remaining columns to english translation

    :param str path: path to the shape file to be imported
    :param rename_cols: dictionary with the columns to keep and their new name
    '''
    gdf = gpd.read_file(filepath)

    # Drop non-important columns and rename
    gdf = gdf[rename_cols.keys()]
    gdf.rename(columns=rename_cols, inplace=True)

    # Change casing
    gdf['neighborhood'] = gdf['neighborhood'].str.capitalize()
    gdf['county'] = gdf['county'].str.capitalize()

    # Filter Guadalajara and Zapopan only
    gdf = gdf[gdf['county'].isin(counties_filter)] 

    return gdf

if __name__ == '__main__':
    gdf = get_gdf(filepath=shp_filepath, rename_cols=rename_cols)
    gdf.to_file('../../database/neighborhoods.gpkg', driver='GPKG')
