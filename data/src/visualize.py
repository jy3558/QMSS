"""
Mapping and plotting utilities.

- folium_map_by_zip: creates a choropleth map by zipcode using GeoDataFrame of ZIP boundaries.
- plot_time_series: small helper to create time series plots (plotly).
"""
import folium
import pandas as pd
import json
import os
import plotly.express as px

def folium_map_by_zip(zip_gdf, agg_df, zip_field_geo="ZIPCODE", zip_field_agg="zipcode", value_field="mean_hygiene_index", out_html="results/maps/zip_map.html"):
    """
    zip_gdf: GeoDataFrame of ZIP polygons
    agg_df: DataFrame with zipcode and value_field columns (single time slice)
    """
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    # pick most recent period if period column exists
    if "period" in agg_df.columns:
        latest = agg_df.groupby(zip_field_agg).apply(lambda g: g.sort_values("period", ascending=False).head(1)).reset_index(drop=True)
    else:
        latest = agg_df
    merged = zip_gdf.merge(latest, left_on=zip_field_geo, right_on=zip_field_agg, how="left")
    merged_json = merged.to_crs(epsg=4326).__geo_interface__
    center = merged.geometry.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=10, tiles="cartodbpositron")
    folium.Choropleth(
        geo_data=merged_json,
        data=merged,
        columns=[zip_field_geo, value_field],
        key_on=f"feature.properties.{zip_field_geo}",
        fill_color="YlOrRd",
        fill_opacity=0.8,
        line_opacity=0.2,
        legend_name=value_field,
    ).add_to(m)
    m.save(out_html)
    print("Saved map to", out_html)

def plot_time_series(agg_df, zipcode, value_field="mean_hygiene_index", out_png=None):
    df = agg_df[agg_df["zipcode"] == zipcode].sort_values("period")
    fig = px.line(df, x="period", y=value_field, title=f"Time series for {zipcode}")
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.write_image(out_png)
        print("Wrote", out_png)
    return fig
