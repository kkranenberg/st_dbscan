import datetime as datetime
import pandas as pd
import numpy as np
import plotly.express as px
from .st_dbscan import ST_DBSCAN

def kkr_prepare_acled_for_st_dbscan(df_acled_api):
    # create DF for ST-DBSCAN
    df_timedelta_days = df_acled_api[['event_date', 'latitude', 'longitude']].copy()
    df_timedelta_days['event_date'] = df_timedelta_days['event_date'].astype('datetime64[ns]')
    # spherical to cartehsian

    # df_timedelta_days['x'], df_timedelta_days['y'], z = coordinates.spherical_to_cartesian(1,
    #                                                                                   np.deg2rad(
    #                                                                                       df_timedelta_days['latitude']),
    #                                                                                   np.deg2rad(df_timedelta_days[
    #                                                                                                  'longitude']))

    # normalize x & y
    #df_timedelta_days['latitude'] = (df_timedelta_days['latitude'] - df_timedelta_days['latitude'].min()) / (
    #        df_timedelta_days['latitude'].max() - df_timedelta_days['latitude'].min())
    #df_timedelta_days['longitude'] = (df_timedelta_days['longitude'] - df_timedelta_days['longitude'].min()) / (
    #       df_timedelta_days['longitude'].max() - df_timedelta_days['longitude'].min())

    # convert date into days since beginning of the dataset
    df_timedelta_days['event_date'] = df_timedelta_days['event_date'] - df_timedelta_days['event_date'].min()
    df_timedelta_days['event_date'] = df_timedelta_days['event_date'].dt.days
    #    print(resumetable(df_timedelta_days))
    np_timedelta_days = df_timedelta_days.loc[:, ['event_date', 'latitude', 'longitude']].values
    return np_timedelta_days

df_acled_api = pd.read_csv('./acled_api_20201027_141757.csv')

df_acled_ame = df_acled_api.loc[df_acled_api['region'].str.contains('Middle East')].head(40000).copy()
np_acled_ame = kkr_prepare_acled_for_st_dbscan(df_acled_ame)

df_st_dbscan_params = pd.DataFrame(
    columns=['eps1', 'eps2', 'min_samples', 'frame_size', 'frame_overlap' ])


print('MinPts = ln(', len(np_acled_ame), ') = ', np.round(np.log(len(np_acled_ame))))


for eps1 in [0.0078,50]:
    for eps2 in [10,2]:
        row = dict(
            zip(list(df_st_dbscan_params.columns),
                [eps1, eps2, np.round(np.log(len(df_acled_ame))), 180, None]))
        df_st_dbscan_params = df_st_dbscan_params.append(row, ignore_index=True)

for i, row in df_st_dbscan_params.iterrows():
    start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(start, 'Start ST-DBSCAN ', row.name + 1, 'of', len(df_st_dbscan_params), 'eps1:', row['eps1'], 'eps2',
          row['eps2'], 'min_samples:', row['min_samples']
          )

    stdbscan = ST_DBSCAN(row['eps1'], row['eps2'], row['min_samples'], metric='haversine')
    stdbscan.fit_frame_split(np_acled_ame, row['frame_size'], row['frame_overlap'])

    end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(end, 'Finish ST-DBSCAN ', row.name + 1, 'of', len(df_st_dbscan_params), 'eps1:', row['eps1'], 'eps2',
          row['eps2'], 'min_samples:', row['min_samples'],
          'Clusters:', str(stdbscan.labels.max()), 'Label_count:', len(stdbscan.labels))
    # save to result to df_st_dbscan_params

    df_acled_ame['cluster']  = stdbscan.labels


    fig = px.scatter_mapbox(df_acled_ame,
                            lat="latitude", lon="longitude", color='cluster', animation_frame='cluster',
                            zoom=2,
                            # center=cent,
                            color_continuous_scale=px.colors.cyclical.IceFire)
    fig.update_layout(mapbox_style="open-street-map")
    fig.write_html('./maps/fig_'+row['eps1']+'_'+row['eps2']+'.html')