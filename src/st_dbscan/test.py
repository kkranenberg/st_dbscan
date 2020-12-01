import datetime as datetime
import pandas as pd
import numpy as np
import plotly.express as px
from astropy import coordinates

# pip install -e git+https://github.com/kkranenberg/st_dbscan#egg=ST_DBSCAN


from st_dbscan import ST_DBSCAN


def kkr_prepare_acled_for_st_dbscan(df_acled_api, to_cartesian=False):
    # create DF for ST-DBSCAN
    df_timedelta_days = df_acled_api[['event_date', 'latitude', 'longitude']].copy()
    df_timedelta_days['event_date'] = df_timedelta_days['event_date'].astype('datetime64[ns]')
    # convert date into days since beginning of the dataset
    df_timedelta_days['event_date'] = df_timedelta_days['event_date'] - df_timedelta_days['event_date'].min()
    df_timedelta_days['event_date'] = df_timedelta_days['event_date'].dt.days

    if to_cartesian:
        # spherical to cartesian
        df_timedelta_days['x'], df_timedelta_days['y'], z = coordinates.spherical_to_cartesian(1,
                                                                                               np.deg2rad(
                                                                                                   df_timedelta_days[
                                                                                                       'latitude']),
                                                                                               np.deg2rad(
                                                                                                   df_timedelta_days[
                                                                                                       'longitude']))

        # normalize x& y
        df_timedelta_days['x'] = (df_timedelta_days['x'] - df_timedelta_days['x'].min()) / (
                df_timedelta_days['x'].max() - df_timedelta_days['x'].min())
        df_timedelta_days['y'] = (df_timedelta_days['y'] - df_timedelta_days['y'].min()) / (
                df_timedelta_days['y'].max() - df_timedelta_days['y'].min())

        np_timedelta_days = df_timedelta_days.loc[:, ['event_date', 'x', 'y']].values
    else:
        np_timedelta_days = df_timedelta_days.loc[:, ['event_date', 'latitude', 'longitude']].values
    # normalize latitutde and Longitutde
    # df_timedelta_days['latitude'] = (df_timedelta_days['latitude'] - df_timedelta_days['latitude'].min()) / (
    #        df_timedelta_days['latitude'].max() - df_timedelta_days['latitude'].min())
    # df_timedelta_days['longitude'] = (df_timedelta_days['longitude'] - df_timedelta_days['longitude'].min()) / (
    #       df_timedelta_days['longitude'].max() - df_timedelta_days['longitude'].min())

    return np_timedelta_days


df_acled_api = pd.read_csv('acled_api_20201027_141757.csv')

df_acled_ame = df_acled_api.loc[df_acled_api['region'].str.contains('Middle East')].copy()
df_acled_ame.sort_values(['event_date'], kind='stable', inplace=True, ignore_index=True)
np_acled_ame = kkr_prepare_acled_for_st_dbscan(df_acled_ame)

print('MinPts = ln(', len(np_acled_ame), ') = ', np.round(np.log(len(np_acled_ame))))

fit=True
split=True

frame_size= 100

for eps1 in [50]:
    for eps2 in [7]:
        start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(start, 'Start ST-DBSCAN ', 'eps1:', eps1, 'eps2',
              eps2, 'min_samples:', np.round(np.log(len(np_acled_ame))), 'frame_size:', frame_size
              )

        stdbscan = ST_DBSCAN(eps1, eps2, np.round(np.log(len(np_acled_ame))), metric='haversine')
        if fit:
            stdbscan_fit = ST_DBSCAN(eps1, eps2, np.round(np.log(len(np_acled_ame))), metric='haversine')
            stdbscan_fit.fit(np_acled_ame)

            df_acled_ame['cluster_fit'] = stdbscan_fit.labels

            fig = px.scatter_mapbox(df_acled_ame,
                                    lat="latitude", lon="longitude",  animation_frame='cluster_fit',
                                    zoom=4,
                                    # center=cent,
                                    #color_continuous_scale=px.colors.cyclical.IceFire,
                                    hover_data=['cluster_fit','latitude','longitude', 'event_date','country','actor1','actor2']
                                    )
            fig.update_layout(mapbox_style="open-street-map")
            fig.write_html('fig_'+str(eps1)+'_'+str(eps2)+'.html')
            end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(end, 'Finish ST-DBSCAN ',
                  'eps1:', eps1,
                  'eps2', eps2,
                  'min_samples:', np.round(np.log(len(np_acled_ame))),
                  #      'Clusters:', str(stdbscan_fit.labels.max()), 'Label_count:',
                  'Clusters_fit:', str(stdbscan_fit.labels.max()),
                  #      'Label_count:',len(stdbscan_fit.labels),
                  'Label_count_fit:', len(stdbscan_fit.labels))
            df_acled_ame['cluster_fit'] =  stdbscan_fit.labels
        if split:
            stdbscan_fit_split = ST_DBSCAN(eps1, eps2, np.round(np.log(len(np_acled_ame))), metric='haversine')
            stdbscan_fit_split.fit_frame_split(np_acled_ame, frame_size)

            df_acled_ame['cluster_split'] = stdbscan_fit_split.labels

            fig1 = px.scatter_mapbox(df_acled_ame,
                                     lat="latitude", lon="longitude", animation_frame='cluster_split',
                                     zoom=4,
                                     # center=cent,
                                     # color_continuous_scale=px.colors.cyclical.IceFire,
                                     hover_data=['cluster_split', 'latitude', 'longitude', 'event_date', 'country', 'actor1',
                                                 'actor2']
                                     )
            fig1.update_layout(mapbox_style="open-street-map")
            fig1.write_html('fig_split_hav_' + str(eps1) + '_' + str(eps2) + '.html')

            end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(end, 'Finish ST-DBSCAN ',
                  'eps1:', eps1,
                  'eps2', eps2,
                  'min_samples:', np.round(np.log(len(np_acled_ame))),
                  #      'Clusters:', str(stdbscan_fit.labels.max()), 'Label_count:',
                  'Clusters_split:', str(stdbscan_fit_split.labels.max()),
                  #      'Label_count:',len(stdbscan_fit.labels),
                  'Label_count_split:', len(stdbscan_fit_split.labels))

        # save to result to df_st_dbscan_params


        if fit and split:
            same = stdbscan_fit.labels == stdbscan_fit_split.labels
            unqiue = pd.DataFrame(np.unique(same, return_counts=True))
            print(unqiue)
            # same = stdbscan_fit_split.labels==stdbscan_fit
            # print(same)
            # df_acled_ame['cluster']  = stdbscan_fit.labels




