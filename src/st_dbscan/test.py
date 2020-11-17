import datetime as datetime
import pandas as pd
import numpy as np
import plotly.express as px
#pip install -e git+https://github.com/kkranenberg/st_dbscan#egg=ST_DBSCAN

from st_dbscan  import ST_DBSCAN

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

df_acled_api = pd.read_csv('acled_api_20201027_141757.csv')

df_acled_ame = df_acled_api.loc[df_acled_api['region'].str.contains('Middle East')].head(10000).copy()
df_acled_ame.sort_values(['event_date'],kind= 'stable',inplace=True,ignore_index=True)
np_acled_ame = kkr_prepare_acled_for_st_dbscan(df_acled_ame)

df_st_dbscan_params = pd.DataFrame(
    columns=['eps1', 'eps2', 'min_samples', 'frame_size', 'frame_overlap'])


print('MinPts = ln(', len(np_acled_ame), ') = ', np.round(np.log(len(np_acled_ame))))


for eps1 in [50,100]:
    for eps2 in [2,10]:

        start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(start, 'Start ST-DBSCAN ',  'eps1:', eps1, 'eps2',
              eps2, 'min_samples:', np.round(np.log(len(np_acled_ame))),'frame_size:', 100
              )

        stdbscan = ST_DBSCAN( eps1, eps2, np.round(np.log(len(np_acled_ame))))
        stdbscan_fit= stdbscan.fit_predict(np_acled_ame)
        stdbscan_fit_split=stdbscan.fit_frame_split(np_acled_ame, 100)

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
        df_acled_ame_split = df_acled_ame.copy()
        split_labels = stdbscan_fit_split.labels
        df_acled_ame_split['cluster'] = split_labels

        same = stdbscan_fit_split.labels==stdbscan_fit
        print(same)
        #df_acled_ame['cluster']  = stdbscan_fit.labels


        #fig = px.scatter_mapbox(df_acled_ame,
        #                        lat="latitude", lon="longitude",  animation_frame='cluster',
        #                        zoom=4,
        #                        # center=cent,
        #                        #color_continuous_scale=px.colors.cyclical.IceFire,
        #                        hover_data=['cluster','latitude','longitude', 'event_date','country','actor1','actor2']
        #                        )
        #fig.update_layout(mapbox_style="open-street-map")
        #fig.write_html('fig_'+str(eps1)+'_'+str(eps2)+'.html')

        fig1 = px.scatter_mapbox(df_acled_ame_split,
                                lat="latitude", lon="longitude",  animation_frame='cluster',
                                zoom=4,
                                # center=cent,
                                #color_continuous_scale=px.colors.cyclical.IceFire,
                                hover_data=['cluster','latitude','longitude', 'event_date','country','actor1','actor2']
                                )
        fig1.update_layout(mapbox_style="open-street-map")
        fig1.write_html('fig_split_'+str(eps1)+'_'+str(eps2)+'.html')