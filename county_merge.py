#!/usr/bin/env python

import os
import sys
import glob
import datetime
from datetime import datetime
from io import StringIO
import argparse
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import ffmpy
from colour import Color
from PIL import Image

import timeline


data_file = 'data.csv'
upper_end = .9
image_width = 1920
image_height = 1100
crossfade = 2
slide_time = 2
per_capita_unit = 100000
per_capita_string = '100K'
start_date = '2020-03-01'
metric = 'cases_pc'
title = 'US County COVID-19 New Cases Per Day Per ' + per_capita_string
# title = 'US County COVID-19 Deaths Per Day Per ' + per_capita_string
# title = 'US County COVID-19 Total Cases'
# title = 'US County COVID-19 Total Deaths'


def log(msg):
    """Log provided message to console along with timestamp."""
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' + msg)


def gen_data():
    """Retrieve data from datasources and merge into dataframe."""
    if os.path.exists(data_file):
        log('reading in existing data file')
        new_df = pd.read_csv(data_file, dtype={'fips': str})
        return new_df

    log('retrieving data file from nytimes github')
    url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
    log('data file retrieved')
    log('creating base pandas dataframe')
    dat_df = pd.read_csv(url, dtype={'fips': str})

    dat_df = dat_df[dat_df.date >= start_date]

    dates = dat_df['date'].unique().tolist()

    log('retrieving data file from census.gov')
    url = 'https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals' \
          '/co-est2019-alldata.csv'
    log('data file retrieved')
    log('creating population dataframe')
    pop_df = pd.read_csv(
        url, encoding='ISO-8859-1',
        dtype={'SUBLEV': str, 'REGION': str, 'DIVISION': str, 'STATE': str, 'COUNTY': str}
    )


    pop_df = pop_df[pop_df['COUNTY'] != '000']
    pop_df['fips'] = pop_df['STATE'] + pop_df['COUNTY']
    pop_df = pop_df[['fips', 'POPESTIMATE2019', 'CTYNAME', 'STNAME']]
    pop_df['CTYNAME'] = pop_df['CTYNAME'].apply(lambda x: x.replace(' County', ''))
    pop_df.columns = ['fips', 'population', 'county', 'state']


    csv = 'date,fips,population,county,state\n'
    for date in dates:
        for idx, row in pop_df.iterrows():
            csv += '%s,%s,%s,"%s","%s"\n' % (date, row['fips'], row['population'],
                                             row['county'], row['state'])

    log('merging data frames')
    new_df = pd.read_csv(StringIO(csv), dtype={'fips': str})
    new_df = new_df.merge(dat_df, how='left', left_on=['date', 'fips'],
                          right_on=['date', 'fips'])

    # new_df = dat_df.join(pop_df.set_index('fips'), on='fips', lsuffix='_x', rsuffix='_y')

    new_df = new_df[['date', 'fips', 'population', 'county_x', 'state_x', 'cases', 'deaths']]
    new_df.columns = ['date', 'fips', 'population', 'county', 'state', 'cases', 'deaths']
    new_df = new_df.fillna(0)

    log('calculating new columns')
    new_df['cases'] = new_df['cases'].astype(int)
    new_df['deaths'] = new_df['deaths'].astype(int)

    new_df['cases_new'] = new_df.groupby(['fips'])['cases'].diff().fillna(0)
    new_df['deaths_new'] = new_df.groupby('fips')['deaths'].diff().fillna(0)

    new_df['cases_new'] = new_df['cases_new'].apply(lambda x: 0 if x < 0 else x)
    new_df['deaths_new'] = new_df['deaths_new'].apply(lambda x: 0 if x < 0 else x)

    new_df['cases_roll'] = new_df.groupby('fips')['cases_new'].rolling(3).mean() \
                                 .reset_index(0, drop=True).fillna(0)
    new_df['deaths_roll'] = new_df.groupby('fips')['deaths_new'].rolling(3).mean() \
                                  .reset_index(0, drop=True).fillna(0)

    new_df['cases_pc'] = new_df['cases_roll'] / new_df['population'] * per_capita_unit
    new_df['deaths_pc'] = new_df['deaths_roll'] / new_df['population'] * per_capita_unit

    log('saving merged dataframe')
    new_df.to_csv('data.csv', index=False)

    log('done processing dataframe')
    return new_df


def gen_image(date, dimension, new_df):
    """Create map image for specific date and dimension."""
    fips = new_df['fips'][new_df['date'] == date].unique().tolist()
    values = new_df[dimension][new_df['date'] == date].tolist()

    colorscale = gen_colorscale()

    max_range = new_df[dimension][new_df[dimension] > 0].quantile(upper_end)
    endpts = list(np.linspace(0.01, max_range, len(colorscale) - 1))

    fig = ff.create_choropleth(
        fips=fips, values=values,
        binning_endpoints=endpts,
        # plot_bgcolor='rgb(0, 0, 0)',
        colorscale=colorscale,
        show_state_data=True,
        # county_outline={'color': 'rgb(255, 255, 255)', 'width': .7},
        state_outline={'color': 'rgb(0, 0, 0)', 'width': .5},
        show_hover=True, centroid_marker={'opacity': 0},
        asp=2.9, width=image_width, height=image_height,
        title_text=title
    )

    fig.update_layout(dict(
        margin={'pad': 20, 't': 80, 'b': 200},
        legend={'font': {'size': 30}, 'itemsizing': 'constant'},
        title={'font': {'size': 30}},
        geo={'landcolor': 'white'},
    ))

    fig.add_annotation(
        x=1,
        y=.07,
        showarrow=False,
        font=dict(size=20),
        text="Produced by Bryan Spaulding",
        xref="paper",
        yref="paper"
    )
    fig.add_annotation(
        x=1,
        y=.02,
        showarrow=False,
        font=dict(size=15),
        text="Data source: NY Times",
        xref="paper",
        yref="paper"
    )
    fig.add_annotation(
        x=1,
        y=0,
        showarrow=False,
        font=dict(size=15),
        text="https://www.nytimes.com/interactive/2020/us/coronavirus-us-cases.html",
        xref="paper",
        yref="paper"
    )



    log('Generating %s_%s.png' % (dimension, date))
    fig.write_image('images/%s_%s.png' % (dimension, date))


def gen_all_images(dimension, new_df):
    """Iterate over dataframe and generate associated map images."""
    dates = new_df['date'].unique().tolist()
    for date in dates:
        gen_image(date, dimension, new_df)


def gen_indiv_video(date, dimension):
    ff = ffmpy.FFmpeg(
        inputs={
            'images/cases_pc_2020-04-05.png':
            ['-hide_banner', '-loglevel', 'warning', '-loop', '1']
        },
        outputs={
            'output.mp4': [
                '-c:v', 'libx264', '-t', '1', '-r', '30', '-pix_fmt',
                'yuv420p', '-vf', 'scale=%s:%s' % (image_width, image_height)
            ]
        }
    )
    print(ff.cmd)
    ff.run()


def gen_crossfade_frames(file_1, file_2, dimension, start_num, time):
    ff = ffmpy.FFmpeg(
        inputs={
            file_1:
            ['-hide_banner', '-loglevel', 'warning', '-loop', '1'],
            file_2:
            ['-loop', '1']
        },
        outputs={
            'frames/{dimension}_frame_%05d.png'.format(dimension=dimension):
            ['-start_number', str(start_num), '-filter_complex',
             '[1:v][0:v]blend=all_expr=\'A*(if(gte(T,{crossfade}),1,T/{crossfade}))' \
             '+B*(1-(if(gte(T,{crossfade}),1,T/{crossfade})))\''.format(crossfade=crossfade),
             '-t', str(time)]
        }
    )
    log('Generating crossfade frames from %s to %s' % (file_1, file_2))
    ff.run()


def del_crossfade_frames(dimension):
    log('Removing frames for ' + dimension)
    for f in glob.glob('frames/%s_frame_*.png' % dimension):
        os.remove(f)


def gen_all_crossfade_frames(images, dimension):
    prev_image = None
    for idx, image in enumerate(images):
        file_start = ((idx - 1) * 25 * slide_time) + 1
        if idx == 0:
            prev_image = image
            continue
        gen_crossfade_frames(prev_image, image, dimension, file_start, slide_time)
        prev_image = image


def convert_frames_to_video(dimension):
    # ffmpeg -i frames/frames_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4
    ff = ffmpy.FFmpeg(
        inputs={
            'frames/{dimension}_tl_frame_%05d.png'.format(dimension=dimension):
            ['-hide_banner', '-loglevel', 'warning']
        },
        outputs={
            'videos/%s.mp4' % dimension:
            ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y']
        }
    )
    log('Converting %s frames to video' % dimension)
    ff.run()


def get_images_list(dimension):
    result = []
    for file in os.listdir('images'):
        if file.startswith(dimension + '_20'):
            result.append(os.path.join('images', file))
    result.sort()
    return result


def gen_colorscale():
    red = Color('darkred')
    white = Color('blanchedalmond')
    colors = list(red.range_to(white, 7))
    colors_converted = []
    for color in colors:
        colors_converted.append(color.hex)
    colors_converted.append('#fff')
    return colors_converted[::-1]


def integrate_frame(dim_frame_num, tl_frame_num, dimension):
    frame_file = 'frames/%s_frame_%s.png' % (dimension, dim_frame_num)
    tl_file = 'frames/timeline_%s.png' % tl_frame_num
    final_file = 'frames/%s_tl_frame_%s.png' % (dimension, tl_frame_num)

    log('generating integrated file: ' + final_file)

    img_main = Image.open(frame_file).convert('RGBA')
    img_tl = Image.open(tl_file).convert('RGBA')
    main_w, main_h = img_main.size
    tl_w, tl_h = img_tl.size
    paste_x1 = (main_w / 2) - (tl_w / 2)
    paste_y1 = main_h - tl_h - 30
    img_main.paste(img_tl, (int(paste_x1), int(paste_y1)))
    img_main.save(final_file)


def gen_integrated_frames(images, dimension):
    num_frames = (len(images) - 1) * 50
    for n in range(1, num_frames + 1):
        integrate_frame('%05d' % n, '%05d' % n, dimension)
    integrate_frame('%05d' % num_frames, '%05d' % (num_frames + 1), dimension)


def delete_all_frames():
    log('deleting all frame files')
    for file in glob.glob('frames/*.png'):
        os.remove(file)
    sys.exit(0)


def delete_all_images():
    log('deleting all image files')
    for file in glob.glob('images/*.png'):
        os.remove(file)
    sys.exit(0)


def overlay_full_legend(dimension, df):
    date_df = df[df[dimension] > 0]
    date_df = date_df.groupby('date')[dimension].size().reset_index(name='count')
    max_df = date_df.loc[date_df['count'].idxmax()]
    max_date = max_df['date']

    sel_coordinates = (1684, 80, 1900, 426)

    max_file = 'images/%s_%s.png' % (dimension, max_date)
    max_img = Image.open(max_file).convert('RGBA')
    selection = max_img.crop(sel_coordinates)

    for file in sorted(glob.glob('images/%s_20*.png' % dimension)):
        if file != max_file:
            log('updating legend for ' + file)
            this_file = Image.open(file).convert('RGBA')
            this_file.paste(selection, sel_coordinates)
            this_file.save(file)


def main():
    parser = argparse.ArgumentParser(
        description='Script for processing COVID-19 data and creating video')
    parser.add_argument('--delete-all-frames', action="store_true", default=False,
                        dest='delete_all_frames', help='delete all frame files')
    parser.add_argument('--delete-all-images', action="store_true", default=False,
                        dest='delete_all_images', help='delete all image files')
    args = parser.parse_args()


    if args.delete_all_frames:
        delete_all_frames()

    if args.delete_all_images:
        delete_all_images()


    new_df = gen_data()

    date_list = new_df['date'].unique().tolist()
    timeline.gen_base_timeline_image(date_list)
    timeline.gen_timeline_frames(date_list, 50)

    gen_all_images(metric, new_df)
    overlay_full_legend(metric, new_df)

    images = get_images_list(metric)
    gen_all_crossfade_frames(images, metric)
    gen_integrated_frames(images, metric)
    convert_frames_to_video(metric)


    # gen_image('2020-03-01', 'cases_pc', new_df)
    # gen_image('2020-03-15', 'cases_pc', new_df)
    # gen_image('2020-03-01', 'cases_pc', new_df)
    # gen_image('2020-04-12', 'cases_pc', new_df)

    # integrate_frame('00001', 'cases_pc')


    # TODO: command line arguments to run in different modes, dimensions (whatare these?)
    # TODO: variablize / configuration (blocks will be best for each dimension)
    # ..... https://www.reddit.com/r/learnpython/comments/2hjxk5/ \
    #       whats_the_proper_way_to_use_configparser_across/
    # TODO: eliminate global variables
    # TODO: log file / verbosity (print to screen too)
    # TODO: exception handling
    # TODO: comments
    # TODO: refactor methods, variables to be cleaner, better named
    # TODO: switch to delete image files (all or specific dimension)

    # TODO: incorporate credit for Census datasource
    # TODO: adjust timing (make configuration / cl argument driven)
    # TODO: possible bug in deaths vs. deaths_pc video. seemed to concatenate
    # TODO: delete previous data before run?  after run?
    # TODO: store working data in directory that's not backed up by timemachine

    # TODO: rename this file to something more logical
    # TODO: merge timeline.py into this file
    # TODO: make image / frame / video sizes all driven off same value
    # TODO: argument for "full do-over" mode
    # TODO: build in ability to do truncated run (either starting from x date, or between x and
    #       y dates)
    # TODO: idea: create multi-paned view of COVID stats running on simultaneous timelines
    # TODO: script to temporarily utilize ec2 instance, pushing result to s3


if __name__ == '__main__':
    log('START')
    main()
    log('END')
