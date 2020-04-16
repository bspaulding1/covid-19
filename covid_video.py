#!/usr/bin/env python

import os
from shutil import copyfile
import sys
import glob
import datetime
from datetime import datetime, date, timedelta
import requests
import filecmp
from io import StringIO
import argparse
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import ffmpy
from colour import Color
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

from config import CONFIG


vert_font_size = 18
big_date_font = 48
diamond_size = 18
diamond_offset = diamond_size / 2
tl_width = 1800
tl_hash_height = 16
tl_side_buffer = 50
tl_img_width = tl_width + (tl_side_buffer * 2)
tl_img_height = tl_hash_height + (tl_side_buffer * 2)

tl_mid_width = tl_img_width / 2
tl_mid_height = tl_img_height / 4
tl_hash_top = tl_mid_height - (tl_hash_height / 2)
tl_hash_bot = tl_mid_height + (tl_hash_height / 2)
tl_hash_start = tl_side_buffer
tl_hash_end = tl_img_width - tl_side_buffer

tl_base_file = 'images/timeline_base.png'



def log(msg):
    """Log provided message to console along with timestamp."""
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' + msg)


def retrieve_file(url, file_name):
    log('retrieving ' + file_name)
    tmp_file = file_name + '.tmp'
    r = requests.get(url)
    with open(tmp_file, 'wb') as f:
        f.write(r.content)
    if os.path.exists(file_name):
        same_file = filecmp.cmp(file_name, tmp_file)
        os.remove(file_name)
    else:
        same_file = False
    os.rename(tmp_file, file_name)
    if same_file: 
        log(file_name + ' has not changed')
        return True

    log(file_name + ' updated')
    return False


def gen_data():
    """Retrieve data from datasources and merge into dataframe."""

    nytimes_retrieve = retrieve_file(
        CONFIG.get('default', 'nytimes_csv_url'),
        CONFIG.get('default', 'nytimes_file_name'),
    )

    census_retrieve = retrieve_file(
        CONFIG.get('default', 'census_csv_url'),
        CONFIG.get('default', 'census_file_name'),
    )

    if os.path.exists(CONFIG.get('default', 'data_file')) and nytimes_retrieve and census_retrieve:
        log('data file exists and is current')
        new_df = pd.read_csv(CONFIG.get('default', 'data_file'), dtype={'fips': str})
        return

    log('creating base pandas dataframe')
    dat_df = pd.read_csv(
        CONFIG.get('default', 'nytimes_file_name'), dtype={'fips': str})

    dates = dat_df['date'].unique().tolist()

    log('creating population dataframe')
    pop_df = pd.read_csv(
        CONFIG.get('default', 'census_file_name'), encoding='ISO-8859-1',
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

    per_capita_unit = int(CONFIG.get('default', 'per_capita_unit'))
    new_df['cases_pc'] = new_df['cases_roll'] / new_df['population'] * per_capita_unit
    new_df['deaths_pc'] = new_df['deaths_roll'] / new_df['population'] * per_capita_unit

    log('saving merged dataframe')
    new_df.to_csv('data.csv', index=False)

    log('done processing dataframe')


def get_df_slice(start_date=None, end_date=None):
    df = pd.read_csv(
        CONFIG.get('default', 'data_file'), dtype={'fips': str})
    if start_date is None:
        df = df[df['date'] >= CONFIG.get('default', 'start_date')]
    else:
        df = df[df['date'] >= start_date]
    if end_date is not None:
        df = df[df['date'] <= end_date]
    return df


def gen_image(date, metric, new_df):
    """Create map image for specific date and metric."""
    fips = new_df['fips'][new_df['date'] == date].unique().tolist()
    values = new_df[metric][new_df['date'] == date].tolist()

    colorscale = gen_colorscale()

    max_range = new_df[metric][new_df[metric] > 0].quantile(
        float(CONFIG.get('default', 'upper_end')))
    endpts = list(np.linspace(0.01, max_range, len(colorscale) - 1))

    fig = ff.create_choropleth(
        fips=fips, values=values,
        binning_endpoints=endpts,
        colorscale=colorscale,
        show_state_data=True,
        state_outline={'color': 'rgb(0, 0, 0)', 'width': .5},
        show_hover=True, centroid_marker={'opacity': 0},
        asp=2.9, width=CONFIG.getint('default', 'image_width'), height=CONFIG.getint('default', 'image_height'),
        title_text=CONFIG.get(metric, 'slide_title')
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



    fig.write_image('images/%s_%s.png' % (metric, date))


def gen_all_images(metric, df):
    """Iterate over dataframe and generate associated map images."""
    date_list = df['date'].unique().tolist()
    with tqdm(total=len(date_list)) as pbar:
        for date_string in date_list:
            pbar.set_description(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' +
                                 'generating main map images')
            gen_image(date_string, metric, df)
            pbar.update(1)
    overlay_full_legend(metric, df)


def gen_crossfade_frames(file_1, file_2, metric, start_num, time):
    ff = ffmpy.FFmpeg(
        inputs={
            file_1:
            ['-hide_banner', '-loglevel', 'warning', '-loop', '1'],
            file_2:
            ['-loop', '1']
        },
        outputs={
            'frames/{metric}_frame_%05d.png'.format(metric=metric):
            ['-start_number', str(start_num), '-filter_complex',
             '[1:v][0:v]blend=all_expr=\'A*(if(gte(T,{crossfade}),1,T/{crossfade}))' \
             '+B*(1-(if(gte(T,{crossfade}),1,T/{crossfade})))\''.format(
                 crossfade=CONFIG.get('default', 'crossfade')),
             '-t', str(time)]
        }
    )
    ff.run()


def del_crossfade_frames(metric):
    log('Removing frames for ' + metric)
    for f in glob.glob('frames/%s_frame_*.png' % metric):
        os.remove(f)


def gen_all_crossfade_frames(metric):
    images_list = get_images_list(metric)
    prev_image = None
    frames_per_day = CONFIG.getint('default', 'frames_per_day')
    slide_time = CONFIG.getint('default', 'slide_time')
    total_frames = frames_per_day * (len(images_list) - 1)
    with tqdm(total=len(images_list) - 1) as pbar:
        for idx, image in enumerate(images_list):
            pbar.set_description(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' +
                                 'generating crossfade frames')
            file_start = ((idx - 1) * frames_per_day) + 1
            if idx == 0:
                prev_image = image
                continue
            gen_crossfade_frames(prev_image, image, metric, file_start, slide_time)
            prev_image = image
            pbar.update(1)
    extra_frame_num = total_frames + 1
    copyfile(
        'frames/%s_frame_%05d.png' % (metric, total_frames),
        'frames/%s_frame_%05d.png' % (metric, extra_frame_num)
    )


def convert_frames_to_video(metric):
    ff = ffmpy.FFmpeg(
        inputs={
            'frames/{metric}_tl_frame_%05d.png'.format(metric=metric):
            ['-hide_banner', '-loglevel', 'warning']
        },
        outputs={
            'videos/%s.mp4' % metric:
            ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y']
        }
    )
    log('Converting %s frames to video' % metric)
    ff.run()


def get_images_list(metric):
    result = []
    for file in os.listdir('images'):
        if file.startswith(metric + '_20'):
            result.append(os.path.join('images', file))
    result.sort()
    return result


def gen_colorscale():
    large_color = Color(CONFIG.get('default', 'large_color'))
    small_color = Color(CONFIG.get('default', 'small_color'))
    colors = list(large_color.range_to(small_color, 7))
    colors_converted = []
    for color in colors:
        colors_converted.append(color.hex)
    colors_converted.append('#fff')
    return colors_converted[::-1]


def integrate_frame(dim_frame_num, tl_frame_num, metric):
    frame_file = 'frames/%s_frame_%s.png' % (metric, dim_frame_num)
    tl_file = 'frames/timeline_%s.png' % tl_frame_num
    final_file = 'frames/%s_tl_frame_%s.png' % (metric, tl_frame_num)

    img_main = Image.open(frame_file).convert('RGBA')
    img_tl = Image.open(tl_file).convert('RGBA')
    main_w, main_h = img_main.size
    tl_w, tl_h = img_tl.size
    paste_x1 = (main_w / 2) - (tl_w / 2)
    paste_y1 = main_h - tl_h - 30
    img_main.paste(img_tl, (int(paste_x1), int(paste_y1)))
    img_main.save(final_file)


def gen_integrated_frames(metric):
    images = get_images_list(metric)
    num_frames = (len(images) - 1) * 50 + 1
    with tqdm(total=num_frames) as pbar:
        for n in range(1, num_frames + 1):
            pbar.set_description(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' +
                                 'generating integrated frames')
            pbar.update(1)
            integrate_frame('%05d' % n, '%05d' % n, metric)


def delete_all_frames():
    log('deleting all frame files')
    for file in glob.glob('frames/*.png'):
        os.remove(file)


def delete_all_images():
    log('deleting all image files')
    for file in glob.glob('images/*.png'):
        os.remove(file)


def overlay_full_legend(metric, df):
    date_df = df[df[metric] > 0]
    date_df = date_df.groupby('date')[metric].size().reset_index(name='count')
    max_df = date_df.loc[date_df['count'].idxmax()]
    max_date = max_df['date']

    sel_coordinates = (
        int(CONFIG.get('default', 'legend_x1')),
        int(CONFIG.get('default', 'legend_y1')),
        int(CONFIG.get('default', 'legend_x2')),
        int(CONFIG.get('default', 'legend_y2'))
    )

    max_file = 'images/%s_%s.png' % (metric, max_date)
    max_img = Image.open(max_file).convert('RGBA')
    selection = max_img.crop(sel_coordinates)

    file_list = sorted(glob.glob('images/%s_20*.png' % metric))
    with tqdm(total=len(file_list) - 1) as pbar:
        for idx, file in enumerate(file_list):
            if file != max_file:
                pbar.set_description(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' +
                                     'updating map image legends')
                pbar.update(1)
                this_file = Image.open(file).convert('RGBA')
                this_file.paste(selection, sel_coordinates)
                this_file.save(file)


def valid_date(s):
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def gen_date_list(beg, end):
    beg_date = datetime.strptime(beg, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')
    delta = end_date - beg_date

    days_list = []

    for i in range(delta.days + 1):
        this_date = beg_date + timedelta(days=i)
        days_list.append(this_date)

    return days_list


def gen_date_string_list(date_list):
    date_string_list = []

    for d in date_list:
        date_string_list.append(datetime.strftime(d, '%Y-%m-%d'))

    return date_string_list


def gen_timeline_frame(date_string, idx, tot_frames):
    frame_number = idx + 1
    file_name = 'timeline_%05d.png' % frame_number

    percent_complete = idx / tot_frames
    x_position = tl_hash_start + (tl_width * percent_complete)

    img = Image.open(tl_base_file).convert('RGBA')
    img_f = ImageFont.truetype(
        font='Courier', size=big_date_font, index=0, encoding='')
    draw = ImageDraw.Draw(img)

    draw.polygon([x_position, tl_mid_height - diamond_offset, x_position - diamond_offset, tl_mid_height, x_position,
               tl_mid_height + diamond_offset, x_position + diamond_offset, tl_mid_height], outline=(246, 49, 0), fill=(246, 49, 0))

    txt_w, txt_h = draw.textsize(date_string, font=img_f)
    draw.text((tl_img_width / 2 - txt_w / 2, tl_img_height - 10 - txt_h),
              date_string, font=img_f, fill=(246, 49, 0))

    img.save('frames/' + file_name)


def gen_timeline_frames(df):
    interval = CONFIG.getint('default', 'frames_per_day')
    date_list = df['date'].unique().tolist()
    gen_base_timeline_image(date_list)
    tot_frames = (len(date_list) - 1) * interval
    with tqdm(total=tot_frames) as pbar:
        for idx, date_string in enumerate(date_list):
            if idx == len(date_list) - 1:
                gen_timeline_frame(date_string, tot_frames, tot_frames)
                break
            for n in range((idx * interval), (idx * interval) + interval):
                gen_timeline_frame(date_string, n, tot_frames)
                pbar.set_description(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' + 
                    'generating timeline frames')
                pbar.update(1)


def gen_base_timeline_image(date_list):
    log('generating base timeline image')

    num_hashes = len(date_list)
    hash_space = tl_width / (num_hashes - 1)

    f = ImageFont.truetype(font='Courier', size=vert_font_size, index=0, encoding='')
    img_txt = Image.new('L', (tl_img_height, tl_img_width))
    img_dra = ImageDraw.Draw(img_txt)
    img_dra.text((10, tl_hash_start - (vert_font_size / 2) + 1),
                 date_list[0].replace('2020-', ''), font=f, fill=255)
    img_dra.text((10, tl_hash_start + (hash_space * (num_hashes - 1)) - (vert_font_size / 2) + 1),
                 date_list[-1].replace('2020-', ''), font=f, fill=255)
    w = img_txt.rotate(90,  expand=1)

    img = Image.new('RGB', (tl_img_width, tl_img_height), color='white')
    draw = ImageDraw.Draw(img)
    draw.line((tl_hash_start, tl_mid_height, tl_hash_end,
            tl_mid_height), fill=(0, 0, 0), width=2)
    for x in range(0, num_hashes):
        spot = tl_hash_start + (hash_space * x)
        spot = int(spot)
        draw.line((spot, tl_hash_top, spot, tl_hash_bot),
                  fill=(0, 0, 0), width=2)

    img.paste(ImageOps.colorize(w, (0, 0, 0), (0, 0, 0)), (0, 0),  w)

    img.save('images/timeline_base.png')



def main():

    metrics_list = CONFIG.get('default', 'metrics_list').split(',')

    parser = argparse.ArgumentParser(
        description='Script for processing COVID-19 data and creating video')
    parser.add_argument('--delete-all-frames', action="store_true", default=False,
                        dest='delete_all_frames', help='delete all frame files')
    parser.add_argument('--delete-all-images', action="store_true", default=False,
                        dest='delete_all_images', help='delete all image files')
    parser.add_argument('-b', default=None, dest='begin_date', type=valid_date,
                        help='start date of date range - format YYYY-MM-DD')
    parser.add_argument('-e', default=None, dest='end_date', type=valid_date,
                        help='end date of date range - format YYYY-MM-DD')
    parser.add_argument('-i', default=None, dest='specific_date', type=valid_date,
                        help='create image for specific date - format YYYY-MM-DD')
    parser.add_argument('-m', default=None, dest='metric', choices=metrics_list,
                        help='specify which metric (default: cases_pc)')
    args = parser.parse_args()
    # -f for frame
    # -d for deleting of some ilk
    # -y for auto-confirming
    # -r refresh data calculations
    # -o output file path + name
    # -l leave working files in place

    quick_exit = False

    if args.metric:
        metric = args.metric
    else:
        metric = CONFIG.get('default', 'metric')

    if args.delete_all_frames or args.delete_all_images or args.specific_date:
        quick_exit = True

    if args.delete_all_frames:
        delete_all_frames()

    if args.delete_all_images:
        delete_all_images()

    if args.specific_date:
        gen_data()
        vid_df = get_df_slice(start_date=args.start_date, end_date=args.end_date)
        gen_image(args.specific_date, metric, vid_df)

    if not quick_exit:
        gen_data()
        vid_df = get_df_slice(start_date=args.start_date, end_date=args.end_date)
        gen_timeline_frames(vid_df)
        gen_all_images(metric, vid_df)
        gen_all_crossfade_frames(metric)
        gen_integrated_frames(metric)
        convert_frames_to_video(metric)

    # integrate_frame('00001', 'cases_pc')


    # TODO: log file / verbosity (print to screen too)
    # TODO: exception handling
    # TODO: comments
    # TODO: refactor methods, variables to be cleaner, better named, more flexibly executed
    # TODO: switch to delete image files (all or specific metric)
    # TODO: sanity check my calculations (daily new events per fips; colorscale ranges)
    # TODO: possible bug in deaths vs. deaths_pc video. seemed to concatenate
    # TODO: Purge unused methods
    # TODO: Identify and run various Python code quality static analyzers

    # TODO: incorporate credit for Census datasource into image display
    # TODO: migrate variables to config file
    # TODO: make image / frame / video sizes all driven off same values
    # TODO: confirm that changing configs work: speed, metric
    # TODO: organize config file sections
    # TODO: delete previous data before run?  after run?
    # TODO: store working data in directory that's not backed up by timemachine
    # TODO: create 'data' directory and store csv's there
    # TODO: prompt for confirmation if no data has changed and no options have been set (-y override)
    # TODO: style progress bars https://pypi.org/project/tqdm/#parameters
    #       https://github.com/tqdm/tqdm/issues/585

    # TODO: idea: create multi-paned view of COVID stats running on simultaneous timelines
    # TODO: script to temporarily utilize ec2 instance, pushing result to s3
    # TODO: prompt for confirmation whether or not to run if data hasn't changed

    # TODO: Configuration &/or CLI Argument options [VISUALLY MAP METHOD INTERDEPENDENCIES]
    # * ability to keep or delete all working files *
    # * integrate frame for specific frame number *
    # * refresh source data &/or calculated data *
    # * console logging verbosity *
    # * video output file name / location *
    # * full do-over mode *
    # * force run flag (but otherwise prompt if no change to data) *
    # * image dimension (downstream variables!)
    # * transition timing (downstream variables! > frames)
    # * upper binendings (sp?) quantile
    # * temp storage directory for images, frames
    # * base file names / patterns (?)
    # * clean up working files after done OR leave 


if __name__ == '__main__':
    log('START')
    main()
    log('END')
