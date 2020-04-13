#!/usr/bin/env python

import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
from io import StringIO
import datetime
import os
import glob
import ffmpy
from colour import Color
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageOps

import timeline


data_file = 'data.csv'
upper_end = .85
width = 1920
height = 1100
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
	print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' + msg)


def gen_data():
	if os.path.exists(data_file):
		new_df = pd.read_csv(data_file, dtype={'fips': str})
		return new_df
	else:
		url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
		dat_df = pd.read_csv(url, dtype={'fips': str})

		dat_df = dat_df[dat_df.date >= start_date]

		dates = dat_df['date'].unique().tolist()

		url = 'https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv'
		pop_df = pd.read_csv(url, encoding='ISO-8859-1', dtype={'SUBLEV': str, 'REGION': str, 'DIVISION': str, 'STATE': str, 'COUNTY': str})


		pop_df = pop_df[pop_df['COUNTY'] != '000']
		pop_df['fips'] = pop_df['STATE'] + pop_df['COUNTY']
		pop_df = pop_df[['fips', 'POPESTIMATE2019', 'CTYNAME', 'STNAME']]
		pop_df['CTYNAME'] = pop_df['CTYNAME'].apply(lambda x: x.replace(' County', ''))
		pop_df.columns = ['fips', 'population', 'county', 'state']


		csv = 'date,fips,population,county,state\n'
		for date in dates:
			for idx, row in pop_df.iterrows():
				csv += '%s,%s,%s,"%s","%s"\n' % (date, row['fips'], row['population'], row['county'], row['state'])

		new_df = pd.read_csv(StringIO(csv), dtype={'fips': str})
		new_df = new_df.merge(dat_df, how='left', left_on=['date', 'fips'], right_on=['date', 'fips'])

		# new_df = dat_df.join(pop_df.set_index('fips'), on='fips', lsuffix='_x', rsuffix='_y')

		new_df = new_df[['date', 'fips', 'population', 'county_x', 'state_x', 'cases', 'deaths']]
		new_df.columns = ['date', 'fips', 'population', 'county', 'state', 'cases', 'deaths']
		new_df = new_df.fillna(0)

		new_df['cases'] = new_df['cases'].astype(int)
		new_df['deaths'] = new_df['deaths'].astype(int)

		new_df['cases_new'] = new_df.groupby(['fips'])['cases'].diff().fillna(0)
		new_df['deaths_new'] = new_df.groupby('fips')['deaths'].diff().fillna(0)

		new_df['cases_new'] = new_df['cases_new'].apply(lambda x: 0 if x < 0 else x)
		new_df['deaths_new'] = new_df['deaths_new'].apply(lambda x: 0 if x < 0 else x)

		new_df['cases_roll'] = new_df.groupby('fips')['cases_new'].rolling(3).mean().reset_index(0, drop=True).fillna(0)
		new_df['deaths_roll'] = new_df.groupby('fips')['deaths_new'].rolling(3).mean().reset_index(0, drop=True).fillna(0)

		new_df['cases_pc'] = new_df['cases_roll'] / new_df['population'] * per_capita_unit
		new_df['deaths_pc'] = new_df['deaths_roll'] / new_df['population'] * per_capita_unit


		new_df.to_csv('data.csv', index=False)

		return new_df


def gen_image(date, dimension, new_df):
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
	    asp=2.9, width=width, height=height,
	    title_text=title
	)

	fig.update_layout(dict(
		margin={'pad': 20, 't': 80, 'b': 200},
		legend={'font': {'size': 30}, 'itemsizing': 'constant'},
		title={'font': {'size': 30}},
		geo={'landcolor': 'white'}

	))

	log('Generating %s_%s.png' % (dimension, date))
	fig.write_image('images/%s_%s.png' % (dimension, date))


def gen_all_images(dimension, new_df):
	dates = new_df['date'].unique().tolist()
	for date in dates:
		gen_image(date, dimension, new_df)


def gen_indiv_video(date, dimension):
	# https://ffmpy.readthedocs.io/en/latest/examples.html#complex-command-lines
	ff = ffmpy.FFmpeg(
		inputs={'images/cases_pc_2020-04-05.png': 
			['-hide_banner', '-loglevel', 'warning', '-loop', '1']
		},
		outputs={'output.mp4': 
			['-c:v', 'libx264', '-t', '1', '-r', '30', '-pix_fmt', 'yuv420p', '-vf', 'scale=%s:%s' % (width, height)]
		}
	)
	print(ff.cmd)
	ff.run()

# 
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
			['-start_number', str(start_num), '-filter_complex', '[1:v][0:v]blend=all_expr=\'A*(if(gte(T,{crossfade}),1,T/{crossfade}))+B*(1-(if(gte(T,{crossfade}),1,T/{crossfade})))\''.format(crossfade=crossfade), '-t', str(time)]
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
	images_len = len(images)
	for idx, image in enumerate(images):
		# print(prev_image, image, idx)
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
	white = Color('cornsilk')
	colors = list(red.range_to(white,14))
	colors_converted = []
	for color in colors:
		colors_converted.append(color.hex)
	colors_converted.append('#fff')
	return colors_converted[::-1]


def integrate_frame(frame_num, dimension):
	frame_file = 'frames/%s_frame_%s.png' % (dimension, frame_num)
	tl_file = 'frames/timeline_%s.png' % frame_num
	final_file = 'frames/%s_tl_frame_%s.png' % (dimension, frame_num)

	log('generating integrated file: ' + final_file)

	img_main = Image.open(frame_file).convert('RGBA')
	img_tl = Image.open(tl_file).convert('RGBA')
	# TODO: calculate image sizes and placement
	main_w, main_h = img_main.size
	tl_w, tl_h = img_tl.size
	paste_x1 = (main_w / 2) - (tl_w / 2)
	paste_y1 = main_h - tl_h - 30
	img_main.paste(img_tl, (int(paste_x1), int(paste_y1)))
	img_main.save(final_file)


def gen_integrated_frames(images, dimension):
	num_frames = (len(images) - 1) * 50
	for n in range(1, num_frames + 1):
		integrate_frame('%05d' % n, dimension)


def main():
	new_df = gen_data()

	date_list = new_df['date'].unique().tolist()
	timeline.gen_base_timeline_image(date_list)
	timeline.gen_timeline_frames(date_list, 50)

	gen_all_images(metric, new_df)

	images = get_images_list(metric)
	gen_all_crossfade_frames(images, metric)
	gen_integrated_frames(images, metric)
	convert_frames_to_video(metric)
	# del_crossfade_frames(metric)


	# gen_image('2020-03-01', 'cases_pc', new_df)
	# gen_image('2020-03-15', 'cases_pc', new_df)
	# gen_image('2020-03-01', 'cases_pc', new_df)

	# integrate_frame('00001', 'cases_pc')

	# TODO: possible bug in deaths vs. deaths_pc video. seemed to concatenate.

	# TODO: variablize / configuration (blocks will be best for each dimension)
	# TODO: command line arguments (dimension, dates, etc.)
	# TODO: log file / verbosity (print to screen too)
	# TODO: exception handling
	# TODO: comments
	# TODO: refactor methods to be cleaner, better named
	# TODO: switch to delete image files (all or specific dimension)

	# TODO: intelligence on retrieving new data automatically (i.e., if ~24 hours old)
	# TODO: build in ability to do truncated run (either starting from x date, or between x and y dates)
	# TODO: build in a method to diff daily stats, and only generate images / frames for deltas (of course overall config has to have stayed constant)
	# TODO: argument for "full do-over" mode
	# TODO: consider rounding numbers to make the legend more clean (?)
	# TODO: copy/paste image contents https://kite.com/python/examples/3039/pil-copy-a-region-of-an-image-to-another-area
	# ..... x: 1644, y: 79, w:257, h: 642
	# ..... if i can programmatically place legend, I can also effectively control section I copy/paste (to mitigate resizing in the future)
	# TODO: timeling along the bottom
	# ..... https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/timeline.html
	# ..... https://stackoverflow.com/questions/44951911/plot-a-binary-timeline-in-matplotlib
	# TODO: make image / frame / video sizes all driven off same value
	# TODO: weave another color into the scale, and increase the number of bands in order to show more nuance

	# TODO: link to NY Tracking data

	# TODO: idea: create multi-paned view of COVID stats running on simultaneous timelines
	pass


if __name__ == '__main__':
	log('START')
	main()
	log('END')
