#!/usr/bin/env python

import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
from io import StringIO
import datetime
import os
import ffmpy
from colour import Color
from datetime import datetime


data_file = 'data.csv'
per_capita_unit = 1000000
per_capita_string = '1M'
start_date = '2020-03-01'
upper_end = .85
width = 1920
height = 900
crossfade = 2
metric = 'cases_pc'
title = 'US County COVID-19 New Cases Per Day Per ' + per_capita_string
slide_time = 2


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
		margin={'pad': 20, 't': 80},
		legend={'font': {'size': 30}, 'itemsizing': 'constant'},
		title={'font': {'size': 30}},
		geo={'landcolor': 'white'}

	))

	fig.write_image('images/%s_%s.png' % (dimension, date))


def gen_all_images(dimension, new_df):
	dates = new_df['date'].unique().tolist()
	for date in dates:
		log('Generating image for ' + date)
		gen_image(date, dimension, new_df)


def gen_indiv_video(date, dimension):
	# https://ffmpy.readthedocs.io/en/latest/examples.html#complex-command-lines
	ff = ffmpy.FFmpeg(
		inputs={'images/cases_pc_2020-04-05.png': 
			['-loop', '1']
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
			['-loop', '1'],
			file_2: 
			['-loop', '1']
		},
		outputs={
			'frames/frame_%05d.png': 
			['-start_number', str(start_num), '-filter_complex', '[1:v][0:v]blend=all_expr=\'A*(if(gte(T,{crossfade}),1,T/{crossfade}))+B*(1-(if(gte(T,{crossfade}),1,T/{crossfade})))\''.format(crossfade=crossfade), '-t', str(time)]
		}
	)
	print(ff.cmd)
	ff.run()


def gen_all_crossfade_frames(images, dimension):
	prev_image = None
	images_len = len(images)
	for idx, image in enumerate(images):
		file_start = ((idx - 1) * 25 * slide_time) + 1
		if idx == 0:
			prev_image = image
			continue

		gen_crossfade_frames(prev_image, image, dimension, file_start, slide_time)

		prev_image = image


def convert_frames_to_video(directory, dimension):
	# ffmpeg -i frames/frames_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4
	ff = ffmpy.FFmpeg(
		inputs={
			'frames/frame_%05d.png': 
			None
		},
		outputs={
			'%s/%s.mp4' % (directory, dimension): 
			['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y']
		}
	)
	print(ff.cmd)
	ff.run()


def get_images_list(dimension):
	result = []
	for file in os.listdir('images'):
		if file.startswith(dimension):
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


def jh(data_type):
	csv_file = 'time_series_covid19_%s_US.csv' % data_type

	if os.path.exists(csv_file):
		url = csv_file
		dat_df = pd.read_csv(url, dtype={'FIPS': str})
	else:
		url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/' + csv_file
		dat_df = pd.read_csv(url, dtype={'FIPS': str})
		dat_df.to_csv(csv_file, index=False)

	base_cols = dat_df.columns.tolist()[:11]
	date_cols = dat_df.columns.tolist()[11:]

	new_date_cols = []
	for col in date_cols:
		col_content = col.split('/')
		m = col_content[0]
		d = col_content[1]
		if len(col_content[0]) == 1:
			m = '0' + col_content[0]
		if len(col_content[1]) == 1:
			d = '0' + col_content[1]
		new_date_cols.append('20%s-%s-%s' % (col_content[2], m, d))

	dat_df.columns = base_cols + new_date_cols
	dat_df.rename(columns={'FIPS': 'fips'}, inplace=True)

	print(dat_df)


def main():
	new_df = gen_data()
	gen_all_images('cases_pc', new_df)
	images = get_images_list(metric)
	gen_all_crossfade_frames(images, metric)
	convert_frames_to_video('videos', metric)

	# gen_image('2020-04-01', 'cases_pc', new_df)
	# gen_image('2020-03-15', 'cases_pc', new_df)
	# gen_image('2020-03-01', 'cases_pc', new_df)

	# TODO: metric-based naming convention for frames (refactor all methods)
	# TODO: delete frames after video produced?
	# TODO: configuration
	# TODO: log file / verbosity (print to screen too)
	# TODO: exception handling
	# TODO: consider rounding numbers to make the legend more clean (?)
	# TODO: copy/paste image contents https://kite.com/python/examples/3039/pil-copy-a-region-of-an-image-to-another-area
	# TODO: should be able to get x/y from Gimp
	# TODO: how to pin bottom number at zero - OR, display white in background by default
	# ...... https://community.plotly.com/t/how-to-change-colors-for-na-values-to-gray-in-a-choropleth-map/15746/4
	# TODO: fix the fact that my diff is factoring in negative numbers
	# TODO: link to NY Tracking data
	# TODO: reduce ffmpeg verbosity and log https://superuser.com/questions/326629/how-can-i-make-ffmpeg-be-quieter-less-verbose
	# TODO: intelligence on retrieving new data automatically (i.e., if ~24 hours old)

	# TODO: idea: create multi-paned view of COVID stats running on simultaneous timelines


if __name__ == '__main__':
	log('START')
	main()
	log('END')
