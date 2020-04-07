#!/usr/bin/env python

import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
from io import StringIO
import datetime
import os


data_file = 'data.csv'
per_capita_unit = 1000000
start_date = '2020-03-01'
upper_end = .90


def print_datetime():
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def gen_data():
	if os.path.exists(data_file):
		new_df = pd.read_csv(data_file, dtype={'fips': str, 'cases': int, 'deaths': int})
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

		new_df = new_df[['date', 'fips', 'population', 'county_x', 'state_x', 'cases', 'deaths']]
		new_df.columns = ['date', 'fips', 'population', 'county', 'state', 'cases', 'deaths']
		new_df = new_df.fillna(0)

		new_df['cases'] = new_df['cases'].astype(int)
		new_df['deaths'] = new_df['deaths'].astype(int)

		new_df['cases_new'] = new_df.groupby('county')['cases'].diff().fillna(0)
		new_df['deaths_new'] = new_df.groupby('county')['deaths'].diff().fillna(0)

		new_df['cases_roll'] = new_df.groupby('county')['cases_new'].rolling(3).mean().reset_index(0, drop=True).fillna(0)
		new_df['deaths_roll'] = new_df.groupby('county')['deaths_new'].rolling(3).mean().reset_index(0, drop=True).fillna(0)

		new_df['cases_pc'] = new_df['cases_roll'] / new_df['population'] * per_capita_unit
		new_df['deaths_pc'] = new_df['deaths_roll'] / new_df['population'] * per_capita_unit

		new_df.to_csv('data.csv')

		return new_df

def gen_image(date, dimension, new_df):
	fips = new_df['fips'][new_df['date'] == date].unique().tolist()
	values = new_df[dimension][new_df['date'] == date].tolist()

	colorscale = ['rgb(255,255,255)']
	for color in enumerate(px.colors.sequential.OrRd):
	    colorscale.append(color[1])

	    
	# TODO: probably need to scope this to the results on the peak day
	max_range = new_df[dimension][new_df[dimension] > 0].quantile(upper_end)


	# TODO: is there a way to make beginning = 0 and white? Or how about background gray?
	endpts = list(np.linspace(0.01, max_range, len(colorscale) - 1))


	# TODO: incorporate title and date, including measurement
	# TODO: state lines?
	# TODO: drop county lines but build in state lines once the complete counties are incorporated
	fig = ff.create_choropleth(
	    fips=fips, values=values,
	    binning_endpoints=endpts,
	    colorscale=colorscale,
	    show_state_data=False,
	    county_outline={'color': 'rgb(255, 255, 255)', 'width': 0.5},
	    show_hover=True, centroid_marker={'opacity': 0},
	    asp=2.9, width=3840, height=1700
	)

	fig.update_layout(dict(
		margin={'pad': 6},
		legend = {'font': {'size': 30}, 'itemsizing': 'constant'}
	))

	fig.write_image('images/%s_%s.png' % (dimension, date))


def gen_all_images(dimension, new_df):
	dates = new_df['date'].unique().tolist()
	for date in dates:
		print('Generating image for ' + date)
		gen_image(date, dimension, new_df)


def main():
	new_df = gen_data()
	# gen_image('2020-04-05', 'cases_pc', new_df)
	gen_all_images('cases_pc', new_df)


if __name__ == '__main__':
	print_datetime()
	main()
	print_datetime()
