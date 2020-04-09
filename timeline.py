#!/usr/bin/env python


from datetime import datetime, date, timedelta



def gen_date_list(beg, end):
	beg_date = datetime.strptime(beg, '%Y-%m-%d')
	end_date = datetime.strptime(end, '%Y-%m-%d')
	delta = end_date - beg_date

	days_list = []

	for i in range(delta.days + 1):
		this_date = beg_date + timedelta(days=i)
		days_list.append(datetime.strftime(this_date, '%Y-%m-%d'))

	print(days_list)


def gen_timeline(date_list):
	pass


def main():
	gen_date_list('2020-03-01', '2020-04-07')


if __name__ == '__main__':
	main()