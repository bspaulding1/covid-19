#!/usr/bin/env python

from datetime import datetime, date, timedelta
from PIL import Image, ImageDraw, ImageFont, ImageOps


font_size = 18
big_font = 48
ellipse_diam = 18
ellipse_offset = ellipse_diam / 2
line_width = 1800
hash_height = 16
side_buffer = 50
img_width = line_width + (side_buffer * 2)
img_height = hash_height + (side_buffer * 2)

mid_width = img_width / 2
mid_height = img_height / 4
hash_top = mid_height - (hash_height / 2)
hash_bot = mid_height + (hash_height / 2)
hash_start = side_buffer
hash_end = img_width - side_buffer

timeline_base_file = 'images/timeline_base.png'


def log(msg):
	print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' + msg)


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


def gen_timeline_frame(date_string, frame_number, tot_frames):
	file_name = 'timeline_%05d.png' % frame_number
	log('generating ' + file_name)

	percent_complete = frame_number / tot_frames
	x_position = hash_start + (line_width * percent_complete)

	# TODO: change to load timeline_base_file
	img = Image.open(timeline_base_file).convert('RGBA')
	img_f = ImageFont.truetype(font='Courier', size=big_font, index=0, encoding='')
	draw = ImageDraw.Draw(img)

	draw.polygon([x_position, mid_height - ellipse_offset, x_position - ellipse_offset, mid_height, x_position, mid_height + ellipse_offset, x_position + ellipse_offset, mid_height], outline=(255,0,0), fill=(255,0,0))

	txt_w, txt_h = draw.textsize(date_string, font=img_f)
	draw.text((img_width / 2 - txt_w / 2, img_height - 10 - txt_h), date_string, font=img_f, fill=(255,0,0))

	img.save('frames/' + file_name)


def gen_timeline_frames(date_string_list, interval):
	tot_frames = (len(date_string_list) - 1) * interval
	for idx, date_string in enumerate(date_string_list):
		for n in range((idx * interval) + 1, (idx * interval) + interval + 1):
			gen_timeline_frame(date_string, n, tot_frames)


def gen_base_timeline_image(date_string_list):
	log('generating base timeline image')

	num_hashes = len(date_string_list) - 1
	hash_space = line_width / (num_hashes - 1)

	f = ImageFont.truetype(font='Courier', size=font_size, index=0, encoding='')
	img_txt = Image.new('L', (img_height, img_width))
	img_dra = ImageDraw.Draw(img_txt)
	img_dra.text((10, hash_start - (font_size / 2) + 1), date_string_list[0].replace('2020-', ''), font=f, fill=255)
	img_dra.text((10, hash_start + (hash_space * (num_hashes - 1)) - (font_size / 2) + 1), date_string_list[-1].replace('2020-', ''), font=f, fill=255)
	w=img_txt.rotate(90,  expand=1)

 
	img = Image.new('RGB', (img_width, img_height), color = 'white')
	img_f = ImageFont.truetype(font='Courier', size=big_font, index=0, encoding='')
	draw = ImageDraw.Draw(img)
	draw.line((hash_start, mid_height, hash_end, mid_height), fill=(0,0,0), width=2)
	for x in range(0, num_hashes):
		spot = hash_start + (hash_space * x)
		spot = int(spot)
		draw.line((spot, hash_top, spot, hash_bot), fill=(0,0,0), width=2)

	img.paste(ImageOps.colorize(w, (0,0,0), (0,0,0)), (0,0),  w)

	img.save('images/timeline_base.png')



def main():
	date_list = gen_date_list('2020-03-01', '2020-04-07')
	# print(date_list)
	date_string_list = gen_date_string_list(date_list)
	# print(date_string_list)
	gen_base_timeline_image(date_string_list)
	gen_timeline_frames(date_string_list, 50)
	# gen_image(date_string_list)
	# date_iter(date_string_list, 50)
	# gen_timeline(date_list)

	# TODO: can probably print diamond as a function of percentage complete along line


if __name__ == '__main__':
	main()