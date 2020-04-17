#!/usr/bin/env python

from configparser import ConfigParser, ExtendedInterpolation


CONFIG_FILE = 'covid_video.ini'

def calculate(parser):
    parser.set('default', 'frames_per_day', str(
        25 * parser.getint('default', 'slide_time')))
    return parser

def create_config(config_file=None):
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(CONFIG_FILE)
    parser = calculate(parser)
    return parser


CONFIG = create_config()
