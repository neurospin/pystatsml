#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 18:05:38 2018

@author: edouard.duchesnay@gmail.com

./replace.py -i /tmp/brainvol/data/* -p wm -r WM
"""

import os
import os.path
import argparse
import re
import shutil

if __name__ == "__main__":

    # parse command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='list of input files or root directory', nargs='+', type=str)
    parser.add_argument('--backup', action='store_true', help='save backup .bak file')
    parser.add_argument('--noaction', action='store_true', help='dry run')
    parser.add_argument('-p', '--pattern', help='list of input files or root directory', type=str)
    parser.add_argument('-r', '--replacement', help='list of input files or root directory', type=str)

    options = parser.parse_args()

    if options.input is None or options.pattern is None:
        parser.print_help()
        raise SystemExit("Error: files are missing")

    if options.replacement is None :
        options.replacement = ""

    if len(options.input) == 1 and os.path.isdir(options.input[0]):
        filenames = [os.path.join(curdir, file) \
             for curdir, subdirs, files in os.walk(options.input[0]) for file in files]
    else:
        filenames = [f for f in options.input if os.path.isfile(f)]


    regex = re.compile(options.pattern)

    for filename in filenames:
        lines = ""
        touch = False
        with open(filename, 'r') as infile:
            try:
                for line in infile:
                    if len(regex.findall(line)) > 0:
                        touch = True
                        line = regex.sub(options.replacement, line)
                        # print(line)
                    lines += line
            except Exception as e:
                print(filename, ":", e)

        if touch and not options.noaction:
            shutil.copy(filename, filename + ".bak")
            with open(filename, 'w') as f:
                f.write(lines)
