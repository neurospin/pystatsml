# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:19:07 2016

@author: edouard.duchesnay@cea.fr
"""
from __future__ import print_function
import sys, os, argparse

doc_tag = "'''"
skip_tag = '## SKIP'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input python file')

    options = parser.parse_args()

    if not options.input:
        print >> sys.stderr, 'Required input file'
        sys.exit(os.EX_USAGE)
    input_filename = options.input
    #input_filename = "/home/ed203246/git/pylearn-doc/src/tools_numpy.py"
    output_filename = os.path.splitext(input_filename)[0] + ".rst"
    input_fd = open(input_filename, 'r')
    output_fd = open(output_filename, 'w')
    
    #line_in = '## Pandas data manipulation'
    code_block = True
    skip = False
    for line_in in input_fd:
        #print(line_in)
        ## Switch state
        if skip_tag in line_in:
            skip = not skip
            continue
        if skip:
            continue
        if doc_tag in line_in and not code_block:  # end doc start code block
            code_block = True
            output_fd.write('\n') # write new line instead of doc_tag
            #line_in = line_in.replace(doc_tag, '')
            output_fd.write('.. code:: python\n')
            continue
        elif doc_tag in line_in and code_block:  # start doc end code block
            code_block = False
            line_in = line_in.replace(doc_tag, '')

        if code_block:
            output_fd.write('    ' + line_in)
        else:
            output_fd.write(line_in)

    input_fd.close()
    output_fd.close()
