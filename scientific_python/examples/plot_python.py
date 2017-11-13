"""
Python language
===============

**Source** Kevin Markham https://github.com/justmarkham/python-reference

"""


######################################################################
# Set up your programming environment using Anaconda
# --------------------------------------------------
#
# 1. Download anaconda (Python 3.x) http://continuum.io/downloads
#
# 2. Install it, on Linux:
#
# ::
#
#     bash Anaconda3-2.4.1-Linux-x86_64.sh
#
# 3. Add anaconda path in your PATH variable in your ``.bashrc`` file:
#
# ::
#
#     export PATH="${HOME}/anaconda3/bin:$PATH"
#
# 4. Optional: install additionnal packages:
#
# Using ``conda``:
#
# ::
#
#     conda install seaborn
#
# Using ``pip``:
#
# ::
#
#     pip install -U --user seaborn
#
# Optional:
#
# ::
#
#     pip install -U --user nibabel
#     pip install -U --user nilearn
#
# 5. Python editor ``spyder``:
#
#    -  Consoles/Open IPython console.
#    -  Left panel text editor
#    -  Right panel ipython console
#    -  F9 run selection or curent line (in recent version of spyder)
#
# 6. Python interpreter: one can use either ``python`` or ``ipython``,
#    with later being an interpretive shell within python that provides
#    additional features: https://docs.python.org/3/tutorial/ &
#    http://ipython.readthedocs.io/en/stable/interactive/tutorial.html.
#


######################################################################
# Import libraries
# ----------------
#

# 'generic import' of math module
import math
math.sqrt(25)

# import a function
from math import sqrt
sqrt(25)    # no longer have to reference the module

# import multiple functions at once
from math import cos, floor

# import all functions in a module (generally discouraged)
# from os import *

# define an alias
import numpy as np

# show all functions in math module
content = dir(math)


######################################################################
# Data types
# ----------
#

# determine the type of an object
type(2)         # returns 'int'
type(2.0)       # returns 'float'
type('two')     # returns 'str'
type(True)      # returns 'bool'
type(None)      # returns 'NoneType'

# check if an object is of a given type
isinstance(2.0, int)            # returns False
isinstance(2.0, (int, float))   # returns True

# convert an object to a given type
float(2)
int(2.9)
str(2.9)

# zero, None, and empty containers are converted to False
bool(0)
bool(None)
bool('')    # empty string
bool([])    # empty list
bool({})    # empty dictionary

# non-empty containers and non-zeros are converted to True
bool(2)
bool('two')
bool([2])


######################################################################
# Math
# ----
#

# basic operations
10 + 4          # add (returns 14)
10 - 4          # subtract (returns 6)
10 * 4          # multiply (returns 40)
10 ** 4         # exponent (returns 10000)
10 / 4          # divide (returns 2 because both types are 'int')
10 / float(4)   # divide (returns 2.5)
5 % 4           # modulo (returns 1) - also known as the remainder

10 / 4          # true division (returns 2.5)
10 // 4         # floor division (returns 2)


######################################################################
# Comparisons and boolean operations
# ----------------------------------
#

# comparisons (these return True)
5 > 3
5 >= 3
5 != 3
5 == 5

# boolean operations (these return True)
5 > 3 and 6 > 3
5 > 3 or 5 < 3
not False
False or not False and True     # evaluation order: not, and, or


######################################################################
# Conditional statements
# ----------------------
#

x = 3
# if statement
if x > 0:
    print('positive')

# if/else statement
if x > 0:
    print('positive')
else:
    print('zero or negative')

# if/elif/else statement
if x > 0:
    print('positive')
elif x == 0:
    print('zero')
else:
    print('negative')

# single-line if statement (sometimes discouraged)
if x > 0: print('positive')

# single-line if/else statement (sometimes discouraged)
# known as a 'ternary operator'
'positive' if x > 0 else 'zero or negative'


######################################################################
# Lists
# -----
#
# Different objects categorized along a certain ordered sequence, lists
# are ordered, iterable, mutable (adding or removing objects changes the
# list size), can contain multiple data types
#


# create an empty list (two ways)
empty_list = []
empty_list = list()

# create a list
simpsons = ['homer', 'marge', 'bart']

# examine a list
simpsons[0]     # print element 0 ('homer')
len(simpsons)   # returns the length (3)

# modify a list (does not return the list)
simpsons.append('lisa')                 # append element to end
simpsons.extend(['itchy', 'scratchy'])  # append multiple elements to end
simpsons.insert(0, 'maggie')            # insert element at index 0 (shifts everything right)
simpsons.remove('bart')                 # searches for first instance and removes it
simpsons.pop(0)                         # removes element 0 and returns it
del simpsons[0]                         # removes element 0 (does not return it)
simpsons[0] = 'krusty'                  # replace element 0

# concatenate lists (slower than 'extend' method)
neighbors = simpsons + ['ned','rod','todd']

# find elements in a list
simpsons.count('lisa')      # counts the number of instances
simpsons.index('itchy')     # returns index of first instance

# list slicing [start:end:stride]
weekdays = ['mon','tues','wed','thurs','fri']
weekdays[0]         # element 0
weekdays[0:3]       # elements 0, 1, 2
weekdays[:3]        # elements 0, 1, 2
weekdays[3:]        # elements 3, 4
weekdays[-1]        # last element (element 4)
weekdays[::2]       # every 2nd element (0, 2, 4)
weekdays[::-1]      # backwards (4, 3, 2, 1, 0)

# alternative method for returning the list backwards
list(reversed(weekdays))

# sort a list in place (modifies but does not return the list)
simpsons.sort()
simpsons.sort(reverse=True)     # sort in reverse
simpsons.sort(key=len)          # sort by a key

# return a sorted list (but does not modify the original list)
sorted(simpsons)
sorted(simpsons, reverse=True)
sorted(simpsons, key=len)

# create a second reference to the same list
num = [1, 2, 3]
same_num = num
same_num[0] = 0         # modifies both 'num' and 'same_num'

# copy a list (three ways)
new_num = num.copy()
new_num = num[:]
new_num = list(num)

# examine objects
id(num) == id(same_num) # returns True
id(num) == id(new_num)  # returns False
num is same_num         # returns True
num is new_num          # returns False
num == same_num         # returns True
num == new_num          # returns True (their contents are equivalent)

# conatenate +, replicate *
[1, 2, 3] + [4, 5, 6]
["a"] * 2 + ["b"] * 3


######################################################################
# Tuples
# ------
#
# Like lists, but their size cannot change: ordered, iterable, immutable,
# can contain multiple data types
#

# create a tuple
digits = (0, 1, 'two')          # create a tuple directly
digits = tuple([0, 1, 'two'])   # create a tuple from a list
zero = (0,)                     # trailing comma is required to indicate it's a tuple

# examine a tuple
digits[2]           # returns 'two'
len(digits)         # returns 3
digits.count(0)     # counts the number of instances of that value (1)
digits.index(1)     # returns the index of the first instance of that value (1)

# elements of a tuple cannot be modified
# digits[2] = 2       # throws an error

# concatenate tuples
digits = digits + (3, 4)

# create a single tuple with elements repeated (also works with lists)
(3, 4) * 2          # returns (3, 4, 3, 4)

# tuple unpacking
bart = ('male', 10, 'simpson')  # create a tuple


######################################################################
# Strings
# -------
#
# A sequence of characters, they are iterable, immutable
#

from __future__ import print_function
# create a string
s = str(42)         # convert another data type into a string
s = 'I like you'

# examine a string
s[0]                # returns 'I'
len(s)              # returns 10

# string slicing like lists
s[:6]               # returns 'I like'
s[7:]               # returns 'you'
s[-1]               # returns 'u'

# basic string methods (does not modify the original string)
s.lower()           # returns 'i like you'
s.upper()           # returns 'I LIKE YOU'
s.startswith('I')   # returns True
s.endswith('you')   # returns True
s.isdigit()         # returns False (returns True if every character in the string is a digit)
s.find('like')      # returns index of first occurrence (2), but doesn't support regex
s.find('hate')      # returns -1 since not found
s.replace('like','love')    # replaces all instances of 'like' with 'love'

# split a string into a list of substrings separated by a delimiter
s.split(' ')        # returns ['I','like','you']
s.split()           # same thing
s2 = 'a, an, the'
s2.split(',')       # returns ['a',' an',' the']

# join a list of strings into one string using a delimiter
stooges = ['larry','curly','moe']
' '.join(stooges)   # returns 'larry curly moe'

# concatenate strings
s3 = 'The meaning of life is'
s4 = '42'
s3 + ' ' + s4       # returns 'The meaning of life is 42'
s3 + ' ' + str(42)  # same thing

# remove whitespace from start and end of a string
s5 = '  ham and cheese  '
s5.strip()          # returns 'ham and cheese'

# string substitutions: all of these return 'raining cats and dogs'
'raining %s and %s' % ('cats','dogs')                       # old way
'raining {} and {}'.format('cats','dogs')                   # new way
'raining {arg1} and {arg2}'.format(arg1='cats',arg2='dogs') # named arguments

# string formatting
# more examples: http://mkaz.com/2012/10/10/python-string-format/
'pi is {:.2f}'.format(3.14159)      # returns 'pi is 3.14'

# normal strings versus raw strings
print('first line\nsecond line')     # normal strings allow for escaped characters
print(r'first line\nfirst line')     # raw strings treat backslashes as literal characters


######################################################################
# Dictionaries
# ------------
#
# Dictionaries are structures which can contain multiple data types, and
# is ordered with key-value pairs: for each (unique) key, the dictionary
# outputs one value. Keys can be strings, numbers, or tuples, while the
# corresponding values can be any Python object. Dictionaries are:
# unordered, iterable, mutable
#

# create an empty dictionary (two ways)
empty_dict = {}
empty_dict = dict()

# create a dictionary (two ways)
family = {'dad':'homer', 'mom':'marge', 'size':6}
family = dict(dad='homer', mom='marge', size=6)

# convert a list of tuples into a dictionary
list_of_tuples = [('dad','homer'), ('mom','marge'), ('size', 6)]
family = dict(list_of_tuples)

# examine a dictionary
family['dad']       # returns 'homer'
len(family)         # returns 3
family.keys()       # returns list: ['dad', 'mom', 'size']
family.values()     # returns list: ['homer', 'marge', 6]
family.items()      # returns list of tuples:
                    #   [('dad', 'homer'), ('mom', 'marge'), ('size', 6)]
'mom' in family     # returns True
'marge' in family   # returns False (only checks keys)

# modify a dictionary (does not return the dictionary)
family['cat'] = 'snowball'              # add a new entry
family['cat'] = 'snowball ii'           # edit an existing entry
del family['cat']                       # delete an entry
family['kids'] = ['bart', 'lisa']       # value can be a list
family.pop('dad')                       # removes an entry and returns the value ('homer')
family.update({'baby':'maggie', 'grandpa':'abe'})   # add multiple entries

# accessing values more safely with 'get'
family['mom']                       # returns 'marge'
family.get('mom')                   # same thing
try:
    family['grandma']               # throws an error
except  KeyError as e:
    print("Error", e)

family.get('grandma')               # returns None
family.get('grandma', 'not found')  # returns 'not found' (the default)

# accessing a list element within a dictionary
family['kids'][0]                   # returns 'bart'
family['kids'].remove('lisa')       # removes 'lisa'

# string substitution using a dictionary
'youngest child is %(baby)s' % family   # returns 'youngest child is maggie'


######################################################################
# Sets
# ----
#
# Like dictionaries, but with unique keys only (no corresponding values).
# They are: unordered, iterable, mutable, can contain multiple data types
# made up of unique elements (strings, numbers, or tuples)
#

# create an empty set
empty_set = set()

# create a set
languages = {'python', 'r', 'java'}         # create a set directly
snakes = set(['cobra', 'viper', 'python'])  # create a set from a list

# examine a set
len(languages)              # returns 3
'python' in languages       # returns True

# set operations
languages & snakes          # returns intersection: {'python'}
languages | snakes          # returns union: {'cobra', 'r', 'java', 'viper', 'python'}
languages - snakes          # returns set difference: {'r', 'java'}
snakes - languages          # returns set difference: {'cobra', 'viper'}

# modify a set (does not return the set)
languages.add('sql')        # add a new element
languages.add('r')          # try to add an existing element (ignored, no error)
languages.remove('java')    # remove an element
try:
    languages.remove('c')       # try to remove a non-existing element (throws an error)
except  KeyError as e:
    print("Error", e)
languages.discard('c')      # removes an element if present, but ignored otherwise
languages.pop()             # removes and returns an arbitrary element
languages.clear()           # removes all elements
languages.update('go', 'spark') # add multiple elements (can also pass a list or set)

# get a sorted list of unique elements from a list
sorted(set([9, 0, 2, 1, 0]))    # returns [0, 1, 2, 9]


######################################################################
# Functions
# ---------
#
# Functions are sets of instructions launched when called upon, they can
# have multiple input values and a return value
#

# define a function with no arguments and no return values
def print_text():
    print('this is text')

# call the function
print_text()

# define a function with one argument and no return values
def print_this(x):
    print(x)

# call the function
print_this(3)       # prints 3
n = print_this(3)   # prints 3, but doesn't assign 3 to n
                    #   because the function has no return statement

# define a function with one argument and one return value
def square_this(x):
    return x ** 2

# include an optional docstring to describe the effect of a function
def square_this(x):
    """Return the square of a number."""
    return x ** 2

# call the function
square_this(3)          # prints 9
var = square_this(3)    # assigns 9 to var, but does not print 9

# default arguments
def power_this(x, power=2):
    return x ** power

power_this(2)    # 4
power_this(2, 3) # 8

# use 'pass' as a placeholder if you haven't written the function body
def stub():
    pass

# return two values from a single function
def min_max(nums):
    return min(nums), max(nums)

# return values can be assigned to a single variable as a tuple
nums = [1, 2, 3]
min_max_num = min_max(nums)         # min_max_num = (1, 3)

# return values can be assigned into multiple variables using tuple unpacking
min_num, max_num = min_max(nums)    # min_num = 1, max_num = 3


######################################################################
# Loops
# -----
#
# Loops are a set of instructions which repeat until termination
# conditions are met. This can include iterating through all values in an
# object, go through a range of values, etc
#

# range returns a list of integers
range(0, 3)     # returns [0, 1, 2]: includes first value but excludes second value
range(3)        # same thing: starting at zero is the default
range(0, 5, 2)  # returns [0, 2, 4]: third argument specifies the 'stride'

# for loop (not recommended)
fruits = ['apple', 'banana', 'cherry']
for i in range(len(fruits)):
    print(fruits[i].upper())

# alternative for loop (recommended style)
for fruit in fruits:
    print(fruit.upper())

# use range when iterating over a large sequence to avoid actually creating the integer list in memory
for i in range(10**6):
    pass

# iterate through two things at once (using tuple unpacking)
family = {'dad':'homer', 'mom':'marge', 'size':6}
for key, value in family.items():
    print(key, value)

# use enumerate if you need to access the index value within the loop
for index, fruit in enumerate(fruits):
    print(index, fruit)

# for/else loop
for fruit in fruits:
    if fruit == 'banana':
        print("Found the banana!")
        break   # exit the loop and skip the 'else' block
else:
    # this block executes ONLY if the for loop completes without hitting 'break'
    print("Can't find the banana")

# while loop
count = 0
while count < 5:
    print("This will print 5 times")
    count += 1      # equivalent to 'count = count + 1'


######################################################################
# List comprehensions
# -------------------
#
# Process which affects whole lists without iterating through loops. For
# more:
# http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Comprehensions.html
#

# for loop to create a list of cubes
nums = [1, 2, 3, 4, 5]
cubes = []
for num in nums:
    cubes.append(num**3)

# equivalent list comprehension
cubes = [num**3 for num in nums]    # [1, 8, 27, 64, 125]

# for loop to create a list of cubes of even numbers
cubes_of_even = []
for num in nums:
    if num % 2 == 0:
        cubes_of_even.append(num**3)

# equivalent list comprehension
# syntax: [expression for variable in iterable if condition]
cubes_of_even = [num**3 for num in nums if num % 2 == 0]    # [8, 64]

# for loop to cube even numbers and square odd numbers
cubes_and_squares = []
for num in nums:
    if num % 2 == 0:
        cubes_and_squares.append(num**3)
    else:
        cubes_and_squares.append(num**2)

# equivalent list comprehension (using a ternary expression)
# syntax: [true_condition if condition else false_condition for variable in iterable]
cubes_and_squares = [num**3 if num % 2 == 0 else num**2 for num in nums]    # [1, 8, 9, 64, 25]

# for loop to flatten a 2d-matrix
matrix = [[1, 2], [3, 4]]
items = []
for row in matrix:
    for item in row:
        items.append(item)

# equivalent list comprehension
items = [item for row in matrix
              for item in row]      # [1, 2, 3, 4]

# set comprehension
fruits = ['apple', 'banana', 'cherry']
unique_lengths = {len(fruit) for fruit in fruits}   # {5, 6}

# dictionary comprehension
fruit_lengths = {fruit:len(fruit) for fruit in fruits}              # {'apple': 5, 'banana': 6, 'cherry': 6}


######################################################################
# Exceptions handling
# -------------------
#

dct = dict(a=[1, 2], b=[4, 5])

key = 'c'
try:
    dct[key]
except:
    print("Key %s is missing. Add it with empty value" % key)
    dct['c'] = []

print(dct)


######################################################################
# Basic operating system interfaces (os)
# --------------------------------------
#

import os
import tempfile

tmpdir = tempfile.gettempdir()

# list containing the names of the entries in the directory given by path.
os.listdir(tmpdir)

# Change the current working directory to path.
os.chdir(tmpdir)

# Get current working directory.
print('Working dir:', os.getcwd())

# Join paths
mytmpdir = os.path.join(tmpdir, "foobar")

# Create a directory
if not os.path.exists(mytmpdir):
    os.mkdir(mytmpdir)

filename = os.path.join(mytmpdir, "myfile.txt")
print(filename)

# Write
lines = ["Dans python tout est bon", "Enfin, presque"]

## write line by line
fd = open(filename, "w")
fd.write(lines[0] + "\n")
fd.write(lines[1]+ "\n")
fd.close()

## use a context manager to automatically close your file
with open(filename, 'w') as f:
    for line in lines:
        f.write(line + '\n')

# Read
## read one line at a time (entire file does not have to fit into memory)
f = open(filename, "r")
f.readline()    # one string per line (including newlines)
f.readline()    # next line
f.close()

## read one line at a time (entire file does not have to fit into memory)
f = open(filename, 'r')
f.readline()    # one string per line (including newlines)
f.readline()    # next line
f.close()

## read the whole file at once, return a list of lines
f = open(filename, 'r')
f.readlines()   # one list, each line is one string
f.close()

## use list comprehension to duplicate readlines without reading entire file at once
f = open(filename, 'r')
[line for line in f]
f.close()

## use a context manager to automatically close your file
with open(filename, 'r') as f:
    lines = [line for line in f]


######################################################################
# Object Oriented Programing (OOP)
# --------------------------------
#
# **Sources**
#
# -  http://python-textbok.readthedocs.org/en/latest/Object\_Oriented\_Programming.html
#
# **Principles**
#
# -  **Encapsulate** data (attributes) and code (methods) into objects.
#
# -  **Class** = template or blueprint that can be used to create objects.
#
# -  An **object** is a specific instance of a class.
#
# -  **Inheritance**: OOP allows classes to inherit commonly used state
#    and behaviour from other classes. Reduce code duplication
#
# -  **Polymorphism**: (usually obtained through polymorphism) calling
#    code is agnostic as to whether an object belongs to a parent class or
#    one of its descendants (abstraction, modularity). The same method
#    called on 2 objects of 2 different classes will behave differently.
#

import math

class Shape2D:
    def area(self):
        raise NotImplementedError()

# __init__ is a special method called the constructor

# Inheritance + Encapsulation
class Square(Shape2D):
    def __init__(self, width):
        self.width = width

    def area(self):
        return self.width ** 2

class Disk(Shape2D):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

shapes = [Square(2), Disk(3)]

# Polymorphism
print([s.area() for s in shapes])

s = Shape2D()
try:
    s.area()
except NotImplementedError as e:
    print("NotImplementedError")


######################################################################
# Exercises
# ---------
#


######################################################################
# Exercise 1: functions
# ~~~~~~~~~~~~~~~~~~~~~
#
# Create a function that acts as a simple calulator If the operation is
# not specified, default to addition If the operation is misspecified,
# return an prompt message Ex: ``calc(4,5,"multiply")`` returns 20 Ex:
# ``calc(3,5)`` returns 8 Ex: ``calc(1, 2, "something")`` returns error
# message
#


######################################################################
# Exercise 2: functions + list + loop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Given a list of numbers, return a list where all adjacent duplicate
# elements have been reduced to a single element. Ex: ``[1, 2, 2, 3, 2]``
# returns ``[1, 2, 3, 2]``. You may create a new list or modify the passed
# in list.
#
# Remove all duplicate values (adjacent or not) Ex: ``[1, 2, 2, 3, 2]``
# returns ``[1, 2, 3]``
#


######################################################################
# Exercise 3: File I/O
# ~~~~~~~~~~~~~~~~~~~~
#
# Copy/past the bsd 4 clause license into a text file. Read, the file
# (assuming this file could be huge) and count the occurrences of each
# word within the file. Words are separated by whitespace or new line
# characters.
#


######################################################################
# Exercise 4: OOP
# ~~~~~~~~~~~~~~~
#
# 1. Create a class ``Employee`` with 2 attributes provided in the
#    constructor: ``name``, ``years_of_service``. With one method
#    ``salary`` with is obtained by ``1500 + 100 * years_of_service``.
#
# 2. Create a subclass ``Manager`` which redefine ``salary`` method
#    ``2500 + 120 * years_of_service``.
#
# 3. Create a small dictionary-nosed database where the key is the
#    employee's name. Populate the database with: samples =
#    Employee('lucy', 3), Employee('john', 1), Manager('julie', 10),
#    Manager('paul', 3)
#
# 4. Return a table of made name, salary rows, i.e. a list of list [[name,
#    salary]]
#
# 5. Compute the average salary
