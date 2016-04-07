# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 10:03:29 2016

@author: edouard.duchesnay@gmail.com
"""

'''
## Exercise 1: functions

define a function with two 'positional arguments' (no default values) and
one 'keyword argument' (has a default value)
'''
def calc(a, b, op='add'):
    if op == 'add':
        return a + b
    elif op == 'sub':
        return a - b
    else:
        print('valid operations are add and sub')

# call the function
calc(10, 4, op='add')   # returns 14
calc(10, 4, 'add')      # also returns 14: unnamed arguments are inferred by position
calc(10, 4)             # also returns 14: default for 'op' is 'add'
calc(10, 4, 'sub')      # returns 6
calc(10, 4, 'div')      # prints 'valid operations are add and sub'


'''
Exercise 2: functions + list + loop

Given a list of numbers, return a list where
all adjacent duplicate elements have been reduced to a single element.
Ex: `[1, 2, 2, 3, 2]` returns `[1, 2, 3, 2]`. 
You may create a new list or modify the passed in list.

Remove all duplicate values (adjacent or not)
Ex: `[1, 2, 2, 3, 2]` returns `[1, 2, 3]`
'''

def remove_adjacent_duplicates(original_list):
    new_list = []
    new_list.append(original_list[0])
    for num in original_list[1:]:
        if num != new_list[-1]:
            new_list.append(num)
    return new_list

remove_adjacent_duplicates([1, 2, 2, 3, 2])

def remove_duplicates(original_list):
    new_list = []
    for num in original_list:
        if num not in new_list:
            new_list.append(num)
    return new_list

remove_duplicates([3, 2, 2, 1, 2])

# or this solution mights modify the order

def remove_duplicates(original_list):
    return(list(set(original_list)))
    
remove_duplicates([3, 2, 2, 1, 2])

'''
Exercise 4: OOP

    Create a class Employee with 2 attributes provided in the constructor: name, years_of_service. With one method salary with is obtained by 1500 + 100 * years_of_service.

    Create a subclass Manager which redefine salary method 2500 + 120 * years_of_service.

    Create a small dictionnary-nased database where the key is the employee's name. Populate the database with: Employee("lucy", 3), Employee("john", 1), Manager('julie', 10), Manager('paul', 3)

    Return a table of made name, salary rows, ie. a list of list  [[name, salary]]

    Compute the average salary
'''

class Employee:
    def __init__(self, name, years_of_service):
        self.name = name
        self.years_of_service = years_of_service

    def salary(self):
        return 1500 + 100 * self.years_of_service

class Manager(Employee):
    def salary(self):
        return 2500 + 120 * self.years_of_service

employees = dict()
samples = Employee("lucy", 3), Employee("john", 1), Manager('julie', 3),  Manager('paul', 1)

for e in samples:
    employees[e.name] = e

[[name, employees[name].salary()] for name in employees]

sum([e.salary() for e in employees.values()]) / len(employees)


'''
Exercie 3: File I/O

Copy/past the bsd 4 clause license into a text file. Read, the file 
(assuming this file could be huge) and cound the occurences of each word
within the file. Words are separated by whitespace or new line characters.
'''

bsd_4clause = """
Copyright (c) <year>, <copyright holder>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software
   must display the following acknowledgement:
   This product includes software developed by the <organization>.
4. Neither the name of the <organization> nor the
   names of its contributors may be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY <COPYRIGHT HOLDER> ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import tempfile

tmpfile = os.path.join(tempfile.gettempdir(), "bsd.txt")

fd = open(tmpfile, "w")
fd.write(bsd_4clause)
fd.close()

fd = open(tmpfile, "r")

count = dict()
for line in fd:
    for word in line.split():
        if not word in count:
            count[word] = 1
        else:
            count[word] += 1

print(count)
