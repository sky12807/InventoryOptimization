string = '''
'''
import numpy as np
r = []
for i in string.split('\n'):
    if 'test' in i:
        line = i.split(':  ')[-1].split(',')
        r.append([float(p) for p in line])
r = np.array(r)

for i in r[:,1]:print(i)