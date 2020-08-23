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
    
    
string = ''' {{}} &
		{\includegraphics[width=0.2\\textwidth]{photo/table_ctm_nn_city/{{}}-pm25-1.png}} &
        {\includegraphics[width=0.2\\textwidth]{photo/table_ctm_nn_city/{{}}-pm25-4.png}} &
        {\includegraphics[width=0.2\\textwidth]{photo/table_ctm_nn_city/{{}}-pm25-7.png}} &
        {\includegraphics[width=0.2\\textwidth]{photo/table_ctm_nn_city/{{}}-pm25-10.png}} \\\\
        \hline
        '''
def format_Lin(*r):
    s = string.split('{{}}')
    res = ''
    for i in range(len(s)):
        if i<len(r):
            res = res + s[i] + r[i]
        else:
            res = res+s[i]
    return res

for c1,c2 in [['TianJin','tj'],['ShiJiazhuang','sjz'],['NingBo','nb'],['LanZhou','lz'],\
              ['XiAn','xa'],['XinJiang','xj'],['SanYa','sy'],['HeFei','hf'],['ShangHai','sh'],\
              ['GuangZhou','gz'],['XiaMen','xm']]:
    print(format_Lin(c1,c2,c2,c2,c2))