import isdhic

data = isdhic.HiCData(np.array([(1,1),(1,2),(2,1),(3,1)]))
print data.data

data.remove_self_contacts()
print data.data

data.remove_redundant_contacts()
print data.data

filename = '../data/GSM1173493_cell-1.txt'
parser = isdhic.HiCParser(filename)
datasets = parser.parse()
msg = '#contacts between chr{0} and chr{1}: {2}'

for k, v in datasets.items():
    print msg.format(*(k + (len(v),)))
