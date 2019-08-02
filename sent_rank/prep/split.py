import xml.etree.ElementTree
import copy

e = xml.etree.ElementTree.parse(f'query_res.xml').getroot()

n = len(e.findall('row'))
print(n)

n_per_chunk = n // 300

id = fid = cur_id = 0
top = xml.etree.ElementTree.Element('content')
rows = []
for row in e.findall('row'):
    id += 1
    cur_id += 1

    top.append(copy.deepcopy(row))

    if id == n_per_chunk or id >= n:
        tree = xml.etree.ElementTree.ElementTree(top)
        tree.write(f'data_s/output{fid}.xml', encoding="utf-8")
        id = 0
        fid += 1
        top = xml.etree.ElementTree.Element('content')

    if cur_id % 100 == 0:
        print(f'{cur_id} is split')
