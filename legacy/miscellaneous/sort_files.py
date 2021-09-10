
# coding: utf-8

# In[17]:


import re
from collections import defaultdict
import os
import shutil

def search(file):
    machine, slot, port = None, None, None
    regex_machine_slot = re.compile(r"(ilcomp[a-z0-9]+),\s+slot\s([0-9])") 
    regex_port = re.compile(r"-p\s(\d+)")
    with open(file) as f:
        for line in f:
            line = line.rstrip('\n')
            r = re.search(regex_machine_slot, line)
            if r is not None:
                machine = r.group(1)
                slot = r.group(2)
            r = re.search(regex_port, line)
            if r is not None:
                port = r.group(1)
    return machine, slot, port


# In[19]:


files = [f[:-4] for f in os.listdir('.') if f.endswith('.out')]
print(files)
alloc_dict = defaultdict(list)
machine_log_dict = defaultdict(list)
for f in files:
    machine, slot, port = search(f+".out")
    if machine is None or slot is None or port is None:
        continue
    alloc_dict[machine].append((slot, port))
    machine_log_dict[machine].extend([f+".out", f+".err"])
print(list(alloc_dict.keys()))
print(list(alloc_dict.values()))
print(machine_log_dict)

for machine, alloc in alloc_dict.items():
    slot_lst = []
    port_lst = []
    for a in alloc:
        slot_lst.append(str(int(a[0])-1))
        port_lst.append(a[1])
    slot_lst.sort(key=int)
    port_lst.sort(key=int)
    folder_name = '{}_{}_{}'.format(machine, ''.join(slot_lst), ':'.join(port_lst))
    os.makedirs(folder_name)
    for log in machine_log_dict[machine]:
        shutil.move(log, os.path.join(folder_name, log))

