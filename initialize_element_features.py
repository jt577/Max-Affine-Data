import numpy as np
import json


OH_elements = []
with open('elements_list.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        OH_elements.append(line.strip())

print(OH_elements)

ppmat_oh = np.eye(len(lines))
p_dict_oh = {}
for el_index, element in enumerate(OH_elements):
    p_dict_oh[element] = ppmat_oh[:, el_index].tolist()  # Convert numpy array to list

# Save the dictionary to a file
with open('p_dict_oh.json', 'w') as json_file:
    json.dump(p_dict_oh, json_file)

# Save the list to a file
with open('OH_elements.json', 'w') as file:
    json.dump(OH_elements, file)
