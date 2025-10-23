# obtain materials project formation energies for all 1-5 element materials
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mp_api.client import MPRester
import json

# Load the list from the file
with open('OH_elements.json', 'r') as file:
    OH_elements = json.load(file)

# Load the dictionary from the file
with open('p_dict_oh.json', 'r') as file:
    p_dict_oh = json.load(file)

api_key = "wE930r8O24TLEjNhZEHT8LBivEdm9CqA"

max_en = 0.10  # eV/atom
features = []
formation_energies_valid = []

#######################################################################################
# 1 Element Materials
with MPRester(api_key) as mpr: # 
    docs = mpr.summary.search(chemsys="*", fields=["energy_above_hull", "composition_reduced", "formation_energy_per_atom", "density", "efermi"])

print(docs[0].fields_not_requested)

compositions = []
hulls = []

formation_energies_unfiltered = []
for doc in docs:
    if doc.energy_above_hull <= max_en:
        hulls.append(doc.energy_above_hull)
        compositions.append(str(doc.composition_reduced))
        formation_energies_unfiltered.append(doc.formation_energy_per_atom)

print(compositions)
print(formation_energies_unfiltered)

# Creating a set to keep track of seen elements
seen = set()

# List comprehension to filter out duplicates while preserving order
unique_compositions = [x for x in compositions if x not in seen and not seen.add(x)]

formation_energies = []
for comp in unique_compositions:
    matching_indices = [index for index, s in enumerate(compositions) if s == comp]
    temp_energies = []
    for i in matching_indices:
        temp_energies.append(formation_energies_unfiltered[i])
    formation_energies.append(min(temp_energies))

print(unique_compositions)  
print(formation_energies)

def extract_elements_and_stoichiometry(composition):
    elements = []
    stoichiometry = []
    matches = re.findall(r'([A-Z][a-z]*)(\d+)', composition)
    for match in matches:
        elements.append(match[0])
        stoichiometry.append(int(match[1]))
    return elements, stoichiometry

elements_list = []
stoichiometry_list = []
for comp in unique_compositions:
    elements, stoichiometry = extract_elements_and_stoichiometry(comp)
    elements_list.append(elements)
    stoichiometry_list.append(stoichiometry)


for index, elements in enumerate(elements_list):
    stoichs = []
    ps = []
    if all(entry in OH_elements for entry in elements):
        sum_stoichs = sum(stoichiometry_list[index])
        for ind, el in enumerate(elements):
            ps.append(p_dict_oh[el])
            stoichs.append(stoichiometry_list[index][ind] / sum_stoichs)
        feature_0 = np.zeros(len(ps[0]))
        for i in range(len(stoichs)):
            feature_0 += np.array(ps[i]) * stoichs[i]
        features.append(feature_0)
        
        formation_energies_valid.append(formation_energies[index])

#######################################################################################
# 2 Element Materials
with MPRester(api_key) as mpr: # 
    docs = mpr.summary.search(chemsys="*-*", fields=["energy_above_hull", "composition_reduced", "formation_energy_per_atom", "density", "efermi"])

print(docs[0].fields_not_requested)

compositions = []
hulls = []

formation_energies_unfiltered = []
for doc in docs:
    if doc.energy_above_hull <= max_en:
        hulls.append(doc.energy_above_hull)
        compositions.append(str(doc.composition_reduced))
        formation_energies_unfiltered.append(doc.formation_energy_per_atom)

print(compositions)
print(formation_energies_unfiltered)

# Creating a set to keep track of seen elements
seen = set()

# List comprehension to filter out duplicates while preserving order
unique_compositions = [x for x in compositions if x not in seen and not seen.add(x)]

formation_energies = []
for comp in unique_compositions:
    matching_indices = [index for index, s in enumerate(compositions) if s == comp]
    temp_energies = []
    for i in matching_indices:
        temp_energies.append(formation_energies_unfiltered[i])
    formation_energies.append(min(temp_energies))

print(unique_compositions)  
print(formation_energies)

def extract_elements_and_stoichiometry(composition):
    elements = []
    stoichiometry = []
    matches = re.findall(r'([A-Z][a-z]*)(\d+)', composition)
    for match in matches:
        elements.append(match[0])
        stoichiometry.append(int(match[1]))
    return elements, stoichiometry

elements_list = []
stoichiometry_list = []
for comp in unique_compositions:
    elements, stoichiometry = extract_elements_and_stoichiometry(comp)
    elements_list.append(elements)
    stoichiometry_list.append(stoichiometry)


for index, elements in enumerate(elements_list):
    stoichs = []
    ps = []
    if all(entry in OH_elements for entry in elements):
        sum_stoichs = sum(stoichiometry_list[index])
        for ind, el in enumerate(elements):
            ps.append(p_dict_oh[el])
            stoichs.append(stoichiometry_list[index][ind] / sum_stoichs)
        feature_0 = np.zeros(len(ps[0]))
        for i in range(len(stoichs)):
            feature_0 += np.array(ps[i]) * stoichs[i]
        features.append(feature_0)
        
        formation_energies_valid.append(formation_energies[index])

#############################################################################################
# 3 Element Materials
with MPRester(api_key) as mpr: # 
    docs = mpr.summary.search(chemsys="*-*-*", fields=["energy_above_hull", "composition_reduced", "formation_energy_per_atom", "density", "efermi"])

print(docs[0].fields_not_requested )

compositions = []
hulls = []

formation_energies_unfiltered = []
for doc in docs:
    if doc.energy_above_hull <= max_en:
        hulls.append(doc.energy_above_hull)
        compositions.append(str(doc.composition_reduced))
        formation_energies_unfiltered.append(doc.formation_energy_per_atom)

print(compositions)
print(formation_energies_unfiltered)

# Creating a set to keep track of seen elements
seen = set()

# List comprehension to filter out duplicates while preserving order
unique_compositions = [x for x in compositions if x not in seen and not seen.add(x)]

formation_energies = []
for comp in unique_compositions:
    matching_indices = [index for index, s in enumerate(compositions) if s == comp]
    temp_energies = []
    for i in matching_indices:
        temp_energies.append(formation_energies_unfiltered[i])
    formation_energies.append(min(temp_energies))

print(unique_compositions)  
print(formation_energies)


def extract_elements_and_stoichiometry(composition):
    elements = []
    stoichiometry = []
    matches = re.findall(r'([A-Z][a-z]*)(\d+)', composition)
    for match in matches:
        elements.append(match[0])
        stoichiometry.append(int(match[1]))
    return elements, stoichiometry

elements_list = []
stoichiometry_list = []
for comp in unique_compositions:
    elements, stoichiometry = extract_elements_and_stoichiometry(comp)
    elements_list.append(elements)
    stoichiometry_list.append(stoichiometry)

for index, elements in enumerate(elements_list):
    stoichs = []
    ps = []
    if all(entry in OH_elements for entry in elements):
        sum_stoichs = sum(stoichiometry_list[index])
        for ind, el in enumerate(elements):
            ps.append(p_dict_oh[el])
            stoichs.append(stoichiometry_list[index][ind] / sum_stoichs)
        feature_0 = np.zeros(len(ps[0]))
        for i in range(len(stoichs)):
            feature_0 += np.array(ps[i]) * stoichs[i]
        features.append(feature_0)
        
        formation_energies_valid.append(formation_energies[index])

#############################################################################################
# 4 Element Materials
with MPRester(api_key) as mpr: # 
    docs = mpr.summary.search(chemsys="*-*-*-*", fields=["energy_above_hull", "composition_reduced", "formation_energy_per_atom", "density", "efermi"])

print(docs[0].fields_not_requested )

compositions = []
hulls = []

formation_energies_unfiltered = []
for doc in docs:
    if doc.energy_above_hull <= max_en:
        hulls.append(doc.energy_above_hull)
        compositions.append(str(doc.composition_reduced))
        formation_energies_unfiltered.append(doc.formation_energy_per_atom)

print(compositions)
print(formation_energies_unfiltered)

# Creating a set to keep track of seen elements
seen = set()

# List comprehension to filter out duplicates while preserving order
unique_compositions = [x for x in compositions if x not in seen and not seen.add(x)]

formation_energies = []
for comp in unique_compositions:
    matching_indices = [index for index, s in enumerate(compositions) if s == comp]
    temp_energies = []
    for i in matching_indices:
        temp_energies.append(formation_energies_unfiltered[i])
    formation_energies.append(min(temp_energies))

print(unique_compositions)  
print(formation_energies)


def extract_elements_and_stoichiometry(composition):
    elements = []
    stoichiometry = []
    matches = re.findall(r'([A-Z][a-z]*)(\d+)', composition)
    for match in matches:
        elements.append(match[0])
        stoichiometry.append(int(match[1]))
    return elements, stoichiometry

elements_list = []
stoichiometry_list = []
for comp in unique_compositions:
    elements, stoichiometry = extract_elements_and_stoichiometry(comp)
    elements_list.append(elements)
    stoichiometry_list.append(stoichiometry)

for index, elements in enumerate(elements_list):
    stoichs = []
    ps = []
    if all(entry in OH_elements for entry in elements):
        sum_stoichs = sum(stoichiometry_list[index])
        for ind, el in enumerate(elements):
            ps.append(p_dict_oh[el])
            stoichs.append(stoichiometry_list[index][ind] / sum_stoichs)
        feature_0 = np.zeros(len(ps[0]))
        for i in range(len(stoichs)):
            feature_0 += np.array(ps[i]) * stoichs[i]
        features.append(feature_0)
        
        formation_energies_valid.append(formation_energies[index])

#############################################################################################
# 5 Element Materials
with MPRester(api_key) as mpr: # 
    docs = mpr.summary.search(chemsys="*-*-*-*-*", fields=["energy_above_hull", "composition_reduced", "formation_energy_per_atom", "density", "efermi"])

print(docs[0].fields_not_requested )

compositions = []
hulls = []

formation_energies_unfiltered = []
for doc in docs:
    if doc.energy_above_hull <= max_en:
        hulls.append(doc.energy_above_hull)
        compositions.append(str(doc.composition_reduced))
        formation_energies_unfiltered.append(doc.formation_energy_per_atom)

print(compositions)
print(formation_energies_unfiltered)

# Creating a set to keep track of seen elements
seen = set()

# List comprehension to filter out duplicates while preserving order
unique_compositions = [x for x in compositions if x not in seen and not seen.add(x)]

formation_energies = []
for comp in unique_compositions:
    matching_indices = [index for index, s in enumerate(compositions) if s == comp]
    temp_energies = []
    for i in matching_indices:
        temp_energies.append(formation_energies_unfiltered[i])
    formation_energies.append(min(temp_energies))

print(unique_compositions)  
print(formation_energies)


def extract_elements_and_stoichiometry(composition):
    elements = []
    stoichiometry = []
    matches = re.findall(r'([A-Z][a-z]*)(\d+)', composition)
    for match in matches:
        elements.append(match[0])
        stoichiometry.append(int(match[1]))
    return elements, stoichiometry

elements_list = []
stoichiometry_list = []
for comp in unique_compositions:
    elements, stoichiometry = extract_elements_and_stoichiometry(comp)
    elements_list.append(elements)
    stoichiometry_list.append(stoichiometry)

for index, elements in enumerate(elements_list):
    stoichs = []
    ps = []
    if all(entry in OH_elements for entry in elements):
        sum_stoichs = sum(stoichiometry_list[index])
        for ind, el in enumerate(elements):
            ps.append(p_dict_oh[el])
            stoichs.append(stoichiometry_list[index][ind] / sum_stoichs)
        feature_0 = np.zeros(len(ps[0]))
        for i in range(len(stoichs)):
            feature_0 += np.array(ps[i]) * stoichs[i]
        features.append(feature_0)
        
        formation_energies_valid.append(formation_energies[index])

# Convert features to a PyTorch tensor
features_tensor = [torch.tensor(f, dtype=torch.float32) for f in features]


# Save the energies to a file
with open('formation_energies_valid.json', 'w') as json_file:
    json.dump(formation_energies_valid, json_file)

# save features_tensor
torch.save(features_tensor, 'features_tensor.pt')

