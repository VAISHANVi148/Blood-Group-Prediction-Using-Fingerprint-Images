# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:21:38 2025

@author: vaish
"""

import pickle

# Define your blood group classes (adjust if needed)
blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Save the list to 'class_names.pkl'
with open('class_names.pkl', 'wb') as f:
    pickle.dump(blood_groups, f)

print("class_names.pkl file created successfully!")
