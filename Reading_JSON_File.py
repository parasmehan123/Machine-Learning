"""SCRIPT FOR READING JSON FILE"""


import json

data = [json.loads(line) for line in open('C:/Users/priya/Desktop/GeneToVec/Dataset for Gene2Vec Project/data01.json', 'r')]

text = []
text = [item['abstract'] for item in data]

  