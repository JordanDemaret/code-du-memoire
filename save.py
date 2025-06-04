import json

dico={}

with open("numcol.csv","r") as f:
    f.readline()
    line = f.readline()
    while line:
        line = line.split(",")
        dico[line[0]] = int(line[1])
        line = f.readline()

# Sauvegarder les données
with open("save.json", "w") as f:
    json.dump(dico, f)