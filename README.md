# Code utilisé dans le cadre de mon mémoire

Code utilisé et résultats obtenus dans le cadre de mon mémoire en sciences de l'informatique.  
Pour compléter le code, veuillez télécharger le fichier numcol.csv suivant :  
https://drive.google.com/file/d/1NqfEUQegR9k2iLx9rUmVeFru35Q7BX8r/view?usp=drive_link  
Vous pouvez ensuite exécuter le code save.py pour obtenir la version JSON, ou directement télécharger le fichier via le lien ci-dessous : 

https://drive.google.com/file/d/1bFOkgaeaLjNzXNX1Wd2qhlajrs_kxEQI/view?usp=drive_link

Ce mémoire s'inspire de l'article de Wagner concernant la recherche de contre-exemples dans la théorie des graphes :  
https://arxiv.org/pdf/2104.14516

Il reprend le code template de Wagner (https://github.com/zawagner22/cross-entropy-for-combinatorics/tree/main) tout en apportant des modifications à son code.  
Pour des raisons de clarté, seules les parties modifiées ou ajoutées seront documentées.

## Structure du répertoire

Le répertoire est divisé comme suit :

./
├── README.md                    # Ce fichier
├── fonction.py                  # Contient plusieurs fonctions comme les calculs d'invariants
├── Test.py                      # Tests unitaires pour valider les fonctions
├── save.py                      # Script transformant le fichier numcol.csv en JSON
├── afficheGraphique.py          # Génération des graphiques d'évolution
├── conjecture1.py               # Implémentation de la première conjecture avec l'approche de Wagner
├── Conjecture2.py               # Implémentation de la deuxième conjecture avec l'approche de Wagner
├── Conjecture2_modi_sup_edge.py # Variante avec suppression d'arêtes pour la deuxième conjecture
├── Conjecture2_modi_add_edge.py # Variante avec ajout d'arêtes pour la deuxième conjecture
├── Conjecture3.py               # Implémentation de la troisième conjecture avec l'approche de Wagner
├── Conjecture3_modi_sup_edge.py # Variante avec suppression d'arêtes pour la troisième conjecture
├── Conjecture3_modi_add_edge.py # Variante avec ajout d'arêtes pour la troisième conjecture
└── resultat/                    # Dossier contenant les résultats des expérimentations
    ├── conjecture1/
    ├── conjecture2/
    └── conjecture3/

## Auteur

**Demaret Jordan**  
Mémoire du Master en Sciences Informatiques, finalité spécialisée