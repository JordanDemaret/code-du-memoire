
import matplotlib.pyplot as plt
import numpy as np




listMeanAllReward = []  # Liste pour stocker les moyennes des scores

chemmin="./resultat/conjecture2/Conjectur2_n_7_m_6/best_100_rewards_2.txt"

with open(chemmin, 'r') as file:
    for line in file:
        if line.strip():  # Vérifie si la ligne n'est pas vide
            score = float(line.strip())
            listMeanAllReward.append(score)  # Ajoute le score à la liste
    

listMeanAllReward.pop(-1)  # Supprimer le premier élément (0.0) de la liste






# Permet de créer un graphique de l'évolution des scores et de l'enregistrer
iterations = list(range(1, len(listMeanAllReward) * 20, 20))  
plt.plot(iterations, listMeanAllReward)  
plt.plot([1, iterations[-1]], [0.1, 0.1], color='r', linestyle='--')  
if len(iterations) <= 10:
    # Si moins de 10 points, afficher tous les points
    x_ticks = iterations
else:
    # Sinon, créer environ 10 graduations
    pas = max(1, iterations[-1] // 10)
x_ticks = list(range(1, iterations[-1] + 1, pas))
    
plt.xticks(x_ticks)
plt.xlabel("Kéme itération")  
plt.ylabel("Scores") 
plt.title("Évolution des scores des 100 meilleurs individus par itération") 
plt.savefig('Evolution_score_trouver.png')
plt.close()  