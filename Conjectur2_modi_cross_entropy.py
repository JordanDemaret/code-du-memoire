import networkx as nx 
import random
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.models import load_model
from statistics import mean
import pickle
import time
import math
import matplotlib.pyplot as plt

#TODO
from fonction import *  
import copy
import json
import pynauty
import os


N = 13

DECISIONS = int(N*(N-1)/2)  
#TODO
M=DECISIONS//2 -10

LEARNING_RATE = 0.0001 
n_sessions =1000 
percentile = 93 
super_percentile = 94 

FIRST_LAYER_NEURONS = 128
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS = 4

n_actions = M 

observation_space = DECISIONS
	  
len_game = DECISIONS 
state_dim = (observation_space,)

INF = 1000000


#TODO modi
model = Sequential()
model.add(Dense(FIRST_LAYER_NEURONS,  activation="relu", input_shape=(observation_space,)))
model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
model.add(Dense(DECISIONS, activation="sigmoid"))
model.compile(loss="crossentropy", optimizer=SGD(learning_rate = LEARNING_RATE)) 

print(model.summary())

GraphRef = G_star(N, M)
colorationOfGraphRef = colorationNonEquival(GraphRef, N)

found = False
def calc_score(state):
	"""
	Reward function for your problem.

    Input: a 0-1 vector of length DECISIONS. It represents the graph (or other object) you have created.

    Output: the reward/score for your construction. See files in the *demos* folder for examples.	
	"""
     #TODO
	G= nx.Graph()
	global found
	G.add_nodes_from(list(range(N)))
	Gdeg = np.zeros(N,dtype=np.int16) #degree sequence
	count = 0
	for i in range(N):
		for j in range(i+1,N):
			if state[count] == 1:	
				G.add_edge(i,j)
				Gdeg[i] += 1
				Gdeg[j] += 1
			count += 1
			
	ColG = colorationNonEquival(G,N)
	myScore = colorationOfGraphRef - ColG
			
	if myScore > 0.1 and not (found) :
		found = True
		print(myScore)
		print(state)
		nx.draw_kamada_kawai(G)
		plt.show()
		exit()
	return myScore


def play_game(n_sessions, actions,state_next,states,prob, step, total_score):
	#TODO
	for i in range(n_sessions):
		indices = np.arange(DECISIONS)
		selected_indices = np.random.choice(indices, size=step, replace=False, p=prob[i]/np.sum(prob[i])) #Choose M indices with probabilities given by the neural network
	
		j=0
		elem=selected_indices[j]
		state_next[i] = states[i,:,step-1]
		while state_next[i][elem]==1:
			j += 1
			elem=selected_indices[j]
			
		pos=elem #Choose the first index no 
		
		actions[i][pos] = 1 
		state_next[i][pos] = 1
		#calculate final score
		terminal = step == M
		if terminal:
			total_score[i] = calc_score(state_next[i])
				# record sessions 
		if not terminal:
			states[i,:,step] = state_next[i]
	
	
	return actions, state_next,states, total_score, terminal

#jitted_play_game = njit()(play_game)						
def generate_session(agent, n_sessions, verbose = 1):	
	states =  np.zeros([n_sessions, observation_space, len_game], dtype=int)
	actions = np.zeros([n_sessions, len_game], dtype = int)
	state_next = np.zeros([n_sessions,observation_space], dtype = int)
	prob = np.zeros(n_sessions)
	step = 0
	total_score = np.zeros([n_sessions])
	pred_time = 0
	play_time = 0
	
	while (True):
		step += 1		
		tic = time.time()
		prob = agent.predict(states[:,:,step-1], batch_size = n_sessions) 
		pred_time += time.time()-tic
		tic = time.time()
		actions, state_next,states, total_score, terminal = play_game(n_sessions, actions,state_next,states,prob, step, total_score)
		play_time += time.time()-tic
		
		if terminal:
			break
	if (verbose):
		print("Predict: "+str(pred_time)+", play: " + str(play_time))
	return states, actions, total_score
	
#ici
def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
	counter = n_sessions * (100.0 - percentile) / 100.0
	reward_threshold = np.percentile(rewards_batch,percentile)

	elite_states = []
	elite_actions = []
	elite_rewards = []
	for i in range(len(states_batch)):
		if rewards_batch[i] >= reward_threshold-0.0000001:		
			if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
				for item in states_batch[i]:
					elite_states.append(item.tolist())
				for item in actions_batch[i]:
					elite_actions.append(item)			
			counter -= 1
	elite_states = np.array(elite_states, dtype = int)	
	elite_actions = np.array(elite_actions, dtype = int)	
	return elite_states, elite_actions
	
def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
	counter = n_sessions * (100.0 - percentile) / 100.0
	reward_threshold = np.percentile(rewards_batch,percentile)

	super_states = []
	super_actions = []
	super_rewards = []
	for i in range(len(states_batch)):
		if rewards_batch[i] >= reward_threshold-0.0000001:
			if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
				super_states.append(states_batch[i])
				super_actions.append(actions_batch[i])
				super_rewards.append(rewards_batch[i])
				counter -= 1
	super_states = np.array(super_states, dtype = int)


	super_actions = np.array(super_actions, dtype = int)
	super_rewards = np.array(super_rewards)
	return super_states, super_actions, super_rewards
	

super_states =  np.empty((0,len_game,observation_space), dtype = int)
super_actions = np.array([], dtype = int)
super_rewards = np.array([])
sessgen_time = 0
fit_time = 0
score_time = 0

#TODO
listGraph = []
inter=[]
listGraphScore = []

myRand = random.randint(0,1000) #used in the filename
debut = 0 #used to continue the algorithm from a certain point
if __name__ == "__main__":   
    # VTODO
    if os.path.exists("2_2_M_value.txt") and os.path.exists("2_2_model.weights.h5"):
        # Charger le modèle existant
        model.build(input_shape=(None, observation_space))  # Construire le modèle avec la forme d'entrée
        model.load_weights("2_2_model.weights.h5")  # Charger les poids
        # Charger la valeur de M et d'autres paramètres à partir du fichier texte
        with open("2_2_M_value.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("M ="):
                    M = int(line.split("=")[1].strip())  
                elif line.startswith("i ="):
                    debut = int(line.split("=")[1].strip())+1
                elif line.startswith("myRand ="):
                    myRand = int(line.split("=")[1].strip())  
    #TODO
    if not os.path.exists(str(myRand)):
        os.makedirs(str(myRand))   
    #TODO
    nx.draw_kamada_kawai(GraphRef)
    plt.savefig(str(myRand)+'/graph_ref_'+str(myRand)+'.png')
    plt.close()  # Fermer la figure précédente pour éviter les superpositions

    # TODO
    GraphRef_complement = nx.complement(GraphRef)
    nx.draw_kamada_kawai(GraphRef_complement)
    plt.savefig(str(myRand)+'/graph_ref_complement_'+str(myRand)+'.png')
    plt.close()

       

    for i in range(debut,100000): 
        tic = time.time()
        sessions = generate_session(model,n_sessions,0) #change 0 to 1 to print out how much time each step in generate_session takes
        sessgen_time = time.time()-tic
        tic = time.time()
       
        states_batch = np.array(sessions[0], dtype = int)
        actions_batch = np.array(sessions[1], dtype = int)
        rewards_batch = np.array(sessions[2])
        states_batch = np.transpose(states_batch,axes=[0,2,1])
       
        states_batch = np.append(states_batch,super_states,axis=0)

        if i > 0 and len(super_actions) > 0:
            actions_batch = np.append(actions_batch,np.array(super_actions),axis=0)
        rewards_batch = np.append(rewards_batch,super_rewards)
           
        randomcomp_time = time.time()-tic
        tic = time.time()

       
        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile) #pick the sessions to learn from
        select1_time = time.time()-tic

        tic = time.time()
        super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=super_percentile) #pick the sessions to survive
        select2_time = time.time()-tic
       
        tic = time.time()
       
        super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
        super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
        select3_time = time.time()-tic
       
        tic = time.time()
        model.fit(elite_states, elite_actions) #learn from the elite sessions
        fit_time = time.time()-tic
       
        tic = time.time()
       
        super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
        super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
        super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]
       
        rewards_batch.sort()
        mean_all_reward = np.mean(rewards_batch[-100:])
        mean_best_reward = np.mean(super_rewards)  

        score_time = time.time()-tic
       
        print("\n" + str(i) +  ". Best individuals: " + str(np.flip(np.sort(super_rewards))))
       
        print(  "Mean reward: " + str(mean_all_reward) + "\nSessgen: " + str(sessgen_time) + ", other: " + str(randomcomp_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time))
       
       
        if (i%20 == 1): #Write all important info to files every 20 iterations
            with open(str(myRand)+'/best_species_pickle_'+str(myRand)+'.txt', 'wb') as fp:
                pickle.dump(super_actions, fp)
            with open(str(myRand)+'/best_species_txt_'+str(myRand)+'.txt', 'w') as f:
                for item in super_actions:
                    f.write(str(item))
                    f.write("\n")
            with open(str(myRand)+'/best_species_rewards_'+str(myRand)+'.txt', 'w') as f:
                for item in super_rewards:
                    f.write(str(item))
                    f.write("\n")
            with open(str(myRand)+'/best_100_rewards_'+str(myRand)+'.txt', 'a') as f:
                f.write(str(mean_all_reward)+"\n")
            with open(str(myRand)+'/best_elite_rewards_'+str(myRand)+'.txt', 'a') as f:
                f.write(str(mean_best_reward)+"\n")
            #TODO
            with open(str(myRand)+'/layer_weights_'+str(myRand)+'.txt', 'w') as f:
                for layer in model.layers:
                    weights, biases = layer.get_weights()
                    f.write(f"Layer: {layer.name}\n")
                    f.write(f"Weights: {weights}\n")
                    f.write(f"Biases: {biases}\n")

        if (i%200==2): # To create a timeline, like in Figure 3
            with open(str(myRand)+'/best_species_timeline_txt_'+str(myRand)+'.txt', 'a') as f:
                f.write(str(i)+"\n")
                f.write(str(super_actions[0]))
                f.write("\n")
        #TODO TOUTLA FIN 
            inter.append(i)
            G= nx.Graph()
            G.add_nodes_from(list(range(N)))
            count = 0
            for i in range(N):
                for j in range(i+1,N):
                    if super_actions[0][count] == 1:
                        G.add_edge(i,j)
                    count += 1
            listGraph.append(G)
            listGraphScore.append(super_rewards[0])


            # Affichage des graphes
            num_graphs = len(listGraph)
            cols = 5  # Nombre de colonnes dans la grille
            rows = math.ceil(num_graphs / cols)  # Calcul du nombre de lignes nécessaires

            fig, axes = plt.subplots(rows, cols, figsize=(min(20, 5 * cols), min(20, 5 * rows)))
            axes = axes.flatten()  # Aplatir pour un accès facile

            for i, G in enumerate(listGraph):
                pos = nx.circular_layout(G)
                #pos = nx.shell_layout(G)
           
                # Identifier les nœuds isolés
           
                # Dessiner le graphe
                nx.draw(G, pos, ax=axes[i], with_labels=True, node_color='lightblue', edge_color='gray')
                axes[i].set_title(f"Itération n° {inter[i]} (Score: {listGraphScore[i]:.2f})")  # Titre pour chaque sous-graphe
               
               
                # Ajouter une grille manuellement
                axes[i].set_xticks([])  # Supprimer les ticks x
                axes[i].set_yticks([])  # Supprimer les ticks y
                axes[i].grid(visible=True, color='gray', linestyle='--', linewidth=0.5)  # Ajouter une grille
            plt.savefig(str(myRand)+'/graphes_'+str(myRand)+'.png')
            plt.close(fig)  # Fermer la figure après l'enregistrement pour éviter d'afficher les graphes à chaque itération
        
            fig, axes = plt.subplots(rows, cols, figsize=(min(20, 5 * cols), min(20, 5 * rows)))
            axes = axes.flatten()  # Aplatir pour un accès facile

            for i, G in enumerate(listGraph):
                G_complement = nx.complement(G)
                pos = nx.circular_layout(G_complement)
                # Dessiner le graphe courant 
                nx.draw(G_complement, pos, ax=axes[i], with_labels=True, node_color='lightblue', edge_color='gray')
                axes[i].set_title(f"Itération n° {inter[i]} (Score: {listGraphScore[i]:.2f})")
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                axes[i].grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

            plt.savefig(f"{myRand}/graphe_complement_{myRand}.png")
            plt.close(fig)
       
        super_actions = np.array(super_actions)
        
        # Redimensionnez super_actions pour qu'il ait le même nombre de dimensions que actions_batch
        if super_actions.ndim == 1:
            super_actions = super_actions.reshape(-1, actions_batch.shape[1])

        # Concaténez les tableaux
        actions_batch = np.append(actions_batch, np.array(super_actions), axis=0)

        # Sauvegarder les poids du modèle à la fino
        model.save_weights(f"2_2_model.weights.h5")
        print(f"Les poids du modèle ont été sauvegardés dans 2_2_model.weights.h5")
        # Sauvegarder la valeur de M dans un fichier texte
        with open(f"2_2_M_value.txt", "w") as f:
            f.write(f"M = {M}\n")
            f.write(f"i = {i}\n")
            f.write(f"myRand = {myRand}\n")
        print(f"La valeur de M a été sauvegardée dans 2_2_M_value.txt")

