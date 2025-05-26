import numpy as np
import networkx as nx
import math
import copy

def construction_graphe(state,N):
    """
        Fonction permetant de construire un graphe à partir d'un état donné.
        Cette fonction reprent la façon que Wagner a de construire un graphe à partir d'un état.
        L'ajout apporté est la création d'un tableau Gdeg qui contient le degré de chaque sommet du graphe.
    Args:
        state (list): Liste d'entiers représentant l'état du graphe.
        N (int): Nombre de sommets dans le graphe.
    Returns:
        G (networkx.Graph): Le graphe construit à partir de l'état.
        Gdeg (numpy.ndarray): Un tableau contenant le degré de chaque sommet du graphe.    
    """
    G= nx.Graph()
    G.add_nodes_from(list(range(N)))
    Gdeg = np.zeros(N,dtype=np.int16) 
    count = 0
    for i in range(N):
        for j in range(i+1,N):
            if state[count] == 1:    
                G.add_edge(i,j)
                Gdeg[i] += 1
                Gdeg[j] += 1
            count += 1
    return G, Gdeg

#TODO
def RandicIndex(G, Gdeg):
	result = 0
	for u,v in G.edges():
		result += 1/(math.sqrt(Gdeg[u]*Gdeg[v]))
	return result
#TODO
def AGindex(G,Gdeg):
	result = 0
	for u,v in G.edges():
		result += (Gdeg[u]+Gdeg[v])/(2*math.sqrt(Gdeg[u]*Gdeg[v]))
	return result

def colorationNonEquival(G,n):
    if G.number_of_edges() == int(n*(n-1)/2):
        return 1
    else:
        
        """G=graphe_canonique(G)
        signiature = nx.to_graph6_bytes(G).decode()[10:]
        signiature = signiature[ :len(signiature)-1]
        if signiature in dico:
            return dico[signiature]
        else:"""
        for node, degree in G.degree():
            if degree == n - 1: # Vérifie si le degré du sommet est égal à n-1
                G1 = copy.deepcopy(G)
                G1.remove_node(node)
                mapping = {n-1: node}
                G1= nx.relabel_nodes(G1, mapping)
                return colorationNonEquival(G1, n - 1)
        degres = dict(G.degree()) # Obtenir les degrés des sommets
        sommets_tries = sorted(degres.items(), key=lambda x: x[1], reverse=True) # Trier par degré décroissant
        iChosen = sommets_tries[0][0]
        jChosen = sommets_tries[1][0]
        for j in sommets_tries[1:]:
            node = j[0]
            if not G.has_edge(iChosen,node):
                jChosen = node
                break
        G1 = copy.deepcopy(G)
        G2 = copy.deepcopy(G)

        G1.add_edge(iChosen,jChosen)
        # Fusionner les sommets iChosen et jChosen dans G2
        G2 = nx.contracted_nodes(G2, iChosen, jChosen, self_loops=False)
        if n-1 in G2.nodes() and jChosen != n-1:
            mapping = {n-1: jChosen}
            G2 = nx.relabel_nodes(G2, mapping)
        return colorationNonEquival(G1,n) + colorationNonEquival(G2,n-1)
