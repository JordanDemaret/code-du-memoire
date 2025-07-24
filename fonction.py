import numpy as np
import networkx as nx
import math
import copy
from pynauty import graph , canon_graph
import json


dico={}           
# Charger les données
with open("save.json", "r") as f:
    dico = json.load(f)

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

def RandicIndex(G, Gdeg):
    """
    Fonction pour calculer l'indice de Randic d'un graphe.
    Args:
        G (networkx.Graph): Le graphe pour lequel l'indice de Randic doit être calculé.
        Gdeg (numpy.ndarray): Un tableau contenant le degré de chaque sommet du graphe.
    Returns:
        result (float): L'indice de Randic du graphe.
    """
    result = 0
    for u,v in G.edges():
        result += 1/(math.sqrt(Gdeg[u]*Gdeg[v]))
    return result

def AGindex(G,Gdeg):
    """
    Fonction pour calculer l'indice AG d'un graphe.
    Args:
        G (networkx.Graph): Le graphe pour lequel l'indice AG doit être calculé.
        Gdeg (numpy.ndarray): Un tableau contenant le degré de chaque sommet du graphe.
    Returns:    
        result (float): L'indice AG du graphe.
    """
    result = 0
    for u,v in G.edges():
        result += (Gdeg[u]+Gdeg[v])/(2*math.sqrt(Gdeg[u]*Gdeg[v]))
    return result

def colorationNonEquival(G,n):
    """
    Fonction pour calculer la coloration non équivalente d'un graphe.
    Args:
        G (networkx.Graph): Le graphe pour lequel la coloration non équivalente doit être calculée.
        n (int): Le nombre de sommets dans le graphe.
    Returns:
        int: Le nombre de colorations non équivalentes du graphe.
    """
    if G.number_of_edges() == int(n*(n-1)/2):
        return 1
    else:
        
        G=graphe_canonique(G)
        signiature = nx.to_graph6_bytes(G).decode()[10:]
        signiature = signiature[ :len(signiature)-1]
        if signiature in dico:
            return dico[signiature]
        else:
            for node, degree in G.degree():
                # Vérifie si le degré du sommet est égal à n-1
                if degree == n - 1: 
                    G1 = copy.deepcopy(G)
                    G1.remove_node(node)
                    mapping = {n-1: node}
                    G1= nx.relabel_nodes(G1, mapping)
                    return colorationNonEquival(G1, n - 1)
            # Obtenir les degrés des sommets
            degres = dict(G.degree()) 
            # Trier par degré décroissant
            sommets_tries = sorted(degres.items(), key=lambda x: x[1], reverse=True) 
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

def G_star(n, m):
    """
        Fonction pour créer un graphe G* avec n sommets et m arêtes.
    Args:
        n (int): Le nombre de sommets dans le graphe.
        m (int): Le nombre d'arêtes dans le graphe.
    Returns:
        G (networkx.Graph): Le graphe G* construit avec n sommets et m arêtes.
    """
    k_m=0
    while (k_m * (k_m - 1)) // 2 <= m:
        k_m += 1
    k_m -= 1
    
    r_m = m - (k_m * (k_m - 1)) // 2
    print(r_m)

    # Créer un graphe vide
    G = nx.Graph()

    # Ajouter les sommets de la clique
    clique_nodes = list(range(k_m))
    G.add_nodes_from(clique_nodes)

    # Ajouter les arêtes de la clique
    for i in range(k_m):
        for j in range(i + 1, k_m):
            G.add_edge(i, j)

    # Ajouter les sommets isolés
    isolated_nodes = list(range(k_m, n))
    G.add_nodes_from(isolated_nodes)

    # Connecter un sommet isolé à r_m sommets de la clique
    if n - k_m > 0:
        isolated_node = k_m
        for i in range(r_m):
            G.add_edge(isolated_node, i)

    return G
	
def new_G(listeArc, n):
    """
    Fonction pour créer un nouveau graphe à partir d'une liste d'arêtes et d'un nombre de sommets.
    Args:
        listeArc (list): Liste d'arêtes du graphe.
        n (int): Nombre de sommets dans le graphe.
    Returns:
        G (networkx.Graph): Le nouveau graphe construit à partir de la liste d'arêtes.
    """
    G = nx.Graph()
    G.add_nodes_from(list(range(n)))
    G.add_edges_from(listeArc)
    return G

def graphe_canonique(G):
    """    
        Fonction pour obtenir le graphe canonique d'un graphe donné.
    Args:
        G (networkx.Graph): Le graphe pour lequel le graphe canonique doit être calculé.        
    Returns:
        G (networkx.Graph): Le graphe canonique du graphe d'entrée.
    """
    nauty_graph = graph.Graph(    
        number_of_vertices=G.number_of_nodes(),
        adjacency_dict={node: list(neighbors) for node, neighbors in G.adjacency()}
    )

    tab=[]
    a= canon_graph(nauty_graph).adjacency_dict
    for i in a.keys():
        for j in a[i]:
            if (j,i) not in tab:
                tab.append((i,j))
    G=new_G(tab, G.number_of_nodes())
    return G

def complit_split_graph(n,verticesInClique):
    if verticesInClique < 1 or verticesInClique > n:
        raise ValueError("verticesInClique doit être compris entre 1 et n (inclusif).")

    Gdeg = [0] * n
    # Créer un graphe vide
    G = nx.Graph()

    # Ajouter les sommets de la clique
    clique_nodes = list(range(verticesInClique))
    G.add_nodes_from(clique_nodes)

    # Ajouter les arêtes de la clique
    for i in range(verticesInClique):
        for j in range(i + 1, verticesInClique):
            G.add_edge(i, j)
            Gdeg[i] += 1
            Gdeg[j] += 1

    # Ajouter les sommets isolés
    isolated_nodes = list(range(verticesInClique, n))
    G.add_nodes_from(isolated_nodes)

    # Connecter xhacun sommet isolé au k_m remiers sommets de la clique
    for i in range(verticesInClique, n):
        for j in range(verticesInClique):
            G.add_edge(i, j)
            Gdeg[i] += 1
            Gdeg[j] += 1
    return G, Gdeg




