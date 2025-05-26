from fonction import *


EPSILON = 1e-3  # Valeur un peu plus grande pour tolérer les arrondis

def test_AG():
    G1, G1degree = construction_graphe([1,1,1,1,1,1], 4)
    G2, G2degree = construction_graphe([1,1,1,0,0,0,0,0,1,1], 5)

    assert abs(AGindex(G1, G1degree) - 6.0) < EPSILON, "Test failed for AGindex with complete graph"
    assert abs(AGindex(G2, G2degree) - 5.196) < EPSILON, "Test failed for AGindex with incomplete graph"

def test_Randic():
    G1, G1degree = construction_graphe([1, 1, 1, 1, 1, 1], 4)
    G2, G2degree = construction_graphe([1,1,1,0,0,0,0,0,1,1], 5)

    assert abs(RandicIndex(G1, G1degree) - 2.0) < EPSILON, "Test failed for RandicIndex with complete graph"
    assert abs(RandicIndex(G2, G2degree) - 2.394) < EPSILON, "Test failed for RandicIndex with incomplete graph"

def test_colorationNonEquival1():
    G1, _ = construction_graphe([1,1,1,0,0,1,0,0,1,0,1,0,1,1,1], 6)
    G2, _ = construction_graphe([1,1,1,0,0,1,1,0,0,1,0,0,1,1,0], 6)

    assert colorationNonEquival(G1, 6) == 18, "Test failed for colorationNonEquival with G1"
    assert colorationNonEquival(G2, 6) == 17, "Test failed for colorationNonEquival with G2"

def test_colorationNonEquival2():
    G3, _ = construction_graphe([1,1,1,0,1,0,1,1,1,1], 5)
    G4, _ = construction_graphe([1,1,1,0,1,1,0,1,0,1], 5)
    G5, _ = construction_graphe([1,1,1,0,1,0,1,0,1,1], 5)
    G6, _ = construction_graphe([1,1,1,0,1,1,0,1,0,0], 5)

    assert colorationNonEquival(G3, 5) == 4, "Test failed for colorationNonEquival with G3"
    assert colorationNonEquival(G4, 5) == 4, "Test failed for colorationNonEquival with G4"
    assert colorationNonEquival(G5, 5) == 6, "Test failed for colorationNonEquival with G5"
    assert colorationNonEquival(G6, 5) == 5, "Test failed for colorationNonEquival with G6"

"""
def test_colorationNonEquival3():
    G7, _ = construction_graphe([1,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,0,1,0,0],7)
    G8, _ = construction_graphe([1,1,1,0,1,1,0,1,0,1], 5)
    G9, _ = construction_graphe([1,1,1,0,1,0,1,0,1,1], 5)

    assert colorationNonEquival(G7, 5) == 7, "Test failed for colorationNonEquival with G7"
    #assert colorationNonEquival(G8, 5) == 209, "Test failed for colorationNonEquival with G8"
    #assert colorationNonEquival(G9, 5) == 456, "Test failed for colorationNonEquival with G9
"""