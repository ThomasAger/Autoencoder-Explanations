import data as dt
import seaborn as sns
import numpy as np
import rank
import matplotlib as plt
# Input: 2 directions in an array, all entities, amt of entities to add
def directionGraph(directions, entities, d_names, e_names, e_amt):
    # Get the dot products of the entities on the directions
    ranks, r_names = rank.getRankings(directions, entities, d_names, e_names)
    # Arrange by highest ranking
    # Create graph with X coordinates equal to the dot products on the 1st cluster and Y to the 2nd cluster


    y = ranks[0]
    z = ranks[1]
    n = r_names

    fig, ax = plt.subplots()
    ax.scatter(z, y)

    for i, txt in enumerate(n):
        ax.annotate(txt, (z[i], y[i]))

dir_ids = [118,35]
# Create direction graph
file_name = "movies mds CV1 S0 LEFalse SFT0L0100ndcg0.95 Breakoff CA200 MC1 MS0.4 ATS1000 DS600"
data_type = "movies"
directions = "../data/"+data_type+"/cluster/hierarchy_directions/" + file_name + ".txt"
names = "../data/"+data_type+"/cluster/hierarchy_names/" + file_name + ".txt"
entities = "../data/"+data_type+"/nnet/spaces/films200-genres.txt"
directionGraph()