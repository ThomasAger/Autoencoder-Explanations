import matplotlib.pyplot as plt
import numpy as np

import rank
from util import proj_util as dt


# Input: 2 directions in an array, all entities, amt of entities to add
def directionGraph(directions, entities, d_names, e_names):
    # Get the dot products of the entities on the directions
    ranks, r_names = rank.getRankings(directions, entities, d_names, e_names)
    # Arrange by highest ranking
    # Create graph with X coordinates equal to the dot products on the 1st cluster and Y to the 2nd cluster


    y = ranks[0]
    z = ranks[1]

    fig, ax = plt.subplots()

    # Change size, color and marker style
    ax.scatter(z, y, s=0, c="black", marker="x")

    for i, txt in enumerate(e_names):
        ax.annotate(txt, (z[i], y[i]))
    # Add axis labels
    plt.ylabel(d_names[0], fontsize=20)
    plt.xlabel(d_names[1], fontsize=20)
    # Remove numbers
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # Turn grid off
    ax.grid()
    # Change background colour to white
    ax.set_axis_bgcolor('#ffffff')
    # Change colour and width of axis lines
    ax.spines['bottom'].set_color('#6dcff6')
    ax.spines['left'].set_color('#f9ad81')
    ax.spines['bottom'].set_linewidth("4")
    ax.spines['left'].set_linewidth("4")

    plt.plot()
    plt.savefig("../data/movies/figures/description.png", bbox_inches="tight")
    plt.show()
    y = ax.spines['bottom']
    print(y)



dir_ids = [212,368]
classes = ["horror", "comedy"]

# Create direction graph
file_name = "f200geE300DS[200]DN0.5CTgenresHAtanhCV1 S0 SFT0 allL0"
cluster_fn = "100ndcg KMeans CA400 MC1 MS0.4 ATS1000 DS400"

class1 = np.asarray(dt.import1dArray("../data/movies/classify/genres/class-" + classes[0]), "i")
class2 = np.asarray(dt.import1dArray("../data/movies/classify/genres/class-" + classes[1]), "i")


top_indexes = dt.import1dArray("../data/movies/top_250_imdb.txt")

data_type = "movies"
directions = dt.import2dArray("../data/"+data_type+"/cluster/clusters/" + file_name + cluster_fn + ".txt")
d_names = dt.import1dArray("../data/"+data_type+"/cluster/names/" + file_name + cluster_fn + ".txt")
entities = np.asarray(dt.import2dArray("../data/"+data_type+"/nnet/spaces/"+file_name+".txt"))
e_names = np.asarray(dt.import1dArray("../data/" +data_type+"/nnet/spaces/entitynames.txt"))

class1 = class1[top_indexes]
class2 = class2[top_indexes]

confirmed_indexes = []
for c in range(len(class1)):
    if class1[c] == 1:
        confirmed_indexes.append(c)
for c in range(len(class2)):
    if class2[c] == 1:
        confirmed_indexes.append(c)

confirmed_indexes = np.unique(np.asarray(confirmed_indexes))

confirmed_indexes = [4436,4017,12637,13926,7466,
                     783, 576, 1872,
                     2492, 12525, 4810, 12096, 6864]

e_names = e_names[confirmed_indexes]
entities = entities[confirmed_indexes]

chosen_dir = [directions[dir_ids[0]]]
chosen_dir.append(directions[dir_ids[1]])
chosen_names = [d_names[dir_ids[0]]]
chosen_names.append(d_names[dir_ids[1]])



directionGraph(chosen_dir, entities, chosen_names, e_names)