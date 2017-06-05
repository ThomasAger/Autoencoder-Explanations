import data as dt
from sklearn.neighbors import KDTree

"""
file_name = "wines100-" + classify
space = import2dArray("../data/"+data_type+"/nnet/spaces/"+ file_name + ".txt", "f")
entity_names = import1dArray("../data/" + data_type + "/classify/"+classify+"/available_entities.txt", "s")
"""

data_type = "movies"
classify = "genres"
file_name = "places100"
space = dt.import2dArray("../data/"+data_type+"/nnet/spaces/"+ file_name + "-"+classify+".txt", "f")
entity_names = dt.import1dArray("../data/" + data_type + "/classify/"+classify+"/available_entities.txt", "s")


inds_to_check = range(0,400,20)

for i in inds_to_check:
    print(entity_names[i])
    tree = KDTree(space, leaf_size=2)
    dist, ind = tree.query([space[i]], k=5)
    ind = ind[0][:]
    for j in ind:
        print(entity_names[j])
    print("-------------------------")