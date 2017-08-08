import data as dt
from sklearn.neighbors import KDTree
import os
from shutil import copyfile


def kdTree(entity_names, space):
    inds_to_check = range(0,400,20)

    for i in inds_to_check:
        print(entity_names[i])
        tree = KDTree(space, leaf_size=2)
        dist, ind = tree.query([space[i]], k=5)
        ind = ind[0][:]
        for j in ind:
            print(entity_names[j])
        print("-------------------------")
"""
data_type = "movies"
classify = "genres"
file_name = "places100"
space = dt.import2dArray("../data/"+data_type+"/nnet/spaces/"+ file_name + "-"+classify+".txt", "f")
entity_names = dt.import1dArray("../data/" + data_type + "/classify/"+classify+"/available_entities.txt", "s")
"""
def treeImages(loc, names,class_name):
    for n in names:
        copyfile(loc + class_name + " " + n + "CV0" + ".png",   output_loc + class_name + " " +  n + "CV0" + ".png")
"""
file_name = "wines100-" + classify
space = import2dArray("../data/"+data_type+"/nnet/spaces/"+ file_name + ".txt", "f")
entity_names = import1dArray("../data/" + data_type + "/classify/"+classify+"/available_entities.txt", "s")
"""

data_type = "placetypes"
class_name = "TravelAndTransport"
name1 = "places NONNETCV5S4 SFT0 allL050kappa KMeans CA100 MC1 MS0.4 ATS2000 DS200 foursquare tdev3"
name2 = "places NONNETCV5S4 SFT0 allL050kappa KMeans CA100 MC1 MS0.4 ATS2000 DS200 foursquare tdev3FT BOCFi IT1300"
names = [name1, name2]
loc = "../data/" + data_type + "/rules/tree_images/"
output_loc = "../data/" + data_type + "/rules/tree_investigate/"
treeImages(loc, names, class_name)