import itertools

range = range(15000)

comb = itertools.combinations(range, 2)

lister = list(comb)

print(len(lister))