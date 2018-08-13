

def expandTuple(*vals):
    array = []
    for i in vals:
        array.append(i)
    return tuple(array)
x = 1
y = 2
z = 3
x,y,z = expandTuple(x,y,z)
print(x,y,z)
