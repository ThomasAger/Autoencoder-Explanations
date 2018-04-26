notes = ["A", "B", "C", "D", "E", "F", "G"]
types = ["major", "minor"]
import random
for i in range(30):
    print(random.choice(notes), random.choice(types))