#Nathan Brahmstadt and Jordan Crane

#part1
training_file = open("housing_train.txt", 'r')
training_data = []
for line in training_file:
	print(line)
	training_data.append(line.split())
	