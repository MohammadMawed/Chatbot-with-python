from tensorflow.python.framework import ops
from nltk.stem.lancaster import LancasterStemmer
import nltk 
import tensorflow
import numpy 
import tflearn
import random
import simplejson
import pickle 

nltk.download('punkt')
stemmer = LancasterStemmer()

#Opening the json files which contains all our data 

with open("intents.json") as file:
    data = simplejson.load(file)

try:
	with open("data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)

except:    
   
	words = []
	labels = []
	docs_x = []
	docs_y = []

	#We are looping through all intents in our json file for as 
	#For e.g "greeting" tag 

	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern)
			#Here, we are stemming the words.
			#Stemming means that we are taking each word and the model brings  down to its root let the model tryies
			#For e.g the word "Whats up?"
			#We let the model takes the root word of "Whats" and work with that and ignores all letters behides
			#The reason for that is, we are training our model, we dont care what is attached to the word
			#By eliminating extra characters It is  making our model more accurate 
			words.extend(wrds)
			#Adding all words in
			docs_x.append(wrds)
			docs_y.append(intent["tag"])

		if intent["tag"] not in labels:
			labels.append(intent["tag"])

	#Stemming all words that we have in the words list words[] and removing any duplicate elements
	#We want to figure ou the vocabulary size of our model

	words = [stemmer.stem(w.lower()) for w in words]
	words = sorted(list(set(words))) 
	
	labels = sorted(labels)	

	training = []
	output = []

	#The input is won"t work, because we have right now strings and neural networks understand only numbers
	#So we will create a bag if words that represents all of the words in any given pattern
	#Bag of words is known as one hot encoded
	#For e.g we are going to have a list like this [0,0,0,0,0,1], these numbers could be 2,3,4
	#categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction
	#In our case we have 9 tags which are "greeting" and "shop" and "hours" and so one 
	#In this we the algorithm will take the input from the user and it will look up if the iput is already in the dataset
	#For e.g the user enters "hi" our algrithm will look up in our tags and will put by "greeting" tag 1 and all other tags 0 because they dont contains that word

	out_empty = [0 for _ in range (len(labels))]

	for x, doc in enumerate(docs_x):
		bag = []

		wrds = [stemmer.stem(w) for w in doc if w != "?"]
		#Steming all words in patterns 

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)		


		output_row = out_empty[:]
		output_row[labels.index(docs_y[x])] = 1

		training.append(bag)
		output.append(output_row)

	training = numpy.array(training)
	output = numpy.array(output)

	with open("data.pickle", "wb") as f:
		pickle.dump((words, labels, training, output), f)

#Building the model
ops.reset_default_graph()

#Defining the input shape that we are expecting of our model
net = tflearn.input_data(shape = [None, len(training[0])])
#Adding the connected layer to our neural network	
#The network starts at the input data 
#8 neurons for our hidden layer
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
#Softmax is going to give us the probabilities for each output
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
	model.load("model.tflearn")
except:	
	model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric = True)
	model.save("model.tflearn")


def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(w.lower()) for w in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)


def chat():
	print("________________________________________________")
	print("Start talking with the bot(Type quit to stop)!")
	print("________________________________________________")
	print("Hello there, Ask something!")
	while True:
		inp = input("You: ")
		if inp.lower() == "quit":
			#If the user enters "quit" the program will stop
			break
		#Getting the input from the user and feeding that value to the model to make prediction	
		result = model.predict([bag_of_words(inp, words)])[0]
		result_index = numpy.argmax(result)
		tag = labels[result_index]

		if result[result_index] > 0.7:
			for tg in data["intents"]:
				if tg['tag'] == tag:
					responses = tg['responses']
			print(random.choice(responses))		 

		else:
			#If the bot does not understand what he've been asked
			print("I didn't get that, try again or ask something else!")
chat()		

