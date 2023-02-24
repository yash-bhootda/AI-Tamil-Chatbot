
import random
import json

import torch

from model import NeuralNet
from nltk_util import bag_of_words, tokenize

from googletrans import Translator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "YAAS"


def get_response(msg):

    translator = Translator()
    string = translator.translate(msg)
    string = string.text

    sentence = tokenize(string)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:

                string = random.choice(intent['responses'])
                if(tag == "menu"):
                    file = open('yourmenu.txt', 'w')
                    x = msg.split("My order is ")
                    file.write(x[1])
                resp_tamil=translator.translate(string , dest='ta')

                return resp_tamil.text

    
    return "எனக்கு புரியவில்லை ......."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You:")
        from googletrans import Translator
        translator = Translator()
        result=translator.translate(sentence)
        
        if sentence == "quit":
            break

        resp = get_response(result.text)
        resp_tamil=translator.translate(resp , dest='ta')
        print(resp_tamil.text)