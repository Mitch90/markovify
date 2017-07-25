import markovify
import nltk
import re

class POSifiedText(marcovify.Text):
    def word_split(self, sentence):
        words = re.split(self.word_split_pattern, sentence)
        words = [ "::".join(tag) for tag in nltk.pos_tag(words) ]
        return words

    def word_join(self, words):
        sentence = " ".join(word.split("::")[0] for word in words)
        return sentence

# Get raw text as string.
with open("textSources/lupi.txt") as f:
    text = f.read()

# Build the model.
# text_model = markovify.Text(text)

text_POSmodel = POSifiedText(text)

# Print three randomly-generated sentences of no more than 140 characters
for i in range(1):
    print(text_POSmodel.make_short_sentence(140))
