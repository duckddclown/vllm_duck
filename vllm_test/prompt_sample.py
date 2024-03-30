import random
import string

class sentence_pool:
    def __init__(self) -> None:
        self.freq_sentences = ["'What is your name?',\n",
                               "'The capital of USA is',\n",
                               "'The center of the universe is',\n"]
    
    def generate_random_sequence(self):
        N = 15
        res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k = N))
        res = "'" + str(res) + "',\n"
        return res
    
    def generate_prompts(self):
        file = open("./prompt.txt", "w")
        for i in range(1000):
            prompt = self.freq_sentences[2]
            #randval = random.uniform(0,1)
            #if (randval > 0.5):
                #index = random.randint(0,2)
                #prompt = self.freq_sentences[index]
            #else:
                #prompt = self.generate_random_sequence()
            file.write(prompt)
        file.close()

sent_pool = sentence_pool()
sent_pool.generate_prompts()

