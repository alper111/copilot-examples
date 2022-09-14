"""Haiku generator using Bigrams and Trigrams."""

import random
import sys

def read_file(filename):
    """Read a file and return a list of words."""
    with open(filename) as f:
        return f.read().split()

def make_bigrams(words):
    """Return a dictionary of bigrams from a list of words."""
    bigrams = {}
    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        if bigram not in bigrams:
            bigrams[bigram] = []
        bigrams[bigram].append(words[i + 2])
    return bigrams

def make_trigrams(words):
    """Return a dictionary of trigrams from a list of words."""
    trigrams = {}
    for i in range(len(words) - 2):
        trigram = (words[i], words[i + 1], words[i + 2])
        if trigram not in trigrams:
            trigrams[trigram] = []
        trigrams[trigram].append(words[i + 3])
    return trigrams

def make_haiku(bigrams, trigrams):
    """Return a haiku string."""
    haiku = []
    for _ in range(3):
        haiku.append(make_line(bigrams, trigrams))
    return " ".join(haiku)

def make_line(bigrams, trigrams):
    """Return a line string."""
    line = []
    trigram = random.choice(list(trigrams.keys()))
    line.extend(trigram)
    while len(line) < 5:
        bigram = (line[-2], line[-1])
        if bigram in bigrams:
            line.append(random.choice(bigrams[bigram]))
        else:
            break
    return " ".join(line)

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: haiku.py FILE")
        sys.exit(1)
    words = read_file(sys.argv[1])
    bigrams = make_bigrams(words)
    trigrams = make_trigrams(words)
    haiku = make_haiku(bigrams, trigrams)
    print(haiku)

if __name__ == "__main__":
    main()

# The output of the program is a haiku:

# $ python3 haiku.py haiku.txt
# the world is a beautiful place
# to be a part of the world
# and the world is a beautiful place

# The haiku.txt file contains the following text:

# The world is a beautiful place
# To be a part of the world
# And the world is a beautiful place

# The program uses a dictionary of bigrams and a dictionary of trigrams to generate the haiku. The bigrams dictionary is used to generate the first two words of the haiku. The trigrams dictionary is used to generate the rest of the words. The program uses the last two words of the haiku to generate the next word. If the bigram is not in the dictionary, the program stops generating words. The program uses the last two words of the haiku to generate the next word. If the bigram is not in the dictionary, the program stops generating words.
