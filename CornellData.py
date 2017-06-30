import os
import ast
import string
from gensim.models import Word2Vec
import re
import gensim
import numpy
import torch

class CornellData:
    def __init__(self):
        dirName='data/'
        self.lines = {}
        self.conversations = []

        MOVIE_LINES_FIELDS = ["lineID","characterID","movieID","character","text"]
        MOVIE_CONVERSATIONS_FIELDS = ["character1ID","character2ID","movieID","utteranceIDs"]

        self.lines = self.loadLines(os.path.join(dirName, "movie_lines.txt"), MOVIE_LINES_FIELDS)
        self.conversations = self.loadConversations(os.path.join(dirName, "movie_conversations.txt"), MOVIE_CONVERSATIONS_FIELDS)

    def loadLines(self, fileName, fields):
        lines = {}
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                values = [''.join(c for c in s if c not in string.punctuation) for s in values]
                # Extract fields
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]
                lines[lineObj['lineID']] = lineObj['text'].strip('\n')
        return lines

    def loadConversations(self, fileName, fields):
        conversations = []
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                convObj = {}
                for i, field in enumerate(fields):
                    convObj['utteranceIDs'] = values[i]
                lineIds = ast.literal_eval(convObj["utteranceIDs"])
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(self.lines[lineId]),
                conversations.append(convObj)
            conversations_list = [d.get('lines') for d in conversations]

            lenght=5
            co=[c for c in conversations_list if len(c) > lenght]
            co2 = co[:25]  #temporaneo, sarà poi co direttamente.

        return co2

    def getConversations(self):
        return self.conversations
