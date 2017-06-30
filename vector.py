import os
import ast
import string
from gensim.models import Word2Vec
import re
import gensim
import numpy
import torch
from CornellData import CornellData

class vector:
    def vettorizzazione():
        a = CornellData()
        listaconversazioni = a.loadConversations(os.path.join(dirName, "movie_conversations.txt"), MOVIE_CONVERSATIONS_FIELDS)
        MOVIE_CONVERSATIONS_FIELDS = ["character1ID","character2ID","movieID","utteranceIDs"]
        token = torch.FloatTensor(1,300)
        token.fill_(0)

        uno = torch.FloatTensor(1,1)
        uno.fill_(1)

        zero = torch.FloatTensor(1,1)
        zero.fill_(0)

        inizio = torch.cat([uno, zero], 1)
        fine = torch.cat([zero, uno], 1)

        fineparola = torch.cat([zero, zero], 1)

        iniziobattuta = torch.cat([token, inizio], 1)

        finebattuta = torch.cat([token, fine], 1)

        vettorefinale=[]
        for conversazione in listaconversazioni:
            vettorebattute=[]
            for battuta in conversazione:
                vettoreparole=torch.FloatTensor()
                vettoreparole = torch.cat([iniziobattuta, vettoreparole])
                for parola in battuta:
                    try:
                        p = model[parola]
                        p = torch.from_numpy(p)
                        p = p.view(1, 300)
                    except:
                        pass
                    parolatokenizzata = torch.cat([p, fineparola], 1)
                    vettoreparole = torch.cat([vettoreparole, parolatokenizzata])
                    #vettoreparole = vettoreparole.view(len(battuta),302)
                vettoreparole = torch.cat([vettoreparole, finebattuta])
                vettorebattute.append(vettoreparole)
            vettorefinale.append(vettorebattute)

        return vettorefinale
