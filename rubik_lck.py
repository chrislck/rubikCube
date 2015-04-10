import math
import random
import string

random.seed(0)

# ordre arbitraire: on regarde les faces du cube dans l'ordre suivant: Up, Front, Left, Back, Right, (on revient frontside) Down
# taille du cube va de 2 à 7 soit 294 autocollants
class BaseCube(object): 

    order =[ #UpFace   
            [ 0,150, 54, 24, 96,216, 1,
            151,174,175,176,177,238,152,
            55,178, 78, 79,120,239, 56,
            25,179, 80, 48,121,240, 26,
            97,180,122,123,124,241, 98,
            217,242,243,244,245,246,218,
            2,153, 57, 27, 99,219,  3],            
            #FrontFace  
            [4,154, 58, 28,100,220,  5,
            155,181,182,183,184,247,156,
            59,185, 81, 82,125,248, 60,
            29,186, 83, 49,126,249, 30,
            101,187,127,128,129,250,102,
            221,251,252,253,254,255,222,
            6,157, 61, 31,103,223,  7],
            #LeftFace
            [  8,158, 62, 32,104,224,  9,
            159,188,189,190,191,256,160,  
            63,192, 84, 85,130,257, 64,   
            33,193, 86, 50,131,258, 34,  
            105,194,132,133,134,259,106,
            225,260,261,262,263,264,226,
            10,161, 65, 35,107,227, 11], 
            #BackFace
            [ 12,162, 66, 36,108,227 ,13,
            163,195,196,197,198,265,164,
            67,199, 87, 88,135,266, 68,
            37,200, 89, 51,136,267, 38,
            109,201,137,138,139,268,110,
            228,269,270,271,272,273,229,
            14,165, 69, 39,111,230, 15],
            #RightFace
            [ 16,166, 70, 40,112,230, 17,
            167,202,203,204,205,274,168,
            71,206, 90, 91,140,275, 72,
            41,207, 92, 52,141,276, 42,
            113,208,142,143,144,277,114,
            231,278,279,280,281,282,232,
            18,169, 73, 43,115,233, 19],
            #DownFace
            [ 20,170, 74, 44,116,234, 21,
            171,209,210,211,212,283,172,
            75,213, 93, 94,145,284, 76,
            45,214, 95, 53,146,285, 46,
            117,215,147,148,149,286,118,
            235,287,288,289,290,291,236,
            22,173, 77, 47,119,237, 23]
        ]
#n=2
#permutations des coins C (8 exemplaires, 3x8 facettes)
    co= [
        [ # coins
          # X   .   .   .   .   .   X
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          # X   .   .   .   .   .   X
          
         [[  0 ,  1 ,  3 ,  2 ],[  4 ,  8 , 12 , 16 ],[  5 ,  9 , 15 , 17 ]]
        ,[[  4 ,  5 ,  7 ,  6 ],[  3 , 18 , 20 ,  9 ],[  2 , 16 , 21 , 11 ]]
        ,[[  8 ,  9 , 11 , 10 ],[  2 ,  6 , 22 , 13 ],[  0 ,  4 , 20 , 15 ]]
        ,[[ 12 , 13 , 15 , 14 ],[  1 , 19 , 20 ,  8 ],[  0 , 17 , 21 , 10 ]]
        ,[[ 16 , 17 , 19 , 18 ],[  1 , 14 , 21 ,  5 ],[  3 , 12 , 23 ,  7 ]]
        ,[[ 20 , 21 , 23 , 22 ],[  6 , 18 , 14 , 10 ],[  7 , 19 , 15 , 11 ]]	
        ]
        ,[[],[],[],[],[],[]]
        ,[[],[],[],[],[],[]]
        ,[[],[],[],[],[],[]] # rien, quand on bouge une sous couche, les coins ne bougent pas
    ]
#n=3
#permutations des coins C (8 exemplaires, 3x8 facettes)
#permutations des aretes centrales  Ac (12 exemplaires ou 0 si n est pair, 2 facettes par arete)
    ace=[
        [ #arete_centrale
          # .   .   .   X   .   .   .
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          # X   .   .   .   .   .   X
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          # .   .   .   X   .   .   .
          
         [[ 24 , 26 , 27 , 25 ],[ 28 , 32 , 36 , 40 ]]
        ,[[ 28 , 30 , 31 , 29 ],[ 27 , 42 , 44 , 34 ]]
        ,[[ 33 , 32 , 34 , 35 ],[ 25 , 29 , 46 , 38 ]]
        ,[[ 36 , 38 , 39 , 37 ],[ 24 , 33 , 45 , 43 ]]
        ,[[ 40 , 43 , 41 , 42 ],[ 26 , 37 , 47 , 30 ]]
        ,[[ 44 , 47 , 45 , 46 ],[ 31 , 41 , 39 , 35 ]]
        ]
        ,[
         [[ 29, 33, 37, 41],[ 30, 34, 38, 42]]
        ,[[ 25, 40, 46, 35],[ 26, 43, 45, 32]]
        ,[[ 24, 28, 44, 39],[ 27, 31, 47, 36]]
        ,[[ 25, 35, 46, 40],[ 26, 32, 45, 43]]
        ,[[ 24, 39, 44, 28],[ 27, 36, 47, 31]]
        ,[[ 29, 41, 37, 33],[ 30, 42, 38, 34]]
        ]
        ,[
         [[ 29, 33, 37, 41],[ 30, 34, 38, 42]]
        ,[[ 25, 40, 46, 35],[ 26, 43, 45, 32]]
        ,[[ 24, 28, 44, 39],[ 27, 31, 47, 36]]
        ,[[ 25, 35, 46, 40],[ 26, 32, 45, 43]]
        ,[[ 24, 39, 44, 28],[ 27, 36, 47, 31]]
        ,[[ 29, 41, 37, 33],[ 30, 42, 38, 34]]
        ]
        ,[
         [[ 29, 33, 37, 41],[ 30, 34, 38, 42]]
        ,[[ 25, 40, 46, 35],[ 26, 43, 45, 32]]
        ,[[ 24, 28, 44, 39],[ 27, 31, 47, 36]]
        ,[[ 25, 35, 46, 40],[ 26, 32, 45, 43]]
        ,[[ 24, 39, 44, 28],[ 27, 36, 47, 31]]
        ,[[ 29, 41, 37, 33],[ 30, 42, 38, 34]]
        ]
    ]
#permutations des centres pour une rotation souscouche centrale Fc (6 centres ou 0 centre si n est pair)
    fce=[
          # centres
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          # .   .   .   X   .   .   .
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .

        [[],[],[],[],[],[]], #rien, la piece centrale tourne sur elle-même
        [
         [[ 49 , 50 , 51 , 52 ]]
        ,[[ 48 , 52 , 53 , 50 ]]
        ,[[ 48 , 49 , 53 , 51 ]]
        ,[[ 48 , 50 , 53 , 52 ]] # inverse de F 
        ,[[ 48 , 51 , 53 , 49 ]] # inverse de L
        ,[[ 49 , 52 , 51 , 50 ]] # inverse de U
        ],
        [
         [[ 49 , 50 , 51 , 52 ]]
        ,[[ 48 , 52 , 53 , 50 ]]
        ,[[ 48 , 49 , 53 , 51 ]]
        ,[[ 48 , 50 , 53 , 52 ]] # inverse de F 
        ,[[ 48 , 51 , 53 , 49 ]] # inverse de L
        ,[[ 49 , 52 , 51 , 50 ]] # inverse de U
        ],
        [
         [[ 49 , 50 , 51 , 52 ]]
        ,[[ 48 , 52 , 53 , 50 ]]
        ,[[ 48 , 49 , 53 , 51 ]]
        ,[[ 48 , 50 , 53 , 52 ]] # inverse de F 
        ,[[ 48 , 51 , 53 , 49 ]] # inverse de L
        ,[[ 49 , 52 , 51 , 50 ]] # inverse de U
        ]        
    ]
#n=4
#permutations des coins C (8 exemplaires, 3x8 facettes)
#permutations des aretes adjacentes1 A1 (24 exemplaires)
#permutations des facettes coins1 Fc1 (24 exemplaires)
#perte de fc et ac
    a1 = [
          #pour n=4
          # .   X   X   .
          # X   .   .   X
          # X   .   .   X
          # .   X   X   .

          #pour n=5
          # .   X   .   X   .
          # X   .   .   .   X
          # .   .   .   .   .
          # X   .   .   .   X
          # .   X   .   X   .

          #pour n=6
          # .   X   .   .   X   .
          # X   .   .   .   .   X
          # .   .   .   .   .   .
          # .   .   .   .   .   .
          # X   .   .   .   .   X
          # .   X   .   .   X   .

          #pour n=7
          # .   X   .   .   .   X   .
          # X   .   .   .   .   .   X
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          # X   .   .   .   .   .   X
          # .   X   .   .   .   X   .
     [[],[],[],[],[],[]] # a completer
    ,[[],[],[],[],[],[]] # a completer
    ,[[],[],[],[],[],[]] # a completer
    ,[[],[],[],[],[],[]] # a completer
    ]
    
    fco1 = [
        [
          #pour n=4
          # .   .   .   .
          # .   X   X   .
          # .   X   X   .
          # .   .   .   .

          #pour n=5
          # .   .   .   .   .
          # .   X   .   X   .
          # .   .   .   .   .
          # .   X   .   X   .
          # .   .   .   .   .

          #pour n=6
          # .   .   .   .   .   .
          # .   X   .   .   X   .
          # .   .   .   .   .   .
          # .   .   .   .   .   .
          # .   X   .   .   X   .
          # .   .   .   .   .   .

          #pour n=7
          # .   .   .   .   .   .   .
          # .   X   .   .   .   X   .
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          # .   X   .   .   .   X   .
          # .   .   .   .   .   .   .    
        [[],[],[],[],[],[]] # a completer
        ,[[],[],[],[],[],[]] # a completer
        ,[[],[],[],[],[],[]] # a completer
        ,[[],[],[],[],[],[]] # a completer
        ]]
#n=5
#permutations des coins C (8 exemplaires, 3x8 facettes)
#permutations des aretes centrales  Ac (12 exemplaires ou 0 si n est pair, 2 facettes par arete)
#permutations des centres pour une rotation souscouche centrale Fc (6 centres ou 0 centre si n est pair)
#permutations des facettes coins1 Fc1 (24 exemplaires)
#permutations des facettes adjacentes centrales1 Fac1(24 exemplaires)

    fa1 = [[
          #pour n=5
          # .   .   .   .   .
          # .   .   X   .   .
          # .   X   .   X   .
          # .   .   X   .   .
          # .   .   .   .   .

          #pour n=6
          # .   .   .   .   .   .
          # .   .   X   X   .   .
          # .   X   .   .   X   .
          # .   X   .   .   X   .
          # .   .   X   X   .   .
          # .   .   .   .   .   .

          #pour n=7
          # .   .   .   .   .   .   .
          # .   .   X   .   X   .   .
          # .   X   .   .   .   X   .
          # .   .   .   .   .   .   .
          # .   X   .   .   .   X   .
          # .   .   X   .   X   .   .
          # .   .   .   .   .   .   .
        [[],[],[],[],[],[]], # a completer
        [[],[],[],[],[],[]], # a completer
        [[],[],[],[],[],[]]  # a completer
    ]]
#n=6
#permutations des coins C (8 exemplaires, 3x8 facettes)
#permutations des aretes adjacentes1 A1 (24 exemplaires)
#permutations des facettes coins1 Fc1 (24 exemplaires)
#permutations des facettes adjacentes1 Fa1(48 exemplaires)
#permutations des facettes coins2 Fc2 (24 exemplaires)
#perte de fc et ac
    a2 = [
         [
          #pour n=6
          # .   .   X   X   .   .
          # .   .   .   .   .   .
          # X   .   .   .   ..  X
          # X   .   .   .   .   X
          # .   .   .   .   .   .
          # .   .   X   X   .   .

          #pour n=7
          # .   .   X   .   X   .   .
          # .   .   .   .   .   .   .
          # X   .   .   .   .   .   X
          # .   .   .   .   .   .   .
          # X   .   .   .   .   .   X
          # .   .   .   .   .   .   .
          # .   .   X   .   X   .   .
          [],[],[],[],[],[]] # a completer
    ]
    fco2 = [
          [
          # .   .   .   .   .   .
          # .   .   .   .   .   .
          # .   .   X   X   ..  .
          # .   .   X   X   .   .
          # .   .   .   .   .   .
          # .   .   .   .   .   .

          #pour n=7
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          # .   .   X   .   X   .   .
          # .   .   .   .   .   .   .
          # .   .   X   .   X   .   .
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          [],[],[],[],[],[]] # a completer
    ]
#n=7
#permutations des coins C (8 exemplaires, 3x8 facettes)
#permutations des aretes centrales  Ac (12 exemplaires ou 0 si n est pair, 2 facettes par arete)
#permutations des centres pour une rotation souscouche centrale Fc (6 centres ou 0 centre si n est pair)
#permutations des facettes coins1 Fc1 (24 exemplaires)
#permutations des facettes adjacentes centrales1 Fac1(24 exemplaires)
#permutations des facettes coins2 Fc2 (24 exemplaires)
#permutations des facettes adjacentes centrales2 Fac2(24 exemplaires)
    fa2 = [
          [
          #pour n=7
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          # .   .   .   X   .   .   .
          # .   .   X   .   X   .   .
          # .   .   .   X   .   .   .
          # .   .   .   .   .   .   .
          # .   .   .   .   .   .   .
          [],[],[],[],[],[]] # a completer
    ]

    fa1ce= [
          [
          #pour n=7
          # .   .   .   .   .   .   .
          # .   .   .   X   .   .   .
          # .   .   .   .   .   .   .
          # .   X   .   .   .   X   .
          # .   .   .   .   .   .   .
          # .   .   .   X   .   .   .
          # .   .   .   .   .   .   .
          [],[],[],[],[],[]] # a completer
    ]
    
    
    facetsgroup = [  []
                    ,[]
                    ,[co]
                    ,[co,ace,fce]
                    ,[co,a1,fco1]
                    ,[co,ace,a1,fce,fco1,fa1]
                    ,[co,a1,a2,fco1,fa1,fco2]
                    ,[co,ace,a1,a2,fce,fco1,fco2,fa1,fa2,fa1ce]
                    ]
                
#size:     2 .. 7
#sens:     0 .. 1
#qty:      1 .. n
#sublayer: 0 pour size=2 .. size DIV 2 
    rotations = ['U','F','L','B','R','D']
    couleurs=['B','R','W','O','Y','G']
    oppositions=[5,3,4,1,2,0]
    
    def __init__(self,size=2,color=couleurs,facettes=[]):
        """ version 0.1 alpha par chrislck@free.fr """
        if size <2:
            self.size = 2
        else:
            if size >7:
                self.size = 7
            else:
                self.size = size
        self.colors = color
        self.facets =   [None] * self.size * self.size * 6
        if len(facettes) == len(self.facets):
            self.facets = facettes[:]
        for i in range(6): # up front left back right down
            for j in BaseCube.order[i]:
                if j< len(self.facets):
                    self.facets[j] = i
        
    def rotate(self,letter,wide=1,sublayer=0,sens=True,qty=1):            

        def permut(tab,sens=True):
            p = len(tab)
            tmp = [None] * p
            if sens:
                for i in range(p):
                    tmp[i] = self.facets[tab[i][0]]
                for i in range(p):
                    for j in range(3):
                        self.facets[tab[i][j]] = self.facets[tab[i][j+1]]
                    self.facets[tab[i][3]] = tmp[i]
            else:
                for i in range(p):
                    tmp[i] = self.facets[tab[i][3]]
                for i in range(p):
                    for j in range(3):
                        self.facets[tab[i][3-j]] = self.facets[tab[i][3-j-1]]
                    self.facets[tab[i][0]] = tmp[i]
                
        def permutList(fg,letter,wide=1,sublayer=0):
            result = []
            for i in fg:
                result += i[sublayer][BaseCube.rotations.index(letter)]
            if wide>1:
                result += permutlist(fg,BaseCube.rotations.index(letter),wide-1,sublayer+1)
            return result
        
        for i in range(qty):
            permut(permutList(BaseCube.facetsgroup[self.size],letter,wide,sublayer),sens)
    
    def __eq__(self,c=None):
        #  24 classes d'équivalence selon les rotations sur les 3 axes
        if issubclass(c,BaseCube):
            tmp = BaseCube(c.size,c.colors,c.facets)
            for i in range(6):
                for j in range(4):
                    if self.__str__().__eq__(tmp.__str__()):
                        return True
                if i<3:
                    tmp.rotateAll('Y')
                elif i == 3:
                    tmp.rotateAll('X')
                elif i == 4:
                    tmp.rotateAll('Y',True,2)
            return False
        return NotImplemented
    
    def __ne__(self,c=None):
        result = self.__eq__(c)
        if result is NotImplemented:
            return result
        return not result

    def __str__(self):
        return ''.join([BaseCube.couleurs[i]+' ' for i in self.facets])

    def faceOK(self,list=rotations):
#        result = True
        for a in list:
            i = BaseCube.rotations.index(a)
            for j in BaseCube.order[i-1]:
                if j <len(self.facets):
                    if self.facets[BaseCube.order[i][0]] != self.facets[j]:
#                        result = False
                        return False
        return True

    def rotateAll(self,letter,sens=True,qty=1):
        if   letter =='X':
            rotate('R',(self.size // 2)+1,0,False)
            rotate('L',self.size // 2,0,True)
        elif letter =='Y':
            rotate('F',(self.size // 2)+1,0,False)
            rotate('B',self.size // 2,0,True)            
        elif letter =='Z':
            rotate('U',(self.size // 2)+1,0,False)
            rotate('D',self.size // 2,0,True)            

# Back-Propagation Neural Networks
# 
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>

#    import math
#    import random
#    import string

#    random.seed(0)


# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2


class NN():


    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = int(ni + 1) # +1 for bias node
        self.nh = int(nh)
        self.no = int(no)

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)
# end of NN code
        
class ExtendedCube(BaseCube):

    def __init__(self,size=2,colors=BaseCube.couleurs,facettes=[]):
        super(ExtendedCube,self).__init__(size,colors,facettes)
        self.history = []
        nbn = self.size*self.size*6*4   #4 bits, 3 pour coder les 7 couleurs (6 couleurs + & couleur nulle) et 1 bit pour dire si c'est la bonne couleur ou non
        self.neunet = NN(nbn,2*nbn,8)
        self.trainingList = []
        
    def rotate(self,letter,wide=1,sublayer=0,sens=True,qty=1):
        super().rotate(letter,wide,sublayer,sens,qty)
        self.history.append([(letter,wide,sublayer,sens,qty)])
        
    def validColorMorphList(self):
        result = []
        tmp1 = self.colors[:]
        for i in tmp1:
            tmp2 = tmp1[:]
            tmp2.remove(tmp1[oppositions[tmp1.index(i)]])
            tmp2.remove(i)
            for j in tmp2:
                tmp3 = tmp2[:]
                tmp3.remove(tmp2[oppositions[tmp2.index(j)]])
                tmp3.remove(j)
                for k in tmp3:
                    result.append([  i
                                    ,j
                                    ,k
                                    ,tmp1[oppositions[tmp1.index(j)]]
                                    ,tmp1[oppositions[tmp1.index(k)]]
                                    ,tmp1[oppositions[tmp1.index(i)]]
                                   ])
        return result
    
    def isColorMorph(self,c=None):
        #  24 permutations de couleurs : meme forme à une ou plusieurs permutations de couleurs près
        if issubclass(c,BaseCube):
            for l in self.validColorMorphList:
                tmp = BaseCube(self.size,self.colors,l)
                if self.__eq__(tmp):
                    return True
            return False

    def setGoalAndTrain(self):
        
        # random nb rotations (20-50)
        # random axe (1-6)
        # random size and sublayers (1-4) (0-3)
        # random sens and quantité (true/False) (1-2)
        
        pass
    
if __name__ == "__main__":
    a = BaseCube()
    b = ExtendedCube(3)



       
