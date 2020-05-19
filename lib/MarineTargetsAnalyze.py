from .nnLayer import *
from torch.nn import functional as F
from tqdm import tqdm
import pickle

class StateAnalyzer:
    def __init__(self, weightPath, classNum=3, 
                 feaSize=224, filterNum=256, contextSizeList=[1,3,5],hiddenList=[],map_location="cpu",device="cpu", 
                 xySize=64, vSize=32, dSize=32, tSize=32,
                 xydiff=0.001, vdiff=1, ddiff=10, tdiff=10800):
        stateDict = torch.load(weightPath, map_location=map_location)
        self.lab2id,self.id2lab = stateDict['lab2id'],stateDict['id2lab']
        self.x2id,self.id2x = stateDict['x2id'],stateDict['id2x']
        self.y2id,self.id2y = stateDict['y2id'],stateDict['id2y']
        self.v2id,self.id2v = stateDict['v2id'],stateDict['id2v']
        self.d2id,self.id2d = stateDict['d2id'],stateDict['id2d']
        self.t2id,self.id2t = stateDict['t2id'],stateDict['id2t']
        self.xEmbedding = TextEmbedding( torch.zeros((len(self.id2x),xySize), dtype=torch.float),name='xEmbedding' ).to(device)
        self.yEmbedding = TextEmbedding( torch.zeros((len(self.id2y),xySize), dtype=torch.float),name='yEmbedding' ).to(device)
        self.vEmbedding = TextEmbedding( torch.zeros((len(self.id2v),vSize), dtype=torch.float),name='vEmbedding' ).to(device)
        self.dEmbedding = TextEmbedding( torch.zeros((len(self.id2d),dSize), dtype=torch.float),name='dEmbedding' ).to(device)
        self.tEmbedding = TextEmbedding( torch.zeros((len(self.id2t),tSize), dtype=torch.float),name='tEmbedding' ).to(device)
        self.textCNN = TextCNN(feaSize, contextSizeList, filterNum).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterNum, classNum, hiddenList).to(device)
        self.moduleList = nn.ModuleList([self.xEmbedding,self.yEmbedding,self.vEmbedding,self.dEmbedding,self.tEmbedding, self.textCNN, self.fcLinear])
        for module in self.moduleList:
            module.load_state_dict(stateDict[module.name])
            module.eval()

        self.xydiff,self.vdiff,self.ddiff,self.tdiff = xydiff,vdiff,ddiff,tdiff
        self.device = device
        print("%d epochs and %.3lf val Score 's model load finished."%(stateDict['epochs'], stateDict['bestMtc']))

    def predict(self, STATE, batchSize=64):
        if type(STATE)==str:
            with open(STATE, 'rb') as f:
                STATE = pickle.load(f)
        print('Transforming...')
        STATE = [[[x//self.xydiff,y//self.xydiff,v//self.vdiff,d//self.ddiff,t//self.tdiff] for x,y,v,d,t in ship] for ship in STATE]
        tokenizedSTATE = np.array([[[
                                        self.x2id[x] if x in self.x2id else 1,
                                        self.y2id[y] if y in self.y2id else 1,
                                        self.v2id[v] if v in self.v2id else 1,
                                        self.d2id[d] if d in self.d2id else 1,
                                        self.t2id[t] if t in self.t2id else 1
                                    ] for x,y,v,d,t in ship] for ship in STATE])
        STATELen = np.array([len(r)+1 for r in STATE],dtype='int32')

        Ypre = []
        idList = list(range(len(tokenizedSTATE)))
        print('Predicting...')
        for i in tqdm(range((len(idList)+batchSize-1)//batchSize)):
            samples = idList[i*batchSize:(i+1)*batchSize]
            STATESeqMaxLen = STATELen[samples].max()
            batchSeq = torch.tensor([i+[[0,0,0,0,0]]*(STATESeqMaxLen-len(i)) for i in tokenizedSTATE[samples]], dtype=torch.long).to(self.device)
            batchY = F.softmax(self._calculate_y_logit(batchSeq), dim=1).cpu().data.numpy()
            Ypre.append(batchY)
        Ypre = np.vstack(Ypre).astype('float32')
        print('Finished!')
        return Ypre

    def _calculate_y_logit(self, X):
        x,y,v,d,t = self.xEmbedding(X[:,:,0]),self.yEmbedding(X[:,:,1]),self.vEmbedding(X[:,:,2]),self.dEmbedding(X[:,:,3]),self.tEmbedding(X[:,:,4])
        X = torch.cat([x,y,v,d,t], axis=2) # => batchSize × seqLen × feaSize
        X = X.transpose(1,2)
        X = self.textCNN(X)
        return self.fcLinear(X)

class StateAnalyzer2:
    def __init__(self, weightPath, classNum=3, 
                 feaSize=160, filterNum=192, contextSizeList=[1,3,5],hiddenList=[],map_location="cpu",device="cpu", 
                 xySize=32, rSize=32, thetaSize=16, vSize=16, dSize=16, tSize=16,
                 xydiff=500, rdiff=300, thetadiff=2, vdiff=1, ddiff=10, tdiff=10800):
        stateDict = torch.load(weightPath, map_location=map_location)
        self.lab2id,self.id2lab = stateDict['lab2id'],stateDict['id2lab']
        self.x2id,self.id2x = stateDict['x2id'],stateDict['id2x']
        self.y2id,self.id2y = stateDict['y2id'],stateDict['id2y']
        self.r2id,self.id2r = stateDict['r2id'],stateDict['id2r']
        self.theta2id,self.id2theta = stateDict['theta2id'],stateDict['id2theta']
        self.v2id,self.id2v = stateDict['v2id'],stateDict['id2v']
        self.d2id,self.id2d = stateDict['d2id'],stateDict['id2d']
        self.t2id,self.id2t = stateDict['t2id'],stateDict['id2t']
        self.xEmbedding = TextEmbedding( torch.zeros((len(self.id2x),xySize), dtype=torch.float),name='xEmbedding' ).to(device)
        self.yEmbedding = TextEmbedding( torch.zeros((len(self.id2y),xySize), dtype=torch.float),name='yEmbedding' ).to(device)
        self.rEmbedding = TextEmbedding( torch.zeros((len(self.id2r),rSize), dtype=torch.float),name='rEmbedding' ).to(device)
        self.thetaEmbedding = TextEmbedding( torch.zeros((len(self.id2theta),thetaSize), dtype=torch.float),name='thetaEmbedding' ).to(device)
        self.vEmbedding = TextEmbedding( torch.zeros((len(self.id2v),vSize), dtype=torch.float),name='vEmbedding' ).to(device)
        self.dEmbedding = TextEmbedding( torch.zeros((len(self.id2d),dSize), dtype=torch.float),name='dEmbedding' ).to(device)
        self.tEmbedding = TextEmbedding( torch.zeros((len(self.id2t),tSize), dtype=torch.float),name='tEmbedding' ).to(device)
        self.textCNN = TextCNN(feaSize, contextSizeList, filterNum).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterNum, classNum, hiddenList).to(device)
        self.moduleList = nn.ModuleList([self.xEmbedding,self.yEmbedding,self.rEmbedding,self.thetaEmbedding,self.vEmbedding,self.dEmbedding,self.tEmbedding, self.textCNN, self.fcLinear])
        for module in self.moduleList:
            module.load_state_dict(stateDict[module.name])
            module.eval()

        self.xydiff,self.rdiff,self.thetadiff,self.vdiff,self.ddiff,self.tdiff = xydiff,rdiff,thetadiff,vdiff,ddiff,tdiff
        self.device = device
        print("%d epochs and %.3lf val Score 's model load finished."%(stateDict['epochs'], stateDict['bestMtc']))

    def predict(self, STATE, batchSize=64):
        if type(STATE)==str:
            with open(STATE, 'rb') as f:
                STATE = pickle.load(f)
        print('Transforming...')
        STATE = [[[x//self.xydiff,y//self.xydiff,r//self.rdiff,theta//self.thetadiff,v//self.vdiff,d//self.ddiff,t//self.tdiff] for x,y,r,theta,v,d,t in ship] for ship in STATE]
        tokenizedSTATE = np.array([[[
                                        self.x2id[x] if x in self.x2id else 1,
                                        self.y2id[y] if y in self.y2id else 1,
                                        self.r2id[r] if r in self.r2id else 1,
                                        self.theta2id[theta] if theta in self.theta2id else 1,
                                        self.v2id[v] if v in self.v2id else 1,
                                        self.d2id[d] if d in self.d2id else 1,
                                        self.t2id[t] if t in self.t2id else 1
                                    ] for x,y,r,theta,v,d,t in ship] for ship in STATE])
        STATELen = np.array([len(r)+1 for r in STATE],dtype='int32')

        Ypre = []
        idList = list(range(len(tokenizedSTATE)))
        print('Predicting...')
        for i in tqdm(range((len(idList)+batchSize-1)//batchSize)):
            samples = idList[i*batchSize:(i+1)*batchSize]
            STATESeqMaxLen = STATELen[samples].max()
            batchSeq = torch.tensor([i+[[0,0,0,0,0,0,0]]*(STATESeqMaxLen-len(i)) for i in tokenizedSTATE[samples]], dtype=torch.long).to(self.device)
            batchY = F.softmax(self._calculate_y_logit(batchSeq), dim=1).cpu().data.numpy()
            Ypre.append(batchY)
        Ypre = np.vstack(Ypre).astype('float32')
        print('Finished!')
        return Ypre

    def _calculate_y_logit(self, X):
        x,y,r,theta,v,d,t = self.xEmbedding(X[:,:,0]),self.yEmbedding(X[:,:,1]),self.rEmbedding(X[:,:,2]),self.thetaEmbedding(X[:,:,3]),self.vEmbedding(X[:,:,4]),self.dEmbedding(X[:,:,5]),self.tEmbedding(X[:,:,6])
        X = torch.cat([x,y,r,theta,v,d,t], axis=2) # => batchSize × seqLen × feaSize
        X = X.transpose(1,2)
        X = self.textCNN(X)
        return self.fcLinear(X)