from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
import os,logging,pickle,random,torch
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class DataClass_state:
    def __init__(self, trainPath, testPath, labelPath, xydiff=500, vdiff=1, ddiff=10, tdiff=10800, validSize=0.2, minCount=10):
        # Open files and load data
        print('Loading the raw data...')
        with open(trainPath,'rb') as f:
            trainSTATE = pickle.load(f)
        with open(labelPath,'rb') as f:
            trainLab = pickle.load(f)
        if testPath is not None:
            with open(testPath,'rb') as f:
                testSTATE = pickle.load(f)
        else:
            testSTATE = []
        # Get the mapping variables for label and label_id
        print('Getting the mapping variables for label and label id......')
        self.lab2id,self.id2lab = {},[]
        cnt = 0
        for lab in tqdm(trainLab):
            if lab not in self.lab2id:
                self.lab2id[lab] = cnt
                self.id2lab.append(lab)
                cnt += 1
        self.classNum = cnt
        # Discretization
        print('Starting to discretize features......')
        tmp = np.vstack(trainSTATE + testSTATE)
        trainSTATE = [[[x//xydiff,y//xydiff,v//vdiff,d//ddiff,t//tdiff] for x,y,v,d,t in ship] for ship in trainSTATE]
        testSTATE  = [[[x//xydiff,y//xydiff,v//vdiff,d//ddiff,t//tdiff] for x,y,v,d,t in ship] for ship in testSTATE]
        # Dropping uncommon items......
        print('Dropping uncommon items...')
        xCounter,yCounter,vCounter,dCounter,tCounter = {},{},{},{},{}
        for ship in tqdm(trainSTATE+testSTATE):
            for x,y,v,d,t in ship:
                xCounter[x],yCounter[y],vCounter[v],dCounter[d],tCounter[t] = xCounter.get(x,0)+1,yCounter.get(y,0)+1,vCounter.get(v,0)+1,dCounter.get(d,0)+1,tCounter.get(t,0)+1
        trainSTATE = [[ [
                            x if xCounter[x]>=minCount else "<UNK>", 
                            y if yCounter[y]>=minCount else "<UNK>",
                            v if vCounter[v]>=minCount else "<UNK>",
                            d if dCounter[d]>=minCount else "<UNK>",
                            t if tCounter[t]>=minCount else "<UNK>",
                        ] for x,y,v,d,t in ship] for ship in trainSTATE]
        testSTATE = [[  [
                            x if xCounter[x]>=minCount else "<UNK>", 
                            y if yCounter[y]>=minCount else "<UNK>",
                            v if vCounter[v]>=minCount else "<UNK>",
                            d if dCounter[d]>=minCount else "<UNK>",
                            t if tCounter[t]>=minCount else "<UNK>",
                        ] for x,y,v,d,t in ship] for ship in testSTATE]
        # Get the mapping variables for state and state_id
        print('Getting the mapping variables for state and state id......')
        self.x2id,self.id2x = {"<EOS>":0, "<UNK>":1},["<EOS>", "<UNK>"]
        self.y2id,self.id2y = {"<EOS>":0, "<UNK>":1},["<EOS>", "<UNK>"]
        self.v2id,self.id2v = {"<EOS>":0, "<UNK>":1},["<EOS>", "<UNK>"]
        self.d2id,self.id2d = {"<EOS>":0, "<UNK>":1},["<EOS>", "<UNK>"]
        self.t2id,self.id2t = {"<EOS>":0, "<UNK>":1},["<EOS>", "<UNK>"]
        xCnt,yCnt,vCnt,dCnt,tCnt = 2,2,2,2,2
        for ship in tqdm(trainSTATE+testSTATE):
            for x,y,v,d,t in ship:
                if x not in self.x2id and xCounter[x]>=minCount:
                    self.x2id[x] = xCnt
                    self.id2x.append(x)
                    xCnt += 1
                if y not in self.y2id and yCounter[y]>=minCount:
                    self.y2id[y] = yCnt
                    self.id2y.append(y)
                    yCnt += 1
                if v not in self.v2id and vCounter[v]>=minCount:
                    self.v2id[v] = vCnt
                    self.id2v.append(v)
                    vCnt += 1
                if d not in self.d2id and dCounter[d]>=minCount:
                    self.d2id[d] = dCnt
                    self.id2d.append(d)
                    dCnt += 1
                if t not in self.t2id and tCounter[t]>=minCount:
                    self.t2id[t] = tCnt
                    self.id2t.append(t)
                    tCnt += 1
        self.xNum,self.yNum,self.vNum,self.dNum,self.tNum = xCnt,yCnt,vCnt,dCnt,tCnt
        # Tokenize the sentences and labels
        self.trainSTATE,self.testSTATE = trainSTATE,testSTATE
        self.trainTokenizedSTATE = np.array([[[self.x2id[x],self.y2id[y],self.v2id[v],self.d2id[d],self.t2id[t]] for x,y,v,d,t in ship] for ship in trainSTATE])
        self.testTokenizedSTATE = np.array([[[self.x2id[x],self.y2id[y],self.v2id[v],self.d2id[d],self.t2id[t]] for x,y,v,d,t in ship] for ship in testSTATE])
        self.trainLab,self.testLab = np.array( [self.lab2id[i] for i in trainLab],dtype='int32' ),np.array( [-1 for i in trainLab],dtype='int32' )
        self.trainSTATELen,self.testSTATELen = np.array([len(r)+1 for r in self.trainSTATE],dtype='int32'),np.array([len(r)+1 for r in self.testSTATE],dtype='int32')
        self.vector = {}
        print('Start train_valid_test split......')
        trainIdList,validIdList = train_test_split(range(len(trainSTATE)), test_size=validSize, stratify=self.trainLab) if validSize>0.0 else (trainIdList,[])
        testIdList = list(range(len(testSTATE)))
        trainSampleNum,validSampleNum,testSampleNum = len(trainIdList),len(validIdList),len(testIdList)
        totalSampleNum = trainSampleNum + validSampleNum + testSampleNum
        self.trainIdList,self.validIdList,self.testIdList = trainIdList,validIdList,testIdList
        self.trainSampleNum,self.validSampleNum,self.testSampleNum = len(trainIdList),len(validIdList),len(testIdList)
        self.totalSampleNum = self.trainSampleNum+self.validSampleNum+self.testSampleNum
        print('classNum:',self.classNum)
        print(f'xNum:{self.xNum}, yNum:{self.yNum}, vNum:{self.vNum}, dNum:{self.dNum}, tNum:{self.tNum}')
        print('train sample size:',len(self.trainIdList))
        print('valid sample size:',len(self.validIdList))
        print('test sample size:',len(self.testIdList))
        self.xCounter,self.yCounter,self.vCounter,self.dCounter,self.tCounter = xCounter,yCounter,vCounter,dCounter,tCounter
        self.minCount = minCount

    def describe(self):
        print(f'===========DataClass Describe===========')
        print(f'{"CLASS":<16}{"TRAIN":<8}{"VALID":<8}')
        for i,c in enumerate(self.id2lab):
            trainIsC = sum(self.trainLab[self.trainIdList]==i)/self.trainSampleNum if self.trainSampleNum>0 else -1.0
            validIsC = sum(self.trainLab[self.validIdList]==i)/self.validSampleNum if self.validSampleNum>0 else -1.0
            print(f'{c:<16}{trainIsC:<8.3f}{validIsC:<8.3f}')
        print(f'========================================')

    def vectorize(self, method="char2vec", xySize=64, vSize=32, dSize=32, tSize=32, window=13, sg=1, 
                        workers=8, iters=10, loadCache=True):
        if os.path.exists(f'cache/{method}_xy{xySize}_v{vSize}_d{dSize}_t{tSize}.pkl') and loadCache:
            with open(f'cache/{method}_xy{xySize}_v{vSize}_d{dSize}_t{tSize}.pkl', 'rb') as f:
                self.vector['embedding'] = pickle.load(f)
            print(f'Loaded cache from cache/{method}_xy{xySize}_v{vSize}_d{dSize}_t{tSize}.pkl.')
            return
        self.vector['embedding'] = {}
        if method == 'char2vec':
            print('training x chars...')
            xDoc = [[str(j[0]) for j in i]+['<EOS>'] for i in self.trainSTATE+self.testSTATE]
            model = Word2Vec(xDoc, min_count=self.minCount, window=window, size=xySize, workers=workers, sg=sg, iter=iters)
            char2vec = np.zeros((self.xNum, xySize), dtype=np.float32)
            for i in range(self.xNum):
                char2vec[i] = model.wv[str(self.id2x[i])] if str(self.id2x[i]) in model.wv else np.random.random((1,xySize))
            self.vector['embedding']['x'] = char2vec
            print('training y chars...')
            yDoc = [[str(j[1]) for j in i]+['<EOS>'] for i in self.trainSTATE+self.testSTATE]
            model = Word2Vec(yDoc, min_count=self.minCount, window=window, size=xySize, workers=workers, sg=sg, iter=iters)
            char2vec = np.zeros((self.yNum, xySize), dtype=np.float32)
            for i in range(self.yNum):
                char2vec[i] = model.wv[str(self.id2y[i])] if str(self.id2y[i]) in model.wv else np.random.random((1,xySize))
            self.vector['embedding']['y'] = char2vec
            print('training v chars...')
            vDoc = [[str(j[2]) for j in i]+['<EOS>'] for i in self.trainSTATE+self.testSTATE]
            model = Word2Vec(vDoc, min_count=self.minCount, window=window, size=vSize, workers=workers, sg=sg, iter=iters)
            char2vec = np.zeros((self.vNum, vSize), dtype=np.float32)
            for i in range(self.vNum):
                char2vec[i] = model.wv[str(self.id2v[i])] if str(self.id2v[i]) in model.wv else np.random.random((1,vSize))
            self.vector['embedding']['v'] = char2vec
            print('training d chars...')
            dDoc = [[str(j[3]) for j in i]+['<EOS>'] for i in self.trainSTATE+self.testSTATE]
            model = Word2Vec(dDoc, min_count=self.minCount, window=window, size=dSize, workers=workers, sg=sg, iter=iters)
            char2vec = np.zeros((self.dNum, dSize), dtype=np.float32)
            for i in range(self.dNum):
                char2vec[i] = model.wv[str(self.id2d[i])] if str(self.id2d[i]) in model.wv else np.random.random((1,dSize))
            self.vector['embedding']['d'] = char2vec
            print('training t chars...')
            tDoc = [[str(j[4]) for j in i]+['<EOS>'] for i in self.trainSTATE+self.testSTATE]
            model = Word2Vec(tDoc, min_count=self.minCount, window=window, size=tSize, workers=workers, sg=sg, iter=iters)
            char2vec = np.zeros((self.tNum, tSize), dtype=np.float32)
            for i in range(self.tNum):
                char2vec[i] = model.wv[str(self.id2t[i])] if str(self.id2t[i]) in model.wv else np.random.random((1,tSize))
            self.vector['embedding']['t'] = char2vec

        with open(f'cache/{method}_xy{xySize}_v{vSize}_d{dSize}_t{tSize}.pkl', 'wb') as f:
            pickle.dump(self.vector['embedding'], f, protocol=4)

    def random_batch_data_stream(self, batchSize=128, type='train', device=torch.device('cpu')):
        idList = [i for i in self.trainIdList] if type=='train' else [i for i in self.validIdList]
        X,XLen,Lab = self.trainTokenizedSTATE,self.trainSTATELen,self.trainLab
        while True:
            random.shuffle(idList)
            for i in range((len(idList)+batchSize-1)//batchSize):
                samples = idList[i*batchSize:(i+1)*batchSize]
                STATESeqMaxLen = XLen[samples].max()
                yield {
                        "seqArr":torch.tensor([i+[[0,0,0,0,0]]*(STATESeqMaxLen-len(i)) for i in X[samples]], dtype=torch.long).to(device), \
                        "seqLenArr":torch.tensor(XLen[samples], dtype=torch.int).to(device)
                      }, torch.tensor(Lab[samples], dtype=torch.long).to(device)
    
    def one_epoch_batch_data_stream(self, batchSize=128, type='valid', device=torch.device('cpu')):
        if type == 'train':
            X,XLen,Lab = self.trainTokenizedSTATE,self.trainSTATELen,self.trainLab
            idList = self.trainIdList
        elif type == 'valid':
            X,XLen,Lab = self.trainTokenizedSTATE,self.trainSTATELen,self.trainLab
            idList = self.validIdList
        elif type == 'test':
            X,XLen,Lab = self.testTokenizedSTATE,self.testSTATELen,self.testLab
            idList = self.testIdList
        for i in range((len(idList)+batchSize-1)//batchSize):
            samples = idList[i*batchSize:(i+1)*batchSize]
            STATESeqMaxLen = XLen[samples].max()
            yield {
                    "seqArr":torch.tensor([i+[[0,0,0,0,0]]*(STATESeqMaxLen-len(i)) for i in X[samples]], dtype=torch.long).to(device), \
                    "seqLenArr":torch.tensor(XLen[samples], dtype=torch.int).to(device)
                  }, torch.tensor(Lab[samples], dtype=torch.long).to(device)

class DataClass_state2:
    def __init__(self, trainPath, testPath, labelPath, xydiff=500, rdiff=300, thetadiff=2, vdiff=1, ddiff=10, tdiff=10800, validSize=0.2, minCount=10):
        # Open files and load data
        print('Loading the raw data...')
        with open(trainPath,'rb') as f:
            trainSTATE = pickle.load(f)
        with open(labelPath,'rb') as f:
            trainLab = pickle.load(f)
        if testPath is not None:
            with open(testPath,'rb') as f:
                testSTATE = pickle.load(f)
        else:
            testSTATE = []
        # Get the mapping variables for label and label_id
        print('Getting the mapping variables for label and label id......')
        self.lab2id,self.id2lab = {},[]
        cnt = 0
        for lab in tqdm(trainLab):
            if lab not in self.lab2id:
                self.lab2id[lab] = cnt
                self.id2lab.append(lab)
                cnt += 1
        self.classNum = cnt
        # Discretization
        print('Starting to discretize features......')
        tmp = np.vstack(trainSTATE + testSTATE)
        trainSTATE = [[[x//xydiff,y//xydiff,r//rdiff,theta//thetadiff,v//vdiff,d//ddiff,t//tdiff] for x,y,r,theta,v,d,t in ship] for ship in trainSTATE]
        testSTATE  = [[[x//xydiff,y//xydiff,r//rdiff,theta//thetadiff,v//vdiff,d//ddiff,t//tdiff] for x,y,r,theta,v,d,t in ship] for ship in testSTATE]
        # Dropping uncommon items......
        print('Dropping uncommon items...')
        xCounter,yCounter,rCounter,thetaCounter,vCounter,dCounter,tCounter = {},{},{},{},{},{},{}
        for ship in tqdm(trainSTATE+testSTATE):
            for x,y,r,theta,v,d,t in ship:
                xCounter[x],yCounter[y],rCounter[r],thetaCounter[theta],vCounter[v],dCounter[d],tCounter[t] = xCounter.get(x,0)+1,yCounter.get(y,0)+1,rCounter.get(r,0)+1,thetaCounter.get(theta,0)+1,vCounter.get(v,0)+1,dCounter.get(d,0)+1,tCounter.get(t,0)+1
        trainSTATE = [[ [
                            x if xCounter[x]>=minCount else "<UNK>", 
                            y if yCounter[y]>=minCount else "<UNK>",
                            r if rCounter[r]>=minCount else "<UNK>",
                            theta if thetaCounter[theta]>=minCount else "<UNK>",
                            v if vCounter[v]>=minCount else "<UNK>",
                            d if dCounter[d]>=minCount else "<UNK>",
                            t if tCounter[t]>=minCount else "<UNK>",
                        ] for x,y,r,theta,v,d,t in ship] for ship in trainSTATE]
        testSTATE = [[  [
                            x if xCounter[x]>=minCount else "<UNK>", 
                            y if yCounter[y]>=minCount else "<UNK>",
                            r if rCounter[r]>=minCount else "<UNK>",
                            theta if thetaCounter[theta]>=minCount else "<UNK>",
                            v if vCounter[v]>=minCount else "<UNK>",
                            d if dCounter[d]>=minCount else "<UNK>",
                            t if tCounter[t]>=minCount else "<UNK>",
                        ] for x,y,r,theta,v,d,t in ship] for ship in testSTATE]
        # Get the mapping variables for state and state_id
        print('Getting the mapping variables for state and state id......')
        self.x2id,self.id2x = {"<EOS>":0, "<UNK>":1},["<EOS>", "<UNK>"]
        self.y2id,self.id2y = {"<EOS>":0, "<UNK>":1},["<EOS>", "<UNK>"]
        self.r2id,self.id2r = {"<EOS>":0, "<UNK>":1},["<EOS>", "<UNK>"]
        self.theta2id,self.id2theta = {"<EOS>":0, "<UNK>":1},["<EOS>", "<UNK>"]
        self.v2id,self.id2v = {"<EOS>":0, "<UNK>":1},["<EOS>", "<UNK>"]
        self.d2id,self.id2d = {"<EOS>":0, "<UNK>":1},["<EOS>", "<UNK>"]
        self.t2id,self.id2t = {"<EOS>":0, "<UNK>":1},["<EOS>", "<UNK>"]
        xCnt,yCnt,rCnt,thetaCnt,vCnt,dCnt,tCnt = 2,2,2,2,2,2,2
        for ship in tqdm(trainSTATE+testSTATE):
            for x,y,r,theta,v,d,t in ship:
                if x not in self.x2id and xCounter[x]>=minCount:
                    self.x2id[x] = xCnt
                    self.id2x.append(x)
                    xCnt += 1
                if y not in self.y2id and yCounter[y]>=minCount:
                    self.y2id[y] = yCnt
                    self.id2y.append(y)
                    yCnt += 1
                if r not in self.r2id and rCounter[r]>=minCount:
                    self.r2id[r] = rCnt
                    self.id2r.append(r)
                    rCnt += 1
                if theta not in self.theta2id and thetaCounter[theta]>=minCount:
                    self.theta2id[theta] = thetaCnt
                    self.id2theta.append(theta)
                    thetaCnt += 1
                if v not in self.v2id and vCounter[v]>=minCount:
                    self.v2id[v] = vCnt
                    self.id2v.append(v)
                    vCnt += 1
                if d not in self.d2id and dCounter[d]>=minCount:
                    self.d2id[d] = dCnt
                    self.id2d.append(d)
                    dCnt += 1
                if t not in self.t2id and tCounter[t]>=minCount:
                    self.t2id[t] = tCnt
                    self.id2t.append(t)
                    tCnt += 1
        self.xNum,self.yNum,self.rNum,self.thetaNum,self.vNum,self.dNum,self.tNum = xCnt,yCnt,rCnt,thetaCnt,vCnt,dCnt,tCnt
        # Tokenize the sentences and labels
        self.trainSTATE,self.testSTATE = trainSTATE,testSTATE
        self.trainTokenizedSTATE = np.array([[[self.x2id[x],self.y2id[y],self.r2id[r],self.theta2id[theta],self.v2id[v],self.d2id[d],self.t2id[t]] for x,y,r,theta,v,d,t in ship] for ship in trainSTATE])
        self.testTokenizedSTATE = np.array([[[self.x2id[x],self.y2id[y],self.r2id[r],self.theta2id[theta],self.v2id[v],self.d2id[d],self.t2id[t]] for x,y,r,theta,v,d,t in ship] for ship in testSTATE])
        self.trainLab,self.testLab = np.array( [self.lab2id[i] for i in trainLab],dtype='int32' ),np.array( [-1 for i in trainLab],dtype='int32' )
        self.trainSTATELen,self.testSTATELen = np.array([len(r)+1 for r in self.trainSTATE],dtype='int32'),np.array([len(r)+1 for r in self.testSTATE],dtype='int32')
        self.vector = {}
        print('Start train_valid_test split......')
        trainIdList,validIdList = train_test_split(range(len(trainSTATE)), test_size=validSize) if validSize>0.0 else (trainIdList,[])
        testIdList = list(range(len(testSTATE)))
        trainSampleNum,validSampleNum,testSampleNum = len(trainIdList),len(validIdList),len(testIdList)
        totalSampleNum = trainSampleNum + validSampleNum + testSampleNum
        self.trainIdList,self.validIdList,self.testIdList = trainIdList,validIdList,testIdList
        self.trainSampleNum,self.validSampleNum,self.testSampleNum = len(trainIdList),len(validIdList),len(testIdList)
        self.totalSampleNum = self.trainSampleNum+self.validSampleNum+self.testSampleNum
        print('classNum:',self.classNum)
        print(f'xNum:{self.xNum}, yNum:{self.yNum}, rNum:{self.rNum}, thetaNum:{self.thetaNum}, vNum:{self.vNum}, dNum:{self.dNum}, tNum:{self.tNum}')
        print('train sample size:',len(self.trainIdList))
        print('valid sample size:',len(self.validIdList))
        print('test sample size:',len(self.testIdList))
        self.xCounter,self.yCounter,self.rCounter,self.thetaCounter,self.vCounter,self.dCounter,self.tCounter = xCounter,yCounter,rCounter,thetaCounter,vCounter,dCounter,tCounter
        self.minCount = minCount

    def describe(self):
        print(f'===========DataClass Describe===========')
        print(f'{"CLASS":<16}{"TRAIN":<8}{"VALID":<8}')
        for i,c in enumerate(self.id2lab):
            trainIsC = sum(self.trainLab[self.trainIdList]==i)/self.trainSampleNum if self.trainSampleNum>0 else -1.0
            validIsC = sum(self.trainLab[self.validIdList]==i)/self.validSampleNum if self.validSampleNum>0 else -1.0
            print(f'{c:<16}{trainIsC:<8.3f}{validIsC:<8.3f}')
        print(f'========================================')

    def vectorize(self, method="char2vec", xySize=32, rSize=32, thetaSize=16, vSize=16, dSize=16, tSize=16, window=13, sg=1, 
                        workers=8, loadCache=True):
        if os.path.exists(f'cache/{method}_xy{xySize}_r{rSize}_theta{thetaSize}_v{vSize}_d{dSize}_t{tSize}.pkl') and loadCache:
            with open(f'cache/{method}_xy{xySize}_r{rSize}_theta{thetaSize}_v{vSize}_d{dSize}_t{tSize}.pkl', 'rb') as f:
                self.vector['embedding'] = pickle.load(f)
            print(f'Loaded cache from cache/{method}_xy{xySize}_r{rSize}_theta{thetaSize}_v{vSize}_d{dSize}_t{tSize}.pkl.')
            return
        self.vector['embedding'] = {}
        if method == 'char2vec':
            print('training x chars...')
            xDoc = [[str(j[0]) for j in i]+['<EOS>'] for i in self.trainSTATE+self.testSTATE]
            model = Word2Vec(xDoc, min_count=self.minCount, window=window, size=xySize, workers=workers, sg=sg, iter=10)
            char2vec = np.zeros((self.xNum, xySize), dtype=np.float32)
            for i in range(self.xNum):
                char2vec[i] = model.wv[str(self.id2x[i])] if str(self.id2x[i]) in model.wv else np.random.random((1,xySize))
            self.vector['embedding']['x'] = char2vec
            print('training y chars...')
            yDoc = [[str(j[1]) for j in i]+['<EOS>'] for i in self.trainSTATE+self.testSTATE]
            model = Word2Vec(yDoc, min_count=self.minCount, window=window, size=xySize, workers=workers, sg=sg, iter=10)
            char2vec = np.zeros((self.yNum, xySize), dtype=np.float32)
            for i in range(self.yNum):
                char2vec[i] = model.wv[str(self.id2y[i])] if str(self.id2y[i]) in model.wv else np.random.random((1,xySize))
            self.vector['embedding']['y'] = char2vec
            print('training r chars...')
            rDoc = [[str(j[2]) for j in i]+['<EOS>'] for i in self.trainSTATE+self.testSTATE]
            model = Word2Vec(rDoc, min_count=self.minCount, window=window, size=rSize, workers=workers, sg=sg, iter=10)
            char2vec = np.zeros((self.rNum, rSize), dtype=np.float32)
            for i in range(self.rNum):
                char2vec[i] = model.wv[str(self.id2r[i])] if str(self.id2r[i]) in model.wv else np.random.random((1,rSize))
            self.vector['embedding']['r'] = char2vec
            print('training theta chars...')
            thetaDoc = [[str(j[3]) for j in i]+['<EOS>'] for i in self.trainSTATE+self.testSTATE]
            model = Word2Vec(thetaDoc, min_count=self.minCount, window=window, size=thetaSize, workers=workers, sg=sg, iter=10)
            char2vec = np.zeros((self.thetaNum, thetaSize), dtype=np.float32)
            for i in range(self.thetaNum):
                char2vec[i] = model.wv[str(self.id2theta[i])] if str(self.id2theta[i]) in model.wv else np.random.random((1,thetaSize))
            self.vector['embedding']['theta'] = char2vec
            print('training v chars...')
            vDoc = [[str(j[4]) for j in i]+['<EOS>'] for i in self.trainSTATE+self.testSTATE]
            model = Word2Vec(vDoc, min_count=self.minCount, window=window, size=vSize, workers=workers, sg=sg, iter=10)
            char2vec = np.zeros((self.vNum, vSize), dtype=np.float32)
            for i in range(self.vNum):
                char2vec[i] = model.wv[str(self.id2v[i])] if str(self.id2v[i]) in model.wv else np.random.random((1,vSize))
            self.vector['embedding']['v'] = char2vec
            print('training d chars...')
            dDoc = [[str(j[5]) for j in i]+['<EOS>'] for i in self.trainSTATE+self.testSTATE]
            model = Word2Vec(dDoc, min_count=self.minCount, window=window, size=dSize, workers=workers, sg=sg, iter=10)
            char2vec = np.zeros((self.dNum, dSize), dtype=np.float32)
            for i in range(self.dNum):
                char2vec[i] = model.wv[str(self.id2d[i])] if str(self.id2d[i]) in model.wv else np.random.random((1,dSize))
            self.vector['embedding']['d'] = char2vec
            print('training t chars...')
            tDoc = [[str(j[6]) for j in i]+['<EOS>'] for i in self.trainSTATE+self.testSTATE]
            model = Word2Vec(tDoc, min_count=self.minCount, window=window, size=tSize, workers=workers, sg=sg, iter=10)
            char2vec = np.zeros((self.tNum, tSize), dtype=np.float32)
            for i in range(self.tNum):
                char2vec[i] = model.wv[str(self.id2t[i])] if str(self.id2t[i]) in model.wv else np.random.random((1,tSize))
            self.vector['embedding']['t'] = char2vec

        with open(f'cache/{method}_xy{xySize}_r{rSize}_theta{thetaSize}_v{vSize}_d{dSize}_t{tSize}.pkl', 'wb') as f:
            pickle.dump(self.vector['embedding'], f, protocol=4)

    def random_batch_data_stream(self, batchSize=128, type='train', device=torch.device('cpu')):
        idList = [i for i in self.trainIdList] if type=='train' else [i for i in self.validIdList]
        X,XLen,Lab = self.trainTokenizedSTATE,self.trainSTATELen,self.trainLab
        while True:
            random.shuffle(idList)
            for i in range((len(idList)+batchSize-1)//batchSize):
                samples = idList[i*batchSize:(i+1)*batchSize]
                STATESeqMaxLen = XLen[samples].max()
                yield {
                        "seqArr":torch.tensor([i+[[0,0,0,0,0,0,0]]*(STATESeqMaxLen-len(i)) for i in X[samples]], dtype=torch.long).to(device), \
                        "seqLenArr":torch.tensor(XLen[samples], dtype=torch.int).to(device)
                      }, torch.tensor(Lab[samples], dtype=torch.long).to(device)
    
    def one_epoch_batch_data_stream(self, batchSize=128, type='valid', device=torch.device('cpu')):
        if type == 'train':
            X,XLen,Lab = self.trainTokenizedSTATE,self.trainSTATELen,self.trainLab
            idList = self.trainIdList
        elif type == 'valid':
            X,XLen,Lab = self.trainTokenizedSTATE,self.trainSTATELen,self.trainLab
            idList = self.validIdList
        elif type == 'test':
            X,XLen,Lab = self.testTokenizedSTATE,self.testSTATELen,self.testLab
            idList = self.testIdList
        for i in range((len(idList)+batchSize-1)//batchSize):
            samples = idList[i*batchSize:(i+1)*batchSize]
            STATESeqMaxLen = XLen[samples].max()
            yield {
                    "seqArr":torch.tensor([i+[[0,0,0,0,0,0,0]]*(STATESeqMaxLen-len(i)) for i in X[samples]], dtype=torch.long).to(device), \
                    "seqLenArr":torch.tensor(XLen[samples], dtype=torch.int).to(device)
                  }, torch.tensor(Lab[samples], dtype=torch.long).to(device)