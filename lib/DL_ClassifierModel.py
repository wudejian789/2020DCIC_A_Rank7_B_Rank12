import numpy as np
import torch,time,os,pickle,random
from torch import nn as nn
from .nnLayer import *
from .metrics import *
from collections import Counter,Iterable
from sklearn.model_selection import StratifiedKFold

class BaseClassifier:
    def __init__(self):
        pass
    def calculate_y_logit(self, X, XLen):
        pass
    def cv_train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, 
                 optimType='Adam', lr=0.001, weightDecay=0, kFold=5, isHigherBetter=True, metrics="MaF", report=["ACC", "MaF"], 
                 savePath='model', seed=9527):
        skf = StratifiedKFold(n_splits=kFold, random_state=seed, shuffle=True)
        validRes,testRes = [],[]
        tvIdList = list(range(dataClass.totalSampleNum))#dataClass.trainIdList+dataClass.validIdList
        for i,(trainIndices,validIndices) in enumerate(skf.split(tvIdList, dataClass.trainLab)):
            print(f'CV_{i+1}:')
            self.reset_parameters()
            dataClass.trainIdList,dataClass.validIdList = trainIndices,validIndices
            dataClass.trainSampleNum,dataClass.validSampleNum = len(trainIndices),len(validIndices)
            res = self.train(dataClass,trainSize,batchSize,epoch,stopRounds,earlyStop,saveRounds,optimType,lr,weightDecay,
                             isHigherBetter,metrics,report,f"{savePath}_cv{i+1}")
            validRes.append(res)
        Metrictor.table_show(validRes, report)
    def train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, 
              optimType='Adam', lr=0.001, weightDecay=0, isHigherBetter=True, metrics="MaF", report=["ACC", "MiF"], 
              savePath='model'):
        dataClass.describe()
        assert batchSize%trainSize==0
        metrictor = Metrictor(dataClass.classNum)
        self.stepCounter = 0
        self.stepUpdate = batchSize//trainSize
        optimizer = torch.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        trainStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='train', device=self.device)
        itersPerEpoch = (dataClass.trainSampleNum+trainSize-1)//trainSize
        mtc,bestMtc,stopSteps = 0.0,0.0,0
        if dataClass.validSampleNum>0: validStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='valid', device=self.device)
        st = time.time()
        for e in range(epoch):
            for i in range(itersPerEpoch):
                self.to_train_mode()
                X,Y = next(trainStream)
                loss = self._train_step(X,Y, optimizer)
                if stopRounds>0 and (e*itersPerEpoch+i+1)%stopRounds==0:
                    self.to_eval_mode()
                    print(f"After iters {e*itersPerEpoch+i+1}: [train] loss= {loss:.3f};", end='')
                    if dataClass.validSampleNum>0:
                        X,Y = next(validStream)
                        loss = self.calculate_loss(X,Y)
                        print(f' [valid] loss= {loss:.3f};', end='')
                    restNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*trainSize
                    speed = (e*itersPerEpoch+i+1)*trainSize/(time.time()-st)
                    print(" speed: %.3lf items/s; remaining time: %.3lfs;"%(speed, restNum/speed))
            if dataClass.validSampleNum>0 and (e+1)%saveRounds==0:
                self.to_eval_mode()
                print(f'========== Epoch:{e+1:5d} ==========')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
                metrictor.set_data(Y_pre, Y)
                print(f'[Total Train]',end='')
                metrictor(report)
                print(f'[Total Valid]',end='')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
                metrictor.set_data(Y_pre, Y)
                res = metrictor(report)
                mtc = res[metrics]
                print('=================================')
                if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                    print(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                    bestMtc = mtc
                    self.save("%s.pkl"%savePath, e+1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps>=earlyStop:
                        print(f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e+1}, stop training.')
                        break
        self.load("%s.pkl"%savePath)
        os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, ("%.3lf"%bestMtc)[2:]))
        print(f'============ Result ============')
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Train]',end='')
        metrictor(report)
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Valid]',end='')
        res = metrictor(report)
        metrictor.each_class_indictor_show(dataClass.id2lab)
        print(f'================================')
        return res
    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            stateDict['lab2id'],stateDict['id2lab'] = dataClass.lab2id,dataClass.id2lab
            stateDict['loc2id'],stateDict['id2loc'] = dataClass.loc2id,dataClass.id2loc
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            if "trainIdList" in parameters:
                dataClass.trainIdList = parameters['trainIdList']
            if "validIdList" in parameters:
                dataClass.validIdList = parameters['validIdList']
            if "testIdList" in parameters:
                dataClass.testIdList = parameters['testIdList']
            dataClass.lab2id,dataClass.id2lab = parameters['lab2id'],parameters['id2lab']
            dataClass.loc2id,dataClass.id2loc = parameters['loc2id'],parameters['id2loc']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=1)
    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=1)
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        return self.criterion(Y_logit, Y)
    def calculate_indicator_by_iterator(self, dataStream, classNum, report):
        metrictor = Metrictor(classNum)
        Y_prob_pre,Y = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(Y_prob_pre, Y)
        return metrictor(report)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.hstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr
    def calculate_y_by_iterator(self, dataStream):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return Y_preArr.argmax(axis=1), YArr
    def to_train_mode(self):
        for module in self.moduleList:
            module.train()
    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()
    def _train_step(self, X, Y, optimizer):
        self.stepCounter += 1
        if self.stepCounter<self.stepUpdate:
            p = False
        else:
            self.stepCounter = 0
            p = True
        loss = self.calculate_loss(X, Y)/self.stepUpdate
        loss.backward()
        if p:
            optimizer.step()
            optimizer.zero_grad()
        return loss*self.stepUpdate

class TextClassifier_MLP(BaseClassifier):
    def __init__(self, classNum, feaSize=256, dropout=0.5, hiddenList=[], 
                 useFocalLoss=False, weight=None, device=torch.device("cuda:0")):
        self.fcLinear = MLP(feaSize, classNum, hiddenList, dropout).to(device)
        self.moduleList = nn.ModuleList([self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight)
    def calculate_y_logit(self, X):
        return self.fcLinear(X['textVector'])

class TextClassifier_FastText(BaseClassifier):
    def __init__(self, classNum, embedding, feaSize=512, hiddenList=[], 
                 embDropout=0.3, fcDropout=0.5, 
                 useFocalLoss=False, weight=None, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding( torch.tensor(embedding, dtype=torch.float),dropout=embDropout ).to(device)
        self.fastText = FastText(feaSize).to(device)
        self.fcLinear = MLP(feaSize, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding, self.fastText, self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight)
    def calculate_y_logit(self, X):
        X,XLen = X['seqArr'],X['seqLenArr']
        X = self.textEmbedding(X)
        X = self.fastText(X, XLen)
        return self.fcLinear(X)

'''
class TextClassifier_StateCNN(BaseClassifier):
    def __init__(self, classNum, feaSize=4, filterNum=64, contextSizeList=[1,3,5], hiddenList=[], 
                 fcDropout=0.5, 
                 useFocalLoss=False, weight=None, device=torch.device("cuda:0")):
        self.textCNN = TextCNN(feaSize, contextSizeList, filterNum).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterNum, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textCNN, self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight)
    def calculate_y_logit(self, X):
        X = X['seqArr'].transpose(1,2)
        X = self.textCNN(X)
        return self.fcLinear(X)
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            stateDict['lab2id'],stateDict['id2lab'] = dataClass.lab2id,dataClass.id2lab
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            if "trainIdList" in parameters:
                dataClass.trainIdList = parameters['trainIdList']
            if "validIdList" in parameters:
                dataClass.validIdList = parameters['validIdList']
            if "testIdList" in parameters:
                dataClass.testIdList = parameters['testIdList']
            dataClass.lab2id,dataClass.id2lab = parameters['lab2id'],parameters['id2lab']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))
'''
class TextClassifier_StateCNN(BaseClassifier):
    def __init__(self, classNum, embedding, feaSize=416, filterNum=448, contextSizeList=[1,3,5], hiddenList=[], 
                 embDropout=0.3, fcDropout=0.5, 
                 useFocalLoss=False, weight=None, device=torch.device("cuda:0")):
        self.xEmbedding = TextEmbedding( torch.tensor(embedding['x'], dtype=torch.float),dropout=embDropout,name='xEmbedding' ).to(device)
        self.yEmbedding = TextEmbedding( torch.tensor(embedding['y'], dtype=torch.float),dropout=embDropout,name='yEmbedding' ).to(device)
        self.vEmbedding = TextEmbedding( torch.tensor(embedding['v'], dtype=torch.float),dropout=embDropout,name='vEmbedding' ).to(device)
        self.dEmbedding = TextEmbedding( torch.tensor(embedding['d'], dtype=torch.float),dropout=embDropout,name='dEmbedding' ).to(device)
        self.tEmbedding = TextEmbedding( torch.tensor(embedding['t'], dtype=torch.float),dropout=embDropout,name='tEmbedding' ).to(device)
        self.textCNN = TextCNN(feaSize, contextSizeList, filterNum).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterNum, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.xEmbedding,self.yEmbedding,self.vEmbedding,self.dEmbedding,self.tEmbedding, self.textCNN, self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight)
    def calculate_y_logit(self, X):
        X = X['seqArr']
        x,y,v,d,t = self.xEmbedding(X[:,:,0]),self.yEmbedding(X[:,:,1]),self.vEmbedding(X[:,:,2]),self.dEmbedding(X[:,:,3]),self.tEmbedding(X[:,:,4])
        X = torch.cat([x,y,v,d,t], axis=2) # => batchSize × seqLen × feaSize
        X = X.transpose(1,2)
        X = self.textCNN(X)
        return self.fcLinear(X)
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            stateDict['lab2id'],stateDict['id2lab'] = dataClass.lab2id,dataClass.id2lab
            stateDict['x2id'],stateDict['id2x'] = dataClass.x2id,dataClass.id2x
            stateDict['y2id'],stateDict['id2y'] = dataClass.y2id,dataClass.id2y
            stateDict['v2id'],stateDict['id2v'] = dataClass.v2id,dataClass.id2v
            stateDict['d2id'],stateDict['id2d'] = dataClass.d2id,dataClass.id2d
            stateDict['t2id'],stateDict['id2t'] = dataClass.t2id,dataClass.id2t
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            if "trainIdList" in parameters:
                dataClass.trainIdList = parameters['trainIdList']
            if "validIdList" in parameters:
                dataClass.validIdList = parameters['validIdList']
            if "testIdList" in parameters:
                dataClass.testIdList = parameters['testIdList']
            dataClass.lab2id,dataClass.id2lab = parameters['lab2id'],parameters['id2lab']
            dataClass.x2id,dataClass.id2x = parameters['x2id'],parameters['id2x']
            dataClass.y2id,dataClass.id2y = parameters['y2id'],parameters['id2y']
            dataClass.v2id,dataClass.id2v = parameters['v2id'],parameters['id2v']
            dataClass.d2id,dataClass.id2d = parameters['d2id'],parameters['id2d']
            dataClass.t2id,dataClass.id2t = parameters['t2id'],parameters['id2t']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))

class TextClassifier_StateCNN2(BaseClassifier):
    def __init__(self, classNum, embedding, feaSize=608, filterNum=448, contextSizeList=[1,3,5], hiddenList=[], 
                 embDropout=0.3, fcDropout=0.5, 
                 useFocalLoss=False, weight=None, device=torch.device("cuda:0")):
        self.xEmbedding = TextEmbedding( torch.tensor(embedding['x'], dtype=torch.float),dropout=embDropout,name='xEmbedding' ).to(device)
        self.yEmbedding = TextEmbedding( torch.tensor(embedding['y'], dtype=torch.float),dropout=embDropout,name='yEmbedding' ).to(device)
        self.rEmbedding = TextEmbedding( torch.tensor(embedding['r'], dtype=torch.float),dropout=embDropout,name='rEmbedding' ).to(device)
        self.thetaEmbedding = TextEmbedding( torch.tensor(embedding['theta'], dtype=torch.float),dropout=embDropout,name='thetaEmbedding' ).to(device)
        self.vEmbedding = TextEmbedding( torch.tensor(embedding['v'], dtype=torch.float),dropout=embDropout,name='vEmbedding' ).to(device)
        self.dEmbedding = TextEmbedding( torch.tensor(embedding['d'], dtype=torch.float),dropout=embDropout,name='dEmbedding' ).to(device)
        self.tEmbedding = TextEmbedding( torch.tensor(embedding['t'], dtype=torch.float),dropout=embDropout,name='tEmbedding' ).to(device)
        self.textCNN = TextCNN(feaSize, contextSizeList, filterNum).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterNum, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.xEmbedding,self.yEmbedding,self.rEmbedding,self.thetaEmbedding,self.vEmbedding,self.dEmbedding,self.tEmbedding, self.textCNN, self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight)
    def calculate_y_logit(self, X):
        X = X['seqArr']
        x,y,r,theta,v,d,t = self.xEmbedding(X[:,:,0]),self.yEmbedding(X[:,:,1]),self.rEmbedding(X[:,:,2]),self.thetaEmbedding(X[:,:,3]),self.vEmbedding(X[:,:,4]),self.dEmbedding(X[:,:,5]),self.tEmbedding(X[:,:,6])
        X = torch.cat([x,y,r,theta,v,d,t], axis=2) # => batchSize × seqLen × feaSize
        X = X.transpose(1,2)
        X = self.textCNN(X)
        return self.fcLinear(X)
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            stateDict['lab2id'],stateDict['id2lab'] = dataClass.lab2id,dataClass.id2lab
            stateDict['x2id'],stateDict['id2x'] = dataClass.x2id,dataClass.id2x
            stateDict['y2id'],stateDict['id2y'] = dataClass.y2id,dataClass.id2y
            stateDict['r2id'],stateDict['id2r'] = dataClass.r2id,dataClass.id2r
            stateDict['theta2id'],stateDict['id2theta'] = dataClass.theta2id,dataClass.id2theta
            stateDict['v2id'],stateDict['id2v'] = dataClass.v2id,dataClass.id2v
            stateDict['d2id'],stateDict['id2d'] = dataClass.d2id,dataClass.id2d
            stateDict['t2id'],stateDict['id2t'] = dataClass.t2id,dataClass.id2t
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            if "trainIdList" in parameters:
                dataClass.trainIdList = parameters['trainIdList']
            if "validIdList" in parameters:
                dataClass.validIdList = parameters['validIdList']
            if "testIdList" in parameters:
                dataClass.testIdList = parameters['testIdList']
            dataClass.lab2id,dataClass.id2lab = parameters['lab2id'],parameters['id2lab']
            dataClass.x2id,dataClass.id2x = parameters['x2id'],parameters['id2x']
            dataClass.y2id,dataClass.id2y = parameters['y2id'],parameters['id2y']
            dataClass.r2id,dataClass.id2r = parameters['r2id'],parameters['id2r']
            dataClass.theta2id,dataClass.id2theta = parameters['theta2id'],parameters['id2theta']
            dataClass.v2id,dataClass.id2v = parameters['v2id'],parameters['id2v']
            dataClass.d2id,dataClass.id2d = parameters['d2id'],parameters['id2d']
            dataClass.t2id,dataClass.id2t = parameters['t2id'],parameters['id2t']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))

class TextClassifier_CNN(BaseClassifier):
    def __init__(self, classNum, embedding, feaSize=512, filterNum=448, contextSizeList=[1,3,5], hiddenList=[], 
                 embDropout=0.3, fcDropout=0.5, isFreezon=False,
                 useFocalLoss=False, weight=None, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding( torch.tensor(embedding, dtype=torch.float),dropout=embDropout,freeze=isFreezon ).to(device)
        self.textCNN = TextCNN(feaSize, contextSizeList, filterNum).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterNum, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding, self.textCNN, self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight)
    def calculate_y_logit(self, X):
        X = self.textEmbedding(X['seqArr'])
        X = X.transpose(1,2)
        X = self.textCNN(X)
        return self.fcLinear(X)

'''
class TextClassifier_StateBiLSTM(BaseClassifier):
    def __init__(self, classNum, feaSize=512, hiddenSize=512, hiddenList=[], 
                 fcDropout=0.5, 
                 useFocalLoss=False, weight=None, device=torch.device("cuda:0")):
        self.textBiLSTM = TextBiLSTM(feaSize, hiddenSize).to(device)
        self.simpleAttn = SimpleAttention(hiddenSize*2).to(device)
        self.fcLinear = MLP(hiddenSize*2, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textBiLSTM, self.simpleAttn, self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight)
    def calculate_y_logit(self, X):
        X,XLen = X['seqArr'],X['seqLenArr']
        X = self.textBiLSTM(X, XLen)
        X = self.simpleAttn(X)
        return self.fcLinear(X)
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            stateDict['lab2id'],stateDict['id2lab'] = dataClass.lab2id,dataClass.id2lab
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            if "trainIdList" in parameters:
                dataClass.trainIdList = parameters['trainIdList']
            if "validIdList" in parameters:
                dataClass.validIdList = parameters['validIdList']
            if "testIdList" in parameters:
                dataClass.testIdList = parameters['testIdList']
            dataClass.lab2id,dataClass.id2lab = parameters['lab2id'],parameters['id2lab']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc'])) 
'''

class TextClassifier_StateGRU(BaseClassifier):
    def __init__(self, classNum, embedding, feaSize=320, hiddenSize=448, hiddenList=[], 
                 embDropout=0.3, fcDropout=0.5, 
                 useFocalLoss=False, weight=None, device=torch.device("cuda:0")):
        self.xEmbedding = TextEmbedding( torch.tensor(embedding['x'], dtype=torch.float),dropout=embDropout,name='xEmbedding' ).to(device)
        self.yEmbedding = TextEmbedding( torch.tensor(embedding['y'], dtype=torch.float),dropout=embDropout,name='yEmbedding' ).to(device)
        self.vEmbedding = TextEmbedding( torch.tensor(embedding['v'], dtype=torch.float),dropout=embDropout,name='vEmbedding' ).to(device)
        self.dEmbedding = TextEmbedding( torch.tensor(embedding['d'], dtype=torch.float),dropout=embDropout,name='dEmbedding' ).to(device)
        self.tEmbedding = TextEmbedding( torch.tensor(embedding['t'], dtype=torch.float),dropout=embDropout,name='tEmbedding' ).to(device)
        self.textGRU = TextGRU(feaSize, hiddenSize).to(device)
        self.fcLinear = MLP(hiddenSize, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.xEmbedding,self.yEmbedding,self.vEmbedding,self.dEmbedding,self.tEmbedding, 
                                         self.textGRU, self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight)
    def calculate_y_logit(self, X):
        X,XLen = X['seqArr'],X['seqLenArr']
        x,y,v,d,t = self.xEmbedding(X[:,:,0]),self.yEmbedding(X[:,:,1]),self.vEmbedding(X[:,:,2]),self.dEmbedding(X[:,:,3]),self.tEmbedding(X[:,:,4])
        X = torch.cat([x,y,v,d,t], dim=2) # => batchSize × seqLen × feaSize
        X = self.textGRU(X, XLen)[:,-1,:] # => batchSize × hiddenSize
        return self.fcLinear(X)
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            stateDict['lab2id'],stateDict['id2lab'] = dataClass.lab2id,dataClass.id2lab
            stateDict['x2id'],stateDict['id2x'] = dataClass.x2id,dataClass.id2x
            stateDict['y2id'],stateDict['id2y'] = dataClass.y2id,dataClass.id2y
            stateDict['v2id'],stateDict['id2v'] = dataClass.v2id,dataClass.id2v
            stateDict['d2id'],stateDict['id2d'] = dataClass.d2id,dataClass.id2d
            stateDict['t2id'],stateDict['id2t'] = dataClass.t2id,dataClass.id2t
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            if "trainIdList" in parameters:
                dataClass.trainIdList = parameters['trainIdList']
            if "validIdList" in parameters:
                dataClass.validIdList = parameters['validIdList']
            if "testIdList" in parameters:
                dataClass.testIdList = parameters['testIdList']
            dataClass.lab2id,dataClass.id2lab = parameters['lab2id'],parameters['id2lab']
            dataClass.x2id,dataClass.id2x = parameters['x2id'],parameters['id2x']
            dataClass.y2id,dataClass.id2y = parameters['y2id'],parameters['id2y']
            dataClass.v2id,dataClass.id2v = parameters['v2id'],parameters['id2v']
            dataClass.d2id,dataClass.id2d = parameters['d2id'],parameters['id2d']
            dataClass.t2id,dataClass.id2t = parameters['t2id'],parameters['id2t']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))

class TextClassifier_StateBiGRU(BaseClassifier):
    def __init__(self, classNum, embedding, feaSize=320, hiddenSize=448, hiddenList=[], 
                 embDropout=0.3, fcDropout=0.5, 
                 useFocalLoss=False, weight=None, device=torch.device("cuda:0")):
        self.xEmbedding = TextEmbedding( torch.tensor(embedding['x'], dtype=torch.float),dropout=embDropout,name='xEmbedding' ).to(device)
        self.yEmbedding = TextEmbedding( torch.tensor(embedding['y'], dtype=torch.float),dropout=embDropout,name='yEmbedding' ).to(device)
        self.vEmbedding = TextEmbedding( torch.tensor(embedding['v'], dtype=torch.float),dropout=embDropout,name='vEmbedding' ).to(device)
        self.dEmbedding = TextEmbedding( torch.tensor(embedding['d'], dtype=torch.float),dropout=embDropout,name='dEmbedding' ).to(device)
        self.tEmbedding = TextEmbedding( torch.tensor(embedding['t'], dtype=torch.float),dropout=embDropout,name='tEmbedding' ).to(device)
        self.textBiLSTM = TextBiGRU(feaSize, hiddenSize).to(device)
        self.simpleAttn = SimpleAttention(hiddenSize*2).to(device)
        self.fcLinear = MLP(hiddenSize*2, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.xEmbedding,self.yEmbedding,self.vEmbedding,self.dEmbedding,self.tEmbedding, 
                                         self.textBiLSTM, self.simpleAttn, self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight)
    def calculate_y_logit(self, X):
        X,XLen = X['seqArr'],X['seqLenArr']
        x,y,v,d,t = self.xEmbedding(X[:,:,0]),self.yEmbedding(X[:,:,1]),self.vEmbedding(X[:,:,2]),self.dEmbedding(X[:,:,3]),self.tEmbedding(X[:,:,4])
        X = torch.cat([x,y,v,d,t], dim=2) # => batchSize × seqLen × feaSize
        X = self.textBiLSTM(X, XLen)
        X = self.simpleAttn(X)
        return self.fcLinear(X)
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            stateDict['lab2id'],stateDict['id2lab'] = dataClass.lab2id,dataClass.id2lab
            stateDict['x2id'],stateDict['id2x'] = dataClass.x2id,dataClass.id2x
            stateDict['y2id'],stateDict['id2y'] = dataClass.y2id,dataClass.id2y
            stateDict['v2id'],stateDict['id2v'] = dataClass.v2id,dataClass.id2v
            stateDict['d2id'],stateDict['id2d'] = dataClass.d2id,dataClass.id2d
            stateDict['t2id'],stateDict['id2t'] = dataClass.t2id,dataClass.id2t
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            if "trainIdList" in parameters:
                dataClass.trainIdList = parameters['trainIdList']
            if "validIdList" in parameters:
                dataClass.validIdList = parameters['validIdList']
            if "testIdList" in parameters:
                dataClass.testIdList = parameters['testIdList']
            dataClass.lab2id,dataClass.id2lab = parameters['lab2id'],parameters['id2lab']
            dataClass.x2id,dataClass.id2x = parameters['x2id'],parameters['id2x']
            dataClass.y2id,dataClass.id2y = parameters['y2id'],parameters['id2y']
            dataClass.v2id,dataClass.id2v = parameters['v2id'],parameters['id2v']
            dataClass.d2id,dataClass.id2d = parameters['d2id'],parameters['id2d']
            dataClass.t2id,dataClass.id2t = parameters['t2id'],parameters['id2t']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))

class TextClassifier_StateBiGRU2(BaseClassifier):
    def __init__(self, classNum, embedding, feaSize=320, hiddenSize=448, hiddenList=[], 
                 embDropout=0.3, fcDropout=0.5, 
                 useFocalLoss=False, weight=None, device=torch.device("cuda:0")):
        self.xEmbedding = TextEmbedding( torch.tensor(embedding['x'], dtype=torch.float),dropout=embDropout,name='xEmbedding' ).to(device)
        self.yEmbedding = TextEmbedding( torch.tensor(embedding['y'], dtype=torch.float),dropout=embDropout,name='yEmbedding' ).to(device)
        self.rEmbedding = TextEmbedding( torch.tensor(embedding['r'], dtype=torch.float),dropout=embDropout,name='rEmbedding' ).to(device)
        self.thetaEmbedding = TextEmbedding( torch.tensor(embedding['theta'], dtype=torch.float),dropout=embDropout,name='thetaEmbedding' ).to(device)
        self.vEmbedding = TextEmbedding( torch.tensor(embedding['v'], dtype=torch.float),dropout=embDropout,name='vEmbedding' ).to(device)
        self.dEmbedding = TextEmbedding( torch.tensor(embedding['d'], dtype=torch.float),dropout=embDropout,name='dEmbedding' ).to(device)
        self.tEmbedding = TextEmbedding( torch.tensor(embedding['t'], dtype=torch.float),dropout=embDropout,name='tEmbedding' ).to(device)
        self.textBiLSTM = TextBiGRU(feaSize, hiddenSize).to(device)
        self.simpleAttn = SimpleAttention(hiddenSize*2).to(device)
        self.fcLinear = MLP(hiddenSize*2, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.xEmbedding,self.yEmbedding,self.rEmbedding,self.thetaEmbedding,self.vEmbedding,self.dEmbedding,self.tEmbedding, 
                                         self.textBiLSTM, self.simpleAttn, self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight)
    def calculate_y_logit(self, X):
        X,XLen = X['seqArr'],X['seqLenArr']
        x,y,r,theta,v,d,t = self.xEmbedding(X[:,:,0]),self.yEmbedding(X[:,:,1]),self.rEmbedding(X[:,:,2]),self.thetaEmbedding(X[:,:,3]),self.vEmbedding(X[:,:,4]),self.dEmbedding(X[:,:,5]),self.tEmbedding(X[:,:,6])
        X = torch.cat([x,y,r,theta,v,d,t], axis=2) # => batchSize × seqLen × feaSize
        X = self.textBiLSTM(X, XLen)
        X = self.simpleAttn(X)
        return self.fcLinear(X)
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            stateDict['lab2id'],stateDict['id2lab'] = dataClass.lab2id,dataClass.id2lab
            stateDict['x2id'],stateDict['id2x'] = dataClass.x2id,dataClass.id2x
            stateDict['y2id'],stateDict['id2y'] = dataClass.y2id,dataClass.id2y
            stateDict['r2id'],stateDict['id2r'] = dataClass.r2id,dataClass.id2r
            stateDict['theta2id'],stateDict['id2theta'] = dataClass.theta2id,dataClass.id2theta
            stateDict['v2id'],stateDict['id2v'] = dataClass.v2id,dataClass.id2v
            stateDict['d2id'],stateDict['id2d'] = dataClass.d2id,dataClass.id2d
            stateDict['t2id'],stateDict['id2t'] = dataClass.t2id,dataClass.id2t
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            if "trainIdList" in parameters:
                dataClass.trainIdList = parameters['trainIdList']
            if "validIdList" in parameters:
                dataClass.validIdList = parameters['validIdList']
            if "testIdList" in parameters:
                dataClass.testIdList = parameters['testIdList']
            dataClass.lab2id,dataClass.id2lab = parameters['lab2id'],parameters['id2lab']
            dataClass.x2id,dataClass.id2x = parameters['x2id'],parameters['id2x']
            dataClass.y2id,dataClass.id2y = parameters['y2id'],parameters['id2y']
            dataClass.r2id,dataClass.id2r = parameters['r2id'],parameters['id2r']
            dataClass.theta2id,dataClass.id2theta = parameters['theta2id'],parameters['id2theta']
            dataClass.v2id,dataClass.id2v = parameters['v2id'],parameters['id2v']
            dataClass.d2id,dataClass.id2d = parameters['d2id'],parameters['id2d']
            dataClass.t2id,dataClass.id2t = parameters['t2id'],parameters['id2t']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))

class TextClassifier_BiLSTM(BaseClassifier):
    def __init__(self, classNum, embedding, feaSize=512, hiddenSize=512, hiddenList=[], 
                 embDropout=0.3, fcDropout=0.5, 
                 useFocalLoss=False, weight=None, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding( torch.tensor(embedding, dtype=torch.float),dropout=embDropout ).to(device)
        self.textBiLSTM = TextBiLSTM(feaSize, hiddenSize).to(device)
        self.simpleAttn = SimpleAttention(hiddenSize*2).to(device)
        self.fcLinear = MLP(hiddenSize*2, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding, self.textBiLSTM, self.simpleAttn, self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight)
    def calculate_y_logit(self, X):
        X,XLen = X['seqArr'],X['seqLenArr']
        X = self.textEmbedding(X)
        X = self.textBiLSTM(X, XLen)
        X = self.simpleAttn(X)
        return self.fcLinear(X)   

class TextClassifier_BiLSTM_CNN(BaseClassifier):
    def __init__(self, classNum, embedding, seqMaxLen, feaSize=512, hiddenSize=512, filterNum=448, contextSizeList=[1,3,5], hiddenList=[], 
                 embDropout=0.3, fcDropout=0.5, 
                 useFocalLoss=False, weight=None, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding( torch.tensor(embedding, dtype=torch.float),dropout=embDropout ).to(device)
        self.textBiLSTM = TextBiLSTM(feaSize, hiddenSize).to(device)
        self.textCNN = TextCNN(None, hiddenSize*2, contextSizeList, filterNum, seqMaxLen).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterNum, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding, self.textBiLSTM, self.textCNN, self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight)
    def calculate_y_logit(self, X):
        X,XLen = X['seqArr'],X['seqLenArr']
        X = self.textEmbedding(X)
        X = self.textBiLSTM(X, XLen)
        X = self.textCNN(X)
        return self.fcLinear(X)

class TextClassifier_HAN(BaseClassifier):
    def __init__(self, classNum, embedding, feaSize=512, hiddenSize=256, hiddenList=[], 
                 embDropout=0.3, attnDropout=0.3, fcDropout=0.5, 
                 useFocalLoss=False, weight=None, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding( torch.tensor(embedding, dtype=torch.float),dropout=embDropout ).to(device)
        self.charBiLSTM = TextBiLSTM(feaSize, hiddenSize, name='charBiLSTM').to(device)
        self.charAttn = SimpleAttention(hiddenSize*2, hiddenSize//2, name='charAttn').to(device)
        self.wordBiLSTM = TextBiLSTM(hiddenSize*2, hiddenSize, isDropout=False, name='wordBiLSTM').to(device)
        self.wordAttn = SimpleAttention(hiddenSize*2, hiddenSize//2, name='wordAttn').to(device)
        self.fcLinear = MLP(hiddenSize*2, classNum, hiddenList, fcDropout).to(device)
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight)
        self.dropout = nn.Dropout(p=attnDropout)
        self.moduleList = nn.ModuleList([self.charBiLSTM, self.charAttn, self.wordBiLSTM, self.wordAttn, self.fcLinear])
    def calculate_y_logit(self, X):
        X, XLen, wordLen = X['seqArr'],X['seqLenArr'],['unitLenArr']
        # X: batchSize × seqLen × wdLen; XLen: batchSize; wordLen: batchSize × seqLen
        XLen, indices = torch.sort(XLen, descending=True)
        _, desortedIndices = torch.sort(indices)
        packedX = nn.utils.rnn.pack_padded_sequence(X[indices], XLen, batch_first=True) # batchSize*seqLen × wdLen
        packedWordLen = nn.utils.rnn.pack_padded_sequence(wordLen[indices], XLen, batch_first=True)# batchSize*seqLen
        X = self.charBiLSTM(self.textEmbedding(packedX.data), packedWordLen.data) # => batchSize*seqLen × wdLen × hiddenSize*2
        X = self.dropout(self.charAttn(X)) # => batchSize*seqLen × hiddenSize*2
        packedX = nn.utils.rnn.PackedSequence(data=X, 
                                              batch_sizes=packedX.batch_sizes, 
                                              sorted_indices=packedX.sorted_indices, 
                                              unsorted_indices=packedX.unsorted_indices)
        packedX = self.wordBiLSTM(packedX, None) # => batchSize*seqLen × hiddenSize*2
        X, _ = nn.utils.rnn.pad_packed_sequencep(packedX, batch_first=True) # => batchSize × seqLen × hiddenSize*2
        X = X[desortedIndices]
        X = self.wordAttn(X) # => batchSize × hiddenSize*2
        return self.fcLinear(X) # => batchSize × classNum