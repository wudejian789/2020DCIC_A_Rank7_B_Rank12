from lib.utils import *
from lib.DL_ClassifierModel import *
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
import pickle,random,os

seed = 9527#562264457

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True

device = 'cuda'

dataClass = DataClass_state('./data/nnTrain2.pkl', None, './data/nnLabel2.pkl', xydiff=500)
dataClass.vectorize(loadCache=False, iters=10, workers=cpu_count())

model = TextClassifier_StateCNN(3, dataClass.vector['embedding'], feaSize=224, filterNum=256, contextSizeList=[1,3,5],
                                weight=[1/0.505,1/0.340,1/0.155], useFocalLoss=True, device=device, embDropout=0.3, fcDropout=0.5)

model.cv_train(dataClass, trainSize=64, batchSize=64, epoch=50, stopRounds=0, earlyStop=10,
              isHigherBetter=True, metrics="MaF", report=["ACC", "MaF"], savePath='model/nn/StateCNN', 
              kFold=5, seed=seed)