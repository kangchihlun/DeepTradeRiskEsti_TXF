import sys,os
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from windpuller import WindPuller
from dataset import DataSet
from datasetLoader import feature as ft




def evaluate_model(model_path, windSize=30):
    
    train_set, test_set = read_feature(".", input_shape, code)
    
    scores = saved_wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Test loss:', scores[0])
    print('test accuracy:', scores[1])
    pred = saved_wp.predict(test_set.images, 1024)
    cr = calculate_cumulative_return(test_set.labels, pred)
    print("changeRate\tpositionAdvice\tprincipal\tcumulativeReturn")
    for i in range(len(test_set.labels)):
        print(str(test_set.labels[i]) + "\t" + str(pred[i]) + "\t" + str(cr[i] + 1.) + "\t" + str(cr[i]))

def make_model_type(input_shape, _modelTyp = 0, nb_epochs=100, batch_size=128, lr=0.01, n_layers=1, n_hidden=16, rate_dropout=0.3):
    model_path = 'model.%s' % input_shape[0]
    train_set,validation_set,test_set,numFeatures = ft.generateDataSetTXF(os.getcwd(),input_wind_size = input_shape[0] ,toDataSet=True)

    input_shape[1]=numFeatures
    wp = WindPuller(input_shape=input_shape,modelType=_modelTyp, lr=lr , n_layers=n_layers, n_hidden=n_hidden, rate_dropout=rate_dropout)

    wp.fit(train_set.images, train_set.labels, batch_size=batch_size,
           nb_epoch=nb_epochs, shuffle=True, verbose=1,
           validation_data=(validation_set.images, validation_set.labels))
    
    scores = wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    wp.model.save(model_path)


def load_model_type(model_path,input_shape, _modelTyp = 0, nb_epochs=100, batch_size=128, lr=0.01, n_layers=1, n_hidden=16, rate_dropout=0.3):
    train_set,validation_set,test_set,numFeatures = ft.generateDataSetTXF(os.getcwd(),input_wind_size = input_shape[0] ,toDataSet=True)
    input_shape[1]=numFeatures
    wp = WindPuller(input_shape=input_shape, modelType=1, lr=lr, n_layers=n_layers, n_hidden=n_hidden, rate_dropout=rate_dropout)
    saved_wp = wp.load_model(model_path)
    scores = saved_wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Test loss:', scores[0])
    print('test accuracy:', scores[1])
    pred = saved_wp.predict( test_set.images , 1024)
    pred = np.reshape(pred, [-1])
    result = np.array([pred, test_set.labels]).transpose()
    with open('output.' + str(input_shape[0]), 'w') as fp:
        for i in range(result.shape[0]):
            for val in result[i]:
                fp.write(str(val) + "\t")
            fp.write('\n')    
    

    
#
def make_model_type3(input_shape, nb_epochs=100, batch_size=128, lr=0.01, n_layers=1, n_hidden=16, rate_dropout=0.3):
    model_path = 'model.%s' % input_shape[0]
    windowSize = input_shape[0] # num minutes
    X_train, y_train, X_val, Y_val, X_test, y_test, numFeatures = ft.generateDataSetTXF(os.getcwd(),input_wind_size = input_shape[0],toDataSet=False)
    input_shape[0] = numFeatures 
    
    wp = WindPuller(input_shape=input_shape, modelType=3, lr=lr, n_layers=n_layers, n_hidden=n_hidden, rate_dropout=rate_dropout)
    
    wp.fit(X_train, y_train, batch_size=batch_size,
           nb_epoch=nb_epochs, shuffle=True, verbose=1,validation_data=(X_val, Y_val))
    
    scores = wp.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    wp.model.save(model_path)


def load_model_type3(input_shape):
    model_path = 'model.%s' % input_shape[0]
    wp = WindPuller(input_shape=input_shape, modelType=2, lr=lr, n_layers=n_layers, n_hidden=n_hidden, rate_dropout=rate_dropout)
    saved_wp = wp.load_model(model_path)
    scores = saved_wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Test loss:', scores[0])
    print('test accuracy:', scores[1])
    pred = saved_wp.predict( X_test , 1024)
    pred = numpy.reshape(pred, [-1])
    result = numpy.array([pred, y_test]).transpose()
    with open('output.' + str(input_shape[0]), 'w') as fp:
        for i in range(result.shape[0]):
            for val in result[i]:
                fp.write(str(val) + "\t")
            fp.write('\n')
 


  
# 
def listdir_joined(path):
    return [os.path.join(path, entry) for entry in os.listdir(path)]

def getfileRecursive(path,outArr):
    if(os.path.isfile(path)):
        _type_ = (path.split("."))[-1]
        #if(fileType == _type_):
        outArr.append(path)
    elif(os.path.isdir(path)):
        _folders_ = [x for x in listdir_joined(path)]
        for f in _folders_ :
            getfileRecursive(f,outArr)

def checkModelWinningRate():
    searchDir = os.getcwd() + '\\result'
    allfile=[]
    getfileRecursive(searchDir,allfile)
    modelPrecisionDict = {}
    
    
    for j in range(len(allfile)): # fname = allfile[1]
        fname = allfile[j]
        if('output' in fname):
            content = []
            with open(fname) as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            numHit = 0
            for stc in content: # stc = content[23]
                tr = stc.split('\t')
                pred = max( float(tr[0]) , 0.0 )
                real = max( float(tr[1]) , 0.0 )
                if( numpy.sign(pred) == numpy.sign(real) ):
                    numHit += 1
            
            res = float(numHit) / len(content)
            modelPrecisionDict[ os.path.basename(allfile[j-1]) ] = res
    return modelPrecisionDict 



def checkModelPorfit():
    searchDir = os.getcwd() + '\\result'
    allfile=[]
    getfileRecursive(searchDir,allfile)
    modelProfitDict = {}
    
    for j in range(len(allfile)): 
        fname = allfile[j]
        if('output' in fname):
            content = []
            with open(fname) as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            
            cumulative_profit = 0
            for k in range(1,len(content)): 
                cur = content[k].split('\t')
                prv = content[k-1].split('\t')
                
                pred_curr = float(cur[0])
                pred_prv = float(prv[0])
                
                true_curr = float(cur[1])
                true_prv = float(prv[1])
                
                if( (np.abs(pred_curr) < 0.0001 ) and ( pred_prv != 0 ) ):
                    cumulative_profit += pred_prv * true_prv
            
            modelProfitDict[ os.path.basename(allfile[j-1]) ] = cumulative_profit 
    return modelProfitDict



if __name__ == '__main__':
    operation = "train"
    if len(sys.argv) > 1:
        operation = sys.argv[1]
    
    # Perameters Setup
    ModelType =  1
    if len(sys.argv) > 2:
        ModelType = int(sys.argv[2])
    
    numEpoch = 400
    batchSize = 1024
    learningRate = 0.007
    
    
    if operation == "train":
        if(ModelType == 0): # positive only
            make_model_type([30, 0],_modelTyp = 0, nb_epochs=numEpoch , batch_size=batchSize, lr=learningRate, n_layers=1, n_hidden=16, rate_dropout=0.2)
        elif(ModelType == 1): # negative only
            make_model_type([30, 0],_modelTyp = 1, nb_epochs=numEpoch , batch_size=batchSize, lr=learningRate, n_layers=1, n_hidden=16, rate_dropout=0.2)
        elif(ModelType == 3):
            _input_dim = 30
            _output_dim = 50 
            _hidden1_dim = 40
            _dense_dim = 1
            _batchSize = 1024
            input_shape = [ _input_dim , _output_dim , _hidden1_dim , _dense_dim ]
            make_model_type2( input_shape , numEpoch , batchSize, lr = learningRate,rate_dropout=0.2)
    
    elif operation == "predict":
        modDict = { 
                    0:'model0_positive',
                    1:'model1_negative',
                    2:'model3.30_epoch200_Lr0.007',
                    3:'model3.30_epoch200_Lr0.007'
                  }
        
        if(ModelType == 0): # positive only
            load_model_type(modDict[ModelType], [30, 0],_modelTyp = 0, nb_epochs=numEpoch , batch_size=batchSize, lr=learningRate, n_layers=1, n_hidden=16, rate_dropout=0.2)
        elif(ModelType == 1): # negative only
            load_model_type(modDict[ModelType], [30, 0],_modelTyp = 1, nb_epochs=numEpoch , batch_size=batchSize, lr=learningRate, n_layers=1, n_hidden=16, rate_dropout=0.2)
        elif(ModelType == 3):
            _input_dim = 30
            _output_dim = 50 
            _hidden1_dim = 40
            _dense_dim = 1
            _batchSize = 1024
            input_shape = [ _input_dim , _output_dim , _hidden1_dim , _dense_dim ]
            load_model_type2( input_shape , numEpoch , batchSize, lr = learningRate,rate_dropout=0.2)
    else:
        print("Usage: gossip.py [train | predict]")
    
