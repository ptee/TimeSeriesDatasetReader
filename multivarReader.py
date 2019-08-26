# -*- coding: utf-8 -*-
""" Read multivariate time series datasets used by Mustafa Baydogan. The data is in mat format.
    
    http://www.mustafabaydogan.com/files/viewdownload/20-data-sets/69-multivariate-time-series-classification-data-sets-in-matlab-format.html
    
    Author: Pattreeya Tanisaro
"""

import scipy.io as sio
import sys
from os import path, listdir
import numpy as np
import numpy.random as random
import getopt



class MultivarReader(object):
    """ To read Multivariate Time Series Classification Data Sets (in MATLAB format) from Mustafa Baydogan datasets.
    
    Attributes
    ----------
    dataset : str
        name of the dataset
    train : list 
        training data. The sequence length for each instance/sample in the dataset can be unequal.
    train_labels : list
        training data labels. 
    test : list
        testing data. The sequence length for each instance/sample in the dataset can be unequal.
    test_labels : list
         test data labels
    num_features : int
        number of features
    
    Methods
    -------
    numFeatures()
        Get number of features.
    shuffle(train=True)
        Shuffle data. 
    datasetName
        Get dataset name
    trainSet
        Get training data
    testSet
        Get test data
    mergeTrainTest
        Merge the training and test data
    count
        Count number of occurrences of label in Y
    shiftOutput
        Shift the output label from minimum to 0
    minMaxMeanSeqLength(train=True)
        Look into the training/testing data to get min/max of the sequence length
    minMaxMeanValue(train=True)
        Get min, max, mean value of the dataset
    listToString
        Print list of values as string concatenating using 3 digits
    """
    
    def __init__(self, infile ):
        """ create list (samples) of train/test set.
        
            The list length is the number of samples.
            Each list has (nSeq, nFeat) by transpose the original matrix
        
        Parameters
        ----------
        infile : str
            Input (mat) file.
                
        """
        def readSamples( data ):
            
            newData = []
            data = data.squeeze()
            for dat in data:
                newData.append( dat.T )
            
            return newData
                
        print( "Reading: ", infile )
        matfile = sio.loadmat( infile )
        self.dataset = path.splitext( path.basename( infile ))[0]
        mts = matfile['mts']
        self.train = readSamples( mts['train'][0][0] ) # return list as the samples which contain nFeat x nSeq (unequal)
        self.train_labels = (mts['trainlabels'][0][0]).squeeze().tolist()
        print( "Training samples: ", len(self.train), ", or length : ",len(self.train_labels) )
        print( "Training features (fixed): ", self.train[0].shape[1], ", with #classes: ", np.unique(self.train_labels) )
        
        self.test = readSamples( mts['test'][0][0] )
        self.test_labels =  (mts['testlabels'][0][0]).squeeze().tolist()
        print( "Test samples: ", len(self.test), ", or  length : ",len(self.test_labels) )
        print( "Test features (fixed): ", self.test[0].shape[1], ", with #classes: ", np.unique(self.test_labels) )
    
        assert( self.test[0].shape[1] == self.train[0].shape[1] )
        
        self.num_features = self.test[0].shape[1]
    
    
    def numFeatures(self):
        """ Get number of features.
        
        Returns
        -------
        int
            number of features.
        """
        return self.num_features
    
           
    def shuffle(self, train=True):
        """ Shuffle data. The function returns nothing. User can get the new shuffled data by calling trainSet or testSet.
        
        Parameters
        ----------
        train : boolean
            If train is set to True, shuffle the stored training data.
            If train is set to False, shuffle the stored test data.
            
        Returns:
        None
        """
        XY = []
        if train:
            for (X,Y) in zip(self.train, self.train_labels):
                XY.append( (X, Y) )   
        else:
            
            for (X,Y) in zip(self.test, self.test_labels):
                XY.append( (X, Y) )
        
        random.shuffle(XY)
        X = []; Y = []
             
        for xy in XY:
            X.append( xy[0] )
            Y.append( xy[1] )
        
        if train:
            self.train = X
            self.train_labels = Y
        
        else:
            self.test = X
            self.test_labels = Y
        
        
    def datasetName(self):
        """ Get dataset name.
        """
        return self.dataset
    
    
    def trainSet(self):
        """Get training data and traing labels. 
        
        Returns
        -------
        list
            The training data as a list containing number of samples contain [nSeq x nFeat].
        list
            The training labels as a list. 
        """
        return self.train, self.train_labels 
    
    
    def testSet(self):
        """ Get test data and test labels. 
        
        Returns
        -------
        list
            The test data as a list containing number of samples contain [nSeq x nFeat].
        list
            The test labels as a list.
        """
        return self.test, self.test_labels
    
    
    def mergeTrainTest(self):
        """ Merge train and test data
        
        Returns
        -------
        list
            The merged data of training and test set.
        list
            The merged labels.
        """
        data = self.train
        data.extend( self.test )
        labels = self.train_labels
        labels.extend( self.test_labels )
        
        return data, labels
    
    @staticmethod
    def count( Y ):
        """ Count number of occurrences of label in Y
        
        Parameters
        ----------
        Y : list
            Labels.
        
        Returns
        -------
        map
            A map of label and number of occurences in the given Y.
        """
        uniq_Y = set(np.unique( Y ))
        labels = {label: Y.count(label) for label in uniq_Y}
        
        return labels
         
           
    def shiftOutput(self):
        """ Shift the output label from the minimum value to 0 and set the labels from 0 onwards.
        
        Returns
        -------
        list
            Train labels
        list
            Test labels
        """
        uniq_testY = set(np.unique( self.test_labels ))
        uniq_trainY = set(np.unique( self.train_labels ))
        if uniq_testY != uniq_trainY:
            print( "testY != trainY !!!" )
            sys.exit()
        
        old_labels = list(set(uniq_trainY))
        new_labels = range(len(uniq_trainY))
        
        new_train_labels = []
        for y in  self.train_labels:
            idx = old_labels.index(y)
            new_train_labels.append(new_labels[idx])
        self.train_labels = new_train_labels
        
        new_test_labels = []
        for y in self.test_labels:
            idx = old_labels.index(y)
            new_test_labels.append(new_labels[idx])
        self.test_labels = new_test_labels
        
        return self.train_labels, self.test_labels
    
    
    def minMaxMeanSeqLength(self, train=True):
        """ Look into the training/testing data to get min/max of the sequence length.
        
        Parameters
        ----------
        train : boolean
            Set to True for the training set, False for the test set.
        
        Returns
        --------
        int
            minimum length
        int 
            maximum length
        int
            average length
        """
        if train:
            dataSet = self.train
        else:
            dataSet = self.test
            
        minL=np.Inf;maxL=0; totalL = 0
        nData = len(dataSet)
        for data in dataSet:
            # data.shape = (nSeq, nFeat)
            (nSeq, nFeat) = data.shape
            if nSeq > maxL:
                maxL = nSeq
            if nSeq < minL:
                minL = nSeq
            totalL += nSeq
                
        
        return minL, maxL, int(totalL/nData)
    
    
    def minMaxMeanValue(self, train=True):
        """ Get min, max, mean value of the dataset. This can infer the padding value in the dataset.
        
        Parameters
        -----------
        train : boolean
            Set to True for the training set, False for the test set.
        
        Returns
        -------
        list
            Minimum values according to the number of features.
        list 
            Maximum values according to the number of features.
        list
            Mean values according to the number of features.
         
        """
        if train:
            dataSet = self.train
        else:
            dataSet = self.test
            
        
        min_values = []; max_values = []; sum_values = []; sum_seq = 0;
        for i, data in enumerate(dataSet):
            # data.shape = (nSeq, nFeat)
            (nSeq, nFeat) = data.shape
        
            if i==0:
                min_values = [np.inf for j in range(nFeat)]
                max_values = [0 for j in range(nFeat)]
                sum_values = [0 for j in range(nFeat)]
                
            
            for k in range(nFeat):
                min_values[k] = min( min(data[:,k]), min_values[k] )
                max_values[k] = max( max(data[:,k]), max_values[k] )
                sum_values[k] = np.sum( data[:,k] )
                
            sum_seq += nSeq
        
        mean_values =  list(np.array(sum_values)/sum_seq)
        
        return min_values, max_values, mean_values
    
    
    def listToString(self, values):
        """ Print list of values as string concatenating using 3 digits
        
        Parameters
        ----------
        values : list
            A list
        """
        msg = ''
        for v in values:
            if v != values[-1]:
                msg += '{:8.3f}, '.format(v)
            else:
                msg += '{:8.3f}'.format(v)
            
        return msg
    

#################################### END OF CLASS #################################      

def test( dataset ):            
    
    print("Dataset: ",dataset.datasetName()) 
    trainX, trainY = dataset.trainSet()
    testX, testY = dataset.testSet()
    print("Train Y:")
    print(trainY)
    print("Test Y:")
    print(testY)
    
    print("***ShiftOutput****")
    trainY, testY = dataset.shiftOutput()
    print("After shifting Train Y:")
    print(trainY)
    print("After shifting Test Y:")
    print(testY)
    
    print("****Count labels of training Y****")
    print( dataset.count(trainY) )
    
    print("****Count labels of test Y****")
    print( dataset.count(testY) )
    
    msg = ''
    min_values, max_values, mean_values = dataset.minMaxMeanValue(train=True)    
    msg += "TrainData min_values: " + dataset.listToString(min_values) +'\n'
    msg += "TrainData max_values: " + dataset.listToString(max_values) +'\n'
    msg += "TrainData mean_values: " + dataset.listToString( mean_values ) +'\n'
    print( msg )
    
    msg = ''
    min_values, max_values, mean_values = dataset.minMaxMeanValue(train=False)    
    msg += "TestData min_values: " + dataset.listToString(min_values) +'\n'
    msg += "TestData max_values: " + dataset.listToString(max_values) +'\n'
    msg += "TestData mean_values: " + dataset.listToString( mean_values ) +'\n'
    print( msg )
    
    msg = ''
    min_len, max_len, mean_len = dataset.minMaxMeanSeqLength(train=True)    
    msg += "TrainData min length: " + str(min_len) +'\n'
    msg += "TrainData max length: " + str(max_len) +'\n'
    msg += "TrainData mean length: " + str( mean_len ) +'\n'
    print( msg )
    
    msg = ''
    min_len, max_len, mean_len = dataset.minMaxMeanSeqLength(train=False)    
    msg += "TestData min length: " + str(min_len) +'\n'
    msg += "TestData max length: " + str(max_len) +'\n'
    msg += "TestData mean length: " + str( mean_len ) +'\n'
    print( msg )
    
        
def test_loop( inPath ):
    
    for ifile in listdir( inPath ):    
        if path.splitext(ifile)[1] != '.mat':
            continue
        
        print("*****Reading: %s ********"%ifile)
        test( MultivarReader(path.join(inPath, ifile)) )
       

if __name__ == "__main__":

    def usage():
        print( 'Usage: %s -i <input_file> OR -p <input_path>'%(sys.argv[0]) )
        print( '-i input file from Baydogan dataset (.mat)' )
        print( '-p input path which contains various datasets (.mat) in a directory.' )
        sys.exit(2)
        
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:p:",["help","inFile","inPath"])
    except:
        usage()
    inFile = ""; inPath="";
    
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ("-i", "--inFile"):
            inFile = arg
        elif opt in ("-p", "--inPath"):
            inPath = arg
    
    
    if inFile: 
        test( MultivarReader(  inFile ) )
      
    elif inPath:
        test_loop(inPath)
    else:
        usage()


########################################## END OF FILE ###############################################
