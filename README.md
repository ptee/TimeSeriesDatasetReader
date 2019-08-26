# Time-Series Dataset Readers

* multivarReader.py

  Read multivariate time series (MTS) datasets collected by [Mustafa Baydogan](https://asu.pure.elsevier.com/en/publications/time-series-representation-and-similarity-based-on-local-autopatt-2). His archive becomes widely used in the community for benchmarking multivariate time series classification. The archive consists of 15 datasets in "mat" format, therefore, multivarReader.py will read the data and return the data as the python list. This is because the sequence length of each instance in the same dataset can be in arbitrary lengths. 
  
  
  Mustafa Baydogan's MTS datasets can be obtained from:
  http://www.mustafabaydogan.com/files/viewdownload/20-data-sets/69-multivariate-time-series-classification-data-sets-in-matlab-format.html or contact [Mustafa Baydogan](http://www.mustafabaydogan.com).

  The datasets in this archive were used in: 
  * [A Very Concise Feature Representation For Time Series Classification Understanding](https://ieeexplore.ieee.org/document/8757981), In the 16th International Conference on Machine Vision Applications (MVA), 2019, and 
  
  * [Dimensionality Reduction for Visualization of Time Series and Trajectories](https://link.springer.com/chapter/10.1007/978-3-030-20205-7_21), In the 21st Scandinavian Conference on Image Analysis (SCIA), pages: 246-257, 2019.
   

    The characteristics of 15 datasets and their error rates in ten classifiers from [A Very Concise Feature Representation For Time Series Classification Understanding](https://ieeexplore.ieee.org/document/8757981) can be found in the table below. From the left to the right of the table, they are i) the number of the attributes, ii) the lengths of sequences in the dataset, iii) the number of output classes, iv) the number of training data, and v) the number of testing data. 
    
    The datasets according to  [A Very Concise Feature Representation For Time Series Classification Understanding](https://ieeexplore.ieee.org/document/8757981) are grouped into four categories according to the levels of difficulty based on the classification results from the [Dynamic Time Wrapping (DTW)](https://asu.pure.elsevier.com/en/publications/time-series-representation-and-similarity-based-on-local-autopatt-2). The categories of the datasets are represented by small symbols appearing in front of the dataset names. 
    
    
    ![dataset+results](doc/dataset+results.png)
 
