# Eler: Ensemble Learning-based Automated Verification of Code Clones
Eler is a method for automated clone verification of high-precision based on ensemble learning.

Eler combine several clone detection algorithms based on different representations, including token-based, tree-based, and graph-based detection algorithms to accurately verify more difficult-to-detect clone types. 
And combine predicted values from machine learning models by ensemble learning to avoid the high overhead and subjectivity problems associated with manual verification.

Eler is mainly comprised of two main phases: feature extraction and ensemble learning.

1. Extraction of features: The purpose of this phase consists of calculating the similarity scores of the clone pairs under verification using different clone detection tools to form a similarity vector. The code pair enters the phase and the similarity vector leave it.
2. Ensemble learning: The purpose of this phase is to input the similarity vector to multiple machine learning models for categorisation, and combine the categorisation results given by each machine learning model to give a final result as to help determine whether or not the detected code pair is indeed a clone. The input of this phase is the similarity vector and the outcome will be the final result of the clone verification. 

The source code and dataset of Eler are published here.


# Project Structure  
  
```shell  
Eler  
|-- main.py     	// implement the start method of the project
|-- Extraction_of_features.py     // implement the Extraction of features phase  
|-- Ensemble_learning.py   // implement the Ensemble learning phase  
|-- GetSimilarity   // three token-based, three tree-based, and three graph-based code clone detection algorithms we reproduced 
```

### Extraction_of_features.py
- Input: code pairs under verification
- Output: feature vectors of code pairs 
```
python Extraction_of_features.py
```

### Ensemble_learning.py
- Input: feature vectors of code pairs
- Output: final result of the clone verification
```
python Ensemble_learning.py
```
