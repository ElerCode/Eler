# Eler

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

