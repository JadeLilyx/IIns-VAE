# Deep UWB Dataset
          
(https://doi.org/10.5281/zenodo.4290069)


## Sample Structure

  || CIR (157 float values)    ||
  
  || Error (in meters)         ||
  
  || Room (int)                ||
  
  || Obstacle (10 bool values) ||


## Room Encoding

  0 -> cross-room measurements
  
  1 -> big room
  
  2 -> medium room
  
  3 -> small room
  
  4 -> outdoor

## Obstacle encoding (1-hot encoding)

  1000000000 -> wall
  
  0100000000 -> polystyrene plate
  
  0010000000 -> plastic (trash bin and chair)
  
  0001000000 -> plywood plate
  
  0000100000 -> cardboard box
  
  0000010000 -> LCD TV
  
  0000001000 -> metal plate
  
  0000000100 -> wood door
  
  0000000010 -> glass plate
  
  0000000001 -> metal window


## Reading Code
 
```python
# Import libraries
import pandas as pd
import numpy as np

# Extract dataset
dataset = pd.read_pickle('dataset.pkl')

# Select specific obstacle configurations
ds = np.asarray(dataset.loc[dataset['Obstacles']=='011111111'][['CIR','Error']])

# Select specific rooms
ds = np.asarray(dataset.loc[dataset['Room']==1][['CIR','Error']])
  
# Select all samples
ds = np.asarray(dataset[['CIR','Error']])

# Get X,y for training  
X = np.vstack(ds[:,0])
Y = np.array(ds[:,1])
```
  
