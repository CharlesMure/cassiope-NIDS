# cassiope-NIDS
Creating a NIDS for SDN based on a Deep Neural Network

## TODO

* Evaluer les modèles + algortihme d'apprentissage en utilisant le [NSL-KDD Dataset][NSL-KDD]
  * Utilisation de "Feature Extraction"
* Recherhcer les features accessibles sur OpenFlow
* Contacter a.habibi.l@unb.ca pour récupérer des Datasets réels

## Set-Up

We Use Python3, TensorFlow and Keras to generate and test our models

```
 pip3 install jupyter
 pip3 install tensorflow-gpu
 pip3 install keras
```

 
 
## Sytem Details

#### Categories of attacks supported ####
* DoS
* R2L
* U2R
* Probe

#### Models ####
* [CapsNet][CapsNet]


### Utils
* [Real Dataset][Datasets]



[NSL-KDD]: https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
[CapsNet]: https://github.com/XifengGuo/CapsNet-Keras
[Datasets]: http://www.unb.ca/cic/datasets/index.html
