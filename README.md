# cassiope-NIDS
Creating a NIDS based on a Deep Neural Network (CNN) - a technical approach

The project was conducted during my master studies at Télécom SudParis.

![Overview of the project](https://raw.githubusercontent.com/CharlesMure/cassiope-NIDS/master/Poster%20-%20Equipe%2018.jpg)

## Set-Up

We Use Python3, TensorFlow and Keras to generate and test our models. The collection part is handle by 

```
 pip3 install numpy tensorflow keras imblearn
```

In order to collect the data from the network, we use the auditing tool [Argus][Argus]. You need to install Argus and Argus-client.
Follow the instructions on their website.
 
 
## How to use

### Training & Test ###

To train the model, you just need to launch the script model_train

```
 python3 model_train.py
```

This step will export the model of the network as a .json and the associated weight as a .h5 file.

### Prediction ###

First step, you need to monitor and extract the features of your network using Argus.

* Listen on the interface (.ie enp0s3) and stream the result on the port 4444

```
argus -i enp0s3 -P 4444
```

* Extract the features from the stream and populate a small .log file.

```
 ./ra -S localhost:4444 -n -L -1 -c , -u -s saddr daddr sport dport sttl dttl state dur dpkts sbytes dbytes rate sttl dttl sload dload smeansz dmeansz - tcp or udp >> collect.log

  # If you use sFlow
  ./ra -S sflow://localhost:6343 -n -L -1 -c , -u -s saddr daddr sport dport sttl dttl state dur dpkts sbytes dbytes rate sttl dttl sload dload smeansz dmeansz - tcp or udp
```

 * Launch monitoring.py to pre-process the features of the extracted flow

```
 python3 monitoring.py &
```

* Classify in real-time the flow using DeepLInspect.py

```
 python3 DeepLInspect.py
```


#### Categories of attacks classified ####
* Fuzzers
* Backdoor 
* DoS
* Exploits
* Generic
* Reconnaissance
* Shellcode
* Worms


#### Model Used ####
* [CNN][Reference_model] 


### Dataset
* [UNSW_NB15][Dataset]



[Reference_model]: https://www.researchgate.net/publication/319717354_A_Few-shot_Deep_Learning_Approach_for_Improved_Intrusion_Detection
[Argus]: https://qosient.com/argus/
[Dataset]: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/
