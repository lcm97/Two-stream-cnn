# two-stream-cnn

This project utilizes the aggregated cpmputational power of already connected IoT devices to perform DNN-based recognition in real time.It also contains a single device implementation which could run on the TX2 or other embedded-devices as a comparison.

## Dependencies:

  Make sure that you have Python 2.7 running on your device.
### Single device:  
  On TX2, please use NVIDIA JetPack 3.3 to install CUDA and cuDNN,then utilize pip to install Keras with  a Tensorflow-gpu backend,this installation procedure is similar on Raspberry PI as well.
### Multiple devices:
  For Raspberry PIs, apart from tensorflow and keras,please install Apache Avro through pip for managing remote procedure calls (RPC). Moreover, make sure ports number 12345 and 9999 are open on all Raspberry PIs,and edit the resource/ip files to get all of the IP addresses of Raspberry PIs in your network.
  
## Quick Start:
### Multiple devices:
  On all of your device except the initial sender, run the node.
  
  python node.py
  Start the data sender. You should be able to see console log.
  
  python initial.py
  
### Single device:
 
  Execute predict file to run model inference.
  
  python predict.py
 

## References:

[1]: R. Hadidi, J. Cao, M. Woodward, M. Ryoo, and H. Kim, "Musical Chair: Efficient Real-Time Recognition Using Collaborative IoT Devices," ArXiv e-prints:1802.02138.

[2]: M.S.Ryoo, K.Kim, and H.J.Yang ,"Extreme Low Resolution Activity Recognition with Multi-Siamese
Embedding Learning," in Conference on Artificaial Intelligence(AAAI),Feb.2018

[3]: J.Choi, W.J.Jeon, and S.-C.Lee,"Spatial-Temporal Pyramid Matching for Sports Videos,"in International Conference on Multimedia Information Retrieval(ICMR),pp.291-297,2008
