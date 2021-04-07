# High Speed Object Tracking in Racket sports
**Object tracking is implemented using Semantic Image Segmentation technique using SegNet network.**

### The work is inspired from: ###
1. Y. Huang, I. Liao, C. Chen, T. İk and W. Peng, "TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications*," 2019 16th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS), Taipei, Taiwan, 2019, pp. 1-8, doi: 10.1109/AVSS.2019.8909871.
2. V. Badrinarayanan, A. Kendall and R. Cipolla, "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 12, pp. 2481-2495, 1 Dec. 2017, doi: 10.1109/TPAMI.2016.2644615.

### Requirements to run: ###
>Python - 3.8.5 <br />
>numpy - 1.19.2 <br />
>pandas - 1.15.0 <br />
>pillow - 8.1.0 <br />
>pydot - 1.4.2 <br />
>scipy - 1.6.1 <br />
>tensorboard - 2.4.1 <br />
>Tendorflow - 2.4.0 <br />
>CUDA - 11.0 <br />
>cuDNN - 8.0.4 <br />
>Keras - 2.4.3 <br />
>OpenCV - 4.0.1 <br />
>Flask <br />
>flask_cors <br />

**sample_video file has been provided in the given link and should be added to flask_apps/sample_video:** <br />
https://www.dropbox.com/sh/z4jxyktqq8ww1ye/AAATH1xdbNcV_OzaAXqaFF8Ba?dl=0 <br />
Ver7 contains the trained weights for the tennis ball tracking video and the Tennis file conatins the input and output video of the model.
### User interface of the application created:
  **Input video will be fetched from the youtube link and processed to produce the output**
  
  ![image](https://user-images.githubusercontent.com/49316145/113846736-69cecc80-97b4-11eb-8f0c-fd959912ccc0.png)
  
### Frame wise example output:
  
  ![image](https://user-images.githubusercontent.com/49316145/113848247-e6ae7600-97b5-11eb-9113-f6013d762ff3.png)
  
### Additional:
>**Openpose paper:** Z. Cao, G. Hidalgo, T. Simon, S. -E. Wei and Y. Sheikh, "OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields," in IEEE >Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 1, pp. 172-186, 1 Jan. 2021, doi: 10.1109/TPAMI.2019.2929257. <br />
>**Openpose implementation:** https://github.com/CMU-Perceptual-Computing-Lab/openpose <br />
>This was implemented additionally in the object tracking video clips for player pose analysis.

### Input/Output data playlist:
https://www.youtube.com/watch?v=wdDROcUAsiw&list=PLan-29mX9agax7fbI3juHbcQqX1_N_1bm


