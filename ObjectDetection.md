# Models of Object Detection

### Introduction
**Object Detection** is one of the most popular research fields on Machine Learning. To be specific, this is one branch of computer vision for detecting semantic objects of a certain class in images or videos. There are a lot of sub-branches of object detection such as 3D Object Detection, Real-Time Object Detection, Salient Object Detection etc. In this report, I want to introduce and analyze some state-of-the-art models in Object Detection. 
[Oydeep Bhattacharjee](https://medium.com/technology-nineleaps/some-key-machine-learning-definitions-b524eb6cb48)[1] defined a machine learning model as a mathematical representation of a real-world process. To generate a machine learning model, you will need to provide training data to a machine learning algorithm to learn from. 
But in my report, I understand models as algorithms, and will discuss about some classical and state-of-the-art models with some relevant pre-trained model zoos.

### Review of Object Detection Algorithms
##### [R-CNN](http://islab.ulsan.ac.kr/files/announcement/513/rcnn_pami.pdf)
Region-based Convolutional Networks(R-CNN) is a good classical model to detect objects developed by R.Girshick and al.(2014)[2]. They propose a simple and scalable approach to combine region proposals with high-capacity convolutional networks(CNNs) by two ideas:
> * one can apply CNNs to bottom-up region proposals in order to localize and segment objects;
> * when labeled training data are scarce, supervised pre-training for an auxiliary task, followed by
domain-specific fine-tuning, boosts performance significantly. 
>-R.Girshick and al.（2014,p.1）

* Pros:
	* the first algorithm which combine region and CNNs.
	* > improves mean average precision (mAP) by more than 50% relative to the previous best result on VOC 2012—achieving a mAP of 62.4%. -R.Girshick and al.(2014,p.1)
* Cons:
	* huge calculation amount

##### [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
I saw Tensorflow detection model zoo included this model so I want to introduce it. Fast R-CNN was proposed by R.Girshick(2015)[3] which is an update to R-CNN and SPPnet.

* Pros:
	* > Fast R-CNN employs several innovations to improve training and testing speed while also increasing detection accuracy. Fast R-CNN trains the very deep VGG16 network 9× faster than R-CNN, is 213× faster at test-time, and achieves a higher mAP on PASCAL VOC 2012. Compared to SPPnet, Fast R-CNN trains VGG16 3× faster, tests 10× faster, and is more accurate. 
	* > streamline the training process for state-of-the-art ConvNet-based object detectors ,jointly learns to classify object proposals and refine their spatial locations.- R.Girshick(2015,p.1)
	* 4-stage -> 2-stage


##### [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)
S.Ren and al.(2016)[4] have introduced a Region Proposal Network(RPN) to improve the running speed of detection networks like SPPnet and Fast R-CNN.RPN shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. Developers merge RPN and Fast R-CNN into a single network by sharing their convolutional features—using neural networks with “attention” mechanisms, the RPN component tells the unified network where to look.(S.Ren and al.2016, p.1)

* Pros:
	* >For the very deep VGG-16 model, Faster R-CNN has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks.- S.Ren and al.2016, p.1

##### [R-FCN](https://arxiv.org/pdf/1605.06409v2.pdf)
Region-based Fully Convolutional Networks(J.Dai and al.2016)[5] expanded the convolutional area to almost whole computation shared on the entire image. The developers also use Residual Networks(ResNets), a popular and latest image classifier backbone.

* Pros:
	* > R-FCN show competitive results on the PASCAL VOC datasets (e.g., 83.6% mAP on the 2007 set) with the 101-layer ResNet. Meanwhile, result is achieved at a test-time speed of 170ms per image, 2.5-20× faster than the Faster R-CNN counterpart.- J.Dai and al.(2016,p.1) 

##### [YOLO](https://arxiv.org/pdf/1506.02640.pdf)
You only look once model (J.Redmon and al.2016)[6] can detect a unified, real-time object. This model puts object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. This model predicts bounding boxes and class probabilities end-to-end directly from full images in one evaluation instead of repurposing classifiers to perform detection.

* Pros:
	* contains real time speed
	* > YOLO model processes images in real-time at 45 frames per second. A smaller version of the network, Fast YOLO,processes an astounding 155 frames per second while still achieving double the mAP of other real-time detectors. Compared to state-of-the-art detection systems, YOLO makes more localization errors but is less likely to predict false positives on background. Finally, YOLO learns very general representations of objects. It outperforms other detection methods, including DPM and R-CNN, when generalizing from natural images to other domains like artwork. - J.Redmon and al.(2016,p.1)
* Cons:
	* makes moew localization errors.
	* mAP is lower than Fast R-CNN and Faster R-CNN VGG-16.
	* So there are YOLOv2, YOLOv3 published in 2016,2018 respectively.

##### [SSD](https://arxiv.org/pdf/1512.02325.pdf)
Similar to YOLO, Single Shot MultiBox Detector(W.Liu and al.2016)[7] uses a single network to predict bounding boxes and class probabilities with a end-to-end CNN architecture. 

* Pros:
	* real-time Object Detection
	* easy to train and straightforward to integrate into systems that require a detection component.
 	* > Experimental results on the PASCAL VOC, COCO, and ILSVRC datasets confirm that SSD has competitive accuracy to methods that utilize an additional object proposal step and is much faster, while providing a unified framework for both training and inference. For 300 × 300 input, SSD achieves 74.3% mAP1 on VOC2007 test at 59 FPS on a Nvidia Titan X and for 512 × 512 input, SSD achieves 76.9% mAP, outperforming a comparable state-of-the-art Faster R-CNN model. Compared to other single stage methods, SSD has much better accuracy even with a smaller input image size.- W.Liu and al.(2016,p.1)

##### [FPN](https://arxiv.org/pdf/1612.03144.pdf)
Feature Pyramid Networks(FPN) use the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost. A topdown architecture with lateral connections is developed for building high-level semantic feature maps at all scales. This architecture, called a Feature Pyramid Network (FPN), shows significant improvement as a generic feature extractor in several applications. (TY.Lin and al,2017,p.1)[8]

* Pros:
	* cost is pretty small
	* can detect small object.
	* > Using FPN in a basic Faster R-CNN system, our method achieves state-of-the-art singlemodel results on the COCO detection benchmark without bells and whistles, surpassing all existing single-model entries including those from the COCO 2016 challenge winners. In addition, our method can run at 6 FPS on a GPU and thus is a practical and accurate solution to multi-scale object detection. - TY.Lin and al(2017,p.1)

* Cons:
	* feature map on the top layer lacks position information.

##### [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf) 
K.He and al.(2017)[9] developed a framework for object detection and image segmentation based on Faster R-CNN. Actually, they added a branch for predicting segmentation masks on each Region of Interest(RoI) via a small FCN, in parallel with Faster R-CNN's branch for classification and bounding box regression.

* Pros:
	* > detect an object and create a segmentation mask for each instance simultaneously. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework.We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, boundingbox object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. We hope our simple and effective approach will serve as a solid baseline and help ease future research in instance-level recognition. - K.He and al.(2017,p.1)

##### [Cascade R-CNN](https://arxiv.org/pdf/1712.00726v1.pdf)
Cascade R-CNN (Z.Cai and al,2017)[10] is the first model to cascade detectors which upgrade two-stage to four-stage for delving into High Quality Object Detection.

* Pros:
	* guide the cascade ideas into object detection.
* Cons:
	* \#17 best model for Object Detection on COCO minival, So the performance now is not good because the tech is improving.

##### [RetinaNet](https://arxiv.org/pdf/1708.02002v2.pdf)
RetinaNet (TY.Lin and al.2018)[11] is a simple one-stage detector for demonstrating the Focal Loss can increase the accuracy of one-stage detector and keep the speed of previous one-stage detectors such as SSD, YOLO. The developers address the extreme foreground-background class imbalance during training by replacing the Cross-entropy Error with Focal Loss.

* Pros:
	* improve the accuracy of one-stage detectors significantly while match the speed of previous one-stage detectors. 


##### [SNIPER](https://arxiv.org/pdf/1805.09300v3.pdf)
SNIPER (Scale Normalization for Image Pyramids with Efficient Resampling) is a multi-scale algorithm for addressing the speed problem of image pyramids developed by B.Singh and al.(2018)[12]. Instead of processing every pixel in an image pyramid, SNIPER propose chips (scale specific context-regions that cover maximum proposals at a particular scale). The number of chips generated per image during training adaptively changes based on the scene complexity.

* Pros
	* combines the advantage of R-CNN on scale and advantage of Fast R-CNN on speed.
	* improve the training speed
	* match the same accuracy of state-of-the-art networks without using high-resolution images on training.

* Cons
	* The detecting speed doesn't be improved.

##### [PANet](https://paperswithcode.com/paper/path-aggregation-network-for-instance)
Path Aggregation Network (PANet) optimized FCN by improving information flow in proposal-based instance segmentation framework.(S.Liu and al.2018)[13]

* Pros
	* > reaches the 1st place in the COCO 2017 Challenge Instance Segmentation task and the 2nd place in Object Detection task without large-batch training. It is also state-of-the-art on MVD and Cityscapes.- S.Liu and al.(2018,p.1)


##### [TridentNet](https://paperswithcode.com/paper/scale-aware-trident-networks-for-object)
Different from other scale method like image pyramid and feature pyramid, Scale-aware Trident Network (Y.Li and al.2019)[14] solves the scale variation based on COCO dataset in the SimpleDet Framework. 
Image Pyramid’s testing speed is slower, but scaling performance is pretty good. Feature Pyramid is similar to image pyramid on feature for speeding up, but the performance is not as good as image pyramid. So the writers purposed a network to solve the scale variation problem by combining the advantages of various receptive fields on different sizes’ objects.[15]

![TridentNet](https://user-images.githubusercontent.com/9766409/66883640-9bd6e800-ef9c-11e9-8214-1566ecccae57.png)

* Pros:
	* solve the problem of scale variation 


##### [CBNet](https://arxiv.org/pdf/1909.03625v1.pdf)
Y.Liu and al.(2019)[16] purposed a Composite Backbone Network (CBNet) for improving the detection performance from existing backbones. And Recently, the best state-of-the-art model on COCO test-dev is done by CBNet and Cascade Mask R-CNN which combines the Cascade R-CNN for object detection and Mask R-CNN for instance segmentation.

* Pros: 
	* easy to integrate into mast detectors.
	* > Specially, by simply integrating the proposed CBNet into the baseline detector Cascade Mask R-CNN, we achieve a new state-of-the-art result on COCO dataset (mAP of 53.3) with single model, which demonstrates great effectiveness of the proposed CBNet architecture. - Y.Liu and al.(2019,p.1)


### Model Zoo
**Model Zoo** means a collection of pre-trained models on different datasets using various algorithms. There are a lot of open-source model zoos, and I introduce three among them which covered most state-of-the-art object detection methods. I recommend a website called [paperswithcode](https://paperswithcode.com/) which provide a lot of state-of-the-art models with codes and papers.

* [Tensorflow Detection Modul Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
	
	###### pros
	* based on Tensorflow object detection Api, have good image classification model library--slim, guanrantee the quality of feature extraction.
	* based on a lot of datasets like COCO, Kitti, Open Images, AVA...
	* many users, many tutorials, many discussions.
	* good examples for a novice to use object detection api.
	###### cons
	* only have several object detection algorithms such as SSD,Fast R-CNN.
	* learning cost is high.
	* hard to change code.

* [Detectron Model Zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md) or [Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)
	
	###### pros
	* Use a lot of state-of-the-art object detection algorithms like Mask R-CNN, RetinaNet, Cascade R-CNN, etc.

	###### cons
	* Learning cost is high due to large models.
	* have a lot of functions, not for object detection only.

* [SimpleDet Model Zoo](https://github.com/TuSimple/simpledet/blob/master/MODEL_ZOO.md)

	###### pros
	* Also use a lot of state-of-the-art object detection algorithms such as TridentNet, RetinaNet, Mask R-CNN, Cascade R-CNN, FPN, etc.
	* Speed is fast, especially for a starter who lacks high-quality GPU.

	###### cons
	* Only on COCO dataset.

### Recommendations
* SimpleDet Model Zoo is kind for beginners who are interest in state-of-the-art models of object detection. First of all, it provides a lot of new algorithms. Secondly, it’s a simple framework based on MXNet Api.
* But if users are familiar with PyTorch, I recommend them to use Detectron’s Model Zoo.
* Similarly, The TensorFlow detection zoo fit the people who familiar with TensorFlow.


###Reference
1. https://medium.com/technology-nineleaps/some-key-machine-learning-definitions-b524eb6cb48
2. R.Girshick and al.（2014):[Region-based Convolutional Networks for
Accurate Object Detection and Segmentation](http://islab.ulsan.ac.kr/files/announcement/513/rcnn_pami.pdf)
3. R.Girshick(2015):[Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
4. S.Ren and al.(2016):[Faster R-CNN: Towards Real-Time Object
Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)
5. J.Dai and al.(2016):[R-FCN: Object Detection via
Region-based Fully Convolutional Networks](https://arxiv.org/pdf/1605.06409v2.pdf)
6. J.Redmon and al.(2016):[You Only Look Once:
Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
7. W.Liu and al.(2016):[SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)
8. TY.Lin and al(2017):[Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf)
9. K.He and al.(2017):[Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
10. Z.Cai and al(2017):[Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/pdf/1712.00726v1.pdf)
11. TY.Lin and al.(2018):[Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002v2.pdf)
12. B.Singh and al.(2018):[SNIPER: Efficient Multi-Scale Training](https://arxiv.org/pdf/1805.09300v3.pdf)
13. S.Liu and al.(2018):[Path Aggregation Network for Instance Segmentation](https://arxiv.org/pdf/1803.01534v4.pdf)
14. Y.Li and al.(2019):[Scale-Aware Trident Networks for Object Detection](https://arxiv.org/pdf/1901.01892v2.pdf)
15. https://zhuanlan.zhihu.com/p/54334986
16. Y.Liu and al.(2019):[CBNet: A Novel Composite Backbone Network Architecture for Object Detection](https://arxiv.org/pdf/1909.03625v1.pdf)
17. https://www.coursera.org/learn/machine-learning/home/info
18. https://www.zhihu.com/question/61173908
19. https://medium.com/zylapp/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852
20. https://blog.csdn.net/jningwei/article/details/86607160
21. https://ieeexplore-ieee-org.ezproxy.bu.edu/stamp/stamp.jsp?tp=&arnumber=8627998
22. https://arleyzhang.github.io/articles/f0c1556d/
