# Keras implementation of DeepLabV3+ with MobileNetV2 backbone

| IMAGE | TRUE_MASK | PRED MASK <br/> mIoU: .56 | IMG + PRED MASK |
| :---: | :-------: | :-------: | :--------: |
| ![cityscapes_img_1](images/dice-spacity-6_1.png "Cityscapes Image") | ![cityscapes_img_2](images/dice-spacity-6_2.png "Cityscapes True Mask") | ![cityscapes_img_3](images/dice-spacity-6_3.png "Cityscapes Pred Mask") | ![cityscapes_img_4](images/dice-spacity-6_4.png "Cityscapes Image + Pred Mask") |

Epoch 57: val_loss improved from 0.09650 to 0.09584, saving model to model-saved/dice_city_spacity_6.hdf5 <br/>
1488/1488 [==============================] - 854s 574ms/step <br/>
loss: 0.0860 - dice_accuracy_ignoring_last_label: 0.9190 - sparse_accuracy_ignoring_last_label: 0.8966 <br/> 
val_loss: 0.0958 - val_dice_accuracy_ignoring_last_label: 0.9091 - val_sparse_accuracy_ignoring_last_label: 0.8839

I'm still running, on my cpu, really slow...

1. **Rethinking Atrous Convolution for Semantic Image Segmentation** <br/>
    Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam. <br/>
    [[link]](http://arxiv.org/abs/1706.05587). arXiv: 1706.05587, 2017.

2. **DeepLabv3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation** <br/>
    Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam. <br/>
    [[link]](https://arxiv.org/abs/1802.02611). In ECCV, 2018.

3. **MobileNetV2: Inverted Residuals and Linear Bottlenecks** <br/>
    Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen <br/>
    [[link]](https://arxiv.org/abs/1801.04381). In CVPR, 2018.

4. **The Cityscapes Dataset for Semantic Urban Scene Understanding** <br/>
    Cordts, Marius, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, Bernt Schiele.  <br/>
    [[link]](https://www.cityscapes-dataset.com/). In CVPR, 2016.
