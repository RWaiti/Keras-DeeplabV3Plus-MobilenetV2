# Keras implementation of DeepLabV3+ with MobileNetV2 backbone

![cityscapes_img_1](images/dice-spacity-6.png "cityscapes predict")

Epoch 57: val_loss improved from 0.09650 to 0.09584, saving model to model-saved/dice_city_spacity_6.hdf5
1488/1488 [==============================] - 854s 574ms/step - loss: 0.0860 - dice_accuracy_ignoring_last_label: 0.9190 - sparse_accuracy_ignoring_last_label: 0.8966 - val_loss: 0.0958 - val_dice_accuracy_ignoring_last_label: 0.9091 - val_sparse_accuracy_ignoring_last_label: 0.8839 - lr: 5.0056e-06

I'm still running, on my cpu, really slow...


1. **Rethinking Atrous Convolution for Semantic Image Segmentation**<br/>
    Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam.<br/>
    [[link]](http://arxiv.org/abs/1706.05587). arXiv: 1706.05587, 2017.

2. **DeepLabv3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**<br/>
    Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.<br/>
    [[link]](https://arxiv.org/abs/1802.02611). In ECCV, 2018.

3. **MobileNetV2: Inverted Residuals and Linear Bottlenecks**<br/>
    Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen<br/>
    [[link]](https://arxiv.org/abs/1801.04381). In CVPR, 2018.

4. **The Cityscapes Dataset for Semantic Urban Scene Understanding**<br/>
    Cordts, Marius, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, Bernt Schiele. <br/>
    [[link]](https://www.cityscapes-dataset.com/). In CVPR, 2016.
