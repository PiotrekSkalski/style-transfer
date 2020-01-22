# Fast style transfer
<div align = 'center'>
<p align = 'center'>
<img src = 'https://github.com/PiotrekSkalski/style-transfer/blob/master/images/content_images/amber.jpg' height = '250px' width = '250px'>
</p>
<p align = 'center'>
<img src = 'https://github.com/PiotrekSkalski/style-transfer/blob/master/images/style_images/mosaic.jpg' height = '250px' width = '250px'>
<img src = 'https://github.com/PiotrekSkalski/style-transfer/blob/master/images/output_images/mosaic_door.png' height = '250px' width = '250px'>
</p>
<p align = 'center'>
<img src = 'https://github.com/PiotrekSkalski/style-transfer/blob/master/images/style_images/style_edgy.jpg' height = '250px' width = '250px'>
<img src = 'https://github.com/PiotrekSkalski/style-transfer/blob/master/images/output_images/picasso_door.png' height = '250px' width = '250px'>
</p>
<p align = 'center'>
<img src = 'https://github.com/PiotrekSkalski/style-transfer/blob/master/images/style_images/scream.jpg' height = '250px' width = '250px'>
<img src = 'https://github.com/PiotrekSkalski/style-transfer/blob/master/images/output_images/scream.png' height = '250px' width = '250px'>
</p>
</div>
Neural style transfer is an exciting field where neural networks are used to edit an image in order to make it adopt a style of a different image. It was first successfully applied by Gatys: he optimised pixels of a randomly initialised image by minimizing a loss function, which was a measure of content retained from the original image and similarity in style to a target style image. The key to the success of this method was to construct the loss function based on feature maps from a pretrained VGG neural network. The first few layers of the VGG network encode small-scale information from an image, and can be used to construct content loss as well as transfer the fine-grained style like dominant colours and brush strokes. Subsequent layers contain larger-scale features that further encode the style.

While this method gives visually pleasing results, it is time consuming - it can take several minutes on a GPU to stylise an image. Hence researchers tried to find methods that would approximate the pixel optimization process. One successful attempt by Johnson et al. used an encoder-decoder convolutional network that was trained to stylise images in a particular style. Although training this network takes a few hours, stylising images can be done in real time and allows for real-time video style transfer.

In this repository I make my own attemp at neural style transfer using the method from Johnson's paper. As a guiding example I use the [fast-neural-style](https://github.com/pytorch/examples/tree/master/fast_neural_style) repo from Pytorch - I borrow the architecture of the TransformerNet() module as it is. My own contribution includes:
* Using fastai library to train the network. It includes a [one-cycle policy](https://docs.fast.ai/callbacks.one_cycle.html) learning rate scheduler which can speed up training and sometimes gives better stylization results.
* VggLoss module that uses pytorch hooks to create a custom loss function from a list of vgg layer names and corresponding content and style weights. It also includes a total variation regularization which enforces smoother outputs.
* Pretraining the TransformerNet on the task of recreating the input image. This transfer learning method gives better results and reduces training time .

The code is in the form of a jupyter notebook and can be run in Colab or on any other machine, ideally containing a GPU. It depends on [fastai](https://docs.fast.ai/) and [PyTorch](https://pytorch.org/).

### Pretraining
<p align = 'center'>
<img src = 'https://github.com/PiotrekSkalski/style-transfer/blob/master/images/output_images/autoencoder_l2.png' height = '250px' width = '250px'>
</br>
Trained with MSE loss
</br>
<img src = 'https://github.com/PiotrekSkalski/style-transfer/blob/master/images/output_images/autoencoder_vgg.png' height = '250px' width = '250px'>
</br>
Trained with content loss from vgg
</br>
</p>
Pretraining the TransformerNet as an autoencoder, i.e. teaching it to reconstruct the input image, acts like transfer learning. Network initialised in this way reduces training time for style tranfer training by two to five times and can be used on smaller datasets. It also leads to more visually pleasing results.
Using pixel mean squared error as a loss function gives blurred results - small differences in pixels carry very little loss. Interestingly, this problem can be mitigated by using MSE loss of activations from a pretrained vgg network, i.e. content loss from style tranfer. By using activations from the first ReLu layer I managed to achieve much better results - the reconstrructed images are much sharper, which probably stems from the fact that the first feature maps in vgg learn to recognise fine-grained texture of an image.


