# Explore latent variables effects of VAE & Conditional VAE
The project is implemented by ChaoyangZheng, RuiFan, ZiangGuo and WenhuiPeng for PHD course: Machine Learning II Deep Learning and Applications in HEC MONTREAL. 

The complete report can be found in the "report" folder.

## Introduction
This project focused on generating images by variational autoencoder on MNIST dataset. The experiments aims explore the latent variables effect (including changing the number of hidden dimensions, and visualizing learned low dimension manifold), performance of different network architectures as function approximators. In addition, we compare conditional variational autoencoder with variational autoencoder.

## Experiments and discussion
In this section, we experiment VAEs on MNIST handwritten digits dataset, exploring the latent variables effect (including changing the number of hidden dimensions, and low dimension manifold visualization). Then we compare different network architectures as VAEs function approximators. In the end we use conditional VAEs (CVAE) to learn structured output representation, generating images with certain labels. All VAE models in the experiments are modiﬁed based on the vanilla architecture shown in ﬁgure 1.
![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/explore-latent-variables-effects-of-VAE-Conditional-VAE-/master/graphs/1.PNG "Optional title")

### 1 Latent dimension effect 
In this experiment, we investigate generated image quality of VAEs with respect to different latent dimensions to gain insights towards the impacts of hidden dimension to VAE’s performance. Speciﬁcally, we want to compare quality of reconstructed images and generated images on different latent dimensions.We used the vanilla VAE architecture in ﬁgure 1, which has symmetric two layer MLP as encoder (784-400-latent) and decoder (latent-400-784), where latent is the hidden dimension, set to 2,5,10,20 in the experiments.We train this VAE by 20 epochs (loss stopped decreasing) via Adam (learningrate = 1e−3) Results of reconstructed images and generated images are shown in ﬁgure 2 and ﬁgure 3.

### 2 Learned manifold visualization 
In this experiment, we aim to control generated images. Speciﬁcally, we can pick up two hidden dimensions and ﬁx the rest of them, then we investigate the visualization representation of the given dimensions. Since in the theory of VAE, the prior of the latent space is Gaussian distribution, we used the inverse CDF of the Gaussian to produce the latent variable from the random sample. For each of these transformed latent variable, we plotted the generative image with the learned VAE model.

We also explored this effect with different function approximators: 2 layer MLP with hidden dimensions of 2, 3, 5, 10; 2 layer CNN with hidden dimensions of 2, 3, 5, 10; 3 layer CNN with hidden dimensions of 2, 3, 5, 10. 

The visualisation result of three model with hidden dimension 2 are shown in the Figure 4. The result shows that difference of the picture grid by two CNN layers is slightly smaller than that provided by two fully connected layers. This means the generative power of two layer CNN VAE model is not as strong as VAE with two fully connected layers, which is consistent with the result of section 3.3. As can be seen in the ﬁgure, the small change in the latent variable will cause a slight change in the generative images. We can understand how the latent variable impact output digit, such as output "8" change to "2" and then to "6" with one latent dimension change. 

The latent effect of 3 layer-CNN VAE with different hidden dimension is shown in the ﬁgure (5). As can be seen, the more the hidden dimension is, the smaller the difference of latent effect can be seen. This is easy to understand, because the more hidden dimension it has, the more dimension of latent variable remains the same. For example, with hidden dimension to be 10, only 2 out of 10 dimension are changed. Therefore, the difference of the generated images is smaller, comparing to the situation of low hidden dimension.

### 3 Architecture effect 
In this experiment, we investigate the effect of different architecture as function approximators.We change the number of convolution layers on the encoder and decoder, based on the CNN shown in ﬁgure 6. We train the CNN VAEs for 20 epochs with Adam (learningrate = 1e−3). 

The generated images are shown in ﬁgure 7, we noticed the generated image quality improves as increasingthenumberoflayersfromonetofour. However,theimprovementbecomeslesssigniﬁcant when the model has more layers. And when it comes to comparing from four and ﬁve layers, the results are very similar. This is reasonable because model complexity increases with CNN depth, when the model has enough complexity to learn, increasing the depth will no longer help.

### 4 Structured learning via conditional VAE
In this experiment, our task is to generate hand written digits that belong to a certain class. CVAEs [2] make it possible, where the input-to-output mapping is one-to-many, without requiring explicitly specify the structure of the output distribution. 

The CVAE architecture in experiment is shown in ﬁgure 8.Compared with common VAEs, CVAE concatenate the image X , and one hot vector of label Ay as the encoder input. The encoder is composed of three denselayers: 794(28∗28+10)−512−256−2, and thedecoder has symmetric architecture 12(2+10)−256−512−784,except the two dimensional latent variable is concatenated with the one hot label vector as the decoder input. The CVAE model is trained for 50 epochs with Adam (learningrate = 1e−3). 

The generated images are shown in ﬁgure 9, the generated images in each column has ﬁxed label from 0 to 9. As we can see, compared to the vanilla VAE architecture in the ﬁrst experiment, the samples generated by the CVAE models are more realistic and diverse in shape, though they all use dense layers as function approximators. 

Then we explore the latent variables effect on CVAE, by ﬁxing one latent dimension, and change the rest(1 dimension in this experiment, since we used only two hidden dimensions as described in the CVAE architecture). Figure 10 shows the latent variable effect on CVAE, we noticed a clear gradual changing in the patterns of generated digits. For example, the slope of digits change from vertical to more inclined angle (0,1,4,7,8). Moreover, we found that changing the same latent variable can have different effect on different classes: the patterns of (2,5,9) digits change in more diverse way, rather than simply increasing the inclined angle.

## Conclusion 
• The quality of reconstructed images increases with hidden dimension,however this rule does not apply to generated images, since there exist an optimal latent dimension for generated images. 

• The learned manifold visualization shows that the generated image can be controlled via latent variable manipulation, for example the incline angle of digits changes with the latent variable certain dimension. 

• When the VAE has shallow CNN as encoder and decoder, increasing the depth of convolutional layers improves performance. 

• Compared to vanilla VAE, CVAE models are able to generate more realistic and diverse samples; same latent dimension has different effect on conditions (labels).


## Appendix1: Figures
![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/explore-latent-variables-effects-of-VAE-Conditional-VAE-/master/graphs/2.PNG "Optional title")

![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/explore-latent-variables-effects-of-VAE-Conditional-VAE-/master/graphs/3.PNG "Optional title")

![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/explore-latent-variables-effects-of-VAE-Conditional-VAE-/master/graphs/4.PNG "Optional title")

![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/explore-latent-variables-effects-of-VAE-Conditional-VAE-/master/graphs/5.PNG "Optional title")

![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/explore-latent-variables-effects-of-VAE-Conditional-VAE-/master/graphs/6.PNG "Optional title")

![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/explore-latent-variables-effects-of-VAE-Conditional-VAE-/master/graphs/7.PNG "Optional title")

## Appendix2: References
[1] Kingma & Diederik P & Welling, Max (2013) Auto-encoding variational bayes, arXiv preprint arXiv:1312.6114 

[2]Sohn,Kihyuk&Lee,Honglak&Yan,Xinchen (2015)LearningStructuredOutputRepresentation using Deep Conditional Generative Models, Advances in Neural Information Processing Systems 28, pp. 3483–3491. Curran Associates, Inc.



