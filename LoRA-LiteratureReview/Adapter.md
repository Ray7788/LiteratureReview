Parameter-Efficient Transfer Learning for NLP Literature Review
======
Transfer learning Baseline
-----
The two most common transfer learning techniques in NLP are **feature-based transfer** and **fine-tuning**.  
Features-based transfer involves pre-training real-valued embeddings vectors. These  embeddings may be at the word (Mikolov et al., 2013), sentence (Cer et al., 2019), or paragraph level (Le & Mikolov,  2014). The embeddings are then fed to custom downstream  models. 
Fine-tuning involves copying the weights from a  pre-trained network and tuning them on the downstream  task. Recent work shows that fine-tuning often enjoys better  Accuracy performance than feature-based transfer (Howard & Ruder, 2018). 

Both feature-based transfer and fine-tuning require a new set of weights for each task. Fine-tuning is more parameter effificient if the lower layers of a network are shared between tasks

**Adapter-based tuning** requires training two orders of magnitude fewer parameters to fine-tuning, while attaining similar performance.
Adapters are new modules added between layers of a pre-trained network. Adapter-based tuning differs from feature-based transfer and fine-tuning in the following way:
假设原始的预训练模型的参数为ω，加入的adapter 参数为υ，在针对不同下游任务进行调整时，只需要将预训练参数固定住，只针对adapter参数υ进行训练。通常情况下，参数量υ<<ω, 因此在对多个下游任务调整时，只需要调整极小数量的参数，大大的提高了预训练模型的扩展性和实用性。

Adapter-based tuning relates to multi-task and continual learning. Multi-task learning also results in compact models. However, multi-task learning requires simultaneous access to all tasks, which adapter-based tuning does not require. Continual learning systems aim to learn from an endless stream of tasks. This paradigm is challenging because networks forget previous tasks after re-training (McCloskey & Cohen, 1989; French, 1999). Adapters differ in that the tasks do not interact and the shared parameters are frozen. This means that the model has perfect memory of previous tasks using a small number of task-specific parameters.

**Solved questions**
-------
(i) it attains good performance, (ii) it permits training on tasks sequentially, that is, it does not require simultaneous access to all datasets, and (iii) it adds only a small number of additional parameters per task.

Details
-------
Tuning with adapter modules involves adding a small number of new parameters to a model, which are trained on the downstream task. When performing vanilla fine-tuning of deep networks, a modification is made to the **top layer** of the network. Adapter modules perform more general architectural modifications to re-purpose a pre-trained network for a downstream task. In particular, the adapter tuning strategy involves injecting new layers into the original network. The weights of the original network are untouched, whilst the new adapter layers are initialized at random. the parameters of the original network are frozen and therefore may be shared by many tasks.

Adapter modules have two main features: a small number of parameters, and a near-identity initialization.

Instantiation for Transformer Networks
-------
Each layer of the Transformer contains two primary sub-layers: an attention layer and a feedforward layer. Both layers are followed immediately by a projection that maps the features size back to the size of layer’s input. 
A skip-connection is applied across each of the sub-layers. The output of each sub-layer is fed into layer normalization. We insert two serial adapters after each of these sub-layers. The adapter is always applied directly to the output of the sub-layer, after the projection back to the input size, but before adding the skip connection back. The output of the adapter is then passed directly into the following layer normalization.

To limit the number of parameters, we propose a **bottleneck** architecture. The adapters first project the original d-dimensional features into a smaller dimension, m, apply a nonlinearity, then project back to d dimensions.


降维（Downsampling）： 通过使用较小的卷积核或步幅大于1的卷积操作，将输入的空间维度降低。这有助于减少计算成本和参数数量。

扩展（Expansion）： 在降维之后，使用较大的卷积核或扩张卷积（dilated convolution）等操作，将维度扩展。这有助于提取更高级别的特征。

恢复维度（Upsampling）： 最终，通过上采样或者使用恰当的卷积操作，将特征图的维度恢复到较高的水平。

The adapter module itself has a skip-connection internally. With the skip-connection, if the parameters of the projection layers are initialized to near-zero, the module is initialized to an approximate identity function.

Alongside the layers in the adapter module, we also train
new layer normalization parameters per task.