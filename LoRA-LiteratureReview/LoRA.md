LoRA:LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS Literature Review
===========
Achievement
----
* A pre-trained model can be shared and used to build many small LoRA modules for dif ferent tasks. We can freeze the shared model and efficiently switch tasks by replacing the matrices A and B in Figure 1, reducing the storage requirement and task-switching over head significantly. 在原始PLM旁边增加一个旁路，做一个降维再升维的操作，来模拟所谓的 intrinsic rank 。训练的时候固定PLM的参数，只训练降维矩阵A与升维矩阵B。而模型的输入输出维度不变，输出时将BA与PLM的参数叠加。用随机高斯分布初始化A，用0矩阵初始化B，保证训练的开始此旁路矩阵依然是0矩阵。
* LoRA makes training more efficient and lowers the hardware barrier to entry by up to 3 times when using adaptive optimizers since we do not need to calculate the gradients or maintain the optimizer states for most parameters. Instead, we only optimize the injected, much smaller low-rank matrices.
* Our simple linear design allows us to merge the trainable matrices with the frozen weights when deployed, introducing no inference latency compared to a fully fine-tuned model, by construction.
* LoRA is orthogonal to many prior methods and can be combined with many of them, such as prefix-tuning. We provide an example in Appendix E

Solved questions
--------
* Adapter Layers Introduce Inference Latency:   
    However, large neural networks rely on hardware parallelism to keep the latency low, and adapter layers have to be processed sequentially. This makes a difference in the online inference setting where the batch size is typically as small as one. In a generic scenario without model parallelism, such as running inference on GPT-2 (Radford et al., b) medium on a single GPU, we see a noticeable increase in latency when using adapters, even with a very small bottleneck dimension (Table 1). This problem gets worse when we need to shard.

* Directly Optimizing the Prompt is Hard:
    We observe that prefix tuning is difficult to optimize  and that its performance changes non-monotonically in trainable parameters, confirming similar  observations in the original paper. More fundamentally, reserving a part of the sequence length for  adaptation necessarily reduces the sequence length available to process a downstream task, which  we suspect makes tuning the prompt less performant compared to other methods.

Method
--------
The principles outlined here apply  to any dense layers in deep learning models, though we only focus on certain weights in Transformer  language models in our experiments as the motivating use case.

Difficulties and limitations
----------
LoRA also has its limitations. For example, it is not straightforward to batch inputs to different tasks  with different A and B in a single forward pass, if one chooses to absorb A and B into W to eliminate  additional inference latency. Though it is possible to not merge the weights and dynamically choose  the LoRA modules to use for samples in a batch for scenarios where latency is not critical.