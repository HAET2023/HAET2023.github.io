---
layout: default
title: Accepted Papers
---

This is the list of all accepted papers. 


# Spotlights

**GradMax: Gradient Maximizing Neural Network Growth** 
 <br /> Utku Evci; Fabian Pedregosa; Max Vladymyrov; Thomas Unterthiner<br />
<abstract>Neural networks are often optimized in a fixed parameter space, i.e. architectures are chosen before training and kept constant. In this work we instead focus on starting training with a small seed architecture and letting the network grow itself using a simple idea: maximizing gradients. Studying this idea we propose a new technique called Gradient Maximizing Growth (GradMax). GradMax is fast, maximizes gradient norm and improves optimization speed/time.</abstract>

[PDF](https://www.dropbox.com/s/5ip0wp35gdcckgg/GradMax%20Gradient%20Maximizing%20Neural%20Network%20Growth%20Paper.pdf?dl=0) &bull;
[Poster](https://www.dropbox.com/s/t01jsmwv1l5043s/GradMax%20Gradient%20Maximizing%20Neural%20Network%20Growth%20Poster.pdf?dl=0) &bull;
[Presentation](https://youtu.be/IYm0v6OxYU0)


**GRADIENT MATCHING FOR EFFICIENT LEARNING** <br />
Krishnateja Killamsetty; Durga S; Baharan Mirzasoleiman; Ganesh Ramakrishnan; Abir De; Rishabh Iyer<br />
<abstract>The great success of modern machine learning models on large datasets is contingent on extensive computational resources with high financial and environmental
costs. One way to address this is by extracting subsets that generalize on par with the full data. In this work, we propose a general framework, GRAD-MATCH, which
finds subsets that closely match the gradient of the training or validation set. We find such subsets effectively using an orthogonal matching pursuit algorithm. Our
extensive experiments on real-world datasets show that GRAD-MATCH significantly and consistently outperforms several recent data-selection algorithms and is Pareto-optimal with respect to the accuracy-efficiency trade-off.</abstract>




**FULLY QUANTIZING TRANSFORMER-BASED ASR FOR EDGE DEPLOYMENT** <br />
Alex Bie; Bharat Venkitesh; Joao B Monteiro; Md. Akmal Haidar; Mehdi Rezagholizadeh<br />
<abstract>While significant improvements have been made in recent years in terms of end-to-end automatic speech recognition (ASR) performance, such improvements were obtained through the use of very large neural networks, unfit for embedded use on edge devices. That being said, in this paper, we work on simplifying and compressing Transformer-based encoder-decoder architectures for the end-to-end ASR task. We empirically introduce a more compact Speech-Transformer by investigating the impact of discarding particular modules on the performance of the model. Moreover, we evaluate reducing the numerical precision of our network’s weights and activations while maintaining the performance of the full-precision model. Our experiments show that we can reduce the number of parameters of the full-precision model and then further compress the model 4x by fully quantizing to 8-bit fixed point precision.</abstract>

[PDF](https://www.dropbox.com/s/pa03dwhqc12ilfe/FULLY%20QUANTIZING%20TRANSFORMER-BASED%20ASR%20FOR%20EDGE%20DEPLOYMENT%20Paper.pdf?dl=0) &bull;
[Poster](https://www.dropbox.com/s/w1tqjy8rncgue0m/FULLY%20QUANTIZING%20TRANSFORMER-BASED%20ASR%20FOR%20EDGE%20DEPLOYMENT%20Poster.pdf?dl=0) &bull;
[Presentation](https://youtu.be/nI-q33syFHw)


**ACTORQ: QUANTIZATION FOR ACTOR-LEARNER DISTRIBUTED REINFORCEMENT LEARNING** <br />
Maximilian Lam; Sharad Chitlangia; Srivatsan Krishnan; Zishen Wan; Gabriel Barth-Maron; Aleksandra Faust; Vijay Janapa Reddi<br />
<abstract>In this paper, we introduce a novel Reinforcement Learning (RL) training framework, ActorQ, for speeding up actor-learner distributed RL training. ActorQ leverages full precision optimization on the learner, and distributed data collection through lower-precision quantized actors. The quantized, 8-bit (or 16 bit) inference on actors, speeds up data collection without affecting the convergence. The quantized distributed RL training system, ActorQ, demonstrates end to end speedups of > 1.5 × - 2.5 ×, and faster convergence over full precision training on a range of tasks (Deepmind Control Suite) and different RL algorithms (D4PG, DQN). Finally, we break down the various runtime costs of distributed RL training (such as communication time, inference time, model load time, etc) and evaluate the effects of quantization on these system attributes.</abstract>

[PDF](https://www.dropbox.com/s/230tshb5viiuy31/ACTORQ%20QUANTIZATION%20FOR%20ACTOR-LEARNER%20DISTRIBUTED%20REINFORCEMENT%20LEARNING%20Paper.pdf?dl=0) &bull;
[Poster](https://www.dropbox.com/s/79topxjzurqkse9/ACTORQ%20QUANTIZATION%20FOR%20ACTOR-LEARNER%20DISTRIBUTED%20REINFORCEMENT%20LEARNING%20Poster.pdf?dl=0) 



**OPTIMIZER FUSION: EFFICIENT TRAINING WITH BETTER LOCALITY AND PARALLELISM** <br />
Zixuan Jiang; Jiaqi Gu; Mingjie Liu; Keren Zhu; David Z Pan<br />
<abstract>Machine learning frameworks adopt iterative optimizers to train neural networks. Conventional eager execution separates the updating of trainable parameters from forward and backward computations. However, this approach introduces nontrivial training time overhead due to the lack of data locality and computation parallelism. In this work, we propose to fuse the optimizer with forward or backward computation to better leverage locality and parallelism during training. By re-ordering the forward computation, gradient calculation, and parameter updating, our proposed method improves the efficiency of iterative optimizers. Experimental results demonstrate that we can achieve an up to 20% training time reduction on various configurations. Since our methods do not alter the optimizer algorithm, they can be used as a general “plug-in” technique to the training process.</abstract>

[PDF](https://www.dropbox.com/s/7vw3rbikk0e0edi/OPTIMIZER%20FUSION%20EFFICIENT%20TRAINING%20WITH%20BETTER%20LOCALITY%20AND%20PARALLELISM%20Paper.pdf?dl=0) &bull;
[Poster](https://www.dropbox.com/s/wezzjpnmwz853b1/OPTIMIZER%20FUSION%20EFFICIENT%20TRAINING%20WITH%20BETTER%20LOCALITY%20AND%20PARALLELISM%20Poster.pdf?dl=0) &bull;
[Presentation](https://www.youtube.com/watch?v=UfDSUV_pmr8)


**GROUPED SPARSE PROJECTION FOR DEEP LEARNING** <br />
Riyasat Ohib; Nicolas Gillis; Sameena Shah; Vamsi K Potluru; Sergey Plis<br />
<abstract>Accumulating empirical evidence shows that very large deep learning models learn faster and achieve higher accuracy than their smaller counterparts. Yet, smaller models have benefits of energy efficiency and are often easier to interpret. To simultaneously get the benefits of large and small models we often encourage sparsity in the model weights of large models. For this, different approaches have been proposed including weight-pruning and distillation. Unfortunately, most existing approaches do not have a controllable way to request a desired value of sparsity as an interpretable parameter and get it right in a single run. In this work, we design a new sparse projection method for a set of weights in order to achieve a desired average level of sparsity without additional hyperparameter tuning which is measured using the ratio of the l1 and l2 norms. Instead of projecting each vector of the weight matrix individually, or using sparsity as a regularizer, we project all vectors together to achieve an average target sparsity, where the sparsity levels of the individual vectors of the weight matrix are automatically tuned. Our projection operator has the following guarantees – (A) it is fast and enjoys a runtime linear in the size of the vectors; (B) the solution is unique except for a measure set of zero. We utilize our projection operator to obtain the desired sparsity of deep learning models in a single run with a negligible performance hit, while competing methods require sparsity hyperparameter tuning. Even with a single projection of a pre-trained dense model followed by fine-tuning, we show empirical performance competitive to the state of the art. We support these claims with empirical evidence on real-world datasets and on a number of architectures, comparing it to other state of the art methods including DeepHoyer.</abstract>

[PDF](https://www.dropbox.com/s/tk477kyoegntlqq/GROUPED%20SPARSE%20PROJECTION%20FOR%20DEEP%20LEARNING%20Paper.pdf?dl=0) &bull;
[Poster](https://www.dropbox.com/s/yw28r9p0jbgju4d/GROUPED%20SPARSE%20PROJECTION%20FOR%20DEEP%20LEARNING%20Poster.pdf?dl=0) &bull;
[Presentation](https://www.youtube.com/watch?v=ohZq_Xa4diw)


**GRADIENT DESCENT WITH MOMENTUM USING DYNAMIC STOCHASTIC COMPUTING** 
 <br />Siting Liu & Warren J. Gross<br />
<abstract>In this paper, dynamic stochastic computing is used to simplify the computations involved in updating the weights in neural networks. Specifically, a stochastic circuit is proposed to perform the gradient descent algorithm with momentum, including the exponential moving average operations in the optimizer and iterative update of the weights. The use of stochastic circuits can reduce the hardware resources and energy consumption compared to conventional fixed-/floating-point arithmetic circuits, since information is processed in the form of single bits in stochastic circuits instead of fixed-/floating-point numbers. The stochastic circuits are then deployed in the training of VGG16, ResNet18 and MobileNetV2 on CIFAR-10 dataset. A similar test accuracy is obtained compared to their floating-point implementations.</abstract>

[PDF](https://www.dropbox.com/s/v5qlu1tdd11g6y1/GRADIENT%20DESCENT%20WITH%20MOMENTUM%20USING%20DYNAMIC%20STOCHASTIC%20COMPUTING%20Paper.pdf?dl=0) &bull;
[Poster](https://www.dropbox.com/s/veo6niip4mbghwa/GRADIENT%20DESCENT%20WITH%20MOMENTUM%20USING%20DYNAMIC%20STOCHASTIC%20COMPUTING%20Poster.pdf?dl=0) &bull;
[Presentation](https://youtu.be/e8oiwDS-tqo)


**MEMORY-BOUNDED SPARSE TRAINING ON THE EDGE** <br />
Xiaolong Ma; Zhengang Li; Geng Yuan; Wei Niu; Bin Ren; Yanzhi Wang; Xue Lin<br />
<abstract>Recently, a new trend of exploring sparsity for accelerating neural network training has emerged, embracing the paradigm of training on the edge. Different from the existing works for sparse training, this work identifies the memory footprint as a critical limiting factor and reveals the importance of sparsity schemes on the performance of sparse training in terms of accuracy, training speed, and memory footprint. To achieve that, this paper presents a novel solution that can enable the end-to-end sparse training of deep neural networks on edge devices. Specifically, the proposed Memory-Bounded Sparse Training (MBST) framework (i) practically restricts the memory footprint to support the end-to-end training on the edge, and (ii) achieves significant training speedups in relevance to the sparsity schemes, (iii) with the capability of maintaining high accuracy. On CIFAR-100, the MBST consistently outperforms representative SOTA works in all aspects of accuracy, training speed, and memory footprint.</abstract>

[PDF](https://www.dropbox.com/s/ymfqtauq9kwwzey/MEMORY-BOUNDED%20SPARSE%20TRAINING%20ON%20THE%20EDGE%20Paper.pdf?dl=0) &bull;
[Poster](https://www.dropbox.com/s/xofqvi73dvwl8tf/MEMORY-BOUNDED%20SPARSE%20TRAINING%20ON%20THE%20EDGE%20Poster.pdf?dl=0) &bull;
[Presentation](https://youtu.be/nkhv12azSHI)



**A FAST METHOD TO FINE-TUNE NEURAL NETWORKS FOR THE LEAST ENERGY CONSUMPTION ON FPGAS** 
 <br />Morteza Hosseini; Mohammad Ebrahimabadi; Arnab Mazumder; Houman Homayoun; Tinoosh Mohseninn<br />
<abstract>Because of their simple hardware requirements, low bitwidth neural networks (NNs) have gained significant attention over the recent years, and have been extensively employed in electronic devices that seek efficiency and performance. Research has shown that scaled-up low bitwidth NNs can have accuracy levels on par with their full-precision counterparts. As a result, there seems to be a trade-off between quantization (q) and scaling (s) of NNs to maintain the accuracy. In this paper, we propose QS-NAS which is a systematic approach to explore the best quantization and scaling factors for a NN architecture that satisfies a targeted accuracy level and results in the least energy consumption per inference when de-ployed to a hardware–FPGA in this work. Compared to the literature using the same VGG-like NN with different q and s over the same datasets, our selected optimal NNs deployed to a low-cost tiny Xilinx FPGA from the ZedBoard resulted in accuracy levels higher or on par with those of the related work, while giving the least power dissipation and the highest inference/Joule.</abstract>

[PDF](https://www.dropbox.com/s/yqjr4uurswv7foj/A%20FAST%20METHOD%20TO%20FINE-TUNE%20NEURAL%20NETWORKS%20FOR%20THE%20LEAST%20ENERGY%20CONSUMPTION%20ON%20FPGAS%20Paper.pdf?dl=0) &bull;
[Poster](https://www.dropbox.com/s/6wjggoi1tws6x8w/A%20FAST%20METHOD%20TO%20FINE-TUNE%20NEURAL%20NETWORKS%20FOR%20THE%20LEAST%20ENERGY%20CONSUMPTION%20ON%20FPGAS%20Poster.pdf?dl=0) &bull;
[Presentation](https://youtu.be/fjybljxuAVE)


**SELF-REFLECTIVE VARIATIONAL AUTOENCODER** <br />
Ifigeneia Apostolopoulou; Elan Rosenfeld; Artur Dubrawski<br />
<abstract>The Variational Autoencoder (VAE) is a powerful framework for learning probabilistic latent variable generative models. However, typical assumptions on the approximate posterior distributions can substantially restrict its capacity for inference and generative modeling. More importantly, the restricted capacity is usually compensated for by increasing complexity of the latent space, adding significant computational and memory overhead. In this work, we introduce an orthogonal solution which layer-wise intertwines the latent and observed space, a process we call self-reflective inference. By modifying the structure of existing VAE architectures, self-reflection ensures that the stochastic flow preserves the factorization of the exact posterior, updating the latent codes to be consistent with the generative model. We empirically demonstrate the computational advantages of matching the variational posterior to the exact generative posterior. On binarized MNIST, self-reflective inference achieves state of the art performance without resorting to complex, expensive components such as autoregressive layers. On CIFAR-10, our model matches or outperforms very deep architectures with orders of magnitude smaller stochastic layers, achieving a high compression ratio in a fraction of the training time, without diminishing accuracy. Our proposed modification is quite general and complements the existing literature. Self-reflective inference can naturally leverage advances in distribution estimation and generative modeling to improve the capacity of each layer in the hierarchy.</abstract>

[PDF](https://www.dropbox.com/s/g0ygb6radm3pfml/SELF-REFLECTIVE%20VARIATIONAL%20AUTOENCODER%20Paper.pdf?dl=0) &bull;
[Poster](https://www.dropbox.com/s/n0ttlp53kdvl9q4/SELF-REFLECTIVE%20VARIATIONAL%20AUTOENCODER%20Poster.pdf?dl=0) &bull;
[Presentation](https://youtu.be/oBe75OrZQd8)


**ADAPTIVE FILTERS AND AGGREGATOR FUSION FOR EFFICIENT GRAPH CONVOLUTIONS** <br />
Shyam A Tailor; Felix Opolka; Pietro Lió; Nic Lane<br />
<abstract>Training and deploying graph neural networks (GNNs) remains difficult due to their high memory consumption and inference latency. In this work we present a new type of GNN architecture that achieves state-of-the-art performance with lower memory consumption and latency, along with characteristics suited to accelerator implementation. Our proposal uses memory proportional to the number of vertices in the graph, in contrast to competing methods which require memory proportional to the number of edges; we find our efficient approach actually achieves higher accuracy than competing approaches across 5 large and varied datasets against strong baselines. We achieve our results by using a novel adaptive filtering approach inspired by signal processing; it can be interpreted as enabling each vertex to have its own weight matrix, and is not related to attention. Following our focus on efficient hardware usage, we propose aggregator fusion, a technique to enable GNNs to significantly boost their representational power, with only a small increase in latency of 19% over standard sparse matrix multiplication. Code and pretrained models can be found at this URL: https://github.com/shyam196/egc.</abstract>

[PDF](https://www.dropbox.com/s/ezdkzsimbms94u2/ADAPTIVE%20FILTERS%20AND%20AGGREGATOR%20FUSION%20FOR%20EFFICIENT%20GRAPH%20CONVOLUTIONS%20Paper.pdf?dl=0) &bull;
[Poster](https://www.dropbox.com/s/v921ns0qcwkium7/ADAPTIVE%20FILTERS%20AND%20AGGREGATOR%20FUSION%20FOR%20EFFICIENT%20GRAPH%20CONVOLUTIONS%20Poster.pdf?dl=0) &bull;
[Presentation](https://youtu.be/oFX4LDUcDTk)

**An Exact Penalty Method for Binary Training** <br />
Tim Dockhorn; Yaoliang Yu; Vahid Partovi Nia; Eyyüb Sari<br />
<abstract>It is common practice to quantize a subset of the network parameters to deploy deep neural networks on resource-constrained devices, such as cell phones, IoT devices, autonomous cars, etc. Techniques to quantize models include the straightthrough gradient method and model regularization. Generally, these methods do not ensure convergence of the optimization algorithm. We propose an exact penalty method for binary quantization that converges finitely to {−1, 1}. An empirical study shows that our approach leads to competitive performance on CIFAR-10 with ResNets. We further show that the penalty parameter can be used as a knob for training efficiency: larger values give faster convergence whereas smaller values lead to superior performance. As an extension, we demonstrate why our algorithm is well-suited for convex objectives with binary constraints and we show how our method can be extended to other discrete constraint sets.
</abstract>




**TRAINING CNNS FASTER WITH INPUT AND KERNEL DOWNSAMPLING** <br />
Zissis Poulos; Ali Nouri; Andreas Moshovos<br />
<abstract>We reduce the total number of floating-point operations (FLOPs) required for training convolutional neural networks (CNNs) with a method that, for some of the mini-batches, a) scales down the resolution of input images via downsampling, and b) reduces the number of forward pass operations via pooling on the convolution filters. Training is performed in an interleaved fashion; some batches follow the regular forward and backpropagation schedule with original network parameters, whereas others undergo a forward pass with pooled filters and downsampled inputs. Since pooling is differentiable, gradients of the pooled filters flow back to the original network parameters for a standard parameter update. Experiments with residual architectures on CIFAR-10 show that we can achieve up to 23% reduction in training time with up to 2.9% relative increase in validation error.</abstract>

[PDF](https://www.dropbox.com/s/t98w41vfuztv9uw/TRAINING%20CNNS%20FASTER%20WITH%20INPUT%20AND%20KERNEL%20DOWNSAMPLING%20Paper.pdf?dl=0) &bull;
[Poster](https://www.dropbox.com/s/t02fchlhzn3i7zg/TRAINING%20CNNS%20FASTER%20WITH%20INPUT%20AND%20KERNEL%20DOWNSAMPLING%20Poster.pdf?dl=0) &bull;
[Presentation](https://youtu.be/E4iCz46_oJM)


**ON-FPGA TRAINING WITH ULTRA MEMORY REDUCTION: A LOW- PRECISION TENSOR METHOD** <br />
Kaiqi Zhang; Cole Hawkins; Xiyuan Zhang; Cong Hao; Zheng Zhang<br />
<abstract>Various hardware accelerators have been developed for energy-efficient and real-time inference of neural networks on edge devices. However, most training is done on high-performance GPUs or servers, and the huge memory and computing costs prevent training neural networks on edge devices. This paper proposes a novel tensor-based training framework, which offers orders-of-magnitude memory reduction in the training process. We propose a novel rank-adaptive tensorized neural network model, and design a hardware-friendly low-precision algorithm to train this model. We present an FPGA accelerator to demonstrate the benefits of this training method on edge devices. Our preliminary FPGA implementation achieves 59× speedup and 123× energy reduction compared to embedded CPU, and 292× memory reduction over a standard full-size training.</abstract>

[PDF](https://www.dropbox.com/s/5wuoqe357c3gln0/ON-FPGA%20TRAINING%20WITH%20ULTRA%20MEMORY%20REDUCTION%20A%20LOW-%20PRECISION%20TENSOR%20METHOD%20Paper.pdf?dl=0) &bull;
[Poster](https://www.dropbox.com/s/j95luja04z3euf7/ON-FPGA%20TRAINING%20WITH%20ULTRA%20MEMORY%20REDUCTION%20A%20LOW-%20PRECISION%20TENSOR%20METHOD%20Poster.pdf?dl=0) &bull;
[Presentation](https://youtu.be/Xi4PYSKuGG8)



**MoIL: ENABLING EFFICIENT INCREMENTAL TRAINING ON EDGE DEVICES** <br />
Jiacheng Yang; James Gleeson; Mostafa Elhoushi; Gennady Pekhimenko<br />
<abstract>Edge devices such as smartphones are increasingly becoming the end target hardware for deploying compute heavy deep neural networks (DNN). However, given that mobile platforms have less computational power than cloud servers, edge deployments have been limited to inference workloads, with training being performed in the cloud. Unfortunately, forgoing training on-device prevents tailoring of DNN models to user data, and results in lower accuracy on data seen by the user in-the-wild. In this work, we show that training on-device is possible on today’s mobile platforms by training deployed models incrementally, and that it can be done efficiently and accurately. We demonstrate that three optimizations — global dataset mixing, last layer training, and feature map caching — can be applied to reduce the computational overhead of incremental training from 7 months to 38.8 minutes, while still maintaining high accuracy on both curated datasets (e.g., ImageNet) and local user data observed in-the-wild.</abstract>

[PDF](https://www.dropbox.com/s/is9to7cbt4pzt6i/MoIL%20ENABLING%20EFFICIENT%20INCREMENTAL%20TRAINING%20ON%20EDGE%20DEVICES%20Paper.pdf?dl=0) &bull;
[Poster](https://www.dropbox.com/s/x3txue1l6ydzzif/MoIL%20ENABLING%20EFFICIENT%20INCREMENTAL%20TRAINING%20ON%20EDGE%20DEVICES%20Poster.pdf?dl=0) &bull;
[Presentation](https://youtu.be/5d6P1FFG5Vk)


**HETEROGENEOUS ZERO-SHOT FEDERATED LEARNING WITH NEW CLASSES FOR AUDIO CLASSIFICATION** <br />
Gautham Krishna Gudur (Ericsson)*; Satheesh K Perepu (Ericsson)<br />
<abstract>Federated learning is an effective way of extracting insights from different user devices while preserving the privacy of users. However, new classes with completely unseen data distributions can stream across any device in a federated learning setting, whose data cannot be accessed by the global server or other users. To this end, we propose a unified zero-shot framework to handle these aforementioned challenges during federated learning. We simulate two scenarios here – 1) when the new class labels are not reported by the user, the traditional FL setting is used; 2) when new class labels are reported by the user, we synthesize Anonymized Data Impressions by calculating class similarity matrices corresponding to each device’s new classes followed by unsupervised clustering to distinguish between new classes across different users. Moreover, our proposed framework can also handle statistical heterogeneities in both labels and models across the participating users. We empirically evaluate our framework on-device across different communication rounds (FL iterations) with new classes in both local and global updates, along with heterogeneous labels and models, on two widely used audio classification applications – keyword spotting and urban sound classification, and observe an average deterministic accuracy increase of ∼4.041% and ∼4.258% respectively.</abstract>

[PDF](https://www.dropbox.com/s/qbjw17lc0hnzkd4/HETEROGENEOUS%20ZERO-SHOT%20FEDERATED%20LEARNING%20WITH%20NEW%20CLASSES%20FOR%20AUDIO%20CLASSIFICATION%20Paper.pdf?dl=0) &bull;
[Poster](https://www.dropbox.com/s/rvsqf3n1nbm9guc/HETEROGENEOUS%20ZERO-SHOT%20FEDERATED%20LEARNING%20WITH%20NEW%20CLASSES%20FOR%20AUDIO%20CLASSIFICATION%20Poster.pdf?dl=0) &bull;
[Presentation](https://youtu.be/8MQ8bAXT7eQ)

**EFFICIENT TRAINING UNDER LIMITED RESOURCES** <br />
Mahdi Zolnouri, Dounia Lakhmiri, Christophe Tribes, Eyyub Sari, Sébastien Le Digabel<br />
<abstract>Training time budget and size of the dataset are among the factors affecting the performance of a Deep Neural Network (DNN). This paper shows that Neural Architecture Search (NAS), Hyper Parameters Optimization (HPO), and Data Augmentation help DNNs perform much better while these two factors are limited. However, searching for an optimal architecture and the best hyperparameter values besides a good combination of data augmentation techniques under low resources requires many experiments. We present our approach to achieving such a goal in three steps: reducing training epoch time by compressing the model while maintaining the performance compared to the original model, preventing model overfitting when the dataset is small, and performing the hyperparameter tuning. We used NOMAD, which is a blackbox optimization software based on a derivative-free algorithm to do NAS and HPO. Our work achieved an accuracy of 86.0% on a tiny subset of Mini-ImageNet (Vinyals et al., 2016) at the ICLR 2021 Hardware Aware Efficient Training (HAET) Challenge and won second place in the competition.</abstract>

[PDF](https://www.dropbox.com/s/vtod5u35mimxlab/EFFICIENT%20TRAINING%20UNDER%20LIMITED%20RESOURCES%20Paper.pdf?dl=0) 




