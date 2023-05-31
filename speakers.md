
# Speakers

- [Warren Gross](#warren-gross)
- [Julie Grollier](#julie-grollier)
- [Song Han](#song-han)
- [Ehsan Saboori](#ehsan-saboori)
- [Yunhe Wang](#yunhe-wang)
- [Liangwei Ge](#liangwei-ge)

## Warren Gross

| <img src="/speakers_pictures/warren.jpg" alt="Warren Gross" height="250px" /> |
| **Stochastic Computing for Machine Learning towards an Intelligent Edge** |
| Deep learning offers the promise of intelligent devices that are able to perceive, reason and take intuitive actions. The rising adoption of deep learning techniques has motivated the search for low-cost and high-speed software and hardware solutions for deployment on smart edge devices. In recent years work has focused on reducing the complexity of inference in deep neural networks, however, a method enabling low-complexity on-chip learning is still missing. In this talk we explore the promise of a hardware implementation technique called stochastic computing, which represents values in an algorithm as streams of randomly chosen bits. We show how stochastic computing can be used as a binary or ternary compression technique for both inference and training of deep neural networks. We also show how stochastic computing can implement invertible logic circuits that perform learning without backpropagation. |

| **Affiliation** | [McGill University](https://www.mcgill.ca/ece/warren-gross) |
| **Biography** | Warren J. Gross received the B.A.Sc. degree in Electrical Engineering from the University of Waterloo, Waterloo, ON, Canada, in 1996, and the M.A.Sc. and Ph.D. degrees from the University of Toronto, Toronto, ON, Canada, in 1999 and 2003, respectively. He is currently a Professor, the Louis Ho Faculty Scholar in Technological Innovation,  and the Chair of the Department of Electrical and Computer Engineering, McGill University, Montreal, QC, Canada. His research interests are in the design and implementation of signal processing systems and custom computer architectures. Dr. Gross served as the Chair for the IEEE Signal Processing Society Technical Committee on Design and Implementation of Signal Processing Systems. He served as the General Co-Chair for the IEEE GlobalSIP 2017 and the IEEE SiPS 2017 and the Technical Program Co-Chair for SiPS 2012. He served as an Associate Editor for the IEEE TRANSACTIONS ON SIGNAL PROCESSING and as a Senior Area Editor. He is a Licensed Professional Engineer in the Province of Ontario. | 


## Julie Grollier

| <img src="/speakers_pictures/julie.jpg" alt="Julie Grollier" height="250px" /> |
| **Spiking Equilibrium Propagation for Autonomously Learning Hardware** |
| Current hardware such as CPUs/GPUs is not adapted for efficient training of deep networks: orders of magnitude in speed and energy efficiency are lost due to relentless data transfers between memory and processing. A solution to this issue is to build chips where physical components embodying neurons and synapses are organized in deep networks. Memristor nanodevices are nanoscale resistors whose tunable conductance can mimic and memorize a synaptic weight in-situ. Deep network chips combining transistor-based neurons with memristor-based synapses have been developed but they are only capable of inference. Indeed, backpropagation requires heavy additional circuitry and memories to compute, store and write gradients in memristors. To solve this problem, we have developed a spiking version of equilibrium propagation and designed it to run on circuits with hardware LIF neurons that locally update the memristor conductances through their voltage spikes. Our simulations of the system give a test accuracy on MNIST within 2\% of BPTT. They also show that the system is resilient to imperfections and fast: with GHz oscillators, inference can be achieved in nanoseconds and training in tens of milliseconds. We have thus designed an ultrafast hardware Spiking Neural Network where learning is entirely autonomous and local, with state-of-the-art accuracy. Our work opens the path to novel hardware chips for embedded AI. |

| **Affiliation** | [CNRS, Thales](http://julie.grollier.free.fr) |
| **Biography** | Julie Grollier is researcher director in the CNRS/Thales lab in France, where she is leading the “Nanodevices for Neuromorphic Computing” team. Julie has over 100 publications, and is a frequent invited speaker in international conferences. She is a Fellow of the American Physical Society. In 2010 she was awarded the Jacques Herbrand prize of the French Academy of Science. 2018 she received the Silver Medal of CNRS in Physics for her pioneering work on spintronics and brain-inspired computing. | 

## Song Han

| <img src="/speakers_pictures/song.jpeg" alt="Song Han" height="250px" /> |
| **TinyTL: Reduce Memory, Not Parameters for Efficient On-Device Training** |
| On-device learning enables edge devices to continually adapt the AI models to new data, which requires a small memory footprint to fit the tight memory constraint of edge devices. Existing work solves this problem by reducing the number of trainable parameters. However, this doesn't directly translate to memory saving since the major bottleneck is the activations, not parameters. In this work, we present Tiny-Transfer-Learning (TinyTL) for memory-efficient on-device learning. TinyTL freezes the weights while only learns the bias modules, thus no need to store the intermediate activations. To maintain the adaptation capacity, we introduce a new memory-efficient bias module, the lite residual module, to refine the feature extractor by learning small residual feature maps adding only 3.8% memory overhead. Extensive experiments show that TinyTL significantly saves the memory (up to 6.5x) with little accuracy loss compared to fine-tuning the full network. Compared to fine-tuning the last layer, TinyTL provides significant accuracy improvements (up to 33.8%) with little memory overhead. Furthermore, combined with feature extractor adaptation, TinyTL provides 7.5-12.9x memory saving without sacrificing accuracy compared to fine-tuning the full Inception-V3.|

| **Affiliation** | [MIT EECS](https://songhan.mit.edu) |
| **Biography** | Song Han is an assistant professor in MIT EECS. He received his PhD degree from Stanford University. His research focuses on efficient deep learning computing. He proposed “deep compression” technique that can reduce neural network size by an order of magnitude without losing accuracy, and the hardware implementation “efficient inference engine” that first exploited pruning and weight sparsity in deep learning accelerators. His recent research on neural architecture search and TinyML was highlighted by MIT News, Wired, and Venture Beat, and received many low-power computer vision (LPCV) contest awards. Song received Best Paper awards at ICLR’16 and FPGA’17, Amazon Machine Learning Research Award, SONY Faculty Award, Facebook Faculty Award. Song was named “35 Innovators Under 35” by MIT Technology Review for his contribution on “deep compression” technique that “lets powerful artificial intelligence (AI) programs run more efficiently on low-power mobile devices.” Song received the NSF CAREER Award for “efficient algorithms and hardware for accelerated machine learning.”|



## Ehsan Saboori

| <img src="/speakers_pictures/ehsan.jpg" alt="Ehsan Saboori" height="250px" /> |
| **Deep learning model compression using neural network design space exploration** |
| The emergence of deep neural networks (DNNs) in recent years has enabled ground-breaking   abilities   and   applications   for   modern   intelligent   systems. Concurrently, the increasing complexity and sophistication of DNNs is predicated on significant power consumption, model size and computing resources. These factors have been found to limit deep learning’s performance in real-time applications, in large-scale systems, and on low-power devices. Application developers, software engineers and algorithm architects must now create intelligent solution that deal with strict latency, power and  computation  constraints across an  increasingly diverse set of hardware backends. Deeplite researches and develops novel multi-objective optimization methods aimed towards automated software solutions for neural network design space exploration and network compression that is ideal for various target hardware platforms. Increasingly, the need for end to end, holistic optimization solutions that include training and  inference  considerations, with a particular focus on promising solutions that are user-friendly, intuitive, and enable the practical use of neural networks. We will discuss how recent progress in neural network   design   space   exploration   can   benefit   both   algorithm   developers   and hardware designers alike to reduce the complexity in designing resource-efficient deep learning that consumes less power and fewer barriers to adoption. |

| **Affiliation** | [Deeplite](https://www.deeplite.ai/) |
| **Biography** | Ehsan Saboori is a passionate tech entrepreneur and technologist. He obtained his BSc   in   artificial   intelligence   and   completed   his   PhD   in   computer   science   at Concordia university in 2016. He has been a member of ACM and IEEE associations and published several peer-reviewed conferences and journal papers. With several years of experience working in different companies such as Microsoft, SAP and Morgan Stanley he cofounded Deeplite where he assessed emerging challenges for deep learning and formed the technology vision that became the core of Deeplite. |




## Yunhe Wang

| <img src="/speakers_pictures/YunheWang.jpg" alt="Yunhe Wang.jpg" height="250px" /> |
| **AdderNet: Do we really need multiplications in deep learning?** |
| To reduce the costs for launching deep neural networks, we present a new kind of calculation paradigm that only utilizes additions to establish models with high performance, namely, Adder Neural Network (AdderNet). By exploiting a series of optimization approaches (e.g. better optimizer, knowledge distillation, quantization) for enhancing the resulting performance, AdderNet now can achieve a 93.3% Top-5 accuracy using ResNet-50 on the ImageNet. The FPGA hardware implementation also demonstrates that the energy consumption and circuit areas required by networks can be reduced about 70% using 16bit/8bit AdderNets. We further successfully apply the AdderNet on the image super-resolution to show its superiority. |

| **Affiliation** | [Huawei Noah’s Ark Lab](https://www.wangyunhe.site/) |
| **Biography** | Dr. Yunhe Wang is currently a senior researcher at Huawei Noah’s Ark Lab. He received the Ph.D. degree from Peking University, China. His research interests lie in deep learning algorithms and related applications in computer vision. He has published over 40 papers in prestigious journals and top tier conferences, including IEEE T-PAMI, IEEE T-NNLS, IEEE T-IP, ICML, NIPS, CVPR, ICCV, IJCAI and AAAI. He regularly severed as the (senior) PC member for many conferences, e.g. NIPS, ICML, CVPR, ICCV, IJCAI and AAAI. |


## Liangwei Ge

| <img src="/speakers_pictures/luke.jpeg" alt="Liangwei Ge" height="250px" /> |
| **Deep learning challenges and how Intel is addressing them** |
| The presentation provides a comprehensive summary of the challenges of training deep learning models that we face today and shows how Intel is addressing them. Various technologies, solutions, and best practices that Intel provides to the industry for tackling these challenges are introduced. It covers the challenges at full scale: from cluster, node levels down to core, kernel levels. |

| **Affiliation** | [Intel]() |
| **Biography** | Liangwei Ge is an artificial intelligence (AI) Technical Solution Specialist (TSS) within Intel. His particular focus is on AI workload optimizations, like distributed training across a cluster, processing of large, high-resolution, 3D images, and inferencing at the data center and edge devices. He obtained his PhD in Electronics Design Automation (EDA), Waseda Univ., Japan. |
