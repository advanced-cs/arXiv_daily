# 计算机视觉 cs.CV

- **最新发布 109 篇**

- **更新 56 篇**

## 最新发布

#### [new 001] Analysis of Plant Nutrient Deficiencies Using Multi-Spectral Imaging and Optimized Segmentation Model
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，旨在解决植物叶片营养缺乏的精准检测问题。通过多光谱成像与改进的YOLOv5模型，结合注意力机制，提升对叶片异常症状的识别效果。实验表明，该方法在Dice分数和IoU上比基础模型提升约12%，尤其适用于检测黄化和色素积累等症状。**

- **链接: [http://arxiv.org/pdf/2507.14013v1](http://arxiv.org/pdf/2507.14013v1)**

> **作者:** Ji-Yan Wu; Zheng Yong Poh; Anoop C. Patil; Bongsoo Park; Giovanni Volpe; Daisuke Urano
>
> **摘要:** Accurate detection of nutrient deficiency in plant leaves is essential for precision agriculture, enabling early intervention in fertilization, disease, and stress management. This study presents a deep learning framework for leaf anomaly segmentation using multispectral imaging and an enhanced YOLOv5 model with a transformer-based attention head. The model is tailored for processing nine-channel multispectral input and uses self-attention mechanisms to better capture subtle, spatially-distributed symptoms. The plants in the experiments were grown under controlled nutrient stress conditions for evaluation. We carry out extensive experiments to benchmark the proposed model against the baseline YOLOv5. Extensive experiments show that the proposed model significantly outperforms the baseline YOLOv5, with an average Dice score and IoU (Intersection over Union) improvement of about 12%. In particular, this model is effective in detecting challenging symptoms like chlorosis and pigment accumulation. These results highlight the promise of combining multi-spectral imaging with spectral-spatial feature learning for advancing plant phenotyping and precision agriculture.
>
---
#### [new 002] From Binary to Semantic: Utilizing Large-Scale Binary Occupancy Data for 3D Semantic Occupancy Prediction
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于3D语义占用预测任务，旨在解决依赖昂贵标注LiDAR数据的问题。作者提出一种新框架，利用大规模低成本的二值占用数据，通过预训练和自动标注提升语义预测效果。实验表明该方法在相关任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2507.13387v1](http://arxiv.org/pdf/2507.13387v1)**

> **作者:** Chihiro Noguchi; Takaki Yamamoto
>
> **备注:** Accepted to ICCV Workshop 2025
>
> **摘要:** Accurate perception of the surrounding environment is essential for safe autonomous driving. 3D occupancy prediction, which estimates detailed 3D structures of roads, buildings, and other objects, is particularly important for vision-centric autonomous driving systems that do not rely on LiDAR sensors. However, in 3D semantic occupancy prediction -- where each voxel is assigned a semantic label -- annotated LiDAR point clouds are required, making data acquisition costly. In contrast, large-scale binary occupancy data, which only indicate occupied or free space without semantic labels, can be collected at a lower cost. Despite their availability, the potential of leveraging such data remains unexplored. In this study, we investigate the utilization of large-scale binary occupancy data from two perspectives: (1) pre-training and (2) learning-based auto-labeling. We propose a novel binary occupancy-based framework that decomposes the prediction process into binary and semantic occupancy modules, enabling effective use of binary occupancy data. Our experimental results demonstrate that the proposed framework outperforms existing methods in both pre-training and auto-labeling tasks, highlighting its effectiveness in enhancing 3D semantic occupancy prediction. The code is available at https://github.com/ToyotaInfoTech/b2s-occupancy
>
---
#### [new 003] Transformer-Based Framework for Motion Capture Denoising and Anomaly Detection in Medical Rehabilitation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出了一种基于Transformer的端到端深度学习框架，用于医疗康复中的动作捕捉去噪与异常检测。任务是提升远程康复的准确性和安全性，解决数据噪声、缺失及异常动作识别问题。工作包括模型设计、时序建模与实时检测机制，验证了其在康复数据集上的优越性能。**

- **链接: [http://arxiv.org/pdf/2507.13371v1](http://arxiv.org/pdf/2507.13371v1)**

> **作者:** Yeming Cai; Yang Wang; Zhenglin Li
>
> **摘要:** This paper proposes an end-to-end deep learning framework integrating optical motion capture with a Transformer-based model to enhance medical rehabilitation. It tackles data noise and missing data caused by occlusion and environmental factors, while detecting abnormal movements in real time to ensure patient safety. Utilizing temporal sequence modeling, our framework denoises and completes motion capture data, improving robustness. Evaluations on stroke and orthopedic rehabilitation datasets show superior performance in data reconstruction and anomaly detection, providing a scalable, cost-effective solution for remote rehabilitation with reduced on-site supervision.
>
---
#### [new 004] SuperCM: Improving Semi-Supervised Learning and Domain Adaptation through differentiable clustering
- **分类: cs.CV**

- **简介: 该论文属于半监督学习与领域自适应任务，旨在解决在标签数据有限的情况下提升模型性能的问题。论文提出SuperCM方法，通过引入可微聚类模块，显式利用聚类假设，结合有标签数据计算聚类中心，以端到端方式提升SSL与UDA效果，尤其在低监督场景中表现突出。**

- **链接: [http://arxiv.org/pdf/2507.13779v1](http://arxiv.org/pdf/2507.13779v1)**

> **作者:** Durgesh Singh; Ahcène Boubekki; Robert Jenssen; Michael Kampffmeyer
>
> **摘要:** Semi-Supervised Learning (SSL) and Unsupervised Domain Adaptation (UDA) enhance the model performance by exploiting information from labeled and unlabeled data. The clustering assumption has proven advantageous for learning with limited supervision and states that data points belonging to the same cluster in a high-dimensional space should be assigned to the same category. Recent works have utilized different training mechanisms to implicitly enforce this assumption for the SSL and UDA. In this work, we take a different approach by explicitly involving a differentiable clustering module which is extended to leverage the supervised data to compute its centroids. We demonstrate the effectiveness of our straightforward end-to-end training strategy for SSL and UDA over extensive experiments and highlight its benefits, especially in low supervision regimes, both as a standalone model and as a regularizer for existing approaches.
>
---
#### [new 005] Feature Engineering is Not Dead: Reviving Classical Machine Learning with Entropy, HOG, and LBP Feature Fusion for Image Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像分类任务，旨在解决在强调可解释性和计算效率场景下，传统机器学习方法能否媲美深度学习的问题。作者将排列熵扩展到二维图像，结合HOG和LBP提取多尺度、多方向特征，构建780维特征向量，并用SVM分类器实现高效图像分类。**

- **链接: [http://arxiv.org/pdf/2507.13772v1](http://arxiv.org/pdf/2507.13772v1)**

> **作者:** Abhijit Sen; Giridas Maiti; Bikram K. Parida; Bhanu P. Mishra; Mahima Arya; Denys I. Bondar
>
> **摘要:** Feature engineering continues to play a critical role in image classification, particularly when interpretability and computational efficiency are prioritized over deep learning models with millions of parameters. In this study, we revisit classical machine learning based image classification through a novel approach centered on Permutation Entropy (PE), a robust and computationally lightweight measure traditionally used in time series analysis but rarely applied to image data. We extend PE to two-dimensional images and propose a multiscale, multi-orientation entropy-based feature extraction approach that characterizes spatial order and complexity along rows, columns, diagonals, anti-diagonals, and local patches of the image. To enhance the discriminatory power of the entropy features, we integrate two classic image descriptors: the Histogram of Oriented Gradients (HOG) to capture shape and edge structure, and Local Binary Patterns (LBP) to encode micro-texture of an image. The resulting hand-crafted feature set, comprising of 780 dimensions, is used to train Support Vector Machine (SVM) classifiers optimized through grid search. The proposed approach is evaluated on multiple benchmark datasets, including Fashion-MNIST, KMNIST, EMNIST, and CIFAR-10, where it delivers competitive classification performance without relying on deep architectures. Our results demonstrate that the fusion of PE with HOG and LBP provides a compact, interpretable, and effective alternative to computationally expensive and limited interpretable deep learning models. This shows a potential of entropy-based descriptors in image classification and contributes a lightweight and generalizable solution to interpretable machine learning in image classification and computer vision.
>
---
#### [new 006] COREVQA: A Crowd Observation and Reasoning Entailment Visual Question Answering Benchmark
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言模型（VLM）评估任务，旨在解决现有基准测试中缺乏对视觉蕴含推理能力评估的问题。作者构建了COREVQA数据集，包含5608张拥挤场景图像及对应真/假陈述，用于测试模型对图像信息的推理和判断能力。实验表明，即使是性能最佳的VLMs，在该任务上的准确率也低于80%。**

- **链接: [http://arxiv.org/pdf/2507.13405v1](http://arxiv.org/pdf/2507.13405v1)**

> **作者:** Ishant Chintapatla; Kazuma Choji; Naaisha Agarwal; Andrew Lin; Hannah You; Charles Duong; Kevin Zhu; Sean O'Brien; Vasu Sharma
>
> **摘要:** Recently, many benchmarks and datasets have been developed to evaluate Vision-Language Models (VLMs) using visual question answering (VQA) pairs, and models have shown significant accuracy improvements. However, these benchmarks rarely test the model's ability to accurately complete visual entailment, for instance, accepting or refuting a hypothesis based on the image. To address this, we propose COREVQA (Crowd Observations and Reasoning Entailment), a benchmark of 5608 image and synthetically generated true/false statement pairs, with images derived from the CrowdHuman dataset, to provoke visual entailment reasoning on challenging crowded images. Our results show that even the top-performing VLMs achieve accuracy below 80%, with other models performing substantially worse (39.98%-69.95%). This significant performance gap reveals key limitations in VLMs' ability to reason over certain types of image-question pairs in crowded scenes.
>
---
#### [new 007] Efficient Burst Super-Resolution with One-step Diffusion
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决现有方法生成图像模糊的问题。作者提出了一种高效的突发图像超分辨率方法，结合扩散模型与知识蒸馏，实现快速且高质量的图像重建。**

- **链接: [http://arxiv.org/pdf/2507.13607v1](http://arxiv.org/pdf/2507.13607v1)**

> **作者:** Kento Kawai; Takeru Oba; Kyotaro Tokoro; Kazutoshi Akita; Norimichi Ukita
>
> **备注:** NTIRE2025
>
> **摘要:** While burst Low-Resolution (LR) images are useful for improving their Super Resolution (SR) image compared to a single LR image, prior burst SR methods are trained in a deterministic manner, which produces a blurry SR image. Since such blurry images are perceptually degraded, we aim to reconstruct sharp and high-fidelity SR images by a diffusion model. Our method improves the efficiency of the diffusion model with a stochastic sampler with a high-order ODE as well as one-step diffusion using knowledge distillation. Our experimental results demonstrate that our method can reduce the runtime to 1.6 % of its baseline while maintaining the SR quality measured based on image distortion and perceptual quality.
>
---
#### [new 008] TimeNeRF: Building Generalizable Neural Radiance Fields across Time from Few-Shot Input Views
- **分类: cs.CV; cs.MM**

- **简介: 论文提出TimeNeRF，旨在解决基于少量视角输入构建可泛化的时序神经辐射场问题。该工作属于三维场景建模与渲染任务，通过结合多视角立体、神经辐射场与解耦策略，实现无需逐场景优化即可渲染任意视角和时间的高质量3D场景，尤其擅长表现自然光影变化。**

- **链接: [http://arxiv.org/pdf/2507.13929v1](http://arxiv.org/pdf/2507.13929v1)**

> **作者:** Hsiang-Hui Hung; Huu-Phu Do; Yung-Hui Li; Ching-Chun Huang
>
> **备注:** Accepted by MM 2024
>
> **摘要:** We present TimeNeRF, a generalizable neural rendering approach for rendering novel views at arbitrary viewpoints and at arbitrary times, even with few input views. For real-world applications, it is expensive to collect multiple views and inefficient to re-optimize for unseen scenes. Moreover, as the digital realm, particularly the metaverse, strives for increasingly immersive experiences, the ability to model 3D environments that naturally transition between day and night becomes paramount. While current techniques based on Neural Radiance Fields (NeRF) have shown remarkable proficiency in synthesizing novel views, the exploration of NeRF's potential for temporal 3D scene modeling remains limited, with no dedicated datasets available for this purpose. To this end, our approach harnesses the strengths of multi-view stereo, neural radiance fields, and disentanglement strategies across diverse datasets. This equips our model with the capability for generalizability in a few-shot setting, allows us to construct an implicit content radiance field for scene representation, and further enables the building of neural radiance fields at any arbitrary time. Finally, we synthesize novel views of that time via volume rendering. Experiments show that TimeNeRF can render novel views in a few-shot setting without per-scene optimization. Most notably, it excels in creating realistic novel views that transition smoothly across different times, adeptly capturing intricate natural scene changes from dawn to dusk.
>
---
#### [new 009] CoTasks: Chain-of-Thought based Video Instruction Tuning Tasks
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频理解与推理任务，旨在解决视频大模型缺乏基于细粒度对象理解的链式推理能力问题。作者提出CoTasks框架，将复杂视频问题分解为四个实体级基础任务，通过嵌入中间推理步骤提升模型的时空推理能力。实验表明该方法显著提高了多个子任务的性能。**

- **链接: [http://arxiv.org/pdf/2507.13609v1](http://arxiv.org/pdf/2507.13609v1)**

> **作者:** Yanan Wang; Julio Vizcarra; Zhi Li; Hao Niu; Mori Kurokawa
>
> **摘要:** Despite recent progress in video large language models (VideoLLMs), a key open challenge remains: how to equip models with chain-of-thought (CoT) reasoning abilities grounded in fine-grained object-level video understanding. Existing instruction-tuned models, such as the Qwen and LLaVA series, are trained on high-level video-text pairs, often lacking structured annotations necessary for compositional, step-by-step reasoning. We propose CoTasks: Chain-of-Thought based Video Instruction Tuning Tasks, a new framework that decomposes complex video questions of existing datasets (e.g., NeXT-QA, STAR) into four entity-level foundational tasks: frame localization, entity tracking, spatial and temporal relation extraction. By embedding these intermediate CoT-style reasoning steps into the input, CoTasks enables models to explicitly perform object-centric spatiotemporal reasoning. Experiments on the NeXT-QA benchmark show that CoTasks significantly enhance inference performance: LLaVA-video-7B improves by +3.3 points in average GPT-4 evaluation score, and Qwen2.5-VL-3B gains +17.4, with large boosts in causal (+14.6), temporal (+10.9), and descriptive (+48.1) subcategories. These results demonstrate the effectiveness of CoTasks as a structured CoT-style supervision framework for improving compositional video reasoning.
>
---
#### [new 010] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉问答（VQA）任务，旨在解决用户提问中存在的歧义问题。现有方法多通过改写问题解决歧义，但忽略了交互式澄清的重要性。论文提出了ClearVQA基准，评估模型通过交互澄清歧义的能力，并推动模型从“偏好回答”转向“主动提问”。**

- **链接: [http://arxiv.org/pdf/2507.13773v1](http://arxiv.org/pdf/2507.13773v1)**

> **作者:** Pu Jian; Donglei Yu; Wen Yang; Shuo Ren; Jiajun Zhang
>
> **备注:** ACL2025 Main
>
> **摘要:** In visual question answering (VQA) context, users often pose ambiguous questions to visual language models (VLMs) due to varying expression habits. Existing research addresses such ambiguities primarily by rephrasing questions. These approaches neglect the inherently interactive nature of user interactions with VLMs, where ambiguities can be clarified through user feedback. However, research on interactive clarification faces two major challenges: (1) Benchmarks are absent to assess VLMs' capacity for resolving ambiguities through interaction; (2) VLMs are trained to prefer answering rather than asking, preventing them from seeking clarification. To overcome these challenges, we introduce \textbf{ClearVQA} benchmark, which targets three common categories of ambiguity in VQA context, and encompasses various VQA scenarios.
>
---
#### [new 011] Learning Spectral Diffusion Prior for Hyperspectral Image Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于高光谱图像重建任务，旨在从退化的2D测量数据中恢复3D高光谱图像。现有深度学习方法难以准确重建高分辨率细节，为此作者提出光谱扩散先验（SDP）和光谱先验注入模块（SPIM），利用扩散模型提升重建质量。实验表明该方法在MST和BISRNet上均有效提升重建性能。**

- **链接: [http://arxiv.org/pdf/2507.13769v1](http://arxiv.org/pdf/2507.13769v1)**

> **作者:** Mingyang Yu; Zhijian Wu; Dingjiang Huang
>
> **摘要:** Hyperspectral image (HSI) reconstruction aims to recover 3D HSI from its degraded 2D measurements. Recently great progress has been made in deep learning-based methods, however, these methods often struggle to accurately capture high-frequency details of the HSI. To address this issue, this paper proposes a Spectral Diffusion Prior (SDP) that is implicitly learned from hyperspectral images using a diffusion model. Leveraging the powerful ability of the diffusion model to reconstruct details, this learned prior can significantly improve the performance when injected into the HSI model. To further improve the effectiveness of the learned prior, we also propose the Spectral Prior Injector Module (SPIM) to dynamically guide the model to recover the HSI details. We evaluate our method on two representative HSI methods: MST and BISRNet. Experimental results show that our method outperforms existing networks by about 0.5 dB, effectively improving the performance of HSI reconstruction.
>
---
#### [new 012] Using Multiple Input Modalities Can Improve Data-Efficiency and O.O.D. Generalization for ML with Satellite Imagery
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于遥感图像机器学习任务，旨在提升模型的数据效率和地理分布外泛化能力。论文通过融合多种地理数据层（如高程、气象数据）与光学卫星图像，验证多模输入对模型性能的提升效果，尤其在标注数据有限和地理分布外场景下效果显著。**

- **链接: [http://arxiv.org/pdf/2507.13385v1](http://arxiv.org/pdf/2507.13385v1)**

> **作者:** Arjun Rao; Esther Rolf
>
> **备注:** 17 pages, 9 figures, 7 tables. Accepted to TerraBytes@ICML 2025
>
> **摘要:** A large variety of geospatial data layers is available around the world ranging from remotely-sensed raster data like satellite imagery, digital elevation models, predicted land cover maps, and human-annotated data, to data derived from environmental sensors such as air temperature or wind speed data. A large majority of machine learning models trained on satellite imagery (SatML), however, are designed primarily for optical input modalities such as multi-spectral satellite imagery. To better understand the value of using other input modalities alongside optical imagery in supervised learning settings, we generate augmented versions of SatML benchmark tasks by appending additional geographic data layers to datasets spanning classification, regression, and segmentation. Using these augmented datasets, we find that fusing additional geographic inputs with optical imagery can significantly improve SatML model performance. Benefits are largest in settings where labeled data are limited and in geographic out-of-sample settings, suggesting that multi-modal inputs may be especially valuable for data-efficiency and out-of-sample performance of SatML models. Surprisingly, we find that hard-coded fusion strategies outperform learned variants, with interesting implications for future work.
>
---
#### [new 013] Unmasking Performance Gaps: A Comparative Study of Human Anonymization and Its Effects on Video Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于视频异常检测任务，旨在解决隐私保护与检测性能之间的权衡问题。作者在UCF-Crime数据集上评估四种匿名化技术对四种检测方法的影响，发现检测性能与算法设计和学习策略密切相关。实验表明，某些匿名化方法下模型性能反而提升，揭示了隐私保护与检测效用之间的复杂关系。**

- **链接: [http://arxiv.org/pdf/2507.14083v1](http://arxiv.org/pdf/2507.14083v1)**

> **作者:** Sara Abdulaziz; Egor Bondarev
>
> **备注:** ACIVS 2025
>
> **摘要:** Advancements in deep learning have improved anomaly detection in surveillance videos, yet they raise urgent privacy concerns due to the collection of sensitive human data. In this paper, we present a comprehensive analysis of anomaly detection performance under four human anonymization techniques, including blurring, masking, encryption, and avatar replacement, applied to the UCF-Crime dataset. We evaluate four anomaly detection methods, MGFN, UR-DMU, BN-WVAD, and PEL4VAD, on the anonymized UCF-Crime to reveal how each method responds to different obfuscation techniques. Experimental results demonstrate that anomaly detection remains viable under anonymized data and is dependent on the algorithmic design and the learning strategy. For instance, under certain anonymization patterns, such as encryption and masking, some models inadvertently achieve higher AUC performance compared to raw data, due to the strong responsiveness of their algorithmic components to these noise patterns. These results highlight the algorithm-specific sensitivities to anonymization and emphasize the trade-off between preserving privacy and maintaining detection utility. Furthermore, we compare these conventional anonymization techniques with the emerging privacy-by-design solutions, highlighting an often overlooked trade-off between robust privacy protection and utility flexibility. Through comprehensive experiments and analyses, this study provides a compelling benchmark and insights into balancing human privacy with the demands of anomaly detection.
>
---
#### [new 014] Evaluation of Human Visual Privacy Protection: A Three-Dimensional Framework and Benchmark Dataset
- **分类: cs.CV**

- **简介: 该论文属于视觉隐私保护评估任务，旨在解决如何客观评价隐私保护效果的问题。论文提出了一种包含隐私、效用与实用性三个维度的评估框架，并构建了包含多种标签的人类视觉隐私数据集HR-VISPR，用于训练可解释的隐私度量模型。研究通过评估11种隐私保护方法，验证了框架在平衡隐私保护与实用性方面的有效性。**

- **链接: [http://arxiv.org/pdf/2507.13981v1](http://arxiv.org/pdf/2507.13981v1)**

> **作者:** Sara Abdulaziz; Giacomo D'Amicantonio; Egor Bondarev
>
> **备注:** accepted at ICCV'25 workshop CV4BIOM
>
> **摘要:** Recent advances in AI-powered surveillance have intensified concerns over the collection and processing of sensitive personal data. In response, research has increasingly focused on privacy-by-design solutions, raising the need for objective techniques to evaluate privacy protection. This paper presents a comprehensive framework for evaluating visual privacy-protection methods across three dimensions: privacy, utility, and practicality. In addition, it introduces HR-VISPR, a publicly available human-centric dataset with biometric, soft-biometric, and non-biometric labels to train an interpretable privacy metric. We evaluate 11 privacy protection methods, ranging from conventional techniques to advanced deep-learning methods, through the proposed framework. The framework differentiates privacy levels in alignment with human visual perception, while highlighting trade-offs between privacy, utility, and practicality. This study, along with the HR-VISPR dataset, serves as an insightful tool and offers a structured evaluation framework applicable across diverse contexts.
>
---
#### [new 015] "PhyWorldBench": A Comprehensive Evaluation of Physical Realism in Text-to-Video Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成模型评估任务，旨在解决模型在物理真实性方面的不足。作者构建了PhyWorldBench基准，涵盖多种物理现象及“反物理”场景，通过12个主流模型测试，分析其在物理一致性上的表现并提出改进建议。**

- **链接: [http://arxiv.org/pdf/2507.13428v1](http://arxiv.org/pdf/2507.13428v1)**

> **作者:** Jing Gu; Xian Liu; Yu Zeng; Ashwin Nagarajan; Fangrui Zhu; Daniel Hong; Yue Fan; Qianqi Yan; Kaiwen Zhou; Ming-Yu Liu; Xin Eric Wang
>
> **备注:** 31 pages, 21 figures
>
> **摘要:** Video generation models have achieved remarkable progress in creating high-quality, photorealistic content. However, their ability to accurately simulate physical phenomena remains a critical and unresolved challenge. This paper presents PhyWorldBench, a comprehensive benchmark designed to evaluate video generation models based on their adherence to the laws of physics. The benchmark covers multiple levels of physical phenomena, ranging from fundamental principles like object motion and energy conservation to more complex scenarios involving rigid body interactions and human or animal motion. Additionally, we introduce a novel ""Anti-Physics"" category, where prompts intentionally violate real-world physics, enabling the assessment of whether models can follow such instructions while maintaining logical consistency. Besides large-scale human evaluation, we also design a simple yet effective method that could utilize current MLLM to evaluate the physics realism in a zero-shot fashion. We evaluate 12 state-of-the-art text-to-video generation models, including five open-source and five proprietary models, with a detailed comparison and analysis. we identify pivotal challenges models face in adhering to real-world physics. Through systematic testing of their outputs across 1,050 curated prompts-spanning fundamental, composite, and anti-physics scenarios-we identify pivotal challenges these models face in adhering to real-world physics. We then rigorously examine their performance on diverse physical phenomena with varying prompt types, deriving targeted recommendations for crafting prompts that enhance fidelity to physical principles.
>
---
#### [new 016] Low-Light Enhancement via Encoder-Decoder Network with Illumination Guidance
- **分类: cs.CV**

- **简介: 该论文属于图像增强任务，旨在解决低光图像质量差的问题。作者提出EDNIG框架，结合U-Net与亮度引导图（来自BCP），并引入空间金字塔池化和Swish激活函数，通过GAN框架优化，有效提升低光图像质量。**

- **链接: [http://arxiv.org/pdf/2507.13360v1](http://arxiv.org/pdf/2507.13360v1)**

> **作者:** Le-Anh Tran; Chung Nguyen Tran; Ngoc-Luu Nguyen; Nhan Cach Dang; Jordi Carrabina; David Castells-Rufas; Minh Son Nguyen
>
> **备注:** 6 pages, 3 figures, ICCCE 2025
>
> **摘要:** This paper introduces a novel deep learning framework for low-light image enhancement, named the Encoder-Decoder Network with Illumination Guidance (EDNIG). Building upon the U-Net architecture, EDNIG integrates an illumination map, derived from Bright Channel Prior (BCP), as a guidance input. This illumination guidance helps the network focus on underexposed regions, effectively steering the enhancement process. To further improve the model's representational power, a Spatial Pyramid Pooling (SPP) module is incorporated to extract multi-scale contextual features, enabling better handling of diverse lighting conditions. Additionally, the Swish activation function is employed to ensure smoother gradient propagation during training. EDNIG is optimized within a Generative Adversarial Network (GAN) framework using a composite loss function that combines adversarial loss, pixel-wise mean squared error (MSE), and perceptual loss. Experimental results show that EDNIG achieves competitive performance compared to state-of-the-art methods in quantitative metrics and visual quality, while maintaining lower model complexity, demonstrating its suitability for real-world applications. The source code for this work is available at https://github.com/tranleanh/ednig.
>
---
#### [new 017] Multi-Centre Validation of a Deep Learning Model for Scoliosis Assessment
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学影像分析任务，旨在解决脊柱侧弯评估中人工测量Cobb角耗时且存在观察者差异的问题。研究验证了一种全自动深度学习软件（Carebot AI Bones）在10家医院的103例全脊柱X光片上的表现，结果显示其测量精度接近专家水平，具备临床应用潜力。**

- **链接: [http://arxiv.org/pdf/2507.14093v1](http://arxiv.org/pdf/2507.14093v1)**

> **作者:** Šimon Kubov; Simon Klíčník; Jakub Dandár; Zdeněk Straka; Karolína Kvaková; Daniel Kvak
>
> **摘要:** Scoliosis affects roughly 2 to 4 percent of adolescents, and treatment decisions depend on precise Cobb angle measurement. Manual assessment is time consuming and subject to inter observer variation. We conducted a retrospective, multi centre evaluation of a fully automated deep learning software (Carebot AI Bones, Spine Measurement functionality; Carebot s.r.o.) on 103 standing anteroposterior whole spine radiographs collected from ten hospitals. Two musculoskeletal radiologists independently measured each study and served as reference readers. Agreement between the AI and each radiologist was assessed with Bland Altman analysis, mean absolute error (MAE), root mean squared error (RMSE), Pearson correlation coefficient, and Cohen kappa for four grade severity classification. Against Radiologist 1 the AI achieved an MAE of 3.89 degrees (RMSE 4.77 degrees) with a bias of 0.70 degrees and limits of agreement from minus 8.59 to plus 9.99 degrees. Against Radiologist 2 the AI achieved an MAE of 3.90 degrees (RMSE 5.68 degrees) with a bias of 2.14 degrees and limits from minus 8.23 to plus 12.50 degrees. Pearson correlations were r equals 0.906 and r equals 0.880 (inter reader r equals 0.928), while Cohen kappa for severity grading reached 0.51 and 0.64 (inter reader kappa 0.59). These results demonstrate that the proposed software reproduces expert level Cobb angle measurements and categorical grading across multiple centres, suggesting its utility for streamlining scoliosis reporting and triage in clinical workflows.
>
---
#### [new 018] A Quantum-assisted Attention U-Net for Building Segmentation over Tunis using Sentinel-1 Data
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于遥感图像处理任务，旨在解决突尼斯城市区域的建筑物分割问题。为应对高分辨率卫星图像带来的挑战，研究引入量子卷积（Quanvolution）预处理，提升注意力U-Net模型的特征提取能力。实验表明，该方法在保持分割精度的同时显著减少网络参数，提升了计算效率，展示了量子辅助深度学习在城市建筑物大规模分割中的潜力。**

- **链接: [http://arxiv.org/pdf/2507.13852v1](http://arxiv.org/pdf/2507.13852v1)**

> **作者:** Luigi Russo; Francesco Mauro; Babak Memar; Alessandro Sebastianelli; Silvia Liberata Ullo; Paolo Gamba
>
> **备注:** Accepted at IEEE Joint Urban Remote Sensing Event (JURSE) 2025
>
> **摘要:** Building segmentation in urban areas is essential in fields such as urban planning, disaster response, and population mapping. Yet accurately segmenting buildings in dense urban regions presents challenges due to the large size and high resolution of satellite images. This study investigates the use of a Quanvolutional pre-processing to enhance the capability of the Attention U-Net model in the building segmentation. Specifically, this paper focuses on the urban landscape of Tunis, utilizing Sentinel-1 Synthetic Aperture Radar (SAR) imagery. In this work, Quanvolution was used to extract more informative feature maps that capture essential structural details in radar imagery, proving beneficial for accurate building segmentation. Preliminary results indicate that proposed methodology achieves comparable test accuracy to the standard Attention U-Net model while significantly reducing network parameters. This result aligns with findings from previous works, confirming that Quanvolution not only maintains model accuracy but also increases computational efficiency. These promising outcomes highlight the potential of quantum-assisted Deep Learning frameworks for large-scale building segmentation in urban environments.
>
---
#### [new 019] VLMs have Tunnel Vision: Evaluating Nonlocal Visual Reasoning in Leading VLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型评估任务，旨在解决当前VLMs在非局部视觉推理上的表现问题。论文设计了三种非局部视觉推理任务，发现主流模型如Gemini、GPT等在这些任务上表现不佳，远低于人类水平，揭示了它们缺乏核心视觉推理能力。**

- **链接: [http://arxiv.org/pdf/2507.13361v1](http://arxiv.org/pdf/2507.13361v1)**

> **作者:** Shmuel Berman; Jia Deng
>
> **摘要:** Visual Language Models (VLMs) excel at complex visual tasks such as VQA and chart understanding, yet recent work suggests they struggle with simple perceptual tests. We present an evaluation that tests vision-language models' capacity for nonlocal visual reasoning -- reasoning that requires chaining evidence collected from multiple, possibly distant, regions of an image. We isolate three distinct forms of non-local vision: comparative perception, which demands holding two images in working memory and comparing them; saccadic search, which requires making discrete, evidence-driven jumps to locate successive targets; and smooth visual search, which involves searching smoothly along a continuous contour. Flagship models (e.g., Gemini 2.5 Pro, Claude Vision 3.7, GPT-o4-mini), even those that perform well on prior primitive-vision benchmarks, fail these tests and barely exceed random accuracy on two variants of our tasks that are trivial for humans. Our structured evaluation suite allows us to test if VLMs can perform similar visual algorithms to humans. Our findings show that despite gains in raw visual acuity, current models lack core visual reasoning capabilities.
>
---
#### [new 020] Real-Time Fusion of Visual and Chart Data for Enhanced Maritime Vision
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像与数据融合任务，旨在提升海上视觉导航。通过实时融合视觉数据与海图信息，解决动态环境中导航标志（如浮标）的准确定位与匹配问题。提出基于Transformer的端到端网络，实现图像检测与海图标记的直接匹配，提升关联精度。**

- **链接: [http://arxiv.org/pdf/2507.13880v1](http://arxiv.org/pdf/2507.13880v1)**

> **作者:** Marten Kreis; Benjamin Kiefer
>
> **摘要:** This paper presents a novel approach to enhancing marine vision by fusing real-time visual data with chart information. Our system overlays nautical chart data onto live video feeds by accurately matching detected navigational aids, such as buoys, with their corresponding representations in chart data. To achieve robust association, we introduce a transformer-based end-to-end neural network that predicts bounding boxes and confidence scores for buoy queries, enabling the direct matching of image-domain detections with world-space chart markers. The proposed method is compared against baseline approaches, including a ray-casting model that estimates buoy positions via camera projection and a YOLOv7-based network extended with a distance estimation module. Experimental results on a dataset of real-world maritime scenes demonstrate that our approach significantly improves object localization and association accuracy in dynamic and challenging environments.
>
---
#### [new 021] Minimalist Concept Erasure in Generative Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于生成模型安全与可控性任务，旨在解决生成模型因依赖大规模未标注数据带来的安全与版权问题。论文提出了一种极简的概念擦除方法，仅基于生成结果的分布距离构建目标函数，并通过端到端的可微优化实现有效擦除。此外，引入神经元掩码提升擦除鲁棒性，避免过度修改模型结构。实验表明该方法可在不损害模型性能的前提下稳健擦除特定概念，提升生成模型的安全性与可控性。**

- **链接: [http://arxiv.org/pdf/2507.13386v1](http://arxiv.org/pdf/2507.13386v1)**

> **作者:** Yang Zhang; Er Jin; Yanfei Dong; Yixuan Wu; Philip Torr; Ashkan Khakzar; Johannes Stegmaier; Kenji Kawaguchi
>
> **备注:** ICML2025
>
> **摘要:** Recent advances in generative models have demonstrated remarkable capabilities in producing high-quality images, but their reliance on large-scale unlabeled data has raised significant safety and copyright concerns. Efforts to address these issues by erasing unwanted concepts have shown promise. However, many existing erasure methods involve excessive modifications that compromise the overall utility of the model. In this work, we address these issues by formulating a novel minimalist concept erasure objective based \emph{only} on the distributional distance of final generation outputs. Building on our formulation, we derive a tractable loss for differentiable optimization that leverages backpropagation through all generation steps in an end-to-end manner. We also conduct extensive analysis to show theoretical connections with other models and methods. To improve the robustness of the erasure, we incorporate neuron masking as an alternative to model fine-tuning. Empirical evaluations on state-of-the-art flow-matching models demonstrate that our method robustly erases concepts without degrading overall model performance, paving the way for safer and more responsible generative models.
>
---
#### [new 022] Training-free Token Reduction for Vision Mamba
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉与模型压缩任务，旨在解决Vision Mamba模型计算效率低的问题。现有ViT的token压缩方法在Mamba上效果差，因其依赖注意力机制。论文提出无需训练的MTR方法，基于Mamba结构评估token重要性，实现高效压缩。实验表明MTR显著减少计算量且性能损失小。**

- **链接: [http://arxiv.org/pdf/2507.14042v1](http://arxiv.org/pdf/2507.14042v1)**

> **作者:** Qiankun Ma; Ziyao Zhang; Chi Su; Jie Chen; Zhen Song; Hairong Zheng; Wen Gao
>
> **摘要:** Vision Mamba has emerged as a strong competitor to Vision Transformers (ViTs) due to its ability to efficiently capture long-range dependencies with linear computational complexity. While token reduction, an effective compression technique in ViTs, has rarely been explored in Vision Mamba. Exploring Vision Mamba's efficiency is essential for enabling broader applications. However, we find that directly applying existing token reduction techniques for ViTs to Vision Mamba leads to significant performance degradation. This is primarily because Mamba is a sequence model without attention mechanisms, whereas most token reduction techniques for ViTs rely on attention mechanisms for importance measurement and overlook the order of compressed tokens. In this paper, we investigate a Mamba structure-aware importance score to evaluate token importance in a simple and effective manner. Building on this score, we further propose MTR, a training-free \textbf{M}amba \textbf{T}oken \textbf{R}eduction framework. Without the need for training or additional tuning parameters, our method can be seamlessly integrated as a plug-and-play component across various Mamba models. Extensive experiments demonstrate that our approach significantly reduces computational workload while minimizing performance impact across various tasks and multiple backbones. Notably, MTR reduces FLOPs by approximately 40\% on the Vim-B backbone, with only a 1.6\% drop in ImageNet performance without retraining.
>
---
#### [new 023] Localized FNO for Spatiotemporal Hemodynamic Upsampling in Aneurysm MRI
- **分类: cs.CV; cs.AI; physics.comp-ph**

- **简介: 该论文属于医学影像分析任务，旨在解决脑动脉瘤MRI血流数据分辨率低的问题。作者提出了LoFNO模型，结合几何先验与神经算子框架，实现对血流速度和壁面切应力的超分辨率重建，提升诊断精度。**

- **链接: [http://arxiv.org/pdf/2507.13789v1](http://arxiv.org/pdf/2507.13789v1)**

> **作者:** Kyriakos Flouris; Moritz Halter; Yolanne Y. R. Lee; Samuel Castonguay; Luuk Jacobs; Pietro Dirix; Jonathan Nestmann; Sebastian Kozerke; Ender Konukoglu
>
> **摘要:** Hemodynamic analysis is essential for predicting aneurysm rupture and guiding treatment. While magnetic resonance flow imaging enables time-resolved volumetric blood velocity measurements, its low spatiotemporal resolution and signal-to-noise ratio limit its diagnostic utility. To address this, we propose the Localized Fourier Neural Operator (LoFNO), a novel 3D architecture that enhances both spatial and temporal resolution with the ability to predict wall shear stress (WSS) directly from clinical imaging data. LoFNO integrates Laplacian eigenvectors as geometric priors for improved structural awareness on irregular, unseen geometries and employs an Enhanced Deep Super-Resolution Network (EDSR) layer for robust upsampling. By combining geometric priors with neural operator frameworks, LoFNO de-noises and spatiotemporally upsamples flow data, achieving superior velocity and WSS predictions compared to interpolation and alternative deep learning methods, enabling more precise cerebrovascular diagnostics.
>
---
#### [new 024] When Person Re-Identification Meets Event Camera: A Benchmark Dataset and An Attribute-guided Re-Identification Framework
- **分类: cs.CV; cs.AI; cs.LG; cs.NE**

- **简介: 该论文属于行人重识别（ReID）任务，旨在解决事件相机数据在ReID中缺乏大规模真实数据的问题。作者构建了包含118,988图像对的EvReID数据集，并提出了属性引导的TriPro-ReID框架，提升特征学习效果，验证了RGB-Event融合方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.13659v1](http://arxiv.org/pdf/2507.13659v1)**

> **作者:** Xiao Wang; Qian Zhu; Shujuan Wu; Bo Jiang; Shiliang Zhang; Yaowei Wang; Yonghong Tian; Bin Luo
>
> **摘要:** Recent researchers have proposed using event cameras for person re-identification (ReID) due to their promising performance and better balance in terms of privacy protection, event camera-based person ReID has attracted significant attention. Currently, mainstream event-based person ReID algorithms primarily focus on fusing visible light and event stream, as well as preserving privacy. Although significant progress has been made, these methods are typically trained and evaluated on small-scale or simulated event camera datasets, making it difficult to assess their real identification performance and generalization ability. To address the issue of data scarcity, this paper introduces a large-scale RGB-event based person ReID dataset, called EvReID. The dataset contains 118,988 image pairs and covers 1200 pedestrian identities, with data collected across multiple seasons, scenes, and lighting conditions. We also evaluate 15 state-of-the-art person ReID algorithms, laying a solid foundation for future research in terms of both data and benchmarking. Based on our newly constructed dataset, this paper further proposes a pedestrian attribute-guided contrastive learning framework to enhance feature learning for person re-identification, termed TriPro-ReID. This framework not only effectively explores the visual features from both RGB frames and event streams, but also fully utilizes pedestrian attributes as mid-level semantic features. Extensive experiments on the EvReID dataset and MARS datasets fully validated the effectiveness of our proposed RGB-Event person ReID framework. The benchmark dataset and source code will be released on https://github.com/Event-AHU/Neuromorphic_ReID
>
---
#### [new 025] HeCoFuse: Cross-Modal Complementary V2X Cooperative Perception with Heterogeneous Sensors
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **简介: 该论文属于自动驾驶中的车路协同感知任务，旨在解决多车多模态传感器（相机、激光雷达）异构配置下的特征融合与感知可靠性问题。论文提出了HeCoFuse框架，通过分层融合机制与注意力策略，实现跨模态特征对齐与质量平衡，并在真实数据集上验证了其优越性能。**

- **链接: [http://arxiv.org/pdf/2507.13677v1](http://arxiv.org/pdf/2507.13677v1)**

> **作者:** Chuheng Wei; Ziye Qin; Walter Zimmer; Guoyuan Wu; Matthew J. Barth
>
> **备注:** Ranked first in CVPR DriveX workshop TUM-Traf V2X challenge. Accepted by ITSC2025
>
> **摘要:** Real-world Vehicle-to-Everything (V2X) cooperative perception systems often operate under heterogeneous sensor configurations due to cost constraints and deployment variability across vehicles and infrastructure. This heterogeneity poses significant challenges for feature fusion and perception reliability. To address these issues, we propose HeCoFuse, a unified framework designed for cooperative perception across mixed sensor setups where nodes may carry Cameras (C), LiDARs (L), or both. By introducing a hierarchical fusion mechanism that adaptively weights features through a combination of channel-wise and spatial attention, HeCoFuse can tackle critical challenges such as cross-modality feature misalignment and imbalanced representation quality. In addition, an adaptive spatial resolution adjustment module is employed to balance computational cost and fusion effectiveness. To enhance robustness across different configurations, we further implement a cooperative learning strategy that dynamically adjusts fusion type based on available modalities. Experiments on the real-world TUMTraf-V2X dataset demonstrate that HeCoFuse achieves 43.22% 3D mAP under the full sensor configuration (LC+LC), outperforming the CoopDet3D baseline by 1.17%, and reaches an even higher 43.38% 3D mAP in the L+LC scenario, while maintaining 3D mAP in the range of 21.74% to 43.38% across nine heterogeneous sensor configurations. These results, validated by our first-place finish in the CVPR 2025 DriveX challenge, establish HeCoFuse as the current state-of-the-art on TUM-Traf V2X dataset while demonstrating robust performance across diverse sensor deployments.
>
---
#### [new 026] UL-DD: A Multimodal Drowsiness Dataset Using Video, Biometric Signals, and Behavioral Data
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于驾驶员疲劳检测任务，旨在解决现有数据集缺乏多模态连续性数据的问题。作者构建了一个包含3D面部视频、生物信号、行为数据的公开疲劳数据集UL-DD，记录了19名受试者在清醒与疲劳状态下的多模态信息，持续时间达1400分钟，提供更全面的驾驶疲劳分析依据。**

- **链接: [http://arxiv.org/pdf/2507.13403v1](http://arxiv.org/pdf/2507.13403v1)**

> **作者:** Morteza Bodaghi; Majid Hosseini; Raju Gottumukkala; Ravi Teja Bhupatiraju; Iftikhar Ahmad; Moncef Gabbouj
>
> **摘要:** In this study, we present a comprehensive public dataset for driver drowsiness detection, integrating multimodal signals of facial, behavioral, and biometric indicators. Our dataset includes 3D facial video using a depth camera, IR camera footage, posterior videos, and biometric signals such as heart rate, electrodermal activity, blood oxygen saturation, skin temperature, and accelerometer data. This data set provides grip sensor data from the steering wheel and telemetry data from the American truck simulator game to provide more information about drivers' behavior while they are alert and drowsy. Drowsiness levels were self-reported every four minutes using the Karolinska Sleepiness Scale (KSS). The simulation environment consists of three monitor setups, and the driving condition is completely like a car. Data were collected from 19 subjects (15 M, 4 F) in two conditions: when they were fully alert and when they exhibited signs of sleepiness. Unlike other datasets, our multimodal dataset has a continuous duration of 40 minutes for each data collection session per subject, contributing to a total length of 1,400 minutes, and we recorded gradual changes in the driver state rather than discrete alert/drowsy labels. This study aims to create a comprehensive multimodal dataset of driver drowsiness that captures a wider range of physiological, behavioral, and driving-related signals. The dataset will be available upon request to the corresponding author.
>
---
#### [new 027] Encapsulated Composition of Text-to-Image and Text-to-Video Models for High-Quality Video Synthesis
- **分类: cs.CV**

- **简介: 该论文属于文本生成视频任务，旨在解决现有文本到视频模型在画质与运动连贯性上的不足。论文提出EVS方法，结合文本到图像与文本到视频模型，通过去噪优化提升画质，同时保持运动一致性，实现更高质量的视频生成。**

- **链接: [http://arxiv.org/pdf/2507.13753v1](http://arxiv.org/pdf/2507.13753v1)**

> **作者:** Tongtong Su; Chengyu Wang; Bingyan Liu; Jun Huang; Dongming Lu
>
> **摘要:** In recent years, large text-to-video (T2V) synthesis models have garnered considerable attention for their abilities to generate videos from textual descriptions. However, achieving both high imaging quality and effective motion representation remains a significant challenge for these T2V models. Existing approaches often adapt pre-trained text-to-image (T2I) models to refine video frames, leading to issues such as flickering and artifacts due to inconsistencies across frames. In this paper, we introduce EVS, a training-free Encapsulated Video Synthesizer that composes T2I and T2V models to enhance both visual fidelity and motion smoothness of generated videos. Our approach utilizes a well-trained diffusion-based T2I model to refine low-quality video frames by treating them as out-of-distribution samples, effectively optimizing them with noising and denoising steps. Meanwhile, we employ T2V backbones to ensure consistent motion dynamics. By encapsulating the T2V temporal-only prior into the T2I generation process, EVS successfully leverages the strengths of both types of models, resulting in videos of improved imaging and motion quality. Experimental results validate the effectiveness of our approach compared to previous approaches. Our composition process also leads to a significant improvement of 1.6x-4.5x speedup in inference time. Source codes: https://github.com/Tonniia/EVS.
>
---
#### [new 028] CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉生成任务，旨在解决单图内容与风格解耦问题。作者提出CSD-VAR方法，通过尺度感知优化、SVD修正和增强记忆机制，在VAR模型中实现更优的内容保留与风格迁移，并构建了CSD-100数据集进行评估。**

- **链接: [http://arxiv.org/pdf/2507.13984v1](http://arxiv.org/pdf/2507.13984v1)**

> **作者:** Quang-Binh Nguyen; Minh Luu; Quang Nguyen; Anh Tran; Khoi Nguyen
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Disentangling content and style from a single image, known as content-style decomposition (CSD), enables recontextualization of extracted content and stylization of extracted styles, offering greater creative flexibility in visual synthesis. While recent personalization methods have explored the decomposition of explicit content style, they remain tailored for diffusion models. Meanwhile, Visual Autoregressive Modeling (VAR) has emerged as a promising alternative with a next-scale prediction paradigm, achieving performance comparable to that of diffusion models. In this paper, we explore VAR as a generative framework for CSD, leveraging its scale-wise generation process for improved disentanglement. To this end, we propose CSD-VAR, a novel method that introduces three key innovations: (1) a scale-aware alternating optimization strategy that aligns content and style representation with their respective scales to enhance separation, (2) an SVD-based rectification method to mitigate content leakage into style representations, and (3) an Augmented Key-Value (K-V) memory enhancing content identity preservation. To benchmark this task, we introduce CSD-100, a dataset specifically designed for content-style decomposition, featuring diverse subjects rendered in various artistic styles. Experiments demonstrate that CSD-VAR outperforms prior approaches, achieving superior content preservation and stylization fidelity.
>
---
#### [new 029] AortaDiff: Volume-Guided Conditional Diffusion Models for Multi-Branch Aortic Surface Generation
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理与建模任务，旨在解决传统方法在生成高质量、几何一致的三维主动脉模型上的不足。论文提出了AortaDiff，一种基于扩散模型的框架，可从CT/MRI体积数据中生成平滑且适合计算流体动力学（CFD）分析的主动脉表面模型。**

- **链接: [http://arxiv.org/pdf/2507.13404v1](http://arxiv.org/pdf/2507.13404v1)**

> **作者:** Delin An; Pan Du; Jian-Xun Wang; Chaoli Wang
>
> **摘要:** Accurate 3D aortic construction is crucial for clinical diagnosis, preoperative planning, and computational fluid dynamics (CFD) simulations, as it enables the estimation of critical hemodynamic parameters such as blood flow velocity, pressure distribution, and wall shear stress. Existing construction methods often rely on large annotated training datasets and extensive manual intervention. While the resulting meshes can serve for visualization purposes, they struggle to produce geometrically consistent, well-constructed surfaces suitable for downstream CFD analysis. To address these challenges, we introduce AortaDiff, a diffusion-based framework that generates smooth aortic surfaces directly from CT/MRI volumes. AortaDiff first employs a volume-guided conditional diffusion model (CDM) to iteratively generate aortic centerlines conditioned on volumetric medical images. Each centerline point is then automatically used as a prompt to extract the corresponding vessel contour, ensuring accurate boundary delineation. Finally, the extracted contours are fitted into a smooth 3D surface, yielding a continuous, CFD-compatible mesh representation. AortaDiff offers distinct advantages over existing methods, including an end-to-end workflow, minimal dependency on large labeled datasets, and the ability to generate CFD-compatible aorta meshes with high geometric fidelity. Experimental results demonstrate that AortaDiff performs effectively even with limited training data, successfully constructing both normal and pathologically altered aorta meshes, including cases with aneurysms or coarctation. This capability enables the generation of high-quality visualizations and positions AortaDiff as a practical solution for cardiovascular research.
>
---
#### [new 030] PositionIC: Unified Position and Identity Consistency for Image Customization
- **分类: cs.CV**

- **简介: 该论文属于图像定制任务，旨在解决多主体场景中实体级空间控制不足的问题。作者提出PositionIC框架，通过双向生成范式和位置调制层，实现身份与位置一致性，提升定制图像的空间控制精度与视觉质量。**

- **链接: [http://arxiv.org/pdf/2507.13861v1](http://arxiv.org/pdf/2507.13861v1)**

> **作者:** Junjie Hu; Tianyang Han; Kai Ma; Jialin Gao; Hao Dou; Song Yang; Xianhua He; Jianhui Zhang; Junfeng Luo; Xiaoming Wei; Wenqiang Zhang
>
> **摘要:** Recent subject-driven image customization has achieved significant advancements in fidelity, yet fine-grained entity-level spatial control remains elusive, hindering the broader real-world application. This limitation is mainly attributed to scalable datasets that bind identity with precise positional cues are absent. To this end, we introduce PositionIC, a unified framework that enforces position and identity consistency for multi-subject customization. We construct a scalable synthesis pipeline that employs a bidirectional generation paradigm to eliminate subject drift and maintain semantic coherence. On top of these data, we design a lightweight positional modulation layer that decouples spatial embeddings among subjects, enabling independent, accurate placement while preserving visual fidelity. Extensive experiments demonstrate that our approach can achieve precise spatial control while maintaining high consistency in image customization task. PositionIC paves the way for controllable, high-fidelity image customization in open-world, multi-entity scenarios and will be released to foster further research.
>
---
#### [new 031] A Deep Learning-Based Ensemble System for Automated Shoulder Fracture Detection in Clinical Radiographs
- **分类: cs.CV; cs.AI; 68T07; I.2.10**

- **简介: 该论文属于医学图像分析任务，旨在解决肩部骨折在X光片中易被漏诊的问题。作者构建了一个基于多模型深度学习的集成系统，采用Faster R-CNN、EfficientDet和RF-DETR等架构，并结合集成技术提升检测性能。实验表明，该系统在检测肩部骨折方面表现出高准确性和F1分数，适用于临床快速筛查与分诊支持。**

- **链接: [http://arxiv.org/pdf/2507.13408v1](http://arxiv.org/pdf/2507.13408v1)**

> **作者:** Hemanth Kumar M; Karthika M; Saianiruth M; Vasanthakumar Venugopal; Anandakumar D; Revathi Ezhumalai; Charulatha K; Kishore Kumar J; Dayana G; Kalyan Sivasailam; Bargava Subramanian
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Background: Shoulder fractures are often underdiagnosed, especially in emergency and high-volume clinical settings. Studies report up to 10% of such fractures may be missed by radiologists. AI-driven tools offer a scalable way to assist early detection and reduce diagnostic delays. We address this gap through a dedicated AI system for shoulder radiographs. Methods: We developed a multi-model deep learning system using 10,000 annotated shoulder X-rays. Architectures include Faster R-CNN (ResNet50-FPN, ResNeXt), EfficientDet, and RF-DETR. To enhance detection, we applied bounding box and classification-level ensemble techniques such as Soft-NMS, WBF, and NMW fusion. Results: The NMW ensemble achieved 95.5% accuracy and an F1-score of 0.9610, outperforming individual models across all key metrics. It demonstrated strong recall and localization precision, confirming its effectiveness for clinical fracture detection in shoulder X-rays. Conclusion: The results show ensemble-based AI can reliably detect shoulder fractures in radiographs with high clinical relevance. The model's accuracy and deployment readiness position it well for integration into real-time diagnostic workflows. The current model is limited to binary fracture detection, reflecting its design for rapid screening and triage support rather than detailed orthopedic classification.
>
---
#### [new 032] Global Modeling Matters: A Fast, Lightweight and Effective Baseline for Efficient Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像恢复任务，旨在解决恶劣天气导致的图像质量下降问题。现有方法依赖复杂架构影响效率，本文提出PW-FNet，结合小波与傅里叶变换，实现高效全局建模，提升恢复质量与计算效率。**

- **链接: [http://arxiv.org/pdf/2507.13663v1](http://arxiv.org/pdf/2507.13663v1)**

> **作者:** Xingyu Jiang; Ning Gao; Hongkun Dou; Xiuhui Zhang; Xiaoqing Zhong; Yue Deng; Hongjue Li
>
> **摘要:** Natural image quality is often degraded by adverse weather conditions, significantly impairing the performance of downstream tasks. Image restoration has emerged as a core solution to this challenge and has been widely discussed in the literature. Although recent transformer-based approaches have made remarkable progress in image restoration, their increasing system complexity poses significant challenges for real-time processing, particularly in real-world deployment scenarios. To this end, most existing methods attempt to simplify the self-attention mechanism, such as by channel self-attention or state space model. However, these methods primarily focus on network architecture while neglecting the inherent characteristics of image restoration itself. In this context, we explore a pyramid Wavelet-Fourier iterative pipeline to demonstrate the potential of Wavelet-Fourier processing for image restoration. Inspired by the above findings, we propose a novel and efficient restoration baseline, named Pyramid Wavelet-Fourier Network (PW-FNet). Specifically, PW-FNet features two key design principles: 1) at the inter-block level, integrates a pyramid wavelet-based multi-input multi-output structure to achieve multi-scale and multi-frequency bands decomposition; and 2) at the intra-block level, incorporates Fourier transforms as an efficient alternative to self-attention mechanisms, effectively reducing computational complexity while preserving global modeling capability. Extensive experiments on tasks such as image deraining, raindrop removal, image super-resolution, motion deblurring, image dehazing, image desnowing and underwater/low-light enhancement demonstrate that PW-FNet not only surpasses state-of-the-art methods in restoration quality but also achieves superior efficiency, with significantly reduced parameter size, computational cost and inference time.
>
---
#### [new 033] Can Synthetic Images Conquer Forgetting? Beyond Unexplored Doubts in Few-Shot Class-Incremental Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于少样本类增量学习任务，旨在解决训练数据极少下的灾难性遗忘问题。论文提出Diffusion-FSCIL，利用冻结的文本到图像扩散模型生成多尺度特征，结合特征蒸馏，实现高效学习与旧类性能保持。**

- **链接: [http://arxiv.org/pdf/2507.13739v1](http://arxiv.org/pdf/2507.13739v1)**

> **作者:** Junsu Kim; Yunhoe Ku; Seungryul Baek
>
> **备注:** 6th CLVISION ICCV Workshop accepted
>
> **摘要:** Few-shot class-incremental learning (FSCIL) is challenging due to extremely limited training data; while aiming to reduce catastrophic forgetting and learn new information. We propose Diffusion-FSCIL, a novel approach that employs a text-to-image diffusion model as a frozen backbone. Our conjecture is that FSCIL can be tackled using a large generative model's capabilities benefiting from 1) generation ability via large-scale pre-training; 2) multi-scale representation; 3) representational flexibility through the text encoder. To maximize the representation capability, we propose to extract multiple complementary diffusion features to play roles as latent replay with slight support from feature distillation for preventing generative biases. Our framework realizes efficiency through 1) using a frozen backbone; 2) minimal trainable components; 3) batch processing of multiple feature extractions. Extensive experiments on CUB-200, \emph{mini}ImageNet, and CIFAR-100 show that Diffusion-FSCIL surpasses state-of-the-art methods, preserving performance on previously learned classes and adapting effectively to new ones.
>
---
#### [new 034] One Step Closer: Creating the Future to Boost Monocular Semantic Scene Completion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D语义场景补全（SSC）任务，旨在解决单目视觉中因遮挡和视野限制导致的场景理解不完整问题。论文提出CF-SSC框架，通过伪未来帧预测扩展感知范围，结合位姿和深度信息实现时空一致的3D场景补全，提升了遮挡推理与精度。**

- **链接: [http://arxiv.org/pdf/2507.13801v1](http://arxiv.org/pdf/2507.13801v1)**

> **作者:** Haoang Lu; Yuanqi Su; Xiaoning Zhang; Hao Hu
>
> **摘要:** In recent years, visual 3D Semantic Scene Completion (SSC) has emerged as a critical perception task for autonomous driving due to its ability to infer complete 3D scene layouts and semantics from single 2D images. However, in real-world traffic scenarios, a significant portion of the scene remains occluded or outside the camera's field of view -- a fundamental challenge that existing monocular SSC methods fail to address adequately. To overcome these limitations, we propose Creating the Future SSC (CF-SSC), a novel temporal SSC framework that leverages pseudo-future frame prediction to expand the model's effective perceptual range. Our approach combines poses and depths to establish accurate 3D correspondences, enabling geometrically-consistent fusion of past, present, and predicted future frames in 3D space. Unlike conventional methods that rely on simple feature stacking, our 3D-aware architecture achieves more robust scene completion by explicitly modeling spatial-temporal relationships. Comprehensive experiments on SemanticKITTI and SSCBench-KITTI-360 benchmarks demonstrate state-of-the-art performance, validating the effectiveness of our approach, highlighting our method's ability to improve occlusion reasoning and 3D scene completion accuracy.
>
---
#### [new 035] NoiseSDF2NoiseSDF: Learning Clean Neural Fields from Noisy Supervision
- **分类: cs.CV**

- **简介: 该论文属于3D表面重建任务，旨在解决从含噪点云中重建准确隐式表面的问题。受2D图像Noise2Noise方法启发，论文提出NoiseSDF2NoiseSDF，通过最小化含噪SDF间的MSE损失，实现从噪声监督中直接学习干净的神经SDF，从而提升表面重建质量。**

- **链接: [http://arxiv.org/pdf/2507.13595v1](http://arxiv.org/pdf/2507.13595v1)**

> **作者:** Tengkai Wang; Weihao Li; Ruikai Cui; Shi Qiu; Nick Barnes
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** Reconstructing accurate implicit surface representations from point clouds remains a challenging task, particularly when data is captured using low-quality scanning devices. These point clouds often contain substantial noise, leading to inaccurate surface reconstructions. Inspired by the Noise2Noise paradigm for 2D images, we introduce NoiseSDF2NoiseSDF, a novel method designed to extend this concept to 3D neural fields. Our approach enables learning clean neural SDFs directly from noisy point clouds through noisy supervision by minimizing the MSE loss between noisy SDF representations, allowing the network to implicitly denoise and refine surface estimations. We evaluate the effectiveness of NoiseSDF2NoiseSDF on benchmarks, including the ShapeNet, ABC, Famous, and Real datasets. Experimental results demonstrate that our framework significantly improves surface reconstruction quality from noisy inputs.
>
---
#### [new 036] MADI: Masking-Augmented Diffusion with Inference-Time Scaling for Visual Editing
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉编辑任务，旨在提升扩散模型在图像生成中的可控性与可编辑性。通过提出MAgD训练策略和基于Pause Tokens的推理时容量扩展机制，增强模型对局部结构的编辑能力及整体控制效果，推动扩散模型向通用生成架构发展。**

- **链接: [http://arxiv.org/pdf/2507.13401v1](http://arxiv.org/pdf/2507.13401v1)**

> **作者:** Shreya Kadambi; Risheek Garrepalli; Shubhankar Borse; Munawar Hyatt; Fatih Porikli
>
> **备注:** 26 pages
>
> **摘要:** Despite the remarkable success of diffusion models in text-to-image generation, their effectiveness in grounded visual editing and compositional control remains challenging. Motivated by advances in self-supervised learning and in-context generative modeling, we propose a series of simple yet powerful design choices that significantly enhance diffusion model capacity for structured, controllable generation and editing. We introduce Masking-Augmented Diffusion with Inference-Time Scaling (MADI), a framework that improves the editability, compositionality and controllability of diffusion models through two core innovations. First, we introduce Masking-Augmented gaussian Diffusion (MAgD), a novel training strategy with dual corruption process which combines standard denoising score matching and masked reconstruction by masking noisy input from forward process. MAgD encourages the model to learn discriminative and compositional visual representations, thus enabling localized and structure-aware editing. Second, we introduce an inference-time capacity scaling mechanism based on Pause Tokens, which act as special placeholders inserted into the prompt for increasing computational capacity at inference time. Our findings show that adopting expressive and dense prompts during training further enhances performance, particularly for MAgD. Together, these contributions in MADI substantially enhance the editability of diffusion models, paving the way toward their integration into more general-purpose, in-context generative diffusion architectures.
>
---
#### [new 037] DreamScene: 3D Gaussian-based End-to-end Text-to-3D Scene Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到3D场景生成任务，旨在解决现有方法在自动化、3D一致性与细粒度控制方面的不足。作者提出DreamScene框架，通过场景规划、布局生成、几何采样与相机策略，实现高质量、可编辑的3D场景生成。**

- **链接: [http://arxiv.org/pdf/2507.13985v1](http://arxiv.org/pdf/2507.13985v1)**

> **作者:** Haoran Li; Yuli Tian; Kun Lan; Yong Liao; Lin Wang; Pan Hui; Peng Yuan Zhou
>
> **备注:** Extended version of ECCV 2024 paper "DreamScene"
>
> **摘要:** Generating 3D scenes from natural language holds great promise for applications in gaming, film, and design. However, existing methods struggle with automation, 3D consistency, and fine-grained control. We present DreamScene, an end-to-end framework for high-quality and editable 3D scene generation from text or dialogue. DreamScene begins with a scene planning module, where a GPT-4 agent infers object semantics and spatial constraints to construct a hybrid graph. A graph-based placement algorithm then produces a structured, collision-free layout. Based on this layout, Formation Pattern Sampling (FPS) generates object geometry using multi-timestep sampling and reconstructive optimization, enabling fast and realistic synthesis. To ensure global consistent, DreamScene employs a progressive camera sampling strategy tailored to both indoor and outdoor settings. Finally, the system supports fine-grained scene editing, including object movement, appearance changes, and 4D dynamic motion. Experiments demonstrate that DreamScene surpasses prior methods in quality, consistency, and flexibility, offering a practical solution for open-domain 3D content creation. Code and demos are available at https://dreamscene-project.github.io.
>
---
#### [new 038] Open-Vocabulary Object Detection in UAV Imagery: A Review and Future Perspectives
- **分类: cs.CV**

- **简介: 该论文属于无人机（UAV）图像中的开放词汇目标检测（OVOD）任务，旨在解决传统方法仅能检测预定义类别的局限性。论文系统综述了基于跨模态图文对齐（如CLIP）的OVOD方法，构建了分类体系，分析了挑战与问题，并展望了未来研究方向，为该领域提供了结构化参考。**

- **链接: [http://arxiv.org/pdf/2507.13359v1](http://arxiv.org/pdf/2507.13359v1)**

> **作者:** Yang Zhou; Junjie Li; CongYang Ou; Dawei Yan; Haokui Zhang; Xizhe Xue
>
> **备注:** 27 pages, 5 figures
>
> **摘要:** Due to its extensive applications, aerial image object detection has long been a hot topic in computer vision. In recent years, advancements in Unmanned Aerial Vehicles (UAV) technology have further propelled this field to new heights, giving rise to a broader range of application requirements. However, traditional UAV aerial object detection methods primarily focus on detecting predefined categories, which significantly limits their applicability. The advent of cross-modal text-image alignment (e.g., CLIP) has overcome this limitation, enabling open-vocabulary object detection (OVOD), which can identify previously unseen objects through natural language descriptions. This breakthrough significantly enhances the intelligence and autonomy of UAVs in aerial scene understanding. This paper presents a comprehensive survey of OVOD in the context of UAV aerial scenes. We begin by aligning the core principles of OVOD with the unique characteristics of UAV vision, setting the stage for a specialized discussion. Building on this foundation, we construct a systematic taxonomy that categorizes existing OVOD methods for aerial imagery and provides a comprehensive overview of the relevant datasets. This structured review enables us to critically dissect the key challenges and open problems at the intersection of these fields. Finally, based on this analysis, we outline promising future research directions and application prospects. This survey aims to provide a clear road map and a valuable reference for both newcomers and seasoned researchers, fostering innovation in this rapidly evolving domain. We keep tracing related works at https://github.com/zhouyang2002/OVOD-in-UVA-imagery
>
---
#### [new 039] Enhancing Spatial Reasoning in Vision-Language Models via Chain-of-Thought Prompting and Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL; I.2.10; I.4.8; I.2.6; I.2.7; I.5.4; I.5.1**

- **简介: 该论文属于视觉-语言模型（VLM）任务，旨在提升模型的空间推理能力。论文探讨了思维链（CoT）提示和强化学习方法，发现结构化场景图提示（SceneGraph CoT）与Group Relative Policy Optimization（GRPO）强化学习结合，能显著提高模型在空间推理任务上的准确性和泛化能力，优于传统监督微调（SFT）。**

- **链接: [http://arxiv.org/pdf/2507.13362v1](http://arxiv.org/pdf/2507.13362v1)**

> **作者:** Binbin Ji; Siddharth Agrawal; Qiance Tang; Yvonne Wu
>
> **备注:** 10 pages, 5 figures, submitted to a conference (IEEE formate). Authored by students from the Courant Institute, NYU
>
> **摘要:** This study investigates the spatial reasoning capabilities of vision-language models (VLMs) through Chain-of-Thought (CoT) prompting and reinforcement learning. We begin by evaluating the impact of different prompting strategies and find that simple CoT formats, where the model generates a reasoning step before the answer, not only fail to help, but can even harm the model's original performance. In contrast, structured multi-stage prompting based on scene graphs (SceneGraph CoT) significantly improves spatial reasoning accuracy. Furthermore, to improve spatial reasoning ability, we fine-tune models using Group Relative Policy Optimization (GRPO) on the SAT dataset and evaluate their performance on CVBench. Compared to supervised fine-tuning (SFT), GRPO achieves higher accuracy on Pass@1 evaluations and demonstrates superior robustness under out-of-distribution (OOD) conditions. In particular, we find that SFT overfits to surface-level linguistic patterns and may degrade performance when test-time phrasing changes (e.g., from "closer to" to "farther from"). GRPO, on the other hand, generalizes more reliably and maintains stable performance under such shifts. Our findings provide insights into how reinforcement learning and structured prompting improve the spatial reasoning capabilities and generalization behavior of modern VLMs. All code is open source at: https://github.com/Yvonne511/spatial-vlm-investigator
>
---
#### [new 040] DynFaceRestore: Balancing Fidelity and Quality in Diffusion-Guided Blind Face Restoration with Dynamic Blur-Level Mapping and Guidance
- **分类: cs.CV**

- **简介: 该论文属于盲脸修复任务，旨在从未知退化的人脸图像中恢复高质量细节。现有方法因固定扩散步长和全局引导易导致过度或不足修复。论文提出DynFaceRestore，通过动态模糊级别映射和局部引导调整，在扩散过程中平衡保真度与质量，提升修复效果。**

- **链接: [http://arxiv.org/pdf/2507.13797v1](http://arxiv.org/pdf/2507.13797v1)**

> **作者:** Huu-Phu Do; Yu-Wei Chen; Yi-Cheng Liao; Chi-Wei Hsiao; Han-Yang Wang; Wei-Chen Chiu; Ching-Chun Huang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Blind Face Restoration aims to recover high-fidelity, detail-rich facial images from unknown degraded inputs, presenting significant challenges in preserving both identity and detail. Pre-trained diffusion models have been increasingly used as image priors to generate fine details. Still, existing methods often use fixed diffusion sampling timesteps and a global guidance scale, assuming uniform degradation. This limitation and potentially imperfect degradation kernel estimation frequently lead to under- or over-diffusion, resulting in an imbalance between fidelity and quality. We propose DynFaceRestore, a novel blind face restoration approach that learns to map any blindly degraded input to Gaussian blurry images. By leveraging these blurry images and their respective Gaussian kernels, we dynamically select the starting timesteps for each blurry image and apply closed-form guidance during the diffusion sampling process to maintain fidelity. Additionally, we introduce a dynamic guidance scaling adjuster that modulates the guidance strength across local regions, enhancing detail generation in complex areas while preserving structural fidelity in contours. This strategy effectively balances the trade-off between fidelity and quality. DynFaceRestore achieves state-of-the-art performance in both quantitative and qualitative evaluations, demonstrating robustness and effectiveness in blind face restoration.
>
---
#### [new 041] Gaussian kernel-based motion measurement
- **分类: cs.CV**

- **简介: 该论文属于结构健康监测中的视觉运动测量任务，旨在解决现有方法在亚像素级测量中精度不足或依赖手动调参的问题。作者提出了一种基于高斯核的运动测量方法，通过跟踪高斯核位置提取运动，并引入运动一致性与超分辨率约束，提高了测量的精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.13693v1](http://arxiv.org/pdf/2507.13693v1)**

> **作者:** Hongyi Liu; Haifeng Wang
>
> **摘要:** The growing demand for structural health monitoring has driven increasing interest in high-precision motion measurement, as structural information derived from extracted motions can effectively reflect the current condition of the structure. Among various motion measurement techniques, vision-based methods stand out due to their low cost, easy installation, and large-scale measurement. However, when it comes to sub-pixel-level motion measurement, current vision-based methods either lack sufficient accuracy or require extensive manual parameter tuning (e.g., pyramid layers, target pixels, and filter parameters) to reach good precision. To address this issue, we developed a novel Gaussian kernel-based motion measurement method, which can extract the motion between different frames via tracking the location of Gaussian kernels. The motion consistency, which fits practical structural conditions, and a super-resolution constraint, are introduced to increase accuracy and robustness of our method. Numerical and experimental validations show that it can consistently reach high accuracy without customized parameter setup for different test samples.
>
---
#### [new 042] Tackling fake images in cybersecurity -- Interpretation of a StyleGAN and lifting its black-box
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于图像生成与网络安全任务，旨在分析StyleGAN模型的工作机制并探讨其潜在风险。论文通过研究生成器结构、权重修剪和潜在向量操控，揭示了模型的可解释性与安全性问题，强调AI生成图像可能被滥用于制造虚假身份，从而带来网络安全威胁。**

- **链接: [http://arxiv.org/pdf/2507.13722v1](http://arxiv.org/pdf/2507.13722v1)**

> **作者:** Julia Laubmann; Johannes Reschke
>
> **摘要:** In today's digital age, concerns about the dangers of AI-generated images are increasingly common. One powerful tool in this domain is StyleGAN (style-based generative adversarial networks), a generative adversarial network capable of producing highly realistic synthetic faces. To gain a deeper understanding of how such a model operates, this work focuses on analyzing the inner workings of StyleGAN's generator component. Key architectural elements and techniques, such as the Equalized Learning Rate, are explored in detail to shed light on the model's behavior. A StyleGAN model is trained using the PyTorch framework, enabling direct inspection of its learned weights. Through pruning, it is revealed that a significant number of these weights can be removed without drastically affecting the output, leading to reduced computational requirements. Moreover, the role of the latent vector -- which heavily influences the appearance of the generated faces -- is closely examined. Global alterations to this vector primarily affect aspects like color tones, while targeted changes to individual dimensions allow for precise manipulation of specific facial features. This ability to finetune visual traits is not only of academic interest but also highlights a serious ethical concern: the potential misuse of such technology. Malicious actors could exploit this capability to fabricate convincing fake identities, posing significant risks in the context of digital deception and cybercrime.
>
---
#### [new 043] Franca: Nested Matryoshka Clustering for Scalable Visual Representation Learning
- **分类: cs.CV**

- **简介: 该论文属于视觉表征学习任务，旨在解决自监督学习中聚类语义模糊和模型不可靠问题。作者提出Franca模型，采用嵌套Matryoshka聚类和位置解耦策略，提升性能与可解释性，并实现开源。**

- **链接: [http://arxiv.org/pdf/2507.14137v1](http://arxiv.org/pdf/2507.14137v1)**

> **作者:** Shashanka Venkataramanan; Valentinos Pariza; Mohammadreza Salehi; Lukas Knobel; Spyros Gidaris; Elias Ramzi; Andrei Bursuc; Yuki M. Asano
>
> **摘要:** We present Franca (pronounced Fran-ka): free one; the first fully open-source (data, code, weights) vision foundation model that matches and in many cases surpasses the performance of state-of-the-art proprietary models, e.g., DINOv2, CLIP, SigLIPv2, etc. Our approach is grounded in a transparent training pipeline inspired by Web-SSL and uses publicly available data: ImageNet-21K and a subset of ReLAION-2B. Beyond model release, we tackle critical limitations in SSL clustering methods. While modern models rely on assigning image features to large codebooks via clustering algorithms like Sinkhorn-Knopp, they fail to account for the inherent ambiguity in clustering semantics. To address this, we introduce a parameter-efficient, multi-head clustering projector based on nested Matryoshka representations. This design progressively refines features into increasingly fine-grained clusters without increasing the model size, enabling both performance and memory efficiency. Additionally, we propose a novel positional disentanglement strategy that explicitly removes positional biases from dense representations, thereby improving the encoding of semantic content. This leads to consistent gains on several downstream benchmarks, demonstrating the utility of cleaner feature spaces. Our contributions establish a new standard for transparent, high-performance vision models and open a path toward more reproducible and generalizable foundation models for the broader AI community. The code and model checkpoints are available at https://github.com/valeoai/Franca.
>
---
#### [new 044] GOSPA and T-GOSPA quasi-metrics for evaluation of multi-object tracking algorithms
- **分类: cs.CV; math.ST; stat.TH**

- **简介: 论文属于多目标跟踪评估任务，旨在解决现有评估指标对漏检、误检和定位误差的惩罚不够灵活的问题。提出了两种准度量GOSPA和T-GOSPA，可分别衡量目标集和轨迹集之间的差异，支持非对称定位误差和不同代价的漏检与误检，提升了多目标跟踪算法评估的灵活性和适用性。**

- **链接: [http://arxiv.org/pdf/2507.13706v1](http://arxiv.org/pdf/2507.13706v1)**

> **作者:** Ángel F. García-Fernández; Jinhao Gu; Lennart Svensson; Yuxuan Xia; Jan Krejčí; Oliver Kost; Ondřej Straka
>
> **摘要:** This paper introduces two quasi-metrics for performance assessment of multi-object tracking (MOT) algorithms. In particular, one quasi-metric is an extension of the generalised optimal subpattern assignment (GOSPA) metric and measures the discrepancy between sets of objects. The other quasi-metric is an extension of the trajectory GOSPA (T-GOSPA) metric and measures the discrepancy between sets of trajectories. Similar to the GOSPA-based metrics, these quasi-metrics include costs for localisation error for properly detected objects, the number of false objects and the number of missed objects. The T-GOSPA quasi-metric also includes a track switching cost. Differently from the GOSPA and T-GOSPA metrics, the proposed quasi-metrics have the flexibility of penalising missed and false objects with different costs, and the localisation costs are not required to be symmetric. These properties can be useful in MOT evaluation in certain applications. The performance of several Bayesian MOT algorithms is assessed with the T-GOSPA quasi-metric via simulations.
>
---
#### [new 045] A Comprehensive Survey for Real-World Industrial Defect Detection: Challenges, Approaches, and Prospects
- **分类: cs.CV**

- **简介: 该论文属于工业缺陷检测任务，旨在解决传统方法在精度、自动化和可扩展性方面的不足。论文综述了基于计算机视觉和深度学习的2D与3D缺陷检测方法，特别强调从封闭集到开放集检测框架的转变，减少了对大量标注数据的依赖，并能识别新类型缺陷。同时分析了实际应用中的挑战并展望了未来趋势。**

- **链接: [http://arxiv.org/pdf/2507.13378v1](http://arxiv.org/pdf/2507.13378v1)**

> **作者:** Yuqi Cheng; Yunkang Cao; Haiming Yao; Wei Luo; Cheng Jiang; Hui Zhang; Weiming Shen
>
> **备注:** 27 pages, 7 figures
>
> **摘要:** Industrial defect detection is vital for upholding product quality across contemporary manufacturing systems. As the expectations for precision, automation, and scalability intensify, conventional inspection approaches are increasingly found wanting in addressing real-world demands. Notable progress in computer vision and deep learning has substantially bolstered defect detection capabilities across both 2D and 3D modalities. A significant development has been the pivot from closed-set to open-set defect detection frameworks, which diminishes the necessity for extensive defect annotations and facilitates the recognition of novel anomalies. Despite such strides, a cohesive and contemporary understanding of industrial defect detection remains elusive. Consequently, this survey delivers an in-depth analysis of both closed-set and open-set defect detection strategies within 2D and 3D modalities, charting their evolution in recent years and underscoring the rising prominence of open-set techniques. We distill critical challenges inherent in practical detection environments and illuminate emerging trends, thereby providing a current and comprehensive vista of this swiftly progressing field.
>
---
#### [new 046] VLA-Mark: A cross modal watermark for large vision-language alignment model
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态水印任务，旨在解决视觉-语言模型中水印影响语义连贯性的问题。作者提出VLA-Mark框架，通过跨模态协调嵌入可检测水印，结合多尺度视觉-文本对齐指标和熵敏感机制，实现在不损害语义一致性的同时提升水印鲁棒性与检测性能。**

- **链接: [http://arxiv.org/pdf/2507.14067v1](http://arxiv.org/pdf/2507.14067v1)**

> **作者:** Shuliang Liu; Qi Zheng; Jesse Jiaxi Xu; Yibo Yan; He Geng; Aiwei Liu; Peijie Jiang; Jia Liu; Yik-Cheung Tam; Xuming Hu
>
> **摘要:** Vision-language models demand watermarking solutions that protect intellectual property without compromising multimodal coherence. Existing text watermarking methods disrupt visual-textual alignment through biased token selection and static strategies, leaving semantic-critical concepts vulnerable. We propose VLA-Mark, a vision-aligned framework that embeds detectable watermarks while preserving semantic fidelity through cross-modal coordination. Our approach integrates multiscale visual-textual alignment metrics, combining localized patch affinity, global semantic coherence, and contextual attention patterns, to guide watermark injection without model retraining. An entropy-sensitive mechanism dynamically balances watermark strength and semantic preservation, prioritizing visual grounding during low-uncertainty generation phases. Experiments show 7.4% lower PPL and 26.6% higher BLEU than conventional methods, with near-perfect detection (98.8% AUC). The framework demonstrates 96.1\% attack resilience against attacks such as paraphrasing and synonym substitution, while maintaining text-visual consistency, establishing new standards for quality-preserving multimodal watermarking
>
---
#### [new 047] Depth3DLane: Fusing Monocular 3D Lane Detection with Self-Supervised Monocular Depth Estimation
- **分类: cs.CV; cs.RO**

- **简介: 论文提出Depth3DLane，用于单目三维车道检测任务，旨在解决缺乏空间信息、依赖昂贵传感器或真实深度数据的问题。方法融合自监督深度估计，提取场景点云与语义信息，结合3D车道锚点预测几何结构，并可估计相机参数，提升在无标定场景下的适用性与稳定性。**

- **链接: [http://arxiv.org/pdf/2507.13857v1](http://arxiv.org/pdf/2507.13857v1)**

> **作者:** Max van den Hoven; Kishaan Jeeveswaran; Pieter Piscaer; Thijs Wensveen; Elahe Arani; Bahram Zonooz
>
> **摘要:** Monocular 3D lane detection is essential for autonomous driving, but challenging due to the inherent lack of explicit spatial information. Multi-modal approaches rely on expensive depth sensors, while methods incorporating fully-supervised depth networks rely on ground-truth depth data that is impractical to collect at scale. Additionally, existing methods assume that camera parameters are available, limiting their applicability in scenarios like crowdsourced high-definition (HD) lane mapping. To address these limitations, we propose Depth3DLane, a novel dual-pathway framework that integrates self-supervised monocular depth estimation to provide explicit structural information, without the need for expensive sensors or additional ground-truth depth data. Leveraging a self-supervised depth network to obtain a point cloud representation of the scene, our bird's-eye view pathway extracts explicit spatial information, while our front view pathway simultaneously extracts rich semantic information. Depth3DLane then uses 3D lane anchors to sample features from both pathways and infer accurate 3D lane geometry. Furthermore, we extend the framework to predict camera parameters on a per-frame basis and introduce a theoretically motivated fitting procedure to enhance stability on a per-segment basis. Extensive experiments demonstrate that Depth3DLane achieves competitive performance on the OpenLane benchmark dataset. Furthermore, experimental results show that using learned parameters instead of ground-truth parameters allows Depth3DLane to be applied in scenarios where camera calibration is infeasible, unlike previous methods.
>
---
#### [new 048] Enhancing LiDAR Point Features with Foundation Model Priors for 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，旨在解决LiDAR点云特征表达能力不足的问题。通过引入DepthAnything模型提供的深度先验，融合原始LiDAR数据，提升点特征表达能力，并设计双路径特征提取框架及双向门控融合模块，有效结合全局语义与局部结构信息，增强了检测精度。**

- **链接: [http://arxiv.org/pdf/2507.13899v1](http://arxiv.org/pdf/2507.13899v1)**

> **作者:** Yujian Mo; Yan Wu; Junqiao Zhao; Jijun Wang; Yinghao Hu; Jun Yan
>
> **摘要:** Recent advances in foundation models have opened up new possibilities for enhancing 3D perception. In particular, DepthAnything offers dense and reliable geometric priors from monocular RGB images, which can complement sparse LiDAR data in autonomous driving scenarios. However, such priors remain underutilized in LiDAR-based 3D object detection. In this paper, we address the limited expressiveness of raw LiDAR point features, especially the weak discriminative capability of the reflectance attribute, by introducing depth priors predicted by DepthAnything. These priors are fused with the original LiDAR attributes to enrich each point's representation. To leverage the enhanced point features, we propose a point-wise feature extraction module. Then, a Dual-Path RoI feature extraction framework is employed, comprising a voxel-based branch for global semantic context and a point-based branch for fine-grained structural details. To effectively integrate the complementary RoI features, we introduce a bidirectional gated RoI feature fusion module that balances global and local cues. Extensive experiments on the KITTI benchmark show that our method consistently improves detection accuracy, demonstrating the value of incorporating visual foundation model priors into LiDAR-based 3D object detection.
>
---
#### [new 049] Team of One: Cracking Complex Video QA with Model Synergy
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频问答（Video QA）任务，旨在解决现有视频多模态模型在复杂真实场景中理解能力不足、时序建模弱、泛化性差等问题。论文提出一种协同多模型的集成机制，通过结构化思维链协调多个视频语言模型，并利用大语言模型融合可靠回答，提升推理深度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.13820v1](http://arxiv.org/pdf/2507.13820v1)**

> **作者:** Jun Xie; Zhaoran Zhao; Xiongjun Guan; Yingjian Zhu; Hongzhu Yi; Xinming Wang; Feng Chen; Zhepeng Wang
>
> **摘要:** We propose a novel framework for open-ended video question answering that enhances reasoning depth and robustness in complex real-world scenarios, as benchmarked on the CVRR-ES dataset. Existing Video-Large Multimodal Models (Video-LMMs) often exhibit limited contextual understanding, weak temporal modeling, and poor generalization to ambiguous or compositional queries. To address these challenges, we introduce a prompting-and-response integration mechanism that coordinates multiple heterogeneous Video-Language Models (VLMs) via structured chains of thought, each tailored to distinct reasoning pathways. An external Large Language Model (LLM) serves as an evaluator and integrator, selecting and fusing the most reliable responses. Extensive experiments demonstrate that our method significantly outperforms existing baselines across all evaluation metrics, showcasing superior generalization and robustness. Our approach offers a lightweight, extensible strategy for advancing multimodal reasoning without requiring model retraining, setting a strong foundation for future Video-LMM development.
>
---
#### [new 050] PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations
- **分类: cs.CV**

- **简介: 该论文属于三维重建任务，旨在解决无姿态图像的3D高斯点绘（3D-GS）在复杂相机轨迹下重建效果差的问题。作者提出PCR-GS方法，通过特征重投影正则化和小波频率正则化实现相机姿态协同优化，提升了重建质量和姿态估计准确性。**

- **链接: [http://arxiv.org/pdf/2507.13891v1](http://arxiv.org/pdf/2507.13891v1)**

> **作者:** Yu Wei; Jiahui Zhang; Xiaoqin Zhang; Ling Shao; Shijian Lu
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** COLMAP-free 3D Gaussian Splatting (3D-GS) has recently attracted increasing attention due to its remarkable performance in reconstructing high-quality 3D scenes from unposed images or videos. However, it often struggles to handle scenes with complex camera trajectories as featured by drastic rotation and translation across adjacent camera views, leading to degraded estimation of camera poses and further local minima in joint optimization of camera poses and 3D-GS. We propose PCR-GS, an innovative COLMAP-free 3DGS technique that achieves superior 3D scene modeling and camera pose estimation via camera pose co-regularization. PCR-GS achieves regularization from two perspectives. The first is feature reprojection regularization which extracts view-robust DINO features from adjacent camera views and aligns their semantic information for camera pose regularization. The second is wavelet-based frequency regularization which exploits discrepancy in high-frequency details to further optimize the rotation matrix in camera poses. Extensive experiments over multiple real-world scenes show that the proposed PCR-GS achieves superior pose-free 3D-GS scene modeling under dramatic changes of camera trajectories.
>
---
#### [new 051] SkySense V2: A Unified Foundation Model for Multi-modal Remote Sensing
- **分类: cs.CV**

- **简介: 论文提出SkySense V2，一个统一的多模态遥感基础模型，旨在解决现有方法需为每种模态训练独立网络、参数利用低效及预训练方法未充分适配遥感图像特性的问题。其任务属于多模态遥感图像处理，用于城市规划、环境监测等。工作包括设计统一Transformer骨干网络、自适应patch合并模块、可学习模态提示令牌及融合专家模块，并通过新自监督策略预训练，提升模型性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.13812v1](http://arxiv.org/pdf/2507.13812v1)**

> **作者:** Yingying Zhang; Lixiang Ru; Kang Wu; Lei Yu; Lei Liang; Yansheng Li; Jingdong Chen
>
> **备注:** Accepted by ICCV25
>
> **摘要:** The multi-modal remote sensing foundation model (MM-RSFM) has significantly advanced various Earth observation tasks, such as urban planning, environmental monitoring, and natural disaster management. However, most existing approaches generally require the training of separate backbone networks for each data modality, leading to redundancy and inefficient parameter utilization. Moreover, prevalent pre-training methods typically apply self-supervised learning (SSL) techniques from natural images without adequately accommodating the characteristics of remote sensing (RS) images, such as the complicated semantic distribution within a single RS image. In this work, we present SkySense V2, a unified MM-RSFM that employs a single transformer backbone to handle multiple modalities. This backbone is pre-trained with a novel SSL strategy tailored to the distinct traits of RS data. In particular, SkySense V2 incorporates an innovative adaptive patch merging module and learnable modality prompt tokens to address challenges related to varying resolutions and limited feature diversity across modalities. In additional, we incorporate the mixture of experts (MoE) module to further enhance the performance of the foundation model. SkySense V2 demonstrates impressive generalization abilities through an extensive evaluation involving 16 datasets over 7 tasks, outperforming SkySense by an average of 1.8 points.
>
---
#### [new 052] Generalist Forecasting with Frozen Video Models via Latent Diffusion
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视频预测任务，旨在解决通用场景下的短期未来预测问题。作者提出一种基于冻结视频模型的通用预测框架，通过在视觉特征空间中训练潜在扩散模型进行未来特征预测，并结合任务特定解码器实现多任务预测，验证了视觉表征能力与预测性能的强相关性。**

- **链接: [http://arxiv.org/pdf/2507.13942v1](http://arxiv.org/pdf/2507.13942v1)**

> **作者:** Jacob C Walker; Pedro Vélez; Luisa Polania Cabrera; Guangyao Zhou; Rishabh Kabra; Carl Doersch; Maks Ovsjanikov; João Carreira; Shiry Ginosar
>
> **摘要:** Forecasting what will happen next is a critical skill for general-purpose systems that plan or act in the world at different levels of abstraction. In this paper, we identify a strong correlation between a vision model's perceptual ability and its generalist forecasting performance over short time horizons. This trend holds across a diverse set of pretrained models-including those trained generatively-and across multiple levels of abstraction, from raw pixels to depth, point tracks, and object motion. The result is made possible by a novel generalist forecasting framework that operates on any frozen vision backbone: we train latent diffusion models to forecast future features in the frozen representation space, which are then decoded via lightweight, task-specific readouts. To enable consistent evaluation across tasks, we introduce distributional metrics that compare distributional properties directly in the space of downstream tasks and apply this framework to nine models and four tasks. Our results highlight the value of bridging representation learning and generative modeling for temporally grounded video understanding.
>
---
#### [new 053] MaskHOI: Robust 3D Hand-Object Interaction Estimation via Masked Pre-training
- **分类: cs.CV**

- **简介: 该论文属于3D手物交互（HOI）任务，旨在从单目RGB图像中准确估计手和物体的三维姿态。由于图像几何模糊和交互中的严重遮挡，该任务极具挑战。论文提出MaskHOI，通过基于掩码的预训练方法，结合区域特定掩码策略和3D SDF预测，提升模型对几何结构和遮挡的鲁棒性，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.13673v1](http://arxiv.org/pdf/2507.13673v1)**

> **作者:** Yuechen Xie; Haobo Jiang; Jian Yang; Yigong Zhang; Jin Xie
>
> **备注:** 10 pages, 8 figures, 6 tables
>
> **摘要:** In 3D hand-object interaction (HOI) tasks, estimating precise joint poses of hands and objects from monocular RGB input remains highly challenging due to the inherent geometric ambiguity of RGB images and the severe mutual occlusions that occur during interaction.To address these challenges, we propose MaskHOI, a novel Masked Autoencoder (MAE)-driven pretraining framework for enhanced HOI pose estimation. Our core idea is to leverage the masking-then-reconstruction strategy of MAE to encourage the feature encoder to infer missing spatial and structural information, thereby facilitating geometric-aware and occlusion-robust representation learning. Specifically, based on our observation that human hands exhibit far greater geometric complexity than rigid objects, conventional uniform masking fails to effectively guide the reconstruction of fine-grained hand structures. To overcome this limitation, we introduce a Region-specific Mask Ratio Allocation, primarily comprising the region-specific masking assignment and the skeleton-driven hand masking guidance. The former adaptively assigns lower masking ratios to hand regions than to rigid objects, balancing their feature learning difficulty, while the latter prioritizes masking critical hand parts (e.g., fingertips or entire fingers) to realistically simulate occlusion patterns in real-world interactions. Furthermore, to enhance the geometric awareness of the pretrained encoder, we introduce a novel Masked Signed Distance Field (SDF)-driven multimodal learning mechanism. Through the self-masking 3D SDF prediction, the learned encoder is able to perceive the global geometric structure of hands and objects beyond the 2D image plane, overcoming the inherent limitations of monocular input and alleviating self-occlusion issues. Extensive experiments demonstrate that our method significantly outperforms existing state-of-the-art approaches.
>
---
#### [new 054] PoemTale Diffusion: Minimising Information Loss in Poem to Image Generation with Multi-Stage Prompt Refinement
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决诗歌生成图像时信息丢失的问题。作者提出PoemTale Diffusion方法，通过多阶段提示优化和改进扩散模型的注意力机制，提升对抽象诗歌的理解与图像生成质量，并发布诗歌图像数据集P4I。**

- **链接: [http://arxiv.org/pdf/2507.13708v1](http://arxiv.org/pdf/2507.13708v1)**

> **作者:** Sofia Jamil; Bollampalli Areen Reddy; Raghvendra Kumar; Sriparna Saha; Koustava Goswami; K. J. Joseph
>
> **备注:** ECAI 2025
>
> **摘要:** Recent advancements in text-to-image diffusion models have achieved remarkable success in generating realistic and diverse visual content. A critical factor in this process is the model's ability to accurately interpret textual prompts. However, these models often struggle with creative expressions, particularly those involving complex, abstract, or highly descriptive language. In this work, we introduce a novel training-free approach tailored to improve image generation for a unique form of creative language: poetic verse, which frequently features layered, abstract, and dual meanings. Our proposed PoemTale Diffusion approach aims to minimise the information that is lost during poetic text-to-image conversion by integrating a multi stage prompt refinement loop into Language Models to enhance the interpretability of poetic texts. To support this, we adapt existing state-of-the-art diffusion models by modifying their self-attention mechanisms with a consistent self-attention technique to generate multiple consistent images, which are then collectively used to convey the poem's meaning. Moreover, to encourage research in the field of poetry, we introduce the P4I (PoemForImage) dataset, consisting of 1111 poems sourced from multiple online and offline resources. We engaged a panel of poetry experts for qualitative assessments. The results from both human and quantitative evaluations validate the efficacy of our method and contribute a novel perspective to poem-to-image generation with enhanced information capture in the generated images.
>
---
#### [new 055] SparseC-AFM: a deep learning method for fast and accurate characterization of MoS$_2$ with C-AFM
- **分类: cs.CV; cond-mat.mtrl-sci**

- **简介: 该论文属于材料科学与深度学习交叉任务，旨在解决传统导电原子力显微镜（C-AFM）扫描二维材料（如MoS₂）时速度慢的问题。作者提出SparseC-AFM方法，通过深度学习从稀疏扫描数据中快速准确重建导电图，显著减少数据采集时间，并保持关键材料参数的提取精度。**

- **链接: [http://arxiv.org/pdf/2507.13527v1](http://arxiv.org/pdf/2507.13527v1)**

> **作者:** Levi Harris; Md Jayed Hossain; Mufan Qiu; Ruichen Zhang; Pingchuan Ma; Tianlong Chen; Jiaqi Gu; Seth Ariel Tongay; Umberto Celano
>
> **摘要:** The increasing use of two-dimensional (2D) materials in nanoelectronics demands robust metrology techniques for electrical characterization, especially for large-scale production. While atomic force microscopy (AFM) techniques like conductive AFM (C-AFM) offer high accuracy, they suffer from slow data acquisition speeds due to the raster scanning process. To address this, we introduce SparseC-AFM, a deep learning model that rapidly and accurately reconstructs conductivity maps of 2D materials like MoS$_2$ from sparse C-AFM scans. Our approach is robust across various scanning modes, substrates, and experimental conditions. We report a comparison between (a) classic flow implementation, where a high pixel density C-AFM image (e.g., 15 minutes to collect) is manually parsed to extract relevant material parameters, and (b) our SparseC-AFM method, which achieves the same operation using data that requires substantially less acquisition time (e.g., under 5 minutes). SparseC-AFM enables efficient extraction of critical material parameters in MoS$_2$, including film coverage, defect density, and identification of crystalline island boundaries, edges, and cracks. We achieve over 11x reduction in acquisition time compared to manual extraction from a full-resolution C-AFM image. Moreover, we demonstrate that our model-predicted samples exhibit remarkably similar electrical properties to full-resolution data gathered using classic-flow scanning. This work represents a significant step toward translating AI-assisted 2D material characterization from laboratory research to industrial fabrication. Code and model weights are available at github.com/UNITES-Lab/sparse-cafm.
>
---
#### [new 056] Uncertainty Quantification Framework for Aerial and UAV Photogrammetry through Error Propagation
- **分类: cs.CV**

- **简介: 该论文属于摄影测量任务，旨在解决无人机和航空摄影测量中点云精度不确定的问题。通过构建误差传播模型，提出了一种在多视角立体（MVS）阶段估计不确定性的新方法，实现了对整个摄影测量过程的不确定性量化。**

- **链接: [http://arxiv.org/pdf/2507.13486v1](http://arxiv.org/pdf/2507.13486v1)**

> **作者:** Debao Huang; Rongjun Qin
>
> **备注:** 16 pages, 9 figures, this manuscript has been submitted to ISPRS Journal of Photogrammetry and Remote Sensing for consideration
>
> **摘要:** Uncertainty quantification of the photogrammetry process is essential for providing per-point accuracy credentials of the point clouds. Unlike airborne LiDAR, which typically delivers consistent accuracy across various scenes, the accuracy of photogrammetric point clouds is highly scene-dependent, since it relies on algorithm-generated measurements (i.e., stereo or multi-view stereo). Generally, errors of the photogrammetric point clouds propagate through a two-step process: Structure-from-Motion (SfM) with Bundle adjustment (BA), followed by Multi-view Stereo (MVS). While uncertainty estimation in the SfM stage has been well studied using the first-order statistics of the reprojection error function, that in the MVS stage remains largely unsolved and non-standardized, primarily due to its non-differentiable and multi-modal nature (i.e., from pixel values to geometry). In this paper, we present an uncertainty quantification framework closing this gap by associating an error covariance matrix per point accounting for this two-step photogrammetry process. Specifically, to estimate the uncertainty in the MVS stage, we propose a novel, self-calibrating method by taking reliable n-view points (n>=6) per-view to regress the disparity uncertainty using highly relevant cues (such as matching cost values) from the MVS stage. Compared to existing approaches, our method uses self-contained, reliable 3D points extracted directly from the MVS process, with the benefit of being self-supervised and naturally adhering to error propagation path of the photogrammetry process, thereby providing a robust and certifiable uncertainty quantification across diverse scenes. We evaluate the framework using a variety of publicly available airborne and UAV imagery datasets. Results demonstrate that our method outperforms existing approaches by achieving high bounding rates without overestimating uncertainty.
>
---
#### [new 057] $\nabla$NABLA: Neighborhood Adaptive Block-Level Attention
- **分类: cs.CV**

- **简介: 论文提出NABLA，一种自适应块级注意力机制，用于视频生成任务。旨在解决Transformer中全注意力机制计算复杂度高的问题，通过动态调整稀疏性降低计算开销，同时保持生成质量。方法无需低级算子设计，兼容PyTorch，实验证明其训练和推理速度快2.7倍，且性能几乎无损。**

- **链接: [http://arxiv.org/pdf/2507.13546v1](http://arxiv.org/pdf/2507.13546v1)**

> **作者:** Dmitrii Mikhailov; Aleksey Letunovskiy; Maria Kovaleva; Vladimir Arkhipkin; Vladimir Korviakov; Vladimir Polovnikov; Viacheslav Vasilev; Evelina Sidorova; Denis Dimitrov
>
> **摘要:** Recent progress in transformer-based architectures has demonstrated remarkable success in video generation tasks. However, the quadratic complexity of full attention mechanisms remains a critical bottleneck, particularly for high-resolution and long-duration video sequences. In this paper, we propose NABLA, a novel Neighborhood Adaptive Block-Level Attention mechanism that dynamically adapts to sparsity patterns in video diffusion transformers (DiTs). By leveraging block-wise attention with adaptive sparsity-driven threshold, NABLA reduces computational overhead while preserving generative quality. Our method does not require custom low-level operator design and can be seamlessly integrated with PyTorch's Flex Attention operator. Experiments demonstrate that NABLA achieves up to 2.7x faster training and inference compared to baseline almost without compromising quantitative metrics (CLIP score, VBench score, human evaluation score) and visual quality drop. The code and model weights are available here: https://github.com/gen-ai-team/Wan2.1-NABLA
>
---
#### [new 058] CaSTFormer: Causal Spatio-Temporal Transformer for Driving Intention Prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于驾驶意图预测任务，旨在提升人机共驾系统的安全性和交互效率。针对现有方法难以准确建模复杂时空依赖性和人类驾驶行为不确定性的问题，论文提出了CaSTFormer，包含RSF机制、CPE模块和FSN网络，以显式建模因果关系并提升预测性能。**

- **链接: [http://arxiv.org/pdf/2507.13425v1](http://arxiv.org/pdf/2507.13425v1)**

> **作者:** Sirui Wang; Zhou Guan; Bingxi Zhao; Tongjia Gu
>
> **摘要:** Accurate prediction of driving intention is key to enhancing the safety and interactive efficiency of human-machine co-driving systems. It serves as a cornerstone for achieving high-level autonomous driving. However, current approaches remain inadequate for accurately modeling the complex spatio-temporal interdependencies and the unpredictable variability of human driving behavior. To address these challenges, we propose CaSTFormer, a Causal Spatio-Temporal Transformer to explicitly model causal interactions between driver behavior and environmental context for robust intention prediction. Specifically, CaSTFormer introduces a novel Reciprocal Shift Fusion (RSF) mechanism for precise temporal alignment of internal and external feature streams, a Causal Pattern Extraction (CPE) module that systematically eliminates spurious correlations to reveal authentic causal dependencies, and an innovative Feature Synthesis Network (FSN) that adaptively synthesizes these purified representations into coherent spatio-temporal inferences. We evaluate the proposed CaSTFormer on the public Brain4Cars dataset, and it achieves state-of-the-art performance. It effectively captures complex causal spatio-temporal dependencies and enhances both the accuracy and transparency of driving intention prediction.
>
---
#### [new 059] EPSilon: Efficient Point Sampling for Lightening of Hybrid-based 3D Avatar Generation
- **分类: cs.CV**

- **简介: 该论文属于3D人像生成任务，旨在解决基于混合表示的神经渲染方法推理速度慢的问题。现有方法因对采样点进行形变计算导致效率低下。论文提出EPSilon，通过空点剔除策略（ERO和EIO）减少冗余计算，在保持生成质量的同时大幅提升推理和训练速度。**

- **链接: [http://arxiv.org/pdf/2507.13648v1](http://arxiv.org/pdf/2507.13648v1)**

> **作者:** Seungjun Moon; Sangjoon Yu; Gyeong-Moon Park
>
> **摘要:** The rapid advancement of neural radiance fields (NeRF) has paved the way to generate animatable human avatars from a monocular video. However, the sole usage of NeRF suffers from a lack of details, which results in the emergence of hybrid representation that utilizes SMPL-based mesh together with NeRF representation. While hybrid-based models show photo-realistic human avatar generation qualities, they suffer from extremely slow inference due to their deformation scheme: to be aligned with the mesh, hybrid-based models use the deformation based on SMPL skinning weights, which needs high computational costs on each sampled point. We observe that since most of the sampled points are located in empty space, they do not affect the generation quality but result in inference latency with deformation. In light of this observation, we propose EPSilon, a hybrid-based 3D avatar generation scheme with novel efficient point sampling strategies that boost both training and inference. In EPSilon, we propose two methods to omit empty points at rendering; empty ray omission (ERO) and empty interval omission (EIO). In ERO, we wipe out rays that progress through the empty space. Then, EIO narrows down the sampling interval on the ray, which wipes out the region not occupied by either clothes or mesh. The delicate sampling scheme of EPSilon enables not only great computational cost reduction during deformation but also the designation of the important regions to be sampled, which enables a single-stage NeRF structure without hierarchical sampling. Compared to existing methods, EPSilon maintains the generation quality while using only 3.9% of sampled points and achieves around 20 times faster inference, together with 4 times faster training convergence. We provide video results on https://github.com/seungjun-moon/epsilon.
>
---
#### [new 060] Enhancing Breast Cancer Detection with Vision Transformers and Graph Neural Networks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在提高乳腺癌检测的准确性。为解决传统方法性能不足的问题，论文提出了一种结合视觉变换器（ViT）和图神经网络（GNN）的新框架，利用CBIS-DDSM数据集实现乳腺癌检测，取得了84.2%的准确率，并通过注意力热图提升模型可解释性。**

- **链接: [http://arxiv.org/pdf/2507.13372v1](http://arxiv.org/pdf/2507.13372v1)**

> **作者:** Yeming Cai; Zhenglin Li; Yang Wang
>
> **摘要:** Breast cancer is a leading cause of death among women globally, and early detection is critical for improving survival rates. This paper introduces an innovative framework that integrates Vision Transformers (ViT) and Graph Neural Networks (GNN) to enhance breast cancer detection using the CBIS-DDSM dataset. Our framework leverages ViT's ability to capture global image features and GNN's strength in modeling structural relationships, achieving an accuracy of 84.2%, outperforming traditional methods. Additionally, interpretable attention heatmaps provide insights into the model's decision-making process, aiding radiologists in clinical settings.
>
---
#### [new 061] Foundation Models as Class-Incremental Learners for Dermatological Image Classification
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决皮肤疾病分类中的类增量学习问题。作者评估了预训练基础模型在不遗忘旧知识的前提下，逐步学习新类别疾病的能力，并提出了一种简单有效的方法，通过冻结模型主干并增量训练轻量MLP，取得了优异性能。**

- **链接: [http://arxiv.org/pdf/2507.14050v1](http://arxiv.org/pdf/2507.14050v1)**

> **作者:** Mohamed Elkhayat; Mohamed Mahmoud; Jamil Fayyad; Nourhan Bayasi
>
> **备注:** Accepted at the MICCAI EMERGE 2025 workshop
>
> **摘要:** Class-Incremental Learning (CIL) aims to learn new classes over time without forgetting previously acquired knowledge. The emergence of foundation models (FM) pretrained on large datasets presents new opportunities for CIL by offering rich, transferable representations. However, their potential for enabling incremental learning in dermatology remains largely unexplored. In this paper, we systematically evaluate frozen FMs pretrained on large-scale skin lesion datasets for CIL in dermatological disease classification. We propose a simple yet effective approach where the backbone remains frozen, and a lightweight MLP is trained incrementally for each task. This setup achieves state-of-the-art performance without forgetting, outperforming regularization, replay, and architecture based methods. To further explore the capabilities of frozen FMs, we examine zero training scenarios using nearest mean classifiers with prototypes derived from their embeddings. Through extensive ablation studies, we demonstrate that this prototype based variant can also achieve competitive results. Our findings highlight the strength of frozen FMs for continual learning in dermatology and support their broader adoption in real world medical applications. Our code and datasets are available here.
>
---
#### [new 062] AI-ming backwards: Vanishing archaeological landscapes in Mesopotamia and automatic detection of sites on CORONA imagery
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于计算机视觉与考古学交叉任务，旨在解决因城市化导致考古遗址消失的问题。研究者利用CORONA历史卫星图像，重新训练AI模型，成功提升了对已消失遗址的识别精度，并发现了四个新遗址，实现了对不可见考古景观的自动检测。**

- **链接: [http://arxiv.org/pdf/2507.13420v1](http://arxiv.org/pdf/2507.13420v1)**

> **作者:** Alessandro Pistola; Valentina Orru'; Nicolo' Marchetti; Marco Roccetti
>
> **备注:** 25 pages, 9 Figures
>
> **摘要:** By upgrading an existing deep learning model with the knowledge provided by one of the oldest sets of grayscale satellite imagery, known as CORONA, we improved the AI model attitude towards the automatic identification of archaeological sites in an environment which has been completely transformed in the last five decades, including the complete destruction of many of those same sites. The initial Bing based convolutional network model was retrained using CORONA satellite imagery for the district of Abu Ghraib, west of Baghdad, central Mesopotamian floodplain. The results were twofold and surprising. First, the detection precision obtained on the area of interest increased sensibly: in particular, the Intersection over Union (IoU) values, at the image segmentation level, surpassed 85 percent, while the general accuracy in detecting archeological sites reached 90 percent. Second, our retrained model allowed the identification of four new sites of archaeological interest (confirmed through field verification), previously not identified by archaeologists with traditional techniques. This has confirmed the efficacy of using AI techniques and the CORONA imagery from the 1960 to discover archaeological sites currently no longer visible, a concrete breakthrough with significant consequences for the study of landscapes with vanishing archaeological evidence induced by anthropization
>
---
#### [new 063] When Seeing Overrides Knowing: Disentangling Knowledge Conflicts in Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉-语言模型（VLM）在面对视觉输入与内部知识冲突时的处理机制。任务是分析跨模态知识冲突的解决方式。论文构建了包含矛盾信息的多模态数据集，通过logit分析定位控制冲突的关键注意力头，并展示其在引导模型依赖内部知识或视觉输入中的作用。**

- **链接: [http://arxiv.org/pdf/2507.13868v1](http://arxiv.org/pdf/2507.13868v1)**

> **作者:** Francesco Ortu; Zhijing Jin; Diego Doimo; Alberto Cazzaniga
>
> **摘要:** Vision-language models (VLMs) increasingly leverage diverse knowledge sources to address complex tasks, often encountering conflicts between their internal parametric knowledge and external information. Knowledge conflicts can result in hallucinations and unreliable responses, but the mechanisms governing such interactions remain unknown. To address this gap, we analyze the mechanisms that VLMs use to resolve cross-modal conflicts by introducing a dataset of multimodal counterfactual queries that deliberately contradict internal commonsense knowledge. We localize with logit inspection a small set of heads that control the conflict. Moreover, by modifying these heads, we can steer the model towards its internal knowledge or the visual inputs. Finally, we show that attention from such heads pinpoints localized image regions driving visual overrides, outperforming gradient-based attribution in precision.
>
---
#### [new 064] Smart Routing for Multimodal Video Retrieval: When to Search What
- **分类: cs.CV; cs.AI; cs.IR**

- **简介: 该论文属于多模态视频检索任务，旨在解决传统方法依赖密集文本、忽略视觉信息且计算昂贵的问题。作者提出ModaRoute，利用GPT-4.1动态选择最优模态（ASR、OCR、视觉），降低计算开销1.78模态/查询，实现高效检索。**

- **链接: [http://arxiv.org/pdf/2507.13374v1](http://arxiv.org/pdf/2507.13374v1)**

> **作者:** Kevin Dela Rosa
>
> **备注:** Accepted to ICCV 2025 Multimodal Representation and Retrieval Workshop
>
> **摘要:** We introduce ModaRoute, an LLM-based intelligent routing system that dynamically selects optimal modalities for multimodal video retrieval. While dense text captions can achieve 75.9% Recall@5, they require expensive offline processing and miss critical visual information present in 34% of clips with scene text not captured by ASR. By analyzing query intent and predicting information needs, ModaRoute reduces computational overhead by 41% while achieving 60.9% Recall@5. Our approach uses GPT-4.1 to route queries across ASR (speech), OCR (text), and visual indices, averaging 1.78 modalities per query versus exhaustive 3.0 modality search. Evaluation on 1.8M video clips demonstrates that intelligent routing provides a practical solution for scaling multimodal retrieval systems, reducing infrastructure costs while maintaining competitive effectiveness for real-world deployment.
>
---
#### [new 065] GRAM-MAMBA: Holistic Feature Alignment for Wireless Perception with Adaptive Low-Rank Compensation
- **分类: cs.CV**

- **简介: 论文提出GRAM-MAMBA，用于物联网多模态感知的高效特征对齐方法。该方法基于Mamba模型处理传感器时序数据，结合GRAM矩阵优化模态间对齐，并引入低秩补偿策略应对模态缺失问题。任务为多模态融合，旨在提升资源受限环境下的感知效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.13803v1](http://arxiv.org/pdf/2507.13803v1)**

> **作者:** Weiqi Yang; Xu Zhou; Jingfu Guan; Hao Du; Tianyu Bai
>
> **摘要:** Multi-modal fusion is crucial for Internet of Things (IoT) perception, widely deployed in smart homes, intelligent transport, industrial automation, and healthcare. However, existing systems often face challenges: high model complexity hinders deployment in resource-constrained environments, unidirectional modal alignment neglects inter-modal relationships, and robustness suffers when sensor data is missing. These issues impede efficient and robust multimodal perception in real-world IoT settings. To overcome these limitations, we propose GRAM-MAMBA. This framework utilizes the linear-complexity Mamba model for efficient sensor time-series processing, combined with an optimized GRAM matrix strategy for pairwise alignment among modalities, addressing the shortcomings of traditional single-modality alignment. Inspired by Low-Rank Adaptation (LoRA), we introduce an adaptive low-rank layer compensation strategy to handle missing modalities post-training. This strategy freezes the pre-trained model core and irrelevant adaptive layers, fine-tuning only those related to available modalities and the fusion process. Extensive experiments validate GRAM-MAMBA's effectiveness. On the SPAWC2021 indoor positioning dataset, the pre-trained model shows lower error than baselines; adapting to missing modalities yields a 24.5% performance boost by training less than 0.2% of parameters. On the USC-HAD human activity recognition dataset, it achieves 93.55% F1 and 93.81% Overall Accuracy (OA), outperforming prior work; the update strategy increases F1 by 23% while training less than 0.3% of parameters. These results highlight GRAM-MAMBA's potential for achieving efficient and robust multimodal perception in resource-constrained environments.
>
---
#### [new 066] LoRA-Loop: Closing the Synthetic Replay Cycle for Continual VLM Learning
- **分类: cs.CV**

- **简介: 论文提出LoRA-Loop，属于持续学习任务，旨在解决视觉语言模型在持续学习中因合成样本误导导致的知识遗忘问题。通过在Stable Diffusion中引入任务特定的低秩适配器，并结合置信度筛选机制，提升合成样本质量，从而增强模型的稳定性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.13568v1](http://arxiv.org/pdf/2507.13568v1)**

> **作者:** Kaihong Wang; Donghyun Kim; Margrit Betke
>
> **摘要:** Continual learning for vision-language models has achieved remarkable performance through synthetic replay, where samples are generated using Stable Diffusion to regularize during finetuning and retain knowledge. However, real-world downstream applications often exhibit domain-specific nuances and fine-grained semantics not captured by generators, causing synthetic-replay methods to produce misaligned samples that misguide finetuning and undermine retention of prior knowledge. In this work, we propose a LoRA-enhanced synthetic-replay framework that injects task-specific low-rank adapters into a frozen Stable Diffusion model, efficiently capturing each new task's unique visual and semantic patterns. Specifically, we introduce a two-stage, confidence-based sample selection: we first rank real task data by post-finetuning VLM confidence to focus LoRA finetuning on the most representative examples, then generate synthetic samples and again select them by confidence for distillation. Our approach integrates seamlessly with existing replay pipelines-simply swap in the adapted generator to boost replay fidelity. Extensive experiments on the Multi-domain Task Incremental Learning (MTIL) benchmark show that our method outperforms previous synthetic-replay techniques, achieving an optimal balance among plasticity, stability, and zero-shot capability. These results demonstrate the effectiveness of generator adaptation via LoRA for robust continual learning in VLMs.
>
---
#### [new 067] Automatic Classification and Segmentation of Tunnel Cracks Based on Deep Learning and Visual Explanations
- **分类: cs.CV**

- **简介: 该论文属于图像分类与语义分割任务，旨在解决隧道裂缝自动检测与分析问题。论文提出了一种基于深度学习的两步方法：第一阶段使用DenseNet-169对隧道图像进行分类，筛选含裂缝图像；第二阶段采用DeepLabV3+模型实现裂缝精确分割，并通过可视化解释增强模型可解释性。实验表明该方法在分类和分割性能上均优于现有模型，有助于隧道健康状态的快速评估。**

- **链接: [http://arxiv.org/pdf/2507.14010v1](http://arxiv.org/pdf/2507.14010v1)**

> **作者:** Yong Feng; Xiaolei Zhang; Shijin Feng; Yong Zhao; Yihan Chen
>
> **备注:** 8 pages, 10 figures, 3 tables
>
> **摘要:** Tunnel lining crack is a crucial indicator of tunnels' safety status. Aiming to classify and segment tunnel cracks with enhanced accuracy and efficiency, this study proposes a two-step deep learning-based method. An automatic tunnel image classification model is developed using the DenseNet-169 in the first step. The proposed crack segmentation model in the second step is based on the DeepLabV3+, whose internal logic is evaluated via a score-weighted visual explanation technique. Proposed method combines tunnel image classification and segmentation together, so that the selected images containing cracks from the first step are segmented in the second step to improve the detection accuracy and efficiency. The superior performances of the two-step method are validated by experiments. The results show that the accuracy and frames per second (FPS) of the tunnel crack classification model are 92.23% and 39.80, respectively, which are higher than other convolutional neural networks (CNN) based and Transformer based models. Also, the intersection over union (IoU) and F1 score of the tunnel crack segmentation model are 57.01% and 67.44%, respectively, outperforming other state-of-the-art models. Moreover, the provided visual explanations in this study are conducive to understanding the "black box" of deep learning-based models. The developed two-stage deep learning-based method integrating visual explanations provides a basis for fast and accurate quantitative assessment of tunnel health status.
>
---
#### [new 068] Augmented Reality in Cultural Heritage: A Dual-Model Pipeline for 3D Artwork Reconstruction
- **分类: cs.CV**

- **简介: 论文提出了一种用于博物馆环境的增强现实（AR）管线，旨在识别艺术品并从单张图像生成精确的3D模型。该方法结合两种深度估计模型（GLPN和Depth-Anything），优化深度图以提升重建精度和视觉真实感，解决艺术品不规则轮廓和纹理变化带来的挑战，为博物馆提供增强游客体验的交互式数字内容方案。**

- **链接: [http://arxiv.org/pdf/2507.13719v1](http://arxiv.org/pdf/2507.13719v1)**

> **作者:** Daniele Pannone; Alessia Castronovo; Maurizio Mancini; Gian Luca Foresti; Claudio Piciarelli; Rossana Gabrieli; Muhammad Yasir Bilal; Danilo Avola
>
> **摘要:** This paper presents an innovative augmented reality pipeline tailored for museum environments, aimed at recognizing artworks and generating accurate 3D models from single images. By integrating two complementary pre-trained depth estimation models, i.e., GLPN for capturing global scene structure and Depth-Anything for detailed local reconstruction, the proposed approach produces optimized depth maps that effectively represent complex artistic features. These maps are then converted into high-quality point clouds and meshes, enabling the creation of immersive AR experiences. The methodology leverages state-of-the-art neural network architectures and advanced computer vision techniques to overcome challenges posed by irregular contours and variable textures in artworks. Experimental results demonstrate significant improvements in reconstruction accuracy and visual realism, making the system a highly robust tool for museums seeking to enhance visitor engagement through interactive digital content.
>
---
#### [new 069] Sugar-Beet Stress Detection using Satellite Image Time Series
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于农业遥感任务，旨在解决糖用甜菜田胁迫检测问题。作者提出一种无监督方法，利用3D卷积自编码器提取Sentinel-2图像序列特征，并结合时间编码进行聚类分析，实现跨年份的胁迫识别。**

- **链接: [http://arxiv.org/pdf/2507.13514v1](http://arxiv.org/pdf/2507.13514v1)**

> **作者:** Bhumika Laxman Sadbhave; Philipp Vaeth; Denise Dejon; Gunther Schorcht; Magda Gregorová
>
> **摘要:** Satellite Image Time Series (SITS) data has proven effective for agricultural tasks due to its rich spectral and temporal nature. In this study, we tackle the task of stress detection in sugar-beet fields using a fully unsupervised approach. We propose a 3D convolutional autoencoder model to extract meaningful features from Sentinel-2 image sequences, combined with acquisition-date-specific temporal encodings to better capture the growth dynamics of sugar-beets. The learned representations are used in a downstream clustering task to separate stressed from healthy fields. The resulting stress detection system can be directly applied to data from different years, offering a practical and accessible tool for stress detection in sugar-beets.
>
---
#### [new 070] QuantEIT: Ultra-Lightweight Quantum-Assisted Inference for Chest Electrical Impedance Tomography
- **分类: cs.CV; cs.ET; cs.LG**

- **简介: 该论文属于医学成像任务，旨在解决胸电抗断层成像（EIT）中图像重建的不适定逆问题。现有深度学习方法因模型复杂、参数多影响效率。论文提出QuantEIT框架，结合量子电路与轻量网络，在无需训练数据的情况下实现高效图像重建，参数仅需0.2%，且抗噪性强。**

- **链接: [http://arxiv.org/pdf/2507.14031v1](http://arxiv.org/pdf/2507.14031v1)**

> **作者:** Hao Fang; Sihao Teng; Hao Yu; Siyi Yuan; Huaiwu He; Zhe Liu; Yunjie Yang
>
> **备注:** 10 pages, 12 figures
>
> **摘要:** Electrical Impedance Tomography (EIT) is a non-invasive, low-cost bedside imaging modality with high temporal resolution, making it suitable for bedside monitoring. However, its inherently ill-posed inverse problem poses significant challenges for accurate image reconstruction. Deep learning (DL)-based approaches have shown promise but often rely on complex network architectures with a large number of parameters, limiting efficiency and scalability. Here, we propose an Ultra-Lightweight Quantum-Assisted Inference (QuantEIT) framework for EIT image reconstruction. QuantEIT leverages a Quantum-Assisted Network (QA-Net), combining parallel 2-qubit quantum circuits to generate expressive latent representations that serve as implicit nonlinear priors, followed by a single linear layer for conductivity reconstruction. This design drastically reduces model complexity and parameter number. Uniquely, QuantEIT operates in an unsupervised, training-data-free manner and represents the first integration of quantum circuits into EIT image reconstruction. Extensive experiments on simulated and real-world 2D and 3D EIT lung imaging data demonstrate that QuantEIT outperforms conventional methods, achieving comparable or superior reconstruction accuracy using only 0.2% of the parameters, with enhanced robustness to noise.
>
---
#### [new 071] Just Add Geometry: Gradient-Free Open-Vocabulary 3D Detection Without Human-in-the-Loop
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D目标检测任务，旨在解决现有数据集类别有限、标注成本高的问题。利用2D视觉语言模型实现无需人工标注和训练的开放词汇3D检测，通过几何方法生成3D边界框，并构建新数据集验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2507.13363v1](http://arxiv.org/pdf/2507.13363v1)**

> **作者:** Atharv Goel; Mehar Khurana
>
> **摘要:** Modern 3D object detection datasets are constrained by narrow class taxonomies and costly manual annotations, limiting their ability to scale to open-world settings. In contrast, 2D vision-language models trained on web-scale image-text pairs exhibit rich semantic understanding and support open-vocabulary detection via natural language prompts. In this work, we leverage the maturity and category diversity of 2D foundation models to perform open-vocabulary 3D object detection without any human-annotated 3D labels. Our pipeline uses a 2D vision-language detector to generate text-conditioned proposals, which are segmented with SAM and back-projected into 3D using camera geometry and either LiDAR or monocular pseudo-depth. We introduce a geometric inflation strategy based on DBSCAN clustering and Rotating Calipers to infer 3D bounding boxes without training. To simulate adverse real-world conditions, we construct Pseudo-nuScenes, a fog-augmented, RGB-only variant of the nuScenes dataset. Experiments demonstrate that our method achieves competitive localization performance across multiple settings, including LiDAR-based and purely RGB-D inputs, all while remaining training-free and open-vocabulary. Our results highlight the untapped potential of 2D foundation models for scalable 3D perception. We open-source our code and resources at https://github.com/atharv0goel/open-world-3D-det.
>
---
#### [new 072] Total Generalized Variation of the Normal Vector Field and Applications to Mesh Denoising
- **分类: cs.CV; math.DG; math.OC**

- **简介: 该论文属于三维网格去噪任务，旨在解决网格法向量场的高阶正则化问题。作者提出了一种基于切线Raviart-Thomas有限元空间的离散广义全变差（TGV）模型，用于处理球面值法向量场，提升了网格去噪效果。**

- **链接: [http://arxiv.org/pdf/2507.13530v1](http://arxiv.org/pdf/2507.13530v1)**

> **作者:** Lukas Baumgärtner; Ronny Bergmann; Roland Herzog; Stephan Schmidt; Manuel Weiß
>
> **摘要:** We propose a novel formulation for the second-order total generalized variation (TGV) of the normal vector on an oriented, triangular mesh embedded in $\mathbb{R}^3$. The normal vector is considered as a manifold-valued function, taking values on the unit sphere. Our formulation extends previous discrete TGV models for piecewise constant scalar data that utilize a Raviart-Thomas function space. To exctend this formulation to the manifold setting, a tailor-made tangential Raviart-Thomas type finite element space is constructed in this work. The new regularizer is compared to existing methods in mesh denoising experiments.
>
---
#### [new 073] Butter: Frequency Consistency and Hierarchical Fusion for Autonomous Driving Object Detection
- **分类: cs.CV; I.4.8; I.2.10; H.5.1; I.2.6**

- **简介: 论文提出了一种名为Butter的目标检测框架，用于自动驾驶中的物体检测任务。它旨在解决现有方法在多尺度特征一致性、检测精度与计算效率之间的平衡问题。工作主要包括两部分：频率自适应特征一致性增强（FAFCE）和渐进式层次特征融合网络（PHFFNet），以提升特征表示能力。实验表明其在多个数据集上效果良好，兼顾准确性与效率。**

- **链接: [http://arxiv.org/pdf/2507.13373v1](http://arxiv.org/pdf/2507.13373v1)**

> **作者:** Xiaojian Lin; Wenxin Zhang; Yuchu Jiang; Wangyu Wu; Yiran Guo; Kangxu Wang; Zongzheng Zhang; Guijin Wang; Lei Jin; Hao Zhao
>
> **备注:** 10 pages, 6 figures. Supplementary material: 8 pages, 7 figures. Accepted at ACM Multimedia 2025
>
> **摘要:** Hierarchical feature representations play a pivotal role in computer vision, particularly in object detection for autonomous driving. Multi-level semantic understanding is crucial for accurately identifying pedestrians, vehicles, and traffic signs in dynamic environments. However, existing architectures, such as YOLO and DETR, struggle to maintain feature consistency across different scales while balancing detection precision and computational efficiency. To address these challenges, we propose Butter, a novel object detection framework designed to enhance hierarchical feature representations for improving detection robustness. Specifically, Butter introduces two key innovations: Frequency-Adaptive Feature Consistency Enhancement (FAFCE) Component, which refines multi-scale feature consistency by leveraging adaptive frequency filtering to enhance structural and boundary precision, and Progressive Hierarchical Feature Fusion Network (PHFFNet) Module, which progressively integrates multi-level features to mitigate semantic gaps and strengthen hierarchical feature learning. Through extensive experiments on BDD100K, KITTI, and Cityscapes, Butter demonstrates superior feature representation capabilities, leading to notable improvements in detection accuracy while reducing model complexity. By focusing on hierarchical feature refinement and integration, Butter provides an advanced approach to object detection that achieves a balance between accuracy, deployability, and computational efficiency in real-time autonomous driving scenarios. Our model and implementation are publicly available at https://github.com/Aveiro-Lin/Butter, facilitating further research and validation within the autonomous driving community.
>
---
#### [new 074] Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于图像去模糊任务，旨在解决缺乏成对模糊-清晰图像的问题。通过使用扩散模型，从非配对数据中学习纹理先验知识，提出了一种新框架 \ours，包含纹理先验编码器和纹理传输变压器层，有效恢复图像细节，超越现有方法。**

- **链接: [http://arxiv.org/pdf/2507.13599v1](http://arxiv.org/pdf/2507.13599v1)**

> **作者:** Chengxu Liu; Lu Qi; Jinshan Pan; Xueming Qian; Ming-Hsuan Yang
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Since acquiring large amounts of realistic blurry-sharp image pairs is difficult and expensive, learning blind image deblurring from unpaired data is a more practical and promising solution. Unfortunately, dominant approaches rely heavily on adversarial learning to bridge the gap from blurry domains to sharp domains, ignoring the complex and unpredictable nature of real-world blur patterns. In this paper, we propose a novel diffusion model (DM)-based framework, dubbed \ours, for image deblurring by learning spatially varying texture prior from unpaired data. In particular, \ours performs DM to generate the prior knowledge that aids in recovering the textures of blurry images. To implement this, we propose a Texture Prior Encoder (TPE) that introduces a memory mechanism to represent the image textures and provides supervision for DM training. To fully exploit the generated texture priors, we present the Texture Transfer Transformer layer (TTformer), in which a novel Filter-Modulated Multi-head Self-Attention (FM-MSA) efficiently removes spatially varying blurring through adaptive filtering. Furthermore, we implement a wavelet-based adversarial loss to preserve high-frequency texture details. Extensive evaluations show that \ours provides a promising unsupervised deblurring solution and outperforms SOTA methods in widely-used benchmarks.
>
---
#### [new 075] OmniVec2 -- A Novel Transformer based Network for Large Scale Multimodal and Multitask Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态多任务学习任务，旨在解决大规模多模态数据的统一建模问题。作者提出了OmniVec2网络，采用模态专用分词器、共享Transformer和跨模态注意力机制，实现12种模态数据在统一嵌入空间中的处理。通过模态特定任务头支持多任务学习，并设计了迭代模态切换的预训练策略和多模态训练算法，最终在25个数据集上取得SOTA性能。**

- **链接: [http://arxiv.org/pdf/2507.13364v1](http://arxiv.org/pdf/2507.13364v1)**

> **作者:** Siddharth Srivastava; Gaurav Sharma
>
> **摘要:** We present a novel multimodal multitask network and associated training algorithm. The method is capable of ingesting data from approximately 12 different modalities namely image, video, audio, text, depth, point cloud, time series, tabular, graph, X-ray, infrared, IMU, and hyperspectral. The proposed approach utilizes modality specialized tokenizers, a shared transformer architecture, and cross-attention mechanisms to project the data from different modalities into a unified embedding space. It addresses multimodal and multitask scenarios by incorporating modality-specific task heads for different tasks in respective modalities. We propose a novel pretraining strategy with iterative modality switching to initialize the network, and a training algorithm which trades off fully joint training over all modalities, with training on pairs of modalities at a time. We provide comprehensive evaluation across 25 datasets from 12 modalities and show state of the art performances, demonstrating the effectiveness of the proposed architecture, pretraining strategy and adapted multitask training.
>
---
#### [new 076] Moodifier: MLLM-Enhanced Emotion-Driven Image Editing
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决基于情绪的图像编辑中情绪表达抽象、编辑困难的问题。论文构建了大规模情绪标注数据集MoodArchive，训练了情绪到视觉属性的模型MoodifyCLIP，并提出了无需训练的编辑框架Moodifier，实现跨领域的情绪化图像编辑。**

- **链接: [http://arxiv.org/pdf/2507.14024v1](http://arxiv.org/pdf/2507.14024v1)**

> **作者:** Jiarong Ye; Sharon X. Huang
>
> **摘要:** Bridging emotions and visual content for emotion-driven image editing holds great potential in creative industries, yet precise manipulation remains challenging due to the abstract nature of emotions and their varied manifestations across different contexts. We tackle this challenge with an integrated approach consisting of three complementary components. First, we introduce MoodArchive, an 8M+ image dataset with detailed hierarchical emotional annotations generated by LLaVA and partially validated by human evaluators. Second, we develop MoodifyCLIP, a vision-language model fine-tuned on MoodArchive to translate abstract emotions into specific visual attributes. Third, we propose Moodifier, a training-free editing model leveraging MoodifyCLIP and multimodal large language models (MLLMs) to enable precise emotional transformations while preserving content integrity. Our system works across diverse domains such as character expressions, fashion design, jewelry, and home d\'ecor, enabling creators to quickly visualize emotional variations while preserving identity and structure. Extensive experimental evaluations show that Moodifier outperforms existing methods in both emotional accuracy and content preservation, providing contextually appropriate edits. By linking abstract emotions to concrete visual changes, our solution unlocks new possibilities for emotional content creation in real-world applications. We will release the MoodArchive dataset, MoodifyCLIP model, and make the Moodifier code and demo publicly available upon acceptance.
>
---
#### [new 077] NoHumansRequired: Autonomous High-Quality Image Editing Triplet Mining
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于图像编辑数据挖掘任务，旨在解决生成高质量训练数据困难的问题。作者提出了一种全自动的三元组挖掘系统，无需人工干预，可跨领域生成高质量的图像编辑数据。论文还发布了包含358k三元组的开源数据集和一个开源微调模型。**

- **链接: [http://arxiv.org/pdf/2507.14119v1](http://arxiv.org/pdf/2507.14119v1)**

> **作者:** Maksim Kuprashevich; Grigorii Alekseenko; Irina Tolstykh; Georgii Fedorov; Bulat Suleimanov; Vladimir Dokholyan; Aleksandr Gordeev
>
> **摘要:** Recent advances in generative modeling enable image editing assistants that follow natural language instructions without additional user input. Their supervised training requires millions of triplets: original image, instruction, edited image. Yet mining pixel-accurate examples is hard. Each edit must affect only prompt-specified regions, preserve stylistic coherence, respect physical plausibility, and retain visual appeal. The lack of robust automated edit-quality metrics hinders reliable automation at scale. We present an automated, modular pipeline that mines high-fidelity triplets across domains, resolutions, instruction complexities, and styles. Built on public generative models and running without human intervention, our system uses a task-tuned Gemini validator to score instruction adherence and aesthetics directly, removing any need for segmentation or grounding models. Inversion and compositional bootstrapping enlarge the mined set by approximately 2.2x, enabling large-scale high-fidelity training data. By automating the most repetitive annotation steps, the approach allows a new scale of training without human labeling effort. To democratize research in this resource-intensive area, we release NHR-Edit: an open dataset of 358k high-quality triplets. In the largest cross-dataset evaluation, it surpasses all public alternatives. We also release Bagel-NHR-Edit, an open-source fine-tuned Bagel model, which achieves state-of-the-art metrics in our experiments.
>
---
#### [new 078] Moving Object Detection from Moving Camera Using Focus of Expansion Likelihood and Segmentation
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决从移动相机视角中分离动态与静态物体的问题。现有方法依赖光流，但在复杂场景中效果不佳。论文提出FoELS方法，结合光流与纹理信息，通过计算扩展焦点（FoE）并融合分割先验，提升动态物体检测效果，适用于复杂场景与相机旋转等情况。**

- **链接: [http://arxiv.org/pdf/2507.13628v1](http://arxiv.org/pdf/2507.13628v1)**

> **作者:** Masahiro Ogawa; Qi An; Atsushi Yamashita
>
> **备注:** 8 pages, 15 figures, RA-L submission
>
> **摘要:** Separating moving and static objects from a moving camera viewpoint is essential for 3D reconstruction, autonomous navigation, and scene understanding in robotics. Existing approaches often rely primarily on optical flow, which struggles to detect moving objects in complex, structured scenes involving camera motion. To address this limitation, we propose Focus of Expansion Likelihood and Segmentation (FoELS), a method based on the core idea of integrating both optical flow and texture information. FoELS computes the focus of expansion (FoE) from optical flow and derives an initial motion likelihood from the outliers of the FoE computation. This likelihood is then fused with a segmentation-based prior to estimate the final moving probability. The method effectively handles challenges including complex structured scenes, rotational camera motion, and parallel motion. Comprehensive evaluations on the DAVIS 2016 dataset and real-world traffic videos demonstrate its effectiveness and state-of-the-art performance.
>
---
#### [new 079] IConMark: Robust Interpretable Concept-Based Watermark For AI Images
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文属于图像水印任务，旨在解决AI生成图像真实性验证问题。提出IConMark方法，在生成过程中嵌入可解释语义水印，增强对对抗攻击的鲁棒性，同时支持人工验证。结合现有技术提升鲁棒性，并验证了其检测准确性与图像质量保持能力。**

- **链接: [http://arxiv.org/pdf/2507.13407v1](http://arxiv.org/pdf/2507.13407v1)**

> **作者:** Vinu Sankar Sadasivan; Mehrdad Saberi; Soheil Feizi
>
> **备注:** Accepted at ICLR 2025 Workshop on GenAI Watermarking (WMARK)
>
> **摘要:** With the rapid rise of generative AI and synthetic media, distinguishing AI-generated images from real ones has become crucial in safeguarding against misinformation and ensuring digital authenticity. Traditional watermarking techniques have shown vulnerabilities to adversarial attacks, undermining their effectiveness in the presence of attackers. We propose IConMark, a novel in-generation robust semantic watermarking method that embeds interpretable concepts into AI-generated images, as a first step toward interpretable watermarking. Unlike traditional methods, which rely on adding noise or perturbations to AI-generated images, IConMark incorporates meaningful semantic attributes, making it interpretable to humans and hence, resilient to adversarial manipulation. This method is not only robust against various image augmentations but also human-readable, enabling manual verification of watermarks. We demonstrate a detailed evaluation of IConMark's effectiveness, demonstrating its superiority in terms of detection accuracy and maintaining image quality. Moreover, IConMark can be combined with existing watermarking techniques to further enhance and complement its robustness. We introduce IConMark+SS and IConMark+TM, hybrid approaches combining IConMark with StegaStamp and TrustMark, respectively, to further bolster robustness against multiple types of image manipulations. Our base watermarking technique (IConMark) and its variants (+TM and +SS) achieve 10.8%, 14.5%, and 15.9% higher mean area under the receiver operating characteristic curve (AUROC) scores for watermark detection, respectively, compared to the best baseline on various datasets.
>
---
#### [new 080] DiViD: Disentangled Video Diffusion for Static-Dynamic Factorization
- **分类: cs.CV**

- **简介: 该论文属于视频生成与分解任务，旨在无监督地分离视频中的静态外观与动态内容。现有方法存在信息泄露与重建模糊问题。论文提出DiViD，首个端到端视频扩散框架，通过序列编码器与条件解码器实现静态-动态显式分解，引入多种归纳偏置与正则化策略，提升分解准确性与生成质量。**

- **链接: [http://arxiv.org/pdf/2507.13934v1](http://arxiv.org/pdf/2507.13934v1)**

> **作者:** Marzieh Gheisari; Auguste Genovesio
>
> **摘要:** Unsupervised disentanglement of static appearance and dynamic motion in video remains a fundamental challenge, often hindered by information leakage and blurry reconstructions in existing VAE- and GAN-based approaches. We introduce DiViD, the first end-to-end video diffusion framework for explicit static-dynamic factorization. DiViD's sequence encoder extracts a global static token from the first frame and per-frame dynamic tokens, explicitly removing static content from the motion code. Its conditional DDPM decoder incorporates three key inductive biases: a shared-noise schedule for temporal consistency, a time-varying KL-based bottleneck that tightens at early timesteps (compressing static information) and relaxes later (enriching dynamics), and cross-attention that routes the global static token to all frames while keeping dynamic tokens frame-specific. An orthogonality regularizer further prevents residual static-dynamic leakage. We evaluate DiViD on real-world benchmarks using swap-based accuracy and cross-leakage metrics. DiViD outperforms state-of-the-art sequential disentanglement methods: it achieves the highest swap-based joint accuracy, preserves static fidelity while improving dynamic transfer, and reduces average cross-leakage.
>
---
#### [new 081] C-DOG: Training-Free Multi-View Multi-Object Association in Dense Scenes Without Visual Feature via Connected δ-Overlap Graphs
- **分类: cs.CV**

- **简介: 该论文属于3D重建中的多视角多目标关联任务，旨在解决视觉特征不可靠或缺失时的目标匹配问题。作者提出了C-DOG方法，通过结合δ-重叠图建模和极线几何，实现无需训练的鲁棒目标关联，提升了在高密度、低重叠等复杂场景下的重建效果。**

- **链接: [http://arxiv.org/pdf/2507.14095v1](http://arxiv.org/pdf/2507.14095v1)**

> **作者:** Yung-Hong Sun; Ting-Hung Lin; Jiangang Chen; Hongrui Jiang; Yu Hen Hu
>
> **摘要:** Multi-view multi-object association is a fundamental step in 3D reconstruction pipelines, enabling consistent grouping of object instances across multiple camera views. Existing methods often rely on appearance features or geometric constraints such as epipolar consistency. However, these approaches can fail when objects are visually indistinguishable or observations are corrupted by noise. We propose C-DOG, a training-free framework that serves as an intermediate module bridging object detection (or pose estimation) and 3D reconstruction, without relying on visual features. It combines connected delta-overlap graph modeling with epipolar geometry to robustly associate detections across views. Each 2D observation is represented as a graph node, with edges weighted by epipolar consistency. A delta-neighbor-overlap clustering step identifies strongly consistent groups while tolerating noise and partial connectivity. To further improve robustness, we incorporate Interquartile Range (IQR)-based filtering and a 3D back-projection error criterion to eliminate inconsistent observations. Extensive experiments on synthetic benchmarks demonstrate that C-DOG outperforms geometry-based baselines and remains robust under challenging conditions, including high object density, without visual features, and limited camera overlap, making it well-suited for scalable 3D reconstruction in real-world scenarios.
>
---
#### [new 082] InSyn: Modeling Complex Interactions for Pedestrian Trajectory Prediction
- **分类: cs.CV**

- **简介: 该论文属于行人轨迹预测任务，旨在解决复杂交互场景下预测精度不足的问题。现有方法忽略特定交互模式，导致拥挤场景效果差。作者提出InSyn模型，结合Transformer与方向敏感行为建模，并引入SSOS训练策略，有效提升预测准确性，尤其在高密度场景中表现突出。**

- **链接: [http://arxiv.org/pdf/2507.13397v1](http://arxiv.org/pdf/2507.13397v1)**

> **作者:** Kaiyuan Zhai; Juan Chen; Chao Wang; Zeyi Xu
>
> **摘要:** Accurate pedestrian trajectory prediction is crucial for intelligent applications, yet it remains highly challenging due to the complexity of interactions among pedestrians. Previous methods have primarily relied on relative positions to model pedestrian interactions; however, they tend to overlook specific interaction patterns such as paired walking or conflicting behaviors, limiting the prediction accuracy in crowded scenarios. To address this issue, we propose InSyn (Interaction-Synchronization Network), a novel Transformer-based model that explicitly captures diverse interaction patterns (e.g., walking in sync or conflicting) while effectively modeling direction-sensitive social behaviors. Additionally, we introduce a training strategy termed Seq-Start of Seq (SSOS), designed to alleviate the common issue of initial-step divergence in numerical time-series prediction. Experiments on the ETH and UCY datasets demonstrate that our model outperforms recent baselines significantly, especially in high-density scenarios. Furthermore, the SSOS strategy proves effective in improving sequential prediction performance, reducing the initial-step prediction error by approximately 6.58%.
>
---
#### [new 083] Food safety trends across Europe: insights from the 392-million-entry CompreHensive European Food Safety (CHEFS) database
- **分类: cs.CY; cs.AI; cs.CV**

- **简介: 该论文旨在整合欧盟食品安全监测数据，解决数据分散、难以分析的问题。作者构建了包含392百万条记录的CHEFS数据库，统一管理农药残留、兽药残留和化学污染物数据，并分析2000至2024年趋势，为食品安全政策与研究提供支持。**

- **链接: [http://arxiv.org/pdf/2507.13802v1](http://arxiv.org/pdf/2507.13802v1)**

> **作者:** Nehir Kizililsoley; Floor van Meer; Osman Mutlu; Wouter F Hoenderdaal; Rosan G. Hobé; Wenjuan Mu; Arjen Gerssen; H. J. van der Fels-Klerx; Ákos Jóźwiak; Ioannis Manikas; Ali Hürriyetoǧlu; Bas H. M. van der Velden
>
> **摘要:** In the European Union, official food safety monitoring data collected by member states are submitted to the European Food Safety Authority (EFSA) and published on Zenodo. This data includes 392 million analytical results derived from over 15.2 million samples covering more than 4,000 different types of food products, offering great opportunities for artificial intelligence to analyze trends, predict hazards, and support early warning systems. However, the current format with data distributed across approximately 1000 files totaling several hundred gigabytes hinders accessibility and analysis. To address this, we introduce the CompreHensive European Food Safety (CHEFS) database, which consolidates EFSA monitoring data on pesticide residues, veterinary medicinal product residues, and chemical contaminants into a unified and structured dataset. We describe the creation and structure of the CHEFS database and demonstrate its potential by analyzing trends in European food safety monitoring data from 2000 to 2024. Our analyses explore changes in monitoring activities, the most frequently tested products, which products were most often non-compliant and which contaminants were most often found, and differences across countries. These findings highlight the CHEFS database as both a centralized data source and a strategic tool for guiding food safety policy, research, and regulation.
>
---
#### [new 084] A Novel APVD Steganography Technique Incorporating Pseudorandom Pixel Selection for Robust Image Security
- **分类: cs.CR; cs.CV; cs.MM; eess.IV; 68Q80; I.4.2**

- **简介: 该论文属于图像隐写任务，旨在解决APVD隐写方法中存在的“未使用块”问题，通过结合伪随机像素选择策略，提升安全性、嵌入容量和图像质量。实验表明，新方法在多种图像上表现优异，有效改善PSNR、UIQ和SSIM指标。**

- **链接: [http://arxiv.org/pdf/2507.13367v1](http://arxiv.org/pdf/2507.13367v1)**

> **作者:** Mehrab Hosain; Rajiv Kapoor
>
> **备注:** Accepted COMITCON 2023. Lecture Notes in Electrical Engineering, vol 1191. Springer
>
> **摘要:** Steganography is the process of embedding secret information discreetly within a carrier, ensuring secure exchange of confidential data. The Adaptive Pixel Value Differencing (APVD) steganography method, while effective, encounters certain challenges like the "unused blocks" issue. This problem can cause a decrease in security, compromise the embedding capacity, and lead to lower visual quality. This research presents a novel steganographic strategy that integrates APVD with pseudorandom pixel selection to effectively mitigate these issues. The results indicate that the new method outperforms existing techniques in aspects of security, data hiding capacity, and the preservation of image quality. Empirical results reveal that the combination of APVD with pseudorandom pixel selection significantly enhances key image quality metrics such as Peak Signal-to-Noise Ratio (PSNR), Universal Image Quality Index (UIQ), and Structural Similarity Index (SSIM), surpassing other contemporary methods in performance. The newly proposed method is versatile, able to handle a variety of cover and secret images in both color and grayscale, thereby ensuring secure data transmission without compromising the aesthetic quality of the image.
>
---
#### [new 085] Multiresolution local smoothness detection in non-uniformly sampled multivariate signals
- **分类: math.NA; cs.CV; cs.LG; cs.NA**

- **简介: 该论文属于信号处理任务，旨在解决非均匀采样多变量信号的局部正则性检测问题。作者提出一种基于samplet变换的线性时间算法，通过分析samplet系数的衰减特性，有效检测高维散乱数据的局部光滑性，并验证其在不同维度信号上的性能。**

- **链接: [http://arxiv.org/pdf/2507.13480v1](http://arxiv.org/pdf/2507.13480v1)**

> **作者:** Sara Avesani; Gianluca Giacchi; Michael Multerer
>
> **摘要:** Inspired by edge detection based on the decay behavior of wavelet coefficients, we introduce a (near) linear-time algorithm for detecting the local regularity in non-uniformly sampled multivariate signals. Our approach quantifies regularity within the framework of microlocal spaces introduced by Jaffard. The central tool in our analysis is the fast samplet transform, a distributional wavelet transform tailored to scattered data. We establish a connection between the decay of samplet coefficients and the pointwise regularity of multivariate signals. As a by product, we derive decay estimates for functions belonging to classical H\"older spaces and Sobolev-Slobodeckij spaces. While traditional wavelets are effective for regularity detection in low-dimensional structured data, samplets demonstrate robust performance even for higher dimensional and scattered data. To illustrate our theoretical findings, we present extensive numerical studies detecting local regularity of one-, two- and three-dimensional signals, ranging from non-uniformly sampled time series over image segmentation to edge detection in point clouds.
>
---
#### [new 086] Blind Super Resolution with Reference Images and Implicit Degradation Representation
- **分类: eess.IV; cs.CV**

- **简介: 论文属于盲超分辨率（BSR）任务，旨在解决传统方法中 degradation kernel 估计不准确且忽略缩放因子的问题。作者提出一种新方法，利用高分辨率参考图像，自适应地学习 degradation 过程并生成 LR-HR 对，提升超分辨率效果。该方法适用于多种 BSR 模型，表现优于先前方法。**

- **链接: [http://arxiv.org/pdf/2507.13915v1](http://arxiv.org/pdf/2507.13915v1)**

> **作者:** Huu-Phu Do; Po-Chih Hu; Hao-Chien Hsueh; Che-Kai Liu; Vu-Hoang Tran; Ching-Chun Huang
>
> **备注:** Accepted by ACCV 2024
>
> **摘要:** Previous studies in blind super-resolution (BSR) have primarily concentrated on estimating degradation kernels directly from low-resolution (LR) inputs to enhance super-resolution. However, these degradation kernels, which model the transition from a high-resolution (HR) image to its LR version, should account for not only the degradation process but also the downscaling factor. Applying the same degradation kernel across varying super-resolution scales may be impractical. Our research acknowledges degradation kernels and scaling factors as pivotal elements for the BSR task and introduces a novel strategy that utilizes HR images as references to establish scale-aware degradation kernels. By employing content-irrelevant HR reference images alongside the target LR image, our model adaptively discerns the degradation process. It is then applied to generate additional LR-HR pairs through down-sampling the HR reference images, which are keys to improving the SR performance. Our reference-based training procedure is applicable to proficiently trained blind SR models and zero-shot blind SR methods, consistently outperforming previous methods in both scenarios. This dual consideration of blur kernels and scaling factors, coupled with the use of a reference image, contributes to the effectiveness of our approach in blind super-resolution tasks.
>
---
#### [new 087] Software architecture and manual for novel versatile CT image analysis toolbox -- AnatomyArchive
- **分类: eess.IV; cs.CV; 62H35, 68U10; I.4.10; I.4.7; J.3**

- **简介: 该论文设计并开发了一款名为AnatomyArchive的CT图像分析工具箱，基于TotalSegmentator模型，旨在提升医学影像分析效率与精度。论文属于医学图像处理任务，解决自动分割、体积计算与特征提取等问题，实现知识图谱管理、自动裁剪、渲染及统计分析等功能，助力机器学习模型开发。**

- **链接: [http://arxiv.org/pdf/2507.13901v1](http://arxiv.org/pdf/2507.13901v1)**

> **作者:** Lei Xu; Torkel B Brismar
>
> **备注:** 24 pages, 7 figures
>
> **摘要:** We have developed a novel CT image analysis package named AnatomyArchive, built on top of the recent full body segmentation model TotalSegmentator. It provides automatic target volume selection and deselection capabilities according to user-configured anatomies for volumetric upper- and lower-bounds. It has a knowledge graph-based and time efficient tool for anatomy segmentation mask management and medical image database maintenance. AnatomyArchive enables automatic body volume cropping, as well as automatic arm-detection and exclusion, for more precise body composition analysis in both 2D and 3D formats. It provides robust voxel-based radiomic feature extraction, feature visualization, and an integrated toolchain for statistical tests and analysis. A python-based GPU-accelerated nearly photo-realistic segmentation-integrated composite cinematic rendering is also included. We present here its software architecture design, illustrate its workflow and working principle of algorithms as well provide a few examples on how the software can be used to assist development of modern machine learning models. Open-source codes will be released at https://github.com/lxu-medai/AnatomyArchive for only research and educational purposes.
>
---
#### [new 088] Domain-randomized deep learning for neuroimage analysis
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决深度学习模型在神经影像分析中泛化能力不足的问题。论文介绍了基于域随机化的深度学习方法，通过合成具有多样化强度和解剖内容的图像提升模型鲁棒性，并探讨了其应用效果与实际挑战。**

- **链接: [http://arxiv.org/pdf/2507.13458v1](http://arxiv.org/pdf/2507.13458v1)**

> **作者:** Malte Hoffmann
>
> **备注:** 12 pages, 6 figures, 2 tables, deep learning, domain generalization, domain randomization, neuroimaging, medical image analysis, accepted for publication in IEEE Signal Processing Magazine
>
> **摘要:** Deep learning has revolutionized neuroimage analysis by delivering unprecedented speed and accuracy. However, the narrow scope of many training datasets constrains model robustness and generalizability. This challenge is particularly acute in magnetic resonance imaging (MRI), where image appearance varies widely across pulse sequences and scanner hardware. A recent domain-randomization strategy addresses the generalization problem by training deep neural networks on synthetic images with randomized intensities and anatomical content. By generating diverse data from anatomical segmentation maps, the approach enables models to accurately process image types unseen during training, without retraining or fine-tuning. It has demonstrated effectiveness across modalities including MRI, computed tomography, positron emission tomography, and optical coherence tomography, as well as beyond neuroimaging in ultrasound, electron and fluorescence microscopy, and X-ray microtomography. This tutorial paper reviews the principles, implementation, and potential of the synthesis-driven training paradigm. It highlights key benefits, such as improved generalization and resistance to overfitting, while discussing trade-offs such as increased computational demands. Finally, the article explores practical considerations for adopting the technique, aiming to accelerate the development of generalizable tools that make deep learning more accessible to domain experts without extensive computational resources or machine learning knowledge.
>
---
#### [new 089] Convergent transformations of visual representation in brains and models
- **分类: q-bio.NC; cs.AI; cs.CV; eess.IV; I.2.10**

- **简介: 该论文研究视觉表征在大脑和深度神经网络中的收敛性转化，旨在揭示视觉感知是由外部世界结构还是大脑内部结构主导。通过结合跨被试相似性和模型层级对齐的统一框架，分析fMRI数据，发现大脑皮层存在保守的功能网络，并分为两个路径：内侧腹侧流处理场景结构，外侧背侧流处理社交和生物内容。深度视觉模型能捕捉这一组织，而语言模型不能，表明人工视觉系统可模拟人脑视觉处理机制。**

- **链接: [http://arxiv.org/pdf/2507.13941v1](http://arxiv.org/pdf/2507.13941v1)**

> **作者:** Pablo Marcos-Manchón; Lluís Fuentemilla
>
> **备注:** for associate code, see https://github.com/memory-formation/convergent-transformations
>
> **摘要:** A fundamental question in cognitive neuroscience is what shapes visual perception: the external world's structure or the brain's internal architecture. Although some perceptual variability can be traced to individual differences, brain responses to naturalistic stimuli evoke similar activity patterns across individuals, suggesting a convergent representational principle. Here, we test if this stimulus-driven convergence follows a common trajectory across people and deep neural networks (DNNs) during its transformation from sensory to high-level internal representations. We introduce a unified framework that traces representational flow by combining inter-subject similarity with alignment to model hierarchies. Applying this framework to three independent fMRI datasets of visual scene perception, we reveal a cortex-wide network, conserved across individuals, organized into two pathways: a medial-ventral stream for scene structure and a lateral-dorsal stream tuned for social and biological content. This functional organization is captured by the hierarchies of vision DNNs but not language models, reinforcing the specificity of the visual-to-semantic transformation. These findings show a convergent computational solution for visual encoding in both human and artificial vision, driven by the structure of the external world.
>
---
#### [new 090] Leveraging the Spatial Hierarchy: Coarse-to-fine Trajectory Generation via Cascaded Hybrid Diffusion
- **分类: cs.SI; cs.CV**

- **简介: 该论文属于轨迹生成任务，旨在解决隐私保护下高质量轨迹合成的问题。现有方法难以处理轨迹复杂结构，生成效果不佳。论文提出Cardiff框架，采用分层扩散模型，先在道路段层面生成粗粒度轨迹，再在GPS层面细化生成高保真轨迹，并实现隐私与效用的平衡。**

- **链接: [http://arxiv.org/pdf/2507.13366v1](http://arxiv.org/pdf/2507.13366v1)**

> **作者:** Baoshen Guo; Zhiqing Hong; Junyi Li; Shenhao Wang; Jinhua Zhao
>
> **摘要:** Urban mobility data has significant connections with economic growth and plays an essential role in various smart-city applications. However, due to privacy concerns and substantial data collection costs, fine-grained human mobility trajectories are difficult to become publicly available on a large scale. A promising solution to address this issue is trajectory synthesizing. However, existing works often ignore the inherent structural complexity of trajectories, unable to handle complicated high-dimensional distributions and generate realistic fine-grained trajectories. In this paper, we propose Cardiff, a coarse-to-fine Cascaded hybrid diffusion-based trajectory synthesizing framework for fine-grained and privacy-preserving mobility generation. By leveraging the hierarchical nature of urban mobility, Cardiff decomposes the generation process into two distinct levels, i.e., discrete road segment-level and continuous fine-grained GPS-level: (i) In the segment-level, to reduce computational costs and redundancy in raw trajectories, we first encode the discrete road segments into low-dimensional latent embeddings and design a diffusion transformer-based latent denoising network for segment-level trajectory synthesis. (ii) Taking the first stage of generation as conditions, we then design a fine-grained GPS-level conditional denoising network with a noise augmentation mechanism to achieve robust and high-fidelity generation. Additionally, the Cardiff framework not only progressively generates high-fidelity trajectories through cascaded denoising but also flexibly enables a tunable balance between privacy preservation and utility. Experimental results on three large real-world trajectory datasets demonstrate that our method outperforms state-of-the-art baselines in various metrics.
>
---
#### [new 091] TexGS-VolVis: Expressive Scene Editing for Volume Visualization via Textured Gaussian Splatting
- **分类: cs.GR; cs.CL; cs.CV**

- **简介: 该论文属于体积可视化任务，旨在解决现有方法风格迁移单一、灵活性差的问题。论文提出TexGS-VolVis，采用纹理高斯点框架，实现高质量、可控制的非真实感渲染与实时编辑。通过2D高斯扩展与图像/文本驱动编辑，提升了可视化效果与交互能力。**

- **链接: [http://arxiv.org/pdf/2507.13586v1](http://arxiv.org/pdf/2507.13586v1)**

> **作者:** Kaiyuan Tang; Kuangshi Ai; Jun Han; Chaoli Wang
>
> **备注:** Accepted by IEEE VIS 2025
>
> **摘要:** Advancements in volume visualization (VolVis) focus on extracting insights from 3D volumetric data by generating visually compelling renderings that reveal complex internal structures. Existing VolVis approaches have explored non-photorealistic rendering techniques to enhance the clarity, expressiveness, and informativeness of visual communication. While effective, these methods often rely on complex predefined rules and are limited to transferring a single style, restricting their flexibility. To overcome these limitations, we advocate the representation of VolVis scenes using differentiable Gaussian primitives combined with pretrained large models to enable arbitrary style transfer and real-time rendering. However, conventional 3D Gaussian primitives tightly couple geometry and appearance, leading to suboptimal stylization results. To address this, we introduce TexGS-VolVis, a textured Gaussian splatting framework for VolVis. TexGS-VolVis employs 2D Gaussian primitives, extending each Gaussian with additional texture and shading attributes, resulting in higher-quality, geometry-consistent stylization and enhanced lighting control during inference. Despite these improvements, achieving flexible and controllable scene editing remains challenging. To further enhance stylization, we develop image- and text-driven non-photorealistic scene editing tailored for TexGS-VolVis and 2D-lift-3D segmentation to enable partial editing with fine-grained control. We evaluate TexGS-VolVis both qualitatively and quantitatively across various volume rendering scenes, demonstrating its superiority over existing methods in terms of efficiency, visual quality, and editing flexibility.
>
---
#### [new 092] UGPL: Uncertainty-Guided Progressive Learning for Evidence-Based Classification in Computed Tomography
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决CT图像中病理特征微弱且分布多样的分类难题。论文提出了UGPL方法，通过不确定性引导的渐进学习框架，先进行全局分析，再聚焦关键区域，结合证据深度学习与自适应融合机制，提升分类准确性。**

- **链接: [http://arxiv.org/pdf/2507.14102v1](http://arxiv.org/pdf/2507.14102v1)**

> **作者:** Shravan Venkatraman; Pavan Kumar S; Rakesh Raj Madavan; Chandrakala S
>
> **备注:** 18 pages, 10 figures, 5 tables, 2025 ICCV Workshops
>
> **摘要:** Accurate classification of computed tomography (CT) images is essential for diagnosis and treatment planning, but existing methods often struggle with the subtle and spatially diverse nature of pathological features. Current approaches typically process images uniformly, limiting their ability to detect localized abnormalities that require focused analysis. We introduce UGPL, an uncertainty-guided progressive learning framework that performs a global-to-local analysis by first identifying regions of diagnostic ambiguity and then conducting detailed examination of these critical areas. Our approach employs evidential deep learning to quantify predictive uncertainty, guiding the extraction of informative patches through a non-maximum suppression mechanism that maintains spatial diversity. This progressive refinement strategy, combined with an adaptive fusion mechanism, enables UGPL to integrate both contextual information and fine-grained details. Experiments across three CT datasets demonstrate that UGPL consistently outperforms state-of-the-art methods, achieving improvements of 3.29%, 2.46%, and 8.08% in accuracy for kidney abnormality, lung cancer, and COVID-19 detection, respectively. Our analysis shows that the uncertainty-guided component provides substantial benefits, with performance dramatically increasing when the full progressive learning pipeline is implemented. Our code is available at: https://github.com/shravan-18/UGPL
>
---
#### [new 093] Divide and Conquer: A Large-Scale Dataset and Model for Left-Right Breast MRI Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决乳腺MRI中左右乳腺分割缺乏公开数据的问题。作者构建了包含13,000多例标注数据的公开数据集，并训练了用于左右乳腺分割的深度学习模型，为女性健康研究提供资源。**

- **链接: [http://arxiv.org/pdf/2507.13830v1](http://arxiv.org/pdf/2507.13830v1)**

> **作者:** Maximilian Rokuss; Benjamin Hamm; Yannick Kirchhoff; Klaus Maier-Hein
>
> **备注:** Accepted at MICCAI 2025 WOMEN
>
> **摘要:** We introduce the first publicly available breast MRI dataset with explicit left and right breast segmentation labels, encompassing more than 13,000 annotated cases. Alongside this dataset, we provide a robust deep-learning model trained for left-right breast segmentation. This work addresses a critical gap in breast MRI analysis and offers a valuable resource for the development of advanced tools in women's health. The dataset and trained model are publicly available at: www.github.com/MIC-DKFZ/BreastDivider
>
---
#### [new 094] Converting T1-weighted MRI from 3T to 7T quality using deep learning
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决3T MRI图像分辨率和细节不足的问题。通过深度学习模型（U-Net与GAN U-Net），从3T图像合成高质量7T MRI图像，提升视觉效果和分割精度，同时保持在下游任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.13782v1](http://arxiv.org/pdf/2507.13782v1)**

> **作者:** Malo Gicquel; Ruoyi Zhao; Anika Wuestefeld; Nicola Spotorno; Olof Strandberg; Kalle Åström; Yu Xiao; Laura EM Wisse; Danielle van Westen; Rik Ossenkoppele; Niklas Mattsson-Carlgren; David Berron; Oskar Hansson; Gabrielle Flood; Jacob Vogel
>
> **摘要:** Ultra-high resolution 7 tesla (7T) magnetic resonance imaging (MRI) provides detailed anatomical views, offering better signal-to-noise ratio, resolution and tissue contrast than 3T MRI, though at the cost of accessibility. We present an advanced deep learning model for synthesizing 7T brain MRI from 3T brain MRI. Paired 7T and 3T T1-weighted images were acquired from 172 participants (124 cognitively unimpaired, 48 impaired) from the Swedish BioFINDER-2 study. To synthesize 7T MRI from 3T images, we trained two models: a specialized U-Net, and a U-Net integrated with a generative adversarial network (GAN U-Net). Our models outperformed two additional state-of-the-art 3T-to-7T models in image-based evaluation metrics. Four blinded MRI professionals judged our synthetic 7T images as comparable in detail to real 7T images, and superior in subjective visual quality to 7T images, apparently due to the reduction of artifacts. Importantly, automated segmentations of the amygdalae of synthetic GAN U-Net 7T images were more similar to manually segmented amygdalae (n=20), than automated segmentations from the 3T images that were used to synthesize the 7T images. Finally, synthetic 7T images showed similar performance to real 3T images in downstream prediction of cognitive status using MRI derivatives (n=3,168). In all, we show that synthetic T1-weighted brain images approaching 7T quality can be generated from 3T images, which may improve image quality and segmentation, without compromising performance in downstream tasks. Future directions, possible clinical use cases, and limitations are discussed.
>
---
#### [new 095] Whose View of Safety? A Deep DIVE Dataset for Pluralistic Alignment of Text-to-Image Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于文本到图像（T2I）模型对齐任务，旨在解决模型在安全性和价值观方面缺乏多样化对齐的问题。作者构建了一个名为DIVE的多模态数据集，通过多维度人群的反馈，实现对不同安全观点的深度理解与引导，推动T2I系统更公平、更具包容性。**

- **链接: [http://arxiv.org/pdf/2507.13383v1](http://arxiv.org/pdf/2507.13383v1)**

> **作者:** Charvi Rastogi; Tian Huey Teh; Pushkar Mishra; Roma Patel; Ding Wang; Mark Díaz; Alicia Parrish; Aida Mostafazadeh Davani; Zoe Ashwood; Michela Paganini; Vinodkumar Prabhakaran; Verena Rieser; Lora Aroyo
>
> **备注:** 28 pages, 16 figures
>
> **摘要:** Current text-to-image (T2I) models often fail to account for diverse human experiences, leading to misaligned systems. We advocate for pluralistic alignment, where an AI understands and is steerable towards diverse, and often conflicting, human values. Our work provides three core contributions to achieve this in T2I models. First, we introduce a novel dataset for Diverse Intersectional Visual Evaluation (DIVE) -- the first multimodal dataset for pluralistic alignment. It enable deep alignment to diverse safety perspectives through a large pool of demographically intersectional human raters who provided extensive feedback across 1000 prompts, with high replication, capturing nuanced safety perceptions. Second, we empirically confirm demographics as a crucial proxy for diverse viewpoints in this domain, revealing significant, context-dependent differences in harm perception that diverge from conventional evaluations. Finally, we discuss implications for building aligned T2I models, including efficient data collection strategies, LLM judgment capabilities, and model steerability towards diverse perspectives. This research offers foundational tools for more equitable and aligned T2I systems. Content Warning: The paper includes sensitive content that may be harmful.
>
---
#### [new 096] Improving Out-of-distribution Human Activity Recognition via IMU-Video Cross-modal Representation Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于人体活动识别（HAR）任务，旨在提升在不同环境或人群中基于IMU数据的模型泛化能力。现有方法依赖特定标签且泛化性差，为此，论文提出一种基于IMU与视频数据的跨模态自监督预训练方法，有效提升了在分布外（OOD）数据上的识别表现，特别是在帕金森患者数据集上。**

- **链接: [http://arxiv.org/pdf/2507.13482v1](http://arxiv.org/pdf/2507.13482v1)**

> **作者:** Seyyed Saeid Cheshmi; Buyao Lyu; Thomas Lisko; Rajesh Rajamani; Robert A. McGovern; Yogatheesan Varatharajah
>
> **摘要:** Human Activity Recognition (HAR) based on wearable inertial sensors plays a critical role in remote health monitoring. In patients with movement disorders, the ability to detect abnormal patient movements in their home environments can enable continuous optimization of treatments and help alert caretakers as needed. Machine learning approaches have been proposed for HAR tasks using Inertial Measurement Unit (IMU) data; however, most rely on application-specific labels and lack generalizability to data collected in different environments or populations. To address this limitation, we propose a new cross-modal self-supervised pretraining approach to learn representations from large-sale unlabeled IMU-video data and demonstrate improved generalizability in HAR tasks on out of distribution (OOD) IMU datasets, including a dataset collected from patients with Parkinson's disease. Specifically, our results indicate that the proposed cross-modal pretraining approach outperforms the current state-of-the-art IMU-video pretraining approach and IMU-only pretraining under zero-shot and few-shot evaluations. Broadly, our study provides evidence that in highly dynamic data modalities, such as IMU signals, cross-modal pretraining may be a useful tool to learn generalizable data representations. Our software is available at https://github.com/scheshmi/IMU-Video-OOD-HAR.
>
---
#### [new 097] GIFT: Gradient-aware Immunization of diffusion models against malicious Fine-Tuning with safe concepts retention
- **分类: cs.CR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于安全增强任务，旨在解决扩散模型易受恶意微调攻击的问题。作者提出GIFT方法，通过双层优化框架，在破坏有害概念表示的同时保留安全内容生成能力，提升模型对恶意微调的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.13598v1](http://arxiv.org/pdf/2507.13598v1)**

> **作者:** Amro Abdalla; Ismail Shaheen; Dan DeGenaro; Rupayan Mallick; Bogdan Raita; Sarah Adel Bargal
>
> **备注:** Warning: This paper contains NSFW content. Reader discretion is advised
>
> **摘要:** We present GIFT: a {G}radient-aware {I}mmunization technique to defend diffusion models against malicious {F}ine-{T}uning while preserving their ability to generate safe content. Existing safety mechanisms like safety checkers are easily bypassed, and concept erasure methods fail under adversarial fine-tuning. GIFT addresses this by framing immunization as a bi-level optimization problem: the upper-level objective degrades the model's ability to represent harmful concepts using representation noising and maximization, while the lower-level objective preserves performance on safe data. GIFT achieves robust resistance to malicious fine-tuning while maintaining safe generative quality. Experimental results show that our method significantly impairs the model's ability to re-learn harmful concepts while maintaining performance on safe content, offering a promising direction for creating inherently safer generative models resistant to adversarial fine-tuning attacks.
>
---
#### [new 098] Leveraging Pathology Foundation Models for Panoptic Segmentation of Melanoma in H&E Images
- **分类: eess.IV; cs.CV; q-bio.QM**

- **简介: 该论文属于医学图像分割任务，旨在解决黑色素瘤组织自动分割问题。为提升准确性和效率，研究使用病理基础模型Virchow2提取特征，并与RGB图像融合，通过Efficient-UNet网络实现五类组织的全景分割，取得了优异性能。**

- **链接: [http://arxiv.org/pdf/2507.13974v1](http://arxiv.org/pdf/2507.13974v1)**

> **作者:** Jiaqi Lv; Yijie Zhu; Carmen Guadalupe Colin Tenorio; Brinder Singh Chohan; Mark Eastwood; Shan E Ahmed Raza
>
> **备注:** Accepted by MIUA 2025
>
> **摘要:** Melanoma is an aggressive form of skin cancer with rapid progression and high metastatic potential. Accurate characterisation of tissue morphology in melanoma is crucial for prognosis and treatment planning. However, manual segmentation of tissue regions from haematoxylin and eosin (H&E) stained whole-slide images (WSIs) is labour-intensive and prone to inter-observer variability, this motivates the need for reliable automated tissue segmentation methods. In this study, we propose a novel deep learning network for the segmentation of five tissue classes in melanoma H&E images. Our approach leverages Virchow2, a pathology foundation model trained on 3.1 million histopathology images as a feature extractor. These features are fused with the original RGB images and subsequently processed by an encoder-decoder segmentation network (Efficient-UNet) to produce accurate segmentation maps. The proposed model achieved first place in the tissue segmentation task of the PUMA Grand Challenge, demonstrating robust performance and generalizability. Our results show the potential and efficacy of incorporating pathology foundation models into segmentation networks to accelerate computational pathology workflows.
>
---
#### [new 099] Generative AI-Driven High-Fidelity Human Motion Simulation
- **分类: cs.AI; cs.CV**

- **简介: 论文提出G-AI-HMS系统，属于人机运动仿真任务，旨在提升工业任务中人体运动模拟的真实性。通过结合文本生成与动作生成模型，解决传统方法运动逼真度低的问题，并利用计算机视觉验证生成动作的质量。**

- **链接: [http://arxiv.org/pdf/2507.14097v1](http://arxiv.org/pdf/2507.14097v1)**

> **作者:** Hari Iyer; Neel Macwan; Atharva Jitendra Hude; Heejin Jeong; Shenghan Guo
>
> **摘要:** Human motion simulation (HMS) supports cost-effective evaluation of worker behavior, safety, and productivity in industrial tasks. However, existing methods often suffer from low motion fidelity. This study introduces Generative-AI-Enabled HMS (G-AI-HMS), which integrates text-to-text and text-to-motion models to enhance simulation quality for physical tasks. G-AI-HMS tackles two key challenges: (1) translating task descriptions into motion-aware language using Large Language Models aligned with MotionGPT's training vocabulary, and (2) validating AI-enhanced motions against real human movements using computer vision. Posture estimation algorithms are applied to real-time videos to extract joint landmarks, and motion similarity metrics are used to compare them with AI-enhanced sequences. In a case study involving eight tasks, the AI-enhanced motions showed lower error than human created descriptions in most scenarios, performing better in six tasks based on spatial accuracy, four tasks based on alignment after pose normalization, and seven tasks based on overall temporal similarity. Statistical analysis showed that AI-enhanced prompts significantly (p $<$ 0.0001) reduced joint error and temporal misalignment while retaining comparable posture accuracy.
>
---
#### [new 100] StructInbet: Integrating Explicit Structural Guidance into Inbetween Frame Generation
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决中间帧生成中像素轨迹模糊和角色外观不一致的问题。论文提出StructInbet，引入显式结构引导以减少模糊性，并采用时间注意力机制融合前后关键帧的视觉信息，提升生成帧的结构准确性和外观一致性。**

- **链接: [http://arxiv.org/pdf/2507.13377v1](http://arxiv.org/pdf/2507.13377v1)**

> **作者:** Zhenglin Pan; Haoran Xie
>
> **备注:** 3 pages, 3 figures. SIGGRAPH 2025 Poster
>
> **摘要:** In this paper, we propose StructInbet, an inbetweening system designed to generate controllable transitions over explicit structural guidance. StructInbet introduces two key contributions. First, we propose explicit structural guidance to the inbetweening problem to reduce the ambiguity inherent in pixel trajectories. Second, we adopt a temporal attention mechanism that incorporates visual identity from both the preceding and succeeding keyframes, ensuring consistency in character appearance.
>
---
#### [new 101] Cross-modal Causal Intervention for Alzheimer's Disease Prediction
- **分类: cs.AI; cs.CV; cs.MM**

- **简介: 该论文属于医学影像与语言处理的跨模态学习任务，旨在解决阿尔茨海默病（AD）早期诊断中因数据偏差和混杂因素导致的不可靠预测问题。作者提出ADPC框架，结合MRI、fMRI和临床文本数据，利用因果干预消除混杂因素，提升AD、MCI和认知正常（CN）分类的准确性。**

- **链接: [http://arxiv.org/pdf/2507.13956v1](http://arxiv.org/pdf/2507.13956v1)**

> **作者:** Yutao Jin; Haowen Xiao; Jielei Chu; Fengmao Lv; Yuxiao Li; Tianrui Li
>
> **摘要:** Mild Cognitive Impairment (MCI) serves as a prodromal stage of Alzheimer's Disease (AD), where early identification and intervention can effectively slow the progression to dementia. However, diagnosing AD remains a significant challenge in neurology due to the confounders caused mainly by the selection bias of multimodal data and the complex relationships between variables. To address these issues, we propose a novel visual-language causal intervention framework named Alzheimer's Disease Prediction with Cross-modal Causal Intervention (ADPC) for diagnostic assistance. Our ADPC employs large language model (LLM) to summarize clinical data under strict templates, maintaining structured text outputs even with incomplete or unevenly distributed datasets. The ADPC model utilizes Magnetic Resonance Imaging (MRI), functional MRI (fMRI) images and textual data generated by LLM to classify participants into Cognitively Normal (CN), MCI, and AD categories. Because of the presence of confounders, such as neuroimaging artifacts and age-related biomarkers, non-causal models are likely to capture spurious input-output correlations, generating less reliable results. Our framework implicitly eliminates confounders through causal intervention. Experimental results demonstrate the outstanding performance of our method in distinguishing CN/MCI/AD cases, achieving state-of-the-art (SOTA) metrics across most evaluation metrics. The study showcases the potential of integrating causal reasoning with multi-modal learning for neurological disease diagnosis.
>
---
#### [new 102] OrthoInsight: Rib Fracture Diagnosis and Report Generation Based on Multi-Modal Large Models
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出OrthoInsight，一个基于多模态大模型的肋骨骨折诊断与报告生成系统。它属于医学图像分析任务，旨在解决人工解读CT图像效率低且易出错的问题。论文工作包括整合YOLOv9骨折检测、医学知识图谱和微调LLaVA语言模型，以实现高效、准确的诊断与报告生成。**

- **链接: [http://arxiv.org/pdf/2507.13993v1](http://arxiv.org/pdf/2507.13993v1)**

> **作者:** Ningyong Wu; Jinzhi Wang; Wenhong Zhao; Chenzhan Yu; Zhigang Xiu; Duwei Dai
>
> **摘要:** The growing volume of medical imaging data has increased the need for automated diagnostic tools, especially for musculoskeletal injuries like rib fractures, commonly detected via CT scans. Manual interpretation is time-consuming and error-prone. We propose OrthoInsight, a multi-modal deep learning framework for rib fracture diagnosis and report generation. It integrates a YOLOv9 model for fracture detection, a medical knowledge graph for retrieving clinical context, and a fine-tuned LLaVA language model for generating diagnostic reports. OrthoInsight combines visual features from CT images with expert textual data to deliver clinically useful outputs. Evaluated on 28,675 annotated CT images and expert reports, it achieves high performance across Diagnostic Accuracy, Content Completeness, Logical Coherence, and Clinical Guidance Value, with an average score of 4.28, outperforming models like GPT-4 and Claude-3. This study demonstrates the potential of multi-modal learning in transforming medical image analysis and providing effective support for radiologists.
>
---
#### [new 103] Flatten Wisely: How Patch Order Shapes Mamba-Powered Vision for MRI Segmentation
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 论文研究视觉Mamba模型在MRI分割任务中的表现，探讨图像块扫描顺序对结果的影响。提出MS2D模块，评估21种扫描策略，发现空间连续路径效果更优，证明扫描顺序是影响性能的重要因素。**

- **链接: [http://arxiv.org/pdf/2507.13384v1](http://arxiv.org/pdf/2507.13384v1)**

> **作者:** Osama Hardan; Omar Elshenhabi; Tamer Khattab; Mohamed Mabrok
>
> **备注:** Submitted to the 2025 IEEE International Conference on Future Machine Learning and Data Science (FMLDS)
>
> **摘要:** Vision Mamba models promise transformer-level performance at linear computational cost, but their reliance on serializing 2D images into 1D sequences introduces a critical, yet overlooked, design choice: the patch scan order. In medical imaging, where modalities like brain MRI contain strong anatomical priors, this choice is non-trivial. This paper presents the first systematic study of how scan order impacts MRI segmentation. We introduce Multi-Scan 2D (MS2D), a parameter-free module for Mamba-based architectures that facilitates exploring diverse scan paths without additional computational cost. We conduct a large-scale benchmark of 21 scan strategies on three public datasets (BraTS 2020, ISLES 2022, LGG), covering over 70,000 slices. Our analysis shows conclusively that scan order is a statistically significant factor (Friedman test: $\chi^{2}_{20}=43.9, p=0.0016$), with performance varying by as much as 27 Dice points. Spatially contiguous paths -- simple horizontal and vertical rasters -- consistently outperform disjointed diagonal scans. We conclude that scan order is a powerful, cost-free hyperparameter, and provide an evidence-based shortlist of optimal paths to maximize the performance of Mamba models in medical imaging.
>
---
#### [new 104] BreastSegNet: Multi-label Segmentation of Breast MRI
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决乳腺MRI中多种解剖结构的多标签分割问题。现有方法覆盖结构有限，作者提出BreastSegNet算法，覆盖9种结构，并标注了1123个MRI切片。他们评估了9种模型，其中nnU-Net ResEncM表现最佳，平均Dice得分为0.694。代码和权重已公开。**

- **链接: [http://arxiv.org/pdf/2507.13604v1](http://arxiv.org/pdf/2507.13604v1)**

> **作者:** Qihang Li; Jichen Yang; Yaqian Chen; Yuwen Chen; Hanxue Gu; Lars J. Grimm; Maciej A. Mazurowski
>
> **摘要:** Breast MRI provides high-resolution imaging critical for breast cancer screening and preoperative staging. However, existing segmentation methods for breast MRI remain limited in scope, often focusing on only a few anatomical structures, such as fibroglandular tissue or tumors, and do not cover the full range of tissues seen in scans. This narrows their utility for quantitative analysis. In this study, we present BreastSegNet, a multi-label segmentation algorithm for breast MRI that covers nine anatomical labels: fibroglandular tissue (FGT), vessel, muscle, bone, lesion, lymph node, heart, liver, and implant. We manually annotated a large set of 1123 MRI slices capturing these structures with detailed review and correction from an expert radiologist. Additionally, we benchmark nine segmentation models, including U-Net, SwinUNet, UNet++, SAM, MedSAM, and nnU-Net with multiple ResNet-based encoders. Among them, nnU-Net ResEncM achieves the highest average Dice scores of 0.694 across all labels. It performs especially well on heart, liver, muscle, FGT, and bone, with Dice scores exceeding 0.73, and approaching 0.90 for heart and liver. All model code and weights are publicly available, and we plan to release the data at a later date.
>
---
#### [new 105] Neural Architecture Search with Mixed Bio-inspired Learning Rules
- **分类: cs.NE; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于神经网络架构搜索任务，旨在解决生物启发神经网络在准确性和可扩展性上落后于反向传播模型的问题。通过设计混合生物启发学习规则并结合神经架构搜索，自动发现不同层使用不同学习规则的网络结构，从而提升了生物启发模型的准确性和可扩展性。**

- **链接: [http://arxiv.org/pdf/2507.13485v1](http://arxiv.org/pdf/2507.13485v1)**

> **作者:** Imane Hamzaoui; Riyadh Baghdadi
>
> **备注:** ECAI 2025
>
> **摘要:** Bio-inspired neural networks are attractive for their adversarial robustness, energy frugality, and closer alignment with cortical physiology, yet they often lag behind back-propagation (BP) based models in accuracy and ability to scale. We show that allowing the use of different bio-inspired learning rules in different layers, discovered automatically by a tailored neural-architecture-search (NAS) procedure, bridges this gap. Starting from standard NAS baselines, we enlarge the search space to include bio-inspired learning rules and use NAS to find the best architecture and learning rule to use in each layer. We show that neural networks that use different bio-inspired learning rules for different layers have better accuracy than those that use a single rule across all the layers. The resulting NN that uses a mix of bio-inspired learning rules sets new records for bio-inspired models: 95.16% on CIFAR-10, 76.48% on CIFAR-100, 43.42% on ImageNet16-120, and 60.51% top-1 on ImageNet. In some regimes, they even surpass comparable BP-based networks while retaining their robustness advantages. Our results suggest that layer-wise diversity in learning rules allows better scalability and accuracy, and motivates further research on mixing multiple bio-inspired learning rules in the same network.
>
---
#### [new 106] Enhanced DeepLab Based Nerve Segmentation with Optimized Tuning
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决超声神经图像中神经结构精准识别问题。作者基于DeepLabV3模型构建分割流程，引入自动阈值微调方法，优化预处理和参数选择，提升了分割效果，在多项指标上取得显著改进。**

- **链接: [http://arxiv.org/pdf/2507.13394v1](http://arxiv.org/pdf/2507.13394v1)**

> **作者:** Akhil John Thomas; Christiaan Boerkamp
>
> **摘要:** Nerve segmentation is crucial in medical imaging for precise identification of nerve structures. This study presents an optimized DeepLabV3-based segmentation pipeline that incorporates automated threshold fine-tuning to improve segmentation accuracy. By refining preprocessing steps and implementing parameter optimization, we achieved a Dice Score of 0.78, an IoU of 0.70, and a Pixel Accuracy of 0.95 on ultrasound nerve imaging. The results demonstrate significant improvements over baseline models and highlight the importance of tailored parameter selection in automated nerve detection.
>
---
#### [new 107] D2IP: Deep Dynamic Image Prior for 3D Time-sequence Pulmonary Impedance Imaging
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于3D时间序列肺部阻抗成像任务，旨在解决无监督学习方法在该领域计算成本高、收敛慢的问题。论文提出了D2IP框架，结合参数热启动、时序传播和轻量网络结构，实现了更快速、准确且时间连续的成像重建。**

- **链接: [http://arxiv.org/pdf/2507.14046v1](http://arxiv.org/pdf/2507.14046v1)**

> **作者:** Hao Fang; Hao Yu; Sihao Teng; Tao Zhang; Siyi Yuan; Huaiwu He; Zhe Liu; Yunjie Yang
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** Unsupervised learning methods, such as Deep Image Prior (DIP), have shown great potential in tomographic imaging due to their training-data-free nature and high generalization capability. However, their reliance on numerous network parameter iterations results in high computational costs, limiting their practical application, particularly in complex 3D or time-sequence tomographic imaging tasks. To overcome these challenges, we propose Deep Dynamic Image Prior (D2IP), a novel framework for 3D time-sequence imaging. D2IP introduces three key strategies - Unsupervised Parameter Warm-Start (UPWS), Temporal Parameter Propagation (TPP), and a customized lightweight reconstruction backbone, 3D-FastResUNet - to accelerate convergence, enforce temporal coherence, and improve computational efficiency. Experimental results on both simulated and clinical pulmonary datasets demonstrate that D2IP enables fast and accurate 3D time-sequence Electrical Impedance Tomography (tsEIT) reconstruction. Compared to state-of-the-art baselines, D2IP delivers superior image quality, with a 24.8% increase in average MSSIM and an 8.1% reduction in ERR, alongside significantly reduced computational time (7.1x faster), highlighting its promise for clinical dynamic pulmonary imaging.
>
---
#### [new 108] Safety Certification in the Latent space using Control Barrier Functions and World Models
- **分类: cs.RO; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 该论文属于安全控制任务，旨在解决从视觉数据中合成安全控制器时依赖大量标注安全关键数据的问题。论文提出了一种半监督框架，结合控制屏障函数与世界模型，在潜在空间中学习安全控制器，减少对标注数据的依赖，提升安全控制的可扩展性与数据效率。**

- **链接: [http://arxiv.org/pdf/2507.13871v1](http://arxiv.org/pdf/2507.13871v1)**

> **作者:** Mehul Anand; Shishir Kolathaya
>
> **备注:** 6 pages, 6 figures. arXiv admin note: text overlap with arXiv:2409.12616
>
> **摘要:** Synthesising safe controllers from visual data typically requires extensive supervised labelling of safety-critical data, which is often impractical in real-world settings. Recent advances in world models enable reliable prediction in latent spaces, opening new avenues for scalable and data-efficient safe control. In this work, we introduce a semi-supervised framework that leverages control barrier certificates (CBCs) learned in the latent space of a world model to synthesise safe visuomotor policies. Our approach jointly learns a neural barrier function and a safe controller using limited labelled data, while exploiting the predictive power of modern vision transformers for latent dynamics modelling.
>
---
#### [new 109] Generalist Bimanual Manipulation via Foundation Video Diffusion Models
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于机器人双臂操作任务，旨在解决数据稀缺和异构性问题。论文提出了VIDAR框架，结合视频扩散模型与掩码逆动力学模型，利用多视角视频预训练，实现跨任务和背景的动作预测。使用少量人类示范，即可在新任务中取得良好泛化效果。**

- **链接: [http://arxiv.org/pdf/2507.12898v1](http://arxiv.org/pdf/2507.12898v1)**

> **作者:** Yao Feng; Hengkai Tan; Xinyi Mao; Guodong Liu; Shuhe Huang; Chendong Xiang; Hang Su; Jun Zhu
>
> **摘要:** Bimanual robotic manipulation, which involves the coordinated control of two robotic arms, is foundational for solving challenging tasks. Despite recent progress in general-purpose manipulation, data scarcity and embodiment heterogeneity remain serious obstacles to further scaling up in bimanual settings. In this paper, we introduce VIdeo Diffusion for Action Reasoning (VIDAR), a two-stage framework that leverages large-scale, diffusion-based video pre-training and a novel masked inverse dynamics model for action prediction. We pre-train the video diffusion model on 750K multi-view videos from three real-world bimanual robot platforms, utilizing a unified observation space that encodes robot, camera, task, and scene contexts. Our masked inverse dynamics model learns masks to extract action-relevant information from generated trajectories without requiring pixel-level labels, and the masks can effectively generalize to unseen backgrounds. Our experiments demonstrate that with only 20 minutes of human demonstrations on an unseen robot platform (only 1% of typical data requirements), VIDAR generalizes to unseen tasks and backgrounds with strong semantic understanding, surpassing state-of-the-art methods. Our findings highlight the potential of video foundation models, coupled with masked action prediction, to enable scalable and generalizable robotic manipulation in diverse real-world settings.
>
---
## 更新

#### [replaced 001] Align Your Rhythm: Generating Highly Aligned Dance Poses with Gating-Enhanced Rhythm-Aware Feature Representation
- **分类: cs.MM; cs.AI; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.17340v2](http://arxiv.org/pdf/2503.17340v2)**

> **作者:** Congyi Fan; Jian Guan; Xuanjia Zhao; Dongli Xu; Youtian Lin; Tong Ye; Pengming Feng; Haiwei Pan
>
> **备注:** ICCV 2025 Accept, Project page: https://danceba.github.io/
>
> **摘要:** Automatically generating natural, diverse and rhythmic human dance movements driven by music is vital for virtual reality and film industries. However, generating dance that naturally follows music remains a challenge, as existing methods lack proper beat alignment and exhibit unnatural motion dynamics. In this paper, we propose Danceba, a novel framework that leverages gating mechanism to enhance rhythm-aware feature representation for music-driven dance generation, which achieves highly aligned dance poses with enhanced rhythmic sensitivity. Specifically, we introduce Phase-Based Rhythm Extraction (PRE) to precisely extract rhythmic information from musical phase data, capitalizing on the intrinsic periodicity and temporal structures of music. Additionally, we propose Temporal-Gated Causal Attention (TGCA) to focus on global rhythmic features, ensuring that dance movements closely follow the musical rhythm. We also introduce Parallel Mamba Motion Modeling (PMMM) architecture to separately model upper and lower body motions along with musical features, thereby improving the naturalness and diversity of generated dance movements. Extensive experiments confirm that Danceba outperforms state-of-the-art methods, achieving significantly better rhythmic alignment and motion diversity. Project page: https://danceba.github.io/ .
>
---
#### [replaced 002] OD-VIRAT: A Large-Scale Benchmark for Object Detection in Realistic Surveillance Environments
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.12396v2](http://arxiv.org/pdf/2507.12396v2)**

> **作者:** Hayat Ullah; Abbas Khan; Arslan Munir; Hari Kalva
>
> **备注:** 14 pages
>
> **摘要:** Realistic human surveillance datasets are crucial for training and evaluating computer vision models under real-world conditions, facilitating the development of robust algorithms for human and human-interacting object detection in complex environments. These datasets need to offer diverse and challenging data to enable a comprehensive assessment of model performance and the creation of more reliable surveillance systems for public safety. To this end, we present two visual object detection benchmarks named OD-VIRAT Large and OD-VIRAT Tiny, aiming at advancing visual understanding tasks in surveillance imagery. The video sequences in both benchmarks cover 10 different scenes of human surveillance recorded from significant height and distance. The proposed benchmarks offer rich annotations of bounding boxes and categories, where OD-VIRAT Large has 8.7 million annotated instances in 599,996 images and OD-VIRAT Tiny has 288,901 annotated instances in 19,860 images. This work also focuses on benchmarking state-of-the-art object detection architectures, including RETMDET, YOLOX, RetinaNet, DETR, and Deformable-DETR on this object detection-specific variant of VIRAT dataset. To the best of our knowledge, it is the first work to examine the performance of these recently published state-of-the-art object detection architectures on realistic surveillance imagery under challenging conditions such as complex backgrounds, occluded objects, and small-scale objects. The proposed benchmarking and experimental settings will help in providing insights concerning the performance of selected object detection models and set the base for developing more efficient and robust object detection architectures.
>
---
#### [replaced 003] GaVS: 3D-Grounded Video Stabilization via Temporally-Consistent Local Reconstruction and Rendering
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23957v2](http://arxiv.org/pdf/2506.23957v2)**

> **作者:** Zinuo You; Stamatios Georgoulis; Anpei Chen; Siyu Tang; Dengxin Dai
>
> **备注:** siggraph 2025, project website: https://sinoyou.github.io/gavs. version 2, update discussion
>
> **摘要:** Video stabilization is pivotal for video processing, as it removes unwanted shakiness while preserving the original user motion intent. Existing approaches, depending on the domain they operate, suffer from several issues (e.g. geometric distortions, excessive cropping, poor generalization) that degrade the user experience. To address these issues, we introduce \textbf{GaVS}, a novel 3D-grounded approach that reformulates video stabilization as a temporally-consistent `local reconstruction and rendering' paradigm. Given 3D camera poses, we augment a reconstruction model to predict Gaussian Splatting primitives, and finetune it at test-time, with multi-view dynamics-aware photometric supervision and cross-frame regularization, to produce temporally-consistent local reconstructions. The model are then used to render each stabilized frame. We utilize a scene extrapolation module to avoid frame cropping. Our method is evaluated on a repurposed dataset, instilled with 3D-grounded information, covering samples with diverse camera motions and scene dynamics. Quantitatively, our method is competitive with or superior to state-of-the-art 2D and 2.5D approaches in terms of conventional task metrics and new geometry consistency. Qualitatively, our method produces noticeably better results compared to alternatives, validated by the user study.
>
---
#### [replaced 004] CorMulT: A Semi-supervised Modality Correlation-aware Multimodal Transformer for Sentiment Analysis
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.07046v3](http://arxiv.org/pdf/2407.07046v3)**

> **作者:** Yangmin Li; Ruiqi Zhu; Wengen Li
>
> **摘要:** Multimodal sentiment analysis is an active research area that combines multiple data modalities, e.g., text, image and audio, to analyze human emotions and benefits a variety of applications. Existing multimodal sentiment analysis methods can be classified as modality interaction-based methods, modality transformation-based methods and modality similarity-based methods. However, most of these methods highly rely on the strong correlations between modalities, and cannot fully uncover and utilize the correlations between modalities to enhance sentiment analysis. Therefore, these methods usually achieve bad performance for identifying the sentiment of multimodal data with weak correlations. To address this issue, we proposed a two-stage semi-supervised model termed Correlation-aware Multimodal Transformer (CorMulT) which consists pre-training stage and prediction stage. At the pre-training stage, a modality correlation contrastive learning module is designed to efficiently learn modality correlation coefficients between different modalities. At the prediction stage, the learned correlation coefficients are fused with modality representations to make the sentiment prediction. According to the experiments on the popular multimodal dataset CMU-MOSEI, CorMulT obviously surpasses state-of-the-art multimodal sentiment analysis methods.
>
---
#### [replaced 005] Progressively Exploring and Exploiting Cost-Free Data to Break Fine-Grained Classification Barriers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.20383v2](http://arxiv.org/pdf/2412.20383v2)**

> **作者:** Li-Jun Zhao; Zhen-Duo Chen; Zhi-Yuan Xue; Xin Luo; Xin-Shun Xu
>
> **摘要:** Current fine-grained classification research primarily focuses on fine-grained feature learning. However, in real-world scenarios, fine-grained data annotation is challenging, and the features and semantics are highly diverse and frequently changing. These issues create inherent barriers between traditional experimental settings and real-world applications, limiting the effectiveness of conventional fine-grained classification methods. Although some recent studies have provided potential solutions to these issues, most of them still rely on limited supervised information and thus fail to offer effective solutions. In this paper, based on theoretical analysis, we propose a novel learning paradigm to break the barriers in fine-grained classification. This paradigm enables the model to progressively learn during inference, thereby leveraging cost-free data to more accurately represent fine-grained categories and adapt to dynamic semantic changes. On this basis, an efficient EXPloring and EXPloiting strategy and method (EXP2) is designed. Thereinto, useful inference data samples are explored according to class representations and exploited to optimize classifiers. Experimental results demonstrate the general effectiveness of our method, providing guidance for future in-depth understanding and exploration of real-world fine-grained classification.
>
---
#### [replaced 006] DVFL-Net: A Lightweight Distilled Video Focal Modulation Network for Spatio-Temporal Action Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.12426v2](http://arxiv.org/pdf/2507.12426v2)**

> **作者:** Hayat Ullah; Muhammad Ali Shafique; Abbas Khan; Arslan Munir
>
> **备注:** 17 pages
>
> **摘要:** The landscape of video recognition has evolved significantly, shifting from traditional Convolutional Neural Networks (CNNs) to Transformer-based architectures for improved accuracy. While 3D CNNs have been effective at capturing spatiotemporal dynamics, recent Transformer models leverage self-attention to model long-range spatial and temporal dependencies. Despite achieving state-of-the-art performance on major benchmarks, Transformers remain computationally expensive, particularly with dense video data. To address this, we propose a lightweight Video Focal Modulation Network, DVFL-Net, which distills spatiotemporal knowledge from a large pre-trained teacher into a compact nano student model, enabling efficient on-device deployment. DVFL-Net utilizes knowledge distillation and spatial-temporal feature modulation to significantly reduce computation while preserving high recognition performance. We employ forward Kullback-Leibler (KL) divergence alongside spatio-temporal focal modulation to effectively transfer both local and global context from the Video-FocalNet Base (teacher) to the proposed VFL-Net (student). We evaluate DVFL-Net on UCF50, UCF101, HMDB51, SSV2, and Kinetics-400, benchmarking it against recent state-of-the-art methods in Human Action Recognition (HAR). Additionally, we conduct a detailed ablation study analyzing the impact of forward KL divergence. The results confirm the superiority of DVFL-Net in achieving an optimal balance between performance and efficiency, demonstrating lower memory usage, reduced GFLOPs, and strong accuracy, making it a practical solution for real-time HAR applications.
>
---
#### [replaced 007] Accelerating Diffusion Transformer via Error-Optimized Cache
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.19243v3](http://arxiv.org/pdf/2501.19243v3)**

> **作者:** Junxiang Qiu; Shuo Wang; Jinda Lu; Lin Liu; Houcheng Jiang; Xingyu Zhu; Yanbin Hao
>
> **摘要:** Diffusion Transformer (DiT) is a crucial method for content generation. However, it needs a lot of time to sample. Many studies have attempted to use caching to reduce the time consumption of sampling. Existing caching methods accelerate generation by reusing DiT features from the previous time step and skipping calculations in the next, but they tend to locate and cache low-error modules without focusing on reducing caching-induced errors, resulting in a sharp decline in generated content quality when increasing caching intensity. To solve this problem, we propose the \textbf{E}rror-\textbf{O}ptimized \textbf{C}ache (\textbf{EOC}). This method introduces three key improvements: \textbf{(1)} Prior knowledge extraction: Extract and process the caching differences; \textbf{(2)} A judgment method for cache optimization: Determine whether certain caching steps need to be optimized; \textbf{(3)} Cache optimization: reduce caching errors. Experiments show that this algorithm significantly reduces the error accumulation caused by caching, especially excessive caching. On the ImageNet dataset, without substantially increasing the computational load, this method improves the FID of the generated images when the rule-based model FORA has a caching level of \textbf{75}\%, \textbf{50}\%, and \textbf{25}\%, and the training-based model Learning-to-cache has a caching level of \textbf{22}\%. Specifically, the FID values change from 30.454 to 21.690 (\textbf{28.8}\%), from 6.857 to 5.821 (\textbf{15.1}\%), from 3.870 to 3.692 (\textbf{4.6}\%), and from 3.539 to 3.451 (\textbf{2.5}\%) respectively. Code is available at https://github.com/qiujx0520/EOC_MM2025.git.
>
---
#### [replaced 008] Multi-Objective Reinforcement Learning for Adaptable Personalized Autonomous Driving
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05223v2](http://arxiv.org/pdf/2505.05223v2)**

> **作者:** Hendrik Surmann; Jorge de Heuvel; Maren Bennewitz
>
> **摘要:** Human drivers exhibit individual preferences regarding driving style. Adapting autonomous vehicles to these preferences is essential for user trust and satisfaction. However, existing end-to-end driving approaches often rely on predefined driving styles or require continuous user feedback for adaptation, limiting their ability to support dynamic, context-dependent preferences. We propose a novel approach using multi-objective reinforcement learning (MORL) with preference-driven optimization for end-to-end autonomous driving that enables runtime adaptation to driving style preferences. Preferences are encoded as continuous weight vectors to modulate behavior along interpretable style objectives$\unicode{x2013}$including efficiency, comfort, speed, and aggressiveness$\unicode{x2013}$without requiring policy retraining. Our single-policy agent integrates vision-based perception in complex mixed-traffic scenarios and is evaluated in diverse urban environments using the CARLA simulator. Experimental results demonstrate that the agent dynamically adapts its driving behavior according to changing preferences while maintaining performance in terms of collision avoidance and route completion.
>
---
#### [replaced 009] How Far Have Medical Vision-Language Models Come? A Comprehensive Benchmarking Study
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.11200v2](http://arxiv.org/pdf/2507.11200v2)**

> **作者:** Che Liu; Jiazhen Pan; Weixiang Shen; Wenjia Bai; Daniel Rueckert; Rossella Arcucci
>
> **备注:** Technical report
>
> **摘要:** Vision-Language Models (VLMs) trained on web-scale corpora excel at natural image tasks and are increasingly repurposed for healthcare; however, their competence in medical tasks remains underexplored. We present a comprehensive evaluation of open-source general-purpose and medically specialised VLMs, ranging from 3B to 72B parameters, across eight benchmarks: MedXpert, OmniMedVQA, PMC-VQA, PathVQA, MMMU, SLAKE, and VQA-RAD. To observe model performance across different aspects, we first separate it into understanding and reasoning components. Three salient findings emerge. First, large general-purpose models already match or surpass medical-specific counterparts on several benchmarks, demonstrating strong zero-shot transfer from natural to medical images. Second, reasoning performance is consistently lower than understanding, highlighting a critical barrier to safe decision support. Third, performance varies widely across benchmarks, reflecting differences in task design, annotation quality, and knowledge demands. No model yet reaches the reliability threshold for clinical deployment, underscoring the need for stronger multimodal alignment and more rigorous, fine-grained evaluation protocols.
>
---
#### [replaced 010] Critiques of World Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.05169v2](http://arxiv.org/pdf/2507.05169v2)**

> **作者:** Eric Xing; Mingkai Deng; Jinyu Hou; Zhiting Hu
>
> **摘要:** World Model, the supposed algorithmic surrogate of the real-world environment which biological agents experience with and act upon, has been an emerging topic in recent years because of the rising needs to develop virtual agents with artificial (general) intelligence. There has been much debate on what a world model really is, how to build it, how to use it, and how to evaluate it. In this essay, starting from the imagination in the famed Sci-Fi classic Dune, and drawing inspiration from the concept of "hypothetical thinking" in psychology literature, we offer critiques of several schools of thoughts on world modeling, and argue the primary goal of a world model to be simulating all actionable possibilities of the real world for purposeful reasoning and acting. Building on the critiques, we propose a new architecture for a general-purpose world model, based on hierarchical, multi-level, and mixed continuous/discrete representations, and a generative and self-supervision learning framework, with an outlook of a Physical, Agentic, and Nested (PAN) AGI system enabled by such a model.
>
---
#### [replaced 011] Hierarchical Multi-Stage Transformer Architecture for Context-Aware Temporal Action Localization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06411v2](http://arxiv.org/pdf/2507.06411v2)**

> **作者:** Hayat Ullah; Arslan Munir; Oliver Nina
>
> **备注:** 17 pages, 6 figures,
>
> **摘要:** Inspired by the recent success of transformers and multi-stage architectures in video recognition and object detection domains. We thoroughly explore the rich spatio-temporal properties of transformers within a multi-stage architecture paradigm for the temporal action localization (TAL) task. This exploration led to the development of a hierarchical multi-stage transformer architecture called PCL-Former, where each subtask is handled by a dedicated transformer module with a specialized loss function. Specifically, the Proposal-Former identifies candidate segments in an untrimmed video that may contain actions, the Classification-Former classifies the action categories within those segments, and the Localization-Former precisely predicts the temporal boundaries (i.e., start and end) of the action instances. To evaluate the performance of our method, we have conducted extensive experiments on three challenging benchmark datasets: THUMOS-14, ActivityNet-1.3, and HACS Segments. We also conducted detailed ablation experiments to assess the impact of each individual module of our PCL-Former. The obtained quantitative results validate the effectiveness of the proposed PCL-Former, outperforming state-of-the-art TAL approaches by 2.8%, 1.2%, and 4.8% on THUMOS14, ActivityNet-1.3, and HACS datasets, respectively.
>
---
#### [replaced 012] Lost in Tracking Translation: A Comprehensive Analysis of Visual SLAM in Human-Centered XR and IoT Ecosystems
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.07146v2](http://arxiv.org/pdf/2411.07146v2)**

> **作者:** Yasra Chandio; Khotso Selialia; Joseph DeGol; Luis Garcia; Fatima M. Anwar
>
> **摘要:** Advancements in tracking algorithms have empowered nascent applications across various domains, from steering autonomous vehicles to guiding robots to enhancing augmented reality experiences for users. However, these algorithms are application-specific and do not work across applications with different types of motion; even a tracking algorithm designed for a given application does not work in scenarios deviating from highly standard conditions. For example, a tracking algorithm designed for robot navigation inside a building will not work for tracking the same robot in an outdoor environment. To demonstrate this problem, we evaluate the performance of the state-of-the-art tracking methods across various applications and scenarios. To inform our analysis, we first categorize algorithmic, environmental, and locomotion-related challenges faced by tracking algorithms. We quantitatively evaluate the performance using multiple tracking algorithms and representative datasets for a wide range of Internet of Things (IoT) and Extended Reality (XR) applications, including autonomous vehicles, drones, and humans. Our analysis shows that no tracking algorithm works across different applications and scenarios within applications. Ultimately, using the insights generated from our analysis, we discuss multiple approaches to improving the tracking performance using input data characterization, leveraging intermediate information, and output evaluation.
>
---
#### [replaced 013] A Mixture of Experts (MoE) model to improve AI-based computational pathology prediction performance under variable levels of histopathology image blur
- **分类: eess.IV; cs.CV; I.4; J.3**

- **链接: [http://arxiv.org/pdf/2405.09298v5](http://arxiv.org/pdf/2405.09298v5)**

> **作者:** Yujie Xiang; Bojing Liu; Mattias Rantalainen
>
> **摘要:** AI-based models for histopathology whole slide image (WSI) analysis are increasingly common, but unsharp or blurred areas within WSI can significantly reduce prediction performance. In this study, we investigated the effect of image blur on deep learning models and introduced a mixture of experts (MoE) strategy that combines predictions from multiple expert models trained on data with varying blur levels. Using H&E-stained WSIs from 2,093 breast cancer patients, we benchmarked performance on grade classification and IHC biomarker prediction with both CNN- (CNN_CLAM and MoE-CNN_CLAM) and Vision Transformer-based (UNI_CLAM and MoE-UNI_CLAM) models. Our results show that baseline models' performance consistently decreased with increasing blur, but expert models trained on blurred tiles and especially our proposed MoE approach substantially improved performance, and outperformed baseline models in a range of simulated scenarios. MoE-CNN_CLAM outperformed the baseline CNN_CLAM under moderate (AUC: 0.868 vs. 0.702) and mixed blur conditions (AUC: 0.890 vs. 0.875). MoE-UNI_CLAM outperformed the baseline UNI_CLAM model in both moderate (AUC: 0.950 vs. 0.928) and mixed blur conditions (AUC: 0.944 vs. 0.931). This MoE method has the potential to enhance the reliability of AI-based pathology models under variable image quality, supporting broader application in both research and clinical settings.
>
---
#### [replaced 014] CDUPatch: Color-Driven Universal Adversarial Patch Attack for Dual-Modal Visible-Infrared Detectors
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.10888v2](http://arxiv.org/pdf/2504.10888v2)**

> **作者:** Jiahuan Long; Wen Yao; Tingsong Jiang; Chao Ma
>
> **备注:** Accepted by ACMMM 2025
>
> **摘要:** Adversarial patches are widely used to evaluate the robustness of object detection systems in real-world scenarios. These patches were initially designed to deceive single-modal detectors (e.g., visible or infrared) and have recently been extended to target visible-infrared dual-modal detectors. However, existing dual-modal adversarial patch attacks have limited attack effectiveness across diverse physical scenarios. To address this, we propose CDUPatch, a universal cross-modal patch attack against visible-infrared object detectors across scales, views, and scenarios. Specifically, we observe that color variations lead to different levels of thermal absorption, resulting in temperature differences in infrared imaging. Leveraging this property, we propose an RGB-to-infrared adapter that maps RGB patches to infrared patches, enabling unified optimization of cross-modal patches. By learning an optimal color distribution on the adversarial patch, we can manipulate its thermal response and generate an adversarial infrared texture. Additionally, we introduce a multi-scale clipping strategy and construct a new visible-infrared dataset, MSDrone, which contains aerial vehicle images in varying scales and perspectives. These data augmentation strategies enhance the robustness of our patch in real-world conditions. Experiments on four benchmark datasets (e.g., DroneVehicle, LLVIP, VisDrone, MSDrone) show that our method outperforms existing patch attacks in the digital domain. Extensive physical tests further confirm strong transferability across scales, views, and scenarios.
>
---
#### [replaced 015] SecurePose: Automated Face Blurring and Human Movement Kinematics Extraction from Videos Recorded in Clinical Settings
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2402.14143v2](http://arxiv.org/pdf/2402.14143v2)**

> **作者:** Rishabh Bajpai; Bhooma Aravamuthan
>
> **摘要:** Movement disorder diagnosis often relies on expert evaluation of patient videos, but sharing these videos poses privacy risks. Current methods for de-identifying videos, such as blurring faces, are often manual, inconsistent, or inaccurate. Furthermore, these methods can compromise objective kinematic analysis - a crucial component of diagnosis. To address these challenges, we developed SecurePose, an open-source software that simultaneously provides reliable de-identification and automated kinematic extraction from videos recorded in clinic settings using smartphones/tablets. SecurePose utilizes pose estimation (using OpenPose) to extract full body kinematics, track individuals, identify the patient, and then accurately blur faces in the videos. We validated SecurePose on gait videos recorded in outpatient clinic visits of 116 children with cerebral palsy, assessing both the accuracy of its de-identification compared to the ground truth (manual blurring) and the reliability of the intermediate steps of kinematics extraction. Results demonstrate that SecurePose outperformed six existing methods in automated face detection and achieved comparable accuracy to robust manual blurring, but in significantly less time (91.08% faster). Ten experienced researchers also confirmed SecurePose's usability via System Usability Scale scores. These findings validate SecurePose as a practical and effective tool for protecting patient privacy while enabling accurate kinematics extraction in clinical settings.
>
---
#### [replaced 016] Revisiting Data Augmentation for Ultrasound Images
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.13193v2](http://arxiv.org/pdf/2501.13193v2)**

> **作者:** Adam Tupper; Christian Gagné
>
> **备注:** Published in the Transacations of Machine Learning Research (TMLR, 2025), see https://openreview.net/forum?id=iGcxlTLIL5 . For the associated source code see https://github.com/adamtupper/ultrasound-augmentation
>
> **摘要:** Data augmentation is a widely used and effective technique to improve the generalization performance of deep neural networks. Yet, despite often facing limited data availability when working with medical images, it is frequently underutilized. This appears to come from a gap in our collective understanding of the efficacy of different augmentation techniques across different tasks and modalities. One modality where this is especially true is ultrasound imaging. This work addresses this gap by analyzing the effectiveness of different augmentation techniques at improving model performance across a wide range of ultrasound image analysis tasks. To achieve this, we introduce a new standardized benchmark of 14 ultrasound image classification and semantic segmentation tasks from 10 different sources and covering 11 body regions. Our results demonstrate that many of the augmentations commonly used for tasks on natural images are also effective on ultrasound images, even more so than augmentations developed specifically for ultrasound images in some cases. We also show that diverse augmentation using TrivialAugment, which is widely used for natural images, is also effective for ultrasound images. Moreover, our proposed methodology represents a structured approach for assessing various data augmentations that can be applied to other contexts and modalities.
>
---
#### [replaced 017] Large-Vocabulary Segmentation for Medical Images with Text Prompts
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2312.17183v5](http://arxiv.org/pdf/2312.17183v5)**

> **作者:** Ziheng Zhao; Yao Zhang; Chaoyi Wu; Xiaoman Zhang; Xiao Zhou; Ya Zhang; Yanfeng Wang; Weidi Xie
>
> **备注:** 74 pages
>
> **摘要:** This paper aims to build a model that can Segment Anything in 3D medical images, driven by medical terminologies as Text prompts, termed as SAT. Our main contributions are three-fold: (i) We construct the first multimodal knowledge tree on human anatomy, including 6502 anatomical terminologies; Then, we build the largest and most comprehensive segmentation dataset for training, collecting over 22K 3D scans from 72 datasets, across 497 classes, with careful standardization on both image and label space; (ii) We propose to inject medical knowledge into a text encoder via contrastive learning and formulate a large-vocabulary segmentation model that can be prompted by medical terminologies in text form; (iii) We train SAT-Nano (110M parameters) and SAT-Pro (447M parameters). SAT-Pro achieves comparable performance to 72 nnU-Nets -- the strongest specialist models trained on each dataset (over 2.2B parameters combined) -- over 497 categories. Compared with the interactive approach MedSAM, SAT-Pro consistently outperforms across all 7 human body regions with +7.1% average Dice Similarity Coefficient (DSC) improvement, while showing enhanced scalability and robustness. On 2 external (cross-center) datasets, SAT-Pro achieves higher performance than all baselines (+3.7% average DSC), demonstrating superior generalization ability.
>
---
#### [replaced 018] MuteSwap: Visual-informed Silent Video Identity Conversion
- **分类: cs.SD; cs.CV; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.00498v2](http://arxiv.org/pdf/2507.00498v2)**

> **作者:** Yifan Liu; Yu Fang; Zhouhan Lin
>
> **摘要:** Conventional voice conversion modifies voice characteristics from a source speaker to a target speaker, relying on audio input from both sides. However, this process becomes infeasible when clean audio is unavailable, such as in silent videos or noisy environments. In this work, we focus on the task of Silent Face-based Voice Conversion (SFVC), which does voice conversion entirely from visual inputs. i.e., given images of a target speaker and a silent video of a source speaker containing lip motion, SFVC generates speech aligning the identity of the target speaker while preserving the speech content in the source silent video. As this task requires generating intelligible speech and converting identity using only visual cues, it is particularly challenging. To address this, we introduce MuteSwap, a novel framework that employs contrastive learning to align cross-modality identities and minimize mutual information to separate shared visual features. Experimental results show that MuteSwap achieves impressive performance in both speech synthesis and identity conversion, especially under noisy conditions where methods dependent on audio input fail to produce intelligible results, demonstrating both the effectiveness of our training approach and the feasibility of SFVC.
>
---
#### [replaced 019] VFaith: Do Large Multimodal Models Really Reason on Seen Images Rather than Previous Memories?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.11571v2](http://arxiv.org/pdf/2506.11571v2)**

> **作者:** Jiachen Yu; Yufei Zhan; Ziheng Wu; Yousong Zhu; Jinqiao Wang; Minghui Qiu
>
> **摘要:** Recent extensive works have demonstrated that by introducing long CoT, the capabilities of MLLMs to solve complex problems can be effectively enhanced. However, the reasons for the effectiveness of such paradigms remain unclear. It is challenging to analysis with quantitative results how much the model's specific extraction of visual cues and its subsequent so-called reasoning during inference process contribute to the performance improvements. Therefore, evaluating the faithfulness of MLLMs' reasoning to visual information is crucial. To address this issue, we first present a cue-driven automatic and controllable editing pipeline with the help of GPT-Image-1. It enables the automatic and precise editing of specific visual cues based on the instruction. Furthermore, we introduce VFaith-Bench, the first benchmark to evaluate MLLMs' visual reasoning capabilities and analyze the source of such capabilities with an emphasis on the visual faithfulness. Using the designed pipeline, we constructed comparative question-answer pairs by altering the visual cues in images that are crucial for solving the original reasoning problem, thereby changing the question's answer. By testing similar questions with images that have different details, the average accuracy reflects the model's visual reasoning ability, while the difference in accuracy before and after editing the test set images effectively reveals the relationship between the model's reasoning ability and visual perception. We further designed specific metrics to expose this relationship. VFaith-Bench includes 755 entries divided into five distinct subsets, along with an additional human-labeled perception task. We conducted in-depth testing and analysis of existing mainstream flagship models and prominent open-source model series/reasoning models on VFaith-Bench, further investigating the underlying factors of their reasoning capabilities.
>
---
#### [replaced 020] Cycle-Consistent Multi-Graph Matching for Self-Supervised Annotation of C.Elegans
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07348v2](http://arxiv.org/pdf/2503.07348v2)**

> **作者:** Christoph Karg; Sebastian Stricker; Lisa Hutschenreiter; Bogdan Savchynskyy; Dagmar Kainmueller
>
> **摘要:** In this work we present a novel approach for unsupervised multi-graph matching, which applies to problems for which a Gaussian distribution of keypoint features can be assumed. We leverage cycle consistency as loss for self-supervised learning, and determine Gaussian parameters through Bayesian Optimization, yielding a highly efficient approach that scales to large datasets. Our fully unsupervised approach enables us to reach the accuracy of state-of-the-art supervised methodology for the biomedical use case of semantic cell annotation in 3D microscopy images of the worm C. elegans. To this end, our approach yields the first unsupervised atlas of C. elegans, i.e. a model of the joint distribution of all of its cell nuclei, without the need for any ground truth cell annotation. This advancement enables highly efficient semantic annotation of cells in large microscopy datasets, overcoming a current key bottleneck. Beyond C. elegans, our approach offers fully unsupervised construction of cell-level atlases for any model organism with a stereotyped body plan down to the level of unique semantic cell labels, and thus bears the potential to catalyze respective biomedical studies in a range of further species.
>
---
#### [replaced 021] Horticultural Temporal Fruit Monitoring via 3D Instance Segmentation and Re-Identification using Colored Point Clouds
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.07799v2](http://arxiv.org/pdf/2411.07799v2)**

> **作者:** Daniel Fusaro; Federico Magistri; Jens Behley; Alberto Pretto; Cyrill Stachniss
>
> **备注:** Submitted to Computers and Electronics in Agriculture
>
> **摘要:** Accurate and consistent fruit monitoring over time is a key step toward automated agricultural production systems. However, this task is inherently difficult due to variations in fruit size, shape, occlusion, orientation, and the dynamic nature of orchards where fruits may appear or disappear between observations. In this article, we propose a novel method for fruit instance segmentation and re-identification on 3D terrestrial point clouds collected over time. Our approach directly operates on dense colored point clouds, capturing fine-grained 3D spatial detail. We segment individual fruits using a learning-based instance segmentation method applied directly to the point cloud. For each segmented fruit, we extract a compact and discriminative descriptor using a 3D sparse convolutional neural network. To track fruits across different times, we introduce an attention-based matching network that associates fruits with their counterparts from previous sessions. Matching is performed using a probabilistic assignment scheme, selecting the most likely associations across time. We evaluate our approach on real-world datasets of strawberries and apples, demonstrating that it outperforms existing methods in both instance segmentation and temporal re-identification, enabling robust and precise fruit monitoring across complex and dynamic orchard environments.
>
---
#### [replaced 022] UniEmoX: Cross-modal Semantic-Guided Large-Scale Pretraining for Universal Scene Emotion Perception
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.18877v3](http://arxiv.org/pdf/2409.18877v3)**

> **作者:** Chuang Chen; Xiao Sun; Zhi Liu
>
> **备注:** Accepted by IEEE TIP
>
> **摘要:** Visual emotion analysis holds significant research value in both computer vision and psychology. However, existing methods for visual emotion analysis suffer from limited generalizability due to the ambiguity of emotion perception and the diversity of data scenarios. To tackle this issue, we introduce UniEmoX, a cross-modal semantic-guided large-scale pretraining framework. Inspired by psychological research emphasizing the inseparability of the emotional exploration process from the interaction between individuals and their environment, UniEmoX integrates scene-centric and person-centric low-level image spatial structural information, aiming to derive more nuanced and discriminative emotional representations. By exploiting the similarity between paired and unpaired image-text samples, UniEmoX distills rich semantic knowledge from the CLIP model to enhance emotional embedding representations more effectively. To the best of our knowledge, this is the first large-scale pretraining framework that integrates psychological theories with contemporary contrastive learning and masked image modeling techniques for emotion analysis across diverse scenarios. Additionally, we develop a visual emotional dataset titled Emo8. Emo8 samples cover a range of domains, including cartoon, natural, realistic, science fiction and advertising cover styles, covering nearly all common emotional scenes. Comprehensive experiments conducted on six benchmark datasets across two downstream tasks validate the effectiveness of UniEmoX. The source code is available at https://github.com/chincharles/u-emo.
>
---
#### [replaced 023] VAPO: Visibility-Aware Keypoint Localization for Efficient 6DoF Object Pose Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.14559v5](http://arxiv.org/pdf/2403.14559v5)**

> **作者:** Ruyi Lian; Yuewei Lin; Longin Jan Latecki; Haibin Ling
>
> **备注:** accepted for publication in the Proceedings of the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025) as oral presentation
>
> **摘要:** Localizing predefined 3D keypoints in a 2D image is an effective way to establish 3D-2D correspondences for instance-level 6DoF object pose estimation. However, unreliable localization results of invisible keypoints degrade the quality of correspondences. In this paper, we address this issue by localizing the important keypoints in terms of visibility. Since keypoint visibility information is currently missing in the dataset collection process, we propose an efficient way to generate binary visibility labels from available object-level annotations, for keypoints of both asymmetric objects and symmetric objects. We further derive real-valued visibility-aware importance from binary labels based on the PageRank algorithm. Taking advantage of the flexibility of our visibility-aware importance, we construct VAPO (Visibility-Aware POse estimator) by integrating the visibility-aware importance with a state-of-the-art pose estimation algorithm, along with additional positional encoding. VAPO can work in both CAD-based and CAD-free settings. Extensive experiments are conducted on popular pose estimation benchmarks including Linemod, Linemod-Occlusion, and YCB-V, demonstrating that VAPO clearly achieves state-of-the-art performances. Project page: https://github.com/RuyiLian/VAPO.
>
---
#### [replaced 024] PhenoBench: A Comprehensive Benchmark for Cell Phenotyping
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.03532v4](http://arxiv.org/pdf/2507.03532v4)**

> **作者:** Jannik Franzen; Fabian H. Reith; Claudia Winklmayr; Jerome Luescher; Nora Koreuber; Elias Baumann; Christian M. Schuerch; Dagmar Kainmueller; Josef Lorenz Rumberger
>
> **备注:** accepted for presentation at MICCAI 2025
>
> **摘要:** Digital pathology has seen the advent of a wealth of foundational models (FM), yet to date their performance on cell phenotyping has not been benchmarked in a unified manner. We therefore propose PhenoBench: A comprehensive benchmark for cell phenotyping on Hematoxylin and Eosin (H&E) stained histopathology images. We provide both PhenoCell, a new H&E dataset featuring 14 granular cell types identified by using multiplexed imaging, and ready-to-use fine-tuning and benchmarking code that allows the systematic evaluation of multiple prominent pathology FMs in terms of dense cell phenotype predictions in different generalization scenarios. We perform extensive benchmarking of existing FMs, providing insights into their generalization behavior under technical vs. medical domain shifts. Furthermore, while FMs achieve macro F1 scores > 0.70 on previously established benchmarks such as Lizard and PanNuke, on PhenoCell, we observe scores as low as 0.20. This indicates a much more challenging task not captured by previous benchmarks, establishing PhenoCell as a prime asset for future benchmarking of FMs and supervised models alike. Code and data are available on GitHub.
>
---
#### [replaced 025] Consistency Trajectory Matching for One-Step Generative Super-Resolution
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.20349v4](http://arxiv.org/pdf/2503.20349v4)**

> **作者:** Weiyi You; Mingyang Zhang; Leheng Zhang; Xingyu Zhou; Kexuan Shi; Shuhang Gu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Current diffusion-based super-resolution (SR) approaches achieve commendable performance at the cost of high inference overhead. Therefore, distillation techniques are utilized to accelerate the multi-step teacher model into one-step student model. Nevertheless, these methods significantly raise training costs and constrain the performance of the student model by the teacher model. To overcome these tough challenges, we propose Consistency Trajectory Matching for Super-Resolution (CTMSR), a distillation-free strategy that is able to generate photo-realistic SR results in one step. Concretely, we first formulate a Probability Flow Ordinary Differential Equation (PF-ODE) trajectory to establish a deterministic mapping from low-resolution (LR) images with noise to high-resolution (HR) images. Then we apply the Consistency Training (CT) strategy to directly learn the mapping in one step, eliminating the necessity of pre-trained diffusion model. To further enhance the performance and better leverage the ground-truth during the training process, we aim to align the distribution of SR results more closely with that of the natural images. To this end, we propose to minimize the discrepancy between their respective PF-ODE trajectories from the LR image distribution by our meticulously designed Distribution Trajectory Matching (DTM) loss, resulting in improved realism of our recovered HR images. Comprehensive experimental results demonstrate that the proposed methods can attain comparable or even superior capabilities on both synthetic and real datasets while maintaining minimal inference latency.
>
---
#### [replaced 026] TextDiffuser-RL: Efficient and Robust Text Layout Optimization for High-Fidelity Text-to-Image Synthesis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.19291v2](http://arxiv.org/pdf/2505.19291v2)**

> **作者:** Kazi Mahathir Rahman; Showrin Rahman; Sharmin Sultana Srishty
>
> **备注:** 14 pages, 26 figures. Submitted to arXiv for dissemination. Intended for future submission to a Generative AI conference
>
> **摘要:** Text-embedded image generation plays a critical role in industries such as graphic design, advertising, and digital content creation. Text-to-Image generation methods leveraging diffusion models, such as TextDiffuser-2, have demonstrated promising results in producing images with embedded text. TextDiffuser-2 effectively generates bounding box layouts that guide the rendering of visual text, achieving high fidelity and coherence. However, existing approaches often rely on resource-intensive processes and are limited in their ability to run efficiently on both CPU and GPU platforms. To address these challenges, we propose a novel two-stage pipeline that integrates reinforcement learning (RL) for rapid and optimized text layout generation with a diffusion-based image synthesis model. Our RL-based approach significantly accelerates the bounding box prediction step while reducing overlaps, allowing the system to run efficiently on both CPUs and GPUs. Extensive evaluations demonstrate that our framework maintains or surpasses TextDiffuser-2's quality in text placement and image synthesis, with markedly faster runtime and increased flexibility. Extensive evaluations demonstrate that our framework maintains or surpasses TextDiffuser-2's quality in text placement and image synthesis, with markedly faster runtime and increased flexibility. Our approach has been evaluated on the MARIOEval benchmark, achieving OCR and CLIPScore metrics close to state-of-the-art models, while being 97.64% more faster and requiring only 2MB of memory to run.
>
---
#### [replaced 027] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.01312v2](http://arxiv.org/pdf/2502.01312v2)**

> **作者:** Xiao Lin; Yun Peng; Liuyi Wang; Xianyou Zhong; Minghao Zhu; Jingwei Yang; Yi Feng; Chengju Liu; Qijun Chen
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Category-level object pose estimation aims to recover the rotation, translation and size of unseen instances within predefined categories. In this task, deep neural network-based methods have demonstrated remarkable performance. However, previous studies show they suffer from spurious correlations raised by "unclean" confounders in models, hindering their performance on novel instances with significant variations. To address this issue, we propose CleanPose, a novel approach integrating causal learning and knowledge distillation to enhance category-level pose estimation. To mitigate the negative effect of unobserved confounders, we develop a causal inference module based on front-door adjustment, which promotes unbiased estimation by reducing potential spurious correlations. Additionally, to further improve generalization ability, we devise a residual-based knowledge distillation method that has proven effective in providing comprehensive category information guidance. Extensive experiments across multiple benchmarks (REAL275, CAMERA25 and HouseCat6D) hightlight the superiority of proposed CleanPose over state-of-the-art methods. Code will be available at https://github.com/chrislin0621/CleanPose.
>
---
#### [replaced 028] Improved DDIM Sampling with Moment Matching Gaussian Mixtures
- **分类: cs.CV; cs.AI; cs.LG; I.2, I.4**

- **链接: [http://arxiv.org/pdf/2311.04938v3](http://arxiv.org/pdf/2311.04938v3)**

> **作者:** Prasad Gabbur
>
> **备注:** 29 pages, 14 figures; Analysis of DDIM-GMM as a multimodal denoiser; Additional experiments on LSUN datasets and text-to-image generation with Stable Diffusion; Comparison with DPM-Solver; Ablations on GMM parameters; Updated equations with bold font for vectors and matrices
>
> **摘要:** We propose using a Gaussian Mixture Model (GMM) as reverse transition operator (kernel) within the Denoising Diffusion Implicit Models (DDIM) framework, which is one of the most widely used approaches for accelerated sampling from pre-trained Denoising Diffusion Probabilistic Models (DDPM). Specifically we match the first and second order central moments of the DDPM forward marginals by constraining the parameters of the GMM. We see that moment matching is sufficient to obtain samples with equal or better quality than the original DDIM with Gaussian kernels. We provide experimental results with unconditional models trained on CelebAHQ and FFHQ and class-conditional models trained on ImageNet datasets respectively. Our results suggest that using the GMM kernel leads to significant improvements in the quality of the generated samples when the number of sampling steps is small, as measured by FID and IS metrics. For example on ImageNet 256x256, using 10 sampling steps, we achieve a FID of 6.94 and IS of 207.85 with a GMM kernel compared to 10.15 and 196.73 respectively with a Gaussian kernel.
>
---
#### [replaced 029] Inverse Synthetic Aperture Fourier Ptychography
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.03733v2](http://arxiv.org/pdf/2507.03733v2)**

> **作者:** Matthew A. Chan; Casey J. Pellizzari; Christopher A. Metzler
>
> **摘要:** Fourier ptychography (FP) is a powerful light-based synthetic aperture imaging technique that allows one to reconstruct a high-resolution, wide field-of-view image by computationally integrating a diverse collection of low-resolution, far-field measurements. Typically, FP measurement diversity is introduced by changing the angle of the illumination or the position of the camera; either approach results in sampling different portions of the target's spatial frequency content, but both approaches introduce substantial costs and complexity to the acquisition process. In this work, we introduce Inverse Synthetic Aperture Fourier Ptychography, a novel approach to FP that foregoes changing the illumination angle or camera position and instead generates measurement diversity through target motion. Critically, we also introduce a novel learning-based method for estimating k-space coordinates from dual plane intensity measurements, thereby enabling synthetic aperture imaging without knowing the rotation of the target. We experimentally validate our method in simulation and on a tabletop optical system.
>
---
#### [replaced 030] FBSDiff: Plug-and-Play Frequency Band Substitution of Diffusion Features for Highly Controllable Text-Driven Image Translation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2408.00998v3](http://arxiv.org/pdf/2408.00998v3)**

> **作者:** Xiang Gao; Jiaying Liu
>
> **备注:** Accepted conference paper of ACM MM 2024
>
> **摘要:** Large-scale text-to-image diffusion models have been a revolutionary milestone in the evolution of generative AI and multimodal technology, allowing wonderful image generation with natural-language text prompt. However, the issue of lacking controllability of such models restricts their practical applicability for real-life content creation. Thus, attention has been focused on leveraging a reference image to control text-to-image synthesis, which is also regarded as manipulating (or editing) a reference image as per a text prompt, namely, text-driven image-to-image translation. This paper contributes a novel, concise, and efficient approach that adapts pre-trained large-scale text-to-image (T2I) diffusion model to the image-to-image (I2I) paradigm in a plug-and-play manner, realizing high-quality and versatile text-driven I2I translation without any model training, model fine-tuning, or online optimization process. To guide T2I generation with a reference image, we propose to decompose diverse guiding factors with different frequency bands of diffusion features in the DCT spectral space, and accordingly devise a novel frequency band substitution layer which realizes dynamic control of the reference image to the T2I generation result in a plug-and-play manner. We demonstrate that our method allows flexible control over both guiding factor and guiding intensity of the reference image simply by tuning the type and bandwidth of the substituted frequency band, respectively. Extensive qualitative and quantitative experiments verify superiority of our approach over related methods in I2I translation visual quality, versatility, and controllability. The code is publicly available at: https://github.com/XiangGao1102/FBSDiff.
>
---
#### [replaced 031] EvolveNav: Self-Improving Embodied Reasoning for LLM-Based Vision-Language Navigation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01551v2](http://arxiv.org/pdf/2506.01551v2)**

> **作者:** Bingqian Lin; Yunshuang Nie; Khun Loun Zai; Ziming Wei; Mingfei Han; Rongtao Xu; Minzhe Niu; Jianhua Han; Liang Lin; Cewu Lu; Xiaodan Liang
>
> **摘要:** Building Vision-Language Navigation (VLN) agents which can navigate following natural language instructions is a long-standing goal in human-robot interaction applications. Recent studies have revealed the potential of training open-source Large Language Models (LLMs) to unleash LLMs' reasoning ability for improving navigation, and simultaneously mitigate the domain gap between LLMs' training corpus and the VLN task. However, these approaches primarily adopt direct input-output mapping paradigms, causing the mapping learning difficult and the navigational decisions unexplainable. Chain-of-Thought (CoT) training is a promising way to improve both navigational decision accuracy and interpretability, while the complexity of the navigation task makes the perfect CoT labels unavailable and may lead to overfitting through pure CoT supervised fine-tuning. In this paper, we propose a novel sElf-improving embodied reasoning framework for boosting LLM-based vision-language Navigation, dubbed EvolveNav. Our EvolveNav consists of two stages: (1) Formalized CoT Supervised Fine-Tuning, where we train the model with formalized CoT labels to both activate the model's navigational reasoning capabilities and increase the reasoning speed; (2) Self-Reflective Post-Training, where the model is iteratively trained with its own reasoning outputs as self-enriched CoT labels to enhance the supervision diversity. A self-reflective auxiliary task is also introduced to encourage learning correct reasoning patterns by contrasting with wrong ones. Experimental results on the popular VLN benchmarks demonstrate the superiority of EvolveNav over previous LLM-based VLN approaches. Code is available at https://github.com/expectorlin/EvolveNav.
>
---
#### [replaced 032] BeetleVerse: A Study on Taxonomic Classification of Ground Beetles
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13393v2](http://arxiv.org/pdf/2504.13393v2)**

> **作者:** S M Rayeed; Alyson East; Samuel Stevens; Sydne Record; Charles V Stewart
>
> **备注:** Paper Accepted at Computer Vision and Pattern Recognition 2025 (Workshop CV4Animals: Computer Vision for Animal Behavior Tracking and Modeling)
>
> **摘要:** Ground beetles are a highly sensitive and speciose biological indicator, making them vital for monitoring biodiversity. However, they are currently an underutilized resource due to the manual effort required by taxonomic experts to perform challenging species differentiations based on subtle morphological differences, precluding widespread applications. In this paper, we evaluate 12 vision models on taxonomic classification across four diverse, long-tailed datasets spanning over 230 genera and 1769 species, with images ranging from controlled laboratory settings to challenging field-collected (in-situ) photographs. We further explore taxonomic classification in two important real-world contexts: sample efficiency and domain adaptation. Our results show that the Vision and Language Transformer combined with an MLP head is the best performing model, with 97% accuracy at genus and 94% at species level. Sample efficiency analysis shows that we can reduce train data requirements by up to 50% with minimal compromise in performance. The domain adaptation experiments reveal significant challenges when transferring models from lab to in-situ images, highlighting a critical domain gap. Overall, our study lays a foundation for large-scale automated taxonomic classification of beetles, and beyond that, advances sample-efficient learning and cross-domain adaptation for diverse long-tailed ecological datasets.
>
---
#### [replaced 033] Understanding Dataset Bias in Medical Imaging: A Case Study on Chest X-rays
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07722v4](http://arxiv.org/pdf/2507.07722v4)**

> **作者:** Ethan Dack; Chengliang Dai
>
> **摘要:** Recent works have revisited the infamous task ``Name That Dataset'', demonstrating that non-medical datasets contain underlying biases and that the dataset origin task can be solved with high accuracy. In this work, we revisit the same task applied to popular open-source chest X-ray datasets. Medical images are naturally more difficult to release for open-source due to their sensitive nature, which has led to certain open-source datasets being extremely popular for research purposes. By performing the same task, we wish to explore whether dataset bias also exists in these datasets. To extend our work, we apply simple transformations to the datasets, repeat the same task, and perform an analysis to identify and explain any detected biases. Given the importance of AI applications in medical imaging, it's vital to establish whether modern methods are taking shortcuts or are focused on the relevant pathology. We implement a range of different network architectures on the datasets: NIH, CheXpert, MIMIC-CXR and PadChest. We hope this work will encourage more explainable research being performed in medical imaging and the creation of more open-source datasets in the medical domain. Our code can be found here: https://github.com/eedack01/x_ray_ds_bias.
>
---
#### [replaced 034] Exploiting Label Skewness for Spiking Neural Networks in Federated Learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.17305v3](http://arxiv.org/pdf/2412.17305v3)**

> **作者:** Di Yu; Xin Du; Linshan Jiang; Huijing Zhang; Shuiguang Deng
>
> **备注:** This work has been accepted on the International Joint Conference on Artificial Intelligence 2025
>
> **摘要:** The energy efficiency of deep spiking neural networks (SNNs) aligns with the constraints of resource-limited edge devices, positioning SNNs as a promising foundation for intelligent applications leveraging the extensive data collected by these devices. To address data privacy concerns when deploying SNNs on edge devices, federated learning (FL) facilitates collaborative model training by leveraging data distributed across edge devices without transmitting local data to a central server. However, existing FL approaches struggle with label-skewed data across devices, which leads to drift in local SNN models and degrades the performance of the global SNN model. In this paper, we propose a novel framework called FedLEC, which incorporates intra-client label weight calibration to balance the learning intensity across local labels and inter-client knowledge distillation to mitigate local SNN model bias caused by label absence. Extensive experiments with three different structured SNNs across five datasets (i.e., three non-neuromorphic and two neuromorphic datasets) demonstrate the efficiency of FedLEC. Compared to eight state-of-the-art FL algorithms, FedLEC achieves an average accuracy improvement of approximately 11.59% for the global SNN model under various label skew distribution settings.
>
---
#### [replaced 035] EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.12440v3](http://arxiv.org/pdf/2507.12440v3)**

> **作者:** Ruihan Yang; Qinxi Yu; Yecheng Wu; Rui Yan; Borui Li; An-Chieh Cheng; Xueyan Zou; Yunhao Fang; Xuxin Cheng; Ri-Zhao Qiu; Hongxu Yin; Sifei Liu; Song Han; Yao Lu; Xiaolong Wang
>
> **备注:** More videos can be found on our website: https://rchalyang.github.io/EgoVLA
>
> **摘要:** Real robot data collection for imitation learning has led to significant advancements in robotic manipulation. However, the requirement for robot hardware in the process fundamentally constrains the scale of the data. In this paper, we explore training Vision-Language-Action (VLA) models using egocentric human videos. The benefit of using human videos is not only for their scale but more importantly for the richness of scenes and tasks. With a VLA trained on human video that predicts human wrist and hand actions, we can perform Inverse Kinematics and retargeting to convert the human actions to robot actions. We fine-tune the model using a few robot manipulation demonstrations to obtain the robot policy, namely EgoVLA. We propose a simulation benchmark called Ego Humanoid Manipulation Benchmark, where we design diverse bimanual manipulation tasks with demonstrations. We fine-tune and evaluate EgoVLA with Ego Humanoid Manipulation Benchmark and show significant improvements over baselines and ablate the importance of human data. Videos can be found on our website: https://rchalyang.github.io/EgoVLA
>
---
#### [replaced 036] Towards scientific discovery with dictionary learning: Extracting biological concepts from microscopy foundation models
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2412.16247v3](http://arxiv.org/pdf/2412.16247v3)**

> **作者:** Konstantin Donhauser; Kristina Ulicna; Gemma Elyse Moran; Aditya Ravuri; Kian Kenyon-Dean; Cian Eastwood; Jason Hartford
>
> **摘要:** Sparse dictionary learning (DL) has emerged as a powerful approach to extract semantically meaningful concepts from the internals of large language models (LLMs) trained mainly in the text domain. In this work, we explore whether DL can extract meaningful concepts from less human-interpretable scientific data, such as vision foundation models trained on cell microscopy images, where limited prior knowledge exists about which high-level concepts should arise. We propose a novel combination of a sparse DL algorithm, Iterative Codebook Feature Learning (ICFL), with a PCA whitening pre-processing step derived from control data. Using this combined approach, we successfully retrieve biologically meaningful concepts, such as cell types and genetic perturbations. Moreover, we demonstrate how our method reveals subtle morphological changes arising from human-interpretable interventions, offering a promising new direction for scientific discovery via mechanistic interpretability in bioimaging.
>
---
#### [replaced 037] Geometry-Informed Neural Networks
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2402.14009v4](http://arxiv.org/pdf/2402.14009v4)**

> **作者:** Arturs Berzins; Andreas Radler; Eric Volkmann; Sebastian Sanokowski; Sepp Hochreiter; Johannes Brandstetter
>
> **备注:** Code available at https://github.com/ml-jku/GINNs-Geometry-informed-Neural-Networks
>
> **摘要:** Geometry is a ubiquitous tool in computer graphics, design, and engineering. However, the lack of large shape datasets limits the application of state-of-the-art supervised learning methods and motivates the exploration of alternative learning strategies. To this end, we introduce geometry-informed neural networks (GINNs) -- a framework for training shape-generative neural fields without data by leveraging user-specified design requirements in the form of objectives and constraints. By adding diversity as an explicit constraint, GINNs avoid mode-collapse and can generate multiple diverse solutions, often required in geometry tasks. Experimentally, we apply GINNs to several problems spanning physics, geometry, and engineering design, showing control over geometrical and topological properties, such as surface smoothness or the number of holes. These results demonstrate the potential of training shape-generative models without data, paving the way for new generative design approaches without large datasets.
>
---
#### [replaced 038] PosePilot: Steering Camera Pose for Generative World Models with Self-supervised Depth
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01729v2](http://arxiv.org/pdf/2505.01729v2)**

> **作者:** Bu Jin; Weize Li; Baihan Yang; Zhenxin Zhu; Junpeng Jiang; Huan-ang Gao; Haiyang Sun; Kun Zhan; Hengtong Hu; Xueyang Zhang; Peng Jia; Hao Zhao
>
> **备注:** Accepted at IEEE/RSJ IROS 2025
>
> **摘要:** Recent advancements in autonomous driving (AD) systems have highlighted the potential of world models in achieving robust and generalizable performance across both ordinary and challenging driving conditions. However, a key challenge remains: precise and flexible camera pose control, which is crucial for accurate viewpoint transformation and realistic simulation of scene dynamics. In this paper, we introduce PosePilot, a lightweight yet powerful framework that significantly enhances camera pose controllability in generative world models. Drawing inspiration from self-supervised depth estimation, PosePilot leverages structure-from-motion principles to establish a tight coupling between camera pose and video generation. Specifically, we incorporate self-supervised depth and pose readouts, allowing the model to infer depth and relative camera motion directly from video sequences. These outputs drive pose-aware frame warping, guided by a photometric warping loss that enforces geometric consistency across synthesized frames. To further refine camera pose estimation, we introduce a reverse warping step and a pose regression loss, improving viewpoint precision and adaptability. Extensive experiments on autonomous driving and general-domain video datasets demonstrate that PosePilot significantly enhances structural understanding and motion reasoning in both diffusion-based and auto-regressive world models. By steering camera pose with self-supervised depth, PosePilot sets a new benchmark for pose controllability, enabling physically consistent, reliable viewpoint synthesis in generative world models.
>
---
#### [replaced 039] SimpleGVR: A Simple Baseline for Latent-Cascaded Video Super-Resolution
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.19838v2](http://arxiv.org/pdf/2506.19838v2)**

> **作者:** Liangbin Xie; Yu Li; Shian Du; Menghan Xia; Xintao Wang; Fanghua Yu; Ziyan Chen; Pengfei Wan; Jiantao Zhou; Chao Dong
>
> **备注:** Project webpage available at https://simplegvr.github.io/
>
> **摘要:** Latent diffusion models have emerged as a leading paradigm for efficient video generation. However, as user expectations shift toward higher-resolution outputs, relying solely on latent computation becomes inadequate. A promising approach involves decoupling the process into two stages: semantic content generation and detail synthesis. The former employs a computationally intensive base model at lower resolutions, while the latter leverages a lightweight cascaded video super-resolution (VSR) model to achieve high-resolution output. In this work, we focus on studying key design principles for latter cascaded VSR models, which are underexplored currently. First, we propose two degradation strategies to generate training pairs that better mimic the output characteristics of the base model, ensuring alignment between the VSR model and its upstream generator. Second, we provide critical insights into VSR model behavior through systematic analysis of (1) timestep sampling strategies, (2) noise augmentation effects on low-resolution (LR) inputs. These findings directly inform our architectural and training innovations. Finally, we introduce interleaving temporal unit and sparse local attention to achieve efficient training and inference, drastically reducing computational overhead. Extensive experiments demonstrate the superiority of our framework over existing methods, with ablation studies confirming the efficacy of each design choice. Our work establishes a simple yet effective baseline for cascaded video super-resolution generation, offering practical insights to guide future advancements in efficient cascaded synthesis systems.
>
---
#### [replaced 040] Accelerating Diffusion Transformer via Gradient-Optimized Cache
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05156v2](http://arxiv.org/pdf/2503.05156v2)**

> **作者:** Junxiang Qiu; Lin Liu; Shuo Wang; Jinda Lu; Kezhou Chen; Yanbin Hao
>
> **摘要:** Feature caching has emerged as an effective strategy to accelerate diffusion transformer (DiT) sampling through temporal feature reuse. It is a challenging problem since (1) Progressive error accumulation from cached blocks significantly degrades generation quality, particularly when over 50\% of blocks are cached; (2) Current error compensation approaches neglect dynamic perturbation patterns during the caching process, leading to suboptimal error correction. To solve these problems, we propose the Gradient-Optimized Cache (GOC) with two key innovations: (1) Cached Gradient Propagation: A gradient queue dynamically computes the gradient differences between cached and recomputed features. These gradients are weighted and propagated to subsequent steps, directly compensating for the approximation errors introduced by caching. (2) Inflection-Aware Optimization: Through statistical analysis of feature variation patterns, we identify critical inflection points where the denoising trajectory changes direction. By aligning gradient updates with these detected phases, we prevent conflicting gradient directions during error correction. Extensive evaluations on ImageNet demonstrate GOC's superior trade-off between efficiency and quality. With 50\% cached blocks, GOC achieves IS 216.28 (26.3\% higher) and FID 3.907 (43\% lower) compared to baseline DiT, while maintaining identical computational costs. These improvements persist across various cache ratios, demonstrating robust adaptability to different acceleration requirements. Code is available at https://github.com/qiujx0520/GOC_ICCV2025.git.
>
---
#### [replaced 041] Hands-On: Segmenting Individual Signs from Continuous Sequences
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.08593v3](http://arxiv.org/pdf/2504.08593v3)**

> **作者:** JianHe Low; Harry Walsh; Ozge Mercanoglu Sincan; Richard Bowden
>
> **备注:** Accepted in the 19th IEEE International Conference on Automatic Face and Gesture Recognition
>
> **摘要:** This work tackles the challenge of continuous sign language segmentation, a key task with huge implications for sign language translation and data annotation. We propose a transformer-based architecture that models the temporal dynamics of signing and frames segmentation as a sequence labeling problem using the Begin-In-Out (BIO) tagging scheme. Our method leverages the HaMeR hand features, and is complemented with 3D Angles. Extensive experiments show that our model achieves state-of-the-art results on the DGS Corpus, while our features surpass prior benchmarks on BSLCorpus.
>
---
#### [replaced 042] ZonUI-3B: A Lightweight Vision-Language Model for Cross-Resolution GUI Grounding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.23491v3](http://arxiv.org/pdf/2506.23491v3)**

> **作者:** ZongHan Hsieh; Tzer-Jen Wei; ShengJing Yang
>
> **摘要:** In this paper, we present ZonUI-3B, a lightweight Vision-Language Model (VLM) that can be fully trained on a single consumer-grade GPU (RTX 4090) while delivering performance comparable to significantly larger models on GUI grounding tasks. The model incorporates several key innovations: (i) combine cross-platform, multi-resolution dataset of 24K examples from diverse sources including mobile, desktop, and web GUI screenshots to effectively address data scarcity in high-resolution desktop environments; (ii) a two-stage fine-tuning strategy, where initial cross-platform training establishes robust GUI understanding, followed by specialized fine-tuning on high-resolution data to significantly enhance model adaptability; and (iii) data curation and redundancy reduction strategies, demonstrating that randomly sampling a smaller subset with reduced redundancy achieves performance comparable to larger datasets, emphasizing data diversity over sheer volume. Empirical evaluation on standard GUI grounding benchmarks, including ScreenSpot, ScreenSpot-v2, and the challenging ScreenSpot-Pro, highlights ZonUI-3B's exceptional accuracy, achieving 84.9% on ScreenSpot and 86.4% on ScreenSpot-v2, surpassing prior models under 4B parameters. Ablation studies validate the critical role of balanced sampling and two-stage fine-tuning in enhancing robustness, particularly in high-resolution desktop scenarios. The ZonUI-3B is available at: https://github.com/Han1018/ZonUI-3B
>
---
#### [replaced 043] Scalable Frame Sampling for Video Classification: A Semi-Optimal Policy Approach with Reduced Search Space
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.05260v2](http://arxiv.org/pdf/2409.05260v2)**

> **作者:** Junho Lee; Jeongwoo Shin; Seung Woo Ko; Seongsu Ha; Joonseok Lee
>
> **摘要:** Given a video with $T$ frames, frame sampling is a task to select $N \ll T$ frames, so as to maximize the performance of a fixed video classifier. Not just brute-force search, but most existing methods suffer from its vast search space of $\binom{T}{N}$, especially when $N$ gets large. To address this challenge, we introduce a novel perspective of reducing the search space from $O(T^N)$ to $O(T)$. Instead of exploring the entire $O(T^N)$ space, our proposed semi-optimal policy selects the top $N$ frames based on the independently estimated value of each frame using per-frame confidence, significantly reducing the computational complexity. We verify that our semi-optimal policy can efficiently approximate the optimal policy, particularly under practical settings. Additionally, through extensive experiments on various datasets and model architectures, we demonstrate that learning our semi-optimal policy ensures stable and high performance regardless of the size of $N$ and $T$.
>
---
#### [replaced 044] DiffAD: A Unified Diffusion Modeling Approach for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.12170v2](http://arxiv.org/pdf/2503.12170v2)**

> **作者:** Tao Wang; Cong Zhang; Xingguang Qu; Kun Li; Weiwei Liu; Chang Huang
>
> **备注:** 8 pages, 6 figures; Code released
>
> **摘要:** End-to-end autonomous driving (E2E-AD) has rapidly emerged as a promising approach toward achieving full autonomy. However, existing E2E-AD systems typically adopt a traditional multi-task framework, addressing perception, prediction, and planning tasks through separate task-specific heads. Despite being trained in a fully differentiable manner, they still encounter issues with task coordination, and the system complexity remains high. In this work, we introduce DiffAD, a novel diffusion probabilistic model that redefines autonomous driving as a conditional image generation task. By rasterizing heterogeneous targets onto a unified bird's-eye view (BEV) and modeling their latent distribution, DiffAD unifies various driving objectives and jointly optimizes all driving tasks in a single framework, significantly reducing system complexity and harmonizing task coordination. The reverse process iteratively refines the generated BEV image, resulting in more robust and realistic driving behaviors. Closed-loop evaluations in Carla demonstrate the superiority of the proposed method, achieving a new state-of-the-art Success Rate and Driving Score.
>
---
#### [replaced 045] Mind the Modality Gap: Towards a Remote Sensing Vision-Language Model via Cross-modal Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2402.09816v2](http://arxiv.org/pdf/2402.09816v2)**

> **作者:** Angelos Zavras; Dimitrios Michail; Begüm Demir; Ioannis Papoutsis
>
> **备注:** Accepted at the ISPRS Journal of Photogrammetry and Remote Sensing. Our code implementation and weights for all experiments are publicly available at https://github.com/Orion-AI-Lab/MindTheModalityGap
>
> **摘要:** Deep Learning (DL) is undergoing a paradigm shift with the emergence of foundation models. In this work, we focus on Contrastive Language-Image Pre-training (CLIP), a Vision-Language foundation model that achieves high accuracy across various image classification tasks and often rivals fully supervised baselines, despite not being explicitly trained for those tasks. Nevertheless, there are still domains where zero-shot CLIP performance is far from optimal, such as Remote Sensing (RS) and medical imagery. These domains do not only exhibit fundamentally different distributions compared to natural images, but also commonly rely on complementary modalities, beyond RGB, to derive meaningful insights. To this end, we propose a methodology to align distinct RS image modalities with the visual and textual modalities of CLIP. Our two-stage procedure addresses the aforementioned distribution shift, extends the zero-shot capabilities of CLIP and enriches CLIP's shared embedding space with domain-specific knowledge. Initially, we robustly fine-tune CLIP according to the PAINT (Ilharco et al., 2022) patching protocol, in order to deal with the distribution shift. Building upon this foundation, we facilitate the cross-modal alignment of a RS modality encoder by distilling knowledge from the CLIP visual and textual encoders. We empirically show that both patching and cross-modal alignment translate to significant performance gains, across several RS imagery classification and cross-modal retrieval benchmark datasets. Notably, these enhancements are achieved without the reliance on textual descriptions, without introducing any task-specific parameters, without training from scratch and without catastrophic forgetting. We make our code implementation and weights for all experiments publicly available at https://github.com/Orion-AI-Lab/MindTheModalityGap.
>
---
#### [replaced 046] LLM-driven Medical Report Generation via Communication-efficient Heterogeneous Federated Learning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.17562v2](http://arxiv.org/pdf/2506.17562v2)**

> **作者:** Haoxuan Che; Haibo Jin; Zhengrui Guo; Yi Lin; Cheng Jin; Hao Chen
>
> **备注:** Accepted by IEEE TMI
>
> **摘要:** LLMs have demonstrated significant potential in Medical Report Generation (MRG), yet their development requires large amounts of medical image-report pairs, which are commonly scattered across multiple centers. Centralizing these data is exceptionally challenging due to privacy regulations, thereby impeding model development and broader adoption of LLM-driven MRG models. To address this challenge, we present FedMRG, the first framework that leverages Federated Learning (FL) to enable privacy-preserving, multi-center development of LLM-driven MRG models, specifically designed to overcome the critical challenge of communication-efficient LLM training under multi-modal data heterogeneity. To start with, our framework tackles the fundamental challenge of communication overhead in FL-LLM tuning by employing low-rank factorization to efficiently decompose parameter updates, significantly reducing gradient transmission costs and making LLM-driven MRG feasible in bandwidth-constrained FL settings. Furthermore, we observed the dual heterogeneity in MRG under the FL scenario: varying image characteristics across medical centers, as well as diverse reporting styles and terminology preferences. To address this, we further enhance FedMRG with (1) client-aware contrastive learning in the MRG encoder, coupled with diagnosis-driven prompts, which capture both globally generalizable and locally distinctive features while maintaining diagnostic accuracy; and (2) a dual-adapter mutual boosting mechanism in the MRG decoder that harmonizes generic and specialized adapters to address variations in reporting styles and terminology. Through extensive evaluation of our established FL-MRG benchmark, we demonstrate the generalizability and adaptability of FedMRG, underscoring its potential in harnessing multi-center data and generating clinically accurate reports while maintaining communication efficiency.
>
---
#### [replaced 047] Inversion-DPO: Precise and Efficient Post-Training for Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.11554v2](http://arxiv.org/pdf/2507.11554v2)**

> **作者:** Zejian Li; Yize Li; Chenye Meng; Zhongni Liu; Yang Ling; Shengyuan Zhang; Guang Yang; Changyuan Yang; Zhiyuan Yang; Lingyun Sun
>
> **摘要:** Recent advancements in diffusion models (DMs) have been propelled by alignment methods that post-train models to better conform to human preferences. However, these approaches typically require computation-intensive training of a base model and a reward model, which not only incurs substantial computational overhead but may also compromise model accuracy and training efficiency. To address these limitations, we propose Inversion-DPO, a novel alignment framework that circumvents reward modeling by reformulating Direct Preference Optimization (DPO) with DDIM inversion for DMs. Our method conducts intractable posterior sampling in Diffusion-DPO with the deterministic inversion from winning and losing samples to noise and thus derive a new post-training paradigm. This paradigm eliminates the need for auxiliary reward models or inaccurate appromixation, significantly enhancing both precision and efficiency of training. We apply Inversion-DPO to a basic task of text-to-image generation and a challenging task of compositional image generation. Extensive experiments show substantial performance improvements achieved by Inversion-DPO compared to existing post-training methods and highlight the ability of the trained generative models to generate high-fidelity compositionally coherent images. For the post-training of compostitional image geneation, we curate a paired dataset consisting of 11,140 images with complex structural annotations and comprehensive scores, designed to enhance the compositional capabilities of generative models. Inversion-DPO explores a new avenue for efficient, high-precision alignment in diffusion models, advancing their applicability to complex realistic generation tasks. Our code is available at https://github.com/MIGHTYEZ/Inversion-DPO
>
---
#### [replaced 048] GeoMag: A Vision-Language Model for Pixel-level Fine-Grained Remote Sensing Image Parsing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.05887v2](http://arxiv.org/pdf/2507.05887v2)**

> **作者:** Xianzhi Ma; Jianhui Li; Changhua Pei; Hao Liu
>
> **摘要:** The application of Vision-Language Models (VLMs) in remote sensing (RS) image understanding has achieved notable progress, demonstrating the basic ability to recognize and describe geographical entities. However, existing RS-VLMs are mostly limited to image-level and region-level tasks, lacking the capability to handle pixel-level tasks and performing poorly in small-object recognition scenarios. Moreover, RS-VLMs consume significant computational resources when processing high-resolution RS images, further restricting their practical applicability. In this context, we propose GeoMag (Geographical Magnifier), an end-to-end general-purpose large model framework for RS. GeoMag dynamically focuses the attention scope based on prompt semantics to effectively perform remote sensing image parsing across multiple levels of granularity. This method introduces Task-driven Multi-granularity Resolution Adjustment (TMRA) and Prompt-guided Semantic-aware Cropping (PSC), which adaptively reduce the spatial resolution of task-irrelevant regions while enhancing the visual representation of task-relevant areas. This approach improves the model's perception of critical target regions, suppresses background redundancy, and reduces the computational cost of interpreting high-resolution RS imagery. Extensive comparative experiments on 10 benchmarks demonstrate that GeoMag not only excels in handling pixel-level tasks but also maintains competitive performance across tasks of other granularities compared to existing RS-VLMs.
>
---
#### [replaced 049] FDSG: Forecasting Dynamic Scene Graphs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.01487v2](http://arxiv.org/pdf/2506.01487v2)**

> **作者:** Yi Yang; Yuren Cong; Hao Cheng; Bodo Rosenhahn; Michael Ying Yang
>
> **备注:** 16 pages, 8 figures, 12 tables
>
> **摘要:** Dynamic scene graph generation extends scene graph generation from images to videos by modeling entity relationships and their temporal evolution. However, existing methods either generate scene graphs from observed frames without explicitly modeling temporal dynamics, or predict only relationships while assuming static entity labels and locations. These limitations hinder effective extrapolation of both entity and relationship dynamics, restricting video scene understanding. We propose Forecasting Dynamic Scene Graphs (FDSG), a novel framework that predicts future entity labels, bounding boxes, and relationships, for unobserved frames, while also generating scene graphs for observed frames. Our scene graph forecast module leverages query decomposition and neural stochastic differential equations to model entity and relationship dynamics. A temporal aggregation module further refines predictions by integrating forecasted and observed information via cross-attention. To benchmark FDSG, we introduce Scene Graph Forecasting, a new task for full future scene graph prediction. Experiments on Action Genome show that FDSG outperforms state-of-the-art methods on dynamic scene graph generation, scene graph anticipation, and scene graph forecasting. Codes will be released upon publication.
>
---
#### [replaced 050] A General Framework for Inference-time Scaling and Steering of Diffusion Models
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.06848v5](http://arxiv.org/pdf/2501.06848v5)**

> **作者:** Raghav Singhal; Zachary Horvitz; Ryan Teehan; Mengye Ren; Zhou Yu; Kathleen McKeown; Rajesh Ranganath
>
> **摘要:** Diffusion models produce impressive results in modalities ranging from images and video to protein design and text. However, generating samples with user-specified properties remains a challenge. Recent research proposes fine-tuning models to maximize rewards that capture desired properties, but these methods require expensive training and are prone to mode collapse. In this work, we present Feynman-Kac (FK) steering, an inference-time framework for steering diffusion models with reward functions. FK steering works by sampling a system of multiple interacting diffusion processes, called particles, and resampling particles at intermediate steps based on scores computed using functions called potentials. Potentials are defined using rewards for intermediate states and are selected such that a high value indicates that the particle will yield a high-reward sample. We explore various choices of potentials, intermediate rewards, and samplers. We evaluate FK steering on text-to-image and text diffusion models. For steering text-to-image models with a human preference reward, we find that FK steering a 0.8B parameter model outperforms a 2.6B parameter fine-tuned model on prompt fidelity, with faster sampling and no training. For steering text diffusion models with rewards for text quality and specific text attributes, we find that FK steering generates lower perplexity, more linguistically acceptable outputs and enables gradient-free control of attributes like toxicity. Our results demonstrate that inference-time scaling and steering of diffusion models - even with off-the-shelf rewards - can provide significant sample quality gains and controllability benefits. Code is available at https://github.com/zacharyhorvitz/Fk-Diffusion-Steering .
>
---
#### [replaced 051] Demographic-aware fine-grained classification of pediatric wrist fractures
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.12964v2](http://arxiv.org/pdf/2507.12964v2)**

> **作者:** Ammar Ahmed; Ali Shariq Imran; Zenun Kastrati; Sher Muhammad Daudpota
>
> **摘要:** Wrist pathologies are frequently observed, particularly among children who constitute the majority of fracture cases. However, diagnosing these conditions is time-consuming and requires specialized expertise. Computer vision presents a promising avenue, contingent upon the availability of extensive datasets, a notable challenge in medical imaging. Therefore, reliance solely on one modality, such as images, proves inadequate, especially in an era of diverse and plentiful data types. In this study, we employ a multifaceted approach to address the challenge of recognizing wrist pathologies using an extremely limited dataset. Initially, we approach the problem as a fine-grained recognition task, aiming to identify subtle X-ray pathologies that conventional CNNs overlook. Secondly, we enhance network performance by fusing patient metadata with X-ray images. Thirdly, rather than pre-training on a coarse-grained dataset like ImageNet, we utilize weights trained on a fine-grained dataset. While metadata integration has been used in other medical domains, this is a novel application for wrist pathologies. Our results show that a fine-grained strategy and metadata integration improve diagnostic accuracy by 2% with a limited dataset and by over 10% with a larger fracture-focused dataset.
>
---
#### [replaced 052] A Simple Baseline for Stable and Plastic Neural Networks
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.10637v2](http://arxiv.org/pdf/2507.10637v2)**

> **作者:** Étienne Künzel; Achref Jaziri; Visvanathan Ramesh
>
> **备注:** 11 pages, 50 figures
>
> **摘要:** Continual learning in computer vision requires that models adapt to a continuous stream of tasks without forgetting prior knowledge, yet existing approaches often tip the balance heavily toward either plasticity or stability. We introduce RDBP, a simple, low-overhead baseline that unites two complementary mechanisms: ReLUDown, a lightweight activation modification that preserves feature sensitivity while preventing neuron dormancy, and Decreasing Backpropagation, a biologically inspired gradient-scheduling scheme that progressively shields early layers from catastrophic updates. Evaluated on the Continual ImageNet benchmark, RDBP matches or exceeds the plasticity and stability of state-of-the-art methods while reducing computational cost. RDBP thus provides both a practical solution for real-world continual learning and a clear benchmark against which future continual learning strategies can be measured.
>
---
#### [replaced 053] On Pre-training of Multimodal Language Models Customized for Chart Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.14506v3](http://arxiv.org/pdf/2407.14506v3)**

> **作者:** Wan-Cyuan Fan; Yen-Chun Chen; Mengchen Liu; Lu Yuan; Leonid Sigal
>
> **备注:** NeurIPS 2024 Workshop on Adaptive Foundation Models
>
> **摘要:** Recent studies customizing Multimodal Large Language Models (MLLMs) for domain-specific tasks have yielded promising results, especially in the field of scientific chart comprehension. These studies generally utilize visual instruction tuning with specialized datasets to enhance question and answer (QA) accuracy within the chart domain. However, they often neglect the fundamental discrepancy between natural image-caption pre-training data and digital chart image-QA data, particularly in the models' capacity to extract underlying numeric values from charts. This paper tackles this oversight by exploring the training processes necessary to improve MLLMs' comprehension of charts. We present three key findings: (1) Incorporating raw data values in alignment pre-training markedly improves comprehension of chart data. (2) Replacing images with their textual representation randomly during end-to-end fine-tuning transfer the language reasoning capability to chart interpretation skills. (3) Requiring the model to first extract the underlying chart data and then answer the question in the fine-tuning can further improve the accuracy. Consequently, we introduce CHOPINLLM, an MLLM tailored for in-depth chart comprehension. CHOPINLLM effectively interprets various types of charts, including unannotated ones, while maintaining robust reasoning abilities. Furthermore, we establish a new benchmark to evaluate MLLMs' understanding of different chart types across various comprehension levels. Experimental results show that CHOPINLLM exhibits strong performance in understanding both annotated and unannotated charts across a wide range of types.
>
---
#### [replaced 054] SIC: Similarity-Based Interpretable Image Classification with Neural Networks
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.17328v3](http://arxiv.org/pdf/2501.17328v3)**

> **作者:** Tom Nuno Wolf; Emre Kavak; Fabian Bongratz; Christian Wachinger
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** The deployment of deep learning models in critical domains necessitates a balance between high accuracy and interpretability. We introduce SIC, an inherently interpretable neural network that provides local and global explanations of its decision-making process. Leveraging the concept of case-based reasoning, SIC extracts class-representative support vectors from training images, ensuring they capture relevant features while suppressing irrelevant ones. Classification decisions are made by calculating and aggregating similarity scores between these support vectors and the input's latent feature vector. We employ B-Cos transformations, which align model weights with inputs, to yield coherent pixel-level explanations in addition to global explanations of case-based reasoning. We evaluate SIC on three tasks: fine-grained classification on Stanford Dogs and FunnyBirds, multi-label classification on Pascal VOC, and pathology detection on the RSNA dataset. Results indicate that SIC not only achieves competitive accuracy compared to state-of-the-art black-box and inherently interpretable models but also offers insightful explanations verified through practical evaluation on the FunnyBirds benchmark. Our theoretical analysis proves that these explanations fulfill established axioms for explanations. Our findings underscore SIC's potential for applications where understanding model decisions is as critical as the decisions themselves.
>
---
#### [replaced 055] Computer-Vision-Enabled Worker Video Analysis for Motion Amount Quantification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.13999v3](http://arxiv.org/pdf/2405.13999v3)**

> **作者:** Hari Iyer; Neel Macwan; Shenghan Guo; Heejin Jeong
>
> **摘要:** The performance of physical workers is significantly influenced by the extent of their motions. However, monitoring and assessing these motions remains a challenge. Recent advancements have enabled in-situ video analysis for real-time observation of worker behaviors. This paper introduces a novel framework for tracking and quantifying upper and lower limb motions, issuing alerts when critical thresholds are reached. Using joint position data from posture estimation, the framework employs Hotelling's $T^2$ statistic to quantify and monitor motion amounts. A significant positive correlation was noted between motion warnings and the overall NASA Task Load Index (TLX) workload rating (\textit{r} = 0.218, \textit{p} = 0.0024). A supervised Random Forest model trained on the collected motion data was benchmarked against multiple datasets including UCF Sports Action and UCF50, and was found to effectively generalize across environments, identifying ergonomic risk patterns with accuracies up to 94\%.
>
---
#### [replaced 056] Entropy Loss: An Interpretability Amplifier of 3D Object Detection Network for Intelligent Driving
- **分类: cs.CV; cs.AI; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2409.00839v2](http://arxiv.org/pdf/2409.00839v2)**

> **作者:** Haobo Yang; Shiyan Zhang; Zhuoyi Yang; Xinyu Zhang; Jilong Guo; Zongyou Yang; Jun Li
>
> **摘要:** With the increasing complexity of the traffic environment, the significance of safety perception in intelligent driving is intensifying. Traditional methods in the field of intelligent driving perception rely on deep learning, which suffers from limited interpretability, often described as a "black box." This paper introduces a novel type of loss function, termed "Entropy Loss," along with an innovative training strategy. Entropy Loss is formulated based on the functionality of feature compression networks within the perception model. Drawing inspiration from communication systems, the information transmission process in a feature compression network is expected to demonstrate steady changes in information volume and a continuous decrease in information entropy. By modeling network layer outputs as continuous random variables, we construct a probabilistic model that quantifies changes in information volume. Entropy Loss is then derived based on these expectations, guiding the update of network parameters to enhance network interpretability. Our experiments indicate that the Entropy Loss training strategy accelerates the training process. Utilizing the same 60 training epochs, the accuracy of 3D object detection models using Entropy Loss on the KITTI test set improved by up to 4.47\% compared to models without Entropy Loss, underscoring the method's efficacy. The implementation code is available at https://github.com/yhbcode000/Eloss-Interpretability.
>
---
