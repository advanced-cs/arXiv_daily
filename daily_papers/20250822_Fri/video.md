# 计算机视觉 cs.CV

- **最新发布 91 篇**

- **更新 68 篇**

## 最新发布

#### [new 001] MExECON: Multi-view Extended Explicit Clothed humans Optimized via Normal integration
- **分类: cs.CV**

- **简介: 该论文提出MExECON，用于多视角RGB图像重建穿着人类的3D模型，通过联合多视角优化和法线图整合提升几何与姿态估计精度。**

- **链接: [http://arxiv.org/pdf/2508.15500v1](http://arxiv.org/pdf/2508.15500v1)**

> **作者:** Fulden Ece Uğur; Rafael Redondo; Albert Barreiro; Stefan Hristov; Roger Marí
>
> **摘要:** This work presents MExECON, a novel pipeline for 3D reconstruction of clothed human avatars from sparse multi-view RGB images. Building on the single-view method ECON, MExECON extends its capabilities to leverage multiple viewpoints, improving geometry and body pose estimation. At the core of the pipeline is the proposed Joint Multi-view Body Optimization (JMBO) algorithm, which fits a single SMPL-X body model jointly across all input views, enforcing multi-view consistency. The optimized body model serves as a low-frequency prior that guides the subsequent surface reconstruction, where geometric details are added via normal map integration. MExECON integrates normal maps from both front and back views to accurately capture fine-grained surface details such as clothing folds and hairstyles. All multi-view gains are achieved without requiring any network re-training. Experimental results show that MExECON consistently improves fidelity over the single-view baseline and achieves competitive performance compared to modern few-shot 3D reconstruction methods.
>
---
#### [new 002] RATopo: Improving Lane Topology Reasoning via Redundancy Assignment
- **分类: cs.CV**

- **简介: 论文针对自动驾驶中车道拓扑推理的监督不足问题，提出RATopo策略，通过重构Transformer解码器与多并行注意力块，实现冗余预测保留与多对一分配，提升车道间及与交通元素的拓扑推理性能。**

- **链接: [http://arxiv.org/pdf/2508.15272v1](http://arxiv.org/pdf/2508.15272v1)**

> **作者:** Han Li; Shaofei Huang; Longfei Xu; Yulu Gao; Beipeng Mu; Si Liu
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Lane topology reasoning plays a critical role in autonomous driving by modeling the connections among lanes and the topological relationships between lanes and traffic elements. Most existing methods adopt a first-detect-then-reason paradigm, where topological relationships are supervised based on the one-to-one assignment results obtained during the detection stage. This supervision strategy results in suboptimal topology reasoning performance due to the limited range of valid supervision. In this paper, we propose RATopo, a Redundancy Assignment strategy for lane Topology reasoning that enables quantity-rich and geometry-diverse topology supervision. Specifically, we restructure the Transformer decoder by swapping the cross-attention and self-attention layers. This allows redundant lane predictions to be retained before suppression, enabling effective one-to-many assignment. We also instantiate multiple parallel cross-attention blocks with independent parameters, which further enhances the diversity of detected lanes. Extensive experiments on OpenLane-V2 demonstrate that our RATopo strategy is model-agnostic and can be seamlessly integrated into existing topology reasoning frameworks, consistently improving both lane-lane and lane-traffic topology performance.
>
---
#### [new 003] First RAG, Second SEG: A Training-Free Paradigm for Camouflaged Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 论文针对伪装物体检测（COD）任务，解决传统方法依赖训练和高质量提示的问题，提出RAG-SEG分两阶段处理：先用RAG生成粗掩码，再用SAM精修，无需训练，高效完成检测。**

- **链接: [http://arxiv.org/pdf/2508.15313v1](http://arxiv.org/pdf/2508.15313v1)**

> **作者:** Wutao Liu; YiDan Wang; Pan Gao
>
> **摘要:** Camouflaged object detection (COD) poses a significant challenge in computer vision due to the high similarity between objects and their backgrounds. Existing approaches often rely on heavy training and large computational resources. While foundation models such as the Segment Anything Model (SAM) offer strong generalization, they still struggle to handle COD tasks without fine-tuning and require high-quality prompts to yield good performance. However, generating such prompts manually is costly and inefficient. To address these challenges, we propose \textbf{First RAG, Second SEG (RAG-SEG)}, a training-free paradigm that decouples COD into two stages: Retrieval-Augmented Generation (RAG) for generating coarse masks as prompts, followed by SAM-based segmentation (SEG) for refinement. RAG-SEG constructs a compact retrieval database via unsupervised clustering, enabling fast and effective feature retrieval. During inference, the retrieved features produce pseudo-labels that guide precise mask generation using SAM2. Our method eliminates the need for conventional training while maintaining competitive performance. Extensive experiments on benchmark COD datasets demonstrate that RAG-SEG performs on par with or surpasses state-of-the-art methods. Notably, all experiments are conducted on a \textbf{personal laptop}, highlighting the computational efficiency and practicality of our approach. We present further analysis in the Appendix, covering limitations, salient object detection extension, and possible improvements.
>
---
#### [new 004] Backpropagation-Free Test-Time Adaptation via Probabilistic Gaussian Alignment
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出ADAPT方法，解决测试时适应（TTA）中依赖反向传播和缺乏类条件分布建模的问题。通过概率高斯对齐建模，实现无梯度、无源数据的测试时适应，提升跨分布鲁棒性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2508.15568v1](http://arxiv.org/pdf/2508.15568v1)**

> **作者:** Youjia Zhang; Youngeun Kim; Young-Geun Choi; Hongyeob Kim; Huiling Liu; Sungeun Hong
>
> **摘要:** Test-time adaptation (TTA) enhances the zero-shot robustness under distribution shifts by leveraging unlabeled test data during inference. Despite notable advances, several challenges still limit its broader applicability. First, most methods rely on backpropagation or iterative optimization, which limits scalability and hinders real-time deployment. Second, they lack explicit modeling of class-conditional feature distributions. This modeling is crucial for producing reliable decision boundaries and calibrated predictions, but it remains underexplored due to the lack of both source data and supervision at test time. In this paper, we propose ADAPT, an Advanced Distribution-Aware and backPropagation-free Test-time adaptation method. We reframe TTA as a Gaussian probabilistic inference task by modeling class-conditional likelihoods using gradually updated class means and a shared covariance matrix. This enables closed-form, training-free inference. To correct potential likelihood bias, we introduce lightweight regularization guided by CLIP priors and a historical knowledge bank. ADAPT requires no source data, no gradient updates, and no full access to target data, supporting both online and transductive settings. Extensive experiments across diverse benchmarks demonstrate that our method achieves state-of-the-art performance under a wide range of distribution shifts with superior scalability and robustness.
>
---
#### [new 005] High-Frequency First: A Two-Stage Approach for Improving Image INR
- **分类: cs.CV**

- **简介: 该论文针对隐式神经表示（INR）的频谱偏差问题，提出两阶段训练策略，通过邻居感知的软掩码优先学习高频细节，提升图像重建质量。**

- **链接: [http://arxiv.org/pdf/2508.15582v1](http://arxiv.org/pdf/2508.15582v1)**

> **作者:** Sumit Kumar Dam; Mrityunjoy Gain; Eui-Nam Huh; Choong Seon Hong
>
> **备注:** Paper on INR; 4 figures, 8 pages
>
> **摘要:** Implicit Neural Representations (INRs) have emerged as a powerful alternative to traditional pixel-based formats by modeling images as continuous functions over spatial coordinates. A key challenge, however, lies in the spectral bias of neural networks, which tend to favor low-frequency components while struggling to capture high-frequency (HF) details such as sharp edges and fine textures. While prior approaches have addressed this limitation through architectural modifications or specialized activation functions, we propose an orthogonal direction by directly guiding the training process. Specifically, we introduce a two-stage training strategy where a neighbor-aware soft mask adaptively assigns higher weights to pixels with strong local variations, encouraging early focus on fine details. The model then transitions to full-image training. Experimental results show that our approach consistently improves reconstruction quality and complements existing INR methods. As a pioneering attempt to assign frequency-aware importance to pixels in image INR, our work offers a new avenue for mitigating the spectral bias problem.
>
---
#### [new 006] DyMorph-B2I: Dynamic and Morphology-Guided Binary-to-Instance Segmentation for Renal Pathology
- **分类: cs.CV**

- **简介: 论文提出DyMorph-B2I，解决肾脏病理中二值掩码到实例分割的难题，通过整合形态学方法与参数优化，提升分割精度。**

- **链接: [http://arxiv.org/pdf/2508.15208v1](http://arxiv.org/pdf/2508.15208v1)**

> **作者:** Leiyue Zhao; Yuechen Yang; Yanfan Zhu; Haichun Yang; Yuankai Huo; Paul D. Simonson; Kenji Ikemura; Mert R. Sabuncu; Yihe Yang; Ruining Deng
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Accurate morphological quantification of renal pathology functional units relies on instance-level segmentation, yet most existing datasets and automated methods provide only binary (semantic) masks, limiting the precision of downstream analyses. Although classical post-processing techniques such as watershed, morphological operations, and skeletonization, are often used to separate semantic masks into instances, their individual effectiveness is constrained by the diverse morphologies and complex connectivity found in renal tissue. In this study, we present DyMorph-B2I, a dynamic, morphology-guided binary-to-instance segmentation pipeline tailored for renal pathology. Our approach integrates watershed, skeletonization, and morphological operations within a unified framework, complemented by adaptive geometric refinement and customizable hyperparameter tuning for each class of functional unit. Through systematic parameter optimization, DyMorph-B2I robustly separates adherent and heterogeneous structures present in binary masks. Experimental results demonstrate that our method outperforms individual classical approaches and na\"ive combinations, enabling superior instance separation and facilitating more accurate morphometric analysis in renal pathology workflows. The pipeline is publicly available at: https://github.com/ddrrnn123/DyMorph-B2I.
>
---
#### [new 007] XDR-LVLM: An Explainable Vision-Language Large Model for Diabetic Retinopathy Diagnosis
- **分类: cs.CV**

- **简介: 该论文针对糖尿病视网膜病变（DR）诊断中的模型可解释性问题，提出XDR-LVLM框架，结合视觉-语言大模型实现高精度诊断，并生成包含病理特征解释的可读报告，提升临床应用可靠性。**

- **链接: [http://arxiv.org/pdf/2508.15168v1](http://arxiv.org/pdf/2508.15168v1)**

> **作者:** Masato Ito; Kaito Tanaka; Keisuke Matsuda; Aya Nakayama
>
> **摘要:** Diabetic Retinopathy (DR) is a major cause of global blindness, necessitating early and accurate diagnosis. While deep learning models have shown promise in DR detection, their black-box nature often hinders clinical adoption due to a lack of transparency and interpretability. To address this, we propose XDR-LVLM (eXplainable Diabetic Retinopathy Diagnosis with LVLM), a novel framework that leverages Vision-Language Large Models (LVLMs) for high-precision DR diagnosis coupled with natural language-based explanations. XDR-LVLM integrates a specialized Medical Vision Encoder, an LVLM Core, and employs Multi-task Prompt Engineering and Multi-stage Fine-tuning to deeply understand pathological features within fundus images and generate comprehensive diagnostic reports. These reports explicitly include DR severity grading, identification of key pathological concepts (e.g., hemorrhages, exudates, microaneurysms), and detailed explanations linking observed features to the diagnosis. Extensive experiments on the Diabetic Retinopathy (DDR) dataset demonstrate that XDR-LVLM achieves state-of-the-art performance, with a Balanced Accuracy of 84.55% and an F1 Score of 79.92% for disease diagnosis, and superior results for concept detection (77.95% BACC, 66.88% F1). Furthermore, human evaluations confirm the high fluency, accuracy, and clinical utility of the generated explanations, showcasing XDR-LVLM's ability to bridge the gap between automated diagnosis and clinical needs by providing robust and interpretable insights.
>
---
#### [new 008] Multi-perspective monitoring of wildlife and human activities from camera traps and drones with deep learning models
- **分类: cs.CV**

- **简介: 该论文通过结合相机陷阱与无人机图像，利用深度学习模型监测野生动物与人类活动的空间分布，解决人兽冲突检测问题，通过模型训练与空间分析识别活动热点及潜在冲突区。**

- **链接: [http://arxiv.org/pdf/2508.15629v1](http://arxiv.org/pdf/2508.15629v1)**

> **作者:** Hao Chen; Fang Qiu; Li An; Douglas Stow; Eve Bohnett; Haitao Lyu; Shuang Tian
>
> **摘要:** Wildlife and human activities are key components of landscape systems. Understanding their spatial distribution is essential for evaluating human wildlife interactions and informing effective conservation planning. Multiperspective monitoring of wildlife and human activities by combining camera traps and drone imagery. Capturing the spatial patterns of their distributions, which allows the identification of the overlap of their activity zones and the assessment of the degree of human wildlife conflict. The study was conducted in Chitwan National Park (CNP), Nepal, and adjacent regions. Images collected by visible and nearinfrared camera traps and thermal infrared drones from February to July 2022 were processed to create training and testing datasets, which were used to build deep learning models to automatic identify wildlife and human activities. Drone collected thermal imagery was used for detecting targets to provide a multiple monitoring perspective. Spatial pattern analysis was performed to identify animal and resident activity hotspots and delineation potential human wildlife conflict zones. Among the deep learning models tested, YOLOv11s achieved the highest performance with a precision of 96.2%, recall of 92.3%, mAP50 of 96.7%, and mAP50 of 81.3%, making it the most effective for detecting objects in camera trap imagery. Drone based thermal imagery, analyzed with an enhanced Faster RCNN model, added a complementary aerial viewpoint for camera trap detections. Spatial pattern analysis identified clear hotspots for both wildlife and human activities and their overlapping patterns within certain areas in the CNP and buffer zones indicating potential conflict. This study reveals human wildlife conflicts within the conserved landscape. Integrating multiperspective monitoring with automated object detection enhances wildlife surveillance and landscape management.
>
---
#### [new 009] D3FNet: A Differential Attention Fusion Network for Fine-Grained Road Structure Extraction in Remote Perception Systems
- **分类: cs.CV**

- **简介: 论文提出D3FNet，用于细粒度道路结构提取，解决狭窄、遮挡道路检测难题。引入DADE模块增强特征、DDFM融合特征、多尺度空洞策略提升连续性，实验验证其优越性能。**

- **链接: [http://arxiv.org/pdf/2508.15537v1](http://arxiv.org/pdf/2508.15537v1)**

> **作者:** Chang Liu; Yang Xu; Tamas Sziranyi
>
> **备注:** 10 pages, 6 figures, International Conference on Computer Vision, ICCV 2025 (DriveX) paper id 5
>
> **摘要:** Extracting narrow roads from high-resolution remote sensing imagery remains a significant challenge due to their limited width, fragmented topology, and frequent occlusions. To address these issues, we propose D3FNet, a Dilated Dual-Stream Differential Attention Fusion Network designed for fine-grained road structure segmentation in remote perception systems. Built upon the encoder-decoder backbone of D-LinkNet, D3FNet introduces three key innovations:(1) a Differential Attention Dilation Extraction (DADE) module that enhances subtle road features while suppressing background noise at the bottleneck; (2) a Dual-stream Decoding Fusion Mechanism (DDFM) that integrates original and attention-modulated features to balance spatial precision with semantic context; and (3) a multi-scale dilation strategy (rates 1, 3, 5, 9) that mitigates gridding artifacts and improves continuity in narrow road prediction. Unlike conventional models that overfit to generic road widths, D3FNet specifically targets fine-grained, occluded, and low-contrast road segments. Extensive experiments on the DeepGlobe and CHN6-CUG benchmarks show that D3FNet achieves superior IoU and recall on challenging road regions, outperforming state-of-the-art baselines. Ablation studies further verify the complementary synergy of attention-guided encoding and dual-path decoding. These results confirm D3FNet as a robust solution for fine-grained narrow road extraction in complex remote and cooperative perception scenarios.
>
---
#### [new 010] StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding
- **分类: cs.CV; cs.AI**

- **简介: 论文针对长视频理解中的内存与计算开销问题，提出StreamMem机制，通过流式编码与注意力压缩实现查询无关的KV缓存，提升多轮对话场景下的效率，达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2508.15717v1](http://arxiv.org/pdf/2508.15717v1)**

> **作者:** Yanlai Yang; Zhuokai Zhao; Satya Narayan Shukla; Aashu Singh; Shlok Kumar Mishra; Lizhu Zhang; Mengye Ren
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** Multimodal large language models (MLLMs) have made significant progress in visual-language reasoning, but their ability to efficiently handle long videos remains limited. Despite recent advances in long-context MLLMs, storing and attending to the key-value (KV) cache for long visual contexts incurs substantial memory and computational overhead. Existing visual compression methods require either encoding the entire visual context before compression or having access to the questions in advance, which is impractical for long video understanding and multi-turn conversational settings. In this work, we propose StreamMem, a query-agnostic KV cache memory mechanism for streaming video understanding. Specifically, StreamMem encodes new video frames in a streaming manner, compressing the KV cache using attention scores between visual tokens and generic query tokens, while maintaining a fixed-size KV memory to enable efficient question answering (QA) in memory-constrained, long-video scenarios. Evaluation on three long video understanding and two streaming video question answering benchmarks shows that StreamMem achieves state-of-the-art performance in query-agnostic KV cache compression and is competitive with query-aware compression approaches.
>
---
#### [new 011] TAIGen: Training-Free Adversarial Image Generation via Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出TAIGen，通过扩散模型实现无需训练的高效对抗图像生成，解决传统方法低效高耗的问题，采用选择性通道策略与局部扰动，提升生成速度与攻击成功率。**

- **链接: [http://arxiv.org/pdf/2508.15020v1](http://arxiv.org/pdf/2508.15020v1)**

> **作者:** Susim Roy; Anubhooti Jain; Mayank Vatsa; Richa Singh
>
> **备注:** Accepted at ICCVW-CV4BIOM 2025
>
> **摘要:** Adversarial attacks from generative models often produce low-quality images and require substantial computational resources. Diffusion models, though capable of high-quality generation, typically need hundreds of sampling steps for adversarial generation. This paper introduces TAIGen, a training-free black-box method for efficient adversarial image generation. TAIGen produces adversarial examples using only 3-20 sampling steps from unconditional diffusion models. Our key finding is that perturbations injected during the mixing step interval achieve comparable attack effectiveness without processing all timesteps. We develop a selective RGB channel strategy that applies attention maps to the red channel while using GradCAM-guided perturbations on green and blue channels. This design preserves image structure while maximizing misclassification in target models. TAIGen maintains visual quality with PSNR above 30 dB across all tested datasets. On ImageNet with VGGNet as source, TAIGen achieves 70.6% success against ResNet, 80.8% against MNASNet, and 97.8% against ShuffleNet. The method generates adversarial examples 10x faster than existing diffusion-based attacks. Our method achieves the lowest robust accuracy, indicating it is the most impactful attack as the defense mechanism is least successful in purifying the images generated by TAIGen.
>
---
#### [new 012] Center-Oriented Prototype Contrastive Clustering
- **分类: cs.CV**

- **简介: 该论文针对对比学习聚类中类别冲突和原型偏差问题，提出中心导向的原型对比聚类框架，通过软原型模块和双一致性学习模块优化原型计算与特征对齐，提升聚类效果。**

- **链接: [http://arxiv.org/pdf/2508.15231v1](http://arxiv.org/pdf/2508.15231v1)**

> **作者:** Shihao Dong; Xiaotong Zhou; Yuhui Zheng; Huiying Xu; Xinzhong Zhu
>
> **摘要:** Contrastive learning is widely used in clustering tasks due to its discriminative representation. However, the conflict problem between classes is difficult to solve effectively. Existing methods try to solve this problem through prototype contrast, but there is a deviation between the calculation of hard prototypes and the true cluster center. To address this problem, we propose a center-oriented prototype contrastive clustering framework, which consists of a soft prototype contrastive module and a dual consistency learning module. In short, the soft prototype contrastive module uses the probability that the sample belongs to the cluster center as a weight to calculate the prototype of each category, while avoiding inter-class conflicts and reducing prototype drift. The dual consistency learning module aligns different transformations of the same sample and the neighborhoods of different samples respectively, ensuring that the features have transformation-invariant semantic information and compact intra-cluster distribution, while providing reliable guarantees for the calculation of prototypes. Extensive experiments on five datasets show that the proposed method is effective compared to the SOTA. Our code is published on https://github.com/LouisDong95/CPCC.
>
---
#### [new 013] From Linearity to Non-Linearity: How Masked Autoencoders Capture Spatial Correlations
- **分类: cs.CV**

- **简介: 论文研究MAEs如何通过不同超参数捕捉图像空间相关性，分析线性与非线性MAE的特征学习机制，探讨超参数对空间相关性捕捉的影响，并提出实践选择策略。**

- **链接: [http://arxiv.org/pdf/2508.15404v1](http://arxiv.org/pdf/2508.15404v1)**

> **作者:** Anthony Bisulco; Rahul Ramesh; Randall Balestriero; Pratik Chaudhari
>
> **摘要:** Masked Autoencoders (MAEs) have emerged as a powerful pretraining technique for vision foundation models. Despite their effectiveness, they require extensive hyperparameter tuning (masking ratio, patch size, encoder/decoder layers) when applied to novel datasets. While prior theoretical works have analyzed MAEs in terms of their attention patterns and hierarchical latent variable models, the connection between MAE hyperparameters and performance on downstream tasks is relatively unexplored. This work investigates how MAEs learn spatial correlations in the input image. We analytically derive the features learned by a linear MAE and show that masking ratio and patch size can be used to select for features that capture short- and long-range spatial correlations. We extend this analysis to non-linear MAEs to show that MAE representations adapt to spatial correlations in the dataset, beyond second-order statistics. Finally, we discuss some insights on how to select MAE hyper-parameters in practice.
>
---
#### [new 014] CM2LoD3: Reconstructing LoD3 Building Models Using Semantic Conflict Maps
- **分类: cs.CV; eess.IV**

- **简介: 该论文旨在解决LoD3建筑模型自动化重建难题，通过语义冲突图与合成冲突图生成相结合，提升3D城市建模效率与精度。**

- **链接: [http://arxiv.org/pdf/2508.15672v1](http://arxiv.org/pdf/2508.15672v1)**

> **作者:** Franz Hanke; Antonia Bieringer; Olaf Wysocki; Boris Jutzi
>
> **备注:** This paper was accepted for the 20th 3D GeoInfo & 9th Smart Data Smart Cities Conference
>
> **摘要:** Detailed 3D building models are crucial for urban planning, digital twins, and disaster management applications. While Level of Detail 1 (LoD)1 and LoD2 building models are widely available, they lack detailed facade elements essential for advanced urban analysis. In contrast, LoD3 models address this limitation by incorporating facade elements such as windows, doors, and underpasses. However, their generation has traditionally required manual modeling, making large-scale adoption challenging. In this contribution, CM2LoD3, we present a novel method for reconstructing LoD3 building models leveraging Conflict Maps (CMs) obtained from ray-to-model-prior analysis. Unlike previous works, we concentrate on semantically segmenting real-world CMs with synthetically generated CMs from our developed Semantic Conflict Map Generator (SCMG). We also observe that additional segmentation of textured models can be fused with CMs using confidence scores to further increase segmentation performance and thus increase 3D reconstruction accuracy. Experimental results demonstrate the effectiveness of our CM2LoD3 method in segmenting and reconstructing building openings, with the 61% performance with uncertainty-aware fusion of segmented building textures. This research contributes to the advancement of automated LoD3 model reconstruction, paving the way for scalable and efficient 3D city modeling. Our project is available: https://github.com/InFraHank/CM2LoD3
>
---
#### [new 015] Task-Generalized Adaptive Cross-Domain Learning for Multimodal Image Fusion
- **分类: cs.CV**

- **简介: 论文针对多模态图像融合中的模态对齐、高频细节丢失及任务限制问题，提出AdaSFFuse框架，通过自适应波let变换与频域Mamba模块实现跨域协同融合，提升融合效果与效率。**

- **链接: [http://arxiv.org/pdf/2508.15505v1](http://arxiv.org/pdf/2508.15505v1)**

> **作者:** Mengyu Wang; Zhenyu Liu; Kun Li; Yu Wang; Yuwei Wang; Yanyan Wei; Fei Wang
>
> **备注:** Accepted by IEEE Transactions on Multimedia
>
> **摘要:** Multimodal Image Fusion (MMIF) aims to integrate complementary information from different imaging modalities to overcome the limitations of individual sensors. It enhances image quality and facilitates downstream applications such as remote sensing, medical diagnostics, and robotics. Despite significant advancements, current MMIF methods still face challenges such as modality misalignment, high-frequency detail destruction, and task-specific limitations. To address these challenges, we propose AdaSFFuse, a novel framework for task-generalized MMIF through adaptive cross-domain co-fusion learning. AdaSFFuse introduces two key innovations: the Adaptive Approximate Wavelet Transform (AdaWAT) for frequency decoupling, and the Spatial-Frequency Mamba Blocks for efficient multimodal fusion. AdaWAT adaptively separates the high- and low-frequency components of multimodal images from different scenes, enabling fine-grained extraction and alignment of distinct frequency characteristics for each modality. The Spatial-Frequency Mamba Blocks facilitate cross-domain fusion in both spatial and frequency domains, enhancing this process. These blocks dynamically adjust through learnable mappings to ensure robust fusion across diverse modalities. By combining these components, AdaSFFuse improves the alignment and integration of multimodal features, reduces frequency loss, and preserves critical details. Extensive experiments on four MMIF tasks -- Infrared-Visible Image Fusion (IVF), Multi-Focus Image Fusion (MFF), Multi-Exposure Image Fusion (MEF), and Medical Image Fusion (MIF) -- demonstrate AdaSFFuse's superior fusion performance, ensuring both low computational cost and a compact network, offering a strong balance between performance and efficiency. The code will be publicly available at https://github.com/Zhen-yu-Liu/AdaSFFuse.
>
---
#### [new 016] An Empirical Study on How Video-LLMs Answer Video Questions
- **分类: cs.CV**

- **简介: 该论文研究Video-LLMs回答视频问题的内部机制，通过设计三种变体与分层分析，揭示视频信息提取、中间层关键作用及语言引导检索的重要性，为模型优化提供依据。**

- **链接: [http://arxiv.org/pdf/2508.15360v1](http://arxiv.org/pdf/2508.15360v1)**

> **作者:** Chenhui Gou; Ziyu Ma; Zicheng Duan; Haoyu He; Feng Chen; Akide Liu; Bohan Zhuang; Jianfei Cai; Hamid Rezatofighi
>
> **摘要:** Taking advantage of large-scale data and pretrained language models, Video Large Language Models (Video-LLMs) have shown strong capabilities in answering video questions. However, most existing efforts focus on improving performance, with limited attention to understanding their internal mechanisms. This paper aims to bridge this gap through a systematic empirical study. To interpret existing VideoLLMs, we adopt attention knockouts as our primary analytical tool and design three variants: Video Temporal Knockout, Video Spatial Knockout, and Language-to-Video Knockout. Then, we apply these three knockouts on different numbers of layers (window of layers). By carefully controlling the window of layers and types of knockouts, we provide two settings: a global setting and a fine-grained setting. Our study reveals three key findings: (1) Global setting indicates Video information extraction primarily occurs in early layers, forming a clear two-stage process -- lower layers focus on perceptual encoding, while higher layers handle abstract reasoning; (2) In the fine-grained setting, certain intermediate layers exert an outsized impact on video question answering, acting as critical outliers, whereas most other layers contribute minimally; (3) In both settings, we observe that spatial-temporal modeling relies more on language-guided retrieval than on intra- and inter-frame self-attention among video tokens, despite the latter's high computational cost. Finally, we demonstrate that these insights can be leveraged to reduce attention computation in Video-LLMs. To our knowledge, this is the first work to systematically uncover how Video-LLMs internally process and understand video content, offering interpretability and efficiency perspectives for future research.
>
---
#### [new 017] Fine-grained Multi-class Nuclei Segmentation with Molecular-empowered All-in-SAM Model
- **分类: cs.CV**

- **简介: 论文针对细粒度多类核分割任务，解决传统模型在识别特定核亚型时的不足，提出分子增强的All-in-SAM模型，通过标注参与、SAM适配与MOCL优化提升分割精度并降低标注需求。**

- **链接: [http://arxiv.org/pdf/2508.15751v1](http://arxiv.org/pdf/2508.15751v1)**

> **作者:** Xueyuan Li; Can Cui; Ruining Deng; Yucheng Tang; Quan Liu; Tianyuan Yao; Shunxing Bao; Naweed Chowdhury; Haichun Yang; Yuankai Huo
>
> **备注:** 25 pages, 3 figures, accepted by Journal of Medical Imaging
>
> **摘要:** Purpose: Recent developments in computational pathology have been driven by advances in Vision Foundation Models, particularly the Segment Anything Model (SAM). This model facilitates nuclei segmentation through two primary methods: prompt-based zero-shot segmentation and the use of cell-specific SAM models for direct segmentation. These approaches enable effective segmentation across a range of nuclei and cells. However, general vision foundation models often face challenges with fine-grained semantic segmentation, such as identifying specific nuclei subtypes or particular cells. Approach: In this paper, we propose the molecular-empowered All-in-SAM Model to advance computational pathology by leveraging the capabilities of vision foundation models. This model incorporates a full-stack approach, focusing on: (1) annotation-engaging lay annotators through molecular-empowered learning to reduce the need for detailed pixel-level annotations, (2) learning-adapting the SAM model to emphasize specific semantics, which utilizes its strong generalizability with SAM adapter, and (3) refinement-enhancing segmentation accuracy by integrating Molecular-Oriented Corrective Learning (MOCL). Results: Experimental results from both in-house and public datasets show that the All-in-SAM model significantly improves cell classification performance, even when faced with varying annotation quality. Conclusions: Our approach not only reduces the workload for annotators but also extends the accessibility of precise biomedical image analysis to resource-limited settings, thereby advancing medical diagnostics and automating pathology image analysis.
>
---
#### [new 018] LLM-empowered Dynamic Prompt Routing for Vision-Language Models Tuning under Long-Tailed Distributions
- **分类: cs.CV; I.4.10**

- **简介: 该论文针对视觉语言模型（VLM）在长尾分布数据下的微调问题，提出多维动态提示路由（MDPR）框架，通过构建五维知识库与动态路由机制，平衡语义并提升预测稳定性，有效缓解类别不平衡带来的偏差。**

- **链接: [http://arxiv.org/pdf/2508.15688v1](http://arxiv.org/pdf/2508.15688v1)**

> **作者:** Yongju Jia; Jiarui Ma; Xiangxian Li; Baiqiao Zhang; Xianhui Cao; Juan Liu; Yulong Bian
>
> **备注:** accepted by EMNLP 2025
>
> **摘要:** Pre-trained vision-language models (VLMs), such as CLIP, have demonstrated impressive capability in visual tasks, but their fine-tuning often suffers from bias in class-imbalanced scene. Recent works have introduced large language models (LLMs) to enhance VLM fine-tuning with supplementing semantic information. However, they often overlook inherent class imbalance in VLMs' pre-training, which may lead to bias accumulation in downstream tasks. To address this problem, this paper proposes a Multi-dimensional Dynamic Prompt Routing (MDPR) framework. MDPR constructs a comprehensive knowledge base for classes, spanning five visual-semantic dimensions. During fine-tuning, the dynamic routing mechanism aligns global visual classes, retrieves optimal prompts, and balances fine-grained semantics, yielding stable predictions through logits fusion. Extensive experiments on long-tailed benchmarks, including CIFAR-LT, ImageNet-LT, and Places-LT, demonstrate that MDPR achieves comparable results with current SOTA methods. Ablation studies further confirm the effectiveness of our semantic library for tail classes, and show that our dynamic routing incurs minimal computational overhead, making MDPR a flexible and efficient enhancement for VLM fine-tuning under data imbalance.
>
---
#### [new 019] Pretrained Diffusion Models Are Inherently Skipped-Step Samplers
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对扩散模型生成任务中的高效采样问题，提出跳步采样机制，证明其源自原始训练目标，无需非马尔可夫过程即可实现加速生成，结合DDIM提升效果。**

- **链接: [http://arxiv.org/pdf/2508.15233v1](http://arxiv.org/pdf/2508.15233v1)**

> **作者:** Wenju Xu
>
> **摘要:** Diffusion models have been achieving state-of-the-art results across various generation tasks. However, a notable drawback is their sequential generation process, requiring long-sequence step-by-step generation. Existing methods, such as DDIM, attempt to reduce sampling steps by constructing a class of non-Markovian diffusion processes that maintain the same training objective. However, there remains a gap in understanding whether the original diffusion process can achieve the same efficiency without resorting to non-Markovian processes. In this paper, we provide a confirmative answer and introduce skipped-step sampling, a mechanism that bypasses multiple intermediate denoising steps in the iterative generation process, in contrast with the traditional step-by-step refinement of standard diffusion inference. Crucially, we demonstrate that this skipped-step sampling mechanism is derived from the same training objective as the standard diffusion model, indicating that accelerated sampling via skipped-step sampling via a Markovian way is an intrinsic property of pretrained diffusion models. Additionally, we propose an enhanced generation method by integrating our accelerated sampling technique with DDIM. Extensive experiments on popular pretrained diffusion models, including the OpenAI ADM, Stable Diffusion, and Open Sora models, show that our method achieves high-quality generation with significantly reduced sampling steps.
>
---
#### [new 020] Reliable Multi-view 3D Reconstruction for `Just-in-time' Edge Environments
- **分类: cs.CV; cs.DC**

- **简介: 论文针对动态边缘环境中的多视角3D重建可靠性问题，提出基于投资组合理论的资源管理策略，利用遗传算法优化，确保在系统中断时仍能保持重建质量。**

- **链接: [http://arxiv.org/pdf/2508.15158v1](http://arxiv.org/pdf/2508.15158v1)**

> **作者:** Md. Nurul Absur; Abhinav Kumar; Swastik Brahma; Saptarshi Debroy
>
> **备注:** 11 Pages, 7 Figures
>
> **摘要:** Multi-view 3D reconstruction applications are revolutionizing critical use cases that require rapid situational-awareness, such as emergency response, tactical scenarios, and public safety. In many cases, their near-real-time latency requirements and ad-hoc needs for compute resources necessitate adoption of `Just-in-time' edge environments where the system is set up on the fly to support the applications during the mission lifetime. However, reliability issues can arise from the inherent dynamism and operational adversities of such edge environments, resulting in spatiotemporally correlated disruptions that impact the camera operations, which can lead to sustained degradation of reconstruction quality. In this paper, we propose a novel portfolio theory inspired edge resource management strategy for reliable multi-view 3D reconstruction against possible system disruptions. Our proposed methodology can guarantee reconstruction quality satisfaction even when the cameras are prone to spatiotemporally correlated disruptions. The portfolio theoretic optimization problem is solved using a genetic algorithm that converges quickly for realistic system settings. Using publicly available and customized 3D datasets, we demonstrate the proposed camera selection strategy's benefits in guaranteeing reliable 3D reconstruction against traditional baseline strategies, under spatiotemporal disruptions.
>
---
#### [new 021] Predicting Road Crossing Behaviour using Pose Detection and Sequence Modelling
- **分类: cs.CV; cs.AI**

- **简介: 该论文通过姿态检测与序列建模预测行人过马路意图，解决自动驾驶远距离预判问题，对比GRU/LSTM/1D CNN后构建端到端框架。**

- **链接: [http://arxiv.org/pdf/2508.15336v1](http://arxiv.org/pdf/2508.15336v1)**

> **作者:** Subhasis Dasgupta; Preetam Saha; Agniva Roy; Jaydip Sen
>
> **备注:** This is a pre-print version of the original paper accepted in the IEEE conference INDISCON 2025. It contains 8 figures and 1 table. The length of the paper is 7 pages
>
> **摘要:** The world is constantly moving towards AI based systems and autonomous vehicles are now reality in different parts of the world. These vehicles require sensors and cameras to detect objects and maneuver according to that. It becomes important to for such vehicles to also predict from a distant if a person is about to cross a road or not. The current study focused on predicting the intent of crossing the road by pedestrians in an experimental setup. The study involved working with deep learning models to predict poses and sequence modelling for temporal predictions. The study analysed three different sequence modelling to understand the prediction behaviour and it was found out that GRU was better in predicting the intent compared to LSTM model but 1D CNN was the best model in terms of speed. The study involved video analysis, and the output of pose detection model was integrated later on to sequence modelling techniques for an end-to-end deep learning framework for predicting road crossing intents.
>
---
#### [new 022] Comp-X: On Defining an Interactive Learned Image Compression Paradigm With Expert-driven LLM Agent
- **分类: cs.CV**

- **简介: 本文提出Comp-X，解决传统图像压缩需手动选择模式的问题，通过多功能框架、交互式LLM代理及专用基准，实现智能交互压缩，提升用户体验与压缩性能。**

- **链接: [http://arxiv.org/pdf/2508.15243v1](http://arxiv.org/pdf/2508.15243v1)**

> **作者:** Yixin Gao; Xin Li; Xiaohan Pan; Runsen Feng; Bingchen Li; Yunpeng Qi; Yiting Lu; Zhengxue Cheng; Zhibo Chen; Jörn Ostermann
>
> **摘要:** We present Comp-X, the first intelligently interactive image compression paradigm empowered by the impressive reasoning capability of large language model (LLM) agent. Notably, commonly used image codecs usually suffer from limited coding modes and rely on manual mode selection by engineers, making them unfriendly for unprofessional users. To overcome this, we advance the evolution of image coding paradigm by introducing three key innovations: (i) multi-functional coding framework, which unifies different coding modes of various objective/requirements, including human-machine perception, variable coding, and spatial bit allocation, into one framework. (ii) interactive coding agent, where we propose an augmented in-context learning method with coding expert feedback to teach the LLM agent how to understand the coding request, mode selection, and the use of the coding tools. (iii) IIC-bench, the first dedicated benchmark comprising diverse user requests and the corresponding annotations from coding experts, which is systematically designed for intelligently interactive image compression evaluation. Extensive experimental results demonstrate that our proposed Comp-X can understand the coding requests efficiently and achieve impressive textual interaction capability. Meanwhile, it can maintain comparable compression performance even with a single coding framework, providing a promising avenue for artificial general intelligence (AGI) in image compression.
>
---
#### [new 023] GasTwinFormer: A Hybrid Vision Transformer for Livestock Methane Emission Segmentation and Dietary Classification in Optical Gas Imaging
- **分类: cs.CV**

- **简介: 该论文提出GasTwinFormer，用于实时牛甲烷排放分割与饮食分类。通过混合视觉Transformer结合空间全局与局部注意力机制，利用OGI数据集实现高精度（74.47% mIoU）与高效推理（114.9 FPS），并达100%饮食分类准确率。**

- **链接: [http://arxiv.org/pdf/2508.15057v1](http://arxiv.org/pdf/2508.15057v1)**

> **作者:** Toqi Tahamid Sarker; Mohamed Embaby; Taminul Islam; Amer AbuGhazaleh; Khaled R Ahmed
>
> **备注:** Accepted for publication at ICCVW 2025
>
> **摘要:** Livestock methane emissions represent 32% of human-caused methane production, making automated monitoring critical for climate mitigation strategies. We introduce GasTwinFormer, a hybrid vision transformer for real-time methane emission segmentation and dietary classification in optical gas imaging through a novel Mix Twin encoder alternating between spatially-reduced global attention and locally-grouped attention mechanisms. Our architecture incorporates a lightweight LR-ASPP decoder for multi-scale feature aggregation and enables simultaneous methane segmentation and dietary classification in a unified framework. We contribute the first comprehensive beef cattle methane emission dataset using OGI, containing 11,694 annotated frames across three dietary treatments. GasTwinFormer achieves 74.47% mIoU and 83.63% mF1 for segmentation while maintaining exceptional efficiency with only 3.348M parameters, 3.428G FLOPs, and 114.9 FPS inference speed. Additionally, our method achieves perfect dietary classification accuracy (100%), demonstrating the effectiveness of leveraging diet-emission correlations. Extensive ablation studies validate each architectural component, establishing GasTwinFormer as a practical solution for real-time livestock emission monitoring. Please see our project page at gastwinformer.github.io.
>
---
#### [new 024] Normal and Abnormal Pathology Knowledge-Augmented Vision-Language Model for Anomaly Detection in Pathology Images
- **分类: cs.CV**

- **简介: 该论文针对病理图像异常检测任务，解决数据稀缺、结构复杂及可解释性差等问题，提出融合正常与异常病理知识的轻量Vision-Language模型Ano-NAViLa，提升检测精度与可解释性。**

- **链接: [http://arxiv.org/pdf/2508.15256v1](http://arxiv.org/pdf/2508.15256v1)**

> **作者:** Jinsol Song; Jiamu Wang; Anh Tien Nguyen; Keunho Byeon; Sangjeong Ahn; Sung Hak Lee; Jin Tae Kwak
>
> **备注:** Accepted at ICCV 2025. \c{opyright} IEEE 2025. This is the author's accepted version (camera-ready) of the paper. The definitive version is published in the Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2025). DOI will be updated when available
>
> **摘要:** Anomaly detection in computational pathology aims to identify rare and scarce anomalies where disease-related data are often limited or missing. Existing anomaly detection methods, primarily designed for industrial settings, face limitations in pathology due to computational constraints, diverse tissue structures, and lack of interpretability. To address these challenges, we propose Ano-NAViLa, a Normal and Abnormal pathology knowledge-augmented Vision-Language model for Anomaly detection in pathology images. Ano-NAViLa is built on a pre-trained vision-language model with a lightweight trainable MLP. By incorporating both normal and abnormal pathology knowledge, Ano-NAViLa enhances accuracy and robustness to variability in pathology images and provides interpretability through image-text associations. Evaluated on two lymph node datasets from different organs, Ano-NAViLa achieves the state-of-the-art performance in anomaly detection and localization, outperforming competing models.
>
---
#### [new 025] You Only Pose Once: A Minimalist's Detection Transformer for Monocular RGB Category-level 9D Multi-Object Pose Estimation
- **分类: cs.CV; cs.RO**

- **简介: 论文提出YOPO框架，解决单目RGB图像中多物体类别级9D姿态估计问题，通过统一检测与姿态估计，无需伪深度或CAD模型，实现端到端训练，刷新基准性能。**

- **链接: [http://arxiv.org/pdf/2508.14965v1](http://arxiv.org/pdf/2508.14965v1)**

> **作者:** Hakjin Lee; Junghoon Seo; Jaehoon Sim
>
> **备注:** https://mikigom.github.io/YOPO-project-page
>
> **摘要:** Accurately recovering the full 9-DoF pose of unseen instances within specific categories from a single RGB image remains a core challenge for robotics and automation. Most existing solutions still rely on pseudo-depth, CAD models, or multi-stage cascades that separate 2D detection from pose estimation. Motivated by the need for a simpler, RGB-only alternative that learns directly at the category level, we revisit a longstanding question: Can object detection and 9-DoF pose estimation be unified with high performance, without any additional data? We show that they can with our method, YOPO, a single-stage, query-based framework that treats category-level 9-DoF estimation as a natural extension of 2D detection. YOPO augments a transformer detector with a lightweight pose head, a bounding-box-conditioned translation module, and a 6D-aware Hungarian matching cost. The model is trained end-to-end only with RGB images and category-level pose labels. Despite its minimalist design, YOPO sets a new state of the art on three benchmarks. On the REAL275 dataset, it achieves 79.6% $\rm{IoU}_{50}$ and 54.1% under the $10^\circ$$10{\rm{cm}}$ metric, surpassing prior RGB-only methods and closing much of the gap to RGB-D systems. The code, models, and additional qualitative results can be found on our project.
>
---
#### [new 026] TPA: Temporal Prompt Alignment for Fetal Congenital Heart Defect Classification
- **分类: cs.CV**

- **简介: 论文针对胎儿先天性心脏病（CHD）超声视频分类任务，解决图像噪声、探头偏差及时间信息缺失等问题，提出TPA框架，融合时序建模与对比学习，引入CVAESM模块量化不确定性，实现高精度诊断与校准优化。**

- **链接: [http://arxiv.org/pdf/2508.15298v1](http://arxiv.org/pdf/2508.15298v1)**

> **作者:** Darya Taratynova; Alya Almsouti; Beknur Kalmakhanbet; Numan Saeed; Mohammad Yaqub
>
> **摘要:** Congenital heart defect (CHD) detection in ultrasound videos is hindered by image noise and probe positioning variability. While automated methods can reduce operator dependence, current machine learning approaches often neglect temporal information, limit themselves to binary classification, and do not account for prediction calibration. We propose Temporal Prompt Alignment (TPA), a method leveraging foundation image-text model and prompt-aware contrastive learning to classify fetal CHD on cardiac ultrasound videos. TPA extracts features from each frame of video subclips using an image encoder, aggregates them with a trainable temporal extractor to capture heart motion, and aligns the video representation with class-specific text prompts via a margin-hinge contrastive loss. To enhance calibration for clinical reliability, we introduce a Conditional Variational Autoencoder Style Modulation (CVAESM) module, which learns a latent style vector to modulate embeddings and quantifies classification uncertainty. Evaluated on a private dataset for CHD detection and on a large public dataset, EchoNet-Dynamic, for systolic dysfunction, TPA achieves state-of-the-art macro F1 scores of 85.40% for CHD diagnosis, while also reducing expected calibration error by 5.38% and adaptive ECE by 6.8%. On EchoNet-Dynamic's three-class task, it boosts macro F1 by 4.73% (from 53.89% to 58.62%). Temporal Prompt Alignment (TPA) is a framework for fetal congenital heart defect (CHD) classification in ultrasound videos that integrates temporal modeling, prompt-aware contrastive learning, and uncertainty quantification.
>
---
#### [new 027] LGMSNet: Thinning a medical image segmentation model via dual-level multiscale fusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医学图像分割中的轻量化模型性能与全局感知不足问题，提出LGMSNet框架，通过双尺度融合、异构卷积核和稀疏Transformer-卷积分支，有效缓解通道冗余并提升泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.15476v1](http://arxiv.org/pdf/2508.15476v1)**

> **作者:** Chengqi Dong; Fenghe Tang; Rongge Mao; Xinpei Gao; S. Kevin Zhou
>
> **备注:** Accepted by ECAI 2025
>
> **摘要:** Medical image segmentation plays a pivotal role in disease diagnosis and treatment planning, particularly in resource-constrained clinical settings where lightweight and generalizable models are urgently needed. However, existing lightweight models often compromise performance for efficiency and rarely adopt computationally expensive attention mechanisms, severely restricting their global contextual perception capabilities. Additionally, current architectures neglect the channel redundancy issue under the same convolutional kernels in medical imaging, which hinders effective feature extraction. To address these challenges, we propose LGMSNet, a novel lightweight framework based on local and global dual multiscale that achieves state-of-the-art performance with minimal computational overhead. LGMSNet employs heterogeneous intra-layer kernels to extract local high-frequency information while mitigating channel redundancy. In addition, the model integrates sparse transformer-convolutional hybrid branches to capture low-frequency global information. Extensive experiments across six public datasets demonstrate LGMSNet's superiority over existing state-of-the-art methods. In particular, LGMSNet maintains exceptional performance in zero-shot generalization tests on four unseen datasets, underscoring its potential for real-world deployment in resource-limited medical scenarios. The whole project code is in https://github.com/cq-dong/LGMSNet.
>
---
#### [new 028] Paired-Sampling Contrastive Framework for Joint Physical-Digital Face Attack Detection
- **分类: cs.CV**

- **简介: 该论文提出统一框架，通过配对采样学习模态无关活体线索，解决传统分离模型导致的系统复杂度高、易受组合攻击问题，实现高效物理-数字人脸攻击联合检测。**

- **链接: [http://arxiv.org/pdf/2508.14980v1](http://arxiv.org/pdf/2508.14980v1)**

> **作者:** Andrei Balykin; Anvar Ganiev; Denis Kondranin; Kirill Polevoda; Nikolai Liudkevich; Artem Petrov
>
> **备注:** Accepted to ICCV2025 FAS workshop
>
> **摘要:** Modern face recognition systems remain vulnerable to spoofing attempts, including both physical presentation attacks and digital forgeries. Traditionally, these two attack vectors have been handled by separate models, each targeting its own artifacts and modalities. However, maintaining distinct detectors increases system complexity and inference latency and leaves systems exposed to combined attack vectors. We propose the Paired-Sampling Contrastive Framework, a unified training approach that leverages automatically matched pairs of genuine and attack selfies to learn modality-agnostic liveness cues. Evaluated on the 6th Face Anti-Spoofing Challenge Unified Physical-Digital Attack Detection benchmark, our method achieves an average classification error rate (ACER) of 2.10 percent, outperforming prior solutions. The framework is lightweight (4.46 GFLOPs) and trains in under one hour, making it practical for real-world deployment. Code and pretrained models are available at https://github.com/xPONYx/iccv2025_deepfake_challenge.
>
---
#### [new 029] CineScale: Free Lunch in High-Resolution Cinematic Visual Generation
- **分类: cs.CV**

- **简介: 论文提出CineScale框架，解决高分辨率视觉生成中因超训练分辨率导致的重复模式问题，通过无微调或少量LoRA调整，实现8k图像与4k视频生成，拓展了图像到视频和视频到视频的合成能力。**

- **链接: [http://arxiv.org/pdf/2508.15774v1](http://arxiv.org/pdf/2508.15774v1)**

> **作者:** Haonan Qiu; Ning Yu; Ziqi Huang; Paul Debevec; Ziwei Liu
>
> **备注:** CineScale is an extended work of FreeScale (ICCV 2025). Project Page: https://eyeline-labs.github.io/CineScale/, Code Repo: https://github.com/Eyeline-Labs/CineScale
>
> **摘要:** Visual diffusion models achieve remarkable progress, yet they are typically trained at limited resolutions due to the lack of high-resolution data and constrained computation resources, hampering their ability to generate high-fidelity images or videos at higher resolutions. Recent efforts have explored tuning-free strategies to exhibit the untapped potential higher-resolution visual generation of pre-trained models. However, these methods are still prone to producing low-quality visual content with repetitive patterns. The key obstacle lies in the inevitable increase in high-frequency information when the model generates visual content exceeding its training resolution, leading to undesirable repetitive patterns deriving from the accumulated errors. In this work, we propose CineScale, a novel inference paradigm to enable higher-resolution visual generation. To tackle the various issues introduced by the two types of video generation architectures, we propose dedicated variants tailored to each. Unlike existing baseline methods that are confined to high-resolution T2I and T2V generation, CineScale broadens the scope by enabling high-resolution I2V and V2V synthesis, built atop state-of-the-art open-source video generation frameworks. Extensive experiments validate the superiority of our paradigm in extending the capabilities of higher-resolution visual generation for both image and video models. Remarkably, our approach enables 8k image generation without any fine-tuning, and achieves 4k video generation with only minimal LoRA fine-tuning. Generated video samples are available at our website: https://eyeline-labs.github.io/CineScale/.
>
---
#### [new 030] Aligning Moments in Time using Video Queries
- **分类: cs.CV**

- **简介: 论文提出视频到视频时刻检索（Vid2VidMR）任务，解决跨视频语义帧对齐与复杂依赖建模难题，设计基于Transformer的MATR模型，通过双阶段序列对齐与自监督预训练实现精准时刻定位，实验表明在ActivityNet-VRL和SportsMoments数据集上性能显著提升。**

- **链接: [http://arxiv.org/pdf/2508.15439v1](http://arxiv.org/pdf/2508.15439v1)**

> **作者:** Yogesh Kumar; Uday Agarwal; Manish Gupta; Anand Mishra
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Video-to-video moment retrieval (Vid2VidMR) is the task of localizing unseen events or moments in a target video using a query video. This task poses several challenges, such as the need for semantic frame-level alignment and modeling complex dependencies between query and target videos. To tackle this challenging problem, we introduce MATR (Moment Alignment TRansformer), a transformer-based model designed to capture semantic context as well as the temporal details necessary for precise moment localization. MATR conditions target video representations on query video features using dual-stage sequence alignment that encodes the required correlations and dependencies. These representations are then used to guide foreground/background classification and boundary prediction heads, enabling the model to accurately identify moments in the target video that semantically match with the query video. Additionally, to provide a strong task-specific initialization for MATR, we propose a self-supervised pre-training technique that involves training the model to localize random clips within videos. Extensive experiments demonstrate that MATR achieves notable performance improvements of 13.1% in R@1 and 8.1% in mIoU on an absolute scale compared to state-of-the-art methods on the popular ActivityNet-VRL dataset. Additionally, on our newly proposed dataset, SportsMoments, MATR shows a 14.7% gain in R@1 and a 14.4% gain in mIoU on an absolute scale over strong baselines.
>
---
#### [new 031] MapKD: Unlocking Prior Knowledge with Cross-Modal Distillation for Efficient Online HD Map Construction
- **分类: cs.CV**

- **简介: 论文针对在线HD地图构建任务，解决依赖过时离线地图和高计算成本的问题，提出MapKD框架，通过跨模态知识蒸馏（教师-教练-学生结构）与两种蒸馏策略，提升轻量化模型性能，实测提升mIoU和mAP。**

- **链接: [http://arxiv.org/pdf/2508.15653v1](http://arxiv.org/pdf/2508.15653v1)**

> **作者:** Ziyang Yan; Ruikai Li; Zhiyong Cui; Bohan Li; Han Jiang; Yilong Ren; Aoyong Li; Zhenning Li; Sijia Wen; Haiyang Yu
>
> **摘要:** Online HD map construction is a fundamental task in autonomous driving systems, aiming to acquire semantic information of map elements around the ego vehicle based on real-time sensor inputs. Recently, several approaches have achieved promising results by incorporating offline priors such as SD maps and HD maps or by fusing multi-modal data. However, these methods depend on stale offline maps and multi-modal sensor suites, resulting in avoidable computational overhead at inference. To address these limitations, we employ a knowledge distillation strategy to transfer knowledge from multimodal models with prior knowledge to an efficient, low-cost, and vision-centric student model. Specifically, we propose MapKD, a novel multi-level cross-modal knowledge distillation framework with an innovative Teacher-Coach-Student (TCS) paradigm. This framework consists of: (1) a camera-LiDAR fusion model with SD/HD map priors serving as the teacher; (2) a vision-centric coach model with prior knowledge and simulated LiDAR to bridge the cross-modal knowledge transfer gap; and (3) a lightweight vision-based student model. Additionally, we introduce two targeted knowledge distillation strategies: Token-Guided 2D Patch Distillation (TGPD) for bird's eye view feature alignment and Masked Semantic Response Distillation (MSRD) for semantic learning guidance. Extensive experiments on the challenging nuScenes dataset demonstrate that MapKD improves the student model by +6.68 mIoU and +10.94 mAP while simultaneously accelerating inference speed. The code is available at:https://github.com/2004yan/MapKD2026.
>
---
#### [new 032] MeSS: City Mesh-Guided Outdoor Scene Generation with Cross-View Consistent Diffusion
- **分类: cs.CV**

- **简介: 该论文旨在解决城市网格模型缺乏真实纹理及跨视图不一致的问题，提出MeSS方法通过改进扩散模型，结合三阶段处理（稀疏视图生成、密集视图传播、全局对齐）与3DGS重建，实现高几何精度的户外场景生成。**

- **链接: [http://arxiv.org/pdf/2508.15169v1](http://arxiv.org/pdf/2508.15169v1)**

> **作者:** Xuyang Chen; Zhijun Zhai; Kaixuan Zhou; Zengmao Wang; Jianan He; Dong Wang; Yanfeng Zhang; mingwei Sun; Rüdiger Westermann; Konrad Schindler; Liqiu Meng
>
> **摘要:** Mesh models have become increasingly accessible for numerous cities; however, the lack of realistic textures restricts their application in virtual urban navigation and autonomous driving. To address this, this paper proposes MeSS (Meshbased Scene Synthesis) for generating high-quality, styleconsistent outdoor scenes with city mesh models serving as the geometric prior. While image and video diffusion models can leverage spatial layouts (such as depth maps or HD maps) as control conditions to generate street-level perspective views, they are not directly applicable to 3D scene generation. Video diffusion models excel at synthesizing consistent view sequences that depict scenes but often struggle to adhere to predefined camera paths or align accurately with rendered control videos. In contrast, image diffusion models, though unable to guarantee cross-view visual consistency, can produce more geometry-aligned results when combined with ControlNet. Building on this insight, our approach enhances image diffusion models by improving cross-view consistency. The pipeline comprises three key stages: first, we generate geometrically consistent sparse views using Cascaded Outpainting ControlNets; second, we propagate denser intermediate views via a component dubbed AGInpaint; and third, we globally eliminate visual inconsistencies (e.g., varying exposure) using the GCAlign module. Concurrently with generation, a 3D Gaussian Splatting (3DGS) scene is reconstructed by initializing Gaussian balls on the mesh surface. Our method outperforms existing approaches in both geometric alignment and generation quality. Once synthesized, the scene can be rendered in diverse styles through relighting and style transfer techniques.
>
---
#### [new 033] Bidirectional Temporal Information Propagation for Moving Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文针对移动红外小目标检测任务，解决传统滑动窗口方法忽略全局时间信息导致的性能不足问题。提出双向时间传播框架，通过局部-全局时空特征融合（LTMF/GTMF模块）与STF损失，提升检测精度与速度。**

- **链接: [http://arxiv.org/pdf/2508.15415v1](http://arxiv.org/pdf/2508.15415v1)**

> **作者:** Dengyan Luo; Yanping Xiang; Hu Wang; Luping Ji. Shuai Li; Mao Ye
>
> **摘要:** Moving infrared small target detection is broadly adopted in infrared search and track systems, and has attracted considerable research focus in recent years. The existing learning-based multi-frame methods mainly aggregate the information of adjacent frames in a sliding window fashion to assist the detection of the current frame. However, the sliding-window-based methods do not consider joint optimization of the entire video clip and ignore the global temporal information outside the sliding window, resulting in redundant computation and sub-optimal performance. In this paper, we propose a Bidirectional temporal information propagation method for moving InfraRed small target Detection, dubbed BIRD. The bidirectional propagation strategy simultaneously utilizes local temporal information of adjacent frames and global temporal information of past and future frames in a recursive fashion. Specifically, in the forward and backward propagation branches, we first design a Local Temporal Motion Fusion (LTMF) module to model local spatio-temporal dependency between a target frame and its two adjacent frames. Then, a Global Temporal Motion Fusion (GTMF) module is developed to further aggregate the global propagation feature with the local fusion feature. Finally, the bidirectional aggregated features are fused and input into the detection head for detection. In addition, the entire video clip is jointly optimized by the traditional detection loss and the additional Spatio-Temporal Fusion (STF) loss. Extensive experiments demonstrate that the proposed BIRD method not only achieves the state-of-the-art performance but also shows a fast inference speed.
>
---
#### [new 034] DriveSplat: Decoupled Driving Scene Reconstruction with Geometry-enhanced Partitioned Neural Gaussians
- **分类: cs.CV**

- **简介: 该论文针对驾驶场景3D重建中的动态-静态分离与几何精度问题，提出DriveSplat方法，通过分区神经高斯表示、可变形高斯建模动态对象及深度法线先验监督，实现高精度驾驶场景重建。**

- **链接: [http://arxiv.org/pdf/2508.15376v1](http://arxiv.org/pdf/2508.15376v1)**

> **作者:** Cong Wang; Xianda Guo; Wenbo Xu; Wei Tian; Ruiqi Song; Chenming Zhang; Lingxi Li; Long Chen
>
> **摘要:** In the realm of driving scenarios, the presence of rapidly moving vehicles, pedestrians in motion, and large-scale static backgrounds poses significant challenges for 3D scene reconstruction. Recent methods based on 3D Gaussian Splatting address the motion blur problem by decoupling dynamic and static components within the scene. However, these decoupling strategies overlook background optimization with adequate geometry relationships and rely solely on fitting each training view by adding Gaussians. Therefore, these models exhibit limited robustness in rendering novel views and lack an accurate geometric representation. To address the above issues, we introduce DriveSplat, a high-quality reconstruction method for driving scenarios based on neural Gaussian representations with dynamic-static decoupling. To better accommodate the predominantly linear motion patterns of driving viewpoints, a region-wise voxel initialization scheme is employed, which partitions the scene into near, middle, and far regions to enhance close-range detail representation. Deformable neural Gaussians are introduced to model non-rigid dynamic actors, whose parameters are temporally adjusted by a learnable deformation network. The entire framework is further supervised by depth and normal priors from pre-trained models, improving the accuracy of geometric structures. Our method has been rigorously evaluated on the Waymo and KITTI datasets, demonstrating state-of-the-art performance in novel-view synthesis for driving scenarios.
>
---
#### [new 035] BasketLiDAR: The First LiDAR-Camera Multimodal Dataset for Professional Basketball MOT
- **分类: cs.CV**

- **简介: 本论文针对篮球比赛多目标跟踪难题，构建首个LiDAR-相机多模态数据集，提出融合感知框架，实现高精度实时跟踪。**

- **链接: [http://arxiv.org/pdf/2508.15299v1](http://arxiv.org/pdf/2508.15299v1)**

> **作者:** Ryunosuke Hayashi; Kohei Torimi; Rokuto Nagata; Kazuma Ikeda; Ozora Sako; Taichi Nakamura; Masaki Tani; Yoshimitsu Aoki; Kentaro Yoshioka
>
> **备注:** Accepted to MMSports
>
> **摘要:** Real-time 3D trajectory player tracking in sports plays a crucial role in tactical analysis, performance evaluation, and enhancing spectator experience. Traditional systems rely on multi-camera setups, but are constrained by the inherently two-dimensional nature of video data and the need for complex 3D reconstruction processing, making real-time analysis challenging. Basketball, in particular, represents one of the most difficult scenarios in the MOT field, as ten players move rapidly and complexly within a confined court space, with frequent occlusions caused by intense physical contact. To address these challenges, this paper constructs BasketLiDAR, the first multimodal dataset in the sports MOT field that combines LiDAR point clouds with synchronized multi-view camera footage in a professional basketball environment, and proposes a novel MOT framework that simultaneously achieves improved tracking accuracy and reduced computational cost. The BasketLiDAR dataset contains a total of 4,445 frames and 3,105 player IDs, with fully synchronized IDs between three LiDAR sensors and three multi-view cameras. We recorded 5-on-5 and 3-on-3 game data from actual professional basketball players, providing complete 3D positional information and ID annotations for each player. Based on this dataset, we developed a novel MOT algorithm that leverages LiDAR's high-precision 3D spatial information. The proposed method consists of a real-time tracking pipeline using LiDAR alone and a multimodal tracking pipeline that fuses LiDAR and camera data. Experimental results demonstrate that our approach achieves real-time operation, which was difficult with conventional camera-only methods, while achieving superior tracking performance even under occlusion conditions. The dataset is available upon request at: https://sites.google.com/keio.jp/keio-csg/projects/basket-lidar
>
---
#### [new 036] SurgWound-Bench: A Benchmark for Surgical Wound Diagnosis
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SurgWound数据集及基准，解决手术伤口诊断数据不足问题，通过三阶段框架实现细粒度诊断与报告生成，推动个性化医疗。**

- **链接: [http://arxiv.org/pdf/2508.15189v1](http://arxiv.org/pdf/2508.15189v1)**

> **作者:** Jiahao Xu; Changchang Yin; Odysseas Chatzipanagiotou; Diamantis Tsilimigras; Kevin Clear; Bingsheng Yao; Dakuo Wang; Timothy Pawlik; Ping Zhang
>
> **摘要:** Surgical site infection (SSI) is one of the most common and costly healthcare-associated infections and and surgical wound care remains a significant clinical challenge in preventing SSIs and improving patient outcomes. While recent studies have explored the use of deep learning for preliminary surgical wound screening, progress has been hindered by concerns over data privacy and the high costs associated with expert annotation. Currently, no publicly available dataset or benchmark encompasses various types of surgical wounds, resulting in the absence of an open-source Surgical-Wound screening tool. To address this gap: (1) we present SurgWound, the first open-source dataset featuring a diverse array of surgical wound types. It contains 697 surgical wound images annotated by 3 professional surgeons with eight fine-grained clinical attributes. (2) Based on SurgWound, we introduce the first benchmark for surgical wound diagnosis, which includes visual question answering (VQA) and report generation tasks to comprehensively evaluate model performance. (3) Furthermore, we propose a three-stage learning framework, WoundQwen, for surgical wound diagnosis. In the first stage, we employ five independent MLLMs to accurately predict specific surgical wound characteristics. In the second stage, these predictions serve as additional knowledge inputs to two MLLMs responsible for diagnosing outcomes, which assess infection risk and guide subsequent interventions. In the third stage, we train a MLLM that integrates the diagnostic results from the previous two stages to produce a comprehensive report. This three-stage framework can analyze detailed surgical wound characteristics and provide subsequent instructions to patients based on surgical images, paving the way for personalized wound care, timely intervention, and improved patient outcomes.
>
---
#### [new 037] ATLAS: Decoupling Skeletal and Shape Parameters for Expressive Parametric Human Modeling
- **分类: cs.CV**

- **简介: 该论文针对参数化人体建模中姿态与形状表达受限的问题，提出ATLAS模型，通过解耦骨骼与形状参数，基于高分辨率扫描数据构建高保真人体模型，提升对复杂姿态和个性化属性的表达能力。**

- **链接: [http://arxiv.org/pdf/2508.15767v1](http://arxiv.org/pdf/2508.15767v1)**

> **作者:** Jinhyung Park; Javier Romero; Shunsuke Saito; Fabian Prada; Takaaki Shiratori; Yichen Xu; Federica Bogo; Shoou-I Yu; Kris Kitani; Rawal Khirodkar
>
> **备注:** ICCV 2025; Website: https://jindapark.github.io/projects/atlas/
>
> **摘要:** Parametric body models offer expressive 3D representation of humans across a wide range of poses, shapes, and facial expressions, typically derived by learning a basis over registered 3D meshes. However, existing human mesh modeling approaches struggle to capture detailed variations across diverse body poses and shapes, largely due to limited training data diversity and restrictive modeling assumptions. Moreover, the common paradigm first optimizes the external body surface using a linear basis, then regresses internal skeletal joints from surface vertices. This approach introduces problematic dependencies between internal skeleton and outer soft tissue, limiting direct control over body height and bone lengths. To address these issues, we present ATLAS, a high-fidelity body model learned from 600k high-resolution scans captured using 240 synchronized cameras. Unlike previous methods, we explicitly decouple the shape and skeleton bases by grounding our mesh representation in the human skeleton. This decoupling enables enhanced shape expressivity, fine-grained customization of body attributes, and keypoint fitting independent of external soft-tissue characteristics. ATLAS outperforms existing methods by fitting unseen subjects in diverse poses more accurately, and quantitative evaluations show that our non-linear pose correctives more effectively capture complex poses compared to linear models.
>
---
#### [new 038] DesignCLIP: Multimodal Learning with CLIP for Design Patent Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文旨在通过CLIP模型开发DesignCLIP框架，解决设计专利图像信息不全导致的检索与分类歧义问题，利用多模态学习提升专利分析准确性。**

- **链接: [http://arxiv.org/pdf/2508.15297v1](http://arxiv.org/pdf/2508.15297v1)**

> **作者:** Zhu Wang; Homaira Huda Shomee; Sathya N. Ravi; Sourav Medya
>
> **备注:** Accepted by EMNLP 2025. 22 pages, 14 figures
>
> **摘要:** In the field of design patent analysis, traditional tasks such as patent classification and patent image retrieval heavily depend on the image data. However, patent images -- typically consisting of sketches with abstract and structural elements of an invention -- often fall short in conveying comprehensive visual context and semantic information. This inadequacy can lead to ambiguities in evaluation during prior art searches. Recent advancements in vision-language models, such as CLIP, offer promising opportunities for more reliable and accurate AI-driven patent analysis. In this work, we leverage CLIP models to develop a unified framework DesignCLIP for design patent applications with a large-scale dataset of U.S. design patents. To address the unique characteristics of patent data, DesignCLIP incorporates class-aware classification and contrastive learning, utilizing generated detailed captions for patent images and multi-views image learning. We validate the effectiveness of DesignCLIP across various downstream tasks, including patent classification and patent retrieval. Additionally, we explore multimodal patent retrieval, which provides the potential to enhance creativity and innovation in design by offering more diverse sources of inspiration. Our experiments show that DesignCLIP consistently outperforms baseline and SOTA models in the patent domain on all tasks. Our findings underscore the promise of multimodal approaches in advancing patent analysis. The codebase is available here: https://anonymous.4open.science/r/PATENTCLIP-4661/README.md.
>
---
#### [new 039] Multi-Object Sketch Animation with Grouping and Motion Trajectory Priors
- **分类: cs.CV**

- **简介: 论文任务为多物体草图动画生成，解决现有方法处理多物体交互和复杂运动时出现的时间不一致与泛化差问题。提出GroupSketch方法，采用两阶段流程（Motion Initialization和Motion Refinement），通过Group-based Displacement Network（GDN）与Context-conditioned Feature Enhancement模块提升动画质量与一致性。**

- **链接: [http://arxiv.org/pdf/2508.15535v1](http://arxiv.org/pdf/2508.15535v1)**

> **作者:** Guotao Liang; Juncheng Hu; Ximing Xing; Jing Zhang; Qian Yu
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** We introduce GroupSketch, a novel method for vector sketch animation that effectively handles multi-object interactions and complex motions. Existing approaches struggle with these scenarios, either being limited to single-object cases or suffering from temporal inconsistency and poor generalization. To address these limitations, our method adopts a two-stage pipeline comprising Motion Initialization and Motion Refinement. In the first stage, the input sketch is interactively divided into semantic groups and key frames are defined, enabling the generation of a coarse animation via interpolation. In the second stage, we propose a Group-based Displacement Network (GDN), which refines the coarse animation by predicting group-specific displacement fields, leveraging priors from a text-to-video model. GDN further incorporates specialized modules, such as Context-conditioned Feature Enhancement (CCFE), to improve temporal consistency. Extensive experiments demonstrate that our approach significantly outperforms existing methods in generating high-quality, temporally consistent animations for complex, multi-object sketches, thus expanding the practical applications of sketch animation.
>
---
#### [new 040] DIO: Refining Mutual Information and Causal Chain to Enhance Machine Abstract Reasoning Ability
- **分类: cs.CV**

- **简介: 本文针对机器抽象推理能力不足问题，基于RPM任务设计DIO模型，通过优化互信息与因果链建模提升推理能力，改进传统目标函数以更准确捕捉人类推理逻辑。（99字）**

- **链接: [http://arxiv.org/pdf/2508.15387v1](http://arxiv.org/pdf/2508.15387v1)**

> **作者:** Ruizhuo Song; Beiming Yuan
>
> **备注:** 15 pages, 9 figures, 8 tables
>
> **摘要:** Despite the outstanding performance of current deep learning models across various domains, their fundamental bottleneck in abstract reasoning remains unresolved. To address this challenge, the academic community has introduced Raven's Progressive Matrices (RPM) problems as an authoritative benchmark for evaluating the abstract reasoning capabilities of deep learning algorithms, with a focus on core intelligence dimensions such as abstract reasoning, pattern recognition, and complex problem-solving. Therefore, this paper centers on solving RPM problems, aiming to contribute to enhancing the abstract reasoning abilities of machine intelligence. Firstly, this paper adopts a ``causal chain modeling'' perspective to systematically analyze the complete causal chain in RPM tasks: image $\rightarrow$ abstract attributes $\rightarrow$ progressive attribute patterns $\rightarrow$ pattern consistency $\rightarrow$ correct answer. Based on this analysis, the network architecture of the baseline model DIO is designed. However, experiments reveal that the optimization objective formulated for DIO, namely maximizing the variational lower bound of mutual information between the context and the correct option, fails to enable the model to genuinely acquire the predefined human reasoning logic. This is attributed to two main reasons: the tightness of the lower bound significantly impacts the effectiveness of mutual information maximization, and mutual information, as a statistical measure, does not capture the causal relationship between subjects and objects. To overcome these limitations, this paper progressively proposes three improvement methods:
>
---
#### [new 041] SceneGen: Single-Image 3D Scene Generation in One Feedforward Pass
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SceneGen，通过单图像生成多3D资产，无需优化或检索，结合特征聚合与位置头实现单次前向传播生成，并支持多图输入，验证了高效鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.15769v1](http://arxiv.org/pdf/2508.15769v1)**

> **作者:** Yanxu Meng; Haoning Wu; Ya Zhang; Weidi Xie
>
> **备注:** Technical Report; Project Page: https://mengmouxu.github.io/SceneGen
>
> **摘要:** 3D content generation has recently attracted significant research interest due to its applications in VR/AR and embodied AI. In this work, we address the challenging task of synthesizing multiple 3D assets within a single scene image. Concretely, our contributions are fourfold: (i) we present SceneGen, a novel framework that takes a scene image and corresponding object masks as input, simultaneously producing multiple 3D assets with geometry and texture. Notably, SceneGen operates with no need for optimization or asset retrieval; (ii) we introduce a novel feature aggregation module that integrates local and global scene information from visual and geometric encoders within the feature extraction module. Coupled with a position head, this enables the generation of 3D assets and their relative spatial positions in a single feedforward pass; (iii) we demonstrate SceneGen's direct extensibility to multi-image input scenarios. Despite being trained solely on single-image inputs, our architectural design enables improved generation performance with multi-image inputs; and (iv) extensive quantitative and qualitative evaluations confirm the efficiency and robust generation abilities of our approach. We believe this paradigm offers a novel solution for high-quality 3D content generation, potentially advancing its practical applications in downstream tasks. The code and model will be publicly available at: https://mengmouxu.github.io/SceneGen.
>
---
#### [new 042] A Curated Dataset and Deep Learning Approach for Minor Dent Detection in Vehicles
- **分类: cs.CV**

- **简介: 论文提出基于YOLOv8的深度学习方法，解决车辆微小凹痕检测难题，构建定制数据集并优化模型，实现高精度实时检测。**

- **链接: [http://arxiv.org/pdf/2508.15431v1](http://arxiv.org/pdf/2508.15431v1)**

> **作者:** Danish Zia Baig; Mohsin Kamal
>
> **摘要:** Conventional car damage inspection techniques are labor-intensive, manual, and frequently overlook tiny surface imperfections like microscopic dents. Machine learning provides an innovative solution to the increasing demand for quicker and more precise inspection methods. The paper uses the YOLOv8 object recognition framework to provide a deep learning-based solution for automatically detecting microscopic surface flaws, notably tiny dents, on car exteriors. Traditional automotive damage inspection procedures are manual, time-consuming, and frequently unreliable at detecting tiny flaws. To solve this, a bespoke dataset containing annotated photos of car surfaces under various lighting circumstances, angles, and textures was created. To improve robustness, the YOLOv8m model and its customized variants, YOLOv8m-t4 and YOLOv8m-t42, were trained employing real-time data augmentation approaches. Experimental results show that the technique has excellent detection accuracy and low inference latency, making it suited for real-time applications such as automated insurance evaluations and automobile inspections. Evaluation parameters such as mean Average Precision (mAP), precision, recall, and F1-score verified the model's efficacy. With a precision of 0.86, recall of 0.84, and F1-score of 0.85, the YOLOv8m-t42 model outperformed the YOLOv8m-t4 model (precision: 0.81, recall: 0.79, F1-score: 0.80) in identifying microscopic surface defects. With a little reduced mAP@0.5:0.95 of 0.20, the mAP@0.5 for YOLOv8m-t42 stabilized at 0.60. Furthermore, YOLOv8m-t42's PR curve area was 0.88, suggesting more consistent performance than YOLOv8m-t4 (0.82). YOLOv8m-t42 has greater accuracy and is more appropriate for practical dent detection applications, even though its convergence is slower.
>
---
#### [new 043] Visual Autoregressive Modeling for Instruction-Guided Image Editing
- **分类: cs.CV; cs.MM**

- **简介: 该论文针对基于指令的图像编辑任务，解决扩散模型因全局去噪导致的编辑偏差问题，提出VAREdit框架，通过视觉自回归建模与多尺度预测结合，引入Scale-Aligned Reference模块提升编辑精度与效率。**

- **链接: [http://arxiv.org/pdf/2508.15772v1](http://arxiv.org/pdf/2508.15772v1)**

> **作者:** Qingyang Mao; Qi Cai; Yehao Li; Yingwei Pan; Mingyue Cheng; Ting Yao; Qi Liu; Tao Mei
>
> **备注:** Source codes and models are available at https://github.com/HiDream-ai/VAREdit
>
> **摘要:** Recent advances in diffusion models have brought remarkable visual fidelity to instruction-guided image editing. However, their global denoising process inherently entangles the edited region with the entire image context, leading to unintended spurious modifications and compromised adherence to editing instructions. In contrast, autoregressive models offer a distinct paradigm by formulating image synthesis as a sequential process over discrete visual tokens. Their causal and compositional mechanism naturally circumvents the adherence challenges of diffusion-based methods. In this paper, we present VAREdit, a visual autoregressive (VAR) framework that reframes image editing as a next-scale prediction problem. Conditioned on source image features and text instructions, VAREdit generates multi-scale target features to achieve precise edits. A core challenge in this paradigm is how to effectively condition the source image tokens. We observe that finest-scale source features cannot effectively guide the prediction of coarser target features. To bridge this gap, we introduce a Scale-Aligned Reference (SAR) module, which injects scale-matched conditioning information into the first self-attention layer. VAREdit demonstrates significant advancements in both editing adherence and efficiency. On standard benchmarks, it outperforms leading diffusion-based methods by 30\%+ higher GPT-Balance score. Moreover, it completes a $512\times512$ editing in 1.2 seconds, making it 2.2$\times$ faster than the similarly sized UltraEdit. The models are available at https://github.com/HiDream-ai/VAREdit.
>
---
#### [new 044] Transfer learning optimization based on evolutionary selective fine tuning
- **分类: cs.CV**

- **简介: 该论文针对迁移学习中传统微调导致的过拟合与高成本问题，提出基于进化算法的选择性微调方法BioTune，通过优化层选择提升效率。**

- **链接: [http://arxiv.org/pdf/2508.15367v1](http://arxiv.org/pdf/2508.15367v1)**

> **作者:** Jacinto Colan; Ana Davila; Yasuhisa Hasegawa
>
> **备注:** Presented at the Workshop artiFicial And bio-inspIred netwoRked intelliGence foR cOnstrained aUtoNomous Devices (FAIRGROUND). 2025 International Joint Conference on Neural Networks (IJCNN)
>
> **摘要:** Deep learning has shown substantial progress in image analysis. However, the computational demands of large, fully trained models remain a consideration. Transfer learning offers a strategy for adapting pre-trained models to new tasks. Traditional fine-tuning often involves updating all model parameters, which can potentially lead to overfitting and higher computational costs. This paper introduces BioTune, an evolutionary adaptive fine-tuning technique that selectively fine-tunes layers to enhance transfer learning efficiency. BioTune employs an evolutionary algorithm to identify a focused set of layers for fine-tuning, aiming to optimize model performance on a given target task. Evaluation across nine image classification datasets from various domains indicates that BioTune achieves competitive or improved accuracy and efficiency compared to existing fine-tuning methods such as AutoRGN and LoRA. By concentrating the fine-tuning process on a subset of relevant layers, BioTune reduces the number of trainable parameters, potentially leading to decreased computational cost and facilitating more efficient transfer learning across diverse data characteristics and distributions.
>
---
#### [new 045] VideoEraser: Concept Erasure in Text-to-Video Diffusion Models
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 论文针对文本到视频扩散模型中的概念擦除任务，解决生成有害内容问题，提出无需训练的VideoEraser框架，通过调整提示嵌入与对抗性噪声引导，有效抑制不良内容生成。**

- **链接: [http://arxiv.org/pdf/2508.15314v1](http://arxiv.org/pdf/2508.15314v1)**

> **作者:** Naen Xu; Jinghuai Zhang; Changjiang Li; Zhi Chen; Chunyi Zhou; Qingming Li; Tianyu Du; Shouling Ji
>
> **备注:** To appear in the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)
>
> **摘要:** The rapid growth of text-to-video (T2V) diffusion models has raised concerns about privacy, copyright, and safety due to their potential misuse in generating harmful or misleading content. These models are often trained on numerous datasets, including unauthorized personal identities, artistic creations, and harmful materials, which can lead to uncontrolled production and distribution of such content. To address this, we propose VideoEraser, a training-free framework that prevents T2V diffusion models from generating videos with undesirable concepts, even when explicitly prompted with those concepts. Designed as a plug-and-play module, VideoEraser can seamlessly integrate with representative T2V diffusion models via a two-stage process: Selective Prompt Embedding Adjustment (SPEA) and Adversarial-Resilient Noise Guidance (ARNG). We conduct extensive evaluations across four tasks, including object erasure, artistic style erasure, celebrity erasure, and explicit content erasure. Experimental results show that VideoEraser consistently outperforms prior methods regarding efficacy, integrity, fidelity, robustness, and generalizability. Notably, VideoEraser achieves state-of-the-art performance in suppressing undesirable content during T2V generation, reducing it by 46% on average across four tasks compared to baselines.
>
---
#### [new 046] ExtraGS: Geometric-Aware Trajectory Extrapolation with Uncertainty-Guided Generative Priors
- **分类: cs.CV**

- **简介: 论文针对自动驾驶中轨迹外推的几何不一致与过度平滑问题，提出ExtraGS框架，融合几何与生成先验，采用RSG和FFG表示，并利用自监督不确定性估计，提升外推视图的几何一致性与真实感。**

- **链接: [http://arxiv.org/pdf/2508.15529v1](http://arxiv.org/pdf/2508.15529v1)**

> **作者:** Kaiyuan Tan; Yingying Shen; Haohui Zhu; Zhiwei Zhan; Shan Zhao; Mingfei Tu; Hongcheng Luo; Haiyang Sun; Bing Wang; Guang Chen; Hangjun Ye
>
> **摘要:** Synthesizing extrapolated views from recorded driving logs is critical for simulating driving scenes for autonomous driving vehicles, yet it remains a challenging task. Recent methods leverage generative priors as pseudo ground truth, but often lead to poor geometric consistency and over-smoothed renderings. To address these limitations, we propose ExtraGS, a holistic framework for trajectory extrapolation that integrates both geometric and generative priors. At the core of ExtraGS is a novel Road Surface Gaussian(RSG) representation based on a hybrid Gaussian-Signed Distance Function (SDF) design, and Far Field Gaussians (FFG) that use learnable scaling factors to efficiently handle distant objects. Furthermore, we develop a self-supervised uncertainty estimation framework based on spherical harmonics that enables selective integration of generative priors only where extrapolation artifacts occur. Extensive experiments on multiple datasets, diverse multi-camera setups, and various generative priors demonstrate that ExtraGS significantly enhances the realism and geometric consistency of extrapolated views, while preserving high fidelity along the original trajectory.
>
---
#### [new 047] Reversible Unfolding Network for Concealed Visual Perception with Generative Refinement
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对隐藏视觉感知任务，解决现有方法仅限mask域的局限，提出RUN++网络通过可逆建模与生成扩散模型，分阶段处理遮挡区域，融合三模块提升分割精度，有效减少误检。**

- **链接: [http://arxiv.org/pdf/2508.15027v1](http://arxiv.org/pdf/2508.15027v1)**

> **作者:** Chunming He; Fengyang Xiao; Rihan Zhang; Chengyu Fang; Deng-Ping Fan; Sina Farsiu
>
> **备注:** 18 pages, 21 tables, 13 figures
>
> **摘要:** Existing methods for concealed visual perception (CVP) often leverage reversible strategies to decrease uncertainty, yet these are typically confined to the mask domain, leaving the potential of the RGB domain underexplored. To address this, we propose a reversible unfolding network with generative refinement, termed RUN++. Specifically, RUN++ first formulates the CVP task as a mathematical optimization problem and unfolds the iterative solution into a multi-stage deep network. This approach provides a principled way to apply reversible modeling across both mask and RGB domains while leveraging a diffusion model to resolve the resulting uncertainty. Each stage of the network integrates three purpose-driven modules: a Concealed Object Region Extraction (CORE) module applies reversible modeling to the mask domain to identify core object regions; a Context-Aware Region Enhancement (CARE) module extends this principle to the RGB domain to foster better foreground-background separation; and a Finetuning Iteration via Noise-based Enhancement (FINE) module provides a final refinement. The FINE module introduces a targeted Bernoulli diffusion model that refines only the uncertain regions of the segmentation mask, harnessing the generative power of diffusion for fine-detail restoration without the prohibitive computational cost of a full-image process. This unique synergy, where the unfolding network provides a strong uncertainty prior for the diffusion model, allows RUN++ to efficiently direct its focus toward ambiguous areas, significantly mitigating false positives and negatives. Furthermore, we introduce a new paradigm for building robust CVP systems that remain effective under real-world degradations and extend this concept into a broader bi-level optimization framework.
>
---
#### [new 048] Weakly-Supervised Learning for Tree Instances Segmentation in Airborne Lidar Point Clouds
- **分类: cs.CV**

- **简介: 该论文针对LiDAR点云中树实例分割的挑战，提出弱监督学习方法，通过人类评分优化模型，提升分割精度并减少误判，解决数据变化和标注成本高的问题。**

- **链接: [http://arxiv.org/pdf/2508.15646v1](http://arxiv.org/pdf/2508.15646v1)**

> **作者:** Swann Emilien Céleste Destouches; Jesse Lahaye; Laurent Valentin Jospin; Jan Skaloud
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** Tree instance segmentation of airborne laser scanning (ALS) data is of utmost importance for forest monitoring, but remains challenging due to variations in the data caused by factors such as sensor resolution, vegetation state at acquisition time, terrain characteristics, etc. Moreover, obtaining a sufficient amount of precisely labeled data to train fully supervised instance segmentation methods is expensive. To address these challenges, we propose a weakly supervised approach where labels of an initial segmentation result obtained either by a non-finetuned model or a closed form algorithm are provided as a quality rating by a human operator. The labels produced during the quality assessment are then used to train a rating model, whose task is to classify a segmentation output into the same classes as specified by the human operator. Finally, the segmentation model is finetuned using feedback from the rating model. This in turn improves the original segmentation model by 34\% in terms of correctly identified tree instances while considerably reducing the number of non-tree instances predicted. Challenges still remain in data over sparsely forested regions characterized by small trees (less than two meters in height) or within complex surroundings containing shrubs, boulders, etc. which can be confused as trees where the performance of the proposed method is reduced.
>
---
#### [new 049] Fast Graph Neural Network for Image Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文针对图像分类任务，解决传统CNN在复杂场景下的效率与精度瓶颈，提出结合GCNs与Voronoi图的方法，将图像转换为图结构并优化，提升分类效率和精度。**

- **链接: [http://arxiv.org/pdf/2508.14958v1](http://arxiv.org/pdf/2508.14958v1)**

> **作者:** Mustafa Mohammadi Gharasuie; Luis Rueda
>
> **备注:** 12 pages, proceeding into CanadianAI 2025
>
> **摘要:** The rapid progress in image classification has been largely driven by the adoption of Graph Convolutional Networks (GCNs), which offer a robust framework for handling complex data structures. This study introduces a novel approach that integrates GCNs with Voronoi diagrams to enhance image classification by leveraging their ability to effectively model relational data. Unlike conventional convolutional neural networks (CNNs), our method represents images as graphs, where pixels or regions function as vertices. These graphs are then refined using corresponding Delaunay triangulations, optimizing their representation. The proposed model achieves significant improvements in both preprocessing efficiency and classification accuracy across various benchmark datasets, surpassing state-of-the-art approaches, particularly in challenging scenarios involving intricate scenes and fine-grained categories. Experimental results, validated through cross-validation, underscore the effectiveness of combining GCNs with Voronoi diagrams for advancing image classification. This research not only presents a novel perspective on image classification but also expands the potential applications of graph-based learning paradigms in computer vision and unstructured data analysis.
>
---
#### [new 050] When and What: Diffusion-Grounded VideoLLM with Entity Aware Segmentation for Long Video Understanding
- **分类: cs.CV**

- **简介: 论文提出Grounded VideoDiT，针对长视频理解中时间感知粗略和实体对齐不足的问题，通过扩散编码、对象绑定表示和混合标记方案，提升时间敏感性和实体关联能力。**

- **链接: [http://arxiv.org/pdf/2508.15641v1](http://arxiv.org/pdf/2508.15641v1)**

> **作者:** Pengcheng Fang; Yuxia Chen; Rui Guo
>
> **摘要:** Understanding videos requires more than answering open ended questions, it demands the ability to pinpoint when events occur and how entities interact across time. While recent Video LLMs have achieved remarkable progress in holistic reasoning, they remain coarse in temporal perception: timestamps are encoded only implicitly, frame level features are weak in capturing continuity, and language vision alignment often drifts from the entities of interest. In this paper, we present Grounded VideoDiT, a Video LLM designed to overcome these limitations by introducing three key innovations. First, a Diffusion Temporal Latent (DTL) encoder enhances boundary sensitivity and maintains temporal consistency. Second, object grounded representations explicitly bind query entities to localized visual evidence, strengthening alignment. Third, a mixed token scheme with discrete temporal tokens provides explicit timestamp modeling, enabling fine grained temporal reasoning. Together, these designs equip Grounded VideoDiT with robust grounding capabilities, as validated by state of the art results on Charades STA, NExT GQA, and multiple VideoQA benchmarks.
>
---
#### [new 051] Collaborative Multi-Modal Coding for High-Quality 3D Generation
- **分类: cs.CV**

- **简介: 该论文针对3D生成中的多模态数据利用问题，提出TriMM模型，通过协作编码和辅助监督整合RGB、RGBD、点云等多模态数据，结合三平面扩散模型生成高质量3D资产，提升纹理与几何细节。**

- **链接: [http://arxiv.org/pdf/2508.15228v1](http://arxiv.org/pdf/2508.15228v1)**

> **作者:** Ziang Cao; Zhaoxi Chen; Liang Pan; Ziwei Liu
>
> **摘要:** 3D content inherently encompasses multi-modal characteristics and can be projected into different modalities (e.g., RGB images, RGBD, and point clouds). Each modality exhibits distinct advantages in 3D asset modeling: RGB images contain vivid 3D textures, whereas point clouds define fine-grained 3D geometries. However, most existing 3D-native generative architectures either operate predominantly within single-modality paradigms-thus overlooking the complementary benefits of multi-modality data-or restrict themselves to 3D structures, thereby limiting the scope of available training datasets. To holistically harness multi-modalities for 3D modeling, we present TriMM, the first feed-forward 3D-native generative model that learns from basic multi-modalities (e.g., RGB, RGBD, and point cloud). Specifically, 1) TriMM first introduces collaborative multi-modal coding, which integrates modality-specific features while preserving their unique representational strengths. 2) Furthermore, auxiliary 2D and 3D supervision are introduced to raise the robustness and performance of multi-modal coding. 3) Based on the embedded multi-modal code, TriMM employs a triplane latent diffusion model to generate 3D assets of superior quality, enhancing both the texture and the geometric detail. Extensive experiments on multiple well-known datasets demonstrate that TriMM, by effectively leveraging multi-modality, achieves competitive performance with models trained on large-scale datasets, despite utilizing a small amount of training data. Furthermore, we conduct additional experiments on recent RGB-D datasets, verifying the feasibility of incorporating other multi-modal datasets into 3D generation.
>
---
#### [new 052] Heatmap Regression without Soft-Argmax for Facial Landmark Detection
- **分类: cs.CV; cs.AI**

- **简介: 论文针对面部关键点检测任务，解决热图回归中依赖Soft-Argmax的可微性问题，提出基于结构化预测的替代训练目标，实现更高效准确的端到端训练。**

- **链接: [http://arxiv.org/pdf/2508.14929v1](http://arxiv.org/pdf/2508.14929v1)**

> **作者:** Chiao-An Yang; Raymond A. Yeh
>
> **摘要:** Facial landmark detection is an important task in computer vision with numerous applications, such as head pose estimation, expression analysis, face swapping, etc. Heatmap regression-based methods have been widely used to achieve state-of-the-art results in this task. These methods involve computing the argmax over the heatmaps to predict a landmark. Since argmax is not differentiable, these methods use a differentiable approximation, Soft-argmax, to enable end-to-end training on deep-nets. In this work, we revisit this long-standing choice of using Soft-argmax and demonstrate that it is not the only way to achieve strong performance. Instead, we propose an alternative training objective based on the classic structured prediction framework. Empirically, our method achieves state-of-the-art performance on three facial landmark benchmarks (WFLW, COFW, and 300W), converging 2.2x faster during training while maintaining better/competitive accuracy. Our code is available here: https://github.com/ca-joe-yang/regression-without-softarg.
>
---
#### [new 053] AeroDuo: Aerial Duo for UAV-based Vision and Language Navigation
- **分类: cs.CV**

- **简介: 论文提出AeroDuo框架，解决无人机在复杂环境中基于视觉与语言的导航难题。通过双高度无人机协作（高海拔环境推理，低海拔精准导航）及HaL-13k数据集，提升导航可靠性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.15232v1](http://arxiv.org/pdf/2508.15232v1)**

> **作者:** Ruipu Wu; Yige Zhang; Jinyu Chen; Linjiang Huang; Shifeng Zhang; Xu Zhou; Liang Wang; Si Liu
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Aerial Vision-and-Language Navigation (VLN) is an emerging task that enables Unmanned Aerial Vehicles (UAVs) to navigate outdoor environments using natural language instructions and visual cues. However, due to the extended trajectories and complex maneuverability of UAVs, achieving reliable UAV-VLN performance is challenging and often requires human intervention or overly detailed instructions. To harness the advantages of UAVs' high mobility, which could provide multi-grained perspectives, while maintaining a manageable motion space for learning, we introduce a novel task called Dual-Altitude UAV Collaborative VLN (DuAl-VLN). In this task, two UAVs operate at distinct altitudes: a high-altitude UAV responsible for broad environmental reasoning, and a low-altitude UAV tasked with precise navigation. To support the training and evaluation of the DuAl-VLN, we construct the HaL-13k, a dataset comprising 13,838 collaborative high-low UAV demonstration trajectories, each paired with target-oriented language instructions. This dataset includes both unseen maps and an unseen object validation set to systematically evaluate the model's generalization capabilities across novel environments and unfamiliar targets. To consolidate their complementary strengths, we propose a dual-UAV collaborative VLN framework, AeroDuo, where the high-altitude UAV integrates a multimodal large language model (Pilot-LLM) for target reasoning, while the low-altitude UAV employs a lightweight multi-stage policy for navigation and target grounding. The two UAVs work collaboratively and only exchange minimal coordinate information to ensure efficiency.
>
---
#### [new 054] WorldWeaver: Generating Long-Horizon Video Worlds via Rich Perception
- **分类: cs.CV**

- **简介: 论文任务为长时序视频生成，解决结构与时间一致性问题。通过联合建模RGB与感知条件、利用深度线索构建记忆库及分段噪声调度，提升生成视频的稳定性与质量。**

- **链接: [http://arxiv.org/pdf/2508.15720v1](http://arxiv.org/pdf/2508.15720v1)**

> **作者:** Zhiheng Liu; Xueqing Deng; Shoufa Chen; Angtian Wang; Qiushan Guo; Mingfei Han; Zeyue Xue; Mengzhao Chen; Ping Luo; Linjie Yang
>
> **备注:** Project page: https://johanan528.github.io/worldweaver_web/
>
> **摘要:** Generative video modeling has made significant strides, yet ensuring structural and temporal consistency over long sequences remains a challenge. Current methods predominantly rely on RGB signals, leading to accumulated errors in object structure and motion over extended durations. To address these issues, we introduce WorldWeaver, a robust framework for long video generation that jointly models RGB frames and perceptual conditions within a unified long-horizon modeling scheme. Our training framework offers three key advantages. First, by jointly predicting perceptual conditions and color information from a unified representation, it significantly enhances temporal consistency and motion dynamics. Second, by leveraging depth cues, which we observe to be more resistant to drift than RGB, we construct a memory bank that preserves clearer contextual information, improving quality in long-horizon video generation. Third, we employ segmented noise scheduling for training prediction groups, which further mitigates drift and reduces computational cost. Extensive experiments on both diffusion- and rectified flow-based models demonstrate the effectiveness of WorldWeaver in reducing temporal drift and improving the fidelity of generated videos.
>
---
#### [new 055] Image-Conditioned 3D Gaussian Splat Quantization
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 论文针对3D场景压缩与更新，解决存储效率及长期适应性问题，提出ICGS-Quantizer，通过联合相关性、共享代码本和图像条件解码提升效率与适应性。**

- **链接: [http://arxiv.org/pdf/2508.15372v1](http://arxiv.org/pdf/2508.15372v1)**

> **作者:** Xinshuang Liu; Runfa Blark Li; Keito Suzuki; Truong Nguyen
>
> **摘要:** 3D Gaussian Splatting (3DGS) has attracted considerable attention for enabling high-quality real-time rendering. Although 3DGS compression methods have been proposed for deployment on storage-constrained devices, two limitations hinder archival use: (1) they compress medium-scale scenes only to the megabyte range, which remains impractical for large-scale scenes or extensive scene collections; and (2) they lack mechanisms to accommodate scene changes after long-term archival. To address these limitations, we propose an Image-Conditioned Gaussian Splat Quantizer (ICGS-Quantizer) that substantially enhances compression efficiency and provides adaptability to scene changes after archiving. ICGS-Quantizer improves quantization efficiency by jointly exploiting inter-Gaussian and inter-attribute correlations and by using shared codebooks across all training scenes, which are then fixed and applied to previously unseen test scenes, eliminating the overhead of per-scene codebooks. This approach effectively reduces the storage requirements for 3DGS to the kilobyte range while preserving visual fidelity. To enable adaptability to post-archival scene changes, ICGS-Quantizer conditions scene decoding on images captured at decoding time. The encoding, quantization, and decoding processes are trained jointly, ensuring that the codes, which are quantized representations of the scene, are effective for conditional decoding. We evaluate ICGS-Quantizer on 3D scene compression and 3D scene updating. Experimental results show that ICGS-Quantizer consistently outperforms state-of-the-art methods in compression efficiency and adaptability to scene changes. Our code, model, and data will be publicly available on GitHub.
>
---
#### [new 056] Adversarial Agent Behavior Learning in Autonomous Driving Using Deep Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文任务是自动驾驶中对抗性行为学习，解决自主驾驶系统对规则代理对抗行为的鲁棒性问题，通过深度强化学习生成对抗性行为并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.15207v1](http://arxiv.org/pdf/2508.15207v1)**

> **作者:** Arjun Srinivasan; Anubhav Paras; Aniket Bera
>
> **摘要:** Existing approaches in reinforcement learning train an agent to learn desired optimal behavior in an environment with rule based surrounding agents. In safety critical applications such as autonomous driving it is crucial that the rule based agents are modelled properly. Several behavior modelling strategies and IDM models are used currently to model the surrounding agents. We present a learning based method to derive the adversarial behavior for the rule based agents to cause failure scenarios. We evaluate our adversarial agent against all the rule based agents and show the decrease in cumulative reward.
>
---
#### [new 057] Spiking Variational Graph Representation Inference for Video Summarization
- **分类: cs.CV**

- **简介: 该论文针对视频摘要任务，解决全局时间依赖与语义连贯性不足及多通道噪声问题，提出基于脉冲神经网络的关键帧提取、动态聚合图推理及变分推断模块，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.15389v1](http://arxiv.org/pdf/2508.15389v1)**

> **作者:** Wenrui Li; Wei Han; Liang-Jian Deng; Ruiqin Xiong; Xiaopeng Fan
>
> **备注:** Accepted by IEEE TIP
>
> **摘要:** With the rise of short video content, efficient video summarization techniques for extracting key information have become crucial. However, existing methods struggle to capture the global temporal dependencies and maintain the semantic coherence of video content. Additionally, these methods are also influenced by noise during multi-channel feature fusion. We propose a Spiking Variational Graph (SpiVG) Network, which enhances information density and reduces computational complexity. First, we design a keyframe extractor based on Spiking Neural Networks (SNN), leveraging the event-driven computation mechanism of SNNs to learn keyframe features autonomously. To enable fine-grained and adaptable reasoning across video frames, we introduce a Dynamic Aggregation Graph Reasoner, which decouples contextual object consistency from semantic perspective coherence. We present a Variational Inference Reconstruction Module to address uncertainty and noise arising during multi-channel feature fusion. In this module, we employ Evidence Lower Bound Optimization (ELBO) to capture the latent structure of multi-channel feature distributions, using posterior distribution regularization to reduce overfitting. Experimental results show that SpiVG surpasses existing methods across multiple datasets such as SumMe, TVSum, VideoXum, and QFVS. Our codes and pre-trained models are available at https://github.com/liwrui/SpiVG.
>
---
#### [new 058] RCDINO: Enhancing Radar-Camera 3D Object Detection with DINOv2 Semantic Features
- **分类: cs.CV**

- **简介: 论文提出RCDINO，用于雷达-摄像头3D物体检测，通过融合DINOv2语义特征增强视觉表示，提升检测性能，在nuScenes数据集上达到SOTA。**

- **链接: [http://arxiv.org/pdf/2508.15353v1](http://arxiv.org/pdf/2508.15353v1)**

> **作者:** Olga Matykina; Dmitry Yudin
>
> **备注:** Accepted for publication in Optical Memory and Neural Networks, 2025
>
> **摘要:** Three-dimensional object detection is essential for autonomous driving and robotics, relying on effective fusion of multimodal data from cameras and radar. This work proposes RCDINO, a multimodal transformer-based model that enhances visual backbone features by fusing them with semantically rich representations from the pretrained DINOv2 foundation model. This approach enriches visual representations and improves the model's detection performance while preserving compatibility with the baseline architecture. Experiments on the nuScenes dataset demonstrate that RCDINO achieves state-of-the-art performance among radar-camera models, with 56.4 NDS and 48.1 mAP. Our implementation is available at https://github.com/OlgaMatykina/RCDINO.
>
---
#### [new 059] Fast globally optimal Truncated Least Squares point cloud registration with fixed rotation axis
- **分类: cs.CV**

- **简介: 论文针对点云配准中的TLS问题，提出高效算法，在固定旋转轴下实现全局最优，速度快于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.15613v1](http://arxiv.org/pdf/2508.15613v1)**

> **作者:** Ivo Ivanov; Carsten Markgraf
>
> **摘要:** Recent results showed that point cloud registration with given correspondences can be made robust to outlier rates of up to 95\% using the truncated least squares (TLS) formulation. However, solving this combinatorial optimization problem to global optimality is challenging. Provably globally optimal approaches using semidefinite programming (SDP) relaxations take hundreds of seconds for 100 points. In this paper, we propose a novel linear time convex relaxation as well as a contractor method to speed up Branch and Bound (BnB). Our solver can register two 3D point clouds with 100 points to provable global optimality in less than half a second when the axis of rotation is provided. Although it currently cannot solve the full 6DoF problem, it is two orders of magnitude faster than the state-of-the-art SDP solver STRIDE when solving the rotation-only TLS problem. In addition to providing a formal proof for global optimality, we present empirical evidence of global optimality using adversarial instances with local minimas close to the global minimum.
>
---
#### [new 060] Enhancing Novel View Synthesis from extremely sparse views with SfM-free 3D Gaussian Splatting Framework
- **分类: cs.CV**

- **简介: 论文针对极稀疏视图下的新型视图合成任务，解决传统3DGS依赖SfM初始化导致的重建误差问题。提出SfM-free框架，通过密集立体模块估计相机姿态并重建点云，结合视图插值与多尺度正则化提升几何精度和渲染质量。**

- **链接: [http://arxiv.org/pdf/2508.15457v1](http://arxiv.org/pdf/2508.15457v1)**

> **作者:** Zongqi He; Hanmin Li; Kin-Chung Chan; Yushen Zuo; Hao Xie; Zhe Xiao; Jun Xiao; Kin-Man Lam
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** 3D Gaussian Splatting (3DGS) has demonstrated remarkable real-time performance in novel view synthesis, yet its effectiveness relies heavily on dense multi-view inputs with precisely known camera poses, which are rarely available in real-world scenarios. When input views become extremely sparse, the Structure-from-Motion (SfM) method that 3DGS depends on for initialization fails to accurately reconstruct the 3D geometric structures of scenes, resulting in degraded rendering quality. In this paper, we propose a novel SfM-free 3DGS-based method that jointly estimates camera poses and reconstructs 3D scenes from extremely sparse-view inputs. Specifically, instead of SfM, we propose a dense stereo module to progressively estimates camera pose information and reconstructs a global dense point cloud for initialization. To address the inherent problem of information scarcity in extremely sparse-view settings, we propose a coherent view interpolation module that interpolates camera poses based on training view pairs and generates viewpoint-consistent content as additional supervision signals for training. Furthermore, we introduce multi-scale Laplacian consistent regularization and adaptive spatial-aware multi-scale geometry regularization to enhance the quality of geometrical structures and rendered content. Experiments show that our method significantly outperforms other state-of-the-art 3DGS-based approaches, achieving a remarkable 2.75dB improvement in PSNR under extremely sparse-view conditions (using only 2 training views). The images synthesized by our method exhibit minimal distortion while preserving rich high-frequency details, resulting in superior visual quality compared to existing techniques.
>
---
#### [new 061] Towards a 3D Transfer-based Black-box Attack via Critical Feature Guidance
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对三维点云对抗攻击任务，解决无目标模型信息情况下生成有效对抗样本的问题。提出CFG方法，通过关键特征引导提升对抗样本的跨模型转移能力与隐蔽性。**

- **链接: [http://arxiv.org/pdf/2508.15650v1](http://arxiv.org/pdf/2508.15650v1)**

> **作者:** Shuchao Pang; Zhenghan Chen; Shen Zhang; Liming Lu; Siyuan Liang; Anan Du; Yongbin Zhou
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Deep neural networks for 3D point clouds have been demonstrated to be vulnerable to adversarial examples. Previous 3D adversarial attack methods often exploit certain information about the target models, such as model parameters or outputs, to generate adversarial point clouds. However, in realistic scenarios, it is challenging to obtain any information about the target models under conditions of absolute security. Therefore, we focus on transfer-based attacks, where generating adversarial point clouds does not require any information about the target models. Based on our observation that the critical features used for point cloud classification are consistent across different DNN architectures, we propose CFG, a novel transfer-based black-box attack method that improves the transferability of adversarial point clouds via the proposed Critical Feature Guidance. Specifically, our method regularizes the search of adversarial point clouds by computing the importance of the extracted features, prioritizing the corruption of critical features that are likely to be adopted by diverse architectures. Further, we explicitly constrain the maximum deviation extent of the generated adversarial point clouds in the loss function to ensure their imperceptibility. Extensive experiments conducted on the ModelNet40 and ScanObjectNN benchmark datasets demonstrate that the proposed CFG outperforms the state-of-the-art attack methods by a large margin.
>
---
#### [new 062] Waver: Wave Your Way to Lifelike Video Generation
- **分类: cs.CV**

- **简介: 论文提出Waver模型，解决多模态视频生成难题，通过Hybrid Stream DiT架构实现文本、图像到视频的统一生成，结合高质量数据筛选与训练方案，在T2V/I2V任务中超越开源模型，达成高精度视频合成。**

- **链接: [http://arxiv.org/pdf/2508.15761v1](http://arxiv.org/pdf/2508.15761v1)**

> **作者:** Yifu Zhang; Hao Yang; Yuqi Zhang; Yifei Hu; Fengda Zhu; Chuang Lin; Xiaofeng Mei; Yi Jiang; Zehuan Yuan; Bingyue Peng
>
> **摘要:** We present Waver, a high-performance foundation model for unified image and video generation. Waver can directly generate videos with durations ranging from 5 to 10 seconds at a native resolution of 720p, which are subsequently upscaled to 1080p. The model simultaneously supports text-to-video (T2V), image-to-video (I2V), and text-to-image (T2I) generation within a single, integrated framework. We introduce a Hybrid Stream DiT architecture to enhance modality alignment and accelerate training convergence. To ensure training data quality, we establish a comprehensive data curation pipeline and manually annotate and train an MLLM-based video quality model to filter for the highest-quality samples. Furthermore, we provide detailed training and inference recipes to facilitate the generation of high-quality videos. Building on these contributions, Waver excels at capturing complex motion, achieving superior motion amplitude and temporal consistency in video synthesis. Notably, it ranks among the Top 3 on both the T2V and I2V leaderboards at Artificial Analysis (data as of 2025-07-30 10:00 GMT+8), consistently outperforming existing open-source models and matching or surpassing state-of-the-art commercial solutions. We hope this technical report will help the community more efficiently train high-quality video generation models and accelerate progress in video generation technologies. Official page: https://github.com/FoundationVision/Waver.
>
---
#### [new 063] STAGNet: A Spatio-Temporal Graph and LSTM Framework for Accident Anticipation
- **分类: cs.CV**

- **简介: 该论文提出STAGNet框架，融合时空图与LSTM，用于从车载摄像头视频预测事故，提升事故预警的平均精度和碰撞时间。**

- **链接: [http://arxiv.org/pdf/2508.15216v1](http://arxiv.org/pdf/2508.15216v1)**

> **作者:** Vipooshan Vipulananthan; Kumudu Mohottala; Kavindu Chinthana; Nimsara Paramulla; Charith D Chitraranjan
>
> **摘要:** Accident prediction and timely warnings play a key role in improving road safety by reducing the risk of injury to road users and minimizing property damage. Advanced Driver Assistance Systems (ADAS) are designed to support human drivers and are especially useful when they can anticipate potential accidents before they happen. While many existing systems depend on a range of sensors such as LiDAR, radar, and GPS, relying solely on dash-cam video input presents a more challenging but a more cost-effective and easily deployable solution. In this work, we incorporate better spatio-temporal features and aggregate them through a recurrent network to improve upon state-of-the-art graph neural networks for predicting accidents from dash-cam videos. Experiments using three publicly available datasets show that our proposed STAGNet model achieves higher average precision and mean time-to-collision values than previous methods, both when cross-validated on a given dataset and when trained and tested on different datasets.
>
---
#### [new 064] CurveFlow: Curvature-Guided Flow Matching for Image Generation
- **分类: cs.CV**

- **简介: 该论文针对文本到图像生成任务，解决现有线性轨迹模型导致语义不一致问题，提出CurveFlow框架，通过曲率引导学习非线性轨迹并引入曲率正则化，提升生成图像与描述的语义一致性。**

- **链接: [http://arxiv.org/pdf/2508.15093v1](http://arxiv.org/pdf/2508.15093v1)**

> **作者:** Yan Luo; Drake Du; Hao Huang; Yi Fang; Mengyu Wang
>
> **摘要:** Existing rectified flow models are based on linear trajectories between data and noise distributions. This linearity enforces zero curvature, which can inadvertently force the image generation process through low-probability regions of the data manifold. A key question remains underexplored: how does the curvature of these trajectories correlate with the semantic alignment between generated images and their corresponding captions, i.e., instructional compliance? To address this, we introduce CurveFlow, a novel flow matching framework designed to learn smooth, non-linear trajectories by directly incorporating curvature guidance into the flow path. Our method features a robust curvature regularization technique that penalizes abrupt changes in the trajectory's intrinsic dynamics.Extensive experiments on MS COCO 2014 and 2017 demonstrate that CurveFlow achieves state-of-the-art performance in text-to-image generation, significantly outperforming both standard rectified flow variants and other non-linear baselines like Rectified Diffusion. The improvements are especially evident in semantic consistency metrics such as BLEU, METEOR, ROUGE, and CLAIR. This confirms that our curvature-aware modeling substantially enhances the model's ability to faithfully follow complex instructions while simultaneously maintaining high image quality. The code is made publicly available at https://github.com/Harvard-AI-and-Robotics-Lab/CurveFlow.
>
---
#### [new 065] HiRQA: Hierarchical Ranking and Quality Alignment for Opinion-Unaware Image Quality Assessment
- **分类: cs.CV**

- **简介: 该论文提出HiRQA框架，解决无参考图像质量评估中数据偏差和依赖主观标签的问题，通过自监督排序与对比学习，结合高阶损失和合成数据训练，实现强泛化能力与实时部署。**

- **链接: [http://arxiv.org/pdf/2508.15130v1](http://arxiv.org/pdf/2508.15130v1)**

> **作者:** Vaishnav Ramesh; Haining Wang; Md Jahidul Islam
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** Despite significant progress in no-reference image quality assessment (NR-IQA), dataset biases and reliance on subjective labels continue to hinder their generalization performance. We propose HiRQA, Hierarchical Ranking and Quality Alignment), a self-supervised, opinion-unaware framework that offers a hierarchical, quality-aware embedding through a combination of ranking and contrastive learning. Unlike prior approaches that depend on pristine references or auxiliary modalities at inference time, HiRQA predicts quality scores using only the input image. We introduce a novel higher-order ranking loss that supervises quality predictions through relational ordering across distortion pairs, along with an embedding distance loss that enforces consistency between feature distances and perceptual differences. A training-time contrastive alignment loss, guided by structured textual prompts, further enhances the learned representation. Trained only on synthetic distortions, HiRQA generalizes effectively to authentic degradations, as demonstrated through evaluation on various distortions such as lens flare, haze, motion blur, and low-light conditions. For real-time deployment, we introduce \textbf{HiRQA-S}, a lightweight variant with an inference time of only 3.5 ms per image. Extensive experiments across synthetic and authentic benchmarks validate HiRQA's state-of-the-art (SOTA) performance, strong generalization ability, and scalability.
>
---
#### [new 066] Scaling Group Inference for Diverse and High-Quality Generation
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 该论文针对生成模型多输出样本重复问题，提出群体推理方法，通过二次整数规划优化样本质量与多样性，支持大规模候选集处理，适用于文本到图像等多任务生成。**

- **链接: [http://arxiv.org/pdf/2508.15773v1](http://arxiv.org/pdf/2508.15773v1)**

> **作者:** Gaurav Parmar; Or Patashnik; Daniil Ostashev; Kuan-Chieh Wang; Kfir Aberman; Srinivasa Narasimhan; Jun-Yan Zhu
>
> **备注:** Project website: https://www.cs.cmu.edu/~group-inference, GitHub: https://github.com/GaParmar/group-inference
>
> **摘要:** Generative models typically sample outputs independently, and recent inference-time guidance and scaling algorithms focus on improving the quality of individual samples. However, in real-world applications, users are often presented with a set of multiple images (e.g., 4-8) for each prompt, where independent sampling tends to lead to redundant results, limiting user choices and hindering idea exploration. In this work, we introduce a scalable group inference method that improves both the diversity and quality of a group of samples. We formulate group inference as a quadratic integer assignment problem: candidate outputs are modeled as graph nodes, and a subset is selected to optimize sample quality (unary term) while maximizing group diversity (binary term). To substantially improve runtime efficiency, we progressively prune the candidate set using intermediate predictions, allowing our method to scale up to large candidate sets. Extensive experiments show that our method significantly improves group diversity and quality compared to independent sampling baselines and recent inference algorithms. Our framework generalizes across a wide range of tasks, including text-to-image, image-to-image, image prompting, and video generation, enabling generative models to treat multiple outputs as cohesive groups rather than independent samples.
>
---
#### [new 067] Scalable FPGA Framework for Real-Time Denoising in High-Throughput Imaging: A DRAM-Optimized Pipeline using High-Level Synthesis
- **分类: cs.AR; cs.CV; cs.DC; eess.IV; eess.SP; physics.ins-det**

- **简介: 论文提出一种基于FPGA的实时去噪框架，用于处理高吞吐成像中的高速数据，通过HLS优化DRAM缓冲和突发模式接口，降低延迟并减少后续处理数据量。**

- **链接: [http://arxiv.org/pdf/2508.14917v1](http://arxiv.org/pdf/2508.14917v1)**

> **作者:** Weichien Liao
>
> **备注:** FPGA-based denoising pipeline for PRISM-scale imaging. Real-time frame subtraction and averaging via burst-mode AXI4 and DRAM buffering. Benchmarked against CPU/GPU workflows; scalable across multi-bank FPGA setups
>
> **摘要:** High-throughput imaging workflows, such as Parallel Rapid Imaging with Spectroscopic Mapping (PRISM), generate data at rates that exceed conventional real-time processing capabilities. We present a scalable FPGA-based preprocessing pipeline for real-time denoising, implemented via High-Level Synthesis (HLS) and optimized for DRAM-backed buffering. Our architecture performs frame subtraction and averaging directly on streamed image data, minimizing latency through burst-mode AXI4 interfaces. The resulting kernel operates below the inter-frame interval, enabling inline denoising and reducing dataset size for downstream CPU/GPU analysis. Validated under PRISM-scale acquisition, this modular FPGA framework offers a practical solution for latency-sensitive imaging workflows in spectroscopy and microscopy.
>
---
#### [new 068] Label Uncertainty for Ultrasound Segmentation
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; stat.ML**

- **简介: 该论文针对超声分割中的标注不确定性问题，提出利用专家提供的像素级置信度值进行模型训练，通过引入置信度信号提升分割精度及下游临床任务性能。**

- **链接: [http://arxiv.org/pdf/2508.15635v1](http://arxiv.org/pdf/2508.15635v1)**

> **作者:** Malini Shivaram; Gautam Rajendrakumar Gare; Laura Hutchins; Jacob Duplantis; Thomas Deiss; Thales Nogueira Gomes; Thong Tran; Keyur H. Patel; Thomas H Fox; Amita Krishnan; Deva Ramanan; Bennett DeBoisblanc; Ricardo Rodriguez; John Galeotti
>
> **备注:** Paper under review
>
> **摘要:** In medical imaging, inter-observer variability among radiologists often introduces label uncertainty, particularly in modalities where visual interpretation is subjective. Lung ultrasound (LUS) is a prime example-it frequently presents a mixture of highly ambiguous regions and clearly discernible structures, making consistent annotation challenging even for experienced clinicians. In this work, we introduce a novel approach to both labeling and training AI models using expert-supplied, per-pixel confidence values. Rather than treating annotations as absolute ground truth, we design a data annotation protocol that captures the confidence that radiologists have in each labeled region, modeling the inherent aleatoric uncertainty present in real-world clinical data. We demonstrate that incorporating these confidence values during training leads to improved segmentation performance. More importantly, we show that this enhanced segmentation quality translates into better performance on downstream clinically-critical tasks-specifically, estimating S/F oxygenation ratio values, classifying S/F ratio change, and predicting 30-day patient readmission. While we empirically evaluate many methods for exposing the uncertainty to the learning model, we find that a simple approach that trains a model on binarized labels obtained with a (60%) confidence threshold works well. Importantly, high thresholds work far better than a naive approach of a 50% threshold, indicating that training on very confident pixels is far more effective. Our study systematically investigates the impact of training with varying confidence thresholds, comparing not only segmentation metrics but also downstream clinical outcomes. These results suggest that label confidence is a valuable signal that, when properly leveraged, can significantly enhance the reliability and clinical utility of AI in medical imaging.
>
---
#### [new 069] \textit{adder-viz}: Real-Time Visualization Software for Transcoding Event Video
- **分类: cs.MM; cs.CV; cs.HC; eess.IV**

- **简介: 论文提出adder-viz软件，用于实时可视化事件视频转码，解决传统方法在灵活性、速度和压缩性上的局限，通过改进ADΔER表示和软件实现高效可视化。**

- **链接: [http://arxiv.org/pdf/2508.14996v1](http://arxiv.org/pdf/2508.14996v1)**

> **作者:** Andrew C. Freeman; Luke Reinkensmeyer
>
> **备注:** Accepted to the Open-Source Track at ACM Multimedia 2025
>
> **摘要:** Recent years have brought about a surge in neuromorphic ``event'' video research, primarily targeting computer vision applications. Event video eschews video frames in favor of asynchronous, per-pixel intensity samples. While much work has focused on a handful of representations for specific event cameras, these representations have shown limitations in flexibility, speed, and compressibility. We previously proposed the unified AD{\Delta}ER representation to address these concerns. This paper introduces numerous improvements to the \textit{adder-viz} software for visualizing real-time event transcode processes and applications in-the-loop. The MIT-licensed software is available from a centralized repository at \href{https://github.com/ac-freeman/adder-codec-rs}{https://github.com/ac-freeman/adder-codec-rs}.
>
---
#### [new 070] On the Effectiveness of Graph Reordering for Accelerating Approximate Nearest Neighbor Search on GPU
- **分类: cs.IR; cs.CV; cs.DB; cs.DC; cs.DS**

- **简介: 论文研究图重排对GPU上基于图的近似最近邻搜索（ANNS）的加速效果，通过统一框架评估不同重排策略，优化内存布局以提升QPS。**

- **链接: [http://arxiv.org/pdf/2508.15436v1](http://arxiv.org/pdf/2508.15436v1)**

> **作者:** Yutaro Oguri; Mai Nishimura; Yusuke Matsui
>
> **摘要:** We present the first systematic investigation of graph reordering effects for graph-based Approximate Nearest Neighbor Search (ANNS) on a GPU. While graph-based ANNS has become the dominant paradigm for modern AI applications, recent approaches focus on algorithmic innovations while neglecting memory layout considerations that significantly affect execution time. Our unified evaluation framework enables comprehensive evaluation of diverse reordering strategies across different graph indices through a graph adapter that converts arbitrary graph topologies into a common representation and a GPU-optimized graph traversal engine. We conduct a comprehensive analysis across diverse datasets and state-of-the-art graph indices, introducing analysis metrics that quantify the relationship between structural properties and memory layout effectiveness. Our GPU-targeted reordering achieves up to 15$\%$ QPS improvements while preserving search accuracy, demonstrating that memory layout optimization operates orthogonally to existing algorithmic innovations. We will release all code upon publication to facilitate reproducibility and foster further research.
>
---
#### [new 071] A Vision-Based Shared-Control Teleoperation Scheme for Controlling the Robotic Arm of a Four-Legged Robot
- **分类: cs.RO; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 该论文提出基于视觉的共享控制方案，通过外部相机和机器学习模型实时映射操作员手腕动作到四足机器人机械臂，结合轨迹规划避免碰撞，提升工业遥控安全性与易用性。**

- **链接: [http://arxiv.org/pdf/2508.14994v1](http://arxiv.org/pdf/2508.14994v1)**

> **作者:** Murilo Vinicius da Silva; Matheus Hipolito Carvalho; Juliano Negri; Thiago Segreto; Gustavo J. G. Lahr; Ricardo V. Godoy; Marcelo Becker
>
> **摘要:** In hazardous and remote environments, robotic systems perform critical tasks demanding improved safety and efficiency. Among these, quadruped robots with manipulator arms offer mobility and versatility for complex operations. However, teleoperating quadruped robots is challenging due to the lack of integrated obstacle detection and intuitive control methods for the robotic arm, increasing collision risks in confined or dynamically changing workspaces. Teleoperation via joysticks or pads can be non-intuitive and demands a high level of expertise due to its complexity, culminating in a high cognitive load on the operator. To address this challenge, a teleoperation approach that directly maps human arm movements to the robotic manipulator offers a simpler and more accessible solution. This work proposes an intuitive remote control by leveraging a vision-based pose estimation pipeline that utilizes an external camera with a machine learning-based model to detect the operator's wrist position. The system maps these wrist movements into robotic arm commands to control the robot's arm in real-time. A trajectory planner ensures safe teleoperation by detecting and preventing collisions with both obstacles and the robotic arm itself. The system was validated on the real robot, demonstrating robust performance in real-time control. This teleoperation approach provides a cost-effective solution for industrial applications where safety, precision, and ease of use are paramount, ensuring reliable and intuitive robotic control in high-risk environments.
>
---
#### [new 072] End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对医疗诊断中知识缺失与推理不可追溯问题，提出端到端代理RAG系统Deep-DxSearch，通过强化学习训练实现可追溯的检索增强推理，提升诊断准确性并优于现有基线模型。**

- **链接: [http://arxiv.org/pdf/2508.15746v1](http://arxiv.org/pdf/2508.15746v1)**

> **作者:** Qiaoyu Zheng; Yuze Sun; Chaoyi Wu; Weike Zhao; Pengcheng Qiu; Yongguo Yu; Kun Sun; Yanfeng Wang; Ya Zhang; Weidi Xie
>
> **备注:** 35 pages, 5 figures, 3 tables
>
> **摘要:** Accurate diagnosis with medical large language models is hindered by knowledge gaps and hallucinations. Retrieval and tool-augmented methods help, but their impact is limited by weak use of external knowledge and poor feedback-reasoning traceability. To address these challenges, We introduce Deep-DxSearch, an agentic RAG system trained end-to-end with reinforcement learning (RL) that enables steer tracebale retrieval-augmented reasoning for medical diagnosis. In Deep-DxSearch, we first construct a large-scale medical retrieval corpus comprising patient records and reliable medical knowledge sources to support retrieval-aware reasoning across diagnostic scenarios. More crutially, we frame the LLM as the core agent and the retrieval corpus as its environment, using tailored rewards on format, retrieval, reasoning structure, and diagnostic accuracy, thereby evolving the agentic RAG policy from large-scale data through RL. Experiments demonstrate that our end-to-end agentic RL training framework consistently outperforms prompt-engineering and training-free RAG approaches across multiple data centers. After training, Deep-DxSearch achieves substantial gains in diagnostic accuracy, surpassing strong diagnostic baselines such as GPT-4o, DeepSeek-R1, and other medical-specific frameworks for both common and rare disease diagnosis under in-distribution and out-of-distribution settings. Moreover, ablation studies on reward design and retrieval corpus components confirm their critical roles, underscoring the uniqueness and effectiveness of our approach compared with traditional implementations. Finally, case studies and interpretability analyses highlight improvements in Deep-DxSearch's diagnostic policy, providing deeper insight into its performance gains and supporting clinicians in delivering more reliable and precise preliminary diagnoses. See https://github.com/MAGIC-AI4Med/Deep-DxSearch.
>
---
#### [new 073] DoSReMC: Domain Shift Resilient Mammography Classification using Batch Normalization Adaptation
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对乳腺X光分类中的域移位问题，提出DoSReMC框架，通过调整批归一化层和对抗训练提升跨域泛化能力，无需重新训练模型。**

- **链接: [http://arxiv.org/pdf/2508.15452v1](http://arxiv.org/pdf/2508.15452v1)**

> **作者:** Uğurcan Akyüz; Deniz Katircioglu-Öztürk; Emre K. Süslü; Burhan Keleş; Mete C. Kaya; Gamze Durhan; Meltem G. Akpınar; Figen B. Demirkazık; Gözde B. Akar
>
> **摘要:** Numerous deep learning-based solutions have been developed for the automatic recognition of breast cancer using mammography images. However, their performance often declines when applied to data from different domains, primarily due to domain shift - the variation in data distributions between source and target domains. This performance drop limits the safe and equitable deployment of AI in real-world clinical settings. In this study, we present DoSReMC (Domain Shift Resilient Mammography Classification), a batch normalization (BN) adaptation framework designed to enhance cross-domain generalization without retraining the entire model. Using three large-scale full-field digital mammography (FFDM) datasets - including HCTP, a newly introduced, pathologically confirmed in-house dataset - we conduct a systematic cross-domain evaluation with convolutional neural networks (CNNs). Our results demonstrate that BN layers are a primary source of domain dependence: they perform effectively when training and testing occur within the same domain, and they significantly impair model generalization under domain shift. DoSReMC addresses this limitation by fine-tuning only the BN and fully connected (FC) layers, while preserving pretrained convolutional filters. We further integrate this targeted adaptation with an adversarial training scheme, yielding additional improvements in cross-domain generalizability. DoSReMC can be readily incorporated into existing AI pipelines and applied across diverse clinical environments, providing a practical pathway toward more robust and generalizable mammography classification systems.
>
---
#### [new 074] Pathology-Informed Latent Diffusion Model for Anomaly Detection in Lymph Node Metastasis
- **分类: eess.IV; cs.CV**

- **简介: 本文提出基于病理学提示的扩散模型，用于数字病理学中淋巴结转移的无监督异常检测，结合视觉-语言模型提升区分能力，并在胃和乳腺数据集上验证有效性。**

- **链接: [http://arxiv.org/pdf/2508.15236v1](http://arxiv.org/pdf/2508.15236v1)**

> **作者:** Jiamu Wang; Keunho Byeon; Jinsol Song; Anh Nguyen; Sangjeong Ahn; Sung Hak Lee; Jin Tae Kwak
>
> **摘要:** Anomaly detection is an emerging approach in digital pathology for its ability to efficiently and effectively utilize data for disease diagnosis. While supervised learning approaches deliver high accuracy, they rely on extensively annotated datasets, suffering from data scarcity in digital pathology. Unsupervised anomaly detection, however, offers a viable alternative by identifying deviations from normal tissue distributions without requiring exhaustive annotations. Recently, denoising diffusion probabilistic models have gained popularity in unsupervised anomaly detection, achieving promising performance in both natural and medical imaging datasets. Building on this, we incorporate a vision-language model with a diffusion model for unsupervised anomaly detection in digital pathology, utilizing histopathology prompts during reconstruction. Our approach employs a set of pathology-related keywords associated with normal tissues to guide the reconstruction process, facilitating the differentiation between normal and abnormal tissues. To evaluate the effectiveness of the proposed method, we conduct experiments on a gastric lymph node dataset from a local hospital and assess its generalization ability under domain shift using a public breast lymph node dataset. The experimental results highlight the potential of the proposed method for unsupervised anomaly detection across various organs in digital pathology. Code: https://github.com/QuIIL/AnoPILaD.
>
---
#### [new 075] Are Virtual DES Images a Valid Alternative to the Real Ones?
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文任务是评估虚拟DES图像在CESM病变分类中的有效性。解决是否能替代真实DES图像的问题，通过三种模型生成虚拟DES图像并测试分类性能，结果显示预训练U-Net表现最佳，但仍有提升空间。**

- **链接: [http://arxiv.org/pdf/2508.15594v1](http://arxiv.org/pdf/2508.15594v1)**

> **作者:** Ana C. Perre; Luís A. Alexandre; Luís C. Freire
>
> **备注:** 10 pages, 4 figures, 3 tables
>
> **摘要:** Contrast-enhanced spectral mammography (CESM) is an imaging modality that provides two types of images, commonly known as low-energy (LE) and dual-energy subtracted (DES) images. In many domains, particularly in medicine, the emergence of image-to-image translation techniques has enabled the artificial generation of images using other images as input. Within CESM, applying such techniques to generate DES images from LE images could be highly beneficial, potentially reducing patient exposure to radiation associated with high-energy image acquisition. In this study, we investigated three models for the artificial generation of DES images (virtual DES): a pre-trained U-Net model, a U-Net trained end-to-end model, and a CycleGAN model. We also performed a series of experiments to assess the impact of using virtual DES images on the classification of CESM examinations into malignant and non-malignant categories. To our knowledge, this is the first study to evaluate the impact of virtual DES images on CESM lesion classification. The results demonstrate that the best performance was achieved with the pre-trained U-Net model, yielding an F1 score of 85.59% when using the virtual DES images, compared to 90.35% with the real DES images. This discrepancy likely results from the additional diagnostic information in real DES images, which contributes to a higher classification accuracy. Nevertheless, the potential for virtual DES image generation is considerable and future advancements may narrow this performance gap to a level where exclusive reliance on virtual DES images becomes clinically viable.
>
---
#### [new 076] Side Effects of Erasing Concepts from Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究概念擦除技术（CETs）在扩散模型中的副作用，旨在解决其易被绕过的问题。通过提出Side Effect Evaluation基准，量化了CETs在邻近概念影响、目标规避和属性泄露方面的效果，揭示了超类-子类层级和语义相似提示可绕过CETs的机制，并公开了相关数据集与工具。**

- **链接: [http://arxiv.org/pdf/2508.15124v1](http://arxiv.org/pdf/2508.15124v1)**

> **作者:** Shaswati Saha; Sourajit Saha; Manas Gaur; Tejas Gokhale
>
> **备注:** Findings of the Association for Computational Linguistics: EMNLP 2025
>
> **摘要:** Concerns about text-to-image (T2I) generative models infringing on privacy, copyright, and safety have led to the development of Concept Erasure Techniques (CETs). The goal of an effective CET is to prohibit the generation of undesired ``target'' concepts specified by the user, while preserving the ability to synthesize high-quality images of the remaining concepts. In this work, we demonstrate that CETs can be easily circumvented and present several side effects of concept erasure. For a comprehensive measurement of the robustness of CETs, we present Side Effect Evaluation (\see), an evaluation benchmark that consists of hierarchical and compositional prompts that describe objects and their attributes. This dataset and our automated evaluation pipeline quantify side effects of CETs across three aspects: impact on neighboring concepts, evasion of targets, and attribute leakage. Our experiments reveal that CETs can be circumvented by using superclass-subclass hierarchy and semantically similar prompts, such as compositional variants of the target. We show that CETs suffer from attribute leakage and counterintuitive phenomena of attention concentration or dispersal. We release our dataset, code, and evaluation tools to aid future work on robust concept erasure.
>
---
#### [new 077] Fusing Structural Phenotypes with Functional Data for Early Prediction of Primary Angle Closure Glaucoma Progression
- **分类: q-bio.QM; cs.AI; cs.CV; eess.IV**

- **简介: 该论文通过融合视神经结构（OCT）与功能数据（视觉场），利用机器学习预测原发性闭角型青光眼（PACG）进展速度，识别关键预测因素以提高疾病风险分类准确性。**

- **链接: [http://arxiv.org/pdf/2508.14922v1](http://arxiv.org/pdf/2508.14922v1)**

> **作者:** Swati Sharma; Thanadet Chuangsuwanich; Royston K. Y. Tan; Shimna C. Prasad; Tin A. Tun; Shamira A. Perera; Martin L. Buist; Tin Aung; Monisha E. Nongpiur; Michaël J. A. Girard
>
> **备注:** 23 pages, 5 figures, 3 tables
>
> **摘要:** Purpose: To classify eyes as slow or fast glaucoma progressors in patients with primary angle closure glaucoma (PACG) using an integrated approach combining optic nerve head (ONH) structural features and sector-based visual field (VF) functional parameters. Methods: PACG patients with >5 reliable VF tests over >5 years were included. Progression was assessed in Zeiss Forum, with baseline VF within six months of OCT. Fast progression was VFI decline <-2.0% per year; slow progression >-2.0% per year. OCT volumes were AI-segmented to extract 31 ONH parameters. The Glaucoma Hemifield Test defined five regions per hemifield, aligned with RNFL distribution. Mean sensitivity per region was combined with structural parameters to train ML classifiers. Multiple models were tested, and SHAP identified key predictors. Main outcome measures: Classification of slow versus fast progressors using combined structural and functional data. Results: We analyzed 451 eyes from 299 patients. Mean VFI progression was -0.92% per year; 369 eyes progressed slowly and 82 rapidly. The Random Forest model combining structural and functional features achieved the best performance (AUC = 0.87, 2000 Monte Carlo iterations). SHAP identified six key predictors: inferior MRW, inferior and inferior-temporal RNFL thickness, nasal-temporal LC curvature, superior nasal VF sensitivity, and inferior RNFL and GCL+IPL thickness. Models using only structural or functional features performed worse with AUC of 0.82 and 0.78, respectively. Conclusions: Combining ONH structural and VF functional parameters significantly improves classification of progression risk in PACG. Inferior ONH features, MRW and RNFL thickness, were the most predictive, highlighting the critical role of ONH morphology in monitoring disease progression.
>
---
#### [new 078] Zero-shot Volumetric CT Super-Resolution using 3D Gaussian Splatting with Upsampled 2D X-ray Projection Priors
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对零样本体积CT超分辨率重建任务，解决无需配对数据且恢复细节困难的问题。提出利用扩散模型生成上采样2D X射线投影先验，结合3D高斯点绘制与负alpha混合技术，提升重建质量。**

- **链接: [http://arxiv.org/pdf/2508.15151v1](http://arxiv.org/pdf/2508.15151v1)**

> **作者:** Jeonghyun Noh; Hyun-Jic Oh; Byungju Chae; Won-Ki Jeong
>
> **摘要:** Computed tomography (CT) is widely used in clinical diagnosis, but acquiring high-resolution (HR) CT is limited by radiation exposure risks. Deep learning-based super-resolution (SR) methods have been studied to reconstruct HR from low-resolution (LR) inputs. While supervised SR approaches have shown promising results, they require large-scale paired LR-HR volume datasets that are often unavailable. In contrast, zero-shot methods alleviate the need for paired data by using only a single LR input, but typically struggle to recover fine anatomical details due to limited internal information. To overcome these, we propose a novel zero-shot 3D CT SR framework that leverages upsampled 2D X-ray projection priors generated by a diffusion model. Exploiting the abundance of HR 2D X-ray data, we train a diffusion model on large-scale 2D X-ray projection and introduce a per-projection adaptive sampling strategy. It selects the generative process for each projection, thus providing HR projections as strong external priors for 3D CT reconstruction. These projections serve as inputs to 3D Gaussian splatting for reconstructing a 3D CT volume. Furthermore, we propose negative alpha blending (NAB-GS) that allows negative values in Gaussian density representation. NAB-GS enables residual learning between LR and diffusion-based projections, thereby enhancing high-frequency structure reconstruction. Experiments on two datasets show that our method achieves superior quantitative and qualitative results for 3D CT SR.
>
---
#### [new 079] Lang2Lift: A Framework for Language-Guided Pallet Detection and Pose Estimation Integrated in Autonomous Outdoor Forklift Operation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Lang2Lift框架，通过自然语言指导实现户外叉车自主托盘检测与6D位姿估计，解决复杂环境下的自动化搬运问题，结合Florence-2/SAM-2和FoundationPose提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.15427v1](http://arxiv.org/pdf/2508.15427v1)**

> **作者:** Huy Hoang Nguyen; Johannes Huemer; Markus Murschitz; Tobias Glueck; Minh Nhat Vu; Andreas Kugi
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** The logistics and construction industries face persistent challenges in automating pallet handling, especially in outdoor environments with variable payloads, inconsistencies in pallet quality and dimensions, and unstructured surroundings. In this paper, we tackle automation of a critical step in pallet transport: the pallet pick-up operation. Our work is motivated by labor shortages, safety concerns, and inefficiencies in manually locating and retrieving pallets under such conditions. We present Lang2Lift, a framework that leverages foundation models for natural language-guided pallet detection and 6D pose estimation, enabling operators to specify targets through intuitive commands such as "pick up the steel beam pallet near the crane." The perception pipeline integrates Florence-2 and SAM-2 for language-grounded segmentation with FoundationPose for robust pose estimation in cluttered, multi-pallet outdoor scenes under variable lighting. The resulting poses feed into a motion planning module for fully autonomous forklift operation. We validate Lang2Lift on the ADAPT autonomous forklift platform, achieving 0.76 mIoU pallet segmentation accuracy on a real-world test dataset. Timing and error analysis demonstrate the system's robustness and confirm its feasibility for deployment in operational logistics and construction environments. Video demonstrations are available at https://eric-nguyen1402.github.io/lang2lift.github.io/
>
---
#### [new 080] Hessian-based lightweight neural network for brain vessel segmentation on a minimal training dataset
- **分类: eess.IV; cs.CV; I.4.6; I.5.4; J.3**

- **简介: 该论文针对脑MRA图像血管分割任务，解决小样本标注难题。提出HessNet模型，基于Hessian矩阵设计轻量网络（6000参数），在CPU上实现高效训练，仅需少量数据即可达到SOTA精度，并构建了半自动标注的IXI数据集。**

- **链接: [http://arxiv.org/pdf/2508.15660v1](http://arxiv.org/pdf/2508.15660v1)**

> **作者:** Alexandra Bernadotte; Elfimov Nikita; Mikhail Shutov; Ivan Menshikov
>
> **备注:** 11 pages, 2 figures
>
> **摘要:** Accurate segmentation of blood vessels in brain magnetic resonance angiography (MRA) is essential for successful surgical procedures, such as aneurysm repair or bypass surgery. Currently, annotation is primarily performed through manual segmentation or classical methods, such as the Frangi filter, which often lack sufficient accuracy. Neural networks have emerged as powerful tools for medical image segmentation, but their development depends on well-annotated training datasets. However, there is a notable lack of publicly available MRA datasets with detailed brain vessel annotations. To address this gap, we propose a novel semi-supervised learning lightweight neural network with Hessian matrices on board for 3D segmentation of complex structures such as tubular structures, which we named HessNet. The solution is a Hessian-based neural network with only 6000 parameters. HessNet can run on the CPU and significantly reduces the resource requirements for training neural networks. The accuracy of vessel segmentation on a minimal training dataset reaches state-of-the-art results. It helps us create a large, semi-manually annotated brain vessel dataset of brain MRA images based on the IXI dataset (annotated 200 images). Annotation was performed by three experts under the supervision of three neurovascular surgeons after applying HessNet. It provides high accuracy of vessel segmentation and allows experts to focus only on the most complex important cases. The dataset is available at https://git.scinalytics.com/terilat/VesselDatasetPartly.
>
---
#### [new 081] Exploring the Landscape of Non-Equilibrium Memories with Neural Cellular Automata
- **分类: cond-mat.stat-mech; cs.CV; cs.LG; nlin.CG**

- **简介: 该论文通过结合数学证明与机器学习，研究二维非平衡记忆的多样性，揭示了在扰动下长期存储信息的多种机制，发现超越Toom规则的纠错方式及噪声依赖的存储特性。**

- **链接: [http://arxiv.org/pdf/2508.15726v1](http://arxiv.org/pdf/2508.15726v1)**

> **作者:** Ethan Lake; Ehsan Pajouheshgar
>
> **备注:** 4+9 pages
>
> **摘要:** We investigate the landscape of many-body memories: families of local non-equilibrium dynamics that retain information about their initial conditions for thermodynamically long time scales, even in the presence of arbitrary perturbations. In two dimensions, the only well-studied memory is Toom's rule. Using a combination of rigorous proofs and machine learning methods, we show that the landscape of 2D memories is in fact quite vast. We discover memories that correct errors in ways qualitatively distinct from Toom's rule, have ordered phases stabilized by fluctuations, and preserve information only in the presence of noise. Taken together, our results show that physical systems can perform robust information storage in many distinct ways, and demonstrate that the physics of many-body memories is richer than previously realized. Interactive visualizations of the dynamics studied in this work are available at https://memorynca.github.io/2D.
>
---
#### [new 082] Bladder Cancer Diagnosis with Deep Learning: A Multi-Task Framework and Online Platform
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文通过多任务深度学习框架和在线平台实现膀胱癌诊断，解决传统膀胱镜检查依赖医生经验导致的主观性问题，提出分类、分割与分子分型模型，实验显示高准确率，提升诊断效率与可及性。**

- **链接: [http://arxiv.org/pdf/2508.15379v1](http://arxiv.org/pdf/2508.15379v1)**

> **作者:** Jinliang Yu; Mingduo Xie; Yue Wang; Tianfan Fu; Xianglai Xu; Jiajun Wang
>
> **摘要:** Clinical cystoscopy, the current standard for bladder cancer diagnosis, suffers from significant reliance on physician expertise, leading to variability and subjectivity in diagnostic outcomes. There is an urgent need for objective, accurate, and efficient computational approaches to improve bladder cancer diagnostics. Leveraging recent advancements in deep learning, this study proposes an integrated multi-task deep learning framework specifically designed for bladder cancer diagnosis from cystoscopic images. Our framework includes a robust classification model using EfficientNet-B0 enhanced with Convolutional Block Attention Module (CBAM), an advanced segmentation model based on ResNet34-UNet++ architecture with self-attention mechanisms and attention gating, and molecular subtyping using ConvNeXt-Tiny to classify molecular markers such as HER-2 and Ki-67. Additionally, we introduce a Gradio-based online diagnostic platform integrating all developed models, providing intuitive features including multi-format image uploads, bilingual interfaces, and dynamic threshold adjustments. Extensive experimentation demonstrates the effectiveness of our methods, achieving outstanding accuracy (93.28%), F1-score (82.05%), and AUC (96.41%) for classification tasks, and exceptional segmentation performance indicated by a Dice coefficient of 0.9091. The online platform significantly improved the accuracy, efficiency, and accessibility of clinical bladder cancer diagnostics, enabling practical and user-friendly deployment. The code is publicly available. Our multi-task framework and integrated online tool collectively advance the field of intelligent bladder cancer diagnosis by improving clinical reliability, supporting early tumor detection, and enabling real-time diagnostic feedback. These contributions mark a significant step toward AI-assisted decision-making in urology.
>
---
#### [new 083] Deep Equilibrium Convolutional Sparse Coding for Hyperspectral Image Denoising
- **分类: eess.IV; cs.CV**

- **简介: 本文针对超光谱图像去噪任务，提出基于Deep Equilibrium模型的DECSC框架，融合局部-非局部-全局特征，通过卷积稀疏编码与Transformer模块提升去噪鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.15553v1](http://arxiv.org/pdf/2508.15553v1)**

> **作者:** Jin Ye; Jingran Wang; Fengchao Xiong; Jingzhou Chen; Yuntao Qian
>
> **摘要:** Hyperspectral images (HSIs) play a crucial role in remote sensing but are often degraded by complex noise patterns. Ensuring the physical property of the denoised HSIs is vital for robust HSI denoising, giving the rise of deep unfolding-based methods. However, these methods map the optimization of a physical model to a learnable network with a predefined depth, which lacks convergence guarantees. In contrast, Deep Equilibrium (DEQ) models treat the hidden layers of deep networks as the solution to a fixed-point problem and models them as infinite-depth networks, naturally consistent with the optimization. Under the framework of DEQ, we propose a Deep Equilibrium Convolutional Sparse Coding (DECSC) framework that unifies local spatial-spectral correlations, nonlocal spatial self-similarities, and global spatial consistency for robust HSI denoising. Within the convolutional sparse coding (CSC) framework, we enforce shared 2D convolutional sparse representation to ensure global spatial consistency across bands, while unshared 3D convolutional sparse representation captures local spatial-spectral details. To further exploit nonlocal self-similarities, a transformer block is embedded after the 2D CSC. Additionally, a detail enhancement module is integrated with the 3D CSC to promote image detail preservation. We formulate the proximal gradient descent of the CSC model as a fixed-point problem and transform the iterative updates into a learnable network architecture within the framework of DEQ. Experimental results demonstrate that our DECSC method achieves superior denoising performance compared to state-of-the-art methods.
>
---
#### [new 084] Probability Density from Latent Diffusion Models for Out-of-Distribution Detection
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对分布外检测（OOD）问题，探讨似然方法在实践中的失效原因，通过在表示空间训练扩散模型评估基于密度估计的检测器性能。**

- **链接: [http://arxiv.org/pdf/2508.15737v1](http://arxiv.org/pdf/2508.15737v1)**

> **作者:** Joonas Järve; Karl Kaspar Haavel; Meelis Kull
>
> **备注:** ECAI 2025
>
> **摘要:** Despite rapid advances in AI, safety remains the main bottleneck to deploying machine-learning systems. A critical safety component is out-of-distribution detection: given an input, decide whether it comes from the same distribution as the training data. In generative models, the most natural OOD score is the data likelihood. Actually, under the assumption of uniformly distributed OOD data, the likelihood is even the optimal OOD detector, as we show in this work. However, earlier work reported that likelihood often fails in practice, raising doubts about its usefulness. We explore whether, in practice, the representation space also suffers from the inability to learn good density estimation for OOD detection, or if it is merely a problem of the pixel space typically used in generative models. To test this, we trained a Variational Diffusion Model not on images, but on the representation space of a pre-trained ResNet-18 to assess the performance of our likelihood-based detector in comparison to state-of-the-art methods from the OpenOOD suite.
>
---
#### [new 085] Decentralized Vision-Based Autonomous Aerial Wildlife Monitoring
- **分类: cs.RO; cs.AI; cs.CV; cs.MA; I.2.9**

- **简介: 该论文提出一种去中心化多无人机系统，用于自主监测野生动物。通过单相机视觉算法，在动态环境中实现大规模物种的鲁棒识别与跟踪，解决传统方法效率低、依赖集中控制的问题。**

- **链接: [http://arxiv.org/pdf/2508.15038v1](http://arxiv.org/pdf/2508.15038v1)**

> **作者:** Makram Chahine; William Yang; Alaa Maalouf; Justin Siriska; Ninad Jadhav; Daniel Vogt; Stephanie Gil; Robert Wood; Daniela Rus
>
> **摘要:** Wildlife field operations demand efficient parallel deployment methods to identify and interact with specific individuals, enabling simultaneous collective behavioral analysis, and health and safety interventions. Previous robotics solutions approach the problem from the herd perspective, or are manually operated and limited in scale. We propose a decentralized vision-based multi-quadrotor system for wildlife monitoring that is scalable, low-bandwidth, and sensor-minimal (single onboard RGB camera). Our approach enables robust identification and tracking of large species in their natural habitat. We develop novel vision-based coordination and tracking algorithms designed for dynamic, unstructured environments without reliance on centralized communication or control. We validate our system through real-world experiments, demonstrating reliable deployment in diverse field conditions.
>
---
#### [new 086] Scalable Event-Based Video Streaming for Machines with MoQ
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 该论文针对机器端事件视频流媒体的数据传输问题，提出基于Media Over QUIC协议的低延迟新格式，解决传统视频流在神经形态传感器下的传输效率与兼容性挑战。**

- **链接: [http://arxiv.org/pdf/2508.15003v1](http://arxiv.org/pdf/2508.15003v1)**

> **作者:** Andrew C. Freeman
>
> **备注:** Accepted to ACM Mile High Video 2025
>
> **摘要:** Lossy compression and rate-adaptive streaming are a mainstay in traditional video steams. However, a new class of neuromorphic ``event'' sensors records video with asynchronous pixel samples rather than image frames. These sensors are designed for computer vision applications, rather than human video consumption. Until now, researchers have focused their efforts primarily on application development, ignoring the crucial problem of data transmission. We survey the landscape of event-based video systems, discuss the technical issues with our recent scalable event streaming work, and propose a new low-latency event streaming format based on the latest additions to the Media Over QUIC protocol draft.
>
---
#### [new 087] See it. Say it. Sorted: Agentic System for Compositional Diagram Generation
- **分类: cs.AI; cs.CV; cs.MA**

- **简介: 该论文研究手绘草图到结构化图表的生成任务，解决扩散模型在空间精度和符号结构上的不足。提出无需训练的代理系统，融合视觉-语言模型与大语言模型，通过迭代编辑生成可编辑的SVG代码，实现精准布局与符号组成。**

- **链接: [http://arxiv.org/pdf/2508.15222v1](http://arxiv.org/pdf/2508.15222v1)**

> **作者:** Hantao Zhang; Jingyang Liu; Ed Li
>
> **摘要:** We study sketch-to-diagram generation: converting rough hand sketches into precise, compositional diagrams. Diffusion models excel at photorealism but struggle with the spatial precision, alignment, and symbolic structure required for flowcharts. We introduce See it. Say it. Sorted., a training-free agentic system that couples a Vision-Language Model (VLM) with Large Language Models (LLMs) to produce editable Scalable Vector Graphics (SVG) programs. The system runs an iterative loop in which a Critic VLM proposes a small set of qualitative, relational edits; multiple candidate LLMs synthesize SVG updates with diverse strategies (conservative->aggressive, alternative, focused); and a Judge VLM selects the best candidate, ensuring stable improvement. This design prioritizes qualitative reasoning over brittle numerical estimates, preserves global constraints (e.g., alignment, connectivity), and naturally supports human-in-the-loop corrections. On 10 sketches derived from flowcharts in published papers, our method more faithfully reconstructs layout and structure than two frontier closed-source image generation LLMs (GPT-5 and Gemini-2.5-Pro), accurately composing primitives (e.g., multi-headed arrows) without inserting unwanted text. Because outputs are programmatic SVGs, the approach is readily extensible to presentation tools (e.g., PowerPoint) via APIs and can be specialized with improved prompts and task-specific tools. The codebase is open-sourced at https://github.com/hantaoZhangrichard/see_it_say_it_sorted.git.
>
---
#### [new 088] Explainable Knowledge Distillation for Efficient Medical Image Classification
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 本论文针对医学图像分类任务，提出基于知识蒸馏的高效可解释模型，结合教师模型与混合监督，设计紧凑学生模型并使用可视化方法解释决策，验证其在资源受限环境下的有效性。**

- **链接: [http://arxiv.org/pdf/2508.15251v1](http://arxiv.org/pdf/2508.15251v1)**

> **作者:** Aqib Nazir Mir; Danish Raza Rizvi
>
> **摘要:** This study comprehensively explores knowledge distillation frameworks for COVID-19 and lung cancer classification using chest X-ray (CXR) images. We employ high-capacity teacher models, including VGG19 and lightweight Vision Transformers (Visformer-S and AutoFormer-V2-T), to guide the training of a compact, hardware-aware student model derived from the OFA-595 supernet. Our approach leverages hybrid supervision, combining ground-truth labels with teacher models' soft targets to balance accuracy and computational efficiency. We validate our models on two benchmark datasets: COVID-QU-Ex and LCS25000, covering multiple classes, including COVID-19, healthy, non-COVID pneumonia, lung, and colon cancer. To interpret the spatial focus of the models, we employ Score-CAM-based visualizations, which provide insight into the reasoning process of both teacher and student networks. The results demonstrate that the distilled student model maintains high classification performance with significantly reduced parameters and inference time, making it an optimal choice in resource-constrained clinical environments. Our work underscores the importance of combining model efficiency with explainability for practical, trustworthy medical AI solutions.
>
---
#### [new 089] Intern-S1: A Scientific Multimodal Foundation Model
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 论文提出Intern-S1，旨在解决科学领域基础模型性能不足问题，通过多模态Mixture-of-Experts架构与强化学习提升科学推理能力，在分子合成、热力学预测等任务中超越闭源模型。**

- **链接: [http://arxiv.org/pdf/2508.15763v1](http://arxiv.org/pdf/2508.15763v1)**

> **作者:** Lei Bai; Zhongrui Cai; Maosong Cao; Weihan Cao; Chiyu Chen; Haojiong Chen; Kai Chen; Pengcheng Chen; Ying Chen; Yongkang Chen; Yu Cheng; Yu Cheng; Pei Chu; Tao Chu; Erfei Cui; Ganqu Cui; Long Cui; Ziyun Cui; Nianchen Deng; Ning Ding; Nanqin Dong; Peijie Dong; Shihan Dou; Sinan Du; Haodong Duan; Caihua Fan; Ben Gao; Changjiang Gao; Jianfei Gao; Songyang Gao; Yang Gao; Zhangwei Gao; Jiaye Ge; Qiming Ge; Lixin Gu; Yuzhe Gu; Aijia Guo; Qipeng Guo; Xu Guo; Conghui He; Junjun He; Yili Hong; Siyuan Hou; Caiyu Hu; Hanglei Hu; Jucheng Hu; Ming Hu; Zhouqi Hua; Haian Huang; Junhao Huang; Xu Huang; Zixian Huang; Zhe Jiang; Lingkai Kong; Linyang Li; Peiji Li; Pengze Li; Shuaibin Li; Tianbin Li; Wei Li; Yuqiang Li; Dahua Lin; Junyao Lin; Tianyi Lin; Zhishan Lin; Hongwei Liu; Jiangning Liu; Jiyao Liu; Junnan Liu; Kai Liu; Kaiwen Liu; Kuikun Liu; Shichun Liu; Shudong Liu; Wei Liu; Xinyao Liu; Yuhong Liu; Zhan Liu; Yinquan Lu; Haijun Lv; Hongxia Lv; Huijie Lv; Qidang Lv; Ying Lv; Chengqi Lyu; Chenglong Ma; Jianpeng Ma; Ren Ma; Runmin Ma; Runyuan Ma; Xinzhu Ma; Yichuan Ma; Zihan Ma; Sixuan Mi; Junzhi Ning; Wenchang Ning; Xinle Pang; Jiahui Peng; Runyu Peng; Yu Qiao; Jiantao Qiu; Xiaoye Qu; Yuan Qu; Yuchen Ren; Fukai Shang; Wenqi Shao; Junhao Shen; Shuaike Shen; Chunfeng Song; Demin Song; Diping Song; Chenlin Su; Weijie Su; Weigao Sun; Yu Sun; Qian Tan; Cheng Tang; Huanze Tang; Kexian Tang; Shixiang Tang; Jian Tong; Aoran Wang; Bin Wang; Dong Wang; Lintao Wang; Rui Wang; Weiyun Wang; Wenhai Wang; Yi Wang; Ziyi Wang; Ling-I Wu; Wen Wu; Yue Wu; Zijian Wu; Linchen Xiao; Shuhao Xing; Chao Xu; Huihui Xu; Jun Xu; Ruiliang Xu; Wanghan Xu; GanLin Yang; Yuming Yang; Haochen Ye; Jin Ye; Shenglong Ye; Jia Yu; Jiashuo Yu; Jing Yu; Fei Yuan; Bo Zhang; Chao Zhang; Chen Zhang; Hongjie Zhang; Jin Zhang; Qiaosheng Zhang; Qiuyinzhe Zhang; Songyang Zhang; Taolin Zhang; Wenlong Zhang; Wenwei Zhang; Yechen Zhang; Ziyang Zhang; Haiteng Zhao; Qian Zhao; Xiangyu Zhao; Xiangyu Zhao; Bowen Zhou; Dongzhan Zhou; Peiheng Zhou; Yuhao Zhou; Yunhua Zhou; Dongsheng Zhu; Lin Zhu; Yicheng Zou
>
> **摘要:** In recent years, a plethora of open-source foundation models have emerged, achieving remarkable progress in some widely attended fields, with performance being quite close to that of closed-source models. However, in high-value but more challenging scientific professional fields, either the fields still rely on expert models, or the progress of general foundation models lags significantly compared to those in popular areas, far from sufficient for transforming scientific research and leaving substantial gap between open-source models and closed-source models in these scientific domains. To mitigate this gap and explore a step further toward Artificial General Intelligence (AGI), we introduce Intern-S1, a specialized generalist equipped with general understanding and reasoning capabilities with expertise to analyze multiple science modal data. Intern-S1 is a multimodal Mixture-of-Experts (MoE) model with 28 billion activated parameters and 241 billion total parameters, continually pre-trained on 5T tokens, including over 2.5T tokens from scientific domains. In the post-training stage, Intern-S1 undergoes offline and then online reinforcement learning (RL) in InternBootCamp, where we propose Mixture-of-Rewards (MoR) to synergize the RL training on more than 1000 tasks simultaneously. Through integrated innovations in algorithms, data, and training systems, Intern-S1 achieved top-tier performance in online RL training.On comprehensive evaluation benchmarks, Intern-S1 demonstrates competitive performance on general reasoning tasks among open-source models and significantly outperforms open-source models in scientific domains, surpassing closed-source state-of-the-art models in professional tasks, such as molecular synthesis planning, reaction condition prediction, predicting thermodynamic stabilities for crystals. Our models are available at https://huggingface.co/internlm/Intern-S1.
>
---
#### [new 090] "Does the cafe entrance look accessible? Where is the door?" Towards Geospatial AI Agents for Visual Inquiries
- **分类: cs.HC; cs.AI; cs.CV; H.5; I.2**

- **简介: 论文提出Geo-Visual Agents，通过整合地理图像与GIS数据，解决复杂视觉空间查询（如评估咖啡馆入口可达性），拓展交互式地图对非结构化视觉信息的处理能力。**

- **链接: [http://arxiv.org/pdf/2508.15752v1](http://arxiv.org/pdf/2508.15752v1)**

> **作者:** Jon E. Froehlich; Jared Hwang; Zeyu Wang; John S. O'Meara; Xia Su; William Huang; Yang Zhang; Alex Fiannaca; Philip Nelson; Shaun Kane
>
> **备注:** Accepted to the ICCV'25 Workshop "Vision Foundation Models and Generative AI for Accessibility: Challenges and Opportunities"
>
> **摘要:** Interactive digital maps have revolutionized how people travel and learn about the world; however, they rely on pre-existing structured data in GIS databases (e.g., road networks, POI indices), limiting their ability to address geo-visual questions related to what the world looks like. We introduce our vision for Geo-Visual Agents--multimodal AI agents capable of understanding and responding to nuanced visual-spatial inquiries about the world by analyzing large-scale repositories of geospatial images, including streetscapes (e.g., Google Street View), place-based photos (e.g., TripAdvisor, Yelp), and aerial imagery (e.g., satellite photos) combined with traditional GIS data sources. We define our vision, describe sensing and interaction approaches, provide three exemplars, and enumerate key challenges and opportunities for future work.
>
---
#### [new 091] Self-supervised physics-informed generative networks for phase retrieval from a single X-ray hologram
- **分类: physics.optics; cs.CV; eess.IV; physics.comp-ph; physics.ins-det**

- **简介: 该论文解决从单张X射线全息图中恢复相位与吸收信息的任务，针对传统方法需人工调参及依赖特定条件的问题，提出基于物理信息的自监督GAN模型，无需训练数据即可实现高效、准确的相位恢复。**

- **链接: [http://arxiv.org/pdf/2508.15530v1](http://arxiv.org/pdf/2508.15530v1)**

> **作者:** Xiaogang Yang; Dawit Hailu; Vojtěch Kulvait; Thomas Jentschke; Silja Flenner; Imke Greving; Stuart I. Campbell; Johannes Hagemann; Christian G. Schroer; Tak Ming Wong; Julian Moosmann
>
> **备注:** Version of record published in Optics Express, Vol. 33, Issue 17, pp. 35832-35851 (2025). Merged article, 20 pages of main text, 1 page of supplement header, and 7 pages of supplement (total 28 pages). Contains 10 figures in the main article and 5 figures in the supplement
>
> **摘要:** X-ray phase contrast imaging significantly improves the visualization of structures with weak or uniform absorption, broadening its applications across a wide range of scientific disciplines. Propagation-based phase contrast is particularly suitable for time- or dose-critical in vivo/in situ/operando (tomography) experiments because it requires only a single intensity measurement. However, the phase information of the wave field is lost during the measurement and must be recovered. Conventional algebraic and iterative methods often rely on specific approximations or boundary conditions that may not be met by many samples or experimental setups. In addition, they require manual tuning of reconstruction parameters by experts, making them less adaptable for complex or variable conditions. Here we present a self-learning approach for solving the inverse problem of phase retrieval in the near-field regime of Fresnel theory using a single intensity measurement (hologram). A physics-informed generative adversarial network is employed to reconstruct both the phase and absorbance of the unpropagated wave field in the sample plane from a single hologram. Unlike most deep learning approaches for phase retrieval, our approach does not require paired, unpaired, or simulated training data. This significantly broadens the applicability of our approach, as acquiring or generating suitable training data remains a major challenge due to the wide variability in sample types and experimental configurations. The algorithm demonstrates robust and consistent performance across diverse imaging conditions and sample types, delivering quantitative, high-quality reconstructions for both simulated data and experimental datasets acquired at beamline P05 at PETRA III (DESY, Hamburg), operated by Helmholtz-Zentrum Hereon. Furthermore, it enables the simultaneous retrieval of both phase and absorption information.
>
---
## 更新

#### [replaced 001] A Systematic Study of Deep Learning Models and xAI Methods for Region-of-Interest Detection in MRI Scans
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.14151v2](http://arxiv.org/pdf/2508.14151v2)**

> **作者:** Justin Yiu; Kushank Arora; Daniel Steinberg; Rohit Ghiya
>
> **摘要:** Magnetic Resonance Imaging (MRI) is an essential diagnostic tool for assessing knee injuries. However, manual interpretation of MRI slices remains time-consuming and prone to inter-observer variability. This study presents a systematic evaluation of various deep learning architectures combined with explainable AI (xAI) techniques for automated region of interest (ROI) detection in knee MRI scans. We investigate both supervised and self-supervised approaches, including ResNet50, InceptionV3, Vision Transformers (ViT), and multiple U-Net variants augmented with multi-layer perceptron (MLP) classifiers. To enhance interpretability and clinical relevance, we integrate xAI methods such as Grad-CAM and Saliency Maps. Model performance is assessed using AUC for classification and PSNR/SSIM for reconstruction quality, along with qualitative ROI visualizations. Our results demonstrate that ResNet50 consistently excels in classification and ROI identification, outperforming transformer-based models under the constraints of the MRNet dataset. While hybrid U-Net + MLP approaches show potential for leveraging spatial features in reconstruction and interpretability, their classification performance remains lower. Grad-CAM consistently provided the most clinically meaningful explanations across architectures. Overall, CNN-based transfer learning emerges as the most effective approach for this dataset, while future work with larger-scale pretraining may better unlock the potential of transformer models.
>
---
#### [replaced 002] Learning Motion Blur Robust Vision Transformers for Real-Time UAV Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.05383v2](http://arxiv.org/pdf/2407.05383v2)**

> **作者:** You Wu; Xucheng Wang; Dan Zeng; Hengzhou Ye; Xiaolan Xie; Qijun Zhao; Shuiwang Li
>
> **摘要:** Unmanned aerial vehicle (UAV) tracking is critical for applications like surveillance, search-and-rescue, and autonomous navigation. However, the high-speed movement of UAVs and targets introduces unique challenges, including real-time processing demands and severe motion blur, which degrade the performance of existing generic trackers. While single-stream vision transformer (ViT) architectures have shown promise in visual tracking, their computational inefficiency and lack of UAV-specific optimizations limit their practicality in this domain. In this paper, we boost the efficiency of this framework by tailoring it into an adaptive computation framework that dynamically exits Transformer blocks for real-time UAV tracking. The motivation behind this is that tracking tasks with fewer challenges can be adequately addressed using low-level feature representations. Simpler tasks can often be handled with less demanding, lower-level features. This approach allows the model use computational resources more efficiently by focusing on complex tasks and conserving resources for easier ones. Another significant enhancement introduced in this paper is the improved effectiveness of ViTs in handling motion blur, a common issue in UAV tracking caused by the fast movements of either the UAV, the tracked objects, or both. This is achieved by acquiring motion blur robust representations through enforcing invariance in the feature representation of the target with respect to simulated motion blur. We refer to our proposed approach as BDTrack. Extensive experiments conducted on four tracking benchmarks validate the effectiveness and versatility of our approach, demonstrating its potential as a practical and effective approach for real-time UAV tracking. Code is released at: https://github.com/wuyou3474/BDTrack.
>
---
#### [replaced 003] Human-Object Interaction from Human-Level Instructions
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.17840v3](http://arxiv.org/pdf/2406.17840v3)**

> **作者:** Zhen Wu; Jiaman Li; Pei Xu; C. Karen Liu
>
> **备注:** ICCV 2025, project page: https://hoifhli.github.io/
>
> **摘要:** Intelligent agents must autonomously interact with the environments to perform daily tasks based on human-level instructions. They need a foundational understanding of the world to accurately interpret these instructions, along with precise low-level movement and interaction skills to execute the derived actions. In this work, we propose the first complete system for synthesizing physically plausible, long-horizon human-object interactions for object manipulation in contextual environments, driven by human-level instructions. We leverage large language models (LLMs) to interpret the input instructions into detailed execution plans. Unlike prior work, our system is capable of generating detailed finger-object interactions, in seamless coordination with full-body movements. We also train a policy to track generated motions in physics simulation via reinforcement learning (RL) to ensure physical plausibility of the motion. Our experiments demonstrate the effectiveness of our system in synthesizing realistic interactions with diverse objects in complex environments, highlighting its potential for real-world applications.
>
---
#### [replaced 004] Real-Time Beach Litter Detection and Counting: A Comparative Analysis of RT-DETR Model Variants
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.13101v2](http://arxiv.org/pdf/2508.13101v2)**

> **作者:** Miftahul Huda; Arsyiah Azahra; Putri Maulida Chairani; Dimas Rizky Ramadhani; Nabila Azhari; Ade Lailani
>
> **摘要:** Coastal pollution is a pressing global environmental issue, necessitating scalable and automated solutions for monitoring and management. This study investigates the efficacy of the Real-Time Detection Transformer (RT-DETR), a state-of-the-art, end-to-end object detection model, for the automated detection and counting of beach litter. A rigorous comparative analysis is conducted between two model variants, RT-DETR-Large (RT-DETR-L) and RT-DETR-Extra-Large (RT-DETR-X), trained on a publicly available dataset of coastal debris. The evaluation reveals that the RT-DETR-X model achieves marginally superior accuracy, with a mean Average Precision at 50\% IoU (mAP@50) of 0.816 and a mAP@50-95 of 0.612, compared to the RT-DETR-L model's 0.810 and 0.606, respectively. However, this minor performance gain is realized at a significant computational cost; the RT-DETR-L model demonstrates a substantially faster inference time of 20.1 ms versus 34.5 ms for the RT-DETR-X. The findings suggest that the RT-DETR-L model offers a more practical and efficient solution for real-time, in-field deployment due to its superior balance of processing speed and detection accuracy. This research provides valuable insights into the application of advanced Transformer-based detectors for environmental conservation, highlighting the critical trade-offs between model complexity and operational viability.
>
---
#### [replaced 005] Creating a Historical Migration Dataset from Finnish Church Records, 1800-1920
- **分类: cs.CV; I.4.6, J.5**

- **链接: [http://arxiv.org/pdf/2506.07960v3](http://arxiv.org/pdf/2506.07960v3)**

> **作者:** Ari Vesalainen; Jenna Kanerva; Aida Nitsch; Kiia Korsu; Ilari Larkiola; Laura Ruotsalainen; Filip Ginter
>
> **摘要:** This article presents a large-scale effort to create a structured dataset of internal migration in Finland between 1800 and 1920 using digitized church moving records. These records, maintained by Evangelical-Lutheran parishes, document the migration of individuals and families and offer a valuable source for studying historical demographic patterns. The dataset includes over six million entries extracted from approximately 200,000 images of handwritten migration records. The data extraction process was automated using a deep learning pipeline that included layout analysis, table detection, cell classification, and handwriting recognition. The complete pipeline was applied to all images, resulting in a structured dataset suitable for research. The dataset can be used to study internal migration, urbanization, and family migration, and the spread of disease in preindustrial Finland. A case study from the Elim\"aki parish shows how local migration histories can be reconstructed. The work demonstrates how large volumes of handwritten archival material can be transformed into structured data to support historical and demographic research.
>
---
#### [replaced 006] Referring Expression Instance Retrieval and A Strong End-to-End Baseline
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18246v4](http://arxiv.org/pdf/2506.18246v4)**

> **作者:** Xiangzhao Hao; Kuan Zhu; Hongyu Guo; Haiyun Guo; Ning Jiang; Quan Lu; Ming Tang; Jinqiao Wang
>
> **备注:** ACMMM2025
>
> **摘要:** Using natural language to query visual information is a fundamental need in real-world applications. Text-Image Retrieval (TIR) retrieves a target image from a gallery based on an image-level description, while Referring Expression Comprehension (REC) localizes a target object within a given image using an instance-level description. However, real-world applications often present more complex demands. Users typically query an instance-level description across a large gallery and expect to receive both relevant image and the corresponding instance location. In such scenarios, TIR struggles with fine-grained descriptions and object-level localization, while REC is limited in its ability to efficiently search large galleries and lacks an effective ranking mechanism. In this paper, we introduce a new task called \textbf{Referring Expression Instance Retrieval (REIR)}, which supports both instance-level retrieval and localization based on fine-grained referring expressions. First, we propose a large-scale benchmark for REIR, named REIRCOCO, constructed by prompting advanced vision-language models to generate high-quality referring expressions for instances in the MSCOCO and RefCOCO datasets. Second, we present a baseline method, Contrastive Language-Instance Alignment with Relation Experts (CLARE), which employs a dual-stream architecture to address REIR in an end-to-end manner. Given a referring expression, the textual branch encodes it into a query embedding. The visual branch detects candidate objects and extracts their instance-level visual features. The most similar candidate to the query is selected for bounding box prediction. CLARE is first trained on object detection and REC datasets to establish initial grounding capabilities, then optimized via Contrastive Language-Instance Alignment (CLIA) for improved retrieval across images. We will release our code and benchmark publicly.
>
---
#### [replaced 007] MCA-RG: Enhancing LLMs with Medical Concept Alignment for Radiology Report Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.06992v2](http://arxiv.org/pdf/2507.06992v2)**

> **作者:** Qilong Xing; Zikai Song; Youjia Zhang; Na Feng; Junqing Yu; Wei Yang
>
> **备注:** MICCAI 2025
>
> **摘要:** Despite significant advancements in adapting Large Language Models (LLMs) for radiology report generation (RRG), clinical adoption remains challenging due to difficulties in accurately mapping pathological and anatomical features to their corresponding text descriptions. Additionally, semantic agnostic feature extraction further hampers the generation of accurate diagnostic reports. To address these challenges, we introduce Medical Concept Aligned Radiology Report Generation (MCA-RG), a knowledge-driven framework that explicitly aligns visual features with distinct medical concepts to enhance the report generation process. MCA-RG utilizes two curated concept banks: a pathology bank containing lesion-related knowledge, and an anatomy bank with anatomical descriptions. The visual features are aligned with these medical concepts and undergo tailored enhancement. We further propose an anatomy-based contrastive learning procedure to improve the generalization of anatomical features, coupled with a matching loss for pathological features to prioritize clinically relevant regions. Additionally, a feature gating mechanism is employed to filter out low-quality concept features. Finally, the visual features are corresponding to individual medical concepts, and are leveraged to guide the report generation process. Experiments on two public benchmarks (MIMIC-CXR and CheXpert Plus) demonstrate that MCA-RG achieves superior performance, highlighting its effectiveness in radiology report generation.
>
---
#### [replaced 008] Handle-based Mesh Deformation Guided By Vision Language Model
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04562v2](http://arxiv.org/pdf/2506.04562v2)**

> **作者:** Xingpeng Sun; Shiyang Jia; Zherong Pan; Kui Wu; Aniket Bera
>
> **备注:** 19 pages
>
> **摘要:** Mesh deformation is a fundamental tool in 3D content manipulation. Despite extensive prior research, existing approaches often suffer from low output quality, require significant manual tuning, or depend on data-intensive training. To address these limitations, we introduce a training-free, handle-based mesh deformation method. % Our core idea is to leverage a Vision-Language Model (VLM) to interpret and manipulate a handle-based interface through prompt engineering. We begin by applying cone singularity detection to identify a sparse set of potential handles. The VLM is then prompted to select both the deformable sub-parts of the mesh and the handles that best align with user instructions. Subsequently, we query the desired deformed positions of the selected handles in screen space. To reduce uncertainty inherent in VLM predictions, we aggregate the results from multiple camera views using a novel multi-view voting scheme. % Across a suite of benchmarks, our method produces deformations that align more closely with user intent, as measured by CLIP and GPTEval3D scores, while introducing low distortion -- quantified via membrane energy. In summary, our approach is training-free, highly automated, and consistently delivers high-quality mesh deformations.
>
---
#### [replaced 009] CMAMRNet: A Contextual Mask-Aware Network Enhancing Mural Restoration Through Comprehensive Mask Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.07140v2](http://arxiv.org/pdf/2508.07140v2)**

> **作者:** Yingtie Lei; Fanghai Yi; Yihang Dong; Weihuang Liu; Xiaofeng Zhang; Zimeng Li; Chi-Man Pun; Xuhang Chen
>
> **备注:** Accepted by BMVC 2025
>
> **摘要:** Murals, as invaluable cultural artifacts, face continuous deterioration from environmental factors and human activities. Digital restoration of murals faces unique challenges due to their complex degradation patterns and the critical need to preserve artistic authenticity. Existing learning-based methods struggle with maintaining consistent mask guidance throughout their networks, leading to insufficient focus on damaged regions and compromised restoration quality. We propose CMAMRNet, a Contextual Mask-Aware Mural Restoration Network that addresses these limitations through comprehensive mask guidance and multi-scale feature extraction. Our framework introduces two key components: (1) the Mask-Aware Up/Down-Sampler (MAUDS), which ensures consistent mask sensitivity across resolution scales through dedicated channel-wise feature selection and mask-guided feature fusion; and (2) the Co-Feature Aggregator (CFA), operating at both the highest and lowest resolutions to extract complementary features for capturing fine textures and global structures in degraded regions. Experimental results on benchmark datasets demonstrate that CMAMRNet outperforms state-of-the-art methods, effectively preserving both structural integrity and artistic details in restored murals. The code is available at~\href{https://github.com/CXH-Research/CMAMRNet}{https://github.com/CXH-Research/CMAMRNet}.
>
---
#### [replaced 010] Preacher: Paper-to-Video Agentic System
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.09632v4](http://arxiv.org/pdf/2508.09632v4)**

> **作者:** Jingwei Liu; Ling Yang; Hao Luo; Fan Wang; Hongyan Li; Mengdi Wang
>
> **备注:** Include some mistakes
>
> **摘要:** The paper-to-video task converts a research paper into a structured video abstract, distilling key concepts, methods, and conclusions into an accessible, well-organized format. While state-of-the-art video generation models demonstrate potential, they are constrained by limited context windows, rigid video duration constraints, limited stylistic diversity, and an inability to represent domain-specific knowledge. To address these limitations, we introduce Preacher, the first paper-to-video agentic system. Preacher employs a topdown approach to decompose, summarize, and reformulate the paper, followed by bottom-up video generation, synthesizing diverse video segments into a coherent abstract. To align cross-modal representations, we define key scenes and introduce a Progressive Chain of Thought (P-CoT) for granular, iterative planning. Preacher successfully generates high-quality video abstracts across five research fields, demonstrating expertise beyond current video generation models. Code will be released at: https://github.com/GenVerse/Paper2Video
>
---
#### [replaced 011] ReconDreamer-RL: Enhancing Reinforcement Learning via Diffusion-based Scene Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.08170v2](http://arxiv.org/pdf/2508.08170v2)**

> **作者:** Chaojun Ni; Guosheng Zhao; Xiaofeng Wang; Zheng Zhu; Wenkang Qin; Xinze Chen; Guanghong Jia; Guan Huang; Wenjun Mei
>
> **摘要:** Reinforcement learning for training end-to-end autonomous driving models in closed-loop simulations is gaining growing attention. However, most simulation environments differ significantly from real-world conditions, creating a substantial simulation-to-reality (sim2real) gap. To bridge this gap, some approaches utilize scene reconstruction techniques to create photorealistic environments as a simulator. While this improves realistic sensor simulation, these methods are inherently constrained by the distribution of the training data, making it difficult to render high-quality sensor data for novel trajectories or corner case scenarios. Therefore, we propose ReconDreamer-RL, a framework designed to integrate video diffusion priors into scene reconstruction to aid reinforcement learning, thereby enhancing end-to-end autonomous driving training. Specifically, in ReconDreamer-RL, we introduce ReconSimulator, which combines the video diffusion prior for appearance modeling and incorporates a kinematic model for physical modeling, thereby reconstructing driving scenarios from real-world data. This narrows the sim2real gap for closed-loop evaluation and reinforcement learning. To cover more corner-case scenarios, we introduce the Dynamic Adversary Agent (DAA), which adjusts the trajectories of surrounding vehicles relative to the ego vehicle, autonomously generating corner-case traffic scenarios (e.g., cut-in). Finally, the Cousin Trajectory Generator (CTG) is proposed to address the issue of training data distribution, which is often biased toward simple straight-line movements. Experiments show that ReconDreamer-RL improves end-to-end autonomous driving training, outperforming imitation learning methods with a 5x reduction in the Collision Ratio.
>
---
#### [replaced 012] LV-Net: Anatomy-aware lateral ventricle shape modeling with a case study on Alzheimer's disease
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2508.06055v2](http://arxiv.org/pdf/2508.06055v2)**

> **作者:** Wonjung Park; Suhyun Ahn; Jinah Park
>
> **摘要:** Lateral ventricle (LV) shape analysis holds promise as a biomarker for neurological diseases; however, challenges remain due to substantial shape variability across individuals and segmentation difficulties arising from limited MRI resolution. We introduce LV-Net, a novel framework for producing individualized 3D LV meshes from brain MRI by deforming an anatomy-aware joint LV-hippocampus template mesh. By incorporating anatomical relationships embedded within the joint template, LV-Net reduces boundary segmentation artifacts and improves reconstruction robustness. In addition, by classifying the vertices of the template mesh based on their anatomical adjacency, our method enhances point correspondence across subjects, leading to more accurate LV shape statistics. We demonstrate that LV-Net achieves superior reconstruction accuracy, even in the presence of segmentation imperfections, and delivers more reliable shape descriptors across diverse datasets. Finally, we apply LV-Net to Alzheimer's disease analysis, identifying LV subregions that show significantly associations with the disease relative to cognitively normal controls. The codes for LV shape modeling are available at https://github.com/PWonjung/LV_Shape_Modeling.
>
---
#### [replaced 013] Evaluation Agent: Efficient and Promptable Evaluation Framework for Visual Generative Models
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.09645v3](http://arxiv.org/pdf/2412.09645v3)**

> **作者:** Fan Zhang; Shulin Tian; Ziqi Huang; Yu Qiao; Ziwei Liu
>
> **备注:** Equal contributions from first three authors. Project page: https://vchitect.github.io/Evaluation-Agent-project Code: https://github.com/Vchitect/Evaluation-Agent
>
> **摘要:** Recent advancements in visual generative models have enabled high-quality image and video generation, opening diverse applications. However, evaluating these models often demands sampling hundreds or thousands of images or videos, making the process computationally expensive, especially for diffusion-based models with inherently slow sampling. Moreover, existing evaluation methods rely on rigid pipelines that overlook specific user needs and provide numerical results without clear explanations. In contrast, humans can quickly form impressions of a model's capabilities by observing only a few samples. To mimic this, we propose the Evaluation Agent framework, which employs human-like strategies for efficient, dynamic, multi-round evaluations using only a few samples per round, while offering detailed, user-tailored analyses. It offers four key advantages: 1) efficiency, 2) promptable evaluation tailored to diverse user needs, 3) explainability beyond single numerical scores, and 4) scalability across various models and tools. Experiments show that Evaluation Agent reduces evaluation time to 10% of traditional methods while delivering comparable results. The Evaluation Agent framework is fully open-sourced to advance research in visual generative models and their efficient evaluation.
>
---
#### [replaced 014] Inverse Problem Sampling in Latent Space Using Sequential Monte Carlo
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.05908v3](http://arxiv.org/pdf/2502.05908v3)**

> **作者:** Idan Achituve; Hai Victor Habi; Amir Rosenfeld; Arnon Netzer; Idit Diamant; Ethan Fetaya
>
> **摘要:** In image processing, solving inverse problems is the task of finding plausible reconstructions of an image that was corrupted by some (usually known) degradation operator. Commonly, this process is done using a generative image model that can guide the reconstruction towards solutions that appear natural. The success of diffusion models over the last few years has made them a leading candidate for this task. However, the sequential nature of diffusion models makes this conditional sampling process challenging. Furthermore, since diffusion models are often defined in the latent space of an autoencoder, the encoder-decoder transformations introduce additional difficulties. To address these challenges, we suggest a novel sampling method based on sequential Monte Carlo (SMC) in the latent space of diffusion models. We name our method LD-SMC. We define a generative model for the data using additional auxiliary observations and perform posterior inference with SMC sampling based on a reverse diffusion process. Empirical evaluations on ImageNet and FFHQ show the benefits of LD-SMC over competing methods in various inverse problem tasks and especially in challenging inpainting tasks.
>
---
#### [replaced 015] TextSplat: Text-Guided Semantic Fusion for Generalizable Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.09588v2](http://arxiv.org/pdf/2504.09588v2)**

> **作者:** Zhicong Wu; Hongbin Xu; Gang Xu; Ping Nie; Zhixin Yan; Jinkai Zheng; Liangqiong Qu; Ming Li; Liqiang Nie
>
> **摘要:** Recent advancements in Generalizable Gaussian Splatting have enabled robust 3D reconstruction from sparse input views by utilizing feed-forward Gaussian Splatting models, achieving superior cross-scene generalization. However, while many methods focus on geometric consistency, they often neglect the potential of text-driven guidance to enhance semantic understanding, which is crucial for accurately reconstructing fine-grained details in complex scenes. To address this limitation, we propose TextSplat--the first text-driven Generalizable Gaussian Splatting framework. By employing a text-guided fusion of diverse semantic cues, our framework learns robust cross-modal feature representations that improve the alignment of geometric and semantic information, producing high-fidelity 3D reconstructions. Specifically, our framework employs three parallel modules to obtain complementary representations: the Diffusion Prior Depth Estimator for accurate depth information, the Semantic Aware Segmentation Network for detailed semantic information, and the Multi-View Interaction Network for refined cross-view features. Then, in the Text-Guided Semantic Fusion Module, these representations are integrated via the text-guided and attention-based feature aggregation mechanism, resulting in enhanced 3D Gaussian parameters enriched with detailed semantic cues. Experimental results on various benchmark datasets demonstrate improved performance compared to existing methods across multiple evaluation metrics, validating the effectiveness of our framework. The code will be publicly available.
>
---
#### [replaced 016] Architectural Co-Design for Zero-Shot Anomaly Detection: Decoupling Representation and Dynamically Fusing Features in CLIP
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.07819v2](http://arxiv.org/pdf/2508.07819v2)**

> **作者:** Ke Ma; Jun Long; Hongxiao Fei; Liujie Hua; Yiran Qian; Zhen Dai; Yueyi Luo
>
> **备注:** 4 pages, 1 reference, 3 figures, icassp 2026
>
> **摘要:** Pre-trained Vision-Language Models (VLMs) face a significant adaptation gap when applied to Zero-Shot Anomaly Detection (ZSAD), stemming from their lack of local inductive biases for dense prediction and their reliance on inflexible feature fusion paradigms. We address these limitations through an Architectural Co-Design framework that jointly refines feature representation and cross-modal fusion. Our method integrates a parameter-efficient Convolutional Low-Rank Adaptation (Conv-LoRA) adapter to inject local inductive biases for fine-grained representation, and introduces a Dynamic Fusion Gateway (DFG) that leverages visual context to adaptively modulate text prompts, enabling a powerful bidirectional fusion. Extensive experiments on diverse industrial and medical benchmarks demonstrate superior accuracy and robustness, validating that this synergistic co-design is critical for robustly adapting foundation models to dense perception tasks.
>
---
#### [replaced 017] Exploring Spatial-Temporal Dynamics in Event-based Facial Micro-Expression Analysis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11988v2](http://arxiv.org/pdf/2508.11988v2)**

> **作者:** Nicolas Mastropasqua; Ignacio Bugueno-Cordova; Rodrigo Verschae; Daniel Acevedo; Pablo Negri; Maria E. Buemi
>
> **摘要:** Micro-expression analysis has applications in domains such as Human-Robot Interaction and Driver Monitoring Systems. Accurately capturing subtle and fast facial movements remains difficult when relying solely on RGB cameras, due to limitations in temporal resolution and sensitivity to motion blur. Event cameras offer an alternative, with microsecond-level precision, high dynamic range, and low latency. However, public datasets featuring event-based recordings of Action Units are still scarce. In this work, we introduce a novel, preliminary multi-resolution and multi-modal micro-expression dataset recorded with synchronized RGB and event cameras under variable lighting conditions. Two baseline tasks are evaluated to explore the spatial-temporal dynamics of micro-expressions: Action Unit classification using Spiking Neural Networks (51.23\% accuracy with events vs. 23.12\% with RGB), and frame reconstruction using Conditional Variational Autoencoders, achieving SSIM = 0.8513 and PSNR = 26.89 dB with high-resolution event input. These promising results show that event-based data can be used for micro-expression recognition and frame reconstruction.
>
---
#### [replaced 018] CaLiV: LiDAR-to-Vehicle Calibration of Arbitrary Sensor Setups
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.01987v2](http://arxiv.org/pdf/2504.01987v2)**

> **作者:** Ilir Tahiraj; Markus Edinger; Dominik Kulmer; Markus Lienkamp
>
> **摘要:** In autonomous systems, sensor calibration is essential for safe and efficient navigation in dynamic environments. Accurate calibration is a prerequisite for reliable perception and planning tasks such as object detection and obstacle avoidance. Many existing LiDAR calibration methods require overlapping fields of view, while others use external sensing devices or postulate a feature-rich environment. In addition, Sensor-to-Vehicle calibration is not supported by the vast majority of calibration algorithms. In this work, we propose a novel target-based technique for extrinsic Sensor-to-Sensor and Sensor-to-Vehicle calibration of multi-LiDAR systems called CaLiV. This algorithm works for non-overlapping fields of view and does not require any external sensing devices. First, we apply motion to produce field of view overlaps and utilize a simple Unscented Kalman Filter to obtain vehicle poses. Then, we use the Gaussian mixture model-based registration framework GMMCalib to align the point clouds in a common calibration frame. Finally, we reduce the task of recovering the sensor extrinsics to a minimization problem. We show that both translational and rotational Sensor-to-Sensor errors can be solved accurately by our method. In addition, all Sensor-to-Vehicle rotation angles can also be calibrated with high accuracy. We validate the simulation results in real-world experiments. The code is open-source and available on https://github.com/TUMFTM/CaLiV.
>
---
#### [replaced 019] NucleiMix: Realistic Data Augmentation for Nuclei Instance Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.16671v2](http://arxiv.org/pdf/2410.16671v2)**

> **作者:** Jiamu Wang; Jin Tae Kwak
>
> **摘要:** Nuclei instance segmentation is an essential task in pathology image analysis, serving as the foundation for many downstream applications. The release of several public datasets has significantly advanced research in this area, yet many existing methods struggle with data imbalance issues. To address this challenge, this study introduces a data augmentation method, called NucleiMix, which is designed to balance the distribution of nuclei types by increasing the number of rare-type nuclei within datasets. NucleiMix operates in two phases. In the first phase, it identifies candidate locations similar to the surroundings of rare-type nuclei and inserts rare-type nuclei into the candidate locations. In the second phase, it employs a progressive inpainting strategy using a pre-trained diffusion model to seamlessly integrate rare-type nuclei into their new environments in replacement of major-type nuclei or background locations. We systematically evaluate the effectiveness of NucleiMix on three public datasets using two popular nuclei instance segmentation models. The results demonstrate the superior ability of NucleiMix to synthesize realistic rare-type nuclei and to enhance the quality of nuclei segmentation and classification in an accurate and robust manner.
>
---
#### [replaced 020] Physics-Driven Autoregressive State Space Models for Medical Image Reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.09331v3](http://arxiv.org/pdf/2412.09331v3)**

> **作者:** Bilal Kabas; Fuat Arslan; Valiyeh A. Nezhad; Saban Ozturk; Emine U. Saritas; Tolga Çukur
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** Medical image reconstruction from undersampled acquisitions is an ill-posed inverse problem requiring accurate recovery of anatomical structures from incomplete measurements. Physics-driven (PD) network models have gained prominence for this task by integrating data-consistency mechanisms with learned priors, enabling improved performance over purely data-driven approaches. However, reconstruction quality still hinges on the network's ability to disentangle artifacts from true anatomical signals-both of which exhibit complex, multi-scale contextual structure. Convolutional neural networks (CNNs) capture local correlations but often struggle with non-local dependencies. While transformers aim to alleviate this limitation, practical implementations involve design compromises to reduce computational cost by balancing local and non-local sensitivity, occasionally resulting in performance comparable to CNNs. To address these challenges, we propose MambaRoll, a novel physics-driven autoregressive state space model (SSM) for high-fidelity and efficient image reconstruction. MambaRoll employs an unrolled architecture where each cascade autoregressively predicts finer-scale feature maps conditioned on coarser-scale representations, enabling consistent multi-scale context propagation. Each stage is built on a hierarchy of scale-specific PD-SSM modules that capture spatial dependencies while enforcing data consistency through residual correction. To further improve scale-aware learning, we introduce a Deep Multi-Scale Decoding (DMSD) loss, which provides supervision at intermediate spatial scales in alignment with the autoregressive design. Demonstrations on accelerated MRI and sparse-view CT reconstructions show that MambaRoll consistently outperforms state-of-the-art CNN-, transformer-, and SSM-based methods.
>
---
#### [replaced 021] TrackID3x3: A Dataset and Algorithm for Multi-Player Tracking with Identification and Pose Estimation in 3x3 Basketball Full-court Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18282v2](http://arxiv.org/pdf/2503.18282v2)**

> **作者:** Kazuhiro Yamada; Li Yin; Qingrui Hu; Ning Ding; Shunsuke Iwashita; Jun Ichikawa; Kiwamu Kotani; Calvin Yeung; Keisuke Fujii
>
> **备注:** Accepted in MMSports'25
>
> **摘要:** Multi-object tracking, player identification, and pose estimation are fundamental components of sports analytics, essential for analyzing player movements, performance, and tactical strategies. However, existing datasets and methodologies primarily target mainstream team sports such as soccer and conventional 5-on-5 basketball, often overlooking scenarios involving fixed-camera setups commonly used at amateur levels, less mainstream sports, or datasets that explicitly incorporate pose annotations. In this paper, we propose the TrackID3x3 dataset, the first publicly available comprehensive dataset specifically designed for multi-player tracking, player identification, and pose estimation in 3x3 basketball scenarios. The dataset comprises three distinct subsets (Indoor fixed-camera, Outdoor fixed-camera, and Drone camera footage), capturing diverse full-court camera perspectives and environments. We also introduce the Track-ID task, a simplified variant of the game state reconstruction task that excludes field detection and focuses exclusively on fixed-camera scenarios. To evaluate performance, we propose a baseline algorithm called Track-ID algorithm, tailored to assess tracking and identification quality. Furthermore, our benchmark experiments, utilizing recent multi-object tracking algorithms (e.g., BoT-SORT-ReID) and top-down pose estimation methods (HRNet, RTMPose, and SwinPose), demonstrate robust results and highlight remaining challenges. Our dataset and evaluation benchmarks provide a solid foundation for advancing automated analytics in 3x3 basketball. Dataset and code will be available at https://github.com/open-starlab/TrackID3x3.
>
---
#### [replaced 022] Towards Comprehensive Cellular Characterisation of H&E slides
- **分类: cs.CV; q-bio.QM; I.2.10; I.4.8**

- **链接: [http://arxiv.org/pdf/2508.09926v2](http://arxiv.org/pdf/2508.09926v2)**

> **作者:** Benjamin Adjadj; Pierre-Antoine Bannier; Guillaume Horent; Sebastien Mandela; Aurore Lyon; Kathryn Schutte; Ulysse Marteau; Valentin Gaury; Laura Dumont; Thomas Mathieu; Reda Belbahri; Benoît Schmauch; Eric Durand; Katharina Von Loga; Lucie Gillet
>
> **备注:** 25 pages, 4 figures
>
> **摘要:** Cell detection, segmentation and classification are essential for analyzing tumor microenvironments (TME) on hematoxylin and eosin (H&E) slides. Existing methods suffer from poor performance on understudied cell types (rare or not present in public datasets) and limited cross-domain generalization. To address these shortcomings, we introduce HistoPLUS, a state-of-the-art model for cell analysis, trained on a novel curated pan-cancer dataset of 108,722 nuclei covering 13 cell types. In external validation across 4 independent cohorts, HistoPLUS outperforms current state-of-the-art models in detection quality by 5.2% and overall F1 classification score by 23.7%, while using 5x fewer parameters. Notably, HistoPLUS unlocks the study of 7 understudied cell types and brings significant improvements on 8 of 13 cell types. Moreover, we show that HistoPLUS robustly transfers to two oncology indications unseen during training. To support broader TME biomarker research, we release the model weights and inference code at https://github.com/owkin/histoplus/.
>
---
#### [replaced 023] Revisiting Out-of-Distribution Detection in Real-time Object Detection: From Benchmark Pitfalls to a New Mitigation Paradigm
- **分类: cs.CV; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2503.07330v3](http://arxiv.org/pdf/2503.07330v3)**

> **作者:** Changshun Wu; Weicheng He; Chih-Hong Cheng; Xiaowei Huang; Saddek Bensalem
>
> **备注:** Expanded journal version of our IROS 2025 paper, adding automated OoD benchmarking, generalization to multiple object detectors, few-shot fine-tuning, and in-depth analysis
>
> **摘要:** Out-of-distribution (OoD) inputs pose a persistent challenge to deep learning models, often triggering overconfident predictions on non-target objects. While prior work has primarily focused on refining scoring functions and adjusting test-time thresholds, such algorithmic improvements offer only incremental gains. We argue that a rethinking of the entire development lifecycle is needed to mitigate these risks effectively. This work addresses two overlooked dimensions of OoD detection in object detection. First, we reveal fundamental flaws in widely used evaluation benchmarks: contrary to their design intent, up to 13% of objects in the OoD test sets actually belong to in-distribution classes, and vice versa. These quality issues severely distort the reported performance of existing methods and contribute to their high false positive rates. Second, we introduce a novel training-time mitigation paradigm that operates independently of external OoD detectors. Instead of relying solely on post-hoc scoring, we fine-tune the detector using a carefully synthesized OoD dataset that semantically resembles in-distribution objects. This process shapes a defensive decision boundary by suppressing objectness on OoD objects, leading to a 91% reduction in hallucination error of a YOLO model on BDD-100K. Our methodology generalizes across detection paradigms such as YOLO, Faster R-CNN, and RT-DETR, and supports few-shot adaptation. Together, these contributions offer a principled and effective way to reduce OoD-induced hallucination in object detectors. Code and data are available at: https://gricad-gitlab.univ-grenoble-alpes.fr/dnn-safety/m-hood.
>
---
#### [replaced 024] Statistical analysis of multivariate planar curves and applications to X-ray classification
- **分类: stat.ME; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2508.11780v2](http://arxiv.org/pdf/2508.11780v2)**

> **作者:** Issam-Ali Moindjié; Marie-Hélène Descary; Cédric Beaulac
>
> **摘要:** Recent developments in computer vision have enabled the availability of segmented images across various domains, such as medicine, where segmented radiography images play an important role in diagnosis-making. As prediction problems are common in medical image analysis, this work explores the use of segmented images (through the associated contours they highlight) as predictors in a supervised classification context. Consequently, we develop a new approach for image analysis that takes into account the shape of objects within images. For this aim, we introduce a new formalism that extends the study of single random planar curves to the joint analysis of multiple planar curves-referred to here as multivariate planar curves. In this framework, we propose a solution to the alignment issue in statistical shape analysis. The obtained multivariate shape variables are then used in functional classification methods through tangent projections. Detection of cardiomegaly in segmented X-rays and numerical experiments on synthetic data demonstrate the appeal and robustness of the proposed method.
>
---
#### [replaced 025] Omni-Video: Democratizing Unified Video Understanding and Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06119v3](http://arxiv.org/pdf/2507.06119v3)**

> **作者:** Zhiyu Tan; Hao Yang; Luozheng Qin; Jia Gong; Mengping Yang; Hao Li
>
> **备注:** Technical report, project page: https://howellyoung-s.github.io/OmniVideo_project/
>
> **摘要:** Notable breakthroughs in unified understanding and generation modeling have led to remarkable advancements in image understanding, reasoning, production and editing, yet current foundational models predominantly focus on processing images, creating a gap in the development of unified models for video understanding and generation. This report presents Omni-Video, an efficient and effective unified framework for video understanding, generation, as well as instruction-based editing. Our key insight is to teach existing multimodal large language models (MLLMs) to produce continuous visual clues that are used as the input of diffusion decoders, which produce high-quality videos conditioned on these visual clues. To fully unlock the potential of our system for unified video modeling, we integrate several technical improvements: 1) a lightweight architectural design that respectively attaches a vision head on the top of MLLMs and a adapter before the input of diffusion decoders, the former produce visual tokens for the latter, which adapts these visual tokens to the conditional space of diffusion decoders; and 2) an efficient multi-stage training scheme that facilitates a fast connection between MLLMs and diffusion decoders with limited data and computational resources. We empirically demonstrate that our model exhibits satisfactory generalization abilities across video generation, editing and understanding tasks.
>
---
#### [replaced 026] Diffusion MRI with Machine Learning
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.00019v5](http://arxiv.org/pdf/2402.00019v5)**

> **作者:** Davood Karimi; Simon K. Warfield
>
> **摘要:** \hspace{2mm} Diffusion-weighted magnetic resonance imaging (dMRI) of the brain offers unique capabilities including noninvasive probing of tissue microstructure and structural connectivity. It is widely used for clinical assessment of disease and injury, and for neuroscience research. Analyzing the dMRI data to extract useful information for medical and scientific purposes can be challenging. The dMRI measurements may suffer from strong noise and artifacts, and may exhibit high inter-session and inter-scanner variability in the data, as well as inter-subject heterogeneity in brain structure. Moreover, the relationship between measurements and the phenomena of interest can be highly complex. Recent years have witnessed increasing use of machine learning methods for dMRI analysis. This manuscript aims to assess these efforts, with a focus on methods that have addressed data preprocessing and harmonization, microstructure mapping, tractography, and white matter tract analysis. We study the main findings, strengths, and weaknesses of the existing methods and suggest topics for future research. We find that machine learning may be exceptionally suited to tackle some of the difficult tasks in dMRI analysis. However, for this to happen, several shortcomings of existing methods and critical unresolved issues need to be addressed. There is a pressing need to improve evaluation practices, to increase the availability of rich training datasets and validation benchmarks, as well as model generalizability, reliability, and explainability concerns.
>
---
#### [replaced 027] MaskSDM with Shapley values to improve flexibility, robustness, and explainability in species distribution modeling
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13057v2](http://arxiv.org/pdf/2503.13057v2)**

> **作者:** Robin Zbinden; Nina van Tiel; Gencer Sumbul; Chiara Vanalli; Benjamin Kellenberger; Devis Tuia
>
> **摘要:** Species Distribution Models (SDMs) play a vital role in biodiversity research, conservation planning, and ecological niche modeling by predicting species distributions based on environmental conditions. The selection of predictors is crucial, strongly impacting both model accuracy and how well the predictions reflect ecological patterns. To ensure meaningful insights, input variables must be carefully chosen to match the study objectives and the ecological requirements of the target species. However, existing SDMs, including both traditional and deep learning-based approaches, often lack key capabilities for variable selection: (i) flexibility to choose relevant predictors at inference without retraining; (ii) robustness to handle missing predictor values without compromising accuracy; and (iii) explainability to interpret and accurately quantify each predictor's contribution. To overcome these limitations, we introduce MaskSDM, a novel deep learning-based SDM that enables flexible predictor selection by employing a masked training strategy. This approach allows the model to make predictions with arbitrary subsets of input variables while remaining robust to missing data. It also provides a clearer understanding of how adding or removing a given predictor affects model performance and predictions. Additionally, MaskSDM leverages Shapley values for precise predictor contribution assessments, improving upon traditional approximations. We evaluate MaskSDM on the global sPlotOpen dataset, modeling the distributions of 12,738 plant species. Our results show that MaskSDM outperforms imputation-based methods and approximates models trained on specific subsets of variables. These findings underscore MaskSDM's potential to increase the applicability and adoption of SDMs, laying the groundwork for developing foundation models in SDMs that can be readily applied to diverse ecological applications.
>
---
#### [replaced 028] ABC: Achieving Better Control of Multimodal Embeddings using VLMs
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.00329v2](http://arxiv.org/pdf/2503.00329v2)**

> **作者:** Benjamin Schneider; Florian Kerschbaum; Wenhu Chen
>
> **备注:** TMLR 2025
>
> **摘要:** Visual embedding models excel at zero-shot tasks like visual retrieval and classification. However, these models cannot be used for tasks that contain ambiguity or require user instruction. These tasks necessitate an embedding model which outputs can use a natural language instruction to control the representation of a visual embedding. Existing CLIP-based approaches embed images and text independently, and fuse the result. We find that this results in weak interactions between modalities, and poor user control over the representation. We introduce ABC, an open-source multimodal embedding model that uses a vision-language model backbone to deeply integrate image features with natural language instructions. ABC achieves best-for-size performance on MSCOCO image-to-text retrieval and is the top performing model on classification and VQA tasks in the Massive Multimodal Embedding Benchmark. With a strongly unified vision-language representation, ABC can use natural language to solve subtle and potentially ambiguous visual retrieval problems. To evaluate this capability, we design CtrlBench, a benchmark that requires interleaving textual instructions with image content for correct retrieval. ABC advances the state of visual embeddings, outputting high-quality visual representations with natural language control. Our model and datasets are available at our project page: https://tiger-ai-lab.github.io/ABC/
>
---
#### [replaced 029] BoostTrack++: using tracklet information to detect more objects in multiple object tracking
- **分类: cs.CV; cs.AI; 68T20 (Primary) 68T45, 68U10 (Secondary); F.2.2; I.4.8**

- **链接: [http://arxiv.org/pdf/2408.13003v2](http://arxiv.org/pdf/2408.13003v2)**

> **作者:** Vukašin Stanojević; Branimir Todorović
>
> **备注:** To be published in Filomat, Vol 39, No 16 (2025)
>
> **摘要:** Multiple object tracking (MOT) depends heavily on selection of true positive detected bounding boxes. However, this aspect of the problem is mostly overlooked or mitigated by employing two-stage association and utilizing low confidence detections in the second stage. Recently proposed BoostTrack attempts to avoid the drawbacks of multiple stage association approach and use low-confidence detections by applying detection confidence boosting. In this paper, we identify the limitations of the confidence boost used in BoostTrack and propose a method to improve its performance. To construct a richer similarity measure and enable a better selection of true positive detections, we propose to use a combination of shape, Mahalanobis distance and novel soft BIoU similarity. We propose a soft detection confidence boost technique which calculates new confidence scores based on the similarity measure and the previous confidence scores, and we introduce varying similarity threshold to account for lower similarity measure between detections and tracklets which are not regularly updated. The proposed additions are mutually independent and can be used in any MOT algorithm. Combined with the BoostTrack+ baseline, our method achieves near state of the art results on the MOT17 dataset and new state of the art HOTA and IDF1 scores on the MOT20 dataset. The source code is available at: https://github.com/vukasin-stanojevic/BoostTrack .
>
---
#### [replaced 030] AURA: A Fine-Grained Benchmark and Decomposed Metric for Audio-Visual Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.07470v2](http://arxiv.org/pdf/2508.07470v2)**

> **作者:** Siminfar Samakoush Galougah; Rishie Raj; Sanjoy Chowdhury; Sayan Nag; Ramani Duraiswami
>
> **摘要:** Current audio-visual (AV) benchmarks focus on final answer accuracy, overlooking the underlying reasoning process. This makes it difficult to distinguish genuine comprehension from correct answers derived through flawed reasoning or hallucinations. To address this, we introduce AURA (Audio-visual Understanding and Reasoning Assessment), a benchmark for evaluating the cross-modal reasoning capabilities of Audio-Visual Large Language Models (AV-LLMs) and Omni-modal Language Models (OLMs). AURA includes questions across six challenging cognitive domains, such as causality, timbre and pitch, tempo and AV synchronization, unanswerability, implicit distractions, and skill profiling, explicitly designed to be unanswerable from a single modality. This forces models to construct a valid logical path grounded in both audio and video, setting AURA apart from AV datasets that allow uni-modal shortcuts. To assess reasoning traces, we propose a novel metric, AuraScore, which addresses the lack of robust tools for evaluating reasoning fidelity. It decomposes reasoning into two aspects: (i) Factual Consistency - whether reasoning is grounded in perceptual evidence, and (ii) Core Inference - the logical validity of each reasoning step. Evaluations of SOTA models on AURA reveal a critical reasoning gap: although models achieve high accuracy (up to 92% on some tasks), their Factual Consistency and Core Inference scores fall below 45%. This discrepancy highlights that models often arrive at correct answers through flawed logic, underscoring the need for our benchmark and paving the way for more robust multimodal evaluation.
>
---
#### [replaced 031] UAV-ON: A Benchmark for Open-World Object Goal Navigation with Aerial Agents
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.00288v3](http://arxiv.org/pdf/2508.00288v3)**

> **作者:** Jianqiang Xiao; Yuexuan Sun; Yixin Shao; Boxi Gan; Rongqiang Liu; Yanjing Wu; Weili Gua; Xiang Deng
>
> **备注:** Accepted to ACM MM Dataset Track 2025
>
> **摘要:** Aerial navigation is a fundamental yet underexplored capability in embodied intelligence, enabling agents to operate in large-scale, unstructured environments where traditional navigation paradigms fall short. However, most existing research follows the Vision-and-Language Navigation (VLN) paradigm, which heavily depends on sequential linguistic instructions, limiting its scalability and autonomy. To address this gap, we introduce UAV-ON, a benchmark for large-scale Object Goal Navigation (ObjectNav) by aerial agents in open-world environments, where agents operate based on high-level semantic goals without relying on detailed instructional guidance as in VLN. UAV-ON comprises 14 high-fidelity Unreal Engine environments with diverse semantic regions and complex spatial layouts, covering urban, natural, and mixed-use settings. It defines 1270 annotated target objects, each characterized by an instance-level instruction that encodes category, physical footprint, and visual descriptors, allowing grounded reasoning. These instructions serve as semantic goals, introducing realistic ambiguity and complex reasoning challenges for aerial agents. To evaluate the benchmark, we implement several baseline methods, including Aerial ObjectNav Agent (AOA), a modular policy that integrates instruction semantics with egocentric observations for long-horizon, goal-directed exploration. Empirical results show that all baselines struggle in this setting, highlighting the compounded challenges of aerial navigation and semantic goal grounding. UAV-ON aims to advance research on scalable UAV autonomy driven by semantic goal descriptions in complex real-world environments.
>
---
#### [replaced 032] Understanding Co-speech Gestures in-the-wild
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22668v2](http://arxiv.org/pdf/2503.22668v2)**

> **作者:** Sindhu B Hegde; K R Prajwal; Taein Kwon; Andrew Zisserman
>
> **备注:** Main paper - 11 pages, 4 figures, Supplementary - 6 pages, 6 figures
>
> **摘要:** Co-speech gestures play a vital role in non-verbal communication. In this paper, we introduce a new framework for co-speech gesture understanding in the wild. Specifically, we propose three new tasks and benchmarks to evaluate a model's capability to comprehend gesture-speech-text associations: (i) gesture based retrieval, (ii) gesture word spotting, and (iii) active speaker detection using gestures. We present a new approach that learns a tri-modal video-gesture-speech-text representation to solve these tasks. By leveraging a combination of global phrase contrastive loss and local gesture-word coupling loss, we demonstrate that a strong gesture representation can be learned in a weakly supervised manner from videos in the wild. Our learned representations outperform previous methods, including large vision-language models (VLMs). Further analysis reveals that speech and text modalities capture distinct gesture related signals, underscoring the advantages of learning a shared tri-modal embedding space. The dataset, model, and code are available at: https://www.robots.ox.ac.uk/~vgg/research/jegal.
>
---
#### [replaced 033] Cross-Modality Masked Learning for Survival Prediction in ICI Treated NSCLC Patients
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.06994v2](http://arxiv.org/pdf/2507.06994v2)**

> **作者:** Qilong Xing; Zikai Song; Bingxin Gong; Lian Yang; Junqing Yu; Wei Yang
>
> **备注:** MICCAI 2025
>
> **摘要:** Accurate prognosis of non-small cell lung cancer (NSCLC) patients undergoing immunotherapy is essential for personalized treatment planning, enabling informed patient decisions, and improving both treatment outcomes and quality of life. However, the lack of large, relevant datasets and effective multi-modal feature fusion strategies pose significant challenges in this domain. To address these challenges, we present a large-scale dataset and introduce a novel framework for multi-modal feature fusion aimed at enhancing the accuracy of survival prediction. The dataset comprises 3D CT images and corresponding clinical records from NSCLC patients treated with immune checkpoint inhibitors (ICI), along with progression-free survival (PFS) and overall survival (OS) data. We further propose a cross-modality masked learning approach for medical feature fusion, consisting of two distinct branches, each tailored to its respective modality: a Slice-Depth Transformer for extracting 3D features from CT images and a graph-based Transformer for learning node features and relationships among clinical variables in tabular data. The fusion process is guided by a masked modality learning strategy, wherein the model utilizes the intact modality to reconstruct missing components. This mechanism improves the integration of modality-specific features, fostering more effective inter-modality relationships and feature interactions. Our approach demonstrates superior performance in multi-modal integration for NSCLC survival prediction, surpassing existing methods and setting a new benchmark for prognostic models in this context.
>
---
#### [replaced 034] Omni$^2$: Unifying Omnidirectional Image Generation and Editing in an Omni Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.11379v2](http://arxiv.org/pdf/2504.11379v2)**

> **作者:** Liu Yang; Huiyu Duan; Yucheng Zhu; Xiaohong Liu; Lu Liu; Zitong Xu; Guangji Ma; Xiongkuo Min; Guangtao Zhai; Patrick Le Callet
>
> **备注:** 19 pages
>
> **摘要:** $360^{\circ}$ omnidirectional images (ODIs) have gained considerable attention recently, and are widely used in various virtual reality (VR) and augmented reality (AR) applications. However, capturing such images is expensive and requires specialized equipment, making ODI synthesis increasingly important. While common 2D image generation and editing methods are rapidly advancing, these models struggle to deliver satisfactory results when generating or editing ODIs due to the unique format and broad 360$^{\circ}$ Field-of-View (FoV) of ODIs. To bridge this gap, we construct \textbf{\textit{Any2Omni}}, the first comprehensive ODI generation-editing dataset comprises 60,000+ training data covering diverse input conditions and up to 9 ODI generation and editing tasks. Built upon Any2Omni, we propose an \textbf{\underline{Omni}} model for \textbf{\underline{Omni}}-directional image generation and editing (\textbf{\textit{Omni$^2$}}), with the capability of handling various ODI generation and editing tasks under diverse input conditions using one model. Extensive experiments demonstrate the superiority and effectiveness of the proposed Omni$^2$ model for both the ODI generation and editing tasks. Both the Any2Omni dataset and the Omni$^2$ model are publicly available at: https://github.com/IntMeGroup/Omni2.
>
---
#### [replaced 035] Translating Images to Road Network: A Sequence-to-Sequence Perspective
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2402.08207v3](http://arxiv.org/pdf/2402.08207v3)**

> **作者:** Jiachen Lu; Ming Nie; Bozhou Zhang; Reyuan Peng; Xinyue Cai; Hang Xu; Feng Wen; Wei Zhang; Li Zhang
>
> **备注:** V1 is the ICCV 2023 conference version, and V2 is the extended version
>
> **摘要:** The extraction of road network is essential for the generation of high-definition maps since it enables the precise localization of road landmarks and their interconnections. However, generating road network poses a significant challenge due to the conflicting underlying combination of Euclidean (e.g., road landmarks location) and non-Euclidean (e.g., road topological connectivity) structures. Existing methods struggle to merge the two types of data domains effectively, but few of them address it properly. Instead, our work establishes a unified representation of both types of data domain by projecting both Euclidean and non-Euclidean data into an integer series called RoadNet Sequence. Further than modeling an auto-regressive sequence-to-sequence Transformer model to understand RoadNet Sequence, we decouple the dependency of RoadNet Sequence into a mixture of auto-regressive and non-autoregressive dependency. Building on this, our proposed non-autoregressive sequence-to-sequence approach leverages non-autoregressive dependencies while fixing the gap towards auto-regressive dependencies, resulting in success in both efficiency and accuracy. We further identify two main bottlenecks in the current RoadNetTransformer on a non-overfitting split of the dataset: poor landmark detection limited by the BEV Encoder and error propagation to topology reasoning. Therefore, we propose Topology-Inherited Training to inherit better topology knowledge into RoadNetTransformer. Additionally, we collect SD-Maps from open-source map datasets and use this prior information to significantly improve landmark detection and reachability. Extensive experiments on the nuScenes dataset demonstrate the superiority of RoadNet Sequence representation and the non-autoregressive approach compared to existing state-of-the-art alternatives.
>
---
#### [replaced 036] Latent Interpolation Learning Using Diffusion Models for Cardiac Volume Reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.13826v3](http://arxiv.org/pdf/2508.13826v3)**

> **作者:** Niklas Bubeck; Suprosanna Shit; Chen Chen; Can Zhao; Pengfei Guo; Dong Yang; Georg Zitzlsberger; Daguang Xu; Bernhard Kainz; Daniel Rueckert; Jiazhen Pan
>
> **摘要:** Cardiac Magnetic Resonance (CMR) imaging is a critical tool for diagnosing and managing cardiovascular disease, yet its utility is often limited by the sparse acquisition of 2D short-axis slices, resulting in incomplete volumetric information. Accurate 3D reconstruction from these sparse slices is essential for comprehensive cardiac assessment, but existing methods face challenges, including reliance on predefined interpolation schemes (e.g., linear or spherical), computational inefficiency, and dependence on additional semantic inputs such as segmentation labels or motion data. To address these limitations, we propose a novel Cardiac Latent Interpolation Diffusion (CaLID) framework that introduces three key innovations. First, we present a data-driven interpolation scheme based on diffusion models, which can capture complex, non-linear relationships between sparse slices and improves reconstruction accuracy. Second, we design a computationally efficient method that operates in the latent space and speeds up 3D whole-heart upsampling time by a factor of 24, reducing computational overhead compared to previous methods. Third, with only sparse 2D CMR images as input, our method achieves SOTA performance against baseline methods, eliminating the need for auxiliary input such as morphological guidance, thus simplifying workflows. We further extend our method to 2D+T data, enabling the effective modeling of spatiotemporal dynamics and ensuring temporal coherence. Extensive volumetric evaluations and downstream segmentation tasks demonstrate that CaLID achieves superior reconstruction quality and efficiency. By addressing the fundamental limitations of existing approaches, our framework advances the state of the art for spatio and spatiotemporal whole-heart reconstruction, offering a robust and clinically practical solution for cardiovascular imaging.
>
---
#### [replaced 037] TripleMixer: A 3D Point Cloud Denoising Model for Adverse Weather
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2408.13802v2](http://arxiv.org/pdf/2408.13802v2)**

> **作者:** Xiongwei Zhao; Congcong Wen; Xu Zhu; Yang Wang; Haojie Bai; Wenhao Dou
>
> **备注:** 15 pages, submit to IEEE TIP
>
> **摘要:** Adverse weather conditions such as snow, fog, and rain pose significant challenges to LiDAR-based perception models by introducing noise and corrupting point cloud measurements. To address this issue, we propose TripleMixer, a robust and efficient point cloud denoising network that integrates spatial, frequency, and channel-wise processing through three specialized mixer modules. TripleMixer effectively suppresses high-frequency noise while preserving essential geometric structures and can be seamlessly deployed as a plug-and-play module within existing LiDAR perception pipelines. To support the development and evaluation of denoising methods, we construct two large-scale simulated datasets, Weather-KITTI and Weather-NuScenes, covering diverse weather scenarios with dense point-wise semantic and noise annotations. Based on these datasets, we establish four benchmarks: Denoising, Semantic Segmentation (SS), Place Recognition (PR), and Object Detection (OD). These benchmarks enable systematic evaluation of denoising generalization, transferability, and downstream impact under both simulated and real-world adverse weather conditions. Extensive experiments demonstrate that TripleMixer achieves state-of-the-art denoising performance and yields substantial improvements across all downstream tasks without requiring retraining. Our results highlight the potential of denoising as a task-agnostic preprocessing strategy to enhance LiDAR robustness in real-world autonomous driving applications.
>
---
#### [replaced 038] Toward Errorless Training ImageNet-1k
- **分类: cs.CV; cs.LG; 68T07**

- **链接: [http://arxiv.org/pdf/2508.04941v4](http://arxiv.org/pdf/2508.04941v4)**

> **作者:** Bo Deng; Levi Heath
>
> **备注:** 14 pages, 2 figures, 5 tables
>
> **摘要:** In this paper, we describe a feedforward artificial neural network trained on the ImageNet 2012 contest dataset [7] with the new method of [5] to an accuracy rate of 98.3% with a 99.69 Top-1 rate, and an average of 285.9 labels that are perfectly classified over the 10 batch partitions of the dataset. The best performing model uses 322,430,160 parameters, with 4 decimal places precision. We conjecture that the reason our model does not achieve a 100% accuracy rate is due to a double-labeling problem, by which there are duplicate images in the dataset with different labels.
>
---
#### [replaced 039] Hadamard Attention Recurrent Transformer: A Strong Baseline for Stereo Matching Transformer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.01023v4](http://arxiv.org/pdf/2501.01023v4)**

> **作者:** Ziyang Chen; Wenting Li; Yongjun Zhang; Yabo Wu; Bingshu Wang; Yong Zhao; C. L. Philip Chen
>
> **摘要:** Constrained by the low-rank bottleneck inherent in attention mechanisms, current stereo matching transformers suffer from limited nonlinear expressivity, which renders their feature representations sensitive to challenging conditions such as reflections. To overcome this difficulty, we present the Hadamard Attention Recurrent Stereo Transformer (HART). HART includes a novel attention mechanism that incorporates the following components: 1) The Dense Attention Kernel (DAK) maps the attention weight distribution into a high-dimensional space over (0, +$\infty$). By removing the upper bound constraint on attention weights, DAK enables more flexible modeling of complex feature interactions. This reduces feature collinearity. 2) The Multi Kernel & Order Interaction (MKOI) module extends the attention mechanism by unifying semantic and spatial knowledge learning. This integration improves the ability of HART to learn features in binocular images. Experimental results demonstrate the effectiveness of our HART. In reflective area, HART ranked 1st on the KITTI 2012 benchmark among all published methods at the time of submission. Code is available at https://github.com/ZYangChen/HART.
>
---
#### [replaced 040] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.12892v2](http://arxiv.org/pdf/2409.12892v2)**

> **作者:** Lukas Höllein; Aljaž Božič; Michael Zollhöfer; Matthias Nießner
>
> **备注:** Accepted to ICCV 2025. Project page: https://lukashoel.github.io/3DGS-LM, Video: https://www.youtube.com/watch?v=tDiGuGMssg8, Code: https://github.com/lukasHoel/3DGS-LM
>
> **摘要:** We present 3DGS-LM, a new method that accelerates the reconstruction of 3D Gaussian Splatting (3DGS) by replacing its ADAM optimizer with a tailored Levenberg-Marquardt (LM). Existing methods reduce the optimization time by decreasing the number of Gaussians or by improving the implementation of the differentiable rasterizer. However, they still rely on the ADAM optimizer to fit Gaussian parameters of a scene in thousands of iterations, which can take up to an hour. To this end, we change the optimizer to LM that runs in conjunction with the 3DGS differentiable rasterizer. For efficient GPU parallization, we propose a caching data structure for intermediate gradients that allows us to efficiently calculate Jacobian-vector products in custom CUDA kernels. In every LM iteration, we calculate update directions from multiple image subsets using these kernels and combine them in a weighted mean. Overall, our method is 20% faster than the original 3DGS while obtaining the same reconstruction quality. Our optimization is also agnostic to other methods that acclerate 3DGS, thus enabling even faster speedups compared to vanilla 3DGS.
>
---
#### [replaced 041] Vision Transformers for Kidney Stone Image Classification: A Comparative Study with CNNs
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.13461v2](http://arxiv.org/pdf/2508.13461v2)**

> **作者:** Ivan Reyes-Amezcua; Francisco Lopez-Tiro; Clement Larose; Andres Mendez-Vazquez; Gilberto Ochoa-Ruiz; Christian Daul
>
> **摘要:** Kidney stone classification from endoscopic images is critical for personalized treatment and recurrence prevention. While convolutional neural networks (CNNs) have shown promise in this task, their limited ability to capture long-range dependencies can hinder performance under variable imaging conditions. This study presents a comparative analysis between Vision Transformers (ViTs) and CNN-based models, evaluating their performance on two ex vivo datasets comprising CCD camera and flexible ureteroscope images. The ViT-base model pretrained on ImageNet-21k consistently outperformed a ResNet50 baseline across multiple imaging conditions. For instance, in the most visually complex subset (Section patches from endoscopic images), the ViT model achieved 95.2% accuracy and 95.1% F1-score, compared to 64.5% and 59.3% with ResNet50. In the mixed-view subset from CCD-camera images, ViT reached 87.1% accuracy versus 78.4% with CNN. These improvements extend across precision and recall as well. The results demonstrate that ViT-based architectures provide superior classification performance and offer a scalable alternative to conventional CNNs for kidney stone image analysis.
>
---
#### [replaced 042] GALA: Guided Attention with Language Alignment for Open Vocabulary Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.14278v2](http://arxiv.org/pdf/2508.14278v2)**

> **作者:** Elena Alegret; Kunyi Li; Sen Wang; Siyun Liang; Michael Niemeyer; Stefano Gasperini; Nassir Navab; Federico Tombari
>
> **摘要:** 3D scene reconstruction and understanding have gained increasing popularity, yet existing methods still struggle to capture fine-grained, language-aware 3D representations from 2D images. In this paper, we present GALA, a novel framework for open-vocabulary 3D scene understanding with 3D Gaussian Splatting (3DGS). GALA distills a scene-specific 3D instance feature field via self-supervised contrastive learning. To extend to generalized language feature fields, we introduce the core contribution of GALA, a cross-attention module with two learnable codebooks that encode view-independent semantic embeddings. This design not only ensures intra-instance feature similarity but also supports seamless 2D and 3D open-vocabulary queries. It reduces memory consumption by avoiding per-Gaussian high-dimensional feature learning. Extensive experiments on real-world datasets demonstrate GALA's remarkable open-vocabulary performance on both 2D and 3D.
>
---
#### [replaced 043] TiP4GEN: Text to Immersive Panorama 4D Scene Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.12415v2](http://arxiv.org/pdf/2508.12415v2)**

> **作者:** Ke Xing; Hanwen Liang; Dejia Xu; Yuyang Yin; Konstantinos N. Plataniotis; Yao Zhao; Yunchao Wei
>
> **备注:** Accepted In Proceedings of the 33rd ACM International Conference on Multimedia (MM' 25)
>
> **摘要:** With the rapid advancement and widespread adoption of VR/AR technologies, there is a growing demand for the creation of high-quality, immersive dynamic scenes. However, existing generation works predominantly concentrate on the creation of static scenes or narrow perspective-view dynamic scenes, falling short of delivering a truly 360-degree immersive experience from any viewpoint. In this paper, we introduce \textbf{TiP4GEN}, an advanced text-to-dynamic panorama scene generation framework that enables fine-grained content control and synthesizes motion-rich, geometry-consistent panoramic 4D scenes. TiP4GEN integrates panorama video generation and dynamic scene reconstruction to create 360-degree immersive virtual environments. For video generation, we introduce a \textbf{Dual-branch Generation Model} consisting of a panorama branch and a perspective branch, responsible for global and local view generation, respectively. A bidirectional cross-attention mechanism facilitates comprehensive information exchange between the branches. For scene reconstruction, we propose a \textbf{Geometry-aligned Reconstruction Model} based on 3D Gaussian Splatting. By aligning spatial-temporal point clouds using metric depth maps and initializing scene cameras with estimated poses, our method ensures geometric consistency and temporal coherence for the reconstructed scenes. Extensive experiments demonstrate the effectiveness of our proposed designs and the superiority of TiP4GEN in generating visually compelling and motion-coherent dynamic panoramic scenes. Our project page is at https://ke-xing.github.io/TiP4GEN/.
>
---
#### [replaced 044] ILeSiA: Interactive Learning of Robot Situational Awareness from Camera Input
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.20173v2](http://arxiv.org/pdf/2409.20173v2)**

> **作者:** Petr Vanc; Giovanni Franzese; Jan Kristof Behrens; Cosimo Della Santina; Karla Stepanova; Jens Kober; Robert Babuska
>
> **备注:** 8 pages, 9 figures. Accepted to IEEE Robotics and Automation Letters (Early Access)
>
> **摘要:** Learning from demonstration is a promising approach for teaching robots new skills. However, a central challenge in the execution of acquired skills is the ability to recognize faults and prevent failures. This is essential because demonstrations typically cover only a limited set of scenarios and often only the successful ones. During task execution, unforeseen situations may arise, such as changes in the robot's environment or interaction with human operators. To recognize such situations, this paper focuses on teaching the robot situational awareness by using a camera input and labeling frames as safe or risky. We train a Gaussian Process (GP) regression model fed by a low-dimensional latent space representation of the input images. The model outputs a continuous risk score ranging from zero to one, quantifying the degree of risk at each timestep. This allows for pausing task execution in unsafe situations and directly adding new training data, labeled by the human user. Our experiments on a robotic manipulator show that the proposed method can reliably detect both known and novel faults using only a single example for each new fault. In contrast, a standard multi-layer perceptron (MLP) performs well only on faults it has encountered during training. Our method enables the next generation of cobots to be rapidly deployed with easy-to-set-up, vision-based risk assessment, proactively safeguarding humans and detecting misaligned parts or missing objects before failures occur. We provide all the code and data required to reproduce our experiments at imitrob.ciirc.cvut.cz/publications/ilesia.
>
---
#### [replaced 045] Hybrid Autoregressive-Diffusion Model for Real-Time Streaming Sign Language Production
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09105v2](http://arxiv.org/pdf/2507.09105v2)**

> **作者:** Maoxiao Ye; Xinfeng Ye; Mano Manoharan
>
> **备注:** The authors have withdrawn this manuscript because the current version requires substantial revisions and is no longer suitable for posting
>
> **摘要:** Earlier Sign Language Production (SLP) models typically relied on autoregressive methods that generate output tokens one by one, which inherently provide temporal alignment. Although techniques like Teacher Forcing can prevent model collapse during training, they still cannot solve the problem of error accumulation during inference, since ground truth is unavailable at that stage. In contrast, more recent approaches based on diffusion models leverage step-by-step denoising to enable high-quality generation. However, the iterative nature of these models and the requirement to denoise entire sequences limit their applicability in real-time tasks like SLP. To address it, we apply a hybrid approach combining autoregressive and diffusion models to SLP for the first time, leveraging the strengths of both models in sequential dependency modeling and output refinement. To capture fine-grained body movements, we design a Multi-Scale Pose Representation module that separately extracts detailed features from distinct articulators and integrates them via a Multi-Scale Fusion module. Furthermore, we introduce a Confidence-Aware Causal Attention mechanism that utilizes joint-level confidence scores to dynamically guide the pose generation process, improving accuracy and robustness. Extensive experiments on the PHOENIX14T and How2Sign datasets demonstrate the effectiveness of our method in both generation quality and real-time streaming efficiency.
>
---
#### [replaced 046] MoCHA-former: Moiré-Conditioned Hybrid Adaptive Transformer for Video Demoiréing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.14423v2](http://arxiv.org/pdf/2508.14423v2)**

> **作者:** Jeahun Sung; Changhyun Roh; Chanho Eom; Jihyong Oh
>
> **备注:** Please visit our project page at [this http URL link](https://cmlab-korea.github.io/MoCHAformer-Demo/)
>
> **摘要:** Recent advances in portable imaging have made camera-based screen capture ubiquitous. Unfortunately, frequency aliasing between the camera's color filter array (CFA) and the display's sub-pixels induces moir\'e patterns that severely degrade captured photos and videos. Although various demoir\'eing models have been proposed to remove such moir\'e patterns, these approaches still suffer from several limitations: (i) spatially varying artifact strength within a frame, (ii) large-scale and globally spreading structures, (iii) channel-dependent statistics and (iv) rapid temporal fluctuations across frames. We address these issues with the Moir\'e Conditioned Hybrid Adaptive Transformer (MoCHA-former), which comprises two key components: Decoupled Moir\'e Adaptive Demoir\'eing (DMAD) and Spatio-Temporal Adaptive Demoir\'eing (STAD). DMAD separates moir\'e and content via a Moir\'e Decoupling Block (MDB) and a Detail Decoupling Block (DDB), then produces moir\'e-adaptive features using a Moir\'e Conditioning Block (MCB) for targeted restoration. STAD introduces a Spatial Fusion Block (SFB) with window attention to capture large-scale structures, and a Feature Channel Attention (FCA) to model channel dependence in RAW frames. To ensure temporal consistency, MoCHA-former performs implicit frame alignment without any explicit alignment module. We analyze moir\'e characteristics through qualitative and quantitative studies, and evaluate on two video datasets covering RAW and sRGB domains. MoCHA-former consistently surpasses prior methods across PSNR, SSIM, and LPIPS.
>
---
#### [replaced 047] Grounded-VideoLLM: Sharpening Fine-grained Temporal Grounding in Video Large Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.03290v2](http://arxiv.org/pdf/2410.03290v2)**

> **作者:** Haibo Wang; Zhiyang Xu; Yu Cheng; Shizhe Diao; Yufan Zhou; Yixin Cao; Qifan Wang; Weifeng Ge; Lifu Huang
>
> **备注:** Accepted by EMNLP 2025 Findings
>
> **摘要:** Video Large Language Models (Video-LLMs) have demonstrated remarkable capabilities in coarse-grained video understanding, however, they struggle with fine-grained temporal grounding. In this paper, we introduce Grounded-VideoLLM, a novel Video-LLM adept at perceiving and reasoning over specific video moments in a fine-grained manner. We identify that current Video-LLMs have limitations for fine-grained video understanding since they lack effective temporal modeling and timestamp representation. In light of this, we sharpen our model by incorporating (1) an additional temporal stream to encode the relationships between frames and (2) discrete temporal tokens enriched with specific time knowledge to represent timestamps. To optimize the training of Grounded-VideoLLM, we employ a multi-stage training scheme, beginning with simple video-captioning tasks and progressively introducing video temporal grounding tasks of increasing complexity. To further enhance Grounded-VideoLLM's temporal reasoning capability, we also curate a grounded VideoQA dataset by an automatic annotation pipeline. Extensive experiments demonstrate that Grounded-VideoLLM not only excels in fine-grained grounding tasks such as temporal sentence grounding, dense video captioning, and grounded VideoQA, but also shows great potential as a versatile video assistant for general video understanding.
>
---
#### [replaced 048] The Devil is in the EOS: Sequence Training for Detailed Image Captioning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20077v2](http://arxiv.org/pdf/2507.20077v2)**

> **作者:** Abdelrahman Mohamed; Yova Kementchedjhieva
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Despite significant advances in vision-language models (VLMs), image captioning often suffers from a lack of detail, with base models producing short, generic captions. This limitation persists even though VLMs are equipped with strong vision and language backbones. While supervised data and complex reward functions have been proposed to improve detailed image captioning, we identify a simpler underlying issue: a bias towards the end-of-sequence (EOS) token, which is introduced during cross-entropy training. We propose an unsupervised method to debias the model's tendency to predict the EOS token prematurely. By reducing this bias, we encourage the generation of longer, more detailed captions without the need for intricate reward functions or supervision. Our approach is straightforward, effective, and easily applicable to any pretrained model. We demonstrate its effectiveness through experiments with three VLMs and on three detailed captioning benchmarks. Our results show a substantial increase in caption length and relevant details, albeit with an expected increase in the rate of hallucinations.
>
---
#### [replaced 049] LaMP-Cap: Personalized Figure Caption Generation With Multimodal Figure Profiles
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.06561v3](http://arxiv.org/pdf/2506.06561v3)**

> **作者:** Ho Yin 'Sam' Ng; Ting-Yao Hsu; Aashish Anantha Ramakrishnan; Branislav Kveton; Nedim Lipka; Franck Dernoncourt; Dongwon Lee; Tong Yu; Sungchul Kim; Ryan A. Rossi; Ting-Hao 'Kenneth' Huang
>
> **备注:** Accepted to EMNLP 2025 Findings. The LaMP-CAP dataset is publicly available at: https://github.com/Crowd-AI-Lab/lamp-cap
>
> **摘要:** Figure captions are crucial for helping readers understand and remember a figure's key message. Many models have been developed to generate these captions, helping authors compose better quality captions more easily. Yet, authors almost always need to revise generic AI-generated captions to match their writing style and the domain's style, highlighting the need for personalization. Despite language models' personalization (LaMP) advances, these technologies often focus on text-only settings and rarely address scenarios where both inputs and profiles are multimodal. This paper introduces LaMP-Cap, a dataset for personalized figure caption generation with multimodal figure profiles. For each target figure, LaMP-Cap provides not only the needed inputs, such as figure images, but also up to three other figures from the same document--each with its image, caption, and figure-mentioning paragraphs--as a profile to characterize the context. Experiments with four LLMs show that using profile information consistently helps generate captions closer to the original author-written ones. Ablation studies reveal that images in the profile are more helpful than figure-mentioning paragraphs, highlighting the advantage of using multimodal profiles over text-only ones.
>
---
#### [replaced 050] MultiRef: Controllable Image Generation with Multiple Visual References
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.06905v2](http://arxiv.org/pdf/2508.06905v2)**

> **作者:** Ruoxi Chen; Dongping Chen; Siyuan Wu; Sinan Wang; Shiyun Lang; Petr Sushko; Gaoyang Jiang; Yao Wan; Ranjay Krishna
>
> **备注:** Accepted to ACM MM 2025 Datasets
>
> **摘要:** Visual designers naturally draw inspiration from multiple visual references, combining diverse elements and aesthetic principles to create artwork. However, current image generative frameworks predominantly rely on single-source inputs -- either text prompts or individual reference images. In this paper, we focus on the task of controllable image generation using multiple visual references. We introduce MultiRef-bench, a rigorous evaluation framework comprising 990 synthetic and 1,000 real-world samples that require incorporating visual content from multiple reference images. The synthetic samples are synthetically generated through our data engine RefBlend, with 10 reference types and 33 reference combinations. Based on RefBlend, we further construct a dataset MultiRef containing 38k high-quality images to facilitate further research. Our experiments across three interleaved image-text models (i.e., OmniGen, ACE, and Show-o) and six agentic frameworks (e.g., ChatDiT and LLM + SD) reveal that even state-of-the-art systems struggle with multi-reference conditioning, with the best model OmniGen achieving only 66.6% in synthetic samples and 79.0% in real-world cases on average compared to the golden answer. These findings provide valuable directions for developing more flexible and human-like creative tools that can effectively integrate multiple sources of visual inspiration. The dataset is publicly available at: https://multiref.github.io/.
>
---
#### [replaced 051] AlphaDent: A dataset for automated tooth pathology detection
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.22512v2](http://arxiv.org/pdf/2507.22512v2)**

> **作者:** Evgeniy I. Sosnin; Yuriy L. Vasilev; Roman A. Solovyev; Aleksandr L. Stempkovskiy; Dmitry V. Telpukhov; Artem A. Vasilev; Aleksandr A. Amerikanov; Aleksandr Y. Romanov
>
> **摘要:** In this article, we present a new unique dataset for dental research - AlphaDent. This dataset is based on the DSLR camera photographs of the teeth of 295 patients and contains over 1200 images. The dataset is labeled for solving the instance segmentation problem and is divided into 9 classes. The article provides a detailed description of the dataset and the labeling format. The article also provides the details of the experiment on neural network training for the Instance Segmentation problem using this dataset. The results obtained show high quality of predictions. The dataset is published under an open license; and the training/inference code and model weights are also available under open licenses.
>
---
#### [replaced 052] Parallel transport on matrix manifolds and Exponential Action
- **分类: math.NA; cs.CV; cs.NA; 15A16, 15A18, 15B10, 22E70, 51F25, 53C80, 53Z99**

- **链接: [http://arxiv.org/pdf/2408.06054v2](http://arxiv.org/pdf/2408.06054v2)**

> **作者:** Du Nguyen; Stefan Sommer
>
> **摘要:** We express parallel transport for several common matrix Lie groups with a family of pseudo-Riemannian metrics in terms of matrix exponential and exponential actions. The metrics are constructed from a deformation of a bi-invariant metric and are naturally reductive. There is a similar picture for homogeneous spaces when taking quotients satisfying a general condition. In particular, for a Stiefel manifold of orthogonal matrices of size $n\times d$, we give an expression for parallel transport along a geodesic from time zero to $t$, that could be computed with time complexity of $O(n d^2)$ for small $t$, and of $O(td^3)$ for large $t$, contributing a step in a long-standing open problem in matrix manifolds. A similar result holds for {\it flag manifolds} with the canonical metric. We also show the parallel transport formulas for the {\it general linear group} and the {\it special orthogonal group} under these metrics.
>
---
#### [replaced 053] Neuro Symbolic Knowledge Reasoning for Procedural Video Question Answering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.14957v2](http://arxiv.org/pdf/2503.14957v2)**

> **作者:** Thanh-Son Nguyen; Hong Yang; Tzeh Yuan Neoh; Hao Zhang; Ee Yeo Keat; Basura Fernando
>
> **摘要:** We introduce \dataset (Procedural Knowledge Reasoning Question Answering), a new benchmark for question answering over procedural tasks that require structured reasoning. PKR-QA is constructed semi-automatically using a procedural knowledge graph (PKG), which encodes task-specific knowledge across diverse domains. The PKG is built by curating and linking information from the COIN instructional video dataset and the ontology, enriched with commonsense knowledge from ConceptNet and structured outputs from Large Language Models (LLMs), followed by manual verification. To generate question-answer pairs, we design graph traversal templates where each template is applied systematically over PKG. To enable interpretable reasoning, we propose a neurosymbolic approach called Knowledge Module Learning (KML), which learns procedural relations via neural modules and composes them for structured reasoning with LLMs. Experiments demonstrate that this paradigm improves reasoning performance on our dataset and enables step-by-step reasoning traces that facilitate interpretability. Our theoretical analysis on KML learning shows that our trained models satisfy near optimal conditions for learning KG relations as neural network mapping models. Code and dataset will be released soon.
>
---
#### [replaced 054] RESfM: Robust Deep Equivariant Structure from Motion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.14280v2](http://arxiv.org/pdf/2404.14280v2)**

> **作者:** Fadi Khatib; Yoni Kasten; Dror Moran; Meirav Galun; Ronen Basri
>
> **备注:** Accepted to ICLR 2025. Project page: https://robust-equivariant-sfm.github.io/
>
> **摘要:** Multiview Structure from Motion is a fundamental and challenging computer vision problem. A recent deep-based approach utilized matrix equivariant architectures for simultaneous recovery of camera pose and 3D scene structure from large image collections. That work, however, made the unrealistic assumption that the point tracks given as input are almost clean of outliers. Here, we propose an architecture suited to dealing with outliers by adding a multiview inlier/outlier classification module that respects the model equivariance and by utilizing a robust bundle adjustment step. Experiments demonstrate that our method can be applied successfully in realistic settings that include large image collections and point tracks extracted with common heuristics that include many outliers, achieving state-of-the-art accuracies in almost all runs, superior to existing deep-based methods and on-par with leading classical (non-deep) sequential and global methods.
>
---
#### [replaced 055] BannerAgency: Advertising Banner Design with Multimodal LLM Agents
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11060v2](http://arxiv.org/pdf/2503.11060v2)**

> **作者:** Heng Wang; Yotaro Shimose; Shingo Takamatsu
>
> **备注:** Accepted as a main conference paper at EMNLP 2025
>
> **摘要:** Advertising banners are critical for capturing user attention and enhancing advertising campaign effectiveness. Creating aesthetically pleasing banner designs while conveying the campaign messages is challenging due to the large search space involving multiple design elements. Additionally, advertisers need multiple sizes for different displays and various versions to target different sectors of audiences. Since design is intrinsically an iterative and subjective process, flexible editability is also in high demand for practical usage. While current models have served as assistants to human designers in various design tasks, they typically handle only segments of the creative design process or produce pixel-based outputs that limit editability. This paper introduces a training-free framework for fully automated banner ad design creation, enabling frontier multimodal large language models (MLLMs) to streamline the production of effective banners with minimal manual effort across diverse marketing contexts. We present BannerAgency, an MLLM agent system that collaborates with advertisers to understand their brand identity and banner objectives, generates matching background images, creates blueprints for foreground design elements, and renders the final creatives as editable components in Figma or SVG formats rather than static pixels. To facilitate evaluation and future research, we introduce BannerRequest400, a benchmark featuring 100 unique logos paired with 400 diverse banner requests. Through quantitative and qualitative evaluations, we demonstrate the framework's effectiveness, emphasizing the quality of the generated banner designs, their adaptability to various banner requests, and their strong editability enabled by this component-based approach.
>
---
#### [replaced 056] Synthesizing Near-Boundary OOD Samples for Out-of-Distribution Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.10225v3](http://arxiv.org/pdf/2507.10225v3)**

> **作者:** Jinglun Li; Kaixun Jiang; Zhaoyu Chen; Bo Lin; Yao Tang; Weifeng Ge; Wenqiang Zhang
>
> **备注:** Accepted by ICCV 2025 (Highlight)
>
> **摘要:** Pre-trained vision-language models have exhibited remarkable abilities in detecting out-of-distribution (OOD) samples. However, some challenging OOD samples, which lie close to in-distribution (InD) data in image feature space, can still lead to misclassification. The emergence of foundation models like diffusion models and multimodal large language models (MLLMs) offers a potential solution to this issue. In this work, we propose SynOOD, a novel approach that harnesses foundation models to generate synthetic, challenging OOD data for fine-tuning CLIP models, thereby enhancing boundary-level discrimination between InD and OOD samples. Our method uses an iterative in-painting process guided by contextual prompts from MLLMs to produce nuanced, boundary-aligned OOD samples. These samples are refined through noise adjustments based on gradients from OOD scores like the energy score, effectively sampling from the InD/OOD boundary. With these carefully synthesized images, we fine-tune the CLIP image encoder and negative label features derived from the text encoder to strengthen connections between near-boundary OOD samples and a set of negative labels. Finally, SynOOD achieves state-of-the-art performance on the large-scale ImageNet benchmark, with minimal increases in parameters and runtime. Our approach significantly surpasses existing methods, and the code is available at https://github.com/Jarvisgivemeasuit/SynOOD.
>
---
#### [replaced 057] Capturing Stable HDR Videos Using a Dual-Camera System
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.06593v2](http://arxiv.org/pdf/2507.06593v2)**

> **作者:** Qianyu Zhang; Bolun Zheng; Lingyu Zhu; Hangjia Pan; Zunjie Zhu; Zongpeng Li; Shiqi Wang
>
> **摘要:** High Dynamic Range (HDR) video acquisition using the alternating exposure (AE) paradigm has garnered significant attention due to its cost-effectiveness with a single consumer camera. However, despite progress driven by deep neural networks, these methods remain prone to temporal flicker in real-world applications due to inter-frame exposure inconsistencies. To address this challenge while maintaining the cost-effectiveness of the AE paradigm, we propose a novel learning-based HDR video generation solution. Specifically, we propose a dual-stream HDR video generation paradigm that decouples temporal luminance anchoring from exposure-variant detail reconstruction, overcoming the inherent limitations of the AE paradigm. To support this, we design an asynchronous dual-camera system (DCS), which enables independent exposure control across two cameras, eliminating the need for synchronization typically required in traditional multi-camera setups. Furthermore, an exposure-adaptive fusion network (EAFNet) is formulated for the DCS system. EAFNet integrates a pre-alignment subnetwork that aligns features across varying exposures, ensuring robust feature extraction for subsequent fusion, an asymmetric cross-feature fusion subnetwork that emphasizes reference-based attention to effectively merge these features across exposures, and a reconstruction subnetwork to mitigate ghosting artifacts and preserve fine details. Extensive experimental evaluations demonstrate that the proposed method achieves state-of-the-art performance across various datasets, showing the remarkable potential of our solution in HDR video reconstruction. The codes and data captured by DCS will be available at https://zqqqyu.github.io/DCS-HDR/.
>
---
#### [replaced 058] DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.13560v2](http://arxiv.org/pdf/2508.13560v2)**

> **作者:** Zhen Qu; Xian Tao; Xinyi Gong; ShiChen Qu; Xiaopei Zhang; Xingang Wang; Fei Shen; Zhengtao Zhang; Mukesh Prasad; Guiguang Ding
>
> **备注:** Accepted by ICCV 2025, Project: https://github.com/xiaozhen228/DictAS
>
> **摘要:** Recent vision-language models (e.g., CLIP) have demonstrated remarkable class-generalizable ability to unseen classes in few-shot anomaly segmentation (FSAS), leveraging supervised prompt learning or fine-tuning on seen classes. However, their cross-category generalization largely depends on prior knowledge of real seen anomaly samples. In this paper, we propose a novel framework, namely DictAS, which enables a unified model to detect visual anomalies in unseen object categories without any retraining on the target data, only employing a few normal reference images as visual prompts. The insight behind DictAS is to transfer dictionary lookup capabilities to the FSAS task for unseen classes via self-supervised learning, instead of merely memorizing the normal and abnormal feature patterns from the training set. Specifically, DictAS mainly consists of three components: (1) Dictionary Construction - to simulate the index and content of a real dictionary using features from normal reference images. (2) Dictionary Lookup - to retrieve queried region features from the dictionary via a sparse lookup strategy. When a query feature cannot be retrieved, it is classified as an anomaly. (3) Query Discrimination Regularization - to enhance anomaly discrimination by making abnormal features harder to retrieve from the dictionary. To achieve this, Contrastive Query Constraint and Text Alignment Constraint are further proposed. Extensive experiments on seven public industrial and medical datasets demonstrate that DictAS consistently outperforms state-of-the-art FSAS methods.
>
---
#### [replaced 059] Flexible Tool Selection through Low-dimensional Attribute Alignment of Vision and Language
- **分类: cs.CV; cs.AI; cs.CL; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2505.22146v4](http://arxiv.org/pdf/2505.22146v4)**

> **作者:** Guangfu Hao; Haojie Wen; Liangxuan Guo; Yang Chen; Yanchao Bi; Shan Yu
>
> **摘要:** Flexible tool selection reflects a complex cognitive ability that distinguishes humans from other species, yet computational models that capture this ability remain underdeveloped. We developed a framework using low-dimensional attribute representations to bridge visual tool perception and linguistic task understanding. We constructed a comprehensive dataset (ToolNet) containing 115 common tools labeled with 13 carefully designed attributes spanning physical, functional, and psychological properties, paired with natural language scenarios describing tool usage. Visual encoders (ResNet or ViT) extract attributes from tool images while fine-tuned language models (GPT-2, LLaMA, DeepSeek) derive required attributes from task descriptions. Our approach achieves 74% accuracy in tool selection tasks-significantly outperforming direct tool matching (20%) and smaller multimodal models (21%-58%), while approaching performance of much larger models like GPT-4o (73%) with substantially fewer parameters. Human evaluation studies validate our framework's alignment with human decision-making patterns, and generalization experiments demonstrate effective performance on novel tool categories. Ablation studies revealed that manipulation-related attributes (graspability, elongation, hand-relatedness) consistently prove most critical across modalities. This work provides a parameter-efficient, interpretable solution that mimics human-like tool cognition, advancing both cognitive science understanding and practical applications in tool selection tasks.
>
---
#### [replaced 060] Fast-DDPM: Fast Denoising Diffusion Probabilistic Models for Medical Image-to-Image Generation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.14802v3](http://arxiv.org/pdf/2405.14802v3)**

> **作者:** Hongxu Jiang; Muhammad Imran; Teng Zhang; Yuyin Zhou; Muxuan Liang; Kuang Gong; Wei Shao
>
> **摘要:** Denoising diffusion probabilistic models (DDPMs) have achieved unprecedented success in computer vision. However, they remain underutilized in medical imaging, a field crucial for disease diagnosis and treatment planning. This is primarily due to the high computational cost associated with (1) the use of large number of time steps (e.g., 1,000) in diffusion processes and (2) the increased dimensionality of medical images, which are often 3D or 4D. Training a diffusion model on medical images typically takes days to weeks, while sampling each image volume takes minutes to hours. To address this challenge, we introduce Fast-DDPM, a simple yet effective approach capable of improving training speed, sampling speed, and generation quality simultaneously. Unlike DDPM, which trains the image denoiser across 1,000 time steps, Fast-DDPM trains and samples using only 10 time steps. The key to our method lies in aligning the training and sampling procedures to optimize time-step utilization. Specifically, we introduced two efficient noise schedulers with 10 time steps: one with uniform time step sampling and another with non-uniform sampling. We evaluated Fast-DDPM across three medical image-to-image generation tasks: multi-image super-resolution, image denoising, and image-to-image translation. Fast-DDPM outperformed DDPM and current state-of-the-art methods based on convolutional networks and generative adversarial networks in all tasks. Additionally, Fast-DDPM reduced the training time to 0.2x and the sampling time to 0.01x compared to DDPM. Our code is publicly available at: https://github.com/mirthAI/Fast-DDPM.
>
---
#### [replaced 061] EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.11709v2](http://arxiv.org/pdf/2505.11709v2)**

> **作者:** Ryan Hoque; Peide Huang; David J. Yoon; Mouli Sivapurapu; Jian Zhang
>
> **摘要:** Imitation learning for manipulation has a well-known data scarcity problem. Unlike natural language and 2D computer vision, there is no Internet-scale corpus of data for dexterous manipulation. One appealing option is egocentric human video, a passively scalable data source. However, existing large-scale datasets such as Ego4D do not have native hand pose annotations and do not focus on object manipulation. To this end, we use Apple Vision Pro to collect EgoDex: the largest and most diverse dataset of dexterous human manipulation to date. EgoDex has 829 hours of egocentric video with paired 3D hand and finger tracking data collected at the time of recording, where multiple calibrated cameras and on-device SLAM can be used to precisely track the pose of every joint of each hand. The dataset covers a wide range of diverse manipulation behaviors with everyday household objects in 194 different tabletop tasks ranging from tying shoelaces to folding laundry. Furthermore, we train and systematically evaluate imitation learning policies for hand trajectory prediction on the dataset, introducing metrics and benchmarks for measuring progress in this increasingly important area. By releasing this large-scale dataset, we hope to push the frontier of robotics, computer vision, and foundation models. EgoDex is publicly available for download at https://github.com/apple/ml-egodex.
>
---
#### [replaced 062] LORE: Latent Optimization for Precise Semantic Control in Rectified Flow-based Image Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.03144v2](http://arxiv.org/pdf/2508.03144v2)**

> **作者:** Liangyang Ouyang; Jiafeng Mao
>
> **备注:** Our implementation is available at https://github.com/oyly16/LORE
>
> **摘要:** Text-driven image editing enables users to flexibly modify visual content through natural language instructions, and is widely applied to tasks such as semantic object replacement, insertion, and removal. While recent inversion-based editing methods using rectified flow models have achieved promising results in image quality, we identify a structural limitation in their editing behavior: the semantic bias toward the source concept encoded in the inverted noise tends to suppress attention to the target concept. This issue becomes particularly critical when the source and target semantics are dissimilar, where the attention mechanism inherently leads to editing failure or unintended modifications in non-target regions. In this paper, we systematically analyze and validate this structural flaw, and introduce LORE, a training-free and efficient image editing method. LORE directly optimizes the inverted noise, addressing the core limitations in generalization and controllability of existing approaches, enabling stable, controllable, and general-purpose concept replacement, without requiring architectural modification or model fine-tuning. We conduct comprehensive evaluations on three challenging benchmarks: PIEBench, SmartEdit, and GapEdit. Experimental results show that LORE significantly outperforms strong baselines in terms of semantic alignment, image quality, and background fidelity, demonstrating the effectiveness and scalability of latent-space optimization for general-purpose image editing. Our implementation is available at https://github.com/oyly16/LORE.
>
---
#### [replaced 063] Label Anything: Multi-Class Few-Shot Semantic Segmentation with Visual Prompts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.02075v4](http://arxiv.org/pdf/2407.02075v4)**

> **作者:** Pasquale De Marinis; Nicola Fanelli; Raffaele Scaringi; Emanuele Colonna; Giuseppe Fiameni; Gennaro Vessio; Giovanna Castellano
>
> **备注:** ECAI 2025 - 28th European Conference on Artificial Intelligence
>
> **摘要:** Few-shot semantic segmentation aims to segment objects from previously unseen classes using only a limited number of labeled examples. In this paper, we introduce Label Anything, a novel transformer-based architecture designed for multi-prompt, multi-way few-shot semantic segmentation. Our approach leverages diverse visual prompts -- points, bounding boxes, and masks -- to create a highly flexible and generalizable framework that significantly reduces annotation burden while maintaining high accuracy. Label Anything makes three key contributions: ($\textit{i}$) we introduce a new task formulation that relaxes conventional few-shot segmentation constraints by supporting various types of prompts, multi-class classification, and enabling multiple prompts within a single image; ($\textit{ii}$) we propose a novel architecture based on transformers and attention mechanisms; and ($\textit{iii}$) we design a versatile training procedure allowing our model to operate seamlessly across different $N$-way $K$-shot and prompt-type configurations with a single trained model. Our extensive experimental evaluation on the widely used COCO-$20^i$ benchmark demonstrates that Label Anything achieves state-of-the-art performance among existing multi-way few-shot segmentation methods, while significantly outperforming leading single-class models when evaluated in multi-class settings. Code and trained models are available at https://github.com/pasqualedem/LabelAnything.
>
---
#### [replaced 064] Cross multiscale vision transformer for deep fake detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.00833v2](http://arxiv.org/pdf/2502.00833v2)**

> **作者:** Akhshan P; Taneti Sanjay; Chandrakala S
>
> **备注:** This version of the manuscript contains errors in wording and explanation, which may cause confusion in interpreting the methodology and results. The authors are preparing a revised version with corrected and clearer descriptions
>
> **摘要:** The proliferation of deep fake technology poses significant challenges to digital media authenticity, necessitating robust detection mechanisms. This project evaluates deep fake detection using the SP Cup's 2025 deep fake detection challenge dataset. We focused on exploring various deep learning models for detecting deep fake content, utilizing traditional deep learning techniques alongside newer architectures. Our approach involved training a series of models and rigorously assessing their performance using metrics such as accuracy.
>
---
#### [replaced 065] FastMap: Revisiting Structure from Motion through First-Order Optimization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.04612v3](http://arxiv.org/pdf/2505.04612v3)**

> **作者:** Jiahao Li; Haochen Wang; Muhammad Zubair Irshad; Igor Vasiljevic; Matthew R. Walter; Vitor Campagnolo Guizilini; Greg Shakhnarovich
>
> **备注:** Project webpage: https://jiahao.ai/fastmap
>
> **摘要:** We propose FastMap, a new global structure from motion method focused on speed and simplicity. Previous methods like COLMAP and GLOMAP are able to estimate high-precision camera poses, but suffer from poor scalability when the number of matched keypoint pairs becomes large, mainly due to the time-consuming process of second-order Gauss-Newton optimization. Instead, we design our method solely based on first-order optimizers. To obtain maximal speedup, we identify and eliminate two key performance bottlenecks: computational complexity and the kernel implementation of each optimization step. Through extensive experiments, we show that FastMap is up to 10 times faster than COLMAP and GLOMAP with GPU acceleration and achieves comparable pose accuracy.
>
---
#### [replaced 066] Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Model and Light-Weight Edge Model
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.13787v2](http://arxiv.org/pdf/2411.13787v2)**

> **作者:** Zewei Xin; Qinya Li; Chaoyue Niu; Fan Wu; Guihai Chen
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Large text-to-image models demonstrate impressive generation capabilities; however, their substantial size necessitates expensive cloud servers for deployment. Conversely, light-weight models can be deployed on edge devices at lower cost but often with inferior generation quality for complex user prompts. To strike a balance between performance and cost, we propose a routing framework, called RouteT2I, which dynamically selects either the large cloud model or the light-weight edge model for each user prompt. Since generated image quality is challenging to measure and compare directly, RouteT2I establishes multi-dimensional quality metrics, particularly, by evaluating the similarity between the generated images and both positive and negative texts that describe each specific quality metric. RouteT2I then predicts the expected quality of the generated images by identifying key tokens in the prompt and comparing their impact on the quality. RouteT2I further introduces the Pareto relative superiority to compare the multi-metric quality of the generated images. Based on this comparison and predefined cost constraints, RouteT2I allocates prompts to either the edge or the cloud. Evaluation reveals that RouteT2I significantly reduces the number of requesting large cloud model while maintaining high-quality image generation.
>
---
#### [replaced 067] When Better Eyes Lead to Blindness: A Diagnostic Study of the Information Bottleneck in CNN-LSTM Image Captioning Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.18788v2](http://arxiv.org/pdf/2507.18788v2)**

> **作者:** Hitesh Kumar Gupta
>
> **备注:** This paper is published in International Journal of Computer Applications (IJCA), Vol. 187, No. 31, August 2025
>
> **摘要:** Image captioning, situated at the intersection of computer vision and natural language processing, requires a sophisticated understanding of both visual scenes and linguistic structure. While modern approaches are dominated by large-scale Transformer architectures, this paper documents a systematic, iterative development of foundational image captioning models, progressing from a simple CNN-LSTM encoder-decoder to a competitive attention-based system. This paper presents a series of five models, beginning with Genesis and concluding with Nexus, an advanced model featuring an EfficientNetV2B3 backbone and a dynamic attention mechanism. The experiments chart the impact of architectural enhancements and demonstrate a key finding within the classic CNN-LSTM paradigm: merely upgrading the visual backbone without a corresponding attention mechanism can degrade performance, as the single-vector bottleneck cannot transmit the richer visual detail. This insight validates the architectural shift to attention. Trained on the MS COCO 2017 dataset, the final model, Nexus, achieves a BLEU-4 score of 31.4, surpassing several foundational benchmarks and validating the iterative design process. This work provides a clear, replicable blueprint for understanding the core architectural principles that underpin modern vision-language tasks.
>
---
#### [replaced 068] Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2407.20836v4](http://arxiv.org/pdf/2407.20836v4)**

> **作者:** Yunfeng Diao; Naixin Zhai; Changtao Miao; Zitong Yu; Xingxing Wei; Xun Yang; Meng Wang
>
> **摘要:** Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g. transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we show that adversarial attack is truly a real threat to AIGI detectors, because FPBA can deliver successful black-box attacks across models, generators, defense methods, and even evade cross-generator detection, which is a crucial real-world detection scenario. The code will be shared upon acceptance.
>
---
