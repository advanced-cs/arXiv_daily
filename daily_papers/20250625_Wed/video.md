# 计算机视觉 cs.CV

- **最新发布 117 篇**

- **更新 68 篇**

## 最新发布

#### [new 001] AMF-MedIT: An Efficient Align-Modulation-Fusion Framework for Medical Image-Tabular Data
- **分类: cs.CV**

- **简介: 该论文属于多模态医学分析任务，旨在解决图像与表格数据融合中的维度不匹配和噪声问题。提出AMF-MedIT框架，结合自适应调制与高效编码器，提升融合效果与数据效率。**

- **链接: [http://arxiv.org/pdf/2506.19439v1](http://arxiv.org/pdf/2506.19439v1)**

> **作者:** Congjing Yu; Jing Ye; Yang Liu; Xiaodong Zhang; Zhiyong Zhang
>
> **摘要:** Multimodal medical analysis combining image and tabular data has gained increasing attention. However, effective fusion remains challenging due to cross-modal discrepancies in feature dimensions and modality contributions, as well as the noise from high-dimensional tabular inputs. To address these problems, we present AMF-MedIT, an efficient Align-Modulation-Fusion framework for medical image and tabular data integration, particularly under data-scarce conditions. To harmonize dimension discrepancies and dynamically adjust modality contributions, we propose the Adaptive Modulation and Fusion (AMF) module, a novel modulation-based fusion paradigm with a streamlined architecture. We first derive the modulation objectives and introduce a modality confidence ratio, enabling the incorporation of prior knowledge into the fusion process. Then, the feature masks, density and leakage losses are proposed to achieve the modulation objectives. Additionally, we introduce FT-Mamba, a powerful tabular encoder leveraging a selective mechanism to handle noisy medical tabular data efficiently. Furthermore, interpretability studies are conducted to explore how different tabular encoders supervise the imaging modality during contrastive pretraining for the first time. Extensive experiments demonstrate that AMF-MedIT achieves a superior balance between multimodal performance and data efficiency while showing strong adaptability to incomplete tabular data. Interpretability analysis also highlights FT-Mamba's capabilities in extracting distinct tabular features and guiding the image encoder toward more accurate and flexible attention patterns.
>
---
#### [new 002] Stylized Structural Patterns for Improved Neural Network Pre-training
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于计算机视觉任务，旨在解决合成数据训练模型效果不佳的问题。通过引入新型合成数据和反向风格化技术，减少与真实数据的分布差距，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.19465v1](http://arxiv.org/pdf/2506.19465v1)**

> **作者:** Farnood Salehi; Vandit Sharma; Amirhossein Askari Farsangi; Tunç Ozan Aydın
>
> **摘要:** Modern deep learning models in computer vision require large datasets of real images, which are difficult to curate and pose privacy and legal concerns, limiting their commercial use. Recent works suggest synthetic data as an alternative, yet models trained with it often underperform. This paper proposes a two-step approach to bridge this gap. First, we propose an improved neural fractal formulation through which we introduce a new class of synthetic data. Second, we propose reverse stylization, a technique that transfers visual features from a small, license-free set of real images onto synthetic datasets, enhancing their effectiveness. We analyze the domain gap between our synthetic datasets and real images using Kernel Inception Distance (KID) and show that our method achieves a significantly lower distributional gap compared to existing synthetic datasets. Furthermore, our experiments across different tasks demonstrate the practical impact of this reduced gap. We show that pretraining the EDM2 diffusion model on our synthetic dataset leads to an 11% reduction in FID during image generation, compared to models trained on existing synthetic datasets, and a 20% decrease in autoencoder reconstruction error, indicating improved performance in data representation. Furthermore, a ViT-S model trained for classification on this synthetic data achieves over a 10% improvement in ImageNet-100 accuracy. Our work opens up exciting possibilities for training practical models when sufficiently large real training sets are not available.
>
---
#### [new 003] MambaOutRS: A Hybrid CNN-Fourier Architecture for Remote Sensing Image Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感图像分类任务，旨在解决传统模型效率低的问题。提出MambaOutRS架构，结合卷积和傅里叶门控模块，提升分类性能。**

- **链接: [http://arxiv.org/pdf/2506.19561v1](http://arxiv.org/pdf/2506.19561v1)**

> **作者:** Minjong Cheon; Changbae Mun
>
> **摘要:** Recent advances in deep learning for vision tasks have seen the rise of State Space Models (SSMs) like Mamba, celebrated for their linear scalability. However, their adaptation to 2D visual data often necessitates complex modifications that may diminish efficiency. In this paper, we introduce MambaOutRS, a novel hybrid convolutional architecture for remote sensing image classification that re-evaluates the necessity of recurrent SSMs. MambaOutRS builds upon stacked Gated CNN blocks for local feature extraction and introduces a novel Fourier Filter Gate (FFG) module that operates in the frequency domain to capture global contextual information efficiently. Our architecture employs a four-stage hierarchical design and was extensively evaluated on challenging remote sensing datasets: UC Merced, AID, NWPU-RESISC45, and EuroSAT. MambaOutRS consistently achieved state-of-the-art (SOTA) performance across these benchmarks. Notably, our MambaOutRS-t variant (24.0M parameters) attained the highest F1-scores of 98.41\% on UC Merced and 95.99\% on AID, significantly outperforming existing baselines, including larger transformer models and Mamba-based architectures, despite using considerably fewer parameters. An ablation study conclusively demonstrates the critical role of the Fourier Filter Gate in enhancing the model's ability to capture global spatial patterns, leading to robust and accurate classification. These results strongly suggest that the complexities of recurrent SSMs can be effectively superseded by a judicious combination of gated convolutions for spatial mixing and frequency-based gates for spectral global context. Thus, MambaOutRS provides a compelling and efficient paradigm for developing high-performance deep learning models in remote sensing and other vision domains, particularly where computational efficiency is paramount.
>
---
#### [new 004] HoliGS: Holistic Gaussian Splatting for Embodied View Synthesis
- **分类: cs.CV**

- **简介: 该论文属于环境视图合成任务，解决长视频中动态场景的高效重建问题。提出HoliGS框架，通过分解场景并利用高斯点云实现快速准确的视图渲染。**

- **链接: [http://arxiv.org/pdf/2506.19291v1](http://arxiv.org/pdf/2506.19291v1)**

> **作者:** Xiaoyuan Wang; Yizhou Zhao; Botao Ye; Xiaojun Shan; Weijie Lyu; Lu Qi; Kelvin C. K. Chan; Yinxiao Li; Ming-Hsuan Yang
>
> **摘要:** We propose HoliGS, a novel deformable Gaussian splatting framework that addresses embodied view synthesis from long monocular RGB videos. Unlike prior 4D Gaussian splatting and dynamic NeRF pipelines, which struggle with training overhead in minute-long captures, our method leverages invertible Gaussian Splatting deformation networks to reconstruct large-scale, dynamic environments accurately. Specifically, we decompose each scene into a static background plus time-varying objects, each represented by learned Gaussian primitives undergoing global rigid transformations, skeleton-driven articulation, and subtle non-rigid deformations via an invertible neural flow. This hierarchical warping strategy enables robust free-viewpoint novel-view rendering from various embodied camera trajectories by attaching Gaussians to a complete canonical foreground shape (\eg, egocentric or third-person follow), which may involve substantial viewpoint changes and interactions between multiple actors. Our experiments demonstrate that \ourmethod~ achieves superior reconstruction quality on challenging datasets while significantly reducing both training and rendering time compared to state-of-the-art monocular deformable NeRFs. These results highlight a practical and scalable solution for EVS in real-world scenarios. The source code will be released.
>
---
#### [new 005] Vision Transformer-Based Time-Series Image Reconstruction for Cloud-Filling Applications
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文属于遥感图像重建任务，旨在解决云层遮挡导致的多光谱影像缺失问题，通过结合时间序列信息与SAR数据，利用ViT模型提升重建效果。**

- **链接: [http://arxiv.org/pdf/2506.19591v1](http://arxiv.org/pdf/2506.19591v1)**

> **作者:** Lujun Li; Yiqun Wang; Radu State
>
> **备注:** This paper has been accepted as a conference paper at the 2025 IEEE International Geoscience and Remote Sensing Symposium (IGARSS)
>
> **摘要:** Cloud cover in multispectral imagery (MSI) poses significant challenges for early season crop mapping, as it leads to missing or corrupted spectral information. Synthetic aperture radar (SAR) data, which is not affected by cloud interference, offers a complementary solution, but lack sufficient spectral detail for precise crop mapping. To address this, we propose a novel framework, Time-series MSI Image Reconstruction using Vision Transformer (ViT), to reconstruct MSI data in cloud-covered regions by leveraging the temporal coherence of MSI and the complementary information from SAR from the attention mechanism. Comprehensive experiments, using rigorous reconstruction evaluation metrics, demonstrate that Time-series ViT framework significantly outperforms baselines that use non-time-series MSI and SAR or time-series MSI without SAR, effectively enhancing MSI image reconstruction in cloud-covered regions.
>
---
#### [new 006] SceneCrafter: Controllable Multi-View Driving Scene Editing
- **分类: cs.CV**

- **简介: 该论文提出SceneCrafter，用于可控的多视角驾驶场景编辑。解决真实感、3D一致性及场景编辑质量问题，通过多视图扩散模型实现高效场景修改。**

- **链接: [http://arxiv.org/pdf/2506.19488v1](http://arxiv.org/pdf/2506.19488v1)**

> **作者:** Zehao Zhu; Yuliang Zou; Chiyu Max Jiang; Bo Sun; Vincent Casser; Xiukun Huang; Jiahao Wang; Zhenpei Yang; Ruiqi Gao; Leonidas Guibas; Mingxing Tan; Dragomir Anguelov
>
> **备注:** CVPR 2025
>
> **摘要:** Simulation is crucial for developing and evaluating autonomous vehicle (AV) systems. Recent literature builds on a new generation of generative models to synthesize highly realistic images for full-stack simulation. However, purely synthetically generated scenes are not grounded in reality and have difficulty in inspiring confidence in the relevance of its outcomes. Editing models, on the other hand, leverage source scenes from real driving logs, and enable the simulation of different traffic layouts, behaviors, and operating conditions such as weather and time of day. While image editing is an established topic in computer vision, it presents fresh sets of challenges in driving simulation: (1) the need for cross-camera 3D consistency, (2) learning ``empty street" priors from driving data with foreground occlusions, and (3) obtaining paired image tuples of varied editing conditions while preserving consistent layout and geometry. To address these challenges, we propose SceneCrafter, a versatile editor for realistic 3D-consistent manipulation of driving scenes captured from multiple cameras. We build on recent advancements in multi-view diffusion models, using a fully controllable framework that scales seamlessly to multi-modality conditions like weather, time of day, agent boxes and high-definition maps. To generate paired data for supervising the editing model, we propose a novel framework on top of Prompt-to-Prompt to generate geometrically consistent synthetic paired data with global edits. We also introduce an alpha-blending framework to synthesize data with local edits, leveraging a model trained on empty street priors through novel masked training and multi-view repaint paradigm. SceneCrafter demonstrates powerful editing capabilities and achieves state-of-the-art realism, controllability, 3D consistency, and scene editing quality compared to existing baselines.
>
---
#### [new 007] From Pixels and Words to Waves: A Unified Framework for Spectral Dictionary vLLMs
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在解决传统模型中计算复杂度高的问题。通过引入频谱字典混合器，替代卷积和自注意力机制，提升效率与可解释性。**

- **链接: [http://arxiv.org/pdf/2506.18943v1](http://arxiv.org/pdf/2506.18943v1)**

> **作者:** Andrew Kiruluta; Priscilla Burity
>
> **摘要:** Vision-language models (VLMs) unify computer vision and natural language processing in a single architecture capable of interpreting and describing images. Most state-of-the-art systems rely on two computationally intensive components: convolutions in the vision encoder and quadratic self-attention for multimodal fusion. This work removes both by introducing a spectral dictionary token mixer, which represents each image patch or wordpiece as a sparse combination of learnable frequency atoms. Our 1.1B-parameter prototype, SDict-VLM, achieves BLEU-4 of 39.2, CIDEr of 127.5, and SPICE of 27.0 on MS-COCO captioning, along with 50.3 percent accuracy on VQAv2. These results close approximately 85 percent of the performance gap to BLIP-2 while using 60 percent fewer parameters, 2.3 times less peak GPU memory, and 2.2 times faster inference than PaLI-3. To our knowledge, this is the first VLM to eliminate both convolutions and self-attention while matching mid-scale transformer baselines. In addition to its O(L log L) complexity, the shared frequency dictionary enables transparent cross-modal alignment and offers a tunable trade-off between accuracy and compute, paving the way for efficient and interpretable VLMs.
>
---
#### [new 008] Progressive Modality Cooperation for Multi-Modality Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文属于多模态域适应任务，旨在解决跨域视觉识别中模态信息不匹配的问题。提出PMC框架，通过模态协作选择可靠样本，并在缺失模态时生成缺失模态数据以提升性能。**

- **链接: [http://arxiv.org/pdf/2506.19316v1](http://arxiv.org/pdf/2506.19316v1)**

> **作者:** Weichen Zhang; Dong Xu; Jing Zhang; Wanli Ouyang
>
> **摘要:** In this work, we propose a new generic multi-modality domain adaptation framework called Progressive Modality Cooperation (PMC) to transfer the knowledge learned from the source domain to the target domain by exploiting multiple modality clues (\eg, RGB and depth) under the multi-modality domain adaptation (MMDA) and the more general multi-modality domain adaptation using privileged information (MMDA-PI) settings. Under the MMDA setting, the samples in both domains have all the modalities. In two newly proposed modules of our PMC, the multiple modalities are cooperated for selecting the reliable pseudo-labeled target samples, which captures the modality-specific information and modality-integrated information, respectively. Under the MMDA-PI setting, some modalities are missing in the target domain. Hence, to better exploit the multi-modality data in the source domain, we further propose the PMC with privileged information (PMC-PI) method by proposing a new multi-modality data generation (MMG) network. MMG generates the missing modalities in the target domain based on the source domain data by considering both domain distribution mismatch and semantics preservation, which are respectively achieved by using adversarial learning and conditioning on weighted pseudo semantics. Extensive experiments on three image datasets and eight video datasets for various multi-modality cross-domain visual recognition tasks under both MMDA and MMDA-PI settings clearly demonstrate the effectiveness of our proposed PMC framework.
>
---
#### [new 009] Unified Vision-Language-Action Model
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出UniVLA，一种统一的视觉-语言-动作模型，解决机器人操作中动作生成问题，通过联合建模视觉、语言和动作信号，提升长时任务性能。**

- **链接: [http://arxiv.org/pdf/2506.19850v1](http://arxiv.org/pdf/2506.19850v1)**

> **作者:** Yuqi Wang; Xinghang Li; Wenxuan Wang; Junbo Zhang; Yingyan Li; Yuntao Chen; Xinlong Wang; Zhaoxiang Zhang
>
> **备注:** technical report
>
> **摘要:** Vision-language-action models (VLAs) have garnered significant attention for their potential in advancing robotic manipulation. However, previous approaches predominantly rely on the general comprehension capabilities of vision-language models (VLMs) to generate action signals, often overlooking the rich temporal and causal structure embedded in visual observations. In this paper, we present UniVLA, a unified and native multimodal VLA model that autoregressively models vision, language, and action signals as discrete token sequences. This formulation enables flexible multimodal tasks learning, particularly from large-scale video data. By incorporating world modeling during post-training, UniVLA captures causal dynamics from videos, facilitating effective transfer to downstream policy learning--especially for long-horizon tasks. Our approach sets new state-of-the-art results across several widely used simulation benchmarks, including CALVIN, LIBERO, and Simplenv-Bridge, significantly surpassing previous methods. For example, UniVLA achieves 95.5% average success rate on LIBERO benchmark, surpassing pi0-FAST's 85.5%. We further demonstrate its broad applicability on real-world ALOHA manipulation and autonomous driving.
>
---
#### [new 010] Mem4Nav: Boosting Vision-and-Language Navigation in Urban Environments with a Hierarchical Spatial-Cognition Long-Short Memory System
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言导航任务，旨在提升城市环境中智能体对语言指令的理解与空间推理能力。通过引入分层记忆系统Mem4Nav，融合细粒度空间索引与语义拓扑图，增强长期与短期记忆模块，显著提升了导航性能。**

- **链接: [http://arxiv.org/pdf/2506.19433v1](http://arxiv.org/pdf/2506.19433v1)**

> **作者:** Lixuan He; Haoyu Dong; Zhenxing Chen; Yangcheng Yu; Jie Feng; Yong Li
>
> **摘要:** Vision-and-Language Navigation (VLN) in large-scale urban environments requires embodied agents to ground linguistic instructions in complex scenes and recall relevant experiences over extended time horizons. Prior modular pipelines offer interpretability but lack unified memory, while end-to-end (M)LLM agents excel at fusing vision and language yet remain constrained by fixed context windows and implicit spatial reasoning. We introduce \textbf{Mem4Nav}, a hierarchical spatial-cognition long-short memory system that can augment any VLN backbone. Mem4Nav fuses a sparse octree for fine-grained voxel indexing with a semantic topology graph for high-level landmark connectivity, storing both in trainable memory tokens embedded via a reversible Transformer. Long-term memory (LTM) compresses and retains historical observations at both octree and graph nodes, while short-term memory (STM) caches recent multimodal entries in relative coordinates for real-time obstacle avoidance and local planning. At each step, STM retrieval sharply prunes dynamic context, and, when deeper history is needed, LTM tokens are decoded losslessly to reconstruct past embeddings. Evaluated on Touchdown and Map2Seq across three backbones (modular, state-of-the-art VLN with prompt-based LLM, and state-of-the-art VLN with strided-attention MLLM), Mem4Nav yields 7-13 pp gains in Task Completion, sufficient SPD reduction, and >10 pp nDTW improvement. Ablations confirm the indispensability of both the hierarchical map and dual memory modules. Our codes are open-sourced via https://github.com/tsinghua-fib-lab/Mem4Nav.
>
---
#### [new 011] Visual hallucination detection in large vision-language models via evidential conflict
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉幻觉检测任务，旨在解决LVLMs在视觉与文本输出不一致的问题。构建了PRE-HAL数据集，并提出基于DST的检测方法，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.19513v1](http://arxiv.org/pdf/2506.19513v1)**

> **作者:** Tao Huang; Zhekun Liu; Rui Wang; Yang Zhang; Liping Jing
>
> **摘要:** Despite the remarkable multimodal capabilities of Large Vision-Language Models (LVLMs), discrepancies often occur between visual inputs and textual outputs--a phenomenon we term visual hallucination. This critical reliability gap poses substantial risks in safety-critical Artificial Intelligence (AI) applications, necessitating a comprehensive evaluation benchmark and effective detection methods. Firstly, we observe that existing visual-centric hallucination benchmarks mainly assess LVLMs from a perception perspective, overlooking hallucinations arising from advanced reasoning capabilities. We develop the Perception-Reasoning Evaluation Hallucination (PRE-HAL) dataset, which enables the systematic evaluation of both perception and reasoning capabilities of LVLMs across multiple visual semantics, such as instances, scenes, and relations. Comprehensive evaluation with this new benchmark exposed more visual vulnerabilities, particularly in the more challenging task of relation reasoning. To address this issue, we propose, to the best of our knowledge, the first Dempster-Shafer theory (DST)-based visual hallucination detection method for LVLMs through uncertainty estimation. This method aims to efficiently capture the degree of conflict in high-level features at the model inference phase. Specifically, our approach employs simple mass functions to mitigate the computational complexity of evidence combination on power sets. We conduct an extensive evaluation of state-of-the-art LVLMs, LLaVA-v1.5, mPLUG-Owl2 and mPLUG-Owl3, with the new PRE-HAL benchmark. Experimental results indicate that our method outperforms five baseline uncertainty metrics, achieving average AUROC improvements of 4%, 10%, and 7% across three LVLMs. Our code is available at https://github.com/HT86159/Evidential-Conflict.
>
---
#### [new 012] Radial Attention: $O(n\log n)$ Sparse Attention with Energy Decay for Long Video Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视频生成任务，解决长视频生成计算成本高的问题。提出Radial Attention机制，实现高效稀疏注意力，提升生成速度与长度。**

- **链接: [http://arxiv.org/pdf/2506.19852v1](http://arxiv.org/pdf/2506.19852v1)**

> **作者:** Xingyang Li; Muyang Li; Tianle Cai; Haocheng Xi; Shuo Yang; Yujun Lin; Lvmin Zhang; Songlin Yang; Jinbo Hu; Kelly Peng; Maneesh Agrawala; Ion Stoica; Kurt Keutzer; Song Han
>
> **备注:** Code: https://github.com/mit-han-lab/radial-attention
>
> **摘要:** Recent advances in diffusion models have enabled high-quality video generation, but the additional temporal dimension significantly increases computational costs, making training and inference on long videos prohibitively expensive. In this paper, we identify a phenomenon we term Spatiotemporal Energy Decay in video diffusion models: post-softmax attention scores diminish as spatial and temporal distance between tokens increase, akin to the physical decay of signal or waves over space and time in nature. Motivated by this, we propose Radial Attention, a scalable sparse attention mechanism with $O(n \log n)$ complexity that translates energy decay into exponentially decaying compute density, which is significantly more efficient than standard $O(n^2)$ dense attention and more expressive than linear attention. Specifically, Radial Attention employs a simple, static attention mask where each token attends to spatially nearby tokens, with the attention window size shrinking with temporal distance. Moreover, it allows pre-trained video diffusion models to extend their generation length with efficient LoRA-based fine-tuning. Extensive experiments show that Radial Attention maintains video quality across Wan2.1-14B, HunyuanVideo, and Mochi 1, achieving up to a 1.9$\times$ speedup over the original dense attention. With minimal tuning, it enables video generation up to 4$\times$ longer while reducing training costs by up to 4.4$\times$ compared to direct fine-tuning and accelerating inference by up to 3.7$\times$ compared to dense attention inference.
>
---
#### [new 013] General Methods Make Great Domain-specific Foundation Models: A Case-study on Fetal Ultrasound
- **分类: cs.CV; cs.AI; cs.LG; I.4**

- **简介: 该论文属于医学图像分析任务，探讨在胎儿超声数据上预训练专用基础模型的有效性。研究对比了自定义预训练与迁移学习的效果，验证了通用方法在医疗领域的可行性。**

- **链接: [http://arxiv.org/pdf/2506.19552v1](http://arxiv.org/pdf/2506.19552v1)**

> **作者:** Jakob Ambsdorf; Asbjørn Munk; Sebastian Llambias; Anders Nymark Christensen; Kamil Mikolaj; Randall Balestriero; Martin Tolsgaard; Aasa Feragen; Mads Nielsen
>
> **备注:** Submitted version of paper accepted at MICCAI 2025
>
> **摘要:** With access to large-scale, unlabeled medical datasets, researchers are confronted with two questions: Should they attempt to pretrain a custom foundation model on this medical data, or use transfer-learning from an existing generalist model? And, if a custom model is pretrained, are novel methods required? In this paper we explore these questions by conducting a case-study, in which we train a foundation model on a large regional fetal ultrasound dataset of 2M images. By selecting the well-established DINOv2 method for pretraining, we achieve state-of-the-art results on three fetal ultrasound datasets, covering data from different countries, classification, segmentation, and few-shot tasks. We compare against a series of models pretrained on natural images, ultrasound images, and supervised baselines. Our results demonstrate two key insights: (i) Pretraining on custom data is worth it, even if smaller models are trained on less data, as scaling in natural image pretraining does not translate to ultrasound performance. (ii) Well-tuned methods from computer vision are making it feasible to train custom foundation models for a given medical domain, requiring no hyperparameter tuning and little methodological adaptation. Given these findings, we argue that a bias towards methodological innovation should be avoided when developing domain specific foundation models under common computational resource constraints.
>
---
#### [new 014] Segment Any 3D-Part in a Scene from a Sentence
- **分类: cs.CV**

- **简介: 该论文属于3D场景中基于自然语言的部件分割任务，旨在解决数据不足和方法局限问题，提出3D-PU数据集和OpenPart3D框架。**

- **链接: [http://arxiv.org/pdf/2506.19331v1](http://arxiv.org/pdf/2506.19331v1)**

> **作者:** Hongyu Wu; Pengwan Yang; Yuki M. Asano; Cees G. M. Snoek
>
> **摘要:** This paper aims to achieve the segmentation of any 3D part in a scene based on natural language descriptions, extending beyond traditional object-level 3D scene understanding and addressing both data and methodological challenges. Due to the expensive acquisition and annotation burden, existing datasets and methods are predominantly limited to object-level comprehension. To overcome the limitations of data and annotation availability, we introduce the 3D-PU dataset, the first large-scale 3D dataset with dense part annotations, created through an innovative and cost-effective method for constructing synthetic 3D scenes with fine-grained part-level annotations, paving the way for advanced 3D-part scene understanding. On the methodological side, we propose OpenPart3D, a 3D-input-only framework to effectively tackle the challenges of part-level segmentation. Extensive experiments demonstrate the superiority of our approach in open-vocabulary 3D scene understanding tasks at the part level, with strong generalization capabilities across various 3D scene datasets.
>
---
#### [new 015] Assessing Risk of Stealing Proprietary Models for Medical Imaging Tasks
- **分类: cs.CV; cs.CR**

- **简介: 该论文研究医疗影像模型的模型窃取攻击问题，提出QueryWise方法在有限查询预算下实现有效攻击。**

- **链接: [http://arxiv.org/pdf/2506.19464v1](http://arxiv.org/pdf/2506.19464v1)**

> **作者:** Ankita Raj; Harsh Swaika; Deepankar Varma; Chetan Arora
>
> **备注:** Accepted to MICCAI 2024
>
> **摘要:** The success of deep learning in medical imaging applications has led several companies to deploy proprietary models in diagnostic workflows, offering monetized services. Even though model weights are hidden to protect the intellectual property of the service provider, these models are exposed to model stealing (MS) attacks, where adversaries can clone the model's functionality by querying it with a proxy dataset and training a thief model on the acquired predictions. While extensively studied on general vision tasks, the susceptibility of medical imaging models to MS attacks remains inadequately explored. This paper investigates the vulnerability of black-box medical imaging models to MS attacks under realistic conditions where the adversary lacks access to the victim model's training data and operates with limited query budgets. We demonstrate that adversaries can effectively execute MS attacks by using publicly available datasets. To further enhance MS capabilities with limited query budgets, we propose a two-step model stealing approach termed QueryWise. This method capitalizes on unlabeled data obtained from a proxy distribution to train the thief model without incurring additional queries. Evaluation on two medical imaging models for Gallbladder Cancer and COVID-19 classification substantiates the effectiveness of the proposed attack. The source code is available at https://github.com/rajankita/QueryWise.
>
---
#### [new 016] Orthogonal Projection Subspace to Aggregate Online Prior-knowledge for Continual Test-time Adaptation
- **分类: cs.CV**

- **简介: 该论文属于持续测试时适应（CTTA）任务，旨在解决灾难性遗忘和错误累积问题。提出OoPk方法，通过正交投影和在线先验知识聚合提升模型适应能力。**

- **链接: [http://arxiv.org/pdf/2506.19022v1](http://arxiv.org/pdf/2506.19022v1)**

> **作者:** Jinlong Li; Dong Zhao; Qi Zang; Zequn Jie; Lin Ma; Nicu Sebe
>
> **摘要:** Continual Test Time Adaptation (CTTA) is a task that requires a source pre-trained model to continually adapt to new scenarios with changing target distributions. Existing CTTA methods primarily focus on mitigating the challenges of catastrophic forgetting and error accumulation. Though there have been emerging methods based on forgetting adaptation with parameter-efficient fine-tuning, they still struggle to balance competitive performance and efficient model adaptation, particularly in complex tasks like semantic segmentation. In this paper, to tackle the above issues, we propose a novel pipeline, Orthogonal Projection Subspace to aggregate online Prior-knowledge, dubbed OoPk. Specifically, we first project a tuning subspace orthogonally which allows the model to adapt to new domains while preserving the knowledge integrity of the pre-trained source model to alleviate catastrophic forgetting. Then, we elaborate an online prior-knowledge aggregation strategy that employs an aggressive yet efficient image masking strategy to mimic potential target dynamism, enhancing the student model's domain adaptability. This further gradually ameliorates the teacher model's knowledge, ensuring high-quality pseudo labels and reducing error accumulation. We demonstrate our method with extensive experiments that surpass previous CTTA methods and achieve competitive performances across various continual TTA benchmarks in semantic segmentation tasks.
>
---
#### [new 017] Ancient Script Image Recognition and Processing: A Review
- **分类: cs.CV**

- **简介: 该论文属于古代文字图像识别任务，旨在解决古文字识别中的挑战，如数据不平衡和图像退化。工作包括分类研究、分析方法并探讨解决方案。**

- **链接: [http://arxiv.org/pdf/2506.19208v1](http://arxiv.org/pdf/2506.19208v1)**

> **作者:** Xiaolei Diao; Rite Bo; Yanling Xiao; Lida Shi; Zhihan Zhou; Hao Xu; Chuntao Li; Xiongfeng Tang; Massimo Poesio; Cédric M. John; Daqian Shi
>
> **摘要:** Ancient scripts, e.g., Egyptian hieroglyphs, Oracle Bone Inscriptions, and Ancient Greek inscriptions, serve as vital carriers of human civilization, embedding invaluable historical and cultural information. Automating ancient script image recognition has gained importance, enabling large-scale interpretation and advancing research in archaeology and digital humanities. With the rise of deep learning, this field has progressed rapidly, with numerous script-specific datasets and models proposed. While these scripts vary widely, spanning phonographic systems with limited glyphs to logographic systems with thousands of complex symbols, they share common challenges and methodological overlaps. Moreover, ancient scripts face unique challenges, including imbalanced data distribution and image degradation, which have driven the development of various dedicated methods. This survey provides a comprehensive review of ancient script image recognition methods. We begin by categorizing existing studies based on script types and analyzing respective recognition methods, highlighting both their differences and shared strategies. We then focus on challenges unique to ancient scripts, systematically examining their impact and reviewing recent solutions, including few-shot learning and noise-robust techniques. Finally, we summarize current limitations and outline promising future directions. Our goal is to offer a structured, forward-looking perspective to support ongoing advancements in the recognition, interpretation, and decipherment of ancient scripts.
>
---
#### [new 018] Genome-Anchored Foundation Model Embeddings Improve Molecular Prediction from Histology Images
- **分类: cs.CV**

- **简介: 该论文属于分子预测任务，旨在通过病理图像准确预测分子特征和患者预后。工作是提出PathLUPI模型，利用转录组信息提升WSI的分子预测性能。**

- **链接: [http://arxiv.org/pdf/2506.19681v1](http://arxiv.org/pdf/2506.19681v1)**

> **作者:** Cheng Jin; Fengtao Zhou; Yunfang Yu; Jiabo Ma; Yihui Wang; Yingxue Xu; Huajun Zhou; Hao Jiang; Luyang Luo; Luhui Mao; Zifan He; Xiuming Zhang; Jing Zhang; Ronald Chan; Herui Yao; Hao Chen
>
> **备注:** Under Review
>
> **摘要:** Precision oncology requires accurate molecular insights, yet obtaining these directly from genomics is costly and time-consuming for broad clinical use. Predicting complex molecular features and patient prognosis directly from routine whole-slide images (WSI) remains a major challenge for current deep learning methods. Here we introduce PathLUPI, which uses transcriptomic privileged information during training to extract genome-anchored histological embeddings, enabling effective molecular prediction using only WSIs at inference. Through extensive evaluation across 49 molecular oncology tasks using 11,257 cases among 20 cohorts, PathLUPI demonstrated superior performance compared to conventional methods trained solely on WSIs. Crucially, it achieves AUC $\geq$ 0.80 in 14 of the biomarker prediction and molecular subtyping tasks and C-index $\geq$ 0.70 in survival cohorts of 5 major cancer types. Moreover, PathLUPI embeddings reveal distinct cellular morphological signatures associated with specific genotypes and related biological pathways within WSIs. By effectively encoding molecular context to refine WSI representations, PathLUPI overcomes a key limitation of existing models and offers a novel strategy to bridge molecular insights with routine pathology workflows for wider clinical application.
>
---
#### [new 019] Correspondence-Free Multiview Point Cloud Registration via Depth-Guided Joint Optimisation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多视角点云配准任务，旨在解决复杂环境中特征提取与数据关联困难的问题。提出一种无需对应关系的方法，通过深度图联合优化估计点云位姿和全局地图。**

- **链接: [http://arxiv.org/pdf/2506.18922v1](http://arxiv.org/pdf/2506.18922v1)**

> **作者:** Yiran Zhou; Yingyu Wang; Shoudong Huang; Liang Zhao
>
> **备注:** 8 pages, accepted for publication in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Multiview point cloud registration is a fundamental task for constructing globally consistent 3D models. Existing approaches typically rely on feature extraction and data association across multiple point clouds; however, these processes are challenging to obtain global optimal solution in complex environments. In this paper, we introduce a novel correspondence-free multiview point cloud registration method. Specifically, we represent the global map as a depth map and leverage raw depth information to formulate a non-linear least squares optimisation that jointly estimates poses of point clouds and the global map. Unlike traditional feature-based bundle adjustment methods, which rely on explicit feature extraction and data association, our method bypasses these challenges by associating multi-frame point clouds with a global depth map through their corresponding poses. This data association is implicitly incorporated and dynamically refined during the optimisation process. Extensive evaluations on real-world datasets demonstrate that our method outperforms state-of-the-art approaches in accuracy, particularly in challenging environments where feature extraction and data association are difficult.
>
---
#### [new 020] Da Yu: Towards USV-Based Image Captioning for Waterway Surveillance and Scene Understanding
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于图像描述生成任务，旨在解决水道环境的语义理解问题。通过构建WaterCaption数据集和提出Da Yu模型，提升USV的场景认知能力。**

- **链接: [http://arxiv.org/pdf/2506.19288v1](http://arxiv.org/pdf/2506.19288v1)**

> **作者:** Runwei Guan; Ningwei Ouyang; Tianhao Xu; Shaofeng Liang; Wei Dai; Yafeng Sun; Shang Gao; Songning Lai; Shanliang Yao; Xuming Hu; Ryan Wen Liu; Yutao Yue; Hui Xiong
>
> **备注:** 14 pages, 13 figures
>
> **摘要:** Automated waterway environment perception is crucial for enabling unmanned surface vessels (USVs) to understand their surroundings and make informed decisions. Most existing waterway perception models primarily focus on instance-level object perception paradigms (e.g., detection, segmentation). However, due to the complexity of waterway environments, current perception datasets and models fail to achieve global semantic understanding of waterways, limiting large-scale monitoring and structured log generation. With the advancement of vision-language models (VLMs), we leverage image captioning to introduce WaterCaption, the first captioning dataset specifically designed for waterway environments. WaterCaption focuses on fine-grained, multi-region long-text descriptions, providing a new research direction for visual geo-understanding and spatial scene cognition. Exactly, it includes 20.2k image-text pair data with 1.8 million vocabulary size. Additionally, we propose Da Yu, an edge-deployable multi-modal large language model for USVs, where we propose a novel vision-to-language projector called Nano Transformer Adaptor (NTA). NTA effectively balances computational efficiency with the capacity for both global and fine-grained local modeling of visual features, thereby significantly enhancing the model's ability to generate long-form textual outputs. Da Yu achieves an optimal balance between performance and efficiency, surpassing state-of-the-art models on WaterCaption and several other captioning benchmarks.
>
---
#### [new 021] CoCo4D: Comprehensive and Complex 4D Scene Generation
- **分类: cs.CV**

- **简介: 该论文属于4D场景生成任务，旨在解决多视角一致性和动态场景生成问题。提出CoCo4D框架，分离处理动态前景与背景，提升生成效果。**

- **链接: [http://arxiv.org/pdf/2506.19798v1](http://arxiv.org/pdf/2506.19798v1)**

> **作者:** Junwei Zhou; Xueting Li; Lu Qi; Ming-Hsuan Yang
>
> **备注:** 16 pages,10 figures
>
> **摘要:** Existing 4D synthesis methods primarily focus on object-level generation or dynamic scene synthesis with limited novel views, restricting their ability to generate multi-view consistent and immersive dynamic 4D scenes. To address these constraints, we propose a framework (dubbed as CoCo4D) for generating detailed dynamic 4D scenes from text prompts, with the option to include images. Our method leverages the crucial observation that articulated motion typically characterizes foreground objects, whereas background alterations are less pronounced. Consequently, CoCo4D divides 4D scene synthesis into two responsibilities: modeling the dynamic foreground and creating the evolving background, both directed by a reference motion sequence. Given a text prompt and an optional reference image, CoCo4D first generates an initial motion sequence utilizing video diffusion models. This motion sequence then guides the synthesis of both the dynamic foreground object and the background using a novel progressive outpainting scheme. To ensure seamless integration of the moving foreground object within the dynamic background, CoCo4D optimizes a parametric trajectory for the foreground, resulting in realistic and coherent blending. Extensive experiments show that CoCo4D achieves comparable or superior performance in 4D scene generation compared to existing methods, demonstrating its effectiveness and efficiency. More results are presented on our website https://colezwhy.github.io/coco4d/.
>
---
#### [new 022] 3D-SSM: A Novel 3D Selective Scan Module for Remote Sensing Change Detection
- **分类: cs.CV**

- **简介: 该论文属于遥感变化检测任务，旨在解决现有方法难以捕捉图像通道间长距离依赖的问题。提出3D-SSM模块及两个关键组件，提升特征表示与变化检测性能。**

- **链接: [http://arxiv.org/pdf/2506.19263v1](http://arxiv.org/pdf/2506.19263v1)**

> **作者:** Rui Huang; Jincheng Zeng; Sen Gao; Yan Xing
>
> **摘要:** Existing Mamba-based approaches in remote sensing change detection have enhanced scanning models, yet remain limited by their inability to capture long-range dependencies between image channels effectively, which restricts their feature representation capabilities. To address this limitation, we propose a 3D selective scan module (3D-SSM) that captures global information from both the spatial plane and channel perspectives, enabling a more comprehensive understanding of the data.Based on the 3D-SSM, we present two key components: a spatiotemporal interaction module (SIM) and a multi-branch feature extraction module (MBFEM). The SIM facilitates bi-temporal feature integration by enabling interactions between global and local features across images from different time points, thereby enhancing the detection of subtle changes. Meanwhile, the MBFEM combines features from the frequency domain, spatial domain, and 3D-SSM to provide a rich representation of contextual information within the image. Our proposed method demonstrates favourable performance compared to state-of-the-art change detection methods on five benchmark datasets through extensive experiments. Code is available at https://github.com/VerdantMist/3D-SSM
>
---
#### [new 023] Comparative Performance of Finetuned ImageNet Pre-trained Models for Electronic Component Classification
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决电子元件分类问题。通过比较12种ImageNet预训练模型，验证其在该任务中的性能表现。**

- **链接: [http://arxiv.org/pdf/2506.19330v1](http://arxiv.org/pdf/2506.19330v1)**

> **作者:** Yidi Shao; Longfei Zhou; Fangshuo Tang; Xinyi Shi; Dalang Chen; Shengtao Xia
>
> **备注:** This is the author's version of the accepted paper. The final version will appear in IEEE UV 2024
>
> **摘要:** Electronic component classification and detection are crucial in manufacturing industries, significantly reducing labor costs and promoting technological and industrial development. Pre-trained models, especially those trained on ImageNet, are highly effective in image classification, allowing researchers to achieve excellent results even with limited data. This paper compares the performance of twelve ImageNet pre-trained models in classifying electronic components. Our findings show that all models tested delivered respectable accuracies. MobileNet-V2 recorded the highest at 99.95%, while EfficientNet-B0 had the lowest at 92.26%. These results underscore the substantial benefits of using ImageNet pre-trained models in image classification tasks and confirm the practical applicability of these methods in the electronics manufacturing sector.
>
---
#### [new 024] Diffusion Transformer-to-Mamba Distillation for High-Resolution Image Generation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决高分辨率生成中计算成本高的问题。通过知识蒸馏将Transformer迁移到Mamba模型，提升效率并保持质量。**

- **链接: [http://arxiv.org/pdf/2506.18999v1](http://arxiv.org/pdf/2506.18999v1)**

> **作者:** Yuan Yao; Yicong Hong; Difan Liu; Long Mai; Feng Liu; Jiebo Luo
>
> **摘要:** The quadratic computational complexity of self-attention in diffusion transformers (DiT) introduces substantial computational costs in high-resolution image generation. While the linear-complexity Mamba model emerges as a potential alternative, direct Mamba training remains empirically challenging. To address this issue, this paper introduces diffusion transformer-to-mamba distillation (T2MD), forming an efficient training pipeline that facilitates the transition from the self-attention-based transformer to the linear complexity state-space model Mamba. We establish a diffusion self-attention and Mamba hybrid model that simultaneously achieves efficiency and global dependencies. With the proposed layer-level teacher forcing and feature-based knowledge distillation, T2MD alleviates the training difficulty and high cost of a state space model from scratch. Starting from the distilled 512$\times$512 resolution base model, we push the generation towards 2048$\times$2048 images via lightweight adaptation and high-resolution fine-tuning. Experiments demonstrate that our training path leads to low overhead but high-quality text-to-image generation. Importantly, our results also justify the feasibility of using sequential and causal Mamba models for generating non-causal visual output, suggesting the potential for future exploration.
>
---
#### [new 025] MSR-Align: Policy-Grounded Multimodal Alignment for Safety-Aware Reasoning in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型的安全对齐任务，旨在解决多模态输入带来的安全风险。工作包括构建MSR-Align数据集，提升模型对恶意提示的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.19257v1](http://arxiv.org/pdf/2506.19257v1)**

> **作者:** Yinan Xia; Yilei Jiang; Yingshui Tan; Xiaoyong Zhu; Xiangyu Yue; Bo Zheng
>
> **摘要:** Vision-Language Models (VLMs) have achieved remarkable progress in multimodal reasoning tasks through enhanced chain-of-thought capabilities. However, this advancement also introduces novel safety risks, as these models become increasingly vulnerable to harmful multimodal prompts that can trigger unethical or unsafe behaviors. Existing safety alignment approaches, primarily designed for unimodal language models, fall short in addressing the complex and nuanced threats posed by multimodal inputs. Moreover, current safety datasets lack the fine-grained, policy-grounded reasoning required to robustly align reasoning-capable VLMs. In this work, we introduce {MSR-Align}, a high-quality Multimodal Safety Reasoning dataset tailored to bridge this gap. MSR-Align supports fine-grained, deliberative reasoning over standardized safety policies across both vision and text modalities. Our data generation pipeline emphasizes multimodal diversity, policy-grounded reasoning, and rigorous quality filtering using strong multimodal judges. Extensive experiments demonstrate that fine-tuning VLMs on MSR-Align substantially improves robustness against both textual and vision-language jailbreak attacks, while preserving or enhancing general reasoning performance. MSR-Align provides a scalable and effective foundation for advancing the safety alignment of reasoning-capable VLMs. Our dataset is made publicly available at https://huggingface.co/datasets/Leigest/MSR-Align.
>
---
#### [new 026] Recurrent Visual Feature Extraction and Stereo Attentions for CT Report Generation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医学图像报告生成任务，旨在解决CT图像与文本对齐及特征融合问题。提出基于循环视觉特征提取和立体注意力机制的方法，提升报告生成效果。**

- **链接: [http://arxiv.org/pdf/2506.19665v1](http://arxiv.org/pdf/2506.19665v1)**

> **作者:** Yuanhe Tian; Lei Mao; Yan Song
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Generating reports for computed tomography (CT) images is a challenging task, while similar to existing studies for medical image report generation, yet has its unique characteristics, such as spatial encoding of multiple images, alignment between image volume and texts, etc. Existing solutions typically use general 2D or 3D image processing techniques to extract features from a CT volume, where they firstly compress the volume and then divide the compressed CT slices into patches for visual encoding. These approaches do not explicitly account for the transformations among CT slices, nor do they effectively integrate multi-level image features, particularly those containing specific organ lesions, to instruct CT report generation (CTRG). In considering the strong correlation among consecutive slices in CT scans, in this paper, we propose a large language model (LLM) based CTRG method with recurrent visual feature extraction and stereo attentions for hierarchical feature modeling. Specifically, we use a vision Transformer to recurrently process each slice in a CT volume, and employ a set of attentions over the encoded slices from different perspectives to selectively obtain important visual information and align them with textual features, so as to better instruct an LLM for CTRG. Experiment results and further analysis on the benchmark M3D-Cap dataset show that our method outperforms strong baseline models and achieves state-of-the-art results, demonstrating its validity and effectiveness.
>
---
#### [new 027] Connecting Vision and Emissions: A Behavioural AI Approach to Carbon Estimation in Road Design
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于智能交通任务，旨在通过行为AI估算道路设计中的碳排放。工作包括改进YOLOv8进行车辆检测与分类，并结合OCR识别车牌以计算碳排放。**

- **链接: [http://arxiv.org/pdf/2506.18924v1](http://arxiv.org/pdf/2506.18924v1)**

> **作者:** Ammar K Al Mhdawi; Nonso Nnamoko; Safanah Mudheher Raafat; M. K. S. Al-Mhdawi; Amjad J Humaidi
>
> **摘要:** We present an enhanced YOLOv8 real time vehicle detection and classification framework, for estimating carbon emissions in urban environments. The system enhances YOLOv8 architecture to detect, segment, and track vehicles from live traffic video streams. Once a vehicle is localized, a dedicated deep learning-based identification module is employed to recognize license plates and classify vehicle types. Since YOLOv8 lacks the built-in capacity for fine grained recognition tasks such as reading license plates or determining vehicle attributes beyond class labels, our framework incorporates a hybrid pipeline where each detected vehicle is tracked and its bounding box is cropped and passed to a deep Optical Character Recognition (OCR) module. This OCR system, composed of multiple convolutional neural network (CNN) layers, is trained specifically for character-level detection and license plate decoding under varied conditions such as motion blur, occlusion, and diverse font styles. Additionally, the recognized plate information is validated using a real time API that cross references with an external vehicle registration database to ensure accurate classification and emission estimation. This multi-stage approach enables precise, automated calculation of per vehicle carbon emissions. Extensive evaluation was conducted using a diverse vehicle dataset enriched with segmentation masks and annotated license plates. The YOLOv8 detector achieved a mean Average Precision (mAP@0.5) of approximately 71% for bounding boxes and 70% for segmentation masks. Character level OCR accuracy reached up to 99% with the best performing CNN model. These results affirm the feasibility of combining real time object detection with deep OCR for practical deployment in smart transportation systems, offering a scalable solution for automated, vehicle specific carbon emission monitoring.
>
---
#### [new 028] GLIMPSE: Gradient-Layer Importance Mapping for Prompted Visual Saliency Explanation for Generative LVLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型解释任务，旨在解决LVLM在生成文本时的视觉注意力解释问题。提出GLIMPSE框架，通过融合梯度加权和多层传播，生成跨模态的可解释热图。**

- **链接: [http://arxiv.org/pdf/2506.18985v1](http://arxiv.org/pdf/2506.18985v1)**

> **作者:** Guanxi Shen
>
> **摘要:** Recent advances in large vision language models (LVLMs) have unlocked unprecedented capabilities in generating coherent responses from visual inputs. However, interpreting where LVLMs direct their visual attention while generating free-form textual responses remains a significant challenge, yet is essential for understanding model behavior, diagnosing hallucination, exposing bias and ensuring transparency. We introduce GLIMPSE (Gradient-Layer Importance Mapping for Prompted Visual Saliency Explanation), a lightweight, model-agnostic framework for visualizing the salient image regions that LVLMs rely upon during open-ended visual question answering (VQA), while concurrently revealing the multimodal textual saliency. GLIMPSE fuses gradient-weighted attention, adaptive layer propagation, and weighted token aggregation to produce holistic response-level attribution heat maps for interpreting cross-modal reasoning, outperforming prior interpretability methods in human-alignment. We demonstrate an analytic explainable AI (XAI) approach using GLIMPSE to uncover fine-grained insights into LVLM cross-modal attribution, trace token-level reasoning dynamics, and analyze systematic human-attention misalignment, hallucination, and bias.
>
---
#### [new 029] Bird's-eye view safety monitoring for the construction top under the tower crane
- **分类: cs.CV; cs.SY; eess.SY**

- **简介: 该论文属于建筑安全监控任务，旨在解决塔吊作业中人员与构件碰撞风险。通过融合摄像头和LiDAR数据，实现三维定位与实时预警。**

- **链接: [http://arxiv.org/pdf/2506.18938v1](http://arxiv.org/pdf/2506.18938v1)**

> **作者:** Yanke Wang; Yu Hin Ng; Haobo Liang; Ching-Wei Chang; Hao Chen
>
> **摘要:** The tower crane is involving more automated and intelligent operation procedure, and importantly, the application of automation technologies to the safety issues is imperative ahead of the utilization of any other advances. Among diverse risk management tasks on site, it is essential to protect the human workers on the workspace between the tower crane and constructed building top area (construction top) from the bird's-eye view, especially with Modular Integrated Construction (MiC) lifted. Also, the camera and Light Detection And Ranging (LiDAR) can capture abundant 3D information on site, which is however yet made the best use. Considering the safety protection for humans and tower cranes, we present an AI-based fully automated safety monitoring system for tower crane lifting from the bird's-eye view, surveilling to shield the human workers on the construction top and avoid cranes' collision by alarming the crane operator. The system achieved a 3D data fusion for localization of humans and MiCs by integrating the captured information from camera and LiDAR. The state-of-the-art methods were explored and implemented into our proposed software pipeline coupled with the hardware and display systems. Furthermore, we conducted an analysis of the components in the pipeline to verify the accuracy and effectiveness of the involved methods. The display and visualization on the real site proved that our system can serve as a valuable safety monitoring toolkit on site.
>
---
#### [new 030] PEVLM: Parallel Encoding for Vision-Language Models
- **分类: cs.CV; cs.LG; cs.PF**

- **简介: 该论文属于视频语言模型任务，解决长视频理解中的计算效率问题。提出PEVLM方法，通过并行编码提升预填充效率，减少注意力计算复杂度，提升准确率与速度。**

- **链接: [http://arxiv.org/pdf/2506.19651v1](http://arxiv.org/pdf/2506.19651v1)**

> **作者:** Letian Kang; Shixian Luo; Yiqiang Li; Xiaoyang Yu; Shenxuan Zhou; Yong Wu
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated strong performance in video-language tasks, yet their application to long video understanding remains constrained by the quadratic complexity of standard attention mechanisms. In this paper, we propose \textbf{PEVLM}, a parallel encoding strategy specifically designed to improve the prefill efficiency of VLMs without requiring model finetuning. PEVLM partitions the input into block-wise segments with a shared sink, preserves full-attention positional embeddings, and aligns attention weights to mimic full-attention distributions. This design reduces attention computation from $O((T \times N)^2)$ to $O(T \times N)$ while maintaining high accuracy. Extensive experiments on the LongVideoBench benchmark show that PEVLM achieves up to 8.37\% accuracy improvement over existing inference-efficient methods and delivers up to 7.47x speedup in attention computation and 40\% reduction in end-to-end latency. Under strict latency constraints, PEVLM significantly outperforms baselines, raising accuracy from 23.26\% to 61.03\%. These results highlight PEVLM's effectiveness for low-latency, long-context video understanding, making it well-suited for real-world applications such as autonomous driving.
>
---
#### [new 031] Automated Image Recognition Framework
- **分类: cs.CV**

- **简介: 该论文属于图像识别任务，旨在解决数据收集与标注困难的问题。提出AIR框架，通过生成和增强数据提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.19261v1](http://arxiv.org/pdf/2506.19261v1)**

> **作者:** Quang-Binh Nguyen; Trong-Vu Hoang; Ngoc-Do Tran; Tam V. Nguyen; Minh-Triet Tran; Trung-Nghia Le
>
> **备注:** ICCCI 2025
>
> **摘要:** While the efficacy of deep learning models heavily relies on data, gathering and annotating data for specific tasks, particularly when addressing novel or sensitive subjects lacking relevant datasets, poses significant time and resource challenges. In response to this, we propose a novel Automated Image Recognition (AIR) framework that harnesses the power of generative AI. AIR empowers end-users to synthesize high-quality, pre-annotated datasets, eliminating the necessity for manual labeling. It also automatically trains deep learning models on the generated datasets with robust image recognition performance. Our framework includes two main data synthesis processes, AIR-Gen and AIR-Aug. The AIR-Gen enables end-users to seamlessly generate datasets tailored to their specifications. To improve image quality, we introduce a novel automated prompt engineering module that leverages the capabilities of large language models. We also introduce a distribution adjustment algorithm to eliminate duplicates and outliers, enhancing the robustness and reliability of generated datasets. On the other hand, the AIR-Aug enhances a given dataset, thereby improving the performance of deep classifier models. AIR-Aug is particularly beneficial when users have limited data for specific tasks. Through comprehensive experiments, we demonstrated the efficacy of our generated data in training deep learning models and showcased the system's potential to provide image recognition models for a wide range of objects. We also conducted a user study that achieved an impressive score of 4.4 out of 5.0, underscoring the AI community's positive perception of AIR.
>
---
#### [new 032] A Comparative Study of NAFNet Baselines for Image Restoration
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像修复任务，研究NAFNet模型的组件影响。通过实验分析不同结构对恢复效果的影响，验证了SimpleGate和LayerNorm的有效性。**

- **链接: [http://arxiv.org/pdf/2506.19845v1](http://arxiv.org/pdf/2506.19845v1)**

> **作者:** Vladislav Esaulov; M. Moein Esfahani
>
> **摘要:** We study NAFNet (Nonlinear Activation Free Network), a simple and efficient deep learning baseline for image restoration. By using CIFAR10 images corrupted with noise and blur, we conduct an ablation study of NAFNet's core components. Our baseline model implements SimpleGate activation, Simplified Channel Activation (SCA), and LayerNormalization. We compare this baseline to different variants that replace or remove components. Quantitative results (PSNR, SSIM) and examples illustrate how each modification affects restoration performance. Our findings support the NAFNet design: the SimpleGate and simplified attention mechanisms yield better results than conventional activations and attention, while LayerNorm proves to be important for stable training. We conclude with recommendations for model design, discuss potential improvements, and future work.
>
---
#### [new 033] Continual Retinal Vision-Language Pre-training upon Incremental Imaging Modalities
- **分类: cs.CV**

- **简介: 该论文属于医学图像与自然语言处理交叉任务，旨在解决多模态眼底图像分析中模型泛化与遗忘问题，提出RetCoP框架实现持续预训练。**

- **链接: [http://arxiv.org/pdf/2506.19320v1](http://arxiv.org/pdf/2506.19320v1)**

> **作者:** Yuang Yao; Ruiqi Wu; Yi Zhou; Tao Zhou
>
> **备注:** Accepted by MICCAI 2025
>
> **摘要:** Traditional fundus image analysis models focus on single-modal tasks, ignoring fundus modality complementarity, which limits their versatility. Recently, retinal foundation models have emerged, but most still remain modality-specific. Integrating multiple fundus imaging modalities into a single foundation model is valuable. However, in dynamic environments, data from different modalities often arrive incrementally, necessitating continual pre-training. To address this, we propose RetCoP, the first continual vision-language pre-training framework in the fundus domain, which incrementally integrates image and text features from different imaging modalities into a single unified foundation model. To mitigate catastrophic forgetting in continual pre-training, we introduce a rehearsal strategy utilizing representative image-text pairs and an off-diagonal information distillation approach. The former allows the model to revisit knowledge from previous stages, while the latter explicitly preserves the alignment between image and text representations. Experiments show that RetCoP outperforms all the compared methods, achieving the best generalization and lowest forgetting rate. The code can be found at https://github.com/Yuang-Yao/RetCoP.
>
---
#### [new 034] Surgery-R1: Advancing Surgical-VQLA with Reasoning Multimodal Large Language Model via Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于手术场景理解任务，解决Surgical-VQLA模型缺乏深度推理和可解释性的问题。通过构建数据集并提出Surgery-R1模型，提升其推理能力与临床适用性。**

- **链接: [http://arxiv.org/pdf/2506.19469v1](http://arxiv.org/pdf/2506.19469v1)**

> **作者:** Pengfei Hao; Shuaibo Li; Hongqiu Wang; Zhizhuo Kou; Junhang Zhang; Guang Yang; Lei Zhu
>
> **摘要:** In recent years, significant progress has been made in the field of surgical scene understanding, particularly in the task of Visual Question Localized-Answering in robotic surgery (Surgical-VQLA). However, existing Surgical-VQLA models lack deep reasoning capabilities and interpretability in surgical scenes, which limits their reliability and potential for development in clinical applications. To address this issue, inspired by the development of Reasoning Multimodal Large Language Models (MLLMs), we first build the Surgery-R1-54k dataset, including paired data for Visual-QA, Grounding-QA, and Chain-of-Thought (CoT). Then, we propose the first Reasoning MLLM for Surgical-VQLA (Surgery-R1). In our Surgery-R1, we design a two-stage fine-tuning mechanism to enable the basic MLLM with complex reasoning abilities by utilizing supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT). Furthermore, for an efficient and high-quality rule-based reward system in our RFT, we design a Multimodal Coherence reward mechanism to mitigate positional illusions that may arise in surgical scenarios. Experiment results demonstrate that Surgery-R1 outperforms other existing state-of-the-art (SOTA) models in the Surgical-VQLA task and widely-used MLLMs, while also validating its reasoning capabilities and the effectiveness of our approach. The code and dataset will be organized in https://github.com/FiFi-HAO467/Surgery-R1.
>
---
#### [new 035] DiffRIS: Enhancing Referring Remote Sensing Image Segmentation with Pre-trained Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感图像分割任务，旨在解决自然语言描述与遥感图像之间语义对齐的问题。提出DiffRIS框架，利用预训练扩散模型提升分割精度。**

- **链接: [http://arxiv.org/pdf/2506.18946v1](http://arxiv.org/pdf/2506.18946v1)**

> **作者:** Zhe Dong; Yuzhe Sun; Tianzhu Liu; Yanfeng Gu
>
> **摘要:** Referring remote sensing image segmentation (RRSIS) enables the precise delineation of regions within remote sensing imagery through natural language descriptions, serving critical applications in disaster response, urban development, and environmental monitoring. Despite recent advances, current approaches face significant challenges in processing aerial imagery due to complex object characteristics including scale variations, diverse orientations, and semantic ambiguities inherent to the overhead perspective. To address these limitations, we propose DiffRIS, a novel framework that harnesses the semantic understanding capabilities of pre-trained text-to-image diffusion models for enhanced cross-modal alignment in RRSIS tasks. Our framework introduces two key innovations: a context perception adapter (CP-adapter) that dynamically refines linguistic features through global context modeling and object-aware reasoning, and a progressive cross-modal reasoning decoder (PCMRD) that iteratively aligns textual descriptions with visual regions for precise segmentation. The CP-adapter bridges the domain gap between general vision-language understanding and remote sensing applications, while PCMRD enables fine-grained semantic alignment through multi-scale feature interaction. Comprehensive experiments on three benchmark datasets-RRSIS-D, RefSegRS, and RISBench-demonstrate that DiffRIS consistently outperforms existing methods across all standard metrics, establishing a new state-of-the-art for RRSIS tasks. The significant performance improvements validate the effectiveness of leveraging pre-trained diffusion models for remote sensing applications through our proposed adaptive framework.
>
---
#### [new 036] HAWAII: Hierarchical Visual Knowledge Transfer for Efficient Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉-语言模型任务，旨在提升模型的视觉理解能力。针对多视觉专家训练成本高的问题，提出HAWAII框架，通过知识蒸馏和LoRA适配器高效融合多专家知识。**

- **链接: [http://arxiv.org/pdf/2506.19072v1](http://arxiv.org/pdf/2506.19072v1)**

> **作者:** Yimu Wang; Mozhgan Nasr Azadani; Sean Sedwards; Krzysztof Czarnecki
>
> **备注:** Work in progress
>
> **摘要:** Improving the visual understanding ability of vision-language models (VLMs) is crucial for enhancing their performance across various tasks. While using multiple pretrained visual experts has shown great promise, it often incurs significant computational costs during training and inference. To address this challenge, we propose HAWAII, a novel framework that distills knowledge from multiple visual experts into a single vision encoder, enabling it to inherit the complementary strengths of several experts with minimal computational overhead. To mitigate conflicts among different teachers and switch between different teacher-specific knowledge, instead of using a fixed set of adapters for multiple teachers, we propose to use teacher-specific Low-Rank Adaptation (LoRA) adapters with a corresponding router. Each adapter is aligned with a specific teacher, avoiding noisy guidance during distillation. To enable efficient knowledge distillation, we propose fine-grained and coarse-grained distillation. At the fine-grained level, token importance scores are employed to emphasize the most informative tokens from each teacher adaptively. At the coarse-grained level, we summarize the knowledge from multiple teachers and transfer it to the student using a set of general-knowledge LoRA adapters with a router. Extensive experiments on various vision-language tasks demonstrate the superiority of HAWAII, compared to the popular open-source VLMs.
>
---
#### [new 037] Inverse-and-Edit: Effective and Fast Image Editing by Cycle Consistency Models
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，解决扩散模型计算量大、编辑能力弱的问题，提出基于一致性模型的快速高效编辑方法。**

- **链接: [http://arxiv.org/pdf/2506.19103v1](http://arxiv.org/pdf/2506.19103v1)**

> **作者:** Ilia Beletskii; Andrey Kuznetsov; Aibek Alanov
>
> **备注:** The code of our method is available on GitHub at https://github.com/ControlGenAI/Inverse-and-Edit
>
> **摘要:** Recent advances in image editing with diffusion models have achieved impressive results, offering fine-grained control over the generation process. However, these methods are computationally intensive because of their iterative nature. While distilled diffusion models enable faster inference, their editing capabilities remain limited, primarily because of poor inversion quality. High-fidelity inversion and reconstruction are essential for precise image editing, as they preserve the structural and semantic integrity of the source image. In this work, we propose a novel framework that enhances image inversion using consistency models, enabling high-quality editing in just four steps. Our method introduces a cycle-consistency optimization strategy that significantly improves reconstruction accuracy and enables a controllable trade-off between editability and content preservation. We achieve state-of-the-art performance across various image editing tasks and datasets, demonstrating that our method matches or surpasses full-step diffusion models while being substantially more efficient. The code of our method is available on GitHub at https://github.com/ControlGenAI/Inverse-and-Edit.
>
---
#### [new 038] Training-Free Motion Customization for Distilled Video Generators with Adaptive Test-Time Distillation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决训练-free 设置下运动定制难题。提出MotionEcho框架，通过扩散教师强制提升生成质量与运动保真度。**

- **链接: [http://arxiv.org/pdf/2506.19348v1](http://arxiv.org/pdf/2506.19348v1)**

> **作者:** Jintao Rong; Xin Xie; Xinyi Yu; Linlin Ou; Xinyu Zhang; Chunhua Shen; Dong Gong
>
> **摘要:** Distilled video generation models offer fast and efficient synthesis but struggle with motion customization when guided by reference videos, especially under training-free settings. Existing training-free methods, originally designed for standard diffusion models, fail to generalize due to the accelerated generative process and large denoising steps in distilled models. To address this, we propose MotionEcho, a novel training-free test-time distillation framework that enables motion customization by leveraging diffusion teacher forcing. Our approach uses high-quality, slow teacher models to guide the inference of fast student models through endpoint prediction and interpolation. To maintain efficiency, we dynamically allocate computation across timesteps according to guidance needs. Extensive experiments across various distilled video generation models and benchmark datasets demonstrate that our method significantly improves motion fidelity and generation quality while preserving high efficiency. Project page: https://euminds.github.io/motionecho/
>
---
#### [new 039] Bind-Your-Avatar: Multi-Talking-Character Video Generation with Dynamic 3D-mask-based Embedding Router
- **分类: cs.CV**

- **简介: 该论文属于多角色对话视频生成任务，解决多角色音频对应控制与缺乏数据集的问题，提出新框架、3D掩码方法及首个相关数据集。**

- **链接: [http://arxiv.org/pdf/2506.19833v1](http://arxiv.org/pdf/2506.19833v1)**

> **作者:** Yubo Huang; Weiqiang Wang; Sirui Zhao; Tong Xu; Lin Liu; Enhong Chen
>
> **摘要:** Recent years have witnessed remarkable advances in audio-driven talking head generation. However, existing approaches predominantly focus on single-character scenarios. While some methods can create separate conversation videos between two individuals, the critical challenge of generating unified conversation videos with multiple physically co-present characters sharing the same spatial environment remains largely unaddressed. This setting presents two key challenges: audio-to-character correspondence control and the lack of suitable datasets featuring multi-character talking videos within the same scene. To address these challenges, we introduce Bind-Your-Avatar, an MM-DiT-based model specifically designed for multi-talking-character video generation in the same scene. Specifically, we propose (1) A novel framework incorporating a fine-grained Embedding Router that binds `who' and `speak what' together to address the audio-to-character correspondence control. (2) Two methods for implementing a 3D-mask embedding router that enables frame-wise, fine-grained control of individual characters, with distinct loss functions based on observed geometric priors and a mask refinement strategy to enhance the accuracy and temporal smoothness of the predicted masks. (3) The first dataset, to the best of our knowledge, specifically constructed for multi-talking-character video generation, and accompanied by an open-source data processing pipeline, and (4) A benchmark for the dual-talking-characters video generation, with extensive experiments demonstrating superior performance over multiple state-of-the-art methods.
>
---
#### [new 040] AnimaX: Animating the Inanimate in 3D with Joint Video-Pose Diffusion Models
- **分类: cs.CV**

- **简介: 该论文提出AnimaX，解决3D动画生成问题。通过结合视频扩散模型与骨骼动画，实现多样化的3D物体动画生成。**

- **链接: [http://arxiv.org/pdf/2506.19851v1](http://arxiv.org/pdf/2506.19851v1)**

> **作者:** Zehuan Huang; Haoran Feng; Yangtian Sun; Yuanchen Guo; Yanpei Cao; Lu Sheng
>
> **备注:** Project page: https://anima-x.github.io/
>
> **摘要:** We present AnimaX, a feed-forward 3D animation framework that bridges the motion priors of video diffusion models with the controllable structure of skeleton-based animation. Traditional motion synthesis methods are either restricted to fixed skeletal topologies or require costly optimization in high-dimensional deformation spaces. In contrast, AnimaX effectively transfers video-based motion knowledge to the 3D domain, supporting diverse articulated meshes with arbitrary skeletons. Our method represents 3D motion as multi-view, multi-frame 2D pose maps, and enables joint video-pose diffusion conditioned on template renderings and a textual motion prompt. We introduce shared positional encodings and modality-aware embeddings to ensure spatial-temporal alignment between video and pose sequences, effectively transferring video priors to motion generation task. The resulting multi-view pose sequences are triangulated into 3D joint positions and converted into mesh animation via inverse kinematics. Trained on a newly curated dataset of 160,000 rigged sequences, AnimaX achieves state-of-the-art results on VBench in generalization, motion fidelity, and efficiency, offering a scalable solution for category-agnostic 3D animation. Project page: \href{https://anima-x.github.io/}{https://anima-x.github.io/}.
>
---
#### [new 041] Self-Supervised Multimodal NeRF for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的多模态场景重建任务，解决动态场景下无需3D标签的神经辐射场建模问题。提出自监督框架，结合LiDAR与相机数据，提升合成视图质量。**

- **链接: [http://arxiv.org/pdf/2506.19615v1](http://arxiv.org/pdf/2506.19615v1)**

> **作者:** Gaurav Sharma; Ravi Kothari; Josef Schmid
>
> **摘要:** In this paper, we propose a Neural Radiance Fields (NeRF) based framework, referred to as Novel View Synthesis Framework (NVSF). It jointly learns the implicit neural representation of space and time-varying scene for both LiDAR and Camera. We test this on a real-world autonomous driving scenario containing both static and dynamic scenes. Compared to existing multimodal dynamic NeRFs, our framework is self-supervised, thus eliminating the need for 3D labels. For efficient training and faster convergence, we introduce heuristic-based image pixel sampling to focus on pixels with rich information. To preserve the local features of LiDAR points, a Double Gradient based mask is employed. Extensive experiments on the KITTI-360 dataset show that, compared to the baseline models, our framework has reported best performance on both LiDAR and Camera domain. Code of the model is available at https://github.com/gaurav00700/Selfsupervised-NVSF
>
---
#### [new 042] Systematic Comparison of Projection Methods for Monocular 3D Human Pose Estimation on Fisheye Images
- **分类: cs.CV; cs.RO; I.2.10; I.2.9; I.4.8; I.4.9**

- **简介: 该论文属于单目3D人体姿态估计任务，旨在解决鱼眼图像中人体姿态准确检测的问题。通过系统比较不同投影方法，提出基于边界框的模型选择策略，并构建了新数据集FISHnCHIPS。**

- **链接: [http://arxiv.org/pdf/2506.19747v1](http://arxiv.org/pdf/2506.19747v1)**

> **作者:** Stephanie Käs; Sven Peter; Henrik Thillmann; Anton Burenko; David Benjamin Adrian; Dennis Mack; Timm Linder; Bastian Leibe
>
> **备注:** Presented at IEEE International Conference on Robotics and Automation 2025
>
> **摘要:** Fisheye cameras offer robots the ability to capture human movements across a wider field of view (FOV) than standard pinhole cameras, making them particularly useful for applications in human-robot interaction and automotive contexts. However, accurately detecting human poses in fisheye images is challenging due to the curved distortions inherent to fisheye optics. While various methods for undistorting fisheye images have been proposed, their effectiveness and limitations for poses that cover a wide FOV has not been systematically evaluated in the context of absolute human pose estimation from monocular fisheye images. To address this gap, we evaluate the impact of pinhole, equidistant and double sphere camera models, as well as cylindrical projection methods, on 3D human pose estimation accuracy. We find that in close-up scenarios, pinhole projection is inadequate, and the optimal projection method varies with the FOV covered by the human pose. The usage of advanced fisheye models like the double sphere model significantly enhances 3D human pose estimation accuracy. We propose a heuristic for selecting the appropriate projection model based on the detection bounding box to enhance prediction quality. Additionally, we introduce and evaluate on our novel dataset FISHnCHIPS, which features 3D human skeleton annotations in fisheye images, including images from unconventional angles, such as extreme close-ups, ground-mounted cameras, and wide-FOV poses, available at: https://www.vision.rwth-aachen.de/fishnchips
>
---
#### [new 043] Video-XL-2: Towards Very Long-Video Understanding Through Task-Aware KV Sparsification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频理解任务，旨在解决长视频处理中的高内存和计算成本问题。通过引入任务感知的KV稀疏化方法，提升模型效率与性能。**

- **链接: [http://arxiv.org/pdf/2506.19225v1](http://arxiv.org/pdf/2506.19225v1)**

> **作者:** Minghao Qin; Xiangrui Liu; Zhengyang Liang; Yan Shu; Huaying Yuan; Juenjie Zhou; Shitao Xiao; Bo Zhao; Zheng Liu
>
> **备注:** 12 pages, 5 Figure, 3 Table
>
> **摘要:** Multi-modal large language models (MLLMs) models have made significant progress in video understanding over the past few years. However, processing long video inputs remains a major challenge due to high memory and computational costs. This makes it difficult for current models to achieve both strong performance and high efficiency in long video understanding. To address this challenge, we propose Video-XL-2, a novel MLLM that delivers superior cost-effectiveness for long-video understanding based on task-aware KV sparsification. The proposed framework operates with two key steps: chunk-based pre-filling and bi-level key-value decoding. Chunk-based pre-filling divides the visual token sequence into chunks, applying full attention within each chunk and sparse attention across chunks. This significantly reduces computational and memory overhead. During decoding, bi-level key-value decoding selectively reloads either dense or sparse key-values for each chunk based on its relevance to the task. This approach further improves memory efficiency and enhances the model's ability to capture fine-grained information. Video-XL-2 achieves state-of-the-art performance on various long video understanding benchmarks, outperforming existing open-source lightweight models. It also demonstrates exceptional efficiency, capable of processing over 10,000 frames on a single NVIDIA A100 (80GB) GPU and thousands of frames in just a few seconds.
>
---
#### [new 044] Improving Progressive Generation with Decomposable Flow Matching
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉生成任务，旨在提升多阶段生成效果。提出DFM框架，在多尺度表示中独立应用流匹配，解决传统多阶段架构复杂的问题，实现更优的生成质量与效率。**

- **链接: [http://arxiv.org/pdf/2506.19839v1](http://arxiv.org/pdf/2506.19839v1)**

> **作者:** Moayed Haji-Ali; Willi Menapace; Ivan Skorokhodov; Arpit Sahni; Sergey Tulyakov; Vicente Ordonez; Aliaksandr Siarohin
>
> **备注:** Project Webpage: https://snap-research.github.io/dfm/
>
> **摘要:** Generating high-dimensional visual modalities is a computationally intensive task. A common solution is progressive generation, where the outputs are synthesized in a coarse-to-fine spectral autoregressive manner. While diffusion models benefit from the coarse-to-fine nature of denoising, explicit multi-stage architectures are rarely adopted. These architectures have increased the complexity of the overall approach, introducing the need for a custom diffusion formulation, decomposition-dependent stage transitions, add-hoc samplers, or a model cascade. Our contribution, Decomposable Flow Matching (DFM), is a simple and effective framework for the progressive generation of visual media. DFM applies Flow Matching independently at each level of a user-defined multi-scale representation (such as Laplacian pyramid). As shown by our experiments, our approach improves visual quality for both images and videos, featuring superior results compared to prior multistage frameworks. On Imagenet-1k 512px, DFM achieves 35.2% improvements in FDD scores over the base architecture and 26.4% over the best-performing baseline, under the same training compute. When applied to finetuning of large models, such as FLUX, DFM shows faster convergence speed to the training distribution. Crucially, all these advantages are achieved with a single model, architectural simplicity, and minimal modifications to existing training pipelines.
>
---
#### [new 045] Emergence of Text Readability in Vision Language Models
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型中文本可读性出现的机制，探讨如何提升模型对图像中文本的理解能力。**

- **链接: [http://arxiv.org/pdf/2506.19389v1](http://arxiv.org/pdf/2506.19389v1)**

> **作者:** Jaeyoo Park; Sanghyuk Chun; Wonjae Kim; Sangdoo Yun; Bohyung Han
>
> **备注:** EVAL-FoMo Workshop @ CVPR 2025
>
> **摘要:** We investigate how the ability to recognize textual content within images emerges during the training of Vision-Language Models (VLMs). Our analysis reveals a critical phenomenon: the ability to read textual information in a given image \textbf{(text readability)} emerges abruptly after substantial training iterations, in contrast to semantic content understanding which develops gradually from the early stages of training. This delayed emergence may reflect how contrastive learning tends to initially prioritize general semantic understanding, with text-specific symbolic processing developing later. Interestingly, the ability to match images with rendered text develops even slower, indicating a deeper need for semantic integration. These findings highlight the need for tailored training strategies to accelerate robust text comprehension in VLMs, laying the groundwork for future research on optimizing multimodal learning.
>
---
#### [new 046] GenHSI: Controllable Generation of Human-Scene Interaction Videos
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决长视频中人-场景交互不自然、身份不一致的问题。提出GenHSI方法，通过分阶段生成实现可控、高质量的人-场景互动视频。**

- **链接: [http://arxiv.org/pdf/2506.19840v1](http://arxiv.org/pdf/2506.19840v1)**

> **作者:** Zekun Li; Rui Zhou; Rahul Sajnani; Xiaoyan Cong; Daniel Ritchie; Srinath Sridhar
>
> **摘要:** Large-scale pre-trained video diffusion models have exhibited remarkable capabilities in diverse video generation. However, existing solutions face several challenges in using these models to generate long movie-like videos with rich human-object interactions that include unrealistic human-scene interaction, lack of subject identity preservation, and require expensive training. We propose GenHSI, a training-free method for controllable generation of long human-scene interaction videos (HSI). Taking inspiration from movie animation, our key insight is to overcome the limitations of previous work by subdividing the long video generation task into three stages: (1) script writing, (2) pre-visualization, and (3) animation. Given an image of a scene, a user description, and multiple images of a person, we use these three stages to generate long-videos that preserve human-identity and provide rich human-scene interactions. Script writing converts complex human tasks into simple atomic tasks that are used in the pre-visualization stage to generate 3D keyframes (storyboards). These 3D keyframes are rendered and animated by off-the-shelf video diffusion models for consistent long video generation with rich contacts in a 3D-aware manner. A key advantage of our work is that we alleviate the need for scanned, accurate scenes and create 3D keyframes from single-view images. We are the first to generate a long video sequence with a consistent camera pose that contains arbitrary numbers of character actions without training. Experiments demonstrate that our method can generate long videos that effectively preserve scene content and character identity with plausible human-scene interaction from a single image scene. Visit our project homepage https://kunkun0w0.github.io/project/GenHSI/ for more information.
>
---
#### [new 047] MOSCARD -- Causal Reasoning and De-confounding for Multimodal Opportunistic Screening of Cardiovascular Adverse Events
- **分类: cs.CV**

- **简介: 该论文属于心血管事件预测任务，旨在通过多模态数据（CXR与ECG）进行因果推理和去混杂，提升风险评估准确性。**

- **链接: [http://arxiv.org/pdf/2506.19174v1](http://arxiv.org/pdf/2506.19174v1)**

> **作者:** Jialu Pi; Juan Maria Farina; Rimita Lahiri; Jiwoong Jeong; Archana Gurudu; Hyung-Bok Park; Chieh-Ju Chao; Chadi Ayoub; Reza Arsanjani; Imon Banerjee
>
> **摘要:** Major Adverse Cardiovascular Events (MACE) remain the leading cause of mortality globally, as reported in the Global Disease Burden Study 2021. Opportunistic screening leverages data collected from routine health check-ups and multimodal data can play a key role to identify at-risk individuals. Chest X-rays (CXR) provide insights into chronic conditions contributing to major adverse cardiovascular events (MACE), while 12-lead electrocardiogram (ECG) directly assesses cardiac electrical activity and structural abnormalities. Integrating CXR and ECG could offer a more comprehensive risk assessment than conventional models, which rely on clinical scores, computed tomography (CT) measurements, or biomarkers, which may be limited by sampling bias and single modality constraints. We propose a novel predictive modeling framework - MOSCARD, multimodal causal reasoning with co-attention to align two distinct modalities and simultaneously mitigate bias and confounders in opportunistic risk estimation. Primary technical contributions are - (i) multimodal alignment of CXR with ECG guidance; (ii) integration of causal reasoning; (iii) dual back-propagation graph for de-confounding. Evaluated on internal, shift data from emergency department (ED) and external MIMIC datasets, our model outperformed single modality and state-of-the-art foundational models - AUC: 0.75, 0.83, 0.71 respectively. Proposed cost-effective opportunistic screening enables early intervention, improving patient outcomes and reducing disparities.
>
---
#### [new 048] EvDetMAV: Generalized MAV Detection from Moving Event Cameras
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于目标检测任务，旨在解决MAV在复杂场景下的检测难题。通过分析事件相机中的螺旋桨特征，提出新方法并构建首个事件数据集，提升检测精度。**

- **链接: [http://arxiv.org/pdf/2506.19416v1](http://arxiv.org/pdf/2506.19416v1)**

> **作者:** Yin Zhang; Zian Ning; Xiaoyu Zhang; Shiliang Guo; Peidong Liu; Shiyu Zhao
>
> **备注:** 8 pages, 7 figures. This paper is accepted by IEEE Robotics and Automation Letters
>
> **摘要:** Existing micro aerial vehicle (MAV) detection methods mainly rely on the target's appearance features in RGB images, whose diversity makes it difficult to achieve generalized MAV detection. We notice that different types of MAVs share the same distinctive features in event streams due to their high-speed rotating propellers, which are hard to see in RGB images. This paper studies how to detect different types of MAVs from an event camera by fully exploiting the features of propellers in the original event stream. The proposed method consists of three modules to extract the salient and spatio-temporal features of the propellers while filtering out noise from background objects and camera motion. Since there are no existing event-based MAV datasets, we introduce a novel MAV dataset for the community. This is the first event-based MAV dataset comprising multiple scenarios and different types of MAVs. Without training, our method significantly outperforms state-of-the-art methods and can deal with challenging scenarios, achieving a precision rate of 83.0\% (+30.3\%) and a recall rate of 81.5\% (+36.4\%) on the proposed testing dataset. The dataset and code are available at: https://github.com/WindyLab/EvDetMAV.
>
---
#### [new 049] SAM2-SGP: Enhancing SAM2 for Medical Image Segmentation via Support-Set Guided Prompting
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决SAM2在医疗领域依赖人工提示和域偏移的问题。通过引入支持集引导提示、伪掩码生成与注意力机制，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2506.19658v1](http://arxiv.org/pdf/2506.19658v1)**

> **作者:** Yang Xing; Jiong Wu; Yuheng Bu; Kuang Gong
>
> **摘要:** Although new vision foundation models such as Segment Anything Model 2 (SAM2) have significantly enhanced zero-shot image segmentation capabilities, reliance on human-provided prompts poses significant challenges in adapting SAM2 to medical image segmentation tasks. Moreover, SAM2's performance in medical image segmentation was limited by the domain shift issue, since it was originally trained on natural images and videos. To address these challenges, we proposed SAM2 with support-set guided prompting (SAM2-SGP), a framework that eliminated the need for manual prompts. The proposed model leveraged the memory mechanism of SAM2 to generate pseudo-masks using image-mask pairs from a support set via a Pseudo-mask Generation (PMG) module. We further introduced a novel Pseudo-mask Attention (PMA) module, which used these pseudo-masks to automatically generate bounding boxes and enhance localized feature extraction by guiding attention to relevant areas. Furthermore, a low-rank adaptation (LoRA) strategy was adopted to mitigate the domain shift issue. The proposed framework was evaluated on both 2D and 3D datasets across multiple medical imaging modalities, including fundus photography, X-ray, computed tomography (CT), magnetic resonance imaging (MRI), positron emission tomography (PET), and ultrasound. The results demonstrated a significant performance improvement over state-of-the-art models, such as nnUNet and SwinUNet, as well as foundation models, such as SAM2 and MedSAM2, underscoring the effectiveness of the proposed approach. Our code is publicly available at https://github.com/astlian9/SAM_Support.
>
---
#### [new 050] VideoPCDNet: Video Parsing and Prediction with Phase Correlation Networks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频解析与预测任务，旨在解决无监督对象表示和动态建模问题。提出VideoPCDNet框架，利用相位相关技术实现对象分解与跟踪。**

- **链接: [http://arxiv.org/pdf/2506.19621v1](http://arxiv.org/pdf/2506.19621v1)**

> **作者:** Noel José Rodrigues Vicente; Enrique Lehner; Angel Villar-Corrales; Jan Nogga; Sven Behnke
>
> **备注:** Accepted for Publication at ICANN 2025
>
> **摘要:** Understanding and predicting video content is essential for planning and reasoning in dynamic environments. Despite advancements, unsupervised learning of object representations and dynamics remains challenging. We present VideoPCDNet, an unsupervised framework for object-centric video decomposition and prediction. Our model uses frequency-domain phase correlation techniques to recursively parse videos into object components, which are represented as transformed versions of learned object prototypes, enabling accurate and interpretable tracking. By explicitly modeling object motion through a combination of frequency domain operations and lightweight learned modules, VideoPCDNet enables accurate unsupervised object tracking and prediction of future video frames. In our experiments, we demonstrate that VideoPCDNet outperforms multiple object-centric baseline models for unsupervised tracking and prediction on several synthetic datasets, while learning interpretable object and motion representations.
>
---
#### [new 051] RareSpot: Spotting Small and Rare Wildlife in Aerial Imagery with Multi-Scale Consistency and Context-Aware Augmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于目标检测任务，旨在解决小而稀有野生动物在航拍图像中的检测难题。通过多尺度一致性和上下文感知增强方法提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.19087v1](http://arxiv.org/pdf/2506.19087v1)**

> **作者:** Bowen Zhang; Jesse T. Boulerice; Nikhil Kuniyil; Charvi Mendiratta; Satish Kumar; Hila Shamon; B. S. Manjunath
>
> **备注:** Accepted to the CVPR 2025 Workshop on Computer Vision for Animal Behavior Tracking and Modeling (CV4Animals)
>
> **摘要:** Automated detection of small and rare wildlife in aerial imagery is crucial for effective conservation, yet remains a significant technical challenge. Prairie dogs exemplify this issue: their ecological importance as keystone species contrasts sharply with their elusive presence--marked by small size, sparse distribution, and subtle visual features--which undermines existing detection approaches. To address these challenges, we propose RareSpot, a robust detection framework integrating multi-scale consistency learning and context-aware augmentation. Our multi-scale consistency approach leverages structured alignment across feature pyramids, enhancing fine-grained object representation and mitigating scale-related feature loss. Complementarily, context-aware augmentation strategically synthesizes challenging training instances by embedding difficult-to-detect samples into realistic environmental contexts, significantly boosting model precision and recall. Evaluated on an expert-annotated prairie dog drone imagery benchmark, our method achieves state-of-the-art performance, improving detection accuracy by over 35% compared to baseline methods. Importantly, it generalizes effectively across additional wildlife datasets, demonstrating broad applicability. The RareSpot benchmark and approach not only support critical ecological monitoring but also establish a new foundation for detecting small, rare species in complex aerial scenes.
>
---
#### [new 052] A Global-Local Cross-Attention Network for Ultra-high Resolution Remote Sensing Image Semantic Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感图像语义分割任务，旨在解决超高清图像分割中的计算效率与多尺度特征融合问题。提出GLCANet网络，通过双流结构和注意力机制提升分割精度与效率。**

- **链接: [http://arxiv.org/pdf/2506.19406v1](http://arxiv.org/pdf/2506.19406v1)**

> **作者:** Chen Yi; Shan LianLei
>
> **摘要:** With the rapid development of ultra-high resolution (UHR) remote sensing technology, the demand for accurate and efficient semantic segmentation has increased significantly. However, existing methods face challenges in computational efficiency and multi-scale feature fusion. To address these issues, we propose GLCANet (Global-Local Cross-Attention Network), a lightweight segmentation framework designed for UHR remote sensing imagery.GLCANet employs a dual-stream architecture to efficiently fuse global semantics and local details while minimizing GPU usage. A self-attention mechanism enhances long-range dependencies, refines global features, and preserves local details for better semantic consistency. A masked cross-attention mechanism also adaptively fuses global-local features, selectively enhancing fine-grained details while exploiting global context to improve segmentation accuracy. Experimental results show that GLCANet outperforms state-of-the-art methods regarding accuracy and computational efficiency. The model effectively processes large, high-resolution images with a small memory footprint, providing a promising solution for real-world remote sensing applications.
>
---
#### [new 053] Interpretable and Granular Video-Based Quantification of Motor Characteristics from the Finger Tapping Test in Parkinson Disease
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学影像分析任务，旨在客观量化帕金森病患者的运动特征。通过视频分析提取四个关键特征，提升MDS-UPDRS评分预测精度并提供可解释性。**

- **链接: [http://arxiv.org/pdf/2506.18925v1](http://arxiv.org/pdf/2506.18925v1)**

> **作者:** Tahereh Zarrat Ehsan; Michael Tangermann; Yağmur Güçlütürk; Bastiaan R. Bloem; Luc J. W. Evers
>
> **摘要:** Accurately quantifying motor characteristics in Parkinson disease (PD) is crucial for monitoring disease progression and optimizing treatment strategies. The finger-tapping test is a standard motor assessment. Clinicians visually evaluate a patient's tapping performance and assign an overall severity score based on tapping amplitude, speed, and irregularity. However, this subjective evaluation is prone to inter- and intra-rater variability, and does not offer insights into individual motor characteristics captured during this test. This paper introduces a granular computer vision-based method for quantifying PD motor characteristics from video recordings. Four sets of clinically relevant features are proposed to characterize hypokinesia, bradykinesia, sequence effect, and hesitation-halts. We evaluate our approach on video recordings and clinical evaluations of 74 PD patients from the Personalized Parkinson Project. Principal component analysis with varimax rotation shows that the video-based features corresponded to the four deficits. Additionally, video-based analysis has allowed us to identify further granular distinctions within sequence effect and hesitation-halts deficits. In the following, we have used these features to train machine learning classifiers to estimate the Movement Disorder Society Unified Parkinson Disease Rating Scale (MDS-UPDRS) finger-tapping score. Compared to state-of-the-art approaches, our method achieves a higher accuracy in MDS-UPDRS score prediction, while still providing an interpretable quantification of individual finger-tapping motor characteristics. In summary, the proposed framework provides a practical solution for the objective assessment of PD motor characteristics, that can potentially be applied in both clinical and remote settings. Future work is needed to assess its responsiveness to symptomatic treatment and disease progression.
>
---
#### [new 054] Implementing blind navigation through multi-modal sensing and gait guidance
- **分类: cs.CV; cs.SY; eess.SY**

- **简介: 该论文属于无障碍导航任务，旨在解决视障人士路径寻找与障碍避让问题。通过多模态感知和步态引导系统实现更优的导航辅助。**

- **链接: [http://arxiv.org/pdf/2506.19593v1](http://arxiv.org/pdf/2506.19593v1)**

> **作者:** Feifan Yan; Tianle Zeng; Meixi He
>
> **摘要:** By the year 2023, the global population of individuals with impaired vision has surpassed 220 million. People with impaired vision will find it difficult while finding path or avoiding obstacles, and must ask for auxiliary tools for help. Although traditional aids such as guide canes and guide dogs exist, they still have some shortcomings. In this paper, we present our wearable blind guiding device, what perform navigation guidance through our proposed Gait-based Guiding System. Our device innovatively integrates gait phase analysis for walking guide, and in terms of environmental perception, we use multimodal sensing to acquire diverse environment information. During the experiment, we conducted both indoor and outdoor experiments, and compared with the standard guide cane. The result shows superior performance of our device in blind guidance.
>
---
#### [new 055] OpenWildlife: Open-Vocabulary Multi-Species Wildlife Detector for Geographically-Diverse Aerial Imagery
- **分类: cs.CV**

- **简介: 该论文提出OpenWildlife，解决多物种野生动物检测问题，通过语言感知嵌入和改进的DINO框架，提升跨环境识别能力。**

- **链接: [http://arxiv.org/pdf/2506.19204v1](http://arxiv.org/pdf/2506.19204v1)**

> **作者:** Muhammed Patel; Javier Noa Turnes; Jayden Hsiao; Linlin Xu; David Clausi
>
> **摘要:** We introduce OpenWildlife (OW), an open-vocabulary wildlife detector designed for multi-species identification in diverse aerial imagery. While existing automated methods perform well in specific settings, they often struggle to generalize across different species and environments due to limited taxonomic coverage and rigid model architectures. In contrast, OW leverages language-aware embeddings and a novel adaptation of the Grounding-DINO framework, enabling it to identify species specified through natural language inputs across both terrestrial and marine environments. Trained on 15 datasets, OW outperforms most existing methods, achieving up to \textbf{0.981} mAP50 with fine-tuning and \textbf{0.597} mAP50 on seven datasets featuring novel species. Additionally, we introduce an efficient search algorithm that combines k-nearest neighbors and breadth-first search to prioritize areas where social species are likely to be found. This approach captures over \textbf{95\%} of species while exploring only \textbf{33\%} of the available images. To support reproducibility, we publicly release our source code and dataset splits, establishing OW as a flexible, cost-effective solution for global biodiversity assessments.
>
---
#### [new 056] Active View Selector: Fast and Accurate Active View Selection with Cross Reference Image Quality Assessment
- **分类: cs.CV**

- **简介: 该论文属于新型视图合成与3D重建任务，解决主动视图选择问题。通过2D图像质量评估方法，提升视图选择效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.19844v1](http://arxiv.org/pdf/2506.19844v1)**

> **作者:** Zirui Wang; Yash Bhalgat; Ruining Li; Victor Adrian Prisacariu
>
> **备注:** Project page: https://avs.active.vision/
>
> **摘要:** We tackle active view selection in novel view synthesis and 3D reconstruction. Existing methods like FisheRF and ActiveNeRF select the next best view by minimizing uncertainty or maximizing information gain in 3D, but they require specialized designs for different 3D representations and involve complex modelling in 3D space. Instead, we reframe this as a 2D image quality assessment (IQA) task, selecting views where current renderings have the lowest quality. Since ground-truth images for candidate views are unavailable, full-reference metrics like PSNR and SSIM are inapplicable, while no-reference metrics, such as MUSIQ and MANIQA, lack the essential multi-view context. Inspired by a recent cross-referencing quality framework CrossScore, we train a model to predict SSIM within a multi-view setup and use it to guide view selection. Our cross-reference IQA framework achieves substantial quantitative and qualitative improvements across standard benchmarks, while being agnostic to 3D representations, and runs 14-33 times faster than previous methods.
>
---
#### [new 057] ScaleCap: Inference-Time Scalable Image Captioning via Dual-Modality Debiasing
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于图像描述任务，旨在解决LVLMs的多模态和语言偏差问题。通过提出双模态去偏策略，提升生成描述的准确性和平衡性。**

- **链接: [http://arxiv.org/pdf/2506.19848v1](http://arxiv.org/pdf/2506.19848v1)**

> **作者:** Long Xing; Qidong Huang; Xiaoyi Dong; Pan Zhang; Yuhang Zang; Yuhang Cao; Jinsong Li; Shuangrui Ding; Weiming Zhang; Nenghai Yu; Jiaqi Wang; Feng Wu; Dahua Lin
>
> **备注:** Code is available at https://github.com/Cooperx521/ScaleCap
>
> **摘要:** This paper presents ScaleCap, an inference-time scalable image captioning strategy that generates comprehensive and detailed image captions. The key challenges of high-quality image captioning lie in the inherent biases of LVLMs: multimodal bias resulting in imbalanced descriptive granularity, offering detailed accounts of some elements while merely skimming over others; linguistic bias leading to hallucinated descriptions of non-existent objects. To address these issues, we propose a scalable debiased captioning strategy, which continuously enriches and calibrates the caption with increased inference budget. Specifically, we propose two novel components: heuristic question answering and contrastive sentence rating. The former generates content-specific questions based on the image and answers them to progressively inject relevant information into the caption. The latter employs sentence-level offline contrastive decoding to effectively identify and eliminate hallucinations caused by linguistic biases. With increased inference cost, more heuristic questions are raised by ScaleCap to progressively capture additional visual details, generating captions that are more accurate, balanced, and informative. Extensive modality alignment experiments demonstrate the effectiveness of ScaleCap. Annotating 450K images with ScaleCap and using them for LVLM pretraining leads to consistent performance gains across 11 widely used benchmarks. Furthermore, ScaleCap showcases superb richness and fidelity of generated captions with two additional tasks: replacing images with captions in VQA task, and reconstructing images from captions to assess semantic coverage. Code is available at https://github.com/Cooperx521/ScaleCap.
>
---
#### [new 058] HMSViT: A Hierarchical Masked Self-Supervised Vision Transformer for Corneal Nerve Segmentation and Diabetic Neuropathy Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割与诊断任务，旨在解决糖尿病周围神经病变的早期检测问题。提出HMSViT模型，通过自监督学习和多尺度特征提取，提升角膜神经分割和诊断准确性。**

- **链接: [http://arxiv.org/pdf/2506.19474v1](http://arxiv.org/pdf/2506.19474v1)**

> **作者:** Xin Zhang; Liangxiu Han; Yue Shi; Yanlin Zheng; Alam Uazman; Maryam Ferdousi; Rayaz Malik
>
> **摘要:** Diabetic Peripheral Neuropathy (DPN) affects nearly half of diabetes patients, requiring early detection. Corneal Confocal Microscopy (CCM) enables non-invasive diagnosis, but automated methods suffer from inefficient feature extraction, reliance on handcrafted priors, and data limitations. We propose HMSViT, a novel Hierarchical Masked Self-Supervised Vision Transformer (HMSViT) designed for corneal nerve segmentation and DPN diagnosis. Unlike existing methods, HMSViT employs pooling-based hierarchical and dual attention mechanisms with absolute positional encoding, enabling efficient multi-scale feature extraction by capturing fine-grained local details in early layers and integrating global context in deeper layers, all at a lower computational cost. A block-masked self supervised learning framework is designed for the HMSViT that reduces reliance on labelled data, enhancing feature robustness, while a multi-scale decoder is used for segmentation and classification by fusing hierarchical features. Experiments on clinical CCM datasets showed HMSViT achieves state-of-the-art performance, with 61.34% mIoU for nerve segmentation and 70.40% diagnostic accuracy, outperforming leading hierarchical models like the Swin Transformer and HiViT by margins of up to 6.39% in segmentation accuracy while using fewer parameters. Detailed ablation studies further reveal that integrating block-masked SSL with hierarchical multi-scale feature extraction substantially enhances performance compared to conventional supervised training. Overall, these comprehensive experiments confirm that HMSViT delivers excellent, robust, and clinically viable results, demonstrating its potential for scalable deployment in real-world diagnostic applications.
>
---
#### [new 059] Trajectory Prediction in Dynamic Object Tracking: A Critical Study
- **分类: cs.CV**

- **简介: 该论文属于动态目标跟踪与轨迹预测任务，旨在分析现有方法的优缺点及应用挑战，提出未来研究方向以提升系统性能与伦理安全性。**

- **链接: [http://arxiv.org/pdf/2506.19341v1](http://arxiv.org/pdf/2506.19341v1)**

> **作者:** Zhongping Dong; Liming Chen; Mohand Tahar Kechadi
>
> **摘要:** This study provides a detailed analysis of current advancements in dynamic object tracking (DOT) and trajectory prediction (TP) methodologies, including their applications and challenges. It covers various approaches, such as feature-based, segmentation-based, estimation-based, and learning-based methods, evaluating their effectiveness, deployment, and limitations in real-world scenarios. The study highlights the significant impact of these technologies in automotive and autonomous vehicles, surveillance and security, healthcare, and industrial automation, contributing to safety and efficiency. Despite the progress, challenges such as improved generalization, computational efficiency, reduced data dependency, and ethical considerations still exist. The study suggests future research directions to address these challenges, emphasizing the importance of multimodal data integration, semantic information fusion, and developing context-aware systems, along with ethical and privacy-preserving frameworks.
>
---
#### [new 060] SimpleGVR: A Simple Baseline for Latent-Cascaded Video Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于视频超分辨率任务，旨在提升视频分辨率。通过设计新模型和训练策略，解决高分辨率生成效率问题。**

- **链接: [http://arxiv.org/pdf/2506.19838v1](http://arxiv.org/pdf/2506.19838v1)**

> **作者:** Liangbin Xie; Yu Li; Shian Du; Menghan Xia; Xintao Wang; Fanghua Yu; Ziyan Chen; Pengfei Wan; Jiantao Zhou; Chao Dong
>
> **备注:** Project webpage available at https://simplegvr.github.io/
>
> **摘要:** Latent diffusion models have emerged as a leading paradigm for efficient video generation. However, as user expectations shift toward higher-resolution outputs, relying solely on latent computation becomes inadequate. A promising approach involves decoupling the process into two stages: semantic content generation and detail synthesis. The former employs a computationally intensive base model at lower resolutions, while the latter leverages a lightweight cascaded video super-resolution (VSR) model to achieve high-resolution output. In this work, we focus on studying key design principles for latter cascaded VSR models, which are underexplored currently. First, we propose two degradation strategies to generate training pairs that better mimic the output characteristics of the base model, ensuring alignment between the VSR model and its upstream generator. Second, we provide critical insights into VSR model behavior through systematic analysis of (1) timestep sampling strategies, (2) noise augmentation effects on low-resolution (LR) inputs. These findings directly inform our architectural and training innovations. Finally, we introduce interleaving temporal unit and sparse local attention to achieve efficient training and inference, drastically reducing computational overhead. Extensive experiments demonstrate the superiority of our framework over existing methods, with ablation studies confirming the efficacy of each design choice. Our work establishes a simple yet effective baseline for cascaded video super-resolution generation, offering practical insights to guide future advancements in efficient cascaded synthesis systems.
>
---
#### [new 061] Identifying Physically Realizable Triggers for Backdoored Face Recognition Networks
- **分类: cs.CV; cs.CR; cs.LG**

- **简介: 该论文属于后门攻击检测任务，旨在识别受感染人脸识别网络中的物理可实现触发器。工作包括检测和定位此类触发器。**

- **链接: [http://arxiv.org/pdf/2506.19533v1](http://arxiv.org/pdf/2506.19533v1)**

> **作者:** Ankita Raj; Ambar Pal; Chetan Arora
>
> **备注:** Accepted to ICIP 2021
>
> **摘要:** Backdoor attacks embed a hidden functionality into deep neural networks, causing the network to display anomalous behavior when activated by a predetermined pattern in the input Trigger, while behaving well otherwise on public test data. Recent works have shown that backdoored face recognition (FR) systems can respond to natural-looking triggers like a particular pair of sunglasses. Such attacks pose a serious threat to the applicability of FR systems in high-security applications. We propose a novel technique to (1) detect whether an FR network is compromised with a natural, physically realizable trigger, and (2) identify such triggers given a compromised network. We demonstrate the effectiveness of our methods with a compromised FR network, where we are able to identify the trigger (e.g., green sunglasses or red hat) with a top-5 accuracy of 74%, whereas a naive brute force baseline achieves 56% accuracy.
>
---
#### [new 062] USIS16K: High-Quality Dataset for Underwater Salient Instance Segmentation
- **分类: cs.CV; I.4.6**

- **简介: 该论文提出USIS16K数据集，用于解决水下显著实例分割任务，旨在提升水下场景中目标检测与分割的准确性。**

- **链接: [http://arxiv.org/pdf/2506.19472v1](http://arxiv.org/pdf/2506.19472v1)**

> **作者:** Lin Hong; Xin Wang; Yihao Li; Xia Wang
>
> **备注:** 8 pages 10 figures
>
> **摘要:** Inspired by the biological visual system that selectively allocates attention to efficiently identify salient objects or regions, underwater salient instance segmentation (USIS) aims to jointly address the problems of where to look (saliency prediction) and what is there (instance segmentation) in underwater scenarios. However, USIS remains an underexplored challenge due to the inaccessibility and dynamic nature of underwater environments, as well as the scarcity of large-scale, high-quality annotated datasets. In this paper, we introduce USIS16K, a large-scale dataset comprising 16,151 high-resolution underwater images collected from diverse environmental settings and covering 158 categories of underwater objects. Each image is annotated with high-quality instance-level salient object masks, representing a significant advance in terms of diversity, complexity, and scalability. Furthermore, we provide benchmark evaluations on underwater object detection and USIS tasks using USIS16K. To facilitate future research in this domain, the dataset and benchmark models are publicly available.
>
---
#### [new 063] Semantic Scene Graph for Ultrasound Image Explanation and Scanning Guidance
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文属于医学图像解释与引导任务，旨在提升非专业用户对超声图像的理解和操作。通过引入语义场景图，实现图像解释与扫描指导。**

- **链接: [http://arxiv.org/pdf/2506.19683v1](http://arxiv.org/pdf/2506.19683v1)**

> **作者:** Xuesong Li; Dianye Huang; Yameng Zhang; Nassir Navab; Zhongliang Jiang
>
> **摘要:** Understanding medical ultrasound imaging remains a long-standing challenge due to significant visual variability caused by differences in imaging and acquisition parameters. Recent advancements in large language models (LLMs) have been used to automatically generate terminology-rich summaries orientated to clinicians with sufficient physiological knowledge. Nevertheless, the increasing demand for improved ultrasound interpretability and basic scanning guidance among non-expert users, e.g., in point-of-care settings, has not yet been explored. In this study, we first introduce the scene graph (SG) for ultrasound images to explain image content to ordinary and provide guidance for ultrasound scanning. The ultrasound SG is first computed using a transformer-based one-stage method, eliminating the need for explicit object detection. To generate a graspable image explanation for ordinary, the user query is then used to further refine the abstract SG representation through LLMs. Additionally, the predicted SG is explored for its potential in guiding ultrasound scanning toward missing anatomies within the current imaging view, assisting ordinary users in achieving more standardized and complete anatomical exploration. The effectiveness of this SG-based image explanation and scanning guidance has been validated on images from the left and right neck regions, including the carotid and thyroid, across five volunteers. The results demonstrate the potential of the method to maximally democratize ultrasound by enhancing its interpretability and usability for ordinaries.
>
---
#### [new 064] PRISM: Perceptual Recognition for Identifying Standout Moments in Human-Centric Keyframe Extraction
- **分类: cs.CV**

- **简介: 该论文提出PRISM，用于识别视频中的关键帧，解决在线视频内容中突出时刻检测问题。通过感知颜色差异实现高效、可解释的关键帧提取。**

- **链接: [http://arxiv.org/pdf/2506.19168v1](http://arxiv.org/pdf/2506.19168v1)**

> **作者:** Mert Can Cakmak; Nitin Agarwal; Diwash Poudel
>
> **摘要:** Online videos play a central role in shaping political discourse and amplifying cyber social threats such as misinformation, propaganda, and radicalization. Detecting the most impactful or "standout" moments in video content is crucial for content moderation, summarization, and forensic analysis. In this paper, we introduce PRISM (Perceptual Recognition for Identifying Standout Moments), a lightweight and perceptually-aligned framework for keyframe extraction. PRISM operates in the CIELAB color space and uses perceptual color difference metrics to identify frames that align with human visual sensitivity. Unlike deep learning-based approaches, PRISM is interpretable, training-free, and computationally efficient, making it well suited for real-time and resource-constrained environments. We evaluate PRISM on four benchmark datasets: BBC, TVSum, SumMe, and ClipShots, and demonstrate that it achieves strong accuracy and fidelity while maintaining high compression ratios. These results highlight PRISM's effectiveness in both structured and unstructured video content, and its potential as a scalable tool for analyzing and moderating harmful or politically sensitive media in online platforms.
>
---
#### [new 065] SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images
- **分类: cs.CV**

- **简介: 该论文提出SMARTIES模型，解决多传感器遥感图像处理中模型适应性差的问题。通过统一的Transformer架构实现跨传感器数据重建，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.19585v1](http://arxiv.org/pdf/2506.19585v1)**

> **作者:** Gencer Sumbul; Chang Xu; Emanuele Dalsasso; Devis Tuia
>
> **摘要:** From optical sensors to microwave radars, leveraging the complementary strengths of remote sensing (RS) sensors is crucial for achieving dense spatio-temporal monitoring of our planet. In contrast, recent deep learning models, whether task-specific or foundational, are often specific to single sensors or to fixed combinations: adapting such models to different sensory inputs requires both architectural changes and re-training, limiting scalability and generalization across multiple RS sensors. On the contrary, a single model able to modulate its feature representations to accept diverse sensors as input would pave the way to agile and flexible multi-sensor RS data processing. To address this, we introduce SMARTIES, a generic and versatile foundation model lifting sensor-specific/dependent efforts and enabling scalability and generalization to diverse RS sensors: SMARTIES projects data from heterogeneous sensors into a shared spectrum-aware space, enabling the use of arbitrary combinations of bands both for training and inference. To obtain sensor-agnostic representations, we train a single, unified transformer model reconstructing masked multi-sensor data with cross-sensor token mixup. On both single- and multi-modal tasks across diverse sensors, SMARTIES outperforms previous models that rely on sensor-specific pretraining. Our code and pretrained models are available at https://gsumbul.github.io/SMARTIES.
>
---
#### [new 066] HOIverse: A Synthetic Scene Graph Dataset With Human Object Interactions
- **分类: cs.CV**

- **简介: 该论文提出HOIverse，一个包含人类与物体交互的合成场景图数据集，旨在解决室内环境中的场景理解问题，提升人机协作任务的性能。**

- **链接: [http://arxiv.org/pdf/2506.19639v1](http://arxiv.org/pdf/2506.19639v1)**

> **作者:** Mrunmai Vivek Phatak; Julian Lorenz; Nico Hörmann; Jörg Hähner; Rainer Lienhart
>
> **摘要:** When humans and robotic agents coexist in an environment, scene understanding becomes crucial for the agents to carry out various downstream tasks like navigation and planning. Hence, an agent must be capable of localizing and identifying actions performed by the human. Current research lacks reliable datasets for performing scene understanding within indoor environments where humans are also a part of the scene. Scene Graphs enable us to generate a structured representation of a scene or an image to perform visual scene understanding. To tackle this, we present HOIverse a synthetic dataset at the intersection of scene graph and human-object interaction, consisting of accurate and dense relationship ground truths between humans and surrounding objects along with corresponding RGB images, segmentation masks, depth images and human keypoints. We compute parametric relations between various pairs of objects and human-object pairs, resulting in an accurate and unambiguous relation definitions. In addition, we benchmark our dataset on state-of-the-art scene graph generation models to predict parametric relations and human-object interactions. Through this dataset, we aim to accelerate research in the field of scene understanding involving people.
>
---
#### [new 067] Lightweight RGB-T Tracking with Mobile Vision Transformers
- **分类: cs.CV**

- **简介: 该论文属于目标跟踪任务，解决单模态跟踪在恶劣环境下的性能问题，提出基于MobileViT的轻量级RGB-T跟踪算法。**

- **链接: [http://arxiv.org/pdf/2506.19154v1](http://arxiv.org/pdf/2506.19154v1)**

> **作者:** Mahdi Falaki; Maria A. Amer
>
> **摘要:** Single-modality object tracking (e.g., RGB-only) encounters difficulties in challenging imaging conditions, such as low illumination and adverse weather conditions. To solve this, multimodal tracking (e.g., RGB-T models) aims to leverage complementary data such as thermal infrared features. While recent Vision Transformer-based multimodal trackers achieve strong performance, they are often computationally expensive due to large model sizes. In this work, we propose a novel lightweight RGB-T tracking algorithm based on Mobile Vision Transformers (MobileViT). Our tracker introduces a progressive fusion framework that jointly learns intra-modal and inter-modal interactions between the template and search regions using separable attention. This design produces effective feature representations that support more accurate target localization while achieving a small model size and fast inference speed. Compared to state-of-the-art efficient multimodal trackers, our model achieves comparable accuracy while offering significantly lower parameter counts (less than 4 million) and the fastest GPU inference speed of 122 frames per second. This paper is the first to propose a tracker using Mobile Vision Transformers for RGB-T tracking and multimodal tracking at large. Tracker code and model weights will be made publicly available upon acceptance.
>
---
#### [new 068] Video Compression for Spatiotemporal Earth System Data
- **分类: cs.CV; cs.DL; eess.IV; physics.geo-ph**

- **简介: 该论文属于地球系统数据压缩任务，解决大规模遥感数据存储与传输问题，通过视频编码技术实现高效压缩并保持高保真。**

- **链接: [http://arxiv.org/pdf/2506.19656v1](http://arxiv.org/pdf/2506.19656v1)**

> **作者:** Oscar J. Pellicer-Valero; Cesar Aybar; Gustau Camps Valls
>
> **摘要:** Large-scale Earth system datasets, from high-resolution remote sensing imagery to spatiotemporal climate model outputs, exhibit characteristics analogous to those of standard videos. Their inherent spatial, temporal, and spectral redundancies can thus be readily exploited by established video compression techniques. Here, we present xarrayvideo, a Python library for compressing multichannel spatiotemporal datasets by encoding them as videos. Our approach achieves compression ratios of up to 250x while maintaining high fidelity by leveraging standard, well-optimized video codecs through ffmpeg. We demonstrate the library's effectiveness on four real-world multichannel spatiotemporal datasets: DynamicEarthNet (very high resolution Planet images), DeepExtremeCubes (high resolution Sentinel-2 images), ERA5 (weather reanalysis data), and the SimpleS2 dataset (high resolution multichannel Sentinel-2 images), achieving Peak Signal-to-Noise Ratios (PSNRs) of 55.86, 40.60, 46.58, and 43.23 dB at 0.1 bits per pixel per band (bpppb) and 65.91, 54.28, 62.90, and 55.04 dB at 1 bpppb. We are redistributing two of these datasets, DeepExtremeCubes (2.3 Tb) and DynamicEarthNet (525 Gb), in the machine-learning-ready and cloud-ready TACO format through HuggingFace at significantly reduced sizes (270 Gb and 8.5 Gb, respectively) without compromising quality (PSNR 55.77-56.65 and 60.15). No performance loss is observed when the compressed versions of these datasets are used in their respective deep learning-based downstream tasks (next step reflectance prediction and landcover segmentation). In conclusion, xarrayvideo presents an efficient solution for handling the rapidly growing size of Earth observation datasets, making advanced compression techniques accessible and practical to the Earth science community. The library is available for use at https://github.com/IPL-UV/xarrayvideo
>
---
#### [new 069] ReMAR-DS: Recalibrated Feature Learning for Metal Artifact Reduction and CT Domain Transformation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像处理任务，旨在解决kVCT中的金属伪影问题，并实现kVCT到MVCT的域转换。通过深度学习框架ReMAR-DS提升图像质量，减少辐射暴露。**

- **链接: [http://arxiv.org/pdf/2506.19531v1](http://arxiv.org/pdf/2506.19531v1)**

> **作者:** Mubashara Rehman; Niki Martinel; Michele Avanzo; Riccardo Spizzo; Christian Micheloni
>
> **备注:** Accepted in 23rd International Conference on Image Analysis and Processing (ICIAP) 2025, Italy
>
> **摘要:** Artifacts in kilo-Voltage CT (kVCT) imaging degrade image quality, impacting clinical decisions. We propose a deep learning framework for metal artifact reduction (MAR) and domain transformation from kVCT to Mega-Voltage CT (MVCT). The proposed framework, ReMAR-DS, utilizes an encoder-decoder architecture with enhanced feature recalibration, effectively reducing artifacts while preserving anatomical structures. This ensures that only relevant information is utilized in the reconstruction process. By infusing recalibrated features from the encoder block, the model focuses on relevant spatial regions (e.g., areas with artifacts) and highlights key features across channels (e.g., anatomical structures), leading to improved reconstruction of artifact-corrupted regions. Unlike traditional MAR methods, our approach bridges the gap between high-resolution kVCT and artifact-resistant MVCT, enhancing radiotherapy planning. It produces high-quality MVCT-like reconstructions, validated through qualitative and quantitative evaluations. Clinically, this enables oncologists to rely on kVCT alone, reducing repeated high-dose MVCT scans and lowering radiation exposure for cancer patients.
>
---
#### [new 070] Reinforcement Learning-Based Dynamic Grouping for Tubular Structure Tracking
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像处理任务，解决管状结构跟踪问题。针对现有方法计算效率低和依赖先验知识的问题，提出基于强化学习的动态分组方法，提升跟踪效果与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.18930v1](http://arxiv.org/pdf/2506.18930v1)**

> **作者:** Chong Di; Shuwang Zhou; Da Chen; Jean-Marie Mirebeau; Minglei Shu; Laurent D. Cohen
>
> **摘要:** The computation of minimal paths for the applications in tracking tubular structures such as blood vessels and roads is challenged by complex morphologies and environmental variations. Existing approaches can be roughly categorized into two research lines: the point-wise based models and the segment-wise based models. Although segment-wise approaches have obtained promising results in many scenarios, they often suffer from computational inefficiency and heavily rely on a prescribed prior to fit the target elongated shapes. We propose a novel framework that casts segment-wise tracking as a Markov Decision Process (MDP), enabling a reinforcement learning approach. Our method leverages Q-Learning to dynamically explore a graph of segments, computing edge weights on-demand and adaptively expanding the search space. This strategy avoids the high cost of a pre-computed graph and proves robust to incomplete initial information. Experimental reuslts on typical tubular structure datasets demonstrate that our method significantly outperforms state-of-the-art point-wise and segment-wise approaches. The proposed method effectively handles complex topologies and maintains global path coherence without depending on extensive prior structural knowledge.
>
---
#### [new 071] UltraAD: Fine-Grained Ultrasound Anomaly Classification via Few-Shot CLIP Adaptation
- **分类: cs.CV**

- **简介: 该论文属于医学图像异常检测任务，旨在解决超声图像中细粒度异常分类问题。通过少样本CLIP适配方法提升定位与分类性能。**

- **链接: [http://arxiv.org/pdf/2506.19694v1](http://arxiv.org/pdf/2506.19694v1)**

> **作者:** Yue Zhou; Yuan Bi; Wenjuan Tong; Wei Wang; Nassir Navab; Zhongliang Jiang
>
> **摘要:** Precise anomaly detection in medical images is critical for clinical decision-making. While recent unsupervised or semi-supervised anomaly detection methods trained on large-scale normal data show promising results, they lack fine-grained differentiation, such as benign vs. malignant tumors. Additionally, ultrasound (US) imaging is highly sensitive to devices and acquisition parameter variations, creating significant domain gaps in the resulting US images. To address these challenges, we propose UltraAD, a vision-language model (VLM)-based approach that leverages few-shot US examples for generalized anomaly localization and fine-grained classification. To enhance localization performance, the image-level token of query visual prototypes is first fused with learnable text embeddings. This image-informed prompt feature is then further integrated with patch-level tokens, refining local representations for improved accuracy. For fine-grained classification, a memory bank is constructed from few-shot image samples and corresponding text descriptions that capture anatomical and abnormality-specific features. During training, the stored text embeddings remain frozen, while image features are adapted to better align with medical data. UltraAD has been extensively evaluated on three breast US datasets, outperforming state-of-the-art methods in both lesion localization and fine-grained medical classification. The code will be released upon acceptance.
>
---
#### [new 072] AirV2X: Unified Air-Ground Vehicle-to-Everything Collaboration
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决传统V2X系统在农村和郊区的覆盖不足问题。通过无人机辅助，构建了AirV2X-Perception数据集，支持V2D算法开发与评估。**

- **链接: [http://arxiv.org/pdf/2506.19283v1](http://arxiv.org/pdf/2506.19283v1)**

> **作者:** Xiangbo Gao; Yuheng Wu; Xuewen Luo; Keshu Wu; Xinghao Chen; Yuping Wang; Chenxi Liu; Yang Zhou; Zhengzhong Tu
>
> **摘要:** While multi-vehicular collaborative driving demonstrates clear advantages over single-vehicle autonomy, traditional infrastructure-based V2X systems remain constrained by substantial deployment costs and the creation of "uncovered danger zones" in rural and suburban areas. We present AirV2X-Perception, a large-scale dataset that leverages Unmanned Aerial Vehicles (UAVs) as a flexible alternative or complement to fixed Road-Side Units (RSUs). Drones offer unique advantages over ground-based perception: complementary bird's-eye-views that reduce occlusions, dynamic positioning capabilities that enable hovering, patrolling, and escorting navigation rules, and significantly lower deployment costs compared to fixed infrastructure. Our dataset comprises 6.73 hours of drone-assisted driving scenarios across urban, suburban, and rural environments with varied weather and lighting conditions. The AirV2X-Perception dataset facilitates the development and standardized evaluation of Vehicle-to-Drone (V2D) algorithms, addressing a critical gap in the rapidly expanding field of aerial-assisted autonomous driving systems. The dataset and development kits are open-sourced at https://github.com/taco-group/AirV2X-Perception.
>
---
#### [new 073] PrITTI: Primitive-based Generation of Controllable and Editable 3D Semantic Scenes
- **分类: cs.CV**

- **简介: 该论文属于3D语义场景生成任务，旨在解决传统体素表示的高内存消耗和难编辑问题，提出PrITTI框架，利用基础几何体实现可控制、可编辑的3D场景生成。**

- **链接: [http://arxiv.org/pdf/2506.19117v1](http://arxiv.org/pdf/2506.19117v1)**

> **作者:** Christina Ourania Tze; Daniel Dauner; Yiyi Liao; Dzmitry Tsishkou; Andreas Geiger
>
> **备注:** Project page: https://raniatze.github.io/pritti/
>
> **摘要:** Large-scale 3D semantic scene generation has predominantly relied on voxel-based representations, which are memory-intensive, bound by fixed resolutions, and challenging to edit. In contrast, primitives represent semantic entities using compact, coarse 3D structures that are easy to manipulate and compose, making them an ideal representation for this task. In this paper, we introduce PrITTI, a latent diffusion-based framework that leverages primitives as the main foundational elements for generating compositional, controllable, and editable 3D semantic scene layouts. Our method adopts a hybrid representation, modeling ground surfaces in a rasterized format while encoding objects as vectorized 3D primitives. This decomposition is also reflected in a structured latent representation that enables flexible scene manipulation of ground and object components. To overcome the orientation ambiguities in conventional encoding methods, we introduce a stable Cholesky-based parameterization that jointly encodes object size and orientation. Experiments on the KITTI-360 dataset show that PrITTI outperforms a voxel-based baseline in generation quality, while reducing memory requirements by up to $3\times$. In addition, PrITTI enables direct instance-level manipulation of objects in the scene and supports a range of downstream applications, including scene inpainting, outpainting, and photo-realistic street-view synthesis.
>
---
#### [new 074] LEGATO: Large-scale End-to-end Generalizable Approach to Typeset OMR
- **分类: cs.CV; cs.DL**

- **简介: 该论文提出Legato，一个端到端的Transformer模型，用于光学音乐识别（OMR），解决多页类型乐谱识别与ABC符号生成问题。**

- **链接: [http://arxiv.org/pdf/2506.19065v1](http://arxiv.org/pdf/2506.19065v1)**

> **作者:** Guang Yang; Victoria Ebert; Nazif Tamer; Luiza Pozzobon; Noah A. Smith
>
> **摘要:** We propose Legato, a new end-to-end transformer model for optical music recognition (OMR). Legato is the first large-scale pretrained OMR model capable of recognizing full-page or multi-page typeset music scores and the first to generate documents in ABC notation, a concise, human-readable format for symbolic music. Bringing together a pretrained vision encoder with an ABC decoder trained on a dataset of more than 214K images, our model exhibits the strong ability to generalize across various typeset scores. We conduct experiments on a range of datasets and demonstrate that our model achieves state-of-the-art performance. Given the lack of a standardized evaluation for end-to-end OMR, we comprehensively compare our model against the previous state of the art using a diverse set of metrics.
>
---
#### [new 075] One Prototype Is Enough: Single-Prototype Activation for Interpretable Image Classification
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在提升模型解释性。提出ProtoSolo，仅用一个原型完成分类与解释，降低认知复杂度。**

- **链接: [http://arxiv.org/pdf/2506.19808v1](http://arxiv.org/pdf/2506.19808v1)**

> **作者:** Yitao Peng; Lianghua He; Die Hu
>
> **摘要:** In this paper, we propose ProtoSolo, a novel deep neural architecture for interpretable image classification inspired by prototypical networks such as ProtoPNet. Existing prototype networks usually rely on the collaborative decision-making of multiple prototypes to achieve the classification and interpretation of a single category. In contrast, ProtoSolo only requires the activation of a single prototype to complete the classification. This allows the network to explain each category decision by only providing the features that are most similar to the prototype of that category, significantly reducing the cognitive complexity of the explanation. Secondly, we propose a feature-based comparison method, which uses feature map instead of full-channel feature vector as the object of similarity comparison and prototype learning. This design enables ProtoSolo to utilize richer global information for classification while relying on a single prototype activation. In addition, we propose a non-prototype projection learning strategy, which preserves the information association between the prototype and the training image patches while avoiding the sharp change of the network structure caused by the projection operation, thus avoiding its negative impact on the classification performance. Experiments on the CUB-200-2011 and Stanford Cars datasets show that ProtoSolo achieves superior performance in classification tasks and reaches the best level in terms of cognitive complexity of explanations compared to state-of-the-art interpretable methods. The code is available at https://github.com/pyt19/ProtoSolo.
>
---
#### [new 076] Open-Vocabulary Camouflaged Object Segmentation with Cascaded Vision Language Models
- **分类: cs.CV**

- **简介: 该论文属于开放词汇伪装目标分割任务，解决视觉模糊和未见类别带来的挑战。提出级联框架，结合VLM提升分割与分类精度。**

- **链接: [http://arxiv.org/pdf/2506.19300v1](http://arxiv.org/pdf/2506.19300v1)**

> **作者:** Kai Zhao; Wubang Yuan; Zheng Wang; Guanyi Li; Xiaoqiang Zhu; Deng-ping Fan; Dan Zeng
>
> **摘要:** Open-Vocabulary Camouflaged Object Segmentation (OVCOS) seeks to segment and classify camouflaged objects from arbitrary categories, presenting unique challenges due to visual ambiguity and unseen categories.Recent approaches typically adopt a two-stage paradigm: first segmenting objects, then classifying the segmented regions using Vision Language Models (VLMs).However, these methods (1) suffer from a domain gap caused by the mismatch between VLMs' full-image training and cropped-region inference, and (2) depend on generic segmentation models optimized for well-delineated objects, making them less effective for camouflaged objects.Without explicit guidance, generic segmentation models often overlook subtle boundaries, leading to imprecise segmentation.In this paper,we introduce a novel VLM-guided cascaded framework to address these issues in OVCOS.For segmentation, we leverage the Segment Anything Model (SAM), guided by the VLM.Our framework uses VLM-derived features as explicit prompts to SAM, effectively directing attention to camouflaged regions and significantly improving localization accuracy.For classification, we avoid the domain gap introduced by hard cropping.Instead, we treat the segmentation output as a soft spatial prior via the alpha channel, which retains the full image context while providing precise spatial guidance, leading to more accurate and context-aware classification of camouflaged objects.The same VLM is shared across both segmentation and classification to ensure efficiency and semantic consistency.Extensive experiments on both OVCOS and conventional camouflaged object segmentation benchmarks demonstrate the clear superiority of our method, highlighting the effectiveness of leveraging rich VLM semantics for both segmentation and classification of camouflaged objects.
>
---
#### [new 077] Airway Skill Assessment with Spatiotemporal Attention Mechanisms Using Human Gaze
- **分类: cs.CV**

- **简介: 该论文属于医疗技能评估任务，旨在解决传统主观评估ETI方法的不足。通过结合人眼动数据与视频，使用注意力机制提升评估准确性和客观性。**

- **链接: [http://arxiv.org/pdf/2506.19306v1](http://arxiv.org/pdf/2506.19306v1)**

> **作者:** Jean-Paul Ainam; Rahul; Lora Cavuoto; Matthew Hackett; Jack Norfleet; Suvranu De
>
> **备注:** 13 pages, 6 figures, 14 equations,
>
> **摘要:** Airway management skills are critical in emergency medicine and are typically assessed through subjective evaluation, often failing to gauge competency in real-world scenarios. This paper proposes a machine learning-based approach for assessing airway skills, specifically endotracheal intubation (ETI), using human gaze data and video recordings. The proposed system leverages an attention mechanism guided by the human gaze to enhance the recognition of successful and unsuccessful ETI procedures. Visual masks were created from gaze points to guide the model in focusing on task-relevant areas, reducing irrelevant features. An autoencoder network extracts features from the videos, while an attention module generates attention from the visual masks, and a classifier outputs a classification score. This method, the first to use human gaze for ETI, demonstrates improved accuracy and efficiency over traditional methods. The integration of human gaze data not only enhances model performance but also offers a robust, objective assessment tool for clinical skills, particularly in high-stress environments such as military settings. The results show improvements in prediction accuracy, sensitivity, and trustworthiness, highlighting the potential for this approach to improve clinical training and patient outcomes in emergency medicine.
>
---
#### [new 078] Deblurring in the Wild: A Real-World Dataset from Smartphone High-Speed Videos
- **分类: cs.CV**

- **简介: 该论文属于图像去模糊任务，旨在解决真实场景下的运动模糊问题。通过手机高速视频构建大规模数据集，包含42,000对模糊与清晰图像，用于评估和提升去模糊模型的性能。**

- **链接: [http://arxiv.org/pdf/2506.19445v1](http://arxiv.org/pdf/2506.19445v1)**

> **作者:** Mahdi Mohd Hossain Noki; Syed Mumtahin Mahmud; Prothito Shovon Majumder; Abdul Mohaimen Al Radi; Md. Haider Ali; Md. Mosaddek Khan
>
> **备注:** 8 pages (without references), 3 figures. Dataset https://huggingface.co/datasets/masterda/SloMoBlur
>
> **摘要:** We introduce the largest real-world image deblurring dataset constructed from smartphone slow-motion videos. Using 240 frames captured over one second, we simulate realistic long-exposure blur by averaging frames to produce blurry images, while using the temporally centered frame as the sharp reference. Our dataset contains over 42,000 high-resolution blur-sharp image pairs, making it approximately 10 times larger than widely used datasets, with 8 times the amount of different scenes, including indoor and outdoor environments, with varying object and camera motions. We benchmark multiple state-of-the-art (SOTA) deblurring models on our dataset and observe significant performance degradation, highlighting the complexity and diversity of our benchmark. Our dataset serves as a challenging new benchmark to facilitate robust and generalizable deblurring models.
>
---
#### [new 079] Image Segmentation using Chan-Vese Active Contours
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，旨在解决噪声和弱边界图像的分割问题。通过改进Chan-Vese模型，利用区域强度差异进行轮廓演化，实现更准确的分割效果。**

- **链接: [http://arxiv.org/pdf/2506.19344v1](http://arxiv.org/pdf/2506.19344v1)**

> **作者:** Pranav Shenoy K. P
>
> **摘要:** This paper presents a comprehensive derivation and implementation of the Chan-Vese active contour model for image segmentation. The model, derived from the Mumford-Shah variational framework, evolves contours based on regional intensity differences rather than image gradients, making it highly effective for segmenting noisy images or images with weak boundaries. We provide a rigorous mathematical derivation of the level set formulation, including detailed treatment of each energy term using the divergence theorem and curve evolution theory. The resulting algorithm is implemented in Python using finite difference methods with special care to numerical stability, including an upwind entropy scheme and curvature-based regularization. Experimental results on medical and synthetic images demonstrate accurate segmentation, robustness to noise, and superior performance compared to classical edge-based methods. This study confirms the suitability of the Chan-Vese model for complex segmentation tasks and highlights its potential for use in real-world imaging applications.
>
---
#### [new 080] Capturing Fine-Grained Alignments Improves 3D Affordance Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D affordance检测任务，旨在解决点云与文本间细粒度对齐问题。提出LM-AD方法和AQM模块，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2506.19312v1](http://arxiv.org/pdf/2506.19312v1)**

> **作者:** Junsei Tokumitsu; Yuiga Wada
>
> **备注:** MVA 2025 (Oral)
>
> **摘要:** In this work, we address the challenge of affordance detection in 3D point clouds, a task that requires effectively capturing fine-grained alignments between point clouds and text. Existing methods often struggle to model such alignments, resulting in limited performance on standard benchmarks. A key limitation of these approaches is their reliance on simple cosine similarity between point cloud and text embeddings, which lacks the expressiveness needed for fine-grained reasoning. To address this limitation, we propose LM-AD, a novel method for affordance detection in 3D point clouds. Moreover, we introduce the Affordance Query Module (AQM), which efficiently captures fine-grained alignment between point clouds and text by leveraging a pretrained language model. We demonstrated that our method outperformed existing approaches in terms of accuracy and mean Intersection over Union on the 3D AffordanceNet dataset.
>
---
#### [new 081] Reading Smiles: Proxy Bias in Foundation Models for Facial Emotion Recognition
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文属于面部情绪识别任务，探讨基础模型在情绪推断中依赖的视觉线索是否具有心理依据。研究发现模型更关注眉毛位置等特征，存在潜在偏差和快捷学习问题。**

- **链接: [http://arxiv.org/pdf/2506.19079v1](http://arxiv.org/pdf/2506.19079v1)**

> **作者:** Iosif Tsangko; Andreas Triantafyllopoulos; Adem Abdelmoula; Adria Mallol-Ragolta; Bjoern W. Schuller
>
> **摘要:** Foundation Models (FMs) are rapidly transforming Affective Computing (AC), with Vision Language Models (VLMs) now capable of recognising emotions in zero shot settings. This paper probes a critical but underexplored question: what visual cues do these models rely on to infer affect, and are these cues psychologically grounded or superficially learnt? We benchmark varying scale VLMs on a teeth annotated subset of AffectNet dataset and find consistent performance shifts depending on the presence of visible teeth. Through structured introspection of, the best-performing model, i.e., GPT-4o, we show that facial attributes like eyebrow position drive much of its affective reasoning, revealing a high degree of internal consistency in its valence-arousal predictions. These patterns highlight the emergent nature of FMs behaviour, but also reveal risks: shortcut learning, bias, and fairness issues especially in sensitive domains like mental health and education.
>
---
#### [new 082] Memory-Augmented Incomplete Multimodal Survival Prediction via Cross-Slide and Gene-Attentive Hypergraph Learning
- **分类: cs.CV**

- **简介: 该论文属于癌症生存预测任务，解决多模态数据不平衡与缺失问题，提出融合病理与基因数据的框架，并引入记忆机制提升预测性能。**

- **链接: [http://arxiv.org/pdf/2506.19324v1](http://arxiv.org/pdf/2506.19324v1)**

> **作者:** Mingcheng Qu; Guang Yang; Donglin Di; Yue Gao; Tonghua Su; Yang Song; Lei Fan
>
> **备注:** accepted by MICCAI2025 code: https://github.com/MCPathology/M2Surv
>
> **摘要:** Multimodal pathology-genomic analysis is critical for cancer survival prediction. However, existing approaches predominantly integrate formalin-fixed paraffin-embedded (FFPE) slides with genomic data, while neglecting the availability of other preservation slides, such as Fresh Froze (FF) slides. Moreover, as the high-resolution spatial nature of pathology data tends to dominate the cross-modality fusion process, it hinders effective multimodal fusion and leads to modality imbalance challenges between pathology and genomics. These methods also typically require complete data modalities, limiting their clinical applicability with incomplete modalities, such as missing either pathology or genomic data. In this paper, we propose a multimodal survival prediction framework that leverages hypergraph learning to effectively integrate multi-WSI information and cross-modality interactions between pathology slides and genomics data while addressing modality imbalance. In addition, we introduce a memory mechanism that stores previously learned paired pathology-genomic features and dynamically compensates for incomplete modalities. Experiments on five TCGA datasets demonstrate that our model outperforms advanced methods by over 2.3% in C-Index. Under incomplete modality scenarios, our approach surpasses pathology-only (3.3%) and gene-only models (7.9%). Code: https://github.com/MCPathology/M2Surv
>
---
#### [new 083] MedErr-CT: A Visual Question Answering Benchmark for Identifying and Correcting Errors in CT Reports
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MedErr-CT基准，用于评估医学多模态大模型在CT报告错误识别与修正中的能力，解决医疗诊断准确性问题。**

- **链接: [http://arxiv.org/pdf/2506.19217v1](http://arxiv.org/pdf/2506.19217v1)**

> **作者:** Sunggu Kyung; Hyungbin Park; Jinyoung Seo; Jimin Sung; Jihyun Kim; Dongyeong Kim; Wooyoung Jo; Yoojin Nam; Sangah Park; Taehee Kwon; Sang Min Lee; Namkug Kim
>
> **备注:** 14 pages, 5 figures, submitted to CVPR 2025
>
> **摘要:** Computed Tomography (CT) plays a crucial role in clinical diagnosis, but the growing demand for CT examinations has raised concerns about diagnostic errors. While Multimodal Large Language Models (MLLMs) demonstrate promising comprehension of medical knowledge, their tendency to produce inaccurate information highlights the need for rigorous validation. However, existing medical visual question answering (VQA) benchmarks primarily focus on simple visual recognition tasks, lacking clinical relevance and failing to assess expert-level knowledge. We introduce MedErr-CT, a novel benchmark for evaluating medical MLLMs' ability to identify and correct errors in CT reports through a VQA framework. The benchmark includes six error categories - four vision-centric errors (Omission, Insertion, Direction, Size) and two lexical error types (Unit, Typo) - and is organized into three task levels: classification, detection, and correction. Using this benchmark, we quantitatively assess the performance of state-of-the-art 3D medical MLLMs, revealing substantial variation in their capabilities across different error types. Our benchmark contributes to the development of more reliable and clinically applicable MLLMs, ultimately helping reduce diagnostic errors and improve accuracy in clinical practice. The code and datasets are available at https://github.com/babbu3682/MedErr-CT.
>
---
#### [new 084] Self-Paced Collaborative and Adversarial Network for Unsupervised Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文属于无监督域适应任务，旨在解决源域与目标域分布不匹配的问题。提出CAN网络，结合协同与对抗学习，提升特征表示的域不变性和判别性。**

- **链接: [http://arxiv.org/pdf/2506.19267v1](http://arxiv.org/pdf/2506.19267v1)**

> **作者:** Weichen Zhang; Dong Xu; Wanli Ouyang; Wen Li
>
> **摘要:** This paper proposes a new unsupervised domain adaptation approach called Collaborative and Adversarial Network (CAN), which uses the domain-collaborative and domain-adversarial learning strategy for training the neural network. The domain-collaborative learning aims to learn domain-specific feature representation to preserve the discriminability for the target domain, while the domain adversarial learning aims to learn domain-invariant feature representation to reduce the domain distribution mismatch between the source and target domains. We show that these two learning strategies can be uniformly formulated as domain classifier learning with positive or negative weights on the losses. We then design a collaborative and adversarial training scheme, which automatically learns domain-specific representations from lower blocks in CNNs through collaborative learning and domain-invariant representations from higher blocks through adversarial learning. Moreover, to further enhance the discriminability in the target domain, we propose Self-Paced CAN (SPCAN), which progressively selects pseudo-labeled target samples for re-training the classifiers. We employ a self-paced learning strategy to select pseudo-labeled target samples in an easy-to-hard fashion. Comprehensive experiments on different benchmark datasets, Office-31, ImageCLEF-DA, and VISDA-2017 for the object recognition task, and UCF101-10 and HMDB51-10 for the video action recognition task, show our newly proposed approaches achieve the state-of-the-art performance, which clearly demonstrates the effectiveness of our proposed approaches for unsupervised domain adaptation.
>
---
#### [new 085] Damba-ST: Domain-Adaptive Mamba for Efficient Urban Spatio-Temporal Prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于城市时空预测任务，旨在解决跨域泛化与模型效率问题。提出Damba-ST模型，通过域自适应机制提升Mamba在不同城市环境中的性能。**

- **链接: [http://arxiv.org/pdf/2506.18939v1](http://arxiv.org/pdf/2506.18939v1)**

> **作者:** Rui An; Yifeng Zhang; Ziran Liang; Wenqi Fan; Yuxuan Liang; Xuequn Shang; Qing Li
>
> **摘要:** Training urban spatio-temporal foundation models that generalize well across diverse regions and cities is critical for deploying urban services in unseen or data-scarce regions. Recent studies have typically focused on fusing cross-domain spatio-temporal data to train unified Transformer-based models. However, these models suffer from quadratic computational complexity and high memory overhead, limiting their scalability and practical deployment. Inspired by the efficiency of Mamba, a state space model with linear time complexity, we explore its potential for efficient urban spatio-temporal prediction. However, directly applying Mamba as a spatio-temporal backbone leads to negative transfer and severe performance degradation. This is primarily due to spatio-temporal heterogeneity and the recursive mechanism of Mamba's hidden state updates, which limit cross-domain generalization. To overcome these challenges, we propose Damba-ST, a novel domain-adaptive Mamba-based model for efficient urban spatio-temporal prediction. Damba-ST retains Mamba's linear complexity advantage while significantly enhancing its adaptability to heterogeneous domains. Specifically, we introduce two core innovations: (1) a domain-adaptive state space model that partitions the latent representation space into a shared subspace for learning cross-domain commonalities and independent, domain-specific subspaces for capturing intra-domain discriminative features; (2) three distinct Domain Adapters, which serve as domain-aware proxies to bridge disparate domain distributions and facilitate the alignment of cross-domain commonalities. Extensive experiments demonstrate the generalization and efficiency of Damba-ST. It achieves state-of-the-art performance on prediction tasks and demonstrates strong zero-shot generalization, enabling seamless deployment in new urban environments without extensive retraining or fine-tuning.
>
---
#### [new 086] Sampling Matters in Explanations: Towards Trustworthy Attribution Analysis Building Block in Visual Models through Maximizing Explanation Certainty
- **分类: cs.CV**

- **简介: 该论文属于视觉模型解释任务，旨在提升归因分析的可信度。针对样本分布不匹配导致解释不确定性问题，提出一种通过抑制特征的采样方法，增强解释效果。**

- **链接: [http://arxiv.org/pdf/2506.19442v1](http://arxiv.org/pdf/2506.19442v1)**

> **作者:** Róisín Luo; James McDermott; Colm O'Riordan
>
> **备注:** Code: https://anonymous.4open.science/r/sampling_matters_reproducibility-BB60/
>
> **摘要:** Image attribution analysis seeks to highlight the feature representations learned by visual models such that the highlighted feature maps can reflect the pixel-wise importance of inputs. Gradient integration is a building block in the attribution analysis by integrating the gradients from multiple derived samples to highlight the semantic features relevant to inferences. Such a building block often combines with other information from visual models such as activation or attention maps to form ultimate explanations. Yet, our theoretical analysis demonstrates that the extent to the alignment of the sample distribution in gradient integration with respect to natural image distribution gives a lower bound of explanation certainty. Prior works add noise into images as samples and the noise distributions can lead to low explanation certainty. Counter-intuitively, our experiment shows that extra information can saturate neural networks. To this end, building trustworthy attribution analysis needs to settle the sample distribution misalignment problem. Instead of adding extra information into input images, we present a semi-optimal sampling approach by suppressing features from inputs. The sample distribution by suppressing features is approximately identical to the distribution of natural images. Our extensive quantitative evaluation on large scale dataset ImageNet affirms that our approach is effective and able to yield more satisfactory explanations against state-of-the-art baselines throughout all experimental models.
>
---
#### [new 087] Generate the Forest before the Trees -- A Hierarchical Diffusion model for Climate Downscaling
- **分类: cs.CV**

- **简介: 该论文属于气候降尺度任务，旨在解决传统方法计算成本高、效率低的问题。提出一种分层扩散模型（HDD），通过粗到细的采样策略提升效率并保持精度。**

- **链接: [http://arxiv.org/pdf/2506.19391v1](http://arxiv.org/pdf/2506.19391v1)**

> **作者:** Declan J. Curran; Sanaa Hobeichi; Hira Saleem; Hao Xue; Flora D. Salim
>
> **备注:** 8 pages
>
> **摘要:** Downscaling is essential for generating the high-resolution climate data needed for local planning, but traditional methods remain computationally demanding. Recent years have seen impressive results from AI downscaling models, particularly diffusion models, which have attracted attention due to their ability to generate ensembles and overcome the smoothing problem common in other AI methods. However, these models typically remain computationally intensive. We introduce a Hierarchical Diffusion Downscaling (HDD) model, which introduces an easily-extensible hierarchical sampling process to the diffusion framework. A coarse-to-fine hierarchy is imposed via a simple downsampling scheme. HDD achieves competitive accuracy on ERA5 reanalysis datasets and CMIP6 models, significantly reducing computational load by running on up to half as many pixels with competitive results. Additionally, a single model trained at 0.25{\deg} resolution transfers seamlessly across multiple CMIP6 models with much coarser resolution. HDD thus offers a lightweight alternative for probabilistic climate downscaling, facilitating affordable large-ensemble high-resolution climate projections. See a full code implementation at: https://github.com/HDD-Hierarchical-Diffusion-Downscaling/HDD-Hierarchical-Diffusion-Downscaling.
>
---
#### [new 088] Online camera-pose-free stereo endoscopic tissue deformation recovery with tissue-invariant vision-biomechanics consistency
- **分类: cs.CV**

- **简介: 该论文属于手术导航中的软组织变形恢复任务，解决因相机运动、遮挡等导致的变形建模难题。通过在线优化方法，实现无需相机位姿估计的稳定组织几何与变形建模。**

- **链接: [http://arxiv.org/pdf/2506.19388v1](http://arxiv.org/pdf/2506.19388v1)**

> **作者:** Jiahe Chen; Naoki Tomii; Ichiro Sakuma; Etsuko Kobayashi
>
> **摘要:** Tissue deformation recovery based on stereo endoscopic images is crucial for tool-tissue interaction analysis and benefits surgical navigation and autonomous soft tissue manipulation. Previous research suffers from the problems raised from camera motion, occlusion, large tissue deformation, lack of tissue-specific biomechanical priors, and reliance on offline processing. Unlike previous studies where the tissue geometry and deformation are represented by 3D points and displacements, the proposed method models tissue geometry as the 3D point and derivative map and tissue deformation as the 3D displacement and local deformation map. For a single surface point, 6 parameters are used to describe its rigid motion and 3 parameters for its local deformation. The method is formulated under the camera-centric setting, where all motions are regarded as the scene motion with respect to the camera. Inter-frame alignment is realized by optimizing the inter-frame deformation, making it unnecessary to estimate camera pose. The concept of the canonical map is introduced to optimize tissue geometry and deformation in an online approach. Quantitative and qualitative experiments were conducted using in vivo and ex vivo laparoscopic datasets. With the inputs of depth and optical flow, the method stably models tissue geometry and deformation even when the tissue is partially occluded or moving outside the field of view. Results show that the 3D reconstruction accuracy in the non-occluded and occluded areas reaches 0.37$\pm$0.27 mm and 0.39$\pm$0.21 mm in terms of surface distance, respectively. The method can also estimate surface strain distribution during various manipulations as an extra modality for mechanical-based analysis.
>
---
#### [new 089] Filling of incomplete sinograms from sparse PET detector configurations using a residual U-Net
- **分类: eess.IV; cs.CV; physics.med-ph**

- **简介: 该论文属于图像重建任务，旨在解决稀疏PET探测器配置导致的图像质量下降问题。通过引入残差U-Net网络填补缺失的sinogram数据，提升重建图像质量。**

- **链接: [http://arxiv.org/pdf/2506.19600v1](http://arxiv.org/pdf/2506.19600v1)**

> **作者:** Klara Leffler; Luigi Tommaso Luppino; Samuel Kuttner; Karin Söderkvist; Jan Axelsson
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Long axial field-of-view PET scanners offer increased field-of-view and sensitivity compared to traditional PET scanners. However, a significant cost is associated with the densely packed photodetectors required for the extended-coverage systems, limiting clinical utilisation. To mitigate the cost limitations, alternative sparse system configurations have been proposed, allowing an extended field-of-view PET design with detector costs similar to a standard PET system, albeit at the expense of image quality. In this work, we propose a deep sinogram restoration network to fill in the missing sinogram data. Our method utilises a modified Residual U-Net, trained on clinical PET scans from a GE Signa PET/MR, simulating the removal of 50% of the detectors in a chessboard pattern (retaining only 25% of all lines of response). The model successfully recovers missing counts, with a mean absolute error below two events per pixel, outperforming 2D interpolation in both sinogram and reconstructed image domain. Notably, the predicted sinograms exhibit a smoothing effect, leading to reconstructed images lacking sharpness in finer details. Despite these limitations, the model demonstrates a substantial capacity for compensating for the undersampling caused by the sparse detector configuration. This proof-of-concept study suggests that sparse detector configurations, combined with deep learning techniques, offer a viable alternative to conventional PET scanner designs. This approach supports the development of cost-effective, total body PET scanners, allowing a significant step forward in medical imaging technology.
>
---
#### [new 090] Noise Consistency Training: A Native Approach for One-Step Generator in Learning Additional Controls
- **分类: cs.LG; cs.CV; stat.ML**

- **简介: 该论文属于可控生成任务，旨在解决一阶段生成器适应新控制条件的问题。提出NCT方法，在不重新训练模型的情况下，通过噪声一致性损失实现高效可控生成。**

- **链接: [http://arxiv.org/pdf/2506.19741v1](http://arxiv.org/pdf/2506.19741v1)**

> **作者:** Yihong Luo; Shuchen Xue; Tianyang Hu; Jing Tang
>
> **摘要:** The pursuit of efficient and controllable high-quality content generation remains a central challenge in artificial intelligence-generated content (AIGC). While one-step generators, enabled by diffusion distillation techniques, offer excellent generation quality and computational efficiency, adapting them to new control conditions--such as structural constraints, semantic guidelines, or external inputs--poses a significant challenge. Conventional approaches often necessitate computationally expensive modifications to the base model and subsequent diffusion distillation. This paper introduces Noise Consistency Training (NCT), a novel and lightweight approach to directly integrate new control signals into pre-trained one-step generators without requiring access to original training images or retraining the base diffusion model. NCT operates by introducing an adapter module and employs a noise consistency loss in the noise space of the generator. This loss aligns the adapted model's generation behavior across noises that are conditionally dependent to varying degrees, implicitly guiding it to adhere to the new control. Theoretically, this training objective can be understood as minimizing the distributional distance between the adapted generator and the conditional distribution induced by the new conditions. NCT is modular, data-efficient, and easily deployable, relying only on the pre-trained one-step generator and a control signal model. Extensive experiments demonstrate that NCT achieves state-of-the-art controllable generation in a single forward pass, surpassing existing multi-step and distillation-based methods in both generation quality and computational efficiency. Code is available at https://github.com/Luo-Yihong/NCT
>
---
#### [new 091] SoK: Can Synthetic Images Replace Real Data? A Survey of Utility and Privacy of Synthetic Image Generation
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于隐私保护数据合成任务，旨在评估合成图像替代真实数据的可行性及隐私风险，通过系统分类与实验比较不同生成方法。**

- **链接: [http://arxiv.org/pdf/2506.19360v1](http://arxiv.org/pdf/2506.19360v1)**

> **作者:** Yunsung Chung; Yunbei Zhang; Nassir Marrouche; Jihun Hamm
>
> **备注:** Accepted at the 34th USENIX Security Symposium (USENIX Security '25). 21 pages, plus a 6-page appendix
>
> **摘要:** Advances in generative models have transformed the field of synthetic image generation for privacy-preserving data synthesis (PPDS). However, the field lacks a comprehensive survey and comparison of synthetic image generation methods across diverse settings. In particular, when we generate synthetic images for the purpose of training a classifier, there is a pipeline of generation-sampling-classification which takes private training as input and outputs the final classifier of interest. In this survey, we systematically categorize existing image synthesis methods, privacy attacks, and mitigations along this generation-sampling-classification pipeline. To empirically compare diverse synthesis approaches, we provide a benchmark with representative generative methods and use model-agnostic membership inference attacks (MIAs) as a measure of privacy risk. Through this study, we seek to answer critical questions in PPDS: Can synthetic data effectively replace real data? Which release strategy balances utility and privacy? Do mitigations improve the utility-privacy tradeoff? Which generative models perform best across different scenarios? With a systematic evaluation of diverse methods, our study provides actionable insights into the utility-privacy tradeoffs of synthetic data generation methods and guides the decision on optimal data releasing strategies for real-world applications.
>
---
#### [new 092] ConCM: Consistency-Driven Calibration and Matching for Few-Shot Class-Incremental Learning
- **分类: cs.LG; cs.CV; 68T40; I.2.6; I.4.9**

- **简介: 该论文属于少样本类增量学习任务，解决知识冲突问题。提出ConCM框架，通过一致性校准和动态结构匹配提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.19558v1](http://arxiv.org/pdf/2506.19558v1)**

> **作者:** QinZhe Wang; Zixuan Chen; Keke Huang; Xiu Su; Chunhua Yang; Chang Xu
>
> **备注:** 9 pages, 5 figures(Excluding the appendix)
>
> **摘要:** Few-Shot Class-Incremental Learning (FSCIL) requires models to adapt to novel classes with limited supervision while preserving learned knowledge. Existing prospective learning-based space construction methods reserve space to accommodate novel classes. However, prototype deviation and structure fixity limit the expressiveness of the embedding space. In contrast to fixed space reservation, we explore the optimization of feature-structure dual consistency and propose a Consistency-driven Calibration and Matching Framework (ConCM) that systematically mitigate the knowledge conflict inherent in FSCIL. Specifically, inspired by hippocampal associative memory, we design a memory-aware prototype calibration that extracts generalized semantic attributes from base classes and reintegrates them into novel classes to enhance the conceptual center consistency of features. Further, we propose dynamic structure matching, which adaptively aligns the calibrated features to a session-specific optimal manifold space, ensuring cross-session structure consistency. Theoretical analysis shows that our method satisfies both geometric optimality and maximum matching, thereby overcoming the need for class-number priors. On large-scale FSCIL benchmarks including mini-ImageNet and CUB200, ConCM achieves state-of-the-art performance, surpassing current optimal method by 3.20% and 3.68% in harmonic accuracy of incremental sessions.
>
---
#### [new 093] Fake or Real, Can Robots Tell? Evaluating Embodied Vision-Language Models on Real and 3D-Printed Objects
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于机器人场景理解任务，研究如何利用视觉-语言模型识别真实与3D打印物体，比较不同描述策略的效果。**

- **链接: [http://arxiv.org/pdf/2506.19579v1](http://arxiv.org/pdf/2506.19579v1)**

> **作者:** Federico Tavella; Kathryn Mearns; Angelo Cangelosi
>
> **摘要:** Robotic scene understanding increasingly relies on vision-language models (VLMs) to generate natural language descriptions of the environment. In this work, we present a comparative study of captioning strategies for tabletop scenes captured by a robotic arm equipped with an RGB camera. The robot collects images of objects from multiple viewpoints, and we evaluate several models that generate scene descriptions. We compare the performance of various captioning models, like BLIP and VLMs. Our experiments examine the trade-offs between single-view and multi-view captioning, and difference between recognising real-world and 3D printed objects. We quantitatively evaluate object identification accuracy, completeness, and naturalness of the generated captions. Results show that VLMs can be used in robotic settings where common objects need to be recognised, but fail to generalise to novel representations. Our findings provide practical insights into deploying foundation models for embodied agents in real-world settings.
>
---
#### [new 094] Xray2Xray: World Model from Chest X-rays with Volumetric Context
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出Xray2Xray模型，旨在从胸片中学习三维结构信息，解决二维图像的结构重叠问题，提升疾病诊断与风险预测效果。**

- **链接: [http://arxiv.org/pdf/2506.19055v1](http://arxiv.org/pdf/2506.19055v1)**

> **作者:** Zefan Yang; Xinrui Song; Xuanang Xu; Yongyi Shi; Ge Wang; Mannudeep K. Kalra; Pingkun Yan
>
> **摘要:** Chest X-rays (CXRs) are the most widely used medical imaging modality and play a pivotal role in diagnosing diseases. However, as 2D projection images, CXRs are limited by structural superposition, which constrains their effectiveness in precise disease diagnosis and risk prediction. To address the limitations of 2D CXRs, this study introduces Xray2Xray, a novel World Model that learns latent representations encoding 3D structural information from chest X-rays. Xray2Xray captures the latent representations of the chest volume by modeling the transition dynamics of X-ray projections across different angular positions with a vision model and a transition model. We employed the latent representations of Xray2Xray for downstream risk prediction and disease diagnosis tasks. Experimental results showed that Xray2Xray outperformed both supervised methods and self-supervised pretraining methods for cardiovascular disease risk estimation and achieved competitive performance in classifying five pathologies in CXRs. We also assessed the quality of Xray2Xray's latent representations through synthesis tasks and demonstrated that the latent representations can be used to reconstruct volumetric context.
>
---
#### [new 095] NAADA: A Noise-Aware Attention Denoising Autoencoder for Dental Panoramic Radiographs
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于图像去噪任务，旨在解决 dental radiographs 中细节丢失问题。提出 NAADA 网络，通过噪声感知注意力机制提升去噪效果。**

- **链接: [http://arxiv.org/pdf/2506.19387v1](http://arxiv.org/pdf/2506.19387v1)**

> **作者:** Khuram Naveed; Bruna Neves de Freitas; Ruben Pauwels
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** Convolutional denoising autoencoders (DAEs) are powerful tools for image restoration. However, they inherit a key limitation of convolutional neural networks (CNNs): they tend to recover low-frequency features, such as smooth regions, more effectively than high-frequency details. This leads to the loss of fine details, which is particularly problematic in dental radiographs where preserving subtle anatomical structures is crucial. While self-attention mechanisms can help mitigate this issue by emphasizing important features, conventional attention methods often prioritize features corresponding to cleaner regions and may overlook those obscured by noise. To address this limitation, we propose a noise-aware self-attention method, which allows the model to effectively focus on and recover key features even within noisy regions. Building on this approach, we introduce the noise-aware attention-enhanced denoising autoencoder (NAADA) network for enhancing noisy panoramic dental radiographs. Compared with the recent state of the art (and much heavier) methods like Uformer, MResDNN etc., our method improves the reconstruction of fine details, ensuring better image quality and diagnostic accuracy.
>
---
#### [new 096] Quantitative Benchmarking of Anomaly Detection Methods in Digital Pathology
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于数字病理学中的异常检测任务，旨在解决病理图像独特挑战下的检测问题。通过实验评估多种方法，建立基准以指导后续研究。**

- **链接: [http://arxiv.org/pdf/2506.19234v1](http://arxiv.org/pdf/2506.19234v1)**

> **作者:** Can Cui; Xindong Zheng; Ruining Deng; Quan Liu; Tianyuan Yao; Keith T Wilson; Lori A Coburn; Bennett A Landman; Haichun Yang; Yaohong Wang; Yuankai Huo
>
> **摘要:** Anomaly detection has been widely studied in the context of industrial defect inspection, with numerous methods developed to tackle a range of challenges. In digital pathology, anomaly detection holds significant potential for applications such as rare disease identification, artifact detection, and biomarker discovery. However, the unique characteristics of pathology images, such as their large size, multi-scale structures, stain variability, and repetitive patterns, introduce new challenges that current anomaly detection algorithms struggle to address. In this quantitative study, we benchmark over 20 classical and prevalent anomaly detection methods through extensive experiments. We curated five digital pathology datasets, both real and synthetic, to systematically evaluate these approaches. Our experiments investigate the influence of image scale, anomaly pattern types, and training epoch selection strategies on detection performance. The results provide a detailed comparison of each method's strengths and limitations, establishing a comprehensive benchmark to guide future research in anomaly detection for digital pathology images.
>
---
#### [new 097] Staining normalization in histopathology: Method benchmarking using multicenter dataset
- **分类: eess.IV; cs.CV; q-bio.TO; I.2.1; I.4.0**

- **简介: 该论文属于图像处理任务，旨在解决H&E染色病理图像的染色差异问题。通过多中心数据集对比不同归一化方法的效果。**

- **链接: [http://arxiv.org/pdf/2506.19106v1](http://arxiv.org/pdf/2506.19106v1)**

> **作者:** Umair Khan; Jouni Härkönen; Marjukka Friman; Leena Latonen; Teijo Kuopio; Pekka Ruusuvuori
>
> **备注:** 18 pages, 9 figures
>
> **摘要:** Hematoxylin and Eosin (H&E) has been the gold standard in tissue analysis for decades, however, tissue specimens stained in different laboratories vary, often significantly, in appearance. This variation poses a challenge for both pathologists' and AI-based downstream analysis. Minimizing stain variation computationally is an active area of research. To further investigate this problem, we collected a unique multi-center tissue image dataset, wherein tissue samples from colon, kidney, and skin tissue blocks were distributed to 66 different labs for routine H&E staining. To isolate staining variation, other factors affecting the tissue appearance were kept constant. Further, we used this tissue image dataset to compare the performance of eight different stain normalization methods, including four traditional methods, namely, histogram matching, Macenko, Vahadane, and Reinhard normalization, and two deep learning-based methods namely CycleGAN and Pixp2pix, both with two variants each. We used both quantitative and qualitative evaluation to assess the performance of these methods. The dataset's inter-laboratory staining variation could also guide strategies to improve model generalizability through varied training data
>
---
#### [new 098] Explicit Residual-Based Scalable Image Coding for Humans and Machines
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像压缩任务，旨在提升人类和机器视觉的可扩展编码效率。通过引入显式残差机制，提出两种方法以优化压缩性能与灵活性。**

- **链接: [http://arxiv.org/pdf/2506.19297v1](http://arxiv.org/pdf/2506.19297v1)**

> **作者:** Yui Tatsumi; Ziyue Zeng; Hiroshi Watanabe
>
> **摘要:** Scalable image compression is a technique that progressively reconstructs multiple versions of an image for different requirements. In recent years, images have increasingly been consumed not only by humans but also by image recognition models. This shift has drawn growing attention to scalable image compression methods that serve both machine and human vision (ICMH). Many existing models employ neural network-based codecs, known as learned image compression, and have made significant strides in this field by carefully designing the loss functions. In some cases, however, models are overly reliant on their learning capacity, and their architectural design is not sufficiently considered. In this paper, we enhance the coding efficiency and interpretability of ICMH framework by integrating an explicit residual compression mechanism, which is commonly employed in resolution scalable coding methods such as JPEG2000. Specifically, we propose two complementary methods: Feature Residual-based Scalable Coding (FR-ICMH) and Pixel Residual-based Scalable Coding (PR-ICMH). These proposed methods are applicable to various machine vision tasks. Moreover, they provide flexibility to choose between encoder complexity and compression performance, making it adaptable to diverse application requirements. Experimental results demonstrate the effectiveness of our proposed methods, with PR-ICMH achieving up to 29.57% BD-rate savings over the previous work.
>
---
#### [new 099] Angio-Diff: Learning a Self-Supervised Adversarial Diffusion Model for Angiographic Geometry Generation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在解决血管造影图像合成难题。通过自监督扩散模型和参数化血管模型，提升合成图像的几何准确性。**

- **链接: [http://arxiv.org/pdf/2506.19455v1](http://arxiv.org/pdf/2506.19455v1)**

> **作者:** Zhifeng Wang; Renjiao Yi; Xin Wen; Chenyang Zhu; Kai Xu; Kunlun He
>
> **摘要:** Vascular diseases pose a significant threat to human health, with X-ray angiography established as the gold standard for diagnosis, allowing for detailed observation of blood vessels. However, angiographic X-rays expose personnel and patients to higher radiation levels than non-angiographic X-rays, which are unwanted. Thus, modality translation from non-angiographic to angiographic X-rays is desirable. Data-driven deep approaches are hindered by the lack of paired large-scale X-ray angiography datasets. While making high-quality vascular angiography synthesis crucial, it remains challenging. We find that current medical image synthesis primarily operates at pixel level and struggles to adapt to the complex geometric structure of blood vessels, resulting in unsatisfactory quality of blood vessel image synthesis, such as disconnections or unnatural curvatures. To overcome this issue, we propose a self-supervised method via diffusion models to transform non-angiographic X-rays into angiographic X-rays, mitigating data shortages for data-driven approaches. Our model comprises a diffusion model that learns the distribution of vascular data from diffusion latent, a generator for vessel synthesis, and a mask-based adversarial module. To enhance geometric accuracy, we propose a parametric vascular model to fit the shape and distribution of blood vessels. The proposed method contributes a pipeline and a synthetic dataset for X-ray angiography. We conducted extensive comparative and ablation experiments to evaluate the Angio-Diff. The results demonstrate that our method achieves state-of-the-art performance in synthetic angiography image quality and more accurately synthesizes the geometric structure of blood vessels. The code is available at https://github.com/zfw-cv/AngioDiff.
>
---
#### [new 100] Convergent and divergent connectivity patterns of the arcuate fasciculus in macaques and humans
- **分类: q-bio.NC; cs.CV; eess.IV**

- **简介: 该论文属于神经科学领域，旨在比较人类与猕猴弓状束的连接模式。通过多种成像技术分析其结构差异，揭示人类语言网络的进化基础。**

- **链接: [http://arxiv.org/pdf/2506.19266v1](http://arxiv.org/pdf/2506.19266v1)**

> **作者:** Jiahao Huang; Ruifeng Li; Wenwen Yu; Anan Li; Xiangning Li; Mingchao Yan; Lei Xie; Qingrun Zeng; Xueyan Jia; Shuxin Wang; Ronghui Ju; Feng Chen; Qingming Luo; Hui Gong; Xiaoquan Yang; Yuanjing Feng; Zheng Wang
>
> **备注:** 34 pages, 6 figures
>
> **摘要:** The organization and connectivity of the arcuate fasciculus (AF) in nonhuman primates remain contentious, especially concerning how its anatomy diverges from that of humans. Here, we combined cross-scale single-neuron tracing - using viral-based genetic labeling and fluorescence micro-optical sectioning tomography in macaques (n = 4; age 3 - 11 years) - with whole-brain tractography from 11.7T diffusion MRI. Complemented by spectral embedding analysis of 7.0T MRI in humans, we performed a comparative connectomic analysis of the AF across species. We demonstrate that the macaque AF originates in the temporal-parietal cortex, traverses the auditory cortex and parietal operculum, and projects into prefrontal regions. In contrast, the human AF exhibits greater expansion into the middle temporal gyrus and stronger prefrontal and parietal operculum connectivity - divergences quantified by Kullback-Leibler analysis that likely underpin the evolutionary specialization of human language networks. These interspecies differences - particularly the human AF's broader temporal integration and strengthened frontoparietal linkages - suggest a connectivity-based substrate for the emergence of advanced language processing unique to humans. Furthermore, our findings offer a neuroanatomical framework for understanding AF-related disorders such as aphasia and dyslexia, where aberrant connectivity disrupts language function.
>
---
#### [new 101] ReCoGNet: Recurrent Context-Guided Network for 3D MRI Prostate Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决3D MRI前列腺分割中的准确性与鲁棒性问题。提出一种结合深度学习与循环结构的混合模型，提升分割效果。**

- **链接: [http://arxiv.org/pdf/2506.19687v1](http://arxiv.org/pdf/2506.19687v1)**

> **作者:** Ahmad Mustafa; Reza Rastegar; Ghassan AlRegib
>
> **摘要:** Prostate gland segmentation from T2-weighted MRI is a critical yet challenging task in clinical prostate cancer assessment. While deep learning-based methods have significantly advanced automated segmentation, most conventional approaches-particularly 2D convolutional neural networks (CNNs)-fail to leverage inter-slice anatomical continuity, limiting their accuracy and robustness. Fully 3D models offer improved spatial coherence but require large amounts of annotated data, which is often impractical in clinical settings. To address these limitations, we propose a hybrid architecture that models MRI sequences as spatiotemporal data. Our method uses a deep, pretrained DeepLabV3 backbone to extract high-level semantic features from each MRI slice and a recurrent convolutional head, built with ConvLSTM layers, to integrate information across slices while preserving spatial structure. This combination enables context-aware segmentation with improved consistency, particularly in data-limited and noisy imaging conditions. We evaluate our method on the PROMISE12 benchmark under both clean and contrast-degraded test settings. Compared to state-of-the-art 2D and 3D segmentation models, our approach demonstrates superior performance in terms of precision, recall, Intersection over Union (IoU), and Dice Similarity Coefficient (DSC), highlighting its potential for robust clinical deployment.
>
---
#### [new 102] Systematic Review of Pituitary Gland and Pituitary Adenoma Automatic Segmentation Techniques in Magnetic Resonance Imaging
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在提高MRI中垂体腺瘤和垂体的自动分割精度。通过综述34项研究，分析不同方法的性能。**

- **链接: [http://arxiv.org/pdf/2506.19797v1](http://arxiv.org/pdf/2506.19797v1)**

> **作者:** Mubaraq Yakubu; Navodini Wijethilake; Jonathan Shapey; Andrew King; Alexander Hammers
>
> **摘要:** Purpose: Accurate segmentation of both the pituitary gland and adenomas from magnetic resonance imaging (MRI) is essential for diagnosis and treatment of pituitary adenomas. This systematic review evaluates automatic segmentation methods for improving the accuracy and efficiency of MRI-based segmentation of pituitary adenomas and the gland itself. Methods: We reviewed 34 studies that employed automatic and semi-automatic segmentation methods. We extracted and synthesized data on segmentation techniques and performance metrics (such as Dice overlap scores). Results: The majority of reviewed studies utilized deep learning approaches, with U-Net-based models being the most prevalent. Automatic methods yielded Dice scores of 0.19--89.00\% for pituitary gland and 4.60--96.41\% for adenoma segmentation. Semi-automatic methods reported 80.00--92.10\% for pituitary gland and 75.90--88.36\% for adenoma segmentation. Conclusion: Most studies did not report important metrics such as MR field strength, age and adenoma size. Automated segmentation techniques such as U-Net-based models show promise, especially for adenoma segmentation, but further improvements are needed to achieve consistently good performance in small structures like the normal pituitary gland. Continued innovation and larger, diverse datasets are likely critical to enhancing clinical applicability.
>
---
#### [new 103] KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MA**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型的幻觉问题。通过引入事实奖励机制，KnowRL提升模型推理过程中的事实准确性。**

- **链接: [http://arxiv.org/pdf/2506.19807v1](http://arxiv.org/pdf/2506.19807v1)**

> **作者:** Baochang Ren; Shuofei Qiao; Wenhao Yu; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** Large Language Models (LLMs), particularly slow-thinking models, often exhibit severe hallucination, outputting incorrect content due to an inability to accurately recognize knowledge boundaries during reasoning. While Reinforcement Learning (RL) can enhance complex reasoning abilities, its outcome-oriented reward mechanism often lacks factual supervision over the thinking process, further exacerbating the hallucination problem. To address the high hallucination in slow-thinking models, we propose Knowledge-enhanced RL, KnowRL. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. This targeted factual input during RL training enables the model to learn and internalize fact-based reasoning strategies. By directly rewarding adherence to facts within the reasoning steps, KnowRL fosters a more reliable thinking process. Experimental results on three hallucination evaluation datasets and two reasoning evaluation datasets demonstrate that KnowRL effectively mitigates hallucinations in slow-thinking models while maintaining their original strong reasoning capabilities. Our code is available at https://github.com/zjunlp/KnowRL.
>
---
#### [new 104] Uncovering Conceptual Blindspots in Generative Image Models Using Sparse Autoencoders
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文属于图像生成模型研究，旨在解决模型在生成图像时存在的概念盲点问题。通过稀疏自编码器分析概念偏差，识别模型与真实数据间的差异。**

- **链接: [http://arxiv.org/pdf/2506.19708v1](http://arxiv.org/pdf/2506.19708v1)**

> **作者:** Matyas Bohacek; Thomas Fel; Maneesh Agrawala; Ekdeep Singh Lubana
>
> **摘要:** Despite their impressive performance, generative image models trained on large-scale datasets frequently fail to produce images with seemingly simple concepts -- e.g., human hands or objects appearing in groups of four -- that are reasonably expected to appear in the training data. These failure modes have largely been documented anecdotally, leaving open the question of whether they reflect idiosyncratic anomalies or more structural limitations of these models. To address this, we introduce a systematic approach for identifying and characterizing "conceptual blindspots" -- concepts present in the training data but absent or misrepresented in a model's generations. Our method leverages sparse autoencoders (SAEs) to extract interpretable concept embeddings, enabling a quantitative comparison of concept prevalence between real and generated images. We train an archetypal SAE (RA-SAE) on DINOv2 features with 32,000 concepts -- the largest such SAE to date -- enabling fine-grained analysis of conceptual disparities. Applied to four popular generative models (Stable Diffusion 1.5/2.1, PixArt, and Kandinsky), our approach reveals specific suppressed blindspots (e.g., bird feeders, DVD discs, and whitespaces on documents) and exaggerated blindspots (e.g., wood background texture and palm trees). At the individual datapoint level, we further isolate memorization artifacts -- instances where models reproduce highly specific visual templates seen during training. Overall, we propose a theoretically grounded framework for systematically identifying conceptual blindspots in generative models by assessing their conceptual fidelity with respect to the underlying data-generating process.
>
---
#### [new 105] Orthogonal Finetuning Made Scalable
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于模型微调任务，解决OFT计算效率低的问题，通过输入中心化和参数化改进，提升训练速度和降低内存消耗。**

- **链接: [http://arxiv.org/pdf/2506.19847v1](http://arxiv.org/pdf/2506.19847v1)**

> **作者:** Zeju Qiu; Weiyang Liu; Adrian Weller; Bernhard Schölkopf
>
> **备注:** Technical report (17 pages, 7 figures, project page: https://spherelab.ai/oftv2/)
>
> **摘要:** Orthogonal finetuning (OFT) offers highly parameter-efficient adaptation while preventing catastrophic forgetting, but its high runtime and memory demands limit practical deployment. We identify the core computational bottleneck in OFT as its weight-centric implementation, which relies on costly matrix-matrix multiplications with cubic complexity. To overcome this, we propose OFTv2, an input-centric reformulation that instead uses matrix-vector multiplications (i.e., matrix-free computation), reducing the computational cost to quadratic. We further introduce the Cayley-Neumann parameterization, an efficient orthogonal parameterization that approximates the matrix inversion in Cayley transform via a truncated Neumann series. These modifications allow OFTv2 to achieve up to 10x faster training and 3x lower GPU memory usage without compromising performance. In addition, we extend OFTv2 to support finetuning quantized foundation models and show that it outperforms the popular QLoRA in training stability, efficiency, and memory usage.
>
---
#### [new 106] A Deep Learning Based Method for Fast Registration of Cardiac Magnetic Resonance Images
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学图像配准任务，旨在解决心脏MRI快速准确配准问题。提出一种轻量级深度学习模型FLIR，提升配准效率与精度。**

- **链接: [http://arxiv.org/pdf/2506.19167v1](http://arxiv.org/pdf/2506.19167v1)**

> **作者:** Benjamin Graham
>
> **摘要:** Image registration is used in many medical image analysis applications, such as tracking the motion of tissue in cardiac images, where cardiac kinematics can be an indicator of tissue health. Registration is a challenging problem for deep learning algorithms because ground truth transformations are not feasible to create, and because there are potentially multiple transformations that can produce images that appear correlated with the goal. Unsupervised methods have been proposed to learn to predict effective transformations, but these methods take significantly longer to predict than established baseline methods. For a deep learning method to see adoption in wider research and clinical settings, it should be designed to run in a reasonable time on common, mid-level hardware. Fast methods have been proposed for the task of image registration but often use patch-based methods which can affect registration accuracy for a highly dynamic organ such as the heart. In this thesis, a fast, volumetric registration model is proposed for the use of quantifying cardiac strain. The proposed Deep Learning Neural Network (DLNN) is designed to utilize an architecture that can compute convolutions incredibly efficiently, allowing the model to achieve registration fidelity similar to other state-of-the-art models while taking a fraction of the time to perform inference. The proposed fast and lightweight registration (FLIR) model is used to predict tissue motion which is then used to quantify the non-uniform strain experienced by the tissue. For acquisitions taken from the same patient at approximately the same time, it would be expected that strain values measured between the acquisitions would have very small differences. Using this metric, strain values computed using the FLIR method are shown to be very consistent.
>
---
#### [new 107] Look to Locate: Vision-Based Multisensory Navigation with 3-D Digital Maps for GNSS-Challenged Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于导航任务，解决GNSS受限环境下的车辆定位问题。通过融合视觉与3D地图，提升定位精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.19827v1](http://arxiv.org/pdf/2506.19827v1)**

> **作者:** Ola Elmaghraby; Eslam Mounier; Paulo Ricardo Marques de Araujo; Aboelmagd Noureldin
>
> **摘要:** In Global Navigation Satellite System (GNSS)-denied environments such as indoor parking structures or dense urban canyons, achieving accurate and robust vehicle positioning remains a significant challenge. This paper proposes a cost-effective, vision-based multi-sensor navigation system that integrates monocular depth estimation, semantic filtering, and visual map registration (VMR) with 3-D digital maps. Extensive testing in real-world indoor and outdoor driving scenarios demonstrates the effectiveness of the proposed system, achieving sub-meter accuracy of 92% indoors and more than 80% outdoors, with consistent horizontal positioning and heading average root mean-square errors of approximately 0.98 m and 1.25 {\deg}, respectively. Compared to the baselines examined, the proposed solution significantly reduced drift and improved robustness under various conditions, achieving positioning accuracy improvements of approximately 88% on average. This work highlights the potential of cost-effective monocular vision systems combined with 3D maps for scalable, GNSS-independent navigation in land vehicles.
>
---
#### [new 108] Learning from Anatomy: Supervised Anatomical Pretraining (SAP) for Improved Metastatic Bone Disease Segmentation in Whole-Body MRI
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决全身影像中转移性骨病的准确分割问题。通过监督解剖预训练方法提升分割效果。**

- **链接: [http://arxiv.org/pdf/2506.19590v1](http://arxiv.org/pdf/2506.19590v1)**

> **作者:** Joris Wuts; Jakub Ceranka; Nicolas Michoux; Frédéric Lecouvet; Jef Vandemeulebroucke
>
> **备注:** This preprint is currently under review at *Computers in Biology and Medicine* (Elsevier). This version has not been peer-reviewed
>
> **摘要:** The segmentation of metastatic bone disease (MBD) in whole-body MRI (WB-MRI) is a challenging problem. Due to varying appearances and anatomical locations of lesions, ambiguous boundaries, and severe class imbalance, obtaining reliable segmentations requires large, well-annotated datasets capturing lesion variability. Generating such datasets requires substantial time and expertise, and is prone to error. While self-supervised learning (SSL) can leverage large unlabeled datasets, learned generic representations often fail to capture the nuanced features needed for accurate lesion detection. In this work, we propose a Supervised Anatomical Pretraining (SAP) method that learns from a limited dataset of anatomical labels. First, an MRI-based skeletal segmentation model is developed and trained on WB-MRI scans from healthy individuals for high-quality skeletal delineation. Then, we compare its downstream efficacy in segmenting MBD on a cohort of 44 patients with metastatic prostate cancer, against both a baseline random initialization and a state-of-the-art SSL method. SAP significantly outperforms both the baseline and SSL-pretrained models, achieving a normalized surface Dice of 0.76 and a Dice coefficient of 0.64. The method achieved a lesion detection F2 score of 0.44, improving on 0.24 (baseline) and 0.31 (SSL). When considering only clinically relevant lesions larger than 1~ml, SAP achieves a detection sensitivity of 100% in 28 out of 32 patients. Learning bone morphology from anatomy yields an effective and domain-relevant inductive bias that can be leveraged for the downstream segmentation task of bone lesions. All code and models are made publicly available.
>
---
#### [new 109] Experimental Assessment of Neural 3D Reconstruction for Small UAV-based Applications
- **分类: cs.ET; cs.AI; cs.CV; cs.NI; eess.IV**

- **简介: 该论文属于3D重建任务，旨在解决小型无人机在受限环境中高精度三维建模的问题。通过集成神经3D重建技术提升重建质量。**

- **链接: [http://arxiv.org/pdf/2506.19491v1](http://arxiv.org/pdf/2506.19491v1)**

> **作者:** Genís Castillo Gómez-Raya; Álmos Veres-Vitályos; Filip Lemic; Pablo Royo; Mario Montagud; Sergi Fernández; Sergi Abadal; Xavier Costa-Pérez
>
> **备注:** 6 pages, 7 figures, 2 tables, accepted at IEEE International Symposium on Personal, Indoor and Mobile Radio Communications 2025
>
> **摘要:** The increasing miniaturization of Unmanned Aerial Vehicles (UAVs) has expanded their deployment potential to indoor and hard-to-reach areas. However, this trend introduces distinct challenges, particularly in terms of flight dynamics and power consumption, which limit the UAVs' autonomy and mission capabilities. This paper presents a novel approach to overcoming these limitations by integrating Neural 3D Reconstruction (N3DR) with small UAV systems for fine-grained 3-Dimensional (3D) digital reconstruction of small static objects. Specifically, we design, implement, and evaluate an N3DR-based pipeline that leverages advanced models, i.e., Instant-ngp, Nerfacto, and Splatfacto, to improve the quality of 3D reconstructions using images of the object captured by a fleet of small UAVs. We assess the performance of the considered models using various imagery and pointcloud metrics, comparing them against the baseline Structure from Motion (SfM) algorithm. The experimental results demonstrate that the N3DR-enhanced pipeline significantly improves reconstruction quality, making it feasible for small UAVs to support high-precision 3D mapping and anomaly detection in constrained environments. In more general terms, our results highlight the potential of N3DR in advancing the capabilities of miniaturized UAV systems.
>
---
#### [new 110] Reconsidering Explicit Longitudinal Mammography Alignment for Enhanced Breast Cancer Risk Prediction
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在提升乳腺癌风险预测。解决纵向乳腺X光片对齐问题，通过对比输入空间与表示空间对齐效果，提出更优的联合优化方法。**

- **链接: [http://arxiv.org/pdf/2506.19363v1](http://arxiv.org/pdf/2506.19363v1)**

> **作者:** Solveig Thrun; Stine Hansen; Zijun Sun; Nele Blum; Suaiba A. Salahuddin; Kristoffer Wickstrøm; Elisabeth Wetzer; Robert Jenssen; Maik Stille; Michael Kampffmeyer
>
> **备注:** MICCAI 2025, early accepted
>
> **摘要:** Regular mammography screening is essential for early breast cancer detection. Deep learning-based risk prediction methods have sparked interest to adjust screening intervals for high-risk groups. While early methods focused only on current mammograms, recent approaches leverage the temporal aspect of screenings to track breast tissue changes over time, requiring spatial alignment across different time points. Two main strategies for this have emerged: explicit feature alignment through deformable registration and implicit learned alignment using techniques like transformers, with the former providing more control. However, the optimal approach for explicit alignment in mammography remains underexplored. In this study, we provide insights into where explicit alignment should occur (input space vs. representation space) and if alignment and risk prediction should be jointly optimized. We demonstrate that jointly learning explicit alignment in representation space while optimizing risk estimation performance, as done in the current state-of-the-art approach, results in a trade-off between alignment quality and predictive performance and show that image-level alignment is superior to representation-level alignment, leading to better deformation field quality and enhanced risk prediction accuracy. The code is available at https://github.com/sot176/Longitudinal_Mammogram_Alignment.git.
>
---
#### [new 111] CronusVLA: Transferring Latent Motion Across Time for Multi-Frame Prediction in Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机械操作中的多帧预测任务，解决单帧观察限制问题，通过引入多帧运动信息提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.19816v1](http://arxiv.org/pdf/2506.19816v1)**

> **作者:** Hao Li; Shuai Yang; Yilun Chen; Yang Tian; Xiaoda Yang; Xinyi Chen; Hanqing Wang; Tai Wang; Feng Zhao; Dahua Lin; Jiangmiao Pang
>
> **备注:** 36 pages, 21 figures
>
> **摘要:** Recent vision-language-action (VLA) models built on pretrained vision-language models (VLMs) have demonstrated strong generalization across manipulation tasks. However, they remain constrained by a single-frame observation paradigm and cannot fully benefit from the motion information offered by aggregated multi-frame historical observations, as the large vision-language backbone introduces substantial computational cost and inference latency. We propose CronusVLA, a unified framework that extends single-frame VLA models to the multi-frame paradigm through an efficient post-training stage. CronusVLA comprises three key components: (1) single-frame pretraining on large-scale embodied datasets with autoregressive action tokens prediction, which establishes an embodied vision-language foundation; (2) multi-frame encoding, adapting the prediction of vision-language backbones from discrete action tokens to motion features during post-training, and aggregating motion features from historical frames into a feature chunking; (3) cross-frame decoding, which maps the feature chunking to accurate actions via a shared decoder with cross-attention. By reducing redundant token computation and caching past motion features, CronusVLA achieves efficient inference. As an application of motion features, we further propose an action adaptation mechanism based on feature-action retrieval to improve model performance during finetuning. CronusVLA achieves state-of-the-art performance on SimplerEnv with 70.9% success rate, and 12.7% improvement over OpenVLA on LIBERO. Real-world Franka experiments also show the strong performance and robustness.
>
---
#### [new 112] Deformable Medical Image Registration with Effective Anatomical Structure Representation and Divide-and-Conquer Network
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像配准任务，旨在解决传统方法在ROI表示和独立对齐上的不足。提出EASR-DCN方法，通过有效ROI表示和分治网络实现无监督配准，提升精度与变形减少效果。**

- **链接: [http://arxiv.org/pdf/2506.19222v1](http://arxiv.org/pdf/2506.19222v1)**

> **作者:** Xinke Ma; Yongsheng Pan; Qingjie Zeng; Mengkang Lu; Bolysbek Murat Yerzhanuly; Bazargul Matkerim; Yong Xia
>
> **摘要:** Effective representation of Regions of Interest (ROI) and independent alignment of these ROIs can significantly enhance the performance of deformable medical image registration (DMIR). However, current learning-based DMIR methods have limitations. Unsupervised techniques disregard ROI representation and proceed directly with aligning pairs of images, while weakly-supervised methods heavily depend on label constraints to facilitate registration. To address these issues, we introduce a novel ROI-based registration approach named EASR-DCN. Our method represents medical images through effective ROIs and achieves independent alignment of these ROIs without requiring labels. Specifically, we first used a Gaussian mixture model for intensity analysis to represent images using multiple effective ROIs with distinct intensities. Furthermore, we propose a novel Divide-and-Conquer Network (DCN) to process these ROIs through separate channels to learn feature alignments for each ROI. The resultant correspondences are seamlessly integrated to generate a comprehensive displacement vector field. Extensive experiments were performed on three MRI and one CT datasets to showcase the superior accuracy and deformation reduction efficacy of our EASR-DCN. Compared to VoxelMorph, our EASR-DCN achieved improvements of 10.31\% in the Dice score for brain MRI, 13.01\% for cardiac MRI, and 5.75\% for hippocampus MRI, highlighting its promising potential for clinical applications. The code for this work will be released upon acceptance of the paper.
>
---
#### [new 113] Virtual Memory for 3D Gaussian Splatting
- **分类: cs.GR; cs.CV; cs.HC**

- **简介: 该论文属于3D重建任务，解决大场景渲染内存不足问题，通过虚拟内存技术动态加载可见高斯分布，提升渲染效率。**

- **链接: [http://arxiv.org/pdf/2506.19415v1](http://arxiv.org/pdf/2506.19415v1)**

> **作者:** Jonathan Haberl; Philipp Fleck; Clemens Arth
>
> **备注:** Based on the Master Thesis from Jonathan Haberl from 2024, Submitted to TVCG in Feb. 2025;
>
> **摘要:** 3D Gaussian Splatting represents a breakthrough in the field of novel view synthesis. It establishes Gaussians as core rendering primitives for highly accurate real-world environment reconstruction. Recent advances have drastically increased the size of scenes that can be created. In this work, we present a method for rendering large and complex 3D Gaussian Splatting scenes using virtual memory. By leveraging well-established virtual memory and virtual texturing techniques, our approach efficiently identifies visible Gaussians and dynamically streams them to the GPU just in time for real-time rendering. Selecting only the necessary Gaussians for both storage and rendering results in reduced memory usage and effectively accelerates rendering, especially for highly complex scenes. Furthermore, we demonstrate how level of detail can be integrated into our proposed method to further enhance rendering speed for large-scale scenes. With an optimized implementation, we highlight key practical considerations and thoroughly evaluate the proposed technique and its impact on desktop and mobile devices.
>
---
#### [new 114] MemeMind: A Large-Scale Multimodal Dataset with Chain-of-Thought Reasoning for Harmful Meme Detection
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于有害表情包检测任务，旨在解决现有数据集不足和检测模型解释性差的问题。研究构建了MemeMind数据集并提出MemeGuard框架，提升检测效果与可解释性。**

- **链接: [http://arxiv.org/pdf/2506.18919v1](http://arxiv.org/pdf/2506.18919v1)**

> **作者:** Hexiang Gu; Qifan Yu; Saihui Hou; Zhiqin Fang; Huijia Wu; Zhaofeng He
>
> **摘要:** The rapid development of social media has intensified the spread of harmful content. Harmful memes, which integrate both images and text, pose significant challenges for automated detection due to their implicit semantics and complex multimodal interactions. Although existing research has made progress in detection accuracy and interpretability, the lack of a systematic, large-scale, diverse, and highly explainable dataset continues to hinder further advancement in this field. To address this gap, we introduce MemeMind, a novel dataset featuring scientifically rigorous standards, large scale, diversity, bilingual support (Chinese and English), and detailed Chain-of-Thought (CoT) annotations. MemeMind fills critical gaps in current datasets by offering comprehensive labeling and explicit reasoning traces, thereby providing a solid foundation for enhancing harmful meme detection. In addition, we propose an innovative detection framework, MemeGuard, which effectively integrates multimodal information with reasoning process modeling, significantly improving models' ability to understand and identify harmful memes. Extensive experiments conducted on the MemeMind dataset demonstrate that MemeGuard consistently outperforms existing state-of-the-art methods in harmful meme detection tasks.
>
---
#### [new 115] NIC-RobustBench: A Comprehensive Open-Source Toolkit for Neural Image Compression and Robustness Analysis
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 该论文属于神经图像压缩与鲁棒性分析任务，旨在解决NIC模型的鲁棒性评估问题，提出NIC-RobustBench框架进行高效评估与比较。**

- **链接: [http://arxiv.org/pdf/2506.19051v1](http://arxiv.org/pdf/2506.19051v1)**

> **作者:** Georgii Bychkov; Khaled Abud; Egor Kovalev; Alexander Gushchin; Dmitriy Vatolin; Anastasia Antsiferova
>
> **备注:** arXiv admin note: text overlap with arXiv:2411.11795
>
> **摘要:** Adversarial robustness of neural networks is an increasingly important area of research, combining studies on computer vision models, large language models (LLMs), and others. With the release of JPEG AI -- the first standard for end-to-end neural image compression (NIC) methods -- the question of evaluating NIC robustness has become critically significant. However, previous research has been limited to a narrow range of codecs and attacks. To address this, we present \textbf{NIC-RobustBench}, the first open-source framework to evaluate NIC robustness and adversarial defenses' efficiency, in addition to comparing Rate-Distortion (RD) performance. The framework includes the largest number of codecs among all known NIC libraries and is easily scalable. The paper demonstrates a comprehensive overview of the NIC-RobustBench framework and employs it to analyze NIC robustness. Our code is available online at https://github.com/msu-video-group/NIC-RobustBench.
>
---
#### [new 116] SOF: Sorted Opacity Fields for Fast Unbounded Surface Reconstruction
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决从3D高斯中快速精确提取表面的问题。提出SOF方法，通过层次化排序和优化深度，提升重建精度与速度。**

- **链接: [http://arxiv.org/pdf/2506.19139v1](http://arxiv.org/pdf/2506.19139v1)**

> **作者:** Lukas Radl; Felix Windisch; Thomas Deixelberger; Jozef Hladky; Michael Steiner; Dieter Schmalstieg; Markus Steinberger
>
> **摘要:** Recent advances in 3D Gaussian representations have significantly improved the quality and efficiency of image-based scene reconstruction. Their explicit nature facilitates real-time rendering and fast optimization, yet extracting accurate surfaces - particularly in large-scale, unbounded environments - remains a difficult task. Many existing methods rely on approximate depth estimates and global sorting heuristics, which can introduce artifacts and limit the fidelity of the reconstructed mesh. In this paper, we present Sorted Opacity Fields (SOF), a method designed to recover detailed surfaces from 3D Gaussians with both speed and precision. Our approach improves upon prior work by introducing hierarchical resorting and a robust formulation of Gaussian depth, which better aligns with the level-set. To enhance mesh quality, we incorporate a level-set regularizer operating on the opacity field and introduce losses that encourage geometrically-consistent primitive shapes. In addition, we develop a parallelized Marching Tetrahedra algorithm tailored to our opacity formulation, reducing meshing time by up to an order of magnitude. As demonstrated by our quantitative evaluation, SOF achieves higher reconstruction accuracy while cutting total processing time by more than a factor of three. These results mark a step forward in turning efficient Gaussian-based rendering into equally efficient geometry extraction.
>
---
#### [new 117] NeRF-based CBCT Reconstruction needs Normalization and Initialization
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于CBCT图像重建任务，解决NeRF方法中局部-全局训练不匹配问题，提出归一化哈希编码器和映射一致性初始化策略以提升重建质量与训练稳定性。**

- **链接: [http://arxiv.org/pdf/2506.19742v1](http://arxiv.org/pdf/2506.19742v1)**

> **作者:** Zhuowei Xu; Han Li; Dai Sun; Zhicheng Li; Yujia Li; Qingpeng Kong; Zhiwei Cheng; Nassir Navab; S. Kevin Zhou
>
> **摘要:** Cone Beam Computed Tomography (CBCT) is widely used in medical imaging. However, the limited number and intensity of X-ray projections make reconstruction an ill-posed problem with severe artifacts. NeRF-based methods have achieved great success in this task. However, they suffer from a local-global training mismatch between their two key components: the hash encoder and the neural network. Specifically, in each training step, only a subset of the hash encoder's parameters is used (local sparse), whereas all parameters in the neural network participate (global dense). Consequently, hash features generated in each step are highly misaligned, as they come from different subsets of the hash encoder. These misalignments from different training steps are then fed into the neural network, causing repeated inconsistent global updates in training, which leads to unstable training, slower convergence, and degraded reconstruction quality. Aiming to alleviate the impact of this local-global optimization mismatch, we introduce a Normalized Hash Encoder, which enhances feature consistency and mitigates the mismatch. Additionally, we propose a Mapping Consistency Initialization(MCI) strategy that initializes the neural network before training by leveraging the global mapping property from a well-trained model. The initialized neural network exhibits improved stability during early training, enabling faster convergence and enhanced reconstruction performance. Our method is simple yet effective, requiring only a few lines of code while substantially improving training efficiency on 128 CT cases collected from 4 different datasets, covering 7 distinct anatomical regions.
>
---
## 更新

#### [replaced 001] Referring Expression Instance Retrieval and A Strong End-to-End Baseline
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18246v2](http://arxiv.org/pdf/2506.18246v2)**

> **作者:** Xiangzhao Hao; Kuan Zhu; Hongyu Guo; Haiyun Guo; Ning Jiang; Quan Lu; Ming Tang; JinQiao Wang
>
> **摘要:** Natural language querying of visual content underpins many vision-language tasks, typically categorized by text granularity and visual search scope. Text-Image Retrieval (TIR) retrieves whole images using coarse descriptions, while Referring Expression Comprehension (REC) localizes objects using fine-grained expressions within a single image. However, real-world scenarios often require both instance-level retrieval and localization across large galleries -- tasks where TIR lacks precision and REC lacks scalability. To address this gap, we propose a new task: Referring Expression Instance Retrieval (REIR), which jointly supports instance-level retrieval and localization. We introduce REIRCOCO, a large-scale benchmark constructed by prompting vision-language models to generate fine-grained expressions for MSCOCO and RefCOCO instances. We also present a baseline method, CLARE, featuring a dual-stream architecture with a Mix of Relation Experts (MORE) module for capturing inter-instance relationships. CLARE integrates object detection and REC pretraining with Contrastive Language-Instance Alignment (CLIA) for end-to-end optimization. Experiments show that CLARE achieves state-of-the-art performance on REIR and generalizes well to TIR and REC, highlighting its effectiveness and versatility.
>
---
#### [replaced 002] PicoSAM2: Low-Latency Segmentation In-Sensor for Edge Vision Applications
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18807v2](http://arxiv.org/pdf/2506.18807v2)**

> **作者:** Pietro Bonazzi; Nicola Farronato; Stefan Zihlmann; Haotong Qin; Michele Magno
>
> **摘要:** Real-time, on-device segmentation is critical for latency-sensitive and privacy-aware applications like smart glasses and IoT devices. We introduce PicoSAM2, a lightweight (1.3M parameters, 336M MACs) promptable segmentation model optimized for edge and in-sensor execution, including the Sony IMX500. It builds on a depthwise separable U-Net, with knowledge distillation and fixed-point prompt encoding to learn from the Segment Anything Model 2 (SAM2). On COCO and LVIS, it achieves 51.9% and 44.9% mIoU, respectively. The quantized model (1.22MB) runs at 14.3 ms on the IMX500-achieving 86 MACs/cycle, making it the only model meeting both memory and compute constraints for in-sensor deployment. Distillation boosts LVIS performance by +3.5% mIoU and +5.1% mAP. These results demonstrate that efficient, promptable segmentation is feasible directly on-camera, enabling privacy-preserving vision without cloud or host processing.
>
---
#### [replaced 003] LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.10082v3](http://arxiv.org/pdf/2506.10082v3)**

> **作者:** Chenjian Gao; Lihe Ding; Xin Cai; Zhanpeng Huang; Zibin Wang; Tianfan Xue
>
> **备注:** 12 pages
>
> **摘要:** Video editing using diffusion models has achieved remarkable results in generating high-quality edits for videos. However, current methods often rely on large-scale pretraining, limiting flexibility for specific edits. First-frame-guided editing provides control over the first frame, but lacks flexibility over subsequent frames. To address this, we propose a mask-based LoRA (Low-Rank Adaptation) tuning method that adapts pretrained Image-to-Video (I2V) models for flexible video editing. Our approach preserves background regions while enabling controllable edits propagation. This solution offers efficient and adaptable video editing without altering the model architecture. To better steer this process, we incorporate additional references, such as alternate viewpoints or representative scene states, which serve as visual anchors for how content should unfold. We address the control challenge using a mask-driven LoRA tuning strategy that adapts a pre-trained image-to-video model to the editing context. The model must learn from two distinct sources: the input video provides spatial structure and motion cues, while reference images offer appearance guidance. A spatial mask enables region-specific learning by dynamically modulating what the model attends to, ensuring that each area draws from the appropriate source. Experimental results show our method achieves superior video editing performance compared to state-of-the-art methods. Project Page: https://cjeen.github.io/LoraEditPaper
>
---
#### [replaced 004] MagicPose4D: Crafting Articulated Models with Appearance and Motion Control
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.14017v3](http://arxiv.org/pdf/2405.14017v3)**

> **作者:** Hao Zhang; Di Chang; Fang Li; Mohammad Soleymani; Narendra Ahuja
>
> **备注:** Project Page: https://magicpose4d.github.io/
>
> **摘要:** With the success of 2D and 3D visual generative models, there is growing interest in generating 4D content. Existing methods primarily rely on text prompts to produce 4D content, but they often fall short of accurately defining complex or rare motions. To address this limitation, we propose MagicPose4D, a novel framework for refined control over both appearance and motion in 4D generation. Unlike current 4D generation methods, MagicPose4D accepts monocular videos or mesh sequences as motion prompts, enabling precise and customizable motion control. MagicPose4D comprises two key modules: (i) Dual-Phase 4D Reconstruction Module, which operates in two phases. The first phase focuses on capturing the model's shape using accurate 2D supervision and less accurate but geometrically informative 3D pseudo-supervision without imposing skeleton constraints. The second phase extracts the 3D motion (skeleton poses) using more accurate pseudo-3D supervision, obtained in the first phase and introduces kinematic chain-based skeleton constraints to ensure physical plausibility. Additionally, we propose a Global-local Chamfer loss that aligns the overall distribution of predicted mesh vertices with the supervision while maintaining part-level alignment without extra annotations. (ii) Cross-category Motion Transfer Module, which leverages the extracted motion from the 4D reconstruction module and uses a kinematic-chain-based skeleton to achieve cross-category motion transfer. It ensures smooth transitions between frames through dynamic rigidity, facilitating robust generalization without additional training. Through extensive experiments, we demonstrate that MagicPose4D significantly improves the accuracy and consistency of 4D content generation, outperforming existing methods in various benchmarks.
>
---
#### [replaced 005] Object-aware Sound Source Localization via Audio-Visual Scene Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18557v2](http://arxiv.org/pdf/2506.18557v2)**

> **作者:** Sung Jin Um; Dongjin Kim; Sangmin Lee; Jung Uk Kim
>
> **备注:** Accepted at CVPR 2025
>
> **摘要:** Audio-visual sound source localization task aims to spatially localize sound-making objects within visual scenes by integrating visual and audio cues. However, existing methods struggle with accurately localizing sound-making objects in complex scenes, particularly when visually similar silent objects coexist. This limitation arises primarily from their reliance on simple audio-visual correspondence, which does not capture fine-grained semantic differences between sound-making and silent objects. To address these challenges, we propose a novel sound source localization framework leveraging Multimodal Large Language Models (MLLMs) to generate detailed contextual information that explicitly distinguishes between sound-making foreground objects and silent background objects. To effectively integrate this detailed information, we introduce two novel loss functions: Object-aware Contrastive Alignment (OCA) loss and Object Region Isolation (ORI) loss. Extensive experimental results on MUSIC and VGGSound datasets demonstrate the effectiveness of our approach, significantly outperforming existing methods in both single-source and multi-source localization scenarios. Code and generated detailed contextual information are available at: https://github.com/VisualAIKHU/OA-SSL.
>
---
#### [replaced 006] Improving Out-of-Distribution Detection via Dynamic Covariance Calibration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09399v3](http://arxiv.org/pdf/2506.09399v3)**

> **作者:** Kaiyu Guo; Zijian Wang; Tan Pan; Brian C. Lovell; Mahsa Baktashmotlagh
>
> **备注:** Accepted by ICML25
>
> **摘要:** Out-of-Distribution (OOD) detection is essential for the trustworthiness of AI systems. Methods using prior information (i.e., subspace-based methods) have shown effective performance by extracting information geometry to detect OOD data with a more appropriate distance metric. However, these methods fail to address the geometry distorted by ill-distributed samples, due to the limitation of statically extracting information geometry from the training distribution. In this paper, we argue that the influence of ill-distributed samples can be corrected by dynamically adjusting the prior geometry in response to new data. Based on this insight, we propose a novel approach that dynamically updates the prior covariance matrix using real-time input features, refining its information. Specifically, we reduce the covariance along the direction of real-time input features and constrain adjustments to the residual space, thus preserving essential data characteristics and avoiding effects on unintended directions in the principal space. We evaluate our method on two pre-trained models for the CIFAR dataset and five pre-trained models for ImageNet-1k, including the self-supervised DINO model. Extensive experiments demonstrate that our approach significantly enhances OOD detection across various models. The code is released at https://github.com/workerbcd/ooddcc.
>
---
#### [replaced 007] Dynamic PET Image Reconstruction via Non-negative INR Factorization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08025v2](http://arxiv.org/pdf/2503.08025v2)**

> **作者:** Chaozhi Zhang; Wenxiang Ding; Roy Y. He; Xiaoqun Zhang; Qiaoqiao Ding
>
> **摘要:** The reconstruction of dynamic positron emission tomography (PET) images from noisy projection data is a significant but challenging problem. In this paper, we introduce an unsupervised learning approach, Non-negative Implicit Neural Representation Factorization (\texttt{NINRF}), based on low rank matrix factorization of unknown images and employing neural networks to represent both coefficients and bases. Mathematically, we demonstrate that if a sequence of dynamic PET images satisfies a generalized non-negative low-rank property, it can be decomposed into a set of non-negative continuous functions varying in the temporal-spatial domain. This bridges the well-established non-negative matrix factorization (NMF) with continuous functions and we propose using implicit neural representations (INRs) to connect matrix with continuous functions. The neural network parameters are obtained by minimizing the KL divergence, with additional sparsity regularization on coefficients and bases. Extensive experiments on dynamic PET reconstruction with Poisson noise demonstrate the effectiveness of the proposed method compared to other methods, while giving continuous representations for object's detailed geometric features and regional concentration variation.
>
---
#### [replaced 008] Unfolding the Past: A Comprehensive Deep Learning Approach to Analyzing Incunabula Pages
- **分类: cs.DL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18069v2](http://arxiv.org/pdf/2506.18069v2)**

> **作者:** Klaudia Ropel; Krzysztof Kutt; Luiz do Valle Miranda; Grzegorz J. Nalepa
>
> **备注:** 10 pages, 8 figures; submitted to TPDL 2025; change in v2: updated e-mail address
>
> **摘要:** We developed a proof-of-concept method for the automatic analysis of the structure and content of incunabula pages. A custom dataset comprising 500 annotated pages from five different incunabula was created using resources from the Jagiellonian Digital Library. Each page was manually labeled with five predefined classes: Text, Title, Picture, Table, and Handwriting. Additionally, the publicly available DocLayNet dataset was utilized as supplementary training data. To perform object detection, YOLO11n and YOLO11s models were employed and trained using two strategies: a combined dataset (DocLayNet and the custom dataset) and the custom dataset alone. The highest performance (F1 = 0.94) was achieved by the YOLO11n model trained exclusively on the custom data. Optical character recognition was then conducted on regions classified as Text, using both Tesseract and Kraken OCR, with Tesseract demonstrating superior results. Subsequently, image classification was applied to the Picture class using a ResNet18 model, achieving an accuracy of 98.7% across five subclasses: Decorative_letter, Illustration, Other, Stamp, and Wrong_detection. Furthermore, the CLIP model was utilized to generate semantic descriptions of illustrations. The results confirm the potential of machine learning in the analysis of early printed books, while emphasizing the need for further advancements in OCR performance and visual content interpretation.
>
---
#### [replaced 009] crossMoDA Challenge: Evolution of Cross-Modality Domain Adaptation Techniques for Vestibular Schwannoma and Cochlea Segmentation from 2021 to 2023
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12006v2](http://arxiv.org/pdf/2506.12006v2)**

> **作者:** Navodini Wijethilake; Reuben Dorent; Marina Ivory; Aaron Kujawa; Stefan Cornelissen; Patrick Langenhuizen; Mohamed Okasha; Anna Oviedova; Hexin Dong; Bogyeong Kang; Guillaume Sallé; Luyi Han; Ziyuan Zhao; Han Liu; Tao Yang; Shahad Hardan; Hussain Alasmawi; Santosh Sanjeev; Yuzhou Zhuang; Satoshi Kondo; Maria Baldeon Calisto; Shaikh Muhammad Uzair Noman; Cancan Chen; Ipek Oguz; Rongguo Zhang; Mina Rezaei; Susana K. Lai-Yuen; Satoshi Kasai; Chih-Cheng Hung; Mohammad Yaqub; Lisheng Wang; Benoit M. Dawant; Cuntai Guan; Ritse Mann; Vincent Jaouen; Ji-Wung Han; Li Zhang; Jonathan Shapey; Tom Vercauteren
>
> **摘要:** The cross-Modality Domain Adaptation (crossMoDA) challenge series, initiated in 2021 in conjunction with the International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), focuses on unsupervised cross-modality segmentation, learning from contrast-enhanced T1 (ceT1) and transferring to T2 MRI. The task is an extreme example of domain shift chosen to serve as a meaningful and illustrative benchmark. From a clinical application perspective, it aims to automate Vestibular Schwannoma (VS) and cochlea segmentation on T2 scans for more cost-effective VS management. Over time, the challenge objectives have evolved to enhance its clinical relevance. The challenge evolved from using single-institutional data and basic segmentation in 2021 to incorporating multi-institutional data and Koos grading in 2022, and by 2023, it included heterogeneous routine data and sub-segmentation of intra- and extra-meatal tumour components. In this work, we report the findings of the 2022 and 2023 editions and perform a retrospective analysis of the challenge progression over the years. The observations from the successive challenge contributions indicate that the number of outliers decreases with an expanding dataset. This is notable since the diversity of scanning protocols of the datasets concurrently increased. The winning approach of the 2023 edition reduced the number of outliers on the 2021 and 2022 testing data, demonstrating how increased data heterogeneity can enhance segmentation performance even on homogeneous data. However, the cochlea Dice score declined in 2023, likely due to the added complexity from tumour sub-annotations affecting overall segmentation performance. While progress is still needed for clinically acceptable VS segmentation, the plateauing performance suggests that a more challenging cross-modal task may better serve future benchmarking.
>
---
#### [replaced 010] Not All Thats Rare Is Lost: Causal Paths to Rare Concept Synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20808v2](http://arxiv.org/pdf/2505.20808v2)**

> **作者:** Bo-Kai Ruan; Zi-Xiang Ni; Bo-Lun Huang; Teng-Fang Hsiao; Hong-Han Shuai
>
> **摘要:** Diffusion models have shown strong capabilities in high-fidelity image generation but often falter when synthesizing rare concepts, i.e., prompts that are infrequently observed in the training distribution. In this paper, we introduce RAP, a principled framework that treats rare concept generation as navigating a latent causal path: a progressive, model-aligned trajectory through the generative space from frequent concepts to rare targets. Rather than relying on heuristic prompt alternation, we theoretically justify that rare prompt guidance can be approximated by semantically related frequent prompts. We then formulate prompt switching as a dynamic process based on score similarity, enabling adaptive stage transitions. Furthermore, we reinterpret prompt alternation as a second-order denoising mechanism, promoting smooth semantic progression and coherent visual synthesis. Through this causal lens, we align input scheduling with the model's internal generative dynamics. Experiments across diverse diffusion backbones demonstrate that RAP consistently enhances rare concept generation, outperforming strong baselines in both automated evaluations and human studies.
>
---
#### [replaced 011] Dataset of soil images with corresponding particle size distributions for photogranulometry
- **分类: cs.CV; I.5.4; I.2.10**

- **链接: [http://arxiv.org/pdf/2506.17469v2](http://arxiv.org/pdf/2506.17469v2)**

> **作者:** Thomas Plante St-Cyr; François Duhaime; Jean-Sébastien Dubé; Simon Grenier
>
> **备注:** 8 pages, 10 figures, conference
>
> **摘要:** Traditional particle size distribution (PSD) analyses create significant downtime and are expensive in labor and maintenance. These drawbacks could be alleviated using optical grain size analysis integrated into routine geotechnical laboratory workflow. This paper presents a high-resolution dataset of 12,714 images of 321 different soil samples collected in the Montreal, Quebec region, alongside their PSD analysis. It is designed to provide a robust starting point for training convolutional neural networks (CNN) in geotechnical applications. Soil samples were photographed in a standardized top-view position with a resolution of 45 MP and a minimum scale of 39.4 micrometers per pixel, both in their moist and dry states. A custom test bench employing 13x9 inch white aluminum trays, on which the samples are spread in a thin layer, was used. For samples exceeding a size limit, a coning and quartering method was employed for mass reduction.
>
---
#### [replaced 012] FusionSAM: Visual Multi-Modal Learning with Segment Anything
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.13980v2](http://arxiv.org/pdf/2408.13980v2)**

> **作者:** Daixun Li; Weiying Xie; Mingxiang Cao; Yunke Wang; Yusi Zhang; Leyuan Fang; Yunsong Li; Chang Xu
>
> **摘要:** Multimodal image fusion and semantic segmentation are critical for autonomous driving. Despite advancements, current models often struggle with segmenting densely packed elements due to a lack of comprehensive fusion features for guidance during training. While the Segment Anything Model (SAM) allows precise control during fine-tuning through its flexible prompting encoder, its potential remains largely unexplored in the context of multimodal segmentation for natural images. In this paper, we introduce SAM into multimodal image segmentation for the first time, proposing a novel framework that combines Latent Space Token Generation (LSTG) and Fusion Mask Prompting (FMP) modules. This approach transforms the training methodology for multimodal segmentation from a traditional black-box approach to a controllable, prompt-based mechanism. Specifically, we obtain latent space features for both modalities through vector quantization and embed them into a cross-attention-based inter-domain fusion module to establish long-range dependencies between modalities. We then use these comprehensive fusion features as prompts to guide precise pixel-level segmentation. Extensive experiments on multiple public datasets demonstrate that our method significantly outperforms SAM and SAM2 in multimodal autonomous driving scenarios, achieving an average improvement of 4.1$\%$ over the state-of-the-art method in segmentation mIoU, and the performance is also optimized in other multi-modal visual scenes.
>
---
#### [replaced 013] Flopping for FLOPs: Leveraging equivariance for computational efficiency
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.05169v2](http://arxiv.org/pdf/2502.05169v2)**

> **作者:** Georg Bökman; David Nordström; Fredrik Kahl
>
> **备注:** ICML 2025
>
> **摘要:** Incorporating geometric invariance into neural networks enhances parameter efficiency but typically increases computational costs. This paper introduces new equivariant neural networks that preserve symmetry while maintaining a comparable number of floating-point operations (FLOPs) per parameter to standard non-equivariant networks. We focus on horizontal mirroring (flopping) invariance, common in many computer vision tasks. The main idea is to parametrize the feature spaces in terms of mirror-symmetric and mirror-antisymmetric features, i.e., irreps of the flopping group. This decomposes the linear layers to be block-diagonal, requiring half the number of FLOPs. Our approach reduces both FLOPs and wall-clock time, providing a practical solution for efficient, scalable symmetry-aware architectures.
>
---
#### [replaced 014] RA-NeRF: Robust Neural Radiance Field Reconstruction with Accurate Camera Pose Estimation under Complex Trajectories
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.15242v2](http://arxiv.org/pdf/2506.15242v2)**

> **作者:** Qingsong Yan; Qiang Wang; Kaiyong Zhao; Jie Chen; Bo Li; Xiaowen Chu; Fei Deng
>
> **备注:** IROS 2025
>
> **摘要:** Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have emerged as powerful tools for 3D reconstruction and SLAM tasks. However, their performance depends heavily on accurate camera pose priors. Existing approaches attempt to address this issue by introducing external constraints but fall short of achieving satisfactory accuracy, particularly when camera trajectories are complex. In this paper, we propose a novel method, RA-NeRF, capable of predicting highly accurate camera poses even with complex camera trajectories. Following the incremental pipeline, RA-NeRF reconstructs the scene using NeRF with photometric consistency and incorporates flow-driven pose regulation to enhance robustness during initialization and localization. Additionally, RA-NeRF employs an implicit pose filter to capture the camera movement pattern and eliminate the noise for pose estimation. To validate our method, we conduct extensive experiments on the Tanks\&Temple dataset for standard evaluation, as well as the NeRFBuster dataset, which presents challenging camera pose trajectories. On both datasets, RA-NeRF achieves state-of-the-art results in both camera pose estimation and visual quality, demonstrating its effectiveness and robustness in scene reconstruction under complex pose trajectories.
>
---
#### [replaced 015] Brain Mapping with Dense Features: Grounding Cortical Semantic Selectivity in Natural Images With Vision Transformers
- **分类: cs.CV; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2410.05266v2](http://arxiv.org/pdf/2410.05266v2)**

> **作者:** Andrew F. Luo; Jacob Yeung; Rushikesh Zawar; Shaurya Dewan; Margaret M. Henderson; Leila Wehbe; Michael J. Tarr
>
> **备注:** Accepted at ICLR 2025, code: https://github.com/aluo-x/BrainSAIL
>
> **摘要:** We introduce BrainSAIL, a method for linking neural selectivity with spatially distributed semantic visual concepts in natural scenes. BrainSAIL leverages recent advances in large-scale artificial neural networks, using them to provide insights into the functional topology of the brain. To overcome the challenge presented by the co-occurrence of multiple categories in natural images, BrainSAIL exploits semantically consistent, dense spatial features from pre-trained vision models, building upon their demonstrated ability to robustly predict neural activity. This method derives clean, spatially dense embeddings without requiring any additional training, and employs a novel denoising process that leverages the semantic consistency of images under random augmentations. By unifying the space of whole-image embeddings and dense visual features and then applying voxel-wise encoding models to these features, we enable the identification of specific subregions of each image which drive selectivity patterns in different areas of the higher visual cortex. This provides a powerful tool for dissecting the neural mechanisms that underlie semantic visual processing for natural images. We validate BrainSAIL on cortical regions with known category selectivity, demonstrating its ability to accurately localize and disentangle selectivity to diverse visual concepts. Next, we demonstrate BrainSAIL's ability to characterize high-level visual selectivity to scene properties and low-level visual features such as depth, luminance, and saturation, providing insights into the encoding of complex visual information. Finally, we use BrainSAIL to directly compare the feature selectivity of different brain encoding models across different regions of interest in visual cortex. Our innovative method paves the way for significant advances in mapping and decomposing high-level visual representations in the human brain.
>
---
#### [replaced 016] Compositional Scene Understanding through Inverse Generative Modeling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21780v4](http://arxiv.org/pdf/2505.21780v4)**

> **作者:** Yanbo Wang; Justin Dauwels; Yilun Du
>
> **备注:** ICML 2025, Webpage: https://energy-based-model.github.io/compositional-inference
>
> **摘要:** Generative models have demonstrated remarkable abilities in generating high-fidelity visual content. In this work, we explore how generative models can further be used not only to synthesize visual content but also to understand the properties of a scene given a natural image. We formulate scene understanding as an inverse generative modeling problem, where we seek to find conditional parameters of a visual generative model to best fit a given natural image. To enable this procedure to infer scene structure from images substantially different than those seen during training, we further propose to build this visual generative model compositionally from smaller models over pieces of a scene. We illustrate how this procedure enables us to infer the set of objects in a scene, enabling robust generalization to new test scenes with an increased number of objects of new shapes. We further illustrate how this enables us to infer global scene factors, likewise enabling robust generalization to new scenes. Finally, we illustrate how this approach can be directly applied to existing pretrained text-to-image generative models for zero-shot multi-object perception. Code and visualizations are at https://energy-based-model.github.io/compositional-inference.
>
---
#### [replaced 017] FineCLIPER: Multi-modal Fine-grained CLIP for Dynamic Facial Expression Recognition with AdaptERs
- **分类: cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2407.02157v3](http://arxiv.org/pdf/2407.02157v3)**

> **作者:** Haodong Chen; Haojian Huang; Junhao Dong; Mingzhe Zheng; Dian Shao
>
> **备注:** Accepted to ACM MM 2024
>
> **摘要:** Dynamic Facial Expression Recognition (DFER) is crucial for understanding human behavior. However, current methods exhibit limited performance mainly due to the scarcity of high-quality data, the insufficient utilization of facial dynamics, and the ambiguity of expression semantics, etc. To this end, we propose a novel framework, named Multi-modal Fine-grained CLIP for Dynamic Facial Expression Recognition with AdaptERs (FineCLIPER), incorporating the following novel designs: 1) To better distinguish between similar facial expressions, we extend the class labels to textual descriptions from both positive and negative aspects, and obtain supervision by calculating the cross-modal similarity based on the CLIP model; 2) Our FineCLIPER adopts a hierarchical manner to effectively mine useful cues from DFE videos. Specifically, besides directly embedding video frames as input (low semantic level), we propose to extract the face segmentation masks and landmarks based on each frame (middle semantic level) and utilize the Multi-modal Large Language Model (MLLM) to further generate detailed descriptions of facial changes across frames with designed prompts (high semantic level). Additionally, we also adopt Parameter-Efficient Fine-Tuning (PEFT) to enable efficient adaptation of large pre-trained models (i.e., CLIP) for this task. Our FineCLIPER achieves SOTA performance on the DFEW, FERV39k, and MAFW datasets in both supervised and zero-shot settings with few tunable parameters. Project Page: https://haroldchen19.github.io/FineCLIPER-Page/
>
---
#### [replaced 018] MDeRainNet: An Efficient Macro-pixel Image Rain Removal Network
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.10652v3](http://arxiv.org/pdf/2406.10652v3)**

> **作者:** Tao Yan; Weijiang He; Chenglong Wang; Cihang Wei; Xiangjie Zhu; Yinghui Wang; Rynson W. H. Lau
>
> **备注:** 14 pages, 14 figures, 4 tables
>
> **摘要:** Since rainy weather always degrades image quality and poses significant challenges to most computer vision-based intelligent systems, image de-raining has been a hot research topic. Fortunately, in a rainy light field (LF) image, background obscured by rain streaks in one sub-view may be visible in the other sub-views, and implicit depth information and recorded 4D structural information may benefit rain streak detection and removal. However, existing LF image rain removal methods either do not fully exploit the global correlations of 4D LF data or only utilize partial sub-views, resulting in sub-optimal rain removal performance and no-equally good quality for all de-rained sub-views. In this paper, we propose an efficient network, called MDeRainNet, for rain streak removal from LF images. The proposed network adopts a multi-scale encoder-decoder architecture, which directly works on Macro-pixel images (MPIs) to improve the rain removal performance. To fully model the global correlation between the spatial and the angular information, we propose an Extended Spatial-Angular Interaction (ESAI) module to merge them, in which a simple and effective Transformer-based Spatial-Angular Interaction Attention (SAIA) block is also proposed for modeling long-range geometric correlations and making full use of the angular information. Furthermore, to improve the generalization performance of our network on real-world rainy scenes, we propose a novel semi-supervised learning framework for our MDeRainNet, which utilizes multi-level KL loss to bridge the domain gap between features of synthetic and real-world rain streaks and introduces colored-residue image guided contrastive regularization to reconstruct rain-free images. Extensive experiments conducted on synthetic and real-world LFIs demonstrate that our method outperforms the state-of-the-art methods both quantitatively and qualitatively.
>
---
#### [replaced 019] VideoMathQA: Benchmarking Mathematical Reasoning via Multimodal Understanding in Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05349v2](http://arxiv.org/pdf/2506.05349v2)**

> **作者:** Hanoona Rasheed; Abdelrahman Shaker; Anqi Tang; Muhammad Maaz; Ming-Hsuan Yang; Salman Khan; Fahad Shahbaz Khan
>
> **备注:** VideoMathQA Technical Report
>
> **摘要:** Mathematical reasoning in real-world video settings presents a fundamentally different challenge than in static images or text. It requires interpreting fine-grained visual information, accurately reading handwritten or digital text, and integrating spoken cues, often dispersed non-linearly over time. In such multimodal contexts, success hinges not just on perception, but on selectively identifying and integrating the right contextual details from a rich and noisy stream of content. To this end, we introduce VideoMathQA, a benchmark designed to evaluate whether models can perform such temporally extended cross-modal reasoning on videos. The benchmark spans 10 diverse mathematical domains, covering videos ranging from 10 seconds to over 1 hour. It requires models to interpret structured visual content, understand instructional narratives, and jointly ground concepts across visual, audio, and textual modalities. We employ graduate-level experts to ensure high quality, totaling over $920$ man-hours of annotation. To reflect real-world scenarios, questions are designed around three core reasoning challenges: direct problem solving, where answers are grounded in the presented question; conceptual transfer, which requires applying learned methods to new problems; and deep instructional comprehension, involving multi-step reasoning over extended explanations and partially worked-out solutions. Each question includes multi-step reasoning annotations, enabling fine-grained diagnosis of model capabilities. Through this benchmark, we highlight the limitations of existing approaches and establish a systematic evaluation framework for models that must reason, rather than merely perceive, across temporally extended and modality-rich mathematical problem settings. Our benchmark and evaluation code are available at: https://mbzuai-oryx.github.io/VideoMathQA
>
---
#### [replaced 020] FOCoOp: Enhancing Out-of-Distribution Robustness in Federated Prompt Learning for Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.16218v2](http://arxiv.org/pdf/2506.16218v2)**

> **作者:** Xinting Liao; Weiming Liu; Jiaming Qian; Pengyang Zhou; Jiahe Xu; Wenjie Wang; Chaochao Chen; Xiaolin Zheng; Tat-Seng Chua
>
> **备注:** Accepted by ICML25
>
> **摘要:** Federated prompt learning (FPL) for vision-language models is a powerful approach to collaboratively adapt models across distributed clients while preserving data privacy. However, existing FPL approaches suffer from a trade-off between performance and robustness, particularly in out-of-distribution (OOD) shifts, limiting their reliability in real-world scenarios. The inherent in-distribution (ID) data heterogeneity among different clients makes it more challenging to maintain this trade-off. To fill this gap, we introduce a Federated OOD-aware Context Optimization (FOCoOp) framework, which captures diverse distributions among clients using ID global prompts, local prompts, and OOD prompts. Specifically, FOCoOp leverages three sets of prompts to create both class-level and distribution-level separations, which adapt to OOD shifts through bi-level distributionally robust optimization. Additionally, FOCoOp improves the discrimination consistency among clients, i.e., calibrating global prompts, seemingly OOD prompts, and OOD prompts by semi-unbalanced optimal transport. The extensive experiments on real-world datasets demonstrate that FOCoOp effectively captures decentralized heterogeneous distributions and enhances robustness of different OOD shifts. The project is available at GitHub.
>
---
#### [replaced 021] Light of Normals: Unified Feature Representation for Universal Photometric Stereo
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18882v2](http://arxiv.org/pdf/2506.18882v2)**

> **作者:** Hong Li; Houyuan Chen; Chongjie Ye; Zhaoxi Chen; Bohan Li; Shaocong Xu; Xianda Guo; Xuhui Liu; Yikai Wang; Baochang Zhang; Satoshi Ikehata; Boxin Shi; Anyi Rao; Hao Zhao
>
> **备注:** Home: https://houyuanchen111.github.io/lino.github.io Github: https://github.com/houyuanchen111/LINO_UniPS HuggingFace Demo: https://huggingface.co/spaces/houyuanchen/lino
>
> **摘要:** Universal photometric stereo (PS) aims to recover high-quality surface normals from objects under arbitrary lighting conditions without relying on specific illumination models. Despite recent advances such as SDM-UniPS and Uni MS-PS, two fundamental challenges persist: 1) the deep coupling between varying illumination and surface normal features, where ambiguity in observed intensity makes it difficult to determine whether brightness variations stem from lighting changes or surface orientation; and 2) the preservation of high-frequency geometric details in complex surfaces, where intricate geometries create self-shadowing, inter-reflections, and subtle normal variations that conventional feature processing operations struggle to capture accurately.
>
---
#### [replaced 022] Classification in Japanese Sign Language Based on Dynamic Facial Expressions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.06347v2](http://arxiv.org/pdf/2411.06347v2)**

> **作者:** Yui Tatsumi; Shoko Tanaka; Shunsuke Akamatsu; Takahiro Shindo; Hiroshi Watanabe
>
> **备注:** Accepted by 2024 IEEE 13th Global Conference on Consumer Electronics (GCCE 2024)
>
> **摘要:** Sign language is a visual language expressed through hand movements and non-manual markers. Non-manual markers include facial expressions and head movements. These expressions vary across different nations. Therefore, specialized analysis methods for each sign language are necessary. However, research on Japanese Sign Language (JSL) recognition is limited due to a lack of datasets. The development of recognition models that consider both manual and non-manual features of JSL is crucial for precise and smooth communication with deaf individuals. In JSL, sentence types such as affirmative statements and questions are distinguished by facial expressions. In this paper, we propose a JSL recognition method that focuses on facial expressions. Our proposed method utilizes a neural network to analyze facial features and classify sentence types. Through the experiments, we confirm our method's effectiveness by achieving a classification accuracy of 96.05%.
>
---
#### [replaced 023] DeltaSpace: A Semantic-aligned Feature Space for Flexible Text-guided Image Editing
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2310.08785v2](http://arxiv.org/pdf/2310.08785v2)**

> **作者:** Yueming Lyu; Kang Zhao; Bo Peng; Huafeng Chen; Yue Jiang; Yingya Zhang; Jing Dong; Caifeng Shan
>
> **备注:** 18 pages. arXiv admin note: text overlap with arXiv:2303.06285
>
> **摘要:** Text-guided image editing faces significant challenges when considering training and inference flexibility. Much literature collects large amounts of annotated image-text pairs to train text-conditioned generative models from scratch, which is expensive and not efficient. After that, some approaches that leverage pre-trained vision-language models have been proposed to avoid data collection, but they are limited by either per text-prompt optimization or inference-time hyper-parameters tuning. To address these issues, we investigate and identify a specific space, referred to as CLIP DeltaSpace, where the CLIP visual feature difference of two images is semantically aligned with the CLIP textual feature difference of their corresponding text descriptions. Based on DeltaSpace, we propose a novel framework called DeltaEdit, which maps the CLIP visual feature differences to the latent space directions of a generative model during the training phase, and predicts the latent space directions from the CLIP textual feature differences during the inference phase. And this design endows DeltaEdit with two advantages: (1) text-free training; (2) generalization to various text prompts for zero-shot inference. Extensive experiments validate the effectiveness and versatility of DeltaEdit with different generative models, including both the GAN model and the diffusion model, in achieving flexible text-guided image editing. Code is available at https://github.com/Yueming6568/DeltaEdit.
>
---
#### [replaced 024] Impact of Visual Context on Noisy Multimodal NMT: An Empirical Study for English to Indian Languages
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2308.16075v2](http://arxiv.org/pdf/2308.16075v2)**

> **作者:** Baban Gain; Dibyanayan Bandyopadhyay; Samrat Mukherjee; Chandranath Adak; Asif Ekbal
>
> **摘要:** Neural Machine Translation (NMT) has made remarkable progress using large-scale textual data, but the potential of incorporating multimodal inputs, especially visual information, remains underexplored in high-resource settings. While prior research has focused on using multimodal data in low-resource scenarios, this study examines how image features impact translation when added to a large-scale, pre-trained unimodal NMT system. Surprisingly, the study finds that images might be redundant in this context. Additionally, the research introduces synthetic noise to assess whether images help the model handle textual noise. Multimodal models slightly outperform text-only models in noisy settings, even when random images are used. The study's experiments translate from English to Hindi, Bengali, and Malayalam, significantly outperforming state-of-the-art benchmarks. Interestingly, the effect of visual context varies with the level of source text noise: no visual context works best for non-noisy translations, cropped image features are optimal for low noise, and full image features perform better in high-noise scenarios. This sheds light on the role of visual context, especially in noisy settings, and opens up a new research direction for Noisy Neural Machine Translation in multimodal setups. The research emphasizes the importance of combining visual and textual information to improve translation across various environments. Our code is publicly available at https://github.com/babangain/indicMMT.
>
---
#### [replaced 025] Controllable Video Generation with Provable Disentanglement
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.02690v2](http://arxiv.org/pdf/2502.02690v2)**

> **作者:** Yifan Shen; Peiyuan Zhu; Zijian Li; Shaoan Xie; Zeyu Tang; Namrata Deka; Zongfang Liu; Guangyi Chen; Kun Zhang
>
> **摘要:** Controllable video generation remains a significant challenge, despite recent advances in generating high-quality and consistent videos. Most existing methods for controlling video generation treat the video as a whole, neglecting intricate fine-grained spatiotemporal relationships, which limits both control precision and efficiency. In this paper, we propose Controllable Video Generative Adversarial Networks (CoVoGAN) to disentangle the video concepts, thus facilitating efficient and independent control over individual concepts. Specifically, following the minimal change principle, we first disentangle static and dynamic latent variables. We then leverage the sufficient change property to achieve component-wise identifiability of dynamic latent variables, enabling disentangled control of video generation. To establish the theoretical foundation, we provide a rigorous analysis demonstrating the identifiability of our approach. Building on these theoretical insights, we design a Temporal Transition Module to disentangle latent dynamics. To enforce the minimal change principle and sufficient change property, we minimize the dimensionality of latent dynamic variables and impose temporal conditional independence. To validate our approach, we integrate this module as a plug-in for GANs. Extensive qualitative and quantitative experiments on various video generation benchmarks demonstrate that our method significantly improves generation quality and controllability across diverse real-world scenarios.
>
---
#### [replaced 026] GCE-Pose: Global Context Enhancement for Category-level Object Pose Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.04293v2](http://arxiv.org/pdf/2502.04293v2)**

> **作者:** Weihang Li; Hongli Xu; Junwen Huang; Hyunjun Jung; Peter KT Yu; Nassir Navab; Benjamin Busam
>
> **备注:** CVPR 2025 accepted
>
> **摘要:** A key challenge in model-free category-level pose estimation is the extraction of contextual object features that generalize across varying instances within a specific category. Recent approaches leverage foundational features to capture semantic and geometry cues from data. However, these approaches fail under partial visibility. We overcome this with a first-complete-then-aggregate strategy for feature extraction utilizing class priors. In this paper, we present GCE-Pose, a method that enhances pose estimation for novel instances by integrating category-level global context prior. GCE-Pose performs semantic shape reconstruction with a proposed Semantic Shape Reconstruction (SSR) module. Given an unseen partial RGB-D object instance, our SSR module reconstructs the instance's global geometry and semantics by deforming category-specific 3D semantic prototypes through a learned deep Linear Shape Model. We further introduce a Global Context Enhanced (GCE) feature fusion module that effectively fuses features from partial RGB-D observations and the reconstructed global context. Extensive experiments validate the impact of our global context prior and the effectiveness of the GCE fusion module, demonstrating that GCE-Pose significantly outperforms existing methods on challenging real-world datasets HouseCat6D and NOCS-REAL275. Our project page is available at https://colin-de.github.io/GCE-Pose/.
>
---
#### [replaced 027] Beyond Reconstruction: A Physics Based Neural Deferred Shader for Photo-realistic Rendering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.12273v2](http://arxiv.org/pdf/2504.12273v2)**

> **作者:** Zhuo He; Paul Henderson; Nicolas Pugeault
>
> **摘要:** Deep learning based rendering has achieved major improvements in photo-realistic image synthesis, with potential applications including visual effects in movies and photo-realistic scene building in video games. However, a significant limitation is the difficulty of decomposing the illumination and material parameters, which limits such methods to reconstructing an input scene, without any possibility to control these parameters. This paper introduces a novel physics based neural deferred shading pipeline to decompose the data-driven rendering process, learn a generalizable shading function to produce photo-realistic results for shading and relighting tasks; we also propose a shadow estimator to efficiently mimic shadowing effects. Our model achieves improved performance compared to classical models and a state-of-art neural shading model, and enables generalizable photo-realistic shading from arbitrary illumination input.
>
---
#### [replaced 028] Cross-Level Multi-Instance Distillation for Self-Supervised Fine-Grained Visual Categorization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2401.08860v3](http://arxiv.org/pdf/2401.08860v3)**

> **作者:** Qi Bi; Wei Ji; Jingjun Yi; Haolan Zhan; Gui-Song Xia
>
> **备注:** Accepted by IEEE Transactions on Image Processing (TIP)
>
> **摘要:** High-quality annotation of fine-grained visual categories demands great expert knowledge, which is taxing and time consuming. Alternatively, learning fine-grained visual representation from enormous unlabeled images (e.g., species, brands) by self-supervised learning becomes a feasible solution. However, recent researches find that existing self-supervised learning methods are less qualified to represent fine-grained categories. The bottleneck lies in that the pre-text representation is built from every patch-wise embedding, while fine-grained categories are only determined by several key patches of an image. In this paper, we propose a Cross-level Multi-instance Distillation (CMD) framework to tackle the challenge. Our key idea is to consider the importance of each image patch in determining the fine-grained pre-text representation by multiple instance learning. To comprehensively learn the relation between informative patches and fine-grained semantics, the multi-instance knowledge distillation is implemented on both the region/image crop pairs from the teacher and student net, and the region-image crops inside the teacher / student net, which we term as intra-level multi-instance distillation and inter-level multi-instance distillation. Extensive experiments on CUB-200-2011, Stanford Cars and FGVC Aircraft show that the proposed method outperforms the contemporary method by upto 10.14% and existing state-of-the-art self-supervised learning approaches by upto 19.78% on both top-1 accuracy and Rank-1 retrieval metric.
>
---
#### [replaced 029] Multimodal Fusion SLAM with Fourier Attention
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.18204v2](http://arxiv.org/pdf/2506.18204v2)**

> **作者:** Youjie Zhou; Guofeng Mei; Yiming Wang; Yi Wan; Fabio Poiesi
>
> **备注:** Accepted in IEEE RAL
>
> **摘要:** Visual SLAM is particularly challenging in environments affected by noise, varying lighting conditions, and darkness. Learning-based optical flow algorithms can leverage multiple modalities to address these challenges, but traditional optical flow-based visual SLAM approaches often require significant computational resources.To overcome this limitation, we propose FMF-SLAM, an efficient multimodal fusion SLAM method that utilizes fast Fourier transform (FFT) to enhance the algorithm efficiency. Specifically, we introduce a novel Fourier-based self-attention and cross-attention mechanism to extract features from RGB and depth signals. We further enhance the interaction of multimodal features by incorporating multi-scale knowledge distillation across modalities. We also demonstrate the practical feasibility of FMF-SLAM in real-world scenarios with real time performance by integrating it with a security robot by fusing with a global positioning module GNSS-RTK and global Bundle Adjustment. Our approach is validated using video sequences from TUM, TartanAir, and our real-world datasets, showcasing state-of-the-art performance under noisy, varying lighting, and dark conditions.Our code and datasets are available at https://github.com/youjie-zhou/FMF-SLAM.git.
>
---
#### [replaced 030] Align and Distill: Unifying and Improving Domain Adaptive Object Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.12029v4](http://arxiv.org/pdf/2403.12029v4)**

> **作者:** Justin Kay; Timm Haucke; Suzanne Stathatos; Siqi Deng; Erik Young; Pietro Perona; Sara Beery; Grant Van Horn
>
> **备注:** TMLR camera ready (Featured Certification). 33 pages, 15 figures
>
> **摘要:** Object detectors often perform poorly on data that differs from their training set. Domain adaptive object detection (DAOD) methods have recently demonstrated strong results on addressing this challenge. Unfortunately, we identify systemic benchmarking pitfalls that call past results into question and hamper further progress: (a) Overestimation of performance due to underpowered baselines, (b) Inconsistent implementation practices preventing transparent comparisons of methods, and (c) Lack of generality due to outdated backbones and lack of diversity in benchmarks. We address these problems by introducing: (1) A unified benchmarking and implementation framework, Align and Distill (ALDI), enabling comparison of DAOD methods and supporting future development, (2) A fair and modern training and evaluation protocol for DAOD that addresses benchmarking pitfalls, (3) A new DAOD benchmark dataset, CFC-DAOD, enabling evaluation on diverse real-world data, and (4) A new method, ALDI++, that achieves state-of-the-art results by a large margin. ALDI++ outperforms the previous state-of-the-art by +3.5 AP50 on Cityscapes to Foggy Cityscapes, +5.7 AP50 on Sim10k to Cityscapes (where ours is the only method to outperform a fair baseline), and +0.6 AP50 on CFC Kenai to Channel. ALDI and ALDI++ are architecture-agnostic, setting a new state-of-the-art for YOLO and DETR-based DAOD as well without additional hyperparameter tuning. Our framework, dataset, and state-of-the-art method offer a critical reset for DAOD and provide a strong foundation for future research. Code and data are available: https://github.com/justinkay/aldi and https://github.com/visipedia/caltech-fish-counting.
>
---
#### [replaced 031] ClimateIQA: A New Dataset and Benchmark to Advance Vision-Language Models in Meteorology Anomalies Analysis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.09838v2](http://arxiv.org/pdf/2406.09838v2)**

> **作者:** Jian Chen; Peilin Zhou; Yining Hua; Dading Chong; Meng Cao; Yaowei Li; Zixuan Yuan; Bing Zhu; Junwei Liang
>
> **摘要:** Meteorological heatmaps play a vital role in deciphering extreme weather phenomena, yet their inherent complexities marked by irregular contours, unstructured patterns, and complex color variations present unique analytical hurdles for state-of-the-art Vision-Language Models (VLMs). Current state-of-the-art models like GPT-4o, Qwen-VL, and LLaVA 1.6 struggle with tasks such as precise color identification and spatial localization, resulting in inaccurate or incomplete interpretations. To address these challenges, we introduce Sparse Position and Outline Tracking (SPOT), a novel algorithm specifically designed to process irregularly shaped colored regions in visual data. SPOT identifies and localizes these regions by extracting their spatial coordinates, enabling structured representations of irregular shapes. Building on SPOT, we construct ClimateIQA, a novel meteorological visual question answering (VQA) dataset, comprising 26,280 high-resolution heatmaps and 762,120 instruction samples for wind gust, total precipitation, wind chill index and heat index analysis. ClimateIQA enhances VLM training by incorporating spatial cues, geographic metadata, and reanalysis data, improving model accuracy in interpreting and describing extreme weather features. Furthermore, we develop Climate-Zoo, a suite of fine-tuned VLMs based on SPOT-empowered ClimateIQA, which significantly outperforms existing models in meteorological heatmap tasks.
>
---
#### [replaced 032] ASR-enhanced Multimodal Representation Learning for Cross-Domain Product Retrieval
- **分类: cs.MM; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2408.02978v2](http://arxiv.org/pdf/2408.02978v2)**

> **作者:** Ruixiang Zhao; Jian Jia; Yan Li; Xuehan Bai; Quan Chen; Han Li; Peng Jiang; Xirong Li
>
> **备注:** accepted for publication as a REGULAR paper in the IEEE Transactions on Multimedia
>
> **摘要:** E-commerce is increasingly multimedia-enriched, with products exhibited in a broad-domain manner as images, short videos, or live stream promotions. A unified and vectorized cross-domain production representation is essential. Due to large intra-product variance and high inter-product similarity in the broad-domain scenario, a visual-only representation is inadequate. While Automatic Speech Recognition (ASR) text derived from the short or live-stream videos is readily accessible, how to de-noise the excessively noisy text for multimodal representation learning is mostly untouched. We propose ASR-enhanced Multimodal Product Representation Learning (AMPere). In order to extract product-specific information from the raw ASR text, AMPere uses an easy-to-implement LLM-based ASR text summarizer. The LLM-summarized text, together with visual data, is then fed into a multi-branch network to generate compact multimodal embeddings. Extensive experiments on a large-scale tri-domain dataset verify the effectiveness of AMPere in obtaining a unified multimodal product representation that clearly improves cross-domain product retrieval.
>
---
#### [replaced 033] Two-Stream Spatial-Temporal Transformer Framework for Person Identification via Natural Conversational Keypoints
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20803v2](http://arxiv.org/pdf/2502.20803v2)**

> **作者:** Masoumeh Chapariniya; Hossein Ranjbar; Teodora Vukovic; Sarah Ebling; Volker Dellwo
>
> **备注:** I would like to withdraw this submission due to the need for substantial revisions in the results and analysis. I plan to correct and improve the study and submit a more complete version in the near future
>
> **摘要:** In the age of AI-driven generative technologies, traditional biometric recognition systems face unprecedented challenges, particularly from sophisticated deepfake and face reenactment techniques. In this study, we propose a Two-Stream Spatial-Temporal Transformer Framework for person identification using upper body keypoints visible during online conversations, which we term conversational keypoints. Our framework processes both spatial relationships between keypoints and their temporal evolution through two specialized branches: a Spatial Transformer (STR) that learns distinctive structural patterns in keypoint configurations, and a Temporal Transformer (TTR) that captures sequential motion patterns. Using the state-of-the-art Sapiens pose estimator, we extract 133 keypoints (based on COCO-WholeBody format) representing facial features, head pose, and hand positions. The framework was evaluated on a dataset of 114 individuals engaged in natural conversations, achieving recognition accuracies of 80.12% for the spatial stream, 63.61% for the temporal stream. We then explored two fusion strategies: a shared loss function approach achieving 82.22% accuracy, and a feature-level fusion method that concatenates feature maps from both streams, significantly improving performance to 94.86%. By jointly modeling both static anatomical relationships and dynamic movement patterns, our approach learns comprehensive identity signatures that are more robust to spoofing than traditional appearance-based methods.
>
---
#### [replaced 034] Privacy Attacks on Image AutoRegressive Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.02514v4](http://arxiv.org/pdf/2502.02514v4)**

> **作者:** Antoni Kowalczuk; Jan Dubiński; Franziska Boenisch; Adam Dziedzic
>
> **备注:** Accepted at ICML2025
>
> **摘要:** Image AutoRegressive generation has emerged as a new powerful paradigm with image autoregressive models (IARs) matching state-of-the-art diffusion models (DMs) in image quality (FID: 1.48 vs. 1.58) while allowing for a higher generation speed. However, the privacy risks associated with IARs remain unexplored, raising concerns regarding their responsible deployment. To address this gap, we conduct a comprehensive privacy analysis of IARs, comparing their privacy risks to the ones of DMs as reference points. Concretely, we develop a novel membership inference attack (MIA) that achieves a remarkably high success rate in detecting training images (with a True Positive Rate at False Positive Rate = 1% of 86.38% vs. 6.38% for DMs with comparable attacks). We leverage our novel MIA to provide dataset inference (DI) for IARs, and show that it requires as few as 6 samples to detect dataset membership (compared to 200 for DI in DMs), confirming a higher information leakage in IARs. Finally, we are able to extract hundreds of training data points from an IAR (e.g., 698 from VAR-d30). Our results suggest a fundamental privacy-utility trade-off: while IARs excel in image generation quality and speed, they are empirically significantly more vulnerable to privacy attacks compared to DMs that achieve similar performance. We release the code at https://github.com/sprintml/privacy_attacks_against_iars for reproducibility.
>
---
#### [replaced 035] TD-Paint: Faster Diffusion Inpainting Through Time Aware Pixel Conditioning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.09306v2](http://arxiv.org/pdf/2410.09306v2)**

> **作者:** Tsiry Mayet; Pourya Shamsolmoali; Simon Bernard; Eric Granger; Romain Hérault; Clement Chatelain
>
> **摘要:** Diffusion models have emerged as highly effective techniques for inpainting, however, they remain constrained by slow sampling rates. While recent advances have enhanced generation quality, they have also increased sampling time, thereby limiting scalability in real-world applications. We investigate the generative sampling process of diffusion-based inpainting models and observe that these models make minimal use of the input condition during the initial sampling steps. As a result, the sampling trajectory deviates from the data manifold, requiring complex synchronization mechanisms to realign the generation process. To address this, we propose Time-aware Diffusion Paint (TD-Paint), a novel approach that adapts the diffusion process by modeling variable noise levels at the pixel level. This technique allows the model to efficiently use known pixel values from the start, guiding the generation process toward the target manifold. By embedding this information early in the diffusion process, TD-Paint significantly accelerates sampling without compromising image quality. Unlike conventional diffusion-based inpainting models, which require a dedicated architecture or an expensive generation loop, TD-Paint achieves faster sampling times without architectural modifications. Experimental results across three datasets show that TD-Paint outperforms state-of-the-art diffusion models while maintaining lower complexity.
>
---
#### [replaced 036] Temporal-Spectral-Spatial Unified Remote Sensing Dense Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12280v2](http://arxiv.org/pdf/2505.12280v2)**

> **作者:** Sijie Zhao; Feng Liu; Enzhuo Zhang; Yiqing Guo; Pengfeng Xiao; Lei Bai; Xueliang Zhang; Hao Chen; Zhenwei Shi; Wanli Ouyang
>
> **备注:** 14 pages, 6 figures, Code link:https://github.com/walking-shadow/Official_TSSUN
>
> **摘要:** The proliferation of multi-source remote sensing data has propelled the development of deep learning for dense prediction, yet significant challenges in data and task unification persist. Current deep learning architectures for remote sensing are fundamentally rigid. They are engineered for fixed input-output configurations, restricting their adaptability to the heterogeneous spatial, temporal, and spectral dimensions inherent in real-world data. Furthermore, these models neglect the intrinsic correlations among semantic segmentation, binary change detection, and semantic change detection, necessitating the development of distinct models or task-specific decoders. This paradigm is also constrained to a predefined set of output semantic classes, where any change to the classes requires costly retraining. To overcome these limitations, we introduce the Spatial-Temporal-Spectral Unified Network (STSUN) for unified modeling. STSUN can adapt to input and output data with arbitrary spatial sizes, temporal lengths, and spectral bands by leveraging their metadata for a unified representation. Moreover, STSUN unifies disparate dense prediction tasks within a single architecture by conditioning the model on trainable task embeddings. Similarly, STSUN facilitates flexible prediction across any set of semantic categories by integrating trainable category embeddings as metadata. Extensive experiments on multiple datasets with diverse STS configurations in multiple scenarios demonstrate that a single STSUN model effectively adapts to heterogeneous inputs and outputs, unifying various dense prediction tasks and diverse semantic class predictions. The proposed approach consistently achieves state-of-the-art performance, highlighting its robustness and generalizability for complex remote sensing applications.
>
---
#### [replaced 037] Progressive Cross-Stream Cooperation in Spatial and Temporal Domain for Action Localization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/1905.11575v2](http://arxiv.org/pdf/1905.11575v2)**

> **作者:** Rui Su; Dong Xu; Luping Zhou; Wanli Ouyang
>
> **备注:** TPAMI
>
> **摘要:** Spatio-temporal action localization consists of three levels of tasks: spatial localization, action classification, and temporal localization. In this work, we propose a new progressive cross-stream cooperation (PCSC) framework that improves all three tasks above. The basic idea is to utilize both spatial region (resp., temporal segment proposals) and features from one stream (i.e., the Flow/RGB stream) to help another stream (i.e., the RGB/Flow stream) to iteratively generate better bounding boxes in the spatial domain (resp., temporal segments in the temporal domain). In this way, not only the actions could be more accurately localized both spatially and temporally, but also the action classes could be predicted more precisely. Specifically, we first combine the latest region proposals (for spatial detection) or segment proposals (for temporal localization) from both streams to form a larger set of labelled training samples to help learn better action detection or segment detection models. Second, to learn better representations, we also propose a new message passing approach to pass information from one stream to another stream, which also leads to better action detection and segment detection models. By first using our newly proposed PCSC framework for spatial localization at the frame-level and then applying our temporal PCSC framework for temporal localization at the tube-level, the action localization results are progressively improved at both the frame level and the video level. Comprehensive experiments on two benchmark datasets UCF-101-24 and J-HMDB demonstrate the effectiveness of our newly proposed approaches for spatio-temporal action localization in realistic scenarios.
>
---
#### [replaced 038] SycnMapV2: Robust and Adaptive Unsupervised Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.16297v2](http://arxiv.org/pdf/2506.16297v2)**

> **作者:** Heng Zhang; Zikang Wan; Danilo Vasconcellos Vargas
>
> **摘要:** Human vision excels at segmenting visual cues without the need for explicit training, and it remains remarkably robust even as noise severity increases. In contrast, existing AI algorithms struggle to maintain accuracy under similar conditions. Here, we present SyncMapV2, the first to solve unsupervised segmentation with state-of-the-art robustness. SyncMapV2 exhibits a minimal drop in mIoU, only 0.01%, under digital corruption, compared to a 23.8% drop observed in SOTA methods. This superior performance extends across various types of corruption: noise (7.3% vs. 37.7%), weather (7.5% vs. 33.8%), and blur (7.0% vs. 29.5%). Notably, SyncMapV2 accomplishes this without any robust training, supervision, or loss functions. It is based on a learning paradigm that uses self-organizing dynamical equations combined with concepts from random networks. Moreover, unlike conventional methods that require re-initialization for each new input, SyncMapV2 adapts online, mimicking the continuous adaptability of human vision. Thus, we go beyond the accurate and robust results, and present the first algorithm that can do all the above online, adapting to input rather than re-initializing. In adaptability tests, SyncMapV2 demonstrates near-zero performance degradation, which motivates and fosters a new generation of robust and adaptive intelligence in the near future.
>
---
#### [replaced 039] DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11558v2](http://arxiv.org/pdf/2506.11558v2)**

> **作者:** Bo-Cheng Chiu; Jen-Jee Chen; Yu-Chee Tseng; Feng-Chi Chen
>
> **备注:** I would like to request the withdrawal of this submission because the current version contains significant errors and incomplete results. I intend to revise the manuscript thoroughly before resubmitting. I apologize for the oversight and appreciate your understanding
>
> **摘要:** Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with GPT-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling.
>
---
#### [replaced 040] VesselSAM: Leveraging SAM for Aortic Vessel Segmentation with AtrousLoRA
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.18185v4](http://arxiv.org/pdf/2502.18185v4)**

> **作者:** Adnan Iltaf; Rayan Merghani Ahmed; Zhenxi Zhang; Bin Li; Shoujun Zhou
>
> **备注:** Work in progress
>
> **摘要:** Medical image segmentation is crucial for clinical diagnosis and treatment planning, especially when dealing with complex anatomical structures such as vessels. However, accurately segmenting vessels remains challenging due to their small size, intricate edge structures, and susceptibility to artifacts and imaging noise. In this work, we propose VesselSAM, an enhanced version of the Segment Anything Model (SAM), specifically tailored for aortic vessel segmentation. VesselSAM incorporates AtrousLoRA, a novel module integrating Atrous Attention and Low-Rank Adaptation (LoRA), to enhance segmentation performance. Atrous Attention enables the model to capture multi-scale contextual information, preserving both fine-grained local details and broader global context. Additionally, LoRA facilitates efficient fine-tuning of the frozen SAM image encoder, reducing the number of trainable parameters and thereby enhancing computational efficiency. We evaluate VesselSAM using two challenging datasets: the Aortic Vessel Tree (AVT) dataset and the Type-B Aortic Dissection (TBAD) dataset. VesselSAM achieves state-of-the-art performance, attaining DSC scores of 93.50\%, 93.25\%, 93.02\%, and 93.26\% across multi-center datasets. Our results demonstrate that VesselSAM delivers high segmentation accuracy while significantly reducing computational overhead compared to existing large-scale models. This development paves the way for enhanced AI-based aortic vessel segmentation in clinical environments. The code and models will be released at https://github.com/Adnan-CAS/AtrousLora.
>
---
#### [replaced 041] MIFNet: Learning Modality-Invariant Features for Generalizable Multimodal Image Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.11299v3](http://arxiv.org/pdf/2501.11299v3)**

> **作者:** Yepeng Liu; Zhichao Sun; Baosheng Yu; Yitian Zhao; Bo Du; Yongchao Xu; Jun Cheng
>
> **备注:** Accept by IEEE TIP 2025
>
> **摘要:** Many keypoint detection and description methods have been proposed for image matching or registration. While these methods demonstrate promising performance for single-modality image matching, they often struggle with multimodal data because the descriptors trained on single-modality data tend to lack robustness against the non-linear variations present in multimodal data. Extending such methods to multimodal image matching often requires well-aligned multimodal data to learn modality-invariant descriptors. However, acquiring such data is often costly and impractical in many real-world scenarios. To address this challenge, we propose a modality-invariant feature learning network (MIFNet) to compute modality-invariant features for keypoint descriptions in multimodal image matching using only single-modality training data. Specifically, we propose a novel latent feature aggregation module and a cumulative hybrid aggregation module to enhance the base keypoint descriptors trained on single-modality data by leveraging pre-trained features from Stable Diffusion models. %, our approach generates robust and invariant features across diverse and unknown modalities. We validate our method with recent keypoint detection and description methods in three multimodal retinal image datasets (CF-FA, CF-OCT, EMA-OCTA) and two remote sensing datasets (Optical-SAR and Optical-NIR). Extensive experiments demonstrate that the proposed MIFNet is able to learn modality-invariant feature for multimodal image matching without accessing the targeted modality and has good zero-shot generalization ability. The code will be released at https://github.com/lyp-deeplearning/MIFNet.
>
---
#### [replaced 042] Contactless Cardiac Pulse Monitoring Using Event Cameras
- **分类: cs.CV; cs.ET; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.09529v2](http://arxiv.org/pdf/2505.09529v2)**

> **作者:** Mohamed Moustafa; Joseph Lemley; Peter Corcoran
>
> **摘要:** Time event cameras are a novel technology for recording scene information at extremely low latency and with low power consumption. Event cameras output a stream of events that encapsulate pixel-level light intensity changes within the scene, capturing information with a higher dynamic range and temporal resolution than traditional cameras. This study investigates the contact-free reconstruction of an individual's cardiac pulse signal from time event recording of their face using a supervised convolutional neural network (CNN) model. An end-to-end model is trained to extract the cardiac signal from a two-dimensional representation of the event stream, with model performance evaluated based on the accuracy of the calculated heart rate. The experimental results confirm that physiological cardiac information in the facial region is effectively preserved within the event stream, showcasing the potential of this novel sensor for remote heart rate monitoring. The model trained on event frames achieves a root mean square error (RMSE) of 3.32 beats per minute (bpm) compared to the RMSE of 2.92 bpm achieved by the baseline model trained on standard camera frames. Furthermore, models trained on event frames generated at 60 and 120 FPS outperformed the 30 FPS standard camera results, achieving an RMSE of 2.54 and 2.13 bpm, respectively.
>
---
#### [replaced 043] A Contrastive Learning Foundation Model Based on Perfectly Aligned Sample Pairs for Remote Sensing Images
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19447v2](http://arxiv.org/pdf/2505.19447v2)**

> **作者:** Hengtong Shen; Haiyan Gu; Haitao Li; Yi Yang; Agen Qiu
>
> **摘要:** Self-Supervised Learning (SSL) enables us to pre-train foundation models without costly labeled data. Among SSL methods, Contrastive Learning (CL) methods are better at obtaining accurate semantic representations in noise interference. However, due to the significant domain gap, while CL methods have achieved great success in many computer vision tasks, they still require specific adaptation for Remote Sensing (RS) images. To this end, we present a novel self-supervised method called PerA, which produces all-purpose RS features through semantically Perfectly Aligned sample pairs. Specifically, PerA obtains features from sampled views by applying spatially disjoint masks to augmented images rather than random cropping. Our framework provides high-quality features by ensuring consistency between teacher and student and predicting learnable mask tokens. Compared to previous contrastive methods, our method demonstrates higher memory efficiency and can be trained with larger batches due to its sparse inputs. Additionally, the proposed method demonstrates remarkable adaptability to uncurated RS data and reduce the impact of the potential semantic inconsistency. We also collect an unlabeled pre-training dataset, which contains about 5 million RS images. We conducted experiments on multiple downstream task datasets and achieved performance comparable to previous state-of-the-art methods with a limited model scale, demonstrating the effectiveness of our approach. We hope this work will contribute to practical remote sensing interpretation works.
>
---
#### [replaced 044] DDS-NAS: Dynamic Data Selection within Neural Architecture Search via On-line Hard Example Mining applied to Image Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14667v2](http://arxiv.org/pdf/2506.14667v2)**

> **作者:** Matt Poyser; Toby P. Breckon
>
> **备注:** 27 single-column pages, 8 figures, to be published in Pattern Recognition
>
> **摘要:** In order to address the scalability challenge within Neural Architecture Search (NAS), we speed up NAS training via dynamic hard example mining within a curriculum learning framework. By utilizing an autoencoder that enforces an image similarity embedding in latent space, we construct an efficient kd-tree structure to order images by furthest neighbour dissimilarity in a low-dimensional embedding. From a given query image from our subsample dataset, we can identify the most dissimilar image within the global dataset in logarithmic time. Via curriculum learning, we then dynamically re-formulate an unbiased subsample dataset for NAS optimisation, upon which the current NAS solution architecture performs poorly. We show that our DDS-NAS framework speeds up gradient-based NAS strategies by up to 27x without loss in performance. By maximising the contribution of each image sample during training, we reduce the duration of a NAS training cycle and the number of iterations required for convergence.
>
---
#### [replaced 045] MAMMA: Markerless & Automatic Multi-Person Motion Action Capture
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13040v2](http://arxiv.org/pdf/2506.13040v2)**

> **作者:** Hanz Cuevas-Velasquez; Anastasios Yiannakidis; Soyong Shin; Giorgio Becherini; Markus Höschle; Joachim Tesch; Taylor Obersat; Tsvetelina Alexiadis; Michael J. Black
>
> **摘要:** We present MAMMA, a markerless motion-capture pipeline that accurately recovers SMPL-X parameters from multi-view video of two-person interaction sequences. Traditional motion-capture systems rely on physical markers. Although they offer high accuracy, their requirements of specialized hardware, manual marker placement, and extensive post-processing make them costly and time-consuming. Recent learning-based methods attempt to overcome these limitations, but most are designed for single-person capture, rely on sparse keypoints, or struggle with occlusions and physical interactions. In this work, we introduce a method that predicts dense 2D surface landmarks conditioned on segmentation masks, enabling person-specific correspondence estimation even under heavy occlusion. We employ a novel architecture that exploits learnable queries for each landmark. We demonstrate that our approach can handle complex person--person interaction and offers greater accuracy than existing methods. To train our network, we construct a large, synthetic multi-view dataset combining human motions from diverse sources, including extreme poses, hand motions, and close interactions. Our dataset yields high-variability synthetic sequences with rich body contact and occlusion, and includes SMPL-X ground-truth annotations with dense 2D landmarks. The result is a system capable of capturing human motion without the need for markers. Our approach offers competitive reconstruction quality compared to commercial marker-based motion-capture solutions, without the extensive manual cleanup. Finally, we address the absence of common benchmarks for dense-landmark prediction and markerless motion capture by introducing two evaluation settings built from real multi-view sequences. We will release our dataset, benchmark, method, training code, and pre-trained model weights for research purposes.
>
---
#### [replaced 046] FusionForce: End-to-end Differentiable Neural-Symbolic Layer for Trajectory Prediction
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.10156v4](http://arxiv.org/pdf/2502.10156v4)**

> **作者:** Ruslan Agishev; Karel Zimmermann
>
> **备注:** Code: https://github.com/ctu-vras/fusionforce
>
> **摘要:** We propose end-to-end differentiable model that predicts robot trajectories on rough offroad terrain from camera images and/or lidar point clouds. The model integrates a learnable component that predicts robot-terrain interaction forces with a neural-symbolic layer that enforces the laws of classical mechanics and consequently improves generalization on out-of-distribution data. The neural-symbolic layer includes a differentiable physics engine that computes the robot's trajectory by querying these forces at the points of contact with the terrain. As the proposed architecture comprises substantial geometrical and physics priors, the resulting model can also be seen as a learnable physics engine conditioned on real sensor data that delivers $10^4$ trajectories per second. We argue and empirically demonstrate that this architecture reduces the sim-to-real gap and mitigates out-of-distribution sensitivity. The differentiability, in conjunction with the rapid simulation speed, makes the model well-suited for various applications including model predictive control, trajectory shooting, supervised and reinforcement learning, or SLAM.
>
---
#### [replaced 047] ReconX: Reconstruct Any Scene from Sparse Views with Video Diffusion Model
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [http://arxiv.org/pdf/2408.16767v3](http://arxiv.org/pdf/2408.16767v3)**

> **作者:** Fangfu Liu; Wenqiang Sun; Hanyang Wang; Yikai Wang; Haowen Sun; Junliang Ye; Jun Zhang; Yueqi Duan
>
> **备注:** Project page: https://liuff19.github.io/ReconX
>
> **摘要:** Advancements in 3D scene reconstruction have transformed 2D images from the real world into 3D models, producing realistic 3D results from hundreds of input photos. Despite great success in dense-view reconstruction scenarios, rendering a detailed scene from insufficient captured views is still an ill-posed optimization problem, often resulting in artifacts and distortions in unseen areas. In this paper, we propose ReconX, a novel 3D scene reconstruction paradigm that reframes the ambiguous reconstruction challenge as a temporal generation task. The key insight is to unleash the strong generative prior of large pre-trained video diffusion models for sparse-view reconstruction. However, 3D view consistency struggles to be accurately preserved in directly generated video frames from pre-trained models. To address this, given limited input views, the proposed ReconX first constructs a global point cloud and encodes it into a contextual space as the 3D structure condition. Guided by the condition, the video diffusion model then synthesizes video frames that are both detail-preserved and exhibit a high degree of 3D consistency, ensuring the coherence of the scene from various perspectives. Finally, we recover the 3D scene from the generated video through a confidence-aware 3D Gaussian Splatting optimization scheme. Extensive experiments on various real-world datasets show the superiority of our ReconX over state-of-the-art methods in terms of quality and generalizability.
>
---
#### [replaced 048] Stepping Out of Similar Semantic Space for Open-Vocabulary Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.16058v2](http://arxiv.org/pdf/2506.16058v2)**

> **作者:** Yong Liu; SongLi Wu; Sule Bai; Jiahao Wang; Yitong Wang; Yansong Tang
>
> **摘要:** Open-vocabulary segmentation aims to achieve segmentation of arbitrary categories given unlimited text inputs as guidance. To achieve this, recent works have focused on developing various technical routes to exploit the potential of large-scale pre-trained vision-language models and have made significant progress on existing benchmarks. However, we find that existing test sets are limited in measuring the models' comprehension of ``open-vocabulary" concepts, as their semantic space closely resembles the training space, even with many overlapping categories. To this end, we present a new benchmark named OpenBench that differs significantly from the training semantics. It is designed to better assess the model's ability to understand and segment a wide range of real-world concepts. When testing existing methods on OpenBench, we find that their performance diverges from the conclusions drawn on existing test sets. In addition, we propose a method named OVSNet to improve the segmentation performance for diverse and open scenarios. Through elaborate fusion of heterogeneous features and cost-free expansion of the training space, OVSNet achieves state-of-the-art results on both existing datasets and our proposed OpenBench. Corresponding analysis demonstrate the soundness and effectiveness of our proposed benchmark and method.
>
---
#### [replaced 049] Improved and Explainable Cervical Cancer Classification using Ensemble Pooling of Block Fused Descriptors
- **分类: eess.IV; cs.CV; cs.LG; I.2.1; I.5.2**

- **链接: [http://arxiv.org/pdf/2405.01600v2](http://arxiv.org/pdf/2405.01600v2)**

> **作者:** Saurabh Saini; Kapil Ahuja; Akshat S. Chauhan
>
> **备注:** 26 Pages, 10 figures, and 8 tables
>
> **摘要:** Cervical cancer is the second most common cancer in women and causes high death rates. Earlier models for detecting cervical cancer had limited success. In this work, we propose new models that substantially outperform previous models. Previous studies show that pretrained ResNets extract features from cervical cancer images well. Hence, our first model involves working with three ResNets (50, 101, 152). All the existing works use only the last convolution block of their respective ResNet, which captures abstract features (e.g., shapes, objects). However, we believe that detailed features (e.g., color, edges, texture), coming from earlier convolution blocks, are equally important for cancer (specifically cervical cancer) classification. Since now the number of features become large, we use a novel feature selection technique of Global Max Pooling for detailed features and Global Average Pooling for abstract features. Hence, our second model consists of the resulting Cascaded Block Fused variants of the three ResNets. To improve the performance further, we combine and normalize the features of the three standard ResNets as well as our proposed three Cascaded Block Fused ResNets. This type of combination is also new in cancer classification domain (also in cervical cancer), and results in our third and fourth models, respectively. We use a linear SVM for classification. We exhaustively perform experiments on two public datasets, IARC and AnnoCerv, achieving an average performance of 97.92% and 92.97% surpassing standard ResNets performance of 90.89% and 87.97%, respectively. We outperform the competitive approach available on IARC dataset with an average gain of 13.20%, while no prior competitive work available on AnnoCerv. Additionally, we introduce a novel SHAP+LIME explainability method, accurately identifying the cancerous region in 97% of cases.
>
---
#### [replaced 050] Diff-Def: Diffusion-Generated Deformation Fields for Conditional Atlases
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.16776v3](http://arxiv.org/pdf/2403.16776v3)**

> **作者:** Sophie Starck; Vasiliki Sideri-Lampretsa; Bernhard Kainz; Martin J. Menten; Tamara T. Mueller; Daniel Rueckert
>
> **摘要:** Anatomical atlases are widely used for population studies and analysis. Conditional atlases target a specific sub-population defined via certain conditions, such as demographics or pathologies, and allow for the investigation of fine-grained anatomical differences like morphological changes associated with ageing or disease. Existing approaches use either registration-based methods that are often unable to handle large anatomical variations or generative adversarial models, which are challenging to train since they can suffer from training instabilities. Instead of generating atlases directly in as intensities, we propose using latent diffusion models to generate deformation fields, which transform a general population atlas into one representing a specific sub-population. Our approach ensures structural integrity, enhances interpretability and avoids hallucinations that may arise during direct image synthesis by generating this deformation field and regularising it using a neighbourhood of images. We compare our method to several state-of-the-art atlas generation methods using brain MR images from the UK Biobank. Our method generates highly realistic atlases with smooth transformations and high anatomical fidelity, outperforming existing baselines. We demonstrate the quality of these atlases through comprehensive evaluations, including quantitative metrics for anatomical accuracy, perceptual similarity, and qualitative analyses displaying the consistency and realism of the generated atlases.
>
---
#### [replaced 051] Aligning Anime Video Generation with Human Feedback
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10044v2](http://arxiv.org/pdf/2504.10044v2)**

> **作者:** Bingwen Zhu; Yudong Jiang; Baohan Xu; Siqian Yang; Mingyu Yin; Yidi Wu; Huyang Sun; Zuxuan Wu
>
> **备注:** 10 pages, 7 figures, 7 tables
>
> **摘要:** Anime video generation faces significant challenges due to the scarcity of anime data and unusual motion patterns, leading to issues such as motion distortion and flickering artifacts, which result in misalignment with human preferences. Existing reward models, designed primarily for real-world videos, fail to capture the unique appearance and consistency requirements of anime. In this work, we propose a pipeline to enhance anime video generation by leveraging human feedback for better alignment. Specifically, we construct the first multi-dimensional reward dataset for anime videos, comprising 30k human-annotated samples that incorporating human preferences for both visual appearance and visual consistency. Based on this, we develop AnimeReward, a powerful reward model that employs specialized vision-language models for different evaluation dimensions to guide preference alignment. Furthermore, we introduce Gap-Aware Preference Optimization (GAPO), a novel training method that explicitly incorporates preference gaps into the optimization process, enhancing alignment performance and efficiency. Extensive experiment results show that AnimeReward outperforms existing reward models, and the inclusion of GAPO leads to superior alignment in both quantitative benchmarks and human evaluations, demonstrating the effectiveness of our pipeline in enhancing anime video quality. Our code and dataset are publicly available at https://github.com/bilibili/Index-anisora.
>
---
#### [replaced 052] Pro-AD: Learning Comprehensive Prototypes with Prototype-based Constraint for Multi-class Unsupervised Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13097v3](http://arxiv.org/pdf/2506.13097v3)**

> **作者:** Ziqing Zhou; Yurui Pan; Lidong Wang; Wenbing Zhu; Mingmin Chi; Dong Wu; Bo Peng
>
> **摘要:** Prototype-based reconstruction methods for unsupervised anomaly detection utilize a limited set of learnable prototypes which only aggregates insufficient normal information, resulting in undesirable reconstruction. However, increasing the number of prototypes may lead to anomalies being well reconstructed through the attention mechanism, which we refer to as the "Soft Identity Mapping" problem. In this paper, we propose Pro-AD to address these issues and fully utilize the prototypes to boost the performance of anomaly detection. Specifically, we first introduce an expanded set of learnable prototypes to provide sufficient capacity for semantic information. Then we employ a Dynamic Bidirectional Decoder which integrates the process of the normal information aggregation and the target feature reconstruction via prototypes, with the aim of allowing the prototypes to aggregate more comprehensive normal semantic information from different levels of the image features and the target feature reconstruction to not only utilize its contextual information but also dynamically leverage the learned comprehensive prototypes. Additionally, to prevent the anomalies from being well reconstructed using sufficient semantic information through the attention mechanism, Pro-AD introduces a Prototype-based Constraint that applied within the target feature reconstruction process of the decoder, which further improves the performance of our approach. Extensive experiments on multiple challenging benchmarks demonstrate that our Pro-AD achieve state-of-the-art performance, highlighting its superior robustness and practical effectiveness for Multi-class Unsupervised Anomaly Detection task.
>
---
#### [replaced 053] LAuReL: Learned Augmented Residual Layer
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.07501v4](http://arxiv.org/pdf/2411.07501v4)**

> **作者:** Gaurav Menghani; Ravi Kumar; Sanjiv Kumar
>
> **备注:** Accepted at 42nd International Conference on Machine Learning (2025), Vancouver, Canada
>
> **摘要:** One of the core pillars of efficient deep learning methods is architectural improvements such as the residual/skip connection, which has led to significantly better model convergence and quality. Since then the residual connection has become ubiquitous in not just convolutional neural networks but also transformer-based architectures, the backbone of LLMs. In this paper we introduce Learned Augmented Residual Layer (LAuReL) -- a novel generalization of the canonical residual connection -- with the goal to be an in-situ replacement of the latter while outperforming on both model quality and footprint metrics. Our experiments show that using LAuReL can help boost performance for both vision and language models. For example, on the ResNet-50, ImageNet 1K task, it achieves 60% of the gains from adding an extra layer, while only adding 0.003% more parameters, and matches it while adding 2.6 times fewer parameters. Similarly, when pre-training 1B and 4B parameter LLMs, LAuReL improves performance on a variety of challenging downstream evaluation tasks by 2.54% to 20.05%, while adding only 0.012% and 0.1% additional parameters, respectively.
>
---
#### [replaced 054] Why Sample Space Matters: Keyframe Sampling Optimization for LiDAR-based Place Recognition
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.02643v3](http://arxiv.org/pdf/2410.02643v3)**

> **作者:** Nikolaos Stathoulopoulos; Vidya Sumathy; Christoforos Kanellakis; George Nikolakopoulos
>
> **备注:** The work is no longer intended for consideration in its current form. Readers are instead encouraged to refer to our related and more complete study, arXiv:2501.01791, which should be considered as a stand-alone contribution
>
> **摘要:** Recent advances in robotics are driving real-world autonomy for long-term and large-scale missions, where loop closures via place recognition are vital for mitigating pose estimation drift. However, achieving real-time performance remains challenging for resource-constrained mobile robots and multi-robot systems due to the computational burden of high-density sampling, which increases the complexity of comparing and verifying query samples against a growing map database. Conventional methods often retain redundant information or miss critical data by relying on fixed sampling intervals or operating in 3-D space instead of the descriptor feature space. To address these challenges, we introduce the concept of sample space and propose a novel keyframe sampling approach for LiDAR-based place recognition. Our method minimizes redundancy while preserving essential information in the hyper-dimensional descriptor space, supporting both learning-based and handcrafted descriptors. The proposed approach incorporates a sliding window optimization strategy to ensure efficient keyframe selection and real-time performance, enabling seamless integration into robotic pipelines. In sum, our approach demonstrates robust performance across diverse datasets, with the ability to adapt seamlessly from indoor to outdoor scenarios without parameter tuning, reducing loop closure detection times and memory requirements.
>
---
#### [replaced 055] DivTrackee versus DynTracker: Promoting Diversity in Anti-Facial Recognition against Dynamic FR Strategy
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2501.06533v2](http://arxiv.org/pdf/2501.06533v2)**

> **作者:** Wenshu Fan; Minxing Zhang; Hongwei Li; Wenbo Jiang; Hanxiao Chen; Xiangyu Yue; Michael Backes; Xiao Zhang
>
> **摘要:** The widespread adoption of facial recognition (FR) models raises serious concerns about their potential misuse, motivating the development of anti-facial recognition (AFR) to protect user facial privacy. In this paper, we argue that the static FR strategy, predominantly adopted in prior literature for evaluating AFR efficacy, cannot faithfully characterize the actual capabilities of determined trackers who aim to track a specific target identity. In particular, we introduce DynTracker, a dynamic FR strategy where the model's gallery database is iteratively updated with newly recognized target identity images. Surprisingly, such a simple approach renders all the existing AFR protections ineffective. To mitigate the privacy threats posed by DynTracker, we advocate for explicitly promoting diversity in the AFR-protected images. We hypothesize that the lack of diversity is the primary cause of the failure of existing AFR methods. Specifically, we develop DivTrackee, a novel method for crafting diverse AFR protections that builds upon a text-guided image generation framework and diversity-promoting adversarial losses. Through comprehensive experiments on various image benchmarks and feature extractors, we demonstrate DynTracker's strength in breaking existing AFR methods and the superiority of DivTrackee in preventing user facial images from being identified by dynamic FR strategies. We believe our work can act as an important initial step towards developing more effective AFR methods for protecting user facial privacy against determined trackers.
>
---
#### [replaced 056] RRCANet: Recurrent Reusable-Convolution Attention Network for Infrared Small Target Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02393v2](http://arxiv.org/pdf/2506.02393v2)**

> **作者:** Yongxian Liu; Boyang Li; Ting Liu; Zaiping Lin; Wei An
>
> **备注:** We have corrected some annotation errors in the figures
>
> **摘要:** Infrared small target detection is a challenging task due to its unique characteristics (e.g., small, dim, shapeless and changeable). Recently published CNN-based methods have achieved promising performance with heavy feature extraction and fusion modules. To achieve efficient and effective detection, we propose a recurrent reusable-convolution attention network (RRCA-Net) for infrared small target detection. Specifically, RRCA-Net incorporates reusable-convolution block (RuCB) in a recurrent manner without introducing extra parameters. With the help of the repetitive iteration in RuCB, the high-level information of small targets in the deep layers can be well maintained and further refined. Then, a dual interactive attention aggregation module (DIAAM) is proposed to promote the mutual enhancement and fusion of refined information. In this way, RRCA-Net can both achieve high-level feature refinement and enhance the correlation of contextual information between adjacent layers. Moreover, to achieve steady convergence, we design a target characteristic inspired loss function (DpT-k loss) by integrating physical and mathematical constraints. Experimental results on three benchmark datasets (e.g. NUAA-SIRST, IRSTD-1k, DenseSIRST) demonstrate that our RRCA-Net can achieve comparable performance to the state-of-the-art methods while maintaining a small number of parameters, and act as a plug and play module to introduce consistent performance improvement for several popular IRSTD methods. Our code will be available at https://github.com/yongxianLiu/ soon.
>
---
#### [replaced 057] Grounding Beyond Detection: Enhancing Contextual Understanding in Embodied 3D Grounding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05199v2](http://arxiv.org/pdf/2506.05199v2)**

> **作者:** Yani Zhang; Dongming Wu; Hao Shi; Yingfei Liu; Tiancai Wang; Haoqiang Fan; Xingping Dong
>
> **备注:** 1st place on EmbodiedScan visual grounding
>
> **摘要:** Embodied 3D grounding aims to localize target objects described in human instructions from ego-centric viewpoint. Most methods typically follow a two-stage paradigm where a trained 3D detector's optimized backbone parameters are used to initialize a grounding model. In this study, we explore a fundamental question: Does embodied 3D grounding benefit enough from detection? To answer this question, we assess the grounding performance of detection models using predicted boxes filtered by the target category. Surprisingly, these detection models without any instruction-specific training outperform the grounding models explicitly trained with language instructions. This indicates that even category-level embodied 3D grounding may not be well resolved, let alone more fine-grained context-aware grounding. Motivated by this finding, we propose DEGround, which shares DETR queries as object representation for both DEtection and Grounding and enables the grounding to benefit from basic category classification and box detection. Based on this framework, we further introduce a regional activation grounding module that highlights instruction-related regions and a query-wise modulation module that incorporates sentence-level semantic into the query representation, strengthening the context-aware understanding of language instructions. Remarkably, DEGround outperforms state-of-the-art model BIP3D by 7.52% at overall accuracy on the EmbodiedScan validation set. The source code will be publicly available at https://github.com/zyn213/DEGround.
>
---
#### [replaced 058] Hadamard Attention Recurrent Transformer: A Strong Baseline for Stereo Matching Transformer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.01023v3](http://arxiv.org/pdf/2501.01023v3)**

> **作者:** Ziyang Chen; Wenting Li; Yongjun Zhang; Yabo Wu; Bingshu Wang; Yong Zhao; C. L. Philip Chen
>
> **摘要:** Constrained by the low-rank bottleneck inherent in attention mechanisms, current stereo matching transformers suffer from limited nonlinear expressivity, which renders their feature representations sensitive to challenging conditions such as reflections. To overcome this difficulty, we present the Hadamard Attention Recurrent Stereo Transformer (HART). HART includes a novel attention mechanism that incorporates the following components: 1) The Dense Attention Kernel (DAK) maps the attention weight distribution into a high-dimensional space over (0, +$\infty$). By removing the upper bound constraint on attention weights, DAK enables more flexible modeling of complex feature interactions. This reduces feature collinearity. 2) The Multi Kernel & Order Interaction (MKOI) module extends the attention mechanism by unifying semantic and spatial knowledge learning. This integration improves the ability of HART to learn features in binocular images. Experimental results demonstrate the effectiveness of our HART. In reflective area, HART ranked 1st on the KITTI 2012 benchmark among all published methods at the time of submission. Code is available at https://github.com/ZYangChen/HART.
>
---
#### [replaced 059] Privacy-Shielded Image Compression: Defending Against Exploitation from Vision-Language Pretrained Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.15201v2](http://arxiv.org/pdf/2506.15201v2)**

> **作者:** Xuelin Shen; Jiayin Xu; Kangsheng Yin; Wenhan Yang
>
> **备注:** 11 pages, 6 figures, publised to ICML 2025
>
> **摘要:** The improved semantic understanding of vision-language pretrained (VLP) models has made it increasingly difficult to protect publicly posted images from being exploited by search engines and other similar tools. In this context, this paper seeks to protect users' privacy by implementing defenses at the image compression stage to prevent exploitation. Specifically, we propose a flexible coding method, termed Privacy-Shielded Image Compression (PSIC), that can produce bitstreams with multiple decoding options. By default, the bitstream is decoded to preserve satisfactory perceptual quality while preventing interpretation by VLP models. Our method also retains the original image compression functionality. With a customizable input condition, the proposed scheme can reconstruct the image that preserves its full semantic information. A Conditional Latent Trigger Generation (CLTG) module is proposed to produce bias information based on customizable conditions to guide the decoding process into different reconstructed versions, and an Uncertainty-Aware Encryption-Oriented (UAEO) optimization function is designed to leverage the soft labels inferred from the target VLP model's uncertainty on the training data. This paper further incorporates an adaptive multi-objective optimization strategy to obtain improved encrypting performance and perceptual quality simultaneously within a unified training process. The proposed scheme is plug-and-play and can be seamlessly integrated into most existing Learned Image Compression (LIC) models. Extensive experiments across multiple downstream tasks have demonstrated the effectiveness of our design.
>
---
#### [replaced 060] ObjCtrl-2.5D: Training-free Object Control with Camera Poses
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.07721v2](http://arxiv.org/pdf/2412.07721v2)**

> **作者:** Zhouxia Wang; Yushi Lan; Shangchen Zhou; Chen Change Loy
>
> **备注:** Project Page: https://wzhouxiff.github.io/projects/ObjCtrl-2.5D/
>
> **摘要:** This study aims to achieve more precise and versatile object control in image-to-video (I2V) generation. Current methods typically represent the spatial movement of target objects with 2D trajectories, which often fail to capture user intention and frequently produce unnatural results. To enhance control, we present ObjCtrl-2.5D, a training-free object control approach that uses a 3D trajectory, extended from a 2D trajectory with depth information, as a control signal. By modeling object movement as camera movement, ObjCtrl-2.5D represents the 3D trajectory as a sequence of camera poses, enabling object motion control using an existing camera motion control I2V generation model (CMC-I2V) without training. To adapt the CMC-I2V model originally designed for global motion control to handle local object motion, we introduce a module to isolate the target object from the background, enabling independent local control. In addition, we devise an effective way to achieve more accurate object control by sharing low-frequency warped latent within the object's region across frames. Extensive experiments demonstrate that ObjCtrl-2.5D significantly improves object control accuracy compared to training-free methods and offers more diverse control capabilities than training-based approaches using 2D trajectories, enabling complex effects like object rotation. Code and results are available at https://wzhouxiff.github.io/projects/ObjCtrl-2.5D/.
>
---
#### [replaced 061] ConciseHint: Boosting Efficient Reasoning via Continuous Concise Hints during Generation
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18810v2](http://arxiv.org/pdf/2506.18810v2)**

> **作者:** Siao Tang; Xinyin Ma; Gongfan Fang; Xinchao Wang
>
> **备注:** Codes are available at https://github.com/tsa18/ConciseHint
>
> **摘要:** Recent advancements in large reasoning models (LRMs) like DeepSeek-R1 and OpenAI o1 series have achieved notable performance enhancements on complex reasoning tasks by scaling up the generation length by Chain-of-Thought (CoT). However, an emerging issue is their inclination to produce excessively verbose reasoning processes, leading to the inefficiency problem. Existing literature on improving efficiency mainly adheres to the before-reasoning paradigms such as prompting and reasoning or fine-tuning and reasoning, but ignores the promising direction of directly encouraging the model to speak concisely by intervening during the generation of reasoning. In order to fill the blank, we propose a framework dubbed ConciseHint, which continuously encourages the reasoning model to speak concisely by injecting the textual hint (manually designed or trained on the concise data) during the token generation of the reasoning process. Besides, ConciseHint is adaptive to the complexity of the query by adaptively adjusting the hint intensity, which ensures it will not undermine model performance. Experiments on the state-of-the-art LRMs, including DeepSeek-R1 and Qwen-3 series, demonstrate that our method can effectively produce concise reasoning processes while maintaining performance well. For instance, we achieve a reduction ratio of 65\% for the reasoning length on GSM8K benchmark with Qwen-3 4B with nearly no accuracy loss.
>
---
#### [replaced 062] Exclusive Style Removal for Cross Domain Novel Class Discovery
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.18140v4](http://arxiv.org/pdf/2406.18140v4)**

> **作者:** Yicheng Wang; Feng Liu; Junmin Liu; Kai Sun
>
> **摘要:** As a promising field in open-world learning, \textit{Novel Class Discovery} (NCD) is usually a task to cluster unseen novel classes in an unlabeled set based on the prior knowledge of labeled data within the same domain. However, the performance of existing NCD methods could be severely compromised when novel classes are sampled from a different distribution with the labeled ones. In this paper, we explore and establish the solvability of NCD with cross domain setting under the necessary condition that the style information needs to be removed. Based on the theoretical analysis, we introduce an exclusive style removal module for extracting style information that is distinctive from the baseline features, thereby facilitating inference. Moreover, this module is easy to integrate with other NCD methods, acting as a plug-in to improve performance on novel classes with different distributions compared to the labeled set. Additionally, recognizing the non-negligible influence of different backbones and pre-training strategies on the performance of the NCD methods, we build a fair benchmark for future NCD research. Extensive experiments on three common datasets demonstrate the effectiveness of our proposed style removal strategy.
>
---
#### [replaced 063] SemGauss-SLAM: Dense Semantic Gaussian Splatting SLAM
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2403.07494v4](http://arxiv.org/pdf/2403.07494v4)**

> **作者:** Siting Zhu; Renjie Qin; Guangming Wang; Jiuming Liu; Hesheng Wang
>
> **备注:** IROS 2025
>
> **摘要:** We propose SemGauss-SLAM, a dense semantic SLAM system utilizing 3D Gaussian representation, that enables accurate 3D semantic mapping, robust camera tracking, and high-quality rendering simultaneously. In this system, we incorporate semantic feature embedding into 3D Gaussian representation, which effectively encodes semantic information within the spatial layout of the environment for precise semantic scene representation. Furthermore, we propose feature-level loss for updating 3D Gaussian representation, enabling higher-level guidance for 3D Gaussian optimization. In addition, to reduce cumulative drift in tracking and improve semantic reconstruction accuracy, we introduce semantic-informed bundle adjustment. By leveraging multi-frame semantic associations, this strategy enables joint optimization of 3D Gaussian representation and camera poses, resulting in low-drift tracking and accurate semantic mapping. Our SemGauss-SLAM demonstrates superior performance over existing radiance field-based SLAM methods in terms of mapping and tracking accuracy on Replica and ScanNet datasets, while also showing excellent capabilities in high-precision semantic segmentation and dense semantic mapping.
>
---
#### [replaced 064] Cross-sensor self-supervised training and alignment for remote sensing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.09922v2](http://arxiv.org/pdf/2405.09922v2)**

> **作者:** Valerio Marsocci; Nicolas Audebert
>
> **摘要:** Large-scale ''foundation models'' have gained traction as a way to leverage the vast amounts of unlabeled remote sensing data collected every day. However, due to the multiplicity of Earth Observation satellites, these models should learn ''sensor agnostic'' representations, that generalize across sensor characteristics with minimal fine-tuning. This is complicated by data availability, as low-resolution imagery, such as Sentinel-2 and Landsat-8 data, are available in large amounts, while very high-resolution aerial or satellite data is less common. To tackle these challenges, we introduce cross-sensor self-supervised training and alignment for remote sensing (X-STARS). We design a self-supervised training loss, the Multi-Sensor Alignment Dense loss (MSAD), to align representations across sensors, even with vastly different resolutions. Our X-STARS can be applied to train models from scratch, or to adapt large models pretrained on e.g low-resolution EO data to new high-resolution sensors, in a continual pretraining framework. We collect and release MSC-France, a new multi-sensor dataset, on which we train our X-STARS models, then evaluated on seven downstream classification and segmentation tasks. We demonstrate that X-STARS outperform s the state-of-the-art by a significant margin with less data across various conditions of data availability and resolutions.
>
---
#### [replaced 065] Overlap-Aware Feature Learning for Robust Unsupervised Domain Adaptation for 3D Semantic Segmentation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.01668v3](http://arxiv.org/pdf/2504.01668v3)**

> **作者:** Junjie Chen; Yuecong Xu; Haosheng Li; Kemi Ding
>
> **备注:** This paper has been accepted to the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** 3D point cloud semantic segmentation (PCSS) is a cornerstone for environmental perception in robotic systems and autonomous driving, enabling precise scene understanding through point-wise classification. While unsupervised domain adaptation (UDA) mitigates label scarcity in PCSS, existing methods critically overlook the inherent vulnerability to real-world perturbations (e.g., snow, fog, rain) and adversarial distortions. This work first identifies two intrinsic limitations that undermine current PCSS-UDA robustness: (a) unsupervised features overlap from unaligned boundaries in shared-class regions and (b) feature structure erosion caused by domain-invariant learning that suppresses target-specific patterns. To address the proposed problems, we propose a tripartite framework consisting of: 1) a robustness evaluation model quantifying resilience against adversarial attack/corruption types through robustness metrics; 2) an invertible attention alignment module (IAAM) enabling bidirectional domain mapping while preserving discriminative structure via attention-guided overlap suppression; and 3) a contrastive memory bank with quality-aware contrastive learning that progressively refines pseudo-labels with feature quality for more discriminative representations. Extensive experiments on SynLiDAR-to-SemanticPOSS adaptation demonstrate a maximum mIoU improvement of 14.3\% under adversarial attack.
>
---
#### [replaced 066] IgCONDA-PET: Weakly-Supervised PET Anomaly Detection using Implicitly-Guided Attention-Conditional Counterfactual Diffusion Modeling -- a Multi-Center, Multi-Cancer, and Multi-Tracer Study
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.00239v3](http://arxiv.org/pdf/2405.00239v3)**

> **作者:** Shadab Ahamed; Arman Rahmim
>
> **备注:** 48 pages, 13 figures, 4 tables
>
> **摘要:** Minimizing the need for pixel-level annotated data to train PET lesion detection and segmentation networks is highly desired and can be transformative, given time and cost constraints associated with expert annotations. Current unsupervised or weakly-supervised anomaly detection methods rely on autoencoder or generative adversarial networks (GANs) trained only on healthy data. While these approaches reduce annotation dependency, GAN-based methods are notably more challenging to train than non-GAN alternatives (such as autoencoders) due to issues such as the simultaneous optimization of two competing networks, mode collapse, and training instability. In this paper, we present the weakly-supervised $\textbf{I}$mplicitly-$\textbf{g}$uided $\textbf{CO}$u$\textbf{N}$terfactual diffusion model for $\textbf{D}$etecting $\textbf{A}$nomalies in $\textbf{PET}$ images (IgCONDA-PET). The solution is developed and validated using PET scans from six retrospective cohorts consisting of a total of 2652 cases (multi-cancer, multi-tracer) containing both local and public datasets (spanning multiple centers). The training is conditioned on image class labels (healthy vs. unhealthy) via attention modules, and we employ implicit diffusion guidance. We perform counterfactual generation which facilitates "unhealthy-to-healthy" domain translation by generating a synthetic, healthy version of an unhealthy input image, enabling the detection of anomalies through the calculated differences. The performance of our method was compared against several other deep learning based weakly-supervised or unsupervised methods as well as traditional methods like 41% SUV$_\text{max}$ thresholding. We also highlight the importance of incorporating attention modules in our network for the detection of small anomalies. The code is publicly available at: https://github.com/ahxmeds/IgCONDA-PET.git.
>
---
#### [replaced 067] Super-Resolution with Structured Motion
- **分类: cs.CV; I.4.1; I.4.3**

- **链接: [http://arxiv.org/pdf/2505.15961v2](http://arxiv.org/pdf/2505.15961v2)**

> **作者:** Gabby Litterio; Juan-David Lizarazo-Ferro; Pedro Felzenszwalb; Rashid Zia
>
> **摘要:** We consider the limits of super-resolution using imaging constraints. Due to various theoretical and practical limitations, reconstruction-based methods have been largely restricted to small increases in resolution. In addition, motion-blur is usually seen as a nuisance that impedes super-resolution. We show that by using high-precision motion information, sparse image priors, and convex optimization, it is possible to increase resolution by large factors. A key operation in super-resolution is deconvolution with a box. In general, convolution with a box is not invertible. However, we obtain perfect reconstructions of sparse signals using convex optimization. We also show that motion blur can be helpful for super-resolution. We demonstrate that using pseudo-random motion it is possible to reconstruct a high-resolution target using a single low-resolution image. We present numerical experiments with simulated data and results with real data captured by a camera mounted on a computer controlled stage.
>
---
#### [replaced 068] AI-based Multimodal Biometrics for Detecting Smartphone Distractions: Application to Online Learning
- **分类: cs.CY; cs.AI; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.17364v2](http://arxiv.org/pdf/2506.17364v2)**

> **作者:** Alvaro Becerra; Roberto Daza; Ruth Cobos; Aythami Morales; Mutlu Cukurova; Julian Fierrez
>
> **备注:** Accepted in EC-TEL25: 20th European Conference on Technology Enhanced Learning, Newcastle and Durham, UK, 15-19 September 2025
>
> **摘要:** This work investigates the use of multimodal biometrics to detect distractions caused by smartphone use during tasks that require sustained attention, with a focus on computer-based online learning. Although the methods are applicable to various domains, such as autonomous driving, we concentrate on the challenges learners face in maintaining engagement amid internal (e.g., motivation), system-related (e.g., course design) and contextual (e.g., smartphone use) factors. Traditional learning platforms often lack detailed behavioral data, but Multimodal Learning Analytics (MMLA) and biosensors provide new insights into learner attention. We propose an AI-based approach that leverages physiological signals and head pose data to detect phone use. Our results show that single biometric signals, such as brain waves or heart rate, offer limited accuracy, while head pose alone achieves 87%. A multimodal model combining all signals reaches 91% accuracy, highlighting the benefits of integration. We conclude by discussing the implications and limitations of deploying these models for real-time support in online learning environments.
>
---
