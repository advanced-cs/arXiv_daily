# 计算机视觉 cs.CV

- **最新发布 110 篇**

- **更新 79 篇**

## 最新发布

#### [new 001] MOR-VIT: Efficient Vision Transformer with Mixture-of-Recursions
- **分类: cs.CV**

- **简介: 该论文属于图像识别任务，旨在解决视觉Transformer（ViT）参数冗余和计算成本高的问题。作者提出MoR-ViT，首次引入基于Mixture-of-Recursions的动态递归机制，使每个token自适应处理深度，从而提升效率。实验表明其在精度、参数量和推理速度上均优于现有高效ViT方法。**

- **链接: [http://arxiv.org/pdf/2507.21761v1](http://arxiv.org/pdf/2507.21761v1)**

> **作者:** YiZhou Li
>
> **备注:** 18 pages,9 figuers
>
> **摘要:** Vision Transformers (ViTs) have achieved remarkable success in image recognition, yet standard ViT architectures are hampered by substantial parameter redundancy and high computational cost, limiting their practical deployment. While recent efforts on efficient ViTs primarily focus on static model compression or token-level sparsification, they remain constrained by fixed computational depth for all tokens. In this work, we present MoR-ViT, a novel vision transformer framework that, for the first time, incorporates a token-level dynamic recursion mechanism inspired by the Mixture-of-Recursions (MoR) paradigm. This approach enables each token to adaptively determine its processing depth, yielding a flexible and input-dependent allocation of computational resources. Extensive experiments on ImageNet-1K and transfer benchmarks demonstrate that MoR-ViT not only achieves state-of-the-art accuracy with up to 70% parameter reduction and 2.5x inference acceleration, but also outperforms leading efficient ViT baselines such as DynamicViT and TinyViT under comparable conditions. These results establish dynamic recursion as an effective strategy for efficient vision transformers and open new avenues for scalable and deployable deep learning models in real-world scenarios.
>
---
#### [new 002] VeS: Teaching Pixels to Listen Without Supervision
- **分类: cs.CV; I.2.10**

- **简介: 该论文属于音频-视觉（AV）跨模态检索任务，旨在解决低资源、多语言环境下模型性能下降的问题。作者通过对比三种对比学习目标，发现密集令牌匹配方法在多语言、噪声环境下表现更优，且无需微调视觉主干即可实现精准零样本定位。**

- **链接: [http://arxiv.org/pdf/2507.22008v1](http://arxiv.org/pdf/2507.22008v1)**

> **作者:** Sajay Raj
>
> **备注:** 6 pages, 1 figure, 1 table. Code and models are released
>
> **摘要:** Recent dense audio-visual (AV) models achieve impressive retrieval and emergent localization, but almost all evidence comes from English-centric, caption-rich web video. It is unclear whether these objectives survive in low-resource, code-switched, and noisy multilingual settings that typify developing regions. We show they do**-**and that the choice of aggregation function becomes even more critical. Using a multilingual subset of Project Vaani spanning dozens of Indian languages and dialectal variants, we compare three contrastive objectives: (i) a global mean-pooled loss (CLIP-style), (ii) a dense max-mean token matcher (DenseAV-style), and (iii) a simple hybrid (motivated by frozen-vision alignment strategies). The dense objective delivers a +59% relative R@1 (Audio Visual) improvement over global pooling and substantially lower mean/median ranks, while consistently producing sharp zero-shot localization heatmaps of spoken objects-despite keeping the vision backbone entirely frozen (no LoRA / partial fine-tuning). Our results demonstrate that dense token routing is not a luxury of high-resource English corpora; it is more decisive when annotations and acoustic cleanliness are scarce. We release the codebase and trained models.
>
---
#### [new 003] Ov3R: Open-Vocabulary Semantic 3D Reconstruction from RGB Videos
- **分类: cs.CV**

- **简介: 该论文提出Ov3R框架，用于从RGB视频中进行开放词汇语义3D重建，属于空间AI任务。旨在解决现有方法在几何一致性与语义对齐上的不足。工作包括设计CLIP3R模块实现语义嵌入的3D重建，以及2D-3D OVS模块融合多线索特征提升语义分割性能。**

- **链接: [http://arxiv.org/pdf/2507.22052v1](http://arxiv.org/pdf/2507.22052v1)**

> **作者:** Ziren Gong; Xiaohan Li; Fabio Tosi; Jiawei Han; Stefano Mattoccia; Jianfei Cai; Matteo Poggi
>
> **摘要:** We present Ov3R, a novel framework for open-vocabulary semantic 3D reconstruction from RGB video streams, designed to advance Spatial AI. The system features two key components: CLIP3R, a CLIP-informed 3D reconstruction module that predicts dense point maps from overlapping clips while embedding object-level semantics; and 2D-3D OVS, a 2D-3D open-vocabulary semantic module that lifts 2D features into 3D by learning fused descriptors integrating spatial, geometric, and semantic cues. Unlike prior methods, Ov3R incorporates CLIP semantics directly into the reconstruction process, enabling globally consistent geometry and fine-grained semantic alignment. Our framework achieves state-of-the-art performance in both dense 3D reconstruction and open-vocabulary 3D segmentation, marking a step forward toward real-time, semantics-aware Spatial AI.
>
---
#### [new 004] Evaluating Deep Learning Models for African Wildlife Image Classification: From DenseNet to Vision Transformers
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分类任务，旨在解决非洲野生动物保护中自动识别物种的问题。通过比较DenseNet-201、ResNet-152、EfficientNet-B4和Vision Transformer ViT-H/14的性能，发现DenseNet-201在卷积模型中表现最佳（67%准确率），而ViT-H/14准确率最高（99%），但计算成本高。研究强调准确率与部署可行性的权衡，并展示了DenseNet-201在实际场景中的应用。**

- **链接: [http://arxiv.org/pdf/2507.21364v1](http://arxiv.org/pdf/2507.21364v1)**

> **作者:** Lukman Jibril Aliyu; Umar Sani Muhammad; Bilqisu Ismail; Nasiru Muhammad; Almustapha A Wakili; Seid Muhie Yimam; Shamsuddeen Hassan Muhammad; Mustapha Abdullahi
>
> **备注:** Accepted as a camera-ready paper at Deep Learning Indaba 2025 (Kigali, Rwanda)
>
> **摘要:** Wildlife populations in Africa face severe threats, with vertebrate numbers declining by over 65% in the past five decades. In response, image classification using deep learning has emerged as a promising tool for biodiversity monitoring and conservation. This paper presents a comparative study of deep learning models for automatically classifying African wildlife images, focusing on transfer learning with frozen feature extractors. Using a public dataset of four species: buffalo, elephant, rhinoceros, and zebra; we evaluate the performance of DenseNet-201, ResNet-152, EfficientNet-B4, and Vision Transformer ViT-H/14. DenseNet-201 achieved the best performance among convolutional networks (67% accuracy), while ViT-H/14 achieved the highest overall accuracy (99%), but with significantly higher computational cost, raising deployment concerns. Our experiments highlight the trade-offs between accuracy, resource requirements, and deployability. The best-performing CNN (DenseNet-201) was integrated into a Hugging Face Gradio Space for real-time field use, demonstrating the feasibility of deploying lightweight models in conservation settings. This work contributes to African-grounded AI research by offering practical insights into model selection, dataset preparation, and responsible deployment of deep learning tools for wildlife conservation.
>
---
#### [new 005] Motion Matters: Motion-guided Modulation Network for Skeleton-based Micro-Action Recognition
- **分类: cs.CV**

- **简介: 该论文属于骨架微动作识别任务，旨在解决现有方法难以捕捉微动作中细微运动变化的问题。论文提出了运动引导调制网络（MMN），通过在骨骼级别和帧级别注入运动线索，并采用运动一致性学习策略，提升微动作识别的准确性。**

- **链接: [http://arxiv.org/pdf/2507.21977v1](http://arxiv.org/pdf/2507.21977v1)**

> **作者:** Jihao Gu; Kun Li; Fei Wang; Yanyan Wei; Zhiliang Wu; Hehe Fan; Meng Wang
>
> **摘要:** Micro-Actions (MAs) are an important form of non-verbal communication in social interactions, with potential applications in human emotional analysis. However, existing methods in Micro-Action Recognition often overlook the inherent subtle changes in MAs, which limits the accuracy of distinguishing MAs with subtle changes. To address this issue, we present a novel Motion-guided Modulation Network (MMN) that implicitly captures and modulates subtle motion cues to enhance spatial-temporal representation learning. Specifically, we introduce a Motion-guided Skeletal Modulation module (MSM) to inject motion cues at the skeletal level, acting as a control signal to guide spatial representation modeling. In parallel, we design a Motion-guided Temporal Modulation module (MTM) to incorporate motion information at the frame level, facilitating the modeling of holistic motion patterns in micro-actions. Finally, we propose a motion consistency learning strategy to aggregate the motion cues from multi-scale features for micro-action classification. Experimental results on the Micro-Action 52 and iMiGUE datasets demonstrate that MMN achieves state-of-the-art performance in skeleton-based micro-action recognition, underscoring the importance of explicitly modeling subtle motion cues. The code will be available at https://github.com/momiji-bit/MMN.
>
---
#### [new 006] Aether Weaver: Multimodal Affective Narrative Co-Generation with Dynamic Scene Graphs
- **分类: cs.CV**

- **简介: 该论文提出“Aether Weaver”框架，用于多模态情感叙事协同生成。旨在解决传统串行文本到视觉流程的局限性，实现文本、场景图、视觉场景与情感音景的同步生成。通过整合叙事生成、场景管理、情感控制等模块，提升叙事深度、视觉质量和情感一致性，支持快速创意原型设计与沉浸式叙事体验。**

- **链接: [http://arxiv.org/pdf/2507.21893v1](http://arxiv.org/pdf/2507.21893v1)**

> **作者:** Saeed Ghorbani
>
> **摘要:** We introduce Aether Weaver, a novel, integrated framework for multimodal narrative co-generation that overcomes limitations of sequential text-to-visual pipelines. Our system concurrently synthesizes textual narratives, dynamic scene graph representations, visual scenes, and affective soundscapes, driven by a tightly integrated, co-generation mechanism. At its core, the Narrator, a large language model, generates narrative text and multimodal prompts, while the Director acts as a dynamic scene graph manager, and analyzes the text to build and maintain a structured representation of the story's world, ensuring spatio-temporal and relational consistency for visual rendering and subsequent narrative generation. Additionally, a Narrative Arc Controller guides the high-level story structure, influencing multimodal affective consistency, further complemented by an Affective Tone Mapper that ensures congruent emotional expression across all modalities. Through qualitative evaluations on a diverse set of narrative prompts encompassing various genres, we demonstrate that Aether Weaver significantly enhances narrative depth, visual fidelity, and emotional resonance compared to cascaded baseline approaches. This integrated framework provides a robust platform for rapid creative prototyping and immersive storytelling experiences.
>
---
#### [new 007] HDR Environment Map Estimation with Latent Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在从单视角图像估计高动态范围（HDR）环境图。主要解决ERP表示中的极点失真和边界接缝问题。作者提出了一种基于潜扩散模型（LDM）的新方法，并设计了适用于ERP格式的卷积填充策略和全景适应的Diffusion Transformer架构（PanoDiT），以提升环境图质量和光照准确性。**

- **链接: [http://arxiv.org/pdf/2507.21261v1](http://arxiv.org/pdf/2507.21261v1)**

> **作者:** Jack Hilliard; Adrian Hilton; Jean-Yves Guillemaut
>
> **摘要:** We advance the field of HDR environment map estimation from a single-view image by establishing a novel approach leveraging the Latent Diffusion Model (LDM) to produce high-quality environment maps that can plausibly light mirror-reflective surfaces. A common issue when using the ERP representation, the format used by the vast majority of approaches, is distortions at the poles and a seam at the sides of the environment map. We remove the border seam artefact by proposing an ERP convolutional padding in the latent autoencoder. Additionally, we investigate whether adapting the diffusion network architecture to the ERP format can improve the quality and accuracy of the estimated environment map by proposing a panoramically-adapted Diffusion Transformer architecture. Our proposed PanoDiT network reduces ERP distortions and artefacts, but at the cost of image quality and plausibility. We evaluate with standard benchmarks to demonstrate that our models estimate high-quality environment maps that perform competitively with state-of-the-art approaches in both image quality and lighting accuracy.
>
---
#### [new 008] Automated Detection of Antarctic Benthic Organisms in High-Resolution In Situ Imagery to Aid Biodiversity Monitoring
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉与生态监测任务，旨在解决南极海底生物多样性监测中人工标注图像效率低的问题。作者提出了一个目标检测框架，并发布了首个用于南极威德尔海生物监测的公开数据集，结合多种技术提升检测效果，尤其在中大型生物分类上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.21665v1](http://arxiv.org/pdf/2507.21665v1)**

> **作者:** Cameron Trotter; Huw Griffiths; Tasnuva Ming Khan; Rowan Whittle
>
> **备注:** Accepted to ICCV 2025's Joint Workshop on Marine Vision (ICCVW, CVAUI&AAMVEM). Main paper (11 pages, 3 figures, 3 tables) plus supplementary (7 pages, 5 figures, 2 tables)
>
> **摘要:** Monitoring benthic biodiversity in Antarctica is vital for understanding ecological change in response to climate-driven pressures. This work is typically performed using high-resolution imagery captured in situ, though manual annotation of such data remains laborious and specialised, impeding large-scale analysis. We present a tailored object detection framework for identifying and classifying Antarctic benthic organisms in high-resolution towed camera imagery, alongside the first public computer vision dataset for benthic biodiversity monitoring in the Weddell Sea. Our approach addresses key challenges associated with marine ecological imagery, including limited annotated data, variable object sizes, and complex seafloor structure. The proposed framework combines resolution-preserving patching, spatial data augmentation, fine-tuning, and postprocessing via Slicing Aided Hyper Inference. We benchmark multiple object detection architectures and demonstrate strong performance in detecting medium and large organisms across 25 fine-grained morphotypes, significantly more than other works in this area. Detection of small and rare taxa remains a challenge, reflecting limitations in current detection architectures. Our framework provides a scalable foundation for future machine-assisted in situ benthic biodiversity monitoring research.
>
---
#### [new 009] EMIT: Enhancing MLLMs for Industrial Anomaly Detection via Difficulty-Aware GRPO
- **分类: cs.CV**

- **简介: 该论文属于工业异常检测任务，旨在解决多模态大语言模型（MLLM）在该领域表现不佳的问题。作者提出EMIT框架，结合多任务数据集、GPT生成文本描述和难度感知的GRPO优化方法，提升MLLM在少量样本下的检测性能，实验表明效果显著。**

- **链接: [http://arxiv.org/pdf/2507.21619v1](http://arxiv.org/pdf/2507.21619v1)**

> **作者:** Wei Guan; Jun Lan; Jian Cao; Hao Tan; Huijia Zhu; Weiqiang Wang
>
> **摘要:** Industrial anomaly detection (IAD) plays a crucial role in maintaining the safety and reliability of manufacturing systems. While multimodal large language models (MLLMs) show strong vision-language reasoning abilities, their effectiveness in IAD remains limited without domain-specific adaptation. In this work, we propose EMIT, a unified framework that enhances MLLMs for IAD via difficulty-aware group relative policy optimization (GRPO). EMIT constructs a multi-task IAD dataset and utilizes GPT-generated object text descriptions to compensate for missing defective images. For few-shot anomaly detection, it integrates a soft prompt and heatmap-guided contrastive embeddings derived from patch-level comparisons. To better handle difficult data samples, i.e., cases where the MLLM struggles to generate correct answers, we propose a difficulty-aware GRPO that extends the original GRPO by incorporating a response resampling strategy to ensure the inclusion of correct answers in the sampled responses, as well as an advantage reweighting mechanism to strengthen learning from such difficult data samples. Extensive experiments on the MMAD benchmark demonstrate that EMIT significantly enhances the IAD performance of MLLMs, achieving an average improvement of 7.77\% over the base model (InternVL3-8B) across seven tasks.
>
---
#### [new 010] Sun sensor calibration algorithms: A systematic mapping and survey
- **分类: cs.CV; astro-ph.IM**

- **简介: 该论文属于系统综述任务，旨在解决太阳敏感器校准算法缺乏系统性总结的问题。论文系统梳理了太阳敏感器建模与校准算法的研究进展，分析了现有方法的不足，并提出了未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.21541v1](http://arxiv.org/pdf/2507.21541v1)**

> **作者:** Michael Herman; Olivia J. Pinon Fischer; Dimitri N. Mavris
>
> **备注:** Submitted to Acta Astronautica
>
> **摘要:** Attitude sensors determine the spacecraft attitude through the sensing of an astronomical object, field or other phenomena. The Sun and fixed stars are the two primary astronomical sensing objects. Attitude sensors are critical components for the survival and knowledge improvement of spacecraft. Of these, sun sensors are the most common and important sensor for spacecraft attitude determination. The sun sensor measures the Sun vector in spacecraft coordinates. The sun sensor calibration process is particularly difficult due to the complex nature of the uncertainties involved. The uncertainties are small, difficult to observe, and vary spatio-temporally over the lifecycle of the sensor. In addition, the sensors are affected by numerous sources of uncertainties, including manufacturing, electrical, environmental, and interference sources. This motivates the development of advanced calibration algorithms to minimize uncertainty over the sensor lifecycle and improve accuracy. Although modeling and calibration techniques for sun sensors have been explored extensively in the literature over the past two decades, there is currently no resource that consolidates and systematically reviews this body of work. The present review proposes a systematic mapping of sun sensor modeling and calibration algorithms across a breadth of sensor configurations. It specifically provides a comprehensive survey of each methodology, along with an analysis of research gaps and recommendations for future directions in sun sensor modeling and calibration techniques.
>
---
#### [new 011] MAGE: Multimodal Alignment and Generation Enhancement via Bridging Visual and Semantic Spaces
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于多模态学习任务，旨在解决视觉与语义空间对齐问题。通过提出MAGE框架及智能对齐网络（IAN），结合交叉熵与均方误差训练策略，实现视觉与文本语义的高效对齐，并提升模型“Any-to-Any”生成能力，在多个基准测试中表现出色。**

- **链接: [http://arxiv.org/pdf/2507.21741v1](http://arxiv.org/pdf/2507.21741v1)**

> **作者:** Shaojun E; Yuchen Yang; Jiaheng Wu; Yan Zhang; Tiejun Zhao; Ziyan Chen
>
> **备注:** 9 pages
>
> **摘要:** In the latest advancements in multimodal learning, effectively addressing the spatial and semantic losses of visual data after encoding remains a critical challenge. This is because the performance of large multimodal models is positively correlated with the coupling between visual encoders and large language models. Existing approaches often face issues such as vector gaps or semantic disparities, resulting in information loss during the propagation process. To address these issues, we propose MAGE (Multimodal Alignment and Generation Enhancement), a novel framework that bridges the semantic spaces of vision and text through an innovative alignment mechanism. By introducing the Intelligent Alignment Network (IAN), MAGE achieves dimensional and semantic alignment. To reduce the gap between synonymous heterogeneous data, we employ a training strategy that combines cross-entropy and mean squared error, significantly enhancing the alignment effect. Moreover, to enhance MAGE's "Any-to-Any" capability, we developed a fine-tuning dataset for multimodal tool-calling instructions to expand the model's output capability boundaries. Finally, our proposed multimodal large model architecture, MAGE, achieved significantly better performance compared to similar works across various evaluation benchmarks, including MME, MMBench, and SEED. Complete code and appendix are available at: https://github.com/GTCOM-NLP/MAGE.
>
---
#### [new 012] Group Relative Augmentation for Data Efficient Action Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于动作检测任务，旨在解决使用少量样本适配大型视频-语言模型（VLMs）时的过拟合和粒度不匹配问题。论文提出结合参数高效微调（LoRA）与可学习特征增强，并通过组加权损失函数提升模型鲁棒性，实现在AVA和MOMA数据集上的高效动作检测。**

- **链接: [http://arxiv.org/pdf/2507.21353v1](http://arxiv.org/pdf/2507.21353v1)**

> **作者:** Deep Anil Patel; Iain Melvin; Zachary Izzo; Martin Renqiang Min
>
> **摘要:** Adapting large Video-Language Models (VLMs) for action detection using only a few examples poses challenges like overfitting and the granularity mismatch between scene-level pre-training and required person-centric understanding. We propose an efficient adaptation strategy combining parameter-efficient tuning (LoRA) with a novel learnable internal feature augmentation. Applied within the frozen VLM backbone using FiLM, these augmentations generate diverse feature variations directly relevant to the task. Additionally, we introduce a group-weighted loss function that dynamically modulates the training contribution of each augmented sample based on its prediction divergence relative to the group average. This promotes robust learning by prioritizing informative yet reasonable augmentations. We demonstrate our method's effectiveness on complex multi-label, multi-person action detection datasets (AVA, MOMA), achieving strong mAP performance and showcasing significant data efficiency for adapting VLMs from limited examples.
>
---
#### [new 013] On Explaining Visual Captioning with Hybrid Markov Logic Networks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像描述生成任务，旨在解决深度神经网络在生成图像标题时缺乏可解释性的问题。作者提出了一种基于混合马尔可夫逻辑网络的解释框架，通过分析训练数据分布的变化，量化哪些训练样例对生成标题有重要影响，从而提升模型的可解释性。**

- **链接: [http://arxiv.org/pdf/2507.21246v1](http://arxiv.org/pdf/2507.21246v1)**

> **作者:** Monika Shah; Somdeb Sarkhel; Deepak Venugopal
>
> **摘要:** Deep Neural Networks (DNNs) have made tremendous progress in multimodal tasks such as image captioning. However, explaining/interpreting how these models integrate visual information, language information and knowledge representation to generate meaningful captions remains a challenging problem. Standard metrics to measure performance typically rely on comparing generated captions with human-written ones that may not provide a user with a deep insights into this integration. In this work, we develop a novel explanation framework that is easily interpretable based on Hybrid Markov Logic Networks (HMLNs) - a language that can combine symbolic rules with real-valued functions - where we hypothesize how relevant examples from the training data could have influenced the generation of the observed caption. To do this, we learn a HMLN distribution over the training instances and infer the shift in distributions over these instances when we condition on the generated sample which allows us to quantify which examples may have been a source of richer information to generate the observed caption. Our experiments on captions generated for several state-of-the-art captioning models using Amazon Mechanical Turk illustrate the interpretability of our explanations, and allow us to compare these models along the dimension of explainability.
>
---
#### [new 014] Describe, Adapt and Combine: Empowering CLIP Encoders for Open-set 3D Object Retrieval
- **分类: cs.CV**

- **简介: 该论文属于开放集3D物体检索任务，旨在提升模型对训练中未见类别的泛化能力。现有方法受限于3D数据不足，难以学习通用表示。论文提出DAC框架，利用CLIP和多模态大语言模型（MLLM）协同学习，并引入AB-LoRA缓解过拟合，显著提升检索性能。**

- **链接: [http://arxiv.org/pdf/2507.21489v1](http://arxiv.org/pdf/2507.21489v1)**

> **作者:** Zhichuan Wang; Yang Zhou; Zhe Liu; Rui Yu; Song Bai; Yulong Wang; Xinwei He; Xiang Bai
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Open-set 3D object retrieval (3DOR) is an emerging task aiming to retrieve 3D objects of unseen categories beyond the training set. Existing methods typically utilize all modalities (i.e., voxels, point clouds, multi-view images) and train specific backbones before fusion. However, they still struggle to produce generalized representations due to insufficient 3D training data. Being contrastively pre-trained on web-scale image-text pairs, CLIP inherently produces generalized representations for a wide range of downstream tasks. Building upon it, we present a simple yet effective framework named Describe, Adapt and Combine (DAC) by taking only multi-view images for open-set 3DOR. DAC innovatively synergizes a CLIP model with a multi-modal large language model (MLLM) to learn generalized 3D representations, where the MLLM is used for dual purposes. First, it describes the seen category information to align with CLIP's training objective for adaptation during training. Second, it provides external hints about unknown objects complementary to visual cues during inference. To improve the synergy, we introduce an Additive-Bias Low-Rank adaptation (AB-LoRA), which alleviates overfitting and further enhances the generalization to unseen categories. With only multi-view images, DAC significantly surpasses prior arts by an average of +10.01\% mAP on four open-set 3DOR datasets. Moreover, its generalization is also validated on image-based and cross-dataset setups. Code is available at https://github.com/wangzhichuan123/DAC.
>
---
#### [new 015] Attention-Driven Multimodal Alignment for Long-term Action Quality Assessment
- **分类: cs.CV**

- **简介: 该论文属于长期动作质量评估任务，旨在解决现有方法在多模态融合与时间动态建模上的不足。作者提出LMAC-Net，通过多模态注意力一致性机制对齐视觉与音频特征，并设计局部查询编码器和双层评分策略，提升长视频中动作质量评估的准确性。**

- **链接: [http://arxiv.org/pdf/2507.21945v1](http://arxiv.org/pdf/2507.21945v1)**

> **作者:** Xin Wang; Peng-Jie Li; Yuan-Yuan Shen
>
> **备注:** Accepted to Applied Soft Computing
>
> **摘要:** Long-term action quality assessment (AQA) focuses on evaluating the quality of human activities in videos lasting up to several minutes. This task plays an important role in the automated evaluation of artistic sports such as rhythmic gymnastics and figure skating, where both accurate motion execution and temporal synchronization with background music are essential for performance assessment. However, existing methods predominantly fall into two categories: unimodal approaches that rely solely on visual features, which are inadequate for modeling multimodal cues like music; and multimodal approaches that typically employ simple feature-level contrastive fusion, overlooking deep cross-modal collaboration and temporal dynamics. As a result, they struggle to capture complex interactions between modalities and fail to accurately track critical performance changes throughout extended sequences. To address these challenges, we propose the Long-term Multimodal Attention Consistency Network (LMAC-Net). LMAC-Net introduces a multimodal attention consistency mechanism to explicitly align multimodal features, enabling stable integration of visual and audio information and enhancing feature representations. Specifically, we introduce a multimodal local query encoder module to capture temporal semantics and cross-modal relations, and use a two-level score evaluation for interpretable results. In addition, attention-based and regression-based losses are applied to jointly optimize multimodal alignment and score fusion. Experiments conducted on the RG and Fis-V datasets demonstrate that LMAC-Net significantly outperforms existing methods, validating the effectiveness of our proposed approach.
>
---
#### [new 016] Recursive Visual Imagination and Adaptive Linguistic Grounding for Vision Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航（VLN）任务，旨在解决导航代理在理解复杂场景与语言指令时易受细节干扰、导致行为偏差的问题。作者提出递归视觉想象（RVI）和自适应语言接地（ALG）技术，通过递归总结视觉感知、增强语言对齐，提升导航决策准确性。实验表明该方法在VLN-CE和ObjectNav任务上优于现有技术。**

- **链接: [http://arxiv.org/pdf/2507.21450v1](http://arxiv.org/pdf/2507.21450v1)**

> **作者:** Bolei Chen; Jiaxu Kang; Yifei Wang; Ping Zhong; Qi Wu; Jianxin Wang
>
> **备注:** Submitted to AAAI 2026
>
> **摘要:** Vision Language Navigation (VLN) typically requires agents to navigate to specified objects or remote regions in unknown scenes by obeying linguistic commands. Such tasks require organizing historical visual observations for linguistic grounding, which is critical for long-sequence navigational decisions. However, current agents suffer from overly detailed scene representation and ambiguous vision-language alignment, which weaken their comprehension of navigation-friendly high-level scene priors and easily lead to behaviors that violate linguistic commands. To tackle these issues, we propose a navigation policy by recursively summarizing along-the-way visual perceptions, which are adaptively aligned with commands to enhance linguistic grounding. In particular, by structurally modeling historical trajectories as compact neural grids, several Recursive Visual Imagination (RVI) techniques are proposed to motivate agents to focus on the regularity of visual transitions and semantic scene layouts, instead of dealing with misleading geometric details. Then, an Adaptive Linguistic Grounding (ALG) technique is proposed to align the learned situational memories with different linguistic components purposefully. Such fine-grained semantic matching facilitates the accurate anticipation of navigation actions and progress. Our navigation policy outperforms the state-of-the-art methods on the challenging VLN-CE and ObjectNav tasks, showing the superiority of our RVI and ALG techniques for VLN.
>
---
#### [new 017] ChartM$^3$: Benchmarking Chart Editing with Multimodal Instructions
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态图表编辑任务，旨在解决自然语言指令模糊导致的细粒度图表编辑难题。论文提出ChartM³基准，包含1,000个样本及多级复杂度与多视角评估，并构建了含24,000样本的训练集，显著提升多模态大模型编辑能力。**

- **链接: [http://arxiv.org/pdf/2507.21167v1](http://arxiv.org/pdf/2507.21167v1)**

> **作者:** Danglu Yang; Liang Zhang; Zihao Yue; Liangyu Chen; Yichen Xu; Wenxuan Wang; Qin Jin
>
> **摘要:** Charts are a fundamental visualization format widely used in data analysis across research and industry. While enabling users to edit charts based on high-level intentions is of great practical value, existing methods primarily rely on natural language instructions, which are often too ambiguous to support fine-grained editing. In this work, we introduce a novel paradigm for multimodal chart editing, where user intent is expressed through a combination of natural language and visual indicators that explicitly highlight the elements to be modified. To support this paradigm, we present Chart$\text{M}^3$, a new benchmark for Multimodal chart editing with Multi-level complexity and Multi-perspective evaluation. Chart$\text{M}^3$ contains 1,000 samples spanning four levels of editing difficulty. Each sample includes triplets in the form of (chart, code, multimodal instructions). To comprehensively evaluate chart editing models, Chart$\text{M}^3$ provides metrics that assess both visual appearance and code correctness. Our benchmark reveals significant limitations in current multimodal large language models (MLLMs), including GPT-4o, particularly in their ability to interpret and act on visual indicators. To address this, we construct Chart$\text{M}^3$-Train, a large-scale training set with 24,000 multimodal chart editing samples. Fine-tuning MLLMs on this dataset leads to substantial improvements, demonstrating the importance of multimodal supervision in building practical chart editing systems. Our datasets, codes, and evaluation tools are available at https://github.com/MLrollIT/ChartM3. %https://github.com/MLrollIT/ChartM3Our datasets, codes, and evaluation tools are available at https://github.com/yaolinli/VCE.
>
---
#### [new 018] GLCP: Global-to-Local Connectivity Preservation for Tubular Structure Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决管状结构（如血管网络）分割中的结构碎片化问题。现有方法忽略局部不连续区域，导致效果不佳。为此，论文提出GLCP框架，包含IMS模块同时学习全局分割、骨架图和局部不连续图，以及DAR模块优化分割结果，从而提升管状结构的分割准确性与连续性。**

- **链接: [http://arxiv.org/pdf/2507.21328v1](http://arxiv.org/pdf/2507.21328v1)**

> **作者:** Feixiang Zhou; Zhuangzhi Gao; He Zhao; Jianyang Xie; Yanda Meng; Yitian Zhao; Gregory Y. H. Lip; Yalin Zheng
>
> **备注:** MICCAI 2025 (Oral)
>
> **摘要:** Accurate segmentation of tubular structures, such as vascular networks, plays a critical role in various medical domains. A remaining significant challenge in this task is structural fragmentation, which can adversely impact downstream applications. Existing methods primarily focus on designing various loss functions to constrain global topological structures. However, they often overlook local discontinuity regions, leading to suboptimal segmentation results. To overcome this limitation, we propose a novel Global-to-Local Connectivity Preservation (GLCP) framework that can simultaneously perceive global and local structural characteristics of tubular networks. Specifically, we propose an Interactive Multi-head Segmentation (IMS) module to jointly learn global segmentation, skeleton maps, and local discontinuity maps, respectively. This enables our model to explicitly target local discontinuity regions while maintaining global topological integrity. In addition, we design a lightweight Dual-Attention-based Refinement (DAR) module to further improve segmentation quality by refining the resulting segmentation maps. Extensive experiments on both 2D and 3D datasets demonstrate that our GLCP achieves superior accuracy and continuity in tubular structure segmentation compared to several state-of-the-art approaches. The source codes will be available at https://github.com/FeixiangZhou/GLCP.
>
---
#### [new 019] MOVE: Motion-Guided Few-Shot Video Object Segmentation
- **分类: cs.CV**

- **简介: 该论文属于视频目标分割任务，旨在解决基于少量标注示例和运动模式的视频动态目标分割问题。现有方法忽略视频中的丰富动态信息，限制了其在运动理解场景中的应用。为此，论文提出了MOVE数据集和一种新方法DMA，以推动运动引导的少样本视频目标分割研究。**

- **链接: [http://arxiv.org/pdf/2507.22061v1](http://arxiv.org/pdf/2507.22061v1)**

> **作者:** Kaining Ying; Hengrui Hu; Henghui Ding
>
> **备注:** ICCV 2025, Project Page: https://henghuiding.com/MOVE/
>
> **摘要:** This work addresses motion-guided few-shot video object segmentation (FSVOS), which aims to segment dynamic objects in videos based on a few annotated examples with the same motion patterns. Existing FSVOS datasets and methods typically focus on object categories, which are static attributes that ignore the rich temporal dynamics in videos, limiting their application in scenarios requiring motion understanding. To fill this gap, we introduce MOVE, a large-scale dataset specifically designed for motion-guided FSVOS. Based on MOVE, we comprehensively evaluate 6 state-of-the-art methods from 3 different related tasks across 2 experimental settings. Our results reveal that current methods struggle to address motion-guided FSVOS, prompting us to analyze the associated challenges and propose a baseline method, Decoupled Motion Appearance Network (DMA). Experiments demonstrate that our approach achieves superior performance in few shot motion understanding, establishing a solid foundation for future research in this direction.
>
---
#### [new 020] Low-Cost Test-Time Adaptation for Robust Video Editing
- **分类: cs.CV**

- **简介: 该论文属于视频编辑任务，旨在解决视频编辑中时间不一致和提示过拟合问题。提出了Vid-TTA方法，通过自监督辅助任务实现测试时轻量级个性化优化，提升视频编辑质量与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.21858v1](http://arxiv.org/pdf/2507.21858v1)**

> **作者:** Jianhui Wang; Yinda Chen; Yangfan He; Xinyuan Song; Yi Xin; Dapeng Zhang; Zhongwei Wan; Bin Li; Rongchao Zhang
>
> **摘要:** Video editing is a critical component of content creation that transforms raw footage into coherent works aligned with specific visual and narrative objectives. Existing approaches face two major challenges: temporal inconsistencies due to failure in capturing complex motion patterns, and overfitting to simple prompts arising from limitations in UNet backbone architectures. While learning-based methods can enhance editing quality, they typically demand substantial computational resources and are constrained by the scarcity of high-quality annotated data. In this paper, we present Vid-TTA, a lightweight test-time adaptation framework that personalizes optimization for each test video during inference through self-supervised auxiliary tasks. Our approach incorporates a motion-aware frame reconstruction mechanism that identifies and preserves crucial movement regions, alongside a prompt perturbation and reconstruction strategy that strengthens model robustness to diverse textual descriptions. These innovations are orchestrated by a meta-learning driven dynamic loss balancing mechanism that adaptively adjusts the optimization process based on video characteristics. Extensive experiments demonstrate that Vid-TTA significantly improves video temporal consistency and mitigates prompt overfitting while maintaining low computational overhead, offering a plug-and-play performance boost for existing video editing models.
>
---
#### [new 021] Impact of Underwater Image Enhancement on Feature Matching
- **分类: cs.CV**

- **简介: 该论文属于图像处理与计算机视觉任务，旨在解决水下图像因光线吸收、散射等因素导致的视觉退化问题。作者提出了新的评估指标和框架，分析增强技术对特征匹配的影响，并验证其在SLAM算法中的实际效果。**

- **链接: [http://arxiv.org/pdf/2507.21715v1](http://arxiv.org/pdf/2507.21715v1)**

> **作者:** Jason M. Summers; Mark W. Jones
>
> **摘要:** We introduce local matching stability and furthest matchable frame as quantitative measures for evaluating the success of underwater image enhancement. This enhancement process addresses visual degradation caused by light absorption, scattering, marine growth, and debris. Enhanced imagery plays a critical role in downstream tasks such as path detection and autonomous navigation for underwater vehicles, relying on robust feature extraction and frame matching. To assess the impact of enhancement techniques on frame-matching performance, we propose a novel evaluation framework tailored to underwater environments. Through metric-based analysis, we identify strengths and limitations of existing approaches and pinpoint gaps in their assessment of real-world applicability. By incorporating a practical matching strategy, our framework offers a robust, context-aware benchmark for comparing enhancement methods. Finally, we demonstrate how visual improvements affect the performance of a complete real-world algorithm -- Simultaneous Localization and Mapping (SLAM) -- reinforcing the framework's relevance to operational underwater scenarios.
>
---
#### [new 022] Emerging Trends in Pseudo-Label Refinement for Weakly Supervised Semantic Segmentation with Image-Level Supervision
- **分类: cs.CV**

- **简介: 该论文属于弱监督语义分割任务，旨在利用图像级标注生成像素级预测。论文重点综述了伪标签优化的最新方法，分析了现有技术的挑战与局限，并探讨了未来研究方向。工作包括分类总结最新进展、评估方法在特定领域数据上的应用，并指出研究空白。**

- **链接: [http://arxiv.org/pdf/2507.21587v1](http://arxiv.org/pdf/2507.21587v1)**

> **作者:** Zheyuan Zhang; Wang Zhang
>
> **摘要:** Unlike fully supervised semantic segmentation, weakly supervised semantic segmentation (WSSS) relies on weaker forms of supervision to perform dense prediction tasks. Among the various types of weak supervision, WSSS with image level annotations is considered both the most challenging and the most practical, attracting significant research attention. Therefore, in this review, we focus on WSSS with image level annotations. Additionally, this review concentrates on mainstream research directions, deliberately omitting less influential branches. Given the rapid development of new methods and the limitations of existing surveys in capturing recent trends, there is a pressing need for an updated and comprehensive review. Our goal is to fill this gap by synthesizing the latest advancements and state-of-the-art techniques in WSSS with image level labels. Basically, we provide a comprehensive review of recent advancements in WSSS with image level labels, categorizing existing methods based on the types and levels of additional supervision involved. We also examine the challenges of applying advanced methods to domain specific datasets in WSSS,a topic that remains underexplored. Finally, we discuss the current challenges, evaluate the limitations of existing approaches, and outline several promising directions for future research. This review is intended for researchers who are already familiar with the fundamental concepts of WSSS and are seeking to deepen their understanding of current advances and methodological innovations.
>
---
#### [new 023] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于文本到图像生成评估任务，旨在解决现有方法依赖人工标注、训练成本高的问题。作者提出LLaVA-Reward，利用多模态大模型的隐藏状态自动评估生成质量，并引入SkipCA模块增强图文交互。模型支持多种偏好数据训练，在多个评估维度上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.21391v1](http://arxiv.org/pdf/2507.21391v1)**

> **作者:** Shijie Zhou; Ruiyi Zhang; Huaisheng Zhu; Branislav Kveton; Yufan Zhou; Jiuxiang Gu; Jian Chen; Changyou Chen
>
> **备注:** Accepted at ICCV 2025. Code available at https://github.com/sjz5202/LLaVA-Reward
>
> **摘要:** We introduce LLaVA-Reward, an efficient reward model designed to automatically evaluate text-to-image (T2I) generations across multiple perspectives, leveraging pretrained multimodal large language models (MLLMs). Existing MLLM-based approaches require instruction-following data for supervised fine-tuning and evaluate generation quality on analyzing text response, which is time-consuming and difficult to train. To address this problem, we propose LLaVA-Reward, which directly utilizes the hidden states of MLLMs given text-image pairs. To enhance the bidirectional interaction between visual and textual representations in decoder-only MLLMs, we further propose adding a Skip-connection Cross Attention (SkipCA) module. This design enhances text-image correlation reasoning by connecting early-layer visual features with later-layer hidden representations.In addition, LLaVA-Reward supports different types of preference data for efficient fine-tuning, including paired preference data and unpaired data. We train LLaVA-Reward on four evaluation perspectives: text-image alignment, fidelity/artifact, safety, and overall ranking. Empirical results demonstrate that LLaVA-Reward outperforms conventional and MLLM-based methods in generating human-aligned scores for automatic evaluations and inference-time scaling in text-to-image generations.
>
---
#### [new 024] ReGATE: Learning Faster and Better with Fewer Tokens in MLLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态大语言模型（MLLM）训练效率优化任务，旨在解决训练过程中计算成本过高的问题。作者提出ReGATE方法，通过参考模型指导的自适应令牌剪枝，减少前向传播中的冗余计算。实验表明该方法在多个多模态基准上提升了训练速度和性能，同时减少了令牌数量。**

- **链接: [http://arxiv.org/pdf/2507.21420v1](http://arxiv.org/pdf/2507.21420v1)**

> **作者:** Chaoyu Li; Yogesh Kulkarni; Pooyan Fazli
>
> **摘要:** The computational cost of training multimodal large language models (MLLMs) rapidly increases with the number of tokens involved. Existing efficiency methods primarily target inference and rely on token reduction or merging, offering limited benefit during training. In this paper, we propose ReGATE (Reference$-$Guided Adaptive Token Elision), an adaptive token pruning method for accelerating MLLM training. Specifically, ReGATE adopts a teacher-student framework in which the MLLM being trained serves as the student, and a frozen reference large language model (LLM) acts as the teacher. The teacher computes per-token reference losses, which are combined with an exponential moving average (EMA) of the student's own difficulty scores. This adaptive difficulty-based scoring enables the selective processing of crucial tokens while bypassing less informative ones in the forward pass, significantly reducing computational overhead. Experiments demonstrate that ReGATE, when applied to VideoLLaMA2, matches the peak accuracy of standard training on MVBench up to 2$\times$ faster, using only 35% of the tokens. With additional training, it even surpasses the baseline on several multimodal benchmarks, all while reducing the total token count by over 41%. Code and models will be released soon.
>
---
#### [new 025] AU-LLM: Micro-Expression Action Unit Detection via Enhanced LLM-Based Feature Fusion
- **分类: cs.CV**

- **简介: 该论文属于微表情动作单元检测任务，旨在解决微表情中低强度、细粒度动作单元识别困难的问题。作者提出AU-LLM框架，首次引入大语言模型进行微表情分析，通过增强融合投影器（EFP）融合多级视觉特征，提升了检测精度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.21778v1](http://arxiv.org/pdf/2507.21778v1)**

> **作者:** Zhishu Liu; Kaishen Yuan; Bo Zhao; Yong Xu; Zitong Yu
>
> **摘要:** The detection of micro-expression Action Units (AUs) is a formidable challenge in affective computing, pivotal for decoding subtle, involuntary human emotions. While Large Language Models (LLMs) demonstrate profound reasoning abilities, their application to the fine-grained, low-intensity domain of micro-expression AU detection remains unexplored. This paper pioneers this direction by introducing \textbf{AU-LLM}, a novel framework that for the first time uses LLM to detect AUs in micro-expression datasets with subtle intensities and the scarcity of data. We specifically address the critical vision-language semantic gap, the \textbf{Enhanced Fusion Projector (EFP)}. The EFP employs a Multi-Layer Perceptron (MLP) to intelligently fuse mid-level (local texture) and high-level (global semantics) visual features from a specialized 3D-CNN backbone into a single, information-dense token. This compact representation effectively empowers the LLM to perform nuanced reasoning over subtle facial muscle movements.Through extensive evaluations on the benchmark CASME II and SAMM datasets, including stringent Leave-One-Subject-Out (LOSO) and cross-domain protocols, AU-LLM establishes a new state-of-the-art, validating the significant potential and robustness of LLM-based reasoning for micro-expression analysis. The codes are available at https://github.com/ZS-liu-JLU/AU-LLMs.
>
---
#### [new 026] Staining and locking computer vision models without retraining
- **分类: cs.CV; cs.AI; cs.LG; 68T07, 68T45, 68W40; I.2.10; F.2.0; K.5.1; K.6.5**

- **简介: 该论文属于计算机视觉模型安全任务，旨在保护模型知识产权。论文提出无需重新训练即可对预训练模型进行“染色”（水印）和“锁定”的方法，通过修改少量权重嵌入秘密行为，并在输入图像中加入触发补丁以解锁，确保模型性能不受影响，同时提供可证明的安全保证。**

- **链接: [http://arxiv.org/pdf/2507.22000v1](http://arxiv.org/pdf/2507.22000v1)**

> **作者:** Oliver J. Sutton; Qinghua Zhou; George Leete; Alexander N. Gorban; Ivan Y. Tyukin
>
> **备注:** 10 pages, 9 pages of appendices, 10 figures
>
> **摘要:** We introduce new methods of staining and locking computer vision models, to protect their owners' intellectual property. Staining, also known as watermarking, embeds secret behaviour into a model which can later be used to identify it, while locking aims to make a model unusable unless a secret trigger is inserted into input images. Unlike existing methods, our algorithms can be used to stain and lock pre-trained models without requiring fine-tuning or retraining, and come with provable, computable guarantees bounding their worst-case false positive rates. The stain and lock are implemented by directly modifying a small number of the model's weights and have minimal impact on the (unlocked) model's performance. Locked models are unlocked by inserting a small `trigger patch' into the corner of the input image. We present experimental results showing the efficacy of our methods and demonstrating their practical performance on a variety of computer vision models.
>
---
#### [new 027] Enhancing Generalization in Data-free Quantization via Mixup-class Prompting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于数据无关量化（DFQ）任务，旨在解决量化模型泛化能力不足的问题。通过提出“mixup-class prompt”策略，在文本提示层面融合多类别标签生成多样化合成数据，提升量化模型的泛化性和优化稳定性，实验证明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.21947v1](http://arxiv.org/pdf/2507.21947v1)**

> **作者:** Jiwoong Park; Chaeun Lee; Yongseok Choi; Sein Park; Deokki Hong; Jungwook Choi
>
> **摘要:** Post-training quantization (PTQ) improves efficiency but struggles with limited calibration data, especially under privacy constraints. Data-free quantization (DFQ) mitigates this by generating synthetic images using generative models such as generative adversarial networks (GANs) and text-conditioned latent diffusion models (LDMs), while applying existing PTQ algorithms. However, the relationship between generated synthetic images and the generalizability of the quantized model during PTQ remains underexplored. Without investigating this relationship, synthetic images generated by previous prompt engineering methods based on single-class prompts suffer from issues such as polysemy, leading to performance degradation. We propose \textbf{mixup-class prompt}, a mixup-based text prompting strategy that fuses multiple class labels at the text prompt level to generate diverse, robust synthetic data. This approach enhances generalization, and improves optimization stability in PTQ. We provide quantitative insights through gradient norm and generalization error analysis. Experiments on convolutional neural networks (CNNs) and vision transformers (ViTs) show that our method consistently outperforms state-of-the-art DFQ methods like GenQ. Furthermore, it pushes the performance boundary in extremely low-bit scenarios, achieving new state-of-the-art accuracy in challenging 2-bit weight, 4-bit activation (W2A4) quantization.
>
---
#### [new 028] PanoGAN A Deep Generative Model for Panoramic Dental Radiographs
- **分类: cs.CV; cs.ET; cs.LG; eess.IV**

- **简介: 该论文属于医学图像生成任务，旨在解决牙科数据稀缺问题。作者使用深度卷积GAN（DCGAN）结合Wasserstein损失和梯度惩罚，训练生成全景牙科X光片。通过预处理标准化输入并保持解剖多样性，探索四种模型变体。临床专家评估结果显示生成图像具有中等解剖可见性，部分存在伪影，但模型在细节与清晰度上各有优势，为牙科图像生成提供了初步基础。**

- **链接: [http://arxiv.org/pdf/2507.21200v1](http://arxiv.org/pdf/2507.21200v1)**

> **作者:** Soren Pedersen; Sanyam Jain; Mikkel Chavez; Viktor Ladehoff; Bruna Neves de Freitas; Ruben Pauwels
>
> **摘要:** This paper presents the development of a generative adversarial network (GAN) for synthesizing dental panoramic radiographs. Although exploratory in nature, the study aims to address the scarcity of data in dental research and education. We trained a deep convolutional GAN (DCGAN) using a Wasserstein loss with gradient penalty (WGANGP) on a dataset of 2322 radiographs of varying quality. The focus was on the dentoalveolar regions, other anatomical structures were cropped out. Extensive preprocessing and data cleaning were performed to standardize the inputs while preserving anatomical variability. We explored four candidate models by varying critic iterations, feature depth, and the use of denoising prior to training. A clinical expert evaluated the generated radiographs based on anatomical visibility and realism, using a 5-point scale (1 very poor 5 excellent). Most images showed moderate anatomical depiction, although some were degraded by artifacts. A trade-off was observed the model trained on non-denoised data yielded finer details especially in structures like the mandibular canal and trabecular bone, while a model trained on denoised data offered superior overall image clarity and sharpness. These findings provide a foundation for future work on GAN-based methods in dental imaging.
>
---
#### [new 029] Seeing Beyond Frames: Zero-Shot Pedestrian Intention Prediction with Raw Temporal Video and Multimodal Cues
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于自动驾驶中的行人意图预测任务，旨在解决复杂城市环境中对行人穿越意图的准确预测问题。论文提出了BF-PIP方法，基于Gemini 2.5 Pro，利用连续视频片段和多模态提示（如边界框和车辆速度），在无需训练的情况下实现零样本预测，准确率达73%，优于基于GPT-4V的方法。**

- **链接: [http://arxiv.org/pdf/2507.21161v1](http://arxiv.org/pdf/2507.21161v1)**

> **作者:** Pallavi Zambare; Venkata Nikhil Thanikella; Ying Liu
>
> **备注:** Accepted in IEEE 3rd International Conference on Artificial Intelligence, Blockchain, and Internet of Things (AIBThings 2025)
>
> **摘要:** Pedestrian intention prediction is essential for autonomous driving in complex urban environments. Conventional approaches depend on supervised learning over frame sequences and require extensive retraining to adapt to new scenarios. Here, we introduce BF-PIP (Beyond Frames Pedestrian Intention Prediction), a zero-shot approach built upon Gemini 2.5 Pro. It infers crossing intentions directly from short, continuous video clips enriched with structured JAAD metadata. In contrast to GPT-4V based methods that operate on discrete frames, BF-PIP processes uninterrupted temporal clips. It also incorporates bounding-box annotations and ego-vehicle speed via specialized multimodal prompts. Without any additional training, BF-PIP achieves 73% prediction accuracy, outperforming a GPT-4V baseline by 18 %. These findings illustrate that combining temporal video inputs with contextual cues enhances spatiotemporal perception and improves intent inference under ambiguous conditions. This approach paves the way for agile, retraining-free perception module in intelligent transportation system.
>
---
#### [new 030] StepAL: Step-aware Active Learning for Cataract Surgical Videos
- **分类: cs.CV**

- **简介: 该论文属于医疗视频分析任务，旨在解决白内障手术视频标注成本高的问题。传统主动学习方法不适用于长视频中的手术步骤识别。作者提出StepAL框架，结合伪标签与熵加权聚类策略，有效选择需标注的完整视频，提升模型性能，减少标注工作量。**

- **链接: [http://arxiv.org/pdf/2507.22059v1](http://arxiv.org/pdf/2507.22059v1)**

> **作者:** Nisarg A. Shah; Bardia Safaei; Shameema Sikder; S. Swaroop Vedula; Vishal M. Patel
>
> **备注:** Accepted to MICCAI 2025
>
> **摘要:** Active learning (AL) can reduce annotation costs in surgical video analysis while maintaining model performance. However, traditional AL methods, developed for images or short video clips, are suboptimal for surgical step recognition due to inter-step dependencies within long, untrimmed surgical videos. These methods typically select individual frames or clips for labeling, which is ineffective for surgical videos where annotators require the context of the entire video for annotation. To address this, we propose StepAL, an active learning framework designed for full video selection in surgical step recognition. StepAL integrates a step-aware feature representation, which leverages pseudo-labels to capture the distribution of predicted steps within each video, with an entropy-weighted clustering strategy. This combination prioritizes videos that are both uncertain and exhibit diverse step compositions for annotation. Experiments on two cataract surgery datasets (Cataract-1k and Cataract-101) demonstrate that StepAL consistently outperforms existing active learning approaches, achieving higher accuracy in step recognition with fewer labeled videos. StepAL offers an effective approach for efficient surgical video analysis, reducing the annotation burden in developing computer-assisted surgical systems.
>
---
#### [new 031] LiteFat: Lightweight Spatio-Temporal Graph Learning for Real-Time Driver Fatigue Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于实时驾驶员疲劳检测任务，旨在解决现有方法计算量大、延迟高，难以在嵌入式设备上运行的问题。论文提出LiteFat，通过轻量级时空图学习模型，结合MobileNet提取面部特征，构建时空图进行疲劳检测，实现在低资源设备上的高效准确检测。**

- **链接: [http://arxiv.org/pdf/2507.21756v1](http://arxiv.org/pdf/2507.21756v1)**

> **作者:** Jing Ren; Suyu Ma; Hong Jia; Xiwei Xu; Ivan Lee; Haytham Fayek; Xiaodong Li; Feng Xia
>
> **备注:** 6 pages, 1 figure
>
> **摘要:** Detecting driver fatigue is critical for road safety, as drowsy driving remains a leading cause of traffic accidents. Many existing solutions rely on computationally demanding deep learning models, which result in high latency and are unsuitable for embedded robotic devices with limited resources (such as intelligent vehicles/cars) where rapid detection is necessary to prevent accidents. This paper introduces LiteFat, a lightweight spatio-temporal graph learning model designed to detect driver fatigue efficiently while maintaining high accuracy and low computational demands. LiteFat involves converting streaming video data into spatio-temporal graphs (STG) using facial landmark detection, which focuses on key motion patterns and reduces unnecessary data processing. LiteFat uses MobileNet to extract facial features and create a feature matrix for the STG. A lightweight spatio-temporal graph neural network is then employed to identify signs of fatigue with minimal processing and low latency. Experimental results on benchmark datasets show that LiteFat performs competitively while significantly decreasing computational complexity and latency as compared to current state-of-the-art methods. This work enables the development of real-time, resource-efficient human fatigue detection systems that can be implemented upon embedded robotic devices.
>
---
#### [new 032] MSGCoOp: Multiple Semantic-Guided Context Optimization for Few-Shot Learning
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的小样本学习任务，旨在解决现有方法在新类别泛化上的过拟合与遗忘问题。作者提出MSGCoOp框架，通过多语义引导的上下文优化、LLM生成的类描述对齐及多样性正则化，提升少样本泛化能力，同时保持计算效率。实验表明其在跨领域任务中表现更优。**

- **链接: [http://arxiv.org/pdf/2507.21786v1](http://arxiv.org/pdf/2507.21786v1)**

> **作者:** Zhaolong Wang; Tongfeng Sun; Mingzheng Du; Yachao Huang
>
> **摘要:** Vision-language pre-trained models (VLMs) such as CLIP have demonstrated remarkable zero-shot generalization, and prompt learning has emerged as an efficient alternative to full fine-tuning. However, existing methods often struggle with generalization to novel classes, a phenomenon attributed to overfitting on seen classes and forgetting general knowledge. Furthermore, recent approaches that improve generalization often introduce complex architectures or heavy computational overhead. In this paper, we propose a Multiple Semantic-Guided Context Optimization (MSGCoOp) framework to enhance few-shot generalization while maintaining computational efficiency. Our approach leverages an ensemble of parallel learnable context vectors to capture diverse semantic aspects. To enrich these prompts, we introduce a semantic guidance mechanism that aligns them with comprehensive class descriptions automatically generated by a Large Language Model (LLM). Furthermore, a diversity regularization loss encourages the prompts to learn complementary and orthogonal features, preventing them from collapsing into redundant representations. Extensive experiments on 11 benchmark datasets show that MSGCoOp significantly improves performance on base-to-novel generalization, achieving an average harmonic mean improvement of 1.10\% over the strong KgCoOp baseline. Our method also demonstrates enhanced robustness in cross-domain generalization tasks. Our code is avaliable at: \href{https://github.com/Rain-Bus/MSGCoOp}{https://github.com/Rain-Bus/MSGCoOp}.
>
---
#### [new 033] SwinECAT: A Transformer-based fundus disease classification model with Shifted Window Attention and Efficient Channel Attention
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在解决眼底图像中病变区域小、疾病间差异细微导致的分类精度低问题。论文提出SwinECAT模型，结合Shifted Window Attention与Efficient Channel Attention机制，实现对9种眼底疾病的精细分类，显著提升了分类性能。**

- **链接: [http://arxiv.org/pdf/2507.21922v1](http://arxiv.org/pdf/2507.21922v1)**

> **作者:** Peiran Gu; Teng Yao; Mengshen He; Fuhao Duan; Feiyan Liu; RenYuan Peng; Bao Ge
>
> **备注:** 17 pages
>
> **摘要:** In recent years, artificial intelligence has been increasingly applied in the field of medical imaging. Among these applications, fundus image analysis presents special challenges, including small lesion areas in certain fundus diseases and subtle inter-disease differences, which can lead to reduced prediction accuracy and overfitting in the models. To address these challenges, this paper proposes the Transformer-based model SwinECAT, which combines the Shifted Window (Swin) Attention with the Efficient Channel Attention (ECA) Attention. SwinECAT leverages the Swin Attention mechanism in the Swin Transformer backbone to effectively capture local spatial structures and long-range dependencies within fundus images. The lightweight ECA mechanism is incorporated to guide the SwinECAT's attention toward critical feature channels, enabling more discriminative feature representation. In contrast to previous studies that typically classify fundus images into 4 to 6 categories, this work expands fundus disease classification to 9 distinct types, thereby enhancing the granularity of diagnosis. We evaluate our method on the Eye Disease Image Dataset (EDID) containing 16,140 fundus images for 9-category classification. Experimental results demonstrate that SwinECAT achieves 88.29\% accuracy, with weighted F1-score of 0.88 and macro F1-score of 0.90. The classification results of our proposed model SwinECAT significantly outperform the baseline Swin Transformer and multiple compared baseline models. To our knowledge, this represents the highest reported performance for 9-category classification on this public dataset.
>
---
#### [new 034] See Different, Think Better: Visual Variations Mitigating Hallucinations in LVLMs
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型中的幻觉缓解任务，旨在解决大型视觉语言模型（LVLMs）在细粒度视觉理解中产生的文本与视觉内容不一致问题。论文提出了ViHallu框架，通过生成可控的视觉变化图像和构建视觉指令，提升模型的视觉-语义对齐能力，从而减少幻觉现象。**

- **链接: [http://arxiv.org/pdf/2507.22003v1](http://arxiv.org/pdf/2507.22003v1)**

> **作者:** Ziyun Dai; Xiaoqiang Li; Shaohua Zhang; Yuanchen Wu; Jide Li
>
> **备注:** Accepted by ACM MM25
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities in visual understanding and multimodal reasoning. However, LVLMs frequently exhibit hallucination phenomena, manifesting as the generated textual responses that demonstrate inconsistencies with the provided visual content. Existing hallucination mitigation methods are predominantly text-centric, the challenges of visual-semantic alignment significantly limit their effectiveness, especially when confronted with fine-grained visual understanding scenarios. To this end, this paper presents ViHallu, a Vision-Centric Hallucination mitigation framework that enhances visual-semantic alignment through Visual Variation Image Generation and Visual Instruction Construction. ViHallu introduces \textbf{\textit{visual variation images}} with controllable visual alterations while maintaining the overall image structure. These images, combined with carefully constructed visual instructions, enable LVLMs to better understand fine-grained visual content through fine-tuning, allowing models to more precisely capture the correspondence between visual content and text, thereby enhancing visual-semantic alignment. Extensive experiments on multiple benchmarks show that ViHallu effectively enhances models' fine-grained visual understanding while significantly reducing hallucination tendencies. Furthermore, we release ViHallu-Instruction, a visual instruction dataset specifically designed for hallucination mitigation and visual-semantic alignment. Code is available at https://github.com/oliviadzy/ViHallu.
>
---
#### [new 035] Dual Guidance Semi-Supervised Action Detection
- **分类: cs.CV**

- **简介: 该论文属于时空动作定位任务，旨在解决标注数据有限下的动作检测问题。作者提出了一种半监督方法，通过双引导网络选择更优的伪边界框，结合帧级分类和边界框预测，提升模型性能。实验表明该方法在多个数据集上优于基于图像的半监督基线。**

- **链接: [http://arxiv.org/pdf/2507.21247v1](http://arxiv.org/pdf/2507.21247v1)**

> **作者:** Ankit Singh; Efstratios Gavves; Cees G. M. Snoek; Hilde Kuehne
>
> **摘要:** Semi-Supervised Learning (SSL) has shown tremendous potential to improve the predictive performance of deep learning models when annotations are hard to obtain. However, the application of SSL has so far been mainly studied in the context of image classification. In this work, we present a semi-supervised approach for spatial-temporal action localization. We introduce a dual guidance network to select better pseudo-bounding boxes. It combines a frame-level classification with a bounding-box prediction to enforce action class consistency across frames and boxes. Our evaluation across well-known spatial-temporal action localization datasets, namely UCF101-24 , J-HMDB-21 and AVA shows that the proposed module considerably enhances the model's performance in limited labeled data settings. Our framework achieves superior results compared to extended image-based semi-supervised baselines.
>
---
#### [new 036] Bridging Synthetic and Real-World Domains: A Human-in-the-Loop Weakly-Supervised Framework for Industrial Toxic Emission Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于工业烟雾分割任务，旨在解决真实场景中像素级标注稀缺的问题。作者提出CEDANet框架，结合弱监督和领域自适应方法，利用公民提供的视频级标签优化伪标签，并通过对抗特征对齐提升模型性能。实验表明其效果接近全监督模型，验证了方法在环境监测中的有效性与成本优势。**

- **链接: [http://arxiv.org/pdf/2507.22002v1](http://arxiv.org/pdf/2507.22002v1)**

> **作者:** Yida Tao; Yen-Chia Hsu
>
> **摘要:** Industrial smoke segmentation is critical for air-quality monitoring and environmental protection but is often hampered by the high cost and scarcity of pixel-level annotations in real-world settings. We introduce CEDANet, a human-in-the-loop, class-aware domain adaptation framework that uniquely integrates weak, citizen-provided video-level labels with adversarial feature alignment. Specifically, we refine pseudo-labels generated by a source-trained segmentation model using citizen votes, and employ class-specific domain discriminators to transfer rich source-domain representations to the industrial domain. Comprehensive experiments on SMOKE5K and custom IJmond datasets demonstrate that CEDANet achieves an F1-score of 0.414 and a smoke-class IoU of 0.261 with citizen feedback, vastly outperforming the baseline model, which scored 0.083 and 0.043 respectively. This represents a five-fold increase in F1-score and a six-fold increase in smoke-class IoU. Notably, CEDANet with citizen-constrained pseudo-labels achieves performance comparable to the same architecture trained on limited 100 fully annotated images with F1-score of 0.418 and IoU of 0.264, demonstrating its ability to reach small-sampled fully supervised-level accuracy without target-domain annotations. Our research validates the scalability and cost-efficiency of combining citizen science with weakly supervised domain adaptation, offering a practical solution for complex, data-scarce environmental monitoring applications.
>
---
#### [new 037] Mitigating Spurious Correlations in Weakly Supervised Semantic Segmentation via Cross-architecture Consistency Regularization
- **分类: cs.CV**

- **简介: 该论文属于弱监督语义分割任务，旨在解决因缺乏像素级标注导致的前景覆盖不全、边界不准及虚假关联问题，尤其针对工业烟雾与烟囱空间耦合的场景。论文提出一种跨架构一致性正则化方法，结合CNN与ViT的教师-学生框架，并引入知识迁移损失及后处理技术，以提升伪标签质量，减少模型偏差。**

- **链接: [http://arxiv.org/pdf/2507.21959v1](http://arxiv.org/pdf/2507.21959v1)**

> **作者:** Zheyuan Zhang; Yen-chia Hsu
>
> **摘要:** Scarcity of pixel-level labels is a significant challenge in practical scenarios. In specific domains like industrial smoke, acquiring such detailed annotations is particularly difficult and often requires expert knowledge. To alleviate this, weakly supervised semantic segmentation (WSSS) has emerged as a promising approach. However, due to the supervision gap and inherent bias in models trained with only image level labels, existing WSSS methods suffer from limitations such as incomplete foreground coverage, inaccurate object boundaries, and spurious correlations, especially in our domain, where emissions are always spatially coupled with chimneys. Previous solutions typically rely on additional priors or external knowledge to mitigate these issues, but they often lack scalability and fail to address the model's inherent bias toward co-occurring context. To address this, we propose a novel WSSS framework that directly targets the co-occurrence problem without relying on external supervision. Unlike prior methods that adopt a single network, we employ a teacher-student framework that combines CNNs and ViTs. We introduce a knowledge transfer loss that enforces cross-architecture consistency by aligning internal representations. Additionally, we incorporate post-processing techniques to address partial coverage and further improve pseudo mask quality.
>
---
#### [new 038] PanoSplatt3R: Leveraging Perspective Pretraining for Generalized Unposed Wide-Baseline Panorama Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于全景图像重建任务，旨在解决无准确姿态信息下的宽基线全景重建问题。作者提出PanoSplatt3R，通过将视角域的预训练迁移至全景域，并引入RoPE滚动机制，实现了高质量的新视角生成与深度估计，提升了方法的实用性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.21960v1](http://arxiv.org/pdf/2507.21960v1)**

> **作者:** Jiahui Ren; Mochu Xiang; Jiajun Zhu; Yuchao Dai
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Wide-baseline panorama reconstruction has emerged as a highly effective and pivotal approach for not only achieving geometric reconstruction of the surrounding 3D environment, but also generating highly realistic and immersive novel views. Although existing methods have shown remarkable performance across various benchmarks, they are predominantly reliant on accurate pose information. In real-world scenarios, the acquisition of precise pose often requires additional computational resources and is highly susceptible to noise. These limitations hinder the broad applicability and practicality of such methods. In this paper, we present PanoSplatt3R, an unposed wide-baseline panorama reconstruction method. We extend and adapt the foundational reconstruction pretrainings from the perspective domain to the panoramic domain, thus enabling powerful generalization capabilities. To ensure a seamless and efficient domain-transfer process, we introduce RoPE rolling that spans rolled coordinates in rotary positional embeddings across different attention heads, maintaining a minimal modification to RoPE's mechanism, while modeling the horizontal periodicity of panorama images. Comprehensive experiments demonstrate that PanoSplatt3R, even in the absence of pose information, significantly outperforms current state-of-the-art methods. This superiority is evident in both the generation of high-quality novel views and the accuracy of depth estimation, thereby showcasing its great potential for practical applications. Project page: https://npucvr.github.io/PanoSplatt3R
>
---
#### [new 039] Top2Pano: Learning to Generate Indoor Panoramas from Top-Down View
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在从2D俯视图生成室内全景图。为解决几何一致性与真实感难题，论文提出Top2Pano模型，先估计体积占据以推断3D结构，再通过体积渲染生成初步全景图，最后用扩散模型优化结果。实验表明其在多个数据集上优于基线方法。**

- **链接: [http://arxiv.org/pdf/2507.21371v1](http://arxiv.org/pdf/2507.21371v1)**

> **作者:** Zitong Zhang; Suranjan Gautam; Rui Yu
>
> **备注:** ICCV 2025. Project page: https://top2pano.github.io/
>
> **摘要:** Generating immersive 360{\deg} indoor panoramas from 2D top-down views has applications in virtual reality, interior design, real estate, and robotics. This task is challenging due to the lack of explicit 3D structure and the need for geometric consistency and photorealism. We propose Top2Pano, an end-to-end model for synthesizing realistic indoor panoramas from top-down views. Our method estimates volumetric occupancy to infer 3D structures, then uses volumetric rendering to generate coarse color and depth panoramas. These guide a diffusion-based refinement stage using ControlNet, enhancing realism and structural fidelity. Evaluations on two datasets show Top2Pano outperforms baselines, effectively reconstructing geometry, occlusions, and spatial arrangements. It also generalizes well, producing high-quality panoramas from schematic floorplans. Our results highlight Top2Pano's potential in bridging top-down views with immersive indoor synthesis.
>
---
#### [new 040] The Evolution of Video Anomaly Detection: A Unified Framework from DNN to MLLM
- **分类: cs.CV**

- **简介: 该论文属于视频异常检测任务，旨在识别和定位视频中的异常行为。论文系统回顾了基于深度学习和大语言模型的方法进展，提出了统一框架，分析新范式，构建分类体系，并探讨未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.21649v1](http://arxiv.org/pdf/2507.21649v1)**

> **作者:** Shibo Gao; Peipei Yang; Haiyang Guo; Yangyang Liu; Yi Chen; Shuai Li; Han Zhu; Jian Xu; Xu-Yao Zhang; Linlin Huang
>
> **摘要:** Video anomaly detection (VAD) aims to identify and ground anomalous behaviors or events in videos, serving as a core technology in the fields of intelligent surveillance and public safety. With the advancement of deep learning, the continuous evolution of deep model architectures has driven innovation in VAD methodologies, significantly enhancing feature representation and scene adaptability, thereby improving algorithm generalization and expanding application boundaries. More importantly, the rapid development of multi-modal large language (MLLMs) and large language models (LLMs) has introduced new opportunities and challenges to the VAD field. Under the support of MLLMs and LLMs, VAD has undergone significant transformations in terms of data annotation, input modalities, model architectures, and task objectives. The surge in publications and the evolution of tasks have created an urgent need for systematic reviews of recent advancements. This paper presents the first comprehensive survey analyzing VAD methods based on MLLMs and LLMs, providing an in-depth discussion of the changes occurring in the VAD field in the era of large models and their underlying causes. Additionally, this paper proposes a unified framework that encompasses both deep neural network (DNN)-based and LLM-based VAD methods, offering a thorough analysis of the new VAD paradigms empowered by LLMs, constructing a classification system, and comparing their strengths and weaknesses. Building on this foundation, this paper focuses on current VAD methods based on MLLMs/LLMs. Finally, based on the trajectory of technological advancements and existing bottlenecks, this paper distills key challenges and outlines future research directions, offering guidance for the VAD community.
>
---
#### [new 041] Multi-View Reconstruction with Global Context for 3D Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于3D异常检测任务，旨在解决工业质检中高精度检测因全局信息不足导致性能下降的问题。论文提出多视角重建（MVR）方法，将点云转换为多视角图像，通过重建框架增强全局信息学习，取得了良好检测效果。**

- **链接: [http://arxiv.org/pdf/2507.21555v1](http://arxiv.org/pdf/2507.21555v1)**

> **作者:** Yihan Sun; Yuqi Cheng; Yunkang Cao; Yuxin Zhang; Weiming Shen
>
> **备注:** 6 pages, 5 figures, IEEE International Conference on Systems, Man, and Cybernetics (IEEE SMC), 2025
>
> **摘要:** 3D anomaly detection is critical in industrial quality inspection. While existing methods achieve notable progress, their performance degrades in high-precision 3D anomaly detection due to insufficient global information. To address this, we propose Multi-View Reconstruction (MVR), a method that losslessly converts high-resolution point clouds into multi-view images and employs a reconstruction-based anomaly detection framework to enhance global information learning. Extensive experiments demonstrate the effectiveness of MVR, achieving 89.6\% object-wise AU-ROC and 95.7\% point-wise AU-ROC on the Real3D-AD benchmark.
>
---
#### [new 042] Evaluating Deepfake Detectors in the Wild
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像识别与安全任务，旨在评估深度伪造检测器在真实场景中的性能。研究者构建了包含50万张高质量深伪图像的数据集，测试显示多数检测器效果有限，且易受图像处理影响。**

- **链接: [http://arxiv.org/pdf/2507.21905v1](http://arxiv.org/pdf/2507.21905v1)**

> **作者:** Viacheslav Pirogov; Maksim Artemev
>
> **备注:** Accepted to the ICML 2025 Workshop 'DataWorld: Unifying Data Curation Frameworks Across Domains'
>
> **摘要:** Deepfakes powered by advanced machine learning models present a significant and evolving threat to identity verification and the authenticity of digital media. Although numerous detectors have been developed to address this problem, their effectiveness has yet to be tested when applied to real-world data. In this work we evaluate modern deepfake detectors, introducing a novel testing procedure designed to mimic real-world scenarios for deepfake detection. Using state-of-the-art deepfake generation methods, we create a comprehensive dataset containing more than 500,000 high-quality deepfake images. Our analysis shows that detecting deepfakes still remains a challenging task. The evaluation shows that in fewer than half of the deepfake detectors tested achieved an AUC score greater than 60%, with the lowest being 50%. We demonstrate that basic image manipulations, such as JPEG compression or image enhancement, can significantly reduce model performance. All code and data are publicly available at https://github.com/messlav/Deepfake-Detectors-in-the-Wild.
>
---
#### [new 043] XAI for Point Cloud Data using Perturbations based on Meaningful Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于点云分类的可解释人工智能（XAI）任务，旨在解决如何生成易于理解的模型解释问题。论文提出了一种基于点云分割的解释方法，通过新的点移动机制引入扰动，生成更具可解释性的显著图，优于传统聚类方法。**

- **链接: [http://arxiv.org/pdf/2507.22020v1](http://arxiv.org/pdf/2507.22020v1)**

> **作者:** Raju Ningappa Mulawade; Christoph Garth; Alexander Wiebel
>
> **备注:** 18 pages, 14 figures
>
> **摘要:** We propose a novel segmentation-based explainable artificial intelligence (XAI) method for neural networks working on point cloud classification. As one building block of this method, we propose a novel point-shifting mechanism to introduce perturbations in point cloud data. Recently, AI has seen an exponential growth. Hence, it is important to understand the decision-making process of AI algorithms when they are applied in critical areas. Our work focuses on explaining AI algorithms that classify point cloud data. An important aspect of the methods used for explaining AI algorithms is their ability to produce explanations that are easy for humans to understand. This allows them to analyze the AI algorithms better and make appropriate decisions based on that analysis. Therefore, in this work, we intend to generate meaningful explanations that can be easily interpreted by humans. The point cloud data we consider represents 3D objects such as cars, guitars, and laptops. We make use of point cloud segmentation models to generate explanations for the working of classification models. The segments are used to introduce perturbations into the input point cloud data and generate saliency maps. The perturbations are introduced using the novel point-shifting mechanism proposed in this work which ensures that the shifted points no longer influence the output of the classification algorithm. In contrast to previous methods, the segments used by our method are meaningful, i.e. humans can easily interpret the meaning of the segments. Thus, the benefit of our method over other methods is its ability to produce more meaningful saliency maps. We compare our method with the use of classical clustering algorithms to generate explanations. We also analyze the saliency maps generated for example inputs using our method to demonstrate the usefulness of the method in generating meaningful explanations.
>
---
#### [new 044] Chain-of-Cooking:Cooking Process Visualization via Bidirectional Chain-of-Thought Guidance
- **分类: cs.CV**

- **简介: 该论文属于图像生成与食品分析交叉任务，旨在根据食谱生成每步烹饪过程的图像。主要解决生成图像与文本描述语义不一致及步骤间上下文不连贯的问题。论文提出了Chain-of-Cooking模型，包含动态补丁选择模块、语义演化模块与双向思维链引导，以提升图像生成的语义一致性与上下文连贯性，并构建了CookViz数据集用于实验验证。**

- **链接: [http://arxiv.org/pdf/2507.21529v1](http://arxiv.org/pdf/2507.21529v1)**

> **作者:** Mengling Xu; Ming Tao; Bing-Kun Bao
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Cooking process visualization is a promising task in the intersection of image generation and food analysis, which aims to generate an image for each cooking step of a recipe. However, most existing works focus on generating images of finished foods based on the given recipes, and face two challenges to visualize the cooking process. First, the appearance of ingredients changes variously across cooking steps, it is difficult to generate the correct appearances of foods that match the textual description, leading to semantic inconsistency. Second, the current step might depend on the operations of previous step, it is crucial to maintain the contextual coherence of images in sequential order. In this work, we present a cooking process visualization model, called Chain-of-Cooking. Specifically, to generate correct appearances of ingredients, we present a Dynamic Patch Selection Module to retrieve previously generated image patches as references, which are most related to current textual contents. Furthermore, to enhance the coherence and keep the rational order of generated images, we propose a Semantic Evolution Module and a Bidirectional Chain-of-Thought (CoT) Guidance. To better utilize the semantics of previous texts, the Semantic Evolution Module establishes the semantical association between latent prompts and current cooking step, and merges it with the latent features. Then the CoT Guidance updates the merged features to guide the current cooking step remain coherent with the previous step. Moreover, we construct a dataset named CookViz, consisting of intermediate image-text pairs for the cooking process. Quantitative and qualitative experiments show that our method outperforms existing methods in generating coherent and semantic consistent cooking process.
>
---
#### [new 045] Wind Turbine Feature Detection Using Deep Learning and Synthetic Data
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉与深度学习任务，旨在解决风力涡轮机特征检测问题。为提升无人机自主巡检的安全性与准确性，论文提出通过合成数据生成方法，构建多样化训练集，并训练YOLOv11网络检测风力涡轮机及其关键特征，最终在真实图像中取得良好检测效果。**

- **链接: [http://arxiv.org/pdf/2507.21611v1](http://arxiv.org/pdf/2507.21611v1)**

> **作者:** Arash Shahirpour; Jakob Gebler; Manuel Sanders; Tim Reuscher
>
> **备注:** 8 pages, 5 figures, accepted at ICMV 2025
>
> **摘要:** For the autonomous drone-based inspection of wind turbine (WT) blades, accurate detection of the WT and its key features is essential for safe drone positioning and collision avoidance. Existing deep learning methods typically rely on manually labeled real-world images, which limits both the quantity and the diversity of training datasets in terms of weather conditions, lighting, turbine types, and image complexity. In this paper, we propose a method to generate synthetic training data that allows controlled variation of visual and environmental factors, increasing the diversity and hence creating challenging learning scenarios. Furthermore, we train a YOLOv11 feature detection network solely on synthetic WT images with a modified loss function, to detect WTs and their key features within an image. The resulting network is evaluated both using synthetic images and a set of real-world WT images and shows promising performance across both synthetic and real-world data, achieving a Pose mAP50-95 of 0.97 on real images never seen during training.
>
---
#### [new 046] Fairness and Robustness of CLIP-Based Models for Chest X-rays
- **分类: cs.CV**

- **简介: 该论文研究CLIP模型在胸片分类任务中的公平性和鲁棒性问题。作者在多个数据集上评估六种CLIP模型，分析其在不同年龄、性别、种族患者中的表现差异，并测试模型对捷径学习的鲁棒性。结果显示模型存在年龄相关性能差异，且依赖伪相关特征（如胸管）。论文旨在揭示CLIP模型在医疗应用中的潜在偏差与局限。**

- **链接: [http://arxiv.org/pdf/2507.21291v1](http://arxiv.org/pdf/2507.21291v1)**

> **作者:** Théo Sourget; David Restrepo; Céline Hudelot; Enzo Ferrante; Stergios Christodoulidis; Maria Vakalopoulou
>
> **备注:** Accepted for publication at the FAIMI MICCAI workshop 2025
>
> **摘要:** Motivated by the strong performance of CLIP-based models in natural image-text domains, recent efforts have adapted these architectures to medical tasks, particularly in radiology, where large paired datasets of images and reports, such as chest X-rays, are available. While these models have shown encouraging results in terms of accuracy and discriminative performance, their fairness and robustness in the different clinical tasks remain largely underexplored. In this study, we extensively evaluate six widely used CLIP-based models on chest X-ray classification using three publicly available datasets: MIMIC-CXR, NIH-CXR14, and NEATX. We assess the models fairness across six conditions and patient subgroups based on age, sex, and race. Additionally, we assess the robustness to shortcut learning by evaluating performance on pneumothorax cases with and without chest drains. Our results indicate performance gaps between patients of different ages, but more equitable results for the other attributes. Moreover, all models exhibit lower performance on images without chest drains, suggesting reliance on spurious correlations. We further complement the performance analysis with a study of the embeddings generated by the models. While the sensitive attributes could be classified from the embeddings, we do not see such patterns using PCA, showing the limitations of these visualisation techniques when assessing models. Our code is available at https://github.com/TheoSourget/clip_cxr_fairness
>
---
#### [new 047] GAITEX: Human motion dataset from impaired gait and rehabilitation exercises of inertial and optical sensor data
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文属于人体运动分析任务，旨在解决缺乏高质量、多样化的康复运动与步态数据问题。论文构建了包含惯性与光学传感器数据的多模态数据集GAITEX，涵盖正常与异常步态及康复训练动作，提供精准标注和多种分析工具，助力运动评估、步态分析等模型开发与验证。**

- **链接: [http://arxiv.org/pdf/2507.21069v1](http://arxiv.org/pdf/2507.21069v1)**

> **作者:** Andreas Spilz; Heiko Oppel; Jochen Werner; Kathrin Stucke-Straub; Felix Capanni; Michael Munz
>
> **摘要:** Wearable inertial measurement units (IMUs) offer a cost-effective and scalable means to assess human movement quality in clinical and everyday settings. However, the development of robust sensor-based classification models for physiotherapeutic exercises and gait analysis requires large, diverse datasets, which are costly and time-consuming to collect. Here, we present a multimodal dataset of physiotherapeutic exercises - including correct and clinically relevant variants - and gait-related exercises - including both normal and impaired gait patterns - recorded from 19 participants using synchronized IMUs and marker-based motion capture (MoCap). The dataset includes raw data from nine IMUs and thirty-five optical markers capturing full-body kinematics. Each IMU is additionally equipped with four optical markers, enabling precise comparison between IMU-derived orientation estimates and reference values from the MoCap system. To support further analysis, we also provide processed IMU orientations aligned with common segment coordinate systems, subject-specific OpenSim models, inverse kinematics results, and tools for visualizing IMU orientations in the musculoskeletal context. Detailed annotations of movement execution quality and time-stamped segmentations support diverse analysis goals. This dataset supports the development and benchmarking of machine learning models for tasks such as automatic exercise evaluation, gait analysis, temporal activity segmentation, and biomechanical parameter estimation. To facilitate reproducibility, we provide code for postprocessing, sensor-to-segment alignment, inverse kinematics computation, and technical validation. This resource is intended to accelerate research in machine learning-driven human movement analysis.
>
---
#### [new 048] HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels
- **分类: cs.CV**

- **简介: 该论文属于3D世界生成任务，旨在解决从文本或图像生成沉浸式、可探索和交互的3D场景问题。论文提出了HunyuanWorld 1.0框架，采用全景世界代理和分层3D网格表示，实现语义感知的世界分解与重建，支持多样化的3D世界生成，并具备良好的兼容性和交互性。**

- **链接: [http://arxiv.org/pdf/2507.21809v1](http://arxiv.org/pdf/2507.21809v1)**

> **作者:** HunyuanWorld Team; Zhenwei Wang; Yuhao Liu; Junta Wu; Zixiao Gu; Haoyuan Wang; Xuhui Zuo; Tianyu Huang; Wenhuan Li; Sheng Zhang; Yihang Lian; Yulin Tsai; Lifu Wang; Sicong Liu; Puhua Jiang; Xianghui Yang; Dongyuan Guo; Yixuan Tang; Xinyue Mao; Jiaao Yu; Junlin Yu; Jihong Zhang; Meng Chen; Liang Dong; Yiwen Jia; Chao Zhang; Yonghao Tan; Hao Zhang; Zheng Ye; Peng He; Runzhou Wu; Minghui Chen; Zhan Li; Wangchen Qin; Lei Wang; Yifu Sun; Lin Niu; Xiang Yuan; Xiaofeng Yang; Yingping He; Jie Xiao; Yangyu Tao; Jianchen Zhu; Jinbao Xue; Kai Liu; Chongqing Zhao; Xinming Wu; Tian Liu; Peng Chen; Di Wang; Yuhong Liu; Linus; Jie Jiang; Tengfei Wang; Chunchao Guo
>
> **备注:** Technical Report; Project Page: https://3d-models.hunyuan.tencent.com/world/
>
> **摘要:** Creating immersive and playable 3D worlds from texts or images remains a fundamental challenge in computer vision and graphics. Existing world generation approaches typically fall into two categories: video-based methods that offer rich diversity but lack 3D consistency and rendering efficiency, and 3D-based methods that provide geometric consistency but struggle with limited training data and memory-inefficient representations. To address these limitations, we present HunyuanWorld 1.0, a novel framework that combines the best of both worlds for generating immersive, explorable, and interactive 3D scenes from text and image conditions. Our approach features three key advantages: 1) 360{\deg} immersive experiences via panoramic world proxies; 2) mesh export capabilities for seamless compatibility with existing computer graphics pipelines; 3) disentangled object representations for augmented interactivity. The core of our framework is a semantically layered 3D mesh representation that leverages panoramic images as 360{\deg} world proxies for semantic-aware world decomposition and reconstruction, enabling the generation of diverse 3D worlds. Extensive experiments demonstrate that our method achieves state-of-the-art performance in generating coherent, explorable, and interactive 3D worlds while enabling versatile applications in virtual reality, physical simulation, game development, and interactive content creation.
>
---
#### [new 049] From Seeing to Experiencing: Scaling Navigation Foundation Models with Reinforcement Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决导航基础模型在现实城市环境中缺乏交互性和安全行为的问题。作者提出Seeing-to-Experiencing（S2E）框架，结合视频预训练与强化学习后训练，提升模型的交互能力，同时保持其泛化能力。论文还构建了评估基准NavBench-GS，验证了S2E在提升模型性能方面的有效性。**

- **链接: [http://arxiv.org/pdf/2507.22028v1](http://arxiv.org/pdf/2507.22028v1)**

> **作者:** Honglin He; Yukai Ma; Wayne Wu; Bolei Zhou
>
> **摘要:** Navigation foundation models trained on massive webscale data enable agents to generalize across diverse environments and embodiments. However, these models trained solely on offline data, often lack the capacity to reason about the consequences of their actions or adapt through counterfactual understanding. They thus face significant limitations in the real-world urban navigation where interactive and safe behaviors, such as avoiding obstacles and moving pedestrians, are critical. To tackle these challenges, we introduce the Seeing-to-Experiencing framework to scale the capability of navigation foundation models with reinforcement learning. S2E combines the strengths of pre-training on videos and post-training through RL. It maintains the generalizability acquired from large-scale real-world videos while enhancing its interactivity through RL in simulation environments. Specifically, we introduce two innovations: an Anchor-Guided Distribution Matching strategy, which stabilizes learning and models diverse motion patterns through anchor-based supervision; and a Residual-Attention Module, which obtains reactive behaviors from simulation environments without erasing the model's pretrained knowledge. Moreover, we establish a comprehensive end-to-end evaluation benchmark, NavBench-GS, built on photorealistic 3DGS reconstructions of real-world scenes that incorporate physical interactions. It can systematically assess the generalizability and safety of navigation foundation models. Extensive experiments show that S2E mitigates the diminishing returns often seen when scaling with offline data alone. We perform a thorough analysis of the benefits of Reinforcement Learning compared to Supervised Fine-Tuning in the context of post-training for robot learning. Our findings emphasize the crucial role of integrating interactive online experiences to effectively scale foundation models in Robotics.
>
---
#### [new 050] APT: Improving Diffusion Models for High Resolution Image Generation with Adaptive Path Tracing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型在高分辨率图像生成中的局限性。现有方法存在分布偏移和单调性问题。论文提出APT方法，结合统计匹配和尺度感知调度，提升生成质量与速度。**

- **链接: [http://arxiv.org/pdf/2507.21690v1](http://arxiv.org/pdf/2507.21690v1)**

> **作者:** Sangmin Han; Jinho Jeong; Jinwoo Kim; Seon Joo Kim
>
> **摘要:** Latent Diffusion Models (LDMs) are generally trained at fixed resolutions, limiting their capability when scaling up to high-resolution images. While training-based approaches address this limitation by training on high-resolution datasets, they require large amounts of data and considerable computational resources, making them less practical. Consequently, training-free methods, particularly patch-based approaches, have become a popular alternative. These methods divide an image into patches and fuse the denoising paths of each patch, showing strong performance on high-resolution generation. However, we observe two critical issues for patch-based approaches, which we call ``patch-level distribution shift" and ``increased patch monotonicity." To address these issues, we propose Adaptive Path Tracing (APT), a framework that combines Statistical Matching to ensure patch distributions remain consistent in upsampled latents and Scale-aware Scheduling to deal with the patch monotonicity. As a result, APT produces clearer and more refined details in high-resolution images. In addition, APT enables a shortcut denoising process, resulting in faster sampling with minimal quality degradation. Our experimental results confirm that APT produces more detailed outputs with improved inference speed, providing a practical approach to high-resolution image generation.
>
---
#### [new 051] Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于语义分割任务，旨在解决跨域场景下模型性能下降的问题。通过引入概率扩散对齐框架PDAF，挖掘潜在域先验信息，优化特征表示，以增强模型在未见域中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.21367v1](http://arxiv.org/pdf/2507.21367v1)**

> **作者:** I-Hsiang Chen; Hua-En Chang; Wei-Ting Chen; Jenq-Neng Hwang; Sy-Yen Kuo
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Domain Generalized Semantic Segmentation (DGSS) is a critical yet challenging task, as domain shifts in unseen environments can severely compromise model performance. While recent studies enhance feature alignment by projecting features into the source domain, they often neglect intrinsic latent domain priors, leading to suboptimal results. In this paper, we introduce PDAF, a Probabilistic Diffusion Alignment Framework that enhances the generalization of existing segmentation networks through probabilistic diffusion modeling. PDAF introduces a Latent Domain Prior (LDP) to capture domain shifts and uses this prior as a conditioning factor to align both source and unseen target domains. To achieve this, PDAF integrates into a pre-trained segmentation model and utilizes paired source and pseudo-target images to simulate latent domain shifts, enabling LDP modeling. The framework comprises three modules: the Latent Prior Extractor (LPE) predicts the LDP by supervising domain shifts; the Domain Compensation Module (DCM) adjusts feature representations to mitigate domain shifts; and the Diffusion Prior Estimator (DPE) leverages a diffusion process to estimate the LDP without requiring paired samples. This design enables PDAF to iteratively model domain shifts, progressively refining feature representations to enhance generalization under complex target conditions. Extensive experiments validate the effectiveness of PDAF across diverse and challenging urban scenes.
>
---
#### [new 052] Decoupled Spatio-Temporal Consistency Learning for Self-Supervised Tracking
- **分类: cs.CV**

- **简介: 该论文属于视觉跟踪任务，旨在解决依赖大量人工标注边界框的问题。通过提出一种新的自监督跟踪框架{\tracker}，利用解耦的时空一致性学习和实例对比损失，在无需边界框标注的情况下，实现高效跟踪表示学习。**

- **链接: [http://arxiv.org/pdf/2507.21606v1](http://arxiv.org/pdf/2507.21606v1)**

> **作者:** Yaozong Zheng; Bineng Zhong; Qihua Liang; Ning Li; Shuxiang Song
>
> **备注:** Accepted by AAAI2025
>
> **摘要:** The success of visual tracking has been largely driven by datasets with manual box annotations. However, these box annotations require tremendous human effort, limiting the scale and diversity of existing tracking datasets. In this work, we present a novel Self-Supervised Tracking framework named \textbf{{\tracker}}, designed to eliminate the need of box annotations. Specifically, a decoupled spatio-temporal consistency training framework is proposed to learn rich target information across timestamps through global spatial localization and local temporal association. This allows for the simulation of appearance and motion variations of instances in real-world scenarios. Furthermore, an instance contrastive loss is designed to learn instance-level correspondences from a multi-view perspective, offering robust instance supervision without additional labels. This new design paradigm enables {\tracker} to effectively learn generic tracking representations in a self-supervised manner, while reducing reliance on extensive box annotations. Extensive experiments on nine benchmark datasets demonstrate that {\tracker} surpasses \textit{SOTA} self-supervised tracking methods, achieving an improvement of more than 25.3\%, 20.4\%, and 14.8\% in AUC (AO) score on the GOT10K, LaSOT, TrackingNet datasets, respectively. Code: https://github.com/GXNU-ZhongLab/SSTrack.
>
---
#### [new 053] LinDeps: A Fine-tuning Free Post-Pruning Method to Remove Layer-Wise Linear Dependencies with Guaranteed Performance Preservation
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务中的神经网络模型压缩方向，旨在解决卷积神经网络（CNN）在资源受限平台部署时的冗余参数问题。现有剪枝方法忽略层内特征图的结构依赖关系，导致剪枝效果不佳。为此，作者提出了LinDeps方法，通过线性依赖分析识别并移除冗余卷积滤波器，并引入信号恢复机制保证模型性能，无需微调。实验表明，LinDeps在多种网络和数据集上提升了现有剪枝技术的压缩率和推理速度，达到新的剪枝效果。**

- **链接: [http://arxiv.org/pdf/2507.21573v1](http://arxiv.org/pdf/2507.21573v1)**

> **作者:** Maxim Henry; Adrien Deliège; Anthony Cioppa; Marc Van Droogenbroeck
>
> **备注:** 10 pages, 4 figures, 5 tables, 45 references
>
> **摘要:** Convolutional Neural Networks (CNN) are widely used in many computer vision tasks. Yet, their increasing size and complexity pose significant challenges for efficient deployment on resource-constrained platforms. Hence, network pruning has emerged as an effective way of reducing the size and computational requirements of neural networks by removing redundant or unimportant parameters. However, a fundamental challenge with pruning consists in optimally removing redundancies without degrading performance. Most existing pruning techniques overlook structural dependencies across feature maps within a layer, resulting in suboptimal pruning decisions. In this work, we introduce LinDeps, a novel post-pruning method, i.e., a pruning method that can be applied on top of any pruning technique, which systematically identifies and removes redundant filters via linear dependency analysis. Particularly, LinDeps applies pivoted QR decomposition to feature maps to detect and prune linearly dependent filters. Then, a novel signal recovery mechanism adjusts the next layer's kernels to preserve compatibility and performance without requiring any fine-tuning. Our experiments on CIFAR-10 and ImageNet with VGG and ResNet backbones demonstrate that LinDeps improves compression rates of existing pruning techniques while preserving performances, leading to a new state of the art in CNN pruning. We also benchmark LinDeps in low-resource setups where no retraining can be performed, which shows significant pruning improvements and inference speedups over a state-of-the-art method. LinDeps therefore constitutes an essential add-on for any current or future pruning technique.
>
---
#### [new 054] Enhancing and Accelerating Brain MRI through Deep Learning Reconstruction Using Prior Subject-Specific Imaging
- **分类: cs.CV; physics.med-ph**

- **简介: 该论文属于医学图像重建任务，旨在解决脑部MRI成像时间长的问题。作者提出了一种结合深度学习与先验影像信息的新框架，包含初始重建、深度配准和增强网络，提升了重建质量与速度，并改善了脑分割效果。**

- **链接: [http://arxiv.org/pdf/2507.21349v1](http://arxiv.org/pdf/2507.21349v1)**

> **作者:** Amirmohammad Shamaei; Alexander Stebner; Salome; Bosshart; Johanna Ospel; Gouri Ginde; Mariana Bento; Roberto Souza
>
> **摘要:** Magnetic resonance imaging (MRI) is a crucial medical imaging modality. However, long acquisition times remain a significant challenge, leading to increased costs, and reduced patient comfort. Recent studies have shown the potential of using deep learning models that incorporate information from prior subject-specific MRI scans to improve reconstruction quality of present scans. Integrating this prior information requires registration of the previous scan to the current image reconstruction, which can be time-consuming. We propose a novel deep-learning-based MRI reconstruction framework which consists of an initial reconstruction network, a deep registration model, and a transformer-based enhancement network. We validated our method on a longitudinal dataset of T1-weighted MRI scans with 2,808 images from 18 subjects at four acceleration factors (R5, R10, R15, R20). Quantitative metrics confirmed our approach's superiority over existing methods (p < 0.05, Wilcoxon signed-rank test). Furthermore, we analyzed the impact of our MRI reconstruction method on the downstream task of brain segmentation and observed improved accuracy and volumetric agreement with reference segmentations. Our approach also achieved a substantial reduction in total reconstruction time compared to methods that use traditional registration algorithms, making it more suitable for real-time clinical applications. The code associated with this work is publicly available at https://github.com/amirshamaei/longitudinal-mri-deep-recon.
>
---
#### [new 055] Shallow Deep Learning Can Still Excel in Fine-Grained Few-Shot Learning
- **分类: cs.CV**

- **简介: 该论文属于细粒度小样本学习任务，旨在解决浅层网络性能不足的问题。作者提出了一种位置感知的星座网络（LCN-4），通过空间特征融合、特征聚类和位置补偿机制，提升了浅层网络的性能，取得了与深层网络相当甚至更好的效果。**

- **链接: [http://arxiv.org/pdf/2507.22041v1](http://arxiv.org/pdf/2507.22041v1)**

> **作者:** Chaofei Qi; Chao Ye; Zhitai Liu; Weiyang Lin; Jianbin Qiu
>
> **摘要:** Deep learning has witnessed the extensive utilization across a wide spectrum of domains, including fine-grained few-shot learning (FGFSL) which heavily depends on deep backbones. Nonetheless, shallower deep backbones such as ConvNet-4, are not commonly preferred because they're prone to extract a larger quantity of non-abstract visual attributes. In this paper, we initially re-evaluate the relationship between network depth and the ability to fully encode few-shot instances, and delve into whether shallow deep architecture could effectuate comparable or superior performance to mainstream deep backbone. Fueled by the inspiration from vanilla ConvNet-4, we introduce a location-aware constellation network (LCN-4), equipped with a cutting-edge location-aware feature clustering module. This module can proficiently encoder and integrate spatial feature fusion, feature clustering, and recessive feature location, thereby significantly minimizing the overall loss. Specifically, we innovatively put forward a general grid position encoding compensation to effectively address the issue of positional information missing during the feature extraction process of specific ordinary convolutions. Additionally, we further propose a general frequency domain location embedding technique to offset for the location loss in clustering features. We have carried out validation procedures on three representative fine-grained few-shot benchmarks. Relevant experiments have established that LCN-4 notably outperforms the ConvNet-4 based State-of-the-Arts and achieves performance that is on par with or superior to most ResNet12-based methods, confirming the correctness of our conjecture.
>
---
#### [new 056] GuidPaint: Class-Guided Image Inpainting with Diffusion Models
- **分类: cs.CV; I.4.4**

- **简介: 该论文属于图像修复任务，旨在解决现有方法在无训练情况下缺乏对遮挡区域精细控制的问题。作者提出GuidPaint，通过引入分类器引导和结合随机与确定性采样，实现对修复过程的精确控制，提升修复结果的语义一致性和视觉真实性。**

- **链接: [http://arxiv.org/pdf/2507.21627v1](http://arxiv.org/pdf/2507.21627v1)**

> **作者:** Qimin Wang; Xinda Liu; Guohua Geng
>
> **摘要:** In recent years, diffusion models have been widely adopted for image inpainting tasks due to their powerful generative capabilities, achieving impressive results. Existing multimodal inpainting methods based on diffusion models often require architectural modifications and retraining, resulting in high computational cost. In contrast, context-aware diffusion inpainting methods leverage the model's inherent priors to adjust intermediate denoising steps, enabling high-quality inpainting without additional training and significantly reducing computation. However, these methods lack fine-grained control over the masked regions, often leading to semantically inconsistent or visually implausible content. To address this issue, we propose GuidPaint, a training-free, class-guided image inpainting framework. By incorporating classifier guidance into the denoising process, GuidPaint enables precise control over intermediate generations within the masked areas, ensuring both semantic consistency and visual realism. Furthermore, it integrates stochastic and deterministic sampling, allowing users to select preferred intermediate results and deterministically refine them. Experimental results demonstrate that GuidPaint achieves clear improvements over existing context-aware inpainting methods in both qualitative and quantitative evaluations.
>
---
#### [new 057] ZIUM: Zero-Shot Intent-Aware Adversarial Attack on Unlearned Models
- **分类: cs.CV; cs.CR**

- **简介: 论文提出ZIUM，用于对未学习模型进行零样本意图感知的对抗攻击，旨在解决现有方法在攻击效果和计算成本上的不足。该工作属于对抗攻击任务，目标是提升攻击的意图定制能力和效率。**

- **链接: [http://arxiv.org/pdf/2507.21985v1](http://arxiv.org/pdf/2507.21985v1)**

> **作者:** Hyun Jun Yook; Ga San Jhun; Jae Hyun Cho; Min Jeon; Donghyun Kim; Tae Hyung Kim; Youn Kyu Lee
>
> **备注:** Accepted to ICCV2025
>
> **摘要:** Machine unlearning (MU) removes specific data points or concepts from deep learning models to enhance privacy and prevent sensitive content generation. Adversarial prompts can exploit unlearned models to generate content containing removed concepts, posing a significant security risk. However, existing adversarial attack methods still face challenges in generating content that aligns with an attacker's intent while incurring high computational costs to identify successful prompts. To address these challenges, we propose ZIUM, a Zero-shot Intent-aware adversarial attack on Unlearned Models, which enables the flexible customization of target attack images to reflect an attacker's intent. Additionally, ZIUM supports zero-shot adversarial attacks without requiring further optimization for previously attacked unlearned concepts. The evaluation across various MU scenarios demonstrated ZIUM's effectiveness in successfully customizing content based on user-intent prompts while achieving a superior attack success rate compared to existing methods. Moreover, its zero-shot adversarial attack significantly reduces the attack time for previously attacked unlearned concepts.
>
---
#### [new 058] Cross-Architecture Distillation Made Simple with Redundancy Suppression
- **分类: cs.CV**

- **简介: 该论文属于知识蒸馏任务，旨在解决跨架构知识迁移效率低、设计复杂的问题。作者提出了一种简单的方法（RSD），通过抑制冗余信息实现知识迁移，并设计轻量模块保留学生模型特性。方法在多个基准上表现优异，且参数开销更低。**

- **链接: [http://arxiv.org/pdf/2507.21844v1](http://arxiv.org/pdf/2507.21844v1)**

> **作者:** Weijia Zhang; Yuehao Liu; Wu Ran; Chao Ma
>
> **备注:** Accepted by ICCV 2025 (Highlight)
>
> **摘要:** We describe a simple method for cross-architecture knowledge distillation, where the knowledge transfer is cast into a redundant information suppression formulation. Existing methods introduce sophisticated modules, architecture-tailored designs, and excessive parameters, which impair their efficiency and applicability. We propose to extract the architecture-agnostic knowledge in heterogeneous representations by reducing the redundant architecture-exclusive information. To this end, we present a simple redundancy suppression distillation (RSD) loss, which comprises cross-architecture invariance maximisation and feature decorrelation objectives. To prevent the student from entirely losing its architecture-specific capabilities, we further design a lightweight module that decouples the RSD objective from the student's internal representations. Our method is devoid of the architecture-specific designs and complex operations in the pioneering method of OFA. It outperforms OFA on CIFAR-100 and ImageNet-1k benchmarks with only a fraction of their parameter overhead, which highlights its potential as a simple and strong baseline to the cross-architecture distillation community.
>
---
#### [new 059] RelMap: Enhancing Online Map Construction with Class-Aware Spatial Relation and Semantic Priors
- **分类: cs.CV**

- **简介: 该论文属于在线高精地图构建任务，旨在提升自动驾驶系统的地图构建效果。论文提出RelMap框架，通过引入类别感知的空间关系先验和基于专家混合的语义先验，建模地图元素间的空间与语义关系，提升精度与泛化能力。方法兼容单帧与时序感知模型，在nuScenes和Argoverse 2数据集上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2507.21567v1](http://arxiv.org/pdf/2507.21567v1)**

> **作者:** Tianhui Cai; Yun Zhang; Zewei Zhou; Zhiyu Huang; Jiaqi Ma
>
> **摘要:** Online high-definition (HD) map construction plays an increasingly important role in scaling autonomous driving systems. Transformer-based methods have become prevalent in online HD map construction; however, existing approaches often neglect the inherent spatial and semantic relationships among map elements, which limits their accuracy and generalization. To address this, we propose RelMap, an end-to-end framework that enhances online map construction by incorporating spatial relations and semantic priors. We introduce a Class-aware Spatial Relation Prior, which explicitly encodes relative positional dependencies between map elements using a learnable class-aware relation encoder. Additionally, we propose a Mixture-of-Experts (MoE)-based Semantic Prior, which routes features to class-specific experts based on predicted class probabilities, refining instance feature decoding. Our method is compatible with both single-frame and temporal perception backbones, achieving state-of-the-art performance on both the nuScenes and Argoverse 2 datasets.
>
---
#### [new 060] Analyzing the Sensitivity of Vision Language Models in Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型（VLMs）在视觉问答任务中对违反Grice会话原则的敏感性。通过在问题中添加修饰语，测试GPT-4o、Claude-3.5-Sonnet和Gemini-1.5-Flash等模型的表现，发现其性能下降，表明该方法有助于揭示VLMs的局限性。**

- **链接: [http://arxiv.org/pdf/2507.21335v1](http://arxiv.org/pdf/2507.21335v1)**

> **作者:** Monika Shah; Sudarshan Balaji; Somdeb Sarkhel; Sanorita Dey; Deepak Venugopal
>
> **摘要:** We can think of Visual Question Answering as a (multimodal) conversation between a human and an AI system. Here, we explore the sensitivity of Vision Language Models (VLMs) through the lens of cooperative principles of conversation proposed by Grice. Specifically, even when Grice's maxims of conversation are flouted, humans typically do not have much difficulty in understanding the conversation even though it requires more cognitive effort. Here, we study if VLMs are capable of handling violations to Grice's maxims in a manner that is similar to humans. Specifically, we add modifiers to human-crafted questions and analyze the response of VLMs to these modifiers. We use three state-of-the-art VLMs in our study, namely, GPT-4o, Claude-3.5-Sonnet and Gemini-1.5-Flash on questions from the VQA v2.0 dataset. Our initial results seem to indicate that the performance of VLMs consistently diminish with the addition of modifiers which indicates our approach as a promising direction to understand the limitations of VLMs.
>
---
#### [new 061] Semantics versus Identity: A Divide-and-Conquer Approach towards Adjustable Medical Image De-Identification
- **分类: cs.CV**

- **简介: 该论文属于医学图像去标识化任务，旨在解决现有方法无法兼顾隐私保护与医学语义保留的问题。论文提出分治框架，包含身份区域屏蔽和语义补偿两步，并引入特征解耦策略，以实现灵活隐私保护与语义保留的平衡。**

- **链接: [http://arxiv.org/pdf/2507.21703v1](http://arxiv.org/pdf/2507.21703v1)**

> **作者:** Yuan Tian; Shuo Wang; Rongzhao Zhang; Zijian Chen; Yankai Jiang; Chunyi Li; Xiangyang Zhu; Fang Yan; Qiang Hu; XiaoSong Wang; Guangtao Zhai
>
> **备注:** Accepted to ICCV2025;
>
> **摘要:** Medical imaging has significantly advanced computer-aided diagnosis, yet its re-identification (ReID) risks raise critical privacy concerns, calling for de-identification (DeID) techniques. Unfortunately, existing DeID methods neither particularly preserve medical semantics, nor are flexibly adjustable towards different privacy levels. To address these issues, we propose a divide-and-conquer framework comprising two steps: (1) Identity-Blocking, which blocks varying proportions of identity-related regions, to achieve different privacy levels; and (2) Medical-Semantics-Compensation, which leverages pre-trained Medical Foundation Models (MFMs) to extract medical semantic features to compensate the blocked regions. Moreover, recognizing that features from MFMs may still contain residual identity information, we introduce a Minimum Description Length principle-based feature decoupling strategy, to effectively decouple and discard such identity components. Extensive evaluations against existing approaches across seven datasets and three downstream tasks, demonstrates our state-of-the-art performance.
>
---
#### [new 062] Detection Transformers Under the Knife: A Neuroscience-Inspired Approach to Ablations
- **分类: cs.CV; cs.AI; I.2; I.4**

- **简介: 该论文属于计算机视觉与可解释AI任务，旨在提升检测Transformer模型的透明度与效率。通过神经科学启发的方法，系统性地分析并量化了检测Transformer内部组件的作用，揭示了模型在组件缺失时的鲁棒性模式，为优化模型结构提供了依据。**

- **链接: [http://arxiv.org/pdf/2507.21723v1](http://arxiv.org/pdf/2507.21723v1)**

> **作者:** Nils Hütten; Florian Hölken; Hasan Tercan; Tobias Meisen
>
> **摘要:** In recent years, Explainable AI has gained traction as an approach to enhancing model interpretability and transparency, particularly in complex models such as detection transformers. Despite rapid advancements, a substantial research gap remains in understanding the distinct roles of internal components - knowledge that is essential for improving transparency and efficiency. Inspired by neuroscientific ablation studies, which investigate the functions of brain regions through selective impairment, we systematically analyze the impact of ablating key components in three state-of-the-art detection transformer models: Detection transformer (DETR), deformable detection transformer (DDETR), and DETR with improved denoising anchor boxes (DINO). The ablations target query embeddings, encoder and decoder multi-head self-attentions (MHSA) as well as decoder multi-head cross-attention (MHCA) layers. We evaluate the effects of these ablations on the performance metrics gIoU and F1-score, quantifying effects on both the classification and regression sub-tasks on the COCO dataset. To facilitate reproducibility and future research, we publicly release the DeepDissect library. Our findings reveal model-specific resilience patterns: while DETR is particularly sensitive to ablations in encoder MHSA and decoder MHCA, DDETR's multi-scale deformable attention enhances robustness, and DINO exhibits the greatest resilience due to its look-forward twice update rule, which helps distributing knowledge across blocks. These insights also expose structural redundancies, particularly in DDETR's and DINO's decoder MHCA layers, highlighting opportunities for model simplification without sacrificing performance. This study advances XAI for DETRs by clarifying the contributions of internal components to model performance, offering insights to optimize and improve transparency and efficiency in critical applications.
>
---
#### [new 063] Anyone Can Jailbreak: Prompt-Based Attacks on LLMs and T2Is
- **分类: cs.CV**

- **简介: 该论文研究提示攻击（jailbreaks）如何被非专家用户用于绕过大型语言模型和文本到图像系统的安全机制。属于安全与内容审核任务，旨在揭示当前防御体系的不足，并提出攻击策略分类。论文分析了多种攻击方法，并强调需加强现实场景中的上下文防御能力。**

- **链接: [http://arxiv.org/pdf/2507.21820v1](http://arxiv.org/pdf/2507.21820v1)**

> **作者:** Ahmed B Mustafa; Zihan Ye; Yang Lu; Michael P Pound; Shreyank N Gowda
>
> **摘要:** Despite significant advancements in alignment and content moderation, large language models (LLMs) and text-to-image (T2I) systems remain vulnerable to prompt-based attacks known as jailbreaks. Unlike traditional adversarial examples requiring expert knowledge, many of today's jailbreaks are low-effort, high-impact crafted by everyday users with nothing more than cleverly worded prompts. This paper presents a systems-style investigation into how non-experts reliably circumvent safety mechanisms through techniques such as multi-turn narrative escalation, lexical camouflage, implication chaining, fictional impersonation, and subtle semantic edits. We propose a unified taxonomy of prompt-level jailbreak strategies spanning both text-output and T2I models, grounded in empirical case studies across popular APIs. Our analysis reveals that every stage of the moderation pipeline, from input filtering to output validation, can be bypassed with accessible strategies. We conclude by highlighting the urgent need for context-aware defenses that reflect the ease with which these jailbreaks can be reproduced in real-world settings.
>
---
#### [new 064] X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决离散自回归模型在图像生成中视觉保真度低、细节失真等问题。作者提出X-Omni框架，结合强化学习、语义图像分词器和离线扩散解码器，提升了生成质量和指令遵循能力，实现图像与语言的统一建模。**

- **链接: [http://arxiv.org/pdf/2507.22058v1](http://arxiv.org/pdf/2507.22058v1)**

> **作者:** Zigang Geng; Yibing Wang; Yeyao Ma; Chen Li; Yongming Rao; Shuyang Gu; Zhao Zhong; Qinglin Lu; Han Hu; Xiaosong Zhang; Linus; Di Wang; Jie Jiang
>
> **摘要:** Numerous efforts have been made to extend the ``next token prediction'' paradigm to visual contents, aiming to create a unified approach for both image generation and understanding. Nevertheless, attempts to generate images through autoregressive modeling with discrete tokens have been plagued by issues such as low visual fidelity, distorted outputs, and failure to adhere to complex instructions when rendering intricate details. These shortcomings are likely attributed to cumulative errors during autoregressive inference or information loss incurred during the discretization process. Probably due to this challenge, recent research has increasingly shifted toward jointly training image generation with diffusion objectives and language generation with autoregressive objectives, moving away from unified modeling approaches. In this work, we demonstrate that reinforcement learning can effectively mitigate artifacts and largely enhance the generation quality of a discrete autoregressive modeling method, thereby enabling seamless integration of image and language generation. Our framework comprises a semantic image tokenizer, a unified autoregressive model for both language and images, and an offline diffusion decoder for image generation, termed X-Omni. X-Omni achieves state-of-the-art performance in image generation tasks using a 7B language model, producing images with high aesthetic quality while exhibiting strong capabilities in following instructions and rendering long texts.
>
---
#### [new 065] Predict Patient Self-reported Race from Skin Histological Images
- **分类: cs.CV; cs.CE**

- **简介: 该论文属于计算病理学任务，旨在研究AI模型是否会无意中从皮肤组织图像中学习种族信息。作者通过深度学习模型预测患者自报种族，并分析模型依赖的形态特征，发现表皮区域是关键预测特征。研究强调了数据整理和偏见缓解在病理学AI部署中的重要性。**

- **链接: [http://arxiv.org/pdf/2507.21912v1](http://arxiv.org/pdf/2507.21912v1)**

> **作者:** Shengjia Chen; Ruchika Verma; Kevin Clare; Jannes Jegminat; Kuan-lin Huang; Brandon Veremis; Thomas Fuchs; Gabriele Campanella
>
> **备注:** Accepted to the MICCAI Workshop on Fairness of AI in Medical Imaging (FAIMI), 2025
>
> **摘要:** Artificial Intelligence (AI) has demonstrated success in computational pathology (CPath) for disease detection, biomarker classification, and prognosis prediction. However, its potential to learn unintended demographic biases, particularly those related to social determinants of health, remains understudied. This study investigates whether deep learning models can predict self-reported race from digitized dermatopathology slides and identifies potential morphological shortcuts. Using a multisite dataset with a racially diverse population, we apply an attention-based mechanism to uncover race-associated morphological features. After evaluating three dataset curation strategies to control for confounding factors, the final experiment showed that White and Black demographic groups retained high prediction performance (AUC: 0.799, 0.762), while overall performance dropped to 0.663. Attention analysis revealed the epidermis as a key predictive feature, with significant performance declines when these regions were removed. These findings highlight the need for careful data curation and bias mitigation to ensure equitable AI deployment in pathology. Code available at: https://github.com/sinai-computational-pathology/CPath_SAIF.
>
---
#### [new 066] Suppressing Gradient Conflict for Generalizable Deepfake Detection
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测任务，旨在解决模型在源域和在线合成伪造数据联合训练时出现的梯度冲突问题。作者提出CS-DFD框架，包含UVS和CGR两个模块，分别通过优化更新向量和降低特征空间梯度冲突，提升模型在源域和目标域的检测性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.21530v1](http://arxiv.org/pdf/2507.21530v1)**

> **作者:** Ming-Hui Liu; Harry Cheng; Xin Luo; Xin-Shun Xu
>
> **备注:** V1
>
> **摘要:** Robust deepfake detection models must be capable of generalizing to ever-evolving manipulation techniques beyond training data. A promising strategy is to augment the training data with online synthesized fake images containing broadly generalizable artifacts. However, in the context of deepfake detection, it is surprising that jointly training on both original and online synthesized forgeries may result in degraded performance. This contradicts the common belief that incorporating more source-domain data should enhance detection accuracy. Through empirical analysis, we trace this degradation to gradient conflicts during backpropagation which force a trade-off between source domain accuracy and target domain generalization. To overcome this issue, we propose a Conflict-Suppressed Deepfake Detection (CS-DFD) framework that explicitly mitigates the gradient conflict via two synergistic modules. First, an Update Vector Search (UVS) module searches for an alternative update vector near the initial gradient vector to reconcile the disparities of the original and online synthesized forgeries. By further transforming the search process into an extremum optimization problem, UVS yields the uniquely update vector, which maximizes the simultaneous loss reductions for each data type. Second, a Conflict Gradient Reduction (CGR) module enforces a low-conflict feature embedding space through a novel Conflict Descent Loss. This loss penalizes misaligned gradient directions and guides the learning of representations with aligned, non-conflicting gradients. The synergy of UVS and CGR alleviates gradient interference in both parameter optimization and representation learning. Experiments on multiple deepfake benchmarks demonstrate that CS-DFD achieves state-of-the-art performance in both in-domain detection accuracy and cross-domain generalization.
>
---
#### [new 067] MetaLab: Few-Shot Game Changer for Image Recognition
- **分类: cs.CV**

- **简介: 该论文属于图像识别任务，旨在解决小样本（few-shot）图像识别问题。作者提出了MetaLab方法，通过CIELab颜色空间转换和协同学习策略，提升模型在仅有少量样本时的识别性能，实验表明其准确率接近人类识别水平。**

- **链接: [http://arxiv.org/pdf/2507.22057v1](http://arxiv.org/pdf/2507.22057v1)**

> **作者:** Chaofei Qi; Zhitai Liu; Jianbin Qiu
>
> **摘要:** Difficult few-shot image recognition has significant application prospects, yet remaining the substantial technical gaps with the conventional large-scale image recognition. In this paper, we have proposed an efficient original method for few-shot image recognition, called CIELab-Guided Coherent Meta-Learning (MetaLab). Structurally, our MetaLab comprises two collaborative neural networks: LabNet, which can perform domain transformation for the CIELab color space and extract rich grouped features, and coherent LabGNN, which can facilitate mutual learning between lightness graph and color graph. For sufficient certification, we have implemented extensive comparative studies on four coarse-grained benchmarks, four fine-grained benchmarks, and four cross-domain few-shot benchmarks. Specifically, our method can achieve high accuracy, robust performance, and effective generalization capability with one-shot sample per class. Overall, all experiments have demonstrated that our MetaLab can approach 99\% $\uparrow\downarrow$ accuracy, reaching the human recognition ceiling with little visual deviation.
>
---
#### [new 068] Locally Controlled Face Aging with Latent Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决现有方法将面部老化视为全局同质过程的问题。作者提出一种基于潜在扩散模型的局部控制老化方法，实现对特定面部区域的精细化老化处理，并通过潜在扩散优化器确保整体一致性，从而生成更真实、个性化的老化效果。**

- **链接: [http://arxiv.org/pdf/2507.21600v1](http://arxiv.org/pdf/2507.21600v1)**

> **作者:** Lais Isabelle Alves dos Santos; Julien Despois; Thibaut Chauffier; Sileye O. Ba; Giovanni Palma
>
> **摘要:** We present a novel approach to face aging that addresses the limitations of current methods which treat aging as a global, homogeneous process. Existing techniques using GANs and diffusion models often condition generation on a reference image and target age, neglecting that facial regions age heterogeneously due to both intrinsic chronological factors and extrinsic elements like sun exposure. Our method leverages latent diffusion models to selectively age specific facial regions using local aging signs. This approach provides significantly finer-grained control over the generation process, enabling more realistic and personalized aging. We employ a latent diffusion refiner to seamlessly blend these locally aged regions, ensuring a globally consistent and natural-looking synthesis. Experimental results demonstrate that our method effectively achieves three key criteria for successful face aging: robust identity preservation, high-fidelity and realistic imagery, and a natural, controllable aging progression.
>
---
#### [new 069] Tracking Moose using Aerial Object Detection
- **分类: cs.CV; I.4.8**

- **简介: 该论文属于计算机视觉与野生动物追踪任务，旨在解决在无人机航拍图像中准确检测小型目标（如驼鹿）的问题。为提升检测效果，研究应用了图像分块增强方法，并比较了三种不同架构的目标检测模型性能。结果显示，轻量级模型在特定设置下表现优异，适合用于计算受限的无人机部署。**

- **链接: [http://arxiv.org/pdf/2507.21256v1](http://arxiv.org/pdf/2507.21256v1)**

> **作者:** Christopher Indris; Raiyan Rahman; Goetz Bramesfeld; Guanghui Wang
>
> **备注:** 18 pages, 6 figures, 8 tables
>
> **摘要:** Aerial wildlife tracking is critical for conservation efforts and relies on detecting small objects on the ground below the aircraft. It presents technical challenges: crewed aircraft are expensive, risky and disruptive; autonomous drones have limited computational capacity for onboard AI systems. Since the objects of interest may appear only a few pixels wide, small object detection is an inherently challenging computer vision subfield compounded by computational efficiency needs. This paper applies a patching augmentation to datasets to study model performance under various settings. A comparative study of three common yet architecturally diverse object detectors is conducted using the data, varying the patching method's hyperparameters against detection accuracy. Each model achieved at least 93\% mAP@IoU=0.5 on at least one patching configuration. Statistical analyses provide an in-depth commentary on the effects of various factors. Analysis also shows that faster, simpler models are about as effective as models that require more computational power for this task and perform well given limited patch scales, encouraging UAV deployment. Datasets and models will be made available via https://github.com/chrisindris/Moose.
>
---
#### [new 070] An Angular-Temporal Interaction Network for Light Field Object Tracking in Low-Light Scenes
- **分类: cs.CV**

- **简介: 该论文属于低光照场景下的光场目标跟踪任务，旨在解决现有方法在时间域中难以可靠建模角度特征的问题。论文提出了一种新的光场极平面结构图像（ESI）表示方法，并设计了角度-时间交互网络（ATINet）以学习角度感知的表示。此外，构建了一个大规模低光光场目标跟踪数据集。实验表明，该方法在单目标和多目标跟踪中均表现出色。**

- **链接: [http://arxiv.org/pdf/2507.21460v1](http://arxiv.org/pdf/2507.21460v1)**

> **作者:** Mianzhao Wang; Fan Shi; Xu Cheng; Feifei Zhang; Shengyong Chen
>
> **摘要:** High-quality 4D light field representation with efficient angular feature modeling is crucial for scene perception, as it can provide discriminative spatial-angular cues to identify moving targets. However, recent developments still struggle to deliver reliable angular modeling in the temporal domain, particularly in complex low-light scenes. In this paper, we propose a novel light field epipolar-plane structure image (ESI) representation that explicitly defines the geometric structure within the light field. By capitalizing on the abrupt changes in the angles of light rays within the epipolar plane, this representation can enhance visual expression in low-light scenes and reduce redundancy in high-dimensional light fields. We further propose an angular-temporal interaction network (ATINet) for light field object tracking that learns angular-aware representations from the geometric structural cues and angular-temporal interaction cues of light fields. Furthermore, ATINet can also be optimized in a self-supervised manner to enhance the geometric feature interaction across the temporal domain. Finally, we introduce a large-scale light field low-light dataset for object tracking. Extensive experimentation demonstrates that ATINet achieves state-of-the-art performance in single object tracking. Furthermore, we extend the proposed method to multiple object tracking, which also shows the effectiveness of high-quality light field angular-temporal modeling.
>
---
#### [new 071] Collaborative Perceiver: Elevating Vision-based 3D Object Detection via Local Density-Aware Spatial Occupancy
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中基于视觉的3D目标检测任务。旨在解决现有方法忽略环境上下文信息的问题。作者提出了一种多任务学习框架Collaborative Perceiver (CoP)，通过引入空间占据信息和局部密度感知策略，提升BEV表示能力，实现了更准确的3D目标检测。**

- **链接: [http://arxiv.org/pdf/2507.21358v1](http://arxiv.org/pdf/2507.21358v1)**

> **作者:** Jicheng Yuan; Manh Nguyen Duc; Qian Liu; Manfred Hauswirth; Danh Le Phuoc
>
> **摘要:** Vision-based bird's-eye-view (BEV) 3D object detection has advanced significantly in autonomous driving by offering cost-effectiveness and rich contextual information. However, existing methods often construct BEV representations by collapsing extracted object features, neglecting intrinsic environmental contexts, such as roads and pavements. This hinders detectors from comprehensively perceiving the characteristics of the physical world. To alleviate this, we introduce a multi-task learning framework, Collaborative Perceiver (CoP), that leverages spatial occupancy as auxiliary information to mine consistent structural and conceptual similarities shared between 3D object detection and occupancy prediction tasks, bridging gaps in spatial representations and feature refinement. To this end, we first propose a pipeline to generate dense occupancy ground truths incorporating local density information (LDO) for reconstructing detailed environmental information. Next, we employ a voxel-height-guided sampling (VHS) strategy to distill fine-grained local features according to distinct object properties. Furthermore, we develop a global-local collaborative feature fusion (CFF) module that seamlessly integrates complementary knowledge between both tasks, thus composing more robust BEV representations. Extensive experiments on the nuScenes benchmark demonstrate that CoP outperforms existing vision-based frameworks, achieving 49.5\% mAP and 59.2\% NDS on the test set. Code and supplementary materials are available at this link https://github.com/jichengyuan/Collaborative-Perceiver.
>
---
#### [new 072] EIFNet: Leveraging Event-Image Fusion for Robust Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于事件相机语义分割任务，旨在解决事件流数据稀疏噪声大、多模态融合困难的问题。作者提出了EIFNet网络，包含事件特征优化、模态自适应校准和注意力门控融合模块，实现事件与图像数据的有效融合，在两个数据集上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2507.21971v1](http://arxiv.org/pdf/2507.21971v1)**

> **作者:** Zhijiang Li; Haoran He
>
> **摘要:** Event-based semantic segmentation explores the potential of event cameras, which offer high dynamic range and fine temporal resolution, to achieve robust scene understanding in challenging environments. Despite these advantages, the task remains difficult due to two main challenges: extracting reliable features from sparse and noisy event streams, and effectively fusing them with dense, semantically rich image data that differ in structure and representation. To address these issues, we propose EIFNet, a multi-modal fusion network that combines the strengths of both event and frame-based inputs. The network includes an Adaptive Event Feature Refinement Module (AEFRM), which improves event representations through multi-scale activity modeling and spatial attention. In addition, we introduce a Modality-Adaptive Recalibration Module (MARM) and a Multi-Head Attention Gated Fusion Module (MGFM), which align and integrate features across modalities using attention mechanisms and gated fusion strategies. Experiments on DDD17-Semantic and DSEC-Semantic datasets show that EIFNet achieves state-of-the-art performance, demonstrating its effectiveness in event-based semantic segmentation.
>
---
#### [new 073] Semantic Segmentation of iPS Cells: Case Study on Model Complexity in Biomedical Imaging
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决低对比度、细微边界条件下的iPS细胞分割问题。作者通过优化DeepLabv3模型，在无需结构修改的情况下，取得了优于SAM2和MedSAM2等大模型的性能，证明模型复杂度并非越高越好。论文还提供了适用于小数据集和领域特定编码的开源实现，推动再生医学图像分割的发展。**

- **链接: [http://arxiv.org/pdf/2507.21608v1](http://arxiv.org/pdf/2507.21608v1)**

> **作者:** Maoquan Zhang; Bisser Raytchev; Xiujuan Sun
>
> **备注:** 19th International Conference on Machine Vision Applications MVA2025
>
> **摘要:** Medical image segmentation requires not only accuracy but also robustness under challenging imaging conditions. In this study, we show that a carefully configured DeepLabv3 model can achieve high performance in segmenting induced pluripotent stem (iPS) cell colonies, and, under our experimental conditions, outperforms large-scale foundation models such as SAM2 and its medical variant MedSAM2 without structural modifications. These results suggest that, for specialized tasks characterized by subtle, low-contrast boundaries, increased model complexity does not necessarily translate to better performance. Our work revisits the assumption that ever-larger and more generalized architectures are always preferable, and provides evidence that appropriately adapted, simpler models may offer strong accuracy and practical reliability in domain-specific biomedical applications. We also offer an open-source implementation that includes strategies for small datasets and domain-specific encoding, with the aim of supporting further advances in semantic segmentation for regenerative medicine and related fields.
>
---
#### [new 074] MetaCLIP 2: A Worldwide Scaling Recipe
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态预训练任务，旨在解决跨语言图像-文本理解中的数据非英语化和性能下降问题。作者提出了MetaCLIP 2，通过全新训练方法利用全球网络数据，在不依赖翻译或架构改动的情况下，提升了多语言场景下的图像分类和检索性能，实现了多项新纪录。**

- **链接: [http://arxiv.org/pdf/2507.22062v1](http://arxiv.org/pdf/2507.22062v1)**

> **作者:** Yung-Sung Chuang; Yang Li; Dong Wang; Ching-Feng Yeh; Kehan Lyu; Ramya Raghavendra; James Glass; Lifei Huang; Jason Weston; Luke Zettlemoyer; Xinlei Chen; Zhuang Liu; Saining Xie; Wen-tau Yih; Shang-Wen Li; Hu Xu
>
> **备注:** 10 pages
>
> **摘要:** Contrastive Language-Image Pretraining (CLIP) is a popular foundation model, supporting from zero-shot classification, retrieval to encoders for multimodal large language models (MLLMs). Although CLIP is successfully trained on billion-scale image-text pairs from the English world, scaling CLIP's training further to learning from the worldwide web data is still challenging: (1) no curation method is available to handle data points from non-English world; (2) the English performance from existing multilingual CLIP is worse than its English-only counterpart, i.e., "curse of multilinguality" that is common in LLMs. Here, we present MetaCLIP 2, the first recipe training CLIP from scratch on worldwide web-scale image-text pairs. To generalize our findings, we conduct rigorous ablations with minimal changes that are necessary to address the above challenges and present a recipe enabling mutual benefits from English and non-English world data. In zero-shot ImageNet classification, MetaCLIP 2 ViT-H/14 surpasses its English-only counterpart by 0.8% and mSigLIP by 0.7%, and surprisingly sets new state-of-the-art without system-level confounding factors (e.g., translation, bespoke architecture changes) on multilingual benchmarks, such as CVQA with 57.4%, Babel-ImageNet with 50.2% and XM3600 with 64.3% on image-to-text retrieval.
>
---
#### [new 075] MMAT-1M: A Large Reasoning Dataset for Multimodal Agent Tuning
- **分类: cs.CV**

- **简介: 该论文属于多模态大模型训练任务，旨在解决当前缺乏大规模高质量多模态代理调优数据集的问题。论文构建了包含百万级数据的MMAT-1M数据集，支持思维链、反思和工具调用。通过四阶段数据引擎生成带推理路径和反思的多轮对话，并可压缩为单轮格式。实验证明该数据集有效提升了模型在多模态推理和工具使用方面的能力。**

- **链接: [http://arxiv.org/pdf/2507.21924v1](http://arxiv.org/pdf/2507.21924v1)**

> **作者:** Tianhong Gao; Yannian Fu; Weiqun Wu; Haixiao Yue; Shanshan Liu; Gang Zhang
>
> **摘要:** Large Language Models (LLMs), enhanced through agent tuning, have demonstrated remarkable capabilities in Chain-of-Thought (CoT) and tool utilization, significantly surpassing the performance of standalone models. However, the multimodal domain still lacks a large-scale, high-quality agent tuning dataset to unlock the full potential of multimodal large language models. To bridge this gap, we introduce MMAT-1M, the first million-scale multimodal agent tuning dataset designed to support CoT, reflection, and dynamic tool usage. Our dataset is constructed through a novel four-stage data engine: 1) We first curate publicly available multimodal datasets containing question-answer pairs; 2) Then, leveraging GPT-4o, we generate rationales for the original question-answer pairs and dynamically integrate API calls and Retrieval Augmented Generation (RAG) information through a multi-turn paradigm; 3) Furthermore, we refine the rationales through reflection to ensure logical consistency and accuracy, creating a multi-turn dialogue dataset with both Rationale and Reflection (RR); 4) Finally, to enhance efficiency, we optionally compress multi-turn dialogues into a One-turn Rationale and Reflection (ORR) format. By fine-tuning open-source multimodal models on the MMAT-1M, we observe significant performance gains. For instance, the InternVL2.5-8B-RR model achieves an average improvement of 2.7% across eight public benchmarks and 8.8% on the RAG benchmark Dyn-VQA, demonstrating the dataset's effectiveness in enhancing multimodal reasoning and tool-based capabilities. The dataset is publicly available at https://github.com/VIS-MPU-Agent/MMAT-1M.
>
---
#### [new 076] SAMITE: Position Prompted SAM2 with Calibrated Memory for Visual Object Tracking
- **分类: cs.CV**

- **简介: 该论文属于视觉目标跟踪任务，旨在解决目标遮挡和干扰物影响导致跟踪误差传播的问题。论文提出了SAMITE模型，在SAM2基础上增加原型记忆库和位置提示生成模块，提升跟踪鲁棒性与准确性。**

- **链接: [http://arxiv.org/pdf/2507.21732v1](http://arxiv.org/pdf/2507.21732v1)**

> **作者:** Qianxiong Xu; Lanyun Zhu; Chenxi Liu; Guosheng Lin; Cheng Long; Ziyue Li; Rui Zhao
>
> **摘要:** Visual Object Tracking (VOT) is widely used in applications like autonomous driving to continuously track targets in videos. Existing methods can be roughly categorized into template matching and autoregressive methods, where the former usually neglects the temporal dependencies across frames and the latter tends to get biased towards the object categories during training, showing weak generalizability to unseen classes. To address these issues, some methods propose to adapt the video foundation model SAM2 for VOT, where the tracking results of each frame would be encoded as memory for conditioning the rest of frames in an autoregressive manner. Nevertheless, existing methods fail to overcome the challenges of object occlusions and distractions, and do not have any measures to intercept the propagation of tracking errors. To tackle them, we present a SAMITE model, built upon SAM2 with additional modules, including: (1) Prototypical Memory Bank: We propose to quantify the feature-wise and position-wise correctness of each frame's tracking results, and select the best frames to condition subsequent frames. As the features of occluded and distracting objects are feature-wise and position-wise inaccurate, their scores would naturally be lower and thus can be filtered to intercept error propagation; (2) Positional Prompt Generator: To further reduce the impacts of distractors, we propose to generate positional mask prompts to provide explicit positional clues for the target, leading to more accurate tracking. Extensive experiments have been conducted on six benchmarks, showing the superiority of SAMITE. The code is available at https://github.com/Sam1224/SAMITE.
>
---
#### [new 077] ArtSeek: Deep artwork understanding via multimodal in-context reasoning and late interaction retrieval
- **分类: cs.CV**

- **简介: 该论文属于艺术分析任务，旨在解决缺乏外部知识链接的数字艺术品理解问题。作者提出了ArtSeek框架，结合多模态大模型与检索增强生成，通过图像输入实现艺术作品的分类、检索与解释。论文工作包括构建多任务分类网络、检索模块与推理策略，并提出了WikiFragments数据集，显著提升了艺术分析性能。**

- **链接: [http://arxiv.org/pdf/2507.21917v1](http://arxiv.org/pdf/2507.21917v1)**

> **作者:** Nicola Fanelli; Gennaro Vessio; Giovanna Castellano
>
> **摘要:** Analyzing digitized artworks presents unique challenges, requiring not only visual interpretation but also a deep understanding of rich artistic, contextual, and historical knowledge. We introduce ArtSeek, a multimodal framework for art analysis that combines multimodal large language models with retrieval-augmented generation. Unlike prior work, our pipeline relies only on image input, enabling applicability to artworks without links to Wikidata or Wikipedia-common in most digitized collections. ArtSeek integrates three key components: an intelligent multimodal retrieval module based on late interaction retrieval, a contrastive multitask classification network for predicting artist, genre, style, media, and tags, and an agentic reasoning strategy enabled through in-context examples for complex visual question answering and artwork explanation via Qwen2.5-VL. Central to this approach is WikiFragments, a Wikipedia-scale dataset of image-text fragments curated to support knowledge-grounded multimodal reasoning. Our framework achieves state-of-the-art results on multiple benchmarks, including a +8.4% F1 improvement in style classification over GraphCLIP and a +7.1 BLEU@1 gain in captioning on ArtPedia. Qualitative analyses show that ArtSeek can interpret visual motifs, infer historical context, and retrieve relevant knowledge, even for obscure works. Though focused on visual arts, our approach generalizes to other domains requiring external knowledge, supporting scalable multimodal AI research. Both the dataset and the source code will be made publicly available at https://github.com/cilabuniba/artseek.
>
---
#### [new 078] VoluMe -- Authentic 3D Video Calls from Live Gaussian Splat Prediction
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D视频通信任务，旨在解决远程会议中缺乏真实3D交互的问题。现有方法依赖复杂硬件或预训练模型，限制了实用性。论文提出VoluMe方法，通过单个2D摄像头实时预测3D高斯重建，实现真实、稳定的3D视频通话，具备高视觉质量与时间稳定性，适用于普通设备，提升了远程沟通的沉浸感与真实感。**

- **链接: [http://arxiv.org/pdf/2507.21311v1](http://arxiv.org/pdf/2507.21311v1)**

> **作者:** Martin de La Gorce; Charlie Hewitt; Tibor Takacs; Robert Gerdisch; Zafiirah Hosenie; Givi Meishvili; Marek Kowalski; Thomas J. Cashman; Antonio Criminisi
>
> **摘要:** Virtual 3D meetings offer the potential to enhance copresence, increase engagement and thus improve effectiveness of remote meetings compared to standard 2D video calls. However, representing people in 3D meetings remains a challenge; existing solutions achieve high quality by using complex hardware, making use of fixed appearance via enrolment, or by inverting a pre-trained generative model. These approaches lead to constraints that are unwelcome and ill-fitting for videoconferencing applications. We present the first method to predict 3D Gaussian reconstructions in real time from a single 2D webcam feed, where the 3D representation is not only live and realistic, but also authentic to the input video. By conditioning the 3D representation on each video frame independently, our reconstruction faithfully recreates the input video from the captured viewpoint (a property we call authenticity), while generalizing realistically to novel viewpoints. Additionally, we introduce a stability loss to obtain reconstructions that are temporally stable on video sequences. We show that our method delivers state-of-the-art accuracy in visual quality and stability metrics compared to existing methods, and demonstrate our approach in live one-to-one 3D meetings using only a standard 2D camera and display. This demonstrates that our approach can allow anyone to communicate volumetrically, via a method for 3D videoconferencing that is not only highly accessible, but also realistic and authentic.
>
---
#### [new 079] Few-Shot Vision-Language Reasoning for Satellite Imagery via Verifiable Rewards
- **分类: cs.CV**

- **简介: 该论文属于遥感领域视觉-语言推理任务，旨在解决标注数据稀缺下的模型训练问题。提出了一种基于可验证奖励的少样本强化学习（RLVR）框架，无需依赖大量标注数据，仅用少量样本即可提升模型性能。实验表明该方法在多种遥感任务中表现优异，具备高效、通用的特点。**

- **链接: [http://arxiv.org/pdf/2507.21745v1](http://arxiv.org/pdf/2507.21745v1)**

> **作者:** Aybora Koksal; A. Aydin Alatan
>
> **备注:** ICCV 2025 Workshop on Curated Data for Efficient Learning (CDEL). 10 pages, 3 figures, 6 tables. Our model, training code and dataset will be at https://github.com/aybora/FewShotReasoning
>
> **摘要:** Recent advances in large language and vision-language models have enabled strong reasoning capabilities, yet they remain impractical for specialized domains like remote sensing, where annotated data is scarce and expensive. We present the first few-shot reinforcement learning with verifiable reward (RLVR) framework for satellite imagery that eliminates the need for caption supervision--relying solely on lightweight, rule-based binary or IoU-based rewards. Adapting the "1-shot RLVR" paradigm from language models to vision-language models, we employ policy-gradient optimization with as few as one curated example to align model outputs for satellite reasoning tasks. Comprehensive experiments across multiple remote sensing benchmarks--including classification, visual question answering, and grounding--show that even a single example yields substantial improvements over the base model. Scaling to 128 examples matches or exceeds models trained on thousands of annotated samples. While the extreme one-shot setting can induce mild, task-specific overfitting, our approach consistently demonstrates robust generalization and efficiency across diverse tasks. Further, we find that prompt design and loss weighting significantly influence training stability and final accuracy. Our method enables cost-effective and data-efficient development of domain-specialist vision-language reasoning models, offering a pragmatic recipe for data-scarce fields: start from a compact VLM, curate a handful of reward-checkable cases, and train via RLVR.
>
---
#### [new 080] Dual Cross-image Semantic Consistency with Self-aware Pseudo Labeling for Semi-supervised Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于半监督医学图像分割任务，旨在解决标注数据不足及伪标签不一致问题。方法提出双跨图像语义一致性框架（DuCiSC），通过区域级语义对齐和自感知置信估计提升模型性能，在多个医学数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.21440v1](http://arxiv.org/pdf/2507.21440v1)**

> **作者:** Han Wu; Chong Wang; Zhiming Cui
>
> **备注:** IEEE TMI
>
> **摘要:** Semi-supervised learning has proven highly effective in tackling the challenge of limited labeled training data in medical image segmentation. In general, current approaches, which rely on intra-image pixel-wise consistency training via pseudo-labeling, overlook the consistency at more comprehensive semantic levels (e.g., object region) and suffer from severe discrepancy of extracted features resulting from an imbalanced number of labeled and unlabeled data. To overcome these limitations, we present a new \underline{Du}al \underline{C}ross-\underline{i}mage \underline{S}emantic \underline{C}onsistency (DuCiSC) learning framework, for semi-supervised medical image segmentation. Concretely, beyond enforcing pixel-wise semantic consistency, DuCiSC proposes dual paradigms to encourage region-level semantic consistency across: 1) labeled and unlabeled images; and 2) labeled and fused images, by explicitly aligning their prototypes. Relying on the dual paradigms, DuCiSC can effectively establish consistent cross-image semantics via prototype representations, thereby addressing the feature discrepancy issue. Moreover, we devise a novel self-aware confidence estimation strategy to accurately select reliable pseudo labels, allowing for exploiting the training dynamics of unlabeled data. Our DuCiSC method is extensively validated on four datasets, including two popular binary benchmarks in segmenting the left atrium and pancreas, a multi-class Automatic Cardiac Diagnosis Challenge dataset, and a challenging scenario of segmenting the inferior alveolar nerve that features complicated anatomical structures, showing superior segmentation results over previous state-of-the-art approaches. Our code is publicly available at \href{https://github.com/ShanghaiTech-IMPACT/DuCiSC}{https://github.com/ShanghaiTech-IMPACT/DuCiSC}.
>
---
#### [new 081] MapDiffusion: Generative Diffusion for Vectorized Online HD Map Construction and Uncertainty Estimation in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 论文任务为自动驾驶中的在线矢量高精地图构建。解决传统方法无法捕捉环境不确定性与模糊性问题。工作提出MapDiffusion，采用扩散模型生成多可能地图样本，实现不确定性估计与精度提升，在nuScenes数据集上表现优越。**

- **链接: [http://arxiv.org/pdf/2507.21423v1](http://arxiv.org/pdf/2507.21423v1)**

> **作者:** Thomas Monninger; Zihan Zhang; Zhipeng Mo; Md Zafar Anwar; Steffen Staab; Sihao Ding
>
> **备注:** Accepted for 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Autonomous driving requires an understanding of the static environment from sensor data. Learned Bird's-Eye View (BEV) encoders are commonly used to fuse multiple inputs, and a vector decoder predicts a vectorized map representation from the latent BEV grid. However, traditional map construction models provide deterministic point estimates, failing to capture uncertainty and the inherent ambiguities of real-world environments, such as occlusions and missing lane markings. We propose MapDiffusion, a novel generative approach that leverages the diffusion paradigm to learn the full distribution of possible vectorized maps. Instead of predicting a single deterministic output from learned queries, MapDiffusion iteratively refines randomly initialized queries, conditioned on a BEV latent grid, to generate multiple plausible map samples. This allows aggregating samples to improve prediction accuracy and deriving uncertainty estimates that directly correlate with scene ambiguity. Extensive experiments on the nuScenes dataset demonstrate that MapDiffusion achieves state-of-the-art performance in online map construction, surpassing the baseline by 5% in single-sample performance. We further show that aggregating multiple samples consistently improves performance along the ROC curve, validating the benefit of distribution modeling. Additionally, our uncertainty estimates are significantly higher in occluded areas, reinforcing their value in identifying regions with ambiguous sensor input. By modeling the full map distribution, MapDiffusion enhances the robustness and reliability of online vectorized HD map construction, enabling uncertainty-aware decision-making for autonomous vehicles in complex environments.
>
---
#### [new 082] Unleashing the Power of Motion and Depth: A Selective Fusion Strategy for RGB-D Video Salient Object Detection
- **分类: cs.CV**

- **简介: 该论文属于RGB-D视频显著目标检测（RGB-D VSOD）任务，旨在解决如何有效融合光流与深度信息以辅助RGB模态进行显著目标检测的问题。论文提出了SMFNet框架，包含像素级选择性融合策略（PSF）与多维选择性注意力模块（MSAM），优化多模态特征融合，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.21857v1](http://arxiv.org/pdf/2507.21857v1)**

> **作者:** Jiahao He; Daerji Suolang; Keren Fu; Qijun Zhao
>
> **备注:** submitted to TMM on 11-Jun-2024, ID: MM-020522, still in peer review
>
> **摘要:** Applying salient object detection (SOD) to RGB-D videos is an emerging task called RGB-D VSOD and has recently gained increasing interest, due to considerable performance gains of incorporating motion and depth and that RGB-D videos can be easily captured now in daily life. Existing RGB-D VSOD models have different attempts to derive motion cues, in which extracting motion information explicitly from optical flow appears to be a more effective and promising alternative. Despite this, there remains a key issue that how to effectively utilize optical flow and depth to assist the RGB modality in SOD. Previous methods always treat optical flow and depth equally with respect to model designs, without explicitly considering their unequal contributions in individual scenarios, limiting the potential of motion and depth. To address this issue and unleash the power of motion and depth, we propose a novel selective cross-modal fusion framework (SMFNet) for RGB-D VSOD, incorporating a pixel-level selective fusion strategy (PSF) that achieves optimal fusion of optical flow and depth based on their actual contributions. Besides, we propose a multi-dimensional selective attention module (MSAM) to integrate the fused features derived from PSF with the remaining RGB modality at multiple dimensions, effectively enhancing feature representation to generate refined features. We conduct comprehensive evaluation of SMFNet against 19 state-of-the-art models on both RDVS and DVisal datasets, making the evaluation the most comprehensive RGB-D VSOD benchmark up to date, and it also demonstrates the superiority of SMFNet over other models. Meanwhile, evaluation on five video benchmark datasets incorporating synthetic depth validates the efficacy of SMFNet as well. Our code and benchmark results are made publicly available at https://github.com/Jia-hao999/SMFNet.
>
---
#### [new 083] Distribution-Based Masked Medical Vision-Language Model Using Structured Reports
- **分类: cs.CV**

- **简介: 该论文属于医学图像-文本预训练任务，旨在解决医疗数据中的不确定性和模糊性问题。作者提出了一种基于分布的掩码医学视觉-语言模型，利用大型语言模型生成的结构化文本报告，增强图像数据的临床语义信息，提升下游任务的性能。**

- **链接: [http://arxiv.org/pdf/2507.21794v1](http://arxiv.org/pdf/2507.21794v1)**

> **作者:** Shreyank N Gowda; Ruichi Zhang; Xiao Gu; Ying Weng; Lu Yang
>
> **备注:** Accepted in MICCAI-W 2025
>
> **摘要:** Medical image-language pre-training aims to align medical images with clinically relevant text to improve model performance on various downstream tasks. However, existing models often struggle with the variability and ambiguity inherent in medical data, limiting their ability to capture nuanced clinical information and uncertainty. This work introduces an uncertainty-aware medical image-text pre-training model that enhances generalization capabilities in medical image analysis. Building on previous methods and focusing on Chest X-Rays, our approach utilizes structured text reports generated by a large language model (LLM) to augment image data with clinically relevant context. These reports begin with a definition of the disease, followed by the `appearance' section to highlight critical regions of interest, and finally `observations' and `verdicts' that ground model predictions in clinical semantics. By modeling both inter- and intra-modal uncertainty, our framework captures the inherent ambiguity in medical images and text, yielding improved representations and performance on downstream tasks. Our model demonstrates significant advances in medical image-text pre-training, obtaining state-of-the-art performance on multiple downstream tasks.
>
---
#### [new 084] Boost Self-Supervised Dataset Distillation via Parameterization, Predefined Augmentation, and Approximation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于数据集蒸馏任务，旨在解决大规模数据集训练成本高的问题。通过自监督学习方法，将图像及其表示蒸馏为紧凑集合，提升跨架构泛化与迁移学习性能。提出了参数化、预定义增强和近似建模等技术，增强蒸馏效果。**

- **链接: [http://arxiv.org/pdf/2507.21455v1](http://arxiv.org/pdf/2507.21455v1)**

> **作者:** Sheng-Feng Yu; Jia-Jiun Yao; Wei-Chen Chiu
>
> **摘要:** Although larger datasets are crucial for training large deep models, the rapid growth of dataset size has brought a significant challenge in terms of considerable training costs, which even results in prohibitive computational expenses. Dataset Distillation becomes a popular technique recently to reduce the dataset size via learning a highly compact set of representative exemplars, where the model trained with these exemplars ideally should have comparable performance with respect to the one trained with the full dataset. While most of existing works upon dataset distillation focus on supervised datasets, we instead aim to distill images and their self-supervisedly trained representations into a distilled set. This procedure, named as Self-Supervised Dataset Distillation, effectively extracts rich information from real datasets, yielding the distilled sets with enhanced cross-architecture generalizability. Particularly, in order to preserve the key characteristics of original dataset more faithfully and compactly, several novel techniques are proposed: 1) we introduce an innovative parameterization upon images and representations via distinct low-dimensional bases, where the base selection for parameterization is experimentally shown to play a crucial role; 2) we tackle the instability induced by the randomness of data augmentation -- a key component in self-supervised learning but being underestimated in the prior work of self-supervised dataset distillation -- by utilizing predetermined augmentations; 3) we further leverage a lightweight network to model the connections among the representations of augmented views from the same image, leading to more compact pairs of distillation. Extensive experiments conducted on various datasets validate the superiority of our approach in terms of distillation efficiency, cross-architecture generalization, and transfer learning performance.
>
---
#### [new 085] TARS: MinMax Token-Adaptive Preference Strategy for Hallucination Reduction in MLLMs
- **分类: cs.CV**

- **简介: 该论文属于多模态大语言模型（MLLMs）中的幻觉抑制任务，旨在解决模型生成内容与视觉输入不一致或事实错误的问题。作者提出TARS方法，通过将偏好优化建模为Min-Max问题，动态调整词元级分布，减少对静态偏好模式的过拟合，从而提升因果推理能力并降低幻觉率。**

- **链接: [http://arxiv.org/pdf/2507.21584v1](http://arxiv.org/pdf/2507.21584v1)**

> **作者:** Kejia Zhang; Keda Tao; Zhiming Luo; Chang Liu; Jiasheng Tang; Huan Wang
>
> **摘要:** Multimodal large language models (MLLMs) enable vision-language reasoning, yet often generate plausible outputs that are factually incorrect or visually ungrounded, thereby compromising their reliability. Direct preference optimization (DPO) is a common strategy for correcting hallucinations by aligning model outputs with human preferences. Existing DPO strategies typically treat hallucination-related preferences as fixed targets, relying on static supervision signals during training. This approach tends to overfit to superficial linguistic cues in preference data, leading to distributional rigidity and spurious correlations that impair grounding in causally relevant visual information. To overcome this limitation, we propose TARS, a token-adaptive preference strategy that reformulates DPO as a min-max optimization problem. TARS maximizes token-level distributional shifts under semantic constraints to simulate alignment uncertainty, and simultaneously minimizes the expected preference loss under these controlled perturbations. This joint objective preserves causal grounding while mitigating overfitting to preference patterns, thereby reducing hallucinations in multimodal reasoning. We evaluate TARS on multiple hallucination benchmarks and find consistently strong performance. Using only 4.8k preference samples and no expert feedback, TARS reduces hallucination rates from 26.4% to 13.2% and decreases cognition value from 2.5 to 0.4. It outperforms standard DPO and matches GPT-4o on several key metrics.
>
---
#### [new 086] CAPE: A CLIP-Aware Pointing Ensemble of Complementary Heatmap Cues for Embodied Reference Understanding
- **分类: cs.CV**

- **简介: 该论文属于具身指代表达理解任务，旨在通过结合手势和语言准确识别说话人所指物体。现有方法难以有效利用视觉线索，论文提出双模型框架，分别学习头部-指尖与手腕-指尖方向，并引入高斯射线热图和CLIP感知集成模块，提升指代表达的理解效果。**

- **链接: [http://arxiv.org/pdf/2507.21888v1](http://arxiv.org/pdf/2507.21888v1)**

> **作者:** Fevziye Irem Eyiokur; Dogucan Yaman; Hazım Kemal Ekenel; Alexander Waibel
>
> **摘要:** We address the problem of Embodied Reference Understanding, which involves predicting the object that a person in the scene is referring to through both pointing gesture and language. Accurately identifying the referent requires multimodal understanding: integrating textual instructions, visual pointing, and scene context. However, existing methods often struggle to effectively leverage visual clues for disambiguation. We also observe that, while the referent is often aligned with the head-to-fingertip line, it occasionally aligns more closely with the wrist-to-fingertip line. Therefore, relying on a single line assumption can be overly simplistic and may lead to suboptimal performance. To address this, we propose a dual-model framework, where one model learns from the head-to-fingertip direction and the other from the wrist-to-fingertip direction. We further introduce a Gaussian ray heatmap representation of these lines and use them as input to provide a strong supervisory signal that encourages the model to better attend to pointing cues. To combine the strengths of both models, we present the CLIP-Aware Pointing Ensemble module, which performs a hybrid ensemble based on CLIP features. Additionally, we propose an object center prediction head as an auxiliary task to further enhance referent localization. We validate our approach through extensive experiments and analysis on the benchmark YouRefIt dataset, achieving an improvement of approximately 4 mAP at the 0.25 IoU threshold.
>
---
#### [new 087] VAGU & GtS: LLM-Based Benchmark and Framework for Joint Video Anomaly Grounding and Understanding
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视频异常检测任务，旨在同时解决异常理解与定位问题。现有方法多侧重单一方向，缺乏统一框架。作者提出VAGU基准，首次集成两类任务，并设计无需训练的GtS框架，结合新指标JeAUG，实现更全面的视频异常检测。**

- **链接: [http://arxiv.org/pdf/2507.21507v1](http://arxiv.org/pdf/2507.21507v1)**

> **作者:** Shibo Gao; Peipei Yang; Yangyang Liu; Yi Chen; Han Zhu; Xuyao Zhang; Linlin Huang
>
> **备注:** 21 pages, 19 figures, 8 tables
>
> **摘要:** Video Anomaly Detection (VAD) aims to identify anomalous events in videos and accurately determine their time intervals. Current VAD methods mainly fall into two categories: traditional DNN-based approaches that focus on temporal localization, and LLM-based approaches that emphasize semantic understanding. Both anomaly understanding and grounding are essential for comprehensive video anomaly detection and can complement each other. However, no existing model or dataset supports both tasks simultaneously. To address this, we introduce VAGU (Video Anomaly Grounding and Understanding), the first benchmark to integrate both tasks. Each VAGU instance includes annotations for anomaly category, semantic explanation, precise temporal grounding and Video QA. We also provide multiple-choice Video QA for objective evaluation. Based on this dataset, we propose Glance then Scrutinize (GtS), a training-free framework guided by textual prompts. The framework first enables coarse localization of high-probability anomalous regions, followed by detailed anomaly interpretation and temporal boundary refinement. Additionally, we propose the JeAUG metric, which jointly evaluates semantic interpretability and temporal precision, overcoming the limitations of traditional metrics. Extensive experiments verify the effectiveness of our benchmark, framework, and evaluation metric.
>
---
#### [new 088] Adversarial Reconstruction Feedback for Robust Fine-grained Generalization
- **分类: cs.CV**

- **简介: 该论文属于细粒度图像检索任务，旨在解决现有方法依赖预定义类别语义、泛化能力弱的问题。作者提出AdvRF框架，通过对抗重建反馈机制，实现类别无关的差异表示学习，提升对未见类别的检索性能。**

- **链接: [http://arxiv.org/pdf/2507.21742v1](http://arxiv.org/pdf/2507.21742v1)**

> **作者:** Shijie Wang; Jian Shi; Haojie Li
>
> **备注:** ICCV 2025
>
> **摘要:** Existing fine-grained image retrieval (FGIR) methods predominantly rely on supervision from predefined categories to learn discriminative representations for retrieving fine-grained objects. However, they inadvertently introduce category-specific semantics into the retrieval representation, creating semantic dependencies on predefined classes that critically hinder generalization to unseen categories. To tackle this, we propose AdvRF, a novel adversarial reconstruction feedback framework aimed at learning category-agnostic discrepancy representations. Specifically, AdvRF reformulates FGIR as a visual discrepancy reconstruction task via synergizing category-aware discrepancy localization from retrieval models with category-agnostic feature learning from reconstruction models. The reconstruction model exposes residual discrepancies overlooked by the retrieval model, forcing it to improve localization accuracy, while the refined signals from the retrieval model guide the reconstruction model to improve its reconstruction ability. Consequently, the retrieval model localizes visual differences, while the reconstruction model encodes these differences into category-agnostic representations. This representation is then transferred to the retrieval model through knowledge distillation for efficient deployment. Quantitative and qualitative evaluations demonstrate that our AdvRF achieves impressive performance on both widely-used fine-grained and coarse-grained datasets.
>
---
#### [new 089] A Deep Learning Pipeline Using Synthetic Data to Improve Interpretation of Paper ECG Images
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决纸质心电图（ECG）图像自动诊断问题。针对图像中的视觉噪声和细微波形模式识别难题，作者提出了一种深度学习框架，结合合成数据与真实数据，采用两阶段微调策略及ConvNeXt架构，实现了高精度的ECG图像分类。**

- **链接: [http://arxiv.org/pdf/2507.21968v1](http://arxiv.org/pdf/2507.21968v1)**

> **作者:** Xiaoyu Wang; Ramesh Nadarajah; Zhiqiang Zhang; David Wong
>
> **摘要:** Cardiovascular diseases (CVDs) are the leading global cause of death, and early detection is essential to improve patient outcomes. Electrocardiograms (ECGs), especially 12-lead ECGs, play a key role in the identification of CVDs. These are routinely interpreted by human experts, a process that is time-consuming and requires expert knowledge. Historical research in this area has focused on automatic ECG interpretation from digital signals, with recent deep learning approaches achieving strong results. In practice, however, most ECG data in clinical practice are stored or shared in image form. To bridge this gap, we propose a deep learning framework designed specifically to classify paper-like ECG images into five main diagnostic categories. Our method was the winning entry to the 2024 British Heart Foundation Open Data Science Challenge. It addresses two main challenges of paper ECG classification: visual noise (e.g., shadows or creases) and the need to detect fine-detailed waveform patterns. We propose a pre-processing pipeline that reduces visual noise and a two-stage fine-tuning strategy: the model is first fine-tuned on synthetic and external ECG image datasets to learn domain-specific features, and then further fine-tuned on the target dataset to enhance disease-specific recognition. We adopt the ConvNeXt architecture as the backbone of our model. Our method achieved AUROC scores of 0.9688 on the public validation set and 0.9677 on the private test set of the British Heart Foundation Open Data Science Challenge, highlighting its potential as a practical tool for automated ECG interpretation in clinical workflows.
>
---
#### [new 090] Contrast-Prior Enhanced Duality for Mask-Free Shadow Removal
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像处理任务，旨在解决无遮罩阴影去除问题。现有方法依赖难以获取的阴影掩码，而本文提出利用自适应门控双分支注意力机制和频域对比融合网络，动态优化对比先验信息，以更准确地区分阴影与低反射物体，提升阴影去除效果。**

- **链接: [http://arxiv.org/pdf/2507.21949v1](http://arxiv.org/pdf/2507.21949v1)**

> **作者:** Jiyu Wu; Yifan Liu; Jiancheng Huang; Mingfu Yan; Shifeng Chen
>
> **摘要:** Existing shadow removal methods often rely on shadow masks, which are challenging to acquire in real-world scenarios. Exploring intrinsic image cues, such as local contrast information, presents a potential alternative for guiding shadow removal in the absence of explicit masks. However, the cue's inherent ambiguity becomes a critical limitation in complex scenes, where it can fail to distinguish true shadows from low-reflectance objects and intricate background textures. To address this motivation, we propose the Adaptive Gated Dual-Branch Attention (AGBA) mechanism. AGBA dynamically filters and re-weighs the contrast prior to effectively disentangle shadow features from confounding visual elements. Furthermore, to tackle the persistent challenge of restoring soft shadow boundaries and fine-grained details, we introduce a diffusion-based Frequency-Contrast Fusion Network (FCFN) that leverages high-frequency and contrast cues to guide the generative process. Extensive experiments demonstrate that our method achieves state-of-the-art results among mask-free approaches while maintaining competitive performance relative to mask-based methods.
>
---
#### [new 091] Optimizing Active Learning in Vision-Language Models via Parameter-Efficient Uncertainty Calibration
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的主动学习任务，旨在降低标注成本。通过提出一种参数高效且结合不确定性校准的主动学习方法，优化数据采样过程，减少计算开销并提升性能。实验验证了方法在多个数据集上的有效性，并比较了Prompt学习与LoRA在样本选择中的效果。**

- **链接: [http://arxiv.org/pdf/2507.21521v1](http://arxiv.org/pdf/2507.21521v1)**

> **作者:** Athmanarayanan Lakshmi Narayanan; Amrutha Machireddy; Ranganath Krishnan
>
> **备注:** International Joint Conference on Neural Networks 2025 (Accepted)
>
> **摘要:** Active Learning (AL) has emerged as a powerful approach for minimizing labeling costs by selectively sampling the most informative data for neural network model development. Effective AL for large-scale vision-language models necessitates addressing challenges in uncertainty estimation and efficient sampling given the vast number of parameters involved. In this work, we introduce a novel parameter-efficient learning methodology that incorporates uncertainty calibration loss within the AL framework. We propose a differentiable loss function that promotes uncertainty calibration for effectively selecting fewer and most informative data samples for fine-tuning. Through extensive experiments across several datasets and vision backbones, we demonstrate that our solution can match and exceed the performance of complex feature-based sampling techniques while being computationally very efficient. Additionally, we investigate the efficacy of Prompt learning versus Low-rank adaptation (LoRA) in sample selection, providing a detailed comparative analysis of these methods in the context of efficient AL.
>
---
#### [new 092] ST-DAI: Single-shot 2.5D Spatial Transcriptomics with Intra-Sample Domain Adaptive Imputation for Cost-efficient 3D Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于空间转录组学任务，旨在解决3D空间转录组高成本和域差异问题。作者提出ST-DAI方法，通过2.5D采样降低成本，并设计单次3D填补模型结合域自适应策略，实现高效准确的3D基因表达预测。**

- **链接: [http://arxiv.org/pdf/2507.21516v1](http://arxiv.org/pdf/2507.21516v1)**

> **作者:** Jiahe Qian; Yaoyu Fang; Xinkun Wang; Lee A. Cooper; Bo Zhou
>
> **备注:** 21 pages, 4 figures, 3 tables, under review
>
> **摘要:** For 3D spatial transcriptomics (ST), the high per-section acquisition cost of fully sampling every tissue section remains a significant challenge. Although recent approaches predict gene expression from histology images, these methods require large external datasets, which leads to high-cost and suffers from substantial domain discrepancies that lead to poor generalization on new samples. In this work, we introduce ST-DAI, a single-shot framework for 3D ST that couples a cost-efficient 2.5D sampling scheme with an intra-sample domain-adaptive imputation framework. First, in the cost-efficient 2.5D sampling stage, one reference section (central section) is fully sampled while other sections (adjacent sections) is sparsely sampled, thereby capturing volumetric context at significantly reduced experimental cost. Second, we propose a single-shot 3D imputation learning method that allows us to generate fully sampled 3D ST from this cost-efficient 2.5D ST scheme, using only sample-specific training. We observe position misalignment and domain discrepancy between sections. To address those issues, we adopt a pipeline that first aligns the central section to the adjacent section, thereafter generates dense pseudo-supervision on the central section, and then performs Fast Multi-Domain Refinement (FMDR), which adapts the network to the domain of the adjacent section while fine-tuning only a few parameters through the use of Parameter-Efficient Domain-Alignment Layers (PDLs). During this refinement, a Confidence Score Generator (CSG) reweights the pseudo-labels according to their estimated reliability, thereby directing imputation toward trustworthy regions. Our experimental results demonstrate that ST-DAI achieves gene expression prediction performance comparable to fully sampled approaches while substantially reducing the measurement burden.
>
---
#### [new 093] Supervised Quantum Image Processing
- **分类: quant-ph; cs.AI; cs.CV; cs.LG; 81P68, 81P70, 81P40, 68Q12, 68T01; I.2; I.4; J.2**

- **简介: 该论文属于图像处理与量子计算交叉领域任务，旨在解决大数据背景下图像存储与分类效率问题。工作内容包括对比四种量子图像表示方法的压缩性能，并评估基于这些表示的量子核在分类任务中的准确率与资源消耗，结果显示量子核在准确率相当的情况下可大幅减少存储需求。**

- **链接: [http://arxiv.org/pdf/2507.22039v1](http://arxiv.org/pdf/2507.22039v1)**

> **作者:** Marco Parigi; Mehran Khosrojerdi; Filippo Caruso; Leonardo Banchi
>
> **备注:** 13 pages, 11 figures
>
> **摘要:** In the era of big data and artificial intelligence, the increasing volume of data and the demand to solve more and more complex computational challenges are two driving forces for improving the efficiency of data storage, processing and analysis. Quantum image processing (QIP) is an interdisciplinary field between quantum information science and image processing, which has the potential to alleviate some of these challenges by leveraging the power of quantum computing. In this work, we compare and examine the compression properties of four different Quantum Image Representations (QImRs): namely, Tensor Network Representation (TNR), Flexible Representation of Quantum Image (FRQI), Novel Enhanced Quantum Representation NEQR, and Quantum Probability Image Encoding (QPIE). Our simulations show that FRQI performs a higher compression of image information than TNR, NEQR, and QPIE. Furthermore, we investigate the trade-off between accuracy and memory in binary classification problems, evaluating the performance of quantum kernels based on QImRs compared to the classical linear kernel. Our results indicate that quantum kernels provide comparable classification average accuracy but require exponentially fewer resources for image storage.
>
---
#### [new 094] Page image classification for content-specific data processing
- **分类: cs.IR; cs.AI; cs.CV; 68T10, 68T09, 62H30; I.7.5; H.3.7**

- **简介: 该论文属于图像分类任务，旨在解决历史文档数字化中页面内容多样导致的手动处理效率低问题。作者开发了一个基于人工智能的分类系统，自动识别页面内容类型，如文本、图形和布局，从而实现针对性的数据处理与分析。**

- **链接: [http://arxiv.org/pdf/2507.21114v1](http://arxiv.org/pdf/2507.21114v1)**

> **作者:** Kateryna Lutsai; Pavel Straňák
>
> **备注:** 65 pages, 57 figures, 20 tables
>
> **摘要:** Digitization projects in humanities often generate vast quantities of page images from historical documents, presenting significant challenges for manual sorting and analysis. These archives contain diverse content, including various text types (handwritten, typed, printed), graphical elements (drawings, maps, photos), and layouts (plain text, tables, forms). Efficiently processing this heterogeneous data requires automated methods to categorize pages based on their content, enabling tailored downstream analysis pipelines. This project addresses this need by developing and evaluating an image classification system specifically designed for historical document pages, leveraging advancements in artificial intelligence and machine learning. The set of categories was chosen to facilitate content-specific processing workflows, separating pages requiring different analysis techniques (e.g., OCR for text, image analysis for graphics)
>
---
#### [new 095] Progressive Homeostatic and Plastic Prompt Tuning for Audio-Visual Multi-Task Incremental Learning
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于音频-视觉多任务增量学习任务，旨在持续学习多个相关任务而不需联合训练。论文提出渐进式稳态与可塑性提示调优方法（PHP），通过三个阶段设计解决知识遗忘与任务特异性平衡问题，实现跨任务与跨模态的有效学习。**

- **链接: [http://arxiv.org/pdf/2507.21588v1](http://arxiv.org/pdf/2507.21588v1)**

> **作者:** Jiong Yin; Liang Li; Jiehua Zhang; Yuhan Gao; Chenggang Yan; Xichun Sheng
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Audio-visual multi-task incremental learning aims to continuously learn from multiple audio-visual tasks without the need for joint training on all tasks. The challenge of the problem is how to preserve the old task knowledge while facilitating the learning of new task with previous experiences. To address these challenges, we introduce a three-stage Progressive Homeostatic and Plastic audio-visual prompt (PHP) method. In the shallow phase, we design the task-shared modality aggregating adapter to foster cross-task and cross-modal audio-visual representation learning to enhance shared understanding between tasks. In the middle phase, we propose the task-specific modality-shared dynamic generating adapter, which constructs prompts that are tailored to individual tasks while remaining general across modalities, which balances the models ability to retain knowledge against forgetting with its potential for versatile multi-task transferability. In the deep phase, we introduce the task-specific modality-independent prompts to further refine the understand ability by targeting individual information for each task and modality. By incorporating these three phases, PHP retains task-specific prompts while adapting shared parameters for new tasks to effectively balance knowledge sharing and specificity. Our method achieves SOTA performance in different orders of four tasks (AVE, AVVP, AVS and AVQA). Our code can be available at https://github.com/ENJOY-Yin-jiong/PHP.
>
---
#### [new 096] MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE
- **分类: cs.AI; cs.CV**

- **简介: 论文提出MixGRPO，属于图像生成中基于流匹配模型的人类偏好对齐任务，旨在解决现有方法（如FlowGRPO）因需优化所有去噪步骤导致的效率低下问题。工作包括设计混合ODE-SDE采样策略，引入滑动窗口机制，仅在窗口内进行SDE采样与优化，减少计算开销，并提出更高效的变体MixGRPO-Flash。**

- **链接: [http://arxiv.org/pdf/2507.21802v1](http://arxiv.org/pdf/2507.21802v1)**

> **作者:** Junzhe Li; Yutao Cui; Tao Huang; Yinping Ma; Chun Fan; Miles Yang; Zhao Zhong
>
> **摘要:** Although GRPO substantially enhances flow matching models in human preference alignment of image generation, methods such as FlowGRPO still exhibit inefficiency due to the necessity of sampling and optimizing over all denoising steps specified by the Markov Decision Process (MDP). In this paper, we propose $\textbf{MixGRPO}$, a novel framework that leverages the flexibility of mixed sampling strategies through the integration of stochastic differential equations (SDE) and ordinary differential equations (ODE). This streamlines the optimization process within the MDP to improve efficiency and boost performance. Specifically, MixGRPO introduces a sliding window mechanism, using SDE sampling and GRPO-guided optimization only within the window, while applying ODE sampling outside. This design confines sampling randomness to the time-steps within the window, thereby reducing the optimization overhead, and allowing for more focused gradient updates to accelerate convergence. Additionally, as time-steps beyond the sliding window are not involved in optimization, higher-order solvers are supported for sampling. So we present a faster variant, termed $\textbf{MixGRPO-Flash}$, which further improves training efficiency while achieving comparable performance. MixGRPO exhibits substantial gains across multiple dimensions of human preference alignment, outperforming DanceGRPO in both effectiveness and efficiency, with nearly 50% lower training time. Notably, MixGRPO-Flash further reduces training time by 71%. Codes and models are available at $\href{https://github.com/Tencent-Hunyuan/MixGRPO}{MixGRPO}$.
>
---
#### [new 097] Learning from Limited and Imperfect Data
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于机器学习任务，旨在解决现实世界中有限且不完美数据（如长尾分布、数据分布偏移）下的模型训练问题。论文提出了四种应对不同场景的深度学习算法，分别处理生成模型训练、尾类泛化、半监督优化和领域自适应，以提升模型在真实复杂数据上的性能。**

- **链接: [http://arxiv.org/pdf/2507.21205v1](http://arxiv.org/pdf/2507.21205v1)**

> **作者:** Harsh Rangwani
>
> **备注:** PhD Thesis
>
> **摘要:** The distribution of data in the world (eg, internet, etc.) significantly differs from the well-curated datasets and is often over-populated with samples from common categories. The algorithms designed for well-curated datasets perform suboptimally when used for learning from imperfect datasets with long-tailed imbalances and distribution shifts. To expand the use of deep models, it is essential to overcome the labor-intensive curation process by developing robust algorithms that can learn from diverse, real-world data distributions. Toward this goal, we develop practical algorithms for Deep Neural Networks which can learn from limited and imperfect data present in the real world. This thesis is divided into four segments, each covering a scenario of learning from limited or imperfect data. The first part of the thesis focuses on Learning Generative Models from Long-Tail Data, where we mitigate the mode-collapse and enable diverse aesthetic image generations for tail (minority) classes. In the second part, we enable effective generalization on tail classes through Inductive Regularization schemes, which allow tail classes to generalize as effectively as the head classes without requiring explicit generation of images. In the third part, we develop algorithms for Optimizing Relevant Metrics for learning from long-tailed data with limited annotation (semi-supervised), followed by the fourth part, which focuses on the Efficient Domain Adaptation of the model to various domains with very few to zero labeled samples.
>
---
#### [new 098] ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出了ReXGroundingCT，首个将自由文本放射学报告与3D胸部CT扫描中像素级分割手动标注关联的公开数据集。旨在解决医学AI中连接复杂描述性文本与三维空间解剖位置能力不足的关键问题。通过三阶段流程，利用GPT-4提取发现，并由专家进行手动分割，建立了用于开发和评估胸部CT自由文本分割模型的新基准。**

- **链接: [http://arxiv.org/pdf/2507.22030v1](http://arxiv.org/pdf/2507.22030v1)**

> **作者:** Mohammed Baharoon; Luyang Luo; Michael Moritz; Abhinav Kumar; Sung Eun Kim; Xiaoman Zhang; Miao Zhu; Mahmoud Hussain Alabbad; Maha Sbayel Alhazmi; Neel P. Mistry; Kent Ryan Kleinschmidt; Brady Chrisler; Sathvik Suryadevara; Sri Sai Dinesh Jaliparthi; Noah Michael Prudlo; Mark David Marino; Jeremy Palacio; Rithvik Akula; Hong-Yu Zhou; Ibrahim Ethem Hamamci; Scott J. Adams; Hassan Rayhan AlOmaish; Pranav Rajpurkar
>
> **摘要:** We present ReXGroundingCT, the first publicly available dataset to link free-text radiology findings with pixel-level segmentations in 3D chest CT scans that is manually annotated. While prior datasets have relied on structured labels or predefined categories, ReXGroundingCT captures the full expressiveness of clinical language represented in free text and grounds it to spatially localized 3D segmentation annotations in volumetric imaging. This addresses a critical gap in medical AI: the ability to connect complex, descriptive text, such as "3 mm nodule in the left lower lobe", to its precise anatomical location in three-dimensional space, a capability essential for grounded radiology report generation systems. The dataset comprises 3,142 non-contrast chest CT scans paired with standardized radiology reports from the CT-RATE dataset. Using a systematic three-stage pipeline, GPT-4 was used to extract positive lung and pleural findings, which were then manually segmented by expert annotators. A total of 8,028 findings across 16,301 entities were annotated, with quality control performed by board-certified radiologists. Approximately 79% of findings are focal abnormalities, while 21% are non-focal. The training set includes up to three representative segmentations per finding, while the validation and test sets contain exhaustive labels for each finding entity. ReXGroundingCT establishes a new benchmark for developing and evaluating sentence-level grounding and free-text medical segmentation models in chest CT. The dataset can be accessed at https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT.
>
---
#### [new 099] Comparative Analysis of Vision Transformers and Convolutional Neural Networks for Medical Image Classification
- **分类: eess.IV; cs.CV; cs.LG; I.2.10; I.4.8**

- **简介: 该论文属于医学图像分类任务，旨在比较卷积神经网络（CNN）与视觉Transformer（ViT）在医疗图像分析中的性能差异。研究通过评估四种先进模型在三种医学图像分类任务中的表现，揭示了不同任务下模型的优劣，为临床决策支持系统提供架构选择依据。**

- **链接: [http://arxiv.org/pdf/2507.21156v1](http://arxiv.org/pdf/2507.21156v1)**

> **作者:** Kunal Kawadkar
>
> **备注:** 9 pages, 8 figures, 3 tables. Submitted to IEEE Access
>
> **摘要:** The emergence of Vision Transformers (ViTs) has revolutionized computer vision, yet their effectiveness compared to traditional Convolutional Neural Networks (CNNs) in medical imaging remains under-explored. This study presents a comprehensive comparative analysis of CNN and ViT architectures across three critical medical imaging tasks: chest X-ray pneumonia detection, brain tumor classification, and skin cancer melanoma detection. We evaluated four state-of-the-art models - ResNet-50, EfficientNet-B0, ViT-Base, and DeiT-Small - across datasets totaling 8,469 medical images. Our results demonstrate task-specific model advantages: ResNet-50 achieved 98.37% accuracy on chest X-ray classification, DeiT-Small excelled at brain tumor detection with 92.16% accuracy, and EfficientNet-B0 led skin cancer classification at 81.84% accuracy. These findings provide crucial insights for practitioners selecting architectures for medical AI applications, highlighting the importance of task-specific architecture selection in clinical decision support systems.
>
---
#### [new 100] Querying GI Endoscopy Images: A VQA Approach
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像问答（VQA）任务，旨在解决通用模型在胃肠内镜图像问答中效果差的问题。作者探索了Florence2模型在此领域的适应性，并使用ROUGE、BLEU等指标评估其性能。**

- **链接: [http://arxiv.org/pdf/2507.21165v1](http://arxiv.org/pdf/2507.21165v1)**

> **作者:** Gaurav Parajuli
>
> **摘要:** VQA (Visual Question Answering) combines Natural Language Processing (NLP) with image understanding to answer questions about a given image. It has enormous potential for the development of medical diagnostic AI systems. Such a system can help clinicians diagnose gastro-intestinal (GI) diseases accurately and efficiently. Although many of the multimodal LLMs available today have excellent VQA capabilities in the general domain, they perform very poorly for VQA tasks in specialized domains such as medical imaging. This study is a submission for ImageCLEFmed-MEDVQA-GI 2025 subtask 1 that explores the adaptation of the Florence2 model to answer medical visual questions on GI endoscopy images. We also evaluate the model performance using standard metrics like ROUGE, BLEU and METEOR
>
---
#### [new 101] VidFuncta: Towards Generalizable Neural Representations for Ultrasound Videos
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学视频分析任务，旨在解决超声视频因采集不标准和操作者偏差导致的分析困难。论文提出VidFuncta框架，基于隐式神经表示（INRs）对超声视频进行高效建模，通过分离静态与动态特征实现视频重建与下游任务（如射血分数预测、乳腺病变分类等），验证了其泛化能力与有效性。**

- **链接: [http://arxiv.org/pdf/2507.21863v1](http://arxiv.org/pdf/2507.21863v1)**

> **作者:** Julia Wolleb; Florentin Bieder; Paul Friedrich; Hemant D. Tagare; Xenophon Papademetris
>
> **备注:** Accepted 6th International Workshop of Advances in Simplifying Medical UltraSound (ASMUS) to be held at MICCAI 2025
>
> **摘要:** Ultrasound is widely used in clinical care, yet standard deep learning methods often struggle with full video analysis due to non-standardized acquisition and operator bias. We offer a new perspective on ultrasound video analysis through implicit neural representations (INRs). We build on Functa, an INR framework in which each image is represented by a modulation vector that conditions a shared neural network. However, its extension to the temporal domain of medical videos remains unexplored. To address this gap, we propose VidFuncta, a novel framework that leverages Functa to encode variable-length ultrasound videos into compact, time-resolved representations. VidFuncta disentangles each video into a static video-specific vector and a sequence of time-dependent modulation vectors, capturing both temporal dynamics and dataset-level redundancies. Our method outperforms 2D and 3D baselines on video reconstruction and enables downstream tasks to directly operate on the learned 1D modulation vectors. We validate VidFuncta on three public ultrasound video datasets -- cardiac, lung, and breast -- and evaluate its downstream performance on ejection fraction prediction, B-line detection, and breast lesion classification. These results highlight the potential of VidFuncta as a generalizable and efficient representation framework for ultrasound videos. Our code is publicly available under https://github.com/JuliaWolleb/VidFuncta_public.
>
---
#### [new 102] Cardiac-CLIP: A Vision-Language Foundation Model for 3D Cardiac CT Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决心血管疾病诊断中多模态理解不足的问题。作者提出了Cardiac-CLIP，通过自监督和对比学习，结合3D心脏CT图像与文本报告，提升复杂心血管异常的分类、检索和临床分析性能。**

- **链接: [http://arxiv.org/pdf/2507.22024v1](http://arxiv.org/pdf/2507.22024v1)**

> **作者:** Yutao Hu; Ying Zheng; Shumei Miao; Xiaolei Zhang; Jiahao Xia; Yaolei Qi; Yiyang Zhang; Yuting He; Qian Chen; Jing Ye; Hongyan Qiao; Xiuhua Hu; Lei Xu; Jiayin Zhang; Hui Liu; Minwen Zheng; Yining Wang; Daimin Zhang; Ji Zhang; Wenqi Shao; Yun Liu; Longjiang Zhang; Guanyu Yang
>
> **摘要:** Foundation models have demonstrated remarkable potential in medical domain. However, their application to complex cardiovascular diagnostics remains underexplored. In this paper, we present Cardiac-CLIP, a multi-modal foundation model designed for 3D cardiac CT images. Cardiac-CLIP is developed through a two-stage pre-training strategy. The first stage employs a 3D masked autoencoder (MAE) to perform self-supervised representation learning from large-scale unlabeled volumetric data, enabling the visual encoder to capture rich anatomical and contextual features. In the second stage, contrastive learning is introduced to align visual and textual representations, facilitating cross-modal understanding. To support the pre-training, we collect 16641 real clinical CT scans, supplemented by 114k publicly available data. Meanwhile, we standardize free-text radiology reports into unified templates and construct the pathology vectors according to diagnostic attributes, based on which the soft-label matrix is generated to supervise the contrastive learning process. On the other hand, to comprehensively evaluate the effectiveness of Cardiac-CLIP, we collect 6,722 real-clinical data from 12 independent institutions, along with the open-source data to construct the evaluation dataset. Specifically, Cardiac-CLIP is comprehensively evaluated across multiple tasks, including cardiovascular abnormality classification, information retrieval and clinical analysis. Experimental results demonstrate that Cardiac-CLIP achieves state-of-the-art performance across various downstream tasks in both internal and external data. Particularly, Cardiac-CLIP exhibits great effectiveness in supporting complex clinical tasks such as the prospective prediction of acute coronary syndrome, which is notoriously difficult in real-world scenarios.
>
---
#### [new 103] Cyst-X: AI-Powered Pancreatic Cancer Risk Prediction from Multicenter MRI in Centralized and Federated Learning
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出Cyst-X，一种基于多中心MRI数据的AI框架，用于预测胰腺囊肿恶性风险。任务是医学图像分析中的疾病风险预测，旨在解决胰腺癌早期检测难题。通过集中式和联邦学习训练模型，性能优于现有指南和专家判断，并发布首个大规模多中心胰腺囊肿MRI数据集。**

- **链接: [http://arxiv.org/pdf/2507.22017v1](http://arxiv.org/pdf/2507.22017v1)**

> **作者:** Hongyi Pan; Gorkem Durak; Elif Keles; Deniz Seyithanoglu; Zheyuan Zhang; Alpay Medetalibeyoglu; Halil Ertugrul Aktas; Andrea Mia Bejar; Ziliang Hong; Yavuz Taktak; Gulbiz Dagoglu Kartal; Mehmet Sukru Erturk; Timurhan Cebeci; Maria Jaramillo Gonzalez; Yury Velichko; Lili Zhao; Emil Agarunov; Federica Proietto Salanitri; Concetto Spampinato; Pallavi Tiwari; Ziyue Xu; Sachin Jambawalikar; Ivo G. Schoots; Marco J. Bruno; Chenchang Huang; Candice Bolan; Tamas Gonda; Frank H. Miller; Rajesh N. Keswani; Michael B. Wallace; Ulas Bagci
>
> **摘要:** Pancreatic cancer is projected to become the second-deadliest malignancy in Western countries by 2030, highlighting the urgent need for better early detection. Intraductal papillary mucinous neoplasms (IPMNs), key precursors to pancreatic cancer, are challenging to assess with current guidelines, often leading to unnecessary surgeries or missed malignancies. We present Cyst-X, an AI framework that predicts IPMN malignancy using multicenter MRI data, leveraging MRI's superior soft tissue contrast over CT. Trained on 723 T1- and 738 T2-weighted scans from 764 patients across seven institutions, our models (AUC=0.82) significantly outperform both Kyoto guidelines (AUC=0.75) and expert radiologists. The AI-derived imaging features align with known clinical markers and offer biologically meaningful insights. We also demonstrate strong performance in a federated learning setting, enabling collaborative training without sharing patient data. To promote privacy-preserving AI development and improve IPMN risk stratification, the Cyst-X dataset is released as the first large-scale, multi-center pancreatic cysts MRI dataset.
>
---
#### [new 104] UI-AGILE: Advancing GUI Agents with Effective Reinforcement Learning and Precise Inference-Time Grounding
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于GUI智能代理任务，旨在提升代理在图形用户界面中的推理与定位能力。论文提出UI-AGILE框架，改进训练阶段的奖励机制与推理阶段的定位方法，解决了现有方法在复杂任务中定位不准、奖励稀疏等问题。**

- **链接: [http://arxiv.org/pdf/2507.22025v1](http://arxiv.org/pdf/2507.22025v1)**

> **作者:** Shuquan Lian; Yuhang Wu; Jia Ma; Zihan Song; Bingqi Chen; Xiawu Zheng; Hui Li
>
> **摘要:** The emergence of Multimodal Large Language Models (MLLMs) has driven significant advances in Graphical User Interface (GUI) agent capabilities. Nevertheless, existing GUI agent training and inference techniques still suffer from a dilemma for reasoning designs, ineffective reward, and visual noise. To address these issues, we introduce UI-AGILE, a comprehensive framework enhancing GUI agents at both the training and inference stages. For training, we propose a suite of improvements to the Supervised Fine-Tuning (SFT) process: 1) a Continuous Reward function to incentivize high-precision grounding; 2) a "Simple Thinking" reward to balance planning with speed and grounding accuracy; and 3) a Cropping-based Resampling strategy to mitigate the sparse reward problem and improve learning on complex tasks. For inference, we present Decomposed Grounding with Selection, a novel method that dramatically improves grounding accuracy on high-resolution displays by breaking the image into smaller, manageable parts. Experiments show that UI-AGILE achieves the state-of-the-art performance on two benchmarks ScreenSpot-Pro and ScreenSpot-v2. For instance, using both our proposed training and inference enhancement methods brings 23% grounding accuracy improvement over the best baseline on ScreenSpot-Pro.
>
---
#### [new 105] Unmasking Synthetic Realities in Generative AI: A Comprehensive Review of Adversarially Robust Deepfake Detection Systems
- **分类: cs.CR; cs.CV; F.2.2; I.2.7**

- **简介: 该论文属于深度伪造检测任务，旨在解决生成式AI带来的虚假内容泛滥问题。论文系统评估了现有检测方法，指出其在对抗性攻击下的脆弱性，并强调需提升模型的鲁棒性与跨模态适应能力。作者还提供了开源代码库以促进研究复现与测试。**

- **链接: [http://arxiv.org/pdf/2507.21157v1](http://arxiv.org/pdf/2507.21157v1)**

> **作者:** Naseem Khan; Tuan Nguyen; Amine Bermak; Issa Khalil
>
> **备注:** 27 pages, 4 Tables, 3 Figures
>
> **摘要:** The rapid advancement of Generative Artificial Intelligence has fueled deepfake proliferation-synthetic media encompassing fully generated content and subtly edited authentic material-posing challenges to digital security, misinformation mitigation, and identity preservation. This systematic review evaluates state-of-the-art deepfake detection methodologies, emphasizing reproducible implementations for transparency and validation. We delineate two core paradigms: (1) detection of fully synthetic media leveraging statistical anomalies and hierarchical feature extraction, and (2) localization of manipulated regions within authentic content employing multi-modal cues such as visual artifacts and temporal inconsistencies. These approaches, spanning uni-modal and multi-modal frameworks, demonstrate notable precision and adaptability in controlled settings, effectively identifying manipulations through advanced learning techniques and cross-modal fusion. However, comprehensive assessment reveals insufficient evaluation of adversarial robustness across both paradigms. Current methods exhibit vulnerability to adversarial perturbations-subtle alterations designed to evade detection-undermining reliability in real-world adversarial contexts. This gap highlights critical disconnect between methodological development and evolving threat landscapes. To address this, we contribute a curated GitHub repository aggregating open-source implementations, enabling replication and testing. Our findings emphasize urgent need for future work prioritizing adversarial resilience, advocating scalable, modality-agnostic architectures capable of withstanding sophisticated manipulations. This review synthesizes strengths and shortcomings of contemporary deepfake detection while charting paths toward robust trustworthy systems.
>
---
#### [new 106] PRISM: Programmatic Reasoning with Image Sequence Manipulation for LVLM Jailbreaking
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于视觉-语言模型安全任务，旨在解决现有安全机制无法防御复杂对抗攻击的问题。作者提出PRISM框架，受软件安全ROP技术启发，将有害指令分解为多个视觉组件，通过组合诱导模型生成有害输出。实验表明其攻击效果优于现有方法，揭示了LVLM推理过程中的潜在漏洞。**

- **链接: [http://arxiv.org/pdf/2507.21540v1](http://arxiv.org/pdf/2507.21540v1)**

> **作者:** Quanchen Zou; Zonghao Ying; Moyang Chen; Wenzhuo Xu; Yisong Xiao; Yakai Li; Deyue Zhang; Dongdong Yang; Zhao Liu; Xiangzheng Zhang
>
> **摘要:** The increasing sophistication of large vision-language models (LVLMs) has been accompanied by advances in safety alignment mechanisms designed to prevent harmful content generation. However, these defenses remain vulnerable to sophisticated adversarial attacks. Existing jailbreak methods typically rely on direct and semantically explicit prompts, overlooking subtle vulnerabilities in how LVLMs compose information over multiple reasoning steps. In this paper, we propose a novel and effective jailbreak framework inspired by Return-Oriented Programming (ROP) techniques from software security. Our approach decomposes a harmful instruction into a sequence of individually benign visual gadgets. A carefully engineered textual prompt directs the sequence of inputs, prompting the model to integrate the benign visual gadgets through its reasoning process to produce a coherent and harmful output. This makes the malicious intent emergent and difficult to detect from any single component. We validate our method through extensive experiments on established benchmarks including SafeBench and MM-SafetyBench, targeting popular LVLMs. Results show that our approach consistently and substantially outperforms existing baselines on state-of-the-art models, achieving near-perfect attack success rates (over 0.90 on SafeBench) and improving ASR by up to 0.39. Our findings reveal a critical and underexplored vulnerability that exploits the compositional reasoning abilities of LVLMs, highlighting the urgent need for defenses that secure the entire reasoning process.
>
---
#### [new 107] A Tactical Behaviour Recognition Framework Based on Causal Multimodal Reasoning: A Study on Covert Audio-Video Analysis Combining GAN Structure Enhancement and Phonetic Accent Modelling
- **分类: cs.CY; cs.AI; cs.CV; 05C82, 68T07, 68T05, 62H30; I.2.10; I.4.8; H.5.1; H.2.8**

- **简介: 论文提出TACTIC-GRAPHS框架，用于战术视频中的语义理解和威胁检测。该框架结合图神经网络与多模态推理，解决高噪声和弱结构环境下音视频分析与威胁识别问题。通过融合视觉、语音和动作信息，实现跨模态加权与因果信号分析，提升了威胁链识别准确率与系统可解释性。**

- **链接: [http://arxiv.org/pdf/2507.21100v1](http://arxiv.org/pdf/2507.21100v1)**

> **作者:** Wei Meng
>
> **备注:** This paper introduces a structurally innovative and mathematically rigorous framework for multimodal tactical reasoning, offering a significant advance in causal inference and graph-based threat recognition under noisy conditions
>
> **摘要:** This paper introduces TACTIC-GRAPHS, a system that combines spectral graph theory and multimodal graph neural reasoning for semantic understanding and threat detection in tactical video under high noise and weak structure. The framework incorporates spectral embedding, temporal causal edge modeling, and discriminative path inference across heterogeneous modalities. A semantic-aware keyframe extraction method fuses visual, acoustic, and action cues to construct temporal graphs. Using graph attention and Laplacian spectral mapping, the model performs cross-modal weighting and causal signal analysis. Experiments on TACTIC-AVS and TACTIC-Voice datasets show 89.3 percent accuracy in temporal alignment and over 85 percent recognition of complete threat chains, with node latency within plus-minus 150 milliseconds. The approach enhances structural interpretability and supports applications in surveillance, defense, and intelligent security systems.
>
---
#### [new 108] MoHoBench: Assessing Honesty of Multimodal Large Language Models via Unanswerable Visual Questions
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于多模态大语言模型（MLLM）诚实性评估任务，旨在解决模型在面对视觉上无法回答的问题时可能产生不可信内容的问题。论文构建了包含12k+样本的基准数据集MoHoBench，并对28个MLLM进行了诚实性评估，发现模型在需要拒绝回答时表现不佳，且视觉信息显著影响其诚实性。基于此，作者尝试通过监督学习和偏好学习方法提升模型的诚实表现。**

- **链接: [http://arxiv.org/pdf/2507.21503v1](http://arxiv.org/pdf/2507.21503v1)**

> **作者:** Yanxu Zhu; Shitong Duan; Xiangxu Zhang; Jitao Sang; Peng Zhang; Tun Lu; Xiao Zhou; Jing Yao; Xiaoyuan Yi; Xing Xie
>
> **摘要:** Recently Multimodal Large Language Models (MLLMs) have achieved considerable advancements in vision-language tasks, yet produce potentially harmful or untrustworthy content. Despite substantial work investigating the trustworthiness of language models, MMLMs' capability to act honestly, especially when faced with visually unanswerable questions, remains largely underexplored. This work presents the first systematic assessment of honesty behaviors across various MLLMs. We ground honesty in models' response behaviors to unanswerable visual questions, define four representative types of such questions, and construct MoHoBench, a large-scale MMLM honest benchmark, consisting of 12k+ visual question samples, whose quality is guaranteed by multi-stage filtering and human verification. Using MoHoBench, we benchmarked the honesty of 28 popular MMLMs and conducted a comprehensive analysis. Our findings show that: (1) most models fail to appropriately refuse to answer when necessary, and (2) MMLMs' honesty is not solely a language modeling issue, but is deeply influenced by visual information, necessitating the development of dedicated methods for multimodal honesty alignment. Therefore, we implemented initial alignment methods using supervised and preference learning to improve honesty behavior, providing a foundation for future work on trustworthy MLLMs. Our data and code can be found at https://github.com/DSTTSD/MoHoBench.
>
---
#### [new 109] Hot-Swap MarkBoard: An Efficient Black-box Watermarking Approach for Large-scale Model Distribution
- **分类: cs.CR; cs.AI; cs.CV**

- **简介: 论文提出Hot-Swap MarkBoard，一种高效的黑盒水印方法，用于大规模模型分发中的版权保护。该方法通过多分支LoRA模块嵌入用户特定水印，支持无需重新训练的快速定制，并防止水印被移除。适用于多种模型架构和任务，实验表明其在效率和适应性方面优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.20650v1](http://arxiv.org/pdf/2507.20650v1)**

> **作者:** Zhicheng Zhang; Peizhuo Lv; Mengke Wan; Jiang Fang; Diandian Guo; Yezeng Chen; Yinlong Liu; Wei Ma; Jiyan Sun; Liru Geng
>
> **摘要:** Recently, Deep Learning (DL) models have been increasingly deployed on end-user devices as On-Device AI, offering improved efficiency and privacy. However, this deployment trend poses more serious Intellectual Property (IP) risks, as models are distributed on numerous local devices, making them vulnerable to theft and redistribution. Most existing ownership protection solutions (e.g., backdoor-based watermarking) are designed for cloud-based AI-as-a-Service (AIaaS) and are not directly applicable to large-scale distribution scenarios, where each user-specific model instance must carry a unique watermark. These methods typically embed a fixed watermark, and modifying the embedded watermark requires retraining the model. To address these challenges, we propose Hot-Swap MarkBoard, an efficient watermarking method. It encodes user-specific $n$-bit binary signatures by independently embedding multiple watermarks into a multi-branch Low-Rank Adaptation (LoRA) module, enabling efficient watermark customization without retraining through branch swapping. A parameter obfuscation mechanism further entangles the watermark weights with those of the base model, preventing removal without degrading model performance. The method supports black-box verification and is compatible with various model architectures and DL tasks, including classification, image generation, and text generation. Extensive experiments across three types of tasks and six backbone models demonstrate our method's superior efficiency and adaptability compared to existing approaches, achieving 100\% verification accuracy.
>
---
#### [new 110] Research Challenges and Progress in the End-to-End V2X Cooperative Autonomous Driving Competition
- **分类: cs.RO; cs.CV; I.4.9**

- **简介: 该论文属于自动驾驶任务，旨在解决V2X协同驾驶中的多源数据融合与通信限制问题。论文组织了V2X协同自动驾驶挑战赛，设立两个赛道，基于UniV2X框架和V2X-Seq-SPD数据集，评估协同驾驶系统，分析关键技术趋势，推动可扩展、可靠V2X协同自动驾驶的发展。**

- **链接: [http://arxiv.org/pdf/2507.21610v1](http://arxiv.org/pdf/2507.21610v1)**

> **作者:** Ruiyang Hao; Haibao Yu; Jiaru Zhong; Chuanye Wang; Jiahao Wang; Yiming Kan; Wenxian Yang; Siqi Fan; Huilin Yin; Jianing Qiu; Yao Mu; Jiankai Sun; Li Chen; Walter Zimmer; Dandan Zhang; Shanghang Zhang; Mac Schwager; Wei Huang; Xiaobo Zhang; Ping Luo; Zaiqing Nie
>
> **备注:** 10 pages, 4 figures, accepted by ICCVW
>
> **摘要:** With the rapid advancement of autonomous driving technology, vehicle-to-everything (V2X) communication has emerged as a key enabler for extending perception range and enhancing driving safety by providing visibility beyond the line of sight. However, integrating multi-source sensor data from both ego-vehicles and infrastructure under real-world constraints, such as limited communication bandwidth and dynamic environments, presents significant technical challenges. To facilitate research in this area, we organized the End-to-End Autonomous Driving through V2X Cooperation Challenge, which features two tracks: cooperative temporal perception and cooperative end-to-end planning. Built on the UniV2X framework and the V2X-Seq-SPD dataset, the challenge attracted participation from over 30 teams worldwide and established a unified benchmark for evaluating cooperative driving systems. This paper describes the design and outcomes of the challenge, highlights key research problems including bandwidth-aware fusion, robust multi-agent planning, and heterogeneous sensor integration, and analyzes emerging technical trends among top-performing solutions. By addressing practical constraints in communication and data fusion, the challenge contributes to the development of scalable and reliable V2X-cooperative autonomous driving systems.
>
---
## 更新

#### [replaced 001] Bias Analysis for Synthetic Face Detection: A Case Study of the Impact of Facial Attributes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.19705v2](http://arxiv.org/pdf/2507.19705v2)**

> **作者:** Asmae Lamsaf; Lucia Cascone; Hugo Proença; João Neves
>
> **备注:** Accepted at IJCB2025
>
> **摘要:** Bias analysis for synthetic face detection is bound to become a critical topic in the coming years. Although many detection models have been developed and several datasets have been released to reliably identify synthetic content, one crucial aspect has been largely overlooked: these models and training datasets can be biased, leading to failures in detection for certain demographic groups and raising significant social, legal, and ethical issues. In this work, we introduce an evaluation framework to contribute to the analysis of bias of synthetic face detectors with respect to several facial attributes. This framework exploits synthetic data generation, with evenly distributed attribute labels, for mitigating any skew in the data that could otherwise influence the outcomes of bias analysis. We build on the proposed framework to provide an extensive case study of the bias level of five state-of-the-art detectors in synthetic datasets with 25 controlled facial attributes. While the results confirm that, in general, synthetic face detectors are biased towards the presence/absence of specific facial attributes, our study also sheds light on the origins of the observed bias through the analysis of the correlations with the balancing of facial attributes in the training sets of the detectors, and the analysis of detectors activation maps in image pairs with controlled attribute modifications.
>
---
#### [replaced 002] LoRA-Loop: Closing the Synthetic Replay Cycle for Continual VLM Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13568v2](http://arxiv.org/pdf/2507.13568v2)**

> **作者:** Kaihong Wang; Donghyun Kim; Margrit Betke
>
> **摘要:** Continual learning for vision-language models has achieved remarkable performance through synthetic replay, where samples are generated using Stable Diffusion to regularize during finetuning and retain knowledge. However, real-world downstream applications often exhibit domain-specific nuances and fine-grained semantics not captured by generators, causing synthetic-replay methods to produce misaligned samples that misguide finetuning and undermine retention of prior knowledge. In this work, we propose a LoRA-enhanced synthetic-replay framework that injects task-specific low-rank adapters into a frozen Stable Diffusion model, efficiently capturing each new task's unique visual and semantic patterns. Specifically, we introduce a two-stage, confidence-based sample selection: we first rank real task data by post-finetuning VLM confidence to focus LoRA finetuning on the most representative examples, then generate synthetic samples and again select them by confidence for distillation. Our approach integrates seamlessly with existing replay pipelines-simply swap in the adapted generator to boost replay fidelity. Extensive experiments on the Multi-domain Task Incremental Learning (MTIL) benchmark show that our method outperforms previous synthetic-replay techniques, achieving an optimal balance among plasticity, stability, and zero-shot capability. These results demonstrate the effectiveness of generator adaptation via LoRA for robust continual learning in VLMs.
>
---
#### [replaced 003] Addressing High Class Imbalance in Multi-Class Diabetic Retinopathy Severity Grading with Augmentation and Transfer Learning
- **分类: cs.CV; cs.LG; F.2.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2507.17121v2](http://arxiv.org/pdf/2507.17121v2)**

> **作者:** Faisal Ahmed
>
> **备注:** 9 pages, 1 Figure
>
> **摘要:** Diabetic retinopathy (DR) is a leading cause of vision loss worldwide, and early diagnosis through automated retinal image analysis can significantly reduce the risk of blindness. This paper presents a robust deep learning framework for both binary and five-class DR classification, leveraging transfer learning and extensive data augmentation to address the challenges of class imbalance and limited training data. We evaluate a range of pretrained convolutional neural network architectures, including variants of ResNet and EfficientNet, on the APTOS 2019 dataset. For binary classification, our proposed model achieves a state-of-the-art accuracy of 98.9%, with a precision of 98.6%, recall of 99.3%, F1-score of 98.9%, and an AUC of 99.4%. In the more challenging five-class severity classification task, our model obtains a competitive accuracy of 84.6% and an AUC of 94.1%, outperforming several existing approaches. Our findings also demonstrate that EfficientNet-B0 and ResNet34 offer optimal trade-offs between accuracy and computational efficiency across both tasks. These results underscore the effectiveness of combining class-balanced augmentation with transfer learning for high-performance DR diagnosis. The proposed framework provides a scalable and accurate solution for DR screening, with potential for deployment in real-world clinical environments.
>
---
#### [replaced 004] A Survey on Wi-Fi Sensing Generalizability: Taxonomy, Techniques, Datasets, and Future Research Prospects
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08008v2](http://arxiv.org/pdf/2503.08008v2)**

> **作者:** Fei Wang; Tingting Zhang; Wei Xi; Han Ding; Ge Wang; Di Zhang; Yuanhao Cui; Fan Liu; Jinsong Han; Jie Xu; Tony Xiao Han
>
> **备注:** Under Review; 30 pages, 322 references
>
> **摘要:** Wi-Fi sensing has emerged as a powerful non-intrusive technology for recognizing human activities, monitoring vital signs, and enabling context-aware applications using commercial wireless devices. However, the performance of Wi-Fi sensing often degrades when applied to new users, devices, or environments due to significant domain shifts. To address this challenge, researchers have proposed a wide range of generalization techniques aimed at enhancing the robustness and adaptability of Wi-Fi sensing systems. In this survey, we provide a comprehensive and structured review of over 200 papers published since 2015, categorizing them according to the Wi-Fi sensing pipeline: experimental setup, signal preprocessing, feature learning, and model deployment. We analyze key techniques, including signal preprocessing, domain adaptation, meta-learning, metric learning, data augmentation, cross-modal alignment, federated learning, and continual learning. Furthermore, we summarize publicly available datasets across various tasks,such as activity recognition, user identification, indoor localization, and pose estimation, and provide insights into their domain diversity. We also discuss emerging trends and future directions, including large-scale pretraining, integration with multimodal foundation models, and continual deployment. To foster community collaboration, we introduce the Sensing Dataset Platform (SDP) for sharing datasets and models. This survey aims to serve as a valuable reference and practical guide for researchers and practitioners dedicated to improving the generalizability of Wi-Fi sensing systems.
>
---
#### [replaced 005] SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.16268v3](http://arxiv.org/pdf/2410.16268v3)**

> **作者:** Shuangrui Ding; Rui Qian; Xiaoyi Dong; Pan Zhang; Yuhang Zang; Yuhang Cao; Yuwei Guo; Dahua Lin; Jiaqi Wang
>
> **备注:** ICCV 2025, Project page: https://mark12ding.github.io/project/SAM2Long/ ; github page: https://github.com/Mark12Ding/SAM2Long/
>
> **摘要:** The Segment Anything Model 2 (SAM 2) has emerged as a powerful foundation model for object segmentation in both images and videos, paving the way for various downstream video applications. The crucial design of SAM 2 for video segmentation is its memory module, which prompts object-aware memories from previous frames for current frame prediction. However, its greedy-selection memory design suffers from the "error accumulation" problem, where an errored or missed mask will cascade and influence the segmentation of the subsequent frames, which limits the performance of SAM 2 toward complex long-term videos. To this end, we introduce SAM2Long, an improved training-free video object segmentation strategy, which considers the segmentation uncertainty within each frame and chooses the video-level optimal results from multiple segmentation pathways in a constrained tree search manner. In practice, we maintain a fixed number of segmentation pathways throughout the video. For each frame, multiple masks are proposed based on the existing pathways, creating various candidate branches. We then select the same fixed number of branches with higher cumulative scores as the new pathways for the next frame. After processing the final frame, the pathway with the highest cumulative score is chosen as the final segmentation result. Benefiting from its heuristic search design, SAM2Long is robust toward occlusions and object reappearances, and can effectively segment and track objects for complex long-term videos. Notably, SAM2Long achieves an average improvement of 3.0 points across all 24 head-to-head comparisons, with gains of up to 5.3 points in J&F on long-term video object segmentation benchmarks such as SA-V and LVOS. The code is released at https://github.com/Mark12Ding/SAM2Long.
>
---
#### [replaced 006] YOLO-PRO: Enhancing Instance-Specific Object Detection with Full-Channel Global Self-Attention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.02348v3](http://arxiv.org/pdf/2503.02348v3)**

> **作者:** Lin Huang; Yujuan Tan; Weisheng Li; Shitai Shan; Liu Liu; Linlin Shen; Jing Yu; Yue Niu
>
> **摘要:** This paper addresses the inherent limitations of conventional bottleneck structures (diminished instance discriminability due to overemphasis on batch statistics) and decoupled heads (computational redundancy) in object detection frameworks by proposing two novel modules: the Instance-Specific Bottleneck with full-channel global self-attention (ISB) and the Instance-Specific Asymmetric Decoupled Head (ISADH). The ISB module innovatively reconstructs feature maps to establish an efficient full-channel global attention mechanism through synergistic fusion of batch-statistical and instance-specific features. Complementing this, the ISADH module pioneers an asymmetric decoupled architecture enabling hierarchical multi-dimensional feature integration via dual-stream batch-instance representation fusion. Extensive experiments on the MS-COCO benchmark demonstrate that the coordinated deployment of ISB and ISADH in the YOLO-PRO framework achieves state-of-the-art performance across all computational scales. Specifically, YOLO-PRO surpasses YOLOv8 by 1.0-1.6% AP (N/S/M/L/X scales) and outperforms YOLO11 by 0.1-0.5% AP in critical N/M/L/X groups, while maintaining competitive computational efficiency. This work provides practical insights for developing high-precision detectors deployable on edge devices.
>
---
#### [replaced 007] DreamScene: 3D Gaussian-based End-to-end Text-to-3D Scene Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13985v2](http://arxiv.org/pdf/2507.13985v2)**

> **作者:** Haoran Li; Yuli Tian; Kun Lan; Yong Liao; Lin Wang; Pan Hui; Peng Yuan Zhou
>
> **备注:** Extended version of ECCV 2024 paper "DreamScene"
>
> **摘要:** Generating 3D scenes from natural language holds great promise for applications in gaming, film, and design. However, existing methods struggle with automation, 3D consistency, and fine-grained control. We present DreamScene, an end-to-end framework for high-quality and editable 3D scene generation from text or dialogue. DreamScene begins with a scene planning module, where a GPT-4 agent infers object semantics and spatial constraints to construct a hybrid graph. A graph-based placement algorithm then produces a structured, collision-free layout. Based on this layout, Formation Pattern Sampling (FPS) generates object geometry using multi-timestep sampling and reconstructive optimization, enabling fast and realistic synthesis. To ensure global consistent, DreamScene employs a progressive camera sampling strategy tailored to both indoor and outdoor settings. Finally, the system supports fine-grained scene editing, including object movement, appearance changes, and 4D dynamic motion. Experiments demonstrate that DreamScene surpasses prior methods in quality, consistency, and flexibility, offering a practical solution for open-domain 3D content creation. Code and demos are available at https://jahnsonblack.github.io/DreamScene-Full/.
>
---
#### [replaced 008] Fuse Before Transfer: Knowledge Fusion for Heterogeneous Distillation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.12342v2](http://arxiv.org/pdf/2410.12342v2)**

> **作者:** Guopeng Li; Qiang Wang; Ke Yan; Shouhong Ding; Yuan Gao; Gui-Song Xia
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Most knowledge distillation (KD) methodologies predominantly focus on teacher-student pairs with similar architectures, such as both being convolutional neural networks (CNNs). However, the potential and flexibility of KD can be greatly improved by expanding it to novel Cross-Architecture KD (CAKD), where the knowledge of homogeneous and heterogeneous teachers can be transferred flexibly to a given student. The primary challenge in CAKD lies in the substantial feature gaps between heterogeneous models, originating from the distinction of their inherent inductive biases and module functions. To this end, we introduce an assistant model as a bridge to facilitate smooth feature knowledge transfer between heterogeneous teachers and students. More importantly, within our proposed design principle, the assistant model combines the advantages of cross-architecture inductive biases and module functions by merging convolution and attention modules derived from both student and teacher module functions. Furthermore, we observe that heterogeneous features exhibit diverse spatial distributions in CAKD, hindering the effectiveness of conventional pixel-wise mean squared error (MSE) loss. Therefore, we leverage a spatial-agnostic InfoNCE loss to align features after spatial smoothing, thereby improving the feature alignments in CAKD. Our proposed method is evaluated across some homogeneous model pairs and arbitrary heterogeneous combinations of CNNs, ViTs, and MLPs, achieving state-of-the-art performance for distilled models with a maximum gain of 11.47% on CIFAR-100 and 3.67% on ImageNet-1K. Our code and models will be released.
>
---
#### [replaced 009] Knowledge Regularized Negative Feature Tuning of Vision-Language Models for Out-of-Distribution Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.19847v2](http://arxiv.org/pdf/2507.19847v2)**

> **作者:** Wenjie Zhu; Yabin Zhang; Xin Jin; Wenjun Zeng; Lei Zhang
>
> **备注:** accepted by ACMMM 2025
>
> **摘要:** Out-of-distribution (OOD) detection is crucial for building reliable machine learning models. Although negative prompt tuning has enhanced the OOD detection capabilities of vision-language models, these tuned models often suffer from reduced generalization performance on unseen classes and styles. To address this challenge, we propose a novel method called Knowledge Regularized Negative Feature Tuning (KR-NFT), which integrates an innovative adaptation architecture termed Negative Feature Tuning (NFT) and a corresponding knowledge-regularization (KR) optimization strategy. Specifically, NFT applies distribution-aware transformations to pre-trained text features, effectively separating positive and negative features into distinct spaces. This separation maximizes the distinction between in-distribution (ID) and OOD images. Additionally, we introduce image-conditional learnable factors through a lightweight meta-network, enabling dynamic adaptation to individual images and mitigating sensitivity to class and style shifts. Compared to traditional negative prompt tuning, NFT demonstrates superior efficiency and scalability. To optimize this adaptation architecture, the KR optimization strategy is designed to enhance the discrimination between ID and OOD sets while mitigating pre-trained knowledge forgetting. This enhances OOD detection performance on trained ID classes while simultaneously improving OOD detection on unseen ID datasets. Notably, when trained with few-shot samples from ImageNet dataset, KR-NFT not only improves ID classification accuracy and OOD detection but also significantly reduces the FPR95 by 5.44\% under an unexplored generalization setting with unseen ID categories. Codes can be found at \href{https://github.com/ZhuWenjie98/KRNFT}.
>
---
#### [replaced 010] Fast Globally Optimal and Geometrically Consistent 3D Shape Matching
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06385v3](http://arxiv.org/pdf/2504.06385v3)**

> **作者:** Paul Roetzer; Florian Bernard
>
> **备注:** 8 pages main paper, 9 pages supplementary
>
> **摘要:** Geometric consistency, i.e. the preservation of neighbourhoods, is a natural and strong prior in 3D shape matching. Geometrically consistent matchings are crucial for many downstream applications, such as texture transfer or statistical shape modelling. Yet, in practice, geometric consistency is often overlooked, or only achieved under severely limiting assumptions (e.g. a good initialisation). In this work, we propose a novel formalism for computing globally optimal and geometrically consistent matchings between 3D shapes which is scalable in practice. Our key idea is to represent the surface of the source shape as a collection of cyclic paths, which are then consistently matched to the target shape. Mathematically, we construct a hyper product graph (between source and target shape), and then cast 3D shape matching as a minimum-cost circulation flow problem in this hyper graph, which yields global geometrically consistent matchings between both shapes. We empirically show that our formalism is efficiently solvable and that it leads to high-quality results.
>
---
#### [replaced 011] UncertainSAM: Fast and Efficient Uncertainty Quantification of the Segment Anything Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05049v4](http://arxiv.org/pdf/2505.05049v4)**

> **作者:** Timo Kaiser; Thomas Norrenbrock; Bodo Rosenhahn
>
> **备注:** Accepted to ICML'25
>
> **摘要:** The introduction of the Segment Anything Model (SAM) has paved the way for numerous semantic segmentation applications. For several tasks, quantifying the uncertainty of SAM is of particular interest. However, the ambiguous nature of the class-agnostic foundation model SAM challenges current uncertainty quantification (UQ) approaches. This paper presents a theoretically motivated uncertainty quantification model based on a Bayesian entropy formulation jointly respecting aleatoric, epistemic, and the newly introduced task uncertainty. We use this formulation to train USAM, a lightweight post-hoc UQ method. Our model traces the root of uncertainty back to under-parameterised models, insufficient prompts or image ambiguities. Our proposed deterministic USAM demonstrates superior predictive capabilities on the SA-V, MOSE, ADE20k, DAVIS, and COCO datasets, offering a computationally cheap and easy-to-use UQ alternative that can support user-prompting, enhance semi-supervised pipelines, or balance the tradeoff between accuracy and cost efficiency.
>
---
#### [replaced 012] Semantic segmentation of SEM images of lower bainitic and tempered martensitic steels
- **分类: cs.CV; cond-mat.mtrl-sci; cs.LG**

- **链接: [http://arxiv.org/pdf/2312.17251v2](http://arxiv.org/pdf/2312.17251v2)**

> **作者:** Xiaohan Bie; Manoj Arthanari; Evelin Barbosa de Melo; Baihua Ren; Juancheng Li; Stephen Yue; Salim Brahimi; Jun Song
>
> **摘要:** This study employs deep learning techniques to segment scanning electron microscope images, enabling a quantitative analysis of carbide precipitates in lower bainite and tempered martensite steels with comparable strength. Following segmentation, carbides are investigated, and their volume percentage, size distribution, and orientations are probed within the image dataset. Our findings reveal that lower bainite and tempered martensite exhibit comparable volume percentages of carbides, albeit with a more uniform distribution of carbides in tempered martensite. Carbides in lower bainite demonstrate a tendency for better alignment than those in tempered martensite, aligning with the observations of other researchers. However, both microstructures display a scattered carbide orientation, devoid of any discernible pattern. Comparative analysis of aspect ratios and sizes of carbides in lower bainite and tempered martensite unveils striking similarities. The deep learning model achieves an impressive pixelwise accuracy of 98.0% in classifying carbide/iron matrix at the individual pixel level. The semantic segmentation derived from deep learning extends its applicability to the analysis of secondary phases in various materials, offering a time-efficient, versatile AI-powered workflow for quantitative microstructure analysis.
>
---
#### [replaced 013] Latent Swap Joint Diffusion for 2D Long-Form Latent Generation
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.05130v3](http://arxiv.org/pdf/2502.05130v3)**

> **作者:** Yusheng Dai; Chenxi Wang; Chang Li; Chen Wang; Jun Du; Kewei Li; Ruoyu Wang; Jiefeng Ma; Lei Sun; Jianqing Gao
>
> **摘要:** This paper introduces Swap Forward (SaFa), a modality-agnostic and efficient method to generate seamless and coherence long spectrum and panorama through latent swap joint diffusion across multi-views. We first investigate the spectrum aliasing problem in spectrum-based audio generation caused by existing joint diffusion methods. Through a comparative analysis of the VAE latent representation of Mel-spectra and RGB images, we identify that the failure arises from excessive suppression of high-frequency components during the spectrum denoising process due to the averaging operator. To address this issue, we propose Self-Loop Latent Swap, a frame-level bidirectional swap applied to the overlapping region of adjacent views. Leveraging stepwise differentiated trajectories of adjacent subviews, this swap operator adaptively enhances high-frequency components and avoid spectrum distortion. Furthermore, to improve global cross-view consistency in non-overlapping regions, we introduce Reference-Guided Latent Swap, a unidirectional latent swap operator that provides a centralized reference trajectory to synchronize subview diffusions. By refining swap timing and intervals, we can achieve a cross-view similarity-diversity balance in a forward-only manner. Quantitative and qualitative experiments demonstrate that SaFa significantly outperforms existing joint diffusion methods and even training-based methods in audio generation using both U-Net and DiT models, along with effective longer length adaptation. It also adapts well to panorama generation, achieving comparable performance with 2 $\sim$ 20 $\times$ faster speed and greater model generalizability. More generation demos are available at https://swapforward.github.io/
>
---
#### [replaced 014] PPJudge: Towards Human-Aligned Assessment of Artistic Painting Process
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09242v2](http://arxiv.org/pdf/2507.09242v2)**

> **作者:** Shiqi Jiang; Xinpeng Li; Xi Mao; Changbo Wang; Chenhui Li
>
> **备注:** ACM International Conference on Multimedia 2025
>
> **摘要:** Artistic image assessment has become a prominent research area in computer vision. In recent years, the field has witnessed a proliferation of datasets and methods designed to evaluate the aesthetic quality of paintings. However, most existing approaches focus solely on static final images, overlooking the dynamic and multi-stage nature of the artistic painting process. To address this gap, we propose a novel framework for human-aligned assessment of painting processes. Specifically, we introduce the Painting Process Assessment Dataset (PPAD), the first large-scale dataset comprising real and synthetic painting process images, annotated by domain experts across eight detailed attributes. Furthermore, we present PPJudge (Painting Process Judge), a Transformer-based model enhanced with temporally-aware positional encoding and a heterogeneous mixture-of-experts architecture, enabling effective assessment of the painting process. Experimental results demonstrate that our method outperforms existing baselines in accuracy, robustness, and alignment with human judgment, offering new insights into computational creativity and art education.
>
---
#### [replaced 015] FIX-CLIP: Dual-Branch Hierarchical Contrastive Learning via Synthetic Captions for Better Understanding of Long Text
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.10095v2](http://arxiv.org/pdf/2507.10095v2)**

> **作者:** Bingchao Wang; Zhiwei Ning; Jianyu Ding; Xuanang Gao; Yin Li; Dongsheng Jiang; Jie Yang; Wei Liu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** CLIP has shown promising performance across many short-text tasks in a zero-shot manner. However, limited by the input length of the text encoder, CLIP struggles on under-stream tasks with long-text inputs ($>77$ tokens). To remedy this issue, we propose FIX-CLIP, which includes three novel modules: (1) A dual-branch training pipeline that aligns short and long texts with masked and raw images, respectively, which boosts the long-text representation while preserving the short-text ability. (2) Multiple learnable regional prompts with unidirectional masks in Transformer layers for regional information extraction. (3) A hierarchical feature alignment module in the intermediate encoder layers to promote the consistency of multi-scale features. Furthermore, we collect 30M images and utilize existing MLLMs to synthesize long-text captions for training. Extensive experiments show that FIX-CLIP achieves state-of-the-art performance on both long-text and short-text retrieval benchmarks. For downstream applications, we reveal that FIX-CLIP's text encoder delivers promising performance in a plug-and-play manner for diffusion models with long-text input. The code is available at https://github.com/bcwang-sjtu/Fix-CLIP.
>
---
#### [replaced 016] IRASim: A Fine-Grained World Model for Robot Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.14540v2](http://arxiv.org/pdf/2406.14540v2)**

> **作者:** Fangqi Zhu; Hongtao Wu; Song Guo; Yuxiao Liu; Chilam Cheang; Tao Kong
>
> **备注:** Opensource, project website: https://gen-irasim.github.io
>
> **摘要:** World models allow autonomous agents to plan and explore by predicting the visual outcomes of different actions. However, for robot manipulation, it is challenging to accurately model the fine-grained robot-object interaction within the visual space using existing methods which overlooks precise alignment between each action and the corresponding frame. In this paper, we present IRASim, a novel world model capable of generating videos with fine-grained robot-object interaction details, conditioned on historical observations and robot action trajectories. We train a diffusion transformer and introduce a novel frame-level action-conditioning module within each transformer block to explicitly model and strengthen the action-frame alignment. Extensive experiments show that: (1) the quality of the videos generated by our method surpasses all the baseline methods and scales effectively with increased model size and computation; (2) policy evaluations using IRASim exhibit a strong correlation with those using the ground-truth simulator, highlighting its potential to accelerate real-world policy evaluation; (3) testing-time scaling through model-based planning with IRASim significantly enhances policy performance, as evidenced by an improvement in the IoU metric on the Push-T benchmark from 0.637 to 0.961; (4) IRASim provides flexible action controllability, allowing virtual robotic arms in datasets to be controlled via a keyboard or VR controller.
>
---
#### [replaced 017] RANa: Retrieval-Augmented Navigation
- **分类: cs.CV; cs.IR; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.03524v2](http://arxiv.org/pdf/2504.03524v2)**

> **作者:** Gianluca Monaci; Rafael S. Rezende; Romain Deffayet; Gabriela Csurka; Guillaume Bono; Hervé Déjean; Stéphane Clinchant; Christian Wolf
>
> **摘要:** Methods for navigation based on large-scale learning typically treat each episode as a new problem, where the agent is spawned with a clean memory in an unknown environment. While these generalization capabilities to an unknown environment are extremely important, we claim that, in a realistic setting, an agent should have the capacity of exploiting information collected during earlier robot operations. We address this by introducing a new retrieval-augmented agent, trained with RL, capable of querying a database collected from previous episodes in the same environment and learning how to integrate this additional context information. We introduce a unique agent architecture for the general navigation task, evaluated on ImageNav, Instance-ImageNav and ObjectNav. Our retrieval and context encoding methods are data-driven and employ vision foundation models (FM) for both semantic and geometric understanding. We propose new benchmarks for these settings and we show that retrieval allows zero-shot transfer across tasks and environments while significantly improving performance.
>
---
#### [replaced 018] SCALAR: Scale-wise Controllable Visual Autoregressive Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.19946v2](http://arxiv.org/pdf/2507.19946v2)**

> **作者:** Ryan Xu; Dongyang Jin; Yancheng Bai; Rui Lan; Xu Duan; Lei Sun; Xiangxiang Chu
>
> **摘要:** Controllable image synthesis, which enables fine-grained control over generated outputs, has emerged as a key focus in visual generative modeling. However, controllable generation remains challenging for Visual Autoregressive (VAR) models due to their hierarchical, next-scale prediction style. Existing VAR-based methods often suffer from inefficient control encoding and disruptive injection mechanisms that compromise both fidelity and efficiency. In this work, we present SCALAR, a controllable generation method based on VAR, incorporating a novel Scale-wise Conditional Decoding mechanism. SCALAR leverages a pretrained image encoder to extract semantic control signal encodings, which are projected into scale-specific representations and injected into the corresponding layers of the VAR backbone. This design provides persistent and structurally aligned guidance throughout the generation process. Building on SCALAR, we develop SCALAR-Uni, a unified extension that aligns multiple control modalities into a shared latent space, supporting flexible multi-conditional guidance in a single model. Extensive experiments show that SCALAR achieves superior generation quality and control precision across various tasks.
>
---
#### [replaced 019] Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.17799v3](http://arxiv.org/pdf/2411.17799v3)**

> **作者:** Ronglai Zuo; Rolandos Alexandros Potamias; Evangelos Ververas; Jiankang Deng; Stefanos Zafeiriou
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Sign language is a visual language that encompasses all linguistic features of natural languages and serves as the primary communication method for the deaf and hard-of-hearing communities. Although many studies have successfully adapted pretrained language models (LMs) for sign language translation (sign-to-text), the reverse task-sign language generation (text-to-sign)-remains largely unexplored. In this work, we introduce a multilingual sign language model, Signs as Tokens (SOKE), which can generate 3D sign avatars autoregressively from text inputs using a pretrained LM. To align sign language with the LM, we leverage a decoupled tokenizer that discretizes continuous signs into token sequences representing various body parts. During decoding, unlike existing approaches that flatten all part-wise tokens into a single sequence and predict one token at a time, we propose a multi-head decoding method capable of predicting multiple tokens simultaneously. This approach improves inference efficiency while maintaining effective information fusion across different body parts. To further ease the generation process, we propose a retrieval-enhanced SLG approach, which incorporates external sign dictionaries to provide accurate word-level signs as auxiliary conditions, significantly improving the precision of generated signs. Extensive qualitative and quantitative evaluations demonstrate the effectiveness of SOKE.
>
---
#### [replaced 020] Diffusion Beats Autoregressive in Data-Constrained Settings
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.15857v3](http://arxiv.org/pdf/2507.15857v3)**

> **作者:** Mihir Prabhudesai; Mengning Wu; Amir Zadeh; Katerina Fragkiadaki; Deepak Pathak
>
> **备注:** Project Webpage: https://diffusion-scaling.github.io
>
> **摘要:** Autoregressive (AR) models have long dominated the landscape of large language models, driving progress across a wide range of tasks. Recently, diffusion-based language models have emerged as a promising alternative, though their advantages over AR models remain underexplored. In this paper, we systematically study masked diffusion models in data-constrained settings-where training involves repeated passes over limited data-and find that they significantly outperform AR models when compute is abundant but data is scarce. Diffusion models make better use of repeated data, achieving lower validation loss and superior downstream performance. We interpret this advantage as implicit data augmentation: masked diffusion exposes the model to a diverse distribution of token orderings and prediction tasks, unlike AR's fixed left-to-right factorization. We find new scaling laws for diffusion models and derive a closed-form expression for the critical compute threshold at which diffusion begins to outperform AR. These results suggest that when data, not compute, is the bottleneck, diffusion models offer a compelling alternative to the standard AR paradigm. Our code is available at: https://diffusion-scaling.github.io.
>
---
#### [replaced 021] Motion Diffusion Autoencoders: Enabling Attribute Manipulation in Human Motion Demonstrated on Karate Techniques
- **分类: cs.CV; cs.LG; 68T07 (Primary) 68T30, 92C99 (Secondary); I.2.4; I.2.6**

- **链接: [http://arxiv.org/pdf/2501.18729v2](http://arxiv.org/pdf/2501.18729v2)**

> **作者:** Anthony Richardson; Felix Putze
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Attribute manipulation deals with the problem of changing individual attributes of a data point or a time series, while leaving all other aspects unaffected. This work focuses on the domain of human motion, more precisely karate movement patterns. To the best of our knowledge, it presents the first success at manipulating attributes of human motion data. One of the key requirements for achieving attribute manipulation on human motion is a suitable pose representation. Therefore, we design a novel continuous, rotation-based pose representation that enables the disentanglement of the human skeleton and the motion trajectory, while still allowing an accurate reconstruction of the original anatomy. The core idea of the manipulation approach is to use a transformer encoder for discovering high-level semantics, and a diffusion probabilistic model for modeling the remaining stochastic variations. We show that the embedding space obtained from the transformer encoder is semantically meaningful and linear. This enables the manipulation of high-level attributes, by discovering their linear direction of change in the semantic embedding space and moving the embedding along said direction. All code and data is made publicly available.
>
---
#### [replaced 022] InfiniteYou: Flexible Photo Recrafting While Preserving Your Identity
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.16418v2](http://arxiv.org/pdf/2503.16418v2)**

> **作者:** Liming Jiang; Qing Yan; Yumin Jia; Zichuan Liu; Hao Kang; Xin Lu
>
> **备注:** ICCV 2025 (Highlight). Project page: https://bytedance.github.io/InfiniteYou/ Code and model: https://github.com/bytedance/InfiniteYou
>
> **摘要:** Achieving flexible and high-fidelity identity-preserved image generation remains formidable, particularly with advanced Diffusion Transformers (DiTs) like FLUX. We introduce InfiniteYou (InfU), one of the earliest robust frameworks leveraging DiTs for this task. InfU addresses significant issues of existing methods, such as insufficient identity similarity, poor text-image alignment, and low generation quality and aesthetics. Central to InfU is InfuseNet, a component that injects identity features into the DiT base model via residual connections, enhancing identity similarity while maintaining generation capabilities. A multi-stage training strategy, including pretraining and supervised fine-tuning (SFT) with synthetic single-person-multiple-sample (SPMS) data, further improves text-image alignment, ameliorates image quality, and alleviates face copy-pasting. Extensive experiments demonstrate that InfU achieves state-of-the-art performance, surpassing existing baselines. In addition, the plug-and-play design of InfU ensures compatibility with various existing methods, offering a valuable contribution to the broader community.
>
---
#### [replaced 023] Posture-Driven Action Intent Inference for Playing style and Fatigue Assessment
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.11642v2](http://arxiv.org/pdf/2507.11642v2)**

> **作者:** Abhishek Jaiswal; Nisheeth Srivastava
>
> **摘要:** Posture-based mental state inference has significant potential in diagnosing fatigue, preventing injury, and enhancing performance across various domains. Such tools must be research-validated with large datasets before being translated into practice. Unfortunately, such vision diagnosis faces serious challenges due to the sensitivity of human subject data. To address this, we identify sports settings as a viable alternative for accumulating data from human subjects experiencing diverse emotional states. We test our hypothesis in the game of cricket and present a posture-based solution to identify human intent from activity videos. Our method achieves over 75\% F1 score and over 80\% AUC-ROC in discriminating aggressive and defensive shot intent through motion analysis. These findings indicate that posture leaks out strong signals for intent inference, even with inherent noise in the data pipeline. Furthermore, we utilize existing data statistics as weak supervision to validate our findings, offering a potential solution for overcoming data labelling limitations. This research contributes to generalizable techniques for sports analytics and also opens possibilities for applying human behavior analysis across various fields.
>
---
#### [replaced 024] Back Home: A Computer Vision Solution to Seashell Identification for Ecological Restoration
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.04873v4](http://arxiv.org/pdf/2501.04873v4)**

> **作者:** Alexander Valverde; Luis Solano; André Montoya
>
> **备注:** ICCV 2025 (CV4E Workshop)
>
> **摘要:** Illegal souvenir collection strips an estimated five tonnes of seashells from Costa Rica's beaches each year. Yet, once these specimens are seized, their coastal origin -- Pacific or Caribbean -- cannot be verified easily due to the lack of information, preventing their return when confiscated by local authorities. To solve this issue, we introduce BackHome19K, the first large-scale image corpus (19,058 photographs, 516 species) annotated with coast-level labels, and propose a lightweight pipeline that infers provenance in real time on a mobile-grade CPU. A trained anomaly filter pre-screens uploads, increasing robustness to user-generated noise. On a held-out test set, the classifier attains 86.3% balanced accuracy, while the filter rejects 93% of 180 out-of-domain objects with zero false negatives. Deployed as a web application, the system has already processed 70,000 shells for wildlife officers in under three seconds per image, enabling confiscated specimens to be safely repatriated to their native ecosystems. The dataset is available at https://huggingface.co/datasets/FIFCO/BackHome19K
>
---
#### [replaced 025] Enhancing Glass Defect Detection with Diffusion Models: Addressing Imbalanced Datasets in Manufacturing Quality Control
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.03134v3](http://arxiv.org/pdf/2505.03134v3)**

> **作者:** Sajjad Rezvani Boroujeni; Hossein Abedi; Tom Bush
>
> **备注:** 12 pages, 7 figures, published in Computer and Decision Making - An International Journal (COMDEM)
>
> **摘要:** Visual defect detection in industrial glass manufacturing remains a critical challenge due to the low frequency of defective products, leading to imbalanced datasets that limit the performance of deep learning models and computer vision systems. This paper presents a novel approach using Denoising Diffusion Probabilistic Models (DDPMs) to generate synthetic defective glass product images for data augmentation, effectively addressing class imbalance issues in manufacturing quality control and automated visual inspection. The methodology significantly enhances image classification performance of standard CNN architectures (ResNet50V2, EfficientNetB0, and MobileNetV2) in detecting anomalies by increasing the minority class representation. Experimental results demonstrate substantial improvements in key machine learning metrics, particularly in recall for defective samples across all tested deep neural network architectures while maintaining perfect precision on the validation set. The most dramatic improvement was observed in ResNet50V2's overall classification accuracy, which increased from 78\% to 93\% when trained with the augmented data. This work provides a scalable, cost-effective approach to enhancing automated defect detection in glass manufacturing that can potentially be extended to other industrial quality assurance systems and industries with similar class imbalance challenges.
>
---
#### [replaced 026] LinkTo-Anime: A 2D Animation Optical Flow Dataset from 3D Model Rendering
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.02733v2](http://arxiv.org/pdf/2506.02733v2)**

> **作者:** Xiaoyi Feng; Kaifeng Zou; Caichun Cen; Tao Huang; Hui Guo; Zizhou Huang; Yingli Zhao; Mingqing Zhang; Ziyuan Zheng; Diwei Wang; Yuntao Zou; Dagang Li
>
> **摘要:** Existing optical flow datasets focus primarily on real-world simulation or synthetic human motion, but few are tailored to Celluloid(cel) anime character motion: a domain with unique visual and motion characteristics. To bridge this gap and facilitate research in optical flow estimation and downstream tasks such as anime video generation and line drawing colorization, we introduce LinkTo-Anime, the first high-quality dataset specifically designed for cel anime character motion generated with 3D model rendering. LinkTo-Anime provides rich annotations including forward and backward optical flow, occlusion masks, and Mixamo Skeleton. The dataset comprises 395 video sequences, totally 24,230 training frames, 720 validation frames, and 4,320 test frames. Furthermore, a comprehensive benchmark is constructed with various optical flow estimation methods to analyze the shortcomings and limitations across multiple datasets.
>
---
#### [replaced 027] DASH: 4D Hash Encoding with Self-Supervised Decomposition for Real-Time Dynamic Scene Rendering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.19141v2](http://arxiv.org/pdf/2507.19141v2)**

> **作者:** Jie Chen; Zhangchi Hu; Peixi Wu; Huyue Zhu; Hebei Li; Xiaoyan Sun
>
> **备注:** ICCV 2025
>
> **摘要:** Dynamic scene reconstruction is a long-term challenge in 3D vision. Existing plane-based methods in dynamic Gaussian splatting suffer from an unsuitable low-rank assumption, causing feature overlap and poor rendering quality. Although 4D hash encoding provides an explicit representation without low-rank constraints, directly applying it to the entire dynamic scene leads to substantial hash collisions and redundancy. To address these challenges, we present DASH, a real-time dynamic scene rendering framework that employs 4D hash encoding coupled with self-supervised decomposition. Our approach begins with a self-supervised decomposition mechanism that separates dynamic and static components without manual annotations or precomputed masks. Next, we introduce a multiresolution 4D hash encoder for dynamic elements, providing an explicit representation that avoids the low-rank assumption. Finally, we present a spatio-temporal smoothness regularization strategy to mitigate unstable deformation artifacts. Experiments on real-world datasets demonstrate that DASH achieves state-of-the-art dynamic rendering performance, exhibiting enhanced visual quality at real-time speeds of 264 FPS on a single 4090 GPU. Code: https://github.com/chenj02/DASH.
>
---
#### [replaced 028] Humanoid Occupancy: Enabling A Generalized Multimodal Occupancy Perception System on Humanoid Robots
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20217v2](http://arxiv.org/pdf/2507.20217v2)**

> **作者:** Wei Cui; Haoyu Wang; Wenkang Qin; Yijie Guo; Gang Han; Wen Zhao; Jiahang Cao; Zhang Zhang; Jiaru Zhong; Jingkai Sun; Pihai Sun; Shuai Shi; Botuo Jiang; Jiahao Ma; Jiaxu Wang; Hao Cheng; Zhichao Liu; Yang Wang; Zheng Zhu; Guan Huang; Jian Tang; Qiang Zhang
>
> **备注:** Tech Report
>
> **摘要:** Humanoid robot technology is advancing rapidly, with manufacturers introducing diverse heterogeneous visual perception modules tailored to specific scenarios. Among various perception paradigms, occupancy-based representation has become widely recognized as particularly suitable for humanoid robots, as it provides both rich semantic and 3D geometric information essential for comprehensive environmental understanding. In this work, we present Humanoid Occupancy, a generalized multimodal occupancy perception system that integrates hardware and software components, data acquisition devices, and a dedicated annotation pipeline. Our framework employs advanced multi-modal fusion techniques to generate grid-based occupancy outputs encoding both occupancy status and semantic labels, thereby enabling holistic environmental understanding for downstream tasks such as task planning and navigation. To address the unique challenges of humanoid robots, we overcome issues such as kinematic interference and occlusion, and establish an effective sensor layout strategy. Furthermore, we have developed the first panoramic occupancy dataset specifically for humanoid robots, offering a valuable benchmark and resource for future research and development in this domain. The network architecture incorporates multi-modal feature fusion and temporal information integration to ensure robust perception. Overall, Humanoid Occupancy delivers effective environmental perception for humanoid robots and establishes a technical foundation for standardizing universal visual modules, paving the way for the widespread deployment of humanoid robots in complex real-world scenarios.
>
---
#### [replaced 029] NarrLV: Towards a Comprehensive Narrative-Centric Evaluation for Long Video Generation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.11245v2](http://arxiv.org/pdf/2507.11245v2)**

> **作者:** X. Feng; H. Yu; M. Wu; S. Hu; J. Chen; C. Zhu; J. Wu; X. Chu; K. Huang
>
> **备注:** Project Page: https://amap-ml.github.io/NarrLV-Website/
>
> **摘要:** With the rapid development of foundation video generation technologies, long video generation models have exhibited promising research potential thanks to expanded content creation space. Recent studies reveal that the goal of long video generation tasks is not only to extend video duration but also to accurately express richer narrative content within longer videos. However, due to the lack of evaluation benchmarks specifically designed for long video generation models, the current assessment of these models primarily relies on benchmarks with simple narrative prompts (e.g., VBench). To the best of our knowledge, our proposed NarrLV is the first benchmark to comprehensively evaluate the Narrative expression capabilities of Long Video generation models. Inspired by film narrative theory, (i) we first introduce the basic narrative unit maintaining continuous visual presentation in videos as Temporal Narrative Atom (TNA), and use its count to quantitatively measure narrative richness. Guided by three key film narrative elements influencing TNA changes, we construct an automatic prompt generation pipeline capable of producing evaluation prompts with a flexibly expandable number of TNAs. (ii) Then, based on the three progressive levels of narrative content expression, we design an effective evaluation metric using the MLLM-based question generation and answering framework. (iii) Finally, we conduct extensive evaluations on existing long video generation models and the foundation generation models. Experimental results demonstrate that our metric aligns closely with human judgments. The derived evaluation outcomes reveal the detailed capability boundaries of current video generation models in narrative content expression.
>
---
#### [replaced 030] LookCloser: Frequency-aware Radiance Field for Tiny-Detail Scene
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18513v3](http://arxiv.org/pdf/2503.18513v3)**

> **作者:** Xiaoyu Zhang; Weihong Pan; Chong Bao; Xiyu Zhang; Xiaojun Xiang; Hanqing Jiang; Hujun Bao
>
> **备注:** Accepted to CVPR 2025. Project page: https://coscatter.github.io/LookCloser
>
> **摘要:** Humans perceive and comprehend their surroundings through information spanning multiple frequencies. In immersive scenes, people naturally scan their environment to grasp its overall structure while examining fine details of objects that capture their attention. However, current NeRF frameworks primarily focus on modeling either high-frequency local views or the broad structure of scenes with low-frequency information, which is limited to balancing both. We introduce FA-NeRF, a novel frequency-aware framework for view synthesis that simultaneously captures the overall scene structure and high-definition details within a single NeRF model. To achieve this, we propose a 3D frequency quantification method that analyzes the scene's frequency distribution, enabling frequency-aware rendering. Our framework incorporates a frequency grid for fast convergence and querying, a frequency-aware feature re-weighting strategy to balance features across different frequency contents. Extensive experiments show that our method significantly outperforms existing approaches in modeling entire scenes while preserving fine details. Project page: https://coscatter.github.io/LookCloser/
>
---
#### [replaced 031] Image Captioning via Compact Bidirectional Architecture
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2201.01984v2](http://arxiv.org/pdf/2201.01984v2)**

> **作者:** Zijie Song; Yuanen Zhou; Zhenzhen Hu; Daqing Liu; Huixia Ben; Richang Hong; Meng Wang
>
> **摘要:** Most current image captioning models typically generate captions from left-to-right. This unidirectional property makes them can only leverage past context but not future context. Though refinement-based models can exploit both past and future context by generating a new caption in the second stage based on pre-retrieved or pre-generated captions in the first stage, the decoder of these models generally consists of two networks~(i.e. a retriever or captioner in the first stage and a captioner in the second stage), which can only be executed sequentially. In this paper, we introduce a Compact Bidirectional Transformer model for image captioning that can leverage bidirectional context implicitly and explicitly while the decoder can be executed parallelly. Specifically, it is implemented by tightly coupling left-to-right(L2R) and right-to-left(R2L) flows into a single compact model to serve as a regularization for implicitly exploiting bidirectional context and optionally allowing explicit interaction of the bidirectional flows, while the final caption is chosen from either L2R or R2L flow in a sentence-level ensemble manner. We conduct extensive ablation studies on MSCOCO benchmark and find that the compact bidirectional architecture and the sentence-level ensemble play more important roles than the explicit interaction mechanism. By combining with word-level ensemble seamlessly, the effect of sentence-level ensemble is further enlarged. We further extend the conventional one-flow self-critical training to the two-flows version under this architecture and achieve new state-of-the-art results in comparison with non-vision-language-pretraining models. Finally, we verify the generality of this compact bidirectional architecture by extending it to LSTM backbone. Source code is available at https://github.com/YuanEZhou/cbtic.
>
---
#### [replaced 032] GLIMPSE: Holistic Cross-Modal Explainability for Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.18985v3](http://arxiv.org/pdf/2506.18985v3)**

> **作者:** Guanxi Shen
>
> **备注:** Keywords: Explainable Computer Vision, Large Vision-Language Models, AI Interpretability, Explainable AI, Visual Saliency, Attribution Maps, Cross-Modal Attribution, Human Attention Alignment, AI Transparency
>
> **摘要:** Recent large vision-language models (LVLMs) have advanced capabilities in visual question answering (VQA). However, interpreting where LVLMs direct their visual attention remains a significant challenge, yet is essential for understanding model behavior. We introduce GLIMPSE (Gradient-Layer Importance Mapping for Prompted Visual Saliency Explanation), a lightweight, model-agnostic framework that jointly attributes LVLM outputs to the most relevant visual evidence and textual signals that support open-ended generation. GLIMPSE fuses gradient-weighted attention, adaptive layer propagation, and relevance-weighted token aggregation to produce holistic response-level heat maps for interpreting cross-modal reasoning, outperforming prior methods in faithfulness and pushing the state-of-the-art in human-attention alignment. We demonstrate an analytic approach to uncover fine-grained insights into LVLM cross-modal attribution, trace reasoning dynamics, analyze systematic misalignment, diagnose hallucination and bias, and ensure transparency.
>
---
#### [replaced 033] VLM-CPL: Consensus Pseudo Labels from Vision-Language Models for Annotation-Free Pathological Image Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.15836v3](http://arxiv.org/pdf/2403.15836v3)**

> **作者:** Lanfeng Zhong; Zongyao Huang; Yang Liu; Wenjun Liao; Shichuan Zhang; Guotai Wang; Shaoting Zhang
>
> **备注:** Accepted at TMI
>
> **摘要:** Classification of pathological images is the basis for automatic cancer diagnosis. Despite that deep learning methods have achieved remarkable performance, they heavily rely on labeled data, demanding extensive human annotation efforts. In this study, we present a novel human annotation-free method by leveraging pre-trained Vision-Language Models (VLMs). Without human annotation, pseudo-labels of the training set are obtained by utilizing the zero-shot inference capabilities of VLM, which may contain a lot of noise due to the domain gap between the pre-training and target datasets. To address this issue, we introduce VLM-CPL, a novel approach that contains two noisy label filtering techniques with a semi-supervised learning strategy. Specifically, we first obtain prompt-based pseudo-labels with uncertainty estimation by zero-shot inference with the VLM using multiple augmented views of an input. Then, by leveraging the feature representation ability of VLM, we obtain feature-based pseudo-labels via sample clustering in the feature space. Prompt-feature consensus is introduced to select reliable samples based on the consensus between the two types of pseudo-labels. We further propose High-confidence Cross Supervision by to learn from samples with reliable pseudo-labels and the remaining unlabeled samples. Additionally, we present an innovative open-set prompting strategy that filters irrelevant patches from whole slides to enhance the quality of selected patches. Experimental results on five public pathological image datasets for patch-level and slide-level classification showed that our method substantially outperformed zero-shot classification by VLMs, and was superior to existing noisy label learning methods. The code is publicly available at https://github.com/HiLab-git/VLM-CPL.
>
---
#### [replaced 034] DEPTHOR: Depth Enhancement from a Practical Light-Weight dToF Sensor and RGB Image
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.01596v2](http://arxiv.org/pdf/2504.01596v2)**

> **作者:** Jijun Xiang; Xuan Zhu; Xianqi Wang; Yu Wang; Hong Zhang; Fei Guo; Xin Yang
>
> **备注:** 16 pages, 15 figures, 7 tables
>
> **摘要:** Depth enhancement, which uses RGB images as guidance to convert raw signals from dToF into high-precision, dense depth maps, is a critical task in computer vision. Although existing super-resolution-based methods show promising results on public datasets, they often rely on idealized assumptions like accurate region correspondences and reliable dToF inputs, overlooking calibration errors that cause misalignment and anomaly signals inherent to dToF imaging, limiting real-world applicability. To address these challenges, we propose a novel completion-based method, named DEPTHOR, featuring advances in both the training strategy and model architecture. First, we propose a method to simulate real-world dToF data from the accurate ground truth in synthetic datasets to enable noise-robust training. Second, we design a novel network that incorporates monocular depth estimation (MDE), leveraging global depth relationships and contextual information to improve prediction in challenging regions. On the ZJU-L5 dataset, our training strategy significantly enhances depth completion models, achieving results comparable to depth super-resolution methods, while our model achieves state-of-the-art results, improving Rel and RMSE by 27% and 18%, respectively. On a more challenging set of dToF samples we collected, our method outperforms SOTA methods on preliminary stereo-based GT, improving Rel and RMSE by 23% and 22%, respectively. Our Code is available at https://github.com/ShadowBbBb/Depthor
>
---
#### [replaced 035] Generalizable Neural Electromagnetic Inverse Scattering
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2506.21349v3](http://arxiv.org/pdf/2506.21349v3)**

> **作者:** Yizhe Cheng; Chunxun Tian; Haoru Wang; Wentao Zhu; Xiaoxuan Ma; Yizhou Wang
>
> **摘要:** Solving Electromagnetic Inverse Scattering Problems (EISP) is fundamental in applications such as medical imaging, where the goal is to reconstruct the relative permittivity from scattered electromagnetic field. This inverse process is inherently ill-posed and highly nonlinear, making it particularly challenging. A recent machine learning-based approach, Img-Interiors, shows promising results by leveraging continuous implicit functions. However, it requires case-specific optimization, lacks generalization to unseen data, and fails under sparse transmitter setups (e.g., with only one transmitter). To address these limitations, we revisit EISP from a physics-informed perspective, reformulating it as a two stage inverse transmission-scattering process. This formulation reveals the induced current as a generalizable intermediate representation, effectively decoupling the nonlinear scattering process from the ill-posed inverse problem. Built on this insight, we propose the first generalizable physics-driven framework for EISP, comprising a current estimator and a permittivity solver, working in an end-to-end manner. The current estimator explicitly learns the induced current as a physical bridge between the incident and scattered field, while the permittivity solver computes the relative permittivity directly from the estimated induced current. This design enables data-driven training and generalizable feed-forward prediction of relative permittivity on unseen data while maintaining strong robustness to transmitter sparsity. Extensive experiments show that our method outperforms state-of-the-art approaches in reconstruction accuracy, generalization, and robustness. This work offers a fundamentally new perspective on electromagnetic inverse scattering and represents a major step toward cost-effective practical solutions for electromagnetic imaging.
>
---
#### [replaced 036] Few-shot Online Anomaly Detection and Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.18201v2](http://arxiv.org/pdf/2403.18201v2)**

> **作者:** Shenxing Wei; Xing Wei; Zhiheng Ma; Songlin Dong; Shaochen Zhang; Yihong Gong
>
> **摘要:** Detecting anomaly patterns from images is a crucial artificial intelligence technique in industrial applications. Recent research in this domain has emphasized the necessity of a large volume of training data, overlooking the practical scenario where, post-deployment of the model, unlabeled data containing both normal and abnormal samples can be utilized to enhance the model's performance. Consequently, this paper focuses on addressing the challenging yet practical few-shot online anomaly detection and segmentation (FOADS) task. Under the FOADS framework, models are trained on a few-shot normal dataset, followed by inspection and improvement of their capabilities by leveraging unlabeled streaming data containing both normal and abnormal samples simultaneously. To tackle this issue, we propose modeling the feature distribution of normal images using a Neural Gas network, which offers the flexibility to adapt the topology structure to identify outliers in the data flow. In order to achieve improved performance with limited training samples, we employ multi-scale feature embedding extracted from a CNN pre-trained on ImageNet to obtain a robust representation. Furthermore, we introduce an algorithm that can incrementally update parameters without the need to store previous samples. Comprehensive experimental results demonstrate that our method can achieve substantial performance under the FOADS setting, while ensuring that the time complexity remains within an acceptable range on MVTec AD and BTAD datasets.
>
---
#### [replaced 037] DIVE: Taming DINO for Subject-Driven Video Editing
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.03347v2](http://arxiv.org/pdf/2412.03347v2)**

> **作者:** Yi Huang; Wei Xiong; He Zhang; Chaoqi Chen; Jianzhuang Liu; Mingfu Yan; Shifeng Chen
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Building on the success of diffusion models in image generation and editing, video editing has recently gained substantial attention. However, maintaining temporal consistency and motion alignment still remains challenging. To address these issues, this paper proposes DINO-guided Video Editing (DIVE), a framework designed to facilitate subject-driven editing in source videos conditioned on either target text prompts or reference images with specific identities. The core of DIVE lies in leveraging the powerful semantic features extracted from a pretrained DINOv2 model as implicit correspondences to guide the editing process. Specifically, to ensure temporal motion consistency, DIVE employs DINO features to align with the motion trajectory of the source video. For precise subject editing, DIVE incorporates the DINO features of reference images into a pretrained text-to-image model to learn Low-Rank Adaptations (LoRAs), effectively registering the target subject's identity. Extensive experiments on diverse real-world videos demonstrate that our framework can achieve high-quality editing results with robust motion consistency, highlighting the potential of DINO to contribute to video editing. Project page: https://dino-video-editing.github.io
>
---
#### [replaced 038] Differential-UMamba: Rethinking Tumor Segmentation Under Limited Data Scenarios
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.18177v2](http://arxiv.org/pdf/2507.18177v2)**

> **作者:** Dhruv Jain; Romain Modzelewski; Romain Herault; Clement Chatelain; Eva Torfeh; Sebastien Thureau
>
> **摘要:** In data-scarce scenarios, deep learning models often overfit to noise and irrelevant patterns, which limits their ability to generalize to unseen samples. To address these challenges in medical image segmentation, we introduce Diff-UMamba, a novel architecture that combines the UNet framework with the mamba mechanism to model long-range dependencies. At the heart of Diff-UMamba is a noise reduction module, which employs a signal differencing strategy to suppress noisy or irrelevant activations within the encoder. This encourages the model to filter out spurious features and enhance task-relevant representations, thereby improving its focus on clinically significant regions. As a result, the architecture achieves improved segmentation accuracy and robustness, particularly in low-data settings. Diff-UMamba is evaluated on multiple public datasets, including medical segmentation decathalon dataset (lung and pancreas) and AIIB23, demonstrating consistent performance gains of 1-3% over baseline methods in various segmentation tasks. To further assess performance under limited data conditions, additional experiments are conducted on the BraTS-21 dataset by varying the proportion of available training samples. The approach is also validated on a small internal non-small cell lung cancer dataset for the segmentation of gross tumor volume in cone beam CT, where it achieves a 4-5% improvement over baseline.
>
---
#### [replaced 039] Sparfels: Fast Reconstruction from Sparse Unposed Imagery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02178v3](http://arxiv.org/pdf/2505.02178v3)**

> **作者:** Shubhendu Jena; Amine Ouasfi; Mae Younes; Adnane Boukhayma
>
> **备注:** ICCV 2025. Project page : https://shubhendu-jena.github.io/Sparfels-web/
>
> **摘要:** We present a method for Sparse view reconstruction with surface element splatting that runs within 3 minutes on a consumer grade GPU. While few methods address sparse radiance field learning from noisy or unposed sparse cameras, shape recovery remains relatively underexplored in this setting. Several radiance and shape learning test-time optimization methods address the sparse posed setting by learning data priors or using combinations of external monocular geometry priors. Differently, we propose an efficient and simple pipeline harnessing a single recent 3D foundation model. We leverage its various task heads, notably point maps and camera initializations to instantiate a bundle adjusting 2D Gaussian Splatting (2DGS) model, and image correspondences to guide camera optimization midst 2DGS training. Key to our contribution is a novel formulation of splatted color variance along rays, which can be computed efficiently. Reducing this moment in training leads to more accurate shape reconstructions. We demonstrate state-of-the-art performances in the sparse uncalibrated setting in reconstruction and novel view benchmarks based on established multi-view datasets.
>
---
#### [replaced 040] SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.14811v3](http://arxiv.org/pdf/2507.14811v3)**

> **作者:** Jiaji Zhang; Ruichao Sun; Hailiang Zhao; Jiaju Wu; Peng Chen; Hao Li; Yuying Liu; Kingsum Chow; Gang Xiong; Shuiguang Deng
>
> **摘要:** Diffusion models have demonstrated exceptional generative capabilities but are computationally intensive, posing significant challenges for deployment in resource-constrained or latency-sensitive environments. Quantization offers an effective means to reduce model size and computational cost, with post-training quantization (PTQ) being particularly appealing due to its compatibility with pre-trained models without requiring retraining or training data. However, existing PTQ methods for diffusion models often rely on architecture-specific heuristics that limit their generalizability and hinder integration with industrial deployment pipelines. To address these limitations, we propose SegQuant, a unified quantization framework that adaptively combines complementary techniques to enhance cross-model versatility. SegQuant consists of a segment-aware, graph-based quantization strategy (SegLinear) that captures structural semantics and spatial heterogeneity, along with a dual-scale quantization scheme (DualScale) that preserves polarity-asymmetric activations, which is crucial for maintaining visual fidelity in generated outputs. SegQuant is broadly applicable beyond Transformer-based diffusion models, achieving strong performance while ensuring seamless compatibility with mainstream deployment tools.
>
---
#### [replaced 041] UniPaint: Unified Space-time Video Inpainting via Mixture-of-Experts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.06340v2](http://arxiv.org/pdf/2412.06340v2)**

> **作者:** Zhen Wan; Chenyang Qi; Zhiheng Liu; Tao Gui; Yue Ma
>
> **备注:** ICCV 1st Workshop on Human-Interactive Generation and Editing (poster)
>
> **摘要:** In this paper, we present UniPaint, a unified generative space-time video inpainting framework that enables spatial-temporal inpainting and interpolation. Different from existing methods that treat video inpainting and video interpolation as two distinct tasks, we leverage a unified inpainting framework to tackle them and observe that these two tasks can mutually enhance synthesis performance. Specifically, we first introduce a plug-and-play space-time video inpainting adapter, which can be employed in various personalized models. The key insight is to propose a Mixture of Experts (MoE) attention to cover various tasks. Then, we design a spatial-temporal masking strategy during the training stage to mutually enhance each other and improve performance. UniPaint produces high-quality and aesthetically pleasing results, achieving the best quantitative results across various tasks and scale setups. The code and checkpoints are available at $\href{https://github.com/mmmmm-w/UniPaint}{this \ repository}$.
>
---
#### [replaced 042] AI-ming backwards: Vanishing archaeological landscapes in Mesopotamia and automatic detection of sites on CORONA imagery
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.13420v2](http://arxiv.org/pdf/2507.13420v2)**

> **作者:** Alessandro Pistola; Valentina Orru'; Nicolo' Marchetti; Marco Roccetti
>
> **备注:** 25 pages, 9 Figures
>
> **摘要:** By upgrading an existing deep learning model with the knowledge provided by one of the oldest sets of grayscale satellite imagery, known as CORONA, we improved the AI model attitude towards the automatic identification of archaeological sites in an environment which has been completely transformed in the last five decades, including the complete destruction of many of those same sites. The initial Bing based convolutional network model was retrained using CORONA satellite imagery for the district of Abu Ghraib, west of Baghdad, central Mesopotamian floodplain. The results were twofold and surprising. First, the detection precision obtained on the area of interest increased sensibly: in particular, the Intersection over Union (IoU) values, at the image segmentation level, surpassed 85 percent, while the general accuracy in detecting archeological sites reached 90 percent. Second, our retrained model allowed the identification of four new sites of archaeological interest (confirmed through field verification), previously not identified by archaeologists with traditional techniques. This has confirmed the efficacy of using AI techniques and the CORONA imagery from the 1960 to discover archaeological sites currently no longer visible, a concrete breakthrough with significant consequences for the study of landscapes with vanishing archaeological evidence induced by anthropization
>
---
#### [replaced 043] Language Driven Occupancy Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16072v2](http://arxiv.org/pdf/2411.16072v2)**

> **作者:** Zhu Yu; Bowen Pang; Lizhe Liu; Runmin Zhang; Qiang Li; Si-Yuan Cao; Maochun Luo; Mingxia Chen; Sheng Yang; Hui-Liang Shen
>
> **备注:** ICCV 2025; Project Page: https://github.com/pkqbajng/LOcc
>
> **摘要:** We introduce LOcc, an effective and generalizable framework for open-vocabulary occupancy (OVO) prediction. Previous approaches typically supervise the networks through coarse voxel-to-text correspondences via image features as intermediates or noisy and sparse correspondences from voxel-based model-view projections. To alleviate the inaccurate supervision, we propose a semantic transitive labeling pipeline to generate dense and fine-grained 3D language occupancy ground truth. Our pipeline presents a feasible way to dig into the valuable semantic information of images, transferring text labels from images to LiDAR point clouds and ultimately to voxels, to establish precise voxel-to-text correspondences. By replacing the original prediction head of supervised occupancy models with a geometry head for binary occupancy states and a language head for language features, LOcc effectively uses the generated language ground truth to guide the learning of 3D language volume. Through extensive experiments, we demonstrate that our transitive semantic labeling pipeline can produce more accurate pseudo-labeled ground truth, diminishing labor-intensive human annotations. Additionally, we validate LOcc across various architectures, where all models consistently outperform state-of-the-art zero-shot occupancy prediction approaches on the Occ3D-nuScenes dataset.
>
---
#### [replaced 044] An Integrated Approach to Robotic Object Grasping and Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.13205v3](http://arxiv.org/pdf/2411.13205v3)**

> **作者:** Owais Ahmed; M Huzaifa; M Areeb; Hamza Ali Khan
>
> **摘要:** In response to the growing challenges of manual labor and efficiency in warehouse operations, Amazon has embarked on a significant transformation by incorporating robotics to assist with various tasks. While a substantial number of robots have been successfully deployed for tasks such as item transportation within warehouses, the complex process of object picking from shelves remains a significant challenge. This project addresses the issue by developing an innovative robotic system capable of autonomously fulfilling a simulated order by efficiently selecting specific items from shelves. A distinguishing feature of the proposed robotic system is its capacity to navigate the challenge of uncertain object positions within each bin of the shelf. The system is engineered to autonomously adapt its approach, employing strategies that enable it to efficiently locate and retrieve the desired items, even in the absence of pre-established knowledge about their placements.
>
---
#### [replaced 045] FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09081v3](http://arxiv.org/pdf/2506.09081v3)**

> **作者:** Zheqi He; Yesheng Liu; Jing-shu Zheng; Xuejing Li; Jin-Ge Yao; Bowen Qin; Richeng Xuan; Xi Yang
>
> **备注:** Accepted by ACL 2025 Demo
>
> **摘要:** We present FlagEvalMM, an open-source evaluation framework designed to comprehensively assess multimodal models across a diverse range of vision-language understanding and generation tasks, such as visual question answering, text-to-image/video generation, and image-text retrieval. We decouple model inference from evaluation through an independent evaluation service, thus enabling flexible resource allocation and seamless integration of new tasks and models. Moreover, FlagEvalMM utilizes advanced inference acceleration tools (e.g., vLLM, SGLang) and asynchronous data loading to significantly enhance evaluation efficiency. Extensive experiments show that FlagEvalMM offers accurate and efficient insights into model strengths and limitations, making it a valuable tool for advancing multimodal research. The framework is publicly accessible at https://github.com/flageval-baai/FlagEvalMM.
>
---
#### [replaced 046] Efficacy of Image Similarity as a Metric for Augmenting Small Dataset Retinal Image Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04862v3](http://arxiv.org/pdf/2507.04862v3)**

> **作者:** Thomas Wallace; Ik Siong Heng; Senad Subasic; Chris Messenger
>
> **备注:** 30 pages, 10 figures
>
> **摘要:** Synthetic images are an option for augmenting limited medical imaging datasets to improve the performance of various machine learning models. A common metric for evaluating synthetic image quality is the Fr\'echet Inception Distance (FID) which measures the similarity of two image datasets. In this study we evaluate the relationship between this metric and the improvement which synthetic images, generated by a Progressively Growing Generative Adversarial Network (PGGAN), grant when augmenting Diabetes-related Macular Edema (DME) intraretinal fluid segmentation performed by a U-Net model with limited amounts of training data. We find that the behaviour of augmenting with standard and synthetic images agrees with previously conducted experiments. Additionally, we show that dissimilar (high FID) datasets do not improve segmentation significantly. As FID between the training and augmenting datasets decreases, the augmentation datasets are shown to contribute to significant and robust improvements in image segmentation. Finally, we find that there is significant evidence to suggest that synthetic and standard augmentations follow separate log-normal trends between FID and improvements in model performance, with synthetic data proving more effective than standard augmentation techniques. Our findings show that more similar datasets (lower FID) will be more effective at improving U-Net performance, however, the results also suggest that this improvement may only occur when images are sufficiently dissimilar.
>
---
#### [replaced 047] Puzzle Similarity: A Perceptually-guided Cross-Reference Metric for Artifact Detection in 3D Scene Reconstructions
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; 68T07, 68T45, 68T10; I.4; I.3; I.2**

- **链接: [http://arxiv.org/pdf/2411.17489v3](http://arxiv.org/pdf/2411.17489v3)**

> **作者:** Nicolai Hermann; Jorge Condor; Piotr Didyk
>
> **摘要:** Modern reconstruction techniques can effectively model complex 3D scenes from sparse 2D views. However, automatically assessing the quality of novel views and identifying artifacts is challenging due to the lack of ground truth images and the limitations of no-reference image metrics in predicting reliable artifact maps. The absence of such metrics hinders assessment of the quality of novel views and limits the adoption of post-processing techniques, such as inpainting, to enhance reconstruction quality. To tackle this, recent work has established a new category of metrics (cross-reference), predicting image quality solely by leveraging context from alternate viewpoint captures (arXiv:2404.14409). In this work, we propose a new cross-reference metric, Puzzle Similarity, which is designed to localize artifacts in novel views. Our approach utilizes image patch statistics from the training views to establish a scene-specific distribution, later used to identify poorly reconstructed regions in the novel views. Given the lack of good measures to evaluate cross-reference methods in the context of 3D reconstruction, we collected a novel human-labeled dataset of artifact and distortion maps in unseen reconstructed views. Through this dataset, we demonstrate that our method achieves state-of-the-art localization of artifacts in novel views, correlating with human assessment, even without aligned references. We can leverage our new metric to enhance applications like automatic image restoration, guided acquisition, or 3D reconstruction from sparse inputs. Find the project page at https://nihermann.github.io/puzzlesim/ .
>
---
#### [replaced 048] Ensuring Medical AI Safety: Interpretability-Driven Detection and Mitigation of Spurious Model Behavior and Associated Data
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.13818v2](http://arxiv.org/pdf/2501.13818v2)**

> **作者:** Frederik Pahde; Thomas Wiegand; Sebastian Lapuschkin; Wojciech Samek
>
> **摘要:** Deep neural networks are increasingly employed in high-stakes medical applications, despite their tendency for shortcut learning in the presence of spurious correlations, which can have potentially fatal consequences in practice. Whereas a multitude of works address either the detection or mitigation of such shortcut behavior in isolation, the Reveal2Revise approach provides a comprehensive bias mitigation framework combining these steps. However, effectively addressing these biases often requires substantial labeling efforts from domain experts. In this work, we review the steps of the Reveal2Revise framework and enhance it with semi-automated interpretability-based bias annotation capabilities. This includes methods for the sample- and feature-level bias annotation, providing valuable information for bias mitigation methods to unlearn the undesired shortcut behavior. We show the applicability of the framework using four medical datasets across two modalities, featuring controlled and real-world spurious correlations caused by data artifacts. We successfully identify and mitigate these biases in VGG16, ResNet50, and contemporary Vision Transformer models, ultimately increasing their robustness and applicability for real-world medical tasks. Our code is available at https://github.com/frederikpahde/medical-ai-safety.
>
---
#### [replaced 049] RISEE: A Highly Interactive Naturalistic Driving Trajectories Dataset with Human Subjective Risk Perception and Eye-tracking Information
- **分类: cs.HC; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.19490v2](http://arxiv.org/pdf/2507.19490v2)**

> **作者:** Xinzheng Wu; Junyi Chen; Peiyi Wang; Shunxiang Chen; Haolan Meng; Yong Shen
>
> **备注:** Preprint accepted by ITSC 2025
>
> **摘要:** In the research and development (R&D) and verification and validation (V&V) phases of autonomous driving decision-making and planning systems, it is necessary to integrate human factors to achieve decision-making and evaluation that align with human cognition. However, most existing datasets primarily focus on vehicle motion states and trajectories, neglecting human-related information. In addition, current naturalistic driving datasets lack sufficient safety-critical scenarios while simulated datasets suffer from low authenticity. To address these issues, this paper constructs the Risk-Informed Subjective Evaluation and Eye-tracking (RISEE) dataset which specifically contains human subjective evaluations and eye-tracking data apart from regular naturalistic driving trajectories. By leveraging the complementary advantages of drone-based (high realism and extensive scenario coverage) and simulation-based (high safety and reproducibility) data collection methods, we first conduct drone-based traffic video recording at a highway ramp merging area. After that, the manually selected highly interactive scenarios are reconstructed in simulation software, and drivers' first-person view (FPV) videos are generated, which are then viewed and evaluated by recruited participants. During the video viewing process, participants' eye-tracking data is collected. After data processing and filtering, 3567 valid subjective risk ratings from 101 participants across 179 scenarios are retained, along with 2045 qualified eye-tracking data segments. The collected data and examples of the generated FPV videos are available in our website.
>
---
#### [replaced 050] ZeroStereo: Zero-shot Stereo Matching from Single Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.08654v4](http://arxiv.org/pdf/2501.08654v4)**

> **作者:** Xianqi Wang; Hao Yang; Gangwei Xu; Junda Cheng; Min Lin; Yong Deng; Jinliang Zang; Yurui Chen; Xin Yang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** State-of-the-art supervised stereo matching methods have achieved remarkable performance on various benchmarks. However, their generalization to real-world scenarios remains challenging due to the scarcity of annotated real-world stereo data. In this paper, we propose ZeroStereo, a novel stereo image generation pipeline for zero-shot stereo matching. Our approach synthesizes high-quality right images from arbitrary single images by leveraging pseudo disparities generated by a monocular depth estimation model. Unlike previous methods that address occluded regions by filling missing areas with neighboring pixels or random backgrounds, we fine-tune a diffusion inpainting model to recover missing details while preserving semantic structure. Additionally, we propose Training-Free Confidence Generation, which mitigates the impact of unreliable pseudo labels without additional training, and Adaptive Disparity Selection, which ensures a diverse and realistic disparity distribution while preventing excessive occlusion and foreground distortion. Experiments demonstrate that models trained with our pipeline achieve state-of-the-art zero-shot generalization across multiple datasets with only a dataset volume comparable to Scene Flow. Code: https://github.com/Windsrain/ZeroStereo.
>
---
#### [replaced 051] Probabilistic Directed Distance Fields for Ray-Based Shape Representations
- **分类: cs.CV; cs.LG; I.2.10**

- **链接: [http://arxiv.org/pdf/2404.09081v2](http://arxiv.org/pdf/2404.09081v2)**

> **作者:** Tristan Aumentado-Armstrong; Stavros Tsogkas; Sven Dickinson; Allan Jepson
>
> **备注:** Extension of arXiv:2112.05300. Accepted to TPAMI
>
> **摘要:** In modern computer vision, the optimal representation of 3D shape continues to be task-dependent. One fundamental operation applied to such representations is differentiable rendering, as it enables inverse graphics approaches in learning frameworks. Standard explicit shape representations (voxels, point clouds, or meshes) are often easily rendered, but can suffer from limited geometric fidelity, among other issues. On the other hand, implicit representations (occupancy, distance, or radiance fields) preserve greater fidelity, but suffer from complex or inefficient rendering processes, limiting scalability. In this work, we devise Directed Distance Fields (DDFs), a novel neural shape representation that builds upon classical distance fields. The fundamental operation in a DDF maps an oriented point (position and direction) to surface visibility and depth. This enables efficient differentiable rendering, obtaining depth with a single forward pass per pixel, as well as differential geometric quantity extraction (e.g., surface normals), with only additional backward passes. Using probabilistic DDFs (PDDFs), we show how to model inherent discontinuities in the underlying field. We then apply DDFs to several applications, including single-shape fitting, generative modelling, and single-image 3D reconstruction, showcasing strong performance with simple architectural components via the versatility of our representation. Finally, since the dimensionality of DDFs permits view-dependent geometric artifacts, we conduct a theoretical investigation of the constraints necessary for view consistency. We find a small set of field properties that are sufficient to guarantee a DDF is consistent, without knowing, for instance, which shape the field is expressing.
>
---
#### [replaced 052] SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.12857v2](http://arxiv.org/pdf/2507.12857v2)**

> **作者:** Shiqi Huang; Shuting He; Huaiyuan Qin; Bihan Wen
>
> **备注:** ICCV 2025 (Highlight), code see https://github.com/HuangShiqi128/SCORE
>
> **摘要:** Most existing remote sensing instance segmentation approaches are designed for close-vocabulary prediction, limiting their ability to recognize novel categories or generalize across datasets. This restricts their applicability in diverse Earth observation scenarios. To address this, we introduce open-vocabulary (OV) learning for remote sensing instance segmentation. While current OV segmentation models perform well on natural image datasets, their direct application to remote sensing faces challenges such as diverse landscapes, seasonal variations, and the presence of small or ambiguous objects in aerial imagery. To overcome these challenges, we propose $\textbf{SCORE}$ ($\textbf{S}$cene $\textbf{C}$ontext matters in $\textbf{O}$pen-vocabulary $\textbf{RE}$mote sensing instance segmentation), a framework that integrates multi-granularity scene context, i.e., regional context and global context, to enhance both visual and textual representations. Specifically, we introduce Region-Aware Integration, which refines class embeddings with regional context to improve object distinguishability. Additionally, we propose Global Context Adaptation, which enriches naive text embeddings with remote sensing global context, creating a more adaptable and expressive linguistic latent space for the classifier. We establish new benchmarks for OV remote sensing instance segmentation across diverse datasets. Experimental results demonstrate that, our proposed method achieves SOTA performance, which provides a robust solution for large-scale, real-world geospatial analysis. Our code is available at https://github.com/HuangShiqi128/SCORE.
>
---
#### [replaced 053] Category-level Meta-learned NeRF Priors for Efficient Object Mapping
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.01582v3](http://arxiv.org/pdf/2503.01582v3)**

> **作者:** Saad Ejaz; Hriday Bavle; Laura Ribeiro; Holger Voos; Jose Luis Sanchez-Lopez
>
> **摘要:** In 3D object mapping, category-level priors enable efficient object reconstruction and canonical pose estimation, requiring only a single prior per semantic category (e.g., chair, book, laptop, etc.). DeepSDF has been used predominantly as a category-level shape prior, but it struggles to reconstruct sharp geometry and is computationally expensive. In contrast, NeRFs capture fine details but have yet to be effectively integrated with category-level priors in a real-time multi-object mapping framework. To bridge this gap, we introduce PRENOM, a Prior-based Efficient Neural Object Mapper that integrates category-level priors with object-level NeRFs to enhance reconstruction efficiency and enable canonical object pose estimation. PRENOM gets to know objects on a first-name basis by meta-learning on synthetic reconstruction tasks generated from open-source shape datasets. To account for object category variations, it employs a multi-objective genetic algorithm to optimize the NeRF architecture for each category, balancing reconstruction quality and training time. Additionally, prior-based probabilistic ray sampling directs sampling toward expected object regions, accelerating convergence and improving reconstruction quality under constrained resources. Experimental results highlight the ability of PRENOM to achieve high-quality reconstructions while maintaining computational feasibility. Specifically, comparisons with prior-free NeRF-based approaches on a synthetic dataset show a 21\% lower Chamfer distance. Furthermore, evaluations against other approaches using shape priors on a noisy real-world dataset indicate a 13\% improvement averaged across all reconstruction metrics, and comparable pose and size estimation accuracy, while being trained for 5$\times$ less time. Code available at: https://github.com/snt-arg/PRENOM
>
---
#### [replaced 054] MedViT V2: Medical Image Classification with KAN-Integrated Transformers and Dilated Neighborhood Attention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.13693v2](http://arxiv.org/pdf/2502.13693v2)**

> **作者:** Omid Nejati Manzari; Hojat Asgariandehkordi; Taha Koleilat; Yiming Xiao; Hassan Rivaz
>
> **摘要:** Convolutional networks, transformers, hybrid models, and Mamba-based architectures have demonstrated strong performance across various medical image classification tasks. However, these methods were primarily designed to classify clean images using labeled data. In contrast, real-world clinical data often involve image corruptions that are unique to multi-center studies and stem from variations in imaging equipment across manufacturers. In this paper, we introduce the Medical Vision Transformer (MedViTV2), a novel architecture incorporating Kolmogorov-Arnold Network (KAN) layers into the transformer architecture for the first time, aiming for generalized medical image classification. We have developed an efficient KAN block to reduce computational load while enhancing the accuracy of the original MedViT. Additionally, to counteract the fragility of our MedViT when scaled up, we propose an enhanced Dilated Neighborhood Attention (DiNA), an adaptation of the efficient fused dot-product attention kernel capable of capturing global context and expanding receptive fields to scale the model effectively and addressing feature collapse issues. Moreover, a hierarchical hybrid strategy is introduced to stack our Local Feature Perception and Global Feature Perception blocks in an efficient manner, which balances local and global feature perceptions to boost performance. Extensive experiments on 17 medical image classification datasets and 12 corrupted medical image datasets demonstrate that MedViTV2 achieved state-of-the-art results in 27 out of 29 experiments with reduced computational complexity. MedViTV2 is 44\% more computationally efficient than the previous version and significantly enhances accuracy, achieving improvements of 4.6\% on MedMNIST, 5.8\% on NonMNIST, and 13.4\% on the MedMNIST-C benchmark.
>
---
#### [replaced 055] RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02751v3](http://arxiv.org/pdf/2506.02751v3)**

> **作者:** Chuanyu Fu; Yuqi Zhang; Kunbin Yao; Guanying Chen; Yuan Xiong; Chuan Huang; Shuguang Cui; Xiaochun Cao
>
> **备注:** ICCV 2025. Project page: https://fcyycf.github.io/RobustSplat/
>
> **摘要:** 3D Gaussian Splatting (3DGS) has gained significant attention for its real-time, photo-realistic rendering in novel-view synthesis and 3D modeling. However, existing methods struggle with accurately modeling scenes affected by transient objects, leading to artifacts in the rendered images. We identify that the Gaussian densification process, while enhancing scene detail capture, unintentionally contributes to these artifacts by growing additional Gaussians that model transient disturbances. To address this, we propose RobustSplat, a robust solution based on two critical designs. First, we introduce a delayed Gaussian growth strategy that prioritizes optimizing static scene structure before allowing Gaussian splitting/cloning, mitigating overfitting to transient objects in early optimization. Second, we design a scale-cascaded mask bootstrapping approach that first leverages lower-resolution feature similarity supervision for reliable initial transient mask estimation, taking advantage of its stronger semantic consistency and robustness to noise, and then progresses to high-resolution supervision to achieve more precise mask prediction. Extensive experiments on multiple challenging datasets show that our method outperforms existing methods, clearly demonstrating the robustness and effectiveness of our method. Our project page is https://fcyycf.github.io/RobustSplat/.
>
---
#### [replaced 056] A Multi-Agent System Enables Versatile Information Extraction from the Chemical Literature
- **分类: cs.AI; cs.CV; cs.MA**

- **链接: [http://arxiv.org/pdf/2507.20230v2](http://arxiv.org/pdf/2507.20230v2)**

> **作者:** Yufan Chen; Ching Ting Leung; Bowen Yu; Jianwei Sun; Yong Huang; Linyan Li; Hao Chen; Hanyu Gao
>
> **摘要:** To fully expedite AI-powered chemical research, high-quality chemical databases are the cornerstone. Automatic extraction of chemical information from the literature is essential for constructing reaction databases, but it is currently limited by the multimodality and style variability of chemical information. In this work, we developed a multimodal large language model (MLLM)-based multi-agent system for robust and automated chemical information extraction. It utilizes the MLLM's strong reasoning capability to understand the structure of diverse chemical graphics, decompose the extraction task into sub-tasks, and coordinate a set of specialized agents, each combining the capabilities of the MLLM with the precise, domain-specific strengths of dedicated tools, to solve them accurately and integrate the results into a unified output. Our system achieved an F1 score of 80.8% on a benchmark dataset of sophisticated multimodal chemical reaction graphics from the literature, surpassing the previous state-of-the-art model (F1 score of 35.6%) by a significant margin. Additionally, it demonstrated consistent improvements in key sub-tasks, including molecular image recognition, reaction image parsing, named entity recognition and text-based reaction extraction. This work is a critical step toward automated chemical information extraction into structured datasets, which will be a strong promoter of AI-driven chemical research.
>
---
#### [replaced 057] ZERO: Industry-ready Vision Foundation Model with Multi-modal Prompts
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.04270v3](http://arxiv.org/pdf/2507.04270v3)**

> **作者:** Sangbum Choi; Kyeongryeol Go; Taewoong Jang
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** Foundation models have revolutionized AI, yet they struggle with zero-shot deployment in real-world industrial settings due to a lack of high-quality, domain-specific datasets. To bridge this gap, Superb AI introduces ZERO, an industry-ready vision foundation model that leverages multi-modal prompting (textual and visual) for generalization without retraining. Trained on a compact yet representative 0.9 million annotated samples from a proprietary billion-scale industrial dataset, ZERO demonstrates competitive performance on academic benchmarks like LVIS-Val and significantly outperforms existing models across 37 diverse industrial datasets. Furthermore, ZERO achieved 2nd place in the CVPR 2025 Object Instance Detection Challenge and 4th place in the Foundational Few-shot Object Detection Challenge, highlighting its practical deployability and generalizability with minimal adaptation and limited data. To the best of our knowledge, ZERO is the first vision foundation model explicitly built for domain-specific, zero-shot industrial applications.
>
---
#### [replaced 058] From Gallery to Wrist: Realistic 3D Bracelet Insertion in Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20331v2](http://arxiv.org/pdf/2507.20331v2)**

> **作者:** Chenjian Gao; Lihe Ding; Rui Han; Zhanpeng Huang; Zibin Wang; Tianfan Xue
>
> **备注:** 12 pages
>
> **摘要:** Inserting 3D objects into videos is a longstanding challenge in computer graphics with applications in augmented reality, virtual try-on, and video composition. Achieving both temporal consistency, or realistic lighting remains difficult, particularly in dynamic scenarios with complex object motion, perspective changes, and varying illumination. While 2D diffusion models have shown promise for producing photorealistic edits, they often struggle with maintaining temporal coherence across frames. Conversely, traditional 3D rendering methods excel in spatial and temporal consistency but fall short in achieving photorealistic lighting. In this work, we propose a hybrid object insertion pipeline that combines the strengths of both paradigms. Specifically, we focus on inserting bracelets into dynamic wrist scenes, leveraging the high temporal consistency of 3D Gaussian Splatting (3DGS) for initial rendering and refining the results using a 2D diffusion-based enhancement model to ensure realistic lighting interactions. Our method introduces a shading-driven pipeline that separates intrinsic object properties (albedo, shading, reflectance) and refines both shading and sRGB images for photorealism. To maintain temporal coherence, we optimize the 3DGS model with multi-frame weighted adjustments. This is the first approach to synergize 3D rendering and 2D diffusion for video object insertion, offering a robust solution for realistic and consistent video editing. Project Page: https://cjeen.github.io/BraceletPaper/
>
---
#### [replaced 059] JWB-DH-V1: Benchmark for Joint Whole-Body Talking Avatar and Speech Generation Version 1
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.20987v2](http://arxiv.org/pdf/2507.20987v2)**

> **作者:** Xinhan Di; Kristin Qi; Pengqian Yu
>
> **备注:** WiCV @ ICCV 2025
>
> **摘要:** Recent advances in diffusion-based video generation have enabled photo-realistic short clips, but current methods still struggle to achieve multi-modal consistency when jointly generating whole-body motion and natural speech. Current approaches lack comprehensive evaluation frameworks that assess both visual and audio quality, and there are insufficient benchmarks for region-specific performance analysis. To address these gaps, we introduce the Joint Whole-Body Talking Avatar and Speech Generation Version I(JWB-DH-V1), comprising a large-scale multi-modal dataset with 10,000 unique identities across 2 million video samples, and an evaluation protocol for assessing joint audio-video generation of whole-body animatable avatars. Our evaluation of SOTA models reveals consistent performance disparities between face/hand-centric and whole-body performance, which incidates essential areas for future research. The dataset and evaluation tools are publicly available at https://github.com/deepreasonings/WholeBodyBenchmark.
>
---
#### [replaced 060] C2-Evo: Co-Evolving Multimodal Data and Model for Self-Improving Reasoning
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.16518v2](http://arxiv.org/pdf/2507.16518v2)**

> **作者:** Xiuwei Chen; Wentao Hu; Hanhui Li; Jun Zhou; Zisheng Chen; Meng Cao; Yihan Zeng; Kui Zhang; Yu-Jie Yuan; Jianhua Han; Hang Xu; Xiaodan Liang
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have shown impressive reasoning capabilities. However, further enhancing existing MLLMs necessitates high-quality vision-language datasets with carefully curated task complexities, which are both costly and challenging to scale. Although recent self-improving models that iteratively refine themselves offer a feasible solution, they still suffer from two core challenges: (i) most existing methods augment visual or textual data separately, resulting in discrepancies in data complexity (e.g., over-simplified diagrams paired with redundant textual descriptions); and (ii) the evolution of data and models is also separated, leading to scenarios where models are exposed to tasks with mismatched difficulty levels. To address these issues, we propose C2-Evo, an automatic, closed-loop self-improving framework that jointly evolves both training data and model capabilities. Specifically, given a base dataset and a base model, C2-Evo enhances them by a cross-modal data evolution loop and a data-model evolution loop. The former loop expands the base dataset by generating complex multimodal problems that combine structured textual sub-problems with iteratively specified geometric diagrams, while the latter loop adaptively selects the generated problems based on the performance of the base model, to conduct supervised fine-tuning and reinforcement learning alternately. Consequently, our method continuously refines its model and training data, and consistently obtains considerable performance gains across multiple mathematical reasoning benchmarks. Our code, models, and datasets will be released.
>
---
#### [replaced 061] One-stage Modality Distillation for Incomplete Multimodal Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2309.08204v2](http://arxiv.org/pdf/2309.08204v2)**

> **作者:** Shicai Wei; Yang Luo; Chunbo Luo
>
> **摘要:** Learning based on multimodal data has attracted increasing interest recently. While a variety of sensory modalities can be collected for training, not all of them are always available in development scenarios, which raises the challenge to infer with incomplete modality. To address this issue, this paper presents a one-stage modality distillation framework that unifies the privileged knowledge transfer and modality information fusion into a single optimization procedure via multi-task learning. Compared with the conventional modality distillation that performs them independently, this helps to capture the valuable representation that can assist the final model inference directly. Specifically, we propose the joint adaptation network for the modality transfer task to preserve the privileged information. This addresses the representation heterogeneity caused by input discrepancy via the joint distribution adaptation. Then, we introduce the cross translation network for the modality fusion task to aggregate the restored and available modality features. It leverages the parameters-sharing strategy to capture the cross-modal cues explicitly. Extensive experiments on RGB-D classification and segmentation tasks demonstrate the proposed multimodal inheritance framework can overcome the problem of incomplete modality input in various scenes and achieve state-of-the-art performance.
>
---
#### [replaced 062] Generative Ghost: Investigating Ranking Bias Hidden in AI-Generated Videos
- **分类: cs.IR; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.07327v2](http://arxiv.org/pdf/2502.07327v2)**

> **作者:** Haowen Gao; Liang Pang; Shicheng Xu; Leigang Qu; Tat-Seng Chua; Huawei Shen; Xueqi Cheng
>
> **备注:** 13 pages, Accepted at ACMMM2025
>
> **摘要:** With the rapid development of AI-generated content (AIGC), the creation of high-quality AI-generated videos has become faster and easier, resulting in the Internet being flooded with all kinds of video content. However, the impact of these videos on the content ecosystem remains largely unexplored. Video information retrieval remains a fundamental approach for accessing video content. Building on the observation that retrieval models often favor AI-generated content in ad-hoc and image retrieval tasks, we investigate whether similar biases emerge in the context of challenging video retrieval, where temporal and visual factors may further influence model behavior. To explore this, we first construct a comprehensive benchmark dataset containing both real and AI-generated videos, along with a set of fair and rigorous metrics to assess bias. This benchmark consists of 13,000 videos generated by two state-of-the-art open-source video generation models. We meticulously design a suite of rigorous metrics to accurately measure this preference, accounting for potential biases arising from the limited frame rate and suboptimal quality of AIGC videos. We then applied three off-the-shelf video retrieval models to perform retrieval tasks on this hybrid dataset. Our findings reveal a clear preference for AI-generated videos in retrieval. Further investigation shows that incorporating AI-generated videos into the training set of retrieval models exacerbates this bias. Unlike the preference observed in image modalities, we find that video retrieval bias arises from both unseen visual and temporal information, making the root causes of video bias a complex interplay of these two factors. To mitigate this bias, we fine-tune the retrieval models using a contrastive learning approach. The results of this study highlight the potential implications of AI-generated videos on retrieval systems.
>
---
#### [replaced 063] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.01413v5](http://arxiv.org/pdf/2506.01413v5)**

> **作者:** Yulei Qin; Gang Li; Zongyi Li; Zihan Xu; Yuchen Shi; Zhekai Lin; Xiao Cui; Ke Li; Xing Sun
>
> **备注:** 15 pages of main body, 5 tables, 5 figures, 42 pages of appendix
>
> **摘要:** Existing large language models (LLMs) face challenges of following complex instructions, especially when multiple constraints are present and organized in paralleling, chaining, and branching structures. One intuitive solution, namely chain-of-thought (CoT), is expected to universally improve capabilities of LLMs. However, we find that the vanilla CoT exerts a negative impact on performance due to its superficial reasoning pattern of simply paraphrasing the instructions. It fails to peel back the compositions of constraints for identifying their relationship across hierarchies of types and dimensions. To this end, we propose RAIF, a systematic method to boost LLMs in dealing with complex instructions via incentivizing reasoning for test-time compute scaling. First, we stem from the decomposition of complex instructions under existing taxonomies and propose a reproducible data acquisition method. Second, we exploit reinforcement learning (RL) with verifiable rule-centric reward signals to cultivate reasoning specifically for instruction following. We address the shallow, non-essential nature of reasoning under complex instructions via sample-wise contrast for superior CoT enforcement. We also exploit behavior cloning of experts to facilitate steady distribution shift from fast-thinking LLMs to skillful reasoners. Extensive evaluations on seven comprehensive benchmarks confirm the validity of the proposed method, where a 1.5B LLM achieves 11.74% gains with performance comparable to a 8B LLM. Evaluation on OOD constraints also confirms the generalizability of our RAIF. Codes and data are available at https://github.com/yuleiqin/RAIF. Keywords: reinforcement learning with verifiable rewards (RLVR), instruction following, complex instructions
>
---
#### [replaced 064] Unified 3D MRI Representations via Sequence-Invariant Contrastive Learning
- **分类: cs.CV; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2501.12057v3](http://arxiv.org/pdf/2501.12057v3)**

> **作者:** Liam Chalcroft; Jenny Crinion; Cathy J. Price; John Ashburner
>
> **摘要:** Self-supervised deep learning has accelerated 2D natural image analysis but remains difficult to translate into 3D MRI, where data are scarce and pre-trained 2D backbones cannot capture volumetric context. We present a \emph{sequence-invariant} self-supervised framework leveraging quantitative MRI (qMRI). By simulating multiple MRI contrasts from a single 3D qMRI scan and enforcing consistent representations across these contrasts, we learn anatomy-centric rather than sequence-specific features. The result is a single 3D encoder that excels across tasks and protocols. Experiments on healthy brain segmentation (IXI), stroke lesion segmentation (ARC), and MRI denoising show significant gains over baseline SSL approaches, especially in low-data settings (up to +8.3\% Dice, +4.2 dB PSNR). It also generalises to unseen sites, supporting scalable clinical use. Code and trained models are publicly available at https://github.com/liamchalcroft/contrast-squared
>
---
#### [replaced 065] Beyond Class Tokens: LLM-guided Dominant Property Mining for Few-shot Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20511v2](http://arxiv.org/pdf/2507.20511v2)**

> **作者:** Wei Zhuo; Runjie Luo; Wufeng Xue; Linlin Shen
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Few-shot Learning (FSL), which endeavors to develop the generalization ability for recognizing novel classes using only a few images, faces significant challenges due to data scarcity. Recent CLIP-like methods based on contrastive language-image pertaining mitigate the issue by leveraging textual representation of the class name for unseen image discovery. Despite the achieved success, simply aligning visual representations to class name embeddings would compromise the visual diversity for novel class discrimination. To this end, we proposed a novel Few-Shot Learning (FSL) method (BCT-CLIP) that explores \textbf{dominating properties} via contrastive learning beyond simply using class tokens. Through leveraging LLM-based prior knowledge, our method pushes forward FSL with comprehensive structural image representations, including both global category representation and the patch-aware property embeddings. In particular, we presented a novel multi-property generator (MPG) with patch-aware cross-attentions to generate multiple visual property tokens, a Large-Language Model (LLM)-assistant retrieval procedure with clustering-based pruning to obtain dominating property descriptions, and a new contrastive learning strategy for property-token learning. The superior performances on the 11 widely used datasets demonstrate that our investigation of dominating properties advances discriminative class-specific representation learning and few-shot classification.
>
---
#### [replaced 066] Fine-Grained Perturbation Guidance via Attention Head Selection
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.10978v3](http://arxiv.org/pdf/2506.10978v3)**

> **作者:** Donghoon Ahn; Jiwon Kang; Sanghyun Lee; Minjae Kim; Jaewon Min; Wooseok Jang; Sangwu Lee; Sayak Paul; Susung Hong; Seungryong Kim
>
> **备注:** Project page: https://cvlab-kaist.github.io/HeadHunter/
>
> **摘要:** Recent guidance methods in diffusion models steer reverse sampling by perturbing the model to construct an implicit weak model and guide generation away from it. Among these approaches, attention perturbation has demonstrated strong empirical performance in unconditional scenarios where classifier-free guidance is not applicable. However, existing attention perturbation methods lack principled approaches for determining where perturbations should be applied, particularly in Diffusion Transformer (DiT) architectures where quality-relevant computations are distributed across layers. In this paper, we investigate the granularity of attention perturbations, ranging from the layer level down to individual attention heads, and discover that specific heads govern distinct visual concepts such as structure, style, and texture quality. Building on this insight, we propose "HeadHunter", a systematic framework for iteratively selecting attention heads that align with user-centric objectives, enabling fine-grained control over generation quality and visual attributes. In addition, we introduce SoftPAG, which linearly interpolates each selected head's attention map toward an identity matrix, providing a continuous knob to tune perturbation strength and suppress artifacts. Our approach not only mitigates the oversmoothing issues of existing layer-level perturbation but also enables targeted manipulation of specific visual styles through compositional head selection. We validate our method on modern large-scale DiT-based text-to-image models including Stable Diffusion 3 and FLUX.1, demonstrating superior performance in both general quality enhancement and style-specific guidance. Our work provides the first head-level analysis of attention perturbation in diffusion models, uncovering interpretable specialization within attention layers and enabling practical design of effective perturbation strategies.
>
---
#### [replaced 067] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.03248v2](http://arxiv.org/pdf/2412.03248v2)**

> **作者:** Yiwu Zhong; Zhuoming Liu; Yin Li; Liwei Wang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Large language models (LLMs) have enabled the creation of multi-modal LLMs that exhibit strong comprehension of visual data such as images and videos. However, these models usually rely on extensive visual tokens from visual encoders, leading to high computational demands, which limits their applicability in resource-constrained environments and for long-context tasks. In this work, we propose a training-free adaptive inference method for multi-modal LLMs that can accommodate a broad range of efficiency requirements with a minimum performance drop. Our method consists of a) iterative token merging based on embedding similarity before LLMs, and b) progressive token pruning within LLM layers based on multi-modal importance. With a minimalist design, our method can be applied to both video and image LLMs. Extensive experiments on diverse video and image benchmarks demonstrate that our method substantially reduces computation load (e.g., a $\textbf{7-fold}$ reduction in FLOPs) while preserving the performance of video and image LLMs. Further, at a similar computational cost, our method outperforms the state-of-the-art methods in long video understanding (e.g., $\textbf{+4.6}$ on MLVU). Additionally, our in-depth analysis provides insights into token redundancy and LLM layer behaviors, offering guidance for future research in designing efficient multi-modal LLMs. Our code is available at https://github.com/LaVi-Lab/AIM.
>
---
#### [replaced 068] Texture, Shape, Order, and Relation Matter: A New Transformer Design for Sequential DeepFake Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.13873v5](http://arxiv.org/pdf/2404.13873v5)**

> **作者:** Yunfei Li; Yuezun Li; Baoyuan Wu; Junyu Dong; Guopu Zhu; Siwei Lyu
>
> **备注:** An extension of WACV 2025 (Oral)
>
> **摘要:** Sequential DeepFake detection is an emerging task that predicts the manipulation sequence in order. Existing methods typically formulate it as an image-to-sequence problem, employing conventional Transformer architectures. However, these methods lack dedicated design and consequently result in limited performance. As such, this paper describes a new Transformer design, called {TSOM}, by exploring three perspectives: Texture, Shape, and Order of Manipulations. Our method features four major improvements: \ding{182} we describe a new texture-aware branch that effectively captures subtle manipulation traces with a Diversiform Pixel Difference Attention module. \ding{183} Then we introduce a Multi-source Cross-attention module to seek deep correlations among spatial and sequential features, enabling effective modeling of complex manipulation traces. \ding{184} To further enhance the cross-attention, we describe a Shape-guided Gaussian mapping strategy, providing initial priors of the manipulation shape. \ding{185} Finally, observing that the subsequent manipulation in a sequence may influence traces left in the preceding one, we intriguingly invert the prediction order from forward to backward, leading to notable gains as expected. Building upon TSOM, we introduce an extended method, {TSOM++}, which additionally explores Relation of manipulations: \ding{186} we propose a new sequential contrastive learning scheme to capture relationships between various manipulation types in sequence, further enhancing the detection of manipulation traces. We conduct extensive experiments in comparison with several state-of-the-art methods, demonstrating the superiority of our method. The code has been released at https://github.com/OUC-VAS/TSOM.
>
---
#### [replaced 069] Adversarial attacks and defenses in explainable artificial intelligence: A survey
- **分类: cs.CR; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2306.06123v4](http://arxiv.org/pdf/2306.06123v4)**

> **作者:** Hubert Baniecki; Przemyslaw Biecek
>
> **备注:** Accepted by Information Fusion
>
> **摘要:** Explainable artificial intelligence (XAI) methods are portrayed as a remedy for debugging and trusting statistical and deep learning models, as well as interpreting their predictions. However, recent advances in adversarial machine learning (AdvML) highlight the limitations and vulnerabilities of state-of-the-art explanation methods, putting their security and trustworthiness into question. The possibility of manipulating, fooling or fairwashing evidence of the model's reasoning has detrimental consequences when applied in high-stakes decision-making and knowledge discovery. This survey provides a comprehensive overview of research concerning adversarial attacks on explanations of machine learning models, as well as fairness metrics. We introduce a unified notation and taxonomy of methods facilitating a common ground for researchers and practitioners from the intersecting research fields of AdvML and XAI. We discuss how to defend against attacks and design robust interpretation methods. We contribute a list of existing insecurities in XAI and outline the emerging research directions in adversarial XAI (AdvXAI). Future work should address improving explanation methods and evaluation protocols to take into account the reported safety issues.
>
---
#### [replaced 070] Very High-Resolution Bridge Deformation Monitoring Using UAV-based Photogrammetry
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2410.18984v2](http://arxiv.org/pdf/2410.18984v2)**

> **作者:** Mehdi Maboudi; Jan Backhaus; Inka Mai; Yahya Ghassoun; Yogesh Khedar; Dirk Lowke; Bjoern Riedel; Ulf Bestmann; Markus Gerke
>
> **摘要:** Accurate and efficient structural health monitoring of infrastructure objects such as bridges is a vital task, as many existing constructions have already reached or are approaching their planned service life. In this contribution, we address the question of the suitability of UAV-based monitoring for SHM, in particular focusing on the geometric deformation under load. Such an advanced technology is becoming increasingly popular due to its ability to decrease the cost and risk of tedious traditional inspection methods. To this end, we performed extensive tests employing a research reinforced concrete bridge that can be exposed to a predefined load via ground anchors. Very high-resolution image blocks have been captured before, during, and after the application of controlled loads. From those images, the motion of distinct points on the bridge has been monitored, and in addition, dense image point clouds were computed to evaluate the performance of surface-based data acquisition. Moreover, a geodetic control network in stable regions is used as control information for bundle adjustment. We applied different sensing technologies in order to be able to judge the image-based deformation results: displacement transducers, tachymetry, and laser profiling. As a platform for the photogrammetric measurements, a multi-rotor UAV DJI Matrice 600 Pro was employed, equipped with two RTK-GNSS receivers. The mounted camera was a PhaseOne iXM-100 (100MP) with an 80 mm lens. With a flying height of 30 m above the terrain, this resulted in a GSD of 1.3 mm while a forward and sideward overlap of 80% was maintained. The comparison with reference data (displacement transducers) reveals a difference of less than 1 mm. We show that by employing the introduced UAV-based monitoring approach, a full area-wide quantification of deformation is possible in contrast to classical point or profile measurements.
>
---
#### [replaced 071] T2I-Copilot: A Training-Free Multi-Agent Text-to-Image System for Enhanced Prompt Interpretation and Interactive Generation
- **分类: cs.CV; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.20536v2](http://arxiv.org/pdf/2507.20536v2)**

> **作者:** Chieh-Yun Chen; Min Shi; Gong Zhang; Humphrey Shi
>
> **备注:** ICCV 2025
>
> **摘要:** Text-to-Image (T2I) generative models have revolutionized content creation but remain highly sensitive to prompt phrasing, often requiring users to repeatedly refine prompts multiple times without clear feedback. While techniques such as automatic prompt engineering, controlled text embeddings, denoising, and multi-turn generation mitigate these issues, they offer limited controllability, or often necessitate additional training, restricting the generalization abilities. Thus, we introduce T2I-Copilot, a training-free multi-agent system that leverages collaboration between (Multimodal) Large Language Models to automate prompt phrasing, model selection, and iterative refinement. This approach significantly simplifies prompt engineering while enhancing generation quality and text-image alignment compared to direct generation. Specifically, T2I-Copilot consists of three agents: (1) Input Interpreter, which parses the input prompt, resolves ambiguities, and generates a standardized report; (2) Generation Engine, which selects the appropriate model from different types of T2I models and organizes visual and textual prompts to initiate generation; and (3) Quality Evaluator, which assesses aesthetic quality and text-image alignment, providing scores and feedback for potential regeneration. T2I-Copilot can operate fully autonomously while also supporting human-in-the-loop intervention for fine-grained control. On GenAI-Bench, using open-source generation models, T2I-Copilot achieves a VQA score comparable to commercial models RecraftV3 and Imagen 3, surpasses FLUX1.1-pro by 6.17% at only 16.59% of its cost, and outperforms FLUX.1-dev and SD 3.5 Large by 9.11% and 6.36%. Code will be released at: https://github.com/SHI-Labs/T2I-Copilot.
>
---
#### [replaced 072] Geometric Algebra Meets Large Language Models: Instruction-Based Transformations of Separate Meshes in 3D, Interactive and Controllable Scenes
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [http://arxiv.org/pdf/2408.02275v2](http://arxiv.org/pdf/2408.02275v2)**

> **作者:** Prodromos Kolyvakis; Manos Kamarianakis; George Papagiannakis
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** This paper introduces a novel integration of Large Language Models (LLMs) with Conformal Geometric Algebra (CGA) to revolutionize controllable 3D scene editing, particularly for object repositioning tasks, which traditionally requires intricate manual processes and specialized expertise. These conventional methods typically suffer from reliance on large training datasets or lack a formalized language for precise edits. Utilizing CGA as a robust formal language, our system, Shenlong, precisely models spatial transformations necessary for accurate object repositioning. Leveraging the zero-shot learning capabilities of pre-trained LLMs, Shenlong translates natural language instructions into CGA operations which are then applied to the scene, facilitating exact spatial transformations within 3D scenes without the need for specialized pre-training. Implemented in a realistic simulation environment, Shenlong ensures compatibility with existing graphics pipelines. To accurately assess the impact of CGA, we benchmark against robust Euclidean Space baselines, evaluating both latency and accuracy. Comparative performance evaluations indicate that Shenlong significantly reduces LLM response times by 16% and boosts success rates by 9.6% on average compared to the traditional methods. Notably, Shenlong achieves a 100% perfect success rate in common practical queries, a benchmark where other systems fall short. These advancements underscore Shenlong's potential to democratize 3D scene editing, enhancing accessibility and fostering innovation across sectors such as education, digital entertainment, and virtual reality.
>
---
#### [replaced 073] Improving Visual Place Recognition with Sequence-Matching Receptiveness Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06840v2](http://arxiv.org/pdf/2503.06840v2)**

> **作者:** Somayeh Hussaini; Tobias Fischer; Michael Milford
>
> **备注:** 8 pages, 5 figures, Accepted to the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** In visual place recognition (VPR), filtering and sequence-based matching approaches can improve performance by integrating temporal information across image sequences, especially in challenging conditions. While these methods are commonly applied, their effects on system behavior can be unpredictable and can actually make performance worse in certain situations. In this work, we present a new supervised learning approach that learns to predict the per-frame sequence matching receptiveness (SMR) of VPR techniques, enabling the system to selectively decide when to trust the output of a sequence matching system. Our approach is agnostic to the underlying VPR technique and effectively predicts SMR, and hence significantly improves VPR performance across a large range of state-of-the-art and classical VPR techniques (namely CosPlace, MixVPR, EigenPlaces, SALAD, AP-GeM, NetVLAD and SAD), and across three benchmark VPR datasets (Nordland, Oxford RobotCar, and SFU-Mountain). We also provide insights into a complementary approach that uses the predictor to replace discarded matches, and present ablation studies including an analysis of the interactions between our SMR predictor and the selected sequence length.
>
---
#### [replaced 074] PEVLM: Parallel Encoding for Vision-Language Models
- **分类: cs.CV; cs.LG; cs.PF**

- **链接: [http://arxiv.org/pdf/2506.19651v3](http://arxiv.org/pdf/2506.19651v3)**

> **作者:** Letian Kang; Shixian Luo; Yiqiang Li; Yuxin Yin; Shenxuan Zhou; Xiaoyang Yu; Jin Yang; Yong Wu
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated strong capabilities in multimodal understanding and generation tasks. However, their application to long video understanding remains hindered by the quadratic complexity of standard attention mechanisms. In this work, we introduce \textbf{PEVLM}, a fine-tuning-free parallel encoding method designed to enhance the prefilling efficiency of VLMs in long video scenarios. PEVLM partitions the input video into context blocks with a shared sink block, while preserving sequential position embeddings to align the attention weight distribution with that of Full-Attention. This design reduces attention complexity from $O((T \times N)^2)$ to $O(T \times N)$ where $T$ is the number of frames and $N$ the number of tokens per frame, without sacrificing accuracy. Extensive experiments across multiple state-of-the-art models and benchmarks demonstrate that PEVLM consistently outperforms existing parallel encoding approaches, achieving up to \textbf{7.47x} speedup in attention computation and reducing end-to-end latency by \textbf{40\%}. Remarkably, PEVLM not only maintains high accuracy, but in some settings even surpasses Full-Attention performance. Under strict latency constraints, it achieves substantial gains, improving accuracy from \textbf{23.26\%} to \textbf{61.03\%}. These results underscore the effectiveness of PEVLM for low-latency, long-context video understanding, making it a promising solution for real-world applications.
>
---
#### [replaced 075] When Tokens Talk Too Much: A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20198v2](http://arxiv.org/pdf/2507.20198v2)**

> **作者:** Kele Shao; Keda Tao; Kejia Zhang; Sicheng Feng; Mu Cai; Yuzhang Shang; Haoxuan You; Can Qin; Yang Sui; Huan Wang
>
> **备注:** For ongoing updates and to track the latest advances in this promising area, we maintain a public repository: https://github.com/cokeshao/Awesome-Multimodal-Token-Compression
>
> **摘要:** Multimodal large language models (MLLMs) have made remarkable strides, largely driven by their ability to process increasingly long and complex contexts, such as high-resolution images, extended video sequences, and lengthy audio input. While this ability significantly enhances MLLM capabilities, it introduces substantial computational challenges, primarily due to the quadratic complexity of self-attention mechanisms with numerous input tokens. To mitigate these bottlenecks, token compression has emerged as an auspicious and critical approach, efficiently reducing the number of tokens during both training and inference. In this paper, we present the first systematic survey and synthesis of the burgeoning field of multimodal long context token compression. Recognizing that effective compression strategies are deeply tied to the unique characteristics and redundancies of each modality, we categorize existing approaches by their primary data focus, enabling researchers to quickly access and learn methods tailored to their specific area of interest: (1) image-centric compression, which addresses spatial redundancy in visual data; (2) video-centric compression, which tackles spatio-temporal redundancy in dynamic sequences; and (3) audio-centric compression, which handles temporal and spectral redundancy in acoustic signals. Beyond this modality-driven categorization, we further dissect methods based on their underlying mechanisms, including transformation-based, similarity-based, attention-based, and query-based approaches. By providing a comprehensive and structured overview, this survey aims to consolidate current progress, identify key challenges, and inspire future research directions in this rapidly evolving domain. We also maintain a public repository to continuously track and update the latest advances in this promising area.
>
---
#### [replaced 076] SurgiSR4K: A High-Resolution Endoscopic Video Dataset for Robotic-Assisted Minimally Invasive Procedures
- **分类: eess.IV; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.00209v3](http://arxiv.org/pdf/2507.00209v3)**

> **作者:** Fengyi Jiang; Xiaorui Zhang; Lingbo Jin; Ruixing Liang; Yuxin Chen; Adi Chola Venkatesh; Jason Culman; Tiantian Wu; Lirong Shao; Wenqing Sun; Cong Gao; Hallie McNamara; Jingpei Lu; Omid Mohareri
>
> **摘要:** High-resolution imaging is crucial for enhancing visual clarity and enabling precise computer-assisted guidance in minimally invasive surgery (MIS). Despite the increasing adoption of 4K endoscopic systems, there remains a significant gap in publicly available native 4K datasets tailored specifically for robotic-assisted MIS. We introduce SurgiSR4K, the first publicly accessible surgical imaging and video dataset captured at a native 4K resolution, representing realistic conditions of robotic-assisted procedures. SurgiSR4K comprises diverse visual scenarios including specular reflections, tool occlusions, bleeding, and soft tissue deformations, meticulously designed to reflect common challenges faced during laparoscopic and robotic surgeries. This dataset opens up possibilities for a broad range of computer vision tasks that might benefit from high resolution data, such as super resolution (SR), smoke removal, surgical instrument detection, 3D tissue reconstruction, monocular depth estimation, instance segmentation, novel view synthesis, and vision-language model (VLM) development. SurgiSR4K provides a robust foundation for advancing research in high-resolution surgical imaging and fosters the development of intelligent imaging technologies aimed at enhancing performance, safety, and usability in image-guided robotic surgeries.
>
---
#### [replaced 077] Towards Facilitated Fairness Assessment of AI-based Skin Lesion Classifiers Through GenAI-based Image Synthesis
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.17860v2](http://arxiv.org/pdf/2507.17860v2)**

> **作者:** Ko Watanabe; Stanislav Frolov; Adriano Lucieri; Andreas Dengel
>
> **摘要:** Recent advancements in Deep Learning and its application on the edge hold great potential for the revolution of routine screenings for skin cancers like Melanoma. Along with the anticipated benefits of this technology, potential dangers arise from unforseen and inherent biases. Thus, assessing and improving the fairness of such systems is of utmost importance. A key challenge in fairness assessment is to ensure that the evaluation dataset is sufficiently representative of different Personal Identifiable Information (PII) (sex, age, and race) and other minority groups. Against the backdrop of this challenge, this study leverages the state-of-the-art Generative AI (GenAI) LightningDiT model to assess the fairness of publicly available melanoma classifiers. The results suggest that fairness assessment using highly realistic synthetic data is a promising direction. Yet, our findings indicate that verifying fairness becomes difficult when the melanoma-detection model used for evaluation is trained on data that differ from the dataset underpinning the synthetic images. Nonetheless, we propose that our approach offers a valuable new avenue for employing synthetic data to gauge and enhance fairness in medical-imaging GenAI systems.
>
---
#### [replaced 078] GS-SDF: LiDAR-Augmented Gaussian Splatting and Neural SDF for Geometrically Consistent Rendering and Reconstruction
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10170v2](http://arxiv.org/pdf/2503.10170v2)**

> **作者:** Jianheng Liu; Yunfei Wan; Bowen Wang; Chunran Zheng; Jiarong Lin; Fu Zhang
>
> **备注:** 8 pages, IROS 2025
>
> **摘要:** Digital twins are fundamental to the development of autonomous driving and embodied artificial intelligence. However, achieving high-granularity surface reconstruction and high-fidelity rendering remains a challenge. Gaussian splatting offers efficient photorealistic rendering but struggles with geometric inconsistencies due to fragmented primitives and sparse observational data in robotics applications. Existing regularization methods, which rely on render-derived constraints, often fail in complex environments. Moreover, effectively integrating sparse LiDAR data with Gaussian splatting remains challenging. We propose a unified LiDAR-visual system that synergizes Gaussian splatting with a neural signed distance field. The accurate LiDAR point clouds enable a trained neural signed distance field to offer a manifold geometry field. This motivates us to offer an SDF-based Gaussian initialization for physically grounded primitive placement and a comprehensive geometric regularization for geometrically consistent rendering and reconstruction. Experiments demonstrate superior reconstruction accuracy and rendering quality across diverse trajectories. To benefit the community, the codes are released at https://github.com/hku-mars/GS-SDF.
>
---
#### [replaced 079] From Semantics, Scene to Instance-awareness: Distilling Foundation Model for Open-vocabulary Situation Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.14686v2](http://arxiv.org/pdf/2507.14686v2)**

> **作者:** Chen Cai; Tianyi Liu; Jianjun Gao; Wenyang Liu; Kejun Wu; Ruoyu Wang; Yi Wang; Soo Chin Liew
>
> **摘要:** Recent Multimodal Large Language Models (MLLMs) exhibit strong zero-shot abilities but struggle with complex Grounded Situation Recognition (GSR) and are resource-intensive for edge device deployment. Meanwhile, conventional GSR models often lack generalization ability, falling short in recognizing unseen and rare situations. In this paper, we exploit transferring knowledge from a teacher MLLM to a small GSR model to enhance its generalization and zero-shot abilities, thereby introducing the task of Open-vocabulary Grounded Situation Recognition (Ov-GSR). To achieve this, we propose Multimodal Interactive Prompt Distillation (MIPD), a novel framework that distills enriched multimodal knowledge from the foundation model, enabling the student Ov-GSR model to recognize unseen situations and be better aware of rare situations. Specifically, the MIPD framework first leverages the LLM-based Judgmental Rationales Generator (JRG) to construct positive and negative glimpse and gaze rationales enriched with contextual semantic information. The proposed scene-aware and instance-perception prompts are then introduced to align rationales with visual information from the MLLM teacher via the Negative-Guided Multimodal Prompting Alignment (NMPA) module, effectively capturing holistic and perceptual multimodal knowledge. Finally, the aligned multimodal knowledge is distilled into the student Ov-GSR model, providing a stronger foundation for generalization that enhances situation understanding, bridges the gap between seen and unseen scenarios, and mitigates prediction bias in rare cases. We evaluate MIPD on the refined Ov-SWiG dataset, achieving superior performance on seen, rare, and unseen situations, and further demonstrate improved unseen detection on the HICO-DET dataset.
>
---
