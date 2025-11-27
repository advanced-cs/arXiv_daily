# 计算机视觉 cs.CV

- **最新发布 138 篇**

- **更新 107 篇**

## 最新发布

#### [new 001] Shift-Equivariant Complex-Valued Convolutional Neural Networks
- **分类: cs.CV**

- **简介: 该论文针对卷积神经网络缺乏平移等变与不变性的问题，提出适用于复值神经网络的可学习多相上/下采样方法（LPS），引入从复数到实数的投影层，理论保障平移等变性。在极化合成孔径雷达图像的分类、重建与语义分割任务中验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2511.21250v1](https://arxiv.org/pdf/2511.21250v1)**

> **作者:** Quentin Gabot; Teck-Yian Lim; Jérémy Fix; Joana Frontera-Pons; Chengfang Ren; Jean-Philippe Ovarlez
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Convolutional neural networks have shown remarkable performance in recent years on various computer vision problems. However, the traditional convolutional neural network architecture lacks a critical property: shift equivariance and invariance, broken by downsampling and upsampling operations. Although data augmentation techniques can help the model learn the latter property empirically, a consistent and systematic way to achieve this goal is by designing downsampling and upsampling layers that theoretically guarantee these properties by construction. Adaptive Polyphase Sampling (APS) introduced the cornerstone for shift invariance, later extended to shift equivariance with Learnable Polyphase up/downsampling (LPS) applied to real-valued neural networks. In this paper, we extend the work on LPS to complex-valued neural networks both from a theoretical perspective and with a novel building block of a projection layer from $\mathbb{C}$ to $\mathbb{R}$ before the Gumbel Softmax. We finally evaluate this extension on several computer vision problems, specifically for either the invariance property in classification tasks or the equivariance property in both reconstruction and semantic segmentation problems, using polarimetric Synthetic Aperture Radar images.
>
---
#### [new 002] HTTM: Head-wise Temporal Token Merging for Faster VGGT
- **分类: cs.CV**

- **简介: 该论文针对3D场景重建中VGGT模型因全局注意力导致的高延迟问题，提出无训练的头级别时序令牌合并方法HTTM。通过在多头粒度上融合令牌，保留特征唯一性并利用头级空间与时间对应关系，实现高达7倍加速，同时保持性能几乎不变。**

- **链接: [https://arxiv.org/pdf/2511.21317v1](https://arxiv.org/pdf/2511.21317v1)**

> **作者:** Weitian Wang; Lukas Meiner; Rai Shubham; Cecilia De La Parra; Akash Kumar
>
> **摘要:** The Visual Geometry Grounded Transformer (VGGT) marks a significant leap forward in 3D scene reconstruction, as it is the first model that directly infers all key 3D attributes (camera poses, depths, and dense geometry) jointly in one pass. However, this joint inference mechanism requires global attention layers that perform all-to-all attention computation on tokens from all views. For reconstruction of large scenes with long-sequence inputs, this causes a significant latency bottleneck. In this paper, we propose head-wise temporal merging (HTTM), a training-free 3D token merging method for accelerating VGGT. Existing merging techniques merge tokens uniformly across different attention heads, resulting in identical tokens in the layers' output, which hinders the model's representational ability. HTTM tackles this problem by merging tokens in multi-head granularity, which preserves the uniqueness of feature tokens after head concatenation. Additionally, this enables HTTM to leverage the spatial locality and temporal correspondence observed at the head level to achieve higher merging ratios with lower merging costs compared to existing methods. Thus, HTTM achieves up to 7x acceleration with negligible performance drops in a GPU-based inference.
>
---
#### [new 003] One Patch is All You Need: Joint Surface Material Reconstruction and Classification from Minimal Visual Cues
- **分类: cs.CV**

- **简介: 该论文提出SMARC模型，解决从单个10%图像补丁中联合重建与分类材料表面的难题。针对视觉线索稀疏的场景，采用部分卷积U-Net结合分类头，实现端到端的表面修复与材质识别，在真实数据集上达到优异性能。**

- **链接: [https://arxiv.org/pdf/2511.20784v1](https://arxiv.org/pdf/2511.20784v1)**

> **作者:** Sindhuja Penchala; Gavin Money; Gabriel Marques; Samuel Wood; Jessica Kirschman; Travis Atkison; Shahram Rahimi; Noorbakhsh Amiri Golilarz
>
> **备注:** 9 pages,3 figures, 5 tables
>
> **摘要:** Understanding material surfaces from sparse visual cues is critical for applications in robotics, simulation, and material perception. However, most existing methods rely on dense or full-scene observations, limiting their effectiveness in constrained or partial view environment. To address this challenge, we introduce SMARC, a unified model for Surface MAterial Reconstruction and Classification from minimal visual input. By giving only a single 10% contiguous patch of the image, SMARC recognizes and reconstructs the full RGB surface while simultaneously classifying the material category. Our architecture combines a Partial Convolutional U-Net with a classification head, enabling both spatial inpainting and semantic understanding under extreme observation sparsity. We compared SMARC against five models including convolutional autoencoders [17], Vision Transformer (ViT) [13], Masked Autoencoder (MAE) [5], Swin Transformer [9], and DETR [2] using Touch and Go dataset [16] of real-world surface textures. SMARC achieves state-of-the-art results with a PSNR of 17.55 dB and a material classification accuracy of 85.10%. Our findings highlight the advantages of partial convolution in spatial reasoning under missing data and establish a strong foundation for minimal-vision surface understanding.
>
---
#### [new 004] PFF-Net: Patch Feature Fitting for Point Cloud Normal Estimation
- **分类: cs.CV**

- **简介: 该论文针对点云法向估计任务，解决局部邻域尺寸选择困难的问题。提出PFF-Net，通过多尺度特征融合与跨尺度补偿，实现自适应特征拟合，提升法向估计精度与效率，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.21365v1](https://arxiv.org/pdf/2511.21365v1)**

> **作者:** Qing Li; Huifang Feng; Kanle Shi; Yue Gao; Yi Fang; Yu-Shen Liu; Zhizhong Han
>
> **备注:** Accepted by TVCG
>
> **摘要:** Estimating the normal of a point requires constructing a local patch to provide center-surrounding context, but determining the appropriate neighborhood size is difficult when dealing with different data or geometries. Existing methods commonly employ various parameter-heavy strategies to extract a full feature description from the input patch. However, they still have difficulties in accurately and efficiently predicting normals for various point clouds. In this work, we present a new idea of feature extraction for robust normal estimation of point clouds. We use the fusion of multi-scale features from different neighborhood sizes to address the issue of selecting reasonable patch sizes for various data or geometries. We seek to model a patch feature fitting (PFF) based on multi-scale features to approximate the optimal geometric description for normal estimation and implement the approximation process via multi-scale feature aggregation and cross-scale feature compensation. The feature aggregation module progressively aggregates the patch features of different scales to the center of the patch and shrinks the patch size by removing points far from the center. It not only enables the network to precisely capture the structure characteristic in a wide range, but also describes highly detailed geometries. The feature compensation module ensures the reusability of features from earlier layers of large scales and reveals associated information in different patch sizes. Our approximation strategy based on aggregating the features of multiple scales enables the model to achieve scale adaptation of varying local patches and deliver the optimal feature description. Extensive experiments demonstrate that our method achieves state-of-the-art performance on both synthetic and real-world datasets with fewer network parameters and running time.
>
---
#### [new 005] EvRainDrop: HyperGraph-guided Completion for Effective Frame and Event Stream Aggregation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对事件相机数据的时空稀疏性问题，提出基于超图的事件流补全方法EvRainDrop。通过超图建模跨时空事件关联，利用上下文信息传递完成稀疏事件补全，并融合RGB信息实现多模态特征学习与聚合，在事件分类任务中验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2511.21439v1](https://arxiv.org/pdf/2511.21439v1)**

> **作者:** Futian Wang; Fan Zhang; Xiao Wang; Mengqi Wang; Dexing Huang; Jin Tang
>
> **摘要:** Event cameras produce asynchronous event streams that are spatially sparse yet temporally dense. Mainstream event representation learning algorithms typically use event frames, voxels, or tensors as input. Although these approaches have achieved notable progress, they struggle to address the undersampling problem caused by spatial sparsity. In this paper, we propose a novel hypergraph-guided spatio-temporal event stream completion mechanism, which connects event tokens across different times and spatial locations via hypergraphs and leverages contextual information message passing to complete these sparse events. The proposed method can flexibly incorporate RGB tokens as nodes in the hypergraph within this completion framework, enabling multi-modal hypergraph-based information completion. Subsequently, we aggregate hypergraph node information across different time steps through self-attention, enabling effective learning and fusion of multi-modal features. Extensive experiments on both single- and multi-label event classification tasks fully validated the effectiveness of our proposed framework. The source code of this paper will be released on https://github.com/Event-AHU/EvRainDrop.
>
---
#### [new 006] MobileI2V: Fast and High-Resolution Image-to-Video on Mobile Devices
- **分类: cs.CV**

- **简介: 该论文针对移动设备上图像到视频（I2V）生成速度慢、计算资源受限的问题，提出轻量级模型MobileI2V。通过线性混合去噪器、两步采样蒸馏和移动端注意力优化，实现720p视频实时生成，单帧生成时间小于100毫秒，显著提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2511.21475v1](https://arxiv.org/pdf/2511.21475v1)**

> **作者:** Shuai Zhang; Bao Tang; Siyuan Yu; Yueting Zhu; Jingfeng Yao; Ya Zou; Shanglin Yuan; Li Yu; Wenyu Liu; Xinggang Wang
>
> **备注:** Our Demo and code:https://github.com/hustvl/MobileI2V
>
> **摘要:** Recently, video generation has witnessed rapid advancements, drawing increasing attention to image-to-video (I2V) synthesis on mobile devices. However, the substantial computational complexity and slow generation speed of diffusion models pose significant challenges for real-time, high-resolution video generation on resource-constrained mobile devices. In this work, we propose MobileI2V, a 270M lightweight diffusion model for real-time image-to-video generation on mobile devices. The core lies in: (1) We analyzed the performance of linear attention modules and softmax attention modules on mobile devices, and proposed a linear hybrid architecture denoiser that balances generation efficiency and quality. (2) We design a time-step distillation strategy that compresses the I2V sampling steps from more than 20 to only two without significant quality loss, resulting in a 10-fold increase in generation speed. (3) We apply mobile-specific attention optimizations that yield a 2-fold speed-up for attention operations during on-device inference. MobileI2V enables, for the first time, fast 720p image-to-video generation on mobile devices, with quality comparable to existing models. Under one-step conditions, the generation speed of each frame of 720p video is less than 100 ms. Our code is available at: https://github.com/hustvl/MobileI2V.
>
---
#### [new 007] RefOnce: Distilling References into a Prototype Memory for Referring Camouflaged Object Detection
- **分类: cs.CV**

- **简介: 该论文针对参照性伪装目标检测（Ref-COD）任务，解决现有方法需测试时提供参考图像导致部署困难、延迟高的问题。提出RefOnce框架，通过训练时将参考信息蒸馏为类别原型记忆，并在推理时根据查询条件生成参考向量，实现无需测试参考图像的高效检测。**

- **链接: [https://arxiv.org/pdf/2511.20989v1](https://arxiv.org/pdf/2511.20989v1)**

> **作者:** Yu-Huan Wu; Zi-Xuan Zhu; Yan Wang; Liangli Zhen; Deng-Ping Fan
>
> **备注:** 11 pages, 5 figure, 6 tables
>
> **摘要:** Referring Camouflaged Object Detection (Ref-COD) segments specified camouflaged objects in a scene by leveraging a small set of referring images. Though effective, current systems adopt a dual-branch design that requires reference images at test time, which limits deployability and adds latency and data-collection burden. We introduce a Ref-COD framework that distills references into a class-prototype memory during training and synthesizes a reference vector at inference via a query-conditioned mixture of prototypes. Concretely, we maintain an EMA-updated prototype per category and predict mixture weights from the query to produce a guidance vector without any test-time references. To bridge the representation gap between reference statistics and camouflaged query features, we propose a bidirectional attention alignment module that adapts both the query features and the class representation. Thus, our approach yields a simple, efficient path to Ref-COD without mandatory references. We evaluate the proposed method on the large-scale R2C7K benchmark. Extensive experiments demonstrate competitive or superior performance of the proposed method compared with recent state-of-the-arts. Code is available at https://github.com/yuhuan-wu/RefOnce.
>
---
#### [new 008] LLaVA-UHD v3: Progressive Visual Compression for Efficient Native-Resolution Encoding in MLLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态大模型（MLLMs）中高分辨率视觉编码带来的计算开销问题，提出基于渐进式视觉压缩（PVC）的高效原生分辨率编码方法。通过改进嵌入与分窗令牌压缩，实现低延迟、高性能的视觉建模，显著降低首次生成时间（TTFT），提升推理效率。**

- **链接: [https://arxiv.org/pdf/2511.21150v1](https://arxiv.org/pdf/2511.21150v1)**

> **作者:** Shichu Sun; Yichen Zhang; Haolin Song; Zonghao Guo; Chi Chen; Yidan Zhang; Yuan Yao; Zhiyuan Liu; Maosong Sun
>
> **摘要:** Visual encoding followed by token condensing has become the standard architectural paradigm in multi-modal large language models (MLLMs). Many recent MLLMs increasingly favor global native- resolution visual encoding over slice-based methods. To investigate this trend, we systematically compare their behavior on vision-language understanding and attention patterns, revealing that global encoding enhances overall capability but at the expense of greater computational overhead. To address this issue, we present LLaVA-UHD v3, an MLLM centered upon our proposed Progressive Visual Compression (PVC) method, which can be seamlessly integrated into standard Vision Transformer (ViT) to enable efficient native-resolution encoding. The PVC approach consists of two key modules: (i) refined patch embedding, which supports flexible patch-size scaling for fine-grained visual model- ing, (ii) windowed token compression, hierarchically deployed across ViT layers to progressively aggregate local token representations. Jointly modulated by these two modules, a widely pretrained ViT can be reconfigured into an efficient architecture while largely preserving generality. Evaluated across extensive benchmarks, the transformed ViT, termed ViT-UHD, demonstrates competitive performance with MoonViT while reducing TTFT (time-to-first-token) by 2.4x, when developed within an identical MLLM architecture. Building upon ViT-UHD, LLaVA-UHD v3 also achieves competitive performance to Qwen2-VL, while further reducing TTFT by 1.9x. We will release all code and checkpoints to support future research on efficient MLLMs.
>
---
#### [new 009] Are Neuro-Inspired Multi-Modal Vision-Language Models Resilient to Membership Inference Privacy Leakage?
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文研究多模态视觉语言模型（VLMs）在成员推理攻击（MIA）下的隐私泄露问题，属于隐私安全任务。针对现有研究忽视神经启发模型在隐私攻击中的鲁棒性问题，提出拓扑正则化框架（tau），通过实验验证神经启发的VLMs在保持模型性能的同时，显著降低隐私泄露风险，提升对MIA的防御能力。**

- **链接: [https://arxiv.org/pdf/2511.20710v1](https://arxiv.org/pdf/2511.20710v1)**

> **作者:** David Amebley; Sayanton Dibbo
>
> **摘要:** In the age of agentic AI, the growing deployment of multi-modal models (MMs) has introduced new attack vectors that can leak sensitive training data in MMs, causing privacy leakage. This paper investigates a black-box privacy attack, i.e., membership inference attack (MIA) on multi-modal vision-language models (VLMs). State-of-the-art research analyzes privacy attacks primarily to unimodal AI-ML systems, while recent studies indicate MMs can also be vulnerable to privacy attacks. While researchers have demonstrated that biologically inspired neural network representations can improve unimodal model resilience against adversarial attacks, it remains unexplored whether neuro-inspired MMs are resilient against privacy attacks. In this work, we introduce a systematic neuroscience-inspired topological regularization (tau) framework to analyze MM VLMs resilience against image-text-based inference privacy attacks. We examine this phenomenon using three VLMs: BLIP, PaliGemma 2, and ViT-GPT2, across three benchmark datasets: COCO, CC3M, and NoCaps. Our experiments compare the resilience of baseline and neuro VLMs (with topological regularization), where the tau > 0 configuration defines the NEURO variant of VLM. Our results on the BLIP model using the COCO dataset illustrate that MIA attack success in NEURO VLMs drops by 24% mean ROC-AUC, while achieving similar model utility (similarities between generated and reference captions) in terms of MPNet and ROUGE-2 metrics. This shows neuro VLMs are comparatively more resilient against privacy attacks, while not significantly compromising model utility. Our extensive evaluation with PaliGemma 2 and ViT-GPT2 models, on two additional datasets: CC3M and NoCaps, further validates the consistency of the findings. This work contributes to the growing understanding of privacy risks in MMs and provides evidence on neuro VLMs privacy threat resilience.
>
---
#### [new 010] Intriguing Properties of Dynamic Sampling Networks
- **分类: cs.CV**

- **简介: 该论文研究动态采样网络的理论机制，针对现有方法缺乏统一分析的问题，提出“扭曲”算子统一建模多种动态采样结构。通过统计分析与实验，揭示其前向/反向不对称性及非平移不变特性，阐明稳定训练条件，并提出基于梯度更新的损失景观可视化方法。**

- **链接: [https://arxiv.org/pdf/2511.20800v1](https://arxiv.org/pdf/2511.20800v1)**

> **作者:** Dario Morle; Reid Zaffino
>
> **摘要:** Dynamic sampling mechanisms in deep learning architectures have demonstrated utility across many computer vision models, though the theoretical analysis of these structures has not yet been unified. In this paper we connect the various dynamic sampling methods by developing and analyzing a novel operator which generalizes existing methods, which we term "warping". Warping provides a minimal implementation of dynamic sampling which is amenable to analysis, and can be used to reconstruct existing architectures including deformable convolutions, active convolutional units, and spatial transformer networks. Using our formalism, we provide statistical analysis of the operator by modeling the inputs as both IID variables and homogeneous random fields. Extending this analysis, we discover a unique asymmetry between the forward and backward pass of the model training. We demonstrate that these mechanisms represent an entirely different class of orthogonal operators to the traditional translationally invariant operators defined by convolutions. With a combination of theoretical analysis and empirical investigation, we find the conditions necessary to ensure stable training of dynamic sampling networks. In addition, statistical analysis of discretization effects are studied. Finally, we introduce a novel loss landscape visualization which utilizes gradient update information directly, to better understand learning behavior.
>
---
#### [new 011] Unsupervised Memorability Modeling from Tip-of-the-Tongue Retrieval Queries
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对视觉内容记忆性研究中人工标注成本高、数据稀缺的问题，提出首个大规模无监督记忆性数据集，利用Reddit上的“舌尖效应”检索查询构建8.2万条视频与回忆描述。通过对比学习训练模型实现跨模态舌尖检索，并推动开放回忆生成任务，显著提升生成质量。**

- **链接: [https://arxiv.org/pdf/2511.20854v1](https://arxiv.org/pdf/2511.20854v1)**

> **作者:** Sree Bhattacharyya; Yaman Kumar Singla; Sudhir Yarram; Somesh Kumar Singh; Harini S; James Z. Wang
>
> **备注:** Accepted at WACV 2026
>
> **摘要:** Visual content memorability has intrigued the scientific community for decades, with applications ranging widely, from understanding nuanced aspects of human memory to enhancing content design. A significant challenge in progressing the field lies in the expensive process of collecting memorability annotations from humans. This limits the diversity and scalability of datasets for modeling visual content memorability. Most existing datasets are limited to collecting aggregate memorability scores for visual content, not capturing the nuanced memorability signals present in natural, open-ended recall descriptions. In this work, we introduce the first large-scale unsupervised dataset designed explicitly for modeling visual memorability signals, containing over 82,000 videos, accompanied by descriptive recall data. We leverage tip-of-the-tongue (ToT) retrieval queries from online platforms such as Reddit. We demonstrate that our unsupervised dataset provides rich signals for two memorability-related tasks: recall generation and ToT retrieval. Large vision-language models fine-tuned on our dataset outperform state-of-the-art models such as GPT-4o in generating open-ended memorability descriptions for visual content. We also employ a contrastive training strategy to create the first model capable of performing multimodal ToT retrieval. Our dataset and models present a novel direction, facilitating progress in visual content memorability research.
>
---
#### [new 012] Inversion-Free Style Transfer with Dual Rectified Flows
- **分类: cs.CV**

- **简介: 该论文针对风格迁移任务，解决现有方法依赖耗时且易出错的反演过程的问题。提出无反演的双修正流框架，通过前向传播并行预测内容与风格轨迹，动态融合生成高质量风格化图像，提升效率与视觉质量。**

- **链接: [https://arxiv.org/pdf/2511.20986v1](https://arxiv.org/pdf/2511.20986v1)**

> **作者:** Yingying Deng; Xiangyu He; Fan Tang; Weiming Dong; Xucheng Yin
>
> **摘要:** Style transfer, a pivotal task in image processing, synthesizes visually compelling images by seamlessly blending realistic content with artistic styles, enabling applications in photo editing and creative design. While mainstream training-free diffusion-based methods have greatly advanced style transfer in recent years, their reliance on computationally inversion processes compromises efficiency and introduces visual distortions when inversion is inaccurate. To address these limitations, we propose a novel \textit{inversion-free} style transfer framework based on dual rectified flows, which tackles the challenge of finding an unknown stylized distribution from two distinct inputs (content and style images), \textit{only with forward pass}. Our approach predicts content and style trajectories in parallel, then fuses them through a dynamic midpoint interpolation that integrates velocities from both paths while adapting to the evolving stylized image. By jointly modeling the content, style, and stylized distributions, our velocity field design achieves robust fusion and avoids the shortcomings of naive overlays. Attention injection further guides style integration, enhancing visual fidelity, content preservation, and computational efficiency. Extensive experiments demonstrate generalization across diverse styles and content, providing an effective and efficient pipeline for style transfer.
>
---
#### [new 013] You Can Trust Your Clustering Model: A Parameter-free Self-Boosting Plug-in for Deep Clustering
- **分类: cs.CV**

- **简介: 该论文针对深度聚类中全局特征结构分离度差的问题，提出无需调参的DCBoost插件。通过自适应局部一致性筛选高置信样本，构建判别损失以增强类内紧凑性和类间可分性，显著提升现有模型性能。**

- **链接: [https://arxiv.org/pdf/2511.21193v1](https://arxiv.org/pdf/2511.21193v1)**

> **作者:** Hanyang Li; Yuheng Jia; Hui Liu; Junhui Hou
>
> **备注:** The paper is accepted by NeurIPS 2025
>
> **摘要:** Recent deep clustering models have produced impressive clustering performance. However, a common issue with existing methods is the disparity between global and local feature structures. While local structures typically show strong consistency and compactness within class samples, global features often present intertwined boundaries and poorly separated clusters. Motivated by this observation, we propose DCBoost, a parameter-free plug-in designed to enhance the global feature structures of current deep clustering models. By harnessing reliable local structural cues, our method aims to elevate clustering performance effectively. Specifically, we first identify high-confidence samples through adaptive $k$-nearest neighbors-based consistency filtering, aiming to select a sufficient number of samples with high label reliability to serve as trustworthy anchors for self-supervision. Subsequently, these samples are utilized to compute a discriminative loss, which promotes both intra-class compactness and inter-class separability, to guide network optimization. Extensive experiments across various benchmark datasets showcase that our DCBoost significantly improves the clustering performance of diverse existing deep clustering models. Notably, our method improves the performance of current state-of-the-art baselines (e.g., ProPos) by more than 3% and amplifies the silhouette coefficient by over $7\times$. Code is available at <https://github.com/l-h-y168/DCBoost>.
>
---
#### [new 014] Hybrid SIFT-SNN for Efficient Anomaly Detection of Traffic Flow-Control Infrastructure
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对交通基础设施异常检测任务，提出SIFT-SNN框架，融合SIFT特征与脉冲神经网络，实现低延迟、低功耗的实时检测。通过真实与合成数据训练，在桥梁数据集上达92.3%准确率，推理仅9.5ms，支持边缘部署与可解释决策。**

- **链接: [https://arxiv.org/pdf/2511.21337v1](https://arxiv.org/pdf/2511.21337v1)**

> **作者:** Munish Rathee; Boris Bačić; Maryam Doborjeh
>
> **备注:** 8 pages, 6 figures. This is a preprint of a paper accepted for presentation at the 2025 International Conference on Image and Vision Computing New Zealand (IVCNZ). The final version will appear in IEEE Xplore
>
> **摘要:** This paper presents the SIFT-SNN framework, a low-latency neuromorphic signal-processing pipeline for real-time detection of structural anomalies in transport infrastructure. The proposed approach integrates Scale-Invariant Feature Transform (SIFT) for spatial feature encoding with a latency-driven spike conversion layer and a Leaky Integrate-and-Fire (LIF) Spiking Neural Network (SNN) for classification. The Auckland Harbour Bridge dataset is recorded under various weather and lighting conditions, comprising 6,000 labelled frames that include both real and synthetically augmented unsafe cases. The presented system achieves a classification accuracy of 92.3% (+- 0.8%) with a per-frame inference time of 9.5 ms. Achieved sub-10 millisecond latency, combined with sparse spike activity (8.1%), enables real-time, low-power edge deployment. Unlike conventional CNN-based approaches, the hybrid SIFT-SNN pipeline explicitly preserves spatial feature grounding, enhances interpretability, supports transparent decision-making, and operates efficiently on embedded hardware. Although synthetic augmentation improved robustness, generalisation to unseen field conditions remains to be validated. The SIFT-SNN framework is validated through a working prototype deployed on a consumer-grade system and framed as a generalisable case study in structural safety monitoring for movable concrete barriers, which, as a traffic flow-control infrastructure, is deployed in over 20 cities worldwide.
>
---
#### [new 015] GaINeR: Geometry-Aware Implicit Network Representation
- **分类: cs.CV**

- **简介: 该论文提出GaINeR，一种面向2D图像的几何感知隐式网络表示方法。针对传统隐式神经表示缺乏显式几何结构、难以支持局部编辑与物理模拟的问题，通过结合可训练高斯分布与神经网络，实现连续图像表示与可解释几何结构，支持灵活局部编辑与交互式操作。**

- **链接: [https://arxiv.org/pdf/2511.20924v1](https://arxiv.org/pdf/2511.20924v1)**

> **作者:** Weronika Jakubowska; Mikołaj Zieliński; Rafał Tobiasz; Krzysztof Byrski; Maciej Zięba; Dominik Belter; Przemysław Spurek
>
> **备注:** 16 pages, 16 figures
>
> **摘要:** Implicit Neural Representations (INRs) have become an essential tool for modeling continuous 2D images, enabling high-fidelity reconstruction, super-resolution, and compression. Popular architectures such as SIREN, WIRE, and FINER demonstrate the potential of INR for capturing fine-grained image details. However, traditional INRs often lack explicit geometric structure and have limited capabilities for local editing or integration with physical simulation, restricting their applicability in dynamic or interactive settings. To address these limitations, we propose GaINeR: Geometry-Aware Implicit Network Representation, a novel framework for 2D images that combines trainable Gaussian distributions with a neural network-based INR. For a given image coordinate, the model retrieves the K nearest Gaussians, aggregates distance-weighted embeddings, and predicts the RGB value via a neural network. This design enables continuous image representation, interpretable geometric structure, and flexible local editing, providing a foundation for physically aware and interactive image manipulation. The official implementation of our method is publicly available at https://github.com/WJakubowska/GaINeR.
>
---
#### [new 016] MIRA: Multimodal Iterative Reasoning Agent for Image Editing
- **分类: cs.CV**

- **简介: 该论文针对指令引导的图像编辑中复杂语义理解困难的问题，提出轻量级多模态迭代推理代理MIRA。通过感知-推理-行动循环，逐步生成原子编辑指令，结合视觉反馈提升准确性。基于自建数据集与两阶段训练，显著改善编辑的语义一致性和感知质量，性能媲美或超越商用系统。**

- **链接: [https://arxiv.org/pdf/2511.21087v1](https://arxiv.org/pdf/2511.21087v1)**

> **作者:** Ziyun Zeng; Hang Hua; Jiebo Luo
>
> **摘要:** Instruction-guided image editing offers an intuitive way for users to edit images with natural language. However, diffusion-based editing models often struggle to accurately interpret complex user instructions, especially those involving compositional relationships, contextual cues, or referring expressions, leading to edits that drift semantically or fail to reflect the intended changes. We tackle this problem by proposing MIRA (Multimodal Iterative Reasoning Agent), a lightweight, plug-and-play multimodal reasoning agent that performs editing through an iterative perception-reasoning-action loop, effectively simulating multi-turn human-model interaction processes. Instead of issuing a single prompt or static plan, MIRA predicts atomic edit instructions step by step, using visual feedback to make its decisions. Our 150K multimodal tool-use dataset, MIRA-Editing, combined with a two-stage SFT + GRPO training pipeline, enables MIRA to perform reasoning and editing over complex editing instructions. When paired with open-source image editing models such as Flux.1-Kontext, Step1X-Edit, and Qwen-Image-Edit, MIRA significantly improves both semantic consistency and perceptual quality, achieving performance comparable to or exceeding proprietary systems such as GPT-Image and Nano-Banana.
>
---
#### [new 017] Thinking With Bounding Boxes: Enhancing Spatio-Temporal Video Grounding via Reinforcement Fine-Tuning
- **分类: cs.CV**

- **简介: 该论文针对视频时空定位任务，解决多模态大模型因训练目标错位和区域-词汇对齐弱导致性能不佳的问题。提出STVG-o1框架，通过边界框思维链与多维强化奖励机制，实现无需架构修改的SOTA性能，显著提升定位精度与跨数据集泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.21375v1](https://arxiv.org/pdf/2511.21375v1)**

> **作者:** Xin Gu; Haoji Zhang; Qihang Fan; Jingxuan Niu; Zhipeng Zhang; Libo Zhang; Guang Chen; Fan Chen; Longyin Wen; Sijie Zhu
>
> **摘要:** Spatio-temporal video grounding (STVG) requires localizing a target object in untrimmed videos both temporally and spatially from natural language descriptions. Despite their strong language understanding, multimodal large language models (MLLMs) underperform on STVG due to misaligned training objectives and weak fine-grained region-word alignment in standard visual encoders. To address this, we propose STVG-o1, the first framework that enables off-the-shelf MLLMs to achieve state-of-the-art STVG performance without any architectural modifications. Our method introduces a bounding-box chain-of-thought mechanism that explicitly reasons about spatio-temporal locations in an intermediate step before producing the final prediction. We further design a multi-dimensional reinforcement reward function consisting of format, consistency, temporal, spatial, and think rewards, which provides geometry-aware supervision through reinforcement fine-tuning. Evaluated on HCSTVG-v1/v2 and VidSTG, STVG-o1 sets new state-of-the-art results on HCSTVG, outperforming the best task-specific method by 7.3\% m\_tIoU on HCSTVG-v1, matching specialized models on VidSTG, and surpassing all existing MLLM-based approaches by large margins. It also demonstrates strong open-vocabulary generalization across datasets, establishing MLLMs as viable and powerful backbones for precise spatio-temporal grounding. Our code and models will be released.
>
---
#### [new 018] G$^2$VLM: Geometry Grounded Vision Language Model with Unified 3D Reconstruction and Spatial Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出G²VLM，一种基于几何的视觉语言模型，旨在解决视觉语言模型在空间理解与推理上的不足。通过融合3D重建与空间推理，利用多视角图像学习3D几何特征，提升空间智能，实现高效、可扩展的统一框架。**

- **链接: [https://arxiv.org/pdf/2511.21688v1](https://arxiv.org/pdf/2511.21688v1)**

> **作者:** Wenbo Hu; Jingli Lin; Yilin Long; Yunlong Ran; Lihan Jiang; Yifan Wang; Chenming Zhu; Runsen Xu; Tai Wang; Jiangmiao Pang
>
> **备注:** code are released at https://github.com/InternRobotics/G2VLM
>
> **摘要:** Vision-Language Models (VLMs) still lack robustness in spatial intelligence, demonstrating poor performance on spatial understanding and reasoning tasks. We attribute this gap to the absence of a visual geometry learning process capable of reconstructing 3D space from 2D images. We present G$^2$VLM, a geometry grounded vision-language model that bridges two fundamental aspects of spatial intelligence: spatial 3D reconstruction and spatial understanding. G$^2$VLM natively leverages learned 3D visual geometry features to directly predict 3D attributes and enhance spatial reasoning tasks via in-context learning and interleaved reasoning. Our unified design is highly scalable for spatial understanding: it trains on abundant multi-view image and video data, while simultaneously leveraging the benefits of 3D visual priors that are typically only derived from hard-to-collect annotations. Experimental results demonstrate G$^2$VLM is proficient in both tasks, achieving comparable results to state-of-the-art feed-forward 3D reconstruction models and achieving better or competitive results across spatial understanding and reasoning tasks. By unifying a semantically strong VLM with low-level 3D vision tasks, we hope G$^2$VLM can serve as a strong baseline for the community and unlock more future applications, such as 3D scene editing.
>
---
#### [new 019] SAM Guided Semantic and Motion Changed Region Mining for Remote Sensing Change Captioning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对遥感变化描述任务，解决现有方法区域感知弱、时序对齐不足的问题。提出基于SAM模型挖掘语义与运动变化区域，融合全局特征与知识图谱，通过跨注意力机制实现多源信息融合，生成更精准的变化描述。**

- **链接: [https://arxiv.org/pdf/2511.21420v1](https://arxiv.org/pdf/2511.21420v1)**

> **作者:** Futian Wang; Mengqi Wang; Xiao Wang; Haowen Wang; Jin Tang
>
> **摘要:** Remote sensing change captioning is an emerging and popular research task that aims to describe, in natural language, the content of interest that has changed between two remote sensing images captured at different times. Existing methods typically employ CNNs/Transformers to extract visual representations from the given images or incorporate auxiliary tasks to enhance the final results, with weak region awareness and limited temporal alignment. To address these issues, this paper explores the use of the SAM (Segment Anything Model) foundation model to extract region-level representations and inject region-of-interest knowledge into the captioning framework. Specifically, we employ a CNN/Transformer model to extract global-level vision features, leverage the SAM foundation model to delineate semantic- and motion-level change regions, and utilize a specially constructed knowledge graph to provide information about objects of interest. These heterogeneous sources of information are then fused via cross-attention, and a Transformer decoder is used to generate the final natural language description of the observed changes. Extensive experimental results demonstrate that our method achieves state-of-the-art performance across multiple widely used benchmark datasets. The source code of this paper will be released on https://github.com/Event-AHU/SAM_ChangeCaptioning
>
---
#### [new 020] Multimodal Robust Prompt Distillation for 3D Point Cloud Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对3D点云模型易受对抗攻击的问题，提出多模态鲁棒提示蒸馏（MRPD）框架。通过融合视觉、3D与文本模态的稳健特征，实现轻量级提示学习，在训练阶段完成知识迁移，无需推理开销。实验表明，MRPD在多种攻击下显著提升防御性能，同时保持干净数据下的高精度。**

- **链接: [https://arxiv.org/pdf/2511.21574v1](https://arxiv.org/pdf/2511.21574v1)**

> **作者:** Xiang Gu; Liming Lu; Xu Zheng; Anan Du; Yongbin Zhou; Shuchao Pang
>
> **摘要:** Adversarial attacks pose a significant threat to learning-based 3D point cloud models, critically undermining their reliability in security-sensitive applications. Existing defense methods often suffer from (1) high computational overhead and (2) poor generalization ability across diverse attack types. To bridge these gaps, we propose a novel yet efficient teacher-student framework, namely Multimodal Robust Prompt Distillation (MRPD) for distilling robust 3D point cloud model. It learns lightweight prompts by aligning student point cloud model's features with robust embeddings from three distinct teachers: a vision model processing depth projections, a high-performance 3D model, and a text encoder. To ensure a reliable knowledge transfer, this distillation is guided by a confidence-gated mechanism which dynamically balances the contribution of all input modalities. Notably, since the distillation is all during the training stage, there is no additional computational cost at inference. Extensive experiments demonstrate that MRPD substantially outperforms state-of-the-art defense methods against a wide range of white-box and black-box attacks, while even achieving better performance on clean data. Our work presents a new, practical paradigm for building robust 3D vision systems by efficiently harnessing multimodal knowledge.
>
---
#### [new 021] A deep learning model to reduce agent dose for contrast-enhanced MRI of the cerebellopontine angle cistern
- **分类: cs.CV**

- **简介: 该论文针对脑桥小脑角池增强MRI造影剂剂量过高问题，提出一种深度学习模型，通过低剂量模拟数据恢复高质量图像。实验表明，使用10%-30%标准剂量结合DL重建，可显著提升图像质量与病灶分割精度，实现低剂量下的准确诊断。**

- **链接: [https://arxiv.org/pdf/2511.20926v1](https://arxiv.org/pdf/2511.20926v1)**

> **作者:** Yunjie Chen; Rianne A. Weber; Olaf M. Neve; Stephan R. Romeijn; Erik F. Hensen; Jelmer M. Wolterink; Qian Tao; Marius Staring; Berit M. Verbist
>
> **摘要:** Objectives: To evaluate a deep learning (DL) model for reducing the agent dose of contrast-enhanced T1-weighted MRI (T1ce) of the cerebellopontine angle (CPA) cistern. Materials and methods: In this multi-center retrospective study, T1 and T1ce of vestibular schwannoma (VS) patients were used to simulate low-dose T1ce with varying reductions of contrast agent dose. DL models were trained to restore standard-dose T1ce from the low-dose simulation. The image quality and segmentation performance of the DL-restored T1ce were evaluated. A head and neck radiologist was asked to rate DL-restored images in multiple aspects, including image quality and diagnostic characterization. Results: 203 MRI studies from 72 VS patients (mean age, 58.51 \pm 14.73, 39 men) were evaluated. As the input dose increased, the structural similarity index measure of the restored T1ce increased from 0.639 \pm 0.113 to 0.993 \pm 0.009, and the peak signal-to-noise ratio increased from 21.6 \pm 3.73 dB to 41.4 \pm 4.84 dB. At 10% input dose, using DL-restored T1ce for segmentation improved the Dice from 0.673 to 0.734, the 95% Hausdorff distance from 2.38 mm to 2.07 mm, and the average surface distance from 1.00 mm to 0.59 mm. Both DL-restored T1ce from 10% and 30% input doses showed excellent images, with the latter being considered more informative. Conclusion: The DL model improved the image quality of low-dose MRI of the CPA cistern, which makes lesion detection and diagnostic characterization possible with 10% - 30% of the standard dose.
>
---
#### [new 022] Long-Term Alzheimers Disease Prediction: A Novel Image Generation Method Using Temporal Parameter Estimation with Normal Inverse Gamma Distribution on Uneven Time Series
- **分类: cs.CV**

- **简介: 该论文针对阿尔茨海默病（AD）长期预测中不规则时间序列导致特征失真的问题，提出基于正态逆高斯分布的时序参数估计模型T-NIG。通过融合时间参数与坐标邻域特征，生成中间及未来脑图像，有效降低模型不确定性，提升短长期预测性能，保持疾病相关特征。**

- **链接: [https://arxiv.org/pdf/2511.21057v1](https://arxiv.org/pdf/2511.21057v1)**

> **作者:** Xin Hong; Xinze Sun; Yinhao Li; Yen-Wei Chen
>
> **备注:** 13pages, 6 figures
>
> **摘要:** Image generation can provide physicians with an imaging diagnosis basis in the prediction of Alzheimer's Disease (AD). Recent research has shown that long-term AD predictions by image generation often face difficulties maintaining disease-related characteristics when dealing with irregular time intervals in sequential data. Considering that the time-related aspects of the distribution can reflect changes in disease-related characteristics when images are distributed unevenly, this research proposes a model to estimate the temporal parameter within the Normal Inverse Gamma Distribution (T-NIG) to assist in generating images over the long term. The T-NIG model employs brain images from two different time points to create intermediate brain images, forecast future images, and predict the disease. T-NIG is designed by identifying features using coordinate neighborhoods. It incorporates a time parameter into the normal inverse gamma distribution to understand how features change in brain imaging sequences that have varying time intervals. Additionally, T-NIG utilizes uncertainty estimation to reduce both epistemic and aleatoric uncertainties in the model, which arise from insufficient temporal data. In particular, the T-NIG model demonstrates state-of-the-art performance in both short-term and long-term prediction tasks within the dataset. Experimental results indicate that T-NIG is proficient in forecasting disease progression while maintaining disease-related characteristics, even when faced with an irregular temporal data distribution.
>
---
#### [new 023] Efficient Training for Human Video Generation with Entropy-Guided Prioritized Progressive Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对高分辨率人体视频生成中扩散模型训练成本高的问题，提出熵引导的优先级渐进学习框架（Ent-Prog）。通过条件熵膨胀评估模块重要性，实现关键组件优先训练，并设计自适应渐进调度策略，有效降低训练时间与显存消耗，显著提升训练效率。**

- **链接: [https://arxiv.org/pdf/2511.21136v1](https://arxiv.org/pdf/2511.21136v1)**

> **作者:** Changlin Li; Jiawei Zhang; Shuhao Liu; Sihao Lin; Zeyi Shi; Zhihui Li; Xiaojun Chang
>
> **备注:** Project page: https://github.com/changlin31/Ent-Prog
>
> **摘要:** Human video generation has advanced rapidly with the development of diffusion models, but the high computational cost and substantial memory consumption associated with training these models on high-resolution, multi-frame data pose significant challenges. In this paper, we propose Entropy-Guided Prioritized Progressive Learning (Ent-Prog), an efficient training framework tailored for diffusion models on human video generation. First, we introduce Conditional Entropy Inflation (CEI) to assess the importance of different model components on the target conditional generation task, enabling prioritized training of the most critical components. Second, we introduce an adaptive progressive schedule that adaptively increases computational complexity during training by measuring the convergence efficiency. Ent-Prog reduces both training time and GPU memory consumption while maintaining model performance. Extensive experiments across three datasets, demonstrate the effectiveness of Ent-Prog, achieving up to 2.2$\times$ training speedup and 2.4$\times$ GPU memory reduction without compromising generative performance.
>
---
#### [new 024] Seeing without Pixels: Perception from Camera Trajectories
- **分类: cs.CV**

- **简介: 该论文研究如何仅通过相机轨迹感知视频内容，而非依赖像素。提出CamFormer模型，利用对比学习将相机位姿轨迹映射到与自然语言对齐的嵌入空间，实现跨模态理解。验证了轨迹作为轻量、鲁棒、通用的视频感知模态的有效性，解决了无像素视频理解问题。**

- **链接: [https://arxiv.org/pdf/2511.21681v1](https://arxiv.org/pdf/2511.21681v1)**

> **作者:** Zihui Xue; Kristen Grauman; Dima Damen; Andrew Zisserman; Tengda Han
>
> **备注:** Project website: https://sites.google.com/view/seeing-without-pixels
>
> **摘要:** Can one perceive a video's content without seeing its pixels, just from the camera trajectory-the path it carves through space? This paper is the first to systematically investigate this seemingly implausible question. Towards this end, we propose a contrastive learning framework to train CamFormer, a dedicated encoder that projects camera pose trajectories into a joint embedding space, aligning them with natural language. We find that, contrary to its apparent simplicity, the camera trajectory is a remarkably informative signal to uncover video content. In other words, "how you move" can indeed reveal "what you are doing" (egocentric) or "observing" (exocentric). We demonstrate the versatility of our learned CamFormer embeddings on a diverse suite of downstream tasks, ranging from cross-modal alignment to classification and temporal analysis. Importantly, our representations are robust across diverse camera pose estimation methods, including both high-fidelity multi-sensored and standard RGB-only estimators. Our findings establish camera trajectory as a lightweight, robust, and versatile modality for perceiving video content.
>
---
#### [new 025] AnchorOPT: Towards Optimizing Dynamic Anchors for Adaptive Prompt Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对提示学习中静态锚点缺乏适应性的问题，提出AnchorOPT框架。通过动态学习锚点值与可优化的位置矩阵，实现跨任务和训练阶段的自适应。在CLIP基础上，分两阶段训练，显著提升模型泛化能力，且无需额外模块即可取得优异性能。**

- **链接: [https://arxiv.org/pdf/2511.21188v1](https://arxiv.org/pdf/2511.21188v1)**

> **作者:** Zheng Li; Yibing Song; Xin Zhang; Lei Luo; Xiang Li; Jian Yang
>
> **备注:** Technical Report
>
> **摘要:** Existing prompt learning methods, which are built upon CLIP models, leverage textual tokens as anchors to guide the learnable soft tokens. This guidance improves CLIP generalizations. However, these anchors-static in both value and position-lack cross-task and stage-adaptive flexibility. To address this limitation, we propose AnchorOPT, a dynamic anchor-based prompt learning framework. Specifically, AnchorOPT introduces dynamism in two key dimensions: (i) anchor values eschew handcrafted explicit textual tokens (e.g., "shape", "color"), instead learning dynamically from task-specific data; and (ii) the positional relationship between anchor and soft tokens is no longer fixed but adaptively optimized via a learnable position matrix conditioned on the training stage and task context. Training occurs in two stages: we first learn the anchor tokens, then freeze and transfer them to the second stage for optimization of soft tokens and the position matrix. Extensive experiments demonstrate that using only a simple learnable anchor and position matrix achieves performance comparable to or exceeding some methods incorporating additional learnable modules or regularization techniques. As a plug-and-play module, AnchorOPT integrates seamlessly into existing frameworks, yielding consistent performance gains across diverse datasets. Code is publicly available at https://github.com/zhengli97/ATPrompt.
>
---
#### [new 026] TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models
- **分类: cs.CV**

- **简介: 该论文针对文本到视频（T2V）模型的安全风险，提出TEAR框架，旨在发现由动态时序引发的潜在违规内容。通过时序感知的自动化红队测试，生成隐蔽且高效的恶意提示，显著提升攻击成功率至80%以上，有效评估T2V模型在复杂时间序列中的安全漏洞。**

- **链接: [https://arxiv.org/pdf/2511.21145v1](https://arxiv.org/pdf/2511.21145v1)**

> **作者:** Jiaming He; Guanyu Hou; Hongwei Li; Zhicong Huang; Kangjie Chen; Yi Yu; Wenbo Jiang; Guowen Xu; Tianwei Zhang
>
> **摘要:** Text-to-Video (T2V) models are capable of synthesizing high-quality, temporally coherent dynamic video content, but the diverse generation also inherently introduces critical safety challenges. Existing safety evaluation methods,which focus on static image and text generation, are insufficient to capture the complex temporal dynamics in video generation. To address this, we propose a TEmporal-aware Automated Red-teaming framework, named TEAR, an automated framework designed to uncover safety risks specifically linked to the dynamic temporal sequencing of T2V models. TEAR employs a temporal-aware test generator optimized via a two-stage approach: initial generator training and temporal-aware online preference learning, to craft textually innocuous prompts that exploit temporal dynamics to elicit policy-violating video output. And a refine model is adopted to improve the prompt stealthiness and adversarial effectiveness cyclically. Extensive experimental evaluation demonstrates the effectiveness of TEAR across open-source and commercial T2V systems with over 80% attack success rate, a significant boost from prior best result of 57%.
>
---
#### [new 027] Foundry: Distilling 3D Foundation Models for the Edge
- **分类: cs.CV; cs.AI; cs.LG; cs.NE**

- **简介: 该论文针对大模型在边缘设备部署困难的问题，提出FMD范式与Foundry方法。通过知识蒸馏压缩3D点云基础模型，使学生模型以少量超令牌保留教师模型的通用表征能力，实现高效、通用的边缘部署。**

- **链接: [https://arxiv.org/pdf/2511.20721v1](https://arxiv.org/pdf/2511.20721v1)**

> **作者:** Guillaume Letellier; Siddharth Srivastava; Frédéric Jurie; Gaurav Sharma
>
> **摘要:** Foundation models pre-trained with self-supervised learning (SSL) on large-scale datasets have become powerful general-purpose feature extractors. However, their immense size and computational cost make them prohibitive for deployment on edge devices such as robots and AR/VR headsets. Existing compression techniques like standard knowledge distillation create efficient 'specialist' models but sacrifice the crucial, downstream-agnostic generality that makes foundation models so valuable. In this paper, we introduce Foundation Model Distillation (FMD), a new paradigm for compressing large SSL models into compact, efficient, and faithful proxies that retain their general-purpose representational power. We present Foundry, the first implementation of FMD for 3D point clouds. Our approach, Foundry, trains a student to learn a compressed set of SuperTokens that reconstruct the teacher's token-level representations, capturing a compact basis of its latent space. A single distilled model maintains strong transferability across diverse downstream tasks-classification, part segmentation, and few-shot scenarios-approaching full foundation-model performance while using significantly fewer tokens and FLOPs, making such models more practical for deployment on resourceconstrained hardware.
>
---
#### [new 028] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition
- **分类: cs.CV**

- **简介: 该论文将生成式图像修复模型用于图像层分解任务，解决单图分层难题。通过轻量微调扩散模型，并引入线性复杂度的多模态上下文融合模块，在合成数据上实现高效细节保留，提升物体移除与遮挡恢复效果，推动图像编辑与创作应用。**

- **链接: [https://arxiv.org/pdf/2511.20996v1](https://arxiv.org/pdf/2511.20996v1)**

> **作者:** Jingxi Chen; Yixiao Zhang; Xiaoye Qian; Zongxia Li; Cornelia Fermuller; Caren Chen; Yiannis Aloimonos
>
> **摘要:** Images can be viewed as layered compositions, foreground objects over background, with potential occlusions. This layered representation enables independent editing of elements, offering greater flexibility for content creation. Despite the progress in large generative models, decomposing a single image into layers remains challenging due to limited methods and data. We observe a strong connection between layer decomposition and in/outpainting tasks, and propose adapting a diffusion-based inpainting model for layer decomposition using lightweight finetuning. To further preserve detail in the latent space, we introduce a novel multi-modal context fusion module with linear attention complexity. Our model is trained purely on a synthetic dataset constructed from open-source assets and achieves superior performance in object removal and occlusion recovery, unlocking new possibilities in downstream editing and creative applications.
>
---
#### [new 029] V$^{2}$-SAM: Marrying SAM2 with Multi-Prompt Experts for Cross-View Object Correspondence
- **分类: cs.CV**

- **简介: 该论文针对跨视角物体对应任务，解决因视角与外观差异导致的匹配难题。提出V²-SAM框架，融合几何与视觉双提示生成器，并引入多专家选择机制，实现精准跨视角物体对应，显著提升在多个基准上的性能。**

- **链接: [https://arxiv.org/pdf/2511.20886v1](https://arxiv.org/pdf/2511.20886v1)**

> **作者:** Jiancheng Pan; Runze Wang; Tianwen Qian; Mohammad Mahdi; Yanwei Fu; Xiangyang Xue; Xiaomeng Huang; Luc Van Gool; Danda Pani Paudel; Yuqian Fu
>
> **备注:** 19 pages
>
> **摘要:** Cross-view object correspondence, exemplified by the representative task of ego-exo object correspondence, aims to establish consistent associations of the same object across different viewpoints (e.g., ego-centric and exo-centric). This task poses significant challenges due to drastic viewpoint and appearance variations, making existing segmentation models, such as SAM2, non-trivial to apply directly. To address this, we present V^2-SAM, a unified cross-view object correspondence framework that adapts SAM2 from single-view segmentation to cross-view correspondence through two complementary prompt generators. Specifically, the Cross-View Anchor Prompt Generator (V^2-Anchor), built upon DINOv3 features, establishes geometry-aware correspondences and, for the first time, unlocks coordinate-based prompting for SAM2 in cross-view scenarios, while the Cross-View Visual Prompt Generator (V^2-Visual) enhances appearance-guided cues via a novel visual prompt matcher that aligns ego-exo representations from both feature and structural perspectives. To effectively exploit the strengths of both prompts, we further adopt a multi-expert design and introduce a Post-hoc Cyclic Consistency Selector (PCCS) that adaptively selects the most reliable expert based on cyclic consistency. Extensive experiments validate the effectiveness of V^2-SAM, achieving new state-of-the-art performance on Ego-Exo4D (ego-exo object correspondence), DAVIS-2017 (video object tracking), and HANDAL-X (robotic-ready cross-view correspondence).
>
---
#### [new 030] LaGen: Towards Autoregressive LiDAR Scene Generation
- **分类: cs.CV**

- **简介: 该论文提出LaGen，首个支持长时序交互式生成的自回归LiDAR场景生成框架。针对现有方法无法实现逐帧交互生成的问题，提出基于单帧输入与边界框条件的4D点云生成方案，并引入场景解耦与噪声调制模块，有效提升生成质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2511.21256v1](https://arxiv.org/pdf/2511.21256v1)**

> **作者:** Sizhuo Zhou; Xiaosong Jia; Fanrui Zhang; Junjie Li; Juyong Zhang; Yukang Feng; Jianwen Sun; Songbur Wong; Junqi You; Junchi Yan
>
> **摘要:** Generative world models for autonomous driving (AD) have become a trending topic. Unlike the widely studied image modality, in this work we explore generative world models for LiDAR data. Existing generation methods for LiDAR data only support single frame generation, while existing prediction approaches require multiple frames of historical input and can only deterministically predict multiple frames at once, lacking interactivity. Both paradigms fail to support long-horizon interactive generation. To this end, we introduce LaGen, which to the best of our knowledge is the first framework capable of frame-by-frame autoregressive generation of long-horizon LiDAR scenes. LaGen is able to take a single-frame LiDAR input as a starting point and effectively utilize bounding box information as conditions to generate high-fidelity 4D scene point clouds. In addition, we introduce a scene decoupling estimation module to enhance the model's interactive generation capability for object-level content, as well as a noise modulation module to mitigate error accumulation during long-horizon generation. We construct a protocol based on nuScenes for evaluating long-horizon LiDAR scene generation. Experimental results comprehensively demonstrate LaGen outperforms state-of-the-art LiDAR generation and prediction models, especially on the later frames.
>
---
#### [new 031] CanKD: Cross-Attention-based Non-local operation for Feature-based Knowledge Distillation
- **分类: cs.CV**

- **简介: 该论文针对计算机视觉中的知识蒸馏任务，提出CanKD框架，通过跨注意力机制实现非局部特征知识迁移，使学生模型动态关注教师模型所有像素，增强特征表达。仅引入额外损失函数，显著提升检测与分割性能，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.21503v1](https://arxiv.org/pdf/2511.21503v1)**

> **作者:** Shizhe Sun; Wataru Ohyama
>
> **备注:** WACV 2026 Accepted
>
> **摘要:** We propose Cross-Attention-based Non-local Knowledge Distillation (CanKD), a novel feature-based knowledge distillation framework that leverages cross-attention mechanisms to enhance the knowledge transfer process. Unlike traditional self-attention-based distillation methods that align teacher and student feature maps independently, CanKD enables each pixel in the student feature map to dynamically consider all pixels in the teacher feature map. This non-local knowledge transfer more thoroughly captures pixel-wise relationships, improving feature representation learning. Our method introduces only an additional loss function to achieve superior performance compared with existing attention-guided distillation methods. Extensive experiments on object detection and image segmentation tasks demonstrate that CanKD outperforms state-of-the-art feature and hybrid distillation methods. These experimental results highlight CanKD's potential as a new paradigm for attention-guided distillation in computer vision tasks. Code is available at https://github.com/tori-hotaru/CanKD
>
---
#### [new 032] CaptionQA: Is Your Caption as Useful as the Image Itself?
- **分类: cs.CV**

- **简介: 该论文提出CaptionQA基准，评估图像描述在下游任务中的实用性。针对现有评价忽视描述能否替代图像的问题，构建涵盖4个领域的33,027个需视觉信息的多选题，通过LLM仅用描述回答来衡量其效用。结果显示当前模型在传统评测中表现相近，但在描述实用性上差距达32%。**

- **链接: [https://arxiv.org/pdf/2511.21025v1](https://arxiv.org/pdf/2511.21025v1)**

> **作者:** Shijia Yang; Yunong Liu; Bohan Zhai; Ximeng Sun; Zicheng Liu; Emad Barsoum; Manling Li; Chenfeng Xu
>
> **摘要:** Image captions serve as efficient surrogates for visual content in multimodal systems such as retrieval, recommendation, and multi-step agentic inference pipelines. Yet current evaluation practices miss a fundamental question: Can captions stand-in for images in real downstream tasks? We propose a utility-based benchmark, CaptionQA, to evaluate model-generated captions, where caption quality is measured by how well it supports downstream tasks. CaptionQA is an extensible domain-dependent benchmark covering 4 domains--Natural, Document, E-commerce, and Embodied AI--each with fine-grained taxonomies (25 top-level and 69 subcategories) that identify useful information for domain-specific tasks. CaptionQA builds 33,027 densely annotated multiple-choice questions (50.3 per image on average) that explicitly require visual information to answer, providing a comprehensive probe of caption utility. In our evaluation protocol, an LLM answers these questions using captions alone, directly measuring whether captions preserve image-level utility and are utilizable by a downstream LLM. Evaluating state-of-the-art MLLMs reveals substantial gaps between the image and its caption utility. Notably, models nearly identical on traditional image-QA benchmarks lower by up to 32% in caption utility. We release CaptionQA along with an open-source pipeline for extension to new domains. The code is available at https://github.com/bronyayang/CaptionQA.
>
---
#### [new 033] MetaRank: Task-Aware Metric Selection for Model Transferability Estimation
- **分类: cs.CV**

- **简介: 该论文针对迁移学习中预训练模型选择效率低的问题，提出MetaRank框架，实现任务感知的模型可迁移性评估（MTE）指标自动选择。通过文本编码与元学习，基于目标数据集描述动态排序候选指标，提升选型准确性，避免盲目试错。**

- **链接: [https://arxiv.org/pdf/2511.21007v1](https://arxiv.org/pdf/2511.21007v1)**

> **作者:** Yuhang Liu; Wenjie Zhao; Yunhui Guo
>
> **备注:** 10 figures
>
> **摘要:** Selecting an appropriate pre-trained source model is a critical, yet computationally expensive, task in transfer learning. Model Transferability Estimation (MTE) methods address this by providing efficient proxy metrics to rank models without full fine-tuning. In practice, the choice of which MTE metric to use is often ad hoc or guided simply by a metric's average historical performance. However, we observe that the effectiveness of MTE metrics is highly task-dependent and no single metric is universally optimal across all target datasets. To address this gap, we introduce MetaRank, a meta-learning framework for automatic, task-aware MTE metric selection. We formulate metric selection as a learning-to-rank problem. Rather than relying on conventional meta-features, MetaRank encodes textual descriptions of both datasets and MTE metrics using a pretrained language model, embedding them into a shared semantic space. A meta-predictor is then trained offline on diverse meta-tasks to learn the intricate relationship between dataset characteristics and metric mechanisms, optimized with a listwise objective that prioritizes correctly ranking the top-performing metrics. During the subsequent online phase, MetaRank efficiently ranks the candidate MTE metrics for a new, unseen target dataset based on its textual description, enabling practitioners to select the most appropriate metric a priori. Extensive experiments across 11 pretrained models and 11 target datasets demonstrate the strong effectiveness of our approach.
>
---
#### [new 034] Canvas-to-Image: Compositional Image Generation with Multimodal Controls
- **分类: cs.CV**

- **简介: 该论文提出Canvas-to-Image框架，解决扩散模型在多模态控制下生成图像时的高保真度与意图一致性问题。通过将文本、参考图、布局等异构控制信号编码为统一画布，实现端到端的视觉-空间联合推理，显著提升多对象、姿态、布局等复杂场景下的生成质量与控制精度。**

- **链接: [https://arxiv.org/pdf/2511.21691v1](https://arxiv.org/pdf/2511.21691v1)**

> **作者:** Yusuf Dalva; Guocheng Gordon Qian; Maya Goldenberg; Tsai-Shien Chen; Kfir Aberman; Sergey Tulyakov; Pinar Yanardag; Kuan-Chieh Jackson Wang
>
> **备注:** 24 pages; webpage: https://snap-research.github.io/canvas-to-image/
>
> **摘要:** While modern diffusion models excel at generating high-quality and diverse images, they still struggle with high-fidelity compositional and multimodal control, particularly when users simultaneously specify text prompts, subject references, spatial arrangements, pose constraints, and layout annotations. We introduce Canvas-to-Image, a unified framework that consolidates these heterogeneous controls into a single canvas interface, enabling users to generate images that faithfully reflect their intent. Our key idea is to encode diverse control signals into a single composite canvas image that the model can directly interpret for integrated visual-spatial reasoning. We further curate a suite of multi-task datasets and propose a Multi-Task Canvas Training strategy that optimizes the diffusion model to jointly understand and integrate heterogeneous controls into text-to-image generation within a unified learning paradigm. This joint training enables Canvas-to-Image to reason across multiple control modalities rather than relying on task-specific heuristics, and it generalizes well to multi-control scenarios during inference. Extensive experiments show that Canvas-to-Image significantly outperforms state-of-the-art methods in identity preservation and control adherence across challenging benchmarks, including multi-person composition, pose-controlled composition, layout-constrained generation, and multi-control generation.
>
---
#### [new 035] PathMamba: A Hybrid Mamba-Transformer for Topologically Coherent Road Segmentation in Satellite Imagery
- **分类: cs.CV**

- **简介: 该论文针对卫星影像中道路分割的高精度与拓扑连续性难题，提出PathMamba混合架构。融合Mamba的线性序列建模与Transformer的全局感知能力，用Mamba保持道路连通性，用Transformer增强上下文理解，显著提升分割拓扑质量，同时保持高效计算，在两个数据集上达到新SOTA。**

- **链接: [https://arxiv.org/pdf/2511.21298v1](https://arxiv.org/pdf/2511.21298v1)**

> **作者:** Jules Decaestecker; Nicolas Vigne
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Achieving both high accuracy and topological continuity in road segmentation from satellite imagery is a critical goal for applications ranging from urban planning to disaster response. State-of-the-art methods often rely on Vision Transformers, which excel at capturing global context, yet their quadratic complexity is a significant barrier to efficient deployment, particularly for on-board processing in resource-constrained platforms. In contrast, emerging State Space Models like Mamba offer linear-time efficiency and are inherently suited to modeling long, continuous structures. We posit that these architectures have complementary strengths. To this end, we introduce PathMamba, a novel hybrid architecture that integrates Mamba's sequential modeling with the Transformer's global reasoning. Our design strategically uses Mamba blocks to trace the continuous nature of road networks, preserving topological structure, while integrating Transformer blocks to refine features with global context. This approach yields topologically superior segmentation maps without the prohibitive scaling costs of pure attention-based models. Our experiments on the DeepGlobe Road Extraction and Massachusetts Roads datasets demonstrate that PathMamba sets a new state-of-the-art. Notably, it significantly improves topological continuity, as measured by the APLS metric, setting a new benchmark while remaining computationally competitive.
>
---
#### [new 036] Deep Learning-Based Multiclass Classification of Oral Lesions with Stratified Augmentation
- **分类: cs.CV**

- **简介: 该论文针对口腔病变早期诊断难题，提出基于深度学习的多分类模型。针对数据少且不平衡问题，采用分层分割与增强、过采样策略，实现16类口腔病变准确分类，显著提升少数类识别性能，为临床辅助诊断提供有效解决方案。**

- **链接: [https://arxiv.org/pdf/2511.21582v1](https://arxiv.org/pdf/2511.21582v1)**

> **作者:** Joy Naoum; Revana Salama; Ali Hamdi
>
> **备注:** 12 pages, 3 figures,
>
> **摘要:** Oral cancer is highly common across the globe and is mostly diagnosed during the later stages due to the close visual similarity to benign, precancerous, and malignant lesions in the oral cavity. Implementing computer aided diagnosis systems early on has the potential to greatly improve clinical outcomes. This research intends to use deep learning to build a multiclass classifier for sixteen different oral lesions. To overcome the challenges of limited and imbalanced datasets, the proposed technique combines stratified data splitting and advanced data augmentation and oversampling to perform the classification. The experimental results, which achieved 83.33 percent accuracy, 89.12 percent precision, and 77.31 percent recall, demonstrate the superiority of the suggested model over state of the art methods now in use. The suggested model effectively conveys the effectiveness of oversampling and augmentation strategies in situations where the minority class classification performance is noteworthy. As a first step toward trustworthy computer aided diagnostic systems for the early detection of oral cancer in clinical settings, the suggested framework shows promise.
>
---
#### [new 037] Estimating Fog Parameters from a Sequence of Stereo Images
- **分类: cs.CV**

- **简介: 该论文针对雾天视觉感知中雾参数估计不准确的问题，提出一种基于立体图像序列的实时雾参数联合估计方法。通过构建新数据集SDIRF并设计优化模型，实现对局部均匀雾的动态建模，提升雾参数估计精度与鲁棒性，可集成至SLAM系统中。**

- **链接: [https://arxiv.org/pdf/2511.20865v1](https://arxiv.org/pdf/2511.20865v1)**

> **作者:** Yining Ding; João F. C. Mota; Andrew M. Wallace; Sen Wang
>
> **摘要:** We propose a method which, given a sequence of stereo foggy images, estimates the parameters of a fog model and updates them dynamically. In contrast with previous approaches, which estimate the parameters sequentially and thus are prone to error propagation, our algorithm estimates all the parameters simultaneously by solving a novel optimisation problem. By assuming that fog is only locally homogeneous, our method effectively handles real-world fog, which is often globally inhomogeneous. The proposed algorithm can be easily used as an add-on module in existing visual Simultaneous Localisation and Mapping (SLAM) or odometry systems in the presence of fog. In order to assess our method, we also created a new dataset, the Stereo Driving In Real Fog (SDIRF), consisting of high-quality, consecutive stereo frames of real, foggy road scenes under a variety of visibility conditions, totalling over 40 minutes and 34k frames. As a first-of-its-kind, SDIRF contains the camera's photometric parameters calibrated in a lab environment, which is a prerequisite for correctly applying the atmospheric scattering model to foggy images. The dataset also includes the counterpart clear data of the same routes recorded in overcast weather, which is useful for companion work in image defogging and depth reconstruction. We conducted extensive experiments using both synthetic foggy data and real foggy sequences from SDIRF to demonstrate the superiority of the proposed algorithm over prior methods. Our method not only produces the most accurate estimates on synthetic data, but also adapts better to real fog. We make our code and SDIRF publicly available\footnote{https://github.com/SenseRoboticsLab/estimating-fog-parameters} to the community with the aim of advancing the research on visual perception in fog.
>
---
#### [new 038] Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following
- **分类: cs.CV**

- **简介: 该论文针对多模态模型在复杂评价标准下一致性不足的问题，提出Multi-Crit基准，涵盖开放生成与可验证推理任务，通过精心构建的数据集和新指标评估模型对多元标准的遵循能力。研究揭示了当前大模型在多准则判断中的局限性，并探讨了微调与推理策略的影响，为可调控、可靠的多模态评估提供基础。**

- **链接: [https://arxiv.org/pdf/2511.21662v1](https://arxiv.org/pdf/2511.21662v1)**

> **作者:** Tianyi Xiong; Yi Ge; Ming Li; Zuolong Zhang; Pranav Kulkarni; Kaishen Wang; Qi He; Zeying Zhu; Chenxi Liu; Ruibo Chen; Tong Zheng; Yanshuo Chen; Xiyao Wang; Renrui Zhang; Wenhu Chen; Heng Huang
>
> **摘要:** Large multimodal models (LMMs) are increasingly adopted as judges in multimodal evaluation systems due to their strong instruction following and consistency with human preferences. However, their ability to follow diverse, fine-grained evaluation criteria remains underexplored. We develop Multi-Crit, a benchmark for evaluating multimodal judges on their capacity to follow pluralistic criteria and produce reliable criterion-level judgments. Covering both open-ended generation and verifiable reasoning tasks, Multi-Crit is built through a rigorous data curation pipeline that gathers challenging response pairs with multi-criterion human annotations. It further introduces three novel metrics for systematically assessing pluralistic adherence, criterion-switching flexibility, and the ability to recognize criterion-level preference conflicts. Comprehensive analysis of 25 LMMs reveals that 1) proprietary models still struggle to maintain consistent adherence to pluralistic criteria--especially in open-ended evaluation; 2) open-source models lag further behind in flexibly following diverse criteria; and 3) critic fine-tuning with holistic judgment signals enhances visual grounding but fails to generalize to pluralistic criterion-level judgment. Additional analyses on reasoning fine-tuning, test-time scaling, and boundary consistency between open-source and proprietary models further probe the limits of current multimodal judges. As a pioneering study, Multi-Crit lays the foundation for building reliable and steerable multimodal AI evaluation.
>
---
#### [new 039] Inferix: A Block-Diffusion based Next-Generation Inference Engine for World Simulation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Inferix，一种基于块扩散（block-diffusion）的下一代推理引擎，用于世界模拟任务。针对现有视频生成模型在长序列、交互性与效率上的不足，通过引入半自回归解码与KV缓存机制，实现高效、高质量、可交互的长时视频生成，并集成细粒度评估工具，推动世界模型发展。**

- **链接: [https://arxiv.org/pdf/2511.20714v1](https://arxiv.org/pdf/2511.20714v1)**

> **作者:** Inferix Team; Tianyu Feng; Yizeng Han; Jiahao He; Yuanyu He; Xi Lin; Teng Liu; Hanfeng Lu; Jiasheng Tang; Wei Wang; Zhiyuan Wang; Jichao Wu; Mingyang Yang; Yinghao Yu; Zeyu Zhang; Bohan Zhuang
>
> **摘要:** World models serve as core simulators for fields such as agentic AI, embodied AI, and gaming, capable of generating long, physically realistic, and interactive high-quality videos. Moreover, scaling these models could unlock emergent capabilities in visual perception, understanding, and reasoning, paving the way for a new paradigm that moves beyond current LLM-centric vision foundation models. A key breakthrough empowering them is the semi-autoregressive (block-diffusion) decoding paradigm, which merges the strengths of diffusion and autoregressive methods by generating video tokens in block-applying diffusion within each block while conditioning on previous ones, resulting in more coherent and stable video sequences. Crucially, it overcomes limitations of standard video diffusion by reintroducing LLM-style KV Cache management, enabling efficient, variable-length, and high-quality generation. Therefore, Inferix is specifically designed as a next-generation inference engine to enable immersive world synthesis through optimized semi-autoregressive decoding processes. This dedicated focus on world simulation distinctly sets it apart from systems engineered for high-concurrency scenarios (like vLLM or SGLang) and from classic video diffusion models (such as xDiTs). Inferix further enhances its offering with interactive video streaming and profiling, enabling real-time interaction and realistic simulation to accurately model world dynamics. Additionally, it supports efficient benchmarking through seamless integration of LV-Bench, a new fine-grained evaluation benchmark tailored for minute-long video generation scenarios. We hope the community will work together to advance Inferix and foster world model exploration.
>
---
#### [new 040] BUSTR: Breast Ultrasound Text Reporting with a Descriptor-Aware Vision-Language Model
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出BUSTR，一种无需配对图像-报告数据的乳腺超声文本报告生成框架。针对缺乏标注数据和大模型幻觉问题，通过结构化描述符与影像特征融合，利用多头Swin编码器学习描述符感知的视觉表征，并结合词元级交叉熵与表示相似性对齐损失，实现精准报告生成。在两个公开数据集上验证了其在自然语言与临床指标上的优越性。**

- **链接: [https://arxiv.org/pdf/2511.20956v1](https://arxiv.org/pdf/2511.20956v1)**

> **作者:** Rawa Mohammed; Mina Attin; Bryar Shareef
>
> **备注:** 13 pages, 2 figures, 6 tables
>
> **摘要:** Automated radiology report generation (RRG) for breast ultrasound (BUS) is limited by the lack of paired image-report datasets and the risk of hallucinations from large language models. We propose BUSTR, a multitask vision-language framework that generates BUS reports without requiring paired image-report supervision. BUSTR constructs reports from structured descriptors (e.g., BI-RADS, pathology, histology) and radiomics features, learns descriptor-aware visual representations with a multi-head Swin encoder trained using a multitask loss over dataset-specific descriptor sets, and aligns visual and textual tokens via a dual-level objective that combines token-level cross-entropy with a cosine-similarity alignment loss between input and output representations. We evaluate BUSTR on two public BUS datasets, BrEaST and BUS-BRA, which differ in size and available descriptors. Across both datasets, BUSTR consistently improves standard natural language generation metrics and clinical efficacy metrics, particularly for key targets such as BI-RADS category and pathology. Our results show that this descriptor-aware vision model, trained with a combined token-level and alignment loss, improves both automatic report metrics and clinical efficacy without requiring paired image-report data. The source code can be found at https://github.com/AAR-UNLV/BUSTR
>
---
#### [new 041] CLRecogEye : Curriculum Learning towards exploiting convolution features for Dynamic Iris Recognition
- **分类: cs.CV**

- **简介: 该论文针对动态虹膜识别中旋转、尺度、反光和模糊等挑战，提出基于课程学习的CLRecogEye框架。通过3D-CNN对分割后的虹膜序列建模时空特征，结合课程学习与三元组、ArcFace损失，增强特征判别力，提升识别鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.21097v1](https://arxiv.org/pdf/2511.21097v1)**

> **作者:** Geetanjali Sharma; Gaurav Jaswal; Aditya Nigam; Raghavendra Ramachandra
>
> **备注:** 12 Pages, 3 figures, ISVC conference 2025
>
> **摘要:** Iris authentication algorithms have achieved impressive recognition performance, making them highly promising for real-world applications such as border control, citizen identification, and both criminal investigations and commercial systems. However, their robustness is still challenged by variations in rotation, scale, specular reflections, and defocus blur. In addition, most existing approaches rely on straightforward point-to-point comparisons, typically using cosine or L2 distance, without effectively leveraging the spatio-spatial-temporal structure of iris patterns. To address these limitations, we propose a novel and generalized matching pipeline that learns rich spatio-spatial-temporal representations of iris features. Our approach first splits each iris image along one dimension, generating a sequence of sub-images that serve as input to a 3D-CNN, enabling the network to capture both spatial and spatio-spatial-temporal cues. To further enhance the modeling of spatio-spatial-temporal feature dynamics, we train the model in curriculum manner. This design allows the network to embed temporal dependencies directly into the feature space, improving discriminability in the deep metric domain. The framework is trained end-to-end with triplet and ArcFace loss in a curriculum manner, enforcing highly discriminative embeddings despite challenges like rotation, scale, reflections, and blur. This design yields a robust and generalizable solution for iris authentication.Github code: https://github.com/GeetanjaliGTZ/CLRecogEye
>
---
#### [new 042] LongVT: Incentivizing "Thinking with Long Videos" via Native Tool Calling
- **分类: cs.CV**

- **简介: 该论文针对长视频理解中因证据稀疏导致的幻觉问题，提出LongVT框架。通过利用大模型的时序定位能力作为原生工具，实现全局到局部的多轮视频裁剪与细粒度分析，提升推理准确性。研究构建了VideoSIAH数据集，并设计三阶段训练策略，显著优于现有基线。**

- **链接: [https://arxiv.org/pdf/2511.20785v1](https://arxiv.org/pdf/2511.20785v1)**

> **作者:** Zuhao Yang; Sudong Wang; Kaichen Zhang; Keming Wu; Sicong Leng; Yifan Zhang; Chengwei Qin; Shijian Lu; Xingxuan Li; Lidong Bing
>
> **摘要:** Large multimodal models (LMMs) have shown great potential for video reasoning with textual Chain-of-Thought. However, they remain vulnerable to hallucinations, especially when processing long-form videos where evidence is sparse and temporally dispersed. Inspired by how humans comprehend long videos - by first skimming globally and then examining relevant clips for details - we introduce LongVT, an end-to-end agentic framework that enables "Thinking with Long Videos" via interleaved Multimodal Chain-of-Tool-Thought. Specifically, we exploit LMMs' inherent temporal grounding ability as a native video cropping tool to zoom in on a specific video clip and resample finer-grained video frames. This global-to-local reasoning loop continues until answers are grounded in retrieved visual evidence. Given the scarcity of fine-grained question-answering (QA) data for the long video reasoning task, we curate and will release a data suite named VideoSIAH to facilitate both training and evaluation. Specifically, our training dataset consists of 247.9K samples for tool-integrated cold-start supervised fine-tuning, 1.6K samples for agentic reinforcement learning, and 15.4K samples for agentic reinforcement fine-tuning, respectively. Our evaluation benchmark consists of 1,280 QA pairs that are carefully curated through a semi-automatic data pipeline with human-in-the-loop validation. With a meticulously designed three-stage training strategy and extensive empirical validation, LongVT consistently outperforms existing strong baselines across four challenging long-video understanding and reasoning benchmarks. Our codes, data, and model checkpoints are publicly available at https://github.com/EvolvingLMMs-Lab/LongVT .
>
---
#### [new 043] Test-Time Alignment of Text-to-Image Diffusion Models via Null-Text Embedding Optimisation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对文本到图像扩散模型的测试时对齐（TTA）任务，解决现有方法易过优化或奖励劫持的问题。提出Null-TTA，通过优化分类器自由引导中的无条件嵌入，实现语义空间内的生成分布对齐，避免非语义噪声干扰，提升对齐效果与跨奖励泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.20889v1](https://arxiv.org/pdf/2511.20889v1)**

> **作者:** Taehoon Kim; Henry Gouk; Timothy Hospedales
>
> **摘要:** Test-time alignment (TTA) aims to adapt models to specific rewards during inference. However, existing methods tend to either under-optimise or over-optimise (reward hack) the target reward function. We propose Null-Text Test-Time Alignment (Null-TTA), which aligns diffusion models by optimising the unconditional embedding in classifier-free guidance, rather than manipulating latent or noise variables. Due to the structured semantic nature of the text embedding space, this ensures alignment occurs on a semantically coherent manifold and prevents reward hacking (exploiting non-semantic noise patterns to improve the reward). Since the unconditional embedding in classifier-free guidance serves as the anchor for the model's generative distribution, Null-TTA directly steers model's generative distribution towards the target reward rather than just adjusting the samples, even without updating model parameters. Thanks to these desirable properties, we show that Null-TTA achieves state-of-the-art target test-time alignment while maintaining strong cross-reward generalisation. This establishes semantic-space optimisation as an effective and principled novel paradigm for TTA.
>
---
#### [new 044] Wavefront-Constrained Passive Obscured Object Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对模糊光场中被遮挡物体的精准定位与分割难题，提出物理驱动的WavePCNet模型。通过复振幅传播约束与动量记忆机制，有效抑制多重散射噪声，提升低信噪比下的感知稳定性。引入高频跨层补偿增强结构，实现多尺度特征融合与结构一致性建模，显著提升模型鲁棒性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.20991v1](https://arxiv.org/pdf/2511.20991v1)**

> **作者:** Zhiwen Zheng; Yiwei Ouyang; Zhao Huang; Tao Zhang; Xiaoshuai Zhang; Huiyu Zhou; Wenwen Tang; Shaowei Jiang; Jin Liu; Xingru Huang
>
> **摘要:** Accurately localizing and segmenting obscured objects from faint light patterns beyond the field of view is highly challenging due to multiple scattering and medium-induced perturbations. Most existing methods, based on real-valued modeling or local convolutional operations, are inadequate for capturing the underlying physics of coherent light propagation. Moreover, under low signal-to-noise conditions, these methods often converge to non-physical solutions, severely compromising the stability and reliability of the observation. To address these challenges, we propose a novel physics-driven Wavefront Propagating Compensation Network (WavePCNet) to simulate wavefront propagation and enhance the perception of obscured objects. This WavePCNet integrates the Tri-Phase Wavefront Complex-Propagation Reprojection (TriWCP) to incorporate complex amplitude transfer operators to precisely constrain coherent propagation behavior, along with a momentum memory mechanism to effectively suppress the accumulation of perturbations. Additionally, a High-frequency Cross-layer Compensation Enhancement is introduced to construct frequency-selective pathways with multi-scale receptive fields and dynamically model structural consistency across layers, further boosting the model's robustness and interpretability under complex environmental conditions. Extensive experiments conducted on four physically collected datasets demonstrate that WavePCNet consistently outperforms state-of-the-art methods across both accuracy and robustness.
>
---
#### [new 045] $Δ$-NeRF: Incremental Refinement of Neural Radiance Fields through Residual Control and Knowledge Transfer
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对NeRF增量更新中需重训练、易遗忘的问题，提出Δ-NeRF框架。通过残差控制器、不确定性门控与视图选择策略，实现无需历史数据的高效增量优化，并结合知识蒸馏压缩模型。实验表明其在卫星影像上性能接近联合训练，训练提速30-42%。**

- **链接: [https://arxiv.org/pdf/2511.20804v1](https://arxiv.org/pdf/2511.20804v1)**

> **作者:** Kriti Ghosh; Devjyoti Chakraborty; Lakshmish Ramaswamy; Suchendra M. Bhandarkar; In Kee Kim; Nancy O'Hare; Deepak Mishra
>
> **摘要:** Neural Radiance Fields (NeRFs) have demonstrated remarkable capabilities in 3D reconstruction and novel view synthesis. However, most existing NeRF frameworks require complete retraining when new views are introduced incrementally, limiting their applicability in domains where data arrives sequentially. This limitation is particularly problematic in satellite-based terrain analysis, where regions are repeatedly observed over time. Incremental refinement of NeRFs remains underexplored, and naive approaches suffer from catastrophic forgetting when past data is unavailable. We propose $Δ$-NeRF, a unique modular residual framework for incremental NeRF refinement. $Δ$-NeRF introduces several novel techniques including: (1) a residual controller that injects per-layer corrections into a frozen base NeRF, enabling refinement without access to past data; (2) an uncertainty-aware gating mechanism that prevents overcorrection by adaptively combining base and refined predictions; and (3) a view selection strategy that reduces training data by up to 47\% while maintaining performance. Additionally, we employ knowledge distillation to compress the enhanced model into a compact student network (20\% of original size). Experiments on satellite imagery demonstrate that $Δ$-NeRF achieves performance comparable to joint training while reducing training time by 30-42\%. $Δ$-NeRF consistently outperforms existing baselines, achieving an improvement of up to 43.5\% in PSNR over naive fine-tuning and surpassing joint training on some metrics.
>
---
#### [new 046] SPHINX: A Synthetic Environment for Visual Perception and Reasoning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出SPHINX，一个用于视觉感知与推理的合成环境，旨在解决多模态认知能力评估难题。通过程序化生成包含25类任务的可视化谜题，支持精确评估与大规模数据构建。实验表明当前顶尖模型表现有限，而基于可验证奖励的强化学习显著提升性能，推动多模态推理发展。**

- **链接: [https://arxiv.org/pdf/2511.20814v1](https://arxiv.org/pdf/2511.20814v1)**

> **作者:** Md Tanvirul Alam; Saksham Aggarwal; Justin Yang Chae; Nidhi Rastogi
>
> **摘要:** We present Sphinx, a synthetic environment for visual perception and reasoning that targets core cognitive primitives. Sphinx procedurally generates puzzles using motifs, tiles, charts, icons, and geometric primitives, each paired with verifiable ground-truth solutions, enabling both precise evaluation and large-scale dataset construction. The benchmark covers 25 task types spanning symmetry detection, geometric transformations, spatial reasoning, chart interpretation, and sequence prediction. Evaluating recent large vision-language models (LVLMs) shows that even state-of-the-art GPT-5 attains only 51.1% accuracy, well below human performance. Finally, we demonstrate that reinforcement learning with verifiable rewards (RLVR) substantially improves model accuracy on these tasks and yields gains on external visual reasoning benchmarks, highlighting its promise for advancing multimodal reasoning.
>
---
#### [new 047] LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules
- **分类: cs.CV**

- **简介: 该论文针对肺结节精准诊断任务，解决现有模型在结节形态描述与医学知识融合上的不足。提出LungNoduleAgent多智能体系统，通过检测、报告生成与恶性度推理三模块协同，实现区域级语义对齐与多专家协作，显著提升诊断精度。**

- **链接: [https://arxiv.org/pdf/2511.21042v1](https://arxiv.org/pdf/2511.21042v1)**

> **作者:** Cheng Yang; Hui Jin; Xinlei Yu; Zhipeng Wang; Yaoqun Liu; Fenglei Fan; Dajiang Lei; Gangyong Jia; Changmiao Wang; Ruiquan Ge
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Diagnosing lung cancer typically involves physicians identifying lung nodules in Computed tomography (CT) scans and generating diagnostic reports based on their morphological features and medical expertise. Although advancements have been made in using multimodal large language models for analyzing lung CT scans, challenges remain in accurately describing nodule morphology and incorporating medical expertise. These limitations affect the reliability and effectiveness of these models in clinical settings. Collaborative multi-agent systems offer a promising strategy for achieving a balance between generality and precision in medical applications, yet their potential in pathology has not been thoroughly explored. To bridge these gaps, we introduce LungNoduleAgent, an innovative collaborative multi-agent system specifically designed for analyzing lung CT scans. LungNoduleAgent streamlines the diagnostic process into sequential components, improving precision in describing nodules and grading malignancy through three primary modules. The first module, the Nodule Spotter, coordinates clinical detection models to accurately identify nodules. The second module, the Radiologist, integrates localized image description techniques to produce comprehensive CT reports. Finally, the Doctor Agent System performs malignancy reasoning by using images and CT reports, supported by a pathology knowledge base and a multi-agent system framework. Extensive testing on two private datasets and the public LIDC-IDRI dataset indicates that LungNoduleAgent surpasses mainstream vision-language models, agent systems, and advanced expert models. These results highlight the importance of region-level semantic alignment and multi-agent collaboration in diagnosing nodules. LungNoduleAgent stands out as a promising foundational tool for supporting clinical analyses of lung nodules.
>
---
#### [new 048] Progress by Pieces: Test-Time Scaling for Autoregressive Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对自回归图像生成中的测试时缩放问题，提出GridAR框架。通过网格分区渐进生成与布局引导的提示重构，提升生成质量并降低计算成本，有效解决传统方法在轨迹纠错和全局蓝图缺失上的不足。**

- **链接: [https://arxiv.org/pdf/2511.21185v1](https://arxiv.org/pdf/2511.21185v1)**

> **作者:** Joonhyung Park; Hyeongwon Jang; Joowon Kim; Eunho Yang
>
> **备注:** Project page: https://grid-ar.github.io/
>
> **摘要:** Recent visual autoregressive (AR) models have shown promising capabilities in text-to-image generation, operating in a manner similar to large language models. While test-time computation scaling has brought remarkable success in enabling reasoning-enhanced outputs for challenging natural language tasks, its adaptation to visual AR models remains unexplored and poses unique challenges. Naively applying test-time scaling strategies such as Best-of-N can be suboptimal: they consume full-length computation on erroneous generation trajectories, while the raster-scan decoding scheme lacks a blueprint of the entire canvas, limiting scaling benefits as only a few prompt-aligned candidates are generated. To address these, we introduce GridAR, a test-time scaling framework designed to elicit the best possible results from visual AR models. GridAR employs a grid-partitioned progressive generation scheme in which multiple partial candidates for the same position are generated within a canvas, infeasible ones are pruned early, and viable ones are fixed as anchors to guide subsequent decoding. Coupled with this, we present a layout-specified prompt reformulation strategy that inspects partial views to infer a feasible layout for satisfying the prompt. The reformulated prompt then guides subsequent image generation to mitigate the blueprint deficiency. Together, GridAR achieves higher-quality results under limited test-time scaling: with N=4, it even outperforms Best-of-N (N=8) by 14.4% on T2I-CompBench++ while reducing cost by 25.6%. It also generalizes to autoregressive image editing, showing comparable edit quality and a 13.9% gain in semantic preservation on PIE-Bench over larger-N baselines.
>
---
#### [new 049] DeeAD: Dynamic Early Exit of Vision-Language Action for Efficient Autonomous Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文针对自动驾驶中视觉-语言-动作（VLA）模型推理延迟高的问题，提出无需训练的DeeAD框架。通过评估中间轨迹的物理可行性，动态提前退出计算，并结合多跳控制器跳过冗余层，实现28%层稀疏度与29%延迟降低，保持规划质量与安全。**

- **链接: [https://arxiv.org/pdf/2511.20720v1](https://arxiv.org/pdf/2511.20720v1)**

> **作者:** Haibo HU; Lianming Huang; Nan Guan; Chun Jason Xue
>
> **摘要:** Vision-Language Action (VLA) models unify perception, reasoning, and trajectory generation for autonomous driving, but suffer from significant inference latency due to deep transformer stacks. We present DeeAD, a training-free, action-guided early-exit framework that accelerates VLA planning by evaluating the physical feasibility of intermediate trajectories. Instead of relying on confidence scores, DeeAD terminates inference when predicted trajectories align with lightweight planning priors (e.g., Navigation or Low-precision Planning) within a tolerable deviation (<2m). To improve efficiency, we introduce a multi-hop controller that adaptively skips redundant layers based on the change rate of scores. DeeAD integrates into existing VLA models, such as ORION, without requiring retraining. Experiments on the Bench2Drive benchmark demonstrate up to 28% transformer-layer sparsity and 29% latency reduction, while preserving planning quality and safety.
>
---
#### [new 050] Structure-Aware Prototype Guided Trusted Multi-View Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多视图分类中的可信性问题，提出结构感知原型引导的可信多视图分类框架。通过引入原型简化视图内邻域关系学习，动态对齐跨视图结构，提升一致性与效率，解决现有方法计算成本高、视图间不一致的问题。**

- **链接: [https://arxiv.org/pdf/2511.21021v1](https://arxiv.org/pdf/2511.21021v1)**

> **作者:** Haojian Huang; Jiahao Shi; Zhe Liu; Harold Haodong Chen; Han Fang; Hao Sun; Zhongjiang He
>
> **备注:** 12 pages, 8 figures, 7 tables, Ongoing Work
>
> **摘要:** Trustworthy multi-view classification (TMVC) addresses the challenge of achieving reliable decision-making in complex scenarios where multi-source information is heterogeneous, inconsistent, or even conflicting. Existing TMVC approaches predominantly rely on globally dense neighbor relationships to model intra-view dependencies, leading to high computational costs and an inability to directly ensure consistency across inter-view relationships. Furthermore, these methods typically aggregate evidence from different views through manually assigned weights, lacking guarantees that the learned multi-view neighbor structures are consistent within the class space, thus undermining the trustworthiness of classification outcomes. To overcome these limitations, we propose a novel TMVC framework that introduces prototypes to represent the neighbor structures of each view. By simplifying the learning of intra-view neighbor relations and enabling dynamic alignment of intra- and inter-view structure, our approach facilitates more efficient and consistent discovery of cross-view consensus. Extensive experiments on multiple public multi-view datasets demonstrate that our method achieves competitive downstream performance and robustness compared to prevalent TMVC methods.
>
---
#### [new 051] BotaCLIP: Contrastive Learning for Botany-Aware Representation of Earth Observation Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对地球观测数据中缺乏领域知识的问题，提出BotaCLIP框架，通过对比学习将高分辨率航拍图像与植物标本数据对齐，实现轻量级、低成本的生态知识注入。解决了预训练模型在生物多样性建模中因数据稀缺导致的表征能力不足问题，显著提升了植物存在预测、蝴蝶分布和土壤营养级群丰度等任务的性能。**

- **链接: [https://arxiv.org/pdf/2511.21194v1](https://arxiv.org/pdf/2511.21194v1)**

> **作者:** Selene Cerna; Sara Si-Moussi; Wilfried Thuiller; Hadrien Hendrikx; Vincent Miele
>
> **摘要:** Foundation models have demonstrated a remarkable ability to learn rich, transferable representations across diverse modalities such as images, text, and audio. In modern machine learning pipelines, these representations often replace raw data as the primary input for downstream tasks. In this paper, we address the challenge of adapting a pre-trained foundation model to inject domain-specific knowledge, without retraining from scratch or incurring significant computational costs. To this end, we introduce BotaCLIP, a lightweight multimodal contrastive framework that adapts a pre-trained Earth Observation foundation model (DOFA) by aligning high-resolution aerial imagery with botanical relevés. Unlike generic embeddings, BotaCLIP internalizes ecological structure through contrastive learning with a regularization strategy that mitigates catastrophic forgetting. Once trained, the resulting embeddings serve as transferable representations for downstream predictors. Motivated by real-world applications in biodiversity modeling, we evaluated BotaCLIP representations in three ecological tasks: plant presence prediction, butterfly occurrence modeling, and soil trophic group abundance estimation. The results showed consistent improvements over those derived from DOFA and supervised baselines. More broadly, this work illustrates how domain-aware adaptation of foundation models can inject expert knowledge into data-scarce settings, enabling frugal representation learning.
>
---
#### [new 052] Training-Free Diffusion Priors for Text-to-Image Generation via Optimization-based Visual Inversion
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对文本到图像生成中依赖昂贵训练的扩散先验问题，提出无需训练和数据的优化视觉反演（OVI）方法。通过优化随机伪标记生成图像潜在表示，并引入两种新约束提升图像真实性。实验表明其性能可媲美先进先验，揭示现有评估基准的缺陷。**

- **链接: [https://arxiv.org/pdf/2511.20821v1](https://arxiv.org/pdf/2511.20821v1)**

> **作者:** Samuele Dell'Erba; Andrew D. Bagdanov
>
> **备注:** 11 pages, 7 figures, technical report (preprint)
>
> **摘要:** Diffusion models have established the state-of-the-art in text-to-image generation, but their performance often relies on a diffusion prior network to translate text embeddings into the visual manifold for easier decoding. These priors are computationally expensive and require extensive training on massive datasets. In this work, we challenge the necessity of a trained prior at all by employing Optimization-based Visual Inversion (OVI), a training-free and data-free alternative, to replace the need for a prior. OVI initializes a latent visual representation from random pseudo-tokens and iteratively optimizes it to maximize the cosine similarity with input textual prompt embedding. We further propose two novel constraints, a Mahalanobis-based and a Nearest-Neighbor loss, to regularize the OVI optimization process toward the distribution of realistic images. Our experiments, conducted on Kandinsky 2.2, show that OVI can serve as an alternative to traditional priors. More importantly, our analysis reveals a critical flaw in current evaluation benchmarks like T2I-CompBench++, where simply using the text embedding as a prior achieves surprisingly high scores, despite lower perceptual quality. Our constrained OVI methods improve visual fidelity over this baseline, with the Nearest-Neighbor approach proving particularly effective, achieving quantitative scores comparable to or higher than the state-of-the-art data-efficient prior, indicating that the idea merits further investigation. The code will be publicly available upon acceptance.
>
---
#### [new 053] Do Reasoning Vision-Language Models Inversely Scale in Test-Time Compute? A Distractor-centric Empirical Analysis
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究视觉语言模型在测试时计算资源增加下的表现，聚焦于无关信息（干扰项）的影响。通过构建Idiv数据集，系统分析视觉干扰项对推理长度与准确率的作用，发现其导致逆向缩放但不延长推理，且提出有效提示策略缓解偏差。**

- **链接: [https://arxiv.org/pdf/2511.21397v1](https://arxiv.org/pdf/2511.21397v1)**

> **作者:** Jiyun Bae; Hyunjong Ok; Sangwoo Mo; Jaeho Lee
>
> **备注:** preprint
>
> **摘要:** How does irrelevant information (i.e., distractors) affect test-time scaling in vision-language models (VLMs)? Prior studies on language models have reported an inverse scaling effect, where textual distractors lead to longer but less effective reasoning. To investigate whether similar phenomena occur in multimodal settings, we introduce Idis (Images with distractors), a visual question-answering dataset that systematically varies distractors along semantic, numerical, and spatial dimensions. Our analyses reveal that visual distractors differ fundamentally from textual ones: although inverse scaling persists, adding visual distractors reduces accuracy without increasing reasoning length. We further show that tracking attribute counts within reasoning traces provides key insights into how distractors, reasoning length, and accuracy interact. Finally, we demonstrate that these trends extend to established visual bias benchmarks such as Waterbirds, and we propose a simple prompting strategy to mitigate bias-driven predictions in reasoning models.
>
---
#### [new 054] Frequency-Aware Token Reduction for Efficient Vision Transformer
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉Transformer计算复杂度高的问题，提出一种频率感知的令牌压缩方法。通过区分高频与低频令牌，保留高频信息并聚合低频信息为直流令牌，有效降低计算量，缓解秩坍缩与过平滑问题，提升效率与性能。属于计算机视觉中的模型优化任务。**

- **链接: [https://arxiv.org/pdf/2511.21477v1](https://arxiv.org/pdf/2511.21477v1)**

> **作者:** Dong-Jae Lee; Jiwan Hur; Jaehyun Choi; Jaemyung Yu; Junmo Kim
>
> **备注:** Neurips 2025
>
> **摘要:** Vision Transformers have demonstrated exceptional performance across various computer vision tasks, yet their quadratic computational complexity concerning token length remains a significant challenge. To address this, token reduction methods have been widely explored. However, existing approaches often overlook the frequency characteristics of self-attention, such as rank collapsing and over-smoothing phenomenon. In this paper, we propose a frequency-aware token reduction strategy that improves computational efficiency while preserving performance by mitigating rank collapsing. Our method partitions tokens into high-frequency tokens and low-frequency tokens. high-frequency tokens are selectively preserved, while low-frequency tokens are aggregated into a compact direct current token to retain essential low-frequency components. Through extensive experiments and analysis, we demonstrate that our approach significantly improves accuracy while reducing computational overhead and mitigating rank collapsing and over smoothing. Furthermore, we analyze the previous methods, shedding light on their implicit frequency characteristics and limitations.
>
---
#### [new 055] CtrlVDiff: Controllable Video Generation via Unified Multimodal Video Diffusion
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出CtrlVDiff，一个统一的多模态视频扩散模型，旨在解决视频生成中控制精度与时空一致性难题。通过融合深度、法向、语义分割、材质等多模态线索，实现可控视频生成与精细编辑，构建了对齐的MMVideo数据集，显著提升生成质量与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.21129v1](https://arxiv.org/pdf/2511.21129v1)**

> **作者:** Dianbing Xi; Jiepeng Wang; Yuanzhi Liang; Xi Qiu; Jialun Liu; Hao Pan; Yuchi Huo; Rui Wang; Haibin Huang; Chi Zhang; Xuelong Li
>
> **备注:** 27 pages, 18 figures, 9 tables. Project page: https://tele-ai.github.io/CtrlVDiff/
>
> **摘要:** We tackle the dual challenges of video understanding and controllable video generation within a unified diffusion framework. Our key insights are two-fold: geometry-only cues (e.g., depth, edges) are insufficient: they specify layout but under-constrain appearance, materials, and illumination, limiting physically meaningful edits such as relighting or material swaps and often causing temporal drift. Enriching the model with additional graphics-based modalities (intrinsics and semantics) provides complementary constraints that both disambiguate understanding and enable precise, predictable control during generation. However, building a single model that uses many heterogeneous cues introduces two core difficulties. Architecturally, the model must accept any subset of modalities, remain robust to missing inputs, and inject control signals without sacrificing temporal consistency. Data-wise, training demands large-scale, temporally aligned supervision that ties real videos to per-pixel multimodal annotations. We then propose CtrlVDiff, a unified diffusion model trained with a Hybrid Modality Control Strategy (HMCS) that routes and fuses features from depth, normals, segmentation, edges, and graphics-based intrinsics (albedo, roughness, metallic), and re-renders videos from any chosen subset with strong temporal coherence. To enable this, we build MMVideo, a hybrid real-and-synthetic dataset aligned across modalities and captions. Across understanding and generation benchmarks, CtrlVDiff delivers superior controllability and fidelity, enabling layer-wise edits (relighting, material adjustment, object insertion) and surpassing state-of-the-art baselines while remaining robust when some modalities are unavailable.
>
---
#### [new 056] Text-Guided Semantic Image Encoder
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型中图像编码器忽略文本查询的问题，提出文本引导的语义图像编码器（TIE），使图像表示基于输入文本生成。TIE在多个图文任务上显著提升性能，减少一半图像令牌数，提高效率与可解释性，有效实现查询相关视觉特征捕捉。**

- **链接: [https://arxiv.org/pdf/2511.20770v1](https://arxiv.org/pdf/2511.20770v1)**

> **作者:** Raghuveer Thirukovalluru; Xiaochuang Han; Bhuwan Dhingra; Emily Dinan; Maha Elbayad
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Image encoders, a fundamental component of vision-language models (VLMs), are typically pretrained independently before being aligned with a language model. This standard paradigm results in encoders that process images agnostically, without regard to the specific downstream task or text query. To address this limitation, we propose the Text-Guided Semantic Image Encoder (TIE), which generates image representations conditioned on the input text query. VLMs equipped with TIE outperform their conventional counterparts by +1.5 and +1.3 points on average across nine image-to-text benchmarks at the 1B and 3B scales, respectively, with gains reaching up to 6 points on tasks such as DocVQA and InfoVQA. Moreover, TIE-based VLMs attain superior performance while utilizing only half as many image tiles (tokens), resulting in notably improved inference efficiency. TIE also generalizes well with generic queries, indicating that text-conditioned training effectively optimizes the encoder to capture key visual features. Qualitative analysis confirms that TIE consistently attends to query-relevant regions, enhancing both interpretability and query-specific grounding.
>
---
#### [new 057] Unlocking Zero-shot Potential of Semi-dense Image Matching via Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文针对图像匹配任务中的数据依赖问题，提出MatchGS框架，通过几何修正的3DGS生成高精度对应标签，并融合3D知识指导2D匹配器学习视角不变表示，实现零样本图像匹配性能显著提升。**

- **链接: [https://arxiv.org/pdf/2511.21265v1](https://arxiv.org/pdf/2511.21265v1)**

> **作者:** Juncheng Chen; Chao Xu; Yanjun Cao
>
> **摘要:** Learning-based image matching critically depends on large-scale, diverse, and geometrically accurate training data. 3D Gaussian Splatting (3DGS) enables photorealistic novel-view synthesis and thus is attractive for data generation. However, its geometric inaccuracies and biased depth rendering currently prevent robust correspondence labeling. To address this, we introduce MatchGS, the first framework designed to systematically correct and leverage 3DGS for robust, zero-shot image matching. Our approach is twofold: (1) a geometrically-faithful data generation pipeline that refines 3DGS geometry to produce highly precise correspondence labels, enabling the synthesis of a vast and diverse range of viewpoints without compromising rendering fidelity; and (2) a 2D-3D representation alignment strategy that infuses 3DGS' explicit 3D knowledge into the 2D matcher, guiding 2D semi-dense matchers to learn viewpoint-invariant 3D representations. Our generated ground-truth correspondences reduce the epipolar error by up to 40 times compared to existing datasets, enable supervision under extreme viewpoint changes, and provide self-supervisory signals through Gaussian attributes. Consequently, state-of-the-art matchers trained solely on our data achieve significant zero-shot performance gains on public benchmarks, with improvements of up to 17.7%. Our work demonstrates that with proper geometric refinement, 3DGS can serve as a scalable, high-fidelity, and structurally-rich data source, paving the way for a new generation of robust zero-shot image matchers.
>
---
#### [new 058] MoGAN: Improving Motion Quality in Video Diffusion via Few-Step Motion Adversarial Post-Training
- **分类: cs.CV**

- **简介: 该论文针对视频生成中运动不连贯、失真等问题，提出MoGAN框架。基于三步蒸馏视频扩散模型，引入基于DiT的光流判别器与分布匹配正则化，提升运动真实性。实验表明，MoGAN显著改善运动质量，同时保持视觉保真度与生成效率。**

- **链接: [https://arxiv.org/pdf/2511.21592v1](https://arxiv.org/pdf/2511.21592v1)**

> **作者:** Haotian Xue; Qi Chen; Zhonghao Wang; Xun Huang; Eli Shechtman; Jinrong Xie; Yongxin Chen
>
> **摘要:** Video diffusion models achieve strong frame-level fidelity but still struggle with motion coherence, dynamics and realism, often producing jitter, ghosting, or implausible dynamics. A key limitation is that the standard denoising MSE objective provides no direct supervision on temporal consistency, allowing models to achieve low loss while still generating poor motion. We propose MoGAN, a motion-centric post-training framework that improves motion realism without reward models or human preference data. Built atop a 3-step distilled video diffusion model, we train a DiT-based optical-flow discriminator to differentiate real from generated motion, combined with a distribution-matching regularizer to preserve visual fidelity. With experiments on Wan2.1-T2V-1.3B, MoGAN substantially improves motion quality across benchmarks. On VBench, MoGAN boosts motion score by +7.3% over the 50-step teacher and +13.3% over the 3-step DMD model. On VideoJAM-Bench, MoGAN improves motion score by +7.4% over the teacher and +8.8% over DMD, while maintaining comparable or even better aesthetic and image-quality scores. A human study further confirms that MoGAN is preferred for motion quality (52% vs. 38% for the teacher; 56% vs. 29% for DMD). Overall, MoGAN delivers significantly more realistic motion without sacrificing visual fidelity or efficiency, offering a practical path toward fast, high-quality video generation. Project webpage is: https://xavihart.github.io/mogan.
>
---
#### [new 059] CANVAS: A Benchmark for Vision-Language Models on Tool-Based User Interface Design
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出CANVAS基准，用于评估视觉语言模型（VLMs）在工具调用下的用户界面（UI）设计能力。针对现有缺乏工具驱动设计评测基准的问题，构建了598个任务，涵盖设计复制与修改，通过真实软件操作流程测试模型表现，揭示其策略性工具调用能力及常见错误，推动VLM在设计协作中的应用。**

- **链接: [https://arxiv.org/pdf/2511.20737v1](https://arxiv.org/pdf/2511.20737v1)**

> **作者:** Daeheon Jeong; Seoyeon Byun; Kihoon Son; Dae Hyun Kim; Juho Kim
>
> **摘要:** User interface (UI) design is an iterative process in which designers progressively refine their work with design software such as Figma or Sketch. Recent advances in vision language models (VLMs) with tool invocation suggest these models can operate design software to edit a UI design through iteration. Understanding and enhancing this capacity is important, as it highlights VLMs' potential to collaborate with designers within conventional software. However, as no existing benchmark evaluates tool-based design performance, the capacity remains unknown. To address this, we introduce CANVAS, a benchmark for VLMs on tool-based user interface design. Our benchmark contains 598 tool-based design tasks paired with ground-truth references sampled from 3.3K mobile UI designs across 30 function-based categories (e.g., onboarding, messaging). In each task, a VLM updates the design step-by-step through context-based tool invocations (e.g., create a rectangle as a button background), linked to design software. Specifically, CANVAS incorporates two task types: (i) design replication evaluates the ability to reproduce a whole UI screen; (ii) design modification evaluates the ability to modify a specific part of an existing screen. Results suggest that leading models exhibit more strategic tool invocations, improving design quality. Furthermore, we identify common error patterns models exhibit, guiding future work in enhancing tool-based design capabilities.
>
---
#### [new 060] Revisiting KRISP: A Lightweight Reproduction and Analysis of Knowledge-Enhanced Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对知识增强视觉问答（VQA）模型计算成本高、难以部署的问题，复现并轻量化了KRISP模型。通过减少参数量，在保持75%性能的同时，提升了可扩展性与鲁棒性，有效防止AI幻觉，实现边缘设备上的离线视觉推理。**

- **链接: [https://arxiv.org/pdf/2511.20795v1](https://arxiv.org/pdf/2511.20795v1)**

> **作者:** Souradeep Dutta; Keshav Bulia; Neena S Nair
>
> **备注:** 7 pages , 4 figures
>
> **摘要:** Facebook AI Research introduced KRISP [4], which integrates structured external knowledge into pipelines for vision-language reasoning. Despite its effectiveness, the original model has been developed for industrial-scale training, is computationally demanding, and is tightly connected to a large backbone. In this work, we reexamine KRISP from a different angle and offer a lightweight reproduction with significantly fewer parameters. Even though our replicated model performs about 75 % of the original, the replication process uncovers a number of design flaws, real-world pitfalls, and implicit problems that were not fully covered in the original paper. We offer insights into the scalability and efficacy of knowledge-enhanced VQA architectures under resource constraints through systematic ablation studies, which include a proof-of-concept on synthetic VQA data and evaluation on the DAQUAR dataset. Our model, configured with a low parameter setup and constrained by the external Knowledge graph domain, prevents AI hallucinations and generates outputs solely within that domain. Minimal parameters allow us to function on edge devices like smartphones and AR-VR, further improving offline visual reasoning.
>
---
#### [new 061] DinoLizer: Learning from the Best for Generative Inpainting Localization
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DinoLizer，一种基于DINOv2的生成式修复图像篡改区域定位方法。针对现有技术在检测生成式修复篡改时鲁棒性不足的问题，利用预训练模型提取语义特征，通过线性分类头与滑动窗口策略实现高精度局部定位，显著提升对多种后处理操作的抗干扰能力。**

- **链接: [https://arxiv.org/pdf/2511.20722v1](https://arxiv.org/pdf/2511.20722v1)**

> **作者:** Minh Thong Doi; Jan Butora; Vincent Itier; Jérémie Boulanger; Patrick Bas
>
> **摘要:** We introduce DinoLizer, a DINOv2-based model for localizing manipulated regions in generative inpainting. Our method builds on a DINOv2 model pretrained to detect synthetic images on the B-Free dataset. We add a linear classification head on top of the Vision Transformer's patch embeddings to predict manipulations at a $14\times 14$ patch resolution. The head is trained to focus on semantically altered regions, treating non-semantic edits as part of the original content. Because the ViT accepts only fixed-size inputs, we use a sliding-window strategy to aggregate predictions over larger images; the resulting heatmaps are post-processed to refine the estimated binary manipulation masks. Empirical results show that DinoLizer surpasses state-of-the-art local manipulation detectors on a range of inpainting datasets derived from different generative models. It remains robust to common post-processing operations such as resizing, noise addition, and JPEG (double) compression. On average, DinoLizer achieves a 12\% higher Intersection-over-Union (IoU) than the next best model, with even greater gains after post-processing. Our experiments with off-the-shelf DINOv2 demonstrate the strong representational power of Vision Transformers for this task. Finally, extensive ablation studies comparing DINOv2 and its successor, DINOv3, in deepfake localization confirm DinoLizer's superiority. The code will be publicly available upon acceptance of the paper.
>
---
#### [new 062] FIELDS: Face reconstruction with accurate Inference of Expression using Learning with Direct Supervision
- **分类: cs.CV**

- **简介: 该论文提出FIELDS方法，解决3D人脸重建中因依赖2D监督而丢失细微表情的问题。通过引入直接的3D表情参数监督和情绪识别分支，结合真实4D扫描数据，实现更精准的表情还原，显著提升自然表情建模与野外表情识别性能。**

- **链接: [https://arxiv.org/pdf/2511.21245v1](https://arxiv.org/pdf/2511.21245v1)**

> **作者:** Chen Ling; Henglin Shi; Hedvig Kjellström
>
> **摘要:** Facial expressions convey the bulk of emotional information in human communication, yet existing 3D face reconstruction methods often miss subtle affective details due to reliance on 2D supervision and lack of 3D ground truth. We propose FIELDS (Face reconstruction with accurate Inference of Expression using Learning with Direct Supervision) to address these limitations by extending self-supervised 2D image consistency cues with direct 3D expression parameter supervision and an auxiliary emotion recognition branch. Our encoder is guided by authentic expression parameters from spontaneous 4D facial scans, while an intensity-aware emotion loss encourages the 3D expression parameters to capture genuine emotion content without exaggeration. This dual-supervision strategy bridges the 2D/3D domain gap and mitigates expression-intensity bias, yielding high-fidelity 3D reconstructions that preserve subtle emotional cues. From a single image, FIELDS produces emotion-rich face models with highly realistic expressions, significantly improving in-the-wild facial expression recognition performance without sacrificing naturalness.
>
---
#### [new 063] Co-Training Vision Language Models for Remote Sensing Multi-task Learning
- **分类: cs.CV**

- **简介: 该论文面向遥感多任务学习，旨在构建统一的视觉语言模型。针对遥感数据多样性和计算负担重的问题，提出RSCoVLM，通过动态分辨率策略与Zoom-in Chain机制提升模型性能，并设计新评估协议，实现跨模型公平比较。**

- **链接: [https://arxiv.org/pdf/2511.21272v1](https://arxiv.org/pdf/2511.21272v1)**

> **作者:** Qingyun Li; Shuran Ma; Junwei Luo; Yi Yu; Yue Zhou; Fengxiang Wang; Xudong Lu; Xiaoxing Wang; Xin He; Yushi Chen; Xue Yang; Junchi Yan
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** With Transformers achieving outstanding performance on individual remote sensing (RS) tasks, we are now approaching the realization of a unified model that excels across multiple tasks through multi-task learning (MTL). Compared to single-task approaches, MTL methods offer improved generalization, enhanced scalability, and greater practical applicability. Recently, vision language models (VLMs) have achieved promising results in RS image understanding, grounding, and ultra-high-resolution (UHR) image reasoning, respectively. Moreover, the unified text-based interface demonstrates significant potential for MTL. Hence, in this work, we present RSCoVLM, a simple yet flexible VLM baseline for RS MTL. Firstly, we create the data curation engine, including data acquisition, offline processing and integrating, as well as online loading and weighting. This data engine effectively addresses complex RS data enviroment and generates flexible vision-language conversations. Furthermore, we propose a unified dynamic-resolution strategy to address the diverse image scales inherent in RS imagery. For UHR images, we introduce the Zoom-in Chain mechanism together with its corresponding dataset, LRS-VQA-Zoom. The strategies are flexible and effectively mitigate the computational burdens. Additionally, we significantly enhance the model's object detection capability and propose a novel evaluation protocol that ensures fair comparison between VLMs and conventional detection models. Extensive experiments demonstrate that RSCoVLM achieves state-of-the-art performance across diverse tasks, outperforming existing RS VLMs and even rivaling specialized expert models. All the training and evaluating tools, model weights, and datasets have been fully open-sourced to support reproducibility. We expect that this baseline will promote further progress toward general-purpose RS models.
>
---
#### [new 064] 3-Tracer: A Tri-level Temporal-Aware Framework for Audio Forgery Detection and Localization
- **分类: cs.CV**

- **简介: 该论文针对部分音频伪造检测任务，解决现有方法难以捕捉多层级时序异常的问题。提出T3-Tracer框架，通过帧、段、音频三层联合分析，结合特征聚合与多尺度差异感知模块，有效检测伪造帧及边界，实现精准定位与检测。**

- **链接: [https://arxiv.org/pdf/2511.21237v1](https://arxiv.org/pdf/2511.21237v1)**

> **作者:** Shuhan Xia; Xuannan Liu; Xing Cui; Peipei Li
>
> **摘要:** Recently, partial audio forgery has emerged as a new form of audio manipulation. Attackers selectively modify partial but semantically critical frames while preserving the overall perceptual authenticity, making such forgeries particularly difficult to detect. Existing methods focus on independently detecting whether a single frame is forged, lacking the hierarchical structure to capture both transient and sustained anomalies across different temporal levels. To address these limitations, We identify three key levels relevant to partial audio forgery detection and present T3-Tracer, the first framework that jointly analyzes audio at the frame, segment, and audio levels to comprehensively detect forgery traces. T3-Tracer consists of two complementary core modules: the Frame-Audio Feature Aggregation Module (FA-FAM) and the Segment-level Multi-Scale Discrepancy-Aware Module (SMDAM). FA-FAM is designed to detect the authenticity of each audio frame. It combines both frame-level and audio-level temporal information to detect intra-frame forgery cues and global semantic inconsistencies. To further refine and correct frame detection, we introduce SMDAM to detect forgery boundaries at the segment level. It adopts a dual-branch architecture that jointly models frame features and inter-frame differences across multi-scale temporal windows, effectively identifying abrupt anomalies that appeared on the forged boundaries. Extensive experiments conducted on three challenging datasets demonstrate that our approach achieves state-of-the-art performance.
>
---
#### [new 065] Endo-G$^{2}$T: Geometry-Guided & Temporally Aware Time-Embedded 4DGS For Endoscopic Scenes
- **分类: cs.CV**

- **简介: 该论文针对内窥镜视频的4D高斯溅射重建任务，解决视图依赖效应导致的几何漂移问题。提出Endo-G²T框架，通过几何引导先验蒸馏、时间嵌入高斯场与关键帧约束流式优化，实现早期几何锚定、时序一致性与高效动态建模，在单目重建中达到最优性能。**

- **链接: [https://arxiv.org/pdf/2511.21367v1](https://arxiv.org/pdf/2511.21367v1)**

> **作者:** Yangle Liu; Fengze Li; Kan Liu; Jieming Ma
>
> **摘要:** Endoscopic (endo) video exhibits strong view-dependent effects such as specularities, wet reflections, and occlusions. Pure photometric supervision misaligns with geometry and triggers early geometric drift, where erroneous shapes are reinforced during densification and become hard to correct. We ask how to anchor geometry early for 4D Gaussian splatting (4DGS) while maintaining temporal consistency and efficiency in dynamic endoscopic scenes. Thus, we present Endo-G$^{2}$T, a geometry-guided and temporally aware training scheme for time-embedded 4DGS. First, geo-guided prior distillation converts confidence-gated monocular depth into supervision with scale-invariant depth and depth-gradient losses, using a warm-up-to-cap schedule to inject priors softly and avoid early overfitting. Second, a time-embedded Gaussian field represents dynamics in XYZT with a rotor-like rotation parameterization, yielding temporally coherent geometry with lightweight regularization that favors smooth motion and crisp opacity boundaries. Third, keyframe-constrained streaming improves efficiency and long-horizon stability through keyframe-focused optimization under a max-points budget, while non-keyframes advance with lightweight updates. Across EndoNeRF and StereoMIS-P1 datasets, Endo-G$^{2}$T achieves state-of-the-art results among monocular reconstruction baselines.
>
---
#### [new 066] From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对工业场景中大量未标注视频数据难以用于视觉-语言-动作（VLA）模型预训练的问题，提出一种端到端的无监督框架。通过轻量级运动编码器与基于“潜在动作能量”的分割机制，自动提取语义一致的动作基元，生成可用于VLA预训练的结构化数据，实现从原始视频到可训练数据的高效转化。**

- **链接: [https://arxiv.org/pdf/2511.21428v1](https://arxiv.org/pdf/2511.21428v1)**

> **作者:** Jiajie Zhang; Sören Schwertfeger; Alexander Kleiner
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** We present a novel unsupervised framework to unlock vast unlabeled human demonstration data from continuous industrial video streams for Vision-Language-Action (VLA) model pre-training. Our method first trains a lightweight motion tokenizer to encode motion dynamics, then employs an unsupervised action segmenter leveraging a novel "Latent Action Energy" metric to discover and segment semantically coherent action primitives. The pipeline outputs both segmented video clips and their corresponding latent action sequences, providing structured data directly suitable for VLA pre-training. Evaluations on public benchmarks and a proprietary electric motor assembly dataset demonstrate effective segmentation of key tasks performed by humans at workstations. Further clustering and quantitative assessment via a Vision-Language Model confirm the semantic coherence of the discovered action primitives. To our knowledge, this is the first fully automated end-to-end system for extracting and organizing VLA pre-training data from unstructured industrial videos, offering a scalable solution for embodied AI integration in manufacturing.
>
---
#### [new 067] EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens
- **分类: cs.CV**

- **简介: 该论文针对高效多模态大模型因压缩视觉令牌导致视觉信息损失的问题，提出EM-KD框架。通过曼哈顿距离与匈牙利匹配对齐异构视觉令牌，设计视觉-语言亲和性与视觉语义蒸馏策略，提升学生模型的细粒度理解能力。在多个基准上验证了其优越的准确率与效率。**

- **链接: [https://arxiv.org/pdf/2511.21106v1](https://arxiv.org/pdf/2511.21106v1)**

> **作者:** Ze Feng; Sen Yang; Boqiang Duan; Wankou Yang; Jingdong Wang
>
> **备注:** accepted by AAAI 2026
>
> **摘要:** Efficient Multimodal Large Language Models (MLLMs) compress vision tokens to reduce resource consumption, but the loss of visual information can degrade comprehension capabilities. Although some priors introduce Knowledge Distillation to enhance student models, they overlook the fundamental differences in fine-grained vision comprehension caused by unbalanced vision tokens between the efficient student and vanilla teacher. In this paper, we propose EM-KD, a novel paradigm that enhances the Efficient MLLMs with Knowledge Distillation. To overcome the challenge of unbalanced vision tokens, we first calculate the Manhattan distance between the vision logits of teacher and student, and then align them in the spatial dimension with the Hungarian matching algorithm. After alignment, EM-KD introduces two distillation strategies: 1) Vision-Language Affinity Distillation (VLAD) and 2) Vision Semantic Distillation (VSD). Specifically, VLAD calculates the affinity matrix between text tokens and aligned vision tokens, and minimizes the smooth L1 distance of the student and the teacher affinity matrices. Considering the semantic richness of vision logits in the final layer, VSD employs the reverse KL divergence to measure the discrete probability distributions of the aligned vision logits over the vocabulary space. Comprehensive evaluation on diverse benchmarks demonstrates that EM-KD trained model outperforms prior Efficient MLLMs on both accuracy and efficiency with a large margin, validating its effectiveness. Compared with previous distillation methods, which are equipped with our proposed vision token matching strategy for fair comparison, EM-KD also achieves better performance.
>
---
#### [new 068] CameraMaster: Unified Camera Semantic-Parameter Control for Photography Retouching
- **分类: cs.CV**

- **简介: 该论文提出CameraMaster，解决文本引导图像编辑中相机参数控制不精确、多参数组合困难的问题。通过解耦摄影师意图与相机参数，联合建模指令与参数嵌入，实现统一、精准的语义-参数对齐，支持线性响应与多参数协同调整，显著提升图像修图的物理一致性与可控性。**

- **链接: [https://arxiv.org/pdf/2511.21024v1](https://arxiv.org/pdf/2511.21024v1)**

> **作者:** Qirui Yang; Yang Yang; Ying Zeng; Xiaobin Hu; Bo Li; Huanjing Yue; Jingyu Yang; Peng-Tao Jiang
>
> **摘要:** Text-guided diffusion models have greatly advanced image editing and generation. However, achieving physically consistent image retouching with precise parameter control (e.g., exposure, white balance, zoom) remains challenging. Existing methods either rely solely on ambiguous and entangled text prompts, which hinders precise camera control, or train separate heads/weights for parameter adjustment, which compromises scalability, multi-parameter composition, and sensitivity to subtle variations. To address these limitations, we propose CameraMaster, a unified camera-aware framework for image retouching. The key idea is to explicitly decouple the camera directive and then coherently integrate two critical information streams: a directive representation that captures the photographer's intent, and a parameter embedding that encodes precise camera settings. CameraMaster first uses the camera parameter embedding to modulate both the camera directive and the content semantics. The modulated directive is then injected into the content features via cross-attention, yielding a strongly camera-sensitive semantic context. In addition, the directive and camera embeddings are injected as conditioning and gating signals into the time embedding, enabling unified, layer-wise modulation throughout the denoising process and enforcing tight semantic-parameter alignment. To train and evaluate CameraMaster, we construct a large-scale dataset of 78K image-prompt pairs annotated with camera parameters. Extensive experiments show that CameraMaster produces monotonic and near-linear responses to parameter variations, supports seamless multi-parameter composition, and significantly outperforms existing methods.
>
---
#### [new 069] From Diffusion to One-Step Generation: A Comparative Study of Flow-Based Models with Application to Image Inpainting
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究生成模型在图像修复中的应用。针对扩散模型采样慢的问题，比较了DDPM、CFM与MeanFlow三种方法，提出基于统一架构的高效生成方案。实验表明，MeanFlow实现单步生成，推理速度提升50倍；扩展至图像修复任务，显著提升重建质量。**

- **链接: [https://arxiv.org/pdf/2511.21215v1](https://arxiv.org/pdf/2511.21215v1)**

> **作者:** Umang Agarwal; Rudraksh Sangore; Sumit Laddha
>
> **摘要:** We present a comprehensive comparative study of three generative modeling paradigms: Denoising Diffusion Probabilistic Models (DDPM), Conditional Flow Matching (CFM), and MeanFlow. While DDPM and CFM require iterative sampling, MeanFlow enables direct one-step generation by modeling the average velocity over time intervals. We implement all three methods using a unified TinyUNet architecture (<1.5M parameters) on CIFAR-10, demonstrating that CFM achieves an FID of 24.15 with 50 steps, significantly outperforming DDPM (FID 402.98). MeanFlow achieves FID 29.15 with single-step sampling -- a 50X reduction in inference time. We further extend CFM to image inpainting, implementing mask-guided sampling with four mask types (center, random bbox, irregular, half). Our fine-tuned inpainting model achieves substantial improvements: PSNR increases from 4.95 to 8.57 dB on center masks (+73%), and SSIM improves from 0.289 to 0.418 (+45%), demonstrating the effectiveness of inpainting-aware training.
>
---
#### [new 070] Layer-Aware Video Composition via Split-then-Merge
- **分类: cs.CV**

- **简介: 该论文提出Split-then-Merge框架，解决生成视频合成中控制力弱与数据稀缺问题。通过无监督分割视频为前景与背景层，自合成训练学习动态交互，结合多层融合与身份保持损失，提升合成真实性。实验证明其优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.20809v1](https://arxiv.org/pdf/2511.20809v1)**

> **作者:** Ozgur Kara; Yujia Chen; Ming-Hsuan Yang; James M. Rehg; Wen-Sheng Chu; Du Tran
>
> **备注:** Project Webpage: https://split-then-merge.github.io
>
> **摘要:** We present Split-then-Merge (StM), a novel framework designed to enhance control in generative video composition and address its data scarcity problem. Unlike conventional methods relying on annotated datasets or handcrafted rules, StM splits a large corpus of unlabeled videos into dynamic foreground and background layers, then self-composes them to learn how dynamic subjects interact with diverse scenes. This process enables the model to learn the complex compositional dynamics required for realistic video generation. StM introduces a novel transformation-aware training pipeline that utilizes a multi-layer fusion and augmentation to achieve affordance-aware composition, alongside an identity-preservation loss that maintains foreground fidelity during blending. Experiments show StM outperforms SoTA methods in both quantitative benchmarks and in humans/VLLM-based qualitative evaluations. More details are available at our project page: https://split-then-merge.github.io
>
---
#### [new 071] MODEST: Multi-Optics Depth-of-Field Stereo Dataset
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文提出MODEST数据集，解决真实光学条件下深度估计缺乏高质量实拍数据的问题。针对立体视觉与深度估计任务，构建了高分辨率、多光学参数（焦距、光圈）的实拍立体图像数据集，支持真实场景下的深度估计、3D重建等研究，推动模型在真实世界中的泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.20853v1](https://arxiv.org/pdf/2511.20853v1)**

> **作者:** Nisarg K. Trivedi; Vinayak A. Belludi; Li-Yun Wang; Pardis Taghavi; Dante Lok
>
> **摘要:** Reliable depth estimation under real optical conditions remains a core challenge for camera vision in systems such as autonomous robotics and augmented reality. Despite recent progress in depth estimation and depth-of-field rendering, research remains constrained by the lack of large-scale, high-fidelity, real stereo DSLR datasets, limiting real-world generalization and evaluation of models trained on synthetic data as shown extensively in literature. We present the first high-resolution (5472$\times$3648px) stereo DSLR dataset with 18000 images, systematically varying focal length and aperture across complex real scenes and capturing the optical realism and complexity of professional camera systems. For 9 scenes with varying scene complexity, lighting and background, images are captured with two identical camera assemblies at 10 focal lengths (28-70mm) and 5 apertures (f/2.8-f/22), spanning 50 optical configurations in 2000 images per scene. This full-range optics coverage enables controlled analysis of geometric and optical effects for monocular and stereo depth estimation, shallow depth-of-field rendering, deblurring, 3D scene reconstruction and novel view synthesis. Each focal configuration has a dedicated calibration image set, supporting evaluation of classical and learning based methods for intrinsic and extrinsic calibration. The dataset features challenging visual elements such as multi-scale optical illusions, reflective surfaces, mirrors, transparent glass walls, fine-grained details, and natural / artificial ambient light variations. This work attempts to bridge the realism gap between synthetic training data and real camera optics, and demonstrates challenges with the current state-of-the-art monocular, stereo depth and depth-of-field methods. We release the dataset, calibration files, and evaluation code to support reproducible research on real-world optical generalization.
>
---
#### [new 072] SurgMLLMBench: A Multimodal Large Language Model Benchmark Dataset for Surgical Scene Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对手术场景理解中多模态大模型评估缺乏统一基准的问题，提出SurgMLLMBench基准数据集。它整合了腹腔镜、机器人辅助和显微手术的像素级分割与结构化VQA标注，支持跨域评估与交互式推理。通过新构建的MAVIS数据集，实现模型在多任务、多领域的一致性能与泛化能力，推动可复现的手术AI研究。**

- **链接: [https://arxiv.org/pdf/2511.21339v1](https://arxiv.org/pdf/2511.21339v1)**

> **作者:** Tae-Min Choi; Tae Kyeong Jeong; Garam Kim; Jaemin Lee; Yeongyoon Koh; In Cheul Choi; Jae-Ho Chung; Jong Woong Park; Juyoun Park
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Recent advances in multimodal large language models (LLMs) have highlighted their potential for medical and surgical applications. However, existing surgical datasets predominantly adopt a Visual Question Answering (VQA) format with heterogeneous taxonomies and lack support for pixel-level segmentation, limiting consistent evaluation and applicability. We present SurgMLLMBench, a unified multimodal benchmark explicitly designed for developing and evaluating interactive multimodal LLMs for surgical scene understanding, including the newly collected Micro-surgical Artificial Vascular anastomosIS (MAVIS) dataset. It integrates pixel-level instrument segmentation masks and structured VQA annotations across laparoscopic, robot-assisted, and micro-surgical domains under a unified taxonomy, enabling comprehensive evaluation beyond traditional VQA tasks and richer visual-conversational interactions. Extensive baseline experiments show that a single model trained on SurgMLLMBench achieves consistent performance across domains and generalizes effectively to unseen datasets. SurgMLLMBench will be publicly released as a robust resource to advance multimodal surgical AI research, supporting reproducible evaluation and development of interactive surgical reasoning models.
>
---
#### [new 073] Qwen3-VL Technical Report
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Qwen3-VL，一种先进的多模态大模型，解决长文本与跨模态内容理解难题。通过增强的时空建模、深度视觉-语言对齐及文本时间对齐技术，实现256K令牌长上下文下图文视频的高效融合与推理，在多模态理解与数学推理任务中表现领先。**

- **链接: [https://arxiv.org/pdf/2511.21631v1](https://arxiv.org/pdf/2511.21631v1)**

> **作者:** Shuai Bai; Yuxuan Cai; Ruizhe Chen; Keqin Chen; Xionghui Chen; Zesen Cheng; Lianghao Deng; Wei Ding; Chang Gao; Chunjiang Ge; Wenbin Ge; Zhifang Guo; Qidong Huang; Jie Huang; Fei Huang; Binyuan Hui; Shutong Jiang; Zhaohai Li; Mingsheng Li; Mei Li; Kaixin Li; Zicheng Lin; Junyang Lin; Xuejing Liu; Jiawei Liu; Chenglong Liu; Yang Liu; Dayiheng Liu; Shixuan Liu; Dunjie Lu; Ruilin Luo; Chenxu Lv; Rui Men; Lingchen Meng; Xuancheng Ren; Xingzhang Ren; Sibo Song; Yuchong Sun; Jun Tang; Jianhong Tu; Jianqiang Wan; Peng Wang; Pengfei Wang; Qiuyue Wang; Yuxuan Wang; Tianbao Xie; Yiheng Xu; Haiyang Xu; Jin Xu; Zhibo Yang; Mingkun Yang; Jianxin Yang; An Yang; Bowen Yu; Fei Zhang; Hang Zhang; Xi Zhang; Bo Zheng; Humen Zhong; Jingren Zhou; Fan Zhou; Jing Zhou; Yuanzhi Zhu; Ke Zhu
>
> **备注:** 42 pages
>
> **摘要:** We introduce Qwen3-VL, the most capable vision-language model in the Qwen series to date, achieving superior performance across a broad range of multimodal benchmarks. It natively supports interleaved contexts of up to 256K tokens, seamlessly integrating text, images, and video. The model family includes both dense (2B/4B/8B/32B) and mixture-of-experts (30B-A3B/235B-A22B) variants to accommodate diverse latency-quality trade-offs. Qwen3-VL delivers three core pillars: (i) markedly stronger pure-text understanding, surpassing comparable text-only backbones in several cases; (ii) robust long-context comprehension with a native 256K-token window for both text and interleaved multimodal inputs, enabling faithful retention, retrieval, and cross-referencing across long documents and videos; and (iii) advanced multimodal reasoning across single-image, multi-image, and video tasks, demonstrating leading performance on comprehensive evaluations such as MMMU and visual-math benchmarks (e.g., MathVista and MathVision). Architecturally, we introduce three key upgrades: (i) an enhanced interleaved-MRoPE for stronger spatial-temporal modeling across images and video; (ii) DeepStack integration, which effectively leverages multi-level ViT features to tighten vision-language alignment; and (iii) text-based time alignment for video, evolving from T-RoPE to explicit textual timestamp alignment for more precise temporal grounding. Under comparable token budgets and latency constraints, Qwen3-VL achieves superior performance in both dense and Mixture-of-Experts (MoE) architectures. We envision Qwen3-VL serving as a foundational engine for image-grounded reasoning, agentic decision-making, and multimodal code intelligence in real-world workflows.
>
---
#### [new 074] PG-ControlNet: A Physics-Guided ControlNet for Generative Spatially Varying Image Deblurring
- **分类: cs.CV**

- **简介: 该论文针对复杂空间变异性图像去模糊任务，解决现有方法物理准确性与感知真实感难以兼顾的问题。提出PG-ControlNet，通过建模高维密集退化核场，以物理约束引导生成过程，有效融合模型与生成优势，在严重模糊场景下实现更精确且自然的去模糊效果。**

- **链接: [https://arxiv.org/pdf/2511.21043v1](https://arxiv.org/pdf/2511.21043v1)**

> **作者:** Hakki Motorcu; Mujdat Cetin
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Spatially varying image deblurring remains a fundamentally ill-posed problem, especially when degradations arise from complex mixtures of motion and other forms of blur under significant noise. State-of-the-art learning-based approaches generally fall into two paradigms: model-based deep unrolling methods that enforce physical constraints by modeling the degradations, but often produce over-smoothed, artifact-laden textures, and generative models that achieve superior perceptual quality yet hallucinate details due to weak physical constraints. In this paper, we propose a novel framework that uniquely reconciles these paradigms by taming a powerful generative prior with explicit, dense physical constraints. Rather than oversimplifying the degradation field, we model it as a dense continuum of high-dimensional compressed kernels, ensuring that minute variations in motion and other degradation patterns are captured. We leverage this rich descriptor field to condition a ControlNet architecture, strongly guiding the diffusion sampling process. Extensive experiments demonstrate that our method effectively bridges the gap between physical accuracy and perceptual realism, outperforming state-of-the-art model-based methods as well as generative baselines in challenging, severely blurred scenarios.
>
---
#### [new 075] Towards an Effective Action-Region Tracking Framework for Fine-grained Video Action Recognition
- **分类: cs.CV**

- **简介: 该论文针对细粒度视频动作识别任务，解决现有方法难以捕捉局部细微动态变化的问题。提出动作区域追踪（ART）框架，通过文本约束的语义查询追踪关键区域动态，构建时序连贯的动作轨迹，并设计多层级对比约束与任务优化机制，有效区分相似动作。**

- **链接: [https://arxiv.org/pdf/2511.21202v1](https://arxiv.org/pdf/2511.21202v1)**

> **作者:** Baoli Sun; Yihan Wang; Xinzhu Ma; Zhihui Wang; Kun Lu; Zhiyong Wang
>
> **摘要:** Fine-grained action recognition (FGAR) aims to identify subtle and distinctive differences among fine-grained action categories. However, current recognition methods often capture coarse-grained motion patterns but struggle to identify subtle details in local regions evolving over time. In this work, we introduce the Action-Region Tracking (ART) framework, a novel solution leveraging a query-response mechanism to discover and track the dynamics of distinctive local details, enabling effective distinction of similar actions. Specifically, we propose a region-specific semantic activation module that employs discriminative and text-constrained semantics as queries to capture the most action-related region responses in each video frame, facilitating interaction among spatial and temporal dimensions with corresponding video features. The captured region responses are organized into action tracklets, which characterize region-based action dynamics by linking related responses across video frames in a coherent sequence. The text-constrained queries encode nuanced semantic representations derived from textual descriptions of action labels extracted by language branches within Visual Language Models (VLMs). To optimize the action tracklets, we design a multi-level tracklet contrastive constraint among region responses at spatial and temporal levels, enabling effective discrimination within each frame and correlation between adjacent frames. Additionally, a task-specific fine-tuning mechanism refines textual semantics such that semantic representations encoded by VLMs are preserved while optimized for task preferences. Comprehensive experiments on widely used action recognition benchmarks demonstrate the superiority to previous state-of-the-art baselines.
>
---
#### [new 076] FlowerDance: MeanFlow for Efficient and Refined 3D Dance Generation
- **分类: cs.CV**

- **简介: 该论文研究音乐到舞蹈的生成任务，旨在提升3D舞蹈生成的质量与效率。针对现有方法生成慢、资源消耗大的问题，提出FlowerDance，结合MeanFlow与物理约束，实现少步采样高效生成，并采用轻量架构支持非自回归生成与交互式编辑，显著提升速度与质量。**

- **链接: [https://arxiv.org/pdf/2511.21029v1](https://arxiv.org/pdf/2511.21029v1)**

> **作者:** Kaixing Yang; Xulong Tang; Ziqiao Peng; Xiangyue Zhang; Puwei Wang; Jun He; Hongyan Liu
>
> **摘要:** Music-to-dance generation aims to translate auditory signals into expressive human motion, with broad applications in virtual reality, choreography, and digital entertainment. Despite promising progress, the limited generation efficiency of existing methods leaves insufficient computational headroom for high-fidelity 3D rendering, thereby constraining the expressiveness of 3D characters during real-world applications. Thus, we propose FlowerDance, which not only generates refined motion with physical plausibility and artistic expressiveness, but also achieves significant generation efficiency on inference speed and memory utilization . Specifically, FlowerDance combines MeanFlow with Physical Consistency Constraints, which enables high-quality motion generation with only a few sampling steps. Moreover, FlowerDance leverages a simple but efficient model architecture with BiMamba-based backbone and Channel-Level Cross-Modal Fusion, which generates dance with efficient non-autoregressive manner. Meanwhile, FlowerDance supports motion editing, enabling users to interactively refine dance sequences. Extensive experiments on AIST++ and FineDance show that FlowerDance achieves state-of-the-art results in both motion quality and generation efficiency. Code will be released upon acceptance.
>
---
#### [new 077] Scenes as Tokens: Multi-Scale Normal Distributions Transform Tokenizer for General 3D Vision-Language Understanding
- **分类: cs.CV**

- **简介: 该论文提出NDTokenizer3D，针对3D视觉语言理解中场景令牌化困难的问题，基于多尺度正态分布变换（NDT）构建三阶段令牌化流程。通过多尺度NDT编码器与解码器，实现从点云到统一场景令牌的转换，并支持交互式提示与分割任务，显著提升3D指代分割、视觉问答与密集描述等任务性能。**

- **链接: [https://arxiv.org/pdf/2511.21191v1](https://arxiv.org/pdf/2511.21191v1)**

> **作者:** Yutao Tang; Cheng Zhao; Gaurav Mittal; Rohith Kukkala; Rama Chellappa; Cheng Peng; Mei Chen
>
> **摘要:** Recent advances in 3D vision-language models (VLMs) highlight a strong potential for 3D scene understanding and reasoning. However, effectively tokenizing 3D scenes into holistic scene tokens, and leveraging these tokens across diverse 3D understanding tasks, remain highly challenging. We present NDTokenizer3D, a generalist 3D VLM that performs a wide range of 3D scene understanding tasks while naturally supporting human interactions, thereby bridging language-level reasoning with 3D spatial understanding. The core of our approach is a novel three-stage scene tokenization pipeline built upon a Multi-Scale Normal Distributions Transform (NDT) representation, paired with a Multi-Scale NDT Decoder (MSDec). Specifically, NDTokenizer3D first constructs a multi-scale NDT representation from raw high-resolution point clouds, preserving both global context and fine-grained geometric details. Next, the MSDec progressively fuses cross-scale NDT features, producing holistic scene tokens consumable by LLM endpoints. Beyond tokenization, MSDec is repurposed as a general interface for human-interactive prompting (points, boxes, masks) and segmentation-mask decoding, unifying diverse 3D scene understanding tasks within a single architecture. With this compact and unified design, NDTokenizer3D offers a fine-grained, general-purpose 3D VLM, achieving remarkable improvements in 3D Referring Segmentation, 3D Visual Question Answering, and 3D Dense Captioning.
>
---
#### [new 078] Smooth regularization for efficient video recognition
- **分类: cs.CV**

- **简介: 该论文针对视频识别任务，解决轻量级模型难以捕捉复杂时序动态的问题。提出一种基于高斯随机游走的平滑正则化方法，强制连续帧间特征表示平滑过渡，增强时间一致性。在Kinetics-600上提升3.8%–6.4%，显著优于现有轻量模型。**

- **链接: [https://arxiv.org/pdf/2511.20928v1](https://arxiv.org/pdf/2511.20928v1)**

> **作者:** Gil Goldman; Raja Giryes; Mahadev Satyanarayanan
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** We propose a smooth regularization technique that instills a strong temporal inductive bias in video recognition models, particularly benefiting lightweight architectures. Our method encourages smoothness in the intermediate-layer embeddings of consecutive frames by modeling their changes as a Gaussian Random Walk (GRW). This penalizes abrupt representational shifts, thereby promoting low-acceleration solutions that better align with the natural temporal coherence inherent in videos. By leveraging this enforced smoothness, lightweight models can more effectively capture complex temporal dynamics. Applied to such models, our technique yields a 3.8% to 6.4% accuracy improvement on Kinetics-600. Notably, the MoViNets model family trained with our smooth regularization improves the current state of the art by 3.8% to 6.1% within their respective FLOP constraints, while MobileNetV3 and the MoViNets-Stream family achieve gains of 4.9% to 6.4% over prior state-of-the-art models with comparable memory footprints. Our code and models are available at https://github.com/gilgoldm/grw-smoothing.
>
---
#### [new 079] E-M3RF: An Equivariant Multimodal 3D Re-assembly Framework
- **分类: cs.CV**

- **简介: 该论文针对3D碎片重装任务，解决几何信息不足时的装配难题及物理重叠问题。提出E-M3RF框架，融合点云位置与颜色的多模态特征，利用旋转等变编码与Transformer增强表示，通过SE(3)流匹配预测变换。在多个数据集上验证，显著降低误差并提升重装精度。**

- **链接: [https://arxiv.org/pdf/2511.21422v1](https://arxiv.org/pdf/2511.21422v1)**

> **作者:** Adeela Islam; Stefano Fiorini; Manuel Lecha; Theodore Tsesmelis; Stuart James; Pietro Morerio; Alessio Del Bue
>
> **摘要:** 3D reassembly is a fundamental geometric problem, and in recent years it has increasingly been challenged by deep learning methods rather than classical optimization. While learning approaches have shown promising results, most still rely primarily on geometric features to assemble a whole from its parts. As a result, methods struggle when geometry alone is insufficient or ambiguous, for example, for small, eroded, or symmetric fragments. Additionally, solutions do not impose physical constraints that explicitly prevent overlapping assemblies. To address these limitations, we introduce E-M3RF, an equivariant multimodal 3D reassembly framework that takes as input the point clouds, containing both point positions and colors of fractured fragments, and predicts the transformations required to reassemble them using SE(3) flow matching. Each fragment is represented by both geometric and color features: i) 3D point positions are encoded as rotationconsistent geometric features using a rotation-equivariant encoder, ii) the colors at each 3D point are encoded with a transformer. The two feature sets are then combined to form a multimodal representation. We experimented on four datasets: two synthetic datasets, Breaking Bad and Fantastic Breaks, and two real-world cultural heritage datasets, RePAIR and Presious, demonstrating that E-M3RF on the RePAIR dataset reduces rotation error by 23.1% and translation error by 13.2%, while Chamfer Distance decreases by 18.4% compared to competing methods.
>
---
#### [new 080] TrafficLens: Multi-Camera Traffic Video Analysis Using LLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出TrafficLens，针对多摄像头交通视频分析中视频转文本效率低的问题。通过序列化处理重叠区域视频，结合VLM与相似性检测，减少冗余计算，实现快速生成高精度文本描述，显著提升处理速度。**

- **链接: [https://arxiv.org/pdf/2511.20965v1](https://arxiv.org/pdf/2511.20965v1)**

> **作者:** Md Adnan Arefeen; Biplob Debnath; Srimat Chakradhar
>
> **备注:** 2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC)
>
> **摘要:** Traffic cameras are essential in urban areas, playing a crucial role in intelligent transportation systems. Multiple cameras at intersections enhance law enforcement capabilities, traffic management, and pedestrian safety. However, efficiently managing and analyzing multi-camera feeds poses challenges due to the vast amount of data. Analyzing such huge video data requires advanced analytical tools. While Large Language Models (LLMs) like ChatGPT, equipped with retrieval-augmented generation (RAG) systems, excel in text-based tasks, integrating them into traffic video analysis demands converting video data into text using a Vision-Language Model (VLM), which is time-consuming and delays the timely utilization of traffic videos for generating insights and investigating incidents. To address these challenges, we propose TrafficLens, a tailored algorithm for multi-camera traffic intersections. TrafficLens employs a sequential approach, utilizing overlapping coverage areas of cameras. It iteratively applies VLMs with varying token limits, using previous outputs as prompts for subsequent cameras, enabling rapid generation of detailed textual descriptions while reducing processing time. Additionally, TrafficLens intelligently bypasses redundant VLM invocations through an object-level similarity detector. Experimental results with real-world datasets demonstrate that TrafficLens reduces video-to-text conversion time by up to $4\times$ while maintaining information accuracy.
>
---
#### [new 081] The Age-specific Alzheimer 's Disease Prediction with Characteristic Constraints in Nonuniform Time Span
- **分类: cs.CV**

- **简介: 该论文针对非均匀时间间隔下阿尔茨海默病的年龄特异性预测问题，提出一种基于定量指标约束的序列图像生成方法。通过引入年龄缩放因子和像素损失优化，提升MRI图像生成质量，增强疾病进展预测准确性，显著改善长期预后图像的结构相似性。**

- **链接: [https://arxiv.org/pdf/2511.21530v1](https://arxiv.org/pdf/2511.21530v1)**

> **作者:** Xin Hong; Kaifeng Huang
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** Alzheimer's disease is a debilitating disorder marked by a decline in cognitive function. Timely identification of the disease is essential for the development of personalized treatment strategies that aim to mitigate its progression. The application of generated images for the prediction of Alzheimer's disease poses challenges, particularly in accurately representing the disease's characteristics when input sequences are captured at irregular time intervals. This study presents an innovative methodology for sequential image generation, guided by quantitative metrics, to maintain the essential features indicative of disease progression. Furthermore, an age-scaling factor is integrated into the process to produce age-specific MRI images, facilitating the prediction of advanced stages of the disease. The results obtained from the ablation study suggest that the inclusion of quantitative metrics significantly improves the accuracy of MRI image synthesis. Furthermore, the application of age-scaled pixel loss contributed to the enhanced iterative generation of MRI images. In terms of long-term disease prognosis, the Structural Similarity Index reached a peak value of 0.882, indicating a substantial degree of similarity in the synthesized images.
>
---
#### [new 082] Privacy-Preserving Federated Vision Transformer Learning Leveraging Lightweight Homomorphic Encryption in Medical AI
- **分类: cs.CV; cs.CR**

- **简介: 该论文针对医疗图像联邦学习中的隐私泄露问题，提出基于视觉变压器（ViT）与轻量同态加密（CKKS）的隐私保护框架。通过加密ViT的CLS token实现安全聚合，显著降低通信开销并抵御模型逆向攻击，有效保护患者数据隐私，同时保持高分类精度。**

- **链接: [https://arxiv.org/pdf/2511.20983v1](https://arxiv.org/pdf/2511.20983v1)**

> **作者:** Al Amin; Kamrul Hasan; Liang Hong; Sharif Ullah
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Collaborative machine learning across healthcare institutions promises improved diagnostic accuracy by leveraging diverse datasets, yet privacy regulations such as HIPAA prohibit direct patient data sharing. While federated learning (FL) enables decentralized training without raw data exchange, recent studies show that model gradients in conventional FL remain vulnerable to reconstruction attacks, potentially exposing sensitive medical information. This paper presents a privacy-preserving federated learning framework combining Vision Transformers (ViT) with homomorphic encryption (HE) for secure multi-institutional histopathology classification. The approach leverages the ViT CLS token as a compact 768-dimensional feature representation for secure aggregation, encrypting these tokens using CKKS homomorphic encryption before transmission to the server. We demonstrate that encrypting CLS tokens achieves a 30-fold communication reduction compared to gradient encryption while maintaining strong privacy guarantees. Through evaluation on a three-client federated setup for lung cancer histopathology classification, we show that gradients are highly susceptible to model inversion attacks (PSNR: 52.26 dB, SSIM: 0.999, NMI: 0.741), enabling near-perfect image reconstruction. In contrast, the proposed CLS-protected HE approach prevents such attacks while enabling encrypted inference directly on ciphertexts, requiring only 326 KB of encrypted data transmission per aggregation round. The framework achieves 96.12 percent global classification accuracy in the unencrypted domain and 90.02 percent in the encrypted domain.
>
---
#### [new 083] Generalized Design Choices for Deepfake Detectors
- **分类: cs.CV**

- **简介: 该论文聚焦于深度伪造检测任务，旨在解决现有方法性能差异主要由实现细节而非核心设计导致的问题。通过系统实验，分离并评估训练、推理及增量更新中的各项设计因素，识别出提升检测精度与泛化能力的关键实践，为未来检测系统提供架构无关的最佳设计指南。**

- **链接: [https://arxiv.org/pdf/2511.21507v1](https://arxiv.org/pdf/2511.21507v1)**

> **作者:** Lorenzo Pellegrini; Serafino Pandolfini; Davide Maltoni; Matteo Ferrara; Marco Prati; Marco Ramilli
>
> **备注:** 12 pages, 9 figures, 10 tables, code available: https://github.com/MI-BioLab/AI-GenBench
>
> **摘要:** The effectiveness of deepfake detection methods often depends less on their core design and more on implementation details such as data preprocessing, augmentation strategies, and optimization techniques. These factors make it difficult to fairly compare detectors and to understand which factors truly contribute to their performance. To address this, we systematically investigate how different design choices influence the accuracy and generalization capabilities of deepfake detection models, focusing on aspects related to training, inference, and incremental updates. By isolating the impact of individual factors, we aim to establish robust, architecture-agnostic best practices for the design and development of future deepfake detection systems. Our experiments identify a set of design choices that consistently improve deepfake detection and enable state-of-the-art performance on the AI-GenBench benchmark.
>
---
#### [new 084] DeepRFTv2: Kernel-level Learning for Image Deblurring
- **分类: cs.CV**

- **简介: 该论文针对图像去模糊任务，解决现有深度网络仅在像素级学习、难以理解模糊本质的问题。提出傅里叶核估计器（FKE），在频域实现核级模糊过程学习，结合特征与估计核的卷积，提升去模糊性能，并设计高效多尺度架构，实现低内存训练。**

- **链接: [https://arxiv.org/pdf/2511.21132v1](https://arxiv.org/pdf/2511.21132v1)**

> **作者:** Xintian Mao; Haofei Song; Yin-Nian Liu; Qingli Li; Yan Wang
>
> **摘要:** It is well-known that if a network aims to learn how to deblur, it should understand the blur process. Blurring is naturally caused by the convolution of the sharp image with the blur kernel. Thus, allowing the network to learn the blur process in the kernel-level can significantly improve the image deblurring performance. But, current deep networks are still at the pixel-level learning stage, either performing end-to-end pixel-level restoration or stage-wise pseudo kernel-level restoration, failing to enable the deblur model to understand the essence of the blur. To this end, we propose Fourier Kernel Estimator (FKE), which considers the activation operation in Fourier space and converts the convolution problem in the spatial domain to a multiplication problem in Fourier space. Our FKE, jointly optimized with the deblur model, enables the network to learn the kernel-level blur process with low complexity and without any additional supervision. Furthermore, we change the convolution object of the kernel from ``image" to network extracted ``feature", whose rich semantic and structural information is more suitable to blur process learning. With the convolution of the feature and the estimated kernel, our model can learn the essence of blur in kernel-level. To further improve the efficiency of feature extraction, we design a decoupled multi-scale architecture with multiple hierarchical sub-unets with a reversible strategy, which allows better multi-scale encoding and decoding in low training memory. Extensive experiments indicate that our method achieves state-of-the-art motion deblurring results and show potential for handling other kernel-related problems. Analysis also shows our kernel estimator is able to learn physically meaningful kernels. The code will be available at https://github.com/DeepMed-Lab-ECNU/Single-Image-Deblur.
>
---
#### [new 085] Attention-Guided Patch-Wise Sparse Adversarial Attacks on Vision-Language-Action Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉-语言-动作（VLA）模型的对抗攻击问题，提出ADVLA框架。通过在视觉编码器投影的文本特征空间中施加稀疏、聚焦的对抗扰动，实现低幅值、局部稀疏攻击。无需端到端训练，单步攻击仅需0.06秒，扰动不足10%像素，成功率近100%，有效提升攻击隐蔽性与效率。**

- **链接: [https://arxiv.org/pdf/2511.21663v1](https://arxiv.org/pdf/2511.21663v1)**

> **作者:** Naifu Zhang; Wei Tao; Xi Xiao; Qianpu Sun; Yuxin Zheng; Wentao Mo; Peiqiang Wang; Nan Zhang
>
> **摘要:** In recent years, Vision-Language-Action (VLA) models in embodied intelligence have developed rapidly. However, existing adversarial attack methods require costly end-to-end training and often generate noticeable perturbation patches. To address these limitations, we propose ADVLA, a framework that directly applies adversarial perturbations on features projected from the visual encoder into the textual feature space. ADVLA efficiently disrupts downstream action predictions under low-amplitude constraints, and attention guidance allows the perturbations to be both focused and sparse. We introduce three strategies that enhance sensitivity, enforce sparsity, and concentrate perturbations. Experiments demonstrate that under an $L_{\infty}=4/255$ constraint, ADVLA combined with Top-K masking modifies less than 10% of the patches while achieving an attack success rate of nearly 100%. The perturbations are concentrated on critical regions, remain almost imperceptible in the overall image, and a single-step iteration takes only about 0.06 seconds, significantly outperforming conventional patch-based attacks. In summary, ADVLA effectively weakens downstream action predictions of VLA models under low-amplitude and locally sparse conditions, avoiding the high training costs and conspicuous perturbations of traditional patch attacks, and demonstrates unique effectiveness and practical value for attacking VLA feature spaces.
>
---
#### [new 086] FaithFusion: Harmonizing Reconstruction and Generation via Pixel-wise Information Gain
- **分类: cs.CV**

- **简介: 该论文针对可控驾驶场景重建与3D生成中几何保真与视角变化下外观合理性难以兼顾的问题，提出FaithFusion框架。通过像素级期望信息增益（EIG）统一驱动3DGS与扩散模型融合，实现高不确定性区域的精准修复与几何一致性保持，显著提升重建质量与视觉真实性。**

- **链接: [https://arxiv.org/pdf/2511.21113v1](https://arxiv.org/pdf/2511.21113v1)**

> **作者:** YuAn Wang; Xiaofan Li; Chi Huang; Wenhao Zhang; Hao Li; Bosheng Wang; Xun Sun; Jun Wang
>
> **备注:** 16 pages, 10 figures
>
> **摘要:** In controllable driving-scene reconstruction and 3D scene generation, maintaining geometric fidelity while synthesizing visually plausible appearance under large viewpoint shifts is crucial. However, effective fusion of geometry-based 3DGS and appearance-driven diffusion models faces inherent challenges, as the absence of pixel-wise, 3D-consistent editing criteria often leads to over-restoration and geometric drift. To address these issues, we introduce \textbf{FaithFusion}, a 3DGS-diffusion fusion framework driven by pixel-wise Expected Information Gain (EIG). EIG acts as a unified policy for coherent spatio-temporal synthesis: it guides diffusion as a spatial prior to refine high-uncertainty regions, while its pixel-level weighting distills the edits back into 3DGS. The resulting plug-and-play system is free from extra prior conditions and structural modifications.Extensive experiments on the Waymo dataset demonstrate that our approach attains SOTA performance across NTA-IoU, NTL-IoU, and FID, maintaining an FID of 107.47 even at 6 meters lane shift. Our code is available at https://github.com/wangyuanbiubiubiu/FaithFusion.
>
---
#### [new 087] Beyond Realism: Learning the Art of Expressive Composition with StickerNet
- **分类: cs.CV**

- **简介: 该论文提出表达性图像合成任务，旨在模拟用户在创作平台上的非现实主义编辑行为。针对传统方法过度追求真实感的问题，作者构建了基于180万真实编辑行为的StickerNet框架，通过两阶段学习预测贴纸的风格与布局参数，有效捕捉用户意图，显著提升生成内容的表达力与自然性。**

- **链接: [https://arxiv.org/pdf/2511.20957v1](https://arxiv.org/pdf/2511.20957v1)**

> **作者:** Haoming Lu; David Kocharian; Humphrey Shi
>
> **摘要:** As a widely used operation in image editing workflows, image composition has traditionally been studied with a focus on achieving visual realism and semantic plausibility. However, in practical editing scenarios of the modern content creation landscape, many compositions are not intended to preserve realism. Instead, users of online platforms motivated by gaining community recognition often aim to create content that is more artistic, playful, or socially engaging. Taking inspiration from this observation, we define the expressive composition task, a new formulation of image composition that embraces stylistic diversity and looser placement logic, reflecting how users edit images on real-world creative platforms. To address this underexplored problem, we present StickerNet, a two-stage framework that first determines the composition type, then predicts placement parameters such as opacity, mask, location, and scale accordingly. Unlike prior work that constructs datasets by simulating object placements on real images, we directly build our dataset from 1.8 million editing actions collected on an anonymous online visual creation and editing platform, each reflecting user-community validated placement decisions. This grounding in authentic editing behavior ensures strong alignment between task definition and training supervision. User studies and quantitative evaluations show that StickerNet outperforms common baselines and closely matches human placement behavior, demonstrating the effectiveness of learning from real-world editing patterns despite the inherent ambiguity of the task. This work introduces a new direction in visual understanding that emphasizes expressiveness and user intent over realism.
>
---
#### [new 088] Pygmalion Effect in Vision: Image-to-Clay Translation for Reflective Geometry Reconstruction
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文针对反射导致的3D重建难题，提出“视觉中的皮格马利翁效应”框架，通过图像到陶土状形态的转换，消除镜面信息并保留几何一致性。采用双分支网络联合训练，利用无反射的合成陶土图像作为监督信号，显著提升表面法向精度与网格完整性，实现更鲁棒的反射物体三维重建。**

- **链接: [https://arxiv.org/pdf/2511.21098v1](https://arxiv.org/pdf/2511.21098v1)**

> **作者:** Gayoung Lee; Junho Kim; Jin-Hwa Kim; Junmo Kim
>
> **摘要:** Understanding reflection remains a long-standing challenge in 3D reconstruction due to the entanglement of appearance and geometry under view-dependent reflections. In this work, we present the Pygmalion Effect in Vision, a novel framework that metaphorically "sculpts" reflective objects into clay-like forms through image-to-clay translation. Inspired by the myth of Pygmalion, our method learns to suppress specular cues while preserving intrinsic geometric consistency, enabling robust reconstruction from multi-view images containing complex reflections. Specifically, we introduce a dual-branch network in which a BRDF-based reflective branch is complemented by a clay-guided branch that stabilizes geometry and refines surface normals. The two branches are trained jointly using the synthesized clay-like images, which provide a neutral, reflection-free supervision signal that complements the reflective views. Experiments on both synthetic and real datasets demonstrate substantial improvement in normal accuracy and mesh completeness over existing reflection-handling methods. Beyond technical gains, our framework reveals that seeing by unshining, translating radiance into neutrality, can serve as a powerful inductive bias for reflective object geometry learning.
>
---
#### [new 089] DiverseVAR: Balancing Diversity and Quality of Next-Scale Visual Autoregressive Models
- **分类: cs.CV**

- **简介: 该论文针对文本条件视觉自回归模型（VAR）在生成时多样性不足的问题，提出DiverseVAR框架。通过在测试阶段注入文本嵌入噪声提升多样性，并引入尺度旅行（scale-travel）技术在不牺牲图像质量的前提下优化生成结果，实现了多样性与质量的更好平衡。**

- **链接: [https://arxiv.org/pdf/2511.21415v1](https://arxiv.org/pdf/2511.21415v1)**

> **作者:** Mingue Park; Prin Phunyaphibarn; Phillip Y. Lee; Minhyuk Sung
>
> **摘要:** We introduce DiverseVAR, a framework that enhances the diversity of text-conditioned visual autoregressive models (VAR) at test time without requiring retraining, fine-tuning, or substantial computational overhead. While VAR models have recently emerged as strong competitors to diffusion and flow models for image generation, they suffer from a critical limitation in diversity, often producing nearly identical images even for simple prompts. This issue has largely gone unnoticed amid the predominant focus on image quality. We address this limitation at test time in two stages. First, inspired by diversity enhancement techniques in diffusion models, we propose injecting noise into the text embedding. This introduces a trade-off between diversity and image quality: as diversity increases, the image quality sharply declines. To preserve quality, we propose scale-travel: a novel latent refinement technique inspired by time-travel strategies in diffusion models. Specifically, we use a multi-scale autoencoder to extract coarse-scale tokens that enable us to resume generation at intermediate stages. Extensive experiments show that combining text-embedding noise injection with our scale-travel refinement significantly enhances diversity while minimizing image-quality degradation, achieving a new Pareto frontier in the diversity-quality trade-off.
>
---
#### [new 090] CaFlow: Enhancing Long-Term Action Quality Assessment with Causal Counterfactual Flow
- **分类: cs.CV**

- **简介: 该论文针对长期动作质量评估（Long-term AQA）任务，解决现有方法对标注依赖强、易受上下文混淆因素干扰的问题。提出CaFlow框架，通过因果反事实正则化模块分离因果与混淆特征，结合双向时序流建模与循环一致性约束，提升模型鲁棒性与长时序表示稳定性，实现更准确的细粒度评分预测。**

- **链接: [https://arxiv.org/pdf/2511.21653v1](https://arxiv.org/pdf/2511.21653v1)**

> **作者:** Ruisheng Han; Kanglei Zhou; Shuang Chen; Amir Atapour-Abarghouei; Hubert P. H. Shum
>
> **摘要:** Action Quality Assessment (AQA) predicts fine-grained execution scores from action videos and is widely applied in sports, rehabilitation, and skill evaluation. Long-term AQA, as in figure skating or rhythmic gymnastics, is especially challenging since it requires modeling extended temporal dynamics while remaining robust to contextual confounders. Existing approaches either depend on costly annotations or rely on unidirectional temporal modeling, making them vulnerable to spurious correlations and unstable long-term representations. To this end, we propose CaFlow, a unified framework that integrates counterfactual de-confounding with bidirectional time-conditioned flow. The Causal Counterfactual Regularization (CCR) module disentangles causal and confounding features in a self-supervised manner and enforces causal robustness through counterfactual interventions, while the BiT-Flow module models forward and backward dynamics with a cycle-consistency constraint to produce smoother and more coherent representations. Extensive experiments on multiple long-term AQA benchmarks demonstrate that CaFlow achieves state-of-the-art performance. Code is available at https://github.com/Harrison21/CaFlow
>
---
#### [new 091] Video Generation Models Are Good Latent Reward Models
- **分类: cs.CV**

- **简介: 该论文针对视频生成中奖励反馈学习（ReFL）效率低的问题，提出在噪声潜空间进行偏好优化的PRFL框架。解决了传统像素空间方法内存开销大、训练慢且缺乏早期监督的难题，实现了高效、端到端的潜空间优化，显著提升生成质量与训练效率。**

- **链接: [https://arxiv.org/pdf/2511.21541v1](https://arxiv.org/pdf/2511.21541v1)**

> **作者:** Xiaoyue Mi; Wenqing Yu; Jiesong Lian; Shibo Jie; Ruizhe Zhong; Zijun Liu; Guozhen Zhang; Zixiang Zhou; Zhiyong Xu; Yuan Zhou; Qinglin Lu; Fan Tang
>
> **摘要:** Reward feedback learning (ReFL) has proven effective for aligning image generation with human preferences. However, its extension to video generation faces significant challenges. Existing video reward models rely on vision-language models designed for pixel-space inputs, confining ReFL optimization to near-complete denoising steps after computationally expensive VAE decoding. This pixel-space approach incurs substantial memory overhead and increased training time, and its late-stage optimization lacks early-stage supervision, refining only visual quality rather than fundamental motion dynamics and structural coherence. In this work, we show that pre-trained video generation models are naturally suited for reward modeling in the noisy latent space, as they are explicitly designed to process noisy latent representations at arbitrary timesteps and inherently preserve temporal information through their sequential modeling capabilities. Accordingly, we propose Process Reward Feedback Learning~(PRFL), a framework that conducts preference optimization entirely in latent space, enabling efficient gradient backpropagation throughout the full denoising chain without VAE decoding. Extensive experiments demonstrate that PRFL significantly improves alignment with human preferences, while achieving substantial reductions in memory consumption and training time compared to RGB ReFL.
>
---
#### [new 092] UAVLight: A Benchmark for Illumination-Robust 3D Reconstruction in Unmanned Aerial Vehicle (UAV) Scenes
- **分类: cs.CV**

- **简介: 该论文针对无人机（UAV）场景中光照不一致导致的3D重建失真问题，提出UAVLight基准数据集。通过在固定飞行路径上多时段采集，实现光照变化下的几何一致性，解决现有数据集光照多样性与时空稳定性难以兼顾的问题，为光照鲁棒的3D重建方法提供可控、真实的评估平台。**

- **链接: [https://arxiv.org/pdf/2511.21565v1](https://arxiv.org/pdf/2511.21565v1)**

> **作者:** Kang Du; Xue Liao; Junpeng Xia; Chaozheng Guo; Yi Gu; Yirui Guan; Duotun Wang; ShengHuang; Zeyu Wang
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Illumination inconsistency is a fundamental challenge in multi-view 3D reconstruction. Variations in sunlight direction, cloud cover, and shadows break the constant-lighting assumption underlying both classical multi-view stereo (MVS) and structure from motion (SfM) pipelines and recent neural rendering methods, leading to geometry drift, color inconsistency, and shadow imprinting. This issue is especially critical in UAV-based reconstruction, where long flight durations and outdoor environments make lighting changes unavoidable. However, existing datasets either restrict capture to short time windows, thus lacking meaningful illumination diversity, or span months and seasons, where geometric and semantic changes confound the isolated study of lighting robustness. We introduce UAVLight, a controlled-yet-real benchmark for illumination-robust 3D reconstruction. Each scene is captured along repeatable, geo-referenced flight paths at multiple fixed times of day, producing natural lighting variation under consistent geometry, calibration, and viewpoints. With standardized evaluation protocols across lighting conditions, UAVLight provides a reliable foundation for developing and benchmarking reconstruction methods that are consistent, faithful, and relightable in real outdoor environments.
>
---
#### [new 093] Self-Paced Learning for Images of Antinuclear Antibodies
- **分类: cs.CV**

- **简介: 该论文针对抗核抗体（ANA）检测中的多实例多标签学习难题，提出一种自适应的自步学习框架。通过实例采样、概率伪标签分发与自步权重学习，实现无需预处理的端到端训练，有效提升复杂荧光模式识别准确率，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.21519v1](https://arxiv.org/pdf/2511.21519v1)**

> **作者:** Yiyang Jiang; Guangwu Qian; Jiaxin Wu; Qi Huang; Qing Li; Yongkang Wu; Xiao-Yong Wei
>
> **备注:** IEEE Transactions on Medical Imaging
>
> **摘要:** Antinuclear antibody (ANA) testing is a crucial method for diagnosing autoimmune disorders, including lupus, Sjögren's syndrome, and scleroderma. Despite its importance, manual ANA detection is slow, labor-intensive, and demands years of training. ANA detection is complicated by over 100 coexisting antibody types, resulting in vast fluorescent pattern combinations. Although machine learning and deep learning have enabled automation, ANA detection in real-world clinical settings presents unique challenges as it involves multi-instance, multi-label (MIML) learning. In this paper, a novel framework for ANA detection is proposed that handles the complexities of MIML tasks using unaltered microscope images without manual preprocessing. Inspired by human labeling logic, it identifies consistent ANA sub-regions and assigns aggregated labels accordingly. These steps are implemented using three task-specific components: an instance sampler, a probabilistic pseudo-label dispatcher, and self-paced weight learning rate coefficients. The instance sampler suppresses low-confidence instances by modeling pattern confidence, while the dispatcher adaptively assigns labels based on instance distinguishability. Self-paced learning adjusts training according to empirical label observations. Our framework overcomes limitations of traditional MIML methods and supports end-to-end optimization. Extensive experiments on one ANA dataset and three public medical MIML benchmarks demonstrate the superiority of our framework. On the ANA dataset, our model achieves up to +7.0% F1-Macro and +12.6% mAP gains over the best prior method, setting new state-of-the-art results. It also ranks top-2 across all key metrics on public datasets, reducing Hamming loss and one-error by up to 18.2% and 26.9%, respectively. The source code can be accessed at https://github.com/fletcherjiang/ANA-SelfPacedLearning.
>
---
#### [new 094] Referring Video Object Segmentation with Cross-Modality Proxy Queries
- **分类: cs.CV**

- **简介: 该论文研究指代视频目标分割（RVOS）任务，旨在精准定位文本描述的视频目标。针对现有方法在跨模态对齐中缺乏帧间依赖与文本约束延迟的问题，提出ProxyFormer架构，通过动态更新的代理查询融合视觉与文本语义，实现跨模态协同与帧间一致性，提升分割精度与跟踪稳定性。**

- **链接: [https://arxiv.org/pdf/2511.21139v1](https://arxiv.org/pdf/2511.21139v1)**

> **作者:** Baoli Sun; Xinzhu Ma; Ning Wang; Zhihui Wang; Zhiyong Wang
>
> **摘要:** Referring video object segmentation (RVOS) is an emerging cross-modality task that aims to generate pixel-level maps of the target objects referred by given textual expressions. The main concept involves learning an accurate alignment of visual elements and language expressions within a semantic space. Recent approaches address cross-modality alignment through conditional queries, tracking the target object using a query-response based mechanism built upon transformer structure. However, they exhibit two limitations: (1) these conditional queries lack inter-frame dependency and variation modeling, making accurate target tracking challenging amid significant frame-to-frame variations; and (2) they integrate textual constraints belatedly, which may cause the video features potentially focus on the non-referred objects. Therefore, we propose a novel RVOS architecture called ProxyFormer, which introduces a set of proxy queries to integrate visual and text semantics and facilitate the flow of semantics between them. By progressively updating and propagating proxy queries across multiple stages of video feature encoder, ProxyFormer ensures that the video features are focused on the object of interest. This dynamic evolution also enables the establishment of inter-frame dependencies, enhancing the accuracy and coherence of object tracking. To mitigate high computational costs, we decouple cross-modality interactions into temporal and spatial dimensions. Additionally, we design a Joint Semantic Consistency (JSC) training strategy to align semantic consensus between the proxy queries and the combined video-text pairs. Comprehensive experiments on four widely used RVOS benchmarks demonstrate the superiority of our ProxyFormer to the state-of-the-art methods.
>
---
#### [new 095] Active Learning for GCN-based Action Recognition
- **分类: cs.CV**

- **简介: 该论文针对骨架动作识别中标签数据稀缺的问题，提出一种标签高效的GCN模型。通过设计基于对抗策略的采样函数，平衡代表性、多样性和不确定性，精选关键样本；同时引入双向稳定GCN架构，提升特征空间映射能力。实验表明，该方法在两个基准上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.21625v1](https://arxiv.org/pdf/2511.21625v1)**

> **作者:** Hichem Sahbi
>
> **摘要:** Despite the notable success of graph convolutional networks (GCNs) in skeleton-based action recognition, their performance often depends on large volumes of labeled data, which are frequently scarce in practical settings. To address this limitation, we propose a novel label-efficient GCN model. Our work makes two primary contributions. First, we develop a novel acquisition function that employs an adversarial strategy to identify a compact set of informative exemplars for labeling. This selection process balances representativeness, diversity, and uncertainty. Second, we introduce bidirectional and stable GCN architectures. These enhanced networks facilitate a more effective mapping between the ambient and latent data spaces, enabling a better understanding of the learned exemplar distribution. Extensive evaluations on two challenging skeleton-based action recognition benchmarks reveal significant improvements achieved by our label-efficient GCNs compared to prior work.
>
---
#### [new 096] MUSE: Manipulating Unified Framework for Synthesizing Emotions in Images via Test-Time Optimization
- **分类: cs.CV**

- **简介: 该论文提出MUSE，首个统一的图像情感合成框架，解决生成与编辑分离导致的效率低、应用受限问题。通过测试时优化，利用预训练情绪分类器引导情感生成，动态确定引导时机，采用多情感损失减少干扰，实现高效精准的情感图像合成与编辑。**

- **链接: [https://arxiv.org/pdf/2511.21051v1](https://arxiv.org/pdf/2511.21051v1)**

> **作者:** Yingjie Xia; Xi Wang; Jinglei Shi; Vicky Kalogeiton; Jian Yang
>
> **摘要:** Images evoke emotions that profoundly influence perception, often prioritized over content. Current Image Emotional Synthesis (IES) approaches artificially separate generation and editing tasks, creating inefficiencies and limiting applications where these tasks naturally intertwine, such as therapeutic interventions or storytelling. In this work, we introduce MUSE, the first unified framework capable of both emotional generation and editing. By adopting a strategy conceptually aligned with Test-Time Scaling (TTS) that widely used in both LLM and diffusion model communities, it avoids the requirement for additional updating diffusion model and specialized emotional synthesis datasets. More specifically, MUSE addresses three key questions in emotional synthesis: (1) HOW to stably guide synthesis by leveraging an off-the-shelf emotion classifier with gradient-based optimization of emotional tokens; (2) WHEN to introduce emotional guidance by identifying the optimal timing using semantic similarity as a supervisory signal; and (3) WHICH emotion to guide synthesis through a multi-emotion loss that reduces interference from inherent and similar emotions. Experimental results show that MUSE performs favorably against all methods for both generation and editing, improving emotional accuracy and semantic diversity while maintaining an optimal balance between desired content, adherence to text prompts, and realistic emotional expression. It establishes a new paradigm for emotion synthesis.
>
---
#### [new 097] UruDendro4: A Benchmark Dataset for Automatic Tree-Ring Detection in Cross-Section Images of Pinus taeda L
- **分类: cs.CV**

- **简介: 该论文提出UruDendro4数据集，解决针叶树横截面图像中树轮自动检测的数据稀缺问题。通过102张带人工标注的松树横截面图像，支持树轮体积建模与算法训练，提供基准性能，验证了深度学习方法在树轮检测中的有效性及数据对模型泛化能力的提升。**

- **链接: [https://arxiv.org/pdf/2511.20935v1](https://arxiv.org/pdf/2511.20935v1)**

> **作者:** Henry Marichal; Joaquin Blanco; Diego Passarella; Gregory Randall
>
> **备注:** Accepted at IEEE 15th International Conference on Pattern Recognition Systems (ICPRS-25)
>
> **摘要:** Tree-ring growth represents the annual wood increment for a tree, and quantifying it allows researchers to assess which silvicultural practices are best suited for each species. Manual measurement of this growth is time-consuming and often imprecise, as it is typically performed along 4 to 8 radial directions on a cross-sectional disc. In recent years, automated algorithms and datasets have emerged to enhance accuracy and automate the delineation of annual rings in cross-sectional images. To address the scarcity of wood cross-section data, we introduce the UruDendro4 dataset, a collection of 102 image samples of Pinus taeda L., each manually annotated with annual growth rings. Unlike existing public datasets, UruDendro4 includes samples extracted at multiple heights along the stem, allowing for the volumetric modeling of annual growth using manually delineated rings. This dataset (images and annotations) allows the development of volumetric models for annual wood estimation based on cross-sectional imagery. Additionally, we provide a performance baseline for automatic ring detection on this dataset using state-of-the-art methods. The highest performance was achieved by the DeepCS-TRD method, with a mean Average Precision of 0.838, a mean Average Recall of 0.782, and an Adapted Rand Error score of 0.084. A series of ablation experiments were conducted to empirically validate the final parameter configuration. Furthermore, we empirically demonstrate that training a learning model including this dataset improves the model's generalization in the tree-ring detection task.
>
---
#### [new 098] EoS-FM: Can an Ensemble of Specialist Models act as a Generalist Feature Extractor?
- **分类: cs.CV**

- **简介: 该论文针对遥感领域基础模型训练资源消耗大的问题，提出EoS-FM框架。通过轻量级任务专用模型（ConvNeXtV2）的集成，实现高效、可解释且可扩展的特征提取。解决了大模型依赖海量数据与算力的瓶颈，支持联邦学习与持续集成，推动可持续遥感AI发展。**

- **链接: [https://arxiv.org/pdf/2511.21523v1](https://arxiv.org/pdf/2511.21523v1)**

> **作者:** Pierre Adorni; Minh-Tan Pham; Stéphane May; Sébastien Lefèvre
>
> **摘要:** Recent advances in foundation models have shown great promise in domains such as natural language processing and computer vision, and similar efforts are now emerging in the Earth Observation community. These models aim to generalize across tasks with limited supervision, reducing the need for training separate models for each task. However, current strategies, which largely focus on scaling model size and dataset volume, require prohibitive computational and data resources, limiting accessibility to only a few large institutions. Moreover, this paradigm of ever-larger models stands in stark contrast with the principles of sustainable and environmentally responsible AI, as it leads to immense carbon footprints and resource inefficiency. In this work, we present a novel and efficient alternative: an Ensemble-of-Specialists framework for building Remote Sensing Foundation Models (RSFMs). Our method decomposes the training process into lightweight, task-specific ConvNeXtV2 specialists that can be frozen and reused. This modular approach offers strong advantages in efficiency, interpretability, and extensibility. Moreover, it naturally supports federated training, pruning, and continuous specialist integration, making it particularly well-suited for collaborative and resource-constrained settings. Our framework sets a new direction for building scalable and efficient RSFMs.
>
---
#### [new 099] Video Object Recognition in Mobile Edge Networks: Local Tracking or Edge Detection?
- **分类: cs.CV; eess.IV**

- **简介: 该论文研究移动边缘网络中的视频目标识别任务，针对资源受限设备在帧级分析中如何决策本地跟踪与边缘检测的问题，提出基于深度强化学习的LTED-Ada算法。通过考虑帧率、精度与延迟需求，实现自适应选择，并在多设备场景下结合联邦学习提升泛化能力，实验验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2511.20716v1](https://arxiv.org/pdf/2511.20716v1)**

> **作者:** Kun Guo; Yun Shen; Xijun Wang; Chaoqun You; Yun Rui; Tony Q. S. Quek
>
> **摘要:** Fast and accurate video object recognition, which relies on frame-by-frame video analytics, remains a challenge for resource-constrained devices such as traffic cameras. Recent advances in mobile edge computing have made it possible to offload computation-intensive object detection to edge servers equipped with high-accuracy neural networks, while lightweight and fast object tracking algorithms run locally on devices. This hybrid approach offers a promising solution but introduces a new challenge: deciding when to perform edge detection versus local tracking. To address this, we formulate two long-term optimization problems for both single-device and multi-device scenarios, taking into account the temporal correlation of consecutive frames and the dynamic conditions of mobile edge networks. Based on the formulation, we propose the LTED-Ada in single-device setting, a deep reinforcement learning-based algorithm that adaptively selects between local tracking and edge detection, according to the frame rate as well as recognition accuracy and delay requirement. In multi-device setting, we further enhance LTED-Ada using federated learning to enable collaborative policy training across devices, thereby improving its generalization to unseen frame rates and performance requirements. Finally, we conduct extensive hardware-in-the-loop experiments using multiple Raspberry Pi 4B devices and a personal computer as the edge server, demonstrating the superiority of LTED-Ada.
>
---
#### [new 100] RefTr: Recurrent Refinement of Confluent Trajectories for 3D Vascular Tree Centerline Graphs
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对3D血管树中心线生成任务，解决现有方法召回率低、拓扑错误等问题。提出RefTr模型，采用生产-精炼架构，通过递归精炼共汇轨迹，保证树结构拓扑正确性，提升召回率与推理效率，显著减少参数量，实现更高精度与速度。**

- **链接: [https://arxiv.org/pdf/2511.20823v1](https://arxiv.org/pdf/2511.20823v1)**

> **作者:** Roman Naeem; David Hagerman; Jennifer Alvén; Fredrik Kahl
>
> **摘要:** Tubular trees, such as blood vessels and lung airways, are essential for material transport within the human body. Accurately detecting their centerlines with correct tree topology is critical for clinical tasks such as diagnosis, treatment planning, and surgical navigation. In these applications, maintaining high recall is crucial, as missing small branches can result in fatal mistakes caused by incomplete assessments or undetected abnormalities. We present RefTr, a 3D image-to-graph model for centerline generation of vascular trees via recurrent refinement of confluent trajectories. RefTr uses a Producer-Refiner architecture based on a Transformer decoder, where the Producer proposes a set of initial confluent trajectories that are recurrently refined by the Refiner to produce final trajectories, which forms the centerline graph. The confluent trajectory representation enables refinement of complete trajectories while explicitly enforcing a valid tree topology. The recurrent refinement scheme improves precision and reuses the same Refiner block across multiple steps, yielding a 2.4x reduction in decoder parameters compared to previous SOTA. We also introduce an efficient non-maximum suppression algorithm for spatial tree graphs to merge duplicate branches and boost precision. Across multiple public centerline datasets, RefTr achieves superior recall and comparable precision to previous SOTA, while offering faster inference and substantially fewer parameters, demonstrating its potential as a new state-of-the-art framework for vascular tree analysis in 3D medical imaging.
>
---
#### [new 101] ReSAM: Refine, Requery, and Reinforce: Self-Prompting Point-Supervised Segmentation for Remote Sensing Images
- **分类: cs.CV**

- **简介: 该论文针对遥感图像分割中领域差异大、密集标注稀缺的问题，提出基于自提示的点监督分割框架ReSAM。通过“精炼-重查询-强化”循环，利用稀疏点标注迭代优化伪掩码，提升SAM在遥感图像上的泛化能力，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.21606v1](https://arxiv.org/pdf/2511.21606v1)**

> **作者:** M. Naseer Subhani
>
> **摘要:** Interactive segmentation models such as the Segment Anything Model (SAM) have demonstrated remarkable generalization on natural images, but perform suboptimally on remote sensing imagery (RSI) due to severe domain shift and the scarcity of dense annotations. To address this, we propose a self-prompting, point-supervised framework that adapts SAM to RSIs using only sparse point annotations. Our method employs a Refine-Requery-Reinforce loop, where coarse pseudo-masks are generated from initial points (Refine), improved with self-constructed box prompts (Requery), and embeddings are aligned across iterations to reduce confirmation bias (Reinforce). Without relying on full-mask supervision, our approach progressively enhances SAM's segmentation quality and domain robustness through self-guided prompt adaptation . We evaluate our proposed method on three RSI benchmark datasets, including WHU, HRSID, and NWPU VHR-10, showing that our method consistently surpasses pretrained SAM and recent point-supervised segmentation methods. Our results demonstrate that self-prompting and semantic alignment provide an efficient path towards scalable, point-level adaptation of foundation segmentation models for remote sensing applications.
>
---
#### [new 102] Scaling Foundation Models for Radar Scene Understanding
- **分类: cs.CV**

- **简介: 该论文面向雷达场景理解任务，解决现有雷达方法碎片化、难迁移的问题。提出RadarFM基础模型，通过结构化语言监督和哈希感知对比学习，实现统一的场景表征，并在CARLA仿真环境中构建大规模标注数据集，引入定位感知评估指标，提升空间推理能力。**

- **链接: [https://arxiv.org/pdf/2511.21105v1](https://arxiv.org/pdf/2511.21105v1)**

> **作者:** Pushkal Mishra; Kshitiz Bansal; Dinesh Bharadia
>
> **摘要:** Radar sensors provide reliable perception across adverse weather, lighting, and long-range conditions. Recent advances in foundation models have transformed visual and language understanding, yet their integration with radar sensing remains largely underexplored. Existing radar approaches are fragmented and task-specific; each downstream task employs distinct architectures and training objectives, preventing transfer across tasks. In this work, we introduce RadarFM: a radar foundation model that learns unified scene-level representations through structured spatial language supervision. We make two key contributions: (1) a structured caption framework that encodes vehicle distributions in native radar coordinates, and (2) a hash-aware contrastive learning objective that quantifies continuous scene similarity rather than binary matching, enabling fine-grained spatial reasoning. Leveraging the CARLA simulator, we generate large-scale, well-annotated radar datasets across diverse driving scenarios. We also propose localization-aware metrics that assess spatial accuracy beyond traditional detection measures.
>
---
#### [new 103] Harmony: Harmonizing Audio and Video Generation through Cross-Task Synergy
- **分类: cs.CV**

- **简介: 该论文聚焦于音视频生成同步任务，针对开源模型在音频-视频对齐上的不稳定性问题，提出Harmony框架。通过跨任务协同训练缓解对应漂移，设计全局-局部解耦模块提升时序对齐精度，并引入同步增强的无分类器引导（SyncCFG），显著提升生成质量与音视频同步性。**

- **链接: [https://arxiv.org/pdf/2511.21579v1](https://arxiv.org/pdf/2511.21579v1)**

> **作者:** Teng Hu; Zhentao Yu; Guozhen Zhang; Zihan Su; Zhengguang Zhou; Youliang Zhang; Yuan Zhou; Qinglin Lu; Ran Yi
>
> **摘要:** The synthesis of synchronized audio-visual content is a key challenge in generative AI, with open-source models facing challenges in robust audio-video alignment. Our analysis reveals that this issue is rooted in three fundamental challenges of the joint diffusion process: (1) Correspondence Drift, where concurrently evolving noisy latents impede stable learning of alignment; (2) inefficient global attention mechanisms that fail to capture fine-grained temporal cues; and (3) the intra-modal bias of conventional Classifier-Free Guidance (CFG), which enhances conditionality but not cross-modal synchronization. To overcome these challenges, we introduce Harmony, a novel framework that mechanistically enforces audio-visual synchronization. We first propose a Cross-Task Synergy training paradigm to mitigate drift by leveraging strong supervisory signals from audio-driven video and video-driven audio generation tasks. Then, we design a Global-Local Decoupled Interaction Module for efficient and precise temporal-style alignment. Finally, we present a novel Synchronization-Enhanced CFG (SyncCFG) that explicitly isolates and amplifies the alignment signal during inference. Extensive experiments demonstrate that Harmony establishes a new state-of-the-art, significantly outperforming existing methods in both generation fidelity and, critically, in achieving fine-grained audio-visual synchronization.
>
---
#### [new 104] Enhanced Landmark Detection Model in Pelvic Fluoroscopy using 2D/3D Registration Loss
- **分类: cs.CV**

- **简介: 该论文针对骨盆术中透视影像中因患者体位变化导致的地标检测误差问题，提出基于2D/3D注册损失的U-Net改进框架，通过引入三维解剖结构与二维影像的配准信息，提升模型在非标准视角下的地标定位精度。**

- **链接: [https://arxiv.org/pdf/2511.21575v1](https://arxiv.org/pdf/2511.21575v1)**

> **作者:** Chou Mo; Yehyun Suh; J. Ryan Martin; Daniel Moyer
>
> **备注:** 9 pages, 3 figures, 1 table
>
> **摘要:** Automated landmark detection offers an efficient approach for medical professionals to understand patient anatomic structure and positioning using intra-operative imaging. While current detection methods for pelvic fluoroscopy demonstrate promising accuracy, most assume a fixed Antero-Posterior view of the pelvis. However, orientation often deviates from this standard view, either due to repositioning of the imaging unit or of the target structure itself. To address this limitation, we propose a novel framework that incorporates 2D/3D landmark registration into the training of a U-Net landmark prediction model. We analyze the performance difference by comparing landmark detection accuracy between the baseline U-Net, U-Net trained with Pose Estimation Loss, and U-Net fine-tuned with Pose Estimation Loss under realistic intra-operative conditions where patient pose is variable.
>
---
#### [new 105] Deformation-aware Temporal Generation for Early Prediction of Alzheimers Disease
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对阿尔茨海默病（AD）早期预测任务，解决因脑部形态变化难以自动捕捉及影像数据缺失的问题。提出变形感知时序生成网络（DATGN），通过插值补全缺失数据，并利用双向时序模块生成符合疾病进展趋势的未来MRI图像，提升分类准确率，实现早期预测。**

- **链接: [https://arxiv.org/pdf/2511.21114v1](https://arxiv.org/pdf/2511.21114v1)**

> **作者:** Xin Honga; Jie Lin; Minghui Wang
>
> **备注:** 29 pages,6figures,one column
>
> **摘要:** Alzheimer's disease (AD), a degenerative brain condition, can benefit from early prediction to slow its progression. As the disease progresses, patients typically undergo brain atrophy. Current prediction methods for Alzheimers disease largely involve analyzing morphological changes in brain images through manual feature extraction. This paper proposes a novel method, the Deformation-Aware Temporal Generative Network (DATGN), to automate the learning of morphological changes in brain images about disease progression for early prediction. Given the common occurrence of missing data in the temporal sequences of MRI images, DATGN initially interpolates incomplete sequences. Subsequently, a bidirectional temporal deformation-aware module guides the network in generating future MRI images that adhere to the disease's progression, facilitating early prediction of Alzheimer's disease. DATGN was tested for the generation of temporal sequences of future MRI images using the ADNI dataset, and the experimental results are competitive in terms of PSNR and MMSE image quality metrics. Furthermore, when DATGN-generated synthetic data was integrated into the SVM vs. CNN vs. 3DCNN-based classification methods, significant improvements were achieved from 6. 21\% to 16\% in AD vs. NC classification accuracy and from 7. 34\% to 21. 25\% in AD vs. MCI vs. NC classification accuracy. The qualitative visualization results indicate that DATGN produces MRI images consistent with the brain atrophy trend in Alzheimer's disease, enabling early disease prediction.
>
---
#### [new 106] Continual Error Correction on Low-Resource Devices
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对低资源设备上AI模型预测错误问题，提出一种轻量级持续纠错系统。通过服务器端知识蒸馏与设备端原型更新结合，实现高效少样本纠错，显著降低计算开销与遗忘率，适用于图像分类与检测任务，验证了其在真实场景中的可行性。**

- **链接: [https://arxiv.org/pdf/2511.21652v1](https://arxiv.org/pdf/2511.21652v1)**

> **作者:** Kirill Paramonov; Mete Ozay; Aristeidis Mystakidis; Nikolaos Tsalikidis; Dimitrios Sotos; Anastasios Drosou; Dimitrios Tzovaras; Hyunjun Kim; Kiseok Chang; Sangdok Mo; Namwoong Kim; Woojong Yoo; Jijoong Moon; Umberto Michieli
>
> **备注:** ACM MMSys 2025
>
> **摘要:** The proliferation of AI models in everyday devices has highlighted a critical challenge: prediction errors that degrade user experience. While existing solutions focus on error detection, they rarely provide efficient correction mechanisms, especially for resource-constrained devices. We present a novel system enabling users to correct AI misclassifications through few-shot learning, requiring minimal computational resources and storage. Our approach combines server-side foundation model training with on-device prototype-based classification, enabling efficient error correction through prototype updates rather than model retraining. The system consists of two key components: (1) a server-side pipeline that leverages knowledge distillation to transfer robust feature representations from foundation models to device-compatible architectures, and (2) a device-side mechanism that enables ultra-efficient error correction through prototype adaptation. We demonstrate our system's effectiveness on both image classification and object detection tasks, achieving over 50% error correction in one-shot scenarios on Food-101 and Flowers-102 datasets while maintaining minimal forgetting (less than 0.02%) and negligible computational overhead. Our implementation, validated through an Android demonstration app, proves the system's practicality in real-world scenarios.
>
---
#### [new 107] Open Vocabulary Compositional Explanations for Neuron Alignment
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对神经网络中神经元解释的局限性，提出一种开放词汇的组合解释框架。旨在突破依赖人工标注数据的限制，通过开放词汇语义分割生成掩码，实现对任意概念和数据集的神经元激活解释，提升解释的灵活性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.20931v1](https://arxiv.org/pdf/2511.20931v1)**

> **作者:** Biagio La Rosa; Leilani H. Gilpin
>
> **备注:** 47 pages, 11 figures
>
> **摘要:** Neurons are the fundamental building blocks of deep neural networks, and their interconnections allow AI to achieve unprecedented results. Motivated by the goal of understanding how neurons encode information, compositional explanations leverage logical relationships between concepts to express the spatial alignment between neuron activations and human knowledge. However, these explanations rely on human-annotated datasets, restricting their applicability to specific domains and predefined concepts. This paper addresses this limitation by introducing a framework for the vision domain that allows users to probe neurons for arbitrary concepts and datasets. Specifically, the framework leverages masks generated by open vocabulary semantic segmentation to compute open vocabulary compositional explanations. The proposed framework consists of three steps: specifying arbitrary concepts, generating semantic segmentation masks using open vocabulary models, and deriving compositional explanations from these masks. The paper compares the proposed framework with previous methods for computing compositional explanations both in terms of quantitative metrics and human interpretability, analyzes the differences in explanations when shifting from human-annotated data to model-annotated data, and showcases the additional capabilities provided by the framework in terms of flexibility of the explanations with respect to the tasks and properties of interest.
>
---
#### [new 108] Monet: Reasoning in Latent Visual Space Beyond Images and Language
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Monet框架，解决多模态大模型在隐空间中进行抽象视觉推理的难题。通过三阶段蒸馏SFT与VLPO强化学习，使模型在连续嵌入空间生成中间视觉思维，提升真实世界与抽象视觉任务的推理能力。**

- **链接: [https://arxiv.org/pdf/2511.21395v1](https://arxiv.org/pdf/2511.21395v1)**

> **作者:** Qixun Wang; Yang Shi; Yifei Wang; Yuanxing Zhang; Pengfei Wan; Kun Gai; Xianghua Ying; Yisen Wang
>
> **摘要:** "Thinking with images" has emerged as an effective paradigm for advancing visual reasoning, extending beyond text-only chains of thought by injecting visual evidence into intermediate reasoning steps. However, existing methods fall short of human-like abstract visual thinking, as their flexibility is fundamentally limited by external tools. In this work, we introduce Monet, a training framework that enables multimodal large language models (MLLMs) to reason directly within the latent visual space by generating continuous embeddings that function as intermediate visual thoughts. We identify two core challenges in training MLLMs for latent visual reasoning: high computational cost in latent-vision alignment and insufficient supervision over latent embeddings, and address them with a three-stage distillation-based supervised fine-tuning (SFT) pipeline. We further reveal a limitation of applying GRPO to latent reasoning: it primarily enhances text-based reasoning rather than latent reasoning. To overcome this, we propose VLPO (Visual-latent Policy Optimization), a reinforcement learning method that explicitly incorporates latent embeddings into policy gradient updates. To support SFT, we construct Monet-SFT-125K, a high-quality text-image interleaved CoT dataset containing 125K real-world, chart, OCR, and geometry CoTs. Our model, Monet-7B, shows consistent gains across real-world perception and reasoning benchmarks and exhibits strong out-of-distribution generalization on challenging abstract visual reasoning tasks. We also empirically analyze the role of each training component and discuss our early unsuccessful attempts, providing insights for future developments in visual latent reasoning. Our model, data, and code are available at https://github.com/NOVAglow646/Monet.
>
---
#### [new 109] The More, the Merrier: Contrastive Fusion for Higher-Order Multimodal Alignment
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态学习中高阶交互建模不足的问题，提出对比融合（ConFu）框架。通过联合嵌入单模态与融合模态，引入融合对比项，同时保持成对对齐，有效捕捉如XOR类高阶依赖，提升跨模态互补性利用与任务性能。**

- **链接: [https://arxiv.org/pdf/2511.21331v1](https://arxiv.org/pdf/2511.21331v1)**

> **作者:** Stefanos Koutoupis; Michaela Areti Zervou; Konstantinos Kontras; Maarten De Vos; Panagiotis Tsakalides; Grigorios Tsagatakis
>
> **摘要:** Learning joint representations across multiple modalities remains a central challenge in multimodal machine learning. Prevailing approaches predominantly operate in pairwise settings, aligning two modalities at a time. While some recent methods aim to capture higher-order interactions among multiple modalities, they often overlook or insufficiently preserve pairwise relationships, limiting their effectiveness on single-modality tasks. In this work, we introduce Contrastive Fusion (ConFu), a framework that jointly embeds both individual modalities and their fused combinations into a unified representation space, where modalities and their fused counterparts are aligned. ConFu extends traditional pairwise contrastive objectives with an additional fused-modality contrastive term, encouraging the joint embedding of modality pairs with a third modality. This formulation enables ConFu to capture higher-order dependencies, such as XOR-like relationships, that cannot be recovered through pairwise alignment alone, while still maintaining strong pairwise correspondence. We evaluate ConFu on synthetic and real-world multimodal benchmarks, assessing its ability to exploit cross-modal complementarity, capture higher-order dependencies, and scale with increasing multimodal complexity. Across these settings, ConFu demonstrates competitive performance on retrieval and classification tasks, while supporting unified one-to-one and two-to-one retrieval within a single contrastive framework.
>
---
#### [new 110] Revolutionizing Glioma Segmentation & Grading Using 3D MRI - Guided Hybrid Deep Learning Models
- **分类: cs.CV**

- **简介: 该论文针对胶质瘤的精准诊断问题，提出一种3D MRI引导的混合深度学习框架。结合U-Net分割与注意力增强的DenseNet-VGG分类网络，实现肿瘤精准分割与分级。通过多头及空间-通道注意力机制提升关键区域识别能力，实验表明分割Dice达98%，分类准确率99%，显著优于传统方法。**

- **链接: [https://arxiv.org/pdf/2511.21673v1](https://arxiv.org/pdf/2511.21673v1)**

> **作者:** Pandiyaraju V; Sreya Mynampati; Abishek Karthik; Poovarasan L; D. Saraswathi
>
> **摘要:** Gliomas are brain tumor types that have a high mortality rate which means early and accurate diagnosis is important for therapeutic intervention for the tumors. To address this difficulty, the proposed research will develop a hybrid deep learning model which integrates U-Net based segmentation and a hybrid DenseNet-VGG classification network with multihead attention and spatial-channel attention capabilities. The segmentation model will precisely demarcate the tumors in a 3D volume of MRI data guided by spatial and contextual information. The classification network which combines a branch of both DenseNet and VGG, will incorporate the demarcated tumor on which features with attention mechanisms would be focused on clinically relevant features. High-dimensional 3D MRI data could successfully be utilized in the model through preprocessing steps which are normalization, resampling, and data augmentation. Through a variety of measures the framework is evaluated: measures of performance in segmentation are Dice coefficient and Mean Intersection over Union (IoU) and measures of performance in classification are accuracy precision, recall, and F1-score. The hybrid framework that has been proposed has demonstrated through physical testing that it has the capability of obtaining a Dice coefficient of 98% in tumor segmentation, and 99% on classification accuracy, outperforming traditional CNN models and attention-free methods. Utilizing multi-head attention mechanisms enhances notions of priority in aspects of the tumor that are clinically significant, and enhances interpretability and accuracy. The results suggest a great potential of the framework in facilitating the timely and reliable diagnosis and grading of glioma by clinicians is promising, allowing for better planning of patient treatment.
>
---
#### [new 111] When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉-语言-动作（VLA）模型的通用可转移对抗补丁攻击。针对现有攻击多为特定模型且难以跨模型迁移的问题，提出UPA-RFAS框架，通过特征空间优化、鲁棒双阶段对抗训练及专用损失函数，实现物理补丁在不同VLA模型、任务和视角间的强迁移性，揭示了VLA机器人的实际攻击风险。**

- **链接: [https://arxiv.org/pdf/2511.21192v1](https://arxiv.org/pdf/2511.21192v1)**

> **作者:** Hui Lu; Yi Yu; Yiming Yang; Chenyu Yi; Qixin Zhang; Bingquan Shen; Alex C. Kot; Xudong Jiang
>
> **摘要:** Vision-Language-Action (VLA) models are vulnerable to adversarial attacks, yet universal and transferable attacks remain underexplored, as most existing patches overfit to a single model and fail in black-box settings. To address this gap, we present a systematic study of universal, transferable adversarial patches against VLA-driven robots under unknown architectures, finetuned variants, and sim-to-real shifts. We introduce UPA-RFAS (Universal Patch Attack via Robust Feature, Attention, and Semantics), a unified framework that learns a single physical patch in a shared feature space while promoting cross-model transfer. UPA-RFAS combines (i) a feature-space objective with an $\ell_1$ deviation prior and repulsive InfoNCE loss to induce transferable representation shifts, (ii) a robustness-augmented two-phase min-max procedure where an inner loop learns invisible sample-wise perturbations and an outer loop optimizes the universal patch against this hardened neighborhood, and (iii) two VLA-specific losses: Patch Attention Dominance to hijack text$\to$vision attention and Patch Semantic Misalignment to induce image-text mismatch without labels. Experiments across diverse VLA models, manipulation suites, and physical executions show that UPA-RFAS consistently transfers across models, tasks, and viewpoints, exposing a practical patch-based attack surface and establishing a strong baseline for future defenses.
>
---
#### [new 112] Which Layer Causes Distribution Deviation? Entropy-Guided Adaptive Pruning for Diffusion and Flow Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对扩散与流模型在下游任务中参数冗余问题，提出熵引导的渐进式剪枝框架EntPruner。通过条件熵偏差度量模块重要性，实现数据依赖的自适应剪枝，有效提升推理速度并保持生成质量。**

- **链接: [https://arxiv.org/pdf/2511.21122v1](https://arxiv.org/pdf/2511.21122v1)**

> **作者:** Changlin Li; Jiawei Zhang; Zeyi Shi; Zongxin Yang; Zhihui Li; Xiaojun Chang
>
> **备注:** Project page: https://github.com/changlin31/EntPruner
>
> **摘要:** Large-scale vision generative models, including diffusion and flow models, have demonstrated remarkable performance in visual generation tasks. However, transferring these pre-trained models to downstream tasks often results in significant parameter redundancy. In this paper, we propose EntPruner, an entropy-guided automatic progressive pruning framework for diffusion and flow models. First, we introduce entropy-guided pruning, a block-level importance assessment strategy specifically designed for generative models. Unlike discriminative models, generative models require preserving the diversity and condition-fidelity of the output distribution. As the importance of each module can vary significantly across downstream tasks, EntPruner prioritizes pruning of less important blocks using data-dependent Conditional Entropy Deviation (CED) as a guiding metric. CED quantifies how much the distribution diverges from the learned conditional data distribution after removing a block. Second, we propose a zero-shot adaptive pruning framework to automatically determine when and how much to prune during training. This dynamic strategy avoids the pitfalls of one-shot pruning, mitigating mode collapse, and preserving model performance. Extensive experiments on DiT and SiT models demonstrate the effectiveness of EntPruner, achieving up to 2.22$\times$ inference speedup while maintaining competitive generation quality on ImageNet and three downstream datasets.
>
---
#### [new 113] Merge and Bound: Direct Manipulations on Weights for Class Incremental Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对类增量学习（CIL）中的灾难性遗忘问题，提出直接在参数空间操作权重的Merge-and-Bound（M&B）方法。通过任务间与任务内权重合并，并结合有界更新策略，实现新旧知识有效融合，无需修改模型结构或学习目标，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.21490v1](https://arxiv.org/pdf/2511.21490v1)**

> **作者:** Taehoon Kim; Donghwan Jang; Bohyung Han
>
> **摘要:** We present a novel training approach, named Merge-and-Bound (M&B) for Class Incremental Learning (CIL), which directly manipulates model weights in the parameter space for optimization. Our algorithm involves two types of weight merging: inter-task weight merging and intra-task weight merging. Inter-task weight merging unifies previous models by averaging the weights of models from all previous stages. On the other hand, intra-task weight merging facilitates the learning of current task by combining the model parameters within current stage. For reliable weight merging, we also propose a bounded update technique that aims to optimize the target model with minimal cumulative updates and preserve knowledge from previous tasks; this strategy reveals that it is possible to effectively obtain new models near old ones, reducing catastrophic forgetting. M&B is seamlessly integrated into existing CIL methods without modifying architecture components or revising learning objectives. We extensively evaluate our algorithm on standard CIL benchmarks and demonstrate superior performance compared to state-of-the-art methods.
>
---
#### [new 114] GuardTrace-VL: Detecting Unsafe Multimodel Reasoning via Iterative Safety Supervision
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文针对多模态大模型推理过程中的安全风险问题，提出GuardTrace-VL，通过联合图文分析监控完整问答链，检测中间推理阶段的不安全内容。构建了高质量标注数据集，并设计渐进式训练方法，显著提升安全检测性能，F1达93.1%，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.20994v1](https://arxiv.org/pdf/2511.20994v1)**

> **作者:** Yuxiao Xiang; Junchi Chen; Zhenchao Jin; Changtao Miao; Haojie Yuan; Qi Chu; Tao Gong; Nenghai Yu
>
> **摘要:** Multimodal large reasoning models (MLRMs) are increasingly deployed for vision-language tasks that produce explicit intermediate rationales. However, reasoning traces can contain unsafe content even when the final answer is non-harmful, creating deployment risks. Existing multimodal safety guards primarily evaluate only the input question and the final answer, neglecting the intermediate reasoning process. This oversight allows undetected harm, such as biased inferences or policy-violating use of visual context, to emerge during reasoning. We introduce GuardTrace-VL, a vision-aware safety auditor that monitors the full Question-Thinking-Answer (QTA) pipeline via joint image-text analysis, enabling detection of unsafe content as it emerges in the reasoning stage. To support training and evaluation, we construct the GuardTrace dataset, which is generated through diverse prompting strategies and refined via a MLRM- and human-based voting and verification pipeline. Furthermore, we propose a three-stage progressive training scheme combined with the data refinement process, enabling the model to learn nuanced and context-dependent safety preferences according to different risk levels. On our proposed test set covering both in-domain and out-of-domain scenarios, GuardTrace-VL model achieves an F1 score of 93.1% on unsafe reasoning detection tasks, representing a 13.5% improvement in F1 score compared to the previous strongest multimodal safety defense methods. The codes will be made publicly available.
>
---
#### [new 115] Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对新闻图像字幕生成任务，解决信息不全、跨模态对齐弱、视觉实体定位不准问题。提出MERGE框架，构建以实体为中心的多模态知识库，通过检索增强生成提升图文对齐与实体识别，显著优于现有方法，且在新数据集上表现 robust。**

- **链接: [https://arxiv.org/pdf/2511.21002v1](https://arxiv.org/pdf/2511.21002v1)**

> **作者:** Xiaoxing You; Qiang Huang; Lingyu Li; Chi Zhang; Xiaopeng Liu; Min Zhang; Jun Yu
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** News image captioning aims to produce journalistically informative descriptions by combining visual content with contextual cues from associated articles. Despite recent advances, existing methods struggle with three key challenges: (1) incomplete information coverage, (2) weak cross-modal alignment, and (3) suboptimal visual-entity grounding. To address these issues, we introduce MERGE, the first Multimodal Entity-aware Retrieval-augmented GEneration framework for news image captioning. MERGE constructs an entity-centric multimodal knowledge base (EMKB) that integrates textual, visual, and structured knowledge, enabling enriched background retrieval. It improves cross-modal alignment through a multistage hypothesis-caption strategy and enhances visual-entity matching via dynamic retrieval guided by image content. Extensive experiments on GoodNews and NYTimes800k show that MERGE significantly outperforms state-of-the-art baselines, with CIDEr gains of +6.84 and +1.16 in caption quality, and F1-score improvements of +4.14 and +2.64 in named entity recognition. Notably, MERGE also generalizes well to the unseen Visual News dataset, achieving +20.17 in CIDEr and +6.22 in F1-score, demonstrating strong robustness and domain adaptability.
>
---
#### [new 116] AVFakeBench: A Comprehensive Audio-Video Forgery Detection Benchmark for AV-LMMs
- **分类: cs.CV**

- **简介: 该论文针对音视频伪造检测任务，解决现有基准难以覆盖真实复杂伪造场景的问题。提出AVFakeBench，首个涵盖人类与通用主体、七类伪造和四层标注的综合性基准，包含12K高质量音视频问题，并构建多任务评估框架。通过评测11个AV-LMMs，揭示其在细粒度感知与推理上的不足。**

- **链接: [https://arxiv.org/pdf/2511.21251v1](https://arxiv.org/pdf/2511.21251v1)**

> **作者:** Shuhan Xia; Peipei Li; Xuannan Liu; Dongsen Zhang; Xinyu Guo; Zekun Li
>
> **摘要:** The threat of Audio-Video (AV) forgery is rapidly evolving beyond human-centric deepfakes to include more diverse manipulations across complex natural scenes. However, existing benchmarks are still confined to DeepFake-based forgeries and single-granularity annotations, thus failing to capture the diversity and complexity of real-world forgery scenarios. To address this, we introduce AVFakeBench, the first comprehensive audio-video forgery detection benchmark that spans rich forgery semantics across both human subject and general subject. AVFakeBench comprises 12K carefully curated audio-video questions, covering seven forgery types and four levels of annotations. To ensure high-quality and diverse forgeries, we propose a multi-stage hybrid forgery framework that integrates proprietary models for task planning with expert generative models for precise manipulation. The benchmark establishes a multi-task evaluation framework covering binary judgment, forgery types classification, forgery detail selection, and explanatory reasoning. We evaluate 11 Audio-Video Large Language Models (AV-LMMs) and 2 prevalent detection methods on AVFakeBench, demonstrating the potential of AV-LMMs as emerging forgery detectors while revealing their notable weaknesses in fine-grained perception and reasoning.
>
---
#### [new 117] CaliTex: Geometry-Calibrated Attention for View-Coherent 3D Texture Generation
- **分类: cs.CV**

- **简介: 该论文针对3D纹理生成中的视图不一致性问题，提出CaliTex框架。通过几何校准注意力机制，引入部件对齐与条件路由模块，强化几何结构与外观的耦合，实现视图一致的高质量纹理生成。**

- **链接: [https://arxiv.org/pdf/2511.21309v1](https://arxiv.org/pdf/2511.21309v1)**

> **作者:** Chenyu Liu; Hongze Chen; Jingzhi Bao; Lingting Zhu; Runze Zhang; Weikai Chen; Zeyu Hu; Yingda Yin; Keyang Luo; Xin Wang
>
> **摘要:** Despite major advances brought by diffusion-based models, current 3D texture generation systems remain hindered by cross-view inconsistency -- textures that appear convincing from one viewpoint often fail to align across others. We find that this issue arises from attention ambiguity, where unstructured full attention is applied indiscriminately across tokens and modalities, causing geometric confusion and unstable appearance-structure coupling. To address this, we introduce CaliTex, a framework of geometry-calibrated attention that explicitly aligns attention with 3D structure. It introduces two modules: Part-Aligned Attention that enforces spatial alignment across semantically matched parts, and Condition-Routed Attention which routes appearance information through geometry-conditioned pathways to maintain spatial fidelity. Coupled with a two-stage diffusion transformer, CaliTex makes geometric coherence an inherent behavior of the network rather than a byproduct of optimization. Empirically, CaliTex produces seamless and view-consistent textures and outperforms both open-source and commercial baselines.
>
---
#### [new 118] Prompt-Aware Adaptive Elastic Weight Consolidation for Continual Learning in Medical Vision-Language Models
- **分类: cs.MM; cs.CV**

- **简介: 该论文针对医疗视觉-语言模型在持续学习中面临的灾难性遗忘问题，提出Prompt-Aware Adaptive EWC方法。通过分层参数保护与自适应权重调整，有效保留跨模态医学知识，提升新任务适应能力，在多个医学影像数据集上显著降低遗忘并提升诊断性能。**

- **链接: [https://arxiv.org/pdf/2511.20732v1](https://arxiv.org/pdf/2511.20732v1)**

> **作者:** Ziyuan Gao; Philippe Morel
>
> **备注:** Accepted by 32nd International Conference on MultiMedia Modeling (MMM 2026)
>
> **摘要:** Medical AI systems face catastrophic forgetting when deployed in clinical settings, where models must learn new imaging protocols while retaining prior diagnostic capabilities. This challenge is particularly acute for medical vision-language models that must preserve complex cross-modal alignments between medical images and clinical terminology across diverse imaging modalities. We introduce Prompt- Aware Adaptive Elastic Weight Consolidation (PA-EWC), a novel continual learning approach that addresses catastrophic forgetting through prompt-guided parameter specialization. Our method systematically categorizes model parameters based on their functional roles in processing visual-descriptive, spatial-guided, and medical-semantic information, enabling targeted protection of critical knowledge while allowing adaptation to new clinical requirements. PA-EWC incorporates adaptive Fisher Information computation with gradient stability analysis and develops weighted complexity metrics based on medical terminology density. We evaluate our approach across five medical imaging datasets (Kvasir-SEG, ISIC 2018, CheXlocalize, BUSI, CAMUS) representing diverse modalities including endoscopy, dermoscopy, radiography, and ultrasound. Experimental results demonstrate that PA-EWC reduces catastrophic forgetting by up to 17.58% compared to baseline methods, with performance improvements of 4.30% on chest X-ray pathology localization and 6.06% on polyp segmentation.
>
---
#### [new 119] BanglaMM-Disaster: A Multimodal Transformer-Based Deep Learning Framework for Multiclass Disaster Classification in Bangla
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出BanglaMM-Disaster，一种用于孟加拉语多模态灾难分类的深度学习框架。针对孟加拉国灾害实时监测需求，融合文本与图像数据，构建了5037条标注数据集，采用Transformer与CNN结合的早融合模型，显著提升分类准确率，解决低资源环境下多模态灾害识别难题。**

- **链接: [https://arxiv.org/pdf/2511.21364v1](https://arxiv.org/pdf/2511.21364v1)**

> **作者:** Ariful Islam; Md Rifat Hossen; Md. Mahmudul Arif; Abdullah Al Noman; Md Arifur Rahman
>
> **备注:** Presented at the 2025 IEEE International Conference on Signal Processing, Information, Communication and Systems (SPICSCON), November 21-22, 2025, University of Rajshahi, Bangladesh. 6 pages, 9 disaster classes, multimodal dataset with 5,037 samples
>
> **摘要:** Natural disasters remain a major challenge for Bangladesh, so real-time monitoring and quick response systems are essential. In this study, we present BanglaMM-Disaster, an end-to-end deep learning-based multimodal framework for disaster classification in Bangla, using both textual and visual data from social media. We constructed a new dataset of 5,037 Bangla social media posts, each consisting of a caption and a corresponding image, annotated into one of nine disaster-related categories. The proposed model integrates transformer-based text encoders, including BanglaBERT, mBERT, and XLM-RoBERTa, with CNN backbones such as ResNet50, DenseNet169, and MobileNetV2, to process the two modalities. Using early fusion, the best model achieves 83.76% accuracy. This surpasses the best text-only baseline by 3.84% and the image-only baseline by 16.91%. Our analysis also shows reduced misclassification across all classes, with noticeable improvements for ambiguous examples. This work fills a key gap in Bangla multimodal disaster analysis and demonstrates the benefits of combining multiple data types for real-time disaster response in low-resource settings.
>
---
#### [new 120] Probabilistic Wildfire Spread Prediction Using an Autoregressive Conditional Generative Adversarial Network
- **分类: cs.LG; cs.AI; cs.CE; cs.CV**

- **简介: 该论文针对实时 wildfire 播散预测难题，提出一种自回归条件生成对抗网络（CGAN）。通过建模序列状态转移，提升长期预测稳定性与边界精度，有效捕捉火灾非线性与不确定性，优于传统深度学习模型，为应急响应提供高精度、可解释的时序预测支持。**

- **链接: [https://arxiv.org/pdf/2511.21019v1](https://arxiv.org/pdf/2511.21019v1)**

> **作者:** Taehoon Kang; Taeyong Kim
>
> **备注:** 22 pages, 15 figures, Submitted to Journal of Environmental Management
>
> **摘要:** Climate change has intensified the frequency and severity of wildfires, making rapid and accurate prediction of fire spread essential for effective mitigation and response. Physics-based simulators such as FARSITE offer high-fidelity predictions but are computationally intensive, limiting their applicability in real-time decision-making, while existing deep learning models often yield overly smooth predictions that fail to capture the complex, nonlinear dynamics of wildfire propagation. This study proposes an autoregressive conditional generative adversarial network (CGAN) for probabilistic wildfire spread prediction. By formulating the prediction task as an autoregressive problem, the model learns sequential state transitions, ensuring long-term prediction stability. Experimental results demonstrate that the proposed CGAN-based model outperforms conventional deep learning models in both overall predictive accuracy and boundary delineation of fire perimeters. These results demonstrate that adversarial learning allows the model to capture the strong nonlinearity and uncertainty of wildfire spread, instead of simply fitting the pixel average. Furthermore, the autoregressive framework facilitates systematic temporal forecasting of wildfire evolution. The proposed CGAN-based autoregressive framework enhances both the accuracy and physical interpretability of wildfire spread prediction, offering a promising foundation for time-sensitive response and evacuation planning.
>
---
#### [new 121] Automated Histopathologic Assessment of Hirschsprung Disease Using a Multi-Stage Vision Transformer Framework
- **分类: q-bio.QM; cs.CV; eess.IV**

- **简介: 该论文针对先天性巨结肠症的病理诊断，提出一种基于ViT-B/16的多阶段分割框架，依次识别肌层、肌间神经丛及神经节细胞。通过30例全切片图像训练与验证，显著提升分割精度与临床一致性，有效支持数字病理诊断，减少人为差异。**

- **链接: [https://arxiv.org/pdf/2511.20734v1](https://arxiv.org/pdf/2511.20734v1)**

> **作者:** Youssef Megahed; Saleh Abou-Alwan; Anthony Fuller; Dina El Demellawy; Steven Hawken; Adrian D. C. Chan
>
> **备注:** 16 pages, 8 figures, 6 tables
>
> **摘要:** Hirschsprung Disease is characterized by the absence of ganglion cells in the myenteric plexus. Therefore, their correct identification is crucial for diagnosing Hirschsprung disease. We introduce a three-stage segmentation framework based on a Vision Transformer (ViT-B/16) that mimics the pathologist's diagnostic approach. The framework sequentially segments the muscularis propria, delineates the myenteric plexus, and identifies ganglion cells within anatomically valid regions. 30 whole-slide images of colon tissue were used, each containing expert manual annotations of muscularis, plexus, and ganglion cells at varying levels of certainty. A 5-fold cross-validation scheme was applied to each stage, along with resolution-specific tiling strategies and tailored postprocessing to ensure anatomical consistency. The proposed method achieved a Dice coefficient of 89.9% and a Plexus Inclusion Rate of 100% for muscularis segmentation. Plexus segmentation reached a recall of 94.8%, a precision of 84.2% and a Ganglia Inclusion Rate of 99.7%. For high-certainty ganglion cells, the model achieved 62.1% precision and 89.1% recall, while joint certainty scores yielded 67.0% precision. These results indicate that ViT-based models are effective at leveraging global tissue context and capturing cellular morphology at small scales, even within complex histological tissue structures. This multi-stage methodology has great potential to support digital pathology workflows by reducing inter-observer variability and assisting in the evaluation of Hirschsprung disease. The clinical impact will be evaluated in future work with larger multi-center datasets and additional expert annotations.
>
---
#### [new 122] Bangla Sign Language Translation: Dataset Creation Challenges, Benchmarking and Prospects
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对低资源的孟加拉手语翻译（BdSLT）任务，提出数据集IsharaKhobor及两个子集，解决数据匮乏问题。通过基准测试与词汇规范处理，构建了可公开使用的高质量数据集，推动AI辅助工具发展。**

- **链接: [https://arxiv.org/pdf/2511.21533v1](https://arxiv.org/pdf/2511.21533v1)**

> **作者:** Husne Ara Rubaiyeat; Hasan Mahmud; Md Kamrul Hasan
>
> **备注:** 14 pages, 8 tables
>
> **摘要:** Bangla Sign Language Translation (BdSLT) has been severely constrained so far as the language itself is very low resource. Standard sentence level dataset creation for BdSLT is of immense importance for developing AI based assistive tools for deaf and hard of hearing people of Bangla speaking community. In this paper, we present a dataset, IsharaKhobor , and two subset of it for enabling research. We also present the challenges towards developing the dataset and present some way forward by benchmarking with landmark based raw and RQE embedding. We do some ablation on vocabulary restriction and canonicalization of the same within the dataset, which resulted in two more datasets, IsharaKhobor_small and IsharaKhobor_canonical_small. The dataset is publicly available at: www.kaggle.com/datasets/hasanssl/isharakhobor [1].
>
---
#### [new 123] Deep Parameter Interpolation for Scalar Conditioning
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出深度参数插值（DPI），用于在不改变网络结构的前提下，使神经网络接受额外的标量输入。针对生成模型中时间或噪声水平依赖向量场建模困难的问题，DPI通过动态插值两组可学习参数实现标量条件化，提升去噪性能与样本质量，兼具高效性与通用性。**

- **链接: [https://arxiv.org/pdf/2511.21028v1](https://arxiv.org/pdf/2511.21028v1)**

> **作者:** Chicago Y. Park; Michael T. McCann; Cristina Garcia-Cardona; Brendt Wohlberg; Ulugbek S. Kamilov
>
> **摘要:** We propose deep parameter interpolation (DPI), a general-purpose method for transforming an existing deep neural network architecture into one that accepts an additional scalar input. Recent deep generative models, including diffusion models and flow matching, employ a single neural network to learn a time- or noise level-dependent vector field. Designing a network architecture to accurately represent this vector field is challenging because the network must integrate information from two different sources: a high-dimensional vector (usually an image) and a scalar. Common approaches either encode the scalar as an additional image input or combine scalar and vector information in specific network components, which restricts architecture choices. Instead, we propose to maintain two learnable parameter sets within a single network and to introduce the scalar dependency by dynamically interpolating between the parameter sets based on the scalar value during training and sampling. DPI is a simple, architecture-agnostic method for adding scalar dependence to a neural network. We demonstrate that our method improves denoising performance and enhances sample quality for both diffusion and flow matching models, while achieving computational efficiency comparable to standard scalar conditioning techniques. Code is available at https://github.com/wustl-cig/parameter_interpolation.
>
---
#### [new 124] Adversarial Multi-Task Learning for Liver Tumor Segmentation, Dynamic Enhancement Regression, and Classification
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出MTI-Net框架，旨在同时完成肝肿瘤分割、动态增强回归和分类任务。针对多任务间关联性弱及动态MRI信息提取困难的问题，引入熵融合特征、任务交互模块与任务驱动判别器，提升跨任务协同与动态数据建模能力，实现在238例数据上的优异性能。**

- **链接: [https://arxiv.org/pdf/2511.20793v1](https://arxiv.org/pdf/2511.20793v1)**

> **作者:** Xiaojiao Xiao; Qinmin Vivian Hu; Tae Hyun Kim; Guanghui Wang
>
> **摘要:** Liver tumor segmentation, dynamic enhancement regression, and classification are critical for clinical assessment and diagnosis. However, no prior work has attempted to achieve these tasks simultaneously in an end-to-end framework, primarily due to the lack of an effective framework that captures inter-task relevance for mutual improvement and the absence of a mechanism to extract dynamic MRI information effectively. To address these challenges, we propose the Multi-Task Interaction adversarial learning Network (MTI-Net), a novel integrated framework designed to tackle these tasks simultaneously. MTI-Net incorporates Multi-domain Information Entropy Fusion (MdIEF), which utilizes entropy-aware, high-frequency spectral information to effectively integrate features from both frequency and spectral domains, enhancing the extraction and utilization of dynamic MRI data. The network also introduces a task interaction module that establishes higher-order consistency between segmentation and regression, thus fostering inter-task synergy and improving overall performance. Additionally, we designed a novel task-driven discriminator (TDD) to capture internal high-order relationships between tasks. For dynamic MRI information extraction, we employ a shallow Transformer network to perform positional encoding, which captures the relationships within dynamic MRI sequences. In experiments on a dataset of 238 subjects, MTI-Net demonstrates high performance across multiple tasks, indicating its strong potential for assisting in the clinical assessment of liver tumors. The code is available at: https://github.com/xiaojiao929/MTI-Net.
>
---
#### [new 125] CNN-LSTM Hybrid Architecture for Over-the-Air Automatic Modulation Classification Using SDR
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对无线通信中的自动调制识别（AMC）任务，提出一种基于CNN-LSTM混合架构的系统，结合软件定义无线电平台，有效处理时变信号。通过融合公开与自建数据集，在0–30dB SNR下实现93.48%准确率，验证了其在复杂环境下的鲁棒性与实用性。**

- **链接: [https://arxiv.org/pdf/2511.21040v1](https://arxiv.org/pdf/2511.21040v1)**

> **作者:** Dinanath Padhya; Krishna Acharya; Bipul Kumar Dahal; Dinesh Baniya Kshatri
>
> **备注:** 8 Pages, 10 figures, 2 Tables, Accepted in Journal (Journal of Innovations in Engineering Education), Issue is not Published Yet
>
> **摘要:** Automatic Modulation Classification (AMC) is a core technology for future wireless communication systems, enabling the identification of modulation schemes without prior knowledge. This capability is essential for applications in cognitive radio, spectrum monitoring, and intelligent communication networks. We propose an AMC system based on a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) architecture, integrated with a Software Defined Radio (SDR) platform. The proposed architecture leverages CNNs for spatial feature extraction and LSTMs for capturing temporal dependencies, enabling efficient handling of complex, time-varying communication signals. The system's practical ability was demonstrated by identifying over-the-air (OTA) signals from a custom-built FM transmitter alongside other modulation schemes. The system was trained on a hybrid dataset combining the RadioML2018 dataset with a custom-generated dataset, featuring samples at Signal-to-Noise Ratios (SNRs) from 0 to 30dB. System performance was evaluated using accuracy, precision, recall, F1 score, and the Area Under the Receiver Operating Characteristic Curve (AUC-ROC). The optimized model achieved 93.48% accuracy, 93.53% precision, 93.48% recall, and an F1 score of 93.45%. The AUC-ROC analysis confirmed the model's discriminative power, even in noisy conditions. This paper's experimental results validate the effectiveness of the hybrid CNN-LSTM architecture for AMC, suggesting its potential application in adaptive spectrum management and advanced cognitive radio systems.
>
---
#### [new 126] Multi-Reward GRPO for Stable and Prosodic Single-Codebook TTS LLMs at Scale
- **分类: cs.SD; cs.CV**

- **简介: 该论文研究单码本文本到语音大模型的稳定性问题。针对其存在的语调不稳、说话人漂移和自然度下降问题，提出多奖励分组相对策略优化框架，融合长度惩罚、熵正则与基于大模型推理的韵律对齐奖励，显著提升语音质量与稳定性，并验证了方法在不同规模下的有效性。**

- **链接: [https://arxiv.org/pdf/2511.21270v1](https://arxiv.org/pdf/2511.21270v1)**

> **作者:** Yicheng Zhong; Peiji Yang; Zhisheng Wang
>
> **备注:** 4 pages, 2 figures
>
> **摘要:** Recent advances in Large Language Models (LLMs) have transformed text-to-speech (TTS) synthesis, inspiring autoregressive frameworks that represent speech as sequences of discrete codec tokens. Among them, single-codebook TTS LLMs have emerged as compact and streamable architectures that jointly model semantic and acoustic integration. However, despite their efficiency, these models often exhibit unstable prosody, speaker drift, and degraded naturalness. To address these issues, we propose a multi-reward Group Relative Policy Optimization (GRPO) framework that directly optimizes the token generation policy of single-codebook TTS LLMs. Beyond standard intelligibility and speaker similarity objectives, our design integrates three rule-based rewards: a length penalty for duration consistency, an entropy regularization reward for decoding stability, and an LLM-annotated prosody alignment reward that explicitly supervises rhythm. In this prosody reward, an external reasoning LLM predicts multiple plausible pause structures via in-context learning, providing a human-preference-aligned supervisory signal for GRPO training. To assess universality, we further attach a flow-matching (FM) decoder on top of the GRPO-optimized AR backbone and observe consistent additional gains, indicating that our reinforcement optimization enhances the intrinsic AR policy. We further conduct a scalability analysis across data sizes and model scales, revealing that the proposed method consistently enhances prosodic stability, speaker similarity, and overall speech naturalness in single-codebook TTS LLMs.
>
---
#### [new 127] Guaranteed Optimal Compositional Explanations for Neurons
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对深度神经网络中神经元的可解释性问题，提出首个能保证最优性的组合解释框架。通过分解空间对齐因素、设计启发式估计算法，实现高效精确的逻辑规则搜索。在计算机视觉领域验证了现有方法存在10-40%子优解，并证明新方法在效率与灵活性上优于传统束搜索。**

- **链接: [https://arxiv.org/pdf/2511.20934v1](https://arxiv.org/pdf/2511.20934v1)**

> **作者:** Biagio La Rosa; Leilani H. Gilpin
>
> **备注:** 41 pages, 10 figures
>
> **摘要:** While neurons are the basic units of deep neural networks, it is still unclear what they learn and if their knowledge is aligned with that of humans. Compositional explanations aim to answer this question by describing the spatial alignment between neuron activations and concepts through logical rules. These logical descriptions are typically computed via a search over all possible concept combinations. Since computing the spatial alignment over the entire state space is computationally infeasible, the literature commonly adopts beam search to restrict the space. However, beam search cannot provide any theoretical guarantees of optimality, and it remains unclear how close current explanations are to the true optimum. In this theoretical paper, we address this gap by introducing the first framework for computing guaranteed optimal compositional explanations. Specifically, we propose: (i) a decomposition that identifies the factors influencing the spatial alignment, (ii) a heuristic to estimate the alignment at any stage of the search, and (iii) the first algorithm that can compute optimal compositional explanations within a feasible time. Using this framework, we analyze the differences between optimal and non-optimal explanations in the most popular settings for compositional explanations, the computer vision domain and Convolutional Neural Networks. In these settings, we demonstrate that 10-40 percent of explanations obtained with beam search are suboptimal when overlapping concepts are involved. Finally, we evaluate a beam-search variant guided by our proposed decomposition and heuristic, showing that it matches or improves runtime over prior methods while offering greater flexibility in hyperparameters and computational resources.
>
---
#### [new 128] AV-Edit: Multimodal Generative Sound Effect Editing via Audio-Visual Semantic Joint Control
- **分类: cs.MM; cs.CV; cs.SD**

- **简介: 该论文提出AV-Edit，解决音效编辑中依赖低层信号或粗略文本导致灵活性差的问题。通过联合视觉、音频与文本语义，利用对比音频-视觉掩码自编码器预训练，并引入相关性特征门控的多模态扩散变换器，实现精准音效编辑。构建专用视频音效数据集进行评估，显著提升音效编辑质量与一致性。**

- **链接: [https://arxiv.org/pdf/2511.21146v1](https://arxiv.org/pdf/2511.21146v1)**

> **作者:** Xinyue Guo; Xiaoran Yang; Lipan Zhang; Jianxuan Yang; Zhao Wang; Jian Luan
>
> **摘要:** Sound effect editing-modifying audio by adding, removing, or replacing elements-remains constrained by existing approaches that rely solely on low-level signal processing or coarse text prompts, often resulting in limited flexibility and suboptimal audio quality. To address this, we propose AV-Edit, a generative sound effect editing framework that enables fine-grained editing of existing audio tracks in videos by jointly leveraging visual, audio, and text semantics. Specifically, the proposed method employs a specially designed contrastive audio-visual masking autoencoder (CAV-MAE-Edit) for multimodal pre-training, learning aligned cross-modal representations. These representations are then used to train an editorial Multimodal Diffusion Transformer (MM-DiT) capable of removing visually irrelevant sounds and generating missing audio elements consistent with video content through a correlation-based feature gating training strategy. Furthermore, we construct a dedicated video-based sound editing dataset as an evaluation benchmark. Experiments demonstrate that the proposed AV-Edit generates high-quality audio with precise modifications based on visual content, achieving state-of-the-art performance in the field of sound effect editing and exhibiting strong competitiveness in the domain of audio generation.
>
---
#### [new 129] SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对遵循社会规范的具身导航任务，提出SocialNav基础模型。通过构建大规模数据集与分阶段训练，融合社会推理与导航技能，实现高成功率与高社会合规性。**

- **链接: [https://arxiv.org/pdf/2511.21135v1](https://arxiv.org/pdf/2511.21135v1)**

> **作者:** Ziyi Chen; Yingnan Guo; Zedong Chu; Minghua Luo; Yanfen Shen; Mingchao Sun; Junjun Hu; Shichao Xie; Kuan Yang; Pei Shi; Zhining Gu; Lu Liu; Honglin Han; Xiaolong Wu; Mu Xu; Yu Zhang
>
> **摘要:** Embodied navigation that adheres to social norms remains an open research challenge. Our \textbf{SocialNav} is a foundational model for socially-aware navigation with a hierarchical "brain-action" architecture, capable of understanding high-level social norms and generating low-level, socially compliant trajectories. To enable such dual capabilities, we construct the SocNav Dataset, a large-scale collection of 7 million samples, comprising (1) a Cognitive Activation Dataset providing social reasoning signals such as chain-of-thought explanations and social traversability prediction, and (2) an Expert Trajectories Pyramid aggregating diverse navigation demonstrations from internet videos, simulated environments, and real-world robots. A multi-stage training pipeline is proposed to gradually inject and refine navigation intelligence: we first inject general navigation skills and social norms understanding into the model via imitation learning, and then refine such skills through a deliberately designed Socially-Aware Flow Exploration GRPO (SAFE-GRPO), the first flow-based reinforcement learning framework for embodied navigation that explicitly rewards socially compliant behaviors. SocialNav achieves +38% success rate and +46% social compliance rate compared to the state-of-the-art method, demonstrating strong gains in both navigation performance and social compliance. Our project page: https://amap-eai.github.io/SocialNav/
>
---
#### [new 130] Uncertainty Quantification for Visual Object Pose Estimation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对单目视觉目标位姿估计中的不确定性量化问题，提出无需分布假设的分布自由不确定性边界方法。通过像素检测噪声约束，构建非凸不确定性集，并基于S-lemma设计凸优化算法SLUE，得到高概率包含真实位姿的椭球边界。可有效分解为平移与姿态独立边界，实验验证其在翻译不确定性上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.21666v1](https://arxiv.org/pdf/2511.21666v1)**

> **作者:** Lorenzo Shaikewitz; Charis Georgiou; Luca Carlone
>
> **备注:** 18 pages, 9 figures. Code available: https://github.com/MIT-SPARK/PoseUncertaintySets
>
> **摘要:** Quantifying the uncertainty of an object's pose estimate is essential for robust control and planning. Although pose estimation is a well-studied robotics problem, attaching statistically rigorous uncertainty is not well understood without strict distributional assumptions. We develop distribution-free pose uncertainty bounds about a given pose estimate in the monocular setting. Our pose uncertainty only requires high probability noise bounds on pixel detections of 2D semantic keypoints on a known object. This noise model induces an implicit, non-convex set of pose uncertainty constraints. Our key contribution is SLUE (S-Lemma Uncertainty Estimation), a convex program to reduce this set to a single ellipsoidal uncertainty bound that is guaranteed to contain the true object pose with high probability. SLUE solves a relaxation of the minimum volume bounding ellipsoid problem inspired by the celebrated S-lemma. It requires no initial guess of the bound's shape or size and is guaranteed to contain the true object pose with high probability. For tighter uncertainty bounds at the same confidence, we extend SLUE to a sum-of-squares relaxation hierarchy which is guaranteed to converge to the minimum volume ellipsoidal uncertainty bound for a given set of keypoint constraints. We show this pose uncertainty bound can easily be projected to independent translation and axis-angle orientation bounds. We evaluate SLUE on two pose estimation datasets and a real-world drone tracking scenario. Compared to prior work, SLUE generates substantially smaller translation bounds and competitive orientation bounds. We release code at https://github.com/MIT-SPARK/PoseUncertaintySets.
>
---
#### [new 131] ENACT: Evaluating Embodied Cognition with World Modeling of Egocentric Interaction
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文提出ENACT基准，评估视觉语言模型是否具备具身认知能力。通过双任务（正向与逆向世界建模）在部分可观测的视角交互中检验模型对动作-效应、空间意识及长时记忆的理解。基于机器人仿真生成数据，揭示当前模型在长交互序列中显著落后于人类，且存在人为偏见。**

- **链接: [https://arxiv.org/pdf/2511.20937v1](https://arxiv.org/pdf/2511.20937v1)**

> **作者:** Qineng Wang; Wenlong Huang; Yu Zhou; Hang Yin; Tianwei Bao; Jianwen Lyu; Weiyu Liu; Ruohan Zhang; Jiajun Wu; Li Fei-Fei; Manling Li
>
> **备注:** Preprint version
>
> **摘要:** Embodied cognition argues that intelligence arises from sensorimotor interaction rather than passive observation. It raises an intriguing question: do modern vision-language models (VLMs), trained largely in a disembodied manner, exhibit signs of embodied cognition? We introduce ENACT, a benchmark that casts evaluation of embodied cognition as world modeling from egocentric interaction in a visual question answering (VQA) format. Framed as a partially observable Markov decision process (POMDP) whose actions are scene graph changes, ENACT comprises two complementary sequence reordering tasks: forward world modeling (reorder shuffled observations given actions) and inverse world modeling (reorder shuffled actions given observations). While conceptually simple, solving these tasks implicitly demands capabilities central to embodied cognition-affordance recognition, action-effect reasoning, embodied awareness, and interactive, long-horizon memory from partially observable egocentric input, while avoiding low-level image synthesis that could confound the evaluation. We provide a scalable pipeline that synthesizes QA pairs from robotics simulation (BEHAVIOR) and evaluates models on 8,972 QA pairs spanning long-horizon home-scale activities. Experiments reveal a performance gap between frontier VLMs and humans that widens with interaction horizon. Models consistently perform better on the inverse task than the forward one and exhibit anthropocentric biases, including a preference for right-handed actions and degradation when camera intrinsics or viewpoints deviate from human vision. Website at https://enact-embodied-cognition.github.io/.
>
---
#### [new 132] AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出AerialMind，首个面向无人机场景的指代多目标跟踪（RMOT）基准。针对现有研究局限于地面场景、难以捕捉大范围上下文的问题，构建了大规模数据集并提出COALA标注框架降低人力成本。同时设计HETrack方法，提升视觉-语言联合表征能力，增强无人机环境下的目标感知与跟踪性能。**

- **链接: [https://arxiv.org/pdf/2511.21053v1](https://arxiv.org/pdf/2511.21053v1)**

> **作者:** Chenglizhao Chen; Shaofeng Liang; Runwei Guan; Xiaolou Sun; Haocheng Zhao; Haiyun Jiang; Tao Huang; Henghui Ding; Qing-Long Han
>
> **备注:** AAAI 2026
>
> **摘要:** Referring Multi-Object Tracking (RMOT) aims to achieve precise object detection and tracking through natural language instructions, representing a fundamental capability for intelligent robotic systems. However, current RMOT research remains mostly confined to ground-level scenarios, which constrains their ability to capture broad-scale scene contexts and perform comprehensive tracking and path planning. In contrast, Unmanned Aerial Vehicles (UAVs) leverage their expansive aerial perspectives and superior maneuverability to enable wide-area surveillance. Moreover, UAVs have emerged as critical platforms for Embodied Intelligence, which has given rise to an unprecedented demand for intelligent aerial systems capable of natural language interaction. To this end, we introduce AerialMind, the first large-scale RMOT benchmark in UAV scenarios, which aims to bridge this research gap. To facilitate its construction, we develop an innovative semi-automated collaborative agent-based labeling assistant (COALA) framework that significantly reduces labor costs while maintaining annotation quality. Furthermore, we propose HawkEyeTrack (HETrack), a novel method that collaboratively enhances vision-language representation learning and improves the perception of UAV scenarios. Comprehensive experiments validated the challenging nature of our dataset and the effectiveness of our method.
>
---
#### [new 133] A Fractional Variational Approach to Spectral Filtering Using the Fourier Transform
- **分类: eess.IV; cs.CV; math-ph**

- **简介: 该论文针对拉曼光谱中荧光干扰与噪声掩盖关键化学特征的问题，提出一种基于傅里叶变换的分数阶变分滤波方法。通过在频域优化正则化参数与分数阶导数阶次，平衡去噪与特征保留，实现高效、鲁棒的谱图处理。**

- **链接: [https://arxiv.org/pdf/2511.20675v1](https://arxiv.org/pdf/2511.20675v1)**

> **作者:** Nelson H. T. Lemes; José Claudinei Ferreira; Higor V. M. Ferreira
>
> **备注:** 31 pages, 3 figures, 2 tables
>
> **摘要:** The interference of fluorescence signals and noise remains a significant challenge in Raman spectrum analysis, often obscuring subtle spectral features that are critical for accurate analysis. Inspired by variational methods similar to those used in image denoising, our approach minimizes a functional involving fractional derivatives to balance noise suppression with the preservation of essential chemical features of the signal, such as peak position, intensity, and area. The original problem is reformulated in the frequency domain through the Fourier transform, making the implementation simple and fast. In this work, we discuss the theoretical framework, practical implementation, and the advantages and limitations of this method in the context of {simulated} Raman data, as well as in image processing. The main contribution of this article is the combination of a variational approach in the frequency domain, the use of fractional derivatives, and the optimization of the {regularization parameter and} derivative order through the concept of Shannon entropy. This work explores how the fractional order, combined with the regularization parameter, affects noise removal and preserves the essential features of the spectrum {and image}. Finally, the study shows that the combination of the proposed strategies produces an efficient, robust, and easily implementable filter.
>
---
#### [new 134] TraceGen: World Modeling in 3D Trace Space Enables Learning from Cross-Embodiment Videos
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文针对机器人在新平台、新场景下仅凭少量示范学习的难题，提出TraceGen世界模型。通过构建统一的3D轨迹空间表示，实现跨主体、跨环境视频的高效利用。基于123K视频数据预训练，仅需5段目标视频即可达80%成功率，显著提升泛化与推理效率。**

- **链接: [https://arxiv.org/pdf/2511.21690v1](https://arxiv.org/pdf/2511.21690v1)**

> **作者:** Seungjae Lee; Yoonkyo Jung; Inkook Chun; Yao-Chih Lee; Zikui Cai; Hongjia Huang; Aayush Talreja; Tan Dat Dao; Yongyuan Liang; Jia-Bin Huang; Furong Huang
>
> **摘要:** Learning new robot tasks on new platforms and in new scenes from only a handful of demonstrations remains challenging. While videos of other embodiments - humans and different robots - are abundant, differences in embodiment, camera, and environment hinder their direct use. We address the small-data problem by introducing a unifying, symbolic representation - a compact 3D "trace-space" of scene-level trajectories - that enables learning from cross-embodiment, cross-environment, and cross-task videos. We present TraceGen, a world model that predicts future motion in trace-space rather than pixel space, abstracting away appearance while retaining the geometric structure needed for manipulation. To train TraceGen at scale, we develop TraceForge, a data pipeline that transforms heterogeneous human and robot videos into consistent 3D traces, yielding a corpus of 123K videos and 1.8M observation-trace-language triplets. Pretraining on this corpus produces a transferable 3D motion prior that adapts efficiently: with just five target robot videos, TraceGen attains 80% success across four tasks while offering 50-600x faster inference than state-of-the-art video-based world models. In the more challenging case where only five uncalibrated human demonstration videos captured on a handheld phone are available, it still reaches 67.5% success on a real robot, highlighting TraceGen's ability to adapt across embodiments without relying on object detectors or heavy pixel-space generation.
>
---
#### [new 135] STAR: Smartphone-analogous Typing in Augmented Reality
- **分类: cs.HC; cs.CV**

- **简介: 该论文提出STAR，一种类智能手机双指输入的增强现实文本输入方法。针对AR中高效易用文本输入难的问题，通过在手部皮肤上显示虚拟键盘，利用用户对手机打字的熟悉度。实验表明，用户经30分钟练习后可达到21.9 WPM的输入速度，错误率仅0.3%。**

- **链接: [https://arxiv.org/pdf/2511.21143v1](https://arxiv.org/pdf/2511.21143v1)**

> **作者:** Taejun Kim; Amy Karlson; Aakar Gupta; Tovi Grossman; Jason Wu; Parastoo Abtahi; Christopher Collins; Michael Glueck; Hemant Bhaskar Surale
>
> **备注:** ACM UIST 2023
>
> **摘要:** While text entry is an essential and frequent task in Augmented Reality (AR) applications, devising an efficient and easy-to-use text entry method for AR remains an open challenge. This research presents STAR, a smartphone-analogous AR text entry technique that leverages a user's familiarity with smartphone two-thumb typing. With STAR, a user performs thumb typing on a virtual QWERTY keyboard that is overlain on the skin of their hands. During an evaluation study of STAR, participants achieved a mean typing speed of 21.9 WPM (i.e., 56% of their smartphone typing speed), and a mean error rate of 0.3% after 30 minutes of practice. We further analyze the major factors implicated in the performance gap between STAR and smartphone typing, and discuss ways this gap could be narrowed.
>
---
#### [new 136] Mechanisms of Non-Monotonic Scaling in Vision Transformers
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究视觉Transformer深度扩展中的非单调性能下降问题。通过分析ViT-S/B/L在ImageNet上的表现，发现存在“悬崖-平台-爬升”三阶段表征演化模式，揭示[CLS] token被弱化、patch tokens协同增强的现象，并提出信息混淆指数量化信息混合，指出额外深度主要促进信息扩散而非任务性能提升，建议以相变调控替代盲目增深。**

- **链接: [https://arxiv.org/pdf/2511.21635v1](https://arxiv.org/pdf/2511.21635v1)**

> **作者:** Anantha Padmanaban Krishna Kumar
>
> **备注:** 16 pages total (11 pages main text, 1 pages references, 4 pages appendix), 5 figures, 11 tables. Code available at https://github.com/AnanthaPadmanaban-KrishnaKumar/Cliff-Plateau-Climb
>
> **摘要:** Deeper Vision Transformers often perform worse than shallower ones, which challenges common scaling assumptions. Through a systematic empirical analysis of ViT-S, ViT-B, and ViT-L on ImageNet, we identify a consistent three-phase Cliff-Plateau-Climb pattern that governs how representations evolve with depth. We observe that better performance is associated with progressive marginalization of the [CLS] token, originally designed as a global aggregation hub, in favor of distributed consensus among patch tokens. We quantify patterns of information mixing with an Information Scrambling Index, and show that in ViT-L the information-task tradeoff emerges roughly 10 layers later than in ViT-B, and that these additional layers correlate with increased information diffusion rather than improved task performance. Taken together, these results suggest that transformer architectures in this regime may benefit more from carefully calibrated depth that executes clean phase transitions than from simply increasing parameter count. The Information Scrambling Index provides a useful diagnostic for existing models and suggests a potential design target for future architectures. All code is available at: https://github.com/AnanthaPadmanaban-KrishnaKumar/Cliff-Plateau-Climb.
>
---
#### [new 137] CHiQPM: Calibrated Hierarchical Interpretable Image Classification
- **分类: cs.LG; cs.CV; cs.HC**

- **简介: 该论文提出CHiQPM模型，解决图像分类中可解释性与准确率的平衡问题。通过层次化对比解释实现全局与局部可解释性，结合校准的置信集预测，提升人类与AI协作效率，兼具高精度与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.20779v1](https://arxiv.org/pdf/2511.20779v1)**

> **作者:** Thomas Norrenbrock; Timo Kaiser; Sovan Biswas; Neslihan Kose; Ramesh Manuvinakurike; Bodo Rosenhahn
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Globally interpretable models are a promising approach for trustworthy AI in safety-critical domains. Alongside global explanations, detailed local explanations are a crucial complement to effectively support human experts during inference. This work proposes the Calibrated Hierarchical QPM (CHiQPM) which offers uniquely comprehensive global and local interpretability, paving the way for human-AI complementarity. CHiQPM achieves superior global interpretability by contrastively explaining the majority of classes and offers novel hierarchical explanations that are more similar to how humans reason and can be traversed to offer a built-in interpretable Conformal prediction (CP) method. Our comprehensive evaluation shows that CHiQPM achieves state-of-the-art accuracy as a point predictor, maintaining 99% accuracy of non-interpretable models. This demonstrates a substantial improvement, where interpretability is incorporated without sacrificing overall accuracy. Furthermore, its calibrated set prediction is competitively efficient to other CP methods, while providing interpretable predictions of coherent sets along its hierarchical explanation.
>
---
#### [new 138] OVOD-Agent: A Markov-Bandit Framework for Proactive Visual Reasoning and Self-Evolving Detection
- **分类: cs.AI; cs.CV**

- **简介: 该论文针对开放词汇目标检测（OVOD）中类别泛化能力不足的问题，提出OVOD-Agent框架。通过将被动匹配转为主动视觉推理，构建基于弱马尔可夫决策过程与强化学习的自进化检测机制，实现对稀有类别的有效识别，显著提升检测性能。**

- **链接: [https://arxiv.org/pdf/2511.21064v1](https://arxiv.org/pdf/2511.21064v1)**

> **作者:** Chujie Wang; Jianyu Lu; Zhiyuan Luo; Xi Chen; Chu He
>
> **摘要:** Open-Vocabulary Object Detection (OVOD) aims to enable detectors to generalize across categories by leveraging semantic information. Although existing methods are pretrained on large vision-language datasets, their inference is still limited to fixed category names, creating a gap between multimodal training and unimodal inference. Previous work has shown that improving textual representation can significantly enhance OVOD performance, indicating that the textual space is still underexplored. To this end, we propose OVOD-Agent, which transforms passive category matching into proactive visual reasoning and self-evolving detection. Inspired by the Chain-of-Thought (CoT) paradigm, OVOD-Agent extends the textual optimization process into an interpretable Visual-CoT with explicit actions. OVOD's lightweight nature makes LLM-based management unsuitable; instead, we model visual context transitions as a Weakly Markovian Decision Process (w-MDP) over eight state spaces, which naturally represents the agent's state, memory, and interaction dynamics. A Bandit module generates exploration signals under limited supervision, helping the agent focus on uncertain regions and adapt its detection policy. We further integrate Markov transition matrices with Bandit trajectories for self-supervised Reward Model (RM) optimization, forming a closed loop from Bandit exploration to RM learning. Experiments on COCO and LVIS show that OVOD-Agent provides consistent improvements across OVOD backbones, particularly on rare categories, confirming the effectiveness of the proposed framework.
>
---
## 更新

#### [replaced 001] LogicOCR: Do Your Large Multimodal Models Excel at Logical Reasoning on Text-Rich Images?
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对大模型在图文逻辑推理任务中的表现不足，提出LogicOCR基准，包含生成与真实图像的2780个问题。通过构建高质量数据集并评估多种模型，发现其视觉阅读与推理能力仍落后于纯文本输入。为此提出TextCue方法，利用注意力图增强关键文本区域感知，显著提升推理准确率。**

- **链接: [https://arxiv.org/pdf/2505.12307v2](https://arxiv.org/pdf/2505.12307v2)**

> **作者:** Maoyuan Ye; Haibin He; Qihuang Zhong; Jing Zhang; Juhua Liu; Bo Du
>
> **备注:** GitHub: https://github.com/MiliLab/LogicOCR
>
> **摘要:** Recent advances in Large Multimodal Models (LMMs) have revolutionized their reasoning and Optical Character Recognition (OCR) capabilities. However, their complex logical reasoning performance on text-rich images remains underexplored. To bridge this gap, we introduce LogicOCR, a benchmark comprising 2780 questions with two subsets, i.e., LogicOCR-Gen with 1100 multi-choice questions on generated images, and LogicOCR-Real with 1680 meticulously designed free-form questions on real-world images. For constructing LogicOCR-Gen, we first curate a text corpus from the Chinese National Civil Servant Examination, and customize an automatic pipeline to steer GPT-Image-1 to generate images with varied layouts and fonts, ensuring contextual relevance and visual realism. Then, the generated images are manually verified. We evaluate a range of representative LMMs under Chain-of-Thought (CoT) and direct-answer settings. Our multi-dimensional analysis reveals key insights, such as the impact of test-time scaling, input modality differences, and sensitivity to visual-text orientation. Notably, LMMs still lag in multimodal reasoning compared to text-only inputs, indicating that they have not fully bridged visual reading with reasoning. Moreover, we propose TextCue, a training-free method that enhances LMMs' perception of image regions containing important text cues for solving questions. We leverage LMMs' attention maps and an off-the-shelf text segmentation specialist to determine the region, which is then cropped and enlarged to augment the original image. Experiments show its effectiveness, e.g., a 1.8% accuracy gain over LLaVA-OV-1.5-8B under the CoT setting. Our benchmark is available at https://github.com/MiliLab/LogicOCR.
>
---
#### [replaced 002] CroMe: Multimodal Fake News Detection using Cross-Modal Tri-Transformer and Metric Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2501.12422v2](https://arxiv.org/pdf/2501.12422v2)**

> **作者:** Eunjee Choi; Junhyun Ahn; XinYu Piao; Jong-Kook Kim
>
> **摘要:** Multimodal Fake News Detection has received increasing attention recently. Existing methods rely on independently encoded unimodal data and overlook the advantages of capturing intra-modality relationships and integrating inter-modal similarities using advanced techniques. To address these issues, Cross-Modal Tri-Transformer and Metric Learning for Multimodal Fake News Detection (CroMe) is proposed. CroMe utilizes Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models (BLIP2) as encoders to capture detailed text, image and combined image-text representations. The metric learning module employs a proxy anchor method to capture intra-modality relationships while the feature fusion module uses a Cross-Modal and Tri-Transformer for effective integration. The final fake news detector processes the fused features through a classifier to predict the authenticity of the content. Experiments on datasets show that CroMe excels in multimodal fake news detection.
>
---
#### [replaced 003] MetricHMSR:Metric Human Mesh and Scene Recovery from Monocular Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.09919v2](https://arxiv.org/pdf/2506.09919v2)**

> **作者:** Chentao Song; He Zhang; Haolei Yuan; Haozhe Lin; Jianhua Tao; Hongwen Zhang; Tao Yu
>
> **摘要:** We introduce MetricHMSR (Metric Human Mesh and Scene Recovery), a novel approach for metric human mesh and scene recovery from monocular images. Due to unrealistic assumptions in the camera model and inherent challenges in metric perception, existing approaches struggle to achieve human pose and metric 3D position estimation through a unified module. To address this limitation, MetricHMSR incorporates camera rays to comprehensively encode both the bounding box information and the intrinsic parameters of perspective projection. Then we proposed Human Mixture-of-Experts (MoE), the model dynamically routes image features and ray features to task-specific experts for specialized understanding of different data aspects, enabling a unified framework that simultaneously perceives the local pose and the global 3D position. Based on the results above, we further refine the existing monocular metric depth estimation method to achieve more accurate results, ultimately enabling the seamless overlay of humans and scenes in 3D space. Comprehensive experimental results demonstrate that the proposed method achieves state-of-the-art performance on both human mesh and scene recovery.
>
---
#### [replaced 004] GFT-GCN: Privacy-Preserving 3D Face Mesh Recognition with Spectral Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19958v2](https://arxiv.org/pdf/2511.19958v2)**

> **作者:** Hichem Felouat; Hanrui Wang; Isao Echizen
>
> **备注:** 13 pages, 8 figures, WACV 2026
>
> **摘要:** 3D face recognition offers a robust biometric solution by capturing facial geometry, providing resilience to variations in illumination, pose changes, and presentation attacks. Its strong spoof resistance makes it suitable for high-security applications, but protecting stored biometric templates remains critical. We present GFT-GCN, a privacy-preserving 3D face recognition framework that combines spectral graph learning with diffusion-based template protection. Our approach integrates the Graph Fourier Transform (GFT) and Graph Convolutional Networks (GCN) to extract compact, discriminative spectral features from 3D face meshes. To secure these features, we introduce a spectral diffusion mechanism that produces irreversible, renewable, and unlinkable templates. A lightweight client-server architecture ensures that raw biometric data never leaves the client device. Experiments on the BU-3DFE and FaceScape datasets demonstrate high recognition accuracy and strong resistance to reconstruction attacks. Results show that GFT-GCN effectively balances privacy and performance, offering a practical solution for secure 3D face authentication.
>
---
#### [replaced 005] Earth-Adapter: Bridge the Geospatial Domain Gaps with Mixture of Frequency Adaptation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.06220v4](https://arxiv.org/pdf/2504.06220v4)**

> **作者:** Xiaoxing Hu; Ziyang Gong; Yupei Wang; Yuru Jia; Fei Lin; Dexiang Gao; Ke An; Jianhong Han; Zhuoran Sun; Gen Luo; Gen Luo; Xue Yang
>
> **备注:** AAAI 2026 camera ready
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT) is a technique that allows us to adapt powerful Foundation Models (FMs) to diverse downstream tasks while preserving and unleashing their inherent capabilities. However, we have observed that existing PEFT methods, which are often designed with natural imagery in mind, struggle when applied to Remote Sensing (RS) scenarios. This is primarily due to their inability to handle artifact influences, a problem particularly severe in RS image features. To tackle this challenge, we introduce Earth-Adapter, the first PEFT method specifically designed for RS artifacts conquering. Earth-Adapter introduces a novel Mixture of Frequency Adaptation process that combines a Mixture of Adapter (MoA) with Discrete Fourier Transformation (DFT). By utilizing DFT, Earth-Adapter can decompose features into different frequency components, precisely separating artifacts from original features. The MoA then dynamically assigns weights to each adapter expert, allowing for the combination of features across various frequency domains. These simple-yet-effective approaches enable Earth-Adapter to more efficiently overcome the disturbances caused by artifacts than previous PEFT methods, significantly enhancing the FMs' performance on RS scenarios. Experiments on Domain Adaptation (DA), and Domain Generalization (DG) semantic segmentation benchmarks showcase the Earth-Adapter's effectiveness. Compared with baseline Rein, Earth-Adapter significantly improves 9.0% mIoU in DA and 3.1% mIoU in DG benchmarks. Our code will be released at https://github.com/VisionXLab/Earth-Adapter.
>
---
#### [replaced 006] Comparison of Generative Learning Methods for Turbulence Surrogates
- **分类: physics.flu-dyn; cs.CV**

- **链接: [https://arxiv.org/pdf/2411.16417v3](https://arxiv.org/pdf/2411.16417v3)**

> **作者:** Claudia Drygala; Edmund Ross; Francesca di Mare; Hanno Gottschalk
>
> **摘要:** Numerical simulations of turbulent flows present significant challenges in fluid dynamics due to their complexity and high computational cost. High resolution techniques such as Direct Numerical Simulation (DNS) and Large Eddy Simulation (LES) are generally not computationally affordable, particularly for technologically relevant problems. Recent advances in machine learning, specifically in generative probabilistic models, offer promising alternatives as surrogates for turbulence. This paper investigates the application of three generative models - Variational Autoencoders (VAE), Deep Convolutional Generative Adversarial Networks (DCGAN), and Denoising Diffusion Probabilistic Models (DDPM) - in simulating a von Kármán vortex street around a fixed cylinder projected into 2D, as well as a real-world experimental dataset of the wake flow of a cylinder array. Training data was obtained by means of LES in the simulated case and Particle Image Velocimetry (PIV) in the experimental case. We evaluate each model's ability to capture the statistical properties and spatial structures of the turbulent flow. Our results demonstrate that DDPM and DCGAN effectively replicate all flow distributions, highlighting their potential as efficient and accurate tools for turbulence surrogacy. We find a strong argument for DCGAN, as although they are more difficult to train (due to problems such as mode collapse), they show the fastest inference and training time, require less data to train compared to VAE and DDPM, and provide the results most closely aligned with the input stream. In contrast, VAE train quickly (and can generate samples quickly) but do not produce adequate results, and DDPM, whilst effective, are significantly slower at both, inference and training time.
>
---
#### [replaced 007] SOAP: Enhancing Spatio-Temporal Relation and Motion Information Capturing for Few-Shot Action Recognition
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2407.16344v4](https://arxiv.org/pdf/2407.16344v4)**

> **作者:** Wenbo Huang; Jinghui Zhang; Xuwei Qian; Zhen Wu; Meng Wang; Lei Zhang
>
> **备注:** Accepted by ACM MM 2024
>
> **摘要:** High frame-rate (HFR) videos of action recognition improve fine-grained expression while reducing the spatio-temporal relation and motion information density. Thus, large amounts of video samples are continuously required for traditional data-driven training. However, samples are not always sufficient in real-world scenarios, promoting few-shot action recognition (FSAR) research. We observe that most recent FSAR works build spatio-temporal relation of video samples via temporal alignment after spatial feature extraction, cutting apart spatial and temporal features within samples. They also capture motion information via narrow perspectives between adjacent frames without considering density, leading to insufficient motion information capturing. Therefore, we propose a novel plug-and-play architecture for FSAR called Spatio-tempOral frAme tuPle enhancer (SOAP) in this paper. The model we designed with such architecture refers to SOAP-Net. Temporal connections between different feature channels and spatio-temporal relation of features are considered instead of simple feature extraction. Comprehensive motion information is also captured, using frame tuples with multiple frames containing more motion information than adjacent frames. Combining frame tuples of diverse frame counts further provides a broader perspective. SOAP-Net achieves new state-of-the-art performance across well-known benchmarks such as SthSthV2, Kinetics, UCF101, and HMDB51. Extensive empirical evaluations underscore the competitiveness, pluggability, generalization, and robustness of SOAP. The code is released at https://github.com/wenbohuang1002/SOAP.
>
---
#### [replaced 008] PointNSP: Autoregressive 3D Point Cloud Generation with Next-Scale Level-of-Detail Prediction
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.05613v2](https://arxiv.org/pdf/2510.05613v2)**

> **作者:** Ziqiao Meng; Qichao Wang; Zhiyang Dou; Zixing Song; Zhipeng Zhou; Irwin King; Peilin Zhao
>
> **备注:** This work was intended as a replacement of arXiv:2503.08594 and any subsequent updates will appear there
>
> **摘要:** Autoregressive point cloud generation has long lagged behind diffusion-based approaches in quality. The performance gap stems from the fact that autoregressive models impose an artificial ordering on inherently unordered point sets, forcing shape generation to proceed as a sequence of local predictions. This sequential bias emphasizes short-range continuity but undermines the model's capacity to capture long-range dependencies, hindering its ability to enforce global structural properties such as symmetry, consistent topology, and large-scale geometric regularities. Inspired by the level-of-detail (LOD) principle in shape modeling, we propose PointNSP, a coarse-to-fine generative framework that preserves global shape structure at low resolutions and progressively refines fine-grained geometry at higher scales through a next-scale prediction paradigm. This multi-scale factorization aligns the autoregressive objective with the permutation-invariant nature of point sets, enabling rich intra-scale interactions while avoiding brittle fixed orderings. Experiments on ShapeNet show that PointNSP establishes state-of-the-art (SOTA) generation quality for the first time within the autoregressive paradigm. In addition, it surpasses strong diffusion-based baselines in parameter, training, and inference efficiency. Finally, in dense generation with 8,192 points, PointNSP's advantages become even more pronounced, underscoring its scalability potential.
>
---
#### [replaced 009] FlowTok: Flowing Seamlessly Across Text and Image Tokens
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.10772v3](https://arxiv.org/pdf/2503.10772v3)**

> **作者:** Ju He; Qihang Yu; Qihao Liu; Liang-Chieh Chen
>
> **备注:** Project page at https://tacju.github.io/projects/flowtok.html
>
> **摘要:** Bridging different modalities lies at the heart of cross-modality generation. While conventional approaches treat the text modality as a conditioning signal that gradually guides the denoising process from Gaussian noise to the target image modality, we explore a much simpler paradigm-directly evolving between text and image modalities through flow matching. This requires projecting both modalities into a shared latent space, which poses a significant challenge due to their inherently different representations: text is highly semantic and encoded as 1D tokens, whereas images are spatially redundant and represented as 2D latent embeddings. To address this, we introduce FlowTok, a minimal framework that seamlessly flows across text and images by encoding images into a compact 1D token representation. Compared to prior methods, this design reduces the latent space size by 3.3x at an image resolution of 256, eliminating the need for complex conditioning mechanisms or noise scheduling. Moreover, FlowTok naturally extends to image-to-text generation under the same formulation. With its streamlined architecture centered around compact 1D tokens, FlowTok is highly memory-efficient, requires significantly fewer training resources, and achieves much faster sampling speeds-all while delivering performance comparable to state-of-the-art models. Code is available at https://github.com/TACJu/FlowTok.
>
---
#### [replaced 010] Point-Supervised Facial Expression Spotting with Gaussian-Based Instance-Adaptive Intensity Modeling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16952v2](https://arxiv.org/pdf/2511.16952v2)**

> **作者:** Yicheng Deng; Hideaki Hayashi; Hajime Nagahara
>
> **摘要:** Automatic facial expression spotting, which aims to identify facial expression instances in untrimmed videos, is crucial for facial expression analysis. Existing methods primarily focus on fully-supervised learning and rely on costly, time-consuming temporal boundary annotations. In this paper, we investigate point-supervised facial expression spotting (P-FES), where only a single timestamp annotation per instance is required for training. We propose a unique two-branch framework for P-FES. First, to mitigate the limitation of hard pseudo-labeling, which often confuses neutral and expression frames with various intensities, we propose a Gaussian-based instance-adaptive intensity modeling (GIM) module to model instance-level expression intensity distribution for soft pseudo-labeling. By detecting the pseudo-apex frame around each point label, estimating the duration, and constructing an instance-level Gaussian distribution, GIM assigns soft pseudo-labels to expression frames for more reliable intensity supervision. The GIM module is incorporated into our framework to optimize the class-agnostic expression intensity branch. Second, we design a class-aware apex classification branch that distinguishes macro- and micro-expressions solely based on their pseudo-apex frames. During inference, the two branches work independently: the class-agnostic expression intensity branch generates expression proposals, while the class-aware apex-classification branch is responsible for macro- and micro-expression classification. Furthermore, we introduce an intensity-aware contrastive loss to enhance discriminative feature learning and suppress neutral noise by contrasting neutral frames with expression frames with various intensities. Extensive experiments on the SAMM-LV, CAS(ME)$^2$, and CAS(ME)$^3$ datasets demonstrate the effectiveness of our proposed framework.
>
---
#### [replaced 011] A Simple Framework Towards Vision-based Traffic Signal Control with Microscopic Simulation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2403.06884v2](https://arxiv.org/pdf/2403.06884v2)**

> **作者:** Pan He; Quanyi Li; Xiaoyong Yuan; Bolei Zhou
>
> **备注:** Accepted for presentation at the Transportation Research Board (TRB) 105th Annual Meeting
>
> **摘要:** Traffic signal control (TSC) is crucial for reducing traffic congestion leading to smoother traffic flow, reduced idle time, and mitigated CO2 emissions. In this paper, we explore the computer vision approach for TSC that modulates on-road traffic flows through visual observation. Unlike traditional feature-based approaches, vision-based methods depend much less on heuristics and predefined features, bringing promising potentials for end-to-end learning and optimization of traffic signals. Thus, we introduce a simple traffic simulation framework called TrafficDojo towards vision-based TSC and its benchmark by integrating the microscopic traffic flow provided in SUMO into the 3D driving simulator MetaDrive. This proposed framework offers a versatile traffic environment for in-depth analysis and comprehensive evaluation of traffic signal controllers across diverse traffic conditions and scenarios. We establish and compare baseline algorithms including both traditional and Reinforcement Learning (RL) approaches. This work sheds light on the design and development of vision-based TSC approaches and opens up new research opportunities
>
---
#### [replaced 012] Unsupervised Segmentation by Diffusing, Walking and Cutting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.04678v2](https://arxiv.org/pdf/2412.04678v2)**

> **作者:** Daniela Ivanova; Marco Aversa; Paul Henderson; John Williamson
>
> **备注:** Accepted to The IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** We propose an unsupervised image segmentation method using features from pre-trained text-to-image diffusion models. Inspired by classic spectral clustering approaches, we construct adjacency matrices from self-attention layers between image patches and recursively partition using Normalised Cuts. A key insight is that self-attention probability distributions, which capture semantic relations between patches, can be interpreted as a transition matrix for random walks across the image. We leverage this by first using Random Walk Normalized Cuts directly on these self-attention activations to partition the image, minimizing transition probabilities between clusters while maximizing coherence within clusters. Applied recursively, this yields a hierarchical segmentation that reflects the rich semantics in the pre-trained attention layers, without any additional training. Next, we explore other ways to build the NCuts adjacency matrix from features, and how we can use the random walk interpretation of self-attention to capture long-range relationships. Finally, we propose an approach to automatically determine the NCut cost criterion, avoiding the need to tune this manually. We quantitatively analyse the effect incorporating different features, a constant versus dynamic NCut threshold, and incorporating multi-node paths when constructing the NCuts adjacency matrix. We show that our approach surpasses all existing methods for zero-shot unsupervised segmentation, achieving state-of-the-art results on COCO-Stuff-27 and Cityscapes.
>
---
#### [replaced 013] Think Visually, Reason Textually: Vision-Language Synergy in ARC
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对基础模型在少量示例下进行抽象推理的能力不足问题，提出视觉-语言协同推理框架。通过将ARC-AGI任务分解为模态对齐子任务，并引入模态切换自校正机制，利用视觉全局抽象与语言精确执行的优势，显著提升模型在复杂推理任务中的表现。**

- **链接: [https://arxiv.org/pdf/2511.15703v2](https://arxiv.org/pdf/2511.15703v2)**

> **作者:** Beichen Zhang; Yuhang Zang; Xiaoyi Dong; Yuhang Cao; Haodong Duan; Dahua Lin; Jiaqi Wang
>
> **摘要:** Abstract reasoning from minimal examples remains a core unsolved problem for frontier foundation models such as GPT-5 and Grok 4. These models still fail to infer structured transformation rules from a handful of examples, which is a key hallmark of human intelligence. The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) provides a rigorous testbed for this capability, demanding conceptual rule induction and transfer to novel tasks. Most existing methods treat ARC-AGI as a purely textual reasoning task, overlooking the fact that humans rely heavily on visual abstraction when solving such puzzles. However, our pilot experiments reveal a paradox: naively rendering ARC-AGI grids as images degrades performance due to imprecise rule execution. This leads to our central hypothesis that vision and language possess complementary strengths across distinct reasoning stages: vision supports global pattern abstraction and verification, whereas language specializes in symbolic rule formulation and precise execution. Building on this insight, we introduce two synergistic strategies: (1) Vision-Language Synergy Reasoning (VLSR), which decomposes ARC-AGI into modality-aligned subtasks; and (2) Modality-Switch Self-Correction (MSSC), which leverages vision to verify text-based reasoning for intrinsic error correction. Extensive experiments demonstrate that our approach yields up to a 4.33\% improvement over text-only baselines across diverse flagship models and multiple ARC-AGI tasks. Our findings suggest that unifying visual abstraction with linguistic reasoning is a crucial step toward achieving generalizable, human-like intelligence in future foundation models. Source code is released at https://github.com/InternLM/ARC-VL.
>
---
#### [replaced 014] Vision Remember: Recovering Visual Information in Efficient LVLM with Vision Feature Resampling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.03928v2](https://arxiv.org/pdf/2506.03928v2)**

> **作者:** Ze Feng; Jiang-jiang Liu; Sen Yang; Lingyu Xiao; Zhibin Quan; Zhenhua Feng; Wankou Yang; Jingdong Wang
>
> **摘要:** The computational expense of redundant vision tokens in Large Vision-Language Models (LVLMs) has led many existing methods to compress them via a vision projector. However, this compression may lose visual information that is crucial for tasks relying on fine-grained spatial relationships, such as OCR and Chart&Table Understanding. In this paper, we propose to resample original vision features across the LLM decoder layers to recover visual information and attain efficiency. Following this principle, we introduce Vision Remember, which includes two key modules: (1) Token-Feature Cross-Attention Layer and (2) Token Bidirectional Self-Attention Layer. In the Token bidirectional attention, we employ self-attention mechanism to maintain the bidirectional interaction between vision tokens and the text-guided token. In the Token-Feature interaction attention, we introduce local cross-attention to resample the visual feature and utilize the multi-level fusion to enrich the visual representation. We conduct comprehensive experiments on multiple visual understanding benchmarks and the results with the LLaVA-NeXT baseline show that Vision Remember outperforms TokenPacker by +2.7 and FastV by +5.7 across nearly all the settings. Compared with previous vision feature re-fusion methods, our approach also surpasses DeepStack by +3.9 and SVA Aggregator by +3.4 on the same baseline. The experimental results validate the generalization capability of the proposed method when combined with various efficient vision projectors and LVLMs.
>
---
#### [replaced 015] Dynamic Epsilon Scheduling: A Multi-Factor Adaptive Perturbation Budget for Adversarial Training
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.04263v2](https://arxiv.org/pdf/2506.04263v2)**

> **作者:** Alan Mitkiy; James Smith; Myungseo wong; Hana Satou; Hiroshi Tanaka; Emily Johnson
>
> **摘要:** Adversarial training is among the most effective strategies for defending deep neural networks against adversarial examples. A key limitation of existing adversarial training approaches lies in their reliance on a fixed perturbation budget, which fails to account for instance-specific robustness characteristics. While prior works such as IAAT and MMA introduce instance-level adaptations, they often rely on heuristic or static approximations of data robustness. In this paper, we propose Dynamic Epsilon Scheduling (DES), a novel framework that adaptively adjusts the adversarial perturbation budget per instance and per training iteration. DES integrates three key factors: (1) the distance to the decision boundary approximated via gradient-based proxies, (2) prediction confidence derived from softmax entropy, and (3) model uncertainty estimated via Monte Carlo dropout. By combining these cues into a unified scheduling strategy, DES tailors the perturbation budget dynamically to guide more effective adversarial learning. Experimental results on CIFAR-10 and CIFAR-100 show that our method consistently improves both adversarial robustness and standard accuracy compared to fixed-epsilon baselines and prior adaptive methods. Moreover, we provide theoretical insights into the stability and convergence of our scheduling policy. This work opens a new avenue for instance-aware, data-driven adversarial training methods.
>
---
#### [replaced 016] DensiCrafter: Physically-Constrained Generation and Fabrication of Self-Supporting Hollow Structures
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.09298v2](https://arxiv.org/pdf/2511.09298v2)**

> **作者:** Shengqi Dang; Fu Chai; Jiaxin Li; Chao Yuan; Wei Ye; Nan Cao
>
> **摘要:** The rise of 3D generative models has enabled automatic 3D geometry and texture synthesis from multimodal inputs (e.g., text or images). However, these methods often ignore physical constraints and manufacturability considerations. In this work, we address the challenge of producing 3D designs that are both lightweight and self-supporting. We present DensiCrafter, a framework for generating lightweight, self-supporting 3D hollow structures by optimizing the density field. Starting from coarse voxel grids produced by Trellis, we interpret these as continuous density fields to optimize and introduce three differentiable, physically constrained, and simulation-free loss terms. Additionally, a mass regularization penalizes unnecessary material, while a restricted optimization domain preserves the outer surface. Our method seamlessly integrates with pretrained Trellis-based models (e.g., Trellis, DSO) without any architectural changes. In extensive evaluations, we achieve up to 43% reduction in material mass on the text-to-3D task. Compared to state-of-the-art baselines, our method could improve the stability and maintain high geometric fidelity. Real-world 3D-printing experiments confirm that our hollow designs can be reliably fabricated and could be self-supporting.
>
---
#### [replaced 017] ISAC: Training-Free Instance-to-Semantic Attention Control for Improving Multi-Instance Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.20935v2](https://arxiv.org/pdf/2505.20935v2)**

> **作者:** Sanghyun Jo; Wooyeol Lee; Ziseok Lee; Kyungsu Kim
>
> **备注:** 36 pages
>
> **摘要:** Text-to-image diffusion models have recently become highly capable, yet their behavior in multi-object scenes remains unreliable: models often produce an incorrect number of instances and exhibit semantics leaking across objects. We trace these failures to vague instance boundaries; self-attention already reveals instance layouts early in the denoising process, but existing approaches act only on semantic signals. We introduce $\textbf{ISAC}$ ($\textbf{I}$nstance-to-$\textbf{S}$emantic $\textbf{A}$ttention $\textbf{C}$ontrol), a training-free, model-agnostic objective that performs hierarchical attention control by first carving out instance layouts from self-attention and then binding semantics to these instances. In Phase 1, ISAC clusters self-attention into the number of instances and repels overlaps, establishing an instance-level structural hierarchy; in Phase 2, it injects these instance cues into cross-attention to obtain instance-aware semantic masks and decomposes mixing semantics by tying attributes within each instance. ISAC yields consistent gains on T2I-CompBench, HRS-Bench, and IntraCompBench, our new benchmark for intra-class compositions where failures are most frequent, with improvements of at least 50% in multi-class accuracy and 7% in multi-instance accuracy on IntraCompBench, without any fine-tuning or external models. Beyond text-to-image setups, ISAC also strengthens layout-to-image controllers under overlapping boxes by refining coarse box layouts into dense instance masks, indicating that hierarchical decoupling of instance formation and semantic assignment is a key principle for robust, controllable multi-object generation. Code will be released upon publication.
>
---
#### [replaced 018] DiffSeg30k: A Multi-Turn Diffusion Editing Benchmark for Localized AIGC Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19111v2](https://arxiv.org/pdf/2511.19111v2)**

> **作者:** Hai Ci; Ziheng Peng; Pei Yang; Yingxin Xuan; Mike Zheng Shou
>
> **备注:** 16 pages, 10 figures; typos corrected, references added
>
> **摘要:** Diffusion-based editing enables realistic modification of local image regions, making AI-generated content harder to detect. Existing AIGC detection benchmarks focus on classifying entire images, overlooking the localization of diffusion-based edits. We introduce DiffSeg30k, a publicly available dataset of 30k diffusion-edited images with pixel-level annotations, designed to support fine-grained detection. DiffSeg30k features: 1) In-the-wild images--we collect images or image prompts from COCO to reflect real-world content diversity; 2) Diverse diffusion models--local edits using eight SOTA diffusion models; 3) Multi-turn editing--each image undergoes up to three sequential edits to mimic real-world sequential editing; and 4) Realistic editing scenarios--a vision-language model (VLM)-based pipeline automatically identifies meaningful regions and generates context-aware prompts covering additions, removals, and attribute changes. DiffSeg30k shifts AIGC detection from binary classification to semantic segmentation, enabling simultaneous localization of edits and identification of the editing models. We benchmark three baseline segmentation approaches, revealing significant challenges in semantic segmentation tasks, particularly concerning robustness to image distortions. Experiments also reveal that segmentation models, despite being trained for pixel-level localization, emerge as highly reliable whole-image classifiers of diffusion edits, outperforming established forgery classifiers while showing great potential in cross-generator generalization. We believe DiffSeg30k will advance research in fine-grained localization of AI-generated content by demonstrating the promise and limitations of segmentation-based methods. DiffSeg30k is released at: https://huggingface.co/datasets/Chaos2629/Diffseg30k
>
---
#### [replaced 019] DWFF-Net : A Multi-Scale Farmland System Habitat Identification Method with Adaptive Dynamic Weight
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11659v2](https://arxiv.org/pdf/2511.11659v2)**

> **作者:** Kesong Zheng; Zhi Song; Peizhou Li; Shuyi Yao; Zhenxing Bian
>
> **备注:** 30 pages,13 figures
>
> **摘要:** Addressing the current lack of a standardized habitat classification system for cultivated land ecosystems, incomplete coverage of the habitat types, and the inability of existing models to effectively integrate semantic and texture features-resulting in insufficient segmentation accuracy and blurred boundaries for multi-scale habitats (e.g., large-scale field plots and micro-habitats)-this study developed a comprehensively annotated ultra-high-resolution remote sensing image dataset encompassing 15 categories of cultivated land system habitats. Furthermore, we propose a Dynamic-Weighted Feature Fusion Network (DWFF-Net). The encoder of this model utilizes a frozen-parameter DINOv3 to extract foundational features. By analyzing the relationships between different category images and feature maps, we introduce a data-level adaptive dynamic weighting strategy for feature fusion. The decoder incorporates a dynamic weight computation network to achieve thorough integration of multi-layer features, and a hybrid loss function is adopted to optimize model training. Experimental results on the constructed dataset demonstrate that the proposed model achieves a mean Intersection over Union (mIoU) of 69.79% and an F1-score of 80.49%, outperforming the baseline network by 2.1% and 1.61%, respectively. Ablation studies further confirm the complementary nature of multi-layer feature fusion, which effectively improves the IoU for micro-habitat categories such as field ridges. This study establishes a habitat identification framework for cultivated land systems based on adaptive multi-layer feature fusion, enabling sub-meter precision habitat mapping at a low cost and providing robust technical support for fine-grained habitat monitoring in cultivated landscapes. (The complete code repository can be accessed via GitHub at the following URL: https://github.com/sysau/DWFF-Net)
>
---
#### [replaced 020] One-Step Diffusion Transformer for Controllable Real-World Image Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17138v2](https://arxiv.org/pdf/2511.17138v2)**

> **作者:** Yushun Fang; Yuxiang Chen; Shibo Yin; Qiang Hu; Jiangchao Yao; Ya Zhang; Xiaoyun Zhang; Yanfeng Wang
>
> **摘要:** Recent advances in diffusion-based real-world image super-resolution (Real-ISR) have demonstrated remarkable perceptual quality, yet the balance between fidelity and controllability remains a problem: multi-step diffusion-based methods suffer from generative diversity and randomness, resulting in low fidelity, while one-step methods lose control flexibility due to fidelity-specific finetuning. In this paper, we present ODTSR, a one-step diffusion transformer based on Qwen-Image that performs Real-ISR considering fidelity and controllability simultaneously: a newly introduced visual stream receives low-quality images (LQ) with adjustable noise (Control Noise), and the original visual stream receives LQs with consistent noise (Prior Noise), forming the Noise-hybrid Visual Stream (NVS) design. ODTSR further employs Fidelity-aware Adversarial Training (FAA) to enhance controllability and achieve one-step inference. Extensive experiments demonstrate that ODTSR not only achieves state-of-the-art (SOTA) performance on generic Real-ISR, but also enables prompt controllability on challenging scenarios such as real-world scene text image super-resolution (STISR) of Chinese characters without training on specific datasets. Codes are available at $\href{https://github.com/RedMediaTech/ODTSR}{\text{this url}}$.
>
---
#### [replaced 021] Activator: GLU Activation Function as the Core Component of a Vision Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.15953v4](https://arxiv.org/pdf/2405.15953v4)**

> **作者:** Abdullah Nazhat Abdullah; Tarkan Aydin
>
> **摘要:** The transformer architecture has driven many successes in a variety of tasks within the field of deep learning, in particular the recent advances in natural language processing (NLP) culminating with large language models (LLM). Adding to that success, transformer architecture has found widespread interest from computer vision (CV) researchers and practitioners, allowing for many advancements in vision-related tasks and opening the door for multitask and multi-modal deep learning architectures that share the same principle of operation. One drawback to these architectures is their reliance on the scaled dot product attention mechanism with the softmax activation function, which is computationally expensive and requires large compute capabilities for both training and inference. This paper investigates substituting the MLP and attention mechanism usually adopted for transformer architecture with an architecture based on incorporating a gated linear unit (GLU) activation function structure with the aim of reducing the computational cost. The equalized experimental assessments conducted in this work show that the proposed modification with the targeted reductions in computational complexity offers competitive performance compared to the selected baseline architectures. The results are significantly in support of the aims of this work, in which the focus was to extensively utilize GLU-based MLPs, establishing a more efficient but capable alternative to the traditional MLP and the attention mechanism as the core component in the design of transformer architectures.
>
---
#### [replaced 022] LMLCC-Net: A Semi-Supervised Deep Learning Model for Lung Nodule Malignancy Prediction from CT Scans using a Novel Hounsfield Unit-Based Intensity Filtering
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.06370v2](https://arxiv.org/pdf/2505.06370v2)**

> **作者:** Tasnia Binte Mamun; Adhora Madhuri; Nusaiba Sobir; Taufiq Hasan
>
> **备注:** 12 pages, 9 figures, 7 tables
>
> **摘要:** Lung cancer is the leading cause of patient mortality in the world. Early diagnosis of malignant pulmonary nodules in CT images can have a significant impact on reducing disease mortality and morbidity. In this work, we propose LMLCC-Net, a novel deep learning framework for classifying nodules from CT scan images using a 3D CNN, considering Hounsfield Unit (HU)-based intensity filtering. Benign and malignant nodules have significant differences in their intensity profile of HU, which was not exploited in the literature. Our method considers the intensity pattern as well as the texture for the prediction of malignancies. LMLCC-Net extracts features from multiple branches that each use a separate learnable HU-based intensity filtering stage. Various combinations of branches and learnable ranges of filters were explored to finally produce the best-performing model. In addition, we propose a semi-supervised learning scheme for labeling ambiguous cases and also developed a lightweight model to classify the nodules. The experimental evaluations are carried out on the LUNA16 dataset. The proposed LMLCC-Net was evaluated using the LUNA16 dataset. Our proposed method achieves a classification accuracy of 91.96%, a sensitivity of 92.94%, and an area under the curve of 94.07%, showing improved performance compared to existing methods The proposed method can have a significant impact in helping radiologists in the classification of pulmonary nodules and improving patient care.
>
---
#### [replaced 023] Gen-3Diffusion: Realistic Image-to-3D Generation via 2D & 3D Diffusion Synergy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.06698v2](https://arxiv.org/pdf/2412.06698v2)**

> **作者:** Yuxuan Xue; Xianghui Xie; Riccardo Marin; Gerard Pons-Moll
>
> **备注:** Accepted to Transaction on Pattern Analysis and Machine Intelligence (T-PAMI). Project Page: https://yuxuan-xue.com/gen-3diffusion. arXiv admin note: substantial text overlap with arXiv:2406.08475
>
> **摘要:** Creating realistic 3D objects and clothed avatars from a single RGB image is an attractive yet challenging problem. Due to its ill-posed nature, recent works leverage powerful prior from 2D diffusion models pretrained on large datasets. Although 2D diffusion models demonstrate strong generalization capability, they cannot guarantee the generated multi-view images are 3D consistent. In this paper, we propose Gen-3Diffusion: Realistic Image-to-3D Generation via 2D & 3D Diffusion Synergy. We leverage a pre-trained 2D diffusion model and a 3D diffusion model via our elegantly designed process that synchronizes two diffusion models at both training and sampling time. The synergy between the 2D and 3D diffusion models brings two major advantages: 1) 2D helps 3D in generalization: the pretrained 2D model has strong generalization ability to unseen images, providing strong shape priors for the 3D diffusion model; 2) 3D helps 2D in multi-view consistency: the 3D diffusion model enhances the 3D consistency of 2D multi-view sampling process, resulting in more accurate multi-view generation. We validate our idea through extensive experiments in image-based objects and clothed avatar generation tasks. Results show that our method generates realistic 3D objects and avatars with high-fidelity geometry and texture. Extensive ablations also validate our design choices and demonstrate the strong generalization ability to diverse clothing and compositional shapes. Our code and pretrained models will be publicly released on https://yuxuan-xue.com/gen-3diffusion.
>
---
#### [replaced 024] Without Paired Labeled Data: End-to-End Self-Supervised Learning for Drone-view Geo-Localization
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2502.11381v5](https://arxiv.org/pdf/2502.11381v5)**

> **作者:** Zhongwei Chen; Zhao-Xu Yang; Hai-Jun Rong; Guoqi Li
>
> **摘要:** Drone-view Geo-Localization (DVGL) aims to achieve accurate localization of drones by retrieving the most relevant GPS-tagged satellite images. However, most existing methods heavily rely on strictly pre-paired drone-satellite images for supervised learning. When the target region shifts, new paired samples are typically required to adapt to the distribution changes. The high cost of annotation and the limited transferability of these methods significantly hinder the practical deployment of DVGL in open-world scenarios. To address these limitations, we propose a novel end-to-end self-supervised learning method with a shallow backbone network, called the dynamic memory-driven and neighborhood information learning (DMNIL) method. It employs a clustering algorithm to generate pseudo-labels and adopts a dual-path contrastive learning framework to learn discriminative intra-view representations. Furthermore, DMNIL incorporates two core modules, including the dynamic hierarchical memory learning (DHML) module and the information consistency evolution learning (ICEL) module. The DHML module combines short-term and long-term memory to enhance intra-view feature consistency and discriminability. Meanwhile, the ICEL module utilizes a neighborhood-driven dynamic constraint mechanism to systematically capture implicit cross-view semantic correlations, consequently improving cross-view feature alignment. To further stabilize and strengthen the self-supervised training process, a pseudo-label enhancement strategy is introduced to enhance the quality of pseudo supervision. Extensive experiments on three public benchmark datasets demonstrate that the proposed method consistently outperforms existing self-supervised methods and even surpasses several state-of-the-art supervised methods. Our code is available at https://github.com/ISChenawei/DMNIL.
>
---
#### [replaced 025] Stream and Query-guided Feature Aggregation for Efficient and Effective 3D Occupancy Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.22087v2](https://arxiv.org/pdf/2503.22087v2)**

> **作者:** Seokha Moon; Janghyun Baek; Giseop Kim; Jinkyu Kim; Sunwook Choi
>
> **摘要:** 3D occupancy prediction has become a key perception task in autonomous driving, as it enables comprehensive scene understanding. Recent methods enhance this understanding by incorporating spatiotemporal information through multi-frame fusion, but they suffer from a trade-off: dense voxel-based representations provide high accuracy at significant computational cost, whereas sparse representations improve efficiency but lose spatial detail. To mitigate this trade-off, we introduce DuOcc, which employs a dual aggregation strategy that retains dense voxel representations to preserve spatial fidelity while maintaining high efficiency. DuOcc consists of two key components: (i) Stream-based Voxel Aggregation, which recurrently accumulates voxel features over time and refines them to suppress warping-induced distortions, preserving a clear separation between occupied and free space. (ii) Query-guided Aggregation, which complements the limitations of voxel accumulation by selectively injecting instance-level query features into the voxel regions occupied by dynamic objects. Experiments on the widely used Occ3D-nuScenes and SurroundOcc datasets demonstrate that DuOcc achieves state-of-the-art performance in real-time settings, while reducing memory usage by over 40% compared to prior methods.
>
---
#### [replaced 026] Automated Neural Architecture Design for Industrial Defect Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.06669v2](https://arxiv.org/pdf/2510.06669v2)**

> **作者:** Yuxi Liu; Yunfeng Ma; Yi Tang; Min Liu; Shuai Jiang; Yaonan Wang
>
> **摘要:** Industrial surface defect detection (SDD) is critical for ensuring product quality and manufacturing reliability. Due to the diverse shapes and sizes of surface defects, SDD faces two main challenges: intraclass difference and interclass similarity. Existing methods primarily utilize manually designed models, which require extensive trial and error and often struggle to address both challenges effectively. To overcome this, we propose AutoNAD, an automated neural architecture design framework for SDD that jointly searches over convolutions, transformers, and multi-layer perceptrons. This hybrid design enables the model to capture both fine-grained local variations and long-range semantic context, addressing the two key challenges while reducing the cost of manual network design. To support efficient training of such a diverse search space, AutoNAD introduces a cross weight sharing strategy, which accelerates supernet convergence and improves subnet performance. Additionally, a searchable multi-level feature aggregation module (MFAM) is integrated to enhance multi-scale feature learning. Beyond detection accuracy, runtime efficiency is essential for industrial deployment. To this end, AutoNAD incorporates a latency-aware prior to guide the selection of efficient architectures. The effectiveness of AutoNAD is validated on three industrial defect datasets and further applied within a defect imaging and detection platform. Code is available at https://github.com/Yuxi104/AutoNAD.
>
---
#### [replaced 027] Interactive Occlusion Boundary Estimation through Exploitation of Synthetic Data
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2408.15038v3](https://arxiv.org/pdf/2408.15038v3)**

> **作者:** Lintao Xu; Chaohui Wang
>
> **备注:** BMVC 2025
>
> **摘要:** Occlusion boundaries (OBs) geometrically localize occlusion events in 2D images and provide critical cues for scene understanding. In this paper, we present the first systematic study of Interactive Occlusion Boundary Estimation (IOBE), introducing MS\textsuperscript{3}PE, a novel multi-scribble-guided deep-learning framework that advances IOBE through two key innovations: (1) an intuitive multi-scribble interaction mechanism, and (2) a 3-encoding-path network enhanced with multi-scale strip convolutions. Our MS\textsuperscript{3}PE surpasses adapted baselines from seven state-of-the-art interactive segmentation methods, and demonstrates strong potential for OB benchmark construction through our real-user experiment. Besides, to address the scarcity of well-annotated real-world data, we propose using synthetic data for training IOBE models, and developed Mesh2OB, the first automated tool for generating precise ground-truth OBs from 3D scenes with self-occlusions explicitly handled, enabling creation of the OB-FUTURE synthetic benchmark that facilitates generalizable training without domain adaptation. Finally, we introduce OB-LIGM, a high-quality real-world benchmark comprising 120 meticulously annotated high-resolution images advancing evaluation standards in OB research. Source code and resources are available at https://github.com/xul-ops/IOBE.
>
---
#### [replaced 028] MANGO: Multimodal Attention-based Normalizing Flow Approach to Fusion Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.10133v2](https://arxiv.org/pdf/2508.10133v2)**

> **作者:** Thanh-Dat Truong; Christophe Bobda; Nitin Agarwal; Khoa Luu
>
> **备注:** Accepted to NeurIPS'25
>
> **摘要:** Multimodal learning has gained much success in recent years. However, current multimodal fusion methods adopt the attention mechanism of Transformers to implicitly learn the underlying correlation of multimodal features. As a result, the multimodal model cannot capture the essential features of each modality, making it difficult to comprehend complex structures and correlations of multimodal inputs. This paper introduces a novel Multimodal Attention-based Normalizing Flow (MANGO) approach to developing explicit, interpretable, and tractable multimodal fusion learning. In particular, we propose a new Invertible Cross-Attention (ICA) layer to develop the Normalizing Flow-based Model for multimodal data. To efficiently capture the complex, underlying correlations in multimodal data in our proposed invertible cross-attention layer, we propose three new cross-attention mechanisms: Modality-to-Modality Cross-Attention (MMCA), Inter-Modality Cross-Attention (IMCA), and Learnable Inter-Modality Cross-Attention (LICA). Finally, we introduce a new Multimodal Attention-based Normalizing Flow to enable the scalability of our proposed method to high-dimensional multimodal data. Our experimental results on three different multimodal learning tasks, i.e., semantic segmentation, image-to-image translation, and movie genre classification, have illustrated the state-of-the-art (SoTA) performance of the proposed approach.
>
---
#### [replaced 029] Saliency-R1: Incentivizing Unified Saliency Reasoning Capability in MLLM with Confidence-Guided Reinforcement Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.00396v3](https://arxiv.org/pdf/2511.00396v3)**

> **作者:** Long Li; Shuichen Ji; Ziyang Luo; Zhihui Li; Dingwen Zhang; Junwei Han; Nian Liu
>
> **备注:** Main text (excluding references): 8 pages, 4 figures; Supplementary Materials (excluding references): 9 pages, 10 figures
>
> **摘要:** Although multimodal large language models (MLLMs) excel in high-level vision-language reasoning, they lack inherent awareness of visual saliency, making it difficult to identify key visual elements. To bridge this gap, we propose Saliency-R1, the first unified MLLM framework that jointly tackles three representative and heterogeneous saliency tasks: Salient Object Detection (SOD), Salient Instance Segmentation (SIS), and Co-salient Object Detection (CoSOD), enhancing the model's capacity for saliency reasoning. We introduce a textual interface with structured tags (<rg>, <ins>) to encode region- and instance-level referring expressions, enabling a single referring segmenter to produce task-appropriate masks. To train the MLLM efficiently, we propose Confidence-Guided Policy Optimization (CGPO), a novel single-sample reinforcement learning algorithm. CGPO improves on GRPO by replacing group-normalized advantages with a per-sample signal based on reward-confidence discrepancy, thereby reducing computational waste, mitigating signal dilution, and lowering training overhead. Our model exceeds or matches the performance of robust open/closed-source MLLMs and specialized state-of-the-art methods across all three tasks, demonstrating the efficacy of our framework in saliency reasoning.
>
---
#### [replaced 030] EvoEmpirBench: Dynamic Spatial Reasoning with Agent-ExpVer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.12718v2](https://arxiv.org/pdf/2509.12718v2)**

> **作者:** Pukun Zhao; Longxiang Wang; Miaowei Wang; Chen Chen; Fanqing Zhou; Haojian Huang
>
> **备注:** Accepted by AAAI 2026, 29 pages, 3 figures, 7 tables
>
> **摘要:** Most existing spatial reasoning benchmarks focus on static or globally observable environments, failing to capture the challenges of long-horizon reasoning and memory utilization under partial observability and dynamic changes. We introduce two dynamic spatial benchmarks, locally observable maze navigation and match-2 elimination that systematically evaluate models' abilities in spatial understanding and adaptive planning when local perception, environment feedback, and global objectives are tightly coupled. Each action triggers structural changes in the environment, requiring continuous update of cognition and strategy. We further propose a subjective experience-based memory mechanism for cross-task experience transfer and validation. Experiments show that our benchmarks reveal key limitations of mainstream models in dynamic spatial reasoning and long-term memory, providing a comprehensive platform for future methodological advances. Our code and data are available at https://anonymous.4open.science/r/EvoEmpirBench-143C/.
>
---
#### [replaced 031] Vision-Language Enhanced Foundation Model for Semi-supervised Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19759v2](https://arxiv.org/pdf/2511.19759v2)**

> **作者:** Jiaqi Guo; Mingzhen Li; Hanyu Su; Santiago López; Lexiaozi Fan; Daniel Kim; Aggelos Katsaggelos
>
> **摘要:** Semi-supervised learning (SSL) has emerged as an effective paradigm for medical image segmentation, reducing the reliance on extensive expert annotations. Meanwhile, vision-language models (VLMs) have demonstrated strong generalization and few-shot capabilities across diverse visual domains. In this work, we integrate VLM-based segmentation into semi-supervised medical image segmentation by introducing a Vision-Language Enhanced Semi-supervised Segmentation Assistant (VESSA) that incorporates foundation-level visual-semantic understanding into SSL frameworks. Our approach consists of two stages. In Stage 1, the VLM-enhanced segmentation foundation model VESSA is trained as a reference-guided segmentation assistant using a template bank containing gold-standard exemplars, simulating learning from limited labeled data. Given an input-template pair, VESSA performs visual feature matching to extract representative semantic and spatial cues from exemplar segmentations, generating structured prompts for a SAM2-inspired mask decoder to produce segmentation masks. In Stage 2, VESSA is integrated into a state-of-the-art SSL framework, enabling dynamic interaction with the student model: as student predictions become more refined, they are fed back to VESSA as prompts, allowing it to generate higher-quality pseudo-labels and stronger guidance. Extensive experiments across multiple segmentation datasets and domains show that VESSA-augmented SSL significantly enhances segmentation accuracy, outperforming state-of-the-art baselines under extremely limited annotation conditions.
>
---
#### [replaced 032] DEMIST: Decoupled Multi-stream latent diffusion for Quantitative Myelin Map Synthesis
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12396v2](https://arxiv.org/pdf/2511.12396v2)**

> **作者:** Jiacheng Wang; Hao Li; Xing Yao; Ahmad Toubasi; Taegan Vinarsky; Caroline Gheen; Joy Derwenskus; Chaoyang Jin; Richard Dortch; Junzhong Xu; Francesca Bagnato; Ipek Oguz
>
> **摘要:** Quantitative magnetization transfer (qMT) imaging provides myelin-sensitive biomarkers, such as the pool size ratio (PSR), which is valuable for multiple sclerosis (MS) assessment. However, qMT requires specialized 20-30 minute scans. We propose DEMIST to synthesize PSR maps from standard T1w and FLAIR images using a 3D latent diffusion model with three complementary conditioning mechanisms. Our approach has two stages: first, we train separate autoencoders for PSR and anatomical images to learn aligned latent representations. Second, we train a conditional diffusion model in this latent space on top of a frozen diffusion foundation backbone. Conditioning is decoupled into: (i) \textbf{semantic} tokens via cross-attention, (ii) \textbf{spatial} per-scale residual hints via a 3D ControlNet branch, and (iii) \textbf{adaptive} LoRA-modulated attention. We include edge-aware loss terms to preserve lesion boundaries and alignment losses to maintain quantitative consistency, while keeping the number of trainable parameters low and retaining the inductive bias of the pretrained model. We evaluate on 163 scans from 99 subjects using 5-fold cross-validation. Our method outperforms VAE, GAN and diffusion baselines on multiple metrics, producing sharper boundaries and better quantitative agreement with ground truth. Our code is publicly available at https://github.com/MedICL-VU/MS-Synthesis-3DcLDM.
>
---
#### [replaced 033] ControlEvents: Controllable Synthesis of Event Camera Datawith Foundational Prior from Image Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.22864v2](https://arxiv.org/pdf/2509.22864v2)**

> **作者:** Yixuan Hu; Yuxuan Xue; Simon Klenk; Daniel Cremers; Gerard Pons-Moll
>
> **备注:** Accepted to WACV2026. Project website: https://https://yuxuan-xue.com/controlevents/
>
> **摘要:** In recent years, event cameras have gained significant attention due to their bio-inspired properties, such as high temporal resolution and high dynamic range. However, obtaining large-scale labeled ground-truth data for event-based vision tasks remains challenging and costly. In this paper, we present ControlEvents, a diffusion-based generative model designed to synthesize high-quality event data guided by diverse control signals such as class text labels, 2D skeletons, and 3D body poses. Our key insight is to leverage the diffusion prior from foundation models, such as Stable Diffusion, enabling high-quality event data generation with minimal fine-tuning and limited labeled data. Our method streamlines the data generation process and significantly reduces the cost of producing labeled event datasets. We demonstrate the effectiveness of our approach by synthesizing event data for visual recognition, 2D skeleton estimation, and 3D body pose estimation. Our experiments show that the synthesized labeled event data enhances model performance in all tasks. Additionally, our approach can generate events based on unseen text labels during training, illustrating the powerful text-based generation capabilities inherited from foundation models.
>
---
#### [replaced 034] Decorrelation Speeds Up Vision Transformers
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.14657v2](https://arxiv.org/pdf/2510.14657v2)**

> **作者:** Kieran Carrigg; Rob van Gastel; Melda Yeghaian; Sander Dalm; Faysal Boughorbel; Marcel van Gerven
>
> **备注:** 16 pages, 12 figures, submitted to CVC 2026
>
> **摘要:** Masked Autoencoder (MAE) pre-training of vision transformers (ViTs) yields strong performance in low-label data regimes but comes with substantial computational costs, making it impractical in time- and resource-constrained industrial settings. We address this by nitegrating Decorrelated Backpropagation (DBP) into MAE pre-training, an optimization method that iteratively reduces input correlations at each layer to accelerate convergence. Applied selectively to the encoder, DBP achieves faster pre-training without loss of stability. To mimic constrained-data scenarios, we evaluate our approach on ImageNet-1K pre-training and ADE20K fine-tuning using randomly sampled subsets of each dataset. Under this setting, DBP-MAE reduces wall-clock time to baseline performance by 21.1%, lowers carbon emissions by 21.4%, and improves segmentation mIoU by 1.1 points. We observe similar gains when pre-training and fine-tuning on proprietary industrial data, confirming the method's applicability in real-world scenarios. These results demonstrate that DBP can reduce training time and energy use while improving downstream performance for large-scale ViT pre-training. Keywords: Deep learning, Vision transformers, Efficient AI, Decorrelation
>
---
#### [replaced 035] Generalizable cardiac substructures segmentation from contrast and non-contrast CTs using pretrained transformers
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.10855v2](https://arxiv.org/pdf/2505.10855v2)**

> **作者:** Aneesh Rangnekar; Nikhil Mankuzhy; Jonas Willmann; Chloe Choi; Abraham Wu; Maria Thor; Andreas Rimner; Harini Veeraraghavan
>
> **摘要:** Automated AI segmentations for radiation treatment planning deteriorate when applied to cases with different characteristics than the training dataset. We developed a hybrid transformer convolutional network to segment cardiac substructures in lung and breast cancer patients with varying imaging contrasts and scan positions. Cohort I (56 contrast-enhanced CT [CECT], 124 non-contrast CT [NCCT] scans from lung cancer patients, supine position) was used to train an oracle model (180 cases), contrast-only model (56 CECTs), and balanced model (32 CECT, 32 NCCT). All models were evaluated on 60 held-out cohort I patients and 66 cohort II breast cancer patients (45 supine, 21 prone). Accuracy was measured using Dice similarity coefficient (DSC), 95th percentile Hausdorff distance (HD95), and dosimetric metrics, with TotalSegmentator as benchmark. Oracle and balanced models achieved similar accuracy (DSC: Oracle vs Balanced: Cohort I: 0.84 $\pm$ 0.10 vs 0.82 $\pm$ 0.10; Cohort II: 0.81 $\pm$ 0.12 vs 0.80 $\pm$ 0.13), both outperforming TotalSegmentator and the contrast-only models. The balanced model, using 64% fewer training cases, produced dosimetrically equivalent contours to manual delineations. It was robust to contrast variations (6 out of 8 substructures) and positioning variations (5 out of 8 substructures), with low correlation to patient age or body mass index. Our balanced model demonstrated robust geometric and dosimetric accuracy across varying imaging protocols and patient characteristics, which is essential for clinical deployment. Combining pretraining with balanced NCCT/CECT distribution enabled reliable segmentation with substantially fewer labeled cases than conventional approaches.
>
---
#### [replaced 036] Geometrically Regularized Transfer Learning with On-Manifold and Off-Manifold Perturbation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.15191v2](https://arxiv.org/pdf/2505.15191v2)**

> **作者:** Hana Satou; Alan Mitkiy; Emma Collins; Finn Kingston
>
> **摘要:** Transfer learning under domain shift remains a fundamental challenge due to the divergence between source and target data manifolds. In this paper, we propose MAADA (Manifold-Aware Adversarial Data Augmentation), a novel framework that decomposes adversarial perturbations into on-manifold and off-manifold components to simultaneously capture semantic variation and model brittleness. We theoretically demonstrate that enforcing on-manifold consistency reduces hypothesis complexity and improves generalization, while off-manifold regularization smooths decision boundaries in low-density regions. Moreover, we introduce a geometry-aware alignment loss that minimizes geodesic discrepancy between source and target manifolds. Experiments on DomainNet, VisDA, and Office-Home show that MAADA consistently outperforms existing adversarial and adaptation methods in both unsupervised and few-shot settings, demonstrating superior structural robustness and cross-domain generalization.
>
---
#### [replaced 037] AMLP: Adjustable Masking Lesion Patches for Self-Supervised Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2309.04312v2](https://arxiv.org/pdf/2309.04312v2)**

> **作者:** Xiangtao Wang; Ruizhi Wang; Thomas Lukasiewicz; Zhenghua Xu
>
> **备注:** © 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Self-supervised masked image modeling (MIM) methods have shown promising performances on analyzing natural images. However, directly applying such methods to medical image segmentation tasks still cannot achieve satisfactory results. The challenges arise from the facts that (i) medical images are inherently more complex compared to natural images, and the subjects in medical images often exhibit more distinct contour features; (ii) moreover, the conventional high and fixed masking ratio in MIM is likely to mask the background, limiting the scope of learnable information. To address these problems, we propose a new self-supervised medical image segmentation framework, called Adjustable Masking Lesion Patches (AMLP), which employs Masked Patch Selection~(MPS) strategy to identify patches with high probabilities of containing lesions to help model achieve precise lesion reconstruction. To improve the categorization of patches in MPS, we further introduce Relative Reconstruction Loss (RRL) to better learn hard-to-reconstruct lesion patches. Then, Category Consistency Loss (CCL) is proposed to refine patch categorization based on reconstruction difficulty, enhancing difference between lesions and backgrounds. Moreover, an Adjustable Masking Ratio (AMR) strategy is proposed to gradually increase the masking ratio over training to expand~the scope of learnable mutual information. Extensive~experiments on two medical segmentation datasets demonstrate the superior performances of the proposed AMLP w.r.t. the SOTA self-supervised methods; the results prove that AMLP effectively addresses the challenges of applying masked modeling to medical images and capturing accurate lesion details that are crucial for segmentation tasks.
>
---
#### [replaced 038] CrossEarth-Gate: Fisher-Guided Adaptive Tuning Engine for Efficient Adaptation of Cross-Domain Remote Sensing Semantic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20302v2](https://arxiv.org/pdf/2511.20302v2)**

> **作者:** Shilei Cao; Ziyang Gong; Hehai Lin; Yang Liu; Jiashun Cheng; Xiaoxing Hu; Haoyuan Liang; Guowen Li; Chengwei Qin; Hong Cheng; Xue Yang; Juepeng Zheng; Haohuan Fu
>
> **摘要:** In Remote Sensing (RS), Parameter-Efficient Fine-Tuning (PEFT) has emerged as a key approach to activate the generalizable representation ability of foundation models for downstream tasks. However, existing specialized PEFT methods often fail when applied to large-scale Earth observation tasks, as they are unable to fully handle the multifaceted and unpredictable domain gaps (\eg, spatial, semantic, and frequency shifts) inherent in RS data. To overcome this, we propose CrossEarth-Gate, which introduces two primary contributions. First, we establish a comprehensive RS module toolbox to address multifaceted domain gaps, comprising spatial, semantic, and frequency modules. Second, we develop a Fisher-guided adaptive selection mechanism that operates on this toolbox. This selection is guided by Fisher Information to quantify each module's importance by measuring its contribution to the task-specific gradient flow. It dynamically activates only the most critical modules at the appropriate layers, guiding the gradient flow to maximize adaptation effectiveness and efficiency. Comprehensive experiments validate the efficacy and generalizability of our method, where CrossEarth-Gate achieves state-of-the-art performance across 16 cross-domain benchmarks for RS semantic segmentation. The code of the work will be released.
>
---
#### [replaced 039] Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.19900v2](https://arxiv.org/pdf/2511.19900v2)**

> **作者:** Jiaqi Liu; Kaiwen Xiong; Peng Xia; Yiyang Zhou; Haonian Ji; Lu Feng; Siwei Han; Mingyu Ding; Huaxiu Yao
>
> **摘要:** Vision-language agents have achieved remarkable progress in a variety of multimodal reasoning tasks; however, their learning remains constrained by the limitations of human-annotated supervision. Recent self-rewarding approaches attempt to overcome this constraint by allowing models to act as their own critics or reward providers. Yet, purely text-based self-evaluation struggles to verify complex visual reasoning steps and often suffers from evaluation hallucinations. To address these challenges, inspired by recent advances in tool-integrated reasoning, we propose Agent0-VL, a self-evolving vision-language agent that achieves continual improvement with tool-integrated reasoning. Agent0-VL incorporates tool usage not only into reasoning but also into self-evaluation and self-repair, enabling the model to introspect, verify, and refine its reasoning through evidence-grounded analysis. It unifies two synergistic roles within a single LVLM: a Solver that performs multi-turn tool-integrated reasoning, and a Verifier that generates structured feedback and fine-grained self-rewards through tool-grounded critique. These roles interact through a Self-Evolving Reasoning Cycle, where tool-based verification and reinforcement learning jointly align the reasoning and evaluation distributions for stable self-improvement. Through this zero-external-reward evolution, Agent0-VL aligns its reasoning and verification behaviors without any human annotation or external reward models, achieving continual self-improvement. Experiments on geometric problem solving and visual scientific analysis show that Agent0-VL achieves an 12.5% improvement over the base model. Our code is available at https://github.com/aiming-lab/Agent0.
>
---
#### [replaced 040] Bayesian Neural Networks for One-to-Many Mapping in Image Enhancement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.14265v3](https://arxiv.org/pdf/2501.14265v3)**

> **作者:** Guoxi Huang; Qirui Yang; Ruirui Lin; Zipeng Qi; David Bull; Nantheera Anantrasirichai
>
> **摘要:** In image enhancement tasks, such as low-light and underwater image enhancement, a degraded image can correspond to multiple plausible target images due to dynamic photography conditions. This naturally results in a one-to-many mapping problem. To address this, we propose a Bayesian Enhancement Model (BEM) that incorporates Bayesian Neural Networks (BNNs) to capture data uncertainty and produce diverse outputs. To enable fast inference, we introduce a BNN-DNN framework: a BNN is first employed to model the one-to-many mapping in a low-dimensional space, followed by a Deterministic Neural Network (DNN) that refines fine-grained image details. Extensive experiments on multiple low-light and underwater image enhancement benchmarks demonstrate the effectiveness of our method.
>
---
#### [replaced 041] BRIC: Bridging Kinematic Plans and Physical Control at Test Time
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出BRIC框架，解决扩散模型生成的运动规划与物理控制器间执行偏差问题。针对长期人类运动生成任务，通过测试时动态调整控制器并轻量引导扩散模型，实现物理上合理且连贯的动作执行，显著提升复杂场景下的运动一致性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.20431v2](https://arxiv.org/pdf/2511.20431v2)**

> **作者:** Dohun Lim; Minji Kim; Jaewoon Lim; Sungchan Kim
>
> **备注:** Accepted to AAAI'26
>
> **摘要:** We propose BRIC, a novel test-time adaptation (TTA) framework that enables long-term human motion generation by resolving execution discrepancies between diffusion-based kinematic motion planners and reinforcement learning-based physics controllers. While diffusion models can generate diverse and expressive motions conditioned on text and scene context, they often produce physically implausible outputs, leading to execution drift during simulation. To address this, BRIC dynamically adapts the physics controller to noisy motion plans at test time, while preserving pre-trained skills via a loss function that mitigates catastrophic forgetting. In addition, BRIC introduces a lightweight test-time guidance mechanism that steers the diffusion model in the signal space without updating its parameters. By combining both adaptation strategies, BRIC ensures consistent and physically plausible long-term executions across diverse environments in an effective and efficient manner. We validate the effectiveness of BRIC on a variety of long-term tasks, including motion composition, obstacle avoidance, and human-scene interaction, achieving state-of-the-art performance across all tasks.
>
---
#### [replaced 042] Learning Normals of Noisy Points by Local Gradient-Aware Surface Filtering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.03394v2](https://arxiv.org/pdf/2507.03394v2)**

> **作者:** Qing Li; Huifang Feng; Xun Gong; Yu-Shen Liu
>
> **备注:** Accepted by ICCV 2025. Project page: https://leoqli.github.io/LGSF/
>
> **摘要:** Estimating normals for noisy point clouds is a persistent challenge in 3D geometry processing, particularly for end-to-end oriented normal estimation. Existing methods generally address relatively clean data and rely on supervised priors to fit local surfaces within specific neighborhoods. In this paper, we propose a novel approach for learning normals from noisy point clouds through local gradient-aware surface filtering. Our method projects noisy points onto the underlying surface by utilizing normals and distances derived from an implicit function constrained by local gradients. We start by introducing a distance measurement operator for global surface fitting on noisy data, which integrates projected distances along normals. Following this, we develop an implicit field-based filtering approach for surface point construction, adding projection constraints on these points during filtering. To address issues of over-smoothing and gradient degradation, we further incorporate local gradient consistency constraints, as well as local gradient orientation and aggregation. Comprehensive experiments on normal estimation, surface reconstruction, and point cloud denoising demonstrate the state-of-the-art performance of our method. The source code and trained models are available at https://github.com/LeoQLi/LGSF.
>
---
#### [replaced 043] CAPability: A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Thoroughness
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文针对多模态大模型视觉描述任务，解决传统评估基准无法全面衡量生成描述的准确性和完整性问题。提出CAPability基准，涵盖12维度多视角评估，基于近1.1万张图文数据，引入精度与命中率等指标，揭示模型在问答与描述间的能力差距，推动模型能力精细化提升。**

- **链接: [https://arxiv.org/pdf/2502.14914v4](https://arxiv.org/pdf/2502.14914v4)**

> **作者:** Zhihang Liu; Chen-Wei Xie; Bin Wen; Feiwu Yu; Jixuan Chen; Pandeng Li; Boqiang Zhang; Nianzu Yang; Yinglu Li; Zuan Gao; Yun Zheng; Hongtao Xie
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Visual captioning benchmarks have become outdated with the emergence of modern multimodal large language models (MLLMs), as the brief ground-truth sentences and traditional metrics fail to assess detailed captions effectively. While recent benchmarks attempt to address this by focusing on keyword extraction or object-centric evaluation, they remain limited to vague-view or object-view analyses and incomplete visual element coverage. In this paper, we introduce CAPability, a comprehensive multi-view benchmark for evaluating visual captioning across 12 dimensions spanning six critical views. We curate nearly 11K human-annotated images and videos with visual element annotations to evaluate the generated captions. CAPability stably assesses both the correctness and thoroughness of captions with \textit{precision} and \textit{hit} metrics. By converting annotations to QA pairs, we further introduce a heuristic metric, \textit{know but cannot tell} ($K\bar{T}$), indicating a significant performance gap between QA and caption capabilities. Our work provides a holistic analysis of MLLMs' captioning abilities, as we identify their strengths and weaknesses across various dimensions, guiding future research to enhance specific aspects of their capabilities.
>
---
#### [replaced 044] A Gray-box Attack against Latent Diffusion Model-based Image Editing by Posterior Collapse
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2408.10901v4](https://arxiv.org/pdf/2408.10901v4)**

> **作者:** Zhongliang Guo; Chun Tong Lei; Lei Fang; Shuai Zhao; Yifei Qian; Jingyu Lin; Zeyu Wang; Cunjian Chen; Ognjen Arandjelović; Chun Pong Lau
>
> **备注:** 15 pages, 9 figures, 9 tables
>
> **摘要:** Recent advancements in Latent Diffusion Models (LDMs) have revolutionized image synthesis and manipulation, raising significant concerns about data misappropriation and intellectual property infringement. While adversarial attacks have been extensively explored as a protective measure against such misuse of generative AI, current approaches are severely limited by their heavy reliance on model-specific knowledge and substantial computational costs. Drawing inspiration from the posterior collapse phenomenon observed in VAE training, we propose the Posterior Collapse Attack (PCA), a novel framework for protecting images from unauthorized manipulation. Through comprehensive theoretical analysis and empirical validation, we identify two distinct collapse phenomena during VAE inference: diffusion collapse and concentration collapse. Based on this discovery, we design a unified loss function that can flexibly achieve both types of collapse through parameter adjustment, each corresponding to different protection objectives in preventing image manipulation. Our method significantly reduces dependence on model-specific knowledge by requiring access to only the VAE encoder, which constitutes less than 4\% of LDM parameters. Notably, PCA achieves prompt-invariant protection by operating on the VAE encoder before text conditioning occurs, eliminating the need for empty prompt optimization required by existing methods. This minimal requirement enables PCA to maintain adequate transferability across various VAE-based LDM architectures while effectively preventing unauthorized image editing. Extensive experiments show PCA outperforms existing techniques in protection effectiveness, computational efficiency (runtime and VRAM), and generalization across VAE-based LDM variants. Our code is available at https://github.com/ZhongliangGuo/PosteriorCollapseAttack.
>
---
#### [replaced 045] Sequence-Adaptive Video Prediction in Continuous Streams using Diffusion Noise Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18255v2](https://arxiv.org/pdf/2511.18255v2)**

> **作者:** Sina Mokhtarzadeh Azar; Emad Bahrami; Enrico Pallotta; Gianpiero Francesca; Radu Timofte; Juergen Gall
>
> **摘要:** In this work, we investigate diffusion-based video prediction models, which forecast future video frames, for continuous video streams. In this context, the models observe continuously new training samples, and we aim to leverage this to improve their predictions. We thus propose an approach that continuously adapts a pre-trained diffusion model to a video stream. Since fine-tuning the parameters of a large diffusion model is too expensive, we refine the diffusion noise during inference while keeping the model parameters frozen, allowing the model to adaptively determine suitable sampling noise. We term the approach Sequence Adaptive Video Prediction with Diffusion Noise Optimization (SAVi-DNO). To validate our approach, we introduce a new evaluation setting on the Ego4D dataset, focusing on simultaneous adaptation and evaluation on long continuous videos. Empirical results demonstrate improved performance based on FVD, SSIM, and PSNR metrics on long videos of Ego4D and OpenDV-YouTube, as well as videos of UCF-101 and SkyTimelapse, showcasing SAVi-DNO's effectiveness.
>
---
#### [replaced 046] Diffusion-Denoised Hyperspectral Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.21890v3](https://arxiv.org/pdf/2505.21890v3)**

> **作者:** Sunil Kumar Narayanan; Lingjun Zhao; Lu Gan; Yongsheng Chen
>
> **备注:** Accepted to 3DV 2026
>
> **摘要:** Hyperspectral imaging (HSI) has been widely used in agricultural applications for non-destructive estimation of plant nutrient composition and precise quantification of sample nutritional elements. Recently, 3D reconstruction methods, such as Neural Radiance Field (NeRF), have been used to create implicit neural representations of HSI scenes. This capability enables the rendering of hyperspectral channel compositions at every spatial location, thereby helping localize the target object's nutrient composition both spatially and spectrally. However, it faces limitations in training time and rendering speed. In this paper, we propose Diffusion-Denoised Hyperspectral Gaussian Splatting (DD-HGS), which enhances the state-of-the-art 3D Gaussian Splatting (3DGS) method with wavelength-aware spherical harmonics, a Kullback-Leibler divergence-based spectral loss, and a diffusion-based denoiser to enable 3D explicit reconstruction of the hyperspectral scenes for the entire spectral range. We present extensive evaluations on diverse real-world hyperspectral scenes from the Hyper-NeRF dataset to show the effectiveness of our DD-HGS. The results demonstrate that DD-HGS achieves the new state-of-the-art performance compared to all the previously published methods. Project page: https://dragonpg2000.github.io/DDHGS-website/
>
---
#### [replaced 047] LASER: Lip Landmark Assisted Speaker Detection for Robustness
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2501.11899v2](https://arxiv.org/pdf/2501.11899v2)**

> **作者:** Le Thien Phuc Nguyen; Zhuoran Yu; Yong Jae Lee
>
> **备注:** WACV 2026
>
> **摘要:** Active Speaker Detection (ASD) aims to identify who is speaking in complex visual scenes. While humans naturally rely on lip-audio synchronization, existing ASD models often misclassify non-speaking instances when lip movements and audio are unsynchronized. To address this, we propose Lip landmark Assisted Speaker dEtection for Robustness (LASER), which explicitly incorporates lip landmarks during training to guide the model's attention to speech-relevant regions. Given a face track, LASER extracts visual features and encodes 2D lip landmarks into dense maps. To handle failure cases such as low resolution or occlusion, we introduce an auxiliary consistency loss that aligns lip-aware and face-only predictions, removing the need for landmark detectors at test time. LASER outperforms state-of-the-art models across both in-domain and out-of-domain benchmarks. To further evaluate robustness in realistic conditions, we introduce LASER-bench, a curated dataset of modern video clips with varying levels of background noise. On the high-noise subset, LASER improves mAP by 3.3 and 4.3 points over LoCoNet and TalkNet, respectively, demonstrating strong resilience to real-world acoustic challenges.
>
---
#### [replaced 048] Video-R4: Reinforcing Text-Rich Video Reasoning with Visual Rumination
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17490v3](https://arxiv.org/pdf/2511.17490v3)**

> **作者:** Yolo Y. Tang; Daiki Shimada; Hang Hua; Chao Huang; Jing Bi; Rogerio Feris; Chenliang Xu
>
> **摘要:** Understanding text-rich videos requires reading small, transient textual cues that often demand repeated inspection. Yet most video QA models rely on single-pass perception over fixed frames, leading to hallucinations and failures on fine-grained evidence. Inspired by how humans pause, zoom, and re-read critical regions, we introduce Video-R4 (Reinforcing Text-Rich Video Reasoning with Visual Rumination), a video reasoning LMM that performs visual rumination: iteratively selecting frames, zooming into informative regions, re-encoding retrieved pixels, and updating its reasoning state. We construct two datasets with executable rumination trajectories: Video-R4-CoT-17k for supervised practice and Video-R4-RL-30k for reinforcement learning. We propose a multi-stage rumination learning framework that progressively finetunes a 7B LMM to learn atomic and mixing visual operations via SFT and GRPO-based RL. Video-R4-7B achieves state-of-the-art results on M4-ViteVQA and further generalizes to multi-page document QA, slides QA, and generic video QA, demonstrating that iterative rumination is an effective paradigm for pixel-grounded multimodal reasoning. Project Page: https://yunlong10.github.io/Video-R4/
>
---
#### [replaced 049] Probabilistic Robustness for Free? Revisiting Training via a Benchmark
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.01724v2](https://arxiv.org/pdf/2511.01724v2)**

> **作者:** Yi Zhang; Zheng Wang; Zhen Chen; Wenjie Ruan; Qing Guo; Siddartha Khastgir; Carsten Maple; Xingyu Zhao
>
> **摘要:** Deep learning models are notoriously vulnerable to imperceptible perturbations. Most existing research centers on adversarial robustness (AR), which evaluates models under worst-case scenarios by examining the existence of deterministic adversarial examples (AEs). In contrast, probabilistic robustness (PR) adopts a statistical perspective, measuring the probability that predictions remain correct under stochastic perturbations. While PR is widely regarded as a practical complement to AR, dedicated training methods for improving PR are still relatively underexplored, albeit with emerging progress. Among the few PR-targeted training methods, we identify three limitations: i non-comparable evaluation protocols; ii limited comparisons to strong AT baselines despite anecdotal PR gains from AT; and iii no unified framework to compare the generalization of these methods. Thus, we introduce PRBench, the first benchmark dedicated to evaluating improvements in PR achieved by different robustness training methods. PRBench empirically compares most common AT and PR-targeted training methods using a comprehensive set of metrics, including clean accuracy, PR and AR performance, training efficiency, and generalization error (GE). We also provide theoretical analysis on the GE of PR performance across different training methods. Main findings revealed by PRBench include: AT methods are more versatile than PR-targeted training methods in terms of improving both AR and PR performance across diverse hyperparameter settings, while PR-targeted training methods consistently yield lower GE and higher clean accuracy. A leaderboard comprising 222 trained models across 7 datasets and 10 model architectures is publicly available at https://tmpspace.github.io/PRBenchLeaderboard/.
>
---
#### [replaced 050] Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.19386v2](https://arxiv.org/pdf/2505.19386v2)**

> **作者:** Nate Gillman; Charles Herrmann; Michael Freeman; Daksh Aggarwal; Evan Luo; Deqing Sun; Chen Sun
>
> **备注:** Camera ready version (NeurIPS 2025). Code and interactive demos at https://force-prompting.github.io/
>
> **摘要:** Recent advances in video generation models have sparked interest in world models capable of simulating realistic environments. While navigation has been well-explored, physically meaningful interactions that mimic real-world forces remain largely understudied. In this work, we investigate using physical forces as a control signal for video generation and propose force prompts which enable users to interact with images through both localized point forces, such as poking a plant, and global wind force fields, such as wind blowing on fabric. We demonstrate that these force prompts can enable videos to respond realistically to physical control signals by leveraging the visual and motion prior in the original pretrained model, without using any 3D asset or physics simulator at inference. The primary challenge of force prompting is the difficulty in obtaining high quality paired force-video training data, both in the real world due to the difficulty of obtaining force signals, and in synthetic data due to limitations in the visual quality and domain diversity of physics simulators. Our key finding is that video generation models can generalize remarkably well when adapted to follow physical force conditioning from videos synthesized by Blender, even with limited demonstrations of few objects. Our method can generate videos which simulate forces across diverse geometries, settings, and materials. We also try to understand the source of this generalization and perform ablations that reveal two key elements: visual diversity and the use of specific text keywords during training. Our approach is trained on only around 15k training examples for a single day on four A100 GPUs, and outperforms existing methods on force adherence and physics realism, bringing world models closer to real-world physics interactions. We release all datasets, code, weights, and interactive video demos at our project page.
>
---
#### [replaced 051] Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.17358v4](https://arxiv.org/pdf/2503.17358v4)**

> **作者:** Jerred Chen; Ronald Clark
>
> **备注:** Project page: https://jerredchen.github.io/image-as-imu/
>
> **摘要:** In many robotics and VR/AR applications, fast camera motions lead to a high level of motion blur, causing existing camera pose estimation methods to fail. In this work, we propose a novel framework that leverages motion blur as a rich cue for motion estimation rather than treating it as an unwanted artifact. Our approach works by predicting a dense motion flow field and a monocular depth map directly from a single motion-blurred image. We then recover the instantaneous camera velocity by solving a linear least squares problem under the small motion assumption. In essence, our method produces an IMU-like measurement that robustly captures fast and aggressive camera movements. To train our model, we construct a large-scale dataset with realistic synthetic motion blur derived from ScanNet++v2 and further refine our model by training end-to-end on real data using our fully differentiable pipeline. Extensive evaluations on real-world benchmarks demonstrate that our method achieves state-of-the-art angular and translational velocity estimates, outperforming current methods like MASt3R and COLMAP.
>
---
#### [replaced 052] SARVLM: A Vision Language Foundation Model for Semantic Understanding and Target Recognition in SAR Imagery
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.22665v2](https://arxiv.org/pdf/2510.22665v2)**

> **作者:** Qiwei Ma; Zhiyu Wang; Wang Liu; Xukun Lu; Bin Deng; Puhong Duan; Xudong Kang; Shutao Li
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** Synthetic Aperture Radar (SAR) is a crucial imaging modality thanks to its all-weather capability. Although recent advances in self-supervised learning and masked image modeling (MIM) have enabled SAR foundation models, these methods largely emphasize low-level visual features and often overlook multimodal alignment and zero-shot target recognition in SAR imagery. To address this, we construct SARVLM-1M, a large-scale vision-language dataset with over one million image-text pairs aggregated from existing datasets. We further propose a domain transfer training strategy to mitigate the large gap between natural and SAR imagery. Building on this, we develop SARVLM, the first vision language foundation model (VLM) tailored to SAR, comprising SARCLIP and SARCap. SARVLM is trained with a vision-language contrastive objective under the proposed domain transfer strategy, bridging SAR imagery and textual descriptions. Extensive experiments on image text retrieval, zero-shot classification, semantic localization, and imagery captioning demonstrate that SARVLM delivers superior feature extraction and interpretation, outperforming state-of-the-art VLMs and advancing SAR semantic understanding. Code and datasets will be released soon.
>
---
#### [replaced 053] Restoration-Oriented Video Frame Interpolation with Region-Distinguishable Priors from SAM
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2312.15868v2](https://arxiv.org/pdf/2312.15868v2)**

> **作者:** Yan Han; Xiaogang Xu; Yingqi Lin; Jiafei Wu; Zhe Liu; Ming-Hsuan Yang
>
> **备注:** Code will be released
>
> **摘要:** In existing restoration-oriented Video Frame Interpolation (VFI) approaches, the motion estimation between neighboring frames plays a crucial role. However, the estimation accuracy in existing methods remains a challenge, primarily due to the inherent ambiguity in identifying corresponding areas in adjacent frames for interpolation. Therefore, enhancing accuracy by distinguishing different regions before motion estimation is of utmost importance. In this paper, we introduce a novel solution involving the utilization of open-world segmentation models, e.g., SAM2 (Segment Anything Model2) for frames, to derive Region-Distinguishable Priors (RDPs) in different frames. These RDPs are represented as spatial-varying Gaussian mixtures, distinguishing an arbitrary number of areas with a unified modality. RDPs can be integrated into existing motion-based VFI methods to enhance features for motion estimation, facilitated by our designed play-and-plug Hierarchical Region-aware Feature Fusion Module (HRFFM). HRFFM incorporates RDP into various hierarchical stages of VFI's encoder, using RDP-guided Feature Normalization (RDPFN) in a residual learning manner. With HRFFM and RDP, the features within VFI's encoder exhibit similar representations for matched regions in neighboring frames, thus improving the synthesis of intermediate frames. Extensive experiments demonstrate that HRFFM consistently enhances VFI performance across various scenes.
>
---
#### [replaced 054] VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.11473v2](https://arxiv.org/pdf/2510.11473v2)**

> **作者:** Qing Li; Huifang Feng; Xun Gong; Yu-Shen Liu
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** 3D Gaussian Splatting has recently emerged as an efficient solution for high-quality and real-time novel view synthesis. However, its capability for accurate surface reconstruction remains underexplored. Due to the discrete and unstructured nature of Gaussians, supervision based solely on image rendering loss often leads to inaccurate geometry and inconsistent multi-view alignment. In this work, we propose a novel method that enhances the geometric representation of 3D Gaussians through view alignment (VA). Specifically, we incorporate edge-aware image cues into the rendering loss to improve surface boundary delineation. To enforce geometric consistency across views, we introduce a visibility-aware photometric alignment loss that models occlusions and encourages accurate spatial relationships among Gaussians. To further mitigate ambiguities caused by lighting variations, we incorporate normal-based constraints to refine the spatial orientation of Gaussians and improve local surface estimation. Additionally, we leverage deep image feature embeddings to enforce cross-view consistency, enhancing the robustness of the learned geometry under varying viewpoints and illumination. Extensive experiments on standard benchmarks demonstrate that our method achieves state-of-the-art performance in both surface reconstruction and novel view synthesis. The source code is available at https://github.com/LeoQLi/VA-GS.
>
---
#### [replaced 055] Class-Independent Increment: An Efficient Approach for Multi-label Class-Incremental Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.00515v2](https://arxiv.org/pdf/2503.00515v2)**

> **作者:** Chenhao Ding; Songlin Dong; Zhengdong Zhou; Jizhou Han; Qiang Wang; Yuhang He; Yihong Gong
>
> **摘要:** Current research on class-incremental learning primarily focuses on single-label classification tasks. However, real-world applications often involve multi-label scenarios, such as image retrieval and medical imaging. Therefore, this paper focuses on the challenging yet practical multi-label class-incremental learning (MLCIL) problem. In addition to the challenge of catastrophic forgetting, MLCIL encounters issues related to feature confusion, encompassing inter-session and intra-feature confusion. To address these problems, we propose a novel MLCIL approach called class-independent increment (CLIN). Specifically, in contrast to existing methods that extract image-level features, we propose a class-independent incremental network (CINet) to extract multiple class-level embeddings for multi-label samples. It learns and preserves the knowledge of different classes by constructing class-specific tokens. On this basis, we develop two novel loss functions, optimizing the learning of class-specific tokens and class-level embeddings, respectively. These losses aim to distinguish between new and old classes, further alleviating the problem of feature confusion. Extensive experiments on MS-COCO and PASCAL VOC datasets demonstrate the effectiveness of our method for improving recognition performance and mitigating forgetting on various MLCIL tasks.
>
---
#### [replaced 056] Multi-scale Temporal Prediction via Incremental Generation and Multi-agent Collaboration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.17429v3](https://arxiv.org/pdf/2509.17429v3)**

> **作者:** Zhitao Zeng; Guojian Yuan; Junyuan Mao; Yuxuan Wang; Xiaoshuang Jia; Yueming Jin
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Accurate temporal prediction is the bridge between comprehensive scene understanding and embodied artificial intelligence. However, predicting multiple fine-grained states of a scene at multiple temporal scales is difficult for vision-language models. We formalize the Multi-Scale Temporal Prediction (MSTP) task in general and surgical scenes by decomposing multi-scale into two orthogonal dimensions: the temporal scale, forecasting states of humans and surgery at varying look-ahead intervals, and the state scale, modeling a hierarchy of states in general and surgical scenes. For example, in general scenes, states of contact relationships are finer-grained than states of spatial relationships. In surgical scenes, medium-level steps are finer-grained than high-level phases yet remain constrained by their encompassing phase. To support this unified task, we introduce the first MSTP Benchmark, featuring synchronized annotations across multiple state scales and temporal scales. We further propose a method, Incremental Generation and Multi-agent Collaboration (IG-MC), which integrates two key innovations. First, we present a plug-and-play incremental generation module that continuously synthesizes up-to-date visual previews at expanding temporal scales to inform multiple decision-making agents, keeping decisions and generated visuals synchronized and preventing performance degradation as look-ahead intervals lengthen. Second, we present a decision-driven multi-agent collaboration framework for multi-state prediction, comprising generation, initiation, and multi-state assessment agents that dynamically trigger and evaluate prediction cycles to balance global coherence and local fidelity.
>
---
#### [replaced 057] Disentangled Geometric Alignment with Adaptive Contrastive Perturbation for Reliable Domain Transfer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.15241v2](https://arxiv.org/pdf/2505.15241v2)**

> **作者:** Emma Collins; Myungseo wong; Kim Yun; Finn Kingston; Hana Satou
>
> **摘要:** Despite progress in geometry-aware domain adaptation, current methods such as GAMA still suffer from two unresolved issues: (1) insufficient disentanglement of task-relevant and task-irrelevant manifold dimensions, and (2) rigid perturbation schemes that ignore per-class alignment asymmetries. To address this, we propose GAMA++, a novel framework that introduces (i) latent space disentanglement to isolate label-consistent manifold directions from nuisance factors, and (ii) an adaptive contrastive perturbation strategy that tailors both on- and off-manifold exploration to class-specific manifold curvature and alignment discrepancy. We further propose a cross-domain contrastive consistency loss that encourages local semantic clusters to align while preserving intra-domain diversity. Our method achieves state-of-the-art results on DomainNet, Office-Home, and VisDA benchmarks under both standard and few-shot settings, with notable improvements in class-level alignment fidelity and boundary robustness. GAMA++ sets a new standard for semantic geometry alignment in transfer learning.
>
---
#### [replaced 058] ReMatch: Boosting Representation through Matching for Multimodal Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19278v2](https://arxiv.org/pdf/2511.19278v2)**

> **作者:** Qianying Liu; Xiao Liang; Zhiqiang Zhang; Zhongfei Qing; Fengfan Zhou; Yibo Chen; Xu Tang; Yao Hu; Paul Henderson
>
> **摘要:** We present ReMatch, a framework that leverages the generative strength of MLLMs for multimodal retrieval. Previous approaches treated an MLLM as a simple encoder, ignoring its generative nature, and under-utilising its compositional reasoning and world knowledge. We instead train the embedding MLLM end-to-end with a chat-style generative matching stage. The matching stage uses the same MLLM to autoregressively decide relevance from multi-view inputs, including both raw data and its own projected embeddings for each query and document. It provides instance-wise discrimination supervision that complements a standard contrastive loss, offering stronger gradients on hard negatives and preserving the compositional strengths of the original MLLM. To obtain semantically richer multimodal embeddings, we use multiple learnable tokens to augment each input, generating fine-grained contextual, mutually orthogonal embeddings with low inference cost. Leveraging our established high-performance baseline,we assemble the ideas mentioned above into a powerful training recipe and achieve a new state-of-the-art on the Massive Multimodal Embedding Benchmark (MMEB). Our experiments show particularly strong zero-shot generalization results on five datasets, highlighting the robustness and transferability of ReMatch.
>
---
#### [replaced 059] Modular, On-Site Solutions with Lightweight Anomaly Detection for Sustainable Nutrient Management in Agriculture
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.12247v2](https://arxiv.org/pdf/2509.12247v2)**

> **作者:** Abigail R. Cohen; Yuming Sun; Zhihao Qin; Harsh S. Muriki; Zihao Xiao; Yeonju Lee; Matthew Housley; Andrew F. Sharkey; Rhuanito S. Ferrarezi; Jing Li; Lu Gan; Yongsheng Chen
>
> **摘要:** Efficient nutrient management is critical for crop growth and sustainable resource consumption (e.g., nitrogen, energy). Current approaches require lengthy analyses, preventing real-time optimization; similarly, imaging facilitates rapid phenotyping but can be computationally intensive, preventing deployment under resource constraints. This study proposes a flexible, tiered pipeline for anomaly detection and status estimation (fresh weight, dry mass, and tissue nutrients), including a comprehensive energy analysis of approaches that span the efficiency-accuracy spectrum. Using a nutrient depletion experiment with three treatments (T1-100%, T2-50%, and T3-25% fertilizer strength) and multispectral imaging (MSI), we developed a hierarchical pipeline using an autoencoder (AE) for early warning. Further, we compared two status estimation modules of different complexity for more detailed analysis: vegetation index (VI) features with machine learning (Random Forest, RF) and raw whole-image deep learning (Vision Transformer, ViT). Results demonstrated high-efficiency anomaly detection (73% net detection of T3 samples 9 days after transplanting) at substantially lower energy than embodied energy in wasted nitrogen. The state estimation modules show trade-offs, with ViT outperforming RF on phosphorus and calcium estimation (R2 0.61 vs. 0.58, 0.48 vs. 0.35) at higher energy cost. With our modular pipeline, this work opens opportunities for edge diagnostics and practical opportunities for agricultural sustainability.
>
---
#### [replaced 060] Not All Splits Are Equal: Rethinking Attribute Generalization Across Unrelated Categories
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.06998v2](https://arxiv.org/pdf/2509.06998v2)**

> **作者:** Liviu Nicolae Fircă; Antonio Bărbălau; Dan Oneata; Elena Burceanu
>
> **备注:** Accepted at NeurIPS 2025 Workshop: CauScien - Uncovering Causality in Science and NeurIPS 2025 Workshop: Reliable ML from Unreliable Data
>
> **摘要:** Can models generalize attribute knowledge across semantically and perceptually dissimilar categories? While prior work has addressed attribute prediction within narrow taxonomic or visually similar domains, it remains unclear whether current models can abstract attributes and apply them to conceptually distant categories. This work presents the first explicit evaluation for the robustness of the attribute prediction task under such conditions, testing whether models can correctly infer shared attributes between unrelated object types: e.g., identifying that the attribute "has four legs" is common to both "dogs" and "chairs". To enable this evaluation, we introduce train-test split strategies that progressively reduce correlation between training and test sets, based on: LLM-driven semantic grouping, embedding similarity thresholding, embedding-based clustering, and supercategory-based partitioning using ground-truth labels. Results show a sharp drop in performance as the correlation between training and test categories decreases, indicating strong sensitivity to split design. Among the evaluated methods, clustering yields the most effective trade-off, reducing hidden correlations while preserving learnability. These findings offer new insights into the limitations of current representations and inform future benchmark construction for attribute reasoning.
>
---
#### [replaced 061] ConceptGuard: Proactive Safety in Text-and-Image-to-Video Generation through Multimodal Risk Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.18780v3](https://arxiv.org/pdf/2511.18780v3)**

> **作者:** Ruize Ma; Minghong Cai; Yilei Jiang; Jiaming Han; Yi Feng; Yingshui Tan; Xiaoyong Zhu; Bo Zhang; Bo Zheng; Xiangyu Yue
>
> **摘要:** Recent progress in video generative models has enabled the creation of high-quality videos from multimodal prompts that combine text and images. While these systems offer enhanced controllability, they also introduce new safety risks, as harmful content can emerge from individual modalities or their interaction. Existing safety methods are often text-only, require prior knowledge of the risk category, or operate as post-generation auditors, struggling to proactively mitigate such compositional, multimodal risks. To address this challenge, we present ConceptGuard, a unified safeguard framework for proactively detecting and mitigating unsafe semantics in multimodal video generation. ConceptGuard operates in two stages: First, a contrastive detection module identifies latent safety risks by projecting fused image-text inputs into a structured concept space; Second, a semantic suppression mechanism steers the generative process away from unsafe concepts by intervening in the prompt's multimodal conditioning. To support the development and rigorous evaluation of this framework, we introduce two novel benchmarks: ConceptRisk, a large-scale dataset for training on multimodal risks, and T2VSafetyBench-TI2V, the first benchmark adapted from T2VSafetyBench for the Text-and-Image-to-Video (TI2V) safety setting. Comprehensive experiments on both benchmarks show that ConceptGuard consistently outperforms existing baselines, achieving state-of-the-art results in both risk detection and safe video generation. Our code is available at https://github.com/Ruize-Ma/ConceptGuard.
>
---
#### [replaced 062] Towards Consistent and Controllable Image Synthesis for Face Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.02465v3](https://arxiv.org/pdf/2502.02465v3)**

> **作者:** Mengting Wei; Tuomas Varanka; Yante Li; Xingxun Jiang; Huai-Qian Khor; Guoying Zhao
>
> **摘要:** Face editing methods, essential for tasks like virtual avatars, digital human synthesis and identity preservation, have traditionally been built upon GAN-based techniques, while recent focus has shifted to diffusion-based models due to their success in image reconstruction. However, diffusion models still face challenges in controlling specific attributes and preserving the consistency of other unchanged attributes especially the identity characteristics. To address these issues and facilitate more convenient editing of face images, we propose a novel approach that leverages the power of Stable-Diffusion (SD) models and crude 3D face models to control the lighting, facial expression and head pose of a portrait photo. We observe that this task essentially involves the combinations of target background, identity and face attributes aimed to edit. We strive to sufficiently disentangle the control of these factors to enable consistency of face editing. Specifically, our method, coined as RigFace, contains: 1) A Spatial Attribute Encoder that provides presise and decoupled conditions of background, pose, expression and lighting; 2) A high-consistency FaceFusion method that transfers identity features from the Identity Encoder to the denoising UNet of a pre-trained SD model; 3) An Attribute Rigger that injects those conditions into the denoising UNet. Our model achieves comparable or even superior performance in both identity preservation and photorealism compared to existing face editing models.
>
---
#### [replaced 063] Uncovering and Mitigating Destructive Multi-Embedding Attacks in Deepfake Proactive Forensics
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.17247v2](https://arxiv.org/pdf/2508.17247v2)**

> **作者:** Lixin Jia; Haiyang Sun; Zhiqing Guo; Yunfeng Diao; Dan Ma; Gaobo Yang
>
> **摘要:** With the rapid evolution of deepfake technologies and the wide dissemination of digital media, personal privacy is facing increasingly serious security threats. Deepfake proactive forensics, which involves embedding imperceptible watermarks to enable reliable source tracking, serves as a crucial defense against these threats. Although existing methods show strong forensic ability, they rely on an idealized assumption of single watermark embedding, which proves impractical in real-world scenarios. In this paper, we formally define and demonstrate the existence of Multi-Embedding Attacks (MEA) for the first time. When a previously protected image undergoes additional rounds of watermark embedding, the original forensic watermark can be destroyed or removed, rendering the entire proactive forensic mechanism ineffective. To address this vulnerability, we propose a general training paradigm named Adversarial Interference Simulation (AIS). Rather than modifying the network architecture, AIS explicitly simulates MEA scenarios during fine-tuning and introduces a resilience-driven loss function to enforce the learning of sparse and stable watermark representations. Our method enables the model to maintain the ability to extract the original watermark correctly even after a second embedding. Extensive experiments demonstrate that our plug-and-play AIS training paradigm significantly enhances the robustness of various existing methods against MEA.
>
---
#### [replaced 064] RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.17159v5](https://arxiv.org/pdf/2502.17159v5)**

> **作者:** Fanhu Zeng; Haiyang Guo; Fei Zhu; Li Shen; Hao Tang
>
> **备注:** NeurIPS 2025 (Spotlight) Fix some typos
>
> **摘要:** Fine-tuning pre-trained models with custom data leads to numerous expert models on specific tasks. Merging models into one universal model to empower multi-task ability refraining from data leakage has gained popularity. With the expansion in data and model size, parameter-efficient tuning becomes the common practice for obtaining task-specific models efficiently. However, few methods are dedicated to efficient merging, and existing methods designed for full fine-tuning merging fail under efficient merging. To address the issue, we analyze from low-rank decomposition and reveal that direction robustness during merging is crucial for merging efficient modules. We furthermore uncover that compensating for the gap between stark singular values contributes to direction robustness. Therefore, we propose RobustMerge, a training-free parameter-efficient merging method with complementary parameter adaptation to maintain direction robustness. Specifically, we (1) prune parameters and scale coefficients from inter-parameter relation for singular values to maintain direction stability away from task interference, and (2) perform cross-task normalization to enhance unseen task generalization. We establish a benchmark consisting of diverse multimodal tasks, on which we conduct experiments to certify the outstanding performance and generalizability of our method. Additional studies and extensive analyses further showcase the effectiveness. Code is available at https://github.com/AuroraZengfh/RobustMerge.
>
---
#### [replaced 065] WeatherDiffusion: Controllable Weather Editing in Intrinsic Space
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.06982v3](https://arxiv.org/pdf/2508.06982v3)**

> **作者:** Yixin Zhu; Zuoliang Zhu; Jian Yang; Miloš Hašan; Jin Xie; Beibei Wang
>
> **摘要:** We present WeatherDiffusion, a diffusion-based framework for controllable weather editing in intrinsic space. Our framework includes two components based on diffusion priors: an inverse renderer that estimates material properties, scene geometry, and lighting as intrinsic maps from an input image, and a forward renderer that utilizes these geometry and material maps along with a text prompt that describes specific weather conditions to generate a final image. The intrinsic maps enhance controllability compared to traditional pixel-space editing approaches.We propose an intrinsic map-aware attention mechanism that improves spatial correspondence and decomposition quality in large outdoor scenes. For forward rendering, we leverage CLIP-space interpolation of weather prompts to achieve fine-grained weather control. We also introduce a synthetic and a real-world dataset, containing 38k and 18k images under various weather conditions, each with intrinsic map annotations. WeatherDiffusion outperforms state-of-the-art pixel-space editing approaches, weather restoration methods, and rendering-based methods, showing promise for downstream tasks such as autonomous driving, enhancing the robustness of detection and segmentation in challenging weather scenarios.
>
---
#### [replaced 066] LightMem: Lightweight and Efficient Memory-Augmented Generation
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文针对大语言模型在动态环境中难以有效利用历史交互信息的问题，提出轻量高效的LightMem记忆系统。受人类记忆模型启发，将记忆分为感知、短时和长时三阶段，通过压缩、主题感知与离线更新提升效率。实验表明，LightMem显著提升问答准确率，大幅降低计算与调用开销。**

- **链接: [https://arxiv.org/pdf/2510.18866v3](https://arxiv.org/pdf/2510.18866v3)**

> **作者:** Jizhan Fang; Xinle Deng; Haoming Xu; Ziyan Jiang; Yuqi Tang; Ziwen Xu; Shumin Deng; Yunzhi Yao; Mengru Wang; Shuofei Qiao; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** Despite their remarkable capabilities, Large Language Models (LLMs) struggle to effectively leverage historical interaction information in dynamic and complex environments. Memory systems enable LLMs to move beyond stateless interactions by introducing persistent information storage, retrieval, and utilization mechanisms. However, existing memory systems often introduce substantial time and computational overhead. To this end, we introduce a new memory system called LightMem, which strikes a balance between the performance and efficiency of memory systems. Inspired by the Atkinson-Shiffrin model of human memory, LightMem organizes memory into three complementary stages. First, cognition-inspired sensory memory rapidly filters irrelevant information through lightweight compression and groups information according to their topics. Next, topic-aware short-term memory consolidates these topic-based groups, organizing and summarizing content for more structured access. Finally, long-term memory with sleep-time update employs an offline procedure that decouples consolidation from online inference. On LongMemEval and LoCoMo, using GPT and Qwen backbones, LightMem consistently surpasses strong baselines, improving QA accuracy by up to 7.7% / 29.3%, reducing total token usage by up to 38x / 20.9x and API calls by up to 30x / 55.5x, while purely online test-time costs are even lower, achieving up to 106x / 117x token reduction and 159x / 310x fewer API calls. The code is available at https://github.com/zjunlp/LightMem.
>
---
#### [replaced 067] Filter Like You Test: Data-Driven Data Filtering for CLIP Pretraining
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.08805v3](https://arxiv.org/pdf/2503.08805v3)**

> **作者:** Mikey Shechter; Yair Carmon
>
> **摘要:** We introduce Filter Like You Test (FLYT), an algorithm for curating large-scale vision-language datasets that learns the usefulness of each data point as a pretraining example. FLYT trains a scoring model that learns to weigh each example's features using gradient signals from downstream tasks training sets. Based on FLYT, we implement Mixing-FLYT (M-FLYT), which takes the per-example scores generated by different scoring methods as features, and learns to unify them into a single score. FLYT naturally produces a distribution over the training examples, which we leverage through Soft Cap Sampling (SCS), a strategy for obtaining a filtered pretraining dataset from per-example probabilities that samples examples while preventing over-representation through a repetition penalty. Using these methods, we achieve 40.1% ImageNet zero-shot accuracy on the DataComp medium scale filtering benchmark, a 2% absolute accuracy increase over all previous results and a 5.5% increase over results that - like us - use only public resources. Our approach also yields 37.7\% on the average of 38 DataComp evaluation tasks, outperforming previous public-resource approaches by 0.4\%.
>
---
#### [replaced 068] XYZCylinder: Towards Compatible Feed-Forward 3D Gaussian Splatting for Driving Scenes via Unified Cylinder Lifting Method
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.07856v2](https://arxiv.org/pdf/2510.07856v2)**

> **作者:** Haochen Yu; Qiankun Liu; Hongyuan Liu; Jianfei Jiang; Juntao Lyu; Jiansheng Chen; Huimin Ma
>
> **备注:** Feed-Forward, 3D Gaussian Splatting, Project page: https://yuyuyu223.github.io/XYZCYlinder-projectpage/
>
> **摘要:** Feed-forward paradigms for 3D reconstruction have become a focus of recent research, which learn implicit, fixed view transformations to generate a single scene representation. However, their application to complex driving scenes reveals significant limitations. Two core challenges are responsible for this performance gap. First, the reliance on a fixed view transformation hinders compatibility to varying camera configurations. Second, the inherent difficulty of learning complex driving scenes from sparse 360° views with minimal overlap compromises the final reconstruction fidelity. To handle these difficulties, we introduce XYZCylinder, a novel method built upon a unified cylinder lifting method that integrates camera modeling and feature lifting. To tackle the compatibility problem, we design a Unified Cylinder Camera Modeling (UCCM) strategy. This strategy explicitly models projection parameters to unify diverse camera setups, thus bypassing the need for learning viewpoint-dependent correspondences. To improve the reconstruction accuracy, we propose a hybrid representation with several dedicated modules based on newly designed Cylinder Plane Feature Group (CPFG) to lift 2D image features to 3D space. Extensive evaluations confirm that XYZCylinder not only achieves state-of-the-art performance under different evaluation settings but also demonstrates remarkable compatibility in entirely new scenes with different camera settings in a zero-shot manner. Project page: \href{https://yuyuyu223.github.io/XYZCYlinder-projectpage/}{here}
>
---
#### [replaced 069] Contrast-Prior Enhanced Duality for Mask-Free Shadow Removal
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.21949v2](https://arxiv.org/pdf/2507.21949v2)**

> **作者:** Jiyu Wu; Yifan Liu; Jiancheng Huang; Mingfu Yan; Shifeng Chen
>
> **备注:** There are unresolved authorship disputes related to this submission, and the current version does not reflect an agreed authorship list
>
> **摘要:** Existing shadow removal methods often rely on shadow masks, which are challenging to acquire in real-world scenarios. Exploring intrinsic image cues, such as local contrast information, presents a potential alternative for guiding shadow removal in the absence of explicit masks. However, the cue's inherent ambiguity becomes a critical limitation in complex scenes, where it can fail to distinguish true shadows from low-reflectance objects and intricate background textures. To address this motivation, we propose the Adaptive Gated Dual-Branch Attention (AGBA) mechanism. AGBA dynamically filters and re-weighs the contrast prior to effectively disentangle shadow features from confounding visual elements. Furthermore, to tackle the persistent challenge of restoring soft shadow boundaries and fine-grained details, we introduce a diffusion-based Frequency-Contrast Fusion Network (FCFN) that leverages high-frequency and contrast cues to guide the generative process. Extensive experiments demonstrate that our method achieves state-of-the-art results among mask-free approaches while maintaining competitive performance relative to mask-based methods.
>
---
#### [replaced 070] SKEL-CF: Coarse-to-Fine Biomechanical Skeleton and Surface Mesh Recovery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20157v2](https://arxiv.org/pdf/2511.20157v2)**

> **作者:** Da Li; Jiping Jin; Xuanlong Yu; Wei Liu; Xiaodong Cun; Kai Chen; Rui Fan; Jiangang Kong; Xi Shen
>
> **备注:** Project page: https://pokerman8.github.io/SKEL-CF/
>
> **摘要:** Parametric 3D human models such as SMPL have driven significant advances in human pose and shape estimation, yet their simplified kinematics limit biomechanical realism. The recently proposed SKEL model addresses this limitation by re-rigging SMPL with an anatomically accurate skeleton. However, estimating SKEL parameters directly remains challenging due to limited training data, perspective ambiguities, and the inherent complexity of human articulation. We introduce SKEL-CF, a coarse-to-fine framework for SKEL parameter estimation. SKEL-CF employs a transformer-based encoder-decoder architecture, where the encoder predicts coarse camera and SKEL parameters, and the decoder progressively refines them in successive layers. To ensure anatomically consistent supervision, we convert the existing SMPL-based dataset 4DHuman into a SKEL-aligned version, 4DHuman-SKEL, providing high-quality training data for SKEL estimation. In addition, to mitigate depth and scale ambiguities, we explicitly incorporate camera modeling into the SKEL-CF pipeline and demonstrate its importance across diverse viewpoints. Extensive experiments validate the effectiveness of the proposed design. On the challenging MOYO dataset, SKEL-CF achieves 85.0 MPJPE / 51.4 PA-MPJPE, significantly outperforming the previous SKEL-based state-of-the-art HSMR (104.5 / 79.6). These results establish SKEL-CF as a scalable and anatomically faithful framework for human motion analysis, bridging the gap between computer vision and biomechanics. Our implementation is available on the project page: https://pokerman8.github.io/SKEL-CF/.
>
---
#### [replaced 071] STARFlow-V: End-to-End Video Generative Modeling with Normalizing Flows
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.20462v2](https://arxiv.org/pdf/2511.20462v2)**

> **作者:** Jiatao Gu; Ying Shen; Tianrong Chen; Laurent Dinh; Yuyang Wang; Miguel Angel Bautista; David Berthelot; Josh Susskind; Shuangfei Zhai
>
> **备注:** 21 pages, 9 figures. Code and samples are available at https://github.com/apple/ml-starflow
>
> **摘要:** Normalizing flows (NFs) are end-to-end likelihood-based generative models for continuous data, and have recently regained attention with encouraging progress on image generation. Yet in the video generation domain, where spatiotemporal complexity and computational cost are substantially higher, state-of-the-art systems almost exclusively rely on diffusion-based models. In this work, we revisit this design space by presenting STARFlow-V, a normalizing flow-based video generator with substantial benefits such as end-to-end learning, robust causal prediction, and native likelihood estimation. Building upon the recently proposed STARFlow, STARFlow-V operates in the spatiotemporal latent space with a global-local architecture which restricts causal dependencies to a global latent space while preserving rich local within-frame interactions. This eases error accumulation over time, a common pitfall of standard autoregressive diffusion model generation. Additionally, we propose flow-score matching, which equips the model with a light-weight causal denoiser to improve the video generation consistency in an autoregressive fashion. To improve the sampling efficiency, STARFlow-V employs a video-aware Jacobi iteration scheme that recasts inner updates as parallelizable iterations without breaking causality. Thanks to the invertible structure, the same model can natively support text-to-video, image-to-video as well as video-to-video generation tasks. Empirically, STARFlow-V achieves strong visual fidelity and temporal consistency with practical sampling throughput relative to diffusion-based baselines. These results present the first evidence, to our knowledge, that NFs are capable of high-quality autoregressive video generation, establishing them as a promising research direction for building world models. Code and generated samples are available at https://github.com/apple/ml-starflow.
>
---
#### [replaced 072] LTD: Low Temperature Distillation for Gradient Masking-free Adversarial Training
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2111.02331v4](https://arxiv.org/pdf/2111.02331v4)**

> **作者:** Erh-Chung Chen; Che-Rung Lee
>
> **摘要:** Adversarial training is a widely adopted strategy to bolster the robustness of neural network models against adversarial attacks. This paper revisits the fundamental assumptions underlying image classification and suggests that representing data as one-hot labels is a key factor that leads to vulnerabilities. However, in real-world datasets, data ambiguity often arises, with samples exhibiting characteristics of multiple classes, rendering one-hot label representations imprecise. To address this, we introduce a novel approach, Low-Temperature Distillation (LTD), designed to refine label representations. Unlike previous approaches, LTD incorporates a relatively low temperature in the teacher model, while maintaining a fixed temperature for the student model during both training and inference. This strategy not only refines assumptions about data distribution but also strengthens model robustness and avoids the gradient masking problem commonly encountered in defensive distillation. Experimental results demonstrate the efficacy of the proposed method when combined with existing frameworks, achieving robust accuracy rates of 58.19%, 31.13%, and 42.08% on the CIFAR-10, CIFAR-100, and ImageNet datasets, respectively, without the need for additional data.
>
---
#### [replaced 073] Adapting Segment Anything Model for Power Transmission Corridor Hazard Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.22105v2](https://arxiv.org/pdf/2505.22105v2)**

> **作者:** Hang Chen; Maoyuan Ye; Peng Yang; Haibin He; Juhua Liu; Bo Du
>
> **摘要:** Power transmission corridor hazard segmentation (PTCHS) aims to separate transmission equipment and surrounding hazards from complex background, conveying great significance to maintaining electric power transmission safety. Recently, the Segment Anything Model (SAM) has emerged as a foundational vision model and pushed the boundaries of segmentation tasks. However, SAM struggles to deal with the target objects in complex transmission corridor scenario, especially those with fine structure. In this paper, we propose ELE-SAM, adapting SAM for the PTCHS task. Technically, we develop a Context-Aware Prompt Adapter to achieve better prompt tokens via incorporating global-local features and focusing more on key regions. Subsequently, to tackle the hazard objects with fine structure in complex background, we design a High-Fidelity Mask Decoder by leveraging multi-granularity mask features and then scaling them to a higher resolution. Moreover, to train ELE-SAM and advance this field, we construct the ELE-40K benchmark, the first large-scale and real-world dataset for PTCHS including 44,094 image-mask pairs. Experimental results for ELE-40K demonstrate the superior performance that ELE-SAM outperforms the baseline model with the average 16.8% mIoU and 20.6% mBIoU performance improvement. Moreover, compared with the state-of-the-art method on HQSeg-44K, the average 2.9% mIoU and 3.8% mBIoU absolute improvements further validate the effectiveness of our method on high-quality generic object segmentation. The source code and dataset are available at https://github.com/Hhaizee/ELE-SAM.
>
---
#### [replaced 074] From Limited Labels to Open Domains:An Efficient Learning Method for Drone-view Geo-Localization
- **分类: cs.CV; cs.IR**

- **链接: [https://arxiv.org/pdf/2503.07520v3](https://arxiv.org/pdf/2503.07520v3)**

> **作者:** Zhongwei Chen; Zhao-Xu Yang; Hai-Jun Rong; Jiawei Lang; Guoqi Li
>
> **摘要:** Traditional supervised drone-view geo-localization (DVGL) methods heavily depend on paired training data and encounter difficulties in learning cross-view correlations from unpaired data. Moreover, when deployed in a new domain, these methods require obtaining the new paired data and subsequent retraining for model adaptation, which significantly increases computational overhead. Existing unsupervised methods have enabled to generate pseudo-labels based on cross-view similarity to infer the pairing relationships. However, geographical similarity and spatial continuity often cause visually analogous features at different geographical locations. The feature confusion compromises the reliability of pseudo-label generation, where incorrect pseudo-labels drive negative optimization. Given these challenges inherent in both supervised and unsupervised DVGL methods, we propose a novel cross-domain invariant knowledge transfer network (CDIKTNet) with limited supervision, whose architecture consists of a cross-domain invariance sub-network (CDIS) and a cross-domain transfer sub-network (CDTS). This architecture facilitates a closed-loop framework for invariance feature learning and knowledge transfer. The CDIS is designed to learn cross-view structural and spatial invariance from a small amount of paired data that serves as prior knowledge. It endows the shared feature space of unpaired data with similar implicit cross-view correlations at initialization, which alleviates feature confusion. Based on this, the CDTS employs dual-path contrastive learning to further optimize each subspace while preserving consistency in a shared feature space. Extensive experiments demonstrate that CDIKTNet achieves state-of-the-art performance under full supervision compared with those supervised methods, and further surpasses existing unsupervised methods in both few-shot and cross-domain initialization.
>
---
#### [replaced 075] Systematic Evaluation and Guidelines for Segment Anything Model in Surgical Video Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.00525v3](https://arxiv.org/pdf/2501.00525v3)**

> **作者:** Cheng Yuan; Jian Jiang; Kunyi Yang; Lv Wu; Rui Wang; Zi Meng; Haonan Ping; Ziyu Xu; Yifan Zhou; Wanli Song; Hesheng Wang; Yueming Jin; Qi Dou; Yutong Ban
>
> **摘要:** Surgical video segmentation is critical for AI to interpret spatial-temporal dynamics in surgery, yet model performance is constrained by limited annotated data. The SAM2 model, pretrained on natural videos, offers potential for zero-shot surgical segmentation, but its applicability in complex surgical environments, with challenges like tissue deformation and instrument variability, remains unexplored. We present the first comprehensive evaluation of the zero-shot capability of SAM2 in 9 surgical datasets (17 surgery types), covering laparoscopic, endoscopic, and robotic procedures. We analyze various prompting (points, boxes, mask) and {finetuning (dense, sparse) strategies}, robustness to surgical challenges, and generalization across procedures and anatomies. Key findings reveal that while SAM2 demonstrates notable zero-shot adaptability in structured scenarios (e.g., instrument segmentation, {multi-organ segmentation}, and scene segmentation), its performance varies under dynamic surgical conditions, highlighting gaps in handling temporal coherence and domain-specific artifacts. These results highlight future pathways to adaptive data-efficient solutions for the surgical data science field.
>
---
#### [replaced 076] TimeViper: A Hybrid Mamba-Transformer Vision-Language Model for Efficient Long Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对长视频理解任务，解决高效处理超长视频（>10,000帧）时的计算效率与上下文建模难题。提出混合Mamba-Transformer架构TimeViper，发现视觉信息向文本传递导致冗余，并设计TransV模块压缩视觉信息，实现高效多模态理解。**

- **链接: [https://arxiv.org/pdf/2511.16595v2](https://arxiv.org/pdf/2511.16595v2)**

> **作者:** Boshen Xu; Zihan Xiao; Jiaze Li; Jianzhong Ju; Zhenbo Luo; Jian Luan; Qin Jin
>
> **备注:** Project page: https://xuboshen.github.io/TimeViper; Code: https://github.com/xiaomi-research/timeviper
>
> **摘要:** We introduce TimeViper, a hybrid vision-language model designed to tackle challenges of long video understanding. Processing long videos demands both an efficient model architecture and an effective mechanism for handling extended temporal contexts. To this end, TimeViper adopts a hybrid Mamba-Transformer backbone that combines the efficiency of state-space models with the expressivity of attention mechanisms. Through this hybrid design, we reveal the vision-to-text information aggregation phenomenon, where information progressively flows from vision tokens to text tokens across increasing LLM depth, resulting in severe vision token redundancy. Motivated by this observation, we propose TransV, a token information transfer module that transfers and compresses vision tokens into instruction tokens while maintaining multimodal understanding capabilities. This design enables TimeViper to process hour-long videos exceeding 10,000 frames. Extensive experiments across multiple benchmarks demonstrate that TimeViper competes with state-of-the-art models while extending frame numbers. We further analyze attention behaviors of both Mamba and Transformer layers, offering new insights into hybrid model interpretability. This work represents an initial step towards developing, interpreting, and compressing hybrid Mamba-Transformer architectures.
>
---
#### [replaced 077] ProtoPFormer: Concentrating on Prototypical Parts in Vision Transformers for Interpretable Image Recognition
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2208.10431v3](https://arxiv.org/pdf/2208.10431v3)**

> **作者:** Mengqi Xue; Qihan Huang; Haofei Zhang; Jingwen Hu; Jie Song; Mingli Song; Canghong Jin
>
> **备注:** Arxiv preprint; 18 pages, 12 figures, 7 tables
>
> **摘要:** Prototypical part network (ProtoPNet) has drawn wide attention and boosted many follow-up studies due to its self-explanatory property for explainable artificial intelligence (XAI). However, when directly applying ProtoPNet on vision transformer (ViT) backbones, learned prototypes have a "distraction" problem: they have a relatively high probability of being activated by the background and pay less attention to the foreground. The powerful capability of modeling long-term dependency makes the transformer-based ProtoPNet hard to focus on prototypical parts, thus severely impairing its inherent interpretability. This paper proposes prototypical part transformer (ProtoPFormer) for appropriately and effectively applying the prototype-based method with ViTs for interpretable image recognition. The proposed method introduces global and local prototypes for capturing and highlighting the representative holistic and partial features of targets according to the architectural characteristics of ViTs. The global prototypes are adopted to provide the global view of objects to guide local prototypes to concentrate on the foreground while eliminating the influence of the background. Afterwards, local prototypes are explicitly supervised to concentrate on their respective prototypical visual parts, increasing the overall interpretability. Extensive experiments demonstrate that our proposed global and local prototypes can mutually correct each other and jointly make final decisions, which faithfully and transparently reason the decision-making processes associatively from the whole and local perspectives, respectively. Moreover, ProtoPFormer consistently achieves superior performance and visualization results over the state-of-the-art (SOTA) prototype-based baselines. Our code has been released at https://github.com/zju-vipa/ProtoPFormer.
>
---
#### [replaced 078] Pistachio: Towards Synthetic, Balanced, and Long-Form Video Anomaly Benchmarks
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [https://arxiv.org/pdf/2511.19474v2](https://arxiv.org/pdf/2511.19474v2)**

> **作者:** Jie Li; Hongyi Cai; Mingkang Dong; Muxin Pu; Shan You; Fei Wang; Tao Huang
>
> **摘要:** Automatically detecting abnormal events in videos is crucial for modern autonomous systems, yet existing Video Anomaly Detection (VAD) benchmarks lack the scene diversity, balanced anomaly coverage, and temporal complexity needed to reliably assess real-world performance. Meanwhile, the community is increasingly moving toward Video Anomaly Understanding (VAU), which requires deeper semantic and causal reasoning but remains difficult to benchmark due to the heavy manual annotation effort it demands. In this paper, we introduce Pistachio, a new VAD/VAU benchmark constructed entirely through a controlled, generation-based pipeline. By leveraging recent advances in video generation models, Pistachio provides precise control over scenes, anomaly types, and temporal narratives, effectively eliminating the biases and limitations of Internet-collected datasets. Our pipeline integrates scene-conditioned anomaly assignment, multi-step storyline generation, and a temporally consistent long-form synthesis strategy that produces coherent 41-second videos with minimal human intervention. Extensive experiments demonstrate the scale, diversity, and complexity of Pistachio, revealing new challenges for existing methods and motivating future research on dynamic and multi-event anomaly understanding.
>
---
#### [replaced 079] Thinking in 360°: Humanoid Visual Search in the Wild
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20351v2](https://arxiv.org/pdf/2511.20351v2)**

> **作者:** Heyang Yu; Yinan Han; Xiangyu Zhang; Baiqiao Yin; Bowen Chang; Xiangyu Han; Xinhao Liu; Jing Zhang; Marco Pavone; Chen Feng; Saining Xie; Yiming Li
>
> **备注:** Website: https://humanoid-vstar.github.io/ ; Code: https://github.com/humanoid-vstar/hstar
>
> **摘要:** Humans rely on the synergistic control of head (cephalomotor) and eye (oculomotor) to efficiently search for visual information in 360°. However, prior approaches to visual search are limited to a static image, neglecting the physical embodiment and its interaction with the 3D world. How can we develop embodied visual search agents as efficient as humans while bypassing the constraints imposed by real-world hardware? To this end, we propose humanoid visual search where a humanoid agent actively rotates its head to search for objects or paths in an immersive world represented by a 360° panoramic image. To study visual search in visually-crowded real-world scenarios, we build H* Bench, a new benchmark that moves beyond household scenes to challenging in-the-wild scenes that necessitate advanced visual-spatial reasoning capabilities, such as transportation hubs, large-scale retail spaces, urban streets, and public institutions. Our experiments first reveal that even top-tier proprietary models falter, achieving only ~30% success in object and path search. We then use post-training techniques to enhance the open-source Qwen2.5-VL, increasing its success rate by over threefold for both object search (14.83% to 47.38%) and path search (6.44% to 24.94%). Notably, the lower ceiling of path search reveals its inherent difficulty, which we attribute to the demand for sophisticated spatial commonsense. Our results not only show a promising path forward but also quantify the immense challenge that remains in building MLLM agents that can be seamlessly integrated into everyday human life.
>
---
#### [replaced 080] Towards Spatially Consistent Image Generation: On Incorporating Intrinsic Scene Properties into Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.10382v2](https://arxiv.org/pdf/2508.10382v2)**

> **作者:** Hyundo Lee; Suhyung Choi; Inwoo Hwang; Byoung-Tak Zhang
>
> **摘要:** Image generation models trained on large datasets can synthesize high-quality images but often produce spatially inconsistent and distorted images due to limited information about the underlying structures and spatial layouts. In this work, we leverage intrinsic scene properties (e.g., depth, segmentation maps) that provide rich information about the underlying scene, unlike prior approaches that solely rely on image-text pairs or use intrinsics as conditional inputs. Our approach aims to co-generate both images and their corresponding intrinsics, enabling the model to implicitly capture the underlying scene structure and generate more spatially consistent and realistic images. Specifically, we first extract rich intrinsic scene properties from a large image dataset with pre-trained estimators, eliminating the need for additional scene information or explicit 3D representations. We then aggregate various intrinsic scene properties into a single latent variable using an autoencoder. Building upon pre-trained large-scale Latent Diffusion Models (LDMs), our method simultaneously denoises the image and intrinsic domains by carefully sharing mutual information so that the image and intrinsic reflect each other without degrading image quality. Experimental results demonstrate that our method corrects spatial inconsistencies and produces a more natural layout of scenes while maintaining the fidelity and textual alignment of the base model (e.g., Stable Diffusion).
>
---
#### [replaced 081] Leveraging Contrast Information for Efficient Document Shadow Removal
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.00385v2](https://arxiv.org/pdf/2504.00385v2)**

> **作者:** Yifan Liu; Jiancheng Huang; Na Liu; Mingfu Yan; Yi Huang; Shifeng Chen
>
> **备注:** There are unresolved authorship disputes related to this submission, and the current version does not reflect an agreed authorship list
>
> **摘要:** Document shadows are a major obstacle in the digitization process. Due to the dense information in text and patterns covered by shadows, document shadow removal requires specialized methods. Existing document shadow removal methods, although showing some progress, still rely on additional information such as shadow masks or lack generalization and effectiveness across different shadow scenarios. This often results in incomplete shadow removal or loss of original document content and tones. Moreover, these methods tend to underutilize the information present in the original shadowed document image. In this paper, we refocus our approach on the document images themselves, which inherently contain rich information.We propose an end-to-end document shadow removal method guided by contrast representation, following a coarse-to-fine refinement approach. By extracting document contrast information, we can effectively and quickly locate shadow shapes and positions without the need for additional masks. This information is then integrated into the refined shadow removal process, providing better guidance for network-based removal and feature fusion. Extensive qualitative and quantitative experiments show that our method achieves state-of-the-art performance.
>
---
#### [replaced 082] Benchmarking the Trustworthiness in Multimodal LLMs for Video Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.12336v3](https://arxiv.org/pdf/2506.12336v3)**

> **作者:** Youze Wang; Zijun Chen; Ruoyu Chen; Shishen Gu; Wenbo Hu; Jiayang Liu; Yinpeng Dong; Hang Su; Jun Zhu; Meng Wang; Richang Hong
>
> **摘要:** Recent advancements in multimodal large language models for video understanding (videoLLMs) have enhanced their capacity to process complex spatiotemporal data. However, challenges such as factual inaccuracies, harmful content, biases, hallucinations, and privacy risks compromise their reliability. This study introduces Trust-videoLLMs, a first comprehensive benchmark evaluating 23 state-of-the-art videoLLMs (5 commercial, 18 open-source) across five critical dimensions: truthfulness, robustness, safety, fairness, and privacy. Comprising 30 tasks with adapted, synthetic, and annotated videos, the framework assesses spatiotemporal risks, temporal consistency and cross-modal impact. Results reveal significant limitations in dynamic scene comprehension, cross-modal perturbation resilience and real-world risk mitigation. While open-source models occasionally outperform, proprietary models generally exhibit superior credibility, though scaling does not consistently improve performance. These findings underscore the need for enhanced training datat diversity and robust multimodal alignment. Trust-videoLLMs provides a publicly available, extensible toolkit for standardized trustworthiness assessments, addressing the critical gap between accuracy-focused benchmarks and demands for robustness, safety, fairness, and privacy.
>
---
#### [replaced 083] Improved Visually Prompted Keyword Localisation in Real Low-Resource Settings
- **分类: cs.CL; cs.CV; eess.AS**

- **简介: 该论文研究视觉提示关键词定位任务，旨在无转录条件下定位语音中出现的图像所指示的词。针对低资源语言缺乏标注的问题，提出无需转录的少样本配对挖掘方法。在英语上性能略有下降，在真实低资源语言尤鲁巴语上表现仍可接受，但因配对不准确导致性能下降更明显。**

- **链接: [https://arxiv.org/pdf/2409.06013v2](https://arxiv.org/pdf/2409.06013v2)**

> **作者:** Leanne Nortje; Dan Oneata; Gabriel Pirlogeanu; Herman Kamper
>
> **备注:** Accepted at SpeD 2025
>
> **摘要:** Given an image query, visually prompted keyword localisation (VPKL) aims to find occurrences of the depicted word in a speech collection. This can be useful when transcriptions are not available for a low-resource language (e.g. if it is unwritten). Previous work showed that VPKL can be performed with a visually grounded speech model trained on paired images and unlabelled speech. But all experiments were done on English. Moreover, transcriptions were used to get positive and negative pairs for the contrastive loss. This paper introduces a few-shot learning scheme to mine pairs automatically without transcriptions. On English, this results in only a small drop in performance. We also - for the first time - consider VPKL on a real low-resource language, Yoruba. While scores are reasonable, here we see a bigger drop in performance compared to using ground truth pairs because the mining is less accurate in Yoruba.
>
---
#### [replaced 084] GreenHyperSpectra: A multi-source hyperspectral dataset for global vegetation trait prediction
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2507.06806v3](https://arxiv.org/pdf/2507.06806v3)**

> **作者:** Eya Cherif; Arthur Ouaknine; Luke A. Brown; Phuong D. Dao; Kyle R. Kovach; Bing Lu; Daniel Mederer; Hannes Feilhauer; Teja Kattenborn; David Rolnick
>
> **备注:** Accepted at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Plant traits such as leaf carbon content and leaf mass are essential variables in the study of biodiversity and climate change. However, conventional field sampling cannot feasibly cover trait variation at ecologically meaningful spatial scales. Machine learning represents a valuable solution for plant trait prediction across ecosystems, leveraging hyperspectral data from remote sensing. Nevertheless, trait prediction from hyperspectral data is challenged by label scarcity and substantial domain shifts (\eg across sensors, ecological distributions), requiring robust cross-domain methods. Here, we present GreenHyperSpectra, a pretraining dataset encompassing real-world cross-sensor and cross-ecosystem samples designed to benchmark trait prediction with semi- and self-supervised methods. We adopt an evaluation framework encompassing in-distribution and out-of-distribution scenarios. We successfully leverage GreenHyperSpectra to pretrain label-efficient multi-output regression models that outperform the state-of-the-art supervised baseline. Our empirical analyses demonstrate substantial improvements in learning spectral representations for trait prediction, establishing a comprehensive methodological framework to catalyze research at the intersection of representation learning and plant functional traits assessment. All code and data are available at: https://github.com/echerif18/HyspectraSSL.
>
---
#### [replaced 085] Scale Where It Matters: Training-Free Localized Scaling for Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19917v2](https://arxiv.org/pdf/2511.19917v2)**

> **作者:** Qin Ren; Yufei Wang; Lanqing Guo; Wen Zhang; Zhiwen Fan; Chenyu You
>
> **摘要:** Diffusion models have become the dominant paradigm in text-to-image generation, and test-time scaling (TTS) further improves quality by allocating more computation during inference. However, existing TTS methods operate at the full-image level, overlooking the fact that image quality is often spatially heterogeneous. This leads to unnecessary computation on already satisfactory regions and insufficient correction of localized defects. In this paper, we explore a new direction - Localized TTS - that adaptively resamples defective regions while preserving high-quality regions, thereby substantially reducing the search space. This paradigm poses two central challenges: accurately localizing defects and maintaining global consistency. We propose LoTTS, the first fully training-free framework for localized TTS. For defect localization, LoTTS contrasts cross- and self-attention signals under quality-aware prompts (e.g., high-quality vs. low-quality) to identify defective regions, and then refines them into coherent masks. For consistency, LoTTS perturbs only defective regions and denoises them locally, ensuring that corrections remain confined while the rest of the image remains undisturbed. Extensive experiments on SD2.1, SDXL, and FLUX demonstrate that LoTTS achieves state-of-the-art performance: it consistently improves both local quality and global fidelity, while reducing GPU cost by 2-4x compared to Best-of-N sampling. These findings establish localized TTS as a promising new direction for scaling diffusion models at inference time.
>
---
#### [replaced 086] SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.10160v2](https://arxiv.org/pdf/2510.10160v2)**

> **作者:** Zhenjie Mao; Yuhuan Yang; Chaofan Ma; Dongsheng Jiang; Jiangchao Yao; Ya Zhang; Yanfeng Wang
>
> **备注:** NeurIPS 2025; Project page: https://zhenjiemao.github.io/SaFiRe/
>
> **摘要:** Referring Image Segmentation (RIS) aims to segment the target object in an image given a natural language expression. While recent methods leverage pre-trained vision backbones and more training corpus to achieve impressive results, they predominantly focus on simple expressions--short, clear noun phrases like "red car" or "left girl". This simplification often reduces RIS to a key word/concept matching problem, limiting the model's ability to handle referential ambiguity in expressions. In this work, we identify two challenging real-world scenarios: object-distracting expressions, which involve multiple entities with contextual cues, and category-implicit expressions, where the object class is not explicitly stated. To address the challenges, we propose a novel framework, SaFiRe, which mimics the human two-phase cognitive process--first forming a global understanding, then refining it through detail-oriented inspection. This is naturally supported by Mamba's scan-then-update property, which aligns with our phased design and enables efficient multi-cycle refinement with linear complexity. We further introduce aRefCOCO, a new benchmark designed to evaluate RIS models under ambiguous referring expressions. Extensive experiments on both standard and proposed datasets demonstrate the superiority of SaFiRe over state-of-the-art baselines.
>
---
#### [replaced 087] Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19516v2](https://arxiv.org/pdf/2511.19516v2)**

> **作者:** Liqin Luo; Guangyao Chen; Xiawu Zheng; Yongxing Dai; Yixiong Zou; Yonghong Tian
>
> **备注:** AAAI 2026
>
> **摘要:** Visual grounding, the task of linking textual queries to specific regions within images, plays a pivotal role in vision-language integration. Existing methods typically rely on extensive task-specific annotations and fine-tuning, limiting their ability to generalize effectively to novel or out-of-distribution scenarios. To address these limitations, we introduce GroundingAgent, a novel agentic visual grounding framework that operates without any task-specific fine-tuning. GroundingAgent employs a structured, iterative reasoning mechanism that integrates pretrained open-vocabulary object detectors, multimodal large language models (MLLMs), and large language models (LLMs) to progressively refine candidate regions through joint semantic and spatial analyses. Remarkably, GroundingAgent achieves an average zero-shot grounding accuracy of 65.1 % on widely-used benchmarks (RefCOCO, RefCOCO+, RefCOCOg), entirely without fine-tuning. Furthermore, by substituting MLLM-generated captions with the original query texts, the accuracy at the selection stage alone reaches approximately 90 %, closely matching supervised performance and underscoring the critical role of LLM reasoning capabilities. GroundingAgent also offers strong interpretability, transparently illustrating each reasoning step and providing clear insights into its decision-making process.
>
---
#### [replaced 088] MoEGCL: Mixture of Ego-Graphs Contrastive Representation Learning for Multi-View Clustering
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.05876v2](https://arxiv.org/pdf/2511.05876v2)**

> **作者:** Jian Zhu; Xin Zou; Jun Sun; Cheng Luo; Lei Liu; Lingfang Zeng; Ning Zhang; Bian Wu; Chang Tang; Lirong Dai
>
> **摘要:** In recent years, the advancement of Graph Neural Networks (GNNs) has significantly propelled progress in Multi-View Clustering (MVC). However, existing methods face the problem of coarse-grained graph fusion. Specifically, current approaches typically generate a separate graph structure for each view and then perform weighted fusion of graph structures at the view level, which is a relatively rough strategy. To address this limitation, we present a novel Mixture of Ego-Graphs Contrastive Representation Learning (MoEGCL). It mainly consists of two modules. In particular, we propose an innovative Mixture of Ego-Graphs Fusion (MoEGF), which constructs ego graphs and utilizes a Mixture-of-Experts network to implement fine-grained fusion of ego graphs at the sample level, rather than the conventional view-level fusion. Additionally, we present the Ego Graph Contrastive Learning (EGCL) module to align the fused representation with the view-specific representation. The EGCL module enhances the representation similarity of samples from the same cluster, not merely from the same sample, further boosting fine-grained graph representation. Extensive experiments demonstrate that MoEGCL achieves state-of-the-art results in deep multi-view clustering tasks. The source code is publicly available at https://github.com/HackerHyper/MoEGCL.
>
---
#### [replaced 089] ReasonAct: Progressive Training for Fine-Grained Video Reasoning in Small Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01533v2](https://arxiv.org/pdf/2508.01533v2)**

> **作者:** Jiaxin Liu; Zhaolu Kang
>
> **摘要:** While recent multimodal models have shown progress in vision-language tasks, small-scale variants still struggle with the fine-grained temporal reasoning required for video understanding. We introduce ReasonAct, a method that enhances video reasoning in smaller models through a three-stage training process: first building a foundation with text-only reasoning, then fine-tuning on video, and finally refining with temporal-aware reinforcement learning. We build upon Temporal Group Relative Policy Optimization (T-GRPO) by incorporating temporal consistency modeling into policy optimization. We also propose a biomechanically-motivated sub-action decomposition mechanism that provides graduated rewards for constituent action phases. Through experiments on HMDB51, UCF-101, and Kinetics-400, our 3B-parameter model achieves 67.2%, 94.1%, and 78.9% accuracy respectively, demonstrating improvements of 17.9, 15.8, and 12.3 points over baselines. Ablation studies validate that our progressive training enables smaller models to achieve competitive video reasoning performance while maintaining computational efficiency.
>
---
#### [replaced 090] TinyChemVL: Advancing Chemical Vision-Language Models via Efficient Visual Token Reduction and Complex Reaction Tasks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06283v2](https://arxiv.org/pdf/2511.06283v2)**

> **作者:** Xuanle Zhao; Shuxin Zeng; Xinyuan Cai; Xiang Cheng; Duzhen Zhang; Xiuyi Chen; Bo Xu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** While Vision Language Models (VLMs) have demonstrated remarkable capabilities in general visual understanding, their application in the chemical domain has been limited, with previous works predominantly focusing on text and thus overlooking critical visual information, such as molecular structures. Current approaches that directly adopt standard VLMs for chemical tasks suffer from two primary issues: (i) computational inefficiency of processing entire chemical images with non-informative backgrounds. (ii) a narrow scope on molecular-level tasks that restricts progress in chemical reasoning. In this work, we propose \textbf{TinyChemVL}, an efficient and powerful chemical VLM that leverages visual token reduction and reaction-level tasks to improve model efficiency and reasoning capacity. Also, we propose \textbf{ChemRxn-V}, a reaction-level benchmark for assessing vision-based reaction recognition and prediction tasks. Directly predicting reaction products from molecular images poses a non-trivial challenge, as it requires models to integrate both recognition and reasoning capacities. Our results demonstrate that with only 4B parameters, TinyChemVL achieves superior performance on both molecular and reaction tasks while demonstrating faster inference and training speeds compared to existing models. Notably, TinyChemVL outperforms ChemVLM while utilizing only 1/16th of the visual tokens. This work builds efficient yet powerful VLMs for chemical domains by co-designing model architecture and task complexity.
>
---
#### [replaced 091] One-Step Diffusion-Based Image Compression with Semantic Distillation
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2505.16687v2](https://arxiv.org/pdf/2505.16687v2)**

> **作者:** Naifu Xue; Zhaoyang Jia; Jiahao Li; Bin Li; Yuan Zhang; Yan Lu
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** While recent diffusion-based generative image codecs have shown impressive performance, their iterative sampling process introduces unpleasing latency. In this work, we revisit the design of a diffusion-based codec and argue that multi-step sampling is not necessary for generative compression. Based on this insight, we propose OneDC, a One-step Diffusion-based generative image Codec -- that integrates a latent compression module with a one-step diffusion generator. Recognizing the critical role of semantic guidance in one-step diffusion, we propose using the hyperprior as a semantic signal, overcoming the limitations of text prompts in representing complex visual content. To further enhance the semantic capability of the hyperprior, we introduce a semantic distillation mechanism that transfers knowledge from a pretrained generative tokenizer to the hyperprior codec. Additionally, we adopt a hybrid pixel- and latent-domain optimization to jointly enhance both reconstruction fidelity and perceptual realism. Extensive experiments demonstrate that OneDC achieves SOTA perceptual quality even with one-step generation, offering over 39% bitrate reduction and 20x faster decoding compared to prior multi-step diffusion-based codecs. Project: https://onedc-codec.github.io/
>
---
#### [replaced 092] OuroMamba: A Data-Free Quantization Framework for Vision Mamba
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.10959v2](https://arxiv.org/pdf/2503.10959v2)**

> **作者:** Akshat Ramachandran; Mingyu Lee; Huan Xu; Souvik Kundu; Tushar Krishna
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** We present OuroMamba, the first data-free post-training quantization (DFQ) method for vision Mamba-based models (VMMs). We identify two key challenges in enabling DFQ for VMMs, (1) VMM's recurrent state transitions restricts capturing of long-range interactions and leads to semantically weak synthetic data, (2) VMM activations exhibit dynamic outlier variations across time-steps, rendering existing static PTQ techniques ineffective. To address these challenges, OuroMamba presents a two-stage framework: (1) OuroMamba-Gen to generate semantically rich and meaningful synthetic data. It applies contrastive learning on patch level VMM features generated through neighborhood interactions in the latent state space, (2) OuroMamba-Quant to employ mixed-precision quantization with lightweight dynamic outlier detection during inference. In specific, we present a thresholding based outlier channel selection strategy for activations that gets updated every time-step. Extensive experiments across vision and generative tasks show that our data-free OuroMamba surpasses existing data-driven PTQ techniques, achieving state-of-the-art performance across diverse quantization settings. Additionally, we implement efficient GPU kernels to achieve practical latency speedup of up to 2.36x. Code and synthetic dataset are available here: https://github.com/georgia-tech-synergy-lab/ICCV-OuroMamba
>
---
#### [replaced 093] MeshCone: Second-Order Cone Programming for Geometrically-Constrained Mesh Enhancement
- **分类: cs.GR; cs.CV; math.OC**

- **链接: [https://arxiv.org/pdf/2412.08484v4](https://arxiv.org/pdf/2412.08484v4)**

> **作者:** Alexander Valverde
>
> **摘要:** Modern mesh generation pipelines whether learning-based or classical often produce outputs requiring post-processing to achieve production-quality geometry. This work introduces MeshCone, a convex optimization framework for guided mesh refinement that leverages reference geometry to correct deformed or degraded meshes. We formulate the problem as a second-order cone program where vertex positions are optimized to align with target geometry while enforcing smoothness through convex edge-length regularization. MeshCone performs geometry-aware optimization that preserves fine details while correcting structural defects. We demonstrate robust performance across 56 diverse object categories from ShapeNet and ThreeDScans, achieving superior refinement quality compared to Laplacian smoothing and unoptimized baselines while maintaining sub-second inference times. MeshCone is particularly suited for applications where reference geometry is available, such as mesh-from-template workflows, scan-to-CAD alignment, and quality assurance in asset production pipelines.
>
---
#### [replaced 094] Passive Dementia Screening via Facial Temporal Micro-Dynamics Analysis of In-the-Wild Talking-Head Video
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.13802v2](https://arxiv.org/pdf/2511.13802v2)**

> **作者:** Filippo Cenacchi; Longbing Cao; Mitchell McEwan; Deborah Richards
>
> **摘要:** We target passive dementia screening from short camera-facing talking head video, developing a facial temporal micro dynamics analysis for language free detection of early neuro cognitive change. This enables unscripted, in the wild video analysis at scale to capture natural facial behaviors, transferrable across devices, topics, and cultures without active intervention by clinicians or researchers during recording. Most existing resources prioritize speech or scripted interviews, limiting use outside clinics and coupling predictions to language and transcription. In contrast, we identify and analyze whether temporal facial kinematics, including blink dynamics, small mouth jaw motions, gaze variability, and subtle head adjustments, are sufficient for dementia screening without speech or text. By stabilizing facial signals, we convert these micro movements into interpretable facial microdynamic time series, smooth them, and summarize short windows into compact clip level statistics for screening. Each window is encoded by its activity mix (the relative share of motion across streams), thus the predictor analyzes the distribution of motion across streams rather than its magnitude, making per channel effects transparent. We also introduce YT DemTalk, a new dataset curated from publicly available, in the wild camera facing videos. It contains 300 clips (150 with self reported dementia, 150 controls) to test our model and offer a first benchmarking of the corpus. On YT DemTalk, ablations identify gaze lability and mouth/jaw dynamics as the most informative cues, and light weighted shallow classifiers could attain a dementia prediction performance of (AUROC) 0.953, 0.961 Average Precision (AP), 0.851 F1-score, and 0.857 accuracy.
>
---
#### [replaced 095] Active Negative Loss: A Robust Framework for Learning with Noisy Labels
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.02373v2](https://arxiv.org/pdf/2412.02373v2)**

> **作者:** Xichen Ye; Yifan Wu; Yiqi Wang; Xiaoqiang Li; Weizhong Zhang; Yifan Chen
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Deep supervised learning has achieved remarkable success across a wide range of tasks, yet it remains susceptible to overfitting when confronted with noisy labels. To address this issue, noise-robust loss functions offer an effective solution for enhancing learning in the presence of label noise. In this work, we systematically investigate the limitation of the recently proposed Active Passive Loss (APL), which employs Mean Absolute Error (MAE) as its passive loss function. Despite the robustness brought by MAE, one of its key drawbacks is that it pays equal attention to clean and noisy samples; this feature slows down convergence and potentially makes training difficult, particularly in large-scale datasets. To overcome these challenges, we introduce a novel loss function class, termed Normalized Negative Loss Functions (NNLFs), which serve as passive loss functions within the APL framework. NNLFs effectively address the limitations of MAE by concentrating more on memorized clean samples. By replacing MAE in APL with our proposed NNLFs, we enhance APL and present a new framework called Active Negative Loss (ANL). Moreover, in non-symmetric noise scenarios, we propose an entropy-based regularization technique to mitigate the vulnerability to the label imbalance. Extensive experiments demonstrate that the new loss functions adopted by our ANL framework can achieve better or comparable performance to state-of-the-art methods across various label noise types and in image segmentation tasks. The source code is available at: https://github.com/Virusdoll/Active-Negative-Loss.
>
---
#### [replaced 096] EmoFeedback$^2$: Reinforcement of Continuous Emotional Image Generation via LVLM-based Reward and Textual Feedback
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.19982v2](https://arxiv.org/pdf/2511.19982v2)**

> **作者:** Jingyang Jia; Kai Shu; Gang Yang; Long Xing; Xun Chen; Aiping Liu
>
> **摘要:** Continuous emotional image generation (C-EICG) is emerging rapidly due to its ability to produce images aligned with both user descriptions and continuous emotional values. However, existing approaches lack emotional feedback from generated images, limiting the control of emotional continuity. Additionally, their simple alignment between emotions and naively generated texts fails to adaptively adjust emotional prompts according to image content, leading to insufficient emotional fidelity. To address these concerns, we propose a novel generation-understanding-feedback reinforcement paradigm (EmoFeedback$^2$) for C-EICG, which exploits the reasoning capability of the fine-tuned large vision-language model (LVLM) to provide reward and textual feedback for generating high-quality images with continuous emotions. Specifically, we introduce an emotion-aware reward feedback strategy, where the LVLM evaluates the emotional values of generated images and computes the reward against target emotions, guiding the reinforcement fine-tuning of the generative model and enhancing the emotional continuity of images. Furthermore, we design a self-promotion textual feedback framework, in which the LVLM iteratively analyzes the emotional content of generated images and adaptively produces refinement suggestions for the next-round prompt, improving the emotional fidelity with fine-grained content. Extensive experimental results demonstrate that our approach effectively generates high-quality images with the desired emotions, outperforming existing state-of-the-art methods in our custom dataset. The code and dataset will be released soon.
>
---
#### [replaced 097] Cheating Stereo Matching in Full-scale: Physical Adversarial Attack against Binocular Depth Estimation in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.14386v3](https://arxiv.org/pdf/2511.14386v3)**

> **作者:** Kangqiao Zhao; Shuo Huai; Xurui Song; Jun Luo
>
> **备注:** AAAI 2026
>
> **摘要:** Though deep neural models adopted to realize the perception of autonomous driving have proven vulnerable to adversarial examples, known attacks often leverage 2D patches and target mostly monocular perception. Therefore, the effectiveness of Physical Adversarial Examples (PAEs) on stereo-based binocular depth estimation remains largely unexplored. To this end, we propose the first texture-enabled physical adversarial attack against stereo matching models in the context of autonomous driving. Our method employs a 3D PAE with global camouflage texture rather than a local 2D patch-based one, ensuring both visual consistency and attack effectiveness across different viewpoints of stereo cameras. To cope with the disparity effect of these cameras, we also propose a new 3D stereo matching rendering module that allows the PAE to be aligned with real-world positions and headings in binocular vision. We further propose a novel merging attack that seamlessly blends the target into the environment through fine-grained PAE optimization. It has significantly enhanced stealth and lethality upon existing hiding attacks that fail to get seamlessly merged into the background. Extensive evaluations show that our PAEs can successfully fool the stereo models into producing erroneous depth information.
>
---
#### [replaced 098] FastAvatar: Instant 3D Gaussian Splatting for Faces from Single Unconstrained Poses
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.18389v2](https://arxiv.org/pdf/2508.18389v2)**

> **作者:** Hao Liang; Zhixuan Ge; Soumendu Majee; Ashish Tiwari; G. M. Dilshan Godaliyadda; Ashok Veeraraghavan; Guha Balakrishnan
>
> **备注:** 11 pages, 5 figures, website: https://hliang2.github.io/FastAvatar/
>
> **摘要:** We present FastAvatar, a fast and robust algorithm for single-image 3D face reconstruction using 3D Gaussian Splatting (3DGS). Given a single input image from an arbitrary pose, FastAvatar recovers a high-quality, full-head 3DGS avatar in approximately 3 seconds on a single NVIDIA A100 GPU. We use a two-stage design: a feed-forward encoder-decoder predicts coarse face geometry by regressing Gaussian structure from a pose-invariant identity embedding, and a lightweight test-time refinement stage then optimizes the appearance parameters for photorealistic rendering. This hybrid strategy combines the speed and stability of direct prediction with the accuracy of optimization, enabling strong identity preservation even under extreme input poses. FastAvatar achieves state-of-the-art reconstruction quality (24.01 dB PSNR, 0.91 SSIM) while running over 600x faster than existing per-subject optimization methods (e.g., FlashAvatar, GaussianAvatars, GASP). Once reconstructed, our avatars support photorealistic novel-view synthesis and FLAME-guided expression animation, enabling controllable reenactment from a single image. By jointly offering high fidelity, robustness to pose, and rapid reconstruction, FastAvatar significantly broadens the applicability of 3DGS-based facial avatars.
>
---
#### [replaced 099] IVY-FAKE: A Unified Explainable Framework and Benchmark for Image and Video AIGC Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.00979v3](https://arxiv.org/pdf/2506.00979v3)**

> **作者:** Changjiang Jiang; Wenhui Dong; Zhonghao Zhang; Chenyang Si; Fengchang Yu; Wei Peng; Xinbin Yuan; Yifei Bi; Ming Zhao; Zian Zhou; Caifeng Shan
>
> **备注:** 30 pages
>
> **摘要:** The rapid development of Artificial Intelligence Generated Content (AIGC) techniques has enabled the creation of high-quality synthetic content, but it also raises significant security concerns. Current detection methods face two major limitations: (1) the lack of multidimensional explainable datasets for generated images and videos. Existing open-source datasets (e.g., WildFake, GenVideo) rely on oversimplified binary annotations, which restrict the explainability and trustworthiness of trained detectors. (2) Prior MLLM-based forgery detectors (e.g., FakeVLM) exhibit insufficiently fine-grained interpretability in their step-by-step reasoning, which hinders reliable localization and explanation. To address these challenges, we introduce Ivy-Fake, the first large-scale multimodal benchmark for explainable AIGC detection. It consists of over 106K richly annotated training samples (images and videos) and 5,000 manually verified evaluation examples, sourced from multiple generative models and real world datasets through a carefully designed pipeline to ensure both diversity and quality. Furthermore, we propose Ivy-xDetector, a reinforcement learning model based on Group Relative Policy Optimization (GRPO), capable of producing explainable reasoning chains and achieving robust performance across multiple synthetic content detection benchmarks. Extensive experiments demonstrate the superiority of our dataset and confirm the effectiveness of our approach. Notably, our method improves performance on GenImage from 86.88% to 96.32%, surpassing prior state-of-the-art methods by a clear margin.
>
---
#### [replaced 100] Directed-Tokens: A Robust Multi-Modality Alignment Approach to Large Language-Vision Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.14264v2](https://arxiv.org/pdf/2508.14264v2)**

> **作者:** Thanh-Dat Truong; Huu-Thien Tran; Tran Thai Son; Bhiksha Raj; Khoa Luu
>
> **备注:** Accepted to NeurIPS'25
>
> **摘要:** Large multimodal models (LMMs) have gained impressive performance due to their outstanding capability in various understanding tasks. However, these models still suffer from some fundamental limitations related to robustness and generalization due to the alignment and correlation between visual and textual features. In this paper, we introduce a simple but efficient learning mechanism for improving the robust alignment between visual and textual modalities by solving shuffling problems. In particular, the proposed approach can improve reasoning capability, visual understanding, and cross-modality alignment by introducing two new tasks: reconstructing the image order and the text order into the LMM's pre-training and fine-tuning phases. In addition, we propose a new directed-token approach to capture visual and textual knowledge, enabling the capability to reconstruct the correct order of visual inputs. Then, we introduce a new Image-to-Response Guided loss to further improve the visual understanding of the LMM in its responses. The proposed approach consistently achieves state-of-the-art (SoTA) performance compared with prior LMMs on academic task-oriented and instruction-following LMM benchmarks.
>
---
#### [replaced 101] PointNSP: Autoregressive 3D Point Cloud Generation with Next-Scale Level-of-Detail Prediction
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.08594v2](https://arxiv.org/pdf/2503.08594v2)**

> **作者:** Ziqiao Meng; Qichao Wang; Zhiyang Dou; Zixing Song; Zhipeng Zhou; Irwin King; Peilin Zhao
>
> **备注:** 24 pages; Previously this version appeared as arXiv:2510.05613 which was submitted as a new work by accident
>
> **摘要:** Autoregressive point cloud generation has long lagged behind diffusion-based approaches in quality. The performance gap stems from the fact that autoregressive models impose an artificial ordering on inherently unordered point sets, forcing shape generation to proceed as a sequence of local predictions. This sequential bias emphasizes short-range continuity but undermines the model's capacity to capture long-range dependencies, hindering its ability to enforce global structural properties such as symmetry, consistent topology, and large-scale geometric regularities. Inspired by the level-of-detail (LOD) principle in shape modeling, we propose PointNSP, a coarse-to-fine generative framework that preserves global shape structure at low resolutions and progressively refines fine-grained geometry at higher scales through a next-scale prediction paradigm. This multi-scale factorization aligns the autoregressive objective with the permutation-invariant nature of point sets, enabling rich intra-scale interactions while avoiding brittle fixed orderings. Experiments on ShapeNet show that PointNSP establishes state-of-the-art (SOTA) generation quality for the first time within the autoregressive paradigm. In addition, it surpasses strong diffusion-based baselines in parameter, training, and inference efficiency. Finally, in dense generation with 8,192 points, PointNSP's advantages become even more pronounced, underscoring its scalability potential.
>
---
#### [replaced 102] UniChange: Unifying Change Detection with Multimodal Large Language Model
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对遥感图像变化检测任务，解决现有模型无法统一处理二值与语义变化检测、知识融合能力弱的问题。提出UniChange框架，利用多模态大模型的语言先验与生成能力，通过特殊标记和文本提示实现多源数据统一建模，显著提升泛化性与性能。**

- **链接: [https://arxiv.org/pdf/2511.02607v2](https://arxiv.org/pdf/2511.02607v2)**

> **作者:** Xu Zhang; Danyang Li; Xiaohang Dong; Tianhao Wu; Hualong Yu; Jianye Wang; Qicheng Li; Xiang Li
>
> **摘要:** Change detection (CD) is a fundamental task for monitoring and analyzing land cover dynamics. While recent high performance models and high quality datasets have significantly advanced the field, a critical limitation persists. Current models typically acquire limited knowledge from single-type annotated data and cannot concurrently leverage diverse binary change detection (BCD) and semantic change detection (SCD) datasets. This constraint leads to poor generalization and limited versatility. The recent advancements in Multimodal Large Language Models (MLLMs) introduce new possibilities for a unified CD framework. We leverage the language priors and unification capabilities of MLLMs to develop UniChange, the first MLLM-based unified change detection model. UniChange integrates generative language abilities with specialized CD functionalities. Our model successfully unifies both BCD and SCD tasks through the introduction of three special tokens: [T1], [T2], and [CHANGE]. Furthermore, UniChange utilizes text prompts to guide the identification of change categories, eliminating the reliance on predefined classification heads. This design allows UniChange to effectively acquire knowledge from multi-source datasets, even when their class definitions conflict. Experiments on four public benchmarks (WHU-CD, S2Looking, LEVIR-CD+, and SECOND) demonstrate SOTA performance, achieving IoU scores of 90.41, 53.04, 78.87, and 57.62, respectively, surpassing all previous methods. The code is available at https://github.com/Erxucomeon/UniChange.
>
---
#### [replaced 103] VGGTFace: Topologically Consistent Facial Geometry Reconstruction in the Wild
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20366v2](https://arxiv.org/pdf/2511.20366v2)**

> **作者:** Xin Ming; Yuxuan Han; Tianyu Huang; Feng Xu
>
> **摘要:** Reconstructing topologically consistent facial geometry is crucial for the digital avatar creation pipelines. Existing methods either require tedious manual efforts, lack generalization to in-the-wild data, or are constrained by the limited expressiveness of 3D Morphable Models. To address these limitations, we propose VGGTFace, an automatic approach that innovatively applies the 3D foundation model, i.e. VGGT, for topologically consistent facial geometry reconstruction from in-the-wild multi-view images captured by everyday users. Our key insight is that, by leveraging VGGT, our method naturally inherits strong generalization ability and expressive power from its large-scale training and point map representation. However, it is unclear how to reconstruct a topologically consistent mesh from VGGT, as the topology information is missing in its prediction. To this end, we augment VGGT with Pixel3DMM for injecting topology information via pixel-aligned UV values. In this manner, we convert the pixel-aligned point map of VGGT to a point cloud with topology. Tailored to this point cloud with known topology, we propose a novel Topology-Aware Bundle Adjustment strategy to fuse them, where we construct a Laplacian energy for the Bundle Adjustment objective. Our method achieves high-quality reconstruction in 10 seconds for 16 views on a single NVIDIA RTX 4090. Experiments demonstrate state-of-the-art results on benchmarks and impressive generalization to in-the-wild data. Code is available at https://github.com/grignarder/vggtface.
>
---
#### [replaced 104] Adaptive Object Detection for Indoor Navigation Assistance: A Performance Evaluation of Real-Time Algorithms
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2501.18444v2](https://arxiv.org/pdf/2501.18444v2)**

> **作者:** Abhinav Pratap; Sushant Kumar; Suchinton Chakravarty
>
> **备注:** 5 pages, 2 figures, 3 tables
>
> **摘要:** This study addresses the need for accurate and efficient object detection in assistive technologies for visually impaired individuals. We evaluate four real-time object detection algorithms YOLO, SSD, Faster R-CNN, and Mask R-CNN within the context of indoor navigation assistance. Using the Indoor Objects Detection dataset, we analyze detection accuracy, processing speed, and adaptability to indoor environments. Our findings highlight the trade-offs between precision and efficiency, offering insights into selecting optimal algorithms for realtime assistive navigation. This research advances adaptive machine learning applications, enhancing indoor navigation solutions for the visually impaired and promoting accessibility.
>
---
#### [replaced 105] Open Vocabulary Monocular 3D Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.16833v2](https://arxiv.org/pdf/2411.16833v2)**

> **作者:** Jin Yao; Hao Gu; Xuweiyi Chen; Jiayun Wang; Zezhou Cheng
>
> **备注:** 3DV 2026, Project page: https://cvlab.cs.virginia.edu/ovmono3d
>
> **摘要:** We propose and study open-vocabulary monocular 3D detection, a novel task that aims to detect objects of any categores in metric 3D space from a single RGB image. Existing 3D object detectors either rely on costly sensors such as LiDAR or multi-view setups, or remain confined to closed vocabularies settings with limited categories, restricting their applicability. We identify two key challenges in this new setting. First, the scarcity of 3D bounding box annotations limits the ability to train generalizable models. To reduce dependence on 3D supervision, we propose a framework that effectively integrates pretrained 2D and 3D vision foundation models. Second, missing labels and semantic ambiguities (\eg, table vs. desk) in existing datasets hinder reliable evaluation. To address this, we design a novel metric that captures model performance while mitigating annotation issues. Our approach achieves state-of-the-art results in zero-shot 3D detection of novel categories as well as in-domain detection on seen classes. We hope our method provides a strong baseline and our evaluation protocol establishes a reliable benchmark for future research.
>
---
#### [replaced 106] UniGame: Turning a Unified Multimodal Model Into Its Own Adversary
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19413v2](https://arxiv.org/pdf/2511.19413v2)**

> **作者:** Zhaolong Su; Wang Lu; Hao Chen; Sharon Li; Jindong Wang
>
> **摘要:** Unified Multimodal Models (UMMs) have shown impressive performance in both understanding and generation with a single architecture. However, UMMs still exhibit a fundamental inconsistency: understanding favors compact embeddings, whereas generation favors reconstruction-rich representations. This structural trade-off produces misaligned decision boundaries, degraded cross-modal coherence, and heightened vulnerability under distributional and adversarial shifts. In this paper, we present UniGame, a self-adversarial post-training framework that directly targets the inconsistencies. By applying a lightweight perturber at the shared token interface, UniGame enables the generation branch to actively seek and challenge fragile understanding, turning the model itself into its own adversary. Experiments demonstrate that UniGame significantly improves the consistency (+4.6%). Moreover, it also achieves substantial improvements in understanding (+3.6%), generation (+0.02), out-of-distribution and adversarial robustness (+4.8% and +6.2% on NaturalBench and AdVQA). The framework is architecture-agnostic, introduces less than 1% additional parameters, and is complementary to existing post-training methods. These results position adversarial self-play as a general and effective principle for enhancing the coherence, stability, and unified competence of future multimodal foundation models. The official code is available at: https://github.com/AIFrontierLab/UniGame
>
---
#### [replaced 107] Restora-Flow: Mask-Guided Image Restoration with Flow Matching
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20152v2](https://arxiv.org/pdf/2511.20152v2)**

> **作者:** Arnela Hadzic; Franz Thaler; Lea Bogensperger; Simon Johannes Joham; Martin Urschler
>
> **备注:** Accepted for WACV 2026
>
> **摘要:** Flow matching has emerged as a promising generative approach that addresses the lengthy sampling times associated with state-of-the-art diffusion models and enables a more flexible trajectory design, while maintaining high-quality image generation. This capability makes it suitable as a generative prior for image restoration tasks. Although current methods leveraging flow models have shown promising results in restoration, some still suffer from long processing times or produce over-smoothed results. To address these challenges, we introduce Restora-Flow, a training-free method that guides flow matching sampling by a degradation mask and incorporates a trajectory correction mechanism to enforce consistency with degraded inputs. We evaluate our approach on both natural and medical datasets across several image restoration tasks involving a mask-based degradation, i.e., inpainting, super-resolution and denoising. We show superior perceptual quality and processing time compared to diffusion and flow matching-based reference methods.
>
---
