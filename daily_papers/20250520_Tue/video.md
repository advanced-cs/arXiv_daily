# 计算机视觉 cs.CV

- **最新发布 264 篇**

- **更新 158 篇**

## 最新发布

#### [new 001] Questioning Representational Optimism in Deep Learning: The Fractured Entangled Representation Hypothesis
- **分类: cs.CV; cs.LG; cs.NE**

- **简介: 该论文属于表示学习分析，质疑性能提升是否改善模型内部表示。比较演化与SGD训练网络在生成单图像任务中的神经元行为，发现SGD导致断裂纠缠表示(FER)，可能损害泛化等能力，而演化网络接近统一分解表示。研究挑战了表示乐观主义，主张理解并缓解FER。**

- **链接: [http://arxiv.org/pdf/2505.11581v1](http://arxiv.org/pdf/2505.11581v1)**

> **作者:** Akarsh Kumar; Jeff Clune; Joel Lehman; Kenneth O. Stanley
>
> **备注:** 43 pages, 25 figures
>
> **摘要:** Much of the excitement in modern AI is driven by the observation that scaling up existing systems leads to better performance. But does better performance necessarily imply better internal representations? While the representational optimist assumes it must, this position paper challenges that view. We compare neural networks evolved through an open-ended search process to networks trained via conventional stochastic gradient descent (SGD) on the simple task of generating a single image. This minimal setup offers a unique advantage: each hidden neuron's full functional behavior can be easily visualized as an image, thus revealing how the network's output behavior is internally constructed neuron by neuron. The result is striking: while both networks produce the same output behavior, their internal representations differ dramatically. The SGD-trained networks exhibit a form of disorganization that we term fractured entangled representation (FER). Interestingly, the evolved networks largely lack FER, even approaching a unified factored representation (UFR). In large models, FER may be degrading core model capacities like generalization, creativity, and (continual) learning. Therefore, understanding and mitigating FER could be critical to the future of representation learning.
>
---
#### [new 002] Event-Driven Dynamic Scene Depth Completion
- **分类: cs.CV**

- **简介: 该论文研究动态场景深度补全任务，解决传统RGB-D传感器在快速运动下对齐不准、深度质量差的问题。提出首个事件驱动框架EventDC，含事件调制对齐（EMA）和局部深度滤波（LDF）模块，利用事件数据优化卷积偏移与权重，提升特征对齐与深度估计效果，并构建了首个事件深度补全数据集。实验验证了其优越性。**

- **链接: [http://arxiv.org/pdf/2505.13279v1](http://arxiv.org/pdf/2505.13279v1)**

> **作者:** Zhiqiang Yan; Jianhao Jiao; Zhengxue Wang; Gim Hee Lee
>
> **备注:** 9 pages
>
> **摘要:** Depth completion in dynamic scenes poses significant challenges due to rapid ego-motion and object motion, which can severely degrade the quality of input modalities such as RGB images and LiDAR measurements. Conventional RGB-D sensors often struggle to align precisely and capture reliable depth under such conditions. In contrast, event cameras with their high temporal resolution and sensitivity to motion at the pixel level provide complementary cues that are %particularly beneficial in dynamic environments.To this end, we propose EventDC, the first event-driven depth completion framework. It consists of two key components: Event-Modulated Alignment (EMA) and Local Depth Filtering (LDF). Both modules adaptively learn the two fundamental components of convolution operations: offsets and weights conditioned on motion-sensitive event streams. In the encoder, EMA leverages events to modulate the sampling positions of RGB-D features to achieve pixel redistribution for improved alignment and fusion. In the decoder, LDF refines depth estimations around moving objects by learning motion-aware masks from events. Additionally, EventDC incorporates two loss terms to further benefit global alignment and enhance local depth recovery. Moreover, we establish the first benchmark for event-based depth completion comprising one real-world and two synthetic datasets to facilitate future research. Extensive experiments on this benchmark demonstrate the superiority of our EventDC.
>
---
#### [new 003] Are vision language models robust to uncertain inputs?
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉语言模型（VLMs）对不确定输入的鲁棒性，属于不确定性量化任务。针对模型面对模糊/异常输入时过度自信生成错误响应的问题，通过异常检测和模糊分类实验发现：大规模VLMs虽改进明显，但指令服从性导致幻觉；自然图像通过提示“弃权”可显著提升可靠性，而领域任务因专业知识缺失受限。最终提出基于描述多样性的内部不确定性评估方法，无需标注即可预测模型弃权能力。**

- **链接: [http://arxiv.org/pdf/2505.11804v1](http://arxiv.org/pdf/2505.11804v1)**

> **作者:** Xi Wang; Eric Nalisnick
>
> **摘要:** Robustness against uncertain and ambiguous inputs is a critical challenge for deep learning models. While recent advancements in large scale vision language models (VLMs, e.g. GPT4o) might suggest that increasing model and training dataset size would mitigate this issue, our empirical evaluation shows a more complicated picture. Testing models using two classic uncertainty quantification tasks, anomaly detection and classification under inherently ambiguous conditions, we find that newer and larger VLMs indeed exhibit improved robustness compared to earlier models, but still suffer from a tendency to strictly follow instructions, often causing them to hallucinate confident responses even when faced with unclear or anomalous inputs. Remarkably, for natural images such as ImageNet, this limitation can be overcome without pipeline modifications: simply prompting models to abstain from uncertain predictions enables significant reliability gains, achieving near-perfect robustness in several settings. However, for domain-specific tasks such as galaxy morphology classification, a lack of specialized knowledge prevents reliable uncertainty estimation. Finally, we propose a novel mechanism based on caption diversity to reveal a model's internal uncertainty, enabling practitioners to predict when models will successfully abstain without relying on labeled data.
>
---
#### [new 004] Temporal-Oriented Recipe for Transferring Large Vision-Language Model to Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决大型视觉语言模型（LVLM）在时间建模能力上的不足。通过分析影响时间理解的关键组件（如视觉编码器与语言模型间的接口），提出时间导向的训练方案及升级接口，显著提升了现有模型在视频任务中的性能。**

- **链接: [http://arxiv.org/pdf/2505.12605v1](http://arxiv.org/pdf/2505.12605v1)**

> **作者:** Thong Nguyen; Zhiyuan Hu; Xu Lin; Cong-Duy Nguyen; See-Kiong Ng; Luu Anh Tuan
>
> **备注:** In Progress
>
> **摘要:** Recent years have witnessed outstanding advances of large vision-language models (LVLMs). In order to tackle video understanding, most of them depend upon their implicit temporal understanding capacity. As such, they have not deciphered important components that contribute to temporal understanding ability, which might limit the potential of these LVLMs for video understanding. In this work, we conduct a thorough empirical study to demystify crucial components that influence the temporal understanding of LVLMs. Our empirical study reveals that significant impacts are centered around the intermediate interface between the visual encoder and the large language model. Building on these insights, we propose a temporal-oriented recipe that encompasses temporal-oriented training schemes and an upscaled interface. Our final model developed using our recipe significantly enhances previous LVLMs on standard video understanding tasks.
>
---
#### [new 005] Use as Many Surrogates as You Want: Selective Ensemble Attack to Unleash Transferability without Sacrificing Resource Efficiency
- **分类: cs.CV**

- **简介: 该论文研究对抗攻击中的替代模型集成方法，解决迁移性与效率的权衡问题。传统方法因固定模型组合导致资源浪费，本文提出动态选择策略SEA，解耦迭代内/跨迭代模型多样性，固定单次计算量提升迁移性。实验证明其在ImageNet和真实系统中高效有效。**

- **链接: [http://arxiv.org/pdf/2505.12644v1](http://arxiv.org/pdf/2505.12644v1)**

> **作者:** Bo Yang; Hengwei Zhang; Jindong Wang; Yuchen Ren; Chenhao Lin; Chao Shen; Zhengyu Zhao
>
> **摘要:** In surrogate ensemble attacks, using more surrogate models yields higher transferability but lower resource efficiency. This practical trade-off between transferability and efficiency has largely limited existing attacks despite many pre-trained models are easily accessible online. In this paper, we argue that such a trade-off is caused by an unnecessary common assumption, i.e., all models should be identical across iterations. By lifting this assumption, we can use as many surrogates as we want to unleash transferability without sacrificing efficiency. Concretely, we propose Selective Ensemble Attack (SEA), which dynamically selects diverse models (from easily accessible pre-trained models) across iterations based on our new interpretation of decoupling within-iteration and cross-iteration model diversity.In this way, the number of within-iteration models is fixed for maintaining efficiency, while only cross-iteration model diversity is increased for higher transferability. Experiments on ImageNet demonstrate the superiority of SEA in various scenarios. For example, when dynamically selecting 4 from 20 accessible models, SEA yields 8.5% higher transferability than existing attacks under the same efficiency. The superiority of SEA also generalizes to real-world systems, such as commercial vision APIs and large vision-language models. Overall, SEA opens up the possibility of adaptively balancing transferability and efficiency according to specific resource requirements.
>
---
#### [new 006] SpatialLLM: From Multi-modality Data to Urban Spatial Intelligence
- **分类: cs.CV**

- **简介: 论文提出SpatialLLM，解决复杂城市场景中依赖专业工具或知识的空间智能任务。通过多模态数据生成结构化场景描述，驱动预训练大模型实现零样本分析，支持城市规划、交通管理等任务，无需训练或专家干预。实验验证其有效性，并指出知识广度、上下文和推理能力是性能关键。**

- **链接: [http://arxiv.org/pdf/2505.12703v1](http://arxiv.org/pdf/2505.12703v1)**

> **作者:** Jiabin Chen; Haiping Wang; Jinpeng Li; Yuan Liu; Zhen Dong; Bisheng Yang
>
> **摘要:** We propose SpatialLLM, a novel approach advancing spatial intelligence tasks in complex urban scenes. Unlike previous methods requiring geographic analysis tools or domain expertise, SpatialLLM is a unified language model directly addressing various spatial intelligence tasks without any training, fine-tuning, or expert intervention. The core of SpatialLLM lies in constructing detailed and structured scene descriptions from raw spatial data to prompt pre-trained LLMs for scene-based analysis. Extensive experiments show that, with our designs, pretrained LLMs can accurately perceive spatial distribution information and enable zero-shot execution of advanced spatial intelligence tasks, including urban planning, ecological analysis, traffic management, etc. We argue that multi-field knowledge, context length, and reasoning ability are key factors influencing LLM performances in urban analysis. We hope that SpatialLLM will provide a novel viable perspective for urban intelligent analysis and management. The code and dataset are available at https://github.com/WHU-USI3DV/SpatialLLM.
>
---
#### [new 007] Black-box Adversaries from Latent Space: Unnoticeable Attacks on Human Pose and Shape Estimation
- **分类: cs.CV**

- **简介: 该论文针对人体姿态估计模型的安全漏洞，提出一种隐蔽黑盒攻击方法(UBA)。任务属于对抗攻击研究，解决现有攻击需白盒访问或扰动明显的问题。通过潜在空间生成优化对抗噪声，仅依赖模型输出查询实现隐蔽有效攻击，实验证明其显著增加模型误差，揭示了数字人生成系统的安全风险。**

- **链接: [http://arxiv.org/pdf/2505.12009v1](http://arxiv.org/pdf/2505.12009v1)**

> **作者:** Zhiying Li; Guanggang Geng; Yeying Jin; Zhizhi Guo; Bruce Gu; Jidong Huo; Zhaoxin Fan; Wenjun Wu
>
> **备注:** 17 pages, 6 figures
>
> **摘要:** Expressive human pose and shape (EHPS) estimation is vital for digital human generation, particularly in live-streaming applications. However, most existing EHPS models focus primarily on minimizing estimation errors, with limited attention on potential security vulnerabilities. Current adversarial attacks on EHPS models often require white-box access (e.g., model details or gradients) or generate visually conspicuous perturbations, limiting their practicality and ability to expose real-world security threats. To address these limitations, we propose a novel Unnoticeable Black-Box Attack (UBA) against EHPS models. UBA leverages the latent-space representations of natural images to generate an optimal adversarial noise pattern and iteratively refine its attack potency along an optimized direction in digital space. Crucially, this process relies solely on querying the model's output, requiring no internal knowledge of the EHPS architecture, while guiding the noise optimization toward greater stealth and effectiveness. Extensive experiments and visual analyses demonstrate the superiority of UBA. Notably, UBA increases the pose estimation errors of EHPS models by 17.27%-58.21% on average, revealing critical vulnerabilities. These findings underscore the urgent need to address and mitigate security risks associated with digital human generation systems.
>
---
#### [new 008] Accelerating Diffusion-based Super-Resolution with Dynamic Time-Spatial Sampling
- **分类: cs.CV**

- **简介: 该论文针对扩散模型在超分辨率任务中的高计算成本问题，提出时空动态采样策略(TSS)。通过分析高频信号恢复的时空依赖性，设计时间动态采样优化纹理迭代分配，结合基于图像内容的空间动态采样，在减少50%迭代次数的同时提升重建质量，实现无额外训练的高效超分辨率。**

- **链接: [http://arxiv.org/pdf/2505.12048v1](http://arxiv.org/pdf/2505.12048v1)**

> **作者:** Rui Qin; Qijie Wang; Ming Sun; Haowei Zhu; Chao Zhou; Bin Wang
>
> **摘要:** Diffusion models have gained attention for their success in modeling complex distributions, achieving impressive perceptual quality in SR tasks. However, existing diffusion-based SR methods often suffer from high computational costs, requiring numerous iterative steps for training and inference. Existing acceleration techniques, such as distillation and solver optimization, are generally task-agnostic and do not fully leverage the specific characteristics of low-level tasks like super-resolution (SR). In this study, we analyze the frequency- and spatial-domain properties of diffusion-based SR methods, revealing key insights into the temporal and spatial dependencies of high-frequency signal recovery. Specifically, high-frequency details benefit from concentrated optimization during early and late diffusion iterations, while spatially textured regions demand adaptive denoising strategies. Building on these observations, we propose the Time-Spatial-aware Sampling strategy (TSS) for the acceleration of Diffusion SR without any extra training cost. TSS combines Time Dynamic Sampling (TDS), which allocates more iterations to refining textures, and Spatial Dynamic Sampling (SDS), which dynamically adjusts strategies based on image content. Extensive evaluations across multiple benchmarks demonstrate that TSS achieves state-of-the-art (SOTA) performance with significantly fewer iterations, improving MUSIQ scores by 0.2 - 3.0 and outperforming the current acceleration methods with only half the number of steps.
>
---
#### [new 009] SPKLIP: Aligning Spike Video Streams with Natural Language
- **分类: cs.CV**

- **简介: 该论文研究脉冲视频-语言对齐任务，解决现有模型因模态不匹配导致的性能不足问题。提出SPKLIP架构，通过分层特征提取器建模脉冲流时序动态，结合对比学习实现跨模态对齐，并引入脉冲神经网络提升能效。实验验证其在少样本学习和能效上的优势。**

- **链接: [http://arxiv.org/pdf/2505.12656v1](http://arxiv.org/pdf/2505.12656v1)**

> **作者:** Yongchang Gao; Meiling Jin; Zhaofei Yu; Tiejun Huang; Guozhang Chen
>
> **摘要:** Spike cameras offer unique sensing capabilities but their sparse, asynchronous output challenges semantic understanding, especially for Spike Video-Language Alignment (Spike-VLA) where models like CLIP underperform due to modality mismatch. We introduce SPKLIP, the first architecture specifically for Spike-VLA. SPKLIP employs a hierarchical spike feature extractor that adaptively models multi-scale temporal dynamics in event streams, and uses spike-text contrastive learning to directly align spike video with language, enabling effective few-shot learning. A full-spiking visual encoder variant, integrating SNN components into our pipeline, demonstrates enhanced energy efficiency. Experiments show state-of-the-art performance on benchmark spike datasets and strong few-shot generalization on a newly contributed real-world dataset. SPKLIP's energy efficiency highlights its potential for neuromorphic deployment, advancing event-based multimodal research. The source code and dataset are available at [link removed for anonymity].
>
---
#### [new 010] PRS-Med: Position Reasoning Segmentation with Vision-Language Model in Medical Imaging
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割与多模态推理任务，旨在解决现有方法在自然语言交互和空间关系推理上的不足。提出PRS-Med框架，结合视觉语言模型实现精准分割与空间推理，并构建MMRS数据集填补医学定位推理数据空白，在六种影像模态中表现超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11872v1](http://arxiv.org/pdf/2505.11872v1)**

> **作者:** Quoc-Huy Trinh; Minh-Van Nguyen; Jung Peng; Ulas Bagci; Debesh Jha
>
> **摘要:** Recent advancements in prompt-based medical image segmentation have enabled clinicians to identify tumors using simple input like bounding boxes or text prompts. However, existing methods face challenges when doctors need to interact through natural language or when position reasoning is required - understanding spatial relationships between anatomical structures and pathologies. We present PRS-Med, a framework that integrates vision-language models with segmentation capabilities to generate both accurate segmentation masks and corresponding spatial reasoning outputs. Additionally, we introduce the MMRS dataset (Multimodal Medical in Positional Reasoning Segmentation), which provides diverse, spatially-grounded question-answer pairs to address the lack of position reasoning data in medical imaging. PRS-Med demonstrates superior performance across six imaging modalities (CT, MRI, X-ray, ultrasound, endoscopy, RGB), significantly outperforming state-of-the-art methods in both segmentation accuracy and position reasoning. Our approach enables intuitive doctor-system interaction through natural language, facilitating more efficient diagnoses. Our dataset pipeline, model, and codebase will be released to foster further research in spatially-aware multimodal reasoning for medical applications.
>
---
#### [new 011] GlobalGeoTree: A Multi-Granular Vision-Language Dataset for Global Tree Species Classification
- **分类: cs.CV**

- **简介: 该论文针对全球树种分类任务，解决遥感数据标注不足问题。提出GlobalGeoTree数据集（630万样本，覆盖21,001种），整合卫星时序影像与环境变量，并开发GeoTreeCLIP视觉语言模型，在零/小样本分类性能显著优于现有模型，为生物多样性研究提供基准工具。**

- **链接: [http://arxiv.org/pdf/2505.12513v1](http://arxiv.org/pdf/2505.12513v1)**

> **作者:** Yang Mu; Zhitong Xiong; Yi Wang; Muhammad Shahzad; Franz Essl; Mark van Kleunen; Xiao Xiang Zhu
>
> **摘要:** Global tree species mapping using remote sensing data is vital for biodiversity monitoring, forest management, and ecological research. However, progress in this field has been constrained by the scarcity of large-scale, labeled datasets. To address this, we introduce GlobalGeoTree, a comprehensive global dataset for tree species classification. GlobalGeoTree comprises 6.3 million geolocated tree occurrences, spanning 275 families, 2,734 genera, and 21,001 species across the hierarchical taxonomic levels. Each sample is paired with Sentinel-2 image time series and 27 auxiliary environmental variables, encompassing bioclimatic, geographic, and soil data. The dataset is partitioned into GlobalGeoTree-6M for model pretraining and curated evaluation subsets, primarily GlobalGeoTree-10kEval for zero-shot and few-shot benchmarking. To demonstrate the utility of the dataset, we introduce a baseline model, GeoTreeCLIP, which leverages paired remote sensing data and taxonomic text labels within a vision-language framework pretrained on GlobalGeoTree-6M. Experimental results show that GeoTreeCLIP achieves substantial improvements in zero- and few-shot classification on GlobalGeoTree-10kEval over existing advanced models. By making the dataset, models, and code publicly available, we aim to establish a benchmark to advance tree species classification and foster innovation in biodiversity research and ecological applications.
>
---
#### [new 012] Joint Depth and Reflectivity Estimation using Single-Photon LiDAR
- **分类: cs.CV**

- **简介: 该论文属于单光子LiDAR的3D重建任务，解决动态场景中传统方法分离估计深度/反射率及依赖静态3D直方图的局限性。提出SPLiDER方法，通过理论证明两者相关性，并设计联合估计框架直接处理时间戳数据，在快速运动场景中实现更优的深度与反射率同步重建。**

- **链接: [http://arxiv.org/pdf/2505.13250v1](http://arxiv.org/pdf/2505.13250v1)**

> **作者:** Hashan K. Weerasooriya; Prateek Chennuri; Weijian Zhang; Istvan Gyongy; Stanley H. Chan
>
> **摘要:** Single-Photon Light Detection and Ranging (SP-LiDAR is emerging as a leading technology for long-range, high-precision 3D vision tasks. In SP-LiDAR, timestamps encode two complementary pieces of information: pulse travel time (depth) and the number of photons reflected by the object (reflectivity). Existing SP-LiDAR reconstruction methods typically recover depth and reflectivity separately or sequentially use one modality to estimate the other. Moreover, the conventional 3D histogram construction is effective mainly for slow-moving or stationary scenes. In dynamic scenes, however, it is more efficient and effective to directly process the timestamps. In this paper, we introduce an estimation method to simultaneously recover both depth and reflectivity in fast-moving scenes. We offer two contributions: (1) A theoretical analysis demonstrating the mutual correlation between depth and reflectivity and the conditions under which joint estimation becomes beneficial. (2) A novel reconstruction method, "SPLiDER", which exploits the shared information to enhance signal recovery. On both synthetic and real SP-LiDAR data, our method outperforms existing approaches, achieving superior joint reconstruction quality.
>
---
#### [new 013] DC-Seg: Disentangled Contrastive Learning for Brain Tumor Segmentation with Missing Modalities
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决多模态脑肿瘤分割中模态缺失的问题。提出DC-Seg方法，通过解耦对比学习分离模态不变解剖特征和模态特异性特征，并加入分割正则化提升缺失模态的鲁棒性，在BraTS和WMH数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11921v1](http://arxiv.org/pdf/2505.11921v1)**

> **作者:** Haitao Li; Ziyu Li; Yiheng Mao; Zhengyao Ding; Zhengxing Huang
>
> **摘要:** Accurate segmentation of brain images typically requires the integration of complementary information from multiple image modalities. However, clinical data for all modalities may not be available for every patient, creating a significant challenge. To address this, previous studies encode multiple modalities into a shared latent space. While somewhat effective, it remains suboptimal, as each modality contains distinct and valuable information. In this study, we propose DC-Seg (Disentangled Contrastive Learning for Segmentation), a new method that explicitly disentangles images into modality-invariant anatomical representation and modality-specific representation, by using anatomical contrastive learning and modality contrastive learning respectively. This solution improves the separation of anatomical and modality-specific features by considering the modality gaps, leading to more robust representations. Furthermore, we introduce a segmentation-based regularizer that enhances the model's robustness to missing modalities. Extensive experiments on the BraTS 2020 and a private white matter hyperintensity(WMH) segmentation dataset demonstrate that DC-Seg outperforms state-of-the-art methods in handling incomplete multimodal brain tumor segmentation tasks with varying missing modalities, while also demonstrate strong generalizability in WMH segmentation. The code is available at https://github.com/CuCl-2/DC-Seg.
>
---
#### [new 014] Denoising Diffusion Probabilistic Model for Point Cloud Compression at Low Bit-Rates
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对低码率点云压缩任务，提出基于去噪扩散概率模型（DDPM-PCC）的解决方案。通过PointNet编码器生成条件向量并采用可学习量化器降低码率，在保持重建质量的同时实现高效压缩。实验表明该方法在ShapeNet等数据集上优于现有技术，改善了低码率下的率失真性能，代码已开源。**

- **链接: [http://arxiv.org/pdf/2505.13316v1](http://arxiv.org/pdf/2505.13316v1)**

> **作者:** Gabriele Spadaro; Alberto Presta; Jhony H. Giraldo; Marco Grangetto; Wei Hu; Giuseppe Valenzise; Attilio Fiandrotti; Enzo Tartaglione
>
> **备注:** 6 pages, 5 figures, accepted at ICME 2025
>
> **摘要:** Efficient compression of low-bit-rate point clouds is critical for bandwidth-constrained applications. However, existing techniques mainly focus on high-fidelity reconstruction, requiring many bits for compression. This paper proposes a "Denoising Diffusion Probabilistic Model" (DDPM) architecture for point cloud compression (DDPM-PCC) at low bit-rates. A PointNet encoder produces the condition vector for the generation, which is then quantized via a learnable vector quantizer. This configuration allows to achieve a low bitrates while preserving quality. Experiments on ShapeNet and ModelNet40 show improved rate-distortion at low rates compared to standardized and state-of-the-art approaches. We publicly released the code at https://github.com/EIDOSLAB/DDPM-PCC.
>
---
#### [new 015] Towards Low-Latency Event Stream-based Visual Object Tracking: A Slow-Fast Approach
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉目标跟踪任务，针对传统帧相机方法延迟高、资源消耗大的问题，提出基于事件相机的双模慢快跟踪框架(SFTrack)。结合图表示学习与轻量网络设计，通过知识蒸馏整合高精度慢跟踪器和低延迟快跟踪器，实现在不同资源环境下的高效实时跟踪。**

- **链接: [http://arxiv.org/pdf/2505.12903v1](http://arxiv.org/pdf/2505.12903v1)**

> **作者:** Shiao Wang; Xiao Wang; Liye Jin; Bo Jiang; Lin Zhu; Lan Chen; Yonghong Tian; Bin Luo
>
> **摘要:** Existing tracking algorithms typically rely on low-frame-rate RGB cameras coupled with computationally intensive deep neural network architectures to achieve effective tracking. However, such frame-based methods inherently face challenges in achieving low-latency performance and often fail in resource-constrained environments. Visual object tracking using bio-inspired event cameras has emerged as a promising research direction in recent years, offering distinct advantages for low-latency applications. In this paper, we propose a novel Slow-Fast Tracking paradigm that flexibly adapts to different operational requirements, termed SFTrack. The proposed framework supports two complementary modes, i.e., a high-precision slow tracker for scenarios with sufficient computational resources, and an efficient fast tracker tailored for latency-aware, resource-constrained environments. Specifically, our framework first performs graph-based representation learning from high-temporal-resolution event streams, and then integrates the learned graph-structured information into two FlashAttention-based vision backbones, yielding the slow and fast trackers, respectively. The fast tracker achieves low latency through a lightweight network design and by producing multiple bounding box outputs in a single forward pass. Finally, we seamlessly combine both trackers via supervised fine-tuning and further enhance the fast tracker's performance through a knowledge distillation strategy. Extensive experiments on public benchmarks, including FE240, COESOT, and EventVOT, demonstrate the effectiveness and efficiency of our proposed method across different real-world scenarios. The source code has been released on https://github.com/Event-AHU/SlowFast_Event_Track.
>
---
#### [new 016] Exploring Sparsity for Parameter Efficient Fine Tuning Using Wavelets
- **分类: cs.CV; cs.AI; cs.LG; eess.IV; eess.SP**

- **简介: 该论文研究参数高效微调任务，解决现有方法（如LoRA）在低参数量下性能不足的问题。提出WaveFT方法，通过小波变换在残差矩阵的频域学习稀疏更新，实现参数量的精细控制。相比直接权重域稀疏方法SHiRA，WaveFT在Stable Diffusion XL个性化图像生成任务中显著提升生成质量，尤其在极低参数量时效果突出。**

- **链接: [http://arxiv.org/pdf/2505.12532v1](http://arxiv.org/pdf/2505.12532v1)**

> **作者:** Ahmet Bilican; M. Akın Yılmaz; A. Murat Tekalp; R. Gökberk Cinbiş
>
> **摘要:** Efficiently adapting large foundation models is critical, especially with tight compute and memory budgets. Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA offer limited granularity and effectiveness in few-parameter regimes. We propose Wavelet Fine-Tuning (WaveFT), a novel PEFT method that learns highly sparse updates in the wavelet domain of residual matrices. WaveFT allows precise control of trainable parameters, offering fine-grained capacity adjustment and excelling with remarkably low parameter count, potentially far fewer than LoRA's minimum -- ideal for extreme parameter-efficient scenarios. In order to demonstrate the effect of the wavelet transform, we compare WaveFT with a special case, called SHiRA, that entails applying sparse updates directly in the weight domain. Evaluated on personalized text-to-image generation using Stable Diffusion XL as baseline, WaveFT significantly outperforms LoRA and other PEFT methods, especially at low parameter counts; achieving superior subject fidelity, prompt alignment, and image diversity.
>
---
#### [new 017] Learning to Adapt to Position Bias in Vision Transformer Classifiers
- **分类: cs.CV**

- **简介: 该论文研究图像分类任务中Vision Transformers的位置偏差问题，分析数据集中位置信息对性能的影响。通过提出Position-SHAP量化位置偏差，发现不同数据集对位置嵌入的需求差异，进而设计Auto-PE方法动态调节位置嵌入强度，实现自适应位置信息学习，提升分类精度。**

- **链接: [http://arxiv.org/pdf/2505.13137v1](http://arxiv.org/pdf/2505.13137v1)**

> **作者:** Robert-Jan Bruintjes; Jan van Gemert
>
> **摘要:** How discriminative position information is for image classification depends on the data. On the one hand, the camera position is arbitrary and objects can appear anywhere in the image, arguing for translation invariance. At the same time, position information is key for exploiting capture/center bias, and scene layout, e.g.: the sky is up. We show that position bias, the level to which a dataset is more easily solved when positional information on input features is used, plays a crucial role in the performance of Vision Transformers image classifiers. To investigate, we propose Position-SHAP, a direct measure of position bias by extending SHAP to work with position embeddings. We show various levels of position bias in different datasets, and find that the optimal choice of position embedding depends on the position bias apparent in the dataset. We therefore propose Auto-PE, a single-parameter position embedding extension, which allows the position embedding to modulate its norm, enabling the unlearning of position information. Auto-PE combines with existing PEs to match or improve accuracy on classification datasets.
>
---
#### [new 018] A Study on the Refining Handwritten Font by Mixing Font Styles
- **分类: cs.CV**

- **简介: 该论文属于字体生成任务，旨在解决手写字体可读性差的问题。提出FontFusionGAN方法，通过混合手写体与印刷体特征，利用GAN生成兼具美观与清晰度的新字体。实验证明其能显著提升可读性，同时保留手写风格，适用于文档处理、读写辅助及多语言字体转换等场景。**

- **链接: [http://arxiv.org/pdf/2505.12834v1](http://arxiv.org/pdf/2505.12834v1)**

> **作者:** Avinash Kumar; Kyeolhee Kang; Ammar ul Hassan; Jaeyoung Choi
>
> **备注:** 4 pages, 3 figures, MITA 2023 (The 19th International Conference on Multimedia Information Technology and Applications July. 11 ~ July 14, 2023, Technical University of Ostrava, Ostrava, Czech)
>
> **摘要:** Handwritten fonts have a distinct expressive character, but they are often difficult to read due to unclear or inconsistent handwriting. FontFusionGAN (FFGAN) is a novel method for improving handwritten fonts by combining them with printed fonts. Our method implements generative adversarial network (GAN) to generate font that mix the desirable features of handwritten and printed fonts. By training the GAN on a dataset of handwritten and printed fonts, it can generate legible and visually appealing font images. We apply our method to a dataset of handwritten fonts and demonstrate that it significantly enhances the readability of the original fonts while preserving their unique aesthetic. Our method has the potential to improve the readability of handwritten fonts, which would be helpful for a variety of applications including document creation, letter writing, and assisting individuals with reading and writing difficulties. In addition to addressing the difficulties of font creation for languages with complex character sets, our method is applicable to other text-image-related tasks, such as font attribute control and multilingual font style transfer.
>
---
#### [new 019] Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于自监督学习领域，旨在解决噪声数据下模型鲁棒性问题。作者提出一种无需推理阶段去噪的框架：先训练去噪器构建数据课程（去噪→带噪样本），结合教师正则化引导预训练模型（如DINOv2）学习噪声不变特征。实验显示该方法在极端噪声下线性分类准确率提升4.8%，且部署时无需保留去噪器。**

- **链接: [http://arxiv.org/pdf/2505.12191v1](http://arxiv.org/pdf/2505.12191v1)**

> **作者:** Wenquan Lu; Jiaqi Zhang; Hugues Van Assel; Randall Balestriero
>
> **摘要:** Self-Supervised Learning (SSL) has become a powerful solution to extract rich representations from unlabeled data. Yet, SSL research is mostly focused on clean, curated and high-quality datasets. As a result, applying SSL on noisy data remains a challenge, despite being crucial to applications such as astrophysics, medical imaging, geophysics or finance. In this work, we present a fully self-supervised framework that enables noise-robust representation learning without requiring a denoiser at inference or downstream fine-tuning. Our method first trains an SSL denoiser on noisy data, then uses it to construct a denoised-to-noisy data curriculum (i.e., training first on denoised, then noisy samples) for pretraining a SSL backbone (e.g., DINOv2), combined with a teacher-guided regularization that anchors noisy embeddings to their denoised counterparts. This process encourages the model to internalize noise robustness. Notably, the denoiser can be discarded after pretraining, simplifying deployment. On ImageNet-1k with ViT-B under extreme Gaussian noise ($\sigma=255$, SNR = 0.72 dB), our method improves linear probing accuracy by 4.8% over DINOv2, demonstrating that denoiser-free robustness can emerge from noise-aware pretraining. The code is available at https://github.com/wenquanlu/noisy_dinov2.
>
---
#### [new 020] Context-Aware Autoregressive Models for Multi-Conditional Image Generation
- **分类: cs.CV**

- **简介: 该论文研究多条件图像生成任务，旨在整合多种模态条件（如边缘、深度图）并提升生成效果。提出ContextAR框架，通过混合位置编码保持条件对齐，设计条件感知注意力降低计算量，支持任意条件组合，实验性能优于现有自回归和扩散模型方法。**

- **链接: [http://arxiv.org/pdf/2505.12274v1](http://arxiv.org/pdf/2505.12274v1)**

> **作者:** Yixiao Chen; Zhiyuan Ma; Guoli Jia; Che Jiang; Jianjun Li; Bowen Zhou
>
> **摘要:** Autoregressive transformers have recently shown impressive image generation quality and efficiency on par with state-of-the-art diffusion models. Unlike diffusion architectures, autoregressive models can naturally incorporate arbitrary modalities into a single, unified token sequence--offering a concise solution for multi-conditional image generation tasks. In this work, we propose $\textbf{ContextAR}$, a flexible and effective framework for multi-conditional image generation. ContextAR embeds diverse conditions (e.g., canny edges, depth maps, poses) directly into the token sequence, preserving modality-specific semantics. To maintain spatial alignment while enhancing discrimination among different condition types, we introduce hybrid positional encodings that fuse Rotary Position Embedding with Learnable Positional Embedding. We design Conditional Context-aware Attention to reduces computational complexity while preserving effective intra-condition perception. Without any fine-tuning, ContextAR supports arbitrary combinations of conditions during inference time. Experimental results demonstrate the powerful controllability and versatility of our approach, and show that the competitive perpormance than diffusion-based multi-conditional control approaches the existing autoregressive baseline across diverse multi-condition driven scenarios. Project page: $\href{https://context-ar.github.io/}{https://context-ar.github.io/.}$
>
---
#### [new 021] Attend to Not Attended: Structure-then-Detail Token Merging for Post-training DiT Acceleration
- **分类: cs.CV**

- **简介: 该论文针对扩散变换模型（DiT）计算成本高的问题，提出结构优先细节的SDTM方法，通过动态合并冗余特征实现加速。基于扩散模型去噪先验，设计分阶段token压缩策略，在保持图像质量的同时达到1.55倍加速，可适配各类DiT架构。**

- **链接: [http://arxiv.org/pdf/2505.11707v1](http://arxiv.org/pdf/2505.11707v1)**

> **作者:** Haipeng Fang; Sheng Tang; Juan Cao; Enshuo Zhang; Fan Tang; Tong-Yee Lee
>
> **备注:** Comments: 14 pages, 14 figures. Accepted by the Proceedings of the 42nd IEEE/CVF Conference on Computer Vision and Pattern Recognition
>
> **摘要:** Diffusion transformers have shown exceptional performance in visual generation but incur high computational costs. Token reduction techniques that compress models by sharing the denoising process among similar tokens have been introduced. However, existing approaches neglect the denoising priors of the diffusion models, leading to suboptimal acceleration and diminished image quality. This study proposes a novel concept: attend to prune feature redundancies in areas not attended by the diffusion process. We analyze the location and degree of feature redundancies based on the structure-then-detail denoising priors. Subsequently, we introduce SDTM, a structure-then-detail token merging approach that dynamically compresses feature redundancies. Specifically, we design dynamic visual token merging, compression ratio adjusting, and prompt reweighting for different stages. Served in a post-training way, the proposed method can be integrated seamlessly into any DiT architecture. Extensive experiments across various backbones, schedulers, and datasets showcase the superiority of our method, for example, it achieves 1.55 times acceleration with negligible impact on image quality. Project page: https://github.com/ICTMCG/SDTM.
>
---
#### [new 022] BusterX: MLLM-Powered AI-Generated Video Forgery Detection and Explanation
- **分类: cs.CV**

- **简介: 该论文属于AI生成视频检测任务，旨在解决现有数据集规模小、检测方法缺乏解释性的问题。提出了首个大规模高质量生成视频数据集GenBuster-200K，并开发了BusterX框架，通过多模态大语言模型与强化学习实现可解释的伪造检测和决策分析。**

- **链接: [http://arxiv.org/pdf/2505.12620v1](http://arxiv.org/pdf/2505.12620v1)**

> **作者:** Haiquan Wen; Yiwei He; Zhenglin Huang; Tianxiao Li; Zihan YU; Xingru Huang; Lu Qi; Baoyuan Wu; Xiangtai Li; Guangliang Cheng
>
> **摘要:** Advances in AI generative models facilitate super-realistic video synthesis, amplifying misinformation risks via social media and eroding trust in digital content. Several research works have explored new deepfake detection methods on AI-generated images to alleviate these risks. However, with the fast development of video generation models, such as Sora and WanX, there is currently a lack of large-scale, high-quality AI-generated video datasets for forgery detection. In addition, existing detection approaches predominantly treat the task as binary classification, lacking explainability in model decision-making and failing to provide actionable insights or guidance for the public. To address these challenges, we propose \textbf{GenBuster-200K}, a large-scale AI-generated video dataset featuring 200K high-resolution video clips, diverse latest generative techniques, and real-world scenes. We further introduce \textbf{BusterX}, a novel AI-generated video detection and explanation framework leveraging multimodal large language model (MLLM) and reinforcement learning for authenticity determination and explainable rationale. To our knowledge, GenBuster-200K is the {\it \textbf{first}} large-scale, high-quality AI-generated video dataset that incorporates the latest generative techniques for real-world scenarios. BusterX is the {\it \textbf{first}} framework to integrate MLLM with reinforcement learning for explainable AI-generated video detection. Extensive comparisons with state-of-the-art methods and ablation studies validate the effectiveness and generalizability of BusterX. The code, models, and datasets will be released.
>
---
#### [new 023] Hybrid 3D-4D Gaussian Splatting for Fast Dynamic Scene Representation
- **分类: cs.CV**

- **简介: 该论文提出混合3D-4D高斯溅射方法，用于动态3D场景高效重建任务。针对现有4D高斯方法在静态区域存在计算冗余和内存过高的问题，通过自适应分配3D高斯（静态）和4D高斯（动态），减少参数量并提升训练速度，同时保持动态区域的高保真运动建模。**

- **链接: [http://arxiv.org/pdf/2505.13215v1](http://arxiv.org/pdf/2505.13215v1)**

> **作者:** Seungjun Oh; Younggeun Lee; Hyejin Jeon; Eunbyung Park
>
> **备注:** https://ohsngjun.github.io/3D-4DGS/
>
> **摘要:** Recent advancements in dynamic 3D scene reconstruction have shown promising results, enabling high-fidelity 3D novel view synthesis with improved temporal consistency. Among these, 4D Gaussian Splatting (4DGS) has emerged as an appealing approach due to its ability to model high-fidelity spatial and temporal variations. However, existing methods suffer from substantial computational and memory overhead due to the redundant allocation of 4D Gaussians to static regions, which can also degrade image quality. In this work, we introduce hybrid 3D-4D Gaussian Splatting (3D-4DGS), a novel framework that adaptively represents static regions with 3D Gaussians while reserving 4D Gaussians for dynamic elements. Our method begins with a fully 4D Gaussian representation and iteratively converts temporally invariant Gaussians into 3D, significantly reducing the number of parameters and improving computational efficiency. Meanwhile, dynamic Gaussians retain their full 4D representation, capturing complex motions with high fidelity. Our approach achieves significantly faster training times compared to baseline 4D Gaussian Splatting methods while maintaining or improving the visual quality.
>
---
#### [new 024] Road Segmentation for ADAS/AD Applications
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于自动驾驶中的道路分割任务，旨在探究模型架构和数据集对分割性能的影响。研究通过改进VGG-16（使用Comma10k数据集）和U-Net（使用KITTI数据集）进行训练及跨数据集测试，发现VGG-16精度更高，并采用F1-score等指标分析架构与数据对结果的作用机制。**

- **链接: [http://arxiv.org/pdf/2505.12206v1](http://arxiv.org/pdf/2505.12206v1)**

> **作者:** Mathanesh Vellingiri Ramasamy; Dimas Rizky Kurniasalim
>
> **摘要:** Accurate road segmentation is essential for autonomous driving and ADAS, enabling effective navigation in complex environments. This study examines how model architecture and dataset choice affect segmentation by training a modified VGG-16 on the Comma10k dataset and a modified U-Net on the KITTI Road dataset. Both models achieved high accuracy, with cross-dataset testing showing VGG-16 outperforming U-Net despite U-Net being trained for more epochs. We analyze model performance using metrics such as F1-score, mean intersection over union, and precision, discussing how architecture and dataset impact results.
>
---
#### [new 025] LLaVA-4D: Embedding SpatioTemporal Prompt into LMMs for 4D Scene Understanding
- **分类: cs.CV**

- **简介: 该论文属于4D场景理解任务，旨在解决现有3D大模型无法捕捉动态对象的问题。提出LLaVA-4D框架，将3D空间坐标与时间编码为动态感知的4D时空提示嵌入视觉特征，增强模型对静态背景和动态物体的时空表征，并通过构建4D视觉语言数据集进行指令微调，提升物理世界理解能力。**

- **链接: [http://arxiv.org/pdf/2505.12253v1](http://arxiv.org/pdf/2505.12253v1)**

> **作者:** Hanyu Zhou; Gim Hee Lee
>
> **摘要:** Despite achieving significant progress in 2D image understanding, large multimodal models (LMMs) struggle in the physical world due to the lack of spatial representation. Typically, existing 3D LMMs mainly embed 3D positions as fixed spatial prompts within visual features to represent the scene. However, these methods are limited to understanding the static background and fail to capture temporally varying dynamic objects. In this paper, we propose LLaVA-4D, a general LMM framework with a novel spatiotemporal prompt for visual representation in 4D scene understanding. The spatiotemporal prompt is generated by encoding 3D position and 1D time into a dynamic-aware 4D coordinate embedding. Moreover, we demonstrate that spatial and temporal components disentangled from visual features are more effective in distinguishing the background from objects. This motivates embedding the 4D spatiotemporal prompt into these features to enhance the dynamic scene representation. By aligning visual spatiotemporal embeddings with language embeddings, LMMs gain the ability to understand both spatial and temporal characteristics of static background and dynamic objects in the physical world. Additionally, we construct a 4D vision-language dataset with spatiotemporal coordinate annotations for instruction fine-tuning LMMs. Extensive experiments have been conducted to demonstrate the effectiveness of our method across different tasks in 4D scene understanding.
>
---
#### [new 026] MAGI-1: Autoregressive Video Generation at Scale
- **分类: cs.CV; cs.AI**

- **简介: MAGI-1是自回归视频生成模型，针对文本条件图像转视频（I2V）任务，解决长视频生成的时间一致性与扩展性问题。通过分块预测连续帧、单调时序噪声去噪实现流式生成，结合算法优化与专用架构，支持实时可控生成并保持恒定推理成本。模型参数达240亿，验证了方法的扩展性。**

- **链接: [http://arxiv.org/pdf/2505.13211v1](http://arxiv.org/pdf/2505.13211v1)**

> **作者:** Sand. ai; Hansi Teng; Hongyu Jia; Lei Sun; Lingzhi Li; Maolin Li; Mingqiu Tang; Shuai Han; Tianning Zhang; W. Q. Zhang; Weifeng Luo; Xiaoyang Kang; Yuchen Sun; Yue Cao; Yunpeng Huang; Yutong Lin; Yuxin Fang; Zewei Tao; Zheng Zhang; Zhongshu Wang; Zixun Liu; Dai Shi; Guoli Su; Hanwen Sun; Hong Pan; Jie Wang; Jiexin Sheng; Min Cui; Min Hu; Ming Yan; Shucheng Yin; Siran Zhang; Tingting Liu; Xianping Yin; Xiaoyu Yang; Xin Song; Xuan Hu; Yankai Zhang; Yuqiao Li
>
> **摘要:** We present MAGI-1, a world model that generates videos by autoregressively predicting a sequence of video chunks, defined as fixed-length segments of consecutive frames. Trained to denoise per-chunk noise that increases monotonically over time, MAGI-1 enables causal temporal modeling and naturally supports streaming generation. It achieves strong performance on image-to-video (I2V) tasks conditioned on text instructions, providing high temporal consistency and scalability, which are made possible by several algorithmic innovations and a dedicated infrastructure stack. MAGI-1 facilitates controllable generation via chunk-wise prompting and supports real-time, memory-efficient deployment by maintaining constant peak inference cost, regardless of video length. The largest variant of MAGI-1 comprises 24 billion parameters and supports context lengths of up to 4 million tokens, demonstrating the scalability and robustness of our approach. The code and models are available at https://github.com/SandAI-org/MAGI-1 and https://github.com/SandAI-org/MagiAttention. The product can be accessed at https://sand.ai.
>
---
#### [new 027] VesselGPT: Autoregressive Modeling of Vascular Geometry
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于血管生成任务，旨在解决解剖树复杂几何的高保真建模问题。提出自回归方法：先用VQ-VAE编码血管结构为离散词汇，再用GPT-2建模生成过程，结合B样条保留形态细节，实现高精度血管合成。**

- **链接: [http://arxiv.org/pdf/2505.13318v1](http://arxiv.org/pdf/2505.13318v1)**

> **作者:** Paula Feldman; Martin Sinnona; Viviana Siless; Claudio Delrieux; Emmanuel Iarussi
>
> **摘要:** Anatomical trees are critical for clinical diagnosis and treatment planning, yet their complex and diverse geometry make accurate representation a significant challenge. Motivated by the latest advances in large language models, we introduce an autoregressive method for synthesizing anatomical trees. Our approach first embeds vessel structures into a learned discrete vocabulary using a VQ-VAE architecture, then models their generation autoregressively with a GPT-2 model. This method effectively captures intricate geometries and branching patterns, enabling realistic vascular tree synthesis. Comprehensive qualitative and quantitative evaluations reveal that our technique achieves high-fidelity tree reconstruction with compact discrete representations. Moreover, our B-spline representation of vessel cross-sections preserves critical morphological details that are often overlooked in previous' methods parameterizations. To the best of our knowledge, this work is the first to generate blood vessels in an autoregressive manner. Code, data, and trained models will be made available.
>
---
#### [new 028] SpatialCrafter: Unleashing the Imagination of Video Diffusion Models for Scene Reconstruction from Limited Observations
- **分类: cs.CV**

- **简介: 该论文属于稀疏/单视图3D场景重建任务，解决现有方法依赖密集多视角输入的问题。提出SpatialCrafter框架，利用视频扩散模型生成补充视角，通过可训练相机编码器、极线注意力机制和统一尺度策略保障几何一致性，结合深度先验与语义特征直接回归3D高斯基元，提升稀疏场景重建的真实感。**

- **链接: [http://arxiv.org/pdf/2505.11992v1](http://arxiv.org/pdf/2505.11992v1)**

> **作者:** Songchun Zhang; Huiyao Xu; Sitong Guo; Zhongwei Xie; Pengwei Liu; Hujun Bao; Weiwei Xu; Changqing Zou
>
> **备注:** 18 pages, 16 figures
>
> **摘要:** Novel view synthesis (NVS) boosts immersive experiences in computer vision and graphics. Existing techniques, though progressed, rely on dense multi-view observations, restricting their application. This work takes on the challenge of reconstructing photorealistic 3D scenes from sparse or single-view inputs. We introduce SpatialCrafter, a framework that leverages the rich knowledge in video diffusion models to generate plausible additional observations, thereby alleviating reconstruction ambiguity. Through a trainable camera encoder and an epipolar attention mechanism for explicit geometric constraints, we achieve precise camera control and 3D consistency, further reinforced by a unified scale estimation strategy to handle scale discrepancies across datasets. Furthermore, by integrating monocular depth priors with semantic features in the video latent space, our framework directly regresses 3D Gaussian primitives and efficiently processes long-sequence features using a hybrid network structure. Extensive experiments show our method enhances sparse view reconstruction and restores the realistic appearance of 3D scenes.
>
---
#### [new 029] BandRC: Band Shifted Raised Cosine Activated Implicit Neural Representations
- **分类: cs.CV**

- **简介: 该论文提出新型激活函数BandRC，用于增强隐式神经表示（INRs）的信号建模能力，解决传统激活函数存在的频谱偏差、噪声敏感及局部/全局特征融合困难等问题。通过数学分析与图像重建、去噪、超分辨率等实验验证其性能，在多项任务中超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11640v1](http://arxiv.org/pdf/2505.11640v1)**

> **作者:** Pandula Thennakoon; Avishka Ranasinghe; Mario De Silva; Buwaneka Epakanda; Roshan Godaliyadda; Parakrama Ekanayake; Vijitha Herath
>
> **备注:** Submitted as a conference paper to ICCV 2025
>
> **摘要:** In recent years, implicit neural representations(INRs) have gained popularity in the computer vision community. This is mainly due to the strong performance of INRs in many computer vision tasks. These networks can extract a continuous signal representation given a discrete signal representation. In previous studies, it has been repeatedly shown that INR performance has a strong correlation with the activation functions used in its multilayer perceptrons. Although numerous activation functions have been proposed that are competitive with one another, they share some common set of challenges such as spectral bias(Lack of sensitivity to high-frequency content in signals), limited robustness to signal noise and difficulties in simultaneous capturing both local and global features. and furthermore, the requirement for manual parameter tuning. To address these issues, we introduce a novel activation function, Band Shifted Raised Cosine Activated Implicit Neural Networks \textbf{(BandRC)} tailored to enhance signal representation capacity further. We also incorporate deep prior knowledge extracted from the signal to adjust the activation functions through a task-specific model. Through a mathematical analysis and a series of experiments which include image reconstruction (with a +8.93 dB PSNR improvement over the nearest counterpart), denoising (with a +0.46 dB increase in PSNR), super-resolution (with a +1.03 dB improvement over the nearest State-Of-The-Art (SOTA) method for 6X super-resolution), inpainting, and 3D shape reconstruction we demonstrate the dominance of BandRC over existing state of the art activation functions.
>
---
#### [new 030] RVTBench: A Benchmark for Visual Reasoning Tasks
- **分类: cs.CV**

- **简介: 该论文针对视觉推理任务缺乏高质量基准的问题，提出了RVTBench多模态评测集，覆盖分割、问答等4类任务，通过数字孪生技术构建包含120万标记的复杂时空推理数据，并提出零样本通用框架RVTagent，突破传统LLM生成方法在推理深度和任务泛化上的局限。**

- **链接: [http://arxiv.org/pdf/2505.11838v1](http://arxiv.org/pdf/2505.11838v1)**

> **作者:** Yiqing Shen; Chenjia Li; Chenxiao Fan; Mathias Unberath
>
> **摘要:** Visual reasoning, the capability to interpret visual input in response to implicit text query through multi-step reasoning, remains a challenge for deep learning models due to the lack of relevant benchmarks. Previous work in visual reasoning has primarily focused on reasoning segmentation, where models aim to segment objects based on implicit text queries. This paper introduces reasoning visual tasks (RVTs), a unified formulation that extends beyond traditional video reasoning segmentation to a diverse family of visual language reasoning problems, which can therefore accommodate multiple output formats including bounding boxes, natural language descriptions, and question-answer pairs. Correspondingly, we identify the limitations in current benchmark construction methods that rely solely on large language models (LLMs), which inadequately capture complex spatial-temporal relationships and multi-step reasoning chains in video due to their reliance on token representation, resulting in benchmarks with artificially limited reasoning complexity. To address this limitation, we propose a novel automated RVT benchmark construction pipeline that leverages digital twin (DT) representations as structured intermediaries between perception and the generation of implicit text queries. Based on this method, we construct RVTBench, a RVT benchmark containing 3,896 queries of over 1.2 million tokens across four types of RVT (segmentation, grounding, VQA and summary), three reasoning categories (semantic, spatial, and temporal), and four increasing difficulty levels, derived from 200 video sequences. Finally, we propose RVTagent, an agent framework for RVT that allows for zero-shot generalization across various types of RVT without task-specific fine-tuning.
>
---
#### [new 031] Continuous Subspace Optimization for Continual Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究持续学习任务，旨在解决灾难性遗忘问题。针对现有低秩适应方法固定子空间限制模型能力的问题，提出动态连续子空间优化（CoSO），通过梯度奇异值分解生成正交子空间序列，在高效更新参数的同时隔离任务间干扰，显著提升长任务序列下的性能。**

- **链接: [http://arxiv.org/pdf/2505.11816v1](http://arxiv.org/pdf/2505.11816v1)**

> **作者:** Quan Cheng; Yuanyu Wan; Lingyu Wu; Chenping Hou; Lijun Zhang
>
> **摘要:** Continual learning aims to learn multiple tasks sequentially while preserving prior knowledge, but faces the challenge of catastrophic forgetting when acquiring new knowledge. Recently, approaches leveraging pre-trained models have gained increasing popularity to mitigate this issue, due to the strong generalization ability of foundation models. To adjust pre-trained models for new tasks, existing methods usually employ low-rank adaptation, which restricts parameter updates to a fixed low-rank subspace. However, constraining the optimization space inherently compromises the model's learning capacity, resulting in inferior performance. To address the limitation, we propose Continuous Subspace Optimization for Continual Learning (CoSO) to fine-tune the model in a series of subspaces rather than a single one. These sequential subspaces are dynamically determined through the singular value decomposition of gradients. CoSO updates the model by projecting gradients into these subspaces, ensuring memory-efficient optimization. To mitigate forgetting, the optimization subspaces of each task are set to be orthogonal to the historical task subspace. During task learning, CoSO maintains a task-specific component that captures the critical update directions associated with the current task. Upon completing a task, this component is used to update the historical task subspace, laying the groundwork for subsequent learning. Extensive experiments on multiple datasets demonstrate that CoSO significantly outperforms state-of-the-art methods, especially in challenging scenarios with long task sequences.
>
---
#### [new 032] CoT-Vid: Dynamic Chain-of-Thought Routing with Self Verification for Training-Free Video Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CoT-Vid，针对视频推理任务，解决现有方法过度依赖感知能力、缺乏显式推理机制的问题。通过动态思维链路由、问题解耦和自验证三阶段设计，构建无需训练的推理框架，并建立视频问题分类标准。在多个基准测试中超越基础模型及部分大模型，验证了显式推理机制的有效性。**

- **链接: [http://arxiv.org/pdf/2505.11830v1](http://arxiv.org/pdf/2505.11830v1)**

> **作者:** Hongbo Jin; Ruyang Liu; Wenhao Zhang; Guibo Luo; Ge Li
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** System2 reasoning is developing rapidly these days with the emergence of Deep- Thinking Models and chain-of-thought technology, which has become a centralized discussion point in the AI community. However, there is a relative gap in the research on complex video reasoning at present. In this work, we propose CoT-Vid, a novel training-free paradigm for the video domain with a multistage complex reasoning design. Distinguishing from existing video LLMs, which rely heavily on perceptual abilities, it achieved surprising performance gain with explicit reasoning mechanism. The paradigm consists of three main components: dynamic inference path routing, problem decoupling strategy, and video self-consistency verification. In addition, we propose a new standard for categorization of video questions. CoT- Vid showed outstanding results on a wide range of benchmarks, and outperforms its base model by 9.3% on Egochema and 5.6% on VideoEspresso, rivalling or even surpassing larger and proprietary models, such as GPT-4V, GPT-4o and Gemini-1.5-flash. Our codebase will be publicly available soon.
>
---
#### [new 033] DD-Ranking: Rethinking the Evaluation of Dataset Distillation
- **分类: cs.CV**

- **简介: 该论文针对数据集蒸馏评估问题，指出传统准确率指标因依赖额外技术（如数据增强）而失准，无法真实反映合成数据质量。作者提出DD-Ranking框架及新指标，聚焦数据信息增强程度，为不同方法建立更公平统一的评估标准，推动领域发展。属于机器学习数据压缩任务。**

- **链接: [http://arxiv.org/pdf/2505.13300v1](http://arxiv.org/pdf/2505.13300v1)**

> **作者:** Zekai Li; Xinhao Zhong; Samir Khaki; Zhiyuan Liang; Yuhao Zhou; Mingjia Shi; Ziqiao Wang; Xuanlei Zhao; Wangbo Zhao; Ziheng Qin; Mengxuan Wu; Pengfei Zhou; Haonan Wang; David Junhao Zhang; Jia-Wei Liu; Shaobo Wang; Dai Liu; Linfeng Zhang; Guang Li; Kun Wang; Zheng Zhu; Zhiheng Ma; Joey Tianyi Zhou; Jiancheng Lv; Yaochu Jin; Peihao Wang; Kaipeng Zhang; Lingjuan Lyu; Yiran Huang; Zeynep Akata; Zhiwei Deng; Xindi Wu; George Cazenavette; Yuzhang Shang; Justin Cui; Jindong Gu; Qian Zheng; Hao Ye; Shuo Wang; Xiaobo Wang; Yan Yan; Angela Yao; Mike Zheng Shou; Tianlong Chen; Hakan Bilen; Baharan Mirzasoleiman; Manolis Kellis; Konstantinos N. Plataniotis; Zhangyang Wang; Bo Zhao; Yang You; Kai Wang
>
> **备注:** 20 pages, 4 figures
>
> **摘要:** In recent years, dataset distillation has provided a reliable solution for data compression, where models trained on the resulting smaller synthetic datasets achieve performance comparable to those trained on the original datasets. To further improve the performance of synthetic datasets, various training pipelines and optimization objectives have been proposed, greatly advancing the field of dataset distillation. Recent decoupled dataset distillation methods introduce soft labels and stronger data augmentation during the post-evaluation phase and scale dataset distillation up to larger datasets (e.g., ImageNet-1K). However, this raises a question: Is accuracy still a reliable metric to fairly evaluate dataset distillation methods? Our empirical findings suggest that the performance improvements of these methods often stem from additional techniques rather than the inherent quality of the images themselves, with even randomly sampled images achieving superior results. Such misaligned evaluation settings severely hinder the development of DD. Therefore, we propose DD-Ranking, a unified evaluation framework, along with new general evaluation metrics to uncover the true performance improvements achieved by different methods. By refocusing on the actual information enhancement of distilled datasets, DD-Ranking provides a more comprehensive and fair evaluation standard for future research advancements.
>
---
#### [new 034] VisionReasoner: Unified Visual Perception and Reasoning via Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文提出VisionReasoner，一个通过强化学习统一视觉感知与推理的框架，解决多任务视觉感知模型碎片化问题。其通过多目标认知学习策略和任务重构方法，在检测、分割、计数等10个跨领域任务中实现单一模型处理，推理性能超越Qwen2.5VL（检测+29.1%、分割+22.1%、计数+15.3%）。**

- **链接: [http://arxiv.org/pdf/2505.12081v1](http://arxiv.org/pdf/2505.12081v1)**

> **作者:** Yuqi Liu; Tianyuan Qu; Zhisheng Zhong; Bohao Peng; Shu Liu; Bei Yu; Jiaya Jia
>
> **摘要:** Large vision-language models exhibit inherent capabilities to handle diverse visual perception tasks. In this paper, we introduce VisionReasoner, a unified framework capable of reasoning and solving multiple visual perception tasks within a shared model. Specifically, by designing novel multi-object cognitive learning strategies and systematic task reformulation, VisionReasoner enhances its reasoning capabilities to analyze visual inputs, and addresses diverse perception tasks in a unified framework. The model generates a structured reasoning process before delivering the desired outputs responding to user queries. To rigorously assess unified visual perception capabilities, we evaluate VisionReasoner on ten diverse tasks spanning three critical domains: detection, segmentation, and counting. Experimental results show that VisionReasoner achieves superior performance as a unified model, outperforming Qwen2.5VL by relative margins of 29.1% on COCO (detection), 22.1% on ReasonSeg (segmentation), and 15.3% on CountBench (counting).
>
---
#### [new 035] LoFT: LoRA-fused Training Dataset Generation with Few-shot Guidance
- **分类: cs.CV**

- **简介: 该论文提出LoFT框架，属于合成数据生成任务，旨在解决现有方法生成数据保真度和多样性不足的问题。通过微调单个真实图像的LoRA权重并融合特征，生成高质量合成数据集。实验证明其数据能提升下游模型性能，尤其在大规模数据下准确率显著提高。**

- **链接: [http://arxiv.org/pdf/2505.11703v1](http://arxiv.org/pdf/2505.11703v1)**

> **作者:** Jae Myung Kim; Stephan Alaniz; Cordelia Schmid; Zeynep Akata
>
> **摘要:** Despite recent advances in text-to-image generation, using synthetically generated data seldom brings a significant boost in performance for supervised learning. Oftentimes, synthetic datasets do not faithfully recreate the data distribution of real data, i.e., they lack the fidelity or diversity needed for effective downstream model training. While previous work has employed few-shot guidance to address this issue, existing methods still fail to capture and generate features unique to specific real images. In this paper, we introduce a novel dataset generation framework named LoFT, LoRA-Fused Training-data Generation with Few-shot Guidance. Our method fine-tunes LoRA weights on individual real images and fuses them at inference time, producing synthetic images that combine the features of real images for improved diversity and fidelity of generated data. We evaluate the synthetic data produced by LoFT on 10 datasets, using 8 to 64 real images per class as guidance and scaling up to 1000 images per class. Our experiments show that training on LoFT-generated data consistently outperforms other synthetic dataset methods, significantly increasing accuracy as the dataset size increases. Additionally, our analysis demonstrates that LoFT generates datasets with high fidelity and sufficient diversity, which contribute to the performance improvement. The code is available at https://github.com/ExplainableML/LoFT.
>
---
#### [new 036] Revisiting Residual Connections: Orthogonal Updates for Stable and Efficient Deep Networks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于深度学习网络结构优化任务，旨在解决传统残差连接因沿输入方向更新而限制新特征学习的问题。提出正交残差更新方法，将模块输出分解为与输入流正交的分量进行叠加，促进网络学习新特征方向，提升训练效率和稳定性。实验证明该方法在多种模型和数据集上显著提升精度（如ViT-B在ImageNet提升4.3%）。**

- **链接: [http://arxiv.org/pdf/2505.11881v1](http://arxiv.org/pdf/2505.11881v1)**

> **作者:** Giyeong Oh; Woohyun Cho; Siyeol Kim; Suhwan Choi; Younjae Yu
>
> **备注:** 27 pages, WIP
>
> **摘要:** Residual connections are pivotal for deep neural networks, enabling greater depth by mitigating vanishing gradients. However, in standard residual updates, the module's output is directly added to the input stream. This can lead to updates that predominantly reinforce or modulate the existing stream direction, potentially underutilizing the module's capacity for learning entirely novel features. In this work, we introduce Orthogonal Residual Update: we decompose the module's output relative to the input stream and add only the component orthogonal to this stream. This design aims to guide modules to contribute primarily new representational directions, fostering richer feature learning while promoting more efficient training. We demonstrate that our orthogonal update strategy improves generalization accuracy and training stability across diverse architectures (ResNetV2, Vision Transformers) and datasets (CIFARs, TinyImageNet, ImageNet-1k), achieving, for instance, a +4.3\%p top-1 accuracy gain for ViT-B on ImageNet-1k.
>
---
#### [new 037] Learning to Highlight Audio by Watching Movies
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出视听引导的音频高亮任务，解决视频制作中音频与视觉显著性不匹配的问题。通过基于Transformer的多模态框架，利用电影数据生成模拟低质音频的混合数据集，训练模型以视频指导优化音频，提升视听和谐度。方法在定量和主观评估中优于基线。**

- **链接: [http://arxiv.org/pdf/2505.12154v1](http://arxiv.org/pdf/2505.12154v1)**

> **作者:** Chao Huang; Ruohan Gao; J. M. F. Tsang; Jan Kurcius; Cagdas Bilen; Chenliang Xu; Anurag Kumar; Sanjeel Parekh
>
> **备注:** CVPR 2025. Project page: https://wikichao.github.io/VisAH/
>
> **摘要:** Recent years have seen a significant increase in video content creation and consumption. Crafting engaging content requires the careful curation of both visual and audio elements. While visual cue curation, through techniques like optimal viewpoint selection or post-editing, has been central to media production, its natural counterpart, audio, has not undergone equivalent advancements. This often results in a disconnect between visual and acoustic saliency. To bridge this gap, we introduce a novel task: visually-guided acoustic highlighting, which aims to transform audio to deliver appropriate highlighting effects guided by the accompanying video, ultimately creating a more harmonious audio-visual experience. We propose a flexible, transformer-based multimodal framework to solve this task. To train our model, we also introduce a new dataset -- the muddy mix dataset, leveraging the meticulous audio and video crafting found in movies, which provides a form of free supervision. We develop a pseudo-data generation process to simulate poorly mixed audio, mimicking real-world scenarios through a three-step process -- separation, adjustment, and remixing. Our approach consistently outperforms several baselines in both quantitative and subjective evaluation. We also systematically study the impact of different types of contextual guidance and difficulty levels of the dataset. Our project page is here: https://wikichao.github.io/VisAH/.
>
---
#### [new 038] iSegMan: Interactive Segment-and-Manipulate 3D Gaussians
- **分类: cs.CV**

- **简介: 该论文提出iSegMan框架，解决3D高斯场景交互式分割与操作中区域控制难、反馈延迟及依赖预训练的问题。通过极线约束传播交互（EIP）和基于可见性的投票机制（VGV）实现无需场景训练的实时分割，结合操作工具箱增强可控性，提升3D编辑效率与精度。属于3D场景交互编辑任务。**

- **链接: [http://arxiv.org/pdf/2505.11934v1](http://arxiv.org/pdf/2505.11934v1)**

> **作者:** Yian Zhao; Wanshi Xu; Ruochong Zheng; Pengchong Qiao; Chang Liu; Jie Chen
>
> **备注:** CVPR 2025
>
> **摘要:** The efficient rendering and explicit nature of 3DGS promote the advancement of 3D scene manipulation. However, existing methods typically encounter challenges in controlling the manipulation region and are unable to furnish the user with interactive feedback, which inevitably leads to unexpected results. Intuitively, incorporating interactive 3D segmentation tools can compensate for this deficiency. Nevertheless, existing segmentation frameworks impose a pre-processing step of scene-specific parameter training, which limits the efficiency and flexibility of scene manipulation. To deliver a 3D region control module that is well-suited for scene manipulation with reliable efficiency, we propose interactive Segment-and-Manipulate 3D Gaussians (iSegMan), an interactive segmentation and manipulation framework that only requires simple 2D user interactions in any view. To propagate user interactions to other views, we propose Epipolar-guided Interaction Propagation (EIP), which innovatively exploits epipolar constraint for efficient and robust interaction matching. To avoid scene-specific training to maintain efficiency, we further propose the novel Visibility-based Gaussian Voting (VGV), which obtains 2D segmentations from SAM and models the region extraction as a voting game between 2D Pixels and 3D Gaussians based on Gaussian visibility. Taking advantage of the efficient and precise region control of EIP and VGV, we put forth a Manipulation Toolbox to implement various functions on selected regions, enhancing the controllability, flexibility and practicality of scene manipulation. Extensive results on 3D scene manipulation and segmentation tasks fully demonstrate the significant advantages of iSegMan. Project page is available at https://zhao-yian.github.io/iSegMan.
>
---
#### [new 039] Keypoints as Dynamic Centroids for Unified Human Pose and Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉中的联合人体姿态估计和实例分割任务，旨在解决动态场景下关节重叠及快速姿态变化导致的检测困难。提出KDC方法，通过动态质心表示将关键点检测与像素聚类结合，利用高置信度关键点生成嵌入空间的MaskCentroids，提升复杂场景的检测精度与实时性能。实验验证了其在多数据集上的有效性。**

- **链接: [http://arxiv.org/pdf/2505.12130v1](http://arxiv.org/pdf/2505.12130v1)**

> **作者:** Niaz Ahmad; Jawad Khan; Kang G. Shin; Youngmoon Lee; Guanghui Wang
>
> **摘要:** The dynamic movement of the human body presents a fundamental challenge for human pose estimation and body segmentation. State-of-the-art approaches primarily rely on combining keypoint heatmaps with segmentation masks but often struggle in scenarios involving overlapping joints or rapidly changing poses during instance-level segmentation. To address these limitations, we propose Keypoints as Dynamic Centroid (KDC), a new centroid-based representation for unified human pose estimation and instance-level segmentation. KDC adopts a bottom-up paradigm to generate keypoint heatmaps for both easily distinguishable and complex keypoints and improves keypoint detection and confidence scores by introducing KeyCentroids using a keypoint disk. It leverages high-confidence keypoints as dynamic centroids in the embedding space to generate MaskCentroids, allowing for swift clustering of pixels to specific human instances during rapid body movements in live environments. Our experimental evaluations on the CrowdPose, OCHuman, and COCO benchmarks demonstrate KDC's effectiveness and generalizability in challenging scenarios in terms of both accuracy and runtime performance. The implementation is available at: https://sites.google.com/view/niazahmad/projects/kdc.
>
---
#### [new 040] From Local Details to Global Context: Advancing Vision-Language Models with Attention-Based Selection
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型（VLM）零样本任务中随机增强导致的局部过拟合和全局语义缺失问题，提出注意力选择方法（ABS）。通过注意力引导的图像/特征裁剪和全局特征融合，结合软匹配优化文本对齐，无需训练即在零样本分类和分布外泛化任务达到SOTA，性能媲美少样本学习。**

- **链接: [http://arxiv.org/pdf/2505.13233v1](http://arxiv.org/pdf/2505.13233v1)**

> **作者:** Lincan Cai; Jingxuan Kang; Shuang Li; Wenxuan Ma; Binhui Xie; Zhida Qin; Jian Liang
>
> **摘要:** Pretrained vision-language models (VLMs), e.g., CLIP, demonstrate impressive zero-shot capabilities on downstream tasks. Prior research highlights the crucial role of visual augmentation techniques, like random cropping, in alignment with fine-grained class descriptions generated by large language models (LLMs), significantly enhancing zero-shot performance by incorporating multi-view information. However, the inherent randomness of these augmentations can inevitably introduce background artifacts and cause models to overly focus on local details, compromising global semantic understanding. To address these issues, we propose an \textbf{A}ttention-\textbf{B}ased \textbf{S}election (\textbf{ABS}) method from local details to global context, which applies attention-guided cropping in both raw images and feature space, supplement global semantic information through strategic feature selection. Additionally, we introduce a soft matching technique to effectively filter LLM descriptions for better alignment. \textbf{ABS} achieves state-of-the-art performance on out-of-distribution generalization and zero-shot classification tasks. Notably, \textbf{ABS} is training-free and even rivals few-shot and test-time adaptation methods. Our code is available at \href{https://github.com/BIT-DA/ABS}{\textcolor{darkgreen}{https://github.com/BIT-DA/ABS}}.
>
---
#### [new 041] Beluga Whale Detection from Satellite Imagery with Point Labels
- **分类: cs.CV**

- **简介: 该论文研究基于卫星图像的海洋动物检测任务，旨在解决传统方法依赖人工标注边界框及忽略不确定目标的问题。提出结合点标注与SAM模型自动生成标注，训练YOLOv8实现多类检测（确定/不确定鲸鱼、海豹），提升检测性能并降低标注成本，适用于生态监测。**

- **链接: [http://arxiv.org/pdf/2505.12066v1](http://arxiv.org/pdf/2505.12066v1)**

> **作者:** Yijie Zheng; Jinxuan Yang; Yu Chen; Yaxuan Wang; Yihang Lu; Guoqing Li
>
> **备注:** Accepted for oral presentation at IGARSS 2025. Session at https://www.2025.ieeeigarss.org/view_paper.php?PaperNum=2430&SessionID=1426
>
> **摘要:** Very high-resolution (VHR) satellite imagery has emerged as a powerful tool for monitoring marine animals on a large scale. However, existing deep learning-based whale detection methods usually require manually created, high-quality bounding box annotations, which are labor-intensive to produce. Moreover, existing studies often exclude ``uncertain whales'', individuals that have ambiguous appearances in satellite imagery, limiting the applicability of these models in real-world scenarios. To address these limitations, this study introduces an automated pipeline for detecting beluga whales and harp seals in VHR satellite imagery. The pipeline leverages point annotations and the Segment Anything Model (SAM) to generate precise bounding box annotations, which are used to train YOLOv8 for multiclass detection of certain whales, uncertain whales, and harp seals. Experimental results demonstrated that SAM-generated annotations significantly improved detection performance, achieving higher $\text{F}_\text{1}$-scores compared to traditional buffer-based annotations. YOLOv8 trained on SAM-labeled boxes achieved an overall $\text{F}_\text{1}$-score of 72.2% for whales overall and 70.3% for harp seals, with superior performance in dense scenes. The proposed approach not only reduces the manual effort required for annotation but also enhances the detection of uncertain whales, offering a more comprehensive solution for marine animal monitoring. This method holds great potential for extending to other species, habitats, and remote sensing platforms, as well as for estimating whale biometrics, thereby advancing ecological monitoring and conservation efforts. The codes for our label and detection pipeline are publicly available at http://github.com/voyagerxvoyagerx/beluga-seeker .
>
---
#### [new 042] 3D Visual Illusion Depth Estimation
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉深度估计任务，研究3D视觉错觉对机器视觉系统的误导问题。针对现有单目/双目深度估计方法易受错觉干扰的缺陷，作者构建大规模数据集评估现有模型，并提出融合视觉语言模型常识的鲁棒框架，自适应筛选可靠深度信息，提升抗干扰能力。**

- **链接: [http://arxiv.org/pdf/2505.13061v1](http://arxiv.org/pdf/2505.13061v1)**

> **作者:** CHengtang Yao; Zhidan Liu; Jiaxi Zeng; Lidong Yu; Yuwei Wu; Yunde Jia
>
> **备注:** Project: https://github.com/YaoChengTang/3D-Visual-Illusion-Depth-Estimation
>
> **摘要:** 3D visual illusion is a perceptual phenomenon where a two-dimensional plane is manipulated to simulate three-dimensional spatial relationships, making a flat artwork or object look three-dimensional in the human visual system. In this paper, we reveal that the machine visual system is also seriously fooled by 3D visual illusions, including monocular and binocular depth estimation. In order to explore and analyze the impact of 3D visual illusion on depth estimation, we collect a large dataset containing almost 3k scenes and 200k images to train and evaluate SOTA monocular and binocular depth estimation methods. We also propose a robust depth estimation framework that uses common sense from a vision-language model to adaptively select reliable depth from binocular disparity and monocular depth. Experiments show that SOTA monocular, binocular, and multi-view depth estimation approaches are all fooled by various 3D visual illusions, while our method achieves SOTA performance.
>
---
#### [new 043] GMM-Based Comprehensive Feature Extraction and Relative Distance Preservation For Few-Shot Cross-Modal Retrieval
- **分类: cs.CV; cs.IR**

- **简介: 该论文针对少样本跨模态检索任务，解决现有方法因数据多峰分布建模不足导致的模态内外偏差问题。提出GCRDP方法，利用高斯混合模型捕捉数据分布，结合对比学习增强特征建模，并通过约束跨模态特征相对距离优化语义对齐，在四个基准数据集上验证了性能优势。**

- **链接: [http://arxiv.org/pdf/2505.13306v1](http://arxiv.org/pdf/2505.13306v1)**

> **作者:** Chengsong Sun; Weiping Li; Xiang Li; Yuankun Liu; Lianlei Shan
>
> **摘要:** Few-shot cross-modal retrieval focuses on learning cross-modal representations with limited training samples, enabling the model to handle unseen classes during inference. Unlike traditional cross-modal retrieval tasks, which assume that both training and testing data share the same class distribution, few-shot retrieval involves data with sparse representations across modalities. Existing methods often fail to adequately model the multi-peak distribution of few-shot cross-modal data, resulting in two main biases in the latent semantic space: intra-modal bias, where sparse samples fail to capture intra-class diversity, and inter-modal bias, where misalignments between image and text distributions exacerbate the semantic gap. These biases hinder retrieval accuracy. To address these issues, we propose a novel method, GCRDP, for few-shot cross-modal retrieval. This approach effectively captures the complex multi-peak distribution of data using a Gaussian Mixture Model (GMM) and incorporates a multi-positive sample contrastive learning mechanism for comprehensive feature modeling. Additionally, we introduce a new strategy for cross-modal semantic alignment, which constrains the relative distances between image and text feature distributions, thereby improving the accuracy of cross-modal representations. We validate our approach through extensive experiments on four benchmark datasets, demonstrating superior performance over six state-of-the-art methods.
>
---
#### [new 044] Just Dance with $π$! A Poly-modal Inductor for Weakly-supervised Video Anomaly Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对弱监督视频异常检测任务，解决传统仅用RGB特征导致可靠性不足的问题。提出PI-VAD框架，通过整合姿态、深度、全景掩模、光流和语言五种模态增强RGB特征，并设计伪模态生成与跨模态诱导模块，训练时利用多模态信息生成原型，推理时仅需RGB。在真实场景数据集上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2505.13123v1](http://arxiv.org/pdf/2505.13123v1)**

> **作者:** Snehashis Majhi; Giacomo D'Amicantonio; Antitza Dantcheva; Quan Kong; Lorenzo Garattoni; Gianpiero Francesca; Egor Bondarev; Francois Bremond
>
> **摘要:** Weakly-supervised methods for video anomaly detection (VAD) are conventionally based merely on RGB spatio-temporal features, which continues to limit their reliability in real-world scenarios. This is due to the fact that RGB-features are not sufficiently distinctive in setting apart categories such as shoplifting from visually similar events. Therefore, towards robust complex real-world VAD, it is essential to augment RGB spatio-temporal features by additional modalities. Motivated by this, we introduce the Poly-modal Induced framework for VAD: "PI-VAD", a novel approach that augments RGB representations by five additional modalities. Specifically, the modalities include sensitivity to fine-grained motion (Pose), three dimensional scene and entity representation (Depth), surrounding objects (Panoptic masks), global motion (optical flow), as well as language cues (VLM). Each modality represents an axis of a polygon, streamlined to add salient cues to RGB. PI-VAD includes two plug-in modules, namely Pseudo-modality Generation module and Cross Modal Induction module, which generate modality-specific prototypical representation and, thereby, induce multi-modal information into RGB cues. These modules operate by performing anomaly-aware auxiliary tasks and necessitate five modality backbones -- only during training. Notably, PI-VAD achieves state-of-the-art accuracy on three prominent VAD datasets encompassing real-world scenarios, without requiring the computational overhead of five modality backbones at inference.
>
---
#### [new 045] Informed Mixing -- Improving Open Set Recognition via Attribution-based Augmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对开放集识别任务，旨在提升模型检测未知类别的能力。现有方法难以从未知数据中学习有效特征，提出GradMix数据增强方法：通过梯度归因图动态遮蔽已学概念，迫使模型从同源数据挖掘更全面的特征。实验表明该方法在开放/闭集分类、分布外检测等任务优于现有技术，并增强模型鲁棒性和自监督学习性能。**

- **链接: [http://arxiv.org/pdf/2505.12803v1](http://arxiv.org/pdf/2505.12803v1)**

> **作者:** Jiawen Xu; Odej Kao; Margret Keuper
>
> **摘要:** Open set recognition (OSR) is devised to address the problem of detecting novel classes during model inference. Even in recent vision models, this remains an open issue which is receiving increasing attention. Thereby, a crucial challenge is to learn features that are relevant for unseen categories from given data, for which these features might not be discriminative. To facilitate this process and "optimize to learn" more diverse features, we propose GradMix, a data augmentation method that dynamically leverages gradient-based attribution maps of the model during training to mask out already learned concepts. Thus GradMix encourages the model to learn a more complete set of representative features from the same data source. Extensive experiments on open set recognition, close set classification, and out-of-distribution detection reveal that our method can often outperform the state-of-the-art. GradMix can further increase model robustness to corruptions as well as downstream classification performance for self-supervised learning, indicating its benefit for model generalization.
>
---
#### [new 046] CLIP-aware Domain-Adaptive Super-Resolution
- **分类: cs.CV**

- **简介: 该论文提出CLIP感知域自适应超分辨率（CDASR），解决单图超分辨率中跨领域泛化难题。通过融合CLIP语义特征与元学习少样本适应策略，设计多阶段特征对齐模块及多损失函数，提升极端缩放下的重建效果。实验显示在×16缩放时PSNR提升达0.30dB，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.12391v1](http://arxiv.org/pdf/2505.12391v1)**

> **作者:** Zhengyang Lu; Qian Xia; Weifan Wang; Feng Wang
>
> **摘要:** This work introduces CLIP-aware Domain-Adaptive Super-Resolution (CDASR), a novel framework that addresses the critical challenge of domain generalization in single image super-resolution. By leveraging the semantic capabilities of CLIP (Contrastive Language-Image Pre-training), CDASR achieves unprecedented performance across diverse domains and extreme scaling factors. The proposed method integrates CLIP-guided feature alignment mechanism with a meta-learning inspired few-shot adaptation strategy, enabling efficient knowledge transfer and rapid adaptation to target domains. A custom domain-adaptive module processes CLIP features alongside super-resolution features through a multi-stage transformation process, including CLIP feature processing, spatial feature generation, and feature fusion. This intricate process ensures effective incorporation of semantic information into the super-resolution pipeline. Additionally, CDASR employs a multi-component loss function that combines pixel-wise reconstruction, perceptual similarity, and semantic consistency. Extensive experiments on benchmark datasets demonstrate CDASR's superiority, particularly in challenging scenarios. On the Urban100 dataset at $\times$8 scaling, CDASR achieves a significant PSNR gain of 0.15dB over existing methods, with even larger improvements of up to 0.30dB observed at $\times$16 scaling.
>
---
#### [new 047] A Skull-Adaptive Framework for AI-Based 3D Transcranial Focused Ultrasound Simulation
- **分类: cs.CV**

- **简介: 该论文提出基于深度学习的颅骨自适应3D超声模拟框架，解决经颅聚焦超声因颅骨异质性导致波前畸变的问题。通过构建大规模仿真数据集TFUScapes，并开发融合换能器位置编码的U-Net模型DeepTFUS，实现从CT影像直接预测高精度声场分布，替代传统耗时数值计算，属于医学影像与计算声学的交叉任务。**

- **链接: [http://arxiv.org/pdf/2505.12998v1](http://arxiv.org/pdf/2505.12998v1)**

> **作者:** Vinkle Srivastav; Juliette Puel; Jonathan Vappou; Elijah Van Houten; Paolo Cabras; Nicolas Padoy
>
> **备注:** The project page is available at https://github.com/CAMMA-public/TFUScapes
>
> **摘要:** Transcranial focused ultrasound (tFUS) is an emerging modality for non-invasive brain stimulation and therapeutic intervention, offering millimeter-scale spatial precision and the ability to target deep brain structures. However, the heterogeneous and anisotropic nature of the human skull introduces significant distortions to the propagating ultrasound wavefront, which require time-consuming patient-specific planning and corrections using numerical solvers for accurate targeting. To enable data-driven approaches in this domain, we introduce TFUScapes, the first large-scale, high-resolution dataset of tFUS simulations through anatomically realistic human skulls derived from T1-weighted MRI images. We have developed a scalable simulation engine pipeline using the k-Wave pseudo-spectral solver, where each simulation returns a steady-state pressure field generated by a focused ultrasound transducer placed at realistic scalp locations. In addition to the dataset, we present DeepTFUS, a deep learning model that estimates normalized pressure fields directly from input 3D CT volumes and transducer position. The model extends a U-Net backbone with transducer-aware conditioning, incorporating Fourier-encoded position embeddings and MLP layers to create global transducer embeddings. These embeddings are fused with U-Net encoder features via feature-wise modulation, dynamic convolutions, and cross-attention mechanisms. The model is trained using a combination of spatially weighted and gradient-sensitive loss functions, enabling it to approximate high-fidelity wavefields. The TFUScapes dataset is publicly released to accelerate research at the intersection of computational acoustics, neurotechnology, and deep learning. The project page is available at https://github.com/CAMMA-public/TFUScapes.
>
---
#### [new 048] VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning
- **分类: cs.CV**

- **简介: 该论文属于视频推理任务，旨在解决多模态大语言模型（MLLMs）在复杂视频逻辑推理中的能力不足问题。提出VideoRFT方法，通过两阶段强化微调：先用自动生成的链式思维数据集进行监督训练，再结合语义一致性奖励强化学习，提升模型基于视觉证据的推理能力，构建了102K/310K规模的数据集并验证了SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.12434v1](http://arxiv.org/pdf/2505.12434v1)**

> **作者:** Qi Wang; Yanrui Yu; Ye Yuan; Rui Mao; Tianfei Zhou
>
> **备注:** Code: https://github.com/QiWang98/VideoRFT
>
> **摘要:** Reinforcement fine-tuning (RFT) has shown great promise in achieving humanlevel reasoning capabilities of Large Language Models (LLMs), and has recently been extended to MLLMs. Nevertheless, reasoning about videos, which is a fundamental aspect of human intelligence, remains a persistent challenge due to the complex logic, temporal and causal structures inherent in video data. To fill this gap, we propose VIDEORFT, a novel approach that extends the RFT paradigm to cultivate human-like video reasoning capabilities in MLLMs. VIDEORFT follows the standard two-stage scheme in RFT: supervised fine-tuning (SFT) with chain-of-thought (CoT) annotations, followed by reinforcement learning (RL) to improve generalization. A central challenge to achieve this in the video domain lies in the scarcity of large-scale, high-quality video CoT datasets. We address this by building a fully automatic CoT curation pipeline. First, we devise a cognitioninspired prompting strategy to elicit a reasoning LLM to generate preliminary CoTs based solely on rich, structured, and literal representations of video content. Subsequently, these CoTs are revised by a visual-language model conditioned on the actual video, ensuring visual consistency and reducing visual hallucinations. This pipeline results in two new datasets - VideoRFT-CoT-102K for SFT and VideoRFT-RL-310K for RL. To further strength the RL phase, we introduce a novel semantic-consistency reward that explicitly promotes the alignment between textual reasoning with visual evidence. This reward encourages the model to produce coherent, context-aware reasoning outputs grounded in visual input. Extensive experiments show that VIDEORFT achieves state-of-the-art performance on six video reasoning benchmarks.
>
---
#### [new 049] NOFT: Test-Time Noise Finetune via Information Bottleneck for Highly Correlated Asset Creation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在通过优化扩散模型的噪声微调，解决生成高保真且多样化的2D/3D资产时内容保持与可控变化的平衡问题。提出NOFT模块，基于信息瓶颈理论在测试阶段微调噪声，仅需少量参数和训练时间即可实现拓扑和纹理对齐的高相关图像生成。**

- **链接: [http://arxiv.org/pdf/2505.12235v1](http://arxiv.org/pdf/2505.12235v1)**

> **作者:** Jia Li; Nan Gao; Huaibo Huang; Ran He
>
> **摘要:** The diffusion model has provided a strong tool for implementing text-to-image (T2I) and image-to-image (I2I) generation. Recently, topology and texture control are popular explorations, e.g., ControlNet, IP-Adapter, Ctrl-X, and DSG. These methods explicitly consider high-fidelity controllable editing based on external signals or diffusion feature manipulations. As for diversity, they directly choose different noise latents. However, the diffused noise is capable of implicitly representing the topological and textural manifold of the corresponding image. Moreover, it's an effective workbench to conduct the trade-off between content preservation and controllable variations. Previous T2I and I2I diffusion works do not explore the information within the compressed contextual latent. In this paper, we first propose a plug-and-play noise finetune NOFT module employed by Stable Diffusion to generate highly correlated and diverse images. We fine-tune seed noise or inverse noise through an optimal-transported (OT) information bottleneck (IB) with around only 14K trainable parameters and 10 minutes of training. Our test-time NOFT is good at producing high-fidelity image variations considering topology and texture alignments. Comprehensive experiments demonstrate that NOFT is a powerful general reimagine approach to efficiently fine-tune the 2D/3D AIGC assets with text or image guidance.
>
---
#### [new 050] Towards Visuospatial Cognition via Hierarchical Fusion of Visual Experts
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文属于多模态大语言模型的视觉空间推理任务，针对现有模型空间理解能力不足的问题，提出ViCA2模型（融合语义/空间双编码器）和ViCA-322K数据集，在基准测试中以7B参数超越大模型，实现高效空间认知。**

- **链接: [http://arxiv.org/pdf/2505.12363v1](http://arxiv.org/pdf/2505.12363v1)**

> **作者:** Qi Feng; Hidetoshi Shimodaira
>
> **备注:** 26 pages, 19 figures, 4 tables. Code, models, and dataset are available at our project page: https://github.com/nkkbr/ViCA
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at general vision-language tasks, visuospatial cognition - reasoning about spatial layouts, relations, and dynamics - remains a significant challenge. Existing models often lack the necessary architectural components and specialized training data for fine-grained spatial understanding. We introduce ViCA2 (Visuospatial Cognitive Assistant 2), a novel MLLM designed to enhance spatial reasoning. ViCA2 features a dual vision encoder architecture integrating SigLIP for semantics and Hiera for spatial structure, coupled with a token ratio control mechanism for efficiency. We also developed ViCA-322K, a new large-scale dataset with over 322,000 spatially grounded question-answer pairs for targeted instruction tuning. On the challenging VSI-Bench benchmark, our ViCA2-7B model achieves a state-of-the-art average score of 56.8, significantly surpassing larger open-source models (e.g., LLaVA-NeXT-Video-72B, 40.9) and leading proprietary models (Gemini-1.5 Pro, 45.4). This demonstrates the effectiveness of our approach in achieving strong visuospatial intelligence with a compact model. We release ViCA2, its codebase, and the ViCA-322K dataset to facilitate further research.
>
---
#### [new 051] SMFusion: Semantic-Preserving Fusion of Multimodal Medical Images for Enhanced Clinical Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于多模态医学图像融合任务，旨在解决传统方法忽视医学图像语义信息的问题。通过构建医学图像-文本数据集，利用BiomedGPT生成文本描述，设计语义对齐模块和跨注意力机制，将文本特征融入图像融合过程，并生成诊断报告验证信息保留效果，最终提出语义损失函数提升关键医学信息融合质量。**

- **链接: [http://arxiv.org/pdf/2505.12251v1](http://arxiv.org/pdf/2505.12251v1)**

> **作者:** Haozhe Xiang; Han Zhang; Yu Cheng; Xiongwen Quan; Wanwan Huang
>
> **摘要:** Multimodal medical image fusion plays a crucial role in medical diagnosis by integrating complementary information from different modalities to enhance image readability and clinical applicability. However, existing methods mainly follow computer vision standards for feature extraction and fusion strategy formulation, overlooking the rich semantic information inherent in medical images. To address this limitation, we propose a novel semantic-guided medical image fusion approach that, for the first time, incorporates medical prior knowledge into the fusion process. Specifically, we construct a publicly available multimodal medical image-text dataset, upon which text descriptions generated by BiomedGPT are encoded and semantically aligned with image features in a high-dimensional space via a semantic interaction alignment module. During this process, a cross attention based linear transformation automatically maps the relationship between textual and visual features to facilitate comprehensive learning. The aligned features are then embedded into a text-injection module for further feature-level fusion. Unlike traditional methods, we further generate diagnostic reports from the fused images to assess the preservation of medical information. Additionally, we design a medical semantic loss function to enhance the retention of textual cues from the source images. Experimental results on test datasets demonstrate that the proposed method achieves superior performance in both qualitative and quantitative evaluations while preserving more critical medical information.
>
---
#### [new 052] GenZSL: Generative Zero-Shot Learning Via Inductive Variational Autoencoder
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出GenZSL方法解决生成式零样本学习任务，针对现有方法依赖强专家语义向量导致生成效果差、泛化弱的问题。通过变分自编码器从相似已知类归纳新类样本，采用CLIP文本嵌入作为弱语义向量，结合类多样性促进和目标类信息增强策略，有效提升生成质量与训练效率，在基准数据集上实现24.7%性能提升和60倍加速。**

- **链接: [http://arxiv.org/pdf/2505.11882v1](http://arxiv.org/pdf/2505.11882v1)**

> **作者:** Shiming Chen; Dingjie Fu; Salman Khan; Fahad Shahbaz Khan
>
> **备注:** Accepted to ICML'25
>
> **摘要:** Remarkable progress in zero-shot learning (ZSL) has been achieved using generative models. However, existing generative ZSL methods merely generate (imagine) the visual features from scratch guided by the strong class semantic vectors annotated by experts, resulting in suboptimal generative performance and limited scene generalization. To address these and advance ZSL, we propose an inductive variational autoencoder for generative zero-shot learning, dubbed GenZSL. Mimicking human-level concept learning, GenZSL operates by inducting new class samples from similar seen classes using weak class semantic vectors derived from target class names (i.e., CLIP text embedding). To ensure the generation of informative samples for training an effective ZSL classifier, our GenZSL incorporates two key strategies. Firstly, it employs class diversity promotion to enhance the diversity of class semantic vectors. Secondly, it utilizes target class-guided information boosting criteria to optimize the model. Extensive experiments conducted on three popular benchmark datasets showcase the superiority and potential of our GenZSL with significant efficacy and efficiency over f-VAEGAN, e.g., 24.7% performance gains and more than $60\times$ faster training speed on AWA2. Codes are available at https://github.com/shiming-chen/GenZSL.
>
---
#### [new 053] Dynamic Graph Induced Contour-aware Heat Conduction Network for Event-based Object Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于事件流目标检测任务，旨在解决现有方法在轮廓建模和多尺度特征利用上的不足。提出CvHeat-DET模型，通过动态图热传导网络捕捉事件流轮廓信息，并融合多尺度图特征提升检测性能，在三个基准数据集上验证了有效性。**

- **链接: [http://arxiv.org/pdf/2505.12908v1](http://arxiv.org/pdf/2505.12908v1)**

> **作者:** Xiao Wang; Yu Jin; Lan Chen; Bo Jiang; Lin Zhu; Yonghong Tian; Jin Tang; Bin Luo
>
> **摘要:** Event-based Vision Sensors (EVS) have demonstrated significant advantages over traditional RGB frame-based cameras in low-light conditions, high-speed motion capture, and low latency. Consequently, object detection based on EVS has attracted increasing attention from researchers. Current event stream object detection algorithms are typically built upon Convolutional Neural Networks (CNNs) or Transformers, which either capture limited local features using convolutional filters or incur high computational costs due to the utilization of self-attention. Recently proposed vision heat conduction backbone networks have shown a good balance between efficiency and accuracy; however, these models are not specifically designed for event stream data. They exhibit weak capability in modeling object contour information and fail to exploit the benefits of multi-scale features. To address these issues, this paper proposes a novel dynamic graph induced contour-aware heat conduction network for event stream based object detection, termed CvHeat-DET. The proposed model effectively leverages the clear contour information inherent in event streams to predict the thermal diffusivity coefficients within the heat conduction model, and integrates hierarchical structural graph features to enhance feature learning across multiple scales. Extensive experiments on three benchmark datasets for event stream-based object detection fully validated the effectiveness of the proposed model. The source code of this paper will be released on https://github.com/Event-AHU/OpenEvDET.
>
---
#### [new 054] EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于机器人模仿学习任务，旨在解决灵巧操作数据稀缺问题。研究者利用Apple Vision Pro构建了EgoDex数据集——包含829小时带3D手部追踪的自我中心视频，覆盖194种日常操作任务，并通过训练策略建立轨迹预测基准，推动机器人及计算机视觉发展。**

- **链接: [http://arxiv.org/pdf/2505.11709v1](http://arxiv.org/pdf/2505.11709v1)**

> **作者:** Ryan Hoque; Peide Huang; David J. Yoon; Mouli Sivapurapu; Jian Zhang
>
> **摘要:** Imitation learning for manipulation has a well-known data scarcity problem. Unlike natural language and 2D computer vision, there is no Internet-scale corpus of data for dexterous manipulation. One appealing option is egocentric human video, a passively scalable data source. However, existing large-scale datasets such as Ego4D do not have native hand pose annotations and do not focus on object manipulation. To this end, we use Apple Vision Pro to collect EgoDex: the largest and most diverse dataset of dexterous human manipulation to date. EgoDex has 829 hours of egocentric video with paired 3D hand and finger tracking data collected at the time of recording, where multiple calibrated cameras and on-device SLAM can be used to precisely track the pose of every joint of each hand. The dataset covers a wide range of diverse manipulation behaviors with everyday household objects in 194 different tabletop tasks ranging from tying shoelaces to folding laundry. Furthermore, we train and systematically evaluate imitation learning policies for hand trajectory prediction on the dataset, introducing metrics and benchmarks for measuring progress in this increasingly important area. By releasing this large-scale dataset, we hope to push the frontier of robotics, computer vision, and foundation models.
>
---
#### [new 055] Is Artificial Intelligence Generated Image Detection a Solved Problem?
- **分类: cs.CV; cs.CR**

- **简介: 该论文聚焦AI生成图像检测的鲁棒性评估，属于模型泛化性研究。针对现有检测器在真实场景性能不足的问题，提出AIGIBench基准，通过多源泛化、抗干扰等四项任务测试11种检测器，发现其现实数据性能显著下降，揭示了当前方法的局限性，推动更可靠的检测技术发展。**

- **链接: [http://arxiv.org/pdf/2505.12335v1](http://arxiv.org/pdf/2505.12335v1)**

> **作者:** Ziqiang Li; Jiazhen Yan; Ziwen He; Kai Zeng; Weiwei Jiang; Lizhi Xiong; Zhangjie Fu
>
> **备注:** Under Review
>
> **摘要:** The rapid advancement of generative models, such as GANs and Diffusion models, has enabled the creation of highly realistic synthetic images, raising serious concerns about misinformation, deepfakes, and copyright infringement. Although numerous Artificial Intelligence Generated Image (AIGI) detectors have been proposed, often reporting high accuracy, their effectiveness in real-world scenarios remains questionable. To bridge this gap, we introduce AIGIBench, a comprehensive benchmark designed to rigorously evaluate the robustness and generalization capabilities of state-of-the-art AIGI detectors. AIGIBench simulates real-world challenges through four core tasks: multi-source generalization, robustness to image degradation, sensitivity to data augmentation, and impact of test-time pre-processing. It includes 23 diverse fake image subsets that span both advanced and widely adopted image generation techniques, along with real-world samples collected from social media and AI art platforms. Extensive experiments on 11 advanced detectors demonstrate that, despite their high reported accuracy in controlled settings, these detectors suffer significant performance drops on real-world data, limited benefits from common augmentations, and nuanced effects of pre-processing, highlighting the need for more robust detection strategies. By providing a unified and realistic evaluation framework, AIGIBench offers valuable insights to guide future research toward dependable and generalizable AIGI detection.
>
---
#### [new 056] CL-CaGAN: Capsule differential adversarial continuous learning for cross-domain hyperspectral anomaly detection
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文针对跨域高光谱异常检测任务，解决现有深度学习方法在开放场景中因先验信息不足和灾难性遗忘导致的性能下降问题。提出CL-CaGAN模型，结合胶囊网络与对抗生成网络估计背景分布，并集成聚类样本回放和自蒸馏正则化策略，实现持续学习以保留历史知识，提升跨域检测性能。实验验证其有效性和稳定性。**

- **链接: [http://arxiv.org/pdf/2505.11793v1](http://arxiv.org/pdf/2505.11793v1)**

> **作者:** Jianing Wang; Siying Guo; Zheng Hua; Runhu Huang; Jinyu Hu; Maoguo Gong
>
> **摘要:** Anomaly detection (AD) has attracted remarkable attention in hyperspectral image (HSI) processing fields, and most existing deep learning (DL)-based algorithms indicate dramatic potential for detecting anomaly samples through specific training process under current scenario. However, the limited prior information and the catastrophic forgetting problem indicate crucial challenges for existing DL structure in open scenarios cross-domain detection. In order to improve the detection performance, a novel continual learning-based capsule differential generative adversarial network (CL-CaGAN) is proposed to elevate the cross-scenario learning performance for facilitating the real application of DL-based structure in hyperspectral AD (HAD) task. First, a modified capsule structure with adversarial learning network is constructed to estimate the background distribution for surmounting the deficiency of prior information. To mitigate the catastrophic forgetting phenomenon, clustering-based sample replay strategy and a designed extra self-distillation regularization are integrated for merging the history and future knowledge in continual AD task, while the discriminative learning ability from previous detection scenario to current scenario is retained by the elaborately designed structure with continual learning (CL) strategy. In addition, the differentiable enhancement is enforced to augment the generation performance of the training data. This further stabilizes the training process with better convergence and efficiently consolidates the reconstruction ability of background samples. To verify the effectiveness of our proposed CL-CaGAN, we conduct experiments on several real HSIs, and the results indicate that the proposed CL-CaGAN demonstrates higher detection performance and continuous learning capacity for mitigating the catastrophic forgetting under cross-domain scenarios.
>
---
#### [new 057] RGB-to-Polarization Estimation: A New Task and Benchmark Study
- **分类: cs.CV**

- **简介: 该论文提出RGB图像到偏振信息估计的新任务，解决偏振图像因硬件限制获取困难的问题。通过整合现有数据集并评估多种深度学习模型，建立了首个综合基准，分析不同模型的性能与局限，为未来研究提供基础资源。**

- **链接: [http://arxiv.org/pdf/2505.13050v1](http://arxiv.org/pdf/2505.13050v1)**

> **作者:** Beibei Lin; Zifeng Yuan; Tingting Chen
>
> **摘要:** Polarization images provide rich physical information that is fundamentally absent from standard RGB images, benefiting a wide range of computer vision applications such as reflection separation and material classification. However, the acquisition of polarization images typically requires additional optical components, which increases both the cost and the complexity of the applications. To bridge this gap, we introduce a new task: RGB-to-polarization image estimation, which aims to infer polarization information directly from RGB images. In this work, we establish the first comprehensive benchmark for this task by leveraging existing polarization datasets and evaluating a diverse set of state-of-the-art deep learning models, including both restoration-oriented and generative architectures. Through extensive quantitative and qualitative analysis, our benchmark not only establishes the current performance ceiling of RGB-to-polarization estimation, but also systematically reveals the respective strengths and limitations of different model families -- such as direct reconstruction versus generative synthesis, and task-specific training versus large-scale pre-training. In addition, we provide some potential directions for future research on polarization estimation. This benchmark is intended to serve as a foundational resource to facilitate the design and evaluation of future methods for polarization estimation from standard RGB inputs.
>
---
#### [new 058] Pyramid Sparse Transformer: Enhancing Multi-Scale Feature Fusion with Dynamic Token Selection
- **分类: cs.CV**

- **简介: 该论文提出金字塔稀疏Transformer（PST），用于计算机视觉的多尺度特征融合任务。针对现有注意力方法计算复杂度过高的问题，PST通过动态token选择和参数共享降低计算成本，同时保持空间细节。作为轻量级即插即用模块，PST在YOLO和ResNet系列模型中显著提升检测（COCO mAP）和分类（ImageNet精度）性能，无需重新训练即可部署。**

- **链接: [http://arxiv.org/pdf/2505.12772v1](http://arxiv.org/pdf/2505.12772v1)**

> **作者:** Junyi Hu; Tian Bai; Fengyi Wu; Zhengming Peng; Yi Zhang
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Feature fusion is critical for high-performance vision models but often incurs prohibitive complexity. However, prevailing attention-based fusion methods often involve significant computational complexity and implementation challenges, limiting their efficiency in resource-constrained environments. To address these issues, we introduce the Pyramid Sparse Transformer (PST), a lightweight, plug-and-play module that integrates coarse-to-fine token selection and shared attention parameters to reduce computation while preserving spatial detail. PST can be trained using only coarse attention and seamlessly activated at inference for further accuracy gains without retraining. When added to state-of-the-art real-time detection models, such as YOLOv11-N/S/M, PST yields mAP improvements of 0.9%, 0.5%, and 0.4% on MS COCO with minimal latency impact. Likewise, embedding PST into ResNet-18/50/101 as backbones, boosts ImageNet top-1 accuracy by 6.5%, 1.7%, and 1.0%, respectively. These results demonstrate PST's effectiveness as a simple, hardware-friendly enhancement for both detection and classification tasks.
>
---
#### [new 059] Swin DiT: Diffusion Transformer using Pseudo Shifted Windows
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决Diffusion Transformers（DiTs）计算成本高、全局冗余及注意力低频惯性问题。作者提出伪移位窗口注意力（PSWA）减少全局冗余，结合高频桥接分支增强局部交互，并设计PCCA策略优化注意力相似性，构建Swin DiT模型。实验显示其性能显著提升（如FID↑54%），计算量更低。**

- **链接: [http://arxiv.org/pdf/2505.13219v1](http://arxiv.org/pdf/2505.13219v1)**

> **作者:** Jiafu Wu; Yabiao Wang; Jian Li; Jinlong Peng; Yun Cao; Chengjie Wang; Jiangning Zhang
>
> **摘要:** Diffusion Transformers (DiTs) achieve remarkable performance within the domain of image generation through the incorporation of the transformer architecture. Conventionally, DiTs are constructed by stacking serial isotropic global information modeling transformers, which face significant computational cost when processing high-resolution images. We empirically analyze that latent space image generation does not exhibit a strong dependence on global information as traditionally assumed. Most of the layers in the model demonstrate redundancy in global computation. In addition, conventional attention mechanisms exhibit low-frequency inertia issues. To address these issues, we propose \textbf{P}seudo \textbf{S}hifted \textbf{W}indow \textbf{A}ttention (PSWA), which fundamentally mitigates global model redundancy. PSWA achieves intermediate global-local information interaction through window attention, while employing a high-frequency bridging branch to simulate shifted window operations, supplementing appropriate global and high-frequency information. Furthermore, we propose the Progressive Coverage Channel Allocation(PCCA) strategy that captures high-order attention similarity without additional computational cost. Building upon all of them, we propose a series of Pseudo \textbf{S}hifted \textbf{Win}dow DiTs (\textbf{Swin DiT}), accompanied by extensive experiments demonstrating their superior performance. For example, our proposed Swin-DiT-L achieves a 54%$\uparrow$ FID improvement over DiT-XL/2 while requiring less computational. https://github.com/wujiafu007/Swin-DiT
>
---
#### [new 060] TACOcc:Target-Adaptive Cross-Modal Fusion with Volume Rendering for 3D Semantic Occupancy
- **分类: cs.CV**

- **简介: 该论文研究多模态3D语义占据预测任务，解决固定融合策略导致的几何语义错位及标注稀疏引发的表面细节丢失问题。提出目标尺度自适应的双向融合机制调节特征对齐，并设计基于3D高斯泼溅的体渲染流程增强2D-3D一致性监督，提升小目标重建精度。在主流数据集验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.12693v1](http://arxiv.org/pdf/2505.12693v1)**

> **作者:** Luyao Lei; Shuo Xu; Yifan Bai; Xing Wei
>
> **摘要:** The performance of multi-modal 3D occupancy prediction is limited by ineffective fusion, mainly due to geometry-semantics mismatch from fixed fusion strategies and surface detail loss caused by sparse, noisy annotations. The mismatch stems from the heterogeneous scale and distribution of point cloud and image features, leading to biased matching under fixed neighborhood fusion. To address this, we propose a target-scale adaptive, bidirectional symmetric retrieval mechanism. It expands the neighborhood for large targets to enhance context awareness and shrinks it for small ones to improve efficiency and suppress noise, enabling accurate cross-modal feature alignment. This mechanism explicitly establishes spatial correspondences and improves fusion accuracy. For surface detail loss, sparse labels provide limited supervision, resulting in poor predictions for small objects. We introduce an improved volume rendering pipeline based on 3D Gaussian Splatting, which takes fused features as input to render images, applies photometric consistency supervision, and jointly optimizes 2D-3D consistency. This enhances surface detail reconstruction while suppressing noise propagation. In summary, we propose TACOcc, an adaptive multi-modal fusion framework for 3D semantic occupancy prediction, enhanced by volume rendering supervision. Experiments on the nuScenes and SemanticKITTI benchmarks validate its effectiveness.
>
---
#### [new 061] MVAR: Visual Autoregressive Modeling with Scale and Spatial Markovian Conditioning
- **分类: cs.CV**

- **简介: 该论文属于视觉生成任务，旨在解决传统自回归模型存在的多尺度冗余和计算复杂度高的问题。提出MVAR框架，通过尺度马尔可夫轨迹仅关联相邻尺度特征，并设计空间马尔可夫注意力限制局部区域交互，将注意力复杂度从O(N²)降至O(Nk)，在降低GPU内存消耗的同时保持生成性能。实验验证其在ImageNet上效果优于基线模型。**

- **链接: [http://arxiv.org/pdf/2505.12742v1](http://arxiv.org/pdf/2505.12742v1)**

> **作者:** Jinhua Zhang; Wei Long; Minghao Han; Weiyi You; Shuhang Gu
>
> **摘要:** Essential to visual generation is efficient modeling of visual data priors. Conventional next-token prediction methods define the process as learning the conditional probability distribution of successive tokens. Recently, next-scale prediction methods redefine the process to learn the distribution over multi-scale representations, significantly reducing generation latency. However, these methods condition each scale on all previous scales and require each token to consider all preceding tokens, exhibiting scale and spatial redundancy. To better model the distribution by mitigating redundancy, we propose Markovian Visual AutoRegressive modeling (MVAR), a novel autoregressive framework that introduces scale and spatial Markov assumptions to reduce the complexity of conditional probability modeling. Specifically, we introduce a scale-Markov trajectory that only takes as input the features of adjacent preceding scale for next-scale prediction, enabling the adoption of a parallel training strategy that significantly reduces GPU memory consumption. Furthermore, we propose spatial-Markov attention, which restricts the attention of each token to a localized neighborhood of size k at corresponding positions on adjacent scales, rather than attending to every token across these scales, for the pursuit of reduced modeling complexity. Building on these improvements, we reduce the computational complexity of attention calculation from O(N^2) to O(Nk), enabling training with just eight NVIDIA RTX 4090 GPUs and eliminating the need for KV cache during inference. Extensive experiments on ImageNet demonstrate that MVAR achieves comparable or superior performance with both small model trained from scratch and large fine-tuned models, while reducing the average GPU memory footprint by 3.0x.
>
---
#### [new 062] FEALLM: Advancing Facial Emotion Analysis in Multimodal Large Language Models with Emotional Synergy and Reasoning
- **分类: cs.CV**

- **简介: 该论文针对面部情绪分析任务，解决传统方法可解释性差、现有多模态大模型缺乏专用数据及无法捕捉表情-动作单元关系的问题。提出包含因果推理的指令数据集、FEABench基准及FEALLM模型，通过增强面部细节建模提升性能，验证了泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.13419v1](http://arxiv.org/pdf/2505.13419v1)**

> **作者:** Zhuozhao Hu; Kaishen Yuan; Xin Liu; Zitong Yu; Yuan Zong; Jingang Shi; Huanjing Yue; Jingyu Yang
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Facial Emotion Analysis (FEA) plays a crucial role in visual affective computing, aiming to infer a person's emotional state based on facial data. Scientifically, facial expressions (FEs) result from the coordinated movement of facial muscles, which can be decomposed into specific action units (AUs) that provide detailed emotional insights. However, traditional methods often struggle with limited interpretability, constrained generalization and reasoning abilities. Recently, Multimodal Large Language Models (MLLMs) have shown exceptional performance in various visual tasks, while they still face significant challenges in FEA due to the lack of specialized datasets and their inability to capture the intricate relationships between FEs and AUs. To address these issues, we introduce a novel FEA Instruction Dataset that provides accurate and aligned FE and AU descriptions and establishes causal reasoning relationships between them, followed by constructing a new benchmark, FEABench. Moreover, we propose FEALLM, a novel MLLM architecture designed to capture more detailed facial information, enhancing its capability in FEA tasks. Our model demonstrates strong performance on FEABench and impressive generalization capability through zero-shot evaluation on various datasets, including RAF-DB, AffectNet, BP4D, and DISFA, showcasing its robustness and effectiveness in FEA tasks. The dataset and code will be available at https://github.com/953206211/FEALLM.
>
---
#### [new 063] Towards Open-world Generalized Deepfake Detection: General Feature Extraction via Unsupervised Domain Adaptation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于开放世界深度伪造检测任务，旨在解决有限标记数据下检测未知伪造方法的难题。提出OWG-DS策略，通过域距离优化和类边界分离模块，结合对抗训练实现跨域特征对齐与泛化增强，提升模型在未标记数据中的检测能力。**

- **链接: [http://arxiv.org/pdf/2505.12339v1](http://arxiv.org/pdf/2505.12339v1)**

> **作者:** Midou Guo; Qilin Yin; Wei Lu; Xiangyang Luo
>
> **摘要:** With the development of generative artificial intelligence, new forgery methods are rapidly emerging. Social platforms are flooded with vast amounts of unlabeled synthetic data and authentic data, making it increasingly challenging to distinguish real from fake. Due to the lack of labels, existing supervised detection methods struggle to effectively address the detection of unknown deepfake methods. Moreover, in open world scenarios, the amount of unlabeled data greatly exceeds that of labeled data. Therefore, we define a new deepfake detection generalization task which focuses on how to achieve efficient detection of large amounts of unlabeled data based on limited labeled data to simulate a open world scenario. To solve the above mentioned task, we propose a novel Open-World Deepfake Detection Generalization Enhancement Training Strategy (OWG-DS) to improve the generalization ability of existing methods. Our approach aims to transfer deepfake detection knowledge from a small amount of labeled source domain data to large-scale unlabeled target domain data. Specifically, we introduce the Domain Distance Optimization (DDO) module to align different domain features by optimizing both inter-domain and intra-domain distances. Additionally, the Similarity-based Class Boundary Separation (SCBS) module is used to enhance the aggregation of similar samples to ensure clearer class boundaries, while an adversarial training mechanism is adopted to learn the domain-invariant features. Extensive experiments show that the proposed deepfake detection generalization enhancement training strategy excels in cross-method and cross-dataset scenarios, improving the model's generalization.
>
---
#### [new 064] Adaptive Image Restoration for Video Surveillance: A Real-Time Approach
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像修复任务，针对视频监控中雨雾等导致的图像退化问题，传统方法无法实时处理。研究提出基于迁移学习的ResNet-50模型，自动识别退化类型并匹配修复处理，实现实时、灵活的自适应恢复方案。**

- **链接: [http://arxiv.org/pdf/2505.13130v1](http://arxiv.org/pdf/2505.13130v1)**

> **作者:** Muhammad Awais Amin; Adama Ilboudo; Abdul Samad bin Shahid; Amjad Ali; Waqas Haider Khan Bangyal
>
> **摘要:** One of the major challenges in the field of computer vision especially for detection, segmentation, recognition, monitoring, and automated solutions, is the quality of images. Image degradation, often caused by factors such as rain, fog, lighting, etc., has a negative impact on automated decision-making.Furthermore, several image restoration solutions exist, including restoration models for single degradation and restoration models for multiple degradations. However, these solutions are not suitable for real-time processing. In this study, the aim was to develop a real-time image restoration solution for video surveillance. To achieve this, using transfer learning with ResNet_50, we developed a model for automatically identifying the types of degradation present in an image to reference the necessary treatment(s) for image restoration. Our solution has the advantage of being flexible and scalable.
>
---
#### [new 065] Few-Step Diffusion via Score identity Distillation
- **分类: cs.CV; cs.LG; stat.ML**

- **简介: 该论文属于扩散模型加速任务，解决现有方法依赖真实/合成图像及文本对齐-多样性权衡问题。提出数据无关的Score identity Distillation框架，通过理论分析混合生成步骤输出匹配数据分布，结合对抗损失和Zero/Anti-CFG策略，在SDXL等模型上实现高效少步生成，平衡性能与多样性。**

- **链接: [http://arxiv.org/pdf/2505.12674v1](http://arxiv.org/pdf/2505.12674v1)**

> **作者:** Mingyuan Zhou; Yi Gu; Zhendong Wang
>
> **摘要:** Diffusion distillation has emerged as a promising strategy for accelerating text-to-image (T2I) diffusion models by distilling a pretrained score network into a one- or few-step generator. While existing methods have made notable progress, they often rely on real or teacher-synthesized images to perform well when distilling high-resolution T2I diffusion models such as Stable Diffusion XL (SDXL), and their use of classifier-free guidance (CFG) introduces a persistent trade-off between text-image alignment and generation diversity. We address these challenges by optimizing Score identity Distillation (SiD) -- a data-free, one-step distillation framework -- for few-step generation. Backed by theoretical analysis that justifies matching a uniform mixture of outputs from all generation steps to the data distribution, our few-step distillation algorithm avoids step-specific networks and integrates seamlessly into existing pipelines, achieving state-of-the-art performance on SDXL at 1024x1024 resolution. To mitigate the alignment-diversity trade-off when real text-image pairs are available, we introduce a Diffusion GAN-based adversarial loss applied to the uniform mixture and propose two new guidance strategies: Zero-CFG, which disables CFG in the teacher and removes text conditioning in the fake score network, and Anti-CFG, which applies negative CFG in the fake score network. This flexible setup improves diversity without sacrificing alignment. Comprehensive experiments on SD1.5 and SDXL demonstrate state-of-the-art performance in both one-step and few-step generation settings, along with robustness to the absence of real images. Our efficient PyTorch implementation, along with the resulting one- and few-step distilled generators, will be released publicly as a separate branch at https://github.com/mingyuanzhou/SiD-LSG.
>
---
#### [new 066] Temporal-Spectral-Spatial Unified Remote Sensing Dense Prediction
- **分类: cs.CV**

- **简介: 该论文针对遥感数据在时空谱维度的异构性问题，提出统一网络TSSUN，解决现有模型因输入输出配置差异导致的性能下降和兼容性差问题。通过元信息解耦标准化输入、统一输出结构及局部-全局注意力机制，实现多源遥感数据下密集预测任务（如语义分割）的统一建模，无需任务调整即达到最优性能。**

- **链接: [http://arxiv.org/pdf/2505.12280v1](http://arxiv.org/pdf/2505.12280v1)**

> **作者:** Sijie Zhao; Feng Liu; Xueliang Zhang; Hao Chen; Pengfeng Xiao; Lei Bai
>
> **备注:** 12 pages, 4 figures, Code link:https://github.com/walking-shadow/Official_TSSUN
>
> **摘要:** The proliferation of diverse remote sensing data has spurred advancements in dense prediction tasks, yet significant challenges remain in handling data heterogeneity. Remote sensing imagery exhibits substantial variability across temporal, spectral, and spatial (TSS) dimensions, complicating unified data processing. Current deep learning models for dense prediction tasks, such as semantic segmentation and change detection, are typically tailored to specific input-output configurations. Consequently, variations in data dimensionality or task requirements often lead to significant performance degradation or model incompatibility, necessitating costly retraining or fine-tuning efforts for different application scenarios. This paper introduces the Temporal-Spectral-Spatial Unified Network (TSSUN), a novel architecture designed for unified representation and modeling of remote sensing data across diverse TSS characteristics and task types. TSSUN employs a Temporal-Spectral-Spatial Unified Strategy that leverages meta-information to decouple and standardize input representations from varied temporal, spectral, and spatial configurations, and similarly unifies output structures for different dense prediction tasks and class numbers. Furthermore, a Local-Global Window Attention mechanism is proposed to efficiently capture both local contextual details and global dependencies, enhancing the model's adaptability and feature extraction capabilities. Extensive experiments on multiple datasets demonstrate that a single TSSUN model effectively adapts to heterogeneous inputs and unifies various dense prediction tasks. The proposed approach consistently achieves or surpasses state-of-the-art performance, highlighting its robustness and generalizability for complex remote sensing applications without requiring task-specific modifications.
>
---
#### [new 067] VFRTok: Variable Frame Rates Video Tokenizer with Duration-Proportional Information Assumption
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，针对现有分词器因固定时间压缩率导致计算成本高的问题，提出信息量与时长而非帧数相关的新假设，设计基于Transformer的VFRTok，通过非对称帧率训练和Partial RoPE解耦位置与内容建模，以1/8的token量实现高效高质量生成。**

- **链接: [http://arxiv.org/pdf/2505.12053v1](http://arxiv.org/pdf/2505.12053v1)**

> **作者:** Tianxiong Zhong; Xingye Tian; Boyuan Jiang; Xuebo Wang; Xin Tao; Pengfei Wan; Zhiwei Zhang
>
> **备注:** 11 pages, 10 figures
>
> **摘要:** Modern video generation frameworks based on Latent Diffusion Models suffer from inefficiencies in tokenization due to the Frame-Proportional Information Assumption. Existing tokenizers provide fixed temporal compression rates, causing the computational cost of the diffusion model to scale linearly with the frame rate. The paper proposes the Duration-Proportional Information Assumption: the upper bound on the information capacity of a video is proportional to the duration rather than the number of frames. Based on this insight, the paper introduces VFRTok, a Transformer-based video tokenizer, that enables variable frame rate encoding and decoding through asymmetric frame rate training between the encoder and decoder. Furthermore, the paper proposes Partial Rotary Position Embeddings (RoPE) to decouple position and content modeling, which groups correlated patches into unified tokens. The Partial RoPE effectively improves content-awareness, enhancing the video generation capability. Benefiting from the compact and continuous spatio-temporal representation, VFRTok achieves competitive reconstruction quality and state-of-the-art generation fidelity while using only 1/8 tokens compared to existing tokenizers.
>
---
#### [new 068] MonoMobility: Zero-Shot 3D Mobility Analysis from Monocular Videos
- **分类: cs.CV**

- **简介: 该论文属于3D运动分析任务，解决单目视频中物体运动解析依赖标注/多视角数据的问题。提出零样本框架，通过深度估计、光流分析和动态场景优化，从单目视频自动识别运动部件及其旋转/平移属性，无需训练数据，验证了复杂运动分析的灵活性。**

- **链接: [http://arxiv.org/pdf/2505.11868v1](http://arxiv.org/pdf/2505.11868v1)**

> **作者:** Hongyi Zhou; Xiaogang Wang; Yulan Guo; Kai Xu
>
> **摘要:** Accurately analyzing the motion parts and their motion attributes in dynamic environments is crucial for advancing key areas such as embodied intelligence. Addressing the limitations of existing methods that rely on dense multi-view images or detailed part-level annotations, we propose an innovative framework that can analyze 3D mobility from monocular videos in a zero-shot manner. This framework can precisely parse motion parts and motion attributes only using a monocular video, completely eliminating the need for annotated training data. Specifically, our method first constructs the scene geometry and roughly analyzes the motion parts and their initial motion attributes combining depth estimation, optical flow analysis and point cloud registration method, then employs 2D Gaussian splatting for scene representation. Building on this, we introduce an end-to-end dynamic scene optimization algorithm specifically designed for articulated objects, refining the initial analysis results to ensure the system can handle 'rotation', 'translation', and even complex movements ('rotation+translation'), demonstrating high flexibility and versatility. To validate the robustness and wide applicability of our method, we created a comprehensive dataset comprising both simulated and real-world scenarios. Experimental results show that our framework can effectively analyze articulated object motions in an annotation-free manner, showcasing its significant potential in future embodied intelligence applications.
>
---
#### [new 069] TS-VLM: Text-Guided SoftSort Pooling for Vision-Language Models in Multi-View Driving Reasoning
- **分类: cs.CV**

- **简介: 该论文提出轻量级视觉语言模型TS-VLM，针对自动驾驶中多视图数据融合效率低、计算成本高的问题，设计文本引导的SoftSort池化模块，动态聚合语义相关视图特征，在提升多视图推理准确性的同时减少90%计算量，适用于实时自动驾驶场景。**

- **链接: [http://arxiv.org/pdf/2505.12670v1](http://arxiv.org/pdf/2505.12670v1)**

> **作者:** Lihong Chen; Hossein Hassani; Soodeh Nikan
>
> **摘要:** Vision-Language Models (VLMs) have shown remarkable potential in advancing autonomous driving by leveraging multi-modal fusion in order to enhance scene perception, reasoning, and decision-making. Despite their potential, existing models suffer from computational overhead and inefficient integration of multi-view sensor data that make them impractical for real-time deployment in safety-critical autonomous driving applications. To address these shortcomings, this paper is devoted to designing a lightweight VLM called TS-VLM, which incorporates a novel Text-Guided SoftSort Pooling (TGSSP) module. By resorting to semantics of the input queries, TGSSP ranks and fuses visual features from multiple views, enabling dynamic and query-aware multi-view aggregation without reliance on costly attention mechanisms. This design ensures the query-adaptive prioritization of semantically related views, which leads to improved contextual accuracy in multi-view reasoning for autonomous driving. Extensive evaluations on the DriveLM benchmark demonstrate that, on the one hand, TS-VLM outperforms state-of-the-art models with a BLEU-4 score of 56.82, METEOR of 41.91, ROUGE-L of 74.64, and CIDEr of 3.39. On the other hand, TS-VLM reduces computational cost by up to 90%, where the smallest version contains only 20.1 million parameters, making it more practical for real-time deployment in autonomous vehicles.
>
---
#### [new 070] UGoDIT: Unsupervised Group Deep Image Prior Via Transferable Weights
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文提出UGoDIT，解决逆成像任务（如MRI重建、图像修复）在低数据量下的难题。针对传统深度生成模型依赖大量干净数据及无监督方法易过拟合、计算低效的缺陷，设计无监督共享编码器和多解码器结构，预训练可迁移权重，测试时固定部分参数并优化测量一致性，在医疗和自然图像恢复中实现快速收敛与高质量重建，性能接近监督模型。**

- **链接: [http://arxiv.org/pdf/2505.11720v1](http://arxiv.org/pdf/2505.11720v1)**

> **作者:** Shijun Liang; Ismail R. Alkhouri; Siddhant Gautam; Qing Qu; Saiprasad Ravishankar
>
> **摘要:** Recent advances in data-centric deep generative models have led to significant progress in solving inverse imaging problems. However, these models (e.g., diffusion models (DMs)) typically require large amounts of fully sampled (clean) training data, which is often impractical in medical and scientific settings such as dynamic imaging. On the other hand, training-data-free approaches like the Deep Image Prior (DIP) do not require clean ground-truth images but suffer from noise overfitting and can be computationally expensive as the network parameters need to be optimized for each measurement set independently. Moreover, DIP-based methods often overlook the potential of learning a prior using a small number of sub-sampled measurements (or degraded images) available during training. In this paper, we propose UGoDIT, an Unsupervised Group DIP via Transferable weights, designed for the low-data regime where only a very small number, M, of sub-sampled measurement vectors are available during training. Our method learns a set of transferable weights by optimizing a shared encoder and M disentangled decoders. At test time, we reconstruct the unseen degraded image using a DIP network, where part of the parameters are fixed to the learned weights, while the remaining are optimized to enforce measurement consistency. We evaluate UGoDIT on both medical (multi-coil MRI) and natural (super resolution and non-linear deblurring) image recovery tasks under various settings. Compared to recent standalone DIP methods, UGoDIT provides accelerated convergence and notable improvement in reconstruction quality. Furthermore, our method achieves performance competitive with SOTA DM-based and supervised approaches, despite not requiring large amounts of clean training data.
>
---
#### [new 071] SurveillanceVQA-589K: A Benchmark for Comprehensive Surveillance Video-Language Understanding with Large Models
- **分类: cs.CV**

- **简介: 该论文属于监控视频语言理解任务，旨在解决现有模型在复杂监控场景（如因果推理、异常分析）中理解不足的问题。作者构建了包含58.9万QA对的大规模数据集SurveillanceVQA-589K，涵盖12类认知问题，采用人机协同标注方法，并提出多维评估协议验证模型在时空因果推理等维度的缺陷。**

- **链接: [http://arxiv.org/pdf/2505.12589v1](http://arxiv.org/pdf/2505.12589v1)**

> **作者:** Bo Liu; Pengfei Qiao; Minhan Ma; Xuange Zhang; Yinan Tang; Peng Xu; Kun Liu; Tongtong Yuan
>
> **备注:** The dataset and code are publicly available at: https://huggingface.co/datasets/fei213/SurveillanceVQA-589K
>
> **摘要:** Understanding surveillance video content remains a critical yet underexplored challenge in vision-language research, particularly due to its real-world complexity, irregular event dynamics, and safety-critical implications. In this work, we introduce SurveillanceVQA-589K, the largest open-ended video question answering benchmark tailored to the surveillance domain. The dataset comprises 589,380 QA pairs spanning 12 cognitively diverse question types, including temporal reasoning, causal inference, spatial understanding, and anomaly interpretation, across both normal and abnormal video scenarios. To construct the benchmark at scale, we design a hybrid annotation pipeline that combines temporally aligned human-written captions with Large Vision-Language Model-assisted QA generation using prompt-based techniques. We also propose a multi-dimensional evaluation protocol to assess contextual, temporal, and causal comprehension. We evaluate eight LVLMs under this framework, revealing significant performance gaps, especially in causal and anomaly-related tasks, underscoring the limitations of current models in real-world surveillance contexts. Our benchmark provides a practical and comprehensive resource for advancing video-language understanding in safety-critical applications such as intelligent monitoring, incident analysis, and autonomous decision-making.
>
---
#### [new 072] GTR: Gaussian Splatting Tracking and Reconstruction of Unknown Objects Based on Appearance and Geometric Complexity
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出了一种基于3D高斯溅射的自适应方法，用于单目RGBD视频的6-DoF物体跟踪与高精度3D重建。针对复杂物体（如对称、几何/外观复杂）的跟踪重建难题，结合混合几何/外观跟踪与关键帧选择，提升鲁棒性与精度，并构建涵盖挑战性物体的评测基准，实现了开放环境下单传感器重建的新突破。**

- **链接: [http://arxiv.org/pdf/2505.11905v1](http://arxiv.org/pdf/2505.11905v1)**

> **作者:** Takuya Ikeda; Sergey Zakharov; Muhammad Zubair Irshad; Istvan Balazs Opra; Shun Iwase; Dian Chen; Mark Tjersland; Robert Lee; Alexandre Dilly; Rares Ambrus; Koichi Nishiwaki
>
> **备注:** main contains 10 pages, 9 figures. And supplementary material contains 10 pages, 27 figures
>
> **摘要:** We present a novel method for 6-DoF object tracking and high-quality 3D reconstruction from monocular RGBD video. Existing methods, while achieving impressive results, often struggle with complex objects, particularly those exhibiting symmetry, intricate geometry or complex appearance. To bridge these gaps, we introduce an adaptive method that combines 3D Gaussian Splatting, hybrid geometry/appearance tracking, and key frame selection to achieve robust tracking and accurate reconstructions across a diverse range of objects. Additionally, we present a benchmark covering these challenging object classes, providing high-quality annotations for evaluating both tracking and reconstruction performance. Our approach demonstrates strong capabilities in recovering high-fidelity object meshes, setting a new standard for single-sensor 3D reconstruction in open-world environments.
>
---
#### [new 073] LiDAR MOT-DETR: A LiDAR-based Two-Stage Transformer for 3D Multiple Object Tracking
- **分类: cs.CV**

- **简介: 该论文研究基于LiDAR的3D多目标跟踪任务，针对点云稀疏性及传统方法在复杂场景中身份切换的问题，提出两阶段Transformer模型：平滑阶段优化检测框，跟踪阶段通过DETR注意力机制关联时序目标，在nuScenes/KITTI数据集上实现优于基线的在线跟踪性能（aMOTA 0.722），离线模式进一步提升精度。**

- **链接: [http://arxiv.org/pdf/2505.12753v1](http://arxiv.org/pdf/2505.12753v1)**

> **作者:** Martha Teiko Teye; Ori Maoz; Matthias Rottmann
>
> **摘要:** Multi-object tracking from LiDAR point clouds presents unique challenges due to the sparse and irregular nature of the data, compounded by the need for temporal coherence across frames. Traditional tracking systems often rely on hand-crafted features and motion models, which can struggle to maintain consistent object identities in crowded or fast-moving scenes. We present a lidar-based two-staged DETR inspired transformer; a smoother and tracker. The smoother stage refines lidar object detections, from any off-the-shelf detector, across a moving temporal window. The tracker stage uses a DETR-based attention block to maintain tracks across time by associating tracked objects with the refined detections using the point cloud as context. The model is trained on the datasets nuScenes and KITTI in both online and offline (forward peeking) modes demonstrating strong performance across metrics such as ID-switch and multiple object tracking accuracy (MOTA). The numerical results indicate that the online mode outperforms the lidar-only baseline and SOTA models on the nuScenes dataset, with an aMOTA of 0.722 and an aMOTP of 0.475, while the offline mode provides an additional 3 pp aMOTP
>
---
#### [new 074] MT-CYP-Net: Multi-Task Network for Pixel-Level Crop Yield Prediction Under Very Few Samples
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于农业产量预测任务，旨在解决卫星遥感数据中像素级作物产量预测因地面真值稀缺导致的精度受限问题。研究者提出多任务网络MT-CYP-Net，通过共享特征和协同训练产量预测与作物分类解码器，利用少量样本（1,859个标记点）生成精细产量分布图，在黑龙江农场数据中验证优于传统方法。**

- **链接: [http://arxiv.org/pdf/2505.12069v1](http://arxiv.org/pdf/2505.12069v1)**

> **作者:** Shenzhou Liu; Di Wang; Haonan Guo; Chengxi Han; Wenzhi Zeng
>
> **摘要:** Accurate and fine-grained crop yield prediction plays a crucial role in advancing global agriculture. However, the accuracy of pixel-level yield estimation based on satellite remote sensing data has been constrained by the scarcity of ground truth data. To address this challenge, we propose a novel approach called the Multi-Task Crop Yield Prediction Network (MT-CYP-Net). This framework introduces an effective multi-task feature-sharing strategy, where features extracted from a shared backbone network are simultaneously utilized by both crop yield prediction decoders and crop classification decoders with the ability to fuse information between them. This design allows MT-CYP-Net to be trained with extremely sparse crop yield point labels and crop type labels, while still generating detailed pixel-level crop yield maps. Concretely, we collected 1,859 yield point labels along with corresponding crop type labels and satellite images from eight farms in Heilongjiang Province, China, in 2023, covering soybean, maize, and rice crops, and constructed a sparse crop yield label dataset. MT-CYP-Net is compared with three classical machine learning and deep learning benchmark methods in this dataset. Experimental results not only indicate the superiority of MT-CYP-Net compared to previous methods on multiple types of crops but also demonstrate the potential of deep networks on precise pixel-level crop yield prediction, especially with limited data labels.
>
---
#### [new 075] Event-based Star Tracking under Spacecraft Jitter: the e-STURT Dataset
- **分类: cs.CV; eess.SP**

- **简介: 该论文针对航天器抖动影响光学任务精度的问题，构建首个事件相机抖动星跟踪数据集e-STURT，通过压电执行器模拟高频抖动并采集真实数据，包含200组序列，同时提出基于事件流的抖动估计算法，为空间传感任务提供算法开发基础。**

- **链接: [http://arxiv.org/pdf/2505.12588v1](http://arxiv.org/pdf/2505.12588v1)**

> **作者:** Samya Bagchi; Peter Anastasiou; Matthew Tetlow; Tat-Jun Chin; Yasir Latif
>
> **摘要:** Jitter degrades a spacecraft's fine-pointing ability required for optical communication, earth observation, and space domain awareness. Development of jitter estimation and compensation algorithms requires high-fidelity sensor observations representative of on-board jitter. In this work, we present the Event-based Star Tracking Under Jitter (e-STURT) dataset -- the first event camera based dataset of star observations under controlled jitter conditions. Specialized hardware employed for the dataset emulates an event-camera undergoing on-board jitter. While the event camera provides asynchronous, high temporal resolution star observations, systematic and repeatable jitter is introduced using a micrometer accurate piezoelectric actuator. Various jitter sources are simulated using distinct frequency bands and utilizing both axes of motion. Ground-truth jitter is captured in hardware from the piezoelectric actuator. The resulting dataset consists of 200 sequences and is made publicly available. This work highlights the dataset generation process, technical challenges and the resulting limitations. To serve as a baseline, we propose a high-frequency jitter estimation algorithm that operates directly on the event stream. The e-STURT dataset will enable the development of jitter aware algorithms for mission critical event-based space sensing applications.
>
---
#### [new 076] KinTwin: Imitation Learning with Torque and Muscle Driven Biomechanical Models Enables Precise Replication of Able-Bodied and Impaired Movement from Markerless Motion Capture
- **分类: cs.CV**

- **简介: 该论文属于运动分析的逆动力学任务，旨在通过模仿学习结合生物力学模型，从无标记运动捕捉数据中推断关节扭矩、肌肉激活等物理参数，解决临床运动分析中动力学推断难题。工作包括构建肌肉驱动模型，训练于正常/受损运动数据，验证其精确复制运动及推断临床指标的能力。**

- **链接: [http://arxiv.org/pdf/2505.13436v1](http://arxiv.org/pdf/2505.13436v1)**

> **作者:** R. James Cotton
>
> **摘要:** Broader access to high-quality movement analysis could greatly benefit movement science and rehabilitation, such as allowing more detailed characterization of movement impairments and responses to interventions, or even enabling early detection of new neurological conditions or fall risk. While emerging technologies are making it easier to capture kinematics with biomechanical models, or how joint angles change over time, inferring the underlying physics that give rise to these movements, including ground reaction forces, joint torques, or even muscle activations, is still challenging. Here we explore whether imitation learning applied to a biomechanical model from a large dataset of movements from able-bodied and impaired individuals can learn to compute these inverse dynamics. Although imitation learning in human pose estimation has seen great interest in recent years, our work differences in several ways: we focus on using an accurate biomechanical model instead of models adopted for computer vision, we test it on a dataset that contains participants with impaired movements, we reported detailed tracking metrics relevant for the clinical measurement of movement including joint angles and ground contact events, and finally we apply imitation learning to a muscle-driven neuromusculoskeletal model. We show that our imitation learning policy, KinTwin, can accurately replicate the kinematics of a wide range of movements, including those with assistive devices or therapist assistance, and that it can infer clinically meaningful differences in joint torques and muscle activations. Our work demonstrates the potential for using imitation learning to enable high-quality movement analysis in clinical practice.
>
---
#### [new 077] DNOI-4DRO: Deep 4D Radar Odometry with Differentiable Neural-Optimization Iterations
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出DNOI-4DRO模型，属于4D雷达里程计任务，旨在解决稀疏点云场景下运动估计精度不足的问题。通过融合神经网络与几何优化，设计双流特征提取网络增强点云表征，并构建可微优化模块迭代优化雷达姿态，在多个数据集上超越现有方法，接近激光雷达方案效果。**

- **链接: [http://arxiv.org/pdf/2505.12310v1](http://arxiv.org/pdf/2505.12310v1)**

> **作者:** Shouyi Lu; Huanyu Zhou; Guirong Zhuo
>
> **备注:** 16 pages,10 figures
>
> **摘要:** A novel learning-optimization-combined 4D radar odometry model, named DNOI-4DRO, is proposed in this paper. The proposed model seamlessly integrates traditional geometric optimization with end-to-end neural network training, leveraging an innovative differentiable neural-optimization iteration operator. In this framework, point-wise motion flow is first estimated using a neural network, followed by the construction of a cost function based on the relationship between point motion and pose in 3D space. The radar pose is then refined using Gauss-Newton updates. Additionally, we design a dual-stream 4D radar backbone that integrates multi-scale geometric features and clustering-based class-aware features to enhance the representation of sparse 4D radar point clouds. Extensive experiments on the VoD and Snail-Radar datasets demonstrate the superior performance of our model, which outperforms recent classical and learning-based approaches. Notably, our method even achieves results comparable to A-LOAM with mapping optimization using LiDAR point clouds as input. Our models and code will be publicly released.
>
---
#### [new 078] Towards a Universal Image Degradation Model via Content-Degradation Disentanglement
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像退化合成任务，旨在解决现有模型泛化性差、依赖人工参数的问题。作者提出首个通用退化模型，通过内容-退化解耦方法分离全局与空间退化特征，结合新型模块实现无干预的复杂退化合成，应用于胶片模拟与盲复原任务。**

- **链接: [http://arxiv.org/pdf/2505.12860v1](http://arxiv.org/pdf/2505.12860v1)**

> **作者:** Wenbo Yang; Zhongling Wang; Zhou Wang
>
> **摘要:** Image degradation synthesis is highly desirable in a wide variety of applications ranging from image restoration to simulating artistic effects. Existing models are designed to generate one specific or a narrow set of degradations, which often require user-provided degradation parameters. As a result, they lack the generalizability to synthesize degradations beyond their initial design or adapt to other applications. Here we propose the first universal degradation model that can synthesize a broad spectrum of complex and realistic degradations containing both homogeneous (global) and inhomogeneous (spatially varying) components. Our model automatically extracts and disentangles homogeneous and inhomogeneous degradation features, which are later used for degradation synthesis without user intervention. A disentangle-by-compression method is proposed to separate degradation information from images. Two novel modules for extracting and incorporating inhomogeneous degradations are created to model inhomogeneous components in complex degradations. We demonstrate the model's accuracy and adaptability in film-grain simulation and blind image restoration tasks. The demo video, code, and dataset of this project will be released upon publication at github.com/yangwenbo99/content-degradation-disentanglement.
>
---
#### [new 079] CL-BioGAN: Biologically-Inspired Cross-Domain Continual Learning for Hyperspectral Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文针对跨场景高光谱异常检测（HAD）中持续学习的记忆稳定性与灵活性矛盾问题，提出生物启发的CL-BioGAN模型。通过设计含主动遗忘损失的生物损失函数和自注意力生成对抗网络，平衡新旧任务参数释放与背景分布拟合，在降低计算成本的同时提升跨域检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.11796v1](http://arxiv.org/pdf/2505.11796v1)**

> **作者:** Jianing Wang; Zheng Hua; Wan Zhang; Shengjia Hao; Yuqiong Yao; Maoguo Gong
>
> **摘要:** Memory stability and learning flexibility in continual learning (CL) is a core challenge for cross-scene Hyperspectral Anomaly Detection (HAD) task. Biological neural networks can actively forget history knowledge that conflicts with the learning of new experiences by regulating learning-triggered synaptic expansion and synaptic convergence. Inspired by this phenomenon, we propose a novel Biologically-Inspired Continual Learning Generative Adversarial Network (CL-BioGAN) for augmenting continuous distribution fitting ability for cross-domain HAD task, where Continual Learning Bio-inspired Loss (CL-Bio Loss) and self-attention Generative Adversarial Network (BioGAN) are incorporated to realize forgetting history knowledge as well as involving replay strategy in the proposed BioGAN. Specifically, a novel Bio-Inspired Loss composed with an Active Forgetting Loss (AF Loss) and a CL loss is designed to realize parameters releasing and enhancing between new task and history tasks from a Bayesian perspective. Meanwhile, BioGAN loss with L2-Norm enhances self-attention (SA) to further balance the stability and flexibility for better fitting background distribution for open scenario HAD (OHAD) tasks. Experiment results underscore that the proposed CL-BioGAN can achieve more robust and satisfying accuracy for cross-domain HAD with fewer parameters and computation cost. This dual contribution not only elevates CL performance but also offers new insights into neural adaptation mechanisms in OHAD task.
>
---
#### [new 080] CURE: Concept Unlearning via Orthogonal Representation Editing in Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于扩散模型安全干预任务，旨在解决现有方法无法高效、精准消除有害概念（侵权、隐私等）且影响模型性能的问题。提出CURE框架：通过正交投影模块Spectral Eraser在权重空间分解概念嵌入，单步闭式操作隔离目标特征并保留正常生成能力，2秒内完成无监督概念遗忘。**

- **链接: [http://arxiv.org/pdf/2505.12677v1](http://arxiv.org/pdf/2505.12677v1)**

> **作者:** Shristi Das Biswas; Arani Roy; Kaushik Roy
>
> **摘要:** As Text-to-Image models continue to evolve, so does the risk of generating unsafe, copyrighted, or privacy-violating content. Existing safety interventions - ranging from training data curation and model fine-tuning to inference-time filtering and guidance - often suffer from incomplete concept removal, susceptibility to jail-breaking, computational inefficiency, or collateral damage to unrelated capabilities. In this paper, we introduce CURE, a training-free concept unlearning framework that operates directly in the weight space of pre-trained diffusion models, enabling fast, interpretable, and highly specific suppression of undesired concepts. At the core of our method is the Spectral Eraser, a closed-form, orthogonal projection module that identifies discriminative subspaces using Singular Value Decomposition over token embeddings associated with the concepts to forget and retain. Intuitively, the Spectral Eraser identifies and isolates features unique to the undesired concept while preserving safe attributes. This operator is then applied in a single step update to yield an edited model in which the target concept is effectively unlearned - without retraining, supervision, or iterative optimization. To balance the trade-off between filtering toxicity and preserving unrelated concepts, we further introduce an Expansion Mechanism for spectral regularization which selectively modulates singular vectors based on their relative significance to control the strength of forgetting. All the processes above are in closed-form, guaranteeing extremely efficient erasure in only $2$ seconds. Benchmarking against prior approaches, CURE achieves a more efficient and thorough removal for targeted artistic styles, objects, identities, or explicit content, with minor damage to original generation ability and demonstrates enhanced robustness against red-teaming.
>
---
#### [new 081] X-Edit: Detecting and Localizing Edits in Images Altered by Text-Guided Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像篡改检测任务，旨在定位由文本引导扩散模型修改的图像区域。针对现有方法难以检测细微深度伪造编辑的问题，提出X-Edit方法：通过扩散模型反演图像特征，结合通道/空间注意力分割网络预测编辑掩码，并设计双损失函数优化定位效果。同时构建新数据集，实验显示其PSNR/SSIM指标优于基线。**

- **链接: [http://arxiv.org/pdf/2505.11753v1](http://arxiv.org/pdf/2505.11753v1)**

> **作者:** Valentina Bazyleva; Nicolo Bonettini; Gaurav Bharaj
>
> **备注:** CVPR (XAI4CV) 2025
>
> **摘要:** Text-guided diffusion models have significantly advanced image editing, enabling highly realistic and local modifications based on textual prompts. While these developments expand creative possibilities, their malicious use poses substantial challenges for detection of such subtle deepfake edits. To this end, we introduce Explain Edit (X-Edit), a novel method for localizing diffusion-based edits in images. To localize the edits for an image, we invert the image using a pretrained diffusion model, then use these inverted features as input to a segmentation network that explicitly predicts the edited masked regions via channel and spatial attention. Further, we finetune the model using a combined segmentation and relevance loss. The segmentation loss ensures accurate mask prediction by balancing pixel-wise errors and perceptual similarity, while the relevance loss guides the model to focus on low-frequency regions and mitigate high-frequency artifacts, enhancing the localization of subtle edits. To the best of our knowledge, we are the first to address and model the problem of localizing diffusion-based modified regions in images. We additionally contribute a new dataset of paired original and edited images addressing the current lack of resources for this task. Experimental results demonstrate that X-Edit accurately localizes edits in images altered by text-guided diffusion models, outperforming baselines in PSNR and SSIM metrics. This highlights X-Edit's potential as a robust forensic tool for detecting and pinpointing manipulations introduced by advanced image editing techniques.
>
---
#### [new 082] G1: Bootstrapping Perception and Reasoning Abilities of Vision-Language Model via Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文属于强化学习驱动的视觉语言模型（VLM）决策能力优化任务，旨在解决VLM在交互式视觉环境中感知与决策脱节的“知行差距”。通过构建多游戏训练环境VLM-Gym，提出G1模型，结合感知增强预训练与强化学习微调，实现感知推理协同进化，提升游戏决策性能，超越现有模型。**

- **链接: [http://arxiv.org/pdf/2505.13426v1](http://arxiv.org/pdf/2505.13426v1)**

> **作者:** Liang Chen; Hongcheng Gao; Tianyu Liu; Zhiqi Huang; Flood Sung; Xinyu Zhou; Yuxin Wu; Baobao Chang
>
> **备注:** 21 pages, 14 figures, code released at https://github.com/chenllliang/G1
>
> **摘要:** Vision-Language Models (VLMs) excel in many direct multimodal tasks but struggle to translate this prowess into effective decision-making within interactive, visually rich environments like games. This ``knowing-doing'' gap significantly limits their potential as autonomous agents, as leading VLMs often performing badly in simple games. To address this, we introduce VLM-Gym, a curated reinforcement learning (RL) environment featuring diverse visual games with unified interfaces and adjustable, compositional difficulty, specifically designed for scalable multi-game parallel training. Leveraging VLM-Gym, we train G0 models using pure RL-driven self-evolution, which demonstrate emergent perception and reasoning patterns. To further mitigate challenges arising from game diversity, we develop G1 models. G1 incorporates a perception-enhanced cold start prior to RL fine-tuning. Our resulting G1 models consistently surpass their teacher across all games and outperform leading proprietary models like Claude-3.7-Sonnet-Thinking. Systematic analysis reveals an intriguing finding: perception and reasoning abilities mutually bootstrap each other throughout the RL training process. Source code including VLM-Gym and RL training are released at https://github.com/chenllliang/G1 to foster future research in advancing VLMs as capable interactive agents.
>
---
#### [new 083] Multimodal Cancer Survival Analysis via Hypergraph Learning with Cross-Modality Rebalance
- **分类: cs.CV**

- **简介: 该论文属于癌症生存预测任务，旨在解决现有方法忽略病理图像上下文信息及病理-基因组模态不平衡的问题。提出基于超图学习的框架，捕获病理图像的层次细节，并通过动态重平衡和交互对齐策略协调双模态贡献，实验显示C-Index提升超3.4%。**

- **链接: [http://arxiv.org/pdf/2505.11997v1](http://arxiv.org/pdf/2505.11997v1)**

> **作者:** Mingcheng Qu; Guang Yang; Donglin; Tonghua Su; Yue Gao; Yang Song; Lei Fan
>
> **备注:** Code: https://github.com/MCPathology/MRePath
>
> **摘要:** Multimodal pathology-genomic analysis has become increasingly prominent in cancer survival prediction. However, existing studies mainly utilize multi-instance learning to aggregate patch-level features, neglecting the information loss of contextual and hierarchical details within pathology images. Furthermore, the disparity in data granularity and dimensionality between pathology and genomics leads to a significant modality imbalance. The high spatial resolution inherent in pathology data renders it a dominant role while overshadowing genomics in multimodal integration. In this paper, we propose a multimodal survival prediction framework that incorporates hypergraph learning to effectively capture both contextual and hierarchical details from pathology images. Moreover, it employs a modality rebalance mechanism and an interactive alignment fusion strategy to dynamically reweight the contributions of the two modalities, thereby mitigating the pathology-genomics imbalance. Quantitative and qualitative experiments are conducted on five TCGA datasets, demonstrating that our model outperforms advanced methods by over 3.4\% in C-Index performance.
>
---
#### [new 084] Hyperspectral Image Land Cover Captioning Dataset for Vision Language Models
- **分类: cs.CV**

- **简介: 该论文属于高光谱图像与文本描述结合的多模态任务，旨在解决传统HSI数据集仅支持分类、缺乏语义理解的问题。作者构建了首个大规模高光谱描述数据集HyperCap，融合像素级文本标注与光谱数据，通过混合标注方法提升模型分类性能，为遥感视觉语言模型提供新基准。**

- **链接: [http://arxiv.org/pdf/2505.12217v1](http://arxiv.org/pdf/2505.12217v1)**

> **作者:** Aryan Das; Tanishq Rachamalla; Pravendra Singh; Koushik Biswas; Vinay Kumar Verma; Swalpa Kumar Roy
>
> **摘要:** We introduce HyperCap, the first large-scale hyperspectral captioning dataset designed to enhance model performance and effectiveness in remote sensing applications. Unlike traditional hyperspectral imaging (HSI) datasets that focus solely on classification tasks, HyperCap integrates spectral data with pixel-wise textual annotations, enabling deeper semantic understanding of hyperspectral imagery. This dataset enhances model performance in tasks like classification and feature extraction, providing a valuable resource for advanced remote sensing applications. HyperCap is constructed from four benchmark datasets and annotated through a hybrid approach combining automated and manual methods to ensure accuracy and consistency. Empirical evaluations using state-of-the-art encoders and diverse fusion techniques demonstrate significant improvements in classification performance. These results underscore the potential of vision-language learning in HSI and position HyperCap as a foundational dataset for future research in the field.
>
---
#### [new 085] SGD-Mix: Enhancing Domain-Specific Image Classification with Label-Preserving Data Augmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对领域特定图像分类任务中数据增强存在的多样性、忠实度与标签清晰度不足问题，提出SGD-Mix框架。通过显着性引导混合和优化扩散模型，保留前景语义、增强背景多样性并维持标签一致性，解决了传统方法在生成质量与模型稳定性上的缺陷，实验验证其在多场景分类任务中的优越性。**

- **链接: [http://arxiv.org/pdf/2505.11813v1](http://arxiv.org/pdf/2505.11813v1)**

> **作者:** Yixuan Dong; Fang-Yi Su; Jung-Hsien Chiang
>
> **备注:** 11 pages, 6 figures, 6 tables
>
> **摘要:** Data augmentation for domain-specific image classification tasks often struggles to simultaneously address diversity, faithfulness, and label clarity of generated data, leading to suboptimal performance in downstream tasks. While existing generative diffusion model-based methods aim to enhance augmentation, they fail to cohesively tackle these three critical aspects and often overlook intrinsic challenges of diffusion models, such as sensitivity to model characteristics and stochasticity under strong transformations. In this paper, we propose a novel framework that explicitly integrates diversity, faithfulness, and label clarity into the augmentation process. Our approach employs saliency-guided mixing and a fine-tuned diffusion model to preserve foreground semantics, enrich background diversity, and ensure label consistency, while mitigating diffusion model limitations. Extensive experiments across fine-grained, long-tail, few-shot, and background robustness tasks demonstrate our method's superior performance over state-of-the-art approaches.
>
---
#### [new 086] Predicting Reaction Time to Comprehend Scenes with Foveated Scene Understanding Maps
- **分类: cs.CV**

- **简介: 该论文属于视觉场景理解任务，旨在预测人类理解场景的反应时间。针对现有模型无法有效量化场景理解时间的问题，提出结合视觉语言模型和中央凹视觉处理的F-SUM模型，通过生成空间场景理解图及其评分，其指标与人类反应时间(r=0.47)、眼动次数(r=0.51)和描述准确率(r=-0.56)显著相关，效果优于传统图像指标。**

- **链接: [http://arxiv.org/pdf/2505.12660v1](http://arxiv.org/pdf/2505.12660v1)**

> **作者:** Ziqi Wen; Jonathan Skaza; Shravan Murlidaran; William Y. Wang; Miguel P. Eckstein
>
> **摘要:** Although models exist that predict human response times (RTs) in tasks such as target search and visual discrimination, the development of image-computable predictors for scene understanding time remains an open challenge. Recent advances in vision-language models (VLMs), which can generate scene descriptions for arbitrary images, combined with the availability of quantitative metrics for comparing linguistic descriptions, offer a new opportunity to model human scene understanding. We hypothesize that the primary bottleneck in human scene understanding and the driving source of variability in response times across scenes is the interaction between the foveated nature of the human visual system and the spatial distribution of task-relevant visual information within an image. Based on this assumption, we propose a novel image-computable model that integrates foveated vision with VLMs to produce a spatially resolved map of scene understanding as a function of fixation location (Foveated Scene Understanding Map, or F-SUM), along with an aggregate F-SUM score. This metric correlates with average (N=17) human RTs (r=0.47) and number of saccades (r=0.51) required to comprehend a scene (across 277 scenes). The F-SUM score also correlates with average (N=16) human description accuracy (r=-0.56) in time-limited presentations. These correlations significantly exceed those of standard image-based metrics such as clutter, visual complexity, and scene ambiguity based on language entropy. Together, our work introduces a new image-computable metric for predicting human response times in scene understanding and demonstrates the importance of foveated visual processing in shaping comprehension difficulty.
>
---
#### [new 087] VLC Fusion: Vision-Language Conditioned Sensor Fusion for Robust Object Detection
- **分类: cs.CV**

- **简介: 该论文属于多模态目标检测任务，旨在解决传统传感器融合方法因忽视环境变化导致模态权重调整不佳的问题。提出VLC Fusion框架，利用视觉语言模型捕捉环境上下文（如黑暗、雨雾）动态调整传感器权重，在自动驾驶和军事数据集（含图像/LiDAR/红外）中验证了检测鲁棒性，优于传统方法。**

- **链接: [http://arxiv.org/pdf/2505.12715v1](http://arxiv.org/pdf/2505.12715v1)**

> **作者:** Aditya Taparia; Noel Ngu; Mario Leiva; Joshua Shay Kricheli; John Corcoran; Nathaniel D. Bastian; Gerardo Simari; Paulo Shakarian; Ransalu Senanayake
>
> **备注:** 12 pages, 19 figures
>
> **摘要:** Although fusing multiple sensor modalities can enhance object detection performance, existing fusion approaches often overlook subtle variations in environmental conditions and sensor inputs. As a result, they struggle to adaptively weight each modality under such variations. To address this challenge, we introduce Vision-Language Conditioned Fusion (VLC Fusion), a novel fusion framework that leverages a Vision-Language Model (VLM) to condition the fusion process on nuanced environmental cues. By capturing high-level environmental context such as as darkness, rain, and camera blurring, the VLM guides the model to dynamically adjust modality weights based on the current scene. We evaluate VLC Fusion on real-world autonomous driving and military target detection datasets that include image, LIDAR, and mid-wave infrared modalities. Our experiments show that VLC Fusion consistently outperforms conventional fusion baselines, achieving improved detection accuracy in both seen and unseen scenarios.
>
---
#### [new 088] WriteViT: Handwritten Text Generation with Vision Transformer
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于手写文本生成任务，旨在解决低数据场景下机器难以分离内容与书写风格的问题，尤其针对越南语等复杂文字。提出WriteViT框架：基于视觉Transformer构建风格编码器、多尺度生成器和识别器，通过条件位置编码增强细节捕获能力，实现了跨语言高质量生成与风格适配。**

- **链接: [http://arxiv.org/pdf/2505.13235v1](http://arxiv.org/pdf/2505.13235v1)**

> **作者:** Dang Hoai Nam; Huynh Tong Dang Khoa; Vo Nguyen Le Duy
>
> **摘要:** Humans can quickly generalize handwriting styles from a single example by intuitively separating content from style. Machines, however, struggle with this task, especially in low-data settings, often missing subtle spatial and stylistic cues. Motivated by this gap, we introduce WriteViT, a one-shot handwritten text synthesis framework that incorporates Vision Transformers (ViT), a family of models that have shown strong performance across various computer vision tasks. WriteViT integrates a ViT-based Writer Identifier for extracting style embeddings, a multi-scale generator built with Transformer encoder-decoder blocks enhanced by conditional positional encoding (CPE), and a lightweight ViT-based recognizer. While previous methods typically rely on CNNs or CRNNs, our design leverages transformers in key components to better capture both fine-grained stroke details and higher-level style information. Although handwritten text synthesis has been widely explored, its application to Vietnamese -- a language rich in diacritics and complex typography -- remains limited. Experiments on Vietnamese and English datasets demonstrate that WriteViT produces high-quality, style-consistent handwriting while maintaining strong recognition performance in low-resource scenarios. These results highlight the promise of transformer-based designs for multilingual handwriting generation and efficient style adaptation.
>
---
#### [new 089] MMS-VPR: Multimodal Street-Level Visual Place Recognition Dataset and Benchmark
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出MMS-VPR多模态街景视觉位置识别数据集，解决现有VPR数据集在行人密集场景、多模态数据和非西方城市环境中的不足。通过构建包含7.8万图像和2500视频的中国商业区数据集，提供GPS标注、时空元数据及空间图结构，支持多模态与图神经网络的定位研究，推动复杂场景下的地理空间理解。**

- **链接: [http://arxiv.org/pdf/2505.12254v1](http://arxiv.org/pdf/2505.12254v1)**

> **作者:** Yiwei Ou; Xiaobin Ren; Ronggui Sun; Guansong Gao; Ziyi Jiang; Kaiqi Zhao; Manfredo Manfredini
>
> **摘要:** Existing visual place recognition (VPR) datasets predominantly rely on vehicle-mounted imagery, lack multimodal diversity and underrepresent dense, mixed-use street-level spaces, especially in non-Western urban contexts. To address these gaps, we introduce MMS-VPR, a large-scale multimodal dataset for street-level place recognition in complex, pedestrian-only environments. The dataset comprises 78,575 annotated images and 2,512 video clips captured across 207 locations in a ~70,800 $\mathrm{m}^2$ open-air commercial district in Chengdu, China. Each image is labeled with precise GPS coordinates, timestamp, and textual metadata, and covers varied lighting conditions, viewpoints, and timeframes. MMS-VPR follows a systematic and replicable data collection protocol with minimal device requirements, lowering the barrier for scalable dataset creation. Importantly, the dataset forms an inherent spatial graph with 125 edges, 81 nodes, and 1 subgraph, enabling structure-aware place recognition. We further define two application-specific subsets -- Dataset_Edges and Dataset_Points -- to support fine-grained and graph-based evaluation tasks. Extensive benchmarks using conventional VPR models, graph neural networks, and multimodal baselines show substantial improvements when leveraging multimodal and structural cues. MMS-VPR facilitates future research at the intersection of computer vision, geospatial understanding, and multimodal reasoning. The dataset is publicly available at https://huggingface.co/datasets/Yiwei-Ou/MMS-VPR.
>
---
#### [new 090] Adversarial Robustness for Unified Multi-Modal Encoders via Efficient Calibration
- **分类: cs.CV**

- **简介: 该论文研究统一多模态编码器的对抗鲁棒性，解决其在轻微扰动下性能骤降的问题。提出高效校准框架，通过冻结预训练模型并训练模态专用投影头，结合三种目标函数和正则化策略，在六种模态中提升抗干扰能力达47.3%，同时保持原有性能。**

- **链接: [http://arxiv.org/pdf/2505.11895v1](http://arxiv.org/pdf/2505.11895v1)**

> **作者:** Chih-Ting Liao; Bin Ren; Guofeng Mei; Xu Zheng
>
> **摘要:** Recent unified multi-modal encoders align a wide range of modalities into a shared representation space, enabling diverse cross-modal tasks. Despite their impressive capabilities, the robustness of these models under adversarial perturbations remains underexplored, which is a critical concern for safety-sensitive applications. In this work, we present the first comprehensive study of adversarial vulnerability in unified multi-modal encoders. We find that even mild adversarial perturbations lead to substantial performance drops across all modalities. Non-visual inputs, such as audio and point clouds, are especially fragile, while visual inputs like images and videos also degrade significantly. To address this, we propose an efficient adversarial calibration framework that improves robustness across modalities without modifying pretrained encoders or semantic centers, ensuring compatibility with existing foundation models. Our method introduces modality-specific projection heads trained solely on adversarial examples, while keeping the backbone and embeddings frozen. We explore three training objectives: fixed-center cross-entropy, clean-to-adversarial L2 alignment, and clean-adversarial InfoNCE, and we introduce a regularization strategy to ensure modality-consistent alignment under attack. Experiments on six modalities and three Bind-style models show that our method improves adversarial robustness by up to 47.3 percent at epsilon = 4/255, while preserving or even improving clean zero-shot and retrieval performance with less than 1 percent trainable parameters.
>
---
#### [new 091] Learning Cross-Spectral Point Features with Task-Oriented Training
- **分类: cs.CV**

- **简介: 该论文属于跨光谱图像匹配与配准任务，旨在提升无人机在低可见度环境下的导航能力。针对可见光与热成像相机特征差异问题，提出任务导向训练方法：通过可微分配准管道联合优化特征网络的匹配与配准误差，使模型在MultiPoint数据集上75%估计达到10像素内误差，并兼容传统流程。**

- **链接: [http://arxiv.org/pdf/2505.12593v1](http://arxiv.org/pdf/2505.12593v1)**

> **作者:** Mia Thomas; Trevor Ablett; Jonathan Kelly
>
> **备注:** Proceedings of the {IEEE} International Conference on Robotics and Automation {(ICRA'25)} Thermal Infrared in Robotics (TIRO) Workshop, Atlanta, Georgia, USA, May 19, 2025
>
> **摘要:** Unmanned aerial vehicles (UAVs) enable operations in remote and hazardous environments, yet the visible-spectrum, camera-based navigation systems often relied upon by UAVs struggle in low-visibility conditions. Thermal cameras, which capture long-wave infrared radiation, are able to function effectively in darkness and smoke, where visible-light cameras fail. This work explores learned cross-spectral (thermal-visible) point features as a means to integrate thermal imagery into established camera-based navigation systems. Existing methods typically train a feature network's detection and description outputs directly, which often focuses training on image regions where thermal and visible-spectrum images exhibit similar appearance. Aiming to more fully utilize the available data, we propose a method to train the feature network on the tasks of matching and registration. We run our feature network on thermal-visible image pairs, then feed the network response into a differentiable registration pipeline. Losses are applied to the matching and registration estimates of this pipeline. Our selected model, trained on the task of matching, achieves a registration error (corner error) below 10 pixels for more than 75% of estimates on the MultiPoint dataset. We further demonstrate that our model can also be used with a classical pipeline for matching and registration.
>
---
#### [new 092] AoP-SAM: Automation of Prompts for Efficient Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分割任务，旨在解决SAM模型依赖手动提示的低效问题。提出AoP-SAM方法，通过轻量级Prompt Predictor自动生成最优位置提示，结合自适应采样过滤机制，在保持零样本泛化能力的同时提升分割效率与精度，适用于自动化场景。**

- **链接: [http://arxiv.org/pdf/2505.11980v1](http://arxiv.org/pdf/2505.11980v1)**

> **作者:** Yi Chen; Mu-Young Son; Chuanbo Hua; Joo-Young Kim
>
> **备注:** Accepted at AAAI 2025
>
> **摘要:** The Segment Anything Model (SAM) is a powerful foundation model for image segmentation, showing robust zero-shot generalization through prompt engineering. However, relying on manual prompts is impractical for real-world applications, particularly in scenarios where rapid prompt provision and resource efficiency are crucial. In this paper, we propose the Automation of Prompts for SAM (AoP-SAM), a novel approach that learns to generate essential prompts in optimal locations automatically. AoP-SAM enhances SAM's efficiency and usability by eliminating manual input, making it better suited for real-world tasks. Our approach employs a lightweight yet efficient Prompt Predictor model that detects key entities across images and identifies the optimal regions for placing prompt candidates. This method leverages SAM's image embeddings, preserving its zero-shot generalization capabilities without requiring fine-tuning. Additionally, we introduce a test-time instance-level Adaptive Sampling and Filtering mechanism that generates prompts in a coarse-to-fine manner. This notably enhances both prompt and mask generation efficiency by reducing computational overhead and minimizing redundant mask refinements. Evaluations of three datasets demonstrate that AoP-SAM substantially improves both prompt generation efficiency and mask generation accuracy, making SAM more effective for automated segmentation tasks.
>
---
#### [new 093] Multi-modal Collaborative Optimization and Expansion Network for Event-assisted Single-eye Expression Recognition
- **分类: cs.CV**

- **简介: 该论文针对单眼表情识别任务中低光照、高曝光等挑战，提出多模态协作优化网络MCO-E Net。通过MCO-Mamba实现事件模态与视觉模态的联合优化，利用HCE-MoE动态分配异构专家网络整合互补特征，有效提升恶劣光照条件下的表情语义捕捉能力。**

- **链接: [http://arxiv.org/pdf/2505.12007v1](http://arxiv.org/pdf/2505.12007v1)**

> **作者:** Runduo Han; Xiuping Liu; Shangxuan Yi; Yi Zhang; Hongchen Tan
>
> **摘要:** In this paper, we proposed a Multi-modal Collaborative Optimization and Expansion Network (MCO-E Net), to use event modalities to resist challenges such as low light, high exposure, and high dynamic range in single-eye expression recognition tasks. The MCO-E Net introduces two innovative designs: Multi-modal Collaborative Optimization Mamba (MCO-Mamba) and Heterogeneous Collaborative and Expansion Mixture-of-Experts (HCE-MoE). MCO-Mamba, building upon Mamba, leverages dual-modal information to jointly optimize the model, facilitating collaborative interaction and fusion of modal semantics. This approach encourages the model to balance the learning of both modalities and harness their respective strengths. HCE-MoE, on the other hand, employs a dynamic routing mechanism to distribute structurally varied experts (deep, attention, and focal), fostering collaborative learning of complementary semantics. This heterogeneous architecture systematically integrates diverse feature extraction paradigms to comprehensively capture expression semantics. Extensive experiments demonstrate that our proposed network achieves competitive performance in the task of single-eye expression recognition, especially under poor lighting conditions.
>
---
#### [new 094] Industry-focused Synthetic Segmentation Pre-training
- **分类: cs.CV**

- **简介: 该论文研究工业实例分割预训练任务，解决真实图像使用受限及领域迁移问题。提出InsCore合成数据集，通过公式生成含复杂遮挡、密集掩码的标注图像，无需真实数据或人工标注。实验表明，使用仅10万合成图像预训练的模型在工业数据集上超越COCO、ImageNet-21k及微调SAM，性能提升6.2%，数据效率高百倍。**

- **链接: [http://arxiv.org/pdf/2505.13099v1](http://arxiv.org/pdf/2505.13099v1)**

> **作者:** Shinichi Mae; Ryosuke Yamada; Hirokatsu Kataoka
>
> **摘要:** Pre-training on real-image datasets has been widely proven effective for improving instance segmentation. However, industrial applications face two key challenges: (1) legal and ethical restrictions, such as ImageNet's prohibition of commercial use, and (2) limited transferability due to the domain gap between web images and industrial imagery. Even recent vision foundation models, including the segment anything model (SAM), show notable performance degradation in industrial settings. These challenges raise critical questions: Can we build a vision foundation model for industrial applications without relying on real images or manual annotations? And can such models outperform even fine-tuned SAM on industrial datasets? To address these questions, we propose the Instance Core Segmentation Dataset (InsCore), a synthetic pre-training dataset based on formula-driven supervised learning (FDSL). InsCore generates fully annotated instance segmentation images that reflect key characteristics of industrial data, including complex occlusions, dense hierarchical masks, and diverse non-rigid shapes, distinct from typical web imagery. Unlike previous methods, InsCore requires neither real images nor human annotations. Experiments on five industrial datasets show that models pre-trained with InsCore outperform those trained on COCO and ImageNet-21k, as well as fine-tuned SAM, achieving an average improvement of 6.2 points in instance segmentation performance. This result is achieved using only 100k synthetic images, more than 100 times fewer than the 11 million images in SAM's SA-1B dataset, demonstrating the data efficiency of our approach. These findings position InsCore as a practical and license-free vision foundation model for industrial applications.
>
---
#### [new 095] ProMi: An Efficient Prototype-Mixture Baseline for Few-Shot Segmentation with Bounding-Box Annotations
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于小样本分割任务，旨在解决像素级标注成本高的问题。提出ProMi方法，利用边界框标注代替精细标注，将背景建模为混合分布，无需训练，在多个数据集上取得最佳效果，适用于机器人实际场景。**

- **链接: [http://arxiv.org/pdf/2505.12547v1](http://arxiv.org/pdf/2505.12547v1)**

> **作者:** Florent Chiaroni; Ali Ayub; Ola Ahmad
>
> **摘要:** In robotics applications, few-shot segmentation is crucial because it allows robots to perform complex tasks with minimal training data, facilitating their adaptation to diverse, real-world environments. However, pixel-level annotations of even small amount of images is highly time-consuming and costly. In this paper, we present a novel few-shot binary segmentation method based on bounding-box annotations instead of pixel-level labels. We introduce, ProMi, an efficient prototype-mixture-based method that treats the background class as a mixture of distributions. Our approach is simple, training-free, and effective, accommodating coarse annotations with ease. Compared to existing baselines, ProMi achieves the best results across different datasets with significant gains, demonstrating its effectiveness. Furthermore, we present qualitative experiments tailored to real-world mobile robot tasks, demonstrating the applicability of our approach in such scenarios. Our code: https://github.com/ThalesGroup/promi.
>
---
#### [new 096] MatPredict: a dataset and benchmark for learning material properties of diverse indoor objects
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于材料属性识别任务，解决消费机器人通过视觉识别室内物体材质的难题。通过结合Replica的3D模型和MatSynth材料库，构建了包含18类物体和14种材质的MatPredict数据集，生成不同光照和视角的渲染图像，并建立视觉推断基准，测试神经网络模型性能，通过精确光效模拟提升训练效果，推动机器人感知发展。**

- **链接: [http://arxiv.org/pdf/2505.13201v1](http://arxiv.org/pdf/2505.13201v1)**

> **作者:** Yuzhen Chen; Hojun Son; Arpan Kusari
>
> **摘要:** Determining material properties from camera images can expand the ability to identify complex objects in indoor environments, which is valuable for consumer robotics applications. To support this, we introduce MatPredict, a dataset that combines the high-quality synthetic objects from Replica dataset with MatSynth dataset's material properties classes - to create objects with diverse material properties. We select 3D meshes of specific foreground objects and render them with different material properties. In total, we generate \textbf{18} commonly occurring objects with \textbf{14} different materials. We showcase how we provide variability in terms of lighting and camera placement for these objects. Next, we provide a benchmark for inferring material properties from visual images using these perturbed models in the scene, discussing the specific neural network models involved and their performance based on different image comparison metrics. By accurately simulating light interactions with different materials, we can enhance realism, which is crucial for training models effectively through large-scale simulations. This research aims to revolutionize perception in consumer robotics. The dataset is provided \href{https://huggingface.co/datasets/UMTRI/MatPredict}{here} and the code is provided \href{https://github.com/arpan-kusari/MatPredict}{here}.
>
---
#### [new 097] Multiscale Adaptive Conflict-Balancing Model For Multimedia Deepfake Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究多模态深度伪造检测任务，解决现有方法模态学习不平衡及冲突问题。提出MACB-DF模型，采用对比学习实现跨模态融合平衡，设计正交化模块缓解编码器梯度冲突，在主流数据集实现95.5%平均准确率，显著提升跨数据集泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.12966v1](http://arxiv.org/pdf/2505.12966v1)**

> **作者:** Zihan Xiong; Xiaohua Wu; Lei Chen; Fangqi Lou
>
> **备注:** 9 pages,ICMR accepted
>
> **摘要:** Advances in computer vision and deep learning have blurred the line between deepfakes and authentic media, undermining multimedia credibility through audio-visual forgery. Current multimodal detection methods remain limited by unbalanced learning between modalities. To tackle this issue, we propose an Audio-Visual Joint Learning Method (MACB-DF) to better mitigate modality conflicts and neglect by leveraging contrastive learning to assist in multi-level and cross-modal fusion, thereby fully balancing and exploiting information from each modality. Additionally, we designed an orthogonalization-multimodal pareto module that preserves unimodal information while addressing gradient conflicts in audio-video encoders caused by differing optimization targets of the loss functions. Extensive experiments and ablation studies conducted on mainstream deepfake datasets demonstrate consistent performance gains of our model across key evaluation metrics, achieving an average accuracy of 95.5% across multiple datasets. Notably, our method exhibits superior cross-dataset generalization capabilities, with absolute improvements of 8.0% and 7.7% in ACC scores over the previous best-performing approach when trained on DFDC and tested on DefakeAVMiT and FakeAVCeleb datasets.
>
---
#### [new 098] Anti-Inpainting: A Proactive Defense against Malicious Diffusion-based Inpainters under Unknown Conditions
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于图像安全领域，提出了一种主动防御方法Anti-Inpainting，解决现有防御技术无法抵御未知条件下恶意扩散模型图像修复的问题。通过多级特征提取、多尺度语义保留增强和分布偏差优化三重机制，提升对抗扰动在未知篡改条件和随机种子下的防护效果，实验验证了其跨模型版本的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.13023v1](http://arxiv.org/pdf/2505.13023v1)**

> **作者:** Yimao Guo; Zuomin Qu; Wei Lu; Xiangyang Luo
>
> **摘要:** As diffusion-based malicious image manipulation becomes increasingly prevalent, multiple proactive defense methods are developed to safeguard images against unauthorized tampering. However, most proactive defense methods only can safeguard images against manipulation under known conditions, and fail to protect images from manipulations guided by tampering conditions crafted by malicious users. To tackle this issue, we propose Anti-Inpainting, a proactive defense method that achieves adequate protection under unknown conditions through a triple mechanism to address this challenge. Specifically, a multi-level deep feature extractor is presented to obtain intricate features during the diffusion denoising process to improve protective effectiveness. We design multi-scale semantic-preserving data augmentation to enhance the transferability of adversarial perturbations across unknown conditions by multi-scale transformations while preserving semantic integrity. In addition, we propose a selection-based distribution deviation optimization strategy to improve the protection of adversarial perturbation against manipulation under diverse random seeds. Extensive experiments indicate the proactive defensive performance of Anti-Inpainting against diffusion-based inpainters guided by unknown conditions in InpaintGuardBench and CelebA-HQ. At the same time, we also demonstrate the proposed approach's robustness under various image purification methods and its transferability across different versions of diffusion models.
>
---
#### [new 099] DragLoRA: Online Optimization of LoRA Adapters for Drag-based Image Editing in Diffusion Model
- **分类: cs.CV**

- **简介: 该论文研究基于拖拽的扩散模型图像编辑任务，解决传统方法特征表达能力弱、效率低的问题。提出DragLoRA框架，通过在线优化LoRA适配器增强运动监督特征，结合去噪分数蒸馏损失稳定训练，并设计自适应优化模式平衡精度与效率。**

- **链接: [http://arxiv.org/pdf/2505.12427v1](http://arxiv.org/pdf/2505.12427v1)**

> **作者:** Siwei Xia; Li Sun; Tiantian Sun; Qingli Li
>
> **备注:** Accepted by ICML2025
>
> **摘要:** Drag-based editing within pretrained diffusion model provides a precise and flexible way to manipulate foreground objects. Traditional methods optimize the input feature obtained from DDIM inversion directly, adjusting them iteratively to guide handle points towards target locations. However, these approaches often suffer from limited accuracy due to the low representation ability of the feature in motion supervision, as well as inefficiencies caused by the large search space required for point tracking. To address these limitations, we present DragLoRA, a novel framework that integrates LoRA (Low-Rank Adaptation) adapters into the drag-based editing pipeline. To enhance the training of LoRA adapters, we introduce an additional denoising score distillation loss which regularizes the online model by aligning its output with that of the original model. Additionally, we improve the consistency of motion supervision by adapting the input features using the updated LoRA, giving a more stable and accurate input feature for subsequent operations. Building on this, we design an adaptive optimization scheme that dynamically toggles between two modes, prioritizing efficiency without compromising precision. Extensive experiments demonstrate that DragLoRA significantly enhances the control precision and computational efficiency for drag-based image editing. The Codes of DragLoRA are available at: https://github.com/Sylvie-X/DragLoRA.
>
---
#### [new 100] Unlocking the Potential of Difficulty Prior in RL-based Multimodal Reasoning
- **分类: cs.CV**

- **简介: 该论文研究强化学习多模态推理中利用问题难度先验提升微调效果的方法。针对传统训练中简单/困难样本梯度无效的问题，提出三阶段方案：离线筛选U型难度分布数据，在线自适应优势加权优化学习信号，第二阶段引入难度提示校准推理深度。通过两阶段小规模训练，显著提升了多模态数学推理任务性能。**

- **链接: [http://arxiv.org/pdf/2505.13261v1](http://arxiv.org/pdf/2505.13261v1)**

> **作者:** Mingrui Chen; Haogeng Liu; Hao Liang; Huaibo Huang; Wentao Zhang; Ran He
>
> **摘要:** In this work, we investigate how explicitly modeling problem's difficulty prior information shapes the effectiveness of reinforcement learning based fine-tuning for multimodal reasoning. Our exploration mainly comprises of following three perspective: First, through offline data curation, we analyze the U-shaped difficulty distribution of two given datasets using the base model by multi-round sampling, and then filter out prompts that are either too simple or extremely difficult to provide meaningful gradients and perform subsequent two-stage training. Second, we implement an online advantage differentiation, computing group-wise empirical accuracy as a difficulty proxy to adaptively reweight advantages estimation, providing stronger learning signals for more challenging problems. Finally, we introduce difficulty hints as explicit prompts for more complex samples in the second training stage, encouraging the model to calibrate its reasoning depth and perform reflective validation checks. Our comprehensive approach demonstrates significant performances across various multi-modal mathematical reasoning benchmarks with only 2K+0.6K two-stage training data.
>
---
#### [new 101] Rebalancing Contrastive Alignment with Learnable Semantic Gaps in Text-Video Retrieval
- **分类: cs.CV; cs.IR; cs.MM**

- **简介: 该论文针对文本-视频检索任务，解决对比学习中模态分布差异和采样假阴性引发的梯度冲突问题，提出GARE框架，通过可学习的语义间隙增量Δ_ij分离优化矛盾，结合梯度监督模块和正则化策略优化对齐稳定性，实验验证了其在噪声数据下的鲁棒性提升。**

- **链接: [http://arxiv.org/pdf/2505.12499v1](http://arxiv.org/pdf/2505.12499v1)**

> **作者:** Jian Xiao; Zijie Song; Jialong Hu; Hao Cheng; Zhenzhen Hu; Jia Li; Richang Hong
>
> **摘要:** Recent advances in text-video retrieval have been largely driven by contrastive learning frameworks. However, existing methods overlook a key source of optimization tension: the separation between text and video distributions in the representation space (referred to as the modality gap), and the prevalence of false negatives in batch sampling. These factors lead to conflicting gradients under the InfoNCE loss, impeding stable alignment. To mitigate this, we propose GARE, a Gap-Aware Retrieval framework that introduces a learnable, pair-specific increment Delta_ij between text t_i and video v_j to offload the tension from the global anchor representation. We first derive the ideal form of Delta_ij via a coupled multivariate first-order Taylor approximation of the InfoNCE loss under a trust-region constraint, revealing it as a mechanism for resolving gradient conflicts by guiding updates along a locally optimal descent direction. Due to the high cost of directly computing Delta_ij, we introduce a lightweight neural module conditioned on the semantic gap between each video-text pair, enabling structure-aware correction guided by gradient supervision. To further stabilize learning and promote interpretability, we regularize Delta using three components: a trust-region constraint to prevent oscillation, a directional diversity term to promote semantic coverage, and an information bottleneck to limit redundancy. Experiments across four retrieval benchmarks show that GARE consistently improves alignment accuracy and robustness to noisy supervision, confirming the effectiveness of gap-aware tension mitigation.
>
---
#### [new 102] The Way Up: A Dataset for Hold Usage Detection in Sport Climbing
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决攀岩运动缺乏标注数据集和抓握动作检测难题。作者构建了首个含22段标注视频的攀岩数据集（记录抓握位置、时序），并基于2D姿态估计模型分析关节关键点与岩点的空间重叠关系，评估模型性能，为AI辅助攀岩系统奠定基础。**

- **链接: [http://arxiv.org/pdf/2505.12854v1](http://arxiv.org/pdf/2505.12854v1)**

> **作者:** Anna Maschek; David C. Schedl
>
> **备注:** accepted at the International Workshop on Computer Vision in Sports (CVsports) at CVPR 2025
>
> **摘要:** Detecting an athlete's position on a route and identifying hold usage are crucial in various climbing-related applications. However, no climbing dataset with detailed hold usage annotations exists to our knowledge. To address this issue, we introduce a dataset of 22 annotated climbing videos, providing ground-truth labels for hold locations, usage order, and time of use. Furthermore, we explore the application of keypoint-based 2D pose-estimation models for detecting hold usage in sport climbing. We determine usage by analyzing the key points of certain joints and the corresponding overlap with climbing holds. We evaluate multiple state-of-the-art models and analyze their accuracy on our dataset, identifying and highlighting climbing-specific challenges. Our dataset and results highlight key challenges in climbing-specific pose estimation and establish a foundation for future research toward AI-assisted systems for sports climbing.
>
---
#### [new 103] From Shots to Stories: LLM-Assisted Video Editing with Unified Language Representations
- **分类: cs.CV**

- **简介: 该论文研究LLM辅助视频编辑，属于多模态智能任务，旨在解决视觉信息与语言推理的衔接及发散性任务输出不稳定问题。提出L-Storyboard语言表征框架统一描述视频镜头，通过StoryFlow策略将发散推理转化为收敛选择机制，提升属性分类、镜头选择和排序三类核心任务的逻辑性与稳定性。**

- **链接: [http://arxiv.org/pdf/2505.12237v1](http://arxiv.org/pdf/2505.12237v1)**

> **作者:** Yuzhi Li; Haojun Xu; Fang Tian
>
> **摘要:** Large Language Models (LLMs) and Vision-Language Models (VLMs) have demonstrated remarkable reasoning and generalization capabilities in video understanding; however, their application in video editing remains largely underexplored. This paper presents the first systematic study of LLMs in the context of video editing. To bridge the gap between visual information and language-based reasoning, we introduce L-Storyboard, an intermediate representation that transforms discrete video shots into structured language descriptions suitable for LLM processing. We categorize video editing tasks into Convergent Tasks and Divergent Tasks, focusing on three core tasks: Shot Attributes Classification, Next Shot Selection, and Shot Sequence Ordering. To address the inherent instability of divergent task outputs, we propose the StoryFlow strategy, which converts the divergent multi-path reasoning process into a convergent selection mechanism, effectively enhancing task accuracy and logical coherence. Experimental results demonstrate that L-Storyboard facilitates a more robust mapping between visual information and language descriptions, significantly improving the interpretability and privacy protection of video editing tasks. Furthermore, StoryFlow enhances the logical consistency and output stability in Shot Sequence Ordering, underscoring the substantial potential of LLMs in intelligent video editing.
>
---
#### [new 104] FlowCut: Unsupervised Video Instance Segmentation via Temporal Mask Matching
- **分类: cs.CV**

- **简介: 该论文提出FlowCut，解决无监督视频实例分割任务，旨在无需标注生成高质量伪标签。方法分三阶段：利用图像和光流特征生成伪实例掩模；通过时序匹配构建一致掩模的短视频段；基于YouTubeVIS-2021训练模型。在多个基准测试中达到最优性能。**

- **链接: [http://arxiv.org/pdf/2505.13174v1](http://arxiv.org/pdf/2505.13174v1)**

> **作者:** Alp Eren Sari; Paolo Favaro
>
> **摘要:** We propose FlowCut, a simple and capable method for unsupervised video instance segmentation consisting of a three-stage framework to construct a high-quality video dataset with pseudo labels. To our knowledge, our work is the first attempt to curate a video dataset with pseudo-labels for unsupervised video instance segmentation. In the first stage, we generate pseudo-instance masks by exploiting the affinities of features from both images and optical flows. In the second stage, we construct short video segments containing high-quality, consistent pseudo-instance masks by temporally matching them across the frames. In the third stage, we use the YouTubeVIS-2021 video dataset to extract our training instance segmentation set, and then train a video segmentation model. FlowCut achieves state-of-the-art performance on the YouTubeVIS-2019, YouTubeVIS-2021, DAVIS-2017, and DAVIS-2017 Motion benchmarks.
>
---
#### [new 105] CompBench: Benchmarking Complex Instruction-guided Image Editing
- **分类: cs.CV**

- **简介: 该论文提出CompBench基准测试，针对复杂指令引导的图像编辑任务，解决现有基准任务简单、指令不细的问题。通过构建包含细粒度指令、空间与上下文推理的挑战性场景，并设计协作框架和指令解耦策略（分解为位置、外观、动态、对象四维度），全面评估模型精确编辑能力，揭示当前模型局限。**

- **链接: [http://arxiv.org/pdf/2505.12200v1](http://arxiv.org/pdf/2505.12200v1)**

> **作者:** Bohan Jia; Wenxuan Huang; Yuntian Tang; Junbo Qiao; Jincheng Liao; Shaosheng Cao; Fei Zhao; Zhaopeng Feng; Zhouhong Gu; Zhenfei Yin; Lei Bai; Wanli Ouyang; Lin Chen; Fei Zhao; Zihan Wang; Yuan Xie; Shaohui Lin
>
> **摘要:** While real-world applications increasingly demand intricate scene manipulation, existing instruction-guided image editing benchmarks often oversimplify task complexity and lack comprehensive, fine-grained instructions. To bridge this gap, we introduce, a large-scale benchmark specifically designed for complex instruction-guided image editing. CompBench features challenging editing scenarios that incorporate fine-grained instruction following, spatial and contextual reasoning, thereby enabling comprehensive evaluation of image editing models' precise manipulation capabilities. To construct CompBench, We propose an MLLM-human collaborative framework with tailored task pipelines. Furthermore, we propose an instruction decoupling strategy that disentangles editing intents into four key dimensions: location, appearance, dynamics, and objects, ensuring closer alignment between instructions and complex editing requirements. Extensive evaluations reveal that CompBench exposes fundamental limitations of current image editing models and provides critical insights for the development of next-generation instruction-guided image editing systems.
>
---
#### [new 106] Computer Vision Models Show Human-Like Sensitivity to Geometric and Topological Concepts
- **分类: cs.CV**

- **简介: 该论文属于认知科学与机器学习的交叉研究，探讨计算机视觉模型是否类人地感知几何拓扑(GT)概念。通过43个GT概念的"找不同"任务测试三类模型，发现Transformer模型准确率超儿童且与人类认知模式高度对齐，而视觉-语言模型因多模态干扰降低几何敏感性，为"概念通过环境交互习得"假说提供验证依据。**

- **链接: [http://arxiv.org/pdf/2505.13281v1](http://arxiv.org/pdf/2505.13281v1)**

> **作者:** Zekun Wang; Sashank Varma
>
> **备注:** 10 pages, 4 figures, CosSci 2025
>
> **摘要:** With the rapid improvement of machine learning (ML) models, cognitive scientists are increasingly asking about their alignment with how humans think. Here, we ask this question for computer vision models and human sensitivity to geometric and topological (GT) concepts. Under the core knowledge account, these concepts are innate and supported by dedicated neural circuitry. In this work, we investigate an alternative explanation, that GT concepts are learned ``for free'' through everyday interaction with the environment. We do so using computer visions models, which are trained on large image datasets. We build on prior studies to investigate the overall performance and human alignment of three classes of models -- convolutional neural networks (CNNs), transformer-based models, and vision-language models -- on an odd-one-out task testing 43 GT concepts spanning seven classes. Transformer-based models achieve the highest overall accuracy, surpassing that of young children. They also show strong alignment with children's performance, finding the same classes of concepts easy vs. difficult. By contrast, vision-language models underperform their vision-only counterparts and deviate further from human profiles, indicating that na\"ive multimodality might compromise abstract geometric sensitivity. These findings support the use of computer vision models to evaluate the sufficiency of the learning account for explaining human sensitivity to GT concepts, while also suggesting that integrating linguistic and visual representations might have unpredicted deleterious consequences.
>
---
#### [new 107] Robust Multimodal Segmentation with Representation Regularization and Hybrid Prototype Distillation
- **分类: cs.CV**

- **简介: 该论文研究多模态语义分割任务，旨在解决实际场景中因模态缺失/噪声导致模型性能下降的问题。提出RobustSeg两阶段框架：先预训练完整模态教师模型，再通过混合原型蒸馏（跨模态知识迁移）和表示正则化（优化特征差异）训练抗干扰学生模型，在三个基准上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.12861v1](http://arxiv.org/pdf/2505.12861v1)**

> **作者:** Jiaqi Tan; Xu Zheng; Yang Liu
>
> **摘要:** Multi-modal semantic segmentation (MMSS) faces significant challenges in real-world scenarios due to dynamic environments, sensor failures, and noise interference, creating a gap between theoretical models and practical performance. To address this, we propose a two-stage framework called RobustSeg, which enhances multi-modal robustness through two key components: the Hybrid Prototype Distillation Module (HPDM) and the Representation Regularization Module (RRM). In the first stage, RobustSeg pre-trains a multi-modal teacher model using complete modalities. In the second stage, a student model is trained with random modality dropout while learning from the teacher via HPDM and RRM. HPDM transforms features into compact prototypes, enabling cross-modal hybrid knowledge distillation and mitigating bias from missing modalities. RRM reduces representation discrepancies between the teacher and student by optimizing functional entropy through the log-Sobolev inequality. Extensive experiments on three public benchmarks demonstrate that RobustSeg outperforms previous state-of-the-art methods, achieving improvements of +2.76%, +4.56%, and +0.98%, respectively. Code is available at: https://github.com/RobustSeg/RobustSeg.
>
---
#### [new 108] LatentINDIGO: An INN-Guided Latent Diffusion Algorithm for Image Restoration
- **分类: cs.CV**

- **简介: 该论文研究图像恢复任务，旨在解决现有潜在扩散模型依赖预定义退化模型、潜在空间引导不稳定及计算成本高的问题。提出基于小波启发的可逆神经网络（INN），通过正/逆变换模拟退化与恢复细节，并设计两种潜在扩散框架（PixelINN/LatentINN）交替更新潜在变量与优化退化模型，结合正则化保持图像流形特性，实现复杂退化下的高效复原。**

- **链接: [http://arxiv.org/pdf/2505.12935v1](http://arxiv.org/pdf/2505.12935v1)**

> **作者:** Di You; Daniel Siromani; Pier Luigi Dragotti
>
> **备注:** Submitted to IEEE Transactions on Image Processing (TIP)
>
> **摘要:** There is a growing interest in the use of latent diffusion models (LDMs) for image restoration (IR) tasks due to their ability to model effectively the distribution of natural images. While significant progress has been made, there are still key challenges that need to be addressed. First, many approaches depend on a predefined degradation operator, making them ill-suited for complex or unknown degradations that deviate from standard analytical models. Second, many methods struggle to provide a stable guidance in the latent space and finally most methods convert latent representations back to the pixel domain for guidance at every sampling iteration, which significantly increases computational and memory overhead. To overcome these limitations, we introduce a wavelet-inspired invertible neural network (INN) that simulates degradations through a forward transform and reconstructs lost details via the inverse transform. We further integrate this design into a latent diffusion pipeline through two proposed approaches: LatentINDIGO-PixelINN, which operates in the pixel domain, and LatentINDIGO-LatentINN, which stays fully in the latent space to reduce complexity. Both approaches alternate between updating intermediate latent variables under the guidance of our INN and refining the INN forward model to handle unknown degradations. In addition, a regularization step preserves the proximity of latent variables to the natural image manifold. Experiments demonstrate that our algorithm achieves state-of-the-art performance on synthetic and real-world low-quality images, and can be readily adapted to arbitrary output sizes.
>
---
#### [new 109] From Low Field to High Value: Robust Cortical Mapping from Low-Field MRI
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像处理任务，解决低场MRI（LF-MRI）皮质表面重建精度不足的问题。提出基于3D U-Net的机器学习方法，利用合成数据训练直接预测皮质距离场，结合几何优化提升拓扑准确性。验证显示3mm各向同性T2扫描结果与高场MRI高度吻合（表面积r=0.96），可移植性强，为便携低场MRI的脑结构分析提供了可行方案。**

- **链接: [http://arxiv.org/pdf/2505.12228v1](http://arxiv.org/pdf/2505.12228v1)**

> **作者:** Karthik Gopinath; Annabel Sorby-Adams; Jonathan W. Ramirez; Dina Zemlyanker; Jennifer Guo; David Hunt; Christine L. Mac Donald; C. Dirk Keene; Timothy Coalson; Matthew F. Glasser; David Van Essen; Matthew S. Rosen; Oula Puonti; W. Taylor Kimberly; Juan Eugenio Iglesias
>
> **备注:** 32 pages
>
> **摘要:** Three-dimensional reconstruction of cortical surfaces from MRI for morphometric analysis is fundamental for understanding brain structure. While high-field MRI (HF-MRI) is standard in research and clinical settings, its limited availability hinders widespread use. Low-field MRI (LF-MRI), particularly portable systems, offers a cost-effective and accessible alternative. However, existing cortical surface analysis tools are optimized for high-resolution HF-MRI and struggle with the lower signal-to-noise ratio and resolution of LF-MRI. In this work, we present a machine learning method for 3D reconstruction and analysis of portable LF-MRI across a range of contrasts and resolutions. Our method works "out of the box" without retraining. It uses a 3D U-Net trained on synthetic LF-MRI to predict signed distance functions of cortical surfaces, followed by geometric processing to ensure topological accuracy. We evaluate our method using paired HF/LF-MRI scans of the same subjects, showing that LF-MRI surface reconstruction accuracy depends on acquisition parameters, including contrast type (T1 vs T2), orientation (axial vs isotropic), and resolution. A 3mm isotropic T2-weighted scan acquired in under 4 minutes, yields strong agreement with HF-derived surfaces: surface area correlates at r=0.96, cortical parcellations reach Dice=0.98, and gray matter volume achieves r=0.93. Cortical thickness remains more challenging with correlations up to r=0.70, reflecting the difficulty of sub-mm precision with 3mm voxels. We further validate our method on challenging postmortem LF-MRI, demonstrating its robustness. Our method represents a step toward enabling cortical surface analysis on portable LF-MRI. Code is available at https://surfer.nmr.mgh.harvard.edu/fswiki/ReconAny
>
---
#### [new 110] SafeVid: Toward Safety Aligned Video Large Multimodal Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频多模态模型安全对齐任务，旨在解决静态安全策略在动态视频场景中失效的问题。提出SafeVid框架，通过文本描述桥接视频与文本安全规则，构建350K安全偏好数据集，采用DPO优化对齐模型，并建立评估基准。实验显示模型安全性能提升最高达42.39%，公开了数据集与系统方法。**

- **链接: [http://arxiv.org/pdf/2505.11926v1](http://arxiv.org/pdf/2505.11926v1)**

> **作者:** Yixu Wang; Jiaxin Song; Yifeng Gao; Xin Wang; Yang Yao; Yan Teng; Xingjun Ma; Yingchun Wang; Yu-Gang Jiang
>
> **摘要:** As Video Large Multimodal Models (VLMMs) rapidly advance, their inherent complexity introduces significant safety challenges, particularly the issue of mismatched generalization where static safety alignments fail to transfer to dynamic video contexts. We introduce SafeVid, a framework designed to instill video-specific safety principles in VLMMs. SafeVid uniquely transfers robust textual safety alignment capabilities to the video domain by employing detailed textual video descriptions as an interpretive bridge, facilitating LLM-based rule-driven safety reasoning. This is achieved through a closed-loop system comprising: 1) generation of SafeVid-350K, a novel 350,000-pair video-specific safety preference dataset; 2) targeted alignment of VLMMs using Direct Preference Optimization (DPO); and 3) comprehensive evaluation via our new SafeVidBench benchmark. Alignment with SafeVid-350K significantly enhances VLMM safety, with models like LLaVA-NeXT-Video demonstrating substantial improvements (e.g., up to 42.39%) on SafeVidBench. SafeVid provides critical resources and a structured approach, demonstrating that leveraging textual descriptions as a conduit for safety reasoning markedly improves the safety alignment of VLMMs. We have made SafeVid-350K dataset (https://huggingface.co/datasets/yxwang/SafeVid-350K) publicly available.
>
---
#### [new 111] It's not you, it's me -- Global urban visual perception varies across demographics and personalities
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于城市感知分析任务，旨在解决现有方法忽视人口统计与个性差异导致的感知偏差问题。通过全球街景调查构建SPECS数据集，分析不同人群对街道的感知差异，并验证机器学习模型与人类评价的偏差，强调需结合本地化感知优化城市规划。**

- **链接: [http://arxiv.org/pdf/2505.12758v1](http://arxiv.org/pdf/2505.12758v1)**

> **作者:** Matias Quintana; Youlong Gu; Xiucheng Liang; Yujun Hou; Koichi Ito; Yihan Zhu; Mahmoud Abdelrahman; Filip Biljecki
>
> **备注:** Under review
>
> **摘要:** Understanding people's preferences and needs is crucial for urban planning decisions, yet current approaches often combine them from multi-cultural and multi-city populations, obscuring important demographic differences and risking amplifying biases. We conducted a large-scale urban visual perception survey of streetscapes worldwide using street view imagery, examining how demographics -- including gender, age, income, education, race and ethnicity, and, for the first time, personality traits -- shape perceptions among 1,000 participants, with balanced demographics, from five countries and 45 nationalities. This dataset, introduced as Street Perception Evaluation Considering Socioeconomics (SPECS), exhibits statistically significant differences in perception scores in six traditionally used indicators (safe, lively, wealthy, beautiful, boring, and depressing) and four new ones we propose (live nearby, walk, cycle, green) among demographics and personalities. We revealed that location-based sentiments are carried over in people's preferences when comparing urban streetscapes with other cities. Further, we compared the perception scores based on where participants and streetscapes are from. We found that an off-the-shelf machine learning model trained on an existing global perception dataset tends to overestimate positive indicators and underestimate negative ones compared to human responses, suggesting that targeted intervention should consider locals' perception. Our study aspires to rectify the myopic treatment of street perception, which rarely considers demographics or personality traits.
>
---
#### [new 112] FIGhost: Fluorescent Ink-based Stealthy and Flexible Backdoor Attacks on Physical Traffic Sign Recognition
- **分类: cs.CV**

- **简介: 该论文针对交通标志识别系统，提出隐蔽物理后门攻击任务。解决现有攻击隐蔽性差、控制不灵活及忽略视觉大模型的问题。基于荧光墨水开发紫外激活的隐形触发器，结合涂鸦设计模拟算法增强鲁棒性，构建自动化样本生成方法实现三种攻击目标，实验验证其在物理环境中可绕过先进检测器和防御机制。**

- **链接: [http://arxiv.org/pdf/2505.12045v1](http://arxiv.org/pdf/2505.12045v1)**

> **作者:** Shuai Yuan; Guowen Xu; Hongwei Li; Rui Zhang; Xinyuan Qian; Wenbo Jiang; Hangcheng Cao; Qingchuan Zhao
>
> **摘要:** Traffic sign recognition (TSR) systems are crucial for autonomous driving but are vulnerable to backdoor attacks. Existing physical backdoor attacks either lack stealth, provide inflexible attack control, or ignore emerging Vision-Large-Language-Models (VLMs). In this paper, we introduce FIGhost, the first physical-world backdoor attack leveraging fluorescent ink as triggers. Fluorescent triggers are invisible under normal conditions and activated stealthily by ultraviolet light, providing superior stealthiness, flexibility, and untraceability. Inspired by real-world graffiti, we derive realistic trigger shapes and enhance their robustness via an interpolation-based fluorescence simulation algorithm. Furthermore, we develop an automated backdoor sample generation method to support three attack objectives. Extensive evaluations in the physical world demonstrate FIGhost's effectiveness against state-of-the-art detectors and VLMs, maintaining robustness under environmental variations and effectively evading existing defenses.
>
---
#### [new 113] ARIW-Framework: Adaptive Robust Iterative Watermarking Framework
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于数字图像版权保护任务，解决现有水印技术视觉质量低、鲁棒性差和泛化性弱的问题。提出ARIW框架：通过迭代优化编码器生成抗噪残差，结合并行优化策略和图像梯度调整嵌入强度，平衡水印质量与抗攻击能力。实验证明其在高视觉质量下实现卓越抗噪性和泛化性。**

- **链接: [http://arxiv.org/pdf/2505.13101v1](http://arxiv.org/pdf/2505.13101v1)**

> **作者:** Shaowu Wu; Liting Zeng; Wei Lu; Xiangyang Luo
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** With the rapid rise of large models, copyright protection for generated image content has become a critical security challenge. Although deep learning watermarking techniques offer an effective solution for digital image copyright protection, they still face limitations in terms of visual quality, robustness and generalization. To address these issues, this paper proposes an adaptive robust iterative watermarking framework (ARIW-Framework) that achieves high-quality watermarked images while maintaining exceptional robustness and generalization performance. Specifically, we introduce an iterative approach to optimize the encoder for generating robust residuals. The encoder incorporates noise layers and a decoder to compute robustness weights for residuals under various noise attacks. By employing a parallel optimization strategy, the framework enhances robustness against multiple types of noise attacks. Furthermore, we leverage image gradients to determine the embedding strength at each pixel location, significantly improving the visual quality of the watermarked images. Extensive experiments demonstrate that the proposed method achieves superior visual quality while exhibiting remarkable robustness and generalization against noise attacks.
>
---
#### [new 114] ViEEG: Hierarchical Neural Coding with Cross-Modal Progressive Enhancement for EEG-Based Visual Decoding
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文研究基于脑电图（EEG）的视觉解码任务，旨在解决现有方法忽略大脑视觉层级结构的问题。提出ViEEG框架，将视觉刺激分解为轮廓、对象和场景三个生物层级，通过三流编码器和跨模态注意力融合EEG特征，结合分层对比学习对齐CLIP嵌入，实现了跨被试零样本识别，性能超越基线45%以上。**

- **链接: [http://arxiv.org/pdf/2505.12408v1](http://arxiv.org/pdf/2505.12408v1)**

> **作者:** Minxu Liu; Donghai Guan; Chuhang Zheng; Chunwei Tian; Jie Wen; Qi Zhu
>
> **备注:** 24 pages, 18 figures
>
> **摘要:** Understanding and decoding brain activity into visual representations is a fundamental challenge at the intersection of neuroscience and artificial intelligence. While EEG-based visual decoding has shown promise due to its non-invasive, low-cost nature and millisecond-level temporal resolution, existing methods are limited by their reliance on flat neural representations that overlook the brain's inherent visual hierarchy. In this paper, we introduce ViEEG, a biologically inspired hierarchical EEG decoding framework that aligns with the Hubel-Wiesel theory of visual processing. ViEEG decomposes each visual stimulus into three biologically aligned components-contour, foreground object, and contextual scene-serving as anchors for a three-stream EEG encoder. These EEG features are progressively integrated via cross-attention routing, simulating cortical information flow from V1 to IT to the association cortex. We further adopt hierarchical contrastive learning to align EEG representations with CLIP embeddings, enabling zero-shot object recognition. Extensive experiments on the THINGS-EEG dataset demonstrate that ViEEG achieves state-of-the-art performance, with 40.9% Top-1 accuracy in subject-dependent and 22.9% Top-1 accuracy in cross-subject settings, surpassing existing methods by over 45%. Our framework not only advances the performance frontier but also sets a new paradigm for biologically grounded brain decoding in AI.
>
---
#### [new 115] VTBench: Evaluating Visual Tokenizers for Autoregressive Image Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉分词器评估任务，旨在解决现有基准无法独立衡量自回归图像生成中视觉分词器(VT)性能的问题。作者提出VTBench基准，通过图像重建、细节/文本保留三个核心任务系统评估VT质量，发现连续VAE优于离散VT，并分析了GPT-4o的生成潜力，推动开源VT发展。**

- **链接: [http://arxiv.org/pdf/2505.13439v1](http://arxiv.org/pdf/2505.13439v1)**

> **作者:** Huawei Lin; Tong Geng; Zhaozhuo Xu; Weijie Zhao
>
> **备注:** 24 pages, 13 figures, 3 tables
>
> **摘要:** Autoregressive (AR) models have recently shown strong performance in image generation, where a critical component is the visual tokenizer (VT) that maps continuous pixel inputs to discrete token sequences. The quality of the VT largely defines the upper bound of AR model performance. However, current discrete VTs fall significantly behind continuous variational autoencoders (VAEs), leading to degraded image reconstructions and poor preservation of details and text. Existing benchmarks focus on end-to-end generation quality, without isolating VT performance. To address this gap, we introduce VTBench, a comprehensive benchmark that systematically evaluates VTs across three core tasks: Image Reconstruction, Detail Preservation, and Text Preservation, and covers a diverse range of evaluation scenarios. We systematically assess state-of-the-art VTs using a set of metrics to evaluate the quality of reconstructed images. Our findings reveal that continuous VAEs produce superior visual representations compared to discrete VTs, particularly in retaining spatial structure and semantic detail. In contrast, the degraded representations produced by discrete VTs often lead to distorted reconstructions, loss of fine-grained textures, and failures in preserving text and object integrity. Furthermore, we conduct experiments on GPT-4o image generation and discuss its potential AR nature, offering new insights into the role of visual tokenization. We release our benchmark and codebase publicly to support further research and call on the community to develop strong, general-purpose open-source VTs.
>
---
#### [new 116] Rethinking Features-Fused-Pyramid-Neck for Object Detection
- **分类: cs.CV**

- **简介: 该论文针对目标检测中特征金字塔颈（FPN）因层级特征强制融合导致的错位问题，提出独立层次金字塔（IHP）架构，结合软最近邻插值（SNI）和自适应下采样方法（ESD），通过二次对齐方案（SA）优化多尺度检测，在Pascal VOC和COCO数据集实现SOTA，兼顾实时性与轻量化。**

- **链接: [http://arxiv.org/pdf/2505.12820v1](http://arxiv.org/pdf/2505.12820v1)**

> **作者:** Hulin Li
>
> **备注:** ECCV 2024
>
> **摘要:** Multi-head detectors typically employ a features-fused-pyramid-neck for multi-scale detection and are widely adopted in the industry. However, this approach faces feature misalignment when representations from different hierarchical levels of the feature pyramid are forcibly fused point-to-point. To address this issue, we designed an independent hierarchy pyramid (IHP) architecture to evaluate the effectiveness of the features-unfused-pyramid-neck for multi-head detectors. Subsequently, we introduced soft nearest neighbor interpolation (SNI) with a weight downscaling factor to mitigate the impact of feature fusion at different hierarchies while preserving key textures. Furthermore, we present a features adaptive selection method for down sampling in extended spatial windows (ESD) to retain spatial features and enhance lightweight convolutional techniques (GSConvE). These advancements culminate in our secondary features alignment solution (SA) for real-time detection, achieving state-of-the-art results on Pascal VOC and MS COCO. Code will be released at https://github.com/AlanLi1997/rethinking-fpn. This paper has been accepted by ECCV2024 and published on Springer Nature.
>
---
#### [new 117] VGGT-SLAM: Dense RGB SLAM Optimized on the SL(4) Manifold
- **分类: cs.CV**

- **简介: 该论文提出VGGT-SLAM系统，属于视觉SLAM任务，针对未校准单目相机场景重建的投影歧义问题。通过SL(4)流形优化15自由度单应变换对齐子地图，突破传统相似变换的限制，提升长视频序列下的三维重建质量，解决了原VGGT模型GPU资源过高的问题。**

- **链接: [http://arxiv.org/pdf/2505.12549v1](http://arxiv.org/pdf/2505.12549v1)**

> **作者:** Dominic Maggio; Hyungtae Lim; Luca Carlone
>
> **摘要:** We present VGGT-SLAM, a dense RGB SLAM system constructed by incrementally and globally aligning submaps created from the feed-forward scene reconstruction approach VGGT using only uncalibrated monocular cameras. While related works align submaps using similarity transforms (i.e., translation, rotation, and scale), we show that such approaches are inadequate in the case of uncalibrated cameras. In particular, we revisit the idea of reconstruction ambiguity, where given a set of uncalibrated cameras with no assumption on the camera motion or scene structure, the scene can only be reconstructed up to a 15-degrees-of-freedom projective transformation of the true geometry. This inspires us to recover a consistent scene reconstruction across submaps by optimizing over the SL(4) manifold, thus estimating 15-degrees-of-freedom homography transforms between sequential submaps while accounting for potential loop closure constraints. As verified by extensive experiments, we demonstrate that VGGT-SLAM achieves improved map quality using long video sequences that are infeasible for VGGT due to its high GPU requirements.
>
---
#### [new 118] Safe-Sora: Safe Text-to-Video Generation via Graphical Watermarking
- **分类: cs.CV**

- **简介: 该论文属于视频版权保护任务，解决AI生成视频的水印嵌入问题。提出Safe-Sora框架，通过分层自适应匹配机制将图形水印分割后嵌入相似视频区域，并设计3D小波增强的Mamba架构实现时空融合。首次将状态空间模型用于水印，提升视频质量与水印鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.12667v1](http://arxiv.org/pdf/2505.12667v1)**

> **作者:** Zihan Su; Xuerui Qiu; Hongbin Xu; Tangyu Jiang; Junhao Zhuang; Chun Yuan; Ming Li; Shengfeng He; Fei Richard Yu
>
> **摘要:** The explosive growth of generative video models has amplified the demand for reliable copyright preservation of AI-generated content. Despite its popularity in image synthesis, invisible generative watermarking remains largely underexplored in video generation. To address this gap, we propose Safe-Sora, the first framework to embed graphical watermarks directly into the video generation process. Motivated by the observation that watermarking performance is closely tied to the visual similarity between the watermark and cover content, we introduce a hierarchical coarse-to-fine adaptive matching mechanism. Specifically, the watermark image is divided into patches, each assigned to the most visually similar video frame, and further localized to the optimal spatial region for seamless embedding. To enable spatiotemporal fusion of watermark patches across video frames, we develop a 3D wavelet transform-enhanced Mamba architecture with a novel spatiotemporal local scanning strategy, effectively modeling long-range dependencies during watermark embedding and retrieval. To the best of our knowledge, this is the first attempt to apply state space models to watermarking, opening new avenues for efficient and robust watermark protection. Extensive experiments demonstrate that Safe-Sora achieves state-of-the-art performance in terms of video quality, watermark fidelity, and robustness, which is largely attributed to our proposals. We will release our code upon publication.
>
---
#### [new 119] Visuospatial Cognitive Assistant
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文属于视频空间理解与推理任务，旨在解决现有视觉语言模型在视频时空认知中的不足。提出了ViCA-322K数据集（含32万视频QA对）和ViCA-7B模型，在8项基准任务中刷新SOTA，并通过ViCA-Thinking-2.68K数据集增强模型可解释性，推动具身AI的时空建模研究。**

- **链接: [http://arxiv.org/pdf/2505.12312v1](http://arxiv.org/pdf/2505.12312v1)**

> **作者:** Qi Feng; Hidetoshi Shimodaira
>
> **备注:** 31 pages, 10 figures, 6 tables. The implementation and fine-tuned model (ViCA-7B) are publicly available at https://huggingface.co/nkkbr/ViCA. The ViCA-322K dataset can be found at https://huggingface.co/datasets/nkkbr/ViCA-322K, and the ViCA-Thinking-2.68K dataset is at https://huggingface.co/datasets/nkkbr/ViCA-thinking-2.68k
>
> **摘要:** Video-based spatial cognition is vital for robotics and embodied AI but challenges current Vision-Language Models (VLMs). This paper makes two key contributions. First, we introduce ViCA (Visuospatial Cognitive Assistant)-322K, a diverse dataset of 322,003 QA pairs from real-world indoor videos (ARKitScenes, ScanNet, ScanNet++), offering supervision for 3D metadata-grounded queries and video-based complex reasoning. Second, we develop ViCA-7B, fine-tuned on ViCA-322K, which achieves new state-of-the-art on all eight VSI-Bench tasks, outperforming existing models, including larger ones (e.g., +26.1 on Absolute Distance). For interpretability, we present ViCA-Thinking-2.68K, a dataset with explicit reasoning chains, and fine-tune ViCA-7B to create ViCA-7B-Thinking, a model that articulates its spatial reasoning. Our work highlights the importance of targeted data and suggests paths for improved temporal-spatial modeling. We release all resources to foster research in robust visuospatial intelligence.
>
---
#### [new 120] eStonefish-scenes: A synthetically generated dataset for underwater event-based optical flow prediction tasks
- **分类: cs.CV; I.2.5; I.2.6; I.2.9; I.2.10**

- **简介: 该论文针对水下事件光流预测任务，提出合成数据集eStonefish-scenes，解决真实数据多样性不足及水下标注数据缺失问题。基于Stonefish模拟器构建数据生成管道，模拟动态鱼群和珊瑚地形，并开发eWiz工具库支持数据处理与训练。**

- **链接: [http://arxiv.org/pdf/2505.13309v1](http://arxiv.org/pdf/2505.13309v1)**

> **作者:** Jad Mansour; Sebastian Realpe; Hayat Rajani; Michele Grimaldi; Rafael Garcia; Nuno Gracias
>
> **备注:** Submitted to IJRR
>
> **摘要:** The combined use of event-based vision and Spiking Neural Networks (SNNs) is expected to significantly impact robotics, particularly in tasks like visual odometry and obstacle avoidance. While existing real-world event-based datasets for optical flow prediction, typically captured with Unmanned Aerial Vehicles (UAVs), offer valuable insights, they are limited in diversity, scalability, and are challenging to collect. Moreover, there is a notable lack of labelled datasets for underwater applications, which hinders the integration of event-based vision with Autonomous Underwater Vehicles (AUVs). To address this, synthetic datasets could provide a scalable solution while bridging the gap between simulation and reality. In this work, we introduce eStonefish-scenes, a synthetic event-based optical flow dataset based on the Stonefish simulator. Along with the dataset, we present a data generation pipeline that enables the creation of customizable underwater environments. This pipeline allows for simulating dynamic scenarios, such as biologically inspired schools of fish exhibiting realistic motion patterns, including obstacle avoidance and reactive navigation around corals. Additionally, we introduce a scene generator that can build realistic reef seabeds by randomly distributing coral across the terrain. To streamline data accessibility, we present eWiz, a comprehensive library designed for processing event-based data, offering tools for data loading, augmentation, visualization, encoding, and training data generation, along with loss functions and performance metrics.
>
---
#### [new 121] RoPECraft: Training-Free Motion Transfer with Trajectory-Guided RoPE Optimization on Diffusion Transformers
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究视频运动迁移任务，旨在无需训练地将参考视频运动转移至扩散模型生成过程。通过光流提取运动轨迹，优化旋转位置嵌入（RoPE）编码运动信息，结合轨迹对齐和相位正则化保持文本一致性并抑制伪影，性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.13344v1](http://arxiv.org/pdf/2505.13344v1)**

> **作者:** Ahmet Berke Gokmen; Yigit Ekin; Bahri Batuhan Bilecen; Aysegul Dundar
>
> **备注:** https://berkegokmen1.github.io/RoPECraft/
>
> **摘要:** We propose RoPECraft, a training-free video motion transfer method for diffusion transformers that operates solely by modifying their rotary positional embeddings (RoPE). We first extract dense optical flow from a reference video, and utilize the resulting motion offsets to warp the complex-exponential tensors of RoPE, effectively encoding motion into the generation process. These embeddings are then further optimized during denoising time steps via trajectory alignment between the predicted and target velocities using a flow-matching objective. To keep the output faithful to the text prompt and prevent duplicate generations, we incorporate a regularization term based on the phase components of the reference video's Fourier transform, projecting the phase angles onto a smooth manifold to suppress high-frequency artifacts. Experiments on benchmarks reveal that RoPECraft outperforms all recently published methods, both qualitatively and quantitatively.
>
---
#### [new 122] Touch2Shape: Touch-Conditioned 3D Diffusion for Shape Exploration and Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D形状重建与探索任务，旨在解决现有扩散模型在局部细节捕捉及遮挡/光照限制下的不足。提出Touch2Shape模型，利用触觉图像构建触控条件扩散框架：通过嵌入模块生成紧凑表征，融合模块优化重建；结合扩散模型与强化学习设计触觉探索策略，提升重建性能。实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2505.13091v1](http://arxiv.org/pdf/2505.13091v1)**

> **作者:** Yuanbo Wang; Zhaoxuan Zhang; Jiajin Qiu; Dilong Sun; Zhengyu Meng; Xiaopeng Wei; Xin Yang
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Diffusion models have made breakthroughs in 3D generation tasks. Current 3D diffusion models focus on reconstructing target shape from images or a set of partial observations. While excelling in global context understanding, they struggle to capture the local details of complex shapes and limited to the occlusion and lighting conditions. To overcome these limitations, we utilize tactile images to capture the local 3D information and propose a Touch2Shape model, which leverages a touch-conditioned diffusion model to explore and reconstruct the target shape from touch. For shape reconstruction, we have developed a touch embedding module to condition the diffusion model in creating a compact representation and a touch shape fusion module to refine the reconstructed shape. For shape exploration, we combine the diffusion model with reinforcement learning to train a policy. This involves using the generated latent vector from the diffusion model to guide the touch exploration policy training through a novel reward design. Experiments validate the reconstruction quality thorough both qualitatively and quantitative analysis, and our touch exploration policy further boosts reconstruction performance.
>
---
#### [new 123] Improved Bag-of-Words Image Retrieval with Geometric Constraints for Ground Texture Localization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于地面纹理定位任务，旨在提升基于词袋模型（BoW）的图像检索精度。针对现有BoW系统在全局定位和SLAM闭环检测中精度不足的问题，提出结合软分配近似K均值词汇及地面纹理固有几何约束（方向/尺度）的改进算法，并通过高精度/高速双版本适配不同需求，实验验证其有效性，可无缝替换现有系统提升性能。**

- **链接: [http://arxiv.org/pdf/2505.11620v1](http://arxiv.org/pdf/2505.11620v1)**

> **作者:** Aaron Wilhelm; Nils Napp
>
> **备注:** Accepted to ICRA 2025
>
> **摘要:** Ground texture localization using a downward-facing camera offers a low-cost, high-precision localization solution that is robust to dynamic environments and requires no environmental modification. We present a significantly improved bag-of-words (BoW) image retrieval system for ground texture localization, achieving substantially higher accuracy for global localization and higher precision and recall for loop closure detection in SLAM. Our approach leverages an approximate $k$-means (AKM) vocabulary with soft assignment, and exploits the consistent orientation and constant scale constraints inherent to ground texture localization. Identifying the different needs of global localization vs. loop closure detection for SLAM, we present both high-accuracy and high-speed versions of our algorithm. We test the effect of each of our proposed improvements through an ablation study and demonstrate our method's effectiveness for both global localization and loop closure detection. With numerous ground texture localization systems already using BoW, our method can readily replace other generic BoW systems in their pipeline and immediately improve their results.
>
---
#### [new 124] MedSG-Bench: A Benchmark for Medical Image Sequences Grounding
- **分类: cs.CV**

- **简介: 该论文提出MedSG-Bench，针对医学图像序列的视觉定位任务，解决现有基准仅关注单图而忽略时序跨模态语义对齐的问题。通过构建包含8项任务的评测框架，覆盖76个数据集，并开发MedSG-188K调优数据集与MedSeq-Grounder模型，推动医学序列图像的细粒度理解研究。**

- **链接: [http://arxiv.org/pdf/2505.11852v1](http://arxiv.org/pdf/2505.11852v1)**

> **作者:** Jingkun Yue; Siqi Zhang; Zinan Jia; Huihuan Xu; Zongbo Han; Xiaohong Liu; Guangyu Wang
>
> **摘要:** Visual grounding is essential for precise perception and reasoning in multimodal large language models (MLLMs), especially in medical imaging domains. While existing medical visual grounding benchmarks primarily focus on single-image scenarios, real-world clinical applications often involve sequential images, where accurate lesion localization across different modalities and temporal tracking of disease progression (e.g., pre- vs. post-treatment comparison) require fine-grained cross-image semantic alignment and context-aware reasoning. To remedy the underrepresentation of image sequences in existing medical visual grounding benchmarks, we propose MedSG-Bench, the first benchmark tailored for Medical Image Sequences Grounding. It comprises eight VQA-style tasks, formulated into two paradigms of the grounding tasks, including 1) Image Difference Grounding, which focuses on detecting change regions across images, and 2) Image Consistency Grounding, which emphasizes detection of consistent or shared semantics across sequential images. MedSG-Bench covers 76 public datasets, 10 medical imaging modalities, and a wide spectrum of anatomical structures and diseases, totaling 9,630 question-answer pairs. We benchmark both general-purpose MLLMs (e.g., Qwen2.5-VL) and medical-domain specialized MLLMs (e.g., HuatuoGPT-vision), observing that even the advanced models exhibit substantial limitations in medical sequential grounding tasks. To advance this field, we construct MedSG-188K, a large-scale instruction-tuning dataset tailored for sequential visual grounding, and further develop MedSeq-Grounder, an MLLM designed to facilitate future research on fine-grained understanding across medical sequential images. The benchmark, dataset, and model are available at https://huggingface.co/MedSG-Bench
>
---
#### [new 125] FLASH: Latent-Aware Semi-Autoregressive Speculative Decoding for Multimodal Tasks
- **分类: cs.CV; cs.MM**

- **简介: 该论文针对多模态模型推理速度慢的问题，提出FLASH框架加速解码。通过视觉潜在特征压缩减少冗余，结合半自回归解码生成多视觉token，在保持质量的同时实现2.68倍加速，适用于视频描述等任务。**

- **链接: [http://arxiv.org/pdf/2505.12728v1](http://arxiv.org/pdf/2505.12728v1)**

> **作者:** Zihua Wang; Ruibo Li; Haozhe Du; Joey Tianyi Zhou; Yu Zhang; Xu Yang
>
> **摘要:** Large language and multimodal models (LLMs and LMMs) exhibit strong inference capabilities but are often limited by slow decoding speeds. This challenge is especially acute in LMMs, where visual inputs typically comprise more tokens with lower information density than text -- an issue exacerbated by recent trends toward finer-grained visual tokenizations to boost performance. Speculative decoding has been effective in accelerating LLM inference by using a smaller draft model to generate candidate tokens, which are then selectively verified by the target model, improving speed without sacrificing output quality. While this strategy has been extended to LMMs, existing methods largely overlook the unique properties of visual inputs and depend solely on text-based draft models. In this work, we propose \textbf{FLASH} (Fast Latent-Aware Semi-Autoregressive Heuristics), a speculative decoding framework designed specifically for LMMs, which leverages two key properties of multimodal data to design the draft model. First, to address redundancy in visual tokens, we propose a lightweight latent-aware token compression mechanism. Second, recognizing that visual objects often co-occur within a scene, we employ a semi-autoregressive decoding strategy to generate multiple tokens per forward pass. These innovations accelerate draft decoding while maintaining high acceptance rates, resulting in faster overall inference. Experiments show that FLASH significantly outperforms prior speculative decoding approaches in both unimodal and multimodal settings, achieving up to \textbf{2.68$\times$} speed-up on video captioning and \textbf{2.55$\times$} on visual instruction tuning tasks compared to the original LMM.
>
---
#### [new 126] RB-SCD: A New Benchmark for Semantic Change Detection of Roads and Bridges in Traffic Scenes
- **分类: cs.CV**

- **简介: 该论文属于语义变化检测任务，旨在解决交通场景中道路桥梁细粒度语义变化识别困难及数据集缺乏问题。提出了RB-SCD数据集（含260对遥感图像，覆盖11类变化），并设计多模态频域驱动框架MFDCD，通过动态频率耦合器和文本频率滤波器实现特征融合，在多个基准测试中验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.13212v1](http://arxiv.org/pdf/2505.13212v1)**

> **作者:** Qingling Shu; Sibao Chen; Zhihui You; Wei Lu; Jin Tang; Bin Luo
>
> **摘要:** Accurate detection of changes in roads and bridges, such as construction, renovation, and demolition, is essential for urban planning and traffic management. However, existing methods often struggle to extract fine-grained semantic change information due to the lack of high-quality annotated datasets in traffic scenarios. To address this, we introduce the Road and Bridge Semantic Change Detection (RB-SCD) dataset, a comprehensive benchmark comprising 260 pairs of high-resolution remote sensing images from diverse cities and countries. RB-SCD captures 11 types of semantic changes across varied road and bridge structures, enabling detailed structural and functional analysis. Building on this dataset, we propose a novel framework, Multimodal Frequency-Driven Change Detector (MFDCD), which integrates multimodal features in the frequency domain. MFDCD includes a Dynamic Frequency Coupler (DFC) that fuses hierarchical visual features with wavelet-based frequency components, and a Textual Frequency Filter (TFF) that transforms CLIP-derived textual features into the frequency domain and applies graph-based filtering. Experimental results on RB-SCD and three public benchmarks demonstrate the effectiveness of our approach.
>
---
#### [new 127] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态安全评估任务，旨在解决视频大视觉语言模型(LVLMs)在动态视频诱导攻击下的安全隐患。通过构建含2,264个视频-文本对的Video-SafetyBench基准，结合可控视频合成方法和新型评估指标RJScore，揭示了模型对良性查询视频攻击67.2%的平均漏洞率。**

- **链接: [http://arxiv.org/pdf/2505.11842v1](http://arxiv.org/pdf/2505.11842v1)**

> **作者:** Xuannan Liu; Zekun Li; Zheqi He; Peipei Li; Shuhan Xia; Xing Cui; Huaibo Huang; Xi Yang; Ran He
>
> **备注:** Project page: https://liuxuannan.github.io/Video-SafetyBench.github.io/
>
> **摘要:** The increasing deployment of Large Vision-Language Models (LVLMs) raises safety concerns under potential malicious inputs. However, existing multimodal safety evaluations primarily focus on model vulnerabilities exposed by static image inputs, ignoring the temporal dynamics of video that may induce distinct safety risks. To bridge this gap, we introduce Video-SafetyBench, the first comprehensive benchmark designed to evaluate the safety of LVLMs under video-text attacks. It comprises 2,264 video-text pairs spanning 48 fine-grained unsafe categories, each pairing a synthesized video with either a harmful query, which contains explicit malice, or a benign query, which appears harmless but triggers harmful behavior when interpreted alongside the video. To generate semantically accurate videos for safety evaluation, we design a controllable pipeline that decomposes video semantics into subject images (what is shown) and motion text (how it moves), which jointly guide the synthesis of query-relevant videos. To effectively evaluate uncertain or borderline harmful outputs, we propose RJScore, a novel LLM-based metric that incorporates the confidence of judge models and human-aligned decision threshold calibration. Extensive experiments show that benign-query video composition achieves average attack success rates of 67.2%, revealing consistent vulnerabilities to video-induced attacks. We believe Video-SafetyBench will catalyze future research into video-based safety evaluation and defense strategies.
>
---
#### [new 128] Cross-Model Transfer of Task Vectors via Few-Shot Orthogonal Alignment
- **分类: cs.CV**

- **简介: 该论文属于迁移学习领域，旨在解决不同预训练模型间任务向量迁移的适配问题。针对任务算术方法在异构模型间失效的局限，提出基于少样本正交对齐的跨模型参数空间适配方法，保持任务向量特性。实验表明，该方法在视觉分类任务中优于直接迁移，性能接近少样本微调，同时保留任务向量的模块化优势。**

- **链接: [http://arxiv.org/pdf/2505.12021v1](http://arxiv.org/pdf/2505.12021v1)**

> **作者:** Kazuhiko Kawamoto; Atsuhiro Endo; Hiroshi Kera
>
> **备注:** 8 pages
>
> **摘要:** Task arithmetic enables efficient model editing by representing task-specific changes as vectors in parameter space. Task arithmetic typically assumes that the source and target models are initialized from the same pre-trained parameters. This assumption limits its applicability in cross-model transfer settings, where models are independently pre-trained on different datasets. To address this challenge, we propose a method based on few-shot orthogonal alignment, which aligns task vectors to the parameter space of a differently pre-trained target model. These transformations preserve key properties of task vectors, such as norm and rank, and are learned using only a small number of labeled examples. We evaluate the method using two Vision Transformers pre-trained on YFCC100M and LAION400M, and test on eight classification datasets. Experimental results show that our method improves transfer accuracy over direct task vector application and achieves performance comparable to few-shot fine-tuning, while maintaining the modularity and reusability of task vectors. Our code is available at https://github.com/kawakera-lab/CrossModelTransfer.
>
---
#### [new 129] UniMoCo: Unified Modality Completion for Robust Multi-Modal Embeddings
- **分类: cs.CV**

- **简介: 该论文针对多模态嵌入任务中模态组合多样导致的模型性能下降问题，提出UniMoCo框架。通过模态补全模块生成缺失的视觉特征，结合对齐训练策略统一嵌入空间，解决传统方法因模态不平衡引发的偏差，提升跨模态任务的鲁棒性和效果。**

- **链接: [http://arxiv.org/pdf/2505.11815v1](http://arxiv.org/pdf/2505.11815v1)**

> **作者:** Jiajun Qin; Yuan Pu; Zhuolun He; Seunggeun Kim; David Z. Pan; Bei Yu
>
> **摘要:** Current research has explored vision-language models for multi-modal embedding tasks, such as information retrieval, visual grounding, and classification. However, real-world scenarios often involve diverse modality combinations between queries and targets, such as text and image to text, text and image to text and image, and text to text and image. These diverse combinations pose significant challenges for existing models, as they struggle to align all modality combinations within a unified embedding space during training, which degrades performance at inference. To address this limitation, we propose UniMoCo, a novel vision-language model architecture designed for multi-modal embedding tasks. UniMoCo introduces a modality-completion module that generates visual features from textual inputs, ensuring modality completeness for both queries and targets. Additionally, we develop a specialized training strategy to align embeddings from both original and modality-completed inputs, ensuring consistency within the embedding space. This enables the model to robustly handle a wide range of modality combinations across embedding tasks. Experiments show that UniMoCo outperforms previous methods while demonstrating consistent robustness across diverse settings. More importantly, we identify and quantify the inherent bias in conventional approaches caused by imbalance of modality combinations in training data, which can be mitigated through our modality-completion paradigm. The code is available at https://github.com/HobbitQia/UniMoCo.
>
---
#### [new 130] Single Image Reflection Removal via inter-layer Complementarity
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对单图像反射去除任务，解决双流架构层间互补性不足导致的分离质量受限问题。提出两种改进：设计层间互补模型，利用残差层低频/高频成分增强透射层细节；构建互补注意力机制，通过通道重组与注意力计算优化层间分离。方法在多个数据集达到SOTA，同时降低计算成本。**

- **链接: [http://arxiv.org/pdf/2505.12641v1](http://arxiv.org/pdf/2505.12641v1)**

> **作者:** Yue Huang; Zi'ang Li; Tianle Hu; Jie Wen; Guanbin Li; Jinglin Zhang; Guoxu Zhou; Xiaozhao Fang
>
> **摘要:** Although dual-stream architectures have achieved remarkable success in single image reflection removal, they fail to fully exploit inter-layer complementarity in their physical modeling and network design, which limits the quality of image separation. To address this fundamental limitation, we propose two targeted improvements to enhance dual-stream architectures: First, we introduce a novel inter-layer complementarity model where low-frequency components extracted from the residual layer interact with the transmission layer through dual-stream architecture to enhance inter-layer complementarity. Meanwhile, high-frequency components from the residual layer provide inverse modulation to both streams, improving the detail quality of the transmission layer. Second, we propose an efficient inter-layer complementarity attention mechanism which first cross-reorganizes dual streams at the channel level to obtain reorganized streams with inter-layer complementary structures, then performs attention computation on the reorganized streams to achieve better inter-layer separation, and finally restores the original stream structure for output. Experimental results demonstrate that our method achieves state-of-the-art separation quality on multiple public datasets while significantly reducing both computational cost and model complexity.
>
---
#### [new 131] A Generalized Label Shift Perspective for Cross-Domain Gaze Estimation
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究跨域视线估计（CDGE）任务，旨在解决模型在新目标域泛化时的域适应问题。现有方法依赖特征对齐，但忽略标签偏移。本文提出广义标签偏移视角，将问题建模为标签与条件偏移，设计基于截断高斯分布的重加权策略校正标签偏移，并推导概率感知条件差异估计方法。实验验证了方法的跨域泛化能力及模型普适性。**

- **链接: [http://arxiv.org/pdf/2505.13043v1](http://arxiv.org/pdf/2505.13043v1)**

> **作者:** Hao-Ran Yang; Xiaohui Chen; Chuan-Xian Ren
>
> **摘要:** Aiming to generalize the well-trained gaze estimation model to new target domains, Cross-domain Gaze Estimation (CDGE) is developed for real-world application scenarios. Existing CDGE methods typically extract the domain-invariant features to mitigate domain shift in feature space, which is proved insufficient by Generalized Label Shift (GLS) theory. In this paper, we introduce a novel GLS perspective to CDGE and modelize the cross-domain problem by label and conditional shift problem. A GLS correction framework is presented and a feasible realization is proposed, in which a importance reweighting strategy based on truncated Gaussian distribution is introduced to overcome the continuity challenges in label shift correction. To embed the reweighted source distribution to conditional invariant learning, we further derive a probability-aware estimation of conditional operator discrepancy. Extensive experiments on standard CDGE tasks with different backbone models validate the superior generalization capability across domain and applicability on various models of proposed method.
>
---
#### [new 132] Guiding Diffusion with Deep Geometric Moments: Balancing Fidelity and Variation
- **分类: cs.CV**

- **简介: 该论文研究文本到图像生成任务，针对现有空间引导方法（如分割图）限制生成多样性的问题，提出深度几何矩（DGM）作为新引导机制。DGM通过几何先验捕捉主体特征，相比CLIP/DINO等特征更聚焦局部细节且抗像素扰动，在扩散模型中平衡生成控制与多样性。**

- **链接: [http://arxiv.org/pdf/2505.12486v1](http://arxiv.org/pdf/2505.12486v1)**

> **作者:** Sangmin Jung; Utkarsh Nath; Yezhou Yang; Giulia Pedrielli; Joydeep Biswas; Amy Zhang; Hassan Ghasemzadeh; Pavan Turaga
>
> **备注:** Accepted in CVPR Workshop GMCV 2025
>
> **摘要:** Text-to-image generation models have achieved remarkable capabilities in synthesizing images, but often struggle to provide fine-grained control over the output. Existing guidance approaches, such as segmentation maps and depth maps, introduce spatial rigidity that restricts the inherent diversity of diffusion models. In this work, we introduce Deep Geometric Moments (DGM) as a novel form of guidance that encapsulates the subject's visual features and nuances through a learned geometric prior. DGMs focus specifically on the subject itself compared to DINO or CLIP features, which suffer from overemphasis on global image features or semantics. Unlike ResNets, which are sensitive to pixel-wise perturbations, DGMs rely on robust geometric moments. Our experiments demonstrate that DGM effectively balance control and diversity in diffusion-based image generation, allowing a flexible control mechanism for steering the diffusion process.
>
---
#### [new 133] SRLoRA: Subspace Recomposition in Low-Rank Adaptation via Importance-Based Fusion and Reinitialization
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于参数高效微调任务，针对LoRA方法因固定低秩子空间限制模型表达能力的问题，提出SRLoRA方法。通过动态评估低秩矩阵对的重要性，融合次要参数至主干网络，并沿未使用的主方向重新初始化新参数，实现子空间动态重组，提升模型性能而不增加参数量。实验证明其在语言和视觉任务中优于标准LoRA。**

- **链接: [http://arxiv.org/pdf/2505.12433v1](http://arxiv.org/pdf/2505.12433v1)**

> **作者:** Haodong Yang; Lei Wang; Md Zakir Hossain
>
> **备注:** Research report
>
> **摘要:** Low-Rank Adaptation (LoRA) is a widely adopted parameter-efficient fine-tuning (PEFT) method that injects two trainable low-rank matrices (A and B) into frozen pretrained models. While efficient, LoRA constrains updates to a fixed low-rank subspace (Delta W = BA), which can limit representational capacity and hinder downstream performance. We introduce Subspace Recomposition in Low-Rank Adaptation (SRLoRA) via importance-based fusion and reinitialization, a novel approach that enhances LoRA's expressiveness without compromising its lightweight structure. SRLoRA assigns importance scores to each LoRA pair (a column of B and the corresponding row of A), and dynamically recomposes the subspace during training. Less important pairs are fused into the frozen backbone, freeing capacity to reinitialize new pairs along unused principal directions derived from the pretrained weight's singular value decomposition. This mechanism enables continual subspace refreshment and richer adaptation over time, without increasing the number of trainable parameters. We evaluate SRLoRA on both language and vision tasks, including the GLUE benchmark and various image classification datasets. SRLoRA consistently achieves faster convergence and improved accuracy over standard LoRA, demonstrating its generality, efficiency, and potential for broader PEFT applications.
>
---
#### [new 134] DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文针对开放词汇语义分割任务，解决现有方法因图像-文本嵌入域差异及浅层特征缺失导致的分割精度不足问题。提出DPSeg框架，通过双提示成本体积生成、视觉提示编码器及语义引导优化策略，融合多级特征以减少域差异并提升细节分割能力，实验验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11676v1](http://arxiv.org/pdf/2505.11676v1)**

> **作者:** Ziyu Zhao; Xiaoguang Li; Linjia Shi; Nasrin Imanpour; Song Wang
>
> **备注:** Accepted by CVPR2025
>
> **摘要:** Open-vocabulary semantic segmentation aims to segment images into distinct semantic regions for both seen and unseen categories at the pixel level. Current methods utilize text embeddings from pre-trained vision-language models like CLIP but struggle with the inherent domain gap between image and text embeddings, even after extensive alignment during training. Additionally, relying solely on deep text-aligned features limits shallow-level feature guidance, which is crucial for detecting small objects and fine details, ultimately reducing segmentation accuracy. To address these limitations, we propose a dual prompting framework, DPSeg, for this task. Our approach combines dual-prompt cost volume generation, a cost volume-guided decoder, and a semantic-guided prompt refinement strategy that leverages our dual prompting scheme to mitigate alignment issues in visual prompt generation. By incorporating visual embeddings from a visual prompt encoder, our approach reduces the domain gap between text and image embeddings while providing multi-level guidance through shallow features. Extensive experiments demonstrate that our method significantly outperforms existing state-of-the-art approaches on multiple public datasets.
>
---
#### [new 135] ORQA: A Benchmark and Foundation Model for Holistic Operating Room Modeling
- **分类: cs.CV**

- **简介: 该论文提出ORQA基准和基础模型，用于多模态手术室综合理解任务。针对现有单任务方法（如阶段识别）的局限性，整合四个公共数据集构建统一基准，融合视觉、听觉等多模态数据，并设计渐进知识蒸馏方法生成高效模型，提升泛化能力和应用灵活性。**

- **链接: [http://arxiv.org/pdf/2505.12890v1](http://arxiv.org/pdf/2505.12890v1)**

> **作者:** Ege Özsoy; Chantal Pellegrini; David Bani-Harouni; Kun Yuan; Matthias Keicher; Nassir Navab
>
> **摘要:** The real-world complexity of surgeries necessitates surgeons to have deep and holistic comprehension to ensure precision, safety, and effective interventions. Computational systems are required to have a similar level of comprehension within the operating room. Prior works, limited to single-task efforts like phase recognition or scene graph generation, lack scope and generalizability. In this work, we introduce ORQA, a novel OR question answering benchmark and foundational multimodal model to advance OR intelligence. By unifying all four public OR datasets into a comprehensive benchmark, we enable our approach to concurrently address a diverse range of OR challenges. The proposed multimodal large language model fuses diverse OR signals such as visual, auditory, and structured data, for a holistic modeling of the OR. Finally, we propose a novel, progressive knowledge distillation paradigm, to generate a family of models optimized for different speed and memory requirements. We show the strong performance of ORQA on our proposed benchmark, and its zero-shot generalization, paving the way for scalable, unified OR modeling and significantly advancing multimodal surgical intelligence. We will release our code and data upon acceptance.
>
---
#### [new 136] Improving Out-of-Domain Robustness with Targeted Augmentation in Frequency and Pixel Spaces
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于域适应任务，旨在提升模型在分布偏移下的跨域鲁棒性。针对传统数据增强泛化性差和定向增强依赖专家知识的问题，提出Frequency-Pixel Connect框架，通过在频率和像素空间混合源域/目标域图像生成增强样本，保持语义的同时引入多样性，实现了数据集无关的跨域增强，在四个多领域基准中超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.12317v1](http://arxiv.org/pdf/2505.12317v1)**

> **作者:** Ruoqi Wang; Haitao Wang; Shaojie Guo; Qiong Luo
>
> **摘要:** Out-of-domain (OOD) robustness under domain adaptation settings, where labeled source data and unlabeled target data come from different distributions, is a key challenge in real-world applications. A common approach to improving OOD robustness is through data augmentations. However, in real-world scenarios, models trained with generic augmentations can only improve marginally when generalized under distribution shifts toward unlabeled target domains. While dataset-specific targeted augmentations can address this issue, they typically require expert knowledge and extensive prior data analysis to identify the nature of the datasets and domain shift. To address these challenges, we propose Frequency-Pixel Connect, a domain-adaptation framework that enhances OOD robustness by introducing a targeted augmentation in both the frequency space and pixel space. Specifically, we mix the amplitude spectrum and pixel content of a source image and a target image to generate augmented samples that introduce domain diversity while preserving the semantic structure of the source image. Unlike previous targeted augmentation methods that are both dataset-specific and limited to the pixel space, Frequency-Pixel Connect is dataset-agnostic, enabling broader and more flexible applicability beyond natural image datasets. We further analyze the effectiveness of Frequency-Pixel Connect by evaluating the performance of our method connecting same-class cross-domain samples while separating different-class examples. We demonstrate that Frequency-Pixel Connect significantly improves cross-domain connectivity and outperforms previous generic methods on four diverse real-world benchmarks across vision, medical, audio, and astronomical domains, and it also outperforms other dataset-specific targeted augmentation methods.
>
---
#### [new 137] Long-RVOS: A Comprehensive Benchmark for Long-term Referring Video Object Segmentation
- **分类: cs.CV**

- **简介: 该论文针对长时指代视频目标分割任务，提出Long-RVOS基准数据集（含2000+平均60秒的长视频）和评估指标，解决现有方法在遮挡、消失重现等实际场景的不足。通过构建包含多类型描述的标注体系，并提出融合运动信息的ReferMo基线方法，显著提升了长视频分割的时空一致性表现。**

- **链接: [http://arxiv.org/pdf/2505.12702v1](http://arxiv.org/pdf/2505.12702v1)**

> **作者:** Tianming Liang; Haichao Jiang; Yuting Yang; Chaolei Tan; Shuai Li; Wei-Shi Zheng; Jian-Fang Hu
>
> **备注:** Project Page: \url{https://isee-laboratory.github.io/Long-RVOS}
>
> **摘要:** Referring video object segmentation (RVOS) aims to identify, track and segment the objects in a video based on language descriptions, which has received great attention in recent years. However, existing datasets remain focus on short video clips within several seconds, with salient objects visible in most frames. To advance the task towards more practical scenarios, we introduce \textbf{Long-RVOS}, a large-scale benchmark for long-term referring video object segmentation. Long-RVOS contains 2,000+ videos of an average duration exceeding 60 seconds, covering a variety of objects that undergo occlusion, disappearance-reappearance and shot changing. The objects are manually annotated with three different types of descriptions to individually evaluate the understanding of static attributes, motion patterns and spatiotemporal relationships. Moreover, unlike previous benchmarks that rely solely on the per-frame spatial evaluation, we introduce two new metrics to assess the temporal and spatiotemporal consistency. We benchmark 6 state-of-the-art methods on Long-RVOS. The results show that current approaches struggle severely with the long-video challenges. To address this, we further propose ReferMo, a promising baseline method that integrates motion information to expand the temporal receptive field, and employs a local-to-global architecture to capture both short-term dynamics and long-term dependencies. Despite simplicity, ReferMo achieves significant improvements over current methods in long-term scenarios. We hope that Long-RVOS and our baseline can drive future RVOS research towards tackling more realistic and long-form videos.
>
---
#### [new 138] Self-Learning Hyperspectral and Multispectral Image Fusion via Adaptive Residual Guided Subspace Diffusion Model
- **分类: cs.CV; eess.IV**

- **简介: 该论文研究高光谱与多光谱图像融合任务，解决传统深度学习方法依赖大量训练数据的问题。提出自学习的ARGS-Diff模型，通过轻量级扩散模型分别提取光谱和空间特征，结合自适应残差模块优化重建过程，无需额外数据即实现高效融合。**

- **链接: [http://arxiv.org/pdf/2505.11800v1](http://arxiv.org/pdf/2505.11800v1)**

> **作者:** Jian Zhu; He Wang; Yang Xu; Zebin Wu; Zhihui Wei
>
> **备注:** cvpr
>
> **摘要:** Hyperspectral and multispectral image (HSI-MSI) fusion involves combining a low-resolution hyperspectral image (LR-HSI) with a high-resolution multispectral image (HR-MSI) to generate a high-resolution hyperspectral image (HR-HSI). Most deep learning-based methods for HSI-MSI fusion rely on large amounts of hyperspectral data for supervised training, which is often scarce in practical applications. In this paper, we propose a self-learning Adaptive Residual Guided Subspace Diffusion Model (ARGS-Diff), which only utilizes the observed images without any extra training data. Specifically, as the LR-HSI contains spectral information and the HR-MSI contains spatial information, we design two lightweight spectral and spatial diffusion models to separately learn the spectral and spatial distributions from them. Then, we use these two models to reconstruct HR-HSI from two low-dimensional components, i.e, the spectral basis and the reduced coefficient, during the reverse diffusion process. Furthermore, we introduce an Adaptive Residual Guided Module (ARGM), which refines the two components through a residual guided function at each sampling step, thereby stabilizing the sampling process. Extensive experimental results demonstrate that ARGS-Diff outperforms existing state-of-the-art methods in terms of both performance and computational efficiency in the field of HSI-MSI fusion. Code is available at https://github.com/Zhu1116/ARGS-Diff.
>
---
#### [new 139] Any-to-Any Learning in Computational Pathology via Triplet Multimodal Pretraining
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算病理学的多模态学习任务，旨在解决多模态数据融合困难、模态缺失鲁棒性及任务适配性问题。提出ALTER框架，通过整合全切片图像、基因组和病理报告的三模态预训练，支持任意模态组合输入，学习跨模态表征，在生存预测、癌症分型等临床任务中达到先进性能。**

- **链接: [http://arxiv.org/pdf/2505.12711v1](http://arxiv.org/pdf/2505.12711v1)**

> **作者:** Qichen Sun; Zhengrui Guo; Rui Peng; Hao Chen; Jinzhuo Wang
>
> **摘要:** Recent advances in computational pathology and artificial intelligence have significantly enhanced the utilization of gigapixel whole-slide images and and additional modalities (e.g., genomics) for pathological diagnosis. Although deep learning has demonstrated strong potential in pathology, several key challenges persist: (1) fusing heterogeneous data types requires sophisticated strategies beyond simple concatenation due to high computational costs; (2) common scenarios of missing modalities necessitate flexible strategies that allow the model to learn robustly in the absence of certain modalities; (3) the downstream tasks in CPath are diverse, ranging from unimodal to multimodal, cnecessitating a unified model capable of handling all modalities. To address these challenges, we propose ALTER, an any-to-any tri-modal pretraining framework that integrates WSIs, genomics, and pathology reports. The term "any" emphasizes ALTER's modality-adaptive design, enabling flexible pretraining with any subset of modalities, and its capacity to learn robust, cross-modal representations beyond WSI-centric approaches. We evaluate ALTER across extensive clinical tasks including survival prediction, cancer subtyping, gene mutation prediction, and report generation, achieving superior or comparable performance to state-of-the-art baselines.
>
---
#### [new 140] Cross-modal feature fusion for robust point cloud registration with ambiguous geometry
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究点云配准任务，解决几何信息模糊区域（如对称或平面结构）配准不准的问题。提出CoFF方法，通过两阶段融合3D点云与2D图像特征，利用跨模态特征增强匹配精度，在多个数据集上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2505.13088v1](http://arxiv.org/pdf/2505.13088v1)**

> **作者:** Zhaoyi Wang; Shengyu Huang; Jemil Avers Butt; Yuanzhou Cai; Matej Varga; Andreas Wieser
>
> **备注:** To appear in the ISPRS Journal of Photogrammetry and Remote Sensing. 19 pages, 14 figures
>
> **摘要:** Point cloud registration has seen significant advancements with the application of deep learning techniques. However, existing approaches often overlook the potential of integrating radiometric information from RGB images. This limitation reduces their effectiveness in aligning point clouds pairs, especially in regions where geometric data alone is insufficient. When used effectively, radiometric information can enhance the registration process by providing context that is missing from purely geometric data. In this paper, we propose CoFF, a novel Cross-modal Feature Fusion method that utilizes both point cloud geometry and RGB images for pairwise point cloud registration. Assuming that the co-registration between point clouds and RGB images is available, CoFF explicitly addresses the challenges where geometric information alone is unclear, such as in regions with symmetric similarity or planar structures, through a two-stage fusion of 3D point cloud features and 2D image features. It incorporates a cross-modal feature fusion module that assigns pixel-wise image features to 3D input point clouds to enhance learned 3D point features, and integrates patch-wise image features with superpoint features to improve the quality of coarse matching. This is followed by a coarse-to-fine matching module that accurately establishes correspondences using the fused features. We extensively evaluate CoFF on four common datasets: 3DMatch, 3DLoMatch, IndoorLRS, and the recently released ScanNet++ datasets. In addition, we assess CoFF on specific subset datasets containing geometrically ambiguous cases. Our experimental results demonstrate that CoFF achieves state-of-the-art registration performance across all benchmarks, including remarkable registration recalls of 95.9% and 81.6% on the widely-used 3DMatch and 3DLoMatch datasets, respectively...(Truncated to fit arXiv abstract length)
>
---
#### [new 141] Reasoning-OCR: Can Large Multimodal Models Solve Complex Logical Reasoning Problems from OCR Cues?
- **分类: cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决现有OCR基准在评估大型多模态模型（LMMs）复杂逻辑推理能力上的不足。作者提出Reasoning-OCR新基准，涵盖6类视觉场景和150个结构化问题，弱化领域知识干扰，评估发现现有模型推理性能亟待提升，为增强OCR线索的复杂推理研究提供支持。**

- **链接: [http://arxiv.org/pdf/2505.12766v1](http://arxiv.org/pdf/2505.12766v1)**

> **作者:** Haibin He; Maoyuan Ye; Jing Zhang; Xiantao Cai; Juhua Liu; Bo Du; Dacheng Tao
>
> **摘要:** Large Multimodal Models (LMMs) have become increasingly versatile, accompanied by impressive Optical Character Recognition (OCR) related capabilities. Existing OCR-related benchmarks emphasize evaluating LMMs' abilities of relatively simple visual question answering, visual-text parsing, etc. However, the extent to which LMMs can deal with complex logical reasoning problems based on OCR cues is relatively unexplored. To this end, we introduce the Reasoning-OCR benchmark, which challenges LMMs to solve complex reasoning problems based on the cues that can be extracted from rich visual-text. Reasoning-OCR covers six visual scenarios and encompasses 150 meticulously designed questions categorized into six reasoning challenges. Additionally, Reasoning-OCR minimizes the impact of field-specialized knowledge. Our evaluation offers some insights for proprietary and open-source LMMs in different reasoning challenges, underscoring the urgent to improve the reasoning performance. We hope Reasoning-OCR can inspire and facilitate future research on enhancing complex reasoning ability based on OCR cues. Reasoning-OCR is publicly available at https://github.com/Hxyz-123/ReasoningOCR.
>
---
#### [new 142] Generalizable Vision-Language Few-Shot Adaptation with Predictive Prompts and Negative Learning
- **分类: cs.CV; cs.AI; cs.GR; cs.RO**

- **简介: 该论文研究少样本视觉-语言模型适应任务，解决有限监督和噪声样本下的泛化与效率问题。提出PromptFuseNL框架，融合预测提示调优与双分支正负学习，通过任务条件残差、跨模态协调和语义硬负挖掘优化原型，采用无监督样本加权抑制噪声。方法在15个基准上实现最高精度，训练速度提升300倍，计算量降低1000倍。**

- **链接: [http://arxiv.org/pdf/2505.11758v1](http://arxiv.org/pdf/2505.11758v1)**

> **作者:** Sriram Mandalika
>
> **摘要:** Few-shot adaptation remains a core challenge for vision-language models (VLMs), especially under limited supervision and noisy support samples. We propose PromptFuseNL, a unified framework that enhances few-shot generalization by combining predictive prompt tuning with dual-branch positive and negative learning. The method refines class prototypes through task-conditioned residuals, multi-stage cross-modal coordination, and semantic hard negative mining. To address label noise, we introduce an unsupervised instance reweighting strategy that downweights unreliable support examples without requiring additional labels or structural changes. PromptFuseNL fuses visual and textual cues through lightweight modules for efficient and discriminative prediction. Evaluated across 15 benchmarks, it consistently surpasses existing prompt- and adapter-based methods in all shot settings while remaining highly efficient, achieving up to 300x faster training and 1000x lower FLOPs compared to full prompt tuning, achieving a new state-of-the-art for robust and scalable few-shot vision-language adaptation.
>
---
#### [new 143] Recollection from Pensieve: Novel View Synthesis via Learning from Uncalibrated Videos
- **分类: cs.CV**

- **简介: 该论文研究无校准视频的新视角合成任务，解决现有模型依赖相机参数或几何先验的问题。提出两阶段方法：第一阶段通过隐式潜在空间重建场景，第二阶段引入3D高斯基元与渲染损失增强几何一致性，实现自监督训练，提升合成质量与姿态估计精度。**

- **链接: [http://arxiv.org/pdf/2505.13440v1](http://arxiv.org/pdf/2505.13440v1)**

> **作者:** Ruoyu Wang; Yi Ma; Shenghua Gao
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Currently almost all state-of-the-art novel view synthesis and reconstruction models rely on calibrated cameras or additional geometric priors for training. These prerequisites significantly limit their applicability to massive uncalibrated data. To alleviate this requirement and unlock the potential for self-supervised training on large-scale uncalibrated videos, we propose a novel two-stage strategy to train a view synthesis model from only raw video frames or multi-view images, without providing camera parameters or other priors. In the first stage, we learn to reconstruct the scene implicitly in a latent space without relying on any explicit 3D representation. Specifically, we predict per-frame latent camera and scene context features, and employ a view synthesis model as a proxy for explicit rendering. This pretraining stage substantially reduces the optimization complexity and encourages the network to learn the underlying 3D consistency in a self-supervised manner. The learned latent camera and implicit scene representation have a large gap compared with the real 3D world. To reduce this gap, we introduce the second stage training by explicitly predicting 3D Gaussian primitives. We additionally apply explicit Gaussian Splatting rendering loss and depth projection loss to align the learned latent representations with physically grounded 3D geometry. In this way, Stage 1 provides a strong initialization and Stage 2 enforces 3D consistency - the two stages are complementary and mutually beneficial. Extensive experiments demonstrate the effectiveness of our approach, achieving high-quality novel view synthesis and accurate camera pose estimation, compared to methods that employ supervision with calibration, pose, or depth information. The code is available at https://github.com/Dwawayu/Pensieve.
>
---
#### [new 144] Degradation-Aware Feature Perturbation for All-in-One Image Restoration
- **分类: cs.CV; cs.AI; I.4.5**

- **简介: 该论文属于多合一图像恢复任务，旨在解决不同退化类型导致的模型参数冲突问题。通过引入退化感知特征扰动（通道重排和注意力掩码），设计DGPB模块调整特征空间对齐参数，减少任务干扰，在去噪、去雾等任务中取得最优效果。**

- **链接: [http://arxiv.org/pdf/2505.12630v1](http://arxiv.org/pdf/2505.12630v1)**

> **作者:** Xiangpeng Tian; Xiangyu Liao; Xiao Liu; Meng Li; Chao Ren
>
> **备注:** Accepted to CVPR 2025. 8 pages, 7 figures
>
> **摘要:** All-in-one image restoration aims to recover clear images from various degradation types and levels with a unified model. Nonetheless, the significant variations among degradation types present challenges for training a universal model, often resulting in task interference, where the gradient update directions of different tasks may diverge due to shared parameters. To address this issue, motivated by the routing strategy, we propose DFPIR, a novel all-in-one image restorer that introduces Degradation-aware Feature Perturbations(DFP) to adjust the feature space to align with the unified parameter space. In this paper, the feature perturbations primarily include channel-wise perturbations and attention-wise perturbations. Specifically, channel-wise perturbations are implemented by shuffling the channels in high-dimensional space guided by degradation types, while attention-wise perturbations are achieved through selective masking in the attention space. To achieve these goals, we propose a Degradation-Guided Perturbation Block (DGPB) to implement these two functions, positioned between the encoding and decoding stages of the encoder-decoder architecture. Extensive experimental results demonstrate that DFPIR achieves state-of-the-art performance on several all-in-one image restoration tasks including image denoising, image dehazing, image deraining, motion deblurring, and low-light image enhancement. Our codes are available at https://github.com/TxpHome/DFPIR.
>
---
#### [new 145] EPIC: Explanation of Pretrained Image Classification Networks via Prototype
- **分类: cs.CV**

- **简介: 该论文属于可解释AI任务，旨在解决现有方法中后处理解释粗糙、原型方法依赖定制模型的问题。提出EPIC框架，在不修改预训练模型结构的前提下，结合后处理灵活性和原型解释的直观性，通过提取代表性图像块生成高质量解释，适配多种数据集（如ImageNet），填补两类方法的空白。**

- **链接: [http://arxiv.org/pdf/2505.12897v1](http://arxiv.org/pdf/2505.12897v1)**

> **作者:** Piotr Borycki; Magdalena Trędowicz; Szymon Janusz; Jacek Tabor; Przemysław Spurek; Arkadiusz Lewicki; Łukasz Struski
>
> **摘要:** Explainable AI (XAI) methods generally fall into two categories. Post-hoc approaches generate explanations for pre-trained models and are compatible with various neural network architectures. These methods often use feature importance visualizations, such as saliency maps, to indicate which input regions influenced the model's prediction. Unfortunately, they typically offer a coarse understanding of the model's decision-making process. In contrast, ante-hoc (inherently explainable) methods rely on specially designed model architectures trained from scratch. A notable subclass of these methods provides explanations through prototypes, representative patches extracted from the training data. However, prototype-based approaches have limitations: they require dedicated architectures, involve specialized training procedures, and perform well only on specific datasets. In this work, we propose EPIC (Explanation of Pretrained Image Classification), a novel approach that bridges the gap between these two paradigms. Like post-hoc methods, EPIC operates on pre-trained models without architectural modifications. Simultaneously, it delivers intuitive, prototype-based explanations inspired by ante-hoc techniques. To the best of our knowledge, EPIC is the first post-hoc method capable of fully replicating the core explanatory power of inherently interpretable models. We evaluate EPIC on benchmark datasets commonly used in prototype-based explanations, such as CUB-200-2011 and Stanford Cars, alongside large-scale datasets like ImageNet, typically employed by post-hoc methods. EPIC uses prototypes to explain model decisions, providing a flexible and easy-to-understand tool for creating clear, high-quality explanations.
>
---
#### [new 146] Enhancing Transformers Through Conditioned Embedded Tokens
- **分类: cs.CV**

- **简介: 该论文属于Transformer模型优化任务，旨在解决注意力机制因病态条件导致的训练低效问题。通过理论分析建立嵌入令牌与注意力条件的关系，提出条件化嵌入令牌方法，系统调整嵌入数据以改善模型条件。实验证明该方法提升训练稳定性，在图像分类、NLP等任务中取得显著效果。**

- **链接: [http://arxiv.org/pdf/2505.12789v1](http://arxiv.org/pdf/2505.12789v1)**

> **作者:** Hemanth Saratchandran; Simon Lucey
>
> **摘要:** Transformers have transformed modern machine learning, driving breakthroughs in computer vision, natural language processing, and robotics. At the core of their success lies the attention mechanism, which enables the modeling of global dependencies among input tokens. However, we reveal that the attention block in transformers suffers from inherent ill-conditioning, which hampers gradient-based optimization and leads to inefficient training. To address this, we develop a theoretical framework that establishes a direct relationship between the conditioning of the attention block and that of the embedded tokenized data. Building on this insight, we introduce conditioned embedded tokens, a method that systematically modifies the embedded tokens to improve the conditioning of the attention mechanism. Our analysis demonstrates that this approach significantly mitigates ill-conditioning, leading to more stable and efficient training. We validate our methodology across various transformer architectures, achieving consistent improvements in image classification, object detection, instance segmentation, and natural language processing, highlighting its broad applicability and effectiveness.
>
---
#### [new 147] Image-based Visibility Analysis Replacing Line-of-Sight Simulation: An Urban Landmark Perspective
- **分类: cs.CV**

- **简介: 该论文提出一种基于图像和视觉语言模型的可见性分析方法，替代传统视线模拟，解决城市地标可见性评估中缺乏感知维度的问题。通过街景图像目标检测构建异构图，揭示观察者与地标的互动关系，案例验证了方法在准确性和环境关联分析上的优势，为城市规划提供新视角。**

- **链接: [http://arxiv.org/pdf/2505.11809v1](http://arxiv.org/pdf/2505.11809v1)**

> **作者:** Zicheng Fan; Kunihiko Fujiwara; Pengyuan Liu; Fan Zhang; Filip Biljecki
>
> **摘要:** Visibility analysis is one of the fundamental analytics methods in urban planning and landscape research, traditionally conducted through computational simulations based on the Line-of-Sight (LoS) principle. However, when assessing the visibility of named urban objects such as landmarks, geometric intersection alone fails to capture the contextual and perceptual dimensions of visibility as experienced in the real world. The study challenges the traditional LoS-based approaches by introducing a new, image-based visibility analysis method. Specifically, a Vision Language Model (VLM) is applied to detect the target object within a direction-zoomed Street View Image (SVI). Successful detection represents the object's visibility at the corresponding SVI location. Further, a heterogeneous visibility graph is constructed to address the complex interaction between observers and target objects. In the first case study, the method proves its reliability in detecting the visibility of six tall landmark constructions in global cities, with an overall accuracy of 87%. Furthermore, it reveals broader contextual differences when the landmarks are perceived and experienced. In the second case, the proposed visibility graph uncovers the form and strength of connections for multiple landmarks along the River Thames in London, as well as the places where these connections occur. Notably, bridges on the River Thames account for approximately 30% of total connections. Our method complements and enhances traditional LoS-based visibility analysis, and showcases the possibility of revealing the prevalent connection of any visual objects in the urban environment. It opens up new research perspectives for urban planning, heritage conservation, and computational social science.
>
---
#### [new 148] CacheFlow: Fast Human Motion Prediction by Cached Normalizing Flow
- **分类: cs.CV**

- **简介: 该论文研究3D人体运动预测任务，解决现有密度估计方法推理速度慢的问题。提出CacheFlow方法：使用预计算的无条件流模型生成运动分布，结合轻量级条件映射模型加速推理，在保持精度的同时实现毫秒级预测，速度超越VAE和扩散模型4-30倍，并在Human3.6M数据集达到SOTA水平。**

- **链接: [http://arxiv.org/pdf/2505.13140v1](http://arxiv.org/pdf/2505.13140v1)**

> **作者:** Takahiro Maeda; Jinkun Cao; Norimichi Ukita; Kris Kitani
>
> **摘要:** Many density estimation techniques for 3D human motion prediction require a significant amount of inference time, often exceeding the duration of the predicted time horizon. To address the need for faster density estimation for 3D human motion prediction, we introduce a novel flow-based method for human motion prediction called CacheFlow. Unlike previous conditional generative models that suffer from time efficiency, CacheFlow takes advantage of an unconditional flow-based generative model that transforms a Gaussian mixture into the density of future motions. The results of the computation of the flow-based generative model can be precomputed and cached. Then, for conditional prediction, we seek a mapping from historical trajectories to samples in the Gaussian mixture. This mapping can be done by a much more lightweight model, thus saving significant computation overhead compared to a typical conditional flow model. In such a two-stage fashion and by caching results from the slow flow model computation, we build our CacheFlow without loss of prediction accuracy and model expressiveness. This inference process is completed in approximately one millisecond, making it 4 times faster than previous VAE methods and 30 times faster than previous diffusion-based methods on standard benchmarks such as Human3.6M and AMASS datasets. Furthermore, our method demonstrates improved density estimation accuracy and comparable prediction accuracy to a SOTA method on Human3.6M. Our code and models will be publicly available.
>
---
#### [new 149] SEPT: Standard-Definition Map Enhanced Scene Perception and Topology Reasoning for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文研究自动驾驶中的场景感知与拓扑推理任务，旨在解决现有方法在远距离或遮挡场景中的性能限制。通过将标准清晰度地图作为先验知识，提出SEPT框架，设计混合特征融合策略结合鸟瞰图特征，并引入交叉口感知关键点检测任务，显著提升了环境理解能力。**

- **链接: [http://arxiv.org/pdf/2505.12246v1](http://arxiv.org/pdf/2505.12246v1)**

> **作者:** Muleilan Pei; Jiayao Shan; Peiliang Li; Jieqi Shi; Jing Huo; Yang Gao; Shaojie Shen
>
> **备注:** Accepted by IEEE Robotics and Automation Letters
>
> **摘要:** Online scene perception and topology reasoning are critical for autonomous vehicles to understand their driving environments, particularly for mapless driving systems that endeavor to reduce reliance on costly High-Definition (HD) maps. However, recent advances in online scene understanding still face limitations, especially in long-range or occluded scenarios, due to the inherent constraints of onboard sensors. To address this challenge, we propose a Standard-Definition (SD) Map Enhanced scene Perception and Topology reasoning (SEPT) framework, which explores how to effectively incorporate the SD map as prior knowledge into existing perception and reasoning pipelines. Specifically, we introduce a novel hybrid feature fusion strategy that combines SD maps with Bird's-Eye-View (BEV) features, considering both rasterized and vectorized representations, while mitigating potential misalignment between SD maps and BEV feature spaces. Additionally, we leverage the SD map characteristics to design an auxiliary intersection-aware keypoint detection task, which further enhances the overall scene understanding performance. Experimental results on the large-scale OpenLane-V2 dataset demonstrate that by effectively integrating SD map priors, our framework significantly improves both scene perception and topology reasoning, outperforming existing methods by a substantial margin.
>
---
#### [new 150] Emergence of Fixational and Saccadic Movements in a Multi-Level Recurrent Attention Model for Vision
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉注意力模型研究，旨在解决现有硬注意力模型（如RAM、DRAM）因缺乏视觉系统层级结构导致的非人眼动态行为问题。提出多级循环注意力模型（MRAM），通过解耦注视定位与任务处理的循环层，平衡固定注视与扫视运动，在图像分类任务中实现更接近人类视觉的注意力机制和更高性能。**

- **链接: [http://arxiv.org/pdf/2505.13191v1](http://arxiv.org/pdf/2505.13191v1)**

> **作者:** Pengcheng Pan; Yonekura Shogo; Yasuo Kuniyoshi
>
> **摘要:** Inspired by foveal vision, hard attention models promise interpretability and parameter economy. However, existing models like the Recurrent Model of Visual Attention (RAM) and Deep Recurrent Attention Model (DRAM) failed to model the hierarchy of human vision system, that compromise on the visual exploration dynamics. As a result, they tend to produce attention that are either overly fixational or excessively saccadic, diverging from human eye movement behavior. In this paper, we propose a Multi-Level Recurrent Attention Model (MRAM), a novel hard attention framework that explicitly models the neural hierarchy of human visual processing. By decoupling the function of glimpse location generation and task execution in two recurrent layers, MRAM emergent a balanced behavior between fixation and saccadic movement. Our results show that MRAM not only achieves more human-like attention dynamics, but also consistently outperforms CNN, RAM and DRAM baselines on standard image classification benchmarks.
>
---
#### [new 151] DIMM: Decoupled Multi-hierarchy Kalman Filter for 3D Object Tracking
- **分类: cs.CV**

- **简介: 该论文研究3D目标跟踪中的状态估计问题，针对传统交互多模型（IMM）方法模型组合解空间受限和权重计算不准确的问题，提出DIMM框架。通过解耦多层级滤波器组扩展解空间，并设计自适应融合网络优化权重矩阵，实验表明跟踪精度提升31.61%~99.23%。**

- **链接: [http://arxiv.org/pdf/2505.12340v1](http://arxiv.org/pdf/2505.12340v1)**

> **作者:** Jirong Zha; Yuxuan Fan; Kai Li; Han Li; Chen Gao; Xinlei Chen; Yong Li
>
> **备注:** 10 pages
>
> **摘要:** State estimation is challenging for 3D object tracking with high maneuverability, as the target's state transition function changes rapidly, irregularly, and is unknown to the estimator. Existing work based on interacting multiple model (IMM) achieves more accurate estimation than single-filter approaches through model combination, aligning appropriate models for different motion modes of the target object over time. However, two limitations of conventional IMM remain unsolved. First, the solution space of the model combination is constrained as the target's diverse kinematic properties in different directions are ignored. Second, the model combination weights calculated by the observation likelihood are not accurate enough due to the measurement uncertainty. In this paper, we propose a novel framework, DIMM, to effectively combine estimates from different motion models in each direction, thus increasing the 3D object tracking accuracy. First, DIMM extends the model combination solution space of conventional IMM from a hyperplane to a hypercube by designing a 3D-decoupled multi-hierarchy filter bank, which describes the target's motion with various-order linear models. Second, DIMM generates more reliable combination weight matrices through a differentiable adaptive fusion network for importance allocation rather than solely relying on the observation likelihood; it contains an attention-based twin delayed deep deterministic policy gradient (TD3) method with a hierarchical reward. Experiments demonstrate that DIMM significantly improves the tracking accuracy of existing state estimation methods by 31.61%~99.23%.
>
---
#### [new 152] Are Multimodal Large Language Models Ready for Omnidirectional Spatial Reasoning?
- **分类: cs.CV**

- **简介: 该论文研究多模态大语言模型（MLLMs）在全景空间推理中的能力，属于视觉-语言推理任务。针对现有模型在360度图像理解上的不足，提出首个全景空间推理基准OSR-Bench（含15.3万QA对），设计负采样策略和两阶段评估框架，测试8个主流模型后发现其全景推理能力薄弱，需提升感知基础。**

- **链接: [http://arxiv.org/pdf/2505.11907v1](http://arxiv.org/pdf/2505.11907v1)**

> **作者:** Zihao Dongfang; Xu Zheng; Ziqiao Weng; Yuanhuiyi Lyu; Danda Pani Paudel; Luc Van Gool; Kailun Yang; Xuming Hu
>
> **摘要:** The 180x360 omnidirectional field of view captured by 360-degree cameras enables their use in a wide range of applications such as embodied AI and virtual reality. Although recent advances in multimodal large language models (MLLMs) have shown promise in visual-spatial reasoning, most studies focus on standard pinhole-view images, leaving omnidirectional perception largely unexplored. In this paper, we ask: Are MLLMs ready for omnidirectional spatial reasoning? To investigate this, we introduce OSR-Bench, the first benchmark specifically designed for this setting. OSR-Bench includes over 153,000 diverse question-answer pairs grounded in high-fidelity panoramic indoor scene maps. It covers key reasoning types including object counting, relative distance, and direction. We also propose a negative sampling strategy that inserts non-existent objects into prompts to evaluate hallucination and grounding robustness. For fine-grained analysis, we design a two-stage evaluation framework assessing both cognitive map generation and QA accuracy using rotation-invariant matching and a combination of rule-based and LLM-based metrics. We evaluate eight state-of-the-art MLLMs, including GPT-4o, Gemini 1.5 Pro, and leading open-source models under zero-shot settings. Results show that current models struggle with spatial reasoning in panoramic contexts, highlighting the need for more perceptually grounded MLLMs. OSR-Bench and code will be released at: https://huggingface.co/datasets/UUUserna/OSR-Bench
>
---
#### [new 153] Understanding Complexity in VideoQA via Visual Program Generation
- **分类: cs.CV**

- **简介: 该论文属于视频问答（VideoQA）领域，旨在解决人工设计复杂问题基准的局限性。通过将代码生成复杂度作为问题难度指标，提出自动评估方法，开发算法识别关键困难元素，并生成比现有基准难1.9倍的新测试集，实现模型性能与问题难度的精准匹配。**

- **链接: [http://arxiv.org/pdf/2505.13429v1](http://arxiv.org/pdf/2505.13429v1)**

> **作者:** Cristobal Eyzaguirre; Igor Vasiljevic; Achal Dave; Jiajun Wu; Rares Andrei Ambrus; Thomas Kollar; Juan Carlos Niebles; Pavel Tokmakov
>
> **摘要:** We propose a data-driven approach to analyzing query complexity in Video Question Answering (VideoQA). Previous efforts in benchmark design have relied on human expertise to design challenging questions, yet we experimentally show that humans struggle to predict which questions are difficult for machine learning models. Our automatic approach leverages recent advances in code generation for visual question answering, using the complexity of generated code as a proxy for question difficulty. We demonstrate that this measure correlates significantly better with model performance than human estimates. To operationalize this insight, we propose an algorithm for estimating question complexity from code. It identifies fine-grained primitives that correlate with the hardest questions for any given set of models, making it easy to scale to new approaches in the future. Finally, to further illustrate the utility of our method, we extend it to automatically generate complex questions, constructing a new benchmark that is 1.9 times harder than the popular NExT-QA.
>
---
#### [new 154] Coarse Attribute Prediction with Task Agnostic Distillation for Real World Clothes Changing ReID
- **分类: cs.CV**

- **简介: 该论文针对真实场景下衣物更换的行人重识别任务（CC-ReID），解决低质量图像导致特征混淆的问题。提出RLQ框架，通过交替训练粗粒度属性预测（CAP）增强外部属性感知，结合任务无关蒸馏（TAD）提升HQ/LQ特征一致性，在多个数据集实现1.6%-6%的性能提升。**

- **链接: [http://arxiv.org/pdf/2505.12580v1](http://arxiv.org/pdf/2505.12580v1)**

> **作者:** Priyank Pathak; Yogesh S Rawat
>
> **摘要:** This work focuses on Clothes Changing Re-IDentification (CC-ReID) for the real world. Existing works perform well with high-quality (HQ) images, but struggle with low-quality (LQ) where we can have artifacts like pixelation, out-of-focus blur, and motion blur. These artifacts introduce noise to not only external biometric attributes (e.g. pose, body shape, etc.) but also corrupt the model's internal feature representation. Models usually cluster LQ image features together, making it difficult to distinguish between them, leading to incorrect matches. We propose a novel framework Robustness against Low-Quality (RLQ) to improve CC-ReID model on real-world data. RLQ relies on Coarse Attributes Prediction (CAP) and Task Agnostic Distillation (TAD) operating in alternate steps in a novel training mechanism. CAP enriches the model with external fine-grained attributes via coarse predictions, thereby reducing the effect of noisy inputs. On the other hand, TAD enhances the model's internal feature representation by bridging the gap between HQ and LQ features, via an external dataset through task-agnostic self-supervision and distillation. RLQ outperforms the existing approaches by 1.6%-2.9% Top-1 on real-world datasets like LaST, and DeepChange, while showing consistent improvement of 5.3%-6% Top-1 on PRCC with competitive performance on LTCC. *The code will be made public soon.*
>
---
#### [new 155] FiGKD: Fine-Grained Knowledge Distillation via High-Frequency Detail Transfer
- **分类: cs.CV**

- **简介: 该论文属于知识蒸馏领域，针对传统方法在细粒度视觉任务中因冗余信号导致学生模型无法捕捉教师决策边界的问题，提出FiGKD框架。通过小波变换分解logits，选择性传递高频细节成分，保留语义决策模式，实验证明其在多数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11897v1](http://arxiv.org/pdf/2505.11897v1)**

> **作者:** Seonghak Kim
>
> **备注:** 14 pages, 6 figures. This work has been submitted to the Elsevier for possible publication
>
> **摘要:** Knowledge distillation (KD) is a widely adopted technique for transferring knowledge from a high-capacity teacher model to a smaller student model by aligning their output distributions. However, existing methods often underperform in fine-grained visual recognition tasks, where distinguishing subtle differences between visually similar classes is essential. This performance gap stems from the fact that conventional approaches treat the teacher's output logits as a single, undifferentiated signal-assuming all contained information is equally beneficial to the student. Consequently, student models may become overloaded with redundant signals and fail to capture the teacher's nuanced decision boundaries. To address this issue, we propose Fine-Grained Knowledge Distillation (FiGKD), a novel frequency-aware framework that decomposes a model's logits into low-frequency (content) and high-frequency (detail) components using the discrete wavelet transform (DWT). FiGKD selectively transfers only the high-frequency components, which encode the teacher's semantic decision patterns, while discarding redundant low-frequency content already conveyed through ground-truth supervision. Our approach is simple, architecture-agnostic, and requires no access to intermediate feature maps. Extensive experiments on CIFAR-100, TinyImageNet, and multiple fine-grained recognition benchmarks show that FiGKD consistently outperforms state-of-the-art logit-based and feature-based distillation methods across a variety of teacher-student configurations. These findings confirm that frequency-aware logit decomposition enables more efficient and effective knowledge transfer, particularly in resource-constrained settings.
>
---
#### [new 156] MVPainter: Accurate and Detailed 3D Texture Generation via Multi-View Diffusion with Geometric Control
- **分类: cs.CV**

- **简介: 该论文研究3D纹理生成任务，解决现有方法在纹理对齐、几何一致性及局部细节上的不足。提出MVPainter框架，通过数据增强、ControlNet几何控制提升纹理精度，并提取物理渲染属性生成实用3D网格，同时开源全流程系统实现最优效果。**

- **链接: [http://arxiv.org/pdf/2505.12635v1](http://arxiv.org/pdf/2505.12635v1)**

> **作者:** Mingqi Shao; Feng Xiong; Zhaoxu Sun; Mu Xu
>
> **备注:** Project page: https://amap-cvlab.github.io/MV-Painter
>
> **摘要:** Recently, significant advances have been made in 3D object generation. Building upon the generated geometry, current pipelines typically employ image diffusion models to generate multi-view RGB images, followed by UV texture reconstruction through texture baking. While 3D geometry generation has improved significantly, supported by multiple open-source frameworks, 3D texture generation remains underexplored. In this work, we systematically investigate 3D texture generation through the lens of three core dimensions: reference-texture alignment, geometry-texture consistency, and local texture quality. To tackle these issues, we propose MVPainter, which employs data filtering and augmentation strategies to enhance texture fidelity and detail, and introduces ControlNet-based geometric conditioning to improve texture-geometry alignment. Furthermore, we extract physically-based rendering (PBR) attributes from the generated views to produce PBR meshes suitable for real-world rendering applications. MVPainter achieves state-of-the-art results across all three dimensions, as demonstrated by human-aligned evaluations. To facilitate further research and reproducibility, we also release our full pipeline as an open-source system, including data construction, model architecture, and evaluation tools.
>
---
#### [new 157] Robust Cross-View Geo-Localization via Content-Viewpoint Disentanglement
- **分类: cs.CV**

- **简介: 该论文针对跨视角地理定位任务，解决不同视角（如无人机与卫星）图像因外观差异和空间扭曲导致的匹配难题。提出CVD框架，通过解耦内容和视点信息，引入独立性约束与跨视角重建约束，分离干扰因素，提升现有方法的定位精度与泛化性。实验验证其在多基准中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.11822v1](http://arxiv.org/pdf/2505.11822v1)**

> **作者:** Ke Li; Di Wang; Xiaowei Wang; Zhihong Wu; Yiming Zhang; Yifeng Wang; Quan Wang
>
> **摘要:** Cross-view geo-localization (CVGL) aims to match images of the same geographic location captured from different perspectives, such as drones and satellites. Despite recent advances, CVGL remains highly challenging due to significant appearance changes and spatial distortions caused by viewpoint variations. Existing methods typically assume that cross-view images can be directly aligned within a shared feature space by maximizing feature similarity through contrastive learning. Nonetheless, this assumption overlooks the inherent conflicts induced by viewpoint discrepancies, resulting in extracted features containing inconsistent information that hinders precise localization. In this study, we take a manifold learning perspective and model the feature space of cross-view images as a composite manifold jointly governed by content and viewpoint information. Building upon this insight, we propose $\textbf{CVD}$, a new CVGL framework that explicitly disentangles $\textit{content}$ and $\textit{viewpoint}$ factors. To promote effective disentanglement, we introduce two constraints: $\textit{(i)}$ An intra-view independence constraint, which encourages statistical independence between the two factors by minimizing their mutual information. $\textit{(ii)}$ An inter-view reconstruction constraint that reconstructs each view by cross-combining $\textit{content}$ and $\textit{viewpoint}$ from paired images, ensuring factor-specific semantics are preserved. As a plug-and-play module, CVD can be seamlessly integrated into existing geo-localization pipelines. Extensive experiments on four benchmarks, i.e., University-1652, SUES-200, CVUSA, and CVACT, demonstrate that CVD consistently improves both localization accuracy and generalization across multiple baselines.
>
---
#### [new 158] Diff-MM: Exploring Pre-trained Text-to-Image Generation Model for Unified Multi-modal Object Tracking
- **分类: cs.CV**

- **简介: 该论文属于多模态目标跟踪任务，旨在解决现有方法因训练数据不足导致的性能受限问题。通过利用预训练文本-图像生成模型（Stable Diffusion）的UNet作为特征提取器，设计并行特征提取管道和多模态子模块调优方法，实现统一参数下的RGB-N/D/T/E多模态跟踪，实验显示性能显著提升。**

- **链接: [http://arxiv.org/pdf/2505.12606v1](http://arxiv.org/pdf/2505.12606v1)**

> **作者:** Shiyu Xuan; Zechao Li; Jinhui Tang
>
> **摘要:** Multi-modal object tracking integrates auxiliary modalities such as depth, thermal infrared, event flow, and language to provide additional information beyond RGB images, showing great potential in improving tracking stabilization in complex scenarios. Existing methods typically start from an RGB-based tracker and learn to understand auxiliary modalities only from training data. Constrained by the limited multi-modal training data, the performance of these methods is unsatisfactory. To alleviate this limitation, this work proposes a unified multi-modal tracker Diff-MM by exploiting the multi-modal understanding capability of the pre-trained text-to-image generation model. Diff-MM leverages the UNet of pre-trained Stable Diffusion as a tracking feature extractor through the proposed parallel feature extraction pipeline, which enables pairwise image inputs for object tracking. We further introduce a multi-modal sub-module tuning method that learns to gain complementary information between different modalities. By harnessing the extensive prior knowledge in the generation model, we achieve a unified tracker with uniform parameters for RGB-N/D/T/E tracking. Experimental results demonstrate the promising performance of our method compared with recently proposed trackers, e.g., its AUC outperforms OneTracker by 8.3% on TNL2K.
>
---
#### [new 159] Multi-Resolution Haar Network: Enhancing human motion prediction via Haar transform
- **分类: cs.CV**

- **简介: 该论文属于3D人体运动预测任务，旨在解决现有方法因忽视时空轴变化导致复杂动作预测不准的问题。提出了HaarMoDic网络，利用2D Haar变换将关节投影到高分辨率混合坐标，通过多分辨率Haar块同时提取时空信息，在Human3.6M数据集上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.12631v1](http://arxiv.org/pdf/2505.12631v1)**

> **作者:** Li Lin
>
> **摘要:** The 3D human pose is vital for modern computer vision and computer graphics, and its prediction has drawn attention in recent years. 3D human pose prediction aims at forecasting a human's future motion from the previous sequence. Ignoring that the arbitrariness of human motion sequences has a firm origin in transition in both temporal and spatial axes limits the performance of state-of-the-art methods, leading them to struggle with making precise predictions on complex cases, e.g., arbitrarily posing or greeting. To alleviate this problem, a network called HaarMoDic is proposed in this paper, which utilizes the 2D Haar transform to project joints to higher resolution coordinates where the network can access spatial and temporal information simultaneously. An ablation study proves that the significant contributing module within the HaarModic Network is the Multi-Resolution Haar (MR-Haar) block. Instead of mining in one of two axes or extracting separately, the MR-Haar block projects whole motion sequences to a mixed-up coordinate in higher resolution with 2D Haar Transform, allowing the network to give scope to information from both axes in different resolutions. With the MR-Haar block, the HaarMoDic network can make predictions referring to a broader range of information. Experimental results demonstrate that HaarMoDic surpasses state-of-the-art methods in every testing interval on the Human3.6M dataset in the Mean Per Joint Position Error (MPJPE) metric.
>
---
#### [new 160] ElderFallGuard: Real-Time IoT and Computer Vision-Based Fall Detection System for Elderly Safety
- **分类: cs.CV**

- **简介: 该论文针对老年人跌倒检测问题，提出基于计算机视觉和物联网的实时监测系统ElderFallGuard。通过MediaPipe姿态识别、自定义数据集训练随机森林模型，结合特定姿势持续时间和运动阈值触发报警，实现100%检测准确率并自动推送Telegram警报，保障老人安全。**

- **链接: [http://arxiv.org/pdf/2505.11845v1](http://arxiv.org/pdf/2505.11845v1)**

> **作者:** Tasrifur Riahi; Md. Azizul Hakim Bappy; Md. Mehedi Islam
>
> **备注:** 9 page, 1 table, 5 figure
>
> **摘要:** For the elderly population, falls pose a serious and increasing risk of serious injury and loss of independence. In order to overcome this difficulty, we present ElderFallGuard: A Computer Vision Based IoT Solution for Elderly Fall Detection and Notification, a cutting-edge, non-invasive system intended for quick caregiver alerts and real-time fall detection. Our approach leverages the power of computer vision, utilizing MediaPipe for accurate human pose estimation from standard video streams. We developed a custom dataset comprising 7200 samples across 12 distinct human poses to train and evaluate various machine learning classifiers, with Random Forest ultimately selected for its superior performance. ElderFallGuard employs a specific detection logic, identifying a fall when a designated prone pose ("Pose6") is held for over 3 seconds coupled with a significant drop in motion detected for more than 2 seconds. Upon confirmation, the system instantly dispatches an alert, including a snapshot of the event, to a designated Telegram group via a custom bot, incorporating cooldown logic to prevent notification overload. Rigorous testing on our dataset demonstrated exceptional results, achieving 100% accuracy, precision, recall, and F1-score. ElderFallGuard offers a promising, vision-based IoT solution to enhance elderly safety and provide peace of mind for caregivers through intelligent, timely alerts.
>
---
#### [new 161] SoftPQ: Robust Instance Segmentation Evaluation via Soft Matching and Tunable Thresholds
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对实例分割评估任务，解决传统指标依赖硬性IoU阈值导致误差区分度低的问题。提出SoftPQ方法，通过可调双阈值定义部分匹配区域，结合次线性惩罚函数，将评估转化为连续梯度，增强对结构误差的鲁棒性，提供更细粒度的模型优化反馈。**

- **链接: [http://arxiv.org/pdf/2505.12155v1](http://arxiv.org/pdf/2505.12155v1)**

> **作者:** Ranit Karmakar; Simon F. Nørrelykke
>
> **摘要:** Segmentation evaluation metrics traditionally rely on binary decision logic: predictions are either correct or incorrect, based on rigid IoU thresholds. Detection--based metrics such as F1 and mAP determine correctness at the object level using fixed overlap cutoffs, while overlap--based metrics like Intersection over Union (IoU) and Dice operate at the pixel level, often overlooking instance--level structure. Panoptic Quality (PQ) attempts to unify detection and segmentation assessment, but it remains dependent on hard-threshold matching--treating predictions below the threshold as entirely incorrect. This binary framing obscures important distinctions between qualitatively different errors and fails to reward gradual model improvements. We propose SoftPQ, a flexible and interpretable instance segmentation metric that redefines evaluation as a graded continuum rather than a binary classification. SoftPQ introduces tunable upper and lower IoU thresholds to define a partial matching region and applies a sublinear penalty function to ambiguous or fragmented predictions. These extensions allow SoftPQ to exhibit smoother score behavior, greater robustness to structural segmentation errors, and more informative feedback for model development and evaluation. Through controlled perturbation experiments, we show that SoftPQ captures meaningful differences in segmentation quality that existing metrics overlook, making it a practical and principled alternative for both benchmarking and iterative model refinement.
>
---
#### [new 162] Mamba-Adaptor: State Space Model Adaptor for Visual Recognition
- **分类: cs.CV**

- **简介: 该论文针对视觉任务中状态空间模型（如Mamba）存在的全局上下文缺失、长程遗忘和弱空间建模问题，提出Mamba-Adaptor适配器，包含Adaptor-T（记忆增强缓解遗忘）和Adaptor-S（多尺度卷积增强空间建模）。作为视觉主干网络/性能增强模块/迁移学习工具，在ImageNet和COCO取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2505.12685v1](http://arxiv.org/pdf/2505.12685v1)**

> **作者:** Fei Xie; Jiahao Nie; Yujin Tang; Wenkang Zhang; Hongshen Zhao
>
> **备注:** CVPR paper
>
> **摘要:** Recent State Space Models (SSM), especially Mamba, have demonstrated impressive performance in visual modeling and possess superior model efficiency. However, the application of Mamba to visual tasks suffers inferior performance due to three main constraints existing in the sequential model: 1) Casual computing is incapable of accessing global context; 2) Long-range forgetting when computing the current hidden states; 3) Weak spatial structural modeling due to the transformed sequential input. To address these issues, we investigate a simple yet powerful vision task Adaptor for Mamba models, which consists of two functional modules: Adaptor-T and Adaptor-S. When solving the hidden states for SSM, we apply a lightweight prediction module Adaptor-T to select a set of learnable locations as memory augmentations to ease long-range forgetting issues. Moreover, we leverage Adapator-S, composed of multi-scale dilated convolutional kernels, to enhance the spatial modeling and introduce the image inductive bias into the feature output. Both modules can enlarge the context modeling in casual computing, as the output is enhanced by the inaccessible features. We explore three usages of Mamba-Adaptor: A general visual backbone for various vision tasks; A booster module to raise the performance of pretrained backbones; A highly efficient fine-tuning module that adapts the base model for transfer learning tasks. Extensive experiments verify the effectiveness of Mamba-Adaptor in three settings. Notably, our Mamba-Adaptor achieves state-of the-art performance on the ImageNet and COCO benchmarks.
>
---
#### [new 163] Mitigating Hallucination in VideoLLMs via Temporal-Aware Activation Engineering
- **分类: cs.CV**

- **简介: 该论文针对视频多模态大语言模型（VideoLLMs）的幻觉问题（生成合理但错误的内容），提出时间感知激活工程框架。通过分析发现模型对幻觉的敏感性取决于时间变化而非任务类型，据此自适应调整关键模块，无需微调即可有效减少幻觉，实验验证了方法的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.12826v1](http://arxiv.org/pdf/2505.12826v1)**

> **作者:** Jianfeng Cai; Wengang Zhou; Zongmeng Zhang; Jiale Hong; Nianji Zhan; Houqiang Li
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable progress in video understanding.However, hallucination, where the model generates plausible yet incorrect outputs, persists as a significant and under-addressed challenge in the video domain. Among existing solutions, activation engineering has proven successful in mitigating hallucinations in LLMs and ImageLLMs, yet its applicability to VideoLLMs remains largely unexplored. In this work, we are the first to systematically investigate the effectiveness and underlying mechanisms of activation engineering for mitigating hallucinations in VideoLLMs. We initially conduct an investigation of the key factors affecting the performance of activation engineering and find that a model's sensitivity to hallucination depends on $\textbf{temporal variation}$ rather than task type. Moreover, selecting appropriate internal modules and dataset for activation engineering is critical for reducing hallucination. Guided by these findings, we propose a temporal-aware activation engineering framework for VideoLLMs, which adaptively identifies and manipulates hallucination-sensitive modules based on the temporal variation characteristic, substantially mitigating hallucinations without additional LLM fine-tuning. Experiments across multiple models and benchmarks demonstrate that our method markedly reduces hallucination in VideoLLMs, thereby validating the robustness of our findings.
>
---
#### [new 164] Faster Video Diffusion with Trainable Sparse Attention
- **分类: cs.CV**

- **简介: 该论文针对视频扩散模型（DiTs）因3D注意力计算复杂度过高的问题，提出可训练稀疏注意力机制VSA。通过粗-细两阶段动态筛选关键位置，在保持85%硬件效率的同时，将训练计算量降低2.53倍，生成速度提升近2倍，为视频扩散模型的高效扩展提供解决方案。**

- **链接: [http://arxiv.org/pdf/2505.13389v1](http://arxiv.org/pdf/2505.13389v1)**

> **作者:** Peiyuan Zhang; Haofeng Huang; Yongqi Chen; Will Lin; Zhengzhong Liu; Ion Stoica; Eric P. Xing; Hao Zhang
>
> **摘要:** Scaling video diffusion transformers (DiTs) is limited by their quadratic 3D attention, even though most of the attention mass concentrates on a small subset of positions. We turn this observation into VSA, a trainable, hardware-efficient sparse attention that replaces full attention at \emph{both} training and inference. In VSA, a lightweight coarse stage pools tokens into tiles and identifies high-weight \emph{critical tokens}; a fine stage computes token-level attention only inside those tiles subjecting to block computing layout to ensure hard efficiency. This leads to a single differentiable kernel that trains end-to-end, requires no post-hoc profiling, and sustains 85\% of FlashAttention3 MFU. We perform a large sweep of ablation studies and scaling-law experiments by pretraining DiTs from 60M to 1.4B parameters. VSA reaches a Pareto point that cuts training FLOPS by 2.53$\times$ with no drop in diffusion loss. Retrofitting the open-source Wan-2.1 model speeds up attention time by 6$\times$ and lowers end-to-end generation time from 31s to 18s with comparable quality. These results establish trainable sparse attention as a practical alternative to full attention and a key enabler for further scaling of video diffusion models.
>
---
#### [new 165] Denoising Mutual Knowledge Distillation in Bi-Directional Multiple Instance Learning
- **分类: cs.CV**

- **简介: 该论文针对数字病理学中全幻灯片图像分类任务，解决多示例学习（MIL）在包级/实例级分类时因噪声标签导致的精度不足问题。通过双向去噪互知识蒸馏框架，结合弱监督到强泛化的伪标签校正技术，增强双层次预测能力，实验证明其优于现有MIL方法。**

- **链接: [http://arxiv.org/pdf/2505.12074v1](http://arxiv.org/pdf/2505.12074v1)**

> **作者:** Chen Shu; Boyu Fu; Yiman Li; Ting Yin; Wenchuan Zhang; Jie Chen; Yuhao Yi; Hong Bu
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Multiple Instance Learning is the predominant method for Whole Slide Image classification in digital pathology, enabling the use of slide-level labels to supervise model training. Although MIL eliminates the tedious fine-grained annotation process for supervised learning, whether it can learn accurate bag- and instance-level classifiers remains a question. To address the issue, instance-level classifiers and instance masks were incorporated to ground the prediction on supporting patches. These methods, while practically improving the performance of MIL methods, may potentially introduce noisy labels. We propose to bridge the gap between commonly used MIL and fully supervised learning by augmenting both the bag- and instance-level learning processes with pseudo-label correction capabilities elicited from weak to strong generalization techniques. The proposed algorithm improves the performance of dual-level MIL algorithms on both bag- and instance-level predictions. Experiments on public pathology datasets showcase the advantage of the proposed methods.
>
---
#### [new 166] HiERO: understanding the hierarchy of human behavior enhances reasoning on egocentric videos
- **分类: cs.CV**

- **简介: 该论文属于第一人称视频理解任务，旨在解决复杂人类行为推理难题。提出HiERO弱监督方法，通过视频与文本对齐从非结构化数据中提取层次化活动模式，增强视频特征表示。该方法在多个视频-文本对齐基准和零样本流程学习任务中实现最优性能，显著超越全监督方法。**

- **链接: [http://arxiv.org/pdf/2505.12911v1](http://arxiv.org/pdf/2505.12911v1)**

> **作者:** Simone Alberto Peirone; Francesca Pistilli; Giuseppe Averta
>
> **备注:** Project page https://github.com/sapeirone/hiero
>
> **摘要:** Human activities are particularly complex and variable, and this makes challenging for deep learning models to reason about them. However, we note that such variability does have an underlying structure, composed of a hierarchy of patterns of related actions. We argue that such structure can emerge naturally from unscripted videos of human activities, and can be leveraged to better reason about their content. We present HiERO, a weakly-supervised method to enrich video segments features with the corresponding hierarchical activity threads. By aligning video clips with their narrated descriptions, HiERO infers contextual, semantic and temporal reasoning with an hierarchical architecture. We prove the potential of our enriched features with multiple video-text alignment benchmarks (EgoMCQ, EgoNLQ) with minimal additional training, and in zero-shot for procedure learning tasks (EgoProceL and Ego4D Goal-Step). Notably, HiERO achieves state-of-the-art performance in all the benchmarks, and for procedure learning tasks it outperforms fully-supervised methods by a large margin (+12.5% F1 on EgoProceL) in zero shot. Our results prove the relevance of using knowledge of the hierarchy of human activities for multiple reasoning tasks in egocentric vision.
>
---
#### [new 167] DPCD: A Quality Assessment Database for Dynamic Point Clouds
- **分类: cs.CV; cs.DB**

- **简介: 该论文属于动态点云质量评估任务，旨在解决该领域缺乏基准数据库的问题。作者构建了DPCD数据库，包含15个参考动态点云和525个失真样本，通过主观实验获得质量评分，并验证了数据库的可靠性与评估指标性能，为相关研究提供数据支持。**

- **链接: [http://arxiv.org/pdf/2505.12431v1](http://arxiv.org/pdf/2505.12431v1)**

> **作者:** Yating Liu; Yujie Zhang; Qi Yang; Yiling Xu; Zhu Li; Ye-Kui Wang
>
> **摘要:** Recently, the advancements in Virtual/Augmented Reality (VR/AR) have driven the demand for Dynamic Point Clouds (DPC). Unlike static point clouds, DPCs are capable of capturing temporal changes within objects or scenes, offering a more accurate simulation of the real world. While significant progress has been made in the quality assessment research of static point cloud, little study has been done on Dynamic Point Cloud Quality Assessment (DPCQA), which hinders the development of quality-oriented applications, such as interframe compression and transmission in practical scenarios. In this paper, we introduce a large-scale DPCQA database, named DPCD, which includes 15 reference DPCs and 525 distorted DPCs from seven types of lossy compression and noise distortion. By rendering these samples to Processed Video Sequences (PVS), a comprehensive subjective experiment is conducted to obtain Mean Opinion Scores (MOS) from 21 viewers for analysis. The characteristic of contents, impact of various distortions, and accuracy of MOSs are presented to validate the heterogeneity and reliability of the proposed database. Furthermore, we evaluate the performance of several objective metrics on DPCD. The experiment results show that DPCQA is more challenge than that of static point cloud. The DPCD, which serves as a catalyst for new research endeavors on DPCQA, is publicly available at https://huggingface.co/datasets/Olivialyt/DPCD.
>
---
#### [new 168] Top-Down Compression: Revisit Efficient Vision Token Projection for Visual Instruction Tuning
- **分类: cs.CV**

- **简介: 该论文研究视觉指令调优，解决视觉-语言投影中精度与效率的权衡问题。提出LLaVA-Meteor框架，通过全局融合模块和动态选择机制压缩75-95%视觉token，保持核心信息，在12个基准上实现高效高性能。**

- **链接: [http://arxiv.org/pdf/2505.11945v1](http://arxiv.org/pdf/2505.11945v1)**

> **作者:** Bonan li; Zicheng Zhang; Songhua Liu; Weihao Yu; Xinchao Wang
>
> **备注:** Under Review
>
> **摘要:** Visual instruction tuning aims to enable large language models to comprehend the visual world, with a pivotal challenge lying in establishing an effective vision-to-language projection. However, existing methods often grapple with the intractable trade-off between accuracy and efficiency. In this paper, we present LLaVA-Meteor, a novel approach designed to break this deadlock, equipped with a novel Top-Down Compression paradigm that strategically compresses visual tokens without compromising core information. Specifically, we construct a trainable Flash Global Fusion module based on efficient selective state space operators, which aligns the feature space while enabling each token to perceive holistic visual context and instruction preference at low cost. Furthermore, a local-to-single scanning manner is employed to effectively capture local dependencies, thereby enhancing the model's capability in vision modeling. To alleviate computational overhead, we explore a Visual-Native Selection mechanism that independently assesses token significance by both the visual and native experts, followed by aggregation to retain the most critical subset. Extensive experiments show that our approach reduces visual tokens by 75--95% while achieving comparable or superior performance across 12 benchmarks, significantly improving efficiency.
>
---
#### [new 169] LOVE: Benchmarking and Evaluating Text-to-Video Generation and Video-to-Text Interpretation
- **分类: cs.CV**

- **简介: 该论文属于多模态评估任务，旨在解决AI生成视频（AIGV）的感知质量与文本对齐不足问题。通过构建AIGVE-60K数据集（含6万人工标注），提出基于大模型的LOVE评估指标，从感知、对齐和任务准确性多维度衡量文本-视频双向生成与解析能力，验证其泛化性并开源资源。**

- **链接: [http://arxiv.org/pdf/2505.12098v1](http://arxiv.org/pdf/2505.12098v1)**

> **作者:** Jiarui Wang; Huiyu Duan; Ziheng Jia; Yu Zhao; Woo Yi Yang; Zicheng Zhang; Zijian Chen; Juntong Wang; Yuke Xing; Guangtao Zhai; Xiongkuo Min
>
> **摘要:** Recent advancements in large multimodal models (LMMs) have driven substantial progress in both text-to-video (T2V) generation and video-to-text (V2T) interpretation tasks. However, current AI-generated videos (AIGVs) still exhibit limitations in terms of perceptual quality and text-video alignment. Therefore, a reliable and scalable automatic model for AIGV evaluation is desirable, which heavily relies on the scale and quality of human annotations. To this end, we present AIGVE-60K, a comprehensive dataset and benchmark for AI-Generated Video Evaluation, which features (i) comprehensive tasks, encompassing 3,050 extensive prompts across 20 fine-grained task dimensions, (ii) the largest human annotations, including 120K mean-opinion scores (MOSs) and 60K question-answering (QA) pairs annotated on 58,500 videos generated from 30 T2V models, and (iii) bidirectional benchmarking and evaluating for both T2V generation and V2T interpretation capabilities. Based on AIGVE-60K, we propose LOVE, a LMM-based metric for AIGV Evaluation from multiple dimensions including perceptual preference, text-video correspondence, and task-specific accuracy in terms of both instance level and model level. Comprehensive experiments demonstrate that LOVE not only achieves state-of-the-art performance on the AIGVE-60K dataset, but also generalizes effectively to a wide range of other AIGV evaluation benchmarks. These findings highlight the significance of the AIGVE-60K dataset. Database and codes are anonymously available at https://github.com/IntMeGroup/LOVE.
>
---
#### [new 170] Can Large Multimodal Models Understand Agricultural Scenes? Benchmarking with AgroMind
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出农业遥感多模态基准AgroMind，解决现有数据集场景单一、任务简化问题。通过整合多源数据构建含25k问答的评测集，涵盖4维度13类任务，评估21个LMMs性能，揭示其在空间推理等领域的不足，建立标准化评估框架。**

- **链接: [http://arxiv.org/pdf/2505.12207v1](http://arxiv.org/pdf/2505.12207v1)**

> **作者:** Qingmei Li; Yang Zhang; Zurong Mai; Yuhang Chen; Shuohong Lou; Henglian Huang; Jiarui Zhang; Zhiwei Zhang; Yibin Wen; Weijia Li; Haohuan Fu; Jianxi Huang; Juepeng Zheng
>
> **摘要:** Large Multimodal Models (LMMs) has demonstrated capabilities across various domains, but comprehensive benchmarks for agricultural remote sensing (RS) remain scarce. Existing benchmarks designed for agricultural RS scenarios exhibit notable limitations, primarily in terms of insufficient scene diversity in the dataset and oversimplified task design. To bridge this gap, we introduce AgroMind, a comprehensive agricultural remote sensing benchmark covering four task dimensions: spatial perception, object understanding, scene understanding, and scene reasoning, with a total of 13 task types, ranging from crop identification and health monitoring to environmental analysis. We curate a high-quality evaluation set by integrating eight public datasets and one private farmland plot dataset, containing 25,026 QA pairs and 15,556 images. The pipeline begins with multi-source data preprocessing, including collection, format standardization, and annotation refinement. We then generate a diverse set of agriculturally relevant questions through the systematic definition of tasks. Finally, we employ LMMs for inference, generating responses, and performing detailed examinations. We evaluated 18 open-source LMMs and 3 closed-source models on AgroMind. Experiments reveal significant performance gaps, particularly in spatial reasoning and fine-grained recognition, it is notable that human performance lags behind several leading LMMs. By establishing a standardized evaluation framework for agricultural RS, AgroMind reveals the limitations of LMMs in domain knowledge and highlights critical challenges for future work. Data and code can be accessed at https://rssysu.github.io/AgroMind/.
>
---
#### [new 171] Improving Open-Set Semantic Segmentation in 3D Point Clouds by Conditional Channel Capacity Maximization: Preliminary Results
- **分类: cs.CV; eess.SP**

- **简介: 该论文研究开放集3D点云语义分割（O3S），解决现有闭集模型无法识别训练外类别的问题。提出条件通道容量最大化（3CM）正则化方法，通过最大化特征与预测的条件互信息，增强编码器保留类别相关特征的能力，从而提升未知类别的检测与分割效果。实验验证了方法的有效性，并探讨动态开放世界适应的未来方向。**

- **链接: [http://arxiv.org/pdf/2505.11521v1](http://arxiv.org/pdf/2505.11521v1)**

> **作者:** Wang Fang; Shirin Rahimi; Olivia Bennett; Sophie Carter; Mitra Hassani; Xu Lan; Omid Javadi; Lucas Mitchell
>
> **摘要:** Point-cloud semantic segmentation underpins a wide range of critical applications. Although recent deep architectures and large-scale datasets have driven impressive closed-set performance, these models struggle to recognize or properly segment objects outside their training classes. This gap has sparked interest in Open-Set Semantic Segmentation (O3S), where models must both correctly label known categories and detect novel, unseen classes. In this paper, we propose a plug and play framework for O3S. By modeling the segmentation pipeline as a conditional Markov chain, we derive a novel regularizer term dubbed Conditional Channel Capacity Maximization (3CM), that maximizes the mutual information between features and predictions conditioned on each class. When incorporated into standard loss functions, 3CM encourages the encoder to retain richer, label-dependent features, thereby enhancing the network's ability to distinguish and segment previously unseen categories. Experimental results demonstrate effectiveness of proposed method on detecting unseen objects. We further outline future directions for dynamic open-world adaptation and efficient information-theoretic estimation.
>
---
#### [new 172] Benchmarking Unified Face Attack Detection via Hierarchical Prompt Tuning
- **分类: cs.CV**

- **简介: 该论文研究统一人脸攻击检测任务，解决现有模型无法同时防御物理呈现攻击和数字深度伪造的问题。提出了大规模数据集UniAttackData+（含697k视频），并设计分层提示调优框架HiPTune，通过视觉提示树和动态语义融合，实现多语义空间自适应分类，在12个数据集验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.13327v1](http://arxiv.org/pdf/2505.13327v1)**

> **作者:** Ajian Liu; Haocheng Yuan; Xiao Guo; Hui Ma; Wanyi Zhuang; Changtao Miao; Yan Hong; Chuanbiao Song; Jun Lan; Qi Chu; Tao Gong; Yanyan Liang; Weiqiang Wang; Jun Wan; Xiaoming Liu; Zhen Lei
>
> **摘要:** Presentation Attack Detection and Face Forgery Detection are designed to protect face data from physical media-based Presentation Attacks and digital editing-based DeepFakes respectively. But separate training of these two models makes them vulnerable to unknown attacks and burdens deployment environments. The lack of a Unified Face Attack Detection model to handle both types of attacks is mainly due to two factors. First, there's a lack of adequate benchmarks for models to explore. Existing UAD datasets have limited attack types and samples, restricting the model's ability to address advanced threats. To address this, we propose UniAttackDataPlus (UniAttackData+), the most extensive and sophisticated collection of forgery techniques to date. It includes 2,875 identities and their 54 kinds of falsified samples, totaling 697,347 videos. Second, there's a lack of a reliable classification criterion. Current methods try to find an arbitrary criterion within the same semantic space, which fails when encountering diverse attacks. So, we present a novel Visual-Language Model-based Hierarchical Prompt Tuning Framework (HiPTune) that adaptively explores multiple classification criteria from different semantic spaces. We build a Visual Prompt Tree to explore various classification rules hierarchically. Then, by adaptively pruning the prompts, the model can select the most suitable prompts to guide the encoder to extract discriminative features at different levels in a coarse-to-fine way. Finally, to help the model understand the classification criteria in visual space, we propose a Dynamically Prompt Integration module to project the visual prompts to the text encoder for more accurate semantics. Experiments on 12 datasets have shown the potential to inspire further innovations in the UAD field.
>
---
#### [new 173] SSR: Enhancing Depth Perception in Vision-Language Models via Rationale-Guided Spatial Reasoning
- **分类: cs.CV**

- **简介: 该论文属于多模态视觉-语言模型的空间推理任务，旨在解决现有模型因依赖RGB输入导致的深度感知不足问题。提出SSR框架，将原始深度数据转化为结构化文本解释作为中间表征，并通过知识蒸馏压缩为轻量嵌入，实现无需重训练的高效集成。构建了百万级标注数据集SSR-CoT和评测基准SSRBench，实验验证其显著提升了空间推理能力。**

- **链接: [http://arxiv.org/pdf/2505.12448v1](http://arxiv.org/pdf/2505.12448v1)**

> **作者:** Yang Liu; Ming Ma; Xiaomin Yu; Pengxiang Ding; Han Zhao; Mingyang Sun; Siteng Huang; Donglin Wang
>
> **摘要:** Despite impressive advancements in Visual-Language Models (VLMs) for multi-modal tasks, their reliance on RGB inputs limits precise spatial understanding. Existing methods for integrating spatial cues, such as point clouds or depth, either require specialized sensors or fail to effectively exploit depth information for higher-order reasoning. To this end, we propose a novel Spatial Sense and Reasoning method, dubbed SSR, a novel framework that transforms raw depth data into structured, interpretable textual rationales. These textual rationales serve as meaningful intermediate representations to significantly enhance spatial reasoning capabilities. Additionally, we leverage knowledge distillation to compress the generated rationales into compact latent embeddings, which facilitate resource-efficient and plug-and-play integration into existing VLMs without retraining. To enable comprehensive evaluation, we introduce a new dataset named SSR-CoT, a million-scale visual-language reasoning dataset enriched with intermediate spatial reasoning annotations, and present SSRBench, a comprehensive multi-task benchmark. Extensive experiments on multiple benchmarks demonstrate SSR substantially improves depth utilization and enhances spatial reasoning, thereby advancing VLMs toward more human-like multi-modal understanding. Our project page is at https://yliu-cs.github.io/SSR.
>
---
#### [new 174] EarthSynth: Generating Informative Earth Observation with Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出EarthSynth扩散模型，解决遥感图像标注数据稀缺问题，属于数据生成任务。通过多类别跨卫星合成标注数据，采用反事实组合训练策略增强多样性，设计R-Filter筛选有效数据，并在分类、检测、分割任务验证效果，推动遥感图像解译发展。**

- **链接: [http://arxiv.org/pdf/2505.12108v1](http://arxiv.org/pdf/2505.12108v1)**

> **作者:** Jiancheng Pan; Shiye Lei; Yuqian Fu; Jiahao Li; Yanxing Liu; Yuze Sun; Xiao He; Long Peng; Xiaomeng Huang; Bo Zhao
>
> **备注:** 23 pages
>
> **摘要:** Remote sensing image (RSI) interpretation typically faces challenges due to the scarcity of labeled data, which limits the performance of RSI interpretation tasks. To tackle this challenge, we propose EarthSynth, a diffusion-based generative foundation model that enables synthesizing multi-category, cross-satellite labeled Earth observation for downstream RSI interpretation tasks. To the best of our knowledge, EarthSynth is the first to explore multi-task generation for remote sensing. EarthSynth, trained on the EarthSynth-180K dataset, employs the Counterfactual Composition training strategy to improve training data diversity and enhance category control. Furthermore, a rule-based method of R-Filter is proposed to filter more informative synthetic data for downstream tasks. We evaluate our EarthSynth on scene classification, object detection, and semantic segmentation in open-world scenarios, offering a practical solution for advancing RSI interpretation.
>
---
#### [new 175] Uniformity First: Uniformity-aware Test-time Adaptation of Vision-language Models against Image Corruption
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型（如CLIP）在传感器退化（如天气/噪声等图像损坏）下的测试时适应（TTA），解决分布偏移导致的性能下降问题。提出UnInfo方法，通过均匀性感知置信度最大化、信息平衡损失和EMA知识蒸馏，保留嵌入均匀性以提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.12912v1](http://arxiv.org/pdf/2505.12912v1)**

> **作者:** Kazuki Adachi; Shin'ya Yamaguchi; Tomoki Hamagami
>
> **备注:** Code is available at https://github.com/kzkadc/uninfo
>
> **摘要:** Pre-trained vision-language models such as contrastive language-image pre-training (CLIP) have demonstrated a remarkable generalizability, which has enabled a wide range of applications represented by zero-shot classification. However, vision-language models still suffer when they face datasets with large gaps from training ones, i.e., distribution shifts. We found that CLIP is especially vulnerable to sensor degradation, a type of realistic distribution shift caused by sensor conditions such as weather, light, or noise. Collecting a new dataset from a test distribution for fine-tuning highly costs since sensor degradation occurs unexpectedly and has a range of variety. Thus, we investigate test-time adaptation (TTA) of zero-shot classification, which enables on-the-fly adaptation to the test distribution with unlabeled test data. Existing TTA methods for CLIP mainly focus on modifying image and text embeddings or predictions to address distribution shifts. Although these methods can adapt to domain shifts, such as fine-grained labels spaces or different renditions in input images, they fail to adapt to distribution shifts caused by sensor degradation. We found that this is because image embeddings are "corrupted" in terms of uniformity, a measure related to the amount of information. To make models robust to sensor degradation, we propose a novel method called uniformity-aware information-balanced TTA (UnInfo). To address the corruption of image embeddings, we introduce uniformity-aware confidence maximization, information-aware loss balancing, and knowledge distillation from the exponential moving average (EMA) teacher. Through experiments, we demonstrate that our UnInfo improves accuracy under sensor degradation by retaining information in terms of uniformity.
>
---
#### [new 176] DB3D-L: Depth-aware BEV Feature Transformation for Accurate 3D Lane Detection
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的3D车道检测任务，旨在解决传统鸟瞰图（BEV）特征构建因缺乏深度信息导致的精度不足问题。通过设计融合深度估计的特征提取模块，结合降维与特征融合方法，有效整合前视图特征与深度信息，提升BEV构建准确性，在主流数据集上达到先进水平。**

- **链接: [http://arxiv.org/pdf/2505.13266v1](http://arxiv.org/pdf/2505.13266v1)**

> **作者:** Yehao Liu; Xiaosu Xu; Zijian Wang; Yiqing Yao
>
> **摘要:** 3D Lane detection plays an important role in autonomous driving. Recent advances primarily build Birds-Eye-View (BEV) feature from front-view (FV) images to perceive 3D information of Lane more effectively. However, constructing accurate BEV information from FV image is limited due to the lacking of depth information, causing previous works often rely heavily on the assumption of a flat ground plane. Leveraging monocular depth estimation to assist in constructing BEV features is less constrained, but existing methods struggle to effectively integrate the two tasks. To address the above issue, in this paper, an accurate 3D lane detection method based on depth-aware BEV feature transtormation is proposed. In detail, an effective feature extraction module is designed, in which a Depth Net is integrated to obtain the vital depth information for 3D perception, thereby simplifying the complexity of view transformation. Subquently a feature reduce module is proposed to reduce height dimension of FV features and depth features, thereby enables effective fusion of crucial FV features and depth features. Then a fusion module is designed to build BEV feature from prime FV feature and depth information. The proposed method performs comparably with state-of-the-art methods on both synthetic Apollo, realistic OpenLane datasets.
>
---
#### [new 177] TinyRS-R1: Compact Multimodal Language Model for Remote Sensing
- **分类: cs.CV**

- **简介: 该论文针对遥感领域大模型在边缘设备部署困难的问题，提出首个2B参数多模态小语言模型TinyRS-R1。通过四阶段训练（预训练、指令调优、思维链微调、GRPO对齐），模型在分类/视觉问答等任务上性能持平或超越7B模型，内存和延迟降低三分之二，兼具推理增强与实时响应能力，实现了遥感专用轻量级多模态推理。**

- **链接: [http://arxiv.org/pdf/2505.12099v1](http://arxiv.org/pdf/2505.12099v1)**

> **作者:** Aybora Koksal; A. Aydin Alatan
>
> **备注:** Submitted to BMVC 2025. Code, models, and the captions for datasets will be released
>
> **摘要:** Remote-sensing applications often run on edge hardware that cannot host today's 7B-parameter multimodal language models. This paper introduces TinyRS, the first 2B-parameter multimodal small language model (MSLM) optimized for remote sensing tasks, and TinyRS-R1, its reasoning-augmented variant. Built upon Qwen2-VL-2B, TinyRS is trained through a four-stage pipeline: pre-training on million satellite images, instruction tuning on visual instruction examples, fine-tuning with Chain-of-Thought (CoT) annotations from the proposed reasoning dataset, and alignment via Group Relative Policy Optimization (GRPO). TinyRS-R1 achieves or surpasses the performance of recent 7B-parameter remote sensing models across classification, VQA, visual grounding, and open-ended question answering-while requiring just one-third of the memory and latency. Our analysis shows that CoT reasoning substantially benefits spatial grounding and scene understanding, while the non-reasoning TinyRS excels in concise, latency-sensitive VQA tasks. TinyRS-R1 represents the first domain-specialized MSLM with GRPO-aligned CoT reasoning for general-purpose remote sensing.
>
---
#### [new 178] Expert-Like Reparameterization of Heterogeneous Pyramid Receptive Fields in Efficient CNNs for Fair Medical Image Classification
- **分类: cs.CV**

- **简介: 该论文针对高效CNN在医学图像分类中难以捕捉多样病灶特征及预测不公平的问题，提出ERoHPRF方法。通过异构金字塔感受野模拟多专家会诊模式提取不同病变特征，结合结构重参数化技术平衡计算效率，提升分类性能与公平性。**

- **链接: [http://arxiv.org/pdf/2505.13039v1](http://arxiv.org/pdf/2505.13039v1)**

> **作者:** Xiao Wu; Xiaoqing Zhang; Zunjie Xiao; Lingxi Hu; Risa Higashita; Jiang Liu
>
> **摘要:** Efficient convolutional neural network (CNN) architecture designs have attracted growing research interests. However, they usually apply single receptive field (RF), small asymmetric RFs, or pyramid RFs to learn different feature representations, still encountering two significant challenges in medical image classification tasks: 1) They have limitations in capturing diverse lesion characteristics efficiently, e.g., tiny, coordination, small and salient, which have unique roles on results, especially imbalanced medical image classification. 2) The predictions generated by those CNNs are often unfair/biased, bringing a high risk by employing them to real-world medical diagnosis conditions. To tackle these issues, we develop a new concept, Expert-Like Reparameterization of Heterogeneous Pyramid Receptive Fields (ERoHPRF), to simultaneously boost medical image classification performance and fairness. This concept aims to mimic the multi-expert consultation mode by applying the well-designed heterogeneous pyramid RF bags to capture different lesion characteristics effectively via convolution operations with multiple heterogeneous kernel sizes. Additionally, ERoHPRF introduces an expert-like structural reparameterization technique to merge its parameters with the two-stage strategy, ensuring competitive computation cost and inference speed through comparisons to a single RF. To manifest the effectiveness and generalization ability of ERoHPRF, we incorporate it into mainstream efficient CNN architectures. The extensive experiments show that our method maintains a better trade-off than state-of-the-art methods in terms of medical image classification, fairness, and computation overhead. The codes of this paper will be released soon.
>
---
#### [new 179] Advanced Integration of Discrete Line Segments in Digitized P&ID for Continuous Instrument Connectivity
- **分类: cs.CV**

- **简介: 该论文属于P&ID数字化处理任务，旨在解决人工映射效率低、易出错的问题。通过计算机视觉检测线段并整合，构建设备与管线的连接关系，形成数字化图表，存储为知识图谱以支持路径优化等应用。**

- **链接: [http://arxiv.org/pdf/2505.11976v1](http://arxiv.org/pdf/2505.11976v1)**

> **作者:** Soumya Swarup Prusty; Astha Agarwal; Srinivasan Iyenger
>
> **备注:** 6 pages, 13 figures
>
> **摘要:** Piping and Instrumentation Diagrams (P&IDs) constitute the foundational blueprint of a plant, depicting the interconnections among process equipment, instrumentation for process control, and the flow of fluids and control signals. In their existing setup, the manual mapping of information from P&ID sheets holds a significant challenge. This is a time-consuming process, taking around 3-6 months, and is susceptible to errors. It also depends on the expertise of the domain experts and often requires multiple rounds of review. The digitization of P&IDs entails merging detected line segments, which is essential for linking various detected instruments, thereby creating a comprehensive digitized P&ID. This paper focuses on explaining how line segments which are detected using a computer vision model are merged and eventually building the connection between equipment and merged lines. Hence presenting a digitized form of information stating the interconnection between process equipment, instrumentation, flow of fluids and control signals. Eventually, which can be stored in a knowledge graph and that information along with the help of advanced algorithms can be leveraged for tasks like finding optimal routes, detecting system cycles, computing transitive closures, and more.
>
---
#### [new 180] Semantically-Aware Game Image Quality Assessment
- **分类: cs.CV; eess.IV**

- **简介: 该论文针对游戏图像无参考质量评估任务，解决现有方法无法处理锯齿、纹理模糊等游戏特有失真的问题。提出结合知识蒸馏游戏失真特征提取器（GDFE）和CLIP语义门控的模型，通过训练不同画质预设数据，使质量评分贴合人类感知，在同类游戏中展现强泛化性和稳定性。**

- **链接: [http://arxiv.org/pdf/2505.11724v1](http://arxiv.org/pdf/2505.11724v1)**

> **作者:** Kai Zhu; Vignesh Edithal; Le Zhang; Ilia Blank; Imran Junejo
>
> **备注:** 16 pages, 12 figures
>
> **摘要:** Assessing the visual quality of video game graphics presents unique challenges due to the absence of reference images and the distinct types of distortions, such as aliasing, texture blur, and geometry level of detail (LOD) issues, which differ from those in natural images or user-generated content. Existing no-reference image and video quality assessment (NR-IQA/VQA) methods fail to generalize to gaming environments as they are primarily designed for distortions like compression artifacts. This study introduces a semantically-aware NR-IQA model tailored to gaming. The model employs a knowledge-distilled Game distortion feature extractor (GDFE) to detect and quantify game-specific distortions, while integrating semantic gating via CLIP embeddings to dynamically weight feature importance based on scene content. Training on gameplay data recorded across graphical quality presets enables the model to produce quality scores that align with human perception. Our results demonstrate that the GDFE, trained through knowledge distillation from binary classifiers, generalizes effectively to intermediate distortion levels unseen during training. Semantic gating further improves contextual relevance and reduces prediction variance. In the absence of in-domain NR-IQA baselines, our model outperforms out-of-domain methods and exhibits robust, monotonic quality trends across unseen games in the same genre. This work establishes a foundation for automated graphical quality assessment in gaming, advancing NR-IQA methods in this domain.
>
---
#### [new 181] Facial Recognition Leveraging Generative Adversarial Networks
- **分类: cs.CV; cs.CR**

- **简介: 该论文针对人脸识别中训练数据不足的问题，提出基于生成对抗网络的数据增强方法。通过残差生成器、改进的FaceNet判别器和端到端联合优化框架，提升小样本场景下的识别精度。实验表明LFW基准准确率提高12.7%，验证了方法的有效性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.11884v1](http://arxiv.org/pdf/2505.11884v1)**

> **作者:** Zhongwen Li; Zongwei Li; Xiaoqi Li
>
> **摘要:** Face recognition performance based on deep learning heavily relies on large-scale training data, which is often difficult to acquire in practical applications. To address this challenge, this paper proposes a GAN-based data augmentation method with three key contributions: (1) a residual-embedded generator to alleviate gradient vanishing/exploding problems, (2) an Inception ResNet-V1 based FaceNet discriminator for improved adversarial training, and (3) an end-to-end framework that jointly optimizes data generation and recognition performance. Experimental results demonstrate that our approach achieves stable training dynamics and significantly improves face recognition accuracy by 12.7% on the LFW benchmark compared to baseline methods, while maintaining good generalization capability with limited training samples.
>
---
#### [new 182] LogicOCR: Do Your Large Multimodal Models Excel at Logical Reasoning on Text-Rich Images?
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态逻辑推理评估任务，旨在解决大型多模态模型（LMMs）在文本密集图像上的复杂逻辑推理能力不足的问题。研究构建了LogicOCR基准数据集（含1,100多选题），通过自动化流程将公务员考试文本转化为多样化图文样本，并评估主流LMMs，揭示其视觉-文本推理的局限性。**

- **链接: [http://arxiv.org/pdf/2505.12307v1](http://arxiv.org/pdf/2505.12307v1)**

> **作者:** Maoyuan Ye; Jing Zhang; Juhua Liu; Bo Du; Dacheng Tao
>
> **备注:** GitHub: \url{https://github.com/MiliLab/LogicOCR}
>
> **摘要:** Recent advances in Large Multimodal Models (LMMs) have significantly improved their reasoning and Optical Character Recognition (OCR) capabilities. However, their performance on complex logical reasoning tasks involving text-rich images remains underexplored. To bridge this gap, we introduce LogicOCR, a benchmark comprising 1,100 multiple-choice questions designed to evaluate LMMs' logical reasoning abilities on text-rich images, while minimizing reliance on domain-specific knowledge (e.g., mathematics). We construct LogicOCR by curating a text corpus from the Chinese National Civil Servant Examination and develop a scalable, automated pipeline to convert it into multimodal samples. First, we design prompt templates to steer GPT-Image-1 to generate images with diverse backgrounds, interleaved text-illustration layouts, and varied fonts, ensuring contextual relevance and visual realism. Then, the generated images are manually verified, with low-quality examples discarded. We evaluate a range of representative open-source and proprietary LMMs under both Chain-of-Thought (CoT) and direct-answer settings. Our multi-dimensional analysis reveals key insights, such as the impact of test-time scaling, input modality differences, and sensitivity to visual-text orientation. Notably, LMMs still lag in multimodal reasoning compared to text-only inputs, indicating that they have not fully bridged visual reading with reasoning. We hope LogicOCR will serve as a valuable resource for advancing multimodal reasoning research. The dataset is available at https://github.com/MiliLab/LogicOCR.
>
---
#### [new 183] FinePhys: Fine-grained Human Action Generation by Explicitly Incorporating Physical Laws for Effective Skeletal Guidance
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，旨在解决生成符合物理规律的细粒度人类动作（如体操）的难题。提出FinePhys框架，通过结合数据驱动3D姿态估计与基于欧拉-拉格朗日方程的物理模拟，生成多尺度骨骼指导，利用扩散模型提升动作的自然性和物理合理性。**

- **链接: [http://arxiv.org/pdf/2505.13437v1](http://arxiv.org/pdf/2505.13437v1)**

> **作者:** Dian Shao; Mingfei Shi; Shengda Xu; Haodong Chen; Yongle Huang; Binglu Wang
>
> **备注:** CVPR 2025
>
> **摘要:** Despite significant advances in video generation, synthesizing physically plausible human actions remains a persistent challenge, particularly in modeling fine-grained semantics and complex temporal dynamics. For instance, generating gymnastics routines such as "switch leap with 0.5 turn" poses substantial difficulties for current methods, often yielding unsatisfactory results. To bridge this gap, we propose FinePhys, a Fine-grained human action generation framework that incorporates Physics to obtain effective skeletal guidance. Specifically, FinePhys first estimates 2D poses in an online manner and then performs 2D-to-3D dimension lifting via in-context learning. To mitigate the instability and limited interpretability of purely data-driven 3D poses, we further introduce a physics-based motion re-estimation module governed by Euler-Lagrange equations, calculating joint accelerations via bidirectional temporal updating. The physically predicted 3D poses are then fused with data-driven ones, offering multi-scale 2D heatmap guidance for the diffusion process. Evaluated on three fine-grained action subsets from FineGym (FX-JUMP, FX-TURN, and FX-SALTO), FinePhys significantly outperforms competitive baselines. Comprehensive qualitative results further demonstrate FinePhys's ability to generate more natural and plausible fine-grained human actions.
>
---
#### [new 184] Video-GPT via Next Clip Diffusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Video-GPT，将视频视为"视觉语言"，通过新型next clip扩散范式进行预训练，解决传统语言序列无法描述时空细节的问题。属于视频建模任务，结合自回归去噪实现短生成与长预测，在视频预测基准（34.97分）和6项下游任务中达到SOTA，展现跨任务泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.12489v1](http://arxiv.org/pdf/2505.12489v1)**

> **作者:** Shaobin Zhuang; Zhipeng Huang; Ying Zhang; Fangyikang Wang; Canmiao Fu; Binxin Yang; Chong Sun; Chen Li; Yali Wang
>
> **备注:** 22 pages, 12 figures, 18 tables
>
> **摘要:** GPT has shown its remarkable success in natural language processing. However, the language sequence is not sufficient to describe spatial-temporal details in the visual world. Alternatively, the video sequence is good at capturing such details. Motivated by this fact, we propose a concise Video-GPT in this paper by treating video as new language for visual world modeling. By analogy to next token prediction in GPT, we introduce a novel next clip diffusion paradigm for pretraining Video-GPT. Different from the previous works, this distinct paradigm allows Video-GPT to tackle both short-term generation and long-term prediction, by autoregressively denoising the noisy clip according to the clean clips in the history. Extensive experiments show our Video-GPT achieves the state-of-the-art performance on video prediction, which is the key factor towards world modeling (Physics-IQ Benchmark: Video-GPT 34.97 vs. Kling 23.64 vs. Wan 20.89). Moreover, it can be well adapted on 6 mainstream video tasks in both video generation and understanding, showing its great generalization capacity in downstream. The project page is at https://Video-GPT.github.io.
>
---
#### [new 185] AutoMat: Enabling Automated Crystal Structure Reconstruction from Microscopy via Agentic Tool Use
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于材料科学自动化任务，旨在解决实验晶体结构数据稀缺及电镜图像转原子结构效率低的问题。提出了AutoMat流程，通过智能代理整合去噪、模板检索、对称性重建等步骤，实现电镜图像到仿真结构的端到端转换，并开发专用基准测试STEM2Mat-Bench验证性能，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.12650v1](http://arxiv.org/pdf/2505.12650v1)**

> **作者:** Yaotian Yang; Yiwen Tang; Yizhe Chen; Xiao Chen; Jiangjie Qiu; Hao Xiong; Haoyu Yin; Zhiyao Luo; Yifei Zhang; Sijia Tao; Wentao Li; Qinghua Zhang; Yuqiang Li; Wanli Ouyang; Bin Zhao; Xiaonan Wang; Fei Wei
>
> **备注:** The code and dataset are publicly available at https://github.com/yyt-2378/AutoMat and https://huggingface.co/datasets/yaotianvector/STEM2Mat
>
> **摘要:** Machine learning-based interatomic potentials and force fields depend critically on accurate atomic structures, yet such data are scarce due to the limited availability of experimentally resolved crystals. Although atomic-resolution electron microscopy offers a potential source of structural data, converting these images into simulation-ready formats remains labor-intensive and error-prone, creating a bottleneck for model training and validation. We introduce AutoMat, an end-to-end, agent-assisted pipeline that automatically transforms scanning transmission electron microscopy (STEM) images into atomic crystal structures and predicts their physical properties. AutoMat combines pattern-adaptive denoising, physics-guided template retrieval, symmetry-aware atomic reconstruction, fast relaxation and property prediction via MatterSim, and coordinated orchestration across all stages. We propose the first dedicated STEM2Mat-Bench for this task and evaluate performance using lattice RMSD, formation energy MAE, and structure-matching success rate. By orchestrating external tool calls, AutoMat enables a text-only LLM to outperform vision-language models in this domain, achieving closed-loop reasoning throughout the pipeline. In large-scale experiments over 450 structure samples, AutoMat substantially outperforms existing multimodal large language models and tools. These results validate both AutoMat and STEM2Mat-Bench, marking a key step toward bridging microscopy and atomistic simulation in materials science.The code and dataset are publicly available at https://github.com/yyt-2378/AutoMat and https://huggingface.co/datasets/yaotianvector/STEM2Mat.
>
---
#### [new 186] Bootstrapping Diffusion: Diffusion Model Training Leveraging Partial and Corrupted Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于生成模型训练任务，旨在解决扩散模型依赖大量高质量数据的问题。通过利用低分辨率、带水印等残缺数据，提出分阶段训练方法：先为各残缺数据视图训练独立模型，再学习残差评分函数。理论证明该方法通过正则化可降低泛化误差，提升数据利用效率。**

- **链接: [http://arxiv.org/pdf/2505.11825v1](http://arxiv.org/pdf/2505.11825v1)**

> **作者:** Xudong Ma
>
> **备注:** 21 pages, 1 figure
>
> **摘要:** Training diffusion models requires large datasets. However, acquiring large volumes of high-quality data can be challenging, for example, collecting large numbers of high-resolution images and long videos. On the other hand, there are many complementary data that are usually considered corrupted or partial, such as low-resolution images and short videos. Other examples of corrupted data include videos that contain subtitles, watermarks, and logos. In this study, we investigate the theoretical problem of whether the above partial data can be utilized to train conventional diffusion models. Motivated by our theoretical analysis in this study, we propose a straightforward approach of training diffusion models utilizing partial data views, where we consider each form of complementary data as a view of conventional data. Our proposed approach first trains one separate diffusion model for each individual view, and then trains a model for predicting the residual score function. We prove generalization error bounds, which show that the proposed diffusion model training approach can achieve lower generalization errors if proper regularizations are adopted in the residual score function training. In particular, we prove that the difficulty in training the residual score function scales proportionally with the signal correlations not captured by partial data views. Consequently, the proposed approach achieves near first-order optimal data efficiency.
>
---
#### [new 187] Spectral-Spatial Self-Supervised Learning for Few-Shot Hyperspectral Image Classification
- **分类: cs.CV**

- **简介: 该论文针对小样本高光谱图像分类任务，解决标注稀缺、空间多样性适应及光谱先验不足问题。提出S4L-FSC方法，通过异构数据集旋转镜像自监督学习提取空间特征，结合同构数据集掩膜重建自监督学习获取光谱依赖，融合小样本学习嵌入光谱-空间先验知识，提升分类性能。**

- **链接: [http://arxiv.org/pdf/2505.12482v1](http://arxiv.org/pdf/2505.12482v1)**

> **作者:** Wenchen Chen; Yanmei Zhang; Zhongwei Xiao; Jianping Chu; Xingbo Wang
>
> **备注:** https://github.com/Wenchen-Chen/S4L-FSC
>
> **摘要:** Few-shot classification of hyperspectral images (HSI) faces the challenge of scarce labeled samples. Self-Supervised learning (SSL) and Few-Shot Learning (FSL) offer promising avenues to address this issue. However, existing methods often struggle to adapt to the spatial geometric diversity of HSIs and lack sufficient spectral prior knowledge. To tackle these challenges, we propose a method, Spectral-Spatial Self-Supervised Learning for Few-Shot Hyperspectral Image Classification (S4L-FSC), aimed at improving the performance of few-shot HSI classification. Specifically, we first leverage heterogeneous datasets to pretrain a spatial feature extractor using a designed Rotation-Mirror Self-Supervised Learning (RM-SSL) method, combined with FSL. This approach enables the model to learn the spatial geometric diversity of HSIs using rotation and mirroring labels as supervisory signals, while acquiring transferable spatial meta-knowledge through few-shot learning. Subsequently, homogeneous datasets are utilized to pretrain a spectral feature extractor via a combination of FSL and Masked Reconstruction Self-Supervised Learning (MR-SSL). The model learns to reconstruct original spectral information from randomly masked spectral vectors, inferring spectral dependencies. In parallel, FSL guides the model to extract pixel-level discriminative features, thereby embedding rich spectral priors into the model. This spectral-spatial pretraining method, along with the integration of knowledge from heterogeneous and homogeneous sources, significantly enhances model performance. Extensive experiments on four HSI datasets demonstrate the effectiveness and superiority of the proposed S4L-FSC approach for few-shot HSI classification.
>
---
#### [new 188] IQBench: How "Smart'' Are Vision-Language Models? A Study with Human IQ Tests
- **分类: cs.CV**

- **简介: 该论文提出IQBench基准，评估视觉语言模型(VLMs)在标准化视觉IQ测试中的流体推理能力，属于多模态智能评估任务。解决现有研究忽视模型推理过程、过度依赖文本的问题。通过人工构建500个防数据泄露的视觉问题，结合答案准确性、解释评估和人类评分分析模型。实验显示主流模型在3D空间和字谜推理存在显著缺陷，揭示推理过程与结果的不一致性。**

- **链接: [http://arxiv.org/pdf/2505.12000v1](http://arxiv.org/pdf/2505.12000v1)**

> **作者:** Tan-Hanh Pham; Phu-Vinh Nguyen; Dang The Hung; Bui Trong Duong; Vu Nguyen Thanh; Chris Ngo; Tri Quang Truong; Truong-Son Hy
>
> **备注:** IQ Test for Multimodal Models
>
> **摘要:** Although large Vision-Language Models (VLMs) have demonstrated remarkable performance in a wide range of multimodal tasks, their true reasoning capabilities on human IQ tests remain underexplored. To advance research on the fluid intelligence of VLMs, we introduce **IQBench**, a new benchmark designed to evaluate VLMs on standardized visual IQ tests. We focus on evaluating the reasoning capabilities of VLMs, which we argue are more important than the accuracy of the final prediction. **Our benchmark is visually centric, minimizing the dependence on unnecessary textual content**, thus encouraging models to derive answers primarily from image-based information rather than learned textual knowledge. To this end, we manually collected and annotated 500 visual IQ questions to **prevent unintentional data leakage during training**. Unlike prior work that focuses primarily on the accuracy of the final answer, we evaluate the reasoning ability of the models by assessing their explanations and the patterns used to solve each problem, along with the accuracy of the final prediction and human evaluation. Our experiments show that there are substantial performance disparities between tasks, with models such as `o4-mini`, `gemini-2.5-flash`, and `claude-3.7-sonnet` achieving the highest average accuracies of 0.615, 0.578, and 0.548, respectively. However, all models struggle with 3D spatial and anagram reasoning tasks, highlighting significant limitations in current VLMs' general reasoning abilities. In terms of reasoning scores, `o4-mini`, `gemini-2.5-flash`, and `claude-3.7-sonnet` achieved top averages of 0.696, 0.586, and 0.516, respectively. These results highlight inconsistencies between the reasoning processes of the models and their final answers, emphasizing the importance of evaluating the accuracy of the reasoning in addition to the final predictions.
>
---
#### [new 189] IA-MVS: Instance-Focused Adaptive Depth Sampling for Multi-View Stereo
- **分类: cs.CV**

- **简介: 该论文属于多视角立体视觉的深度估计任务，旨在解决现有方法因未利用实例级深度范围差异导致的精度不足及误差累积问题。提出IA-MVS方法，通过实例自适应深度采样缩小假设范围，结合深度连续性过滤机制和条件概率置信度模型，在MVSNet框架下实现高效优化，取得DTU基准最优性能。**

- **链接: [http://arxiv.org/pdf/2505.12714v1](http://arxiv.org/pdf/2505.12714v1)**

> **作者:** Yinzhe Wang; Yiwen Xiao; Hu Wang; Yiping Xu; Yan Tian
>
> **摘要:** Multi-view stereo (MVS) models based on progressive depth hypothesis narrowing have made remarkable advancements. However, existing methods haven't fully utilized the potential that the depth coverage of individual instances is smaller than that of the entire scene, which restricts further improvements in depth estimation precision. Moreover, inevitable deviations in the initial stage accumulate as the process advances. In this paper, we propose Instance-Adaptive MVS (IA-MVS). It enhances the precision of depth estimation by narrowing the depth hypothesis range and conducting refinement on each instance. Additionally, a filtering mechanism based on intra-instance depth continuity priors is incorporated to boost robustness. Furthermore, recognizing that existing confidence estimation can degrade IA-MVS performance on point clouds. We have developed a detailed mathematical model for confidence estimation based on conditional probability. The proposed method can be widely applied in models based on MVSNet without imposing extra training burdens. Our method achieves state-of-the-art performance on the DTU benchmark. The source code is available at https://github.com/KevinWang73106/IA-MVS.
>
---
#### [new 190] Kornia-rs: A Low-Level 3D Computer Vision Library In Rust
- **分类: cs.CV**

- **简介: 该论文提出基于Rust的高性能3D计算机视觉库Kornia-rs，针对现有C++库（如OpenCV）和包装方案存在的内存/线程安全隐患及Rust生态3D工具缺失问题。通过Rust所有权模型和静态类型张量系统构建底层库，提供模块化组件、Python接口及3D视觉算子，实现3~5倍图像处理加速，填补了Rust生态中3D视觉能力空白。**

- **链接: [http://arxiv.org/pdf/2505.12425v1](http://arxiv.org/pdf/2505.12425v1)**

> **作者:** Edgar Riba; Jian Shi; Aditya Kumar; Andrew Shen; Gary Bradski
>
> **摘要:** We present \textit{kornia-rs}, a high-performance 3D computer vision library written entirely in native Rust, designed for safety-critical and real-time applications. Unlike C++-based libraries like OpenCV or wrapper-based solutions like OpenCV-Rust, \textit{kornia-rs} is built from the ground up to leverage Rust's ownership model and type system for memory and thread safety. \textit{kornia-rs} adopts a statically-typed tensor system and a modular set of crates, providing efficient image I/O, image processing and 3D operations. To aid cross-platform compatibility, \textit{kornia-rs} offers Python bindings, enabling seamless and efficient integration with Rust code. Empirical results show that \textit{kornia-rs} achieves a 3~ 5 times speedup in image transformation tasks over native Rust alternatives, while offering comparable performance to C++ wrapper-based libraries. In addition to 2D vision capabilities, \textit{kornia-rs} addresses a significant gap in the Rust ecosystem by providing a set of 3D computer vision operators. This paper presents the architecture and performance characteristics of \textit{kornia-rs}, demonstrating its effectiveness in real-world computer vision applications.
>
---
#### [new 191] Always Clear Depth: Robust Monocular Depth Estimation under Adverse Weather
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于单目深度估计任务，旨在解决恶劣天气下模型性能下降问题。提出ACDepth方法，通过扩散模型生成多天气训练数据，结合知识蒸馏和对抗训练提升鲁棒性。创新点包括LoRA微调的数据生成、多粒度知识蒸馏策略及排序引导蒸馏机制，实验在nuScenes数据集上超越基线模型。**

- **链接: [http://arxiv.org/pdf/2505.12199v1](http://arxiv.org/pdf/2505.12199v1)**

> **作者:** Kui Jiang; Jing Cao; Zhaocheng Yu; Junjun Jiang; Jingchun Zhou
>
> **摘要:** Monocular depth estimation is critical for applications such as autonomous driving and scene reconstruction. While existing methods perform well under normal scenarios, their performance declines in adverse weather, due to challenging domain shifts and difficulties in extracting scene information. To address this issue, we present a robust monocular depth estimation method called \textbf{ACDepth} from the perspective of high-quality training data generation and domain adaptation. Specifically, we introduce a one-step diffusion model for generating samples that simulate adverse weather conditions, constructing a multi-tuple degradation dataset during training. To ensure the quality of the generated degradation samples, we employ LoRA adapters to fine-tune the generation weights of diffusion model. Additionally, we integrate circular consistency loss and adversarial training to guarantee the fidelity and naturalness of the scene contents. Furthermore, we elaborate on a multi-granularity knowledge distillation strategy (MKD) that encourages the student network to absorb knowledge from both the teacher model and pretrained Depth Anything V2. This strategy guides the student model in learning degradation-agnostic scene information from various degradation inputs. In particular, we introduce an ordinal guidance distillation mechanism (OGD) that encourages the network to focus on uncertain regions through differential ranking, leading to a more precise depth estimation. Experimental results demonstrate that our ACDepth surpasses md4all-DD by 2.50\% for night scene and 2.61\% for rainy scene on the nuScenes dataset in terms of the absRel metric.
>
---
#### [new 192] Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出自动化框架MONDAY，构建大规模跨平台移动OS导航数据集（313K标注帧），解决现有单OS数据集泛化性差的问题。通过OCR检测、UI识别及多步动作提取，自动生成任务数据，提升模型在未见OS上的性能（平均+18.11%），支持持续扩展。**

- **链接: [http://arxiv.org/pdf/2505.12632v1](http://arxiv.org/pdf/2505.12632v1)**

> **作者:** Yunseok Jang; Yeda Song; Sungryull Sohn; Lajanugen Logeswaran; Tiange Luo; Dong-Ki Kim; Kyunghoon Bae; Honglak Lee
>
> **备注:** CVPR 2025
>
> **摘要:** Recent advancements in Large Language Models (LLMs) and Vision-Language Models (VLMs) have sparked significant interest in developing GUI visual agents. We introduce MONDAY (Mobile OS Navigation Task Dataset for Agents from YouTube), a large-scale dataset of 313K annotated frames from 20K instructional videos capturing diverse real-world mobile OS navigation across multiple platforms. Models that include MONDAY in their pre-training phases demonstrate robust cross-platform generalization capabilities, consistently outperforming models trained on existing single OS datasets while achieving an average performance gain of 18.11%p on an unseen mobile OS platform. To enable continuous dataset expansion as mobile platforms evolve, we present an automated framework that leverages publicly available video content to create comprehensive task datasets without manual annotation. Our framework comprises robust OCR-based scene detection (95.04% F1score), near-perfect UI element detection (99.87% hit ratio), and novel multi-step action identification to extract reliable action sequences across diverse interface configurations. We contribute both the MONDAY dataset and our automated collection framework to facilitate future research in mobile OS navigation.
>
---
#### [new 193] Automatic Complementary Separation Pruning Toward Lightweight CNNs
- **分类: cs.CV**

- **简介: 该论文属于神经网络压缩任务，旨在解决传统剪枝方法依赖人工设定剪枝量、冗余度高的问题。提出ACSP方法，结合结构化剪枝与激活分析，通过构建组件分离能力图空间，利用互补选择和聚类算法自动保留关键组件，在保持精度的同时显著降低计算量，实现全自动轻量化CNN部署。**

- **链接: [http://arxiv.org/pdf/2505.13225v1](http://arxiv.org/pdf/2505.13225v1)**

> **作者:** David Levin; Gonen Singer
>
> **摘要:** In this paper, we present Automatic Complementary Separation Pruning (ACSP), a novel and fully automated pruning method for convolutional neural networks. ACSP integrates the strengths of both structured pruning and activation-based pruning, enabling the efficient removal of entire components such as neurons and channels while leveraging activations to identify and retain the most relevant components. Our approach is designed specifically for supervised learning tasks, where we construct a graph space that encodes the separation capabilities of each component with respect to all class pairs. By employing complementary selection principles and utilizing a clustering algorithm, ACSP ensures that the selected components maintain diverse and complementary separation capabilities, reducing redundancy and maintaining high network performance. The method automatically determines the optimal subset of components in each layer, utilizing a knee-finding algorithm to select the minimal subset that preserves performance without requiring user-defined pruning volumes. Extensive experiments on multiple architectures, including VGG-16, ResNet-50, and MobileNet-V2, across datasets like CIFAR-10, CIFAR-100, and ImageNet-1K, demonstrate that ACSP achieves competitive accuracy compared to other methods while significantly reducing computational costs. This fully automated approach not only enhances scalability but also makes ACSP especially practical for real-world deployment by eliminating the need for manually defining the pruning volume.
>
---
#### [new 194] PMQ-VE: Progressive Multi-Frame Quantization for Video Enhancement
- **分类: cs.CV**

- **简介: 该论文针对多帧视频增强任务，解决现有量化方法导致的性能下降和细节丢失问题。提出PMQ-VE框架，包含两阶段量化策略：BMFQ动态调整帧间量化范围，PMTD利用渐进蒸馏结合多精度教师提升低比特模型性能，在保持效率的同时实现SOTA效果。**

- **链接: [http://arxiv.org/pdf/2505.12266v1](http://arxiv.org/pdf/2505.12266v1)**

> **作者:** ZhanFeng Feng; Long Peng; Xin Di; Yong Guo; Wenbo Li; Yulun Zhang; Renjing Pei; Yang Wang; Yang Cao; Zheng-Jun Zha
>
> **摘要:** Multi-frame video enhancement tasks aim to improve the spatial and temporal resolution and quality of video sequences by leveraging temporal information from multiple frames, which are widely used in streaming video processing, surveillance, and generation. Although numerous Transformer-based enhancement methods have achieved impressive performance, their computational and memory demands hinder deployment on edge devices. Quantization offers a practical solution by reducing the bit-width of weights and activations to improve efficiency. However, directly applying existing quantization methods to video enhancement tasks often leads to significant performance degradation and loss of fine details. This stems from two limitations: (a) inability to allocate varying representational capacity across frames, which results in suboptimal dynamic range adaptation; (b) over-reliance on full-precision teachers, which limits the learning of low-bit student models. To tackle these challenges, we propose a novel quantization method for video enhancement: Progressive Multi-Frame Quantization for Video Enhancement (PMQ-VE). This framework features a coarse-to-fine two-stage process: Backtracking-based Multi-Frame Quantization (BMFQ) and Progressive Multi-Teacher Distillation (PMTD). BMFQ utilizes a percentile-based initialization and iterative search with pruning and backtracking for robust clipping bounds. PMTD employs a progressive distillation strategy with both full-precision and multiple high-bit (INT) teachers to enhance low-bit models' capacity and quality. Extensive experiments demonstrate that our method outperforms existing approaches, achieving state-of-the-art performance across multiple tasks and benchmarks.The code will be made publicly available at: https://github.com/xiaoBIGfeng/PMQ-VE.
>
---
#### [new 195] Online Iterative Self-Alignment for Radiology Report Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究放射学报告生成任务，解决现有方法因高质量标注数据有限导致的过拟合和泛化问题。提出在线迭代自对齐方法，通过自生成多样化数据、多目标评估与优化，迭代提升模型性能，实验表明其超越现有方法达到最优效果。**

- **链接: [http://arxiv.org/pdf/2505.11983v1](http://arxiv.org/pdf/2505.11983v1)**

> **作者:** Ting Xiao; Lei Shi; Yang Zhang; HaoFeng Yang; Zhe Wang; Chenjia Bai
>
> **备注:** Accepted by ACL 2025 Main
>
> **摘要:** Radiology Report Generation (RRG) is an important research topic for relieving radiologist' heavy workload. Existing RRG models mainly rely on supervised fine-tuning (SFT) based on different model architectures using data pairs of radiological images and corresponding radiologist-annotated reports. Recent research has shifted focus to post-training improvements, aligning RRG model outputs with human preferences using reinforcement learning (RL). However, the limited data coverage of high-quality annotated data poses risks of overfitting and generalization. This paper proposes a novel Online Iterative Self-Alignment (OISA) method for RRG that consists of four stages: self-generation of diverse data, self-evaluation for multi-objective preference data,self-alignment for multi-objective optimization and self-iteration for further improvement. Our approach allows for generating varied reports tailored to specific clinical objectives, enhancing the overall performance of the RRG model iteratively. Unlike existing methods, our frame-work significantly increases data quality and optimizes performance through iterative multi-objective optimization. Experimental results demonstrate that our method surpasses previous approaches, achieving state-of-the-art performance across multiple evaluation metrics.
>
---
#### [new 196] CHRIS: Clothed Human Reconstruction with Side View Consistency
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于三维着装人体重建任务，旨在解决单视角输入导致的侧视拓扑不合理及表面不一致问题。提出CHRIS框架：1）侧视法线判别器增强全局合理性；2）多对一梯度计算（M2O）通过聚合邻近点梯度实现局部表面平滑。实验表明其性能超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.12005v1](http://arxiv.org/pdf/2505.12005v1)**

> **作者:** Dong Liu; Yifan Yang; Zixiong Huang; Yuxin Gao; Mingkui Tan
>
> **备注:** ICME 2025
>
> **摘要:** Creating a realistic clothed human from a single-view RGB image is crucial for applications like mixed reality and filmmaking. Despite some progress in recent years, mainstream methods often fail to fully utilize side-view information, as the input single-view image contains front-view information only. This leads to globally unrealistic topology and local surface inconsistency in side views. To address these, we introduce Clothed Human Reconstruction with Side View Consistency, namely CHRIS, which consists of 1) A Side-View Normal Discriminator that enhances global visual reasonability by distinguishing the generated side-view normals from the ground truth ones; 2) A Multi-to-One Gradient Computation (M2O) that ensures local surface consistency. M2O calculates the gradient of a sampling point by integrating the gradients of the nearby points, effectively acting as a smooth operation. Experimental results demonstrate that CHRIS achieves state-of-the-art performance on public benchmarks and outperforms the prior work.
>
---
#### [new 197] Accelerate TarFlow Sampling with GS-Jacobi Iteration
- **分类: cs.CV**

- **简介: 该论文针对图像生成模型TarFlow采样速度慢的问题，提出基于GS-Jacidi迭代的加速方法。通过设计收敛排序指标(CRM)和初始猜测指标(IGM)，区分模型中关键/非关键模块，优化迭代次数和初始值配置。实验表明在保持生成质量(FID)的同时，采样效率最高提升5.32倍，实现了高效图像生成任务优化。**

- **链接: [http://arxiv.org/pdf/2505.12849v1](http://arxiv.org/pdf/2505.12849v1)**

> **作者:** Ben Liu; Zhen Qin
>
> **备注:** 17 pages, 7 figures, 5 tables
>
> **摘要:** Image generation models have achieved widespread applications. As an instance, the TarFlow model combines the transformer architecture with Normalizing Flow models, achieving state-of-the-art results on multiple benchmarks. However, due to the causal form of attention requiring sequential computation, TarFlow's sampling process is extremely slow. In this paper, we demonstrate that through a series of optimization strategies, TarFlow sampling can be greatly accelerated by using the Gauss-Seidel-Jacobi (abbreviated as GS-Jacobi) iteration method. Specifically, we find that blocks in the TarFlow model have varying importance: a small number of blocks play a major role in image generation tasks, while other blocks contribute relatively little; some blocks are sensitive to initial values and prone to numerical overflow, while others are relatively robust. Based on these two characteristics, we propose the Convergence Ranking Metric (CRM) and the Initial Guessing Metric (IGM): CRM is used to identify whether a TarFlow block is "simple" (converges in few iterations) or "tough" (requires more iterations); IGM is used to evaluate whether the initial value of the iteration is good. Experiments on four TarFlow models demonstrate that GS-Jacobi sampling can significantly enhance sampling efficiency while maintaining the quality of generated images (measured by FID), achieving speed-ups of 4.53x in Img128cond, 5.32x in AFHQ, 2.96x in Img64uncond, and 2.51x in Img64cond without degrading FID scores or sample quality. Code and checkpoints are accessible on https://github.com/encoreus/GS-Jacobi_for_TarFlow
>
---
#### [new 198] Technical Report for ICRA 2025 GOOSE 2D Semantic Segmentation Challenge: Boosting Off-Road Segmentation via Photometric Distortion and Exponential Moving Average
- **分类: cs.CV**

- **简介: 该论文针对非结构化越野场景的2D语义分割任务，旨在提升复杂户外环境下的分割精度。采用FlashInternImage-B主干网络与UPerNet解码器，结合光度畸变增强模拟光照变化，并通过权重指数移动平均提升泛化能力，在GOOSE验证集实现88.8% mIoU。**

- **链接: [http://arxiv.org/pdf/2505.11769v1](http://arxiv.org/pdf/2505.11769v1)**

> **作者:** Wonjune Kim; Lae-kyoung Lee; Su-Yong An
>
> **备注:** Winners of the GOOSE 2D Semantic Segmentation Challenge at the IEEE ICRA Workshop on Field Robotics 2025
>
> **摘要:** We report on the application of a high-capacity semantic segmentation pipeline to the GOOSE 2D Semantic Segmentation Challenge for unstructured off-road environments. Using a FlashInternImage-B backbone together with a UPerNet decoder, we adapt established techniques, rather than designing new ones, to the distinctive conditions of off-road scenes. Our training recipe couples strong photometric distortion augmentation (to emulate the wide lighting variations of outdoor terrain) with an Exponential Moving Average (EMA) of weights for better generalization. Using only the GOOSE training dataset, we achieve 88.8\% mIoU on the validation set.
>
---
#### [new 199] Self-NPO: Negative Preference Optimization of Diffusion Models by Simply Learning from Itself without Explicit Preference Annotations
- **分类: cs.CV**

- **简介: 该论文提出Self-NPO方法，属于扩散模型的负偏好优化任务，旨在解决传统方法依赖显式人工标注的问题。通过自学习机制从模型内部获取负偏好信号，无需外部标注或奖励模型训练，有效提升生成质量与人类偏好对齐，适用于SD1.5、SDXL等多种扩散模型。**

- **链接: [http://arxiv.org/pdf/2505.11777v1](http://arxiv.org/pdf/2505.11777v1)**

> **作者:** Fu-Yun Wang; Keqiang Sun; Yao Teng; Xihui Liu; Jiaming Song; Hongsheng Li
>
> **摘要:** Diffusion models have demonstrated remarkable success in various visual generation tasks, including image, video, and 3D content generation. Preference optimization (PO) is a prominent and growing area of research that aims to align these models with human preferences. While existing PO methods primarily concentrate on producing favorable outputs, they often overlook the significance of classifier-free guidance (CFG) in mitigating undesirable results. Diffusion-NPO addresses this gap by introducing negative preference optimization (NPO), training models to generate outputs opposite to human preferences and thereby steering them away from unfavorable outcomes. However, prior NPO approaches, including Diffusion-NPO, rely on costly and fragile procedures for obtaining explicit preference annotations (e.g., manual pairwise labeling or reward model training), limiting their practicality in domains where such data are scarce or difficult to acquire. In this work, we introduce Self-NPO, a Negative Preference Optimization approach that learns exclusively from the model itself, thereby eliminating the need for manual data labeling or reward model training. Moreover, our method is highly efficient and does not require exhaustive data sampling. We demonstrate that Self-NPO integrates seamlessly into widely used diffusion models, including SD1.5, SDXL, and CogVideoX, as well as models already optimized for human preferences, consistently enhancing both their generation quality and alignment with human preferences.
>
---
#### [new 200] Deep Unrolled Meta-Learning for Multi-Coil and Multi-Modality MRI with Adaptive Optimization
- **分类: math.OC; cs.CV**

- **简介: 该论文属于医学图像重建与合成任务，旨在解决多线圈MRI欠采样重建及跨模态合成中数据不足和泛化性差的问题。提出结合展开优化与元学习的深度网络，通过自适应优化步骤整合数据保真和正则化，并利用元知识快速适应新采样模式与模态组合，在公开数据集上显著提升了重建质量。**

- **链接: [http://arxiv.org/pdf/2505.11518v1](http://arxiv.org/pdf/2505.11518v1)**

> **作者:** Merham Fouladvand; Peuroly Batra
>
> **摘要:** We propose a unified deep meta-learning framework for accelerated magnetic resonance imaging (MRI) that jointly addresses multi-coil reconstruction and cross-modality synthesis. Motivated by the limitations of conventional methods in handling undersampled data and missing modalities, our approach unrolls a provably convergent optimization algorithm into a structured neural network architecture. Each phase of the network mimics a step of an adaptive forward-backward scheme with extrapolation, enabling the model to incorporate both data fidelity and nonconvex regularization in a principled manner. To enhance generalization across different acquisition settings, we integrate meta-learning, which enables the model to rapidly adapt to unseen sampling patterns and modality combinations using task-specific meta-knowledge. The proposed method is evaluated on the open source datasets, showing significant improvements in PSNR and SSIM over conventional supervised learning, especially under aggressive undersampling and domain shifts. Our results demonstrate the synergy of unrolled optimization, task-aware meta-learning, and modality fusion, offering a scalable and generalizable solution for real-world clinical MRI reconstruction.
>
---
#### [new 201] Observe-R1: Unlocking Reasoning Abilities of MLLMs with Dynamic Progressive Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出Observe-R1框架，通过动态渐进强化学习提升多模态大语言模型（MLLM）的推理能力。针对多模态数据适配和训练效率问题，构建难度分级的NeuraLadder数据集，设计视觉观察约束、简洁奖励机制及动态权重策略，实验证明其在推理链清晰度和任务性能上优于更大模型。**

- **链接: [http://arxiv.org/pdf/2505.12432v1](http://arxiv.org/pdf/2505.12432v1)**

> **作者:** Zirun Guo; Minjie Hong; Tao Jin
>
> **摘要:** Reinforcement Learning (RL) has shown promise in improving the reasoning abilities of Large Language Models (LLMs). However, the specific challenges of adapting RL to multimodal data and formats remain relatively unexplored. In this work, we present Observe-R1, a novel framework aimed at enhancing the reasoning capabilities of multimodal large language models (MLLMs). We draw inspirations from human learning progression--from simple to complex and easy to difficult, and propose a gradual learning paradigm for MLLMs. To this end, we construct the NeuraLadder dataset, which is organized and sampled according to the difficulty and complexity of data samples for RL training. To tackle multimodal tasks, we introduce a multimodal format constraint that encourages careful observation of images, resulting in enhanced visual abilities and clearer and more structured responses. Additionally, we implement a bonus reward system that favors concise, correct answers within a length constraint, alongside a dynamic weighting mechanism that prioritizes uncertain and medium-difficulty problems, ensuring that more informative samples have a greater impact on training. Our experiments with the Qwen2.5-VL-3B and Qwen2.5-VL-7B models on 20k samples from the NeuraLadder dataset show that Observe-R1 outperforms a series of larger reasoning models on both reasoning and general benchmarks, achieving superior clarity and conciseness in reasoning chains. Ablation studies validate the effectiveness of our strategies, highlighting the robustness and generalization of our approach. The dataset and code will be released at https://github.com/zrguo/Observe-R1.
>
---
#### [new 202] Joint Embedding vs Reconstruction: Provable Benefits of Latent Space Prediction for Self Supervised Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于自监督学习领域，对比联合嵌入与重建两种方法的核心机制。通过理论分析，揭示了数据增强对表征学习的影响，证明在无关特征干扰大时，联合嵌入因更弱的对齐条件优于重建方法，为方法选择提供理论依据。**

- **链接: [http://arxiv.org/pdf/2505.12477v1](http://arxiv.org/pdf/2505.12477v1)**

> **作者:** Hugues Van Assel; Mark Ibrahim; Tommaso Biancalani; Aviv Regev; Randall Balestriero
>
> **备注:** 33 pages, 9 figures
>
> **摘要:** Reconstruction and joint embedding have emerged as two leading paradigms in Self Supervised Learning (SSL). Reconstruction methods focus on recovering the original sample from a different view in input space. On the other hand, joint embedding methods align the representations of different views in latent space. Both approaches offer compelling advantages, yet practitioners lack clear guidelines for choosing between them. In this work, we unveil the core mechanisms that distinguish each paradigm. By leveraging closed form solutions for both approaches, we precisely characterize how the view generation process, e.g. data augmentation, impacts the learned representations. We then demonstrate that, unlike supervised learning, both SSL paradigms require a minimal alignment between augmentations and irrelevant features to achieve asymptotic optimality with increasing sample size. Our findings indicate that in scenarios where these irrelevant features have a large magnitude, joint embedding methods are preferable because they impose a strictly weaker alignment condition compared to reconstruction based methods. These results not only clarify the trade offs between the two paradigms but also substantiate the empirical success of joint embedding approaches on real world challenging datasets.
>
---
#### [new 203] Mean Flows for One-step Generative Modeling
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究一步生成建模，旨在提升单步模型的性能。提出"平均速度"概念替代传统瞬时速度，建立两者数学关系指导网络训练。MeanFlow无需预训练，在ImageNet 256×256上实现3.43 FID，显著优于现有单步方法，缩小了与多步模型的性能差距。**

- **链接: [http://arxiv.org/pdf/2505.13447v1](http://arxiv.org/pdf/2505.13447v1)**

> **作者:** Zhengyang Geng; Mingyang Deng; Xingjian Bai; J. Zico Kolter; Kaiming He
>
> **备注:** Tech report
>
> **摘要:** We propose a principled and effective framework for one-step generative modeling. We introduce the notion of average velocity to characterize flow fields, in contrast to instantaneous velocity modeled by Flow Matching methods. A well-defined identity between average and instantaneous velocities is derived and used to guide neural network training. Our method, termed the MeanFlow model, is self-contained and requires no pre-training, distillation, or curriculum learning. MeanFlow demonstrates strong empirical performance: it achieves an FID of 3.43 with a single function evaluation (1-NFE) on ImageNet 256x256 trained from scratch, significantly outperforming previous state-of-the-art one-step diffusion/flow models. Our study substantially narrows the gap between one-step diffusion/flow models and their multi-step predecessors, and we hope it will motivate future research to revisit the foundations of these powerful models.
>
---
#### [new 204] BrainNetMLP: An Efficient and Effective Baseline for Functional Brain Network Classification
- **分类: q-bio.NC; cs.CV**

- **简介: 该论文属于功能性脑网络分类任务，针对复杂模型性能提升有限的问题，提出纯MLP架构BrainNetMLP。通过双分支结构融合时空特征，在HCP和ABIDE数据集上实现高效且高精度的分类，证明简单模型可超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11538v1](http://arxiv.org/pdf/2505.11538v1)**

> **作者:** Jiacheng Hou; Zhenjie Song; Ercan Engin Kuruoglu
>
> **备注:** V1.0
>
> **摘要:** Recent studies have made great progress in functional brain network classification by modeling the brain as a network of Regions of Interest (ROIs) and leveraging their connections to understand brain functionality and diagnose mental disorders. Various deep learning architectures, including Convolutional Neural Networks, Graph Neural Networks, and the recent Transformer, have been developed. However, despite the increasing complexity of these models, the performance gain has not been as salient. This raises a question: Does increasing model complexity necessarily lead to higher classification accuracy? In this paper, we revisit the simplest deep learning architecture, the Multi-Layer Perceptron (MLP), and propose a pure MLP-based method, named BrainNetMLP, for functional brain network classification, which capitalizes on the advantages of MLP, including efficient computation and fewer parameters. Moreover, BrainNetMLP incorporates a dual-branch structure to jointly capture both spatial connectivity and spectral information, enabling precise spatiotemporal feature fusion. We evaluate our proposed BrainNetMLP on two public and popular brain network classification datasets, the Human Connectome Project (HCP) and the Autism Brain Imaging Data Exchange (ABIDE). Experimental results demonstrate pure MLP-based methods can achieve state-of-the-art performance, revealing the potential of MLP-based models as more efficient yet effective alternatives in functional brain network classification. The code will be available at https://github.com/JayceonHo/BrainNetMLP.
>
---
#### [new 205] Structure-based Anomaly Detection and Clustering
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文研究无监督异常检测与聚类，解决结构化/流数据中偏离低维流形的异常识别问题。提出PIF（含Voronoi/RuzHash变体）进行流形嵌入与离群点隔离，Sliding-PIF处理流数据，MultiLink实现多模型聚类，Online-iForest用于动态流检测，并改进恶意软件开集识别方法。**

- **链接: [http://arxiv.org/pdf/2505.12751v1](http://arxiv.org/pdf/2505.12751v1)**

> **作者:** Filippo Leveni
>
> **备注:** Doctoral dissertation at Politecnico di Milano
>
> **摘要:** Anomaly detection is a fundamental problem in domains such as healthcare, manufacturing, and cybersecurity. This thesis proposes new unsupervised methods for anomaly detection in both structured and streaming data settings. In the first part, we focus on structure-based anomaly detection, where normal data follows low-dimensional manifolds while anomalies deviate from them. We introduce Preference Isolation Forest (PIF), which embeds data into a high-dimensional preference space via manifold fitting, and isolates outliers using two variants: Voronoi-iForest, based on geometric distances, and RuzHash-iForest, leveraging Locality Sensitive Hashing for scalability. We also propose Sliding-PIF, which captures local manifold information for streaming scenarios. Our methods outperform existing techniques on synthetic and real datasets. We extend this to structure-based clustering with MultiLink, a novel method for recovering multiple geometric model families in noisy data. MultiLink merges clusters via a model-aware linkage strategy, enabling robust multi-class structure recovery. It offers key advantages over existing approaches, such as speed, reduced sensitivity to thresholds, and improved robustness to poor initial sampling. The second part of the thesis addresses online anomaly detection in evolving data streams. We propose Online Isolation Forest (Online-iForest), which uses adaptive, multi-resolution histograms and dynamically updates tree structures to track changes over time. It avoids retraining while achieving accuracy comparable to offline models, with superior efficiency for real-time applications. Finally, we tackle anomaly detection in cybersecurity via open-set recognition for malware classification. We enhance a Gradient Boosting classifier with MaxLogit to detect unseen malware families, a method now integrated into Cleafy's production system.
>
---
#### [new 206] SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training
- **分类: cs.LG; cs.AI; cs.AR; cs.CV; cs.PF**

- **简介: 该论文针对注意力机制二次计算复杂度问题，提出高效优化方法。通过FP4量化在Blackwell GPU实现推理加速（5倍于FlashAttention），并首次将低比特训练引入注意力机制，设计8位前反向传播，验证了微调无损、预训练收敛慢的特性，属于高效深度学习优化任务。**

- **链接: [http://arxiv.org/pdf/2505.11594v1](http://arxiv.org/pdf/2505.11594v1)**

> **作者:** Jintao Zhang; Jia Wei; Pengle Zhang; Xiaoming Xu; Haofeng Huang; Haoxu Wang; Kai Jiang; Jun Zhu; Jianfei Chen
>
> **摘要:** The efficiency of attention is important due to its quadratic time complexity. We enhance the efficiency of attention through two key contributions: First, we leverage the new FP4 Tensor Cores in Blackwell GPUs to accelerate attention computation. Our implementation achieves 1038 TOPS on RTX5090, which is a 5x speedup over the fastest FlashAttention on RTX5090. Experiments show that our FP4 attention can accelerate inference of various models in a plug-and-play way. Second, we pioneer low-bit attention to training tasks. Existing low-bit attention works like FlashAttention3 and SageAttention focus only on inference. However, the efficiency of training large models is also important. To explore whether low-bit attention can be effectively applied to training tasks, we design an accurate and efficient 8-bit attention for both forward and backward propagation. Experiments indicate that 8-bit attention achieves lossless performance in fine-tuning tasks but exhibits slower convergence in pretraining tasks. The code will be available at https://github.com/thu-ml/SageAttention.
>
---
#### [new 207] Joint Manifold Learning and Optimal Transport for Dynamic Imaging
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对动态成像中时间序列数据稀缺问题，提出联合低维流形学习和最优传输（OT）正则化的方法，属于动态图像建模任务。通过构建潜在模型整合多时间序列信息与时间演化先验，解决现有方法孤立处理时空约束的缺陷，提升图像动态过程的建模与插值效果。**

- **链接: [http://arxiv.org/pdf/2505.11913v1](http://arxiv.org/pdf/2505.11913v1)**

> **作者:** Sven Dummer; Puru Vaish; Christoph Brune
>
> **摘要:** Dynamic imaging is critical for understanding and visualizing dynamic biological processes in medicine and cell biology. These applications often encounter the challenge of a limited amount of time series data and time points, which hinders learning meaningful patterns. Regularization methods provide valuable prior knowledge to address this challenge, enabling the extraction of relevant information despite the scarcity of time-series data and time points. In particular, low-dimensionality assumptions on the image manifold address sample scarcity, while time progression models, such as optimal transport (OT), provide priors on image development to mitigate the lack of time points. Existing approaches using low-dimensionality assumptions disregard a temporal prior but leverage information from multiple time series. OT-prior methods, however, incorporate the temporal prior but regularize only individual time series, ignoring information from other time series of the same image modality. In this work, we investigate the effect of integrating a low-dimensionality assumption of the underlying image manifold with an OT regularizer for time-evolving images. In particular, we propose a latent model representation of the underlying image manifold and promote consistency between this representation, the time series data, and the OT prior on the time-evolving images. We discuss the advantages of enriching OT interpolations with latent models and integrating OT priors into latent models.
>
---
#### [new 208] Advancing Generalization Across a Variety of Abstract Visual Reasoning Tasks
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文研究抽象视觉推理任务（AVR），聚焦提升模型在新数据分布（o.o.d.）下的泛化能力。针对现有方法在跨分布测试中的不足，提出PoNG模型，采用组卷积、归一化和并行结构设计，在Raven矩阵等多样基准测试中验证其优于现有方法的泛化性能。**

- **链接: [http://arxiv.org/pdf/2505.13391v1](http://arxiv.org/pdf/2505.13391v1)**

> **作者:** Mikołaj Małkiński; Jacek Mańdziuk
>
> **备注:** Accepted to the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025)
>
> **摘要:** The abstract visual reasoning (AVR) domain presents a diverse suite of analogy-based tasks devoted to studying model generalization. Recent years have brought dynamic progress in the field, particularly in i.i.d. scenarios, in which models are trained and evaluated on the same data distributions. Nevertheless, o.o.d. setups that assess model generalization to new test distributions remain challenging even for the most recent models. To advance generalization in AVR tasks, we present the Pathways of Normalized Group Convolution model (PoNG), a novel neural architecture that features group convolution, normalization, and a parallel design. We consider a wide set of AVR benchmarks, including Raven's Progressive Matrices and visual analogy problems with both synthetic and real-world images. The experiments demonstrate strong generalization capabilities of the proposed model, which in several settings outperforms the existing literature methods.
>
---
#### [new 209] On the Mechanisms of Adversarial Data Augmentation for Robust and Adaptive Transfer Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究对抗性数据增强在迁移学习中的作用，属于跨领域自适应任务。针对分布偏移导致的模型脆弱性问题，提出将对抗样本转化为增强工具，通过统一框架结合正则化与域不变表征学习，提升目标域在无监督/小样本场景下的性能，实验验证其增强决策边界和抑制源域过拟合的效果。**

- **链接: [http://arxiv.org/pdf/2505.12681v1](http://arxiv.org/pdf/2505.12681v1)**

> **作者:** Hana Satou; Alan Mitkiy
>
> **摘要:** Transfer learning across domains with distribution shift remains a fundamental challenge in building robust and adaptable machine learning systems. While adversarial perturbations are traditionally viewed as threats that expose model vulnerabilities, recent studies suggest that they can also serve as constructive tools for data augmentation. In this work, we systematically investigate the role of adversarial data augmentation (ADA) in enhancing both robustness and adaptivity in transfer learning settings. We analyze how adversarial examples, when used strategically during training, improve domain generalization by enriching decision boundaries and reducing overfitting to source-domain-specific features. We further propose a unified framework that integrates ADA with consistency regularization and domain-invariant representation learning. Extensive experiments across multiple benchmark datasets -- including VisDA, DomainNet, and Office-Home -- demonstrate that our method consistently improves target-domain performance under both unsupervised and few-shot domain adaptation settings. Our results highlight a constructive perspective of adversarial learning, transforming perturbation from a destructive attack into a regularizing force for cross-domain transferability.
>
---
#### [new 210] Emergent Active Perception and Dexterity of Simulated Humanoids from Visual Reinforcement Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究视觉强化学习驱动的模拟人形机器人全身灵巧控制任务，旨在解决传统方法依赖特权状态信息的问题。提出PDC框架，仅通过第一视角视觉输入实现物体搜索、抓取放置等家庭任务，无需3D物体信息。通过强化学习训练单一策略，涌现出主动搜索等类人行为，验证了视觉驱动控制在感知-行动闭环中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.12278v1](http://arxiv.org/pdf/2505.12278v1)**

> **作者:** Zhengyi Luo; Chen Tessler; Toru Lin; Ye Yuan; Tairan He; Wenli Xiao; Yunrong Guo; Gal Chechik; Kris Kitani; Linxi Fan; Yuke Zhu
>
> **备注:** Project page: https://zhengyiluo.github.io/PDC
>
> **摘要:** Human behavior is fundamentally shaped by visual perception -- our ability to interact with the world depends on actively gathering relevant information and adapting our movements accordingly. Behaviors like searching for objects, reaching, and hand-eye coordination naturally emerge from the structure of our sensory system. Inspired by these principles, we introduce Perceptive Dexterous Control (PDC), a framework for vision-driven dexterous whole-body control with simulated humanoids. PDC operates solely on egocentric vision for task specification, enabling object search, target placement, and skill selection through visual cues, without relying on privileged state information (e.g., 3D object positions and geometries). This perception-as-interface paradigm enables learning a single policy to perform multiple household tasks, including reaching, grasping, placing, and articulated object manipulation. We also show that training from scratch with reinforcement learning can produce emergent behaviors such as active search. These results demonstrate how vision-driven control and complex tasks induce human-like behaviors and can serve as the key ingredients in closing the perception-action loop for animation, robotics, and embodied AI.
>
---
#### [new 211] MM-PRM: Enhancing Multimodal Mathematical Reasoning with Scalable Step-Level Supervision
- **分类: cs.AI; cs.CV**

- **简介: 该论文研究多模态数学推理任务，针对现有模型中间步骤监督不足导致逻辑错误的问题，提出MM-PRM方法。通过构建MM-K12数据集和蒙特卡洛树搜索自动生成70万步骤标注，训练过程奖励模型优化推理路径选择，在多个数学基准测试中显著提升性能，验证了过程监督对增强多模态推理鲁棒性的有效性。**

- **链接: [http://arxiv.org/pdf/2505.13427v1](http://arxiv.org/pdf/2505.13427v1)**

> **作者:** Lingxiao Du; Fanqing Meng; Zongkai Liu; Zhixiang Zhou; Ping Luo; Qiaosheng Zhang; Wenqi Shao
>
> **摘要:** While Multimodal Large Language Models (MLLMs) have achieved impressive progress in vision-language understanding, they still struggle with complex multi-step reasoning, often producing logically inconsistent or partially correct solutions. A key limitation lies in the lack of fine-grained supervision over intermediate reasoning steps. To address this, we propose MM-PRM, a process reward model trained within a fully automated, scalable framework. We first build MM-Policy, a strong multimodal model trained on diverse mathematical reasoning data. Then, we construct MM-K12, a curated dataset of 10,000 multimodal math problems with verifiable answers, which serves as seed data. Leveraging a Monte Carlo Tree Search (MCTS)-based pipeline, we generate over 700k step-level annotations without human labeling. The resulting PRM is used to score candidate reasoning paths in the Best-of-N inference setup and achieves significant improvements across both in-domain (MM-K12 test set) and out-of-domain (OlympiadBench, MathVista, etc.) benchmarks. Further analysis confirms the effectiveness of soft labels, smaller learning rates, and path diversity in optimizing PRM performance. MM-PRM demonstrates that process supervision is a powerful tool for enhancing the logical robustness of multimodal reasoning systems. We release all our codes and data at https://github.com/ModalMinds/MM-PRM.
>
---
#### [new 212] CTLformer: A Hybrid Denoising Model Combining Convolutional Layers and Self-Attention for Enhanced CT Image Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于低剂量CT图像去噪任务，旨在解决多尺度特征融合与噪声分布复杂性问题。提出CTLformer模型，结合卷积与自注意力机制，创新设计多尺度注意力（捕捉细节/全局特征）和动态注意力控制（按噪声区域调整权重），并采用重叠推理减少边界伪影。实验表明其在Mayo Clinic数据集上性能优于现有方法，提升图像质量与诊断精度。**

- **链接: [http://arxiv.org/pdf/2505.12203v1](http://arxiv.org/pdf/2505.12203v1)**

> **作者:** Zhiting Zheng; Shuqi Wu; Wen Ding
>
> **摘要:** Low-dose CT (LDCT) images are often accompanied by significant noise, which negatively impacts image quality and subsequent diagnostic accuracy. To address the challenges of multi-scale feature fusion and diverse noise distribution patterns in LDCT denoising, this paper introduces an innovative model, CTLformer, which combines convolutional structures with transformer architecture. Two key innovations are proposed: a multi-scale attention mechanism and a dynamic attention control mechanism. The multi-scale attention mechanism, implemented through the Token2Token mechanism and self-attention interaction modules, effectively captures both fine details and global structures at different scales, enhancing relevant features and suppressing noise. The dynamic attention control mechanism adapts the attention distribution based on the noise characteristics of the input image, focusing on high-noise regions while preserving details in low-noise areas, thereby enhancing robustness and improving denoising performance. Furthermore, CTLformer integrates convolutional layers for efficient feature extraction and uses overlapping inference to mitigate boundary artifacts, further strengthening its denoising capability. Experimental results on the 2016 National Institutes of Health AAPM Mayo Clinic LDCT Challenge dataset demonstrate that CTLformer significantly outperforms existing methods in both denoising performance and model efficiency, greatly improving the quality of LDCT images. The proposed CTLformer not only provides an efficient solution for LDCT denoising but also shows broad potential in medical image analysis, especially for clinical applications dealing with complex noise patterns.
>
---
#### [new 213] Parameter Efficient Continual Learning with Dynamic Low-Rank Adaptation
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究持续学习中的灾难性遗忘问题，提出参数高效框架PEARL。针对低秩适配器（LoRA）对秩选择敏感导致性能受限的缺陷，PEARL通过动态分配任务特定LoRA的秩，基于参数空间内任务权重与参考任务的相似性自适应调整。实验覆盖多种视觉架构和场景，验证其显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11998v1](http://arxiv.org/pdf/2505.11998v1)**

> **作者:** Prashant Shivaram Bhat; Shakib Yazdani; Elahe Arani; Bahram Zonooz
>
> **备注:** 27 pages, 5 figures
>
> **摘要:** Catastrophic forgetting has remained a critical challenge for deep neural networks in Continual Learning (CL) as it undermines consolidated knowledge when learning new tasks. Parameter efficient fine tuning CL techniques are gaining traction for their effectiveness in addressing catastrophic forgetting with a lightweight training schedule while avoiding degradation of consolidated knowledge in pre-trained models. However, low rank adapters (LoRA) in these approaches are highly sensitive to rank selection which can lead to sub-optimal resource allocation and performance. To this end, we introduce PEARL, a rehearsal-free CL framework that entails dynamic rank allocation for LoRA components during CL training. Specifically, PEARL leverages reference task weights and adaptively determines the rank of task-specific LoRA components based on the current tasks' proximity to reference task weights in parameter space. To demonstrate the versatility of PEARL, we evaluate it across three vision architectures (ResNet, Separable Convolutional Network and Vision Transformer) and a multitude of CL scenarios, and show that PEARL outperforms all considered baselines by a large margin.
>
---
#### [new 214] MedVKAN: Efficient Feature Extraction with Mamba and KAN for Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对医学图像分割任务，解决CNN感受野受限和Transformer计算复杂度高的问题。提出MedVKAN模型，结合Mamba（线性复杂度长程建模）和KAN（可学习激活增强非线性），设计EFC-KAN模块强化局部交互，用VKAN替代传统Transformer模块。在五个数据集上实现四项最优，验证了高效特征提取能力。**

- **链接: [http://arxiv.org/pdf/2505.11797v1](http://arxiv.org/pdf/2505.11797v1)**

> **作者:** Hancan Zhu; Jinhao Chen; Guanghua He
>
> **摘要:** Medical image segmentation relies heavily on convolutional neural networks (CNNs) and Transformer-based models. However, CNNs are constrained by limited receptive fields, while Transformers suffer from scalability challenges due to their quadratic computational complexity. To address these limitations, recent advances have explored alternative architectures. The state-space model Mamba offers near-linear complexity while capturing long-range dependencies, and the Kolmogorov-Arnold Network (KAN) enhances nonlinear expressiveness by replacing fixed activation functions with learnable ones. Building on these strengths, we propose MedVKAN, an efficient feature extraction model integrating Mamba and KAN. Specifically, we introduce the EFC-KAN module, which enhances KAN with convolutional operations to improve local pixel interaction. We further design the VKAN module, integrating Mamba with EFC-KAN as a replacement for Transformer modules, significantly improving feature extraction. Extensive experiments on five public medical image segmentation datasets show that MedVKAN achieves state-of-the-art performance on four datasets and ranks second on the remaining one. These results validate the potential of Mamba and KAN for medical image segmentation while introducing an innovative and computationally efficient feature extraction framework. The code is available at: https://github.com/beginner-cjh/MedVKAN.
>
---
#### [new 215] Mutual Evidential Deep Learning for Medical Image Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于半监督医学图像分割任务，旨在解决现有方法因低质量伪标签导致的模型偏差问题。提出互证深度学习方法：使用不同架构网络生成互补证据并融合，设计渐进式证据学习策略，根据不确定性动态调整对伪标签的关注，提升模型鲁棒性。实验表明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.12418v1](http://arxiv.org/pdf/2505.12418v1)**

> **作者:** Yuanpeng He; Yali Bi; Lijian Li; Chi-Man Pun; Wenpin Jiao; Zhi Jin
>
> **摘要:** Existing semi-supervised medical segmentation co-learning frameworks have realized that model performance can be diminished by the biases in model recognition caused by low-quality pseudo-labels. Due to the averaging nature of their pseudo-label integration strategy, they fail to explore the reliability of pseudo-labels from different sources. In this paper, we propose a mutual evidential deep learning (MEDL) framework that offers a potentially viable solution for pseudo-label generation in semi-supervised learning from two perspectives. First, we introduce networks with different architectures to generate complementary evidence for unlabeled samples and adopt an improved class-aware evidential fusion to guide the confident synthesis of evidential predictions sourced from diverse architectural networks. Second, utilizing the uncertainty in the fused evidence, we design an asymptotic Fisher information-based evidential learning strategy. This strategy enables the model to initially focus on unlabeled samples with more reliable pseudo-labels, gradually shifting attention to samples with lower-quality pseudo-labels while avoiding over-penalization of mislabeled classes in high data uncertainty samples. Additionally, for labeled data, we continue to adopt an uncertainty-driven asymptotic learning strategy, gradually guiding the model to focus on challenging voxels. Extensive experiments on five mainstream datasets have demonstrated that MEDL achieves state-of-the-art performance.
>
---
#### [new 216] FlightGPT: Towards Generalizable and Interpretable UAV Vision-and-Language Navigation with Vision-Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究无人机视觉与语言导航任务，解决现有方法多模态融合弱、泛化差、可解释性低的问题。提出FlightGPT框架，基于视觉语言模型构建两阶段训练（监督微调+强化学习优化），结合思维链推理提升决策可解释性，在城市场景数据集实现9.22%成功率提升。**

- **链接: [http://arxiv.org/pdf/2505.12835v1](http://arxiv.org/pdf/2505.12835v1)**

> **作者:** Hengxing Cai; Jinhan Dong; Jingjun Tan; Jingcheng Deng; Sihang Li; Zhifeng Gao; Haidong Wang; Zicheng Su; Agachai Sumalee; Renxin Zhong
>
> **摘要:** Unmanned Aerial Vehicle (UAV) Vision-and-Language Navigation (VLN) is vital for applications such as disaster response, logistics delivery, and urban inspection. However, existing methods often struggle with insufficient multimodal fusion, weak generalization, and poor interpretability. To address these challenges, we propose FlightGPT, a novel UAV VLN framework built upon Vision-Language Models (VLMs) with powerful multimodal perception capabilities. We design a two-stage training pipeline: first, Supervised Fine-Tuning (SFT) using high-quality demonstrations to improve initialization and structured reasoning; then, Group Relative Policy Optimization (GRPO) algorithm, guided by a composite reward that considers goal accuracy, reasoning quality, and format compliance, to enhance generalization and adaptability. Furthermore, FlightGPT introduces a Chain-of-Thought (CoT)-based reasoning mechanism to improve decision interpretability. Extensive experiments on the city-scale dataset CityNav demonstrate that FlightGPT achieves state-of-the-art performance across all scenarios, with a 9.22\% higher success rate than the strongest baseline in unseen environments. Our implementation is publicly available.
>
---
#### [new 217] Bridging the Inter-Domain Gap through Low-Level Features for Cross-Modal Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文研究跨模态医学图像分割，解决不同模态间域差异问题。提出无监督域适应框架LowBridge，利用共享低层特征（如边缘）生成源模态风格的目标图像，再分割。实验表明其性能优于现有方法，且模型无关，可适配不同生成和分割模型。**

- **链接: [http://arxiv.org/pdf/2505.11909v1](http://arxiv.org/pdf/2505.11909v1)**

> **作者:** Pengfei Lyu; Pak-Hei Yeung; Xiaosheng Yu; Jing Xia; Jianning Chi; Chengdong Wu; Jagath C. Rajapakse
>
> **备注:** 11 pages, 2 figures
>
> **摘要:** This paper addresses the task of cross-modal medical image segmentation by exploring unsupervised domain adaptation (UDA) approaches. We propose a model-agnostic UDA framework, LowBridge, which builds on a simple observation that cross-modal images share some similar low-level features (e.g., edges) as they are depicting the same structures. Specifically, we first train a generative model to recover the source images from their edge features, followed by training a segmentation model on the generated source images, separately. At test time, edge features from the target images are input to the pretrained generative model to generate source-style target domain images, which are then segmented using the pretrained segmentation network. Despite its simplicity, extensive experiments on various publicly available datasets demonstrate that \proposed achieves state-of-the-art performance, outperforming eleven existing UDA approaches under different settings. Notably, further ablation studies show that \proposed is agnostic to different types of generative and segmentation models, suggesting its potential to be seamlessly plugged with the most advanced models to achieve even more outstanding results in the future. The code is available at https://github.com/JoshuaLPF/LowBridge.
>
---
#### [new 218] VoiceCloak: A Multi-Dimensional Defense Framework against Unauthorized Diffusion-based Voice Cloning
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于语音安全领域，旨在防御基于扩散模型的未经授权语音克隆。针对现有防御方法不兼容扩散机制的问题，提出VoiceCloak框架：通过对抗扰动混淆说话人身份（扭曲表征、干扰注意力），并降低语音质量（评分放大、噪声破坏语义），阻断克隆过程。实验验证其防御效果显著。**

- **链接: [http://arxiv.org/pdf/2505.12332v1](http://arxiv.org/pdf/2505.12332v1)**

> **作者:** Qianyue Hu; Junyan Wu; Wei Lu; Xiangyang Luo
>
> **摘要:** Diffusion Models (DMs) have achieved remarkable success in realistic voice cloning (VC), while they also increase the risk of malicious misuse. Existing proactive defenses designed for traditional VC models aim to disrupt the forgery process, but they have been proven incompatible with DMs due to the intricate generative mechanisms of diffusion. To bridge this gap, we introduce VoiceCloak, a multi-dimensional proactive defense framework with the goal of obfuscating speaker identity and degrading perceptual quality in potential unauthorized VC. To achieve these goals, we conduct a focused analysis to identify specific vulnerabilities within DMs, allowing VoiceCloak to disrupt the cloning process by introducing adversarial perturbations into the reference audio. Specifically, to obfuscate speaker identity, VoiceCloak first targets speaker identity by distorting representation learning embeddings to maximize identity variation, which is guided by auditory perception principles. Additionally, VoiceCloak disrupts crucial conditional guidance processes, particularly attention context, thereby preventing the alignment of vocal characteristics that are essential for achieving convincing cloning. Then, to address the second objective, VoiceCloak introduces score magnitude amplification to actively steer the reverse trajectory away from the generation of high-quality speech. Noise-guided semantic corruption is further employed to disrupt structural speech semantics captured by DMs, degrading output quality. Extensive experiments highlight VoiceCloak's outstanding defense success rate against unauthorized diffusion-based voice cloning. Audio samples of VoiceCloak are available at https://voice-cloak.github.io/VoiceCloak/.
>
---
#### [new 219] Enhancing Diffusion-Weighted Images (DWI) for Diffusion MRI: Is it Enough without Non-Diffusion-Weighted B=0 Reference?
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对扩散MRI图像增强任务，指出传统方法仅优化扩散加权图像（DWI）而忽略其与非扩散加权b=0图像的比率关系，导致扩散指标计算误差。作者提出基于DWI/b=0对数比值的比率损失函数，有效降低比率误差，改善超分辨率效果并保留扩散特征。**

- **链接: [http://arxiv.org/pdf/2505.12978v1](http://arxiv.org/pdf/2505.12978v1)**

> **作者:** Yinzhe Wu; Jiahao Huang; Fanwen Wang; Mengze Gao; Congyu Liao; Guang Yang; Kawin Setsompop
>
> **备注:** IEEE ISBI 2025
>
> **摘要:** Diffusion MRI (dMRI) is essential for studying brain microstructure, but high-resolution imaging remains challenging due to the inherent trade-offs between acquisition time and signal-to-noise ratio (SNR). Conventional methods often optimize only the diffusion-weighted images (DWIs) without considering their relationship with the non-diffusion-weighted (b=0) reference images. However, calculating diffusion metrics, such as the apparent diffusion coefficient (ADC) and diffusion tensor with its derived metrics like fractional anisotropy (FA) and mean diffusivity (MD), relies on the ratio between each DWI and the b=0 image, which is crucial for clinical observation and diagnostics. In this study, we demonstrate that solely enhancing DWIs using a conventional pixel-wise mean squared error (MSE) loss is insufficient, as the error in ratio between generated DWIs and b=0 diverges. We propose a novel ratio loss, defined as the MSE loss between the predicted and ground-truth log of DWI/b=0 ratios. Our results show that incorporating the ratio loss significantly improves the convergence of this ratio error, achieving lower ratio MSE and slightly enhancing the peak signal-to-noise ratio (PSNR) of generated DWIs. This leads to improved dMRI super-resolution and better preservation of b=0 ratio-based features for the derivation of diffusion metrics.
>
---
#### [new 220] ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于图表问答任务，旨在解决大型视觉-语言模型（LVLM）视觉推理能力不足的问题。通过构建ChartMuseum基准（含1,162真实图表问题），揭示模型在视觉复杂场景下性能显著低于人类（最佳模型63% vs 人类93%），并验证视觉推理是当前LVLM的主要瓶颈。**

- **链接: [http://arxiv.org/pdf/2505.13444v1](http://arxiv.org/pdf/2505.13444v1)**

> **作者:** Liyan Tang; Grace Kim; Xinyu Zhao; Thom Lake; Wenxuan Ding; Fangcong Yin; Prasann Singhal; Manya Wadhwa; Zeyu Leo Liu; Zayne Sprague; Ramya Namuduri; Bodun Hu; Juan Diego Rodriguez; Puyuan Peng; Greg Durrett
>
> **摘要:** Chart understanding presents a unique challenge for large vision-language models (LVLMs), as it requires the integration of sophisticated textual and visual reasoning capabilities. However, current LVLMs exhibit a notable imbalance between these skills, falling short on visual reasoning that is difficult to perform in text. We conduct a case study using a synthetic dataset solvable only through visual reasoning and show that model performance degrades significantly with increasing visual complexity, while human performance remains robust. We then introduce ChartMuseum, a new Chart Question Answering (QA) benchmark containing 1,162 expert-annotated questions spanning multiple reasoning types, curated from real-world charts across 184 sources, specifically built to evaluate complex visual and textual reasoning. Unlike prior chart understanding benchmarks -- where frontier models perform similarly and near saturation -- our benchmark exposes a substantial gap between model and human performance, while effectively differentiating model capabilities: although humans achieve 93% accuracy, the best-performing model Gemini-2.5-Pro attains only 63.0%, and the leading open-source LVLM Qwen2.5-VL-72B-Instruct achieves only 38.5%. Moreover, on questions requiring primarily visual reasoning, all models experience a 35%-55% performance drop from text-reasoning-heavy question performance. Lastly, our qualitative error analysis reveals specific categories of visual reasoning that are challenging for current LVLMs.
>
---
#### [new 221] HISTAI: An Open-Source, Large-Scale Whole Slide Image Dataset for Computational Pathology
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出开源大规模全切片图像数据集HISTAI，属于计算病理学数据资源建设任务。针对现有公开数据集规模小、组织单一、标注不足的问题，构建了包含6万+多组织切片、附带临床元数据和标准化诊断编码的开放数据集，旨在提升AI模型在病理分析中的鲁棒性和临床适用性。**

- **链接: [http://arxiv.org/pdf/2505.12120v1](http://arxiv.org/pdf/2505.12120v1)**

> **作者:** Dmitry Nechaev; Alexey Pchelnikov; Ekaterina Ivanova
>
> **摘要:** Recent advancements in Digital Pathology (DP), particularly through artificial intelligence and Foundation Models, have underscored the importance of large-scale, diverse, and richly annotated datasets. Despite their critical role, publicly available Whole Slide Image (WSI) datasets often lack sufficient scale, tissue diversity, and comprehensive clinical metadata, limiting the robustness and generalizability of AI models. In response, we introduce the HISTAI dataset, a large, multimodal, open-access WSI collection comprising over 60,000 slides from various tissue types. Each case in the HISTAI dataset is accompanied by extensive clinical metadata, including diagnosis, demographic information, detailed pathological annotations, and standardized diagnostic coding. The dataset aims to fill gaps identified in existing resources, promoting innovation, reproducibility, and the development of clinically relevant computational pathology solutions. The dataset can be accessed at https://github.com/HistAI/HISTAI.
>
---
#### [new 222] Walking the Tightrope: Disentangling Beneficial and Detrimental Drifts in Non-Stationary Custom-Tuning
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对多模态大语言模型在非平稳强化微调中思维链推理的"有害概念漂移"问题，提出反事实感知偏好优化方法（CPO），通过解耦有益/有害分布漂移，增强模型鲁棒性。属于模型动态适应任务，贡献包括理论框架、医疗领域数据集CXR-CounterFact及稳定微调方案。**

- **链接: [http://arxiv.org/pdf/2505.13081v1](http://arxiv.org/pdf/2505.13081v1)**

> **作者:** Xiaoyu Yang; Jie Lu; En Yu
>
> **备注:** 17 pages, 5figures
>
> **摘要:** This paper uncovers a critical yet overlooked phenomenon in multi-modal large language models (MLLMs): detrimental concept drift within chain-of-thought (CoT) reasoning during non-stationary reinforcement fine-tuning (RFT), where reasoning token distributions evolve unpredictably, thereby introducing significant biases in final predictions. To address this, we are pioneers in establishing the theoretical bridge between concept drift theory and RFT processes by formalizing CoT's autoregressive token streams as non-stationary distributions undergoing arbitrary temporal shifts. Leveraging this framework, we propose a novel counterfact-aware RFT that systematically decouples beneficial distribution adaptation from harmful concept drift through concept graph-empowered LLM experts generating counterfactual reasoning trajectories. Our solution, Counterfactual Preference Optimization (CPO), enables stable RFT in non-stationary environments, particularly within the medical domain, through custom-tuning of counterfactual-aware preference alignment. Extensive experiments demonstrate our superior performance of robustness, generalization and coordination within RFT. Besides, we also contributed a large-scale dataset CXR-CounterFact (CCF), comprising 320,416 meticulously curated counterfactual reasoning trajectories derived from MIMIC-CXR. Our code and data are public.
>
---
#### [new 223] TinyAlign: Boosting Lightweight Vision-Language Models by Mitigating Modal Alignment Bottlenecks
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对轻量级视觉-语言模型（VLMs）模态对齐受限问题，提出TinyAlign框架。通过互信息分析揭示语言模型容量限制导致对齐瓶颈，引入检索增强机制从记忆库提取上下文优化输入，显著提升数据效率和任务性能，属于多模态对齐优化任务。**

- **链接: [http://arxiv.org/pdf/2505.12884v1](http://arxiv.org/pdf/2505.12884v1)**

> **作者:** Yuanze Hu; Zhaoxin Fan; Xinyu Wang; Gen Li; Ye Qiu; Zhichao Yang; Wenjun Wu; Kejian Wu; Yifan Sun; Xiaotie Deng; Jin Dong
>
> **摘要:** Lightweight Vision-Language Models (VLMs) are indispensable for resource-constrained applications. The prevailing approach to aligning vision and language models involves freezing both the vision encoder and the language model while training small connector modules. However, this strategy heavily depends on the intrinsic capabilities of the language model, which can be suboptimal for lightweight models with limited representational capacity. In this work, we investigate this alignment bottleneck through the lens of mutual information, demonstrating that the constrained capacity of the language model inherently limits the Effective Mutual Information (EMI) between multimodal inputs and outputs, thereby compromising alignment quality. To address this challenge, we propose TinyAlign, a novel framework inspired by Retrieval-Augmented Generation, which strategically retrieves relevant context from a memory bank to enrich multimodal inputs and enhance their alignment. Extensive empirical evaluations reveal that TinyAlign significantly reduces training loss, accelerates convergence, and enhances task performance. Remarkably, it allows models to achieve baseline-level performance with only 40\% of the fine-tuning data, highlighting exceptional data efficiency. Our work thus offers a practical pathway for developing more capable lightweight VLMs while introducing a fresh theoretical lens to better understand and address alignment bottlenecks in constrained multimodal systems.
>
---
#### [new 224] Bridging Human Oversight and Black-box Driver Assistance: Vision-Language Models for Predictive Alerting in Lane Keeping Assist Systems
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于自动驾驶辅助系统的安全预警任务，旨在解决车道保持辅助系统（LKA）因黑盒特性导致失效预测困难及用户信任不足的问题。提出LKAlert系统，通过视觉语言模型融合多模态数据预测风险并生成解释，构建首个基准数据集和方法框架，验证了实时预警能力（69.8%准确率）与解释效果（71.7 ROUGE-L）。**

- **链接: [http://arxiv.org/pdf/2505.11535v1](http://arxiv.org/pdf/2505.11535v1)**

> **作者:** Yuhang Wang; Hao Zhou
>
> **摘要:** Lane Keeping Assist systems, while increasingly prevalent, often suffer from unpredictable real-world failures, largely due to their opaque, black-box nature, which limits driver anticipation and trust. To bridge the gap between automated assistance and effective human oversight, we present LKAlert, a novel supervisory alert system that leverages VLM to forecast potential LKA risk 1-3 seconds in advance. LKAlert processes dash-cam video and CAN data, integrating surrogate lane segmentation features from a parallel interpretable model as automated guiding attention. Unlike traditional binary classifiers, LKAlert issues both predictive alert and concise natural language explanation, enhancing driver situational awareness and trust. To support the development and evaluation of such systems, we introduce OpenLKA-Alert, the first benchmark dataset designed for predictive and explainable LKA failure warnings. It contains synchronized multimodal inputs and human-authored justifications across annotated temporal windows. We further contribute a generalizable methodological framework for VLM-based black-box behavior prediction, combining surrogate feature guidance with LoRA. This framework enables VLM to reason over structured visual context without altering its vision backbone, making it broadly applicable to other complex, opaque systems requiring interpretable oversight. Empirical results correctly predicts upcoming LKA failures with 69.8% accuracy and a 58.6\% F1-score. The system also generates high-quality textual explanations for drivers (71.7 ROUGE-L) and operates efficiently at approximately 2 Hz, confirming its suitability for real-time, in-vehicle use. Our findings establish LKAlert as a practical solution for enhancing the safety and usability of current ADAS and offer a scalable paradigm for applying VLMs to human-centered supervision of black-box automation.
>
---
#### [new 225] RetinaLogos: Fine-Grained Synthesis of High-Resolution Retinal Images Through Captions
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于视网膜图像生成任务，旨在解决眼科数据稀缺及现有方法无法生成细粒度解剖结构的问题。提出RetinaLogos框架：先利用大语言模型构建140万条文本-图像数据集描述病变细节，再通过三步训练实现高分辨率图像合成，支持疾病阶段、解剖变异等精准控制。实验证明合成图像逼真度达62.07%，并提升糖尿病视网膜病变和青光眼检测准确率10%-25%。**

- **链接: [http://arxiv.org/pdf/2505.12887v1](http://arxiv.org/pdf/2505.12887v1)**

> **作者:** Junzhi Ning; Cheng Tang; Kaijin Zhou; Diping Song; Lihao Liu; Ming Hu; Wei Li; Yanzhou Su; Tianbing Li; Jiyao Liu; Yejin; Sheng Zhang; Yuanfeng Ji; Junjun He
>
> **摘要:** The scarcity of high-quality, labelled retinal imaging data, which presents a significant challenge in the development of machine learning models for ophthalmology, hinders progress in the field. To synthesise Colour Fundus Photographs (CFPs), existing methods primarily relying on predefined disease labels face significant limitations. However, current methods remain limited, thus failing to generate images for broader categories with diverse and fine-grained anatomical structures. To overcome these challenges, we first introduce an innovative pipeline that creates a large-scale, synthetic Caption-CFP dataset comprising 1.4 million entries, called RetinaLogos-1400k. Specifically, RetinaLogos-1400k uses large language models (LLMs) to describe retinal conditions and key structures, such as optic disc configuration, vascular distribution, nerve fibre layers, and pathological features. Furthermore, based on this dataset, we employ a novel three-step training framework, called RetinaLogos, which enables fine-grained semantic control over retinal images and accurately captures different stages of disease progression, subtle anatomical variations, and specific lesion types. Extensive experiments demonstrate state-of-the-art performance across multiple datasets, with 62.07% of text-driven synthetic images indistinguishable from real ones by ophthalmologists. Moreover, the synthetic data improves accuracy by 10%-25% in diabetic retinopathy grading and glaucoma detection, thereby providing a scalable solution to augment ophthalmic datasets.
>
---
#### [new 226] Structureless VIO
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究视觉惯性里程计（VIO），针对传统方法中定位与建图强耦合导致的效率瓶颈，提出无结构VIO框架。通过移除视觉地图模块，解耦定位与建图依赖关系，在提升计算效率的同时实现更高精度，解决了传统VIO因结构维护产生的性能限制问题。**

- **链接: [http://arxiv.org/pdf/2505.12337v1](http://arxiv.org/pdf/2505.12337v1)**

> **作者:** Junlin Song; Miguel Olivares-Mendez
>
> **摘要:** Visual odometry (VO) is typically considered as a chicken-and-egg problem, as the localization and mapping modules are tightly-coupled. The estimation of visual map relies on accurate localization information. Meanwhile, localization requires precise map points to provide motion constraints. This classical design principle is naturally inherited by visual-inertial odometry (VIO). Efficient localization solution that does not require a map has not been fully investigated. To this end, we propose a novel structureless VIO, where the visual map is removed from the odometry framework. Experimental results demonstrated that, compared to the structure-based VIO baseline, our structureless VIO not only substantially improves computational efficiency but also has advantages in accuracy.
>
---
#### [new 227] TeleOpBench: A Simulator-Centric Benchmark for Dual-Arm Dexterous Teleoperation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人遥操作评估领域，旨在解决双臂灵巧遥操作硬件系统缺乏统一测评标准的问题。研究者开发了TeleOpBench模拟基准，包含30种任务和四种操作方式，通过仿真与实物实验验证其有效性，为系统比较和创新提供平台。**

- **链接: [http://arxiv.org/pdf/2505.12748v1](http://arxiv.org/pdf/2505.12748v1)**

> **作者:** Hangyu Li; Qin Zhao; Haoran Xu; Xinyu Jiang; Qingwei Ben; Feiyu Jia; Haoyu Zhao; Liang Xu; Jia Zeng; Hanqing Wang; Bo Dai; Junting Dong; Jiangmiao Pang
>
> **备注:** 13 pages
>
> **摘要:** Teleoperation is a cornerstone of embodied-robot learning, and bimanual dexterous teleoperation in particular provides rich demonstrations that are difficult to obtain with fully autonomous systems. While recent studies have proposed diverse hardware pipelines-ranging from inertial motion-capture gloves to exoskeletons and vision-based interfaces-there is still no unified benchmark that enables fair, reproducible comparison of these systems. In this paper, we introduce TeleOpBench, a simulator-centric benchmark tailored to bimanual dexterous teleoperation. TeleOpBench contains 30 high-fidelity task environments that span pick-and-place, tool use, and collaborative manipulation, covering a broad spectrum of kinematic and force-interaction difficulty. Within this benchmark we implement four representative teleoperation modalities-(i) MoCap, (ii) VR device, (iii) arm-hand exoskeletons, and (iv) monocular vision tracking-and evaluate them with a common protocol and metric suite. To validate that performance in simulation is predictive of real-world behavior, we conduct mirrored experiments on a physical dual-arm platform equipped with two 6-DoF dexterous hands. Across 10 held-out tasks we observe a strong correlation between simulator and hardware performance, confirming the external validity of TeleOpBench. TeleOpBench establishes a common yardstick for teleoperation research and provides an extensible platform for future algorithmic and hardware innovation.
>
---
#### [new 228] An approach based on class activation maps for investigating the effects of data augmentation on neural networks for image classification
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于图像分类任务，研究数据增强对卷积网络学习模式的影响。针对现有研究缺乏定量分析数据增强效果的问题，提出基于类激活图的方法，通过提取像素重要性指标，量化比较不同增强策略对模型关注区域的影响，揭示其差异化的作用模式。**

- **链接: [http://arxiv.org/pdf/2505.12581v1](http://arxiv.org/pdf/2505.12581v1)**

> **作者:** Lucas M. Dorneles; Luan Fonseca Garcia; Joel Luís Carbonera
>
> **摘要:** Neural networks have become increasingly popular in the last few years as an effective tool for the task of image classification due to the impressive performance they have achieved on this task. In image classification tasks, it is common to use data augmentation strategies to increase the robustness of trained networks to changes in the input images and to avoid overfitting. Although data augmentation is a widely adopted technique, the literature lacks a body of research analyzing the effects data augmentation methods have on the patterns learned by neural network models working on complex datasets. The primary objective of this work is to propose a methodology and set of metrics that may allow a quantitative approach to analyzing the effects of data augmentation in convolutional networks applied to image classification. An important tool used in the proposed approach lies in the concept of class activation maps for said models, which allow us to identify and measure the importance these models assign to each individual pixel in an image when executing the classification task. From these maps, we may then extract metrics over the similarities and differences between maps generated by these models trained on a given dataset with different data augmentation strategies. Experiments made using this methodology suggest that the effects of these data augmentation techniques not only can be analyzed in this way but also allow us to identify different impact profiles over the trained models.
>
---
#### [new 229] CALM-PDE: Continuous and Adaptive Convolutions for Latent Space Modeling of Time-dependent PDEs
- **分类: cs.LG; cs.AI; cs.CV; cs.NE; physics.comp-ph**

- **简介: 该论文提出CALM-PDE模型，针对时间依赖偏微分方程(PDE)求解任务，解决传统方法计算成本高及现有神经代理模型内存效率低的问题。通过连续卷积编码器-解码器架构，自适应处理任意离散化空间，在保证精度的同时显著提升内存和推理效率，优于Transformer等方法。**

- **链接: [http://arxiv.org/pdf/2505.12944v1](http://arxiv.org/pdf/2505.12944v1)**

> **作者:** Jan Hagnberger; Daniel Musekamp; Mathias Niepert
>
> **摘要:** Solving time-dependent Partial Differential Equations (PDEs) using a densely discretized spatial domain is a fundamental problem in various scientific and engineering disciplines, including modeling climate phenomena and fluid dynamics. However, performing these computations directly in the physical space often incurs significant computational costs. To address this issue, several neural surrogate models have been developed that operate in a compressed latent space to solve the PDE. While these approaches reduce computational complexity, they often use Transformer-based attention mechanisms to handle irregularly sampled domains, resulting in increased memory consumption. In contrast, convolutional neural networks allow memory-efficient encoding and decoding but are limited to regular discretizations. Motivated by these considerations, we propose CALM-PDE, a model class that efficiently solves arbitrarily discretized PDEs in a compressed latent space. We introduce a novel continuous convolution-based encoder-decoder architecture that uses an epsilon-neighborhood-constrained kernel and learns to apply the convolution operator to adaptive and optimized query points. We demonstrate the effectiveness of CALM-PDE on a diverse set of PDEs with both regularly and irregularly sampled spatial domains. CALM-PDE is competitive with or outperforms existing baseline methods while offering significant improvements in memory and inference time efficiency compared to Transformer-based methods.
>
---
#### [new 230] Bayesian Deep Learning Approaches for Uncertainty-Aware Retinal OCT Image Segmentation for Multiple Sclerosis
- **分类: eess.IV; cs.CV; 68U10, 92C55; I.2.10; I.4.6; J.3**

- **简介: 该论文针对视网膜OCT图像分割任务，解决传统深度学习方法缺乏不确定性估计导致误诊的问题。通过贝叶斯卷积神经网络生成分割结果及不确定性图谱，识别噪声样本并量化层厚测量误差，在公开数据集上实现95.65% Dice分数，提升临床可信度和统计稳健性。**

- **链接: [http://arxiv.org/pdf/2505.12061v1](http://arxiv.org/pdf/2505.12061v1)**

> **作者:** Samuel T. M. Ball
>
> **摘要:** Optical Coherence Tomography (OCT) provides valuable insights in ophthalmology, cardiology, and neurology due to high-resolution, cross-sectional images of the retina. One critical task for ophthalmologists using OCT is delineation of retinal layers within scans. This process is time-consuming and prone to human bias, affecting the accuracy and reliability of diagnoses. Previous efforts to automate delineation using deep learning face challenges in uptake from clinicians and statisticians due to the absence of uncertainty estimation, leading to "confidently wrong" models via hallucinations. In this study, we address these challenges by applying Bayesian convolutional neural networks (BCNNs) to segment an openly available OCT imaging dataset containing 35 human retina OCTs split between healthy controls and patients with multiple sclerosis. Our findings demonstrate that Bayesian models can be used to provide uncertainty maps of the segmentation, which can further be used to identify highly uncertain samples that exhibit recording artefacts such as noise or miscalibration at inference time. Our method also allows for uncertainty-estimation for important secondary measurements such as layer thicknesses, that are medically relevant for patients. We show that these features come in addition to greater performance compared to similar work over all delineations; with an overall Dice score of 95.65%. Our work brings greater clinical applicability, statistical robustness, and performance to retinal OCT segmentation.
>
---
#### [new 231] OpenPros: A Large-Scale Dataset for Limited View Prostate Ultrasound Computed Tomography
- **分类: physics.med-ph; cs.CV**

- **简介: 该论文针对有限视角前列腺超声CT成像难题，构建首个大规模数据集OpenPros，包含28万对模拟声速模型与超声波形数据，基于真实医学影像生成并开源仿真工具。研究验证深度学习在重建效率与精度上优于传统方法，但现有模型仍无法满足临床高分辨率需求，旨在推动算法开发以提升前列腺癌检测效果。**

- **链接: [http://arxiv.org/pdf/2505.12261v1](http://arxiv.org/pdf/2505.12261v1)**

> **作者:** Hanchen Wang; Yixuan Wu; Yinan Feng; Peng Jin; Shihang Feng; Yiming Mao; James Wiskin; Baris Turkbey; Peter A. Pinto; Bradford J. Wood; Songting Luo; Yinpeng Chen; Emad Boctor; Youzuo Lin
>
> **摘要:** Prostate cancer is one of the most common and lethal cancers among men, making its early detection critically important. Although ultrasound imaging offers greater accessibility and cost-effectiveness compared to MRI, traditional transrectal ultrasound methods suffer from low sensitivity, especially in detecting anteriorly located tumors. Ultrasound computed tomography provides quantitative tissue characterization, but its clinical implementation faces significant challenges, particularly under anatomically constrained limited-angle acquisition conditions specific to prostate imaging. To address these unmet needs, we introduce OpenPros, the first large-scale benchmark dataset explicitly developed for limited-view prostate USCT. Our dataset includes over 280,000 paired samples of realistic 2D speed-of-sound (SOS) phantoms and corresponding ultrasound full-waveform data, generated from anatomically accurate 3D digital prostate models derived from real clinical MRI/CT scans and ex vivo ultrasound measurements, annotated by medical experts. Simulations are conducted under clinically realistic configurations using advanced finite-difference time-domain and Runge-Kutta acoustic wave solvers, both provided as open-source components. Through comprehensive baseline experiments, we demonstrate that state-of-the-art deep learning methods surpass traditional physics-based approaches in both inference efficiency and reconstruction accuracy. Nevertheless, current deep learning models still fall short of delivering clinically acceptable high-resolution images with sufficient accuracy. By publicly releasing OpenPros, we aim to encourage the development of advanced machine learning algorithms capable of bridging this performance gap and producing clinically usable, high-resolution, and highly accurate prostate ultrasound images. The dataset is publicly accessible at https://open-pros.github.io/.
>
---
#### [new 232] Enhanced Multimodal Hate Video Detection via Channel-wise and Modality-wise Fusion
- **分类: cs.MM; cs.AI; cs.CV**

- **简介: 该论文研究多模态仇恨视频检测任务，解决现有单模态方法特征利用不足及多模态融合低效问题。提出CMFusion模型，通过时序交叉注意力捕捉音视频动态关联，结合通道/模态融合模块整合文本、音频、视频特征，实验验证其检测性能优于主流基线。**

- **链接: [http://arxiv.org/pdf/2505.12051v1](http://arxiv.org/pdf/2505.12051v1)**

> **作者:** Yinghui Zhang; Tailin Chen; Yuchen Zhang; Zeyu Fu
>
> **备注:** ICDMW 2024, Github: https://github.com/EvelynZ10/cmfusion
>
> **摘要:** The rapid rise of video content on platforms such as TikTok and YouTube has transformed information dissemination, but it has also facilitated the spread of harmful content, particularly hate videos. Despite significant efforts to combat hate speech, detecting these videos remains challenging due to their often implicit nature. Current detection methods primarily rely on unimodal approaches, which inadequately capture the complementary features across different modalities. While multimodal techniques offer a broader perspective, many fail to effectively integrate temporal dynamics and modality-wise interactions essential for identifying nuanced hate content. In this paper, we present CMFusion, an enhanced multimodal hate video detection model utilizing a novel Channel-wise and Modality-wise Fusion Mechanism. CMFusion first extracts features from text, audio, and video modalities using pre-trained models and then incorporates a temporal cross-attention mechanism to capture dependencies between video and audio streams. The learned features are then processed by channel-wise and modality-wise fusion modules to obtain informative representations of videos. Our extensive experiments on a real-world dataset demonstrate that CMFusion significantly outperforms five widely used baselines in terms of accuracy, precision, recall, and F1 score. Comprehensive ablation studies and parameter analyses further validate our design choices, highlighting the model's effectiveness in detecting hate videos. The source codes will be made publicly available at https://github.com/EvelynZ10/cmfusion.
>
---
#### [new 233] GLOVER++: Unleashing the Potential of Affordance Learning from Human Behaviors for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究机器人操控中的affordance学习任务，解决人类示范知识迁移中数据集稀缺和情境多样性不足问题。提出了包含50万图像标注的HOVA-500K数据集和全局到局部训练框架GLOVER++，通过多模态推理实现跨场景的鲁棒知识迁移，在基准测试和下游任务中展现优越性能。**

- **链接: [http://arxiv.org/pdf/2505.11865v1](http://arxiv.org/pdf/2505.11865v1)**

> **作者:** Teli Ma; Jia Zheng; Zifan Wang; Ziyao Gao; Jiaming Zhou; Junwei Liang
>
> **摘要:** Learning manipulation skills from human demonstration videos offers a promising path toward generalizable and interpretable robotic intelligence-particularly through the lens of actionable affordances. However, transferring such knowledge remains challenging due to: 1) a lack of large-scale datasets with precise affordance annotations, and 2) insufficient exploration of affordances in diverse manipulation contexts. To address these gaps, we introduce HOVA-500K, a large-scale, affordance-annotated dataset comprising 500,000 images across 1,726 object categories and 675 actions. We also release a standardized benchmarking suite for multi-modal affordance reasoning. Built upon HOVA-500K, we present GLOVER++, a global-to-local affordance training framework that effectively transfers actionable affordance knowledge from human demonstrations to downstream open-vocabulary reasoning tasks. GLOVER++ achieves state-of-the-art results on the HOVA-500K benchmark and demonstrates strong generalization across diverse downstream robotic manipulation tasks. By explicitly modeling actionable affordances, GLOVER++ facilitates robust transfer across scenes, modalities, and tasks. We hope that HOVA-500K and the GLOVER++ framework will serve as valuable resources for bridging the gap between human demonstrations and robotic manipulation capabilities.
>
---
#### [new 234] MIRACL-VISION: A Large, multilingual, visual document retrieval benchmark
- **分类: cs.IR; cs.CV**

- **简介: 该论文属于视觉文档检索任务，旨在解决现有基准局限于英语、依赖合成问题和小规模语料的问题。作者构建了多语言评测基准MIRACL-VISION，覆盖18种语言，优化语料以剔除简单负样本，实验表明视觉模型检索准确率显著低于文本模型，凸显多语言能力不足。**

- **链接: [http://arxiv.org/pdf/2505.11651v1](http://arxiv.org/pdf/2505.11651v1)**

> **作者:** Radek Osmulsk; Gabriel de Souza P. Moreira; Ronay Ak; Mengyao Xu; Benedikt Schifferer; Even Oldridge
>
> **摘要:** Document retrieval is an important task for search and Retrieval-Augmented Generation (RAG) applications. Large Language Models (LLMs) have contributed to improving the accuracy of text-based document retrieval. However, documents with complex layout and visual elements like tables, charts and infographics are not perfectly represented in textual format. Recently, image-based document retrieval pipelines have become popular, which use visual large language models (VLMs) to retrieve relevant page images given a query. Current evaluation benchmarks on visual document retrieval are limited, as they primarily focus only English language, rely on synthetically generated questions and offer a small corpus size. Therefore, we introduce MIRACL-VISION, a multilingual visual document retrieval evaluation benchmark. MIRACL-VISION covers 18 languages, and is an extension of the MIRACL dataset, a popular benchmark to evaluate text-based multilingual retrieval pipelines. MIRACL was built using a human-intensive annotation process to generate high-quality questions. In order to reduce MIRACL-VISION corpus size to make evaluation more compute friendly while keeping the datasets challenging, we have designed a method for eliminating the "easy" negatives from the corpus. We conducted extensive experiments comparing MIRACL-VISION with other benchmarks, using popular public text and image models. We observe a gap in state-of-the-art VLM-based embedding models on multilingual capabilities, with up to 59.7% lower retrieval accuracy than a text-based retrieval models. Even for the English language, the visual models retrieval accuracy is 12.1% lower compared to text-based models. MIRACL-VISION is a challenging, representative, multilingual evaluation benchmark for visual retrieval pipelines and will help the community build robust models for document retrieval.
>
---
#### [new 235] RECON: Robust symmetry discovery via Explicit Canonical Orientation Normalization
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于对称性发现任务，解决现有等变网络需预设固定变换群组导致性能下降的问题。提出RECON框架，通过类姿态分解和数据驱动归一化，将任意参考系对齐至统一自然姿态，自动发现输入数据的对称分布，首次实现3D变换群组的对称建模，提升等变网络的灵活性。**

- **链接: [http://arxiv.org/pdf/2505.13289v1](http://arxiv.org/pdf/2505.13289v1)**

> **作者:** Alonso Urbano; David W. Romero; Max Zimmer; Sebastian Pokutta
>
> **摘要:** Real-world data often exhibits unknown or approximate symmetries, yet existing equivariant networks must commit to a fixed transformation group prior to training, e.g., continuous $SO(2)$ rotations. This mismatch degrades performance when the actual data symmetries differ from those in the transformation group. We introduce RECON, a framework to discover each input's intrinsic symmetry distribution from unlabeled data. RECON leverages class-pose decompositions and applies a data-driven normalization to align arbitrary reference frames into a common natural pose, yielding directly comparable and interpretable symmetry descriptors. We demonstrate effective symmetry discovery on 2D image benchmarks and -- for the first time -- extend it to 3D transformation groups, paving the way towards more flexible equivariant modeling.
>
---
#### [new 236] A generalisable head MRI defacing pipeline: Evaluation on 2,566 meningioma scans
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像隐私保护任务，旨在解决MRI去面部特征时隐私清除不彻底或损伤脑组织的问题。研究者提出融合图谱配准与脑部掩模的鲁棒去身份化方法，在2566例脑膜瘤临床扫描中实现99.92%成功率，并通过Dice系数0.9975验证解剖结构完整性，代码已开源。**

- **链接: [http://arxiv.org/pdf/2505.12999v1](http://arxiv.org/pdf/2505.12999v1)**

> **作者:** Lorena Garcia-Foncillas Macias; Aaron Kujawa; Aya Elshalakany; Jonathan Shapey; Tom Vercauteren
>
> **摘要:** Reliable MRI defacing techniques to safeguard patient privacy while preserving brain anatomy are critical for research collaboration. Existing methods often struggle with incomplete defacing or degradation of brain tissue regions. We present a robust, generalisable defacing pipeline for high-resolution MRI that integrates atlas-based registration with brain masking. Our method was evaluated on 2,566 heterogeneous clinical scans for meningioma and achieved a 99.92 per cent success rate (2,564/2,566) upon visual inspection. Excellent anatomical preservation is demonstrated with a Dice similarity coefficient of 0.9975 plus or minus 0.0023 between brain masks automatically extracted from the original and defaced volumes. Source code is available at https://github.com/cai4cai/defacing_pipeline.
>
---
#### [new 237] Segmentation of temporomandibular joint structures on mri images using neural networks for diagnosis of pathologies
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文研究基于神经网络的颞下颌关节（TMJ）MRI图像分割任务，旨在解决现有工具无法精准分割关节盘的问题。通过构建94张图像的数据集并增强数据，对比了U-Net、YOLO系列和Roboflow模型性能，最终验证Roboflow在分割任务中的潜力，为TMJ病理诊断提供自动化支持。**

- **链接: [http://arxiv.org/pdf/2505.12963v1](http://arxiv.org/pdf/2505.12963v1)**

> **作者:** Maksim I. Ivanov; Olga E. Mendybaeva; Yuri E. Karyakin; Igor N. Glukhikh; Aleksey V. Lebedev
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** This article explores the use of artificial intelligence for the diagnosis of pathologies of the temporomandibular joint (TMJ), in particular, for the segmentation of the articular disc on MRI images. The relevance of the work is due to the high prevalence of TMJ pathologies, as well as the need to improve the accuracy and speed of diagnosis in medical institutions. During the study, the existing solutions (Diagnocat, MandSeg) were analyzed, which, as a result, are not suitable for studying the articular disc due to the orientation towards bone structures. To solve the problem, an original dataset was collected from 94 images with the classes "temporomandibular joint" and "jaw". To increase the amount of data, augmentation methods were used. After that, the models of U-Net, YOLOv8n, YOLOv11n and Roboflow neural networks were trained and compared. The evaluation was carried out according to the Dice Score, Precision, Sensitivity, Specificity, and Mean Average Precision metrics. The results confirm the potential of using the Roboflow model for segmentation of the temporomandibular joint. In the future, it is planned to develop an algorithm for measuring the distance between the jaws and determining the position of the articular disc, which will improve the diagnosis of TMJ pathologies.
>
---
#### [new 238] Patient-Specific Autoregressive Models for Organ Motion Prediction in Radiotherapy
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于放疗器官运动预测任务，解决现有PCA方法依赖配准质量、难捕捉周期性动态的问题。提出患者特定自回归模型，将4D CT扫描的连续相位作为输入，基于历史运动预测未来器官位置。实验使用70名患者数据验证，在肺/心脏运动预测上超越传统方法，提升放疗精准度。**

- **链接: [http://arxiv.org/pdf/2505.11832v1](http://arxiv.org/pdf/2505.11832v1)**

> **作者:** Yuxiang Lai; Jike Zhong; Vanessa Su; Xiaofeng Yang
>
> **摘要:** Radiotherapy often involves a prolonged treatment period. During this time, patients may experience organ motion due to breathing and other physiological factors. Predicting and modeling this motion before treatment is crucial for ensuring precise radiation delivery. However, existing pre-treatment organ motion prediction methods primarily rely on deformation analysis using principal component analysis (PCA), which is highly dependent on registration quality and struggles to capture periodic temporal dynamics for motion modeling.In this paper, we observe that organ motion prediction closely resembles an autoregressive process, a technique widely used in natural language processing (NLP). Autoregressive models predict the next token based on previous inputs, naturally aligning with our objective of predicting future organ motion phases. Building on this insight, we reformulate organ motion prediction as an autoregressive process to better capture patient-specific motion patterns. Specifically, we acquire 4D CT scans for each patient before treatment, with each sequence comprising multiple 3D CT phases. These phases are fed into the autoregressive model to predict future phases based on prior phase motion patterns. We evaluate our method on a real-world test set of 4D CT scans from 50 patients who underwent radiotherapy at our institution and a public dataset containing 4D CT scans from 20 patients (some with multiple scans), totaling over 1,300 3D CT phases. The performance in predicting the motion of the lung and heart surpasses existing benchmarks, demonstrating its effectiveness in capturing motion dynamics from CT images. These results highlight the potential of our method to improve pre-treatment planning in radiotherapy, enabling more precise and adaptive radiation delivery.
>
---
#### [new 239] STAR: Stage-Wise Attention-Guided Token Reduction for Efficient Large Vision-Language Models Inference
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对大视觉语言模型推理时视觉token计算开销高的问题，提出STAR框架。通过视觉自注意力（早期剪枝冗余低级特征）和跨模态注意力（后期剪枝任务无关token）两阶段协同优化，在保持性能的同时显著降低计算成本，属于模型推理加速任务。**

- **链接: [http://arxiv.org/pdf/2505.12359v1](http://arxiv.org/pdf/2505.12359v1)**

> **作者:** Yichen Guo; Hanze Li; Zonghao Zhang; Jinhao You; Kai Tang; Xiande Huang
>
> **摘要:** Although large vision-language models (LVLMs) leverage rich visual token representations to achieve strong performance on multimodal tasks, these tokens also introduce significant computational overhead during inference. Existing training-free token pruning methods typically adopt a single-stage strategy, focusing either on visual self-attention or visual-textual cross-attention. However, such localized perspectives often overlook the broader information flow across the model, leading to substantial performance degradation, especially under high pruning ratios. In this work, we propose STAR (Stage-wise Attention-guided token Reduction), a training-free, plug-and-play framework that approaches token pruning from a global perspective. Instead of pruning at a single point, STAR performs attention-guided reduction in two complementary stages: an early-stage pruning based on visual self-attention to remove redundant low-level features, and a later-stage pruning guided by cross-modal attention to discard task-irrelevant tokens. This holistic approach allows STAR to significantly reduce computational cost while better preserving task-critical information. Extensive experiments across multiple LVLM architectures and benchmarks show that STAR achieves strong acceleration while maintaining comparable, and in some cases even improved performance.
>
---
#### [new 240] StarFT: Robust Fine-tuning of Zero-shot Models via Spuriosity Alignment
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于鲁棒模型微调任务，解决零样本模型在下游任务中因学习虚假特征（如背景）导致性能下降的问题。提出StarFT框架，通过正则化对齐虚假标签的输出分布，并利用语言模型生成干扰性文本描述，抑制模型提取无关特征。实验表明其显著提升了组鲁棒性和分类性能。**

- **链接: [http://arxiv.org/pdf/2505.13232v1](http://arxiv.org/pdf/2505.13232v1)**

> **作者:** Younghyun Kim; Jongheon Jeong; Sangkyung Kwak; Kyungmin Lee; Juho Lee; Jinwoo Shin
>
> **摘要:** Learning robust representations from data often requires scale, which has led to the success of recent zero-shot models such as CLIP. However, the obtained robustness can easily be deteriorated when these models are fine-tuned on other downstream tasks (e.g., of smaller scales). Previous works often interpret this phenomenon in the context of domain shift, developing fine-tuning methods that aim to preserve the original domain as much as possible. However, in a different context, fine-tuned models with limited data are also prone to learning features that are spurious to humans, such as background or texture. In this paper, we propose StarFT (Spurious Textual Alignment Regularization), a novel framework for fine-tuning zero-shot models to enhance robustness by preventing them from learning spuriosity. We introduce a regularization that aligns the output distribution for spuriosity-injected labels with the original zero-shot model, ensuring that the model is not induced to extract irrelevant features further from these descriptions.We leverage recent language models to get such spuriosity-injected labels by generating alternative textual descriptions that highlight potentially confounding features.Extensive experiments validate the robust generalization of StarFT and its emerging properties: zero-shot group robustness and improved zero-shot classification. Notably, StarFT boosts both worst-group and average accuracy by 14.30% and 3.02%, respectively, in the Waterbirds group shift scenario, where other robust fine-tuning baselines show even degraded performance.
>
---
#### [new 241] Behind the Screens: Uncovering Bias in AI-Driven Video Interview Assessments Using Counterfactuals
- **分类: cs.HC; cs.CV; eess.IV**

- **简介: 该论文属于AI公平性评估任务，旨在解决招聘场景中AI人格评估模型的偏见问题。提出基于反事实的框架，利用GAN生成不同受保护属性（性别/种族/年龄）的应聘者多模态数据，量化分析黑盒模型输出的群体差异，为商业平台提供无需模型内部数据的可扩展公平性审计工具。**

- **链接: [http://arxiv.org/pdf/2505.12114v1](http://arxiv.org/pdf/2505.12114v1)**

> **作者:** Dena F. Mujtaba; Nihar R. Mahapatra
>
> **摘要:** AI-enhanced personality assessments are increasingly shaping hiring decisions, using affective computing to predict traits from the Big Five (OCEAN) model. However, integrating AI into these assessments raises ethical concerns, especially around bias amplification rooted in training data. These biases can lead to discriminatory outcomes based on protected attributes like gender, ethnicity, and age. To address this, we introduce a counterfactual-based framework to systematically evaluate and quantify bias in AI-driven personality assessments. Our approach employs generative adversarial networks (GANs) to generate counterfactual representations of job applicants by altering protected attributes, enabling fairness analysis without access to the underlying model. Unlike traditional bias assessments that focus on unimodal or static data, our method supports multimodal evaluation-spanning visual, audio, and textual features. This comprehensive approach is particularly important in high-stakes applications like hiring, where third-party vendors often provide AI systems as black boxes. Applied to a state-of-the-art personality prediction model, our method reveals significant disparities across demographic groups. We also validate our framework using a protected attribute classifier to confirm the effectiveness of our counterfactual generation. This work provides a scalable tool for fairness auditing of commercial AI hiring platforms, especially in black-box settings where training data and model internals are inaccessible. Our results highlight the importance of counterfactual approaches in improving ethical transparency in affective computing.
>
---
#### [new 242] Concept-Guided Interpretability via Neural Chunking
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于神经网络可解释性研究，旨在揭示模型内部机制。针对神经网络"黑箱"问题，提出"反射假说"：网络活动模式反映数据规律。基于认知分块理论，开发三种方法（DSC/PA/UCD）分割高维动态为概念单元，验证其在不同规模模型中提取可解释实体的有效性，并通过操控实体改变网络行为，为理解复杂系统提供新途径。**

- **链接: [http://arxiv.org/pdf/2505.11576v1](http://arxiv.org/pdf/2505.11576v1)**

> **作者:** Shuchen Wu; Stephan Alaniz; Shyamgopal Karthik; Peter Dayan; Eric Schulz; Zeynep Akata
>
> **备注:** 35 pages, 32 figures. arXiv admin note: text overlap with arXiv:2502.01803
>
> **摘要:** Neural networks are often black boxes, reflecting the significant challenge of understanding their internal workings. We propose a different perspective that challenges the prevailing view: rather than being inscrutable, neural networks exhibit patterns in their raw population activity that mirror regularities in the training data. We refer to this as the Reflection Hypothesis and provide evidence for this phenomenon in both simple recurrent neural networks (RNNs) and complex large language models (LLMs). Building on this insight, we propose to leverage cognitively-inspired methods of chunking to segment high-dimensional neural population dynamics into interpretable units that reflect underlying concepts. We propose three methods to extract these emerging entities, complementing each other based on label availability and dimensionality. Discrete sequence chunking (DSC) creates a dictionary of entities; population averaging (PA) extracts recurring entities that correspond to known labels; and unsupervised chunk discovery (UCD) can be used when labels are absent. We demonstrate the effectiveness of these methods in extracting entities across varying model sizes, ranging from inducing compositionality in RNNs to uncovering recurring neural population states in large models with diverse architectures, and illustrate their advantage over other methods. Throughout, we observe a robust correspondence between the extracted entities and concrete or abstract concepts. Artificially inducing the extracted entities in neural populations effectively alters the network's generation of associated concepts. Our work points to a new direction for interpretability, one that harnesses both cognitive principles and the structure of naturalistic data to reveal the hidden computations of complex learning systems, gradually transforming them from black boxes into systems we can begin to understand.
>
---
#### [new 243] MINGLE: Mixtures of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究测试时持续模型合并任务，解决参数干扰和分布变化导致的灾难性遗忘及适应性不足问题。提出MINGLE框架，通过空门控混合低秩专家动态调整模型合并，结合自适应约束策略平衡新旧任务性能，实验显示其显著减少遗忘并提升泛化能力，优于现有方法7-9%。**

- **链接: [http://arxiv.org/pdf/2505.11883v1](http://arxiv.org/pdf/2505.11883v1)**

> **作者:** Zihuan Qiu; Yi Xu; Chiyuan He; Fanman Meng; Linfeng Xu; Qingbo Wu; Hongliang Li
>
> **摘要:** Continual model merging integrates independently fine-tuned models sequentially without access to original training data, providing a scalable and efficient solution to continual learning. However, current methods still face critical challenges, notably parameter interference among tasks and limited adaptability to evolving test distributions. The former causes catastrophic forgetting of integrated tasks, while the latter hinders effective adaptation to new tasks. To address these, we propose MINGLE, a novel framework for test-time continual model merging, which leverages test-time adaptation using a small set of unlabeled test samples from the current task to dynamically guide the merging process. MINGLE employs a mixture-of-experts architecture composed of parameter-efficient, low-rank experts, enabling efficient adaptation and improving robustness to distribution shifts. To mitigate catastrophic forgetting, we propose Null-Space Constrained Gating, which restricts gating updates to subspaces orthogonal to prior task representations. This suppresses activations on old task inputs and preserves model behavior on past tasks. To further balance stability and adaptability, we design an Adaptive Relaxation Strategy, which dynamically adjusts the constraint strength based on interference signals captured during test-time adaptation. Extensive experiments on standard continual merging benchmarks demonstrate that MINGLE achieves robust generalization, reduces forgetting significantly, and consistently surpasses previous state-of-the-art methods by 7-9\% on average across diverse task orders.
>
---
#### [new 244] Experimental Study on Automatically Assembling Custom Catering Packages With a 3-DOF Delta Robot Using Deep Learning Methods
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究基于3自由度Delta机器人的餐饮包自动装配任务，解决传统包装自动化中物体抓取精度问题。通过构建首个波斯产品数据集（1500图），结合YOLOV5检测、FastSAM分割和几何特征计算抓取点，实现实时检测校准，最终由机器人完成自主包装，抓取成功率超80%。**

- **链接: [http://arxiv.org/pdf/2505.11879v1](http://arxiv.org/pdf/2505.11879v1)**

> **作者:** Reihaneh Yourdkhani; Arash Tavoosian; Navid Asadi Khomami; Mehdi Tale Masouleh
>
> **摘要:** This paper introduces a pioneering experimental study on the automated packing of a catering package using a two-fingered gripper affixed to a 3-degree-of-freedom Delta parallel robot. A distinctive contribution lies in the application of a deep learning approach to tackle this challenge. A custom dataset, comprising 1,500 images, is meticulously curated for this endeavor, representing a noteworthy initiative as the first dataset focusing on Persian-manufactured products. The study employs the YOLOV5 model for object detection, followed by segmentation using the FastSAM model. Subsequently, rotation angle calculation is facilitated with segmentation masks, and a rotated rectangle encapsulating the object is generated. This rectangle forms the basis for calculating two grasp points using a novel geometrical approach involving eigenvectors. An extensive experimental study validates the proposed model, where all pertinent information is seamlessly transmitted to the 3-DOF Delta parallel robot. The proposed algorithm ensures real-time detection, calibration, and the fully autonomous packing process of a catering package, boasting an impressive over 80\% success rate in automatic grasping. This study marks a significant stride in advancing the capabilities of robotic systems for practical applications in packaging automation.
>
---
#### [new 245] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文研究大模型高效微调，解决GPU内存瓶颈问题。提出QZO方法，结合零阶优化（消除梯度和优化器状态）与模型量化（4-bit权重），通过扰动量化尺度估计梯度并稳定训练，相比全参数微调减少18倍内存，实现单卡微调Llama-13B等模型。**

- **链接: [http://arxiv.org/pdf/2505.13430v1](http://arxiv.org/pdf/2505.13430v1)**

> **作者:** Sifeng Shang; Jiayi Zhou; Chenyu Lin; Minxian Li; Kaiyang Zhou
>
> **摘要:** As the size of large language models grows exponentially, GPU memory has become a bottleneck for adapting these models to downstream tasks. In this paper, we aim to push the limits of memory-efficient training by minimizing memory usage on model weights, gradients, and optimizer states, within a unified framework. Our idea is to eliminate both gradients and optimizer states using zeroth-order optimization, which approximates gradients by perturbing weights during forward passes to identify gradient directions. To minimize memory usage on weights, we employ model quantization, e.g., converting from bfloat16 to int4. However, directly applying zeroth-order optimization to quantized weights is infeasible due to the precision gap between discrete weights and continuous gradients, which would otherwise require de-quantization and re-quantization. To overcome this challenge, we propose Quantized Zeroth-order Optimization (QZO), a novel approach that perturbs the continuous quantization scale for gradient estimation and uses a directional derivative clipping method to stabilize training. QZO is orthogonal to both scalar-based and codebook-based post-training quantization methods. Compared to full-parameter fine-tuning in bfloat16, QZO can reduce the total memory cost by more than 18$\times$ for 4-bit LLMs, and enables fine-tuning Llama-2-13B and Stable Diffusion 3.5 Large within a single 24GB GPU.
>
---
#### [new 246] AdaToken-3D: Dynamic Spatial Gating for Efficient 3D Large Multimodal-Models Reasoning
- **分类: cs.GR; cs.CV; cs.IR; cs.IT; math.IT**

- **简介: 该论文属于3D多模态模型优化任务，旨在解决现有3D大型多模态模型（LMMs）因空间令牌冗余导致的低效问题。提出AdaToken-3D框架，通过动态空间令牌剪枝和注意力模式分析，自适应优化信息流。实验证明其能在保持精度的同时提升21%推理速度、减少63%计算量，并揭示超60%空间令牌对预测贡献极低的理论依据。**

- **链接: [http://arxiv.org/pdf/2505.12782v1](http://arxiv.org/pdf/2505.12782v1)**

> **作者:** Kai Zhang; Xingyu Chen; Xiaofeng Zhang
>
> **摘要:** Large Multimodal Models (LMMs) have become a pivotal research focus in deep learning, demonstrating remarkable capabilities in 3D scene understanding. However, current 3D LMMs employing thousands of spatial tokens for multimodal reasoning suffer from critical inefficiencies: excessive computational overhead and redundant information flows. Unlike 2D VLMs processing single images, 3D LMMs exhibit inherent architectural redundancy due to the heterogeneous mechanisms between spatial tokens and visual tokens. To address this challenge, we propose AdaToken-3D, an adaptive spatial token optimization framework that dynamically prunes redundant tokens through spatial contribution analysis. Our method automatically tailors pruning strategies to different 3D LMM architectures by quantifying token-level information flows via attention pattern mining. Extensive experiments on LLaVA-3D (a 7B parameter 3D-LMM) demonstrate that AdaToken-3D achieves 21\% faster inference speed and 63\% FLOPs reduction while maintaining original task accuracy. Beyond efficiency gains, this work systematically investigates redundancy patterns in multimodal spatial information flows through quantitative token interaction analysis. Our findings reveal that over 60\% of spatial tokens contribute minimally ($<$5\%) to the final predictions, establishing theoretical foundations for efficient 3D multimodal learning.
>
---
#### [new 247] PRETI: Patient-Aware Retinal Foundation Model via Metadata-Guided Representation Learning
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出PRETI视网膜基础模型，属于医学图像分析任务，旨在解决依赖临床报告的高成本问题。通过整合元数据引导学习和自监督表征学习，引入可学习元数据嵌入(LME)优化患者信息融合，构建患者级数据对增强鲁棒性，并设计视网膜自适应掩蔽策略(RAAM)提升病理特征提取，在多种视网膜疾病诊断中实现最优性能。**

- **链接: [http://arxiv.org/pdf/2505.12233v1](http://arxiv.org/pdf/2505.12233v1)**

> **作者:** Yeonkyung Lee; Woojung Han; Youngjun Jun; Hyeonmin Kim; Jungkyung Cho; Seong Jae Hwang
>
> **备注:** MICCAI2025 early accept
>
> **摘要:** Retinal foundation models have significantly advanced retinal image analysis by leveraging self-supervised learning to reduce dependence on labeled data while achieving strong generalization. Many recent approaches enhance retinal image understanding using report supervision, but obtaining clinical reports is often costly and challenging. In contrast, metadata (e.g., age, gender) is widely available and serves as a valuable resource for analyzing disease progression. To effectively incorporate patient-specific information, we propose PRETI, a retinal foundation model that integrates metadata-aware learning with robust self-supervised representation learning. We introduce Learnable Metadata Embedding (LME), which dynamically refines metadata representations. Additionally, we construct patient-level data pairs, associating images from the same individual to improve robustness against non-clinical variations. To further optimize retinal image representation, we propose Retina-Aware Adaptive Masking (RAAM), a strategy that selectively applies masking within the retinal region and dynamically adjusts the masking ratio during training. PRETI captures both global structures and fine-grained pathological details, resulting in superior diagnostic performance. Extensive experiments demonstrate that PRETI achieves state-of-the-art results across diverse diseases and biomarker predictions using in-house and public data, indicating the importance of metadata-guided foundation models in retinal disease analysis. Our code and pretrained model are available at https://github.com/MICV-yonsei/PRETI
>
---
#### [new 248] Model alignment using inter-modal bridges
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究跨模态模型对齐任务，解决不同模态（如文本与图像）间表征对齐困难导致模型复用受限的问题。提出基于条件流匹配的半监督方法，通过最优传输或标记样本学习潜在空间映射，在数据稀缺（<20%）时匹配端到端模型性能，适用于物体识别和图像生成任务。**

- **链接: [http://arxiv.org/pdf/2505.12322v1](http://arxiv.org/pdf/2505.12322v1)**

> **作者:** Ali Gholamzadeh; Noor Sajid
>
> **摘要:** Foundation models have demonstrated remarkable performance across modalities such as language and vision. However, model reuse across distinct modalities (e.g., text and vision) remains limited due to the difficulty of aligning internal representations. Existing methods require extensive paired training data or are constrained to specific domains. We introduce a semi-supervised approach for model alignment via conditional flow matching. The conditional flow between latent spaces of different modalities (e.g., text-to-image or biological-to-artificial neuronal activity) can be learned in two settings: ($1$) solving a (balanced or unbalanced) optimal transport problem with an inter-space bridge cost, and ($2$) performing memory-efficient alignment using labelled exemplars. Despite being constrained by the original models' capacity, our method--under both settings--matches downstream task performance of end-to-end trained models on object recognition and image generation tasks across MNIST, ImageNet, and \cite{majaj2015simple} datasets, particularly when labelled training data is scarce ($<20\%$). Our method provides a data-efficient solution for inter-modal model alignment with minimal supervision.
>
---
#### [new 249] UniHM: Universal Human Motion Generation with Object Interactions in Indoor Scenes
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文提出UniHM框架，解决复杂室内场景中多模态整合（环境/物体/语言）的人体运动生成问题。通过混合运动表征、新型LFQ-VAE模型及增强数据集，实现文本到动作和人物-物体交互生成，提升场景感知运动的连续性与真实性。属于3D人体运动合成任务。**

- **链接: [http://arxiv.org/pdf/2505.12774v1](http://arxiv.org/pdf/2505.12774v1)**

> **作者:** Zichen Geng; Zeeshan Hayder; Wei Liu; Ajmal Mian
>
> **摘要:** Human motion synthesis in complex scenes presents a fundamental challenge, extending beyond conventional Text-to-Motion tasks by requiring the integration of diverse modalities such as static environments, movable objects, natural language prompts, and spatial waypoints. Existing language-conditioned motion models often struggle with scene-aware motion generation due to limitations in motion tokenization, which leads to information loss and fails to capture the continuous, context-dependent nature of 3D human movement. To address these issues, we propose UniHM, a unified motion language model that leverages diffusion-based generation for synthesizing scene-aware human motion. UniHM is the first framework to support both Text-to-Motion and Text-to-Human-Object Interaction (HOI) in complex 3D scenes. Our approach introduces three key contributions: (1) a mixed-motion representation that fuses continuous 6DoF motion with discrete local motion tokens to improve motion realism; (2) a novel Look-Up-Free Quantization VAE (LFQ-VAE) that surpasses traditional VQ-VAEs in both reconstruction accuracy and generative performance; and (3) an enriched version of the Lingo dataset augmented with HumanML3D annotations, providing stronger supervision for scene-specific motion learning. Experimental results demonstrate that UniHM achieves comparative performance on the OMOMO benchmark for text-to-HOI synthesis and yields competitive results on HumanML3D for general text-conditioned motion generation.
>
---
#### [new 250] GuidedMorph: Two-Stage Deformable Registration for Breast MRI
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像配准任务，旨在解决乳腺MRI中密集组织因非刚性形变导致的配准难题。提出两阶段框架GuidedMorph，通过单尺度网络全局对齐、双空间变换网络融合形变场，结合欧氏距离变换的形变方法保留细节，在ISPY2等数据集上验证了配准精度提升。**

- **链接: [http://arxiv.org/pdf/2505.13414v1](http://arxiv.org/pdf/2505.13414v1)**

> **作者:** Yaqian Chen; Hanxue Gu; Haoyu Dong; Qihang Li; Yuwen Chen; Nicholas Konz; Lin Li; Maciej A. Mazurowski
>
> **摘要:** Accurately registering breast MR images from different time points enables the alignment of anatomical structures and tracking of tumor progression, supporting more effective breast cancer detection, diagnosis, and treatment planning. However, the complexity of dense tissue and its highly non-rigid nature pose challenges for conventional registration methods, which primarily focus on aligning general structures while overlooking intricate internal details. To address this, we propose \textbf{GuidedMorph}, a novel two-stage registration framework designed to better align dense tissue. In addition to a single-scale network for global structure alignment, we introduce a framework that utilizes dense tissue information to track breast movement. The learned transformation fields are fused by introducing the Dual Spatial Transformer Network (DSTN), improving overall alignment accuracy. A novel warping method based on the Euclidean distance transform (EDT) is also proposed to accurately warp the registered dense tissue and breast masks, preserving fine structural details during deformation. The framework supports paradigms that require external segmentation models and with image data only. It also operates effectively with the VoxelMorph and TransMorph backbones, offering a versatile solution for breast registration. We validate our method on ISPY2 and internal dataset, demonstrating superior performance in dense tissue, overall breast alignment, and breast structural similarity index measure (SSIM), with notable improvements by over 13.01% in dense tissue Dice, 3.13% in breast Dice, and 1.21% in breast SSIM compared to the best learning-based baseline.
>
---
#### [new 251] Scalable Strategies for Continual Learning with Replay
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究持续学习任务，解决回放方法效率低、多任务微调技术未整合的问题。提出三阶段方案：应用低秩适应优化参数效率，设计阶段性回放减少55%样本需求，开发顺序合并技术融合任务特征。最终整合策略形成高效工具集，提升模型在连续任务中的知识整合能力。**

- **链接: [http://arxiv.org/pdf/2505.12512v1](http://arxiv.org/pdf/2505.12512v1)**

> **作者:** Truman Hickok
>
> **摘要:** Future deep learning models will be distinguished by systems that perpetually learn through interaction, imagination, and cooperation, blurring the line between training and inference. This makes continual learning a critical challenge, as methods that efficiently maximize bidirectional transfer across learning trajectories will be essential. Replay is on track to play a foundational role in continual learning, allowing models to directly reconcile new information with past knowledge. In practice, however, replay is quite unscalable, doubling the cost of continual learning when applied naively. Moreover, the continual learning literature has not fully synchronized with the multi-task fine-tuning literature, having not fully integrated highly scalable techniques like model merging and low rank adaptation into a replay-enabled toolset that can produce a unified model in the face of many sequential tasks. In this paper, we begin by applying and analyzing low rank adaptation in a continual learning setting. Next, we introduce consolidation, a phasic approach to replay which leads to up to 55\% less replay samples being needed for a given performance target. Then, we propose sequential merging, an offshoot of task arithmetic which is tailored to the continual learning setting and is shown to work well in combination with replay. Finally, we demonstrate that the developed strategies can operate synergistically, resulting in a highly scalable toolset that outperforms standalone variants.
>
---
#### [new 252] The Gaussian Latent Machine: Efficient Prior and Posterior Sampling for Inverse Problems
- **分类: eess.IV; cs.CV; cs.LG; stat.ML; 65C40, 65C05, 68U10, 65C60**

- **简介: 该论文提出高斯潜在机器，解决贝叶斯成像中先验与后验分布的高效采样问题。通过将多专家模型转化为潜变量形式，统一现有采样算法，并设计高效的双块吉布斯采样方法，实验验证了其在成像任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.12836v1](http://arxiv.org/pdf/2505.12836v1)**

> **作者:** Muhamed Kuric; Martin Zach; Andreas Habring; Michael Unser; Thomas Pock
>
> **摘要:** We consider the problem of sampling from a product-of-experts-type model that encompasses many standard prior and posterior distributions commonly found in Bayesian imaging. We show that this model can be easily lifted into a novel latent variable model, which we refer to as a Gaussian latent machine. This leads to a general sampling approach that unifies and generalizes many existing sampling algorithms in the literature. Most notably, it yields a highly efficient and effective two-block Gibbs sampling approach in the general case, while also specializing to direct sampling algorithms in particular cases. Finally, we present detailed numerical experiments that demonstrate the efficiency and effectiveness of our proposed sampling approach across a wide range of prior and posterior sampling problems from Bayesian imaging.
>
---
#### [new 253] EnvInjection: Environmental Prompt Injection Attack to Multi-modal Web Agents
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文针对多模态大语言模型网页代理的环境提示注入攻击任务，解决现有攻击有效性低、隐蔽性差的问题。通过像素级扰动修改网页源码，诱导代理执行指定动作，利用神经网络近似不可微分映射并用梯度下降优化攻击，显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11717v1](http://arxiv.org/pdf/2505.11717v1)**

> **作者:** Xilong Wang; John Bloch; Zedian Shao; Yuepeng Hu; Shuyan Zhou; Neil Zhenqiang Gong
>
> **摘要:** Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. Environmental prompt injection attacks manipulate the environment to induce the web agent to perform a specific, attacker-chosen action--referred to as the target action. However, existing attacks suffer from limited effectiveness or stealthiness, or are impractical in real-world settings. In this work, we propose EnvInjection, a new attack that addresses these limitations. Our attack adds a perturbation to the raw pixel values of the rendered webpage, which can be implemented by modifying the webpage's source code. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the target action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple webpage datasets shows that EnvInjection is highly effective and significantly outperforms existing baselines.
>
---
#### [new 254] FreqSelect: Frequency-Aware fMRI-to-Image Reconstruction
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于fMRI到图像的生成任务，旨在解决现有方法因平等处理所有空间频率导致的噪声抑制和特征提取效率低的问题。提出FreqSelect模块，在编码前自适应筛选关键频段，强调预测性频率并抑制噪声，整合到VAE-扩散模型中，提升了重建质量和神经表征可解释性。**

- **链接: [http://arxiv.org/pdf/2505.12552v1](http://arxiv.org/pdf/2505.12552v1)**

> **作者:** Junliang Ye; Lei Wang; Md Zakir Hossain
>
> **备注:** Research report
>
> **摘要:** Reconstructing natural images from functional magnetic resonance imaging (fMRI) data remains a core challenge in natural decoding due to the mismatch between the richness of visual stimuli and the noisy, low resolution nature of fMRI signals. While recent two-stage models, combining deep variational autoencoders (VAEs) with diffusion models, have advanced this task, they treat all spatial-frequency components of the input equally. This uniform treatment forces the model to extract meaning features and suppress irrelevant noise simultaneously, limiting its effectiveness. We introduce FreqSelect, a lightweight, adaptive module that selectively filters spatial-frequency bands before encoding. By dynamically emphasizing frequencies that are most predictive of brain activity and suppressing those that are uninformative, FreqSelect acts as a content-aware gate between image features and natural data. It integrates seamlessly into standard very deep VAE-diffusion pipelines and requires no additional supervision. Evaluated on the Natural Scenes dataset, FreqSelect consistently improves reconstruction quality across both low- and high-level metrics. Beyond performance gains, the learned frequency-selection patterns offer interpretable insights into how different visual frequencies are represented in the brain. Our method generalizes across subjects and scenes, and holds promise for extension to other neuroimaging modalities, offering a principled approach to enhancing both decoding accuracy and neuroscientific interpretability.
>
---
#### [new 255] Mitigating Hallucinations via Inter-Layer Consistency Aggregation in Large Vision-Language Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对大型视觉语言模型（LVLMs）的幻觉问题（生成与图像不符的内容），提出无需训练的层间一致性聚合解码方法DCLA。通过动态聚合中间层语义表征并校正偏差层，增强模型层间一致性，在MME/POPE基准上有效降低幻觉，提升模型可靠性。**

- **链接: [http://arxiv.org/pdf/2505.12343v1](http://arxiv.org/pdf/2505.12343v1)**

> **作者:** Kai Tang; Jinhao You; Xiuqi Ge; Hanze Li; Yichen Guo; Xiande Huang
>
> **摘要:** Despite the impressive capabilities of Large Vision-Language Models (LVLMs), they remain susceptible to hallucinations-generating content that is inconsistent with the input image. Existing training-free hallucination mitigation methods often suffer from unstable performance and high sensitivity to hyperparameter settings, limiting their practicality and broader adoption. In this paper, we propose a novel decoding mechanism, Decoding with Inter-layer Consistency via Layer Aggregation (DCLA), which requires no retraining, fine-tuning, or access to external knowledge bases. Specifically, our approach constructs a dynamic semantic reference by aggregating representations from previous layers, and corrects semantically deviated layers to enforce inter-layer consistency. The method allows DCLA to robustly mitigate hallucinations across multiple LVLMs. Experiments on hallucination benchmarks such as MME and POPE demonstrate that DCLA effectively reduces hallucinations while enhancing the reliability and performance of LVLMs.
>
---
#### [new 256] RBF++: Quantifying and Optimizing Reasoning Boundaries across Measurable and Unmeasurable Capabilities for Chain-of-Thought Reasoning
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文研究大语言模型的思维链推理优化，解决可测与不可测能力边界的量化评估问题。提出RBF++框架：通过组合定律量化可测推理边界，采用常数假设和边界划分机制处理多模态等不可测场景，经38模型跨13任务验证有效性，扩展了推理评估基准。**

- **链接: [http://arxiv.org/pdf/2505.13307v1](http://arxiv.org/pdf/2505.13307v1)**

> **作者:** Qiguang Chen; Libo Qin; Jinhao Liu; Yue Liao; Jiaqi Wang; Jingxuan Zhou; Wanxiang Che
>
> **备注:** Manuscript
>
> **摘要:** Chain-of-Thought (CoT) reasoning has proven effective in enhancing large language models (LLMs) on complex tasks, spurring research into its underlying mechanisms. However, two primary challenges remain for real-world applications: (1) the lack of quantitative metrics and actionable guidelines for evaluating and optimizing measurable boundaries of CoT capability, and (2) the absence of methods to assess boundaries of unmeasurable CoT capability, such as multimodal perception. To address these gaps, we introduce the Reasoning Boundary Framework++ (RBF++). To tackle the first challenge, we define the reasoning boundary (RB) as the maximum limit of CoT performance. We also propose a combination law for RBs, enabling quantitative analysis and offering actionable guidance across various CoT tasks. For the second challenge, particularly in multimodal scenarios, we introduce a constant assumption, which replaces unmeasurable RBs with scenario-specific constants. Additionally, we propose the reasoning boundary division mechanism, which divides unmeasurable RBs into two sub-boundaries, facilitating the quantification and optimization of both unmeasurable domain knowledge and multimodal perception capabilities. Extensive experiments involving 38 models across 13 tasks validate the feasibility of our framework in cross-modal settings. Additionally, we evaluate 10 CoT strategies, offer insights into optimization and decay from two complementary perspectives, and expand evaluation benchmarks for measuring RBs in LLM reasoning. We hope this work advances the understanding of RBs and optimization strategies in LLMs. Code and data are available at https://github.com/LightChen233/reasoning-boundary.
>
---
#### [new 257] NTIRE 2025 Challenge on Efficient Burst HDR and Restoration: Datasets, Methods, and Results
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文聚焦高效多帧HDR融合及图像复原任务，解决在参数与计算量限制下处理噪声、未对齐多曝光RAW帧的难题。通过组织挑战赛构建新型数据集，评估并对比参赛方案，最优方法PSNR达43.22dB，为高效算法研究提供参考。**

- **链接: [http://arxiv.org/pdf/2505.12089v1](http://arxiv.org/pdf/2505.12089v1)**

> **作者:** Sangmin Lee; Eunpil Park; Angel Canelo; Hyunhee Park; Youngjo Kim; Hyung-Ju Chun; Xin Jin; Chongyi Li; Chun-Le Guo; Radu Timofte; Qi Wu; Tianheng Qiu; Yuchun Dong; Shenglin Ding; Guanghua Pan; Weiyu Zhou; Tao Hu; Yixu Feng; Duwei Dai; Yu Cao; Peng Wu; Wei Dong; Yanning Zhang; Qingsen Yan; Simon J. Larsen; Ruixuan Jiang; Senyan Xu; Xingbo Wang; Xin Lu; Marcos V. Conde; Javier Abad-Hernandez; Alvaro Garcıa-Lara; Daniel Feijoo; Alvaro Garcıa; Zeyu Xiao; Zhuoyuan Li
>
> **摘要:** This paper reviews the NTIRE 2025 Efficient Burst HDR and Restoration Challenge, which aims to advance efficient multi-frame high dynamic range (HDR) and restoration techniques. The challenge is based on a novel RAW multi-frame fusion dataset, comprising nine noisy and misaligned RAW frames with various exposure levels per scene. Participants were tasked with developing solutions capable of effectively fusing these frames while adhering to strict efficiency constraints: fewer than 30 million model parameters and a computational budget under 4.0 trillion FLOPs. A total of 217 participants registered, with six teams finally submitting valid solutions. The top-performing approach achieved a PSNR of 43.22 dB, showcasing the potential of novel methods in this domain. This paper provides a comprehensive overview of the challenge, compares the proposed solutions, and serves as a valuable reference for researchers and practitioners in efficient burst HDR and restoration.
>
---
#### [new 258] Unified Cross-modal Translation of Score Images, Symbolic Music, and Performance Audio
- **分类: cs.SD; cs.AI; cs.CV; eess.AS**

- **简介: 该论文提出统一跨模态音乐翻译模型，解决传统方法需为各任务单独训练的问题。通过构建1300小时配对数据集及统一标记化框架，将乐谱图像、符号音乐和音频转化为序列，使用单一Transformer处理多任务。实验表明模型在光学识谱错误率（降至13.67%）等任务上超越单任务基线，并首次实现乐谱图像生成音频。**

- **链接: [http://arxiv.org/pdf/2505.12863v1](http://arxiv.org/pdf/2505.12863v1)**

> **作者:** Jongmin Jung; Dongmin Kim; Sihun Lee; Seola Cho; Hyungjoon Soh; Irmak Bukey; Chris Donahue; Dasaem Jeong
>
> **备注:** Submitted to IEEE Transactions on Audio, Speech and Language Processing (TASLPRO)
>
> **摘要:** Music exists in various modalities, such as score images, symbolic scores, MIDI, and audio. Translations between each modality are established as core tasks of music information retrieval, such as automatic music transcription (audio-to-MIDI) and optical music recognition (score image to symbolic score). However, most past work on multimodal translation trains specialized models on individual translation tasks. In this paper, we propose a unified approach, where we train a general-purpose model on many translation tasks simultaneously. Two key factors make this unified approach viable: a new large-scale dataset and the tokenization of each modality. Firstly, we propose a new dataset that consists of more than 1,300 hours of paired audio-score image data collected from YouTube videos, which is an order of magnitude larger than any existing music modal translation datasets. Secondly, our unified tokenization framework discretizes score images, audio, MIDI, and MusicXML into a sequence of tokens, enabling a single encoder-decoder Transformer to tackle multiple cross-modal translation as one coherent sequence-to-sequence task. Experimental results confirm that our unified multitask model improves upon single-task baselines in several key areas, notably reducing the symbol error rate for optical music recognition from 24.58% to a state-of-the-art 13.67%, while similarly substantial improvements are observed across the other translation tasks. Notably, our approach achieves the first successful score-image-conditioned audio generation, marking a significant breakthrough in cross-modal music generation.
>
---
#### [new 259] Scaling Computer-Use Grounding via User Interface Decomposition and Synthesis
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文研究图形界面（GUI）的指令映射任务，解决现有基准过于简化、无法反映真实交互复杂性的问题。提出新基准OSWorld-G（564样本）和合成数据集Jedi（400万样本），通过多尺度模型训练提升界面元素识别、布局理解和操作能力，使基础模型在复杂任务成功率从5%提升至27%，并验证了组合泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.13227v1](http://arxiv.org/pdf/2505.13227v1)**

> **作者:** Tianbao Xie; Jiaqi Deng; Xiaochuan Li; Junlin Yang; Haoyuan Wu; Jixuan Chen; Wenjing Hu; Xinyuan Wang; Yuhui Xu; Zekun Wang; Yiheng Xu; Junli Wang; Doyen Sahoo; Tao Yu; Caiming Xiong
>
> **备注:** 49 pages, 13 figures
>
> **摘要:** Graphical user interface (GUI) grounding, the ability to map natural language instructions to specific actions on graphical user interfaces, remains a critical bottleneck in computer use agent development. Current benchmarks oversimplify grounding tasks as short referring expressions, failing to capture the complexity of real-world interactions that require software commonsense, layout understanding, and fine-grained manipulation capabilities. To address these limitations, we introduce OSWorld-G, a comprehensive benchmark comprising 564 finely annotated samples across diverse task types including text matching, element recognition, layout understanding, and precise manipulation. Additionally, we synthesize and release the largest computer use grounding dataset Jedi, which contains 4 million examples through multi-perspective decoupling of tasks. Our multi-scale models trained on Jedi demonstrate its effectiveness by outperforming existing approaches on ScreenSpot-v2, ScreenSpot-Pro, and our OSWorld-G. Furthermore, we demonstrate that improved grounding with Jedi directly enhances agentic capabilities of general foundation models on complex computer tasks, improving from 5% to 27% on OSWorld. Through detailed ablation studies, we identify key factors contributing to grounding performance and verify that combining specialized data for different interface elements enables compositional generalization to novel interfaces. All benchmark, data, checkpoints, and code are open-sourced and available at https://osworld-grounding.github.io.
>
---
#### [new 260] Attention-Enhanced U-Net for Accurate Segmentation of COVID-19 Infected Lung Regions in CT Scans
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决COVID-19患者肺部CT影像中感染区域的精准分割问题。通过改进U-Net架构，引入注意力机制、数据增强和后处理技术，提升了分割精度（Dice 0.8658），验证了方法的有效性，并为临床部署奠定基础。**

- **链接: [http://arxiv.org/pdf/2505.12298v1](http://arxiv.org/pdf/2505.12298v1)**

> **作者:** Amal Lahchim; Lazar Davic
>
> **备注:** 14 pages, 9 figures, created using Google Colab and PyTorch. Compares segmentation models for COVID-19 CT data
>
> **摘要:** In this study, we propose a robust methodology for automatic segmentation of infected lung regions in COVID-19 CT scans using convolutional neural networks. The approach is based on a modified U-Net architecture enhanced with attention mechanisms, data augmentation, and postprocessing techniques. It achieved a Dice coefficient of 0.8658 and mean IoU of 0.8316, outperforming other methods. The dataset was sourced from public repositories and augmented for diversity. Results demonstrate superior segmentation performance. Future work includes expanding the dataset, exploring 3D segmentation, and preparing the model for clinical deployment.
>
---
#### [new 261] Higher fidelity perceptual image and video compression with a latent conditioned residual denoising diffusion model
- **分类: eess.IV; cs.CV**

- **简介: 该论文研究感知优化的图像/视频压缩任务，解决扩散模型压缩方法保真度不足的问题。提出混合方案：先通过解码器生成低失真图像，再用条件扩散模型预测残差优化感知质量，在保持LPIPS/FID指标的同时提升PSNR达2dB，并扩展至视频压缩。**

- **链接: [http://arxiv.org/pdf/2505.13152v1](http://arxiv.org/pdf/2505.13152v1)**

> **作者:** Jonas Brenig; Radu Timofte
>
> **备注:** Accepted at AIM Workshop 2024 at ECCV 2024
>
> **摘要:** Denoising diffusion models achieved impressive results on several image generation tasks often outperforming GAN based models. Recently, the generative capabilities of diffusion models have been employed for perceptual image compression, such as in CDC. A major drawback of these diffusion-based methods is that, while producing impressive perceptual quality images they are dropping in fidelity/increasing the distortion to the original uncompressed images when compared with other traditional or learned image compression schemes aiming for fidelity. In this paper, we propose a hybrid compression scheme optimized for perceptual quality, extending the approach of the CDC model with a decoder network in order to reduce the impact on distortion metrics such as PSNR. After using the decoder network to generate an initial image, optimized for distortion, the latent conditioned diffusion model refines the reconstruction for perceptual quality by predicting the residual. On standard benchmarks, we achieve up to +2dB PSNR fidelity improvements while maintaining comparable LPIPS and FID perceptual scores when compared with CDC. Additionally, the approach is easily extensible to video compression, where we achieve similar results.
>
---
#### [new 262] Two out of Three (ToT): using self-consistency to make robust predictions
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对深度学习模型决策不可解释的问题，提出提升预测鲁棒性的方法。为解决高风险领域错误决策隐患，开发了ToT算法：通过生成两个替代预测，结合原始结果进行自我一致性检测，当存在冲突时主动弃答，模仿人脑对矛盾信息的处理机制，增强模型可靠性。**

- **链接: [http://arxiv.org/pdf/2505.12642v1](http://arxiv.org/pdf/2505.12642v1)**

> **作者:** Jung Hoon Lee; Sujith Vijayan
>
> **备注:** 12 pages, 7 main figures, 1 supplementary table and 2 supplementary figures
>
> **摘要:** Deep learning (DL) can automatically construct intelligent agents, deep neural networks (alternatively, DL models), that can outperform humans in certain tasks. However, the operating principles of DL remain poorly understood, making its decisions incomprehensible. As a result, it poses a great risk to deploy DL in high-stakes domains in which mistakes or errors may lead to critical consequences. Here, we aim to develop an algorithm that can help DL models make more robust decisions by allowing them to abstain from answering when they are uncertain. Our algorithm, named `Two out of Three (ToT)', is inspired by the sensitivity of the human brain to conflicting information. ToT creates two alternative predictions in addition to the original model prediction and uses the alternative predictions to decide whether it should provide an answer or not.
>
---
#### [new 263] Urban Representation Learning for Fine-grained Economic Mapping: A Semi-supervised Graph-based Approach
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于细粒度经济绘图任务，旨在解决现有方法忽视半监督学习及缺乏多任务框架的问题。提出SemiGTX框架，通过图结构整合多源地理数据，结合自监督与监督回归损失，实现三产业GDP联合预测，实验显示高精度与跨区域泛化能力，增强经济分析可解释性。**

- **链接: [http://arxiv.org/pdf/2505.11645v1](http://arxiv.org/pdf/2505.11645v1)**

> **作者:** Jinzhou Cao; Xiangxu Wang; Jiashi Chen; Wei Tu; Zhenhui Li; Xindong Yang; Tianhong Zhao; Qingquan Li
>
> **备注:** Accepted for publication in International Society Journal of Photogrammetry and Remote Sensing (ISPRS). 70 pages, 10 Figures, 15 Tables
>
> **摘要:** Fine-grained economic mapping through urban representation learning has emerged as a crucial tool for evidence-based economic decisions. While existing methods primarily rely on supervised or unsupervised approaches, they often overlook semi-supervised learning in data-scarce scenarios and lack unified multi-task frameworks for comprehensive sectoral economic analysis. To address these gaps, we propose SemiGTX, an explainable semi-supervised graph learning framework for sectoral economic mapping. The framework is designed with dedicated fusion encoding modules for various geospatial data modalities, seamlessly integrating them into a cohesive graph structure. It introduces a semi-information loss function that combines spatial self-supervision with locally masked supervised regression, enabling more informative and effective region representations. Through multi-task learning, SemiGTX concurrently maps GDP across primary, secondary, and tertiary sectors within a unified model. Extensive experiments conducted in the Pearl River Delta region of China demonstrate the model's superior performance compared to existing methods, achieving R2 scores of 0.93, 0.96, and 0.94 for the primary, secondary and tertiary sectors, respectively. Cross-regional experiments in Beijing and Chengdu further illustrate its generality. Systematic analysis reveals how different data modalities influence model predictions, enhancing explainability while providing valuable insights for regional development planning. This representation learning framework advances regional economic monitoring through diverse urban data integration, providing a robust foundation for precise economic forecasting.
>
---
#### [new 264] Modeling Aesthetic Preferences in 3D Shapes: A Large-Scale Paired Comparison Study Across Object Categories
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文属于计算美学任务，旨在解决3D形状审美模型缺乏实证数据的问题。通过收集22,301组人类偏好对比数据，结合Bradley-Terry模型和随机森林分析，识别对称性、曲率等可解释几何特征，揭示跨物体类别（如家具）的通用美学原则与领域差异，为设计提供数据驱动的透明化指导，并公开数据集支持研究复现。**

- **链接: [http://arxiv.org/pdf/2505.12373v1](http://arxiv.org/pdf/2505.12373v1)**

> **作者:** Kapil Dev
>
> **备注:** 11 pages, 8 figures, submitted to IEEE Transactions on Visualization and Computer Graphics (TVCG)
>
> **摘要:** Human aesthetic preferences for 3D shapes are central to industrial design, virtual reality, and consumer product development. However, most computational models of 3D aesthetics lack empirical grounding in large-scale human judgments, limiting their practical relevance. We present a large-scale study of human preferences. We collected 22,301 pairwise comparisons across five object categories (chairs, tables, mugs, lamps, and dining chairs) via Amazon Mechanical Turk. Building on a previously published dataset~\cite{dev2020learning}, we introduce new non-linear modeling and cross-category analysis to uncover the geometric drivers of aesthetic preference. We apply the Bradley-Terry model to infer latent aesthetic scores and use Random Forests with SHAP analysis to identify and interpret the most influential geometric features (e.g., symmetry, curvature, compactness). Our cross-category analysis reveals both universal principles and domain-specific trends in aesthetic preferences. We focus on human interpretable geometric features to ensure model transparency and actionable design insights, rather than relying on black-box deep learning approaches. Our findings bridge computational aesthetics and cognitive science, providing practical guidance for designers and a publicly available dataset to support reproducibility. This work advances the understanding of 3D shape aesthetics through a human-centric, data-driven framework.
>
---
## 更新

#### [replaced 001] Going Beyond Feature Similarity: Effective Dataset Distillation based on Class-Aware Conditional Mutual Information
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.09945v4](http://arxiv.org/pdf/2412.09945v4)**

> **作者:** Xinhao Zhong; Bin Chen; Hao Fang; Xulin Gu; Shu-Tao Xia; En-Hui Yang
>
> **备注:** Accepted to ICLR 2025
>
> **摘要:** Dataset distillation (DD) aims to minimize the time and memory consumption needed for training deep neural networks on large datasets, by creating a smaller synthetic dataset that has similar performance to that of the full real dataset. However, current dataset distillation methods often result in synthetic datasets that are excessively difficult for networks to learn from, due to the compression of a substantial amount of information from the original data through metrics measuring feature similarity, e,g., distribution matching (DM). In this work, we introduce conditional mutual information (CMI) to assess the class-aware complexity of a dataset and propose a novel method by minimizing CMI. Specifically, we minimize the distillation loss while constraining the class-aware complexity of the synthetic dataset by minimizing its empirical CMI from the feature space of pre-trained networks, simultaneously. Conducting on a thorough set of experiments, we show that our method can serve as a general regularization method to existing DD methods and improve the performance and training efficiency.
>
---
#### [replaced 002] GSplatLoc: Ultra-Precise Camera Localization via 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.20056v2](http://arxiv.org/pdf/2412.20056v2)**

> **作者:** Atticus J. Zeller; Haijuan Wu
>
> **备注:** 11 pages, 2 figures. Code available at https://github.com/AtticusZeller/GsplatLoc
>
> **摘要:** We present GSplatLoc, a camera localization method that leverages the differentiable rendering capabilities of 3D Gaussian splatting for ultra-precise pose estimation. By formulating pose estimation as a gradient-based optimization problem that minimizes discrepancies between rendered depth maps from a pre-existing 3D Gaussian scene and observed depth images, GSplatLoc achieves translational errors within 0.01 cm and near-zero rotational errors on the Replica dataset - significantly outperforming existing methods. Evaluations on the Replica and TUM RGB-D datasets demonstrate the method's robustness in challenging indoor environments with complex camera motions. GSplatLoc sets a new benchmark for localization in dense mapping, with important implications for applications requiring accurate real-time localization, such as robotics and augmented reality.
>
---
#### [replaced 003] Rethinking Attention: Polynomial Alternatives to Softmax in Transformers
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2410.18613v2](http://arxiv.org/pdf/2410.18613v2)**

> **作者:** Hemanth Saratchandran; Jianqiao Zheng; Yiping Ji; Wenbo Zhang; Simon Lucey
>
> **摘要:** This paper questions whether the strong performance of softmax attention in transformers stems from producing a probability distribution over inputs. Instead, we argue that softmax's effectiveness lies in its implicit regularization of the Frobenius norm of the attention matrix, which stabilizes training. Motivated by this, we explore alternative activations, specifically polynomials, that achieve a similar regularization effect. Our theoretical analysis shows that certain polynomials can serve as effective substitutes for softmax, achieving strong performance across transformer applications despite violating softmax's typical properties of positivity, normalization, and sparsity. Extensive experiments support these findings, offering a new perspective on attention mechanisms.
>
---
#### [replaced 004] AdaptCLIP: Adapting CLIP for Universal Visual Anomaly Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.09926v2](http://arxiv.org/pdf/2505.09926v2)**

> **作者:** Bin-Bin Gao; Yue Zhou; Jiangtao Yan; Yuezhi Cai; Weixi Zhang; Meng Wang; Jun Liu; Yong Liu; Lei Wang; Chengjie Wang
>
> **备注:** 27 pages, 15 figures, 22 tables
>
> **摘要:** Universal visual anomaly detection aims to identify anomalies from novel or unseen vision domains without additional fine-tuning, which is critical in open scenarios. Recent studies have demonstrated that pre-trained vision-language models like CLIP exhibit strong generalization with just zero or a few normal images. However, existing methods struggle with designing prompt templates, complex token interactions, or requiring additional fine-tuning, resulting in limited flexibility. In this work, we present a simple yet effective method called AdaptCLIP based on two key insights. First, adaptive visual and textual representations should be learned alternately rather than jointly. Second, comparative learning between query and normal image prompt should incorporate both contextual and aligned residual features, rather than relying solely on residual features. AdaptCLIP treats CLIP models as a foundational service, adding only three simple adapters, visual adapter, textual adapter, and prompt-query adapter, at its input or output ends. AdaptCLIP supports zero-/few-shot generalization across domains and possesses a training-free manner on target domains once trained on a base dataset. AdaptCLIP achieves state-of-the-art performance on 12 anomaly detection benchmarks from industrial and medical domains, significantly outperforming existing competitive methods. We will make the code and model of AdaptCLIP available at https://github.com/gaobb/AdaptCLIP.
>
---
#### [replaced 005] Pseudo-Labeling Based Practical Semi-Supervised Meta-Training for Few-Shot Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2207.06817v4](http://arxiv.org/pdf/2207.06817v4)**

> **作者:** Xingping Dong; Tianran Ouyang; Shengcai Liao; Bo Du; Ling Shao
>
> **备注:** This paper has been accepted by IEEE Transactions on Image Processing
>
> **摘要:** Most existing few-shot learning (FSL) methods require a large amount of labeled data in meta-training, which is a major limit. To reduce the requirement of labels, a semi-supervised meta-training (SSMT) setting has been proposed for FSL, which includes only a few labeled samples and numbers of unlabeled samples in base classes. However, existing methods under this setting require class-aware sample selection from the unlabeled set, which violates the assumption of unlabeled set. In this paper, we propose a practical semi-supervised meta-training setting with truly unlabeled data to facilitate the applications of FSL in realistic scenarios. To better utilize both the labeled and truly unlabeled data, we propose a simple and effective meta-training framework, called pseudo-labeling based meta-learning (PLML). Firstly, we train a classifier via common semi-supervised learning (SSL) and use it to obtain the pseudo-labels of unlabeled data. Then we build few-shot tasks from labeled and pseudo-labeled data and design a novel finetuning method with feature smoothing and noise suppression to better learn the FSL model from noise labels. Surprisingly, through extensive experiments across two FSL datasets, we find that this simple meta-training framework effectively prevents the performance degradation of various FSL models under limited labeled data, and also significantly outperforms the state-of-the-art SSMT models. Besides, benefiting from meta-training, our method also improves two representative SSL algorithms as well.
>
---
#### [replaced 006] CDMamba: Incorporating Local Clues into Mamba for Remote Sensing Image Binary Change Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.04207v2](http://arxiv.org/pdf/2406.04207v2)**

> **作者:** Haotian Zhang; Keyan Chen; Chenyang Liu; Hao Chen; Zhengxia Zou; Zhenwei Shi
>
> **摘要:** Recently, the Mamba architecture based on state space models has demonstrated remarkable performance in a series of natural language processing tasks and has been rapidly applied to remote sensing change detection (CD) tasks. However, most methods enhance the global receptive field by directly modifying the scanning mode of Mamba, neglecting the crucial role that local information plays in dense prediction tasks (e.g., binary CD). In this article, we propose a model called CDMamba, which effectively combines global and local features for handling binary CD tasks. Specifically, the Scaled Residual ConvMamba (SRCM) block is proposed to utilize the ability of Mamba to extract global features and convolution to enhance the local details to alleviate the issue that current Mamba-based methods lack detailed clues and are difficult to achieve fine detection in dense prediction tasks. Furthermore, considering the characteristics of bi-temporal feature interaction required for CD, the Adaptive Global Local Guided Fusion (AGLGF) block is proposed to dynamically facilitate the bi-temporal interaction guided by other temporal global/local features. Our intuition is that more discriminative change features can be acquired with the guidance of other temporal features. Extensive experiments on five datasets demonstrate that our proposed CDMamba is comparable to the current methods (such as the F1/IoU scores are improved by 2.10%/3.00% and 2.44%/2.91% on LEVIR+CD and CLCD, respectively). Our code is open-sourced at https://github.com/zmoka-zht/CDMamba.
>
---
#### [replaced 007] JL1-CD: A New Benchmark for Remote Sensing Change Detection and a Robust Multi-Teacher Knowledge Distillation Framework
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.13407v3](http://arxiv.org/pdf/2502.13407v3)**

> **作者:** Ziyuan Liu; Ruifei Zhu; Long Gao; Yuanxiu Zhou; Jingyu Ma; Yuantao Gu
>
> **备注:** 23 pages, 11 figures
>
> **摘要:** Change detection (CD) in remote sensing images plays a vital role in Earth observation. However, the scarcity of high-resolution, comprehensive open-source datasets and the difficulty in achieving robust performance across varying change types remain major challenges. To address these issues, we introduce JL1-CD, a large-scale, sub-meter CD dataset consisting of 5,000 image pairs. We further propose a novel Origin-Partition (O-P) strategy and integrate it into a Multi-Teacher Knowledge Distillation (MTKD) framework to enhance CD performance. The O-P strategy partitions the training set by Change Area Ratio (CAR) and trains specialized teacher models on each subset. The MTKD framework then distills complementary knowledge from these teachers into a single student model, enabling improved detection results across diverse CAR scenarios without additional inference cost. Our MTKD approach demonstrated strong performance in the 2024 "Jilin-1'' Cup challenge, ranking first in the preliminary and second in the final rounds. Extensive experiments on the JL1-CD and SYSU-CD datasets show that the MTKD framework consistently improves the performance of CD models with various network architectures and parameter sizes, establishing new state-of-the-art results. Code and dataset are available at https://anonymous.4open.science/r/MTKD-A-84B8.
>
---
#### [replaced 008] RS-Agent: Automating Remote Sensing Tasks through Intelligent Agent
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.07089v2](http://arxiv.org/pdf/2406.07089v2)**

> **作者:** Wenjia Xu; Zijian Yu; Boyang Mu; Zhiwei Wei; Yuanben Zhang; Guangzuo Li; Mugen Peng
>
> **摘要:** The unprecedented advancements in Multimodal Large Language Models (MLLMs) have demonstrated strong potential in interacting with humans through both language and visual inputs to perform downstream tasks such as visual question answering and scene understanding. However, these models are constrained to basic instruction-following or descriptive tasks, facing challenges in complex real-world remote sensing applications that require specialized tools and knowledge. To address these limitations, we propose RS-Agent, an AI agent designed to interact with human users and autonomously leverage specialized models to address the demands of real-world remote sensing applications. RS-Agent integrates four key components: a Central Controller based on large language models, a dynamic toolkit for tool execution, a Solution Space for task-specific expert guidance, and a Knowledge Space for domain-level reasoning, enabling it to interpret user queries and orchestrate tools for accurate remote sensing task. We introduce two novel mechanisms: Task-Aware Retrieval, which improves tool selection accuracy through expert-guided planning, and DualRAG, a retrieval-augmented generation method that enhances knowledge relevance through weighted, dual-path retrieval. RS-Agent supports flexible integration of new tools and is compatible with both open-source and proprietary LLMs. Extensive experiments across 9 datasets and 18 remote sensing tasks demonstrate that RS-Agent significantly outperforms state-of-the-art MLLMs, achieving over 95% task planning accuracy and delivering superior performance in tasks such as scene classification, object counting, and remote sensing visual question answering. Our work presents RS-Agent as a robust and extensible framework for advancing intelligent automation in remote sensing analysis.
>
---
#### [replaced 009] Dynamic Attention Analysis for Backdoor Detection in Text-to-Image Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.20518v2](http://arxiv.org/pdf/2504.20518v2)**

> **作者:** Zhongqi Wang; Jie Zhang; Shiguang Shan; Xilin Chen
>
> **摘要:** Recent studies have revealed that text-to-image diffusion models are vulnerable to backdoor attacks, where attackers implant stealthy textual triggers to manipulate model outputs. Previous backdoor detection methods primarily focus on the static features of backdoor samples. However, a vital property of diffusion models is their inherent dynamism. This study introduces a novel backdoor detection perspective named Dynamic Attention Analysis (DAA), showing that these dynamic characteristics serve as better indicators for backdoor detection. Specifically, by examining the dynamic evolution of cross-attention maps, we observe that backdoor samples exhibit distinct feature evolution patterns at the $<$EOS$>$ token compared to benign samples. To quantify these dynamic anomalies, we first introduce DAA-I, which treats the tokens' attention maps as spatially independent and measures dynamic feature using the Frobenius norm. Furthermore, to better capture the interactions between attention maps and refine the feature, we propose a dynamical system-based approach, referred to as DAA-S. This model formulates the spatial correlations among attention maps using a graph-based state equation and we theoretically analyze the global asymptotic stability of this method. Extensive experiments across five representative backdoor attack scenarios demonstrate that our approach significantly surpasses existing detection methods, achieving an average F1 Score of 79.49% and an AUC of 87.67%. The code is available at https://github.com/Robin-WZQ/DAA.
>
---
#### [replaced 010] Vision-centric Token Compression in Large Language Model
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.00791v3](http://arxiv.org/pdf/2502.00791v3)**

> **作者:** Ling Xing; Alex Jinpeng Wang; Rui Yan; Xiangbo Shu; Jinhui Tang
>
> **摘要:** Real-world applications are stretching context windows to hundreds of thousand of tokens while Large Language Models (LLMs) swell from billions to trillions of parameters. This dual expansion send compute and memory costs skyrocketing, making token compression indispensable. We introduce Vision Centric Token Compression (Vist), a slow-fast compression framework that mirrors human reading: the fast path renders distant tokens into images, letting a frozen, lightweight vision encoder skim the low-salience context; the slow path feeds the proximal window into the LLM for fine-grained reasoning. A Probability-Informed Visual Enhancement (PVE) objective masks high-frequency tokens during training, steering the Resampler to concentrate on semantically rich regions-just as skilled reader gloss over function words. On eleven in-context learning benchmarks, Vist achieves the same accuracy with 2.3 times fewer tokens, cutting FLOPs by 16% and memory by 50%. This method delivers remarkable results, outperforming the strongest text encoder-based compression method CEPE by 7.6% on average over benchmarks like TriviaQA, NQ, PopQA, NLUI, and CLIN, setting a new standard for token efficiency in LLMs. The source code will be released.
>
---
#### [replaced 011] 3DGen-Bench: Comprehensive Benchmark Suite for 3D Generative Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21745v2](http://arxiv.org/pdf/2503.21745v2)**

> **作者:** Yuhan Zhang; Mengchen Zhang; Tong Wu; Tengfei Wang; Gordon Wetzstein; Dahua Lin; Ziwei Liu
>
> **摘要:** 3D generation is experiencing rapid advancements, while the development of 3D evaluation has not kept pace. How to keep automatic evaluation equitably aligned with human perception has become a well-recognized challenge. Recent advances in the field of language and image generation have explored human preferences and showcased respectable fitting ability. However, the 3D domain still lacks such a comprehensive preference dataset over generative models. To mitigate this absence, we develop 3DGen-Arena, an integrated platform in a battle manner. Then, we carefully design diverse text and image prompts and leverage the arena platform to gather human preferences from both public users and expert annotators, resulting in a large-scale multi-dimension human preference dataset 3DGen-Bench. Using this dataset, we further train a CLIP-based scoring model, 3DGen-Score, and a MLLM-based automatic evaluator, 3DGen-Eval. These two models innovatively unify the quality evaluation of text-to-3D and image-to-3D generation, and jointly form our automated evaluation system with their respective strengths. Extensive experiments demonstrate the efficacy of our scoring model in predicting human preferences, exhibiting a superior correlation with human ranks compared to existing metrics. We believe that our 3DGen-Bench dataset and automated evaluation system will foster a more equitable evaluation in the field of 3D generation, further promoting the development of 3D generative models and their downstream applications.
>
---
#### [replaced 012] BrainPrompt: Multi-Level Brain Prompt Enhancement for Neurological Condition Identification
- **分类: q-bio.NC; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.16096v2](http://arxiv.org/pdf/2504.16096v2)**

> **作者:** Jiaxing Xu; Kai He; Yue Tang; Wei Li; Mengcheng Lan; Xia Dong; Yiping Ke; Mengling Feng
>
> **备注:** Early accepted by MICCAI 2025
>
> **摘要:** Neurological conditions, such as Alzheimer's Disease, are challenging to diagnose, particularly in the early stages where symptoms closely resemble healthy controls. Existing brain network analysis methods primarily focus on graph-based models that rely solely on imaging data, which may overlook important non-imaging factors and limit the model's predictive power and interpretability. In this paper, we present BrainPrompt, an innovative framework that enhances Graph Neural Networks (GNNs) by integrating Large Language Models (LLMs) with knowledge-driven prompts, enabling more effective capture of complex, non-imaging information and external knowledge for neurological disease identification. BrainPrompt integrates three types of knowledge-driven prompts: (1) ROI-level prompts to encode the identity and function of each brain region, (2) subject-level prompts that incorporate demographic information, and (3) disease-level prompts to capture the temporal progression of disease. By leveraging these multi-level prompts, BrainPrompt effectively harnesses knowledge-enhanced multi-modal information from LLMs, enhancing the model's capability to predict neurological disease stages and meanwhile offers more interpretable results. We evaluate BrainPrompt on two resting-state functional Magnetic Resonance Imaging (fMRI) datasets from neurological disorders, showing its superiority over state-of-the-art methods. Additionally, a biomarker study demonstrates the framework's ability to extract valuable and interpretable information aligned with domain knowledge in neuroscience. The code is available at https://github.com/AngusMonroe/BrainPrompt
>
---
#### [replaced 013] Irregular Tensor Low-Rank Representation for Hyperspectral Image Representation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.18388v4](http://arxiv.org/pdf/2410.18388v4)**

> **作者:** Bo Han; Yuheng Jia; Hui Liu; Junhui Hou
>
> **备注:** Accepted by TIP
>
> **摘要:** Spectral variations pose a common challenge in analyzing hyperspectral images (HSI). To address this, low-rank tensor representation has emerged as a robust strategy, leveraging inherent correlations within HSI data. However, the spatial distribution of ground objects in HSIs is inherently irregular, existing naturally in tensor format, with numerous class-specific regions manifesting as irregular tensors. Current low-rank representation techniques are designed for regular tensor structures and overlook this fundamental irregularity in real-world HSIs, leading to performance limitations. To tackle this issue, we propose a novel model for irregular tensor low-rank representation tailored to efficiently model irregular 3D cubes. By incorporating a non-convex nuclear norm to promote low-rankness and integrating a global negative low-rank term to enhance the discriminative ability, our proposed model is formulated as a constrained optimization problem and solved using an alternating augmented Lagrangian method. Experimental validation conducted on four public datasets demonstrates the superior performance of our method compared to existing state-of-the-art approaches. The code is publicly available at https://github.com/hb-studying/ITLRR.
>
---
#### [replaced 014] Multi-modal MRI Translation via Evidential Regression and Distribution Calibration
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.07372v2](http://arxiv.org/pdf/2407.07372v2)**

> **作者:** Jiyao Liu; Shangqi Gao; Yuxin Li; Lihao Liu; Xin Gao; Zhaohu Xing; Junzhi Ning; Yanzhou Su; Xiao-Yong Zhang; Junjun He; Ningsheng Xu; Xiahai Zhuang
>
> **备注:** Early accepted by MICCAI 2025
>
> **摘要:** Multi-modal Magnetic Resonance Imaging (MRI) translation leverages information from source MRI sequences to generate target modalities, enabling comprehensive diagnosis while overcoming the limitations of acquiring all sequences. While existing deep-learning-based multi-modal MRI translation methods have shown promising potential, they still face two key challenges: 1) lack of reliable uncertainty quantification for synthesized images, and 2) limited robustness when deployed across different medical centers. To address these challenges, we propose a novel framework that reformulates multi-modal MRI translation as a multi-modal evidential regression problem with distribution calibration. Our approach incorporates two key components: 1) an evidential regression module that estimates uncertainties from different source modalities and an explicit distribution mixture strategy for transparent multi-modal fusion, and 2) a distribution calibration mechanism that adapts to source-target mapping shifts to ensure consistent performance across different medical centers. Extensive experiments on three datasets from the BraTS2023 challenge demonstrate that our framework achieves superior performance and robustness across domains.
>
---
#### [replaced 015] Beyond Conventional Transformers: The Medical X-ray Attention (MXA) Block for Improved Multi-Label Diagnosis Using Knowledge Distillation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.02277v2](http://arxiv.org/pdf/2504.02277v2)**

> **作者:** Amit Rand; Hadi Ibrahim
>
> **备注:** 19 pages, 9 figures, 6 tables. For supplementary material and code, see https://github.com/Hadi-M-Ibrahim/Beyond-Conventional-Transformers/
>
> **摘要:** Medical imaging, particularly X-ray analysis, often involves detecting multiple conditions simultaneously within a single scan, making multi-label classification crucial for real-world clinical applications. We present the Medical X-ray Attention (MXA) block, a novel attention mechanism tailored specifically to address the unique challenges of X-ray abnormality detection. The MXA block enhances traditional Multi-Head Self Attention (MHSA) by integrating a specialized module that efficiently captures both detailed local information and broader global context. To the best of our knowledge, this is the first work to propose a task-specific attention mechanism for diagnosing chest X-rays, as well as to attempt multi-label classification using an Efficient Vision Transformer (EfficientViT). By embedding the MXA block within the EfficientViT architecture and employing knowledge distillation, our proposed model significantly improves performance on the CheXpert dataset, a widely used benchmark for multi-label chest X-ray abnormality detection. Our approach achieves an area under the curve (AUC) of 0.85, an absolute improvement of 0.19 compared to our baseline model's AUC of 0.66, corresponding to a substantial approximate 233% relative improvement over random guessing (AUC = 0.5).
>
---
#### [replaced 016] Developing a Hybrid Convolutional Neural Network for Automatic Aphid Counting in Sugar Beet Fields
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2308.05257v2](http://arxiv.org/pdf/2308.05257v2)**

> **作者:** Xumin Gao; Wenxin Xue; Callum Lennox; Mark Stevens; Junfeng Gao
>
> **备注:** Published in Computers and Electronics in Agriculture, Volume 220, May 2024, Article 108910
>
> **摘要:** Aphids can cause direct damage and indirect virus transmission to crops. Timely monitoring and control of their populations are thus critical. However, the manual counting of aphids, which is the most common practice, is labor-intensive and time-consuming. Additionally, two of the biggest challenges in aphid counting are that aphids are small objects and their density distributions are varied in different areas of the field. To address these challenges, we proposed a hybrid automatic aphid counting network architecture which integrates the detection network and the density map estimation network. When the distribution density of aphids is low, it utilizes an improved Yolov5 to count aphids. Conversely, when the distribution density of aphids is high, it switches to CSRNet to count aphids. To the best of our knowledge, this is the first framework integrating the detection network and the density map estimation network for counting tasks. Through comparison experiments of counting aphids, it verified that our proposed approach outperforms all other methods in counting aphids. It achieved the lowest MAE and RMSE values for both the standard and high-density aphid datasets: 2.93 and 4.01 (standard), and 34.19 and 38.66 (high-density), respectively. Moreover, the AP of the improved Yolov5 is 5% higher than that of the original Yolov5. Especially for extremely small aphids and densely distributed aphids, the detection performance of the improved Yolov5 is significantly better than the original Yolov5. This work provides an effective early warning caused by aphids in sugar beet fields, offering protection for sugar beet growth and ensuring sugar beet yield. The datasets and project code are released at: https://github.com/JunfengGaolab/Counting-Aphids.
>
---
#### [replaced 017] FIOVA: A Multi-Annotator Benchmark for Human-Aligned Video Captioning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.15270v2](http://arxiv.org/pdf/2410.15270v2)**

> **作者:** Shiyu Hu; Xuchen Li; Xuzhao Li; Jing Zhang; Yipei Wang; Xin Zhao; Kang Hao Cheong
>
> **摘要:** Despite rapid progress in large vision-language models (LVLMs), existing video caption benchmarks remain limited in evaluating their alignment with human understanding. Most rely on a single annotation per video and lexical similarity-based metrics, failing to capture the variability in human perception and the cognitive importance of events. These limitations hinder accurate diagnosis of model capabilities in producing coherent, complete, and human-aligned descriptions. To address this, we introduce FIOVA (Five-In-One Video Annotations), a human-centric benchmark tailored for evaluation. It comprises 3,002 real-world videos (about 33.6s each), each annotated independently by five annotators. This design enables modeling of semantic diversity and inter-subjective agreement, offering a richer foundation for measuring human-machine alignment. We further propose FIOVA-DQ, an event-level evaluation metric that incorporates cognitive weights derived from annotator consensus, providing fine-grained assessment of event relevance and semantic coverage. Leveraging FIOVA, we conduct a comprehensive evaluation of nine representative LVLMs and introduce a complexity-aware analysis framework based on inter-annotator variation (CV). This reveals consistency gaps across difficulty levels and identifies structural issues such as event under-description and template convergence. Our results highlight FIOVA's diagnostic value for understanding LVLM behavior under varying complexity, setting a new standard for cognitively aligned evaluation in long-video captioning. The benchmark, annotations, metric, and model outputs are publicly released to support future evaluation-driven research in video understanding. More detailed information can be found at https://huuuuusy.github.io/fiova/.
>
---
#### [replaced 018] Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning
- **分类: cs.AI; cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.15558v3](http://arxiv.org/pdf/2503.15558v3)**

> **作者:** NVIDIA; :; Alisson Azzolini; Junjie Bai; Hannah Brandon; Jiaxin Cao; Prithvijit Chattopadhyay; Huayu Chen; Jinju Chu; Yin Cui; Jenna Diamond; Yifan Ding; Liang Feng; Francesco Ferroni; Rama Govindaraju; Jinwei Gu; Siddharth Gururani; Imad El Hanafi; Zekun Hao; Jacob Huffman; Jingyi Jin; Brendan Johnson; Rizwan Khan; George Kurian; Elena Lantz; Nayeon Lee; Zhaoshuo Li; Xuan Li; Maosheng Liao; Tsung-Yi Lin; Yen-Chen Lin; Ming-Yu Liu; Xiangyu Lu; Alice Luo; Andrew Mathau; Yun Ni; Lindsey Pavao; Wei Ping; David W. Romero; Misha Smelyanskiy; Shuran Song; Lyne Tchapmi; Andrew Z. Wang; Boxin Wang; Haoxiang Wang; Fangyin Wei; Jiashu Xu; Yao Xu; Dinghao Yang; Xiaodong Yang; Zhuolin Yang; Jingxu Zhang; Xiaohui Zeng; Zhe Zhang
>
> **摘要:** Physical AI systems need to perceive, understand, and perform complex actions in the physical world. In this paper, we present the Cosmos-Reason1 models that can understand the physical world and generate appropriate embodied decisions (e.g., next step action) in natural language through long chain-of-thought reasoning processes. We begin by defining key capabilities for Physical AI reasoning, with a focus on physical common sense and embodied reasoning. To represent physical common sense, we use a hierarchical ontology that captures fundamental knowledge about space, time, and physics. For embodied reasoning, we rely on a two-dimensional ontology that generalizes across different physical embodiments. Building on these capabilities, we develop two multimodal large language models, Cosmos-Reason1-7B and Cosmos-Reason1-56B. We curate data and train our models in two stages: Physical AI supervised fine-tuning (SFT) and Physical AI reinforcement learning (RL). To evaluate our models, we build comprehensive benchmarks for physical common sense and embodied reasoning according to our ontologies. Evaluation results show that Physical AI SFT and RL bring significant improvements. To facilitate the development of Physical AI, we make our code and pre-trained models available under the NVIDIA Open Model License at https://github.com/nvidia-cosmos/cosmos-reason1.
>
---
#### [replaced 019] CAMOT: Camera Angle-aware Multi-Object Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.17533v2](http://arxiv.org/pdf/2409.17533v2)**

> **作者:** Felix Limanta; Kuniaki Uto; Koichi Shinoda
>
> **备注:** https://gitlab.com/felixlimanta/camot
>
> **摘要:** This paper proposes CAMOT, a simple camera angle estimator for multi-object tracking to tackle two problems: 1) occlusion and 2) inaccurate distance estimation in the depth direction. Under the assumption that multiple objects are located on a flat plane in each video frame, CAMOT estimates the camera angle using object detection. In addition, it gives the depth of each object, enabling pseudo-3D MOT. We evaluated its performance by adding it to various 2D MOT methods on the MOT17 and MOT20 datasets and confirmed its effectiveness. Applying CAMOT to ByteTrack, we obtained 63.8% HOTA, 80.6% MOTA, and 78.5% IDF1 in MOT17, which are state-of-the-art results. Its computational cost is significantly lower than the existing deep-learning-based depth estimators for tracking.
>
---
#### [replaced 020] STORYANCHORS: Generating Consistent Multi-Scene Story Frames for Long-Form Narratives
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.08350v2](http://arxiv.org/pdf/2505.08350v2)**

> **作者:** Bo Wang; Haoyang Huang; Zhiying Lu; Fengyuan Liu; Guoqing Ma; Jianlong Yuan; Yuan Zhang; Nan Duan; Daxin Jiang
>
> **摘要:** This paper introduces StoryAnchors, a unified framework for generating high-quality, multi-scene story frames with strong temporal consistency. The framework employs a bidirectional story generator that integrates both past and future contexts to ensure temporal consistency, character continuity, and smooth scene transitions throughout the narrative. Specific conditions are introduced to distinguish story frame generation from standard video synthesis, facilitating greater scene diversity and enhancing narrative richness. To further improve generation quality, StoryAnchors integrates Multi-Event Story Frame Labeling and Progressive Story Frame Training, enabling the model to capture both overarching narrative flow and event-level dynamics. This approach supports the creation of editable and expandable story frames, allowing for manual modifications and the generation of longer, more complex sequences. Extensive experiments show that StoryAnchors outperforms existing open-source models in key areas such as consistency, narrative coherence, and scene diversity. Its performance in narrative consistency and story richness is also on par with GPT-4o. Ultimately, StoryAnchors pushes the boundaries of story-driven frame generation, offering a scalable, flexible, and highly editable foundation for future research.
>
---
#### [replaced 021] Scene-Text Grounding for Text-Based Video Question Answering
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2409.14319v3](http://arxiv.org/pdf/2409.14319v3)**

> **作者:** Sheng Zhou; Junbin Xiao; Xun Yang; Peipei Song; Dan Guo; Angela Yao; Meng Wang; Tat-Seng Chua
>
> **备注:** Accepted by IEEE TMM
>
> **摘要:** Existing efforts in text-based video question answering (TextVideoQA) are criticized for their opaque decisionmaking and heavy reliance on scene-text recognition. In this paper, we propose to study Grounded TextVideoQA by forcing models to answer questions and spatio-temporally localize the relevant scene-text regions, thus decoupling QA from scenetext recognition and promoting research towards interpretable QA. The task has three-fold significance. First, it encourages scene-text evidence versus other short-cuts for answer predictions. Second, it directly accepts scene-text regions as visual answers, thus circumventing the problem of ineffective answer evaluation by stringent string matching. Third, it isolates the challenges inherited in VideoQA and scene-text recognition. This enables the diagnosis of the root causes for failure predictions, e.g., wrong QA or wrong scene-text recognition? To achieve Grounded TextVideoQA, we propose the T2S-QA model that highlights a disentangled temporal-to-spatial contrastive learning strategy for weakly-supervised scene-text grounding and grounded TextVideoQA. To facilitate evaluation, we construct a new dataset ViTXT-GQA which features 52K scene-text bounding boxes within 2.2K temporal segments related to 2K questions and 729 videos. With ViTXT-GQA, we perform extensive experiments and demonstrate the severe limitations of existing techniques in Grounded TextVideoQA. While T2S-QA achieves superior results, the large performance gap with human leaves ample space for improvement. Our further analysis of oracle scene-text inputs posits that the major challenge is scene-text recognition. To advance the research of Grounded TextVideoQA, our dataset and code are at https://github.com/zhousheng97/ViTXT-GQA.git
>
---
#### [replaced 022] Quantifying Context Bias in Domain Adaptation for Object Detection
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2409.14679v2](http://arxiv.org/pdf/2409.14679v2)**

> **作者:** Hojun Son; Asma Almutairi; Arpan Kusari
>
> **备注:** Under review
>
> **摘要:** Domain adaptation for object detection (DAOD) seeks to transfer a trained model from a source to a target domain. Various DAOD methods exist, some of which aim to minimize context bias between foreground-background associations in various domains. However, no prior work has studied context bias in DAOD by analyzing changes in background features during adaptation and how context bias is represented in different domains. Our research experiment highlights the potential usability of context bias in DAOD. We address the problem by varying activation values over different layers of two different trained models, Detectron2 and YOLOv11, and by masking the background, both of which impact the number and quality of detections. We use two synthetic datasets, CARLA and Virtual KITTI, and two different versions of real open-source data, Cityscapes and KITTI semantic, as separate domains to represent and quantify context bias. We utilize different metrics such as Maximum Mean Discrepancy (MMD) and Maximum Variance Discrepancy (MVD) to find the layer-specific conditional probability estimates of foreground given manipulated background regions for separate domains. We further analyze foreground-background associations across various dataset combinations. We find that state-of-the-art domain adaptation methods exhibit some form of context bias and apply a potentially simple way to alleviate the context bias achieving improved accuracy (from 51.189 to 53.646 mAP on Cityscapes foggy validation with 63.207 mAP and 64.233 mAP on Cityscapes validation respectively). We demonstrate through detailed analysis that understanding of the context bias can affect DAOD approach and focusing solely on aligning foreground features is insufficient for effective DAOD.
>
---
#### [replaced 023] $\infty$-Video: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.19098v2](http://arxiv.org/pdf/2501.19098v2)**

> **作者:** Saul Santos; António Farinhas; Daniel C. McNamee; André F. T. Martins
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** Current video-language models struggle with long-video understanding due to limited context lengths and reliance on sparse frame subsampling, often leading to information loss. This paper introduces $\infty$-Video, which can process arbitrarily long videos through a continuous-time long-term memory (LTM) consolidation mechanism. Our framework augments video Q-formers by allowing them to process unbounded video contexts efficiently and without requiring additional training. Through continuous attention, our approach dynamically allocates higher granularity to the most relevant video segments, forming "sticky" memories that evolve over time. Experiments with Video-LLaMA and VideoChat2 demonstrate improved performance in video question-answering tasks, showcasing the potential of continuous-time LTM mechanisms to enable scalable and training-free comprehension of long videos.
>
---
#### [replaced 024] Digital Twin Catalog: A Large-Scale Photorealistic 3D Object Digital Twin Dataset
- **分类: cs.GR; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.08541v2](http://arxiv.org/pdf/2504.08541v2)**

> **作者:** Zhao Dong; Ka Chen; Zhaoyang Lv; Hong-Xing Yu; Yunzhi Zhang; Cheng Zhang; Yufeng Zhu; Stephen Tian; Zhengqin Li; Geordie Moffatt; Sean Christofferson; James Fort; Xiaqing Pan; Mingfei Yan; Jiajun Wu; Carl Yuheng Ren; Richard Newcombe
>
> **备注:** accepted to CVPR 2025 (Highlight). Dataset page: https://www.projectaria.com/datasets/dtc/
>
> **摘要:** We introduce the Digital Twin Catalog (DTC), a new large-scale photorealistic 3D object digital twin dataset. A digital twin of a 3D object is a highly detailed, virtually indistinguishable representation of a physical object, accurately capturing its shape, appearance, physical properties, and other attributes. Recent advances in neural-based 3D reconstruction and inverse rendering have significantly improved the quality of 3D object reconstruction. Despite these advancements, there remains a lack of a large-scale, digital twin-quality real-world dataset and benchmark that can quantitatively assess and compare the performance of different reconstruction methods, as well as improve reconstruction quality through training or fine-tuning. Moreover, to democratize 3D digital twin creation, it is essential to integrate creation techniques with next-generation egocentric computing platforms, such as AR glasses. Currently, there is no dataset available to evaluate 3D object reconstruction using egocentric captured images. To address these gaps, the DTC dataset features 2,000 scanned digital twin-quality 3D objects, along with image sequences captured under different lighting conditions using DSLR cameras and egocentric AR glasses. This dataset establishes the first comprehensive real-world evaluation benchmark for 3D digital twin creation tasks, offering a robust foundation for comparing and improving existing reconstruction methods. The DTC dataset is already released at https://www.projectaria.com/datasets/dtc/ and we will also make the baseline evaluations open-source.
>
---
#### [replaced 025] COMAE: COMprehensive Attribute Exploration for Zero-shot Hashing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2402.16424v5](http://arxiv.org/pdf/2402.16424v5)**

> **作者:** Yuqi Li; Qingqing Long; Yihang Zhou; Ran Zhang; Zhiyuan Ning; Zhihong Zhu; Yuanchun Zhou; Xuezhi Wang; Meng Xiao
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Zero-shot hashing (ZSH) has shown excellent success owing to its efficiency and generalization in large-scale retrieval scenarios. While considerable success has been achieved, there still exist urgent limitations. Existing works ignore the locality relationships of representations and attributes, which have effective transferability between seeable classes and unseeable classes. Also, the continuous-value attributes are not fully harnessed. In response, we conduct a COMprehensive Attribute Exploration for ZSH, named COMAE, which depicts the relationships from seen classes to unseen ones through three meticulously designed explorations, i.e., point-wise, pair-wise and class-wise consistency constraints. By regressing attributes from the proposed attribute prototype network, COMAE learns the local features that are relevant to the visual attributes. Then COMAE utilizes contrastive learning to comprehensively depict the context of attributes, rather than instance-independent optimization. Finally, the class-wise constraint is designed to cohesively learn the hash code, image representation, and visual attributes more effectively. Experimental results on the popular ZSH datasets demonstrate that COMAE outperforms state-of-the-art hashing techniques, especially in scenarios with a larger number of unseen label classes.
>
---
#### [replaced 026] PseudoNeg-MAE: Self-Supervised Point Cloud Learning using Conditional Pseudo-Negative Embeddings
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.15832v2](http://arxiv.org/pdf/2409.15832v2)**

> **作者:** Sutharsan Mahendren; Saimunur Rahman; Piotr Koniusz; Tharindu Fernando; Sridha Sridharan; Clinton Fookes; Peyman Moghadam
>
> **摘要:** We propose PseudoNeg-MAE, a novel self-supervised learning framework that enhances global feature representation of point cloud masked autoencoder by making them both discriminative and sensitive to transformations. Traditional contrastive learning methods focus on achieving invariance, discarding transformation-specific information. Recent approaches incorporate transformation sensitivity by explicitly modeling relationships between original and transformed inputs. However, they report an invariant-collapse phenomenon, where the predictor degenerates into identity mappings, resulting in latent representations that have limited variation across transformations. We propose a novel loss that explicitly penalizes invariant collapse, enabling the network to capture richer transformation cues while preserving discriminative representations. PseudoNeg-MAE uses a parametric network COPE, which learns the localized displacements caused by transformations within the latent space. However, jointly training COPE with the MAE leads to undesirable trivial solutions where COPE outputs collapse to an identity. To address this, we propose a loss that uses transformation-conditioned pseudo-negatives, to penalize such trivial invariant solutions. We validate PseudoNeg-MAE on shape classification and relative pose estimation tasks, where it achieves competitive performance on the ModelNet40 and ScanObjectNN datasets under challenging evaluation protocols and demonstrates superior accuracy in estimating relative poses compared to supervised methods.
>
---
#### [replaced 027] VCM: Vision Concept Modeling Based on Implicit Contrastive Learning with Vision-Language Instruction Fine-Tuning
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.19627v2](http://arxiv.org/pdf/2504.19627v2)**

> **作者:** Run Luo; Renke Shan; Longze Chen; Ziqiang Liu; Lu Wang; Min Yang; Xiaobo Xia
>
> **备注:** VCM
>
> **摘要:** Large Vision-Language Models (LVLMs) are pivotal for real-world AI tasks like embodied intelligence due to their strong vision-language reasoning abilities. However, current LVLMs process entire images at the token level, which is inefficient compared to humans who analyze information and generate content at the conceptual level, extracting relevant visual concepts with minimal effort. This inefficiency, stemming from the lack of a visual concept model, limits LVLMs' usability in real-world applications. To address this, we propose VCM, an end-to-end self-supervised visual concept modeling framework. VCM leverages implicit contrastive learning across multiple sampled instances and vision-language fine-tuning to construct a visual concept model without requiring costly concept-level annotations. Our results show that VCM significantly reduces computational costs (e.g., 85\% fewer FLOPs for LLaVA-1.5-7B) while maintaining strong performance across diverse image understanding tasks. Moreover, VCM enhances visual encoders' capabilities in classic visual concept perception tasks. Extensive quantitative and qualitative experiments validate the effectiveness and efficiency of VCM.
>
---
#### [replaced 028] Low-Light Video Enhancement via Spatial-Temporal Consistent Decomposition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.15660v2](http://arxiv.org/pdf/2405.15660v2)**

> **作者:** Xiaogang Xu; Kun Zhou; Tao Hu; Jiafei Wu; Ruixing Wang; Hao Peng; Bei Yu
>
> **备注:** IJCAI2025
>
> **摘要:** Low-Light Video Enhancement (LLVE) seeks to restore dynamic or static scenes plagued by severe invisibility and noise. In this paper, we present an innovative video decomposition strategy that incorporates view-independent and view-dependent components to enhance the performance of LLVE. We leverage dynamic cross-frame correspondences for the view-independent term (which primarily captures intrinsic appearance) and impose a scene-level continuity constraint on the view-dependent term (which mainly describes the shading condition) to achieve consistent and satisfactory decomposition results. To further ensure consistent decomposition, we introduce a dual-structure enhancement network featuring a cross-frame interaction mechanism. By supervising different frames simultaneously, this network encourages them to exhibit matching decomposition features. This mechanism can seamlessly integrate with encoder-decoder single-frame networks, incurring minimal additional parameter costs. Extensive experiments are conducted on widely recognized LLVE benchmarks, covering diverse scenarios. Our framework consistently outperforms existing methods, establishing a new SOTA performance.
>
---
#### [replaced 029] RefDrone: A Challenging Benchmark for Referring Expression Comprehension in Drone Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.00392v2](http://arxiv.org/pdf/2502.00392v2)**

> **作者:** Zhichao Sun; Yepeng Liu; Huachao Zhu; Yuliang Gu; Yuda Zou; Zelong Liu; Gui-Song Xia; Bo Du; Yongchao Xu
>
> **摘要:** Drones have become prevalent robotic platforms with diverse applications, showing significant potential in Embodied Artificial Intelligence (Embodied AI). Referring Expression Comprehension (REC) enables drones to locate objects based on natural language expressions, a crucial capability for Embodied AI. Despite advances in REC for ground-level scenes, aerial views introduce unique challenges including varying viewpoints, occlusions and scale variations. To address this gap, we introduce RefDrone, a REC benchmark for drone scenes. RefDrone reveals three key challenges in REC: 1) multi-scale and small-scale target detection; 2) multi-target and no-target samples; 3) complex environment with rich contextual expressions. To efficiently construct this dataset, we develop RDAgent (referring drone annotation framework with multi-agent system), a semi-automated annotation tool for REC tasks. RDAgent ensures high-quality contextual expressions and reduces annotation cost. Furthermore, we propose Number GroundingDINO (NGDINO), a novel method designed to handle multi-target and no-target cases. NGDINO explicitly learns and utilizes the number of objects referred to in the expression. Comprehensive experiments with state-of-the-art REC methods demonstrate that NGDINO achieves superior performance on both the proposed RefDrone and the existing gRefCOCO datasets. The dataset and code are be publicly at https://github.com/sunzc-sunny/refdrone.
>
---
#### [replaced 030] DeepFRC: An End-to-End Deep Learning Model for Functional Registration and Classification
- **分类: cs.CV; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2501.18116v2](http://arxiv.org/pdf/2501.18116v2)**

> **作者:** Siyuan Jiang; Yihan Hu; Wenjie Li; Pengcheng Zeng
>
> **备注:** 28 pages, 8 figures
>
> **摘要:** Functional data - observations in the form of curves or trajectories - arise in diverse domains such as biomedical sensing, motion capture, and handwriting recognition. A core challenge in functional data analysis (FDA) is accounting for phase variability, where misaligned temporal patterns hinder accurate inference. We introduce DeepFRC, an end-to-end deep learning framework for joint functional registration and classification. Unlike conventional approaches that decouple alignment and prediction, DeepFRC integrates class-aware elastic warping and a learnable basis representation into a unified architecture. This design enables temporal alignment and dimensionality reduction to be jointly optimized with classification, improving both interpretability and accuracy. We establish the first theoretical connection between alignment quality and generalization error, and validate our model on synthetic and real-world benchmarks. DeepFRC consistently outperforms state-of-the-art methods, especially in scenarios with complex temporal misalignment. Code is available at: https://github.com/Drivergo-93589/DeepFRC.
>
---
#### [replaced 031] REArtGS: Reconstructing and Generating Articulated Objects via 3D Gaussian Splatting with Geometric and Motion Constraints
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.06677v3](http://arxiv.org/pdf/2503.06677v3)**

> **作者:** Di Wu; Liu Liu; Zhou Linli; Anran Huang; Liangtu Song; Qiaojun Yu; Qi Wu; Cewu Lu
>
> **备注:** 11pages, 6 figures
>
> **摘要:** Articulated objects, as prevalent entities in human life, their 3D representations play crucial roles across various applications. However, achieving both high-fidelity textured surface reconstruction and dynamic generation for articulated objects remains challenging for existing methods. In this paper, we present REArtGS, a novel framework that introduces additional geometric and motion constraints to 3D Gaussian primitives, enabling high-quality textured surface reconstruction and generation for articulated objects. Specifically, given multi-view RGB images of arbitrary two states of articulated objects, we first introduce an unbiased Signed Distance Field (SDF) guidance to regularize Gaussian opacity fields, enhancing geometry constraints and improving surface reconstruction quality. Then we establish deformable fields for 3D Gaussians constrained by the kinematic structures of articulated objects, achieving unsupervised generation of surface meshes in unseen states. Extensive experiments on both synthetic and real datasets demonstrate our approach achieves high-quality textured surface reconstruction for given states, and enables high-fidelity surface generation for unseen states. Codes will be released after acceptance and the project website is at https://sites.google.com/view/reartgs/home.
>
---
#### [replaced 032] DynamiCtrl: Rethinking the Basic Structure and the Role of Text for High-quality Human Image Animation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21246v2](http://arxiv.org/pdf/2503.21246v2)**

> **作者:** Haoyu Zhao; Zhongang Qi; Cong Wang; Qingping Zheng; Guansong Lu; Fei Chen; Hang Xu; Zuxuan Wu
>
> **备注:** 16 pages, 11 figures
>
> **摘要:** With diffusion transformer (DiT) excelling in video generation, its use in specific tasks has drawn increasing attention. However, adapting DiT for pose-guided human image animation faces two core challenges: (a) existing U-Net-based pose control methods may be suboptimal for the DiT backbone; and (b) removing text guidance, as in previous approaches, often leads to semantic loss and model degradation. To address these issues, we propose DynamiCtrl, a novel framework for human animation in video DiT architecture. Specifically, we use a shared VAE encoder for human images and driving poses, unifying them into a common latent space, maintaining pose fidelity, and eliminating the need for an expert pose encoder during video denoising. To integrate pose control into the DiT backbone effectively, we propose a novel Pose-adaptive Layer Norm model. It injects normalized pose features into the denoising process via conditioning on visual tokens, enabling seamless and scalable pose control across DiT blocks. Furthermore, to overcome the shortcomings of text removal, we introduce the "Joint-text" paradigm, which preserves the role of text embeddings to provide global semantic context. Through full-attention blocks, image and pose features are aligned with text features, enhancing semantic consistency, leveraging pretrained knowledge, and enabling multi-level control. Experiments verify the superiority of DynamiCtrl on benchmark and self-collected data (e.g., achieving the best LPIPS of 0.166), demonstrating strong character control and high-quality synthesis. The project page is available at https://gulucaptain.github.io/DynamiCtrl/.
>
---
#### [replaced 033] Probing Human Visual Robustness with Neurally-Guided Deep Neural Networks
- **分类: cs.CV; cs.AI; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2405.02564v2](http://arxiv.org/pdf/2405.02564v2)**

> **作者:** Zhenan Shao; Linjian Ma; Yiqing Zhou; Yibo Jacky Zhang; Sanmi Koyejo; Bo Li; Diane M. Beck
>
> **摘要:** Humans effortlessly navigate the dynamic visual world, yet deep neural networks (DNNs), despite excelling at many visual tasks, are surprisingly vulnerable to minor image perturbations. Past theories suggest that human visual robustness arises from a representational space that evolves along the ventral visual stream (VVS) of the brain to increasingly tolerate object transformations. To test whether robustness is supported by such progression as opposed to being confined exclusively to specialized higher-order regions, we trained DNNs to align their representations with human neural responses from consecutive VVS regions while performing visual tasks. We demonstrate a hierarchical improvement in DNN robustness: alignment to higher-order VVS regions leads to greater improvement. To investigate the mechanism behind such robustness gains, we test a prominent hypothesis that attributes human robustness to the unique geometry of neural category manifolds in the VVS. We first reveal that more desirable manifold properties, specifically, smaller extent and better linear separability, indeed emerge across the human VVS. These properties can be inherited by neurally aligned DNNs and predict their subsequent robustness gains. Furthermore, we show that supervision from neural manifolds alone, via manifold guidance, is sufficient to qualitatively reproduce the hierarchical robustness improvements. Together, these results highlight the critical role of the evolving representational space across VVS in achieving robust visual inference, in part through the formation of more linearly separable category manifolds, which may in turn be leveraged to develop more robust AI systems.
>
---
#### [replaced 034] UniTok: A Unified Tokenizer for Visual Generation and Understanding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.20321v2](http://arxiv.org/pdf/2502.20321v2)**

> **作者:** Chuofan Ma; Yi Jiang; Junfeng Wu; Jihan Yang; Xin Yu; Zehuan Yuan; Bingyue Peng; Xiaojuan Qi
>
> **摘要:** Visual generative and understanding models typically rely on distinct tokenizers to process images, presenting a key challenge for unifying them within a single framework. Recent studies attempt to address this by connecting the training of VQVAE (for autoregressive generation) and CLIP (for understanding) to build a unified tokenizer. However, directly combining these training objectives has been observed to cause severe loss conflicts. In this paper, we show that reconstruction and semantic supervision do not inherently conflict. Instead, the underlying bottleneck stems from limited representational capacity of discrete token space. Building on these insights, we introduce UniTok, a unified tokenizer featuring a novel multi-codebook quantization mechanism that effectively scales up the vocabulary size and bottleneck dimension. In terms of final performance, UniTok sets a new record of 0.38 rFID and 78.6% zero-shot accuracy on ImageNet. Besides, UniTok can be seamlessly integrated into MLLMs to unlock native visual generation capability, without compromising the understanding performance. Additionally, we show that UniTok favors cfg-free generation, reducing gFID from 14.6 to 2.5 on ImageNet 256$\times$256 benchmark. GitHub: https://github.com/FoundationVision/UniTok.
>
---
#### [replaced 035] Rolling with the Punches: Resilient Contrastive Pre-training under Non-Stationary Drift
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.07620v2](http://arxiv.org/pdf/2502.07620v2)**

> **作者:** Xiaoyu Yang; Jie Lu; En Yu
>
> **备注:** 17pages, 3 figures
>
> **摘要:** The remarkable success of large-scale contrastive pre-training, fueled by vast and curated datasets, is encountering new frontiers as the scaling paradigm evolves. A critical emerging challenge is the effective pre-training of models on dynamic data streams characterized by concept drift, unpredictable changes in the underlying data distribution. This paper undertakes a foundational investigation of this issue. We first reveal that conventional contrastive pre-training methods are notably vulnerable to concept drift, leading to significant biases in the learned feature space of pre-trained models. To systematically analyze these effects, we construct a structural causal model that elucidates how drift acts as a confounder, distorting learned representations. Based on these causal insights, we propose Resilient Contrastive Pre-training (RCP), a novel method incorporating causal intervention. RCP introduces a causally-informed objective designed to mitigate drift-induced biases by leveraging targeted interventions. RCP is designed for simple and scalable implementation and exhibits notable adaptability, promoting robust pre-training on evolving data. Comprehensive experiments across diverse downstream tasks compellingly demonstrate that RCP effectively alleviates the detrimental impact of concept drift, yielding more resilient and generalizable representations.
>
---
#### [replaced 036] Scalable Density-based Clustering with Random Projections
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2402.15679v2](http://arxiv.org/pdf/2402.15679v2)**

> **作者:** Haochuan Xu; Ninh Pham
>
> **备注:** Appear in NeurIPS 2024 with the new title "Scalable DBSCAN with Random Projections"
>
> **摘要:** We present sDBSCAN, a scalable density-based clustering algorithm in high dimensions with cosine distance. Utilizing the neighborhood-preserving property of random projections, sDBSCAN can quickly identify core points and their neighborhoods, the primary hurdle of density-based clustering. Theoretically, sDBSCAN outputs a clustering structure similar to DBSCAN under mild conditions with high probability. To further facilitate sDBSCAN, we present sOPTICS, a scalable OPTICS for interactive exploration of the intrinsic clustering structure. We also extend sDBSCAN and sOPTICS to L2, L1, $\chi^2$, and Jensen-Shannon distances via random kernel features. Empirically, sDBSCAN is significantly faster and provides higher accuracy than many other clustering algorithms on real-world million-point data sets. On these data sets, sDBSCAN and sOPTICS run in a few minutes, while the scikit-learn's counterparts demand several hours or cannot run due to memory constraints.
>
---
#### [replaced 037] Long-Context Autoregressive Video Modeling with Next-Frame Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19325v3](http://arxiv.org/pdf/2503.19325v3)**

> **作者:** Yuchao Gu; Weijia Mao; Mike Zheng Shou
>
> **备注:** Project page at https://farlongctx.github.io/
>
> **摘要:** Long-context video modeling is essential for enabling generative models to function as world simulators, as they must maintain temporal coherence over extended time spans. However, most existing models are trained on short clips, limiting their ability to capture long-range dependencies, even with test-time extrapolation. While training directly on long videos is a natural solution, the rapid growth of vision tokens makes it computationally prohibitive. To support exploring efficient long-context video modeling, we first establish a strong autoregressive baseline called Frame AutoRegressive (FAR). FAR models temporal dependencies between continuous frames, converges faster than video diffusion transformers, and outperforms token-level autoregressive models. Based on this baseline, we observe context redundancy in video autoregression. Nearby frames are critical for maintaining temporal consistency, whereas distant frames primarily serve as context memory. To eliminate this redundancy, we propose the long short-term context modeling using asymmetric patchify kernels, which apply large kernels to distant frames to reduce redundant tokens, and standard kernels to local frames to preserve fine-grained detail. This significantly reduces the training cost of long videos. Our method achieves state-of-the-art results on both short and long video generation, providing an effective baseline for long-context autoregressive video modeling.
>
---
#### [replaced 038] Bridge the Modality and Capability Gaps in Vision-Language Model Selection
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2403.13797v3](http://arxiv.org/pdf/2403.13797v3)**

> **作者:** Chao Yi; Yu-Hang He; De-Chuan Zhan; Han-Jia Ye
>
> **备注:** fix typo in figure 2 "Capability Gap"
>
> **摘要:** Vision Language Models (VLMs) excel in zero-shot image classification by pairing images with textual category names. The expanding variety of Pre-Trained VLMs enhances the likelihood of identifying a suitable VLM for specific tasks. To better reuse the VLM resource and fully leverage its potential on different zero-shot image classification tasks, a promising strategy is selecting appropriate Pre-Trained VLMs from the VLM Zoo, relying solely on the text data of the target dataset without access to the dataset's images. In this paper, we analyze two inherent challenges in assessing the ability of a VLM in this Language-Only VLM selection: the "Modality Gap" - the disparity in VLM's embeddings across two different modalities, making text a less reliable substitute for images; and the "Capability Gap" - the discrepancy between the VLM's overall ranking and its ranking for target dataset, hindering direct prediction of a model's dataset-specific performance from its general performance. We propose VLM Selection With gAp Bridging (SWAB) to mitigate the negative impact of two gaps. SWAB first adopts optimal transport to capture the relevance between open-source and target datasets with a transportation matrix. It then uses this matrix to transfer useful statistics of VLMs from open-source datasets to the target dataset for bridging two gaps. By bridging two gaps to obtain better substitutes for test images, SWAB can accurately predict the performance ranking of different VLMs on the target task without the need for the dataset's images. Experiments across various VLMs and image classification datasets validate SWAB's effectiveness.
>
---
#### [replaced 039] MoVer: Motion Verification for Motion Graphics Animations
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.13372v2](http://arxiv.org/pdf/2502.13372v2)**

> **作者:** Jiaju Ma; Maneesh Agrawala
>
> **备注:** Accepted to ACM Transactions on Graphics (SIGGRAPH 2025)
>
> **摘要:** While large vision-language models can generate motion graphics animations from text prompts, they regularly fail to include all spatio-temporal properties described in the prompt. We introduce MoVer, a motion verification DSL based on first-order logic that can check spatio-temporal properties of a motion graphics animation. We identify a general set of such properties that people commonly use to describe animations (e.g., the direction and timing of motions, the relative positioning of objects, etc.). We implement these properties as predicates in MoVer and provide an execution engine that can apply a MoVer program to any input SVG-based motion graphics animation. We then demonstrate how MoVer can be used in an LLM-based synthesis and verification pipeline for iteratively refining motion graphics animations. Given a text prompt, our pipeline synthesizes a motion graphics animation and a corresponding MoVer program. Executing the verification program on the animation yields a report of the predicates that failed and the report can be automatically fed back to LLM to iteratively correct the animation. To evaluate our pipeline, we build a synthetic dataset of 5600 text prompts paired with ground truth MoVer verification programs. We find that while our LLM-based pipeline is able to automatically generate a correct motion graphics animation for 58.8% of the test prompts without any iteration, this number raises to 93.6% with up to 50 correction iterations. Our code and dataset are at https://mover-dsl.github.io.
>
---
#### [replaced 040] This&That: Language-Gesture Controlled Video Generation for Robot Planning
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.05530v2](http://arxiv.org/pdf/2407.05530v2)**

> **作者:** Boyang Wang; Nikhil Sridhar; Chao Feng; Mark Van der Merwe; Adam Fishman; Nima Fazeli; Jeong Joon Park
>
> **摘要:** Clear, interpretable instructions are invaluable when attempting any complex task. Good instructions help to clarify the task and even anticipate the steps needed to solve it. In this work, we propose a robot learning framework for communicating, planning, and executing a wide range of tasks, dubbed This&That. This&That solves general tasks by leveraging video generative models, which, through training on internet-scale data, contain rich physical and semantic context. In this work, we tackle three fundamental challenges in video-based planning: 1) unambiguous task communication with simple human instructions, 2) controllable video generation that respects user intent, and 3) translating visual plans into robot actions. This&That uses language-gesture conditioning to generate video predictions, as a succinct and unambiguous alternative to existing language-only methods, especially in complex and uncertain environments. These video predictions are then fed into a behavior cloning architecture dubbed Diffusion Video to Action (DiVA), which outperforms prior state-of-the-art behavior cloning and video-based planning methods by substantial margins.
>
---
#### [replaced 041] Dataset Distillation with Probabilistic Latent Features
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.06647v2](http://arxiv.org/pdf/2505.06647v2)**

> **作者:** Zhe Li; Sarah Cechnicka; Cheng Ouyang; Katharina Breininger; Peter Schüffler; Bernhard Kainz
>
> **备注:** 23 pages
>
> **摘要:** As deep learning models grow in complexity and the volume of training data increases, reducing storage and computational costs becomes increasingly important. Dataset distillation addresses this challenge by synthesizing a compact set of synthetic data that can effectively replace the original dataset in downstream classification tasks. While existing methods typically rely on mapping data from pixel space to the latent space of a generative model, we propose a novel stochastic approach that models the joint distribution of latent features. This allows our method to better capture spatial structures and produce diverse synthetic samples, which benefits model training. Specifically, we introduce a low-rank multivariate normal distribution parameterized by a lightweight network. This design maintains low computational complexity and is compatible with various matching networks used in dataset distillation. After distillation, synthetic images are generated by feeding the learned latent features into a pretrained generator. These synthetic images are then used to train classification models, and performance is evaluated on real test set. We validate our method on several benchmarks, including ImageNet subsets, CIFAR-10, and the MedMNIST histopathological dataset. Our approach achieves state-of-the-art cross architecture performance across a range of backbone architectures, demonstrating its generality and effectiveness.
>
---
#### [replaced 042] PrePrompt: Predictive prompting for class incremental learning
- **分类: cs.CV; I.5.4**

- **链接: [http://arxiv.org/pdf/2505.08586v2](http://arxiv.org/pdf/2505.08586v2)**

> **作者:** Libo Huang; Zhulin An; Chuanguang Yang; Boyu Diao; Fei Wang; Yan Zeng; Zhifeng Hao; Yongjun Xu
>
> **备注:** 16 pages, 29 figures, conference
>
> **摘要:** Class Incremental Learning (CIL) based on pre-trained models offers a promising direction for open-world continual learning. Existing methods typically rely on correlation-based strategies, where an image's classification feature is used as a query to retrieve the most related key prompts and select the corresponding value prompts for training. However, these approaches face an inherent limitation: fitting the entire feature space of all tasks with only a few trainable prompts is fundamentally challenging. We propose Predictive Prompting (PrePrompt), a novel CIL framework that circumvents correlation-based limitations by leveraging pre-trained models' natural classification ability to predict task-specific prompts. Specifically, PrePrompt decomposes CIL into a two-stage prediction framework: task-specific prompt prediction followed by label prediction. While theoretically appealing, this framework risks bias toward recent classes due to missing historical data for older classifier calibration. PrePrompt then mitigates this by incorporating feature translation, dynamically balancing stability and plasticity. Experiments across multiple benchmarks demonstrate PrePrompt's superiority over state-of-the-art prompt-based CIL methods. Code available at \href{github.com/libo-huang/preprompt}{github.com/libo-huang/preprompt}.
>
---
#### [replaced 043] Boosting Diffusion-Based Text Image Super-Resolution Model Towards Generalized Real-World Scenarios
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07232v3](http://arxiv.org/pdf/2503.07232v3)**

> **作者:** Chenglu Pan; Xiaogang Xu; Ganggui Ding; Yunke Zhang; Wenbo Li; Jiarong Xu; Qingbiao Wu
>
> **摘要:** Restoring low-resolution text images presents a significant challenge, as it requires maintaining both the fidelity and stylistic realism of the text in restored images. Existing text image restoration methods often fall short in hard situations, as the traditional super-resolution models cannot guarantee clarity, while diffusion-based methods fail to maintain fidelity. In this paper, we introduce a novel framework aimed at improving the generalization ability of diffusion models for text image super-resolution (SR), especially promoting fidelity. First, we propose a progressive data sampling strategy that incorporates diverse image types at different stages of training, stabilizing the convergence and improving the generalization. For the network architecture, we leverage a pre-trained SR prior to provide robust spatial reasoning capabilities, enhancing the model's ability to preserve textual information. Additionally, we employ a cross-attention mechanism to better integrate textual priors. To further reduce errors in textual priors, we utilize confidence scores to dynamically adjust the importance of textual features during training. Extensive experiments on real-world datasets demonstrate that our approach not only produces text images with more realistic visual appearances but also improves the accuracy of text structure.
>
---
#### [replaced 044] How Panel Layouts Define Manga: Insights from Visual Ablation Experiments
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.19141v2](http://arxiv.org/pdf/2412.19141v2)**

> **作者:** Siyuan Feng; Teruya Yoshinaga; Katsuhiko Hayashi; Koki Washio; Hidetaka Kamigaito
>
> **备注:** 7 pages, Final camera-ready version for CogSci 2025. Minor revision and improved figure quality
>
> **摘要:** Today, manga has gained worldwide popularity. However, the question of how various elements of manga, such as characters, text, and panel layouts, reflect the uniqueness of a particular work, or even define it, remains an unexplored area. In this paper, we aim to quantitatively and qualitatively analyze the visual characteristics of manga works, with a particular focus on panel layout features. As a research method, we used facing page images of manga as input to train a deep learning model for predicting manga titles, examining classification accuracy to quantitatively analyze these features. Specifically, we conducted ablation studies by limiting page image information to panel frames to analyze the characteristics of panel layouts. Through a series of quantitative experiments using all 104 works, 12 genres, and 10,122 facing page images from the Manga109 dataset, as well as qualitative analysis using Grad-CAM, our study demonstrates that the uniqueness of manga works is strongly reflected in their panel layouts.
>
---
#### [replaced 045] Knowledge-Informed Multi-Agent Trajectory Prediction at Signalized Intersections for Infrastructure-to-Everything
- **分类: cs.RO; cs.CV; cs.MA**

- **链接: [http://arxiv.org/pdf/2501.13461v2](http://arxiv.org/pdf/2501.13461v2)**

> **作者:** Huilin Yin; Yangwenhui Xu; Jiaxiang Li; Hao Zhang; Gerhard Rigoll
>
> **摘要:** Multi-agent trajectory prediction at signalized intersections is crucial for developing efficient intelligent transportation systems and safe autonomous driving systems. Due to the complexity of intersection scenarios and the limitations of single-vehicle perception, the performance of vehicle-centric prediction methods has reached a plateau. In this paper, we introduce an Infrastructure-to-Everything (I2X) collaborative prediction scheme. In this scheme, roadside units (RSUs) independently forecast the future trajectories of all vehicles and transmit these predictions unidirectionally to subscribing vehicles. Building on this scheme, we propose I2XTraj, a dedicated infrastructure-based trajectory prediction model. I2XTraj leverages real-time traffic signal states, prior maneuver strategy knowledge, and multi-agent interactions to generate accurate, joint multi-modal trajectory prediction. First, a continuous signal-informed mechanism is proposed to adaptively process real-time traffic signals to guide trajectory proposal generation under varied intersection configurations. Second, a driving strategy awareness mechanism estimates the joint distribution of maneuver strategies by integrating spatial priors of intersection areas with dynamic vehicle states, enabling coverage of the full set of feasible maneuvers. Third, a spatial-temporal-mode attention network models multi-agent interactions to refine and adjust joint trajectory outputs.Finally, I2XTraj is evaluated on two real-world datasets of signalized intersections, the V2X-Seq and the SinD drone dataset. In both single-infrastructure and online collaborative scenarios, our model outperforms state-of-the-art methods by over 30\% on V2X-Seq and 15\% on SinD, demonstrating strong generalizability and robustness.
>
---
#### [replaced 046] Controlled Training Data Generation with Diffusion Models
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.15309v2](http://arxiv.org/pdf/2403.15309v2)**

> **作者:** Teresa Yeo; Andrei Atanov; Harold Benoit; Aleksandr Alekseev; Ruchira Ray; Pooya Esmaeil Akhoondi; Amir Zamir
>
> **备注:** Project page at https://adversarial-prompts.epfl.ch/
>
> **摘要:** We present a method to control a text-to-image generative model to produce training data useful for supervised learning. Unlike previous works that employ an open-loop approach and pre-define prompts to generate new data using either a language model or human expertise, we develop an automated closed-loop system which involves two feedback mechanisms. The first mechanism uses feedback from a given supervised model and finds adversarial prompts that result in image generations that maximize the model loss. While these adversarial prompts result in diverse data informed by the model, they are not informed of the target distribution, which can be inefficient. Therefore, we introduce the second feedback mechanism that guides the generation process towards a certain target distribution. We call the method combining these two mechanisms Guided Adversarial Prompts. We perform our evaluations on different tasks, datasets and architectures, with different types of distribution shifts (spuriously correlated data, unseen domains) and demonstrate the efficiency of the proposed feedback mechanisms compared to open-loop approaches.
>
---
#### [replaced 047] Robust Emotion Recognition via Bi-Level Self-Supervised Continual Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.10575v2](http://arxiv.org/pdf/2505.10575v2)**

> **作者:** Adnan Ahmad; Bahareh Nakisa; Mohammad Naim Rastgoo
>
> **摘要:** Emotion recognition through physiological signals such as electroencephalogram (EEG) has become an essential aspect of affective computing and provides an objective way to capture human emotions. However, physiological data characterized by cross-subject variability and noisy labels hinder the performance of emotion recognition models. Existing domain adaptation and continual learning methods struggle to address these issues, especially under realistic conditions where data is continuously streamed and unlabeled. To overcome these limitations, we propose a novel bi-level self-supervised continual learning framework, SSOCL, based on a dynamic memory buffer. This bi-level architecture iteratively refines the dynamic buffer and pseudo-label assignments to effectively retain representative samples, enabling generalization from continuous, unlabeled physiological data streams for emotion recognition. The assigned pseudo-labels are subsequently leveraged for accurate emotion prediction. Key components of the framework, including a fast adaptation module and a cluster-mapping module, enable robust learning and effective handling of evolving data streams. Experimental validation on two mainstream EEG tasks demonstrates the framework's ability to adapt to continuous data streams while maintaining strong generalization across subjects, outperforming existing approaches.
>
---
#### [replaced 048] LR0.FM: Low-Res Benchmark and Improving Robustness for Zero-Shot Classification in Foundation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.03950v3](http://arxiv.org/pdf/2502.03950v3)**

> **作者:** Priyank Pathak; Shyam Marjit; Shruti Vyas; Yogesh S Rawat
>
> **备注:** Accepted to ICLR 2025
>
> **摘要:** Visual-language foundation Models (FMs) exhibit remarkable zero-shot generalization across diverse tasks, largely attributed to extensive pre-training on largescale datasets. However, their robustness on low-resolution/pixelated (LR) images, a common challenge in real-world scenarios, remains underexplored. We introduce LR0.FM, a comprehensive benchmark evaluating the impact of low resolution on the zero-shot classification performance of 10 FM(s) across 66 backbones and 15 datasets. We propose a novel metric, Weighted Aggregated Robustness, to address the limitations of existing metrics and better evaluate model performance across resolutions and datasets. Our key findings show that: (i) model size positively correlates with robustness to resolution degradation, (ii) pre-training dataset quality is more important than its size, and (iii) fine-tuned and higher resolution models are less robust against LR. Our analysis further reveals that the model makes semantically reasonable predictions at LR, and the lack of fine-grained details in input adversely impacts the model's initial layers more than the deeper layers. We use these insights and introduce a simple strategy, LR-TK0, to enhance the robustness of models without compromising their pre-trained weights. We demonstrate the effectiveness of LR-TK0 for robustness against low-resolution across several datasets and its generalization capability across backbones and other approaches. Code is available at https://github.com/shyammarjit/LR0.FM
>
---
#### [replaced 049] PainFormer: a Vision Foundation Model for Automatic Pain Assessment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01571v2](http://arxiv.org/pdf/2505.01571v2)**

> **作者:** Stefanos Gkikas; Raul Fernandez Rojas; Manolis Tsiknakis
>
> **摘要:** Pain is a manifold condition that impacts a significant percentage of the population. Accurate and reliable pain evaluation for the people suffering is crucial to developing effective and advanced pain management protocols. Automatic pain assessment systems provide continuous monitoring and support decision-making processes, ultimately aiming to alleviate distress and prevent functionality decline. This study introduces PainFormer, a vision foundation model based on multi-task learning principles trained simultaneously on 14 tasks/datasets with a total of 10.9 million samples. Functioning as an embedding extractor for various input modalities, the foundation model provides feature representations to the Embedding-Mixer, a transformer-based module that performs the final pain assessment. Extensive experiments employing behavioral modalities-including RGB, synthetic thermal, and estimated depth videos-and physiological modalities such as ECG, EMG, GSR, and fNIRS revealed that PainFormer effectively extracts high-quality embeddings from diverse input modalities. The proposed framework is evaluated on two pain datasets, BioVid and AI4Pain, and directly compared to 75 different methodologies documented in the literature. Experiments conducted in unimodal and multimodal settings demonstrate state-of-the-art performances across modalities and pave the way toward general-purpose models for automatic pain assessment.
>
---
#### [replaced 050] Adversarial Attacks on Both Face Recognition and Face Anti-spoofing Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.16940v2](http://arxiv.org/pdf/2405.16940v2)**

> **作者:** Fengfan Zhou; Qianyu Zhou; Hefei Ling; Xuequan Lu
>
> **备注:** Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025)
>
> **摘要:** Adversarial attacks on Face Recognition (FR) systems have demonstrated significant effectiveness against standalone FR models. However, their practicality diminishes in complete FR systems that incorporate Face Anti-Spoofing (FAS) models, as these models can detect and mitigate a substantial number of adversarial examples. To address this critical yet under-explored challenge, we introduce a novel attack setting that targets both FR and FAS models simultaneously, thereby enhancing the practicability of adversarial attacks on integrated FR systems. Specifically, we propose a new attack method, termed Reference-free Multi-level Alignment (RMA), designed to improve the capacity of black-box attacks on both FR and FAS models. The RMA framework is built upon three key components. Firstly, we propose an Adaptive Gradient Maintenance module to address the imbalances in gradient contributions between FR and FAS models. Secondly, we develop a Reference-free Intermediate Biasing module to improve the transferability of adversarial examples against FAS models. In addition, we introduce a Multi-level Feature Alignment module to reduce feature discrepancies at various levels of representation. Extensive experiments showcase the superiority of our proposed attack method to state-of-the-art adversarial attacks.
>
---
#### [replaced 051] Gradient descent with generalized Newton's method
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.02772v3](http://arxiv.org/pdf/2407.02772v3)**

> **作者:** Zhiqi Bu; Shiyun Xu
>
> **备注:** Accepted to ICLR 2025
>
> **摘要:** We propose the generalized Newton's method (GeN) -- a Hessian-informed approach that applies to any optimizer such as SGD and Adam, and covers the Newton-Raphson method as a sub-case. Our method automatically and dynamically selects the learning rate that accelerates the convergence, without the intensive tuning of the learning rate scheduler. In practice, our method is easily implementable, since it only requires additional forward passes with almost zero computational overhead (in terms of training time and memory cost), if the overhead is amortized over many iterations. We present extensive experiments on language and vision tasks (e.g. GPT and ResNet) to showcase that GeN optimizers match the state-of-the-art performance, which was achieved with carefully tuned learning rate schedulers.
>
---
#### [replaced 052] High Dynamic Range Novel View Synthesis with Single Exposure
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.01212v2](http://arxiv.org/pdf/2505.01212v2)**

> **作者:** Kaixuan Zhang; Hu Wang; Minxian Li; Mingwu Ren; Mao Ye; Xiatian Zhu
>
> **备注:** It has been accepted by ICML 2025
>
> **摘要:** High Dynamic Range Novel View Synthesis (HDR-NVS) aims to establish a 3D scene HDR model from Low Dynamic Range (LDR) imagery. Typically, multiple-exposure LDR images are employed to capture a wider range of brightness levels in a scene, as a single LDR image cannot represent both the brightest and darkest regions simultaneously. While effective, this multiple-exposure HDR-NVS approach has significant limitations, including susceptibility to motion artifacts (e.g., ghosting and blurring), high capture and storage costs. To overcome these challenges, we introduce, for the first time, the single-exposure HDR-NVS problem, where only single exposure LDR images are available during training. We further introduce a novel approach, Mono-HDR-3D, featuring two dedicated modules formulated by the LDR image formation principles, one for converting LDR colors to HDR counterparts, and the other for transforming HDR images to LDR format so that unsupervised learning is enabled in a closed loop. Designed as a meta-algorithm, our approach can be seamlessly integrated with existing NVS models. Extensive experiments show that Mono-HDR-3D significantly outperforms previous methods. Source code will be released.
>
---
#### [replaced 053] Differentially Private Synthetic Data via APIs 3: Using Simulators Instead of Foundation Model
- **分类: cs.LG; cs.CR; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2502.05505v2](http://arxiv.org/pdf/2502.05505v2)**

> **作者:** Zinan Lin; Tadas Baltrusaitis; Wenyu Wang; Sergey Yekhanin
>
> **摘要:** Differentially private (DP) synthetic data, which closely resembles the original private data while maintaining strong privacy guarantees, has become a key tool for unlocking the value of private data without compromising privacy. Recently, Private Evolution (PE) has emerged as a promising method for generating DP synthetic data. Unlike other training-based approaches, PE only requires access to inference APIs from foundation models, enabling it to harness the power of state-of-the-art (SoTA) models. However, a suitable foundation model for a specific private data domain is not always available. In this paper, we discover that the PE framework is sufficiently general to allow APIs beyond foundation models. In particular, we demonstrate that many SoTA data synthesizers that do not rely on neural networks--such as computer graphics-based image generators, which we refer to as simulators--can be effectively integrated into PE. This insight significantly broadens PE's applicability and unlocks the potential of powerful simulators for DP data synthesis. We explore this approach, named Sim-PE, in the context of image synthesis. Across four diverse simulators, Sim-PE performs well, improving the downstream classification accuracy of PE by up to 3x, reducing FID by up to 80%, and offering much greater efficiency. We also show that simulators and foundation models can be easily leveraged together within PE to achieve further improvements. The code is open-sourced in the Private Evolution Python library: https://github.com/microsoft/DPSDA.
>
---
#### [replaced 054] AUTO: Adaptive Outlier Optimization for Test-Time OOD Detection
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2303.12267v2](http://arxiv.org/pdf/2303.12267v2)**

> **作者:** Puning Yang; Jian Liang; Jie Cao; Ran He
>
> **备注:** 14 pages
>
> **摘要:** Out-of-distribution (OOD) detection aims to detect test samples that do not fall into any training in-distribution (ID) classes. Prior efforts focus on regularizing models with ID data only, largely underperforming counterparts that utilize auxiliary outliers. However, data safety and privacy make it infeasible to collect task-specific outliers in advance for different scenarios. Besides, using task-irrelevant outliers leads to inferior OOD detection performance. To address the above issue, we present a new setup called test-time OOD detection, which allows the deployed model to utilize real OOD data from the unlabeled data stream during testing. We propose Adaptive Outlier Optimization (AUTO) which allows for continuous adaptation of the OOD detector. Specifically, AUTO consists of three key components: 1) an in-out-aware filter to selectively annotate test samples with pseudo-ID and pseudo-OOD and ingeniously trigger the updating process while encountering each pseudo-OOD sample; 2) a dynamic-updated memory to overcome the catastrophic forgetting led by frequent parameter updates; 3) a prediction-aligning objective to calibrate the rough OOD objective during testing. Extensive experiments show that AUTO significantly improves OOD detection performance over state-of-the-art methods. Besides, evaluations on complicated scenarios (e.g. multi-OOD, time-series OOD) also conduct the superiority of AUTO.
>
---
#### [replaced 055] PillarTrack:Boosting Pillar Representation for Transformer-based 3D Single Object Tracking on Point Clouds
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.07495v2](http://arxiv.org/pdf/2404.07495v2)**

> **作者:** Weisheng Xu; Sifan Zhou; Jiaqi Xiong; Ziyu Zhao; Zhihang Yuan
>
> **摘要:** LiDAR-based 3D single object tracking (3D SOT) is a critical issue in robotics and autonomous driving. Existing 3D SOT methods typically adhere to a point-based processing pipeline, wherein the re-sampling operation invariably leads to either redundant or missing information, thereby impacting performance. To address these issues, we propose PillarTrack, a novel pillar-based 3D SOT framework. First, we transform sparse point clouds into dense pillars to preserve the local and global geometrics. Second, we propose a Pyramid-Encoded Pillar Feature Encoder (PE-PFE) design to enhance the robustness of pillar feature for translation/rotation/scale. Third, we present an efficient Transformer-based backbone from the perspective of modality differences. Finally, we construct our PillarTrack based on above designs. Extensive experiments show that our method achieves comparable performance on the KITTI and NuScenes datasets, significantly enhancing the performance of the baseline.
>
---
#### [replaced 056] Vision-Encoders (Already) Know What They See: Mitigating Object Hallucination via Simple Fine-Grained CLIPScore
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20034v2](http://arxiv.org/pdf/2502.20034v2)**

> **作者:** Hongseok Oh; Wonseok Hwang
>
> **备注:** 4 pages
>
> **摘要:** Recently, Large Vision-Language Models (LVLMs) show remarkable performance across various domains. However, these models suffer from object hallucination. This study revisits the previous claim that the primary cause of such hallucination lies in the limited representational capacity of the vision encoder. Our analysis reveals that the capacity of the vision encoder itself is already adequate for detecting object hallucination. Based on this insight, we propose a Fine-grained CLIPScore (F-CLIPScore), a simple yet effective evaluation metric that enhances object-level granularity by incorporating text embeddings at the noun level. Evaluations on the OHD-Caps benchmark show that F-CLIPScore significantly outperforms conventional CLIPScore in accuracy by a large margin of 39.6\% without additional training. We further demonstrate that F-CLIPScore-based data filtering reduces object hallucination in LVLMs (4.9\% in POPE).
>
---
#### [replaced 057] Feedback-Driven Vision-Language Alignment with Minimal Human Supervision
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.04568v2](http://arxiv.org/pdf/2501.04568v2)**

> **作者:** Giorgio Giannone; Ruoteng Li; Qianli Feng; Evgeny Perevodchikov; Rui Chen; Aleix Martinez
>
> **备注:** Preprint
>
> **摘要:** Vision-language models (VLMs) have demonstrated remarkable potential in integrating visual and linguistic information, but their performance is often constrained by the need for extensive, high-quality image-text training data. Curation of these image-text pairs is both time-consuming and computationally expensive. To address this challenge, we introduce SVP (Sampling-based Visual Projection), a novel framework that enhances vision-language alignment without relying on manually curated text-image pairs or preference annotation. SVP leverages a small set of manually selected images, self-captioning and a pre-trained grounding model as a feedback mechanism to elicit latent information in VLMs. We evaluate our approach across six key areas: captioning, referring, visual question answering, multitasking, hallucination control, and object recall. Results demonstrate significant improvements, including a 14 % average improvement in captioning tasks, up to 12 % increase in object recall, and significantly reduced hallucinations, while maintaining question-answering capabilities. Using SVP, a small VLM achieves hallucination reductions similar to a model five times larger, while a VLM with initially poor referring capabilities more than doubles its performance, approaching parity with a model twice its size.
>
---
#### [replaced 058] Parametric PerceptNet: A bio-inspired deep-net trained for Image Quality Assessment
- **分类: cs.CV; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2412.03210v3](http://arxiv.org/pdf/2412.03210v3)**

> **作者:** Jorge Vila-Tomás; Pablo Hernández-Cámara; Valero Laparra; Jesús Malo
>
> **摘要:** Human vision models are at the core of image processing. For instance, classical approaches to the problem of image quality are based on models that include knowledge about human vision. However, nowadays, deep learning approaches have obtained competitive results by simply approaching this problem as regression of human decisions, and training an standard network on human-rated datasets. These approaches have the advantages of being easily adaptable to a particular problem and they fit very efficiently when data is available. However, mainly due to the excess of parameters, they have the problems of lack of interpretability, and over-fitting. Here we propose a vision model that combines the best of both worlds by using a parametric neural network architecture. We parameterize the layers to have bioplausible functionality, and provide a set of bioplausible parameters. We analyzed different versions of the model and compared it with the non-parametric version. The parametric models achieve a three orders of magnitude reduction in the number of parameters without suffering in regression performance. Furthermore, we show that the parametric models behave better during training and are easier to interpret as vision models. Interestingly, we find that, even initialized with bioplausible trained for regression using human rated datasets, which we call the feature-spreading problem. This suggests that the deep learning approach is inherently flawed, and emphasizes the need to evaluate and train models beyond regression.
>
---
#### [replaced 059] Captured by Captions: On Memorization and its Mitigation in CLIP Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.07830v2](http://arxiv.org/pdf/2502.07830v2)**

> **作者:** Wenhao Wang; Adam Dziedzic; Grace C. Kim; Michael Backes; Franziska Boenisch
>
> **备注:** Accepted at ICLR 2025
>
> **摘要:** Multi-modal models, such as CLIP, have demonstrated strong performance in aligning visual and textual representations, excelling in tasks like image retrieval and zero-shot classification. Despite this success, the mechanisms by which these models utilize training data, particularly the role of memorization, remain unclear. In uni-modal models, both supervised and self-supervised, memorization has been shown to be essential for generalization. However, it is not well understood how these findings would apply to CLIP, which incorporates elements from both supervised learning via captions that provide a supervisory signal similar to labels, and from self-supervised learning via the contrastive objective. To bridge this gap in understanding, we propose a formal definition of memorization in CLIP (CLIPMem) and use it to quantify memorization in CLIP models. Our results indicate that CLIP's memorization behavior falls between the supervised and self-supervised paradigms, with "mis-captioned" samples exhibiting highest levels of memorization. Additionally, we find that the text encoder contributes more to memorization than the image encoder, suggesting that mitigation strategies should focus on the text domain. Building on these insights, we propose multiple strategies to reduce memorization while at the same time improving utility--something that had not been shown before for traditional learning paradigms where reducing memorization typically results in utility decrease.
>
---
#### [replaced 060] See What You Seek: Semantic Contextual Integration for Cloth-Changing Person Re-Identification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.01345v2](http://arxiv.org/pdf/2412.01345v2)**

> **作者:** Xiyu Han; Xian Zhong; Wenxin Huang; Xuemei Jia; Xiaohan Yu; Alex Chichung Kot
>
> **备注:** 12 pages
>
> **摘要:** Cloth-changing person re-identification (CC-ReID) aims to match individuals across surveillance cameras despite variations in clothing. Existing methods typically mitigate the impact of clothing changes or enhance identity (ID)-relevant features, but they often struggle to capture complex semantic information. In this paper, we propose a novel prompt learning framework Semantic Contextual Integration (SCI), which leverages the visual-textual representation capabilities of CLIP to reduce clothing-induced discrepancies and strengthen ID cues. Specifically, we introduce the Semantic Separation Enhancement (SSE) module, which employs dual learnable text tokens to disentangle clothing-related semantics from confounding factors, thereby isolating ID-relevant features. Furthermore, we develop a Semantic-Guided Interaction Module (SIM) that uses orthogonalized text features to guide visual representations, sharpening the focus of the model on distinctive ID characteristics. This semantic integration improves the discriminative power of the model and enriches the visual context with high-dimensional insights. Extensive experiments on three CC-ReID datasets demonstrate that our method outperforms state-of-the-art techniques. The code will be released at https://github.com/hxy-499/CCREID-SCI.
>
---
#### [replaced 061] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10143v4](http://arxiv.org/pdf/2504.10143v4)**

> **作者:** Yichao Cai; Yuhang Liu; Erdun Gao; Tianjiao Jiang; Zhen Zhang; Anton van den Hengel; Javen Qinfeng Shi
>
> **摘要:** Multimodal representation learning, exemplified by multimodal contrastive learning (MMCL) using image-text pairs, aims to learn powerful representations by aligning cues across modalities. This approach relies on the core assumption that the exemplar image-text pairs constitute two representations of an identical concept. However, recent research has revealed that real-world datasets often exhibit cross-modal misalignment. There are two distinct viewpoints on how to address this issue: one suggests mitigating the misalignment, and the other leveraging it. We seek here to reconcile these seemingly opposing perspectives, and to provide a practical guide for practitioners. Using latent variable models we thus formalize cross-modal misalignment by introducing two specific mechanisms: Selection bias, where some semantic variables are absent in the text, and perturbation bias, where semantic variables are altered -- both leading to misalignment in data pairs. Our theoretical analysis demonstrates that, under mild assumptions, the representations learned by MMCL capture exactly the information related to the subset of the semantic variables invariant to selection and perturbation biases. This provides a unified perspective for understanding misalignment. Based on this, we further offer actionable insights into how misalignment should inform the design of real-world ML systems. We validate our theoretical findings via extensive empirical studies on both synthetic data and real image-text datasets, shedding light on the nuanced impact of cross-modal misalignment on multimodal representation learning.
>
---
#### [replaced 062] MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10526v2](http://arxiv.org/pdf/2505.10526v2)**

> **作者:** Mugilan Ganesan; Shane Segal; Ankur Aggarwal; Nish Sinnadurai; Sean Lie; Vithursan Thangarasa
>
> **备注:** Main paper: 11 pages, 4 figures, 3 tables. Supplementary: 1 page
>
> **摘要:** Speculative decoding significantly accelerates language model inference by enabling a lightweight draft model to propose multiple tokens that a larger target model verifies simultaneously. However, applying this technique to vision-language models (VLMs) presents two fundamental challenges: small language models that could serve as efficient drafters lack the architectural components to process visual inputs, and their token predictions fail to match those of VLM target models that consider visual context. We introduce Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models (MASSV), which transforms existing small language models into effective multimodal drafters through a two-phase approach. MASSV first connects the target VLM's vision encoder to the draft model via a lightweight trainable projector, then applies self-distilled visual instruction tuning using responses generated by the target VLM to align token predictions. Comprehensive experiments across the Qwen2.5-VL and Gemma3 model families demonstrate that MASSV increases accepted length by up to 30% and delivers end-to-end inference speedups of up to 1.46x on visually-grounded tasks. MASSV provides a scalable, architecture-compatible method for accelerating both current and future VLMs.
>
---
#### [replaced 063] Remote sensing colour image semantic segmentation of trails created by large herbivorous Mammals
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.12121v3](http://arxiv.org/pdf/2504.12121v3)**

> **作者:** Jose Francisco Diez-Pastor; Francisco Javier Gonzalez-Moya; Pedro Latorre-Carmona; Francisco Javier Perez-Barbería; Ludmila I. Kuncheva; Antonio Canepa-Oneto; Alvar Arnaiz-González; Cesar Garcia-Osorio
>
> **备注:** 24 pages, 6 figures. Submitted to "International Journal of Remote Sensing"
>
> **摘要:** Identifying spatial regions where biodiversity is threatened is crucial for effective ecosystem conservation and monitoring. In this stydy, we assessed varios machine learning methods to detect grazing trails automatically. We tested five semantic segmentation models combined with 14 different encoder networks. The best combination was UNet with MambaOut encoder. The solution proposed could be used as the basis for tools aiming at mapping and tracking changes in grazing trails on a continuous temporal basis.
>
---
#### [replaced 064] TexPro: Text-guided PBR Texturing with Procedural Material Modeling
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.15891v2](http://arxiv.org/pdf/2410.15891v2)**

> **作者:** Ziqiang Dang; Wenqi Dong; Zesong Yang; Bangbang Yang; Liang Li; Yuewen Ma; Zhaopeng Cui
>
> **备注:** Accepted by CVM 2025 and CVMJ (Computational Visual Media Journal)
>
> **摘要:** In this paper, we present TexPro, a novel method for high-fidelity material generation for input 3D meshes given text prompts. Unlike existing text-conditioned texture generation methods that typically generate RGB textures with baked lighting, TexPro is able to produce diverse texture maps via procedural material modeling, which enables physically-based rendering, relighting, and additional benefits inherent to procedural materials. Specifically, we first generate multi-view reference images given the input textual prompt by employing the latest text-to-image model. We then derive texture maps through rendering-based optimization with recent differentiable procedural materials. To this end, we design several techniques to handle the misalignment between the generated multi-view images and 3D meshes, and introduce a novel material agent that enhances material classification and matching by exploring both part-level understanding and object-aware material reasoning. Experiments demonstrate the superiority of the proposed method over existing SOTAs, and its capability of relighting.
>
---
#### [replaced 065] Culture-TRIP: Culturally-Aware Text-to-Image Generation with Iterative Prompt Refinement
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.16902v2](http://arxiv.org/pdf/2502.16902v2)**

> **作者:** Suchae Jeong; Inseong Choi; Youngsik Yun; Jihie Kim
>
> **备注:** 31 pages, 23 figures, Accepted by NAACL 2025
>
> **摘要:** Text-to-Image models, including Stable Diffusion, have significantly improved in generating images that are highly semantically aligned with the given prompts. However, existing models may fail to produce appropriate images for the cultural concepts or objects that are not well known or underrepresented in western cultures, such as `hangari' (Korean utensil). In this paper, we propose a novel approach, Culturally-Aware Text-to-Image Generation with Iterative Prompt Refinement (Culture-TRIP), which refines the prompt in order to improve the alignment of the image with such culture nouns in text-to-image models. Our approach (1) retrieves cultural contexts and visual details related to the culture nouns in the prompt and (2) iteratively refines and evaluates the prompt based on a set of cultural criteria and large language models. The refinement process utilizes the information retrieved from Wikipedia and the Web. Our user survey, conducted with 66 participants from eight different countries demonstrates that our proposed approach enhances the alignment between the images and the prompts. In particular, C-TRIP demonstrates improved alignment between the generated images and underrepresented culture nouns. Resource can be found at https://shane3606.github.io/Culture-TRIP.
>
---
#### [replaced 066] ForestSplats: Deformable transient field for Gaussian Splatting in the Wild
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06179v2](http://arxiv.org/pdf/2503.06179v2)**

> **作者:** Wongi Park; Myeongseok Nam; Siwon Kim; Sangwoo Jo; Soomok Lee
>
> **摘要:** Recently, 3D Gaussian Splatting (3D-GS) has emerged, showing real-time rendering speeds and high-quality results in static scenes. Although 3D-GS shows effectiveness in static scenes, their performance significantly degrades in real-world environments due to transient objects, lighting variations, and diverse levels of occlusion. To tackle this, existing methods estimate occluders or transient elements by leveraging pre-trained models or integrating additional transient field pipelines. However, these methods still suffer from two defects: 1) Using semantic features from the Vision Foundation model (VFM) causes additional computational costs. 2) The transient field requires significant memory to handle transient elements with per-view Gaussians and struggles to define clear boundaries for occluders, solely relying on photometric errors. To address these problems, we propose ForestSplats, a novel approach that leverages the deformable transient field and a superpixel-aware mask to efficiently represent transient elements in the 2D scene across unconstrained image collections and effectively decompose static scenes from transient distractors without VFM. We designed the transient field to be deformable, capturing per-view transient elements. Furthermore, we introduce a superpixel-aware mask that clearly defines the boundaries of occluders by considering photometric errors and superpixels. Additionally, we propose uncertainty-aware densification to avoid generating Gaussians within the boundaries of occluders during densification. Through extensive experiments across several benchmark datasets, we demonstrate that ForestSplats outperforms existing methods without VFM and shows significant memory efficiency in representing transient elements.
>
---
#### [replaced 067] Data-centric Prediction Explanation via Kernelized Stein Discrepancy
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2403.15576v3](http://arxiv.org/pdf/2403.15576v3)**

> **作者:** Mahtab Sarvmaili; Hassan Sajjad; Ga Wu
>
> **摘要:** Existing example-based prediction explanation methods often bridge test and training data points through the model's parameters or latent representations. While these methods offer clues to the causes of model predictions, they often exhibit innate shortcomings, such as incurring significant computational overhead or producing coarse-grained explanations. This paper presents a Highly-precise and Data-centric Explan}ation (HD-Explain) prediction explanation method that exploits properties of Kernelized Stein Discrepancy (KSD). Specifically, the KSD uniquely defines a parameterized kernel function for a trained model that encodes model-dependent data correlation. By leveraging the kernel function, one can identify training samples that provide the best predictive support to a test point efficiently. We conducted thorough analyses and experiments across multiple classification domains, where we show that HD-Explain outperforms existing methods from various aspects, including 1) preciseness (fine-grained explanation), 2) consistency, and 3) computation efficiency, leading to a surprisingly simple, effective, and robust prediction explanation solution.
>
---
#### [replaced 068] Rethinking Image Forgery Detection via Soft Contrastive Learning and Unsupervised Clustering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2308.09307v2](http://arxiv.org/pdf/2308.09307v2)**

> **作者:** Haiwei Wu; Yiming Chen; Jiantao Zhou; Yuanman Li
>
> **摘要:** Image forgery detection aims to detect and locate forged regions in an image. Most existing forgery detection algorithms formulate classification problems to classify pixels into forged or pristine. However, the definition of forged and pristine pixels is only relative within one single image, e.g., a forged region in image A is actually a pristine one in its source image B (splicing forgery). Such a relative definition has been severely overlooked by existing methods, which unnecessarily mix forged (pristine) regions across different images into the same category. To resolve this dilemma, we propose the FOrensic ContrAstive cLustering (FOCAL) method, a novel, simple yet very effective paradigm based on soft contrastive learning and unsupervised clustering for the image forgery detection. Specifically, FOCAL 1) designs a soft contrastive learning (SCL) to supervise the high-level forensic feature extraction in an image-by-image manner, explicitly reflecting the above relative definition; 2) employs an on-the-fly unsupervised clustering algorithm (instead of a trained one) to cluster the learned features into forged/pristine categories, further suppressing the cross-image influence from training data; and 3) allows to further boost the detection performance via simple feature-level concatenation without the need of retraining. Extensive experimental results over six public testing datasets demonstrate that our proposed FOCAL significantly outperforms the state-of-the-art competitors by big margins: +24.8% on Coverage, +18.9% on Columbia, +17.3% on FF++, +15.3% on MISD, +15.0% on CASIA and +10.5% on NIST in terms of IoU (see also Fig. 1). The paradigm of FOCAL could bring fresh insights and serve as a novel benchmark for the image forgery detection task. The code is available at https://github.com/HighwayWu/FOCAL.
>
---
#### [replaced 069] DeLoRA: Decoupling Angles and Strength in Low-rank Adaptation
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18225v2](http://arxiv.org/pdf/2503.18225v2)**

> **作者:** Massimo Bini; Leander Girrbach; Zeynep Akata
>
> **备注:** ICLR 2025
>
> **摘要:** Parameter-Efficient FineTuning (PEFT) methods have recently gained significant popularity thanks to the widespread availability of large-scale pretrained models. These methods allow for quick adaptation to downstream tasks with minimal computational cost. However, popular finetuning methods such as LoRA exhibit limited robustness when it comes to hyperparameter choices or extended training regimes, preventing optimal out-of-the-box performance. In contrast, bounded approaches, such as ETHER, provide greater robustness but are limited to extremely low-rank adaptations and fixed-strength transformations, reducing their adaptation expressive power. In this work, we propose Decoupled Low-rank Adaptation (DeLoRA), a novel finetuning method that normalizes and scales learnable low-rank matrices. By bounding the distance of the transformation, DeLoRA effectively decouples the angular learning from the adaptation strength, enhancing robustness without compromising performance. Through evaluations on subject-driven image generation, natural language understanding, and instruction tuning, we show that DeLoRA matches or surpasses performance of competing PEFT methods, while exhibiting stronger robustness. Code is available at https://github.com/ExplainableML/DeLoRA.
>
---
#### [replaced 070] Vision Transformers on the Edge: A Comprehensive Survey of Model Compression and Acceleration Strategies
- **分类: cs.CV; cs.AR**

- **链接: [http://arxiv.org/pdf/2503.02891v3](http://arxiv.org/pdf/2503.02891v3)**

> **作者:** Shaibal Saha; Lanyu Xu
>
> **备注:** Accepted in Neurocomputing, Elsevier
>
> **摘要:** In recent years, vision transformers (ViTs) have emerged as powerful and promising techniques for computer vision tasks such as image classification, object detection, and segmentation. Unlike convolutional neural networks (CNNs), which rely on hierarchical feature extraction, ViTs treat images as sequences of patches and leverage self-attention mechanisms. However, their high computational complexity and memory demands pose significant challenges for deployment on resource-constrained edge devices. To address these limitations, extensive research has focused on model compression techniques and hardware-aware acceleration strategies. Nonetheless, a comprehensive review that systematically categorizes these techniques and their trade-offs in accuracy, efficiency, and hardware adaptability for edge deployment remains lacking. This survey bridges this gap by providing a structured analysis of model compression techniques, software tools for inference on edge, and hardware acceleration strategies for ViTs. We discuss their impact on accuracy, efficiency, and hardware adaptability, highlighting key challenges and emerging research directions to advance ViT deployment on edge platforms, including graphics processing units (GPUs), application-specific integrated circuit (ASICs), and field-programmable gate arrays (FPGAs). The goal is to inspire further research with a contemporary guide on optimizing ViTs for efficient deployment on edge devices.
>
---
#### [replaced 071] Semantic Shift Estimation via Dual-Projection and Classifier Reconstruction for Exemplar-Free Class-Incremental Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.05423v4](http://arxiv.org/pdf/2503.05423v4)**

> **作者:** Run He; Di Fang; Yicheng Xu; Yawen Cui; Ming Li; Cen Chen; Ziqian Zeng; Huiping Zhuang
>
> **备注:** Accepted by ICML 2025; Camera ready version
>
> **摘要:** Exemplar-Free Class-Incremental Learning (EFCIL) aims to sequentially learn from distinct categories without retaining exemplars but easily suffers from catastrophic forgetting of learned knowledge. While existing EFCIL methods leverage knowledge distillation to alleviate forgetting, they still face two critical challenges: semantic shift and decision bias. Specifically, the embeddings of old tasks shift in the embedding space after learning new tasks, and the classifier becomes biased towards new tasks due to training solely with new data, hindering the balance between old and new knowledge. To address these issues, we propose the Dual-Projection Shift Estimation and Classifier Reconstruction (DPCR) approach for EFCIL. DPCR effectively estimates semantic shift through a dual-projection, which combines a learnable transformation with a row-space projection to capture both task-wise and category-wise shifts. Furthermore, to mitigate decision bias, DPCR employs ridge regression to reformulate a classifier reconstruction process. This reconstruction exploits previous in covariance and prototype of each class after calibration with estimated shift, thereby reducing decision bias. Extensive experiments demonstrate that, on various datasets, DPCR effectively balances old and new tasks, outperforming state-of-the-art EFCIL methods. Our codes are available at https://github.com/RHe502/ICML25-DPCR.
>
---
#### [replaced 072] Integrating Extra Modality Helps Segmentor Find Camouflaged Objects Well
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.14471v2](http://arxiv.org/pdf/2502.14471v2)**

> **作者:** Chengyu Fang; Chunming He; Longxiang Tang; Yuelin Zhang; Chenyang Zhu; Yuqi Shen; Chubin Chen; Guoxia Xu; Xiu Li
>
> **备注:** 18 pages, 8 figures, 14 tables
>
> **摘要:** Camouflaged Object Segmentation (COS) remains challenging because camouflaged objects exhibit only subtle visual differences from their backgrounds and single-modality RGB methods provide limited cues, leading researchers to explore multimodal data to improve segmentation accuracy. In this work, we presenet MultiCOS, a novel framework that effectively leverages diverse data modalities to improve segmentation performance. MultiCOS comprises two modules: Bi-space Fusion Segmentor (BFSer), which employs a state space and a latent space fusion mechanism to integrate cross-modal features within a shared representation and employs a fusion-feedback mechanism to refine context-specific features, and Cross-modal Knowledge Learner (CKLer), which leverages external multimodal datasets to generate pseudo-modal inputs and establish cross-modal semantic associations, transferring knowledge to COS models when real multimodal pairs are missing. When real multimodal COS data are unavailable, CKLer yields additional segmentation gains using only non-COS multimodal sources. Experiments on standard COS benchmarks show that BFSer outperforms existing multimodal baselines with both real and pseudo-modal data. Code will be released at \href{https://github.com/cnyvfang/MultiCOS}{GitHub}.
>
---
#### [replaced 073] Omni-ID: Holistic Identity Representation Designed for Generative Tasks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.09694v2](http://arxiv.org/pdf/2412.09694v2)**

> **作者:** Guocheng Qian; Kuan-Chieh Wang; Or Patashnik; Negin Heravi; Daniil Ostashev; Sergey Tulyakov; Daniel Cohen-Or; Kfir Aberman
>
> **备注:** Accepted to CVPR'25. Webpage: https://snap-research.github.io/Omni-ID
>
> **摘要:** We introduce Omni-ID, a novel facial representation designed specifically for generative tasks. Omni-ID encodes holistic information about an individual's appearance across diverse expressions and poses within a fixed-size representation. It consolidates information from a varied number of unstructured input images into a structured representation, where each entry represents certain global or local identity features. Our approach uses a few-to-many identity reconstruction training paradigm, where a limited set of input images is used to reconstruct multiple target images of the same individual in various poses and expressions. A multi-decoder framework is further employed to leverage the complementary strengths of diverse decoders during training. Unlike conventional representations, such as CLIP and ArcFace, which are typically learned through discriminative or contrastive objectives, Omni-ID is optimized with a generative objective, resulting in a more comprehensive and nuanced identity capture for generative tasks. Trained on our MFHQ dataset -- a multi-view facial image collection, Omni-ID demonstrates substantial improvements over conventional representations across various generative tasks.
>
---
#### [replaced 074] Understanding the Effect of using Semantically Meaningful Tokens for Visual Representation Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.16401v2](http://arxiv.org/pdf/2405.16401v2)**

> **作者:** Neha Kalibhat; Priyatham Kattakinda; Sumit Nawathe; Arman Zarei; Nikita Seleznev; Samuel Sharpe; Senthil Kumar; Soheil Feizi
>
> **备注:** Published at CVPR Workshops 2025
>
> **摘要:** Vision transformers have established a precedent of patchifying images into uniformly-sized chunks before processing. We hypothesize that this design choice may limit models in learning comprehensive and compositional representations from visual data. This paper explores the notion of providing semantically-meaningful visual tokens to transformer encoders within a vision-language pre-training framework. Leveraging off-the-shelf segmentation and scene-graph models, we extract representations of instance segmentation masks (referred to as tangible tokens) and relationships and actions (referred to as intangible tokens). Subsequently, we pre-train a vision-side transformer by incorporating these newly extracted tokens and aligning the resultant embeddings with caption embeddings from a text-side encoder. To capture the structural and semantic relationships among visual tokens, we introduce additive attention weights, which are used to compute self-attention scores. Our experiments on COCO demonstrate notable improvements over ViTs in learned representation quality across text-to-image (+47%) and image-to-text retrieval (+44%) tasks. Furthermore, we showcase the advantages on compositionality benchmarks such as ARO (+18%) and Winoground (+10%).
>
---
#### [replaced 075] LightNeuS: Neural Surface Reconstruction in Endoscopy using Illumination Decline
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2309.02777v2](http://arxiv.org/pdf/2309.02777v2)**

> **作者:** Víctor M. Batlle; José M. M. Montiel; Pascal Fua; Juan D. Tardós
>
> **备注:** 13 pages, 7 figures, 1 table
>
> **摘要:** We propose a new approach to 3D reconstruction from sequences of images acquired by monocular endoscopes. It is based on two key insights. First, endoluminal cavities are watertight, a property naturally enforced by modeling them in terms of a signed distance function. Second, the scene illumination is variable. It comes from the endoscope's light sources and decays with the inverse of the squared distance to the surface. To exploit these insights, we build on NeuS, a neural implicit surface reconstruction technique with an outstanding capability to learn appearance and a SDF surface model from multiple views, but currently limited to scenes with static illumination. To remove this limitation and exploit the relation between pixel brightness and depth, we modify the NeuS architecture to explicitly account for it and introduce a calibrated photometric model of the endoscope's camera and light source. Our method is the first one to produce watertight reconstructions of whole colon sections. We demonstrate excellent accuracy on phantom imagery. Remarkably, the watertight prior combined with illumination decline, allows to complete the reconstruction of unseen portions of the surface with acceptable accuracy, paving the way to automatic quality assessment of cancer screening explorations, measuring the global percentage of observed mucosa.
>
---
#### [replaced 076] A Quality-Centric Framework for Generic Deepfake Detection
- **分类: cs.CV; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.05335v3](http://arxiv.org/pdf/2411.05335v3)**

> **作者:** Wentang Song; Zhiyuan Yan; Yuzhen Lin; Taiping Yao; Changsheng Chen; Shen Chen; Yandan Zhao; Shouhong Ding; Bin Li
>
> **摘要:** Detecting AI-generated images, particularly deepfakes, has become increasingly crucial, with the primary challenge being the generalization to previously unseen manipulation methods. This paper tackles this issue by leveraging the forgery quality of training data to improve the generalization performance of existing deepfake detectors. Generally, the forgery quality of different deepfakes varies: some have easily recognizable forgery clues, while others are highly realistic. Existing works often train detectors on a mix of deepfakes with varying forgery qualities, potentially leading detectors to short-cut the easy-to-spot artifacts from low-quality forgery samples, thereby hurting generalization performance. To tackle this issue, we propose a novel quality-centric framework for generic deepfake detection, which is composed of a Quality Evaluator, a low-quality data enhancement module, and a learning pacing strategy that explicitly incorporates forgery quality into the training process. Our framework is inspired by curriculum learning, which is designed to gradually enable the detector to learn more challenging deepfake samples, starting with easier samples and progressing to more realistic ones. We employ both static and dynamic assessments to assess the forgery quality, combining their scores to produce a final rating for each training sample. The rating score guides the selection of deepfake samples for training, with higher-rated samples having a higher probability of being chosen. Furthermore, we propose a novel frequency data augmentation method specifically designed for low-quality forgery samples, which helps to reduce obvious forgery traces and improve their overall realism. Extensive experiments demonstrate that our proposed framework can be applied plug-and-play to existing detection models and significantly enhance their generalization performance in detection.
>
---
#### [replaced 077] High Accuracy Pulmonary Vessel Segmentation for Contrast and Non-contrast CT Images and Clinical Evaluation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.16988v2](http://arxiv.org/pdf/2503.16988v2)**

> **作者:** Ying Ming; Shaoze Luo; Longfei Zhao; Ruijie Zhao; Bing Li; Qiqi Xu; Wei Song
>
> **备注:** Visual clinical evaluation results were added in v2 Comparison with the latest techniques were added Authors were updated
>
> **摘要:** Accurate segmentation of pulmonary vessels plays a very critical role in diagnosing and assessing various lung diseases. Currently, many automated algorithms are primarily targeted at CTPA (Computed Tomography Pulmonary Angiography) types of data. However, the segmentation precision of these methods is insufficient, and support for NCCT (Non-Contrast Computed Tomography) types of data is also a requirement in some clinical scenarios. In this study, we propose a 3D image segmentation algorithm for automated pulmonary vessel segmentation from both contrast-enhanced and non-contrast CT images. In the network, we designed a Vessel Lumen Structure Optimization Module (VLSOM), which extracts the centerline (Cl) of vessels and adjusts the weights based on the positional information and adds a Cl-Dice Loss to supervise the stability of the vessels structure. We used 427 sets of high-precision annotated CT data from multiple vendors and countries to train the model and achieved Cl-DICE, Cl-Recall, and Recall values of 0.892, 0.861, 0.924 for CTPA data and 0.925, 0.903, 0.949 for NCCT data. This shows that our model has achieved good performance in both accuracy and completeness of pulmonary vessel segmentation. We finally conducted a clinical visual assessment on an independent external test dataset. The average score for accuracy and robustness, branch abundance, assistance for diagnosis and vascular continuity are 4.26, 4.17, 4.33, 3.83 respectively while the full score is 5. These results highlight the great potential of this method in clinical application.
>
---
#### [replaced 078] Bootstraping Clustering of Gaussians for View-consistent 3D Scene Understanding
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19551v2](http://arxiv.org/pdf/2411.19551v2)**

> **作者:** Wenbo Zhang; Lu Zhang; Ping Hu; Liqian Ma; Yunzhi Zhuge; Huchuan Lu
>
> **备注:** Accepted to AAAI25
>
> **摘要:** Injecting semantics into 3D Gaussian Splatting (3DGS) has recently garnered significant attention. While current approaches typically distill 3D semantic features from 2D foundational models (e.g., CLIP and SAM) to facilitate novel view segmentation and semantic understanding, their heavy reliance on 2D supervision can undermine cross-view semantic consistency and necessitate complex data preparation processes, therefore hindering view-consistent scene understanding. In this work, we present FreeGS, an unsupervised semantic-embedded 3DGS framework that achieves view-consistent 3D scene understanding without the need for 2D labels. Instead of directly learning semantic features, we introduce the IDentity-coupled Semantic Field (IDSF) into 3DGS, which captures both semantic representations and view-consistent instance indices for each Gaussian. We optimize IDSF with a two-step alternating strategy: semantics help to extract coherent instances in 3D space, while the resulting instances regularize the injection of stable semantics from 2D space. Additionally, we adopt a 2D-3D joint contrastive loss to enhance the complementarity between view-consistent 3D geometry and rich semantics during the bootstrapping process, enabling FreeGS to uniformly perform tasks such as novel-view semantic segmentation, object selection, and 3D object detection. Extensive experiments on LERF-Mask, 3D-OVS, and ScanNet datasets demonstrate that FreeGS performs comparably to state-of-the-art methods while avoiding the complex data preprocessing workload. Our code is publicly available at https://github.com/wb014/FreeGS.
>
---
#### [replaced 079] Cognitive Disentanglement for Referring Multi-Object Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11496v3](http://arxiv.org/pdf/2503.11496v3)**

> **作者:** Shaofeng Liang; Runwei Guan; Wangwang Lian; Daizong Liu; Xiaolou Sun; Dongming Wu; Yutao Yue; Weiping Ding; Hui Xiong
>
> **备注:** 27 pages, 12 figures
>
> **摘要:** As a significant application of multi-source information fusion in intelligent transportation perception systems, Referring Multi-Object Tracking (RMOT) involves localizing and tracking specific objects in video sequences based on language references. However, existing RMOT approaches often treat language descriptions as holistic embeddings and struggle to effectively integrate the rich semantic information contained in language expressions with visual features. This limitation is especially apparent in complex scenes requiring comprehensive understanding of both static object attributes and spatial motion information. In this paper, we propose a Cognitive Disentanglement for Referring Multi-Object Tracking (CDRMT) framework that addresses these challenges. It adapts the "what" and "where" pathways from the human visual processing system to RMOT tasks. Specifically, our framework first establishes cross-modal connections while preserving modality-specific characteristics. It then disentangles language descriptions and hierarchically injects them into object queries, refining object understanding from coarse to fine-grained semantic levels. Finally, we reconstruct language representations based on visual features, ensuring that tracked objects faithfully reflect the referring expression. Extensive experiments on different benchmark datasets demonstrate that CDRMT achieves substantial improvements over state-of-the-art methods, with average gains of 6.0% in HOTA score on Refer-KITTI and 3.2% on Refer-KITTI-V2. Our approach advances the state-of-the-art in RMOT while simultaneously providing new insights into multi-source information fusion.
>
---
#### [replaced 080] Spatial Re-parameterization for N:M Sparsity
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2306.05612v3](http://arxiv.org/pdf/2306.05612v3)**

> **作者:** Yuxin Zhang; Mingbao Lin; Mingliang Xu; Yonghong Tian; Rongrong Ji
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** This paper presents a Spatial Re-parameterization (SpRe) method for the N:M sparsity. SpRe stems from an observation regarding the restricted variety in spatial sparsity of convolution kernels presented in N:M sparsity compared with unstructured sparsity. Particularly, N:M sparsity exhibits a fixed sparsity rate within the spatial domains due to its distinctive pattern that mandates N non-zero components among M successive weights in the input channel dimension of convolution filters. On the contrary, we observe that conventional unstructured sparsity displays a substantial divergence in sparsity across the spatial domains, which we experimentally verify to be very crucial for its robust performance retention compared with N:M sparsity. Therefore, SpRe employs the spatial-sparsity distribution of unstructured sparsity by assigning an extra branch in conjunction with the original N:M branch at training time, which allows the N:M sparse network to sustain a similar distribution of spatial sparsity with unstructured sparsity. During inference, the extra branch can be further re-parameterized into the main N:M branch, without exerting any distortion on the sparse pattern or additional computation costs. SpRe has achieved a commendable feat by matching the performance of N:M sparsity methods with state-of-the-art unstructured sparsity methods across various benchmarks. Our project is available at https://github.com/zyxxmu/SpRE.
>
---
#### [replaced 081] Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation
- **分类: cs.LG; cs.AI; cs.CV; cs.IR**

- **链接: [http://arxiv.org/pdf/2503.01776v4](http://arxiv.org/pdf/2503.01776v4)**

> **作者:** Tiansheng Wen; Yifei Wang; Zequn Zeng; Zhong Peng; Yudi Su; Xinyang Liu; Bo Chen; Hongwei Liu; Stefanie Jegelka; Chenyu You
>
> **备注:** Accepted by ICML2025
>
> **摘要:** Many large-scale systems rely on high-quality deep representations (embeddings) to facilitate tasks like retrieval, search, and generative modeling. Matryoshka Representation Learning (MRL) recently emerged as a solution for adaptive embedding lengths, but it requires full model retraining and suffers from noticeable performance degradations at short lengths. In this paper, we show that sparse coding offers a compelling alternative for achieving adaptive representation with minimal overhead and higher fidelity. We propose Contrastive Sparse Representation (CSR), a method that sparsifies pre-trained embeddings into a high-dimensional but selectively activated feature space. By leveraging lightweight autoencoding and task-aware contrastive objectives, CSR preserves semantic quality while allowing flexible, cost-effective inference at different sparsity levels. Extensive experiments on image, text, and multimodal benchmarks demonstrate that CSR consistently outperforms MRL in terms of both accuracy and retrieval speed-often by large margins-while also cutting training time to a fraction of that required by MRL. Our results establish sparse coding as a powerful paradigm for adaptive representation learning in real-world applications where efficiency and fidelity are both paramount. Code is available at https://github.com/neilwen987/CSR_Adaptive_Rep
>
---
#### [replaced 082] Securing Visually-Aware Recommender Systems: An Adversarial Image Reconstruction and Detection Framework
- **分类: cs.CV; cs.AI; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2306.07992v2](http://arxiv.org/pdf/2306.07992v2)**

> **作者:** Minglei Yin; Bin Liu; Neil Zhenqiang Gong; Xin Li
>
> **摘要:** With rich visual data, such as images, becoming readily associated with items, visually-aware recommendation systems (VARS) have been widely used in different applications. Recent studies have shown that VARS are vulnerable to item-image adversarial attacks, which add human-imperceptible perturbations to the clean images associated with those items. Attacks on VARS pose new security challenges to a wide range of applications such as e-Commerce and social networks where VARS are widely used. How to secure VARS from such adversarial attacks becomes a critical problem. Currently, there is still a lack of systematic study on how to design secure defense strategies against visual attacks on VARS. In this paper, we attempt to fill this gap by proposing an adversarial image reconstruction and detection framework to secure VARS. Our proposed method can simultaneously (1) secure VARS from adversarial attacks characterized by local perturbations by image reconstruction based on global vision transformers; and (2) accurately detect adversarial examples using a novel contrastive learning approach. Meanwhile, our framework is designed to be used as both a filter and a detector so that they can be jointly trained to improve the flexibility of our defense strategy to a variety of attacks and VARS models. We have conducted extensive experimental studies with two popular attack methods (FGSM and PGD). Our experimental results on two real-world datasets show that our defense strategy against visual attacks is effective and outperforms existing methods on different attacks. Moreover, our method can detect adversarial examples with high accuracy.
>
---
#### [replaced 083] A Deeper Look into Second-Order Feature Aggregation for LiDAR Place Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.15919v2](http://arxiv.org/pdf/2409.15919v2)**

> **作者:** Saimunur Rahman; Peyman Moghadam
>
> **备注:** Submitted to CoRL 2025
>
> **摘要:** Efficient LiDAR Place Recognition (LPR) compresses dense pointwise features into compact global descriptors. While first-order aggregators such as GeM and NetVLAD are widely used, they overlook inter-feature correlations that second-order aggregation naturally captures. Full covariance, a common second-order aggregator, is high in dimensionality; as a result, practitioners often insert a learned projection or employ random sketches -- both of which either sacrifice information or increase parameter count. However, no prior work has systematically investigated how first- and second-order aggregation perform under constrained feature and compute budgets. In this paper, we first demonstrate that second-order aggregation retains its superiority for LPR even when channels are pruned and backbone parameters are reduced. Building on this insight, we propose Channel Partition-based Second-order Local Feature Aggregation (CPS): a drop-in, partition-based second-order aggregation module that preserves all channels while producing an order-of-magnitude smaller descriptor. CPS matches or exceeds the performance of full covariance and outperforms random projection variants, delivering new state-of-the-art results with only four additional learnable parameters across four large-scale benchmarks: Oxford RobotCar, In-house, MulRan, and WildPlaces.
>
---
#### [replaced 084] MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.06141v2](http://arxiv.org/pdf/2412.06141v2)**

> **作者:** Kangyu Zhu; Peng Xia; Yun Li; Hongtu Zhu; Sheng Wang; Huaxiu Yao
>
> **备注:** ICML 2025
>
> **摘要:** The advancement of Large Vision-Language Models (LVLMs) has propelled their application in the medical field. However, Medical LVLMs (Med-LVLMs) encounter factuality challenges due to modality misalignment, where the models prioritize textual knowledge over visual input, leading to hallucinations that contradict information in medical images. Previous attempts to enhance modality alignment in Med-LVLMs through preference optimization have inadequately mitigated clinical relevance in preference data, making these samples easily distinguishable and reducing alignment effectiveness. To address this challenge, we propose MMedPO, a novel multimodal medical preference optimization approach that considers the clinical relevance of preference samples to enhance Med-LVLM alignment. MMedPO curates multimodal preference data by introducing two types of dispreference: (1) plausible hallucinations injected through target Med-LVLMs or GPT-4o to produce medically inaccurate responses, and (2) lesion region neglect achieved through local lesion-noising, disrupting visual understanding of critical areas. We then calculate clinical relevance for each sample based on scores from multiple Med-LLMs and visual tools, and integrate these scores into the preference optimization process as weights, enabling effective alignment. Our experiments demonstrate that MMedPO significantly enhances factual accuracy in Med-LVLMs, achieving substantial improvements over existing preference optimization methods by averaging 14.2% and 51.7% across the Med-VQA and report generation tasks. Our code are available in https://github.com/aiming-lab/MMedPO.
>
---
#### [replaced 085] DiffusionAD: Norm-guided One-step Denoising Diffusion for Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2303.08730v4](http://arxiv.org/pdf/2303.08730v4)**

> **作者:** Hui Zhang; Zheng Wang; Dan Zeng; Zuxuan Wu; Yu-Gang Jiang
>
> **备注:** Accepted by TPAMI
>
> **摘要:** Anomaly detection has garnered extensive applications in real industrial manufacturing due to its remarkable effectiveness and efficiency. However, previous generative-based models have been limited by suboptimal reconstruction quality, hampering their overall performance. We introduce DiffusionAD, a novel anomaly detection pipeline comprising a reconstruction sub-network and a segmentation sub-network. A fundamental enhancement lies in our reformulation of the reconstruction process using a diffusion model into a noise-to-norm paradigm. Here, the anomalous region loses its distinctive features after being disturbed by Gaussian noise and is subsequently reconstructed into an anomaly-free one. Afterward, the segmentation sub-network predicts pixel-level anomaly scores based on the similarities and discrepancies between the input image and its anomaly-free reconstruction. Additionally, given the substantial decrease in inference speed due to the iterative denoising nature of diffusion models, we revisit the denoising process and introduce a rapid one-step denoising paradigm. This paradigm achieves hundreds of times acceleration while preserving comparable reconstruction quality. Furthermore, considering the diversity in the manifestation of anomalies, we propose a norm-guided paradigm to integrate the benefits of multiple noise scales, enhancing the fidelity of reconstructions. Comprehensive evaluations on four standard and challenging benchmarks reveal that DiffusionAD outperforms current state-of-the-art approaches and achieves comparable inference speed, demonstrating the effectiveness and broad applicability of the proposed pipeline. Code is released at https://github.com/HuiZhang0812/DiffusionAD
>
---
#### [replaced 086] Enhancing Multimodal Unified Representations for Cross Modal Generalization
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.05168v2](http://arxiv.org/pdf/2403.05168v2)**

> **作者:** Hai Huang; Yan Xia; Shengpeng Ji; Shulei Wang; Hanting Wang; Minghui Fang; Jieming Zhu; Zhenhua Dong; Sashuai Zhou; Zhou Zhao
>
> **摘要:** To enhance the interpretability of multimodal unified representations, many studies have focused on discrete unified representations. These efforts typically start with contrastive learning and gradually extend to the disentanglement of modal information, achieving solid multimodal discrete unified representations. However, existing research often overlooks two critical issues: 1) The use of Euclidean distance for quantization in discrete representations often overlooks the important distinctions among different dimensions of features, resulting in redundant representations after quantization; 2) Different modalities have unique characteristics, and a uniform alignment approach does not fully exploit these traits. To address these issues, we propose Training-free Optimization of Codebook (TOC) and Fine and Coarse cross-modal Information Disentangling (FCID). These methods refine the unified discrete representations from pretraining and perform fine- and coarse-grained information disentanglement tailored to the specific characteristics of each modality, achieving significant performance improvements over previous state-of-the-art models.
>
---
#### [replaced 087] PAHA: Parts-Aware Audio-Driven Human Animation with Diffusion Model
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.03603v4](http://arxiv.org/pdf/2505.03603v4)**

> **作者:** S. Z. Zhou; Y. B. Wang; J. F. Wu; T. Hu; J. N. Zhang; Z. J. Li; Y. Liu
>
> **摘要:** Audio-driven human animation technology is widely used in human-computer interaction, and the emergence of diffusion models has further advanced its development. Currently, most methods rely on multi-stage generation and intermediate representations, resulting in long inference time and issues with generation quality in specific foreground regions and audio-motion consistency. These shortcomings are primarily due to the lack of localized fine-grained supervised guidance. To address above challenges, we propose PAHA, an end-to-end audio-driven upper-body human animation framework with diffusion model. We introduce two key methods: Parts-Aware Re-weighting (PAR) and Parts Consistency Enhancement (PCE). PAR dynamically adjusts regional training loss weights based on pose confidence scores, effectively improving visual quality. PCE constructs and trains diffusion-based regional audio-visual classifiers to improve the consistency of motion and co-speech audio. Afterwards, we design two novel inference guidance methods for the foregoing classifiers, Sequential Guidance (SG) and Differential Guidance (DG), to balance efficiency and quality respectively. Additionally, we build CNAS, the first public Chinese News Anchor Speech dataset, to advance research and validation in this field. Extensive experimental results and user studies demonstrate that PAHA significantly outperforms existing methods in audio-motion alignment and video-related evaluations. The codes and CNAS dataset will be released upon acceptance.
>
---
#### [replaced 088] Incremental Multi-Scene Modeling via Continual Neural Graphics Primitives
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19903v3](http://arxiv.org/pdf/2411.19903v3)**

> **作者:** Prajwal Singh; Ashish Tiwari; Gautam Vashishtha; Shanmuganathan Raman
>
> **摘要:** Neural radiance fields (NeRF) have revolutionized photorealistic rendering of novel views for 3D scenes. Despite their growing popularity and efficiency as 3D resources, NeRFs face scalability challenges due to the need for separate models per scene and the cumulative increase in training time for multiple scenes. The potential for incrementally encoding multiple 3D scenes into a single NeRF model remains largely unexplored. To address this, we introduce Continual-Neural Graphics Primitives (C-NGP), a novel continual learning framework that integrates multiple scenes incrementally into a single neural radiance field. Using a generative replay approach, C-NGP adapts to new scenes without requiring access to old data. We demonstrate that C-NGP can accommodate multiple scenes without increasing the parameter count, producing high-quality novel-view renderings on synthetic and real datasets. Notably, C-NGP models all $8$ scenes from the Real-LLFF dataset together, with only a $2.2\%$ drop in PSNR compared to vanilla NeRF, which models each scene independently. Further, C-NGP allows multiple style edits in the same network.
>
---
#### [replaced 089] DexGarmentLab: Dexterous Garment Manipulation Environment with Generalizable Policy
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11032v2](http://arxiv.org/pdf/2505.11032v2)**

> **作者:** Yuran Wang; Ruihai Wu; Yue Chen; Jiarui Wang; Jiaqi Liang; Ziyu Zhu; Haoran Geng; Jitendra Malik; Pieter Abbeel; Hao Dong
>
> **摘要:** Garment manipulation is a critical challenge due to the diversity in garment categories, geometries, and deformations. Despite this, humans can effortlessly handle garments, thanks to the dexterity of our hands. However, existing research in the field has struggled to replicate this level of dexterity, primarily hindered by the lack of realistic simulations of dexterous garment manipulation. Therefore, we propose DexGarmentLab, the first environment specifically designed for dexterous (especially bimanual) garment manipulation, which features large-scale high-quality 3D assets for 15 task scenarios, and refines simulation techniques tailored for garment modeling to reduce the sim-to-real gap. Previous data collection typically relies on teleoperation or training expert reinforcement learning (RL) policies, which are labor-intensive and inefficient. In this paper, we leverage garment structural correspondence to automatically generate a dataset with diverse trajectories using only a single expert demonstration, significantly reducing manual intervention. However, even extensive demonstrations cannot cover the infinite states of garments, which necessitates the exploration of new algorithms. To improve generalization across diverse garment shapes and deformations, we propose a Hierarchical gArment-manipuLation pOlicy (HALO). It first identifies transferable affordance points to accurately locate the manipulation area, then generates generalizable trajectories to complete the task. Through extensive experiments and detailed analysis of our method and baseline, we demonstrate that HALO consistently outperforms existing methods, successfully generalizing to previously unseen instances even with significant variations in shape and deformation where others fail. Our project page is available at: https://wayrise.github.io/DexGarmentLab/.
>
---
#### [replaced 090] A Preliminary Study for GPT-4o on Image Restoration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05621v2](http://arxiv.org/pdf/2505.05621v2)**

> **作者:** Hao Yang; Yan Yang; Ruikun Zhang; Liyuan Pan
>
> **摘要:** OpenAI's GPT-4o model, integrating multi-modal inputs and outputs within an autoregressive architecture, has demonstrated unprecedented performance in image generation. In this work, we investigate its potential impact on the image restoration community. We present the first systematic evaluation of GPT-4o across diverse restoration tasks. Our experiments reveal that, although restoration outputs from GPT-4o are visually appealing, they often suffer from pixel-level structural fidelity when compared to ground-truth images. Common issues are variations in image proportions, shifts in object positions and quantities, and changes in viewpoint. To address it, taking image dehazing, derainning, and low-light enhancement as representative case studies, we show that GPT-4o's outputs can serve as powerful visual priors, substantially enhancing the performance of existing dehazing networks. It offers practical guidelines and a baseline framework to facilitate the integration of GPT-4o into future image restoration pipelines. We hope the study on GPT-4o image restoration will accelerate innovation in the broader field of image generation areas. To support further research, we will release GPT-4o-restored images.
>
---
#### [replaced 091] Dual-level Fuzzy Learning with Patch Guidance for Image Ordinal Regression
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05834v2](http://arxiv.org/pdf/2505.05834v2)**

> **作者:** Chunlai Dong; Haochao Ying; Qibo Qiu; Jinhong Wang; Danny Chen; Jian Wu
>
> **备注:** Accepted by IJCAI 2025
>
> **摘要:** Ordinal regression bridges regression and classification by assigning objects to ordered classes. While human experts rely on discriminative patch-level features for decisions, current approaches are limited by the availability of only image-level ordinal labels, overlooking fine-grained patch-level characteristics. In this paper, we propose a Dual-level Fuzzy Learning with Patch Guidance framework, named DFPG that learns precise feature-based grading boundaries from ambiguous ordinal labels, with patch-level supervision. Specifically, we propose patch-labeling and filtering strategies to enable the model to focus on patch-level features exclusively with only image-level ordinal labels available. We further design a dual-level fuzzy learning module, which leverages fuzzy logic to quantitatively capture and handle label ambiguity from both patch-wise and channel-wise perspectives. Extensive experiments on various image ordinal regression datasets demonstrate the superiority of our proposed method, further confirming its ability in distinguishing samples from difficult-to-classify categories. The code is available at https://github.com/ZJUMAI/DFPG-ord.
>
---
#### [replaced 092] Complementary Frequency-Varying Awareness Network for Open-Set Fine-Grained Image Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2307.07214v2](http://arxiv.org/pdf/2307.07214v2)**

> **作者:** Qiulei Dong; Jiayin Sun; Mengyu Gao
>
> **摘要:** Open-set image recognition is a challenging topic in computer vision. Most of the existing works in literature focus on learning more discriminative features from the input images, however, they are usually insensitive to the high- or low-frequency components in features, resulting in a decreasing performance on fine-grained image recognition. To address this problem, we propose a Complementary Frequency-varying Awareness Network that could better capture both high-frequency and low-frequency information, called CFAN. The proposed CFAN consists of three sequential modules: (i) a feature extraction module is introduced for learning preliminary features from the input images; (ii) a frequency-varying filtering module is designed to separate out both high- and low-frequency components from the preliminary features in the frequency domain via a frequency-adjustable filter; (iii) a complementary temporal aggregation module is designed for aggregating the high- and low-frequency components via two Long Short-Term Memory networks into discriminative features. Based on CFAN, we further propose an open-set fine-grained image recognition method, called CFAN-OSFGR, which learns image features via CFAN and classifies them via a linear classifier. Experimental results on 3 fine-grained datasets and 2 coarse-grained datasets demonstrate that CFAN-OSFGR performs significantly better than 9 state-of-the-art methods in most cases.
>
---
#### [replaced 093] Towards Enhanced Image Inpainting: Mitigating Unwanted Object Insertion and Preserving Color Consistency
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.04831v3](http://arxiv.org/pdf/2312.04831v3)**

> **作者:** Yikai Wang; Chenjie Cao; Junqiu Yu; Ke Fan; Xiangyang Xue; Yanwei Fu
>
> **备注:** CVPR 2025 Highlight. Project page: https://yikai-wang.github.io/asuka/ where full-size PDF with more qualitative results are available
>
> **摘要:** Recent advances in image inpainting increasingly use generative models to handle large irregular masks. However, these models can create unrealistic inpainted images due to two main issues: (1) Unwanted object insertion: Even with unmasked areas as context, generative models may still generate arbitrary objects in the masked region that don't align with the rest of the image. (2) Color inconsistency: Inpainted regions often have color shifts that causes a smeared appearance, reducing image quality. Retraining the generative model could help solve these issues, but it's costly since state-of-the-art latent-based diffusion and rectified flow models require a three-stage training process: training a VAE, training a generative U-Net or transformer, and fine-tuning for inpainting. Instead, this paper proposes a post-processing approach, dubbed as ASUKA (Aligned Stable inpainting with UnKnown Areas prior), to improve inpainting models. To address unwanted object insertion, we leverage a Masked Auto-Encoder (MAE) for reconstruction-based priors. This mitigates object hallucination while maintaining the model's generation capabilities. To address color inconsistency, we propose a specialized VAE decoder that treats latent-to-image decoding as a local harmonization task, significantly reducing color shifts for color-consistent inpainting. We validate ASUKA on SD 1.5 and FLUX inpainting variants with Places2 and MISATO, our proposed diverse collection of datasets. Results show that ASUKA mitigates object hallucination and improves color consistency over standard diffusion and rectified flow models and other inpainting methods.
>
---
#### [replaced 094] CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18430v3](http://arxiv.org/pdf/2503.18430v3)**

> **作者:** Zhichao Sun; Huazhang Hu; Yidong Ma; Gang Liu; Nemo Chen; Xu Tang; Yao Hu; Yongchao Xu
>
> **摘要:** With the exponential growth of data, traditional object detection methods are increasingly struggling to handle vast vocabulary object detection tasks effectively. We analyze two key limitations of classification-based detectors: positive gradient dilution, where rare positive categories receive insufficient learning signals, and hard negative gradient dilution, where discriminative gradients are overwhelmed by numerous easy negatives. To address these challenges, we propose CQ-DINO, a category query-based object detection framework that reformulates classification as a contrastive task between object queries and learnable category queries. Our method introduces image-guided query selection, which reduces the negative space by adaptively retrieving top-K relevant categories per image via cross-attention, thereby rebalancing gradient distributions and facilitating implicit hard example mining. Furthermore, CQ-DINO flexibly integrates explicit hierarchical category relationships in structured datasets (e.g., V3Det) or learns implicit category correlations via self-attention in generic datasets (e.g., COCO). Experiments demonstrate that CQ-DINO achieves superior performance on the challenging V3Det benchmark (surpassing previous methods by 2.1% AP) while maintaining competitiveness in COCO. Our work provides a scalable solution for real-world detection systems requiring wide category coverage. The code is publicly at https://github.com/RedAIGC/CQ-DINO.
>
---
#### [replaced 095] Redefining non-IID Data in Federated Learning for Computer Vision Tasks: Migrating from Labels to Embeddings for Task-Specific Data Distributions
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.14553v3](http://arxiv.org/pdf/2503.14553v3)**

> **作者:** Kasra Borazjani; Payam Abdisarabshali; Naji Khosravan; Seyyedali Hosseinalipour
>
> **备注:** 14 pages, 9 figures, 1 table, (implementations are included at our GitHub repository: https://github.com/KasraBorazjani/task-perspective-het)
>
> **摘要:** Federated Learning (FL) represents a paradigm shift in distributed machine learning (ML), enabling clients to train models collaboratively while keeping their raw data private. This paradigm shift from traditional centralized ML introduces challenges due to the non-iid (non-independent and identically distributed) nature of data across clients, significantly impacting FL's performance. Existing literature, predominantly model data heterogeneity by imposing label distribution skew across clients. In this paper, we show that label distribution skew fails to fully capture the real-world data heterogeneity among clients in computer vision tasks beyond classification. Subsequently, we demonstrate that current approaches overestimate FL's performance by relying on label/class distribution skew, exposing an overlooked gap in the literature. By utilizing pre-trained deep neural networks to extract task-specific data embeddings, we define task-specific data heterogeneity through the lens of each vision task and introduce a new level of data heterogeneity called embedding-based data heterogeneity. Our methodology involves clustering data points based on embeddings and distributing them among clients using the Dirichlet distribution. Through extensive experiments, we evaluate the performance of different FL methods under our revamped notion of data heterogeneity, introducing new benchmark performance measures to the literature. We further unveil a series of open research directions that can be pursued.
>
---
#### [replaced 096] Adversarial Attacks of Vision Tasks in the Past 10 Years: A Survey
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2410.23687v2](http://arxiv.org/pdf/2410.23687v2)**

> **作者:** Chiyu Zhang; Lu Zhou; Xiaogang Xu; Jiafei Wu; Zhe Liu
>
> **摘要:** With the advent of Large Vision-Language Models (LVLMs), new attack vectors, such as cognitive bias, prompt injection, and jailbreaking, have emerged. Understanding these attacks promotes system robustness improvement and neural networks demystification. However, existing surveys often target attack taxonomy and lack in-depth analysis like 1) unified insights into adversariality, transferability, and generalization; 2) detailed evaluations framework; 3) motivation-driven attack categorizations; and 4) an integrated perspective on both traditional and LVLM attacks. This article addresses these gaps by offering a thorough summary of traditional and LVLM adversarial attacks, emphasizing their connections and distinctions, and providing actionable insights for future research.
>
---
#### [replaced 097] EndoMetric: Near-Light Monocular Metric Scale Estimation in Endoscopy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.15065v2](http://arxiv.org/pdf/2410.15065v2)**

> **作者:** Raúl Iranzo; Víctor M. Batlle; Juan D. Tardós; José M. M. Montiel
>
> **备注:** 10 pages, 3 figures, to be published in MICCAI 2025
>
> **摘要:** Geometric reconstruction and SLAM with endoscopic images have advanced significantly in recent years. In most medical fields, monocular endoscopes are employed, and the algorithms used are typically adaptations of those designed for external environments, resulting in 3D reconstructions with an unknown scale factor. For the first time, we propose a method to estimate the real metric scale of a 3D reconstruction from standard monocular endoscopic images without relying on application-specific learned priors. Our fully model-based approach leverages the near-light sources embedded in endoscopes, positioned at a small but nonzero baseline from the camera, in combination with the inverse-square law of light attenuation, to accurately recover the metric scale from scratch. This enables the transformation of any endoscope into a metric device, which is crucial for applications such as measuring polyps, stenosis, or assessing the extent of diseased tissue.
>
---
#### [replaced 098] FAST: Federated Active Learning with Foundation Models for Communication-efficient Sampling and Training
- **分类: cs.LG; cs.AI; cs.CV; cs.DC**

- **链接: [http://arxiv.org/pdf/2504.03783v4](http://arxiv.org/pdf/2504.03783v4)**

> **作者:** Haoyuan Li; Mathias Funk; Jindong Wang; Aaqib Saeed
>
> **备注:** Accepted at IEEE Internet of Things Journal
>
> **摘要:** Federated Active Learning (FAL) has emerged as a promising framework to leverage large quantities of unlabeled data across distributed clients while preserving data privacy. However, real-world deployments remain limited by high annotation costs and communication-intensive sampling processes, particularly in a cross-silo setting, when clients possess substantial local datasets. This paper addresses the crucial question: What is the best practice to reduce communication costs in human-in-the-loop learning with minimal annotator effort? Existing FAL methods typically rely on iterative annotation processes that separate active sampling from federated updates, leading to multiple rounds of expensive communication and annotation. In response, we introduce FAST, a two-pass FAL framework that harnesses foundation models for weak labeling in a preliminary pass, followed by a refinement pass focused exclusively on the most uncertain samples. By leveraging representation knowledge from foundation models and integrating refinement steps into a streamlined workflow, FAST substantially reduces the overhead incurred by iterative active sampling. Extensive experiments on diverse medical and natural image benchmarks demonstrate that FAST outperforms existing FAL methods by an average of 4.36% while reducing communication rounds eightfold under a limited 5% labeling budget.
>
---
#### [replaced 099] Self-Supervised Learning for Robotic Leaf Manipulation: A Hybrid Geometric-Neural Approach
- **分类: cs.RO; cs.CV; cs.LG; I.2.10**

- **链接: [http://arxiv.org/pdf/2505.03702v3](http://arxiv.org/pdf/2505.03702v3)**

> **作者:** Srecharan Selvam
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Automating leaf manipulation in agricultural settings faces significant challenges, including the variability of plant morphologies and deformable leaves. We propose a novel hybrid geometric-neural approach for autonomous leaf grasping that combines traditional computer vision with neural networks through self-supervised learning. Our method integrates YOLOv8 for instance segmentation and RAFT-Stereo for 3D depth estimation to build rich leaf representations, which feed into both a geometric feature scoring pipeline and a neural refinement module (GraspPointCNN). The key innovation is our confidence-weighted fusion mechanism that dynamically balances the contribution of each approach based on prediction certainty. Our self-supervised framework uses the geometric pipeline as an expert teacher to automatically generate training data. Experiments demonstrate that our approach achieves an 88.0% success rate in controlled environments and 84.7% in real greenhouse conditions, significantly outperforming both purely geometric (75.3%) and neural (60.2%) methods. This work establishes a new paradigm for agricultural robotics where domain expertise is seamlessly integrated with machine learning capabilities, providing a foundation for fully automated crop monitoring systems.
>
---
#### [replaced 100] Underwater Camouflaged Object Tracking Meets Vision-Language SAM2
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.16902v5](http://arxiv.org/pdf/2409.16902v5)**

> **作者:** Chunhui Zhang; Li Liu; Guanjie Huang; Zhipeng Zhang; Hao Wen; Xi Zhou; Shiming Ge; Yanfeng Wang
>
> **备注:** Accepted to CVPR 2025 Workshop on CV4Animals. https://github.com/983632847/Awesome-Multimodal-Object-Tracking
>
> **摘要:** Over the past decade, significant progress has been made in visual object tracking, largely due to the availability of large-scale datasets. However, these datasets have primarily focused on open-air scenarios and have largely overlooked underwater animal tracking-especially the complex challenges posed by camouflaged marine animals. To bridge this gap, we take a step forward by proposing the first large-scale multi-modal underwater camouflaged object tracking dataset, namely UW-COT220. Based on the proposed dataset, this work first comprehensively evaluates current advanced visual object tracking methods, including SAM- and SAM2-based trackers, in challenging underwater environments, \eg, coral reefs. Our findings highlight the improvements of SAM2 over SAM, demonstrating its enhanced ability to handle the complexities of underwater camouflaged objects. Furthermore, we propose a novel vision-language tracking framework called VL-SAM2, based on the video foundation model SAM2. Extensive experimental results demonstrate that the proposed VL-SAM2 achieves state-of-the-art performance across underwater and open-air object tracking datasets. The dataset and codes are available at~{\color{magenta}{https://github.com/983632847/Awesome-Multimodal-Object-Tracking}}.
>
---
#### [replaced 101] Adaptive Extrapolated Proximal Gradient Methods with Variance Reduction for Composite Nonconvex Finite-Sum Minimization
- **分类: math.OC; cs.CV; cs.NA; math.NA**

- **链接: [http://arxiv.org/pdf/2502.21099v2](http://arxiv.org/pdf/2502.21099v2)**

> **作者:** Ganzhao Yuan
>
> **摘要:** This paper proposes {\sf AEPG-SPIDER}, an Adaptive Extrapolated Proximal Gradient (AEPG) method with variance reduction for minimizing composite nonconvex finite-sum functions. It integrates three acceleration techniques: adaptive stepsizes, Nesterov's extrapolation, and the recursive stochastic path-integrated estimator SPIDER. Unlike existing methods that adjust the stepsize factor using historical gradients, {\sf AEPG-SPIDER} relies on past iterate differences for its update. While targeting stochastic finite-sum problems, {\sf AEPG-SPIDER} simplifies to {\sf AEPG} in the full-batch, non-stochastic setting, which is also of independent interest. To our knowledge, {\sf AEPG-SPIDER} and {\sf AEPG} are the first Lipschitz-free methods to achieve optimal iteration complexity for this class of \textit{composite} minimization problems. Specifically, {\sf AEPG} achieves the optimal iteration complexity of $\mathcal{O}(N \epsilon^{-2})$, while {\sf AEPG-SPIDER} achieves $\mathcal{O}(N + \sqrt{N} \epsilon^{-2})$ for finding $\epsilon$-approximate stationary points, where $N$ is the number of component functions. Under the Kurdyka-Lojasiewicz (KL) assumption, we establish non-ergodic convergence rates for both methods. Preliminary experiments on sparse phase retrieval and linear eigenvalue problems demonstrate the superior performance of {\sf AEPG-SPIDER} and {\sf AEPG} compared to existing methods.
>
---
#### [replaced 102] Differentially Private Synthetic Data via Foundation Model APIs 1: Images
- **分类: cs.CV; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2305.15560v4](http://arxiv.org/pdf/2305.15560v4)**

> **作者:** Zinan Lin; Sivakanth Gopi; Janardhan Kulkarni; Harsha Nori; Sergey Yekhanin
>
> **备注:** Published in ICLR 2024
>
> **摘要:** Generating differentially private (DP) synthetic data that closely resembles the original private data is a scalable way to mitigate privacy concerns in the current data-driven world. In contrast to current practices that train customized models for this task, we aim to generate DP Synthetic Data via APIs (DPSDA), where we treat foundation models as blackboxes and only utilize their inference APIs. Such API-based, training-free approaches are easier to deploy as exemplified by the recent surge in the number of API-based apps. These approaches can also leverage the power of large foundation models which are only accessible via their inference APIs. However, this comes with greater challenges due to strictly more restrictive model access and the need to protect privacy from the API provider. In this paper, we present a new framework called Private Evolution (PE) to solve this problem and show its initial promise on synthetic images. Surprisingly, PE can match or even outperform state-of-the-art (SOTA) methods without any model training. For example, on CIFAR10 (with ImageNet as the public data), we achieve FID <= 7.9 with privacy cost {\epsilon} = 0.67, significantly improving the previous SOTA from {\epsilon} = 32. We further demonstrate the promise of applying PE on large foundation models such as Stable Diffusion to tackle challenging private datasets with a small number of high-resolution images. The code and data are released at https://github.com/microsoft/DPSDA.
>
---
#### [replaced 103] M3G: Multi-Granular Gesture Generator for Audio-Driven Full-Body Human Motion Synthesis
- **分类: cs.GR; cs.AI; cs.CV; cs.SD; eess.AS; I.3.6**

- **链接: [http://arxiv.org/pdf/2505.08293v2](http://arxiv.org/pdf/2505.08293v2)**

> **作者:** Zhizhuo Yin; Yuk Hang Tsui; Pan Hui
>
> **备注:** 9 Pages, 4 figures
>
> **摘要:** Generating full-body human gestures encompassing face, body, hands, and global movements from audio is a valuable yet challenging task in virtual avatar creation. Previous systems focused on tokenizing the human gestures framewisely and predicting the tokens of each frame from the input audio. However, one observation is that the number of frames required for a complete expressive human gesture, defined as granularity, varies among different human gesture patterns. Existing systems fail to model these gesture patterns due to the fixed granularity of their gesture tokens. To solve this problem, we propose a novel framework named Multi-Granular Gesture Generator (M3G) for audio-driven holistic gesture generation. In M3G, we propose a novel Multi-Granular VQ-VAE (MGVQ-VAE) to tokenize motion patterns and reconstruct motion sequences from different temporal granularities. Subsequently, we proposed a multi-granular token predictor that extracts multi-granular information from audio and predicts the corresponding motion tokens. Then M3G reconstructs the human gestures from the predicted tokens using the MGVQ-VAE. Both objective and subjective experiments demonstrate that our proposed M3G framework outperforms the state-of-the-art methods in terms of generating natural and expressive full-body human gestures.
>
---
#### [replaced 104] Explicit and Implicit Representations in AI-based 3D Reconstruction for Radiology: A Systematic Review
- **分类: cs.CV; cs.AI; cs.GR; 68T45; I.4.5**

- **链接: [http://arxiv.org/pdf/2504.11349v2](http://arxiv.org/pdf/2504.11349v2)**

> **作者:** Yuezhe Yang; Boyu Yang; Yaqian Wang; Yang He; Xingbo Dong; Zhe Jin
>
> **备注:** 20 pages, 5 figures, submit to Medical Image Analysis
>
> **摘要:** The demand for high-quality medical imaging in clinical practice and assisted diagnosis has made 3D reconstruction in radiological imaging a key research focus. Artificial intelligence (AI) has emerged as a promising approach to enhancing reconstruction accuracy while reducing acquisition and processing time, thereby minimizing patient radiation exposure and discomfort and ultimately benefiting clinical diagnosis. This review explores state-of-the-art AI-based 3D reconstruction algorithms in radiological imaging, categorizing them into explicit and implicit approaches based on their underlying principles. Explicit methods include point-based, volume-based, and Gaussian representations, while implicit methods encompass implicit prior embedding and neural radiance fields. Additionally, we examine commonly used evaluation metrics and benchmark datasets. Finally, we discuss the current state of development, key challenges, and future research directions in this evolving field. Our project available on: https://github.com/Bean-Young/AI4Radiology.
>
---
#### [replaced 105] Bias and Generalizability of Foundation Models across Datasets in Breast Mammography
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10579v2](http://arxiv.org/pdf/2505.10579v2)**

> **作者:** Elodie Germani; Ilayda Selin Türk; Fatima Zeineddine; Charbel Mourad; Shadi Albarqouni
>
> **备注:** Accepted at the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) 2025
>
> **摘要:** Over the past decades, computer-aided diagnosis tools for breast cancer have been developed to enhance screening procedures, yet their clinical adoption remains challenged by data variability and inherent biases. Although foundation models (FMs) have recently demonstrated impressive generalizability and transfer learning capabilities by leveraging vast and diverse datasets, their performance can be undermined by spurious correlations that arise from variations in image quality, labeling uncertainty, and sensitive patient attributes. In this work, we explore the fairness and bias of FMs for breast mammography classification by leveraging a large pool of datasets from diverse sources-including data from underrepresented regions and an in-house dataset. Our extensive experiments show that while modality-specific pre-training of FMs enhances performance, classifiers trained on features from individual datasets fail to generalize across domains. Aggregating datasets improves overall performance, yet does not fully mitigate biases, leading to significant disparities across under-represented subgroups such as extreme breast densities and age groups. Furthermore, while domain-adaptation strategies can reduce these disparities, they often incur a performance trade-off. In contrast, fairness-aware techniques yield more stable and equitable performance across subgroups. These findings underscore the necessity of incorporating rigorous fairness evaluations and mitigation strategies into FM-based models to foster inclusive and generalizable AI.
>
---
#### [replaced 106] Iterative Deployment Exposure for Unsupervised Out-of-Distribution Detection
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.02327v2](http://arxiv.org/pdf/2406.02327v2)**

> **作者:** Lars Doorenbos; Raphael Sznitman; Pablo Márquez-Neila
>
> **备注:** Accepted at MICCAI 2025
>
> **摘要:** Deep learning models are vulnerable to performance degradation when encountering out-of-distribution (OOD) images, potentially leading to misdiagnoses and compromised patient care. These shortcomings have led to great interest in the field of OOD detection. Existing unsupervised OOD (U-OOD) detection methods typically assume that OOD samples originate from an unconcentrated distribution complementary to the training distribution, neglecting the reality that deployed models passively accumulate task-specific OOD samples over time. To better reflect this real-world scenario, we introduce Iterative Deployment Exposure (IDE), a novel and more realistic setting for U-OOD detection. We propose CSO, a method for IDE that starts from a U-OOD detector that is agnostic to the OOD distribution and slowly refines it during deployment using observed unlabeled data. CSO uses a new U-OOD scoring function that combines the Mahalanobis distance with a nearest-neighbor approach, along with a novel confidence-scaled few-shot OOD detector to effectively learn from limited OOD examples. We validate our approach on a dedicated benchmark, showing that our method greatly improves upon strong baselines on three medical imaging modalities.
>
---
#### [replaced 107] Selftok: Discrete Visual Tokens of Autoregression, by Diffusion, and for Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07538v2](http://arxiv.org/pdf/2505.07538v2)**

> **作者:** Bohan Wang; Zhongqi Yue; Fengda Zhang; Shuo Chen; Li'an Bi; Junzhe Zhang; Xue Song; Kennard Yanting Chan; Jiachun Pan; Weijia Wu; Mingze Zhou; Wang Lin; Kaihang Pan; Saining Zhang; Liyu Jia; Wentao Hu; Wei Zhao; Hanwang Zhang
>
> **摘要:** We completely discard the conventional spatial prior in image representation and introduce a novel discrete visual tokenizer: Self-consistency Tokenizer (Selftok). At its design core, we compose an autoregressive (AR) prior -- mirroring the causal structure of language -- into visual tokens by using the reverse diffusion process of image generation. The AR property makes Selftok fundamentally distinct from traditional spatial tokens in the following two key ways: - Selftok offers an elegant and minimalist approach to unify diffusion and AR for vision-language models (VLMs): By representing images with Selftok tokens, we can train a VLM using a purely discrete autoregressive architecture -- like that in LLMs -- without requiring additional modules or training objectives. - We theoretically show that the AR prior satisfies the Bellman equation, whereas the spatial prior does not. Therefore, Selftok supports reinforcement learning (RL) for visual generation with effectiveness comparable to that achieved in LLMs. Besides the AR property, Selftok is also a SoTA tokenizer that achieves a favorable trade-off between high-quality reconstruction and compression rate. We use Selftok to build a pure AR VLM for both visual comprehension and generation tasks. Impressively, without using any text-image training pairs, a simple policy gradient RL working in the visual tokens can significantly boost the visual generation benchmark, surpassing all the existing models by a large margin. Therefore, we believe that Selftok effectively addresses the long-standing challenge that visual tokens cannot support effective RL. When combined with the well-established strengths of RL in LLMs, this brings us one step closer to realizing a truly multimodal LLM. Project Page: https://selftok-team.github.io/report/.
>
---
#### [replaced 108] No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02831v4](http://arxiv.org/pdf/2505.02831v4)**

> **作者:** Dengyang Jiang; Mengmeng Wang; Liuzhuozheng Li; Lei Zhang; Haoyu Wang; Wei Wei; Guang Dai; Yanning Zhang; Jingdong Wang
>
> **备注:** Self-Representation Alignment for Diffusion Transformers. Code: https://github.com/vvvvvjdy/SRA
>
> **摘要:** Recent studies have demonstrated that learning a meaningful internal representation can both accelerate generative training and enhance the generation quality of diffusion transformers. However, existing approaches necessitate to either introduce an external and complex representation training framework or rely on a large-scale, pre-trained representation foundation model to provide representation guidance during the original generative training process. In this study, we posit that the unique discriminative process inherent to diffusion transformers enables them to offer such guidance without requiring external representation components. We therefore propose Self-Representation Alignment (SRA), a simple yet straightforward method that obtains representation guidance through a self-distillation manner. Specifically, SRA aligns the output latent representation of the diffusion transformer in the earlier layer with higher noise to that in the later layer with lower noise to progressively enhance the overall representation learning during only the generative training process. Experimental results indicate that applying SRA to DiTs and SiTs yields consistent performance improvements. Moreover, SRA not only significantly outperforms approaches relying on auxiliary, complex representation training frameworks but also achieves performance comparable to methods that are heavily dependent on powerful external representation priors.
>
---
#### [replaced 109] ReGraP-LLaVA: Reasoning enabled Graph-based Personalized Large Language and Vision Assistant
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.03654v2](http://arxiv.org/pdf/2505.03654v2)**

> **作者:** Yifan Xiang; Zhenxi Zhang; Bin Li; Yixuan Weng; Shoujun Zhou; Yangfan He; Keqin Li
>
> **备注:** Work in progress
>
> **摘要:** Recent advances in personalized MLLMs enable effective capture of user-specific concepts, supporting both recognition of personalized concepts and contextual captioning. However, humans typically explore and reason over relations among objects and individuals, transcending surface-level information to achieve more personalized and contextual understanding. To this end, existing methods may face three main limitations: Their training data lacks multi-object sets in which relations among objects are learnable. Building on the limited training data, their models overlook the relations between different personalized concepts and fail to reason over them. Their experiments mainly focus on a single personalized concept, where evaluations are limited to recognition and captioning tasks. To address the limitations, we present a new dataset named ReGraP, consisting of 120 sets of personalized knowledge. Each set includes images, KGs, and CoT QA pairs derived from the KGs, enabling more structured and sophisticated reasoning pathways. We propose ReGraP-LLaVA, an MLLM trained with the corresponding KGs and CoT QA pairs, where soft and hard graph prompting methods are designed to align KGs within the model's semantic space. We establish the ReGraP Benchmark, which contains diverse task types: multiple-choice, fill-in-the-blank, True/False, and descriptive questions in both open- and closed-ended settings. The proposed benchmark is designed to evaluate the relational reasoning and knowledge-connection capability of personalized MLLMs. We conduct experiments on the proposed ReGraP-LLaVA and other competitive MLLMs. Results show that the proposed model not only learns personalized knowledge but also performs relational reasoning in responses, achieving the SoTA performance compared with the competitive methods. All the codes and datasets are released at: https://github.com/xyfyyds/ReGraP.
>
---
#### [replaced 110] Progressive Autoregressive Video Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.08151v2](http://arxiv.org/pdf/2410.08151v2)**

> **作者:** Desai Xie; Zhan Xu; Yicong Hong; Hao Tan; Difan Liu; Feng Liu; Arie Kaufman; Yang Zhou
>
> **备注:** 15 pages, 7 figures. Code and video results are available at https://desaixie.github.io/pa-vdm/. v2: Accepted to CVPRW 2025. Updated figures, tables, notations, and text in all sections. Added comparison with more baseline methods, FVD metric results, user study, and discussion on parallel works
>
> **摘要:** Current frontier video diffusion models have demonstrated remarkable results at generating high-quality videos. However, they can only generate short video clips, normally around 10 seconds or 240 frames, due to computation limitations during training. Existing methods naively achieve autoregressive long video generation by directly placing the ending of the previous clip at the front of the attention window as conditioning, which leads to abrupt scene changes, unnatural motion, and error accumulation. In this work, we introduce a more natural formulation of autoregressive long video generation by revisiting the noise level assumption in video diffusion models. Our key idea is to 1. assign the frames with per-frame, progressively increasing noise levels rather than a single noise level and 2. denoise and shift the frames in small intervals rather than all at once. This allows for smoother attention correspondence among frames with adjacent noise levels, larger overlaps between the attention windows, and better propagation of information from the earlier to the later frames. Video diffusion models equipped with our progressive noise schedule can autoregressively generate long videos with much improved fidelity compared to the baselines and minimal quality degradation over time. We present the first results on text-conditioned 60-second (1440 frames) long video generation at a quality close to frontier models. Code and video results are available at https://desaixie.github.io/pa-vdm/.
>
---
#### [replaced 111] Anomaly Anything: Promptable Unseen Visual Anomaly Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.01078v3](http://arxiv.org/pdf/2406.01078v3)**

> **作者:** Han Sun; Yunkang Cao; Hao Dong; Olga Fink
>
> **备注:** 8 pages excluding appendix
>
> **摘要:** Visual anomaly detection (AD) presents significant challenges due to the scarcity of anomalous data samples. While numerous works have been proposed to synthesize anomalous samples, these synthetic anomalies often lack authenticity or require extensive training data, limiting their applicability in real-world scenarios. In this work, we propose Anomaly Anything (AnomalyAny), a novel framework that leverages Stable Diffusion (SD)'s image generation capabilities to generate diverse and realistic unseen anomalies. By conditioning on a single normal sample during test time, AnomalyAny is able to generate unseen anomalies for arbitrary object types with text descriptions. Within AnomalyAny, we propose attention-guided anomaly optimization to direct SD attention on generating hard anomaly concepts. Additionally, we introduce prompt-guided anomaly refinement, incorporating detailed descriptions to further improve the generation quality. Extensive experiments on MVTec AD and VisA datasets demonstrate AnomalyAny's ability in generating high-quality unseen anomalies and its effectiveness in enhancing downstream AD performance.
>
---
#### [replaced 112] Multi-Faceted Multimodal Monosemanticity
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.14888v2](http://arxiv.org/pdf/2502.14888v2)**

> **作者:** Hanqi Yan; Xiangxiang Cui; Lu Yin; Paul Pu Liang; Yulan He; Yifei Wang
>
> **摘要:** Humans experience the world through multiple modalities, such as, vision, language, and speech, making it natural to explore the commonality and distinctions among them. In this work, we take a data-driven approach to address this question by analyzing interpretable, monosemantic features extracted from deep multimodal models. Specifically, we investigate CLIP, a prominent visual-language representation model trained on massive image-text pairs. Building on prior research in single-modal interpretability, we develop a set of multi-modal interpretability tools and measures designed to disentangle and analyze features learned from CLIP. Specifically, we introduce the Modality Dominance Score (MDS) to attribute each CLIP feature to a specific modality. We then map CLIP features into a more interpretable space, enabling us to categorize them into three distinct classes: vision features (single-modal), language features (single-modal), and visual-language features (cross-modal). Interestingly, this data-driven categorization closely aligns with human intuitive understandings of different modalities. We further show that this modality decomposition can benefit multiple downstream tasks, including reducing bias in gender detection, generating cross-modal adversarial examples, and enabling modal-specific feature control in text-to-image generation. These results indicate that large-scale multimodal models, when equipped with task-agnostic interpretability tools, can offer valuable insights into the relationships between different data modalities.
>
---
#### [replaced 113] Graph Network for Sign Language Tasks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.12020v2](http://arxiv.org/pdf/2504.12020v2)**

> **作者:** Shiwei Gan; Yafeng Yin; Zhiwei Jiang; Hongkai Wen; Lei Xie; Sanglu Lu
>
> **备注:** 17 pages, 9 figures, submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI). This is a regular paper submission
>
> **摘要:** Recent advances in sign language research have benefited from CNN-based backbones, which are primarily transferred from traditional computer vision tasks (\eg object identification, image recognition). However, these CNN-based backbones usually excel at extracting features like contours and texture, but may struggle with capturing sign-related features. In fact, sign language tasks require focusing on sign-related regions, including the collaboration between different regions (\eg left hand region and right hand region) and the effective content in a single region. To capture such region-related features, we introduce MixSignGraph, which represents sign sequences as a group of mixed graphs and designs the following three graph modules for feature extraction, \ie Local Sign Graph (LSG) module, Temporal Sign Graph (TSG) module and Hierarchical Sign Graph (HSG) module. Specifically, the LSG module learns the correlation of intra-frame cross-region features within one frame, \ie focusing on spatial features. The TSG module tracks the interaction of inter-frame cross-region features among adjacent frames, \ie focusing on temporal features. The HSG module aggregates the same-region features from different-granularity feature maps of a frame, \ie focusing on hierarchical features. In addition, to further improve the performance of sign language tasks without gloss annotations, we propose a simple yet counter-intuitive Text-driven CTC Pre-training (TCP) method, which generates pseudo gloss labels from text labels for model pre-training. Extensive experiments conducted on current five public sign language datasets demonstrate the superior performance of the proposed model. Notably, our model surpasses the SOTA models on multiple sign language tasks across several datasets, without relying on any additional cues.
>
---
#### [replaced 114] EasyInv: Toward Fast and Better DDIM Inversion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.05159v3](http://arxiv.org/pdf/2408.05159v3)**

> **作者:** Ziyue Zhang; Mingbao Lin; Shuicheng Yan; Rongrong Ji
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** This paper introduces EasyInv, an easy yet novel approach that significantly advances the field of DDIM Inversion by addressing the inherent inefficiencies and performance limitations of traditional iterative optimization methods. At the core of our EasyInv is a refined strategy for approximating inversion noise, which is pivotal for enhancing the accuracy and reliability of the inversion process. By prioritizing the initial latent state, which encapsulates rich information about the original images, EasyInv steers clear of the iterative refinement of noise items. Instead, we introduce a methodical aggregation of the latent state from the preceding time step with the current state, effectively increasing the influence of the initial latent state and mitigating the impact of noise. We illustrate that EasyInv is capable of delivering results that are either on par with or exceed those of the conventional DDIM Inversion approach, especially under conditions where the model's precision is limited or computational resources are scarce. Concurrently, our EasyInv offers an approximate threefold enhancement regarding inference efficiency over off-the-shelf iterative optimization techniques. It can be easily combined with most existing inversion methods by only four lines of code. See code at https://github.com/potato-kitty/EasyInv.
>
---
#### [replaced 115] DiTPainter: Efficient Video Inpainting with Diffusion Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.15661v3](http://arxiv.org/pdf/2504.15661v3)**

> **作者:** Xian Wu; Chang Liu
>
> **摘要:** Many existing video inpainting algorithms utilize optical flows to construct the corresponding maps and then propagate pixels from adjacent frames to missing areas by mapping. Despite the effectiveness of the propagation mechanism, they might encounter blurry and inconsistencies when dealing with inaccurate optical flows or large masks. Recently, Diffusion Transformer (DiT) has emerged as a revolutionary technique for video generation tasks. However, pretrained DiT models for video generation all contain a large amount of parameters, which makes it very time consuming to apply to video inpainting tasks. In this paper, we present DiTPainter, an end-to-end video inpainting model based on Diffusion Transformer (DiT). DiTPainter uses an efficient transformer network designed for video inpainting, which is trained from scratch instead of initializing from any large pretrained models. DiTPainter can address videos with arbitrary lengths and can be applied to video decaptioning and video completion tasks with an acceptable time cost. Experiments show that DiTPainter outperforms existing video inpainting algorithms with higher quality and better spatial-temporal consistency.
>
---
#### [replaced 116] SiCo: An Interactive Size-Controllable Virtual Try-On Approach for Informed Decision-Making
- **分类: cs.HC; cs.CV; H.5.2; I.4.9**

- **链接: [http://arxiv.org/pdf/2408.02803v2](http://arxiv.org/pdf/2408.02803v2)**

> **作者:** Sherry X. Chen; Alex Christopher Lim; Yimeng Liu; Pradeep Sen; Misha Sra
>
> **摘要:** Virtual try-on (VTO) applications aim to replicate the in-store shopping experience and enhance online shopping by enabling users to interact with garments. However, many existing tools adopt a one-size-fits-all approach when visualizing clothing items. This approach limits user interaction with garments, particularly regarding size and fit adjustments, and fails to provide direct insights for size recommendations. As a result, these limitations contribute to high return rates in online shopping. To address this, we introduce SiCo, a new online VTO system that allows users to upload images of themselves and interact with garments by visualizing how different sizes would fit their bodies. Our user study demonstrates that our approach significantly improves users' ability to assess how outfits will appear on their bodies and increases their confidence in selecting clothing sizes that align with their preferences. Based on our evaluation, we believe that SiCo has the potential to reduce return rates and transform the online clothing shopping experience.
>
---
#### [replaced 117] InstanceGen: Image Generation with Instance-level Instructions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05678v3](http://arxiv.org/pdf/2505.05678v3)**

> **作者:** Etai Sella; Yanir Kleiman; Hadar Averbuch-Elor
>
> **备注:** Accepted to SIGGRAPH 2025. Project page: https://tau-vailab.github.io/InstanceGen/
>
> **摘要:** Despite rapid advancements in the capabilities of generative models, pretrained text-to-image models still struggle in capturing the semantics conveyed by complex prompts that compound multiple objects and instance-level attributes. Consequently, we are witnessing growing interests in integrating additional structural constraints, typically in the form of coarse bounding boxes, to better guide the generation process in such challenging cases. In this work, we take the idea of structural guidance a step further by making the observation that contemporary image generation models can directly provide a plausible fine-grained structural initialization. We propose a technique that couples this image-based structural guidance with LLM-based instance-level instructions, yielding output images that adhere to all parts of the text prompt, including object counts, instance-level attributes, and spatial relations between instances.
>
---
#### [replaced 118] Origin Identification for Text-Guided Image-to-Image Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.02376v2](http://arxiv.org/pdf/2501.02376v2)**

> **作者:** Wenhao Wang; Yifan Sun; Zongxin Yang; Zhentao Tan; Zhengdong Hu; Yi Yang
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Text-guided image-to-image diffusion models excel in translating images based on textual prompts, allowing for precise and creative visual modifications. However, such a powerful technique can be misused for spreading misinformation, infringing on copyrights, and evading content tracing. This motivates us to introduce the task of origin IDentification for text-guided Image-to-image Diffusion models (ID$^2$), aiming to retrieve the original image of a given translated query. A straightforward solution to ID$^2$ involves training a specialized deep embedding model to extract and compare features from both query and reference images. However, due to visual discrepancy across generations produced by different diffusion models, this similarity-based approach fails when training on images from one model and testing on those from another, limiting its effectiveness in real-world applications. To solve this challenge of the proposed ID$^2$ task, we contribute the first dataset and a theoretically guaranteed method, both emphasizing generalizability. The curated dataset, OriPID, contains abundant Origins and guided Prompts, which can be used to train and test potential IDentification models across various diffusion models. In the method section, we first prove the existence of a linear transformation that minimizes the distance between the pre-trained Variational Autoencoder (VAE) embeddings of generated samples and their origins. Subsequently, it is demonstrated that such a simple linear transformation can be generalized across different diffusion models. Experimental results show that the proposed method achieves satisfying generalization performance, significantly surpassing similarity-based methods ($+31.6\%$ mAP), even those with generalization designs. The project is available at https://id2icml.github.io.
>
---
#### [replaced 119] TAMP: Token-Adaptive Layerwise Pruning in Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.09897v3](http://arxiv.org/pdf/2504.09897v3)**

> **作者:** Jaewoo Lee; Keyang Xuan; Chanakya Ekbote; Sandeep Polisetty; Yi R. Fung; Paul Pu Liang
>
> **备注:** ACL Findings 2025
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown remarkable versatility in understanding diverse multimodal data and tasks. However, these capabilities come with an increased model scale. While post-training pruning reduces model size in unimodal models, its application to MLLMs often yields limited success. Our analysis discovers that conventional methods fail to account for the unique token attributes across layers and modalities inherent to MLLMs. Inspired by this observation, we propose TAMP, a simple yet effective pruning framework tailored for MLLMs, featuring two key components: (1) Diversity-Aware Sparsity, which adjusts sparsity ratio per layer based on diversities among multimodal output tokens, preserving more parameters in high-diversity layers; and (2) Adaptive Multimodal Input Activation, which identifies representative multimodal input tokens using attention scores to guide unstructured weight pruning. We validate our method on two state-of-the-art MLLMs: LLaVA-NeXT, designed for vision-language tasks, and VideoLLaMA2, capable of processing audio, visual, and language modalities. Empirical experiments across various multimodal evaluation benchmarks demonstrate that each component of our approach substantially outperforms existing pruning techniques.
>
---
#### [replaced 120] VISTA: Enhancing Vision-Text Alignment in MLLMs via Cross-Modal Mutual Information Maximization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10917v2](http://arxiv.org/pdf/2505.10917v2)**

> **作者:** Mingxiao Li; Na Su; Fang Qu; Zhizhou Zhong; Ziyang Chen; Yuan Li; Zhaopeng Tu; Xiaolong Li
>
> **摘要:** Current multimodal large language models (MLLMs) face a critical challenge in modality alignment, often exhibiting a bias towards textual information at the expense of other modalities like vision. This paper conducts a systematic information-theoretic analysis of the widely used cross-entropy loss in MLLMs, uncovering its implicit alignment objective. Our theoretical investigation reveals that this implicit objective has inherent limitations, leading to a degradation of cross-modal alignment as text sequence length increases, thereby hindering effective multimodal information fusion. To overcome these drawbacks, we propose Vision-Text Alignment (VISTA), a novel approach guided by our theoretical insights. VISTA introduces an explicit alignment objective designed to maximize cross-modal mutual information, preventing the degradation of visual alignment. Notably, VISTA enhances the visual understanding capabilities of existing MLLMs without requiring any additional trainable modules or extra training data, making it both efficient and practical. Our method significantly outperforms baseline models across more than a dozen benchmark datasets, including VQAv2, MMStar, and MME, paving the way for new directions in MLLM modal alignment research.
>
---
#### [replaced 121] VLSBench: Unveiling Visual Leakage in Multimodal Safety
- **分类: cs.CR; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.19939v3](http://arxiv.org/pdf/2411.19939v3)**

> **作者:** Xuhao Hu; Dongrui Liu; Hao Li; Xuanjing Huang; Jing Shao
>
> **备注:** ACL2025 Main
>
> **摘要:** Safety concerns of Multimodal large language models (MLLMs) have gradually become an important problem in various applications. Surprisingly, previous works indicate a counterintuitive phenomenon that using textual unlearning to align MLLMs achieves comparable safety performances with MLLMs aligned with image text pairs. To explain such a phenomenon, we discover a Visual Safety Information Leakage (VSIL) problem in existing multimodal safety benchmarks, i.e., the potentially risky content in the image has been revealed in the textual query. Thus, MLLMs can easily refuse these sensitive image-text pairs according to textual queries only, leading to unreliable cross-modality safety evaluation of MLLMs. We also conduct a further comparison experiment between textual alignment and multimodal alignment to highlight this drawback. To this end, we construct multimodal Visual Leakless Safety Bench (VLSBench) with 2.2k image-text pairs through an automated data pipeline. Experimental results indicate that VLSBench poses a significant challenge to both open-source and close-source MLLMs, e.g., LLaVA, Qwen2-VL and GPT-4o. Besides, we empirically compare textual and multimodal alignment methods on VLSBench and find that textual alignment is effective enough for multimodal safety scenarios with VSIL, while multimodal alignment is preferable for safety scenarios without VSIL. Code and data are released under https://github.com/AI45Lab/VLSBench
>
---
#### [replaced 122] Submillimeter-Accurate 3D Lumbar Spine Reconstruction from Biplanar X-Ray Images: Incorporating a Multi-Task Network and Landmark-Weighted Loss
- **分类: eess.IV; cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2503.14573v2](http://arxiv.org/pdf/2503.14573v2)**

> **作者:** Wanxin Yu; Zhemin Zhu; Cong Wang; Yihang Bao; Chunjie Xia; Rongshan Cheng; Yan Yu; Tsung-Yuan Tsai
>
> **备注:** 24 pages, 11 figures, 5 tables
>
> **摘要:** Three-dimensional reconstruction of the spine under weight-bearing conditions from biplanar X-ray images is of great importance for the clinical assessment of spinal diseases. However, the current fully automated reconstruction methods only achieve millimeter-level accuracy, making it difficult to meet clinical standards. This study developed and validated a fully automated method for high-accuracy 3D reconstruction of the lumbar spine from biplanar X-ray images. The method involves lumbar decomposition and landmark detection from the raw X-ray images, followed by a deformable model and landmark-weighted 2D-3D registration approach. The reconstruction accuracy was validated by the gold standard obtained through the registration of CT-segmented vertebral models with the biplanar X-ray images. The proposed method achieved a 3D reconstruction accuracy of 0.80mm, representing a significant improvement over the mainstream approaches. This study will contribute to the clinical diagnosis of lumbar in weight-bearing positions.
>
---
#### [replaced 123] JetFormer: An Autoregressive Generative Model of Raw Images and Text
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.19722v2](http://arxiv.org/pdf/2411.19722v2)**

> **作者:** Michael Tschannen; André Susano Pinto; Alexander Kolesnikov
>
> **备注:** ICLR 2025. Code available at https://github.com/google-research/big_vision
>
> **摘要:** Removing modeling constraints and unifying architectures across domains has been a key driver of the recent progress in training large multimodal models. However, most of these models still rely on many separately trained components such as modality-specific encoders and decoders. In this work, we further streamline joint generative modeling of images and text. We propose an autoregressive decoder-only transformer - JetFormer - which is trained to directly maximize the likelihood of raw data, without relying on any separately pretrained components, and can understand and generate both text and images. Specifically, we leverage a normalizing flow model to obtain a soft-token image representation that is jointly trained with an autoregressive multimodal transformer. The normalizing flow model serves as both an image encoder for perception tasks and an image decoder for image generation tasks during inference. JetFormer achieves text-to-image generation quality competitive with recent VQ-VAE- and VAE-based baselines. These baselines rely on pretrained image autoencoders, which are trained with a complex mixture of losses, including perceptual ones. At the same time, JetFormer demonstrates robust image understanding capabilities. To the best of our knowledge, JetFormer is the first model that is capable of generating high-fidelity images and producing strong log-likelihood bounds.
>
---
#### [replaced 124] DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2211.01095v3](http://arxiv.org/pdf/2211.01095v3)**

> **作者:** Cheng Lu; Yuhao Zhou; Fan Bao; Jianfei Chen; Chongxuan Li; Jun Zhu
>
> **备注:** Machine Intelligence Research
>
> **摘要:** Diffusion probabilistic models (DPMs) have achieved impressive success in high-resolution image synthesis, especially in recent large-scale text-to-image generation applications. An essential technique for improving the sample quality of DPMs is guided sampling, which usually needs a large guidance scale to obtain the best sample quality. The commonly-used fast sampler for guided sampling is DDIM, a first-order diffusion ODE solver that generally needs 100 to 250 steps for high-quality samples. Although recent works propose dedicated high-order solvers and achieve a further speedup for sampling without guidance, their effectiveness for guided sampling has not been well-tested before. In this work, we demonstrate that previous high-order fast samplers suffer from instability issues, and they even become slower than DDIM when the guidance scale grows large. To further speed up guided sampling, we propose DPM-Solver++, a high-order solver for the guided sampling of DPMs. DPM-Solver++ solves the diffusion ODE with the data prediction model and adopts thresholding methods to keep the solution matches training data distribution. We further propose a multistep variant of DPM-Solver++ to address the instability issue by reducing the effective step size. Experiments show that DPM-Solver++ can generate high-quality samples within only 15 to 20 steps for guided sampling by pixel-space and latent-space DPMs.
>
---
#### [replaced 125] Exploring the Potential of Encoder-free Architectures in 3D LMMs
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.09620v2](http://arxiv.org/pdf/2502.09620v2)**

> **作者:** Yiwen Tang; Zoey Guo; Zhuhao Wang; Ray Zhang; Qizhi Chen; Junli Liu; Delin Qu; Zhigang Wang; Dong Wang; Xuelong Li; Bin Zhao
>
> **备注:** The code is released at https://github.com/Ivan-Tang-3D/ENEL
>
> **摘要:** Encoder-free architectures have been preliminarily explored in the 2D visual domain, yet it remains an open question whether they can be effectively applied to 3D understanding scenarios. In this paper, we present the first comprehensive investigation into the potential of encoder-free architectures to alleviate the challenges of encoder-based 3D Large Multimodal Models (LMMs). These challenges include the failure to adapt to varying point cloud resolutions and the point features from the encoder not meeting the semantic needs of Large Language Models (LLMs). We identify key aspects for 3D LMMs to remove the encoder and enable the LLM to assume the role of the 3D encoder: 1) We propose the LLM-embedded Semantic Encoding strategy in the pre-training stage, exploring the effects of various point cloud self-supervised losses. And we present the Hybrid Semantic Loss to extract high-level semantics. 2) We introduce the Hierarchical Geometry Aggregation strategy in the instruction tuning stage. This incorporates inductive bias into the LLM layers to focus on the local details of the point clouds. To the end, we present the first Encoder-free 3D LMM, ENEL. Our 7B model rivals the current state-of-the-art model, ShapeLLM-13B, achieving 55.10%, 50.98%, and 43.10% on the classification, captioning, and VQA tasks, respectively. Our results demonstrate that the encoder-free architecture is highly promising for replacing encoder-based architectures in the field of 3D understanding. The code is released at https://github.com/Ivan-Tang-3D/ENEL
>
---
#### [replaced 126] ImageRAG: Enhancing Ultra High Resolution Remote Sensing Imagery Analysis with ImageRAG
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.07688v3](http://arxiv.org/pdf/2411.07688v3)**

> **作者:** Zilun Zhang; Haozhan Shen; Tiancheng Zhao; Zian Guan; Bin Chen; Yuhao Wang; Xu Jia; Yuxiang Cai; Yongheng Shang; Jianwei Yin
>
> **备注:** full paper
>
> **摘要:** Ultra High Resolution (UHR) remote sensing imagery (RSI) (e.g. 100,000 $\times$ 100,000 pixels or more) poses a significant challenge for current Remote Sensing Multimodal Large Language Models (RSMLLMs). If choose to resize the UHR image to standard input image size, the extensive spatial and contextual information that UHR images contain will be neglected. Otherwise, the original size of these images often exceeds the token limits of standard RSMLLMs, making it difficult to process the entire image and capture long-range dependencies to answer the query based on the abundant visual context. In this paper, we introduce ImageRAG for RS, a training-free framework to address the complexities of analyzing UHR remote sensing imagery. By transforming UHR remote sensing image analysis task to image's long context selection task, we design an innovative image contextual retrieval mechanism based on the Retrieval-Augmented Generation (RAG) technique, denoted as ImageRAG. ImageRAG's core innovation lies in its ability to selectively retrieve and focus on the most relevant portions of the UHR image as visual contexts that pertain to a given query. Fast path and slow path are proposed in this framework to handle this task efficiently and effectively. ImageRAG allows RSMLLMs to manage extensive context and spatial information from UHR RSI, ensuring the analysis is both accurate and efficient. Codebase will be released in https://github.com/om-ai-lab/ImageRAG
>
---
#### [replaced 127] RevCD -- Reversed Conditional Diffusion for Generalized Zero-Shot Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.00511v2](http://arxiv.org/pdf/2409.00511v2)**

> **作者:** William Heyden; Habib Ullah; M. Salman Siddiqui; Fadi Al Machot
>
> **备注:** Accepted as Full Paper of DeLTA 2025. The Conference Proceedings will be published by Springer
>
> **摘要:** In Generalized Zero-Shot Learning (GZSL), we aim to recognize both seen and unseen categories using a model trained only on seen categories. In computer vision, this translates into a classification problem, where knowledge from seen categories is transferred to unseen categories by exploiting the relationships between visual features and available semantic information, such as text corpora or manual annotations. However, learning this joint distribution is costly and requires one-to-one training with corresponding semantic information. We present a reversed conditional Diffusion-based model (RevCD) that mitigates this issue by generating semantic features synthesized from visual inputs by leveraging Diffusion models' conditional mechanisms. Our RevCD model consists of a cross Hadamard-Addition embedding of a sinusoidal time schedule and a multi-headed visual transformer for attention-guided embeddings. The proposed approach introduces three key innovations. First, we reverse the process of generating semantic space based on visual data, introducing a novel loss function that facilitates more efficient knowledge transfer. Second, we apply Diffusion models to zero-shot learning - a novel approach that exploits their strengths in capturing data complexity. Third, we demonstrate our model's performance through a comprehensive cross-dataset evaluation. The complete code will be available on GitHub.
>
---
#### [replaced 128] UniversalRAG: Retrieval-Augmented Generation over Corpora of Diverse Modalities and Granularities
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.20734v2](http://arxiv.org/pdf/2504.20734v2)**

> **作者:** Woongyeong Yeo; Kangsan Kim; Soyeong Jeong; Jinheon Baek; Sung Ju Hwang
>
> **备注:** Project page : https://universalrag.github.io
>
> **摘要:** Retrieval-Augmented Generation (RAG) has shown substantial promise in improving factual accuracy by grounding model responses with external knowledge relevant to queries. However, most existing RAG approaches are limited to a text-only corpus, and while recent efforts have extended RAG to other modalities such as images and videos, they typically operate over a single modality-specific corpus. In contrast, real-world queries vary widely in the type of knowledge they require, which a single type of knowledge source cannot address. To address this, we introduce UniversalRAG, a novel RAG framework designed to retrieve and integrate knowledge from heterogeneous sources with diverse modalities and granularities. Specifically, motivated by the observation that forcing all modalities into a unified representation space derived from a single aggregated corpus causes a modality gap, where the retrieval tends to favor items from the same modality as the query, we propose a modality-aware routing mechanism that dynamically identifies the most appropriate modality-specific corpus and performs targeted retrieval within it. Also, beyond modality, we organize each modality into multiple granularity levels, enabling fine-tuned retrieval tailored to the complexity and scope of the query. We validate UniversalRAG on 8 benchmarks spanning multiple modalities, showing its superiority over various modality-specific and unified baselines.
>
---
#### [replaced 129] SOAP: Style-Omniscient Animatable Portraits
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05022v2](http://arxiv.org/pdf/2505.05022v2)**

> **作者:** Tingting Liao; Yujian Zheng; Adilbek Karmanov; Liwen Hu; Leyang Jin; Yuliang Xiu; Hao Li
>
> **摘要:** Creating animatable 3D avatars from a single image remains challenging due to style limitations (realistic, cartoon, anime) and difficulties in handling accessories or hairstyles. While 3D diffusion models advance single-view reconstruction for general objects, outputs often lack animation controls or suffer from artifacts because of the domain gap. We propose SOAP, a style-omniscient framework to generate rigged, topology-consistent avatars from any portrait. Our method leverages a multiview diffusion model trained on 24K 3D heads with multiple styles and an adaptive optimization pipeline to deform the FLAME mesh while maintaining topology and rigging via differentiable rendering. The resulting textured avatars support FACS-based animation, integrate with eyeballs and teeth, and preserve details like braided hair or accessories. Extensive experiments demonstrate the superiority of our method over state-of-the-art techniques for both single-view head modeling and diffusion-based generation of Image-to-3D. Our code and data are publicly available for research purposes at https://github.com/TingtingLiao/soap.
>
---
#### [replaced 130] Does Vector Quantization Fail in Spatio-Temporal Forecasting? Exploring a Differentiable Sparse Soft-Vector Quantization Approach
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2312.03406v4](http://arxiv.org/pdf/2312.03406v4)**

> **作者:** Chao Chen; Tian Zhou; Yanjun Zhao; Hui Liu; Liang Sun; Rong Jin
>
> **备注:** Accepted by KDD2025 research track
>
> **摘要:** Spatio-temporal forecasting is crucial in various fields and requires a careful balance between identifying subtle patterns and filtering out noise. Vector quantization (VQ) appears well-suited for this purpose, as it quantizes input vectors into a set of codebook vectors or patterns. Although VQ has shown promise in various computer vision tasks, it surprisingly falls short in enhancing the accuracy of spatio-temporal forecasting. We attribute this to two main issues: inaccurate optimization due to non-differentiability and limited representation power in hard-VQ. To tackle these challenges, we introduce Differentiable Sparse Soft-Vector Quantization (SVQ), the first VQ method to enhance spatio-temporal forecasting. SVQ balances detail preservation with noise reduction, offering full differentiability and a solid foundation in sparse regression. Our approach employs a two-layer MLP and an extensive codebook to streamline the sparse regression process, significantly cutting computational costs while simplifying training and improving performance. Empirical studies on five spatio-temporal benchmark datasets show SVQ achieves state-of-the-art results, including a 7.9% improvement on the WeatherBench-S temperature dataset and an average mean absolute error reduction of 9.4% in video prediction benchmarks (Human3.6M, KTH, and KittiCaltech), along with a 17.3% enhancement in image quality (LPIPS). Code is publicly available at https://github.com/Pachark/SVQ-Forecasting.
>
---
#### [replaced 131] Ultrasound Report Generation with Multimodal Large Language Models for Standardized Texts
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08838v2](http://arxiv.org/pdf/2505.08838v2)**

> **作者:** Peixuan Ge; Tongkun Su; Faqin Lv; Baoliang Zhao; Peng Zhang; Chi Hong Wong; Liang Yao; Yu Sun; Zenan Wang; Pak Kin Wong; Ying Hu
>
> **摘要:** Ultrasound (US) report generation is a challenging task due to the variability of US images, operator dependence, and the need for standardized text. Unlike X-ray and CT, US imaging lacks consistent datasets, making automation difficult. In this study, we propose a unified framework for multi-organ and multilingual US report generation, integrating fragment-based multilingual training and leveraging the standardized nature of US reports. By aligning modular text fragments with diverse imaging data and curating a bilingual English-Chinese dataset, the method achieves consistent and clinically accurate text generation across organ sites and languages. Fine-tuning with selective unfreezing of the vision transformer (ViT) further improves text-image alignment. Compared to the previous state-of-the-art KMVE method, our approach achieves relative gains of about 2\% in BLEU scores, approximately 3\% in ROUGE-L, and about 15\% in CIDEr, while significantly reducing errors such as missing or incorrect content. By unifying multi-organ and multi-language report generation into a single, scalable framework, this work demonstrates strong potential for real-world clinical workflows.
>
---
#### [replaced 132] Robustness-Reinforced Knowledge Distillation with Correlation Distance and Network Pruning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2311.13934v2](http://arxiv.org/pdf/2311.13934v2)**

> **作者:** Seonghak Kim; Gyeongdo Ham; Yucheol Cho; Daeshik Kim
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** The improvement in the performance of efficient and lightweight models (i.e., the student model) is achieved through knowledge distillation (KD), which involves transferring knowledge from more complex models (i.e., the teacher model). However, most existing KD techniques rely on Kullback-Leibler (KL) divergence, which has certain limitations. First, if the teacher distribution has high entropy, the KL divergence's mode-averaging nature hinders the transfer of sufficient target information. Second, when the teacher distribution has low entropy, the KL divergence tends to excessively focus on specific modes, which fails to convey an abundant amount of valuable knowledge to the student. Consequently, when dealing with datasets that contain numerous confounding or challenging samples, student models may struggle to acquire sufficient knowledge, resulting in subpar performance. Furthermore, in previous KD approaches, we observed that data augmentation, a technique aimed at enhancing a model's generalization, can have an adverse impact. Therefore, we propose a Robustness-Reinforced Knowledge Distillation (R2KD) that leverages correlation distance and network pruning. This approach enables KD to effectively incorporate data augmentation for performance improvement. Extensive experiments on various datasets, including CIFAR-100, FGVR, TinyImagenet, and ImageNet, demonstrate our method's superiority over current state-of-the-art methods.
>
---
#### [replaced 133] Refinement Module based on Parse Graph of Feature Map for Human Pose Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.11069v5](http://arxiv.org/pdf/2501.11069v5)**

> **作者:** Shibang Liu; Xuemei Xie; Guangming Shi
>
> **摘要:** The parse graph play a crucial role in enhancing the performance of human pose estimation (HPE). Its key advantage lies in its hierarchical structure, like a tree structure, and context relations among nodes, which enable more accurate for inference. To equip models with the advantage of parse graphs, many researchers predefine the parse graph of body structure for HPE. However, these frameworks struggle to adapt to instances that deviate from the predefined parse graph and they are often parameter-heavy. Unlike them, we view the feature map holistically, much like the human body. It can be optimized using parse graphs, where nodes' implicit feature representation boosts adaptability, avoiding rigid structural limitations. In this paper, we design the Refinement Module based on the Parse Graph of feature map (RMPG), which includes two stages: top-down decomposition and bottom-up combination. In the first stage, the feature map is constructed into a tree structure through recursive decomposition, with each node representing a sub-feature map, thereby achieving hierarchical modeling of features. In the second stage, context information is calculated and sub-feature maps with context are recursively connected to gradually build a refined feature map. Additionally, we design a hierarchical network with fewer parameters using multiple RMPG modules to model the context relations and hierarchies in the parse graph of body structure for HPE, some of which are supervised to obtain context relations among body parts. Our network achieves excellent results on multiple mainstream human pose datasets and the effectiveness of RMPG is proven on different methods. The code of RMPG will be open.
>
---
#### [replaced 134] DPBridge: Latent Diffusion Bridge for Dense Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.20506v3](http://arxiv.org/pdf/2412.20506v3)**

> **作者:** Haorui Ji; Taojun Lin; Hongdong Li
>
> **摘要:** Diffusion models demonstrate remarkable capabilities in capturing complex data distributions and have achieved compelling results in many generative tasks. While they have recently been extended to dense prediction tasks such as depth estimation and surface normal prediction, their full potential in this area remains under-explored. In dense prediction settings, target signal maps and input images are pixel-wise aligned. This makes conventional noise-to-data generation paradigm inefficient, as input images can serve as more informative prior compared to pure noise. Diffusion bridge models, which support data-to-data generation between two general data distributions, offer a promising alternative, but they typically fail to exploit the rich visual priors embedded in large pretrained foundation models. To address these limitations, we integrate diffusion bridge formulation with structured visual priors and introduce DPBridge, the first latent diffusion bridge framework for dense prediction tasks. Our method presents three key contributions: (1) a tractable reverse transition kernel for diffusion bridge process, enabling maximum likelihood training scheme for better compatibility with pretrained backbones; (2) a distribution-aligned normalization technique to mitigate the discrepancies between the bridge and standard diffusion processes; and (3) an auxiliary image consistency loss to preserve fine-grained details. Experiments across extensive benchmarks validate that our method consistently achieves superior performance, demonstrating its effectiveness and generalization capability under different scenarios.
>
---
#### [replaced 135] SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.18427v4](http://arxiv.org/pdf/2501.18427v4)**

> **作者:** Enze Xie; Junsong Chen; Yuyang Zhao; Jincheng Yu; Ligeng Zhu; Chengyue Wu; Yujun Lin; Zhekai Zhang; Muyang Li; Junyu Chen; Han Cai; Bingchen Liu; Daquan Zhou; Song Han
>
> **摘要:** This paper presents SANA-1.5, a linear Diffusion Transformer for efficient scaling in text-to-image generation. Building upon SANA-1.0, we introduce three key innovations: (1) Efficient Training Scaling: A depth-growth paradigm that enables scaling from 1.6B to 4.8B parameters with significantly reduced computational resources, combined with a memory-efficient 8-bit optimizer. (2) Model Depth Pruning: A block importance analysis technique for efficient model compression to arbitrary sizes with minimal quality loss. (3) Inference-time Scaling: A repeated sampling strategy that trades computation for model capacity, enabling smaller models to match larger model quality at inference time. Through these strategies, SANA-1.5 achieves a text-image alignment score of 0.81 on GenEval, which can be further improved to 0.96 through inference scaling with VILA-Judge, establishing a new SoTA on GenEval benchmark. These innovations enable efficient model scaling across different compute budgets while maintaining high quality, making high-quality image generation more accessible. Our code and pre-trained models are released.
>
---
#### [replaced 136] Low-hallucination Synthetic Captions for Large-Scale Vision-Language Model Pre-training
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.13123v2](http://arxiv.org/pdf/2504.13123v2)**

> **作者:** Xinsong Zhang; Yarong Zeng; Xinting Huang; Hu Hu; Runquan Xie; Han Hu; Zhanhui Kang
>
> **摘要:** In recent years, the field of vision-language model pre-training has experienced rapid advancements, driven primarily by the continuous enhancement of textual capabilities in large language models. However, existing training paradigms for multimodal large language models heavily rely on high-quality image-text pairs. As models and data scales grow exponentially, the availability of such meticulously curated data has become increasingly scarce and saturated, thereby severely limiting further advancements in this domain. This study investigates scalable caption generation techniques for vision-language model pre-training and demonstrates that large-scale low-hallucination synthetic captions can serve dual purposes: 1) acting as a viable alternative to real-world data for pre-training paradigms and 2) achieving superior performance enhancement when integrated into vision-language models through empirical validation. This paper presents following key contributions: 1) a novel pipeline for generating high-quality, low-hallucination, and knowledge-rich synthetic captions. Our continuous DPO methodology yields remarkable results in reducing hallucinations. Specifically, the non-hallucination caption rate on a held-out test set increases from 48.3% to 77.9% for a 7B-size model. 2) Comprehensive empirical validation reveals that our synthetic captions confer superior pre-training advantages over their counterparts. Across 15 vision language tasks, the model trained with our data achieves a significant performance gain of at least 6.2% compared to identical images with alt-text. In 20 common cognitive domains, the model trained with our data outperforms the alt-text data by at least 7.5%. Meanwhile, it also offers considerable support in the text-to-image domain. With our dataset, the FID score is reduced by 17.1 on a real-world validation benchmark and 13.3 on the MSCOCO validation benchmark.
>
---
#### [replaced 137] Grokking at the Edge of Numerical Stability
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2501.04697v2](http://arxiv.org/pdf/2501.04697v2)**

> **作者:** Lucas Prieto; Melih Barsbey; Pedro A. M. Mediano; Tolga Birdal
>
> **摘要:** Grokking, the sudden generalization that occurs after prolonged overfitting, is a surprising phenomenon challenging our understanding of deep learning. Although significant progress has been made in understanding grokking, the reasons behind the delayed generalization and its dependence on regularization remain unclear. In this work, we argue that without regularization, grokking tasks push models to the edge of numerical stability, introducing floating point errors in the Softmax function, which we refer to as Softmax Collapse (SC). We demonstrate that SC prevents grokking and that mitigating SC enables grokking without regularization. Investigating the root cause of SC, we find that beyond the point of overfitting, the gradients strongly align with what we call the na\"ive loss minimization (NLM) direction. This component of the gradient does not alter the model's predictions but decreases the loss by scaling the logits, typically by scaling the weights along their current direction. We show that this scaling of the logits explains the delay in generalization characteristic of grokking and eventually leads to SC, halting further learning. To validate our hypotheses, we introduce two key contributions that address the challenges in grokking tasks: StableMax, a new activation function that prevents SC and enables grokking without regularization, and $\perp$Grad, a training algorithm that promotes quick generalization in grokking tasks by preventing NLM altogether. These contributions provide new insights into grokking, elucidating its delayed generalization, reliance on regularization, and the effectiveness of existing grokking-inducing methods. Code for this paper is available at https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability.
>
---
#### [replaced 138] Super-Resolution Generative Adversarial Networks based Video Enhancement
- **分类: cs.CV; cs.AI; eess.IV; I.4.3**

- **链接: [http://arxiv.org/pdf/2505.10589v2](http://arxiv.org/pdf/2505.10589v2)**

> **作者:** Kağan ÇETİN
>
> **备注:** 28 pages, 14 figures, 3 tables
>
> **摘要:** This study introduces an enhanced approach to video super-resolution by extending ordinary Single-Image Super-Resolution (SISR) Super-Resolution Generative Adversarial Network (SRGAN) structure to handle spatio-temporal data. While SRGAN has proven effective for single-image enhancement, its design does not account for the temporal continuity required in video processing. To address this, a modified framework that incorporates 3D Non-Local Blocks is proposed, which is enabling the model to capture relationships across both spatial and temporal dimensions. An experimental training pipeline is developed, based on patch-wise learning and advanced data degradation techniques, to simulate real-world video conditions and learn from both local and global structures and details. This helps the model generalize better and maintain stability across varying video content while maintaining the general structure besides the pixel-wise correctness. Two model variants-one larger and one more lightweight-are presented to explore the trade-offs between performance and efficiency. The results demonstrate improved temporal coherence, sharper textures, and fewer visual artifacts compared to traditional single-image methods. This work contributes to the development of practical, learning-based solutions for video enhancement tasks, with potential applications in streaming, gaming, and digital restoration.
>
---
#### [replaced 139] IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.21759v3](http://arxiv.org/pdf/2410.21759v3)**

> **作者:** Hang Guo; Yawei Li; Tao Dai; Shu-Tao Xia; Luca Benini
>
> **备注:** ICML 2025
>
> **摘要:** Fine-tuning pre-trained diffusion models under limited budgets has gained great success. In particular, the recent advances that directly fine-tune the quantized weights using Low-rank Adaptation (LoRA) further reduces training costs. Despite these progress, we point out that existing adaptation recipes are not inference-efficient. Specifically, additional post-training quantization (PTQ) on tuned weights is needed during deployment, which results in noticeable performance drop when the bit-width is low. Based on this observation, we introduce IntLoRA, which adapts quantized diffusion models with integer-type low-rank parameters, to include inference efficiency during tuning. Specifically, IntLoRA enables pre-trained weights to remain quantized during training, facilitating fine-tuning on consumer-level GPUs. During inference, IntLoRA weights can be seamlessly merged into pre-trained weights to directly obtain quantized downstream weights without PTQ. Extensive experiments show our IntLoRA achieves significant speedup on both training and inference without losing performance.
>
---
#### [replaced 140] PointArena: Probing Multimodal Grounding Through Language-Guided Pointing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.09990v2](http://arxiv.org/pdf/2505.09990v2)**

> **作者:** Long Cheng; Jiafei Duan; Yi Ru Wang; Haoquan Fang; Boyang Li; Yushan Huang; Elvis Wang; Ainaz Eftekhar; Jason Lee; Wentao Yuan; Rose Hendrix; Noah A. Smith; Fei Xia; Dieter Fox; Ranjay Krishna
>
> **备注:** 10 Pages, Dataset and code:https://pointarena.github.io/
>
> **摘要:** Pointing serves as a fundamental and intuitive mechanism for grounding language within visual contexts, with applications spanning robotics, assistive technologies, and interactive AI systems. While recent multimodal models have started to support pointing capabilities, existing benchmarks typically focus only on referential object localization tasks. We introduce PointArena, a comprehensive platform for evaluating multimodal pointing across diverse reasoning scenarios. PointArena comprises three components: (1) Point-Bench, a curated dataset containing approximately 1,000 pointing tasks across five reasoning categories; (2) Point-Battle, an interactive, web-based arena facilitating blind, pairwise model comparisons, which has already gathered over 4,500 anonymized votes; and (3) Point-Act, a real-world robotic manipulation system allowing users to directly evaluate multimodal model pointing capabilities in practical settings. We conducted extensive evaluations of both state-of-the-art open-source and proprietary multimodal models. Results indicate that Molmo-72B consistently outperforms other models, though proprietary models increasingly demonstrate comparable performance. Additionally, we find that supervised training specifically targeting pointing tasks significantly enhances model performance. Across our multi-stage evaluation pipeline, we also observe strong correlations, underscoring the critical role of precise pointing capabilities in enabling multimodal models to effectively bridge abstract reasoning with concrete, real-world actions. Project page: https://pointarena.github.io/
>
---
#### [replaced 141] Mitigate Language Priors in Large Vision-Language Models by Cross-Images Contrastive Decoding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10634v2](http://arxiv.org/pdf/2505.10634v2)**

> **作者:** Jianfei Zhao; Feng Zhang; Xin Sun; Chong Feng
>
> **摘要:** Language priors are a major cause of hallucinations in Large Vision-Language Models (LVLMs), often leading to text that is linguistically plausible but visually inconsistent. Recent work explores contrastive decoding as a training-free solution, but these methods typically construct negative visual contexts from the original image, resulting in visual information loss and distorted distribution. Motivated by the observation that language priors stem from the LLM backbone and remain consistent across images, we propose Cross-Images Contrastive Decoding (CICD), a simple yet effective training-free method that uses different images to construct negative visual contexts. We further analyze the cross-image behavior of language priors and introduce a distinction between essential priors (supporting fluency) and detrimental priors (causing hallucinations), enabling selective suppression. By selectively preserving essential priors and suppressing detrimental ones, our method reduces hallucinations while maintaining coherent and fluent language generation. Experiments on four benchmarks and six LVLMs across three model families confirm the effectiveness and generalizability of CICD, especially in image captioning, where language priors are particularly pronounced. Code will be released upon acceptance.
>
---
#### [replaced 142] Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.15633v3](http://arxiv.org/pdf/2411.15633v3)**

> **作者:** Zhiyuan Yan; Jiangming Wang; Peng Jin; Ke-Yue Zhang; Chengchun Liu; Shen Chen; Taiping Yao; Shouhong Ding; Baoyuan Wu; Li Yuan
>
> **摘要:** AI-generated images (AIGIs), such as natural or face images, have become increasingly important yet challenging. In this paper, we start from a new perspective to excavate the reason behind the failure generalization in AIGI detection, named the \textit{asymmetry phenomenon}, where a naively trained detector tends to favor overfitting to the limited and monotonous fake patterns, causing the feature space to become highly constrained and low-ranked, which is proved seriously limiting the expressivity and generalization. One potential remedy is incorporating the pre-trained knowledge within the vision foundation models (higher-ranked) to expand the feature space, alleviating the model's overfitting to fake. To this end, we employ Singular Value Decomposition (SVD) to decompose the original feature space into \textit{two orthogonal subspaces}. By freezing the principal components and adapting only the remained components, we preserve the pre-trained knowledge while learning fake patterns. Compared to existing full-parameters and LoRA-based tuning methods, we explicitly ensure orthogonality, enabling the higher rank of the whole feature space, effectively minimizing overfitting and enhancing generalization. We finally identify a crucial insight: our method implicitly learns \textit{a vital prior that fakes are actually derived from the real}, indicating a hierarchical relationship rather than independence. Modeling this prior, we believe, is essential for achieving superior generalization. Our codes are publicly available at \href{https://github.com/YZY-stack/Effort-AIGI-Detection}{GitHub}.
>
---
#### [replaced 143] Generative Pre-trained Autoregressive Diffusion Transformer
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.07344v3](http://arxiv.org/pdf/2505.07344v3)**

> **作者:** Yuan Zhang; Jiacheng Jiang; Guoqing Ma; Zhiying Lu; Haoyang Huang; Jianlong Yuan; Nan Duan
>
> **摘要:** In this work, we present GPDiT, a Generative Pre-trained Autoregressive Diffusion Transformer that unifies the strengths of diffusion and autoregressive modeling for long-range video synthesis, within a continuous latent space. Instead of predicting discrete tokens, GPDiT autoregressively predicts future latent frames using a diffusion loss, enabling natural modeling of motion dynamics and semantic consistency across frames. This continuous autoregressive framework not only enhances generation quality but also endows the model with representation capabilities. Additionally, we introduce a lightweight causal attention variant and a parameter-free rotation-based time-conditioning mechanism, improving both the training and inference efficiency. Extensive experiments demonstrate that GPDiT achieves strong performance in video generation quality, video representation ability, and few-shot learning tasks, highlighting its potential as an effective framework for video modeling in continuous space.
>
---
#### [replaced 144] A Semantic-Aware and Multi-Guided Network for Infrared-Visible Image Fusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.06159v3](http://arxiv.org/pdf/2407.06159v3)**

> **作者:** Xiaoli Zhang; Liying Wang; Libo Zhao; Xiongfei Li; Siwei Ma
>
> **备注:** TMM 2026
>
> **摘要:** Multi-modality image fusion aims at fusing modality-specific (complementarity) and modality-shared (correlation) information from multiple source images. To tackle the problem of the neglect of inter-feature relationships, high-frequency information loss, and the limited attention to downstream tasks, this paper focuses on how to model correlation-driven decomposing features and reason high-level graph representation by efficiently extracting complementary information and aggregating multi-guided features. We propose a three-branch encoder-decoder architecture along with corresponding fusion layers as the fusion strategy. Firstly, shallow features from individual modalities are extracted by a depthwise convolution layer combined with the transformer block. In the three parallel branches of the encoder, Cross Attention and Invertible Block (CAI) extracts local features and preserves high-frequency texture details. Base Feature Extraction Module (BFE) captures long-range dependencies and enhances modality-shared information. Graph Reasoning Module (GR) is introduced to reason high-level cross-modality relations and simultaneously extract low-level detail features as CAI's modality-specific complementary information. Experiments demonstrate the competitive results compared with state-of-the-art methods in visible/infrared image fusion and medical image fusion tasks. Moreover, the proposed algorithm surpasses the state-of-the-art methods in terms of subsequent tasks, averagely scoring 8.27% mAP@0.5 higher in object detection and 5.85% mIoU higher in semantic segmentation. The code is avaliable at https://github.com/Abraham-Einstein/SMFNet/.
>
---
#### [replaced 145] LadderMIL: Multiple Instance Learning with Coarse-to-Fine Self-Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.02707v3](http://arxiv.org/pdf/2502.02707v3)**

> **作者:** Shuyang Wu; Yifu Qiu; Ines P. Nearchou; Sandrine Prost; Jonathan A. Fallowfield; David J. Harrison; Hakan Bilen; Timothy J. Kendall
>
> **摘要:** Multiple Instance Learning (MIL) for whole slide image (WSI) analysis in computational pathology often neglects instance-level learning as supervision is typically provided only at the bag level. In this work, we present LadderMIL, a framework designed to improve MIL through two perspectives: (1) employing instance-level supervision and (2) learning inter-instance contextual information at bag level. Firstly, we propose a novel Coarse-to-Fine Self-Distillation (CFSD) paradigm that probes and distils a network trained with bag-level information to adaptively obtain instance-level labels which could effectively provide the instance-level supervision for the same network in a self-improving way. Secondly, to capture inter-instance contextual information in WSI, we propose a Contextual Ecoding Generator (CEG), which encodes the contextual appearance of instances within a bag. We also theoretically and empirically prove the instance-level learnability of CFSD. Our LadderMIL is evaluated on multiple clinically relevant benchmarking tasks including breast cancer receptor status classification, multi-class subtype classification, tumour classification, and prognosis prediction. Average improvements of 8.1%, 11% and 2.4% in AUC, F1-score, and C-index, respectively, are demonstrated across the five benchmarks, compared to the best baseline.
>
---
#### [replaced 146] Logic-in-Frames: Dynamic Keyframe Search via Visual Semantic-Logical Verification for Long Video Understanding
- **分类: cs.CV; cs.AI; cs.CL; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.13139v2](http://arxiv.org/pdf/2503.13139v2)**

> **作者:** Weiyu Guo; Ziyang Chen; Shaoguang Wang; Jianxiang He; Yijie Xu; Jinhui Ye; Ying Sun; Hui Xiong
>
> **备注:** 32 pages, under review
>
> **摘要:** Understanding long video content is a complex endeavor that often relies on densely sampled frame captions or end-to-end feature selectors, yet these techniques commonly overlook the logical relationships between textual queries and visual elements. In practice, computational constraints necessitate coarse frame subsampling, a challenge analogous to "finding a needle in a haystack." To address this issue, we introduce a semantics-driven search framework that reformulates keyframe selection under the paradigm of Visual Semantic-Logical Search. Specifically, we systematically define four fundamental logical dependencies: 1) spatial co-occurrence, 2) temporal proximity, 3) attribute dependency, and 4) causal order. These relations dynamically update frame sampling distributions through an iterative refinement process, enabling context-aware identification of semantically critical frames tailored to specific query requirements. Our method establishes new SOTA performance on the manually annotated benchmark in key-frame selection metrics. Furthermore, when applied to downstream video question-answering tasks, the proposed approach demonstrates the best performance gains over existing methods on LongVideoBench and Video-MME, validating its effectiveness in bridging the logical gap between textual queries and visual-temporal reasoning. The code will be publicly available.
>
---
#### [replaced 147] Geometric Framework for Cell Oversegmentation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01890v2](http://arxiv.org/pdf/2502.01890v2)**

> **作者:** Peter Chen; Bryan Chang; Olivia Annette Creasey; Julie Beth Sneddon; Zev Gartner; Yining Liu
>
> **备注:** 22 pages
>
> **摘要:** 3D cell segmentation methods are often hindered by \emph{oversegmentation}, where a single cell is incorrectly split into multiple fragments. This degrades the final segmentation quality and is notoriously difficult to resolve, as oversegmentation errors often resemble \emph{natural gaps} between adjacent cells. Our work makes two key contributions. First, for 3D cell segmentation, we are the first work to formulate oversegmentation as a concrete problem and propose a geometric framework to identify and correct these errors. Our approach builds a pre-trained classifier using both 2D geometric and 3D topological features extracted from flawed 3D segmentation results. Second, we introduce a novel metric, \emph{Geo-Wasserstein} divergence, to quantify changes in 2D geometries. This captures the evolving trends in cell mask shape changes in a geometry-aware manner. We validate our method through extensive experiments on in-domain plant datasets, including both synthesized and real cases, as well as on out-of-domain animal datasets to demonstrate transfer learning performance. An ablation study further highlights the contribution of the \emph{Geo-Wasserstein} divergence. A clear pipeline is provided for end-users to build pre-trained models to any labeled dataset.
>
---
#### [replaced 148] WMCopier: Forging Invisible Image Watermarks on Arbitrary Images
- **分类: cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22330v2](http://arxiv.org/pdf/2503.22330v2)**

> **作者:** Ziping Dong; Chao Shuai; Zhongjie Ba; Peng Cheng; Zhan Qin; Qinglong Wang; Kui Ren
>
> **摘要:** Invisible Image Watermarking is crucial for ensuring content provenance and accountability in generative AI. While Gen-AI providers are increasingly integrating invisible watermarking systems, the robustness of these schemes against forgery attacks remains poorly characterized. This is critical, as forging traceable watermarks onto illicit content leads to false attribution, potentially harming the reputation and legal standing of Gen-AI service providers who are not responsible for the content. In this work, we propose WMCopier, an effective watermark forgery attack that operates without requiring any prior knowledge of or access to the target watermarking algorithm. Our approach first models the target watermark distribution using an unconditional diffusion model, and then seamlessly embeds the target watermark into a non-watermarked image via a shallow inversion process. We also incorporate an iterative optimization procedure that refines the reconstructed image to further trade off the fidelity and forgery efficiency. Experimental results demonstrate that WMCopier effectively deceives both open-source and closed-source watermark systems (e.g., Amazon's system), achieving a significantly higher success rate than existing methods. Additionally, we evaluate the robustness of forged samples and discuss the potential defenses against our attack.
>
---
#### [replaced 149] Latent Action Learning Requires Supervision in the Presence of Distractors
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.00379v3](http://arxiv.org/pdf/2502.00379v3)**

> **作者:** Alexander Nikulin; Ilya Zisman; Denis Tarasov; Nikita Lyubaykin; Andrei Polubarov; Igor Kiselev; Vladislav Kurenkov
>
> **备注:** ICML 2025, Poster, Source code: https://github.com/dunnolab/laom
>
> **摘要:** Recently, latent action learning, pioneered by Latent Action Policies (LAPO), have shown remarkable pre-training efficiency on observation-only data, offering potential for leveraging vast amounts of video available on the web for embodied AI. However, prior work has focused on distractor-free data, where changes between observations are primarily explained by ground-truth actions. Unfortunately, real-world videos contain action-correlated distractors that may hinder latent action learning. Using Distracting Control Suite (DCS) we empirically investigate the effect of distractors on latent action learning and demonstrate that LAPO struggle in such scenario. We propose LAOM, a simple LAPO modification that improves the quality of latent actions by 8x, as measured by linear probing. Importantly, we show that providing supervision with ground-truth actions, as few as 2.5% of the full dataset, during latent action learning improves downstream performance by 4.2x on average. Our findings suggest that integrating supervision during Latent Action Models (LAM) training is critical in the presence of distractors, challenging the conventional pipeline of first learning LAM and only then decoding from latent to ground-truth actions.
>
---
#### [replaced 150] UniCMs: A Unified Consistency Model For Efficient Multimodal Generation and Understanding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.05415v2](http://arxiv.org/pdf/2502.05415v2)**

> **作者:** Chenkai Xu; Xu Wang; Zhenyi Liao; Yishun Li; Tianqi Hou; Zhijie Deng
>
> **摘要:** Consistency models (CMs) have shown promise in the efficient generation of both image and text. This raises the natural question of whether we can learn a unified CM for efficient multimodal generation (e.g., text-to-image) and understanding (e.g., image-to-text). Intuitively, such a model could be acquired by applying the consistency distillation (CD) to existing unified multimodal models. However, the key challenge is establishing a unified denoising perspective for both image and text generation, which is essential for establishing the consistency mapping. To tackle this, at the representation level, we advocate for discrete tokens for both modalities to best preserve language modeling capabilities. Critically, instead of defining the text denoising trajectory via recent discrete diffusion language modeling principles, we specify it using the parallel decoding trace of an autoregressive language model, benefiting from the latter's superior performance in general text generation tasks. The denoising trajectory of image tokens adheres to standard discrete diffusion. We train our unified consistency models (UniCMs) on these combined multimodal trajectories simultaneously with a unified objective. We introduce a trajectory segmentation strategy to further improve the training convergence. Empirically, in text-to-image generation, UniCMs outperform SD3 on GenEval, Image Reward, and CLIP Score metrics, while requiring only approximately ${1}/{8}$ of the sampling time. Meanwhile, in image-to-text generation, UniCMs surpass Show-o on the MMMU benchmark while being $1.5 \times$ faster at long-sequence generating speed. The code is available at https://github.com/zhijie-group/UniCMs.
>
---
#### [replaced 151] FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Pretraining
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11192v2](http://arxiv.org/pdf/2505.11192v2)**

> **作者:** Myunsoo Kim; Seong-Woong Shim; Byung-Jun Lee
>
> **备注:** The manuscript contains errors that require substantial revision
>
> **摘要:** False negatives pose a critical challenge in vision-language pretraining (VLP) due to the many-to-many correspondence between images and texts in large-scale datasets. These false negatives introduce conflicting supervision signals that degrade the learned embedding space and diminish the effectiveness of hard negative sampling. In this paper, we propose FALCON (False-negative Aware Learning of COntrastive Negatives), a learning-based mini-batch construction strategy that adaptively balances the trade-off between hard and false negatives during VLP. Rather than relying on fixed heuristics, FALCON employs a negative mining scheduler that dynamically selects negative samples of appropriate hardness for each anchor instance during mini-batch construction, guided by a proxy for cross-modal alignment improvement. Experimental results demonstrate that FALCON significantly improves performance across two widely adopted VLP frameworks (ALBEF, BLIP-2) and a broad range of downstream tasks and evaluation settings, underscoring its effectiveness and robustness in mitigating the impact of false negatives.
>
---
#### [replaced 152] Mamba-MOC: A Multicategory Remote Object Counting via State Space Model
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.06697v2](http://arxiv.org/pdf/2501.06697v2)**

> **作者:** Peng Liu; Sen Lei; Heng-Chao Li
>
> **摘要:** Multicategory remote object counting is a fundamental task in computer vision, aimed at accurately estimating the number of objects of various categories in remote images. Existing methods rely on CNNs and Transformers, but CNNs struggle to capture global dependencies, and Transformers are computationally expensive, which limits their effectiveness in remote applications. Recently, Mamba has emerged as a promising solution in the field of computer vision, offering a linear complexity for modeling global dependencies. To this end, we propose Mamba-MOC, a mamba-based network designed for multi-category remote object counting, which represents the first application of Mamba to remote sensing object counting. Specifically, we propose a cross-scale interaction module to facilitate the deep integration of hierarchical features. Then we design a context state space model to capture both global and local contextual information and provide local neighborhood information during the scan process. Experimental results in large-scale realistic scenarios demonstrate that our proposed method achieves state-of-the-art performance compared with some mainstream counting algorithms.
>
---
#### [replaced 153] Learning to Learn Weight Generation via Local Consistency Diffusion
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.01117v3](http://arxiv.org/pdf/2502.01117v3)**

> **作者:** Yunchuan Guan; Yu Liu; Ke Zhou; Zhiqi Shen; Jenq-Neng Hwang; Lei Li
>
> **摘要:** Diffusion-based algorithms have emerged as promising techniques for weight generation. However, existing solutions are limited by two challenges: generalizability and local target assignment. The former arises from the inherent lack of cross-task transferability in existing single-level optimization methods, limiting the model's performance on new tasks. The latter lies in existing research modeling only global optimal weights, neglecting the supervision signals in local target weights. Moreover, naively assigning local target weights causes local-global inconsistency. To address these issues, we propose Mc-Di, which integrates the diffusion algorithm with meta-learning for better generalizability. Furthermore, we extend the vanilla diffusion into a local consistency diffusion algorithm. Our theory and experiments demonstrate that it can learn from local targets while maintaining consistency with the global optima. We validate Mc-Di's superior accuracy and inference efficiency in tasks that require frequent weight updates, including transfer learning, few-shot learning, domain generalization, and large language model adaptation.
>
---
#### [replaced 154] MoDGS: Dynamic Gaussian Splatting from Casually-captured Monocular Videos with Depth Priors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.00434v3](http://arxiv.org/pdf/2406.00434v3)**

> **作者:** Qingming Liu; Yuan Liu; Jiepeng Wang; Xianqiang Lyv; Peng Wang; Wenping Wang; Junhui Hou
>
> **备注:** Accepted as a poster at ICLR. Project page: https://modgs.github.io
>
> **摘要:** In this paper, we propose MoDGS, a new pipeline to render novel views of dy namic scenes from a casually captured monocular video. Previous monocular dynamic NeRF or Gaussian Splatting methods strongly rely on the rapid move ment of input cameras to construct multiview consistency but struggle to recon struct dynamic scenes on casually captured input videos whose cameras are either static or move slowly. To address this challenging task, MoDGS adopts recent single-view depth estimation methods to guide the learning of the dynamic scene. Then, a novel 3D-aware initialization method is proposed to learn a reasonable deformation field and a new robust depth loss is proposed to guide the learning of dynamic scene geometry. Comprehensive experiments demonstrate that MoDGS is able to render high-quality novel view images of dynamic scenes from just a casually captured monocular video, which outperforms state-of-the-art meth ods by a significant margin. The code will be publicly available.
>
---
#### [replaced 155] Offboard Occupancy Refinement with Hybrid Propagation for Autonomous Driving
- **分类: cs.CV; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2403.08504v4](http://arxiv.org/pdf/2403.08504v4)**

> **作者:** Hao Shi; Song Wang; Jiaming Zhang; Xiaoting Yin; Guangming Wang; Jianke Zhu; Kailun Yang; Kaiwei Wang
>
> **备注:** Accepted to IEEE Transactions on Intelligent Transportation Systems (T-ITS). The source code will be made publicly available at https://github.com/MasterHow/OccFiner
>
> **摘要:** Vision-based occupancy prediction, also known as 3D Semantic Scene Completion (SSC), presents a significant challenge in computer vision. Previous methods, confined to onboard processing, struggle with simultaneous geometric and semantic estimation, continuity across varying viewpoints, and single-view occlusion. Our paper introduces OccFiner, a novel offboard framework designed to enhance the accuracy of vision-based occupancy predictions. OccFiner operates in two hybrid phases: 1) a multi-to-multi local propagation network that implicitly aligns and processes multiple local frames for correcting onboard model errors and consistently enhancing occupancy accuracy across all distances. 2) the region-centric global propagation, focuses on refining labels using explicit multi-view geometry and integrating sensor bias, particularly for increasing the accuracy of distant occupied voxels. Extensive experiments demonstrate that OccFiner improves both geometric and semantic accuracy across various types of coarse occupancy, setting a new state-of-the-art performance on the SemanticKITTI dataset. Notably, OccFiner significantly boosts the performance of vision-based SSC models, achieving accuracy levels competitive with established LiDAR-based onboard SSC methods. Furthermore, OccFiner is the first to achieve automatic annotation of SSC in a purely vision-based approach. Quantitative experiments prove that OccFiner successfully facilitates occupancy data loop-closure in autonomous driving. Additionally, we quantitatively and qualitatively validate the superiority of the offboard approach on city-level SSC static maps. The source code will be made publicly available at https://github.com/MasterHow/OccFiner.
>
---
#### [replaced 156] Vision Transformers in Precision Agriculture: A Comprehensive Survey
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.21706v2](http://arxiv.org/pdf/2504.21706v2)**

> **作者:** Saber Mehdipour; Seyed Abolghasem Mirroshandel; Seyed Amirhossein Tabatabaei
>
> **摘要:** Detecting plant diseases is a crucial aspect of modern agriculture, as it plays a key role in maintaining crop health and increasing overall yield. Traditional approaches, though still valuable, often rely on manual inspection or conventional machine learning techniques, both of which face limitations in scalability and accuracy. Recently, Vision Transformers (ViTs) have emerged as a promising alternative, offering advantages such as improved handling of long-range dependencies and better scalability for visual tasks. This review explores the application of ViTs in precision agriculture, covering a range of tasks. We begin by introducing the foundational architecture of ViTs and discussing their transition from Natural Language Processing (NLP) to Computer Vision. The discussion includes the concept of inductive bias in traditional models like Convolutional Neural Networks (CNNs), and how ViTs mitigate these biases. We provide a comprehensive review of recent literature, focusing on key methodologies, datasets, and performance metrics. This study also includes a comparative analysis of CNNs and ViTs, along with a review of hybrid models and performance enhancements. Technical challenges such as data requirements, computational demands, and model interpretability are addressed, along with potential solutions. Finally, we outline future research directions and technological advancements that could further support the integration of ViTs in real-world agricultural settings. Our goal with this study is to offer practitioners and researchers a deeper understanding of how ViTs are poised to transform smart and precision agriculture.
>
---
#### [replaced 157] Unsupervised Multi-Parameter Inverse Solving for Reducing Ring Artifacts in 3D X-Ray CBCT
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.05853v2](http://arxiv.org/pdf/2412.05853v2)**

> **作者:** Qing Wu; Hongjiang Wei; Jingyi Yu; Yuyao Zhang
>
> **备注:** 15 pages
>
> **摘要:** Ring artifacts are prevalent in 3D cone-beam computed tomography (CBCT) due to non-ideal responses of X-ray detectors, substantially affecting image quality and diagnostic reliability. Existing state-of-the-art (SOTA) ring artifact reduction (RAR) methods rely on supervised learning with large-scale paired CT datasets. While effective in-domain, supervised methods tend to struggle to fully capture the physical characteristics of ring artifacts, leading to pronounced performance drops in complex real-world acquisitions. Moreover, their scalability to 3D CBCT is limited by high memory demands. In this work, we propose Riner, a new unsupervised RAR method. Based on a theoretical analysis of ring artifact formation, we reformulate RAR as a multi-parameter inverse problem, where the non-ideal responses of X-ray detectors are parameterized as solvable physical variables. Using a new differentiable forward model, Riner can jointly learn the implicit neural representation of artifact-free images and estimate the physical parameters directly from CT measurements, without external training data. Additionally, Riner is memory-friendly due to its ray-based optimization, enhancing its usability in large-scale 3D CBCT. Experiments on both simulated and real-world datasets show Riner outperforms existing SOTA supervised methods.
>
---
#### [replaced 158] What's Inside Your Diffusion Model? A Score-Based Riemannian Metric to Explore the Data Manifold
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11128v2](http://arxiv.org/pdf/2505.11128v2)**

> **作者:** Simone Azeglio; Arianna Di Bernardo
>
> **摘要:** Recent advances in diffusion models have demonstrated their remarkable ability to capture complex image distributions, but the geometric properties of the learned data manifold remain poorly understood. We address this gap by introducing a score-based Riemannian metric that leverages the Stein score function from diffusion models to characterize the intrinsic geometry of the data manifold without requiring explicit parameterization. Our approach defines a metric tensor in the ambient space that stretches distances perpendicular to the manifold while preserving them along tangential directions, effectively creating a geometry where geodesics naturally follow the manifold's contours. We develop efficient algorithms for computing these geodesics and demonstrate their utility for both interpolation between data points and extrapolation beyond the observed data distribution. Through experiments on synthetic data with known geometry, Rotated MNIST, and complex natural images via Stable Diffusion, we show that our score-based geodesics capture meaningful transformations that respect the underlying data distribution. Our method consistently outperforms baseline approaches on perceptual metrics (LPIPS) and distribution-level metrics (FID, KID), producing smoother, more realistic image transitions. These results reveal the implicit geometric structure learned by diffusion models and provide a principled way to navigate the manifold of natural images through the lens of Riemannian geometry.
>
---
