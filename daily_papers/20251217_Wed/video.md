# 计算机视觉 cs.CV

- **最新发布 132 篇**

- **更新 68 篇**

## 最新发布

#### [new 001] Human-AI Collaboration Mechanism Study on AIGC Assisted Image Production for Special Coverage
- **分类: cs.CV; cs.AI**

- **简介: 该论文属新闻传播与AIGC交叉任务，旨在解决AIGC图像在特殊报道中因“黑箱”导致的失真、失实与信任危机。通过跨平台实验分析偏差根源，并构建人机协同模块化管线，提出CIS、CEA、U-PA三维度评估机制。**

- **链接: [https://arxiv.org/pdf/2512.13739v1](https://arxiv.org/pdf/2512.13739v1)**

> **作者:** Yajie Yang; Yuqing Zhao; Xiaochao Xi; Yinan Zhu
>
> **备注:** AAAI-AISI 2026
>
> **摘要:** Artificial Intelligence Generated Content (AIGC) assisting image production triggers controversy in journalism while attracting attention from media agencies. Key issues involve misinformation, authenticity, semantic fidelity, and interpretability. Most AIGC tools are opaque "black boxes," hindering the dual demands of content accuracy and semantic alignment and creating ethical, sociotechnical, and trust dilemmas. This paper explores pathways for controllable image production in journalism's special coverage and conducts two experiments with projects from China's media agency: (1) Experiment 1 tests cross-platform adaptability via standardized prompts across three scenes, revealing disparities in semantic alignment, cultural specificity, and visual realism driven by training-corpus bias and platform-level filtering. (2) Experiment 2 builds a human-in-the-loop modular pipeline combining high-precision segmentation (SAM, GroundingDINO), semantic alignment (BrushNet), and style regulating (Style-LoRA, Prompt-to-Prompt), ensuring editorial fidelity through CLIP-based semantic scoring, NSFW/OCR/YOLO filtering, and verifiable content credentials. Traceable deployment preserves semantic representation. Consequently, we propose a human-AI collaboration mechanism for AIGC assisted image production in special coverage and recommend evaluating Character Identity Stability (CIS), Cultural Expression Accuracy (CEA), and User-Public Appropriateness (U-PA).
>
---
#### [new 002] Improving Semantic Uncertainty Quantification in LVLMs with Semantic Gaussian Processes
- **分类: cs.CV**

- **简介: 该论文属不确定性量化任务，旨在解决LVLMs语义不确定性估计不可靠问题。提出SGPU方法：将答案嵌入语义空间，通过Gram矩阵的特征谱表征语义分布，并用高斯过程分类器建模语义一致性与不确定性关系，在多模型多数据集上实现SOTA校准与判别性能。**

- **链接: [https://arxiv.org/pdf/2512.14177v1](https://arxiv.org/pdf/2512.14177v1)**

> **作者:** Joseph Hoche; Andrei Bursuc; David Brellmann; Gilles Louppe; Pavel Izmailov; Angela Yao; Gianni Franchi
>
> **摘要:** Large Vision-Language Models (LVLMs) often produce plausible but unreliable outputs, making robust uncertainty estimation essential. Recent work on semantic uncertainty estimates relies on external models to cluster multiple sampled responses and measure their semantic consistency. However, these clustering methods are often fragile, highly sensitive to minor phrasing variations, and can incorrectly group or separate semantically similar answers, leading to unreliable uncertainty estimates. We propose Semantic Gaussian Process Uncertainty (SGPU), a Bayesian framework that quantifies semantic uncertainty by analyzing the geometric structure of answer embeddings, avoiding brittle clustering. SGPU maps generated answers into a dense semantic space, computes the Gram matrix of their embeddings, and summarizes their semantic configuration via the eigenspectrum. This spectral representation is then fed into a Gaussian Process Classifier that learns to map patterns of semantic consistency to predictive uncertainty, and that can be applied in both black-box and white-box settings. Across six LLMs and LVLMs on eight datasets spanning VQA, image classification, and textual QA, SGPU consistently achieves state-of-the-art calibration (ECE) and discriminative (AUROC, AUARC) performance. We further show that SGPU transfers across models and modalities, indicating that its spectral representation captures general patterns of semantic uncertainty.
>
---
#### [new 003] MemFlow: Flowing Adaptive Memory for Consistent and Efficient Long Video Narratives
- **分类: cs.CV**

- **简介: 该论文面向流式长视频生成任务，解决历史帧记忆固定导致的长程内容不一致问题。提出MemFlow方法：动态检索与当前文本提示最相关的历史帧更新记忆库，并在注意力中稀疏激活相关token，兼顾一致性与高效性。**

- **链接: [https://arxiv.org/pdf/2512.14699v1](https://arxiv.org/pdf/2512.14699v1)**

> **作者:** Sihui Ji; Xi Chen; Shuai Yang; Xin Tao; Pengfei Wan; Hengshuang Zhao
>
> **备注:** Project Page: https://sihuiji.github.io/MemFlow.github.io/
>
> **摘要:** The core challenge for streaming video generation is maintaining the content consistency in long context, which poses high requirement for the memory design. Most existing solutions maintain the memory by compressing historical frames with predefined strategies. However, different to-generate video chunks should refer to different historical cues, which is hard to satisfy with fixed strategies. In this work, we propose MemFlow to address this problem. Specifically, before generating the coming chunk, we dynamically update the memory bank by retrieving the most relevant historical frames with the text prompt of this chunk. This design enables narrative coherence even if new event happens or scenario switches in future frames. In addition, during generation, we only activate the most relevant tokens in the memory bank for each query in the attention layers, which effectively guarantees the generation efficiency. In this way, MemFlow achieves outstanding long-context consistency with negligible computation burden (7.9% speed reduction compared with the memory-free baseline) and keeps the compatibility with any streaming video generation model with KV cache.
>
---
#### [new 004] OmniGen: Unified Multimodal Sensor Generation for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属自动驾驶多模态传感器数据生成任务，旨在解决真实数据采集成本高、corner-case稀缺及现有单模态生成导致的模态不一致问题。提出OmniGen框架，基于统一BEV空间与UAE体积渲染解码器，联合生成对齐的LiDAR与多视图图像，并引入可控DiT+ControlNet实现灵活生成。**

- **链接: [https://arxiv.org/pdf/2512.14225v1](https://arxiv.org/pdf/2512.14225v1)**

> **作者:** Tao Tang; Enhui Ma; xia zhou; Letian Wang; Tianyi Yan; Xueyang Zhang; Kun Zhan; Peng Jia; XianPeng Lang; Jia-Wang Bian; Kaicheng Yu; Xiaodan Liang
>
> **备注:** ACM MM 2025
>
> **摘要:** Autonomous driving has seen remarkable advancements, largely driven by extensive real-world data collection. However, acquiring diverse and corner-case data remains costly and inefficient. Generative models have emerged as a promising solution by synthesizing realistic sensor data. However, existing approaches primarily focus on single-modality generation, leading to inefficiencies and misalignment in multimodal sensor data. To address these challenges, we propose OminiGen, which generates aligned multimodal sensor data in a unified framework. Our approach leverages a shared Bird\u2019s Eye View (BEV) space to unify multimodal features and designs a novel generalizable multimodal reconstruction method, UAE, to jointly decode LiDAR and multi-view camera data. UAE achieves multimodal sensor decoding through volume rendering, enabling accurate and flexible reconstruction. Furthermore, we incorporate a Diffusion Transformer (DiT) with a ControlNet branch to enable controllable multimodal sensor generation. Our comprehensive experiments demonstrate that OminiGen achieves desired performances in unified multimodal sensor data generation with multimodal consistency and flexible sensor adjustments.
>
---
#### [new 005] ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body
- **分类: cs.CV**

- **简介: 论文提出ViBES，一种行为智能的3D对话代理，解决现有系统将语言、语音、动作割裂建模导致的时序僵硬、社交弱等问题。它构建多模态专家混合（MoME）模型，联合规划与生成语音、表情和身体动作，支持混合交互与可控行为，提升多轮对话中动作与语言的对齐与社会性。**

- **链接: [https://arxiv.org/pdf/2512.14234v1](https://arxiv.org/pdf/2512.14234v1)**

> **作者:** Juze Zhang; Changan Chen; Xin Chen; Heng Yu; Tiange Xiang; Ali Sartaz Khan; Shrinidhi K. Lakshmikanth; Ehsan Adeli
>
> **备注:** Project page: https://ai.stanford.edu/~juze/ViBES/
>
> **摘要:** Human communication is inherently multimodal and social: words, prosody, and body language jointly carry intent. Yet most prior systems model human behavior as a translation task co-speech gesture or text-to-motion that maps a fixed utterance to motion clips-without requiring agentic decision-making about when to move, what to do, or how to adapt across multi-turn dialogue. This leads to brittle timing, weak social grounding, and fragmented stacks where speech, text, and motion are trained or inferred in isolation. We introduce ViBES (Voice in Behavioral Expression and Synchrony), a conversational 3D agent that jointly plans language and movement and executes dialogue-conditioned body actions. Concretely, ViBES is a speech-language-behavior (SLB) model with a mixture-of-modality-experts (MoME) backbone: modality-partitioned transformer experts for speech, facial expression, and body motion. The model processes interleaved multimodal token streams with hard routing by modality (parameters are split per expert), while sharing information through cross-expert attention. By leveraging strong pretrained speech-language models, the agent supports mixed-initiative interaction: users can speak, type, or issue body-action directives mid-conversation, and the system exposes controllable behavior hooks for streaming responses. We further benchmark on multi-turn conversation with automatic metrics of dialogue-motion alignment and behavior quality, and observe consistent gains over strong co-speech and text-to-motion baselines. ViBES goes beyond "speech-conditioned motion generation" toward agentic virtual bodies where language, prosody, and movement are jointly generated, enabling controllable, socially competent 3D interaction. Code and data will be made available at: ai.stanford.edu/~juze/ViBES/
>
---
#### [new 006] A Multicenter Benchmark of Multiple Instance Learning Models for Lymphoma Subtyping from HE-stained Whole Slide Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向淋巴瘤亚型分类任务，旨在解决HE染色全切片图像跨中心泛化能力弱的问题。作者构建首个多中心淋巴瘤基准数据集，系统评估5种病理基础模型与两种MIL聚合器在不同放大倍率下的性能，发现40x足够且泛化性仍不足。**

- **链接: [https://arxiv.org/pdf/2512.14640v1](https://arxiv.org/pdf/2512.14640v1)**

> **作者:** Rao Muhammad Umer; Daniel Sens; Jonathan Noll; Christian Matek; Lukas Wolfseher; Rainer Spang; Ralf Huss; Johannes Raffler; Sarah Reinke; Wolfram Klapper; Katja Steiger; Kristina Schwamborn; Carsten Marr
>
> **备注:** 17 pages
>
> **摘要:** Timely and accurate lymphoma diagnosis is essential for guiding cancer treatment. Standard diagnostic practice combines hematoxylin and eosin (HE)-stained whole slide images with immunohistochemistry, flow cytometry, and molecular genetic tests to determine lymphoma subtypes, a process requiring costly equipment, skilled personnel, and causing treatment delays. Deep learning methods could assist pathologists by extracting diagnostic information from routinely available HE-stained slides, yet comprehensive benchmarks for lymphoma subtyping on multicenter data are lacking. In this work, we present the first multicenter lymphoma benchmarking dataset covering four common lymphoma subtypes and healthy control tissue. We systematically evaluate five publicly available pathology foundation models (H-optimus-1, H0-mini, Virchow2, UNI2, Titan) combined with attention-based (AB-MIL) and transformer-based (TransMIL) multiple instance learning aggregators across three magnifications (10x, 20x, 40x). On in-distribution test sets, models achieve multiclass balanced accuracies exceeding 80% across all magnifications, with all foundation models performing similarly and both aggregation methods showing comparable results. The magnification study reveals that 40x resolution is sufficient, with no performance gains from higher resolutions or cross-magnification aggregation. However, on out-of-distribution test sets, performance drops substantially to around 60%, highlighting significant generalization challenges. To advance the field, larger multicenter studies covering additional rare lymphoma subtypes are needed. We provide an automated benchmarking pipeline to facilitate such future research.
>
---
#### [new 007] Semantic Mismatch and Perceptual Degradation: A New Perspective on Image Editing Immunity
- **分类: cs.CV; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属图像免疫任务，旨在防御扩散模型的恶意文本引导编辑。提出SIFM方法，通过扰动中间特征破坏语义对齐并引入感知退化；并设计ISR新指标，用MLLM评估免疫成功率，实现SOTA防护效果。**

- **链接: [https://arxiv.org/pdf/2512.14320v1](https://arxiv.org/pdf/2512.14320v1)**

> **作者:** Shuai Dong; Jie Zhang; Guoying Zhao; Shiguang Shan; Xilin Chen
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Text-guided image editing via diffusion models, while powerful, raises significant concerns about misuse, motivating efforts to immunize images against unauthorized edits using imperceptible perturbations. Prevailing metrics for evaluating immunization success typically rely on measuring the visual dissimilarity between the output generated from a protected image and a reference output generated from the unprotected original. This approach fundamentally overlooks the core requirement of image immunization, which is to disrupt semantic alignment with attacker intent, regardless of deviation from any specific output. We argue that immunization success should instead be defined by the edited output either semantically mismatching the prompt or suffering substantial perceptual degradations, both of which thwart malicious intent. To operationalize this principle, we propose Synergistic Intermediate Feature Manipulation (SIFM), a method that strategically perturbs intermediate diffusion features through dual synergistic objectives: (1) maximizing feature divergence from the original edit trajectory to disrupt semantic alignment with the expected edit, and (2) minimizing feature norms to induce perceptual degradations. Furthermore, we introduce the Immunization Success Rate (ISR), a novel metric designed to rigorously quantify true immunization efficacy for the first time. ISR quantifies the proportion of edits where immunization induces either semantic failure relative to the prompt or significant perceptual degradations, assessed via Multimodal Large Language Models (MLLMs). Extensive experiments show our SIFM achieves the state-of-the-art performance for safeguarding visual content against malicious diffusion-based manipulation.
>
---
#### [new 008] TorchTraceAP: A New Benchmark Dataset for Detecting Performance Anti-Patterns in Computer Vision Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出TorchTraceAP基准数据集，旨在解决CV模型性能反模式检测难的问题。它构建了600+跨模型、跨硬件的PyTorch执行轨迹，提出“轻量模型初筛+LLM细粒度分类”的两阶段方法，显著优于传统无监督与规则方法。**

- **链接: [https://arxiv.org/pdf/2512.14141v1](https://arxiv.org/pdf/2512.14141v1)**

> **作者:** Hanning Chen; Keyu Man; Kevin Zhu; Chenguang Zhu; Haonan Li; Tongbo Luo; Xizhou Feng; Wei Sun; Sreen Tallam; Mohsen Imani; Partha Kanuparthy
>
> **摘要:** Identifying and addressing performance anti-patterns in machine learning (ML) models is critical for efficient training and inference, but it typically demands deep expertise spanning system infrastructure, ML models and kernel development. While large tech companies rely on dedicated ML infrastructure engineers to analyze torch traces and benchmarks, such resource-intensive workflows are largely inaccessible to computer vision researchers in general. Among the challenges, pinpointing problematic trace segments within lengthy execution traces remains the most time-consuming task, and is difficult to automate with current ML models, including LLMs. In this work, we present the first benchmark dataset specifically designed to evaluate and improve ML models' ability to detect anti patterns in traces. Our dataset contains over 600 PyTorch traces from diverse computer vision models classification, detection, segmentation, and generation collected across multiple hardware platforms. We also propose a novel iterative approach: a lightweight ML model first detects trace segments with anti patterns, followed by a large language model (LLM) for fine grained classification and targeted feedback. Experimental results demonstrate that our method significantly outperforms unsupervised clustering and rule based statistical techniques for detecting anti pattern regions. Our method also effectively compensates LLM's limited context length and reasoning inefficiencies.
>
---
#### [new 009] MoLingo: Motion-Language Alignment for Text-to-Motion Generation
- **分类: cs.CV**

- **简介: 该论文属文本生成人体运动任务，旨在提升文本到运动（T2M）生成的质量与对齐度。提出MoLingo模型：构建语义对齐的运动隐空间，采用多token交叉注意力进行文本条件注入，并结合自回归隐式扩散，显著提升运动真实性和文本-动作一致性。**

- **链接: [https://arxiv.org/pdf/2512.13840v1](https://arxiv.org/pdf/2512.13840v1)**

> **作者:** Yannan He; Garvita Tiwari; Xiaohan Zhang; Pankaj Bora; Tolga Birdal; Jan Eric Lenssen; Gerard Pons-Moll
>
> **备注:** Project page: https://hynann.github.io/molingo/MoLingo.html
>
> **摘要:** We introduce MoLingo, a text-to-motion (T2M) model that generates realistic, lifelike human motion by denoising in a continuous latent space. Recent works perform latent space diffusion, either on the whole latent at once or auto-regressively over multiple latents. In this paper, we study how to make diffusion on continuous motion latents work best. We focus on two questions: (1) how to build a semantically aligned latent space so diffusion becomes more effective, and (2) how to best inject text conditioning so the motion follows the description closely. We propose a semantic-aligned motion encoder trained with frame-level text labels so that latents with similar text meaning stay close, which makes the latent space more diffusion-friendly. We also compare single-token conditioning with a multi-token cross-attention scheme and find that cross-attention gives better motion realism and text-motion alignment. With semantically aligned latents, auto-regressive generation, and cross-attention text conditioning, our model sets a new state of the art in human motion generation on standard metrics and in a user study. We will release our code and models for further research and downstream usage.
>
---
#### [new 010] Quality-Aware Framework for Video-Derived Respiratory Signals
- **分类: cs.CV; eess.SP**

- **简介: 该论文面向视频呼吸率（RR）估计任务，解决信号质量不一致导致的不可靠问题。提出质量感知框架：融合10种异构呼吸信号，用4种谱估计算法分析，训练模型预测段级质量并自适应融合或过滤，显著降低RR误差。**

- **链接: [https://arxiv.org/pdf/2512.14093v1](https://arxiv.org/pdf/2512.14093v1)**

> **作者:** Nhi Nguyen; Constantino Álvarez Casado; Le Nguyen; Manuel Lage Cañellas; Miguel Bordallo López
>
> **备注:** 6 pages, 1 figure, 2 tables, conference
>
> **摘要:** Video-based respiratory rate (RR) estimation is often unreliable due to inconsistent signal quality across extraction methods. We present a predictive, quality-aware framework that integrates heterogeneous signal sources with dynamic assessment of reliability. Ten signals are extracted from facial remote photoplethysmography (rPPG), upper-body motion, and deep learning pipelines, and analyzed using four spectral estimators: Welch's method, Multiple Signal Classification (MUSIC), Fast Fourier Transform (FFT), and peak detection. Segment-level quality indices are then used to train machine learning models that predict accuracy or select the most reliable signal. This enables adaptive signal fusion and quality-based segment filtering. Experiments on three public datasets (OMuSense-23, COHFACE, MAHNOB-HCI) show that the proposed framework achieves lower RR estimation errors than individual methods in most cases, with performance gains depending on dataset characteristics. These findings highlight the potential of quality-driven predictive modeling to deliver scalable and generalizable video-based respiratory monitoring solutions.
>
---
#### [new 011] Fracture Morphology Classification: Local Multiclass Modeling for Multilabel Complexity
- **分类: cs.CV**

- **简介: 该论文属医学图像分析任务，旨在自动识别儿童骨折形态。针对全局多标签分类难的问题，提出将AO编码分配至骨折检测框，转为局部多分类，提升F1分数7.89%；但对检测器误差敏感，影响实际部署。**

- **链接: [https://arxiv.org/pdf/2512.14196v1](https://arxiv.org/pdf/2512.14196v1)**

> **作者:** Cassandra Krause; Mattias P. Heinrich; Ron Keuth
>
> **备注:** Accepted as poster at the German Conference on Medical Image Computing 2026
>
> **摘要:** Between $15\,\%$ and $45\,\%$ of children experience a fracture during their growth years, making accurate diagnosis essential. Fracture morphology, alongside location and fragment angle, is a key diagnostic feature. In this work, we propose a method to extract fracture morphology by assigning automatically global AO codes to corresponding fracture bounding boxes. This approach enables the use of public datasets and reformulates the global multilabel task into a local multiclass one, improving the average F1 score by $7.89\,\%$. However, performance declines when using imperfect fracture detectors, highlighting challenges for real-world deployment. Our code is available on GitHub.
>
---
#### [new 012] WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出WorldPlay，属实时交互式世界建模任务，旨在解决长时序几何一致性与速度/内存权衡难题。通过双动作表征、重构上下文记忆和上下文强制蒸馏三项创新，实现720p/24FPS流式视频生成，兼顾实时性、长程一致性和泛化性。**

- **链接: [https://arxiv.org/pdf/2512.14614v1](https://arxiv.org/pdf/2512.14614v1)**

> **作者:** Wenqiang Sun; Haiyu Zhang; Haoyuan Wang; Junta Wu; Zehan Wang; Zhenwei Wang; Yunhong Wang; Jun Zhang; Tengfei Wang; Chunchao Guo
>
> **备注:** project page: https://3d-models.hunyuan.tencent.com/world/, demo: https://3d.hunyuan.tencent.com/sceneTo3D
>
> **摘要:** This paper presents WorldPlay, a streaming video diffusion model that enables real-time, interactive world modeling with long-term geometric consistency, resolving the trade-off between speed and memory that limits current methods. WorldPlay draws power from three key innovations. 1) We use a Dual Action Representation to enable robust action control in response to the user's keyboard and mouse inputs. 2) To enforce long-term consistency, our Reconstituted Context Memory dynamically rebuilds context from past frames and uses temporal reframing to keep geometrically important but long-past frames accessible, effectively alleviating memory attenuation. 3) We also propose Context Forcing, a novel distillation method designed for memory-aware model. Aligning memory context between the teacher and student preserves the student's capacity to use long-range information, enabling real-time speeds while preventing error drift. Taken together, WorldPlay generates long-horizon streaming 720p video at 24 FPS with superior consistency, comparing favorably with existing techniques and showing strong generalization across diverse scenes. Project page and online demo can be found: https://3d-models.hunyuan.tencent.com/world/ and https://3d.hunyuan.tencent.com/sceneTo3D.
>
---
#### [new 013] Enhancing Interpretability for Vision Models via Shapley Value Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属可解释人工智能任务，旨在解决深度视觉模型决策不透明问题。提出一种新自解释框架：训练时引入Shapley值估计作为辅助任务，公平分配预测分至图像块，使解释与模型逻辑一致，在几乎不损性能和兼容性前提下提升可解释性。**

- **链接: [https://arxiv.org/pdf/2512.14354v1](https://arxiv.org/pdf/2512.14354v1)**

> **作者:** Kanglong Fan; Yunqiao Yang; Chen Ma
>
> **备注:** Accepted to AAAI2026
>
> **摘要:** Deep neural networks have demonstrated remarkable performance across various domains, yet their decision-making processes remain opaque. Although many explanation methods are dedicated to bringing the obscurity of DNNs to light, they exhibit significant limitations: post-hoc explanation methods often struggle to faithfully reflect model behaviors, while self-explaining neural networks sacrifice performance and compatibility due to their specialized architectural designs. To address these challenges, we propose a novel self-explaining framework that integrates Shapley value estimation as an auxiliary task during training, which achieves two key advancements: 1) a fair allocation of the model prediction scores to image patches, ensuring explanations inherently align with the model's decision logic, and 2) enhanced interpretability with minor structural modifications, preserving model performance and compatibility. Extensive experiments on multiple benchmarks demonstrate that our method achieves state-of-the-art interpretability.
>
---
#### [new 014] From YOLO to VLMs: Advancing Zero-Shot and Few-Shot Detection of Wastewater Treatment Plants Using Satellite Imagery in MENA Region
- **分类: cs.CV; cs.AI**

- **简介: 该论文属遥感图像分类任务，旨在解决MENA地区污水处理厂（WWTP）自动识别需大量标注的问题。提出基于VLM的零样本/少样本方法，对比多种VLM与YOLOv8，在卫星图像上实现无需人工标注的高效WWTP识别。**

- **链接: [https://arxiv.org/pdf/2512.14312v1](https://arxiv.org/pdf/2512.14312v1)**

> **作者:** Akila Premarathna; Kanishka Hewageegana; Garcia Andarcia Mariangel
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** In regions of the Middle East and North Africa (MENA), there is a high demand for wastewater treatment plants (WWTPs), crucial for sustainable water management. Precise identification of WWTPs from satellite images enables environmental monitoring. Traditional methods like YOLOv8 segmentation require extensive manual labeling. But studies indicate that vision-language models (VLMs) are an efficient alternative to achieving equivalent or superior results through inherent reasoning and annotation. This study presents a structured methodology for VLM comparison, divided into zero-shot and few-shot streams specifically to identify WWTPs. The YOLOv8 was trained on a governmental dataset of 83,566 high-resolution satellite images from Egypt, Saudi Arabia, and UAE: ~85% WWTPs (positives), 15% non-WWTPs (negatives). Evaluated VLMs include LLaMA 3.2 Vision, Qwen 2.5 VL, DeepSeek-VL2, Gemma 3, Gemini, and Pixtral 12B (Mistral), used to identify WWTP components such as circular/rectangular tanks, aeration basins and distinguish confounders via expert prompts producing JSON outputs with confidence and descriptions. The dataset comprises 1,207 validated WWTP locations (198 UAE, 354 KSA, 655 Egypt) and equal non-WWTP sites from field/AI data, as 600mx600m Geo-TIFF images (Zoom 18, EPSG:4326). Zero-shot evaluations on WWTP images showed several VLMs out-performing YOLOv8's true positive rate, with Gemma-3 highest. Results confirm that VLMs, particularly with zero-shot, can replace YOLOv8 for efficient, annotation-free WWTP classification, enabling scalable remote sensing.
>
---
#### [new 015] Repurposing 2D Diffusion Models for 3D Shape Completion
- **分类: cs.CV**

- **简介: 该论文属3D形状补全任务，旨在解决3D扩散模型因数据稀缺和模态不匹配导致性能不足的问题。提出“Shape Atlas”将3D点云转为紧凑2D表示，复用预训练2D扩散模型实现高质量、细节保持的补全，并在PCN和ShapeNet-55上验证有效。**

- **链接: [https://arxiv.org/pdf/2512.13991v1](https://arxiv.org/pdf/2512.13991v1)**

> **作者:** Yao He; Youngjoong Kwon; Tiange Xiang; Wenxiao Cai; Ehsan Adeli
>
> **摘要:** We present a framework that adapts 2D diffusion models for 3D shape completion from incomplete point clouds. While text-to-image diffusion models have achieved remarkable success with abundant 2D data, 3D diffusion models lag due to the scarcity of high-quality 3D datasets and a persistent modality gap between 3D inputs and 2D latent spaces. To overcome these limitations, we introduce the Shape Atlas, a compact 2D representation of 3D geometry that (1) enables full utilization of the generative power of pretrained 2D diffusion models, and (2) aligns the modalities between the conditional input and output spaces, allowing more effective conditioning. This unified 2D formulation facilitates learning from limited 3D data and produces high-quality, detail-preserving shape completions. We validate the effectiveness of our results on the PCN and ShapeNet-55 datasets. Additionally, we show the downstream application of creating artist-created meshes from our completed point clouds, further demonstrating the practicality of our method.
>
---
#### [new 016] Adaptable Segmentation Pipeline for Diverse Brain Tumors with Radiomic-guided Subtyping and Lesion-Wise Model Ensemble
- **分类: cs.CV; eess.IV**

- **简介: 该论文属医学图像分割任务，旨在解决多类型脑肿瘤在多参数MRI上鲁棒、泛化分割难的问题。提出可适配的模块化流水线：基于放射组学引导亚型分类，结合病变级模型集成与定制后处理，在BraTS 2025多挑战数据集上实现高性能、架构无关的分割。**

- **链接: [https://arxiv.org/pdf/2512.14648v1](https://arxiv.org/pdf/2512.14648v1)**

> **作者:** Daniel Capellán-Martín; Abhijeet Parida; Zhifan Jiang; Nishad Kulkarni; Krithika Iyer; Austin Tapp; Syed Muhammad Anwar; María J. Ledesma-Carbayo; Marius George Linguraru
>
> **备注:** 12 pages, 5 figures, 3 tables. Algorithm presented at MICCAI BraTS 2025
>
> **摘要:** Robust and generalizable segmentation of brain tumors on multi-parametric magnetic resonance imaging (MRI) remains difficult because tumor types differ widely. The BraTS 2025 Lighthouse Challenge benchmarks segmentation methods on diverse high-quality datasets of adult and pediatric tumors: multi-consortium international pediatric brain tumor segmentation (PED), preoperative meningioma tumor segmentation (MEN), meningioma radiotherapy segmentation (MEN-RT), and segmentation of pre- and post-treatment brain metastases (MET). We present a flexible, modular, and adaptable pipeline that improves segmentation performance by selecting and combining state-of-the-art models and applying tumor- and lesion-specific processing before and after training. Radiomic features extracted from MRI help detect tumor subtype, ensuring a more balanced training. Custom lesion-level performance metrics determine the influence of each model in the ensemble and optimize post-processing that further refines the predictions, enabling the workflow to tailor every step to each case. On the BraTS testing sets, our pipeline achieved performance comparable to top-ranked algorithms across multiple challenges. These findings confirm that custom lesion-aware processing and model selection yield robust segmentations yet without locking the method to a specific network architecture. Our method has the potential for quantitative tumor measurement in clinical practice, supporting diagnosis and prognosis.
>
---
#### [new 017] ACE-SLAM: Scene Coordinate Regression for Neural Implicit Real-Time SLAM
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文提出ACE-SLAM，一种基于场景坐标回归（SCR）的神经隐式RGB-D实时SLAM方法。旨在解决神经隐式SLAM难以兼顾实时性、低内存与重定位效率的问题，首次将轻量SCR网络作为核心隐式地图表示，并设计专用架构与集成策略，实现严格实时、鲁棒动态场景下的SLAM。**

- **链接: [https://arxiv.org/pdf/2512.14032v1](https://arxiv.org/pdf/2512.14032v1)**

> **作者:** Ignacio Alzugaray; Marwan Taher; Andrew J. Davison
>
> **备注:** Project Page: https://github.com/ialzugaray/ace-slam
>
> **摘要:** We present a novel neural RGB-D Simultaneous Localization And Mapping (SLAM) system that learns an implicit map of the scene in real time. For the first time, we explore the use of Scene Coordinate Regression (SCR) as the core implicit map representation in a neural SLAM pipeline, a paradigm that trains a lightweight network to directly map 2D image features to 3D global coordinates. SCR networks provide efficient, low-memory 3D map representations, enable extremely fast relocalization, and inherently preserve privacy, making them particularly suitable for neural implicit SLAM. Our system is the first one to achieve strict real-time in neural implicit RGB-D SLAM by relying on a SCR-based representation. We introduce a novel SCR architecture specifically tailored for this purpose and detail the critical design choices required to integrate SCR into a live SLAM pipeline. The resulting framework is simple yet flexible, seamlessly supporting both sparse and dense features, and operates reliably in dynamic environments without special adaptation. We evaluate our approach on established synthetic and real-world benchmarks, demonstrating competitive performance against the state of the art. Project Page: https://github.com/ialzugaray/ace-slam
>
---
#### [new 018] XAI-Driven Diagnosis of Generalization Failure in State-Space Cerebrovascular Segmentation Models: A Case Study on Domain Shift Between RSNA and TopCoW Datasets
- **分类: cs.CV**

- **简介: 该论文属医学图像分割任务，旨在解决State-Space模型在跨数据集（RSNA→TopCoW）泛化失败问题。作者量化域差异后，用Seg-XRes-CAM分析注意力偏移，发现模型放弃真实血管特征、聚焦错误预测，揭示其依赖伪相关，验证XAI对诊断域偏移的有效性。**

- **链接: [https://arxiv.org/pdf/2512.13977v1](https://arxiv.org/pdf/2512.13977v1)**

> **作者:** Youssef Abuzeid; Shimaa El-Bana; Ahmad Al-Kabbany
>
> **摘要:** The clinical deployment of deep learning models in medical imaging is severely hindered by domain shift. This challenge, where a high-performing model fails catastrophically on external datasets, is a critical barrier to trustworthy AI. Addressing this requires moving beyond simple performance metrics toward deeper understanding, making Explainable AI (XAI) an essential diagnostic tool in medical image analysis. We present a rigorous, two-phase approach to diagnose the generalization failure of state-of-the-art State-Space Models (SSMs), specifically UMamaba, applied to cerebrovascular segmentation. We first established a quantifiable domain gap between our Source (RSNA CTA Aneurysm) and Target (TopCoW Circle of Willis CT) datasets, noting significant differences in Z-resolution and background noise. The model's Dice score subsequently plummeted from 0.8604 (Source) to 0.2902 (Target). In the second phase, which is our core contribution, we utilized Seg-XRes-CAM to diagnose the cause of this failure. We quantified the model's focus by measuring the overlap between its attention maps and the Ground Truth segmentations, and between its attention maps and its own Prediction Mask. Our analysis proves the model failed to generalize because its attention mechanism abandoned true anatomical features in the Target domain. Quantitative metrics confirm the model's focus shifted away from the Ground Truth vessels (IoU~0.101 at 0.3 threshold) while still aligning with its own wrong predictions (IoU~0.282 at 0.3 threshold). This demonstrates the model learned spurious correlations, confirming XAI is a powerful diagnostic tool for identifying dataset bias in emerging architectures.
>
---
#### [new 019] History-Enhanced Two-Stage Transformer for Aerial Vision-and-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向空中视觉-语言导航（AVLN）任务，旨在解决无人机在大尺度城市环境中依语言指令准确定位目标时，全局推理与局部感知难以兼顾的问题。提出历史增强双阶段Transformer（HETT），通过粗粒度定位到细粒度动作优化的两阶段框架，并引入历史网格地图增强空间记忆，显著提升导航性能。**

- **链接: [https://arxiv.org/pdf/2512.14222v1](https://arxiv.org/pdf/2512.14222v1)**

> **作者:** Xichen Ding; Jianzhe Gao; Cong Pan; Wenguan Wang; Jie Qin
>
> **摘要:** Aerial Vision-and-Language Navigation (AVLN) requires Unmanned Aerial Vehicle (UAV) agents to localize targets in large-scale urban environments based on linguistic instructions. While successful navigation demands both global environmental reasoning and local scene comprehension, existing UAV agents typically adopt mono-granularity frameworks that struggle to balance these two aspects. To address this limitation, this work proposes a History-Enhanced Two-Stage Transformer (HETT) framework, which integrates the two aspects through a coarse-to-fine navigation pipeline. Specifically, HETT first predicts coarse-grained target positions by fusing spatial landmarks and historical context, then refines actions via fine-grained visual analysis. In addition, a historical grid map is designed to dynamically aggregate visual features into a structured spatial memory, enhancing comprehensive scene awareness. Additionally, the CityNav dataset annotations are manually refined to enhance data quality. Experiments on the refined CityNav dataset show that HETT delivers significant performance gains, while extensive ablation studies further verify the effectiveness of each component.
>
---
#### [new 020] Erasing CLIP Memories: Non-Destructive, Data-Free Zero-Shot class Unlearning in CLIP Models
- **分类: cs.CV**

- **简介: 该论文属模型“选择性遗忘”任务，旨在无数据、不重训练地擦除CLIP中特定类别的零样本识别能力。作者提出基于空空间投影的闭式方法，利用目标类文本嵌入构造正交基并投影，削弱图像-文本对齐，实现精准、高效、非破坏性类遗忘。**

- **链接: [https://arxiv.org/pdf/2512.14137v1](https://arxiv.org/pdf/2512.14137v1)**

> **作者:** Ashish Mishra; Tarun Kumar; Gyanaranjan Nayak; Arpit Shah; Suparna Bhattacharya; Martin Foltin
>
> **摘要:** We introduce a novel, closed-form approach for selective unlearning in multimodal models, specifically targeting pretrained models such as CLIP. Our method leverages nullspace projection to erase the target class information embedded in the final projection layer, without requiring any retraining or the use of images from the forget set. By computing an orthonormal basis for the subspace spanned by target text embeddings and projecting these directions, we dramatically reduce the alignment between image features and undesired classes. Unlike traditional unlearning techniques that rely on iterative fine-tuning and extensive data curation, our approach is both computationally efficient and surgically precise. This leads to a pronounced drop in zero-shot performance for the target classes while preserving the overall multimodal knowledge of the model. Our experiments demonstrate that even a partial projection can balance between complete unlearning and retaining useful information, addressing key challenges in model decontamination and privacy preservation.
>
---
#### [new 021] FastDDHPose: Towards Unified, Efficient, and Disentangled 3D Human Pose Estimation
- **分类: cs.CV**

- **简介: 该论文面向单目3D人体姿态估计任务，旨在解决现有方法缺乏统一评估框架及误差累积问题。作者提出Fast3DHPE统一框架以提升复现性与训练效率，并在此基础上设计FastDDHPose——一种解耦式扩散模型，显式建模骨骼长度与方向分布，结合运动学感知的时空去噪器，实现SOTA性能与强泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.14162v1](https://arxiv.org/pdf/2512.14162v1)**

> **作者:** Qingyuan Cai; Linxin Zhang; Xuecai Hu; Saihui Hou; Yongzhen Huang
>
> **摘要:** Recent approaches for monocular 3D human pose estimation (3D HPE) have achieved leading performance by directly regressing 3D poses from 2D keypoint sequences. Despite the rapid progress in 3D HPE, existing methods are typically trained and evaluated under disparate frameworks, lacking a unified framework for fair comparison. To address these limitations, we propose Fast3DHPE, a modular framework that facilitates rapid reproduction and flexible development of new methods. By standardizing training and evaluation protocols, Fast3DHPE enables fair comparison across 3D human pose estimation methods while significantly improving training efficiency. Within this framework, we introduce FastDDHPose, a Disentangled Diffusion-based 3D Human Pose Estimation method which leverages the strong latent distribution modeling capability of diffusion models to explicitly model the distributions of bone length and bone direction while avoiding further amplification of hierarchical error accumulation. Moreover, we design an efficient Kinematic-Hierarchical Spatial and Temporal Denoiser that encourages the model to focus on kinematic joint hierarchies while avoiding unnecessary modeling of overly complex joint topologies. Extensive experiments on Human3.6M and MPI-INF-3DHP show that the Fast3DHPE framework enables fair comparison of all methods while significantly improving training efficiency. Within this unified framework, FastDDHPose achieves state-of-the-art performance with strong generalization and robustness in in-the-wild scenarios. The framework and models will be released at: https://github.com/Andyen512/Fast3DHPE
>
---
#### [new 022] AMD-HookNet++: Evolution of AMD-HookNet with Hybrid CNN-Transformer Feature Enhancement for Glacier Calving Front Segmentation
- **分类: cs.CV**

- **简介: 该论文面向冰川崩解前沿分割任务，解决纯CNN难以建模长程依赖、纯Transformer边缘锯齿的问题。提出AMD-HookNet++：融合CNN（保局部细节）与Transformer（建模全局上下文）的双分支结构，引入增强空间-通道注意力模块和像素级对比深度监督，显著提升分割精度与边界平滑度。**

- **链接: [https://arxiv.org/pdf/2512.14639v1](https://arxiv.org/pdf/2512.14639v1)**

> **作者:** Fei Wu; Marcel Dreier; Nora Gourmelon; Sebastian Wind; Jianlin Zhang; Thorsten Seehaus; Matthias Braun; Andreas Maier; Vincent Christlein
>
> **摘要:** The dynamics of glaciers and ice shelf fronts significantly impact the mass balance of ice sheets and coastal sea levels. To effectively monitor glacier conditions, it is crucial to consistently estimate positional shifts of glacier calving fronts. AMD-HookNet firstly introduces a pure two-branch convolutional neural network (CNN) for glacier segmentation. Yet, the local nature and translational invariance of convolution operations, while beneficial for capturing low-level details, restricts the model ability to maintain long-range dependencies. In this study, we propose AMD-HookNet++, a novel advanced hybrid CNN-Transformer feature enhancement method for segmenting glaciers and delineating calving fronts in synthetic aperture radar images. Our hybrid structure consists of two branches: a Transformer-based context branch to capture long-range dependencies, which provides global contextual information in a larger view, and a CNN-based target branch to preserve local details. To strengthen the representation of the connected hybrid features, we devise an enhanced spatial-channel attention module to foster interactions between the hybrid CNN-Transformer branches through dynamically adjusting the token relationships from both spatial and channel perspectives. Additionally, we develop a pixel-to-pixel contrastive deep supervision to optimize our hybrid model by integrating pixelwise metric learning into glacier segmentation. Through extensive experiments and comprehensive quantitative and qualitative analyses on the challenging glacier segmentation benchmark dataset CaFFe, we show that AMD-HookNet++ sets a new state of the art with an IoU of 78.2 and a HD95 of 1,318 m, while maintaining a competitive MDE of 367 m. More importantly, our hybrid model produces smoother delineations of calving fronts, resolving the issue of jagged edges typically seen in pure Transformer-based approaches.
>
---
#### [new 023] Native Intelligence Emerges from Large-Scale Clinical Practice: A Retinal Foundation Model with Deployment Efficiency
- **分类: cs.CV**

- **简介: 该论文提出ReVision视网膜基础模型，旨在解决现有模型依赖小规模标注数据、泛化差、部署难的问题。它直接从48.6万张真实临床眼底图及对应诊断报告中学习，实现零样本疾病检测与高效迁移，显著提升低资源场景下的部署效率与诊断准确性。**

- **链接: [https://arxiv.org/pdf/2512.14499v1](https://arxiv.org/pdf/2512.14499v1)**

> **作者:** Jia Guo; Jiawei Du; Shengzhu Yang; Shuai Lu; Wenquan Cheng; Kaiwen Zhang; Yihua Sun; Chuhong Yang; Weihang Zhang; Fang Chen; Yilan Wu; Lie Ju; Guochen Ning; Longfei Ma; Huiping Yao; Jinyuan Wang; Peilun Shi; Yukun Zhou; Jie Xu; Pearse A. Keane; Hanruo Liu; Hongen Liao; Ningli Wang; Huiqi Li
>
> **摘要:** Current retinal foundation models remain constrained by curated research datasets that lack authentic clinical context, and require extensive task-specific optimization for each application, limiting their deployment efficiency in low-resource settings. Here, we show that these barriers can be overcome by building clinical native intelligence directly from real-world medical practice. Our key insight is that large-scale telemedicine programs, where expert centers provide remote consultations across distributed facilities, represent a natural reservoir for learning clinical image interpretation. We present ReVision, a retinal foundation model that learns from the natural alignment between 485,980 color fundus photographs and their corresponding diagnostic reports, accumulated through a decade-long telemedicine program spanning 162 medical institutions across China. Through extensive evaluation across 27 ophthalmic benchmarks, we demonstrate that ReVison enables deployment efficiency with minimal local resources. Without any task-specific training, ReVision achieves zero-shot disease detection with an average AUROC of 0.946 across 12 public benchmarks and 0.952 on 3 independent clinical cohorts. When minimal adaptation is feasible, ReVision matches extensively fine-tuned alternatives while requiring orders of magnitude fewer trainable parameters and labeled examples. The learned representations also transfer effectively to new clinical sites, imaging domains, imaging modalities, and systemic health prediction tasks. In a prospective reader study with 33 ophthalmologists, ReVision's zero-shot assistance improved diagnostic accuracy by 14.8% across all experience levels. These results demonstrate that clinical native intelligence can be directly extracted from clinical archives without any further annotation to build medical AI systems suited to various low-resource settings.
>
---
#### [new 024] SAGE: Training Smart Any-Horizon Agents for Long Video Reasoning with Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文聚焦长视频推理任务，旨在解决现有模型缺乏灵活“任意时长”推理能力的问题。提出SAGE智能代理系统，支持单步/多步推理；构建合成数据集与RL后训练方法；设计长时视频评测基准SAGE-Bench，并验证其在长视频上的显著性能提升。**

- **链接: [https://arxiv.org/pdf/2512.13874v1](https://arxiv.org/pdf/2512.13874v1)**

> **作者:** Jitesh Jain; Jialuo Li; Zixian Ma; Jieyu Zhang; Chris Dongjoo Kim; Sangho Lee; Rohun Tripathi; Tanmay Gupta; Christopher Clark; Humphrey Shi
>
> **备注:** Project Page: https://praeclarumjj3.github.io/sage/
>
> **摘要:** As humans, we are natural any-horizon reasoners, i.e., we can decide whether to iteratively skim long videos or watch short ones in full when necessary for a given task. With this in mind, one would expect video reasoning models to reason flexibly across different durations. However, SOTA models are still trained to predict answers in a single turn while processing a large number of frames, akin to watching an entire long video, requiring significant resources. This raises the question: Is it possible to develop performant any-horizon video reasoning systems? Inspired by human behavior, we first propose SAGE, an agent system that performs multi-turn reasoning on long videos while handling simpler problems in a single turn. Secondly, we introduce an easy synthetic data generation pipeline using Gemini-2.5-Flash to train the orchestrator, SAGE-MM, which lies at the core of SAGE. We further propose an effective RL post-training recipe essential for instilling any-horizon reasoning ability in SAGE-MM. Thirdly, we curate SAGE-Bench with an average duration of greater than 700 seconds for evaluating video reasoning ability in real-world entertainment use cases. Lastly, we empirically validate the effectiveness of our system, data, and RL recipe, observing notable improvements of up to 6.1% on open-ended video reasoning tasks, as well as an impressive 8.2% improvement on videos longer than 10 minutes.
>
---
#### [new 025] ART: Articulated Reconstruction Transformer
- **分类: cs.CV**

- **简介: 该论文提出ART模型，解决从稀疏多姿态RGB图像重建3D可动（关节）物体的任务。它采用类别无关的前馈Transformer架构，将物体建模为刚性部件组合，联合预测各部件的3D几何、纹理及关节参数，实现物理可解释、可仿真的完整重建。**

- **链接: [https://arxiv.org/pdf/2512.14671v1](https://arxiv.org/pdf/2512.14671v1)**

> **作者:** Zizhang Li; Cheng Zhang; Zhengqin Li; Henry Howard-Jenkins; Zhaoyang Lv; Chen Geng; Jiajun Wu; Richard Newcombe; Jakob Engel; Zhao Dong
>
> **备注:** Project Page: https://kyleleey.github.io/ART/
>
> **摘要:** We introduce ART, Articulated Reconstruction Transformer -- a category-agnostic, feed-forward model that reconstructs complete 3D articulated objects from only sparse, multi-state RGB images. Previous methods for articulated object reconstruction either rely on slow optimization with fragile cross-state correspondences or use feed-forward models limited to specific object categories. In contrast, ART treats articulated objects as assemblies of rigid parts, formulating reconstruction as part-based prediction. Our newly designed transformer architecture maps sparse image inputs to a set of learnable part slots, from which ART jointly decodes unified representations for individual parts, including their 3D geometry, texture, and explicit articulation parameters. The resulting reconstructions are physically interpretable and readily exportable for simulation. Trained on a large-scale, diverse dataset with per-part supervision, and evaluated across diverse benchmarks, ART achieves significant improvements over existing baselines and establishes a new state of the art for articulated object reconstruction from image inputs.
>
---
#### [new 026] STAR: STacked AutoRegressive Scheme for Unified Multimodal Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属多模态大语言模型（MLLM）任务，旨在解决统一理解与生成间的优化冲突与性能权衡问题。提出STAR框架：分阶段（理解/生成/编辑）堆叠同构自回归模块，冻结基础参数以避免干扰；引入高容量VQ和隐式推理机制提升图像表征与生成质量。**

- **链接: [https://arxiv.org/pdf/2512.13752v1](https://arxiv.org/pdf/2512.13752v1)**

> **作者:** Jie Qin; Jiancheng Huang; Limeng Qiao; Lin Ma
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Multimodal large language models (MLLMs) play a pivotal role in advancing the quest for general artificial intelligence. However, achieving unified target for multimodal understanding and generation remains challenging due to optimization conflicts and performance trade-offs. To effectively enhance generative performance while preserving existing comprehension capabilities, we introduce STAR: a STacked AutoRegressive scheme for task-progressive unified multimodal learning. This approach decomposes multimodal learning into multiple stages: understanding, generation, and editing. By freezing the parameters of the fundamental autoregressive (AR) model and progressively stacking isomorphic AR modules, it avoids cross-task interference while expanding the model's capabilities. Concurrently, we introduce a high-capacity VQ to enhance the granularity of image representations and employ an implicit reasoning mechanism to improve generation quality under complex conditions. Experiments demonstrate that STAR achieves state-of-the-art performance on GenEval (0.91), DPG-Bench (87.44), and ImgEdit (4.34), validating its efficacy for unified multimodal learning.
>
---
#### [new 027] AnchorHOI: Zero-shot Generation of 4D Human-Object Interaction via Anchor-based Prior Distillation
- **分类: cs.CV**

- **简介: 该论文属零-shot 4D人-物交互（HOI）生成任务，旨在解决因4D HOI数据稀缺导致的监督方法泛化性差问题。提出AnchorHOI框架，引入视频扩散先验，并设计NeRF与关键点两类交互感知锚点，通过两步蒸馏实现高质量、多样化的4D HOI生成。**

- **链接: [https://arxiv.org/pdf/2512.14095v1](https://arxiv.org/pdf/2512.14095v1)**

> **作者:** Sisi Dai; Kai Xu
>
> **备注:** AAAI 2026
>
> **摘要:** Despite significant progress in text-driven 4D human-object interaction (HOI) generation with supervised methods, the scalability remains limited by the scarcity of large-scale 4D HOI datasets. To overcome this, recent approaches attempt zero-shot 4D HOI generation with pre-trained image diffusion models. However, interaction cues are minimally distilled during the generation process, restricting their applicability across diverse scenarios. In this paper, we propose AnchorHOI, a novel framework that thoroughly exploits hybrid priors by incorporating video diffusion models beyond image diffusion models, advancing 4D HOI generation. Nevertheless, directly optimizing high-dimensional 4D HOI with such priors remains challenging, particularly for human pose and compositional motion. To address this challenge, AnchorHOI introduces an anchor-based prior distillation strategy, which constructs interaction-aware anchors and then leverages them to guide generation in a tractable two-step process. Specifically, two tailored anchors are designed for 4D HOI generation: anchor Neural Radiance Fields (NeRFs) for expressive interaction composition, and anchor keypoints for realistic motion synthesis. Extensive experiments demonstrate that AnchorHOI outperforms previous methods with superior diversity and generalization.
>
---
#### [new 028] SuperCLIP: CLIP with Simple Classification Supervision
- **分类: cs.CV**

- **简介: 该论文属多模态学习任务，旨在解决CLIP因仅依赖全局对比损失而忽视细粒度文本语义、导致图文对齐不足的问题。作者提出SuperCLIP，在视觉编码器后添加轻量线性分类层，引入词元级分类监督，无需额外标注数据，显著提升零样本分类、图文检索等性能。**

- **链接: [https://arxiv.org/pdf/2512.14480v1](https://arxiv.org/pdf/2512.14480v1)**

> **作者:** Weiheng Zhao; Zilong Huang; Jiashi Feng; Xinggang Wang
>
> **备注:** Accepted by NeurIPS 2025. Code: https://github.com/hustvl/SuperCLIP
>
> **摘要:** Contrastive Language-Image Pretraining (CLIP) achieves strong generalization in vision-language tasks by aligning images and texts in a shared embedding space. However, recent findings show that CLIP-like models still underutilize fine-grained semantic signals in text, and this issue becomes even more pronounced when dealing with long and detailed captions. This stems from CLIP's training objective, which optimizes only global image-text similarity and overlooks token-level supervision - limiting its ability to achieve fine-grained visual-text alignment. To address this, we propose SuperCLIP, a simple yet effective framework that augments contrastive learning with classification-based supervision. By adding only a lightweight linear layer to the vision encoder, SuperCLIP leverages token-level cues to enhance visual-textual alignment - with just a 0.077% increase in total FLOPs, and no need for additional annotated data. Experiments show that SuperCLIP consistently improves zero-shot classification, image-text retrieval, and purely visual tasks. These gains hold regardless of whether the model is trained on original web data or rich re-captioned data, demonstrating SuperCLIP's ability to recover textual supervision in both cases. Furthermore, SuperCLIP alleviates CLIP's small-batch performance drop through classification-based supervision that avoids reliance on large batch sizes. Code and models will be made open source.
>
---
#### [new 029] Enhancing Visual Sentiment Analysis via Semiotic Isotopy-Guided Dataset Construction
- **分类: cs.CV**

- **简介: 该论文属视觉情感分析（VSA）任务，旨在解决数据多样性不足与模型泛化差问题。提出基于符号学“同质性”（isotopy）指导的数据集构建方法，生成更丰富、情感要素组合更显著的新数据集，显著提升模型跨数据集泛化性能。**

- **链接: [https://arxiv.org/pdf/2512.14665v1](https://arxiv.org/pdf/2512.14665v1)**

> **作者:** Marco Blanchini; Giovanna Maria Dimitri; Benedetta Tondi; Tarcisio Lancioni; Mauro Barni
>
> **摘要:** Visual Sentiment Analysis (VSA) is a challenging task due to the vast diversity of emotionally salient images and the inherent difficulty of acquiring sufficient data to capture this variability comprehensively. Key obstacles include building large-scale VSA datasets and developing effective methodologies that enable algorithms to identify emotionally significant elements within an image. These challenges are reflected in the limited generalization performance of VSA algorithms and models when trained and tested across different datasets. Starting from a pool of existing data collections, our approach enables the creation of a new larger dataset that not only contains a wider variety of images than the original ones, but also permits training new models with improved capability to focus on emotionally relevant combinations of image elements. This is achieved through the integration of the semiotic isotopy concept within the dataset creation process, providing deeper insights into the emotional content of images. Empirical evaluations show that models trained on a dataset generated with our method consistently outperform those trained on the original data collections, achieving superior generalization across major VSA benchmarks
>
---
#### [new 030] TUN: Detecting Significant Points in Persistence Diagrams with Deep Learning
- **分类: cs.CV; cs.LG; math.AT**

- **简介: 该论文属拓扑数据分析任务，旨在自动识别一维持久性图（PD）中具有真实拓扑意义的显著点。针对PD信号判别难、人工解释不可靠的问题，提出多模态网络TUN，融合增强PD描述符、自注意力与PointNet编码，实现端到端显著点分类。**

- **链接: [https://arxiv.org/pdf/2512.14274v1](https://arxiv.org/pdf/2512.14274v1)**

> **作者:** Yu Chen; Hongwei Lin
>
> **摘要:** Persistence diagrams (PDs) provide a powerful tool for understanding the topology of the underlying shape of a point cloud. However, identifying which points in PDs encode genuine signals remains challenging. This challenge directly hinders the practical adoption of topological data analysis in many applications, where automated and reliable interpretation of persistence diagrams is essential for downstream decision-making. In this paper, we study automatic significance detection for one-dimensional persistence diagrams. Specifically, we propose Topology Understanding Net (TUN), a multi-modal network that combines enhanced PD descriptors with self-attention, a PointNet-style point cloud encoder, learned fusion, and per-point classification, alongside stable preprocessing and imbalance-aware training. It provides an automated and effective solution for identifying significant points in PDs, which are critical for downstream applications. Experiments show that TUN outperforms classic methods in detecting significant points in PDs, illustrating its effectiveness in real-world applications.
>
---
#### [new 031] Vector Prism: Animating Vector Graphics by Stratifying Semantic Structure
- **分类: cs.CV**

- **简介: 该论文属SVG动画生成任务，旨在解决VLM因无法理解SVG语义结构而导致动画不连贯的问题。提出Vector Prism框架，通过统计聚合弱预测恢复语义分组，使SVG可被VLM更可靠地动画化。**

- **链接: [https://arxiv.org/pdf/2512.14336v1](https://arxiv.org/pdf/2512.14336v1)**

> **作者:** Jooyeol Yun; Jaegul Choo
>
> **备注:** yeolj00.github.io/personal-projects/vector-prism
>
> **摘要:** Scalable Vector Graphics (SVG) are central to modern web design, and the demand to animate them continues to grow as web environments become increasingly dynamic. Yet automating the animation of vector graphics remains challenging for vision-language models (VLMs) despite recent progress in code generation and motion planning. VLMs routinely mis-handle SVGs, since visually coherent parts are often fragmented into low-level shapes that offer little guidance of which elements should move together. In this paper, we introduce a framework that recovers the semantic structure required for reliable SVG animation and reveals the missing layer that current VLM systems overlook. This is achieved through a statistical aggregation of multiple weak part predictions, allowing the system to stably infer semantics from noisy predictions. By reorganizing SVGs into semantic groups, our approach enables VLMs to produce animations with far greater coherence. Our experiments demonstrate substantial gains over existing approaches, suggesting that semantic recovery is the key step that unlocks robust SVG animation and supports more interpretable interactions between VLMs and vector graphics.
>
---
#### [new 032] Score-Based Turbo Message Passing for Plug-and-Play Compressive Imaging
- **分类: cs.CV**

- **简介: 该论文面向压缩成像任务，旨在解决传统PnP方法因先验表达能力弱导致的重建性能瓶颈。提出基于分数模型的涡轮消息传递（STMP）算法，融合分数MMSE去噪器与消息传递；进一步设计Q-STMP以支持量化测量，并给出状态演化分析，实验证明其高效鲁棒。**

- **链接: [https://arxiv.org/pdf/2512.14435v1](https://arxiv.org/pdf/2512.14435v1)**

> **作者:** Chang Cai; Hao Jiang; Xiaojun Yuan; Ying-Jun Angela Zhang
>
> **摘要:** Message-passing algorithms have been adapted for compressive imaging by incorporating various off-the-shelf image denoisers. However, these denoisers rely largely on generic or hand-crafted priors and often fall short in accurately capturing the complex statistical structure of natural images. As a result, traditional plug-and-play (PnP) methods often lead to suboptimal reconstruction, especially in highly underdetermined regimes. Recently, score-based generative models have emerged as a powerful framework for accurately characterizing sophisticated image distribution. Yet, their direct use for posterior sampling typically incurs prohibitive computational complexity. In this paper, by exploiting the close connection between score-based generative modeling and empirical Bayes denoising, we devise a message-passing framework that integrates a score-based minimum mean-squared error (MMSE) denoiser for compressive image recovery. The resulting algorithm, named score-based turbo message passing (STMP), combines the fast convergence of message passing with the expressive power of score-based generative priors. For practical systems with quantized measurements, we further propose quantized STMP (Q-STMP), which augments STMP with a component-wise MMSE dequantization module. We demonstrate that the asymptotic performance of STMP and Q-STMP can be accurately predicted by a set of state-evolution (SE) equations. Experiments on the FFHQ dataset demonstrate that STMP strikes a significantly better performance-complexity tradeoff compared with competing baselines, and that Q-STMP remains robust even under 1-bit quantization. Remarkably, both STMP and Q-STMP typically converge within 10 iterations.
>
---
#### [new 033] TAT: Task-Adaptive Transformer for All-in-One Medical Image Restoration
- **分类: cs.CV**

- **简介: 该论文属医学图像恢复（MedIR）任务，旨在解决All-in-One多任务联合训练中的任务干扰与任务不平衡问题。提出任务自适应Transformer（TAT），通过任务特定权重生成和动态损失平衡策略，提升PET合成、CT去噪、MRI超分性能。**

- **链接: [https://arxiv.org/pdf/2512.14550v1](https://arxiv.org/pdf/2512.14550v1)**

> **作者:** Zhiwen Yang; Jiaju Zhang; Yang Yi; Jian Liang; Bingzheng Wei; Yan Xu
>
> **备注:** This paper has been accepted by MICCAI 2025
>
> **摘要:** Medical image restoration (MedIR) aims to recover high-quality medical images from their low-quality counterparts. Recent advancements in MedIR have focused on All-in-One models capable of simultaneously addressing multiple different MedIR tasks. However, due to significant differences in both modality and degradation types, using a shared model for these diverse tasks requires careful consideration of two critical inter-task relationships: task interference, which occurs when conflicting gradient update directions arise across tasks on the same parameter, and task imbalance, which refers to uneven optimization caused by varying learning difficulties inherent to each task. To address these challenges, we propose a task-adaptive Transformer (TAT), a novel framework that dynamically adapts to different tasks through two key innovations. First, a task-adaptive weight generation strategy is introduced to mitigate task interference by generating task-specific weight parameters for each task, thereby eliminating potential gradient conflicts on shared weight parameters. Second, a task-adaptive loss balancing strategy is introduced to dynamically adjust loss weights based on task-specific learning difficulties, preventing task domination or undertraining. Extensive experiments demonstrate that our proposed TAT achieves state-of-the-art performance in three MedIR tasks--PET synthesis, CT denoising, and MRI super-resolution--both in task-specific and All-in-One settings. Code is available at https://github.com/Yaziwel/TAT.
>
---
#### [new 034] PSMamba: Progressive Self-supervised Vision Mamba for Plant Disease Recognition
- **分类: cs.CV**

- **简介: 该论文面向植物病害识别任务，解决现有自监督方法难以建模多尺度病斑特征的问题。提出PSMamba框架，结合Vision Mamba与双学生分层蒸馏机制，分别学习中尺度（病斑分布、叶脉）和局部尺度（纹理、早期病灶）表征，并通过一致性损失实现跨尺度对齐。**

- **链接: [https://arxiv.org/pdf/2512.14309v1](https://arxiv.org/pdf/2512.14309v1)**

> **作者:** Abdullah Al Mamun; Miaohua Zhang; David Ahmedt-Aristizabal; Zeeshan Hayder; Mohammad Awrangjeb
>
> **摘要:** Self-supervised Learning (SSL) has become a powerful paradigm for representation learning without manual annotations. However, most existing frameworks focus on global alignment and struggle to capture the hierarchical, multi-scale lesion patterns characteristic of plant disease imagery. To address this gap, we propose PSMamba, a progressive self-supervised framework that integrates the efficient sequence modelling of Vision Mamba (VM) with a dual-student hierarchical distillation strategy. Unlike conventional single teacher-student designs, PSMamba employs a shared global teacher and two specialised students: one processes mid-scale views to capture lesion distributions and vein structures, while the other focuses on local views to capture fine-grained cues such as texture irregularities and early-stage lesions. This multi-granular supervision facilitates the joint learning of contextual and detailed representations, with consistency losses ensuring coherent cross-scale alignment. Experiments on three benchmark datasets show that PSMamba consistently outperforms state-of-the-art SSL methods, delivering superior accuracy and robustness in both domain-shifted and fine-grained scenarios.
>
---
#### [new 035] SELECT: Detecting Label Errors in Real-world Scene Text Data
- **分类: cs.CV**

- **简介: 该论文属于场景文本识别（STR）中的标签错误检测任务，旨在解决真实场景文本数据中因标注错误导致的变长序列、错位及字符级错误问题。作者提出SELECT方法，结合多模态训练与SSLC数据增强策略，首次有效检测变长标签错误，提升STR精度。**

- **链接: [https://arxiv.org/pdf/2512.14050v1](https://arxiv.org/pdf/2512.14050v1)**

> **作者:** Wenjun Liu; Qian Wu; Yifeng Hu; Yuke Li
>
> **摘要:** We introduce SELECT (Scene tExt Label Errors deteCTion), a novel approach that leverages multi-modal training to detect label errors in real-world scene text datasets. Utilizing an image-text encoder and a character-level tokenizer, SELECT addresses the issues of variable-length sequence labels, label sequence misalignment, and character-level errors, outperforming existing methods in accuracy and practical utility. In addition, we introduce Similarity-based Sequence Label Corruption (SSLC), a process that intentionally introduces errors into the training labels to mimic real-world error scenarios during training. SSLC not only can cause a change in the sequence length but also takes into account the visual similarity between characters during corruption. Our method is the first to detect label errors in real-world scene text datasets successfully accounting for variable-length labels. Experimental results demonstrate the effectiveness of SELECT in detecting label errors and improving STR accuracy on real-world text datasets, showcasing its practical utility.
>
---
#### [new 036] SS4D: Native 4D Generative Model via Structured Spacetime Latents
- **分类: cs.CV**

- **简介: 该论文提出SS4D，一种原生4D生成模型，旨在从单目视频直接合成动态3D对象。它解决现有方法依赖3D/视频模型拼接导致时序不连贯、结构不一致的问题。工作包括：基于图像到3D预训练模型构建结构化时空隐变量，引入时序层保障时间一致性，并用因子化4D卷积与时间下采样实现高效长序列建模。**

- **链接: [https://arxiv.org/pdf/2512.14284v1](https://arxiv.org/pdf/2512.14284v1)**

> **作者:** Zhibing Li; Mengchen Zhang; Tong Wu; Jing Tan; Jiaqi Wang; Dahua Lin
>
> **备注:** ToG(Siggraph Asia 2025)
>
> **摘要:** We present SS4D, a native 4D generative model that synthesizes dynamic 3D objects directly from monocular video. Unlike prior approaches that construct 4D representations by optimizing over 3D or video generative models, we train a generator directly on 4D data, achieving high fidelity, temporal coherence, and structural consistency. At the core of our method is a compressed set of structured spacetime latents. Specifically, (1) To address the scarcity of 4D training data, we build on a pre-trained single-image-to-3D model, preserving strong spatial consistency. (2) Temporal consistency is enforced by introducing dedicated temporal layers that reason across frames. (3) To support efficient training and inference over long video sequences, we compress the latent sequence along the temporal axis using factorized 4D convolutions and temporal downsampling blocks. In addition, we employ a carefully designed training strategy to enhance robustness against occlusion
>
---
#### [new 037] Zoom-Zero: Reinforced Coarse-to-Fine Video Understanding via Temporal Zoom-in
- **分类: cs.CV**

- **简介: 该论文面向 grounded video question answering（GVQA）任务，旨在解决大视频语言模型时序定位不准、易幻觉的问题。提出Zoom-Zero框架：先粗粒度定位相关视频段，再细粒度“缩放”至关键帧验证；引入缩放准确率奖励和token选择性信用分配，显著提升时序定位与答案准确性。**

- **链接: [https://arxiv.org/pdf/2512.14273v1](https://arxiv.org/pdf/2512.14273v1)**

> **作者:** Xiaoqian Shen; Min-Hung Chen; Yu-Chiang Frank Wang; Mohamed Elhoseiny; Ryo Hachiuma
>
> **备注:** Project page: https://xiaoqian-shen.github.io/Zoom-Zero/
>
> **摘要:** Grounded video question answering (GVQA) aims to localize relevant temporal segments in videos and generate accurate answers to a given question; however, large video-language models (LVLMs) exhibit limited temporal awareness. Although existing approaches based on Group Relative Policy Optimization (GRPO) attempt to improve temporal grounding, they still struggle to faithfully ground their answers in the relevant video evidence, leading to temporal mislocalization and hallucinations. In this work, we present Zoom-Zero, a coarse-to-fine framework that first localizes query-relevant segments and then temporally zooms into the most salient frames for finer-grained visual verification. Our method addresses the limits of GRPO for the GVQA task with two key innovations: (i) a zoom-in accuracy reward that validates the fidelity of temporal grounding prediction and facilitates fine-grained visual verification on grounded frames; (ii) token-selective credit assignment, which attributes rewards to the tokens responsible for temporal localization or answer generation, mitigating GRPO's issue in handling multi-faceted reward signals. Our proposed method advances grounded video question answering, improving temporal grounding by 5.2\% on NExT-GQA and 4.6\% on ReXTime, while also enhancing average answer accuracy by 2.4\%. Additionally, the coarse-to-fine zoom-in during inference further benefits long-form video understanding by preserving critical visual details without compromising global context, yielding an average improvement of 6.4\% on long-video benchmarks.
>
---
#### [new 038] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文聚焦视频时序定位（VTG）任务，旨在提升多模态大模型的时序理解能力。针对现有基准质量差、训练数据噪声大、算法设计不明确三大问题，作者构建高质量基准TimeLens-Bench与数据集TimeLens-100K，并提出文本交错编码、RLVR训练等新方法，显著提升VTG性能。**

- **链接: [https://arxiv.org/pdf/2512.14698v1](https://arxiv.org/pdf/2512.14698v1)**

> **作者:** Jun Zhang; Teng Wang; Yuying Ge; Yixiao Ge; Xinhao Li; Ying Shan; Limin Wang
>
> **备注:** Project Page: https://timelens-arc-lab.github.io/
>
> **摘要:** This paper does not introduce a novel method but instead establishes a straightforward, incremental, yet essential baseline for video temporal grounding (VTG), a core capability in video understanding. While multimodal large language models (MLLMs) excel at various video understanding tasks, the recipes for optimizing them for VTG remain under-explored. In this paper, we present TimeLens, a systematic investigation into building MLLMs with strong VTG ability, along two primary dimensions: data quality and algorithmic design. We first expose critical quality issues in existing VTG benchmarks and introduce TimeLens-Bench, comprising meticulously re-annotated versions of three popular benchmarks with strict quality criteria. Our analysis reveals dramatic model re-rankings compared to legacy benchmarks, confirming the unreliability of prior evaluation standards. We also address noisy training data through an automated re-annotation pipeline, yielding TimeLens-100K, a large-scale, high-quality training dataset. Building on our data foundation, we conduct in-depth explorations of algorithmic design principles, yielding a series of meaningful insights and effective yet efficient practices. These include interleaved textual encoding for time representation, a thinking-free reinforcement learning with verifiable rewards (RLVR) approach as the training paradigm, and carefully designed recipes for RLVR training. These efforts culminate in TimeLens models, a family of MLLMs with state-of-the-art VTG performance among open-source models and even surpass proprietary models such as GPT-5 and Gemini-2.5-Flash. All codes, data, and models will be released to facilitate future research.
>
---
#### [new 039] ChartAgent: A Chart Understanding Framework with Tool Integrated Reasoning
- **分类: cs.CV; cs.LG**

- **简介: 该论文面向图表理解任务，解决现有MLLMs依赖文本标注、缺少数值时性能骤降的问题。提出ChartAgent框架，基于工具集成推理（TIR），通过可复现的多步视觉解析与结构化证据包，提升稀疏标注下的鲁棒性与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.14040v1](https://arxiv.org/pdf/2512.14040v1)**

> **作者:** Boran Wang; Xinming Wang; Yi Chen; Xiang Li; Jian Xu; Jing Yuan; Chenglin Liu
>
> **摘要:** With their high information density and intuitive readability, charts have become the de facto medium for data analysis and communication across disciplines. Recent multimodal large language models (MLLMs) have made notable progress in automated chart understanding, yet they remain heavily dependent on explicit textual annotations and the performance degrades markedly when key numerals are absent. To address this limitation, we introduce ChartAgent, a chart understanding framework grounded in Tool-Integrated Reasoning (TIR). Inspired by human cognition, ChartAgent decomposes complex chart analysis into a sequence of observable, replayable steps. Supporting this architecture is an extensible, modular tool library comprising more than a dozen core tools, such as keyelement detection, instance segmentation, and optical character recognition (OCR), which the agent dynamically orchestrates to achieve systematic visual parsing across diverse chart types. Leveraging TIRs transparency and verifiability, ChartAgent moves beyond the black box paradigm by standardizing and consolidating intermediate outputs into a structured Evidence Package, providing traceable and reproducible support for final conclusions. Experiments show that ChartAgent substantially improves robustness under sparse annotation settings, offering a practical path toward trustworthy and extensible systems for chart understanding.
>
---
#### [new 040] Spherical Voronoi: Directional Appearance as a Differentiable Partition of the Sphere
- **分类: cs.CV**

- **简介: 该论文面向神经渲染中的外观建模任务，解决球谐函数（SH）难以表征高频率、镜面反射等视点相关效应的问题。提出球形Voronoi（SV）方法，将方向域划分为可学习区域，统一建模漫反射与反射，提升效果并简化优化。**

- **链接: [https://arxiv.org/pdf/2512.14180v1](https://arxiv.org/pdf/2512.14180v1)**

> **作者:** Francesco Di Sario; Daniel Rebain; Dor Verbin; Marco Grangetto; Andrea Tagliasacchi
>
> **摘要:** Radiance field methods (e.g. 3D Gaussian Splatting) have emerged as a powerful paradigm for novel view synthesis, yet their appearance modeling often relies on Spherical Harmonics (SH), which impose fundamental limitations. SH struggle with high-frequency signals, exhibit Gibbs ringing artifacts, and fail to capture specular reflections - a key component of realistic rendering. Although alternatives like spherical Gaussians offer improvements, they add significant optimization complexity. We propose Spherical Voronoi (SV) as a unified framework for appearance representation in 3D Gaussian Splatting. SV partitions the directional domain into learnable regions with smooth boundaries, providing an intuitive and stable parameterization for view-dependent effects. For diffuse appearance, SV achieves competitive results while keeping optimization simpler than existing alternatives. For reflections - where SH fail - we leverage SV as learnable reflection probes, taking reflected directions as input following principles from classical graphics. This formulation attains state-of-the-art results on synthetic and real-world datasets, demonstrating that SV offers a principled, efficient, and general solution for appearance modeling in explicit 3D representations.
>
---
#### [new 041] Elastic3D: Controllable Stereo Video Conversion with Guided Latent Decoding
- **分类: cs.CV**

- **简介: 该论文提出Elastic3D，解决单目视频到立体视频的自动转换任务。它基于条件潜在扩散模型，摒弃深度估计与图像扭曲，引入引导式VAE解码器保障视差一致性与清晰度，并支持推理时通过标量参数调控立体强度。**

- **链接: [https://arxiv.org/pdf/2512.14236v1](https://arxiv.org/pdf/2512.14236v1)**

> **作者:** Nando Metzger; Prune Truong; Goutam Bhat; Konrad Schindler; Federico Tombari
>
> **备注:** Project page: elastic3d.github.io
>
> **摘要:** The growing demand for immersive 3D content calls for automated monocular-to-stereo video conversion. We present Elastic3D, a controllable, direct end-to-end method for upgrading a conventional video to a binocular one. Our approach, based on (conditional) latent diffusion, avoids artifacts due to explicit depth estimation and warping. The key to its high-quality stereo video output is a novel, guided VAE decoder that ensures sharp and epipolar-consistent stereo video output. Moreover, our method gives the user control over the strength of the stereo effect (more precisely, the disparity range) at inference time, via an intuitive, scalar tuning knob. Experiments on three different datasets of real-world stereo videos show that our method outperforms both traditional warping-based and recent warping-free baselines and sets a new standard for reliable, controllable stereo video conversion. Please check the project page for the video samples https://elastic3d.github.io.
>
---
#### [new 042] Quality-Driven and Diversity-Aware Sample Expansion for Robust Marine Obstacle Segmentation
- **分类: cs.CV**

- **简介: 该论文面向海洋障碍物分割任务，解决因图像质量差（如阳光耀斑、雾）和训练数据稀缺重复导致的模型鲁棒性不足问题。提出无需重训扩散模型的推理时样本扩增方法：构建类感知高熵风格库生成优质提示，并设计COD引导的自适应退火采样器，在保持布局精度前提下提升合成样本多样性。**

- **链接: [https://arxiv.org/pdf/2512.13970v1](https://arxiv.org/pdf/2512.13970v1)**

> **作者:** Miaohua Zhang; Mohammad Ali Armin; Xuesong Li; Sisi Liang; Lars Petersson; Changming Sun; David Ahmedt-Aristizabal; Zeeshan Hayder
>
> **备注:** 10 pages
>
> **摘要:** Marine obstacle detection demands robust segmentation under challenging conditions, such as sun glitter, fog, and rapidly changing wave patterns. These factors degrade image quality, while the scarcity and structural repetition of marine datasets limit the diversity of available training data. Although mask-conditioned diffusion models can synthesize layout-aligned samples, they often produce low-diversity outputs when conditioned on low-entropy masks and prompts, limiting their utility for improving robustness. In this paper, we propose a quality-driven and diversity-aware sample expansion pipeline that generates training data entirely at inference time, without retraining the diffusion model. The framework combines two key components:(i) a class-aware style bank that constructs high-entropy, semantically grounded prompts, and (ii) an adaptive annealing sampler that perturbs early conditioning, while a COD-guided proportional controller regulates this perturbation to boost diversity without compromising layout fidelity. Across marine obstacle benchmarks, augmenting training data with these controlled synthetic samples consistently improves segmentation performance across multiple backbones and increases visual variation in rare and texture-sensitive classes.
>
---
#### [new 043] An evaluation of SVBRDF Prediction from Generative Image Models for Appearance Modeling of 3D Scenes
- **分类: cs.CV; cs.GR**

- **简介: 该论文属计算机图形学与生成式AI交叉任务，旨在评估生成图像模型预测SVBRDF（空间变化双向反射分布函数）的能力，解决多视角纹理不一致问题。作者对比不同神经架构与条件输入，发现标准UNet在精度与一致性上表现优异。**

- **链接: [https://arxiv.org/pdf/2512.13950v1](https://arxiv.org/pdf/2512.13950v1)**

> **作者:** Alban Gauthier; Valentin Deschaintre; Alexandre Lanvin; Fredo Durand; Adrien Bousseau; George Drettakis
>
> **备注:** Project page: http://repo-sam.inria.fr/nerphys/svbrdf-evaluation Code: http://github.com/graphdeco-inria/svbrdf-evaluation
>
> **摘要:** Digital content creation is experiencing a profound change with the advent of deep generative models. For texturing, conditional image generators now allow the synthesis of realistic RGB images of a 3D scene that align with the geometry of that scene. For appearance modeling, SVBRDF prediction networks recover material parameters from RGB images. Combining these technologies allows us to quickly generate SVBRDF maps for multiple views of a 3D scene, which can be merged to form a SVBRDF texture atlas of that scene. In this paper, we analyze the challenges and opportunities for SVBRDF prediction in the context of such a fast appearance modeling pipeline. On the one hand, single-view SVBRDF predictions might suffer from multiview incoherence and yield inconsistent texture atlases. On the other hand, generated RGB images, and the different modalities on which they are conditioned, can provide additional information for SVBRDF estimation compared to photographs. We compare neural architectures and conditions to identify designs that achieve high accuracy and coherence. We find that, surprisingly, a standard UNet is competitive with more complex designs. Project page: http://repo-sam.inria.fr/nerphys/svbrdf-evaluation
>
---
#### [new 044] TUMTraf EMOT: Event-Based Multi-Object Tracking Dataset and Baseline for Traffic Scenarios
- **分类: cs.CV**

- **简介: 该论文面向智能交通系统中的多目标跟踪任务，旨在解决传统帧式相机在低照度、高速运动下性能差的问题。作者构建了首个面向交通场景的事件相机多目标跟踪数据集TUMTraf EMOT，并提出基于检测跟踪的基线方法，验证了事件相机在该任务上的有效性。**

- **链接: [https://arxiv.org/pdf/2512.14595v1](https://arxiv.org/pdf/2512.14595v1)**

> **作者:** Mengyu Li; Xingcheng Zhou; Guang Chen; Alois Knoll; Hu Cao
>
> **备注:** 10 pages, 9 figures
>
> **摘要:** In Intelligent Transportation Systems (ITS), multi-object tracking is primarily based on frame-based cameras. However, these cameras tend to perform poorly under dim lighting and high-speed motion conditions. Event cameras, characterized by low latency, high dynamic range and high temporal resolution, have considerable potential to mitigate these issues. Compared to frame-based vision, there are far fewer studies on event-based vision. To address this research gap, we introduce an initial pilot dataset tailored for event-based ITS, covering vehicle and pedestrian detection and tracking. We establish a tracking-by-detection benchmark with a specialized feature extractor based on this dataset, achieving excellent performance.
>
---
#### [new 045] HGS: Hybrid Gaussian Splatting with Static-Dynamic Decomposition for Compact Dynamic View Synthesis
- **分类: cs.CV; cs.CG**

- **简介: 该论文面向动态新视角合成任务，旨在解决现有3D高斯泼溅方法模型冗余、体积大、渲染慢的问题。提出混合高斯泼溅（HGS）框架，通过静态-动态分解（SDD）策略，用时变RBF建模动态区域、共享参数优化静态区域，并设计两阶段训练提升边界一致性，实现紧凑高效实时渲染。**

- **链接: [https://arxiv.org/pdf/2512.14352v1](https://arxiv.org/pdf/2512.14352v1)**

> **作者:** Kaizhe Zhang; Yijie Zhou; Weizhan Zhang; Caixia Yan; Haipeng Du; yugui xie; Yu-Hui Wen; Yong-Jin Liu
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** Dynamic novel view synthesis (NVS) is essential for creating immersive experiences. Existing approaches have advanced dynamic NVS by introducing 3D Gaussian Splatting (3DGS) with implicit deformation fields or indiscriminately assigned time-varying parameters, surpassing NeRF-based methods. However, due to excessive model complexity and parameter redundancy, they incur large model sizes and slow rendering speeds, making them inefficient for real-time applications, particularly on resource-constrained devices. To obtain a more efficient model with fewer redundant parameters, in this paper, we propose Hybrid Gaussian Splatting (HGS), a compact and efficient framework explicitly designed to disentangle static and dynamic regions of a scene within a unified representation. The core innovation of HGS lies in our Static-Dynamic Decomposition (SDD) strategy, which leverages Radial Basis Function (RBF) modeling for Gaussian primitives. Specifically, for dynamic regions, we employ time-dependent RBFs to effectively capture temporal variations and handle abrupt scene changes, while for static regions, we reduce redundancy by sharing temporally invariant parameters. Additionally, we introduce a two-stage training strategy tailored for explicit models to enhance temporal coherence at static-dynamic boundaries. Experimental results demonstrate that our method reduces model size by up to 98% and achieves real-time rendering at up to 125 FPS at 4K resolution on a single RTX 3090 GPU. It further sustains 160 FPS at 1352 * 1014 on an RTX 3050 and has been integrated into the VR system. Moreover, HGS achieves comparable rendering quality to state-of-the-art methods while providing significantly improved visual fidelity for high-frequency details and abrupt scene changes.
>
---
#### [new 046] DASP: Self-supervised Nighttime Monocular Depth Estimation with Domain Adaptation of Spatiotemporal Priors
- **分类: cs.CV**

- **简介: 该论文面向夜间单目深度估计任务，解决自监督方法在低照度、纹理缺失和运动模糊下性能下降的问题。提出DASP框架：通过对抗分支学习日间时空先验（含STLM与ASLM模块），结合自监督分支的3D一致性投影损失，实现夜间深度估计。**

- **链接: [https://arxiv.org/pdf/2512.14536v1](https://arxiv.org/pdf/2512.14536v1)**

> **作者:** Yiheng Huang; Junhong Chen; Anqi Ning; Zhanhong Liang; Nick Michiels; Luc Claesen; Wenyin Liu
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Self-supervised monocular depth estimation has achieved notable success under daytime conditions. However, its performance deteriorates markedly at night due to low visibility and varying illumination, e.g., insufficient light causes textureless areas, and moving objects bring blurry regions. To this end, we propose a self-supervised framework named DASP that leverages spatiotemporal priors for nighttime depth estimation. Specifically, DASP consists of an adversarial branch for extracting spatiotemporal priors and a self-supervised branch for learning. In the adversarial branch, we first design an adversarial network where the discriminator is composed of four devised spatiotemporal priors learning blocks (SPLB) to exploit the daytime priors. In particular, the SPLB contains a spatial-based temporal learning module (STLM) that uses orthogonal differencing to extract motion-related variations along the time axis and an axial spatial learning module (ASLM) that adopts local asymmetric convolutions with global axial attention to capture the multiscale structural information. By combining STLM and ASLM, our model can acquire sufficient spatiotemporal features to restore textureless areas and estimate the blurry regions caused by dynamic objects. In the self-supervised branch, we propose a 3D consistency projection loss to bilaterally project the target frame and source frame into a shared 3D space, and calculate the 3D discrepancy between the two projected frames as a loss to optimize the 3D structural consistency and daytime priors. Extensive experiments on the Oxford RobotCar and nuScenes datasets demonstrate that our approach achieves state-of-the-art performance for nighttime depth estimation. Ablation studies further validate the effectiveness of each component.
>
---
#### [new 047] Dual Attention Guided Defense Against Malicious Edits
- **分类: cs.CV; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属图像编辑安全防御任务，旨在抵御文本驱动的恶意图像篡改。提出双注意力引导噪声扰动（DANP）方法，通过动态掩码调控跨注意力与噪声预测，在多步扩散过程中干扰模型语义理解，提升对恶意编辑的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.14333v1](https://arxiv.org/pdf/2512.14333v1)**

> **作者:** Jie Zhang; Shuai Dong; Shiguang Shan; Xilin Chen
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Recent progress in text-to-image diffusion models has transformed image editing via text prompts, yet this also introduces significant ethical challenges from potential misuse in creating deceptive or harmful content. While current defenses seek to mitigate this risk by embedding imperceptible perturbations, their effectiveness is limited against malicious tampering. To address this issue, we propose a Dual Attention-Guided Noise Perturbation (DANP) immunization method that adds imperceptible perturbations to disrupt the model's semantic understanding and generation process. DANP functions over multiple timesteps to manipulate both cross-attention maps and the noise prediction process, using a dynamic threshold to generate masks that identify text-relevant and irrelevant regions. It then reduces attention in relevant areas while increasing it in irrelevant ones, thereby misguides the edit towards incorrect regions and preserves the intended targets. Additionally, our method maximizes the discrepancy between the injected noise and the model's predicted noise to further interfere with the generation. By targeting both attention and noise prediction mechanisms, DANP exhibits impressive immunity against malicious edits, and extensive experiments confirm that our method achieves state-of-the-art performance.
>
---
#### [new 048] ProtoFlow: Interpretable and Robust Surgical Workflow Modeling with Learned Dynamic Scene Graph Prototypes
- **分类: cs.CV; cs.AI**

- **简介: 该论文属手术流程建模任务，旨在解决标注成本高、数据少及模型不可解释问题。提出ProtoFlow框架，用自监督预训练+原型微调的GNN架构，学习动态场景图原型，实现可解释、鲁棒的细粒度手术识别。**

- **链接: [https://arxiv.org/pdf/2512.14092v1](https://arxiv.org/pdf/2512.14092v1)**

> **作者:** Felix Holm; Ghazal Ghazaei; Nassir Navab
>
> **摘要:** Purpose: Detailed surgical recognition is critical for advancing AI-assisted surgery, yet progress is hampered by high annotation costs, data scarcity, and a lack of interpretable models. While scene graphs offer a structured abstraction of surgical events, their full potential remains untapped. In this work, we introduce ProtoFlow, a novel framework that learns dynamic scene graph prototypes to model complex surgical workflows in an interpretable and robust manner. Methods: ProtoFlow leverages a graph neural network (GNN) encoder-decoder architecture that combines self-supervised pretraining for rich representation learning with a prototype-based fine-tuning stage. This process discovers and refines core prototypes that encapsulate recurring, clinically meaningful patterns of surgical interaction, forming an explainable foundation for workflow analysis. Results: We evaluate our approach on the fine-grained CAT-SG dataset. ProtoFlow not only outperforms standard GNN baselines in overall accuracy but also demonstrates exceptional robustness in limited-data, few-shot scenarios, maintaining strong performance when trained on as few as one surgical video. Our qualitative analyses further show that the learned prototypes successfully identify distinct surgical sub-techniques and provide clear, interpretable insights into workflow deviations and rare complications. Conclusion: By uniting robust representation learning with inherent explainability, ProtoFlow represents a significant step toward developing more transparent, reliable, and data-efficient AI systems, accelerating their potential for clinical adoption in surgical training, real-time decision support, and workflow optimization.
>
---
#### [new 049] CAPRMIL: Context-Aware Patch Representations for Multiple Instance Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向计算病理学中的弱监督学习任务，解决全切片图像（WSI）分析中因缺乏像素级标注而依赖多实例学习（MIL）的挑战。提出CAPRMIL框架，通过冻结patch编码器、引入全局上下文感知token和轻量自注意力，生成上下文感知的patch表征，再用简单均值聚合，显著提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2512.14540v1](https://arxiv.org/pdf/2512.14540v1)**

> **作者:** Andreas Lolos; Theofilos Christodoulou; Aris L. Moustakas; Stergios Christodoulidis; Maria Vakalopoulou
>
> **备注:** 24 pages, 12 Figures, 4 Tables
>
> **摘要:** In computational pathology, weak supervision has become the standard for deep learning due to the gigapixel scale of WSIs and the scarcity of pixel-level annotations, with Multiple Instance Learning (MIL) established as the principal framework for slide-level model training. In this paper, we introduce a novel setting for MIL methods, inspired by proceedings in Neural Partial Differential Equation (PDE) Solvers. Instead of relying on complex attention-based aggregation, we propose an efficient, aggregator-agnostic framework that removes the complexity of correlation learning from the MIL aggregator. CAPRMIL produces rich context-aware patch embeddings that promote effective correlation learning on downstream tasks. By projecting patch features -- extracted using a frozen patch encoder -- into a small set of global context/morphology-aware tokens and utilizing multi-head self-attention, CAPRMIL injects global context with linear computational complexity with respect to the bag size. Paired with a simple Mean MIL aggregator, CAPRMIL matches state-of-the-art slide-level performance across multiple public pathology benchmarks, while reducing the total number of trainable parameters by 48%-92.8% versus SOTA MILs, lowering FLOPs during inference by 52%-99%, and ranking among the best models on GPU memory efficiency and training time. Our results indicate that learning rich, context-aware instance representations before aggregation is an effective and scalable alternative to complex pooling for whole-slide analysis. Our code is available at https://github.com/mandlos/CAPRMIL
>
---
#### [new 050] Coarse-to-Fine Hierarchical Alignment for UAV-based Human Detection using Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属UAV图像中的人体检测任务，旨在解决合成数据与真实数据间域偏移导致检测性能下降的问题。提出基于扩散模型的粗到细分层对齐框架（CFHA），含全局风格迁移、局部细节增强和幻觉去除三模块，显著提升跨域检测精度。**

- **链接: [https://arxiv.org/pdf/2512.13869v1](https://arxiv.org/pdf/2512.13869v1)**

> **作者:** Wenda Li; Meng Wu; Sungmin Eum; Heesung Kwon; Qing Qu
>
> **摘要:** Training object detectors demands extensive, task-specific annotations, yet this requirement becomes impractical in UAV-based human detection due to constantly shifting target distributions and the scarcity of labeled images. As a remedy, synthetic simulators are adopted to generate annotated data, with a low annotation cost. However, the domain gap between synthetic and real images hinders the model from being effectively applied to the target domain. Accordingly, we introduce Coarse-to-Fine Hierarchical Alignment (CFHA), a three-stage diffusion-based framework designed to transform synthetic data for UAV-based human detection, narrowing the domain gap while preserving the original synthetic labels. CFHA explicitly decouples global style and local content domain discrepancies and bridges those gaps using three modules: (1) Global Style Transfer -- a diffusion model aligns color, illumination, and texture statistics of synthetic images to the realistic style, using only a small real reference set; (2) Local Refinement -- a super-resolution diffusion model is used to facilitate fine-grained and photorealistic details for the small objects, such as human instances, preserving shape and boundary integrity; (3) Hallucination Removal -- a module that filters out human instances whose visual attributes do not align with real-world data to make the human appearance closer to the target distribution. Extensive experiments on public UAV Sim2Real detection benchmarks demonstrate that our methods significantly improve the detection accuracy compared to the non-transformed baselines. Specifically, our method achieves up to $+14.1$ improvement of mAP50 on Semantic-Drone benchmark. Ablation studies confirm the complementary roles of the global and local stages and highlight the importance of hierarchical alignment. The code is released at \href{https://github.com/liwd190019/CFHA}{this url}.
>
---
#### [new 051] Nexels: Neurally-Textured Surfels for Real-Time Novel View Synthesis with Sparse Geometries
- **分类: cs.CV**

- **简介: 该论文面向新型视角合成任务，旨在解决高纹理场景中点基元（如高斯泼溅）参数量大、内存占用高的问题。提出Nexels表示法：用surfels建模几何，结合全局神经场与逐原始体颜色解耦表征外观，在显著减少 primitives（最高31×）和内存（最高5.5×）的同时，保持甚至提升视觉质量与渲染速度。**

- **链接: [https://arxiv.org/pdf/2512.13796v1](https://arxiv.org/pdf/2512.13796v1)**

> **作者:** Victor Rong; Jan Held; Victor Chu; Daniel Rebain; Marc Van Droogenbroeck; Kiriakos N. Kutulakos; Andrea Tagliasacchi; David B. Lindell
>
> **备注:** Webpage at https://lessvrong.com/cs/nexels
>
> **摘要:** Though Gaussian splatting has achieved impressive results in novel view synthesis, it requires millions of primitives to model highly textured scenes, even when the geometry of the scene is simple. We propose a representation that goes beyond point-based rendering and decouples geometry and appearance in order to achieve a compact representation. We use surfels for geometry and a combination of a global neural field and per-primitive colours for appearance. The neural field textures a fixed number of primitives for each pixel, ensuring that the added compute is low. Our representation matches the perceptual quality of 3D Gaussian splatting while using $9.7\times$ fewer primitives and $5.5\times$ less memory on outdoor scenes and using $31\times$ fewer primitives and $3.7\times$ less memory on indoor scenes. Our representation also renders twice as fast as existing textured primitives while improving upon their visual quality.
>
---
#### [new 052] Unified Semantic Transformer for 3D Scene Understanding
- **分类: cs.CV**

- **简介: 该论文提出UNITE模型，属3D场景理解任务，旨在解决现有方法任务专用、难以泛化的问题。它基于RGB图像，用统一Transformer端到端预测3D语义分割、实例嵌入、开放词汇特征、功能与关节约束等，通过2D蒸馏与多视图自监督训练，实现高效高精度3D理解。**

- **链接: [https://arxiv.org/pdf/2512.14364v1](https://arxiv.org/pdf/2512.14364v1)**

> **作者:** Sebastian Koch; Johanna Wald; Hide Matsuki; Pedro Hermosilla; Timo Ropinski; Federico Tombari
>
> **备注:** Project page: https://unite-page.github.io/
>
> **摘要:** Holistic 3D scene understanding involves capturing and parsing unstructured 3D environments. Due to the inherent complexity of the real world, existing models have predominantly been developed and limited to be task-specific. We introduce UNITE, a Unified Semantic Transformer for 3D scene understanding, a novel feed-forward neural network that unifies a diverse set of 3D semantic tasks within a single model. Our model operates on unseen scenes in a fully end-to-end manner and only takes a few seconds to infer the full 3D semantic geometry. Our approach is capable of directly predicting multiple semantic attributes, including 3D scene segmentation, instance embeddings, open-vocabulary features, as well as affordance and articulations, solely from RGB images. The method is trained using a combination of 2D distillation, heavily relying on self-supervision and leverages novel multi-view losses designed to ensure 3D view consistency. We demonstrate that UNITE achieves state-of-the-art performance on several different semantic tasks and even outperforms task-specific models, in many cases, surpassing methods that operate on ground truth 3D geometry. See the project website at unite-page.github.io
>
---
#### [new 053] A4-Agent: An Agentic Framework for Zero-Shot Affordance Reasoning
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向具身AI中的零样本可供性推理任务，旨在解决现有模型泛化差、依赖标注数据的问题。提出A4-Agent框架，无需训练，通过Dreamer（可视化交互）、Thinker（识别交互部件）、Spotter（精确定位）三阶段解耦推理，利用多类基础模型协同完成语言驱动的交互区域预测。**

- **链接: [https://arxiv.org/pdf/2512.14442v1](https://arxiv.org/pdf/2512.14442v1)**

> **作者:** Zixin Zhang; Kanghao Chen; Hanqing Wang; Hongfei Zhang; Harold Haodong Chen; Chenfei Liao; Litao Guo; Ying-Cong Chen
>
> **摘要:** Affordance prediction, which identifies interaction regions on objects based on language instructions, is critical for embodied AI. Prevailing end-to-end models couple high-level reasoning and low-level grounding into a single monolithic pipeline and rely on training over annotated datasets, which leads to poor generalization on novel objects and unseen environments. In this paper, we move beyond this paradigm by proposing A4-Agent, a training-free agentic framework that decouples affordance prediction into a three-stage pipeline. Our framework coordinates specialized foundation models at test time: (1) a $\textbf{Dreamer}$ that employs generative models to visualize $\textit{how}$ an interaction would look; (2) a $\textbf{Thinker}$ that utilizes large vision-language models to decide $\textit{what}$ object part to interact with; and (3) a $\textbf{Spotter}$ that orchestrates vision foundation models to precisely locate $\textit{where}$ the interaction area is. By leveraging the complementary strengths of pre-trained models without any task-specific fine-tuning, our zero-shot framework significantly outperforms state-of-the-art supervised methods across multiple benchmarks and demonstrates robust generalization to real-world settings.
>
---
#### [new 054] Bridging Fidelity-Reality with Controllable One-Step Diffusion for Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文面向图像超分辨率任务，旨在解决扩散模型在单步推理中 fidelity 低、生成先验区域激活不足、文本提示与语义区域错位三大问题。提出 CODSR 方法：LQ 指导特征调制、区域自适应先验激活、文本匹配引导，实现高保真与高感知质量的高效单步超分。**

- **链接: [https://arxiv.org/pdf/2512.14061v1](https://arxiv.org/pdf/2512.14061v1)**

> **作者:** Hao Chen; Junyang Chen; Jinshan Pan; Jiangxin Dong
>
> **备注:** Project page: https://github.com/Chanson94/CODSR
>
> **摘要:** Recent diffusion-based one-step methods have shown remarkable progress in the field of image super-resolution, yet they remain constrained by three critical limitations: (1) inferior fidelity performance caused by the information loss from compression encoding of low-quality (LQ) inputs; (2) insufficient region-discriminative activation of generative priors; (3) misalignment between text prompts and their corresponding semantic regions. To address these limitations, we propose CODSR, a controllable one-step diffusion network for image super-resolution. First, we propose an LQ-guided feature modulation module that leverages original uncompressed information from LQ inputs to provide high-fidelity conditioning for the diffusion process. We then develop a region-adaptive generative prior activation method to effectively enhance perceptual richness without sacrificing local structural fidelity. Finally, we employ a text-matching guidance strategy to fully harness the conditioning potential of text prompts. Extensive experiments demonstrate that CODSR achieves superior perceptual quality and competitive fidelity compared with state-of-the-art methods with efficient one-step inference.
>
---
#### [new 055] CRISP: Contact-Guided Real2Sim from Monocular Video with Planar Scene Primitives
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 论文提出CRISP方法，解决单目视频中人体-场景联合重建不满足物理仿真需求的问题。通过平面几何拟合、接触引导的遮挡恢复和强化学习驱动的物理验证，生成清洁、凸、可仿真的场景与人体运动，显著降低跟踪失败率并提升仿真效率。**

- **链接: [https://arxiv.org/pdf/2512.14696v1](https://arxiv.org/pdf/2512.14696v1)**

> **作者:** Zihan Wang; Jiashun Wang; Jeff Tan; Yiwen Zhao; Jessica Hodgins; Shubham Tulsiani; Deva Ramanan
>
> **备注:** Project page: https://crisp-real2sim.github.io/CRISP-Real2Sim/
>
> **摘要:** We introduce CRISP, a method that recovers simulatable human motion and scene geometry from monocular video. Prior work on joint human-scene reconstruction relies on data-driven priors and joint optimization with no physics in the loop, or recovers noisy geometry with artifacts that cause motion tracking policies with scene interactions to fail. In contrast, our key insight is to recover convex, clean, and simulation-ready geometry by fitting planar primitives to a point cloud reconstruction of the scene, via a simple clustering pipeline over depth, normals, and flow. To reconstruct scene geometry that might be occluded during interactions, we make use of human-scene contact modeling (e.g., we use human posture to reconstruct the occluded seat of a chair). Finally, we ensure that human and scene reconstructions are physically-plausible by using them to drive a humanoid controller via reinforcement learning. Our approach reduces motion tracking failure rates from 55.2\% to 6.9\% on human-centric video benchmarks (EMDB, PROX), while delivering a 43\% faster RL simulation throughput. We further validate it on in-the-wild videos including casually-captured videos, Internet videos, and even Sora-generated videos. This demonstrates CRISP's ability to generate physically-valid human motion and interaction environments at scale, greatly advancing real-to-sim applications for robotics and AR/VR.
>
---
#### [new 056] Selective, Controlled and Domain-Agnostic Unlearning in Pretrained CLIP: A Training- and Data-Free Approach
- **分类: cs.CV**

- **简介: 该论文研究模型“选择性遗忘”任务，旨在无需训练数据或微调，从预训练CLIP中可控、领域无关地移除特定类别知识。提出一种训练与数据免费的框架，支持全局、领域特定及选择性领域完全遗忘，利用多模态零空间和合成视觉原型实现高效精准遗忘。**

- **链接: [https://arxiv.org/pdf/2512.14113v1](https://arxiv.org/pdf/2512.14113v1)**

> **作者:** Ashish Mishra; Gyanaranjan Nayak; Tarun Kumar; Arpit Shah; Suparna Bhattacharya; Martin Foltin
>
> **摘要:** Pretrained models like CLIP have demonstrated impressive zero-shot classification capabilities across diverse visual domains, spanning natural images, artistic renderings, and abstract representations. However, real-world applications often demand the removal (or "unlearning") of specific object classes without requiring additional data or retraining, or affecting the model's performance on unrelated tasks. In this paper, we propose a novel training- and data-free unlearning framework that enables three distinct forgetting paradigms: (1) global unlearning of selected objects across all domains, (2) domain-specific knowledge removal (e.g., eliminating sketch representations while preserving photo recognition), and (3) complete unlearning in selective domains. By leveraging a multimodal nullspace through synergistic integration of text prompts and synthesized visual prototypes derived from CLIP's joint embedding space, our method efficiently removes undesired class information while preserving the remaining knowledge. This approach overcomes the limitations of existing retraining-based methods and offers a flexible and computationally efficient solution for controlled model forgetting.
>
---
#### [new 057] SignIT: A Comprehensive Dataset and Multimodal Analysis for Italian Sign Language Recognition
- **分类: cs.CV**

- **简介: 该论文面向意大利手语（LIS）识别任务，旨在解决现有数据匮乏与模型性能不足问题。作者构建了含644段视频、94类手势的SignIT多模态数据集，提供2D关键点与RGB帧，并建立基准测试，验证多种SOTA模型在LIS上的局限性。**

- **链接: [https://arxiv.org/pdf/2512.14489v1](https://arxiv.org/pdf/2512.14489v1)**

> **作者:** Alessia Micieli; Giovanni Maria Farinella; Francesco Ragusa
>
> **摘要:** In this work we present SignIT, a new dataset to study the task of Italian Sign Language (LIS) recognition. The dataset is composed of 644 videos covering 3.33 hours. We manually annotated videos considering a taxonomy of 94 distinct sign classes belonging to 5 macro-categories: Animals, Food, Colors, Emotions and Family. We also extracted 2D keypoints related to the hands, face and body of the users. With the dataset, we propose a benchmark for the sign recognition task, adopting several state-of-the-art models showing how temporal information, 2D keypoints and RGB frames can be influence the performance of these models. Results show the limitations of these models on this challenging LIS dataset. We release data and annotations at the following link: https://fpv-iplab.github.io/SignIT/.
>
---
#### [new 058] KFS-Bench: Comprehensive Evaluation of Key Frame Sampling in Long Video Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向长视频问答任务，解决关键帧采样质量难以直接评估的问题。提出首个专用基准KFS-Bench，含多场景标注，支持对采样策略的直接、鲁棒评估；并设计新采样方法与质量度量，提升场景覆盖与问答性能。**

- **链接: [https://arxiv.org/pdf/2512.14017v1](https://arxiv.org/pdf/2512.14017v1)**

> **作者:** Zongyao Li; Kengo Ishida; Satoshi Yamazaki; Xiaotong Ji; Jianquan Liu
>
> **备注:** WACV2026
>
> **摘要:** We propose KFS-Bench, the first benchmark for key frame sampling in long video question answering (QA), featuring multi-scene annotations to enable direct and robust evaluation of sampling strategies. Key frame sampling is crucial for efficient long-form video understanding. In long video QA, selecting informative frames enables multimodal large language models (MLLMs) to improve both accuracy and efficiency. KFS-Bench addresses the limitation of prior works that only indirectly assess frame selection quality via QA accuracy. By providing ground-truth annotations of multiple disjoint scenes required per question, KFS-Bench allows us to directly analyze how different sampling approaches capture essential content across an entire long video. Using KFS-Bench, we conduct a comprehensive study of key frame sampling methods and identify that not only sampling precision but also scene coverage and sampling balance are the key factors influencing QA performance. Regarding all the factors, we design a novel sampling quality metric that correlates with QA accuracy. Furthermore, we develop a novel key frame sampling method that leverages question-video relevance to balance sampling diversity against question-frame similarity, thereby improving coverage of relevant scenes. Our adaptively balanced sampling approach achieves superior performance in both key frame sampling and QA performance. The benchmark is available at https://github.com/NEC-VID/KFS-Bench.
>
---
#### [new 059] LCMem: A Universal Model for Robust Image Memorization Detection
- **分类: cs.CV**

- **简介: 该论文提出LCMem模型，解决生成图像中数据记忆化检测这一隐私审计任务。针对现有方法泛化差、评估弱的问题，将记忆化检测统一为重识别与复制检测的联合问题，通过两阶段对比学习实现跨域鲁棒检测，在多项基准上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.14421v1](https://arxiv.org/pdf/2512.14421v1)**

> **作者:** Mischa Dombrowski; Felix Nützel; Bernhard Kainz
>
> **摘要:** Recent advances in generative image modeling have achieved visual realism sufficient to deceive human experts, yet their potential for privacy preserving data sharing remains insufficiently understood. A central obstacle is the absence of reliable memorization detection mechanisms, limited quantitative evaluation, and poor generalization of existing privacy auditing methods across domains. To address this, we propose to view memorization detection as a unified problem at the intersection of re-identification and copy detection, whose complementary goals cover both identity consistency and augmentation-robust duplication, and introduce Latent Contrastive Memorization Network (LCMem), a cross-domain model evaluated jointly on both tasks. LCMem achieves this through a two-stage training strategy that first learns identity consistency before incorporating augmentation-robust copy detection. Across six benchmark datasets, LCMem achieves improvements of up to 16 percentage points on re-identification and 30 percentage points on copy detection, enabling substantially more reliable memorization detection at scale. Our results show that existing privacy filters provide limited performance and robustness, highlighting the need for stronger protection mechanisms. We show that LCMem sets a new standard for cross-domain privacy auditing, offering reliable and scalable memorization detection. Code and model is publicly available at https://github.com/MischaD/LCMem.
>
---
#### [new 060] Distill Video Datasets into Images
- **分类: cs.CV**

- **简介: 该论文属数据集蒸馏任务，旨在解决视频数据蒸馏难、性能差的问题。提出SFVD框架：将视频蒸馏为单帧图像，通过可微插值生成伪视频，并融合真实视频的时序信息，显著提升蒸馏效果。**

- **链接: [https://arxiv.org/pdf/2512.14621v1](https://arxiv.org/pdf/2512.14621v1)**

> **作者:** Zhenghao Zhao; Haoxuan Wang; Kai Wang; Yuzhang Shang; Yuan Hong; Yan Yan
>
> **摘要:** Dataset distillation aims to synthesize compact yet informative datasets that allow models trained on them to achieve performance comparable to training on the full dataset. While this approach has shown promising results for image data, extending dataset distillation methods to video data has proven challenging and often leads to suboptimal performance. In this work, we first identify the core challenge in video set distillation as the substantial increase in learnable parameters introduced by the temporal dimension of video, which complicates optimization and hinders convergence. To address this issue, we observe that a single frame is often sufficient to capture the discriminative semantics of a video. Leveraging this insight, we propose Single-Frame Video set Distillation (SFVD), a framework that distills videos into highly informative frames for each class. Using differentiable interpolation, these frames are transformed into video sequences and matched with the original dataset, while updates are restricted to the frames themselves for improved optimization efficiency. To further incorporate temporal information, the distilled frames are combined with sampled real videos from real videos during the matching process through a channel reshaping layer. Extensive experiments on multiple benchmarks demonstrate that SFVD substantially outperforms prior methods, achieving improvements of up to 5.3% on MiniUCF, thereby offering a more effective solution.
>
---
#### [new 061] Robust Single-shot Structured Light 3D Imaging via Neural Feature Decoding
- **分类: cs.CV**

- **简介: 该论文针对单次拍摄结构光3D成像中像素域匹配鲁棒性差的问题，提出基于神经特征解码的学习框架：在特征空间构建成本体进行匹配，并引入单目深度先验进行 refine；通过物理渲染生成百万级合成数据训练，仅用合成数据即实现对真实室内场景的泛化。**

- **链接: [https://arxiv.org/pdf/2512.14028v1](https://arxiv.org/pdf/2512.14028v1)**

> **作者:** Jiaheng Li; Qiyu Dai; Lihan Li; Praneeth Chakravarthula; He Sun; Baoquan Chen; Wenzheng Chen
>
> **摘要:** We consider the problem of active 3D imaging using single-shot structured light systems, which are widely employed in commercial 3D sensing devices such as Apple Face ID and Intel RealSense. Traditional structured light methods typically decode depth correspondences through pixel-domain matching algorithms, resulting in limited robustness under challenging scenarios like occlusions, fine-structured details, and non-Lambertian surfaces. Inspired by recent advances in neural feature matching, we propose a learning-based structured light decoding framework that performs robust correspondence matching within feature space rather than the fragile pixel domain. Our method extracts neural features from the projected patterns and captured infrared (IR) images, explicitly incorporating their geometric priors by building cost volumes in feature space, achieving substantial performance improvements over pixel-domain decoding approaches. To further enhance depth quality, we introduce a depth refinement module that leverages strong priors from large-scale monocular depth estimation models, improving fine detail recovery and global structural coherence. To facilitate effective learning, we develop a physically-based structured light rendering pipeline, generating nearly one million synthetic pattern-image pairs with diverse objects and materials for indoor settings. Experiments demonstrate that our method, trained exclusively on synthetic data with multiple structured light patterns, generalizes well to real-world indoor environments, effectively processes various pattern types without retraining, and consistently outperforms both commercial structured light systems and passive stereo RGB-based depth estimation methods. Project page: https://namisntimpot.github.io/NSLweb/.
>
---
#### [new 062] Time-aware UNet and super-resolution deep residual networks for spatial downscaling
- **分类: cs.CV; cs.LG; eess.IV; stat.ML**

- **简介: 该论文面向大气污染物（臭氧）空间降尺度任务，解决卫星数据空间分辨率粗、难用于局地分析的问题。提出两种时间感知模型：在SRDRN和UNet中嵌入轻量级时间编码模块（正弦或RBF），融合时空特征。实验表明其显著提升性能与收敛速度。**

- **链接: [https://arxiv.org/pdf/2512.13753v1](https://arxiv.org/pdf/2512.13753v1)**

> **作者:** Mika Sipilä; Sabrina Maggio; Sandra De Iaco; Klaus Nordhausen; Monica Palma; Sara Taskinen
>
> **摘要:** Satellite data of atmospheric pollutants are often available only at coarse spatial resolution, limiting their applicability in local-scale environmental analysis and decision-making. Spatial downscaling methods aim to transform the coarse satellite data into high-resolution fields. In this work, two widely used deep learning architectures, the super-resolution deep residual network (SRDRN) and the encoder-decoder-based UNet, are considered for spatial downscaling of tropospheric ozone. Both methods are extended with a lightweight temporal module, which encodes observation time using either sinusoidal or radial basis function (RBF) encoding, and fuses the temporal features with the spatial representations in the networks. The proposed time-aware extensions are evaluated against their baseline counterparts in a case study on ozone downscaling over Italy. The results suggest that, while only slightly increasing computational complexity, the temporal modules significantly improve downscaling performance and convergence speed.
>
---
#### [new 063] Unleashing the Power of Image-Tabular Self-Supervised Learning via Breaking Cross-Tabular Barriers
- **分类: cs.CV**

- **简介: 该论文属多模态自监督学习任务，旨在解决现有图像-表格SSL方法因刚性表格建模导致的跨队列泛化能力差问题。提出CITab框架：引入列头语义建模，并设计原型引导的混合线性层（P-MoLin）以提升表格异质性建模与医学概念挖掘能力。**

- **链接: [https://arxiv.org/pdf/2512.14026v1](https://arxiv.org/pdf/2512.14026v1)**

> **作者:** Yibing Fu; Yunpeng Zhao; Zhitao Zeng; Cheng Chen; Yueming Jin
>
> **摘要:** Multi-modal learning integrating medical images and tabular data has significantly advanced clinical decision-making in recent years. Self-Supervised Learning (SSL) has emerged as a powerful paradigm for pretraining these models on large-scale unlabeled image-tabular data, aiming to learn discriminative representations. However, existing SSL methods for image-tabular representation learning are often confined to specific data cohorts, mainly due to their rigid tabular modeling mechanisms when modeling heterogeneous tabular data. This inter-tabular barrier hinders the multi-modal SSL methods from effectively learning transferrable medical knowledge shared across diverse cohorts. In this paper, we propose a novel SSL framework, namely CITab, designed to learn powerful multi-modal feature representations in a cross-tabular manner. We design the tabular modeling mechanism from a semantic-awareness perspective by integrating column headers as semantic cues, which facilitates transferrable knowledge learning and the scalability in utilizing multiple data sources for pretraining. Additionally, we propose a prototype-guided mixture-of-linear layer (P-MoLin) module for tabular feature specialization, empowering the model to effectively handle the heterogeneity of tabular data and explore the underlying medical concepts. We conduct comprehensive evaluations on Alzheimer's disease diagnosis task across three publicly available data cohorts containing 4,461 subjects. Experimental results demonstrate that CITab outperforms state-of-the-art approaches, paving the way for effective and scalable cross-tabular multi-modal learning.
>
---
#### [new 064] Enhancing Visual Programming for Visual Reasoning via Probabilistic Graphs
- **分类: cs.CV**

- **简介: 该论文属视觉推理任务，旨在解决视觉编程（VP）中因非可微性及子任务无标签导致的端到端优化难题。提出EVPG方法，构建依赖关系驱动的有向概率图，将VP执行转化为可微概率推理，从而用最终标签实现梯度优化。**

- **链接: [https://arxiv.org/pdf/2512.14257v1](https://arxiv.org/pdf/2512.14257v1)**

> **作者:** Wentao Wan; Kaiyu Wu; Qingyang Ma; Nan Kang; Yunjie Chen; Liang Lin; Keze Wang
>
> **备注:** 13 Pages, 12 figures
>
> **摘要:** Recently, Visual Programming (VP) based on large language models (LLMs) has rapidly developed and demonstrated significant potential in complex Visual Reasoning (VR) tasks. Previous works to enhance VP have primarily focused on improving the quality of LLM-generated visual programs. However, they have neglected to optimize the VP-invoked pre-trained models, which serve as modules for the visual sub-tasks decomposed from the targeted tasks by VP. The difficulty is that there are only final labels of targeted VR tasks rather than labels of sub-tasks. Besides, the non-differentiable nature of VP impedes the direct use of efficient gradient-based optimization methods to leverage final labels for end-to-end learning of the entire VP framework. To overcome these issues, we propose EVPG, a method to Enhance Visual Programming for visual reasoning via Probabilistic Graphs. Specifically, we creatively build a directed probabilistic graph according to the variable dependency relationships during the VP executing process, which reconstructs the non-differentiable VP executing process into a differentiable exact probability inference process on this directed probabilistic graph. As a result, this enables the VP framework to utilize the final labels for efficient, gradient-based optimization in end-to-end supervised learning on targeted VR tasks. Extensive and comprehensive experiments demonstrate the effectiveness and advantages of our EVPG, showing significant performance improvements for VP on three classical complex VR tasks: GQA, NLVRv2, and Open Images.
>
---
#### [new 065] Beyond a Single Light: A Large-Scale Aerial Dataset for Urban Scene Reconstruction Under Varying Illumination
- **分类: cs.CV**

- **简介: 该论文属大尺度城市场景三维重建任务，旨在解决多时段无人机采集中光照变化导致的色彩伪影与几何失真问题。作者构建了SkyLume数据集：含10城区、10万+多视角图像（三时段采集），配LiDAR真值与新指标TCC，支撑光照鲁棒的逆渲染与重建研究。**

- **链接: [https://arxiv.org/pdf/2512.14200v1](https://arxiv.org/pdf/2512.14200v1)**

> **作者:** Zhuoxiao Li; Wenzong Ma; Taoyu Wu; Jinjing Zhu; Zhenchao Q; Shuai Zhang; Jing Ou; Yinrui Ren; Weiqing Qi; Guobin Shen; Hui Xiong; Wufan Zhao
>
> **摘要:** Recent advances in Neural Radiance Fields and 3D Gaussian Splatting have demonstrated strong potential for large-scale UAV-based 3D reconstruction tasks by fitting the appearance of images. However, real-world large-scale captures are often based on multi-temporal data capture, where illumination inconsistencies across different times of day can significantly lead to color artifacts, geometric inaccuracies, and inconsistent appearance. Due to the lack of UAV datasets that systematically capture the same areas under varying illumination conditions, this challenge remains largely underexplored. To fill this gap, we introduceSkyLume, a large-scale, real-world UAV dataset specifically designed for studying illumination robust 3D reconstruction in urban scene modeling: (1) We collect data from 10 urban regions data comprising more than 100k high resolution UAV images (four oblique views and nadir), where each region is captured at three periods of the day to systematically isolate illumination changes. (2) To support precise evaluation of geometry and appearance, we provide per-scene LiDAR scans and accurate 3D ground-truth for assessing depth, surface normals, and reconstruction quality under varying illumination. (3) For the inverse rendering task, we introduce the Temporal Consistency Coefficient (TCC), a metric that measuress cross-time albedo stability and directly evaluates the robustness of the disentanglement of light and material. We aim for this resource to serve as a foundation that advances research and real-world evaluation in large-scale inverse rendering, geometry reconstruction, and novel view synthesis.
>
---
#### [new 066] TACK Tunnel Data (TTD): A Benchmark Dataset for Deep Learning-Based Defect Detection in Tunnels
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉中的缺陷检测任务，旨在解决隧道巡检中缺乏高质量标注数据的问题。作者构建并公开发布TACK Tunnel Data（TTD）基准数据集，包含三种衬砌类型、标注了裂缝、渗水、析出等典型缺陷的图像，支持监督/半监督/无监督深度学习方法，提升模型泛化与迁移能力。**

- **链接: [https://arxiv.org/pdf/2512.14477v1](https://arxiv.org/pdf/2512.14477v1)**

> **作者:** Andreas Sjölander; Valeria Belloni; Robel Fekadu; Andrea Nascetti
>
> **摘要:** Tunnels are essential elements of transportation infrastructure, but are increasingly affected by ageing and deterioration mechanisms such as cracking. Regular inspections are required to ensure their safety, yet traditional manual procedures are time-consuming, subjective, and costly. Recent advances in mobile mapping systems and Deep Learning (DL) enable automated visual inspections. However, their effectiveness is limited by the scarcity of tunnel datasets. This paper introduces a new publicly available dataset containing annotated images of three different tunnel linings, capturing typical defects: cracks, leaching, and water infiltration. The dataset is designed to support supervised, semi-supervised, and unsupervised DL methods for defect detection and segmentation. Its diversity in texture and construction techniques also enables investigation of model generalization and transferability across tunnel types. By addressing the critical lack of domain-specific data, this dataset contributes to advancing automated tunnel inspection and promoting safer, more efficient infrastructure maintenance strategies.
>
---
#### [new 067] Route-DETR: Pairwise Query Routing in Transformers for Object Detection
- **分类: cs.CV**

- **简介: 该论文属目标检测任务，旨在解决DETR中查询间冗余竞争导致的低效问题。提出Route-DETR，通过解码器自注意力中的自适应成对路由机制，区分竞争与互补查询，引入抑制/委派路由及低秩可学习偏置，并采用双分支训练策略，在不增推理开销下提升精度。**

- **链接: [https://arxiv.org/pdf/2512.13876v1](https://arxiv.org/pdf/2512.13876v1)**

> **作者:** Ye Zhang; Qi Chen; Wenyou Huang; Rui Liu; Zhengjian Kang
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Detection Transformer (DETR) offers an end-to-end solution for object detection by eliminating hand-crafted components like non-maximum suppression. However, DETR suffers from inefficient query competition where multiple queries converge to similar positions, leading to redundant computations. We present Route-DETR, which addresses these issues through adaptive pairwise routing in decoder self-attention layers. Our key insight is distinguishing between competing queries (targeting the same object) versus complementary queries (targeting different objects) using inter-query similarity, confidence scores, and geometry. We introduce dual routing mechanisms: suppressor routes that modulate attention between competing queries to reduce duplication, and delegator routes that encourage exploration of different regions. These are implemented via learnable low-rank attention biases enabling asymmetric query interactions. A dual-branch training strategy incorporates routing biases only during training while preserving standard attention for inference, ensuring no additional computational cost. Experiments on COCO and Cityscapes demonstrate consistent improvements across multiple DETR baselines, achieving +1.7% mAP gain over DINO on ResNet-50 and reaching 57.6% mAP on Swin-L, surpassing prior state-of-the-art models.
>
---
#### [new 068] Complex Mathematical Expression Recognition: Benchmark, Large-Scale Dataset and Strong Baseline
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦数学表达式识别（MER）任务，旨在解决复杂多行表达式识别准确率低的问题。工作包括：构建三难度基准CMER-Bench；发布大规模复杂表达式数据集MER-17M和CMER-3M；提出结构化表示与新tokenizer；设计轻量模型CMERNet，在基准上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.13731v1](https://arxiv.org/pdf/2512.13731v1)**

> **作者:** Weikang Bai; Yongkun Du; Yuchen Su; Yazhen Xie; Zhineng Chen
>
> **摘要:** Mathematical Expression Recognition (MER) has made significant progress in recognizing simple expressions, but the robust recognition of complex mathematical expressions with many tokens and multiple lines remains a formidable challenge. In this paper, we first introduce CMER-Bench, a carefully constructed benchmark that categorizes expressions into three difficulty levels: easy, moderate, and complex. Leveraging CMER-Bench, we conduct a comprehensive evaluation of existing MER models and general-purpose multimodal large language models (MLLMs). The results reveal that while current methods perform well on easy and moderate expressions, their performance degrades significantly when handling complex mathematical expressions, mainly because existing public training datasets are primarily composed of simple samples. In response, we propose MER-17M and CMER-3M that are large-scale datasets emphasizing the recognition of complex mathematical expressions. The datasets provide rich and diverse samples to support the development of accurate and robust complex MER models. Furthermore, to address the challenges posed by the complicated spatial layout of complex expressions, we introduce a novel expression tokenizer, and a new representation called Structured Mathematical Language, which explicitly models the hierarchical and spatial structure of expressions beyond LaTeX format. Based on these, we propose a specialized model named CMERNet, built upon an encoder-decoder architecture and trained on CMER-3M. Experimental results show that CMERNet, with only 125 million parameters, significantly outperforms existing MER models and MLLMs on CMER-Bench.
>
---
#### [new 069] SketchAssist: A Practical Assistant for Semantic Edits and Precise Local Redrawing
- **分类: cs.CV**

- **简介: 该论文面向草图编辑任务，解决现有方法难以兼顾语义修改与局部精准重绘、同时保持结构与风格一致的问题。提出SketchAssist系统：构建可控数据生成管线，设计统一DiT框架，支持指令驱动全局编辑与线条引导局部重绘，并引入任务导向MoE-LoRA提升控制性与保真度。**

- **链接: [https://arxiv.org/pdf/2512.14140v1](https://arxiv.org/pdf/2512.14140v1)**

> **作者:** Han Zou; Yan Zhang; Ruiqi Yu; Cong Xie; Jie Huang; Zhenpeng Zhan
>
> **摘要:** Sketch editing is central to digital illustration, yet existing image editing systems struggle to preserve the sparse, style-sensitive structure of line art while supporting both high-level semantic changes and precise local redrawing. We present SketchAssist, an interactive sketch drawing assistant that accelerates creation by unifying instruction-guided global edits with line-guided region redrawing, while keeping unrelated regions and overall composition intact. To enable this assistant at scale, we introduce a controllable data generation pipeline that (i) constructs attribute-addition sequences from attribute-free base sketches, (ii) forms multi-step edit chains via cross-sequence sampling, and (iii) expands stylistic coverage with a style-preserving attribute-removal model applied to diverse sketches. Building on this data, SketchAssist employs a unified sketch editing framework with minimal changes to DiT-based editors. We repurpose the RGB channels to encode the inputs, enabling seamless switching between instruction-guided edits and line-guided redrawing within a single input interface. To further specialize behavior across modes, we integrate a task-guided mixture-of-experts into LoRA layers, routing by text and visual cues to improve semantic controllability, structural fidelity, and style preservation. Extensive experiments show state-of-the-art results on both tasks, with superior instruction adherence and style/structure preservation compared to recent baselines. Together, our dataset and SketchAssist provide a practical, controllable assistant for sketch creation and revision.
>
---
#### [new 070] 4D-RaDiff: Latent Diffusion for 4D Radar Point Cloud Generation
- **分类: cs.CV**

- **简介: 该论文属雷达点云生成任务，旨在解决4D雷达标注数据稀缺问题。提出4D-RaDiff框架，在潜在空间对稀疏雷达点云建模，支持物体/场景级条件生成，可将无标签框或LiDAR数据转为高质量雷达点云，显著提升检测性能并减少90%真实标注需求。**

- **链接: [https://arxiv.org/pdf/2512.14235v1](https://arxiv.org/pdf/2512.14235v1)**

> **作者:** Jimmie Kwok; Holger Caesar; Andras Palffy
>
> **摘要:** Automotive radar has shown promising developments in environment perception due to its cost-effectiveness and robustness in adverse weather conditions. However, the limited availability of annotated radar data poses a significant challenge for advancing radar-based perception systems. To address this limitation, we propose a novel framework to generate 4D radar point clouds for training and evaluating object detectors. Unlike image-based diffusion, our method is designed to consider the sparsity and unique characteristics of radar point clouds by applying diffusion to a latent point cloud representation. Within this latent space, generation is controlled via conditioning at either the object or scene level. The proposed 4D-RaDiff converts unlabeled bounding boxes into high-quality radar annotations and transforms existing LiDAR point cloud data into realistic radar scenes. Experiments demonstrate that incorporating synthetic radar data of 4D-RaDiff as data augmentation method during training consistently improves object detection performance compared to training on real data only. In addition, pre-training on our synthetic data reduces the amount of required annotated radar data by up to 90% while achieving comparable object detection performance.
>
---
#### [new 071] Optimizing Rank for High-Fidelity Implicit Neural Representations
- **分类: cs.CV**

- **简介: 该论文聚焦隐式神经表示（INR）任务，旨在解决vanilla MLP难以建模高频内容的问题。作者指出主因是训练中网络秩退化，而非架构固有缺陷；提出通过优化器（如Muon）维持高秩、近正交更新来稳定秩，显著提升重建保真度，在多领域实现高达9 dB PSNR提升。**

- **链接: [https://arxiv.org/pdf/2512.14366v1](https://arxiv.org/pdf/2512.14366v1)**

> **作者:** Julian McGinnis; Florian A. Hölzl; Suprosanna Shit; Florentin Bieder; Paul Friedrich; Mark Mühlau; Björn Menze; Daniel Rueckert; Benedikt Wiestler
>
> **摘要:** Implicit Neural Representations (INRs) based on vanilla Multi-Layer Perceptrons (MLPs) are widely believed to be incapable of representing high-frequency content. This has directed research efforts towards architectural interventions, such as coordinate embeddings or specialized activation functions, to represent high-frequency signals. In this paper, we challenge the notion that the low-frequency bias of vanilla MLPs is an intrinsic, architectural limitation to learn high-frequency content, but instead a symptom of stable rank degradation during training. We empirically demonstrate that regulating the network's rank during training substantially improves the fidelity of the learned signal, rendering even simple MLP architectures expressive. Extensive experiments show that using optimizers like Muon, with high-rank, near-orthogonal updates, consistently enhances INR architectures even beyond simple ReLU MLPs. These substantial improvements hold across a diverse range of domains, including natural and medical images, and novel view synthesis, with up to 9 dB PSNR improvements over the previous state-of-the-art. Our project page, which includes code and experimental results, is available at: (https://muon-inrs.github.io).
>
---
#### [new 072] CLNet: Cross-View Correspondence Makes a Stronger Geo-Localizationer
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向跨视角地理定位（IRCVGL）任务，解决卫星与街景图像因视角差异大导致匹配难的问题。提出CLNet框架，含神经对应图、非线性嵌入转换器和全局特征重校准三模块，显式建模空间对应关系，提升定位精度、可解释性与泛化性。**

- **链接: [https://arxiv.org/pdf/2512.14560v1](https://arxiv.org/pdf/2512.14560v1)**

> **作者:** Xianwei Cao; Dou Quan; Shuang Wang; Ning Huyan; Wei Wang; Yunan Li; Licheng Jiao
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Image retrieval-based cross-view geo-localization (IRCVGL) aims to match images captured from significantly different viewpoints, such as satellite and street-level images. Existing methods predominantly rely on learning robust global representations or implicit feature alignment, which often fail to model explicit spatial correspondences crucial for accurate localization. In this work, we propose a novel correspondence-aware feature refinement framework, termed CLNet, that explicitly bridges the semantic and geometric gaps between different views. CLNet decomposes the view alignment process into three learnable and complementary modules: a Neural Correspondence Map (NCM) that spatially aligns cross-view features via latent correspondence fields; a Nonlinear Embedding Converter (NEC) that remaps features across perspectives using an MLP-based transformation; and a Global Feature Recalibration (GFR) module that reweights informative feature channels guided by learned spatial cues. The proposed CLNet can jointly capture both high-level semantics and fine-grained alignments. Extensive experiments on four public benchmarks, CVUSA, CVACT, VIGOR, and University-1652, demonstrate that our proposed CLNet achieves state-of-the-art performance while offering better interpretability and generalizability.
>
---
#### [new 073] The Devil is in Attention Sharing: Improving Complex Non-rigid Image Editing Faithfulness via Attention Synergy
- **分类: cs.CV**

- **简介: 该论文属扩散模型训练-free图像编辑任务，旨在解决复杂非刚性编辑（如姿态/形状变化）中因注意力坍塌导致的过编/欠编问题。提出SynPS方法，通过动态协同位置嵌入与语义信息，自适应调控注意力，提升编辑保真度。**

- **链接: [https://arxiv.org/pdf/2512.14423v1](https://arxiv.org/pdf/2512.14423v1)**

> **作者:** Zhuo Chen; Fanyue Wei; Runze Xu; Jingjing Li; Lixin Duan; Angela Yao; Wen Li
>
> **备注:** Project page:https://synps26.github.io/
>
> **摘要:** Training-free image editing with large diffusion models has become practical, yet faithfully performing complex non-rigid edits (e.g., pose or shape changes) remains highly challenging. We identify a key underlying cause: attention collapse in existing attention sharing mechanisms, where either positional embeddings or semantic features dominate visual content retrieval, leading to over-editing or under-editing.To address this issue, we introduce SynPS, a method that Synergistically leverages Positional embeddings and Semantic information for faithful non-rigid image editing. We first propose an editing measurement that quantifies the required editing magnitude at each denoising step. Based on this measurement, we design an attention synergy pipeline that dynamically modulates the influence of positional embeddings, enabling SynPS to balance semantic modifications and fidelity preservation.By adaptively integrating positional and semantic cues, SynPS effectively avoids both over- and under-editing. Extensive experiments on public and newly curated benchmarks demonstrate the superior performance and faithfulness of our approach.
>
---
#### [new 074] FacEDiT: Unified Talking Face Editing and Generation via Facial Motion Infilling
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FacEDiT，将说话人脸编辑与生成统一为“语音条件下的面部运动补全”任务。通过扩散Transformer与流匹配，实现语音驱动的局部编辑（替换/插入/删除）和生成，兼顾唇同步、身份保持与边界连续性，并构建首个编辑基准FacEDiTBench。**

- **链接: [https://arxiv.org/pdf/2512.14056v1](https://arxiv.org/pdf/2512.14056v1)**

> **作者:** Kim Sung-Bin; Joohyun Chang; David Harwath; Tae-Hyun Oh
>
> **备注:** Project page: https://facedit.github.io/
>
> **摘要:** Talking face editing and face generation have often been studied as distinct problems. In this work, we propose viewing both not as separate tasks but as subtasks of a unifying formulation, speech-conditional facial motion infilling. We explore facial motion infilling as a self-supervised pretext task that also serves as a unifying formulation of dynamic talking face synthesis. To instantiate this idea, we propose FacEDiT, a speech-conditional Diffusion Transformer trained with flow matching. Inspired by masked autoencoders, FacEDiT learns to synthesize masked facial motions conditioned on surrounding motions and speech. This formulation enables both localized generation and edits, such as substitution, insertion, and deletion, while ensuring seamless transitions with unedited regions. In addition, biased attention and temporal smoothness constraints enhance boundary continuity and lip synchronization. To address the lack of a standard editing benchmark, we introduce FacEDiTBench, the first dataset for talking face editing, featuring diverse edit types and lengths, along with new evaluation metrics. Extensive experiments validate that talking face editing and generation emerge as subtasks of speech-conditional motion infilling; FacEDiT produces accurate, speech-aligned facial edits with strong identity preservation and smooth visual continuity while generalizing effectively to talking face generation.
>
---
#### [new 075] Real-time prediction of workplane illuminance distribution for daylight-linked controls using non-intrusive multimodal deep learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属计算机视觉与建筑能源交叉任务，旨在解决动态室内场景下工作面照度实时预测难题。提出非侵入式多模态深度学习框架，仅从侧窗图像提取时空特征，实现高精度、强泛化照度分布预测。**

- **链接: [https://arxiv.org/pdf/2512.14058v1](https://arxiv.org/pdf/2512.14058v1)**

> **作者:** Zulin Zhuang; Yu Bian
>
> **摘要:** Daylight-linked controls (DLCs) have significant potential for energy savings in buildings, especially when abundant daylight is available and indoor workplane illuminance can be accurately predicted in real time. Most existing studies on indoor daylight predictions were developed and tested for static scenes. This study proposes a multimodal deep learning framework that predicts indoor workplane illuminance distributions in real time from non-intrusive images with temporal-spatial features. By extracting image features only from the side-lit window areas rather than interior pixels, the approach remains applicable in dynamically occupied indoor spaces. A field experiment was conducted in a test room in Guangzhou (China), where 17,344 samples were collected for model training and validation. The model achieved R2 > 0.98 with RMSE < 0.14 on the same-distribution test set and R2 > 0.82 with RMSE < 0.17 on an unseen-day test set, indicating high accuracy and acceptable temporal generalization.
>
---
#### [new 076] DRAW2ACT: Turning Depth-Encoded Trajectories into Robotic Demonstration Videos
- **分类: cs.CV; cs.RO**

- **简介: 该论文属机器人视觉-动作生成任务，旨在提升轨迹条件视频生成的可控性与一致性。提出DRAW2ACT框架：利用深度编码轨迹提取多维表征，联合生成对齐的RGB/深度视频，并通过多模态策略模型输出关节角，显著提升视觉质量与操作成功率。**

- **链接: [https://arxiv.org/pdf/2512.14217v1](https://arxiv.org/pdf/2512.14217v1)**

> **作者:** Yang Bai; Liudi Yang; George Eskandar; Fengyi Shen; Mohammad Altillawi; Ziyuan Liu; Gitta Kutyniok
>
> **摘要:** Video diffusion models provide powerful real-world simulators for embodied AI but remain limited in controllability for robotic manipulation. Recent works on trajectory-conditioned video generation address this gap but often rely on 2D trajectories or single modality conditioning, which restricts their ability to produce controllable and consistent robotic demonstrations. We present DRAW2ACT, a depth-aware trajectory-conditioned video generation framework that extracts multiple orthogonal representations from the input trajectory, capturing depth, semantics, shape and motion, and injects them into the diffusion model. Moreover, we propose to jointly generate spatially aligned RGB and depth videos, leveraging cross-modality attention mechanisms and depth supervision to enhance the spatio-temporal consistency. Finally, we introduce a multimodal policy model conditioned on the generated RGB and depth sequences to regress the robot's joint angles. Experiments on Bridge V2, Berkeley Autolab, and simulation benchmarks show that DRAW2ACT achieves superior visual fidelity and consistency while yielding higher manipulation success rates compared to existing baselines.
>
---
#### [new 077] EcoScapes: LLM-Powered Advice for Crafting Sustainable Cities
- **分类: cs.CV**

- **简介: 该论文提出EcoScapes系统，属城市可持续发展辅助决策任务，旨在解决小城市因人力与数据整合能力不足导致的气候适应策略制定困难问题。工作包括构建多层系统：融合专用LLM、卫星影像分析与知识库，支持可持续城市规划。**

- **链接: [https://arxiv.org/pdf/2512.14373v1](https://arxiv.org/pdf/2512.14373v1)**

> **作者:** Martin Röhn; Nora Gourmelon; Vincent Christlein
>
> **摘要:** Climate adaptation is vital for the sustainability and sometimes the mere survival of our urban areas. However, small cities often struggle with limited personnel resources and integrating vast amounts of data from multiple sources for a comprehensive analysis. To overcome these challenges, this paper proposes a multi-layered system combining specialized LLMs, satellite imagery analysis and a knowledge base to aid in developing effective climate adaptation strategies. The corresponding code can be found at https://github.com/Photon-GitHub/EcoScapes.
>
---
#### [new 078] MFE-GAN: Efficient GAN-based Framework for Document Image Enhancement and Binarization with Multi-scale Feature Extraction
- **分类: cs.CV**

- **简介: 该论文面向文档图像增强与二值化任务，旨在解决现有GAN方法训练/推理耗时高的问题。提出MFE-GAN框架，融合多尺度特征提取、Haar小波变换与归一化，并设计新型生成器、判别器及损失函数，在保持SOTA性能的同时显著提速。**

- **链接: [https://arxiv.org/pdf/2512.14114v1](https://arxiv.org/pdf/2512.14114v1)**

> **作者:** Rui-Yang Ju; KokSheik Wong; Yanlin Jin; Jen-Shiun Chiang
>
> **备注:** Extended Journal Version of APSIPA ASC 2025
>
> **摘要:** Document image enhancement and binarization are commonly performed prior to document analysis and recognition tasks for improving the efficiency and accuracy of optical character recognition (OCR) systems. This is because directly recognizing text in degraded documents, particularly in color images, often results in unsatisfactory recognition performance. To address these issues, existing methods train independent generative adversarial networks (GANs) for different color channels to remove shadows and noise, which, in turn, facilitates efficient text information extraction. However, deploying multiple GANs results in long training and inference times. To reduce both training and inference times of document image enhancement and binarization models, we propose MFE-GAN, an efficient GAN-based framework with multi-scale feature extraction (MFE), which incorporates Haar wavelet transformation (HWT) and normalization to process document images before feeding them into GANs for training. In addition, we present novel generators, discriminators, and loss functions to improve the model's performance, and we conduct ablation studies to demonstrate their effectiveness. Experimental results on the Benchmark, Nabuco, and CMATERdb datasets demonstrate that the proposed MFE-GAN significantly reduces the total training and inference times while maintaining comparable performance with respect to state-of-the-art (SOTA) methods. The implementation of this work is available at https://ruiyangju.github.io/MFE-GAN.
>
---
#### [new 079] DISCODE: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向图像描述自动评估任务，旨在解决现有LVLM评估方法在域偏移下鲁棒性差的问题。提出无需微调的分布感知评分解码器DISCODE，通过测试时自适应高斯先验损失提升跨域一致性，并构建多领域MCEval基准验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.14420v1](https://arxiv.org/pdf/2512.14420v1)**

> **作者:** Nakamasa Inoue; Kanoko Goto; Masanari Oi; Martyna Gruszka; Mahiro Ukai; Takumi Hirose; Yusuke Sekikawa
>
> **备注:** Paper accepted to AAAI 2026
>
> **摘要:** Large vision-language models (LVLMs) have shown impressive performance across a broad range of multimodal tasks. However, robust image caption evaluation using LVLMs remains challenging, particularly under domain-shift scenarios. To address this issue, we introduce the Distribution-Aware Score Decoder (DISCODE), a novel finetuning-free method that generates robust evaluation scores better aligned with human judgments across diverse domains. The core idea behind DISCODE lies in its test-time adaptive evaluation approach, which introduces the Adaptive Test-Time (ATT) loss, leveraging a Gaussian prior distribution to improve robustness in evaluation score estimation. This loss is efficiently minimized at test time using an analytical solution that we derive. Furthermore, we introduce the Multi-domain Caption Evaluation (MCEval) benchmark, a new image captioning evaluation benchmark covering six distinct domains, designed to assess the robustness of evaluation metrics. In our experiments, we demonstrate that DISCODE achieves state-of-the-art performance as a reference-free evaluation metric across MCEval and four representative existing benchmarks.
>
---
#### [new 080] OmniDrive-R1: Reinforcement-driven Interleaved Multi-modal Chain-of-Thought for Trustworthy Vision-Language Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向自动驾驶中的视觉-语言推理任务，旨在解决VLM因文本式思维链导致的物体幻觉问题。提出OmniDrive-R1框架，通过强化学习驱动的交错多模态思维链（iMCoT）实现感知与推理端到端联合优化，并设计无标注、过程导向的Clip-GRPO奖励机制提升视觉接地可靠性。**

- **链接: [https://arxiv.org/pdf/2512.14044v1](https://arxiv.org/pdf/2512.14044v1)**

> **作者:** Zhenguo Zhang; Haohan Zhen; Yishen Wang; Le Xu; Tianchen Deng; Xuefeng Chen; Qu Chen; Bo Zhang; Wuxiong Huang
>
> **摘要:** The deployment of Vision-Language Models (VLMs) in safety-critical domains like autonomous driving (AD) is critically hindered by reliability failures, most notably object hallucination. This failure stems from their reliance on ungrounded, text-based Chain-of-Thought (CoT) reasoning.While existing multi-modal CoT approaches attempt mitigation, they suffer from two fundamental flaws: (1) decoupled perception and reasoning stages that prevent end-to-end joint optimization, and (2) reliance on expensive, dense localization labels.Thus we introduce OmniDrive-R1, an end-to-end VLM framework designed for autonomous driving, which unifies perception and reasoning through an interleaved Multi-modal Chain-of-Thought (iMCoT) mechanism. Our core innovation is an Reinforcement-driven visual grounding capability, enabling the model to autonomously direct its attention and "zoom in" on critical regions for fine-grained analysis. This capability is enabled by our pure two-stage reinforcement learning training pipeline and Clip-GRPO algorithm. Crucially, Clip-GRPO introduces an annotation-free, process-based grounding reward. This reward not only eliminates the need for dense labels but also circumvents the instability of external tool calls by enforcing real-time cross-modal consistency between the visual focus and the textual reasoning. Extensive experiments on DriveLMM-o1 demonstrate our model's significant improvements. Compared to the baseline Qwen2.5VL-7B, OmniDrive-R1 improves the overall reasoning score from 51.77% to 80.35%, and the final answer accuracy from 37.81% to 73.62%.
>
---
#### [new 081] VajraV1 -- The most accurate Real Time Object Detector of the YOLO family
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VajraV1，一种基于YOLO架构的实时目标检测模型。旨在提升检测精度同时保持高速推理。通过融合前代YOLO设计优势，VajraV1在COCO上实现SOTA精度（最高56.2% mAP），全面超越YOLOv10–v13各尺寸变体。**

- **链接: [https://arxiv.org/pdf/2512.13834v1](https://arxiv.org/pdf/2512.13834v1)**

> **作者:** Naman Balbir Singh Makkar
>
> **备注:** Technical Report. 20 Pages, 7 figures
>
> **摘要:** Recent years have seen significant advances in real-time object detection, with the release of YOLOv10, YOLO11, YOLOv12, and YOLOv13 between 2024 and 2025. This technical report presents the VajraV1 model architecture, which introduces architectural enhancements over existing YOLO-based detectors. VajraV1 combines effective design choices from prior YOLO models to achieve state-of-the-art accuracy among real-time object detectors while maintaining competitive inference speed. On the COCO validation set, VajraV1-Nano achieves 44.3% mAP, outperforming YOLOv12-N by 3.7% and YOLOv13-N by 2.7% at latency competitive with YOLOv12-N and YOLOv11-N. VajraV1-Small achieves 50.4% mAP, exceeding YOLOv12-S and YOLOv13-S by 2.4%. VajraV1-Medium achieves 52.7% mAP, outperforming YOLOv12-M by 0.2%. VajraV1-Large achieves 53.7% mAP, surpassing YOLOv13-L by 0.3%. VajraV1-Xlarge achieves 56.2% mAP, outperforming all existing real-time object detectors.
>
---
#### [new 082] LLM-driven Knowledge Enhancement for Multimodal Cancer Survival Prediction
- **分类: cs.CV**

- **简介: 该论文面向多模态癌症生存预测任务，解决病理图像与基因数据高维冗余、模态对齐难及监督信号弱的问题。提出LLM驱动的知识增强模型KEMM，融合专家报告与预后背景知识，并设计KECM注意力模块引导特征学习，在五个数据集上达SOTA。**

- **链接: [https://arxiv.org/pdf/2512.14594v1](https://arxiv.org/pdf/2512.14594v1)**

> **作者:** Chenyu Zhao; Yingxue Xu; Fengtao Zhou; Yihui Wang; Hao Chen
>
> **摘要:** Current multimodal survival prediction methods typically rely on pathology images (WSIs) and genomic data, both of which are high-dimensional and redundant, making it difficult to extract discriminative features from them and align different modalities. Moreover, using a simple survival follow-up label is insufficient to supervise such a complex task. To address these challenges, we propose KEMM, an LLM-driven Knowledge-Enhanced Multimodal Model for cancer survival prediction, which integrates expert reports and prognostic background knowledge. 1) Expert reports, provided by pathologists on a case-by-case basis and refined by large language model (LLM), offer succinct and clinically focused diagnostic statements. This information may typically suggest different survival outcomes. 2) Prognostic background knowledge (PBK), generated concisely by LLM, provides valuable prognostic background knowledge on different cancer types, which also enhances survival prediction. To leverage these knowledge, we introduce the knowledge-enhanced cross-modal (KECM) attention module. KECM can effectively guide the network to focus on discriminative and survival-relevant features from highly redundant modalities. Extensive experiments on five datasets demonstrate that KEMM achieves state-of-the-art performance. The code will be released upon acceptance.
>
---
#### [new 083] HyperVL: An Efficient and Dynamic Multimodal Large Language Model for Edge Devices
- **分类: cs.CV; cs.CL**

- **简介: 该论文属边缘设备多模态大模型部署任务，旨在解决ViT编码器高延迟、高内存问题。提出HyperVL模型：采用图像分块降峰存；设计视觉分辨率压缩器（VRC）自适应调分辨率；引入双一致性学习（DCL）统一多尺度ViT与共享LLM，实现动态分支切换。**

- **链接: [https://arxiv.org/pdf/2512.14052v1](https://arxiv.org/pdf/2512.14052v1)**

> **作者:** HyperAI Team; Yuchen Liu; Kaiyang Han; Zhiqiang Xia; Yuhang Dong; Chen Song; Kangyu Tang; Jiaming Xu; Xiushi Feng; WenXuan Yu; Li Peng; Mingyang Wang; Kai Wang; Changpeng Yang; Yang Li; Haoyu Lu; Hao Wang; Bingna Xu; Guangyao Liu; Long Huang; Kaibin Guo; Jinyang Wu; Dan Wu; Hongzhen Wang; Peng Zhou; Shuai Nie; Shande Wang; Runyu Shi; Ying Huang
>
> **备注:** Technical report of Xiaomi HyperAI Team
>
> **摘要:** Current multimodal large lanauge models possess strong perceptual and reasoning capabilities, however high computational and memory requirements make them difficult to deploy directly on on-device environments. While small-parameter models are progressively endowed with strong general capabilities, standard Vision Transformer (ViT) encoders remain a critical bottleneck, suffering from excessive latency and memory consumption when processing high-resolution inputs.To address these challenges, we introduce HyperVL, an efficient multimodal large language model tailored for on-device inference. HyperVL adopts an image-tiling strategy to cap peak memory usage and incorporates two novel techniques: (1) a Visual Resolution Compressor (VRC) that adaptively predicts optimal encoding resolutions to eliminate redundant computation, and (2) Dual Consistency Learning (DCL), which aligns multi-scale ViT encoders within a unified framework, enabling dynamic switching between visual branches under a shared LLM. Extensive experiments demonstrate that HyperVL achieves state-of-the-art performance among models of comparable size across multiple benchmarks. Furthermore, it significantly significantly reduces latency and power consumption on real mobile devices, demonstrating its practicality for on-device multimodal inference.
>
---
#### [new 084] Consistent Instance Field for Dynamic Scene Understanding
- **分类: cs.CV**

- **简介: 该论文面向动态场景理解任务，解决传统方法在时序一致性与实例身份绑定上的不足。提出“一致实例场”，用可变形3D高斯建模时空点的占用概率与条件实例分布，通过可微光栅化联合学习辐射与语义，并引入身份校准与语义重采样机制，提升4D实例一致性。**

- **链接: [https://arxiv.org/pdf/2512.14126v1](https://arxiv.org/pdf/2512.14126v1)**

> **作者:** Junyi Wu; Van Nguyen Nguyen; Benjamin Planche; Jiachen Tao; Changchang Sun; Zhongpai Gao; Zhenghao Zhao; Anwesa Choudhuri; Gengyu Zhang; Meng Zheng; Feiran Wang; Terrence Chen; Yan Yan; Ziyan Wu
>
> **摘要:** We introduce Consistent Instance Field, a continuous and probabilistic spatio-temporal representation for dynamic scene understanding. Unlike prior methods that rely on discrete tracking or view-dependent features, our approach disentangles visibility from persistent object identity by modeling each space-time point with an occupancy probability and a conditional instance distribution. To realize this, we introduce a novel instance-embedded representation based on deformable 3D Gaussians, which jointly encode radiance and semantic information and are learned directly from input RGB images and instance masks through differentiable rasterization. Furthermore, we introduce new mechanisms to calibrate per-Gaussian identities and resample Gaussians toward semantically active regions, ensuring consistent instance representations across space and time. Experiments on HyperNeRF and Neu3D datasets demonstrate that our method significantly outperforms state-of-the-art methods on novel-view panoptic segmentation and open-vocabulary 4D querying tasks.
>
---
#### [new 085] S2D: Sparse-To-Dense Keymask Distillation for Unsupervised Video Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文面向无监督视频实例分割任务，旨在解决合成视频数据无法建模真实运动的问题。提出S2D方法：基于真实视频，利用运动先验筛选高质量关键帧掩码（keymasks），再通过稀疏到稠密蒸馏与时序DropLoss实现隐式掩码传播，最终在真实视频上端到端训练。**

- **链接: [https://arxiv.org/pdf/2512.14440v1](https://arxiv.org/pdf/2512.14440v1)**

> **作者:** Leon Sick; Lukas Hoyer; Dominik Engel; Pedro Hermosilla; Timo Ropinski
>
> **备注:** Project Page with Code/Models/Demo: https://leonsick.github.io/s2d/
>
> **摘要:** In recent years, the state-of-the-art in unsupervised video instance segmentation has heavily relied on synthetic video data, generated from object-centric image datasets such as ImageNet. However, video synthesis by artificially shifting and scaling image instance masks fails to accurately model realistic motion in videos, such as perspective changes, movement by parts of one or multiple instances, or camera motion. To tackle this issue, we propose an unsupervised video instance segmentation model trained exclusively on real video data. We start from unsupervised instance segmentation masks on individual video frames. However, these single-frame segmentations exhibit temporal noise and their quality varies through the video. Therefore, we establish temporal coherence by identifying high-quality keymasks in the video by leveraging deep motion priors. The sparse keymask pseudo-annotations are then used to train a segmentation model for implicit mask propagation, for which we propose a Sparse-To-Dense Distillation approach aided by a Temporal DropLoss. After training the final model on the resulting dense labelset, our approach outperforms the current state-of-the-art across various benchmarks.
>
---
#### [new 086] From Unlearning to UNBRANDING: A Benchmark for Trademark-Safe Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文提出“unbranding”新任务，旨在细粒度去除文本生成图像中的商标及隐性品牌特征（如结构设计），同时保持语义一致。构建了专用基准数据集，并设计基于VLM的评估指标，以检测显性logo与抽象trade dress。实验表明，高保真模型更易复现品牌标识，凸显该问题的紧迫性与独特性。**

- **链接: [https://arxiv.org/pdf/2512.13953v1](https://arxiv.org/pdf/2512.13953v1)**

> **作者:** Dawid Malarz; Artur Kasymov; Filip Manjak; Maciej Zięba; Przemysław Spurek
>
> **摘要:** The rapid progress of text-to-image diffusion models raises significant concerns regarding the unauthorized reproduction of trademarked content. While prior work targets general concepts (e.g., styles, celebrities), it fails to address specific brand identifiers. Crucially, we note that brand recognition is multi-dimensional, extending beyond explicit logos to encompass distinctive structural features (e.g., a car's front grille). To tackle this, we introduce unbranding, a novel task for the fine-grained removal of both trademarks and subtle structural brand features, while preserving semantic coherence. To facilitate research, we construct a comprehensive benchmark dataset. Recognizing that existing brand detectors are limited to logos and fail to capture abstract trade dress (e.g., the shape of a Coca-Cola bottle), we introduce a novel evaluation metric based on Vision Language Models (VLMs). This VLM-based metric uses a question-answering framework to probe images for both explicit logos and implicit, holistic brand characteristics. Furthermore, we observe that as model fidelity increases, with newer systems (SDXL, FLUX) synthesizing brand identifiers more readily than older models (Stable Diffusion), the urgency of the unbranding challenge is starkly highlighted. Our results, validated by our VLM metric, confirm unbranding is a distinct, practically relevant problem requiring specialized techniques. Project Page: https://gmum.github.io/UNBRANDING/.
>
---
#### [new 087] FoodLogAthl-218: Constructing a Real-World Food Image Dataset Using Dietary Management Applications
- **分类: cs.CV; cs.MM**

- **简介: 该论文构建了真实世界食物图像数据集FoodLogAthl-218（218类、6925图），解决现有网络爬取数据与用户实拍餐图差异大导致模型泛化差的问题。提出三个任务：标准分类、时序增量微调、上下文感知多菜分类，并用大模型评估。**

- **链接: [https://arxiv.org/pdf/2512.14574v1](https://arxiv.org/pdf/2512.14574v1)**

> **作者:** Mitsuki Watanabe; Sosuke Amano; Kiyoharu Aizawa; Yoko Yamakata
>
> **摘要:** Food image classification models are crucial for dietary management applications because they reduce the burden of manual meal logging. However, most publicly available datasets for training such models rely on web-crawled images, which often differ from users' real-world meal photos. In this work, we present FoodLogAthl-218, a food image dataset constructed from real-world meal records collected through the dietary management application FoodLog Athl. The dataset contains 6,925 images across 218 food categories, with a total of 14,349 bounding boxes. Rich metadata, including meal date and time, anonymized user IDs, and meal-level context, accompany each image. Unlike conventional datasets-where a predefined class set guides web-based image collection-our data begins with user-submitted photos, and labels are applied afterward. This yields greater intra-class diversity, a natural frequency distribution of meal types, and casual, unfiltered images intended for personal use rather than public sharing. In addition to (1) a standard classification benchmark, we introduce two FoodLog-specific tasks: (2) an incremental fine-tuning protocol that follows the temporal stream of users' logs, and (3) a context-aware classification task where each image contains multiple dishes, and the model must classify each dish by leveraging the overall meal context. We evaluate these tasks using large multimodal models (LMMs). The dataset is publicly available at https://huggingface.co/datasets/FoodLog/FoodLogAthl-218.
>
---
#### [new 088] KLO-Net: A Dynamic K-NN Attention U-Net with CSP Encoder for Efficient Prostate Gland Segmentation from MRI
- **分类: cs.CV; cs.LG**

- **简介: 该论文面向医学图像分割任务，旨在解决前列腺MRI分割中计算开销大、内存占用高及解剖变异导致精度低的问题。提出KLO-Net：融合动态K-NN注意力机制（自适应邻域数）与CSP编码器的轻量U-Net，兼顾效率与精度，并在PROMISE12和PROSTATEx数据集上验证。**

- **链接: [https://arxiv.org/pdf/2512.13902v1](https://arxiv.org/pdf/2512.13902v1)**

> **作者:** Anning Tian; Byunghyun Ko; Kaichen Qu; Mengyuan Liu; Jeongkyu Lee
>
> **备注:** Preprint. Accepted to SPIE Medical Imaging 2026: Image Processing
>
> **摘要:** Real-time deployment of prostate MRI segmentation on clinical workstations is often bottlenecked by computational load and memory footprint. Deep learning-based prostate gland segmentation approaches remain challenging due to anatomical variability. To bridge this efficiency gap while still maintaining reliable segmentation accuracy, we propose KLO-Net, a dynamic K-Nearest Neighbor attention U-Net with Cross Stage Partial, i.e., CSP, encoder for efficient prostate gland segmentation from MRI scan. Unlike the regular K-NN attention mechanism, the proposed dynamic K-NN attention mechanism allows the model to adaptively determine the number of attention connections for each spatial location within a slice. In addition, CSP blocks address the computational load to reduce memory consumption. To evaluate the model's performance, comprehensive experiments and ablation studies are conducted on two public datasets, i.e., PROMISE12 and PROSTATEx, to validate the proposed architecture. The detailed comparative analysis demonstrates the model's advantage in computational efficiency and segmentation quality.
>
---
#### [new 089] SDAR-VL: Stable and Efficient Block-wise Diffusion for Vision-Language Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向视觉-语言理解（VLU）任务，旨在解决块状离散扩散模型训练成本高、收敛慢、不稳定的问题。提出SDAR-VL框架，含异步噪声调度、掩码比缩放和渐进Beta噪声课程三项技术，显著提升训练效率、稳定性与性能，在21个基准上超越现有扩散及强自回归模型。**

- **链接: [https://arxiv.org/pdf/2512.14068v1](https://arxiv.org/pdf/2512.14068v1)**

> **作者:** Shuang Cheng; Yuhua Jiang; Zineng Zhou; Dawei Liu; Wang Tao; Linfeng Zhang; Biqing Qi; Bowen Zhou
>
> **摘要:** Block-wise discrete diffusion offers an attractive balance between parallel generation and causal dependency modeling, making it a promising backbone for vision-language modeling. However, its practical adoption has been limited by high training cost, slow convergence, and instability, which have so far kept it behind strong autoregressive (AR) baselines. We present \textbf{SDAR-VL}, the first systematic application of block-wise discrete diffusion to large-scale vision-language understanding (VLU), together with an \emph{integrated framework for efficient and stable training}. This framework unifies three components: (1) \textbf{Asynchronous Block-wise Noise Scheduling} to diversify supervision within each batch; (2) \textbf{Effective Mask Ratio Scaling} for unbiased loss normalization under stochastic masking; and (3) a \textbf{Progressive Beta Noise Curriculum} that increases effective mask coverage while preserving corruption diversity. Experiments on 21 single-image, multi-image, and video benchmarks show that SDAR-VL consistently improves \emph{training efficiency}, \emph{convergence stability}, and \emph{task performance} over conventional block diffusion. On this evaluation suite, SDAR-VL sets a new state of the art among diffusion-based vision-language models and, under matched settings, matches or surpasses strong AR baselines such as LLaVA-OneVision as well as the global diffusion baseline LLaDA-V, establishing block-wise diffusion as a practical backbone for VLU.
>
---
#### [new 090] Multi-View MRI Approach for Classification of MGMT Methylation in Glioblastoma Patients
- **分类: cs.CV**

- **简介: 该论文属医学影像分析任务，旨在非侵入性预测胶质母细胞瘤（GBM）患者MGMT启动子甲基化状态。提出一种轻量级多视角MRI深度学习方法，融合三平面图像空间关系，避免复杂3D模型；创新肿瘤切片提取技术，并开源可复现流程。**

- **链接: [https://arxiv.org/pdf/2512.14232v1](https://arxiv.org/pdf/2512.14232v1)**

> **作者:** Rawan Alyahya; Asrar Alruwayqi; Atheer Alqarni; Asma Alkhaldi; Metab Alkubeyyer; Xin Gao; Mona Alshahrani
>
> **摘要:** The presence of MGMT promoter methylation significantly affects how well chemotherapy works for patients with Glioblastoma Multiforme (GBM). Currently, confirmation of MGMT promoter methylation relies on invasive brain tumor tissue biopsies. In this study, we explore radiogenomics techniques, a promising approach in precision medicine, to identify genetic markers from medical images. Using MRI scans and deep learning models, we propose a new multi-view approach that considers spatial relationships between MRI views to detect MGMT methylation status. Importantly, our method extracts information from all three views without using a complicated 3D deep learning model, avoiding issues associated with high parameter count, slow convergence, and substantial memory demands. We also introduce a new technique for tumor slice extraction and show its superiority over existing methods based on multiple evaluation metrics. By comparing our approach to state-of-the-art models, we demonstrate the efficacy of our method. Furthermore, we share a reproducible pipeline of published models, encouraging transparency and the development of robust diagnostic tools. Our study highlights the potential of non-invasive methods for identifying MGMT promoter methylation and contributes to advancing precision medicine in GBM treatment.
>
---
#### [new 091] Spherical Leech Quantization for Visual Tokenization and Generation
- **分类: cs.CV; cs.AI; cs.LG; eess.SP**

- **简介: 该论文属视觉表征学习任务，旨在解决非参数向量量化中重建质量与压缩效率的权衡问题。提出基于24维Leech晶格的球面量化方法（$Λ_{24}$-SQ），统一建模非参数量化，简化训练并提升图像重建、压缩及生成性能。**

- **链接: [https://arxiv.org/pdf/2512.14697v1](https://arxiv.org/pdf/2512.14697v1)**

> **作者:** Yue Zhao; Hanwen Jiang; Zhenlin Xu; Chutong Yang; Ehsan Adeli; Philipp Krähenbühl
>
> **备注:** Tech report; project page: https://zhaoyue-zephyrus.github.io/npq/
>
> **摘要:** Non-parametric quantization has received much attention due to its efficiency on parameters and scalability to a large codebook. In this paper, we present a unified formulation of different non-parametric quantization methods through the lens of lattice coding. The geometry of lattice codes explains the necessity of auxiliary loss terms when training auto-encoders with certain existing lookup-free quantization variants such as BSQ. As a step forward, we explore a few possible candidates, including random lattices, generalized Fibonacci lattices, and densest sphere packing lattices. Among all, we find the Leech lattice-based quantization method, which is dubbed as Spherical Leech Quantization ($Λ_{24}$-SQ), leads to both a simplified training recipe and an improved reconstruction-compression tradeoff thanks to its high symmetry and even distribution on the hypersphere. In image tokenization and compression tasks, this quantization approach achieves better reconstruction quality across all metrics than BSQ, the best prior art, while consuming slightly fewer bits. The improvement also extends to state-of-the-art auto-regressive image generation frameworks.
>
---
#### [new 092] Improvise, Adapt, Overcome -- Telescopic Adapters for Efficient Fine-tuning of Vision Language Models in Medical Imaging
- **分类: cs.CV; cs.AI**

- **简介: 该论文属医学图像视觉语言分割（VLSM）的高效微调任务，旨在解决传统微调计算开销大、现有PEFT方法适配能力不足的问题。提出“望远式适配器”，按层深度动态增大适配器容量，仅用61.3k参数即在多个医疗数据集上实现高性能分割。**

- **链接: [https://arxiv.org/pdf/2512.13855v1](https://arxiv.org/pdf/2512.13855v1)**

> **作者:** Ujjwal Mishra; Vinita Shukla; Praful Hambarde; Amit Shukla
>
> **备注:** Accepted at the IEEE/CVF winter conference on applications of computer vision (WACV 2026)
>
> **摘要:** Adapting Vision Language Segmentation Models (VLSMs) to medical imaging domains requires significant computational overhead when using conventional fine-tuning approaches. Existing Parameter-Efficient Fine-Tuning (PEFT) methods apply uniform adapter dimensions across all transformer layers, leading to suboptimal parameter allocation and reduced adaptation efficiency. We introduce Telescopic Adapters, a novel PEFT framework that employs depth-aware scaling to progressively increase adapter capacity from shallow to deep transformer layers. Our method integrates lightweight bottleneck modules within CLIPSeg's vision and text encoders, with adapter dimensions dynamically scaled based on layer depth and semantic relevance. Using only 613k trainable parameters--244x fewer than end-to-end fine-tuning, Telescopic Adapters achieve superior performance across five diverse medical datasets spanning polyp segmentation, skin lesion detection, and breast ultrasound imaging. Comprehensive ablation studies demonstrate that deeper layers require substantially more adaptation capacity than shallow layers, validating our telescopic scaling hypothesis. Our approach establishes a new paradigm for efficient medical VLSM fine-tuning, enabling deployment in resource-constrained clinical environments while maintaining competitive segmentation accuracy.
>
---
#### [new 093] FocalComm: Hard Instance-Aware Multi-Agent Perception
- **分类: cs.CV**

- **简介: 该论文属多智能体协同感知任务，旨在解决现有方法对小目标（如行人）检测性能差、全特征交换冗余的问题。提出FocalComm框架，含可学习硬实例挖掘模块与查询式特征融合机制，聚焦交换关键特征以提升行人等安全敏感目标的检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.13982v1](https://arxiv.org/pdf/2512.13982v1)**

> **作者:** Dereje Shenkut; Vijayakumar Bhagavatula
>
> **备注:** WACV 2026
>
> **摘要:** Multi-agent collaborative perception (CP) is a promising paradigm for improving autonomous driving safety, particularly for vulnerable road users like pedestrians, via robust 3D perception. However, existing CP approaches often optimize for vehicle detection performance metrics, underperforming on smaller, safety-critical objects such as pedestrians, where detection failures can be catastrophic. Furthermore, previous CP methods rely on full feature exchange rather than communicating only salient features that help reduce false negatives. To this end, we present FocalComm, a novel collaborative perception framework that focuses on exchanging hard-instance-oriented features among connected collaborative agents. FocalComm consists of two key novel designs: (1) a learnable progressive hard instance mining (HIM) module to extract hard instance-oriented features per agent, and (2) a query-based feature-level (intermediate) fusion technique that dynamically weights these identified features during collaboration. We show that FocalComm outperforms state-of-the-art collaborative perception methods on two challenging real-world datasets (V2X-Real and DAIR-V2X) across both vehicle-centric and infrastructure-centric collaborative setups. FocalComm also shows a strong performance gain in pedestrian detection in V2X-Real.
>
---
#### [new 094] ViRC: Enhancing Visual Interleaved Mathematical CoT with Reason Chunking
- **分类: cs.CV**

- **简介: 该论文面向多模态数学推理任务，解决现有MLLMs忽视动态视觉获取、难以结构化分步推理的问题。提出ViRC框架，引入Reason Chunking机制，将推理分解为关键逻辑单元（CRUs），并构建CRUX数据集与渐进训练策略，显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2512.14654v1](https://arxiv.org/pdf/2512.14654v1)**

> **作者:** Lihong Wang; Liangqi Li; Weiwei Feng; Jiamin Wu; Changtao Miao; Tieru Wu; Rui Ma; Bo Zhang; Zhe Li
>
> **备注:** Code is available at https://github.com/Leon-LihongWang/ViRC
>
> **摘要:** CoT has significantly enhanced the reasoning ability of LLMs while it faces challenges when extended to multimodal domains, particularly in mathematical tasks. Existing MLLMs typically perform textual reasoning solely from a single static mathematical image, overlooking dynamic visual acquisition during reasoning. In contrast, humans repeatedly examine visual image and employ step-by-step reasoning to prove intermediate propositions. This strategy of decomposing the problem-solving process into key logical nodes adheres to Miller's Law in cognitive science. Inspired by this insight, we propose a ViRC framework for multimodal mathematical tasks, introducing a Reason Chunking mechanism that structures multimodal mathematical CoT into consecutive Critical Reasoning Units (CRUs) to simulate human expert problem-solving patterns. CRUs ensure intra-unit textual coherence for intermediate proposition verification while integrating visual information across units to generate subsequent propositions and support structured reasoning. To this end, we present CRUX dataset by using three visual tools and four reasoning patterns to provide explicitly annotated CRUs across multiple reasoning paths for each mathematical problem. Leveraging the CRUX dataset, we propose a progressive training strategy inspired by human cognitive learning, which includes Instructional SFT, Practice SFT, and Strategic RL, aimed at further strengthening the Reason Chunking ability of the model.The resulting ViRC-7B model achieves a 18.8\% average improvement over baselines across multiple mathematical benchmarks. Code is available at https://github.com/Leon-LihongWang/ViRC.
>
---
#### [new 095] HiFi-Portrait: Zero-shot Identity-preserved Portrait Generation with High-fidelity Multi-face Fusion
- **分类: cs.CV**

- **简介: 该论文属身份保持肖像生成任务，旨在解决多参考图下生成高保真、可控肖像的难题。提出HiFi-Portrait：引入人脸精修器与3D感知关键点生成器，设计HiFi-Net融合多脸特征并对其对齐，构建ID标注数据集，实现零样本、高保真、强可控的肖像生成。**

- **链接: [https://arxiv.org/pdf/2512.14542v1](https://arxiv.org/pdf/2512.14542v1)**

> **作者:** Yifang Xu; Benxiang Zhai; Yunzhuo Sun; Ming Li; Yang Li; Sidan Du
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** Recent advancements in diffusion-based technologies have made significant strides, particularly in identity-preserved portrait generation (IPG). However, when using multiple reference images from the same ID, existing methods typically produce lower-fidelity portraits and struggle to customize face attributes precisely. To address these issues, this paper presents HiFi-Portrait, a high-fidelity method for zero-shot portrait generation. Specifically, we first introduce the face refiner and landmark generator to obtain fine-grained multi-face features and 3D-aware face landmarks. The landmarks include the reference ID and the target attributes. Then, we design HiFi-Net to fuse multi-face features and align them with landmarks, which improves ID fidelity and face control. In addition, we devise an automated pipeline to construct an ID-based dataset for training HiFi-Portrait. Extensive experimental results demonstrate that our method surpasses the SOTA approaches in face similarity and controllability. Furthermore, our method is also compatible with previous SDXL-based works.
>
---
#### [new 096] DriverGaze360: OmniDirectional Driver Attention with Object-Level Guidance
- **分类: cs.CV**

- **简介: 该论文属驾驶员注意力预测任务，旨在解决现有方法视野窄、场景多样性不足导致难以建模全向注意的问题。作者构建了百万级360° gaze数据集DriverGaze360，并提出DriverGaze360-Net模型，通过联合学习注意力图与物体分割提升全景驾驶场景下的注意力预测精度。**

- **链接: [https://arxiv.org/pdf/2512.14266v1](https://arxiv.org/pdf/2512.14266v1)**

> **作者:** Shreedhar Govil; Didier Stricker; Jason Rambach
>
> **摘要:** Predicting driver attention is a critical problem for developing explainable autonomous driving systems and understanding driver behavior in mixed human-autonomous vehicle traffic scenarios. Although significant progress has been made through large-scale driver attention datasets and deep learning architectures, existing works are constrained by narrow frontal field-of-view and limited driving diversity. Consequently, they fail to capture the full spatial context of driving environments, especially during lane changes, turns, and interactions involving peripheral objects such as pedestrians or cyclists. In this paper, we introduce DriverGaze360, a large-scale 360$^\circ$ field of view driver attention dataset, containing $\sim$1 million gaze-labeled frames collected from 19 human drivers, enabling comprehensive omnidirectional modeling of driver gaze behavior. Moreover, our panoramic attention prediction approach, DriverGaze360-Net, jointly learns attention maps and attended objects by employing an auxiliary semantic segmentation head. This improves spatial awareness and attention prediction across wide panoramic inputs. Extensive experiments demonstrate that DriverGaze360-Net achieves state-of-the-art attention prediction performance on multiple metrics on panoramic driving images. Dataset and method available at https://av.dfki.de/drivergaze360.
>
---
#### [new 097] Towards Transferable Defense Against Malicious Image Edits
- **分类: cs.CV; cs.AI; cs.CY; cs.LG**

- **简介: 该论文面向扩散模型图像编辑中的恶意篡改防御任务，解决现有方法跨模型泛化能力弱的问题。提出TDAE框架，含视觉层FlatGrad防御（梯度正则化）和文本层Dynamic Prompt防御（动态优化提示嵌入），实现高转移性鲁棒防御。**

- **链接: [https://arxiv.org/pdf/2512.14341v1](https://arxiv.org/pdf/2512.14341v1)**

> **作者:** Jie Zhang; Shuai Dong; Shiguang Shan; Xilin Chen
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** Recent approaches employing imperceptible perturbations in input images have demonstrated promising potential to counter malicious manipulations in diffusion-based image editing systems. However, existing methods suffer from limited transferability in cross-model evaluations. To address this, we propose Transferable Defense Against Malicious Image Edits (TDAE), a novel bimodal framework that enhances image immunity against malicious edits through coordinated image-text optimization. Specifically, at the visual defense level, we introduce FlatGrad Defense Mechanism (FDM), which incorporates gradient regularization into the adversarial objective. By explicitly steering the perturbations toward flat minima, FDM amplifies immune robustness against unseen editing models. For textual enhancement protection, we propose an adversarial optimization paradigm named Dynamic Prompt Defense (DPD), which periodically refines text embeddings to align the editing outcomes of immunized images with those of the original images, then updates the images under optimized embeddings. Through iterative adversarial updates to diverse embeddings, DPD enforces the generation of immunized images that seek a broader set of immunity-enhancing features, thereby achieving cross-model transferability. Extensive experimental results demonstrate that our TDAE achieves state-of-the-art performance in mitigating malicious edits under both intra- and cross-model evaluations.
>
---
#### [new 098] DL$^3$M: A Vision-to-Language Framework for Expert-Level Medical Reasoning through Deep Learning and Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DL³M框架，解决医疗图像分类模型缺乏可解释性、LLM视觉推理弱且解释不稳定的问题。工作包括：设计MobileCoAtNet高精度胃镜分类模型；用其输出驱动多LLM生成临床推理；构建专家验证的双基准评估32个LLM，揭示当前LLM在医疗决策中的可靠性局限。**

- **链接: [https://arxiv.org/pdf/2512.13742v1](https://arxiv.org/pdf/2512.13742v1)**

> **作者:** Md. Najib Hasan; Imran Ahmad; Sourav Basak Shuvo; Md. Mahadi Hasan Ankon; Sunanda Das; Nazmul Siddique; Hui Wang
>
> **摘要:** Medical image classifiers detect gastrointestinal diseases well, but they do not explain their decisions. Large language models can generate clinical text, yet they struggle with visual reasoning and often produce unstable or incorrect explanations. This leaves a gap between what a model sees and the type of reasoning a clinician expects. We introduce a framework that links image classification with structured clinical reasoning. A new hybrid model, MobileCoAtNet, is designed for endoscopic images and achieves high accuracy across eight stomach-related classes. Its outputs are then used to drive reasoning by several LLMs. To judge this reasoning, we build two expert-verified benchmarks covering causes, symptoms, treatment, lifestyle, and follow-up care. Thirty-two LLMs are evaluated against these gold standards. Strong classification improves the quality of their explanations, but none of the models reach human-level stability. Even the best LLMs change their reasoning when prompts vary. Our study shows that combining DL with LLMs can produce useful clinical narratives, but current LLMs remain unreliable for high-stakes medical decisions. The framework provides a clearer view of their limits and a path for building safer reasoning systems. The complete source code and datasets used in this study are available at https://github.com/souravbasakshuvo/DL3M.
>
---
#### [new 099] Mimicking Human Visual Development for Learning Robust Image Representations
- **分类: cs.CV**

- **简介: 该论文属计算机视觉模型训练任务，旨在提升CNN对分布偏移和噪声的鲁棒性。受人类婴儿视觉发育启发，提出渐进式图像模糊课程：初期用强模糊图像训练，逐步减小模糊程度，使模型先学全局结构、再学细节，显著提升泛化与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.14360v1](https://arxiv.org/pdf/2512.14360v1)**

> **作者:** Ankita Raj; Kaashika Prajaapat; Tapan Kumar Gandhi; Chetan Arora
>
> **备注:** Accepted to ICVGIP 2025
>
> **摘要:** The human visual system is remarkably adept at adapting to changes in the input distribution; a capability modern convolutional neural networks (CNNs) still struggle to match. Drawing inspiration from the developmental trajectory of human vision, we propose a progressive blurring curriculum to improve the generalization and robustness of CNNs. Human infants are born with poor visual acuity, gradually refining their ability to perceive fine details. Mimicking this process, we begin training CNNs on highly blurred images during the initial epochs and progressively reduce the blur as training advances. This approach encourages the network to prioritize global structures over high-frequency artifacts, improving robustness against distribution shifts and noisy inputs. Challenging prior claims that blurring in the initial training epochs imposes a stimulus deficit and irreversibly harms model performance, we reveal that early-stage blurring enhances generalization with minimal impact on in-domain accuracy. Our experiments demonstrate that the proposed curriculum reduces mean corruption error (mCE) by up to 8.30% on CIFAR-10-C and 4.43% on ImageNet-100-C datasets, compared to standard training without blurring. Unlike static blur-based augmentation, which applies blurred images randomly throughout training, our method follows a structured progression, yielding consistent gains across various datasets. Furthermore, our approach complements other augmentation techniques, such as CutMix and MixUp, and enhances both natural and adversarial robustness against common attack methods. Code is available at https://github.com/rajankita/Visual_Acuity_Curriculum.
>
---
#### [new 100] SportsGPT: An LLM-driven Framework for Interpretable Sports Motion Assessment and Training Guidance
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SportsGPT框架，解决智能体育分析中缺乏自动诊断与可解释训练指导的问题。工作包括：1）MotionDTW精准提取关键帧；2）KISMAM生成可解释评估指标；3）SportsRAG基于知识库生成专业训练建议。**

- **链接: [https://arxiv.org/pdf/2512.14121v1](https://arxiv.org/pdf/2512.14121v1)**

> **作者:** Wenbo Tian; Ruting Lin; Hongxian Zheng; Yaodong Yang; Geng Wu; Zihao Zhang; Zhang Zhang
>
> **摘要:** Existing intelligent sports analysis systems mainly focus on "scoring and visualization," often lacking automatic performance diagnosis and interpretable training guidance. Recent advances of Large Language Models (LMMs) and motion analysis techniques provide new opportunities to address the above limitations. In this paper, we propose SportsGPT, an LLM-driven framework for interpretable sports motion assessment and training guidance, which establishes a closed loop from motion time-series input to professional training guidance. First, given a set of high-quality target models, we introduce MotionDTW, a two-stage time series alignment algorithm designed for accurate keyframe extraction from skeleton-based motion sequences. Subsequently, we design a Knowledge-based Interpretable Sports Motion Assessment Model (KISMAM) to obtain a set of interpretable assessment metrics (e.g., insufficient extension) by constrasting the keyframes with the targe models. Finally, we propose SportsRAG, a RAG-based training guidance model based on Qwen3. Leveraging a 6B-token knowledge base, it prompts the LLM to generate professional training guidance by retrieving domain-specific QA pairs. Experimental results demonstrate that MotionDTW significantly outperforms traditional methods with lower temporal error and higher IoU scores. Furthermore, ablation studies validate the KISMAM and SportsRAG, confirming that SportsGPT surpasses general LLMs in diagnostic accuracy and professionalism.
>
---
#### [new 101] ASAP-Textured Gaussians: Enhancing Textured Gaussians with Adaptive Sampling and Anisotropic Parameterization
- **分类: cs.CV**

- **简介: 该论文属3D高斯泼溅纹理建模任务，旨在解决纹理参数内存效率低、采样不均与过参数化问题。提出ASAP方法：基于高斯密度自适应采样，按渲染误差驱动各向异性纹理参数分配，显著提升质量-效率权衡。**

- **链接: [https://arxiv.org/pdf/2512.14039v1](https://arxiv.org/pdf/2512.14039v1)**

> **作者:** Meng Wei; Cheng Zhang; Jianmin Zheng; Hamid Rezatofighi; Jianfei Cai
>
> **摘要:** Recent advances have equipped 3D Gaussian Splatting with texture parameterizations to capture spatially varying attributes, improving the performance of both appearance modeling and downstream tasks. However, the added texture parameters introduce significant memory efficiency challenges. Rather than proposing new texture formulations, we take a step back to examine the characteristics of existing textured Gaussian methods and identify two key limitations in common: (1) Textures are typically defined in canonical space, leading to inefficient sampling that wastes textures' capacity on low-contribution regions; and (2) texture parameterization is uniformly assigned across all Gaussians, regardless of their visual complexity, resulting in over-parameterization. In this work, we address these issues through two simple yet effective strategies: adaptive sampling based on the Gaussian density distribution and error-driven anisotropic parameterization that allocates texture resources according to rendering error. Our proposed ASAP Textured Gaussians, short for Adaptive Sampling and Anisotropic Parameterization, significantly improve the quality efficiency tradeoff, achieving high-fidelity rendering with far fewer texture parameters.
>
---
#### [new 102] Why Text Prevails: Vision May Undermine Multimodal Medical Decision Making
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究医疗决策任务，发现当前多模态大模型在AD分期和胸片分类中，文本输入效果反超图文联合输入。作者通过实证分析揭示模型缺乏视觉接地理解，并提出三种改进策略：上下文学习、视觉描述+文本推理、视觉塔微调。**

- **链接: [https://arxiv.org/pdf/2512.13747v1](https://arxiv.org/pdf/2512.13747v1)**

> **作者:** Siyuan Dai; Lunxiao Li; Kun Zhao; Eardi Lila; Paul K. Crane; Heng Huang; Dongkuan Xu; Haoteng Tang; Liang Zhan
>
> **备注:** Accepted by ICDM 2025 the Workshop on Synergy of AI and Multimodal Biomedical Data Mining
>
> **摘要:** With the rapid progress of large language models (LLMs), advanced multimodal large language models (MLLMs) have demonstrated impressive zero-shot capabilities on vision-language tasks. In the biomedical domain, however, even state-of-the-art MLLMs struggle with basic Medical Decision Making (MDM) tasks. We investigate this limitation using two challenging datasets: (1) three-stage Alzheimer's disease (AD) classification (normal, mild cognitive impairment, dementia), where category differences are visually subtle, and (2) MIMIC-CXR chest radiograph classification with 14 non-mutually exclusive conditions. Our empirical study shows that text-only reasoning consistently outperforms vision-only or vision-text settings, with multimodal inputs often performing worse than text alone. To mitigate this, we explore three strategies: (1) in-context learning with reason-annotated exemplars, (2) vision captioning followed by text-only inference, and (3) few-shot fine-tuning of the vision tower with classification supervision. These findings reveal that current MLLMs lack grounded visual understanding and point to promising directions for improving multimodal decision making in healthcare.
>
---
#### [new 103] Broadening View Synthesis of Dynamic Scenes from Constrained Monocular Videos
- **分类: cs.CV**

- **简介: 该论文属动态场景新视角合成任务，旨在解决单目视频下大角度视角变换时渲染失真问题。提出ExpanDyNeRF框架，融合高斯溅射先验与伪真值生成策略，并构建首个带侧视监督的合成动态多视角数据集SynDM，显著提升极端视角下的渲染质量。**

- **链接: [https://arxiv.org/pdf/2512.14406v1](https://arxiv.org/pdf/2512.14406v1)**

> **作者:** Le Jiang; Shaotong Zhu; Yedi Luo; Shayda Moezzi; Sarah Ostadabbas
>
> **摘要:** In dynamic Neural Radiance Fields (NeRF) systems, state-of-the-art novel view synthesis methods often fail under significant viewpoint deviations, producing unstable and unrealistic renderings. To address this, we introduce Expanded Dynamic NeRF (ExpanDyNeRF), a monocular NeRF framework that leverages Gaussian splatting priors and a pseudo-ground-truth generation strategy to enable realistic synthesis under large-angle rotations. ExpanDyNeRF optimizes density and color features to improve scene reconstruction from challenging perspectives. We also present the Synthetic Dynamic Multiview (SynDM) dataset, the first synthetic multiview dataset for dynamic scenes with explicit side-view supervision-created using a custom GTA V-based rendering pipeline. Quantitative and qualitative results on SynDM and real-world datasets demonstrate that ExpanDyNeRF significantly outperforms existing dynamic NeRF methods in rendering fidelity under extreme viewpoint shifts. Further details are provided in the supplementary materials.
>
---
#### [new 104] ViewMask-1-to-3: Multi-View Consistent Image Generation via Multimodal Diffusion Models
- **分类: cs.CV**

- **简介: 该论文解决单图+文本驱动的多视角一致图像生成任务，旨在克服跨视角几何不一致难题。提出ViewMask-1-to-3：基于离散扩散模型，将多视图合成建模为视觉令牌序列的掩码预测问题，通过文本引导迭代解码，在无需3D先验或复杂结构下实现高效一致生成。**

- **链接: [https://arxiv.org/pdf/2512.14099v1](https://arxiv.org/pdf/2512.14099v1)**

> **作者:** Ruishu Zhu; Zhihao Huang; Jiacheng Sun; Ping Luo; Hongyuan Zhang; Xuelong Li
>
> **摘要:** Multi-view image generation from a single image and text description remains challenging due to the difficulty of maintaining geometric consistency across different viewpoints. Existing approaches typically rely on 3D-aware architectures or specialized diffusion models that require extensive multi-view training data and complex geometric priors. In this work, we introduce ViewMask-1-to-3, a pioneering approach to apply discrete diffusion models to multi-view image generation. Unlike continuous diffusion methods that operate in latent spaces, ViewMask-1-to-3 formulates multi-view synthesis as a discrete sequence modeling problem, where each viewpoint is represented as visual tokens obtained through MAGVIT-v2 tokenization. By unifying language and vision through masked token prediction, our approach enables progressive generation of multiple viewpoints through iterative token unmasking with text input. ViewMask-1-to-3 achieves cross-view consistency through simple random masking combined with self-attention, eliminating the requirement for complex 3D geometric constraints or specialized attention architectures. Our approach demonstrates that discrete diffusion provides a viable and simple alternative to existing multi-view generation methods, ranking first on average across GSO and 3D-FUTURE datasets in terms of PSNR, SSIM, and LPIPS, while maintaining architectural simplicity.
>
---
#### [new 105] Deep Learning Perspective of Scene Understanding in Autonomous Robots
- **分类: cs.CV**

- **简介: 该论文属综述任务，旨在解决自主机器人场景理解中传统几何方法鲁棒性差、实时性弱等问题。工作是系统回顾深度学习在目标检测、分割、深度估计、3D重建和视觉SLAM等方向的应用，分析其优势与挑战，并指出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2512.14020v1](https://arxiv.org/pdf/2512.14020v1)**

> **作者:** Afia Maham; Dur E Nayab Tashfa
>
> **备注:** 11 pages. Review Paper on Deep Learning Perspective of Scene Understanding in Autonomous Robots
>
> **摘要:** This paper provides a review of deep learning applications in scene understanding in autonomous robots, including innovations in object detection, semantic and instance segmentation, depth estimation, 3D reconstruction, and visual SLAM. It emphasizes how these techniques address limitations of traditional geometric models, improve depth perception in real time despite occlusions and textureless surfaces, and enhance semantic reasoning to understand the environment better. When these perception modules are integrated into dynamic and unstructured environments, they become more effective in decisionmaking, navigation and interaction. Lastly, the review outlines the existing problems and research directions to advance learning-based scene understanding of autonomous robots.
>
---
#### [new 106] GaussianPlant: Structure-aligned Gaussian Splatting for 3D Reconstruction of Plants
- **分类: cs.CV**

- **简介: 该论文属3D植物重建任务，旨在解决传统3D高斯泼溅（3DGS）缺乏可解释结构表征、难以支持植物表型分析的问题。提出GaussianPlant：用结构原语（圆柱/圆盘）显式建模枝干与叶片几何，绑定外观原语（高斯）渲染外观，联合优化实现高保真外观与准确结构重建。**

- **链接: [https://arxiv.org/pdf/2512.14087v1](https://arxiv.org/pdf/2512.14087v1)**

> **作者:** Yang Yang; Risa Shinoda; Hiroaki Santo; Fumio Okura
>
> **备注:** Submitted to IEEE TPAMI, under review
>
> **摘要:** We present a method for jointly recovering the appearance and internal structure of botanical plants from multi-view images based on 3D Gaussian Splatting (3DGS). While 3DGS exhibits robust reconstruction of scene appearance for novel-view synthesis, it lacks structural representations underlying those appearances (e.g., branching patterns of plants), which limits its applicability to tasks such as plant phenotyping. To achieve both high-fidelity appearance and structural reconstruction, we introduce GaussianPlant, a hierarchical 3DGS representation, which disentangles structure and appearance. Specifically, we employ structure primitives (StPs) to explicitly represent branch and leaf geometry, and appearance primitives (ApPs) to the plants' appearance using 3D Gaussians. StPs represent a simplified structure of the plant, i.e., modeling branches as cylinders and leaves as disks. To accurately distinguish the branches and leaves, StP's attributes (i.e., branches or leaves) are optimized in a self-organized manner. ApPs are bound to each StP to represent the appearance of branches or leaves as in conventional 3DGS. StPs and ApPs are jointly optimized using a re-rendering loss on the input multi-view images, as well as the gradient flow from ApP to StP using the binding correspondence information. We conduct experiments to qualitatively evaluate the reconstruction accuracy of both appearance and structure, as well as real-world experiments to qualitatively validate the practical performance. Experiments show that the GaussianPlant achieves both high-fidelity appearance reconstruction via ApPs and accurate structural reconstruction via StPs, enabling the extraction of branch structure and leaf instances.
>
---
#### [new 107] CIS-BA: Continuous Interaction Space Based Backdoor Attack for Object Detection in the Real-World
- **分类: cs.CV; cs.CR**

- **简介: 该论文面向目标检测模型的现实世界后门攻击任务，解决现有方法依赖单触发-单目标、鲁棒性差的问题。提出CIS-BA：基于连续交互空间的新范式，通过建模对象间几何交互关系设计空间触发器，实现多触发-多目标、鲁棒且可防御规避的攻击。**

- **链接: [https://arxiv.org/pdf/2512.14158v1](https://arxiv.org/pdf/2512.14158v1)**

> **作者:** Shuxin Zhao; Bo Lang; Nan Xiao; Yilang Zhang
>
> **摘要:** Object detection models deployed in real-world applications such as autonomous driving face serious threats from backdoor attacks. Despite their practical effectiveness,existing methods are inherently limited in both capability and robustness due to their dependence on single-trigger-single-object mappings and fragile pixel-level cues. We propose CIS-BA, a novel backdoor attack paradigm that redefines trigger design by shifting from static object features to continuous inter-object interaction patterns that describe how objects co-occur and interact in a scene. By modeling these patterns as a continuous interaction space, CIS-BA introduces space triggers that, for the first time, enable a multi-trigger-multi-object attack mechanism while achieving robustness through invariant geometric relations. To implement this paradigm, we design CIS-Frame, which constructs space triggers via interaction analysis, formalizes them as class-geometry constraints for sample poisoning, and embeds the backdoor during detector training. CIS-Frame supports both single-object attacks (object misclassification and disappearance) and multi-object simultaneous attacks, enabling complex and coordinated effects across diverse interaction states. Experiments on MS-COCO and real-world videos show that CIS-BA achieves over 97% attack success under complex environments and maintains over 95% effectiveness under dynamic multi-trigger conditions, while evading three state-of-the-art defenses. In summary, CIS-BA extends the landscape of backdoor attacks in interaction-intensive scenarios and provides new insights into the security of object detection systems.
>
---
#### [new 108] FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于深度伪造视频检测任务，旨在解决现有方法在跨域场景下对未知伪造技术泛化能力差的问题。提出FakeRadar框架，通过伪造异常探针（动态子簇建模与边界异常生成）和异常引导三阶段训练，提升对新兴伪造类型的检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.14601v1](https://arxiv.org/pdf/2512.14601v1)**

> **作者:** Zhaolun Li; Jichang Li; Yinqi Cai; Junye Chen; Xiaonan Luo; Guanbin Li; Rushi Lan
>
> **摘要:** In this paper, we propose FakeRadar, a novel deepfake video detection framework designed to address the challenges of cross-domain generalization in real-world scenarios. Existing detection methods typically rely on manipulation-specific cues, performing well on known forgery types but exhibiting severe limitations against emerging manipulation techniques. This poor generalization stems from their inability to adapt effectively to unseen forgery patterns. To overcome this, we leverage large-scale pretrained models (e.g. CLIP) to proactively probe the feature space, explicitly highlighting distributional gaps between real videos, known forgeries, and unseen manipulations. Specifically, FakeRadar introduces Forgery Outlier Probing, which employs dynamic subcluster modeling and cluster-conditional outlier generation to synthesize outlier samples near boundaries of estimated subclusters, simulating novel forgery artifacts beyond known manipulation types. Additionally, we design Outlier-Guided Tri-Training, which optimizes the detector to distinguish real, fake, and outlier samples using proposed outlier-driven contrastive learning and outlier-conditioned cross-entropy losses. Experiments show that FakeRadar outperforms existing methods across various benchmark datasets for deepfake video detection, particularly in cross-domain evaluations, by handling the variety of emerging manipulation techniques.
>
---
#### [new 109] OUSAC: Optimized Guidance Scheduling with Adaptive Caching for DiT Acceleration
- **分类: cs.CV**

- **简介: 该论文属扩散模型加速任务，旨在降低DiT中Classifier-Free Guidance（CFG）的计算开销。提出OUSAC框架：阶段一用进化算法优化动态跳过CFG步骤与调整指导尺度；阶段二设计自适应缓存机制，按块分配校准秩，兼顾效率与质量。**

- **链接: [https://arxiv.org/pdf/2512.14096v1](https://arxiv.org/pdf/2512.14096v1)**

> **作者:** Ruitong Sun; Tianze Yang; Wei Niu; Jin Sun
>
> **备注:** 29 pages
>
> **摘要:** Diffusion models have emerged as the dominant paradigm for high-quality image generation, yet their computational expense remains substantial due to iterative denoising. Classifier-Free Guidance (CFG) significantly enhances generation quality and controllability but doubles the computation by requiring both conditional and unconditional forward passes at every timestep. We present OUSAC (Optimized gUidance Scheduling with Adaptive Caching), a framework that accelerates diffusion transformers (DiT) through systematic optimization. Our key insight is that variable guidance scales enable sparse computation: adjusting scales at certain timesteps can compensate for skipping CFG at others, enabling both fewer total sampling steps and fewer CFG steps while maintaining quality. However, variable guidance patterns introduce denoising deviations that undermine standard caching methods, which assume constant CFG scales across steps. Moreover, different transformer blocks are affected at different levels under dynamic conditions. This paper develops a two-stage approach leveraging these insights. Stage-1 employs evolutionary algorithms to jointly optimize which timesteps to skip and what guidance scale to use, eliminating up to 82% of unconditional passes. Stage-2 introduces adaptive rank allocation that tailors calibration efforts per transformer block, maintaining caching effectiveness under variable guidance. Experiments demonstrate that OUSAC significantly outperforms state-of-the-art acceleration methods, achieving 53% computational savings with 15% quality improvement on DiT-XL/2 (ImageNet 512x512), 60% savings with 16.1% improvement on PixArt-alpha (MSCOCO), and 5x speedup on FLUX while improving CLIP Score over the 50-step baseline.
>
---
#### [new 110] Native and Compact Structured Latents for 3D Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属3D生成任务，旨在解决现有表示难以建模复杂拓扑与精细外观的问题。提出O-Voxel稀疏体素结构，统一编码几何与PBR材质；设计Sparse Compression VAE实现高倍压缩与紧凑隐空间；训练4B参数流匹配模型，显著提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2512.14692v1](https://arxiv.org/pdf/2512.14692v1)**

> **作者:** Jianfeng Xiang; Xiaoxue Chen; Sicheng Xu; Ruicheng Wang; Zelong Lv; Yu Deng; Hongyuan Zhu; Yue Dong; Hao Zhao; Nicholas Jing Yuan; Jiaolong Yang
>
> **备注:** Project Page: https://microsoft.github.io/TRELLIS.2/
>
> **摘要:** Recent advancements in 3D generative modeling have significantly improved the generation realism, yet the field is still hampered by existing representations, which struggle to capture assets with complex topologies and detailed appearance. This paper present an approach for learning a structured latent representation from native 3D data to address this challenge. At its core is a new sparse voxel structure called O-Voxel, an omni-voxel representation that encodes both geometry and appearance. O-Voxel can robustly model arbitrary topology, including open, non-manifold, and fully-enclosed surfaces, while capturing comprehensive surface attributes beyond texture color, such as physically-based rendering parameters. Based on O-Voxel, we design a Sparse Compression VAE which provides a high spatial compression rate and a compact latent space. We train large-scale flow-matching models comprising 4B parameters for 3D generation using diverse public 3D asset datasets. Despite their scale, inference remains highly efficient. Meanwhile, the geometry and material quality of our generated assets far exceed those of existing models. We believe our approach offers a significant advancement in 3D generative modeling.
>
---
#### [new 111] Neurosymbolic Inference On Foundation Models For Remote Sensing Text-to-image Retrieval With Complex Queries
- **分类: cs.CV; cs.AI; cs.IR**

- **简介: 该论文面向遥感图文检索任务，解决现有模型在复杂空间关系推理和可解释性上的不足。提出RUNE方法：用LLM将文本查询转为一阶逻辑表达式，结合神经符号推理模块显式推理检测实体；引入新评估指标，验证其在性能、鲁棒性与可解释性上的优势。**

- **链接: [https://arxiv.org/pdf/2512.14102v1](https://arxiv.org/pdf/2512.14102v1)**

> **作者:** Emanuele Mezzi; Gertjan Burghouts; Maarten Kruithof
>
> **摘要:** Text-to-image retrieval in remote sensing (RS) has advanced rapidly with the rise of large vision-language models (LVLMs) tailored for aerial and satellite imagery, culminating in remote sensing large vision-language models (RS-LVLMS). However, limited explainability and poor handling of complex spatial relations remain key challenges for real-world use. To address these issues, we introduce RUNE (Reasoning Using Neurosymbolic Entities), an approach that combines Large Language Models (LLMs) with neurosymbolic AI to retrieve images by reasoning over the compatibility between detected entities and First-Order Logic (FOL) expressions derived from text queries. Unlike RS-LVLMs that rely on implicit joint embeddings, RUNE performs explicit reasoning, enhancing performance and interpretability. For scalability, we propose a logic decomposition strategy that operates on conditioned subsets of detected entities, guaranteeing shorter execution time compared to neural approaches. Rather than using foundation models for end-to-end retrieval, we leverage them only to generate FOL expressions, delegating reasoning to a neurosymbolic inference module. For evaluation we repurpose the DOTA dataset, originally designed for object detection, by augmenting it with more complex queries than in existing benchmarks. We show the LLM's effectiveness in text-to-logic translation and compare RUNE with state-of-the-art RS-LVLMs, demonstrating superior performance. We introduce two metrics, Retrieval Robustness to Query Complexity (RRQC) and Retrieval Robustness to Image Uncertainty (RRIU), which evaluate performance relative to query complexity and image uncertainty. RUNE outperforms joint-embedding models in complex RS retrieval tasks, offering gains in performance, robustness, and explainability. We show RUNE's potential for real-world RS applications through a use case on post-flood satellite image retrieval.
>
---
#### [new 112] Sparse-LaViDa: Sparse Multimodal Discrete Diffusion Language Models
- **分类: cs.CV**

- **简介: 该论文属多模态生成任务，旨在解决掩码离散扩散模型（MDM）推理慢的问题。提出Sparse-LaViDa框架：动态剪枝冗余掩码token、引入寄存器token保留信息、设计匹配的注意力掩码以对齐训推。在LaViDa-O上实现2倍加速，保持生成质量。**

- **链接: [https://arxiv.org/pdf/2512.14008v1](https://arxiv.org/pdf/2512.14008v1)**

> **作者:** Shufan Li; Jiuxiang Gu; Kangning Liu; Zhe Lin; Zijun Wei; Aditya Grover; Jason Kuen
>
> **备注:** 18 pages (12 pages for the main paper and 6 pages for the appendix), 9 figures
>
> **摘要:** Masked Discrete Diffusion Models (MDMs) have achieved strong performance across a wide range of multimodal tasks, including image understanding, generation, and editing. However, their inference speed remains suboptimal due to the need to repeatedly process redundant masked tokens at every sampling step. In this work, we propose Sparse-LaViDa, a novel modeling framework that dynamically truncates unnecessary masked tokens at each inference step to accelerate MDM sampling. To preserve generation quality, we introduce specialized register tokens that serve as compact representations for the truncated tokens. Furthermore, to ensure consistency between training and inference, we design a specialized attention mask that faithfully matches the truncated sampling procedure during training. Built upon the state-of-the-art unified MDM LaViDa-O, Sparse-LaViDa achieves up to a 2x speedup across diverse tasks including text-to-image generation, image editing, and mathematical reasoning, while maintaining generation quality.
>
---
#### [new 113] VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VASA-3D，属音频驱动的单图生成3D头像任务。旨在解决单张人像重建高保真3D头像及自然表情驱动难题。工作包括：复用VASA-1运动隐表示，设计可条件驱动的3D头模型，并通过基于合成视频帧的优化框架实现单图定制。**

- **链接: [https://arxiv.org/pdf/2512.14677v1](https://arxiv.org/pdf/2512.14677v1)**

> **作者:** Sicheng Xu; Guojun Chen; Jiaolong Yang; Yizhong Zhang; Yu Deng; Steve Lin; Baining Guo
>
> **备注:** NeurIPS 2025 paper. Project webpage: https://www.microsoft.com/en-us/research/project/vasa-3d/
>
> **摘要:** We propose VASA-3D, an audio-driven, single-shot 3D head avatar generator. This research tackles two major challenges: capturing the subtle expression details present in real human faces, and reconstructing an intricate 3D head avatar from a single portrait image. To accurately model expression details, VASA-3D leverages the motion latent of VASA-1, a method that yields exceptional realism and vividness in 2D talking heads. A critical element of our work is translating this motion latent to 3D, which is accomplished by devising a 3D head model that is conditioned on the motion latent. Customization of this model to a single image is achieved through an optimization framework that employs numerous video frames of the reference head synthesized from the input image. The optimization takes various training losses robust to artifacts and limited pose coverage in the generated training data. Our experiment shows that VASA-3D produces realistic 3D talking heads that cannot be achieved by prior art, and it supports the online generation of 512x512 free-viewpoint videos at up to 75 FPS, facilitating more immersive engagements with lifelike 3D avatars.
>
---
#### [new 114] MMGR: Multi-Modal Generative Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出MMGR评估框架，针对视频/图像生成模型缺乏物理、逻辑等推理能力的问题，构建多模态生成式推理评测基准，涵盖五大推理能力与三大领域，揭示当前模型在抽象推理和长程空间规划上的严重缺陷。**

- **链接: [https://arxiv.org/pdf/2512.14691v1](https://arxiv.org/pdf/2512.14691v1)**

> **作者:** Zefan Cai; Haoyi Qiu; Tianyi Ma; Haozhe Zhao; Gengze Zhou; Kung-Hsiang Huang; Parisa Kordjamshidi; Minjia Zhang; Xiao Wen; Jiuxiang Gu; Nanyun Peng; Junjie Hu
>
> **备注:** work in progress
>
> **摘要:** Video foundation models generate visually realistic and temporally coherent content, but their reliability as world simulators depends on whether they capture physical, logical, and spatial constraints. Existing metrics such as Frechet Video Distance (FVD) emphasize perceptual quality and overlook reasoning failures, including violations of causality, physics, and global consistency. We introduce MMGR (Multi-Modal Generative Reasoning Evaluation and Benchmark), a principled evaluation framework based on five reasoning abilities: Physical, Logical, 3D Spatial, 2D Spatial, and Temporal. MMGR evaluates generative reasoning across three domains: Abstract Reasoning (ARC-AGI, Sudoku), Embodied Navigation (real-world 3D navigation and localization), and Physical Commonsense (sports and compositional interactions). MMGR applies fine-grained metrics that require holistic correctness across both video and image generation. We benchmark leading video models (Veo-3, Sora-2, Wan-2.2) and image models (Nano-banana, Nano-banana Pro, GPT-4o-image, Qwen-image), revealing strong performance gaps across domains. Models show moderate success on Physical Commonsense tasks but perform poorly on Abstract Reasoning (below 10 percent accuracy on ARC-AGI) and struggle with long-horizon spatial planning in embodied settings. Our analysis highlights key limitations in current models, including overreliance on perceptual data, weak global state consistency, and objectives that reward visual plausibility over causal correctness. MMGR offers a unified diagnostic benchmark and a path toward reasoning-aware generative world models.
>
---
#### [new 115] A Comprehensive Safety Metric to Evaluate Perception in Autonomous Systems
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶感知安全评估任务，旨在解决现有感知指标忽略目标重要性差异（如速度、距离、碰撞风险等）导致的安全评价不准确问题。作者提出一种综合安全度量指标，融合多维动态参数，输出单一可解释的安全评分，并在真实与虚拟数据上验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.14367v1](https://arxiv.org/pdf/2512.14367v1)**

> **作者:** Georg Volk; Jörg Gamerdinger; Alexander von Bernuth; Oliver Bringmann
>
> **备注:** Accepted at IEEE ITSC 2020
>
> **摘要:** Complete perception of the environment and its correct interpretation is crucial for autonomous vehicles. Object perception is the main component of automotive surround sensing. Various metrics already exist for the evaluation of object perception. However, objects can be of different importance depending on their velocity, orientation, distance, size, or the potential damage that could be caused by a collision due to a missed detection. Thus, these additional parameters have to be considered for safety evaluation. We propose a new safety metric that incorporates all these parameters and returns a single easily interpretable safety assessment score for object perception. This new metric is evaluated with both real world and virtual data sets and compared to state of the art metrics.
>
---
#### [new 116] Expert Switching for Robust AAV Landing: A Dual-Detector Framework in Simulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文面向AAV视觉着陆任务，解决单模型在大尺度变化（高空小目标→低空大目标）下检测鲁棒性差的问题。提出双专家YOLOv8框架，按远/近程尺度分工训练，并用几何门控动态选优，提升着陆精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.14054v1](https://arxiv.org/pdf/2512.14054v1)**

> **作者:** Humaira Tasnim; Ashik E Rasul; Bruce Jo; Hyung-Jin Yoon
>
> **摘要:** Reliable helipad detection is essential for Autonomous Aerial Vehicle (AAV) landing, especially under GPS-denied or visually degraded conditions. While modern detectors such as YOLOv8 offer strong baseline performance, single-model pipelines struggle to remain robust across the extreme scale transitions that occur during descent, where helipads appear small at high altitude and large near touchdown. To address this limitation, we propose a scale-adaptive dual-expert perception framework that decomposes the detection task into far-range and close-range regimes. Two YOLOv8 experts are trained on scale-specialized versions of the HelipadCat dataset, enabling one model to excel at detecting small, low-resolution helipads and the other to provide high-precision localization when the target dominates the field of view. During inference, both experts operate in parallel, and a geometric gating mechanism selects the expert whose prediction is most consistent with the AAV's viewpoint. This adaptive routing prevents the degradation commonly observed in single-detector systems when operating across wide altitude ranges. The dual-expert perception module is evaluated in a closed-loop landing environment that integrates CARLA's photorealistic rendering with NASA's GUAM flight-dynamics engine. Results show substantial improvements in alignment stability, landing accuracy, and overall robustness compared to single-detector baselines. By introducing a scale-aware expert routing strategy tailored to the landing problem, this work advances resilient vision-based perception for autonomous descent and provides a foundation for future multi-expert AAV frameworks.
>
---
#### [new 117] Physics-Guided Deep Learning for Heat Pump Stress Detection: A Comprehensive Analysis on When2Heat Dataset
- **分类: cs.LG; cs.CV; cs.NE**

- **简介: 该论文属工业智能诊断任务，旨在解决热泵系统运行压力检测难问题。提出物理引导深度神经网络（PG-DNN），融合热力学知识进行特征选择与类别定义，在When2Heat数据集上实现78.1%准确率，优于基线模型。**

- **链接: [https://arxiv.org/pdf/2512.13696v1](https://arxiv.org/pdf/2512.13696v1)**

> **作者:** Md Shahabub Alam; Md Asifuzzaman Jishan; Ayan Kumar Ghosh
>
> **摘要:** Heat pump systems are critical components in modern energy-efficient buildings, yet their operational stress detection remains challenging due to complex thermodynamic interactions and limited real-world data. This paper presents a novel Physics-Guided Deep Neural Network (PG-DNN) approach for heat pump stress classification using the When2Heat dataset, containing 131,483 samples with 656 features across 26 European countries. The methodology integrates physics-guided feature selection and class definition with a deep neural network architecture featuring 5 hidden layers and dual regularization strategies. The model achieves 78.1\% test accuracy and 78.5% validation accuracy, demonstrating significant improvements over baseline approaches: +5.0% over shallow networks, +4.0% over limited feature sets, and +2.0% over single regularization strategies. Comprehensive ablation studies validate the effectiveness of physics-guided feature selection, variable thresholding for realistic class distribution, and cross-country energy pattern analysis. The proposed system provides a production-ready solution for heat pump stress detection with 181,348 parameters and 720 seconds training time on AMD Ryzen 9 7950X with RTX 4080 hardware.
>
---
#### [new 118] Composite Classifier-Free Guidance for Multi-Modal Conditioning in Wind Dynamics Super-Resolution
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文面向风动力学超分辨率任务，旨在解决高精度、低成本获取高分辨率风数据的难题。提出复合无分类器引导（CCFG）方法，改进扩散模型对多模态条件输入的利用；构建WindDM模型，实现SOTA重建质量且成本降千倍。**

- **链接: [https://arxiv.org/pdf/2512.13729v1](https://arxiv.org/pdf/2512.13729v1)**

> **作者:** Jacob Schnell; Aditya Makkar; Gunadi Gani; Aniket Srinivasan Ashok; Darren Lo; Mike Optis; Alexander Wong; Yuhao Chen
>
> **摘要:** Various weather modelling problems (e.g., weather forecasting, optimizing turbine placements, etc.) require ample access to high-resolution, highly accurate wind data. Acquiring such high-resolution wind data, however, remains a challenging and expensive endeavour. Traditional reconstruction approaches are typically either cost-effective or accurate, but not both. Deep learning methods, including diffusion models, have been proposed to resolve this trade-off by leveraging advances in natural image super-resolution. Wind data, however, is distinct from natural images, and wind super-resolvers often use upwards of 10 input channels, significantly more than the usual 3-channel RGB inputs in natural images. To better leverage a large number of conditioning variables in diffusion models, we present a generalization of classifier-free guidance (CFG) to multiple conditioning inputs. Our novel composite classifier-free guidance (CCFG) can be dropped into any pre-trained diffusion model trained with standard CFG dropout. We demonstrate that CCFG outputs are higher-fidelity than those from CFG on wind super-resolution tasks. We present WindDM, a diffusion model trained for industrial-scale wind dynamics reconstruction and leveraging CCFG. WindDM achieves state-of-the-art reconstruction quality among deep learning models and costs up to $1000\times$ less than classical methods.
>
---
#### [new 119] Improving the Plausibility of Pressure Distributions Synthesized from Depth through Generative Modeling
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属医学图像生成任务，旨在解决深度图合成压力分布图物理不合理的临床可靠性问题。提出Informed Latent Space与Weight Optimization Loss，并设计BBDM及轻量LBBDM模型，提升生成压力图的物理合理性和实时性。**

- **链接: [https://arxiv.org/pdf/2512.13757v1](https://arxiv.org/pdf/2512.13757v1)**

> **作者:** Neevkumar Manavar; Hanno Gerd Meyer; Joachim Waßmuth; Barbara Hammer; Axel Schneider
>
> **摘要:** Monitoring contact pressure in hospital beds is essential for preventing pressure ulcers and enabling real-time patient assessment. Current methods can predict pressure maps but often lack physical plausibility, limiting clinical reliability. This work proposes a framework that enhances plausibility via Informed Latent Space (ILS) and Weight Optimization Loss (WOL) with generative modeling to produce high-fidelity, physically consistent pressure estimates. This study also applies diffusion based conditional Brownian Bridge Diffusion Model (BBDM) and proposes training strategy for its latent counterpart Latent Brownian Bridge Diffusion Model (LBBDM) tailored for pressure synthesis in lying postures. Experiment results shows proposed method improves physical plausibility and performance over baselines: BBDM with ILS delivers highly detailed maps at higher computational cost and large inference time, whereas LBBDM provides faster inference with competitive performance. Overall, the approach supports non-invasive, vision-based, real-time patient monitoring in clinical environments.
>
---
#### [new 120] Generative AI for Video Translation: A Scalable Architecture for Multilingual Video Conferencing
- **分类: cs.MM; cs.AI; cs.CL; cs.CV**

- **简介: 该论文面向视频翻译任务，解决多用户实时视频会议中因级联生成式AI导致的高延迟与O(N²)计算复杂度问题。提出含轮询机制和分段处理的系统架构，将复杂度降为线性，并在多级GPU上验证实时性与用户体验。**

- **链接: [https://arxiv.org/pdf/2512.13904v1](https://arxiv.org/pdf/2512.13904v1)**

> **作者:** Amirkia Rafiei Oskooei; Eren Caglar; Ibrahim Sahin; Ayse Kayabay; Mehmet S. Aktas
>
> **备注:** Accepted manuscript. Published in Applied Sciences, 2025
>
> **摘要:** The real-time deployment of cascaded generative AI pipelines for applications like video translation is constrained by significant system-level challenges. These include the cumulative latency of sequential model inference and the quadratic ($\mathcal{O}(N^2)$) computational complexity that renders multi-user video conferencing applications unscalable. This paper proposes and evaluates a practical system-level framework designed to mitigate these critical bottlenecks. The proposed architecture incorporates a turn-taking mechanism to reduce computational complexity from quadratic to linear in multi-user scenarios, and a segmented processing protocol to manage inference latency for a perceptually real-time experience. We implement a proof-of-concept pipeline and conduct a rigorous performance analysis across a multi-tiered hardware setup, including commodity (NVIDIA RTX 4060), cloud (NVIDIA T4), and enterprise (NVIDIA A100) GPUs. Our objective evaluation demonstrates that the system achieves real-time throughput ($τ< 1.0$) on modern hardware. A subjective user study further validates the approach, showing that a predictable, initial processing delay is highly acceptable to users in exchange for a smooth, uninterrupted playback experience. The work presents a validated, end-to-end system design that offers a practical roadmap for deploying scalable, real-time generative AI applications in multilingual communication platforms.
>
---
#### [new 121] VICTOR: Dataset Copyright Auditing in Video Recognition Systems
- **分类: cs.CR; cs.CV**

- **简介: 该论文提出VICTOR，首个面向视频识别系统的数据集版权审计方法。针对视频时序特性导致现有图像版权审计失效的问题，设计轻量、隐蔽的样本修改策略（仅改1%样本），通过放大模型对修改/原始样本的输出差异实现侵权检测，具备鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.14439v1](https://arxiv.org/pdf/2512.14439v1)**

> **作者:** Quan Yuan; Zhikun Zhang; Linkang Du; Min Chen; Mingyang Sun; Yunjun Gao; Shibo He; Jiming Chen
>
> **备注:** To appear in the NDSS Symposium 2026, February 2026, San Diego, CA, USA
>
> **摘要:** Video recognition systems are increasingly being deployed in daily life, such as content recommendation and security monitoring. To enhance video recognition development, many institutions have released high-quality public datasets with open-source licenses for training advanced models. At the same time, these datasets are also susceptible to misuse and infringement. Dataset copyright auditing is an effective solution to identify such unauthorized use. However, existing dataset copyright solutions primarily focus on the image domain; the complex nature of video data leaves dataset copyright auditing in the video domain unexplored. Specifically, video data introduces an additional temporal dimension, which poses significant challenges to the effectiveness and stealthiness of existing methods. In this paper, we propose VICTOR, the first dataset copyright auditing approach for video recognition systems. We develop a general and stealthy sample modification strategy that enhances the output discrepancy of the target model. By modifying only a small proportion of samples (e.g., 1%), VICTOR amplifies the impact of published modified samples on the prediction behavior of the target models. Then, the difference in the model's behavior for published modified and unpublished original samples can serve as a key basis for dataset auditing. Extensive experiments on multiple models and datasets highlight the superiority of VICTOR. Finally, we show that VICTOR is robust in the presence of several perturbation mechanisms to the training videos or the target models.
>
---
#### [new 122] Incentivizing Tool-augmented Thinking with Images for Medical Image Analysis
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出Ophiuchus框架，解决医学多模态大模型在复杂诊断中难以动态聚焦图像细粒度区域的问题。通过三阶段训练（冷启动、自反思微调、工具强化学习），实现模型自主决策调用视觉工具、精准定位并融合子图信息，提升VQA、检测与分割等任务性能。**

- **链接: [https://arxiv.org/pdf/2512.14157v1](https://arxiv.org/pdf/2512.14157v1)**

> **作者:** Yankai Jiang; Yujie Zhang; Peng Zhang; Yichen Li; Jintai Chen; Xiaoming Shi; Shihui Zhen
>
> **摘要:** Recent reasoning based medical MLLMs have made progress in generating step by step textual reasoning chains. However, they still struggle with complex tasks that necessitate dynamic and iterative focusing on fine-grained visual regions to achieve precise grounding and diagnosis. We introduce Ophiuchus, a versatile, tool-augmented framework that equips an MLLM to (i) decide when additional visual evidence is needed, (ii) determine where to probe and ground within the medical image, and (iii) seamlessly weave the relevant sub-image content back into an interleaved, multimodal chain of thought. In contrast to prior approaches limited by the performance ceiling of specialized tools, Ophiuchus integrates the model's inherent grounding and perception capabilities with external tools, thereby fostering higher-level reasoning. The core of our method is a three-stage training strategy: cold-start training with tool-integrated reasoning data to achieve basic tool selection and adaptation for inspecting key regions; self-reflection fine-tuning to strengthen reflective reasoning and encourage revisiting tool outputs; and Agentic Tool Reinforcement Learning to directly optimize task-specific rewards and emulate expert-like diagnostic behavior. Extensive experiments show that Ophiuchus consistently outperforms both closed-source and open-source SOTA methods across diverse medical benchmarks, including VQA, detection, and reasoning-based segmentation. Our approach illuminates a path toward medical AI agents that can genuinely "think with images" through tool-integrated reasoning. Datasets, codes, and trained models will be released publicly.
>
---
#### [new 123] Enhancing Semi-Supervised Multi-View Graph Convolutional Networks via Supervised Contrastive Learning and Self-Training
- **分类: cs.LG; cs.CV**

- **简介: 该论文面向半监督多视图图学习任务，旨在解决现有方法难以充分挖掘跨视图互补信息、图结构不鲁棒及未标注数据利用不足的问题。提出MV-SupGCN模型，融合监督对比学习、多策略图构建与自训练伪标签，提升特征判别性与跨视图一致性。**

- **链接: [https://arxiv.org/pdf/2512.13770v1](https://arxiv.org/pdf/2512.13770v1)**

> **作者:** Huaiyuan Xiao; Fadi Dornaika; Jingjun Bi
>
> **摘要:** The advent of graph convolutional network (GCN)-based multi-view learning provides a powerful framework for integrating structural information from heterogeneous views, enabling effective modeling of complex multi-view data. However, existing methods often fail to fully exploit the complementary information across views, leading to suboptimal feature representations and limited performance. To address this, we propose MV-SupGCN, a semi-supervised GCN model that integrates several complementary components with clear motivations and mutual reinforcement. First, to better capture discriminative features and improve model generalization, we design a joint loss function that combines Cross-Entropy loss with Supervised Contrastive loss, encouraging the model to simultaneously minimize intra-class variance and maximize inter-class separability in the latent space. Second, recognizing the instability and incompleteness of single graph construction methods, we combine both KNN-based and semi-supervised graph construction approaches on each view, thereby enhancing the robustness of the data structure representation and reducing generalization error. Third, to effectively utilize abundant unlabeled data and enhance semantic alignment across multiple views, we propose a unified framework that integrates contrastive learning in order to enforce consistency among multi-view embeddings and capture meaningful inter-view relationships, together with pseudo-labeling, which provides additional supervision applied to both the cross-entropy and contrastive loss functions to enhance model generalization. Extensive experiments demonstrate that MV-SupGCN consistently surpasses state-of-the-art methods across multiple benchmarks, validating the effectiveness of our integrated approach. The source code is available at https://github.com/HuaiyuanXiao/MVSupGCN
>
---
#### [new 124] CLAIM: Camera-LiDAR Alignment with Intensity and Monodepth
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出CLAIM方法，解决相机与LiDAR传感器外参标定问题。利用单目深度模型，通过粗到精搜索优化基于结构（Patch Pearson相关）和纹理（互信息）的双损失函数，无需特征提取或匹配，简单鲁棒，在KITTI等数据集上性能领先。**

- **链接: [https://arxiv.org/pdf/2512.14001v1](https://arxiv.org/pdf/2512.14001v1)**

> **作者:** Zhuo Zhang; Yonghui Liu; Meijie Zhang; Feiyang Tan; Yikang Ding
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** In this paper, we unleash the potential of the powerful monodepth model in camera-LiDAR calibration and propose CLAIM, a novel method of aligning data from the camera and LiDAR. Given the initial guess and pairs of images and LiDAR point clouds, CLAIM utilizes a coarse-to-fine searching method to find the optimal transformation minimizing a patched Pearson correlation-based structure loss and a mutual information-based texture loss. These two losses serve as good metrics for camera-LiDAR alignment results and require no complicated steps of data processing, feature extraction, or feature matching like most methods, rendering our method simple and adaptive to most scenes. We validate CLAIM on public KITTI, Waymo, and MIAS-LCEC datasets, and the experimental results demonstrate its superior performance compared with the state-of-the-art methods. The code is available at https://github.com/Tompson11/claim.
>
---
#### [new 125] WaveSim: A Wavelet-based Multi-scale Similarity Metric for Weather and Climate Fields
- **分类: physics.ao-ph; cs.CV; physics.data-an**

- **简介: 该论文提出WaveSim，一种基于小波变换的多尺度相似性度量方法，用于评估天气与气候空间场的相似性。它分解场为幅度、位移、结构三正交分量，按尺度量化差异，解决传统点对点指标无法归因误差到物理尺度的问题。**

- **链接: [https://arxiv.org/pdf/2512.14656v1](https://arxiv.org/pdf/2512.14656v1)**

> **作者:** Gabriele Accarino; Viviana Acquaviva; Sara Shamekh; Duncan Watson-Parris; David Lawrence
>
> **摘要:** We introduce WaveSim, a multi-scale similarity metric for the evaluation of spatial fields in weather and climate applications. WaveSim exploits wavelet transforms to decompose input fields into scale-specific wavelet coefficients. The metric is built by multiplying three orthogonal components derived from these coefficients: Magnitude, which quantifies similarities in the energy distribution of the coefficients, i.e., the intensity of the field; Displacement, which captures spatial shift by comparing the centers of mass of normalized energy distributions; and Structure, which assesses pattern organization independent of location and amplitude. Each component yields a scale-specific similarity score ranging from 0 (no similarity) to 1 (perfect similarity), which are then combined across scales to produce an overall similarity measure. We first evaluate WaveSim using synthetic test cases, applying controlled spatial and temporal perturbations to systematically assess its sensitivity and expected behavior. We then demonstrate its applicability to physically relevant case studies of key modes of climate variability in Earth System Models. Traditional point-wise metrics lack a mechanism for attributing errors to physical scales or modes of dissimilarity. By operating in the wavelet domain and decomposing the signal along independent axes, WaveSim bypasses these limitations and provides an interpretable and diagnostically rich framework for assessing similarity in complex fields. Additionally, the WaveSim framework allows users to place emphasis on a specific scale or component, and lends itself to user-specific model intercomparison, model evaluation, and calibration and training of forecasting systems. We provide a PyTorch-ready implementation of WaveSim, along with all evaluation scripts, at: https://github.com/gabrieleaccarino/wavesim.
>
---
#### [new 126] Establishing Stochastic Object Models from Noisy Data via Ambient Measurement-Integrated Diffusion
- **分类: cs.GR; cs.CV**

- **简介: 该论文属医学影像分析任务，旨在解决从含噪临床数据中构建真实、清洁的随机物体模型（SOMs）难题。提出无监督方法AMID，通过测量-扩散噪声解耦与环境损失函数，直接从噪声CT/乳腺X光数据学习高质量SOMs，提升任务导向图像质量评估可靠性。**

- **链接: [https://arxiv.org/pdf/2512.14187v1](https://arxiv.org/pdf/2512.14187v1)**

> **作者:** Jianwei Sun; Xiaoning Lei; Wenhao Cai; Xichen Xu; Yanshu Wang; Hu Gao
>
> **摘要:** Task-based measures of image quality (IQ) are critical for evaluating medical imaging systems, which must account for randomness including anatomical variability. Stochastic object models (SOMs) provide a statistical description of such variability, but conventional mathematical SOMs fail to capture realistic anatomy, while data-driven approaches typically require clean data rarely available in clinical tasks. To address this challenge, we propose AMID, an unsupervised Ambient Measurement-Integrated Diffusion with noise decoupling, which establishes clean SOMs directly from noisy measurements. AMID introduces a measurement-integrated strategy aligning measurement noise with the diffusion trajectory, and explicitly models coupling between measurement and diffusion noise across steps, an ambient loss is thus designed base on it to learn clean SOMs. Experiments on real CT and mammography datasets show that AMID outperforms existing methods in generation fidelity and yields more reliable task-based IQ evaluation, demonstrating its potential for unsupervised medical imaging analysis.
>
---
#### [new 127] WAM-Flow: Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出WAM-Flow，一种面向自动驾驶的端到端运动规划VLA模型。它将轨迹生成建模为离散流匹配任务，通过并行双向去噪实现粗到细规划，在NAVSIM上显著优于自回归与扩散基线。**

- **链接: [https://arxiv.org/pdf/2512.06112v2](https://arxiv.org/pdf/2512.06112v2)**

> **作者:** Yifang Xu; Jiahao Cui; Feipeng Cai; Zhihao Zhu; Hanlin Shang; Shan Luan; Mingwang Xu; Neng Zhang; Yaoyi Li; Jia Cai; Siyu Zhu
>
> **备注:** 18 pages, 11 figures. Code & Model: https://github.com/fudan-generative-vision/WAM-Flow
>
> **摘要:** We introduce WAM-Flow, a vision-language-action (VLA) model that casts ego-trajectory planning as discrete flow matching over a structured token space. In contrast to autoregressive decoders, WAM-Flow performs fully parallel, bidirectional denoising, enabling coarse-to-fine refinement with a tunable compute-accuracy trade-off. Specifically, the approach combines a metric-aligned numerical tokenizer that preserves scalar geometry via triplet-margin learning, a geometry-aware flow objective and a simulator-guided GRPO alignment that integrates safety, ego progress, and comfort rewards while retaining parallel generation. A multi-stage adaptation converts a pre-trained auto-regressive backbone (Janus-1.5B) from causal decoding to non-causal flow model and strengthens road-scene competence through continued multimodal pretraining. Thanks to the inherent nature of consistency model training and parallel decoding inference, WAM-Flow achieves superior closed-loop performance against autoregressive and diffusion-based VLA baselines, with 1-step inference attaining 89.1 PDMS and 5-step inference reaching 90.3 PDMS on NAVSIM v1 benchmark. These results establish discrete flow matching as a new promising paradigm for end-to-end autonomous driving. The code will be publicly available soon.
>
---
#### [new 128] JMMMU-Pro: Image-based Japanese Multi-discipline Multimodal Understanding Benchmark via Vibe Benchmark Construction
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出JMMMU-Pro——首个面向日语的图像型多学科多模态理解基准，旨在解决现有模型对日语图文联合理解能力评估不足的问题。作者设计Vibe构建法，用图像生成模型（如Nano Banana Pro）自动合成带日文文本的视觉问题，再经人工校验，高效构建高质量基准。**

- **链接: [https://arxiv.org/pdf/2512.14620v1](https://arxiv.org/pdf/2512.14620v1)**

> **作者:** Atsuyuki Miyai; Shota Onohara; Jeonghun Baek; Kiyoharu Aizawa
>
> **备注:** Project page: https://mmmu-japanese-benchmark.github.io/JMMMU_Pro/
>
> **摘要:** This paper introduces JMMMU-Pro, an image-based Japanese Multi-discipline Multimodal Understanding Benchmark, and Vibe Benchmark Construction, a scalable construction method. Following the evolution from MMMU to MMMU-Pro, JMMMU-Pro extends JMMMU by composing the question image and question text into a single image, thereby creating a benchmark that requires integrated visual-textual understanding through visual perception. To build JMMMU-Pro, we propose Vibe Benchmark Construction, a methodology in which an image generative model (e.g., Nano Banana Pro) produces candidate visual questions, and humans verify the outputs and, when necessary, regenerate with adjusted prompts to ensure quality. By leveraging Nano Banana Pro's highly realistic image generation capabilities and its ability to embed clean Japanese text, we construct a high-quality benchmark at low cost, covering a wide range of background and layout designs. Experimental results show that all open-source LMMs struggle substantially with JMMMU-Pro, underscoring JMMMU-Pro as an important benchmark for guiding future efforts in the open-source community. We believe that JMMMU-Pro provides a more rigorous evaluation tool for assessing the Japanese capabilities of LMMs and that our Vibe Benchmark Construction also offers an efficient guideline for future development of image-based VQA benchmarks.
>
---
#### [new 129] EEG-D3: A Solution to the Hidden Overfitting Problem of Deep Learning Models
- **分类: cs.LG; cs.AI; cs.CV; cs.HC**

- **简介: 该论文针对EEG深度学习模型在实际应用中泛化差的“隐式过拟合”问题，提出弱监督方法EEG-D3。它通过预测时间位置实现脑电潜变量解耦，分离真实神经成分与任务相关伪迹，并支持可解释建模与少样本学习。**

- **链接: [https://arxiv.org/pdf/2512.13806v1](https://arxiv.org/pdf/2512.13806v1)**

> **作者:** Siegfried Ludwig; Stylianos Bakas; Konstantinos Barmpas; Georgios Zoumpourlis; Dimitrios A. Adamos; Nikolaos Laskaris; Yannis Panagakis; Stefanos Zafeiriou
>
> **摘要:** Deep learning for decoding EEG signals has gained traction, with many claims to state-of-the-art accuracy. However, despite the convincing benchmark performance, successful translation to real applications is limited. The frequent disconnect between performance on controlled BCI benchmarks and its lack of generalisation to practical settings indicates hidden overfitting problems. We introduce Disentangled Decoding Decomposition (D3), a weakly supervised method for training deep learning models across EEG datasets. By predicting the place in the respective trial sequence from which the input window was sampled, EEG-D3 separates latent components of brain activity, akin to non-linear ICA. We utilise a novel model architecture with fully independent sub-networks for strict interpretability. We outline a feature interpretation paradigm to contrast the component activation profiles on different datasets and inspect the associated temporal and spatial filters. The proposed method reliably separates latent components of brain activity on motor imagery data. Training downstream classifiers on an appropriate subset of these components prevents hidden overfitting caused by task-correlated artefacts, which severely affects end-to-end classifiers. We further exploit the linearly separable latent space for effective few-shot learning on sleep stage classification. The ability to distinguish genuine components of brain activity from spurious features results in models that avoid the hidden overfitting problem and generalise well to real-world applications, while requiring only minimal labelled data. With interest to the neuroscience community, the proposed method gives researchers a tool to separate individual brain processes and potentially even uncover heretofore unknown dynamics.
>
---
#### [new 130] Test Time Optimized Generalized AI-based Medical Image Registration Method
- **分类: eess.IV; cs.CV**

- **简介: 该论文属医学图像配准任务，旨在解决传统非刚性配准方法参数调优复杂、计算开销大及深度学习模型泛化性差的问题。提出一种通用AI框架，支持跨模态（CT/MRI/超声）和跨解剖区域的3D非刚性配准，无需任务定制或重训练。**

- **链接: [https://arxiv.org/pdf/2512.14556v1](https://arxiv.org/pdf/2512.14556v1)**

> **作者:** Sneha Sree C.; Dattesh Shanbhag; Sudhanya Chatterjee
>
> **摘要:** Medical image registration is critical for aligning anatomical structures across imaging modalities such as computed tomography (CT), magnetic resonance imaging (MRI), and ultrasound. Among existing techniques, non-rigid registration (NRR) is particularly challenging due to the need to capture complex anatomical deformations caused by physiological processes like respiration or contrast-induced signal variations. Traditional NRR methods, while theoretically robust, often require extensive parameter tuning and incur high computational costs, limiting their use in real-time clinical workflows. Recent deep learning (DL)-based approaches have shown promise; however, their dependence on task-specific retraining restricts scalability and adaptability in practice. These limitations underscore the need for efficient, generalizable registration frameworks capable of handling heterogeneous imaging contexts. In this work, we introduce a novel AI-driven framework for 3D non-rigid registration that generalizes across multiple imaging modalities and anatomical regions. Unlike conventional methods that rely on application-specific models, our approach eliminates anatomy- or modality-specific customization, enabling streamlined integration into diverse clinical environments.
>
---
#### [new 131] EVOLVE-VLA: Test-Time Training from Environment Feedback for Vision-Language-Action Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出EVOLVE-VLA，属具身智能中的视觉-语言-动作（VLA）模型自适应学习任务。旨在解决VLA模型依赖大量示范、缺乏在线环境反馈下的持续适应能力问题。通过学习进展估计器与累积平滑、渐进视野扩展机制，实现零/少样本下的测试时自主训练与跨任务泛化。**

- **链接: [https://arxiv.org/pdf/2512.14666v1](https://arxiv.org/pdf/2512.14666v1)**

> **作者:** Zechen Bai; Chen Gao; Mike Zheng Shou
>
> **备注:** 15 pages
>
> **摘要:** Achieving truly adaptive embodied intelligence requires agents that learn not just by imitating static demonstrations, but by continuously improving through environmental interaction, which is akin to how humans master skills through practice. Vision-Language-Action (VLA) models have advanced robotic manipulation by leveraging large language models, yet remain fundamentally limited by Supervised Finetuning (SFT): requiring hundreds of demonstrations per task, rigidly memorizing trajectories, and failing to adapt when deployment conditions deviate from training. We introduce EVOLVE-VLA, a test-time training framework enabling VLAs to continuously adapt through environment interaction with minimal or zero task-specific demonstrations. The key technical challenge is replacing oracle reward signals (unavailable at test time) with autonomous feedback. We address this through a learned progress estimator providing dense feedback, and critically, we design our framework to ``tame'' this inherently noisy signal via two mechanisms: (1) an accumulative progress estimation mechanism smoothing noisy point-wise estimates, and (2) a progressive horizon extension strategy enabling gradual policy evolution. EVOLVE-VLA achieves substantial gains: +8.6\% on long-horizon tasks, +22.0\% in 1-shot learning, and enables cross-task generalization -- achieving 20.8\% success on unseen tasks without task-specific demonstrations training (vs. 0\% for pure SFT). Qualitative analysis reveals emergent capabilities absent in demonstrations, including error recovery and novel strategies. This work represents a critical step toward VLAs that truly learn and adapt, moving beyond static imitation toward continuous self-improvements.
>
---
#### [new 132] WAM-Diff: A Masked Diffusion VLA Framework with MoE and Online Reinforcement Learning for Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出WAM-Diff，一种面向自动驾驶的视觉-语言-动作（VLA）框架，解决端到端轨迹生成问题。它创新性地采用离散掩码扩散建模未来自车轨迹，结合MoE架构与在线强化学习（GSPO），提升预测精度与场景适应性。**

- **链接: [https://arxiv.org/pdf/2512.11872v1](https://arxiv.org/pdf/2512.11872v1)**

> **作者:** Mingwang Xu; Jiahao Cui; Feipeng Cai; Hanlin Shang; Zhihao Zhu; Shan Luan; Yifang Xu; Neng Zhang; Yaoyi Li; Jia Cai; Siyu Zhu
>
> **摘要:** End-to-end autonomous driving systems based on vision-language-action (VLA) models integrate multimodal sensor inputs and language instructions to generate planning and control signals. While autoregressive large language models and continuous diffusion policies are prevalent, the potential of discrete masked diffusion for trajectory generation remains largely unexplored. This paper presents WAM-Diff, a VLA framework that employs masked diffusion to iteratively refine a discrete sequence representing future ego-trajectories. Our approach features three key innovations: a systematic adaptation of masked diffusion for autonomous driving that supports flexible, non-causal decoding orders; scalable model capacity via a sparse MoE architecture trained jointly on motion prediction and driving-oriented visual question answering (VQA); and online reinforcement learning using Group Sequence Policy Optimization (GSPO) to optimize sequence-level driving rewards. Remarkably, our model achieves 91.0 PDMS on NAVSIM-v1 and 89.7 EPDMS on NAVSIM-v2, demonstrating the effectiveness of masked diffusion for autonomous driving. The approach provides a promising alternative to autoregressive and diffusion-based policies, supporting scenario-aware decoding strategies for trajectory generation. The code for this paper will be released publicly at: https://github.com/fudan-generative-vision/WAM-Diff
>
---
## 更新

#### [replaced 001] Beyond the Visible: Disocclusion-Aware Editing via Proxy Dynamic Graphs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13392v2](https://arxiv.org/pdf/2512.13392v2)**

> **作者:** Anran Qi; Changjian Li; Adrien Bousseau; Niloy J. Mitra
>
> **摘要:** We address image-to-video generation with explicit user control over the final frame's disoccluded regions. Current image-to-video pipelines produce plausible motion but struggle to generate predictable, articulated motions while enforcing user-specified content in newly revealed areas. Our key idea is to separate motion specification from appearance synthesis: we introduce a lightweight, user-editable Proxy Dynamic Graph (PDG) that deterministically yet approximately drives part motion, while a frozen diffusion prior is used to synthesize plausible appearance that follows that motion. In our training-free pipeline, the user loosely annotates and reposes a PDG, from which we compute a dense motion flow to leverage diffusion as a motion-guided shader. We then let the user edit appearance in the disoccluded areas of the image, and exploit the visibility information encoded by the PDG to perform a latent-space composite that reconciles motion with user intent in these areas. This design yields controllable articulation and user control over disocclusions without fine-tuning. We demonstrate clear advantages against state-of-the-art alternatives towards images turned into short videos of articulated objects, furniture, vehicles, and deformables. Our method mixes generative control, in the form of loose pose and structure, with predictable controls, in the form of appearance specification in the final frame in the disoccluded regions, unlocking a new image-to-video workflow. Code will be released on acceptance. Project page: https://anranqi.github.io/beyond-visible.github.io/
>
---
#### [replaced 002] Echo-CoPilot: A Multi-View, Multi-Task Agent for Echocardiography Interpretation and Reporting
- **分类: cs.AI; cs.CV; cs.LG; eess.IV**

- **链接: [https://arxiv.org/pdf/2512.09944v2](https://arxiv.org/pdf/2512.09944v2)**

> **作者:** Moein Heidari; Mohammad Amin Roohi; Ilker Hacihaliloglu
>
> **摘要:** Echocardiography is central to contemporary cardiovascular care, but full-study interpretation remains a cognitively demanding, multi-view task that is still performed manually. While recent foundation models for echocardiography can achieve strong performance on individual perceptual subtasks such as view classification, segmentation, or disease prediction, they typically operate in isolation and do not provide a unified, clinically coherent assessment. In this work, we introduce Echo-CoPilot, a multi-view, multi-task agent that uses a large language model to orchestrate a suite of specialized echocardiography tools. Within a ReAct-style loop, the agent decomposes clinician queries, invokes tools for view recognition, cardiac structure segmentation, measurement and disease prediction, and report synthesis, and integrates their outputs into guideline-aware answers and narrative summaries. We evaluate Echo-CoPilot on the public MIMIC-EchoQA benchmark, where it achieves an accuracy of 50.8\%, outperforming both general-purpose and biomedical video vision-language models. Qualitative analyses further show that the agent leverages quantitative measurements and physiologic context to resolve challenging cases near clinical decision thresholds, such as borderline left ventricular hypertrophy or pericardial effusion severity. The code will be released upon acceptance of the paper.
>
---
#### [replaced 003] LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.10040v3](https://arxiv.org/pdf/2501.10040v3)**

> **作者:** Wei Lu; Xue Yang; Si-Bao Chen
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** Light-weight neural networks for remote sensing (RS) visual analysis must overcome two inherent redundancies: spatial redundancy from vast, homogeneous backgrounds, and channel redundancy, where extreme scale variations render a single feature space inefficient. Existing models, often designed for natural images, fail to address this dual challenge in RS scenarios. To bridge this gap, we propose LWGANet, a light-weight backbone engineered for RS-specific properties. LWGANet introduces two core innovations: a Top-K Global Feature Interaction (TGFI) module that mitigates spatial redundancy by focusing computation on salient regions, and a Light-Weight Grouped Attention (LWGA) module that resolves channel redundancy by partitioning channels into specialized, scale-specific pathways. By synergistically resolving these core inefficiencies, LWGANet achieves a superior trade-off between feature representation quality and computational cost. Extensive experiments on twelve diverse datasets across four major RS tasks--scene classification, oriented object detection, semantic segmentation, and change detection--demonstrate that LWGANet consistently outperforms state-of-the-art light-weight backbones in both accuracy and efficiency. Our work establishes a new, robust baseline for efficient visual analysis in RS images.
>
---
#### [replaced 004] HQ-DM: Single Hadamard Transformation-Based Quantization-Aware Training for Low-Bit Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.05746v2](https://arxiv.org/pdf/2512.05746v2)**

> **作者:** Shizhuo Mao; Hongtao Zou; Qihu Xie; Song Chen; Yi Kang
>
> **摘要:** Diffusion models have demonstrated significant applications in the field of image generation. However, their high computational and memory costs pose challenges for deployment. Model quantization has emerged as a promising solution to reduce storage overhead and accelerate inference. Nevertheless, existing quantization methods for diffusion models struggle to mitigate outliers in activation matrices during inference, leading to substantial performance degradation under low-bit quantization scenarios. To address this, we propose HQ-DM, a novel Quantization-Aware Training framework that applies Single Hadamard Transformation to activation matrices. This approach effectively reduces activation outliers while preserving model performance under quantization. Compared to traditional Double Hadamard Transformation, our proposed scheme offers distinct advantages by seamlessly supporting INT convolution operations while preventing the amplification of weight outliers. For conditional generation on the ImageNet 256x256 dataset using the LDM-4 model, our W4A4 and W4A3 quantization schemes improve the Inception Score by 12.8% and 467.73%, respectively, over the existing state-of-the-art method.
>
---
#### [replaced 005] AI-GenBench: A New Ongoing Benchmark for AI-Generated Image Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.20865v3](https://arxiv.org/pdf/2504.20865v3)**

> **作者:** Lorenzo Pellegrini; Davide Cozzolino; Serafino Pandolfini; Davide Maltoni; Matteo Ferrara; Luisa Verdoliva; Marco Prati; Marco Ramilli
>
> **备注:** Accepted at Verimedia workshop, IJCNN 2025. 9 pages, 6 figures, 4 tables, code available: https://github.com/MI-BioLab/AI-GenBench
>
> **摘要:** The rapid advancement of generative AI has revolutionized image creation, enabling high-quality synthesis from text prompts while raising critical challenges for media authenticity. We present Ai-GenBench, a novel benchmark designed to address the urgent need for robust detection of AI-generated images in real-world scenarios. Unlike existing solutions that evaluate models on static datasets, Ai-GenBench introduces a temporal evaluation framework where detection methods are incrementally trained on synthetic images, historically ordered by their generative models, to test their ability to generalize to new generative models, such as the transition from GANs to diffusion models. Our benchmark focuses on high-quality, diverse visual content and overcomes key limitations of current approaches, including arbitrary dataset splits, unfair comparisons, and excessive computational demands. Ai-GenBench provides a comprehensive dataset, a standardized evaluation protocol, and accessible tools for both researchers and non-experts (e.g., journalists, fact-checkers), ensuring reproducibility while maintaining practical training requirements. By establishing clear evaluation rules and controlled augmentation strategies, Ai-GenBench enables meaningful comparison of detection methods and scalable solutions. Code and data are publicly available to ensure reproducibility and to support the development of robust forensic detectors to keep pace with the rise of new synthetic generators.
>
---
#### [replaced 006] TextMesh4D: Text-to-4D Mesh Generation via Jacobian Deformation Field
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.24121v2](https://arxiv.org/pdf/2506.24121v2)**

> **作者:** Sisi Dai; Xinxin Su; Ruizhen Hu; Kai Xu
>
> **摘要:** Dynamic 3D (4D) content generation, particularly text-to-4D, remains a challenging and under-explored problem due to its inherent spatiotemporal complexity. Existing text-to-4D methods typically avoid direct mesh generation due to inherent topological constraints, favoring alternative representations like NeRFs or 3DGS. However, these non-mesh approaches, suffer from insufficient geometric fidelity, temporal artifacts, and limited compatibility with modern computer graphics (CG) pipelines. In contrast, directly generating dynamic meshes faces two key challenges: i) deformation inflexibility, as traditional vertex-based optimization is constrained by meshes' explicitly encoded topology, and ii) semantic inconsistency, arising from stochastic noise in distilled priors. In this paper, we introduce TextMesh4D, a pioneering framework for text-to-4D mesh generation that directly addresses these challenges. TextMesh4D features two core innovations: 1) the Jacobian Deformation Field (JDF), which shifts the deformation unit from vertices to faces, using per-face Jacobians to model flexible transformations free from topological constraints. 2) the Local-Global Semantic Regularizer (LGSR), which leverages the mesh's innate geometric properties to enforce semantic coherence both locally and globally across frames. Extensive experiments demonstrate that TextMesh4D achieves state-of-the-art performance in temporal consistency, structural fidelity, and visual realism, while requiring only a single 24GB GPU. Our work establishes a new benchmark for efficient and high-quality text-to-4D mesh generation. The code will be released to facilitate future research.
>
---
#### [replaced 007] Bi-Erasing: A Bidirectional Framework for Concept Removal in Diffusion Models
- **分类: cs.CV; cs.CR**

- **链接: [https://arxiv.org/pdf/2512.13039v2](https://arxiv.org/pdf/2512.13039v2)**

> **作者:** Hao Chen; Yiwei Wang; Songze Li
>
> **备注:** Under Review
>
> **摘要:** Concept erasure, which fine-tunes diffusion models to remove undesired or harmful visual concepts, has become a mainstream approach to mitigating unsafe or illegal image generation in text-to-image models.However, existing removal methods typically adopt a unidirectional erasure strategy by either suppressing the target concept or reinforcing safe alternatives, making it difficult to achieve a balanced trade-off between concept removal and generation quality. To address this limitation, we propose a novel Bidirectional Image-Guided Concept Erasure (Bi-Erasing) framework that performs concept suppression and safety enhancement simultaneously. Specifically, based on the joint representation of text prompts and corresponding images, Bi-Erasing introduces two decoupled image branches: a negative branch responsible for suppressing harmful semantics and a positive branch providing visual guidance for safe alternatives. By jointly optimizing these complementary directions, our approach achieves a balance between erasure efficacy and generation usability. In addition, we apply mask-based filtering to the image branches to prevent interference from irrelevant content during the erasure process. Across extensive experiment evaluations, the proposed Bi-Erasing outperforms baseline methods in balancing concept removal effectiveness and visual fidelity.
>
---
#### [replaced 008] RTR-GS: 3D Gaussian Splatting for Inverse Rendering with Radiance Transfer and Reflection
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.07733v2](https://arxiv.org/pdf/2507.07733v2)**

> **作者:** Yongyang Zhou; Fang-Lue Zhang; Zichen Wang; Lei Zhang
>
> **备注:** 16 pages
>
> **摘要:** 3D Gaussian Splatting (3DGS) has demonstrated impressive capabilities in novel view synthesis. However, rendering reflective objects remains a significant challenge, particularly in inverse rendering and relighting. We introduce RTR-GS, a novel inverse rendering framework capable of robustly rendering objects with arbitrary reflectance properties, decomposing BRDF and lighting, and delivering credible relighting results. Given a collection of multi-view images, our method effectively recovers geometric structure through a hybrid rendering model that combines forward rendering for radiance transfer with deferred rendering for reflections. This approach successfully separates high-frequency and low-frequency appearances, mitigating floating artifacts caused by spherical harmonic overfitting when handling high-frequency details. We further refine BRDF and lighting decomposition using an additional physically-based deferred rendering branch. Experimental results show that our method enhances novel view synthesis, normal estimation, decomposition, and relighting while maintaining efficient training inference process.
>
---
#### [replaced 009] Weakly Supervised Pneumonia Localization from Chest X-Rays Using Deep Neural Network and Grad-CAM Explanations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.00456v4](https://arxiv.org/pdf/2511.00456v4)**

> **作者:** Kiran Shahi; Anup Bagale
>
> **备注:** https://github.com/kiranshahi/pneumonia-analysis
>
> **摘要:** Chest X-ray imaging is commonly used to diagnose pneumonia, but accurately localizing the pneumonia affected regions typically requires detailed pixel-level annotations, which are costly and time consuming to obtain. To address this limitation, this study proposes a weakly supervised deep learning framework for pneumonia classification and localization using Gradient-weighted Class Activation Mapping (Grad-CAM). Instead of relying on costly pixel-level annotations, the proposed method utilizes image-level labels to generate clinically meaningful heatmaps that highlight pneumonia affected regions. Furthermore, we evaluate seven pre-trained deep learning models including a Vision Transformer under identical training conditions, using focal loss and patient-wise splits to prevent data leakage. Experimental results suggest that all models achieved high accuracy (96-98%), with ResNet-18 and EfficientNet-B0 showing the best overall performance and MobileNet-V2 providing an efficient lightweight alternative. Grad-CAM heatmap visualizations in this study confirm that the proposed methods focus on clinically relevant lung regions, supporting the use of explainable AI for radiological diagnostics. Overall, this work highlights the potential of weakly supervised, explainable models that enhance transparency and clinical trust in AI-assisted pneumonia screening.
>
---
#### [replaced 010] Semantic-Drive: Democratizing Long-Tail Data Curation via Open-Vocabulary Grounding and Neuro-Symbolic VLM Consensus
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文提出Semantic-Drive，解决自动驾驶中长尾安全事件（如异常闯入）数据难挖掘的问题。它采用本地化、神经符号融合框架：先用YOLOE进行开放词汇语义定位，再通过多模型共识的推理型VLM做细粒度场景分析，在保护隐私前提下显著提升召回率与风险评估精度。**

- **链接: [https://arxiv.org/pdf/2512.12012v2](https://arxiv.org/pdf/2512.12012v2)**

> **作者:** Antonio Guillen-Perez
>
> **摘要:** The development of robust Autonomous Vehicles (AVs) is bottlenecked by the scarcity of "Long-Tail" training data. While fleets collect petabytes of video logs, identifying rare safety-critical events (e.g., erratic jaywalking, construction diversions) remains a manual, cost-prohibitive process. Existing solutions rely on coarse metadata search, which lacks precision, or cloud-based VLMs, which are privacy-invasive and expensive. We introduce Semantic-Drive, a local-first, neuro-symbolic framework for semantic data mining. Our approach decouples perception into two stages: (1) Symbolic Grounding via a real-time open-vocabulary detector (YOLOE) to anchor attention, and (2) Cognitive Analysis via a Reasoning VLM that performs forensic scene analysis. To mitigate hallucination, we implement a "System 2" inference-time alignment strategy, utilizing a multi-model "Judge-Scout" consensus mechanism. Benchmarked on the nuScenes dataset against the Waymo Open Dataset (WOD-E2E) taxonomy, Semantic-Drive achieves a Recall of 0.966 (vs. 0.475 for CLIP) and reduces Risk Assessment Error by 40% ccompared to the best single scout models. The system runs entirely on consumer hardware (NVIDIA RTX 3090), offering a privacy-preserving alternative to the cloud.
>
---
#### [replaced 011] Exo2Ego: Exocentric Knowledge Guided MLLM for Egocentric Video Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.09143v2](https://arxiv.org/pdf/2503.09143v2)**

> **作者:** Haoyu Zhang; Qiaohui Chu; Meng Liu; Haoxiang Shi; Yaowei Wang; Liqiang Nie
>
> **备注:** This paper is accepted by AAAI 2026
>
> **摘要:** AI personal assistants, deployed through robots or wearables, require embodied understanding to collaborate effectively with humans. However, current Multimodal Large Language Models (MLLMs) primarily focus on third-person (exocentric) vision, overlooking the unique challenges of first-person (egocentric) videos. Additionally, high acquisition costs limit data size, impairing MLLM performance. To address these challenges, we propose learning the mapping between exocentric and egocentric domains, leveraging the extensive exocentric knowledge within existing MLLMs to enhance egocentric video understanding. To this end, we introduce Ego-ExoClip, a pre-training dataset comprising 1.1M synchronized ego-exo clip-text pairs derived from Ego-Exo4D, together with the instruction-tuning dataset EgoIT, which is collected from multiple sources to enhance the model's instruction-following capabilities. Building upon the datasets, we propose a migration strategy and further design a progressive mapping learning pipeline with three stages: Demonstrator Self-Preparation, Demonstrator-Learner Guidance, and Learner Self-Practice. Extensive experiments across diverse egocentric tasks reveal that existing MLLMs perform inadequately in egocentric video understanding, while our model significantly outperforms these leading models.
>
---
#### [replaced 012] OLATverse: A Large-scale Real-world Object Dataset with Precise Lighting Control
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2511.02483v3](https://arxiv.org/pdf/2511.02483v3)**

> **作者:** Xilong Zhou; Jianchun Chen; Pramod Rao; Timo Teufel; Linjie Lyu; Tigran Minasian; Oleksandr Sotnychenko; Xiao-Xiao Long; Marc Habermann; Christian Theobalt
>
> **摘要:** We introduce OLATverse, a large-scale dataset comprising around 9M images of 765 real-world objects, captured from multiple viewpoints under a diverse set of precisely controlled lighting conditions. While recent advances in object-centric inverse rendering, novel view synthesis and relighting have shown promising results, most techniques still heavily rely on the synthetic datasets for training and small-scale real-world datasets for benchmarking, which limits their realism and generalization. To address this gap, OLATverse offers two key advantages over existing datasets: large-scale coverage of real objects and high-fidelity appearance under precisely controlled illuminations. Specifically, OLATverse contains 765 common and uncommon real-world objects, spanning a wide range of material categories. Each object is captured using 35 DSLR cameras and 331 individually controlled light sources, enabling the simulation of diverse illumination conditions. In addition, for each object, we provide well-calibrated camera parameters, accurate object masks, photometric surface normals, and diffuse albedo as auxiliary resources. We also construct an extensive evaluation set, establishing the first comprehensive real-world object-centric benchmark for inverse rendering and normal estimation. We believe that OLATverse represents a pivotal step toward integrating the next generation of inverse rendering and relighting methods with real-world data. The full dataset, along with all post-processing workflows, will be publicly released at https://vcai.mpi-inf.mpg.de/projects/OLATverse/.
>
---
#### [replaced 013] Translating Electrocardiograms to Cardiac Magnetic Resonance Imaging Useful for Cardiac Assessment and Disease Screening: A Multi-Center Study
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2411.13602v3](https://arxiv.org/pdf/2411.13602v3)**

> **作者:** Zhengyao Ding; Ziyu Li; Yujian Hu; Youyao Xu; Chengchen Zhao; Yiheng Mao; Haitao Li; Zhikang Li; Qian Li; Jing Wang; Yue Chen; Mengjia Chen; Longbo Wang; Xuesen Chu; Weichao Pan; Ziyi Liu; Fei Wu; Hongkun Zhang; Ting Chen; Zhengxing Huang
>
> **备注:** 29 pages, 7 figures
>
> **摘要:** Cardiovascular diseases (CVDs) are the leading cause of global mortality, necessitating accessible and accurate diagnostic tools. While cardiac magnetic resonance imaging (CMR) provides gold-standard insights into cardiac structure and function, its clinical utility is limited by high cost and complexity. In contrast, electrocardiography (ECG) is inexpensive and widely available but lacks the granularity of CMR. We propose CardioNets, a deep learning framework that translates 12-lead ECG signals into CMR-level functional parameters and synthetic images, enabling scalable cardiac assessment. CardioNets integrates cross-modal contrastive learning and generative pretraining, aligning ECG with CMR-derived cardiac phenotypes and synthesizing high-resolution CMR images via a masked autoregressive model. Trained on 159,819 samples from five cohorts, including the UK Biobank (n=42,483) and MIMIC-IV-ECG (n=164,550), and externally validated on independent clinical datasets (n=3,767), CardioNets achieved strong performance across disease screening and phenotype estimation tasks. In the UK Biobank, it improved cardiac phenotype regression R2 by 24.8% and cardiomyopathy AUC by up to 39.3% over baseline models. In MIMIC, it increased AUC for pulmonary hypertension detection by 5.6%. Generated CMR images showed 36.6% higher SSIM and 8.7% higher PSNR than prior approaches. In a reader study, ECG-only CardioNets achieved 13.9% higher accuracy than human physicians using both ECG and real CMR. These results suggest that CardioNets offers a promising, low-cost alternative to CMR for large-scale CVD screening, particularly in resource-limited settings. Future efforts will focus on clinical deployment and regulatory validation of ECG-based synthetic imaging.
>
---
#### [replaced 014] INQUIRE-Search: A Framework for Interactive Discovery in Large-Scale Biodiversity Databases
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15656v3](https://arxiv.org/pdf/2511.15656v3)**

> **作者:** Edward Vendrow; Julia Chae; Rupa Kurinchi-Vendhan; Isaac Eckert; Jazlynn Hall; Marta Jarzyna; Reymond Miyajima; Ruth Oliver; Laura Pollock; Lauren Shrack; Scott Yanco; Oisin Mac Aodha; Sara Beery
>
> **备注:** EV, JC, RKV contributed equally
>
> **摘要:** Large community science platforms such as iNaturalist contain hundreds of millions of biodiversity images that often capture ecological context on behaviors, interactions, phenology, and habitat. Yet most ecological workflows rely on metadata filtering or manual inspection, leaving this secondary information inaccessible at scale. We introduce INQUIRE-Search, an open-source system that enables scientists to rapidly and interactively search within an ecological image database for specific concepts using natural language, verify and export relevant observations, and utilize this discovered data for novel scientific analysis. Compared to traditional methods, INQUIRE-Search takes a fraction of the time, opening up new possibilities for scientific questions that can be explored. Through five case studies, we show the diversity of scientific applications that a tool like INQUIRE-Search can support, from seasonal variation in behavior across species to forest regrowth after wildfires. These examples demonstrate a new paradigm for interactive, efficient, and scalable scientific discovery that can begin to unlock previously inaccessible scientific value in large-scale biodiversity datasets. Finally, we emphasize using such AI-enabled discovery tools for science call for experts to reframe the priorities of the scientific process and develop novel methods for experiment design, data collection, survey effort, and uncertainty analysis.
>
---
#### [replaced 015] From My View to Yours: Ego-to-Exo Transfer in VLMs for Understanding Activities of Daily Living
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.05711v3](https://arxiv.org/pdf/2501.05711v3)**

> **作者:** Dominick Reilly; Manish Kumar Govind; Le Xue; Srijan Das
>
> **摘要:** Vision Language Models (VLMs) have achieved strong performance across diverse video understanding tasks. However, their viewpoint invariant training limits their ability to understand egocentric properties (e.g., human object interactions) from exocentric video observations. This limitation is critical for many applications, such as Activities of Daily Living (ADL) monitoring, where the understanding of egocentric properties is essential, and egocentric cameras are impractical to deploy. To address this limitation, we propose Ego2ExoVLM, a VLM that learns to infer egocentric properties from exocentric videos by leveraging time-synchronized ego-exo videos during training. Ego2ExoVLM accomplishes this through the use of two components: Ego2Exo Sequence Distillation, which transfers knowledge from an egocentric teacher to an exocentric student, and Ego Adaptive Visual Tokens, designed to enhance the effectiveness of this knowledge transfer. To measure this capability, we introduce Ego-in-Exo Perception, a benchmark of 3.9K questions curated to explicitly measure the understanding of egocentric properties from exocentric videos. Ego2ExoVLM is evaluated on 10 tasks across Ego-in-Exo Perception and existing ADL benchmarks, achieving state-of-the-art results on the ADL-X benchmark suite and outperforming strong baselines on our proposed benchmark. All code, models, and data will be released at https://github.com/dominickrei/EgoExo4ADL.
>
---
#### [replaced 016] Semantic-Free Procedural 3D Shapes Are Surprisingly Good Teachers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.17467v3](https://arxiv.org/pdf/2411.17467v3)**

> **作者:** Xuweiyi Chen; Zezhou Cheng
>
> **备注:** 3DV | SynData4CV @ CVPR2025 | Project Page: https://point-mae-zero.cs.virginia.edu/
>
> **摘要:** Self-supervised learning has emerged as a promising approach for acquiring transferable 3D representations from unlabeled 3D point clouds. Unlike 2D images, which are widely accessible, acquiring 3D assets requires specialized expertise or professional 3D scanning equipment, making it difficult to scale and raising copyright concerns. To address these challenges, we propose learning 3D representations from procedural 3D programs that automatically generate 3D shapes using simple 3D primitives and augmentations. Remarkably, despite lacking semantic content, the 3D representations learned from the procedurally generated 3D shapes perform on par with state-of-the-art representations learned from semantically recognizable 3D models (e.g., airplanes) across various downstream 3D tasks, such as shape classification, part segmentation, masked point cloud completion, and both scene semantic and instance segmentation. We provide a detailed analysis on factors that make a good 3D procedural programs. Extensive experiments further suggest that current 3D self-supervised learning methods on point clouds do not rely on semantics of 3D shapes, shedding light on the nature of 3D representations learned.
>
---
#### [replaced 017] MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2312.04960v5](https://arxiv.org/pdf/2312.04960v5)**

> **作者:** Xiaoyun Xu; Shujian Yu; Zhuoran Liu; Stjepan Picek
>
> **备注:** Accepted by NDSS 2026
>
> **摘要:** Vision Transformers (ViTs) have emerged as a fundamental architecture and serve as the backbone of modern vision-language models. Despite their impressive performance, ViTs exhibit notable vulnerability to evasion attacks, necessitating the development of specialized Adversarial Training (AT) strategies tailored to their unique architecture. While a direct solution might involve applying existing AT methods to ViTs, our analysis reveals significant incompatibilities, particularly with state-of-the-art (SOTA) approaches such as Generalist (CVPR 2023) and DBAT (USENIX Security 2024). This paper presents a systematic investigation of adversarial robustness in ViTs and provides a novel theoretical Mutual Information (MI) analysis in its autoencoder-based self-supervised pre-training. Specifically, we show that MI between the adversarial example and its latent representation in ViT-based autoencoders should be constrained via derived MI bounds. Building on this insight, we propose a self-supervised AT method, MIMIR, that employs an MI penalty to facilitate adversarial pre-training by masked image modeling with autoencoders. Extensive experiments on CIFAR-10, Tiny-ImageNet, and ImageNet-1K show that MIMIR can consistently provide improved natural and robust accuracy, where MIMIR outperforms SOTA AT results on ImageNet-1K. Notably, MIMIR demonstrates superior robustness against unforeseen attacks and common corruption data and can also withstand adaptive attacks where the adversary possesses full knowledge of the defense mechanism. Our code and trained models are publicly available at: https://github.com/xiaoyunxxy/MIMIR.
>
---
#### [replaced 018] UnCageNet: Tracking and Pose Estimation of Caged Animal
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.07712v2](https://arxiv.org/pdf/2512.07712v2)**

> **作者:** Sayak Dutta; Harish Katti; Shashikant Verma; Shanmuganathan Raman
>
> **备注:** 9 pages, 2 figures, 2 tables. Accepted to the Indian Conference on Computer Vision, Graphics, and Image Processing (ICVGIP 2025), Mandi, India
>
> **摘要:** Animal tracking and pose estimation systems, such as STEP (Simultaneous Tracking and Pose Estimation) and ViTPose, experience substantial performance drops when processing images and videos with cage structures and systematic occlusions. We present a three-stage preprocessing pipeline that addresses this limitation through: (1) cage segmentation using a Gabor-enhanced ResNet-UNet architecture with tunable orientation filters, (2) cage inpainting using CRFill for content-aware reconstruction of occluded regions, and (3) evaluation of pose estimation and tracking on the uncaged frames. Our Gabor-enhanced segmentation model leverages orientation-aware features with 72 directional kernels to accurately identify and segment cage structures that severely impair the performance of existing methods. Experimental validation demonstrates that removing cage occlusions through our pipeline enables pose estimation and tracking performance comparable to that in environments without occlusions. We also observe significant improvements in keypoint detection accuracy and trajectory consistency.
>
---
#### [replaced 019] CCMNet: Leveraging Calibrated Color Correction Matrices for Cross-Camera Color Constancy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.07959v2](https://arxiv.org/pdf/2504.07959v2)**

> **作者:** Dongyoung Kim; Mahmoud Afifi; Dongyun Kim; Michael S. Brown; Seon Joo Kim
>
> **摘要:** Computational color constancy, or white balancing, is a key module in a camera's image signal processor (ISP) that corrects color casts from scene lighting. Because this operation occurs in the camera-specific raw color space, white balance algorithms must adapt to different cameras. This paper introduces a learning-based method for cross-camera color constancy that generalizes to new cameras without retraining. Our method leverages pre-calibrated color correction matrices (CCMs) available on ISPs that map the camera's raw color space to a standard space (e.g., CIE XYZ). Our method uses these CCMs to transform predefined illumination colors (i.e., along the Planckian locus) into the test camera's raw space. The mapped illuminants are encoded into a compact camera fingerprint embedding (CFE) that enables the network to adapt to unseen cameras. To prevent overfitting due to limited cameras and CCMs during training, we introduce a data augmentation technique that interpolates between cameras and their CCMs. Experimental results across multiple datasets and backbones show that our method achieves state-of-the-art cross-camera color constancy while remaining lightweight and relying only on data readily available in camera ISPs.
>
---
#### [replaced 020] Adapting General-Purpose Foundation Models for X-ray Ptychography in Low-Data Regimes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.02503v2](https://arxiv.org/pdf/2511.02503v2)**

> **作者:** Robinson Umeike; Neil Getty; Yin Xiangyu; Yi Jiang
>
> **摘要:** The automation of workflows in advanced microscopy is a key goal where foundation models like Language Models (LLMs) and Vision-Language Models (VLMs) show great potential. However, adapting these general-purpose models for specialized scientific tasks is critical, and the optimal domain adaptation strategy is often unclear. To address this, we introduce PtychoBench, a new multi-modal, multi-task benchmark for ptychographic analysis. Using this benchmark, we systematically compare two specialization strategies: Supervised Fine-Tuning (SFT) and In-Context Learning (ICL). We evaluate these strategies on a visual artifact detection task with VLMs and a textual parameter recommendation task with LLMs in a data-scarce regime. Our findings reveal that the optimal specialization pathway is task-dependent. For the visual task, SFT and ICL are highly complementary, with a fine-tuned model guided by context-aware examples achieving the highest mean performance (Micro-F1 of 0.728). Conversely, for the textual task, ICL on a large base model is the superior strategy, reaching a peak Micro-F1 of 0.847 and outperforming a powerful "super-expert" SFT model (0-shot Micro-F1 of 0.839). We also confirm the superiority of context-aware prompting and identify a consistent contextual interference phenomenon in fine-tuned models. These results, benchmarked against strong baselines including GPT-4o and a DINOv3-based classifier, offer key observations for AI in science: the optimal specialization path in our benchmark is dependent on the task modality, offering a clear framework for developing more effective science-based agentic systems.
>
---
#### [replaced 021] SIGMMA: Hierarchical Graph-Based Multi-Scale Multi-modal Contrastive Alignment of Histopathology Image and Spatial Transcriptome
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.15464v3](https://arxiv.org/pdf/2511.15464v3)**

> **作者:** Dabin Jeong; Amirhossein Vahidi; Ciro Ramírez-Suástegui; Marie Moullet; Kevin Ly; Mohammad Vali Sanian; Sebastian Birk; Yinshui Chang; Adam Boxall; Daniyal Jafree; Lloyd Steele; Vijaya Baskar MS; Muzlifah Haniffa; Mohammad Lotfollahi
>
> **摘要:** Recent advances in computational pathology have leveraged vision-language models to learn joint representations of Hematoxylin and Eosin (HE) images with spatial transcriptomic (ST) profiles. However, existing approaches typically align HE tiles with their corresponding ST profiles at a single scale, overlooking fine-grained cellular structures and their spatial organization. To address this, we propose Sigmma, a multi-modal contrastive alignment framework for learning hierarchical representations of HE images and spatial transcriptome profiles across multiple scales. Sigmma introduces multi-scale contrastive alignment, ensuring that representations learned at different scales remain coherent across modalities. Furthermore, by representing cell interactions as a graph and integrating inter- and intra-subgraph relationships, our approach effectively captures cell-cell interactions, ranging from fine to coarse, within the tissue microenvironment. We demonstrate that Sigmm learns representations that better capture cross-modal correspondences, leading to an improvement of avg. 9.78\% in the gene-expression prediction task and avg. 26.93\% in the cross-modal retrieval task across datasets. We further show that it learns meaningful multi-tissue organization in downstream analyses.
>
---
#### [replaced 022] SAM3-I: Segment Anything with Instructions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04585v2](https://arxiv.org/pdf/2512.04585v2)**

> **作者:** Jingjing Li; Yue Feng; Yuchen Guo; Jincai Huang; Yongri Piao; Qi Bi; Miao Zhang; Xiaoqi Zhao; Qiang Chen; Shihao Zou; Wei Ji; Huchuan Lu; Li Cheng
>
> **备注:** Preliminary results; work in progress
>
> **摘要:** Segment Anything Model 3 (SAM3) has advanced open-vocabulary segmentation through promptable concept segmentation, allowing users to segment all instances corresponding to a given concept, typically specified with short noun-phrase (NP) prompts. While this marks the first integration of language-level concepts within the SAM family, real-world usage typically requires far richer expressions that include attributes, spatial relations, functionalities, actions, states, and even implicit reasoning over instances. Currently, SAM3 relies on external multi-modal agents to convert complex instructions into NPs and then conduct iterative mask filtering. However, these NP-level concepts remain overly coarse, often failing to precisely represent a specific instance. In this work, we present SAM3-I, an enhanced framework that unifies concept-level understanding and instruction-level reasoning within the SAM family. SAM3-I introduces an instruction-aware cascaded adaptation mechanism that progressively aligns expressive instruction semantics with SAM3's existing vision-language representations, enabling direct instruction-following segmentation without sacrificing its original concept-driven capabilities. Furthermore, we design a structured instruction taxonomy spanning concept, simple, and complex levels, and develop a scalable data engine to construct a dataset with diverse instruction-mask pairs. Experiments show that SAM3-I delivers appealing performance, demonstrating that SAM3 can be effectively extended to follow natural-language instructions while preserving its strong concept grounding. We open-source SAM3-I and provide practical fine-tuning workflows, enabling researchers to adapt it to domain-specific applications. The source code is available here.
>
---
#### [replaced 023] Learning neuroimaging models from health system-scale data
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.18638v2](https://arxiv.org/pdf/2509.18638v2)**

> **作者:** Yiwei Lyu; Samir Harake; Asadur Chowdury; Soumyanil Banerjee; Rachel Gologorsky; Shixuan Liu; Anna-Katharina Meissner; Akshay Rao; Chenhui Zhao; Akhil Kondepudi; Cheng Jiang; Xinhai Hou; Rushikesh S. Joshi; Volker Neuschmelting; Ashok Srinivasan; Dawn Kleindorfer; Brian Athey; Vikas Gulani; Aditya Pandey; Honglak Lee; Todd Hollon
>
> **摘要:** Neuroimaging is a ubiquitous tool for evaluating patients with neurological diseases. The global demand for magnetic resonance imaging (MRI) studies has risen steadily, placing significant strain on health systems, prolonging turnaround times, and intensifying physician burnout. These challenges disproportionately impact patients in low-resource and rural settings. Here, we utilized a large academic health system as a data engine to develop Prima, the first vision language model (VLM) serving as an AI foundation for neuroimaging that supports real-world, clinical MRI studies as input. Trained on over 220,000 MRI studies, Prima uses a hierarchical vision architecture that provides general and transferable MRI features. Prima was tested in a 1-year health system-wide study that included 30K MRI studies. Across 52 radiologic diagnoses from the major neurologic disorders, including neoplastic, inflammatory, infectious, and developmental lesions, Prima achieved a mean diagnostic area under the ROC curve of 92.0, outperforming other state-of-the-art general and medical AI models. Prima offers explainable differential diagnoses, worklist priority for radiologists, and clinical referral recommendations across diverse patient demographics and MRI systems. Prima demonstrates algorithmic fairness across sensitive groups and can help mitigate health system biases, such as prolonged turnaround times for low-resource populations. These findings highlight the transformative potential of health system-scale VLMs and Prima's role in advancing AI-driven healthcare.
>
---
#### [replaced 024] On the Design of One-step Diffusion via Shortcutting Flow Paths
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.11831v2](https://arxiv.org/pdf/2512.11831v2)**

> **作者:** Haitao Lin; Peiyan Hu; Minsi Ren; Zhifeng Gao; Zhi-Ming Ma; Guolin ke; Tailin Wu; Stan Z. Li
>
> **备注:** 10 pages of main body, conference paper
>
> **摘要:** Recent advances in few-step diffusion models have demonstrated their efficiency and effectiveness by shortcutting the probabilistic paths of diffusion models, especially in training one-step diffusion models from scratch (\emph{a.k.a.} shortcut models). However, their theoretical derivation and practical implementation are often closely coupled, which obscures the design space. To address this, we propose a common design framework for representative shortcut models. This framework provides theoretical justification for their validity and disentangles concrete component-level choices, thereby enabling systematic identification of improvements. With our proposed improvements, the resulting one-step model achieves a new state-of-the-art FID50k of 2.85 on ImageNet-256x256 under the classifier-free guidance setting with one step generation, and further reaches FID50k of 2.52 with 2x training steps. Remarkably, the model requires no pre-training, distillation, or curriculum learning. We believe our work lowers the barrier to component-level innovation in shortcut models and facilitates principled exploration of their design space.
>
---
#### [replaced 025] VIBE: Can a VLM Read the Room?
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.11162v2](https://arxiv.org/pdf/2506.11162v2)**

> **作者:** Tania Chakraborty; Eylon Caplan; Dan Goldwasser
>
> **备注:** Findings of EMNLP, 2025
>
> **摘要:** Understanding human social behavior such as recognizing emotions and the social dynamics causing them is an important and challenging problem. While LLMs have made remarkable advances, they are limited to the textual domain and cannot account for the major role that non-verbal cues play in understanding social situations. Vision Language Models (VLMs) can potentially account for this gap, however their ability to make correct inferences over such social cues has received little attention. In this paper, we explore the capabilities of VLMs at social reasoning. We identify a previously overlooked limitation in VLMs: the Visual Social-Pragmatic Inference gap. To target this gap, we propose a new task for VLMs: Visual Social-Pragmatic Inference. We construct a high quality dataset to test the abilities of a VLM for this task and benchmark the performance of several VLMs on it.
>
---
#### [replaced 026] Video Generation with Stable Transparency via Shiftable RGB-A Distribution Learner
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.24979v3](https://arxiv.org/pdf/2509.24979v3)**

> **作者:** Haotian Dong; Wenjing Wang; Chen Li; Jing Lyu; Di Lin
>
> **摘要:** Generating RGB-A videos, which include alpha channels for transparency, has wide applications. However, current methods often suffer from low quality due to confusion between RGB and alpha. In this paper, we address this problem by learning shiftable RGB-A distributions. We adjust both the latent space and noise space, shifting the alpha distribution outward while preserving the RGB distribution, thereby enabling stable transparency generation without compromising RGB quality. Specifically, for the latent space, we propose a transparency-aware bidirectional diffusion loss during VAE training, which shifts the RGB-A distribution according to likelihood. For the noise space, we propose shifting the mean of diffusion noise sampling and applying a Gaussian ellipse mask to provide transparency guidance and controllability. Additionally, we construct a high-quality RGB-A video dataset. Compared to state-of-the-art methods, our model excels in visual quality, naturalness, transparency rendering, inference convenience, and controllability. The released model is available on our website: https://donghaotian123.github.io/Wan-Alpha/.
>
---
#### [replaced 027] Seedance 1.5 pro: A Native Audio-Visual Joint Generation Foundation Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13507v2](https://arxiv.org/pdf/2512.13507v2)**

> **作者:** Heyi Chen; Siyan Chen; Xin Chen; Yanfei Chen; Ying Chen; Zhuo Chen; Feng Cheng; Tianheng Cheng; Xinqi Cheng; Xuyan Chi; Jian Cong; Jing Cui; Qinpeng Cui; Qide Dong; Junliang Fan; Jing Fang; Zetao Fang; Chengjian Feng; Han Feng; Mingyuan Gao; Yu Gao; Dong Guo; Qiushan Guo; Boyang Hao; Qingkai Hao; Bibo He; Qian He; Tuyen Hoang; Ruoqing Hu; Xi Hu; Weilin Huang; Zhaoyang Huang; Zhongyi Huang; Donglei Ji; Siqi Jiang; Wei Jiang; Yunpu Jiang; Zhuo Jiang; Ashley Kim; Jianan Kong; Zhichao Lai; Shanshan Lao; Yichong Leng; Ai Li; Feiya Li; Gen Li; Huixia Li; JiaShi Li; Liang Li; Ming Li; Shanshan Li; Tao Li; Xian Li; Xiaojie Li; Xiaoyang Li; Xingxing Li; Yameng Li; Yifu Li; Yiying Li; Chao Liang; Han Liang; Jianzhong Liang; Ying Liang; Zhiqiang Liang; Wang Liao; Yalin Liao; Heng Lin; Kengyu Lin; Shanchuan Lin; Xi Lin; Zhijie Lin; Feng Ling; Fangfang Liu; Gaohong Liu; Jiawei Liu; Jie Liu; Jihao Liu; Shouda Liu; Shu Liu; Sichao Liu; Songwei Liu; Xin Liu; Xue Liu; Yibo Liu; Zikun Liu; Zuxi Liu; Junlin Lyu; Lecheng Lyu; Qian Lyu; Han Mu; Xiaonan Nie; Jingzhe Ning; Xitong Pan; Yanghua Peng; Lianke Qin; Xueqiong Qu; Yuxi Ren; Kai Shen; Guang Shi; Lei Shi; Yan Song; Yinglong Song; Fan Sun; Li Sun; Renfei Sun; Yan Sun; Zeyu Sun; Wenjing Tang; Yaxue Tang; Zirui Tao; Feng Wang; Furui Wang; Jinran Wang; Junkai Wang; Ke Wang; Kexin Wang; Qingyi Wang; Rui Wang; Sen Wang; Shuai Wang; Tingru Wang; Weichen Wang; Xin Wang; Yanhui Wang; Yue Wang; Yuping Wang; Yuxuan Wang; Ziyu Wang; Guoqiang Wei; Wanru Wei; Di Wu; Guohong Wu; Hanjie Wu; Jian Wu; Jie Wu; Ruolan Wu; Xinglong Wu; Yonghui Wu; Ruiqi Xia; Liang Xiang; Fei Xiao; XueFeng Xiao; Pan Xie; Shuangyi Xie; Shuang Xu; Jinlan Xue; Shen Yan; Bangbang Yang; Ceyuan Yang; Jiaqi Yang; Runkai Yang; Tao Yang; Yang Yang; Yihang Yang; ZhiXian Yang; Ziyan Yang; Songting Yao; Yifan Yao; Zilyu Ye; Bowen Yu; Jian Yu; Chujie Yuan; Linxiao Yuan; Sichun Zeng; Weihong Zeng; Xuejiao Zeng; Yan Zeng; Chuntao Zhang; Heng Zhang; Jingjie Zhang; Kuo Zhang; Liang Zhang; Liying Zhang; Manlin Zhang; Ting Zhang; Weida Zhang; Xiaohe Zhang; Xinyan Zhang; Yan Zhang; Yuan Zhang; Zixiang Zhang; Fengxuan Zhao; Huating Zhao; Yang Zhao; Hao Zheng; Jianbin Zheng; Xiaozheng Zheng; Yangyang Zheng; Yijie Zheng; Jiexin Zhou; Jiahui Zhu; Kuan Zhu; Shenhan Zhu; Wenjia Zhu; Benhui Zou; Feilong Zuo
>
> **备注:** Seedance 1.5 pro Technical Report
>
> **摘要:** Recent strides in video generation have paved the way for unified audio-visual generation. In this work, we present Seedance 1.5 pro, a foundational model engineered specifically for native, joint audio-video generation. Leveraging a dual-branch Diffusion Transformer architecture, the model integrates a cross-modal joint module with a specialized multi-stage data pipeline, achieving exceptional audio-visual synchronization and superior generation quality. To ensure practical utility, we implement meticulous post-training optimizations, including Supervised Fine-Tuning (SFT) on high-quality datasets and Reinforcement Learning from Human Feedback (RLHF) with multi-dimensional reward models. Furthermore, we introduce an acceleration framework that boosts inference speed by over 10X. Seedance 1.5 pro distinguishes itself through precise multilingual and dialect lip-syncing, dynamic cinematic camera control, and enhanced narrative coherence, positioning it as a robust engine for professional-grade content creation. Seedance 1.5 pro is now accessible on Volcano Engine at https://console.volcengine.com/ark/region:ark+cn-beijing/experience/vision?type=GenVideo.
>
---
#### [replaced 028] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.07345v2](https://arxiv.org/pdf/2512.07345v2)**

> **作者:** Shilong Jin; Haoran Duan; Litao Hua; Wentao Huang; Yuan Zhou
>
> **备注:** Accepted by AAAI 2026, Code is available at: https://github.com/kimslong/AAAI26-TDAttn
>
> **摘要:** Versatile 3D tasks (e.g., generation or editing) that distill from Text-to-Image (T2I) diffusion models have attracted significant research interest for not relying on extensive 3D training data. However, T2I models exhibit limitations resulting from prior view bias, which produces conflicting appearances between different views of an object. This bias causes subject-words to preferentially activate prior view features during cross-attention (CA) computation, regardless of the target view condition. To overcome this limitation, we conduct a comprehensive mathematical analysis to reveal the root cause of the prior view bias in T2I models. Moreover, we find different UNet layers show different effects of prior view in CA. Therefore, we propose a novel framework, TD-Attn, which addresses multi-view inconsistency via two key components: (1) the 3D-Aware Attention Guidance Module (3D-AAG) constructs a view-consistent 3D attention Gaussian for subject-words to enforce spatial consistency across attention-focused regions, thereby compensating for the limited spatial information in 2D individual view CA maps; (2) the Hierarchical Attention Modulation Module (HAM) utilizes a Semantic Guidance Tree (SGT) to direct the Semantic Response Profiler (SRP) in localizing and modulating CA layers that are highly responsive to view conditions, where the enhanced CA maps further support the construction of more consistent 3D attention Gaussians. Notably, HAM facilitates semantic-specific interventions, enabling controllable and precise 3D editing. Extensive experiments firmly establish that TD-Attn has the potential to serve as a universal plugin, significantly enhancing multi-view consistency across 3D tasks.
>
---
#### [replaced 029] CausalCLIP: Causally-Informed Feature Disentanglement and Filtering for Generalizable Detection of Generated Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13285v2](https://arxiv.org/pdf/2512.13285v2)**

> **作者:** Bo Liu; Qiao Qin; Qinghui He
>
> **备注:** 9 pages,Accepted to AAAI 2026
>
> **摘要:** The rapid advancement of generative models has increased the demand for generated image detectors capable of generalizing across diverse and evolving generation techniques. However, existing methods, including those leveraging pre-trained vision-language models, often produce highly entangled representations, mixing task-relevant forensic cues (causal features) with spurious or irrelevant patterns (non-causal features), thus limiting generalization. To address this issue, we propose CausalCLIP, a framework that explicitly disentangles causal from non-causal features and employs targeted filtering guided by causal inference principles to retain only the most transferable and discriminative forensic cues. By modeling the generation process with a structural causal model and enforcing statistical independence through Gumbel-Softmax-based feature masking and Hilbert-Schmidt Independence Criterion (HSIC) constraints, CausalCLIP isolates stable causal features robust to distribution shifts. When tested on unseen generative models from different series, CausalCLIP demonstrates strong generalization ability, achieving improvements of 6.83% in accuracy and 4.06% in average precision over state-of-the-art methods.
>
---
#### [replaced 030] A Unified Framework with Multimodal Fine-tuning for Remote Sensing Semantic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.11160v2](https://arxiv.org/pdf/2410.11160v2)**

> **作者:** Xianping Ma; Xiaokang Zhang; Man-On Pun; Bo Huang
>
> **备注:** 15 pages, 11 figures
>
> **摘要:** Multimodal remote sensing data, acquired from diverse sensors, offer a comprehensive and integrated perspective of the Earth's surface. Leveraging multimodal fusion techniques, semantic segmentation enables detailed and accurate analysis of geographic scenes, surpassing single-modality approaches. Building on advancements in vision foundation models, particularly the Segment Anything Model (SAM), this study proposes a unified framework incorporating a novel Multimodal Fine-tuning Network (MFNet) for remote sensing semantic segmentation. The proposed framework is designed to seamlessly integrate with various fine-tuning mechanisms, demonstrated through the inclusion of Adapter and Low-Rank Adaptation (LoRA) as representative examples. This extensibility ensures the framework's adaptability to other emerging fine-tuning strategies, allowing models to retain SAM's general knowledge while effectively leveraging multimodal data. Additionally, a pyramid-based Deep Fusion Module (DFM) is introduced to integrate high-level geographic features across multiple scales, enhancing feature representation prior to decoding. This work also highlights SAM's robust generalization capabilities with Digital Surface Model (DSM) data, a novel application. Extensive experiments on three benchmark multimodal remote sensing datasets, ISPRS Vaihingen, ISPRS Potsdam and MMHunan, demonstrate that the proposed MFNet significantly outperforms existing methods in multimodal semantic segmentation, setting a new standard in the field while offering a versatile foundation for future research and applications. The source code for this work is accessible at https://github.com/sstary/SSRS.
>
---
#### [replaced 031] Multimodal Deep Learning for Stroke Prediction and Detection using Retinal Imaging and Clinical Data
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.02677v2](https://arxiv.org/pdf/2505.02677v2)**

> **作者:** Saeed Shurrab; Aadim Nepal; Terrence J. Lee-St. John; Nicola G. Ghazi; Bartlomiej Piechowski-Jozwiak; Farah E. Shamout
>
> **摘要:** Stroke is a major public health problem, affecting millions worldwide. Deep learning has recently demonstrated promise for enhancing the diagnosis and risk prediction of stroke. However, existing methods rely on costly medical imaging modalities, such as computed tomography. Recent studies suggest that retinal imaging could offer a cost-effective alternative for cerebrovascular health assessment due to the shared clinical pathways between the retina and the brain. Hence, this study explores the impact of leveraging retinal images and clinical data for stroke detection and risk prediction. We propose a multimodal deep neural network that processes Optical Coherence Tomography (OCT) and infrared reflectance retinal scans, combined with clinical data, such as demographics, vital signs, and diagnosis codes. We pretrained our model using a self-supervised learning framework using a real-world dataset consisting of $37$ k scans, and then fine-tuned and evaluated the model using a smaller labeled subset. Our empirical findings establish the predictive ability of the considered modalities in detecting lasting effects in the retina associated with acute stroke and forecasting future risk within a specific time horizon. The experimental results demonstrate the effectiveness of our proposed framework by achieving $5$\% AUROC improvement as compared to the unimodal image-only baseline, and $8$\% improvement compared to an existing state-of-the-art foundation model. In conclusion, our study highlights the potential of retinal imaging in identifying high-risk patients and improving long-term outcomes.
>
---
#### [replaced 032] Renal Cell Carcinoma subtyping: learning from multi-resolution localization
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.09471v2](https://arxiv.org/pdf/2411.09471v2)**

> **作者:** Mohamad Mohamad; Francesco Ponzio; Santa Di Cataldo; Damien Ambrosetti; Xavier Descombes
>
> **摘要:** Renal Cell Carcinoma is typically asymptomatic at the early stages for many patients. This leads to a late diagnosis of the tumor, where the curability likelihood is lower, and makes the mortality rate of Renal Cell Carcinoma high, with respect to its incidence rate. To increase the survival chance, a fast and correct categorization of the tumor subtype is paramount. Nowadays, computerized methods, based on artificial intelligence, represent an interesting opportunity to improve the productivity and the objectivity of the microscopy-based Renal Cell Carcinoma diagnosis. Nonetheless, much of their exploitation is hampered by the paucity of annotated dataset, essential for a proficient training of supervised machine learning technologies. This study sets out to investigate a novel self supervised training strategy for machine learning diagnostic tools, based on the multi-resolution nature of the histological samples. We aim at reducing the need of annotated dataset, without significantly reducing the accuracy of the tool. We demonstrate the classification capability of our tool on a whole slide imaging dataset for Renal Cancer subtyping, and we compare our solution with several state-of-the-art classification counterparts.
>
---
#### [replaced 033] Unsupervised Representation Learning from Sparse Transformation Analysis
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2410.05564v2](https://arxiv.org/pdf/2410.05564v2)**

> **作者:** Yue Song; Thomas Anderson Keller; Yisong Yue; Pietro Perona; Max Welling
>
> **备注:** T-PAMI journal paper
>
> **摘要:** There is a vast literature on representation learning based on principles such as coding efficiency, statistical independence, causality, controllability, or symmetry. In this paper we propose to learn representations from sequence data by factorizing the transformations of the latent variables into sparse components. Input data are first encoded as distributions of latent activations and subsequently transformed using a probability flow model, before being decoded to predict a future input state. The flow model is decomposed into a number of rotational (divergence-free) vector fields and a number of potential flow (curl-free) fields. Our sparsity prior encourages only a small number of these fields to be active at any instant and infers the speed with which the probability flows along these fields. Training this model is completely unsupervised using a standard variational objective and results in a new form of disentangled representations where the input is not only represented by a combination of independent factors, but also by a combination of independent transformation primitives given by the learned flow fields. When viewing the transformations as symmetries one may interpret this as learning approximately equivariant representations. Empirically we demonstrate that this model achieves state of the art in terms of both data likelihood and unsupervised approximate equivariance errors on datasets composed of sequence transformations.
>
---
#### [replaced 034] TARDis: Time Attenuated Representation Disentanglement for Incomplete Multi-Modal Tumor Segmentation and Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04576v2](https://arxiv.org/pdf/2512.04576v2)**

> **作者:** Zishuo Wan; Qinqin Kang; Na Li; Yi Huang; Qianru Zhang; Le Lu; Yun Bian; Dawei Ding; Ke Yan
>
> **摘要:** The accurate diagnosis and segmentation of tumors in contrast-enhanced Computed Tomography (CT) are fundamentally driven by the distinctive hemodynamic profiles of contrast agents over time. However, in real-world clinical practice, complete temporal dynamics are often hard to capture by strict radiation dose limits and inconsistent acquisition protocols across institutions, leading to a prevalent missing modality problem. Existing deep learning approaches typically treat missing phases as absent independent channels, ignoring the inherent temporal continuity of hemodynamics. In this work, we propose Time Attenuated Representation Disentanglement (TARDis), a novel physics-aware framework that redefines missing modalities as missing sample points on a continuous Time-Attenuation Curve. We first hypothesize that the latent feature can be disentangled into a time-invariant static component (anatomy) and a time-dependent dynamic component (perfusion). We achieve this via a dual-path architecture: a quantization-based path using a learnable embedding dictionary to extract consistent anatomical structures, and a probabilistic path using a Hemodynamic Conditional Variational Autoencoder to model dynamic enhancement conditioned on the estimated scan time. This design allows the network to infer missing hemodynamic features by sampling from the learned latent distribution. Extensive experiments on a large-scale multi-modal private abdominal CT dataset (2,282 patients) and two public datasets demonstrate that TARDis significantly outperforms state-of-the-art incomplete modality frameworks. Notably, our method maintains robust diagnostic performance even in extreme data-sparsity scenarios, highlighting its potential for reducing radiation exposure while maintaining diagnostic precision.
>
---
#### [replaced 035] Inter- and Intra-image Refinement for Few Shot Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.05838v2](https://arxiv.org/pdf/2507.05838v2)**

> **作者:** Ourui Fu; Hangzhou He; Kaiwen Li; Xinliang Zhang; Lei Zhu; Shuang Zeng; Zhaoheng Xie; Yanye Lu
>
> **摘要:** Deep neural networks for semantic segmentation rely on large-scale annotated datasets, leading to an annotation bottleneck that motivates few shot semantic segmentation (FSS) which aims to generalize to novel classes with minimal labeled exemplars. Most existing FSS methods adopt a prototype-based paradigm, which generates query prior map by extracting masked-area features from support images and then makes predictions guided by the prior map. However, they suffer from two critical limitations induced by inter- and intra-image discrepancies: 1) The intra-class gap between support and query images, caused by single-prototype representation, results in scattered and noisy prior maps; 2) The inter-class interference from visually similar but semantically distinct regions leads to inconsistent support-query feature matching and erroneous predictions. To address these issues, we propose the Inter- and Intra-image Refinement (IIR) model. The model contains an inter-image class activation mapping based method that generates two prototypes for class-consistent region matching, including core discriminative features and local specific features, and yields an accurate and robust prior map. For intra-image refinement, a directional dropout mechanism is introduced to mask inconsistent support-query feature pairs in cross attention, thereby enhancing decoder performance. Extensive experiments demonstrate that IIR achieves state-of-the-art performance on 9 benchmarks, covering standard FSS, part FSS, and cross-domain FSS. Our source code is available at \href{https://github.com/forypipi/IIR}{https://github.com/forypipi/IIR}.
>
---
#### [replaced 036] MindDrive: A Vision-Language-Action Model for Autonomous Driving via Online Reinforcement Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出MindDrive，属自动驾驶中的视觉-语言-动作（VLA）任务，旨在解决模仿学习导致的分布偏移与因果混淆问题。通过双LoRA微调轻量LLM，将在线强化学习从连续动作空间转为离散语言决策空间，实现高效探索与人类化驾驶行为。**

- **链接: [https://arxiv.org/pdf/2512.13636v2](https://arxiv.org/pdf/2512.13636v2)**

> **作者:** Haoyu Fu; Diankun Zhang; Zongchuang Zhao; Jianfeng Cui; Hongwei Xie; Bing Wang; Guang Chen; Dingkang Liang; Xiang Bai
>
> **备注:** 16 pages, 12 figures, 6 tables; Project Page: https://xiaomi-mlab.github.io/MindDrive/
>
> **摘要:** Current Vision-Language-Action (VLA) paradigms in autonomous driving primarily rely on Imitation Learning (IL), which introduces inherent challenges such as distribution shift and causal confusion. Online Reinforcement Learning offers a promising pathway to address these issues through trial-and-error learning. However, applying online reinforcement learning to VLA models in autonomous driving is hindered by inefficient exploration in continuous action spaces. To overcome this limitation, we propose MindDrive, a VLA framework comprising a large language model (LLM) with two distinct sets of LoRA parameters. The one LLM serves as a Decision Expert for scenario reasoning and driving decision-making, while the other acts as an Action Expert that dynamically maps linguistic decisions into feasible trajectories. By feeding trajectory-level rewards back into the reasoning space, MindDrive enables trial-and-error learning over a finite set of discrete linguistic driving decisions, instead of operating directly in a continuous action space. This approach effectively balances optimal decision-making in complex scenarios, human-like driving behavior, and efficient exploration in online reinforcement learning. Using the lightweight Qwen-0.5B LLM, MindDrive achieves Driving Score (DS) of 78.04 and Success Rate (SR) of 55.09% on the challenging Bench2Drive benchmark. To the best of our knowledge, this is the first work to demonstrate the effectiveness of online reinforcement learning for the VLA model in autonomous driving.
>
---
#### [replaced 037] Audio-Visual Camera Pose Estimation with Passive Scene Sounds and In-the-Wild Video
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.12165v2](https://arxiv.org/pdf/2512.12165v2)**

> **作者:** Daniel Adebi; Sagnik Majumder; Kristen Grauman
>
> **摘要:** Understanding camera motion is a fundamental problem in embodied perception and 3D scene understanding. While visual methods have advanced rapidly, they often struggle under visually degraded conditions such as motion blur or occlusions. In this work, we show that passive scene sounds provide complementary cues for relative camera pose estimation for in-the-wild videos. We introduce a simple but effective audio-visual framework that integrates direction-ofarrival (DOA) spectra and binauralized embeddings into a state-of-the-art vision-only pose estimation model. Our results on two large datasets show consistent gains over strong visual baselines, plus robustness when the visual information is corrupted. To our knowledge, this represents the first work to successfully leverage audio for relative camera pose estimation in real-world videos, and it establishes incidental, everyday audio as an unexpected but promising signal for a classic spatial challenge. Project: http://vision.cs.utexas.edu/projects/av_camera_pose.
>
---
#### [replaced 038] Training Multi-Image Vision Agents via End2End Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.08980v2](https://arxiv.org/pdf/2512.08980v2)**

> **作者:** Chengqi Dong; Chuhuai Yue; Hang He; Rongge Mao; Fenghe Tang; S Kevin Zhou; Zekun Xu; Xiaohan Wang; Jiajun Chai; Wei Lin; Guojun Yin
>
> **摘要:** Recent VLM-based agents aim to replicate OpenAI O3's ``thinking with images" via tool use, but most open-source methods limit input to a single image, falling short on real-world multi-image QA tasks. To address this, we propose IMAgent, an open-source vision agent trained via end-to-end reinforcement learning dedicated for complex multi-image tasks. By leveraging a multi-agent system, we generate challenging and visually-rich multi-image QA pairs to fully activate the tool-use potential of the base VLM. Through manual verification, we obtain MIFG-QA, comprising 10k samples for training and evaluation. With deeper reasoning steps, VLMs may increasingly ignore visual inputs. We therefore develop two specialized tools for visual reflection and confirmation, allowing the model to proactively reallocate its attention to image content during inference. Benefiting from our well-designed action-trajectory two-level mask strategy, IMAgent achieves stable tool use behavior via pure RL training without requiring costly supervised fine-tuning data. Extensive experiments demonstrate that IMAgent maintains strong performance on existing single-image benchmarks while achieving substantial improvements on our proposed multi-image dataset, with our analysis providing actionable insights for the research community. Codes and data will be released soon.
>
---
#### [replaced 039] Fast and Explicit: Slice-to-Volume Reconstruction via 3D Gaussian Primitives with Analytic Point Spread Function Modeling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.11624v2](https://arxiv.org/pdf/2512.11624v2)**

> **作者:** Maik Dannecker; Steven Jia; Nil Stolt-Ansó; Nadine Girard; Guillaume Auzias; François Rousseau; Daniel Rueckert
>
> **备注:** Under Review for MIDL 2026
>
> **摘要:** Recovering high-fidelity 3D images from sparse or degraded 2D images is a fundamental challenge in medical imaging, with broad applications ranging from 3D ultrasound reconstruction to MRI super-resolution. In the context of fetal MRI, high-resolution 3D reconstruction of the brain from motion-corrupted low-resolution 2D acquisitions is a prerequisite for accurate neurodevelopmental diagnosis. While implicit neural representations (INRs) have recently established state-of-the-art performance in self-supervised slice-to-volume reconstruction (SVR), they suffer from a critical computational bottleneck: accurately modeling the image acquisition physics requires expensive stochastic Monte Carlo sampling to approximate the point spread function (PSF). In this work, we propose a shift from neural network based implicit representations to Gaussian based explicit representations. By parameterizing the HR 3D image volume as a field of anisotropic Gaussian primitives, we leverage the property of Gaussians being closed under convolution and thus derive a \textit{closed-form analytical solution} for the forward model. This formulation reduces the previously intractable acquisition integral to an exact covariance addition ($\mathbfΣ_{obs} = \mathbfΣ_{HR} + \mathbfΣ_{PSF}$), effectively bypassing the need for compute-intensive stochastic sampling while ensuring exact gradient propagation. We demonstrate that our approach matches the reconstruction quality of self-supervised state-of-the-art SVR frameworks while delivering a 5$\times$--10$\times$ speed-up on neonatal and fetal data. With convergence often reached in under 30 seconds, our framework paves the way towards translation into clinical routine of real-time fetal 3D MRI. Code will be public at {https://github.com/m-dannecker/Gaussian-Primitives-for-Fast-SVR}.
>
---
#### [replaced 040] VesSAM: Efficient Multi-Prompting for Segmenting Complex Vessel
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.00981v2](https://arxiv.org/pdf/2511.00981v2)**

> **作者:** Suzhong Fu; Rui Sun; Xuan Ding; Jingqi Dong; Yiming Yang; Yao Zhu; Min Chang Jordan Ren; Delin Deng; Angelica Aviles-Rivero; Shuguang Cui; Zhen Li
>
> **摘要:** Accurate vessel segmentation is critical for clinical applications such as disease diagnosis and surgical planning, yet remains challenging due to thin, branching structures and low texture contrast. While foundation models like the Segment Anything Model (SAM) have shown promise in generic segmentation, they perform sub-optimally on vascular structures. In this work, we present VesSAM, a powerful and efficient framework tailored for 2D vessel segmentation. VesSAM integrates (1) a convolutional adapter to enhance local texture features, (2) a multi-prompt encoder that fuses anatomical prompts, including skeletons, bifurcation points, and segment midpoints, via hierarchical cross-attention, and (3) a lightweight mask decoder to reduce jagged artifacts. We also introduce an automated pipeline to generate structured multi-prompt annotations, and curate a diverse benchmark dataset spanning 8 datasets across 5 imaging modalities. Experimental results demonstrate that VesSAM consistently outperforms state-of-the-art PEFT-based SAM variants by over 10% Dice and 13% IoU, and achieves competitive performance compared to fully fine-tuned methods, with significantly fewer parameters. VesSAM also generalizes well to out-of-distribution (OoD) settings, outperforming all baselines in average OoD Dice and IoU.
>
---
#### [replaced 041] Multimodal classification of forest biodiversity potential from 2D orthophotos and 3D airborne laser scanning point clouds
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.01728v3](https://arxiv.org/pdf/2501.01728v3)**

> **作者:** Simon B. Jensen; Stefan Oehmcke; Andreas Møgelmose; Meysam Madadi; Christian Igel; Sergio Escalera; Thomas B. Moeslund
>
> **摘要:** Assessment of forest biodiversity is crucial for ecosystem management and conservation. While traditional field surveys provide high-quality assessments, they are labor-intensive and spatially limited. This study investigates whether deep learning-based fusion of close-range sensing data from 2D orthophotos and 3D airborne laser scanning (ALS) point clouds can reliable assess the biodiversity potential of forests. We introduce the BioVista dataset, comprising 44378 paired samples of orthophotos and ALS point clouds from temperate forests in Denmark, designed to explore multimodal fusion approaches. Using deep neural networks (ResNet for orthophotos and PointVector for ALS point clouds), we investigate each data modality's ability to assess forest biodiversity potential, achieving overall accuracies of 76.7% and 75.8%, respectively. We explore various 2D and 3D fusion approaches: confidence-based ensembling, feature-level concatenation, and end-to-end training, with the latter achieving an overall accuracies of 82.0% when separating low- and high potential forest areas. Our results demonstrate that spectral information from orthophotos and structural information from ALS point clouds effectively complement each other in the assessment of forest biodiversity potential.
>
---
#### [replaced 042] Dynamic Prompt Generation for Interactive 3D Medical Image Segmentation Training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.03189v3](https://arxiv.org/pdf/2510.03189v3)**

> **作者:** Tidiane Camaret Ndir; Alexander Pfefferle; Robin Tibor Schirrmeister
>
> **摘要:** Interactive 3D biomedical image segmentation requires efficient models that can iteratively refine predictions based on user prompts. Current foundation models either lack volumetric awareness or suffer from limited interactive capabilities. We propose a training strategy that combines dynamic volumetric prompt generation with content-aware adaptive cropping to optimize the use of the image encoder. Our method simulates realistic user interaction patterns during training while addressing the computational challenges of learning from sequential refinement feedback on a single GPU. For efficient training, we initialize our network using the publicly available weights from the nnInteractive segmentation model. Evaluation on the \textbf{Foundation Models for Interactive 3D Biomedical Image Segmentation} competition demonstrates strong performance with an average final Dice score of 0.6385, normalized surface distance of 0.6614, and area-under-the-curve metrics of 2.4799 (Dice) and 2.5671 (NSD).
>
---
#### [replaced 043] VideoMem: Enhancing Ultra-Long Video Understanding via Adaptive Memory Management
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04540v2](https://arxiv.org/pdf/2512.04540v2)**

> **作者:** Hongbo Jin; Qingyuan Wang; Wenhao Zhang; Yang Liu; Sijie Cheng
>
> **摘要:** Ultra long video understanding remains an open challenge, as existing vision language models (VLMs) falter on such content due to limited context length and inefficient long term memory retention. To address this, recent works have attempted to construct external knowledge bases and corresponding retrieval agumented generation (RAG) systems, yet these incur enormous storage and computational overhead. In this paper, we propose VideoMem, a novel framework that pioneers models long video understanding as a sequential generation task via adaptive memory management. Specifically, VideoMem dynamically updates a global memory buffer, which adaptively retains critical information while discarding redundant content across the video timeline. To efficiently train VLMs for such long-term tasks, VideoMem integrates the Progressive Grouped Relative Policy Optimization (PRPO) algorithm, equipped with two core modules: Progressive State Propagation (PSP) adaptively retains valid current states, propagates them to the next rollout step, and gradually narrows the model exploration space. Temporal Cascading Reward (TCR) further alleviates reward sparsity, improving sample utilization and accelerating convergence. Extensive experiments demonstrate that VideoMem significantly outperforms existing open-source models across diverse benchmarks for ultra-long video understanding tasks.
>
---
#### [replaced 044] Video Reality Test: Can AI-Generated ASMR Videos fool VLMs and Humans?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13281v2](https://arxiv.org/pdf/2512.13281v2)**

> **作者:** Jiaqi Wang; Weijia Wu; Yi Zhan; Rui Zhao; Ming Hu; James Cheng; Wei Liu; Philip Torr; Kevin Qinghong Lin
>
> **备注:** Code is at https://github.com/video-reality-test/video-reality-test, page is at https://video-reality-test.github.io/
>
> **摘要:** Recent advances in video generation have produced vivid content that are often indistinguishable from real videos, making AI-generated video detection an emerging societal challenge. Prior AIGC detection benchmarks mostly evaluate video without audio, target broad narrative domains, and focus on classification solely. Yet it remains unclear whether state-of-the-art video generation models can produce immersive, audio-paired videos that reliably deceive humans and VLMs. To this end, we introduce Video Reality Test, an ASMR-sourced video benchmark suite for testing perceptual realism under tight audio-visual coupling, featuring the following dimensions: \textbf{(i) Immersive ASMR video-audio sources.} Built on carefully curated real ASMR videos, the benchmark targets fine-grained action-object interactions with diversity across objects, actions, and backgrounds. \textbf{(ii) Peer-Review evaluation.} An adversarial creator-reviewer protocol where video generation models act as creators aiming to fool reviewers, while VLMs serve as reviewers seeking to identify fakeness. Our experimental findings show: The best creator Veo3.1-Fast even fools most VLMs: the strongest reviewer (Gemini 2.5-Pro) achieves only 56\% accuracy (random 50\%), far below that of human experts (81.25\%). Adding audio improves real-fake discrimination, yet superficial cues such as watermarks can still significantly mislead models. These findings delineate the current boundary of video generation realism and expose limitations of VLMs in perceptual fidelity and audio-visual consistency. Our code is available at https://github.com/video-reality-test/video-reality-test.
>
---
#### [replaced 045] LSM: A Comprehensive Metric for Assessing the Safety of Lane Detection Systems in Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2407.07740v2](https://arxiv.org/pdf/2407.07740v2)**

> **作者:** Jörg Gamerdinger; Sven Teufel; Stephan Amann; Georg Volk; Oliver Bringmann
>
> **备注:** Accepted at IEEE VTC-Fall 2025
>
> **摘要:** Comprehensive perception of the vehicle's environment and correct interpretation of the environment are crucial for the safe operation of autonomous vehicles. The perception of surrounding objects is the main component for further tasks such as trajectory planning. However, safe trajectory planning requires not only object detection, but also the detection of drivable areas and lane corridors. While first approaches consider an advanced safety evaluation of object detection, the evaluation of lane detection still lacks sufficient safety metrics. Similar to the safety metrics for object detection, additional factors such as the semantics of the scene with road type and road width, the detection range as well as the potential causes of missing detections, incorporated by vehicle speed, should be considered for the evaluation of lane detection. Therefore, we propose the Lane Safety Metric (LSM), which takes these factors into account and allows to evaluate the safety of lane detection systems by determining an easily interpretable safety score. We evaluate our offline safety metric on various virtual scenarios using different lane detection approaches and compare it with state-of-the-art performance metrics.
>
---
#### [replaced 046] MFGDiffusion: Mask-Guided Smoke Synthesis for Enhanced Forest Fire Detection
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2507.11252v2](https://arxiv.org/pdf/2507.11252v2)**

> **作者:** Guanghao Wu; Yunqing Shang; Chen Xu; Hai Song; Chong Wang; Qixing Zhang
>
> **备注:** 14 pages, 11 figures
>
> **摘要:** Smoke is the first visible indicator of a wildfire.With the advancement of deep learning, image-based smoke detection has become a crucial method for detecting and preventing forest fires. However, the scarcity of smoke image data from forest fires is one of the significant factors hindering the detection of forest fire smoke. Image generation models offer a promising solution for synthesizing realistic smoke images. However, current inpainting models exhibit limitations in generating high-quality smoke representations, particularly manifesting as inconsistencies between synthesized smoke and background contexts. To solve these problems, we proposed a comprehensive framework for generating forest fire smoke images. Firstly, we employed the pre-trained segmentation model and the multimodal model to obtain smoke masks and image captions.Then, to address the insufficient utilization of masks and masked images by inpainting models, we introduced a network architecture guided by mask and masked image features. We also proposed a new loss function, the mask random difference loss, which enhances the consistency of the generated effects around the mask by randomly expanding and eroding the mask edges.Finally, to generate a smoke image dataset using random masks for subsequent detection tasks, we incorporated smoke characteristics and use a multimodal large language model as a filtering tool to select diverse and reasonable smoke images, thereby improving the quality of the synthetic dataset. Experiments showed that our generated smoke images are realistic and diverse, and effectively enhance the performance of forest fire smoke detection models. Code is available at https://github.com/wghr123/MFGDiffusion.
>
---
#### [replaced 047] OpenVE-3M: A Large-Scale High-Quality Dataset for Instruction-Guided Video Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.07826v2](https://arxiv.org/pdf/2512.07826v2)**

> **作者:** Haoyang He; Jie Wang; Jiangning Zhang; Zhucun Xue; Xingyuan Bu; Qiangpeng Yang; Shilei Wen; Lei Xie
>
> **备注:** Project page: https://lewandofskee.github.io/projects/OpenVE
>
> **摘要:** The quality and diversity of instruction-based image editing datasets are continuously increasing, yet large-scale, high-quality datasets for instruction-based video editing remain scarce. To address this gap, we introduce OpenVE-3M, an open-source, large-scale, and high-quality dataset for instruction-based video editing. It comprises two primary categories: spatially-aligned edits (Global Style, Background Change, Local Change, Local Remove, Local Add, and Subtitles Edit) and non-spatially-aligned edits (Camera Multi-Shot Edit and Creative Edit). All edit types are generated via a meticulously designed data pipeline with rigorous quality filtering. OpenVE-3M surpasses existing open-source datasets in terms of scale, diversity of edit types, instruction length, and overall quality. Furthermore, to address the lack of a unified benchmark in the field, we construct OpenVE-Bench, containing 431 video-edit pairs that cover a diverse range of editing tasks with three key metrics highly aligned with human judgment. We present OpenVE-Edit, a 5B model trained on our dataset that demonstrates remarkable efficiency and effectiveness by setting a new state-of-the-art on OpenVE-Bench, outperforming all prior open-source models including a 14B baseline. Project page is at https://lewandofskee.github.io/projects/OpenVE.
>
---
#### [replaced 048] Wi-CBR: Salient-aware Adaptive WiFi Sensing for Cross-domain Behavior Recognition
- **分类: cs.CV; eess.SP**

- **链接: [https://arxiv.org/pdf/2506.11616v4](https://arxiv.org/pdf/2506.11616v4)**

> **作者:** Ruobei Zhang; Shengeng Tang; Huan Yan; Xiang Zhang; Jiabao Guo
>
> **摘要:** The challenge in WiFi-based cross-domain Behavior Recognition lies in the significant interference of domain-specific signals on gesture variation. However, previous methods alleviate this interference by mapping the phase from multiple domains into a common feature space. If the Doppler Frequency Shift (DFS) signal is used to dynamically supplement the phase features to achieve better generalization, it enables the model to not only explore a wider feature space but also to avoid potential degradation of gesture semantic information. Specifically, we propose a novel Salient-aware Adaptive WiFi Sensing for Cross-domain Behavior Recognition (Wi-CBR), which constructs a dual-branch self-attention module that captures temporal features from phase information reflecting dynamic path length variations while extracting kinematic features from DFS correlated with motion velocity. Moreover, we design a Saliency Guidance Module that employs group attention mechanisms to mine critical activity features and utilizes gating mechanisms to optimize information entropy, facilitating feature fusion and enabling effective interaction between salient and non-salient behavioral characteristics. Extensive experiments on two large-scale public datasets (Widar3.0 and XRF55) demonstrate the superior performance of our method in both in-domain and cross-domain scenarios.
>
---
#### [replaced 049] Assessing Greenspace Attractiveness with ChatGPT, Claude, and Gemini: Do AI Models Reflect Human Perceptions?
- **分类: cs.CY; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.11827v2](https://arxiv.org/pdf/2512.11827v2)**

> **作者:** Milad Malekzadeh; Magdalena Biernacka; Elias Willberg; Jussi Torkko; Edyta Łaszkiewicz; Tuuli Toivonen
>
> **摘要:** Understanding greenspace attractiveness is essential for designing livable and inclusive urban environments, yet existing assessment approaches often overlook informal or transient spaces and remain too resource intensive to capture subjective perceptions at scale. This study examines the ability of multimodal large language models (MLLMs), ChatGPT GPT-4o, Claude 3.5 Haiku, and Gemini 2.0 Flash, to assess greenspace attractiveness similarly to humans using Google Street View imagery. We compared model outputs with responses from a geo-questionnaire of residents in Lodz, Poland, across both formal (for example, parks and managed greenspaces) and informal (for example, meadows and wastelands) greenspaces. Survey respondents and models indicated whether each greenspace was attractive or unattractive and provided up to three free text explanations. Analyses examined how often their attractiveness judgments aligned and compared their explanations after classifying them into shared reasoning categories. Results show high AI human agreement for attractive formal greenspaces and unattractive informal spaces, but low alignment for attractive informal and unattractive formal greenspaces. Models consistently emphasized aesthetic and design oriented features, underrepresenting safety, functional infrastructure, and locally embedded qualities valued by survey respondents. While these findings highlight the potential for scalable pre-assessment, they also underscore the need for human oversight and complementary participatory approaches. We conclude that MLLMs can support, but not replace, context sensitive greenspace evaluation in planning practice.
>
---
#### [replaced 050] A Semantically Enhanced Generative Foundation Model Improves Pathological Image Synthesis
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.13164v2](https://arxiv.org/pdf/2512.13164v2)**

> **作者:** Xianchao Guan; Zhiyuan Fan; Yifeng Wang; Fuqiang Chen; Yanjiang Zhou; Zengyang Che; Hongxue Meng; Xin Li; Yaowei Wang; Hongpeng Wang; Min Zhang; Heng Tao Shen; Zheng Zhang; Yongbing Zhang
>
> **备注:** 68 pages, 9 figures, 16 tables
>
> **摘要:** The development of clinical-grade artificial intelligence in pathology is limited by the scarcity of diverse, high-quality annotated datasets. Generative models offer a potential solution but suffer from semantic instability and morphological hallucinations that compromise diagnostic reliability. To address this challenge, we introduce a Correlation-Regulated Alignment Framework for Tissue Synthesis (CRAFTS), the first generative foundation model for pathology-specific text-to-image synthesis. By leveraging a dual-stage training strategy on approximately 2.8 million image-caption pairs, CRAFTS incorporates a novel alignment mechanism that suppresses semantic drift to ensure biological accuracy. This model generates diverse pathological images spanning 30 cancer types, with quality rigorously validated by objective metrics and pathologist evaluations. Furthermore, CRAFTS-augmented datasets enhance the performance across various clinical tasks, including classification, cross-modal retrieval, self-supervised learning, and visual question answering. In addition, coupling CRAFTS with ControlNet enables precise control over tissue architecture from inputs such as nuclear segmentation masks and fluorescence images. By overcoming the critical barriers of data scarcity and privacy concerns, CRAFTS provides a limitless source of diverse, annotated histology data, effectively unlocking the creation of robust diagnostic tools for rare and complex cancer phenotypes.
>
---
#### [replaced 051] Adaptive Detector-Verifier Framework for Zero-Shot Polyp Detection in Open-World Settings
- **分类: cs.CV; cs.CL**

- **简介: 该论文面向零-shot开放世界息肉检测任务，解决临床内镜图像因光照变化、运动模糊等导致的域偏移与漏检问题。提出自适应检测-验证框架：YOLOv11检测器结合VLM验证器，通过VLM指导动态调阈值，并用成本敏感的GRPO强化学习微调验证器，显著提升召回率，减少漏检。**

- **链接: [https://arxiv.org/pdf/2512.12492v2](https://arxiv.org/pdf/2512.12492v2)**

> **作者:** Shengkai Xu; Hsiang Lun Kao; Tianxiang Xu; Honghui Zhang; Junqiao Wang; Runmeng Ding; Guanyu Liu; Tianyu Shi; Zhenyu Yu; Guofeng Pan; Ziqian Bi; Yuqi Ouyang
>
> **摘要:** Polyp detectors trained on clean datasets often underperform in real-world endoscopy, where illumination changes, motion blur, and occlusions degrade image quality. Existing approaches struggle with the domain gap between controlled laboratory conditions and clinical practice, where adverse imaging conditions are prevalent. In this work, we propose AdaptiveDetector, a novel two-stage detector-verifier framework comprising a YOLOv11 detector with a vision-language model (VLM) verifier. The detector adaptively adjusts per-frame confidence thresholds under VLM guidance, while the verifier is fine-tuned with Group Relative Policy Optimization (GRPO) using an asymmetric, cost-sensitive reward function specifically designed to discourage missed detections -- a critical clinical requirement. To enable realistic assessment under challenging conditions, we construct a comprehensive synthetic testbed by systematically degrading clean datasets with adverse conditions commonly encountered in clinical practice, providing a rigorous benchmark for zero-shot evaluation. Extensive zero-shot evaluation on synthetically degraded CVC-ClinicDB and Kvasir-SEG images demonstrates that our approach improves recall by 14 to 22 percentage points over YOLO alone, while precision remains within 0.7 points below to 1.7 points above the baseline. This combination of adaptive thresholding and cost-sensitive reinforcement learning achieves clinically aligned, open-world polyp detection with substantially fewer false negatives, thereby reducing the risk of missed precancerous polyps and improving patient outcomes.
>
---
#### [replaced 052] Recent Advances in Multi-Agent Human Trajectory Prediction: A Comprehensive Review
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文是一篇综述，聚焦多智能体人类轨迹预测任务，旨在解决行人交互建模难题。它系统梳理了2020–2025年基于深度学习的最新方法，按架构、输入表示和预测策略分类，重点分析ETH/UCY基准上的模型，并指出关键挑战与未来方向。**

- **链接: [https://arxiv.org/pdf/2506.14831v2](https://arxiv.org/pdf/2506.14831v2)**

> **作者:** Céline Finet; Stephane Da Silva Martins; Jean-Bernard Hayet; Ioannis Karamouzas; Javad Amirian; Sylvie Le Hégarat-Mascle; Julien Pettré; Emanuel Aldea
>
> **备注:** 45 pages
>
> **摘要:** With the emergence of powerful data-driven methods in human trajectory prediction (HTP), gaining a finer understanding of multi-agent interactions lies within hand's reach, with important implications in areas such as social robot navigation, autonomous navigation, and crowd modeling. This survey reviews some of the most recent advancements in deep learning-based multi-agent trajectory prediction, focusing on studies published between 2020 and 2025. We categorize the existing methods based on their architectural design, their input representations, and their overall prediction strategies, placing a particular emphasis on models evaluated using the ETH/UCY benchmark. Furthermore, we highlight key challenges and future research directions in the field of multi-agent HTP.
>
---
#### [replaced 053] Physically Grounded Monocular Depth via Nanophotonic Wavefront Prompting
- **分类: physics.optics; cs.AR; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.15770v2](https://arxiv.org/pdf/2503.15770v2)**

> **作者:** Bingxuan Li; Jiahao Wu; Yuan Xu; Zezheng Zhu; Yunxiang Zhang; Kenneth Chen; Yanqi Liang; Nanfang Yu; Qi Sun
>
> **摘要:** Depth foundation models offer strong learned priors for 3D perception but lack physical depth cues, leading to ambiguities in metric scale. We introduce a birefringent metalens -- a planar nanophotonic lens composed of subwavelength pixels for wavefront shaping with a thickness of 700 nm and a diameter of 3 mm -- to physically prompt depth foundation models. In a single monocular shot, our metalens physically embeds depth information into two polarized optical wavefronts, which we decode through a lightweight prompting and fine-tuning framework that aligns depth foundation models with the optical signals. To scale the training data, we develop a light wave propagation simulator that synthesizes metalens responses from RGB-D datasets, incorporating key physical factors to minimize the sim-to-real gap. Simulated and physical experiments with our fabricated titanium-dioxide metalens demonstrate accurate and consistent metric depth over state-of-the-art monocular depth estimators. The research demonstrates that nanophotonic wavefront formation offers a promising bridge for grounding depth foundation models in physical depth sensing.
>
---
#### [replaced 054] Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.01510v2](https://arxiv.org/pdf/2511.01510v2)**

> **作者:** Derong Kong; Zhixiong Yang; Shengxi Li; Shuaifeng Zhi; Li Liu; Zhen Liu; Jingyuan Xia
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Low-light image enhancement (LLIE) faces persistent challenges in balancing reconstruction fidelity with cross-scenario generalization. While existing methods predominantly focus on deterministic pixel-level mappings between paired low/normal-light images, they often neglect the continuous physical process of luminance transitions in real-world environments, leading to performance drop when normal-light references are unavailable. Inspired by empirical analysis of natural luminance dynamics revealing power-law distributed intensity transitions, this paper introduces Luminance-Aware Statistical Quantification (LASQ), a novel framework that reformulates LLIE as a statistical sampling process over hierarchical luminance distributions. Our LASQ re-conceptualizes luminance transition as a power-law distribution in intensity coordinate space that can be approximated by stratified power functions, therefore, replacing deterministic mappings with probabilistic sampling over continuous luminance layers. A diffusion forward process is designed to autonomously discover optimal transition paths between luminance layers, achieving unsupervised distribution emulation without normal-light references. In this way, it considerably improves the performance in practical situations, enabling more adaptable and versatile light restoration. This framework is also readily applicable to cases with normal-light references, where it achieves superior performance on domain-specific datasets alongside better generalization-ability across non-reference datasets.
>
---
#### [replaced 055] Med3DVLM: An Efficient Vision-Language Model for 3D Medical Image Analysis
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2503.20047v3](https://arxiv.org/pdf/2503.20047v3)**

> **作者:** Yu Xin; Gorkem Can Ates; Kuang Gong; Wei Shao
>
> **摘要:** Vision-language models (VLMs) have shown promise in 2D medical image analysis, but extending them to 3D remains challenging due to the high computational demands of volumetric data and the difficulty of aligning 3D spatial features with clinical text. We present Med3DVLM, a 3D VLM designed to address these challenges through three key innovations: (1) DCFormer, an efficient encoder that uses decomposed 3D convolutions to capture fine-grained spatial features at scale; (2) SigLIP, a contrastive learning strategy with pairwise sigmoid loss that improves image-text alignment without relying on large negative batches; and (3) a dual-stream MLP-Mixer projector that fuses low- and high-level image features with text embeddings for richer multi-modal representations. We evaluate our model on the M3D dataset, which includes radiology reports and VQA data for 120,084 3D medical images. Results show that Med3DVLM achieves superior performance across multiple benchmarks. For image-text retrieval, it reaches 61.00% R@1 on 2,000 samples, significantly outperforming the current state-of-the-art M3D model (19.10%). For report generation, it achieves a METEOR score of 36.42% (vs. 14.38%). In open-ended visual question answering (VQA), it scores 36.76% METEOR (vs. 33.58%), and in closed-ended VQA, it achieves 79.95% accuracy (vs. 75.78%). These results highlight Med3DVLM's ability to bridge the gap between 3D imaging and language, enabling scalable, multi-task reasoning across clinical applications. Our code is publicly available at https://github.com/mirthAI/Med3DVLM.
>
---
#### [replaced 056] GASPACHO: Gaussian Splatting for Controllable Humans and Objects
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.09342v2](https://arxiv.org/pdf/2503.09342v2)**

> **作者:** Aymen Mir; Arthur Moreau; Helisa Dhamo; Zhensong Zhang; Gerard Pons-Moll; Eduardo Pérez-Pellitero
>
> **备注:** Project Page: https://miraymen.github.io/gaspacho/
>
> **摘要:** We present GASPACHO, a method for generating photorealistic, controllable renderings of human-object interactions from multi-view RGB video. Unlike prior work that reconstructs only the human and treats objects as background, GASPACHO simultaneously recovers animatable templates for both the human and the interacting object as distinct sets of Gaussians, thereby allowing for controllable renderings of novel human object interactions in different poses from novel-camera viewpoints. We introduce a novel formulation that learns object Gaussians on an underlying 2D surface manifold rather than in 3D volume, yielding sharper, fine-grained object details for dynamic object reconstruction. We further propose a contact constraint in Gaussian space that regularizes human-object relations and enables natural, physically plausible animation. Across three benchmarks - BEHAVE, NeuralDome, and DNA-Rendering - GASPACHO achieves high-quality reconstructions under heavy occlusion and supports controllable synthesis of novel human-object interactions. We also demonstrate that our method allows for composition of humans and objects in 3D scenes and for the first time showcase that neural rendering can be used for the controllable generation of photoreal humans interacting with dynamic objects in diverse scenes. Our results are available at: https://miraymen.github.io/gaspacho/
>
---
#### [replaced 057] Order Matters: 3D Shape Generation from Sequential VR Sketches
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04761v2](https://arxiv.org/pdf/2512.04761v2)**

> **作者:** Yizi Chen; Sidi Wu; Tianyi Xiao; Nina Wiedemann; Loic Landrieu
>
> **摘要:** VR sketching lets users explore and iterate on ideas directly in 3D, offering a faster and more intuitive alternative to conventional CAD tools. However, existing sketch-to-shape models ignore the temporal ordering of strokes, discarding crucial cues about structure and design intent. We introduce VRSketch2Shape, the first framework and multi-category dataset for generating 3D shapes from sequential VR sketches. Our contributions are threefold: (i) an automated pipeline that generates sequential VR sketches from arbitrary shapes, (ii) a dataset of over 20k synthetic and 900 hand-drawn sketch-shape pairs across four categories, and (iii) an order-aware sketch encoder coupled with a diffusion-based 3D generator. Our approach yields higher geometric fidelity than prior work, generalizes effectively from synthetic to real sketches with minimal supervision, and performs well even on partial sketches. All data and models will be released open-source at https://chenyizi086.github.io/VRSketch2Shape_website.
>
---
#### [replaced 058] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.15208v2](https://arxiv.org/pdf/2505.15208v2)**

> **作者:** Wenjie Liu; Zhongliang Liu; Junwei Shu; Changbo Wang; Yang Li
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Transferring 2D textures to 3D modalities is of great significance for improving the efficiency of multimedia content creation. Existing approaches have rarely focused on transferring image textures onto 3D representations. 3D style transfer methods are capable of transferring abstract artistic styles to 3D scenes. However, these methods often overlook the geometric information of the scene, which makes it challenging to achieve high-quality 3D texture transfer results. In this paper, we present GT^2-GS, a geometry-aware texture transfer framework for gaussian splitting. From the perspective of matching texture features with geometric information in rendered views, we identify the issue of insufficient texture features and propose a geometry-aware texture augmentation module to expand the texture feature set. Moreover, a geometry-consistent texture loss is proposed to optimize texture features into the scene representation. This loss function incorporates both camera pose and 3D geometric information of the scene, enabling controllable texture-oriented appearance editing. Finally, a geometry preservation strategy is introduced. By alternating between the texture transfer and geometry correction stages over multiple iterations, this strategy achieves a balance between learning texture features and preserving geometric integrity. Extensive experiments demonstrate the effectiveness and controllability of our method. Through geometric awareness, our approach achieves texture transfer results that better align with human visual perception. Our homepage is available at https://vpx-ecnu.github.io/GT2-GS-website.
>
---
#### [replaced 059] MMDrive: Interactive Scene Understanding Beyond Vision with Multi-representational Fusion
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向自动驾驶场景理解任务，解决现有视觉语言模型受限于2D图像、难以融合3D空间信息的问题。提出MMDrive框架，融合占用图、LiDAR点云和文本描述，设计文本导向调制器与跨模态抽象器，实现自适应多模态融合，在DriveLM和NuScenes-QA上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.13177v2](https://arxiv.org/pdf/2512.13177v2)**

> **作者:** Minghui Hou; Wei-Hsing Huang; Shaofeng Liang; Daizong Liu; Tai-Hao Wen; Gang Wang; Runwei Guan; Weiping Ding
>
> **摘要:** Vision-language models enable the understanding and reasoning of complex traffic scenarios through multi-source information fusion, establishing it as a core technology for autonomous driving. However, existing vision-language models are constrained by the image understanding paradigm in 2D plane, which restricts their capability to perceive 3D spatial information and perform deep semantic fusion, resulting in suboptimal performance in complex autonomous driving environments. This study proposes MMDrive, an multimodal vision-language model framework that extends traditional image understanding to a generalized 3D scene understanding framework. MMDrive incorporates three complementary modalities, including occupancy maps, LiDAR point clouds, and textual scene descriptions. To this end, it introduces two novel components for adaptive cross-modal fusion and key information extraction. Specifically, the Text-oriented Multimodal Modulator dynamically weights the contributions of each modality based on the semantic cues in the question, guiding context-aware feature integration. The Cross-Modal Abstractor employs learnable abstract tokens to generate compact, cross-modal summaries that highlight key regions and essential semantics. Comprehensive evaluations on the DriveLM and NuScenes-QA benchmarks demonstrate that MMDrive achieves significant performance gains over existing vision-language models for autonomous driving, with a BLEU-4 score of 54.56 and METEOR of 41.78 on DriveLM, and an accuracy score of 62.7% on NuScenes-QA. MMDrive effectively breaks the traditional image-only understanding barrier, enabling robust multimodal reasoning in complex driving environments and providing a new foundation for interpretable autonomous driving scene understanding.
>
---
#### [replaced 060] MMhops-R1: Multimodal Multi-hop Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13573v2](https://arxiv.org/pdf/2512.13573v2)**

> **作者:** Tao Zhang; Ziqi Zhang; Zongyang Ma; Yuxin Chen; Bing Li; Chunfeng Yuan; Guangting Wang; Fengyun Rao; Ying Shan; Weiming Hu
>
> **备注:** Acceped by AAAI 2026
>
> **摘要:** The ability to perform multi-modal multi-hop reasoning by iteratively integrating information across various modalities and external knowledge is critical for addressing complex real-world challenges. However, existing Multi-modal Large Language Models (MLLMs) are predominantly limited to single-step reasoning, as existing benchmarks lack the complexity needed to evaluate and drive multi-hop abilities. To bridge this gap, we introduce MMhops, a novel, large-scale benchmark designed to systematically evaluate and foster multi-modal multi-hop reasoning. MMhops dataset comprises two challenging task formats, Bridging and Comparison, which necessitate that models dynamically construct complex reasoning chains by integrating external knowledge. To tackle the challenges posed by MMhops, we propose MMhops-R1, a novel multi-modal Retrieval-Augmented Generation (mRAG) framework for dynamic reasoning. Our framework utilizes reinforcement learning to optimize the model for autonomously planning reasoning paths, formulating targeted queries, and synthesizing multi-level information. Comprehensive experiments demonstrate that MMhops-R1 significantly outperforms strong baselines on MMhops, highlighting that dynamic planning and multi-modal knowledge integration are crucial for complex reasoning. Moreover, MMhops-R1 demonstrates strong generalization to tasks requiring fixed-hop reasoning, underscoring the robustness of our dynamic planning approach. In conclusion, our work contributes a challenging new benchmark and a powerful baseline model, and we will release the associated code, data, and weights to catalyze future research in this critical area.
>
---
#### [replaced 061] Text Embedded Swin-UMamba for DeepLesion Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.06453v2](https://arxiv.org/pdf/2508.06453v2)**

> **作者:** Ruida Cheng; Tejas Sudharshan Mathai; Pritam Mukherjee; Benjamin Hou; Qingqing Zhu; Zhiyong Lu; Matthew McAuliffe; Ronald M. Summers
>
> **摘要:** Segmentation of lesions on CT enables automatic measurement for clinical assessment of chronic diseases (e.g., lymphoma). Integrating large language models (LLMs) into the lesion segmentation workflow has the potential to combine imaging features with descriptions of lesion characteristics from the radiology reports. In this study, we investigate the feasibility of integrating text into the Swin-UMamba architecture for the task of lesion segmentation. The publicly available ULS23 DeepLesion dataset was used along with short-form descriptions of the findings from the reports. On the test dataset, our method achieved a high Dice score of 82.64, and a low Hausdorff distance of 6.34 pixels was obtained for lesion segmentation. The proposed Text-Swin-U/Mamba model outperformed prior approaches: 37.79% improvement over the LLM-driven LanGuideMedSeg model (p < 0.001), and surpassed the purely image-based XLSTM-UNet and nnUNet models by 2.58% and 1.01%, respectively. The dataset and code can be accessed at https://github.com/ruida/LLM-Swin-UMamba
>
---
#### [replaced 062] POLAR: A Portrait OLAT Dataset and Generative Framework for Illumination-Aware Face Modeling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13192v2](https://arxiv.org/pdf/2512.13192v2)**

> **作者:** Zhuo Chen; Chengqun Yang; Zhuo Su; Zheng Lv; Jingnan Gao; Xiaoyuan Zhang; Xiaokang Yang; Yichao Yan
>
> **备注:** 19 pages, 19 figures
>
> **摘要:** Face relighting aims to synthesize realistic portraits under novel illumination while preserving identity and geometry. However, progress remains constrained by the limited availability of large-scale, physically consistent illumination data. To address this, we introduce POLAR, a large-scale and physically calibrated One-Light-at-a-Time (OLAT) dataset containing over 200 subjects captured under 156 lighting directions, multiple views, and diverse expressions. Building upon POLAR, we develop a flow-based generative model POLARNet that predicts per-light OLAT responses from a single portrait, capturing fine-grained and direction-aware illumination effects while preserving facial identity. Unlike diffusion or background-conditioned methods that rely on statistical or contextual cues, our formulation models illumination as a continuous, physically interpretable transformation between lighting states, enabling scalable and controllable relighting. Together, POLAR and POLARNet form a unified illumination learning framework that links real data, generative synthesis, and physically grounded relighting, establishing a self-sustaining "chicken-and-egg" cycle for scalable and reproducible portrait illumination. Our project page: https://rex0191.github.io/POLAR/.
>
---
#### [replaced 063] Enhancing Geo-localization for Crowdsourced Flood Imagery via LLM-Guided Attention
- **分类: cs.CL; cs.AI; cs.CV; cs.CY**

- **简介: 该论文属视觉地点识别（VPR）任务，旨在解决社交媒体洪水图像因缺乏地理元数据、视觉失真和域偏移导致的定位不准问题。提出VPR-AttLLM框架，利用大语言模型引导注意力增强图像描述符，提升跨源检索精度，无需重训练或新数据。**

- **链接: [https://arxiv.org/pdf/2512.11811v2](https://arxiv.org/pdf/2512.11811v2)**

> **作者:** Fengyi Xu; Jun Ma; Waishan Qiu; Cui Guo; Jack C. P. Cheng
>
> **备注:** Updated author list to include additional contributor. Revised title and improved methodology section based on collaborative feedback
>
> **摘要:** Crowdsourced street-view imagery from social media provides real-time visual evidence of urban flooding and other crisis events, yet it often lacks reliable geographic metadata for emergency response. Existing image geo-localization approaches, also known as Visual Place Recognition (VPR) models, exhibit substantial performance degradation when applied to such imagery due to visual distortions and domain shifts in cross-source scenarios. This paper presents VPR-AttLLM, a model-agnostic framework that integrates the semantic reasoning and geo-knowledge of Large Language Models (LLMs) into established VPR pipelines through attention-guided descriptor enhancement. By leveraging LLMs to identify location-informative regions within the city context and suppress visual noise, VPR-AttLLM improves retrieval performance without requiring model retraining or additional data. Comprehensive evaluations are conducted on extended benchmarks including SF-XL enriched with real social-media flood images, synthetic flooding scenarios over established query sets and Mapillary photos, and a new HK-URBAN dataset capturing morphologically distinct cityscapes. Integrating VPR-AttLLM with three state-of-the-art VPR models-CosPlace, EigenPlaces, and SALAD-consistently improves recall performance, yielding relative gains typically between 1-3% and reaching up to 8% on the most challenging real flood imagery. Beyond measurable gains in retrieval accuracy, this study establishes a generalizable paradigm for LLM-guided multimodal fusion in visual retrieval systems. By embedding principles from urban perception theory into attention mechanisms, VPR-AttLLM bridges human-like spatial reasoning with modern VPR architectures. Its plug-and-play design, strong cross-source robustness, and interpretability highlight its potential for scalable urban monitoring and rapid geo-localization of crowdsourced crisis imagery.
>
---
#### [replaced 064] Guideline-Consistent Segmentation via Multi-Agent Refinement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.04687v2](https://arxiv.org/pdf/2509.04687v2)**

> **作者:** Vanshika Vats; Ashwani Rathee; James Davis
>
> **备注:** To be published in The Fortieth AAAI Conference on Artificial Intelligence (AAAI 2026)
>
> **摘要:** Semantic segmentation in real-world applications often requires not only accurate masks but also strict adherence to textual labeling guidelines. These guidelines are typically complex and long, and both human and automated labeling often fail to follow them faithfully. Traditional approaches depend on expensive task-specific retraining that must be repeated as the guidelines evolve. Although recent open-vocabulary segmentation methods excel with simple prompts, they often fail when confronted with sets of paragraph-length guidelines that specify intricate segmentation rules. To address this, we introduce a multi-agent, training-free framework that coordinates general-purpose vision-language models within an iterative Worker-Supervisor refinement architecture. The Worker performs the segmentation, the Supervisor critiques it against the retrieved guidelines, and a lightweight reinforcement learning stop policy decides when to terminate the loop, ensuring guideline-consistent masks while balancing resource use. Evaluated on the Waymo and ReasonSeg datasets, our method notably outperforms state-of-the-art baselines, demonstrating strong generalization and instruction adherence.
>
---
#### [replaced 065] FutrTrack: A Camera-LiDAR Fusion Transformer for 3D Multiple Object Tracking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.19981v2](https://arxiv.org/pdf/2510.19981v2)**

> **作者:** Martha Teiko Teye; Ori Maoz; Matthias Rottmann
>
> **备注:** Accepted to VISAPP 2026
>
> **摘要:** We propose FutrTrack, a modular camera-LiDAR multi-object tracking framework that builds on existing 3D detectors by introducing a transformer-based smoother and a fusion-driven tracker. Inspired by query-based tracking frameworks, FutrTrack employs a multimodal two-stage transformer refinement and tracking pipeline. Our fusion tracker integrates bounding boxes with multimodal bird's-eye-view (BEV) fusion features from multiple cameras and LiDAR without the need for an explicit motion model. The tracker assigns and propagates identities across frames, leveraging both geometric and semantic cues for robust re-identification under occlusion and viewpoint changes. Prior to tracking, we refine sequences of bounding boxes with a temporal smoother over a moving window to refine trajectories, reduce jitter, and improve spatial consistency. Evaluated on nuScenes and KITTI, FutrTrack demonstrates that query-based transformer tracking methods benefit significantly from multimodal sensor features compared with previous single-sensor approaches. With an aMOTA of 74.7 on the nuScenes test set, FutrTrack achieves strong performance on 3D MOT benchmarks, reducing identity switches while maintaining competitive accuracy. Our approach provides an efficient framework for improving transformer-based trackers to compete with other neural-network-based methods even with limited data and without pretraining.
>
---
#### [replaced 066] One Layer Is Enough: Adapting Pretrained Visual Encoders for Image Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.07829v2](https://arxiv.org/pdf/2512.07829v2)**

> **作者:** Yuan Gao; Chen Chen; Tianrong Chen; Jiatao Gu
>
> **摘要:** Visual generative models (e.g., diffusion models) typically operate in compressed latent spaces to balance training efficiency and sample quality. In parallel, there has been growing interest in leveraging high-quality pre-trained visual representations, either by aligning them inside VAEs or directly within the generative model. However, adapting such representations remains challenging due to fundamental mismatches between understanding-oriented features and generation-friendly latent spaces. Representation encoders benefit from high-dimensional latents that capture diverse hypotheses for masked regions, whereas generative models favor low-dimensional latents that must faithfully preserve injected noise. This discrepancy has led prior work to rely on complex objectives and architectures. In this work, we propose FAE (Feature Auto-Encoder), a simple yet effective framework that adapts pre-trained visual representations into low-dimensional latents suitable for generation using as little as a single attention layer, while retaining sufficient information for both reconstruction and understanding. The key is to couple two separate deep decoders: one trained to reconstruct the original feature space, and a second that takes the reconstructed features as input for image generation. FAE is generic; it can be instantiated with a variety of self-supervised encoders (e.g., DINO, SigLIP) and plugged into two distinct generative families: diffusion models and normalizing flows. Across class-conditional and text-to-image benchmarks, FAE achieves strong performance. For example, on ImageNet 256x256, our diffusion model with CFG attains a near state-of-the-art FID of 1.29 (800 epochs) and 1.70 (80 epochs). Without CFG, FAE reaches the state-of-the-art FID of 1.48 (800 epochs) and 2.08 (80 epochs), demonstrating both high quality and fast learning.
>
---
#### [replaced 067] TransientTrack: Advanced Multi-Object Tracking and Classification of Cancer Cells with Transient Fluorescent Signals
- **分类: cs.CV; q-bio.CB; q-bio.QM**

- **链接: [https://arxiv.org/pdf/2512.01885v2](https://arxiv.org/pdf/2512.01885v2)**

> **作者:** Florian Bürger; Martim Dias Gomes; Nica Gutu; Adrián E. Granada; Noémie Moreau; Katarzyna Bozek
>
> **备注:** 13 pages, 7 figures, 2 tables
>
> **摘要:** Tracking cells in time-lapse videos is an essential technique for monitoring cell population dynamics at a single-cell level. Current methods for cell tracking are developed on videos with mostly single, constant signals and do not detect pivotal events such as cell death. Here, we present TransientTrack, a deep learning-based framework for cell tracking in multi-channel microscopy video data with transient fluorescent signals that fluctuate over time following processes such as the circadian rhythm of cells. By identifying key cellular events - mitosis (cell division) and apoptosis (cell death) our method allows us to build complete trajectories, including cell lineage information. TransientTrack is lightweight and performs matching on cell detection embeddings directly, without the need for quantification of tracking-specific cell features. Furthermore, our approach integrates Transformer Networks, multi-stage matching using all detection boxes, and the interpolation of missing tracklets with the Kalman Filter. This unified framework achieves strong performance across diverse conditions, effectively tracking cells and capturing cell division and death. We demonstrate the use of TransientTrack in an analysis of the efficacy of a chemotherapeutic drug at a single-cell level. The proposed framework could further advance quantitative studies of cancer cell dynamics, enabling detailed characterization of treatment response and resistance mechanisms. The code is available at https://github.com/bozeklab/TransientTrack.
>
---
#### [replaced 068] White Aggregation and Restoration for Few-shot 3D Point Cloud Semantic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.13907v2](https://arxiv.org/pdf/2509.13907v2)**

> **作者:** Jiyun Im; SuBeen Lee; Miso Lee; Jae-Pil Heo
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Few-Shot 3D Point Cloud Semantic Segmentation (FS-PCS) aims to predict per-point labels for an unlabeled point cloud, given only a few labeled examples. To extract discriminative representations from the limited labeled set, existing methods have constructed prototypes using algorithms such as farthest point sampling (FPS). However, we point out that this convention has undesirable effects as performance fluctuates depending on sampling, while the prototype generation process remains underexplored in the field. This motivates us to investigate an advanced prototype generation method based on attention mechanism. Despite its potential, we found that vanilla attention module suffers from the distributional gap between prototypical tokens and support features. To overcome this, we propose White Aggregation and Restoration Module (WARM), which resolves the misalignment by sandwiching cross-attention between whitening and coloring transformations. Specifically, whitening aligns the features to tokens before the attention process, and coloring subsequently restores the original distribution to the attended tokens. This simple yet effective design enables robust attention, thereby generating prototypes that capture the semantic relationships in support features. WARM achieves state-of-the-art performance with a significant margin on FS-PCS benchmarks, and demonstrates its effectiveness through extensive experiments.
>
---
