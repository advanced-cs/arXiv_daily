# 计算机视觉 cs.CV

- **最新发布 125 篇**

- **更新 92 篇**

## 最新发布

#### [new 001] Unintended Bias in 2D+ Image Segmentation and Its Effect on Attention Asymmetry
- **分类: cs.CV**

- **简介: 该论文研究2D+图像分割中预训练模型引入的意外偏差问题，导致特征利用不均和注意力不对称。通过实验比较预训练与随机初始化模型，提出中和颜色通道偏差的方法，提升模型性能与可解释性。**

- **链接: [http://arxiv.org/pdf/2505.14105v1](http://arxiv.org/pdf/2505.14105v1)**

> **作者:** Zsófia Molnár; Gergely Szabó; András Horváth
>
> **摘要:** Supervised pretrained models have become widely used in deep learning, especially for image segmentation tasks. However, when applied to specialized datasets such as biomedical imaging, pretrained weights often introduce unintended biases. These biases cause models to assign different levels of importance to different slices, leading to inconsistencies in feature utilization, which can be observed as asymmetries in saliency map distributions. This transfer of color distributions from natural images to non-natural datasets can compromise model performance and reduce the reliability of results. In this study, we investigate the effects of these biases and propose strategies to mitigate them. Through a series of experiments, we test both pretrained and randomly initialized models, comparing their performance and saliency map distributions. Our proposed methods, which aim to neutralize the bias introduced by pretrained color channel weights, demonstrate promising results, offering a practical approach to improving model explainability while maintaining the benefits of pretrained models. This publication presents our findings, providing insights into addressing pretrained weight biases across various deep learning tasks.
>
---
#### [new 002] Vid2World: Crafting Video Diffusion Models to Interactive World Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出Vid2World，将预训练视频扩散模型转化为交互式世界模型。针对传统世界模型依赖领域数据、预测精度低的问题，通过因果化架构设计、自回归生成及动作引导机制，提升生成可控性。实验表明其在机器人与游戏场景中有效。**

- **链接: [http://arxiv.org/pdf/2505.14357v1](http://arxiv.org/pdf/2505.14357v1)**

> **作者:** Siqiao Huang; Jialong Wu; Qixing Zhou; Shangchen Miao; Mingsheng Long
>
> **备注:** Project page: http://knightnemo.github.io/vid2world/
>
> **摘要:** World models, which predict transitions based on history observation and action sequences, have shown great promise in improving data efficiency for sequential decision making. However, existing world models often require extensive domain-specific training and still produce low-fidelity, coarse predictions, limiting their applicability in complex environments. In contrast, video diffusion models trained on large, internet-scale datasets have demonstrated impressive capabilities in generating high-quality videos that capture diverse real-world dynamics. In this work, we present Vid2World, a general approach for leveraging and transferring pre-trained video diffusion models into interactive world models. To bridge the gap, Vid2World performs casualization of a pre-trained video diffusion model by crafting its architecture and training objective to enable autoregressive generation. Furthermore, it introduces a causal action guidance mechanism to enhance action controllability in the resulting interactive world model. Extensive experiments in robot manipulation and game simulation domains show that our method offers a scalable and effective approach for repurposing highly capable video diffusion models to interactive world models.
>
---
#### [new 003] Visionary-R1: Mitigating Shortcuts in Visual Reasoning with Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文属于视觉推理任务，旨在解决视觉语言模型（VLMs）在强化学习中因依赖捷径而泛化能力差的问题。通过强制模型遵循"图像描述-推理-答案"的输出格式，并用273K无思维链监督的数据训练，提出Visionary-R1，在多个基准上超越现有模型。**

- **链接: [http://arxiv.org/pdf/2505.14677v1](http://arxiv.org/pdf/2505.14677v1)**

> **作者:** Jiaer Xia; Yuhang Zang; Peng Gao; Yixuan Li; Kaiyang Zhou
>
> **摘要:** Learning general-purpose reasoning capabilities has long been a challenging problem in AI. Recent research in large language models (LLMs), such as DeepSeek-R1, has shown that reinforcement learning techniques like GRPO can enable pre-trained LLMs to develop reasoning capabilities using simple question-answer pairs. In this paper, we aim to train visual language models (VLMs) to perform reasoning on image data through reinforcement learning and visual question-answer pairs, without any explicit chain-of-thought (CoT) supervision. Our findings indicate that simply applying reinforcement learning to a VLM -- by prompting the model to produce a reasoning chain before providing an answer -- can lead the model to develop shortcuts from easy questions, thereby reducing its ability to generalize across unseen data distributions. We argue that the key to mitigating shortcut learning is to encourage the model to interpret images prior to reasoning. Therefore, we train the model to adhere to a caption-reason-answer output format: initially generating a detailed caption for an image, followed by constructing an extensive reasoning chain. When trained on 273K CoT-free visual question-answer pairs and using only reinforcement learning, our model, named Visionary-R1, outperforms strong multimodal models, such as GPT-4o, Claude3.5-Sonnet, and Gemini-1.5-Pro, on multiple visual reasoning benchmarks.
>
---
#### [new 004] Dual Precision Quantization for Efficient and Accurate Deep Neural Networks Inference
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文提出Dual Precision Quantization（DPQ），属模型量化任务，解决大模型推理的效率问题。通过W4A8方案（4位权重存储+8位浮点计算）及DPQ算法，在保持较高精度下提升推理速度与内存效率，实验显示性能提升且精度损失可控。**

- **链接: [http://arxiv.org/pdf/2505.14638v1](http://arxiv.org/pdf/2505.14638v1)**

> **作者:** Tomer Gafni; Asaf Karnieli; Yair Hanani
>
> **备注:** Accepted at eLVM Workshop, CVPR, 2025
>
> **摘要:** Deep neural networks have achieved state-of-the-art results in a wide range of applications, from natural language processing and computer vision to speech recognition. However, as tasks become increasingly complex, model sizes continue to grow, posing challenges in latency and memory efficiency. To meet these constraints, post-training quantization has emerged as a promising solution. In this paper, we propose a novel hardware-efficient quantization and inference scheme that exploits hardware advantages with minimal accuracy degradation. Specifically, we introduce a W4A8 scheme, where weights are quantized and stored using 4-bit integer precision, and inference computations are performed using 8-bit floating-point arithmetic, demonstrating significant speedups and improved memory utilization compared to 16-bit operations, applicable on various modern accelerators. To mitigate accuracy loss, we develop a novel quantization algorithm, dubbed Dual Precision Quantization (DPQ), that leverages the unique structure of our scheme without introducing additional inference overhead. Experimental results demonstrate improved performance (i.e., increased throughput) while maintaining tolerable accuracy degradation relative to the full-precision model.
>
---
#### [new 005] ReactDiff: Latent Diffusion for Facial Reaction Generation
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于面部反应生成任务，解决多模态关联与反应多样性问题。提出ReactDiff框架，结合多模态Transformer与潜扩散模型，通过跨模态注意力机制捕捉视听关联，利用潜扩散过程生成多样且自然的面部反应，实验显示性能显著提升。**

- **链接: [http://arxiv.org/pdf/2505.14151v1](http://arxiv.org/pdf/2505.14151v1)**

> **作者:** Jiaming Li; Sheng Wang; Xin Wang; Yitao Zhu; Honglin Xiong; Zixu Zhuang; Qian Wang
>
> **摘要:** Given the audio-visual clip of the speaker, facial reaction generation aims to predict the listener's facial reactions. The challenge lies in capturing the relevance between video and audio while balancing appropriateness, realism, and diversity. While prior works have mostly focused on uni-modal inputs or simplified reaction mappings, recent approaches such as PerFRDiff have explored multi-modal inputs and the one-to-many nature of appropriate reaction mappings. In this work, we propose the Facial Reaction Diffusion (ReactDiff) framework that uniquely integrates a Multi-Modality Transformer with conditional diffusion in the latent space for enhanced reaction generation. Unlike existing methods, ReactDiff leverages intra- and inter-class attention for fine-grained multi-modal interaction, while the latent diffusion process between the encoder and decoder enables diverse yet contextually appropriate outputs. Experimental results demonstrate that ReactDiff significantly outperforms existing approaches, achieving a facial reaction correlation of 0.26 and diversity score of 0.094 while maintaining competitive realism. The code is open-sourced at \href{https://github.com/Hunan-Tiger/ReactDiff}{github}.
>
---
#### [new 006] CAD-Coder: An Open-Source Vision-Language Model for Computer-Aided Design Code Generation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出CAD-Coder，开源视觉语言模型，通过图像生成可编辑CAD代码（CadQuery Python），解决现有模型精度低、无法泛化到真实图像的问题。基于自建16万图像-代码对的GenCAD-Code数据集微调，实现100%语法正确率和高3D相似度，支持未知操作与真实图像输入，开源项目。**

- **链接: [http://arxiv.org/pdf/2505.14646v1](http://arxiv.org/pdf/2505.14646v1)**

> **作者:** Anna C. Doris; Md Ferdous Alam; Amin Heyrani Nobari; Faez Ahmed
>
> **摘要:** Efficient creation of accurate and editable 3D CAD models is critical in engineering design, significantly impacting cost and time-to-market in product innovation. Current manual workflows remain highly time-consuming and demand extensive user expertise. While recent developments in AI-driven CAD generation show promise, existing models are limited by incomplete representations of CAD operations, inability to generalize to real-world images, and low output accuracy. This paper introduces CAD-Coder, an open-source Vision-Language Model (VLM) explicitly fine-tuned to generate editable CAD code (CadQuery Python) directly from visual input. Leveraging a novel dataset that we created--GenCAD-Code, consisting of over 163k CAD-model image and code pairs--CAD-Coder outperforms state-of-the-art VLM baselines such as GPT-4.5 and Qwen2.5-VL-72B, achieving a 100% valid syntax rate and the highest accuracy in 3D solid similarity. Notably, our VLM demonstrates some signs of generalizability, successfully generating CAD code from real-world images and executing CAD operations unseen during fine-tuning. The performance and adaptability of CAD-Coder highlights the potential of VLMs fine-tuned on code to streamline CAD workflows for engineers and designers. CAD-Coder is publicly available at: https://github.com/anniedoris/CAD-Coder.
>
---
#### [new 007] ReSW-VL: Representation Learning for Surgical Workflow Analysis Using Vision-Language Model
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于手术阶段识别任务，针对现有CNN在表征学习中的训练方法不足，提出ReSW-VL方法，利用CLIP模型的图像编码器结合提示学习进行微调，实验在三个数据集验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2505.13746v1](http://arxiv.org/pdf/2505.13746v1)**

> **作者:** Satoshi Kondo
>
> **摘要:** Surgical phase recognition from video is a technology that automatically classifies the progress of a surgical procedure and has a wide range of potential applications, including real-time surgical support, optimization of medical resources, training and skill assessment, and safety improvement. Recent advances in surgical phase recognition technology have focused primarily on Transform-based methods, although methods that extract spatial features from individual frames using a CNN and video features from the resulting time series of spatial features using time series modeling have shown high performance. However, there remains a paucity of research on training methods for CNNs employed for feature extraction or representation learning in surgical phase recognition. In this study, we propose a method for representation learning in surgical workflow analysis using a vision-language model (ReSW-VL). Our proposed method involves fine-tuning the image encoder of a CLIP (Convolutional Language Image Model) vision-language model using prompt learning for surgical phase recognition. The experimental results on three surgical phase recognition datasets demonstrate the effectiveness of the proposed method in comparison to conventional methods.
>
---
#### [new 008] Replace in Translation: Boost Concept Alignment in Counterfactual Text-to-Image
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对反事实文本到图像生成任务，解决反事实场景中对象概念对齐问题。提出通过潜在空间逐步替换对象并利用ELNP策略（基于DeepSeek生成指令），同时设计指标评估概念覆盖，提升生成图像的合理性。**

- **链接: [http://arxiv.org/pdf/2505.14341v1](http://arxiv.org/pdf/2505.14341v1)**

> **作者:** Sifan Li; Ming Tao; Hao Zhao; Ling Shao; Hao Tang
>
> **摘要:** Text-to-Image (T2I) has been prevalent in recent years, with most common condition tasks having been optimized nicely. Besides, counterfactual Text-to-Image is obstructing us from a more versatile AIGC experience. For those scenes that are impossible to happen in real world and anti-physics, we should spare no efforts in increasing the factual feel, which means synthesizing images that people think very likely to be happening, and concept alignment, which means all the required objects should be in the same frame. In this paper, we focus on concept alignment. As controllable T2I models have achieved satisfactory performance for real applications, we utilize this technology to replace the objects in a synthesized image in latent space step-by-step to change the image from a common scene to a counterfactual scene to meet the prompt. We propose a strategy to instruct this replacing process, which is called as Explicit Logical Narrative Prompt (ELNP), by using the newly SoTA language model DeepSeek to generate the instructions. Furthermore, to evaluate models' performance in counterfactual T2I, we design a metric to calculate how many required concepts in the prompt can be covered averagely in the synthesized images. The extensive experiments and qualitative comparisons demonstrate that our strategy can boost the concept alignment in counterfactual T2I.
>
---
#### [new 009] Ground-V: Teaching VLMs to Ground Complex Instructions in Pixels
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型（VLM）的像素级接地任务，解决复杂指令下的物体定位挑战（如幻觉引用、多对象、推理等）。通过知识蒸馏自动生成带像素标注的指令-响应数据集Ground-V，显著提升模型在定位任务（如RefCOCO）的性能，创state-of-the-art纪录。**

- **链接: [http://arxiv.org/pdf/2505.13788v1](http://arxiv.org/pdf/2505.13788v1)**

> **作者:** Yongshuo Zong; Qin Zhang; Dongsheng An; Zhihua Li; Xiang Xu; Linghan Xu; Zhuowen Tu; Yifan Xing; Onkar Dabeer
>
> **备注:** Accepted to CVPR'25
>
> **摘要:** This work presents a simple yet effective workflow for automatically scaling instruction-following data to elicit pixel-level grounding capabilities of VLMs under complex instructions. In particular, we address five critical real-world challenges in text-instruction-based grounding: hallucinated references, multi-object scenarios, reasoning, multi-granularity, and part-level references. By leveraging knowledge distillation from a pre-trained teacher model, our approach generates high-quality instruction-response pairs linked to existing pixel-level annotations, minimizing the need for costly human annotation. The resulting dataset, Ground-V, captures rich object localization knowledge and nuanced pixel-level referring expressions. Experiment results show that models trained on Ground-V exhibit substantial improvements across diverse grounding tasks. Specifically, incorporating Ground-V during training directly achieves an average accuracy boost of 4.4% for LISA and a 7.9% for PSALM across six benchmarks on the gIoU metric. It also sets new state-of-the-art results on standard benchmarks such as RefCOCO/+/g. Notably, on gRefCOCO, we achieve an N-Acc of 83.3%, exceeding the previous state-of-the-art by more than 20%.
>
---
#### [new 010] Training-Free Watermarking for Autoregressive Image Generation
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文属于图像生成模型水印任务，针对自回归模型水印方法缺失的问题，提出训练免去的IndexMark框架。利用代码本冗余性，通过匹配替换相似索引嵌入水印，结合比例验证和辅助验证提升鲁棒性，实验证明其高质量与抗扰动性。**

- **链接: [http://arxiv.org/pdf/2505.14673v1](http://arxiv.org/pdf/2505.14673v1)**

> **作者:** Yu Tong; Zihao Pan; Shuai Yang; Kaiyang Zhou
>
> **摘要:** Invisible image watermarking can protect image ownership and prevent malicious misuse of visual generative models. However, existing generative watermarking methods are mainly designed for diffusion models while watermarking for autoregressive image generation models remains largely underexplored. We propose IndexMark, a training-free watermarking framework for autoregressive image generation models. IndexMark is inspired by the redundancy property of the codebook: replacing autoregressively generated indices with similar indices produces negligible visual differences. The core component in IndexMark is a simple yet effective match-then-replace method, which carefully selects watermark tokens from the codebook based on token similarity, and promotes the use of watermark tokens through token replacement, thereby embedding the watermark without affecting the image quality. Watermark verification is achieved by calculating the proportion of watermark tokens in generated images, with precision further improved by an Index Encoder. Furthermore, we introduce an auxiliary validation scheme to enhance robustness against cropping attacks. Experiments demonstrate that IndexMark achieves state-of-the-art performance in terms of image quality and verification accuracy, and exhibits robustness against various perturbations, including cropping, noises, Gaussian blur, random erasing, color jittering, and JPEG compression.
>
---
#### [new 011] Domain Adaptation of VLM for Soccer Video Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉语言模型（VLM）的领域自适应任务，解决其在特定领域（如足球视频）的性能不足问题。通过结合大规模足球数据集与LLM生成指令数据，采用课程学习逐步微调模型。实验表明，在足球问答和动作分类任务中，模型准确率显著提升（如VQA提升37.5%，分类准确率从11.8%升至63.5%）。**

- **链接: [http://arxiv.org/pdf/2505.13860v1](http://arxiv.org/pdf/2505.13860v1)**

> **作者:** Tiancheng Jiang; Henry Wang; Md Sirajus Salekin; Parmida Atighehchian; Shinan Zhang
>
> **备注:** 8 pages, 5 figures, accepted to the 11th IEEE International Workshop on Computer Vision in Sports (CVSports) at CVPR 2025; supplementary appendix included as ancillary PDF
>
> **摘要:** Vision Language Models (VLMs) have demonstrated strong performance in multi-modal tasks by effectively aligning visual and textual representations. However, most video understanding VLM research has been domain-agnostic, leaving the understanding of their transfer learning capability to specialized domains under-explored. In this work, we address this by exploring the adaptability of open-source VLMs to specific domains, and focusing on soccer as an initial case study. Our approach uses large-scale soccer datasets and LLM to create instruction-following data, and use them to iteratively fine-tune the general-domain VLM in a curriculum learning fashion (first teaching the model key soccer concepts to then question answering tasks). The final adapted model, trained using a curated dataset of 20k video clips, exhibits significant improvement in soccer-specific tasks compared to the base model, with a 37.5% relative improvement for the visual question-answering task and an accuracy improvement from 11.8% to 63.5% for the downstream soccer action classification task.
>
---
#### [new 012] Towards Generating Realistic Underwater Images
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于水下图像生成任务，旨在通过对比学习和生成对抗网络将均匀光照的合成图像转化为逼真水下图像。研究对比了配对（如pix2pix）和非配对方法（如CycleGAN、CUT），并发现结合深度信息的CUT在FID（ realism）最优，但SSIM（结构保真）略降，揭示深度信息提升真实感却可能引入结构偏差。**

- **链接: [http://arxiv.org/pdf/2505.14296v1](http://arxiv.org/pdf/2505.14296v1)**

> **作者:** Abdul-Kazeem Shamba
>
> **摘要:** This paper explores the use of contrastive learning and generative adversarial networks for generating realistic underwater images from synthetic images with uniform lighting. We investigate the performance of image translation models for generating realistic underwater images using the VAROS dataset. Two key evaluation metrics, Fr\'echet Inception Distance (FID) and Structural Similarity Index Measure (SSIM), provide insights into the trade-offs between perceptual quality and structural preservation. For paired image translation, pix2pix achieves the best FID scores due to its paired supervision and PatchGAN discriminator, while the autoencoder model attains the highest SSIM, suggesting better structural fidelity despite producing blurrier outputs. Among unpaired methods, CycleGAN achieves a competitive FID score by leveraging cycle-consistency loss, whereas CUT, which replaces cycle-consistency with contrastive learning, attains higher SSIM, indicating improved spatial similarity retention. Notably, incorporating depth information into CUT results in the lowest overall FID score, demonstrating that depth cues enhance realism. However, the slight decrease in SSIM suggests that depth-aware learning may introduce structural variations.
>
---
#### [new 013] Plane Geometry Problem Solving with Multi-modal Reasoning: A Survey
- **分类: cs.CV; cs.LG**

- **简介: 该论文是关于平面几何问题求解（PGPS）的综述，旨在系统梳理多模态推理方法。针对现有研究缺乏系统性总结及编码幻觉、数据泄露等挑战，论文分类梳理了编码-解码框架下的方法，分析模型架构，并指出未来方向。**

- **链接: [http://arxiv.org/pdf/2505.14340v1](http://arxiv.org/pdf/2505.14340v1)**

> **作者:** Seunghyuk Cho; Zhenyue Qin; Yang Liu; Youngbin Choi; Seungbeom Lee; Dongwoo Kim
>
> **备注:** 18 pages
>
> **摘要:** Plane geometry problem solving (PGPS) has recently gained significant attention as a benchmark to assess the multi-modal reasoning capabilities of large vision-language models. Despite the growing interest in PGPS, the research community still lacks a comprehensive overview that systematically synthesizes recent work in PGPS. To fill this gap, we present a survey of existing PGPS studies. We first categorize PGPS methods into an encoder-decoder framework and summarize the corresponding output formats used by their encoders and decoders. Subsequently, we classify and analyze these encoders and decoders according to their architectural designs. Finally, we outline major challenges and promising directions for future research. In particular, we discuss the hallucination issues arising during the encoding phase within encoder-decoder architectures, as well as the problem of data leakage in current PGPS benchmarks.
>
---
#### [new 014] SparC: Sparse Representation and Construction for High-Resolution 3D Shapes Modeling
- **分类: cs.CV**

- **简介: 该论文属于高分辨率3D形状建模任务，旨在解决现有方法因VAE表征低效及模态不匹配导致的细节丢失问题。提出SparC框架，结合稀疏变形Marching Cubes（SparseCubes）与稀疏卷积VAE（SparConv-VAE），通过稀疏体素场高效重建高保真3D模型，支持潜扩散生成，减少计算成本并保留精细细节。**

- **链接: [http://arxiv.org/pdf/2505.14521v1](http://arxiv.org/pdf/2505.14521v1)**

> **作者:** Zhihao Li; Yufei Wang; Heliang Zheng; Yihao Luo; Bihan Wen
>
> **备注:** Homepage: https://lizhihao6.github.io/SparC
>
> **摘要:** High-fidelity 3D object synthesis remains significantly more challenging than 2D image generation due to the unstructured nature of mesh data and the cubic complexity of dense volumetric grids. Existing two-stage pipelines-compressing meshes with a VAE (using either 2D or 3D supervision), followed by latent diffusion sampling-often suffer from severe detail loss caused by inefficient representations and modality mismatches introduced in VAE. We introduce SparC, a unified framework that combines a sparse deformable marching cubes representation SparseCubes with a novel encoder SparConv-VAE. SparseCubes converts raw meshes into high-resolution ($1024^3$) surfaces with arbitrary topology by scattering signed distance and deformation fields onto a sparse cube, allowing differentiable optimization. SparConv-VAE is the first modality-consistent variational autoencoder built entirely upon sparse convolutional networks, enabling efficient and near-lossless 3D reconstruction suitable for high-resolution generative modeling through latent diffusion. SparC achieves state-of-the-art reconstruction fidelity on challenging inputs, including open surfaces, disconnected components, and intricate geometry. It preserves fine-grained shape details, reduces training and inference cost, and integrates naturally with latent diffusion models for scalable, high-resolution 3D generation.
>
---
#### [new 015] Towards Omnidirectional Reasoning with 360-R1: A Dataset, Benchmark, and GRPO-based Method
- **分类: cs.CV**

- **简介: 该论文聚焦全景视觉问答任务，针对现有多模态模型在360度图像理解中的局限（如定位、特征提取和幻觉问题），构建首个全景VQA数据集OmniVQA并建立基准，提出基于GRPO改进的360-R1方法，通过三类新型奖励函数优化模型，实验显示其性能提升6%。**

- **链接: [http://arxiv.org/pdf/2505.14197v1](http://arxiv.org/pdf/2505.14197v1)**

> **作者:** Xinshen Zhang; Zhen Ye; Xu Zheng
>
> **摘要:** Omnidirectional images (ODIs), with their 360{\deg} field of view, provide unparalleled spatial awareness for immersive applications like augmented reality and embodied AI. However, the capability of existing multi-modal large language models (MLLMs) to comprehend and reason about such panoramic scenes remains underexplored. This paper addresses this gap by introducing OmniVQA, the first dataset and conducting the first benchmark for omnidirectional visual question answering. Our evaluation of state-of-the-art MLLMs reveals significant limitations in handling omnidirectional visual question answering, highlighting persistent challenges in object localization, feature extraction, and hallucination suppression within panoramic contexts. These results underscore the disconnect between current MLLM capabilities and the demands of omnidirectional visual understanding, which calls for dedicated architectural or training innovations tailored to 360{\deg} imagery. Building on the OmniVQA dataset and benchmark, we further introduce a rule-based reinforcement learning method, 360-R1, based on Qwen2.5-VL-Instruct. Concretely, we modify the group relative policy optimization (GRPO) by proposing three novel reward functions: (1) reasoning process similarity reward, (2) answer semantic accuracy reward, and (3) structured format compliance reward. Extensive experiments on our OmniVQA demonstrate the superiority of our proposed method in omnidirectional space (+6% improvement).
>
---
#### [new 016] RETRO: REthinking Tactile Representation Learning with Material PriOrs
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于触觉表示学习任务，旨在解决现有方法忽视材料特性的问题。通过引入材料先验知识到学习框架，使模型更好捕捉表面纹理差异，提升机器人、触觉系统等场景的感知性能。**

- **链接: [http://arxiv.org/pdf/2505.14319v1](http://arxiv.org/pdf/2505.14319v1)**

> **作者:** Weihao Xia; Chenliang Zhou; Cengiz Oztireli
>
> **备注:** Code: https://github.com/weihaox/RETRO
>
> **摘要:** Tactile perception is profoundly influenced by the surface properties of objects in contact. However, despite their crucial role in shaping tactile experiences, these material characteristics have been largely neglected in existing tactile representation learning methods. Most approaches primarily focus on aligning tactile data with visual or textual information, overlooking the richness of tactile feedback that comes from understanding the materials' inherent properties. In this work, we address this gap by revisiting the tactile representation learning framework and incorporating material-aware priors into the learning process. These priors, which represent pre-learned characteristics specific to different materials, allow tactile models to better capture and generalize the nuances of surface texture. Our method enables more accurate, contextually rich tactile feedback across diverse materials and textures, improving performance in real-world applications such as robotics, haptic feedback systems, and material editing.
>
---
#### [new 017] ReservoirTTA: Prolonged Test-time Adaptation for Evolving and Recurring Domains
- **分类: cs.CV**

- **简介: 该论文属于测试时间适应（TTA）任务，针对持续变化或反复出现的领域分布，解决单模型适应中灾难性遗忘、领域干扰等问题。提出ReservoirTTA框架，通过维护多模型水库（在线聚类检测新领域并路由样本），实现动态适应，提升长期非平稳场景下的稳定性与准确性。**

- **链接: [http://arxiv.org/pdf/2505.14511v1](http://arxiv.org/pdf/2505.14511v1)**

> **作者:** Guillaume Vray; Devavrat Tomar; Xufeng Gao; Jean-Philippe Thiran; Evan Shelhamer; Behzad Bozorgtabar
>
> **摘要:** This paper introduces ReservoirTTA, a novel plug-in framework designed for prolonged test-time adaptation (TTA) in scenarios where the test domain continuously shifts over time, including cases where domains recur or evolve gradually. At its core, ReservoirTTA maintains a reservoir of domain-specialized models -- an adaptive test-time model ensemble -- that both detects new domains via online clustering over style features of incoming samples and routes each sample to the appropriate specialized model, and thereby enables domain-specific adaptation. This multi-model strategy overcomes key limitations of single model adaptation, such as catastrophic forgetting, inter-domain interference, and error accumulation, ensuring robust and stable performance on sustained non-stationary test distributions. Our theoretical analysis reveals key components that bound parameter variance and prevent model collapse, while our plug-in TTA module mitigates catastrophic forgetting of previously encountered domains. Extensive experiments on the classification corruption benchmarks, including ImageNet-C and CIFAR-10/100-C, as well as the Cityscapes$\rightarrow$ACDC semantic segmentation task, covering recurring and continuously evolving domain shifts, demonstrate that ReservoirTTA significantly improves adaptation accuracy and maintains stable performance across prolonged, recurring shifts, outperforming state-of-the-art methods.
>
---
#### [new 018] Every Pixel Tells a Story: End-to-End Urdu Newspaper OCR
- **分类: cs.CV**

- **简介: 该论文提出端到端乌尔都语报纸OCR系统，解决复杂多栏布局、低分辨率扫描和多样字体问题。工作包括：文章分割（YOLOv11x模型，精度0.963）、图像超分辨率（SwinIR模型，32.71dB PSNR）、列分割（YOLOv11x，精度0.970）及文本识别（测试Gemini、GPT等LLM，Gemini-2.5-Pro WER最低0.133）。**

- **链接: [http://arxiv.org/pdf/2505.13943v1](http://arxiv.org/pdf/2505.13943v1)**

> **作者:** Samee Arif; Sualeha Farid
>
> **摘要:** This paper introduces a comprehensive end-to-end pipeline for Optical Character Recognition (OCR) on Urdu newspapers. In our approach, we address the unique challenges of complex multi-column layouts, low-resolution archival scans, and diverse font styles. Our process decomposes the OCR task into four key modules: (1) article segmentation, (2) image super-resolution, (3) column segmentation, and (4) text recognition. For article segmentation, we fine-tune and evaluate YOLOv11x to identify and separate individual articles from cluttered layouts. Our model achieves a precision of 0.963 and mAP@50 of 0.975. For super-resolution, we fine-tune and benchmark the SwinIR model (reaching 32.71 dB PSNR) to enhance the quality of degraded newspaper scans. To do our column segmentation, we use YOLOv11x to separate columns in text to further enhance performance - this model reaches a precision of 0.970 and mAP@50 of 0.975. In the text recognition stage, we benchmark a range of LLMs from different families, including Gemini, GPT, Llama, and Claude. The lowest WER of 0.133 is achieved by Gemini-2.5-Pro.
>
---
#### [new 019] VoQA: Visual-only Question Answering
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出纯视觉问答任务VoQA，要求模型仅通过图像中嵌入的文本问题进行推理。针对现有大视觉语言模型性能不足的问题，提出GRT-SFT微调策略，通过结构化训练引导模型完成视觉定位、识别及推理，提升复杂多模态场景下的视觉理解能力。**

- **链接: [http://arxiv.org/pdf/2505.14227v1](http://arxiv.org/pdf/2505.14227v1)**

> **作者:** Luyang Jiang; Jianing An; Jie Luo; Wenjun Wu; Lei Huang
>
> **备注:** 18 pages
>
> **摘要:** We propose Visual-only Question Answering (VoQA), a novel multimodal task in which questions are visually embedded within images, without any accompanying textual input. This requires models to locate, recognize, and reason over visually embedded textual questions, posing challenges for existing large vision-language models (LVLMs), which show notable performance drops even with carefully designed prompts. To bridge this gap, we introduce Guided Response Triggering Supervised Fine-tuning (GRT-SFT), a structured fine-tuning strategy that guides the model to perform step-by-step reasoning purely based on visual input, significantly improving model performance. Our work enhances models' capacity for human-like visual understanding in complex multimodal scenarios, where information, including language, is perceived visually.
>
---
#### [new 020] Egocentric Action-aware Inertial Localization in Point Clouds
- **分类: cs.CV**

- **简介: 该论文提出EAIL框架，解决头戴IMU传感器因噪声和动作多样性导致的3D点云人体定位漂移问题。通过学习IMU动作特征与环境点云的关联，利用动作与空间结构的关联作为锚点，实现鲁棒惯性定位，并副产品动作识别。**

- **链接: [http://arxiv.org/pdf/2505.14346v1](http://arxiv.org/pdf/2505.14346v1)**

> **作者:** Mingfang Zhang; Ryo Yonetani; Yifei Huang; Liangyang Ouyang; Ruicong Liu; Yoichi Sato
>
> **摘要:** This paper presents a novel inertial localization framework named Egocentric Action-aware Inertial Localization (EAIL), which leverages egocentric action cues from head-mounted IMU signals to localize the target individual within a 3D point cloud. Human inertial localization is challenging due to IMU sensor noise that causes trajectory drift over time. The diversity of human actions further complicates IMU signal processing by introducing various motion patterns. Nevertheless, we observe that some actions observed through the head-mounted IMU correlate with spatial environmental structures (e.g., bending down to look inside an oven, washing dishes next to a sink), thereby serving as spatial anchors to compensate for the localization drift. The proposed EAIL framework learns such correlations via hierarchical multi-modal alignment. By assuming that the 3D point cloud of the environment is available, it contrastively learns modality encoders that align short-term egocentric action cues in IMU signals with local environmental features in the point cloud. These encoders are then used in reasoning the IMU data and the point cloud over time and space to perform inertial localization. Interestingly, these encoders can further be utilized to recognize the corresponding sequence of actions as a by-product. Extensive experiments demonstrate the effectiveness of the proposed framework over state-of-the-art inertial localization and inertial action recognition baselines.
>
---
#### [new 021] diffDemorph: Extending Reference-Free Demorphing to Unseen Faces
- **分类: cs.CV**

- **简介: 该论文属于无参考人脸分离任务，旨在解决现有方法对训练测试数据分布（如融合技术、人脸风格）依赖性强的问题。提出diffDemorph，基于扩散模型首次实现跨技术与风格的高保真分离，通过合成数据训练并验证于真实数据，性能提升≥59.46%，六数据集和两种人脸识别器验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.14527v1](http://arxiv.org/pdf/2505.14527v1)**

> **作者:** Nitish Shukla; Arun Ross
>
> **摘要:** A face morph is created by combining two (or more) face images corresponding to two (or more) identities to produce a composite that successfully matches the constituent identities. Reference-free (RF) demorphing reverses this process using only the morph image, without the need for additional reference images. Previous RF demorphing methods were overly constrained, as they rely on assumptions about the distributions of training and testing morphs such as the morphing technique used, face style, and images used to create the morph. In this paper, we introduce a novel diffusion-based approach that effectively disentangles component images from a composite morph image with high visual fidelity. Our method is the first to generalize across morph techniques and face styles, beating the current state of the art by $\geq 59.46\%$ under a common training protocol across all datasets tested. We train our method on morphs created using synthetically generated face images and test on real morphs, thereby enhancing the practicality of the technique. Experiments on six datasets and two face matchers establish the utility and efficacy of our method.
>
---
#### [new 022] RA-Touch: Retrieval-Augmented Touch Understanding with Enriched Visual Data
- **分类: cs.CV**

- **简介: 该论文属于视觉触觉感知任务，旨在解决触觉数据采集成本高及如何利用视觉数据中的材质线索提升触觉理解的问题。提出RA-Touch框架，通过为视觉数据添加触觉描述并检索与触觉输入匹配的视觉-文本表征，整合多模态信息提升材质识别，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14270v1](http://arxiv.org/pdf/2505.14270v1)**

> **作者:** Yoorhim Cho; Hongyeob Kim; Semin Kim; Youjia Zhang; Yunseok Choi; Sungeun Hong
>
> **摘要:** Visuo-tactile perception aims to understand an object's tactile properties, such as texture, softness, and rigidity. However, the field remains underexplored because collecting tactile data is costly and labor-intensive. We observe that visually distinct objects can exhibit similar surface textures or material properties. For example, a leather sofa and a leather jacket have different appearances but share similar tactile properties. This implies that tactile understanding can be guided by material cues in visual data, even without direct tactile supervision. In this paper, we introduce RA-Touch, a retrieval-augmented framework that improves visuo-tactile perception by leveraging visual data enriched with tactile semantics. We carefully recaption a large-scale visual dataset with tactile-focused descriptions, enabling the model to access tactile semantics typically absent from conventional visual datasets. A key challenge remains in effectively utilizing these tactile-aware external descriptions. RA-Touch addresses this by retrieving visual-textual representations aligned with tactile inputs and integrating them to focus on relevant textural and material properties. By outperforming prior methods on the TVL benchmark, our method demonstrates the potential of retrieval-based visual reuse for tactile understanding. Code is available at https://aim-skku.github.io/RA-Touch
>
---
#### [new 023] An Explorative Analysis of SVM Classifier and ResNet50 Architecture on African Food Classification
- **分类: cs.CV**

- **简介: 该论文属于非洲食物分类任务，旨在解决现有食物识别技术在非洲 cuisine 应用不足的问题。研究通过对比微调的ResNet50深度模型与SVM传统方法，基于1658张六类非洲食物图像，采用五项评估指标分析两种算法的优劣，为非洲食品识别技术发展提供实证支持。**

- **链接: [http://arxiv.org/pdf/2505.13923v1](http://arxiv.org/pdf/2505.13923v1)**

> **作者:** Chinedu Emmanuel Mbonu; Kenechukwu Anigbogu; Doris Asogwa; Tochukwu Belonwu
>
> **备注:** 7 pages, 9 figures
>
> **摘要:** Food recognition systems has advanced significantly for Western cuisines, yet its application to African foods remains underexplored. This study addresses this gap by evaluating both deep learning and traditional machine learning methods for African food classification. We compared the performance of a fine-tuned ResNet50 model with a Support Vector Machine (SVM) classifier. The dataset comprises 1,658 images across six selected food categories that are known in Africa. To assess model effectiveness, we utilize five key evaluation metrics: Confusion matrix, F1-score, accuracy, recall and precision. Our findings offer valuable insights into the strengths and limitations of both approaches, contributing to the advancement of food recognition for African cuisines.
>
---
#### [new 024] SuperMapNet for Long-Range and High-Accuracy Vectorized HD Map Construction
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶高精地图构建任务。针对单模态感知局限及多模态融合不足导致的特征范围有限、元素分类定位精度低问题，提出SuperMapNet：通过交叉注意力与流式对齐模块融合相机与LiDAR数据生成长距BEV特征，并利用三级交互机制（点间、元素间、点元素交互）提升要素分类与定位精度。**

- **链接: [http://arxiv.org/pdf/2505.13856v1](http://arxiv.org/pdf/2505.13856v1)**

> **作者:** Ruqin Zhou; San Jiang; Wanshou Jiang; Yongsheng Zhang; Chenguang Dai
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Vectorized HD map is essential for autonomous driving. Remarkable work has been achieved in recent years, but there are still major issues: (1) in the generation of the BEV features, single modality-based methods are of limited perception capability, while direct concatenation-based multi-modal methods fail to capture synergies and disparities between different modalities, resulting in limited ranges with feature holes; (2) in the classification and localization of map elements, only point information is used without the consideration of element infor-mation and neglects the interaction between point information and element information, leading to erroneous shapes and element entanglement with low accuracy. To address above issues, we introduce SuperMapNet for long-range and high-accuracy vectorized HD map construction. It uses both camera images and LiDAR point clouds as input, and first tightly couple semantic information from camera images and geometric information from LiDAR point clouds by a cross-attention based synergy enhancement module and a flow-based disparity alignment module for long-range BEV feature generation. And then, local features from point queries and global features from element queries are tightly coupled by three-level interactions for high-accuracy classification and localization, where Point2Point interaction learns local geometric information between points of the same element and of each point, Element2Element interaction learns relation constraints between different elements and semantic information of each elements, and Point2Element interaction learns complement element information for its constituent points. Experiments on the nuScenes and Argoverse2 datasets demonstrate superior performances, surpassing SOTAs over 14.9/8.8 mAP and 18.5/3.1 mAP under hard/easy settings, respectively. The code is made publicly available1.
>
---
#### [new 025] Blind Restoration of High-Resolution Ultrasound Video
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于超声视频盲超分辨率任务。针对超声视频低信噪比、分辨率不足及设备差异导致模型泛化性差的问题，提出自监督算法DUP，通过视频自适应优化过程实现无配对数据的超分辨率与降噪，提升诊断效果。**

- **链接: [http://arxiv.org/pdf/2505.13915v1](http://arxiv.org/pdf/2505.13915v1)**

> **作者:** Chu Chen; Kangning Cui; Pasquale Cascarano; Wei Tang; Elena Loli Piccolomini; Raymond H. Chan
>
> **摘要:** Ultrasound imaging is widely applied in clinical practice, yet ultrasound videos often suffer from low signal-to-noise ratios (SNR) and limited resolutions, posing challenges for diagnosis and analysis. Variations in equipment and acquisition settings can further exacerbate differences in data distribution and noise levels, reducing the generalizability of pre-trained models. This work presents a self-supervised ultrasound video super-resolution algorithm called Deep Ultrasound Prior (DUP). DUP employs a video-adaptive optimization process of a neural network that enhances the resolution of given ultrasound videos without requiring paired training data while simultaneously removing noise. Quantitative and visual evaluations demonstrate that DUP outperforms existing super-resolution algorithms, leading to substantial improvements for downstream applications.
>
---
#### [new 026] Investigating and Enhancing the Robustness of Large Multimodal Models Against Temporal Inconsistency
- **分类: cs.CV**

- **简介: 该论文属于多模态模型鲁棒性任务，针对其在时间不一致扰动下过度依赖文本和先验知识、忽视视频时序动态的问题，提出TemRobBench基准评估模型，并设计PanoDPO方法，结合视觉与语言特征提升鲁棒性，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.14405v1](http://arxiv.org/pdf/2505.14405v1)**

> **作者:** Jiafeng Liang; Shixin Jiang; Xuan Dong; Ning Wang; Zheng Chu; Hui Su; Jinlan Fu; Ming Liu; See-Kiong Ng; Bing Qin
>
> **摘要:** Large Multimodal Models (LMMs) have recently demonstrated impressive performance on general video comprehension benchmarks. Nevertheless, for broader applications, the robustness of their temporal analysis capability needs to be thoroughly investigated yet predominantly ignored. Motivated by this, we propose a novel temporal robustness benchmark (TemRobBench), which introduces temporal inconsistency perturbations separately at the visual and textual modalities to assess the robustness of models. We evaluate 16 mainstream LMMs and find that they exhibit over-reliance on prior knowledge and textual context in adversarial environments, while ignoring the actual temporal dynamics in the video. To mitigate this issue, we design panoramic direct preference optimization (PanoDPO), which encourages LMMs to incorporate both visual and linguistic feature preferences simultaneously. Experimental results show that PanoDPO can effectively enhance the model's robustness and reliability in temporal analysis.
>
---
#### [new 027] Hunyuan-Game: Industrial-grade Intelligent Game Creation Model
- **分类: cs.CV**

- **简介: 该论文提出Hunyuan-Game模型，属于智能游戏创作任务，旨在解决高质量游戏资产生成与提升设计效率问题。通过图像生成（文本到图像、视觉效果、透明图像、角色生成）和视频生成（图像转视频、360姿态合成、动态插图、超分辨率、交互视频）两大分支，基于大规模数据开发定制模型，融合领域知识，实现高保真游戏内容生成。**

- **链接: [http://arxiv.org/pdf/2505.14135v1](http://arxiv.org/pdf/2505.14135v1)**

> **作者:** Ruihuang Li; Caijin Zhou; Shoujian Zheng; Jianxiang Lu; Jiabin Huang; Comi Chen; Junshu Tang; Guangzheng Xu; Jiale Tao; Hongmei Wang; Donghao Li; Wenqing Yu; Senbo Wang; Zhimin Li; Yetshuan Shi; Haoyu Yang; Yukun Wang; Wenxun Dai; Jiaqi Li; Linqing Wang; Qixun Wang; Zhiyong Xu; Yingfang Zhang; Jiangfeng Xiong; Weijie Kong; Chao Zhang; Hongxin Zhang; Qiaoling Zheng; Weiting Guo; Xinchi Deng; Yixuan Li; Renjia Wei; Yulin Jian; Duojun Huang; Xuhua Ren; Sihuan Lin; Yifu Sun; Yuan Zhou; Joey Wang; Qin Lin; Jingmiao Yu; Jihong Zhang; Caesar Zhong; Di Wang; Yuhong Liu; Linus; Jie Jiang; Longhuang Wu; Shuai Shao; Qinglin Lu
>
> **摘要:** Intelligent game creation represents a transformative advancement in game development, utilizing generative artificial intelligence to dynamically generate and enhance game content. Despite notable progress in generative models, the comprehensive synthesis of high-quality game assets, including both images and videos, remains a challenging frontier. To create high-fidelity game content that simultaneously aligns with player preferences and significantly boosts designer efficiency, we present Hunyuan-Game, an innovative project designed to revolutionize intelligent game production. Hunyuan-Game encompasses two primary branches: image generation and video generation. The image generation component is built upon a vast dataset comprising billions of game images, leading to the development of a group of customized image generation models tailored for game scenarios: (1) General Text-to-Image Generation. (2) Game Visual Effects Generation, involving text-to-effect and reference image-based game visual effect generation. (3) Transparent Image Generation for characters, scenes, and game visual effects. (4) Game Character Generation based on sketches, black-and-white images, and white models. The video generation component is built upon a comprehensive dataset of millions of game and anime videos, leading to the development of five core algorithmic models, each targeting critical pain points in game development and having robust adaptation to diverse game video scenarios: (1) Image-to-Video Generation. (2) 360 A/T Pose Avatar Video Synthesis. (3) Dynamic Illustration Generation. (4) Generative Video Super-Resolution. (5) Interactive Game Video Generation. These image and video generation models not only exhibit high-level aesthetic expression but also deeply integrate domain-specific knowledge, establishing a systematic understanding of diverse game and anime art styles.
>
---
#### [new 028] Aligning Attention Distribution to Information Flow for Hallucination Mitigation in Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型图像描述任务中的幻觉问题，提出通过优化注意力机制对齐信息流与注意力分布。发现模型注意力过度关注视觉表示，而语义已吸收视觉信息，导致幻觉。方法识别核心语义注意力头并优化传播，减少幻觉，实验显示有效且支持保守度调节。**

- **链接: [http://arxiv.org/pdf/2505.14257v1](http://arxiv.org/pdf/2505.14257v1)**

> **作者:** Jianfei Zhao; Feng Zhang; Xin Sun; Chong Feng
>
> **摘要:** Due to the unidirectional masking mechanism, Decoder-Only models propagate information from left to right. LVLMs (Large Vision-Language Models) follow the same architecture, with visual information gradually integrated into semantic representations during forward propagation. Through systematic analysis, we observe that over 80\% of the visual information is absorbed into the semantic representations. However, the model's attention still predominantly focuses on the visual representations. This misalignment between the attention distribution and the actual information flow undermines the model's visual understanding ability and contributes to hallucinations. To address this issue, we enhance the model's visual understanding by leveraging the core information embedded in semantic representations. Specifically, we identify attention heads that focus on core semantic representations based on their attention distributions. Then, through a two-stage optimization paradigm, we propagate the advantages of these attention heads across the entire model, aligning the attention distribution with the actual information flow. We evaluate our method on three image captioning benchmarks using five different LVLMs, demonstrating its effectiveness in significantly reducing hallucinations. Further experiments reveal a trade-off between reduced hallucinations and richer details. Notably, our method allows for manual adjustment of the model's conservativeness, enabling flexible control to meet diverse real-world requirements. Code will be released once accepted.
>
---
#### [new 029] Enhancing Interpretability of Sparse Latent Representations with Class Information
- **分类: cs.CV; cs.LG**

- **简介: 该论文属提升VAE潜在表示可解释性任务。针对VSC方法未能让同类样本保持活跃维度一致的问题，提出新损失函数，通过强制同类样本共享相似活跃维度，构建结构化潜在空间，同时捕捉全局与类特定因素，增强解释性。**

- **链接: [http://arxiv.org/pdf/2505.14476v1](http://arxiv.org/pdf/2505.14476v1)**

> **作者:** Farshad Sangari Abiz; Reshad Hosseini; Babak N. Araabi
>
> **摘要:** Variational Autoencoders (VAEs) are powerful generative models for learning latent representations. Standard VAEs generate dispersed and unstructured latent spaces by utilizing all dimensions, which limits their interpretability, especially in high-dimensional spaces. To address this challenge, Variational Sparse Coding (VSC) introduces a spike-and-slab prior distribution, resulting in sparse latent representations for each input. These sparse representations, characterized by a limited number of active dimensions, are inherently more interpretable. Despite this advantage, VSC falls short in providing structured interpretations across samples within the same class. Intuitively, samples from the same class are expected to share similar attributes while allowing for variations in those attributes. This expectation should manifest as consistent patterns of active dimensions in their latent representations, but VSC does not enforce such consistency. In this paper, we propose a novel approach to enhance the latent space interpretability by ensuring that the active dimensions in the latent space are consistent across samples within the same class. To achieve this, we introduce a new loss function that encourages samples from the same class to share similar active dimensions. This alignment creates a more structured and interpretable latent space, where each shared dimension corresponds to a high-level concept, or "factor." Unlike existing disentanglement-based methods that primarily focus on global factors shared across all classes, our method captures both global and class-specific factors, thereby enhancing the utility and interpretability of latent representations.
>
---
#### [new 030] UniGen: Enhanced Training & Test-Time Strategies for Unified Multimodal Understanding and Generation
- **分类: cs.CV**

- **简介: 该论文提出UniGen模型，属于统一多模态理解与生成任务。旨在提升多模态模型的图像生成质量与语义一致性。工作包括：1）通过多阶段预训练、监督微调及偏好优化改进训练流程；2）提出测试时Chain-of-Thought Verification（CoT-V）策略，使模型自动生成并验证图像与文本的语义匹配性，显著提升生成质量。实验显示其在GenEval和DPG-Bench取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2505.14682v1](http://arxiv.org/pdf/2505.14682v1)**

> **作者:** Rui Tian; Mingfei Gao; Mingze Xu; Jiaming Hu; Jiasen Lu; Zuxuan Wu; Yinfei Yang; Afshin Dehghan
>
> **备注:** Technical report
>
> **摘要:** We introduce UniGen, a unified multimodal large language model (MLLM) capable of image understanding and generation. We study the full training pipeline of UniGen from a data-centric perspective, including multi-stage pre-training, supervised fine-tuning, and direct preference optimization. More importantly, we propose a new Chain-of-Thought Verification (CoT-V) strategy for test-time scaling, which significantly boosts UniGen's image generation quality using a simple Best-of-N test-time strategy. Specifically, CoT-V enables UniGen to act as both image generator and verifier at test time, assessing the semantic alignment between a text prompt and its generated image in a step-by-step CoT manner. Trained entirely on open-source datasets across all stages, UniGen achieves state-of-the-art performance on a range of image understanding and generation benchmarks, with a final score of 0.78 on GenEval and 85.19 on DPG-Bench. Through extensive ablation studies, our work provides actionable insights and addresses key challenges in the full life cycle of building unified MLLMs, contributing meaningful directions to the future research.
>
---
#### [new 031] Instance Segmentation for Point Sets
- **分类: cs.CV; cs.LG; 68T45; I.2.10**

- **简介: 该论文属于3D点云实例分割任务，旨在解决SGPN方法中内存密集型相似度矩阵问题。提出两种采样方法，通过子采样点集进行实例分割后，用最近邻传播标签至全点云。随机采样策略在速度和内存效率上表现更优。**

- **链接: [http://arxiv.org/pdf/2505.14583v1](http://arxiv.org/pdf/2505.14583v1)**

> **作者:** Abhimanyu Talwar; Julien Laasri
>
> **备注:** 6 pages, 11 figures, paper dated 2019
>
> **摘要:** Recently proposed neural network architectures like PointNet [QSMG16] and PointNet++ [QYSG17] have made it possible to apply Deep Learning to 3D point sets. The feature representations of shapes learned by these two networks enabled training classifiers for Semantic Segmentation, and more recently for Instance Segmentation via the Similarity Group Proposal Network (SGPN) [WYHN17]. One area of improvement which has been highlighted by SGPN's authors, pertains to use of memory intensive similarity matrices which occupy memory quadratic in the number of points. In this report, we attempt to tackle this issue through use of two sampling based methods, which compute Instance Segmentation on a sub-sampled Point Set, and then extrapolate labels to the complete set using the nearest neigbhour approach. While both approaches perform equally well on large sub-samples, the random-based strategy gives the most improvements in terms of speed and memory usage.
>
---
#### [new 032] Domain Adaptation for Multi-label Image Classification: a Discriminator-free Approach
- **分类: cs.CV**

- **简介: 该论文针对多标签图像分类的无监督领域自适应任务，提出无判别器方法DDA-MLIC，解决传统对抗方法因添加判别器 subnet 导致分类任务鉴别力下降的问题。通过GMM建模源/目标预测分布，利用DNN估计参数并基于Fréchet距离构建对抗损失，实现高效参数优化，精度更高且参数更少。**

- **链接: [http://arxiv.org/pdf/2505.14333v1](http://arxiv.org/pdf/2505.14333v1)**

> **作者:** Inder Pal Singh; Enjie Ghorbel; Anis Kacem; Djamila Aouada
>
> **备注:** The paper is under consideration at Computer Vision and Image Understanding. arXiv admin note: text overlap with arXiv:2301.10611
>
> **摘要:** This paper introduces a discriminator-free adversarial-based approach termed DDA-MLIC for Unsupervised Domain Adaptation (UDA) in the context of Multi-Label Image Classification (MLIC). While recent efforts have explored adversarial-based UDA methods for MLIC, they typically include an additional discriminator subnet. Nevertheless, decoupling the classification and the discrimination tasks may harm their task-specific discriminative power. Herein, we address this challenge by presenting a novel adversarial critic directly derived from the task-specific classifier. Specifically, we employ a two-component Gaussian Mixture Model (GMM) to model both source and target predictions, distinguishing between two distinct clusters. Instead of using the traditional Expectation Maximization (EM) algorithm, our approach utilizes a Deep Neural Network (DNN) to estimate the parameters of each GMM component. Subsequently, the source and target GMM parameters are leveraged to formulate an adversarial loss using the Fr\'echet distance. The proposed framework is therefore not only fully differentiable but is also cost-effective as it avoids the expensive iterative process usually induced by the standard EM method. The proposed method is evaluated on several multi-label image datasets covering three different types of domain shift. The obtained results demonstrate that DDA-MLIC outperforms existing state-of-the-art methods in terms of precision while requiring a lower number of parameters. The code is made publicly available at github.com/cvi2snt/DDA-MLIC.
>
---
#### [new 033] Decoupling Classifier for Boosting Few-shot Object Detection and Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文针对少样本目标检测（FSOD）和实例分割（FSIS）任务，解决实例级标签缺失导致的分类偏差问题。提出解耦分类器，将标准分类头分为两个独立模块分别处理正样本与噪声负样本，提升模型鲁棒性。方法简单且无需额外计算，在主流数据集上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14239v1](http://arxiv.org/pdf/2505.14239v1)**

> **作者:** Bin-Bin Gao; Xiaochen Chen; Zhongyi Huang; Congchong Nie; Jun Liu; Jinxiang Lai; Guannan Jiang; Xi Wang; Chengjie Wang
>
> **备注:** Accepted by NeurIPS 2022
>
> **摘要:** This paper focus on few-shot object detection~(FSOD) and instance segmentation~(FSIS), which requires a model to quickly adapt to novel classes with a few labeled instances. The existing methods severely suffer from bias classification because of the missing label issue which naturally exists in an instance-level few-shot scenario and is first formally proposed by us. Our analysis suggests that the standard classification head of most FSOD or FSIS models needs to be decoupled to mitigate the bias classification. Therefore, we propose an embarrassingly simple but effective method that decouples the standard classifier into two heads. Then, these two individual heads are capable of independently addressing clear positive samples and noisy negative samples which are caused by the missing label. In this way, the model can effectively learn novel classes while mitigating the effects of noisy negative samples. Without bells and whistles, our model without any additional computation cost and parameters consistently outperforms its baseline and state-of-the-art by a large margin on PASCAL VOC and MS-COCO benchmarks for FSOD and FSIS tasks. The Code is available at https://csgaobb.github.io/Projects/DCFS.
>
---
#### [new 034] Video Compression Commander: Plug-and-Play Inference Acceleration for Video Large Language Models
- **分类: cs.CV**

- **简介: 该论文提出视频压缩框架VidCom2，解决视频大语言模型（VideoLLM）因视觉标记过多导致的推理效率低问题。针对现有方法的信息丢失和兼容性缺陷，通过量化帧独特性自适应压缩，减少70.8%延迟且保持99.6%性能，兼容其他压缩方法。**

- **链接: [http://arxiv.org/pdf/2505.14454v1](http://arxiv.org/pdf/2505.14454v1)**

> **作者:** Xuyang Liu; Yiyu Wang; Junpeng Ma; Linfeng Zhang
>
> **备注:** Our code is available at https://github.com/xuyang-liu16/VidCom2
>
> **摘要:** Video large language models (VideoLLM) excel at video understanding, but face efficiency challenges due to the quadratic complexity of abundant visual tokens. Our systematic analysis of token compression methods for VideoLLMs reveals two critical issues: (i) overlooking distinctive visual signals across frames, leading to information loss; (ii) suffering from implementation constraints, causing incompatibility with modern architectures or efficient operators. To address these challenges, we distill three design principles for VideoLLM token compression and propose a plug-and-play inference acceleration framework "Video Compression Commander" (VidCom2). By quantifying each frame's uniqueness, VidCom2 adaptively adjusts compression intensity across frames, effectively preserving essential information while reducing redundancy in video sequences. Extensive experiments across various VideoLLMs and benchmarks demonstrate the superior performance and efficiency of our VidCom2. With only 25% visual tokens, VidCom2 achieves 99.6% of the original performance on LLaVA-OV while reducing 70.8% of the LLM generation latency. Notably, our Frame Compression Adjustment strategy is compatible with other token compression methods to further improve their performance. Our code is available at https://github.com/xuyang-liu16/VidCom2.
>
---
#### [new 035] Learning Concept-Driven Logical Rules for Interpretable and Generalizable Medical Image Classification
- **分类: cs.CV**

- **简介: 该论文属于医疗图像分类任务，旨在解决概念泄漏及缺乏全局决策逻辑的问题。提出CRL框架，通过学习二值化视觉概念的布尔逻辑规则，捕捉概念关联，提供局部与全局可解释性，提升模型在分布外数据中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.14049v1](http://arxiv.org/pdf/2505.14049v1)**

> **作者:** Yibo Gao; Hangqi Zhou; Zheyao Gao; Bomin Wang; Shangqi Gao; Sihan Wang; Xiahai Zhuang
>
> **备注:** early accepted by MICCAI 2025
>
> **摘要:** The pursuit of decision safety in clinical applications highlights the potential of concept-based methods in medical imaging. While these models offer active interpretability, they often suffer from concept leakages, where unintended information within soft concept representations undermines both interpretability and generalizability. Moreover, most concept-based models focus solely on local explanations (instance-level), neglecting the global decision logic (dataset-level). To address these limitations, we propose Concept Rule Learner (CRL), a novel framework to learn Boolean logical rules from binarized visual concepts. CRL employs logical layers to capture concept correlations and extract clinically meaningful rules, thereby providing both local and global interpretability. Experiments on two medical image classification tasks show that CRL achieves competitive performance with existing methods while significantly improving generalizability to out-of-distribution data. The code of our work is available at https://github.com/obiyoag/crl.
>
---
#### [new 036] Sat2Sound: A Unified Framework for Zero-Shot Soundscape Mapping
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: Sat2Sound提出零样本声音景观映射框架，解决现有方法依赖配对数据难以捕捉声音多样性的问题。通过VLM生成卫星图像的语义声音描述，结合跨模态对比学习，构建共享声音概念代码本，提升卫星图像与音频的跨模态检索性能，并实现基于位置的声音合成应用。**

- **链接: [http://arxiv.org/pdf/2505.13777v1](http://arxiv.org/pdf/2505.13777v1)**

> **作者:** Subash Khanal; Srikumar Sastry; Aayush Dhakal; Adeel Ahmad; Nathan Jacobs
>
> **摘要:** We present Sat2Sound, a multimodal representation learning framework for soundscape mapping, designed to predict the distribution of sounds at any location on Earth. Existing methods for this task rely on satellite image and paired geotagged audio samples, which often fail to capture the diversity of sound sources at a given location. To address this limitation, we enhance existing datasets by leveraging a Vision-Language Model (VLM) to generate semantically rich soundscape descriptions for locations depicted in satellite images. Our approach incorporates contrastive learning across audio, audio captions, satellite images, and satellite image captions. We hypothesize that there is a fixed set of soundscape concepts shared across modalities. To this end, we learn a shared codebook of soundscape concepts and represent each sample as a weighted average of these concepts. Sat2Sound achieves state-of-the-art performance in cross-modal retrieval between satellite image and audio on two datasets: GeoSound and SoundingEarth. Additionally, building on Sat2Sound's ability to retrieve detailed soundscape captions, we introduce a novel application: location-based soundscape synthesis, which enables immersive acoustic experiences. Our code and models will be publicly available.
>
---
#### [new 037] Frozen Backpropagation: Relaxing Weight Symmetry in Temporally-Coded Deep Spiking Neural Networks
- **分类: cs.CV; cs.NE**

- **简介: 论文提出Frozen Backpropagation（fBP），用于神经形态硬件上SNN的高效训练。解决传统BP需权重对称导致高能耗问题，通过周期冻结反馈权重并采用部分传输方案，减少同步开销。实验显示其在CIFAR数据集上降低传输成本千倍，精度仅小幅下降，属SNN训练优化任务。**

- **链接: [http://arxiv.org/pdf/2505.13741v1](http://arxiv.org/pdf/2505.13741v1)**

> **作者:** Gaspard Goupy; Pierre Tirilly; Ioan Marius Bilasco
>
> **摘要:** Direct training of Spiking Neural Networks (SNNs) on neuromorphic hardware can greatly reduce energy costs compared to GPU-based training. However, implementing Backpropagation (BP) on such hardware is challenging because forward and backward passes are typically performed by separate networks with distinct weights. To compute correct gradients, forward and feedback weights must remain symmetric during training, necessitating weight transport between the two networks. This symmetry requirement imposes hardware overhead and increases energy costs. To address this issue, we introduce Frozen Backpropagation (fBP), a BP-based training algorithm relaxing weight symmetry in settings with separate networks. fBP updates forward weights by computing gradients with periodically frozen feedback weights, reducing weight transports during training and minimizing synchronization overhead. To further improve transport efficiency, we propose three partial weight transport schemes of varying computational complexity, where only a subset of weights is transported at a time. We evaluate our methods on image recognition tasks and compare them to existing approaches addressing the weight symmetry requirement. Our results show that fBP outperforms these methods and achieves accuracy comparable to BP. With partial weight transport, fBP can substantially lower transport costs by 1,000x with an accuracy drop of only 0.5pp on CIFAR-10 and 1.1pp on CIFAR-100, or by up to 10,000x at the expense of moderated accuracy loss. This work provides insights for guiding the design of neuromorphic hardware incorporating BP-based on-chip learning.
>
---
#### [new 038] Personalize Your Gaussian: Consistent 3D Scene Personalization from a Single Image
- **分类: cs.CV**

- **简介: 该论文提出CP-GS框架，解决单图驱动3D场景个性化中的视角偏差问题。通过预训练图像-3D生成模型与迭代LoRA微调，扩展单视角参考外观至多视角，并结合几何线索生成一致的3D输出，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14537v1](http://arxiv.org/pdf/2505.14537v1)**

> **作者:** Yuxuan Wang; Xuanyu Yi; Qingshan Xu; Yuan Zhou; Long Chen; Hanwang Zhang
>
> **备注:** 9 pages
>
> **摘要:** Personalizing 3D scenes from a single reference image enables intuitive user-guided editing, which requires achieving both multi-view consistency across perspectives and referential consistency with the input image. However, these goals are particularly challenging due to the viewpoint bias caused by the limited perspective provided in a single image. Lacking the mechanisms to effectively expand reference information beyond the original view, existing methods of image-conditioned 3DGS personalization often suffer from this viewpoint bias and struggle to produce consistent results. Therefore, in this paper, we present Consistent Personalization for 3D Gaussian Splatting (CP-GS), a framework that progressively propagates the single-view reference appearance to novel perspectives. In particular, CP-GS integrates pre-trained image-to-3D generation and iterative LoRA fine-tuning to extract and extend the reference appearance, and finally produces faithful multi-view guidance images and the personalized 3DGS outputs through a view-consistent generation process guided by geometric cues. Extensive experiments on real-world scenes show that our CP-GS effectively mitigates the viewpoint bias, achieving high-quality personalization that significantly outperforms existing methods. The code will be released at https://github.com/Yuxuan-W/CP-GS.
>
---
#### [new 039] DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决视觉语言模型（VLMs）过度依赖文本推理的问题。提出DeepEyes模型，通过强化学习激励"视觉辅助推理"，设计工具导向的数据选择和奖励策略，使模型自主进化出视觉驱动的推理能力，在感知、数学推理等任务中表现提升，行为模式接近人类视觉认知。**

- **链接: [http://arxiv.org/pdf/2505.14362v1](http://arxiv.org/pdf/2505.14362v1)**

> **作者:** Ziwei Zheng; Michael Yang; Jack Hong; Chenxiao Zhao; Guohai Xu; Le Yang; Chao Shen; Xing Yu
>
> **摘要:** Large Vision-Language Models (VLMs) have shown strong capabilities in multimodal understanding and reasoning, yet they are primarily constrained by text-based reasoning processes. However, achieving seamless integration of visual and textual reasoning which mirrors human cognitive processes remains a significant challenge. In particular, effectively incorporating advanced visual input processing into reasoning mechanisms is still an open question. Thus, in this paper, we explore the interleaved multimodal reasoning paradigm and introduce DeepEyes, a model with "thinking with images" capabilities incentivized through end-to-end reinforcement learning without the need for cold-start SFT. Notably, this ability emerges natively within the model itself, leveraging its inherent grounding ability as a tool instead of depending on separate specialized models. Specifically, we propose a tool-use-oriented data selection mechanism and a reward strategy to encourage successful tool-assisted reasoning trajectories. DeepEyes achieves significant performance gains on fine-grained perception and reasoning benchmarks and also demonstrates improvement in grounding, hallucination, and mathematical reasoning tasks. Interestingly, we observe the distinct evolution of tool-calling behavior from initial exploration to efficient and accurate exploitation, and diverse thinking patterns that closely mirror human visual reasoning processes. Code is available at https://github.com/Visual-Agent/DeepEyes.
>
---
#### [new 040] EGFormer: Towards Efficient and Generalizable Multimodal Semantic Segmentation
- **分类: cs.CV**

- **简介: EGFormer提出高效多模态语义分割框架，解决现有方法计算效率低和泛化不足问题。通过Any-modal Scoring Module动态评估模态重要性，结合Modal Dropping Module过滤冗余信息，减少参数（88%）和计算量（50% GFLOPs），同时在跨领域任务中达SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.14014v1](http://arxiv.org/pdf/2505.14014v1)**

> **作者:** Zelin Zhang; Tao Zhang; KediLI; Xu Zheng
>
> **摘要:** Recent efforts have explored multimodal semantic segmentation using various backbone architectures. However, while most methods aim to improve accuracy, their computational efficiency remains underexplored. To address this, we propose EGFormer, an efficient multimodal semantic segmentation framework that flexibly integrates an arbitrary number of modalities while significantly reducing model parameters and inference time without sacrificing performance. Our framework introduces two novel modules. First, the Any-modal Scoring Module (ASM) assigns importance scores to each modality independently, enabling dynamic ranking based on their feature maps. Second, the Modal Dropping Module (MDM) filters out less informative modalities at each stage, selectively preserving and aggregating only the most valuable features. This design allows the model to leverage useful information from all available modalities while discarding redundancy, thus ensuring high segmentation quality. In addition to efficiency, we evaluate EGFormer on a synthetic-to-real transfer task to demonstrate its generalizability. Extensive experiments show that EGFormer achieves competitive performance with up to 88 percent reduction in parameters and 50 percent fewer GFLOPs. Under unsupervised domain adaptation settings, it further achieves state-of-the-art transfer performance compared to existing methods.
>
---
#### [new 041] IPENS:Interactive Unsupervised Framework for Rapid Plant Phenotyping Extraction via NeRF-SAM2 Fusion
- **分类: cs.CV**

- **简介: 论文提出IPENS框架，解决无监督植物表型提取中自遮挡和标注数据依赖问题。通过融合SAM2的2D分割与NeRF的3D辐射场，实现快速多目标点云提取，提升水稻、小麦的分割精度与表型估计，无需标注且交互高效。**

- **链接: [http://arxiv.org/pdf/2505.13633v1](http://arxiv.org/pdf/2505.13633v1)**

> **作者:** Wentao Song; He Huang; Youqiang Sun; Fang Qu; Jiaqi Zhang; Longhui Fang; Yuwei Hao; Chenyang Peng
>
> **摘要:** Advanced plant phenotyping technologies play a crucial role in targeted trait improvement and accelerating intelligent breeding. Due to the species diversity of plants, existing methods heavily rely on large-scale high-precision manually annotated data. For self-occluded objects at the grain level, unsupervised methods often prove ineffective. This study proposes IPENS, an interactive unsupervised multi-target point cloud extraction method. The method utilizes radiance field information to lift 2D masks, which are segmented by SAM2 (Segment Anything Model 2), into 3D space for target point cloud extraction. A multi-target collaborative optimization strategy is designed to effectively resolve the single-interaction multi-target segmentation challenge. Experimental validation demonstrates that IPENS achieves a grain-level segmentation accuracy (mIoU) of 63.72% on a rice dataset, with strong phenotypic estimation capabilities: grain volume prediction yields R2 = 0.7697 (RMSE = 0.0025), leaf surface area R2 = 0.84 (RMSE = 18.93), and leaf length and width predictions achieve R2 = 0.97 and 0.87 (RMSE = 1.49 and 0.21). On a wheat dataset,IPENS further improves segmentation accuracy to 89.68% (mIoU), with equally outstanding phenotypic estimation performance: spike volume prediction achieves R2 = 0.9956 (RMSE = 0.0055), leaf surface area R2 = 1.00 (RMSE = 0.67), and leaf length and width predictions reach R2 = 0.99 and 0.92 (RMSE = 0.23 and 0.15). This method provides a non-invasive, high-quality phenotyping extraction solution for rice and wheat. Without requiring annotated data, it rapidly extracts grain-level point clouds within 3 minutes through simple single-round interactions on images for multiple targets, demonstrating significant potential to accelerate intelligent breeding efficiency.
>
---
#### [new 042] ViC-Bench: Benchmarking Visual-Interleaved Chain-of-Thought Capability in MLLMs with Free-Style Intermediate State Representations
- **分类: cs.CV**

- **简介: 该论文提出ViC-Bench基准，评估多模态大模型（MLLMs）的视觉交织思维链（VI-CoT）能力。针对现有基准使用固定中间视觉状态（IVS）导致推理评估失真的问题，设计包含迷宫导航等四项任务的测试集，支持自由生成IVS，并提出三阶段评估策略与提示注入方法。通过测试18个模型，分析其VI-CoT推理机制，开源工具支持研究。**

- **链接: [http://arxiv.org/pdf/2505.14404v1](http://arxiv.org/pdf/2505.14404v1)**

> **作者:** Xuecheng Wu; Jiaxing Liu; Danlei Huang; Xiaoyu Li; Yifan Wang; Chen Chen; Liya Ma; Xuezhi Cao; Junxiao Xue
>
> **摘要:** Visual-Interleaved Chain-of-Thought (VI-CoT) enables MLLMs to continually update their understanding and decisions based on step-wise intermediate visual states (IVS), much like a human would, which demonstrates impressive success in various tasks, thereby leading to emerged advancements in related benchmarks. Despite promising progress, current benchmarks provide models with relatively fixed IVS, rather than free-style IVS, whch might forcibly distort the original thinking trajectories, failing to evaluate their intrinsic reasoning capabilities. More importantly, existing benchmarks neglect to systematically explore the impact factors that IVS would impart to untamed reasoning performance. To tackle above gaps, we introduce a specialized benchmark termed ViC-Bench, consisting of four representive tasks: maze navigation, jigsaw puzzle, embodied long-horizon planning, and complex counting, where each task has dedicated free-style IVS generation pipeline supporting function calls. To systematically examine VI-CoT capability, we propose a thorough evaluation suite incorporating a progressive three-stage strategy with targeted new metrics. Besides, we establish Incremental Prompting Information Injection (IPII) strategy to ablatively explore the prompting factors for VI-CoT. We extensively conduct evaluations for 18 advanced MLLMs, revealing key insights into their VI-CoT capability. Our proposed benchmark is publicly open at Huggingface.
>
---
#### [new 043] Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting
- **分类: cs.CV**

- **简介: 该论文属于文档图像解析任务，针对现有方法效率低、布局结构差等问题，提出Dolphin模型。其通过分析-解析两阶段流程，先按阅读顺序生成布局元素作为锚点，再结合任务提示并行解析内容，利用超3000万样本训练，在保持高效的同时实现多粒度解析的SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.14059v1](http://arxiv.org/pdf/2505.14059v1)**

> **作者:** Hao Feng; Shu Wei; Xiang Fei; Wei Shi; Yingdong Han; Lei Liao; Jinghui Lu; Binghong Wu; Qi Liu; Chunhui Lin; Jingqun Tang; Hao Liu; Can Huang
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** Document image parsing is challenging due to its complexly intertwined elements such as text paragraphs, figures, formulas, and tables. Current approaches either assemble specialized expert models or directly generate page-level content autoregressively, facing integration overhead, efficiency bottlenecks, and layout structure degradation despite their decent performance. To address these limitations, we present \textit{Dolphin} (\textit{\textbf{Do}cument Image \textbf{P}arsing via \textbf{H}eterogeneous Anchor Prompt\textbf{in}g}), a novel multimodal document image parsing model following an analyze-then-parse paradigm. In the first stage, Dolphin generates a sequence of layout elements in reading order. These heterogeneous elements, serving as anchors and coupled with task-specific prompts, are fed back to Dolphin for parallel content parsing in the second stage. To train Dolphin, we construct a large-scale dataset of over 30 million samples, covering multi-granularity parsing tasks. Through comprehensive evaluations on both prevalent benchmarks and self-constructed ones, Dolphin achieves state-of-the-art performance across diverse page-level and element-level settings, while ensuring superior efficiency through its lightweight architecture and parallel parsing mechanism. The code and pre-trained models are publicly available at https://github.com/ByteDance/Dolphin
>
---
#### [new 044] Breaking Down Video LLM Benchmarks: Knowledge, Spatial Perception, or True Temporal Understanding?
- **分类: cs.CV**

- **简介: 该论文针对视频LLM评估基准混淆知识与时间推理的问题，提出VBenchComp方法，通过自动分类问题为LLM可答（无需视频）、语义（帧可打乱）和时间（需顺序）三类，实现细粒度评估，揭示模型弱点并指导未来基准设计。**

- **链接: [http://arxiv.org/pdf/2505.14321v1](http://arxiv.org/pdf/2505.14321v1)**

> **作者:** Bo Feng; Zhengfeng Lai; Shiyu Li; Zizhen Wang; Simon Wang; Ping Huang; Meng Cao
>
> **摘要:** Existing video understanding benchmarks often conflate knowledge-based and purely image-based questions, rather than clearly isolating a model's temporal reasoning ability, which is the key aspect that distinguishes video understanding from other modalities. We identify two major limitations that obscure whether higher scores truly indicate stronger understanding of the dynamic content in videos: (1) strong language priors, where models can answer questions without watching the video; and (2) shuffling invariance, where models maintain similar performance on certain questions even when video frames are temporally shuffled. To alleviate these issues, we propose VBenchComp, an automated pipeline that categorizes questions into different domains: LLM-Answerable, Semantic, and Temporal. Specifically, LLM-Answerable questions can be answered without viewing the video; Semantic questions remain answerable even when the video frames are shuffled; and Temporal questions require understanding the correct temporal order of frames. The rest of the questions are labeled as Others. This can enable fine-grained evaluation of different capabilities of a video LLM. Our analysis reveals nuanced model weaknesses that are hidden by traditional overall scores, and we offer insights and recommendations for designing future benchmarks that more accurately assess video LLMs.
>
---
#### [new 045] Emerging Properties in Unified Multimodal Pretraining
- **分类: cs.CV**

- **简介: 该论文提出开源模型BAGEL，属于统一多模态预训练任务，旨在提升多模态理解和生成能力。通过大规模跨模态数据预训练，该模型展现复杂推理能力（如图像编辑、3D操作等），性能超越现有开源模型，并公开代码与数据促进研究。**

- **链接: [http://arxiv.org/pdf/2505.14683v1](http://arxiv.org/pdf/2505.14683v1)**

> **作者:** Chaorui Deng; Deyao Zhu; Kunchang Li; Chenhui Gou; Feng Li; Zeyu Wang; Shu Zhong; Weihao Yu; Xiaonan Nie; Ziang Song; Guang Shi; Haoqi Fan
>
> **备注:** 37 pages, 17 figures
>
> **摘要:** Unifying multimodal understanding and generation has shown impressive capabilities in cutting-edge proprietary systems. In this work, we introduce BAGEL, an open0source foundational model that natively supports multimodal understanding and generation. BAGEL is a unified, decoder0only model pretrained on trillions of tokens curated from large0scale interleaved text, image, video, and web data. When scaled with such diverse multimodal interleaved data, BAGEL exhibits emerging capabilities in complex multimodal reasoning. As a result, it significantly outperforms open-source unified models in both multimodal generation and understanding across standard benchmarks, while exhibiting advanced multimodal reasoning abilities such as free-form image manipulation, future frame prediction, 3D manipulation, and world navigation. In the hope of facilitating further opportunities for multimodal research, we share the key findings, pretraining details, data creation protocal, and release our code and checkpoints to the community. The project page is at https://bagel-ai.org/
>
---
#### [new 046] MGStream: Motion-aware 3D Gaussian for Streamable Dynamic Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于流式动态场景重建任务，旨在解决3DGS方法中存在的画面闪烁、存储低效及无法建模新出现物体的问题。提出MGStream方法，通过区分静态和动态的3D高斯体素，对动态部分采用运动掩码、凸 hull 聚类和刚性变形，并结合注意力优化，提升渲染质量、效率及时间一致性。**

- **链接: [http://arxiv.org/pdf/2505.13839v1](http://arxiv.org/pdf/2505.13839v1)**

> **作者:** Zhenyu Bao; Qing Li; Guibiao Liao; Zhongyuan Zhao; Kanglin Liu
>
> **摘要:** 3D Gaussian Splatting (3DGS) has gained significant attention in streamable dynamic novel view synthesis (DNVS) for its photorealistic rendering capability and computational efficiency. Despite much progress in improving rendering quality and optimization strategies, 3DGS-based streamable dynamic scene reconstruction still suffers from flickering artifacts and storage inefficiency, and struggles to model the emerging objects. To tackle this, we introduce MGStream which employs the motion-related 3D Gaussians (3DGs) to reconstruct the dynamic and the vanilla 3DGs for the static. The motion-related 3DGs are implemented according to the motion mask and the clustering-based convex hull algorithm. The rigid deformation is applied to the motion-related 3DGs for modeling the dynamic, and the attention-based optimization on the motion-related 3DGs enables the reconstruction of the emerging objects. As the deformation and optimization are only conducted on the motion-related 3DGs, MGStream avoids flickering artifacts and improves the storage efficiency. Extensive experiments on real-world datasets N3DV and MeetRoom demonstrate that MGStream surpasses existing streaming 3DGS-based approaches in terms of rendering quality, training/storage efficiency and temporal consistency. Our code is available at: https://github.com/pcl3dv/MGStream.
>
---
#### [new 047] LMP: Leveraging Motion Prior in Zero-Shot Video Generation with Diffusion Transformer
- **分类: cs.CV**

- **简介: 该论文属于零样本视频生成任务，旨在解决现有扩散Transformer模型难以通过提示精细控制视频中主体运动（尤其复杂动作）及图像-视频生成中运动不一致的问题。提出LMP框架，通过前景-背景分离、重加权运动转移及外观抑制模块，使生成视频能参考用户提供的运动视频，实现运动控制与主体外观解耦，实验显示其效果达当前最优。**

- **链接: [http://arxiv.org/pdf/2505.14167v1](http://arxiv.org/pdf/2505.14167v1)**

> **作者:** Changgu Chen; Xiaoyan Yang; Junwei Shu; Changbo Wang; Yang Li
>
> **摘要:** In recent years, large-scale pre-trained diffusion transformer models have made significant progress in video generation. While current DiT models can produce high-definition, high-frame-rate, and highly diverse videos, there is a lack of fine-grained control over the video content. Controlling the motion of subjects in videos using only prompts is challenging, especially when it comes to describing complex movements. Further, existing methods fail to control the motion in image-to-video generation, as the subject in the reference image often differs from the subject in the reference video in terms of initial position, size, and shape. To address this, we propose the Leveraging Motion Prior (LMP) framework for zero-shot video generation. Our framework harnesses the powerful generative capabilities of pre-trained diffusion transformers to enable motion in the generated videos to reference user-provided motion videos in both text-to-video and image-to-video generation. To this end, we first introduce a foreground-background disentangle module to distinguish between moving subjects and backgrounds in the reference video, preventing interference in the target video generation. A reweighted motion transfer module is designed to allow the target video to reference the motion from the reference video. To avoid interference from the subject in the reference video, we propose an appearance separation module to suppress the appearance of the reference subject in the target video. We annotate the DAVIS dataset with detailed prompts for our experiments and design evaluation metrics to validate the effectiveness of our method. Extensive experiments demonstrate that our approach achieves state-of-the-art performance in generation quality, prompt-video consistency, and control capability. Our homepage is available at https://vpx-ecnu.github.io/LMP-Website/
>
---
#### [new 048] Flexible-weighted Chamfer Distance: Enhanced Objective Function for Point Cloud Completion
- **分类: cs.CV**

- **简介: 该论文属于点云补全任务，针对传统Chamfer Distance（CD）固定权重导致全局分布差的问题，提出Flexible-Weighted Chamfer Distance（FCD）。通过动态提高全局分布组件的权重，平衡全局与局部性能，提升生成质量。实验显示其在CD、EMD等指标及人评中表现更优。**

- **链接: [http://arxiv.org/pdf/2505.14218v1](http://arxiv.org/pdf/2505.14218v1)**

> **作者:** Jie Li; Shengwei Tian; Long Yu; Xin Ning
>
> **摘要:** Chamfer Distance (CD) comprises two components that can evaluate the global distribution and local performance of generated point clouds, making it widely utilized as a similarity measure between generated and target point clouds in point cloud completion tasks. Additionally, CD's computational efficiency has led to its frequent application as an objective function for guiding point cloud generation. However, using CD directly as an objective function with fixed equal weights for its two components can often result in seemingly high overall performance (i.e., low CD score), while failing to achieve a good global distribution. This is typically reflected in high Earth Mover's Distance (EMD) and Decomposed Chamfer Distance (DCD) scores, alongside poor human assessments. To address this issue, we propose a Flexible-Weighted Chamfer Distance (FCD) to guide point cloud generation. FCD assigns a higher weight to the global distribution component of CD and incorporates a flexible weighting strategy to adjust the balance between the two components, aiming to improve global distribution while maintaining robust overall performance. Experimental results on two state-of-the-art networks demonstrate that our method achieves superior results across multiple evaluation metrics, including CD, EMD, DCD, and F-Score, as well as in human evaluations.
>
---
#### [new 049] OmniStyle: Filtering High Quality Style Transfer Data at Scale
- **分类: cs.CV**

- **简介: 该论文属于风格迁移任务，旨在解决大规模高质量数据稀缺及风格控制精度不足的问题。构建了含百万图像三元组的OmniStyle-1M数据集，提出OmniFilter框架筛选优质样本，并设计基于DiT的OmniStyle模型，支持文本/图像引导，生成高分辨率结果，性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14028v1](http://arxiv.org/pdf/2505.14028v1)**

> **作者:** Ye Wang; Ruiqi Liu; Jiang Lin; Fei Liu; Zili Yi; Yilin Wang; Rui Ma
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** In this paper, we introduce OmniStyle-1M, a large-scale paired style transfer dataset comprising over one million content-style-stylized image triplets across 1,000 diverse style categories, each enhanced with textual descriptions and instruction prompts. We show that OmniStyle-1M can not only enable efficient and scalable of style transfer models through supervised training but also facilitate precise control over target stylization. Especially, to ensure the quality of the dataset, we introduce OmniFilter, a comprehensive style transfer quality assessment framework, which filters high-quality triplets based on content preservation, style consistency, and aesthetic appeal. Building upon this foundation, we propose OmniStyle, a framework based on the Diffusion Transformer (DiT) architecture designed for high-quality and efficient style transfer. This framework supports both instruction-guided and image-guided style transfer, generating high resolution outputs with exceptional detail. Extensive qualitative and quantitative evaluations demonstrate OmniStyle's superior performance compared to existing approaches, highlighting its efficiency and versatility. OmniStyle-1M and its accompanying methodologies provide a significant contribution to advancing high-quality style transfer, offering a valuable resource for the research community.
>
---
#### [new 050] Diving into the Fusion of Monocular Priors for Generalized Stereo Matching
- **分类: cs.CV**

- **简介: 该论文属于立体匹配任务，旨在解决融合视觉基础模型（VFM）单目先验时的对齐偏差、局部最优及迭代噪声问题。提出二进制局部顺序图统一相对与绝对深度表示，并通过像素级线性回归模块对齐单目深度与视差，提升泛化性能与效率。**

- **链接: [http://arxiv.org/pdf/2505.14414v1](http://arxiv.org/pdf/2505.14414v1)**

> **作者:** Chengtang Yao; Lidong Yu; Zhidan Liu; Jiaxi Zeng; Yuwei Wu; Yunde Jia
>
> **备注:** Code: https://github.com/YaoChengTang/Diving-into-the-Fusion-of-Monocular-Priors-for-Generalized-Stereo-Matching
>
> **摘要:** The matching formulation makes it naturally hard for the stereo matching to handle ill-posed regions like occlusions and non-Lambertian surfaces. Fusing monocular priors has been proven helpful for ill-posed matching, but the biased monocular prior learned from small stereo datasets constrains the generalization. Recently, stereo matching has progressed by leveraging the unbiased monocular prior from the vision foundation model (VFM) to improve the generalization in ill-posed regions. We dive into the fusion process and observe three main problems limiting the fusion of the VFM monocular prior. The first problem is the misalignment between affine-invariant relative monocular depth and absolute depth of disparity. Besides, when we use the monocular feature in an iterative update structure, the over-confidence in the disparity update leads to local optima results. A direct fusion of a monocular depth map could alleviate the local optima problem, but noisy disparity results computed at the first several iterations will misguide the fusion. In this paper, we propose a binary local ordering map to guide the fusion, which converts the depth map into a binary relative format, unifying the relative and absolute depth representation. The computed local ordering map is also used to re-weight the initial disparity update, resolving the local optima and noisy problem. In addition, we formulate the final direct fusion of monocular depth to the disparity as a registration problem, where a pixel-wise linear regression module can globally and adaptively align them. Our method fully exploits the monocular prior to support stereo matching results effectively and efficiently. We significantly improve the performance from the experiments when generalizing from SceneFlow to Middlebury and Booster datasets while barely reducing the efficiency.
>
---
#### [new 051] AKRMap: Adaptive Kernel Regression for Trustworthy Visualization of Cross-Modal Embeddings
- **分类: cs.CV; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于跨模态嵌入可视化任务，旨在解决传统降维方法无法有效整合跨模态评估指标（如CLIPScore）的问题。提出AKRMap方法，通过自适应核回归和监督投影网络，学习投影空间中的指标分布，生成更准确的可视化结果，并支持交互操作。实验显示其优于现有方法，尤其在文本到图像模型的嵌入分析中表现突出。**

- **链接: [http://arxiv.org/pdf/2505.14664v1](http://arxiv.org/pdf/2505.14664v1)**

> **作者:** Yilin Ye; Junchao Huang; Xingchen Zeng; Jiazhi Xia; Wei Zeng
>
> **摘要:** Cross-modal embeddings form the foundation for multi-modal models. However, visualization methods for interpreting cross-modal embeddings have been primarily confined to traditional dimensionality reduction (DR) techniques like PCA and t-SNE. These DR methods primarily focus on feature distributions within a single modality, whilst failing to incorporate metrics (e.g., CLIPScore) across multiple modalities.This paper introduces AKRMap, a new DR technique designed to visualize cross-modal embeddings metric with enhanced accuracy by learning kernel regression of the metric landscape in the projection space. Specifically, AKRMap constructs a supervised projection network guided by a post-projection kernel regression loss, and employs adaptive generalized kernels that can be jointly optimized with the projection. This approach enables AKRMap to efficiently generate visualizations that capture complex metric distributions, while also supporting interactive features such as zoom and overlay for deeper exploration. Quantitative experiments demonstrate that AKRMap outperforms existing DR methods in generating more accurate and trustworthy visualizations. We further showcase the effectiveness of AKRMap in visualizing and comparing cross-modal embeddings for text-to-image models. Code and demo are available at https://github.com/yilinye/AKRMap.
>
---
#### [new 052] UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文属于通用视觉定位任务，旨在解决多模态复杂场景下模型推理能力不足的问题。提出UniVG-R1模型，通过构建带推理链的CoT数据集进行监督微调，并结合强化学习优化推理路径，同时采用难度感知权重调整策略提升性能。实验显示其在多个基准测试中达最优效果。**

- **链接: [http://arxiv.org/pdf/2505.14231v1](http://arxiv.org/pdf/2505.14231v1)**

> **作者:** Sule Bai; Mingxing Li; Yong Liu; Jing Tang; Haoji Zhang; Lei Sun; Xiangxiang Chu; Yansong Tang
>
> **摘要:** Traditional visual grounding methods primarily focus on single-image scenarios with simple textual references. However, extending these methods to real-world scenarios that involve implicit and complex instructions, particularly in conjunction with multiple images, poses significant challenges, which is mainly due to the lack of advanced reasoning ability across diverse multi-modal contexts. In this work, we aim to address the more practical universal grounding task, and propose UniVG-R1, a reasoning guided multimodal large language model (MLLM) for universal visual grounding, which enhances reasoning capabilities through reinforcement learning (RL) combined with cold-start data. Specifically, we first construct a high-quality Chain-of-Thought (CoT) grounding dataset, annotated with detailed reasoning chains, to guide the model towards correct reasoning paths via supervised fine-tuning. Subsequently, we perform rule-based reinforcement learning to encourage the model to identify correct reasoning chains, thereby incentivizing its reasoning capabilities. In addition, we identify a difficulty bias arising from the prevalence of easy samples as RL training progresses, and we propose a difficulty-aware weight adjustment strategy to further strengthen the performance. Experimental results demonstrate the effectiveness of UniVG-R1, which achieves state-of-the-art performance on MIG-Bench with a 9.1% improvement over the previous method. Furthermore, our model exhibits strong generalizability, achieving an average improvement of 23.4% in zero-shot performance across four image and video reasoning grounding benchmarks. The project page can be accessed at https://amap-ml.github.io/UniVG-R1-page/.
>
---
#### [new 053] UniCTokens: Boosting Personalized Understanding and Generation via Unified Concept Tokens
- **分类: cs.CV**

- **简介: 该论文属于个性化视觉语言模型任务，旨在解决现有方法分离处理概念理解与生成导致复杂提示生成效果差的问题。提出UniCTokens框架，通过统一概念标记和三阶段渐进训练策略，增强两项任务的互惠提升，并构建UnifyBench基准进行定量评估，实现个性化知识驱动生成的最优效果。**

- **链接: [http://arxiv.org/pdf/2505.14671v1](http://arxiv.org/pdf/2505.14671v1)**

> **作者:** Ruichuan An; Sihan Yang; Renrui Zhang; Zijun Shen; Ming Lu; Gaole Dai; Hao Liang; Ziyu Guo; Shilin Yan; Yulin Luo; Bocheng Zou; Chaoqun Yang; Wentao Zhang
>
> **摘要:** Personalized models have demonstrated remarkable success in understanding and generating concepts provided by users. However, existing methods use separate concept tokens for understanding and generation, treating these tasks in isolation. This may result in limitations for generating images with complex prompts. For example, given the concept $\langle bo\rangle$, generating "$\langle bo\rangle$ wearing its hat" without additional textual descriptions of its hat. We call this kind of generation personalized knowledge-driven generation. To address the limitation, we present UniCTokens, a novel framework that effectively integrates personalized information into a unified vision language model (VLM) for understanding and generation. UniCTokens trains a set of unified concept tokens to leverage complementary semantics, boosting two personalized tasks. Moreover, we propose a progressive training strategy with three stages: understanding warm-up, bootstrapping generation from understanding, and deepening understanding from generation to enhance mutual benefits between both tasks. To quantitatively evaluate the unified VLM personalization, we present UnifyBench, the first benchmark for assessing concept understanding, concept generation, and knowledge-driven generation. Experimental results on UnifyBench indicate that UniCTokens shows competitive performance compared to leading methods in concept understanding, concept generation, and achieving state-of-the-art results in personalized knowledge-driven generation. Our research demonstrates that enhanced understanding improves generation, and the generation process can yield valuable insights into understanding. Our code and dataset will be released at: \href{https://github.com/arctanxarc/UniCTokens}{https://github.com/arctanxarc/UniCTokens}.
>
---
#### [new 054] Instructing Text-to-Image Diffusion Models via Classifier-Guided Semantic Optimization
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成与编辑任务，旨在解决现有方法依赖手动设计文本提示导致效率低、细节干扰和性能受限的问题。提出通过分类器优化语义嵌入的方法，无需文本提示或模型训练，利用数据集级语义嵌入实现解纠缠编辑，理论证明其最优性并验证了跨领域有效性。**

- **链接: [http://arxiv.org/pdf/2505.14254v1](http://arxiv.org/pdf/2505.14254v1)**

> **作者:** Yuanyuan Chang; Yinghua Yao; Tao Qin; Mengmeng Wang; Ivor Tsang; Guang Dai
>
> **摘要:** Text-to-image diffusion models have emerged as powerful tools for high-quality image generation and editing. Many existing approaches rely on text prompts as editing guidance. However, these methods are constrained by the need for manual prompt crafting, which can be time-consuming, introduce irrelevant details, and significantly limit editing performance. In this work, we propose optimizing semantic embeddings guided by attribute classifiers to steer text-to-image models toward desired edits, without relying on text prompts or requiring any training or fine-tuning of the diffusion model. We utilize classifiers to learn precise semantic embeddings at the dataset level. The learned embeddings are theoretically justified as the optimal representation of attribute semantics, enabling disentangled and accurate edits. Experiments further demonstrate that our method achieves high levels of disentanglement and strong generalization across different domains of data.
>
---
#### [new 055] Beyond Words: Multimodal LLM Knows When to Speak
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于对话时机预测任务，解决大语言模型在多模态对话中难以及时生成短回应的问题。研究构建了含同步视听文本数据的多模态数据集，并提出MM-When2Speak模型，融合多模态信息预测回应时机与类型，实验显示其响应时机准确性超现有模型4倍。**

- **链接: [http://arxiv.org/pdf/2505.14654v1](http://arxiv.org/pdf/2505.14654v1)**

> **作者:** Zikai Liao; Yi Ouyang; Yi-Lun Lee; Chen-Ping Yu; Yi-Hsuan Tsai; Zhaozheng Yin
>
> **备注:** Project page: https://github.com/lzk901372/MM-When2Speak
>
> **摘要:** While large language model (LLM)-based chatbots have demonstrated strong capabilities in generating coherent and contextually relevant responses, they often struggle with understanding when to speak, particularly in delivering brief, timely reactions during ongoing conversations. This limitation arises largely from their reliance on text input, lacking the rich contextual cues in real-world human dialogue. In this work, we focus on real-time prediction of response types, with an emphasis on short, reactive utterances that depend on subtle, multimodal signals across vision, audio, and text. To support this, we introduce a new multimodal dataset constructed from real-world conversational videos, containing temporally aligned visual, auditory, and textual streams. This dataset enables fine-grained modeling of response timing in dyadic interactions. Building on this dataset, we propose MM-When2Speak, a multimodal LLM-based model that adaptively integrates visual, auditory, and textual context to predict when a response should occur, and what type of response is appropriate. Experiments show that MM-When2Speak significantly outperforms state-of-the-art unimodal and LLM-based baselines, achieving up to a 4x improvement in response timing accuracy over leading commercial LLMs. These results underscore the importance of multimodal inputs for producing timely, natural, and engaging conversational AI.
>
---
#### [new 056] Multi-Label Stereo Matching for Transparent Scene Depth Estimation
- **分类: cs.CV**

- **简介: 该论文提出多标签立体匹配方法，解决透明场景中同时估计透明物体与被遮挡背景深度的问题。传统方法仅回归单一深度值，而本文采用像素级多维高斯分布建模多深度值，并通过GRU迭代优化均值和协方差，合成数据集验证了其在透明表面深度估计与背景保留上的优势。**

- **链接: [http://arxiv.org/pdf/2505.14008v1](http://arxiv.org/pdf/2505.14008v1)**

> **作者:** Zhidan Liu; Chengtang Yao; Jiaxi Zeng; Yuwei Wu; Yunde Jia
>
> **摘要:** In this paper, we present a multi-label stereo matching method to simultaneously estimate the depth of the transparent objects and the occluded background in transparent scenes.Unlike previous methods that assume a unimodal distribution along the disparity dimension and formulate the matching as a single-label regression problem, we propose a multi-label regression formulation to estimate multiple depth values at the same pixel in transparent scenes. To resolve the multi-label regression problem, we introduce a pixel-wise multivariate Gaussian representation, where the mean vector encodes multiple depth values at the same pixel, and the covariance matrix determines whether a multi-label representation is necessary for a given pixel. The representation is iteratively predicted within a GRU framework. In each iteration, we first predict the update step for the mean parameters and then use both the update step and the updated mean parameters to estimate the covariance matrix. We also synthesize a dataset containing 10 scenes and 89 objects to validate the performance of transparent scene depth estimation. The experiments show that our method greatly improves the performance on transparent surfaces while preserving the background information for scene reconstruction. Code is available at https://github.com/BFZD233/TranScene.
>
---
#### [new 057] Selective Structured State Space for Multispectral-fused Small Target Detection
- **分类: cs.CV**

- **简介: 该论文属于高分辨率遥感图像中小目标检测任务，旨在解决小目标识别精度低和计算成本高的问题。针对Mamba模型在小目标检测上的不足，提出ESTD模块强化局部细节捕捉，CARG模块优化空间与通道信息，结合MEPF模块实现多光谱融合，提升小目标特征表达与检测性能。**

- **链接: [http://arxiv.org/pdf/2505.14043v1](http://arxiv.org/pdf/2505.14043v1)**

> **作者:** Qianqian Zhang; WeiJun Wang; Yunxing Liu; Li Zhou; Hao Zhao; Junshe An; Zihan Wang
>
> **摘要:** Target detection in high-resolution remote sensing imagery faces challenges due to the low recognition accuracy of small targets and high computational costs. The computational complexity of the Transformer architecture increases quadratically with image resolution, while Convolutional Neural Networks (CNN) architectures are forced to stack deeper convolutional layers to expand their receptive fields, leading to an explosive growth in computational demands. To address these computational constraints, we leverage Mamba's linear complexity for efficiency. However, Mamba's performance declines for small targets, primarily because small targets occupy a limited area in the image and have limited semantic information. Accurate identification of these small targets necessitates not only Mamba's global attention capabilities but also the precise capture of fine local details. To this end, we enhance Mamba by developing the Enhanced Small Target Detection (ESTD) module and the Convolutional Attention Residual Gate (CARG) module. The ESTD module bolsters local attention to capture fine-grained details, while the CARG module, built upon Mamba, emphasizes spatial and channel-wise information, collectively improving the model's ability to capture distinctive representations of small targets. Additionally, to highlight the semantic representation of small targets, we design a Mask Enhanced Pixel-level Fusion (MEPF) module for multispectral fusion, which enhances target features by effectively fusing visible and infrared multimodal information.
>
---
#### [new 058] AppleGrowthVision: A large-scale stereo dataset for phenological analysis, fruit detection, and 3D reconstruction in apple orchards
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出AppleGrowthVision数据集，旨在解决苹果园监测中数据不足及立体图像缺失问题。包含覆盖6生长阶段的9317张立体图像和31084个标注果实，提升果实检测、3D重建及生长阶段预测的模型性能，填补农业与计算机视觉结合的空白。**

- **链接: [http://arxiv.org/pdf/2505.14029v1](http://arxiv.org/pdf/2505.14029v1)**

> **作者:** Laura-Sophia von Hirschhausen; Jannes S. Magnusson; Mykyta Kovalenko; Fredrik Boye; Tanay Rawat; Peter Eisert; Anna Hilsmann; Sebastian Pretzsch; Sebastian Bosse
>
> **摘要:** Deep learning has transformed computer vision for precision agriculture, yet apple orchard monitoring remains limited by dataset constraints. The lack of diverse, realistic datasets and the difficulty of annotating dense, heterogeneous scenes. Existing datasets overlook different growth stages and stereo imagery, both essential for realistic 3D modeling of orchards and tasks like fruit localization, yield estimation, and structural analysis. To address these gaps, we present AppleGrowthVision, a large-scale dataset comprising two subsets. The first includes 9,317 high resolution stereo images collected from a farm in Brandenburg (Germany), covering six agriculturally validated growth stages over a full growth cycle. The second subset consists of 1,125 densely annotated images from the same farm in Brandenburg and one in Pillnitz (Germany), containing a total of 31,084 apple labels. AppleGrowthVision provides stereo-image data with agriculturally validated growth stages, enabling precise phenological analysis and 3D reconstructions. Extending MinneApple with our data improves YOLOv8 performance by 7.69 % in terms of F1-score, while adding it to MinneApple and MAD boosts Faster R-CNN F1-score by 31.06 %. Additionally, six BBCH stages were predicted with over 95 % accuracy using VGG16, ResNet152, DenseNet201, and MobileNetv2. AppleGrowthVision bridges the gap between agricultural science and computer vision, by enabling the development of robust models for fruit detection, growth modeling, and 3D analysis in precision agriculture. Future work includes improving annotation, enhancing 3D reconstruction, and extending multimodal analysis across all growth stages.
>
---
#### [new 059] A General Framework for Group Sparsity in Hyperspectral Unmixing Using Endmember Bundles
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于高光谱解混任务，旨在解决传统线性混合模型无法准确表征材料光谱变异性的缺陷。提出基于"端元束"的组稀疏框架，通过将材料表示为多光谱集合（endmember bundles），并引入组间稀疏或组内组间稀疏（SWAG）约束，结合新型变换L1正则化，提升材料光谱及丰度估计精度。实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2505.14634v1](http://arxiv.org/pdf/2505.14634v1)**

> **作者:** Gokul Bhusal; Yifei Lou; Cristina Garcia-Cardona; Ekaterina Merkurjev
>
> **摘要:** Due to low spatial resolution, hyperspectral data often consists of mixtures of contributions from multiple materials. This limitation motivates the task of hyperspectral unmixing (HU), a fundamental problem in hyperspectral imaging. HU aims to identify the spectral signatures (\textit{endmembers}) of the materials present in an observed scene, along with their relative proportions (\textit{fractional abundance}) in each pixel. A major challenge lies in the class variability in materials, which hinders accurate representation by a single spectral signature, as assumed in the conventional linear mixing model. Moreover, To address this issue, we propose using group sparsity after representing each material with a set of spectral signatures, known as endmember bundles, where each group corresponds to a specific material. In particular, we develop a bundle-based framework that can enforce either inter-group sparsity or sparsity within and across groups (SWAG) on the abundance coefficients. Furthermore, our framework offers the flexibility to incorporate a variety of sparsity-promoting penalties, among which the transformed $\ell_1$ (TL1) penalty is a novel regularization in the HU literature. Extensive experiments conducted on both synthetic and real hyperspectral data demonstrate the effectiveness and superiority of the proposed approaches.
>
---
#### [new 060] InstanceBEV: Unifying Instance and BEV Representation for Global Modeling
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶全局感知建模任务，旨在解决BEV方法因数据规模扩大导致的效率问题。提出InstanceBEV，首次通过实例级降维结合Transformer直接聚合全局特征，无需稀疏化处理，实现高效大尺度BEV表示，在OpenOcc-NuScenes数据集达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.13817v1](http://arxiv.org/pdf/2505.13817v1)**

> **作者:** Feng Li; Kun Xu; Zhaoyue Wang; Yunduan Cui; Mohammad Masum Billah; Jia Liu
>
> **摘要:** Occupancy Grid Maps are widely used in navigation for their ability to represent 3D space occupancy. However, existing methods that utilize multi-view cameras to construct Occupancy Networks for perception modeling suffer from cubic growth in data complexity. Adopting a Bird's-Eye View (BEV) perspective offers a more practical solution for autonomous driving, as it provides higher semantic density and mitigates complex object occlusions. Nonetheless, BEV-based approaches still require extensive engineering optimizations to enable efficient large-scale global modeling. To address this challenge, we propose InstanceBEV, the first method to introduce instance-level dimensionality reduction for BEV, enabling global modeling with transformers without relying on sparsification or acceleration operators. Different from other BEV methods, our approach directly employs transformers to aggregate global features. Compared to 3D object detection models, our method samples global feature maps into 3D space. Experiments on OpenOcc-NuScenes dataset show that InstanceBEV achieves state-of-the-art performance while maintaining a simple, efficient framework without requiring additional optimizations.
>
---
#### [new 061] GeoRanker: Distance-Aware Ranking for Worldwide Image Geolocalization
- **分类: cs.CV**

- **简介: 该论文属于全球图像地理定位任务，解决现有方法忽视空间关系、依赖简单相似性匹配的问题。提出GeoRanker框架，利用大视觉语言模型联合编码查询-候选交互，并设计多阶距离损失函数，同时建模绝对/相对地理距离，结合新构建的GeoRanking数据集，实现更优的结构化空间推理，在两个基准上达最优性能。**

- **链接: [http://arxiv.org/pdf/2505.13731v1](http://arxiv.org/pdf/2505.13731v1)**

> **作者:** Pengyue Jia; Seongheon Park; Song Gao; Xiangyu Zhao; Yixuan Li
>
> **摘要:** Worldwide image geolocalization-the task of predicting GPS coordinates from images taken anywhere on Earth-poses a fundamental challenge due to the vast diversity in visual content across regions. While recent approaches adopt a two-stage pipeline of retrieving candidates and selecting the best match, they typically rely on simplistic similarity heuristics and point-wise supervision, failing to model spatial relationships among candidates. In this paper, we propose GeoRanker, a distance-aware ranking framework that leverages large vision-language models to jointly encode query-candidate interactions and predict geographic proximity. In addition, we introduce a multi-order distance loss that ranks both absolute and relative distances, enabling the model to reason over structured spatial relationships. To support this, we curate GeoRanking, the first dataset explicitly designed for geographic ranking tasks with multimodal candidate information. GeoRanker achieves state-of-the-art results on two well-established benchmarks (IM2GPS3K and YFCC4K), significantly outperforming current best methods.
>
---
#### [new 062] Place Recognition: A Comprehensive Review, Current Challenges and Future Directions
- **分类: cs.CV**

- **简介: 该论文属于地方识别任务的综述，旨在解决动态环境下机器人/车辆重复定位与闭环检测问题。工作包括：系统梳理CNN、Transformer及跨模态方法的技术演进；总结 benchmark 数据集与评估指标；分析域适应、实时性等挑战并提出终身学习等未来方向。**

- **链接: [http://arxiv.org/pdf/2505.14068v1](http://arxiv.org/pdf/2505.14068v1)**

> **作者:** Zhenyu Li; Tianyi Shang; Pengjie Xu; Zhaojun Deng
>
> **备注:** 35 pages
>
> **摘要:** Place recognition is a cornerstone of vehicle navigation and mapping, which is pivotal in enabling systems to determine whether a location has been previously visited. This capability is critical for tasks such as loop closure in Simultaneous Localization and Mapping (SLAM) and long-term navigation under varying environmental conditions. In this survey, we comprehensively review recent advancements in place recognition, emphasizing three representative methodological paradigms: Convolutional Neural Network (CNN)-based approaches, Transformer-based frameworks, and cross-modal strategies. We begin by elucidating the significance of place recognition within the broader context of autonomous systems. Subsequently, we trace the evolution of CNN-based methods, highlighting their contributions to robust visual descriptor learning and scalability in large-scale environments. We then examine the emerging class of Transformer-based models, which leverage self-attention mechanisms to capture global dependencies and offer improved generalization across diverse scenes. Furthermore, we discuss cross-modal approaches that integrate heterogeneous data sources such as Lidar, vision, and text description, thereby enhancing resilience to viewpoint, illumination, and seasonal variations. We also summarize standard datasets and evaluation metrics widely adopted in the literature. Finally, we identify current research challenges and outline prospective directions, including domain adaptation, real-time performance, and lifelong learning, to inspire future advancements in this domain. The unified framework of leading-edge place recognition methods, i.e., code library, and the results of their experimental evaluations are available at https://github.com/CV4RA/SOTA-Place-Recognitioner.
>
---
#### [new 063] StPR: Spatiotemporal Preservation and Routing for Exemplar-Free Video Class-Incremental Learning
- **分类: cs.CV**

- **简介: 该论文属于视频类增量学习（VCIL）任务，旨在解决模型在新增动作类别时避免遗忘先前知识及处理时空结构的挑战。提出StPR框架，通过FSSD保留关键语义通道防止知识遗忘，利用TD-MoE动态路由捕捉时序变化，实现无需示例的高效增量学习。**

- **链接: [http://arxiv.org/pdf/2505.13997v1](http://arxiv.org/pdf/2505.13997v1)**

> **作者:** Huaijie Wang; De Cheng; Guozhang Li; Zhipeng Xu; Lingfeng He; Jie Li; Nannan Wang; Xinbo Gao
>
> **摘要:** Video Class-Incremental Learning (VCIL) seeks to develop models that continuously learn new action categories over time without forgetting previously acquired knowledge. Unlike traditional Class-Incremental Learning (CIL), VCIL introduces the added complexity of spatiotemporal structures, making it particularly challenging to mitigate catastrophic forgetting while effectively capturing both frame-shared semantics and temporal dynamics. Existing approaches either rely on exemplar rehearsal, raising concerns over memory and privacy, or adapt static image-based methods that neglect temporal modeling. To address these limitations, we propose Spatiotemporal Preservation and Routing (StPR), a unified and exemplar-free VCIL framework that explicitly disentangles and preserves spatiotemporal information. First, we introduce Frame-Shared Semantics Distillation (FSSD), which identifies semantically stable and meaningful channels by jointly considering semantic sensitivity and classification contribution. These important semantic channels are selectively regularized to maintain prior knowledge while allowing for adaptation. Second, we design a Temporal Decomposition-based Mixture-of-Experts (TD-MoE), which dynamically routes task-specific experts based on their temporal dynamics, enabling inference without task ID or stored exemplars. Together, StPR effectively leverages spatial semantics and temporal dynamics, achieving a unified, exemplar-free VCIL framework. Extensive experiments on UCF101, HMDB51, and Kinetics400 show that our method outperforms existing baselines while offering improved interpretability and efficiency in VCIL. Code is available in the supplementary materials.
>
---
#### [new 064] Unlocking the Power of SAM 2 for Few-Shot Segmentation
- **分类: cs.CV**

- **简介: 该论文属于Few-Shot Segmentation（FSS）任务，旨在利用SAM 2解决视频数据与FSS场景中前景对象身份不匹配及记忆噪声问题。提出伪提示生成器构建伪查询记忆，并通过迭代优化和支持校准注意力机制提升记忆准确性，最终显著提升1-shot分割性能。**

- **链接: [http://arxiv.org/pdf/2505.14100v1](http://arxiv.org/pdf/2505.14100v1)**

> **作者:** Qianxiong Xu; Lanyun Zhu; Xuanyi Liu; Guosheng Lin; Cheng Long; Ziyue Li; Rui Zhao
>
> **备注:** This paper is accepted by ICML'25
>
> **摘要:** Few-Shot Segmentation (FSS) aims to learn class-agnostic segmentation on few classes to segment arbitrary classes, but at the risk of overfitting. To address this, some methods use the well-learned knowledge of foundation models (e.g., SAM) to simplify the learning process. Recently, SAM 2 has extended SAM by supporting video segmentation, whose class-agnostic matching ability is useful to FSS. A simple idea is to encode support foreground (FG) features as memory, with which query FG features are matched and fused. Unfortunately, the FG objects in different frames of SAM 2's video data are always the same identity, while those in FSS are different identities, i.e., the matching step is incompatible. Therefore, we design Pseudo Prompt Generator to encode pseudo query memory, matching with query features in a compatible way. However, the memories can never be as accurate as the real ones, i.e., they are likely to contain incomplete query FG, and some unexpected query background (BG) features, leading to wrong segmentation. Hence, we further design Iterative Memory Refinement to fuse more query FG features into the memory, and devise a Support-Calibrated Memory Attention to suppress the unexpected query BG features in memory. Extensive experiments have been conducted on PASCAL-5$^i$ and COCO-20$^i$ to validate the effectiveness of our design, e.g., the 1-shot mIoU can be 4.2\% better than the best baseline.
>
---
#### [new 065] RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医学影像报告生成任务，针对现有方法忽视大模型内部知识导致冗余的问题，提出RADAR框架：先提取模型与专家图像分类一致的内部知识，再检索补充外部知识，融合二者生成更准确的报告，在多个数据集上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14318v1](http://arxiv.org/pdf/2505.14318v1)**

> **作者:** Wenjun Hou; Yi Cheng; Kaishuai Xu; Heng Li; Yan Hu; Wenjie Li; Jiang Liu
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities in various domains, including radiology report generation. Previous approaches have attempted to utilize multimodal LLMs for this task, enhancing their performance through the integration of domain-specific knowledge retrieval. However, these approaches often overlook the knowledge already embedded within the LLMs, leading to redundant information integration and inefficient utilization of learned representations. To address this limitation, we propose RADAR, a framework for enhancing radiology report generation with supplementary knowledge injection. RADAR improves report generation by systematically leveraging both the internal knowledge of an LLM and externally retrieved information. Specifically, it first extracts the model's acquired knowledge that aligns with expert image-based classification outputs. It then retrieves relevant supplementary knowledge to further enrich this information. Finally, by aggregating both sources, RADAR generates more accurate and informative radiology reports. Extensive experiments on MIMIC-CXR, CheXpert-Plus, and IU X-ray demonstrate that our model outperforms state-of-the-art LLMs in both language quality and clinical accuracy
>
---
#### [new 066] Generalizable Multispectral Land Cover Classification via Frequency-Aware Mixture of Low-Rank Token Experts
- **分类: cs.CV**

- **简介: 该论文属于多光谱地表覆盖分类（MLCC）任务，旨在解决传感器差异和地理条件导致的光谱偏移问题。提出Land-MoE方法，通过频率感知低秩令牌专家混合模型（MoLTE和FAF模块），动态调整特征并抑制噪声，提升跨传感器/地理场景的分类性能，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14088v1](http://arxiv.org/pdf/2505.14088v1)**

> **作者:** Xi Chen; Shen Yan; Juelin Zhu; Chen Chen; Yu Liu; Maojun Zhang
>
> **摘要:** We introduce Land-MoE, a novel approach for multispectral land cover classification (MLCC). Spectral shift, which emerges from disparities in sensors and geospatial conditions, poses a significant challenge in this domain. Existing methods predominantly rely on domain adaptation and generalization strategies, often utilizing small-scale models that exhibit limited performance. In contrast, Land-MoE addresses these issues by hierarchically inserting a Frequency-aware Mixture of Low-rank Token Experts, to fine-tune Vision Foundation Models (VFMs) in a parameter-efficient manner. Specifically, Land-MoE comprises two key modules: the mixture of low-rank token experts (MoLTE) and frequency-aware filters (FAF). MoLTE leverages rank-differentiated tokens to generate diverse feature adjustments for individual instances within multispectral images. By dynamically combining learnable low-rank token experts of varying ranks, it enhances the robustness against spectral shifts. Meanwhile, FAF conducts frequency-domain modulation on the refined features. This process enables the model to effectively capture frequency band information that is strongly correlated with semantic essence, while simultaneously suppressing frequency noise irrelevant to the task. Comprehensive experiments on MLCC tasks involving cross-sensor and cross-geospatial setups demonstrate that Land-MoE outperforms existing methods by a large margin. Additionally, the proposed approach has also achieved state-of-the-art performance in domain generalization semantic segmentation tasks of RGB remote sensing images.
>
---
#### [new 067] Beginning with You: Perceptual-Initialization Improves Vision-Language Representation and Alignment
- **分类: cs.CV; q-bio.NC**

- **简介: 该论文属于视觉语言表征与对齐任务，旨在提升零样本学习性能。提出Perceptual-Initialization（PI）方法，在模型初始化阶段融入人类感知结构（基于NIGHTS数据集），优化CLIP视觉编码器，结合自监督学习，显著提升29个分类与检索任务的零样本表现，无需任务特异性微调，验证早期嵌入人类感知对多模态泛化能力的增强作用。**

- **链接: [http://arxiv.org/pdf/2505.14204v1](http://arxiv.org/pdf/2505.14204v1)**

> **作者:** Yang Hu; Runchen Wang; Stephen Chong Zhao; Xuhui Zhan; Do Hun Kim; Mark Wallace; David A. Tovar
>
> **备注:** 10 pages, 5 figures, 2 tables
>
> **摘要:** We introduce Perceptual-Initialization (PI), a paradigm shift in visual representation learning that incorporates human perceptual structure during the initialization phase rather than as a downstream fine-tuning step. By integrating human-derived triplet embeddings from the NIGHTS dataset to initialize a CLIP vision encoder, followed by self-supervised learning on YFCC15M, our approach demonstrates significant zero-shot performance improvements, without any task-specific fine-tuning, across 29 zero shot classification and 2 retrieval benchmarks. On ImageNet-1K, zero-shot gains emerge after approximately 15 epochs of pretraining. Benefits are observed across datasets of various scales, with improvements manifesting at different stages of the pretraining process depending on dataset characteristics. Our approach consistently enhances zero-shot top-1 accuracy, top-5 accuracy, and retrieval recall (e.g., R@1, R@5) across these diverse evaluation tasks, without requiring any adaptation to target domains. These findings challenge the conventional wisdom of using human-perceptual data primarily for fine-tuning and demonstrate that embedding human perceptual structure during early representation learning yields more capable and vision-language aligned systems that generalize immediately to unseen tasks. Our work shows that "beginning with you", starting with human perception, provides a stronger foundation for general-purpose vision-language intelligence.
>
---
#### [new 068] Dynadiff: Single-stage Decoding of Images from Continuously Evolving fMRI
- **分类: cs.CV**

- **简介: 该论文提出Dynadiff模型，通过单阶段扩散方法从动态fMRI信号重建图像。针对现有方法依赖多阶段流程且丢失时间信息的问题，该模型简化训练流程，提升高语义图像重建精度，在保持对静态fMRI竞争力的同时，精准解析脑活动中的图像表征演化，推动实时脑-图像解码发展。**

- **链接: [http://arxiv.org/pdf/2505.14556v1](http://arxiv.org/pdf/2505.14556v1)**

> **作者:** Marlène Careil; Yohann Benchetrit; Jean-Rémi King
>
> **摘要:** Brain-to-image decoding has been recently propelled by the progress in generative AI models and the availability of large ultra-high field functional Magnetic Resonance Imaging (fMRI). However, current approaches depend on complicated multi-stage pipelines and preprocessing steps that typically collapse the temporal dimension of brain recordings, thereby limiting time-resolved brain decoders. Here, we introduce Dynadiff (Dynamic Neural Activity Diffusion for Image Reconstruction), a new single-stage diffusion model designed for reconstructing images from dynamically evolving fMRI recordings. Our approach offers three main contributions. First, Dynadiff simplifies training as compared to existing approaches. Second, our model outperforms state-of-the-art models on time-resolved fMRI signals, especially on high-level semantic image reconstruction metrics, while remaining competitive on preprocessed fMRI data that collapse time. Third, this approach allows a precise characterization of the evolution of image representations in brain activity. Overall, this work lays the foundation for time-resolved brain-to-image decoding.
>
---
#### [new 069] Intra-class Patch Swap for Self-Distillation
- **分类: cs.CV**

- **简介: 该论文属于模型压缩任务，针对传统知识蒸馏依赖教师网络及现有自蒸馏方法复杂度高的问题，提出基于类内块交换的无教师自蒸馏框架。通过单模型内生成不同置信度样本对并对其预测分布对齐，实现高效模型优化，适用于多视觉任务。**

- **链接: [http://arxiv.org/pdf/2505.14124v1](http://arxiv.org/pdf/2505.14124v1)**

> **作者:** Hongjun Choi; Eun Som Jeon; Ankita Shukla; Pavan Turaga
>
> **备注:** Accepted for publication in Neurocomputing
>
> **摘要:** Knowledge distillation (KD) is a valuable technique for compressing large deep learning models into smaller, edge-suitable networks. However, conventional KD frameworks rely on pre-trained high-capacity teacher networks, which introduce significant challenges such as increased memory/storage requirements, additional training costs, and ambiguity in selecting an appropriate teacher for a given student model. Although a teacher-free distillation (self-distillation) has emerged as a promising alternative, many existing approaches still rely on architectural modifications or complex training procedures, which limit their generality and efficiency. To address these limitations, we propose a novel framework based on teacher-free distillation that operates using a single student network without any auxiliary components, architectural modifications, or additional learnable parameters. Our approach is built on a simple yet highly effective augmentation, called intra-class patch swap augmentation. This augmentation simulates a teacher-student dynamic within a single model by generating pairs of intra-class samples with varying confidence levels, and then applying instance-to-instance distillation to align their predictive distributions. Our method is conceptually simple, model-agnostic, and easy to implement, requiring only a single augmentation function. Extensive experiments across image classification, semantic segmentation, and object detection show that our method consistently outperforms both existing self-distillation baselines and conventional teacher-based KD approaches. These results suggest that the success of self-distillation could hinge on the design of the augmentation itself. Our codes are available at https://github.com/hchoi71/Intra-class-Patch-Swap.
>
---
#### [new 070] CONSIGN: Conformal Segmentation Informed by Spatial Groupings via Decomposition
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像分割不确定性量化任务，旨在解决传统符合预测（CP）忽略空间相关性导致保守估计的问题。提出CONSIGN方法，通过空间分组分解整合像素间关联，生成可靠预测集，兼容多种模型。实验显示其在医疗和COCO数据集上优于传统像素CP方法，提升不确定性估计质量。**

- **链接: [http://arxiv.org/pdf/2505.14113v1](http://arxiv.org/pdf/2505.14113v1)**

> **作者:** Bruno Viti; Elias Karabelas; Martin Holler
>
> **摘要:** Most machine learning-based image segmentation models produce pixel-wise confidence scores - typically derived from softmax outputs - that represent the model's predicted probability for each class label at every pixel. While this information can be particularly valuable in high-stakes domains such as medical imaging, these (uncalibrated) scores are heuristic in nature and do not constitute rigorous quantitative uncertainty estimates. Conformal prediction (CP) provides a principled framework for transforming heuristic confidence scores into statistically valid uncertainty estimates. However, applying CP directly to image segmentation ignores the spatial correlations between pixels, a fundamental characteristic of image data. This can result in overly conservative and less interpretable uncertainty estimates. To address this, we propose CONSIGN (Conformal Segmentation Informed by Spatial Groupings via Decomposition), a CP-based method that incorporates spatial correlations to improve uncertainty quantification in image segmentation. Our method generates meaningful prediction sets that come with user-specified, high-probability error guarantees. It is compatible with any pre-trained segmentation model capable of generating multiple sample outputs - such as those using dropout, Bayesian modeling, or ensembles. We evaluate CONSIGN against a standard pixel-wise CP approach across three medical imaging datasets and two COCO dataset subsets, using three different pre-trained segmentation models. Results demonstrate that accounting for spatial structure significantly improves performance across multiple metrics and enhances the quality of uncertainty estimates.
>
---
#### [new 071] RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态视觉文化理解任务，旨在解决视觉语言模型（VLMs）在文化细微差别解析上的不足。提出基准RAVENEA，包含文化聚焦视觉问答（cVQA）和文化导向图像描述（cIC）任务，整合超1万份人工筛选的维基文档，评估7种检索器与14种VLMs，验证检索增强可显著提升模型性能（cVQA提升3.2%，cIC提升6.2%）。**

- **链接: [http://arxiv.org/pdf/2505.14462v1](http://arxiv.org/pdf/2505.14462v1)**

> **作者:** Jiaang Li; Yifei Yuan; Wenyan Li; Mohammad Aliannejadi; Daniel Hershcovich; Anders Søgaard; Ivan Vulić; Wenxuan Zhang; Paul Pu Liang; Yang Deng; Serge Belongie
>
> **摘要:** As vision-language models (VLMs) become increasingly integrated into daily life, the need for accurate visual culture understanding is becoming critical. Yet, these models frequently fall short in interpreting cultural nuances effectively. Prior work has demonstrated the effectiveness of retrieval-augmented generation (RAG) in enhancing cultural understanding in text-only settings, while its application in multimodal scenarios remains underexplored. To bridge this gap, we introduce RAVENEA (Retrieval-Augmented Visual culturE uNdErstAnding), a new benchmark designed to advance visual culture understanding through retrieval, focusing on two tasks: culture-focused visual question answering (cVQA) and culture-informed image captioning (cIC). RAVENEA extends existing datasets by integrating over 10,000 Wikipedia documents curated and ranked by human annotators. With RAVENEA, we train and evaluate seven multimodal retrievers for each image query, and measure the downstream impact of retrieval-augmented inputs across fourteen state-of-the-art VLMs. Our results show that lightweight VLMs, when augmented with culture-aware retrieval, outperform their non-augmented counterparts (by at least 3.2% absolute on cVQA and 6.2% absolute on cIC). This highlights the value of retrieval-augmented methods and culturally inclusive benchmarks for multimodal understanding.
>
---
#### [new 072] Transfer Learning from Visual Speech Recognition to Mouthing Recognition in German Sign Language
- **分类: cs.CV**

- **简介: 该论文属于迁移学习任务，旨在通过将视觉语音识别（VSR）模型迁移到德语手势语的口型识别，解决SLR系统中非手动特征（如口型）信息利用不足及标注数据稀缺的问题。研究利用三组VSR数据集探索任务相似性影响，发现多任务学习能提升口型识别与VSR的准确性及鲁棒性，证明口型识别应作为独立但关联任务处理。**

- **链接: [http://arxiv.org/pdf/2505.13784v1](http://arxiv.org/pdf/2505.13784v1)**

> **作者:** Dinh Nam Pham; Eleftherios Avramidis
>
> **备注:** Accepted at 19th IEEE International Conference on Automatic Face and Gesture Recognition 2025
>
> **摘要:** Sign Language Recognition (SLR) systems primarily focus on manual gestures, but non-manual features such as mouth movements, specifically mouthing, provide valuable linguistic information. This work directly classifies mouthing instances to their corresponding words in the spoken language while exploring the potential of transfer learning from Visual Speech Recognition (VSR) to mouthing recognition in German Sign Language. We leverage three VSR datasets: one in English, one in German with unrelated words and one in German containing the same target words as the mouthing dataset, to investigate the impact of task similarity in this setting. Our results demonstrate that multi-task learning improves both mouthing recognition and VSR accuracy as well as model robustness, suggesting that mouthing recognition should be treated as a distinct but related task to VSR. This research contributes to the field of SLR by proposing knowledge transfer from VSR to SLR datasets with limited mouthing annotations.
>
---
#### [new 073] 3D Reconstruction from Sketches
- **分类: cs.CV; cs.LG; 68T45; I.2.10**

- **简介: 该论文提出基于多张素描的3D场景重建方法，包含三步骤：通过对应点拼接素描、用CycleGAN生成真实图像、以MegaDepth网络估计深度图。构建了Zurich图像与人工素描配对数据集训练模型。解决了多视图素描到3D重建问题，但拼接模块对真实图纸泛化性不足，单素描重建表现优异。**

- **链接: [http://arxiv.org/pdf/2505.14621v1](http://arxiv.org/pdf/2505.14621v1)**

> **作者:** Abhimanyu Talwar; Julien Laasri
>
> **备注:** 6 pages, 8 figures, paper dated December 12, 2018
>
> **摘要:** We consider the problem of reconstructing a 3D scene from multiple sketches. We propose a pipeline which involves (1) stitching together multiple sketches through use of correspondence points, (2) converting the stitched sketch into a realistic image using a CycleGAN, and (3) estimating that image's depth-map using a pre-trained convolutional neural network based architecture called MegaDepth. Our contribution includes constructing a dataset of image-sketch pairs, the images for which are from the Zurich Building Database, and sketches have been generated by us. We use this dataset to train a CycleGAN for our pipeline's second step. We end up with a stitching process that does not generalize well to real drawings, but the rest of the pipeline that creates a 3D reconstruction from a single sketch performs quite well on a wide variety of drawings.
>
---
#### [new 074] Handloom Design Generation Using Generative Networks
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于生成对抗网络任务，旨在解决手织布设计自动化生成的挑战。通过结合前沿生成模型与风格迁移算法，研究其设计生成性能，并构建新数据集NeuralLoom，利用用户评分评估效果。**

- **链接: [http://arxiv.org/pdf/2505.14330v1](http://arxiv.org/pdf/2505.14330v1)**

> **作者:** Rajat Kanti Bhattacharjee; Meghali Nandi; Amrit Jha; Gunajit Kalita; Ferdous Ahmed Barbhuiya
>
> **摘要:** This paper proposes deep learning techniques of generating designs for clothing, focused on handloom fabric and discusses the associated challenges along with its application. The capability of generative neural network models in understanding artistic designs and synthesizing those is not yet explored well. In this work, multiple methods are employed incorporating the current state of the art generative models and style transfer algorithms to study and observe their performance for the task. The results are then evaluated through user score. This work also provides a new dataset NeuralLoom for the task of the design generation.
>
---
#### [new 075] An Edge AI Solution for Space Object Detection
- **分类: cs.CV; astro-ph.IM; cs.LG; cs.SY; eess.SY**

- **简介: 该论文提出基于SE模块、ViT和YOLOv9的边缘AI方案，解决低地轨道卫星高精度、低延迟空间物体检测难题，通过模型验证其实现了实时碰撞避让需求。**

- **链接: [http://arxiv.org/pdf/2505.13468v1](http://arxiv.org/pdf/2505.13468v1)**

> **作者:** Wenxuan Zhang; Peng Hu
>
> **备注:** Accepted as poster paper at the 2025 IEEE Canadian Conference on Electrical and Computer Engineering (CCECE)
>
> **摘要:** Effective Edge AI for space object detection (SOD) tasks that can facilitate real-time collision assessment and avoidance is essential with the increasing space assets in near-Earth orbits. In SOD, low Earth orbit (LEO) satellites must detect other objects with high precision and minimal delay. We explore an Edge AI solution based on deep-learning-based vision sensing for SOD tasks and propose a deep learning model based on Squeeze-and-Excitation (SE) layers, Vision Transformers (ViT), and YOLOv9 framework. We evaluate the performance of these models across various realistic SOD scenarios, demonstrating their ability to detect multiple satellites with high accuracy and very low latency.
>
---
#### [new 076] Physics-Driven Local-Whole Elastic Deformation Modeling for Point Cloud Representation Learning
- **分类: cs.CV**

- **简介: 该论文属于点云表示学习任务，解决现有方法忽视局部与整体结构关系及弹性变形建模的问题。提出物理驱动的双任务自监督模型，通过隐式场与力传播机制捕捉局部-整体变形关系，提升分类、分割等性能。**

- **链接: [http://arxiv.org/pdf/2505.13812v1](http://arxiv.org/pdf/2505.13812v1)**

> **作者:** Zhongyu Chen; Rong Zhao; Xie Han; Xindong Guo; Song Wang; Zherui Qiao
>
> **摘要:** Existing point cloud representation learning tend to learning the geometric distribution of objects through data-driven approaches, emphasizing structural features while overlooking the relationship between the local information and the whole structure. Local features reflect the fine-grained variations of an object, while the whole structure is determined by the interaction and combination of these local features, collectively defining the object's shape. In real-world, objects undergo elastic deformation under external forces, and this deformation gradually affects the whole structure through the propagation of forces from local regions, thereby altering the object's geometric properties. Inspired by this, we propose a physics-driven self-supervised learning method for point cloud representation, which captures the relationship between parts and the whole by constructing a local-whole force propagation mechanism. Specifically, we employ a dual-task encoder-decoder framework, integrating the geometric modeling capability of implicit fields with physics-driven elastic deformation. The encoder extracts features from the point cloud and its tetrahedral mesh representation, capturing both geometric and physical properties. These features are then fed into two decoders: one learns the whole geometric shape of the point cloud through an implicit field, while the other predicts local deformations using two specifically designed physics information loss functions, modeling the deformation relationship between local and whole shapes. Experimental results show that our method outperforms existing approaches in object classification, few-shot learning, and segmentation, demonstrating its effectiveness.
>
---
#### [new 077] VideoEval-Pro: Robust and Realistic Long Video Understanding Evaluation
- **分类: cs.CV**

- **简介: 该论文提出VideoEval-Pro，解决现有长视频理解(LVU)基准依赖多项选择题、题目存在强先验导致评估不准确的问题。通过开放短问答设计，要求模型真正理解完整视频，评估显示模型性能显著下降且MCQ高分无法预测开放题表现，证明新基准更可靠。**

- **链接: [http://arxiv.org/pdf/2505.14640v1](http://arxiv.org/pdf/2505.14640v1)**

> **作者:** Wentao Ma; Weiming Ren; Yiming Jia; Zhuofeng Li; Ping Nie; Ge Zhang; Wenhu Chen
>
> **备注:** Dataset: https://huggingface.co/datasets/TIGER-Lab/VideoEval-Pro, Project Webpage: https://tiger-ai-lab.github.io/VideoEval-Pro
>
> **摘要:** Large multimodal models (LMMs) have recently emerged as a powerful tool for long video understanding (LVU), prompting the development of standardized LVU benchmarks to evaluate their performance. However, our investigation reveals a rather sober lesson for existing LVU benchmarks. First, most existing benchmarks rely heavily on multiple-choice questions (MCQs), whose evaluation results are inflated due to the possibility of guessing the correct answer; Second, a significant portion of questions in these benchmarks have strong priors to allow models to answer directly without even reading the input video. For example, Gemini-1.5-Pro can achieve over 50\% accuracy given a random frame from a long video on Video-MME. We also observe that increasing the number of frames does not necessarily lead to improvement on existing benchmarks, which is counterintuitive. As a result, the validity and robustness of current LVU benchmarks are undermined, impeding a faithful assessment of LMMs' long-video understanding capability. To tackle this problem, we propose VideoEval-Pro, a realistic LVU benchmark containing questions with open-ended short-answer, which truly require understanding the entire video. VideoEval-Pro assesses both segment-level and full-video understanding through perception and reasoning tasks. By evaluating 21 proprietary and open-source video LMMs, we conclude the following findings: (1) video LMMs show drastic performance ($>$25\%) drops on open-ended questions compared with MCQs; (2) surprisingly, higher MCQ scores do not lead to higher open-ended scores on VideoEval-Pro; (3) compared to other MCQ benchmarks, VideoEval-Pro benefits more from increasing the number of input frames. Our results show that VideoEval-Pro offers a more realistic and reliable measure of long video understanding, providing a clearer view of progress in this domain.
>
---
#### [new 078] LoVR: A Benchmark for Long Video Retrieval in Multimodal Contexts
- **分类: cs.CV; cs.IR**

- **简介: 该论文属于长视频-文本检索任务，针对现有基准视频时长短、字幕质量低、标注粗糙的问题，提出LoVR基准。包含467个长视频及4万余条精细片段，设计自动标注生成与优化框架，及语义融合方法生成高质量字幕，提升数据质量和挑战性，揭示现有方法局限。**

- **链接: [http://arxiv.org/pdf/2505.13928v1](http://arxiv.org/pdf/2505.13928v1)**

> **作者:** Qifeng Cai; Hao Liang; Hejun Dong; Meiyi Qiang; Ruichuan An; Zhaoyang Han; Zhengzhou Zhu; Bin Cui; Wentao Zhang
>
> **摘要:** Long videos contain a vast amount of information, making video-text retrieval an essential and challenging task in multimodal learning. However, existing benchmarks suffer from limited video duration, low-quality captions, and coarse annotation granularity, which hinder the evaluation of advanced video-text retrieval methods. To address these limitations, we introduce LoVR, a benchmark specifically designed for long video-text retrieval. LoVR contains 467 long videos and over 40,804 fine-grained clips with high-quality captions. To overcome the issue of poor machine-generated annotations, we propose an efficient caption generation framework that integrates VLM automatic generation, caption quality scoring, and dynamic refinement. This pipeline improves annotation accuracy while maintaining scalability. Furthermore, we introduce a semantic fusion method to generate coherent full-video captions without losing important contextual information. Our benchmark introduces longer videos, more detailed captions, and a larger-scale dataset, presenting new challenges for video understanding and retrieval. Extensive experiments on various advanced embedding models demonstrate that LoVR is a challenging benchmark, revealing the limitations of current approaches and providing valuable insights for future research. We release the code and dataset link at https://github.com/TechNomad-ds/LoVR-benchmark
>
---
#### [new 079] Grouping First, Attending Smartly: Training-Free Acceleration for Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于加速扩散Transformer的任务，旨在解决其高计算成本问题。提出GRAT方法：通过分组连续token并共享注意力计算，利用预训练模型的局部注意力特性，结合结构化区域限制（如周边块），在无需微调下加速推理（如生成8K图像提速35.8倍），同时保持生成质量。验证于图像及视频生成任务。**

- **链接: [http://arxiv.org/pdf/2505.14687v1](http://arxiv.org/pdf/2505.14687v1)**

> **作者:** Sucheng Ren; Qihang Yu; Ju He; Alan Yuille; Liang-Chieh Chen
>
> **备注:** Project website at oliverrensu.github.io/project/GRAT
>
> **摘要:** Diffusion-based Transformers have demonstrated impressive generative capabilities, but their high computational costs hinder practical deployment, for example, generating an $8192\times 8192$ image can take over an hour on an A100 GPU. In this work, we propose GRAT (\textbf{GR}ouping first, \textbf{AT}tending smartly), a training-free attention acceleration strategy for fast image and video generation without compromising output quality. The key insight is to exploit the inherent sparsity in learned attention maps (which tend to be locally focused) in pretrained Diffusion Transformers and leverage better GPU parallelism. Specifically, GRAT first partitions contiguous tokens into non-overlapping groups, aligning both with GPU execution patterns and the local attention structures learned in pretrained generative Transformers. It then accelerates attention by having all query tokens within the same group share a common set of attendable key and value tokens. These key and value tokens are further restricted to structured regions, such as surrounding blocks or criss-cross regions, significantly reducing computational overhead (e.g., attaining a \textbf{35.8$\times$} speedup over full attention when generating $8192\times 8192$ images) while preserving essential attention patterns and long-range context. We validate GRAT on pretrained Flux and HunyuanVideo for image and video generation, respectively. In both cases, GRAT achieves substantially faster inference without any fine-tuning, while maintaining the performance of full attention. We hope GRAT will inspire future research on accelerating Diffusion Transformers for scalable visual generation.
>
---
#### [new 080] Self-Supervised Learning for Image Segmentation: A Comprehensive Survey
- **分类: cs.CV**

- **简介: 该综述论文聚焦自监督学习在图像分割中的应用，解决标注数据依赖问题。通过分析150+篇论文，分类 pretext任务、下游任务及基准数据集，总结研究进展并提出未来方向，为领域提供系统性指导。（99字）**

- **链接: [http://arxiv.org/pdf/2505.13584v1](http://arxiv.org/pdf/2505.13584v1)**

> **作者:** Thangarajah Akilan; Nusrat Jahan; Wandong Zhang
>
> **备注:** 22 pages, 19 figures, to be submitted for a possible IEEE publication
>
> **摘要:** Supervised learning demands large amounts of precisely annotated data to achieve promising results. Such data curation is labor-intensive and imposes significant overhead regarding time and costs. Self-supervised learning (SSL) partially overcomes these limitations by exploiting vast amounts of unlabeled data and creating surrogate (pretext or proxy) tasks to learn useful representations without manual labeling. As a result, SSL has become a powerful machine learning (ML) paradigm for solving several practical downstream computer vision problems, such as classification, detection, and segmentation. Image segmentation is the cornerstone of many high-level visual perception applications, including medical imaging, intelligent transportation, agriculture, and surveillance. Although there is substantial research potential for developing advanced algorithms for SSL-based semantic segmentation, a comprehensive study of existing methodologies is essential to trace advances and guide emerging researchers. This survey thoroughly investigates over 150 recent image segmentation articles, particularly focusing on SSL. It provides a practical categorization of pretext tasks, downstream tasks, and commonly used benchmark datasets for image segmentation research. It concludes with key observations distilled from a large body of literature and offers future directions to make this research field more accessible and comprehensible for readers.
>
---
#### [new 081] Accuracy and Fairness of Facial Recognition Technology in Low-Quality Police Images: An Experiment With Synthetic Faces
- **分类: cs.CV; stat.AP**

- **简介: 该研究评估低质量图像下FRT的准确性和公平性，针对执法中常见图像退化问题。通过合成人脸模拟对比度、模糊等五种退化，测试发现女性和黑人群体错误率更高（尤其黑人女性），但准确率仍优于传统方法。强调需监管确保公平与可靠性。**

- **链接: [http://arxiv.org/pdf/2505.14320v1](http://arxiv.org/pdf/2505.14320v1)**

> **作者:** Maria Cuellar; Hon Kiu; To; Arush Mehrotra
>
> **摘要:** Facial recognition technology (FRT) is increasingly used in criminal investigations, yet most evaluations of its accuracy rely on high-quality images, unlike those often encountered by law enforcement. This study examines how five common forms of image degradation--contrast, brightness, motion blur, pose shift, and resolution--affect FRT accuracy and fairness across demographic groups. Using synthetic faces generated by StyleGAN3 and labeled with FairFace, we simulate degraded images and evaluate performance using Deepface with ArcFace loss in 1:n identification tasks. We perform an experiment and find that false positive rates peak near baseline image quality, while false negatives increase as degradation intensifies--especially with blur and low resolution. Error rates are consistently higher for women and Black individuals, with Black females most affected. These disparities raise concerns about fairness and reliability when FRT is used in real-world investigative contexts. Nevertheless, even under the most challenging conditions and for the most affected subgroups, FRT accuracy remains substantially higher than that of many traditional forensic methods. This suggests that, if appropriately validated and regulated, FRT should be considered a valuable investigative tool. However, algorithmic accuracy alone is not sufficient: we must also evaluate how FRT is used in practice, including user-driven data manipulation. Such cases underscore the need for transparency and oversight in FRT deployment to ensure both fairness and forensic validity.
>
---
#### [new 082] 4D-ROLLS: 4D Radar Occupancy Learning via LiDAR Supervision
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出4D-ROLLS，首个基于LiDAR监督的4D雷达弱监督占用估计方法。针对传统传感器在恶劣环境性能差的问题，通过生成伪LiDAR标签分阶段训练雷达模型，提升其占用估计的鲁棒性与精度，并验证其跨数据集有效性及在BEV分割等任务中的应用潜力，模型轻量化实现30Hz推理速度。**

- **链接: [http://arxiv.org/pdf/2505.13905v1](http://arxiv.org/pdf/2505.13905v1)**

> **作者:** Ruihan Liu; Xiaoyi Wu; Xijun Chen; Liang Hu; Yunjiang Lou
>
> **摘要:** A comprehensive understanding of 3D scenes is essential for autonomous vehicles (AVs), and among various perception tasks, occupancy estimation plays a central role by providing a general representation of drivable and occupied space. However, most existing occupancy estimation methods rely on LiDAR or cameras, which perform poorly in degraded environments such as smoke, rain, snow, and fog. In this paper, we propose 4D-ROLLS, the first weakly supervised occupancy estimation method for 4D radar using the LiDAR point cloud as the supervisory signal. Specifically, we introduce a method for generating pseudo-LiDAR labels, including occupancy queries and LiDAR height maps, as multi-stage supervision to train the 4D radar occupancy estimation model. Then the model is aligned with the occupancy map produced by LiDAR, fine-tuning its accuracy in occupancy estimation. Extensive comparative experiments validate the exceptional performance of 4D-ROLLS. Its robustness in degraded environments and effectiveness in cross-dataset training are qualitatively demonstrated. The model is also seamlessly transferred to downstream tasks BEV segmentation and point cloud occupancy prediction, highlighting its potential for broader applications. The lightweight network enables 4D-ROLLS model to achieve fast inference speeds at about 30 Hz on a 4060 GPU. The code of 4D-ROLLS will be made available at https://github.com/CLASS-Lab/4D-ROLLS.
>
---
#### [new 083] GeoVLM: Improving Automated Vehicle Geolocalisation Using Vision-Language Matching
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的跨视角地理定位任务，旨在解决现有方法无法将正确匹配图像排为首位的问题。提出GeoVLM模型，利用视觉语言模型的零样本能力生成可解释语言描述，并通过重排序提升最佳匹配准确率。在标准及新提出的Cross-View UK数据集上验证，效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.13669v1](http://arxiv.org/pdf/2505.13669v1)**

> **作者:** Barkin Dagda; Muhammad Awais; Saber Fallah
>
> **摘要:** Cross-view geo-localisation identifies coarse geographical position of an automated vehicle by matching a ground-level image to a geo-tagged satellite image from a database. Despite the advancements in Cross-view geo-localisation, significant challenges still persist such as similar looking scenes which makes it challenging to find the correct match as the top match. Existing approaches reach high recall rates but they still fail to rank the correct image as the top match. To address this challenge, this paper proposes GeoVLM, a novel approach which uses the zero-shot capabilities of vision language models to enable cross-view geo-localisation using interpretable cross-view language descriptions. GeoVLM is a trainable reranking approach which improves the best match accuracy of cross-view geo-localisation. GeoVLM is evaluated on standard benchmark VIGOR and University-1652 and also through real-life driving environments using Cross-View United Kingdom, a new benchmark dataset introduced in this paper. The results of the paper show that GeoVLM improves retrieval performance of cross-view geo-localisation compared to the state-of-the-art methods with the help of explainable natural language descriptions. The code is available at https://github.com/CAV-Research-Lab/GeoVLM
>
---
#### [new 084] VisualQuality-R1: Reasoning-Induced Image Quality Assessment via Reinforcement Learning to Rank
- **分类: cs.CV**

- **简介: 该论文提出VisualQuality-R1，属于无参考图像质量评估（NR-IQA）任务。针对传统方法在推理能力与相对质量建模上的不足，采用强化学习排序算法，通过成对图像比较生成质量评分，结合Thurstone模型计算比较概率，并用连续奖励优化。实验显示其优于现有方法，支持多数据集训练且能生成人类可解释的描述。**

- **链接: [http://arxiv.org/pdf/2505.14460v1](http://arxiv.org/pdf/2505.14460v1)**

> **作者:** Tianhe Wu; Jian Zou; Jie Liang; Lei Zhang; Kede Ma
>
> **摘要:** DeepSeek-R1 has demonstrated remarkable effectiveness in incentivizing reasoning and generalization capabilities of large language models (LLMs) through reinforcement learning. Nevertheless, the potential of reasoning-induced computational modeling has not been thoroughly explored in the context of image quality assessment (IQA), a task critically dependent on visual reasoning. In this paper, we introduce VisualQuality-R1, a reasoning-induced no-reference IQA (NR-IQA) model, and we train it with reinforcement learning to rank, a learning algorithm tailored to the intrinsically relative nature of visual quality. Specifically, for a pair of images, we employ group relative policy optimization to generate multiple quality scores for each image. These estimates are then used to compute comparative probabilities of one image having higher quality than the other under the Thurstone model. Rewards for each quality estimate are defined using continuous fidelity measures rather than discretized binary labels. Extensive experiments show that the proposed VisualQuality-R1 consistently outperforms discriminative deep learning-based NR-IQA models as well as a recent reasoning-induced quality regression method. Moreover, VisualQuality-R1 is capable of generating contextually rich, human-aligned quality descriptions, and supports multi-dataset training without requiring perceptual scale realignment. These features make VisualQuality-R1 especially well-suited for reliably measuring progress in a wide range of image processing tasks like super-resolution and image generation.
>
---
#### [new 085] Unify Graph Learning with Text: Unleashing LLM Potentials for Session Search
- **分类: cs.CV; cs.AI; cs.IR; I.2; H.3.3**

- **简介: 该论文针对会话搜索任务，解决现有方法忽略图结构或文本语义的问题。提出Symbolic Graph Ranker（SGR），通过符号语法规则将会话图转为文本输入LLM，并设计自监督任务（如链接预测、生成对比学习）增强LLM对图结构的理解，实验验证了其优势。**

- **链接: [http://arxiv.org/pdf/2505.14156v1](http://arxiv.org/pdf/2505.14156v1)**

> **作者:** Songhao Wu; Quan Tu; Hong Liu; Jia Xu; Zhongyi Liu; Guannan Zhang; Ran Wang; Xiuying Chen; Rui Yan
>
> **摘要:** Session search involves a series of interactive queries and actions to fulfill user's complex information need. Current strategies typically prioritize sequential modeling for deep semantic understanding, overlooking the graph structure in interactions. While some approaches focus on capturing structural information, they use a generalized representation for documents, neglecting the word-level semantic modeling. In this paper, we propose Symbolic Graph Ranker (SGR), which aims to take advantage of both text-based and graph-based approaches by leveraging the power of recent Large Language Models (LLMs). Concretely, we first introduce a set of symbolic grammar rules to convert session graph into text. This allows integrating session history, interaction process, and task instruction seamlessly as inputs for the LLM. Moreover, given the natural discrepancy between LLMs pre-trained on textual corpora, and the symbolic language we produce using our graph-to-text grammar, our objective is to enhance LLMs' ability to capture graph structures within a textual format. To achieve this, we introduce a set of self-supervised symbolic learning tasks including link prediction, node content generation, and generative contrastive learning, to enable LLMs to capture the topological information from coarse-grained to fine-grained. Experiment results and comprehensive analysis on two benchmark datasets, AOL and Tiangong-ST, confirm the superiority of our approach. Our paradigm also offers a novel and effective methodology that bridges the gap between traditional search strategies and modern LLMs.
>
---
#### [new 086] Vision-Language Modeling Meets Remote Sensing: Models, Datasets and Perspectives
- **分类: cs.CV**

- **简介: 该论文综述了视觉语言模型（VLM）在遥感领域的应用，解决跨模态数据分析与交互问题。通过分类VLM范式（对比学习、指令调优、图像生成），分析模型架构、预训练策略及数据集，总结任务适配方法，并探讨未来方向如跨模态对齐、多模态数据构建等。**

- **链接: [http://arxiv.org/pdf/2505.14361v1](http://arxiv.org/pdf/2505.14361v1)**

> **作者:** Xingxing Weng; Chao Pang; Gui-Song Xia
>
> **备注:** Accepted by IEEE Geoscience and Remote Sensing Magazine
>
> **摘要:** Vision-language modeling (VLM) aims to bridge the information gap between images and natural language. Under the new paradigm of first pre-training on massive image-text pairs and then fine-tuning on task-specific data, VLM in the remote sensing domain has made significant progress. The resulting models benefit from the absorption of extensive general knowledge and demonstrate strong performance across a variety of remote sensing data analysis tasks. Moreover, they are capable of interacting with users in a conversational manner. In this paper, we aim to provide the remote sensing community with a timely and comprehensive review of the developments in VLM using the two-stage paradigm. Specifically, we first cover a taxonomy of VLM in remote sensing: contrastive learning, visual instruction tuning, and text-conditioned image generation. For each category, we detail the commonly used network architecture and pre-training objectives. Second, we conduct a thorough review of existing works, examining foundation models and task-specific adaptation methods in contrastive-based VLM, architectural upgrades, training strategies and model capabilities in instruction-based VLM, as well as generative foundation models with their representative downstream applications. Third, we summarize datasets used for VLM pre-training, fine-tuning, and evaluation, with an analysis of their construction methodologies (including image sources and caption generation) and key properties, such as scale and task adaptability. Finally, we conclude this survey with insights and discussions on future research directions: cross-modal representation alignment, vague requirement comprehension, explanation-driven model reliability, continually scalable model capabilities, and large-scale datasets featuring richer modalities and greater challenges.
>
---
#### [new 087] Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像检测任务，旨在解决现有检测器因训练数据偏差导致泛化性差的问题。提出Dual Data Alignment（DDA）方法，同步对齐像素与频率域特征，消除高频失真引发的虚假关联，并构建新测试集验证检测器在新生成模型上的性能，实验显示显著提升跨数据集泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.14359v1](http://arxiv.org/pdf/2505.14359v1)**

> **作者:** Ruoxin Chen; Junwei Xi; Zhiyuan Yan; Ke-Yue Zhang; Shuang Wu; Jingyi Xie; Xu Chen; Lei Xu; Isabel Guan; Taiping Yao; Shouhong Ding
>
> **备注:** 12 Pages, 9 figures
>
> **摘要:** Existing detectors are often trained on biased datasets, leading to the possibility of overfitting on non-causal image attributes that are spuriously correlated with real/synthetic labels. While these biased features enhance performance on the training data, they result in substantial performance degradation when applied to unbiased datasets. One common solution is to perform dataset alignment through generative reconstruction, matching the semantic content between real and synthetic images. However, we revisit this approach and show that pixel-level alignment alone is insufficient. The reconstructed images still suffer from frequency-level misalignment, which can perpetuate spurious correlations. To illustrate, we observe that reconstruction models tend to restore the high-frequency details lost in real images (possibly due to JPEG compression), inadvertently creating a frequency-level misalignment, where synthetic images appear to have richer high-frequency content than real ones. This misalignment leads to models associating high-frequency features with synthetic labels, further reinforcing biased cues. To resolve this, we propose Dual Data Alignment (DDA), which aligns both the pixel and frequency domains. Moreover, we introduce two new test sets: DDA-COCO, containing DDA-aligned synthetic images for testing detector performance on the most aligned dataset, and EvalGEN, featuring the latest generative models for assessing detectors under new generative architectures such as visual auto-regressive generators. Finally, our extensive evaluations demonstrate that a detector trained exclusively on DDA-aligned MSCOCO could improve across 8 diverse benchmarks by a non-trivial margin, showing a +7.2% on in-the-wild benchmarks, highlighting the improved generalizability of unbiased detectors.
>
---
#### [new 088] M3Depth: Wavelet-Enhanced Depth Estimation on Mars via Mutual Boosting of Dual-Modal Data
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出M3Depth模型，针对火星稀疏纹理环境下传统深度估计性能下降问题，通过小波变换增强的卷积核捕捉低频特征，结合表面法线约束的一致性损失及互促进像素优化模块，显著提升深度估计精度（16%），适用于真实火星场景。**

- **链接: [http://arxiv.org/pdf/2505.14159v1](http://arxiv.org/pdf/2505.14159v1)**

> **作者:** Junjie Li; Jiawei Wang; Miyu Li; Yu Liu; Yumei Wang; Haitao Xu
>
> **摘要:** Depth estimation plays a great potential role in obstacle avoidance and navigation for further Mars exploration missions. Compared to traditional stereo matching, learning-based stereo depth estimation provides a data-driven approach to infer dense and precise depth maps from stereo image pairs. However, these methods always suffer performance degradation in environments with sparse textures and lacking geometric constraints, such as the unstructured terrain of Mars. To address these challenges, we propose M3Depth, a depth estimation model tailored for Mars rovers. Considering the sparse and smooth texture of Martian terrain, which is primarily composed of low-frequency features, our model incorporates a convolutional kernel based on wavelet transform that effectively captures low-frequency response and expands the receptive field. Additionally, we introduce a consistency loss that explicitly models the complementary relationship between depth map and surface normal map, utilizing the surface normal as a geometric constraint to enhance the accuracy of depth estimation. Besides, a pixel-wise refinement module with mutual boosting mechanism is designed to iteratively refine both depth and surface normal predictions. Experimental results on synthetic Mars datasets with depth annotations show that M3Depth achieves a significant 16% improvement in depth estimation accuracy compared to other state-of-the-art methods in depth estimation. Furthermore, the model demonstrates strong applicability in real-world Martian scenarios, offering a promising solution for future Mars exploration missions.
>
---
#### [new 089] A Review of Vision-Based Assistive Systems for Visually Impaired People: Technologies, Applications, and Future Directions
- **分类: cs.CV**

- **简介: 该论文综述了基于视觉的辅助系统最新进展，旨在帮助视障人士独立生活。聚焦障碍检测、导航及用户交互技术，总结现有方法，分析应用并探讨未来方向。（99字）**

- **链接: [http://arxiv.org/pdf/2505.14298v1](http://arxiv.org/pdf/2505.14298v1)**

> **作者:** Fulong Yao; Wenju Zhou; Huosheng Hu
>
> **摘要:** Visually impaired individuals rely heavily on accurate and timely information about obstacles and their surrounding environments to achieve independent living. In recent years, significant progress has been made in the development of assistive technologies, particularly vision-based systems, that enhance mobility and facilitate interaction with the external world in both indoor and outdoor settings. This paper presents a comprehensive review of recent advances in assistive systems designed for the visually impaired, with a focus on state-of-the-art technologies in obstacle detection, navigation, and user interaction. In addition, emerging trends and future directions in visual guidance systems are discussed.
>
---
#### [new 090] UHD Image Dehazing via anDehazeFormer with Atmospheric-aware KV Cache
- **分类: cs.CV**

- **简介: 该论文提出anDehazeFormer框架，针对超高清图像去雾任务中训练速度慢、内存消耗高的问题。通过自适应归一化机制和大气散射感知的KV缓存机制，提升训练效率5倍，实现4K/8K实时处理，同时保持去雾质量并提供可解释性分析。**

- **链接: [http://arxiv.org/pdf/2505.14010v1](http://arxiv.org/pdf/2505.14010v1)**

> **作者:** Pu Wang; Pengwen Dai; Chen Wu; Yeying Jin; Dianjie Lu; Guijuan Zhang; Youshan Zhang; Zhuoran Zheng
>
> **备注:** Under review
>
> **摘要:** In this paper, we propose an efficient visual transformer framework for ultra-high-definition (UHD) image dehazing that addresses the key challenges of slow training speed and high memory consumption for existing methods. Our approach introduces two key innovations: 1) an \textbf{a}daptive \textbf{n}ormalization mechanism inspired by the nGPT architecture that enables ultra-fast and stable training with a network with a restricted range of parameter expressions; and 2) we devise an atmospheric scattering-aware KV caching mechanism that dynamically optimizes feature preservation based on the physical haze formation model. The proposed architecture improves the training convergence speed by \textbf{5 $\times$} while reducing memory overhead, enabling real-time processing of 50 high-resolution images per second on an RTX4090 GPU. Experimental results show that our approach maintains state-of-the-art dehazing quality while significantly improving computational efficiency for 4K/8K image restoration tasks. Furthermore, we provide a new dehazing image interpretable method with the help of an integrated gradient attribution map. Our code can be found here: https://anonymous.4open.science/r/anDehazeFormer-632E/README.md.
>
---
#### [new 091] Scaling Vision Mamba Across Resolutions via Fractal Traversal
- **分类: cs.CV**

- **简介: 该论文针对Vision Mamba模型在视觉任务中因2D序列化导致的空间连续性破坏和分辨率适应性差的问题，提出FractalMamba++。通过Hilbert曲线分形序列化保留空间局部性，设计Cross-State Routing增强长程依赖，添加Positional-Relation模块修复曲线转折处邻接关系。实验显示其在多任务中性能优于原有模型，尤其在高分辨率场景表现突出。（99字）**

- **链接: [http://arxiv.org/pdf/2505.14062v1](http://arxiv.org/pdf/2505.14062v1)**

> **作者:** Bo Li; Haoke Xiao; Lv Tang
>
> **备注:** Work in progressing
>
> **摘要:** Vision Mamba has recently emerged as a promising alternative to Transformer-based architectures, offering linear complexity in sequence length while maintaining strong modeling capacity. However, its adaptation to visual inputs is hindered by challenges in 2D-to-1D patch serialization and weak scalability across input resolutions. Existing serialization strategies such as raster scanning disrupt local spatial continuity and limit the model's ability to generalize across scales. In this paper, we propose FractalMamba++, a robust vision backbone that leverages fractal-based patch serialization via Hilbert curves to preserve spatial locality and enable seamless resolution adaptability. To address long-range dependency fading in high-resolution inputs, we further introduce a Cross-State Routing (CSR) mechanism that enhances global context propagation through selective state reuse. Additionally, we propose a Positional-Relation Capture (PRC) module to recover local adjacency disrupted by curve inflection points. Extensive experiments on image classification, semantic segmentation, object detection, and change detection demonstrate that FractalMamba++ consistently outperforms previous Mamba-based backbones, particularly under high-resolution settings.
>
---
#### [new 092] Visual Agentic Reinforcement Fine-Tuning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Visual-ARFT方法，旨在提升开源视觉语言模型（LVLMs）的多模态代理能力。针对现有模型在图像辅助推理和工具使用上的不足，通过强化微调使模型能实时网页搜索、编写图像处理代码（如裁剪/旋转），并构建MAT基准测试其搜索与编码能力。实验显示其显著超越基线及GPT-4o，验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.14246v1](http://arxiv.org/pdf/2505.14246v1)**

> **作者:** Ziyu Liu; Yuhang Zang; Yushan Zou; Zijian Liang; Xiaoyi Dong; Yuhang Cao; Haodong Duan; Dahua Lin; Jiaqi Wang
>
> **备注:** project url: https://github.com/Liuziyu77/Visual-RFT/tree/main/Visual-ARFT
>
> **摘要:** A key trend in Large Reasoning Models (e.g., OpenAI's o3) is the native agentic ability to use external tools such as web browsers for searching and writing/executing code for image manipulation to think with images. In the open-source research community, while significant progress has been made in language-only agentic abilities such as function calling and tool integration, the development of multi-modal agentic capabilities that involve truly thinking with images, and their corresponding benchmarks, are still less explored. This work highlights the effectiveness of Visual Agentic Reinforcement Fine-Tuning (Visual-ARFT) for enabling flexible and adaptive reasoning abilities for Large Vision-Language Models (LVLMs). With Visual-ARFT, open-source LVLMs gain the ability to browse websites for real-time information updates and write code to manipulate and analyze input images through cropping, rotation, and other image processing techniques. We also present a Multi-modal Agentic Tool Bench (MAT) with two settings (MAT-Search and MAT-Coding) designed to evaluate LVLMs' agentic search and coding abilities. Our experimental results demonstrate that Visual-ARFT outperforms its baseline by +18.6% F1 / +13.0% EM on MAT-Coding and +10.3% F1 / +8.7% EM on MAT-Search, ultimately surpassing GPT-4o. Visual-ARFT also achieves +29.3 F1% / +25.9% EM gains on existing multi-hop QA benchmarks such as 2Wiki and HotpotQA, demonstrating strong generalization capabilities. Our findings suggest that Visual-ARFT offers a promising path toward building robust and generalizable multimodal agents.
>
---
#### [new 093] Speculative Decoding Reimagined for Multimodal Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态大语言模型（MLLM）推理加速任务，旨在解决现有推测解码方法在多模态场景下加速效果不佳的问题。提出Multimodal Speculative Decoding（MSD），通过分离处理文本与视觉token，并采用两阶段训练策略（先语言建模后加入视觉数据），提升draft模型的语言和视觉能力，实现LLaVA模型2.29-2.46倍加速。**

- **链接: [http://arxiv.org/pdf/2505.14260v1](http://arxiv.org/pdf/2505.14260v1)**

> **作者:** Luxi Lin; Zhihang Lin; Zhanpeng Zeng; Rongrong Ji
>
> **备注:** 12 pages
>
> **摘要:** This paper introduces Multimodal Speculative Decoding (MSD) to accelerate Multimodal Large Language Models (MLLMs) inference. Speculative decoding has been shown to accelerate Large Language Models (LLMs) without sacrificing accuracy. However, current speculative decoding methods for MLLMs fail to achieve the same speedup as they do for LLMs. To address this, we reimagine speculative decoding specifically for MLLMs. Our analysis of MLLM characteristics reveals two key design principles for MSD: (1) Text and visual tokens have fundamentally different characteristics and need to be processed separately during drafting. (2) Both language modeling ability and visual perception capability are crucial for the draft model. For the first principle, MSD decouples text and visual tokens in the draft model, allowing each to be handled based on its own characteristics. For the second principle, MSD uses a two-stage training strategy: In stage one, the draft model is trained on text-only instruction-tuning datasets to improve its language modeling ability. In stage two, MSD gradually introduces multimodal data to enhance the visual perception capability of the draft model. Experiments show that MSD boosts inference speed by up to $2.29\times$ for LLaVA-1.5-7B and up to $2.46\times$ for LLaVA-1.5-13B on multimodal benchmarks, demonstrating its effectiveness. Our code is available at https://github.com/Lyn-Lucy/MSD.
>
---
#### [new 094] Model Steering: Learning with a Reference Model Improves Generalization Bounds and Scaling Laws
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文属于模型引导任务，旨在通过参考模型提升目标模型的泛化与数据效率。针对现有方法缺乏理论支持的问题，提出基于分布鲁棒优化的DRRho框架，理论分析其优势，并开发DRRho-CLIP方法，实验验证其优于传统方法。**

- **链接: [http://arxiv.org/pdf/2505.06699v3](http://arxiv.org/pdf/2505.06699v3)**

> **作者:** Xiyuan Wei; Ming Lin; Fanjiang Ye; Fengguang Song; Liangliang Cao; My T. Thai; Tianbao Yang
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** This paper formalizes an emerging learning paradigm that uses a trained model as a reference to guide and enhance the training of a target model through strategic data selection or weighting, named $\textbf{model steering}$. While ad-hoc methods have been used in various contexts, including the training of large foundation models, its underlying principles remain insufficiently understood, leading to sub-optimal performance. In this work, we propose a theory-driven framework for model steering called $\textbf{DRRho risk minimization}$, which is rooted in Distributionally Robust Optimization (DRO). Through a generalization analysis, we provide theoretical insights into why this approach improves generalization and data efficiency compared to training without a reference model. To the best of our knowledge, this is the first time such theoretical insights are provided for the new learning paradigm, which significantly enhance our understanding and practice of model steering. Building on these insights and the connection between contrastive learning and DRO, we introduce a novel method for Contrastive Language-Image Pretraining (CLIP) with a reference model, termed DRRho-CLIP. Extensive experiments validate the theoretical insights, reveal a superior scaling law compared to CLIP without a reference model, and demonstrate its strength over existing heuristic approaches.
>
---
#### [new 095] Towards Efficient Multi-Scale Deformable Attention on NPU
- **分类: cs.PF; cs.CV**

- **简介: 该论文属于视觉任务中的模型优化任务，旨在解决多尺度可变形注意力（MSDA）在NPU加速器上因随机访问采样导致的效率问题。通过协同设计内存访问与计算策略，提出硬件优化方法，实现前向/反向计算加速，较基线分别提升5.9×/8.9×，并优化训练效率达7.3×。**

- **链接: [http://arxiv.org/pdf/2505.14022v1](http://arxiv.org/pdf/2505.14022v1)**

> **作者:** Chenghuan Huang; Zhigeng Xu; Chong Sun; Chen Li; Ziyang Ma
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** Multi-scale deformable attention (MSDA) is a flexible and powerful feature extraction mechanism for visual tasks, but its random-access grid sampling strategy poses significant optimization challenges, especially on domain-specific accelerators such as NPUs. In this work, we present a co-design approach that systematically rethinks memory access and computation strategies for MSDA on the Ascend NPU architecture. With this co-design approach, our implementation supports both efficient forward and backward computation, is fully adapted for training workloads, and incorporates a suite of hardware-aware optimizations. Extensive experiments show that our solution achieves up to $5.9\times$ (forward), $8.9\times$ (backward), and $7.3\times$ (end-to-end training) speedup over the grid sample-based baseline, and $1.9\times$, $2.4\times$, and $2.0\times$ acceleration over the latest vendor library, respectively.
>
---
#### [new 096] Two Experts Are All You Need for Steering Thinking: Reinforcing Cognitive Effort in MoE Reasoning Models Without Additional Training
- **分类: cs.AI; cs.CL; cs.CV; cs.IR; cs.LG**

- **简介: 该论文属于优化MoE推理模型的任务，旨在解决其推理过程中的过思考和欠思考问题。提出RICE方法，通过nPMI识别"认知专家"（如触发<think>的模块），在推理阶段引导结构化思维，提升推理效率与跨领域泛化能力，实验显示其优于现有方法且无需额外训练。**

- **链接: [http://arxiv.org/pdf/2505.14681v1](http://arxiv.org/pdf/2505.14681v1)**

> **作者:** Mengru Wang; Xingyu Chen; Yue Wang; Zhiwei He; Jiahao Xu; Tian Liang; Qiuzhi Liu; Yunzhi Yao; Wenxuan Wang; Ruotian Ma; Haitao Mi; Ningyu Zhang; Zhaopeng Tu; Xiaolong Li; Dong Yu
>
> **备注:** Work in progress
>
> **摘要:** Mixture-of-Experts (MoE) architectures within Large Reasoning Models (LRMs) have achieved impressive reasoning capabilities by selectively activating experts to facilitate structured cognitive processes. Despite notable advances, existing reasoning models often suffer from cognitive inefficiencies like overthinking and underthinking. To address these limitations, we introduce a novel inference-time steering methodology called Reinforcing Cognitive Experts (RICE), designed to improve reasoning performance without additional training or complex heuristics. Leveraging normalized Pointwise Mutual Information (nPMI), we systematically identify specialized experts, termed ''cognitive experts'' that orchestrate meta-level reasoning operations characterized by tokens like ''<think>''. Empirical evaluations with leading MoE-based LRMs (DeepSeek-R1 and Qwen3-235B) on rigorous quantitative and scientific reasoning benchmarks demonstrate noticeable and consistent improvements in reasoning accuracy, cognitive efficiency, and cross-domain generalization. Crucially, our lightweight approach substantially outperforms prevalent reasoning-steering techniques, such as prompt design and decoding constraints, while preserving the model's general instruction-following skills. These results highlight reinforcing cognitive experts as a promising, practical, and interpretable direction to enhance cognitive efficiency within advanced reasoning models.
>
---
#### [new 097] Direction-Aware Neural Acoustic Fields for Few-Shot Interpolation of Ambisonic Impulse Responses
- **分类: eess.AS; cs.AI; cs.CV; cs.LG; cs.SD**

- **简介: 该论文属于声场建模任务，解决现有神经场方法无法准确捕捉声场方向特性的问题。提出方向感知神经声场（DANF），利用Ambisonic格式RIR显式整合方向信息，并设计方向损失函数，同时研究其少样本适配新环境（如低秩调整）的能力。**

- **链接: [http://arxiv.org/pdf/2505.13617v1](http://arxiv.org/pdf/2505.13617v1)**

> **作者:** Christopher Ick; Gordon Wichern; Yoshiki Masuyama; François Germain; Jonathan Le Roux
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** The characteristics of a sound field are intrinsically linked to the geometric and spatial properties of the environment surrounding a sound source and a listener. The physics of sound propagation is captured in a time-domain signal known as a room impulse response (RIR). Prior work using neural fields (NFs) has allowed learning spatially-continuous representations of RIRs from finite RIR measurements. However, previous NF-based methods have focused on monaural omnidirectional or at most binaural listeners, which does not precisely capture the directional characteristics of a real sound field at a single point. We propose a direction-aware neural field (DANF) that more explicitly incorporates the directional information by Ambisonic-format RIRs. While DANF inherently captures spatial relations between sources and listeners, we further propose a direction-aware loss. In addition, we investigate the ability of DANF to adapt to new rooms in various ways including low-rank adaptation.
>
---
#### [new 098] Automated Fetal Biometry Assessment with Deep Ensembles using Sparse-Sampling of 2D Intrapartum Ultrasound Images
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文提出自动化胎儿生物测量系统，解决产科超声评估中人工测量的观察者差异问题。通过分类标准超声平面、分割胎儿头部与耻骨联合、计算进展角（AoP）和头-耻骨距离（HSD），采用深度学习集成与稀疏采样优化，提升测量可靠性与泛化能力，实验结果验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.14572v1](http://arxiv.org/pdf/2505.14572v1)**

> **作者:** Jayroop Ramesh; Valentin Bacher; Mark C. Eid; Hoda Kalabizadeh; Christian Rupprecht; Ana IL Namburete; Pak-Hei Yeung; Madeleine K. Wyburd; Nicola K. Dinsdale
>
> **备注:** Top 5 in MICCAI IUGC 2024: Intrapartum Ultrasound Grand Challenge & Runners up in Classification!
>
> **摘要:** The International Society of Ultrasound advocates Intrapartum Ultrasound (US) Imaging in Obstetrics and Gynecology (ISUOG) to monitor labour progression through changes in fetal head position. Two reliable ultrasound-derived parameters that are used to predict outcomes of instrumental vaginal delivery are the angle of progression (AoP) and head-symphysis distance (HSD). In this work, as part of the Intrapartum Ultrasounds Grand Challenge (IUGC) 2024, we propose an automated fetal biometry measurement pipeline to reduce intra- and inter-observer variability and improve measurement reliability. Our pipeline consists of three key tasks: (i) classification of standard planes (SP) from US videos, (ii) segmentation of fetal head and pubic symphysis from the detected SPs, and (iii) computation of the AoP and HSD from the segmented regions. We perform sparse sampling to mitigate class imbalances and reduce spurious correlations in task (i), and utilize ensemble-based deep learning methods for task (i) and (ii) to enhance generalizability under different US acquisition settings. Finally, to promote robustness in task iii) with respect to the structural fidelity of measurements, we retain the largest connected components and apply ellipse fitting to the segmentations. Our solution achieved ACC: 0.9452, F1: 0.9225, AUC: 0.983, MCC: 0.8361, DSC: 0.918, HD: 19.73, ASD: 5.71, $\Delta_{AoP}$: 8.90 and $\Delta_{HSD}$: 14.35 across an unseen hold-out set of 4 patients and 224 US frames. The results from the proposed automated pipeline can improve the understanding of labour arrest causes and guide the development of clinical risk stratification tools for efficient and effective prenatal care.
>
---
#### [new 099] Adversarial Training from Mean Field Perspective
- **分类: cs.LG; cs.CV; stat.ML**

- **简介: 该论文属于对抗训练的理论分析任务，旨在解析其训练动态与网络结构限制。通过提出基于平均场的新框架，推导不同范数的对抗损失上界，证明无捷径网络难对抗训练、对抗训练降低容量，且宽度与输入输出维度可缓解问题。**

- **链接: [http://arxiv.org/pdf/2505.14021v1](http://arxiv.org/pdf/2505.14021v1)**

> **作者:** Soichiro Kumano; Hiroshi Kera; Toshihiko Yamasaki
>
> **备注:** NeurIPS23
>
> **摘要:** Although adversarial training is known to be effective against adversarial examples, training dynamics are not well understood. In this study, we present the first theoretical analysis of adversarial training in random deep neural networks without any assumptions on data distributions. We introduce a new theoretical framework based on mean field theory, which addresses the limitations of existing mean field-based approaches. Based on this framework, we derive (empirically tight) upper bounds of $\ell_q$ norm-based adversarial loss with $\ell_p$ norm-based adversarial examples for various values of $p$ and $q$. Moreover, we prove that networks without shortcuts are generally not adversarially trainable and that adversarial training reduces network capacity. We also show that network width alleviates these issues. Furthermore, we present the various impacts of the input and output dimensions on the upper bounds and time evolution of the weight variance.
>
---
#### [new 100] Large-Scale Multi-Character Interaction Synthesis
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于大规模多角色互动合成任务，旨在解决现有方法无法生成自然且协调的多角色互动及过渡问题。提出包含互动空间建模和过渡规划网络的生成框架，解决数据不足与时空协调规划难题，实验验证了方法的有效性和可扩展性。**

- **链接: [http://arxiv.org/pdf/2505.14087v1](http://arxiv.org/pdf/2505.14087v1)**

> **作者:** Ziyi Chang; He Wang; George Alex Koulieris; Hubert P. H. Shum
>
> **摘要:** Generating large-scale multi-character interactions is a challenging and important task in character animation. Multi-character interactions involve not only natural interactive motions but also characters coordinated with each other for transition. For example, a dance scenario involves characters dancing with partners and also characters coordinated to new partners based on spatial and temporal observations. We term such transitions as coordinated interactions and decompose them into interaction synthesis and transition planning. Previous methods of single-character animation do not consider interactions that are critical for multiple characters. Deep-learning-based interaction synthesis usually focuses on two characters and does not consider transition planning. Optimization-based interaction synthesis relies on manually designing objective functions that may not generalize well. While crowd simulation involves more characters, their interactions are sparse and passive. We identify two challenges to multi-character interaction synthesis, including the lack of data and the planning of transitions among close and dense interactions. Existing datasets either do not have multiple characters or do not have close and dense interactions. The planning of transitions for multi-character close and dense interactions needs both spatial and temporal considerations. We propose a conditional generative pipeline comprising a coordinatable multi-character interaction space for interaction synthesis and a transition planning network for coordinations. Our experiments demonstrate the effectiveness of our proposed pipeline for multicharacter interaction synthesis and the applications facilitated by our method show the scalability and transferability.
>
---
#### [new 101] FedCTTA: A Collaborative Approach to Continual Test-Time Adaptation in Federated Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出FedCTTA框架，解决联邦学习中模型部署后因数据分布偏移导致的性能下降问题。通过基于噪声样本输出分布的相似性聚合，避免直接特征共享，在保护隐私同时实现计算高效、内存恒定的持续测试时适应，提升异构场景下的模型性能。**

- **链接: [http://arxiv.org/pdf/2505.13643v1](http://arxiv.org/pdf/2505.13643v1)**

> **作者:** Rakibul Hasan Rajib; Md Akil Raihan Iftee; Mir Sazzat Hossain; A. K. M. Mahbubur Rahman; Sajib Mistry; M Ashraful Amin; Amin Ahsan Ali
>
> **备注:** 8 pages, 5 figures, Accepted In IJCNN 2025
>
> **摘要:** Federated Learning (FL) enables collaborative model training across distributed clients without sharing raw data, making it ideal for privacy-sensitive applications. However, FL models often suffer performance degradation due to distribution shifts between training and deployment. Test-Time Adaptation (TTA) offers a promising solution by allowing models to adapt using only test samples. However, existing TTA methods in FL face challenges such as computational overhead, privacy risks from feature sharing, and scalability concerns due to memory constraints. To address these limitations, we propose Federated Continual Test-Time Adaptation (FedCTTA), a privacy-preserving and computationally efficient framework for federated adaptation. Unlike prior methods that rely on sharing local feature statistics, FedCTTA avoids direct feature exchange by leveraging similarity-aware aggregation based on model output distributions over randomly generated noise samples. This approach ensures adaptive knowledge sharing while preserving data privacy. Furthermore, FedCTTA minimizes the entropy at each client for continual adaptation, enhancing the model's confidence in evolving target distributions. Our method eliminates the need for server-side training during adaptation and maintains a constant memory footprint, making it scalable even as the number of clients or training rounds increases. Extensive experiments show that FedCTTA surpasses existing methods across diverse temporal and spatial heterogeneity scenarios.
>
---
#### [new 102] NOVA: A Benchmark for Anomaly Localization and Clinical Reasoning in Brain MRI
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出NOVA基准，用于评估模型在脑MRI中对未知异常的定位与临床推理能力。针对现有测试集无法有效检验罕见或新型病理的问题，构建含900例真实病例的评估数据集，涵盖281种罕见病及多样化扫描协议，并通过专家标注和临床描述，测试模型跨分布泛化能力。实验证明现有模型表现显著下降，验证NOVA对开放世界医学推理的挑战性。**

- **链接: [http://arxiv.org/pdf/2505.14064v1](http://arxiv.org/pdf/2505.14064v1)**

> **作者:** Cosmin I. Bercea; Jun Li; Philipp Raffler; Evamaria O. Riedel; Lena Schmitzer; Angela Kurz; Felix Bitzer; Paula Roßmüller; Julian Canisius; Mirjam L. Beyrle; Che Liu; Wenjia Bai; Bernhard Kainz; Julia A. Schnabel; Benedikt Wiestler
>
> **摘要:** In many real-world applications, deployed models encounter inputs that differ from the data seen during training. Out-of-distribution detection identifies whether an input stems from an unseen distribution, while open-world recognition flags such inputs to ensure the system remains robust as ever-emerging, previously $unknown$ categories appear and must be addressed without retraining. Foundation and vision-language models are pre-trained on large and diverse datasets with the expectation of broad generalization across domains, including medical imaging. However, benchmarking these models on test sets with only a few common outlier types silently collapses the evaluation back to a closed-set problem, masking failures on rare or truly novel conditions encountered in clinical use. We therefore present $NOVA$, a challenging, real-life $evaluation-only$ benchmark of $\sim$900 brain MRI scans that span 281 rare pathologies and heterogeneous acquisition protocols. Each case includes rich clinical narratives and double-blinded expert bounding-box annotations. Together, these enable joint assessment of anomaly localisation, visual captioning, and diagnostic reasoning. Because NOVA is never used for training, it serves as an $extreme$ stress-test of out-of-distribution generalisation: models must bridge a distribution gap both in sample appearance and in semantic space. Baseline results with leading vision-language models (GPT-4o, Gemini 2.0 Flash, and Qwen2.5-VL-72B) reveal substantial performance drops across all tasks, establishing NOVA as a rigorous testbed for advancing models that can detect, localize, and reason about truly unknown anomalies.
>
---
#### [new 103] End-to-end fully-binarized network design: from Generic Learned Thermometer to Block Pruning
- **分类: cs.LG; cs.AR; cs.CV; eess.IV; stat.ML**

- **简介: 该论文属于二值神经网络（BNN）设计任务，旨在解决现有BNN忽略输入数据优化及模型冗余问题。提出Generic Learned Thermometer（GLT）改进输入编码，通过学习非线性量化阈值替代传统ADC；结合块剪枝与知识蒸馏优化网络结构，实现轻量（<1Mb）、全二值化模型，提升准确率并适配实时推理。**

- **链接: [http://arxiv.org/pdf/2505.13462v1](http://arxiv.org/pdf/2505.13462v1)**

> **作者:** Thien Nguyen; William Guicquero
>
> **备注:** Accepted to IEEE AICAS 2025
>
> **摘要:** Existing works on Binary Neural Network (BNN) mainly focus on model's weights and activations while discarding considerations on the input raw data. This article introduces Generic Learned Thermometer (GLT), an encoding technique to improve input data representation for BNN, relying on learning non linear quantization thresholds. This technique consists in multiple data binarizations which can advantageously replace a conventional Analog to Digital Conversion (ADC) that uses natural binary coding. Additionally, we jointly propose a compact topology with light-weight grouped convolutions being trained thanks to block pruning and Knowledge Distillation (KD), aiming at reducing furthermore the model size so as its computational complexity. We show that GLT brings versatility to the BNN by intrinsically performing global tone mapping, enabling significant accuracy gains in practice (demonstrated by simulations on the STL-10 and VWW datasets). Moreover, when combining GLT with our proposed block-pruning technique, we successfully achieve lightweight (under 1Mb), fully-binarized models with limited accuracy degradation while being suitable for in-sensor always-on inference use cases.
>
---
#### [new 104] EuLearn: A 3D database for learning Euler characteristics
- **分类: cs.CG; cs.CV; cs.LG; math.DG; math.GT**

- **简介: 该论文提出EuLearn数据库，旨在通过3D数据（网格、点云、标量场）训练机器学习模型识别拓扑特征。针对传统神经网络在拓扑分类（如属数）上的不足，开发非欧氏采样方法及改进PointNet/Transformer架构，提升模型对复杂拓扑结构（如自结表面）的识别能力。**

- **链接: [http://arxiv.org/pdf/2505.13539v1](http://arxiv.org/pdf/2505.13539v1)**

> **作者:** Rodrigo Fritz; Pablo Suárez-Serrato; Victor Mijangos; Anayanzi D. Martinez-Hernandez; Eduardo Ivan Velazquez Richards
>
> **备注:** 35 pages, many figures. Datasets and source code publicly available at https://huggingface.co/datasets/appliedgeometry/EuLearn and https://github.com/appliedgeometry/EuLearn_db
>
> **摘要:** We present EuLearn, the first surface datasets equitably representing a diversity of topological types. We designed our embedded surfaces of uniformly varying genera relying on random knots, thus allowing our surfaces to knot with themselves. EuLearn contributes new topological datasets of meshes, point clouds, and scalar fields in 3D. We aim to facilitate the training of machine learning systems that can discern topological features. We experimented with specific emblematic 3D neural network architectures, finding that their vanilla implementations perform poorly on genus classification. To enhance performance, we developed a novel, non-Euclidean, statistical sampling method adapted to graph and manifold data. We also introduce adjacency-informed adaptations of PointNet and Transformer architectures that rely on our non-Euclidean sampling strategy. Our results demonstrate that incorporating topological information into deep learning workflows significantly improves performance on these otherwise challenging EuLearn datasets.
>
---
#### [new 105] Neural Inverse Scattering with Score-based Regularization
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于逆散射成像任务，旨在解决高对比度场景下联合估计目标图像与散射场的难题。现有方法（如总变差正则化）成像质量受限，本文提出基于分数的神经场方法，整合去噪分数函数作为图像先验，提升联合估计精度，实验显示优于现有技术。**

- **链接: [http://arxiv.org/pdf/2505.14560v1](http://arxiv.org/pdf/2505.14560v1)**

> **作者:** Yuan Gao; Wenhan Guo; Yu Sun
>
> **摘要:** Inverse scattering is a fundamental challenge in many imaging applications, ranging from microscopy to remote sensing. Solving this problem often requires jointly estimating two unknowns -- the image and the scattering field inside the object -- necessitating effective image prior to regularize the inference. In this paper, we propose a regularized neural field (NF) approach which integrates the denoising score function used in score-based generative models. The neural field formulation offers convenient flexibility to performing joint estimation, while the denoising score function imposes the rich structural prior of images. Our results on three high-contrast simulated objects show that the proposed approach yields a better imaging quality compared to the state-of-the-art NF approach, where regularization is based on total variation.
>
---
#### [new 106] Textual Steering Vectors Can Improve Visual Understanding in Multimodal Large Language Models
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文研究通过文本引导向量提升多模态大模型视觉理解。针对MLLMs缺乏有效行为引导技术的问题，提出利用文本模型生成steering向量（如稀疏自编码器、均值偏移），实验显示其显著提高空间关系和计数任务准确率（+7.3%），优于提示方法且泛化性好。**

- **链接: [http://arxiv.org/pdf/2505.14071v1](http://arxiv.org/pdf/2505.14071v1)**

> **作者:** Woody Haosheng Gan; Deqing Fu; Julian Asilis; Ollie Liu; Dani Yogatama; Vatsal Sharan; Robin Jia; Willie Neiswanger
>
> **摘要:** Steering methods have emerged as effective and targeted tools for guiding large language models' (LLMs) behavior without modifying their parameters. Multimodal large language models (MLLMs), however, do not currently enjoy the same suite of techniques, due in part to their recency and architectural diversity. Inspired by this gap, we investigate whether MLLMs can be steered using vectors derived from their text-only LLM backbone, via sparse autoencoders (SAEs), mean shift, and linear probing. We find that text-derived steering consistently enhances multimodal accuracy across diverse MLLM architectures and visual tasks. In particular, mean shift boosts spatial relationship accuracy on CV-Bench by up to +7.3% and counting accuracy by up to +3.3%, outperforming prompting and exhibiting strong generalization to out-of-distribution datasets. These results highlight textual steering vectors as a powerful, efficient mechanism for enhancing grounding in MLLMs with minimal additional data collection and computational overhead.
>
---
#### [new 107] FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer
- **分类: cs.LG; cs.CV**

- **简介: 论文针对Kolmogorov-Arnold Transformer（KAT）训练速度慢的问题，分析发现内存等待和反向传播中梯度累积低效是主因，提出FlashKAT通过优化内核减少原子操作和慢内存访问，提升训练速度86.5倍并降低误差，属于模型训练效率优化任务。**

- **链接: [http://arxiv.org/pdf/2505.13813v1](http://arxiv.org/pdf/2505.13813v1)**

> **作者:** Matthew Raffel; Lizhong Chen
>
> **摘要:** The Kolmogorov-Arnold Network (KAN) has been gaining popularity as an alternative to the multi-layer perceptron (MLP) with its increased expressiveness and interpretability. However, the KAN can be orders of magnitude slower due to its increased computational cost and training instability, limiting its applicability to larger-scale tasks. Recently, the Kolmogorov-Arnold Transformer (KAT) has been proposed, which can achieve FLOPs similar to the traditional Transformer with MLPs by leveraging Group-Rational KAN (GR-KAN). Unfortunately, despite the comparable FLOPs, our characterizations reveal that the KAT is still 123x slower in training speeds, indicating that there are other performance bottlenecks beyond FLOPs. In this paper, we conduct a series of experiments to understand the root cause of the slowdown in KAT. We uncover that the slowdown can be isolated to memory stalls and, more specifically, in the backward pass of GR-KAN caused by inefficient gradient accumulation. To address this memory bottleneck, we propose FlashKAT, which builds on our restructured kernel that minimizes gradient accumulation with atomic adds and accesses to slow memory. Evaluations demonstrate that FlashKAT can achieve a training speedup of 86.5x compared with the state-of-the-art KAT, while reducing rounding errors in the coefficient gradients. Our code is available at https://github.com/OSU-STARLAB/FlashKAT.
>
---
#### [new 108] Automated Quality Evaluation of Cervical Cytopathology Whole Slide Images Based on Content Analysis
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出基于AI和TBS标准的宫颈细胞病理全玻片图像自动化质量评估方法，解决传统人工评估主观性强、成本高、耗时且可靠性低的问题。通过目标检测、分类、分割模型量化染色质量、细胞计数等指标，结合XGBoost构建综合评分模型，实验显示其速度和一致性优势。**

- **链接: [http://arxiv.org/pdf/2505.13875v1](http://arxiv.org/pdf/2505.13875v1)**

> **作者:** Lanlan Kang; Jian Wang; Jian QIn; Yiqin Liang; Yongjun He
>
> **备注:** 12 pages, 10 figures
>
> **摘要:** The ThinPrep Cytologic Test (TCT) is the most widely used method for cervical cancer screening, and the sample quality directly impacts the accuracy of the diagnosis. Traditional manual evaluation methods rely on the observation of pathologist under microscopes. These methods exhibit high subjectivity, high cost, long duration, and low reliability. With the development of computer-aided diagnosis (CAD), an automated quality assessment system that performs at the level of a professional pathologist is necessary. To address this need, we propose a fully automated quality assessment method for Cervical Cytopathology Whole Slide Images (WSIs) based on The Bethesda System (TBS) diagnostic standards, artificial intelligence algorithms, and the characteristics of clinical data. The method analysis the context of WSIs to quantify quality evaluation metrics which are focused by TBS such as staining quality, cell counts and cell mass proportion through multiple models including object detection, classification and segmentation. Subsequently, the XGBoost model is used to mine the attention paid by pathologists to different quality evaluation metrics when evaluating samples, thereby obtaining a comprehensive WSI sample score calculation model. Experimental results on 100 WSIs demonstrate that the proposed evaluation method has significant advantages in terms of speed and consistency.
>
---
#### [new 109] Learning Wavelet-Sparse FDK for 3D Cone-Beam CT Reconstruction
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对3D锥束CT重建中FDK算法噪声敏感及深度学习方法复杂度高的问题，提出基于波浪变换稀疏化的改进FDK网络。通过在余弦加权与滤波阶段嵌入可学习参数并采用波浪稀疏表示，将参数减少93.75%，保持计算效率的同时提升抗噪性和图像一致性，便于临床部署。**

- **链接: [http://arxiv.org/pdf/2505.13579v1](http://arxiv.org/pdf/2505.13579v1)**

> **作者:** Yipeng Sun; Linda-Sophie Schneider; Chengze Ye; Mingxuan Gu; Siyuan Mei; Siming Bayer; Andreas Maier
>
> **备注:** Accepted by Fully3D 2025
>
> **摘要:** Cone-Beam Computed Tomography (CBCT) is essential in medical imaging, and the Feldkamp-Davis-Kress (FDK) algorithm is a popular choice for reconstruction due to its efficiency. However, FDK is susceptible to noise and artifacts. While recent deep learning methods offer improved image quality, they often increase computational complexity and lack the interpretability of traditional methods. In this paper, we introduce an enhanced FDK-based neural network that maintains the classical algorithm's interpretability by selectively integrating trainable elements into the cosine weighting and filtering stages. Recognizing the challenge of a large parameter space inherent in 3D CBCT data, we leverage wavelet transformations to create sparse representations of the cosine weights and filters. This strategic sparsification reduces the parameter count by $93.75\%$ without compromising performance, accelerates convergence, and importantly, maintains the inference computational cost equivalent to the classical FDK algorithm. Our method not only ensures volumetric consistency and boosts robustness to noise, but is also designed for straightforward integration into existing CT reconstruction pipelines. This presents a pragmatic enhancement that can benefit clinical applications, particularly in environments with computational limitations.
>
---
#### [new 110] Breaking the Compression Ceiling: Data-Free Pipeline for Ultra-Efficient Delta Compression
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出UltraDelta，一种无需数据的高效delta压缩方法，解决存储多任务微调模型的高开销问题。通过三层策略：方差分配稀疏性、分布感知压缩、迹范数重缩放，在LLM等模型中实现133x至40x压缩，同时保持性能，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.13563v1](http://arxiv.org/pdf/2505.13563v1)**

> **作者:** Xiaohui Wang; Peng Ye; Chenyu Huang; Shenghe Zheng; Bo Zhang; Wanli Ouyang; Tao Chen
>
> **摘要:** With the rise of the fine-tuned--pretrained paradigm, storing numerous fine-tuned models for multi-tasking creates significant storage overhead. Delta compression alleviates this by storing only the pretrained model and the highly compressed delta weights (the differences between fine-tuned and pretrained model weights). However, existing methods fail to maintain both high compression and performance, and often rely on data. To address these challenges, we propose UltraDelta, the first data-free delta compression pipeline that achieves both ultra-high compression and strong performance. UltraDelta is designed to minimize redundancy, maximize information, and stabilize performance across inter-layer, intra-layer, and global dimensions, using three key components: (1) Variance-Based Mixed Sparsity Allocation assigns sparsity based on variance, giving lower sparsity to high-variance layers to preserve inter-layer information. (2) Distribution-Aware Compression applies uniform quantization and then groups parameters by value, followed by group-wise pruning, to better preserve intra-layer distribution. (3) Trace-Norm-Guided Rescaling uses the trace norm of delta weights to estimate a global rescaling factor, improving model stability under higher compression. Extensive experiments across (a) large language models (fine-tuned on LLaMA-2 7B and 13B) with up to 133x, (b) general NLP models (RoBERTa-base, T5-base) with up to 800x, (c) vision models (ViT-B/32, ViT-L/14) with up to 400x, and (d) multi-modal models (BEiT-3) with 40x compression ratio, demonstrate that UltraDelta consistently outperforms existing methods, especially under ultra-high compression.
>
---
#### [new 111] KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于个性化食谱推荐与生成任务，旨在解决现有系统缺乏知识图谱与大语言模型深度整合的问题。提出KERL系统，结合食物知识图谱与LLM，通过实体提取、子图检索增强上下文理解，生成符合用户约束的食谱及营养信息，并构建基准数据集验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14629v1](http://arxiv.org/pdf/2505.14629v1)**

> **作者:** Fnu Mohbat; Mohammed J Zaki
>
> **备注:** Accepted at ACL 2025
>
> **摘要:** Recent advances in large language models (LLMs) and the abundance of food data have resulted in studies to improve food understanding using LLMs. Despite several recommendation systems utilizing LLMs and Knowledge Graphs (KGs), there has been limited research on integrating food related KGs with LLMs. We introduce KERL, a unified system that leverages food KGs and LLMs to provide personalized food recommendations and generates recipes with associated micro-nutritional information. Given a natural language question, KERL extracts entities, retrieves subgraphs from the KG, which are then fed into the LLM as context to select the recipes that satisfy the constraints. Next, our system generates the cooking steps and nutritional information for each recipe. To evaluate our approach, we also develop a benchmark dataset by curating recipe related questions, combined with constraints and personal preferences. Through extensive experiments, we show that our proposed KG-augmented LLM significantly outperforms existing approaches, offering a complete and coherent solution for food recommendation, recipe generation, and nutritional analysis. Our code and benchmark datasets are publicly available at https://github.com/mohbattharani/KERL.
>
---
#### [new 112] Bridge the Gap between Past and Future: Siamese Model Optimization for Context-Aware Document Ranking
- **分类: cs.IR; cs.CV; H.3.3**

- **简介: 论文属于信息检索文档排序任务，旨在利用未来行为弥补历史数据局限。提出Siamese模型优化框架，包含历史模型ForeRanker（仅用历史行为）与未来模型（整合预测行为），通过协同训练及动态门控知识蒸馏提升排序效果，解决用户意图演变捕捉难题。**

- **链接: [http://arxiv.org/pdf/2505.14180v1](http://arxiv.org/pdf/2505.14180v1)**

> **作者:** Songhao Wu; Quan Tu; Mingjie Zhong; Hong Liu; Jia Xu; Jinjie Gu; Rui Yan
>
> **摘要:** In the realm of information retrieval, users often engage in multi-turn interactions with search engines to acquire information, leading to the formation of sequences of user feedback behaviors. Leveraging the session context has proven to be beneficial for inferring user search intent and document ranking. A multitude of approaches have been proposed to exploit in-session context for improved document ranking. Despite these advances, the limitation of historical session data for capturing evolving user intent remains a challenge. In this work, we explore the integration of future contextual information into the session context to enhance document ranking. We present the siamese model optimization framework, comprising a history-conditioned model and a future-aware model. The former processes only the historical behavior sequence, while the latter integrates both historical and anticipated future behaviors. Both models are trained collaboratively using the supervised labels and pseudo labels predicted by the other. The history-conditioned model, referred to as ForeRanker, progressively learns future-relevant information to enhance ranking, while it singly uses historical session at inference time. To mitigate inconsistencies during training, we introduce the peer knowledge distillation method with a dynamic gating mechanism, allowing models to selectively incorporate contextual information. Experimental results on benchmark datasets demonstrate the effectiveness of our ForeRanker, showcasing its superior performance compared to existing methods.
>
---
#### [new 113] EmoGist: Efficient In-Context Learning for Visual Emotion Understanding
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出EmoGist方法，属于视觉情绪分类任务。针对图像情绪表达的上下文依赖与细微差异问题，通过预生成多版本情绪标签解释（基于图像聚类分析），测试时检索匹配解释并输入轻量VLM完成分类。无需训练，实验显示在Memotion和FI数据集上F1值提升达8-13分。**

- **链接: [http://arxiv.org/pdf/2505.14660v1](http://arxiv.org/pdf/2505.14660v1)**

> **作者:** Ronald Seoh; Dan Goldwasser
>
> **摘要:** In this paper, we introduce EmoGist, a training-free, in-context learning method for performing visual emotion classification with LVLMs. The key intuition of our approach is that context-dependent definition of emotion labels could allow more accurate predictions of emotions, as the ways in which emotions manifest within images are highly context dependent and nuanced. EmoGist pre-generates multiple explanations of emotion labels, by analyzing the clusters of example images belonging to each category. At test time, we retrieve a version of explanation based on embedding similarity, and feed it to a fast VLM for classification. Through our experiments, we show that EmoGist allows up to 13 points improvement in micro F1 scores with the multi-label Memotion dataset, and up to 8 points in macro F1 in the multi-class FI dataset.
>
---
#### [new 114] Exploring Image Quality Assessment from a New Perspective: Pupil Size
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像质量评估(IQA)任务，研究瞳孔大小与图像质量评价的关系。通过对比自由观察与IQA任务中的瞳孔变化，发现评估时视觉注意机制被激活且瞳孔变化与图像质量相关，为改进客观IQA方法及开发新主观评估手段提供理论支持。**

- **链接: [http://arxiv.org/pdf/2505.13841v1](http://arxiv.org/pdf/2505.13841v1)**

> **作者:** Yixuan Gao; Xiongkuo Min; Guangtao Zhai
>
> **摘要:** This paper explores how the image quality assessment (IQA) task affects the cognitive processes of people from the perspective of pupil size and studies the relationship between pupil size and image quality. Specifically, we first invited subjects to participate in a subjective experiment, which includes two tasks: free observation and IQA. In the free observation task, subjects did not need to perform any action, and they only needed to observe images as they usually do with an album. In the IQA task, subjects were required to score images according to their overall impression of image quality. Then, by analyzing the difference in pupil size between the two tasks, we find that people may activate the visual attention mechanism when evaluating image quality. Meanwhile, we also find that the change in pupil size is closely related to image quality in the IQA task. For future research on IQA, this research can not only provide a theoretical basis for the objective IQA method and promote the development of more effective objective IQA methods, but also provide a new subjective IQA method for collecting the authentic subjective impression of image quality.
>
---
#### [new 115] From stability of Langevin diffusion to convergence of proximal MCMC for non-log-concave sampling
- **分类: stat.ML; cs.CV; cs.LG**

- **简介: 该论文属于非凸非光滑分布采样任务，解决Proximal SGLA在非凸势函数下的收敛性问题。通过证明ULA对漂移近似的稳定性并结合Moreau包络性质，首次理论证明PSGLA的收敛性，实验显示其比SGLD更快且保持恢复效果。**

- **链接: [http://arxiv.org/pdf/2505.14177v1](http://arxiv.org/pdf/2505.14177v1)**

> **作者:** Marien Renaud; Valentin De Bortoli; Arthur Leclaire; Nicolas Papadakis
>
> **摘要:** We consider the problem of sampling distributions stemming from non-convex potentials with Unadjusted Langevin Algorithm (ULA). We prove the stability of the discrete-time ULA to drift approximations under the assumption that the potential is strongly convex at infinity. In many context, e.g. imaging inverse problems, potentials are non-convex and non-smooth. Proximal Stochastic Gradient Langevin Algorithm (PSGLA) is a popular algorithm to handle such potentials. It combines the forward-backward optimization algorithm with a ULA step. Our main stability result combined with properties of the Moreau envelope allows us to derive the first proof of convergence of the PSGLA for non-convex potentials. We empirically validate our methodology on synthetic data and in the context of imaging inverse problems. In particular, we observe that PSGLA exhibits faster convergence rates than Stochastic Gradient Langevin Algorithm for posterior sampling while preserving its restoration properties.
>
---
#### [new 116] Neural Video Compression with Context Modulation
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出基于上下文调制的神经视频压缩方法，解决现有方法对参考帧信息利用不足的问题。通过flow orientation挖掘参考帧与预测帧的关联生成定向时间上下文，并引入context compensation模块优化传播的上下文，减少冗余信息。实验显示其比特率较H.266/VVC降低22.7%，优于前人方法10.1%。**

- **链接: [http://arxiv.org/pdf/2505.14541v1](http://arxiv.org/pdf/2505.14541v1)**

> **作者:** Chuanbo Tang; Zhuoyuan Li; Yifan Bian; Li Li; Dong Liu
>
> **备注:** 11 pages, 8 figures, accepted by CVPR 2025
>
> **摘要:** Efficient video coding is highly dependent on exploiting the temporal redundancy, which is usually achieved by extracting and leveraging the temporal context in the emerging conditional coding-based neural video codec (NVC). Although the latest NVC has achieved remarkable progress in improving the compression performance, the inherent temporal context propagation mechanism lacks the ability to sufficiently leverage the reference information, limiting further improvement. In this paper, we address the limitation by modulating the temporal context with the reference frame in two steps. Specifically, we first propose the flow orientation to mine the inter-correlation between the reference frame and prediction frame for generating the additional oriented temporal context. Moreover, we introduce the context compensation to leverage the oriented context to modulate the propagated temporal context generated from the propagated reference feature. Through the synergy mechanism and decoupling loss supervision, the irrelevant propagated information can be effectively eliminated to ensure better context modeling. Experimental results demonstrate that our codec achieves on average 22.7% bitrate reduction over the advanced traditional video codec H.266/VVC, and offers an average 10.1% bitrate saving over the previous state-of-the-art NVC DCVC-FM. The code is available at https://github.com/Austin4USTC/DCMVC.
>
---
#### [new 117] End-to-end Cortical Surface Reconstruction from Clinical Magnetic Resonance Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出端到端神经网络，通过合成数据训练，解决临床MRI异构性问题，处理任意对比度和分辨率扫描，误差减半，实现快速皮质表面重建。任务为皮质表面建模，解决现有工具依赖高分辨率T1w扫描的局限，方法包括模板网格变形及开源代码。**

- **链接: [http://arxiv.org/pdf/2505.14017v1](http://arxiv.org/pdf/2505.14017v1)**

> **作者:** Jesper Duemose Nielsen; Karthik Gopinath; Andrew Hoopes; Adrian Dalca; Colin Magdamo; Steven Arnold; Sudeshna Das; Axel Thielscher; Juan Eugenio Iglesias; Oula Puonti
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Surface-based cortical analysis is valuable for a variety of neuroimaging tasks, such as spatial normalization, parcellation, and gray matter (GM) thickness estimation. However, most tools for estimating cortical surfaces work exclusively on scans with at least 1 mm isotropic resolution and are tuned to a specific magnetic resonance (MR) contrast, often T1-weighted (T1w). This precludes application using most clinical MR scans, which are very heterogeneous in terms of contrast and resolution. Here, we use synthetic domain-randomized data to train the first neural network for explicit estimation of cortical surfaces from scans of any contrast and resolution, without retraining. Our method deforms a template mesh to the white matter (WM) surface, which guarantees topological correctness. This mesh is further deformed to estimate the GM surface. We compare our method to recon-all-clinical (RAC), an implicit surface reconstruction method which is currently the only other tool capable of processing heterogeneous clinical MR scans, on ADNI and a large clinical dataset (n=1,332). We show a approximately 50 % reduction in cortical thickness error (from 0.50 to 0.24 mm) with respect to RAC and better recovery of the aging-related cortical thinning patterns detected by FreeSurfer on high-resolution T1w scans. Our method enables fast and accurate surface reconstruction of clinical scans, allowing studies (1) with sample sizes far beyond what is feasible in a research setting, and (2) of clinical populations that are difficult to enroll in research studies. The code is publicly available at https://github.com/simnibs/brainnet.
>
---
#### [new 118] Bronchovascular Tree-Guided Weakly Supervised Learning Method for Pulmonary Segment Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文提出弱监督学习方法AHSL，用于肺段分割，解决标注困难及边界不明显问题。通过结合支气管血管树与肺叶层级解剖先验，设计分阶段策略及一致性损失，提升分割精度与边界平滑性，实验验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.13911v1](http://arxiv.org/pdf/2505.13911v1)**

> **作者:** Ruijie Zhao; Zuopeng Tan; Xiao Xue; Longfei Zhao; Bing Li; Zicheng Liao; Ying Ming; Jiaru Wang; Ran Xiao; Sirong Piao; Rui Zhao; Qiqi Xu; Wei Song
>
> **摘要:** Pulmonary segment segmentation is crucial for cancer localization and surgical planning. However, the pixel-wise annotation of pulmonary segments is laborious, as the boundaries between segments are indistinguishable in medical images. To this end, we propose a weakly supervised learning (WSL) method, termed Anatomy-Hierarchy Supervised Learning (AHSL), which consults the precise clinical anatomical definition of pulmonary segments to perform pulmonary segment segmentation. Since pulmonary segments reside within the lobes and are determined by the bronchovascular tree, i.e., artery, airway and vein, the design of the loss function is founded on two principles. First, segment-level labels are utilized to directly supervise the output of the pulmonary segments, ensuring that they accurately encompass the appropriate bronchovascular tree. Second, lobe-level supervision indirectly oversees the pulmonary segment, ensuring their inclusion within the corresponding lobe. Besides, we introduce a two-stage segmentation strategy that incorporates bronchovascular priori information. Furthermore, a consistency loss is proposed to enhance the smoothness of segment boundaries, along with an evaluation metric designed to measure the smoothness of pulmonary segment boundaries. Visual inspection and evaluation metrics from experiments conducted on a private dataset demonstrate the effectiveness of our method.
>
---
#### [new 119] Adversarially Pretrained Transformers may be Universally Robust In-Context Learners
- **分类: cs.LG; cs.CV; stat.ML**

- **简介: 该论文研究对抗预训练Transformer的鲁棒性，解决对抗训练成本高及下游任务泛化问题。提出通过多样任务对抗预训练的Transformer可作为基础模型，利用in-context learning无需参数更新即泛化至多新任务，理论证明其鲁棒性源于稳健特征聚焦，同时指出单层模型局限、准确率-鲁棒性权衡及需大量示例。（99字）**

- **链接: [http://arxiv.org/pdf/2505.14042v1](http://arxiv.org/pdf/2505.14042v1)**

> **作者:** Soichiro Kumano; Hiroshi Kera; Toshihiko Yamasaki
>
> **摘要:** Adversarial training is one of the most effective adversarial defenses, but it incurs a high computational cost. In this study, we show that transformers adversarially pretrained on diverse tasks can serve as robust foundation models and eliminate the need for adversarial training in downstream tasks. Specifically, we theoretically demonstrate that through in-context learning, a single adversarially pretrained transformer can robustly generalize to multiple unseen tasks without any additional training, i.e., without any parameter updates. This robustness stems from the model's focus on robust features and its resistance to attacks that exploit non-predictive features. Besides these positive findings, we also identify several limitations. Under certain conditions (though unrealistic), no universally robust single-layer transformers exist. Moreover, robust transformers exhibit an accuracy--robustness trade-off and require a large number of in-context demonstrations. The code is available at https://github.com/s-kumano/universally-robust-in-context-learner.
>
---
#### [new 120] XDementNET: An Explainable Attention Based Deep Convolutional Network to Detect Alzheimer Progression from MRI data
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出XDementNET，一种结合多残差、空间及注意力机制的深度网络，用于通过MRI精准分类阿尔茨海默病进展阶段（二分类/多分类）。旨在提升诊断精度并增强可解释性，通过多数据集验证，其准确率超97%，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.13906v1](http://arxiv.org/pdf/2505.13906v1)**

> **作者:** Soyabul Islam Lincoln; Mirza Mohd Shahriar Maswood
>
> **备注:** 20 pages, 12 figures,
>
> **摘要:** A common neurodegenerative disease, Alzheimer's disease requires a precise diagnosis and efficient treatment, particularly in light of escalating healthcare expenses and the expanding use of artificial intelligence in medical diagnostics. Many recent studies shows that the combination of brain Magnetic Resonance Imaging (MRI) and deep neural networks have achieved promising results for diagnosing AD. Using deep convolutional neural networks, this paper introduces a novel deep learning architecture that incorporates multiresidual blocks, specialized spatial attention blocks, grouped query attention, and multi-head attention. The study assessed the model's performance on four publicly accessible datasets and concentrated on identifying binary and multiclass issues across various categories. This paper also takes into account of the explainability of AD's progression and compared with state-of-the-art methods namely Gradient Class Activation Mapping (GradCAM), Score-CAM, Faster Score-CAM, and XGRADCAM. Our methodology consistently outperforms current approaches, achieving 99.66\% accuracy in 4-class classification, 99.63\% in 3-class classification, and 100\% in binary classification using Kaggle datasets. For Open Access Series of Imaging Studies (OASIS) datasets the accuracies are 99.92\%, 99.90\%, and 99.95\% respectively. The Alzheimer's Disease Neuroimaging Initiative-1 (ADNI-1) dataset was used for experiments in three planes (axial, sagittal, and coronal) and a combination of all planes. The study achieved accuracies of 99.08\% for axis, 99.85\% for sagittal, 99.5\% for coronal, and 99.17\% for all axis, and 97.79\% and 8.60\% respectively for ADNI-2. The network's ability to retrieve important information from MRI images is demonstrated by its excellent accuracy in categorizing AD stages.
>
---
#### [new 121] GANCompress: GAN-Enhanced Neural Image Compression with Binary Spherical Quantization
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出GANCompress，一种结合二进制球面量化（BSQ）和GAN的神经图像压缩框架。旨在解决高压缩比下保持感知质量、计算效率及内容适应性问题。通过Transformer自编码器与BSQ离散化，结合频域注意力和颜色一致性GAN优化，在ImageNet等数据集上实现100倍压缩，PSNR/SSIM与H.264相当且速度更快，FID降低43%，达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.13542v1](http://arxiv.org/pdf/2505.13542v1)**

> **作者:** Karthik Sivakoti
>
> **摘要:** The exponential growth of visual data in digital communications has intensified the need for efficient compression techniques that balance rate-distortion performance with computational feasibility. While recent neural compression approaches have shown promise, they still struggle with fundamental challenges: preserving perceptual quality at high compression ratios, computational efficiency, and adaptability to diverse visual content. This paper introduces GANCompress, a novel neural compression framework that synergistically combines Binary Spherical Quantization (BSQ) with Generative Adversarial Networks (GANs) to address these challenges. Our approach employs a transformer-based autoencoder with an enhanced BSQ bottleneck that projects latent representations onto a hypersphere, enabling efficient discretization with bounded quantization error. This is followed by a specialized GAN architecture incorporating frequency-domain attention and color consistency optimization. Experimental results demonstrate that GANCompress achieves substantial improvement in compression efficiency -- reducing file sizes by up to 100x with minimal visual distortion. Our method outperforms traditional codecs like H.264 by 12-15% in perceptual metrics while maintaining comparable PSNR/SSIM values, with 2.4x faster encoding and decoding speeds. On standard benchmarks including ImageNet-1k and COCO2017, GANCompress sets a new state-of-the-art, reducing FID from 0.72 to 0.41 (43% improvement) compared to previous methods while maintaining higher throughput. This work presents a significant advancement in neural compression technology with promising applications for real-time visual communication systems.
>
---
#### [new 122] Toward Effective Reinforcement Learning Fine-Tuning for Medical VQA in Vision-Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文聚焦医疗视觉问答任务，研究强化学习微调在医学多模态模型中的应用挑战。针对直接应用RL效果不佳的问题，探讨了基础模型初始化、医学语义对齐、奖励机制及偏差四大维度的影响，通过大量实验验证GRPO方法在准确性和推理质量上优于传统监督微调。**

- **链接: [http://arxiv.org/pdf/2505.13973v1](http://arxiv.org/pdf/2505.13973v1)**

> **作者:** Wenhui Zhu; Xuanzhao Dong; Xin Li; Peijie Qiu; Xiwen Chen; Abolfazl Razi; Aris Sotiras; Yi Su; Yalin Wang
>
> **摘要:** Recently, reinforcement learning (RL)-based tuning has shifted the trajectory of Multimodal Large Language Models (MLLMs), particularly following the introduction of Group Relative Policy Optimization (GRPO). However, directly applying it to medical tasks remains challenging for achieving clinically grounded model behavior. Motivated by the need to align model response with clinical expectations, we investigate four critical dimensions that affect the effectiveness of RL-based tuning in medical visual question answering (VQA): base model initialization strategy, the role of medical semantic alignment, the impact of length-based rewards on long-chain reasoning, and the influence of bias. We conduct extensive experiments to analyze these factors for medical MLLMs, providing new insights into how models are domain-specifically fine-tuned. Additionally, our results also demonstrate that GRPO-based RL tuning consistently outperforms standard supervised fine-tuning (SFT) in both accuracy and reasoning quality.
>
---
#### [new 123] Improving Compositional Generation with Diffusion Models Using Lift Scores
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于扩散模型的组合生成任务，旨在解决多条件生成中条件对齐不足的问题。提出利用lift scores评估样本对单个条件的符合度并组合判断，无需额外训练，开发了高效优化版本，实验证明其提升多条件生成的对齐效果。**

- **链接: [http://arxiv.org/pdf/2505.13740v1](http://arxiv.org/pdf/2505.13740v1)**

> **作者:** Chenning Yu; Sicun Gao
>
> **摘要:** We introduce a novel resampling criterion using lift scores, for improving compositional generation in diffusion models. By leveraging the lift scores, we evaluate whether generated samples align with each single condition and then compose the results to determine whether the composed prompt is satisfied. Our key insight is that lift scores can be efficiently approximated using only the original diffusion model, requiring no additional training or external modules. We develop an optimized variant that achieves relatively lower computational overhead during inference while maintaining effectiveness. Through extensive experiments, we demonstrate that lift scores significantly improved the condition alignment for compositional generation across 2D synthetic data, CLEVR position tasks, and text-to-image synthesis. Our code is available at http://github.com/rainorangelemon/complift.
>
---
#### [new 124] Scaling and Enhancing LLM-based AVSR: A Sparse Mixture of Projectors Approach
- **分类: eess.AS; cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于视听语音识别（AVSR）任务，旨在解决基于大语言模型（LLM）的AVSR计算成本过高问题。提出Llama-SMoP方法，通过稀疏投影器混合（SMoP）模块扩展模型容量而不增加推理成本，采用模态专用路由和专家配置（DEDR），提升多模态LLM的效率与性能，消融实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.14336v1](http://arxiv.org/pdf/2505.14336v1)**

> **作者:** Umberto Cappellazzo; Minsu Kim; Stavros Petridis; Daniele Falavigna; Alessio Brutti
>
> **摘要:** Audio-Visual Speech Recognition (AVSR) enhances robustness in noisy environments by integrating visual cues. While recent advances integrate Large Language Models (LLMs) into AVSR, their high computational cost hinders deployment in resource-constrained settings. To address this, we propose Llama-SMoP, an efficient Multimodal LLM that employs a Sparse Mixture of Projectors (SMoP) module to scale model capacity without increasing inference costs. By incorporating sparsely-gated mixture-of-experts (MoE) projectors, Llama-SMoP enables the use of smaller LLMs while maintaining strong performance. We explore three SMoP configurations and show that Llama-SMoP DEDR (Disjoint-Experts, Disjoint-Routers), which uses modality-specific routers and experts, achieves superior performance on ASR, VSR, and AVSR tasks. Ablation studies confirm its effectiveness in expert activation, scalability, and noise robustness.
>
---
#### [new 125] Open Set Domain Adaptation with Vision-language models via Gradient-aware Separation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于开放集合域适应（OSDA）任务，旨在解决跨领域对齐已知类别分布并识别目标域未知类别的问题。提出基于CLIP的两阶段方法：1）通过动态文本提示调整CLIP的文本编码器，利用语义关系对齐跨域分布；2）利用梯度L2范数差异分离已知/未知样本，缓解误差积累。实验显示优于基线方法。**

- **链接: [http://arxiv.org/pdf/2505.13507v1](http://arxiv.org/pdf/2505.13507v1)**

> **作者:** Haoyang Chen
>
> **摘要:** Open-Set Domain Adaptation (OSDA) confronts the dual challenge of aligning known-class distributions across domains while identifying target-domain-specific unknown categories. Current approaches often fail to leverage semantic relationships between modalities and struggle with error accumulation in unknown sample detection. We propose to harness Contrastive Language-Image Pretraining (CLIP) to address these limitations through two key innovations: 1) Prompt-driven cross-domain alignment: Learnable textual prompts conditioned on domain discrepancy metrics dynamically adapt CLIP's text encoder, enabling semantic consistency between source and target domains without explicit unknown-class supervision. 2) Gradient-aware open-set separation: A gradient analysis module quantifies domain shift by comparing the L2-norm of gradients from the learned prompts, where known/unknown samples exhibit statistically distinct gradient behaviors. Evaluations on Office-Home show that our method consistently outperforms CLIP baseline and standard baseline. Ablation studies confirm the gradient norm's critical role.
>
---
## 更新

#### [replaced 001] ReVLA: Reverting Visual Domain Limitation of Robotic Foundation Models
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2409.15250v3](http://arxiv.org/pdf/2409.15250v3)**

> **作者:** Sombit Dey; Jan-Nico Zaech; Nikolay Nikolov; Luc Van Gool; Danda Pani Paudel
>
> **备注:** Accepted at ICRA-2025, Atlanta
>
> **摘要:** Recent progress in large language models and access to large-scale robotic datasets has sparked a paradigm shift in robotics models transforming them into generalists able to adapt to various tasks, scenes, and robot modalities. A large step for the community are open Vision Language Action models which showcase strong performance in a wide variety of tasks. In this work, we study the visual generalization capabilities of three existing robotic foundation models, and propose a corresponding evaluation framework. Our study shows that the existing models do not exhibit robustness to visual out-of-domain scenarios. This is potentially caused by limited variations in the training data and/or catastrophic forgetting, leading to domain limitations in the vision foundation models. We further explore OpenVLA, which uses two pre-trained vision foundation models and is, therefore, expected to generalize to out-of-domain experiments. However, we showcase catastrophic forgetting by DINO-v2 in OpenVLA through its failure to fulfill the task of depth regression. To overcome the aforementioned issue of visual catastrophic forgetting, we propose a gradual backbone reversal approach founded on model merging. This enables OpenVLA -- which requires the adaptation of the visual backbones during initial training -- to regain its visual generalization ability. Regaining this capability enables our ReVLA model to improve over OpenVLA by a factor of 77\% and 66\% for grasping and lifting in visual OOD tasks. Comprehensive evaluations, episode rollouts and model weights are available on the ReVLA Page
>
---
#### [replaced 002] Generalized Few-shot 3D Point Cloud Segmentation with Vision-Language Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.16282v2](http://arxiv.org/pdf/2503.16282v2)**

> **作者:** Zhaochong An; Guolei Sun; Yun Liu; Runjia Li; Junlin Han; Ender Konukoglu; Serge Belongie
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Generalized few-shot 3D point cloud segmentation (GFS-PCS) adapts models to new classes with few support samples while retaining base class segmentation. Existing GFS-PCS methods enhance prototypes via interacting with support or query features but remain limited by sparse knowledge from few-shot samples. Meanwhile, 3D vision-language models (3D VLMs), generalizing across open-world novel classes, contain rich but noisy novel class knowledge. In this work, we introduce a GFS-PCS framework that synergizes dense but noisy pseudo-labels from 3D VLMs with precise yet sparse few-shot samples to maximize the strengths of both, named GFS-VL. Specifically, we present a prototype-guided pseudo-label selection to filter low-quality regions, followed by an adaptive infilling strategy that combines knowledge from pseudo-label contexts and few-shot samples to adaptively label the filtered, unlabeled areas. Additionally, we design a novel-base mix strategy to embed few-shot samples into training scenes, preserving essential context for improved novel class learning. Moreover, recognizing the limited diversity in current GFS-PCS benchmarks, we introduce two challenging benchmarks with diverse novel classes for comprehensive generalization evaluation. Experiments validate the effectiveness of our framework across models and datasets. Our approach and benchmarks provide a solid foundation for advancing GFS-PCS in the real world. The code is at https://github.com/ZhaochongAn/GFS-VL
>
---
#### [replaced 003] Explaining Uncertainty in Multiple Sclerosis Lesion Segmentation Beyond Prediction Errors
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.04814v2](http://arxiv.org/pdf/2504.04814v2)**

> **作者:** Nataliia Molchanova; Pedro M. Gordaliza; Alessandro Cagol; Mario Ocampo--Pineda; Po--Jui Lu; Matthias Weigel; Xinjie Chen; Erin S. Beck; Haris Tsagkas; Daniel Reich; Anna Stölting; Pietro Maggi; Delphine Ribes; Adrien Depeursinge; Cristina Granziera; Henning Müller; Meritxell Bach Cuadra
>
> **摘要:** Trustworthy artificial intelligence (AI) is essential in healthcare, particularly for high-stakes tasks like medical image segmentation. Explainable AI and uncertainty quantification significantly enhance AI reliability by addressing key attributes such as robustness, usability, and explainability. Despite extensive technical advances in uncertainty quantification for medical imaging, understanding the clinical informativeness and interpretability of uncertainty remains limited. This study introduces a novel framework to explain the potential sources of predictive uncertainty, specifically in cortical lesion segmentation in multiple sclerosis using deep ensembles. The proposed analysis shifts the focus from the uncertainty-error relationship towards relevant medical and engineering factors. Our findings reveal that instance-wise uncertainty is strongly related to lesion size, shape, and cortical involvement. Expert rater feedback confirms that similar factors impede annotator confidence. Evaluations conducted on two datasets (206 patients, almost 2000 lesions) under both in-domain and distribution-shift conditions highlight the utility of the framework in different scenarios.
>
---
#### [replaced 004] DiffDesign: Controllable Diffusion with Meta Prior for Efficient Interior Design Generation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.16301v2](http://arxiv.org/pdf/2411.16301v2)**

> **作者:** Yuxuan Yang; Tao Geng; Jingyao Wang; Changwen Zheng; Fuchun Sun
>
> **备注:** 32 pages
>
> **摘要:** Interior design is a complex and creative discipline involving aesthetics, functionality, ergonomics, and materials science. Effective solutions must meet diverse requirements, typically producing multiple deliverables such as renderings and design drawings from various perspectives. Consequently, interior design processes are often inefficient and demand significant creativity. With advances in machine learning, generative models have emerged as a promising means of improving efficiency by creating designs from text descriptions or sketches. However, few generative works focus on interior design, leading to substantial discrepancies between outputs and practical needs, such as differences in size, spatial scope, and the lack of controllable generation quality. To address these challenges, we propose DiffDesign, a controllable diffusion model with meta priors for efficient interior design generation. Specifically, we utilize the generative priors of a 2D diffusion model pre-trained on a large image dataset as our rendering backbone. We further guide the denoising process by disentangling cross-attention control over design attributes, such as appearance, pose, and size, and introduce an optimal transfer-based alignment module to enforce view consistency. Simultaneously, we construct an interior design-specific dataset, DesignHelper, consisting of over 400 solutions across more than 15 spatial types and 15 design styles. This dataset helps fine-tune DiffDesign. Extensive experiments conducted on various benchmark datasets demonstrate the effectiveness and robustness of DiffDesign.
>
---
#### [replaced 005] Learning Coherent Matrixized Representation in Latent Space for Volumetric 4D Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.13238v2](http://arxiv.org/pdf/2403.13238v2)**

> **作者:** Qitong Yang; Mingtao Feng; Zijie Wu; Shijie Sun; Weisheng Dong; Yaonan Wang; Ajmal Mian
>
> **摘要:** Directly learning to model 4D content, including shape, color, and motion, is challenging. Existing methods rely on pose priors for motion control, resulting in limited motion diversity and continuity in details. To address this, we propose a framework that generates volumetric 4D sequences, where 3D shapes are animated under given conditions (text-image guidance) with dynamic evolution in shape and color across spatial and temporal dimensions, allowing for free navigation and rendering from any direction. We first use a coherent 3D shape and color modeling to encode the shape and color of each detailed 3D geometry frame into a latent space. Then we propose a matrixized 4D sequence representation allowing efficient diffusion model operation. Finally, we introduce spatio-temporal diffusion for 4D volumetric generation under given images and text prompts. Extensive experiments on the ShapeNet, 3DBiCar, DeformingThings4D and Objaverse datasets for several tasks demonstrate that our method effectively learns to generate high quality 3D shapes with consistent color and coherent mesh animations, improving over the current methods. Our code will be publicly available.
>
---
#### [replaced 006] Diffusion based Semantic Outlier Generation via Nuisance Awareness for Out-of-Distribution Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2408.14841v2](http://arxiv.org/pdf/2408.14841v2)**

> **作者:** Suhee Yoon; Sanghyu Yoon; Ye Seul Sim; Sungik Choi; Kyungeun Lee; Hye-Seung Cho; Hankook Lee; Woohyung Lim
>
> **摘要:** Out-of-distribution (OOD) detection, which determines whether a given sample is part of the in-distribution (ID), has recently shown promising results through training with synthetic OOD datasets. Nonetheless, existing methods often produce outliers that are considerably distant from the ID, showing limited efficacy for capturing subtle distinctions between ID and OOD. To address these issues, we propose a novel framework, Semantic Outlier generation via Nuisance Awareness (SONA), which notably produces challenging outliers by directly leveraging pixel-space ID samples through diffusion models. Our approach incorporates SONA guidance, providing separate control over semantic and nuisance regions of ID samples. Thereby, the generated outliers achieve two crucial properties: (i) they present explicit semantic-discrepant information, while (ii) maintaining various levels of nuisance resemblance with ID. Furthermore, the improved OOD detector training with SONA outliers facilitates learning with a focus on semantic distinctions. Extensive experiments demonstrate the effectiveness of our framework, achieving an impressive AUROC of 88% on near-OOD datasets, which surpasses the performance of baseline methods by a significant margin of approximately 6%.
>
---
#### [replaced 007] SpaceJAM: a Lightweight and Regularization-free Method for Fast Joint Alignment of Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.11850v2](http://arxiv.org/pdf/2407.11850v2)**

> **作者:** Nir Barel; Ron Shapira Weber; Nir Mualem; Shahaf E. Finder; Oren Freifeld
>
> **备注:** Accepted to ECCV 2024
>
> **摘要:** The unsupervised task of Joint Alignment (JA) of images is beset by challenges such as high complexity, geometric distortions, and convergence to poor local or even global optima. Although Vision Transformers (ViT) have recently provided valuable features for JA, they fall short of fully addressing these issues. Consequently, researchers frequently depend on expensive models and numerous regularization terms, resulting in long training times and challenging hyperparameter tuning. We introduce the Spatial Joint Alignment Model (SpaceJAM), a novel approach that addresses the JA task with efficiency and simplicity. SpaceJAM leverages a compact architecture with only 16K trainable parameters and uniquely operates without the need for regularization or atlas maintenance. Evaluations on SPair-71K and CUB datasets demonstrate that SpaceJAM matches the alignment capabilities of existing methods while significantly reducing computational demands and achieving at least a 10x speedup. SpaceJAM sets a new standard for rapid and effective image alignment, making the process more accessible and efficient. Our code is available at: https://bgu-cs-vil.github.io/SpaceJAM/.
>
---
#### [replaced 008] View-Invariant Pixelwise Anomaly Detection in Multi-object Scenes with Adaptive View Synthesis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.18012v3](http://arxiv.org/pdf/2406.18012v3)**

> **作者:** Subin Varghese; Vedhus Hoskere
>
> **摘要:** The built environment, encompassing critical infrastructure such as bridges and buildings, requires diligent monitoring of unexpected anomalies or deviations from a normal state in captured imagery. Anomaly detection methods could aid in automating this task; however, deploying anomaly detection effectively in such environments presents significant challenges that have not been evaluated before. These challenges include camera viewpoints that vary, the presence of multiple objects within a scene, and the absence of labeled anomaly data for training. To address these comprehensively, we introduce and formalize Scene Anomaly Detection (Scene AD) as the task of unsupervised, pixel-wise anomaly localization under these specific real-world conditions. Evaluating progress in Scene AD required the development of ToyCity, the first multi-object, multi-view real-image dataset, for unsupervised anomaly detection. Our initial evaluations using ToyCity revealed that established anomaly detection baselines struggle to achieve robust pixel-level localization. To address this, two data augmentation strategies were created to generate additional synthetic images of non-anomalous regions to enhance generalizability. However, the addition of these synthetic images alone only provided minor improvements. Thus, OmniAD, a refinement of the Reverse Distillation methodology, was created to establish a stronger baseline. Our experiments demonstrate that OmniAD, when used with augmented views, yields a 64.33\% increase in pixel-wise \(F_1\) score over Reverse Distillation with no augmentation. Collectively, this work offers the Scene AD task definition, the ToyCity benchmark, the view synthesis augmentation approaches, and the OmniAD method. Project Page: https://drags99.github.io/OmniAD/
>
---
#### [replaced 009] KIND: Knowledge Integration and Diversion for Training Decomposable Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.07337v2](http://arxiv.org/pdf/2408.07337v2)**

> **作者:** Yucheng Xie; Fu Feng; Ruixiao Shi; Jing Wang; Yong Rui; Xin Geng
>
> **摘要:** Pre-trained models have become the preferred backbone due to the increasing complexity of model parameters. However, traditional pre-trained models often face deployment challenges due to their fixed sizes, and are prone to negative transfer when discrepancies arise between training tasks and target tasks. To address this, we propose KIND, a novel pre-training method designed to construct decomposable models. KIND integrates knowledge by incorporating Singular Value Decomposition (SVD) as a structural constraint, with each basic component represented as a combination of a column vector, singular value, and row vector from U, \Sigma, and V^\top matrices. These components are categorized into learngenes for encapsulating class-agnostic knowledge and tailors for capturing class-specific knowledge, with knowledge diversion facilitated by a class gate mechanism during training. Extensive experiments demonstrate that models pre-trained with KIND can be decomposed into learngenes and tailors, which can be adaptively recombined for diverse resource-constrained deployments. Moreover, for tasks with large domain shifts, transferring only learngenes with task-agnostic knowledge, when combined with randomly initialized tailors, effectively mitigates domain shifts. Code will be made available at https://github.com/Te4P0t/KIND.
>
---
#### [replaced 010] Towards Rich Emotions in 3D Avatars: A Text-to-3D Avatar Generation Benchmark
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.02508v2](http://arxiv.org/pdf/2412.02508v2)**

> **作者:** Haidong Xu; Meishan Zhang; Hao Ju; Zhedong Zheng; Erik Cambria; Min Zhang; Hao Fei
>
> **备注:** 19 pages. Project website: https://github.com/WalkerMitty/EmoAva
>
> **摘要:** Producing emotionally dynamic 3D facial avatars with text derived from spoken words (Emo3D) has been a pivotal research topic in 3D avatar generation. While progress has been made in general-purpose 3D avatar generation, the exploration of generating emotional 3D avatars remains scarce, primarily due to the complexities of identifying and rendering rich emotions from spoken words. This paper reexamines Emo3D generation and draws inspiration from human processes, breaking down Emo3D into two cascading steps: Text-to-3D Expression Mapping (T3DEM) and 3D Avatar Rendering (3DAR). T3DEM is the most crucial step in determining the quality of Emo3D generation and encompasses three key challenges: Expression Diversity, Emotion-Content Consistency, and Expression Fluidity. To address these challenges, we introduce a novel benchmark to advance research in Emo3D generation. First, we present EmoAva, a large-scale, high-quality dataset for T3DEM, comprising 15,000 text-to-3D expression mappings that characterize the aforementioned three challenges in Emo3D generation. Furthermore, we develop various metrics to effectively evaluate models against these identified challenges. Next, to effectively model the consistency, diversity, and fluidity of human expressions in the T3DEM step, we propose the Continuous Text-to-Expression Generator, which employs an autoregressive Conditional Variational Autoencoder for expression code generation, enhanced with Latent Temporal Attention and Expression-wise Attention mechanisms. Finally, to further enhance the 3DAR step on rendering higher-quality subtle expressions, we present the Globally-informed Gaussian Avatar (GiGA) model. GiGA incorporates a global information mechanism into 3D Gaussian representations, enabling the capture of subtle micro-expressions and seamless transitions between emotional states.
>
---
#### [replaced 011] CompBench: Benchmarking Complex Instruction-guided Image Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12200v2](http://arxiv.org/pdf/2505.12200v2)**

> **作者:** Bohan Jia; Wenxuan Huang; Yuntian Tang; Junbo Qiao; Jincheng Liao; Shaosheng Cao; Fei Zhao; Zhaopeng Feng; Zhouhong Gu; Zhenfei Yin; Lei Bai; Wanli Ouyang; Lin Chen; Fei Zhao; Zihan Wang; Yuan Xie; Shaohui Lin
>
> **摘要:** While real-world applications increasingly demand intricate scene manipulation, existing instruction-guided image editing benchmarks often oversimplify task complexity and lack comprehensive, fine-grained instructions. To bridge this gap, we introduce, a large-scale benchmark specifically designed for complex instruction-guided image editing. CompBench features challenging editing scenarios that incorporate fine-grained instruction following, spatial and contextual reasoning, thereby enabling comprehensive evaluation of image editing models' precise manipulation capabilities. To construct CompBench, We propose an MLLM-human collaborative framework with tailored task pipelines. Furthermore, we propose an instruction decoupling strategy that disentangles editing intents into four key dimensions: location, appearance, dynamics, and objects, ensuring closer alignment between instructions and complex editing requirements. Extensive evaluations reveal that CompBench exposes fundamental limitations of current image editing models and provides critical insights for the development of next-generation instruction-guided image editing systems. The dataset, code, and models are available in https://comp-bench.github.io/.
>
---
#### [replaced 012] VisBias: Measuring Explicit and Implicit Social Biases in Vision Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.07575v2](http://arxiv.org/pdf/2503.07575v2)**

> **作者:** Jen-tse Huang; Jiantong Qin; Jianping Zhang; Youliang Yuan; Wenxuan Wang; Jieyu Zhao
>
> **备注:** 8 pages of main text; 9 pages of appendix
>
> **摘要:** This research investigates both explicit and implicit social biases exhibited by Vision-Language Models (VLMs). The key distinction between these bias types lies in the level of awareness: explicit bias refers to conscious, intentional biases, while implicit bias operates subconsciously. To analyze explicit bias, we directly pose questions to VLMs related to gender and racial differences: (1) Multiple-choice questions based on a given image (e.g., "What is the education level of the person in the image?") (2) Yes-No comparisons using two images (e.g., "Is the person in the first image more educated than the person in the second image?") For implicit bias, we design tasks where VLMs assist users but reveal biases through their responses: (1) Image description tasks: Models are asked to describe individuals in images, and we analyze disparities in textual cues across demographic groups. (2) Form completion tasks: Models draft a personal information collection form with 20 attributes, and we examine correlations among selected attributes for potential biases. We evaluate Gemini-1.5, GPT-4V, GPT-4o, LLaMA-3.2-Vision and LLaVA-v1.6. Our code and data are publicly available at https://github.com/uscnlp-lime/VisBias.
>
---
#### [replaced 013] Multimodal Cancer Survival Analysis via Hypergraph Learning with Cross-Modality Rebalance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11997v2](http://arxiv.org/pdf/2505.11997v2)**

> **作者:** Mingcheng Qu; Guang Yang; Donglin Di; Tonghua Su; Yue Gao; Yang Song; Lei Fan
>
> **备注:** accepted by IJCAI2025 Code: https://github.com/MCPathology/MRePath
>
> **摘要:** Multimodal pathology-genomic analysis has become increasingly prominent in cancer survival prediction. However, existing studies mainly utilize multi-instance learning to aggregate patch-level features, neglecting the information loss of contextual and hierarchical details within pathology images. Furthermore, the disparity in data granularity and dimensionality between pathology and genomics leads to a significant modality imbalance. The high spatial resolution inherent in pathology data renders it a dominant role while overshadowing genomics in multimodal integration. In this paper, we propose a multimodal survival prediction framework that incorporates hypergraph learning to effectively capture both contextual and hierarchical details from pathology images. Moreover, it employs a modality rebalance mechanism and an interactive alignment fusion strategy to dynamically reweight the contributions of the two modalities, thereby mitigating the pathology-genomics imbalance. Quantitative and qualitative experiments are conducted on five TCGA datasets, demonstrating that our model outperforms advanced methods by over 3.4\% in C-Index performance.
>
---
#### [replaced 014] Spectral-Spatial Self-Supervised Learning for Few-Shot Hyperspectral Image Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12482v2](http://arxiv.org/pdf/2505.12482v2)**

> **作者:** Wenchen Chen; Yanmei Zhang; Zhongwei Xiao; Jianping Chu; Xingbo Wang
>
> **备注:** https://github.com/Wenchen-Chen/S4L-FSC
>
> **摘要:** Few-shot classification of hyperspectral images (HSI) faces the challenge of scarce labeled samples. Self-Supervised learning (SSL) and Few-Shot Learning (FSL) offer promising avenues to address this issue. However, existing methods often struggle to adapt to the spatial geometric diversity of HSIs and lack sufficient spectral prior knowledge. To tackle these challenges, we propose a method, Spectral-Spatial Self-Supervised Learning for Few-Shot Hyperspectral Image Classification (S4L-FSC), aimed at improving the performance of few-shot HSI classification. Specifically, we first leverage heterogeneous datasets to pretrain a spatial feature extractor using a designed Rotation-Mirror Self-Supervised Learning (RM-SSL) method, combined with FSL. This approach enables the model to learn the spatial geometric diversity of HSIs using rotation and mirroring labels as supervisory signals, while acquiring transferable spatial meta-knowledge through few-shot learning. Subsequently, homogeneous datasets are utilized to pretrain a spectral feature extractor via a combination of FSL and Masked Reconstruction Self-Supervised Learning (MR-SSL). The model learns to reconstruct original spectral information from randomly masked spectral vectors, inferring spectral dependencies. In parallel, FSL guides the model to extract pixel-level discriminative features, thereby embedding rich spectral priors into the model. This spectral-spatial pretraining method, along with the integration of knowledge from heterogeneous and homogeneous sources, significantly enhances model performance. Extensive experiments on four HSI datasets demonstrate the effectiveness and superiority of the proposed S4L-FSC approach for few-shot HSI classification.
>
---
#### [replaced 015] Differentially Private Synthetic Data via APIs 3: Using Simulators Instead of Foundation Model
- **分类: cs.LG; cs.CR; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2502.05505v3](http://arxiv.org/pdf/2502.05505v3)**

> **作者:** Zinan Lin; Tadas Baltrusaitis; Wenyu Wang; Sergey Yekhanin
>
> **备注:** Published in: (1) ICLR 2025 Workshop on Data Problems, (2) ICLR 2025 Workshop on Synthetic Data
>
> **摘要:** Differentially private (DP) synthetic data, which closely resembles the original private data while maintaining strong privacy guarantees, has become a key tool for unlocking the value of private data without compromising privacy. Recently, Private Evolution (PE) has emerged as a promising method for generating DP synthetic data. Unlike other training-based approaches, PE only requires access to inference APIs from foundation models, enabling it to harness the power of state-of-the-art (SoTA) models. However, a suitable foundation model for a specific private data domain is not always available. In this paper, we discover that the PE framework is sufficiently general to allow APIs beyond foundation models. In particular, we demonstrate that many SoTA data synthesizers that do not rely on neural networks--such as computer graphics-based image generators, which we refer to as simulators--can be effectively integrated into PE. This insight significantly broadens PE's applicability and unlocks the potential of powerful simulators for DP data synthesis. We explore this approach, named Sim-PE, in the context of image synthesis. Across four diverse simulators, Sim-PE performs well, improving the downstream classification accuracy of PE by up to 3x, reducing FID by up to 80%, and offering much greater efficiency. We also show that simulators and foundation models can be easily leveraged together within PE to achieve further improvements. The code is open-sourced in the Private Evolution Python library: https://github.com/microsoft/DPSDA.
>
---
#### [replaced 016] Contrastive Alignment with Semantic Gap-Aware Corrections in Text-Video Retrieval
- **分类: cs.CV; cs.IR; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.12499v2](http://arxiv.org/pdf/2505.12499v2)**

> **作者:** Jian Xiao; Zijie Song; Jialong Hu; Hao Cheng; Zhenzhen Hu; Jia Li; Richang Hong
>
> **摘要:** Recent advances in text-video retrieval have been largely driven by contrastive learning frameworks. However, existing methods overlook a key source of optimization tension: the separation between text and video distributions in the representation space (referred to as the modality gap), and the prevalence of false negatives in batch sampling. These factors lead to conflicting gradients under the InfoNCE loss, impeding stable alignment. To mitigate this, we propose GARE, a Gap-Aware Retrieval framework that introduces a learnable, pair-specific increment Delta_ij between text t_i and video v_j to offload the tension from the global anchor representation. We first derive the ideal form of Delta_ij via a coupled multivariate first-order Taylor approximation of the InfoNCE loss under a trust-region constraint, revealing it as a mechanism for resolving gradient conflicts by guiding updates along a locally optimal descent direction. Due to the high cost of directly computing Delta_ij, we introduce a lightweight neural module conditioned on the semantic gap between each video-text pair, enabling structure-aware correction guided by gradient supervision. To further stabilize learning and promote interpretability, we regularize Delta using three components: a trust-region constraint to prevent oscillation, a directional diversity term to promote semantic coverage, and an information bottleneck to limit redundancy. Experiments across four retrieval benchmarks show that GARE consistently improves alignment accuracy and robustness to noisy supervision, confirming the effectiveness of gap-aware tension mitigation.
>
---
#### [replaced 017] Industrial Synthetic Segment Pre-training
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13099v2](http://arxiv.org/pdf/2505.13099v2)**

> **作者:** Shinichi Mae; Ryousuke Yamada; Hirokatsu Kataoka
>
> **摘要:** Pre-training on real-image datasets has been widely proven effective for improving instance segmentation. However, industrial applications face two key challenges: (1) legal and ethical restrictions, such as ImageNet's prohibition of commercial use, and (2) limited transferability due to the domain gap between web images and industrial imagery. Even recent vision foundation models, including the segment anything model (SAM), show notable performance degradation in industrial settings. These challenges raise critical questions: Can we build a vision foundation model for industrial applications without relying on real images or manual annotations? And can such models outperform even fine-tuned SAM on industrial datasets? To address these questions, we propose the Instance Core Segmentation Dataset (InsCore), a synthetic pre-training dataset based on formula-driven supervised learning (FDSL). InsCore generates fully annotated instance segmentation images that reflect key characteristics of industrial data, including complex occlusions, dense hierarchical masks, and diverse non-rigid shapes, distinct from typical web imagery. Unlike previous methods, InsCore requires neither real images nor human annotations. Experiments on five industrial datasets show that models pre-trained with InsCore outperform those trained on COCO and ImageNet-21k, as well as fine-tuned SAM, achieving an average improvement of 6.2 points in instance segmentation performance. This result is achieved using only 100k synthetic images, more than 100 times fewer than the 11 million images in SAM's SA-1B dataset, demonstrating the data efficiency of our approach. These findings position InsCore as a practical and license-free vision foundation model for industrial applications.
>
---
#### [replaced 018] Point2Primitive: CAD Reconstruction from Point Cloud by Direct Primitive Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02043v2](http://arxiv.org/pdf/2505.02043v2)**

> **作者:** Cheng Wang; Xinzhu Ma; Bin Wang; Shixiang Tang; Yuan Meng; Ping Jiang
>
> **摘要:** Recovering CAD models from point clouds, especially the sketch-extrusion process, can be seen as the process of rebuilding the topology and extrusion primitives. Previous methods utilize implicit fields for sketch representation, leading to shape reconstruction of curved edges. In this paper, we proposed a CAD reconstruction network that produces editable CAD models from input point clouds (Point2Primitive) by directly predicting every element of the extrusion primitives. Point2Primitive can directly detect and predict sketch curves (type and parameter) from point clouds based on an improved transformer. The sketch curve parameters are formulated as position queries and optimized in an autoregressive way, leading to high parameter accuracy. The topology is rebuilt by extrusion segmentation, and each extrusion parameter (sketch and extrusion operation) is recovered by combining the predicted curves and the computed extrusion operation. Extensive experiments demonstrate that our method is superior in primitive prediction accuracy and CAD reconstruction. The reconstructed shapes are of high geometrical fidelity.
>
---
#### [replaced 019] Customized SAM 2 for Referring Remote Sensing Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07266v2](http://arxiv.org/pdf/2503.07266v2)**

> **作者:** Fu Rong; Meng Lan; Qian Zhang; Lefei Zhang
>
> **摘要:** Referring Remote Sensing Image Segmentation (RRSIS) aims to segment target objects in remote sensing (RS) images based on textual descriptions. Although Segment Anything Model 2 (SAM 2) has shown remarkable performance in various segmentation tasks, its application to RRSIS presents several challenges, including understanding the text-described RS scenes and generating effective prompts from text descriptions. To address these issues, we propose RS2-SAM 2, a novel framework that adapts SAM 2 to RRSIS by aligning the adapted RS features and textual features, providing pseudo-mask-based dense prompts, and enforcing boundary constraints. Specifically, we first employ a union encoder to jointly encode the visual and textual inputs, generating aligned visual and text embeddings as well as multimodal class tokens. Then, we design a bidirectional hierarchical fusion module to adapt SAM 2 to RS scenes and align adapted visual features with the visually enhanced text embeddings, improving the model's interpretation of text-described RS scenes. Additionally, a mask prompt generator is introduced to take the visual embeddings and class tokens as input and produce a pseudo-mask as the dense prompt of SAM 2. To further refine segmentation, we introduce a text-guided boundary loss to optimize segmentation boundaries by computing text-weighted gradient differences. Experimental results on several RRSIS benchmarks demonstrate that RS2-SAM 2 achieves state-of-the-art performance.
>
---
#### [replaced 020] Event-Driven Dynamic Scene Depth Completion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13279v2](http://arxiv.org/pdf/2505.13279v2)**

> **作者:** Zhiqiang Yan; Jianhao Jiao; Zhengxue Wang; Gim Hee Lee
>
> **备注:** 9 pages
>
> **摘要:** Depth completion in dynamic scenes poses significant challenges due to rapid ego-motion and object motion, which can severely degrade the quality of input modalities such as RGB images and LiDAR measurements. Conventional RGB-D sensors often struggle to align precisely and capture reliable depth under such conditions. In contrast, event cameras with their high temporal resolution and sensitivity to motion at the pixel level provide complementary cues that are %particularly beneficial in dynamic environments.To this end, we propose EventDC, the first event-driven depth completion framework. It consists of two key components: Event-Modulated Alignment (EMA) and Local Depth Filtering (LDF). Both modules adaptively learn the two fundamental components of convolution operations: offsets and weights conditioned on motion-sensitive event streams. In the encoder, EMA leverages events to modulate the sampling positions of RGB-D features to achieve pixel redistribution for improved alignment and fusion. In the decoder, LDF refines depth estimations around moving objects by learning motion-aware masks from events. Additionally, EventDC incorporates two loss terms to further benefit global alignment and enhance local depth recovery. Moreover, we establish the first benchmark for event-based depth completion comprising one real-world and two synthetic datasets to facilitate future research. Extensive experiments on this benchmark demonstrate the superiority of our EventDC.
>
---
#### [replaced 021] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.01051v3](http://arxiv.org/pdf/2502.01051v3)**

> **作者:** Tao Zhang; Cheng Da; Kun Ding; Huan Yang; Kun Jin; Yan Li; Tingting Gao; Di Zhang; Shiming Xiang; Chunhong Pan
>
> **备注:** 25 pages, 26 tables, 15 figures
>
> **摘要:** Preference optimization for diffusion models aims to align them with human preferences for images. Previous methods typically use Vision-Language Models (VLMs) as pixel-level reward models to approximate human preferences. However, when used for step-level preference optimization, these models face challenges in handling noisy images of different timesteps and require complex transformations into pixel space. In this work, we show that pre-trained diffusion models are naturally suited for step-level reward modeling in the noisy latent space, as they are explicitly designed to process latent images at various noise levels. Accordingly, we propose the Latent Reward Model (LRM), which repurposes components of the diffusion model to predict preferences of latent images at arbitrary timesteps. Building on LRM, we introduce Latent Preference Optimization (LPO), a step-level preference optimization method conducted directly in the noisy latent space. Experimental results indicate that LPO significantly improves the model's alignment with general, aesthetic, and text-image alignment preferences, while achieving a 2.5-28x training speedup over existing preference optimization methods. Our code and models are available at https://github.com/Kwai-Kolors/LPO.
>
---
#### [replaced 022] Conjuring Positive Pairs for Efficient Unification of Representation Learning and Image Synthesis
- **分类: cs.CV; cs.AI; I.5.4; I.5.1; I.2.10**

- **链接: [http://arxiv.org/pdf/2503.15060v3](http://arxiv.org/pdf/2503.15060v3)**

> **作者:** Imanol G. Estepa; Jesús M. Rodríguez-de-Vera; Ignacio Sarasúa; Bhalaji Nagarajan; Petia Radeva
>
> **备注:** The source code is available in https://github.com/ImaGonEs/Sorcen
>
> **摘要:** While representation learning and generative modeling seek to understand visual data, unifying both domains remains unexplored. Recent Unified Self-Supervised Learning (SSL) methods have started to bridge the gap between both paradigms. However, they rely solely on semantic token reconstruction, which requires an external tokenizer during training -- introducing a significant overhead. In this work, we introduce Sorcen, a novel unified SSL framework, incorporating a synergic Contrastive-Reconstruction objective. Our Contrastive objective, "Echo Contrast", leverages the generative capabilities of Sorcen, eliminating the need for additional image crops or augmentations during training. Sorcen "generates" an echo sample in the semantic token space, forming the contrastive positive pair. Sorcen operates exclusively on precomputed tokens, eliminating the need for an online token transformation during training, thereby significantly reducing computational overhead. Extensive experiments on ImageNet-1k demonstrate that Sorcen outperforms the previous Unified SSL SoTA by 0.4%, 1.48 FID, 1.76%, and 1.53% on linear probing, unconditional image generation, few-shot learning, and transfer learning, respectively, while being 60.8% more efficient. Additionally, Sorcen surpasses previous single-crop MIM SoTA in linear probing and achieves SoTA performance in unconditional image generation, highlighting significant improvements and breakthroughs in Unified SSL models.
>
---
#### [replaced 023] Technical Report: Quantifying and Analyzing the Generalization Power of a DNN
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.06993v2](http://arxiv.org/pdf/2505.06993v2)**

> **作者:** Yuxuan He; Junpeng Zhang; Lei Cheng; Hongyuan Zhang; Quanshi Zhang
>
> **摘要:** This paper proposes a new perspective for analyzing the generalization power of deep neural networks (DNNs), i.e., directly disentangling and analyzing the dynamics of generalizable and non-generalizable interaction encoded by a DNN through the training process. Specifically, this work builds upon the recent theoretical achievement in explainble AI, which proves that the detailed inference logic of DNNs can be can be strictly rewritten as a small number of AND-OR interaction patterns. Based on this, we propose an efficient method to quantify the generalization power of each interaction, and we discover a distinct three-phase dynamics of the generalization power of interactions during training. In particular, the early phase of training typically removes noisy and non-generalizable interactions and learns simple and generalizable ones. The second and the third phases tend to capture increasingly complex interactions that are harder to generalize. Experimental results verify that the learning of non-generalizable interactions is the the direct cause for the gap between the training and testing losses.
>
---
#### [replaced 024] Open3DVQA: A Benchmark for Comprehensive Spatial Reasoning with Multimodal Large Language Model in Open Space
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11094v2](http://arxiv.org/pdf/2503.11094v2)**

> **作者:** Weichen Zhang; Zile Zhou; Zhiheng Zheng; Chen Gao; Jinqiang Cui; Yong Li; Xinlei Chen; Xiao-Ping Zhang
>
> **摘要:** Spatial reasoning is a fundamental capability of embodied agents and has garnered widespread attention in the field of multimodal large language models (MLLMs). In this work, we propose a novel benchmark, Open3DVQA, to comprehensively evaluate the spatial reasoning capacities of current state-of-the-art (SOTA) foundation models in open 3D space. Open3DVQA consists of 9k VQA samples, collected using an efficient semi-automated tool in a high-fidelity urban simulator. We evaluate several SOTA MLLMs across various aspects of spatial reasoning, such as relative and absolute spatial relationships, situational reasoning, and object-centric spatial attributes. Our results reveal that: 1) MLLMs perform better at answering questions regarding relative spatial relationships than absolute spatial relationships, 2) MLLMs demonstrate similar spatial reasoning abilities for both egocentric and allocentric perspectives, and 3) Fine-tuning large models significantly improves their performance across different spatial reasoning tasks. We believe that our open-source data collection tools and in-depth analyses will inspire further research on MLLM spatial reasoning capabilities. The benchmark is available at https://github.com/WeichenZh/Open3DVQA.
>
---
#### [replaced 025] BigReg: An Efficient Registration Pipeline for High-Resolution X-Ray and Light-Sheet Fluorescence Microscopy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.14807v2](http://arxiv.org/pdf/2404.14807v2)**

> **作者:** Siyuan Mei; Fuxin Fan; Mareike Thies; Mingxuan Gu; Fabian Wagner; Oliver Aust; Ina Erceg; Zeynab Mirzaei; Georgiana Neag; Yipeng Sun; Yixing Huang; Andreas Maier
>
> **摘要:** Recently, X-ray microscopy (XRM) and light-sheet fluorescence microscopy (LSFM) have emerged as pivotal tools in preclinical research, particularly for studying bone remodeling diseases such as osteoporosis. These modalities offer micrometer-level resolution, and their integration allows for a complementary examination of bone microstructures which is essential for analyzing functional changes. However, registering high-resolution volumes from these independently scanned modalities poses substantial challenges, especially in real-world and reference-free scenarios. This paper presents BigReg, a fast, two-stage pipeline designed for large-volume registration of XRM and LSFM data. The first stage involves extracting surface features and applying two successive point cloud-based methods for coarse alignment. The subsequent stage refines this alignment using a modified cross-correlation technique, achieving precise volumetric registration. Evaluations using expert-annotated landmarks and augmented test data demonstrate that BigReg approaches the accuracy of landmark-based registration with a landmark distance (LMD) of 8.36\,\textmu m\,$\pm$\,0.12\,\textmu m and a landmark fitness (LM fitness) of 85.71\%\,$\pm$\,1.02\%. Moreover, BigReg can provide an optimal initialization for mutual information-based methods which otherwise fail independently, further reducing LMD to 7.24\,\textmu m\,$\pm$\,0.11\,\textmu m and increasing LM fitness to 93.90\%\,$\pm$\,0.77\%. Ultimately, key microstructures, notably lacunae in XRM and bone cells in LSFM, are accurately aligned, enabling unprecedented insights into the pathology of osteoporosis.
>
---
#### [replaced 026] Deep activity propagation via weight initialization in spiking neural networks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.00580v2](http://arxiv.org/pdf/2410.00580v2)**

> **作者:** Aurora Micheli; Olaf Booij; Jan van Gemert; Nergis Tömen
>
> **摘要:** Spiking Neural Networks (SNNs) and neuromorphic computing offer bio-inspired advantages such as sparsity and ultra-low power consumption, providing a promising alternative to conventional networks. However, training deep SNNs from scratch remains a challenge, as SNNs process and transmit information by quantizing the real-valued membrane potentials into binary spikes. This can lead to information loss and vanishing spikes in deeper layers, impeding effective training. While weight initialization is known to be critical for training deep neural networks, what constitutes an effective initial state for a deep SNN is not well-understood. Existing weight initialization methods designed for conventional networks (ANNs) are often applied to SNNs without accounting for their distinct computational properties. In this work we derive an optimal weight initialization method specifically tailored for SNNs, taking into account the quantization operation. We show theoretically that, unlike standard approaches, this method enables the propagation of activity in deep SNNs without loss of spikes. We demonstrate this behavior in numerical simulations of SNNs with up to 100 layers across multiple time steps. We present an in-depth analysis of the numerical conditions, regarding layer width and neuron hyperparameters, which are necessary to accurately apply our theoretical findings. Furthermore, our experiments on MNIST demonstrate higher accuracy and faster convergence when using the proposed weight initialization scheme. Finally, we show that the newly introduced weight initialization is robust against variations in several network and neuron hyperparameters.
>
---
#### [replaced 027] DragLoRA: Online Optimization of LoRA Adapters for Drag-based Image Editing in Diffusion Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12427v2](http://arxiv.org/pdf/2505.12427v2)**

> **作者:** Siwei Xia; Li Sun; Tiantian Sun; Qingli Li
>
> **备注:** Accepted by ICML2025
>
> **摘要:** Drag-based editing within pretrained diffusion model provides a precise and flexible way to manipulate foreground objects. Traditional methods optimize the input feature obtained from DDIM inversion directly, adjusting them iteratively to guide handle points towards target locations. However, these approaches often suffer from limited accuracy due to the low representation ability of the feature in motion supervision, as well as inefficiencies caused by the large search space required for point tracking. To address these limitations, we present DragLoRA, a novel framework that integrates LoRA (Low-Rank Adaptation) adapters into the drag-based editing pipeline. To enhance the training of LoRA adapters, we introduce an additional denoising score distillation loss which regularizes the online model by aligning its output with that of the original model. Additionally, we improve the consistency of motion supervision by adapting the input features using the updated LoRA, giving a more stable and accurate input feature for subsequent operations. Building on this, we design an adaptive optimization scheme that dynamically toggles between two modes, prioritizing efficiency without compromising precision. Extensive experiments demonstrate that DragLoRA significantly enhances the control precision and computational efficiency for drag-based image editing. The Codes of DragLoRA are available at: https://github.com/Sylvie-X/DragLoRA.
>
---
#### [replaced 028] A Separable Self-attention Inspired by the State Space Model for Computer Vision
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.02040v2](http://arxiv.org/pdf/2501.02040v2)**

> **作者:** Juntao Zhang; Shaogeng Liu; Kun Bian; You Zhou; Pei Zhang; Jianning Liu; Jun Zhou; Bingyan Liu
>
> **摘要:** Mamba is an efficient State Space Model (SSM) with linear computational complexity. Although SSMs are not suitable for handling non-causal data, Vision Mamba (ViM) methods still demonstrate good performance in tasks such as image classification and object detection. Recent studies have shown that there is a rich theoretical connection between state space models and attention variants. We propose a novel separable self attention method, for the first time introducing some excellent design concepts of Mamba into separable self-attention. To ensure a fair comparison with ViMs, we introduce VMINet, a simple yet powerful prototype architecture, constructed solely by stacking our novel attention modules with the most basic down-sampling layers. Notably, VMINet differs significantly from the conventional Transformer architecture. Our experiments demonstrate that VMINet has achieved competitive results on image classification and high-resolution dense prediction tasks.Code is available at: https://github.com/yws-wxs/VMINet.
>
---
#### [replaced 029] Interactive Rendering of Relightable and Animatable Gaussian Avatars
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.10707v2](http://arxiv.org/pdf/2407.10707v2)**

> **作者:** Youyi Zhan; Tianjia Shao; He Wang; Yin Yang; Kun Zhou
>
> **备注:** IEEE Transactions on Visualization and Computer Graphics. Project page https://gapszju.github.io/InteractRAGA . Code https://github.com/1231234zhan/InteractRAGA
>
> **摘要:** Creating relightable and animatable avatars from multi-view or monocular videos is a challenging task for digital human creation and virtual reality applications. Previous methods rely on neural radiance fields or ray tracing, resulting in slow training and rendering processes. By utilizing Gaussian Splatting, we propose a simple and efficient method to decouple body materials and lighting from sparse-view or monocular avatar videos, so that the avatar can be rendered simultaneously under novel viewpoints, poses, and lightings at interactive frame rates (6.9 fps). Specifically, we first obtain the canonical body mesh using a signed distance function and assign attributes to each mesh vertex. The Gaussians in the canonical space then interpolate from nearby body mesh vertices to obtain the attributes. We subsequently deform the Gaussians to the posed space using forward skinning, and combine the learnable environment light with the Gaussian attributes for shading computation. To achieve fast shadow modeling, we rasterize the posed body mesh from dense viewpoints to obtain the visibility. Our approach is not only simple but also fast enough to allow interactive rendering of avatar animation under environmental light changes. Experiments demonstrate that, compared to previous works, our method can render higher quality results at a faster speed on both synthetic and real datasets.
>
---
#### [replaced 030] CraftsMan3D: High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.14979v2](http://arxiv.org/pdf/2405.14979v2)**

> **作者:** Weiyu Li; Jiarui Liu; Rui Chen; Yixun Liang; Xuelin Chen; Ping Tan; Xiaoxiao Long
>
> **备注:** HomePage: https://craftsman3d.github.io/, Code: https://github.com/wyysf-98/CraftsMan3D
>
> **摘要:** We present a novel generative 3D modeling system, coined CraftsMan, which can generate high-fidelity 3D geometries with highly varied shapes, regular mesh topologies, and detailed surfaces, and, notably, allows for refining the geometry in an interactive manner. Despite the significant advancements in 3D generation, existing methods still struggle with lengthy optimization processes, irregular mesh topologies, noisy surfaces, and difficulties in accommodating user edits, consequently impeding their widespread adoption and implementation in 3D modeling software. Our work is inspired by the craftsman, who usually roughs out the holistic figure of the work first and elaborates the surface details subsequently. Specifically, we employ a 3D native diffusion model, which operates on latent space learned from latent set-based 3D representations, to generate coarse geometries with regular mesh topology in seconds. In particular, this process takes as input a text prompt or a reference image and leverages a powerful multi-view (MV) diffusion model to generate multiple views of the coarse geometry, which are fed into our MV-conditioned 3D diffusion model for generating the 3D geometry, significantly improving robustness and generalizability. Following that, a normal-based geometry refiner is used to significantly enhance the surface details. This refinement can be performed automatically, or interactively with user-supplied edits. Extensive experiments demonstrate that our method achieves high efficacy in producing superior-quality 3D assets compared to existing methods. HomePage: https://craftsman3d.github.io/, Code: https://github.com/wyysf-98/CraftsMan
>
---
#### [replaced 031] Uni4D: A Unified Self-Supervised Learning Framework for Point Cloud Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.04837v2](http://arxiv.org/pdf/2504.04837v2)**

> **作者:** Zhi Zuo; Chenyi Zhuang; Pan Gao; Jie Qin; Hao Feng; Nicu Sebe
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Self-supervised representation learning for point cloud videos remains a challenging problem with two key limitations: (1) existing methods rely on explicit knowledge to learn motion, resulting in suboptimal representations; (2) prior Masked AutoEncoder (MAE) frameworks struggle to bridge the gap between low-level geometry and high-level dynamics in 4D data. In this work, we propose a novel self-disentangled MAE for learning expressive, discriminative, and transferable 4D representations. To overcome the first limitation, we learn motion by aligning high-level semantics in the latent space \textit{without any explicit knowledge}. To tackle the second, we introduce a \textit{self-disentangled learning} strategy that incorporates the latent token with the geometry token within a shared decoder, effectively disentangling low-level geometry and high-level semantics. In addition to the reconstruction objective, we employ three alignment objectives to enhance temporal understanding, including frame-level motion and video-level global information. We show that our pre-trained encoder surprisingly discriminates spatio-temporal representation without further fine-tuning. Extensive experiments on MSR-Action3D, NTU-RGBD, HOI4D, NvGesture, and SHREC'17 demonstrate the superiority of our approach in both coarse-grained and fine-grained 4D downstream tasks. Notably, Uni4D improves action segmentation accuracy on HOI4D by $+3.8\%$.
>
---
#### [replaced 032] AS3D: 2D-Assisted Cross-Modal Understanding with Semantic-Spatial Scene Graphs for 3D Visual Grounding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.04058v2](http://arxiv.org/pdf/2505.04058v2)**

> **作者:** Feng Xiao; Hongbin Xu; Guocan Zhao; Wenxiong Kang
>
> **摘要:** 3D visual grounding aims to localize the unique target described by natural languages in 3D scenes. The significant gap between 3D and language modalities makes it a notable challenge to distinguish multiple similar objects through the described spatial relationships. Current methods attempt to achieve cross-modal understanding in complex scenes via a target-centered learning mechanism, ignoring the perception of referred objects. We propose a novel 2D-assisted 3D visual grounding framework that constructs semantic-spatial scene graphs with referred object discrimination for relationship perception. The framework incorporates a dual-branch visual encoder that utilizes 2D pre-trained attributes to guide the multi-modal object encoding. Furthermore, our cross-modal interaction module uses graph attention to facilitate relationship-oriented information fusion. The enhanced object representation and iterative relational learning enable the model to establish effective alignment between 3D vision and referential descriptions. Experimental results on the popular benchmarks demonstrate our superior performance compared to state-of-the-art methods, especially in addressing the challenges of multiple similar distractors.
>
---
#### [replaced 033] IP-Prompter: Training-Free Theme-Specific Image Generation via Dynamic Visual Prompting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.15641v2](http://arxiv.org/pdf/2501.15641v2)**

> **作者:** Yuxin Zhang; Minyan Luo; Weiming Dong; Xiao Yang; Haibin Huang; Chongyang Ma; Oliver Deussen; Tong-Yee Lee; Changsheng Xu
>
> **备注:** Accepted by ACM SIGGRAPH 2025. Project page: https://ip-prompter.github.io/
>
> **摘要:** The stories and characters that captivate us as we grow up shape unique fantasy worlds, with images serving as the primary medium for visually experiencing these realms. Personalizing generative models through fine-tuning with theme-specific data has become a prevalent approach in text-to-image generation. However, unlike object customization, which focuses on learning specific objects, theme-specific generation encompasses diverse elements such as characters, scenes, and objects. Such diversity also introduces a key challenge: how to adaptively generate multi-character, multi-concept, and continuous theme-specific images (TSI). Moreover, fine-tuning approaches often come with significant computational overhead, time costs, and risks of overfitting. This paper explores a fundamental question: Can image generation models directly leverage images as contextual input, similarly to how large language models use text as context? To address this, we present IP-Prompter, a novel training-free TSI generation method. IP-Prompter introduces visual prompting, a mechanism that integrates reference images into generative models, allowing users to seamlessly specify the target theme without requiring additional training. To further enhance this process, we propose a Dynamic Visual Prompting (DVP) mechanism, which iteratively optimizes visual prompts to improve the accuracy and quality of generated images. Our approach enables diverse applications, including consistent story generation, character design, realistic character generation, and style-guided image generation. Comparative evaluations against state-of-the-art personalization methods demonstrate that IP-Prompter achieves significantly better results and excels in maintaining character identity preserving, style consistency and text alignment, offering a robust and flexible solution for theme-specific image generation.
>
---
#### [replaced 034] How Effective Can Dropout Be in Multiple Instance Learning ?
- **分类: cs.CV; cs.AI; eess.IV; stat.ML**

- **链接: [http://arxiv.org/pdf/2504.14783v2](http://arxiv.org/pdf/2504.14783v2)**

> **作者:** Wenhui Zhu; Peijie Qiu; Xiwen Chen; Zhangsihao Yang; Aristeidis Sotiras; Abolfazl Razi; Yalin Wang
>
> **备注:** Accepted by ICML2025
>
> **摘要:** Multiple Instance Learning (MIL) is a popular weakly-supervised method for various applications, with a particular interest in histological whole slide image (WSI) classification. Due to the gigapixel resolution of WSI, applications of MIL in WSI typically necessitate a two-stage training scheme: first, extract features from the pre-trained backbone and then perform MIL aggregation. However, it is well-known that this suboptimal training scheme suffers from "noisy" feature embeddings from the backbone and inherent weak supervision, hindering MIL from learning rich and generalizable features. However, the most commonly used technique (i.e., dropout) for mitigating this issue has yet to be explored in MIL. In this paper, we empirically explore how effective the dropout can be in MIL. Interestingly, we observe that dropping the top-k most important instances within a bag leads to better performance and generalization even under noise attack. Based on this key observation, we propose a novel MIL-specific dropout method, termed MIL-Dropout, which systematically determines which instances to drop. Experiments on five MIL benchmark datasets and two WSI datasets demonstrate that MIL-Dropout boosts the performance of current MIL methods with a negligible computational cost. The code is available at https://github.com/ChongQingNoSubway/MILDropout.
>
---
#### [replaced 035] CRCE: Coreference-Retention Concept Erasure in Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.14232v2](http://arxiv.org/pdf/2503.14232v2)**

> **作者:** Yuyang Xue; Edward Moroshko; Feng Chen; Jingyu Sun; Steven McDonagh; Sotirios A. Tsaftaris
>
> **摘要:** Text-to-Image diffusion models can produce undesirable content that necessitates concept erasure. However, existing methods struggle with under-erasure, leaving residual traces of targeted concepts, or over-erasure, mistakenly eliminating unrelated but visually similar concepts. To address these limitations, we introduce CRCE, a novel concept erasure framework that leverages Large Language Models to identify both semantically related concepts that should be erased alongside the target and distinct concepts that should be preserved. By explicitly modelling coreferential and retained concepts semantically, CRCE enables more precise concept removal, without unintended erasure. Experiments demonstrate that CRCE outperforms existing methods on diverse erasure tasks, including real-world object, person identities, and abstract intellectual property characteristics. The constructed dataset CorefConcept and the source code will be release upon acceptance.
>
---
#### [replaced 036] InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.12368v2](http://arxiv.org/pdf/2501.12368v2)**

> **作者:** Yuhang Zang; Xiaoyi Dong; Pan Zhang; Yuhang Cao; Ziyu Liu; Shengyuan Ding; Shenxi Wu; Yubo Ma; Haodong Duan; Wenwei Zhang; Kai Chen; Dahua Lin; Jiaqi Wang
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Despite the promising performance of Large Vision Language Models (LVLMs) in visual understanding, they occasionally generate incorrect outputs. While reward models (RMs) with reinforcement learning or test-time scaling offer the potential for improving generation quality, a critical gap remains: publicly available multi-modal RMs for LVLMs are scarce, and the implementation details of proprietary models are often unclear. We bridge this gap with InternLM-XComposer2.5-Reward (IXC-2.5-Reward), a simple yet effective multi-modal reward model that aligns LVLMs with human preferences. To ensure the robustness and versatility of IXC-2.5-Reward, we set up a high-quality multi-modal preference corpus spanning text, image, and video inputs across diverse domains, such as instruction following, general understanding, text-rich documents, mathematical reasoning, and video understanding. IXC-2.5-Reward achieves excellent results on the latest multi-modal reward model benchmark and shows competitive performance on text-only reward model benchmarks. We further demonstrate three key applications of IXC-2.5-Reward: (1) Providing a supervisory signal for RL training. We integrate IXC-2.5-Reward with Proximal Policy Optimization (PPO) yields IXC-2.5-Chat, which shows consistent improvements in instruction following and multi-modal open-ended dialogue; (2) Selecting the best response from candidate responses for test-time scaling; and (3) Filtering outlier or noisy samples from existing image and video instruction tuning training data. To ensure reproducibility and facilitate further research, we have open-sourced all model weights and training recipes at https://github.com/InternLM/InternLM-XComposer/tree/main/InternLM-XComposer-2.5-Reward
>
---
#### [replaced 037] Unforgettable Lessons from Forgettable Images: Intra-Class Memorability Matters in Computer Vision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.20761v3](http://arxiv.org/pdf/2412.20761v3)**

> **作者:** Jie Jing; Qing Lin; Shuangpeng Han; Lucia Schiatti; Yen-Ling Kuo; Mengmi Zhang
>
> **摘要:** We introduce intra-class memorability, where certain images within the same class are more memorable than others despite shared category characteristics. To investigate what features make one object instance more memorable than others, we design and conduct human behavior experiments, where participants are shown a series of images, and they must identify when the current image matches the image presented a few steps back in the sequence. To quantify memorability, we propose the Intra-Class Memorability score (ICMscore), a novel metric that incorporates the temporal intervals between repeated image presentations into its calculation. Furthermore, we curate the Intra-Class Memorability Dataset (ICMD), comprising over 5,000 images across ten object classes with their ICMscores derived from 2,000 participants' responses. Subsequently, we demonstrate the usefulness of ICMD by training AI models on this dataset for various downstream tasks: memorability prediction, image recognition, continual learning, and memorability-controlled image editing. Surprisingly, high-ICMscore images impair AI performance in image recognition and continual learning tasks, while low-ICMscore images improve outcomes in these tasks. Additionally, we fine-tune a state-of-the-art image diffusion model on ICMD image pairs with and without masked semantic objects. The diffusion model can successfully manipulate image elements to enhance or reduce memorability. Our contributions open new pathways in understanding intra-class memorability by scrutinizing fine-grained visual features behind the most and least memorable images and laying the groundwork for real-world applications in computer vision. We will release all code, data, and models publicly.
>
---
#### [replaced 038] Structure-Preserving Zero-Shot Image Editing via Stage-Wise Latent Injection in Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.15723v2](http://arxiv.org/pdf/2504.15723v2)**

> **作者:** Dasol Jeong; Donggoo Kang; Jiwon Park; Hyebean Lee; Joonki Paik
>
> **摘要:** We propose a diffusion-based framework for zero-shot image editing that unifies text-guided and reference-guided approaches without requiring fine-tuning. Our method leverages diffusion inversion and timestep-specific null-text embeddings to preserve the structural integrity of the source image. By introducing a stage-wise latent injection strategy-shape injection in early steps and attribute injection in later steps-we enable precise, fine-grained modifications while maintaining global consistency. Cross-attention with reference latents facilitates semantic alignment between the source and reference. Extensive experiments across expression transfer, texture transformation, and style infusion demonstrate state-of-the-art performance, confirming the method's scalability and adaptability to diverse image editing scenarios.
>
---
#### [replaced 039] Universal Incremental Learning: Mitigating Confusion from Inter- and Intra-task Distribution Randomness
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07035v2](http://arxiv.org/pdf/2503.07035v2)**

> **作者:** Sheng Luo; Yi Zhou; Tao Zhou
>
> **备注:** 10 pages, 4 figures, 4 tables
>
> **摘要:** Incremental learning (IL) aims to overcome catastrophic forgetting of previous tasks while learning new ones. Existing IL methods make strong assumptions that the incoming task type will either only increases new classes or domains (i.e. Class IL, Domain IL), or increase by a static scale in a class- and domain-agnostic manner (i.e. Versatile IL (VIL)), which greatly limit their applicability in the unpredictable and dynamic wild. In this work, we investigate $\textbf{Universal Incremental Learning (UIL)}$, where a model neither knows which new classes or domains will increase along sequential tasks, nor the scale of the increments within each task. This uncertainty prevents the model from confidently learning knowledge from all task distributions and symmetrically focusing on the diverse knowledge within each task distribution. Consequently, UIL presents a more general and realistic IL scenario, making the model face confusion arising from inter-task and intra-task distribution randomness. To $\textbf{Mi}$tigate both $\textbf{Co}$nfusion, we propose a simple yet effective framework for UIL, named $\textbf{MiCo}$. At the inter-task distribution level, we employ a multi-objective learning scheme to enforce accurate and deterministic predictions, and its effectiveness is further enhanced by a direction recalibration module that reduces conflicting gradients. Moreover, at the intra-task distribution level, we introduce a magnitude recalibration module to alleviate asymmetrical optimization towards imbalanced class distribution. Extensive experiments on three benchmarks demonstrate the effectiveness of our method, outperforming existing state-of-the-art methods in both the UIL scenario and the VIL scenario. Our code will be available at $\href{https://github.com/rolsheng/UIL}{here}$.
>
---
#### [replaced 040] StarFT: Robust Fine-tuning of Zero-shot Models via Spuriosity Alignment
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13232v2](http://arxiv.org/pdf/2505.13232v2)**

> **作者:** Younghyun Kim; Jongheon Jeong; Sangkyung Kwak; Kyungmin Lee; Juho Lee; Jinwoo Shin
>
> **备注:** IJCAI 2025; Code is available at https://github.com/alinlab/StarFT
>
> **摘要:** Learning robust representations from data often requires scale, which has led to the success of recent zero-shot models such as CLIP. However, the obtained robustness can easily be deteriorated when these models are fine-tuned on other downstream tasks (e.g., of smaller scales). Previous works often interpret this phenomenon in the context of domain shift, developing fine-tuning methods that aim to preserve the original domain as much as possible. However, in a different context, fine-tuned models with limited data are also prone to learning features that are spurious to humans, such as background or texture. In this paper, we propose StarFT (Spurious Textual Alignment Regularization), a novel framework for fine-tuning zero-shot models to enhance robustness by preventing them from learning spuriosity. We introduce a regularization that aligns the output distribution for spuriosity-injected labels with the original zero-shot model, ensuring that the model is not induced to extract irrelevant features further from these descriptions. We leverage recent language models to get such spuriosity-injected labels by generating alternative textual descriptions that highlight potentially confounding features. Extensive experiments validate the robust generalization of StarFT and its emerging properties: zero-shot group robustness and improved zero-shot classification. Notably, StarFT boosts both worst-group and average accuracy by 14.30% and 3.02%, respectively, in the Waterbirds group shift scenario, where other robust fine-tuning baselines show even degraded performance.
>
---
#### [replaced 041] Attentive Eraser: Unleashing Diffusion Model's Object Removal Potential via Self-Attention Redirection Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.12974v5](http://arxiv.org/pdf/2412.12974v5)**

> **作者:** Wenhao Sun; Benlei Cui; Xue-Mei Dong; Jingqun Tang
>
> **备注:** Accepted by AAAI 2025(Oral)
>
> **摘要:** Recently, diffusion models have emerged as promising newcomers in the field of generative models, shining brightly in image generation. However, when employed for object removal tasks, they still encounter issues such as generating random artifacts and the incapacity to repaint foreground object areas with appropriate content after removal. To tackle these problems, we propose Attentive Eraser, a tuning-free method to empower pre-trained diffusion models for stable and effective object removal. Firstly, in light of the observation that the self-attention maps influence the structure and shape details of the generated images, we propose Attention Activation and Suppression (ASS), which re-engineers the self-attention mechanism within the pre-trained diffusion models based on the given mask, thereby prioritizing the background over the foreground object during the reverse generation process. Moreover, we introduce Self-Attention Redirection Guidance (SARG), which utilizes the self-attention redirected by ASS to guide the generation process, effectively removing foreground objects within the mask while simultaneously generating content that is both plausible and coherent. Experiments demonstrate the stability and effectiveness of Attentive Eraser in object removal across a variety of pre-trained diffusion models, outperforming even training-based methods. Furthermore, Attentive Eraser can be implemented in various diffusion model architectures and checkpoints, enabling excellent scalability. Code is available at https://github.com/Anonym0u3/AttentiveEraser.
>
---
#### [replaced 042] VideoVista-CulturalLingo: 360$^\circ$ Horizons-Bridging Cultures, Languages, and Domains in Video Comprehension
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.17821v2](http://arxiv.org/pdf/2504.17821v2)**

> **作者:** Xinyu Chen; Yunxin Li; Haoyuan Shi; Baotian Hu; Wenhan Luo; Yaowei Wang; Min Zhang
>
> **摘要:** Assessing the video comprehension capabilities of multimodal AI systems can effectively measure their understanding and reasoning abilities. Most video evaluation benchmarks are limited to a single language, typically English, and predominantly feature videos rooted in Western cultural contexts. In this paper, we present VideoVista-CulturalLingo, the first video evaluation benchmark designed to bridge cultural, linguistic, and domain divide in video comprehension. Our work differs from existing benchmarks in the following ways: 1) Cultural diversity, incorporating cultures from China, North America, and Europe; 2) Multi-linguistics, with questions presented in Chinese and English-two of the most widely spoken languages; and 3) Broad domain, featuring videos sourced from hundreds of human-created domains. VideoVista-CulturalLingo contains 1,389 videos and 3,134 QA pairs, and we have evaluated 24 recent open-source or proprietary video large models. From the experiment results, we observe that: 1) Existing models perform worse on Chinese-centric questions than Western-centric ones, particularly those related to Chinese history; 2) Current open-source models still exhibit limitations in temporal understanding, especially in the Event Localization task, achieving a maximum score of only 45.2%; 3) Mainstream models demonstrate strong performance in general scientific questions, while open-source models demonstrate weak performance in mathematics.
>
---
#### [replaced 043] DeepForest: Sensing Into Self-Occluding Volumes of Vegetation With Aerial Imaging
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2502.02171v2](http://arxiv.org/pdf/2502.02171v2)**

> **作者:** Mohamed Youssef; Jian Peng; Oliver Bimber
>
> **摘要:** Access to below-canopy volumetric vegetation data is crucial for understanding ecosystem dynamics. We address the long-standing limitation of remote sensing to penetrate deep into dense canopy layers. LiDAR and radar are currently considered the primary options for measuring 3D vegetation structures, while cameras can only extract the reflectance and depth of top layers. Using conventional, high-resolution aerial images, our approach allows sensing deep into self-occluding vegetation volumes, such as forests. It is similar in spirit to the imaging process of wide-field microscopy, but can handle much larger scales and strong occlusion. We scan focal stacks by synthetic-aperture imaging with drones and reduce outof-focus signal contributions using pre-trained 3D convolutional neural networks with mean squared error (MSE) as the loss function. The resulting volumetric reflectance stacks contain low-frequency representations of the vegetation volume. Combining multiple reflectance stacks from various spectral channels provides insights into plant health, growth, and environmental conditions throughout the entire vegetation volume. Compared with simulated ground truth, our correction leads to ~x7 average improvements (min: ~x2, max: ~x12) for forest densities of 200 trees/ha - 1680 trees/ha. In our field experiment, we achieved an MSE of 0.05 when comparing with the top-vegetation layer that was measured with classical multispectral aerial imaging.
>
---
#### [replaced 044] Breaking Language Barriers in Visual Language Models via Multilingual Textual Regularization
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.22577v2](http://arxiv.org/pdf/2503.22577v2)**

> **作者:** Iñigo Pikabea; Iñaki Lacunza; Oriol Pareras; Carlos Escolano; Aitor Gonzalez-Agirre; Javier Hernando; Marta Villegas
>
> **备注:** v2: Expanded model merging experiments. Fix duplicated subsection on limitations
>
> **摘要:** Rapid advancements in Visual Language Models (VLMs) have transformed multimodal understanding but are often constrained by generating English responses regardless of the input language. This phenomenon has been termed as Image-induced Fidelity Loss (IFL) and stems from limited multimodal multilingual training data. To address this, we propose a continuous multilingual integration strategy that injects text-only multilingual data during visual instruction tuning, preserving the language model's original multilingual capabilities. Extensive evaluations demonstrate that our approach significantly improves linguistic fidelity across languages without degradation in visual performance. We also explore model merging, which improves language fidelity but comes at the cost of visual performance. In contrast, our core method achieves robust multilingual alignment without trade-offs, offering a scalable and effective path to mitigating IFL for global VLM adoption.
>
---
#### [replaced 045] A portable diagnosis model for Keratoconus using a smartphone
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.08616v3](http://arxiv.org/pdf/2505.08616v3)**

> **作者:** Yifan Li; Peter Ho; Jo Woon Chong
>
> **摘要:** Keratoconus (KC) is a corneal disorder that results in blurry and distorted vision. Traditional diagnostic tools, while effective, are often bulky, costly, and require professional operation. In this paper, we present a portable and innovative methodology for diagnosing. Our proposed approach first captures the image reflected on the eye's cornea when a smartphone screen-generated Placido disc sheds its light on an eye, then utilizes a two-stage diagnosis for identifying the KC cornea and pinpointing the location of the KC on the cornea. The first stage estimates the height and width of the Placido disc extracted from the captured image to identify whether it has KC. In this KC identification, k-means clustering is implemented to discern statistical characteristics, such as height and width values of extracted Placido discs, from non-KC (control) and KC-affected groups. The second stage involves the creation of a distance matrix, providing a precise localization of KC on the cornea, which is critical for efficient treatment planning. The analysis of these distance matrices, paired with a logistic regression model and robust statistical analysis, reveals a clear distinction between control and KC groups. The logistic regression model, which classifies small areas on the cornea as either control or KC-affected based on the corresponding inter-disc distances in the distance matrix, reported a classification accuracy of 96.94%, which indicates that we can effectively pinpoint the protrusion caused by KC. This comprehensive, smartphone-based method is expected to detect KC and streamline timely treatment.
>
---
#### [replaced 046] Multimodal Fusion of Glucose Monitoring and Food Imagery for Caloric Content Prediction
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.09018v2](http://arxiv.org/pdf/2505.09018v2)**

> **作者:** Adarsh Kumar
>
> **备注:** The manuscript was submitted without proper consideration of institutional policies. Upon review with professor, it was found that the content is subject to licensing restrictions which prohibit public dissemination in its current form. Therefore, I am withdrawing the paper to comply with these requirements
>
> **摘要:** Effective dietary monitoring is critical for managing Type 2 diabetes, yet accurately estimating caloric intake remains a major challenge. While continuous glucose monitors (CGMs) offer valuable physiological data, they often fall short in capturing the full nutritional profile of meals due to inter-individual and meal-specific variability. In this work, we introduce a multimodal deep learning framework that jointly leverages CGM time-series data, Demographic/Microbiome, and pre-meal food images to enhance caloric estimation. Our model utilizes attention based encoding and a convolutional feature extraction for meal imagery, multi-layer perceptrons for CGM and Microbiome data followed by a late fusion strategy for joint reasoning. We evaluate our approach on a curated dataset of over 40 participants, incorporating synchronized CGM, Demographic and Microbiome data and meal photographs with standardized caloric labels. Our model achieves a Root Mean Squared Relative Error (RMSRE) of 0.2544, outperforming the baselines models by over 50%. These findings demonstrate the potential of multimodal sensing to improve automated dietary assessment tools for chronic disease management.
>
---
#### [replaced 047] Latent Action Learning Requires Supervision in the Presence of Distractors
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.00379v4](http://arxiv.org/pdf/2502.00379v4)**

> **作者:** Alexander Nikulin; Ilya Zisman; Denis Tarasov; Nikita Lyubaykin; Andrei Polubarov; Igor Kiselev; Vladislav Kurenkov
>
> **备注:** ICML 2025, Poster, Source code: https://github.com/dunnolab/laom
>
> **摘要:** Recently, latent action learning, pioneered by Latent Action Policies (LAPO), have shown remarkable pre-training efficiency on observation-only data, offering potential for leveraging vast amounts of video available on the web for embodied AI. However, prior work has focused on distractor-free data, where changes between observations are primarily explained by ground-truth actions. Unfortunately, real-world videos contain action-correlated distractors that may hinder latent action learning. Using Distracting Control Suite (DCS) we empirically investigate the effect of distractors on latent action learning and demonstrate that LAPO struggle in such scenario. We propose LAOM, a simple LAPO modification that improves the quality of latent actions by 8x, as measured by linear probing. Importantly, we show that providing supervision with ground-truth actions, as few as 2.5% of the full dataset, during latent action learning improves downstream performance by 4.2x on average. Our findings suggest that integrating supervision during Latent Action Models (LAM) training is critical in the presence of distractors, challenging the conventional pipeline of first learning LAM and only then decoding from latent to ground-truth actions.
>
---
#### [replaced 048] FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Pretraining
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11192v3](http://arxiv.org/pdf/2505.11192v3)**

> **作者:** Myunsoo Kim; Seong-Woong Shim; Byung-Jun Lee
>
> **摘要:** False negatives pose a critical challenge in vision-language pretraining (VLP) due to the many-to-many correspondence between images and texts in large-scale datasets. These false negatives introduce conflicting supervision signals that degrade the learned embedding space and diminish the effectiveness of hard negative sampling. In this paper, we propose FALCON (False-negative Aware Learning of COntrastive Negatives), a learning-based mini-batch construction strategy that adaptively balances the trade-off between hard and false negatives during VLP. Rather than relying on fixed heuristics, FALCON employs a negative mining scheduler that dynamically selects negative samples of appropriate hardness for each anchor instance during mini-batch construction, guided by a proxy for cross-modal alignment improvement. Experimental results demonstrate that FALCON significantly improves performance across two widely adopted VLP frameworks (ALBEF, BLIP-2) and a broad range of downstream tasks and evaluation settings, underscoring its effectiveness and robustness in mitigating the impact of false negatives.
>
---
#### [replaced 049] Automatic Synthetic Data and Fine-grained Adaptive Feature Alignment for Composed Person Retrieval
- **分类: cs.CV; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2311.16515v4](http://arxiv.org/pdf/2311.16515v4)**

> **作者:** Delong Liu; Haiwen Li; Zhaohui Hou; Zhicheng Zhao; Fei Su; Yuan Dong
>
> **摘要:** Person retrieval has attracted rising attention. Existing methods are mainly divided into two retrieval modes, namely image-only and text-only. However, they are unable to make full use of the available information and are difficult to meet diverse application requirements. To address the above limitations, we propose a new Composed Person Retrieval (CPR) task, which combines visual and textual queries to identify individuals of interest from large-scale person image databases. Nevertheless, the foremost difficulty of the CPR task is the lack of available annotated datasets. Therefore, we first introduce a scalable automatic data synthesis pipeline, which decomposes complex multimodal data generation into the creation of textual quadruples followed by identity-consistent image synthesis using fine-tuned generative models. Meanwhile, a multimodal filtering method is designed to ensure the resulting SynCPR dataset retains 1.15 million high-quality and fully synthetic triplets. Additionally, to improve the representation of composed person queries, we propose a novel Fine-grained Adaptive Feature Alignment (FAFA) framework through fine-grained dynamic alignment and masked feature reasoning. Moreover, for objective evaluation, we manually annotate the Image-Text Composed Person Retrieval (ITCPR) test set. The extensive experiments demonstrate the effectiveness of the SynCPR dataset and the superiority of the proposed FAFA framework when compared with the state-of-the-art methods. All code and data will be provided at https://github.com/Delong-liu-bupt/Composed_Person_Retrieval.
>
---
#### [replaced 050] Benchmarking Unified Face Attack Detection via Hierarchical Prompt Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13327v2](http://arxiv.org/pdf/2505.13327v2)**

> **作者:** Ajian Liu; Haocheng Yuan; Xiao Guo; Hui Ma; Wanyi Zhuang; Changtao Miao; Yan Hong; Chuanbiao Song; Jun Lan; Qi Chu; Tao Gong; Yanyan Liang; Weiqiang Wang; Jun Wan; Xiaoming Liu; Zhen Lei
>
> **摘要:** Presentation Attack Detection and Face Forgery Detection are designed to protect face data from physical media-based Presentation Attacks and digital editing-based DeepFakes respectively. But separate training of these two models makes them vulnerable to unknown attacks and burdens deployment environments. The lack of a Unified Face Attack Detection model to handle both types of attacks is mainly due to two factors. First, there's a lack of adequate benchmarks for models to explore. Existing UAD datasets have limited attack types and samples, restricting the model's ability to address advanced threats. To address this, we propose UniAttackDataPlus (UniAttackData+), the most extensive and sophisticated collection of forgery techniques to date. It includes 2,875 identities and their 54 kinds of falsified samples, totaling 697,347 videos. Second, there's a lack of a reliable classification criterion. Current methods try to find an arbitrary criterion within the same semantic space, which fails when encountering diverse attacks. So, we present a novel Visual-Language Model-based Hierarchical Prompt Tuning Framework (HiPTune) that adaptively explores multiple classification criteria from different semantic spaces. We build a Visual Prompt Tree to explore various classification rules hierarchically. Then, by adaptively pruning the prompts, the model can select the most suitable prompts to guide the encoder to extract discriminative features at different levels in a coarse-to-fine way. Finally, to help the model understand the classification criteria in visual space, we propose a Dynamically Prompt Integration module to project the visual prompts to the text encoder for more accurate semantics. Experiments on 12 datasets have shown the potential to inspire further innovations in the UAD field.
>
---
#### [replaced 051] ActiveSSF: An Active-Learning-Guided Self-Supervised Framework for Long-Tailed Megakaryocyte Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.08200v2](http://arxiv.org/pdf/2502.08200v2)**

> **作者:** Linghao Zhuang; Ying Zhang; Gege Yuan; Xingyue Zhao; Zhiping Jiang
>
> **备注:** 6 pages
>
> **摘要:** Precise classification of megakaryocytes is crucial for diagnosing myelodysplastic syndromes. Although self-supervised learning has shown promise in medical image analysis, its application to classifying megakaryocytes in stained slides faces three main challenges: (1) pervasive background noise that obscures cellular details, (2) a long-tailed distribution that limits data for rare subtypes, and (3) complex morphological variations leading to high intra-class variability. To address these issues, we propose the ActiveSSF framework, which integrates active learning with self-supervised pretraining. Specifically, our approach employs Gaussian filtering combined with K-means clustering and HSV analysis (augmented by clinical prior knowledge) for accurate region-of-interest extraction; an adaptive sample selection mechanism that dynamically adjusts similarity thresholds to mitigate class imbalance; and prototype clustering on labeled samples to overcome morphological complexity. Experimental results on clinical megakaryocyte datasets demonstrate that ActiveSSF not only achieves state-of-the-art performance but also significantly improves recognition accuracy for rare subtypes. Moreover, the integration of these advanced techniques further underscores the practical potential of ActiveSSF in clinical settings.
>
---
#### [replaced 052] VLMs as GeoGuessr Masters: Exceptional Performance, Hidden Biases, and Privacy Risks
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11163v2](http://arxiv.org/pdf/2502.11163v2)**

> **作者:** Jingyuan Huang; Jen-tse Huang; Ziyi Liu; Xiaoyuan Liu; Wenxuan Wang; Jieyu Zhao
>
> **备注:** 8 pages of main text; 5 pages of appendix
>
> **摘要:** Visual-Language Models (VLMs) have shown remarkable performance across various tasks, particularly in recognizing geographic information from images. However, VLMs still show regional biases in this task. To systematically evaluate these issues, we introduce a benchmark consisting of 1,200 images paired with detailed geographic metadata. Evaluating four VLMs, we find that while these models demonstrate the ability to recognize geographic information from images, achieving up to 53.8% accuracy in city prediction, they exhibit significant biases. Specifically, performance is substantially higher for economically developed and densely populated regions compared to less developed (-12.5%) and sparsely populated (-17.0%) areas. Moreover, regional biases of frequently over-predicting certain locations remain. For instance, they consistently predict Sydney for images taken in Australia, shown by the low entropy scores for these countries. The strong performance of VLMs also raises privacy concerns, particularly for users who share images online without the intent of being identified. Our code and dataset are publicly available at https://github.com/uscnlp-lime/FairLocator.
>
---
#### [replaced 053] HWA-UNETR: Hierarchical Window Aggregate UNETR for 3D Multimodal Gastric Lesion Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10464v2](http://arxiv.org/pdf/2505.10464v2)**

> **作者:** Jiaming Liang; Lihuan Dai; Xiaoqi Sheng; Xiangguang Chen; Chun Yao; Guihua Tao; Qibin Leng; Hongmin Cai; Xi Zhong
>
> **备注:** This work has been provisionally accepted for MICCAI 2025
>
> **摘要:** Multimodal medical image segmentation faces significant challenges in the context of gastric cancer lesion analysis. This clinical context is defined by the scarcity of independent multimodal datasets and the imperative to amalgamate inherently misaligned modalities. As a result, algorithms are constrained to train on approximate data and depend on application migration, leading to substantial resource expenditure and a potential decline in analysis accuracy. To address those challenges, we have made two major contributions: First, we publicly disseminate the GCM 2025 dataset, which serves as the first large-scale, open-source collection of gastric cancer multimodal MRI scans, featuring professionally annotated FS-T2W, CE-T1W, and ADC images from 500 patients. Second, we introduce HWA-UNETR, a novel 3D segmentation framework that employs an original HWA block with learnable window aggregation layers to establish dynamic feature correspondences between different modalities' anatomical structures, and leverages the innovative tri-orientated fusion mamba mechanism for context modeling and capturing long-range spatial dependencies. Extensive experiments on our GCM 2025 dataset and the publicly BraTS 2021 dataset validate the performance of our framework, demonstrating that the new approach surpasses existing methods by up to 1.68\% in the Dice score while maintaining solid robustness. The dataset and code are public via https://github.com/JeMing-creater/HWA-UNETR.
>
---
#### [replaced 054] Exploring Social Media Image Categorization Using Large Models with Different Adaptation Methods: A Case Study on Cultural Nature's Contributions to People
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.00275v3](http://arxiv.org/pdf/2410.00275v3)**

> **作者:** Rohaifa Khaldi; Domingo Alcaraz-Segura; Ignacio Sánchez-Herrera; Javier Martinez-Lopez; Carlos Javier Navarro; Siham Tabik
>
> **备注:** 23 pages, 7 figures
>
> **摘要:** Social media images provide valuable insights for modeling, mapping, and understanding human interactions with natural and cultural heritage. However, categorizing these images into semantically meaningful groups remains highly complex due to the vast diversity and heterogeneity of their visual content as they contain an open-world human and nature elements. This challenge becomes greater when categories involve abstract concepts and lack consistent visual patterns. Related studies involve human supervision in the categorization process and the lack of public benchmark datasets make comparisons between these works unfeasible. On the other hand, the continuous advances in large models, including Large Language Models (LLMs), Large Visual Models (LVMs), and Large Visual Language Models (LVLMs), provide a large space of unexplored solutions. In this work 1) we introduce FLIPS a dataset of Flickr images that capture the interaction between human and nature, and 2) evaluate various solutions based on different types and combinations of large models using various adaptation methods. We assess and report their performance in terms of cost, productivity, scalability, and result quality to address the challenges of social media image categorization.
>
---
#### [replaced 055] DeepMpMRI: Tensor-decomposition Regularized Learning for Fast and High-Fidelity Multi-Parametric Microstructural MR Imaging
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.03159v2](http://arxiv.org/pdf/2405.03159v2)**

> **作者:** Wenxin Fan; Jian Cheng; Qiyuan Tian; Ruoyou Wu; Juan Zou; Zan Chen; Shanshan Wang
>
> **摘要:** Deep learning has emerged as a promising approach for learning the nonlinear mapping between diffusion-weighted MR images and tissue parameters, which enables automatic and deep understanding of the brain microstructures. However, the efficiency and accuracy in estimating multiple microstructural parameters derived from multiple diffusion models are still limited since previous studies tend to estimate parameter maps from distinct models with isolated signal modeling and dense sampling. This paper proposes DeepMpMRI, an efficient framework for fast and high-fidelity multiple microstructural parameter estimation from multiple models using highly sparse sampled q-space data. DeepMpMRI is equipped with a newly designed tensor-decomposition-based regularizer to effectively capture fine details by exploiting the high-dimensional correlation across microstructural parameters. In addition, we introduce a Nesterov-based adaptive learning algorithm that optimizes the regularization parameter dynamically to enhance the performance. DeepMpMRI is an extendable framework capable of incorporating flexible network architecture. Experimental results on the HCP dataset and the Alzheimer's disease dataset both demonstrate the superiority of our approach over 5 state-of-the-art methods in simultaneously estimating multi-model microstructural parameter maps for DKI and NODDI model with fine-grained details both quantitatively and qualitatively, achieving 4.5 - 15 $\times$ acceleration compared to the dense sampling of a total of 270 diffusion gradients.
>
---
#### [replaced 056] Bayesian Deep Learning Approaches for Uncertainty-Aware Retinal OCT Image Segmentation for Multiple Sclerosis
- **分类: eess.IV; cs.CV; 68U10, 92C55; I.2.10; I.4.6; J.3**

- **链接: [http://arxiv.org/pdf/2505.12061v2](http://arxiv.org/pdf/2505.12061v2)**

> **作者:** Samuel T. M. Ball
>
> **摘要:** Optical Coherence Tomography (OCT) provides valuable insights in ophthalmology, cardiology, and neurology due to high-resolution, cross-sectional images of the retina. One critical task for ophthalmologists using OCT is delineation of retinal layers within scans. This process is time-consuming and prone to human bias, affecting the accuracy and reliability of diagnoses. Previous efforts to automate delineation using deep learning face challenges in uptake from clinicians and statisticians due to the absence of uncertainty estimation, leading to "confidently wrong" models via hallucinations. In this study, we address these challenges by applying Bayesian convolutional neural networks (BCNNs) to segment an openly available OCT imaging dataset containing 35 human retina OCTs split between healthy controls and patients with multiple sclerosis. Our findings demonstrate that Bayesian models can be used to provide uncertainty maps of the segmentation, which can further be used to identify highly uncertain samples that exhibit recording artefacts such as noise or miscalibration at inference time. Our method also allows for uncertainty-estimation for important secondary measurements such as layer thicknesses, that are medically relevant for patients. We show that these features come in addition to greater performance compared to similar work over all delineations; with an overall Dice score of 95.65%. Our work brings greater clinical applicability, statistical robustness, and performance to retinal OCT segmentation.
>
---
#### [replaced 057] LogicQA: Logical Anomaly Detection with Vision Language Model Generated Questions
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.20252v2](http://arxiv.org/pdf/2503.20252v2)**

> **作者:** Yejin Kwon; Daeun Moon; Youngje Oh; Hyunsoo Yoon
>
> **备注:** Accepted Industry Track at ACL 2025
>
> **摘要:** Anomaly Detection (AD) focuses on detecting samples that differ from the standard pattern, making it a vital tool in process control. Logical anomalies may appear visually normal yet violate predefined constraints on object presence, arrangement, or quantity, depending on reasoning and explainability. We introduce LogicQA, a framework that enhances AD by providing industrial operators with explanations for logical anomalies. LogicQA compiles automatically generated questions into a checklist and collects responses to identify violations of logical constraints. LogicQA is training-free, annotation-free, and operates in a few-shot setting. We achieve state-of-the-art (SOTA) Logical AD performance on public benchmarks, MVTec LOCO AD, with an AUROC of 87.6 percent and an F1-max of 87.0 percent along with the explanations of anomalies. Also, our approach has shown outstanding performance on semiconductor SEM corporate data, further validating its effectiveness in industrial applications.
>
---
#### [replaced 058] Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13061v2](http://arxiv.org/pdf/2502.13061v2)**

> **作者:** Jingbiao Mei; Jinghong Chen; Guangyu Yang; Weizhe Lin; Bill Byrne
>
> **备注:** Preprint. Under Review
>
> **摘要:** Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While LMMs have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both SFT and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability.
>
---
#### [replaced 059] MomentSeeker: A Task-Oriented Benchmark For Long-Video Moment Retrieval
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.12558v4](http://arxiv.org/pdf/2502.12558v4)**

> **作者:** Huaying Yuan; Jian Ni; Zheng Liu; Yueze Wang; Junjie Zhou; Zhengyang Liang; Bo Zhao; Zhao Cao; Zhicheng Dou; Ji-Rong Wen
>
> **摘要:** Accurately locating key moments within long videos is crucial for solving long video understanding (LVU) tasks. However, existing benchmarks are either severely limited in terms of video length and task diversity, or they focus solely on the end-to-end LVU performance, making them inappropriate for evaluating whether key moments can be accurately accessed. To address this challenge, we propose MomentSeeker, a novel benchmark for long-video moment retrieval (LMVR), distinguished by the following features. First, it is created based on long and diverse videos, averaging over 1200 seconds in duration and collected from various domains, e.g., movie, anomaly, egocentric, and sports. Second, it covers a variety of real-world scenarios in three levels: global-level, event-level, object-level, covering common tasks like action recognition, object localization, and causal reasoning, etc. Third, it incorporates rich forms of queries, including text-only queries, image-conditioned queries, and video-conditioned queries. On top of MomentSeeker, we conduct comprehensive experiments for both generation-based approaches (directly using MLLMs) and retrieval-based approaches (leveraging video retrievers). Our results reveal the significant challenges in long-video moment retrieval in terms of accuracy and efficiency, despite improvements from the latest long-video MLLMs and task-specific fine-tuning. We have publicly released MomentSeeker(https://yhy-2000.github.io/MomentSeeker/) to facilitate future research in this area.
>
---
#### [replaced 060] Re-Align: Aligning Vision Language Models via Retrieval-Augmented Direct Preference Optimization
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13146v2](http://arxiv.org/pdf/2502.13146v2)**

> **作者:** Shuo Xing; Yuping Wang; Peiran Li; Ruizheng Bai; Yueqi Wang; Chan-wei Hu; Chengxuan Qian; Huaxiu Yao; Zhengzhong Tu
>
> **备注:** 17 pages
>
> **摘要:** The emergence of large Vision Language Models (VLMs) has broadened the scope and capabilities of single-modal Large Language Models (LLMs) by integrating visual modalities, thereby unlocking transformative cross-modal applications in a variety of real-world scenarios. Despite their impressive performance, VLMs are prone to significant hallucinations, particularly in the form of cross-modal inconsistencies. Building on the success of Reinforcement Learning from Human Feedback (RLHF) in aligning LLMs, recent advancements have focused on applying direct preference optimization (DPO) on carefully curated datasets to mitigate these issues. Yet, such approaches typically introduce preference signals in a brute-force manner, neglecting the crucial role of visual information in the alignment process. In this paper, we introduce Re-Align, a novel alignment framework that leverages image retrieval to construct a dual-preference dataset, effectively incorporating both textual and visual preference signals. We further introduce rDPO, an extension of the standard direct preference optimization that incorporates an additional visual preference objective during fine-tuning. Our experimental results demonstrate that Re-Align not only mitigates hallucinations more effectively than previous methods but also yields significant performance gains in general visual question-answering (VQA) tasks. Moreover, we show that Re-Align maintains robustness and scalability across a wide range of VLM sizes and architectures. This work represents a significant step forward in aligning multimodal LLMs, paving the way for more reliable and effective cross-modal applications. We release all the code in https://github.com/taco-group/Re-Align.
>
---
#### [replaced 061] Learning Joint ID-Textual Representation for ID-Preserving Image Synthesis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.14202v2](http://arxiv.org/pdf/2504.14202v2)**

> **作者:** Zichuan Liu; Liming Jiang; Qing Yan; Yumin Jia; Hao Kang; Xin Lu
>
> **摘要:** We propose a novel framework for ID-preserving generation using a multi-modal encoding strategy rather than injecting identity features via adapters into pre-trained models. Our method treats identity and text as a unified conditioning input. To achieve this, we introduce FaceCLIP, a multi-modal encoder that learns a joint embedding space for both identity and textual semantics. Given a reference face and a text prompt, FaceCLIP produces a unified representation that encodes both identity and text, which conditions a base diffusion model to generate images that are identity-consistent and text-aligned. We also present a multi-modal alignment algorithm to train FaceCLIP, using a loss that aligns its joint representation with face, text, and image embedding spaces. We then build FaceCLIP-SDXL, an ID-preserving image synthesis pipeline by integrating FaceCLIP with Stable Diffusion XL (SDXL). Compared to prior methods, FaceCLIP-SDXL enables photorealistic portrait generation with better identity preservation and textual relevance. Extensive experiments demonstrate its quantitative and qualitative superiority.
>
---
#### [replaced 062] GranQ: Granular Zero-Shot Quantization with Channel-Wise Activation Scaling in QAT
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18339v4](http://arxiv.org/pdf/2503.18339v4)**

> **作者:** Inpyo Hong; Youngwan Jo; Hyojeong Lee; Sunghyun Ahn; Sanghyun Park
>
> **摘要:** Zero-shot quantization (ZSQ) enables neural network compression without original training data, making it a promising solution for restricted data access scenarios. To compensate for the lack of data, recent ZSQ methods typically rely on synthetic inputs generated from the full-precision model. However, these synthetic inputs often lead to activation distortion, especially under low-bit settings. As a result, existing methods struggle to mitigate this issue due to coarse activation scaling. To address this issue, we propose GranQ, a novel activation quantization framework that efficiently applies per-channel scaling through vectorized computation. In contrast to conventional channel-wise methods, which apply vectorization only to the quantization step, GranQ improves efficiency by vectorizing the scaling operation. This design allows GranQ to maintain fine-grained quantization granularity with minimal computational overhead, even in low-bit environments. Extensive experiments under quantization-aware training (QAT) settings demonstrate that GranQ consistently outperforms state-of-the-art ZSQ methods across CIFAR and ImageNet. In particular, our method achieves up to 5.45% higher accuracy in the 3-bit setting on CIFAR-100 and even surpasses the full-precision baseline on CIFAR-10. Furthermore, GranQ achieves significant speedup in quantization latency over conventional per-channel methods, demonstrating improved efficiency. With these findings, we anticipate that GranQ will inspire future research beyond conventional ZSQ approaches centered on data generation and model fine-tuning.
>
---
#### [replaced 063] Gradient Leakage Defense with Key-Lock Module for Federated Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2305.04095v2](http://arxiv.org/pdf/2305.04095v2)**

> **作者:** Hanchi Ren; Jingjing Deng; Xianghua Xie; Xiaoke Ma; Jianfeng Ma
>
> **备注:** The source code can be found at https://github.com/Rand2AI/FedKL
>
> **摘要:** Federated Learning (FL) is a widely adopted privacy-preserving machine learning approach where private data remains local, enabling secure computations and the exchange of local model gradients between local clients and third-party parameter servers. However, recent findings reveal that privacy may be compromised and sensitive information potentially recovered from shared gradients. In this study, we offer detailed analysis and a novel perspective on understanding the gradient leakage problem. These theoretical works lead to a new gradient leakage defense technique that secures arbitrary model architectures using a private key-lock module. Only the locked gradient is transmitted to the parameter server for global model aggregation. Our proposed learning method is resistant to gradient leakage attacks, and the key-lock module is designed and trained to ensure that, without the private information of the key-lock module: a) reconstructing private training data from the shared gradient is infeasible; and b) the global model's inference performance is significantly compromised. We discuss the theoretical underpinnings of why gradients can leak private information and provide theoretical proof of our method's effectiveness. We conducted extensive empirical evaluations with many models on several popular benchmarks, demonstrating the robustness of our proposed approach in both maintaining model performance and defending against gradient leakage attacks.
>
---
#### [replaced 064] OT-DETECTOR: Delving into Optimal Transport for Zero-shot Out-of-Distribution Detection
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.06442v2](http://arxiv.org/pdf/2503.06442v2)**

> **作者:** Yu Liu; Hao Tang; Haiqi Zhang; Jing Qin; Zechao Li
>
> **备注:** Accepted to the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025)
>
> **摘要:** Out-of-distribution (OOD) detection is crucial for ensuring the reliability and safety of machine learning models in real-world applications. While zero-shot OOD detection, which requires no training on in-distribution (ID) data, has become feasible with the emergence of vision-language models like CLIP, existing methods primarily focus on semantic matching and fail to fully capture distributional discrepancies. To address these limitations, we propose OT-DETECTOR, a novel framework that employs Optimal Transport (OT) to quantify both semantic and distributional discrepancies between test samples and ID labels. Specifically, we introduce cross-modal transport mass and transport cost as semantic-wise and distribution-wise OOD scores, respectively, enabling more robust detection of OOD samples. Additionally, we present a semantic-aware content refinement (SaCR) module, which utilizes semantic cues from ID labels to amplify the distributional discrepancy between ID and hard OOD samples. Extensive experiments on several benchmarks demonstrate that OT-DETECTOR achieves state-of-the-art performance across various OOD detection tasks, particularly in challenging hard-OOD scenarios.
>
---
#### [replaced 065] Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation
- **分类: cs.LG; cs.AI; cs.CV; cs.IR**

- **链接: [http://arxiv.org/pdf/2503.01776v5](http://arxiv.org/pdf/2503.01776v5)**

> **作者:** Tiansheng Wen; Yifei Wang; Zequn Zeng; Zhong Peng; Yudi Su; Xinyang Liu; Bo Chen; Hongwei Liu; Stefanie Jegelka; Chenyu You
>
> **备注:** Accepted by ICML2025
>
> **摘要:** Many large-scale systems rely on high-quality deep representations (embeddings) to facilitate tasks like retrieval, search, and generative modeling. Matryoshka Representation Learning (MRL) recently emerged as a solution for adaptive embedding lengths, but it requires full model retraining and suffers from noticeable performance degradations at short lengths. In this paper, we show that sparse coding offers a compelling alternative for achieving adaptive representation with minimal overhead and higher fidelity. We propose Contrastive Sparse Representation (CSR), a method that sparsifies pre-trained embeddings into a high-dimensional but selectively activated feature space. By leveraging lightweight autoencoding and task-aware contrastive objectives, CSR preserves semantic quality while allowing flexible, cost-effective inference at different sparsity levels. Extensive experiments on image, text, and multimodal benchmarks demonstrate that CSR consistently outperforms MRL in terms of both accuracy and retrieval speed-often by large margins-while also cutting training time to a fraction of that required by MRL. Our results establish sparse coding as a powerful paradigm for adaptive representation learning in real-world applications where efficiency and fidelity are both paramount. Code is available at https://github.com/neilwen987/CSR_Adaptive_Rep
>
---
#### [replaced 066] MMDocIR: Benchmarking Multi-Modal Retrieval for Long Documents
- **分类: cs.IR; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.08828v2](http://arxiv.org/pdf/2501.08828v2)**

> **作者:** Kuicai Dong; Yujing Chang; Xin Deik Goh; Dexun Li; Ruiming Tang; Yong Liu
>
> **备注:** https://huggingface.co/MMDocIR
>
> **摘要:** Multimodal document retrieval aims to identify and retrieve various forms of multimodal content, such as figures, tables, charts, and layout information from extensive documents. Despite its increasing popularity, there is a notable lack of a comprehensive and robust benchmark to effectively evaluate the performance of systems in such tasks. To address this gap, this work introduces a new benchmark, named MMDocIR, that encompasses two distinct tasks: page-level and layout-level retrieval. The former evaluates the performance of identifying the most relevant pages within a long document, while the later assesses the ability of detecting specific layouts, providing a more fine-grained measure than whole-page analysis. A layout refers to a variety of elements, including textual paragraphs, equations, figures, tables, or charts. The MMDocIR benchmark comprises a rich dataset featuring 1,685 questions annotated by experts and 173,843 questions with bootstrapped labels, making it a valuable resource in multimodal document retrieval for both training and evaluation. Through rigorous experiments, we demonstrate that (i) visual retrievers significantly outperform their text counterparts, (ii) MMDocIR training set effectively enhances the performance of multimodal document retrieval and (iii) text retrievers leveraging VLM-text significantly outperforms retrievers relying on OCR-text. Our dataset is available at https://mmdocrag.github.io/MMDocIR/.
>
---
#### [replaced 067] A Unified Framework for Event-based Frame Interpolation with Ad-hoc Deblurring in the Wild
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2301.05191v3](http://arxiv.org/pdf/2301.05191v3)**

> **作者:** Lei Sun; Daniel Gehrig; Christos Sakaridis; Mathias Gehrig; Jingyun Liang; Peng Sun; Zhijie Xu; Kaiwei Wang; Luc Van Gool; Davide Scaramuzza
>
> **备注:** Accepted to T-PAMI
>
> **摘要:** Effective video frame interpolation hinges on the adept handling of motion in the input scene. Prior work acknowledges asynchronous event information for this, but often overlooks whether motion induces blur in the video, limiting its scope to sharp frame interpolation. We instead propose a unified framework for event-based frame interpolation that performs deblurring ad-hoc and thus works both on sharp and blurry input videos. Our model consists in a bidirectional recurrent network that incorporates the temporal dimension of interpolation and fuses information from the input frames and the events adaptively based on their temporal proximity. To enhance the generalization from synthetic data to real event cameras, we integrate self-supervised framework with the proposed model to enhance the generalization on real-world datasets in the wild. At the dataset level, we introduce a novel real-world high-resolution dataset with events and color videos named HighREV, which provides a challenging evaluation setting for the examined task. Extensive experiments show that our network consistently outperforms previous state-of-the-art methods on frame interpolation, single image deblurring, and the joint task of both. Experiments on domain transfer reveal that self-supervised training effectively mitigates the performance degradation observed when transitioning from synthetic data to real-world data. Code and datasets are available at https://github.com/AHupuJR/REFID.
>
---
#### [replaced 068] Does Acceleration Cause Hidden Instability in Vision Language Models? Uncovering Instance-Level Divergence Through a Large-Scale Empirical Study
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.06794v3](http://arxiv.org/pdf/2503.06794v3)**

> **作者:** Yizheng Sun; Hao Li; Chang Xu; Hongpeng Zhou; Chenghua Lin; Riza Batista-Navarro; Jingyuan Sun
>
> **摘要:** Vision-Language Models (VLMs) are powerful yet computationally intensive for widespread practical deployments. To address such challenge without costly re-training, post-training acceleration techniques like quantization and token reduction are extensively explored. However, current acceleration evaluations primarily target minimal overall performance degradation, overlooking a crucial question: does the accelerated model still give the same answers to the same questions as it did before acceleration? This is vital for stability-centered industrial applications where consistently correct answers for specific, known situations are paramount, such as in AI-based disease diagnosis. We systematically investigate this for accelerated VLMs, testing four leading models (LLaVA-1.5, LLaVA-Next, Qwen2-VL, Qwen2.5-VL) with eight acceleration methods on ten multi-modal benchmarks. Our findings are stark: despite minimal aggregate performance drops, accelerated models changed original answers up to 20% of the time. Critically, up to 6.5% of these changes converted correct answers to incorrect. Input perturbations magnified these inconsistencies, and the trend is confirmed by case studies with the medical VLM LLaVA-Med. This research reveals a significant oversight in VLM acceleration, stressing an urgent need for instance-level stability checks to ensure trustworthy real-world deployment.
>
---
#### [replaced 069] On the Generalizability of Foundation Models for Crop Type Mapping
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.09451v4](http://arxiv.org/pdf/2409.09451v4)**

> **作者:** Yi-Chia Chang; Adam J. Stewart; Favyen Bastani; Piper Wolters; Shreya Kannan; George R. Huber; Jingtong Wang; Arindam Banerjee
>
> **备注:** Accepted to IEEE IGARSS 2025. The final version will appear in the Proceedings of the IEEE International Geoscience and Remote Sensing Symposium (IGARSS) 2025
>
> **摘要:** Foundation models pre-trained using self-supervised learning have shown powerful transfer learning capabilities on various downstream tasks, including language understanding, text generation, and image recognition. The Earth observation (EO) field has produced several foundation models pre-trained directly on multispectral satellite imagery for applications like precision agriculture, wildfire and drought monitoring, and natural disaster response. However, few studies have investigated the ability of these models to generalize to new geographic locations, and potential concerns of geospatial bias -- models trained on data-rich developed nations not transferring well to data-scarce developing nations -- remain. We evaluate three popular EO foundation models, SSL4EO-S12, SatlasPretrain, and ImageNet, on five crop classification datasets across five continents. Results show that pre-trained weights designed explicitly for Sentinel-2, such as SSL4EO-S12, outperform general pre-trained weights like ImageNet. While only 100 labeled images are sufficient for achieving high overall accuracy, 900 images are required to mitigate class imbalance and improve average accuracy.
>
---
#### [replaced 070] Online Iterative Self-Alignment for Radiology Report Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11983v2](http://arxiv.org/pdf/2505.11983v2)**

> **作者:** Ting Xiao; Lei Shi; Yang Zhang; HaoFeng Yang; Zhe Wang; Chenjia Bai
>
> **备注:** Accepted by ACL 2025 Main
>
> **摘要:** Radiology Report Generation (RRG) is an important research topic for relieving radiologist' heavy workload. Existing RRG models mainly rely on supervised fine-tuning (SFT) based on different model architectures using data pairs of radiological images and corresponding radiologist-annotated reports. Recent research has shifted focus to post-training improvements, aligning RRG model outputs with human preferences using reinforcement learning (RL). However, the limited data coverage of high-quality annotated data poses risks of overfitting and generalization. This paper proposes a novel Online Iterative Self-Alignment (OISA) method for RRG that consists of four stages: self-generation of diverse data, self-evaluation for multi-objective preference data,self-alignment for multi-objective optimization and self-iteration for further improvement. Our approach allows for generating varied reports tailored to specific clinical objectives, enhancing the overall performance of the RRG model iteratively. Unlike existing methods, our frame-work significantly increases data quality and optimizes performance through iterative multi-objective optimization. Experimental results demonstrate that our method surpasses previous approaches, achieving state-of-the-art performance across multiple evaluation metrics.
>
---
#### [replaced 071] Evaluating the Correctness of Inference Patterns Used by LLMs for Judgment
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.09083v2](http://arxiv.org/pdf/2410.09083v2)**

> **作者:** Lu Chen; Yuxuan Huang; Yixing Li; Dongrui Liu; Qihan Ren; Shuai Zhao; Kun Kuang; Zilong Zheng; Quanshi Zhang
>
> **摘要:** This paper presents a method to analyze the inference patterns used by Large Language Models (LLMs) for judgment in a case study on legal LLMs, so as to identify potential incorrect representations of the LLM, according to human domain knowledge. Unlike traditional evaluations on language generation results, we propose to evaluate the correctness of the detailed inference patterns of an LLM behind its seemingly correct outputs. To this end, we quantify the interactions between input phrases used by the LLM as primitive inference patterns, because recent theoretical achievements have proven several mathematical guarantees of the faithfulness of the interaction-based explanation. We design a set of metrics to evaluate the detailed inference patterns of LLMs. Experiments show that even when the language generation results appear correct, a significant portion of the inference patterns used by the LLM for the legal judgment may represent misleading or irrelevant logic.
>
---
#### [replaced 072] SG-Reg: Generalizable and Efficient Scene Graph Registration
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.14440v2](http://arxiv.org/pdf/2504.14440v2)**

> **作者:** Chuhao Liu; Zhijian Qiao; Jieqi Shi; Ke Wang; Peize Liu; Shaojie Shen
>
> **备注:** IEEE Transactions Robotics Regular Paper
>
> **摘要:** This paper addresses the challenges of registering two rigid semantic scene graphs, an essential capability when an autonomous agent needs to register its map against a remote agent, or against a prior map. The hand-crafted descriptors in classical semantic-aided registration, or the ground-truth annotation reliance in learning-based scene graph registration, impede their application in practical real-world environments. To address the challenges, we design a scene graph network to encode multiple modalities of semantic nodes: open-set semantic feature, local topology with spatial awareness, and shape feature. These modalities are fused to create compact semantic node features. The matching layers then search for correspondences in a coarse-to-fine manner. In the back-end, we employ a robust pose estimator to decide transformation according to the correspondences. We manage to maintain a sparse and hierarchical scene representation. Our approach demands fewer GPU resources and fewer communication bandwidth in multi-agent tasks. Moreover, we design a new data generation approach using vision foundation models and a semantic mapping module to reconstruct semantic scene graphs. It differs significantly from previous works, which rely on ground-truth semantic annotations to generate data. We validate our method in a two-agent SLAM benchmark. It significantly outperforms the hand-crafted baseline in terms of registration success rate. Compared to visual loop closure networks, our method achieves a slightly higher registration recall while requiring only 52 KB of communication bandwidth for each query frame. Code available at: \href{http://github.com/HKUST-Aerial-Robotics/SG-Reg}{http://github.com/HKUST-Aerial-Robotics/SG-Reg}.
>
---
#### [replaced 073] Swin DiT: Diffusion Transformer using Pseudo Shifted Windows
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13219v2](http://arxiv.org/pdf/2505.13219v2)**

> **作者:** Jiafu Wu; Yabiao Wang; Jian Li; Jinlong Peng; Yun Cao; Chengjie Wang; Jiangning Zhang
>
> **摘要:** Diffusion Transformers (DiTs) achieve remarkable performance within the domain of image generation through the incorporation of the transformer architecture. Conventionally, DiTs are constructed by stacking serial isotropic global information modeling transformers, which face significant computational cost when processing high-resolution images. We empirically analyze that latent space image generation does not exhibit a strong dependence on global information as traditionally assumed. Most of the layers in the model demonstrate redundancy in global computation. In addition, conventional attention mechanisms exhibit low-frequency inertia issues. To address these issues, we propose \textbf{P}seudo \textbf{S}hifted \textbf{W}indow \textbf{A}ttention (PSWA), which fundamentally mitigates global model redundancy. PSWA achieves intermediate global-local information interaction through window attention, while employing a high-frequency bridging branch to simulate shifted window operations, supplementing appropriate global and high-frequency information. Furthermore, we propose the Progressive Coverage Channel Allocation(PCCA) strategy that captures high-order attention similarity without additional computational cost. Building upon all of them, we propose a series of Pseudo \textbf{S}hifted \textbf{Win}dow DiTs (\textbf{Swin DiT}), accompanied by extensive experiments demonstrating their superior performance. For example, our proposed Swin-DiT-L achieves a 54%$\uparrow$ FID improvement over DiT-XL/2 while requiring less computational. https://github.com/wujiafu007/Swin-DiT
>
---
#### [replaced 074] Rethinking Text-Promptable Surgical Instrument Segmentation with Robust Framework
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.12199v3](http://arxiv.org/pdf/2411.12199v3)**

> **作者:** Tae-Min Choi; Juyoun Park
>
> **备注:** 15 pages, 5 figures, 8 tables
>
> **摘要:** Surgical instrument segmentation is an essential component of computer-assisted and robotic surgery systems. Vision-based segmentation models typically produce outputs limited to a predefined set of instrument categories, which restricts their applicability in interactive systems and robotic task automation. Promptable segmentation methods allow selective predictions based on textual prompts. However, they often rely on the assumption that the instruments present in the scene are already known, and prompts are generated accordingly, limiting their ability to generalize to unseen or dynamically emerging instruments. In practical surgical environments, where instrument existence information is not provided, this assumption does not hold consistently, resulting in false-positive segmentation. To address these limitations, we formulate a new task called Robust text-promptable Surgical Instrument Segmentation (R-SIS). Under this setting, prompts are issued for all candidate categories without access to instrument presence information. R-SIS requires distinguishing which prompts refer to visible instruments and generating masks only when such instruments are explicitly present in the scene. This setting reflects practical conditions where uncertainty in instrument presence is inherent. We evaluate existing segmentation methods under the R-SIS protocol using surgical video datasets and observe substantial false-positive predictions in the absence of ground-truth instruments. These findings demonstrate a mismatch between current evaluation protocols and real-world use cases, and support the need for benchmarks that explicitly account for prompt uncertainty and instrument absence.
>
---
#### [replaced 075] Pyramid Sparse Transformer: Enhancing Multi-Scale Feature Fusion with Dynamic Token Selection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12772v2](http://arxiv.org/pdf/2505.12772v2)**

> **作者:** Junyi Hu; Tian Bai; Fengyi Wu; Zhenming Peng; Yi Zhang
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Feature fusion is critical for high-performance vision models but often incurs prohibitive complexity. However, prevailing attention-based fusion methods often involve significant computational complexity and implementation challenges, limiting their efficiency in resource-constrained environments. To address these issues, we introduce the Pyramid Sparse Transformer (PST), a lightweight, plug-and-play module that integrates coarse-to-fine token selection and shared attention parameters to reduce computation while preserving spatial detail. PST can be trained using only coarse attention and seamlessly activated at inference for further accuracy gains without retraining. When added to state-of-the-art real-time detection models, such as YOLOv11-N/S/M, PST yields mAP improvements of 0.9%, 0.5%, and 0.4% on MS COCO with minimal latency impact. Likewise, embedding PST into ResNet-18/50/101 as backbones, boosts ImageNet top-1 accuracy by 6.5%, 1.7%, and 1.0%, respectively. These results demonstrate PST's effectiveness as a simple, hardware-friendly enhancement for both detection and classification tasks.
>
---
#### [replaced 076] FastMap: Revisiting Dense and Scalable Structure from Motion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.04612v2](http://arxiv.org/pdf/2505.04612v2)**

> **作者:** Jiahao Li; Haochen Wang; Muhammad Zubair Irshad; Igor Vasiljevic; Matthew R. Walter; Vitor Campagnolo Guizilini; Greg Shakhnarovich
>
> **备注:** Project webpage: https://jiahao.ai/fastmap
>
> **摘要:** We propose FastMap, a new global structure from motion method focused on speed and simplicity. Previous methods like COLMAP and GLOMAP are able to estimate high-precision camera poses, but suffer from poor scalability when the number of matched keypoint pairs becomes large. We identify two key factors leading to this problem: poor parallelization and computationally expensive optimization steps. To overcome these issues, we design an SfM framework that relies entirely on GPU-friendly operations, making it easily parallelizable. Moreover, each optimization step runs in time linear to the number of image pairs, independent of keypoint pairs or 3D points. Through extensive experiments, we show that FastMap is faster than COLMAP and GLOMAP on large-scale scenes with comparable pose accuracy.
>
---
#### [replaced 077] Cross-Image Contrastive Decoding: Precise, Lossless Suppression of Language Priors in Large Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10634v3](http://arxiv.org/pdf/2505.10634v3)**

> **作者:** Jianfei Zhao; Feng Zhang; Xin Sun; Chong Feng
>
> **摘要:** Language priors are a major cause of hallucinations in Large Vision-Language Models (LVLMs), often leading to text that is linguistically plausible but visually inconsistent. Recent work explores contrastive decoding as a training-free solution, but these methods typically construct negative contexts from the original image, resulting in visual information loss and distorted distribution. Motivated by the observation that language priors stem from the LLM backbone and remain consistent across images, we propose Cross-Images Contrastive Decoding (CICD), a simple yet effective training-free method that uses different images to construct negative contexts. We further analyze the cross-image behavior of language priors and introduce a distinction between essential priors (supporting fluency) and detrimental priors (causing hallucinations). By selectively preserving essential priors and suppressing detrimental ones, our method reduces hallucinations while maintaining coherent and fluent language generation. Experiments on 4 benchmarks and 6 LVLMs across three model families confirm the effectiveness and generalizability of CICD, especially in image captioning, where language priors are particularly pronounced. Code will be released once accepted.
>
---
#### [replaced 078] LED: LLM Enhanced Open-Vocabulary Object Detection without Human Curated Data Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.13794v2](http://arxiv.org/pdf/2503.13794v2)**

> **作者:** Yang Zhou; Shiyu Zhao; Yuxiao Chen; Zhenting Wang; Can Jin; Dimitris N. Metaxas
>
> **摘要:** Large foundation models trained on large-scale vision-language data can boost Open-Vocabulary Object Detection (OVD) via synthetic training data, yet the hand-crafted pipelines often introduce bias and overfit to specific prompts. We sidestep this issue by directly fusing hidden states from Large Language Models (LLMs) into detectors-an avenue surprisingly under-explored. This paper presents a systematic method to enhance visual grounding by utilizing decoder layers of the LLM of an MLLM. We introduce a zero-initialized cross-attention adapter to enable efficient knowledge fusion from LLMs to object detectors, a new approach called LED (LLM Enhanced Open-Vocabulary Object Detection). We find that intermediate LLM layers already encode rich spatial semantics; adapting only the early layers yields most of the gain. With Swin-T as the vision encoder, Qwen2-0.5B + LED lifts GroundingDINO by 3.82 % on OmniLabel at just 8.7 % extra GFLOPs, and a larger vision backbone pushes the improvement to 6.22 %. Extensive ablations on adapter variants, LLM scales and fusion depths further corroborate our design.
>
---
#### [replaced 079] StainDiffuser: MultiTask Dual Diffusion Model for Virtual Staining
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2403.11340v2](http://arxiv.org/pdf/2403.11340v2)**

> **作者:** Tushar Kataria; Beatrice Knudsen; Shireen Y. Elhabian
>
> **摘要:** Hematoxylin and Eosin (H&E) staining is widely regarded as the standard in pathology for diagnosing diseases and tracking tumor recurrence. While H&E staining shows tissue structures, it lacks the ability to reveal specific proteins that are associated with disease severity and treatment response. Immunohistochemical (IHC) stains use antibodies to highlight the expression of these proteins on their respective cell types, improving diagnostic accuracy, and assisting with drug selection for treatment. Despite their value, IHC stains require additional time and resources, limiting their utilization in some clinical settings. Recent advances in deep learning have positioned Image-to-Image (I2I) translation as a computational, cost-effective alternative for IHC. I2I generates high fidelity stain transformations digitally, potentially replacing manual staining in IHC. Diffusion models, the current state of the art in image generation and conditional tasks, are particularly well suited for virtual IHC due to their ability to produce high quality images and resilience to mode collapse. However, these models require extensive and diverse datasets (often millions of samples) to achieve a robust performance, a challenge in virtual staining applications where only thousands of samples are typically available. Inspired by the success of multitask deep learning models in scenarios with limited data, we introduce STAINDIFFUSER, a novel multitask diffusion architecture tailored to virtual staining that achieves convergence with smaller datasets. STAINDIFFUSER simultaneously trains two diffusion processes: (a) generating cell specific IHC stains from H&E images and (b) performing H&E based cell segmentation, utilizing coarse segmentation labels exclusively during training. STAINDIFFUSER generates high-quality virtual stains for two markers, outperforming over twenty I2I baselines.
>
---
#### [replaced 080] Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2403.11083v3](http://arxiv.org/pdf/2403.11083v3)**

> **作者:** Xiaohao Xu; Yunkang Cao; Huaxin Zhang; Nong Sang; Xiaonan Huang
>
> **备注:** Best Student Paper Award at IEEE International Conference on Computer Supported Cooperative Work in Design, 2025
>
> **摘要:** Anomaly detection is vital in various industrial scenarios, including the identification of unusual patterns in production lines and the detection of manufacturing defects for quality control. Existing techniques tend to be specialized in individual scenarios and lack generalization capacities. In this study, our objective is to develop a generic anomaly detection model that can be applied in multiple scenarios. To achieve this, we custom-build generic visual language foundation models that possess extensive knowledge and robust reasoning abilities as anomaly detectors and reasoners. Specifically, we introduce a multi-modal prompting strategy that incorporates domain knowledge from experts as conditions to guide the models. Our approach considers diverse prompt types, including task descriptions, class context, normality rules, and reference images. In addition, we unify the input representation of multi-modality into a 2D image format, enabling multi-modal anomaly detection and reasoning. Our preliminary studies demonstrate that combining visual and language prompts as conditions for customizing the models enhances anomaly detection performance. The customized models showcase the ability to detect anomalies across different data modalities such as images, point clouds, and videos. Qualitative case studies further highlight the anomaly detection and reasoning capabilities, particularly for multi-object scenes and temporal data. Our code is publicly available at https://github.com/Xiaohao-Xu/Customizable-VLM
>
---
#### [replaced 081] Continual Distillation Learning: Knowledge Distillation in Prompt-based Continual Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.13911v4](http://arxiv.org/pdf/2407.13911v4)**

> **作者:** Qifan Zhang; Yunhui Guo; Yu Xiang
>
> **摘要:** We introduce the problem of continual distillation learning (CDL) in order to use knowledge distillation (KD) to improve prompt-based continual learning (CL) models. The CDL problem is valuable to study since the use of a larger vision transformer (ViT) leads to better performance in prompt-based continual learning. The distillation of knowledge from a large ViT to a small ViT improves the inference efficiency for prompt-based CL models. We empirically found that existing KD methods such as logit distillation and feature distillation cannot effectively improve the student model in the CDL setup. To address this issue, we introduce a novel method named Knowledge Distillation based on Prompts (KDP), in which globally accessible prompts specifically designed for knowledge distillation are inserted into the frozen ViT backbone of the student model. We demonstrate that our KDP method effectively enhances the distillation performance in comparison to existing KD methods in the CDL setup.
>
---
#### [replaced 082] Semantics-Oriented Multitask Learning for DeepFake Detection: A Joint Embedding Approach
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.16305v2](http://arxiv.org/pdf/2408.16305v2)**

> **作者:** Mian Zou; Baosheng Yu; Yibing Zhan; Siwei Lyu; Kede Ma
>
> **摘要:** In recent years, the multimedia forensics and security community has seen remarkable progress in multitask learning for DeepFake (i.e., face forgery) detection. The prevailing approach has been to frame DeepFake detection as a binary classification problem augmented by manipulation-oriented auxiliary tasks. This scheme focuses on learning features specific to face manipulations with limited generalizability. In this paper, we delve deeper into semantics-oriented multitask learning for DeepFake detection, capturing the relationships among face semantics via joint embedding. We first propose an automated dataset expansion technique that broadens current face forgery datasets to support semantics-oriented DeepFake detection tasks at both the global face attribute and local face region levels. Furthermore, we resort to the joint embedding of face images and labels (depicted by text descriptions) for prediction. This approach eliminates the need for manually setting task-agnostic and task-specific parameters, which is typically required when predicting multiple labels directly from images. In addition, we employ bi-level optimization to dynamically balance the fidelity loss weightings of various tasks, making the training process fully automated. Extensive experiments on six DeepFake datasets show that our method improves the generalizability of DeepFake detection and renders some degree of model interpretation by providing human-understandable explanations.
>
---
#### [replaced 083] Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.15633v4](http://arxiv.org/pdf/2411.15633v4)**

> **作者:** Zhiyuan Yan; Jiangming Wang; Peng Jin; Ke-Yue Zhang; Chengchun Liu; Shen Chen; Taiping Yao; Shouhong Ding; Baoyuan Wu; Li Yuan
>
> **摘要:** AI-generated images (AIGIs), such as natural or face images, have become increasingly important yet challenging. In this paper, we start from a new perspective to excavate the reason behind the failure generalization in AIGI detection, named the \textit{asymmetry phenomenon}, where a naively trained detector tends to favor overfitting to the limited and monotonous fake patterns, causing the feature space to become highly constrained and low-ranked, which is proved seriously limiting the expressivity and generalization. One potential remedy is incorporating the pre-trained knowledge within the vision foundation models (higher-ranked) to expand the feature space, alleviating the model's overfitting to fake. To this end, we employ Singular Value Decomposition (SVD) to decompose the original feature space into \textit{two orthogonal subspaces}. By freezing the principal components and adapting only the remained components, we preserve the pre-trained knowledge while learning fake patterns. Compared to existing full-parameters and LoRA-based tuning methods, we explicitly ensure orthogonality, enabling the higher rank of the whole feature space, effectively minimizing overfitting and enhancing generalization. We finally identify a crucial insight: our method implicitly learns \textit{a vital prior that fakes are actually derived from the real}, indicating a hierarchical relationship rather than independence. Modeling this prior, we believe, is essential for achieving superior generalization. Our codes are publicly available at \href{https://github.com/YZY-stack/Effort-AIGI-Detection}{GitHub}.
>
---
#### [replaced 084] Multi-granular body modeling with Redundancy-Free Spatiotemporal Fusion for Text-Driven Motion Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06897v2](http://arxiv.org/pdf/2503.06897v2)**

> **作者:** Xingzu Zhan; Chen Xie; Honghang Chen; Haoran Sun; Xiaochun Mai
>
> **备注:** 15pages,5figures,
>
> **摘要:** Text-to-motion generation sits at the intersection of multimodal learning and computer graphics and is gaining momentum because it can simplify content creation for games, animation, robotics and virtual reality. Most current methods stack spatial and temporal features in a straightforward way, which adds redundancy and still misses subtle joint-level cues. We introduce HiSTF Mamba, a framework with three parts: Dual-Spatial Mamba, Bi-Temporal Mamba and a Dynamic Spatiotemporal Fusion Module (DSFM). The Dual-Spatial module runs part-based and whole-body models in parallel, capturing both overall coordination and fine-grained joint motion. The Bi-Temporal module scans sequences forward and backward to encode short-term details and long-term dependencies. DSFM removes redundant temporal information, extracts complementary cues and fuses them with spatial features to build a richer spatiotemporal representation. Experiments on the HumanML3D benchmark show that HiSTF Mamba performs well across several metrics, achieving high fidelity and tight semantic alignment between text and motion.
>
---
#### [replaced 085] Iterative Tool Usage Exploration for Multimodal Agents via Step-wise Preference Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.21561v3](http://arxiv.org/pdf/2504.21561v3)**

> **作者:** Pengxiang Li; Zhi Gao; Bofei Zhang; Yapeng Mi; Xiaojian Ma; Chenrui Shi; Tao Yuan; Yuwei Wu; Yunde Jia; Song-Chun Zhu; Qing Li
>
> **备注:** 24 pages
>
> **摘要:** Multimodal agents, which integrate a controller e.g., a vision language model) with external tools, have demonstrated remarkable capabilities in tackling complex multimodal tasks. Existing approaches for training these agents, both supervised fine-tuning and reinforcement learning, depend on extensive human-annotated task-answer pairs and tool trajectories. However, for complex multimodal tasks, such annotations are prohibitively expensive or impractical to obtain. In this paper, we propose an iterative tool usage exploration method for multimodal agents without any pre-collected data, namely SPORT, via step-wise preference optimization to refine the trajectories of tool usage. Our method enables multimodal agents to autonomously discover effective tool usage strategies through self-exploration and optimization, eliminating the bottleneck of human annotation. SPORT has four iterative components: task synthesis, step sampling, step verification, and preference tuning. We first synthesize multimodal tasks using language models. Then, we introduce a novel trajectory exploration scheme, where step sampling and step verification are executed alternately to solve synthesized tasks. In step sampling, the agent tries different tools and obtains corresponding results. In step verification, we employ a verifier to provide AI feedback to construct step-wise preference data. The data is subsequently used to update the controller for tool usage through preference tuning, producing a SPORT agent. By interacting with real environments, the SPORT agent gradually evolves into a more refined and capable system. Evaluation in the GTA and GAIA benchmarks shows that the SPORT agent achieves 6.41% and 3.64% improvements, underscoring the generalization and effectiveness introduced by our method. The project page is https://SPORT-Agents.github.io.
>
---
#### [replaced 086] MTVCrafter: 4D Motion Tokenization for Open-World Human Image Animation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10238v3](http://arxiv.org/pdf/2505.10238v3)**

> **作者:** Yanbo Ding; Xirui Hu; Zhizhi Guo; Yali Wang
>
> **摘要:** Human image animation has gained increasing attention and developed rapidly due to its broad applications in digital humans. However, existing methods rely largely on 2D-rendered pose images for motion guidance, which limits generalization and discards essential 3D information for open-world animation. To tackle this problem, we propose MTVCrafter (Motion Tokenization Video Crafter), the first framework that directly models raw 3D motion sequences (i.e., 4D motion) for human image animation. Specifically, we introduce 4DMoT (4D motion tokenizer) to quantize 3D motion sequences into 4D motion tokens. Compared to 2D-rendered pose images, 4D motion tokens offer more robust spatio-temporal cues and avoid strict pixel-level alignment between pose image and character, enabling more flexible and disentangled control. Then, we introduce MV-DiT (Motion-aware Video DiT). By designing unique motion attention with 4D positional encodings, MV-DiT can effectively leverage motion tokens as 4D compact yet expressive context for human image animation in the complex 3D world. Hence, it marks a significant step forward in this field and opens a new direction for pose-guided human video generation. Experiments show that our MTVCrafter achieves state-of-the-art results with an FID-VID of 6.98, surpassing the second-best by 65%. Powered by robust motion tokens, MTVCrafter also generalizes well to diverse open-world characters (single/multiple, full/half-body) across various styles and scenarios. Our video demos and code are on: https://github.com/DINGYANB/MTVCrafter.
>
---
#### [replaced 087] Any-to-Any Learning in Computational Pathology via Triplet Multimodal Pretraining
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.12711v2](http://arxiv.org/pdf/2505.12711v2)**

> **作者:** Qichen Sun; Zhengrui Guo; Rui Peng; Hao Chen; Jinzhuo Wang
>
> **摘要:** Recent advances in computational pathology and artificial intelligence have significantly enhanced the utilization of gigapixel whole-slide images and and additional modalities (e.g., genomics) for pathological diagnosis. Although deep learning has demonstrated strong potential in pathology, several key challenges persist: (1) fusing heterogeneous data types requires sophisticated strategies beyond simple concatenation due to high computational costs; (2) common scenarios of missing modalities necessitate flexible strategies that allow the model to learn robustly in the absence of certain modalities; (3) the downstream tasks in CPath are diverse, ranging from unimodal to multimodal, cnecessitating a unified model capable of handling all modalities. To address these challenges, we propose ALTER, an any-to-any tri-modal pretraining framework that integrates WSIs, genomics, and pathology reports. The term "any" emphasizes ALTER's modality-adaptive design, enabling flexible pretraining with any subset of modalities, and its capacity to learn robust, cross-modal representations beyond WSI-centric approaches. We evaluate ALTER across extensive clinical tasks including survival prediction, cancer subtyping, gene mutation prediction, and report generation, achieving superior or comparable performance to state-of-the-art baselines.
>
---
#### [replaced 088] 3D Visual Illusion Depth Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13061v2](http://arxiv.org/pdf/2505.13061v2)**

> **作者:** Chengtang Yao; Zhidan Liu; Jiaxi Zeng; Lidong Yu; Yuwei Wu; Yunde Jia
>
> **备注:** Project: https://github.com/YaoChengTang/3D-Visual-Illusion-Depth-Estimation
>
> **摘要:** 3D visual illusion is a perceptual phenomenon where a two-dimensional plane is manipulated to simulate three-dimensional spatial relationships, making a flat artwork or object look three-dimensional in the human visual system. In this paper, we reveal that the machine visual system is also seriously fooled by 3D visual illusions, including monocular and binocular depth estimation. In order to explore and analyze the impact of 3D visual illusion on depth estimation, we collect a large dataset containing almost 3k scenes and 200k images to train and evaluate SOTA monocular and binocular depth estimation methods. We also propose a robust depth estimation framework that uses common sense from a vision-language model to adaptively select reliable depth from binocular disparity and monocular depth. Experiments show that SOTA monocular, binocular, and multi-view depth estimation approaches are all fooled by various 3D visual illusions, while our method achieves SOTA performance.
>
---
#### [replaced 089] Unified Continuous Generative Models
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07447v2](http://arxiv.org/pdf/2505.07447v2)**

> **作者:** Peng Sun; Yi Jiang; Tao Lin
>
> **备注:** https://github.com/LINs-lab/UCGM
>
> **摘要:** Recent advances in continuous generative models, including multi-step approaches like diffusion and flow-matching (typically requiring 8-1000 sampling steps) and few-step methods such as consistency models (typically 1-8 steps), have demonstrated impressive generative performance. However, existing work often treats these approaches as distinct paradigms, resulting in separate training and sampling methodologies. We introduce a unified framework for training, sampling, and analyzing these models. Our implementation, the Unified Continuous Generative Models Trainer and Sampler (UCGM-{T,S}), achieves state-of-the-art (SOTA) performance. For example, on ImageNet 256x256 using a 675M diffusion transformer, UCGM-T trains a multi-step model achieving 1.30 FID in 20 steps and a few-step model reaching 1.42 FID in just 2 steps. Additionally, applying UCGM-S to a pre-trained model (previously 1.26 FID at 250 steps) improves performance to 1.06 FID in only 40 steps. Code is available at: https://github.com/LINs-lab/UCGM.
>
---
#### [replaced 090] Multi-modal Collaborative Optimization and Expansion Network for Event-assisted Single-eye Expression Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12007v2](http://arxiv.org/pdf/2505.12007v2)**

> **作者:** Runduo Han; Xiuping Liu; Shangxuan Yi; Yi Zhang; Hongchen Tan
>
> **摘要:** In this paper, we proposed a Multi-modal Collaborative Optimization and Expansion Network (MCO-E Net), to use event modalities to resist challenges such as low light, high exposure, and high dynamic range in single-eye expression recognition tasks. The MCO-E Net introduces two innovative designs: Multi-modal Collaborative Optimization Mamba (MCO-Mamba) and Heterogeneous Collaborative and Expansion Mixture-of-Experts (HCE-MoE). MCO-Mamba, building upon Mamba, leverages dual-modal information to jointly optimize the model, facilitating collaborative interaction and fusion of modal semantics. This approach encourages the model to balance the learning of both modalities and harness their respective strengths. HCE-MoE, on the other hand, employs a dynamic routing mechanism to distribute structurally varied experts (deep, attention, and focal), fostering collaborative learning of complementary semantics. This heterogeneous architecture systematically integrates diverse feature extraction paradigms to comprehensively capture expression semantics. Extensive experiments demonstrate that our proposed network achieves competitive performance in the task of single-eye expression recognition, especially under poor lighting conditions.
>
---
#### [replaced 091] RadCLIP: Enhancing Radiologic Image Analysis through Contrastive Language-Image Pre-training
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.09948v3](http://arxiv.org/pdf/2403.09948v3)**

> **作者:** Zhixiu Lu; Hailong Li; Nehal A. Parikh; Jonathan R. Dillman; Lili He
>
> **摘要:** The integration of artificial intelligence (AI) with radiology marks a transformative era in medicine. Vision foundation models have been adopted to enhance radiologic imaging analysis. However, the distinct complexities of radiologic 2D and 3D radiologic data pose unique challenges that existing models, pre-trained on general non-medical images, fail to address adequately. To bridge this gap and capitalize on the diagnostic precision required in radiologic imaging, we introduce Radiologic Contrastive Language-Image Pre-training (RadCLIP): a cross-modal vision-language foundational model that harnesses Vision Language Pre-training (VLP) framework to improve radiologic image analysis. Building upon Contrastive Language-Image Pre-training (CLIP), RadCLIP incorporates a slice pooling mechanism tailored for volumetric image analysis and is pre-trained using a large and diverse dataset of radiologic image-text pairs. The RadCLIP was pre-trained to effectively align radiologic images with their corresponding text annotations, creating a robust vision backbone for radiologic images. Extensive experiments demonstrate RadCLIP's superior performance in both uni-modal radiologic image classification and cross-modal image-text matching, highlighting its significant promise for improving diagnostic accuracy and efficiency in clinical settings. Our Key contributions include curating a large dataset with diverse radiologic 2D/3D radiologic image-text pairs, a slice pooling adapter using an attention mechanism for integrating 2D images, and comprehensive evaluations of RadCLIP on various radiologic downstream tasks.
>
---
#### [replaced 092] RoCoDA: Counterfactual Data Augmentation for Data-Efficient Robot Learning from Demonstrations
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.16959v2](http://arxiv.org/pdf/2411.16959v2)**

> **作者:** Ezra Ameperosa; Jeremy A. Collins; Mrinal Jain; Animesh Garg
>
> **备注:** Accepted to 2025 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Imitation learning in robotics faces significant challenges in generalization due to the complexity of robotic environments and the high cost of data collection. We introduce RoCoDA, a novel method that unifies the concepts of invariance, equivariance, and causality within a single framework to enhance data augmentation for imitation learning. RoCoDA leverages causal invariance by modifying task-irrelevant subsets of the environment state without affecting the policy's output. Simultaneously, we exploit SE(3) equivariance by applying rigid body transformations to object poses and adjusting corresponding actions to generate synthetic demonstrations. We validate RoCoDA through extensive experiments on five robotic manipulation tasks, demonstrating improvements in policy performance, generalization, and sample efficiency compared to state-of-the-art data augmentation methods. Our policies exhibit robust generalization to unseen object poses, textures, and the presence of distractors. Furthermore, we observe emergent behavior such as re-grasping, indicating policies trained with RoCoDA possess a deeper understanding of task dynamics. By leveraging invariance, equivariance, and causality, RoCoDA provides a principled approach to data augmentation in imitation learning, bridging the gap between geometric symmetries and causal reasoning. Project Page: https://rocoda.github.io
>
---
