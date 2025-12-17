# 自然语言处理 cs.CL

- **最新发布 47 篇**

- **更新 41 篇**

## 最新发布

#### [new 001] Dual Language Models: Balancing Training Efficiency and Overfitting Resilience
- **分类: cs.CL; cs.AI**

- **简介: 该论文属语言模型训练方法研究，旨在解决单目标训练中效率与过拟合鲁棒性难以兼顾的问题。作者提出双目标训练框架，融合自回归与掩码扩散目标，无需架构改动；通过50组实验确定最优目标配比，验证其在各类数据重复场景下均优于单目标模型。**

- **链接: [https://arxiv.org/pdf/2512.14549v1](https://arxiv.org/pdf/2512.14549v1)**

> **作者:** David Samuel; Lucas Georges Gabriel Charpentier
>
> **摘要:** This paper combines autoregressive and masked-diffusion training objectives without any architectural modifications, resulting in flexible language models that outperform single-objective models. Autoregressive modeling has been a popular approach, partly because of its training efficiency; however, that comes at the cost of sensitivity to overfitting. On the other hand, masked-diffusion models are less efficient to train while being more resilient to overfitting. In this work, we demonstrate that dual-objective training achieves the best of both worlds. To derive the optimal ratio between both objectives, we train and evaluate 50 language models under varying levels of data repetition. We show that it is optimal to combine both objectives under all evaluated settings and that the optimal ratio is similar whether targeting autoregressive or masked-diffusion downstream performance.
>
---
#### [new 002] TiME: Tiny Monolingual Encoders for Efficient NLP Pipelines
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出TiME（Tiny Monolingual Encoders），面向效率关键型NLP任务，解决大模型在吞吐、延迟、能耗和低资源语言支持上的不足。通过知识蒸馏等现代训练技术，构建轻量单语编码器，在性能与效率间取得更好权衡。**

- **链接: [https://arxiv.org/pdf/2512.14645v1](https://arxiv.org/pdf/2512.14645v1)**

> **作者:** David Schulmeister; Valentin Hartmann; Lars Klein; Robert West
>
> **摘要:** Today, a lot of research on language models is focused on large, general-purpose models. However, many NLP pipelines only require models with a well-defined, small set of capabilities. While large models are capable of performing the tasks of those smaller models, they are simply not fast enough to process large amounts of data or offer real-time responses. Furthermore, they often use unnecessarily large amounts of energy, leading to sustainability concerns and problems when deploying them on battery-powered devices. In our work, we show how to train small models for such efficiency-critical applications. As opposed to many off-the-shelf NLP pipelines, our models use modern training techniques such as distillation, and offer support for low-resource languages. We call our models TiME (Tiny Monolingual Encoders) and comprehensively evaluate them on a range of common NLP tasks, observing an improved trade-off between benchmark performance on one hand, and throughput, latency and energy consumption on the other. Along the way, we show that distilling monolingual models from multilingual teachers is possible, and likewise distilling models with absolute positional embeddings from teachers with relative positional embeddings.
>
---
#### [new 003] Two CFG Nahuatl for automatic corpora expansion
- **分类: cs.CL**

- **简介: 该论文属低资源语言Nawatl的语料扩充任务，旨在解决其数字资源匮乏、缺乏LLM训练语料的问题。作者构建两个上下文无关文法（CFG），生成大量合法人工句子以扩展语料，用于学习非上下文嵌入，并在语义相似度任务中验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.14239v1](https://arxiv.org/pdf/2512.14239v1)**

> **作者:** Juan-José Guzmán-Landa; Juan-Manuel Torres-Moreno; Miguel Figueroa-Saavedra; Ligia Quintana-Torres; Graham Ranger Martha-Lorena Avendaño-Garrido
>
> **备注:** 15 pages, 5 figures, 8 tables
>
> **摘要:** The aim of this article is to introduce two Context-Free Grammars (CFG) for Nawatl Corpora expansion. Nawatl is an Amerindian language (it is a National Language of Mexico) of the $π$-language type, i.e. a language with few digital resources. For this reason the corpora available for the learning of Large Language Models (LLMs) are virtually non-existent, posing a significant challenge. The goal is to produce a substantial number of syntactically valid artificial Nawatl sentences and thereby to expand the corpora for the purpose of learning non contextual embeddings. For this objective, we introduce two new Nawatl CFGs and use them in generative mode. Using these grammars, it is possible to expand Nawatl corpus significantly and subsequently to use it to learn embeddings and to evaluate their relevance in a sentences semantic similarity task. The results show an improvement compared to the results obtained using only the original corpus without artificial expansion, and also demonstrate that economic embeddings often perform better than some LLMs.
>
---
#### [new 004] Step-Tagging: Toward controlling the generation of Language Reasoning Models through step monitoring
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Step-Tagging框架，属语言推理模型（LRM）生成控制任务，旨在解决LRM过量生成冗余推理步骤的问题。通过轻量级句子分类器实时标注推理步类型（ReasonType），实现基于步数统计的可解释早停，显著减少token消耗（20–50%）且不损精度。**

- **链接: [https://arxiv.org/pdf/2512.14332v1](https://arxiv.org/pdf/2512.14332v1)**

> **作者:** Yannis Belkhiter; Seshu Tirupathi; Giulio Zizzo; John D. Kelleher
>
> **摘要:** The field of Language Reasoning Models (LRMs) has been very active over the past few years with advances in training and inference techniques enabling LRMs to reason longer, and more accurately. However, a growing body of studies show that LRMs are still inefficient, over-generating verification and reflection steps. To address this challenge, we introduce the Step-Tagging framework, a lightweight sentence-classifier enabling real-time annotation of the type of reasoning steps that an LRM is generating. To monitor reasoning behaviors, we introduced ReasonType: a novel taxonomy of reasoning steps. Building on this framework, we demonstrated that online monitoring of the count of specific steps can produce effective interpretable early stopping criteria of LRM inferences. We evaluate the Step-tagging framework on three open-source reasoning models across standard benchmark datasets: MATH500, GSM8K, AIME and non-mathematical tasks (GPQA and MMLU-Pro). We achieve 20 to 50\% token reduction while maintaining comparable accuracy to standard generation, with largest gains observed on more computation-heavy tasks. This work offers a novel way to increase control over the generation of LRMs, and a new tool to study behaviors of LRMs.
>
---
#### [new 005] VersatileFFN: Achieving Parameter Efficiency in LLMs via Adaptive Wide-and-Deep Reuse
- **分类: cs.CL**

- **简介: 该论文属大模型参数高效优化任务，旨在缓解LLM内存开销大、表征能力受限问题。提出VersatileFFN：通过宽（子专家混合）深（递归复用）双路径自适应重用同一组FFN参数，以计算换容量，实现难度感知的动态路由。**

- **链接: [https://arxiv.org/pdf/2512.14531v1](https://arxiv.org/pdf/2512.14531v1)**

> **作者:** Ying Nie; Kai Han; Hongguang Li; Hang Zhou; Tianyu Guo; Enhua Wu; Xinghao Chen; Yunhe Wang
>
> **摘要:** The rapid scaling of Large Language Models (LLMs) has achieved remarkable performance, but it also leads to prohibitive memory costs. Existing parameter-efficient approaches such as pruning and quantization mainly compress pretrained models without enhancing architectural capacity, thereby hitting the representational ceiling of the base model. In this work, we propose VersatileFFN, a novel feed-forward network (FFN) that enables flexible reuse of parameters in both width and depth dimensions within a fixed parameter budget. Inspired by the dual-process theory of cognition, VersatileFFN comprises two adaptive pathways: a width-versatile path that generates a mixture of sub-experts from a single shared FFN, mimicking sparse expert routing without increasing parameters, and a depth-versatile path that recursively applies the same FFN to emulate deeper processing for complex tokens. A difficulty-aware gating dynamically balances the two pathways, steering "easy" tokens through the efficient width-wise route and allocating deeper iterative refinement to "hard" tokens. Crucially, both pathways reuse the same parameters, so all additional capacity comes from computation rather than memory. Experiments across diverse benchmarks and model scales demonstrate the effectiveness of the method. The code will be available at https://github.com/huawei-noah/noah-research/tree/master/VersatileFFN.
>
---
#### [new 006] Fast and Accurate Causal Parallel Decoding using Jacobi Forcing
- **分类: cs.CL**

- **简介: 该论文属大模型推理加速任务，旨在解决并行解码中质量与速度难以兼顾的问题。提出Jacobi Forcing蒸馏范式，使AR模型平滑转向高效并行解码，保持因果性；并设计多块解码与拒绝回收策略，实现近4倍加速与低性能损失。**

- **链接: [https://arxiv.org/pdf/2512.14681v1](https://arxiv.org/pdf/2512.14681v1)**

> **作者:** Lanxiang Hu; Siqi Kou; Yichao Fu; Samyam Rajbhandari; Tajana Rosing; Yuxiong He; Zhijie Deng; Hao Zhang
>
> **摘要:** Multi-token generation has emerged as a promising paradigm for accelerating transformer-based large model inference. Recent efforts primarily explore diffusion Large Language Models (dLLMs) for parallel decoding to reduce inference latency. To achieve AR-level generation quality, many techniques adapt AR models into dLLMs to enable parallel decoding. However, they suffer from limited speedup compared to AR models due to a pretrain-to-posttrain mismatch. Specifically, the masked data distribution in post-training deviates significantly from the real-world data distribution seen during pretraining, and dLLMs rely on bidirectional attention, which conflicts with the causal prior learned during pretraining and hinders the integration of exact KV cache reuse. To address this, we introduce Jacobi Forcing, a progressive distillation paradigm where models are trained on their own generated parallel decoding trajectories, smoothly shifting AR models into efficient parallel decoders while preserving their pretrained causal inference property. The models trained under this paradigm, Jacobi Forcing Model, achieves 3.8x wall-clock speedup on coding and math benchmarks with minimal loss in performance. Based on Jacobi Forcing Models' trajectory characteristics, we introduce multi-block decoding with rejection recycling, which enables up to 4.5x higher token acceptance count per iteration and nearly 4.0x wall-clock speedup, effectively trading additional compute for lower inference latency. Our code is available at https://github.com/hao-ai-lab/JacobiForcing.
>
---
#### [new 007] From Context to EDUs: Faithful and Structured Context Compression via Elementary Discourse Unit Decomposition
- **分类: cs.CL; cs.AI**

- **简介: 该论文属自然语言处理中的上下文压缩任务，旨在解决LLM长文本输入的高成本与噪声问题。提出EDU-based压缩框架：先用LingoEDU将文本显式分解为源索引锚定的EDU关系树，再轻量排序选取相关子树线性化。引入StructBench评估，显著提升结构理解与下游任务性能。**

- **链接: [https://arxiv.org/pdf/2512.14244v1](https://arxiv.org/pdf/2512.14244v1)**

> **作者:** Yiqing Zhou; Yu Lei; Shuzheng Si; Qingyan Sun; Wei Wang; Yifei Wu; Hao Wen; Gang Chen; Fanchao Qi; Maosong Sun
>
> **摘要:** Managing extensive context remains a critical bottleneck for Large Language Models (LLMs), particularly in applications like long-document question answering and autonomous agents where lengthy inputs incur high computational costs and introduce noise. Existing compression techniques often disrupt local coherence through discrete token removal or rely on implicit latent encoding that suffers from positional bias and incompatibility with closed-source APIs. To address these limitations, we introduce the EDU-based Context Compressor, a novel explicit compression framework designed to preserve both global structure and fine-grained details. Our approach reformulates context compression as a structure-then-select process. First, our LingoEDU transforms linear text into a structural relation tree of Elementary Discourse Units (EDUs) which are anchored strictly to source indices to eliminate hallucination. Second, a lightweight ranking module selects query-relevant sub-trees for linearization. To rigorously evaluate structural understanding, we release StructBench, a manually annotated dataset of 248 diverse documents. Empirical results demonstrate that our method achieves state-of-the-art structural prediction accuracy and significantly outperforms frontier LLMs while reducing costs. Furthermore, our structure-aware compression substantially enhances performance across downstream tasks ranging from long-context tasks to complex Deep Search scenarios.
>
---
#### [new 008] Low-Resource, High-Impact: Building Corpora for Inclusive Language Technologies
- **分类: cs.CL; cs.AI**

- **简介: 该论文是一篇教程，面向NLP从业者，聚焦低资源语言技术构建。旨在解决数据稀缺、文化差异导致的不平等问题，提供从数据采集、平行句挖掘到MT与下游任务的端到端实践工具与公平、可复现、社区驱动的开发方法。**

- **链接: [https://arxiv.org/pdf/2512.14576v1](https://arxiv.org/pdf/2512.14576v1)**

> **作者:** Ekaterina Artemova; Laurie Burchell; Daryna Dementieva; Shu Okabe; Mariya Shmatova; Pedro Ortiz Suarez
>
> **备注:** Tutorial is accepted to LREC2026
>
> **摘要:** This tutorial (https://tum-nlp.github.io/low-resource-tutorial) is designed for NLP practitioners, researchers, and developers working with multilingual and low-resource languages who seek to create more equitable and socially impactful language technologies. Participants will walk away with a practical toolkit for building end-to-end NLP pipelines for underrepresented languages -- from data collection and web crawling to parallel sentence mining, machine translation, and downstream applications such as text classification and multimodal reasoning. The tutorial presents strategies for tackling the challenges of data scarcity and cultural variance, offering hands-on methods and modeling frameworks. We will focus on fair, reproducible, and community-informed development approaches, grounded in real-world scenarios. We will showcase a diverse set of use cases covering over 10 languages from different language families and geopolitical contexts, including both digitally resource-rich and severely underrepresented languages.
>
---
#### [new 009] Agreement Between Large Language Models and Human Raters in Essay Scoring: A Research Synthesis
- **分类: cs.CL**

- **简介: 该论文属教育测量与AI评估交叉任务，旨在探究LLMs在自动作文评分中与人类评卷者的一致性。作者按PRISMA 2020规范，系统综述2022–2025年65项研究，分析各类一致性指标（如Kappa、Pearson相关等）分布及影响因素。**

- **链接: [https://arxiv.org/pdf/2512.14561v1](https://arxiv.org/pdf/2512.14561v1)**

> **作者:** Hongli Li; Che Han Chen; Kevin Fan; Chiho Young-Johnson; Soyoung Lim; Yali Feng
>
> **备注:** This manuscript is under review as a book chapter
>
> **摘要:** Despite the growing promise of large language models (LLMs) in automatic essay scoring (AES), empirical findings regarding their reliability compared to human raters remain mixed. Following the PRISMA 2020 guidelines, we synthesized 65 published and unpublished studies from January 2022 to August 2025 that examined agreement between LLMs and human raters in AES. Across studies, reported LLM-human agreement was generally moderate to good, with agreement indices (e.g., Quadratic Weighted Kappa, Pearson correlation, and Spearman's rho) mostly ranging between 0.30 and 0.80. Substantial variability in agreement levels was observed across studies, reflecting differences in study-specific factors as well as the lack of standardized reporting practices. Implications and directions for future research are discussed.
>
---
#### [new 010] Multilingual and Continuous Backchannel Prediction: A Cross-lingual Study
- **分类: cs.CL; cs.HC; cs.SD**

- **简介: 该论文研究多语言连续反馈语（backchannel）预测任务，旨在揭示日、英、中三语在反馈时机上的跨语言差异。提出基于Transformer的多语言帧级模型，联合辅助任务训练，分析线索使用、上下文长度影响及零-shot迁移效果，并实现CPU实时推理。**

- **链接: [https://arxiv.org/pdf/2512.14085v1](https://arxiv.org/pdf/2512.14085v1)**

> **作者:** Koji Inoue; Mikey Elmers; Yahui Fu; Zi Haur Pang; Taiga Mori; Divesh Lala; Keiko Ochi; Tatsuya Kawahara
>
> **备注:** This paper has been accepted for presentation at International Workshop on Spoken Dialogue Systems Technology 2026 (IWSDS 2026) and represents the author's version of the work
>
> **摘要:** We present a multilingual, continuous backchannel prediction model for Japanese, English, and Chinese, and use it to investigate cross-linguistic timing behavior. The model is Transformer-based and operates at the frame level, jointly trained with auxiliary tasks on approximately 300 hours of dyadic conversations. Across all three languages, the multilingual model matches or surpasses monolingual baselines, indicating that it learns both language-universal cues and language-specific timing patterns. Zero-shot transfer with two-language training remains limited, underscoring substantive cross-lingual differences. Perturbation analyses reveal distinct cue usage: Japanese relies more on short-term linguistic information, whereas English and Chinese are more sensitive to silence duration and prosodic variation; multilingual training encourages shared yet adaptable representations and reduces overreliance on pitch in Chinese. A context-length study further shows that Japanese is relatively robust to shorter contexts, while Chinese benefits markedly from longer contexts. Finally, we integrate the trained model into a real-time processing software, demonstrating CPU-only inference. Together, these findings provide a unified model and empirical evidence for how backchannel timing differs across languages, informing the design of more natural, culturally-aware spoken dialogue systems.
>
---
#### [new 011] JMMMU-Pro: Image-based Japanese Multi-discipline Multimodal Understanding Benchmark via Vibe Benchmark Construction
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
#### [new 012] Inflation Attitudes of Large Language Models
- **分类: cs.CL; econ.EM**

- **简介: 该论文属社会科学研究任务，探究LLM（GPT-3.5-turbo）能否模拟人类通胀感知与预期。利用其训练截止于2021年9月的特性，对比英国家庭调查与官方数据，分析其对价格信号的响应规律，并用Shapley分解识别驱动因素。**

- **链接: [https://arxiv.org/pdf/2512.14306v1](https://arxiv.org/pdf/2512.14306v1)**

> **作者:** Nikoleta Anesti; Edward Hill; Andreas Joseph
>
> **备注:** 41 pages, 11 figures
>
> **摘要:** This paper investigates the ability of Large Language Models (LLMs), specifically GPT-3.5-turbo (GPT), to form inflation perceptions and expectations based on macroeconomic price signals. We compare the LLM's output to household survey data and official statistics, mimicking the information set and demographic characteristics of the Bank of England's Inflation Attitudes Survey (IAS). Our quasi-experimental design exploits the timing of GPT's training cut-off in September 2021 which means it has no knowledge of the subsequent UK inflation surge. We find that GPT tracks aggregate survey projections and official statistics at short horizons. At a disaggregated level, GPT replicates key empirical regularities of households' inflation perceptions, particularly for income, housing tenure, and social class. A novel Shapley value decomposition of LLM outputs suited for the synthetic survey setting provides well-defined insights into the drivers of model outputs linked to prompt content. We find that GPT demonstrates a heightened sensitivity to food inflation information similar to that of human respondents. However, we also find that it lacks a consistent model of consumer price inflation. More generally, our approach could be used to evaluate the behaviour of LLMs for use in the social sciences, to compare different models, or to assist in survey design.
>
---
#### [new 013] Ladder Up, Memory Down: Low-Cost Fine-Tuning With Side Nets
- **分类: cs.CL; cs.LG**

- **简介: 该论文属大语言模型高效微调任务，旨在解决显存受限下LLM微调难问题。提出Ladder Side Tuning（LST）方法，通过轻量侧网络大幅降低峰值显存（降50%），支持7B模型在12GB GPU上微调；并扩展出xLadder，提升推理深度而不增显存开销。**

- **链接: [https://arxiv.org/pdf/2512.14237v1](https://arxiv.org/pdf/2512.14237v1)**

> **作者:** Estelle Zheng; Nathan Cerisara; Sébastien Warichet; Emmanuel Helbert; Christophe Cerisara
>
> **摘要:** Fine-tuning large language models (LLMs) is often limited by the memory available on commodity GPUs. Parameter-efficient fine-tuning (PEFT) methods such as QLoRA reduce the number of trainable parameters, yet still incur high memory usage induced by the backward pass in the full model. We revisit Ladder Side Tuning (LST), a rarely explored PEFT technique that adds a lightweight side network, and show that it matches QLoRA's compute scaling slope while cutting peak memory by 50\%. Across different downstream benchmarks spanning natural language understanding, mathematical and LLM-critic tasks, LST has competitive performance with QLoRA's accuracy on average while being much more memory-efficient. This efficiency enables fine-tuning of 7B-parameter models on a single 12 GB consumer GPU with 2k-token contexts, requiring no gradient checkpointing\textemdash conditions under which QLoRA exhausts memory. Beyond memory efficiency, we also establish scaling laws showing that LST scales similarly to QLoRA. We exploit Ladder's architectural flexibility by introducing xLadder, a depth-extended variant that increases effective depth via cross-connections and shortens chain-of-thought (CoT) at fixed parameter count. Ladder is strong when memory is the bottleneck; xLadder builds on this by enabling deeper reasoning without additional memory overhead.
>
---
#### [new 014] Effect of Document Packing on the Latent Multi-Hop Reasoning Capabilities of Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属LLM训练优化任务，探究文档打包对模型隐式多跳推理能力的影响。研究发现打包可提升性能但增加算力消耗，并通过消融实验揭示关键影响因素，为高效训练提供实践指导。**

- **链接: [https://arxiv.org/pdf/2512.14427v1](https://arxiv.org/pdf/2512.14427v1)**

> **作者:** Gabriele Prato; Shagun Sodhani; Alessandro Sordoni; Sarath Chandar
>
> **摘要:** The standard practice for training large language models involves packing multiple documents together to optimize computational efficiency. However, the impact of this process on the models' capabilities remains largely unexplored. To address this gap, we investigate how different document-packing strategies influence the latent multi-hop reasoning abilities of LLMs. Our findings indicate that packing can improve model performance compared to training on individual documents, at the expense of more compute. To further understand the underlying mechanisms, we conduct an ablation study, identifying key factors that explain the advantages of packing. Ultimately, our research deepens the understanding of LLM training dynamics and provides practical insights for optimizing model development.
>
---
#### [new 015] Linguists should learn to love speech-based deep learning models
- **分类: cs.CL; cs.SD; eess.AS; q-bio.NC**

- **简介: 该论文属学术评论任务，旨在纠正语言学界过度关注文本型大模型的倾向。它指出文本LLM无法覆盖语音相关的语言现象，主张转向语音驱动的深度学习模型，以更好支撑语言学理论解释与实证研究。**

- **链接: [https://arxiv.org/pdf/2512.14506v1](https://arxiv.org/pdf/2512.14506v1)**

> **作者:** Marianne de Heer Kloots; Paul Boersma; Willem Zuidema
>
> **备注:** Commentary on Futrell, R., & Mahowald, K. arXiv:2501.17047 (in press). How Linguistics Learned to Stop Worrying and Love the Language Models. Behavioural and Brain Sciences
>
> **摘要:** Futrell and Mahowald present a useful framework bridging technology-oriented deep learning systems and explanation-oriented linguistic theories. Unfortunately, the target article's focus on generative text-based LLMs fundamentally limits fruitful interactions with linguistics, as many interesting questions on human language fall outside what is captured by written text. We argue that audio-based deep learning models can and should play a crucial role.
>
---
#### [new 016] Polypersona: Persona-Grounded LLM for Synthetic Survey Responses
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Polypersona框架，解决小模型生成可信、 persona驱动的合成调查响应任务。通过LoRA高效微调量化小模型，构建多领域persona响应数据集，并设计混合评估指标验证效果，实现轻量模型媲美大模型的性能。**

- **链接: [https://arxiv.org/pdf/2512.14562v1](https://arxiv.org/pdf/2512.14562v1)**

> **作者:** Tejaswani Dash; Dinesh Karri; Anudeep Vurity; Gautam Datla; Tazeem Ahmad; Saima Rafi; Rohith Tangudu
>
> **备注:** Accepted in IEEE Bigdata 2025- LLMs4ALL
>
> **摘要:** This paper introduces PolyPersona, a generative framework for synthesizing persona-conditioned survey responses across multiple domains. The framework instruction-tunes compact chat models using parameter-efficient LoRA adapters with 4-bit quantization under a resource-adaptive training setup. A dialogue-based data pipeline explicitly preserves persona cues, ensuring consistent behavioral alignment across generated responses. Using this pipeline, we construct a dataset of 3,568 synthetic survey responses spanning ten domains and 433 distinct personas, enabling controlled instruction tuning and systematic multi-domain evaluation. We evaluate the generated responses using a multi-metric evaluation suite that combines standard text generation metrics, including BLEU, ROUGE, and BERTScore, with survey-specific metrics designed to assess structural coherence, stylistic consistency, and sentiment alignment.Experimental results show that compact models such as TinyLlama 1.1B and Phi-2 achieve performance comparable to larger 7B to 8B baselines, with a highest BLEU score of 0.090 and ROUGE-1 of 0.429. These findings demonstrate that persona-conditioned fine-tuning enables small language models to generate reliable and coherent synthetic survey data. The proposed framework provides an efficient and reproducible approach for survey data generation, supporting scalable evaluation while facilitating bias analysis through transparent and open protocols.
>
---
#### [new 017] SASQ: Static Activation Scaling for Quantization-Aware Training in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属大模型量化任务，旨在解决静态量化精度低、动态量化部署难的问题。提出SASQ框架，仅优化激活量化因子（不更新权重），自适应截断异常值，在保持静态推理高效性的同时提升精度，甚至超越FP16基线。**

- **链接: [https://arxiv.org/pdf/2512.14481v1](https://arxiv.org/pdf/2512.14481v1)**

> **作者:** Shizhuo Mao; Song Chen; Yi Kang
>
> **摘要:** Large language models (LLMs) excel at natural language tasks but face deployment challenges due to their growing size outpacing GPU memory advancements. Model quantization mitigates this issue by lowering weight and activation precision, but existing solutions face fundamental trade-offs: dynamic quantization incurs high computational overhead and poses deployment challenges on edge devices, while static quantization sacrifices accuracy. Existing approaches of quantization-aware training (QAT) further suffer from weight training costs. We propose SASQ: a lightweight QAT framework specifically tailored for activation quantization factors. SASQ exclusively optimizes only the quantization factors (without changing pre-trained weights), enabling static inference with high accuracy while maintaining deployment efficiency. SASQ adaptively truncates some outliers, thereby reducing the difficulty of quantization while preserving the distributional characteristics of the activations. SASQ not only surpasses existing SOTA quantization schemes but also outperforms the corresponding FP16 models. On LLaMA2-7B, it achieves 5.2% lower perplexity than QuaRot and 4.7% lower perplexity than the FP16 model on WikiText2.
>
---
#### [new 018] A Comparative Analysis of Retrieval-Augmented Generation Techniques for Bengali Standard-to-Dialect Machine Translation Using LLMs
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文研究标准语到方言的机器翻译任务，解决 Bengali 低资源方言翻译难题。提出并比较两种无须微调的 RAG 方法：基于音频转录本和基于结构化句对的检索增强生成 pipeline，验证后者显著提升翻译质量，尤其使小模型超越大模型。**

- **链接: [https://arxiv.org/pdf/2512.14179v1](https://arxiv.org/pdf/2512.14179v1)**

> **作者:** K. M. Jubair Sami; Dipto Sumit; Ariyan Hossain; Farig Sadeque
>
> **备注:** Accepted to the Second Workshop on Bangla Language Processing (BLP) at IJCNLP-AACL 2025. 14 pages, 9 figures, 6 tables
>
> **摘要:** Translating from a standard language to its regional dialects is a significant NLP challenge due to scarce data and linguistic variation, a problem prominent in the Bengali language. This paper proposes and compares two novel RAG pipelines for standard-to-dialectal Bengali translation. The first, a Transcript-Based Pipeline, uses large dialect sentence contexts from audio transcripts. The second, a more effective Standardized Sentence-Pairs Pipeline, utilizes structured local\_dialect:standard\_bengali sentence pairs. We evaluated both pipelines across six Bengali dialects and multiple LLMs using BLEU, ChrF, WER, and BERTScore. Our findings show that the sentence-pair pipeline consistently outperforms the transcript-based one, reducing Word Error Rate (WER) from 76\% to 55\% for the Chittagong dialect. Critically, this RAG approach enables smaller models (e.g., Llama-3.1-8B) to outperform much larger models (e.g., GPT-OSS-120B), demonstrating that a well-designed retrieval strategy can be more crucial than model size. This work contributes an effective, fine-tuning-free solution for low-resource dialect translation, offering a practical blueprint for preserving linguistic diversity.
>
---
#### [new 019] CogMem: A Cognitive Memory Architecture for Sustained Multi-Turn Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属AI推理增强任务，旨在解决LLM在多轮交互中出现的推理偏差、任务漂移、幻觉及记忆衰减等问题。作者提出CogMem认知记忆架构，含长时记忆、直接访问记忆和注意力焦点三层机制，以结构化持久记忆提升多轮推理一致性与效率。**

- **链接: [https://arxiv.org/pdf/2512.14118v1](https://arxiv.org/pdf/2512.14118v1)**

> **作者:** Yiran Zhang; Jincheng Hu; Mark Dras; Usman Naseem
>
> **备注:** underreview
>
> **摘要:** Large language models (LLMs) excel at single-turn reasoning but often lose accuracy and coherence over extended, multi-turn interactions. Recent evaluations such as TurnBench highlight recurring failure modes-reasoning bias, task drift, hallucination, overconfidence, and memory decay. Current approaches typically append full conversational histories, causing unbounded context growth, higher computational costs, and degraded reasoning efficiency. We introduce CogMem, a cognitively inspired, memory-augmented LLM architecture that supports sustained iterative reasoning through structured, persistent memory. CogMem incorporates three layers: a Long-Term Memory (LTM) that consolidates cross-session reasoning strategies; a Direct Access (DA) memory that maintains session-level notes and retrieves relevant long-term memories; and a Focus of Attention (FoA) mechanism that dynamically reconstructs concise, task-relevant context at each turn. Experiments on TurnBench show that this layered design mitigates reasoning failures, controls context growth, and improves consistency across extended reasoning chains, moving toward more reliable, human-like reasoning in LLMs.
>
---
#### [new 020] Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属语言模型架构转换任务，旨在解决扩散语言模型（dLMs）训练效率低、速度优势未充分发挥的问题。提出AR-to-dLM高效转换方法：设计块状因果注意力保持预训练权重分布，并引入位置依赖掩码缩小训练-测试分布差异，实现高精度与高吞吐兼顾。**

- **链接: [https://arxiv.org/pdf/2512.14067v1](https://arxiv.org/pdf/2512.14067v1)**

> **作者:** Yonggan Fu; Lexington Whalen; Zhifan Ye; Xin Dong; Shizhe Diao; Jingyu Liu; Chengyue Wu; Hao Zhang; Enze Xie; Song Han; Maksim Khadkevich; Jan Kautz; Yingyan Celine Lin; Pavlo Molchanov
>
> **摘要:** Diffusion language models (dLMs) have emerged as a promising paradigm that enables parallel, non-autoregressive generation, but their learning efficiency lags behind that of autoregressive (AR) language models when trained from scratch. To this end, we study AR-to-dLM conversion to transform pretrained AR models into efficient dLMs that excel in speed while preserving AR models' task accuracy. We achieve this by identifying limitations in the attention patterns and objectives of existing AR-to-dLM methods and then proposing principles and methodologies for more effective AR-to-dLM conversion. Specifically, we first systematically compare different attention patterns and find that maintaining pretrained AR weight distributions is critical for effective AR-to-dLM conversion. As such, we introduce a continuous pretraining scheme with a block-wise attention pattern, which remains causal across blocks while enabling bidirectional modeling within each block. We find that this approach can better preserve pretrained AR models' weight distributions than fully bidirectional modeling, in addition to its known benefit of enabling KV caching, and leads to a win-win in accuracy and efficiency. Second, to mitigate the training-test gap in mask token distributions (uniform vs. highly left-to-right), we propose a position-dependent token masking strategy that assigns higher masking probabilities to later tokens during training to better mimic test-time behavior. Leveraging this framework, we conduct extensive studies of dLMs' attention patterns, training dynamics, and other design choices, providing actionable insights into scalable AR-to-dLM conversion. These studies lead to the Efficient-DLM family, which outperforms state-of-the-art AR models and dLMs, e.g., our Efficient-DLM 8B achieves +5.4%/+2.7% higher accuracy with 4.5x/2.7x higher throughput compared to Dream 7B and Qwen3 4B, respectively.
>
---
#### [new 021] Olmo 3
- **分类: cs.CL; cs.LG**

- **简介: 该论文发布Olmo 3系列全开源大语言模型（7B/32B），聚焦长上下文推理、函数调用、编程等多任务能力。任务为开源大模型研发，旨在解决高质量、全流程可复现的开放模型稀缺问题；工作包括构建并完整公开模型、训练数据、检查点及全部依赖。**

- **链接: [https://arxiv.org/pdf/2512.13961v1](https://arxiv.org/pdf/2512.13961v1)**

> **作者:** Team Olmo; :; Allyson Ettinger; Amanda Bertsch; Bailey Kuehl; David Graham; David Heineman; Dirk Groeneveld; Faeze Brahman; Finbarr Timbers; Hamish Ivison; Jacob Morrison; Jake Poznanski; Kyle Lo; Luca Soldaini; Matt Jordan; Mayee Chen; Michael Noukhovitch; Nathan Lambert; Pete Walsh; Pradeep Dasigi; Robert Berry; Saumya Malik; Saurabh Shah; Scott Geng; Shane Arora; Shashank Gupta; Taira Anderson; Teng Xiao; Tyler Murray; Tyler Romero; Victoria Graf; Akari Asai; Akshita Bhagia; Alexander Wettig; Alisa Liu; Aman Rangapur; Chloe Anastasiades; Costa Huang; Dustin Schwenk; Harsh Trivedi; Ian Magnusson; Jaron Lochner; Jiacheng Liu; Lester James V. Miranda; Maarten Sap; Malia Morgan; Michael Schmitz; Michal Guerquin; Michael Wilson; Regan Huff; Ronan Le Bras; Rui Xin; Rulin Shao; Sam Skjonsberg; Shannon Zejiang Shen; Shuyue Stella Li; Tucker Wilde; Valentina Pyatkin; Will Merrill; Yapei Chang; Yuling Gu; Zhiyuan Zeng; Ashish Sabharwal; Luke Zettlemoyer; Pang Wei Koh; Ali Farhadi; Noah A. Smith; Hannaneh Hajishirzi
>
> **摘要:** We introduce Olmo 3, a family of state-of-the-art, fully-open language models at the 7B and 32B parameter scales. Olmo 3 model construction targets long-context reasoning, function calling, coding, instruction following, general chat, and knowledge recall. This release includes the entire model flow, i.e., the full lifecycle of the family of models, including every stage, checkpoint, data point, and dependency used to build it. Our flagship model, Olmo 3 Think 32B, is the strongest fully-open thinking model released to-date.
>
---
#### [new 022] Towards Nepali-language LLMs: Efficient GPT training with a Nepali BPE tokenizer
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向低资源语言尼泊尔语，解决其缺乏高质量生成式模型的问题。工作包括：构建16k BPE尼泊尔语分词器，基于GPT-2架构改进训练策略（学习率调度、批处理缩放等），使用10.75GB清洗语料预训练，并集成FlashAttention优化训练，最终实现尼泊尔语新闻文本生成。**

- **链接: [https://arxiv.org/pdf/2512.14585v1](https://arxiv.org/pdf/2512.14585v1)**

> **作者:** Adarsha Shrestha; Basanta Pokharel; Binit Shrestha; Smriti Adhikari; Dinesh Gothe
>
> **备注:** Work in progress
>
> **摘要:** Nepali, a low-resource language spoken by over 32 million people, continues to face challenges in natural language processing (NLP) due to its complex grammar, agglutinative morphology, and limited availability of high-quality corpora. Most efforts to date have centered on basic encoder architectures; they remain insufficient for Nepali-specific text generation. This study presents a GPT-2-based Nepali language model trained using several training strategies inspired by GPT-3, including optimized learning rate schedules, batch scaling, and architectural refinements. A custom 16k Byte-Pair Encoding (BPE) tokenizer was trained exclusively on Nepali text to ensure more consistent segmentation and improved input representation. The model was pretrained on a combined dataset comprising a 10.75GB cleaned NepBERTa corpus and additional web-scraped Nepali news articles. FlashAttention was integrated to reduce memory usage and stabilize training. After two epochs, the model achieved a training loss of 3.168177, a validation loss of 3.081982, and a final perplexity of 21.80, demonstrating its capability to generate coherent Nepali news-style text.
>
---
#### [new 023] What Affects the Effective Depth of Large Language Models?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）的“有效深度”影响因素，旨在解决模型深层未被充分利用的问题。通过分析Qwen-2.5系列模型，考察规模、训练方式（如长CoT）和任务难度对有效深度的影响，发现其不随这些因素显著提升，揭示了层利用不足的共性现象。**

- **链接: [https://arxiv.org/pdf/2512.14064v1](https://arxiv.org/pdf/2512.14064v1)**

> **作者:** Yi Hu; Cai Zhou; Muhan Zhang
>
> **摘要:** The scaling of large language models (LLMs) emphasizes increasing depth, yet performance gains diminish with added layers. Prior work introduces the concept of "effective depth", arguing that deeper models fail to fully utilize their layers for meaningful computation. Building on this, we systematically study how effective depth varies with model scale, training type, and task difficulty. First, we analyze the model behavior of Qwen-2.5 family (1.5B-32B) and find that while the number of effective layers grows with model size, the effective depth ratio remains stable. Besides, comparisons between base and corresponding long-CoT models show no increase in effective depth, suggesting that improved reasoning stems from longer context rather than deeper per-token computation. Furthermore, evaluations across tasks of varying difficulty indicate that models do not dynamically use more layers for harder problems. Our results suggest that current LLMs underuse available depth across scales, training paradigms and tasks of varying difficulties, pointing out research opportunities on increasing the layer utilization rate of LLMs, model pruning, and early exiting. Our code is released at https://github.com/AheadOFpotato/what_affects_effective_depth.
>
---
#### [new 024] Astraea: A State-Aware Scheduling Engine for LLM-Powered Agents
- **分类: cs.CL**

- **简介: 该论文属系统优化任务，旨在解决LLM智能体多阶段工作流中端到端延迟（JCT）高的问题。提出状态感知的调度引擎Astraea，含分层调度算法、增强HRRN策略和自适应KV缓存管理，显著降低平均JCT并提升高负载鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.14142v1](https://arxiv.org/pdf/2512.14142v1)**

> **作者:** Hongqiu Ni; Jiabao Zhang; Guopeng Li; Zilong Wang; Ruiqi Wu; Chi Zhang; Haisheng Tan
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** Large Language Models (LLMs) are increasingly being deployed as intelligent agents. Their multi-stage workflows, which alternate between local computation and calls to external network services like Web APIs, introduce a mismatch in their execution pattern and the scheduling granularity of existing inference systems such as vLLM. Existing systems typically focus on per-segment optimization which prevents them from minimizing the end-to-end latency of the complete agentic workflow, i.e., the global Job Completion Time (JCT) over the entire request lifecycle. To address this limitation, we propose Astraea, a service engine designed to shift the optimization from local segments to the global request lifecycle. Astraea employs a state-aware, hierarchical scheduling algorithm that integrates a request's historical state with future predictions. It dynamically classifies requests by their I/O and compute intensive nature and uses an enhanced HRRN policy to balance efficiency and fairness. Astraea also implements an adaptive KV cache manager that intelligently handles the agent state during I/O waits based on the system memory pressure. Extensive experiments show that Astraea reduces average JCT by up to 25.5\% compared to baseline methods. Moreover, our approach demonstrates strong robustness and stability under high load across various model scales.
>
---
#### [new 025] FiNERweb: Datasets and Artifacts for Scalable Multilingual Named Entity Recognition
- **分类: cs.CL**

- **简介: 该论文面向多语言命名实体识别（NER）任务，旨在解决高质量、可复用的多语言NER数据集稀缺问题。作者提出FiNERweb流水线，基于FineWeb-Edu和多语言LLM，自动生成覆盖91种语言、225k篇章的合成标注数据，并开源数据与全部工具。**

- **链接: [https://arxiv.org/pdf/2512.13884v1](https://arxiv.org/pdf/2512.13884v1)**

> **作者:** Jonas Golde; Patrick Haller; Alan Akbik
>
> **摘要:** Recent multilingual named entity recognition (NER) work has shown that large language models (LLMs) can provide effective synthetic supervision, yet such datasets have mostly appeared as by-products of broader experiments rather than as systematic, reusable resources. We introduce FiNERweb, a dataset-creation pipeline that scales the teacher-student paradigm to 91 languages and 25 scripts. Building on FineWeb-Edu, our approach trains regression models to identify NER-relevant passages and annotates them with multilingual LLMs, resulting in about 225k passages with 235k distinct entity labels. Our experiments show that the regression model achieves more than 84 F1, and that models trained on FiNERweb obtain comparable or improved performance in zero shot transfer settings on English, Thai, and Swahili, despite being trained on 19x less data than strong baselines. In addition, we assess annotation quality using LLM-as-a-judge and observe consistently high scores for both faithfulness (3.99 out of 5) and completeness (4.05 out of 5), indicating reliable and informative annotations. Further, we release the dataset with both English labels and translated label sets in the respective target languages because we observe that the performance of current state-of-the-art models drops by 0.02 to 0.09 F1 when evaluated using target language labels instead of English ones. We release FiNERweb together with all accompanying artifacts to the research community in order to facilitate more effective student-teacher training for multilingual named entity recognition.
>
---
#### [new 026] Spoken DialogSum: An Emotion-Rich Conversational Dataset for Spoken Dialogue Summarization
- **分类: cs.CL; cs.AI; cs.LG; eess.AS**

- **简介: 该论文面向**口语对话摘要任务**，旨在解决**情感感知与语音建模缺乏对齐数据**的问题。作者构建了首个含原始音频、事实摘要、情感摘要及细粒度副语言标签（情感/性别/年龄/语速/音高）的对话数据集Spoken DialogSum（13.46k样本），并验证端到端Audio-LLM优于ASR-LLM级联系统。**

- **链接: [https://arxiv.org/pdf/2512.14687v1](https://arxiv.org/pdf/2512.14687v1)**

> **作者:** Yen-Ju Lu; Kunxiao Gao; Mingrui Liang; Helin Wang; Thomas Thebaud; Laureano Moro-Velazquez; Najim Dehak; Jesus Villalba
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Recent audio language models can follow long conversations. However, research on emotion-aware or spoken dialogue summarization is constrained by the lack of data that links speech, summaries, and paralinguistic cues. We introduce Spoken DialogSum, the first corpus aligning raw conversational audio with factual summaries, emotion-rich summaries, and utterance-level labels for speaker age, gender, and emotion. The dataset is built in two stages: first, an LLM rewrites DialogSum scripts with Switchboard-style fillers and back-channels, then tags each utterance with emotion, pitch, and speaking rate. Second, an expressive TTS engine synthesizes speech from the tagged scripts, aligned with paralinguistic labels. Spoken DialogSum comprises 13,460 emotion-diverse dialogues, each paired with both a factual and an emotion-focused summary. The dataset is available online at https://fatfat-emosum.github.io/EmoDialog-Sum-Audio-Samples/. Baselines show that an Audio-LLM raises emotional-summary ROUGE-L by 28% relative to a cascaded ASR-LLM system, confirming the value of end-to-end speech modeling.
>
---
#### [new 027] Structure-Aware Decoding Mechanisms for Complex Entity Extraction with Large-Scale Language Models
- **分类: cs.CL**

- **简介: 该论文面向嵌套与重叠实体抽取任务，解决传统方法语义完整性与结构一致性难兼顾的问题。提出结构感知解码机制：结合候选片段生成、分层结构约束及联合损失优化，在ACE2005上显著提升边界定位与结构建模能力。**

- **链接: [https://arxiv.org/pdf/2512.13980v1](https://arxiv.org/pdf/2512.13980v1)**

> **作者:** Zhimin Qiu; Di Wu; Feng Liu; Chenrui Hu; Yuxiao Wang
>
> **摘要:** This paper proposes a structure-aware decoding method based on large language models to address the difficulty of traditional approaches in maintaining both semantic integrity and structural consistency in nested and overlapping entity extraction tasks. The method introduces a candidate span generation mechanism and structured attention modeling to achieve unified modeling of entity boundaries, hierarchical relationships, and cross-dependencies. The model first uses a pretrained language model to obtain context-aware semantic representations, then captures multi-granular entity span features through candidate representation combinations, and introduces hierarchical structural constraints during decoding to ensure consistency between semantics and structure. To enhance stability in complex scenarios, the model jointly optimizes classification loss and structural consistency loss, maintaining high recognition accuracy under multi-entity co-occurrence and long-sentence dependency conditions. Experiments conducted on the ACE 2005 dataset demonstrate significant improvements in Accuracy, Precision, Recall, and F1-Score, particularly in nested and overlapping entity recognition, where the model shows stronger boundary localization and structural modeling capability. This study verifies the effectiveness of structure-aware decoding in complex semantic extraction tasks, provides a new perspective for developing language models with hierarchical understanding, and establishes a methodological foundation for high-precision information extraction.
>
---
#### [new 028] VLegal-Bench: Cognitively Grounded Benchmark for Vietnamese Legal Reasoning of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文构建了首个面向越南法律的、认知理论驱动的评测基准VLegal-Bench，旨在解决现有LLM在越南法律理解与推理能力评估缺乏系统性、权威性和实用性的问题。工作包括基于Bloom分类法设计多层级任务，由法律专家标注验证10,450个样本，覆盖问答、检索增强生成、多步推理等真实场景。**

- **链接: [https://arxiv.org/pdf/2512.14554v1](https://arxiv.org/pdf/2512.14554v1)**

> **作者:** Nguyen Tien Dong; Minh-Anh Nguyen; Thanh Dat Hoang; Nguyen Tuan Ngoc; Dao Xuan Quang Minh; Phan Phi Hai; Nguyen Thi Ngoc Anh; Dang Van Tu; Binh Vu
>
> **摘要:** The rapid advancement of large language models (LLMs) has enabled new possibilities for applying artificial intelligence within the legal domain. Nonetheless, the complexity, hierarchical organization, and frequent revisions of Vietnamese legislation pose considerable challenges for evaluating how well these models interpret and utilize legal knowledge. To address this gap, Vietnamese Legal Benchmark (VLegal-Bench) is introduced, the first comprehensive benchmark designed to systematically assess LLMs on Vietnamese legal tasks. Informed by Bloom's cognitive taxonomy, VLegal-Bench encompasses multiple levels of legal understanding through tasks designed to reflect practical usage scenarios. The benchmark comprises 10,450 samples generated through a rigorous annotation pipeline, where legal experts label and cross-validate each instance using our annotation system to ensure every sample is grounded in authoritative legal documents and mirrors real-world legal assistant workflows, including general legal questions and answers, retrieval-augmented generation, multi-step reasoning, and scenario-based problem solving tailored to Vietnamese law. By providing a standardized, transparent, and cognitively informed evaluation framework, VLegal-Bench establishes a solid foundation for assessing LLM performance in Vietnamese legal contexts and supports the development of more reliable, interpretable, and ethically aligned AI-assisted legal systems.
>
---
#### [new 029] MMGR: Multi-Modal Generative Reasoning
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
#### [new 030] C-ing Clearly: Enhanced Binary Code Explanations using C code
- **分类: cs.CL; cs.LG**

- **简介: 该论文属二进制代码理解任务，旨在提升LLM对汇编代码的理解能力。针对LLM不擅处理低级语言的问题，提出“C-ing Clearly”方法：利用对应C代码生成合成数据，通过微调增强模型在二进制摘要与漏洞检测上的性能，且跨模型家族和规模均有效。**

- **链接: [https://arxiv.org/pdf/2512.14500v1](https://arxiv.org/pdf/2512.14500v1)**

> **作者:** Teodor Poncu; Ioana Pintilie; Marius Dragoi; Dragos Tantaru; Florin Brad
>
> **备注:** 18 pages, 5 figures
>
> **摘要:** Large Language Models (LLMs) typically excel at coding tasks involving high-level programming languages, as opposed to lower-level programming languages, such as assembly. We propose a synthetic data generation method named C-ing Clearly, which leverages the corresponding C code to enhance an LLM's understanding of assembly. By fine-tuning on data generated through our method, we demonstrate improved LLM performance for binary code summarization and vulnerability detection. Our approach demonstrates consistent gains across different LLM families and model sizes.
>
---
#### [new 031] A Unified Sparse Attention via Multi-Granularity Compression
- **分类: cs.CL**

- **简介: 该论文属高效注意力机制研究，旨在解决LLM中自注意力计算复杂度高（O(n²)）的问题。提出UniSparse方法，通过多粒度压缩与复合令牌动态构建稀疏注意力，在保持≥99%全注意力精度的同时，提速达2.61×。**

- **链接: [https://arxiv.org/pdf/2512.14082v1](https://arxiv.org/pdf/2512.14082v1)**

> **作者:** Siran Liu; Zane Cao; Yongchao He
>
> **摘要:** Efficient long-context understanding and reasoning are increasingly vital for large language model (LLM) applications such as multi-turn dialogue and program analysis. However, the core self-attention mechanism scales quadratically with sequence length, creating a fundamental computational bottleneck. Existing sparse attention methods alleviate this issue but face trade-offs: training-based methods are costly and cannot be directly applied as acceleration plugins for other models, while inference-time methods often compromise efficiency or cross-modal generality. To address these limitations, we present UniSparse, a unified mechanism that introduces the notion of composite tokens--compact representations that aggregate multi-granularity contextual information. Building on this abstraction, UniSparse dynamically constructs sparse attention through multi-granularity compression and block-level selection, enabling efficient and hardware-friendly execution on GPU. Across multiple modalities and tasks ranging from synthetic benchmarks to real-world applications, UniSparse consistently surpasses state-of-the-art sparse attention methods (e.g., MInference, XAttention, FlexPrefill) in both accuracy and efficiency, achieving $\ge$ 99% of full-attention accuracy and up to 2.61$\times$ faster attention computation than FlashAttention.
>
---
#### [new 032] Writing in Symbiosis: Mapping Human Creative Agency in the AI Era
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属人机协作研究任务，旨在探究AI时代人类创作主体性演化问题。作者基于跨LLM前后的纵向写作语料，提出“双轨演化”模型，识别出三类作者创意范式，揭示人类在主题趋同下仍保持风格多样性的适应机制。**

- **链接: [https://arxiv.org/pdf/2512.13697v1](https://arxiv.org/pdf/2512.13697v1)**

> **作者:** Vivan Doshi; Mengyuan Li
>
> **备注:** Advances in Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** The proliferation of Large Language Models (LLMs) raises a critical question about what it means to be human when we share an increasingly symbiotic relationship with persuasive and creative machines. This paper examines patterns of human-AI coevolution in creative writing, investigating how human craft and agency are adapting alongside machine capabilities. We challenge the prevailing notion of stylistic homogenization by examining diverse patterns in longitudinal writing data. Using a large-scale corpus spanning the pre- and post-LLM era, we observe patterns suggestive of a "Dual-Track Evolution": thematic convergence around AI-related topics, coupled with structured stylistic differentiation. Our analysis reveals three emergent adaptation patterns: authors showing increased similarity to AI style, those exhibiting decreased similarity, and those maintaining stylistic stability while engaging with AI-related themes. This Creative Archetype Map illuminates how authorship is coevolving with AI, contributing to discussions about human-AI collaboration, detection challenges, and the preservation of creative diversity.
>
---
#### [new 033] SPARQL-LLM: Real-Time SPARQL Query Generation from Natural Language Questions
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出SPARQL-LLM，解决自然语言到SPARQL查询生成任务中准确率低、不支持联邦查询、响应慢、成本高等问题；通过轻量元数据驱动的架构，实现高精度、快速（快36×）、低成本（$0.01/问）、支持多语言与生物信息学联邦知识图谱的实时查询生成。**

- **链接: [https://arxiv.org/pdf/2512.14277v1](https://arxiv.org/pdf/2512.14277v1)**

> **作者:** Panayiotis Smeros; Vincent Emonet; Ruijie Wang; Ana-Claudia Sima; Tarcisio Mendes de Farias
>
> **备注:** 17 pages, 8 figures, 1 table. Under Review
>
> **摘要:** The advent of large language models is contributing to the emergence of novel approaches that promise to better tackle the challenge of generating structured queries, such as SPARQL queries, from natural language. However, these new approaches mostly focus on response accuracy over a single source while ignoring other evaluation criteria, such as federated query capability over distributed data stores, as well as runtime and cost to generate SPARQL queries. Consequently, they are often not production-ready or easy to deploy over (potentially federated) knowledge graphs with good accuracy. To mitigate these issues, in this paper, we extend our previous work and describe and systematically evaluate SPARQL-LLM, an open-source and triplestore-agnostic approach, powered by lightweight metadata, that generates SPARQL queries from natural language text. First, we describe its architecture, which consists of dedicated components for metadata indexing, prompt building, and query generation and execution. Then, we evaluate it based on a state-of-the-art challenge with multilingual questions, and a collection of questions from three of the most prevalent knowledge graphs within the field of bioinformatics. Our results demonstrate a substantial increase of 24% in the F1 Score on the state-of-the-art challenge, adaptability to high-resource languages such as English and Spanish, as well as ability to form complex and federated bioinformatics queries. Furthermore, we show that SPARQL-LLM is up to 36x faster than other systems participating in the challenge, while costing a maximum of $0.01 per question, making it suitable for real-time, low-cost text-to-SPARQL applications. One such application deployed over real-world decentralized knowledge graphs can be found at https://www.expasy.org/chat.
>
---
#### [new 034] EvoLattice: Persistent Internal-Population Evolution through Multi-Alternative Quality-Diversity Graph Representations for LLM-Guided Program Discovery
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; cs.NE**

- **简介: 该论文提出EvoLattice框架，解决LLM程序进化中单候选覆盖、结构脆弱和反馈稀疏问题。通过有向无环图表示多替代种群，支持路径组合生成候选、替代级评估与自修复，实现稳定、可解释的高质量多样性进化。**

- **链接: [https://arxiv.org/pdf/2512.13857v1](https://arxiv.org/pdf/2512.13857v1)**

> **作者:** Kamer Ali Yuksel
>
> **摘要:** Large language models (LLMs) are increasingly used to evolve programs and multi-agent systems, yet most existing approaches rely on overwrite-based mutations that maintain only a single candidate at a time. Such methods discard useful variants, suffer from destructive edits, and explore a brittle search space prone to structural failure. We introduce EvoLattice, a framework that represents an entire population of candidate programs or agent behaviors within a single directed acyclic graph. Each node stores multiple persistent alternatives, and every valid path through the graph defines a distinct executable candidate, yielding a large combinatorial search space without duplicating structure. EvoLattice enables fine-grained alternative-level evaluation by scoring each alternative across all paths in which it appears, producing statistics that reveal how local design choices affect global performance. These statistics provide a dense, data-driven feedback signal for LLM-guided mutation, recombination, and pruning, while preserving successful components. Structural correctness is guaranteed by a deterministic self-repair mechanism that enforces acyclicity and dependency consistency independently of the LLM. EvoLattice naturally extends to agent evolution by interpreting alternatives as prompt fragments or sub-agent behaviors. Across program synthesis (proxy and optimizer meta-learning), EvoLattice yields more stable evolution, greater expressivity, and stronger improvement trajectories than prior LLM-guided methods. The resulting dynamics resemble quality-diversity optimization, emerging implicitly from EvoLattice's internal multi-alternative representation rather than an explicit external archive.
>
---
#### [new 035] Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对长上下文大语言模型（LLMs）中静态自注意力导致的“分数稀释”问题，提出测试时训练（Test-Time Training）方法：对给定上下文做轻量梯度更新，以提升长上下文信号检索与利用能力。实验表明其显著优于推理时生成更多思维令牌等现有策略。**

- **链接: [https://arxiv.org/pdf/2512.13898v1](https://arxiv.org/pdf/2512.13898v1)**

> **作者:** Rachit Bansal; Aston Zhang; Rishabh Tiwari; Lovish Madaan; Sai Surya Duvvuri; Devvrit Khatri; David Brandfonbrener; David Alvarez-Melis; Prajjwal Bhargava; Mihir Sanjay Kale; Samy Jelassi
>
> **摘要:** Progress on training and architecture strategies has enabled LLMs with millions of tokens in context length. However, empirical evidence suggests that such long-context LLMs can consume far more text than they can reliably use. On the other hand, it has been shown that inference-time compute can be used to scale performance of LLMs, often by generating thinking tokens, on challenging tasks involving multi-step reasoning. Through controlled experiments on sandbox long-context tasks, we find that such inference-time strategies show rapidly diminishing returns and fail at long context. We attribute these failures to score dilution, a phenomenon inherent to static self-attention. Further, we show that current inference-time strategies cannot retrieve relevant long-context signals under certain conditions. We propose a simple method that, through targeted gradient updates on the given context, provably overcomes limitations of static self-attention. We find that this shift in how inference-time compute is spent leads to consistently large performance improvements across models and long-context benchmarks. Our method leads to large 12.6 and 14.1 percentage point improvements for Qwen3-4B on average across subsets of LongBench-v2 and ZeroScrolls benchmarks. The takeaway is practical: for long context, a small amount of context-specific training is a better use of inference compute than current inference-time scaling strategies like producing more thinking tokens.
>
---
#### [new 036] HyperVL: An Efficient and Dynamic Multimodal Large Language Model for Edge Devices
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
#### [new 037] Generative AI for Video Translation: A Scalable Architecture for Multilingual Video Conferencing
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
#### [new 038] RecGPT-V2 Technical Report
- **分类: cs.IR; cs.CL**

- **简介: 该论文属推荐系统任务，旨在解决LLM用于推荐时的效率低、解释单一、泛化弱、评估失准四大问题。提出RecGPT-V2：多智能体协同推理、混合表征压缩、元提示生成、约束强化学习及智能体评估框架，显著提升效果与实用性。**

- **链接: [https://arxiv.org/pdf/2512.14503v1](https://arxiv.org/pdf/2512.14503v1)**

> **作者:** Chao Yi; Dian Chen; Gaoyang Guo; Jiakai Tang; Jian Wu; Jing Yu; Mao Zhang; Wen Chen; Wenjun Yang; Yujie Luo; Yuning Jiang; Zhujin Gao; Bo Zheng; Binbin Cao; Changfa Wu; Dixuan Wang; Han Wu; Haoyi Hu; Kewei Zhu; Lang Tian; Lin Yang; Qiqi Huang; Siqi Yang; Wenbo Su; Xiaoxiao He; Xin Tong; Xu Chen; Xunke Xi; Xiaowei Huang; Yaxuan Wu; Yeqiu Yang; Yi Hu; Yujin Yuan; Yuliang Yan; Zile Zhou
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable potential in transforming recommender systems from implicit behavioral pattern matching to explicit intent reasoning. While RecGPT-V1 successfully pioneered this paradigm by integrating LLM-based reasoning into user interest mining and item tag prediction, it suffers from four fundamental limitations: (1) computational inefficiency and cognitive redundancy across multiple reasoning routes; (2) insufficient explanation diversity in fixed-template generation; (3) limited generalization under supervised learning paradigms; and (4) simplistic outcome-focused evaluation that fails to match human standards. To address these challenges, we present RecGPT-V2 with four key innovations. First, a Hierarchical Multi-Agent System restructures intent reasoning through coordinated collaboration, eliminating cognitive duplication while enabling diverse intent coverage. Combined with Hybrid Representation Inference that compresses user-behavior contexts, our framework reduces GPU consumption by 60% and improves exclusive recall from 9.39% to 10.99%. Second, a Meta-Prompting framework dynamically generates contextually adaptive prompts, improving explanation diversity by +7.3%. Third, constrained reinforcement learning mitigates multi-reward conflicts, achieving +24.1% improvement in tag prediction and +13.0% in explanation acceptance. Fourth, an Agent-as-a-Judge framework decomposes assessment into multi-step reasoning, improving human preference alignment. Online A/B tests on Taobao demonstrate significant improvements: +2.98% CTR, +3.71% IPV, +2.19% TV, and +11.46% NER. RecGPT-V2 establishes both the technical feasibility and commercial viability of deploying LLM-powered intent reasoning at scale, bridging the gap between cognitive exploration and industrial utility.
>
---
#### [new 039] Mitigating Catastrophic Forgetting in Mathematical Reasoning Finetuning through Mixed Training
- **分类: cs.LG; cs.CL**

- **简介: 该论文属大语言模型微调任务，旨在解决数学推理微调导致的灾难性遗忘问题。作者通过在数学数据与MultiNLI数据间采用混合训练（如1:1比例），成功消除遗忘，保持数学性能（12.0%）的同时维持NLI准确率（86.2%）。**

- **链接: [https://arxiv.org/pdf/2512.13706v1](https://arxiv.org/pdf/2512.13706v1)**

> **作者:** John Graham Reynolds
>
> **备注:** 11 pages, 2 figures. Code available at https://github.com/johngrahamreynolds/mathematical_catastrophe_mitigation. Models available at https://huggingface.co/collections/MarioBarbeque/catastrophic-forgetting-in-mathematical-reasoning
>
> **摘要:** When finetuning large language models for specialized tasks such as mathematical reasoning, models exhibit catastrophic forgetting, losing previously learned capabilities. We investigate this by finetuning Flan-T5-Base (250M parameters) on the DeepMind Mathematics dataset and measuring forgetting on MultiNLI. Math-only training improves mathematical accuracy from 3.1\% to 12.0\% but causes NLI accuracy to collapse from 81.0\% to 16.5\%--a 64.5 percentage point drop occurring within the first 1,000 training steps. We propose mixed training strategies that interleave mathematical and NLI examples during training. Our results demonstrate that mixed training completely eliminates catastrophic forgetting while maintaining equivalent mathematical performance: the balanced 1:1 ratio achieves 12.0\% math accuracy (matching math-only) while preserving 86.2\% NLI accuracy. We systematically explore mixing ratios from 1:1 to 15:1, finding that even minimal NLI exposure (6.2\%) provides effective regularization. These findings demonstrate that specialization need not require forgetting general capabilities, with implications for scaling to larger models where mixed training may confer additional benefits beyond forgetting prevention.
>
---
#### [new 040] Scalable Frameworks for Real-World Audio-Visual Speech Recognition
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文面向真实场景下的音视频语音识别（AVSR）任务，解决噪声与视觉干扰导致性能下降的问题。提出三层可扩展框架：鲁棒音视频表征学习、自适应多模态架构设计、与大模型模块化集成，以提升系统在现实环境中的鲁棒性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.14083v1](https://arxiv.org/pdf/2512.14083v1)**

> **作者:** Sungnyun Kim
>
> **备注:** PhD Dissertation
>
> **摘要:** The practical deployment of Audio-Visual Speech Recognition (AVSR) systems is fundamentally challenged by significant performance degradation in real-world environments, characterized by unpredictable acoustic noise and visual interference. This dissertation posits that a systematic, hierarchical approach is essential to overcome these challenges, achieving the robust scalability at the representation, architecture, and system levels. At the representation level, we investigate methods for building a unified model that learns audio-visual features inherently robust to diverse real-world corruptions, thereby enabling generalization to new environments without specialized modules. To address architectural scalability, we explore how to efficiently expand model capacity while ensuring the adaptive and reliable use of multimodal inputs, developing a framework that intelligently allocates computational resources based on the input characteristics. Finally, at the system level, we present methods to expand the system's functionality through modular integration with large-scale foundation models, leveraging their powerful cognitive and generative capabilities to maximize final recognition accuracy. By systematically providing solutions at each of these three levels, this dissertation aims to build a next-generation, robust, and scalable AVSR system with high reliability in real-world applications.
>
---
#### [new 041] Hierarchical Multi-agent Large Language Model Reasoning for Autonomous Functional Materials Discovery
- **分类: cond-mat.mtrl-sci; cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文提出MASTER框架，属自主功能材料发现任务，旨在解决AI缺乏科学推理导致的发现不自主问题。通过多智能体大语言模型协同设计、执行与解读原子级模拟，实现高效、化学可解释的材料筛选，在CO吸附催化应用中减少90%模拟量。**

- **链接: [https://arxiv.org/pdf/2512.13930v1](https://arxiv.org/pdf/2512.13930v1)**

> **作者:** Samuel Rothfarb; Megan C. Davis; Ivana Matanovic; Baikun Li; Edward F. Holby; Wilton J. M. Kort-Kamp
>
> **备注:** Keywords: Multi-agent reasoning; Large language models; Active learning; AI-driven simulation; Materials discovery; Density functional theory; Surface chemistry
>
> **摘要:** Artificial intelligence is reshaping scientific exploration, but most methods automate procedural tasks without engaging in scientific reasoning, limiting autonomy in discovery. We introduce Materials Agents for Simulation and Theory in Electronic-structure Reasoning (MASTER), an active learning framework where large language models autonomously design, execute, and interpret atomistic simulations. In MASTER, a multimodal system translates natural language into density functional theory workflows, while higher-level reasoning agents guide discovery through a hierarchy of strategies, including a single agent baseline and three multi-agent approaches: peer review, triage-ranking, and triage-forms. Across two chemical applications, CO adsorption on Cu-surface transition metal (M) adatoms and on M-N-C catalysts, reasoning-driven exploration reduces required atomistic simulations by up to 90% relative to trial-and-error selection. Reasoning trajectories reveal chemically grounded decisions that cannot be explained by stochastic sampling or semantic bias. Altogether, multi-agent collaboration accelerates materials discovery and marks a new paradigm for autonomous scientific exploration.
>
---
#### [new 042] Grammar Search for Multi-Agent Systems
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属多智能体系统自动构建任务，旨在解决LLM自由搜索成本高、不可解释的问题。提出基于固定可组合组件的语法化搜索框架，在数学与问答任务中四/五项指标超越LLM方法，兼具高效性、模块性与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.14079v1](https://arxiv.org/pdf/2512.14079v1)**

> **作者:** Mayank Singh; Vikas Yadav; Shiva Krishna Reddy Malay; Shravan Nayak; Sai Rajeswar; Sathwik Tejaswi Madhusudhan; Eduardo Blanco
>
> **摘要:** Automatic search for Multi-Agent Systems has recently emerged as a key focus in agentic AI research. Several prior approaches have relied on LLM-based free-form search over the code space. In this work, we propose a more structured framework that explores the same space through a fixed set of simple, composable components. We show that, despite lacking the generative flexibility of LLMs during the candidate generation stage, our method outperforms prior approaches on four out of five benchmarks across two domains: mathematics and question answering. Furthermore, our method offers additional advantages, including a more cost-efficient search process and the generation of modular, interpretable multi-agent systems with simpler logic.
>
---
#### [new 043] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs
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
#### [new 044] Leveraging LLMs for Structured Data Extraction from Unstructured Patient Records
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于临床文本结构化任务，旨在解决EHR非结构化文本中人工提取特征耗时、易错的问题。作者构建了基于本地部署LLM的模块化框架，融合RAG与结构化输出，在合规基础设施上实现安全、可扩展的自动化特征提取，并验证其高准确率与纠错能力。**

- **链接: [https://arxiv.org/pdf/2512.13700v1](https://arxiv.org/pdf/2512.13700v1)**

> **作者:** Mitchell A. Klusty; Elizabeth C. Solie; Caroline N. Leach; W. Vaiden Logan; Lynnet E. Richey; John C. Gensel; David P. Szczykutowicz; Bryan C. McLellan; Emily B. Collier; Samuel E. Armstrong; V. K. Cody Bumgardner
>
> **备注:** 9 pages, 2 figures, 2 tables, submitted to AMIA 2026 Informatics Summit
>
> **摘要:** Manual chart review remains an extremely time-consuming and resource-intensive component of clinical research, requiring experts to extract often complex information from unstructured electronic health record (EHR) narratives. We present a secure, modular framework for automated structured feature extraction from clinical notes leveraging locally deployed large language models (LLMs) on institutionally approved, Health Insurance Portability and Accountability Act (HIPPA)-compliant compute infrastructure. This system integrates retrieval augmented generation (RAG) and structured response methods of LLMs into a widely deployable and scalable container to provide feature extraction for diverse clinical domains. In evaluation, the framework achieved high accuracy across multiple medical characteristics present in large bodies of patient notes when compared against an expert-annotated dataset and identified several annotation errors missed in manual review. This framework demonstrates the potential of LLM systems to reduce the burden of manual chart review through automated extraction and increase consistency in data capture, accelerating clinical research.
>
---
#### [new 045] Segmental Attention Decoding With Long Form Acoustic Encodings
- **分类: eess.AS; cs.CL**

- **简介: 该论文属语音识别任务，旨在解决注意力编码器-解码器（AED）模型在长音频输入下因位置编码失效导致的解码失序问题。提出四点改进：显式绝对位置注入、长上下文训练、分段拼接与语义分段对齐，弥合连续与分段音频的性能差距。**

- **链接: [https://arxiv.org/pdf/2512.14652v1](https://arxiv.org/pdf/2512.14652v1)**

> **作者:** Pawel Swietojanski; Xinwei Li; Mingbin Xu; Takaaki Hori; Dogan Can; Xiaodan Zhuang
>
> **备注:** 5 pages, 1 fig
>
> **摘要:** We address the fundamental incompatibility of attention-based encoder-decoder (AED) models with long-form acoustic encodings. AED models trained on segmented utterances learn to encode absolute frame positions by exploiting limited acoustic context beyond segment boundaries, but fail to generalize when decoding long-form segments where these cues vanish. The model loses ability to order acoustic encodings due to permutation invariance of keys and values in cross-attention. We propose four modifications: (1) injecting explicit absolute positional encodings into cross-attention for each decoded segment, (2) long-form training with extended acoustic context to eliminate implicit absolute position encoding, (3) segment concatenation to cover diverse segmentations needed during training, and (4) semantic segmentation to align AED-decoded segments with training segments. We show these modifications close the accuracy gap between continuous and segmented acoustic encodings, enabling auto-regressive use of the attention decoder.
>
---
#### [new 046] RePo: Language Models with Context Re-Positioning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出RePo方法，属大语言模型上下文建模任务。旨在解决固定位置编码导致的认知负荷过高的问题，通过可学习的非线性位置映射模块动态重定位上下文，提升长程依赖建模与噪声鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.14391v1](https://arxiv.org/pdf/2512.14391v1)**

> **作者:** Huayang Li; Tianyu Zhao; Richard Sproat
>
> **摘要:** In-context learning is fundamental to modern Large Language Models (LLMs); however, prevailing architectures impose a rigid and fixed contextual structure by assigning linear or constant positional indices. Drawing on Cognitive Load Theory (CLT), we argue that this uninformative structure increases extraneous cognitive load, consuming finite working memory capacity that should be allocated to deep reasoning and attention allocation. To address this, we propose RePo, a novel mechanism that reduces extraneous load via context re-positioning. Unlike standard approaches, RePo utilizes a differentiable module, $f_φ$, to assign token positions that capture contextual dependencies, rather than replying on pre-defined integer range. By continually pre-training on the OLMo-2 1B backbone, we demonstrate that RePo significantly enhances performance on tasks involving noisy contexts, structured data, and longer context length, while maintaining competitive performance on general short-context tasks. Detailed analysis reveals that RePo successfully allocate higher attention to distant but relevant information, assign positions in dense and non-linear space, and capture the intrinsic structure of the input context. Our code is available at https://github.com/SakanaAI/repo.
>
---
#### [new 047] Shakespeare, Entropy and Educated Monkeys
- **分类: math.HO; cs.CL; cs.IT**

- **简介: 该论文属信息论应用任务，旨在量化随机与约束随机文本生成的效率差异。通过熵估算，对比“纯随机”与“统计典型”（受英语语言模型约束）猴子生成莎士比亚文本所需时间，揭示语言统计规律对搜索空间的指数级压缩作用。**

- **链接: [https://arxiv.org/pdf/2512.11880v1](https://arxiv.org/pdf/2512.11880v1)**

> **作者:** Ioannis Kontoyiannis
>
> **摘要:** It has often been said, correctly, that a monkey forever randomly typing on a keyboard would eventually produce the complete works of William Shakespeare. Almost just as often it has been pointed out that this "eventually" is well beyond any conceivably relevant time frame. We point out that an educated monkey that still types at random but is constrained to only write "statistically typical" text, would produce any given passage in a much shorter time. Information theory gives a very simple way to estimate that time. For example, Shakespeare's phrase, Better three hours too soon than a minute too late, from The Merry Wives of Windsor, would take the educated monkey only 73 thousand years to produce, compared to the beyond-astronomical $2.7 \times 10^{63}$ years for the randomly typing one. Despite the obvious improvement, it would still take the educated monkey an unimaginably long $10^{42,277}$ years to produce all of Hamlet.
>
---
## 更新

#### [replaced 001] Holistic Utility Preference Learning for Listwise Alignment
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属大语言模型对齐任务，旨在解决现有 pairwise偏好学习（如DPO）无法建模多响应整体排序关系的问题。提出DRPO方法，将对齐建模为Learning-to-Rank任务，用可微diffNDCG损失和自适应排序策略优化列表级效用得分。**

- **链接: [https://arxiv.org/pdf/2410.18127v2](https://arxiv.org/pdf/2410.18127v2)**

> **作者:** Jiacong Zhou; Xianyun Wang; Min Zhang; Jun Yu
>
> **摘要:** Aligning large language models with human preferences is essential for improving interaction quality and safety by ensuring outputs better reflect human values. A promising strategy involves Reinforcement Learning from Human Feedback (RLHF), starting with collecting and ranking responses generated by a supervised fine-tuning model to refine alignment. Existing methods such as Direct Preference Optimization (DPO) focus on pairwise comparisons, categorizing responses into preferred and less preferred pairs and optimizing pairwise margins. However, this pairwise approach cannot capture the holistic ranking relationships among multiple responses or effectively leverage the rich preference information available in list-wise comparisons. To address this challenge, this paper introduces \underline{D}irect \underline{R}anking \underline{P}reference \underline{O}ptimization (DRPO), a novel method that views human preference alignment as a Learning-to-Rank (LTR) task. Unlike pairwise methods, DRPO optimizes the preference ranking of entire response lists by computing holistic utility scores through NDCG, a standard LTR metric. To enable end-to-end optimization with the non-differentiable NDCG, we propose diffNDCG loss, a differentiable approximation facilitated by a sorting network. Furthermore, we introduce a novel margin-based Adaptive Rank Policy Score to enhance the discriminative quality of generated responses. Extensive experiments have shown that DRPO outperforms existing methods, enhancing the quality of the generated responses.
>
---
#### [replaced 002] DIWALI: Diversity and Inclusivity aWare cuLture specific Items for India: Dataset and Assessment of LLMs for Cultural Text Adaptation in Indian Context
- **分类: cs.CL**

- **简介: 该论文聚焦文化文本适配任务，旨在解决LLMs在印度语境下文化对齐不足、缺乏细粒度评估数据的问题。作者构建了覆盖17个文化维度、36个次区域的8k条印度文化特异性数据集DIWALI，并结合自动与人工评估，揭示现有模型仅做表面化、选择性适配。**

- **链接: [https://arxiv.org/pdf/2509.17399v2](https://arxiv.org/pdf/2509.17399v2)**

> **作者:** Pramit Sahoo; Maharaj Brahma; Maunendra Sankar Desarkar
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** Large language models (LLMs) are widely used in various tasks and applications. However, despite their wide capabilities, they are shown to lack cultural alignment \citep{ryan-etal-2024-unintended, alkhamissi-etal-2024-investigating} and produce biased generations \cite{naous-etal-2024-beer} due to a lack of cultural knowledge and competence. Evaluation of LLMs for cultural awareness and alignment is particularly challenging due to the lack of proper evaluation metrics and unavailability of culturally grounded datasets representing the vast complexity of cultures at the regional and sub-regional levels. Existing datasets for culture specific items (CSIs) focus primarily on concepts at the regional level and may contain false positives. To address this issue, we introduce a novel CSI dataset for Indian culture, belonging to 17 cultural facets. The dataset comprises ~8k cultural concepts from 36 sub-regions. To measure the cultural competence of LLMs on a cultural text adaptation task, we evaluate the adaptations using the CSIs created, LLM as Judge, and human evaluations from diverse socio-demographic region. Furthermore, we perform quantitative analysis demonstrating selective sub-regional coverage and surface-level adaptations across all considered LLMs. Our dataset is available here: https://huggingface.co/datasets/nlip/DIWALI, project webpage https://nlip-lab.github.io/nlip/publications/diwali/, and our codebase with model outputs can be found here: https://github.com/pramitsahoo/culture-evaluation
>
---
#### [replaced 003] MatTools: Benchmarking Large Language Models for Materials Science Tools
- **分类: cond-mat.mtrl-sci; cs.CL; cs.DB**

- **简介: 该论文提出MatTools基准，用于评估大语言模型（LLM）在材料科学中调用物理计算工具的能力。任务属AI for Science中的工具调用评测，旨在解决LLM在材料模拟代码生成与安全执行方面的能力评估问题。工作包括构建QA基准（69K对）和真实工具使用基准（49任务），并开展多模型评测。**

- **链接: [https://arxiv.org/pdf/2505.10852v2](https://arxiv.org/pdf/2505.10852v2)**

> **作者:** Siyu Liu; Bo Hu; Beilin Ye; Jiamin Xu; David J. Srolovitz; Tongqi Wen
>
> **备注:** 27 pages, 23 figures
>
> **摘要:** Large language models (LLMs) are increasingly applied to materials science questions, including literature comprehension, property prediction, materials discovery and alloy design. At the same time, a wide range of physics-based computational approaches have been developed in which materials properties can be calculated. Here, we propose a benchmark application to evaluate the proficiency of LLMs to answer materials science questions through the generation and safe execution of codes based on such physics-based computational materials science packages. MatTools is built on two complementary components: a materials simulation tool question-answer (QA) benchmark and a real-world tool-usage benchmark. We designed an automated methodology to efficiently collect real-world materials science tool-use examples. The QA benchmark, derived from the pymatgen (Python Materials Genomics) codebase and documentation, comprises 69,225 QA pairs that assess the ability of an LLM to understand materials science tools. The real-world benchmark contains 49 tasks (138 subtasks) requiring the generation of functional Python code for materials property calculations. Our evaluation of diverse LLMs yields three key insights: (1)Generalists outshine specialists;(2)AI knows AI; and (3)Simpler is better. MatTools provides a standardized framework for assessing and improving LLM capabilities for materials science tool applications, facilitating the development of more effective AI systems for materials science and general scientific research.
>
---
#### [replaced 004] Can Finetuing LLMs on Small Human Samples Increase Heterogeneity, Alignment, and Belief-Action Coherence?
- **分类: cs.CL**

- **简介: 该论文属AI与社会科学交叉任务，探究小样本人类数据微调能否提升LLM在行为模拟中的异质性、对齐度和信念-行动一致性。作者基于信息披露实验，对比人类与微调/基线LLM响应，发现微调可改善前三者，但无法复现真实回归系数，故仍不适用于推断分析。**

- **链接: [https://arxiv.org/pdf/2511.21218v2](https://arxiv.org/pdf/2511.21218v2)**

> **作者:** Steven Wang; Kyle Hunt; Shaojie Tang; Kenneth Joseph
>
> **摘要:** There is ongoing debate about whether large language models (LLMs) can serve as substitutes for human participants in survey and experimental research. While recent work in fields such as marketing and psychology has explored the potential of LLM-based simulation, a growing body of evidence cautions against this practice: LLMs often fail to align with real human behavior, exhibiting limited diversity, systematic misalignment for minority subgroups, insufficient within-group variance, and discrepancies between stated beliefs and actions. This study examines an important and distinct question in this domain: whether fine-tuning on a small subset of human survey data, such as that obtainable from a pilot study, can mitigate these issues and yield realistic simulated outcomes. Using a behavioral experiment on information disclosure, we compare human and LLM-generated responses across multiple dimensions, including distributional divergence, subgroup alignment, belief-action coherence, and the recovery of regression coefficients. We find that fine-tuning on small human samples substantially improves heterogeneity, alignment, and belief-action coherence relative to the base model. However, even the best-performing fine-tuned models fail to reproduce the regression coefficients of the original study, suggesting that LLM-generated data remain unsuitable for replacing human participants in formal inferential analyses.
>
---
#### [replaced 005] Sliding Window Attention Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属LLM推理优化任务，旨在解决滑动窗口注意力（SWA）在全注意力预训练模型上直接应用导致的长上下文性能下降问题。作者提出SWAA方法，融合五种适配策略，无需重预训练即可高效恢复性能，提升推理速度最高达100%。**

- **链接: [https://arxiv.org/pdf/2512.10411v2](https://arxiv.org/pdf/2512.10411v2)**

> **作者:** Yijiong Yu; Jiale Liu; Qingyun Wu; Huazheng Wang; Ji Pei
>
> **摘要:** The self-attention mechanism in Transformer-based Large Language Models (LLMs) scales quadratically with input length, making long-context inference expensive. Sliding window attention (SWA) reduces this cost to linear complexity, but naively enabling complete SWA at inference-time for models pretrained with full attention (FA) causes severe long-context performance degradation due to training-inference mismatch. This makes us wonder: Can FA-pretrained LLMs be well adapted to SWA without pretraining? We investigate this by proposing Sliding Window Attention Adaptation (SWAA), a set of practical recipes that combine five methods for better adaptation: (1) applying SWA only during prefilling; (2) preserving "sink" tokens; (3) interleaving FA/SWA layers; (4) chain-of-thought (CoT); and (5) fine-tuning. Our experiments show that SWA adaptation is feasible while non-trivial: no single method suffices, yet specific synergistic combinations effectively recover the original long-context performance. We further analyze the performance-efficiency trade-offs of different SWAA configurations and provide recommended recipes for diverse scenarios, which can greatly and fundamentally accelerate LLM long-context inference speed by up to 100%. Our code is available at https://github.com/yuyijiong/sliding-window-attention-adaptation
>
---
#### [replaced 006] PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel
- **分类: cs.DC; cs.CL**

- **简介: 该论文面向LLM推理中的解码注意力优化任务，解决共享前缀下KV缓存重复加载与资源利用低效问题。提出PAT：基于前缀感知的多Tile注意力核，采用打包-前向-合并范式，提升内存带宽效率与硬件资源利用率。**

- **链接: [https://arxiv.org/pdf/2511.22333v2](https://arxiv.org/pdf/2511.22333v2)**

> **作者:** Jinjun Yi; Zhixin Zhao; Yitao Hu; Ke Yan; Weiwei Sun; Hao Wang; Laiping Zhao; Yuhao Zhang; Wenxin Li; Keqiu Li
>
> **备注:** Accepted by ASPLOS'26, code available at https://github.com/flashserve/PAT
>
> **摘要:** LLM serving is increasingly dominated by decode attention, which is a memory-bound operation due to massive KV cache loading from global memory. Meanwhile, real-world workloads exhibit substantial, hierarchical shared prefixes across requests (e.g., system prompts, tools/templates, RAG). Existing attention implementations fail to fully exploit prefix sharing: one-query-per-CTA execution repeatedly loads shared prefix KV cache, while one-size-fits-all tiling leaves on-chip resources idle and exacerbates bubbles for uneven KV lengths. These choices amplify memory bandwidth pressure and stall memory-bound decode attention. This paper introduces PAT, a prefix-aware attention kernel implementation for LLM decoding that organizes execution with a pack-forward-merge paradigm. PAT packs queries by shared prefix to reduce repeated memory accesses, runs a customized multi-tile kernel to achieve high resource efficiency. It further applies practical multi-stream forwarding and KV splitting to reduce resource bubbles. The final merge performs online softmax with negligible overhead. We implement PAT as an off-the-shelf plugin for vLLM. Evaluation on both real-world and synthetic workloads shows that PAT reduces attention latency by 53.5% on average and TPOT by 17.0-93.1% under the same configurations against state-of-the-art attention kernels. PAT's source code is publicly available at https://github.com/flashserve/PAT.
>
---
#### [replaced 007] Differentially Private Knowledge Distillation via Synthetic Text Generation
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文提出DistilDP方法，解决差分隐私下大语言模型压缩导致的效用严重下降问题。通过差分私有教师模型生成合成文本，结合硬标签、软标签及隐层对齐进行知识蒸馏，在保障ε=2隐私前提下显著提升学生模型性能。**

- **链接: [https://arxiv.org/pdf/2403.00932v3](https://arxiv.org/pdf/2403.00932v3)**

> **作者:** James Flemings; Murali Annavaram
>
> **摘要:** Large Language models (LLMs) are achieving state-of-the-art performance in many different downstream tasks. However, the increasing urgency of data privacy puts pressure on practitioners to train LLMs with Differential Privacy (DP) on private data. Concurrently, the exponential growth in parameter size of LLMs necessitates model compression before deployment of LLMs on resource-constrained devices or latency-sensitive applications. Differential privacy and model compression generally must trade off utility loss to achieve their objectives. Moreover, simultaneously applying both schemes can compound the utility degradation. To this end, we propose DistilDP: a novel differentially private knowledge distillation algorithm that exploits synthetic data generated by a differentially private teacher LLM. The knowledge of a teacher LLM is transferred onto the student in two ways: one way from the synthetic data itself -- the hard labels, and the other way by the output distribution of the teacher evaluated on the synthetic data -- the soft labels. Furthermore, if the teacher and student share a similar architectural structure, we can further distill knowledge by aligning the hidden representations between both. Our experimental results demonstrate that DistilDP can substantially improve the utility over existing baselines, at least $9.0$ PPL on the Big Patent dataset, with strong privacy parameters, $ε=2$. These promising results progress privacy-preserving compression of autoregressive LLMs. Our code can be accessed here: https://github.com/james-flemings/dp_compress.
>
---
#### [replaced 008] Semantic-Drive: Democratizing Long-Tail Data Curation via Open-Vocabulary Grounding and Neuro-Symbolic VLM Consensus
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文提出Semantic-Drive，解决自动驾驶中长尾安全事件（如异常闯入）数据难挖掘的问题。它采用本地化、神经符号融合框架：先用YOLOE进行开放词汇语义定位，再通过多模型共识的推理型VLM做细粒度场景分析，在保护隐私前提下显著提升召回率与风险评估精度。**

- **链接: [https://arxiv.org/pdf/2512.12012v2](https://arxiv.org/pdf/2512.12012v2)**

> **作者:** Antonio Guillen-Perez
>
> **摘要:** The development of robust Autonomous Vehicles (AVs) is bottlenecked by the scarcity of "Long-Tail" training data. While fleets collect petabytes of video logs, identifying rare safety-critical events (e.g., erratic jaywalking, construction diversions) remains a manual, cost-prohibitive process. Existing solutions rely on coarse metadata search, which lacks precision, or cloud-based VLMs, which are privacy-invasive and expensive. We introduce Semantic-Drive, a local-first, neuro-symbolic framework for semantic data mining. Our approach decouples perception into two stages: (1) Symbolic Grounding via a real-time open-vocabulary detector (YOLOE) to anchor attention, and (2) Cognitive Analysis via a Reasoning VLM that performs forensic scene analysis. To mitigate hallucination, we implement a "System 2" inference-time alignment strategy, utilizing a multi-model "Judge-Scout" consensus mechanism. Benchmarked on the nuScenes dataset against the Waymo Open Dataset (WOD-E2E) taxonomy, Semantic-Drive achieves a Recall of 0.966 (vs. 0.475 for CLIP) and reduces Risk Assessment Error by 40% ccompared to the best single scout models. The system runs entirely on consumer hardware (NVIDIA RTX 3090), offering a privacy-preserving alternative to the cloud.
>
---
#### [replaced 009] Retrieval Enhanced Feedback via In-context Neural Error-book
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属多模态推理任务，旨在解决MLLMs中错误分析与反馈缺乏结构化的问题。提出REFINE框架，通过三个系统性查询（Feed-Target/Check/Path）构建结构化神经错题本，实现高效检索增强反馈，提升推理准确性、效率与泛化性。**

- **链接: [https://arxiv.org/pdf/2508.16313v5](https://arxiv.org/pdf/2508.16313v5)**

> **作者:** Jongyeop Hyun; Bumsoo Kim
>
> **备注:** Accepted at EMNLP 2025 main
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have significantly improved reasoning capabilities, with in-context learning (ICL) emerging as a key technique for adaptation without retraining. While previous works have focused on leveraging correct examples, recent research highlights the importance of learning from errors to enhance performance. However, existing methods lack a structured framework for analyzing and mitigating errors, particularly in Multimodal Large Language Models (MLLMs), where integrating visual and textual inputs adds complexity. To address this issue, we propose REFINE: Retrieval-Enhanced Feedback via In-context Neural Error-book, a teacher-student framework that systematically structures errors and provides targeted feedback. REFINE introduces three systematic queries to construct structured feedback -- Feed-Target, Feed-Check, and Feed-Path -- to enhance multimodal reasoning by prioritizing relevant visual information, diagnosing critical failure points, and formulating corrective actions. Unlike prior approaches that rely on redundant retrievals, REFINE optimizes structured feedback retrieval, improving inference efficiency, token usage, and scalability. Our results demonstrate substantial speedup, reduced computational costs, and successful generalization, highlighting REFINE's potential for enhancing multimodal reasoning.
>
---
#### [replaced 010] Optimizing Large Language Models for ESG Activity Detection in Financial Texts
- **分类: cs.AI; cs.CE; cs.CL; cs.CY; cs.IR**

- **简介: 该论文面向ESG活动检测任务，旨在解决通用大模型在金融文本中识别环境类活动准确率低、领域数据稀缺的问题。作者构建了含1325条样本的ESG-Activities基准数据集（基于欧盟ESG分类），结合真实与合成数据微调开源LLM，显著提升分类性能。**

- **链接: [https://arxiv.org/pdf/2502.21112v2](https://arxiv.org/pdf/2502.21112v2)**

> **作者:** Mattia Birti; Andrea Maurino; Francesco Osborne
>
> **备注:** Published in the Proceedings of the ACM International Conference on AI in Finance (ICAIF). ACM version
>
> **摘要:** The integration of Environmental, Social, and Governance (ESG) factors into corporate decision-making is a fundamental aspect of sustainable finance. However, ensuring that business practices align with evolving regulatory frameworks remains a persistent challenge. AI-driven solutions for automatically assessing the alignment of sustainability reports and non-financial disclosures with specific ESG activities could greatly support this process. Yet, this task remains complex due to the limitations of general-purpose Large Language Models (LLMs) in domain-specific contexts and the scarcity of structured, high-quality datasets. In this paper, we investigate the ability of current-generation LLMs to identify text related to environmental activities. Furthermore, we demonstrate that their performance can be significantly enhanced through fine-tuning on a combination of original and synthetically generated data. To this end, we introduce ESG-Activities, a benchmark dataset containing 1,325 labelled text segments classified according to the EU ESG taxonomy. Our experimental results show that fine-tuning on ESG-Activities significantly enhances classification accuracy, with open models such as Llama 7B and Gemma 7B outperforming large proprietary solutions in specific configurations. These findings have important implications for financial analysts, policymakers, and AI researchers seeking to enhance ESG transparency and compliance through advanced natural language processing techniques.
>
---
#### [replaced 011] TaP: A Taxonomy-Guided Framework for Automated and Scalable Preference Data Generation
- **分类: cs.CL**

- **简介: 该论文属于大语言模型数据构建任务，旨在解决多语言偏好数据集稀缺、人工构建成本高的问题。提出TaP框架，基于结构化分类法自动、可扩展地生成高质量多语言偏好数据，并验证其在监督与偏好微调中显著优于现有开源数据集。**

- **链接: [https://arxiv.org/pdf/2506.23979v2](https://arxiv.org/pdf/2506.23979v2)**

> **作者:** Renren Jin; Tianhao Shen; Xinwei Wu; Dan Shi; Haoran Sun; Yuqi Ren; Wuwei Huang; Quandong Wang; Wei Liu; Jian Luan; Bin Wang; Deyi Xiong
>
> **备注:** 34 pages, 16 tables, 11 figures
>
> **摘要:** Conducting supervised fine-tuning and preference fine-tuning on large language models (LLMs) requires high-quality datasets to improve their ability to follow instructions and align with human preferences and values. However, constructing such datasets is resource-intensive, and most available datasets for supervised and preference fine-tuning are in English. To address these challenges, we propose the \underline{\textbf{Ta}}xonomy-Guided \underline{\textbf{P}}reference Data Generation (TaP) framework, which facilitates automated and scalable construction of preference datasets across various languages. TaP is grounded in a structured taxonomy that allows fine-grained control over dataset composition, thereby ensuring both diversity and comprehensive coverage. We employ TaP-generated datasets to perform supervised and preference fine-tuning on various LLMs. Experimental results demonstrate that LLMs trained on TaP-generated datasets outperform those trained on existing open-source datasets. Remarkably, LLMs trained on TaP-generated datasets surpass the performance of those trained on an open-source dataset that is 180 times larger.
>
---
#### [replaced 012] A stylometric analysis of speaker attribution from speech transcripts
- **分类: cs.CL**

- **简介: 该论文研究语音转录文本的说话人归属任务，解决语音伪装或TTS导致声学特征失效时的身份识别问题。提出StyloSpeaker方法，融合多层级语言风格特征，在不同格式和话题控制的转录文本上评估性能，并对比神经模型、分析关键区分特征。**

- **链接: [https://arxiv.org/pdf/2512.13667v2](https://arxiv.org/pdf/2512.13667v2)**

> **作者:** Cristina Aggazzotti; Elizabeth Allyn Smith
>
> **备注:** added acknowledgments
>
> **摘要:** Forensic scientists often need to identify an unknown speaker or writer in cases such as ransom calls, covert recordings, alleged suicide notes, or anonymous online communications, among many others. Speaker recognition in the speech domain usually examines phonetic or acoustic properties of a voice, and these methods can be accurate and robust under certain conditions. However, if a speaker disguises their voice or employs text-to-speech software, vocal properties may no longer be reliable, leaving only their linguistic content available for analysis. Authorship attribution methods traditionally use syntactic, semantic, and related linguistic information to identify writers of written text (authorship attribution). In this paper, we apply a content-based authorship approach to speech that has been transcribed into text, using what a speaker says to attribute speech to individuals (speaker attribution). We introduce a stylometric method, StyloSpeaker, which incorporates character, word, token, sentence, and style features from the stylometric literature on authorship, to assess whether two transcripts were produced by the same speaker. We evaluate this method on two types of transcript formatting: one approximating prescriptive written text with capitalization and punctuation and another normalized style that removes these conventions. The transcripts' conversation topics are also controlled to varying degrees. We find generally higher attribution performance on normalized transcripts, except under the strongest topic control condition, in which overall performance is highest. Finally, we compare this more explainable stylometric model to black-box neural approaches on the same data and investigate which stylistic features most effectively distinguish speakers.
>
---
#### [replaced 013] CADDesigner: Conceptual Design of CAD Models Based on General-Purpose Agent
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出CADDesigner代理，解决CAD概念设计门槛高、效率低的问题。它融合文本与草图输入，通过交互式需求分析和迭代视觉反馈，基于ECIP范式生成高质量CAD建模代码，并构建知识库实现持续优化。**

- **链接: [https://arxiv.org/pdf/2508.01031v4](https://arxiv.org/pdf/2508.01031v4)**

> **作者:** Fengxiao Fan; Jingzhe Ni; Xiaolong Yin; Sirui Wang; Xingyu Lu; Qiang Zou; Ruofeng Tong; Min Tang; Peng Du
>
> **摘要:** Computer Aided Design (CAD) plays a pivotal role in industrial manufacturing but typically requires a high level of expertise from designers. To lower the entry barrier and improve design efficiency, we present an agent for CAD conceptual design powered by large language models (LLMs). The agent accepts both textual descriptions and sketches as input, engaging in interactive dialogue with users to refine and clarify design requirements through comprehensive requirement analysis. Built upon a novel Explicit Context Imperative Paradigm (ECIP), the agent generates high-quality CAD modeling code. During the generation process, the agent incorporates iterative visual feedback to improve model quality. Generated design cases are stored in a structured knowledge base, enabling continuous improvement of the agent's code generation capabilities. Experimental results demonstrate that our method achieves state-of-the-art performance in CAD code generation.
>
---
#### [replaced 014] Listening Between the Lines: Decoding Podcast Narratives with Language Modeling
- **分类: cs.CL; cs.SI**

- **简介: 该论文属自然语言处理中的叙事分析任务，旨在解决现有大模型难以准确识别播客中隐含叙事框架的问题。作者提出细粒度BERT微调方法，将叙事框架与对话实体绑定，并关联话题以揭示“内容—框架”系统关系。**

- **链接: [https://arxiv.org/pdf/2511.05310v2](https://arxiv.org/pdf/2511.05310v2)**

> **作者:** Shreya Gupta; Ojasva Saxena; Arghodeep Nandi; Sarah Masud; Kiran Garimella; Tanmoy Chakraborty
>
> **备注:** 10 pages, 6 Figures, 5 Tables. Under review
>
> **摘要:** Podcasts have become a central arena for shaping public opinion, making them a vital source for understanding contemporary discourse. Their typically unscripted, multi-themed, and conversational style offers a rich but complex form of data. To analyze how podcasts persuade and inform, we must examine their narrative structures -- specifically, the narrative frames they employ. The fluid and conversational nature of podcasts presents a significant challenge for automated analysis. We show that existing large language models, typically trained on more structured text such as news articles, struggle to capture the subtle cues that human listeners rely on to identify narrative frames. As a result, current approaches fall short of accurately analyzing podcast narratives at scale. To solve this, we develop and evaluate a fine-tuned BERT model that explicitly links narrative frames to specific entities mentioned in the conversation, effectively grounding the abstract frame in concrete details. Our approach then uses these granular frame labels and correlates them with high-level topics to reveal broader discourse trends. The primary contributions of this paper are: (i) a novel frame-labeling methodology that more closely aligns with human judgment for messy, conversational data, and (ii) a new analysis that uncovers the systematic relationship between what is being discussed (the topic) and how it is being presented (the frame), offering a more robust framework for studying influence in digital media.
>
---
#### [replaced 015] Temporal Tokenization Strategies for Event Sequence Modeling with Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究事件序列建模中连续时间表示问题，属时序建模任务。针对LLMs难以处理连续时间的挑战，首次实证比较五种时间分词策略，在多分布真实数据上评估性能，发现需依数据统计特性（如偏态/混合模态）选择适配分词方法。**

- **链接: [https://arxiv.org/pdf/2512.13618v2](https://arxiv.org/pdf/2512.13618v2)**

> **作者:** Zefang Liu; Nam H. Nguyen; Yinzhu Quan; Shi-Xiong Zhang
>
> **摘要:** Representing continuous time is a critical and under-explored challenge in modeling temporal event sequences with large language models (LLMs). Various strategies like byte-level representations or calendar tokens have been proposed. However, the optimal approach remains unclear, especially given the diverse statistical distributions of real-world event data, which range from smooth log-normal to discrete, spiky patterns. This paper presents the first empirical study of temporal tokenization for event sequences, comparing distinct encoding strategies: naive numeric strings, high-precision byte-level representations, human-semantic calendar tokens, classic uniform binning, and adaptive residual scalar quantization. We evaluate these strategies by fine-tuning LLMs on real-world datasets that exemplify these diverse distributions. Our analysis reveals that no single strategy is universally superior; instead, prediction performance depends heavily on aligning the tokenizer with the data's statistical properties, with log-based strategies excelling on skewed distributions and human-centric formats proving robust for mixed modalities.
>
---
#### [replaced 016] Love First, Know Later: Persona-Based Romantic Compatibility Through LLM Text World Engines
- **分类: cs.HC; cs.CL; cs.LG**

- **简介: 该论文提出“先恋爱，后了解”范式，用LLM作为文本世界引擎模拟人格化交互，将匹配转化为基于模拟信号的奖励建模问题，解决传统静态画像匹配不准的问题；理论证明收敛性，实证验证于速配与离婚预测。**

- **链接: [https://arxiv.org/pdf/2512.11844v2](https://arxiv.org/pdf/2512.11844v2)**

> **作者:** Haoyang Shang; Zhengyang Yan; Xuan Liu
>
> **备注:** NeurIPS 2025 Workshop: First Workshop on LLM Persona Modeling (Oral)
>
> **摘要:** We propose Love First, Know Later: a paradigm shift in computational matching that simulates interactions first, then assesses compatibility. Instead of comparing static profiles, our framework leverages LLMs as text world engines that operate in dual capacity-as persona-driven agents following behavioral policies and as the environment modeling interaction dynamics. We formalize compatibility assessment as a reward-modeling problem: given observed matching outcomes, we learn to extract signals from simulations that predict human preferences. Our key insight is that relationships hinge on responses to critical moments-we translate this observation from relationship psychology into mathematical hypotheses, enabling effective simulation. Theoretically, we prove that as LLM policies better approximate human behavior, the induced matching converges to optimal stable matching. Empirically, we validate on speed dating data for initial chemistry and divorce prediction for long-term stability. This paradigm enables interactive, personalized matching systems where users iteratively refine their agents, unlocking future possibilities for transparent and interactive compatibility assessment.
>
---
#### [replaced 017] Rethinking the Reliability of Multi-agent System: A Perspective from Byzantine Fault Tolerance
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文属多智能体系统可靠性研究任务，旨在解决LLM代理在拜占庭故障下的可靠性问题。作者提出CP-WBFT共识机制，利用LLM的反思与判别能力，通过置信探针加权信息流，在高故障率下显著提升MAS鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.10400v2](https://arxiv.org/pdf/2511.10400v2)**

> **作者:** Lifan Zheng; Jiawei Chen; Qinghong Yin; Jingyuan Zhang; Xinyi Zeng; Yu Tian
>
> **摘要:** Ensuring the reliability of agent architectures and effectively identifying problematic agents when failures occur are crucial challenges in multi-agent systems (MAS). Advances in large language models (LLMs) have established LLM-based agents as a major branch of MAS, enabling major breakthroughs in complex problem solving and world modeling. However, the reliability implications of this shift remain largely unexplored. i.e., whether substituting traditional agents with LLM-based agents can effectively enhance the reliability of MAS. In this work, we investigate and quantify the reliability of LLM-based agents from the perspective of Byzantine fault tolerance. We observe that LLM-based agents demonstrate stronger skepticism when processing erroneous message flows, a characteristic that enables them to outperform traditional agents across different topological structures. Motivated by the results of the pilot experiment, we design CP-WBFT, a confidence probe-based weighted Byzantine Fault Tolerant consensus mechanism to enhance the stability of MAS with different topologies. It capitalizes on the intrinsic reflective and discriminative capabilities of LLMs by employing a probe-based, weighted information flow transmission method to improve the reliability of LLM-based agents. Extensive experiments demonstrate that CP-WBFT achieves superior performance across diverse network topologies under extreme Byzantine conditions (85.7\% fault rate). Notably, our approach surpasses traditional methods by attaining remarkable accuracy on various topologies and maintaining strong reliability in both mathematical reasoning and safety assessment tasks.
>
---
#### [replaced 018] Inverse Scaling in Test-Time Compute
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究大推理模型（LRMs）在增加测试时计算量（如推理步数）时性能反而下降的“逆向缩放”现象。它构建四类评测任务，识别五种典型失败模式，揭示延长推理可能加剧分心、过拟合、错误关联与风险行为等问题，强调需多长度评估以提升推理鲁棒性。**

- **链接: [https://arxiv.org/pdf/2507.14417v2](https://arxiv.org/pdf/2507.14417v2)**

> **作者:** Aryo Pradipta Gema; Alexander Hägele; Runjin Chen; Andy Arditi; Jacob Goldman-Wetzler; Kit Fraser-Taliente; Henry Sleight; Linda Petrini; Julian Michael; Beatrice Alex; Pasquale Minervini; Yanda Chen; Joe Benton; Ethan Perez
>
> **备注:** Published in TMLR (12/2025; Featured Certification; J2C Certification), 78 pages
>
> **摘要:** We construct evaluation tasks where extending the reasoning length of Large Reasoning Models (LRMs) deteriorates performance, exhibiting an inverse scaling relationship between test-time compute and accuracy. Our evaluation tasks span four categories: simple counting tasks with distractors, regression tasks with spurious features, deduction tasks with constraint tracking, and advanced AI risks. We identify five distinct failure modes when models reason for longer: 1) Claude models become increasingly distracted by irrelevant information; 2) OpenAI o-series models resist distractors but overfit to problem framings; 3) models shift from reasonable priors to spurious correlations; 4) all models show difficulties in maintaining focus on complex deductive tasks; and 5) extended reasoning may amplify concerning behaviors, with Claude Sonnet 4 showing increased expressions of self-preservation. These findings suggest that while test-time compute scaling remains promising for improving model capabilities, it may inadvertently reinforce problematic reasoning patterns. Our results demonstrate the importance of evaluating models across diverse reasoning lengths to identify and address these failure modes in LRMs.
>
---
#### [replaced 019] MentraSuite: Post-Training Large Language Models for Mental Health Reasoning and Assessment
- **分类: cs.CL**

- **简介: 该论文属心理健康领域可信AI任务，旨在解决现有LLM在精神健康推理中不完整、不一致、不 grounded 的问题。提出MentraSuite框架，含新基准MentraBench、后训练模型Mindora（融合SFT-RL与不一致性奖励）及高质量推理轨迹生成方法。**

- **链接: [https://arxiv.org/pdf/2512.09636v2](https://arxiv.org/pdf/2512.09636v2)**

> **作者:** Mengxi Xiao; Kailai Yang; Pengde Zhao; Enze Zhang; Ziyan Kuang; Zhiwei Liu; Weiguang Han; Shu Liao; Lianting Huang; Jinpeng Hu; Min Peng; Qianqian Xie; Sophia Ananiadou
>
> **摘要:** Mental health disorders affect hundreds of millions globally, and the Web now serves as a primary medium for accessing support, information, and assessment. Large language models (LLMs) offer scalable and accessible assistance, yet their deployment in mental-health settings remains risky when their reasoning is incomplete, inconsistent, or ungrounded. Existing psychological LLMs emphasize emotional understanding or knowledge recall but overlook the step-wise, clinically aligned reasoning required for appraisal, diagnosis, intervention planning, abstraction, and verification. To address these issues, we introduce MentraSuite, a unified framework for advancing reliable mental-health reasoning. We propose MentraBench, a comprehensive benchmark spanning five core reasoning aspects, six tasks, and 13 datasets, evaluating both task performance and reasoning quality across five dimensions: conciseness, coherence, hallucination avoidance, task understanding, and internal consistency. We further present Mindora, a post-trained model optimized through a hybrid SFT-RL framework with an inconsistency-detection reward to enforce faithful and coherent reasoning. To support training, we construct high-quality trajectories using a novel reasoning trajectory generation strategy, that strategically filters difficult samples and applies a structured, consistency-oriented rewriting process to produce concise, readable, and well-balanced trajectories. Across 20 evaluated LLMs, Mindora achieves the highest average performance on MentraBench and shows remarkable performances in reasoning reliability, demonstrating its effectiveness for complex mental-health scenarios.
>
---
#### [replaced 020] RepoTransBench: A Real-World Multilingual Benchmark for Repository-Level Code Translation
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文面向仓库级代码翻译任务，旨在解决现有基准仅支持细粒度（如函数/文件级）翻译、脱离真实需求的问题。作者构建了多语言真实仓库基准RepoTransBench（1897个仓库、13种语言对），提出代理框架RepoTransAgent，并揭示当前LLM在此任务上效果有限（最高仅32.8%成功率）。**

- **链接: [https://arxiv.org/pdf/2412.17744v2](https://arxiv.org/pdf/2412.17744v2)**

> **作者:** Yanli Wang; Yanlin Wang; Suiquan Wang; Daya Guo; Jiachi Chen; John Grundy; Xilin Liu; Yuchi Ma; Mingzhi Mao; Hongyu Zhang; Zibin Zheng
>
> **摘要:** Repository-level code translation refers to translating an entire code repository from one programming language to another while preserving the functionality of the source repository. Many benchmarks have been proposed to evaluate the performance of such code translators. However, previous benchmarks mostly provide fine-grained samples, focusing at either code snippet, function, or file-level code translation. Such benchmarks do not accurately reflect real-world demands, where entire repositories often need to be translated, involving longer code length and more complex functionalities. To address this gap, we propose a new benchmark, named RepoTransBench, which is a real-world multilingual repository-level code translation benchmark featuring 1,897 real-world repository samples across 13 language pairs with automatically executable test suites. Besides, we introduce RepoTransAgent, a general agent framework to perform repository-level code translation. We evaluate both our benchmark's challenges and agent's effectiveness using several methods and backbone LLMs, revealing that repository-level translation remains challenging, where the best-performing method achieves only a 32.8% success rate. Furthermore, our analysis reveals that translation difficulty varies significantly by language pair direction, with dynamic-to-static language translation being much more challenging than the reverse direction (achieving below 10% vs. static-to-dynamic at 45-63%). Finally, we conduct a detailed error analysis and highlight current LLMs' deficiencies in repository-level code translation, which could provide a reference for further improvements. We provide the code and data at https://github.com/DeepSoftwareAnalytics/RepoTransBench.
>
---
#### [replaced 021] Question Answering Over Spatio-Temporal Knowledge Graph
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于时空知识图谱问答（STKGQA）任务，旨在解决现有方法缺乏时空联合建模能力及高质量基准数据的问题。作者构建首个时空问答数据集STQAD，并提出STCQA模型，通过联合嵌入时空特征与约束感知推理，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2402.11542v2](https://arxiv.org/pdf/2402.11542v2)**

> **作者:** Xinbang Dai; Huiying Li; Nan Hu; Yongrui Chen; Rihui Jin; Huikang Hu; Guilin Qi
>
> **备注:** 34 pages, 10 figures. The paper has been accepted by Knowledge-Based Systems
>
> **摘要:** Spatio-temporal knowledge graphs (STKGs) enhance traditional KGs by integrating temporal and spatial annotations, enabling precise reasoning over questions with spatio-temporal dependencies. Despite their potential, research on spatio-temporal knowledge graph question answering (STKGQA) remains limited. This is primarily due to the lack of datasets that simultaneously contain spatio-temporal information, as well as methods capable of handling implicit spatio-temporal reasoning. To bridge this gap, we introduce the spatio-temporal question answering dataset (STQAD), the first comprehensive benchmark comprising 10,000 natural language questions that require both temporal and spatial reasoning. STQAD is constructed with real-world facts containing spatio-temporal information, ensuring that the dataset reflects practical scenarios. Furthermore, our experiments reveal that existing KGQA methods underperform on STQAD, primarily due to their inability to model spatio-temporal interactions. To address this, we propose the spatio-temporal complex question answering (STCQA) method, which jointly embeds temporal and spatial features into KG representations and dynamically filters answers through constraint-aware reasoning. STCQA achieves state-of-the-art performance, significantly outperforming existing baselines. Our work not only provides a valuable resource for future research but also advances the field by offering a robust baseline for answering complex spatio-temporal questions.
>
---
#### [replaced 022] A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属联邦强化学习中RLHF对齐任务，旨在解决LLM在联邦环境下难以兼顾多元人类偏好与公平性的问题。提出系统评估框架，对比多种奖励聚合方法，并设计自适应权重机制，在不访问原始数据前提下提升公平性与对齐质量。**

- **链接: [https://arxiv.org/pdf/2512.08786v2](https://arxiv.org/pdf/2512.08786v2)**

> **作者:** Mahmoud Srewa; Tianyu Zhao; Salma Elmalaki
>
> **备注:** This paper is accepted at the NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle
>
> **摘要:** This paper addresses the challenge of aligning large language models (LLMs) with diverse human preferences within federated learning (FL) environments, where standard methods often fail to adequately represent diverse viewpoints. We introduce a comprehensive evaluation framework that systematically assesses the trade-off between alignment quality and fairness when using different aggregation strategies for human preferences. In our federated setting, each group locally evaluates rollouts and produces reward signals, and the server aggregates these group-level rewards without accessing any raw data. Specifically, we evaluate standard reward aggregation techniques (min, max, and average) and introduce a novel adaptive scheme that dynamically adjusts preference weights based on a group's historical alignment performance. Our experiments on question-answering (Q/A) tasks using a PPO-based RLHF pipeline demonstrate that our adaptive approach consistently achieves superior fairness while maintaining competitive alignment scores. This work offers a robust methodology for evaluating LLM behavior across diverse populations and provides a practical solution for developing truly pluralistic and fairly aligned models.
>
---
#### [replaced 023] Analysing Knowledge Construction in Online Learning: Adapting the Interaction Analysis Model for Unstructured Large-Scale Discourse
- **分类: cs.CL**

- **简介: 该论文属教育数据挖掘任务，旨在解决大规模非结构化在线学习文本中知识构建行为难以自动识别的问题。作者改进交互分析模型，设计四类知识构建编码体系，构建高质量标注数据集，并训练DeBERTa-v3-large模型实现高精度自动分类，验证了理论驱动的半自动化分析可行性。**

- **链接: [https://arxiv.org/pdf/2510.19858v3](https://arxiv.org/pdf/2510.19858v3)**

> **作者:** Jindi Wang; Yidi Zhang; Zhaoxing Li; Pedro Bem Haja; Ioannis Ivrissimtzis; Zichen Zhao; Sebastian Stein
>
> **摘要:** The rapid expansion of online courses and social media has generated large volumes of unstructured learner-generated text. Understanding how learners construct knowledge in these spaces is crucial for analysing learning processes, informing content design, and providing feedback at scale. However, existing approaches typically rely on manual coding of well-structured discussion forums, which does not scale to the fragmented discourse found in online learning. This study proposes and validates a framework that combines a codebook inspired by the Interaction Analysis Model with an automated classifier to enable large-scale analysis of knowledge construction in unstructured online discourse. We adapt four comment-level categories of knowledge construction: Non-Knowledge Construction, Share, Explore, and Integrate. Three trained annotators coded a balanced sample of 20,000 comments from YouTube education channels. The codebook demonstrated strong reliability, with Cohen's kappa = 0.79 on the main dataset and 0.85--0.93 across four additional educational domains. For automated classification, bag-of-words baselines were compared with transformer-based language models using 10-fold cross-validation. A DeBERTa-v3-large model achieved the highest macro-averaged F1 score (0.841), outperforming all baselines and other transformer models. External validation on four domains yielded macro-F1 above 0.705, with stronger transfer in medicine and programming, where discourse was more structured and task-focused, and weaker transfer in language and music, where comments were more varied and context-dependent. Overall, the study shows that theory-driven, semi-automated analysis of knowledge construction at scale is feasible, enabling the integration of knowledge-construction indicators into learning analytics and the design of online learning environments.
>
---
#### [replaced 024] AugServe: Adaptive Request Scheduling for Augmented Large Language Model Inference Serving
- **分类: cs.CL**

- **简介: 该论文属AI系统优化任务，旨在解决增强型大语言模型（AugLLM）推理服务中因FCFS调度和静态批处理导致的队列延迟高、有效吞吐低问题。提出AugServe框架，通过两阶段自适应请求调度与动态令牌批处理，显著提升有效吞吐并降低TTFT。**

- **链接: [https://arxiv.org/pdf/2512.04013v2](https://arxiv.org/pdf/2512.04013v2)**

> **作者:** Ying Wang; Zhen Jin; Jiexiong Xu; Wenhai Lin; Yiquan Chen; Wenzhi Chen
>
> **摘要:** As augmented large language models (LLMs) with external tools become increasingly popular in web applications, improving augmented LLM inference serving efficiency and optimizing service-level objectives (SLOs) are critical for enhancing user experience. To achieve this, inference systems must maximize request handling within latency constraints, referred to as increasing effective throughput. However, existing systems face two major challenges: (i) reliance on first-come-first-served (FCFS) scheduling causes severe head-of-line blocking, leading to queuing delays exceeding the SLOs for many requests; and (ii) static batch token limit, which fails to adapt to fluctuating loads and hardware conditions. Both of these factors degrade effective throughput and service quality. This paper presents AugServe, an efficient inference framework designed to reduce queueing latency and enhance effective throughput for augmented LLM inference services. The core idea of AugServe is a two-stage adaptive request scheduling strategy. Specifically, AugServe combines the inference features of augmented LLM requests to optimize the order of scheduling decisions (stage I). These decisions are continuously refined with runtime information (stage II), adapting to both request characteristics and system capabilities. In addition, AugServe dynamically adjusts the token batching mechanism based on hardware status and real-time load, further enhancing throughput performance. Experimental results show that AugServe achieves 4.7x and 3.3x higher effective throughput than vLLM and InferCept, while reducing time-to-first-token (TTFT) by up to 96.3% and 95.0%, respectively.
>
---
#### [replaced 025] Enhancing Long-term RAG Chatbots with Psychological Models of Memory Importance and Forgetting
- **分类: cs.CL; cs.AI**

- **简介: 该论文属长时对话RAG任务，旨在解决对话记忆增长导致检索精度下降的问题。提出LUFY方法，借鉴心理学记忆模型，仅保留<10%高情绪唤醒记忆，显著提升长期交互体验。**

- **链接: [https://arxiv.org/pdf/2409.12524v2](https://arxiv.org/pdf/2409.12524v2)**

> **作者:** Ryuichi Sumida; Koji Inoue; Tatsuya Kawahara
>
> **备注:** 37 pages, accepted and published in Dialogue & Discourse 16(2) (2025)
>
> **摘要:** While Retrieval-Augmented Generation (RAG) has shown promise in enhancing long-term conversations, the increasing memory load as conversations progress degrades retrieval accuracy. Drawing on psychological insights, we propose LUFY, a simple yet effective method that focuses on emotionally arousing memories and retains less than 10% of the conversation. In the user experiment, participants interacted with three types of RAG chatbots, each for 2 hours over 4 sessions, marking the most extensive assessment of a chatbot's long-term capabilities to date -- more than four times longer than any existing benchmark. The results demonstrate that prioritizing arousing memories while forgetting the majority of the conversation significantly enhances user experience. This study pushes the frontier of long-term conversations and highlights the importance of forgetting unimportant parts of conversations. Code and Dataset: https://github.com/ryuichi-sumida/LUFY
>
---
#### [replaced 026] Estimating Privacy Leakage of Augmented Contextual Knowledge in Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属隐私评估任务，旨在量化语言模型在使用外部上下文时导致的隐私泄露风险。提出“上下文影响”指标，基于差分隐私分离参数知识与上下文贡献，识别分布外上下文引发的泄露，并分析模型规模等因素的影响。**

- **链接: [https://arxiv.org/pdf/2410.03026v3](https://arxiv.org/pdf/2410.03026v3)**

> **作者:** James Flemings; Bo Jiang; Wanrong Zhang; Zafar Takhirov; Murali Annavaram
>
> **摘要:** Language models (LMs) rely on their parametric knowledge augmented with relevant contextual knowledge for certain tasks, such as question answering. However, the contextual knowledge can contain private information that may be leaked when answering queries, and estimating this privacy leakage is not well understood. A straightforward approach of directly comparing an LM's output to the contexts can overestimate the privacy risk, since the LM's parametric knowledge might already contain the augmented contextual knowledge. To this end, we introduce *context influence*, a metric that builds on differential privacy, a widely-adopted privacy notion, to estimate the privacy leakage of contextual knowledge during decoding. Our approach effectively measures how each subset of the context influences an LM's response while separating the specific parametric knowledge of the LM. Using our context influence metric, we demonstrate that context privacy leakage occurs when contextual knowledge is out of distribution with respect to parametric knowledge. Moreover, we experimentally demonstrate how context influence properly attributes the privacy leakage to augmented contexts, and we evaluate how factors -- such as model size, context size, generation position, etc. -- affect context privacy leakage. The practical implications of our results will inform practitioners of the privacy risk associated with augmented contextual knowledge.
>
---
#### [replaced 027] Why Reinforcement Fine-Tuning Enables MLLMs Preserve Prior Knowledge Better: A Data Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文属多模态大模型持续学习任务，探究SFT与RFT对先验知识遗忘的影响。通过引入拼图新任务，在Qwen2.5-VL上实验发现：RFT因训练数据更契合模型原有概率分布，故遗忘更少；数据分布而非算法本身是遗忘主因。**

- **链接: [https://arxiv.org/pdf/2506.23508v3](https://arxiv.org/pdf/2506.23508v3)**

> **作者:** Zhihao Zhang; Qiaole Dong; Qi Zhang; Jun Zhao; Enyu Zhou; Zhiheng Xi; Senjie Jin; Xiaoran Fan; Yuhao Zhou; Mingqi Wu; Yanwei Fu; Tao Ji; Tao Gui; Xuanjing Huang; Kai Chen
>
> **备注:** 28 pages (Preprint.)
>
> **摘要:** Post-training algorithms such as Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT) are widely used to adapt multimodal large language models to downstream tasks. While effective at task adaptation, their impact on prior knowledge remains unclear. In this paper, we introduce jigsaw puzzles as a novel task absent from existing pretraining corpora and systematically study the behavior of SFT and RFT on open-source multimodal model, Qwen2.5-VL series. Our experiments reveal a sharp trade-off: SFT enables rapid task acquisition but leads to catastrophic forgetting, whereas RFT learns more slowly but maintains prior knowledge. We study this phenomenon through learning dynamics by examining both the magnitude and direction of how training data influence prior knowledge. Our analysis shows that RFT mainly reinforces correct samples naturally aligned with the base model's probability landscape, leading to weaker interference with prior knowledge. Moreover, training on RFT-simulated rollouts, which exert a small magnitude of influence and are well aligned in direction to prior knowledge, allows SFT to preserve prior knowledge better while rapidly learning new tasks. These findings suggest that distribution of training data, rather than algorithmic differences, plays a central role in forgetting, and highlight RFT's potential for stable continual learning in multimodal large language models.
>
---
#### [replaced 028] Adaptive Detector-Verifier Framework for Zero-Shot Polyp Detection in Open-World Settings
- **分类: cs.CV; cs.CL**

- **简介: 该论文面向零-shot开放世界息肉检测任务，解决临床内镜图像因光照变化、运动模糊等导致的域偏移与漏检问题。提出自适应检测-验证框架：YOLOv11检测器结合VLM验证器，通过VLM指导动态调阈值，并用成本敏感的GRPO强化学习微调验证器，显著提升召回率，减少漏检。**

- **链接: [https://arxiv.org/pdf/2512.12492v2](https://arxiv.org/pdf/2512.12492v2)**

> **作者:** Shengkai Xu; Hsiang Lun Kao; Tianxiang Xu; Honghui Zhang; Junqiao Wang; Runmeng Ding; Guanyu Liu; Tianyu Shi; Zhenyu Yu; Guofeng Pan; Ziqian Bi; Yuqi Ouyang
>
> **摘要:** Polyp detectors trained on clean datasets often underperform in real-world endoscopy, where illumination changes, motion blur, and occlusions degrade image quality. Existing approaches struggle with the domain gap between controlled laboratory conditions and clinical practice, where adverse imaging conditions are prevalent. In this work, we propose AdaptiveDetector, a novel two-stage detector-verifier framework comprising a YOLOv11 detector with a vision-language model (VLM) verifier. The detector adaptively adjusts per-frame confidence thresholds under VLM guidance, while the verifier is fine-tuned with Group Relative Policy Optimization (GRPO) using an asymmetric, cost-sensitive reward function specifically designed to discourage missed detections -- a critical clinical requirement. To enable realistic assessment under challenging conditions, we construct a comprehensive synthetic testbed by systematically degrading clean datasets with adverse conditions commonly encountered in clinical practice, providing a rigorous benchmark for zero-shot evaluation. Extensive zero-shot evaluation on synthetically degraded CVC-ClinicDB and Kvasir-SEG images demonstrates that our approach improves recall by 14 to 22 percentage points over YOLO alone, while precision remains within 0.7 points below to 1.7 points above the baseline. This combination of adaptive thresholding and cost-sensitive reinforcement learning achieves clinically aligned, open-world polyp detection with substantially fewer false negatives, thereby reducing the risk of missed precancerous polyps and improving patient outcomes.
>
---
#### [replaced 029] Pragmatic Inference for Moral Reasoning Acquisition: Generalization via Distributional Semantics
- **分类: cs.CL**

- **简介: 该论文属道德推理任务，旨在解决LLM依赖分布语义而难以泛化至普适道德推理的问题。作者基于道德基础理论，提出务实推理方法，利用上下文动态桥接语义与道德 pragmatics，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.24102v2](https://arxiv.org/pdf/2509.24102v2)**

> **作者:** Guangliang Liu; Xi Chen; Bocheng Chen; Han Zi; Xitong Zhang; Kristen Johnson
>
> **摘要:** Moral reasoning has emerged as a promising research direction for Large Language Models (LLMs), yet achieving generalization remains a central challenge. From a linguistic standpoint, this difficulty arises because LLMs are adept at capturing distributional semantics, which fundamentally differs from the morals which operate at the pragmatic level. This paper investigates how LLMs can achieve generalized moral reasoning despite their reliance on distributional semantics. We propose pragmatic inference methods grounded in moral foundations theory, which leverage contextual information at each step to bridge the pragmatic gap and guide LLMs in connecting moral foundations with moral reasoning objectives. Experimental results demonstrate that our approach significantly enhances LLMs' generalization in moral reasoning, providing a foundation for future research grounded in moral foundations theory.
>
---
#### [replaced 030] Latent Self-Consistency for Reliable Majority-Set Selection in Short- and Long-Answer Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属大语言模型推理一致性任务，旨在解决LLM输出不一致问题。提出Latent Self-Consistency（LSC），利用可学习token嵌入进行语义层面的多数响应选择，无需修改模型结构，兼顾短/长答案场景，提升准确率与校准性，仅增0.9%计算开销。**

- **链接: [https://arxiv.org/pdf/2508.18395v2](https://arxiv.org/pdf/2508.18395v2)**

> **作者:** Jungsuk Oh; Jay-Yoon Lee
>
> **摘要:** Probabilistic decoding in Large Language Models (LLMs) often yields inconsistent outputs, particularly on complex or long-form questions. Self-Consistency (SC) mitigates this for short-form QA by majority voting over exact strings, whereas Universal Self-Consistency (USC) and Weighted Unigram Consistency Score (WUCS) extend to long-form responses but lose accuracy on short-form benchmarks. We introduce \textbf{Latent Self-Consistency (LSC)}, which selects the most semantically consistent response using learnable token embeddings. LSC's lightweight forward processing of summary tokens only introduces negligible runtime overhead (at most $0.9\%$) on top of standard decoding of the base LLM, and requires no changes to the model architecture. Across 6 short-form and 5 long-form reasoning benchmarks (e.g., MATH, MMLU, TruthfulQA), LSC surpasses SC, USC, and WUCS on both short-form and long-form on average performance, while adding negligible computational overhead on vanilla inference. These results position LSC as a reliable consistency-selection method that works effectively across various answer formats. Additionally, LSC provides well-calibrated confidence estimates, maintaining low expected calibration error across both answer formats.
>
---
#### [replaced 031] Pronunciation-Lexicon Free Training for Phoneme-based Crosslingual ASR via Joint Stochastic Approximation
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文面向跨语言语音识别（ASR）任务，旨在消除对发音词典的依赖。提出JSA-SPG方法：将音素建模为离散隐变量，联合训练S2P、P2G与G2P模型，采用联合随机近似（JSA）算法优化，并引入MLS解码与P2G增强。实验验证其在低资源语言上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2507.06249v2](https://arxiv.org/pdf/2507.06249v2)**

> **作者:** Saierdaer Yusuyin; Te Ma; Hao Huang; Zhijian Ou
>
> **备注:** Accepted by IEEE TASLP
>
> **摘要:** Recently, pre-trained models with phonetic supervision have demonstrated their advantages for crosslingual speech recognition in data efficiency and information sharing across languages. However, a limitation is that a pronunciation lexicon is needed for such phoneme-based crosslingual speech recognition. In this study, we aim to eliminate the need for pronunciation lexicons and propose a latent variable model based method, with phonemes being treated as discrete latent variables. The new method consists of a speech-to-phoneme (S2P) model and a phoneme-to-grapheme (P2G) model, and a grapheme-to-phoneme (G2P) model is introduced as an auxiliary inference model. To jointly train the three models, we utilize the joint stochastic approximation (JSA) algorithm, which is a stochastic extension of the EM (expectation-maximization) algorithm and has demonstrated superior performance particularly in estimating discrete latent variable models. Furthermore, we propose marginal likelihood scoring (MLS) decoding to align inference with the training objective and P2G augmentation to improve the robustness of P2G mapping. Based on the Whistle multilingual pre-trained S2P model, crosslingual experiments are conducted in Polish (130 h) and Indonesian (20 h). With only 10 minutes of phoneme supervision, the new method, JSA-SPG, achieves 5% error rate reductions compared to the best crosslingual fine-tuning approach using subword or full phoneme supervision. Furthermore, it is found that in language domain adaptation (i.e., utilizing cross-domain text-only data), JSA-SPG outperforms the standard practice of language model fusion via the auxiliary support of the G2P model by 9% error rate reductions. To facilitate reproducibility and encourage further exploration in this field, we open-source the JSA-SPG training code and complete pipeline.
>
---
#### [replaced 032] TIBSTC-CoT: A Multi-Domain Instruction Dataset for Chain-of-Thought Reasoning in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向藏语低资源问题，构建了大规模多领域藏语思维链（CoT）指令数据集TIBSTC-CoT，并基于其训练出藏语专用LLM系列Sunshine-thinking，在推理与生成上达SOTA水平。**

- **链接: [https://arxiv.org/pdf/2508.01977v2](https://arxiv.org/pdf/2508.01977v2)**

> **作者:** Fan Gao; Cheng Huang; Nyima Tashi; Yutong Liu; Xiangxiang Wang; Thupten Tsering; Ban Ma-bao; Renzeg Duojie; Gadeng Luosang; Rinchen Dongrub; Dorje Tashi; Xiao Feng; Hao Wang; Yongbin Yu
>
> **备注:** We will merge this paper with arXiv:2503.18288
>
> **摘要:** To address the severe data scarcity in Tibetan, a low-resource language spoken by over six million people, we introduce TIBSTC-CoT, the large-scale, multi-domain Tibetan dataset automatically constructed via chain-of-thought prompting with large language models (LLMs). TIBSTC-CoT establishes a scalable and reproducible framework for dataset creation in low-resource settings, covering diverse domains and reasoning patterns essential for language understanding and generation. Building on this dataset, we develop the Sunshine-thinking LLM family, a series of Tibetan-centric LLMs equipped with chain-of-thought capabilities. Trained entirely on TIBSTC-CoT, Sunshine-thinking has demonstrated strong reasoning and generation performance, comparable to state-of-the-art (SOTA) multilingual LLMs. Our work marks a significant step toward inclusive AI by enabling high-quality Tibetan language processing through both resource creation and model innovation. All data are available: https://github.com/Vicentvankor/sun-shine.
>
---
#### [replaced 033] Non-Resolution Reasoning (NRR): A Computational Framework for Contextual Identity and Ambiguity Preservation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出非解析推理（NRR）框架，解决AI系统过早消解语义歧义的问题。它挑战传统“同一性”假设，通过多向量嵌入、非坍缩注意力和上下文身份追踪，实现歧义保留与并行解释。在合成任务中显著提升分布外泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.13478v2](https://arxiv.org/pdf/2512.13478v2)**

> **作者:** Kei Saito
>
> **备注:** 7 pages, 2 figures, ORCID: 0009-0006-4715-9176
>
> **摘要:** Current artificial intelligence systems, despite remarkable capabilities in text generation and pattern recognition, exhibit a fundamental architectural limitation: they resolve ambiguity prematurely. This premature semantic collapse -- the tendency to collapse multiple valid interpretations into a single output -- stems from classical identity assumptions embedded in standard neural architectures. We propose Non-Resolution Reasoning (NRR), a computational framework that treats ambiguity retention as a valid reasoning mode rather than a defect to be eliminated. NRR introduces three core principles: (1) Non-Identity (A $\ne$ A) -- the same symbol refers to different entities across contexts; (2) Approximate Identity (A $\approx$ A) -- entities share partial structural overlap without being identical; and (3) Non-Resolution -- conflicting interpretations can coexist without forced convergence. We formalize these principles through three architectural components: Multi-Vector Embeddings for context-dependent representation, Non-Collapsing Attention for parallel interpretation retention, and Contextual Identity Tracking (CIT) for maintaining A $\ne$ A across inference. We demonstrate NRR's advantages through case studies in paradox handling, creative generation, and context-dependent reasoning. Crucially, we provide a minimal empirical validation on a synthetic context-shift task where an NRR-lite model achieves 90.9% out-of-distribution accuracy compared to 9.1% for standard architectures, demonstrating that ambiguity preservation enables structural generalization. NRR challenges the assumption that meaning must collapse to be useful, offering a foundation for AI systems capable of sophisticated ambiguity handling and creative reasoning. The question is not whether AI should resolve ambiguity, but when, how, and under whose control.
>
---
#### [replaced 034] Efficient Adaptive Rejection Sampling for Accelerating Speculative Decoding in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属LLM推理加速任务，旨在解决投机解码中因固定阈值导致的高不确定性下随机拒接问题。提出EARS方法，动态依据目标模型预测不确定性（1−max(P)）自适应调整接受阈值，在不确定时放宽标准、确定时严格把关，提升吞吐量且几乎不损精度。**

- **链接: [https://arxiv.org/pdf/2512.13194v2](https://arxiv.org/pdf/2512.13194v2)**

> **作者:** Chendong Sun; mingmin Chen; Lei Xu
>
> **摘要:** Speculative Decoding is a prominent technique for accelerating the autoregressive inference of large language models (LLMs) by employing a fast draft model to propose candidate token sequences and a large target model to verify them in parallel. However, its core component -- the rejection sampling mechanism -- relies on a fixed, context-independent random threshold. This leads to a significant "random rejection" problem in high-uncertainty generation scenarios, where plausible candidate tokens are frequently rejected due to random chance, undermining inference efficiency. This paper introduces Efficient Adaptive Rejection Sampling (EARS), a novel method that dynamically adjusts the acceptance threshold by incorporating the target model's own predictive uncertainty, measured as 1 - max(P_target). By introducing a tolerance term proportional to this uncertainty, EARS intelligently relaxes the acceptance criterion when the model is uncertain, effectively reducing random rejections while maintaining strict standards when the model is confident. Experiments on creative writing and open-domain QA tasks demonstrate that EARS significantly enhances the efficiency of speculative decoding, achieving up to an 18.12% increase in throughput with a negligible 0.84% accuracy drop on the GSM8K benchmark. The method requires no modifications to model architectures and can be seamlessly integrated into existing speculative decoding frameworks.
>
---
#### [replaced 035] Confucius Code Agent: Scalable Agent Scaffolding for Real-World Codebases
- **分类: cs.CL; cs.AI; cs.LG; cs.SE**

- **简介: 该论文提出Confucius Code Agent（CCA）及配套SDK，面向真实大规模代码库的软件工程任务。旨在解决现有编码代理在可扩展性、长程交互、工具协同与可控性上的不足。工作包括构建三层体验驱动的SDK、分层记忆与持续学习机制、模块化工具系统，以及自动化配置优化的meta-agent。**

- **链接: [https://arxiv.org/pdf/2512.10398v3](https://arxiv.org/pdf/2512.10398v3)**

> **作者:** Zhaodong Wang; Zhenting Qi; Sherman Wong; Nathan Hu; Samuel Lin; Jun Ge; Erwin Gao; Wenlin Chen; Yilun Du; Minlan Yu; Ying Zhang
>
> **备注:** The latest draft
>
> **摘要:** Real-world software engineering tasks require coding agents that can operate over massive repositories, sustain long-horizon sessions, and reliably coordinate complex toolchains at test time. Existing research-grade agents offer transparency but struggle when scaled to real-world workloads, while proprietary systems achieve strong practical performance but provide limited extensibility, interpretability, and controllability. We introduce the Confucius Code Agent (CCA), a scalable software engineering agent that can operate at enterprise-level codebases. CCA is built on top of the Confucius SDK, an agent development platform structured around three complementary perspectives: Agent Experience (AX), User Experience (UX), and Developer Experience (DX). The SDK integrates a unified orchestrator with hierarchical working memory for long-context operation, a persistent note-taking mechanism for cross-session continual learning, and a modular extension system for reliable tool use. In addition, we introduce a meta-agent that automates the synthesis, evaluation, and refinement of agent configurations through a build-test-improve loop, enabling rapid adaptation to new tasks, environments, and tool stacks. Instantiated with these mechanisms, CCA demonstrates strong performance on real-world software engineering tasks. On SWE-Bench-Pro, CCA achieves a Resolve@1 of 54.3%, surpassing both research-grade and proprietary coding agents under comparable model conditions. Together, the Confucius SDK and CCA form a general, extensible, and production-grade foundation for building robust coding agents, bridging the gap between research prototypes and practical large-scale deployment.
>
---
#### [replaced 036] Enhancing Geo-localization for Crowdsourced Flood Imagery via LLM-Guided Attention
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
#### [replaced 037] Tree Matching Networks for Natural Language Inference: Parameter-Efficient Semantic Understanding via Dependency Parse Trees
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文面向自然语言推理（NLI）任务，旨在解决大模型参数多、训练慢的问题。提出树匹配网络（TMN），将图匹配网络适配到依赖句法树上，利用显式结构提升语义理解效率。实验表明TMN在SNLI上精度高、参数少、训练快。**

- **链接: [https://arxiv.org/pdf/2512.00204v2](https://arxiv.org/pdf/2512.00204v2)**

> **作者:** Jason Lunder
>
> **备注:** 16 pages, preprint
>
> **摘要:** In creating sentence embeddings for Natural Language Inference (NLI) tasks, using transformer-based models like BERT leads to high accuracy, but require hundreds of millions of parameters. These models take in sentences as a sequence of tokens, and learn to encode the meaning of the sequence into embeddings such that those embeddings can be used reliably for NLI tasks. Essentially, every word is considered against every other word in the sequence, and the transformer model is able to determine the relationships between them, entirely from scratch. However, a model that accepts explicit linguistic structures like dependency parse trees may be able to leverage prior encoded information about these relationships, without having to learn them from scratch, thus improving learning efficiency. To investigate this, we adapt Graph Matching Networks (GMN) to operate on dependency parse trees, creating Tree Matching Networks (TMN). We compare TMN to a BERT based model on the SNLI entailment task and on the SemEval similarity task. TMN is able to achieve significantly better results with a significantly reduced memory footprint and much less training time than the BERT based model on the SNLI task, while both models struggled to preform well on the SemEval. Explicit structural representations significantly outperform sequence-based models at comparable scales, but current aggregation methods limit scalability. We propose multi-headed attention aggregation to address this limitation.
>
---
#### [replaced 038] One Battle After Another: Probing LLMs' Limits on Multi-Turn Instruction Following with a Benchmark Evolving Framework
- **分类: cs.CL**

- **简介: 该论文属大语言模型（LLM）多轮指令遵循能力评估任务，旨在解决现有基准固定轮次、忽视用户耐心与交互体验的问题。作者提出基于Flow理论的演化式评估框架EvolIF，含三层追踪与查询合成机制，动态终止对话，覆盖12类约束，更真实评测模型持续交互能力。**

- **链接: [https://arxiv.org/pdf/2511.03508v2](https://arxiv.org/pdf/2511.03508v2)**

> **作者:** Qi Jia; Ye Shen; Xiujie Song; Kaiwei Zhang; Shibo Wang; Dun Pei; Xiangyang Zhu; Guangtao Zhai
>
> **摘要:** Evaluating LLMs' instruction-following ability in multi-topic dialogues is essential yet challenging. Existing benchmarks are limited to a fixed number of turns, susceptible to saturation and failing to account for users' interactive experience. In this work, we propose a novel framework backed by a three-layer tracking mechanism and a query synthesis agent to mimic sequential user behaviors. Incorporating Flow Theory, we introduce process-centric metrics and terminate a conversational evaluation only upon exhausting user patience. Upon this framework, we present EvolIF, an evolving benchmark covering 12 constraint groups. Results indicate that GPT-5 excels, sustaining 14 turns with 66.40% robustness. It outperforms Gemini-3.0-Pro by a margin of 5.59%, while other models trail behind. Resources are available at https://github.com/JiaQiSJTU/EvolvingInstructionFollowing.
>
---
#### [replaced 039] Language Self-Play For Data-Free Training
- **分类: cs.AI; cs.CL; cs.GT**

- **简介: 该论文提出语言自博弈（LSP）方法，属无数据强化学习任务，旨在解决大模型持续训练依赖海量新数据的瓶颈。通过让模型与自身博弈迭代优化策略，仅用预训练模型即可在指令遵循、数学与编程任务上实现性能提升。**

- **链接: [https://arxiv.org/pdf/2509.07414v2](https://arxiv.org/pdf/2509.07414v2)**

> **作者:** Jakub Grudzien Kuba; Mengting Gu; Qi Ma; Yuandong Tian; Vijai Mohan; Jason Chen
>
> **摘要:** Large language models (LLMs) have advanced rapidly in recent years, driven by scale, abundant high-quality training data, and reinforcement learning. Yet this progress faces a fundamental bottleneck: the need for ever more data from which models can continue to learn. In this work, we propose a reinforcement learning approach that removes this dependency by enabling models to improve without additional data. Our method leverages a game-theoretic framework of self-play, where a model's capabilities are cast as performance in a competitive game and stronger policies emerge by having the model play against itself-a process we call Language Self-Play (LSP). Experiments with Llama-3.2-3B-Instruct on instruction-following, mathematics, and coding benchmarks show that pretrained models can be effectively improved with self-play alone.
>
---
#### [replaced 040] The Instability of Safety: How Random Seeds and Temperature Expose Inconsistent LLM Refusal Behavior
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属AI安全评估任务，旨在揭示LLM安全拒绝行为的不稳定性问题。作者通过多随机种子与温度设置测试四类模型在876个有害提示上的响应一致性，发现显著决策翻转现象，提出Safety Stability Index，并建议至少3样本评估以提升可靠性。**

- **链接: [https://arxiv.org/pdf/2512.12066v2](https://arxiv.org/pdf/2512.12066v2)**

> **作者:** Erik Larsen
>
> **备注:** 16 pages, 7 figures, 9 tables. Code and data available at https://github.com/erikl2/safety-refusal-stability
>
> **摘要:** Current safety evaluations of large language models rely on single-shot testing, implicitly assuming that model responses are deterministic and representative of the model's safety alignment. We challenge this assumption by investigating the stability of safety refusal decisions across random seeds and temperature settings. Testing four instruction-tuned models from three families (Llama 3.1 8B, Qwen 2.5 7B, Qwen 3 8B, Gemma 3 12B) on 876 harmful prompts across 20 different sampling configurations (4 temperatures x 5 random seeds), we find that 18-28% of prompts exhibit decision flips--the model refuses in some configurations but complies in others--depending on the model. Our Safety Stability Index (SSI) reveals that higher temperatures significantly reduce decision stability (Friedman chi-squared = 396.81, p < 0.001), with mean within-temperature SSI dropping from 0.977 at temperature 0.0 to 0.942 at temperature 1.0. We validate our findings across all model families using Claude 3.5 Haiku as a unified external judge, achieving 89.0% inter-judge agreement with our primary Llama 70B judge (Cohen's kappa = 0.62). Within each model, prompts with higher compliance rates exhibit lower stability (Spearman rho = -0.47 to -0.70, all p < 0.001), indicating that models "waver" more on borderline requests. These findings demonstrate that single-shot safety evaluations are insufficient for reliable safety assessment and that evaluation protocols must account for stochastic variation in model behavior. We show that single-shot evaluation agrees with multi-sample ground truth only 92.4% of the time when pooling across temperatures (94.2-97.7% at fixed temperature depending on setting), and recommend using at least 3 samples per prompt for reliable safety assessment.
>
---
#### [replaced 041] FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属LLM推理优化任务，旨在解决长上下文下KV缓存占用大、检索效率低的问题。提出FreeKV框架：算法上采用推测式检索与细粒度校正；系统上设计混合内存布局与双缓冲流式召回，实现高精度、高速KV检索。**

- **链接: [https://arxiv.org/pdf/2505.13109v3](https://arxiv.org/pdf/2505.13109v3)**

> **作者:** Guangda Liu; Chengwei Li; Zhenyu Ning; Jing Lin; Yiwu Yao; Danning Ke; Minyi Guo; Jieru Zhao
>
> **摘要:** Large language models (LLMs) have been widely deployed with rapidly expanding context windows to support increasingly demanding applications. However, long contexts pose significant deployment challenges, primarily due to the KV cache whose size grows proportionally with context length. While KV cache compression methods are proposed to address this issue, KV dropping methods incur considerable accuracy loss, and KV retrieval methods suffer from significant efficiency bottlenecks. We propose FreeKV, an algorithm-system co-optimization framework to enhance KV retrieval efficiency while preserving accuracy. On the algorithm side, FreeKV introduces speculative retrieval to shift the KV selection and recall processes out of the critical path, combined with fine-grained correction to ensure accuracy. On the system side, FreeKV employs hybrid KV layouts across CPU and GPU memory to eliminate fragmented data transfers, and leverages double-buffered streamed recall to further improve efficiency. Experiments demonstrate that FreeKV achieves near-lossless accuracy across various scenarios and models, delivering up to 13$\times$ speedup compared to SOTA KV retrieval methods.
>
---
