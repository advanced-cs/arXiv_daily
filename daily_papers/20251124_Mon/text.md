# 自然语言处理 cs.CL

- **最新发布 59 篇**

- **更新 32 篇**

## 最新发布

#### [new 001] Don't Learn, Ground: A Case for Natural Language Inference with Visual Grounding
- **分类: cs.CL**

- **简介: 该论文针对自然语言推理（NLI）任务，提出一种无需微调的零样本方法。通过文本生成图像构建视觉表征，结合语义相似度与视觉问答进行推理，有效缓解文本偏见和表面启发式问题，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.17358v1](https://arxiv.org/pdf/2511.17358v1)**

> **作者:** Daniil Ignatev; Ayman Santeer; Albert Gatt; Denis Paperno
>
> **摘要:** We propose a zero-shot method for Natural Language Inference (NLI) that leverages multimodal representations by grounding language in visual contexts. Our approach generates visual representations of premises using text-to-image models and performs inference by comparing these representations with textual hypotheses. We evaluate two inference techniques: cosine similarity and visual question answering. Our method achieves high accuracy without task-specific fine-tuning, demonstrating robustness against textual biases and surface heuristics. Additionally, we design a controlled adversarial dataset to validate the robustness of our approach. Our findings suggest that leveraging visual modality as a meaning representation provides a promising direction for robust natural language understanding.
>
---
#### [new 002] How Language Directions Align with Token Geometry in Multilingual LLMs
- **分类: cs.CL**

- **简介: 该论文研究多语言大模型中语言信息的表征结构与几何特性。针对语言编码如何在模型层间演化这一问题，通过线性与非线性探测及新提出的词元-语言对齐分析，揭示语言信息在首层即显著分离，且其结构受训练数据语言分布影响，体现显著的语料偏见。**

- **链接: [https://arxiv.org/pdf/2511.16693v1](https://arxiv.org/pdf/2511.16693v1)**

> **作者:** JaeSeong Kim; Suan Lee
>
> **备注:** 4 pages
>
> **摘要:** Multilingual LLMs demonstrate strong performance across diverse languages, yet there has been limited systematic analysis of how language information is structured within their internal representation space and how it emerges across layers. We conduct a comprehensive probing study on six multilingual LLMs, covering all 268 transformer layers, using linear and nonlinear probes together with a new Token--Language Alignment analysis to quantify the layer-wise dynamics and geometric structure of language encoding. Our results show that language information becomes sharply separated in the first transformer block (+76.4$\pm$8.2 percentage points from Layer 0 to 1) and remains almost fully linearly separable throughout model depth. We further find that the alignment between language directions and vocabulary embeddings is strongly tied to the language composition of the training data. Notably, Chinese-inclusive models achieve a ZH Match@Peak of 16.43\%, whereas English-centric models achieve only 3.90\%, revealing a 4.21$\times$ structural imprinting effect. These findings indicate that multilingual LLMs distinguish languages not by surface script features but by latent representational structures shaped by the training corpus. Our analysis provides practical insights for data composition strategies and fairness in multilingual representation learning. All code and analysis scripts are publicly available at: https://github.com/thisiskorea/How-Language-Directions-Align-with-Token-Geometry-in-Multilingual-LLMs.
>
---
#### [new 003] Towards Hyper-Efficient RAG Systems in VecDBs: Distributed Parallel Multi-Resolution Vector Search
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对RAG系统中向量数据库检索效率与语义精度难以兼顾的问题，提出语义金字塔索引（SPI）框架。通过动态自适应选择多粒度检索分辨率，实现从粗到细的渐进式搜索，显著提升检索速度与内存效率，同时保持高质量语义覆盖。**

- **链接: [https://arxiv.org/pdf/2511.16681v1](https://arxiv.org/pdf/2511.16681v1)**

> **作者:** Dong Liu; Yanxuan Yu
>
> **备注:** Accepted to IEEE International Conference on Parallel and Distributed Systems 2025 (ICPADS 2025 Oral)
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems have become a dominant approach to augment large language models (LLMs) with external knowledge. However, existing vector database (VecDB) retrieval pipelines rely on flat or single-resolution indexing structures, which cannot adapt to the varying semantic granularity required by diverse user queries. This limitation leads to suboptimal trade-offs between retrieval speed and contextual relevance. To address this, we propose \textbf{Semantic Pyramid Indexing (SPI)}, a novel multi-resolution vector indexing framework that introduces query-adaptive resolution control for RAG in VecDBs. Unlike existing hierarchical methods that require offline tuning or separate model training, SPI constructs a semantic pyramid over document embeddings and dynamically selects the optimal resolution level per query through a lightweight classifier. This adaptive approach enables progressive retrieval from coarse-to-fine representations, significantly accelerating search while maintaining semantic coverage. We implement SPI as a plugin for both FAISS and Qdrant backends and evaluate it across multiple RAG tasks including MS MARCO, Natural Questions, and multimodal retrieval benchmarks. SPI achieves up to \textbf{5.7$\times$} retrieval speedup and \textbf{1.8$\times$} memory efficiency gain while improving end-to-end QA F1 scores by up to \textbf{2.5 points} compared to strong baselines. Our theoretical analysis provides guarantees on retrieval quality and latency bounds, while extensive ablation studies validate the contribution of each component. The framework's compatibility with existing VecDB infrastructures makes it readily deployable in production RAG systems. Code is availabe at \href{https://github.com/FastLM/SPI_VecDB}{https://github.com/FastLM/SPI\_VecDB}.
>
---
#### [new 004] Social-Media Based Personas Challenge: Hybrid Prediction of Common and Rare User Actions on Bluesky
- **分类: cs.CL**

- **简介: 该论文针对社交平台用户行为预测任务，解决常见与罕见行为预测不平衡问题。提出混合方法：基于历史模式的查询库、人物画像的LightGBM模型、融合文本与时间特征的神经网络用于稀有行为分类，及自动生成回复。在Bluesky数据集上验证，显著提升罕见行为预测性能，获COLM 2025挑战赛第一名。**

- **链接: [https://arxiv.org/pdf/2511.17241v1](https://arxiv.org/pdf/2511.17241v1)**

> **作者:** Benjamin White; Anastasia Shimorina
>
> **备注:** 1st place at SocialSim: Social-Media Based Personas challenge 2025
>
> **摘要:** Understanding and predicting user behavior on social media platforms is crucial for content recommendation and platform design. While existing approaches focus primarily on common actions like retweeting and liking, the prediction of rare but significant behaviors remains largely unexplored. This paper presents a hybrid methodology for social media user behavior prediction that addresses both frequent and infrequent actions across a diverse action vocabulary. We evaluate our approach on a large-scale Bluesky dataset containing 6.4 million conversation threads spanning 12 distinct user actions across 25 persona clusters. Our methodology combines four complementary approaches: (i) a lookup database system based on historical response patterns; (ii) persona-specific LightGBM models with engineered temporal and semantic features for common actions; (iii) a specialized hybrid neural architecture fusing textual and temporal representations for rare action classification; and (iv) generation of text replies. Our persona-specific models achieve an average macro F1-score of 0.64 for common action prediction, while our rare action classifier achieves 0.56 macro F1-score across 10 rare actions. These results demonstrate that effective social media behavior prediction requires tailored modeling strategies recognizing fundamental differences between action types. Our approach achieved first place in the SocialSim: Social-Media Based Personas challenge organized at the Social Simulation with LLMs workshop at COLM 2025.
>
---
#### [new 005] From Representation to Enactment: The ABC Framework of the Translating Mind
- **分类: cs.CL**

- **简介: 该论文提出ABC框架，挑战传统代表式认知模型，将翻译视为动态的具身实践。基于扩展心智与主动推理理论，强调大脑-身体-环境交互中意义的实时建构，解决翻译认知机制的本质问题，推动非表征主义视角下的翻译研究。**

- **链接: [https://arxiv.org/pdf/2511.16811v1](https://arxiv.org/pdf/2511.16811v1)**

> **作者:** Michael Carl; Takanori Mizowaki; Aishvarya Raj; Masaru Yamada; Devi Sri Bandaru; Yuxiang Wei; Xinyue Ren
>
> **摘要:** Building on the Extended Mind (EM) theory and radical enactivism, this article suggests an alternative to representation-based models of the mind. We lay out a novel ABC framework of the translating mind, in which translation is not the manipulation of static interlingual correspondences but an enacted activity, dynamically integrating affective, behavioral, and cognitive (ABC) processes. Drawing on Predictive Processing and (En)Active Inference, we argue that the translator's mind emerges, rather than being merely extended, through loops of brain-body-environment interactions. This non-representational account reframes translation as skillful participation in sociocultural practice, where meaning is co-created in real time through embodied interaction with texts, tools, and contexts.
>
---
#### [new 006] Shona spaCy: A Morphological Analyzer for an Under-Resourced Bantu Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对资源匮乏的班图语绍纳语，提出Shona spaCy，一个基于规则的形态分析工具。旨在解决其缺乏语言意识工具的问题，通过整合词典与语言学规则，实现词性、词元及形态特征标注，提升NLP可及性，为其他类似语言提供模板。**

- **链接: [https://arxiv.org/pdf/2511.16680v1](https://arxiv.org/pdf/2511.16680v1)**

> **作者:** Happymore Masoka
>
> **摘要:** Despite rapid advances in multilingual natural language processing (NLP), the Bantu language Shona remains under-served in terms of morphological analysis and language-aware tools. This paper presents Shona spaCy, an open-source, rule-based morphological pipeline for Shona built on the spaCy framework. The system combines a curated JSON lexicon with linguistically grounded rules to model noun-class prefixes (Mupanda 1-18), verbal subject concords, tense-aspect markers, ideophones, and clitics, integrating these into token-level annotations for lemma, part-of-speech, and morphological features. The toolkit is available via pip install shona-spacy, with source code at https://github.com/HappymoreMasoka/shona-spacy and a PyPI release at https://pypi.org/project/shona-spacy/0.1.4/. Evaluation on formal and informal Shona corpora yields 90% POS-tagging accuracy and 88% morphological-feature accuracy, while maintaining transparency in its linguistic decisions. By bridging descriptive grammar and computational implementation, Shona spaCy advances NLP accessibility and digital inclusion for Shona speakers and provides a template for morphological analysis tools for other under-resourced Bantu languages.
>
---
#### [new 007] Selective Rotary Position Embedding
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Selective RoPE，一种输入依赖的旋转位置编码机制，用于改进Transformer模型的位置感知。针对传统RoPE固定角度的局限，该方法在软硬件注意力与线性变换器中实现任意角度旋转，增强对序列任务如复制、状态跟踪的建模能力，显著提升语言建模性能。**

- **链接: [https://arxiv.org/pdf/2511.17388v1](https://arxiv.org/pdf/2511.17388v1)**

> **作者:** Sajad Movahedi; Timur Carstensen; Arshia Afzal; Frank Hutter; Antonio Orvieto; Volkan Cevher
>
> **摘要:** Position information is essential for language modeling. In softmax transformers, Rotary Position Embeddings (\textit{RoPE}) encode positions through \textit{fixed-angle} rotations, while in linear transformers, order is handled via input-dependent (selective) gating that decays past key-value associations. Selectivity has generally been shown to improve language-related tasks. Inspired by this, we introduce \textit{Selective RoPE}, an \textit{input-dependent} rotary embedding mechanism, that generalizes \textit{RoPE}, and enables rotation in \textit{arbitrary angles} for both linear and softmax transformers. We show that softmax attention already performs a hidden form of these rotations on query-key pairs, uncovering an implicit positional structure. We further show that in state-space models and gated linear transformers, the real part manages forgetting while the imaginary part encodes positions through rotations. We validate our method by equipping gated transformers with \textit{Selective RoPE}, demonstrating that its input-dependent rotations improve performance in language modeling and on difficult sequence tasks like copying, state tracking, and retrieval.
>
---
#### [new 008] A new kid on the block: Distributional semantics predicts the word-specific tone signatures of monosyllabic words in conversational Taiwan Mandarin
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究普通话单音节词在口语中的声调实现，探究语义对音高的影响。通过广义加性模型分析语料，发现词汇语义是声调轮廓的重要预测因子，基于上下文词嵌入可准确预测具体发音，挑战传统声调理论，支持非对称性词库模型。**

- **链接: [https://arxiv.org/pdf/2511.17337v1](https://arxiv.org/pdf/2511.17337v1)**

> **作者:** Xiaoyun Jin; Mirjam Ernestus; R. Harald Baayen
>
> **备注:** arXiv admin note: text overlap with arXiv:2409.07891
>
> **摘要:** We present a corpus-based investigation of how the pitch contours of monosyllabic words are realized in spontaneous conversational Mandarin, focusing on the effects of words' meanings. We used the generalized additive model to decompose a given observed pitch contour into a set of component pitch contours that are tied to different control variables and semantic predictors. Even when variables such as word duration, gender, speaker identity, tonal context, vowel height, and utterance position are controlled for, the effect of word remains a strong predictor of tonal realization. We present evidence that this effect of word is a semantic effect: word sense is shown to be a better predictor than word, and heterographic homophones are shown to have different pitch contours. The strongest evidence for the importance of semantics is that the pitch contours of individual word tokens can be predicted from their contextualized embeddings with an accuracy that substantially exceeds a permutation baseline. For phonetics, distributional semantics is a new kid on the block. Although our findings challenge standard theories of Mandarin tone, they fit well within the theoretical framework of the Discriminative Lexicon Model.
>
---
#### [new 009] Bench360: Benchmarking Local LLM Inference from 360°
- **分类: cs.CL; cs.AI; cs.LG; cs.PF**

- **简介: 该论文提出Bench360，一个面向本地大语言模型推理的全方位基准测试框架。针对用户在配置选择上面临的复杂性问题，它支持自定义任务与多维度指标（系统性能与任务效果），跨引擎、量化级别与使用场景进行自动化评估，揭示了性能与效率间的权衡，证明无最优配置，凸显框架必要性。**

- **链接: [https://arxiv.org/pdf/2511.16682v1](https://arxiv.org/pdf/2511.16682v1)**

> **作者:** Linus Stuhlmann; Mauricio Fadel Argerich; Jonathan Fürst
>
> **摘要:** Running large language models (LLMs) locally is becoming increasingly common. While the growing availability of small open-source models and inference engines has lowered the entry barrier, users now face an overwhelming number of configuration choices. Identifying an optimal configuration -- balancing functional and non-functional requirements -- requires substantial manual effort. While several benchmarks target LLM inference, they are designed for narrow evaluation goals and not user-focused. They fail to integrate relevant system and task-specific metrics into a unified, easy-to-use benchmark that supports multiple inference engines, usage scenarios, and quantization levels. To address this gap, we present Bench360 -- Benchmarking Local LLM Inference from 360°. Bench360 allows users to easily define their own custom tasks along with datasets and relevant task-specific metrics and then automatically benchmarks selected LLMs, inference engines, and quantization levels across different usage scenarios (single stream, batch & server). Bench360 tracks a wide range of metrics, including (1) system metrics -- such as Computing Performance (e.g., latency, throughput), Resource Usage (e.g., energy per query), and Deployment (e.g., cold start time) -- and (2) task-specific metrics such as ROUGE, F1 score or accuracy. We demonstrate Bench360 on four common LLM tasks -- General Knowledge & Reasoning, QA, Summarization and Text-to-SQL -- across three hardware platforms and four state of the art inference engines. Our results reveal several interesting trade-offs between task performance and system-level efficiency, highlighting the differences in inference engines and models. Most importantly, there is no single best setup for local inference, which strongly motivates the need for a framework such as Bench360.
>
---
#### [new 010] Concept-Based Interpretability for Toxicity Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于毒性语言检测任务，针对现有模型过度依赖特定词汇导致误判的问题，提出基于概念梯度的可解释性方法。通过构建靶向词表集与词-概念对齐评分，识别错误归因的毒性强关联词，并设计无词表增强策略，验证模型在去除显式词汇后是否仍存在偏倚，提升解释的因果性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.16689v1](https://arxiv.org/pdf/2511.16689v1)**

> **作者:** Samarth Garg; Deeksha Varshney; Divya Singh
>
> **备注:** 16 pages
>
> **摘要:** The rise of social networks has not only facilitated communication but also allowed the spread of harmful content. Although significant advances have been made in detecting toxic language in textual data, the exploration of concept-based explanations in toxicity detection remains limited. In this study, we leverage various subtype attributes present in toxicity detection datasets, such as obscene, threat, insult, identity attack, and sexual explicit as concepts that serve as strong indicators to identify whether language is toxic. However, disproportionate attribution of concepts towards the target class often results in classification errors. Our work introduces an interpretability technique based on the Concept Gradient (CG) method which provides a more causal interpretation by measuring how changes in concepts directly affect the output of the model. This is an extension of traditional gradient-based methods in machine learning, which often focus solely on input features. We propose the curation of Targeted Lexicon Set, which captures toxic words that contribute to misclassifications in text classification models. To assess the significance of these lexicon sets in misclassification, we compute Word-Concept Alignment (WCA) scores, which quantify the extent to which these words lead to errors due to over-attribution to toxic concepts. Finally, we introduce a lexicon-free augmentation strategy by generating toxic samples that exclude predefined toxic lexicon sets. This approach allows us to examine whether over-attribution persists when explicit lexical overlap is removed, providing insights into the model's attribution on broader toxic language patterns.
>
---
#### [new 011] Prompt-Based Value Steering of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型的价值导向问题，旨在实现无需微调的动态价值引导。针对静态微调无法适应动态价值观的缺陷，提出一种可复现的提示评估方法，通过量化生成文本中的目标价值观，验证提示能有效引导模型输出符合人类价值的内容。**

- **链接: [https://arxiv.org/pdf/2511.16688v1](https://arxiv.org/pdf/2511.16688v1)**

> **作者:** Giulio Antonio Abbo; Tony Belpaeme
>
> **备注:** 9 pages, 1 figure, 4 tables. Presented at the 3rd International Workshop on Value Engineering in AI (VALE 2025), 28th European Conference on AI. To appear in Springer LNCS
>
> **摘要:** Large language models are increasingly used in applications where alignment with human values is critical. While model fine-tuning is often employed to ensure safe responses, this technique is static and does not lend itself to everyday situations involving dynamic values and preferences. In this paper, we present a practical, reproducible, and model-agnostic procedure to evaluate whether a prompt candidate can effectively steer generated text toward specific human values, formalising a scoring method to quantify the presence and gain of target values in generated responses. We apply our method to a variant of the Wizard-Vicuna language model, using Schwartz's theory of basic human values and a structured evaluation through a dialogue dataset. With this setup, we compare a baseline prompt to one explicitly conditioned on values, and show that value steering is possible even without altering the model or dynamically optimising prompts.
>
---
#### [new 012] PUCP-Metrix: A Comprehensive Open-Source Repository of Linguistic Metrics for Spanish
- **分类: cs.CL**

- **简介: 该论文提出PUCP-Metrix，一个包含182个西班牙语语言学度量的开源库，涵盖词汇多样性、句法语义复杂度等维度。针对西班牙语工具覆盖不足的问题，该工作提供了细粒度可解释的文本分析能力，在可读性评估和机器生成文本检测任务中表现优异，支持多种自然语言处理应用。**

- **链接: [https://arxiv.org/pdf/2511.17402v1](https://arxiv.org/pdf/2511.17402v1)**

> **作者:** Javier Alonso Villegas Luis; Marco Antonio Sobrevilla Cabezudo
>
> **备注:** 1 figure, to be submitted to EACL Demo track
>
> **摘要:** Linguistic features remain essential for interpretability and tasks involving style, structure, and readability, but existing Spanish tools offer limited coverage. We present PUCP-Metrix, an open-source repository of 182 linguistic metrics spanning lexical diversity, syntactic and semantic complexity, cohesion, psycholinguistics, and readability. PUCP-Metrix enables fine-grained, interpretable text analysis. We evaluate its usefulness on Automated Readability Assessment and Machine-Generated Text Detection, showing competitive performance compared to an existing repository and strong neural baselines. PUCP-Metrix offers a comprehensive, extensible resource for Spanish, supporting diverse NLP applications.
>
---
#### [new 013] The PLLuM Instruction Corpus
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍用于微调波兰语大语言模型（PLLuM）的指令数据集。针对语言模型本地化中高质量指令数据稀缺的问题，提出有机、转换和合成三类指令的分类体系，并发布首个代表性子集PLLuMIC，为其他语言模型数据集构建提供参考。**

- **链接: [https://arxiv.org/pdf/2511.17161v1](https://arxiv.org/pdf/2511.17161v1)**

> **作者:** Piotr Pęzik; Filip Żarnecki; Konrad Kaczyński; Anna Cichosz; Zuzanna Deckert; Monika Garnys; Izabela Grabarczyk; Wojciech Janowski; Sylwia Karasińska; Aleksandra Kujawiak; Piotr Misztela; Maria Szymańska; Karolina Walkusz; Igor Siek; Maciej Chrabąszcz; Anna Kołos; Agnieszka Karlińska; Karolina Seweryn; Aleksandra Krasnodębska; Paula Betscher; Zofia Cieślińska; Katarzyna Kowol; Artur Wilczek; Maciej Trzciński; Katarzyna Dziewulska; Roman Roszko; Tomasz Bernaś; Jurgita Vaičenonienė; Danuta Roszko; Paweł Levchuk; Paweł Kowalski; Irena Prawdzic-Jankowska; Marek Kozłowski; Sławomir Dadas; Rafał Poświata; Alina Wróblewska; Katarzyna Krasnowska-Kieraś; Maciej Ogrodniczuk; Michał Rudolf; Piotr Rybak; Karolina Saputa; Joanna Wołoszyn; Marcin Oleksy; Bartłomiej Koptyra; Teddy Ferdinan; Stanisław Woźniak; Maciej Piasecki; Paweł Walkowiak; Konrad Wojtasik; Arkadiusz Janz; Przemysław Kazienko; Julia Moska; Jan Kocoń
>
> **摘要:** This paper describes the instruction dataset used to fine-tune a set of transformer-based large language models (LLMs) developed in the PLLuM (Polish Large Language Model) project. We present a functional typology of the organic, converted, and synthetic instructions used in PLLuM and share some observations about the implications of using human-authored versus synthetic instruction datasets in the linguistic adaptation of base LLMs. Additionally, we release the first representative subset of the PLLuM instruction corpus (PLLuMIC), which we believe to be useful in guiding and planning the development of similar datasets for other LLMs.
>
---
#### [new 014] Interpretable dimensions support an effect of agentivity and telicity on split intransitivity
- **分类: cs.CL**

- **简介: 该论文研究句法中不及物动词的分叉现象（无标记/受标记），旨在验证语义属性（主动性、目的性）对句法类型的影响。针对近期研究质疑人类评分预测力的问题，作者采用可解释维度方法，基于对立语义种子词计算语义维度，结果支持主动性和目的性与句法类型的相关性，证明该方法能有效补充传统评分。**

- **链接: [https://arxiv.org/pdf/2511.16824v1](https://arxiv.org/pdf/2511.16824v1)**

> **作者:** Eva Neu; Brian Dillon; Katrin Erk
>
> **摘要:** Intransitive verbs fall into two different syntactic classes, unergatives and unaccusatives. It has long been argued that verbs describing an agentive action are more likely to appear in an unergative syntax, and those describing a telic event to appear in an unaccusative syntax. However, recent work by Kim et al. (2024) found that human ratings for agentivity and telicity were a poor predictor of the syntactic behavior of intransitives. Here we revisit this question using interpretable dimensions, computed from seed words on opposite poles of the agentive and telic scales. Our findings support the link between unergativity/unaccusativity and agentivity/telicity, and demonstrate that using interpretable dimensions in conjunction with human judgments can offer valuable evidence for semantic properties that are not easily evaluated in rating tasks.
>
---
#### [new 015] ConCISE: A Reference-Free Conciseness Evaluation Metric for LLM-Generated Answers
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型生成回答冗长的问题，提出一种无需参考答案的简洁性评估指标ConCISE。通过计算原文与摘要的压缩比及词移除压缩率，量化非必要内容，有效评估生成文本的简洁性，为对话AI系统提供自动化评价工具。**

- **链接: [https://arxiv.org/pdf/2511.16846v1](https://arxiv.org/pdf/2511.16846v1)**

> **作者:** Seyed Mohssen Ghafari; Ronny Kol; Juan C. Quiroz; Nella Luan; Monika Patial; Chanaka Rupasinghe; Herman Wandabwa; Luiz Pizzato
>
> **摘要:** Large language models (LLMs) frequently generate responses that are lengthy and verbose, filled with redundant or unnecessary details. This diminishes clarity and user satisfaction, and it increases costs for model developers, especially with well-known proprietary models that charge based on the number of output tokens. In this paper, we introduce a novel reference-free metric for evaluating the conciseness of responses generated by LLMs. Our method quantifies non-essential content without relying on gold standard references and calculates the average of three calculations: i) a compression ratio between the original response and an LLM abstractive summary; ii) a compression ratio between the original response and an LLM extractive summary; and iii) wordremoval compression, where an LLM removes as many non-essential words as possible from the response while preserving its meaning, with the number of tokens removed indicating the conciseness score. Experimental results demonstrate that our proposed metric identifies redundancy in LLM outputs, offering a practical tool for automated evaluation of response brevity in conversational AI systems without the need for ground truth human annotations.
>
---
#### [new 016] Deep Improvement Supervision
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究小规模循环模型（TRMs）在复杂推理任务中的效率优化问题。针对训练低效、依赖停顿机制的缺陷，提出基于隐式策略改进的新型训练方案，通过为每轮迭代提供目标，显著提升训练效率，减少18倍前向传播，且无需停顿机制，仅用0.8M参数即在ARC-1上达到24%准确率，超越多数大模型。**

- **链接: [https://arxiv.org/pdf/2511.16886v1](https://arxiv.org/pdf/2511.16886v1)**

> **作者:** Arip Asadulaev; Rayan Banerjee; Fakhri Karray; Martin Takac
>
> **摘要:** Recently, it was shown that small, looped architectures, such as Tiny Recursive Models (TRMs), can outperform Large Language Models (LLMs) on complex reasoning tasks, including the Abstraction and Reasoning Corpus (ARC). In this work, we investigate a core question: how can we further improve the efficiency of these methods with minimal changes? To address this, we frame the latent reasoning of TRMs as a form of classifier-free guidance and implicit policy improvement algorithm. Building on these insights, we propose a novel training scheme that provides a target for each loop during training. We demonstrate that our approach significantly enhances training efficiency. Our method reduces the total number of forward passes by 18x and eliminates halting mechanisms, while maintaining quality comparable to standard TRMs. Notably, we achieve 24% accuracy on ARC-1 with only 0.8M parameters, outperforming most LLMs.
>
---
#### [new 017] Supervised Fine Tuning of Large Language Models for Domain Specific Knowledge Graph Construction:A Case Study on Hunan's Historical Celebrities
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对低资源环境下历史人物知识图谱构建难题，提出基于指令微调的领域适配方法。通过设计领域特定指令模板构建训练数据，对4个大模型进行参数高效微调，显著提升其在湖南历史名人信息抽取中的性能，验证了垂直领域微调在文化遗产数字化中的有效性。**

- **链接: [https://arxiv.org/pdf/2511.17012v1](https://arxiv.org/pdf/2511.17012v1)**

> **作者:** Junjie Hao; Chun Wang; Ying Qiao; Qiuyue Zuo; Qiya Song; Hua Ma; Xieping Gao
>
> **摘要:** Large language models and knowledge graphs offer strong potential for advancing research on historical culture by supporting the extraction, analysis, and interpretation of cultural heritage. Using Hunan's modern historical celebrities shaped by Huxiang culture as a case study, pre-trained large models can help researchers efficiently extract key information, including biographical attributes, life events, and social relationships, from textual sources and construct structured knowledge graphs. However, systematic data resources for Hunan's historical celebrities remain limited, and general-purpose models often underperform in domain knowledge extraction and structured output generation in such low-resource settings. To address these issues, this study proposes a supervised fine-tuning approach for enhancing domain-specific information extraction. First, we design a fine-grained, schema-guided instruction template tailored to the Hunan historical celebrities domain and build an instruction-tuning dataset to mitigate the lack of domain-specific training corpora. Second, we apply parameter-efficient instruction fine-tuning to four publicly available large language models - Qwen2.5-7B, Qwen3-8B, DeepSeek-R1-Distill-Qwen-7B, and Llama-3.1-8B-Instruct - and develop evaluation criteria for assessing their extraction performance. Experimental results show that all models exhibit substantial performance gains after fine-tuning. Among them, Qwen3-8B achieves the strongest results, reaching a score of 89.3866 with 100 samples and 50 training iterations. This study provides new insights into fine-tuning vertical large language models for regional historical and cultural domains and highlights their potential for cost-effective applications in cultural heritage knowledge extraction and knowledge graph construction.
>
---
#### [new 018] NALA_MAINZ at BLP-2025 Task 2: A Multi-agent Approach for Bangla Instruction to Python Code Generation
- **分类: cs.CL; cs.SE**

- **简介: 该论文针对BLP-2025任务2，解决从孟加拉语指令生成Python代码的问题。提出多智能体框架：先由代码生成智能体产出初始代码，再通过调试智能体分析失败测试用例，结合错误信息迭代优化代码。最终以95.4%的Pass@1得分夺冠，并开源代码。**

- **链接: [https://arxiv.org/pdf/2511.16787v1](https://arxiv.org/pdf/2511.16787v1)**

> **作者:** Hossain Shaikh Saadi; Faria Alam; Mario Sanz-Guerrero; Minh Duc Bui; Manuel Mager; Katharina von der Wense
>
> **备注:** BLP 2025 Shared Task 2 - Code Generation in Bangla
>
> **摘要:** This paper presents JGU Mainz's winning system for the BLP-2025 Shared Task on Code Generation from Bangla Instructions. We propose a multi-agent-based pipeline. First, a code-generation agent produces an initial solution from the input instruction. The candidate program is then executed against the provided unit tests (pytest-style, assert-based). Only the failing cases are forwarded to a debugger agent, which reruns the tests, extracts error traces, and, conditioning on the error messages, the current program, and the relevant test cases, generates a revised solution. Using this approach, our submission achieved first place in the shared task with a $Pass@1$ score of 95.4. We also make our code public.
>
---
#### [new 019] A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents
- **分类: cs.CL**

- **简介: 该论文针对大模型对话代理在长期交互中记忆保持困难的问题，提出基于事件语义的事件中心记忆框架。通过将对话分解为带实体和时间线索的事件单元，构建异构图结构实现关联回忆，支持高效检索与证据聚合，在多个基准上优于现有方法，显著缩短问答上下文长度。**

- **链接: [https://arxiv.org/pdf/2511.17208v1](https://arxiv.org/pdf/2511.17208v1)**

> **作者:** Sizhe Zhou
>
> **备注:** Work in progress
>
> **摘要:** LLM-based conversational agents still struggle to maintain coherent, personalized interaction over many sessions: fixed context windows limit how much history can be kept in view, and most external memory approaches trade off between coarse retrieval over large chunks and fine-grained but fragmented views of the dialogue. Motivated by neo-Davidsonian event semantics, we propose an event-centric alternative that represents conversational history as short, event-like propositions which bundle together participants, temporal cues, and minimal local context, rather than as independent relation triples or opaque summaries. In contrast to work that aggressively compresses or forgets past content, our design aims to preserve information in a non-compressive form and make it more accessible, rather than more lossy. Concretely, we instruct an LLM to decompose each session into enriched elementary discourse units (EDUs) -- self-contained statements with normalized entities and source turn attributions -- and organize sessions, EDUs, and their arguments in a heterogeneous graph that supports associative recall. On top of this representation we build two simple retrieval-based variants that use dense similarity search and LLM filtering, with an optional graph-based propagation step to connect and aggregate evidence across related EDUs. Experiments on the LoCoMo and LongMemEval$_S$ benchmarks show that these event-centric memories match or surpass strong baselines, while operating with much shorter QA contexts. Our results suggest that structurally simple, event-level memory provides a principled and practical foundation for long-horizon conversational agents. Our code and data will be released at https://github.com/KevinSRR/EMem.
>
---
#### [new 020] Lost in Translation and Noise: A Deep Dive into the Failure Modes of VLMs on Real-World Tables
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对视觉语言模型（VLMs）在真实表格问答任务中的失效问题，提出多语言、带视觉噪声的基准数据集MirageTVQA。旨在解决现有数据集过于理想化、缺乏真实场景挑战的问题，揭示模型在视觉噪声和跨语言迁移上的严重性能下降。**

- **链接: [https://arxiv.org/pdf/2511.17238v1](https://arxiv.org/pdf/2511.17238v1)**

> **作者:** Anshul Singh; Rohan Chaudhary; Gagneet Singh; Abhay Kumary
>
> **备注:** Accepted as Spotligh Talk at EurIPS 2025 Workshop on AI For Tabular Data
>
> **摘要:** The impressive performance of VLMs is largely measured on benchmarks that fail to capture the complexities of real-world scenarios. Existing datasets for tabular QA, such as WikiTableQuestions and FinQA, are overwhelmingly monolingual (English) and present tables in a digitally perfect, clean format. This creates a significant gap between research and practice. To address this, we present \textbf{MirageTVQA}, a new benchmark designed to evaluate VLMs on these exact dimensions. Featuring nearly 60,000 QA pairs across 24 languages, MirageTVQA challenges models with tables that are not only multilingual but also visually imperfect, incorporating realistic noise to mimic scanned documents. Our evaluation of the leading VLMs reveals two primary failure points: a severe degradation in performance (over 35\% drop for the best models) when faced with visual noise and a consistent English-first bias where reasoning abilities fail to transfer to other languages. MirageTVQA provides a benchmark for measuring and driving progress towards more robust VLM models for table reasoning. The dataset and the code are available at: https://github.com/anshulsc/MirageTVQA.
>
---
#### [new 021] Masked-and-Reordered Self-Supervision for Reinforcement Learning from Verifiable Rewards
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对强化学习中仅结果可验证的数学推理任务，提出MR-RLVR方法，通过“掩码填充”与“步骤重排”构建过程级自监督信号，提升模型长链推理能力。在固定预算下，显著优于原RLVR，在多个数据集上实现性能提升。**

- **链接: [https://arxiv.org/pdf/2511.17473v1](https://arxiv.org/pdf/2511.17473v1)**

> **作者:** Zhen Wang; Zhifeng Gao; Guolin Ke
>
> **摘要:** Test-time scaling has been shown to substantially improve large language models' (LLMs) mathematical reasoning. However, for a large portion of mathematical corpora, especially theorem proving, RLVR's scalability is limited: intermediate reasoning is crucial, while final answers are difficult to directly and reliably verify. Meanwhile, token-level SFT often degenerates into rote memorization rather than inducing longer chains of thought. Inspired by BERT's self-supervised tasks, we propose MR-RLVR (Masked-and-Reordered RLVR), which constructs process-level self-supervised rewards via "masked-then-fill" and "step reordering" to extract learnable signals from intermediate reasoning. Our training pipeline comprises two stages: we first perform self-supervised training on sampled mathematical calculation and proof data; we then conduct RLVR fine-tuning on mathematical calculation datasets where only outcomes are verifiable. We implement MR-RLVR on Qwen2.5-3B and DeepSeek-R1-Distill-Qwen-1.5B, and evaluate on AIME24, AIME25, AMC23, and MATH500. Under a fixed sampling and decoding budget, MR-RLVR achieves average relative gains over the original RLVR of +9.86% Pass@1, +5.27% Pass@5, and +4.00% Pass@8. These results indicate that incorporating process-aware self-supervised signals can effectively enhance RLVR's scalability and performance in only outcome-verifiable settings.
>
---
#### [new 022] Beyond Multiple Choice: A Hybrid Framework for Unifying Robust Evaluation and Verifiable Reasoning Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多选题问答（MCQA）评估中因选项泄露信号导致的指标失真问题，提出ReVeL框架。通过将多选题重写为可验证的开放题，提升训练与评估的可靠性。实验表明，该方法在保持多选性能的同时显著提升开放题表现，降低评估成本与延迟，实现更稳健的模型训练与评价。**

- **链接: [https://arxiv.org/pdf/2511.17405v1](https://arxiv.org/pdf/2511.17405v1)**

> **作者:** Yesheng Liu; Hao Li; Haiyu Xu; Baoqi Pei; Jiahao Wang; Mingxuan Zhao; Jingshu Zheng; Zheqi He; JG Yao; Bowen Qin; Xi Yang; Jiajun Zhang
>
> **备注:** Project url: https://flageval-baai.github.io/ReVeL/
>
> **摘要:** Multiple-choice question answering (MCQA) has been a popular format for evaluating and reinforcement fine-tuning (RFT) of modern multimodal language models. Its constrained output format allows for simplified, deterministic automatic verification. However, we find that the options may leak exploitable signals, which makes the accuracy metrics unreliable for indicating real capabilities and encourages explicit or implicit answer guessing behaviors during RFT. We propose ReVeL (Rewrite and Verify by LLM), a framework that rewrites multiple-choice questions into open-form questions while keeping answers verifiable whenever possible. The framework categorizes questions according to different answer types, apply different rewriting and verification schemes, respectively. When applied for RFT, we converted 20k MCQA examples and use GRPO to finetune Qwen2.5-VL models. Models trained on ReVeL-OpenQA match MCQA accuracy on multiple-choice benchmarks and improve OpenQA accuracy by about six percentage points, indicating better data efficiency and more robust reward signals than MCQA-based training. When used for evaluation, ReVeL also reveals up to 20 percentage points of score inflation in MCQA benchmarks (relative to OpenQA), improves judging accuracy, and reduces both cost and latency. We will release code and data publicly.
>
---
#### [new 023] Predicting the Formation of Induction Heads
- **分类: cs.CL**

- **简介: 该论文研究语言模型中诱导头（IH）的形成机制，旨在揭示训练数据统计特性与IH出现之间的关系。通过分析自然与合成数据，发现批大小与上下文大小的乘积可预测IH形成点，且大二元组重复频率与可靠性决定其形成，存在精确的帕累托前沿。**

- **链接: [https://arxiv.org/pdf/2511.16893v1](https://arxiv.org/pdf/2511.16893v1)**

> **作者:** Tatsuya Aoyama; Ethan Gotlieb Wilcox; Nathan Schneider
>
> **备注:** Accepted to CogInterp @ NeurIPS
>
> **摘要:** Arguably, specialized attention heads dubbed induction heads (IHs) underlie the remarkable in-context learning (ICL) capabilities of modern language models (LMs); yet, a precise characterization of their formation remains unclear. In this study, we investigate the relationship between statistical properties of training data (for both natural and synthetic data) and IH formation. We show that (1) a simple equation combining batch size and context size predicts the point at which IHs form; (2) surface bigram repetition frequency and reliability strongly affect the formation of IHs, and we find a precise Pareto frontier in terms of these two values; and (3) local dependency with high bigram repetition frequency and reliability is sufficient for IH formation, but when the frequency and reliability are low, categoriality and the shape of the marginal distribution matter.
>
---
#### [new 024] Detecting and Steering LLMs' Empathy in Action
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大模型在行动中的共情能力（Empathy-in-Action），旨在检测与调控其共情行为。通过对比不同模型在特定提示下的表现，发现共情可作为激活空间中的线性方向，且安全训练不影响共情编码。研究验证了检测与可控性，揭示模型间实现差异及安全训练对操控鲁棒性的影响。**

- **链接: [https://arxiv.org/pdf/2511.16699v1](https://arxiv.org/pdf/2511.16699v1)**

> **作者:** Juan P. Cadile
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** We investigate empathy-in-action -- the willingness to sacrifice task efficiency to address human needs -- as a linear direction in LLM activation space. Using contrastive prompts grounded in the Empathy-in-Action (EIA) benchmark, we test detection and steering across Phi-3-mini-4k (3.8B), Qwen2.5-7B (safety-trained), and Dolphin-Llama-3.1-8B (uncensored). Detection: All models show AUROC 0.996-1.00 at optimal layers. Uncensored Dolphin matches safety-trained models, demonstrating empathy encoding emerges independent of safety training. Phi-3 probes correlate strongly with EIA behavioral scores (r=0.71, p<0.01). Cross-model probe agreement is limited (Qwen: r=-0.06, Dolphin: r=0.18), revealing architecture-specific implementations despite convergent detection. Steering: Qwen achieves 65.3% success with bidirectional control and coherence at extreme interventions. Phi-3 shows 61.7% success with similar coherence. Dolphin exhibits asymmetric steerability: 94.4% success for pro-empathy steering but catastrophic breakdown for anti-empathy (empty outputs, code artifacts). Implications: The detection-steering gap varies by model. Qwen and Phi-3 maintain bidirectional coherence; Dolphin shows robustness only for empathy enhancement. Safety training may affect steering robustness rather than preventing manipulation, though validation across more models is needed.
>
---
#### [new 025] MUCH: A Multilingual Claim Hallucination Benchmark
- **分类: cs.CL**

- **简介: 该论文提出MUCH，首个多语言声明级不确定性量化（UQ）基准，旨在公平评估LLM可靠性。针对现有方法在多语言、实时性与可复现性上的不足，构建了跨四语种4873样本数据集，提供细粒度生成日志，并设计高效确定性分割算法，支持白盒方法开发与实时监控。**

- **链接: [https://arxiv.org/pdf/2511.17081v1](https://arxiv.org/pdf/2511.17081v1)**

> **作者:** Jérémie Dentan; Alexi Canesse; Davide Buscaldi; Aymen Shabou; Sonia Vanier
>
> **摘要:** Claim-level Uncertainty Quantification (UQ) is a promising approach to mitigate the lack of reliability in Large Language Models (LLMs). We introduce MUCH, the first claim-level UQ benchmark designed for fair and reproducible evaluation of future methods under realistic conditions. It includes 4,873 samples across four European languages (English, French, Spanish, and German) and four instruction-tuned open-weight LLMs. Unlike prior claim-level benchmarks, we release 24 generation logits per token, facilitating the development of future white-box methods without re-generating data. Moreover, in contrast to previous benchmarks that rely on manual or LLM-based segmentation, we propose a new deterministic algorithm capable of segmenting claims using as little as 0.2% of the LLM generation time. This makes our segmentation approach suitable for real-time monitoring of LLM outputs, ensuring that MUCH evaluates UQ methods under realistic deployment constraints. Finally, our evaluations show that current methods still have substantial room for improvement in both performance and efficiency.
>
---
#### [new 026] PEPPER: Perception-Guided Perturbation for Robust Backdoor Defense in Text-to-Image Diffusion Models
- **分类: cs.CL**

- **简介: 该论文针对文本到图像扩散模型的后门攻击问题，提出PEPPER防御方法。通过语义差异大但视觉相似的文本重写，干扰触发词并削弱攻击影响，有效提升模型鲁棒性，同时兼容现有防御方法，保持生成质量。**

- **链接: [https://arxiv.org/pdf/2511.16830v1](https://arxiv.org/pdf/2511.16830v1)**

> **作者:** Oscar Chew; Po-Yi Lu; Jayden Lin; Kuan-Hao Huang; Hsuan-Tien Lin
>
> **摘要:** Recent studies show that text to image (T2I) diffusion models are vulnerable to backdoor attacks, where a trigger in the input prompt can steer generation toward harmful or unintended content. To address this, we introduce PEPPER (PErcePtion Guided PERturbation), a backdoor defense that rewrites the caption into a semantically distant yet visually similar caption while adding unobstructive elements. With this rewriting strategy, PEPPER disrupt the trigger embedded in the input prompt, dilute the influence of trigger tokens and thereby achieve enhanced robustness. Experiments show that PEPPER is particularly effective against text encoder based attacks, substantially reducing attack success while preserving generation quality. Beyond this, PEPPER can be paired with any existing defenses yielding consistently stronger and generalizable robustness than any standalone method. Our code will be released on Github.
>
---
#### [new 027] ARQUSUMM: Argument-aware Quantitative Summarization of Online Conversations
- **分类: cs.CL**

- **简介: 该论文提出“论点感知的量化对话摘要”任务，旨在揭示在线对话中论点与理由的结构及强度。针对现有方法忽视句子内论证结构的问题，提出ARQUSUMM框架，结合大模型少样本学习与论据聚类，实现论点识别、结构解析与量化，显著提升摘要质量与准确性。**

- **链接: [https://arxiv.org/pdf/2511.16985v1](https://arxiv.org/pdf/2511.16985v1)**

> **作者:** An Quang Tang; Xiuzhen Zhang; Minh Ngoc Dinh; Zhuang Li
>
> **备注:** Paper accepted to AAAI2026 Main Technical Track
>
> **摘要:** Online conversations have become more prevalent on public discussion platforms (e.g. Reddit). With growing controversial topics, it is desirable to summarize not only diverse arguments, but also their rationale and justification. Early studies on text summarization focus on capturing general salient information in source documents, overlooking the argumentative nature of online conversations. Recent research on conversation summarization although considers the argumentative relationship among sentences, fail to explicate deeper argument structure within sentences for summarization. In this paper, we propose a novel task of argument-aware quantitative summarization to reveal the claim-reason structure of arguments in conversations, with quantities measuring argument strength. We further propose ARQUSUMM, a novel framework to address the task. To reveal the underlying argument structure within sentences, ARQUSUMM leverages LLM few-shot learning grounded in the argumentation theory to identify propositions within sentences and their claim-reason relationships. For quantitative summarization, ARQUSUMM employs argument structure-aware clustering algorithms to aggregate arguments and quantify their support. Experiments show that ARQUSUMM outperforms existing conversation and quantitative summarization models and generate summaries representing argument structures that are more helpful to users, of high textual quality and quantification accuracy.
>
---
#### [new 028] Training Foundation Models on a Full-Stack AMD Platform: Compute, Networking, and System Design
- **分类: cs.CL; cs.AI; cs.DC**

- **简介: 该论文研究在纯AMD硬件上训练大规模混合专家（MoE）模型，解决高性能计算平台在大模型训练中的适配问题。工作包括系统级微基准测试、模型设计优化及完整训练栈构建，验证了AMD平台具备支持前沿大模型训练的能力。**

- **链接: [https://arxiv.org/pdf/2511.17127v1](https://arxiv.org/pdf/2511.17127v1)**

> **作者:** Quentin Anthony; Yury Tokpanov; Skyler Szot; Srivatsan Rajagopal; Praneeth Medepalli; Rishi Iyer; Vasu Shyam; Anna Golubeva; Ansh Chaurasia; Xiao Yang; Tomas Figliolia; Robert Washbourne; Drew Thorstensen; Amartey Pearson; Zack Grossbart; Jason van Patten; Emad Barsoum; Zhenyu Gu; Yao Fu; Beren Millidge
>
> **摘要:** We report on the first large-scale mixture-of-experts (MoE) pretraining study on pure AMD hardware, utilizing both MI300X GPUs with Pollara interconnect. We distill practical guidance for both systems and model design. On the systems side, we deliver a comprehensive cluster and networking characterization: microbenchmarks for all core collectives (all-reduce, reduce-scatter, all-gather, broadcast) across message sizes and GPU counts on Pollara. To our knowledge, this is the first at this scale. We further provide MI300X microbenchmarks on kernel sizing and memory bandwidth to inform model design. On the modeling side, we introduce and apply MI300X-aware transformer sizing rules for attention and MLP blocks and justify MoE widths that jointly optimize training throughput and inference latency. We describe our training stack in depth, including often-ignored utilities such as fault-tolerance and checkpoint-reshaping, as well as detailed information on our training recipe. We also provide a preview of our model architecture and base model - ZAYA1 (760M active, 8.3B total parameters MoE) - which will be further improved upon in forthcoming papers. ZAYA1-base achieves performance comparable to leading base models such as Qwen3-4B and Gemma3-12B at its scale and larger, and outperforms models including Llama-3-8B and OLMoE across reasoning, mathematics, and coding benchmarks. Together, these results demonstrate that the AMD hardware, network, and software stack are mature and optimized enough for competitive large-scale pretraining.
>
---
#### [new 029] Reproducibility Report: Test-Time Training on Nearest Neighbors for Large Language Models
- **分类: cs.CL**

- **简介: 该论文复现并验证了基于最近邻的测试时训练方法，旨在通过推理时微调提升大语言模型性能。研究在多个模型和数据集上验证其有效性，发现其能显著降低困惑度，尤其在结构化数据上效果更佳，并提出高效检索方案以降低内存消耗。**

- **链接: [https://arxiv.org/pdf/2511.16691v1](https://arxiv.org/pdf/2511.16691v1)**

> **作者:** Boyang Zhou; Johan Lindqvist; Lindsey Li
>
> **摘要:** We reproduce the central claims of Test-Time Training on Nearest Neighbors for Large Language Models (Hardt and Sun, 2024), which proposes adapting a language model at inference time by fine-tuning on retrieved nearest-neighbor sequences. Using pretrained RoBERTa embeddings indexed with Faiss, we retrieve 20 neighbors per test input and apply one gradient update per neighbor across GPT-2 (117M, 774M), GPT-Neo (1.3B), and R1-Distilled-Qwen2.5-1.5B. Our experiments confirm that test-time training significantly reduces perplexity and bits-per-byte metrics across diverse domains from The Pile, with the largest improvements in structured or specialized datasets such as GitHub and EuroParl. We further validate that models not pretrained on The Pile benefit more from this adaptation than models already trained on similar data, allowing smaller models to approach the performance of larger ones. Due to infrastructure limitations, we introduce a memory-efficient retrieval implementation that loads only required line offsets rather than entire files, reducing RAM requirements from over 128 GB per server to 32 GB. We also extend the original study by evaluating R1-Distilled-Qwen2.5-1.5B, showing that test-time training yields consistent gains even for modern reasoning-optimized architectures. Overall, our results support the robustness and generality of nearest-neighbor test-time training while highlighting practical considerations for reproducing large-scale retrieval-augmented adaptation.
>
---
#### [new 030] Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型幻觉问题，提出基于因果推断的早期拒答框架ABCA。通过分析模型内部知识在不同方面（如学科、时间）的多样性，检测知识冲突或不足，实现提前拒答。有效提升拒答可靠性与可解释性，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.17170v1](https://arxiv.org/pdf/2511.17170v1)**

> **作者:** Vy Nguyen; Ziqi Xu; Jeffrey Chan; Estrid He; Feng Xia; Xiuzhen Zhang
>
> **备注:** Accepted to AAAI 2026 (Main Technical Track)
>
> **摘要:** Large Language Models (LLMs) often produce fluent but factually incorrect responses, a phenomenon known as hallucination. Abstention, where the model chooses not to answer and instead outputs phrases such as "I don't know", is a common safeguard. However, existing abstention methods typically rely on post-generation signals, such as generation variations or feedback, which limits their ability to prevent unreliable responses in advance. In this paper, we introduce Aspect-Based Causal Abstention (ABCA), a new framework that enables early abstention by analysing the internal diversity of LLM knowledge through causal inference. This diversity reflects the multifaceted nature of parametric knowledge acquired from various sources, representing diverse aspects such as disciplines, legal contexts, or temporal frames. ABCA estimates causal effects conditioned on these aspects to assess the reliability of knowledge relevant to a given query. Based on these estimates, we enable two types of abstention: Type-1, where aspect effects are inconsistent (knowledge conflict), and Type-2, where aspect effects consistently support abstention (knowledge insufficiency). Experiments on standard benchmarks demonstrate that ABCA improves abstention reliability, achieves state-of-the-art performance, and enhances the interpretability of abstention decisions.
>
---
#### [new 031] E$^3$-Pruner: Towards Efficient, Economical, and Effective Layer Pruning for Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对大语言模型层剪枝中的性能下降、训练成本高和加速有限问题，提出E³-Pruner框架。通过可微分掩码优化与熵感知知识蒸馏，实现高效、经济、有效的层剪枝，在保持高精度的同时显著降低计算开销。**

- **链接: [https://arxiv.org/pdf/2511.17205v1](https://arxiv.org/pdf/2511.17205v1)**

> **作者:** Tao Yuan; Haoli Bai; Yinfei Pan; Xuyang Cao; Tianyu Zhang; Lu Hou; Ting Hu; Xianzhi Yu
>
> **摘要:** With the increasing size of large language models, layer pruning has gained increased attention as a hardware-friendly approach for model compression. However, existing layer pruning methods struggle to simultaneously address key practical deployment challenges, including performance degradation, high training costs, and limited acceleration. To overcome these limitations, we propose \name, a task-\underline{E}ffective, training-\underline{E}conomical and inference-\underline{E}fficient layer pruning framework. \namespace introduces two key innovations: (1) a differentiable mask optimization method using a Gumbel-TopK sampler, enabling efficient and precise pruning mask search; and (2) an entropy-aware adaptive knowledge distillation strategy that enhances task performance. Extensive experiments over diverse model architectures and benchmarks demonstrate the superiority of our method over state-of-the-art approaches. Notably, \namespace achieves 96\% accuracy, a mere 0.8\% drop from the original model (96.8\%) on MATH-500 when pruning 25\% layers of Qwen3-32B, outperforming existing SOTA (95\%), with a 1.33$\times$ inference speedup by consuming merely 0.5B tokens (0.5\% of the post-training data volume).
>
---
#### [new 032] Attention-Guided Feature Fusion (AGFF) Model for Integrating Statistical and Semantic Features in News Text Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对新闻文本分类任务，解决统计特征与语义特征融合不足的问题。提出注意力引导的特征融合（AGFF）模型，动态加权两类特征，提升分类精度。实验表明，该模型有效结合了两者优势，优于传统方法和纯深度学习模型。**

- **链接: [https://arxiv.org/pdf/2511.17184v1](https://arxiv.org/pdf/2511.17184v1)**

> **作者:** Mohammad Zare
>
> **摘要:** News text classification is a crucial task in natural language processing, essential for organizing and filtering the massive volume of digital content. Traditional methods typically rely on statistical features like term frequencies or TF-IDF values, which are effective at capturing word-level importance but often fail to reflect contextual meaning. In contrast, modern deep learning approaches utilize semantic features to understand word usage within context, yet they may overlook simple, high-impact statistical indicators. This paper introduces an Attention-Guided Feature Fusion (AGFF) model that combines statistical and semantic features in a unified framework. The model applies an attention-based mechanism to dynamically determine the relative importance of each feature type, enabling more informed classification decisions. Through evaluation on benchmark news datasets, the AGFF model demonstrates superior performance compared to both traditional statistical models and purely semantic deep learning models. The results confirm that strategic integration of diverse feature types can significantly enhance classification accuracy. Additionally, ablation studies validate the contribution of each component in the fusion process. The findings highlight the model's ability to balance and exploit the complementary strengths of statistical and semantic representations, making it a practical and effective solution for real-world news classification tasks.
>
---
#### [new 033] Ellipsoid-Based Decision Boundaries for Open Intent Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对开放意图分类任务，解决现有方法假设类别分布各向同性、边界僵化的问题。提出EliDecide，通过监督对比学习构建判别特征空间，用可学习矩阵参数化椭球边界，结合双损失优化，实现方向自适应的灵活决策边界，显著提升未知意图检测性能。**

- **链接: [https://arxiv.org/pdf/2511.16685v1](https://arxiv.org/pdf/2511.16685v1)**

> **作者:** Yuetian Zou; Hanlei Zhang; Hua Xu; Songze Li; Long Xiao
>
> **摘要:** Textual open intent classification is crucial for real-world dialogue systems, enabling robust detection of unknown user intents without prior knowledge and contributing to the robustness of the system. While adaptive decision boundary methods have shown great potential by eliminating manual threshold tuning, existing approaches assume isotropic distributions of known classes, restricting boundaries to balls and overlooking distributional variance along different directions. To address this limitation, we propose EliDecide, a novel method that learns ellipsoid decision boundaries with varying scales along different feature directions. First, we employ supervised contrastive learning to obtain a discriminative feature space for known samples. Second, we apply learnable matrices to parameterize ellipsoids as the boundaries of each known class, offering greater flexibility than spherical boundaries defined solely by centers and radii. Third, we optimize the boundaries via a novelly designed dual loss function that balances empirical and open-space risks: expanding boundaries to cover known samples while contracting them against synthesized pseudo-open samples. Our method achieves state-of-the-art performance on multiple text intent benchmarks and further on a question classification dataset. The flexibility of the ellipsoids demonstrates superior open intent detection capability and strong potential for generalization to more text classification tasks in diverse complex open-world scenarios.
>
---
#### [new 034] Principled Design of Interpretable Automated Scoring for Large-Scale Educational Assessments
- **分类: cs.CL**

- **简介: 该论文针对大規模教育評估中自動化評分缺乏可解釋性的問題，提出四項可解釋性原則（FGTI），並設計AnalyticScore框架。該框架通過LLM提取可識別元素、生成人類可理解特徵，並使用直觀的有序邏輯回歸模型進行評分，在準確率上優於多數不可解釋方法，且接近最前沿模型表現。**

- **链接: [https://arxiv.org/pdf/2511.17069v1](https://arxiv.org/pdf/2511.17069v1)**

> **作者:** Yunsung Kim; Mike Hardy; Joseph Tey; Candace Thille; Chris Piech
>
> **备注:** 16 pages, 2 figures
>
> **摘要:** AI-driven automated scoring systems offer scalable and efficient means of evaluating complex student-generated responses. Yet, despite increasing demand for transparency and interpretability, the field has yet to develop a widely accepted solution for interpretable automated scoring to be used in large-scale real-world assessments. This work takes a principled approach to address this challenge. We analyze the needs and potential benefits of interpretable automated scoring for various assessment stakeholders and develop four principles of interpretability -- Faithfulness, Groundedness, Traceability, and Interchangeability (FGTI) -- targeted at those needs. To illustrate the feasibility of implementing these principles, we develop the AnalyticScore framework for short answer scoring as a baseline reference framework for future research. AnalyticScore operates by (1) extracting explicitly identifiable elements of the responses, (2) featurizing each response into human-interpretable values using LLMs, and (3) applying an intuitive ordinal logistic regression model for scoring. In terms of scoring accuracy, AnalyticScore outperforms many uninterpretable scoring methods, and is within only 0.06 QWK of the uninterpretable SOTA on average across 10 items from the ASAP-SAS dataset. By comparing against human annotators conducting the same featurization task, we further demonstrate that the featurization behavior of AnalyticScore aligns well with that of humans.
>
---
#### [new 035] AutoLink: Autonomous Schema Exploration and Expansion for Scalable Schema Linking in Text-to-SQL at Scale
- **分类: cs.CL; cs.DB**

- **简介: 该论文针对工业级文本转SQL中的大规模模式链接问题，提出自主代理框架AutoLink。通过迭代式、基于LLM的动态探索与扩展，实现高效高召回的模式子集筛选，有效解决上下文窗口限制与大库缩放难题，显著提升可扩展性与执行准确率。**

- **链接: [https://arxiv.org/pdf/2511.17190v1](https://arxiv.org/pdf/2511.17190v1)**

> **作者:** Ziyang Wang; Yuanlei Zheng; Zhenbiao Cao; Xiaojin Zhang; Zhongyu Wei; Pei Fu; Zhenbo Luo; Wei Chen; Xiang Bai
>
> **摘要:** For industrial-scale text-to-SQL, supplying the entire database schema to Large Language Models (LLMs) is impractical due to context window limits and irrelevant noise. Schema linking, which filters the schema to a relevant subset, is therefore critical. However, existing methods incur prohibitive costs, struggle to trade off recall and noise, and scale poorly to large databases. We present \textbf{AutoLink}, an autonomous agent framework that reformulates schema linking as an iterative, agent-driven process. Guided by an LLM, AutoLink dynamically explores and expands the linked schema subset, progressively identifying necessary schema components without inputting the full database schema. Our experiments demonstrate AutoLink's superior performance, achieving state-of-the-art strict schema linking recall of \textbf{97.4\%} on Bird-Dev and \textbf{91.2\%} on Spider-2.0-Lite, with competitive execution accuracy, i.e., \textbf{68.7\%} EX on Bird-Dev (better than CHESS) and \textbf{34.9\%} EX on Spider-2.0-Lite (ranking 2nd on the official leaderboard). Crucially, AutoLink exhibits \textbf{exceptional scalability}, \textbf{maintaining high recall}, \textbf{efficient token consumption}, and \textbf{robust execution accuracy} on large schemas (e.g., over 3,000 columns) where existing methods severely degrade-making it a highly scalable, high-recall schema-linking solution for industrial text-to-SQL systems.
>
---
#### [new 036] How Well Do LLMs Understand Tunisian Arabic?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLM）对突尼斯阿拉伯语（Tunizi）的理解能力，属于自然语言处理中的多语言理解任务。针对工业级模型对低资源语言支持不足的问题，作者构建了包含平行翻译与情感标签的新数据集，并在音译、翻译和情感分析三任务上评估多个LLM，揭示其在方言理解上的差距，呼吁未来AI系统应更包容低资源语言。**

- **链接: [https://arxiv.org/pdf/2511.16683v1](https://arxiv.org/pdf/2511.16683v1)**

> **作者:** Mohamed Mahdi
>
> **摘要:** Large Language Models (LLMs) are the engines driving today's AI agents. The better these models understand human languages, the more natural and user-friendly the interaction with AI becomes, from everyday devices like computers and smartwatches to any tool that can act intelligently. Yet, the ability of industrial-scale LLMs to comprehend low-resource languages, such as Tunisian Arabic (Tunizi), is often overlooked. This neglect risks excluding millions of Tunisians from fully interacting with AI in their own language, pushing them toward French or English. Such a shift not only threatens the preservation of the Tunisian dialect but may also create challenges for literacy and influence younger generations to favor foreign languages. In this study, we introduce a novel dataset containing parallel Tunizi, standard Tunisian Arabic, and English translations, along with sentiment labels. We benchmark several popular LLMs on three tasks: transliteration, translation, and sentiment analysis. Our results reveal significant differences between models, highlighting both their strengths and limitations in understanding and processing Tunisian dialects. By quantifying these gaps, this work underscores the importance of including low-resource languages in the next generation of AI systems, ensuring technology remains accessible, inclusive, and culturally grounded.
>
---
#### [new 037] SMILE: A Composite Lexical-Semantic Metric for Question-Answering Evaluation
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对文本与视觉问答评估中传统指标忽视语义、现有方法缺乏灵活性或成本过高的问题，提出SMILE——一种融合句级语义与关键词级语义及精确匹配的复合评估指标。它在保持计算轻量的同时，显著提升与人类判断的相关性，有效平衡了词汇精确性与语义相关性。**

- **链接: [https://arxiv.org/pdf/2511.17432v1](https://arxiv.org/pdf/2511.17432v1)**

> **作者:** Shrikant Kendre; Austin Xu; Honglu Zhou; Michael Ryoo; Shafiq Joty; Juan Carlos Niebles
>
> **备注:** 23 pages, 6 tables, 9 figures
>
> **摘要:** Traditional evaluation metrics for textual and visual question answering, like ROUGE, METEOR, and Exact Match (EM), focus heavily on n-gram based lexical similarity, often missing the deeper semantic understanding needed for accurate assessment. While measures like BERTScore and MoverScore leverage contextual embeddings to address this limitation, they lack flexibility in balancing sentence-level and keyword-level semantics and ignore lexical similarity, which remains important. Large Language Model (LLM) based evaluators, though powerful, come with drawbacks like high costs, bias, inconsistency, and hallucinations. To address these issues, we introduce SMILE: Semantic Metric Integrating Lexical Exactness, a novel approach that combines sentence-level semantic understanding with keyword-level semantic understanding and easy keyword matching. This composite method balances lexical precision and semantic relevance, offering a comprehensive evaluation. Extensive benchmarks across text, image, and video QA tasks show SMILE is highly correlated with human judgments and computationally lightweight, bridging the gap between lexical and semantic evaluation.
>
---
#### [new 038] Falsely Accused: How AI Detectors Misjudge Slightly Polished Arabic Articles
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究AI检测模型在识别阿拉伯语文章时的误判问题。针对人类文本经轻微AI润色后被误判为AI生成的现象，构建了两个数据集，评估14个大模型和商业检测器。结果表明，多数模型性能显著下降，尤其对润色文本误判率高，暴露了现有检测工具的脆弱性。**

- **链接: [https://arxiv.org/pdf/2511.16690v1](https://arxiv.org/pdf/2511.16690v1)**

> **作者:** Saleh Almohaimeed; Saad Almohaimeed; Mousa Jari; Khaled A. Alobaid; Fahad Alotaibi
>
> **备注:** Submitted to Artificial Intelligence Review Journal
>
> **摘要:** Many AI detection models have been developed to counter the presence of articles created by artificial intelligence (AI). However, if a human-authored article is slightly polished by AI, a shift will occur in the borderline decision of these AI detection models, leading them to consider it AI-generated article. This misclassification may result in falsely accusing authors of AI plagiarism and harm the credibility of AI detector models. In English, some efforts were made to meet this challenge, but not in Arabic. In this paper, we generated two datasets. The first dataset contains 800 Arabic articles, half AI-generated and half human-authored. We used it to evaluate 14 Large Language models (LLMs) and commercial AI detectors to assess their ability in distinguishing between human-authored and AI-generated articles. The best 8 models were chosen to act as detectors for our primary concern, which is whether they would consider slightly polished human text as AI-generated. The second dataset, Ar-APT, contains 400 Arabic human-authored articles polished by 10 LLMs using 4 polishing settings, totaling 16400 samples. We use it to evaluate the 8 nominated models and determine whether slight polishing will affect their performance. The results reveal that all AI detectors incorrectly attribute a significant number of articles to AI. The best performing LLM, Claude-4 Sonnet, achieved 83.51%, their performance decreased to 57.63% for articles slightly polished by LLaMA-3. Whereas for the best performing commercial model, originality.AI, that achieves 92% accuracy, dropped to 12% for articles slightly polished by Mistral or Gemma-3.
>
---
#### [new 039] Improving Latent Reasoning in LLMs via Soft Concept Mixing
- **分类: cs.CL**

- **简介: 该论文针对大语言模型（LLM）在抽象推理中因依赖离散标记而表达能力受限的问题，提出软概念混合（SCM）方法。通过将软概念向量融入隐藏状态，并结合强化学习优化，增强模型的潜在推理能力。实验表明，SCM有效提升多任务推理性能，同时保持训练稳定。**

- **链接: [https://arxiv.org/pdf/2511.16885v1](https://arxiv.org/pdf/2511.16885v1)**

> **作者:** Kang Wang; Xiangyu Duan; Tianyi Du
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Unlike human reasoning in abstract conceptual spaces, large language models (LLMs) typically reason by generating discrete tokens, which potentially limit their expressive power. The recent work Soft Thinking has shown that LLMs' latent reasoning via soft concepts is a promising direction, but LLMs are trained on discrete tokens. To reduce this gap between the soft concepts in reasoning and the discrete tokens in training, we propose Soft Concept Mixing (SCM), a soft concept aware training scheme that directly exposes the model to soft representations during training. Specifically, SCM constructs a soft concept vector by forming a probability-weighted average of embeddings. Then, this vector is mixed into the model's hidden states, which embody rich contextual information. Finally, the entire latent reasoning process is optimized with Reinforcement Learning (RL). Experiments on five reasoning benchmarks demonstrate that SCM improves the reasoning performance of LLMs, and simultaneously maintains a stable training dynamic.
>
---
#### [new 040] Learning to Compress: Unlocking the Potential of Large Language Models for Text Representation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型（LLM）在文本表示任务中因因果性设计导致的表征不完整问题，提出以“上下文压缩”为预训练任务，引导模型生成紧凑记忆令牌。通过压缩预训练与对比学习，构建高效文本表示模型LLM2Comp，显著提升表征性能并降低数据需求。**

- **链接: [https://arxiv.org/pdf/2511.17129v1](https://arxiv.org/pdf/2511.17129v1)**

> **作者:** Yeqin Zhang; Yizheng Zhao; Chen Hu; Binxing Jiao; Daxin Jiang; Ruihang Miao; Cam-Tu Nguyen
>
> **备注:** Accepted by AAAI'26
>
> **摘要:** Text representation plays a critical role in tasks like clustering, retrieval, and other downstream applications. With the emergence of large language models (LLMs), there is increasing interest in harnessing their capabilities for this purpose. However, most of the LLMs are inherently causal and optimized for next-token prediction, making them suboptimal for producing holistic representations. To address this, recent studies introduced pretext tasks to adapt LLMs for text representation. Most of these tasks, however, rely on token-level prediction objectives, such as the masked next-token prediction (MNTP) used in LLM2Vec. In this work, we explore the untapped potential of context compression as a pretext task for unsupervised adaptation of LLMs. During compression pre-training, the model learns to generate compact memory tokens, which substitute the whole context for downstream sequence prediction. Experiments demonstrate that a well-designed compression objective can significantly enhance LLM-based text representations, outperforming models trained with token-level pretext tasks. Further improvements through contrastive learning produce a strong representation model (LLM2Comp) that outperforms contemporary LLM-based text encoders on a wide range of tasks while being more sample-efficient, requiring significantly less training data.
>
---
#### [new 041] LangMark: A Multilingual Dataset for Automatic Post-Editing
- **分类: cs.CL**

- **简介: 该论文针对自动后编辑（APE）任务，解决缺乏大规模多语言NMT输出数据集的问题。提出LangMark，一个包含20万+三元组的多语言人工标注数据集，覆盖英译七种语言。基于此，验证了小样本提示的大型语言模型在APE中的有效性，显著提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2511.17153v1](https://arxiv.org/pdf/2511.17153v1)**

> **作者:** Diego Velazquez; Mikaela Grace; Konstantinos Karageorgos; Lawrence Carin; Aaron Schliem; Dimitrios Zaikis; Roger Wechsler
>
> **备注:** 15 pages, 8 figures, ACL 2025
>
> **摘要:** Automatic post-editing (APE) aims to correct errors in machine-translated text, enhancing translation quality, while reducing the need for human intervention. Despite advances in neural machine translation (NMT), the development of effective APE systems has been hindered by the lack of large-scale multilingual datasets specifically tailored to NMT outputs. To address this gap, we present and release LangMark, a new human-annotated multilingual APE dataset for English translation to seven languages: Brazilian Portuguese, French, German, Italian, Japanese, Russian, and Spanish. The dataset has 206,983 triplets, with each triplet consisting of a source segment, its NMT output, and a human post-edited translation. Annotated by expert human linguists, our dataset offers both linguistic diversity and scale. Leveraging this dataset, we empirically show that Large Language Models (LLMs) with few-shot prompting can effectively perform APE, improving upon leading commercial and even proprietary machine translation systems. We believe that this new resource will facilitate the future development and evaluation of APE systems.
>
---
#### [new 042] Do Vision-Language Models Understand Visual Persuasiveness?
- **分类: cs.CL; cs.CV**

- **简介: 该论文探究视觉语言模型（VLMs）对视觉说服力的理解能力。针对“模型是否真正理解视觉如何影响态度与决策”这一问题，构建高共识数据集，提出视觉说服因素分类，并验证不同干预策略。结果表明，模型偏重高召回率，缺乏对低/中层特征的区分力，而语义对齐是关键预测因子，对象锚定的简明推理可显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.17036v1](https://arxiv.org/pdf/2511.17036v1)**

> **作者:** Gyuwon Park
>
> **备注:** 8 pages (except for reference and appendix), 5 figures, 7 tables, to be published in NeurIPS 2025 Workshop: VLM4RWD
>
> **摘要:** Recent advances in vision-language models (VLMs) have enabled impressive multi-modal reasoning and understanding. Yet, whether these models truly grasp visual persuasion-how visual cues shape human attitudes and decisions-remains unclear. To probe this question, we construct a high-consensus dataset for binary persuasiveness judgment and introduce the taxonomy of Visual Persuasive Factors (VPFs), encompassing low-level perceptual, mid-level compositional, and high-level semantic cues. We also explore cognitive steering and knowledge injection strategies for persuasion-relevant reasoning. Empirical analysis across VLMs reveals a recall-oriented bias-models over-predict high persuasiveness-and weak discriminative power for low/mid-level features. In contrast, high-level semantic alignment between message and object presence emerges as the strongest predictor of human judgment. Among intervention strategies, simple instruction or unguided reasoning scaffolds yield marginal or negative effects, whereas concise, object-grounded rationales significantly improve precision and F1 scores. These results indicate that VLMs core limitation lies not in recognizing persuasive objects but in linking them to communicative intent.
>
---
#### [new 043] Large Language Models for Sentiment Analysis to Detect Social Challenges: A Use Case with South African Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言情感分析任务，旨在利用大语言模型（LLMs）检测南非语境下社会媒体中的社会挑战。针对英语、塞佩迪语和茨瓦纳语的热点话题，评估GPT-3.5、GPT-4等模型的零样本情感分析性能，并通过融合多模型结果，实现低于1%错误率的高精度分类，为政府提供可靠决策支持。**

- **链接: [https://arxiv.org/pdf/2511.17301v1](https://arxiv.org/pdf/2511.17301v1)**

> **作者:** Koena Ronny Mabokela; Tim Schlippe; Matthias Wölfel
>
> **备注:** Published in the Proceedings of The Southern African Conference on AI Research (SACAIR 2024), Bloemfontein, South Africa, 2-6 December 2024. ISBN: 978-0-7961-6069-0
>
> **摘要:** Sentiment analysis can aid in understanding people's opinions and emotions on social issues. In multilingual communities sentiment analysis systems can be used to quickly identify social challenges in social media posts, enabling government departments to detect and address these issues more precisely and effectively. Recently, large-language models (LLMs) have become available to the wide public and initial analyses have shown that they exhibit magnificent zero-shot sentiment analysis abilities in English. However, there is no work that has investigated to leverage LLMs for sentiment analysis on social media posts in South African languages and detect social challenges. Consequently, in this work, we analyse the zero-shot performance of the state-of-the-art LLMs GPT-3.5, GPT-4, LlaMa 2, PaLM 2, and Dolly 2 to investigate the sentiment polarities of the 10 most emerging topics in English, Sepedi and Setswana social media posts that fall within the jurisdictional areas of 10 South African government departments. Our results demonstrate that there are big differences between the various LLMs, topics, and languages. In addition, we show that a fusion of the outcomes of different LLMs provides large gains in sentiment classification performance with sentiment classification errors below 1%. Consequently, it is now feasible to provide systems that generate reliable information about sentiment analysis to detect social challenges and draw conclusions about possible needs for actions on specific topics and within different language groups.
>
---
#### [new 044] Parrot: Persuasion and Agreement Robustness Rating of Output Truth -- A Sycophancy Robustness Benchmark for LLMs
- **分类: cs.CL; cs.AI; cs.CE; cs.LG**

- **简介: 该论文提出PARROT框架，评估大模型在权威与说服压力下的鲁棒性，解决“谄媚”现象导致的错误响应问题。通过双盲实验、置信度追踪与行为分类，发现小模型易受误导，准确率与自信度双重下降，强调抗压力应作为模型安全部署的核心目标。**

- **链接: [https://arxiv.org/pdf/2511.17220v1](https://arxiv.org/pdf/2511.17220v1)**

> **作者:** Yusuf Çelebi; Mahmoud El Hussieni; Özay Ezerceli
>
> **摘要:** This study presents PARROT (Persuasion and Agreement Robustness Rating of Output Truth), a robustness focused framework designed to measure the degradation in accuracy that occurs under social pressure exerted on users through authority and persuasion in large language models (LLMs) the phenomenon of sycophancy (excessive conformity). PARROT (i) isolates causal effects by comparing the neutral version of the same question with an authoritatively false version using a double-blind evaluation, (ii) quantifies confidence shifts toward the correct and imposed false responses using log-likelihood-based calibration tracking, and (iii) systematically classifies failure modes (e.g., robust correct, sycophantic agreement, reinforced error, stubborn error, self-correction, etc.) using an eight-state behavioral taxonomy. We evaluated 22 models using 1,302 MMLU-style multiple-choice questions across 13 domains and domain-specific authority templates. Findings show marked heterogeneity: advanced models (e.g., GPT-5, GPT-4.1, Claude Sonnet 4.5) exhibit low "follow rates" ($\leq 11\%$, GPT-5: 4\%) and minimal accuracy loss, while older/smaller models show severe epistemic collapse (GPT-4: 80\%, Qwen 2.5-1.5B: 94\%). The danger is not limited to response changes; weak models reduce confidence in the correct response while increasing confidence in the imposed incorrect response. While international law and global knowledge at the domain level exhibit high fragility, elementary mathematics is relatively resilient. Consequently, we argue that the goal of "resistance to overfitting pressure" should be addressed as a primary objective alongside accuracy, harm avoidance, and privacy for safe deployment in the real world.
>
---
#### [new 045] Estonian WinoGrande Dataset: Comparative Analysis of LLM Performance on Human and Machine Translation
- **分类: cs.CL**

- **简介: 该论文聚焦于多语言常识推理评估任务，针对Estonian语境下WinoGrande数据集的翻译与适配问题。研究通过专业译者完成高质量人工翻译，并探索将人工翻译经验融入提示工程以提升机器翻译质量。结果表明，人工翻译数据上模型表现优于机器翻译，且提示工程改善有限，强调语言专家参与对可靠评估的重要性。**

- **链接: [https://arxiv.org/pdf/2511.17290v1](https://arxiv.org/pdf/2511.17290v1)**

> **作者:** Marii Ojastu; Hele-Andra Kuulmets; Aleksei Dorkin; Marika Borovikova; Dage Särg; Kairit Sirts
>
> **备注:** Preprint
>
> **摘要:** In this paper, we present a localized and culturally adapted Estonian translation of the test set from the widely used commonsense reasoning benchmark, WinoGrande. We detail the translation and adaptation process carried out by translation specialists and evaluate the performance of both proprietary and open source models on the human translated benchmark. Additionally, we explore the feasibility of achieving high-quality machine translation by incorporating insights from the manual translation process into the design of a detailed prompt. This prompt is specifically tailored to address both the linguistic characteristics of Estonian and the unique translation challenges posed by the WinoGrande dataset. Our findings show that model performance on the human translated Estonian dataset is slightly lower than on the original English test set, while performance on machine-translated data is notably worse. Additionally, our experiments indicate that prompt engineering offers limited improvement in translation quality or model accuracy, and highlight the importance of involving language specialists in dataset translation and adaptation to ensure reliable and interpretable evaluations of language competency and reasoning in large language models.
>
---
#### [new 046] Hierarchical Retrieval with Out-Of-Vocabulary Queries: A Case Study on SNOMED CT
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究生物医学本体SNOMED CT中的层级概念检索任务，针对查询词不在词汇表（OOV）导致的检索难题，提出基于语言模型的本体嵌入方法。通过构建标注数据集评估，该方法优于SBERT和传统匹配方法，具有良好的泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.16698v1](https://arxiv.org/pdf/2511.16698v1)**

> **作者:** Jonathon Dilworth; Hui Yang; Jiaoyan Chen; Yongsheng Gao
>
> **备注:** 5 pages, 3 figures, 3 tables, submission to The Web Conference 2026 (WWW'26), Dubai, UAE
>
> **摘要:** SNOMED CT is a biomedical ontology with a hierarchical representation of large-scale concepts. Knowledge retrieval in SNOMED CT is critical for its application, but often proves challenging due to language ambiguity, synonyms, polysemies and so on. This problem is exacerbated when the queries are out-of-vocabulary (OOV), i.e., having no equivalent matchings in the ontology. In this work, we focus on the problem of hierarchical concept retrieval from SNOMED CT with OOV queries, and propose an approach based on language model-based ontology embeddings. For evaluation, we construct OOV queries annotated against SNOMED CT concepts, testing the retrieval of the most direct subsumers and their less relevant ancestors. We find that our method outperforms the baselines including SBERT and two lexical matching methods. While evaluated against SNOMED CT, the approach is generalisable and can be extended to other ontologies. We release code, tools, and evaluation datasets at https://github.com/jonathondilworth/HR-OOV.
>
---
#### [new 047] Humanlike Multi-user Agent (HUMA): Designing a Deceptively Human AI Facilitator for Group Chats
- **分类: cs.CL**

- **简介: 该论文提出HUMA，一种类人多用户对话代理，旨在解决现有AI在异步群聊中缺乏自然交互的问题。通过事件驱动架构与三组件设计，模拟人类响应时间与行为策略，在实验中使用户难以区分AI与真人社区管理员，证明其能实现高拟真度的群组互动。**

- **链接: [https://arxiv.org/pdf/2511.17315v1](https://arxiv.org/pdf/2511.17315v1)**

> **作者:** Mateusz Jacniacki; Martí Carmona Serrat
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Conversational agents built on large language models (LLMs) are becoming increasingly prevalent, yet most systems are designed for one-on-one, turn-based exchanges rather than natural, asynchronous group chats. As AI assistants become widespread throughout digital platforms, from virtual assistants to customer service, developing natural and humanlike interaction patterns seems crucial for maintaining user trust and engagement. We present the Humanlike Multi-user Agent (HUMA), an LLM-based facilitator that participates in multi-party conversations using human-like strategies and timing. HUMA extends prior multi-user chatbot work with an event-driven architecture that handles messages, replies, reactions and introduces realistic response-time simulation. HUMA comprises three components-Router, Action Agent, and Reflection-which together adapt LLMs to group conversation dynamics. We evaluate HUMA in a controlled study with 97 participants in four-person role-play chats, comparing AI and human community managers (CMs). Participants classified CMs as human at near-chance rates in both conditions, indicating they could not reliably distinguish HUMA agents from humans. Subjective experience was comparable across conditions: community-manager effectiveness, social presence, and engagement/satisfaction differed only modestly with small effect sizes. Our results suggest that, in natural group chat settings, an AI facilitator can match human quality while remaining difficult to identify as nonhuman.
>
---
#### [new 048] Planning with Sketch-Guided Verification for Physics-Aware Video Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文面向物理感知视频生成任务，针对现有方法在运动规划中轨迹简单或计算成本高的问题，提出SketchVerify框架。通过测试时采样与验证循环，利用轻量级视频草图快速评估候选轨迹的语义一致性与物理合理性，实现高效高质量运动规划，显著提升视频生成的物理真实性和长期一致性。**

- **链接: [https://arxiv.org/pdf/2511.17450v1](https://arxiv.org/pdf/2511.17450v1)**

> **作者:** Yidong Huang; Zun Wang; Han Lin; Dong-Ki Kim; Shayegan Omidshafiei; Jaehong Yoon; Yue Zhang; Mohit Bansal
>
> **备注:** website: https://sketchverify.github.io/
>
> **摘要:** Recent video generation approaches increasingly rely on planning intermediate control signals such as object trajectories to improve temporal coherence and motion fidelity. However, these methods mostly employ single-shot plans that are typically limited to simple motions, or iterative refinement which requires multiple calls to the video generator, incuring high computational cost. To overcome these limitations, we propose SketchVerify, a training-free, sketch-verification-based planning framework that improves motion planning quality with more dynamically coherent trajectories (i.e., physically plausible and instruction-consistent motions) prior to full video generation by introducing a test-time sampling and verification loop. Given a prompt and a reference image, our method predicts multiple candidate motion plans and ranks them using a vision-language verifier that jointly evaluates semantic alignment with the instruction and physical plausibility. To efficiently score candidate motion plans, we render each trajectory as a lightweight video sketch by compositing objects over a static background, which bypasses the need for expensive, repeated diffusion-based synthesis while achieving comparable performance. We iteratively refine the motion plan until a satisfactory one is identified, which is then passed to the trajectory-conditioned generator for final synthesis. Experiments on WorldModelBench and PhyWorldBench demonstrate that our method significantly improves motion quality, physical realism, and long-term consistency compared to competitive baselines while being substantially more efficient. Our ablation study further shows that scaling up the number of trajectory candidates consistently enhances overall performance.
>
---
#### [new 049] RubiSCoT: A Framework for AI-Supported Academic Assessment
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出RubiSCoT框架，旨在解决学术论文评估耗时、主观性强的问题。通过AI技术实现从选题到终稿的全流程自动化评估，涵盖多维度分析与透明评分，提升评估一致性与效率。**

- **链接: [https://arxiv.org/pdf/2510.17309v1](https://arxiv.org/pdf/2510.17309v1)**

> **作者:** Thorsten Fröhlich; Tim Schlippe
>
> **摘要:** The evaluation of academic theses is a cornerstone of higher education, ensuring rigor and integrity. Traditional methods, though effective, are time-consuming and subject to evaluator variability. This paper presents RubiSCoT, an AI-supported framework designed to enhance thesis evaluation from proposal to final submission. Using advanced natural language processing techniques, including large language models, retrieval-augmented generation, and structured chain-of-thought prompting, RubiSCoT offers a consistent, scalable solution. The framework includes preliminary assessments, multidimensional assessments, content extraction, rubric-based scoring, and detailed reporting. We present the design and implementation of RubiSCoT, discussing its potential to optimize academic assessment processes through consistent, scalable, and transparent evaluation.
>
---
#### [new 050] An Efficient Computational Framework for Discrete Fuzzy Numbers Based on Total Orders
- **分类: cs.LO; cs.CL; cs.DM; cs.LG**

- **简介: 该论文针对离散模糊数的全序关系计算问题，提出高效算法求解pos函数及其逆，实现对模糊数的精确排序。通过利用组合结构，将复杂度降至O(n²m log n)，显著降低计算开销，提升模糊逻辑运算效率，支持大规模应用。**

- **链接: [https://arxiv.org/pdf/2511.17080v1](https://arxiv.org/pdf/2511.17080v1)**

> **作者:** Arnau Mir; Alejandro Mus; Juan Vicente Riera
>
> **备注:** 19 pages, 2 figures. Submitted to Computational and Applied Mathematics (Springer)
>
> **摘要:** Discrete fuzzy numbers, and in particular those defined over a finite chain $L_n = \{0, \ldots, n\}$, have been effectively employed to represent linguistic information within the framework of fuzzy systems. Research on total (admissible) orderings of such types of fuzzy subsets, and specifically those belonging to the set $\mathcal{D}_1^{L_n\rightarrow Y_m}$ consisting of discrete fuzzy numbers $A$ whose support is a closed subinterval of the finite chain $L_n = \{0, 1, \ldots, n\}$ and whose membership values $A(x)$, for $x \in L_n$, belong to the set $Y_m = \{ 0 = y_1 < y_2 < \cdots < y_{m-1} < y_m = 1 \}$, has facilitated the development of new methods for constructing logical connectives, based on a bijective function, called $\textit{pos function}$, that determines the position of each $A \in \mathcal{D}_1^{L_n\rightarrow Y_m}$. For this reason, in this work we revisit the problem by introducing algorithms that exploit the combinatorial structure of total (admissible) orders to compute the $\textit{pos}$ function and its inverse with exactness. The proposed approach achieves a complexity of $\mathcal{O}(n^{2} m \log n)$, which is quadratic in the size of the underlying chain ($n$) and linear in the number of membership levels ($m$). The key point is that the dominant factor is $m$, ensuring scalability with respect to the granularity of membership values. The results demonstrate that this formulation substantially reduces computational cost and enables the efficient implementation of algebraic operations -- such as aggregation and implication -- on the set of discrete fuzzy numbers.
>
---
#### [new 051] Robot Confirmation Generation and Action Planning Using Long-context Q-Former Integrated with Multimodal LLM
- **分类: cs.RO; cs.CL; cs.CV; cs.SD; eess.AS**

- **简介: 该论文聚焦人机协作中的动作确认与规划任务，针对现有方法忽略长视频上下文依赖、文本信息抽象过度的问题，提出融合左右上下文的长程Q-former与文本条件化机制，通过VideoLLaMA3提升多模态理解能力，显著改善动作确认与规划准确率。**

- **链接: [https://arxiv.org/pdf/2511.17335v1](https://arxiv.org/pdf/2511.17335v1)**

> **作者:** Chiori Hori; Yoshiki Masuyama; Siddarth Jain; Radu Corcodel; Devesh Jha; Diego Romeres; Jonathan Le Roux
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Human-robot collaboration towards a shared goal requires robots to understand human action and interaction with the surrounding environment. This paper focuses on human-robot interaction (HRI) based on human-robot dialogue that relies on the robot action confirmation and action step generation using multimodal scene understanding. The state-of-the-art approach uses multimodal transformers to generate robot action steps aligned with robot action confirmation from a single clip showing a task composed of multiple micro steps. Although actions towards a long-horizon task depend on each other throughout an entire video, the current approaches mainly focus on clip-level processing and do not leverage long-context information. This paper proposes a long-context Q-former incorporating left and right context dependency in full videos. Furthermore, this paper proposes a text-conditioning approach to feed text embeddings directly into the LLM decoder to mitigate the high abstraction of the information in text by Q-former. Experiments with the YouCook2 corpus show that the accuracy of confirmation generation is a major factor in the performance of action planning. Furthermore, we demonstrate that the long-context Q-former improves the confirmation and action planning by integrating VideoLLaMA3.
>
---
#### [new 052] OmniScientist: Toward a Co-evolving Ecosystem of Human and AI Scientists
- **分类: cs.CY; cs.CE; cs.CL**

- **简介: 该论文提出OmniScientist框架，旨在构建人与AI科学家协同演化的科研生态系统。针对现有AI科学家忽视科学社会性与协作机制的问题，论文设计了知识网络、协作协议与评估平台，实现从文献到发表的全流程自动化，并支持深度人机协同，推动可持续科研创新。**

- **链接: [https://arxiv.org/pdf/2511.16931v1](https://arxiv.org/pdf/2511.16931v1)**

> **作者:** Chenyang Shao; Dehao Huang; Yu Li; Keyu Zhao; Weiquan Lin; Yining Zhang; Qingbin Zeng; Zhiyu Chen; Tianxing Li; Yifei Huang; Taozhong Wu; Xinyang Liu; Ruotong Zhao; Mengsheng Zhao; Xuhua Zhang; Yue Wang; Yuanyi Zhen; Fengli Xu; Yong Li; Tie-Yan Liu
>
> **摘要:** With the rapid development of Large Language Models (LLMs), AI agents have demonstrated increasing proficiency in scientific tasks, ranging from hypothesis generation and experimental design to manuscript writing. Such agent systems are commonly referred to as "AI Scientists." However, existing AI Scientists predominantly formulate scientific discovery as a standalone search or optimization problem, overlooking the fact that scientific research is inherently a social and collaborative endeavor. Real-world science relies on a complex scientific infrastructure composed of collaborative mechanisms, contribution attribution, peer review, and structured scientific knowledge networks. Due to the lack of modeling for these critical dimensions, current systems struggle to establish a genuine research ecosystem or interact deeply with the human scientific community. To bridge this gap, we introduce OmniScientist, a framework that explicitly encodes the underlying mechanisms of human research into the AI scientific workflow. OmniScientist not only achieves end-to-end automation across data foundation, literature review, research ideation, experiment automation, scientific writing, and peer review, but also provides comprehensive infrastructural support by simulating the human scientific system, comprising: (1) a structured knowledge system built upon citation networks and conceptual correlations; (2) a collaborative research protocol (OSP), which enables seamless multi-agent collaboration and human researcher participation; and (3) an open evaluation platform (ScienceArena) based on blind pairwise user voting and Elo rankings. This infrastructure empowers agents to not only comprehend and leverage human knowledge systems but also to collaborate and co-evolve, fostering a sustainable and scalable innovation ecosystem.
>
---
#### [new 053] Vision Language Models are Confused Tourists
- **分类: cs.CV; cs.CL**

- **简介: 该论文聚焦于视觉语言模型（VLMs）在多元文化输入下的稳定性问题，旨在解决现有评估忽视多文化线索共存的缺陷。作者提出ConfusedTourist评测套件，通过图像叠加等扰动测试模型鲁棒性，发现模型易受干扰导致性能下降，根源在于注意力机制被无关文化线索误导。研究揭示了当前VLMs在跨文化理解中的系统性脆弱性，呼吁提升模型的文化适应能力。**

- **链接: [https://arxiv.org/pdf/2511.17004v1](https://arxiv.org/pdf/2511.17004v1)**

> **作者:** Patrick Amadeus Irawan; Ikhlasul Akmal Hanif; Muhammad Dehan Al Kautsar; Genta Indra Winata; Fajri Koto; Alham Fikri Aji
>
> **摘要:** Although the cultural dimension has been one of the key aspects in evaluating Vision-Language Models (VLMs), their ability to remain stable across diverse cultural inputs remains largely untested, despite being crucial to support diversity and multicultural societies. Existing evaluations often rely on benchmarks featuring only a singular cultural concept per image, overlooking scenarios where multiple, potentially unrelated cultural cues coexist. To address this gap, we introduce ConfusedTourist, a novel cultural adversarial robustness suite designed to assess VLMs' stability against perturbed geographical cues. Our experiments reveal a critical vulnerability, where accuracy drops heavily under simple image-stacking perturbations and even worsens with its image-generation-based variant. Interpretability analyses further show that these failures stem from systematic attention shifts toward distracting cues, diverting the model from its intended focus. These findings highlight a critical challenge: visual cultural concept mixing can substantially impair even state-of-the-art VLMs, underscoring the urgent need for more culturally robust multimodal understanding.
>
---
#### [new 054] Cognitive BASIC: An In-Model Interpreted Reasoning Language for LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Cognitive BASIC，一种基于BASIC风格的可解释推理语言及模型内解释器，用于结构化大模型的多步推理过程。旨在提升推理透明性，解决黑箱决策问题。通过编号指令与自然语言解释器实现知识提取、冲突检测与修正，验证了三类LLM的执行能力。**

- **链接: [https://arxiv.org/pdf/2511.16837v1](https://arxiv.org/pdf/2511.16837v1)**

> **作者:** Oliver Kramer
>
> **备注:** 6 pages, Submitted to ESANN 2026
>
> **摘要:** Cognitive BASIC is a minimal, BASIC-style prompting language and in-model interpreter that structures large language model (LLM) reasoning into explicit, stepwise execution traces. Inspired by the simplicity of retro BASIC, we repurpose numbered lines and simple commands as an interpretable cognitive control layer. Modern LLMs can reliably simulate such short programs, enabling transparent multi-step reasoning inside the model. A natural-language interpreter file specifies command semantics, memory updates, and logging behavior. Our mental-model interpreter extracts declarative and procedural knowledge, detects contradictions, and produces resolutions when necessary. A comparison across three LLMs on a benchmark of knowledge extraction, conflict detection, and reasoning tasks shows that all models can execute Cognitive BASIC programs, with overall strong but not uniform performance.
>
---
#### [new 055] Geometric-Disentangelment Unlearning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究机器模型遗忘任务，旨在消除训练数据子集的影响同时保护保留数据的性能。针对现有方法在遗忘与保留间权衡不佳的问题，提出几何解耦遗忘（GU）：通过将遗忘更新分解为与保留梯度正交的分量，仅执行不影响保留性能的部分，实现理论保证下的无损遗忘。**

- **链接: [https://arxiv.org/pdf/2511.17100v1](https://arxiv.org/pdf/2511.17100v1)**

> **作者:** Duo Zhou; Yuji Zhang; Tianxin Wei; Ruizhong Qiu; Ke Yang; Xiao Lin; Cheng Qian; Jingrui He; Hanghang Tong; Heng Ji; Huan Zhang
>
> **备注:** 27 Pages
>
> **摘要:** Machine unlearning, the removal of a training subset's influence from a deployed model, is critical for privacy preservation and model reliability, yet gradient ascent on forget samples often harms retained knowledge. Existing approaches face a persistent tradeoff between effective forgetting and preservation on the retain set. While previous methods provide useful heuristics, they often lack a formal analysis on how exactly forgetting updates harm retained knowledge, and whether the side effects can be removed with theoretical guarantees. To explore a theoretically sound and simple solution, we start from the first principle on how performance on the retain set is actually affected: a first-order analysis of the local change of the retain loss under small parameter updates during model training. We start from a crisp equivalence: the retain loss is unchanged to first order iff the update direction is orthogonal to the subspace spanned by retain gradients ("retain-invariant"). This identifies the entangled component as the tangential part of forget update within the retain-gradient subspace, and characterizes disentanglement as orthogonality. Guided by this, we propose the Geometric-disentanglement Unlearning (GU) that decomposes any candidate forget gradient update into tangential and normal components to retain space and executes only the normal component. Under a standard trust-region budget, the projected direction aligned with the raw forget gradient is optimal among all first-order retain-invariant moves, and we also derive the optimal projected direction for joint forget-retain updating objectives. Our method is plug-and-play and can be attached to existing gradient-based unlearning procedures to mitigate side effects. GU achieves consistent improvement on various methods across three benchmarks TOFU, MUSE, and WMDP.
>
---
#### [new 056] MusicAIR: A Multimodal AI Music Generation Framework Powered by an Algorithm-Driven Core
- **分类: cs.SD; cs.AI; cs.CL; cs.MM**

- **简介: 该论文提出MusicAIR框架，解决生成式AI音乐因依赖大数据带来的版权与成本问题。基于算法驱动的符号化音乐核心，实现从歌词、文本或图像自动生成符合音乐理论的乐谱，支持多模态输入。实验表明其生成作品在调性一致性上优于人类作曲家，具备教育与创作辅助价值。**

- **链接: [https://arxiv.org/pdf/2511.17323v1](https://arxiv.org/pdf/2511.17323v1)**

> **作者:** Callie C. Liao; Duoduo Liao; Ellie L. Zhang
>
> **备注:** Accepted by IEEE Big Data 2025
>
> **摘要:** Recent advances in generative AI have made music generation a prominent research focus. However, many neural-based models rely on large datasets, raising concerns about copyright infringement and high-performance costs. In contrast, we propose MusicAIR, an innovative multimodal AI music generation framework powered by a novel algorithm-driven symbolic music core, effectively mitigating copyright infringement risks. The music core algorithms connect critical lyrical and rhythmic information to automatically derive musical features, creating a complete, coherent melodic score solely from the lyrics. The MusicAIR framework facilitates music generation from lyrics, text, and images. The generated score adheres to established principles of music theory, lyrical structure, and rhythmic conventions. We developed Generate AI Music (GenAIM), a web tool using MusicAIR for lyric-to-song, text-to-music, and image-to-music generation. In our experiments, we evaluated AI-generated music scores produced by the system using both standard music metrics and innovative analysis that compares these compositions with original works. The system achieves an average key confidence of 85%, outperforming human composers at 79%, and aligns closely with established music theory standards, demonstrating its ability to generate diverse, human-like compositions. As a co-pilot tool, GenAIM can serve as a reliable music composition assistant and a possible educational composition tutor while simultaneously lowering the entry barrier for all aspiring musicians, which is innovative and significantly contributes to AI for music generation.
>
---
#### [new 057] Fantastic Bugs and Where to Find Them in AI Benchmarks
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对AI基准测试中无效题目影响评估可靠性的问题，提出基于响应模式统计分析的系统性修订框架。通过检验题目统计特征是否偏离预期范围，识别潜在问题题项，并引入LLM-judge初筛，提升专家评审效率，实现高效、可扩展的基准测试质量控制。**

- **链接: [https://arxiv.org/pdf/2511.16842v1](https://arxiv.org/pdf/2511.16842v1)**

> **作者:** Sang Truong; Yuheng Tu; Michael Hardy; Anka Reuel; Zeyu Tang; Jirayu Burapacheep; Jonathan Perera; Chibuike Uwakwe; Ben Domingue; Nick Haber; Sanmi Koyejo
>
> **摘要:** Benchmarks are pivotal in driving AI progress, and invalid benchmark questions frequently undermine their reliability. Manually identifying and correcting errors among thousands of benchmark questions is not only infeasible but also a critical bottleneck for reliable evaluation. In this work, we introduce a framework for systematic benchmark revision that leverages statistical analysis of response patterns to flag potentially invalid questions for further expert review. Our approach builds on a core assumption commonly used in AI evaluations that the mean score sufficiently summarizes model performance. This implies a unidimensional latent construct underlying the measurement experiment, yielding expected ranges for various statistics for each item. When empirically estimated values for these statistics fall outside the expected range for an item, the item is more likely to be problematic. Across nine widely used benchmarks, our method guides expert review to identify problematic questions with up to 84\% precision. In addition, we introduce an LLM-judge first pass to review questions, further reducing human effort. Together, these components provide an efficient and scalable framework for systematic benchmark revision.
>
---
#### [new 058] The Shifting Landscape of Vaccine Discourse: Insights From a Decade of Pre- to Post-COVID-19 Vaccine Posts on Social Media
- **分类: cs.SI; cs.CL**

- **简介: 该论文研究社交媒体上英语疫苗话语的演变，聚焦2013–2022年X平台（原Twitter）帖子。通过构建1870万条筛选后数据集，结合社会认知与刻板印象模型，分析疫情前后情绪与语言变化。发现疫情初期负面情绪减少、信任与惊讶词增多，后期负面词汇回升，反映疫苗犹豫加剧。任务为社交媒体话语分析，旨在揭示疫苗舆论动态及其心理机制。**

- **链接: [https://arxiv.org/pdf/2511.16832v1](https://arxiv.org/pdf/2511.16832v1)**

> **作者:** Nikesh Gyawali; Doina Caragea; Cornelia Caragea; Saif M. Mohammad
>
> **摘要:** In this work, we study English-language vaccine discourse in social media posts, specifically posts on X (formerly Twitter), in seven years before the COVID-19 outbreak (2013 to 2019) and three years after the outbreak was first reported (2020 to 2022). Drawing on theories from social cognition and the stereotype content model in Social Psychology, we analyze how English speakers talk about vaccines on social media to understand the evolving narrative around vaccines in social media posts. To do that, we first introduce a novel dataset comprising 18.7 million curated posts on vaccine discourse from 2013 to 2022. This extensive collection-filtered down from an initial 129 million posts through rigorous preprocessing-captures both pre-COVID and COVID-19 periods, offering valuable insights into the evolution of English-speaking X users' perceptions related to vaccines. Our analysis shows that the COVID-19 pandemic led to complex shifts in X users' sentiment and discourse around vaccines. We observe that negative emotion word usage decreased during the pandemic, with notable rises in usage of surprise, and trust related emotion words. Furthermore, vaccine-related language tended to use more warmth-focused words associated with trustworthiness, along with positive, competence-focused words during the early days of the pandemic, with a marked rise in negative word usage towards the end of the pandemic, possibly reflecting a growing vaccine hesitancy and skepticism.
>
---
#### [new 059] Cross-cultural value alignment frameworks for responsible AI governance: Evidence from China-West comparative analysis
- **分类: cs.CY; cs.CL**

- **简介: 该论文聚焦于负责任AI治理中的跨文化价值对齐问题，旨在解决大语言模型在不同文化背景下价值观不一致的挑战。通过构建多层审计平台，采用四种方法对比分析中西起源模型，发现模型普遍存在价值不稳定、代际代表性不足及规模与对齐质量非线性关系等问题，揭示了架构与训练策略对跨文化适应性的关键影响。**

- **链接: [https://arxiv.org/pdf/2511.17256v1](https://arxiv.org/pdf/2511.17256v1)**

> **作者:** Haijiang Liu; Jinguang Gu; Xun Wu; Daniel Hershcovich; Qiaoling Xiao
>
> **备注:** Presented on Academic Conference "Technology for Good: Driving Social Impact" (2025)
>
> **摘要:** As Large Language Models (LLMs) increasingly influence high-stakes decision-making across global contexts, ensuring their alignment with diverse cultural values has become a critical governance challenge. This study presents a Multi-Layered Auditing Platform for Responsible AI that systematically evaluates cross-cultural value alignment in China-origin and Western-origin LLMs through four integrated methodologies: Ethical Dilemma Corpus for assessing temporal stability, Diversity-Enhanced Framework (DEF) for quantifying cultural fidelity, First-Token Probability Alignment for distributional accuracy, and Multi-stAge Reasoning frameworK (MARK) for interpretable decision-making. Our comparative analysis of 20+ leading models, such as Qwen, GPT-4o, Claude, LLaMA, and DeepSeek, reveals universal challenges-fundamental instability in value systems, systematic under-representation of younger demographics, and non-linear relationships between model scale and alignment quality-alongside divergent regional development trajectories. While China-origin models increasingly emphasize multilingual data integration for context-specific optimization, Western models demonstrate greater architectural experimentation but persistent U.S.-centric biases. Neither paradigm achieves robust cross-cultural generalization. We establish that Mistral-series architectures significantly outperform LLaMA3-series in cross-cultural alignment, and that Full-Parameter Fine-Tuning on diverse datasets surpasses Reinforcement Learning from Human Feedback in preserving cultural variation...
>
---
## 更新

#### [replaced 001] ReviewGuard: Enhancing Deficient Peer Review Detection via LLM-Driven Data Augmentation
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2510.16549v2](https://arxiv.org/pdf/2510.16549v2)**

> **作者:** Haoxuan Zhang; Ruochi Li; Sarthak Shrestha; Shree Harshini Mamidala; Revanth Putta; Arka Krishan Aggarwal; Ting Xiao; Junhua Ding; Haihua Chen
>
> **备注:** Accepted as a full paper at the 2025 ACM/IEEE Joint Conference on Digital Libraries (JCDL 2025)
>
> **摘要:** Peer review serves as the gatekeeper of science, yet the surge in submissions and widespread adoption of large language models (LLMs) in scholarly evaluation present unprecedented challenges. While recent work has focused on using LLMs to improve review efficiency, unchecked deficient reviews from both human experts and AI systems threaten to systematically undermine academic integrity. To address this issue, we introduce ReviewGuard, an automated system for detecting and categorizing deficient reviews through a four-stage LLM-driven framework: data collection from ICLR and NeurIPS on OpenReview, GPT-4.1 annotation with human validation, synthetic data augmentation yielding 6,634 papers with 24,657 real and 46,438 synthetic reviews, and fine-tuning of encoder-based models and open-source LLMs. Feature analysis reveals that deficient reviews exhibit lower rating scores, higher self-reported confidence, reduced structural complexity, and more negative sentiment than sufficient reviews. AI-generated text detection shows dramatic increases in AI-authored reviews since ChatGPT's emergence. Mixed training with synthetic and real data substantially improves detection performance - for example, Qwen 3-8B achieves recall of 0.6653 and F1 of 0.7073, up from 0.5499 and 0.5606 respectively. This study presents the first LLM-driven system for detecting deficient peer reviews, providing evidence to inform AI governance in peer review. Code, prompts, and data are available at https://github.com/haoxuan-unt2024/ReviewGuard
>
---
#### [replaced 002] Bridging the Semantic Gap: Contrastive Rewards for Multilingual Text-to-SQL with GRPO
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.13827v2](https://arxiv.org/pdf/2510.13827v2)**

> **作者:** Ashish Kattamuri; Ishita Prasad; Meetu Malhotra; Arpita Vats; Rahul Raja; Albert Lie
>
> **备注:** 20th International Workshop on Semantic and Social Media Adaptation & Personalization
>
> **摘要:** Current Text-to-SQL methods are evaluated and only focused on executable queries, overlooking the semantic alignment challenge -- both in terms of the semantic meaning of the query and the correctness of the execution results. Even execution accuracy itself shows significant drops when moving from English to other languages, with an average decline of 6 percentage points across non-English languages. We address these challenges by presenting a new framework that combines Group Relative Policy Optimization (GRPO) within a multilingual contrastive reward signal to enhance both task efficiency and semantic accuracy in Text-to-SQL systems in cross-lingual scenarios. Our method teaches models to obtain better correspondence between SQL generation and user intent by combining a reward signal based on semantic similarity. On the seven-language MultiSpider dataset, fine-tuning the LLaMA-3-3B model with GRPO improved the execution accuracy up to 87.4 percent (+26 pp over zero-shot) and semantic accuracy up to 52.29 percent (+32.86 pp). Adding our contrastive reward signal in the GRPO framework further improved the average semantic accuracy to 59.14 percent (+6.85 pp, up to +10 pp for Vietnamese). Our experiments showcase that a smaller, parameter-efficient 3B LLaMA model fine-tuned with our contrastive reward signal outperforms a much larger zero-shot 8B LLaMA model, with an uplift of 7.43 pp in execution accuracy (from 81.43 percent on the 8B model to 88.86 percent on the 3B model), and nearly matches its semantic accuracy (59.14 percent vs. 68.57 percent) -- all using just 3,000 reinforcement learning training examples. These results demonstrate how we can improve the performance of Text-to-SQL systems with contrastive rewards for directed semantic alignment, without requiring large-scale training datasets.
>
---
#### [replaced 003] From Hypothesis to Publication: A Comprehensive Survey of AI-Driven Research Support Systems
- **分类: cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2503.01424v4](https://arxiv.org/pdf/2503.01424v4)**

> **作者:** Zekun Zhou; Xiaocheng Feng; Lei Huang; Xiachong Feng; Ziyun Song; Ruihan Chen; Liang Zhao; Weitao Ma; Yuxuan Gu; Baoxin Wang; Dayong Wu; Guoping Hu; Ting Liu; Bing Qin
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Research is a fundamental process driving the advancement of human civilization, yet it demands substantial time and effort from researchers. In recent years, the rapid development of artificial intelligence (AI) technologies has inspired researchers to explore how AI can accelerate and enhance research. To monitor relevant advancements, this paper presents a systematic review of the progress in this domain. Specifically, we organize the relevant studies into three main categories: hypothesis formulation, hypothesis validation, and manuscript publication. Hypothesis formulation involves knowledge synthesis and hypothesis generation. Hypothesis validation includes the verification of scientific claims, theorem proving, and experiment validation. Manuscript publication encompasses manuscript writing and the peer review process. Furthermore, we identify and discuss the current challenges faced in these areas, as well as potential future directions for research. Finally, we also offer a comprehensive overview of existing benchmarks and tools across various domains that support the integration of AI into the research process. We hope this paper serves as an introduction for beginners and fosters future research. Resources have been made publicly available at https://github.com/zkzhou126/AI-for-Research.
>
---
#### [replaced 004] Concise Reasoning via Reinforcement Learning
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2504.05185v3](https://arxiv.org/pdf/2504.05185v3)**

> **作者:** Mehdi Fatemi; Banafsheh Rafiee; Mingjie Tang; Kartik Talamadupula
>
> **摘要:** A major drawback of reasoning models is their excessive token usage, inflating computational cost, resource demand, and latency. We show this verbosity stems not from deeper reasoning but from reinforcement learning loss minimization when models produce incorrect answers. With unsolvable problems dominating training, this effect compounds into a systematic tendency toward longer outputs. Through theoretical analysis of PPO and GRPO, we prove that incorrect answers inherently drive policies toward verbosity \textit{even when} $γ=1$, reframing response lengthening as an optimization artifact. We further uncover a consistent correlation between conciseness and correctness across reasoning and non-reasoning models. Building on these insights, we propose a two-phase RL procedure where a brief secondary stage, trained on a small set of solvable problems, significantly reduces response length while preserving or improving accuracy. Finally, we show that while GRPO shares properties with PPO, it exhibits collapse modes, limiting its reliability for concise reasoning. Our claims are supported by extensive experiments.
>
---
#### [replaced 005] LLM one-shot style transfer for Authorship Attribution and Verification
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.13302v2](https://arxiv.org/pdf/2510.13302v2)**

> **作者:** Pablo Miralles-González; Javier Huertas-Tato; Alejandro Martín; David Camacho
>
> **摘要:** Computational stylometry analyzes writing style through quantitative patterns in text, supporting applications from forensic tasks such as identity linking and plagiarism detection to literary attribution in the humanities. Supervised and contrastive approaches rely on data with spurious correlations and often confuse style with topic. Despite their natural use in AI-generated text detection, the CLM pre-training of modern LLMs has been scarcely leveraged for general authorship problems. We propose a novel unsupervised approach based on this extensive pre-training and the in-context learning capabilities of LLMs, employing the log-probabilities of an LLM to measure style transferability from one text to another. Our method significantly outperforms LLM prompting approaches of comparable scale and achieves higher accuracy than contrastively trained baselines when controlling for topical correlations. Moreover, performance scales fairly consistently with the size of the base model and, in the case of authorship verification, with an additional mechanism that increases test-time computation; enabling flexible trade-offs between computational cost and accuracy.
>
---
#### [replaced 006] Do LLMs produce texts with "human-like" lexical diversity?
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2508.00086v2](https://arxiv.org/pdf/2508.00086v2)**

> **作者:** Kelly Kendro; Jeffrey Maloney; Scott Jarvis
>
> **摘要:** The degree to which large language models (LLMs) produce writing that is truly human-like remains unclear despite the extensive empirical attention that this question has received. The present study addresses this question from the perspective of lexical diversity. Specifically, the study investigates patterns of lexical diversity in LLM-generated texts from four ChatGPT models (ChatGPT-3.5, ChatGPT-4, ChatGPT-o4 mini, and ChatGPT-4.5) in comparison with texts written by L1 and L2 English participants (n = 240) across four education levels. Six dimensions of lexical diversity were measured in each text: volume, abundance, variety-repetition, evenness, disparity, and dispersion. Results from one-way MANOVAs, one-way ANOVAs, and Support Vector Machines revealed that the ChatGPT-generated texts differed significantly from human-written texts for each variable, with ChatGPT-o4 mini and ChatGPT-4.5 differing the most. Within these two groups, ChatGPT-4.5 demonstrated higher levels of lexical diversity than older models despite producing fewer tokens. The human writers' lexical diversity did not differ across subgroups (i.e., education, language status). Altogether, the results indicate that ChatGPT models do not produce human-like texts in relation to lexical diversity, and the newer models produce less human-like text than older models. We discuss the implications of these results for language pedagogy and related applications.
>
---
#### [replaced 007] Overcoming the Generalization Limits of SLM Finetuning for Shape-Based Extraction of Datatype and Object Properties
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.03407v2](https://arxiv.org/pdf/2511.03407v2)**

> **作者:** Célian Ringwald; Fabien Gandon; Catherine Faron; Franck Michel; Hanna Abi Akl
>
> **备注:** Accepted at KCAP 2025
>
> **摘要:** Small language models (SLMs) have shown promises for relation extraction (RE) when extracting RDF triples guided by SHACL shapes focused on common datatype properties. This paper investigates how SLMs handle both datatype and object properties for a complete RDF graph extraction. We show that the key bottleneck is related to long-tail distribution of rare properties. To solve this issue, we evaluate several strategies: stratified sampling, weighted loss, dataset scaling, and template-based synthetic data augmentation. We show that the best strategy to perform equally well over unbalanced target properties is to build a training set where the number of occurrences of each property exceeds a given threshold. To enable reproducibility, we publicly released our datasets, experimental results and code. Our findings offer practical guidance for training shape-aware SLMs and highlight promising directions for future work in semantic RE.
>
---
#### [replaced 008] Emergence of psychopathological computations in large language models
- **分类: q-bio.NC; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2504.08016v2](https://arxiv.org/pdf/2504.08016v2)**

> **作者:** Soo Yong Lee; Hyunjin Hwang; Taekwan Kim; Yuyeong Kim; Kyuri Park; Jaemin Yoo; Denny Borsboom; Kijung Shin
>
> **备注:** pre-print
>
> **摘要:** Can large language models (LLMs) instantiate computations of psychopathology? An effective approach to the question hinges on addressing two factors. First, for conceptual validity, we require a general and computational account of psychopathology that is applicable to computational entities without biological embodiment or subjective experience. Second, psychopathological computations, derived from the adapted theory, need to be empirically identified within the LLM's internal processing. Thus, we establish a computational-theoretical framework to provide an account of psychopathology applicable to LLMs. Based on the framework, we conduct experiments demonstrating two key claims: first, that the computational structure of psychopathology exists in LLMs; and second, that executing this computational structure results in psychopathological functions. We further observe that as LLM size increases, the computational structure of psychopathology becomes denser and that the functions become more effective. Taken together, the empirical results corroborate our hypothesis that network-theoretic computations of psychopathology have already emerged in LLMs. This suggests that certain LLM behaviors mirroring psychopathology may not be a superficial mimicry but a feature of their internal processing. Our work shows the promise of developing a new powerful in silico model of psychopathology and also alludes to the possibility of safety threat from the AI systems with psychopathological behaviors in the near future.
>
---
#### [replaced 009] Improving the Performance of Radiology Report De-identification with Large-Scale Training and Benchmarking Against Cloud Vendor Methods
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.04079v2](https://arxiv.org/pdf/2511.04079v2)**

> **作者:** Eva Prakash; Maayane Attias; Pierre Chambon; Justin Xu; Steven Truong; Jean-Benoit Delbrouck; Tessa Cook; Curtis Langlotz
>
> **备注:** In submission to JAMIA
>
> **摘要:** Objective: To enhance automated de-identification of radiology reports by scaling transformer-based models through extensive training datasets and benchmarking performance against commercial cloud vendor systems for protected health information (PHI) detection. Materials and Methods: In this retrospective study, we built upon a state-of-the-art, transformer-based, PHI de-identification pipeline by fine-tuning on two large annotated radiology corpora from Stanford University, encompassing chest X-ray, chest CT, abdomen/pelvis CT, and brain MR reports and introducing an additional PHI category (AGE) into the architecture. Model performance was evaluated on test sets from Stanford and the University of Pennsylvania (Penn) for token-level PHI detection. We further assessed (1) the stability of synthetic PHI generation using a "hide-in-plain-sight" method and (2) performance against commercial systems. Precision, recall, and F1 scores were computed across all PHI categories. Results: Our model achieved overall F1 scores of 0.973 on the Penn dataset and 0.996 on the Stanford dataset, outperforming or maintaining the previous state-of-the-art model performance. Synthetic PHI evaluation showed consistent detectability (overall F1: 0.959 [0.958-0.960]) across 50 independently de-identified Penn datasets. Our model outperformed all vendor systems on synthetic Penn reports (overall F1: 0.960 vs. 0.632-0.754). Discussion: Large-scale, multimodal training improved cross-institutional generalization and robustness. Synthetic PHI generation preserved data utility while ensuring privacy. Conclusion: A transformer-based de-identification model trained on diverse radiology datasets outperforms prior academic and commercial systems in PHI detection and establishes a new benchmark for secure clinical text processing.
>
---
#### [replaced 010] RAG-BioQA Retrieval-Augmented Generation for Long-Form Biomedical Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.01612v2](https://arxiv.org/pdf/2510.01612v2)**

> **作者:** Lovely Yeswanth Panchumarthi; Sai Prasad Gudari; Atharva Negi; Praveen Raj Budime; Harsit Upadhya
>
> **备注:** Need to work on the methodology more
>
> **摘要:** The exponential growth of biomedical literature creates significant challenges for accessing precise medical information. Current biomedical question-answering systems primarily focus on short-form answers, failing to provide the comprehensive explanations necessary for clinical decision-making. We present RAG-BioQA, a novel framework combining retrieval-augmented generation with domain-specific fine-tuning to produce evidence-based, long-form biomedical answers. Our approach integrates BioBERT embeddings with FAISS indexing and compares various re-ranking strategies (BM25, ColBERT, MonoT5) to optimize context selection before synthesizing evidence through a fine-tuned T5 model. Experimental results on the PubMedQA dataset show significant improvements over baselines, with our best model achieving substantial gains across BLEU, ROUGE, and METEOR metrics, advancing the state of accessible, evidence-based biomedical knowledge retrieval.
>
---
#### [replaced 011] When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.07318v2](https://arxiv.org/pdf/2511.07318v2)**

> **作者:** Shaowen Wang; Yiqi Dong; Ruinian Chang; Tansheng Zhu; Yuebo Sun; Kaifeng Lyu; Jian Li
>
> **摘要:** Despite substantial advances, large language models (LLMs) continue to exhibit hallucinations, generating plausible yet incorrect responses. In this paper, we highlight a critical yet previously underexplored class of hallucinations driven by spurious correlations -- superficial but statistically prominent associations between features (e.g., surnames) and attributes (e.g., nationality) present in the training data. We demonstrate that these spurious correlations induce hallucinations that are confidently generated, immune to model scaling, evade current detection methods, and persist even after refusal fine-tuning. Through systematically controlled synthetic experiments and empirical evaluations on state-of-the-art open-source and proprietary LLMs (including GPT-5), we show that existing hallucination detection methods, such as confidence-based filtering and inner-state probing, fundamentally fail in the presence of spurious correlations. Our theoretical analysis further elucidates why these statistical biases intrinsically undermine confidence-based detection techniques. Our findings thus emphasize the urgent need for new approaches explicitly designed to address hallucinations caused by spurious correlations.
>
---
#### [replaced 012] Beyond Human Judgment: A Bayesian Evaluation of LLMs' Moral Values Understanding
- **分类: cs.CL; cs.HC**

- **链接: [https://arxiv.org/pdf/2508.13804v3](https://arxiv.org/pdf/2508.13804v3)**

> **作者:** Maciej Skorski; Alina Landowska
>
> **备注:** Appears in UncertaiNLP@EMNLP 2025
>
> **摘要:** How do Large Language Models understand moral dimensions compared to humans? This first large-scale Bayesian evaluation of market-leading language models provides the answer. In contrast to prior work using deterministic ground truth (majority or inclusion rules), we model annotator disagreements to capture both aleatoric uncertainty (inherent human disagreement) and epistemic uncertainty (model domain sensitivity). We evaluated the best language models (Claude Sonnet 4, DeepSeek-V3, Llama 4 Maverick) across 250K+ annotations from nearly 700 annotators in 100K+ texts spanning social networks, news and forums. Our GPU-optimized Bayesian framework processed 1M+ model queries, revealing that AI models typically rank among the top 25\% of human annotators, performing much better than average balanced accuracy. Importantly, we find that AI produces far fewer false negatives than humans, highlighting their more sensitive moral detection capabilities.
>
---
#### [replaced 013] AI use in American newspapers is widespread, uneven, and rarely disclosed
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2510.18774v3](https://arxiv.org/pdf/2510.18774v3)**

> **作者:** Jenna Russell; Marzena Karpinska; Destiny Akinode; Katherine Thai; Bradley Emi; Max Spero; Mohit Iyyer
>
> **摘要:** AI is rapidly transforming journalism, but the extent of its use in published newspaper articles remains unclear. We address this gap by auditing a large-scale dataset of 186K articles from online editions of 1.5K American newspapers published in the summer of 2025. Using Pangram, a state-of-the-art AI detector, we discover that approximately 9% of newly-published articles are either partially or fully AI-generated. This AI use is unevenly distributed, appearing more frequently in smaller, local outlets, in specific topics such as weather and technology, and within certain ownership groups. We also analyze 45K opinion pieces from Washington Post, New York Times, and Wall Street Journal, finding that they are 6.4 times more likely to contain AI-generated content than news articles from the same publications, with many AI-flagged op-eds authored by prominent public figures. Despite this prevalence, we find that AI use is rarely disclosed: a manual audit of 100 AI-flagged articles found only five disclosures of AI use. Overall, our audit highlights the immediate need for greater transparency and updated editorial standards regarding the use of AI in journalism to maintain public trust.
>
---
#### [replaced 014] From Perception to Reasoning: Deep Thinking Empowers Multimodal Large Language Models
- **分类: cs.CL; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12861v3](https://arxiv.org/pdf/2511.12861v3)**

> **作者:** Wenxin Zhu; Andong Chen; Yuchen Song; Kehai Chen; Conghui Zhu; Ziyan Chen; Tiejun Zhao
>
> **备注:** Survey; 7 figures, 3 tables, 44 pages
>
> **摘要:** With the remarkable success of Multimodal Large Language Models (MLLMs) in perception tasks, enhancing their complex reasoning capabilities has emerged as a critical research focus. Existing models still suffer from challenges such as opaque reasoning paths and insufficient generalization ability. Chain-of-Thought (CoT) reasoning, which has demonstrated significant efficacy in language models by enhancing reasoning transparency and output interpretability, holds promise for improving model reasoning capabilities when extended to the multimodal domain. This paper provides a systematic review centered on "Multimodal Chain-of-Thought" (MCoT). First, it analyzes the background and theoretical motivations for its inception from the perspectives of technical evolution and task demands. Then, it introduces mainstream MCoT methods from three aspects: CoT paradigms, the post-training stage, and the inference stage, while also analyzing their underlying mechanisms. Furthermore, the paper summarizes existing evaluation benchmarks and metrics, and discusses the application scenarios of MCoT. Finally, it analyzes the challenges currently facing MCoT and provides an outlook on its future research directions.
>
---
#### [replaced 015] Testing Hypotheses from the Social Approval Theory of Online Hate: An Analysis of 110 Million Messages from Parler
- **分类: cs.CL; cs.SI**

- **链接: [https://arxiv.org/pdf/2507.10810v2](https://arxiv.org/pdf/2507.10810v2)**

> **作者:** David M. Markowitz; Samuel Hardman Taylor
>
> **摘要:** We examined how online hate is motivated by receiving social approval via Walther's (2024) social approval theory of online hate, which argues (H1a) more signals of social approval on hate messages predicts more subsequent hate messages, and (H1b) as social approval increases, hate speech becomes more extreme. Using 110 million messages from Parler (2018-2021), we observed the number of upvotes received on a hate speech post was unassociated with hate speech in one's next post and during the next month, three-months, and six-months. The number of upvotes received on (extreme) hate speech comments, however, was positively associated with (extreme) hate speech during the next week, month, three-months, and six-months. Between-person effects revealed an average positive relationship between social approval and hate speech production at all time intervals. For comments, social approval linked more strongly to online hate than social disapproval. Social approval is a critical mechanism facilitating online hate propagation.
>
---
#### [replaced 016] Fairness Evaluation of Large Language Models in Academic Library Reference Services
- **分类: cs.CL; cs.AI; cs.DL**

- **链接: [https://arxiv.org/pdf/2507.04224v3](https://arxiv.org/pdf/2507.04224v3)**

> **作者:** Haining Wang; Jason Clark; Yueru Yan; Star Bradley; Ruiyang Chen; Yiqiong Zhang; Hengyi Fu; Zuoyu Tian
>
> **摘要:** As libraries explore large language models (LLMs) for use in virtual reference services, a key question arises: Can LLMs serve all users equitably, regardless of demographics or social status? While they offer great potential for scalable support, LLMs may also reproduce societal biases embedded in their training data, risking the integrity of libraries' commitment to equitable service. To address this concern, we evaluate whether LLMs differentiate responses across user identities by prompting six state-of-the-art LLMs to assist patrons differing in sex, race/ethnicity, and institutional role. We find no evidence of differentiation by race or ethnicity, and only minor evidence of stereotypical bias against women in one model. LLMs demonstrate nuanced accommodation of institutional roles through the use of linguistic choices related to formality, politeness, and domain-specific vocabularies, reflecting professional norms rather than discriminatory treatment. These findings suggest that current LLMs show a promising degree of readiness to support equitable and contextually appropriate communication in academic library reference services.
>
---
#### [replaced 017] AraFinNews: Arabic Financial Summarisation with Domain-Adapted LLMs
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.01265v2](https://arxiv.org/pdf/2511.01265v2)**

> **作者:** Mo El-Haj; Paul Rayson
>
> **备注:** 9 pages
>
> **摘要:** This paper examines how domain specificity affects abstractive summarisation of Arabic financial texts using large language models (LLMs). We present AraFinNews, the largest publicly available Arabic financial news dataset to date, comprising 212,500 article-headline pairs spanning almost a decade of reporting from October 2015 to July 2025. Developed as an Arabic counterpart to major English summarisation corpora such as CNN/DailyMail, AraFinNews offers a strong benchmark for assessing domain-focused language understanding and generation in financial contexts. Using this resource, we evaluate transformer-based models, including mT5, AraT5 and the domain-adapted FinAraT5, to investigate how financial-domain pretraining influences accuracy, numerical reliability and stylistic alignment with professional reporting. The results show that domain-adapted models produce more coherent summaries, particularly when handling quantitative and entity-centred information. These findings underscore the value of domain-specific adaptation for improving narrative fluency in Arabic financial summarisation. The dataset is freely available for non-commercial research at https://github.com/ArabicNLP-UK/AraFinNews.
>
---
#### [replaced 018] Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.13646v2](https://arxiv.org/pdf/2511.13646v2)**

> **作者:** Chunqiu Steven Xia; Zhe Wang; Yan Yang; Yuxiang Wei; Lingming Zhang
>
> **摘要:** Large Language Models (LLMs) are reshaping almost all industries, including software engineering. In recent years, a number of LLM agents have been proposed to solve real-world software problems. Such software agents are typically equipped with a suite of coding tools and can autonomously decide the next actions to form complete trajectories to solve end-to-end software tasks. While promising, they typically require dedicated design and may still be suboptimal, since it can be extremely challenging and costly to exhaust the entire agent scaffold design space. Recognizing that software agents are inherently software themselves that can be further refined/modified, researchers have proposed a number of self-improving software agents recently, including the Darwin-Gödel Machine (DGM). Meanwhile, such self-improving agents require costly offline training on specific benchmarks and may not generalize well across different LLMs or benchmarks. In this paper, we propose Live-SWE-agent, the first live software agent that can autonomously and continuously evolve itself on-the-fly during runtime when solving real-world software problems. More specifically, Live-SWE-agent starts with the most basic agent scaffold with only access to bash tools (e.g., mini-SWE-agent), and autonomously evolves its own scaffold implementation while solving real-world software problems. Our evaluation on the widely studied SWE-bench Verified benchmark shows that LIVE-SWE-AGENT can achieve an impressive solve rate of 77.4% without test-time scaling, outperforming all existing software agents, including the best proprietary solution. Moreover, Live-SWE-agent outperforms state-of-the-art manually crafted software agents on the recent SWE-Bench Pro benchmark, achieving the best-known solve rate of 45.8%.
>
---
#### [replaced 019] A systematic review of relation extraction task since the emergence of Transformers
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.03610v2](https://arxiv.org/pdf/2511.03610v2)**

> **作者:** Ringwald Celian; Gandon; Fabien; Faron Catherine; Michel Franck; Abi Akl Hanna
>
> **备注:** Submited at ACM-Computing Surveys + The resulting annotated Zotero bibliography : https://www.zotero.org/groups/6070963/scilex_re_systlitreview/library + SciLEx software: https://github.com/Wimmics/SciLEx
>
> **摘要:** This article presents a systematic review of relation extraction (RE) research since the advent of Transformer-based models. Using an automated framework to collect and annotate publications, we analyze 34 surveys, 64 datasets, and 104 models published between 2019 and 2024. The review highlights methodological advances, benchmark resources, and the integration of semantic web technologies. By consolidating results across multiple dimensions, the study identifies current trends, limitations, and open challenges, offering researchers and practitioners a comprehensive reference for understanding the evolution and future directions of RE.
>
---
#### [replaced 020] Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2507.05248v2](https://arxiv.org/pdf/2507.05248v2)**

> **作者:** Ziqi Miao; Lijun Li; Yuan Xiong; Zhenhua Liu; Pengyu Zhu; Jing Shao
>
> **备注:** 20 pages, 10 figures. Code and data available at https://github.com/Dtc7w3PQ/Response-Attack
>
> **摘要:** Contextual priming, where earlier stimuli covertly bias later judgments, offers an unexplored attack surface for large language models (LLMs). We uncover a contextual priming vulnerability in which the previous response in the dialogue can steer its subsequent behavior toward policy-violating content. While existing jailbreak attacks largely rely on single-turn or multi-turn prompt manipulations, or inject static in-context examples, these methods suffer from limited effectiveness, inefficiency, or semantic drift. We introduce Response Attack (RA), a novel framework that strategically leverages intermediate, mildly harmful responses as contextual primers within a dialogue. By reformulating harmful queries and injecting these intermediate responses before issuing a targeted trigger prompt, RA exploits a previously overlooked vulnerability in LLMs. Extensive experiments across eight state-of-the-art LLMs show that RA consistently achieves significantly higher attack success rates than nine leading jailbreak baselines. Our results demonstrate that the success of RA is directly attributable to the strategic use of intermediate responses, which induce models to generate more explicit and relevant harmful content while maintaining stealth, efficiency, and fidelity to the original query. The code and data are available at https://github.com/Dtc7w3PQ/Response-Attack.
>
---
#### [replaced 021] SALT: Steering Activations towards Leakage-free Thinking in Chain of Thought
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.07772v2](https://arxiv.org/pdf/2511.07772v2)**

> **作者:** Shourya Batra; Pierce Tillman; Samarth Gaggar; Shashank Kesineni; Kevin Zhu; Sunishchal Dev; Ashwinee Panda; Vasu Sharma; Maheep Chaudhary
>
> **摘要:** As Large Language Models (LLMs) evolve into personal assistants with access to sensitive user data, they face a critical privacy challenge: while prior work has addressed output-level privacy, recent findings reveal that LLMs often leak private information through their internal reasoning processes, violating contextual privacy expectations. These leaky thoughts occur when models inadvertently expose sensitive details in their reasoning traces, even when final outputs appear safe. The challenge lies in preventing such leakage without compromising the model's reasoning capabilities, requiring a delicate balance between privacy and utility. We introduce Steering Activations towards Leakage-free Thinking (SALT), a lightweight test-time intervention that mitigates privacy leakage in model's Chain of Thought (CoT) by injecting targeted steering vectors into hidden state. We identify the high-leakage layers responsible for this behavior. Through experiments across multiple LLMs, we demonstrate that SALT achieves reductions including $18.2\%$ reduction in CPL on QwQ-32B, $17.9\%$ reduction in CPL on Llama-3.1-8B, and $31.2\%$ reduction in CPL on Deepseek in contextual privacy leakage dataset AirGapAgent-R while maintaining comparable task performance and utility. Our work establishes SALT as a practical approach for test-time privacy protection in reasoning-capable language models, offering a path toward safer deployment of LLM-based personal agents.
>
---
#### [replaced 022] Evaluating Large Language Models for Diacritic Restoration in Romanian Texts: A Comparative Study
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.13182v3](https://arxiv.org/pdf/2511.13182v3)**

> **作者:** Mihai Nadas; Laura Diosan
>
> **备注:** The original submission contained metadata errors and requires correction. A revised and complete version will be submitted as a replacement
>
> **摘要:** Automatic diacritic restoration is crucial for text processing in languages with rich diacritical marks, such as Romanian. This study evaluates the performance of several large language models (LLMs) in restoring diacritics in Romanian texts. Using a comprehensive corpus, we tested models including OpenAI's GPT-3.5, GPT-4, GPT-4o, Google's Gemini 1.0 Pro, Meta's Llama 2 and Llama 3, MistralAI's Mixtral 8x7B Instruct, airoboros 70B, and OpenLLM-Ro's RoLlama 2 7B, under multiple prompt templates ranging from zero-shot to complex multi-shot instructions. Results show that models such as GPT-4o achieve high diacritic restoration accuracy, consistently surpassing a neutral echo baseline, while others, including Meta's Llama family, exhibit wider variability. These findings highlight the impact of model architecture, training data, and prompt design on diacritic restoration performance and outline promising directions for improving NLP tools for diacritic-rich languages.
>
---
#### [replaced 023] WER is Unaware: Assessing How ASR Errors Distort Clinical Understanding in Patient Facing Dialogue
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.16544v2](https://arxiv.org/pdf/2511.16544v2)**

> **作者:** Zachary Ellis; Jared Joselowitz; Yash Deo; Yajie He; Anna Kalygina; Aisling Higham; Mana Rahimzadeh; Yan Jia; Ibrahim Habli; Ernest Lim
>
> **摘要:** As Automatic Speech Recognition (ASR) is increasingly deployed in clinical dialogue, standard evaluations still rely heavily on Word Error Rate (WER). This paper challenges that standard, investigating whether WER or other common metrics correlate with the clinical impact of transcription errors. We establish a gold-standard benchmark by having expert clinicians compare ground-truth utterances to their ASR-generated counterparts, labeling the clinical impact of any discrepancies found in two distinct doctor-patient dialogue datasets. Our analysis reveals that WER and a comprehensive suite of existing metrics correlate poorly with the clinician-assigned risk labels (No, Minimal, or Significant Impact). To bridge this evaluation gap, we introduce an LLM-as-a-Judge, programmatically optimized using GEPA through DSPy to replicate expert clinical assessment. The optimized judge (Gemini-2.5-Pro) achieves human-comparable performance, obtaining 90% accuracy and a strong Cohen's $κ$ of 0.816. This work provides a validated, automated framework for moving ASR evaluation beyond simple textual fidelity to a necessary, scalable assessment of safety in clinical dialogue.
>
---
#### [replaced 024] DiffTester: Accelerating Unit Test Generation for Diffusion LLMs via Repetitive Pattern
- **分类: cs.SE; cs.CL**

- **链接: [https://arxiv.org/pdf/2509.24975v2](https://arxiv.org/pdf/2509.24975v2)**

> **作者:** Lekang Yang; Yuetong Liu; Yitong Zhang; Jia Li
>
> **备注:** Update reference
>
> **摘要:** Software development relies heavily on extensive unit testing, which makes the efficiency of automated Unit Test Generation (UTG) particularly important. However, most existing LLMs generate test cases one token at a time in each forward pass, which leads to inefficient UTG. Recently, diffusion LLMs (dLLMs) have emerged, offering promising parallel generation capabilities and showing strong potential for efficient UTG. Despite this advantage, their application to UTG is still constrained by a clear trade-off between efficiency and test quality, since increasing the number of tokens generated in each step often causes a sharp decline in the quality of test cases. To overcome this limitation, we present DiffTester, an acceleration framework specifically tailored for dLLMs in UTG. The key idea of DiffTester is that unit tests targeting the same focal method often share repetitive structural patterns. By dynamically identifying these common patterns through abstract syntax tree analysis during generation, DiffTester adaptively increases the number of tokens produced at each step without compromising the quality of the output. To enable comprehensive evaluation, we extend the original TestEval benchmark, which was limited to Python, by introducing additional programming languages including Java and C++. Extensive experiments on three benchmarks with two representative models show that DiffTester delivers significant acceleration while preserving test coverage. Moreover, DiffTester generalizes well across different dLLMs and programming languages, providing a practical and scalable solution for efficient UTG in software development. Code and data are publicly available at https://github.com/wellbeingyang/DLM4UTG-open .
>
---
#### [replaced 025] EventWeave: A Dynamic Framework for Capturing Core and Supporting Events in Dialogue Systems
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2503.23078v2](https://arxiv.org/pdf/2503.23078v2)**

> **作者:** Zhengyi Zhao; Shubo Zhang; Yiming Du; Bin Liang; Baojun Wang; Zhongyang Li; Binyang Li; Kam-Fai Wong
>
> **摘要:** Large language models have improved dialogue systems, but often process conversational turns in isolation, overlooking the event structures that guide natural interactions. Hence we introduce \textbf{EventWeave}, a framework that explicitly models relationships between conversational events to generate more contextually appropriate dialogue responses. EventWeave constructs a dynamic event graph that distinguishes between core events (main goals) and supporting events (interconnected details), employing a multi-head attention mechanism to selectively determine which events are most relevant to the current turn. Unlike summarization or standard graph-based approaches, our method captures three distinct relationship types between events, allowing for more nuanced context modeling. Experiments on three dialogue datasets demonstrate that EventWeave produces more natural and contextually appropriate responses while requiring less computational overhead than models processing the entire dialogue history. Ablation studies confirm improvements stem from better event relationship modeling rather than increased information density. Our approach effectively balances comprehensive context understanding with generating concise responses, maintaining strong performance across various dialogue lengths through targeted optimization techniques.
>
---
#### [replaced 026] Task-Aligned Tool Recommendation for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2411.09613v2](https://arxiv.org/pdf/2411.09613v2)**

> **作者:** Hang Gao; Yongfeng Zhang
>
> **备注:** IJCNLP-AACL 2025 Main
>
> **摘要:** By augmenting Large Language Models (LLMs) with external tools, their capacity to solve complex problems has been significantly enhanced. However, despite ongoing advancements in the parsing capabilities of LLMs, incorporating all available tools simultaneously in the prompt remains impractical due to the vast number of external tools. Consequently, it is essential to provide LLMs with a precise set of tools tailored to the specific task, considering both quantity and quality. Current tool retrieval methods primarily focus on refining the ranking list of tools and directly packaging a fixed number of top-ranked tools as the tool set. However, these approaches often fail to equip LLMs with the optimal set of tools prior to execution, since the optimal number of tools for different tasks could be different, resulting in inefficiencies such as redundant or unsuitable tools, which impede immediate access to the most relevant tools. This paper addresses the challenge of recommending precise toolsets for LLMs. We introduce the problem of tool recommendation, define its scope, and propose a novel Precision-driven Tool Recommendation (PTR) approach. PTR captures an initial, concise set of tools by leveraging historical tool bundle usage and dynamically adjusts the tool set by performing tool matching, culminating in a multi-view-based tool addition. Additionally, we present a new dataset, RecTools, and a metric, TRACC, designed to evaluate the effectiveness of tool recommendation for LLMs. We further validate our design choices through comprehensive experiments, demonstrating promising accuracy across two open benchmarks and our RecTools dataset.
>
---
#### [replaced 027] Fine-Grained Reward Optimization for Machine Translation using Error Severity Mappings
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2411.05986v3](https://arxiv.org/pdf/2411.05986v3)**

> **作者:** Miguel Moura Ramos; Tomás Almeida; Daniel Vareta; Filipe Azevedo; Sweta Agrawal; Patrick Fernandes; André F. T. Martins
>
> **摘要:** Reinforcement learning (RL) has been proven to be an effective and robust method for training neural machine translation systems, especially when paired with powerful reward models that accurately assess translation quality. However, most research has focused on RL methods that use sentence-level feedback, leading to inefficient learning signals due to the reward sparsity problem -- the model receives a single score for the entire sentence. To address this, we propose a novel approach that leverages fine-grained, token-level quality assessments along with error severity levels using RL methods. Specifically, we use xCOMET, a state-of-the-art quality estimation system, as our token-level reward model. We conduct experiments on small and large translation datasets with standard encoder-decoder and large language models-based machine translation systems, comparing the impact of sentence-level versus fine-grained reward signals on translation quality. Our results show that training with token-level rewards improves translation quality across language pairs over baselines according to both automatic and human evaluation. Furthermore, token-level reward optimization improves training stability, evidenced by a steady increase in mean rewards over training epochs.
>
---
#### [replaced 028] MiniLLM: Knowledge Distillation of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2306.08543v5](https://arxiv.org/pdf/2306.08543v5)**

> **作者:** Yuxian Gu; Li Dong; Furu Wei; Minlie Huang
>
> **备注:** Published as a conference paper in ICLR 2024
>
> **摘要:** Knowledge Distillation (KD) is a promising technique for reducing the high computational demand of large language models (LLMs). However, previous KD methods are primarily applied to white-box classification models or training small models to imitate black-box model APIs like ChatGPT. How to effectively distill the knowledge of white-box LLMs into small models is still under-explored, which becomes more important with the prosperity of open-source LLMs. In this work, we propose a KD approach that distills LLMs into smaller language models. We first replace the forward Kullback-Leibler divergence (KLD) objective in the standard KD approaches with reverse KLD, which is more suitable for KD on generative language models, to prevent the student model from overestimating the low-probability regions of the teacher distribution. Then, we derive an effective on-policy optimization approach to learn this objective. The student models are named MiniLLM. Extensive experiments in the instruction-following setting show that MiniLLM generates more precise responses with higher overall quality, lower exposure bias, better calibration, and higher long-text generation performance than the baselines. Our method is scalable for different model families with 120M to 13B parameters. Our code, data, and model checkpoints can be found in https://github.com/microsoft/LMOps/tree/main/minillm.
>
---
#### [replaced 029] RPRO: Ranked Preference Reinforcement Optimization for Enhancing Medical QA and Diagnostic Reasoning
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2509.00974v4](https://arxiv.org/pdf/2509.00974v4)**

> **作者:** Chia-Hsuan Hsu; Jun-En Ding; Hsin-Ling Hsu; Chih-Ho Hsu; Li-Hung Yao; Chun-Chieh Liao; Feng Liu; Fang-Ming Hung
>
> **摘要:** Medical question answering requires advanced reasoning that integrates domain knowledge with logical inference. However, existing large language models (LLMs) often generate reasoning chains that lack factual accuracy and clinical reliability. We propose Ranked Preference Reinforcement Optimization (RPRO), a novel framework that combines reinforcement learning with preference-driven reasoning refinement to enhance clinical chain-of-thought (CoT) performance. RPRO distinguishes itself from prior approaches by employing task-adaptive reasoning templates and a probabilistic evaluation mechanism that aligns model outputs with established clinical workflows, while automatically identifying and correcting low-quality reasoning chains. Unlike traditional pairwise preference methods, RPRO introduces a groupwise ranking optimization based on the Bradley--Terry model and incorporates KL-divergence regularization for stable training. Experiments on PubMedQA, MedQA-USMLE, and a real-world clinical dataset from Far Eastern Memorial Hospital (FEMH) demonstrate consistent improvements over strong baselines. Remarkably, our 2B-parameter model outperforms much larger 7B--20B models, including medical-specialized variants. These findings demonstrate that combining preference optimization with quality-driven refinement provides a scalable and clinically grounded approach to building more reliable medical LLMs.
>
---
#### [replaced 030] Resolving Sentiment Discrepancy for Multimodal Sentiment Detection via Semantics Completion and Decomposition
- **分类: cs.CV; cs.CL; cs.MM; cs.SI**

- **链接: [https://arxiv.org/pdf/2407.07026v2](https://arxiv.org/pdf/2407.07026v2)**

> **作者:** Daiqing Wu; Dongbao Yang; Huawen Shen; Can Ma; Yu Zhou
>
> **备注:** Accepted by Pattern Recognition
>
> **摘要:** With the proliferation of social media posts in recent years, the need to detect sentiments in multimodal (image-text) content has grown rapidly. Since posts are user-generated, the image and text from the same post can express different or even contradictory sentiments, leading to potential \textbf{sentiment discrepancy}. However, existing works mainly adopt a single-branch fusion structure that primarily captures the consistent sentiment between image and text. The ignorance or implicit modeling of discrepant sentiment results in compromised unimodal encoding and limited performance. In this paper, we propose a semantics Completion and Decomposition (CoDe) network to resolve the above issue. In the semantics completion module, we complement image and text representations with the semantics of the in-image text, helping bridge the sentiment gap. In the semantics decomposition module, we decompose image and text representations with exclusive projection and contrastive learning, thereby explicitly capturing the discrepant sentiment between modalities. Finally, we fuse image and text representations by cross-attention and combine them with the learned discrepant sentiment for final classification. Extensive experiments on four datasets demonstrate the superiority of CoDe and the effectiveness of each proposed module.
>
---
#### [replaced 031] ToolHaystack: Stress-Testing Tool-Augmented Language Models in Realistic Long-Term Interactions
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2505.23662v2](https://arxiv.org/pdf/2505.23662v2)**

> **作者:** Beong-woo Kwak; Minju Kim; Dongha Lim; Hyungjoo Chae; Dongjin Kang; Sunghwan Kim; Dongil Yang; Jinyoung Yeo
>
> **备注:** Our code and data are available at https://github.com/bwookwak/ToolHaystack Edited for adding acknowledgement section
>
> **摘要:** Large language models (LLMs) have demonstrated strong capabilities in using external tools to address user inquiries. However, most existing evaluations assume tool use in short contexts, offering limited insight into model behavior during realistic long-term interactions. To fill this gap, we introduce ToolHaystack, a benchmark for testing the tool use capabilities in long-term interactions. Each test instance in ToolHaystack includes multiple tasks execution contexts and realistic noise within a continuous conversation, enabling assessment of how well models maintain context and handle various disruptions. By applying this benchmark to 14 state-of-the-art LLMs, we find that while current models perform well in standard multi-turn settings, they often significantly struggle in ToolHaystack, highlighting critical gaps in their long-term robustness not revealed by previous tool benchmarks.
>
---
#### [replaced 032] The Rise of Parameter Specialization for Knowledge Storage in Large Language Models
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2505.17260v2](https://arxiv.org/pdf/2505.17260v2)**

> **作者:** Yihuai Hong; Yiran Zhao; Wei Tang; Yang Deng; Yu Rong; Wenxuan Zhang
>
> **备注:** Accepted in NeurIPS 2025
>
> **摘要:** Over time, a growing wave of large language models from various series has been introduced to the community. Researchers are striving to maximize the performance of language models with constrained parameter sizes. However, from a microscopic perspective, there has been limited research on how to better store knowledge in model parameters, particularly within MLPs, to enable more effective utilization of this knowledge by the model. In this work, we analyze twenty publicly available open-source large language models to investigate the relationship between their strong performance and the way knowledge is stored in their corresponding MLP parameters. Our findings reveal that as language models become more advanced and demonstrate stronger knowledge capabilities, their parameters exhibit increased specialization. Specifically, parameters in the MLPs tend to be more focused on encoding similar types of knowledge. We experimentally validate that this specialized distribution of knowledge contributes to improving the efficiency of knowledge utilization in these models. Furthermore, by conducting causal training experiments, we confirm that this specialized knowledge distribution plays a critical role in improving the model's efficiency in leveraging stored knowledge.
>
---
