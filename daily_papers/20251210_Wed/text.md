# 自然语言处理 cs.CL

- **最新发布 30 篇**

- **更新 42 篇**

## 最新发布

#### [new 001] Ask, Answer, and Detect: Role-Playing LLMs for Personality Detection with Question-Conditioned Mixture-of-Experts
- **分类: cs.CL**

- **简介: 该论文研究人格检测任务，旨在缓解标签稀缺和语义映射不明确问题。提出ROME框架，利用LLM角色扮演生成问卷回答，引入心理知识作为中间监督，通过多任务学习提升检测性能。**

- **链接: [https://arxiv.org/pdf/2512.08814v1](https://arxiv.org/pdf/2512.08814v1)**

> **作者:** Yifan Lyu; Liang Zhang
>
> **摘要:** Understanding human personality is crucial for web applications such as personalized recommendation and mental health assessment. Existing studies on personality detection predominantly adopt a "posts -> user vector -> labels" modeling paradigm, which encodes social media posts into user representations for predicting personality labels (e.g., MBTI labels). While recent advances in large language models (LLMs) have improved text encoding capacities, these approaches remain constrained by limited supervision signals due to label scarcity, and under-specified semantic mappings between user language and abstract psychological constructs. We address these challenges by proposing ROME, a novel framework that explicitly injects psychological knowledge into personality detection. Inspired by standardized self-assessment tests, ROME leverages LLMs' role-play capability to simulate user responses to validated psychometric questionnaires. These generated question-level answers transform free-form user posts into interpretable, questionnaire-grounded evidence linking linguistic cues to personality labels, thereby providing rich intermediate supervision to mitigate label scarcity while offering a semantic reasoning chain that guides and simplifies the text-to-personality mapping learning. A question-conditioned Mixture-of-Experts module then jointly routes over post and question representations, learning to answer questionnaire items under explicit supervision. The predicted answers are summarized into an interpretable answer vector and fused with the user representation for final prediction within a multi-task learning framework, where question answering serves as a powerful auxiliary task for personality detection. Extensive experiments on two real-world datasets demonstrate that ROME consistently outperforms state-of-the-art baselines, achieving improvements (15.41% on Kaggle dataset).
>
---
#### [new 002] HealthcareNLP: where are we and what is next?
- **分类: cs.CL**

- **简介: 该论文属于综述与教程类任务，旨在系统介绍医疗领域NLP（HealthcareNLP）的核心子领域与挑战。它梳理了数据资源、NLP评估、患者应用三层架构，涵盖隐私保护、可解释性、大模型融合等关键问题，填补现有综述空白，并提供实践教程。**

- **链接: [https://arxiv.org/pdf/2512.08617v1](https://arxiv.org/pdf/2512.08617v1)**

> **作者:** Lifeng Han; Paul Rayson; Suzan Verberne; Andrew Moore; Goran Nenadic
>
> **备注:** Accepted Tutorial by LREC 2026 https://lrec2026.info/
>
> **摘要:** This proposed tutorial focuses on Healthcare Domain Applications of NLP, what we have achieved around HealthcareNLP, and the challenges that lie ahead for the future. Existing reviews in this domain either overlook some important tasks, such as synthetic data generation for addressing privacy concerns, or explainable clinical NLP for improved integration and implementation, or fail to mention important methodologies, including retrieval augmented generation and the neural symbolic integration of LLMs and KGs. In light of this, the goal of this tutorial is to provide an introductory overview of the most important sub-areas of a patient- and resource-oriented HealthcareNLP, with three layers of hierarchy: data/resource layer: annotation guidelines, ethical approvals, governance, synthetic data; NLP-Eval layer: NLP tasks such as NER, RE, sentiment analysis, and linking/coding with categorised methods, leading to explainable HealthAI; patients layer: Patient Public Involvement and Engagement (PPIE), health literacy, translation, simplification, and summarisation (also NLP tasks), and shared decision-making support. A hands-on session will be included in the tutorial for the audience to use HealthcareNLP applications. The target audience includes NLP practitioners in the healthcare application domain, NLP researchers who are interested in domain applications, healthcare researchers, and students from NLP fields. The type of tutorial is "Introductory to CL/NLP topics (HealthcareNLP)" and the audience does not need prior knowledge to attend this. Tutorial materials: https://github.com/4dpicture/HealthNLP
>
---
#### [new 003] Universal Adversarial Suffixes for Language Models Using Reinforcement Learning with Calibrated Reward
- **分类: cs.CL**

- **简介: 该论文研究语言模型对对抗性后缀的脆弱性，属安全与鲁棒性任务。提出用强化学习（PPO）生成通用对抗后缀，通过校准奖励提升跨任务和模型的迁移性，在多数据集和模型上验证了有效性。**

- **链接: [https://arxiv.org/pdf/2512.08131v1](https://arxiv.org/pdf/2512.08131v1)**

> **作者:** Sampriti Soor; Suklav Ghosh; Arijit Sur
>
> **备注:** 5 pages
>
> **摘要:** Language models are vulnerable to short adversarial suffixes that can reliably alter predictions. Previous works usually find such suffixes with gradient search or rule-based methods, but these are brittle and often tied to a single task or model. In this paper, a reinforcement learning framework is used where the suffix is treated as a policy and trained with Proximal Policy Optimization against a frozen model as a reward oracle. Rewards are shaped using calibrated cross-entropy, removing label bias and aggregating across surface forms to improve transferability. The proposed method is evaluated on five diverse NLP benchmark datasets, covering sentiment, natural language inference, paraphrase, and commonsense reasoning, using three distinct language models: Qwen2-1.5B Instruct, TinyLlama-1.1B Chat, and Phi-1.5. Results show that RL-trained suffixes consistently degrade accuracy and transfer more effectively across tasks and models than previous adversarial triggers of similar genres.
>
---
#### [new 004] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于RAG中的幻觉检测任务，旨在解决生成内容不忠于检索证据的问题。作者利用稀疏自编码器解析LLM内部激活，提出轻量级、可解释的检测器RAGLens，有效识别并缓解RAG幻觉。**

- **链接: [https://arxiv.org/pdf/2512.08892v1](https://arxiv.org/pdf/2512.08892v1)**

> **作者:** Guangzhi Xiong; Zhenghao He; Bohan Liu; Sanchit Sinha; Aidong Zhang
>
> **摘要:** Retrieval-Augmented Generation (RAG) improves the factuality of large language models (LLMs) by grounding outputs in retrieved evidence, but faithfulness failures, where generations contradict or extend beyond the provided sources, remain a critical challenge. Existing hallucination detection methods for RAG often rely either on large-scale detector training, which requires substantial annotated data, or on querying external LLM judges, which leads to high inference costs. Although some approaches attempt to leverage internal representations of LLMs for hallucination detection, their accuracy remains limited. Motivated by recent advances in mechanistic interpretability, we employ sparse autoencoders (SAEs) to disentangle internal activations, successfully identifying features that are specifically triggered during RAG hallucinations. Building on a systematic pipeline of information-based feature selection and additive feature modeling, we introduce RAGLens, a lightweight hallucination detector that accurately flags unfaithful RAG outputs using LLM internal representations. RAGLens not only achieves superior detection performance compared to existing methods, but also provides interpretable rationales for its decisions, enabling effective post-hoc mitigation of unfaithful RAG. Finally, we justify our design choices and reveal new insights into the distribution of hallucination-related signals within LLMs. The code is available at https://github.com/Teddy-XiongGZ/RAGLens.
>
---
#### [new 005] Segment, Embed, and Align: A Universal Recipe for Aligning Subtitles to Signing
- **分类: cs.CL**

- **简介: 该论文研究手语视频与字幕的对齐任务，旨在解决现有方法语言和数据依赖性强、泛化能力差的问题。提出通用框架SEA，通过分段、嵌入和对齐三步实现跨语言、跨领域的高效对齐，无需端到端训练，生成高质量平行数据。**

- **链接: [https://arxiv.org/pdf/2512.08094v1](https://arxiv.org/pdf/2512.08094v1)**

> **作者:** Zifan Jiang; Youngjoon Jang; Liliane Momeni; Gül Varol; Sarah Ebling; Andrew Zisserman
>
> **摘要:** The goal of this work is to develop a universal approach for aligning subtitles (i.e., spoken language text with corresponding timestamps) to continuous sign language videos. Prior approaches typically rely on end-to-end training tied to a specific language or dataset, which limits their generality. In contrast, our method Segment, Embed, and Align (SEA) provides a single framework that works across multiple languages and domains. SEA leverages two pretrained models: the first to segment a video frame sequence into individual signs and the second to embed the video clip of each sign into a shared latent space with text. Alignment is subsequently performed with a lightweight dynamic programming procedure that runs efficiently on CPUs within a minute, even for hour-long episodes. SEA is flexible and can adapt to a wide range of scenarios, utilizing resources from small lexicons to large continuous corpora. Experiments on four sign language datasets demonstrate state-of-the-art alignment performance, highlighting the potential of SEA to generate high-quality parallel data for advancing sign language processing. SEA's code and models are openly available.
>
---
#### [new 006] Universal Adversarial Suffixes Using Calibrated Gumbel-Softmax Relaxation
- **分类: cs.CL**

- **简介: 该论文研究通用对抗后缀，旨在生成可迁移的文本攻击序列。通过Gumbel-Softmax松弛实现可微优化，结合校准交叉熵与熵正则化训练，避免标签泄漏与分布坍缩。所学后缀能跨任务、跨模型有效降低准确率与置信度。**

- **链接: [https://arxiv.org/pdf/2512.08123v1](https://arxiv.org/pdf/2512.08123v1)**

> **作者:** Sampriti Soor; Suklav Ghosh; Arijit Sur
>
> **备注:** 10 pages
>
> **摘要:** Language models (LMs) are often used as zero-shot or few-shot classifiers by scoring label words, but they remain fragile to adversarial prompts. Prior work typically optimizes task- or model-specific triggers, making results difficult to compare and limiting transferability. We study universal adversarial suffixes: short token sequences (4-10 tokens) that, when appended to any input, broadly reduce accuracy across tasks and models. Our approach learns the suffix in a differentiable "soft" form using Gumbel-Softmax relaxation and then discretizes it for inference. Training maximizes calibrated cross-entropy on the label region while masking gold tokens to prevent trivial leakage, with entropy regularization to avoid collapse. A single suffix trained on one model transfers effectively to others, consistently lowering both accuracy and calibrated confidence. Experiments on sentiment analysis, natural language inference, paraphrase detection, commonsense QA, and physical reasoning with Qwen2-1.5B, Phi-1.5, and TinyLlama-1.1B demonstrate consistent attack effectiveness and transfer across tasks and model families.
>
---
#### [new 007] Short-Context Dominance: How Much Local Context Natural Language Actually Needs?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究自然语言中短上下文主导现象，旨在量化预测所需最小上下文长度，并提出DaMCL指标检测依赖长上下文的序列。通过识别并增强长程相关token，缓解大模型对短上下文的偏好，提升问答等任务性能。**

- **链接: [https://arxiv.org/pdf/2512.08082v1](https://arxiv.org/pdf/2512.08082v1)**

> **作者:** Vala Vakilian; Zimeng Wang; Ankit Singh Rawat; Christos Thrampoulidis
>
> **备注:** 38 pages, 7 figures, includes appendix and references
>
> **摘要:** We investigate the short-context dominance hypothesis: that for most sequences, a small local prefix suffices to predict their next tokens. Using large language models as statistical oracles, we measure the minimum context length (MCL) needed to reproduce accurate full-context predictions across datasets with sequences of varying lengths. For sequences with 1-7k tokens from long-context documents, we consistently find that 75-80% require only the last 96 tokens at most. Given the dominance of short-context tokens, we then ask whether it is possible to detect challenging long-context sequences for which a short local prefix does not suffice for prediction. We introduce a practical proxy to MCL, called Distributionally Aware MCL (DaMCL), that does not require knowledge of the actual next-token and is compatible with sampling strategies beyond greedy decoding. Our experiments validate that simple thresholding of the metric defining DaMCL achieves high performance in detecting long vs. short context sequences. Finally, to counter the bias that short-context dominance induces in LLM output distributions, we develop an intuitive decoding algorithm that leverages our detector to identify and boost tokens that are long-range-relevant. Across Q&A tasks and model architectures, we confirm that mitigating the bias improves performance.
>
---
#### [new 008] Do Depth-Grown Models Overcome the Curse of Depth? An In-Depth Analysis
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究深度增长模型如何缓解Transformer中的“深度诅咒”问题。通过分析发现，渐进式深度增长能更有效地利用深层结构，形成可交换计算模块，并改进推理性能，进而提出一种轻量改进方法提升效果。**

- **链接: [https://arxiv.org/pdf/2512.08819v1](https://arxiv.org/pdf/2512.08819v1)**

> **作者:** Ferdinand Kapl; Emmanouil Angelis; Tobias Höppe; Kaitlin Maile; Johannes von Oswald; Nino Scherrer; Stefan Bauer
>
> **摘要:** Gradually growing the depth of Transformers during training can not only reduce training cost but also lead to improved reasoning performance, as shown by MIDAS (Saunshi et al., 2024). Thus far, however, a mechanistic understanding of these gains has been missing. In this work, we establish a connection to recent work showing that layers in the second half of non-grown, pre-layernorm Transformers contribute much less to the final output distribution than those in the first half - also known as the Curse of Depth (Sun et al., 2025, Csordás et al., 2025). Using depth-wise analyses, we demonstrate that growth via gradual middle stacking yields more effective utilization of model depth, alters the residual stream structure, and facilitates the formation of permutable computational blocks. In addition, we propose a lightweight modification of MIDAS that yields further improvements in downstream reasoning benchmarks. Overall, this work highlights how the gradual growth of model depth can lead to the formation of distinct computational circuits and overcome the limited depth utilization seen in standard non-grown models.
>
---
#### [new 009] QSTN: A Modular Framework for Robust Questionnaire Inference with Large Language Models
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出QSTN框架，属LLM在问卷推理中的应用任务。旨在解决问卷响应生成的可靠性与可复现性问题，通过模块化设计支持系统性评估，并提供低门槛实验工具，提升研究鲁棒性与效率。**

- **链接: [https://arxiv.org/pdf/2512.08646v1](https://arxiv.org/pdf/2512.08646v1)**

> **作者:** Maximilian Kreutner; Jens Rupprecht; Georg Ahnert; Ahmed Salem; Markus Strohmaier
>
> **备注:** The Python package is available at https://github.com/dess-mannheim/QSTN/
>
> **摘要:** We introduce QSTN, an open-source Python framework for systematically generating responses from questionnaire-style prompts to support in-silico surveys and annotation tasks with large language models (LLMs). QSTN enables robust evaluation of questionnaire presentation, prompt perturbations, and response generation methods. Our extensive evaluation ($>40 $ million survey responses) shows that question structure and response generation methods have a significant impact on the alignment of generated survey responses with human answers, and can be obtained for a fraction of the compute cost. In addition, we offer a no-code user interface that allows researchers to set up robust experiments with LLMs without coding knowledge. We hope that QSTN will support the reproducibility and reliability of LLM-based research in the future.
>
---
#### [new 010] Curriculum Guided Massive Multi Agent System Solving For Robust Long Horizon Tasks
- **分类: cs.CL; cs.AI; cs.CV; cs.MA**

- **简介: 该论文研究长时序复杂任务求解，提出一种分层多智能体系统，通过空间课程学习和NLL置信度机制，引导64×64轻量代理网格协同推理，降低计算开销，提升长视野任务的稳定性和准确性。**

- **链接: [https://arxiv.org/pdf/2512.08545v1](https://arxiv.org/pdf/2512.08545v1)**

> **作者:** Indrajit Kar; Kalathur Chenchu Kishore Kumar
>
> **备注:** 22 pages, 2 tables, 9 figures
>
> **摘要:** Large Language Models and multi-agent systems have shown promise in decomposing complex tasks, yet they struggle with long-horizon reasoning tasks and escalating computation cost. This work introduces a hierarchical multi-agent architecture that distributes reasoning across a 64*64 grid of lightweight agents, supported by a selective oracle. A spatial curriculum progressively expands the operational region of the grid, ensuring that agents master easier central tasks before tackling harder peripheral ones. To improve reliability, the system integrates Negative Log-Likelihood as a measure of confidence, allowing the curriculum to prioritize regions where agents are both accurate and well calibrated. A Thompson Sampling curriculum manager adaptively chooses training zones based on competence and NLL-driven reward signals. We evaluate the approach on a spatially grounded Tower of Hanoi benchmark, which mirrors the long-horizon structure of many robotic manipulation and planning tasks. Results demonstrate improved stability, reduced oracle usage, and stronger long-range reasoning from distributed agent cooperation.
>
---
#### [new 011] What Triggers my Model? Contrastive Explanations Inform Gender Choices by Translation Models
- **分类: cs.CL**

- **简介: 该论文属NLP中的模型可解释性任务，旨在探究翻译模型性别偏见的成因。通过对比解释与显著性归因，分析源句中触发性别选择的词汇，发现模型归因与人类感知存在重叠，进而揭示性别决策机制并提出缓解偏见的路径。**

- **链接: [https://arxiv.org/pdf/2512.08440v1](https://arxiv.org/pdf/2512.08440v1)**

> **作者:** Janiça Hackenbuchner; Arda Tezcan; Joke Daems
>
> **摘要:** Interpretability can be implemented as a means to understand decisions taken by (black box) models, such as machine translation (MT) or large language models (LLMs). Yet, research in this area has been limited in relation to a manifested problem in these models: gender bias. With this research, we aim to move away from simply measuring bias to exploring its origins. Working with gender-ambiguous natural source data, this study examines which context, in the form of input tokens in the source sentence, influences (or triggers) the translation model choice of a certain gender inflection in the target language. To analyse this, we use contrastive explanations and compute saliency attribution. We first address the challenge of a lacking scoring threshold and specifically examine different attribution levels of source words on the model gender decisions in the translation. We compare salient source words with human perceptions of gender and demonstrate a noticeable overlap between human perceptions and model attribution. Additionally, we provide a linguistic analysis of salient words. Our work showcases the relevance of understanding model translation decisions in terms of gender, how this compares to human decisions and that this information should be leveraged to mitigate gender bias.
>
---
#### [new 012] Soft Inductive Bias Approach via Explicit Reasoning Perspectives in Inappropriate Utterance Detection Using Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究不当言论检测任务，旨在解决现有大模型在推理过程中易出错的问题。提出一种软归纳偏置方法，通过显式定义推理视角引导模型决策，提升判断准确性。实验表明该方法优于标准监督学习。**

- **链接: [https://arxiv.org/pdf/2512.08480v1](https://arxiv.org/pdf/2512.08480v1)**

> **作者:** Ju-Young Kim; Ji-Hong Park; Se-Yeon Lee; Sujin Park; Gun-Woo Kim
>
> **备注:** in Chinese language, Published in the Proceedings of the 37th Annual Conference on Human and Language Technology, 2025, pp. 714-719. (English translation assisted by GPT)
>
> **摘要:** Recent incidents in certain online games and communities, where anonymity is guaranteed, show that unchecked inappropriate remarks frequently escalate into verbal abuse and even criminal behavior, raising significant social concerns. Consequently, there is a growing need for research on techniques that can detect inappropriate utterances within conversational texts to help build a safer communication environment. Although large-scale language models trained on Korean corpora and chain-of-thought reasoning have recently gained attention, research applying these approaches to inappropriate utterance detection remains limited. In this study, we propose a soft inductive bias approach that explicitly defines reasoning perspectives to guide the inference process, thereby promoting rational decision-making and preventing errors that may arise during reasoning. We fine-tune a Korean large language model using the proposed method and conduct both quantitative performance comparisons and qualitative evaluations across different training strategies. Experimental results show that the Kanana-1.5 model achieves an average accuracy of 87.0046, improving by approximately 3.89 percent over standard supervised learning. These findings indicate that the proposed method goes beyond simple knowledge imitation by large language models and enables more precise and consistent judgments through constrained reasoning perspectives, demonstrating its effectiveness for inappropriate utterance detection.
>
---
#### [new 013] Automatic Essay Scoring and Feedback Generation in Basque Language Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦巴斯克语自动作文评分与反馈生成，构建首个公开C1级数据集，标注多维度分数与错误示例。通过微调开源模型，实现评分一致性与反馈质量超越主流闭源系统，提出新评估方法，推动低资源语言NLP研究。**

- **链接: [https://arxiv.org/pdf/2512.08713v1](https://arxiv.org/pdf/2512.08713v1)**

> **作者:** Ekhi Azurmendi; Xabier Arregi; Oier Lopez de Lacalle
>
> **备注:** Submitted to LREC 2026
>
> **摘要:** This paper introduces the first publicly available dataset for Automatic Essay Scoring (AES) and feedback generation in Basque, targeting the CEFR C1 proficiency level. The dataset comprises 3,200 essays from HABE, each annotated by expert evaluators with criterion specific scores covering correctness, richness, coherence, cohesion, and task alignment enriched with detailed feedback and error examples. We fine-tune open-source models, including RoBERTa-EusCrawl and Latxa 8B/70B, for both scoring and explanation generation. Our experiments show that encoder models remain highly reliable for AES, while supervised fine-tuning (SFT) of Latxa significantly enhances performance, surpassing state-of-the-art (SoTA) closed-source systems such as GPT-5 and Claude Sonnet 4.5 in scoring consistency and feedback quality. We also propose a novel evaluation methodology for assessing feedback generation, combining automatic consistency metrics with expert-based validation of extracted learner errors. Results demonstrate that the fine-tuned Latxa model produces criterion-aligned, pedagogically meaningful feedback and identifies a wider range of error types than proprietary models. This resource and benchmark establish a foundation for transparent, reproducible, and educationally grounded NLP research in low-resource languages such as Basque.
>
---
#### [new 014] An Agentic AI System for Multi-Framework Communication Coding
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对临床医患对话编码任务，解决传统人工标注耗时费力及现有AI模型适应性差的问题。提出MOSAIC系统，基于LangGraph架构构建多智能体协同框架，实现跨多编码体系的自动对话分析，在两个临床领域测试中取得高F1分数，验证了其有效性与可靠性。**

- **链接: [https://arxiv.org/pdf/2512.08659v1](https://arxiv.org/pdf/2512.08659v1)**

> **作者:** Bohao Yang; Rui Yang; Joshua M. Biro; Haoyuan Wang; Jessica L. Handley; Brianna Richardson; Sophia Bessias; Nicoleta Economou-Zavlanos; Armando D. Bedoya; Monica Agrawal; Michael M. Zavlanos; Anand Chowdhury; Raj M. Ratwani; Kai Sun; Kathryn I. Pollak; Michael J. Pencina; Chuan Hong
>
> **摘要:** Clinical communication is central to patient outcomes, yet large-scale human annotation of patient-provider conversation remains labor-intensive, inconsistent, and difficult to scale. Existing approaches based on large language models typically rely on single-task models that lack adaptability, interpretability, and reliability, especially when applied across various communication frameworks and clinical domains. In this study, we developed a Multi-framework Structured Agentic AI system for Clinical Communication (MOSAIC), built on a LangGraph-based architecture that orchestrates four core agents, including a Plan Agent for codebook selection and workflow planning, an Update Agent for maintaining up-to-date retrieval databases, a set of Annotation Agents that applies codebook-guided retrieval-augmented generation (RAG) with dynamic few-shot prompting, and a Verification Agent that provides consistency checks and feedback. To evaluate performance, we compared MOSAIC outputs against gold-standard annotations created by trained human coders. We developed and evaluated MOSAIC using 26 gold standard annotated transcripts for training and 50 transcripts for testing, spanning rheumatology and OB/GYN domains. On the test set, MOSAIC achieved an overall F1 score of 0.928. Performance was highest in the Rheumatology subset (F1 = 0.962) and strongest for Patient Behavior (e.g., patients asking questions, expressing preferences, or showing assertiveness). Ablations revealed that MOSAIC outperforms baseline benchmarking.
>
---
#### [new 015] Fluent Alignment with Disfluent Judges: Post-training for Lower-resource Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究低资源语言的偏好对齐任务，旨在解决缺乏原生数据导致的语言模型流畅性下降问题。提出一种无需目标语言指令数据的在线策略后训练方法，在挪威语上验证其优于机器翻译和多语言微调方法。**

- **链接: [https://arxiv.org/pdf/2512.08777v1](https://arxiv.org/pdf/2512.08777v1)**

> **作者:** David Samuel; Lilja Øvrelid; Erik Velldal; Andrey Kutuzov
>
> **摘要:** We propose a post-training method for lower-resource languages that preserves fluency of language models even when aligned by disfluent reward models. Preference-optimization is now a well-researched topic, but previous work has mostly addressed models for English and Chinese. Lower-resource languages lack both datasets written by native speakers and language models capable of generating fluent synthetic data. Thus, in this work, we focus on developing a fluent preference-aligned language model without any instruction-tuning data in the target language. Our approach uses an on-policy training method, which we compare with two common approaches: supervised finetuning on machine-translated data and multilingual finetuning. We conduct a case study on Norwegian Bokmål and evaluate fluency through native-speaker assessments. The results show that the on-policy aspect is crucial and outperforms the alternatives without relying on any hard-to-obtain data.
>
---
#### [new 016] Are generative AI text annotations systematically biased?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究生成式AI文本标注是否存在系统性偏差，属自然语言处理中的偏见分析任务。通过复现人工标注实验，比较多种大模型在五种概念上的标注差异，发现模型虽F1分数尚可，但存在显著系统性偏差，影响下游结果。**

- **链接: [https://arxiv.org/pdf/2512.08404v1](https://arxiv.org/pdf/2512.08404v1)**

> **作者:** Sjoerd B. Stolwijk; Mark Boukes; Damian Trilling
>
> **备注:** 9 pages, 6 figures, 1 table; version submitted to the International Communication Association Annual Conference in Cape Town 2026
>
> **摘要:** This paper investigates bias in GLLM annotations by conceptually replicating manual annotations of Boukes (2024). Using various GLLMs (Llama3.1:8b, Llama3.3:70b, GPT4o, Qwen2.5:72b) in combination with five different prompts for five concepts (political content, interactivity, rationality, incivility, and ideology). We find GLLMs perform adequate in terms of F1 scores, but differ from manual annotations in terms of prevalence, yield substantively different downstream results, and display systematic bias in that they overlap more with each other than with manual annotations. Differences in F1 scores fail to account for the degree of bias.
>
---
#### [new 017] ClinicalTrialsHub: Bridging Registries and Literature for Comprehensive Clinical Trial Access
- **分类: cs.CL; cs.AI; cs.HC; cs.IR**

- **简介: 该论文提出ClinicalTrialsHub，旨在整合ClinicalTrials.gov数据并从PubMed文献中自动提取结构化临床试验信息。利用大语言模型提升检索与问答能力，解决临床试验数据分散、访问不足的问题，增强数据可及性，支持循证医学。**

- **链接: [https://arxiv.org/pdf/2512.08193v1](https://arxiv.org/pdf/2512.08193v1)**

> **作者:** Jiwoo Park; Ruoqi Liu; Avani Jagdale; Andrew Srisuwananukorn; Jing Zhao; Lang Li; Ping Zhang; Sachin Kumar
>
> **摘要:** We present ClinicalTrialsHub, an interactive search-focused platform that consolidates all data from ClinicalTrials.gov and augments it by automatically extracting and structuring trial-relevant information from PubMed research articles. Our system effectively increases access to structured clinical trial data by 83.8% compared to relying on ClinicalTrials.gov alone, with potential to make access easier for patients, clinicians, researchers, and policymakers, advancing evidence-based medicine. ClinicalTrialsHub uses large language models such as GPT-5.1 and Gemini-3-Pro to enhance accessibility. The platform automatically parses full-text research articles to extract structured trial information, translates user queries into structured database searches, and provides an attributed question-answering system that generates evidence-grounded answers linked to specific source sentences. We demonstrate its utility through a user study involving clinicians, clinical researchers, and PhD students of pharmaceutical sciences and nursing, and a systematic automatic evaluation of its information extraction and question answering capabilities.
>
---
#### [new 018] Adaptation of Embedding Models to Financial Filings via LLM Distillation
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在提升金融领域文档检索效果。针对现有嵌入模型在专业领域表现不足的问题，提出一种基于大模型蒸馏的迭代训练框架，利用无标注语料自动挖掘难例并优化检索模型，显著提升金融 filings 的检索性能。**

- **链接: [https://arxiv.org/pdf/2512.08088v1](https://arxiv.org/pdf/2512.08088v1)**

> **作者:** Eliot Brenner; Dominic Seyler; Manjunath Hegde; Andrei Simion; Koustuv Dasgupta; Bing Xiang
>
> **备注:** In proceedings of LLM-Finance 2025 : The 2nd IEEE International Workshop on Large Language Models for Finance
>
> **摘要:** Despite advances in generative large language models (LLMs), practical application of specialized conversational AI agents remains constrained by computation costs, latency requirements, and the need for precise domain-specific relevance measures. While existing embedding models address the first two constraints, they underperform on information retrieval in specialized domains like finance. This paper introduces a scalable pipeline that trains specialized models from an unlabeled corpus using a general purpose retrieval embedding model as foundation. Our method yields an average of 27.7% improvement in MRR$\texttt{@}$5, 44.6% improvement in mean DCG$\texttt{@}$5 across 14 financial filing types measured over 21,800 query-document pairs, and improved NDCG on 3 of 4 document classes in FinanceBench. We adapt retrieval embeddings (bi-encoder) for RAG, not LLM generators, using LLM-judged relevance to distill domain knowledge into a compact retriever. There are prior works which pair synthetically generated queries with real passages to directly fine-tune the retrieval model. Our pipeline differs from these by introducing interaction between student and teacher models that interleaves retrieval-based mining of hard positive/negative examples from the unlabeled corpus with iterative retraining of the student model's weights using these examples. Each retrieval iteration uses the refined student model to mine the corpus for progressively harder training examples for the subsequent training iteration. The methodology provides a cost-effective solution to bridging the gap between general-purpose models and specialized domains without requiring labor-intensive human annotation.
>
---
#### [new 019] A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究联邦学习下大语言模型的多元对齐问题，提出一种评估框架与自适应聚合方法，通过动态调整各群体偏好权重，在保护数据隐私的同时提升模型公平性与对齐质量。**

- **链接: [https://arxiv.org/pdf/2512.08786v1](https://arxiv.org/pdf/2512.08786v1)**

> **作者:** Mahmoud Srewa; Tianyu Zhao; Salma Elmalaki
>
> **摘要:** This paper addresses the challenge of aligning large language models (LLMs) with diverse human preferences within federated learning (FL) environments, where standard methods often fail to adequately represent diverse viewpoints. We introduce a comprehensive evaluation framework that systematically assesses the trade-off between alignment quality and fairness when using different aggregation strategies for human preferences. In our federated setting, each group locally evaluates rollouts and produces reward signals, and the server aggregates these group-level rewards without accessing any raw data. Specifically, we evaluate standard reward aggregation techniques (min, max, and average) and introduce a novel adaptive scheme that dynamically adjusts preference weights based on a group's historical alignment performance. Our experiments on question-answering (Q/A) tasks using a PPO-based RLHF pipeline demonstrate that our adaptive approach consistently achieves superior fairness while maintaining competitive alignment scores. This work offers a robust methodology for evaluating LLM behavior across diverse populations and provides a practical solution for developing truly pluralistic and fairly aligned models.
>
---
#### [new 020] Revisiting the Scaling Properties of Downstream Metrics in Large Language Model Training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型训练中下游任务性能的缩放规律，旨在直接建模从训练预算到下游性能的关系。提出简单幂律可准确预测固定token/参数比下的准确率，并验证了跨不同规模设置的有效性。**

- **链接: [https://arxiv.org/pdf/2512.08894v1](https://arxiv.org/pdf/2512.08894v1)**

> **作者:** Jakub Krajewski; Amitis Shidani; Dan Busbridge; Sam Wiseman; Jason Ramapuram
>
> **摘要:** While scaling laws for Large Language Models (LLMs) traditionally focus on proxy metrics like pretraining loss, predicting downstream task performance has been considered unreliable. This paper challenges that view by proposing a direct framework to model the scaling of benchmark performance from the training budget. We find that for a fixed token-to-parameter ratio, a simple power law can accurately describe the scaling behavior of log accuracy on multiple popular downstream tasks. Our results show that the direct approach extrapolates better than the previously proposed two-stage procedure, which is prone to compounding errors. Furthermore, we introduce functional forms that predict accuracy across token-to-parameter ratios and account for inference compute under repeated sampling. We validate our findings on models with up to 17B parameters trained on up to 350B tokens across two dataset mixtures. To support reproducibility and encourage future research, we release the complete set of pretraining losses and downstream evaluation results.
>
---
#### [new 021] Pose-Based Sign Language Spotting via an End-to-End Encoder Architecture
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出“手语检测”新任务，旨在从连续手语中识别特定手势是否存在。作者构建端到端编码器模型，直接基于姿态关键点进行二分类判断，避免依赖中间文本标注，降低计算成本与视觉噪声，验证了姿态表示在手语检索中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.08738v1](https://arxiv.org/pdf/2512.08738v1)**

> **作者:** Samuel Ebimobowei Johnny; Blessed Guda; Emmanuel Enejo Aaron; Assane Gueye
>
> **备注:** To appear at AACL-IJCNLP 2025 Workshop WSLP
>
> **摘要:** Automatic Sign Language Recognition (ASLR) has emerged as a vital field for bridging the gap between deaf and hearing communities. However, the problem of sign-to-sign retrieval or detecting a specific sign within a sequence of continuous signs remains largely unexplored. We define this novel task as Sign Language Spotting. In this paper, we present a first step toward sign language retrieval by addressing the challenge of detecting the presence or absence of a query sign video within a sentence-level gloss or sign video. Unlike conventional approaches that rely on intermediate gloss recognition or text-based matching, we propose an end-to-end model that directly operates on pose keypoints extracted from sign videos. Our architecture employs an encoder-only backbone with a binary classification head to determine whether the query sign appears within the target sequence. By focusing on pose representations instead of raw RGB frames, our method significantly reduces computational cost and mitigates visual noise. We evaluate our approach on the Word Presence Prediction dataset from the WSLP 2025 shared task, achieving 61.88\% accuracy and 60.00\% F1-score. These results demonstrate the effectiveness of our pose-based framework for Sign Language Spotting, establishing a strong foundation for future research in automatic sign language retrieval and verification. Code is available at https://github.com/EbimoJohnny/Pose-Based-Sign-Language-Spotting
>
---
#### [new 022] MixLM: High-Throughput and Effective LLM Ranking via Text-Embedding Mix-Interaction
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于信息检索中的排序任务，旨在解决LLM在工业场景下推理开销大、吞吐低的问题。提出MixLM框架，通过文本与嵌入混合输入的方式缩短上下文长度，在保持排序效果的同时提升10倍吞吐，实现高效在线部署。**

- **链接: [https://arxiv.org/pdf/2512.07846v1](https://arxiv.org/pdf/2512.07846v1)**

> **作者:** Guoyao Li; Ran He; Shusen Jing; Kayhan Behdin; Yubo Wang; Sundara Raman Ramachandran; Chanh Nguyen; Jian Sheng; Xiaojing Ma; Chuanrui Zhu; Sriram Vasudevan; Muchen Wu; Sayan Ghosh; Lin Su; Qingquan Song; Xiaoqing Wang; Zhipeng Wang; Qing Lan; Yanning Chen; Jingwei Wu; Luke Simon; Wenjing Zhang; Qi Guo; Fedor Borisyuk
>
> **摘要:** Large language models (LLMs) excel at capturing semantic nuances and therefore show impressive relevance ranking performance in modern recommendation and search systems. However, they suffer from high computational overhead under industrial latency and throughput requirements. In particular, cross-encoder ranking systems often create long context prefill-heavy workloads, as the model has to be presented with the user, query and item information. To this end, we propose MixLM, a novel LLM-based ranking framework, which significantly improves the system throughput via reducing the input context length, while preserving the semantic strength of cross-encoder rankers. In contrast to a standard ranking system where the context is presented to the model as pure text, we propose to use mix-interaction, a mixture of text and embedding tokens to represent the input. Specifically, MixLM encodes all items in the catalog into a few embedding tokens and stores in a nearline cache. The encoded item descriptions are used during online inference, effectively reducing the item length from a few thousand text tokens to a few embedding tokens. We share insights from deploying our MixLM framework to a real-world search application at LinkedIn, including a detailed discussion of our training pipelines, as well as a thorough analysis of our online serving infrastructure optimization. Comparing with strong baselines, MixLM increased throughput by 10.0x under the same latency budget, while maintaining relevance metrics. The efficiency gains delivered by MixLM enabled full-traffic deployment of LLM-powered search, which resulted in a significant 0.47% increase in Daily Active Users (DAU) in online A/B tests.
>
---
#### [new 023] Reasoning Models Ace the CFA Exams
- **分类: cs.AI; cs.CL; q-fin.GN**

- **简介: 该论文评估先进推理模型在CFA三级考试中的表现，解决此前LLMs在此类专业考试中表现不佳的问题。作者使用980道模拟题测试多个模型，发现多数模型通过全部级别，其中Gemini系列表现最佳。**

- **链接: [https://arxiv.org/pdf/2512.08270v1](https://arxiv.org/pdf/2512.08270v1)**

> **作者:** Jaisal Patel; Yunzhe Chen; Kaiwen He; Keyi Wang; David Li; Kairong Xiao; Xiao-Yang Liu
>
> **摘要:** Previous research has reported that large language models (LLMs) demonstrate poor performance on the Chartered Financial Analyst (CFA) exams. However, recent reasoning models have achieved strong results on graduate-level academic and professional examinations across various disciplines. In this paper, we evaluate state-of-the-art reasoning models on a set of mock CFA exams consisting of 980 questions across three Level I exams, two Level II exams, and three Level III exams. Using the same pass/fail criteria from prior studies, we find that most models clear all three levels. The models that pass, ordered by overall performance, are Gemini 3.0 Pro, Gemini 2.5 Pro, GPT-5, Grok 4, Claude Opus 4.1, and DeepSeek-V3.1. Specifically, Gemini 3.0 Pro achieves a record score of 97.6% on Level I. Performance is also strong on Level II, led by GPT-5 at 94.3%. On Level III, Gemini 2.5 Pro attains the highest score with 86.4% on multiple-choice questions while Gemini 3.0 Pro achieves 92.0% on constructed-response questions.
>
---
#### [new 024] Accelerating Urban Science Research with AI Urban Scientist
- **分类: cs.CY; cs.CL; cs.MA**

- **简介: 该论文提出AI城市科学家，旨在解决城市科学中数据碎片化与领域知识不足的难题。通过构建基于多智能体的知识驱动框架，实现从假设生成到实证分析的全流程自动化，推动城市系统机制的解析与可持续城市发展。**

- **链接: [https://arxiv.org/pdf/2512.07849v1](https://arxiv.org/pdf/2512.07849v1)**

> **作者:** Tong Xia; Jiankun Zhang; Ruiwen You; Ao Xu; Linghao Zhang; Tengyao Tu; Jingzhi Wang; Jinghua Piao; Yunke Zhang; Fengli Xu; Yong Li
>
> **摘要:** Cities are complex, adaptive systems whose underlying principles remain difficult to disentangle despite unprecedented data abundance. Urban science therefore faces a fundamental challenge: converting vast, fragmented and interdisciplinary information into coherent explanations of how cities function and evolve. The emergence of AI scientists, i.e., agents capable of autonomous reasoning, hypothesis formation and data-driven experimentation, offers a new pathway toward accelerating this transformation, yet general-purpose systems fall short of the domain knowledge and methodological depth required for urban science research. Here we introduce a knowledge-driven AI Urban Scientist, built from hypotheses, peer-review signals, datasets and analytical patterns distilled from thousands of high-quality studies, and implemented as a coordinated multi-agent framework for end-to-end inquiry. The system generates structured hypotheses, retrieves and harmonizes heterogeneous datasets, conducts automated empirical analysis and simulation, and synthesizes insights in forms compatible with urban scientific reasoning. By providing reusable analytical tools and supporting community-driven extensions, the AI Urban Scientist lowers barriers to advanced urban analytics and acts not merely as an assistant but as an active collaborator in revealing the mechanisms that shape urban systems and in guiding the design of more resilient and equitable cities.
>
---
#### [new 025] The High Cost of Incivility: Quantifying Interaction Inefficiency via Multi-Agent Monte Carlo Simulations
- **分类: cs.AI; cs.CL; cs.CY; cs.MA**

- **简介: 该论文属计算社会科学任务，旨在量化职场无礼行为对沟通效率的影响。通过LLM多智能体蒙特卡洛模拟，比较有毒与非有毒对话的收敛时间，发现毒性使讨论延长约25%，提出“毒性延迟”作为效率损失代理指标。**

- **链接: [https://arxiv.org/pdf/2512.08345v1](https://arxiv.org/pdf/2512.08345v1)**

> **作者:** Benedikt Mangold
>
> **备注:** 8 figures, 3 tables
>
> **摘要:** Workplace toxicity is widely recognized as detrimental to organizational culture, yet quantifying its direct impact on operational efficiency remains methodologically challenging due to the ethical and practical difficulties of reproducing conflict in human subjects. This study leverages Large Language Model (LLM) based Multi-Agent Systems to simulate 1-on-1 adversarial debates, creating a controlled "sociological sandbox". We employ a Monte Carlo method to simulate hundrets of discussions, measuring the convergence time (defined as the number of arguments required to reach a conclusion) between a baseline control group and treatment groups involving agents with "toxic" system prompts. Our results demonstrate a statistically significant increase of approximately 25\% in the duration of conversations involving toxic participants. We propose that this "latency of toxicity" serves as a proxy for financial damage in corporate and academic settings. Furthermore, we demonstrate that agent-based modeling provides a reproducible, ethical alternative to human-subject research for measuring the mechanics of social friction.
>
---
#### [new 026] Beyond Real Weights: Hypercomplex Representations for Stable Quantization
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究多模态语言模型的高效部署问题，提出用渐进式超复数乘法（PHM）层替代密集前馈网络，结合残差插值与知识蒸馏，实现模型压缩与加速，保持性能的同时显著降低参数量与推理延迟。**

- **链接: [https://arxiv.org/pdf/2512.08524v1](https://arxiv.org/pdf/2512.08524v1)**

> **作者:** Jawad Ibn Ahad; Maisha Rahman; Amrijit Biswas; Muhammad Rafsan Kabir; Robin Krambroeckers; Sifat Momen; Nabeel Mohammed; Shafin Rahman
>
> **备注:** Accepted in Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** Multimodal language models (MLLMs) require large parameter capacity to align high-dimensional visual features with linguistic representations, making them computationally heavy and difficult to deploy efficiently. We introduce a progressive reparameterization strategy that compresses these models by gradually replacing dense feed-forward network blocks with compact Parameterized Hypercomplex Multiplication (PHM) layers. A residual interpolation schedule, together with lightweight reconstruction and knowledge distillation losses, ensures that the PHM modules inherit the functional behavior of their dense counterparts during training. This transition yields substantial parameter and FLOP reductions while preserving strong multimodal alignment, enabling faster inference without degrading output quality. We evaluate the approach on multiple vision-language models (VLMs). Our method maintains performance comparable to the base models while delivering significant reductions in model size and inference latency. Progressive PHM substitution thus offers an architecture-compatible path toward more efficient multimodal reasoning and complements existing low-bit quantization techniques.
>
---
#### [new 027] Beyond Unified Models: A Service-Oriented Approach to Low Latency, Context Aware Phonemization for Real Time TTS
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决实时TTS中音素转换质量与速度的权衡问题。提出一种面向服务的轻量级、上下文感知音素化框架，将复杂模块解耦为独立服务，在保证低延迟的同时提升发音准确性，适用于端侧离线应用。**

- **链接: [https://arxiv.org/pdf/2512.08006v1](https://arxiv.org/pdf/2512.08006v1)**

> **作者:** Mahta Fetrat; Donya Navabi; Zahra Dehghanian; Morteza Abolghasemi; Hamid R. Rabiee
>
> **摘要:** Lightweight, real-time text-to-speech systems are crucial for accessibility. However, the most efficient TTS models often rely on lightweight phonemizers that struggle with context-dependent challenges. In contrast, more advanced phonemizers with a deeper linguistic understanding typically incur high computational costs, which prevents real-time performance. This paper examines the trade-off between phonemization quality and inference speed in G2P-aided TTS systems, introducing a practical framework to bridge this gap. We propose lightweight strategies for context-aware phonemization and a service-oriented TTS architecture that executes these modules as independent services. This design decouples heavy context-aware components from the core TTS engine, effectively breaking the latency barrier and enabling real-time use of high-quality phonemization models. Experimental results confirm that the proposed system improves pronunciation soundness and linguistic accuracy while maintaining real-time responsiveness, making it well-suited for offline and end-device TTS applications.
>
---
#### [new 028] Ontology-Based Knowledge Graph Framework for Industrial Standard Documents via Hierarchical and Propositional Structuring
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出一种基于本体的知识图谱框架，解决工业标准文档中复杂结构与语义难以建模的问题。通过分层与命题化结构，结合LLM三元组抽取，构建支持多跳问答与毒条款检测的KG-RAG系统，显著提升问答性能。**

- **链接: [https://arxiv.org/pdf/2512.08398v1](https://arxiv.org/pdf/2512.08398v1)**

> **作者:** Jiin Park; Hyuna Jeon; Yoonseo Lee; Jisu Hong; Misuk Kim
>
> **摘要:** Ontology-based knowledge graph (KG) construction is a core technology that enables multidimensional understanding and advanced reasoning over domain knowledge. Industrial standards, in particular, contain extensive technical information and complex rules presented in highly structured formats that combine tables, scopes of application, constraints, exceptions, and numerical calculations, making KG construction especially challenging. In this study, we propose a method that organizes such documents into a hierarchical semantic structure, decomposes sentences and tables into atomic propositions derived from conditional and numerical rules, and integrates them into an ontology-knowledge graph through LLM-based triple extraction. Our approach captures both the hierarchical and logical structures of documents, effectively representing domain-specific semantics that conventional methods fail to reflect. To verify its effectiveness, we constructed rule, table, and multi-hop QA datasets, as well as a toxic clause detection dataset, from industrial standards, and implemented an ontology-aware KG-RAG framework for comparative evaluation. Experimental results show that our method achieves significant performance improvements across all QA types compared to existing KG-RAG approaches. This study demonstrates that reliable and scalable knowledge representation is feasible even for industrial documents with intertwined conditions, constraints, and scopes, contributing to future domain-specific RAG development and intelligent document management.
>
---
#### [new 029] ThreadWeaver: Adaptive Threading for Efficient Parallel Reasoning in Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属高效推理任务，旨在解决大模型并行推理中精度下降与部署复杂的问题。提出ThreadWeaver框架，通过并行轨迹生成、trie结构设计与强化学习，实现媲美串行模型的精度与显著加速。**

- **链接: [https://arxiv.org/pdf/2512.07843v1](https://arxiv.org/pdf/2512.07843v1)**

> **作者:** Long Lian; Sida Wang; Felix Juefei-Xu; Tsu-Jui Fu; Xiuyu Li; Adam Yala; Trevor Darrell; Alane Suhr; Yuandong Tian; Xi Victoria Lin
>
> **摘要:** Scaling inference-time computation has enabled Large Language Models (LLMs) to achieve strong reasoning performance, but inherently sequential decoding leads to substantial latency, especially on complex tasks. Recent work on adaptive parallel reasoning aims to improve inference efficiency by decomposing the problem-solving process into concurrent reasoning threads when beneficial. However, existing methods on realistic tasks are either limited to supervised behavior cloning or exhibit significant accuracy drops compared to widely-used sequential long chain-of-thought (CoT) baselines. Moreover, many require customized inference engines, complicating deployment. We introduce ThreadWeaver, a framework for adaptive parallel reasoning that achieves accuracy on par with popular sequential reasoning models of comparable size while significantly reducing inference latency. ThreadWeaver's performance stems from three key innovations: 1) a two-stage parallel trajectory generator that produces large-scale, high-quality CoT data with parallel annotations for supervised fine-tuning; 2) a trie-based training-inference co-design that enables parallel reasoning on any off-the-shelf autoregressive inference engine without modifying position embeddings or KV caches; and 3) a parallelization-aware reinforcement learning framework that teaches the model to balance accuracy with effective parallelization. Across six challenging mathematical reasoning benchmarks, ThreadWeaver trained atop Qwen3-8B achieves accuracy comparable to cutting-edge sequential reasoning models (71.9% on average and 79.9% on AIME24) while delivering up to 1.53x average speedup in token latency, establishing a new Pareto frontier between accuracy and efficiency.
>
---
#### [new 030] Balanced Accuracy: The Right Metric for Evaluating LLM Judges -- Explained through Youden's J statistic
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决因类别不平衡导致的LLM裁判评估偏差问题。提出使用Youden's J统计量等价的平衡准确率（Balanced Accuracy）作为更可靠的指标，通过理论与实证证明其在裁判选择中的优越性。**

- **链接: [https://arxiv.org/pdf/2512.08121v1](https://arxiv.org/pdf/2512.08121v1)**

> **作者:** Stephane Collot; Colin Fraser; Justin Zhao; William F. Shen; Timon Willi; Ilias Leontiadis
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Rigorous evaluation of large language models (LLMs) relies on comparing models by the prevalence of desirable or undesirable behaviors, such as task pass rates or policy violations. These prevalence estimates are produced by a classifier, either an LLM-as-a-judge or human annotators, making the choice of classifier central to trustworthy evaluation. Common metrics used for this choice, such as Accuracy, Precision, and F1, are sensitive to class imbalance and to arbitrary choices of positive class, and can favor judges that distort prevalence estimates. We show that Youden's $J$ statistic is theoretically aligned with choosing the best judge to compare models, and that Balanced Accuracy is an equivalent linear transformation of $J$. Through both analytical arguments and empirical examples and simulations, we demonstrate how selecting judges using Balanced Accuracy leads to better, more robust classifier selection.
>
---
## 更新

#### [replaced 001] Shrinking the Generation-Verification Gap with Weak Verifiers
- **分类: cs.CL**

- **简介: 该论文属语言模型验证任务，旨在缩小强验证器与弱验证器间的性能差距。提出Weaver框架，利用弱监督融合多个弱验证器，提升响应选择准确性，并降低对标注数据和计算成本的依赖。**

- **链接: [https://arxiv.org/pdf/2506.18203v2](https://arxiv.org/pdf/2506.18203v2)**

> **作者:** Jon Saad-Falcon; E. Kelly Buchanan; Mayee F. Chen; Tzu-Heng Huang; Brendan McLaughlin; Tanvir Bhathal; Shang Zhu; Ben Athiwaratkun; Frederic Sala; Scott Linderman; Azalia Mirhoseini; Christopher Ré
>
> **备注:** Annual Conference on Neural Information Processing Systems (NeurIPS) 2025
>
> **摘要:** Verifiers can improve language model capabilities by scoring and ranking responses from generated candidates. Currently, high-quality verifiers are either unscalable (e.g., humans) or limited in utility (e.g., tools like Lean). While LM judges and reward models have become broadly useful as general-purpose verifiers, a significant performance gap remains between them and oracle verifiers (verifiers with perfect accuracy). To help close this gap, we introduce Weaver, a framework for designing a strong verifier by combining multiple weak, imperfect verifiers. We find weighted ensembles of verifiers, which typically require learning from labeled data, significantly outperform unweighted combinations due to differences in verifier accuracies. To reduce dependency on labeled data, Weaver leverages weak supervision to estimate each verifier's accuracy and combines outputs into a unified score that better reflects true response quality. However, directly applying weak supervision algorithms poses challenges, including inconsistent verifier output formats and handling low-quality verifiers. Weaver addresses these using dataset statistics to normalize outputs and filter specific verifiers. We study Weaver's effectiveness in test-time repeated sampling, where a model generates multiple candidate responses and selects one. Our evaluations show Weaver significantly improves over Pass@1-performance when selecting the first candidate-across reasoning and math tasks, achieving o3-mini-level accuracy with Llama 3.3 70B Instruct as generator, and an ensemble of 70B or smaller judge and reward models as verifiers (87.7% average). This gain mirrors the jump between GPT-4o and o3-mini (69.0% vs. 86.7%), which required extensive finetuning and post-training. To reduce computational costs of verifier ensembles, we train a 400M cross-encoder using Weaver's combined output scores.
>
---
#### [replaced 002] Fine-grained Spatiotemporal Grounding on Egocentric Videos
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究第一人称视频中的细粒度时空定位任务，旨在解决现有方法在该场景下性能差的问题。作者分析了与第三人称视频的差异，构建了首个像素级基准EgoMask及大规模训练集EgoMask-Train，并验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2508.00518v2](https://arxiv.org/pdf/2508.00518v2)**

> **作者:** Shuo Liang; Yiwu Zhong; Zi-Yuan Hu; Yeyao Tao; Liwei Wang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Spatiotemporal video grounding aims to localize target entities in videos based on textual queries. While existing research has made significant progress in exocentric videos, the egocentric setting remains relatively underexplored, despite its growing importance in applications such as augmented reality and robotics. In this work, we conduct a systematic analysis of the discrepancies between egocentric and exocentric videos, revealing key challenges such as shorter object durations, sparser trajectories, smaller object sizes, and larger positional shifts. To address these challenges, we introduce EgoMask, the first pixel-level benchmark for fine-grained spatiotemporal grounding in egocentric videos. It is constructed by our proposed automatic annotation pipeline, which annotates referring expressions and object masks across short-, medium-, and long-term videos. Additionally, we create EgoMask-Train, a large-scale training dataset to facilitate model development. Experiments demonstrate that the state-of-the-art spatiotemporal grounding models perform poorly on our benchmark EgoMask, but fine-tuning on EgoMask-Train yields significant improvements, while preserving performance on exocentric datasets. Our work thus provides essential resources and insights for advancing egocentric video understanding. Our code is available at https://github.com/LaVi-Lab/EgoMask .
>
---
#### [replaced 003] The AI Consumer Index (ACE)
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出AI消费者指数（ACE），旨在评估前沿AI模型执行日常消费任务的能力。通过构建400个测试用例，覆盖购物、饮食、游戏和DIY四类活动，揭示当前模型在真实性与实用性上与用户需求存在显著差距。**

- **链接: [https://arxiv.org/pdf/2512.04921v3](https://arxiv.org/pdf/2512.04921v3)**

> **作者:** Julien Benchek; Rohit Shetty; Benjamin Hunsberger; Ajay Arun; Zach Richards; Brendan Foody; Osvald Nitski; Bertie Vidgen
>
> **摘要:** We introduce the first version of the AI Consumer Index (ACE), a benchmark for assessing whether frontier AI models can perform everyday consumer tasks. ACE contains a hidden heldout set of 400 test cases, split across four consumer activities: shopping, food, gaming, and DIY. We are also open sourcing 80 cases as a devset with a CC-BY license. For the ACE leaderboard we evaluated 10 frontier models (with websearch turned on) using a novel grading methodology that dynamically checks whether relevant parts of the response are grounded in the retrieved web sources. GPT 5 (Thinking = High) is the top-performing model, scoring 56.1%, followed by o3 Pro (Thinking = On) at 55.2% and GPT 5.1 (Thinking = High) at 55.1%. Model scores differ across domains, and in Shopping the top model scores under 50\%. We find that models are prone to hallucinating key information, such as prices. ACE shows a substantial gap between the performance of even the best models and consumers' AI needs.
>
---
#### [replaced 004] Understanding LLM Reasoning for Abstractive Summarization
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在抽象摘要任务中推理能力的有效性，探究不同推理策略对摘要质量与事实忠实度的影响，发现推理并非总有益，且存在质量与忠实度的权衡。**

- **链接: [https://arxiv.org/pdf/2512.03503v2](https://arxiv.org/pdf/2512.03503v2)**

> **作者:** Haohan Yuan; Haopeng Zhang
>
> **备注:** 26 pages,15 figures
>
> **摘要:** While the reasoning capabilities of Large Language Models (LLMs) excel in analytical tasks such as mathematics and code generation, their utility for abstractive summarization remains widely assumed but largely unverified. To bridge this gap, we first tailor general reasoning strategies to the summarization domain. We then conduct a systematic, large scale comparative study of 8 reasoning strategies and 3 Large Reasoning Models (LRMs) across 8 diverse datasets, assessing both summary quality and faithfulness. Our findings show that reasoning is not a universal solution and its effectiveness is highly dependent on the specific strategy and context. Specifically, we observe a trade-off between summary quality and factual faithfulness: explicit reasoning strategies tend to improve fluency at the expense of factual grounding, while implicit reasoning in LRMs exhibits the inverse pattern. Furthermore, increasing an LRM's internal reasoning budget does not improve, and can even hurt, factual consistency, suggesting that effective summarization demands faithful compression rather than creative over-thinking.
>
---
#### [replaced 005] B-cos LM: Efficiently Transforming Pre-trained Language Models for Improved Explainability
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升模型的可解释性。针对现有语言模型解释性差的问题，提出将预训练语言模型转化为B-cos LM，通过移除偏置项、对齐输入与权重，实现更忠实、易理解的解释，同时保持原有性能。**

- **链接: [https://arxiv.org/pdf/2502.12992v4](https://arxiv.org/pdf/2502.12992v4)**

> **作者:** Yifan Wang; Sukrut Rao; Ji-Ung Lee; Mayank Jobanputra; Vera Demberg
>
> **备注:** TMLR 12/2025
>
> **摘要:** Post-hoc explanation methods for black-box models often struggle with faithfulness and human interpretability due to the lack of explainability in current neural architectures. Meanwhile, B-cos networks have been introduced to improve model explainability by proposing an architecture that removes bias terms and promotes input-weight alignment. Although B-cos networks have shown success in building explainable systems, their application has so far been limited to computer vision models and their associated training pipelines. In this work, we introduce B-cos LMs, i.e., B-cos Language Models (LMs) empowered for natural language processing (NLP) tasks. Our approach directly transforms pre-trained language models into B-cos LMs by combining B-cos conversion and task fine-tuning, improving efficiency compared to previous methods. Automatic and human evaluation results demonstrate that B-cos LMs produce more faithful and human interpretable explanations than post-hoc methods, while maintaining task performance comparable to conventional fine-tuning. Our in-depth analysis explores how B-cos LMs differ from conventionally fine-tuned models in their learning processes and explanation patterns. Finally, we present a first exploration of transforming decoder-only models to B-cos LMs for generation tasks. Our code is available at https://github.com/Ewanwong/bcos_lm.
>
---
#### [replaced 006] SynBullying: A Multi LLM Synthetic Conversational Dataset for Cyberbullying Detection
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于网络安全与NLP交叉任务，旨在解决网络欺凌检测中真实数据稀缺与伦理问题。作者构建了SynBullying——一个基于多LLM生成的对话式合成数据集，具有上下文感知标注和细粒度分类，用于训练和增强欺凌检测模型。**

- **链接: [https://arxiv.org/pdf/2511.11599v2](https://arxiv.org/pdf/2511.11599v2)**

> **作者:** Arefeh Kazemi; Hamza Qadeer; Joachim Wagner; Hossein Hosseini; Sri Balaaji Natarajan Kalaivendan; Brian Davis
>
> **摘要:** We introduce SynBullying, a synthetic multi-LLM conversational dataset for studying and detecting cyberbullying (CB). SynBullying provides a scalable and ethically safe alternative to human data collection by leveraging large language models (LLMs) to simulate realistic bullying interactions. The dataset offers (i) conversational structure, capturing multi-turn exchanges rather than isolated posts; (ii) context-aware annotations, where harmfulness is assessed within the conversational flow considering context, intent, and discourse dynamics; and (iii) fine-grained labeling, covering various CB categories for detailed linguistic and behavioral analysis. We evaluate SynBullying across five dimensions, including conversational structure, lexical patterns, sentiment/toxicity, role dynamics, harm intensity, and CB-type distribution. We further examine its utility by testing its performance as standalone training data and as an augmentation source for CB classification.
>
---
#### [replaced 007] Mortgage Language Model: Domain-Adaptive Pretraining with Residual Instruction, Alignment Tuning, and Task-Specific Routing
- **分类: cs.CL; cs.LG**

- **简介: 该论文聚焦金融领域中的房贷任务，解决通用大模型在专业领域知识不足与多任务性能冲突的问题。提出双专家架构，结合指令残差与智能路由，分别优化问答与结构化任务，提升领域适应性与任务表现。**

- **链接: [https://arxiv.org/pdf/2511.21101v2](https://arxiv.org/pdf/2511.21101v2)**

> **作者:** Manish Jain; Satheesh Kumar Ponnambalam; Salman Faroz; Chandrakanth Lns; Vinay Sharma
>
> **摘要:** Large Language Models (LLMs) demonstrate exceptional capabilities across general domains, yet their application to specialized sectors such as mortgage finance requires domain-specific knowledge augmentation while preserving instruction-following fidelity. We present MortgageLLM, a novel domain-specific large language model that addresses this dual challenge. It is developed using a dual-track specialization framework from a single base model (LLaMA-3.1-8B). We opted for this dual-expert approach as a single multi-task model suffers from performance trade-offs, where optimizing for structured tasks (via SFT) degrades conversational fidelity (via DPO). Our dual-track method solves this by creating two specialists, allowing each to be optimally trained for its distinct capability. Our approach applies the instruction residual technique to restore instruction-following capabilities post-domain adaptation without supervised fine-tuning. We contribute: (1) application of this residual technique to the highly specialized mortgage finance domain; (2) a dual-expert architecture combining a conversational Q&A model and a structured task model for classification and summarization; and (3) an intelligent task routing mechanism using few-shot classification performed by one of the expert models itself. We validate our approach on domain-specific benchmarks, where our final model (MLM v2) significantly outperforms the base LLaMA-3.1-8B-Instruct, achieving an LLM-as-a-Judge summarization score of 4.58 (vs. 3.99), a Q&A score of 4.09 (vs. 4.0), and a classification score of 2.6 (vs. 1.2). On semantic similarity, our model achieved a BERTScore of 0.77 for summarization (vs. 0.74), 0.68 for Q&A (vs. 0.58), and 0.75 for classification (vs. 0.73), substantially outperforming baseline approaches.
>
---
#### [replaced 008] Pay Less Attention to Function Words for Free Robustness of Vision-Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对视觉-语言模型（VLM）在对抗攻击下的脆弱性问题，提出函数词去关注（FDA）方法，通过削弱函数词的注意力提升模型鲁棒性。实验表明其在几乎不影响性能的前提下显著降低攻击成功率。**

- **链接: [https://arxiv.org/pdf/2512.07222v2](https://arxiv.org/pdf/2512.07222v2)**

> **作者:** Qiwei Tian; Chenhao Lin; Zhengyu Zhao; Chao Shen
>
> **摘要:** To address the trade-off between robustness and performance for robust VLM, we observe that function words could incur vulnerability of VLMs against cross-modal adversarial attacks, and propose Function-word De-Attention (FDA) accordingly to mitigate the impact of function words. Similar to differential amplifiers, our FDA calculates the original and the function-word cross-attention within attention heads, and differentially subtracts the latter from the former for more aligned and robust VLMs. Comprehensive experiments include 2 SOTA baselines under 6 different attacks on 2 downstream tasks, 3 datasets, and 3 models. Overall, our FDA yields an average 18/13/53% ASR drop with only 0.2/0.3/0.6% performance drops on the 3 tested models on retrieval, and a 90% ASR drop with a 0.3% performance gain on visual grounding. We demonstrate the scalability, generalization, and zero-shot performance of FDA experimentally, as well as in-depth ablation studies and analysis. Code will be made publicly at https://github.com/michaeltian108/FDA.
>
---
#### [replaced 009] DiscoVerse: Multi-Agent Pharmaceutical Co-Scientist for Traceable Drug Discovery and Reverse Translation
- **分类: cs.CL; cs.MA**

- **简介: 该论文提出DiscoVerse，一种多智能体协作系统，旨在解决药企研发数据复用难问题。它通过角色专业化设计和人机协同，实现对海量药物研发档案的可追溯查询与逆向转化分析，支持证据溯源、知识整合与决策辅助。**

- **链接: [https://arxiv.org/pdf/2511.18259v2](https://arxiv.org/pdf/2511.18259v2)**

> **作者:** Xiaochen Zheng; Alvaro Serra; Ilya Schneider Chernov; Maddalena Marchesi; Eunice Musvasva; Tatyana Y. Doktorova
>
> **备注:** 24 pages, 5 figures, 3 tables. Updated version: added three pharmaceutical industry use cases and revised text for clarity
>
> **摘要:** Pharmaceutical research and development has accumulated vast and heterogeneous archives of data. Much of this knowledge stems from discontinued programs, and reusing these archives is invaluable for reverse translation. However, in practice, such reuse is often infeasible. In this work, we introduce DiscoVerse, a multi-agent co-scientist designed to support pharmaceutical research and development at Roche. Designed as a human-in-the-loop assistant, DiscoVerse enables domain-specific queries by delivering evidence-based answers: it retrieves relevant data, links across documents, summarises key findings and preserves institutional memory. We assess DiscoVerse through expert evaluation of source-linked outputs. Our evaluation spans a selected subset of 180 molecules from Roche's research and development repositories, encompassing over 0.87 billion BPE tokens and more than four decades of research. To our knowledge, this represents the first agentic framework to be systematically assessed on real pharmaceutical data for reverse translation, enabled by authorized access to confidential archives covering the full lifecycle of drug development. Our contributions include: role-specialized agent designs aligned with scientist workflows; human-in-the-loop support for reverse translation; expert evaluation; and a large-scale demonstration showing promising decision-making insights. In brief, across seven benchmark queries, DiscoVerse achieved near-perfect recall ($\geq 0.99$) with moderate precision ($0.71-0.91$). Qualitative assessments and three real-world pharmaceutical use cases further showed faithful, source-linked synthesis across preclinical and clinical evidence.
>
---
#### [replaced 010] CLARIFID: Improving Radiology Report Generation by Reinforcing Clinically Accurate Impressions and Enforcing Detailed Findings
- **分类: cs.CL**

- **简介: 该论文研究自动放射学报告生成，旨在提升报告的临床准确性。针对现有方法事实错误多、依赖单视图的问题，提出CLARIFID框架，通过分步建模、强化学习优化和多视图融合，增强发现与结论的逻辑一致性，提升诊断可靠性。**

- **链接: [https://arxiv.org/pdf/2507.17234v3](https://arxiv.org/pdf/2507.17234v3)**

> **作者:** Kyeongkyu Lee; Seonghwan Yoon; Hongki Lim
>
> **摘要:** Automatic generation of radiology reports has the potential to alleviate radiologists' significant workload, yet current methods struggle to deliver clinically reliable conclusions. In particular, most prior approaches focus on producing fluent text without effectively ensuring the factual correctness of the reports and often rely on single-view images, limiting diagnostic comprehensiveness. We propose CLARIFID, a novel framework that directly optimizes diagnostic correctness by mirroring the two-step workflow of experts. Specifically, CLARIFID (1) learns the logical flow from Findings to Impression through section-aware pretraining, (2) is fine-tuned with Proximal Policy Optimization in which the CheXbert F1 score of the Impression section serves as the reward, (3) employs controlled decoding that completes "Findings" before synthesizing the "Impression", and (4) fuses multiple chest X-ray views via a vision-transformer-based multi-view encoder. During inference, we apply a next-token forcing strategy followed by report-level re-ranking, ensuring that the model first produces a comprehensive "Findings" section before synthesizing the "Impression" and thereby preserving coherent clinical reasoning. Experimental results on the MIMIC-CXR dataset demonstrate that our method achieves superior clinical efficacy and outperforms existing baselines on clinical efficacy scores.
>
---
#### [replaced 011] ProgRAG: Hallucination-Resistant Progressive Retrieval and Reasoning over Knowledge Graphs
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究多跳知识图谱问答（KGQA）任务，旨在解决现有方法因检索不准和推理失败导致的幻觉问题。提出ProgRAG框架，通过分解问题、逐步扩展推理路径，并结合外部检索与不确定性感知剪枝，提升推理准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2511.10240v2](https://arxiv.org/pdf/2511.10240v2)**

> **作者:** Minbae Park; Hyemin Yang; Jeonghyun Kim; Kunsoo Park; Hyunjoon Kim
>
> **摘要:** Large Language Models (LLMs) demonstrate strong reasoning capabilities but struggle with hallucinations and limited transparency. Recently, KG-enhanced LLMs that integrate knowledge graphs (KGs) have been shown to improve reasoning performance, particularly for complex, knowledge-intensive tasks. However, these methods still face significant challenges, including inaccurate retrieval and reasoning failures, often exacerbated by long input contexts that obscure relevant information or by context constructions that struggle to capture the richer logical directions required by different question types. Furthermore, many of these approaches rely on LLMs to directly retrieve evidence from KGs, and to self-assess the sufficiency of this evidence, which often results in premature or incorrect reasoning. To address the retrieval and reasoning failures, we propose ProgRAG, a multi-hop knowledge graph question answering (KGQA) framework that decomposes complex questions into sub-questions, and progressively extends partial reasoning paths by answering each sub-question. At each step, external retrievers gather candidate evidence, which is then refined through uncertainty-aware pruning by the LLM. Finally, the context for LLM reasoning is optimized by organizing and rearranging the partial reasoning paths obtained from the sub-question answers. Experiments on three well-known datasets demonstrate that ProgRAG outperforms existing baselines in multi-hop KGQA, offering improved reliability and reasoning quality.
>
---
#### [replaced 012] Make LVLMs Focus: Context-Aware Attention Modulation for Better Multimodal In-Context Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究多模态上下文学习中LVLMs注意力机制的局限性，提出无需训练的CAMA方法，通过上下文感知的注意力调制增强关键token的关注，提升模型对上下文的利用效率和推理稳定性。**

- **链接: [https://arxiv.org/pdf/2505.17097v3](https://arxiv.org/pdf/2505.17097v3)**

> **作者:** Yanshu Li; Jianjiang Yang; Ziteng Yang; Bozheng Li; Ligong Han; Hongyang He; Zhengtao Yao; Yingjie Victor Chen; Songlin Fei; Dongfang Liu; Ruixiang Tang
>
> **备注:** 14 pages, 8 figures, 5 tables
>
> **摘要:** Multimodal in-context learning (ICL) is becoming a key capability that allows large vision-language models (LVLMs) to adapt to novel tasks without parameter updates, which expands their usefulness in many real-world applications. However, ICL performance remains unstable even when the in-context demonstrations (ICDs) are well matched, showing that LVLMs still struggle to make full use of the provided context. While existing work mainly focuses on prompt engineering or post-hoc logit calibration, we study the attention mechanisms inside LVLMs to address their inherent limitations. We identify two important weaknesses in their self-attention that hinder effective ICL. To address these weaknesses, we propose \textbf{Context-Aware Modulated Attention} (CAMA), a training-free and plug-and-play method that dynamically adjusts attention logits based on the input in-context sequence. CAMA uses a two-stage modulation process that strengthens attention to semantically important tokens, especially visual ones. Across four LVLMs and seven benchmarks, CAMA consistently outperforms vanilla models and baselines, showing clear effectiveness and generalization. It can also activate the intended benefits of prompt engineering methods and remains robust across different sequence configurations. Therefore, CAMA opens up new directions for improving multimodal reasoning through a deeper understanding of attention dynamics.
>
---
#### [replaced 013] Do Natural Language Descriptions of Model Activations Convey Privileged Information?
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究激活向量的自然语言描述是否真能揭示大模型内部机制。它发现现有方法在无内部信息时仍能成功，主因是描述受生成模型自身知识影响，而非目标模型。需更严谨的评估基准。**

- **链接: [https://arxiv.org/pdf/2509.13316v3](https://arxiv.org/pdf/2509.13316v3)**

> **作者:** Millicent Li; Alberto Mario Ceballos Arroyo; Giordano Rogers; Naomi Saphra; Byron C. Wallace
>
> **备注:** 40 pages, 6 figures. Updated and added content
>
> **摘要:** Recent interpretability methods have proposed to translate LLM internal representations into natural language descriptions using a second verbalizer LLM. This is intended to illuminate how the target model represents and operates on inputs. But do such activation verbalization approaches actually provide privileged knowledge about the internal workings of the target model, or do they merely convey information about its inputs? We critically evaluate popular verbalization methods across datasets used in prior work and find that they can succeed at benchmarks without any access to target model internals, suggesting that these datasets may not be ideal for evaluating verbalization methods. We then run controlled experiments which reveal that verbalizations often reflect the parametric knowledge of the verbalizer LLM which generated them, rather than the knowledge of the target LLM whose activations are decoded. Taken together, our results indicate a need for targeted benchmarks and experimental controls to rigorously assess whether verbalization methods provide meaningful insights into the operations of LLMs.
>
---
#### [replaced 014] CLIBD: Bridging Vision and Genomics for Biodiversity Monitoring at Scale
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出CLIBD，旨在融合图像与DNA条形码数据，解决大规模生物多样性监测中物种分类难题。通过对比学习构建多模态统一嵌入空间，实现无需微调的零样本物种识别，显著提升分类准确率。**

- **链接: [https://arxiv.org/pdf/2405.17537v5](https://arxiv.org/pdf/2405.17537v5)**

> **作者:** ZeMing Gong; Austin T. Wang; Xiaoliang Huo; Joakim Bruslund Haurum; Scott C. Lowe; Graham W. Taylor; Angel X. Chang
>
> **备注:** Add Variations of DNA encoding
>
> **摘要:** Measuring biodiversity is crucial for understanding ecosystem health. While prior works have developed machine learning models for taxonomic classification of photographic images and DNA separately, in this work, we introduce a multimodal approach combining both, using CLIP-style contrastive learning to align images, barcode DNA, and text-based representations of taxonomic labels in a unified embedding space. This allows for accurate classification of both known and unknown insect species without task-specific fine-tuning, leveraging contrastive learning for the first time to fuse barcode DNA and image data. Our method surpasses previous single-modality approaches in accuracy by over 8% on zero-shot learning tasks, showcasing its effectiveness in biodiversity studies.
>
---
#### [replaced 015] AEIOU: A Unified Defense Framework against NSFW Prompts in Text-to-Image Models
- **分类: cs.CR; cs.CL**

- **简介: 该论文针对文本到图像模型中NSFW提示词的安全问题，提出AEIOU统一防御框架。通过提取文本编码器隐状态中的NSFW特征，实现高效、可解释、可优化的检测，支持多种模型架构，在准确率和效率上均显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2412.18123v3](https://arxiv.org/pdf/2412.18123v3)**

> **作者:** Yiming Wang; Jiahao Chen; Qingming Li; Tong Zhang; Rui Zeng; Xing Yang; Shouling Ji
>
> **摘要:** As text-to-image (T2I) models advance and gain widespread adoption, their associated safety concerns are becoming increasingly critical. Malicious users exploit these models to generate Not-Safe-for-Work (NSFW) images using harmful or adversarial prompts, underscoring the need for effective safeguards to ensure the integrity and compliance of model outputs. However, existing detection methods often exhibit low accuracy and inefficiency. In this paper, we propose AEIOU, a defense framework that is adaptable, efficient, interpretable, optimizable, and unified against NSFW prompts in T2I models. AEIOU extracts NSFW features from the hidden states of the model's text encoder, utilizing the separable nature of these features to detect NSFW prompts. The detection process is efficient, requiring minimal inference time. AEIOU also offers real-time interpretation of results and supports optimization through data augmentation techniques. The framework is versatile, accommodating various T2I architectures. Our extensive experiments show that AEIOU significantly outperforms both commercial and open-source moderation tools, achieving over 95\% accuracy across all datasets and improving efficiency by at least tenfold. It effectively counters adaptive attacks and excels in few-shot and multi-label scenarios.
>
---
#### [replaced 016] Do Language Models Associate Sound with Meaning? A Multimodal Study of Sound Symbolism
- **分类: cs.CL**

- **简介: 该论文研究多模态大语言模型（MLLMs）是否能关联语音与意义，属于语言与听觉的跨模态任务。通过构建含真实词与伪词的LEX-ICON数据集，分析模型在多种语义维度上的音义关联能力及注意力机制，揭示其对语音象征性的理解。**

- **链接: [https://arxiv.org/pdf/2511.10045v3](https://arxiv.org/pdf/2511.10045v3)**

> **作者:** Jinhong Jeong; Sunghyun Lee; Jaeyoung Lee; Seonah Han; Youngjae Yu
>
> **备注:** 33 pages, 27 tables, 10 figures
>
> **摘要:** Sound symbolism is a linguistic concept that refers to non-arbitrary associations between phonetic forms and their meanings. We suggest that this can be a compelling probe into how Multimodal Large Language Models (MLLMs) interpret auditory information in human languages. We investigate MLLMs' performance on phonetic iconicity across textual (orthographic and IPA) and auditory forms of inputs with up to 25 semantic dimensions (e.g., sharp vs. round), observing models' layer-wise information processing by measuring phoneme-level attention fraction scores. To this end, we present LEX-ICON, an extensive mimetic word dataset consisting of 8,052 words from four natural languages (English, French, Japanese, and Korean) and 2,930 systematically constructed pseudo-words, annotated with semantic features applied across both text and audio modalities. Our key findings demonstrate (1) MLLMs' phonetic intuitions that align with existing linguistic research across multiple semantic dimensions and (2) phonosemantic attention patterns that highlight models' focus on iconic phonemes. These results bridge domains of artificial intelligence and cognitive linguistics, providing the first large-scale, quantitative analyses of phonetic iconicity in terms of MLLMs' interpretability.
>
---
#### [replaced 017] Arbitrage: Efficient Reasoning via Advantage-Aware Speculation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于推理加速任务，旨在解决大模型推理中语义等价步骤被误拒导致的计算浪费问题。提出Arbitrage框架，通过轻量级路由器动态选择草案或目标模型生成的推理步，提升效率与准确性权衡。**

- **链接: [https://arxiv.org/pdf/2512.05033v2](https://arxiv.org/pdf/2512.05033v2)**

> **作者:** Monishwaran Maheswaran; Rishabh Tiwari; Yuezhou Hu; Kerem Dilmen; Coleman Hooper; Haocheng Xi; Nicholas Lee; Mehrdad Farajtabar; Michael W. Mahoney; Kurt Keutzer; Amir Gholami
>
> **备注:** 22 pages
>
> **摘要:** Modern Large Language Models achieve impressive reasoning capabilities with long Chain of Thoughts, but they incur substantial computational cost during inference, and this motivates techniques to improve the performance-cost ratio. Among these techniques, Speculative Decoding accelerates inference by employing a fast but inaccurate draft model to autoregressively propose tokens, which are then verified in parallel by a more capable target model. However, due to unnecessary rejections caused by token mismatches in semantically equivalent steps, traditional token-level Speculative Decoding struggles in reasoning tasks. Although recent works have shifted to step-level semantic verification, which improve efficiency by accepting or rejecting entire reasoning steps, existing step-level methods still regenerate many rejected steps with little improvement, wasting valuable target compute. To address this challenge, we propose Arbitrage, a novel step-level speculative generation framework that routes generation dynamically based on the relative advantage between draft and target models. Instead of applying a fixed acceptance threshold, Arbitrage uses a lightweight router trained to predict when the target model is likely to produce a meaningfully better step. This routing approximates an ideal Arbitrage Oracle that always chooses the higher-quality step, achieving near-optimal efficiency-accuracy trade-offs. Across multiple mathematical reasoning benchmarks, Arbitrage consistently surpasses prior step-level Speculative Decoding baselines, reducing inference latency by up to $\sim2\times$ at matched accuracy.
>
---
#### [replaced 018] Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对信息密集图像的视觉问答任务，解决现有大模型在细粒度图文定位与多步推理上的困难。提出无需训练的Speculative Verdict框架，利用小模型生成多样化推理路径，通过共识机制筛选后由大模型综合得出答案，提升准确率与效率。**

- **链接: [https://arxiv.org/pdf/2510.20812v3](https://arxiv.org/pdf/2510.20812v3)**

> **作者:** Yuhan Liu; Lianhui Qin; Shengjie Wang
>
> **摘要:** Large Vision-Language Models (VLMs) have achieved remarkable progress in multimodal understanding, yet they struggle when reasoning over information-intensive images that densely interleave textual annotations with fine-grained graphical elements. The main challenges lie in precisely localizing critical cues in dense layouts and multi-hop reasoning to integrate dispersed evidence. We propose Speculative Verdict (SV), a training-free framework inspired by speculative decoding that combines multiple lightweight draft experts with a large verdict model. In the draft stage, small VLMs act as draft experts to generate reasoning paths that provide diverse localization candidates; in the verdict stage, a strong VLM synthesizes these paths to produce the final answer, minimizing computational cost while recovering correct answers. To further improve efficiency and accuracy, SV introduces a consensus expert selection mechanism that forwards only high-agreement reasoning paths to the verdict. Empirically, SV achieves consistent gains on challenging information-intensive and high-resolution visual question answering benchmarks, including InfographicVQA, ChartMuseum, ChartQAPro, and HR-Bench 4K. By synthesizing correct insights from multiple partially accurate reasoning paths, SV achieves both error correction and cost-efficiency compared to large proprietary models or training pipelines. Code is available at https://github.com/Tinaliu0123/speculative-verdict.
>
---
#### [replaced 019] Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在选择题中知识与预测不一致的问题，提出一种无参数干预方法KAPPA，通过投影调整隐藏状态，对齐知识与预测方向，提升准确率，并验证其在多任务上的有效性。**

- **链接: [https://arxiv.org/pdf/2509.23782v2](https://arxiv.org/pdf/2509.23782v2)**

> **作者:** Yoonah Park; Haesung Pyun; Yohan Jo
>
> **摘要:** Large Language Models (LLMs) often fail on multiple-choice questions (MCQs) despite demonstrating correct knowledge in other contexts, such as free-form generation. To investigate the mechanism underlying this knowledge-prediction gap on MCQs and alleviate it, we conduct a probing analysis and find that residual streams in certain layers contain a subspace spanned by two important bases: a \emph{knowledge basis} that encodes the probability of the ground-truth answer for a given MCQ and a \emph{prediction basis} that encodes the probability of the answer choice predicted by the model. We observe that incorrect predictions arise from a misalignment of the model's hidden states along these two bases. Hence, we introduce \textbf{KAPPA} (Knowledge-Aligned Prediction through Projection-based Adjustment), a parameter-free intervention that transforms the hidden states to align the prediction coordinate with the knowledge coordinate within this subspace. Experiments on binary-choice reformulations of Big-Bench-Hard and ARC-Challenge show that KAPPA substantially improves accuracy and consistently outperforms baselines. While optimal subspaces differ across tasks, subspaces generalize to some extent, as supported by cross-dataset experiments. Moreover, KAPPA extends its effectiveness to free-form questions beyond MCQs. Our work provides a new geometric understanding of the knowledge-prediction gap and offers a practical method for better aligning model behavior with its latent knowledge.
>
---
#### [replaced 020] Can AI Truly Represent Your Voice in Deliberations? A Comprehensive Study of Large-Scale Opinion Aggregation with LLMs
- **分类: cs.CL**

- **简介: 该论文研究大模型在公共协商摘要中的公平性问题，旨在解决现有方法对少数观点代表性不足及人类评估不可扩展的难题。作者构建了大规模人工标注数据集DeliberationBank，并训练出更贴近人类判断的自动评估模型DeliberationJudge，用于评测和改进协商摘要的代表性与中立性。**

- **链接: [https://arxiv.org/pdf/2510.05154v3](https://arxiv.org/pdf/2510.05154v3)**

> **作者:** Shenzhe Zhu; Shu Yang; Michiel A. Bakker; Alex Pentland; Jiaxin Pei
>
> **摘要:** Large-scale public deliberations generate thousands of free-form contributions that must be synthesized into representative and neutral summaries for policy use. While LLMs have been shown as a promising tool to generate summaries for large-scale deliberations, they also risk underrepresenting minority perspectives and exhibiting bias with respect to the input order, raising fairness concerns in high-stakes contexts. Studying and fixing these issues requires a comprehensive evaluation at a large scale, yet current practice often relies on LLMs as judges, which show weak alignment with human judgments. To address this, we present DeliberationBank, a large-scale human-grounded dataset with (1) opinion data spanning ten deliberation questions created by 3,000 participants and (2) summary judgment data annotated by 4,500 participants across four dimensions (representativeness, informativeness, neutrality, policy approval). Using these datasets, we train DeliberationJudge, a fine-tuned DeBERTa model that can rate deliberation summaries from individual perspectives. DeliberationJudge is more efficient and more aligned with human judgements compared to a wide range of LLM judges. With DeliberationJudge, we evaluate 18 LLMs and reveal persistent weaknesses in deliberation summarization, especially underrepresentation of minority positions. Our framework provides a scalable and reliable way to evaluate deliberation summarization, helping ensure AI systems are more representative and equitable for policymaking.
>
---
#### [replaced 021] Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型推理中KV缓存的内存瓶颈问题，提出Compactor方法，通过近似杠杆分数进行无训练、查询无关的键值缓存压缩，并设计上下文校准策略以最大化压缩率。结合高效推理引擎，显著降低内存占用，提升部署效率。**

- **链接: [https://arxiv.org/pdf/2507.08143v2](https://arxiv.org/pdf/2507.08143v2)**

> **作者:** Vivek Chari; Benjamin Van Durme
>
> **摘要:** Modern Large Language Models (LLMs) are increasingly trained to support very large context windows. We present Compactor, a training-free, query-agnostic KV compression strategy that uses approximate leverage scores to determine token importance. We show that Compactor can achieve the same performance as competing methods while retaining 20% fewer tokens in both synthetic and real-world context tasks, while being more task-robust. We further introduce a procedure for context-calibrated compression: inferring the maximum compression a given context supports before significant performance loss. Using context-calibrated compression, we show that Compactor achieves full KV performance on Longbench while reducing the KV memory burden by 68%, on average. To demonstrate the efficacy and generalizability of our approach, we apply Compactor to 27 synthetic and real-world tasks from RULER and Longbench, with models from both the Qwen 2.5 and Llama 3.1 families. Finally, we release compactor-vllm, an inference engine and suite of optimized Triton kernels designed to efficiently support the sparse, non-contiguous memory access patterns inherent to compressed KV caches. This work demonstrates that Compactor offers a practical, high-performance solution for alleviating the memory bottleneck in modern LLM deployment.
>
---
#### [replaced 022] You May Speak Freely: Improving the Fine-Grained Visual Recognition Capabilities of Multimodal Large Language Models with Answer Extraction
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对细粒度视觉分类中多选项问答的评估难题，提出nlg2choice方法。通过先生成开放式回答再进行约束解码匹配选项，提升MLLM在高数量级相似类别下的分类与检索性能，尤其优化了检索场景的计算效率。**

- **链接: [https://arxiv.org/pdf/2510.14885v2](https://arxiv.org/pdf/2510.14885v2)**

> **作者:** Logan Lawrence; Oindrila Saha; Megan Wei; Chen Sun; Subhransu Maji; Grant Van Horn
>
> **备注:** Accepted to WACV26. 12 pages, 8 tables, 5 figures
>
> **摘要:** Despite the renewed interest in zero-shot visual classification due to the rise of Multimodal Large Language Models (MLLMs), the problem of evaluating free-form responses of auto-regressive models remains a persistent challenge. Most existing works focus on language-only tasks or don't consider Multiple Choice Questions (MCQs) beyond 5-way options, both of which are critical capabilities to solve tasks in Fine-Grained Visual Classification (FGVC) where choice counts are in the hundreds to thousands and the choices are highly related. Furthermore, in this highly multi-way MCQ setting it is not clear how to extend LLM choice extraction to retrieval-based problems, where computing probabilities over the choice set is computationally costly. In this work we investigate nlg2choice, a simple two-stage method which first asks the MLLM an open-ended question for the task with minimal constraints, then uses text-only constrained decoding to predict the most likely choice. In retrieval settings, we compute the probability of the constrained response taking that choice with an early stopping method to significantly improve throughput. Our results show improvement over a suite of seven fine-grained visual datasets when evaluating in terms of classification and retrieval, and show that this performance holds over the various ways that users of LLMs can implement tasks in natural language.
>
---
#### [replaced 023] OMNIGUARD: An Efficient Approach for AI Safety Moderation Across Languages and Modalities
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于AI安全任务，旨在检测跨语言和模态的有害提示。针对现有方法在低资源语言和非文本模态上的不足，提出Omniguard，利用对齐的内部表示构建语言/模态无关的分类器，显著提升检测精度与效率。**

- **链接: [https://arxiv.org/pdf/2505.23856v2](https://arxiv.org/pdf/2505.23856v2)**

> **作者:** Sahil Verma; Keegan Hines; Jeff Bilmes; Charlotte Siska; Luke Zettlemoyer; Hila Gonen; Chandan Singh
>
> **摘要:** The emerging capabilities of large language models (LLMs) have sparked concerns about their immediate potential for harmful misuse. The core approach to mitigate these concerns is the detection of harmful queries to the model. Current detection approaches are fallible, and are particularly susceptible to attacks that exploit mismatched generalization of model capabilities (e.g., prompts in low-resource languages or prompts provided in non-text modalities such as image and audio). To tackle this challenge, we propose Omniguard, an approach for detecting harmful prompts across languages and modalities. Our approach (i) identifies internal representations of an LLM/MLLM that are aligned across languages or modalities and then (ii) uses them to build a language-agnostic or modality-agnostic classifier for detecting harmful prompts. Omniguard improves harmful prompt classification accuracy by 11.57\% over the strongest baseline in a multilingual setting, by 20.44\% for image-based prompts, and sets a new SOTA for audio-based prompts. By repurposing embeddings computed during generation, Omniguard is also very efficient ($\approx\!120 \times$ faster than the next fastest baseline). Code and data are available at: https://github.com/vsahil/OmniGuard.
>
---
#### [replaced 024] Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型中内在价值观与提示激发价值观的表达机制差异。通过分析价值向量和价值神经元，揭示二者既有共性也存在独特成分，导致在可引导性和响应多样性上表现不同。**

- **链接: [https://arxiv.org/pdf/2509.24319v2](https://arxiv.org/pdf/2509.24319v2)**

> **作者:** Jongwook Han; Jongwon Lim; Injin Kong; Yohan Jo
>
> **摘要:** Large language models (LLMs) can express different values in two distinct ways: (1) intrinsic expression, reflecting the model's inherent values learned during training, and (2) prompted expression, elicited by explicit prompts. Given their widespread use in value alignment and persona steering, it is paramount to clearly understand their underlying mechanisms, particularly whether they mostly overlap (as one might expect) or rely on substantially different mechanisms, but this remains largely understudied. We analyze this at the mechanistic level using two approaches: (1) value vectors, feature directions representing value mechanisms extracted from the residual stream, and (2) value neurons, MLP neurons that contribute to value expressions. We demonstrate that intrinsic and prompted value mechanisms partly share common components that are crucial for inducing value expression, but also possess unique elements that manifest in different ways. As a result, these mechanisms lead to different degrees of value steerability (prompted > intrinsic) and response diversity (intrinsic > prompted). In particular, components unique to the intrinsic mechanism seem to promote lexical diversity in responses, whereas those specific to the prompted mechanism primarily strengthen instruction following, taking effect even in distant tasks like jailbreaking.
>
---
#### [replaced 025] Detecting value-expressive text posts in Russian social media
- **分类: cs.CL; cs.AI**

- **简介: 该论文属文本分类任务，旨在解决俄语社交媒体中个人价值观表达识别问题。通过专家、众包与ChatGPT协同标注数据，采用主动学习与预训练模型融合方法，构建高效分类模型，实现对VKontakte平台价值表达帖的准确检测。**

- **链接: [https://arxiv.org/pdf/2312.08968v3](https://arxiv.org/pdf/2312.08968v3)**

> **作者:** Maria Milkova; Maksim Rudnev; Lidia Okolskaya
>
> **摘要:** Basic values are concepts or beliefs which pertain to desirable end-states and transcend specific situations. Studying personal values in social media can illuminate how and why societal values evolve especially when the stimuli-based methods, such as surveys, are inefficient, for instance, in hard-to-reach populations. On the other hand, user-generated content is driven by the massive use of stereotyped, culturally defined speech constructions rather than authentic expressions of personal values. We aimed to find a model that can accurately detect value-expressive posts in Russian social media VKontakte. A training dataset of 5,035 posts was annotated by three experts, 304 crowd-workers and ChatGPT. Crowd-workers and experts showed only moderate agreement in categorizing posts. ChatGPT was more consistent but struggled with spam detection. We applied an ensemble of human- and AI-assisted annotation involving active learning approach, subsequently trained several classification models using embeddings from various pre-trained transformer-based language models. The best performance was achieved with embeddings from a fine-tuned rubert-tiny2 model, yielding high value detection quality (F1 = 0.77, F1-macro = 0.83). This model provides a crucial step to a study of values within and between Russian social media users.
>
---
#### [replaced 026] LLMSQL: Upgrading WikiSQL for the LLM Era of Text-to-SQL
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对WikiSQL数据集在大模型时代的缺陷，提出LLMSQL，通过系统性清洗与重标注，构建面向大语言模型的文本到SQL新基准，解决原数据集的错误与不兼容问题，支持更高效的模型评估与训练。**

- **链接: [https://arxiv.org/pdf/2510.02350v2](https://arxiv.org/pdf/2510.02350v2)**

> **作者:** Dzmitry Pihulski; Karol Charchut; Viktoria Novogrodskaia; Jan Kocoń
>
> **备注:** To appear in the Proceedings of the IEEE International Conference on Data Mining Workshops (ICDMW)
>
> **摘要:** Converting natural language questions into SQL queries enables non-expert users to interact with relational databases and has long been a central task for natural language interfaces to data. While the WikiSQL dataset played a key role in early text-to-SQL research, its usage has declined due to structural and annotation issues, including case sensitivity inconsistencies, data type mismatches, syntax errors, and unanswered questions. We present LLMSQL, a systematic revision and transformation of WikiSQL designed for the large language model era. We classify these errors and implement automated methods for cleaning and re-annotation. To assess the impact of these improvements, we evaluated multiple large language models, including Gemma 3, LLaMA 3.2, Mistral 7B, gpt-oss 20B, Phi-3.5 Mini, Qwen 2.5, OpenAI o4-mini, DeepSeek-R1, and others. Notably, DeepSeek-R1 achieves 88.40% accuracy in a zero-shot setting, and models under 10B parameters surpass 90% accuracy after fine-tuning. Rather than serving as an update, LLMSQL is introduced as an LLM-ready benchmark. Unlike the original WikiSQL, which was tailored for pointer-network models selecting tokens from input, LLMSQL provides clean natural language questions and full SQL queries as plain text, enabling straightforward generation and evaluation for modern natural-language-to-SQL models.
>
---
#### [replaced 027] StreamingThinker: Large Language Models Can Think While Reading
- **分类: cs.CL**

- **简介: 该论文提出“流式思考”范式，解决大模型在输入完整后才开始推理导致的延迟问题。通过StreamingThinker框架实现边读边想，降低等待时间与推理延迟，保持性能的同时提升效率。**

- **链接: [https://arxiv.org/pdf/2510.17238v2](https://arxiv.org/pdf/2510.17238v2)**

> **作者:** Junlong Tong; Yingqi Fan; Anhao Zhao; Yunpu Ma; Xiaoyu Shen
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities in chain of thought (CoT) reasoning. However, the current LLM reasoning paradigm initiates thinking only after the entire input is available, which introduces unnecessary latency and weakens attention to earlier information in dynamic scenarios. Inspired by human cognition of thinking while reading, we first design a \textit{\textbf{streaming thinking}} paradigm for LLMs, where reasoning unfolds in the order of input and further adjusts its depth once reading is complete. We instantiate this paradigm with \textit{StreamingThinker}, a framework that enables LLMs to think while reading through the integration of streaming CoT generation, streaming-constraint training, and streaming parallel inference. Specifically, StreamingThinker employs streaming reasoning units with quality control for CoT generation, enforces order-preserving reasoning through streaming attention masks and position encoding, and leverages parallel KV caches that decouple input encoding from reasoning generation, thereby ensuring alignment and enabling true concurrency. We evaluate StreamingThinker on the Qwen3 model family across math reasoning, logical reasoning, and context-based QA reasoning tasks. Experimental results show that the StreamingThinker preserves performance comparable to batch thinking, while yielding an 80\% reduction in token waiting before the onset of reasoning and a more than 60\% reduction in time-level latency for producing the final answer, demonstrating the effectiveness of the streaming paradigm for LLM reasoning. Code will be released at https://github.com/EIT-NLP/StreamingLLM/tree/main/StreamingThinker.
>
---
#### [replaced 028] LLMs Can't Handle Peer Pressure: Crumbling under Multi-Agent Social Interactions
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在多智能体社交互动中的决策能力，揭示其易受同伴压力影响的问题。通过构建KAIROS基准，评估不同训练方法对模型抗干扰能力的提升效果，发现模型规模和GRPO训练是关键因素。**

- **链接: [https://arxiv.org/pdf/2508.18321v3](https://arxiv.org/pdf/2508.18321v3)**

> **作者:** Maojia Song; Tej Deep Pala; Ruiwen Zhou; Weisheng Jin; Amir Zadeh; Chuan Li; Dorien Herremans; Soujanya Poria
>
> **摘要:** Large language models (LLMs) are increasingly integrated into multi-agent systems (MAS), where peer interactions shape individual decisions. While prior work has mainly examined conformity bias, we broaden the view to include how LLMs build rapport from prior interactions, discern and integrate high-quality peer information, and resist misleading inputs-abilities essential for achieving collective intelligence under complex social dynamics. We introduce KAIROS, a benchmark that simulates quiz-style collaboration with peer agents whose rapport levels and behaviours can be precisely controlled in both historical interactions and the current round. This unified setup enables systematic analysis of how rapport, peer actions, and the model's self-confidence jointly influence decision-making. Using KAIROS, we evaluate prompting, supervised fine-tuning, and reinforcement learning via Group Relative Policy Optimisation (GRPO). Results show that model scale is a primary factor moderating susceptibility to social influence: larger models are more resilient and benefit from prompting-based mitigation, whereas smaller models remain vulnerable. Only carefully configured GRPO training yields consistent robustness and performance gains for small models.
>
---
#### [replaced 029] Bench4KE: Benchmarking Automated Competency Question Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦知识工程自动化中的 competency question（CQ）生成任务，旨在解决现有方法评估缺乏标准化的问题。作者提出 Bench4KE，一个基于 API 的基准测试系统，包含17个真实项目的 CQ 数据集和多种相似度指标，用于评估 LLM 基的 CQ 生成工具，并为未来研究提供基线。**

- **链接: [https://arxiv.org/pdf/2505.24554v3](https://arxiv.org/pdf/2505.24554v3)**

> **作者:** Anna Sofia Lippolis; Minh Davide Ragagni; Paolo Ciancarini; Andrea Giovanni Nuzzolese; Valentina Presutti
>
> **摘要:** The availability of Large Language Models (LLMs) presents a unique opportunity to reinvigorate research on Knowledge Engineering (KE) automation. This trend is already evident in recent efforts developing LLM-based methods and tools for the automatic generation of Competency Questions (CQs), natural language questions used by ontology engineers to define the functional requirements of an ontology. However, the evaluation of these tools lacks standardization. This undermines the methodological rigor and hinders the replication and comparison of results. To address this gap, we introduce Bench4KE, an extensible API-based benchmarking system for KE automation. The presented release focuses on evaluating tools that generate CQs automatically. Bench4KE provides a curated gold standard consisting of CQ datasets from 17 real-world ontology engineering projects and uses a suite of similarity metrics to assess the quality of the CQs generated. We present a comparative analysis of 6 recent CQ generation systems, which are based on LLMs, establishing a baseline for future research. Bench4KE is also designed to accommodate additional KE automation tasks, such as SPARQL query generation, ontology testing and drafting. Code and datasets are publicly available under the Apache 2.0 license.
>
---
#### [replaced 030] When Many-Shot Prompting Fails: An Empirical Study of LLM Code Translation
- **分类: cs.SE; cs.AI; cs.CL; cs.PL**

- **简介: 该论文研究大模型在代码翻译中的多示例提示效果，发现并非示例越多越好。实验表明，少量优质示例（5-25个）在功能正确性上表现最佳，揭示“多示例悖论”，挑战“越多越好”的惯常假设，强调任务依赖的最优提示策略。**

- **链接: [https://arxiv.org/pdf/2510.16809v2](https://arxiv.org/pdf/2510.16809v2)**

> **作者:** Amirkia Rafiei Oskooei; Kaan Baturalp Cosdan; Husamettin Isiktas; Mehmet S. Aktas
>
> **备注:** Accepted to ICSE 2026 (RECODE workshop)
>
> **摘要:** Large Language Models (LLMs) with vast context windows offer new avenues for in-context learning (ICL), where providing many examples ("many-shot" prompting) is often assumed to enhance performance. We investigate this assumption for the complex task of code translation. Through a large-scale empirical study of over 90,000 translations, we systematically evaluate the impact of scaling in-context examples from zero-shot to many-shot configurations of up to 625 examples, with prompts spanning from approximately 100,000 to 800,000 tokens. Our findings reveal a "many-shot paradox": while static similarity metrics may modestly improve with more examples, functional correctness consistently peaks with few-shot prompting (5-25 examples). Providing substantially more examples often degrades this crucial functional performance. This study highlights that for code translation, the quality of a few well-chosen examples outweighs sheer quantity, challenging the universal efficacy of "more is better" for ICL and underscoring the task-dependent nature of optimal prompting strategies. Our results have significant implications for effectively leveraging LLMs in software engineering.
>
---
#### [replaced 031] EEG-to-Text Translation: A Model for Deciphering Human Brain Activity
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究EEG到文本的转换任务，旨在解决现有模型解码性能不足的问题。作者提出R1 Translator模型，结合双向LSTM编码器与预训练Transformer解码器，提升文本生成质量，在ROUGE、CER、WER等指标上均优于现有方法。**

- **链接: [https://arxiv.org/pdf/2505.13936v2](https://arxiv.org/pdf/2505.13936v2)**

> **作者:** Saydul Akbar Murad; Ashim Dahal; Nick Rahimi
>
> **摘要:** With the rapid advancement of large language models like Gemini, GPT, and others, bridging the gap between the human brain and language processing has become an important area of focus. To address this challenge, researchers have developed various models to decode EEG signals into text. However, these models still face significant performance limitations. To overcome these shortcomings, we propose a new model, R1 Translator, which aims to improve the performance of EEG-to-text decoding. The R1 Translator model combines a bidirectional LSTM encoder with a pretrained transformer-based decoder, utilizing EEG features to produce high-quality text outputs. The model processes EEG embeddings through the LSTM to capture sequential dependencies, which are then fed into the transformer decoder for effective text generation. The R1 Translator excels in ROUGE metrics, outperforming both T5 (previous research) and Brain Translator. Specifically, R1 achieves a ROUGE-1 score of 38.00% (P), which is up to 9% higher than T5 (34.89%) and 3% better than Brain (35.69%). It also leads in ROUGE-L, with a F1 score of 32.51%, outperforming T5 by 3% (29.67%) and Brain by 2% (30.38%). In terms of CER, R1 achieves a CER of 0.5795, which is 2% lower than T5 (0.5917) and 4% lower than Brain (0.6001). Additionally, R1 performs better in WER with a score of 0.7280, outperforming T5 by 4.3% (0.7610) and Brain by 3.6% (0.7553). Code is available at https://github.com/Mmurrad/EEG-To-text.
>
---
#### [replaced 032] OS-Sentinel: Towards Safety-Enhanced Mobile GUI Agents via Hybrid Validation in Realistic Workflows
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文研究移动GUI智能体的安全增强，提出OS-Sentinel框架，结合形式化验证与基于VLM的上下文判断，检测操作中的系统风险与隐私泄露。构建了MobileRisk-Live沙箱环境与基准，实验证明其有效性。**

- **链接: [https://arxiv.org/pdf/2510.24411v2](https://arxiv.org/pdf/2510.24411v2)**

> **作者:** Qiushi Sun; Mukai Li; Zhoumianze Liu; Zhihui Xie; Fangzhi Xu; Zhangyue Yin; Kanzhi Cheng; Zehao Li; Zichen Ding; Qi Liu; Zhiyong Wu; Zhuosheng Zhang; Ben Kao; Lingpeng Kong
>
> **备注:** work in progress
>
> **摘要:** Computer-using agents powered by Vision-Language Models (VLMs) have demonstrated human-like capabilities in operating digital environments like mobile platforms. While these agents hold great promise for advancing digital automation, their potential for unsafe operations, such as system compromise and privacy leakage, is raising significant concerns. Detecting these safety concerns across the vast and complex operational space of mobile environments presents a formidable challenge that remains critically underexplored. To establish a foundation for mobile agent safety research, we introduce MobileRisk-Live, a dynamic sandbox environment accompanied by a safety detection benchmark comprising realistic trajectories with fine-grained annotations. Built upon this, we propose OS-Sentinel, a novel hybrid safety detection framework that synergistically combines a Formal Verifier for detecting explicit system-level violations with a VLM-based Contextual Judge for assessing contextual risks and agent actions. Experiments show that OS-Sentinel achieves 10%-30% improvements over existing approaches across multiple metrics. Further analysis provides critical insights that foster the development of safer and more reliable autonomous mobile agents. Our code and data are available at https://github.com/OS-Copilot/OS-Sentinel.
>
---
#### [replaced 033] Representational Stability of Truth in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型对真假及非真非假内容的表征稳定性。通过线性探针分析，发现模型对不熟悉实体的陈述稳定性差，而对虚构但熟悉的陈述更稳定，表明稳定性源于认知熟悉度而非语言形式。**

- **链接: [https://arxiv.org/pdf/2511.19166v2](https://arxiv.org/pdf/2511.19166v2)**

> **作者:** Samantha Dies; Courtney Maynard; Germans Savcisens; Tina Eliassi-Rad
>
> **备注:** 25 pages, 24 figures
>
> **摘要:** Large language models (LLMs) are widely used for factual tasks such as "What treats asthma?" or "What is the capital of Latvia?". However, it remains unclear how stably LLMs encode distinctions between true, false, and neither-true-nor-false content in their internal probabilistic representations. We introduce representational stability as the robustness of an LLM's veracity representations to perturbations in the operational definition of truth. We assess representational stability by (i) training a linear probe on an LLM's activations to separate true from not-true statements and (ii) measuring how its learned decision boundary shifts under controlled label changes. Using activations from sixteen open-source models and three factual domains, we compare two types of neither statements. The first are fact-like assertions about entities we believe to be absent from any training data. We call these unfamiliar neither statements. The second are nonfactual claims drawn from well-known fictional contexts. We call these familiar neither statements. The unfamiliar statements induce the largest boundary shifts, producing up to $40\%$ flipped truth judgements in fragile domains (such as word definitions), while familiar fictional statements remain more coherently clustered and yield smaller changes ($\leq 8.2\%$). These results suggest that representational stability stems more from epistemic familiarity than from linguistic form. More broadly, our approach provides a diagnostic for auditing and training LLMs to preserve coherent truth assignments under semantic uncertainty, rather than optimizing for output accuracy alone.
>
---
#### [replaced 034] Collaborative Causal Sensemaking: Closing the Complementarity Gap in Human-AI Decision Support
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文聚焦人机协同决策支持任务，旨在解决当前AI作为“答题器”导致的互补性差距问题。提出“协作因果意义建构”（CCS）研究框架，通过新训练环境、共享心智模型和以信任与互补为核心的评估，培育AI与人类协同推理能力。**

- **链接: [https://arxiv.org/pdf/2512.07801v2](https://arxiv.org/pdf/2512.07801v2)**

> **作者:** Raunak Jain; Mudita Khurana
>
> **摘要:** LLM-based agents are increasingly deployed for expert decision support, yet human-AI teams in high-stakes settings do not yet reliably outperform the best individual. We argue this complementarity gap reflects a fundamental mismatch: current agents are trained as answer engines, not as partners in the collaborative sensemaking through which experts actually make decisions. Sensemaking (the ability to co-construct causal explanations, surface uncertainties, and adapt goals) is the key capability that current training pipelines do not explicitly develop or evaluate. We propose Collaborative Causal Sensemaking (CCS) as a research agenda to develop this capability from the ground up, spanning new training environments that reward collaborative thinking, representations for shared human-AI mental models, and evaluation centred on trust and complementarity. These directions can advance MAS research toward agents that think with their human partners rather than for them.
>
---
#### [replaced 035] The Necessity of Imperfection:Reversing Model Collapse via Simulating Cognitive Boundedness
- **分类: cs.AI; cs.CL; cs.CY; cs.LG; q-fin.TR**

- **简介: 该论文属AI数据生成任务，旨在解决合成数据导致模型崩溃的问题。作者提出模拟人类认知过程而非复制表面文本，构建认知计算框架，生成含人类典型“不完美”的文本，实验证明其有效降低金融策略回撤并提升防御性收益。**

- **链接: [https://arxiv.org/pdf/2512.01354v3](https://arxiv.org/pdf/2512.01354v3)**

> **作者:** Zhongjie Jiang
>
> **备注:** 60 pages,9 figures. v3: Major update. Added 3D topological visualization (Figure 1) and independent computational verification of the Adaptive Markets Hypothesis (AMH). Includes comprehensive Supplementary Materials (algorithmic pseudocode, system architecture, and real-time GARCH logs) for technical reproducibility
>
> **摘要:** Although synthetic data is widely promoted as a remedy, its prevailing production paradigm -- one optimizing for statistical smoothness -- systematically removes the long-tail, cognitively grounded irregularities that characterize human text. Prolonged training on such statistically optimal but cognitively impoverished data accelerates model collapse. This paper proposes a paradigm shift: instead of imitating the surface properties of data, we simulate the cognitive processes that generate human text. We introduce the Prompt-driven Cognitive Computing Framework (PMCSF), whose core consists of a Cognitive State Decoder (CSD) that reverse-engineers unstructured text into structured cognitive vectors, and a Cognitive Text Encoder (CTE) that re-materializes these states into text enriched with human-typical imperfections via mathematically defined Cognitive Perturbation Operators. The framework is validated through a two-stage objective evaluation pipeline. First, in cognitive codec verification, CTE text yields a Jensen-Shannon divergence of 0.0614 from human text (vs. 0.4431 for standard LLM output), passes double-blind professional media review, and achieves an intraclass correlation coefficient ICC > 0.9 for cognitive profile alignment across heterogeneous models. Second, in functional gain evaluation, isomorphic stress tests in the A-share market show that strategies incorporating CTE-generated data reduce maximum drawdown by 47.4% during the 2015 crash and deliver 8.6% Defensive Alpha, exceeding transaction costs by a factor of 33. Our findings demonstrate that modelling human cognitive limitations -- not copying surface data -- enables synthetic data with genuine functional gain, offering a viable technical pathway toward resolving the AI data-collapse crisis.
>
---
#### [replaced 036] SimSUM: Simulated Benchmark with Structured and Unstructured Medical Records
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于临床信息抽取任务，旨在解决现有数据集中缺乏结构化与非结构化医疗记录关联的问题。作者构建了SimSUM模拟数据集，结合专家定义的贝叶斯网络生成结构化变量，并用大模型生成对应临床文本，支持带背景知识的症状抽取研究。**

- **链接: [https://arxiv.org/pdf/2409.08936v4](https://arxiv.org/pdf/2409.08936v4)**

> **作者:** Paloma Rabaey; Stefan Heytens; Thomas Demeester
>
> **备注:** An earlier version of this dataset was published under the name SynSUM. It has since been renamed to SimSUM to avoid confusion with synthetic data generated from real data, and to emphasize the simulated nature of the dataset. The dataset is available at https://github.com/prabaey/SimSUM
>
> **摘要:** Clinical information extraction, which involves structuring clinical concepts from unstructured medical text, remains a challenging problem that could benefit from the inclusion of tabular background information available in electronic health records. Existing open-source datasets lack explicit links between structured features and clinical concepts in the text, motivating the need for a new research dataset. We introduce SimSUM, a benchmark dataset of 10,000 simulated patient records that link unstructured clinical notes with structured background variables. Each record simulates a patient encounter in the domain of respiratory diseases and includes tabular data (e.g., symptoms, diagnoses, underlying conditions) generated from a Bayesian network whose structure and parameters are defined by domain experts. A large language model (GPT-4o) is prompted to generate a clinical note describing the encounter, including symptoms and relevant context. These notes are annotated with span-level symptom mentions. We conduct an expert evaluation to assess note quality and run baseline predictive models on both the tabular and textual data. The SimSUM dataset is primarily designed to support research on clinical information extraction in the presence of tabular background variables, which can be linked through domain knowledge to concepts of interest to be extracted from the text -- namely, symptoms in the case of SimSUM. Secondary uses include research on the automation of clinical reasoning over both tabular data and text, causal effect estimation in the presence of tabular and/or textual confounders, and multi-modal synthetic data generation. SimSUM is not intended for training clinical decision support systems or production-grade models, but rather to facilitate reproducible research in a simplified and controlled setting.
>
---
#### [replaced 037] ScamAgents: How AI Agents Can Simulate Human-Level Scam Calls
- **分类: cs.CR; cs.AI; cs.CL; cs.MA**

- **简介: 该论文研究AI代理模拟人类级诈骗电话的任务，揭示现有大模型安全机制在多轮对话中易被绕过的问题。作者构建了能动态生成欺骗性对话的ScamAgent，展示其规避防护的能力，并呼吁加强多轮交互下的安全防控。**

- **链接: [https://arxiv.org/pdf/2508.06457v2](https://arxiv.org/pdf/2508.06457v2)**

> **作者:** Sanket Badhe
>
> **备注:** Accepted at CAMLIS 25: Conference on Applied Machine Learning for Information Security. 19 pages, 3 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive fluency and reasoning capabilities, but their potential for misuse has raised growing concern. In this paper, we present ScamAgent, an autonomous multi-turn agent built on top of LLMs, capable of generating highly realistic scam call scripts that simulate real-world fraud scenarios. Unlike prior work focused on single-shot prompt misuse, ScamAgent maintains dialogue memory, adapts dynamically to simulated user responses, and employs deceptive persuasion strategies across conversational turns. We show that current LLM safety guardrails, including refusal mechanisms and content filters, are ineffective against such agent-based threats. Even models with strong prompt-level safeguards can be bypassed when prompts are decomposed, disguised, or delivered incrementally within an agent framework. We further demonstrate the transformation of scam scripts into lifelike voice calls using modern text-to-speech systems, completing a fully automated scam pipeline. Our findings highlight an urgent need for multi-turn safety auditing, agent-level control frameworks, and new methods to detect and disrupt conversational deception powered by generative AI.
>
---
#### [replaced 038] Survey and Experiments on Mental Disorder Detection via Social Media: From Large Language Models and RAG to Agents
- **分类: cs.CL**

- **简介: 该论文聚焦社交媒体上的精神障碍检测任务，旨在解决大语言模型在临床应用中的幻觉与推理局限问题。通过综述与实验，系统评估了RAG和智能体技术对提升模型可靠性与自主推理能力的作用，并建立统一基准，推动可信赖AI在心理健康支持中的应用。**

- **链接: [https://arxiv.org/pdf/2504.02800v3](https://arxiv.org/pdf/2504.02800v3)**

> **作者:** Zhuohan Ge; Nicole Hu; Yubo Wang; Darian Li; Xinyi Zhu; Haoyang Li; Xin Zhang; Mingtao Zhang; Shihao Qi; Yuming Xu; Han Shi; Chen Jason Zhang; Qing Li
>
> **备注:** 20 pages, 10 figures. This is an extension of ICDEW 2025
>
> **摘要:** Mental disorders represent a critical global health challenge, and social media is increasingly viewed as a vital resource for real-time digital phenotyping and intervention. Large Language Models (LLMs) offer stronger semantic understanding and reasoning than traditional deep learning, but their use in high-stakes clinical settings is limited by hallucinations and the lack of persistent memory. However, existing literature has not sufficiently investigated how advanced enhancement techniques, specifically Retrieval-Augmented Generation (RAG) and Agentic systems, can address these reliability and reasoning limitations. Here, we systematically survey the evolving landscape of LLM-based methods for social media mental disorder analysis, spanning standard pretrained language models, RAG to mitigate hallucinations and contextual gaps, and agentic systems for autonomous reasoning and multi-step intervention. We organize existing work by technical paradigm and clinical target, extending beyond common internalizing disorders to include psychotic disorders and externalizing behaviors. Additionally, the paper comprehensively evaluates the performance of LLMs, including the impact of RAG, across various tasks. This work establishes a unified benchmark for the field, paving the way for the development of trustworthy, autonomous AI systems that can deliver precise and explainable mental health support.
>
---
#### [replaced 039] AraLingBench A Human-Annotated Benchmark for Evaluating Arabic Linguistic Capabilities of Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出AraLingBench，旨在评估大语言模型的阿拉伯语语言能力。针对当前模型表面流畅但深层语言理解不足的问题，构建了包含语法、形态等五类共150题的人工标注基准，揭示模型依赖记忆而非真正理解，为阿拉伯语LLM发展提供诊断工具。**

- **链接: [https://arxiv.org/pdf/2511.14295v2](https://arxiv.org/pdf/2511.14295v2)**

> **作者:** Mohammad Zbeeb; Hasan Abed Al Kader Hammoud; Sina Mukalled; Nadine Rizk; Fatima Karnib; Issam Lakkis; Ammar Mohanna; Bernard Ghanem
>
> **摘要:** We present AraLingBench: a fully human annotated benchmark for evaluating the Arabic linguistic competence of large language models (LLMs). The benchmark spans five core categories: grammar, morphology, spelling, reading comprehension, and syntax, through 150 expert-designed multiple choice questions that directly assess structural language understanding. Evaluating 35 Arabic and bilingual LLMs reveals that current models demonstrate strong surface level proficiency but struggle with deeper grammatical and syntactic reasoning. AraLingBench highlights a persistent gap between high scores on knowledge-based benchmarks and true linguistic mastery, showing that many models succeed through memorization or pattern recognition rather than authentic comprehension. By isolating and measuring fundamental linguistic skills, AraLingBench provides a diagnostic framework for developing Arabic LLMs. The full evaluation code is publicly available on GitHub.
>
---
#### [replaced 040] MixtureVitae: Open Web-Scale Pretraining Dataset With High Quality Instruction and Reasoning Data Built from Permissive-First Text Sources
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出MixtureVitae，构建高质量、法律风险低的开源预训练数据集。通过许可优先策略融合公有领域与合规文本，结合指令、推理和合成数据，实现高性能LLM训练，减少对无差别网络爬取的依赖。**

- **链接: [https://arxiv.org/pdf/2509.25531v2](https://arxiv.org/pdf/2509.25531v2)**

> **作者:** Huu Nguyen; Victor May; Harsh Raj; Marianna Nezhurina; Yishan Wang; Yanqi Luo; Minh Chien Vu; Taishi Nakamura; Ken Tsui; Van Khue Nguyen; David Salinas; Aleksandra Krasnodębska; Christoph Schuhmann; Mats Leon Richter; Xuan-Son; Vu; Jenia Jitsev
>
> **备注:** Code: \url{https://github.com/ontocord/mixturevitae}
>
> **摘要:** We present MixtureVitae, an open-access pretraining corpus built to minimize legal risk while providing strong model performance. MixtureVitae follows a risk-mitigated sourcing strategy that combines public-domain and permissively licensed text (e.g., CC-BY/Apache) with carefully justified low-risk additions (e.g., government works and EU TDM-eligible sources), alongside targeted instruction, reasoning and synthetic data with documented provenance. We detail a transparent, multi-stage pipeline for license-aware filtering, safety and quality screening, and domain-aware mixing, and we release the dataset and curation recipes to support reproducible research. In controlled experiments using the open-sci-ref training protocol (fixed architectures at 130M/400M/1.3B/1.7B parameters; training budgets of 50B and 300B tokens), models trained on MixtureVitae consistently outperform other permissive datasets across a suite of standard benchmarks, and at the 1.7B/300B setting they surpass FineWeb-Edu and approach DCLM in the later stages of training. Performance is particularly strong on math/code and competitive on QA tasks. These results demonstrate that permissive-first, risk-mitigated data provides a practical and legally mitigated foundation for training capable LLMs, reducing reliance on indiscriminate web scraping without sacrificing competitiveness. Code: https://github.com/ontocord/mixturevitae
>
---
#### [replaced 041] SENSE models: an open source solution for multilingual and multimodal semantic-based tasks
- **分类: cs.CL**

- **简介: 该论文提出SENSE模型，解决多语言多模态语义对齐问题。基于教师-学生框架，联合训练语音与文本编码器，实现跨模态语义共享。改进了文本教师模型和语音编码器初始化，并开源代码与模型，在多项任务中表现优异。**

- **链接: [https://arxiv.org/pdf/2509.12093v2](https://arxiv.org/pdf/2509.12093v2)**

> **作者:** Salima Mdhaffar; Haroun Elleuch; Chaimae Chellaf; Ha Nguyen; Yannick Estève
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** This paper introduces SENSE (Shared Embedding for N-lingual Speech and tExt), an open-source solution inspired by the SAMU-XLSR framework and conceptually similar to Meta AI's SONAR models. These approaches rely on a teacher-student framework to align a self-supervised speech encoder with the language-agnostic continuous representations of a text encoder at the utterance level. We describe how the original SAMU-XLSR method has been updated by selecting a stronger teacher text model and a better initial speech encoder. The source code for training and using SENSE models has been integrated into the SpeechBrain toolkit, and the first SENSE model we trained has been publicly released. We report experimental results on multilingual and multimodal semantic tasks, where our SENSE model achieves highly competitive performance. Finally, this study offers new insights into how semantics are captured in such semantically aligned speech encoders.
>
---
#### [replaced 042] TS-PEFT: Unveiling Token-Level Redundancy in Parameter-Efficient Fine-Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究参数高效微调（PEFT）中的冗余问题，提出TS-PEFT框架，通过近端优化动态识别并跳过冗余的token更新。实验证明其可减少40%-60%更新量，性能仍优于或媲美LoRA等方法，揭示token级稀疏性更能反映模块重要性。**

- **链接: [https://arxiv.org/pdf/2511.16147v2](https://arxiv.org/pdf/2511.16147v2)**

> **作者:** Dabiao Ma; Ziming Dai; Zhimin Xin; Shu Wang; Ye Wang; Haojun Fei
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Current Parameter-Efficient Fine-Tuning (PEFT) methods typically operate under an implicit assumption: once a target module is selected, every token passing through it contributes equally to the downstream task and requires a parameter update. In this paper, we challenge this convention and unveil a pervasive token-level redundancy in the fine-tuning of large models. We propose TS-PEFT, a theoretically grounded framework utilizing proximal optimization to dynamically identify and skip redundant token updates during training. Our extensive experiments across Natural Language Understanding, Commonsense Reasoning, and Visual Instruction Tuning demonstrate that indiscriminately updating all tokens is not only computationally superfluous but often introduces optimization noise. Strikingly, by discarding 40%-60% of token updates, TS-PEFT consistently matches or surpasses the performance of dense baselines (e.g., LoRA, DoRA). Furthermore, we provide an in-depth analysis revealing that the learned token-level sparsity serves as a superior indicator of module importance compared to traditional weight norms, offering a novel data-driven perspective on the intrinsic adaptation mechanism of large models.
>
---
