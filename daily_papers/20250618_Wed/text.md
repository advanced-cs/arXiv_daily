# 自然语言处理 cs.CL

- **最新发布 84 篇**

- **更新 76 篇**

## 最新发布

#### [new 001] Ace-CEFR -- A Dataset for Automated Evaluation of the Linguistic Difficulty of Conversational Texts for LLM Applications
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Ace-CEFR数据集，用于评估对话文本的语言难度，解决LLM训练中的文本难度评估问题。通过实验验证模型效果优于人类专家。**

- **链接: [http://arxiv.org/pdf/2506.14046v1](http://arxiv.org/pdf/2506.14046v1)**

> **作者:** David Kogan; Max Schumacher; Sam Nguyen; Masanori Suzuki; Melissa Smith; Chloe Sophia Bellows; Jared Bernstein
>
> **摘要:** There is an unmet need to evaluate the language difficulty of short, conversational passages of text, particularly for training and filtering Large Language Models (LLMs). We introduce Ace-CEFR, a dataset of English conversational text passages expert-annotated with their corresponding level of text difficulty. We experiment with several models on Ace-CEFR, including Transformer-based models and LLMs. We show that models trained on Ace-CEFR can measure text difficulty more accurately than human experts and have latency appropriate to production environments. Finally, we release the Ace-CEFR dataset to the public for research and development.
>
---
#### [new 002] Can we train ASR systems on Code-switch without real code-switch data? Case study for Singapore's languages
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决代码切换数据稀缺问题。通过生成合成数据提升ASR性能，验证了在无真实数据情况下的可行性。**

- **链接: [http://arxiv.org/pdf/2506.14177v1](http://arxiv.org/pdf/2506.14177v1)**

> **作者:** Tuan Nguyen; Huy-Dat Tran
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Code-switching (CS), common in multilingual settings, presents challenges for ASR due to scarce and costly transcribed data caused by linguistic complexity. This study investigates building CS-ASR using synthetic CS data. We propose a phrase-level mixing method to generate synthetic CS data that mimics natural patterns. Utilizing monolingual augmented with synthetic phrase-mixed CS data to fine-tune large pretrained ASR models (Whisper, MMS, SeamlessM4T). This paper focuses on three under-resourced Southeast Asian language pairs: Malay-English (BM-EN), Mandarin-Malay (ZH-BM), and Tamil-English (TA-EN), establishing a new comprehensive benchmark for CS-ASR to evaluate the performance of leading ASR models. Experimental results show that the proposed training strategy enhances ASR performance on monolingual and CS tests, with BM-EN showing highest gains, then TA-EN and ZH-BM. This finding offers a cost-effective approach for CS-ASR development, benefiting research and industry.
>
---
#### [new 003] AIn't Nothing But a Survey? Using Large Language Models for Coding German Open-Ended Survey Responses on Survey Motivation
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理中的文本分类任务，旨在解决如何用大语言模型编码德语开放式调查回复的问题。研究比较了不同LLMs和提示方法的效果。**

- **链接: [http://arxiv.org/pdf/2506.14634v1](http://arxiv.org/pdf/2506.14634v1)**

> **作者:** Leah von der Heyde; Anna-Carolina Haensch; Bernd Weiß; Jessika Daikeler
>
> **备注:** to appear in Survey Research Methods
>
> **摘要:** The recent development and wider accessibility of LLMs have spurred discussions about how they can be used in survey research, including classifying open-ended survey responses. Due to their linguistic capacities, it is possible that LLMs are an efficient alternative to time-consuming manual coding and the pre-training of supervised machine learning models. As most existing research on this topic has focused on English-language responses relating to non-complex topics or on single LLMs, it is unclear whether its findings generalize and how the quality of these classifications compares to established methods. In this study, we investigate to what extent different LLMs can be used to code open-ended survey responses in other contexts, using German data on reasons for survey participation as an example. We compare several state-of-the-art LLMs and several prompting approaches, and evaluate the LLMs' performance by using human expert codings. Overall performance differs greatly between LLMs, and only a fine-tuned LLM achieves satisfactory levels of predictive performance. Performance differences between prompting approaches are conditional on the LLM used. Finally, LLMs' unequal classification performance across different categories of reasons for survey participation results in different categorical distributions when not using fine-tuning. We discuss the implications of these findings, both for methodological research on coding open-ended responses and for their substantive analysis, and for practitioners processing or substantively analyzing such data. Finally, we highlight the many trade-offs researchers need to consider when choosing automated methods for open-ended response classification in the age of LLMs. In doing so, our study contributes to the growing body of research about the conditions under which LLMs can be efficiently, accurately, and reliably leveraged in survey research.
>
---
#### [new 004] Explainable Detection of Implicit Influential Patterns in Conversations via Data Augmentation
- **分类: cs.CL**

- **简介: 该论文属于对话中的隐性影响力检测任务，旨在识别嵌入对话中的隐性影响模式。通过数据增强提升模型检测能力，取得显著性能提升。**

- **链接: [http://arxiv.org/pdf/2506.14211v1](http://arxiv.org/pdf/2506.14211v1)**

> **作者:** Sina Abdidizaji; Md Kowsher; Niloofar Yousefi; Ivan Garibay
>
> **备注:** Accepted at the HCI International conference 2025
>
> **摘要:** In the era of digitalization, as individuals increasingly rely on digital platforms for communication and news consumption, various actors employ linguistic strategies to influence public perception. While models have become proficient at detecting explicit patterns, which typically appear in texts as single remarks referred to as utterances, such as social media posts, malicious actors have shifted toward utilizing implicit influential verbal patterns embedded within conversations. These verbal patterns aim to mentally penetrate the victim's mind in order to influence them, enabling the actor to obtain the desired information through implicit means. This paper presents an improved approach for detecting such implicit influential patterns. Furthermore, the proposed model is capable of identifying the specific locations of these influential elements within a conversation. To achieve this, the existing dataset was augmented using the reasoning capabilities of state-of-the-art language models. Our designed framework resulted in a 6% improvement in the detection of implicit influential patterns in conversations. Moreover, this approach improved the multi-label classification tasks related to both the techniques used for influence and the vulnerability of victims by 33% and 43%, respectively.
>
---
#### [new 005] Evaluation Should Not Ignore Variation: On the Impact of Reference Set Choice on Summarization Metrics
- **分类: cs.CL**

- **简介: 该论文属于文本摘要评估任务，旨在解决参考集选择对评估指标稳定性的影响问题。通过分析多个数据集，发现主流指标存在不稳定性，建议引入参考集变化以提高评估可靠性。**

- **链接: [http://arxiv.org/pdf/2506.14335v1](http://arxiv.org/pdf/2506.14335v1)**

> **作者:** Silvia Casola; Yang Janet Liu; Siyao Peng; Oliver Kraus; Albert Gatt; Barbara Plank
>
> **备注:** 17 pages, 13 figures
>
> **摘要:** Human language production exhibits remarkable richness and variation, reflecting diverse communication styles and intents. However, this variation is often overlooked in summarization evaluation. While having multiple reference summaries is known to improve correlation with human judgments, the impact of using different reference sets on reference-based metrics has not been systematically investigated. This work examines the sensitivity of widely used reference-based metrics in relation to the choice of reference sets, analyzing three diverse multi-reference summarization datasets: SummEval, GUMSum, and DUC2004. We demonstrate that many popular metrics exhibit significant instability. This instability is particularly concerning for n-gram-based metrics like ROUGE, where model rankings vary depending on the reference sets, undermining the reliability of model comparisons. We also collect human judgments on LLM outputs for genre-diverse data and examine their correlation with metrics to supplement existing findings beyond newswire summaries, finding weak-to-no correlation. Taken together, we recommend incorporating reference set variation into summarization evaluation to enhance consistency alongside correlation with human judgments, especially when evaluating LLMs.
>
---
#### [new 006] Abstract Meaning Representation for Hospital Discharge Summarization
- **分类: cs.CL**

- **简介: 该论文属于医疗文本生成任务，旨在解决自动生成出院摘要时的可信度问题，通过结合语言图与深度学习模型提升内容可靠性。**

- **链接: [http://arxiv.org/pdf/2506.14101v1](http://arxiv.org/pdf/2506.14101v1)**

> **作者:** Paul Landes; Sitara Rao; Aaron Jeremy Chaise; Barbara Di Eugenio
>
> **摘要:** The Achilles heel of Large Language Models (LLMs) is hallucination, which has drastic consequences for the clinical domain. This is particularly important with regards to automatically generating discharge summaries (a lengthy medical document that summarizes a hospital in-patient visit). Automatically generating these summaries would free physicians to care for patients and reduce documentation burden. The goal of this work is to discover new methods that combine language-based graphs and deep learning models to address provenance of content and trustworthiness in automatic summarization. Our method shows impressive reliability results on the publicly available Medical Information Mart for Intensive III (MIMIC-III) corpus and clinical notes written by physicians at Anonymous Hospital. rovide our method, generated discharge ary output examples, source code and trained models.
>
---
#### [new 007] Investigating the interaction of linguistic and mathematical reasoning in language models using multilingual number puzzles
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言数学推理任务，旨在解决语言模型在跨语言数字符号理解上的困难。通过实验分析语言与数学因素，发现模型需显式符号支持才能正确解答。**

- **链接: [http://arxiv.org/pdf/2506.13886v1](http://arxiv.org/pdf/2506.13886v1)**

> **作者:** Antara Raaghavi Bhattacharya; Isabel Papadimitriou; Kathryn Davidson; David Alvarez-Melis
>
> **摘要:** Across languages, numeral systems vary widely in how they construct and combine numbers. While humans consistently learn to navigate this diversity, large language models (LLMs) struggle with linguistic-mathematical puzzles involving cross-linguistic numeral systems, which humans can learn to solve successfully. We investigate why this task is difficult for LLMs through a series of experiments that untangle the linguistic and mathematical aspects of numbers in language. Our experiments establish that models cannot consistently solve such problems unless the mathematical operations in the problems are explicitly marked using known symbols ($+$, $\times$, etc, as in "twenty + three"). In further ablation studies, we probe how individual parameters of numeral construction and combination affect performance. While humans use their linguistic understanding of numbers to make inferences about the implicit compositional structure of numerals, LLMs seem to lack this notion of implicit numeral structure. We conclude that the ability to flexibly infer compositional rules from implicit patterns in human-scale data remains an open challenge for current reasoning models.
>
---
#### [new 008] From What to Respond to When to Respond: Timely Response Generation for Open-domain Dialogue Agents
- **分类: cs.CL**

- **简介: 该论文属于开放域对话生成任务，解决何时响应的问题。提出TimelyChat基准和Timer模型，通过时间预测生成及时响应。**

- **链接: [http://arxiv.org/pdf/2506.14285v1](http://arxiv.org/pdf/2506.14285v1)**

> **作者:** Seongbo Jang; Minjin Jeon; Jaehoon Lee; Seonghyeon Lee; Dongha Lee; Hwanjo Yu
>
> **备注:** Work in progress
>
> **摘要:** While research on dialogue response generation has primarily focused on generating coherent responses conditioning on textual context, the critical question of when to respond grounded on the temporal context remains underexplored. To bridge this gap, we propose a novel task called timely dialogue response generation and introduce the TimelyChat benchmark, which evaluates the capabilities of language models to predict appropriate time intervals and generate time-conditioned responses. Additionally, we construct a large-scale training dataset by leveraging unlabeled event knowledge from a temporal commonsense knowledge graph and employing a large language model (LLM) to synthesize 55K event-driven dialogues. We then train Timer, a dialogue agent designed to proactively predict time intervals and generate timely responses that align with those intervals. Experimental results show that Timer outperforms prompting-based LLMs and other fine-tuned baselines in both turn-level and dialogue-level evaluations. We publicly release our data, model, and code.
>
---
#### [new 009] LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在解决扩散语言模型的长上下文能力问题。通过分析与改进，提出LongLLaDA方法提升其上下文扩展性能。**

- **链接: [http://arxiv.org/pdf/2506.14429v1](http://arxiv.org/pdf/2506.14429v1)**

> **作者:** Xiaoran Liu; Zhigeng Liu; Zengfeng Huang; Qipeng Guo; Ziwei He; Xipeng Qiu
>
> **备注:** 16 pages, 12 figures, work in progress
>
> **摘要:** Large Language Diffusion Models, or diffusion LLMs, have emerged as a significant focus in NLP research, with substantial effort directed toward understanding their scalability and downstream task performance. However, their long-context capabilities remain unexplored, lacking systematic analysis or methods for context extension. In this work, we present the first systematic investigation comparing the long-context performance of diffusion LLMs and traditional auto-regressive LLMs. We first identify a unique characteristic of diffusion LLMs, unlike auto-regressive LLMs, they maintain remarkably \textbf{\textit{stable perplexity}} during direct context extrapolation. Furthermore, where auto-regressive models fail outright during the Needle-In-A-Haystack task with context exceeding their pretrained length, we discover diffusion LLMs exhibit a distinct \textbf{\textit{local perception}} phenomenon, enabling successful retrieval from recent context segments. We explain both phenomena through the lens of Rotary Position Embedding (RoPE) scaling theory. Building on these observations, we propose LongLLaDA, a training-free method that integrates LLaDA with the NTK-based RoPE extrapolation. Our results validate that established extrapolation scaling laws remain effective for extending the context windows of diffusion LLMs. Furthermore, we identify long-context tasks where diffusion LLMs outperform auto-regressive LLMs and others where they fall short. Consequently, this study establishes the first context extrapolation method for diffusion LLMs while providing essential theoretical insights and empirical benchmarks critical for advancing future research on long-context diffusion LLMs.
>
---
#### [new 010] Capacity Matters: a Proof-of-Concept for Transformer Memorization on Real-World Data
- **分类: cs.CL**

- **简介: 该论文属于生成模型研究，探讨Transformer在真实数据上的记忆能力。通过调整架构和数据配置，分析影响记忆容量的因素。**

- **链接: [http://arxiv.org/pdf/2506.14704v1](http://arxiv.org/pdf/2506.14704v1)**

> **作者:** Anton Changalidis; Aki Härmä
>
> **备注:** This work has been accepted for publication at the First Workshop on Large Language Model Memorization (L2M2) at ACL 2025, Vienna, Austria
>
> **摘要:** This paper studies how the model architecture and data configurations influence the empirical memorization capacity of generative transformers. The models are trained using synthetic text datasets derived from the Systematized Nomenclature of Medicine (SNOMED) knowledge graph: triplets, representing static connections, and sequences, simulating complex relation patterns. The results show that embedding size is the primary determinant of learning speed and capacity, while additional layers provide limited benefits and may hinder performance on simpler datasets. Activation functions play a crucial role, and Softmax demonstrates greater stability and capacity. Furthermore, increasing the complexity of the data set seems to improve the final memorization. These insights improve our understanding of transformer memory mechanisms and provide a framework for optimizing model design with structured real-world data.
>
---
#### [new 011] DCRM: A Heuristic to Measure Response Pair Quality in Preference Optimization
- **分类: cs.CL**

- **简介: 该论文属于偏好优化任务，旨在解决响应对质量评估问题。通过引入DCRM指标，衡量响应对质量，提升模型学习效果。**

- **链接: [http://arxiv.org/pdf/2506.14157v1](http://arxiv.org/pdf/2506.14157v1)**

> **作者:** Chengyu Huang; Tanya Goyal
>
> **摘要:** Recent research has attempted to associate preference optimization (PO) performance with the underlying preference datasets. In this work, our observation is that the differences between the preferred response $y^+$ and dispreferred response $y^-$ influence what LLMs can learn, which may not match the desirable differences to learn. Therefore, we use distance and reward margin to quantify these differences, and combine them to get Distance Calibrated Reward Margin (DCRM), a metric that measures the quality of a response pair for PO. Intuitively, DCRM encourages minimal noisy differences and maximal desired differences. With this, we study 3 types of commonly used preference datasets, classified along two axes: the source of the responses and the preference labeling function. We establish a general correlation between higher DCRM of the training set and better learning outcome. Inspired by this, we propose a best-of-$N^2$ pairing method that selects response pairs with the highest DCRM. Empirically, in various settings, our method produces training datasets that can further improve models' performance on AlpacaEval, MT-Bench, and Arena-Hard over the existing training sets.
>
---
#### [new 012] Thunder-NUBench: A Benchmark for LLMs' Sentence-Level Negation Understanding
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语义理解任务，旨在解决LLMs在句子级否定理解上的不足。工作包括构建Thunder-NUBench基准，评估模型对不同否定形式的理解能力。**

- **链接: [http://arxiv.org/pdf/2506.14397v1](http://arxiv.org/pdf/2506.14397v1)**

> **作者:** Yeonkyoung So; Gyuseong Lee; Sungmok Jung; Joonhak Lee; JiA Kang; Sangho Kim; Jaejin Lee
>
> **摘要:** Negation is a fundamental linguistic phenomenon that poses persistent challenges for Large Language Models (LLMs), particularly in tasks requiring deep semantic understanding. Existing benchmarks often treat negation as a side case within broader tasks like natural language inference, resulting in a lack of benchmarks that exclusively target negation understanding. In this work, we introduce \textbf{Thunder-NUBench}, a novel benchmark explicitly designed to assess sentence-level negation understanding in LLMs. Thunder-NUBench goes beyond surface-level cue detection by contrasting standard negation with structurally diverse alternatives such as local negation, contradiction, and paraphrase. The benchmark consists of manually curated sentence-negation pairs and a multiple-choice dataset that enables in-depth evaluation of models' negation understanding.
>
---
#### [new 013] Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent
- **分类: cs.CL**

- **简介: 该论文属于对话推荐任务，旨在解决多轮对话中用户满意度不足的问题。提出ECPO方法，通过期望确认理论优化用户偏好，提升推荐效果。**

- **链接: [http://arxiv.org/pdf/2506.14302v1](http://arxiv.org/pdf/2506.14302v1)**

> **作者:** Xueyang Feng; Jingsen Zhang; Jiakai Tang; Wei Li; Guohao Cai; Xu Chen; Quanyu Dai; Yue Zhu; Zhenhua Dong
>
> **备注:** Accepted to Findings of ACL 2025
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have significantly propelled the development of Conversational Recommendation Agents (CRAs). However, these agents often generate short-sighted responses that fail to sustain user guidance and meet expectations. Although preference optimization has proven effective in aligning LLMs with user expectations, it remains costly and performs poorly in multi-turn dialogue. To address this challenge, we introduce a novel multi-turn preference optimization (MTPO) paradigm ECPO, which leverages Expectation Confirmation Theory to explicitly model the evolution of user satisfaction throughout multi-turn dialogues, uncovering the underlying causes of dissatisfaction. These causes can be utilized to support targeted optimization of unsatisfactory responses, thereby achieving turn-level preference optimization. ECPO ingeniously eliminates the significant sampling overhead of existing MTPO methods while ensuring the optimization process drives meaningful improvements. To support ECPO, we introduce an LLM-based user simulator, AILO, to simulate user feedback and perform expectation confirmation during conversational recommendations. Experimental results show that ECPO significantly enhances CRA's interaction capabilities, delivering notable improvements in both efficiency and effectiveness over existing MTPO methods.
>
---
#### [new 014] ASMR: Augmenting Life Scenario using Large Generative Models for Robotic Action Reflection
- **分类: cs.CL; cs.AI; cs.RO**

- **简介: 该论文属于多模态分类任务，旨在解决机器人理解用户意图时数据不足的问题。通过生成对话和图像数据增强训练集，提升机器人动作选择能力。**

- **链接: [http://arxiv.org/pdf/2506.13956v1](http://arxiv.org/pdf/2506.13956v1)**

> **作者:** Shang-Chi Tsai; Seiya Kawano; Angel Garcia Contreras; Koichiro Yoshino; Yun-Nung Chen
>
> **备注:** IWSDS 2024 Best Paper Award
>
> **摘要:** When designing robots to assist in everyday human activities, it is crucial to enhance user requests with visual cues from their surroundings for improved intent understanding. This process is defined as a multimodal classification task. However, gathering a large-scale dataset encompassing both visual and linguistic elements for model training is challenging and time-consuming. To address this issue, our paper introduces a novel framework focusing on data augmentation in robotic assistance scenarios, encompassing both dialogues and related environmental imagery. This approach involves leveraging a sophisticated large language model to simulate potential conversations and environmental contexts, followed by the use of a stable diffusion model to create images depicting these environments. The additionally generated data serves to refine the latest multimodal models, enabling them to more accurately determine appropriate actions in response to user interactions with the limited target data. Our experimental results, based on a dataset collected from real-world scenarios, demonstrate that our methodology significantly enhances the robot's action selection capabilities, achieving the state-of-the-art performance.
>
---
#### [new 015] Re-Initialization Token Learning for Tool-Augmented Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于工具增强大语言模型任务，旨在解决工具与词嵌入空间不匹配的问题。通过重新初始化工具令牌，提升工具调用准确性。**

- **链接: [http://arxiv.org/pdf/2506.14248v1](http://arxiv.org/pdf/2506.14248v1)**

> **作者:** Chenghao Li; Liu Liu; Baosheng Yu; Jiayan Qiu; Yibing Zhan
>
> **摘要:** Large language models have demonstrated exceptional performance, yet struggle with complex tasks such as numerical reasoning, plan generation. Integrating external tools, such as calculators and databases, into large language models (LLMs) is crucial for enhancing problem-solving capabilities. Current methods assign a unique token to each tool, enabling LLMs to call tools through token prediction-similar to word generation. However, this approach fails to account for the relationship between tool and word tokens, limiting adaptability within pre-trained LLMs. To address this issue, we propose a novel token learning method that aligns tool tokens with the existing word embedding space from the perspective of initialization, thereby enhancing model performance. We begin by constructing prior token embeddings for each tool based on the tool's name or description, which are used to initialize and regularize the learnable tool token embeddings. This ensures the learned embeddings are well-aligned with the word token space, improving tool call accuracy. We evaluate the method on tasks such as numerical reasoning, knowledge-based question answering, and embodied plan generation using GSM8K-XL, FuncQA, KAMEL, and VirtualHome datasets. The results demonstrate clear improvements over recent baselines, including CoT, REACT, ICL, and ToolkenGPT, indicating that our approach effectively augments LLMs with tools through relevant tokens across diverse domains.
>
---
#### [new 016] Essential-Web v1.0: 24T tokens of organized web data
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Essential-Web v1.0数据集，解决语言模型预训练数据不足问题，通过自动标注和筛选获得高质量多领域数据。**

- **链接: [http://arxiv.org/pdf/2506.14111v1](http://arxiv.org/pdf/2506.14111v1)**

> **作者:** Essential AI; :; Andrew Hojel; Michael Pust; Tim Romanski; Yash Vanjani; Ritvik Kapila; Mohit Parmar; Adarsh Chaluvaraju; Alok Tripathy; Anil Thomas; Ashish Tanwer; Darsh J Shah; Ishaan Shah; Karl Stratos; Khoi Nguyen; Kurt Smith; Michael Callahan; Peter Rushton; Philip Monk; Platon Mazarakis; Saad Jamal; Saurabh Srivastava; Somanshu Singla; Ashish Vaswani
>
> **摘要:** Data plays the most prominent role in how language models acquire skills and knowledge. The lack of massive, well-organized pre-training datasets results in costly and inaccessible data pipelines. We present Essential-Web v1.0, a 24-trillion-token dataset in which every document is annotated with a twelve-category taxonomy covering topic, format, content complexity, and quality. Taxonomy labels are produced by EAI-Distill-0.5b, a fine-tuned 0.5b-parameter model that achieves an annotator agreement within 3% of Qwen2.5-32B-Instruct. With nothing more than SQL-style filters, we obtain competitive web-curated datasets in math (-8.0% relative to SOTA), web code (+14.3%), STEM (+24.5%) and medical (+8.6%). Essential-Web v1.0 is available on HuggingFace: https://huggingface.co/datasets/EssentialAI/essential-web-v1.0
>
---
#### [new 017] EmoNews: A Spoken Dialogue System for Expressive News Conversations
- **分类: cs.CL**

- **简介: 该论文属于任务导向的语音对话系统，旨在解决情感语音调节问题，提升新闻对话的共情效果。通过结合大语言模型和PromptTTS实现情感语音合成，并提出主观评估体系。**

- **链接: [http://arxiv.org/pdf/2506.13894v1](http://arxiv.org/pdf/2506.13894v1)**

> **作者:** Ryuki Matsuura; Shikhar Bharadwaj; Jiarui Liu; Dhatchi Kunde Govindarajan
>
> **摘要:** We develop a task-oriented spoken dialogue system (SDS) that regulates emotional speech based on contextual cues to enable more empathetic news conversations. Despite advancements in emotional text-to-speech (TTS) techniques, task-oriented emotional SDSs remain underexplored due to the compartmentalized nature of SDS and emotional TTS research, as well as the lack of standardized evaluation metrics for social goals. We address these challenges by developing an emotional SDS for news conversations that utilizes a large language model (LLM)-based sentiment analyzer to identify appropriate emotions and PromptTTS to synthesize context-appropriate emotional speech. We also propose subjective evaluation scale for emotional SDSs and judge the emotion regulation performance of the proposed and baseline systems. Experiments showed that our emotional SDS outperformed a baseline system in terms of the emotion regulation and engagement. These results suggest the critical role of speech emotion for more engaging conversations. All our source code is open-sourced at https://github.com/dhatchi711/espnet-emotional-news/tree/emo-sds/egs2/emo_news_sds/sds1
>
---
#### [new 018] Reasoning with Exploration: An Entropy Perspective
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决语言模型推理中探索不足的问题。通过引入熵机制增强推理深度，提升模型表现。**

- **链接: [http://arxiv.org/pdf/2506.14758v1](http://arxiv.org/pdf/2506.14758v1)**

> **作者:** Daixuan Cheng; Shaohan Huang; Xuekai Zhu; Bo Dai; Wayne Xin Zhao; Zhenliang Zhang; Furu Wei
>
> **摘要:** Balancing exploration and exploitation is a central goal in reinforcement learning (RL). Despite recent advances in enhancing language model (LM) reasoning, most methods lean toward exploitation, and increasingly encounter performance plateaus. In this work, we revisit entropy -- a signal of exploration in RL -- and examine its relationship to exploratory reasoning in LMs. Through empirical analysis, we uncover strong positive correlations between high-entropy regions and three types of exploratory reasoning actions: (1) pivotal tokens that determine or connect logical steps, (2) reflective actions such as self-verification and correction, and (3) rare behaviors under-explored by the base LMs. Motivated by this, we introduce a minimal modification to standard RL with only one line of code: augmenting the advantage function with an entropy-based term. Unlike traditional maximum-entropy methods which encourage exploration by promoting uncertainty, we encourage exploration by promoting longer and deeper reasoning chains. Notably, our method achieves significant gains on the Pass@K metric -- an upper-bound estimator of LM reasoning capabilities -- even when evaluated with extremely large K values, pushing the boundaries of LM reasoning.
>
---
#### [new 019] Probabilistic Aggregation and Targeted Embedding Optimization for Collective Moral Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于道德推理任务，旨在解决LLM在复杂道德困境中的分歧问题。通过概率聚合和嵌入优化，提升模型的一致性和准确性。**

- **链接: [http://arxiv.org/pdf/2506.14625v1](http://arxiv.org/pdf/2506.14625v1)**

> **作者:** Chenchen Yuan; Zheyu Zhang; Shuo Yang; Bardh Prenkaj; Gjergji Kasneci
>
> **备注:** 18 pages
>
> **摘要:** Large Language Models (LLMs) have shown impressive moral reasoning abilities. Yet they often diverge when confronted with complex, multi-factor moral dilemmas. To address these discrepancies, we propose a framework that synthesizes multiple LLMs' moral judgments into a collectively formulated moral judgment, realigning models that deviate significantly from this consensus. Our aggregation mechanism fuses continuous moral acceptability scores (beyond binary labels) into a collective probability, weighting contributions by model reliability. For misaligned models, a targeted embedding-optimization procedure fine-tunes token embeddings for moral philosophical theories, minimizing JS divergence to the consensus while preserving semantic integrity. Experiments on a large-scale social moral dilemma dataset show our approach builds robust consensus and improves individual model fidelity. These findings highlight the value of data-driven moral alignment across multiple models and its potential for safer, more consistent AI systems.
>
---
#### [new 020] GenerationPrograms: Fine-grained Attribution with Executable Programs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决模型输出缺乏细粒度溯源的问题。提出GenerationPrograms框架，通过分阶段生成可执行程序提升可解释性和准确性。**

- **链接: [http://arxiv.org/pdf/2506.14580v1](http://arxiv.org/pdf/2506.14580v1)**

> **作者:** David Wan; Eran Hirsch; Elias Stengel-Eskin; Ido Dagan; Mohit Bansal
>
> **备注:** 27 Pages. Code: https://github.com/meetdavidwan/generationprograms
>
> **摘要:** Recent large language models (LLMs) achieve impressive performance in source-conditioned text generation but often fail to correctly provide fine-grained attributions for their outputs, undermining verifiability and trust. Moreover, existing attribution methods do not explain how and why models leverage the provided source documents to generate their final responses, limiting interpretability. To overcome these challenges, we introduce a modular generation framework, GenerationPrograms, inspired by recent advancements in executable "code agent" architectures. Unlike conventional generation methods that simultaneously generate outputs and attributions or rely on post-hoc attribution, GenerationPrograms decomposes the process into two distinct stages: first, creating an executable program plan composed of modular text operations (such as paraphrasing, compression, and fusion) explicitly tailored to the query, and second, executing these operations following the program's specified instructions to produce the final response. Empirical evaluations demonstrate that GenerationPrograms significantly improves attribution quality at both the document level and sentence level across two long-form question-answering tasks and a multi-document summarization task. We further demonstrate that GenerationPrograms can effectively function as a post-hoc attribution method, outperforming traditional techniques in recovering accurate attributions. In addition, the interpretable programs generated by GenerationPrograms enable localized refinement through modular-level improvements that further enhance overall attribution quality.
>
---
#### [new 021] GuiLoMo: Allocating Expert Number and Rank for LoRA-MoE via Bilevel Optimization with GuidedSelection Vectors
- **分类: cs.CL**

- **简介: 该论文属于参数高效微调任务，解决LoRA-MoE中专家数量和秩分配不合理的问题，提出GuiLoMo方法通过双层优化和引导选择向量实现自适应配置。**

- **链接: [http://arxiv.org/pdf/2506.14646v1](http://arxiv.org/pdf/2506.14646v1)**

> **作者:** Hengyuan Zhang; Xinrong Chen; Yingmin Qiu; Xiao Liang; Ziyue Li; Guanyu Wang; Weiping Li; Tong Mo; Wenyue Li; Hayden Kwok-Hay So; Ngai Wong
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) methods, particularly Low-Rank Adaptation (LoRA), offer an efficient way to adapt large language models with reduced computational costs. However, their performance is limited by the small number of trainable parameters. Recent work combines LoRA with the Mixture-of-Experts (MoE), i.e., LoRA-MoE, to enhance capacity, but two limitations remain in hindering the full exploitation of its potential: 1) the influence of downstream tasks when assigning expert numbers, and 2) the uniform rank assignment across all LoRA experts, which restricts representational diversity. To mitigate these gaps, we propose GuiLoMo, a fine-grained layer-wise expert numbers and ranks allocation strategy with GuidedSelection Vectors (GSVs). GSVs are learned via a prior bilevel optimization process to capture both model- and task-specific needs, and are then used to allocate optimal expert numbers and ranks. Experiments on three backbone models across diverse benchmarks show that GuiLoMo consistently achieves superior or comparable performance to all baselines. Further analysis offers key insights into how expert numbers and ranks vary across layers and tasks, highlighting the benefits of adaptive expert configuration. Our code is available at https://github.com/Liar406/Gui-LoMo.git.
>
---
#### [new 022] From Bytes to Ideas: Language Modeling with Autoregressive U-Nets
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种自回归U-Net语言模型，解决传统tokenization固定粒度的问题。通过动态嵌入tokens，实现多尺度序列处理，提升语义理解。**

- **链接: [http://arxiv.org/pdf/2506.14761v1](http://arxiv.org/pdf/2506.14761v1)**

> **作者:** Mathurin Videau; Badr Youbi Idrissi; Alessandro Leite; Marc Schoenauer; Olivier Teytaud; David Lopez-Paz
>
> **摘要:** Tokenization imposes a fixed granularity on the input text, freezing how a language model operates on data and how far in the future it predicts. Byte Pair Encoding (BPE) and similar schemes split text once, build a static vocabulary, and leave the model stuck with that choice. We relax this rigidity by introducing an autoregressive U-Net that learns to embed its own tokens as it trains. The network reads raw bytes, pools them into words, then pairs of words, then up to 4 words, giving it a multi-scale view of the sequence. At deeper stages, the model must predict further into the future -- anticipating the next few words rather than the next byte -- so deeper stages focus on broader semantic patterns while earlier stages handle fine details. When carefully tuning and controlling pretraining compute, shallow hierarchies tie strong BPE baselines, and deeper hierarchies have a promising trend. Because tokenization now lives inside the model, the same system can handle character-level tasks and carry knowledge across low-resource languages.
>
---
#### [new 023] Lost in the Mix: Evaluating LLM Understanding of Code-Switched Text
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLM在代码混杂文本中的理解能力，通过生成测试用例评估其表现，并探索提升方法。**

- **链接: [http://arxiv.org/pdf/2506.14012v1](http://arxiv.org/pdf/2506.14012v1)**

> **作者:** Amr Mohamed; Yang Zhang; Michalis Vazirgiannis; Guokan Shang
>
> **摘要:** Code-switching (CSW) is the act of alternating between two or more languages within a single discourse. This phenomenon is widespread in multilingual communities, and increasingly prevalent in online content, where users naturally mix languages in everyday communication. As a result, Large Language Models (LLMs), now central to content processing and generation, are frequently exposed to code-switched inputs. Given their widespread use, it is crucial to understand how LLMs process and reason about such mixed-language text. This paper presents a systematic evaluation of LLM comprehension under code-switching by generating CSW variants of established reasoning and comprehension benchmarks. While degradation is evident when foreign tokens disrupt English text$\unicode{x2013}$even under linguistic constraints$\unicode{x2013}$embedding English into other languages often improves comprehension. Though prompting yields mixed results, fine-tuning offers a more stable path to degradation mitigation.
>
---
#### [new 024] S$^4$C: Speculative Sampling with Syntactic and Semantic Coherence for Efficient Inference of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理延迟问题。提出S⁴C框架，通过语法和语义一致性提升采样效率。**

- **链接: [http://arxiv.org/pdf/2506.14158v1](http://arxiv.org/pdf/2506.14158v1)**

> **作者:** Tao He; Guang Huang; Yu Yang; Tianshi Xu; Sicheng Zhao; Guiguang Ding; Pengyang Wang; Feng Tian
>
> **摘要:** Large language models (LLMs) exhibit remarkable reasoning capabilities across diverse downstream tasks. However, their autoregressive nature leads to substantial inference latency, posing challenges for real-time applications. Speculative sampling mitigates this issue by introducing a drafting phase followed by a parallel validation phase, enabling faster token generation and verification. Existing approaches, however, overlook the inherent coherence in text generation, limiting their efficiency. To address this gap, we propose a Speculative Sampling with Syntactic and Semantic Coherence (S$^4$C) framework, which extends speculative sampling by leveraging multi-head drafting for rapid token generation and a continuous verification tree for efficient candidate validation and feature reuse. Experimental results demonstrate that S$^4$C surpasses baseline methods across mainstream tasks, offering enhanced efficiency, parallelism, and the ability to generate more valid tokens with fewer computational resources. On Spec-bench benchmarks, S$^4$C achieves an acceleration ratio of 2.26x-2.60x, outperforming state-of-the-art methods.
>
---
#### [new 025] Are manual annotations necessary for statutory interpretations retrieval?
- **分类: cs.CL**

- **简介: 该论文属于法律信息检索任务，旨在探讨手动标注在法律概念解释检索中的必要性。研究分析了标注数量、样本选择及自动化标注的效果。**

- **链接: [http://arxiv.org/pdf/2506.13965v1](http://arxiv.org/pdf/2506.13965v1)**

> **作者:** Aleksander Smywiński-Pohl; Tomer Libal; Adam Kaczmarczyk; Magdalena Król
>
> **摘要:** One of the elements of legal research is looking for cases where judges have extended the meaning of a legal concept by providing interpretations of what a concept means or does not mean. This allow legal professionals to use such interpretations as precedents as well as laymen to better understand the legal concept. The state-of-the-art approach for retrieving the most relevant interpretations for these concepts currently depends on the ranking of sentences and the training of language models over annotated examples. That manual annotation process can be quite expensive and need to be repeated for each such concept, which prompted recent research in trying to automate this process. In this paper, we highlight the results of various experiments conducted to determine the volume, scope and even the need for manual annotation. First of all, we check what is the optimal number of annotations per a legal concept. Second, we check if we can draw the sentences for annotation randomly or there is a gain in the performance of the model, when only the best candidates are annotated. As the last question we check what is the outcome of automating the annotation process with the help of an LLM.
>
---
#### [new 026] How Far Can LLMs Improve from Experience? Measuring Test-Time Learning Ability in LLMs with Human Comparison
- **分类: cs.CL**

- **简介: 该论文属于评估任务，旨在研究大语言模型在测试时的学习能力。通过设计语义游戏，对比模型与人类在不同经验下的表现，揭示模型学习效率的不足。**

- **链接: [http://arxiv.org/pdf/2506.14448v1](http://arxiv.org/pdf/2506.14448v1)**

> **作者:** Jiayin Wang; Zhiquang Guo; Weizhi Ma; Min Zhang
>
> **摘要:** As evaluation designs of large language models may shape our trajectory toward artificial general intelligence, comprehensive and forward-looking assessment is essential. Existing benchmarks primarily assess static knowledge, while intelligence also entails the ability to rapidly learn from experience. To this end, we advocate for the evaluation of Test-time Learning, the capacity to improve performance in experience-based, reasoning-intensive tasks during test time. In this work, we propose semantic games as effective testbeds for evaluating test-time learning, due to their resistance to saturation and inherent demand for strategic reasoning. We introduce an objective evaluation framework that compares model performance under both limited and cumulative experience settings, and contains four forms of experience representation. To provide a comparative baseline, we recruit eight human participants to complete the same task. Results show that LLMs exhibit measurable test-time learning capabilities; however, their improvements are less stable under cumulative experience and progress more slowly than those observed in humans. These findings underscore the potential of LLMs as general-purpose learning machines, while also revealing a substantial intellectual gap between models and humans, irrespective of how well LLMs perform on static benchmarks.
>
---
#### [new 027] Massive Supervised Fine-tuning Experiments Reveal How Data, Layer, and Training Factors Shape LLM Alignment Quality
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究如何通过监督微调提升大模型对齐质量。工作包括训练1000+模型，分析数据、层和训练因素的影响，发现困惑度与效果相关性高。**

- **链接: [http://arxiv.org/pdf/2506.14681v1](http://arxiv.org/pdf/2506.14681v1)**

> **作者:** Yuto Harada; Yusuke Yamauchi; Yusuke Oda; Yohei Oseki; Yusuke Miyao; Yu Takagi
>
> **摘要:** Supervised fine-tuning (SFT) is a critical step in aligning large language models (LLMs) with human instructions and values, yet many aspects of SFT remain poorly understood. We trained a wide range of base models on a variety of datasets including code generation, mathematical reasoning, and general-domain tasks, resulting in 1,000+ SFT models under controlled conditions. We then identified the dataset properties that matter most and examined the layer-wise modifications introduced by SFT. Our findings reveal that some training-task synergies persist across all models while others vary substantially, emphasizing the importance of model-specific strategies. Moreover, we demonstrate that perplexity consistently predicts SFT effectiveness--often surpassing superficial similarity between trained data and benchmark--and that mid-layer weight changes correlate most strongly with performance gains. We will release these 1,000+ SFT models and benchmark results to accelerate further research.
>
---
#### [new 028] AlphaDecay:Module-wise Weight Decay for Heavy-Tailed Balancing in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于深度学习模型优化任务，旨在解决LLMs中权重衰减策略单一的问题。通过模块化自适应衰减方法AlphaDecay，提升模型性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.14562v1](http://arxiv.org/pdf/2506.14562v1)**

> **作者:** Di He; Ajay Jaiswal; Songjun Tu; Li Shen; Ganzhao Yuan; Shiwei Liu; Lu Yin
>
> **摘要:** Weight decay is a standard regularization technique for training large language models (LLMs). While it is common to assign a uniform decay rate to every layer, this approach overlooks the structural diversity of LLMs and the varying spectral properties across modules. In this paper, we introduce AlphaDecay, a simple yet effective method that adaptively assigns different weight decay strengths to each module of an LLM. Our approach is guided by Heavy-Tailed Self-Regularization (HT-SR) theory, which analyzes the empirical spectral density (ESD) of weight correlation matrices to quantify "heavy-tailedness." Modules exhibiting more pronounced heavy-tailed ESDs, reflecting stronger feature learning, are assigned weaker decay, while modules with lighter-tailed spectra receive stronger decay. Our method leverages tailored weight decay assignments to balance the module-wise differences in spectral properties, leading to improved performance. Extensive pre-training tasks with various model sizes from 60M to 1B demonstrate that AlphaDecay achieves better perplexity and generalization than conventional uniform decay and other adaptive decay baselines.
>
---
#### [new 029] Chaining Event Spans for Temporal Relation Grounding
- **分类: cs.CL**

- **简介: 该论文属于时间关系理解任务，旨在解决因答案重叠导致的时序关系判断错误问题。通过引入TRN模型，利用事件时间跨度预测提升时序推理准确性。**

- **链接: [http://arxiv.org/pdf/2506.14213v1](http://arxiv.org/pdf/2506.14213v1)**

> **作者:** Jongho Kim; Dohyeon Lee; Minsoo Kim; Seung-won Hwang
>
> **备注:** In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1689-1700
>
> **摘要:** Accurately understanding temporal relations between events is a critical building block of diverse tasks, such as temporal reading comprehension (TRC) and relation extraction (TRE). For example in TRC, we need to understand the temporal semantic differences between the following two questions that are lexically near-identical: "What finished right before the decision?" or "What finished right after the decision?". To discern the two questions, existing solutions have relied on answer overlaps as a proxy label to contrast similar and dissimilar questions. However, we claim that answer overlap can lead to unreliable results, due to spurious overlaps of two dissimilar questions with coincidentally identical answers. To address the issue, we propose a novel approach that elicits proper reasoning behaviors through a module for predicting time spans of events. We introduce the Timeline Reasoning Network (TRN) operating in a two-step inductive reasoning process: In the first step model initially answers each question with semantic and syntactic information. The next step chains multiple questions on the same event to predict a timeline, which is then used to ground the answers. Results on the TORQUE and TB-dense, TRC and TRE tasks respectively, demonstrate that TRN outperforms previous methods by effectively resolving the spurious overlaps using the predicted timeline.
>
---
#### [new 030] Alignment Quality Index (AQI) : Beyond Refusals: AQI as an Intrinsic Alignment Diagnostic via Latent Geometry, Cluster Divergence, and Layer wise Pooled Representations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型对齐评估任务，旨在解决现有评估方法的不足。提出AQI指标，通过分析潜在空间聚类检测对齐风险，提升安全审计能力。**

- **链接: [http://arxiv.org/pdf/2506.13901v1](http://arxiv.org/pdf/2506.13901v1)**

> **作者:** Abhilekh Borah; Chhavi Sharma; Danush Khanna; Utkarsh Bhatt; Gurpreet Singh; Hasnat Md Abdullah; Raghav Kaushik Ravi; Vinija Jain; Jyoti Patel; Shubham Singh; Vasu Sharma; Arpita Vats; Rahul Raja; Aman Chadha; Amitava Das
>
> **摘要:** Alignment is no longer a luxury, it is a necessity. As large language models (LLMs) enter high-stakes domains like education, healthcare, governance, and law, their behavior must reliably reflect human-aligned values and safety constraints. Yet current evaluations rely heavily on behavioral proxies such as refusal rates, G-Eval scores, and toxicity classifiers, all of which have critical blind spots. Aligned models are often vulnerable to jailbreaking, stochasticity of generation, and alignment faking. To address this issue, we introduce the Alignment Quality Index (AQI). This novel geometric and prompt-invariant metric empirically assesses LLM alignment by analyzing the separation of safe and unsafe activations in latent space. By combining measures such as the Davies-Bouldin Score (DBS), Dunn Index (DI), Xie-Beni Index (XBI), and Calinski-Harabasz Index (CHI) across various formulations, AQI captures clustering quality to detect hidden misalignments and jailbreak risks, even when outputs appear compliant. AQI also serves as an early warning signal for alignment faking, offering a robust, decoding invariant tool for behavior agnostic safety auditing. Additionally, we propose the LITMUS dataset to facilitate robust evaluation under these challenging conditions. Empirical tests on LITMUS across different models trained under DPO, GRPO, and RLHF conditions demonstrate AQI's correlation with external judges and ability to reveal vulnerabilities missed by refusal metrics. We make our implementation publicly available to foster future research in this area.
>
---
#### [new 031] Revisiting Chain-of-Thought Prompting: Zero-shot Can Be Stronger than Few-shot
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的数学推理任务，探讨CoT在ICL中的有效性。研究发现，强模型使用Zero-Shot CoT效果更优，传统CoT示例无显著提升。**

- **链接: [http://arxiv.org/pdf/2506.14641v1](http://arxiv.org/pdf/2506.14641v1)**

> **作者:** Xiang Cheng; Chengyan Pan; Minjun Zhao; Deyang Li; Fangchao Liu; Xinyu Zhang; Xiao Zhang; Yong Liu
>
> **备注:** 19 pages,22 figures
>
> **摘要:** In-Context Learning (ICL) is an essential emergent ability of Large Language Models (LLMs), and recent studies introduce Chain-of-Thought (CoT) to exemplars of ICL to enhance the reasoning capability, especially in mathematics tasks. However, given the continuous advancement of model capabilities, it remains unclear whether CoT exemplars still benefit recent, stronger models in such tasks. Through systematic experiments, we find that for recent strong models such as the Qwen2.5 series, adding traditional CoT exemplars does not improve reasoning performance compared to Zero-Shot CoT. Instead, their primary function is to align the output format with human expectations. We further investigate the effectiveness of enhanced CoT exemplars, constructed using answers from advanced models such as \texttt{Qwen2.5-Max} and \texttt{DeepSeek-R1}. Experimental results indicate that these enhanced exemplars still fail to improve the model's reasoning performance. Further analysis reveals that models tend to ignore the exemplars and focus primarily on the instructions, leading to no observable gain in reasoning ability. Overall, our findings highlight the limitations of the current ICL+CoT framework in mathematical reasoning, calling for a re-examination of the ICL paradigm and the definition of exemplars.
>
---
#### [new 032] AI shares emotion with humans across languages and cultures
- **分类: cs.CL**

- **简介: 该论文属于情感计算任务，旨在解决AI与人类情感对齐问题。通过分析LLM的情感表示，验证其与人类情感的一致性，并展示如何用情感概念引导AI生成特定情绪输出。**

- **链接: [http://arxiv.org/pdf/2506.13978v1](http://arxiv.org/pdf/2506.13978v1)**

> **作者:** Xiuwen Wu; Hao Wang; Zhiang Yan; Xiaohan Tang; Pengfei Xu; Wai-Ting Siok; Ping Li; Jia-Hong Gao; Bingjiang Lyu; Lang Qin
>
> **摘要:** Effective and safe human-machine collaboration requires the regulated and meaningful exchange of emotions between humans and artificial intelligence (AI). Current AI systems based on large language models (LLMs) can provide feedback that makes people feel heard. Yet it remains unclear whether LLMs represent emotion in language as humans do, or whether and how the emotional tone of their output can be controlled. We assess human-AI emotional alignment across linguistic-cultural groups and model-families, using interpretable LLM features translated from concept-sets for over twenty nuanced emotion categories (including six basic emotions). Our analyses reveal that LLM-derived emotion spaces are structurally congruent with human perception, underpinned by the fundamental affective dimensions of valence and arousal. Furthermore, these emotion-related features also accurately predict large-scale behavioural data on word ratings along these two core dimensions, reflecting both universal and language-specific patterns. Finally, by leveraging steering vectors derived solely from human-centric emotion concepts, we show that model expressions can be stably and naturally modulated across distinct emotion categories, which provides causal evidence that human emotion concepts can be used to systematically induce LLMs to produce corresponding affective states when conveying content. These findings suggest AI not only shares emotional representations with humans but its affective outputs can be precisely guided using psychologically grounded emotion concepts.
>
---
#### [new 033] ELLIS Alicante at CQs-Gen 2025: Winning the critical thinking questions shared task: LLM-based question generation and selection
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于自动批判性问题生成任务，旨在解决LLM在促进深度思考中的应用问题。通过构建问答框架，生成并筛选高质量问题，提升对论点的批判性分析能力。**

- **链接: [http://arxiv.org/pdf/2506.14371v1](http://arxiv.org/pdf/2506.14371v1)**

> **作者:** Lucile Favero; Daniel Frases; Juan Antonio Pérez-Ortiz; Tanja Käser; Nuria Oliver
>
> **备注:** Proceedings of the 12th Workshop on Argument Mining
>
> **摘要:** The widespread adoption of chat interfaces based on Large Language Models (LLMs) raises concerns about promoting superficial learning and undermining the development of critical thinking skills. Instead of relying on LLMs purely for retrieving factual information, this work explores their potential to foster deeper reasoning by generating critical questions that challenge unsupported or vague claims in debate interventions. This study is part of a shared task of the 12th Workshop on Argument Mining, co-located with ACL 2025, focused on automatic critical question generation. We propose a two-step framework involving two small-scale open source language models: a Questioner that generates multiple candidate questions and a Judge that selects the most relevant ones. Our system ranked first in the shared task competition, demonstrating the potential of the proposed LLM-based approach to encourage critical engagement with argumentative texts.
>
---
#### [new 034] ELI-Why: Evaluating the Pedagogical Utility of Language Model Explanations
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于教育AI任务，旨在评估语言模型解释的教育适用性。通过构建基准数据集和用户研究，发现模型生成解释在适配不同教育水平上效果有限。**

- **链接: [http://arxiv.org/pdf/2506.14200v1](http://arxiv.org/pdf/2506.14200v1)**

> **作者:** Brihi Joshi; Keyu He; Sahana Ramnath; Sadra Sabouri; Kaitlyn Zhou; Souti Chattopadhyay; Swabha Swayamdipta; Xiang Ren
>
> **备注:** Findings of ACL 2025
>
> **摘要:** Language models today are widely used in education, yet their ability to tailor responses for learners with varied informational needs and knowledge backgrounds remains under-explored. To this end, we introduce ELI-Why, a benchmark of 13.4K "Why" questions to evaluate the pedagogical capabilities of language models. We then conduct two extensive human studies to assess the utility of language model-generated explanatory answers (explanations) on our benchmark, tailored to three distinct educational grades: elementary, high-school and graduate school. In our first study, human raters assume the role of an "educator" to assess model explanations' fit to different educational grades. We find that GPT-4-generated explanations match their intended educational background only 50% of the time, compared to 79% for lay human-curated explanations. In our second study, human raters assume the role of a learner to assess if an explanation fits their own informational needs. Across all educational backgrounds, users deemed GPT-4-generated explanations 20% less suited on average to their informational needs, when compared to explanations curated by lay people. Additionally, automated evaluation metrics reveal that explanations generated across different language model families for different informational needs remain indistinguishable in their grade-level, limiting their pedagogical effectiveness.
>
---
#### [new 035] AsyncSwitch: Asynchronous Text-Speech Adaptation for Code-Switched ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，解决代码转换ASR中的语言歧义和数据不足问题。通过AsyncSwitch框架，利用文本数据预训练并微调模型，提升识别效果。**

- **链接: [http://arxiv.org/pdf/2506.14190v1](http://arxiv.org/pdf/2506.14190v1)**

> **作者:** Tuan Nguyen; Huy-Dat Tran
>
> **备注:** This work has been submitted to the IEEE for possible publication. This paper is a preprint version submitted to the 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU 2025)
>
> **摘要:** Developing code-switched ASR systems is challenging due to language ambiguity and limited exposure to multilingual, code-switched data, while collecting such speech is costly. Prior work generates synthetic audio from text, but these methods are computationally intensive and hard to scale. We introduce AsyncSwitch, a novel asynchronous adaptation framework that leverages large-scale, text-rich web data to pre-expose ASR models to diverse code-switched domains before fine-tuning on paired speech-text corpora. Our three-stage process (1) trains decoder self-attention and feedforward layers on code-switched text, (2) aligns decoder and encoder via cross-attention using limited speech-text data, and (3) fully fine-tunes the entire model. Experiments with Whisper on Malay-English code-switching demonstrate a 9.02% relative WER reduction, while improving monolingual performance in Singlish, Malay, and other English variants.
>
---
#### [new 036] Guaranteed Guess: A Language Modeling Approach for CISC-to-RISC Transpilation with Testing Guarantees
- **分类: cs.CL; cs.AR; cs.LG; cs.PL; cs.SE**

- **简介: 该论文属于CISC到RISC指令集架构翻译任务，旨在提高代码移植的正确性和效率。通过结合大语言模型与软件测试，生成并验证翻译结果。**

- **链接: [http://arxiv.org/pdf/2506.14606v1](http://arxiv.org/pdf/2506.14606v1)**

> **作者:** Ahmed Heakl; Sarim Hashmi; Chaimaa Abi; Celine Lee; Abdulrahman Mahmoud
>
> **备注:** Project page: https://ahmedheakl.github.io/Guaranteed-Guess/
>
> **摘要:** The hardware ecosystem is rapidly evolving, with increasing interest in translating low-level programs across different instruction set architectures (ISAs) in a quick, flexible, and correct way to enhance the portability and longevity of existing code. A particularly challenging class of this transpilation problem is translating between complex- (CISC) and reduced- (RISC) hardware architectures, due to fundamental differences in instruction complexity, memory models, and execution paradigms. In this work, we introduce GG (Guaranteed Guess), an ISA-centric transpilation pipeline that combines the translation power of pre-trained large language models (LLMs) with the rigor of established software testing constructs. Our method generates candidate translations using an LLM from one ISA to another, and embeds such translations within a software-testing framework to build quantifiable confidence in the translation. We evaluate our GG approach over two diverse datasets, enforce high code coverage (>98%) across unit tests, and achieve functional/semantic correctness of 99% on HumanEval programs and 49% on BringupBench programs, respectively. Further, we compare our approach to the state-of-the-art Rosetta 2 framework on Apple Silicon, showcasing 1.73x faster runtime performance, 1.47x better energy efficiency, and 2.41x better memory usage for our transpiled code, demonstrating the effectiveness of GG for real-world CISC-to-RISC translation tasks. We will open-source our codes, data, models, and benchmarks to establish a common foundation for ISA-level code translation research.
>
---
#### [new 037] Treasure Hunt: Real-time Targeting of the Long Tail using Training-Time Markers
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决模型在长尾罕见场景下的性能问题。通过训练时标记优化推理控制，提升模型对低频任务的适应能力。**

- **链接: [http://arxiv.org/pdf/2506.14702v1](http://arxiv.org/pdf/2506.14702v1)**

> **作者:** Daniel D'souza; Julia Kreutzer; Adrien Morisot; Ahmet Üstün; Sara Hooker
>
> **摘要:** One of the most profound challenges of modern machine learning is performing well on the long-tail of rare and underrepresented features. Large general-purpose models are trained for many tasks, but work best on high-frequency use cases. After training, it is hard to adapt a model to perform well on specific use cases underrepresented in the training corpus. Relying on prompt engineering or few-shot examples to maximize the output quality on a particular test case can be frustrating, as models can be highly sensitive to small changes, react in unpredicted ways or rely on a fixed system prompt for maintaining performance. In this work, we ask: "Can we optimize our training protocols to both improve controllability and performance on underrepresented use cases at inference time?" We revisit the divide between training and inference techniques to improve long-tail performance while providing users with a set of control levers the model is trained to be responsive to. We create a detailed taxonomy of data characteristics and task provenance to explicitly control generation attributes and implicitly condition generations at inference time. We fine-tune a base model to infer these markers automatically, which makes them optional at inference time. This principled and flexible approach yields pronounced improvements in performance, especially on examples from the long tail of the training distribution. While we observe an average lift of 5.7% win rates in open-ended generation quality with our markers, we see over 9.1% gains in underrepresented domains. We also observe relative lifts of up to 14.1% on underrepresented tasks like CodeRepair and absolute improvements of 35.3% on length instruction following evaluations.
>
---
#### [new 038] LexiMark: Robust Watermarking via Lexical Substitutions to Enhance Membership Verification of an LLM's Textual Training Data
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于文本水印任务，旨在解决LLM训练数据未经授权使用的验证问题。通过语义替换嵌入水印，提升检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.14474v1](http://arxiv.org/pdf/2506.14474v1)**

> **作者:** Eyal German; Sagiv Antebi; Edan Habler; Asaf Shabtai; Yuval Elovici
>
> **摘要:** Large language models (LLMs) can be trained or fine-tuned on data obtained without the owner's consent. Verifying whether a specific LLM was trained on particular data instances or an entire dataset is extremely challenging. Dataset watermarking addresses this by embedding identifiable modifications in training data to detect unauthorized use. However, existing methods often lack stealth, making them relatively easy to detect and remove. In light of these limitations, we propose LexiMark, a novel watermarking technique designed for text and documents, which embeds synonym substitutions for carefully selected high-entropy words. Our method aims to enhance an LLM's memorization capabilities on the watermarked text without altering the semantic integrity of the text. As a result, the watermark is difficult to detect, blending seamlessly into the text with no visible markers, and is resistant to removal due to its subtle, contextually appropriate substitutions that evade automated and manual detection. We evaluated our method using baseline datasets from recent studies and seven open-source models: LLaMA-1 7B, LLaMA-3 8B, Mistral 7B, Pythia 6.9B, as well as three smaller variants from the Pythia family (160M, 410M, and 1B). Our evaluation spans multiple training settings, including continued pretraining and fine-tuning scenarios. The results demonstrate significant improvements in AUROC scores compared to existing methods, underscoring our method's effectiveness in reliably verifying whether unauthorized watermarked data was used in LLM training.
>
---
#### [new 039] MultiFinBen: A Multilingual, Multimodal, and Difficulty-Aware Benchmark for Financial LLM Evaluation
- **分类: cs.CL**

- **简介: 该论文提出MultiFinBen，一个面向金融领域的多语言、多模态基准，解决现有评估数据单一、复杂度不足的问题，通过新任务和动态机制提升模型在跨语言和多模态金融任务中的评估能力。**

- **链接: [http://arxiv.org/pdf/2506.14028v1](http://arxiv.org/pdf/2506.14028v1)**

> **作者:** Xueqing Peng; Lingfei Qian; Yan Wang; Ruoyu Xiang; Yueru He; Yang Ren; Mingyang Jiang; Jeff Zhao; Huan He; Yi Han; Yun Feng; Yuechen Jiang; Yupeng Cao; Haohang Li; Yangyang Yu; Xiaoyu Wang; Penglei Gao; Shengyuan Lin; Keyi Wang; Shanshan Yang; Yilun Zhao; Zhiwei Liu; Peng Lu; Jerry Huang; Suyuchen Wang; Triantafillos Papadopoulos; Polydoros Giannouris; Efstathia Soufleri; Nuo Chen; Guojun Xiong; Zhiyang Deng; Yijia Zhao; Mingquan Lin; Meikang Qiu; Kaleb E Smith; Arman Cohan; Xiao-Yang Liu; Jimin Huang; Alejandro Lopez-Lira; Xi Chen; Junichi Tsujii; Jian-Yun Nie; Sophia Ananiadou; Qianqian Xie
>
> **摘要:** Recent advances in large language models (LLMs) have accelerated progress in financial NLP and applications, yet existing benchmarks remain limited to monolingual and unimodal settings, often over-relying on simple tasks and failing to reflect the complexity of real-world financial communication. We introduce MultiFinBen, the first multilingual and multimodal benchmark tailored to the global financial domain, evaluating LLMs across modalities (text, vision, audio) and linguistic settings (monolingual, bilingual, multilingual) on domain-specific tasks. We introduce two novel tasks, including PolyFiQA-Easy and PolyFiQA-Expert, the first multilingual financial benchmarks requiring models to perform complex reasoning over mixed-language inputs; and EnglishOCR and SpanishOCR, the first OCR-embedded financial QA tasks challenging models to extract and reason over information from visual-text financial documents. Moreover, we propose a dynamic, difficulty-aware selection mechanism and curate a compact, balanced benchmark rather than simple aggregation existing datasets. Extensive evaluation of 22 state-of-the-art models reveals that even the strongest models, despite their general multimodal and multilingual capabilities, struggle dramatically when faced with complex cross-lingual and multimodal tasks in financial domain. MultiFinBen is publicly released to foster transparent, reproducible, and inclusive progress in financial studies and applications.
>
---
#### [new 040] ClimateChat: Designing Data and Methods for Instruction Tuning LLMs to Answer Climate Change Queries
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于气候问答任务，旨在解决气候指令数据不足的问题。通过自动化方法构建数据集并微调模型，提升LLMs在气候问题上的表现。**

- **链接: [http://arxiv.org/pdf/2506.13796v1](http://arxiv.org/pdf/2506.13796v1)**

> **作者:** Zhou Chen; Xiao Wang; Yuanhong Liao; Ming Lin; Yuqi Bai
>
> **备注:** ICLR 2025 camera ready, 13 pages, 4 figures, 4 tables
>
> **摘要:** As the issue of global climate change becomes increasingly severe, the demand for research in climate science continues to grow. Natural language processing technologies, represented by Large Language Models (LLMs), have been widely applied to climate change-specific research, providing essential information support for decision-makers and the public. Some studies have improved model performance on relevant tasks by constructing climate change-related instruction data and instruction-tuning LLMs. However, current research remains inadequate in efficiently producing large volumes of high-precision instruction data for climate change, which limits further development of climate change LLMs. This study introduces an automated method for constructing instruction data. The method generates instructions using facts and background knowledge from documents and enhances the diversity of the instruction data through web scraping and the collection of seed instructions. Using this method, we constructed a climate change instruction dataset, named ClimateChat-Corpus, which was used to fine-tune open-source LLMs, resulting in an LLM named ClimateChat. Evaluation results show that ClimateChat significantly improves performance on climate change question-and-answer tasks. Additionally, we evaluated the impact of different base models and instruction data on LLM performance and demonstrated its capability to adapt to a wide range of climate change scientific discovery tasks, emphasizing the importance of selecting an appropriate base model for instruction tuning. This research provides valuable references and empirical support for constructing climate change instruction data and training climate change-specific LLMs.
>
---
#### [new 041] GRAM: A Generative Foundation Reward Model for Reward Generalization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习中的奖励建模任务，旨在提升奖励模型的泛化能力。通过结合生成模型与监督学习，提出一种新的基础奖励模型。**

- **链接: [http://arxiv.org/pdf/2506.14175v1](http://arxiv.org/pdf/2506.14175v1)**

> **作者:** Chenglong Wang; Yang Gan; Yifu Huo; Yongyu Mu; Qiaozhi He; Murun Yang; Bei Li; Tong Xiao; Chunliang Zhang; Tongran Liu; Jingbo Zhu
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** In aligning large language models (LLMs), reward models have played an important role, but are standardly trained as discriminative models and rely only on labeled human preference data. In this paper, we explore methods that train reward models using both unlabeled and labeled data. Building on the generative models in LLMs, we develop a generative reward model that is first trained via large-scale unsupervised learning and then fine-tuned via supervised learning. We also show that by using label smoothing, we are in fact optimizing a regularized pairwise ranking loss. This result, in turn, provides a new view of training reward models, which links generative models and discriminative models under the same class of training objectives. The outcome of these techniques is a foundation reward model, which can be applied to a wide range of tasks with little or no further fine-tuning effort. Extensive experiments show that this model generalizes well across several tasks, including response ranking, reinforcement learning from human feedback, and task adaptation with fine-tuning, achieving significant performance improvements over several strong baseline models.
>
---
#### [new 042] Sampling from Your Language Model One Byte at a Time
- **分类: cs.CL; cs.FL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决tokenization带来的生成偏差和模型兼容问题。通过方法将BPE模型转为字节级生成，提升模型互操作性与效果。**

- **链接: [http://arxiv.org/pdf/2506.14123v1](http://arxiv.org/pdf/2506.14123v1)**

> **作者:** Jonathan Hayase; Alisa Liu; Noah A. Smith; Sewoong Oh
>
> **备注:** 23 pages, 8 figures
>
> **摘要:** Tokenization is used almost universally by modern language models, enabling efficient text representation using multi-byte or multi-character tokens. However, prior work has shown that tokenization can introduce distortion into the model's generations. For example, users are often advised not to end their prompts with a space because it prevents the model from including the space as part of the next token. This Prompt Boundary Problem (PBP) also arises in languages such as Chinese and in code generation, where tokens often do not line up with syntactic boundaries. Additionally mismatching tokenizers often hinder model composition and interoperability. For example, it is not possible to directly ensemble models with different tokenizers due to their mismatching vocabularies. To address these issues, we present an inference-time method to convert any autoregressive LM with a BPE tokenizer into a character-level or byte-level LM, without changing its generative distribution at the text level. Our method efficient solves the PBP and is also able to unify the vocabularies of language models with different tokenizers, allowing one to ensemble LMs with different tokenizers at inference time as well as transfer the post-training from one model to another using proxy-tuning. We demonstrate in experiments that the ensemble and proxy-tuned models outperform their constituents on downstream evals.
>
---
#### [new 043] When Does Meaning Backfire? Investigating the Role of AMRs in NLI
- **分类: cs.CL**

- **简介: 该论文属于自然语言推理（NLI）任务，研究AMR对模型泛化的影响。实验表明，AMR在微调中起反作用，在提示中略有提升，但效果源于表面差异而非语义理解。**

- **链接: [http://arxiv.org/pdf/2506.14613v1](http://arxiv.org/pdf/2506.14613v1)**

> **作者:** Junghyun Min; Xiulin Yang; Shira Wein
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** Natural Language Inference (NLI) relies heavily on adequately parsing the semantic content of the premise and hypothesis. In this work, we investigate whether adding semantic information in the form of an Abstract Meaning Representation (AMR) helps pretrained language models better generalize in NLI. Our experiments integrating AMR into NLI in both fine-tuning and prompting settings show that the presence of AMR in fine-tuning hinders model generalization while prompting with AMR leads to slight gains in \texttt{GPT-4o}. However, an ablation study reveals that the improvement comes from amplifying surface-level differences rather than aiding semantic reasoning. This amplification can mislead models to predict non-entailment even when the core meaning is preserved.
>
---
#### [new 044] VL-GenRM: Enhancing Vision-Language Verification via Vision Experts and Iterative Training
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉语言对齐任务，解决VL-RM训练中的数据偏差与幻觉问题，通过迭代框架提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.13888v1](http://arxiv.org/pdf/2506.13888v1)**

> **作者:** Jipeng Zhang; Kehao Miao; Renjie Pi; Zhaowei Wang; Runtao Liu; Rui Pan; Tong Zhang
>
> **摘要:** Reinforcement Fine-Tuning (RFT) with verifiable rewards has advanced large language models but remains underexplored for Vision-Language (VL) models. The Vision-Language Reward Model (VL-RM) is key to aligning VL models by providing structured feedback, yet training effective VL-RMs faces two major challenges. First, the bootstrapping dilemma arises as high-quality training data depends on already strong VL models, creating a cycle where self-generated supervision reinforces existing biases. Second, modality bias and negative example amplification occur when VL models hallucinate incorrect visual attributes, leading to flawed preference data that further misguides training. To address these issues, we propose an iterative training framework leveraging vision experts, Chain-of-Thought (CoT) rationales, and Margin-based Rejection Sampling. Our approach refines preference datasets, enhances structured critiques, and iteratively improves reasoning. Experiments across VL-RM benchmarks demonstrate superior performance in hallucination detection and multimodal reasoning, advancing VL model alignment with reinforcement learning.
>
---
#### [new 045] AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents
- **分类: cs.CL**

- **简介: 该论文提出AgentSynth，用于生成通用计算机使用代理的高质量任务和轨迹数据，解决任务生成效率与成本问题。**

- **链接: [http://arxiv.org/pdf/2506.14205v1](http://arxiv.org/pdf/2506.14205v1)**

> **作者:** Jingxu Xie; Dylan Xu; Xuandong Zhao; Dawn Song
>
> **摘要:** We introduce AgentSynth, a scalable and cost-efficient pipeline for automatically synthesizing high-quality tasks and trajectory datasets for generalist computer-use agents. Leveraging information asymmetry, AgentSynth constructs subtasks that are simple during generation but significantly more challenging when composed into long-horizon tasks, enabling the creation of over 6,000 diverse and realistic tasks. Our pipeline begins with an LLM-based task proposer guided by a persona, followed by an execution agent that completes the task and logs the trajectory. This process is repeated iteratively to form a sequence of subtasks, which are then summarized by a separate agent into a composite task of controllable difficulty. A key strength of AgentSynth is its ability to precisely modulate task complexity by varying the number of subtasks. Empirical evaluations show that state-of-the-art LLM agents suffer a steep performance drop, from 18% success at difficulty level 1 to just 4% at level 6, highlighting the benchmark's difficulty and discriminative power. Moreover, our pipeline achieves a low average cost of \$0.60 per trajectory, orders of magnitude cheaper than human annotations. Our code and data are publicly available at https://github.com/sunblaze-ucb/AgentSynth
>
---
#### [new 046] Ring-lite: Scalable Reasoning via C3PO-Stabilized Reinforcement Learning for LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型推理任务，旨在提升模型效率与稳定性。通过C3PO和知识蒸馏等方法，优化MoE模型，在保持性能的同时减少参数激活量。**

- **链接: [http://arxiv.org/pdf/2506.14731v1](http://arxiv.org/pdf/2506.14731v1)**

> **作者:** Ring Team; Bin Hu; Cai Chen; Deng Zhao; Ding Liu; Dingnan Jin; Feng Zhu; Hao Dai; Hongzhi Luan; Jia Guo; Jiaming Liu; Jiewei Wu; Jun Mei; Jun Zhou; Junbo Zhao; Junwu Xiong; Kaihong Zhang; Kuan Xu; Lei Liang; Liang Jiang; Liangcheng Fu; Longfei Zheng; Qiang Gao; Qing Cui; Quan Wan; Shaomian Zheng; Shuaicheng Li; Tongkai Yang; Wang Ren; Xiaodong Yan; Xiaopei Wan; Xiaoyun Feng; Xin Zhao; Xinxing Yang; Xinyu Kong; Xuemin Yang; Yang Li; Yingting Wu; Yongkang Liu; Zhankai Xu; Zhenduo Zhang; Zhenglei Zhou; Zhenyu Huang; Zhiqiang Zhang; Zihao Wang; Zujie Wen
>
> **备注:** Technical Report
>
> **摘要:** We present Ring-lite, a Mixture-of-Experts (MoE)-based large language model optimized via reinforcement learning (RL) to achieve efficient and robust reasoning capabilities. Built upon the publicly available Ling-lite model, a 16.8 billion parameter model with 2.75 billion activated parameters, our approach matches the performance of state-of-the-art (SOTA) small-scale reasoning models on challenging benchmarks (e.g., AIME, LiveCodeBench, GPQA-Diamond) while activating only one-third of the parameters required by comparable models. To accomplish this, we introduce a joint training pipeline integrating distillation with RL, revealing undocumented challenges in MoE RL training. First, we identify optimization instability during RL training, and we propose Constrained Contextual Computation Policy Optimization(C3PO), a novel approach that enhances training stability and improves computational throughput via algorithm-system co-design methodology. Second, we empirically demonstrate that selecting distillation checkpoints based on entropy loss for RL training, rather than validation metrics, yields superior performance-efficiency trade-offs in subsequent RL training. Finally, we develop a two-stage training paradigm to harmonize multi-domain data integration, addressing domain conflicts that arise in training with mixed dataset. We will release the model, dataset, and code.
>
---
#### [new 047] M2BeamLLM: Multimodal Sensing-empowered mmWave Beam Prediction with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于毫米波通信中的波束预测任务，旨在提升V2I系统的通信性能。通过融合多模态传感器数据并利用大语言模型，提高波束预测的准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.14532v1](http://arxiv.org/pdf/2506.14532v1)**

> **作者:** Can Zheng; Jiguang He; Chung G. Kang; Guofa Cai; Zitong Yu; Merouane Debbah
>
> **备注:** 13 pages, 20 figures
>
> **摘要:** This paper introduces a novel neural network framework called M2BeamLLM for beam prediction in millimeter-wave (mmWave) massive multi-input multi-output (mMIMO) communication systems. M2BeamLLM integrates multi-modal sensor data, including images, radar, LiDAR, and GPS, leveraging the powerful reasoning capabilities of large language models (LLMs) such as GPT-2 for beam prediction. By combining sensing data encoding, multimodal alignment and fusion, and supervised fine-tuning (SFT), M2BeamLLM achieves significantly higher beam prediction accuracy and robustness, demonstrably outperforming traditional deep learning (DL) models in both standard and few-shot scenarios. Furthermore, its prediction performance consistently improves with increased diversity in sensing modalities. Our study provides an efficient and intelligent beam prediction solution for vehicle-to-infrastructure (V2I) mmWave communication systems.
>
---
#### [new 048] A Variational Framework for Improving Naturalness in Generative Spoken Language Models
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音生成任务，旨在提升生成语音的自然度。针对现有方法依赖人工特征的问题，提出一种端到端变分框架，自动编码连续语音属性以增强语义标记。**

- **链接: [http://arxiv.org/pdf/2506.14767v1](http://arxiv.org/pdf/2506.14767v1)**

> **作者:** Li-Wei Chen; Takuya Higuchi; Zakaria Aldeneh; Ahmed Hussen Abdelaziz; Alexander Rudnicky
>
> **备注:** International Conference on Machine Learning (ICML) 2025
>
> **摘要:** The success of large language models in text processing has inspired their adaptation to speech modeling. However, since speech is continuous and complex, it is often discretized for autoregressive modeling. Speech tokens derived from self-supervised models (known as semantic tokens) typically focus on the linguistic aspects of speech but neglect prosodic information. As a result, models trained on these tokens can generate speech with reduced naturalness. Existing approaches try to fix this by adding pitch features to the semantic tokens. However, pitch alone cannot fully represent the range of paralinguistic attributes, and selecting the right features requires careful hand-engineering. To overcome this, we propose an end-to-end variational approach that automatically learns to encode these continuous speech attributes to enhance the semantic tokens. Our approach eliminates the need for manual extraction and selection of paralinguistic features. Moreover, it produces preferred speech continuations according to human raters. Code, samples and models are available at https://github.com/b04901014/vae-gslm.
>
---
#### [new 049] Xolver: Multi-Agent Reasoning with Holistic Experience Learning Just Like an Olympiad Team
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Xolver，一个无需训练的多智能体推理框架，解决LLM孤立推理问题。通过整合经验知识，提升推理能力，实现专家级表现。**

- **链接: [http://arxiv.org/pdf/2506.14234v1](http://arxiv.org/pdf/2506.14234v1)**

> **作者:** Md Tanzib Hosain; Salman Rahman; Md Kishor Morol; Md Rizwan Parvez
>
> **摘要:** Despite impressive progress on complex reasoning, current large language models (LLMs) typically operate in isolation - treating each problem as an independent attempt, without accumulating or integrating experiential knowledge. In contrast, expert problem solvers - such as Olympiad or programming contest teams - leverage a rich tapestry of experiences: absorbing mentorship from coaches, developing intuition from past problems, leveraging knowledge of tool usage and library functionality, adapting strategies based on the expertise and experiences of peers, continuously refining their reasoning through trial and error, and learning from other related problems even during competition. We introduce Xolver, a training-free multi-agent reasoning framework that equips a black-box LLM with a persistent, evolving memory of holistic experience. Xolver integrates diverse experience modalities, including external and self-retrieval, tool use, collaborative interactions, agent-driven evaluation, and iterative refinement. By learning from relevant strategies, code fragments, and abstract reasoning patterns at inference time, Xolver avoids generating solutions from scratch - marking a transition from isolated inference toward experience-aware language agents. Built on both open-weight and proprietary models, Xolver consistently outperforms specialized reasoning agents. Even with lightweight backbones (e.g., QWQ-32B), it often surpasses advanced models including Qwen3-235B, Gemini 2.5 Pro, o3, and o4-mini-high. With o3-mini-high, it achieves new best results on GSM8K (98.1%), AIME'24 (94.4%), AIME'25 (93.7%), Math-500 (99.8%), and LiveCodeBench-V5 (91.6%) - highlighting holistic experience learning as a key step toward generalist agents capable of expert-level reasoning. Code and data are available at https://kagnlp.github.io/xolver.github.io/.
>
---
#### [new 050] A Vision for Geo-Temporal Deep Research Systems: Towards Comprehensive, Transparent, and Reproducible Geo-Temporal Information Synthesis
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于AI信息处理任务，旨在解决深度研究系统中缺乏地理时间推理能力的问题，提出增强检索与综合过程的愿景。**

- **链接: [http://arxiv.org/pdf/2506.14345v1](http://arxiv.org/pdf/2506.14345v1)**

> **作者:** Bruno Martins; Piotr Szymański; Piotr Gramacki
>
> **摘要:** The emergence of Large Language Models (LLMs) has transformed information access, with current LLMs also powering deep research systems that can generate comprehensive report-style answers, through planned iterative search, retrieval, and reasoning. Still, current deep research systems lack the geo-temporal capabilities that are essential for answering context-rich questions involving geographic and/or temporal constraints, frequently occurring in domains like public health, environmental science, or socio-economic analysis. This paper reports our vision towards next generation systems, identifying important technical, infrastructural, and evaluative challenges in integrating geo-temporal reasoning into deep research pipelines. We argue for augmenting retrieval and synthesis processes with the ability to handle geo-temporal constraints, supported by open and reproducible infrastructures and rigorous evaluation protocols. Our vision outlines a path towards more advanced and geo-temporally aware deep research systems, of potential impact to the future of AI-driven information access.
>
---
#### [new 051] Digital Gatekeepers: Google's Role in Curating Hashtags and Subreddits
- **分类: cs.CL**

- **简介: 该论文属于信息过滤研究，探讨Google如何通过算法影响社交媒体内容可见性，揭示其对特定话题的偏见性处理。**

- **链接: [http://arxiv.org/pdf/2506.14370v1](http://arxiv.org/pdf/2506.14370v1)**

> **作者:** Amrit Poudel; Yifan Ding; Jurgen Pfeffer; Tim Weninger
>
> **备注:** Accepted to ACL 2025 Main
>
> **摘要:** Search engines play a crucial role as digital gatekeepers, shaping the visibility of Web and social media content through algorithmic curation. This study investigates how search engines like Google selectively promotes or suppresses certain hashtags and subreddits, impacting the information users encounter. By comparing search engine results with nonsampled data from Reddit and Twitter/X, we reveal systematic biases in content visibility. Google's algorithms tend to suppress subreddits and hashtags related to sexually explicit material, conspiracy theories, advertisements, and cryptocurrencies, while promoting content associated with higher engagement. These findings suggest that Google's gatekeeping practices influence public discourse by curating the social media narratives available to users.
>
---
#### [new 052] An Interdisciplinary Review of Commonsense Reasoning and Intent Detection
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于自然语言理解任务，旨在解决常识推理和意图检测问题。通过综述28篇文献，分析方法与应用，提出模型需更适应、多语言和上下文感知。**

- **链接: [http://arxiv.org/pdf/2506.14040v1](http://arxiv.org/pdf/2506.14040v1)**

> **作者:** Md Nazmus Sakib
>
> **摘要:** This review explores recent advances in commonsense reasoning and intent detection, two key challenges in natural language understanding. We analyze 28 papers from ACL, EMNLP, and CHI (2020-2025), organizing them by methodology and application. Commonsense reasoning is reviewed across zero-shot learning, cultural adaptation, structured evaluation, and interactive contexts. Intent detection is examined through open-set models, generative formulations, clustering, and human-centered systems. By bridging insights from NLP and HCI, we highlight emerging trends toward more adaptive, multilingual, and context-aware models, and identify key gaps in grounding, generalization, and benchmark design.
>
---
#### [new 053] Passing the Turing Test in Political Discourse: Fine-Tuning LLMs to Mimic Polarized Social Media Comments
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理任务，旨在研究LLMs在政治话语中模拟极化评论的能力，解决AI生成偏见内容的问题，通过微调模型并评估其输出的真实性与影响力。**

- **链接: [http://arxiv.org/pdf/2506.14645v1](http://arxiv.org/pdf/2506.14645v1)**

> **作者:** . Pazzaglia; V. Vendetti; L. D. Comencini; F. Deriu; V. Modugno
>
> **摘要:** The increasing sophistication of large language models (LLMs) has sparked growing concerns regarding their potential role in exacerbating ideological polarization through the automated generation of persuasive and biased content. This study explores the extent to which fine-tuned LLMs can replicate and amplify polarizing discourse within online environments. Using a curated dataset of politically charged discussions extracted from Reddit, we fine-tune an open-source LLM to produce context-aware and ideologically aligned responses. The model's outputs are evaluated through linguistic analysis, sentiment scoring, and human annotation, with particular attention to credibility and rhetorical alignment with the original discourse. The results indicate that, when trained on partisan data, LLMs are capable of producing highly plausible and provocative comments, often indistinguishable from those written by humans. These findings raise significant ethical questions about the use of AI in political discourse, disinformation, and manipulation campaigns. The paper concludes with a discussion of the broader implications for AI governance, platform regulation, and the development of detection tools to mitigate adversarial fine-tuning risks.
>
---
#### [new 054] LingoLoop Attack: Trapping MLLMs via Linguistic Context and State Entrapment into Endless Loops
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于安全攻击任务，旨在通过语言上下文和状态陷阱使多模态大模型陷入无限循环，增加输出长度和能耗。**

- **链接: [http://arxiv.org/pdf/2506.14493v1](http://arxiv.org/pdf/2506.14493v1)**

> **作者:** Jiyuan Fu; Kaixun Jiang; Lingyi Hong; Jinglun Li; Haijing Guo; Dingkang Yang; Zhaoyu Chen; Wenqiang Zhang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown great promise but require substantial computational resources during inference. Attackers can exploit this by inducing excessive output, leading to resource exhaustion and service degradation. Prior energy-latency attacks aim to increase generation time by broadly shifting the output token distribution away from the EOS token, but they neglect the influence of token-level Part-of-Speech (POS) characteristics on EOS and sentence-level structural patterns on output counts, limiting their efficacy. To address this, we propose LingoLoop, an attack designed to induce MLLMs to generate excessively verbose and repetitive sequences. First, we find that the POS tag of a token strongly affects the likelihood of generating an EOS token. Based on this insight, we propose a POS-Aware Delay Mechanism to postpone EOS token generation by adjusting attention weights guided by POS information. Second, we identify that constraining output diversity to induce repetitive loops is effective for sustained generation. We introduce a Generative Path Pruning Mechanism that limits the magnitude of hidden states, encouraging the model to produce persistent loops. Extensive experiments demonstrate LingoLoop can increase generated tokens by up to 30 times and energy consumption by a comparable factor on models like Qwen2.5-VL-3B, consistently driving MLLMs towards their maximum generation limits. These findings expose significant MLLMs' vulnerabilities, posing challenges for their reliable deployment. The code will be released publicly following the paper's acceptance.
>
---
#### [new 055] Intended Target Identification for Anomia Patients with Gradient-based Selective Augmentation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在帮助失语症患者识别目标物品。解决术语缺失和语义错误问题，通过梯度增强方法提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.14203v1](http://arxiv.org/pdf/2506.14203v1)**

> **作者:** Jongho Kim; Romain Storaï; Seung-won Hwang
>
> **备注:** EMNLP 2024 Findings (long)
>
> **摘要:** In this study, we investigate the potential of language models (LMs) in aiding patients experiencing anomia, a difficulty identifying the names of items. Identifying the intended target item from patient's circumlocution involves the two challenges of term failure and error: (1) The terms relevant to identifying the item remain unseen. (2) What makes the challenge unique is inherent perturbed terms by semantic paraphasia, which are not exactly related to the target item, hindering the identification process. To address each, we propose robustifying the model from semantically paraphasic errors and enhancing the model with unseen terms with gradient-based selective augmentation. Specifically, the gradient value controls augmented data quality amid semantic errors, while the gradient variance guides the inclusion of unseen but relevant terms. Due to limited domain-specific datasets, we evaluate the model on the Tip-of-the-Tongue dataset as an intermediary task and then apply our findings to real patient data from AphasiaBank. Our results demonstrate strong performance against baselines, aiding anomia patients by addressing the outlined challenges.
>
---
#### [new 056] Automatic Extraction of Clausal Embedding Based on Large-Scale English Text Data
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的句法分析任务，旨在自动提取英语嵌套从句。解决传统研究依赖人工构造例子的问题，通过语料库和解析方法实现自动检测与标注。**

- **链接: [http://arxiv.org/pdf/2506.14064v1](http://arxiv.org/pdf/2506.14064v1)**

> **作者:** Iona Carslaw; Sivan Milton; Nicolas Navarre; Ciyang Qing; Wataru Uegaki
>
> **备注:** Accepted in the Society for Computation in Linguistics
>
> **摘要:** For linguists, embedded clauses have been of special interest because of their intricate distribution of syntactic and semantic features. Yet, current research relies on schematically created language examples to investigate these constructions, missing out on statistical information and naturally-occurring examples that can be gained from large language corpora. Thus, we present a methodological approach for detecting and annotating naturally-occurring examples of English embedded clauses in large-scale text data using constituency parsing and a set of parsing heuristics. Our tool has been evaluated on our dataset Golden Embedded Clause Set (GECS), which includes hand-annotated examples of naturally-occurring English embedded clause sentences. Finally, we present a large-scale dataset of naturally-occurring English embedded clauses which we have extracted from the open-source corpus Dolma using our extraction tool.
>
---
#### [new 057] A Multi-Expert Structural-Semantic Hybrid Framework for Unveiling Historical Patterns in Temporal Knowledge Graphs
- **分类: cs.CL**

- **简介: 该论文属于时间知识图谱推理任务，旨在解决现有方法未能融合结构与语义信息及区分历史与非历史事件的问题。提出MESH框架整合多专家模块以提升预测效果。**

- **链接: [http://arxiv.org/pdf/2506.14235v1](http://arxiv.org/pdf/2506.14235v1)**

> **作者:** Yimin Deng; Yuxia Wu; Yejing Wang; Guoshuai Zhao; Li Zhu; Qidong Liu; Derong Xu; Zichuan Fu; Xian Wu; Yefeng Zheng; Xiangyu Zhao; Xueming Qian
>
> **备注:** ACL25 findings
>
> **摘要:** Temporal knowledge graph reasoning aims to predict future events with knowledge of existing facts and plays a key role in various downstream tasks. Previous methods focused on either graph structure learning or semantic reasoning, failing to integrate dual reasoning perspectives to handle different prediction scenarios. Moreover, they lack the capability to capture the inherent differences between historical and non-historical events, which limits their generalization across different temporal contexts. To this end, we propose a Multi-Expert Structural-Semantic Hybrid (MESH) framework that employs three kinds of expert modules to integrate both structural and semantic information, guiding the reasoning process for different events. Extensive experiments on three datasets demonstrate the effectiveness of our approach.
>
---
#### [new 058] MAS-LitEval : Multi-Agent System for Literary Translation Quality Assessment
- **分类: cs.CL**

- **简介: 该论文属于文学翻译质量评估任务，旨在解决传统指标无法准确评估文学作品风格和叙事一致性的问题。提出MAS-LitEval系统，利用大语言模型进行多维度评估。**

- **链接: [http://arxiv.org/pdf/2506.14199v1](http://arxiv.org/pdf/2506.14199v1)**

> **作者:** Junghwan Kim; Kieun Park; Sohee Park; Hyunggug Kim; Bongwon Suh
>
> **备注:** 4 Pages, 2 tables, EMNLP submitted
>
> **摘要:** Literary translation requires preserving cultural nuances and stylistic elements, which traditional metrics like BLEU and METEOR fail to assess due to their focus on lexical overlap. This oversight neglects the narrative consistency and stylistic fidelity that are crucial for literary works. To address this, we propose MAS-LitEval, a multi-agent system using Large Language Models (LLMs) to evaluate translations based on terminology, narrative, and style. We tested MAS-LitEval on translations of The Little Prince and A Connecticut Yankee in King Arthur's Court, generated by various LLMs, and compared it to traditional metrics. \textbf{MAS-LitEval} outperformed these metrics, with top models scoring up to 0.890 in capturing literary nuances. This work introduces a scalable, nuanced framework for Translation Quality Assessment (TQA), offering a practical tool for translators and researchers.
>
---
#### [new 059] CausalDiffTab: Mixed-Type Causal-Aware Diffusion for Tabular Data Generation
- **分类: cs.CL**

- **简介: 该论文属于数据生成任务，旨在解决高质混合类型表格数据生成难题。提出CausalDiffTab模型，结合因果正则化方法提升生成效果。**

- **链接: [http://arxiv.org/pdf/2506.14206v1](http://arxiv.org/pdf/2506.14206v1)**

> **作者:** Jia-Chen Zhang; Zheng Zhou; Yu-Jie Xiong; Chun-Ming Xia; Fei Dai
>
> **摘要:** Training data has been proven to be one of the most critical components in training generative AI. However, obtaining high-quality data remains challenging, with data privacy issues presenting a significant hurdle. To address the need for high-quality data. Synthesize data has emerged as a mainstream solution, demonstrating impressive performance in areas such as images, audio, and video. Generating mixed-type data, especially high-quality tabular data, still faces significant challenges. These primarily include its inherent heterogeneous data types, complex inter-variable relationships, and intricate column-wise distributions. In this paper, we introduce CausalDiffTab, a diffusion model-based generative model specifically designed to handle mixed tabular data containing both numerical and categorical features, while being more flexible in capturing complex interactions among variables. We further propose a hybrid adaptive causal regularization method based on the principle of Hierarchical Prior Fusion. This approach adaptively controls the weight of causal regularization, enhancing the model's performance without compromising its generative capabilities. Comprehensive experiments conducted on seven datasets demonstrate that CausalDiffTab outperforms baseline methods across all metrics. Our code is publicly available at: https://github.com/Godz-z/CausalDiffTab.
>
---
#### [new 060] MIST: Towards Multi-dimensional Implicit Bias and Stereotype Evaluation of LLMs via Theory of Mind
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的隐性偏见评估任务，旨在解决LLMs在心智理论上的多维偏差问题，通过构建间接测试框架揭示其隐性刻板印象。**

- **链接: [http://arxiv.org/pdf/2506.14161v1](http://arxiv.org/pdf/2506.14161v1)**

> **作者:** Yanlin Li; Hao Liu; Huimin Liu; Yinwei Wei; Yupeng Hu
>
> **摘要:** Theory of Mind (ToM) in Large Language Models (LLMs) refers to their capacity for reasoning about mental states, yet failures in this capacity often manifest as systematic implicit bias. Evaluating this bias is challenging, as conventional direct-query methods are susceptible to social desirability effects and fail to capture its subtle, multi-dimensional nature. To this end, we propose an evaluation framework that leverages the Stereotype Content Model (SCM) to reconceptualize bias as a multi-dimensional failure in ToM across Competence, Sociability, and Morality. The framework introduces two indirect tasks: the Word Association Bias Test (WABT) to assess implicit lexical associations and the Affective Attribution Test (AAT) to measure covert affective leanings, both designed to probe latent stereotypes without triggering model avoidance. Extensive experiments on 8 State-of-the-Art LLMs demonstrate our framework's capacity to reveal complex bias structures, including pervasive sociability bias, multi-dimensional divergence, and asymmetric stereotype amplification, thereby providing a more robust methodology for identifying the structural nature of implicit bias.
>
---
#### [new 061] ImpliRet: Benchmarking the Implicit Fact Retrieval Challenge
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索任务，旨在解决文档侧隐含事实推理问题。通过构建ImpliRet基准，评估模型在时间、算术和常识关系上的推理能力。**

- **链接: [http://arxiv.org/pdf/2506.14407v1](http://arxiv.org/pdf/2506.14407v1)**

> **作者:** Zeinab Sadat Taghavi; Ali Modarressi; Yunpu Ma; Hinrich Schütze
>
> **摘要:** Retrieval systems are central to many NLP pipelines, but often rely on surface-level cues such as keyword overlap and lexical semantic similarity. To evaluate retrieval beyond these shallow signals, recent benchmarks introduce reasoning-heavy queries; however, they primarily shift the burden to query-side processing techniques -- like prompting or multi-hop retrieval -- that can help resolve complexity. In contrast, we present ImpliRet, a benchmark that shifts the reasoning challenge to document-side processing: The queries are simple, but relevance depends on facts stated implicitly in documents through temporal (e.g., resolving "two days ago"), arithmetic, and world knowledge relationships. We evaluate a range of sparse and dense retrievers, all of which struggle in this setting: the best nDCG@10 is only 15.07%. We also test whether long-context models can overcome this limitation. But even with a short context of only ten documents, including the positive document, GPT-4.1 scores only 35.06%, showing that document-side reasoning remains a challenge. Our codes are available at github.com/ZeinabTaghavi/IMPLIRET.Contribution.
>
---
#### [new 062] Optimizing Length Compression in Large Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型优化任务，旨在解决大型推理模型生成冗长推理链的问题。通过提出Brevity和Sufficiency原则，设计LC-R1方法实现高效压缩。**

- **链接: [http://arxiv.org/pdf/2506.14755v1](http://arxiv.org/pdf/2506.14755v1)**

> **作者:** Zhengxiang Cheng; Dongping Chen; Mingyang Fu; Tianyi Zhou
>
> **备注:** 16 pages, 7 figures, 4 tables
>
> **摘要:** Large Reasoning Models (LRMs) have achieved remarkable success, yet they often suffer from producing unnecessary and verbose reasoning chains. We identify a core aspect of this issue as "invalid thinking" -- models tend to repeatedly double-check their work after having derived the correct answer. To address this specific inefficiency, we move beyond the general principles of Efficacy and Efficiency to propose two new, fine-grained principles: Brevity, which advocates for eliminating redundancy, and Sufficiency, which ensures critical reasoning steps are preserved. Guided by these principles, we introduce LC-R1, a post-training method based on Group Relative Policy Optimization (GRPO). LC-R1 employs a novel combination of a Length Reward for overall conciseness and a Compress Reward that is specifically designed to remove the invalid portion of the thinking process. Extensive experiments on multiple reasoning benchmarks demonstrate that LC-R1 achieves a significant reduction in sequence length (~50%) with only a marginal (~2%) drop in accuracy, achieving a favorable trade-off point on the Pareto frontier that prioritizes high compression. Our analysis further validates the robustness of LC-R1 and provides valuable insights for developing more powerful yet computationally efficient LRMs. Our code is released at https://github.com/zxiangx/LC-R1.
>
---
#### [new 063] AssistedDS: Benchmarking How External Domain Knowledge Assists LLMs in Automated Data Science
- **分类: cs.LG; cs.AI; cs.CL; stat.ME; 62-07, 62-08, 68T05, 68T07, 68T01, 68T50; I.2.0; I.2.6; I.2.7; I.5.1; I.5.4; H.2.8; G.3**

- **简介: 该论文属于自动化数据科学任务，旨在研究LLMs如何利用外部领域知识。工作包括构建基准AssistedDS，评估LLMs在表格预测任务中处理领域知识的能力。**

- **链接: [http://arxiv.org/pdf/2506.13992v1](http://arxiv.org/pdf/2506.13992v1)**

> **作者:** An Luo; Xun Xian; Jin Du; Fangqiao Tian; Ganghua Wang; Ming Zhong; Shengchun Zhao; Xuan Bi; Zirui Liu; Jiawei Zhou; Jayanth Srinivasa; Ashish Kundu; Charles Fleming; Mingyi Hong; Jie Ding
>
> **摘要:** Large language models (LLMs) have advanced the automation of data science workflows. Yet it remains unclear whether they can critically leverage external domain knowledge as human data scientists do in practice. To answer this question, we introduce AssistedDS (Assisted Data Science), a benchmark designed to systematically evaluate how LLMs handle domain knowledge in tabular prediction tasks. AssistedDS features both synthetic datasets with explicitly known generative mechanisms and real-world Kaggle competitions, each accompanied by curated bundles of helpful and adversarial documents. These documents provide domain-specific insights into data cleaning, feature engineering, and model selection. We assess state-of-the-art LLMs on their ability to discern and apply beneficial versus harmful domain knowledge, evaluating submission validity, information recall, and predictive performance. Our results demonstrate three key findings: (1) LLMs frequently exhibit an uncritical adoption of provided information, significantly impairing their predictive performance when adversarial content is introduced, (2) helpful guidance is often insufficient to counteract the negative influence of adversarial information, and (3) in Kaggle datasets, LLMs often make errors in handling time-series data, applying consistent feature engineering across different folds, and interpreting categorical variables correctly. These findings highlight a substantial gap in current models' ability to critically evaluate and leverage expert knowledge, underscoring an essential research direction for developing more robust, knowledge-aware automated data science systems.
>
---
#### [new 064] Pushing the Performance of Synthetic Speech Detection with Kolmogorov-Arnold Networks and Self-Supervised Learning Models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音合成检测任务，旨在提升对抗语音欺骗攻击的性能。通过引入Kolmogorov-Arnold网络改进SSL模型，显著提高了检测效果。**

- **链接: [http://arxiv.org/pdf/2506.14153v1](http://arxiv.org/pdf/2506.14153v1)**

> **作者:** Tuan Dat Phuong; Long-Vu Hoang; Huy Dat Tran
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Recent advancements in speech synthesis technologies have led to increasingly advanced spoofing attacks, posing significant challenges for automatic speaker verification systems. While systems based on self-supervised learning (SSL) models, particularly the XLSR-Conformer model, have demonstrated remarkable performance in synthetic speech detection, there remains room for architectural improvements. In this paper, we propose a novel approach that replaces the traditional Multi-Layer Perceptron in the XLSR-Conformer model with a Kolmogorov-Arnold Network (KAN), a novel architecture based on the Kolmogorov-Arnold representation theorem. Our results on ASVspoof2021 demonstrate that integrating KAN into the SSL-based models can improve the performance by 60.55% relatively on LA and DF sets, further achieving 0.70% EER on the 21LA set. These findings suggest that incorporating KAN into SSL-based models is a promising direction for advances in synthetic speech detection.
>
---
#### [new 065] Improving LoRA with Variational Learning
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于模型微调任务，旨在解决LoRA微调中精度和校准不足的问题。通过引入IVON变分算法，提升模型性能并降低计算开销。**

- **链接: [http://arxiv.org/pdf/2506.14280v1](http://arxiv.org/pdf/2506.14280v1)**

> **作者:** Bai Cong; Nico Daheim; Yuesong Shen; Rio Yokota; Mohammad Emtiyaz Khan; Thomas Möllenhoff
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** Bayesian methods have recently been used to improve LoRA finetuning and, although they improve calibration, their effect on other metrics (such as accuracy) is marginal and can sometimes even be detrimental. Moreover, Bayesian methods also increase computational overheads and require additional tricks for them to work well. Here, we fix these issues by using a recently proposed variational algorithm called IVON. We show that IVON is easy to implement and has similar costs to AdamW, and yet it can also drastically improve many metrics by using a simple posterior pruning technique. We present extensive results on billion-scale LLMs (Llama and Qwen series) going way beyond the scale of existing applications of IVON. For example, we finetune a Llama-3.2-3B model on a set of commonsense reasoning tasks and improve accuracy over AdamW by 1.3% and reduce ECE by 5.4%, outperforming AdamW and other recent Bayesian methods like Laplace-LoRA and BLoB. Overall, our results show that variational learning with IVON can effectively improve LoRA finetuning.
>
---
#### [new 066] Improving Practical Aspects of End-to-End Multi-Talker Speech Recognition for Online and Offline Scenarios
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于多说话人语音识别任务，旨在提升在线和离线场景下的实时性和准确性。通过改进SOT框架、引入CSS和双模型结构解决延迟与精度平衡问题。**

- **链接: [http://arxiv.org/pdf/2506.14204v1](http://arxiv.org/pdf/2506.14204v1)**

> **作者:** Aswin Shanmugam Subramanian; Amit Das; Naoyuki Kanda; Jinyu Li; Xiaofei Wang; Yifan Gong
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** We extend the frameworks of Serialized Output Training (SOT) to address practical needs of both streaming and offline automatic speech recognition (ASR) applications. Our approach focuses on balancing latency and accuracy, catering to real-time captioning and summarization requirements. We propose several key improvements: (1) Leveraging Continuous Speech Separation (CSS) single-channel front-end with end-to-end (E2E) systems for highly overlapping scenarios, challenging the conventional wisdom of E2E versus cascaded setups. The CSS framework improves the accuracy of the ASR system by separating overlapped speech from multiple speakers. (2) Implementing dual models -- Conformer Transducer for streaming and Sequence-to-Sequence for offline -- or alternatively, a two-pass model based on cascaded encoders. (3) Exploring segment-based SOT (segSOT) which is better suited for offline scenarios while also enhancing readability of multi-talker transcriptions.
>
---
#### [new 067] ASCD: Attention-Steerable Contrastive Decoding for Reducing Hallucination in MLLM
- **分类: cs.CV; cs.CL; 68T45**

- **简介: 该论文属于多模态大语言模型任务，旨在解决模型幻觉问题。通过改进注意力机制，提出一种新的对比解码方法，有效减少幻觉并提升性能。**

- **链接: [http://arxiv.org/pdf/2506.14766v1](http://arxiv.org/pdf/2506.14766v1)**

> **作者:** Yujun Wang; Jinhe Bi; Yunpu Ma; Soeren Pirk
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Multimodal Large Language Model (MLLM) often suffer from hallucinations. They over-rely on partial cues and generate incorrect responses. Recently, methods like Visual Contrastive Decoding (VCD) and Instruction Contrastive Decoding (ICD) have been proposed to mitigate hallucinations by contrasting predictions from perturbed or negatively prefixed inputs against original outputs. In this work, we uncover that methods like VCD and ICD fundamentally influence internal attention dynamics of the model. This observation suggests that their effectiveness may not stem merely from surface-level modifications to logits but from deeper shifts in attention distribution. Inspired by this insight, we propose an attention-steerable contrastive decoding framework that directly intervenes in attention mechanisms of the model to offer a more principled approach to mitigating hallucinations. Our experiments across multiple MLLM architectures and diverse decoding methods demonstrate that our approach significantly reduces hallucinations and improves the performance on benchmarks such as POPE, CHAIR, and MMHal-Bench, while simultaneously enhancing performance on standard VQA benchmarks.
>
---
#### [new 068] VisText-Mosquito: A Multimodal Dataset and Benchmark for AI-Based Mosquito Breeding Site Detection and Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VisText-Mosquito数据集，用于蚊虫孳生地的AI检测与推理，解决蚊媒疾病预防问题。**

- **链接: [http://arxiv.org/pdf/2506.14629v1](http://arxiv.org/pdf/2506.14629v1)**

> **作者:** Md. Adnanul Islam; Md. Faiyaz Abdullah Sayeedi; Md. Asaduzzaman Shuvo; Muhammad Ziaur Rahman; Shahanur Rahman Bappy; Raiyan Rahman; Swakkhar Shatabda
>
> **摘要:** Mosquito-borne diseases pose a major global health risk, requiring early detection and proactive control of breeding sites to prevent outbreaks. In this paper, we present VisText-Mosquito, a multimodal dataset that integrates visual and textual data to support automated detection, segmentation, and reasoning for mosquito breeding site analysis. The dataset includes 1,828 annotated images for object detection, 142 images for water surface segmentation, and natural language reasoning texts linked to each image. The YOLOv9s model achieves the highest precision of 0.92926 and mAP@50 of 0.92891 for object detection, while YOLOv11n-Seg reaches a segmentation precision of 0.91587 and mAP@50 of 0.79795. For reasoning generation, our fine-tuned BLIP model achieves a final loss of 0.0028, with a BLEU score of 54.7, BERTScore of 0.91, and ROUGE-L of 0.87. This dataset and model framework emphasize the theme "Prevention is Better than Cure", showcasing how AI-based detection can proactively address mosquito-borne disease risks. The dataset and implementation code are publicly available at GitHub: https://github.com/adnanul-islam-jisun/VisText-Mosquito
>
---
#### [new 069] InsertRank: LLMs can reason over BM25 scores to Improve Listwise Reranking
- **分类: cs.IR; cs.AI; cs.CL; I.2.7; H.3.3; I.5.4**

- **简介: 该论文属于信息检索任务，旨在提升复杂查询的文档排序效果。通过结合LLM与BM25分数，提出InsertRank模型，增强推理能力以改善重排序性能。**

- **链接: [http://arxiv.org/pdf/2506.14086v1](http://arxiv.org/pdf/2506.14086v1)**

> **作者:** Rahul Seetharaman; Kaustubh D. Dhole; Aman Bansal
>
> **摘要:** Large Language Models (LLMs) have demonstrated significant strides across various information retrieval tasks, particularly as rerankers, owing to their strong generalization and knowledge-transfer capabilities acquired from extensive pretraining. In parallel, the rise of LLM-based chat interfaces has raised user expectations, encouraging users to pose more complex queries that necessitate retrieval by ``reasoning'' over documents rather than through simple keyword matching or semantic similarity. While some recent efforts have exploited reasoning abilities of LLMs for reranking such queries, considerable potential for improvement remains. In that regards, we introduce InsertRank, an LLM-based reranker that leverages lexical signals like BM25 scores during reranking to further improve retrieval performance. InsertRank demonstrates improved retrieval effectiveness on -- BRIGHT, a reasoning benchmark spanning 12 diverse domains, and R2MED, a specialized medical reasoning retrieval benchmark spanning 8 different tasks. We conduct an exhaustive evaluation and several ablation studies and demonstrate that InsertRank consistently improves retrieval effectiveness across multiple families of LLMs, including GPT, Gemini, and Deepseek models. %In addition, we also conduct ablation studies on normalization by varying the scale of the BM25 scores, and positional bias by shuffling the order of the documents. With Deepseek-R1, InsertRank achieves a score of 37.5 on the BRIGHT benchmark. and 51.1 on the R2MED benchmark, surpassing previous methods.
>
---
#### [new 070] Multimodal Fusion with Semi-Supervised Learning Minimizes Annotation Quantity for Modeling Videoconference Conversation Experience
- **分类: eess.AS; cs.CL; cs.HC; cs.LG; cs.MM**

- **简介: 该论文属于视频会议体验建模任务，旨在解决负向体验检测问题。通过半监督学习融合多模态数据，减少标注依赖，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.13971v1](http://arxiv.org/pdf/2506.13971v1)**

> **作者:** Andrew Chang; Chenkai Hu; Ji Qi; Zhuojian Wei; Kexin Zhang; Viswadruth Akkaraju; David Poeppel; Dustin Freeman
>
> **备注:** Interspeech 2025
>
> **摘要:** Group conversations over videoconferencing are a complex social behavior. However, the subjective moments of negative experience, where the conversation loses fluidity or enjoyment remain understudied. These moments are infrequent in naturalistic data, and thus training a supervised learning (SL) model requires costly manual data annotation. We applied semi-supervised learning (SSL) to leverage targeted labeled and unlabeled clips for training multimodal (audio, facial, text) deep features to predict non-fluid or unenjoyable moments in holdout videoconference sessions. The modality-fused co-training SSL achieved an ROC-AUC of 0.9 and an F1 score of 0.6, outperforming SL models by up to 4% with the same amount of labeled data. Remarkably, the best SSL model with just 8% labeled data matched 96% of the SL model's full-data performance. This shows an annotation-efficient framework for modeling videoconference experience.
>
---
#### [new 071] Computational Studies in Influencer Marketing: A Systematic Literature Review
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于系统综述任务，旨在解决 influencer marketing 中计算研究碎片化问题，通过分析69篇文献，总结研究主题与方法，提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2506.14602v1](http://arxiv.org/pdf/2506.14602v1)**

> **作者:** Haoyang Gui; Thales Bertaglia; Catalina Goanta; Gerasimos Spanakis
>
> **备注:** journal submission, under review
>
> **摘要:** Influencer marketing has become a crucial feature of digital marketing strategies. Despite its rapid growth and algorithmic relevance, the field of computational studies in influencer marketing remains fragmented, especially with limited systematic reviews covering the computational methodologies employed. This makes overarching scientific measurements in the influencer economy very scarce, to the detriment of interested stakeholders outside of platforms themselves, such as regulators, but also researchers from other fields. This paper aims to provide an overview of the state of the art of computational studies in influencer marketing by conducting a systematic literature review (SLR) based on the PRISMA model. The paper analyses 69 studies to identify key research themes, methodologies, and future directions in this research field. The review identifies four major research themes: Influencer identification and characterisation, Advertising strategies and engagement, Sponsored content analysis and discovery, and Fairness. Methodologically, the studies are categorised into machine learning-based techniques (e.g., classification, clustering) and non-machine-learning-based techniques (e.g., statistical analysis, network analysis). Key findings reveal a strong focus on optimising commercial outcomes, with limited attention to regulatory compliance and ethical considerations. The review highlights the need for more nuanced computational research that incorporates contextual factors such as language, platform, and industry type, as well as improved model explainability and dataset reproducibility. The paper concludes by proposing a multidisciplinary research agenda that emphasises the need for further links to regulation and compliance technology, finer granularity in analysis, and the development of standardised datasets.
>
---
#### [new 072] LittleBit: Ultra Low-Bit Quantization via Latent Factorization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩任务，解决大语言模型部署中的高内存和计算成本问题。通过极低比特量化实现模型压缩，提升效率。**

- **链接: [http://arxiv.org/pdf/2506.13771v1](http://arxiv.org/pdf/2506.13771v1)**

> **作者:** Banseok Lee; Dongkyu Kim; Youngcheon You; Youngmin Kim
>
> **摘要:** Deploying large language models (LLMs) often faces challenges from substantial memory and computational costs. Quantization offers a solution, yet performance degradation in the sub-1-bit regime remains particularly difficult. This paper introduces LittleBit, a novel method for extreme LLM compression. It targets levels like 0.1 bits per weight (BPW), achieving nearly 31$\times$ memory reduction, e.g., Llama2-13B to under 0.9 GB. LittleBit represents weights in a low-rank form using latent matrix factorization, subsequently binarizing these factors. To counteract information loss from this extreme precision, it integrates a multi-scale compensation mechanism. This includes row, column, and an additional latent dimension that learns per-rank importance. Two key contributions enable effective training: Dual Sign-Value-Independent Decomposition (Dual-SVID) for stable quantization-aware training (QAT) initialization, and integrated Residual Compensation to mitigate errors. Extensive experiments confirm LittleBit's superiority in sub-1-bit quantization: e.g., its 0.1 BPW performance on Llama2-7B surpasses the leading method's 0.7 BPW. This establishes a superior size-performance trade-off, with kernel-level benchmarks indicating potential for a 5$\times$ speedup compared to FP16. LittleBit paves the way for deploying powerful LLMs in resource-constrained environments.
>
---
#### [new 073] Investigating the Potential of Large Language Model-Based Router Multi-Agent Architectures for Foundation Design Automation: A Task Classification and Expert Selection Study
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文属于基础设计自动化任务，旨在解决工程计算中的智能任务分类与专家选择问题，通过多智能体系统提升设计效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.13811v1](http://arxiv.org/pdf/2506.13811v1)**

> **作者:** Sompote Youwai; David Phim; Vianne Gayl Murcia; Rianne Clair Onas
>
> **摘要:** This study investigates router-based multi-agent systems for automating foundation design calculations through intelligent task classification and expert selection. Three approaches were evaluated: single-agent processing, multi-agent designer-checker architecture, and router-based expert selection. Performance assessment utilized baseline models including DeepSeek R1, ChatGPT 4 Turbo, Grok 3, and Gemini 2.5 Pro across shallow foundation and pile design scenarios. The router-based configuration achieved performance scores of 95.00% for shallow foundations and 90.63% for pile design, representing improvements of 8.75 and 3.13 percentage points over standalone Grok 3 performance respectively. The system outperformed conventional agentic workflows by 10.0 to 43.75 percentage points. Grok 3 demonstrated superior standalone performance without external computational tools, indicating advances in direct LLM mathematical reasoning for engineering applications. The dual-tier classification framework successfully distinguished foundation types, enabling appropriate analytical approaches. Results establish router-based multi-agent systems as optimal for foundation design automation while maintaining professional documentation standards. Given safety-critical requirements in civil engineering, continued human oversight remains essential, positioning these systems as advanced computational assistance tools rather than autonomous design replacements in professional practice.
>
---
#### [new 074] Knowledge Compression via Question Generation: Enhancing Multihop Document Retrieval without Fine-tuning
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，旨在提升多跳文档检索效果。通过生成问题进行知识压缩，无需微调即可提高检索效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.13778v1](http://arxiv.org/pdf/2506.13778v1)**

> **作者:** Anvi Alex Eponon; Moein Shahiki-Tash; Ildar Batyrshin; Christian E. Maldonado-Sifuentes; Grigori Sidorov; Alexander Gelbukh
>
> **摘要:** This study presents a question-based knowledge encoding approach that improves retrieval-augmented generation (RAG) systems without requiring fine-tuning or traditional chunking. We encode textual content using generated questions that span the lexical and semantic space, creating targeted retrieval cues combined with a custom syntactic reranking method. In single-hop retrieval over 109 scientific papers, our approach achieves a Recall@3 of 0.84, outperforming traditional chunking methods by 60 percent. We also introduce "paper-cards", concise paper summaries under 300 characters, which enhance BM25 retrieval, increasing MRR@3 from 0.56 to 0.85 on simplified technical queries. For multihop tasks, our reranking method reaches an F1 score of 0.52 with LLaMA2-Chat-7B on the LongBench 2WikiMultihopQA dataset, surpassing chunking and fine-tuned baselines which score 0.328 and 0.412 respectively. This method eliminates fine-tuning requirements, reduces retrieval latency, enables intuitive question-driven knowledge access, and decreases vector storage demands by 80%, positioning it as a scalable and efficient RAG alternative.
>
---
#### [new 075] AcademicBrowse: Benchmarking Academic Browse Ability of LLMs
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于学术信息检索任务，旨在解决现有基准不适用于学术搜索的问题。提出AcademicBrowse数据集，用于评估LLMs在复杂学术场景中的检索能力。**

- **链接: [http://arxiv.org/pdf/2506.13784v1](http://arxiv.org/pdf/2506.13784v1)**

> **作者:** Junting Zhou; Wang Li; Yiyan Liao; Nengyuan Zhang; Tingjia Miaoand Zhihui Qi; Yuhan Wu; Tong Yang
>
> **摘要:** Large Language Models (LLMs)' search capabilities have garnered significant attention. Existing benchmarks, such as OpenAI's BrowseComp, primarily focus on general search scenarios and fail to adequately address the specific demands of academic search. These demands include deeper literature tracing and organization, professional support for academic databases, the ability to navigate long-tail academic knowledge, and ensuring academic rigor. Here, we proposed AcademicBrowse, the first dataset specifically designed to evaluate the complex information retrieval capabilities of Large Language Models (LLMs) in academic research. AcademicBrowse possesses the following key characteristics: Academic Practicality, where question content closely mirrors real academic learning and research environments, avoiding deliberately misleading models; High Difficulty, with answers that are challenging for single models (e.g., Grok DeepSearch or Gemini Deep Research) to provide directly, often requiring at least three deep searches to derive; Concise Evaluation, where limiting conditions ensure answers are as unique as possible, accompanied by clear sources and brief solution explanations, greatly facilitating subsequent audit and verification, surpassing the current lack of analyzed search datasets both domestically and internationally; and Broad Coverage, as the dataset spans at least 15 different academic disciplines. Through AcademicBrowse, we expect to more precisely measure and promote the performance improvement of LLMs in complex academic information retrieval tasks. The data is available at: https://huggingface.co/datasets/PKU-DS-LAB/AcademicBrowse
>
---
#### [new 076] Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于音乐信息检索任务，解决吉他MIDI转谱表的问题。提出Fretting-Transformer模型，处理弦位歧义与可演奏性，提升转录准确性。**

- **链接: [http://arxiv.org/pdf/2506.14223v1](http://arxiv.org/pdf/2506.14223v1)**

> **作者:** Anna Hamberger; Sebastian Murgul; Jochen Schmidt; Michael Heizmann
>
> **备注:** Accepted to the 50th International Computer Music Conference (ICMC), 2025
>
> **摘要:** Music transcription plays a pivotal role in Music Information Retrieval (MIR), particularly for stringed instruments like the guitar, where symbolic music notations such as MIDI lack crucial playability information. This contribution introduces the Fretting-Transformer, an encoderdecoder model that utilizes a T5 transformer architecture to automate the transcription of MIDI sequences into guitar tablature. By framing the task as a symbolic translation problem, the model addresses key challenges, including string-fret ambiguity and physical playability. The proposed system leverages diverse datasets, including DadaGP, GuitarToday, and Leduc, with novel data pre-processing and tokenization strategies. We have developed metrics for tablature accuracy and playability to quantitatively evaluate the performance. The experimental results demonstrate that the Fretting-Transformer surpasses baseline methods like A* and commercial applications like Guitar Pro. The integration of context-sensitive processing and tuning/capo conditioning further enhances the model's performance, laying a robust foundation for future developments in automated guitar transcription.
>
---
#### [new 077] CRITICTOOL: Evaluating Self-Critique Capabilities of Large Language Models in Tool-Calling Error Scenarios
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于工具调用错误评估任务，旨在解决LLMs在复杂任务中处理工具调用错误的问题。提出CRITICTOOL基准，用于评估模型的自我批判能力。**

- **链接: [http://arxiv.org/pdf/2506.13977v1](http://arxiv.org/pdf/2506.13977v1)**

> **作者:** Shiting Huang; Zhen Fang; Zehui Chen; Siyu Yuan; Junjie Ye; Yu Zeng; Lin Chen; Qi Mao; Feng Zhao
>
> **摘要:** The ability of large language models (LLMs) to utilize external tools has enabled them to tackle an increasingly diverse range of tasks. However, as the tasks become more complex and long-horizon, the intricate tool utilization process may trigger various unexpected errors. Therefore, how to effectively handle such errors, including identifying, diagnosing, and recovering from them, has emerged as a key research direction for advancing tool learning. In this work, we first extensively analyze the types of errors encountered during the function-calling process on several competitive tool evaluation benchmarks. Based on it, we introduce CRITICTOOL, a comprehensive critique evaluation benchmark specialized for tool learning. Building upon a novel evolutionary strategy for dataset construction, CRITICTOOL holds diverse tool-use errors with varying complexities, which better reflects real-world scenarios. We conduct extensive experiments on CRITICTOOL, and validate the generalization and effectiveness of our constructed benchmark strategy. We also provide an in-depth analysis of the tool reflection ability on various LLMs, offering a new perspective on the field of tool learning in LLMs. The code is available at \href{https://github.com/Shellorley0513/CriticTool}{https://github.com/Shellorley0513/CriticTool}.
>
---
#### [new 078] Adaptive Guidance Accelerates Reinforcement Learning of Reasoning Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究强化学习中推理模型的泛化能力，解决模型难以解决新问题的问题。通过自蒸馏和引导算法提升性能。**

- **链接: [http://arxiv.org/pdf/2506.13923v1](http://arxiv.org/pdf/2506.13923v1)**

> **作者:** Vaskar Nath; Elaine Lau; Anisha Gunjal; Manasi Sharma; Nikhil Baharte; Sean Hendryx
>
> **摘要:** We study the process through which reasoning models trained with reinforcement learning on verifiable rewards (RLVR) can learn to solve new problems. We find that RLVR drives performance through two main means: (1) by compressing pass@$k$ into pass@1 and (2) via "capability gain" in which models learn to solve new problems that they previously could not solve even at high $k$. We find that while capability gain exists across model scales, learning to solve new problems is primarily driven through self-distillation. We demonstrate these findings across model scales ranging from 0.5B to 72B on >500,000 reasoning problems with prompts and verifiable final answers across math, science, and code domains. We further show that we can significantly improve pass@$k$ rates by leveraging natural language guidance for the model to consider within context while still requiring the model to derive a solution chain from scratch. Based of these insights, we derive $\text{Guide}$ - a new class of online training algorithms. $\text{Guide}$ adaptively incorporates hints into the model's context on problems for which all rollouts were initially incorrect and adjusts the importance sampling ratio for the "off-policy" trajectories in order to optimize the policy for contexts in which the hints are no longer present. We describe variants of $\text{Guide}$ for GRPO and PPO and empirically show that Guide-GRPO on 7B and 32B parameter models improves generalization over its vanilla counterpart with up to 4$\%$ macro-average improvement across math benchmarks. We include careful ablations to analyze $\text{Guide}$'s components and theoretically analyze Guide's learning efficiency.
>
---
#### [new 079] Acoustic scattering AI for non-invasive object classifications: A case study on hair assessment
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于非侵入式物体分类任务，旨在通过声波散射实现头发类型与湿度的识别，利用AI进行声音分类并验证多种方法。**

- **链接: [http://arxiv.org/pdf/2506.14148v1](http://arxiv.org/pdf/2506.14148v1)**

> **作者:** Long-Vu Hoang; Tuan Nguyen; Tran Huy Dat
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** This paper presents a novel non-invasive object classification approach using acoustic scattering, demonstrated through a case study on hair assessment. When an incident wave interacts with an object, it generates a scattered acoustic field encoding structural and material properties. By emitting acoustic stimuli and capturing the scattered signals from head-with-hair-sample objects, we classify hair type and moisture using AI-driven, deep-learning-based sound classification. We benchmark comprehensive methods, including (i) fully supervised deep learning, (ii) embedding-based classification, (iii) supervised foundation model fine-tuning, and (iv) self-supervised model fine-tuning. Our best strategy achieves nearly 90% classification accuracy by fine-tuning all parameters of a self-supervised model. These results highlight acoustic scattering as a privacy-preserving, non-contact alternative to visual classification, opening huge potential for applications in various industries.
>
---
#### [new 080] Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于机器学习任务，旨在解决RLVR在推理能力提升中的矛盾问题。通过引入新评估指标，证明RLVR能有效激励正确推理。**

- **链接: [http://arxiv.org/pdf/2506.14245v1](http://arxiv.org/pdf/2506.14245v1)**

> **作者:** Xumeng Wen; Zihan Liu; Shun Zheng; Zhijian Xu; Shengyu Ye; Zhirong Wu; Xiao Liang; Yang Wang; Junjie Li; Ziming Miao; Jiang Bian; Mao Yang
>
> **备注:** Preprint
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising paradigm for advancing the reasoning capabilities of Large Language Models (LLMs). However, a critical paradox clouds its efficacy: RLVR-tuned models often underperform their base models on the $Pass@K$ metric for solution-finding, leading to the hypothesis that RLVR merely re-weights existing reasoning paths at the cost of reasoning diversity. In this work, we resolve this contradiction by identifying the source of the problem: the $Pass@K$ metric itself is a flawed measure of reasoning, as it credits correct final answers that probably arise from inaccurate or incomplete chains of thought (CoTs). To address this, we introduce a more precise evaluation metric, $CoT$-$Pass@K$, which mandates that both the reasoning path and the final answer be correct. We provide a new theoretical foundation that formalizes how RLVR, unlike traditional RL, is uniquely structured to incentivize logical integrity. Our empirical results are supportive: using $CoT$-$Pass@K$, we observe that RLVR can incentivize the generalization of correct reasoning for all values of $K$. Furthermore, by analyzing the training dynamics, we find that this enhanced reasoning capability emerges early in the training process and smoothly generalizes. Our work provides a clear perspective on the role of RLVR, offers a more reliable method for its evaluation, and confirms its potential to genuinely advance machine reasoning.
>
---
#### [new 081] Innovating China's Intangible Cultural Heritage with DeepSeek + MidJourney: The Case of Yangliuqing theme Woodblock Prints
- **分类: cs.GR; cs.CL; cs.CY**

- **简介: 该论文属于文化创新任务，旨在将AI技术应用于传统艺术传承。通过结合DeepSeek与MidJourney生成抗疫主题的杨柳青年画，解决传统艺术现代化难题。**

- **链接: [http://arxiv.org/pdf/2506.14104v1](http://arxiv.org/pdf/2506.14104v1)**

> **作者:** RuiKun Yang; ZhongLiang Wei; Longdi Xian
>
> **摘要:** Yangliuqing woodblock prints, a cornerstone of China's intangible cultural heritage, are celebrated for their intricate designs and vibrant colors. However, preserving these traditional art forms while fostering innovation presents significant challenges. This study explores the DeepSeek + MidJourney approach to generating creative, themed Yangliuqing woodblock prints focused on the fight against COVID-19 and depicting joyous winners. Using Fr\'echet Inception Distance (FID) scores for evaluation, the method that combined DeepSeek-generated thematic prompts, MidJourney-generated thematic images, original Yangliuqing prints, and DeepSeek-generated key prompts in MidJourney-generated outputs achieved the lowest mean FID score (150.2) with minimal variability ({\sigma} = 4.9). Additionally, feedback from 62 participants, collected via questionnaires, confirmed that this hybrid approach produced the most representative results. Moreover, the questionnaire data revealed that participants demonstrated the highest willingness to promote traditional culture and the strongest interest in consuming the AI-generated images produced through this method. These findings underscore the effectiveness of an innovative approach that seamlessly blends traditional artistic elements with modern AI-driven creativity, ensuring both cultural preservation and contemporary relevance.
>
---
#### [new 082] ICE-ID: A Novel Historical Census Data Benchmark Comparing NARS against LLMs, \& a ML Ensemble on Longitudinal Identity Resolution
- **分类: cs.AI; cs.CL; cs.LG; stat.AP**

- **简介: 该论文提出ICE-ID基准数据集，用于历史身份解析任务，解决长期人口数据匹配问题。通过对比多种方法，验证了NARS的有效性。**

- **链接: [http://arxiv.org/pdf/2506.13792v1](http://arxiv.org/pdf/2506.13792v1)**

> **作者:** Gonçalo Hora de Carvalho; Lazar S. Popov; Sander Kaatee; Kristinn R. Thórisson; Tangrui Li; Pétur Húni Björnsson; Jilles S. Dibangoye
>
> **摘要:** We introduce ICE-ID, a novel benchmark dataset for historical identity resolution, comprising 220 years (1703-1920) of Icelandic census records. ICE-ID spans multiple generations of longitudinal data, capturing name variations, demographic changes, and rich genealogical links. To the best of our knowledge, this is the first large-scale, open tabular dataset specifically designed to study long-term person-entity matching in a real-world population. We define identity resolution tasks (within and across census waves) with clearly documented metrics and splits. We evaluate a range of methods: handcrafted rule-based matchers, a ML ensemble as well as LLMs for structured data (e.g. transformer-based tabular networks) against a novel approach to tabular data called NARS (Non-Axiomatic Reasoning System) - a general-purpose AI framework designed to reason with limited knowledge and resources. Its core is Non-Axiomatic Logic (NAL), a term-based logic. Our experiments show that NARS is suprisingly simple and competitive with other standard approaches, achieving SOTA at our task. By releasing ICE-ID and our code, we enable reproducible benchmarking of identity resolution approaches in longitudinal settings and hope that ICE-ID opens new avenues for cross-disciplinary research in data linkage and historical analytics.
>
---
#### [new 083] RadFabric: Agentic AI System with Reasoning Capability for Radiology
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医学影像分析任务，旨在解决放射科诊断中病理覆盖不足、准确率低的问题。提出RadFabric系统，融合视觉与文本推理，提升诊断准确性与透明度。**

- **链接: [http://arxiv.org/pdf/2506.14142v1](http://arxiv.org/pdf/2506.14142v1)**

> **作者:** Wenting Chen; Yi Dong; Zhaojun Ding; Yucheng Shi; Yifan Zhou; Fang Zeng; Yijun Luo; Tianyu Lin; Yihang Su; Yichen Wu; Kai Zhang; Zhen Xiang; Tianming Liu; Ninghao Liu; Lichao Sun; Yixuan Yuan; Xiang Li
>
> **备注:** 4 figures, 2 tables
>
> **摘要:** Chest X ray (CXR) imaging remains a critical diagnostic tool for thoracic conditions, but current automated systems face limitations in pathology coverage, diagnostic accuracy, and integration of visual and textual reasoning. To address these gaps, we propose RadFabric, a multi agent, multimodal reasoning framework that unifies visual and textual analysis for comprehensive CXR interpretation. RadFabric is built on the Model Context Protocol (MCP), enabling modularity, interoperability, and scalability for seamless integration of new diagnostic agents. The system employs specialized CXR agents for pathology detection, an Anatomical Interpretation Agent to map visual findings to precise anatomical structures, and a Reasoning Agent powered by large multimodal reasoning models to synthesize visual, anatomical, and clinical data into transparent and evidence based diagnoses. RadFabric achieves significant performance improvements, with near-perfect detection of challenging pathologies like fractures (1.000 accuracy) and superior overall diagnostic accuracy (0.799) compared to traditional systems (0.229 to 0.527). By integrating cross modal feature alignment and preference-driven reasoning, RadFabric advances AI-driven radiology toward transparent, anatomically precise, and clinically actionable CXR analysis.
>
---
#### [new 084] TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型对齐任务，解决DPO难以利用token级奖励的问题，提出TGDPO框架提升性能。**

- **链接: [http://arxiv.org/pdf/2506.14574v1](http://arxiv.org/pdf/2506.14574v1)**

> **作者:** Mingkang Zhu; Xi Chen; Zhongdao Wang; Bei Yu; Hengshuang Zhao; Jiaya Jia
>
> **备注:** ICML 2025
>
> **摘要:** Recent advancements in reinforcement learning from human feedback have shown that utilizing fine-grained token-level reward models can substantially enhance the performance of Proximal Policy Optimization (PPO) in aligning large language models. However, it is challenging to leverage such token-level reward as guidance for Direct Preference Optimization (DPO), since DPO is formulated as a sequence-level bandit problem. To address this challenge, this work decomposes the sequence-level PPO into a sequence of token-level proximal policy optimization problems and then frames the problem of token-level PPO with token-level reward guidance, from which closed-form optimal token-level policy and the corresponding token-level reward can be derived. Using the obtained reward and Bradley-Terry model, this work establishes a framework of computable loss functions with token-level reward guidance for DPO, and proposes a practical reward guidance based on the induced DPO reward. This formulation enables different tokens to exhibit varying degrees of deviation from reference policy based on their respective rewards. Experiment results demonstrate that our method achieves substantial performance improvements over DPO, with win rate gains of up to 7.5 points on MT-Bench, 6.2 points on AlpacaEval 2, and 4.3 points on Arena-Hard. Code is available at https://github.com/dvlab-research/TGDPO.
>
---
## 更新

#### [replaced 001] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.01413v4](http://arxiv.org/pdf/2506.01413v4)**

> **作者:** Yulei Qin; Gang Li; Zongyi Li; Zihan Xu; Yuchen Shi; Zhekai Lin; Xiao Cui; Ke Li; Xing Sun
>
> **备注:** 13 pages of main body, 3 tables, 5 figures, 45 pages of appendix
>
> **摘要:** Existing large language models (LLMs) face challenges of following complex instructions, especially when multiple constraints are present and organized in paralleling, chaining, and branching structures. One intuitive solution, namely chain-of-thought (CoT), is expected to universally improve capabilities of LLMs. However, we find that the vanilla CoT exerts a negative impact on performance due to its superficial reasoning pattern of simply paraphrasing the instructions. It fails to peel back the compositions of constraints for identifying their relationship across hierarchies of types and dimensions. To this end, we propose a systematic method to boost LLMs in dealing with complex instructions via incentivizing reasoning for test-time compute scaling. First, we stem from the decomposition of complex instructions under existing taxonomies and propose a reproducible data acquisition method. Second, we exploit reinforcement learning (RL) with verifiable rule-centric reward signals to cultivate reasoning specifically for instruction following. We address the shallow, non-essential nature of reasoning under complex instructions via sample-wise contrast for superior CoT enforcement. We also exploit behavior cloning of experts to facilitate steady distribution shift from fast-thinking LLMs to skillful reasoners. Extensive evaluations on seven comprehensive benchmarks confirm the validity of the proposed method, where a 1.5B LLM achieves 11.74% gains with performance comparable to a 8B LLM. Codes and data will be available later (under review). Keywords: reinforcement learning with verifiable rewards (RLVR), instruction following, complex instructions
>
---
#### [replaced 002] Seewo's Submission to MLC-SLM: Lessons learned from Speech Reasoning Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.13300v2](http://arxiv.org/pdf/2506.13300v2)**

> **作者:** Bo Li; Chengben Xu; Wufeng Zhang
>
> **摘要:** This paper presents Seewo's systems for both tracks of the Multilingual Conversational Speech Language Model Challenge (MLC-SLM), addressing automatic speech recognition (ASR) and speaker diarization with ASR (SD-ASR). We introduce a multi-stage training pipeline that explicitly enhances reasoning and self-correction in speech language models for ASR. Our approach combines curriculum learning for progressive capability acquisition, Chain-of-Thought data augmentation to foster intermediate reflection, and Reinforcement Learning with Verifiable Rewards (RLVR) to further refine self-correction through reward-driven optimization. This approach achieves substantial improvements over the official challenge baselines. On the evaluation set, our best system attains a WER/CER of 11.57% for Track 1 and a tcpWER/tcpCER of 17.67% for Track 2. Comprehensive ablation studies demonstrate the effectiveness of each component under challenge constraints.
>
---
#### [replaced 003] Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2406.13677v3](http://arxiv.org/pdf/2406.13677v3)**

> **作者:** Erik Derner; Sara Sansalvador de la Fuente; Yoan Gutiérrez; Paloma Moreda; Nuria Oliver
>
> **备注:** Accepted for presentation at the 6th Workshop on Gender Bias in Natural Language Processing (GeBNLP) at ACL 2025
>
> **摘要:** Large language models (LLMs) often inherit and amplify social biases embedded in their training data. A prominent social bias is gender bias. In this regard, prior work has mainly focused on gender stereotyping bias - the association of specific roles or traits with a particular gender - in English and on evaluating gender bias in model embeddings or generated outputs. In contrast, gender representation bias - the unequal frequency of references to individuals of different genders - in the training corpora has received less attention. Yet such imbalances in the training data constitute an upstream source of bias that can propagate and intensify throughout the entire model lifecycle. To fill this gap, we propose a novel LLM-based method to detect and quantify gender representation bias in LLM training data in gendered languages, where grammatical gender challenges the applicability of methods developed for English. By leveraging the LLMs' contextual understanding, our approach automatically identifies and classifies person-referencing words in gendered language corpora. Applied to four Spanish-English benchmarks and five Valencian corpora, our method reveals substantial male-dominant imbalances. We show that such biases in training data affect model outputs, but can surprisingly be mitigated leveraging small-scale training on datasets that are biased towards the opposite gender. Our findings highlight the need for corpus-level gender bias analysis in multilingual NLP. We make our code and data publicly available.
>
---
#### [replaced 004] MultiMatch: Multihead Consistency Regularization Matching for Semi-Supervised Text Classification
- **分类: cs.CL; cs.AI; cs.LG; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.07801v2](http://arxiv.org/pdf/2506.07801v2)**

> **作者:** Iustin Sirbu; Robert-Adrian Popovici; Cornelia Caragea; Stefan Trausan-Matu; Traian Rebedea
>
> **摘要:** We introduce MultiMatch, a novel semi-supervised learning (SSL) algorithm combining the paradigms of co-training and consistency regularization with pseudo-labeling. At its core, MultiMatch features a three-fold pseudo-label weighting module designed for three key purposes: selecting and filtering pseudo-labels based on head agreement and model confidence, and weighting them according to the perceived classification difficulty. This novel module enhances and unifies three existing techniques -- heads agreement from Multihead Co-training, self-adaptive thresholds from FreeMatch, and Average Pseudo-Margins from MarginMatch -- resulting in a holistic approach that improves robustness and performance in SSL settings. Experimental results on benchmark datasets highlight the superior performance of MultiMatch, achieving state-of-the-art results on 9 out of 10 setups from 5 natural language processing datasets and ranking first according to the Friedman test among 19 methods. Furthermore, MultiMatch demonstrates exceptional robustness in highly imbalanced settings, outperforming the second-best approach by 3.26% -- and data imbalance is a key factor for many text classification tasks.
>
---
#### [replaced 005] Rectifying Belief Space via Unlearning to Harness LLMs' Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20620v2](http://arxiv.org/pdf/2502.20620v2)**

> **作者:** Ayana Niwa; Masahiro Kaneko; Kentaro Inui
>
> **备注:** Accepted at ACL2025 Findings (long)
>
> **摘要:** Large language models (LLMs) can exhibit advanced reasoning yet still generate incorrect answers. We hypothesize that such errors frequently stem from spurious beliefs, propositions the model internally considers true but are incorrect. To address this, we propose a method to rectify the belief space by suppressing these spurious beliefs while simultaneously enhancing true ones, thereby enabling more reliable inferences. Our approach first identifies the beliefs that lead to incorrect or correct answers by prompting the model to generate textual explanations, using our Forward-Backward Beam Search (FBBS). We then apply unlearning to suppress the identified spurious beliefs and enhance the true ones, effectively rectifying the model's belief space. Empirical results on multiple QA datasets and LLMs show that our method corrects previously misanswered questions without harming overall model performance. Furthermore, our approach yields improved generalization on unseen data, suggesting that rectifying a model's belief space is a promising direction for mitigating errors and enhancing overall reliability.
>
---
#### [replaced 006] Controllable and Reliable Knowledge-Intensive Task-Oriented Conversational Agents with Declarative Genie Worksheets
- **分类: cs.AI; cs.CL; cs.PL**

- **链接: [http://arxiv.org/pdf/2407.05674v3](http://arxiv.org/pdf/2407.05674v3)**

> **作者:** Harshit Joshi; Shicheng Liu; James Chen; Robert Weigle; Monica S. Lam
>
> **备注:** Accepted at ACL 2025
>
> **摘要:** Large Language Models can carry out human-like conversations in diverse settings, responding to user requests for tasks and knowledge. However, existing conversational agents implemented with LLMs often struggle with hallucination, following instructions with conditional logic, and integrating knowledge from different sources. These shortcomings compromise the agents' effectiveness, rendering them unsuitable for deployment. To address these challenges, we introduce Genie, a programmable framework for creating knowledge-intensive task-oriented conversational agents. Genie can handle involved interactions and answer complex queries. Unlike LLMs, it delivers reliable, grounded responses through advanced dialogue state management and supports controllable agent policies via its declarative specification -- Genie Worksheet. This is achieved through an algorithmic runtime system that implements the developer-supplied policy, limiting LLMs to (1) parse user input using a succinct conversational history, and (2) generate responses according to supplied context. Agents built with Genie outperform SOTA methods on complex logic dialogue datasets. We conducted a user study with 62 participants on three real-life applications: restaurant reservations with Yelp, as well as ticket submission and course enrollment for university students. Genie agents with GPT-4 Turbo outperformed the GPT-4 Turbo agents with function calling, improving goal completion rates from 21.8% to 82.8% across three real-world tasks.
>
---
#### [replaced 007] Bridging Social Media and Search Engines: Dredge Words and the Detection of Unreliable Domains
- **分类: cs.SI; cs.AI; cs.CL; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.11423v4](http://arxiv.org/pdf/2406.11423v4)**

> **作者:** Evan M. Williams; Peter Carragher; Kathleen M. Carley
>
> **摘要:** Proactive content moderation requires platforms to rapidly and continuously evaluate the credibility of websites. Leveraging the direct and indirect paths users follow to unreliable websites, we develop a website credibility classification and discovery system that integrates both webgraph and large-scale social media contexts. We additionally introduce the concept of dredge words, terms or phrases for which unreliable domains rank highly on search engines, and provide the first exploration of their usage on social media. Our graph neural networks that combine webgraph and social media contexts generate to state-of-the-art results in website credibility classification and significantly improves the top-k identification of unreliable domains. Additionally, we release a novel dataset of dredge words, highlighting their strong connections to both social media and online commerce platforms.
>
---
#### [replaced 008] The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with LLMs
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2501.10970v3](http://arxiv.org/pdf/2501.10970v3)**

> **作者:** Nitay Calderon; Roi Reichart; Rotem Dror
>
> **摘要:** The "LLM-as-an-annotator" and "LLM-as-a-judge" paradigms employ Large Language Models (LLMs) as annotators, judges, and evaluators in tasks traditionally performed by humans. LLM annotations are widely used, not only in NLP research but also in fields like medicine, psychology, and social science. Despite their role in shaping study results and insights, there is no standard or rigorous procedure to determine whether LLMs can replace human annotators. In this paper, we propose a novel statistical procedure, the Alternative Annotator Test (alt-test), that requires only a modest subset of annotated examples to justify using LLM annotations. Additionally, we introduce a versatile and interpretable measure for comparing LLM annotators and judges. To demonstrate our procedure, we curated a diverse collection of ten datasets, consisting of language and vision-language tasks, and conducted experiments with six LLMs and four prompting techniques. Our results show that LLMs can sometimes replace humans with closed-source LLMs (such as GPT-4o), outperforming the open-source LLMs we examine, and that prompting techniques yield judges of varying quality. We hope this study encourages more rigorous and reliable practices.
>
---
#### [replaced 009] Towards Geo-Culturally Grounded LLM Generations
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.13497v3](http://arxiv.org/pdf/2502.13497v3)**

> **作者:** Piyawat Lertvittayakumjorn; David Kinney; Vinodkumar Prabhakaran; Donald Martin Jr.; Sunipa Dev
>
> **备注:** ACL 2025 (main conference)
>
> **摘要:** Generative large language models (LLMs) have demonstrated gaps in diverse cultural awareness across the globe. We investigate the effect of retrieval augmented generation and search-grounding techniques on LLMs' ability to display familiarity with various national cultures. Specifically, we compare the performance of standard LLMs, LLMs augmented with retrievals from a bespoke knowledge base (i.e., KB grounding), and LLMs augmented with retrievals from a web search (i.e., search grounding) on multiple cultural awareness benchmarks. We find that search grounding significantly improves the LLM performance on multiple-choice benchmarks that test propositional knowledge (e.g., cultural norms, artifacts, and institutions), while KB grounding's effectiveness is limited by inadequate knowledge base coverage and a suboptimal retriever. However, search grounding also increases the risk of stereotypical judgments by language models and fails to improve evaluators' judgments of cultural familiarity in a human evaluation with adequate statistical power. These results highlight the distinction between propositional cultural knowledge and open-ended cultural fluency when it comes to evaluating LLMs' cultural awareness.
>
---
#### [replaced 010] From tools to thieves: Measuring and understanding public perceptions of AI through crowdsourced metaphors
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2501.18045v3](http://arxiv.org/pdf/2501.18045v3)**

> **作者:** Myra Cheng; Angela Y. Lee; Kristina Rapuano; Kate Niederhoffer; Alex Liebscher; Jeffrey Hancock
>
> **备注:** To appear at the ACM Conference on Fairness, Accountability, and Transparency 2025
>
> **摘要:** How has the public responded to the increasing prevalence of artificial intelligence (AI)-based technologies? We investigate public perceptions of AI by collecting over 12,000 responses over 12 months from a nationally representative U.S. sample. Participants provided open-ended metaphors reflecting their mental models of AI, a methodology that overcomes the limitations of traditional self-reported measures by capturing more nuance. Using a mixed-methods approach combining quantitative clustering and qualitative coding, we identify 20 dominant metaphors shaping public understanding of AI. To analyze these metaphors systematically, we present a scalable framework integrating language modeling (LM)-based techniques to measure key dimensions of public perception: anthropomorphism (attribution of human-like qualities), warmth, and competence. We find that Americans generally view AI as warm and competent, and that over the past year, perceptions of AI's human-likeness and warmth have significantly increased ($+34\%, r = 0.80, p < 0.01; +41\%, r = 0.62, p < 0.05$). These implicit perceptions, along with the identified dominant metaphors, strongly predict trust in and willingness to adopt AI ($r^2 = 0.21, 0.18, p < 0.001$). Moreover, we uncover systematic demographic differences in metaphors and implicit perceptions, such as the higher propensity of women, older individuals, and people of color to anthropomorphize AI, which shed light on demographic disparities in trust and adoption. In addition to our dataset and framework for tracking evolving public attitudes, we provide actionable insights on using metaphors for inclusive and responsible AI development.
>
---
#### [replaced 011] Automated Construction of a Knowledge Graph of Nuclear Fusion Energy for Effective Elicitation and Retrieval of Information
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07738v2](http://arxiv.org/pdf/2504.07738v2)**

> **作者:** Andrea Loreti; Kesi Chen; Ruby George; Robert Firth; Adriano Agnello; Shinnosuke Tanaka
>
> **摘要:** In this document, we discuss a multi-step approach to automated construction of a knowledge graph, for structuring and representing domain-specific knowledge from large document corpora. We apply our method to build the first knowledge graph of nuclear fusion energy, a highly specialized field characterized by vast scope and heterogeneity. This is an ideal benchmark to test the key features of our pipeline, including automatic named entity recognition and entity resolution. We show how pre-trained large language models can be used to address these challenges and we evaluate their performance against Zipf's law, which characterizes human-generated natural language. Additionally, we develop a knowledge-graph retrieval-augmented generation system that combines large language models with a multi-prompt approach. This system provides contextually relevant answers to natural-language queries, including complex multi-hop questions that require reasoning across interconnected entities.
>
---
#### [replaced 012] Assessing Consistency and Reproducibility in the Outputs of Large Language Models: Evidence Across Diverse Finance and Accounting Tasks
- **分类: q-fin.GN; cs.AI; cs.CE; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.16974v3](http://arxiv.org/pdf/2503.16974v3)**

> **作者:** Julian Junyan Wang; Victor Xiaoqi Wang
>
> **备注:** 89 pages, 20 tables, 15 figures
>
> **摘要:** This study provides the first comprehensive assessment of consistency and reproducibility in Large Language Model (LLM) outputs in finance and accounting research. We evaluate how consistently LLMs produce outputs given identical inputs through extensive experimentation with 50 independent runs across five common tasks: classification, sentiment analysis, summarization, text generation, and prediction. Using three OpenAI models (GPT-3.5-turbo, GPT-4o-mini, and GPT-4o), we generate over 3.4 million outputs from diverse financial source texts and data, covering MD&As, FOMC statements, finance news articles, earnings call transcripts, and financial statements. Our findings reveal substantial but task-dependent consistency, with binary classification and sentiment analysis achieving near-perfect reproducibility, while complex tasks show greater variability. More advanced models do not consistently demonstrate better consistency and reproducibility, with task-specific patterns emerging. LLMs significantly outperform expert human annotators in consistency and maintain high agreement even where human experts significantly disagree. We further find that simple aggregation strategies across 3-5 runs dramatically improve consistency. We also find that aggregation may come with an additional benefit of improved accuracy for sentiment analysis when using newer models. Simulation analysis reveals that despite measurable inconsistency in LLM outputs, downstream statistical inferences remain remarkably robust. These findings address concerns about what we term "G-hacking," the selective reporting of favorable outcomes from multiple Generative AI runs, by demonstrating that such risks are relatively low for finance and accounting tasks.
>
---
#### [replaced 013] Scaling Computer-Use Grounding via User Interface Decomposition and Synthesis
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.13227v2](http://arxiv.org/pdf/2505.13227v2)**

> **作者:** Tianbao Xie; Jiaqi Deng; Xiaochuan Li; Junlin Yang; Haoyuan Wu; Jixuan Chen; Wenjing Hu; Xinyuan Wang; Yuhui Xu; Zekun Wang; Yiheng Xu; Junli Wang; Doyen Sahoo; Tao Yu; Caiming Xiong
>
> **备注:** 49 pages, 13 figures
>
> **摘要:** Graphical user interface (GUI) grounding, the ability to map natural language instructions to specific actions on graphical user interfaces, remains a critical bottleneck in computer use agent development. Current benchmarks oversimplify grounding tasks as short referring expressions, failing to capture the complexity of real-world interactions that require software commonsense, layout understanding, and fine-grained manipulation capabilities. To address these limitations, we introduce OSWorld-G, a comprehensive benchmark comprising 564 finely annotated samples across diverse task types including text matching, element recognition, layout understanding, and precise manipulation. Additionally, we synthesize and release the largest computer use grounding dataset Jedi, which contains 4 million examples through multi-perspective decoupling of tasks. Our multi-scale models trained on Jedi demonstrate its effectiveness by outperforming existing approaches on ScreenSpot-v2, ScreenSpot-Pro, and our OSWorld-G. Furthermore, we demonstrate that improved grounding with Jedi directly enhances agentic capabilities of general foundation models on complex computer tasks, improving from 5% to 27% on OSWorld. Through detailed ablation studies, we identify key factors contributing to grounding performance and verify that combining specialized data for different interface elements enables compositional generalization to novel interfaces. All benchmark, data, checkpoints, and code are open-sourced and available at https://osworld-grounding.github.io.
>
---
#### [replaced 014] ClusterChat: Multi-Feature Search for Corpus Exploration
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.14533v2](http://arxiv.org/pdf/2412.14533v2)**

> **作者:** Ashish Chouhan; Saifeldin Mandour; Michael Gertz
>
> **备注:** 5 pages, 1 table, 1 figure, Accepted to SIGIR Demo Paper Track 2025
>
> **摘要:** Exploring large-scale text corpora presents a significant challenge in biomedical, finance, and legal domains, where vast amounts of documents are continuously published. Traditional search methods, such as keyword-based search, often retrieve documents in isolation, limiting the user's ability to easily inspect corpus-wide trends and relationships. We present ClusterChat (The demo video and source code are available at: https://github.com/achouhan93/ClusterChat), an open-source system for corpus exploration that integrates cluster-based organization of documents using textual embeddings with lexical and semantic search, timeline-driven exploration, and corpus and document-level question answering (QA) as multi-feature search capabilities. We validate the system with two case studies on a four million abstract PubMed dataset, demonstrating that ClusterChat enhances corpus exploration by delivering context-aware insights while maintaining scalability and responsiveness on large-scale document collections.
>
---
#### [replaced 015] Evolution of ESG-focused DLT Research: An NLP Analysis of the Literature
- **分类: cs.IR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2308.12420v4](http://arxiv.org/pdf/2308.12420v4)**

> **作者:** Walter Hernandez Cruz; Kamil Tylinski; Alastair Moore; Niall Roche; Nikhil Vadgama; Horst Treiblmaier; Jiangbo Shangguan; Paolo Tasca; Jiahua Xu
>
> **摘要:** Distributed Ledger Technology (DLT) faces increasing environmental scrutiny, particularly concerning the energy consumption of the Proof of Work (PoW) consensus mechanism and broader Environmental, Social, and Governance (ESG) issues. However, existing systematic literature reviews of DLT rely on limited analyses of citations, abstracts, and keywords, failing to fully capture the field's complexity and ESG concerns. We address these challenges by analyzing the full text of 24,539 publications using Natural Language Processing (NLP) with our manually labeled Named Entity Recognition (NER) dataset of 39,427 entities for DLT. This methodology identified 505 key publications at the DLT/ESG intersection, enabling comprehensive domain analysis. Our combined NLP and temporal graph analysis reveals critical trends in DLT evolution and ESG impacts, including cryptography and peer-to-peer networks research's foundational influence, Bitcoin's persistent impact on research and environmental concerns (a "Lindy effect"), Ethereum's catalytic role on Proof of Stake (PoS) and smart contract adoption, and the industry's progressive shift toward energy-efficient consensus mechanisms. Our contributions include the first DLT-specific NER dataset addressing the scarcity of high-quality labeled NLP data in blockchain research, a methodology integrating NLP and temporal graph analysis for large-scale interdisciplinary literature reviews, and the first NLP-driven literature review focusing on DLT's ESG aspects.
>
---
#### [replaced 016] Conformal Linguistic Calibration: Trading-off between Factuality and Specificity
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19110v3](http://arxiv.org/pdf/2502.19110v3)**

> **作者:** Zhengping Jiang; Anqi Liu; Benjamin Van Durme
>
> **摘要:** Language model outputs are not always reliable, thus prompting research into how to adapt model responses based on uncertainty. Common approaches include: \emph{abstention}, where models refrain from generating responses when uncertain; and \emph{linguistic calibration}, where models hedge their statements using uncertainty quantifiers. However, abstention can withhold valuable information, while linguistically calibrated responses are often challenging to leverage in downstream tasks. We propose a unified view, Conformal Linguistic Calibration (CLC), which reinterprets linguistic calibration as \emph{answer set prediction}. First we present a framework connecting abstention and linguistic calibration through the lens of linguistic pragmatics. We then describe an implementation of CLC that allows for controlling the level of imprecision in model responses. Results demonstrate our method produces calibrated outputs with conformal guarantees on factual accuracy. Further, our approach enables fine-tuning models to perform uncertainty-aware adaptive claim rewriting, offering a controllable balance between factuality and specificity.
>
---
#### [replaced 017] CAPTURE: Context-Aware Prompt Injection Testing and Robustness Enhancement
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.12368v2](http://arxiv.org/pdf/2505.12368v2)**

> **作者:** Gauri Kholkar; Ratinder Ahuja
>
> **备注:** Accepted in ACL LLMSec Workshop 2025
>
> **摘要:** Prompt injection remains a major security risk for large language models. However, the efficacy of existing guardrail models in context-aware settings remains underexplored, as they often rely on static attack benchmarks. Additionally, they have over-defense tendencies. We introduce CAPTURE, a novel context-aware benchmark assessing both attack detection and over-defense tendencies with minimal in-domain examples. Our experiments reveal that current prompt injection guardrail models suffer from high false negatives in adversarial cases and excessive false positives in benign scenarios, highlighting critical limitations. To demonstrate our framework's utility, we train CaptureGuard on our generated data. This new model drastically reduces both false negative and false positive rates on our context-aware datasets while also generalizing effectively to external benchmarks, establishing a path toward more robust and practical prompt injection defenses.
>
---
#### [replaced 018] Surprise Calibration for Better In-Context Learning
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.12796v2](http://arxiv.org/pdf/2506.12796v2)**

> **作者:** Zhihang Tan; Jingrui Hou; Ping Wang; Qibiao Hu; Peng Zhu
>
> **备注:** 16 pages, 11 figures
>
> **摘要:** In-context learning (ICL) has emerged as a powerful paradigm for task adaptation in large language models (LLMs), where models infer underlying task structures from a few demonstrations. However, ICL remains susceptible to biases that arise from prior knowledge and contextual demonstrations, which can degrade the performance of LLMs. Existing bias calibration methods typically apply fixed class priors across all inputs, limiting their efficacy in dynamic ICL settings where the context for each query differs. To address these limitations, we adopt implicit sequential Bayesian inference as a framework for interpreting ICL, identify "surprise" as an informative signal for class prior shift, and introduce a novel method--Surprise Calibration (SC). SC leverages the notion of surprise to capture the temporal dynamics of class priors, providing a more adaptive and computationally efficient solution for in-context learning. We empirically demonstrate the superiority of SC over existing bias calibration techniques across a range of benchmark natural language processing tasks.
>
---
#### [replaced 019] Enhancing Goal-oriented Proactive Dialogue Systems via Consistency Reflection and Correction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.13366v2](http://arxiv.org/pdf/2506.13366v2)**

> **作者:** Didi Zhang; Yaxin Fan; Peifeng Li; Qiaoming Zhu
>
> **摘要:** Goal-oriented proactive dialogue systems are designed to guide user conversations seamlessly towards specific objectives by planning a goal-oriented path. However, previous research has focused predominantly on optimizing these paths while neglecting the inconsistencies that may arise between generated responses and dialogue contexts, including user profiles, dialogue history, domain knowledge, and subgoals. To address this issue, we introduce a model-agnostic two-stage Consistency Reflection and Correction (CRC) framework. Specifically, in the consistency reflection stage, the model is prompted to reflect on the discrepancies between generated responses and dialogue contexts, identifying inconsistencies and suggesting possible corrections. In the consistency correction stage, the model generates responses that are more consistent with the dialogue context based on these reflection results. We conducted experiments on various model architectures with different parameter sizes, including encoder-decoder models (BART, T5) and decoder-only models (GPT-2, DialoGPT, Phi3, Mistral and LLaMA3), and the experimental results on three datasets demonstrate that our CRC framework significantly improves the consistency between generated responses and dialogue contexts.
>
---
#### [replaced 020] SOPBench: Evaluating Language Agents at Following Standard Operating Procedures and Constraints
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.08669v2](http://arxiv.org/pdf/2503.08669v2)**

> **作者:** Zekun Li; Shinda Huang; Jiangtian Wang; Nathan Zhang; Antonis Antoniades; Wenyue Hua; Kaijie Zhu; Sirui Zeng; Chi Wang; William Yang Wang; Xifeng Yan
>
> **备注:** Code, data, and over 24k agent trajectories are released at https://github.com/Leezekun/SOPBench
>
> **摘要:** As language agents increasingly automate critical tasks, their ability to follow domain-specific standard operating procedures (SOPs), policies, and constraints when taking actions and making tool calls becomes essential yet remains underexplored. To address this gap, we develop an automated evaluation pipeline SOPBench with: (1) executable environments containing 167 tools/functions across seven customer service domains with service-specific SOPs and rule-based verifiers, (2) an automated test generation framework producing over 900 verified test cases, and (3) an automated evaluation framework to rigorously assess agent adherence from multiple dimensions. Our approach transforms each service-specific SOP code program into a directed graph of executable functions and requires agents to call these functions based on natural language SOP descriptions. The original code serves as oracle rule-based verifiers to assess compliance, reducing reliance on manual annotations and LLM-based evaluations. We evaluate 18 leading models, and results show the task is challenging even for top-tier models (like GPT-4o, Claude-3.7-Sonnet), with variances across domains. Reasoning models like o4-mini-high show superiority while other powerful models perform less effectively (pass rates of 30%-50%), and small models (7B, 8B) perform significantly worse. Additionally, language agents can be easily jailbroken to overlook SOPs and constraints. Code, data, and over 24k agent trajectories are released at https://github.com/Leezekun/SOPBench.
>
---
#### [replaced 021] Do Construction Distributions Shape Formal Language Learning In German BabyLMs?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.11593v2](http://arxiv.org/pdf/2503.11593v2)**

> **作者:** Bastian Bunzeck; Daniel Duran; Sina Zarrieß
>
> **备注:** Accepted at CoNNL 2025
>
> **摘要:** We analyze the influence of utterance-level construction distributions in German child-directed/child-available speech on the resulting word-level, syntactic and semantic competence (and their underlying learning trajectories) in small LMs, which we train on a novel collection of developmentally plausible language data for German. We find that trajectories are surprisingly robust for markedly different distributions of constructions in the training data, which have little effect on final accuracies and almost no effect on global learning trajectories. While syntax learning benefits from more complex utterances, word-level learning culminates in better scores with more fragmentary utterances. We argue that LMs trained on developmentally plausible data can contribute to debates on how conducive different kinds of linguistic stimuli are to language learning.
>
---
#### [replaced 022] ETM: Modern Insights into Perspective on Text-to-SQL Evaluation in the Age of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.07313v4](http://arxiv.org/pdf/2407.07313v4)**

> **作者:** Benjamin G. Ascoli; Yasoda Sai Ram Kandikonda; Jinho D. Choi
>
> **摘要:** The task of Text-to-SQL enables anyone to retrieve information from SQL databases using natural language. While this task has made substantial progress, the two primary evaluation metrics - Execution Accuracy (EXE) and Exact Set Matching Accuracy (ESM) - suffer from inherent limitations that can misrepresent performance. Specifically, ESM's rigid matching overlooks semantically correct but stylistically different queries, whereas EXE can overestimate correctness by ignoring structural errors that yield correct outputs. These shortcomings become especially problematic when assessing outputs from large language model (LLM)-based approaches without fine-tuning, which vary more in style and structure compared to their fine-tuned counterparts. Thus, we introduce a new metric, Enhanced Tree Matching (ETM), which mitigates these issues by comparing queries using both syntactic and semantic elements. Through evaluating nine LLM-based models, we show that EXE and ESM can produce false positive and negative rates as high as 23.0% and 28.9%, while ETM reduces these rates to 0.3% and 2.7%, respectively. We release our ETM script as open source, offering the community a more robust and reliable approach to evaluating Text-to-SQL.
>
---
#### [replaced 023] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15910v2](http://arxiv.org/pdf/2502.15910v2)**

> **作者:** Zheyuan Liu; Guangyao Dou; Xiangchi Yuan; Chunhui Zhang; Zhaoxuan Tan; Meng Jiang
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** Generative models such as Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) trained on massive datasets can lead them to memorize and inadvertently reveal sensitive information, raising ethical and privacy concerns. While some prior works have explored this issue in the context of LLMs, it presents a unique challenge for MLLMs due to the entangled nature of knowledge across modalities, making comprehensive unlearning more difficult. To address this challenge, we propose Modality Aware Neuron Unlearning (MANU), a novel unlearning framework for MLLMs designed to selectively clip neurons based on their relative importance to the targeted forget data, curated for different modalities. Specifically, MANU consists of two stages: important neuron selection and selective pruning. The first stage identifies and collects the most influential neurons across modalities relative to the targeted forget knowledge, while the second stage is dedicated to pruning those selected neurons. MANU effectively isolates and removes the neurons that contribute most to the forget data within each modality, while preserving the integrity of retained knowledge. Our experiments conducted across various MLLM architectures illustrate that MANU can achieve a more balanced and comprehensive unlearning in each modality without largely affecting the overall model utility.
>
---
#### [replaced 024] REAL-Prover: Retrieval Augmented Lean Prover for Mathematical Reasoning
- **分类: cs.CL; cs.AI; cs.LG; cs.LO**

- **链接: [http://arxiv.org/pdf/2505.20613v2](http://arxiv.org/pdf/2505.20613v2)**

> **作者:** Ziju Shen; Naohao Huang; Fanyi Yang; Yutong Wang; Guoxiong Gao; Tianyi Xu; Jiedong Jiang; Wanyi He; Pu Yang; Mengzhou Sun; Haocheng Ju; Peihao Wu; Bryan Dai; Bin Dong
>
> **摘要:** Nowadays, formal theorem provers have made monumental progress on high-school and competition-level mathematics, but few of them generalize to more advanced mathematics. In this paper, we present REAL-Prover, a new open-source stepwise theorem prover for Lean 4 to push this boundary. This prover, based on our fine-tuned large language model (REAL-Prover-v1) and integrated with a retrieval system (Leansearch-PS), notably boosts performance on solving college-level mathematics problems. To train REAL-Prover-v1, we developed HERALD-AF, a data extraction pipeline that converts natural language math problems into formal statements, and a new open-source Lean 4 interactive environment (Jixia-interactive) to facilitate synthesis data collection. In our experiments, our prover using only supervised fine-tune achieves competitive results with a 23.7% success rate (Pass@64) on the ProofNet dataset-comparable to state-of-the-art (SOTA) models. To further evaluate our approach, we introduce FATE-M, a new benchmark focused on algebraic problems, where our prover achieves a SOTA success rate of 56.7% (Pass@64).
>
---
#### [replaced 025] Personalizing Student-Agent Interactions Using Log-Contextualized Retrieval Augmented Generation (RAG)
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17238v2](http://arxiv.org/pdf/2505.17238v2)**

> **作者:** Clayton Cohn; Surya Rayala; Caitlin Snyder; Joyce Fonteles; Shruti Jain; Naveeduddin Mohammed; Umesh Timalsina; Sarah K. Burriss; Ashwin T S; Namrata Srivastava; Menton Deweese; Angela Eeds; Gautam Biswas
>
> **备注:** To appear in the International Conference on Artificial Intelligence in Education (AIED25) Workshop on Epistemics and Decision-Making in AI-Supported Education
>
> **摘要:** Collaborative dialogue offers rich insights into students' learning and critical thinking, which is essential for personalizing pedagogical agent interactions in STEM+C settings. While large language models (LLMs) facilitate dynamic pedagogical interactions, hallucinations undermine confidence, trust, and instructional value. Retrieval-augmented generation (RAG) grounds LLM outputs in curated knowledge but requires a clear semantic link between user input and a knowledge base, which is often weak in student dialogue. We propose log-contextualized RAG (LC-RAG), which enhances RAG retrieval by using environment logs to contextualize collaborative discourse. Our findings show that LC-RAG improves retrieval over a discourse-only baseline and allows our collaborative peer agent, Copa, to deliver relevant, personalized guidance that supports students' critical thinking and epistemic decision-making in a collaborative computational modeling environment, C2STEM.
>
---
#### [replaced 026] Beyond Browsing: API-Based Web Agents
- **分类: cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2410.16464v3](http://arxiv.org/pdf/2410.16464v3)**

> **作者:** Yueqi Song; Frank Xu; Shuyan Zhou; Graham Neubig
>
> **备注:** 20 pages, 8 figures
>
> **摘要:** Web browsers are a portal to the internet, where much of human activity is undertaken. Thus, there has been significant research work in AI agents that interact with the internet through web browsing. However, there is also another interface designed specifically for machine interaction with online content: application programming interfaces (APIs). In this paper we ask -- what if we were to take tasks traditionally tackled by Browsing Agents, and give AI agents access to APIs? To do so, we propose two varieties of agents: (1) an API-calling agent that attempts to perform online tasks through APIs only, similar to traditional coding agents, and (2) a Hybrid Agent that can interact with online data through both web browsing and APIs. In experiments on WebArena, a widely-used and realistic benchmark for web navigation tasks, we find that API-Based Agents outperform web Browsing Agents. Hybrid Agents out-perform both others nearly uniformly across tasks, resulting in a more than 24.0% absolute improvement over web browsing alone, achieving a success rate of 38.9%, the SOTA performance among task-agnostic agents. These results strongly suggest that when APIs are available, they present an attractive alternative to relying on web browsing alone.
>
---
#### [replaced 027] TaskCraft: Automated Generation of Agentic Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10055v2](http://arxiv.org/pdf/2506.10055v2)**

> **作者:** Dingfeng Shi; Jingyi Cao; Qianben Chen; Weichen Sun; Weizhen Li; Hongxuan Lu; Fangchen Dong; Tianrui Qin; King Zhu; Minghao Liu; Jian Yang; Ge Zhang; Jiaheng Liu; Changwang Zhang; Jun Wang; Yuchen Eleanor Jiang; Wangchunshu Zhou
>
> **摘要:** Agentic tasks, which require multi-step problem solving with autonomy, tool use, and adaptive reasoning, are becoming increasingly central to the advancement of NLP and AI. However, existing instruction data lacks tool interaction, and current agentic benchmarks rely on costly human annotation, limiting their scalability. We introduce \textsc{TaskCraft}, an automated workflow for generating difficulty-scalable, multi-tool, and verifiable agentic tasks with execution trajectories. TaskCraft expands atomic tasks using depth-based and width-based extensions to create structurally and hierarchically complex challenges. Empirical results show that these tasks improve prompt optimization in the generation workflow and enhance supervised fine-tuning of agentic foundation models. We present a large-scale synthetic dataset of approximately 36,000 tasks with varying difficulty to support future research on agent tuning and evaluation.
>
---
#### [replaced 028] Geometric Signatures of Compositionality Across a Language Model's Lifetime
- **分类: cs.CL; cs.AI; cs.IT; cs.LG; math.IT**

- **链接: [http://arxiv.org/pdf/2410.01444v5](http://arxiv.org/pdf/2410.01444v5)**

> **作者:** Jin Hwa Lee; Thomas Jiralerspong; Lei Yu; Yoshua Bengio; Emily Cheng
>
> **备注:** Published at ACL 2025
>
> **摘要:** By virtue of linguistic compositionality, few syntactic rules and a finite lexicon can generate an unbounded number of sentences. That is, language, though seemingly high-dimensional, can be explained using relatively few degrees of freedom. An open question is whether contemporary language models (LMs) reflect the intrinsic simplicity of language that is enabled by compositionality. We take a geometric view of this problem by relating the degree of compositionality in a dataset to the intrinsic dimension (ID) of its representations under an LM, a measure of feature complexity. We find not only that the degree of dataset compositionality is reflected in representations' ID, but that the relationship between compositionality and geometric complexity arises due to learned linguistic features over training. Finally, our analyses reveal a striking contrast between nonlinear and linear dimensionality, showing they respectively encode semantic and superficial aspects of linguistic composition.
>
---
#### [replaced 029] Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11425v2](http://arxiv.org/pdf/2502.11425v2)**

> **作者:** Jongho Kim; Seung-won Hwang
>
> **备注:** ACL 2025 main (short)
>
> **摘要:** Despite the advanced capabilities of large language models (LLMs), their temporal reasoning ability remains underdeveloped. Prior works have highlighted this limitation, particularly in maintaining temporal consistency when understanding events. For example, models often confuse mutually exclusive temporal relations like ``before'' and ``after'' between events and make inconsistent predictions. In this work, we tackle the issue of temporal inconsistency in LLMs by proposing a novel counterfactual prompting approach. Our method generates counterfactual questions and enforces collective constraints, enhancing the model's consistency. We evaluate our method on multiple datasets, demonstrating significant improvements in event ordering for explicit and implicit events and temporal commonsense understanding by effectively addressing temporal inconsistencies.
>
---
#### [replaced 030] Hanfu-Bench: A Multimodal Benchmark on Cross-Temporal Cultural Understanding and Transcreation
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.01565v2](http://arxiv.org/pdf/2506.01565v2)**

> **作者:** Li Zhou; Lutong Yu; Dongchu Xie; Shaohuan Cheng; Wenyan Li; Haizhou Li
>
> **备注:** cultural analysis, cultural visual understanding, cultural image transcreation (update dataset license)
>
> **摘要:** Culture is a rich and dynamic domain that evolves across both geography and time. However, existing studies on cultural understanding with vision-language models (VLMs) primarily emphasize geographic diversity, often overlooking the critical temporal dimensions. To bridge this gap, we introduce Hanfu-Bench, a novel, expert-curated multimodal dataset. Hanfu, a traditional garment spanning ancient Chinese dynasties, serves as a representative cultural heritage that reflects the profound temporal aspects of Chinese culture while remaining highly popular in Chinese contemporary society. Hanfu-Bench comprises two core tasks: cultural visual understanding and cultural image transcreation.The former task examines temporal-cultural feature recognition based on single- or multi-image inputs through multiple-choice visual question answering, while the latter focuses on transforming traditional attire into modern designs through cultural element inheritance and modern context adaptation. Our evaluation shows that closed VLMs perform comparably to non-experts on visual cutural understanding but fall short by 10\% to human experts, while open VLMs lags further behind non-experts. For the transcreation task, multi-faceted human evaluation indicates that the best-performing model achieves a success rate of only 42\%. Our benchmark provides an essential testbed, revealing significant challenges in this new direction of temporal cultural understanding and creative adaptation.
>
---
#### [replaced 031] AgentCPM-GUI: Building Mobile-Use Agents with Reinforcement Fine-Tuning
- **分类: cs.AI; cs.CL; cs.CV; cs.HC; I.2.8; I.2.7; I.2.10; H.5.2**

- **链接: [http://arxiv.org/pdf/2506.01391v2](http://arxiv.org/pdf/2506.01391v2)**

> **作者:** Zhong Zhang; Yaxi Lu; Yikun Fu; Yupeng Huo; Shenzhi Yang; Yesai Wu; Han Si; Xin Cong; Haotian Chen; Yankai Lin; Jie Xie; Wei Zhou; Wang Xu; Yuanheng Zhang; Zhou Su; Zhongwu Zhai; Xiaoming Liu; Yudong Mei; Jianming Xu; Hongyan Tian; Chongyi Wang; Chi Chen; Yuan Yao; Zhiyuan Liu; Maosong Sun
>
> **备注:** Updated results in Table 2 and Table 3; The project is available at https://github.com/OpenBMB/AgentCPM-GUI
>
> **摘要:** The recent progress of large language model agents has opened new possibilities for automating tasks through graphical user interfaces (GUIs), especially in mobile environments where intelligent interaction can greatly enhance usability. However, practical deployment of such agents remains constrained by several key challenges. Existing training data is often noisy and lack semantic diversity, which hinders the learning of precise grounding and planning. Models trained purely by imitation tend to overfit to seen interface patterns and fail to generalize in unfamiliar scenarios. Moreover, most prior work focuses on English interfaces while overlooks the growing diversity of non-English applications such as those in the Chinese mobile ecosystem. In this work, we present AgentCPM-GUI, an 8B-parameter GUI agent built for robust and efficient on-device GUI interaction. Our training pipeline includes grounding-aware pre-training to enhance perception, supervised fine-tuning on high-quality Chinese and English trajectories to imitate human-like actions, and reinforcement fine-tuning with GRPO to improve reasoning capability. We also introduce a compact action space that reduces output length and supports low-latency execution on mobile devices. AgentCPM-GUI achieves state-of-the-art performance on five public benchmarks and a new Chinese GUI benchmark called CAGUI, reaching $96.9\%$ Type-Match and $91.3\%$ Exact-Match. To facilitate reproducibility and further research, we publicly release all code, model checkpoint, and evaluation data.
>
---
#### [replaced 032] LongSpec: Long-Context Lossless Speculative Decoding with Efficient Drafting and Verification
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.17421v2](http://arxiv.org/pdf/2502.17421v2)**

> **作者:** Penghui Yang; Cunxiao Du; Fengzhuo Zhang; Haonan Wang; Tianyu Pang; Chao Du; Bo An
>
> **摘要:** As Large Language Models (LLMs) can now process extremely long contexts, efficient inference over these extended inputs has become increasingly important, especially for emerging applications like LLM agents that highly depend on this capability. Speculative decoding (SD) offers a promising lossless acceleration technique compared to lossy alternatives such as quantization and model cascades. However, most state-of-the-art SD methods are trained on short texts (typically fewer than 4k tokens), making them unsuitable for long-context scenarios. Specifically, adapting these methods to long contexts presents three key challenges: (1) the excessive memory demands posed by draft models due to large Key-Value (KV) cache; (2) performance degradation resulting from the mismatch between short-context training and long-context inference; and (3) inefficiencies in tree attention mechanisms when managing long token sequences. This work introduces LongSpec, a framework that addresses these challenges through three core innovations: a memory-efficient draft model with a constant-sized KV cache; novel position indices that mitigate the training-inference mismatch; and an attention aggregation strategy that combines fast prefix computation with standard tree attention to enable efficient decoding. Experimental results confirm the effectiveness of LongSpec, achieving up to a 3.26x speedup over strong Flash Attention baselines across five long-context understanding datasets, as well as a 2.25x reduction in wall-clock time on the AIME24 long reasoning task with the QwQ model, demonstrating significant latency improvements for long-context applications. The code is available at https://github.com/sail-sg/LongSpec.
>
---
#### [replaced 033] Batch-Max: Higher LLM Throughput using Larger Batch Sizes and KV Cache Compression
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.05693v2](http://arxiv.org/pdf/2412.05693v2)**

> **作者:** Michael R. Metel; Boxing Chen; Mehdi Rezagholizadeh
>
> **摘要:** Several works have developed eviction policies to remove key-value (KV) pairs from the KV cache for more efficient inference. The focus has been on compressing the KV cache after the input prompt has been processed for faster token generation. In settings with limited GPU memory, and when the input context is longer than the generation length, we show that by also compressing the KV cache during the input processing phase, larger batch sizes can be used resulting in significantly higher throughput while still maintaining the original model's accuracy.
>
---
#### [replaced 034] Effect of Selection Format on LLM Performance
- **分类: cs.CL; cs.AI; cs.CE; cs.ET; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.06926v2](http://arxiv.org/pdf/2503.06926v2)**

> **作者:** Yuchen Han; Yucheng Wu; Jeffrey Willard
>
> **摘要:** This paper investigates a critical aspect of large language model (LLM) performance: the optimal formatting of classification task options in prompts. Through an extensive experimental study, we compared two selection formats -- bullet points and plain English -- to determine their impact on model performance. Our findings suggest that presenting options via bullet points generally yields better results, although there are some exceptions. Furthermore, our research highlights the need for continued exploration of option formatting to drive further improvements in model performance.
>
---
#### [replaced 035] MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.00555v2](http://arxiv.org/pdf/2506.00555v2)**

> **作者:** Peng Xia; Jinglu Wang; Yibo Peng; Kaide Zeng; Xian Wu; Xiangru Tang; Hongtu Zhu; Yun Li; Shujie Liu; Yan Lu; Huaxiu Yao
>
> **摘要:** Medical Large Vision-Language Models (Med-LVLMs) have shown strong potential in multimodal diagnostic tasks. However, existing single-agent models struggle to generalize across diverse medical specialties, limiting their performance. Recent efforts introduce multi-agent collaboration frameworks inspired by clinical workflows, where general practitioners (GPs) and specialists interact in a fixed sequence. Despite improvements, these static pipelines lack flexibility and adaptability in reasoning. To address this, we propose MMedAgent-RL, a reinforcement learning (RL)-based multi-agent framework that enables dynamic, optimized collaboration among medical agents. Specifically, we train two GP agents based on Qwen2.5-VL via RL: the triage doctor learns to assign patients to appropriate specialties, while the attending physician integrates the judgments from multi-specialists and its own knowledge to make final decisions. To address the inconsistency in specialist outputs, we introduce a curriculum learning (CL)-guided RL strategy that progressively teaches the attending physician to balance between imitating specialists and correcting their mistakes. Experiments on five medical VQA benchmarks demonstrate that MMedAgent-RL not only outperforms both open-source and proprietary Med-LVLMs, but also exhibits human-like reasoning patterns. Notably, it achieves an average performance gain of 20.7% over supervised fine-tuning baselines.
>
---
#### [replaced 036] ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.12376v2](http://arxiv.org/pdf/2506.12376v2)**

> **作者:** Zhaochen Hong; Haofei Yu; Jiaxuan You
>
> **备注:** Accepted at ACL 2025 Main Conference
>
> **摘要:** Evaluating consistency in large language models (LLMs) is crucial for ensuring reliability, particularly in complex, multi-step interactions between humans and LLMs. Traditional self-consistency methods often miss subtle semantic changes in natural language and functional shifts in code or equations, which can accumulate over multiple transformations. To address this, we propose ConsistencyChecker, a tree-based evaluation framework designed to measure consistency through sequences of reversible transformations, including machine translation tasks and AI-assisted programming tasks. In our framework, nodes represent distinct text states, while edges correspond to pairs of inverse operations. Dynamic and LLM-generated benchmarks ensure a fair assessment of the model's generalization ability and eliminate benchmark leakage. Consistency is quantified based on similarity across different depths of the transformation tree. Experiments on eight models from various families and sizes show that ConsistencyChecker can distinguish the performance of different models. Notably, our consistency scores-computed entirely without using WMT paired data-correlate strongly (r > 0.7) with WMT 2024 auto-ranking, demonstrating the validity of our benchmark-free approach. Our implementation is available at: https://github.com/ulab-uiuc/consistencychecker.
>
---
#### [replaced 037] Agent Laboratory: Using LLM Agents as Research Assistants
- **分类: cs.HC; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.04227v2](http://arxiv.org/pdf/2501.04227v2)**

> **作者:** Samuel Schmidgall; Yusheng Su; Ze Wang; Ximeng Sun; Jialian Wu; Xiaodong Yu; Jiang Liu; Michael Moor; Zicheng Liu; Emad Barsoum
>
> **摘要:** Historically, scientific discovery has been a lengthy and costly process, demanding substantial time and resources from initial conception to final results. To accelerate scientific discovery, reduce research costs, and improve research quality, we introduce Agent Laboratory, an autonomous LLM-based framework capable of completing the entire research process. This framework accepts a human-provided research idea and progresses through three stages--literature review, experimentation, and report writing to produce comprehensive research outputs, including a code repository and a research report, while enabling users to provide feedback and guidance at each stage. We deploy Agent Laboratory with various state-of-the-art LLMs and invite multiple researchers to assess its quality by participating in a survey, providing human feedback to guide the research process, and then evaluate the final paper. We found that: (1) Agent Laboratory driven by o1-preview generates the best research outcomes; (2) The generated machine learning code is able to achieve state-of-the-art performance compared to existing methods; (3) Human involvement, providing feedback at each stage, significantly improves the overall quality of research; (4) Agent Laboratory significantly reduces research expenses, achieving an 84% decrease compared to previous autonomous research methods. We hope Agent Laboratory enables researchers to allocate more effort toward creative ideation rather than low-level coding and writing, ultimately accelerating scientific discovery.
>
---
#### [replaced 038] Compression of enumerations and gain
- **分类: cs.CL; cs.IT; math.IT; math.LO**

- **链接: [http://arxiv.org/pdf/2304.03030v2](http://arxiv.org/pdf/2304.03030v2)**

> **作者:** George Barmpalias; Xiaoyan Zhang; Bohua Zhan
>
> **摘要:** We study the compressibility of enumerations in the context of Kolmogorov complexity, focusing on strong and weak forms of compression and their gain: the amount of auxiliary information embedded in the compressed enumeration. The existence of strong compression and weak gainless compression is shown for any computably enumerable (c.e.) set. The density problem of c.e. sets with respect to their prefix complexity is reduced to the question of whether every c.e. set is well-compressible, which we study via enumeration games.
>
---
#### [replaced 039] OWLViz: An Open-World Benchmark for Visual Question Answering
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.07631v2](http://arxiv.org/pdf/2503.07631v2)**

> **作者:** Thuy Nguyen; Dang Nguyen; Hoang Nguyen; Thuan Luong; Long Hoang Dang; Viet Dac Lai
>
> **备注:** ICML 2025 Workshop on Multi-Agent Systems in the Era of Foundation Models: Opportunities, Challenges, and Futures. (8 pages + appendix)
>
> **摘要:** We present a challenging benchmark for the Open WorLd VISual question answering (OWLViz) task. OWLViz presents concise, unambiguous queries that require integrating multiple capabilities, including visual understanding, web exploration, and specialized tool usage. While humans achieve 69.2% accuracy on these intuitive tasks, even state-of-the-art VLMs struggle, with the best model, Gemini 2.0, achieving only 26.6% accuracy. Current agentic VLMs, which rely on limited vision and vision-language models as tools, perform even worse. This performance gap reveals significant limitations in multimodal systems' ability to select appropriate tools and execute complex reasoning sequences, establishing new directions for advancing practical AI research.
>
---
#### [replaced 040] PredictaBoard: Benchmarking LLM Score Predictability
- **分类: cs.CL; cs.AI; stat.ML**

- **链接: [http://arxiv.org/pdf/2502.14445v2](http://arxiv.org/pdf/2502.14445v2)**

> **作者:** Lorenzo Pacchiardi; Konstantinos Voudouris; Ben Slater; Fernando Martínez-Plumed; José Hernández-Orallo; Lexin Zhou; Wout Schellaert
>
> **备注:** Accepted at ACL Findings 2025
>
> **摘要:** Despite possessing impressive skills, Large Language Models (LLMs) often fail unpredictably, demonstrating inconsistent success in even basic common sense reasoning tasks. This unpredictability poses a significant challenge to ensuring their safe deployment, as identifying and operating within a reliable "safe zone" is essential for mitigating risks. To address this, we present PredictaBoard, a novel collaborative benchmarking framework designed to evaluate the ability of score predictors (referred to as assessors) to anticipate LLM errors on specific task instances (i.e., prompts) from existing datasets. PredictaBoard evaluates pairs of LLMs and assessors by considering the rejection rate at different tolerance errors. As such, PredictaBoard stimulates research into developing better assessors and making LLMs more predictable, not only with a higher average performance. We conduct illustrative experiments using baseline assessors and state-of-the-art LLMs. PredictaBoard highlights the critical need to evaluate predictability alongside performance, paving the way for safer AI systems where errors are not only minimised but also anticipated and effectively mitigated. Code for our benchmark can be found at https://github.com/Kinds-of-Intelligence-CFI/PredictaBoard
>
---
#### [replaced 041] Reward Shaping to Mitigate Reward Hacking in RLHF
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18770v3](http://arxiv.org/pdf/2502.18770v3)**

> **作者:** Jiayi Fu; Xuandong Zhao; Chengyuan Yao; Heng Wang; Qi Han; Yanghua Xiao
>
> **备注:** 24 pages
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) is essential for aligning large language models (LLMs) with human values. However, RLHF is susceptible to \emph{reward hacking}, where the agent exploits flaws in the reward function rather than learning the intended behavior, thus degrading alignment. Although reward shaping helps stabilize RLHF and partially mitigate reward hacking, a systematic investigation into shaping techniques and their underlying principles remains lacking. To bridge this gap, we present a comprehensive study of the prevalent reward shaping methods. Our analysis suggests two key design principles: (1) the RL reward should be bounded, and (2) the RL reward benefits from rapid initial growth followed by gradual convergence. Guided by these insights, we propose Preference As Reward (PAR), a novel approach that leverages the latent preferences embedded within the reward model as the signal for reinforcement learning. We evaluated PAR on two base models, Gemma2-2B, and Llama3-8B, using two datasets, Ultrafeedback-Binarized and HH-RLHF. Experimental results demonstrate PAR's superior performance over other reward shaping methods. On the AlpacaEval 2.0 benchmark, PAR achieves a win rate of at least 5 percentage points higher than competing approaches. Furthermore, PAR exhibits remarkable data efficiency, requiring only a single reference reward for optimal performance, and maintains robustness against reward hacking even after two full epochs of training. The code is available at https://github.com/PorUna-byte/PAR, and the Work done during the internship at StepFun by Jiayi Fu.
>
---
#### [replaced 042] FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09081v2](http://arxiv.org/pdf/2506.09081v2)**

> **作者:** Zheqi He; Yesheng Liu; Jing-shu Zheng; Xuejing Li; Jin-Ge Yao; Bowen Qin; Richeng Xuan; Xi Yang
>
> **摘要:** We present FlagEvalMM, an open-source evaluation framework designed to comprehensively assess multimodal models across a diverse range of vision-language understanding and generation tasks, such as visual question answering, text-to-image/video generation, and image-text retrieval. We decouple model inference from evaluation through an independent evaluation service, thus enabling flexible resource allocation and seamless integration of new tasks and models. Moreover, FlagEvalMM utilizes advanced inference acceleration tools (e.g., vLLM, SGLang) and asynchronous data loading to significantly enhance evaluation efficiency. Extensive experiments show that FlagEvalMM offers accurate and efficient insights into model strengths and limitations, making it a valuable tool for advancing multimodal research. The framework is publicly accessible athttps://github.com/flageval-baai/FlagEvalMM.
>
---
#### [replaced 043] Position: Editing Large Language Models Poses Serious Safety Risks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.02958v3](http://arxiv.org/pdf/2502.02958v3)**

> **作者:** Paul Youssef; Zhixue Zhao; Daniel Braun; Jörg Schlötterer; Christin Seifert
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Large Language Models (LLMs) contain large amounts of facts about the world. These facts can become outdated over time, which has led to the development of knowledge editing methods (KEs) that can change specific facts in LLMs with limited side effects. This position paper argues that editing LLMs poses serious safety risks that have been largely overlooked. First, we note the fact that KEs are widely available, computationally inexpensive, highly performant, and stealthy makes them an attractive tool for malicious actors. Second, we discuss malicious use cases of KEs, showing how KEs can be easily adapted for a variety of malicious purposes. Third, we highlight vulnerabilities in the AI ecosystem that allow unrestricted uploading and downloading of updated models without verification. Fourth, we argue that a lack of social and institutional awareness exacerbates this risk, and discuss the implications for different stakeholders. We call on the community to (i) research tamper-resistant models and countermeasures against malicious model editing, and (ii) actively engage in securing the AI ecosystem.
>
---
#### [replaced 044] Ensemble Watermarks for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.19563v2](http://arxiv.org/pdf/2411.19563v2)**

> **作者:** Georg Niess; Roman Kern
>
> **备注:** Accepted to ACL 2025 main conference. This article extends our earlier work arXiv:2405.08400 by introducing an ensemble of stylometric watermarking features and alternative experimental analysis. Code and data are available at http://github.com/CommodoreEU/ensemble-watermark
>
> **摘要:** As large language models (LLMs) reach human-like fluency, reliably distinguishing AI-generated text from human authorship becomes increasingly difficult. While watermarks already exist for LLMs, they often lack flexibility and struggle with attacks such as paraphrasing. To address these issues, we propose a multi-feature method for generating watermarks that combines multiple distinct watermark features into an ensemble watermark. Concretely, we combine acrostica and sensorimotor norms with the established red-green watermark to achieve a 98% detection rate. After a paraphrasing attack, the performance remains high with 95% detection rate. In comparison, the red-green feature alone as a baseline achieves a detection rate of 49% after paraphrasing. The evaluation of all feature combinations reveals that the ensemble of all three consistently has the highest detection rate across several LLMs and watermark strength settings. Due to the flexibility of combining features in the ensemble, various requirements and trade-offs can be addressed. Additionally, the same detection function can be used without adaptations for all ensemble configurations. This method is particularly of interest to facilitate accountability and prevent societal harm.
>
---
#### [replaced 045] AI-Facilitated Analysis of Abstracts and Conclusions: Flagging Unsubstantiated Claims and Ambiguous Pronouns
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.13172v2](http://arxiv.org/pdf/2506.13172v2)**

> **作者:** Evgeny Markhasin
>
> **备注:** 13 pages
>
> **摘要:** We present and evaluate a suite of proof-of-concept (PoC), structured workflow prompts designed to elicit human-like hierarchical reasoning while guiding Large Language Models (LLMs) in the high-level semantic and linguistic analysis of scholarly manuscripts. The prompts target two non-trivial analytical tasks within academic summaries (abstracts and conclusions): identifying unsubstantiated claims (informational integrity) and flagging semantically confusing ambiguous pronoun references (linguistic clarity). We conducted a systematic, multi-run evaluation on two frontier models (Gemini Pro 2.5 Pro and ChatGPT Plus o3) under varied context conditions. Our results for the informational integrity task reveal a significant divergence in model performance: while both models successfully identified an unsubstantiated head of a noun phrase (95% success), ChatGPT consistently failed (0% success) to identify an unsubstantiated adjectival modifier that Gemini correctly flagged (95% success), raising a question regarding the potential influence of the target's syntactic role. For the linguistic analysis task, both models performed well (80-90% success) with full manuscript context. Surprisingly, in a summary-only setting, Gemini's performance was substantially degraded, while ChatGPT achieved a perfect (100%) success rate. Our findings suggest that while structured prompting is a viable methodology for complex textual analysis, prompt performance may be highly dependent on the interplay between the model, task type, and context, highlighting the need for rigorous, model-specific testing.
>
---
#### [replaced 046] Enhancing Clinical Models with Pseudo Data for De-identification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.12674v2](http://arxiv.org/pdf/2506.12674v2)**

> **作者:** Paul Landes; Aaron J Chaise; Tarak Nath Nandi; Ravi K Madduri
>
> **摘要:** Many models are pretrained on redacted text for privacy reasons. Clinical foundation models are often trained on de-identified text, which uses special syntax (masked) text in place of protected health information. Even though these models have increased in popularity, there has been little effort in understanding the effects of training them on redacted text. In this work, we pretrain several encoder-only models on a dataset that contains redacted text and a version with replaced realistic pseudo text. We then fine-tuned models for the protected health information de-identification task and show how our methods significantly outperform previous baselines. The contributions of this work include: a) our novel, and yet surprising findings with training recommendations, b) redacted text replacements used to produce the pseudo dataset, c) pretrained embeddings and fine-tuned task specific models, and d) freely available pseudo training dataset generation and model source code used in our experiments.
>
---
#### [replaced 047] LongCodeBench: Evaluating Coding LLMs at 1M Context Windows
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.07897v2](http://arxiv.org/pdf/2505.07897v2)**

> **作者:** Stefano Rando; Luca Romani; Alessio Sampieri; Luca Franco; John Yang; Yuta Kyuragi; Fabio Galasso; Tatsunori Hashimoto
>
> **摘要:** Context lengths for models have grown rapidly, from thousands to millions of tokens in just a few years. The extreme context sizes of modern long-context models have made it difficult to construct realistic long-context benchmarks -- not only due to the cost of collecting million-context tasks but also in identifying realistic scenarios that require significant contexts. We identify code comprehension and repair as a natural testbed and challenge task for long-context models and introduce LongCodeBench (LCB), a benchmark to test LLM coding abilities in long-context scenarios. Our benchmark tests both the comprehension and repair capabilities of LCLMs in realistic and important settings by drawing from real-world GitHub issues and constructing QA (LongCodeQA) and bug fixing (LongSWE-Bench) tasks. We carefully stratify the complexity of our benchmark, enabling us to evaluate models across different scales -- ranging from Qwen2.5 14B Instruct to Google's flagship Gemini model. We find that long-context remains a weakness for all models, with performance drops such as from 29% to 3% for Claude 3.5 Sonnet, or from 70.2% to 40% for Qwen2.5.
>
---
#### [replaced 048] Assessing the Reasoning Capabilities of LLMs in the context of Evidence-based Claim Verification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2402.10735v4](http://arxiv.org/pdf/2402.10735v4)**

> **作者:** John Dougrez-Lewis; Mahmud Elahi Akhter; Federico Ruggeri; Sebastian Löbbers; Yulan He; Maria Liakata
>
> **备注:** First two authors contributed equally to this work. 25 pages, 3 figure
>
> **摘要:** Although LLMs have shown great performance on Mathematics and Coding related reasoning tasks, the reasoning capabilities of LLMs regarding other forms of reasoning are still an open problem. Here, we examine the issue of reasoning from the perspective of claim verification. We propose a framework designed to break down any claim paired with evidence into atomic reasoning types that are necessary for verification. We use this framework to create RECV, the first claim verification benchmark, incorporating real-world claims, to assess the deductive and abductive reasoning capabilities of LLMs. The benchmark comprises of three datasets, covering reasoning problems of increasing complexity. We evaluate three state-of-the-art proprietary LLMs under multiple prompt settings. Our results show that while LLMs can address deductive reasoning problems, they consistently fail in cases of abductive reasoning. Moreover, we observe that enhancing LLMs with rationale generation is not always beneficial. Nonetheless, we find that generated rationales are semantically similar to those provided by humans, especially in deductive reasoning cases.
>
---
#### [replaced 049] Reparameterized LLM Training via Orthogonal Equivalence Transformation
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08001v3](http://arxiv.org/pdf/2506.08001v3)**

> **作者:** Zeju Qiu; Simon Buchholz; Tim Z. Xiao; Maximilian Dax; Bernhard Schölkopf; Weiyang Liu
>
> **备注:** Technical report v3 (38 pages, 26 figures, project page: https://spherelab.ai/poet/, v3: added singular spectrum and energy analyses in Section 4)
>
> **摘要:** While large language models (LLMs) are driving the rapid advancement of artificial intelligence, effectively and reliably training these large models remains one of the field's most significant challenges. To address this challenge, we propose POET, a novel reParameterized training algorithm that uses Orthogonal Equivalence Transformation to optimize neurons. Specifically, POET reparameterizes each neuron with two learnable orthogonal matrices and a fixed random weight matrix. Because of its provable preservation of spectral properties of weight matrices, POET can stably optimize the objective function with improved generalization. We further develop efficient approximations that make POET flexible and scalable for training large-scale neural networks. Extensive experiments validate the effectiveness and scalability of POET in training LLMs.
>
---
#### [replaced 050] ROSAQ: Rotation-based Saliency-Aware Weight Quantization for Efficiently Compressing Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.13472v2](http://arxiv.org/pdf/2506.13472v2)**

> **作者:** Junho Yoon; Geom Lee; Donghyeon Jeon; Inho Kang; Seung-Hoon Na
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** Quantization has been widely studied as an effective technique for reducing the memory requirement of large language models (LLMs), potentially improving the latency time as well. Utilizing the characteristic of rotational invariance of transformer, we propose the rotation-based saliency-aware weight quantization (ROSAQ), which identifies salient channels in the projection feature space, not in the original feature space, where the projected "principal" dimensions are naturally considered as "salient" features. The proposed ROSAQ consists of 1) PCA-based projection, which first performs principal component analysis (PCA) on a calibration set and transforms via the PCA projection, 2) Salient channel dentification, which selects dimensions corresponding to the K-largest eigenvalues as salient channels, and 3) Saliency-aware quantization with mixed-precision, which uses FP16 for salient dimensions and INT3/4 for other dimensions. Experiment results show that ROSAQ shows improvements over the baseline saliency-aware quantization on the original feature space and other existing quantization methods. With kernel fusion, ROSAQ presents about 2.3x speed up over FP16 implementation in generating 256 tokens with a batch size of 64.
>
---
#### [replaced 051] Convert Language Model into a Value-based Strategic Planner
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.06987v4](http://arxiv.org/pdf/2505.06987v4)**

> **作者:** Xiaoyu Wang; Yue Zhao; Qingqing Gu; Zhonglin Jiang; Xiaokai Chen; Yong Chen; Luo Ji
>
> **备注:** 13 pages, 6 figures, Accepted by ACL 2025 Industry Track
>
> **摘要:** Emotional support conversation (ESC) aims to alleviate the emotional distress of individuals through effective conversations. Although large language models (LLMs) have obtained remarkable progress on ESC, most of these studies might not define the diagram from the state model perspective, therefore providing a suboptimal solution for long-term satisfaction. To address such an issue, we leverage the Q-learning on LLMs, and propose a framework called straQ*. Our framework allows a plug-and-play LLM to bootstrap the planning during ESC, determine the optimal strategy based on long-term returns, and finally guide the LLM to response. Substantial experiments on ESC datasets suggest that straQ* outperforms many baselines, including direct inference, self-refine, chain of thought, finetuning, and finite state machines.
>
---
#### [replaced 052] ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.06745v2](http://arxiv.org/pdf/2412.06745v2)**

> **作者:** Adhiraj Ghosh; Sebastian Dziadzio; Ameya Prabhu; Vishaal Udandarao; Samuel Albanie; Matthias Bethge
>
> **摘要:** Traditional fixed test sets fall short in evaluating open-ended capabilities of foundation models. To address this, we propose ONEBench(OpeN-Ended Benchmarking), a new testing paradigm that consolidates individual evaluation datasets into a unified, ever-expanding sample pool. ONEBench allows users to generate custom, open-ended evaluation benchmarks from this pool, corresponding to specific capabilities of interest. By aggregating samples across test sets, ONEBench enables the assessment of diverse capabilities beyond those covered by the original test sets, while mitigating overfitting and dataset bias. Most importantly, it frames model evaluation as a collective process of selecting and aggregating sample-level tests. The shift from task-specific benchmarks to ONEBench introduces two challenges: (1)heterogeneity and (2)incompleteness. Heterogeneity refers to the aggregation over diverse metrics, while incompleteness describes comparing models evaluated on different data subsets. To address these challenges, we explore algorithms to aggregate sparse measurements into reliable model scores. Our aggregation algorithm ensures identifiability(asymptotically recovering ground-truth scores) and rapid convergence, enabling accurate model ranking with less data. On homogenous datasets, we show our aggregation algorithm provides rankings that highly correlate with those produced by average scores. We also demonstrate robustness to ~95% of measurements missing, reducing evaluation cost by up to 20x with little-to-no change in model rankings. We introduce ONEBench-LLM for language models and ONEBench-LMM for vision-language models, unifying evaluations across these domains. Overall, we present a technique for open-ended evaluation, which can aggregate over incomplete, heterogeneous sample-level measurements to continually grow a benchmark alongside the rapidly developing foundation models.
>
---
#### [replaced 053] EEG2TEXT-CN: An Exploratory Study of Open-Vocabulary Chinese Text-EEG Alignment via Large Language Model and Contrastive Learning on ChineseEEG
- **分类: cs.CL; cs.AI; cs.LG; cs.MM; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2506.00854v2](http://arxiv.org/pdf/2506.00854v2)**

> **作者:** Jacky Tai-Yu Lu; Jung Chiang; Chi-Sheng Chen; Anna Nai-Yun Tung; Hsiang Wei Hu; Yuan Chiao Cheng
>
> **摘要:** We propose EEG2TEXT-CN, which, to the best of our knowledge, represents one of the earliest open-vocabulary EEG-to-text generation frameworks tailored for Chinese. Built on a biologically grounded EEG encoder (NICE-EEG) and a compact pretrained language model (MiniLM), our architecture aligns multichannel brain signals with natural language representations via masked pretraining and contrastive learning. Using a subset of the ChineseEEG dataset, where each sentence contains approximately ten Chinese characters aligned with 128-channel EEG recorded at 256 Hz, we segment EEG into per-character embeddings and predict full sentences in a zero-shot setting. The decoder is trained with teacher forcing and padding masks to accommodate variable-length sequences. Evaluation on over 1,500 training-validation sentences and 300 held-out test samples shows promising lexical alignment, with a best BLEU-1 score of 6.38\%. While syntactic fluency remains a challenge, our findings demonstrate the feasibility of non-phonetic, cross-modal language decoding from EEG. This work opens a new direction in multilingual brain-to-text research and lays the foundation for future cognitive-language interfaces in Chinese.
>
---
#### [replaced 054] Language and Planning in Robotic Navigation: A Multilingual Evaluation of State-of-the-Art Models
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.05478v2](http://arxiv.org/pdf/2501.05478v2)**

> **作者:** Malak Mansour; Ahmed Aly; Bahey Tharwat; Sarim Hashmi; Dong An; Ian Reid
>
> **备注:** This work has been accepted for presentation at LM4Plan@AAAI'25. For more details, please check: https://llmforplanning.github.io/
>
> **摘要:** Large Language Models (LLMs) such as GPT-4, trained on huge amount of datasets spanning multiple domains, exhibit significant reasoning, understanding, and planning capabilities across various tasks. This study presents the first-ever work in Arabic language integration within the Vision-and-Language Navigation (VLN) domain in robotics, an area that has been notably underexplored in existing research. We perform a comprehensive evaluation of state-of-the-art multi-lingual Small Language Models (SLMs), including GPT-4o mini, Llama 3 8B, and Phi-3 medium 14B, alongside the Arabic-centric LLM, Jais. Our approach utilizes the NavGPT framework, a pure LLM-based instruction-following navigation agent, to assess the impact of language on navigation reasoning through zero-shot sequential action prediction using the R2R dataset. Through comprehensive experiments, we demonstrate that our framework is capable of high-level planning for navigation tasks when provided with instructions in both English and Arabic. However, certain models struggled with reasoning and planning in the Arabic language due to inherent limitations in their capabilities, sub-optimal performance, and parsing issues. These findings highlight the importance of enhancing planning and reasoning capabilities in language models for effective navigation, emphasizing this as a key area for further development while also unlocking the potential of Arabic-language models for impactful real-world applications.
>
---
#### [replaced 055] Prefix-Tuning+: Modernizing Prefix-Tuning by Decoupling the Prefix from Attention
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.13674v2](http://arxiv.org/pdf/2506.13674v2)**

> **作者:** Haonan Wang; Brian Chen; Siquan Li; Xinhe Liang; Hwee Kuan Lee; Kenji Kawaguchi; Tianyang Hu
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT) methods have become crucial for rapidly adapting large language models (LLMs) to downstream tasks. Prefix-Tuning, an early and effective PEFT technique, demonstrated the ability to achieve performance comparable to full fine-tuning with significantly reduced computational and memory overhead. However, despite its earlier success, its effectiveness in training modern state-of-the-art LLMs has been very limited. In this work, we demonstrate empirically that Prefix-Tuning underperforms on LLMs because of an inherent tradeoff between input and prefix significance within the attention head. This motivates us to introduce Prefix-Tuning+, a novel architecture that generalizes the principles of Prefix-Tuning while addressing its shortcomings by shifting the prefix module out of the attention head itself. We further provide an overview of our construction process to guide future users when constructing their own context-based methods. Our experiments show that, across a diverse set of benchmarks, Prefix-Tuning+ consistently outperforms existing Prefix-Tuning methods. Notably, it achieves performance on par with the widely adopted LoRA method on several general benchmarks, highlighting the potential modern extension of Prefix-Tuning approaches. Our findings suggest that by overcoming its inherent limitations, Prefix-Tuning can remain a competitive and relevant research direction in the landscape of parameter-efficient LLM adaptation.
>
---
#### [replaced 056] SAE-V: Interpreting Multimodal Models for Enhanced Alignment
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17514v2](http://arxiv.org/pdf/2502.17514v2)**

> **作者:** Hantao Lou; Changye Li; Jiaming Ji; Yaodong Yang
>
> **备注:** 17 pages, 13 figures
>
> **摘要:** With the integration of image modality, the semantic space of multimodal large language models (MLLMs) is more complex than text-only models, making their interpretability more challenging and their alignment less stable, particularly susceptible to low-quality data, which can lead to inconsistencies between modalities, hallucinations, and biased outputs. As a result, developing interpretability methods for MLLMs is crucial for improving alignment quality and efficiency. In text-only LLMs, Sparse Autoencoders (SAEs) have gained attention for their ability to interpret latent representations. However, extending SAEs to multimodal settings presents new challenges due to modality fusion and the difficulty of isolating cross-modal representations. To address these challenges, we introduce SAE-V, a mechanistic interpretability framework that extends the SAE paradigm to MLLMs. By identifying and analyzing interpretable features along with their corresponding data, SAE-V enables fine-grained interpretation of both model behavior and data quality, facilitating a deeper understanding of cross-modal interactions and alignment dynamics. Moreover, by utilizing cross-modal feature weighting, SAE-V provides an intrinsic data filtering mechanism to enhance model alignment without requiring additional models. Specifically, when applied to the alignment process of MLLMs, SAE-V-based data filtering methods could achieve more than 110% performance with less than 50% data. Our results highlight SAE-V's ability to enhance interpretability and alignment in MLLMs, providing insights into their internal mechanisms.
>
---
#### [replaced 057] IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12442v3](http://arxiv.org/pdf/2505.12442v3)**

> **作者:** Liwen Wang; Wenxuan Wang; Shuai Wang; Zongjie Li; Zhenlan Ji; Zongyi Lyu; Daoyuan Wu; Shing-Chi Cheung
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.
>
---
#### [replaced 058] Roboflow100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.20612v2](http://arxiv.org/pdf/2505.20612v2)**

> **作者:** Peter Robicheaux; Matvei Popov; Anish Madan; Isaac Robinson; Joseph Nelson; Deva Ramanan; Neehar Peri
>
> **备注:** The first two authors contributed equally. Project Page: https://rf100-vl.org/
>
> **摘要:** Vision-language models (VLMs) trained on internet-scale data achieve remarkable zero-shot detection performance on common objects like car, truck, and pedestrian. However, state-of-the-art models still struggle to generalize to out-of-distribution classes, tasks and imaging modalities not typically found in their pre-training. Rather than simply re-training VLMs on more visual data, we argue that one should align VLMs to new concepts with annotation instructions containing a few visual examples and rich textual descriptions. To this end, we introduce Roboflow100-VL, a large-scale collection of 100 multi-modal object detection datasets with diverse concepts not commonly found in VLM pre-training. We evaluate state-of-the-art models on our benchmark in zero-shot, few-shot, semi-supervised, and fully-supervised settings, allowing for comparison across data regimes. Notably, we find that VLMs like GroundingDINO and Qwen2.5-VL achieve less than 2% zero-shot accuracy on challenging medical imaging datasets within Roboflow100-VL, demonstrating the need for few-shot concept alignment. Lastly, we discuss our recent CVPR 2025 Foundational FSOD competition and share insights from the community. Notably, the winning team significantly outperforms our baseline by 16.8 mAP! Our code and dataset are available at https://github.com/roboflow/rf100-vl/ and https://universe.roboflow.com/rf100-vl/
>
---
#### [replaced 059] SeqPE: Transformer with Sequential Position Encoding
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13277v2](http://arxiv.org/pdf/2506.13277v2)**

> **作者:** Huayang Li; Yahui Liu; Hongyu Sun; Deng Cai; Leyang Cui; Wei Bi; Peilin Zhao; Taro Watanabe
>
> **摘要:** Since self-attention layers in Transformers are permutation invariant by design, positional encodings must be explicitly incorporated to enable spatial understanding. However, fixed-size lookup tables used in traditional learnable position embeddings (PEs) limit extrapolation capabilities beyond pre-trained sequence lengths. Expert-designed methods such as ALiBi and RoPE, mitigate this limitation but demand extensive modifications for adapting to new modalities, underscoring fundamental challenges in adaptability and scalability. In this work, we present SeqPE, a unified and fully learnable position encoding framework that represents each $n$-dimensional position index as a symbolic sequence and employs a lightweight sequential position encoder to learn their embeddings in an end-to-end manner. To regularize SeqPE's embedding space, we introduce two complementary objectives: a contrastive objective that aligns embedding distances with a predefined position-distance function, and a knowledge distillation loss that anchors out-of-distribution position embeddings to in-distribution teacher representations, further enhancing extrapolation performance. Experiments across language modeling, long-context question answering, and 2D image classification demonstrate that SeqPE not only surpasses strong baselines in perplexity, exact match (EM), and accuracy--particularly under context length extrapolation--but also enables seamless generalization to multi-dimensional inputs without requiring manual architectural redesign. We release our code, data, and checkpoints at https://github.com/ghrua/seqpe.
>
---
#### [replaced 060] BESSTIE: A Benchmark for Sentiment and Sarcasm Classification for Varieties of English
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.04726v3](http://arxiv.org/pdf/2412.04726v3)**

> **作者:** Dipankar Srirag; Aditya Joshi; Jordan Painter; Diptesh Kanojia
>
> **备注:** Findings of ACL: ACL 2025
>
> **摘要:** Despite large language models (LLMs) being known to exhibit bias against non-standard language varieties, there are no known labelled datasets for sentiment analysis of English. To address this gap, we introduce BESSTIE, a benchmark for sentiment and sarcasm classification for three varieties of English: Australian (en-AU), Indian (en-IN), and British (en-UK). We collect datasets for these language varieties using two methods: location-based for Google Places reviews, and topic-based filtering for Reddit comments. To assess whether the dataset accurately represents these varieties, we conduct two validation steps: (a) manual annotation of language varieties and (b) automatic language variety prediction. Native speakers of the language varieties manually annotate the datasets with sentiment and sarcasm labels. We perform an additional annotation exercise to validate the reliance of the annotated labels. Subsequently, we fine-tune nine LLMs (representing a range of encoder/decoder and mono/multilingual models) on these datasets, and evaluate their performance on the two tasks. Our results show that the models consistently perform better on inner-circle varieties (i.e., en-AU and en-UK), in comparison with en-IN, particularly for sarcasm classification. We also report challenges in cross-variety generalisation, highlighting the need for language variety-specific datasets such as ours. BESSTIE promises to be a useful evaluative benchmark for future research in equitable LLMs, specifically in terms of language varieties. The BESSTIE dataset is publicly available at: https://huggingface.co/ datasets/unswnlporg/BESSTIE.
>
---
#### [replaced 061] Chain-of-Thought Reasoning In The Wild Is Not Always Faithful
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.08679v4](http://arxiv.org/pdf/2503.08679v4)**

> **作者:** Iván Arcuschin; Jett Janiak; Robert Krzyzanowski; Senthooran Rajamanoharan; Neel Nanda; Arthur Conmy
>
> **备注:** Accepted to the Reasoning and Planning for LLMs Workshop (ICLR 25), 10 main paper pages, 39 appendix pages
>
> **摘要:** Chain-of-Thought (CoT) reasoning has significantly advanced state-of-the-art AI capabilities. However, recent studies have shown that CoT reasoning is not always faithful when models face an explicit bias in their prompts, i.e., the CoT can give an incorrect picture of how models arrive at conclusions. We go further and show that unfaithful CoT can also occur on realistic prompts with no artificial bias. We find that when separately presented with the questions "Is X bigger than Y?" and "Is Y bigger than X?", models sometimes produce superficially coherent arguments to justify systematically answering Yes to both questions or No to both questions, despite such responses being logically contradictory. We show preliminary evidence that this is due to models' implicit biases towards Yes or No, thus labeling this unfaithfulness as Implicit Post-Hoc Rationalization. Our results reveal that several production models exhibit surprisingly high rates of post-hoc rationalization in our settings: GPT-4o-mini (13%) and Haiku 3.5 (7%). While frontier models are more faithful, especially thinking ones, none are entirely faithful: Gemini 2.5 Flash (2.17%), ChatGPT-4o (0.49%), DeepSeek R1 (0.37%), Gemini 2.5 Pro (0.14%), and Sonnet 3.7 with thinking (0.04%). We also investigate Unfaithful Illogical Shortcuts, where models use subtly illogical reasoning to try to make a speculative answer to hard maths problems seem rigorously proven. Our findings raise challenges for strategies for detecting undesired behavior in LLMs via the chain of thought.
>
---
#### [replaced 062] Exploring news intent and its application: A theory-driven approach
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2312.16490v2](http://arxiv.org/pdf/2312.16490v2)**

> **作者:** Zhengjia Wang; Danding Wang; Qiang Sheng; Juan Cao; Siyuan Ma; Haonan Cheng
>
> **备注:** Accepted to Information Processing & Management. DOI: https://doi.org/10.1016/j.ipm.2025.104229
>
> **摘要:** Understanding the intent behind information is crucial. However, news as a medium of public discourse still lacks a structured investigation of perceived news intent and its application. To advance this field, this paper reviews interdisciplinary studies on intentional action and introduces a conceptual deconstruction-based news intent understanding framework (NINT). This framework identifies the components of intent, facilitating a structured representation of news intent and its applications. Building upon NINT, we contribute a new intent perception dataset. Moreover, we investigate the potential of intent assistance on news-related tasks, such as significant improvement (+2.2% macF1) in the task of fake news detection. We hope that our findings will provide valuable insights into action-based intent cognition and computational social science.
>
---
#### [replaced 063] GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11368v2](http://arxiv.org/pdf/2505.11368v2)**

> **作者:** Lingxiao Diao; Xinyue Xu; Wanxuan Sun; Cheng Yang; Zhuosheng Zhang
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** Large language models (LLMs) have been widely deployed as autonomous agents capable of following user instructions and making decisions in real-world applications. Previous studies have made notable progress in benchmarking the instruction following capabilities of LLMs in general domains, with a primary focus on their inherent commonsense knowledge. Recently, LLMs have been increasingly deployed as domain-oriented agents, which rely on domain-oriented guidelines that may conflict with their commonsense knowledge. These guidelines exhibit two key characteristics: they consist of a wide range of domain-oriented rules and are subject to frequent updates. Despite these challenges, the absence of comprehensive benchmarks for evaluating the domain-oriented guideline following capabilities of LLMs presents a significant obstacle to their effective assessment and further development. In this paper, we introduce GuideBench, a comprehensive benchmark designed to evaluate guideline following performance of LLMs. GuideBench evaluates LLMs on three critical aspects: (i) adherence to diverse rules, (ii) robustness to rule updates, and (iii) alignment with human preferences. Experimental results on a range of LLMs indicate substantial opportunities for improving their ability to follow domain-oriented guidelines.
>
---
#### [replaced 064] FigCaps-HF: A Figure-to-Caption Generative Framework and Benchmark with Human Feedback
- **分类: cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2307.10867v2](http://arxiv.org/pdf/2307.10867v2)**

> **作者:** Ashish Singh; Ashutosh Singh; Prateek Agarwal; Zixuan Huang; Arpita Singh; Tong Yu; Sungchul Kim; Victor Bursztyn; Nesreen K. Ahmed; Puneet Mathur; Erik Learned-Miller; Franck Dernoncourt; Ryan A. Rossi
>
> **备注:** 16 pages, 4 figures. Benchmark Documentation: https://figcapshf.github.io/
>
> **摘要:** Captions are crucial for understanding scientific visualizations and documents. Existing captioning methods for scientific figures rely on figure-caption pairs extracted from documents for training, many of which fall short with respect to metrics like helpfulness, explainability, and visual-descriptiveness [15] leading to generated captions being misaligned with reader preferences. To enable the generation of high-quality figure captions, we introduce FigCaps-HF a new framework for figure-caption generation that can incorporate domain expert feedback in generating captions optimized for reader preferences. Our framework comprises of 1) an automatic method for evaluating quality of figure-caption pairs, 2) a novel reinforcement learning with human feedback (RLHF) method to optimize a generative figure-to-caption model for reader preferences. We demonstrate the effectiveness of our simple learning framework by improving performance over standard fine-tuning across different types of models. In particular, when using BLIP as the base model, our RLHF framework achieves a mean gain of 35.7%, 16.9%, and 9% in ROUGE, BLEU, and Meteor, respectively. Finally, we release a large-scale benchmark dataset with human feedback on figure-caption pairs to enable further evaluation and development of RLHF techniques for this problem.
>
---
#### [replaced 065] SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Modeling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.04619v2](http://arxiv.org/pdf/2503.04619v2)**

> **作者:** Xin Zhang; Qiyu Wei; Yingjie Zhu; Linhai Zhang; Deyu Zhou; Sophia Ananiadou
>
> **备注:** Accepted at ACL 2025
>
> **摘要:** User reviews on e-commerce platforms exhibit dynamic sentiment patterns driven by temporal and contextual factors. Traditional sentiment analysis methods focus on static reviews, failing to capture the evolving temporal relationship between user sentiment rating and textual content. Sentiment analysis on streaming reviews addresses this limitation by modeling and predicting the temporal evolution of user sentiments. However, it suffers from data sparsity, manifesting in temporal, spatial, and combined forms. In this paper, we introduce SynGraph, a novel framework designed to address data sparsity in sentiment analysis on streaming reviews. SynGraph alleviates data sparsity by categorizing users into mid-tail, long-tail, and extreme scenarios and incorporating LLM-augmented enhancements within a dynamic graph-based structure. Experiments on real-world datasets demonstrate its effectiveness in addressing sparsity and improving sentiment modeling in streaming reviews.
>
---
#### [replaced 066] Do Large Language Models Exhibit Cognitive Dissonance? Studying the Difference Between Revealed Beliefs and Stated Answers
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.14986v3](http://arxiv.org/pdf/2406.14986v3)**

> **作者:** Manuel Mondal; Ljiljana Dolamic; Gérôme Bovet; Philippe Cudré-Mauroux; Julien Audiffren
>
> **摘要:** Multiple Choice Questions (MCQ) have become a commonly used approach to assess the capabilities of Large Language Models (LLMs), due to their ease of manipulation and evaluation. The experimental appraisals of the LLMs' Stated Answer (their answer to MCQ) have pointed to their apparent ability to perform probabilistic reasoning or to grasp uncertainty. In this work, we investigate whether these aptitudes are measurable outside tailored prompting and MCQ by reformulating these issues as direct text-completion - the fundamental computational unit of LLMs. We introduce Revealed Belief, an evaluation framework that evaluates LLMs on tasks requiring reasoning under uncertainty, which complements MCQ scoring by analyzing text-completion probability distributions. Our findings suggest that while LLMs frequently state the correct answer, their Revealed Belief shows that they often allocate probability mass inconsistently, exhibit systematic biases, and often fail to update their beliefs appropriately when presented with new evidence, leading to strong potential impacts on downstream tasks. These results suggest that common evaluation methods may only provide a partial picture and that more research is needed to assess the extent and nature of their capabilities.
>
---
#### [replaced 067] Inherent and emergent liability issues in LLM-based agentic systems: a principal-agent perspective
- **分类: cs.CY; cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2504.03255v2](http://arxiv.org/pdf/2504.03255v2)**

> **作者:** Garry A. Gabison; R. Patrick Xian
>
> **备注:** 22 pages (incl. appendix), accepted at REALM workshop, ACL2025
>
> **摘要:** Agentic systems powered by large language models (LLMs) are becoming progressively more complex and capable. Their increasing agency and expanding deployment settings attract growing attention to effective governance policies, monitoring, and control protocols. Based on the emerging landscape of the agentic market, we analyze potential liability issues arising from the delegated use of LLM agents and their extended systems through a principal-agent perspective. Our analysis complements existing risk-based studies on artificial agency and covers the spectrum of important aspects of the principal-agent relationship and their potential consequences at deployment. Furthermore, we motivate method developments for technical governance along the directions of interpretability and behavior evaluations, reward and conflict management, and the mitigation of misalignment and misconduct through principled engineering of detection and fail-safe mechanisms. By illustrating the outstanding issues in AI liability for LLM-based agentic systems, we aim to inform the system design, auditing, and tracing to enhance transparency and liability attribution.
>
---
#### [replaced 068] Uncovering Overfitting in Large Language Model Editing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.07819v2](http://arxiv.org/pdf/2410.07819v2)**

> **作者:** Mengqi Zhang; Xiaotian Ye; Qiang Liu; Pengjie Ren; Shu Wu; Zhumin Chen
>
> **备注:** ICLR 2025
>
> **摘要:** Knowledge editing has been proposed as an effective method for updating and correcting the internal knowledge of Large Language Models (LLMs). However, existing editing methods often struggle with complex tasks, such as multi-hop reasoning. In this paper, we identify and investigate the phenomenon of Editing Overfit, where edited models assign disproportionately high probabilities to the edit target, hindering the generalization of new knowledge in complex scenarios. We attribute this issue to the current editing paradigm, which places excessive emphasis on the direct correspondence between the input prompt and the edit target for each edit sample. To further explore this issue, we introduce a new benchmark, EVOKE (EValuation of Editing Overfit in Knowledge Editing), along with fine-grained evaluation metrics. Through comprehensive experiments and analysis, we demonstrate that Editing Overfit is prevalent in current editing methods and that common overfitting mitigation strategies are ineffective in knowledge editing. To overcome this, inspired by LLMs' knowledge recall mechanisms, we propose a new plug-and-play strategy called Learn the Inference (LTI), which introduce a Multi-stage Inference Constraint module to guide the edited models in recalling new knowledge similarly to how unedited LLMs leverage knowledge through in-context learning. Extensive experimental results across a wide range of tasks validate the effectiveness of LTI in mitigating Editing Overfit.
>
---
#### [replaced 069] Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.18653v3](http://arxiv.org/pdf/2410.18653v3)**

> **作者:** Esteban Garces Arias; Hannah Blocher; Julian Rodemann; Meimingwei Li; Christian Heumann; Matthias Aßenmacher
>
> **备注:** Accepted at the $GEM^2$ Workshop (co-located with ACL 2025)
>
> **摘要:** Open-ended text generation has become a prominent task in natural language processing due to the rise of powerful (large) language models. However, evaluating the quality of these models and the employed decoding strategies remains challenging due to trade-offs among widely used metrics such as coherence, diversity, and perplexity. This paper addresses the specific problem of multicriteria evaluation for open-ended text generation, proposing novel methods for both relative and absolute rankings of decoding methods. Specifically, we employ benchmarking approaches based on partial orderings and present a new summary metric to balance existing automatic indicators, providing a more holistic evaluation of text generation quality. Our experiments demonstrate that the proposed approaches offer a robust way to compare decoding strategies and serve as valuable tools to guide model selection for open-ended text generation tasks. We suggest future directions for improving evaluation methodologies in text generation and make our code, datasets, and models publicly available.
>
---
#### [replaced 070] A Hybrid Multi-Agent Prompting Approach for Simplifying Complex Sentences
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11681v2](http://arxiv.org/pdf/2506.11681v2)**

> **作者:** Pratibha Zunjare; Michael Hsiao
>
> **摘要:** This paper addresses the challenge of transforming complex sentences into sequences of logical, simplified sentences while preserving semantic and logical integrity with the help of Large Language Models. We propose a hybrid approach that combines advanced prompting with multi-agent architectures to enhance the sentence simplification process. Experimental results show that our approach was able to successfully simplify 70% of the complex sentences written for video game design application. In comparison, a single-agent approach attained a 48% success rate on the same task.
>
---
#### [replaced 071] What do Large Language Models Say About Animals? Investigating Risks of Animal Harm in Generated Text
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.04804v4](http://arxiv.org/pdf/2503.04804v4)**

> **作者:** Arturs Kanepajs; Aditi Basu; Sankalpa Ghose; Constance Li; Akshat Mehta; Ronak Mehta; Samuel David Tucker-Davis; Eric Zhou; Bob Fischer; Jacy Reese Anthis
>
> **摘要:** As machine learning systems become increasingly embedded in society, their impact on human and nonhuman life continues to escalate. Technical evaluations have addressed a variety of potential harms from large language models (LLMs) towards humans and the environment, but there is little empirical work regarding harms towards nonhuman animals. Following the growing recognition of animal protection in regulatory and ethical AI frameworks, we present AnimalHarmBench (AHB), a benchmark for risks of animal harm in LLM-generated text. Our benchmark dataset comprises 1,850 curated questions from Reddit post titles and 2,500 synthetic questions based on 50 animal categories (e.g., cats, reptiles) and 50 ethical scenarios with a 70-30 public-private split. Scenarios include open-ended questions about how to treat animals, practical scenarios with potential animal harm, and willingness-to-pay measures for the prevention of animal harm. Using the LLM-as-a-judge framework, responses are evaluated for their potential to increase or decrease harm, and evaluations are debiased for the tendency of judges to judge their own outputs more favorably. AHB reveals significant differences across frontier LLMs, animal categories, scenarios, and subreddits. We conclude with future directions for technical research and addressing the challenges of building evaluations on complex social and moral topics.
>
---
#### [replaced 072] CAPO: Cost-Aware Prompt Optimization
- **分类: cs.CL; cs.AI; cs.NE; stat.ML**

- **链接: [http://arxiv.org/pdf/2504.16005v4](http://arxiv.org/pdf/2504.16005v4)**

> **作者:** Tom Zehle; Moritz Schlager; Timo Heiß; Matthias Feurer
>
> **备注:** Submitted to AutoML 2025
>
> **摘要:** Large language models (LLMs) have revolutionized natural language processing by solving a wide range of tasks simply guided by a prompt. Yet their performance is highly sensitive to prompt formulation. While automatic prompt optimization addresses this challenge by finding optimal prompts, current methods require a substantial number of LLM calls and input tokens, making prompt optimization expensive. We introduce CAPO (Cost-Aware Prompt Optimization), an algorithm that enhances prompt optimization efficiency by integrating AutoML techniques. CAPO is an evolutionary approach with LLMs as operators, incorporating racing to save evaluations and multi-objective optimization to balance performance with prompt length. It jointly optimizes instructions and few-shot examples while leveraging task descriptions for improved robustness. Our extensive experiments across diverse datasets and LLMs demonstrate that CAPO outperforms state-of-the-art discrete prompt optimization methods in 11/15 cases with improvements up to 21%p in accuracy. Our algorithm achieves better performances already with smaller budgets, saves evaluations through racing, and decreases average prompt length via a length penalty, making it both cost-efficient and cost-aware. Even without few-shot examples, CAPO outperforms its competitors and generally remains robust to initial prompts. CAPO represents an important step toward making prompt optimization more powerful and accessible by improving cost-efficiency.
>
---
#### [replaced 073] Discrete Audio Tokens: More Than a Survey!
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.10274v2](http://arxiv.org/pdf/2506.10274v2)**

> **作者:** Pooneh Mousavi; Gallil Maimon; Adel Moumen; Darius Petermann; Jiatong Shi; Haibin Wu; Haici Yang; Anastasia Kuznetsova; Artem Ploujnikov; Ricard Marxer; Bhuvana Ramabhadran; Benjamin Elizalde; Loren Lugosch; Jinyu Li; Cem Subakan; Phil Woodland; Minje Kim; Hung-yi Lee; Shinji Watanabe; Yossi Adi; Mirco Ravanelli
>
> **摘要:** Discrete audio tokens are compact representations that aim to preserve perceptual quality, phonetic content, and speaker characteristics while enabling efficient storage and inference, as well as competitive performance across diverse downstream tasks. They provide a practical alternative to continuous features, enabling the integration of speech and audio into modern large language models (LLMs). As interest in token-based audio processing grows, various tokenization methods have emerged, and several surveys have reviewed the latest progress in the field. However, existing studies often focus on specific domains or tasks and lack a unified comparison across various benchmarks. This paper presents a systematic review and benchmark of discrete audio tokenizers, covering three domains: speech, music, and general audio. We propose a taxonomy of tokenization approaches based on encoder-decoder, quantization techniques, training paradigm, streamability, and application domains. We evaluate tokenizers on multiple benchmarks for reconstruction, downstream performance, and acoustic language modeling, and analyze trade-offs through controlled ablation studies. Our findings highlight key limitations, practical considerations, and open challenges, providing insight and guidance for future research in this rapidly evolving area. For more information, including our main results and tokenizer database, please refer to our website: https://poonehmousavi.github.io/dates-website/.
>
---
#### [replaced 074] EuroLLM-9B: Technical Report
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.04079v2](http://arxiv.org/pdf/2506.04079v2)**

> **作者:** Pedro Henrique Martins; João Alves; Patrick Fernandes; Nuno M. Guerreiro; Ricardo Rei; Amin Farajian; Mateusz Klimaszewski; Duarte M. Alves; José Pombal; Nicolas Boizard; Manuel Faysse; Pierre Colombo; François Yvon; Barry Haddow; José G. C. de Souza; Alexandra Birch; André F. T. Martins
>
> **备注:** 56 pages
>
> **摘要:** This report presents EuroLLM-9B, a large language model trained from scratch to support the needs of European citizens by covering all 24 official European Union languages and 11 additional languages. EuroLLM addresses the issue of European languages being underrepresented and underserved in existing open large language models. We provide a comprehensive overview of EuroLLM-9B's development, including tokenizer design, architectural specifications, data filtering, and training procedures. We describe the pre-training data collection and filtering pipeline, including the creation of EuroFilter, an AI-based multilingual filter, as well as the design of EuroBlocks-Synthetic, a novel synthetic dataset for post-training that enhances language coverage for European languages. Evaluation results demonstrate EuroLLM-9B's competitive performance on multilingual benchmarks and machine translation tasks, establishing it as the leading open European-made LLM of its size. To support open research and adoption, we release all major components of this work, including the base and instruction-tuned models, the EuroFilter classifier, and the synthetic post-training dataset.
>
---
#### [replaced 075] Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.05243v3](http://arxiv.org/pdf/2410.05243v3)**

> **作者:** Boyu Gou; Ruohan Wang; Boyuan Zheng; Yanan Xie; Cheng Chang; Yiheng Shu; Huan Sun; Yu Su
>
> **备注:** Accepted to ICLR 2025 (Oral). Project Homepage: https://osu-nlp-group.github.io/UGround/
>
> **摘要:** Multimodal large language models (MLLMs) are transforming the capabilities of graphical user interface (GUI) agents, facilitating their transition from controlled simulations to complex, real-world applications across various platforms. However, the effectiveness of these agents hinges on the robustness of their grounding capability. Current GUI agents predominantly utilize text-based representations such as HTML or accessibility trees, which, despite their utility, often introduce noise, incompleteness, and increased computational overhead. In this paper, we advocate a human-like embodiment for GUI agents that perceive the environment entirely visually and directly perform pixel-level operations on the GUI. The key is visual grounding models that can accurately map diverse referring expressions of GUI elements to their coordinates on the GUI across different platforms. We show that a simple recipe, which includes web-based synthetic data and slight adaptation of the LLaVA architecture, is surprisingly effective for training such visual grounding models. We collect the largest dataset for GUI visual grounding so far, containing 10M GUI elements and their referring expressions over 1.3M screenshots, and use it to train UGround, a strong universal visual grounding model for GUI agents. Empirical results on six benchmarks spanning three categories (grounding, offline agent, and online agent) show that 1) UGround substantially outperforms existing visual grounding models for GUI agents, by up to 20% absolute, and 2) agents with UGround outperform state-of-the-art agents, despite the fact that existing agents use additional text-based input while ours only uses visual perception. These results provide strong support for the feasibility and promises of GUI agents that navigate the digital world as humans do.
>
---
#### [replaced 076] Graph RAG for Legal Norms: A Hierarchical, Temporal and Deterministic Approach
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2505.00039v3](http://arxiv.org/pdf/2505.00039v3)**

> **作者:** Hudson de Martim
>
> **备注:** This version enhances the theoretical underpinnings of the proposed Graph RAG methodology, including the introduction of a formal, FRBRoo-based model for versioning, and enabling multi-language support for both content and metadata
>
> **摘要:** This article proposes an adaptation of Graph Retrieval-Augmented Generation (Graph RAG) specifically designed for the analysis and comprehension of legal norms. Legal texts are characterized by a predefined hierarchical structure, an extensive network of references and a continuous evolution through multiple temporal versions. This temporal dynamism poses a significant challenge for standard AI systems, demanding a deterministic representation of the law at any given point in time. To address this, our approach grounds the knowledge graph construction in a formal, FRBRoo-inspired model that distinguishes abstract legal works from their concrete textual expressions. We introduce a multi-layered representation of Temporal Versions (capturing date-specific changes) and Language Versions (capturing linguistic variations). By modeling normative evolution as a precise sequence of these versioned entities, we enable the construction of a knowledge graph that serves as a verifiable "ground truth". This allows Large Language Models to generate responses based on accurate, context-aware, and point-in-time correct legal information, overcoming the risk of temporal inaccuracies. Through a detailed analysis of this formal Graph RAG approach and its application to legal norm datasets, this article aims to advance the field of Artificial Intelligence applied to Law, creating opportunities for more effective and reliable systems in legal research, legislative analysis, and decision support.
>
---
