# 自然语言处理 cs.CL

- **最新发布 115 篇**

- **更新 99 篇**

## 最新发布

#### [new 001] NEU-ESC: A Comprehensive Vietnamese dataset for Educational Sentiment analysis and topic Classification toward multitask learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出NEU-ESC数据集，用于越南语教育情感分析与主题分类任务，解决资源不足和领域相关性低的问题。**

- **链接: [http://arxiv.org/pdf/2506.23524v1](http://arxiv.org/pdf/2506.23524v1)**

> **作者:** Phan Quoc Hung Mai; Quang Hung Nguyen; Phuong Giang Duong; Hong Hanh Nguyen; Nguyen Tuan Long
>
> **摘要:** In the field of education, understanding students' opinions through their comments is crucial, especially in the Vietnamese language, where resources remain limited. Existing educational datasets often lack domain relevance and student slang. To address these gaps, we introduce NEU-ESC, a new Vietnamese dataset for Educational Sentiment Classification and Topic Classification, curated from university forums, which offers more samples, richer class diversity, longer texts, and broader vocabulary. In addition, we explore multitask learning using encoder-only language models (BERT), in which we showed that it achieves performance up to 83.7% and 79.8% accuracy for sentiment and topic classification tasks. We also benchmark our dataset and model with other datasets and models, including Large Language Models, and discuss these benchmarks. The dataset is publicly available at: https://huggingface.co/datasets/hung20gg/NEU-ESC.
>
---
#### [new 002] L0: Reinforcement Learning to Become General Agents
- **分类: cs.CL**

- **简介: 该论文提出L0系统，解决大模型作为通用代理的训练难题，通过强化学习提升任务解决能力。**

- **链接: [http://arxiv.org/pdf/2506.23667v1](http://arxiv.org/pdf/2506.23667v1)**

> **作者:** Junjie Zhang; Jingyi Xi; Zhuoyang Song; Junyu Lu; Yuhua Ke; Ting Sun; Yukun Yang; Jiaxing Zhang; Songxin Zhang; Zejian Xie
>
> **摘要:** Training large language models (LLMs) to act as autonomous agents for multi-turn, long-horizon tasks remains significant challenges in scalability and training efficiency. To address this, we introduce L-Zero (L0), a scalable, end-to-end training pipeline for general-purpose agents. Featuring a low-cost, extensible, and sandboxed concurrent agent worker pool, L0 lowers the barrier for applying reinforcement learning in complex environments. We also introduce NB-Agent, the agent scaffold within L0, which operates in a "code-as-action" fashion via a Read-Eval-Print-Loop (REPL). We evaluate L0 on factuality question-answering benchmarks. Our experiments demonstrate that a base model can develop robust problem-solving skills using solely Reinforcement Learning with Verifiable Rewards (RLVR). On the Qwen2.5-7B-Instruct model, our method boosts accuracy on SimpleQA from 30 % to 80 % and on HotpotQA from 22 % to 41 %. We have open-sourced the entire L0 system, including our L0 series models, the NB-Agent, a complete training pipeline, and the corresponding training recipes on (https://github.com/cmriat/l0).
>
---
#### [new 003] Temperature Matters: Enhancing Watermark Robustness Against Paraphrasing Attacks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本检测任务，旨在解决LLM生成文本的水印鲁棒性问题。通过提出新方法并测试其对抗改写攻击的能力，提升水印的可靠性。**

- **链接: [http://arxiv.org/pdf/2506.22623v1](http://arxiv.org/pdf/2506.22623v1)**

> **作者:** Badr Youbi Idrissi; Monica Millunzi; Amelia Sorrenti; Lorenzo Baraldi; Daryna Dementieva
>
> **摘要:** In the present-day scenario, Large Language Models (LLMs) are establishing their presence as powerful instruments permeating various sectors of society. While their utility offers valuable support to individuals, there are multiple concerns over potential misuse. Consequently, some academic endeavors have sought to introduce watermarking techniques, characterized by the inclusion of markers within machine-generated text, to facilitate algorithmic identification. This research project is focused on the development of a novel methodology for the detection of synthetic text, with the overarching goal of ensuring the ethical application of LLMs in AI-driven text generation. The investigation commences with replicating findings from a previous baseline study, thereby underscoring its susceptibility to variations in the underlying generation model. Subsequently, we propose an innovative watermarking approach and subject it to rigorous evaluation, employing paraphrased generated text to asses its robustness. Experimental results highlight the robustness of our proposal compared to the~\cite{aarson} watermarking method.
>
---
#### [new 004] Graft: Integrating the Domain Knowledge via Efficient Parameter Synergy for MLLMs
- **分类: cs.CL**

- **简介: 该论文属于多模态大模型领域，旨在解决领域专用MLLMs知识碎片化问题，提出CAPS框架实现高效参数融合与知识共享。**

- **链接: [http://arxiv.org/pdf/2506.23940v1](http://arxiv.org/pdf/2506.23940v1)**

> **作者:** Yang Dai; Jianxiang An; Tianwei Lin; Hongyang He; Hongzhe Huang; Wenqiao Zhang; Zheqi Lv; Siliang Tang; Yueting Zhuang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved success across various domains. However, their applicability tends to degrade when confronted with different types of data inputs, especially for MLLMs that have been fine-tuned for specific tasks. Despite its importance, the study of knowledge sharing among domain-specific MLLMs--such as those trained for mathematics or code--remains largely underexplored. To address the fragmentation of knowledge across domain-specialized MLLMs, we propose a unified parameter integration framework that enables modular composition of expert capabilities. Our method is grounded in a novel Compatibility-Aware Parameter Splicing (CAPS) strategy, which leverages both local functional attribution and global information-theoretic signals to guide selective parameter fusion. By extending this mechanism to the low-rank adaptation layer granularity, we ensure efficient integration with minimal inference overhead. Furthermore, we introduce a domain compatibility scoring mechanism that quantifies inter-expert alignment at the activation level and correlates with downstream task utility. This principled fusion protocol allows the final model to synergize heterogeneous expertise while preserving structural modularity. Extensive evaluations across diverse multimodal benchmarks validate the effectiveness of our framework, offering a scalable path toward compositional, domain-adaptive MLLMs.
>
---
#### [new 005] Datasets for Fairness in Language Models: An In-Depth Survey
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 该论文属于自然语言处理中的公平性研究任务，旨在解决语言模型评估中数据集偏差问题，通过分析现有数据集并提出统一评估框架来揭示和减少不公平现象。**

- **链接: [http://arxiv.org/pdf/2506.23411v1](http://arxiv.org/pdf/2506.23411v1)**

> **作者:** Jiale Zhang; Zichong Wang; Avash Palikhe; Zhipeng Yin; Wenbin Zhang
>
> **摘要:** Fairness benchmarks play a central role in shaping how we evaluate language models, yet surprisingly little attention has been given to examining the datasets that these benchmarks rely on. This survey addresses that gap by presenting a broad and careful review of the most widely used fairness datasets in current language model research, characterizing them along several key dimensions including their origin, scope, content, and intended use to help researchers better appreciate the assumptions and limitations embedded in these resources. To support more meaningful comparisons and analyses, we introduce a unified evaluation framework that reveals consistent patterns of demographic disparities across datasets and scoring methods. Applying this framework to twenty four common benchmarks, we highlight the often overlooked biases that can influence conclusions about model fairness and offer practical guidance for selecting, combining, and interpreting these datasets. We also point to opportunities for creating new fairness benchmarks that reflect more diverse social contexts and encourage more thoughtful use of these tools going forward. All code, data, and detailed results are publicly available at https://github.com/vanbanTruong/Fairness-in-Large-Language-Models/tree/main/datasets to promote transparency and reproducibility across the research community.
>
---
#### [new 006] Advancing Multi-Step Mathematical Reasoning in Large Language Models through Multi-Layered Self-Reflection with Auto-Prompting
- **分类: cs.CL**

- **简介: 该论文属于多步骤数学推理任务，旨在解决大语言模型在复杂推理中的不足。通过引入MAPS框架，结合自我反思和自动提示，提升模型的推理能力。**

- **链接: [http://arxiv.org/pdf/2506.23888v1](http://arxiv.org/pdf/2506.23888v1)**

> **作者:** André de Souza Loureiro; Jorge Valverde-Rebaza; Julieta Noguez; David Escarcega; Ricardo Marcacini
>
> **备注:** Accepted for publication in: European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD 2025). Research Track
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have significantly improved their problem-solving capabilities. However, these models still struggle when faced with complex multi-step reasoning tasks. In this paper, we propose the Multi-Layered Self-Reflection with Auto-Prompting (MAPS) framework, a novel approach designed to enhance multi-step mathematical reasoning in LLMs by integrating techniques such as Chain of Thought (CoT), Self-Reflection, and Auto-Prompting. Unlike traditional static prompting methods, MAPS employs an iterative refinement process. Initially, the model generates a solution using CoT prompting. When errors are detected, an adaptive self-reflection mechanism identifies and analyzes them, generating tailored prompts to guide corrections. These dynamically adjusted prompts enable the model to iteratively refine its reasoning. Experiments on four well-established benchmarks across multiple LLMs show that MAPS significantly outperforms standard CoT and achieves competitive results with reasoning-optimized models. In addition, MAPS enables general-purpose LLMs to reach performance levels comparable to specialized reasoning models. While deeper reflection layers improve accuracy, they also increase token usage and costs. To balance this trade-off, MAPS strategically limits reflection depth, ensuring an optimal balance between cost and reasoning performance.
>
---
#### [new 007] Jan-nano Technical Report
- **分类: cs.CL**

- **简介: 该论文介绍Jan-nano，一个4B参数的语言模型，通过特殊设计实现高效推理。任务是语言建模，解决计算资源与性能的平衡问题，采用新训练方法提升效率。**

- **链接: [http://arxiv.org/pdf/2506.22760v1](http://arxiv.org/pdf/2506.22760v1)**

> **作者:** Alan Dao; Dinh Bach Vu
>
> **摘要:** Most language models face a fundamental tradeoff where powerful capabilities require substantial computational resources. We shatter this constraint with Jan-nano, a 4B parameter language model that redefines efficiency through radical specialization: instead of trying to know everything, it masters the art of finding anything instantly. Fine-tuned from Qwen3-4B using our novel multi-stage RLVR system that completely eliminates reliance on next token prediction training (SFT), Jan-nano achieves 83.2% on SimpleQA benchmark with MCP integration while running on consumer hardware. With 128K context length, Jan-nano proves that intelligence isn't about scale, it's about strategy.
>
---
#### [new 008] DICE-BENCH: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型工具使用能力评估任务，旨在解决现有基准在多轮多对话场景中的不足，提出DICE-BENCH框架提升真实场景下的模型表现。**

- **链接: [http://arxiv.org/pdf/2506.22853v1](http://arxiv.org/pdf/2506.22853v1)**

> **作者:** Kyochul Jang; Donghyeon Lee; Kyusik Kim; Dongseok Heo; Taewhoo Lee; Woojeong Kim; Bongwon Suh
>
> **备注:** 9 pages, ACL 2025 Vienna
>
> **摘要:** Existing function-calling benchmarks focus on single-turn interactions. However, they overlook the complexity of real-world scenarios. To quantify how existing benchmarks address practical applications, we introduce DICE-SCORE, a metric that evaluates the dispersion of tool-related information such as function name and parameter values throughout the dialogue. Analyzing existing benchmarks through DICE-SCORE reveals notably low scores, highlighting the need for more realistic scenarios. To address this gap, we present DICE-BENCH, a framework that constructs practical function-calling datasets by synthesizing conversations through a tool graph that maintains dependencies across rounds and a multi-agent system with distinct personas to enhance dialogue naturalness. The final dataset comprises 1,607 high-DICE-SCORE instances. Our experiments on 19 LLMs with DICE-BENCH show that significant advances are still required before such models can be deployed effectively in real-world settings. Our code and data are all publicly available: https://snuhcc.github.io/DICE-Bench/.
>
---
#### [new 009] Positional Bias in Binary Question Answering: How Uncertainty Shapes Model Preferences
- **分类: cs.CL**

- **简介: 该论文属于问答任务，研究二分类问答中的位置偏差问题。通过构造不同不确定性数据集，分析模型在不确定情况下的偏好变化。**

- **链接: [http://arxiv.org/pdf/2506.23743v1](http://arxiv.org/pdf/2506.23743v1)**

> **作者:** Tiziano Labruna; Simone Gallo; Giovanni Da San Martino
>
> **摘要:** Positional bias in binary question answering occurs when a model systematically favors one choice over another based solely on the ordering of presented options. In this study, we quantify and analyze positional bias across five large language models under varying degrees of answer uncertainty. We re-adapted the SQuAD-it dataset by adding an extra incorrect answer option and then created multiple versions with progressively less context and more out-of-context answers, yielding datasets that range from low to high uncertainty. Additionally, we evaluate two naturally higher-uncertainty benchmarks: (1) WebGPT - question pairs with unequal human-assigned quality scores, and (2) Winning Arguments - where models predict the more persuasive argument in Reddit's r/ChangeMyView exchanges. Across each dataset, the order of the "correct" (or higher-quality/persuasive) option is systematically flipped (first placed in position 1, then in position 2) to compute both Preference Fairness and Position Consistency. We observe that positional bias is nearly absent under low-uncertainty conditions, but grows exponentially when it becomes doubtful to decide which option is correct.
>
---
#### [new 010] The Trilemma of Truth in Large Language Models
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于模型验证任务，旨在解决评估大语言模型知识真实性的问题。工作包括提出sAwMIL方法，分析模型内部信号，揭示真伪判断的复杂性。**

- **链接: [http://arxiv.org/pdf/2506.23921v1](http://arxiv.org/pdf/2506.23921v1)**

> **作者:** Germans Savcisens; Tina Eliassi-Rad
>
> **摘要:** We often attribute human characteristics to large language models (LLMs) and claim that they "know" certain things. LLMs have an internal probabilistic knowledge that represents information retained during training. How can we assess the veracity of this knowledge? We examine two common methods for probing the veracity of LLMs and discover several assumptions that are flawed. To address these flawed assumptions, we introduce sAwMIL (short for Sparse Aware Multiple-Instance Learning), a probing method that utilizes the internal activations of LLMs to separate statements into true, false, and neither. sAwMIL is based on multiple-instance learning and conformal prediction. We evaluate sAwMIL on 5 validity criteria across 16 open-source LLMs, including both default and chat-based variants, as well as on 3 new datasets. Among the insights we provide are: (1) the veracity signal is often concentrated in the third quarter of an LLM's depth; (2) truth and falsehood signals are not always symmetric; (3) linear probes perform better on chat models than on default models; (4) nonlinear probes may be required to capture veracity signals for some LLMs with reinforcement learning from human feedback or knowledge distillation; and (5) LLMs capture a third type of signal that is distinct from true and false and is neither true nor false. These findings provide a reliable method for verifying what LLMs "know" and how certain they are of their probabilistic internal knowledge.
>
---
#### [new 011] On the Predictive Power of Representation Dispersion in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型的表示分散性与文本预测能力的关系，属于自然语言处理任务。它探讨如何利用分散性提升模型性能和效率。**

- **链接: [http://arxiv.org/pdf/2506.24106v1](http://arxiv.org/pdf/2506.24106v1)**

> **作者:** Yanhong Li; Ming Li; Karen Livescu; Jiawei Zhou
>
> **摘要:** We show that a language model's ability to predict text is tightly linked to the breadth of its embedding space: models that spread their contextual representations more widely tend to achieve lower perplexity. Concretely, we find that representation dispersion - the average pairwise cosine distance among hidden vectors - strongly and negatively correlates with perplexity across diverse model families (LLaMA, Qwen, and others) and domains (Wikipedia, news, scientific abstracts). Beyond illustrating this link, we show how dispersion can be leveraged for a range of practical tasks without requiring labeled data. First, measuring dispersion on unlabeled text allows us to predict downstream accuracy in new domains, offering a data-efficient tool for model selection. Next, we find that identifying layers with higher dispersion pinpoints the best representations for retrieval-based methods such as kNN-LM, bypassing exhaustive layer-by-layer searches. Finally, we integrate a simple push-away objective into training, which increases dispersion in both single-domain and cross-domain scenarios and directly improves perplexity in each.
>
---
#### [new 012] Large Language Models Don't Make Sense of Word Problems. A Scoping Review from a Mathematics Education Perspective
- **分类: cs.CL; math.HO**

- **简介: 该论文属于教育技术任务，探讨LLMs在数学应用题中的表现。研究指出LLMs仅掌握表面解题方法，未真正理解现实情境，限制其教学价值。**

- **链接: [http://arxiv.org/pdf/2506.24006v1](http://arxiv.org/pdf/2506.24006v1)**

> **作者:** Anselm R. Strohmaier; Wim Van Dooren; Kathrin Seßler; Brian Greer; Lieven Verschaffel
>
> **摘要:** The progress of Large Language Models (LLMs) like ChatGPT raises the question of how they can be integrated into education. One hope is that they can support mathematics learning, including word-problem solving. Since LLMs can handle textual input with ease, they appear well-suited for solving mathematical word problems. Yet their real competence, whether they can make sense of the real-world context, and the implications for classrooms remain unclear. We conducted a scoping review from a mathematics-education perspective, including three parts: a technical overview, a systematic review of word problems used in research, and a state-of-the-art empirical evaluation of LLMs on mathematical word problems. First, in the technical overview, we contrast the conceptualization of word problems and their solution processes between LLMs and students. In computer-science research this is typically labeled mathematical reasoning, a term that does not align with usage in mathematics education. Second, our literature review of 213 studies shows that the most popular word-problem corpora are dominated by s-problems, which do not require a consideration of realities of their real-world context. Finally, our evaluation of GPT-3.5-turbo, GPT-4o-mini, GPT-4.1, and o3 on 287 word problems shows that most recent LLMs solve these s-problems with near-perfect accuracy, including a perfect score on 20 problems from PISA. LLMs still showed weaknesses in tackling problems where the real-world context is problematic or non-sensical. In sum, we argue based on all three aspects that LLMs have mastered a superficial solution process but do not make sense of word problems, which potentially limits their value as instructional tools in mathematics classrooms.
>
---
#### [new 013] FairI Tales: Evaluation of Fairness in Indian Contexts with a Focus on Bias and Stereotypes
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的公平性评估任务，旨在解决印度语境下大语言模型的偏见与刻板印象问题。研究构建了INDIC-BIAS基准，评估14个模型在85个身份群体中的公平性表现。**

- **链接: [http://arxiv.org/pdf/2506.23111v1](http://arxiv.org/pdf/2506.23111v1)**

> **作者:** Janki Atul Nawale; Mohammed Safi Ur Rahman Khan; Janani D; Mansi Gupta; Danish Pruthi; Mitesh M. Khapra
>
> **备注:** Accepted in ACL 2025
>
> **摘要:** Existing studies on fairness are largely Western-focused, making them inadequate for culturally diverse countries such as India. To address this gap, we introduce INDIC-BIAS, a comprehensive India-centric benchmark designed to evaluate fairness of LLMs across 85 identity groups encompassing diverse castes, religions, regions, and tribes. We first consult domain experts to curate over 1,800 socio-cultural topics spanning behaviors and situations, where biases and stereotypes are likely to emerge. Grounded in these topics, we generate and manually validate 20,000 real-world scenario templates to probe LLMs for fairness. We structure these templates into three evaluation tasks: plausibility, judgment, and generation. Our evaluation of 14 popular LLMs on these tasks reveals strong negative biases against marginalized identities, with models frequently reinforcing common stereotypes. Additionally, we find that models struggle to mitigate bias even when explicitly asked to rationalize their decision. Our evaluation provides evidence of both allocative and representational harms that current LLMs could cause towards Indian identities, calling for a more cautious usage in practical applications. We release INDIC-BIAS as an open-source benchmark to advance research on benchmarking and mitigating biases and stereotypes in the Indian context.
>
---
#### [new 014] SoMi-ToM: Evaluating Multi-Perspective Theory of Mind in Embodied Social Interactions
- **分类: cs.CL; cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出SoMi-ToM基准，用于评估多视角理论心智在具身社交互动中的能力。旨在解决现有基准与真实社交互动差距大的问题，通过多模态数据和双视角评估方法进行模型评测。**

- **链接: [http://arxiv.org/pdf/2506.23046v1](http://arxiv.org/pdf/2506.23046v1)**

> **作者:** Xianzhe Fan; Xuhui Zhou; Chuanyang Jin; Kolby Nottingham; Hao Zhu; Maarten Sap
>
> **备注:** 23 pages, 6 figures
>
> **摘要:** Humans continuously infer the states, goals, and behaviors of others by perceiving their surroundings in dynamic, real-world social interactions. However, most Theory of Mind (ToM) benchmarks only evaluate static, text-based scenarios, which have a significant gap compared to real interactions. We propose the SoMi-ToM benchmark, designed to evaluate multi-perspective ToM in embodied multi-agent complex social interactions. This benchmark is based on rich multimodal interaction data generated by the interaction environment SoMi, covering diverse crafting goals and social relationships. Our framework supports multi-level evaluation: (1) first-person evaluation provides multimodal (visual, dialogue, action, etc.) input from a first-person perspective during a task for real-time state inference, (2) third-person evaluation provides complete third-person perspective video and text records after a task for goal and behavior inference. This evaluation method allows for a more comprehensive examination of a model's ToM capabilities from both the subjective immediate experience and the objective global observation. We constructed a challenging dataset containing 35 third-person perspective videos, 363 first-person perspective images, and 1225 expert-annotated multiple-choice questions (three options). On this dataset, we systematically evaluated the performance of human subjects and several state-of-the-art large vision-language models (LVLMs). The results show that LVLMs perform significantly worse than humans on SoMi-ToM: the average accuracy gap between humans and models is 40.1% in first-person evaluation and 26.4% in third-person evaluation. This indicates that future LVLMs need to further improve their ToM capabilities in embodied, complex social interactions.
>
---
#### [new 015] RExBench: Can coding agents autonomously implement AI research extensions?
- **分类: cs.CL**

- **简介: 该论文属于AI研究扩展任务，旨在评估编码代理自主实现AI研究扩展的能力。通过构建RExBench基准，测试不同LLM代理的表现，发现其在无大量人工指导时表现不佳。**

- **链接: [http://arxiv.org/pdf/2506.22598v1](http://arxiv.org/pdf/2506.22598v1)**

> **作者:** Nicholas Edwards; Yukyung Lee; Yujun; Mao; Yulu Qin; Sebastian Schuster; Najoung Kim
>
> **摘要:** Agents based on Large Language Models (LLMs) have shown promise for performing sophisticated software engineering tasks autonomously. In addition, there has been progress towards developing agents that can perform parts of the research pipeline in machine learning and the natural sciences. We argue that research extension and its implementation is a critical capability for such systems, and introduce RExBench to support the evaluation of this capability. RExBench is a benchmark consisting of 12 realistic research experiment implementation tasks that aim to investigate research hypotheses that have not previously been implemented. Each task is set up as an extension to an existing research paper and codebase, accompanied by domain expert-written instructions. RExBench is robust to data contamination, and supports an automatic evaluation infrastructure that executes agent outputs to determine whether the success criteria are met. We use this benchmark to evaluate nine LLM agents implemented using three different frameworks: aider, Claude Code, and OpenHands. We find that all agents evaluated fail to autonomously implement the majority of the extensions. Although the success rate improves with additional human-written hints, the best performance under this setting remains below 40%. This indicates that current agents are still short of being able to handle realistic research extension tasks without substantial human guidance.
>
---
#### [new 016] Generalist Reward Models: Found Inside Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型对齐任务，解决奖励模型依赖人工数据的问题。工作发现LLM内隐含强大奖励模型，理论证明其有效性并验证性能优越。**

- **链接: [http://arxiv.org/pdf/2506.23235v1](http://arxiv.org/pdf/2506.23235v1)**

> **作者:** Yi-Chen Li; Tian Xu; Yang Yu; Xuqin Zhang; Xiong-Hui Chen; Zhongxiang Ling; Ningjing Chao; Lei Yuan; Zhi-Hua Zhou
>
> **摘要:** The alignment of Large Language Models (LLMs) is critically dependent on reward models trained on costly human preference data. While recent work explores bypassing this cost with AI feedback, these methods often lack a rigorous theoretical foundation. In this paper, we discover that a powerful generalist reward model is already latently present within any LLM trained via standard next-token prediction. We prove that this endogenous reward is not a heuristic, but is theoretically equivalent to a reward function learned through offline inverse reinforcement learning. This connection allows us to directly elicit a high-quality reward signal from a base (pre-trained or supervised fine-tuned) model without any further training. Critically, we also prove that subsequent reinforcement learning using this endogenous reward leads to a policy with a provably superior error bound compared to the base model. To our best knowledge, this is the first theoretical proof of the effectiveness of reinforcement learning for LLMs. Our experiments validate this theory, demonstrating that our method not only outperforms existing LLM-as-a-judge approaches but can also surpass explicitly trained reward models. These findings suggest that the reward modeling stage can be replaced by a principled method of eliciting the knowledge already captured during pre-training, heralding a more efficient, powerful, and scalable paradigm for LLMs alignment as well as multi-modal models.
>
---
#### [new 017] AI Agents-as-Judge: Automated Assessment of Accuracy, Consistency, Completeness and Clarity for Enterprise Documents
- **分类: cs.CL; cs.AI; 68T07, 68T50; I.2.1; I.2.3; I.2.7; H.3.3**

- **简介: 该论文属于文档质量评估任务，旨在解决企业文档的准确性、一致性等问题。通过构建AI代理系统实现自动化审查与优化。**

- **链接: [http://arxiv.org/pdf/2506.22485v1](http://arxiv.org/pdf/2506.22485v1)**

> **作者:** Sudip Dasgupta; Himanshu Shankar
>
> **备注:** 17 pages, 2 system diagrams, 1 table, no prior conference publication
>
> **摘要:** This study presents a modular, multi-agent system for the automated review of highly structured enterprise business documents using AI agents. Unlike prior solutions focused on unstructured texts or limited compliance checks, this framework leverages modern orchestration tools such as LangChain, CrewAI, TruLens, and Guidance to enable section-by-section evaluation of documents for accuracy, consistency, completeness, and clarity. Specialized agents, each responsible for discrete review criteria such as template compliance or factual correctness, operate in parallel or sequence as required. Evaluation outputs are enforced to a standardized, machine-readable schema, supporting downstream analytics and auditability. Continuous monitoring and a feedback loop with human reviewers allow for iterative system improvement and bias mitigation. Quantitative evaluation demonstrates that the AI Agent-as-Judge system approaches or exceeds human performance in key areas: achieving 99% information consistency (vs. 92% for humans), halving error and bias rates, and reducing average review time from 30 to 2.5 minutes per document, with a 95% agreement rate between AI and expert human judgment. While promising for a wide range of industries, the study also discusses current limitations, including the need for human oversight in highly specialized domains and the operational cost of large-scale LLM usage. The proposed system serves as a flexible, auditable, and scalable foundation for AI-driven document quality assurance in the enterprise context.
>
---
#### [new 018] Text2VectorSQL: Bridging Text-to-SQL and Vector Search for Unified Natural Language Queries
- **分类: cs.CL**

- **简介: 该论文提出Text2VectorSQL框架，融合文本到SQL与向量搜索，解决自然语言查询表达受限问题，提升多样性和准确性。**

- **链接: [http://arxiv.org/pdf/2506.23071v1](http://arxiv.org/pdf/2506.23071v1)**

> **作者:** Zhengren Wang; Bozhou Li; Dongwen Yao; Wentao Zhang
>
> **备注:** Work in progess
>
> **摘要:** While Text-to-SQL enables natural language interaction with structured databases, its effectiveness diminishes with unstructured data or ambiguous queries due to rigid syntax and limited expressiveness. Concurrently, vector search has emerged as a powerful paradigm for semantic retrieval, particularly for unstructured data. However, existing VectorSQL implementations still rely heavily on manual crafting and lack tailored evaluation frameworks, leaving a significant gap between theoretical potential and practical deployment. To bridge these complementary paradigms, we introduces Text2VectorSQL, a novel framework unifying Text-to-SQL and vector search to overcome expressiveness constraints and support more diverse and holistical natural language queries. Specifically, Text2VectorSQL enables semantic filtering, multi-modal matching, and retrieval acceleration. For evaluation, we build vector index on appropriate columns, extend user queries with semantic search, and annotate ground truths via an automatic pipeline with expert review. Furthermore, we develop dedicated Text2VectorSQL models with synthetic data, demonstrating significant performance improvements over baseline methods. Our work establishes the foundation for the Text2VectorSQL task, paving the way for more versatile and intuitive database interfaces. The repository will be publicly available at https://github.com/Open-DataFlow/Text2VectorSQL.
>
---
#### [new 019] Machine Understanding of Scientific Language
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于科学语言理解任务，旨在解决科学文本真实性识别问题。通过构建数据集和方法，提升机器对科学文本的分析能力，以检测虚假信息并促进科学传播。**

- **链接: [http://arxiv.org/pdf/2506.23990v1](http://arxiv.org/pdf/2506.23990v1)**

> **作者:** Dustin Wright
>
> **备注:** PhD Thesis, 210 pages
>
> **摘要:** Scientific information expresses human understanding of nature. This knowledge is largely disseminated in different forms of text, including scientific papers, news articles, and discourse among people on social media. While important for accelerating our pursuit of knowledge, not all scientific text is faithful to the underlying science. As the volume of this text has burgeoned online in recent years, it has become a problem of societal importance to be able to identify the faithfulness of a given piece of scientific text automatically. This thesis is concerned with the cultivation of datasets, methods, and tools for machine understanding of scientific language, in order to analyze and understand science communication at scale. To arrive at this, I present several contributions in three areas of natural language processing and machine learning: automatic fact checking, learning with limited data, and scientific text processing. These contributions include new methods and resources for identifying check-worthy claims, adversarial claim generation, multi-source domain adaptation, learning from crowd-sourced labels, cite-worthiness detection, zero-shot scientific fact checking, detecting exaggerated scientific claims, and modeling degrees of information change in science communication. Critically, I demonstrate how the research outputs of this thesis are useful for effectively learning from limited amounts of scientific text in order to identify misinformative scientific statements and generate new insights into the science communication process
>
---
#### [new 020] Flow-Modulated Scoring for Semantic-Aware Knowledge Graph Completion
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱补全任务，旨在解决静态嵌入方法难以捕捉上下文依赖和关系动态的问题。提出FMS框架，结合语义上下文和动态流匹配，提升关系语义建模效果。**

- **链接: [http://arxiv.org/pdf/2506.23137v1](http://arxiv.org/pdf/2506.23137v1)**

> **作者:** Siyuan Li; Ruitong Liu; Yan Wen; Te Sun
>
> **备注:** 10 pages
>
> **摘要:** Effective modeling of multifaceted relations is pivotal for Knowledge Graph Completion (KGC). However, a majority of existing approaches are predicated on static, embedding-based scoring, exhibiting inherent limitations in capturing contextual dependencies and relational dynamics. Addressing this gap, we propose the Flow-Modulated Scoring (FMS) framework. FMS comprises two principal components: (1) a semantic context learning module that encodes context-sensitive entity representations, and (2) a conditional flow-matching module designed to learn the dynamic transformation from a head to a tail embedding, governed by the aforementioned context. The resultant predictive vector field, representing the context-informed relational path, serves to dynamically refine the initial static score of an entity pair. Through this synergy of context-aware static representations and conditioned dynamic information, FMS facilitates a more profound modeling of relational semantics. Comprehensive evaluations on several standard benchmarks demonstrate that our proposed method surpasses prior state-of-the-art results.
>
---
#### [new 021] Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning
- **分类: cs.CL**

- **简介: 该论文属于分子结构解析任务，旨在解决LLMs在化学知识理解上的不足。通过构建知识库和设计评分模型，提升LLMs的解析性能。**

- **链接: [http://arxiv.org/pdf/2506.23056v1](http://arxiv.org/pdf/2506.23056v1)**

> **作者:** Xiang Zhuang; Bin Wu; Jiyu Cui; Kehua Feng; Xiaotong Li; Huabin Xing; Keyan Ding; Qiang Zhang; Huajun Chen
>
> **备注:** ACL 2025 Main
>
> **摘要:** Molecular structure elucidation involves deducing a molecule's structure from various types of spectral data, which is crucial in chemical experimental analysis. While large language models (LLMs) have shown remarkable proficiency in analyzing and reasoning through complex tasks, they still encounter substantial challenges in molecular structure elucidation. We identify that these challenges largely stem from LLMs' limited grasp of specialized chemical knowledge. In this work, we introduce a Knowledge-enhanced reasoning framework for Molecular Structure Elucidation (K-MSE), leveraging Monte Carlo Tree Search for test-time scaling as a plugin. Specifically, we construct an external molecular substructure knowledge base to extend the LLMs' coverage of the chemical structure space. Furthermore, we design a specialized molecule-spectrum scorer to act as a reward model for the reasoning process, addressing the issue of inaccurate solution evaluation in LLMs. Experimental results show that our approach significantly boosts performance, particularly gaining more than 20% improvement on both GPT-4o-mini and GPT-4o. Our code is available at https://github.com/HICAI-ZJU/K-MSE.
>
---
#### [new 022] Boosting CTC-Based ASR Using LLM-Based Intermediate Loss Regularization
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升CTC模型的语义建模能力。通过引入LLM辅助的中间损失函数，增强语言建模同时保持高效解码。**

- **链接: [http://arxiv.org/pdf/2506.22846v1](http://arxiv.org/pdf/2506.22846v1)**

> **作者:** Duygu Altinok
>
> **备注:** This is the accepted version of an article accepted to the TSD 2025 conference, published in Springer Lecture Notes in Artificial Intelligence (LNAI). The final authenticated version is available online at SpringerLink
>
> **摘要:** End-to-end (E2E) automatic speech recognition (ASR) systems have revolutionized the field by integrating all components into a single neural network, with attention-based encoder-decoder models achieving state-of-the-art performance. However, their autoregressive decoding process limits inference speed, making them unsuitable for real-time applications. In contrast, CTC-based models offer faster, non-autoregressive decoding but struggle to model linguistic dependencies effectively. Addressing this challenge, we propose a novel auxiliary loss framework called Language-Aware Intermediate Loss (LAIL) to enhance CTC-based ASR using the linguistic knowledge of large language models (LLMs). By attaching connector layers to intermediate encoder layers, LAIL maps outputs to the embedding space of an LLM and computes a causal language modeling loss during training. This approach enhances linguistic modeling while preserving the computational efficiency of CTC decoding. Using the Conformer architecture and various LLaMA models, we demonstrate significant improvements in Word Error Rate (WER) on the LibriSpeech, TEDLIUM2, and WSJ corpora, achieving state-of-the-art performance for CTC-based ASR with minimal computational overhead.
>
---
#### [new 023] What to Keep and What to Drop: Adaptive Table Filtering Framework
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于表格问答任务，解决大表格输入受限问题。提出ATF框架，通过过滤冗余信息提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.23463v1](http://arxiv.org/pdf/2506.23463v1)**

> **作者:** Jang Won June
>
> **备注:** 26 pages, 9 figures
>
> **摘要:** Large language models (LLMs) for table-based reasoning often struggle with large tables due to input length limits. We propose ATF (Adaptive Table Filtering Framework), a modular and question-aware filtering pipeline that prunes uninformative columns and rows using LLM-generated column descriptions, clustering, and sparse-dense alignment scores. ATF integrates seamlessly with existing models (e.g., TAPAS, TAPEX) without retraining. Experiments show that ATF reduces table cells by ~70\%, boosting performance on out-of-domain TableQA tasks while causing slight performance drops on Table Fact Verification, where full-table context is more critical. These results highlight ATF's ability to adaptively balance informativeness and minimalism across tasks.
>
---
#### [new 024] ContextCache: Context-Aware Semantic Cache for Multi-Turn Queries in Large Language Models
- **分类: cs.CL; cs.DB**

- **简介: 该论文属于自然语言处理任务，旨在解决多轮对话中语义缓存的上下文感知问题。通过引入ContextCache系统，提升缓存准确性和效率。**

- **链接: [http://arxiv.org/pdf/2506.22791v1](http://arxiv.org/pdf/2506.22791v1)**

> **作者:** Jianxin Yan; Wangze Ni; Lei Chen; Xuemin Lin; Peng Cheng; Zhan Qin; Kui Ren
>
> **摘要:** Semantic caching significantly reduces computational costs and improves efficiency by storing and reusing large language model (LLM) responses. However, existing systems rely primarily on matching individual queries, lacking awareness of multi-turn dialogue contexts, which leads to incorrect cache hits when similar queries appear in different conversational settings. This demonstration introduces ContextCache, a context-aware semantic caching system for multi-turn dialogues. ContextCache employs a two-stage retrieval architecture that first executes vector-based retrieval on the current query to identify potential matches and then integrates current and historical dialogue representations through self-attention mechanisms for precise contextual matching. Evaluation of real-world conversations shows that ContextCache improves precision and recall compared to existing methods. Additionally, cached responses exhibit approximately 10 times lower latency than direct LLM invocation, enabling significant computational cost reductions for LLM conversational applications.
>
---
#### [new 025] Zero-Shot Contextual Embeddings via Offline Synthetic Corpus Generation
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于自然语言处理中的嵌入适配任务，旨在解决无目标语料或微调时的隐私与资源限制问题。通过生成合成语料，实现零样本上下文嵌入适配。**

- **链接: [http://arxiv.org/pdf/2506.23662v1](http://arxiv.org/pdf/2506.23662v1)**

> **作者:** Philip Lippmann; Jie Yang
>
> **摘要:** Context-aware embedding methods boost retrieval accuracy by conditioning on corpus statistics (e.g., term co-occurrence and topical patterns) extracted from neighboring documents. However, this context-aware approach requires access to the target corpus or requires domain-specific finetuning, posing practical barriers in privacy-sensitive or resource-constrained settings. We present ZEST, a zero-shot contextual adaptation framework that replaces real corpus access with a one-time offline synthesis of a compact proxy. Given only a handful exemplar documents representative of the general target domain, we use a multi-step hierarchical procedure to generate a synthetic context corpus of several hundred documents that aims to emulate key domain-specific distributions. At inference, the frozen context-aware encoder uses this proxy corpus -- without any finetuning or target corpus access -- to produce domain-adapted embeddings. Across the MTEB benchmark, ZEST's zero-shot synthetic context adaptation using only five example documents performs within 0.5% of models leveraging full target corpus access -- demonstrating remarkable efficacy without any retraining. ZEST thus provides a practical method for deploying high-performance, adaptable embeddings in constrained environments.
>
---
#### [new 026] Hallucination Detection with Small Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于问答系统中的幻觉检测任务，旨在解决LLM生成回答时出现的不准确问题。通过集成多个小模型验证回答，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.22486v1](http://arxiv.org/pdf/2506.22486v1)**

> **作者:** Ming Cheung
>
> **摘要:** Since the introduction of ChatGPT, large language models (LLMs) have demonstrated significant utility in various tasks, such as answering questions through retrieval-augmented generation. Context can be retrieved using a vectorized database, serving as a foundation for LLMs to generate responses. However, hallucinations in responses can undermine the reliability of LLMs in practical applications, and they are not easily detectable in the absence of ground truth, particularly in question-and-answer scenarios. This paper proposes a framework that integrates multiple small language models to verify responses generated by LLMs using the retrieved context from a vectorized database. By breaking down the responses into individual sentences and utilizing the probability of generating "Yes" tokens from the outputs of multiple models for a given set of questions, responses, and relevant context, hallucinations can be detected. The proposed framework is validated through experiments with real datasets comprising over 100 sets of questions, answers, and contexts, including responses with fully and partially correct sentences. The results demonstrate a 10\% improvement in F1 scores for detecting correct responses compared to hallucinations, indicating that multiple small language models can be effectively employed for answer verification, providing a scalable and efficient solution for both academic and practical applications.
>
---
#### [new 027] Benchmarking Deep Search over Heterogeneous Enterprise Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索任务，旨在解决企业数据中的深度搜索问题。通过构建基准测试，评估RAG系统在多源、复杂数据上的表现，并揭示检索是主要瓶颈。**

- **链接: [http://arxiv.org/pdf/2506.23139v1](http://arxiv.org/pdf/2506.23139v1)**

> **作者:** Prafulla Kumar Choubey; Xiangyu Peng; Shilpa Bhagavath; Kung-Hsiang Huang; Caiming Xiong; Chien-Sheng Wu
>
> **摘要:** We present a new benchmark for evaluating Deep Search--a realistic and complex form of retrieval-augmented generation (RAG) that requires source-aware, multi-hop reasoning over diverse, sparsed, but related sources. These include documents, meeting transcripts, Slack messages, GitHub, and URLs, which vary in structure and often contain human-to-human interactions. We build it using a synthetic data pipeline that simulates business workflows across product planning, development, and support stages, generating interconnected content with realistic noise and multi-hop questions with guaranteed ground-truth answers. We release our benchmark with both answerable and unanswerable queries, and retrieval pool of 39,190 enterprise artifacts, enabling fine-grained evaluation of long-context LLM and RAG systems. Our experiments reveal that even the best-performing agentic RAG methods achieve an average performance score of 32.96 on our benchmark. With further analysis, we highlight retrieval as the main bottleneck: existing methods struggle to conduct deep searches and retrieve all necessary evidence. Consequently, they often reason over partial context, leading to significant performance degradation.
>
---
#### [new 028] Decoding Memes: Benchmarking Narrative Role Classification across Multilingual and Multimodal Models
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于多模态情感分析任务，旨在识别网络迷因中的叙事角色（如英雄、反派等）。研究通过构建多样化数据集并评估多种模型，探索文化背景与跨语言挑战下的角色分类问题。**

- **链接: [http://arxiv.org/pdf/2506.23122v1](http://arxiv.org/pdf/2506.23122v1)**

> **作者:** Shivam Sharma; Tanmoy Chakraborty
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** This work investigates the challenging task of identifying narrative roles - Hero, Villain, Victim, and Other - in Internet memes, across three diverse test sets spanning English and code-mixed (English-Hindi) languages. Building on an annotated dataset originally skewed toward the 'Other' class, we explore a more balanced and linguistically diverse extension, originally introduced as part of the CLEF 2024 shared task. Comprehensive lexical and structural analyses highlight the nuanced, culture-specific, and context-rich language used in real memes, in contrast to synthetically curated hateful content, which exhibits explicit and repetitive lexical markers. To benchmark the role detection task, we evaluate a wide spectrum of models, including fine-tuned multilingual transformers, sentiment and abuse-aware classifiers, instruction-tuned LLMs, and multimodal vision-language models. Performance is assessed under zero-shot settings using precision, recall, and F1 metrics. While larger models like DeBERTa-v3 and Qwen2.5-VL demonstrate notable gains, results reveal consistent challenges in reliably identifying the 'Victim' class and generalising across cultural and code-mixed content. We also explore prompt design strategies to guide multimodal models and find that hybrid prompts incorporating structured instructions and role definitions offer marginal yet consistent improvements. Our findings underscore the importance of cultural grounding, prompt engineering, and multimodal reasoning in modelling subtle narrative framings in visual-textual content.
>
---
#### [new 029] The Translation Barrier Hypothesis: Multilingual Generation with Large Language Models Suffers from Implicit Translation Failure
- **分类: cs.CL**

- **简介: 该论文研究多语言生成任务，探讨低资源语言生成质量差的问题。提出翻译障碍假设，通过实验验证模型在翻译阶段的失败是主要原因。**

- **链接: [http://arxiv.org/pdf/2506.22724v1](http://arxiv.org/pdf/2506.22724v1)**

> **作者:** Niyati Bafna; Tianjian Li; Kenton Murray; David R. Mortensen; David Yarowsky; Hale Sirin; Daniel Khashabi
>
> **备注:** 23 pages incl. appendix
>
> **摘要:** Multilingual generation with large language models (LLMs) is often of poor quality for mid- to low-resource languages. Building on insights from interpretability, we demonstrate the existence of an implicit task-solving-->translation pipeline for generation, whereby the model first solves the required task in a largely target-language-agnostic manner, and subsequently translates answer concepts into the intended target language. We hypothesize that the failure of the translation stage is an important culprit for the observed low quality of final outputs, and formalize this as the translation barrier hypothesis. We test this hypothesis for a word translation task across 108 language pairs, using logit lens to observe model processing in intermediate layers. We find that a significant portion of overall failures indeed stems from translation failure, or the model's inability to translate correctly solved intermediate concepts into the target language. This is especially true for low-resource target languages. Our results highlight an important hurdle for end-to-end multilingual generation, and lend guiding insights for future work seeking to improve multilinguality in LLMs.
>
---
#### [new 030] From Individuals to Interactions: Benchmarking Gender Bias in Multimodal Large Language Models from the Lens of Social Relationship
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态语言模型性别偏见研究任务，旨在解决交互中隐性性别偏见的评估问题。通过构建Genres基准，分析双角色互动中的关系性偏见。**

- **链接: [http://arxiv.org/pdf/2506.23101v1](http://arxiv.org/pdf/2506.23101v1)**

> **作者:** Yue Xu; Wenjie Wang
>
> **摘要:** Multimodal large language models (MLLMs) have shown impressive capabilities across tasks involving both visual and textual modalities. However, growing concerns remain about their potential to encode and amplify gender bias, particularly in socially sensitive applications. Existing benchmarks predominantly evaluate bias in isolated scenarios, overlooking how bias may emerge subtly through interpersonal interactions. We fill this gap by going beyond single-entity evaluation and instead focusing on a deeper examination of relational and contextual gender bias in dual-individual interactions. We introduce Genres, a novel benchmark designed to evaluate gender bias in MLLMs through the lens of social relationships in generated narratives. Genres assesses gender bias through a dual-character profile and narrative generation task that captures rich interpersonal dynamics and supports a fine-grained bias evaluation suite across multiple dimensions. Experiments on both open- and closed-source MLLMs reveal persistent, context-sensitive gender biases that are not evident in single-character settings. Our findings underscore the importance of relationship-aware benchmarks for diagnosing subtle, interaction-driven gender bias in MLLMs and provide actionable insights for future bias mitigation.
>
---
#### [new 031] Objective-Free Local Learning and Emergent Language Structure in Thinking Machines
- **分类: cs.CL; cs.AI; cs.LG; q-bio.NC**

- **简介: 该论文属于自然语言处理任务，旨在解决语言建模与符号结构生成问题。提出一种无目标的局部学习框架，通过层次霍普菲尔记忆链实现语言结构的自组织生成。**

- **链接: [http://arxiv.org/pdf/2506.23293v1](http://arxiv.org/pdf/2506.23293v1)**

> **作者:** P. Myles Eugenio
>
> **备注:** 22 pages, 7 figures
>
> **摘要:** We present a neuro-symbolic framework for generative language modeling based on local, event-driven emergent learning. At its core is a hierarchical Hopfield memory chain acting as a compositional short-term memory and dynamic tokenizer (retokenizer). Rather than relying on predefined tokens or supervision, the model builds structure from scratch, learning symbol sequences as multi-scale representations. It constructs projection tensors that bind co-occurring features into hierarchical tokens, introducing redundancy (i.e an emergent gauge structure) and enabling compression of local activations into long-range dependencies. Curiously, we find that the retokenizer can filter natural language patterns from noise, generating synthetic languages with coherent internal morphology -- quantifiably the same as human language. Language is learned in a local (Hebbian) fashion, where model constraints dictate allowed emergent structure, and new information is retained in alignment with this structure. The absence of a global objective enables a form of plasticity not found in conventional language models, allowing the system to generalize beyond its initial inference class -- even without explicit data. We demonstrate that briefly activating a new neuron during inference binds distributed multi-scale token features into a symbolic embedding. These emergent embedding neurons act as long-term memory and support a key-value mechanism for compositional inference and generalization. This architecture provides a methodological foundation for studying how symbolic structure can emerge from local neural learning. It offers a new pathway for building scalable, interpretable neuro-symbolic systems -- where tokens, grammar, and reasoning arise as compressed memory traces within a Hopfield hierarchy. This approach advances the development of neuromorphic architectures for generative language models.
>
---
#### [new 032] Unveiling Decision-Making in LLMs for Text Classification : Extraction of influential and interpretable concepts with Sparse Autoencoders
- **分类: cs.CL**

- **简介: 该论文研究文本分类中的模型决策解释问题，提出一种基于稀疏自编码器的架构，提升特征的可解释性和因果性。**

- **链接: [http://arxiv.org/pdf/2506.23951v1](http://arxiv.org/pdf/2506.23951v1)**

> **作者:** Mathis Le Bail; Jérémie Dentan; Davide Buscaldi; Sonia Vanier
>
> **摘要:** Sparse Autoencoders (SAEs) have been successfully used to probe Large Language Models (LLMs) and extract interpretable concepts from their internal representations. These concepts are linear combinations of neuron activations that correspond to human-interpretable features. In this paper, we investigate the effectiveness of SAE-based explainability approaches for sentence classification, a domain where such methods have not been extensively explored. We present a novel SAE-based architecture tailored for text classification, leveraging a specialized classifier head and incorporating an activation rate sparsity loss. We benchmark this architecture against established methods such as ConceptShap, Independent Component Analysis, and other SAE-based concept extraction techniques. Our evaluation covers two classification benchmarks and four fine-tuned LLMs from the Pythia family. We further enrich our analysis with two novel metrics for measuring the precision of concept-based explanations, using an external sentence encoder. Our empirical results show that our architecture improves both the causality and interpretability of the extracted features.
>
---
#### [new 033] TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在量化分析微调对大语言模型个体输出的影响。提出TuCo方法，分解模型组件并评估其贡献。**

- **链接: [http://arxiv.org/pdf/2506.23423v1](http://arxiv.org/pdf/2506.23423v1)**

> **作者:** Felipe Nuti; Tim Franzmeyer; João Henriques
>
> **备注:** ICML 2025
>
> **摘要:** Past work has studied the effects of fine-tuning on large language models' (LLMs) overall performance on certain tasks. However, a quantitative and systematic method for analyzing its effect on individual outputs is still lacking. Here, we propose a new method for measuring the contribution that fine-tuning makes to individual LLM responses, assuming access to the original pre-trained model. Our method tracks the model's intermediate hidden states, providing a more fine-grained insight into the effects of fine-tuning than a simple comparison of final outputs from pre-trained and fine-tuned models. We introduce and theoretically analyze an exact decomposition of any fine-tuned LLM into a pre-training component and a fine-tuning component. Empirically, we find that model behavior and performance can be steered by up- or down-scaling the fine-tuning component during the forward pass. Motivated by this finding and our theoretical analysis, we define the Tuning Contribution (TuCo) as the ratio of the magnitudes of the fine-tuning component to the pre-training component. We observe that three prominent adversarial attacks on LLMs circumvent safety measures in a way that reduces TuCo, and that TuCo is consistently lower on prompts where these attacks succeed compared to those where they do not. This suggests that attenuating the effect of fine-tuning on model outputs plays a role in the success of such attacks. In summary, TuCo enables the quantitative study of how fine-tuning influences model behavior and safety, and vice versa.
>
---
#### [new 034] MisinfoTeleGraph: Network-driven Misinformation Detection for German Telegram Messages
- **分类: cs.CL**

- **简介: 该论文属于谣言检测任务，针对德语Telegram平台上的虚假信息问题，构建了首个相关图数据集，并通过图神经网络进行检测研究。**

- **链接: [http://arxiv.org/pdf/2506.22529v1](http://arxiv.org/pdf/2506.22529v1)**

> **作者:** Lu Kalkbrenner; Veronika Solopova; Steffen Zeiler; Robert Nickel; Dorothea Kolossa
>
> **摘要:** Connectivity and message propagation are central, yet often underutilized, sources of information in misinformation detection -- especially on poorly moderated platforms such as Telegram, which has become a critical channel for misinformation dissemination, namely in the German electoral context. In this paper, we introduce Misinfo-TeleGraph, the first German-language Telegram-based graph dataset for misinformation detection. It includes over 5 million messages from public channels, enriched with metadata, channel relationships, and both weak and strong labels. These labels are derived via semantic similarity to fact-checks and news articles using M3-embeddings, as well as manual annotation. To establish reproducible baselines, we evaluate both text-only models and graph neural networks (GNNs) that incorporate message forwarding as a network structure. Our results show that GraphSAGE with LSTM aggregation significantly outperforms text-only baselines in terms of Matthews Correlation Coefficient (MCC) and F1-score. We further evaluate the impact of subscribers, view counts, and automatically versus human-created labels on performance, and highlight both the potential and challenges of weak supervision in this domain. This work provides a reproducible benchmark and open dataset for future research on misinformation detection in German-language Telegram networks and other low-moderation social platforms.
>
---
#### [new 035] Hierarchical Memory Organization for Wikipedia Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自动生成维基百科文章的任务，旨在解决信息准确性和结构化问题。提出MOG框架，通过分层记忆组织提升文章的可信度和可验证性。**

- **链接: [http://arxiv.org/pdf/2506.23393v1](http://arxiv.org/pdf/2506.23393v1)**

> **作者:** Eugene J. Yu; Dawei Zhu; Yifan Song; Xiangyu Wong; Jiebin Zhang; Wenxuan Shi; Xiaoguang Li; Qun Liu; Sujian Li
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** Generating Wikipedia articles autonomously is a challenging task requiring the integration of accurate, comprehensive, and well-structured information from diverse sources. This paper introduces the Memory Organization-based Generation (MOG) framework, a novel approach to address these challenges by leveraging a hierarchical memory architecture. MOG extracts fine-grained memory units from web documents, recursively organizes them into a Wikipedia-style hierarchical structure, and uses this structure to guide the generation process. This ensures alignment between memory and the article outline, improving both informativeness and verifiability while minimizing hallucinations. Additionally, a citation module is implemented to enhance traceability by linking every generated sentence to specific memory units. Evaluations on our newly created WikiStart dataset demonstrate that MOG outperforms baseline methods in producing informative and reliable articles, making it particularly robust in real-world scenarios.
>
---
#### [new 036] Learning-to-Context Slope: Evaluating In-Context Learning Effectiveness Beyond Performance Illusions
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决ICL效果评估不准确的问题，提出LCS指标以更可靠地衡量ICL有效性。**

- **链接: [http://arxiv.org/pdf/2506.23146v1](http://arxiv.org/pdf/2506.23146v1)**

> **作者:** Dingzriui Wang; Xuanliang Zhang; Keyan Xu; Qingfu Zhu; Wanxiang Che; Yang Deng
>
> **摘要:** In-context learning (ICL) has emerged as an effective approach to enhance the performance of large language models (LLMs). However, its effectiveness varies significantly across models and tasks, posing challenges for practitioners to determine when ICL reliably improves performance. Current evaluation approaches, reliant on performance change after applying ICL, suffer from low reliability, poor attribution, and impracticality in data-insufficient scenarios. We propose the Learning-to-Context Slope (LCS), a novel metric that quantifies ICL effectiveness by modeling the slope between learning gain (loss decrease from demonstrations) and contextual relevance (demonstration-input relevance). LCS addresses key limitations of performance-based metrics: (1) it captures continuous loss changes even when outputs are incorrect, improving reliability; (2) its formulation attributes ICL failures to weak contextual alignment (inability to adapt inputs to demonstrations) or strong output calibration (self-verification of correctness); and (3) it minimizes reliance on labeled data via synthetic evaluation. Extensive experiments demonstrate that LCS strongly correlates with performance improvements in labeled settings and reliably reflects true effectiveness in biased or data-scarce scenarios. Further analysis reveals actionable thresholds for LCS and identifies model capabilities critical to ICL success.
>
---
#### [new 037] Agent-to-Agent Theory of Mind: Testing Interlocutor Awareness among Large Language Models
- **分类: cs.CL; cs.AI; cs.CY; cs.MA**

- **简介: 该论文属于自然语言处理任务，研究大语言模型对对话伙伴的识别能力，解决其在多智能体系统中的协作与安全问题，通过实验分析并验证了模型的互为主体意识。**

- **链接: [http://arxiv.org/pdf/2506.22957v1](http://arxiv.org/pdf/2506.22957v1)**

> **作者:** Younwoo Choi; Changling Li; Yongjin Yang; Zhijing Jin
>
> **摘要:** As large language models (LLMs) are increasingly integrated into multi-agent and human-AI systems, understanding their awareness of both self-context and conversational partners is essential for ensuring reliable performance and robust safety. While prior work has extensively studied situational awareness which refers to an LLM's ability to recognize its operating phase and constraints, it has largely overlooked the complementary capacity to identify and adapt to the identity and characteristics of a dialogue partner. In this paper, we formalize this latter capability as interlocutor awareness and present the first systematic evaluation of its emergence in contemporary LLMs. We examine interlocutor inference across three dimensions-reasoning patterns, linguistic style, and alignment preferences-and show that LLMs reliably identify same-family peers and certain prominent model families, such as GPT and Claude. To demonstrate its practical significance, we develop three case studies in which interlocutor awareness both enhances multi-LLM collaboration through prompt adaptation and introduces new alignment and safety vulnerabilities, including reward-hacking behaviors and increased jailbreak susceptibility. Our findings highlight the dual promise and peril of identity-sensitive behavior in LLMs, underscoring the need for further understanding of interlocutor awareness and new safeguards in multi-agent deployments. Our code is open-sourced at https://github.com/younwoochoi/InterlocutorAwarenessLLM.
>
---
#### [new 038] TaP: A Taxonomy-Guided Framework for Automated and Scalable Preference Data Generation
- **分类: cs.CL**

- **简介: 该论文提出TaP框架，用于自动化生成多语言偏好数据集，解决数据稀缺与质量不足问题，提升大语言模型的指令遵循能力。**

- **链接: [http://arxiv.org/pdf/2506.23979v1](http://arxiv.org/pdf/2506.23979v1)**

> **作者:** Renren Jin; Tianhao Shen; Xinwei Wu; Dan Shi; Haoran Sun; Wuwei Huang; Quandong Wang; Wei Liu; Jian Luan; Bin Wang; Deyi Xiong
>
> **备注:** 33 pages, 15 tables, 11 figures
>
> **摘要:** Conducting supervised fine-tuning and preference fine-tuning on large language models (LLMs) requires high-quality datasets to improve their ability to follow instructions and align with human preferences and values. However, constructing such datasets is resource-intensive, and most available datasets for supervised and preference fine-tuning are in English. To address these challenges, we propose the \underline{\textbf{Ta}}xonomy-Guided \underline{\textbf{P}}reference Data Generation (TaP) framework, which facilitates automated and scalable construction of preference datasets across various languages. TaP is grounded in a structured taxonomy that allows fine-grained control over dataset composition, thereby ensuring both diversity and comprehensive coverage. We employ TaP-generated datasets to perform supervised and preference fine-tuning on various LLMs. Experimental results demonstrate that LLMs trained on TaP-generated datasets outperform those trained on existing open-source datasets. Remarkably, LLMs trained on TaP-generated datasets surpass the performance of those trained on an open-source dataset that is 180 times larger.
>
---
#### [new 039] Towards Text-free Graph Foundation Models: Rethinking Multi-Domain Graph Contrastive Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于图预训练任务，旨在解决多领域图对比学习中的领域差异问题。提出MDGCL框架，通过识别领域差异和引入领域注意力机制，提升跨领域知识迁移效果。**

- **链接: [http://arxiv.org/pdf/2506.22510v1](http://arxiv.org/pdf/2506.22510v1)**

> **作者:** Zihao Zhao; Xinlong Zhai; Jinyu Yang; Chuan Shi
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Foundation models have achieved great success in natural language processing (NLP) and computer vision (CV). Their success largely stems from the ability to integrate multi-domain knowledge in pre-training and transfer it to target domains. Considering graph data, especially graphs without textual features, is ubiquitous in real-world applications such as social networks and recommendation systems, some researchers have attempted to extend this paradigm to the graph field, aiming to construct graph foundation models. However, unlike CV and NLP, there are huge gaps among the semantics and properties of graphs in different domains, while current works still adopt traditional contrastive pre-training strategies designed in the single-domain scenario, which regard contrastive samples from different domains as equivalent. From experimental investigations, we discovered that inherent domain-specific differences prevent these strategies from effectively absorbing knowledge from different domains to generate informative representations. In this paper, we propose a novel multi-domain pre-training and cross-domain transfer framework, namely MDGCL.In the pre-training stage, we design a contrastive learning strategy to substantially recognize and capture domain differences, and introduce domain tokens to encode domain-level global information. In the downstream stage, we introduce a domain attention mechanism to enable fine-grained domain knowledge transfer. Extensive experiments on five benchmark datasets have demonstrated that our method outperforms state-of-the-art significantly, with the maximum improvement of 19.33\% on accuracy and 19.13\% on Macro-F1 score.
>
---
#### [new 040] Knowledge Augmented Finetuning Matters in both RAG and Agent Based Dialog Systems
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在提升知识密集场景下的准确性。通过知识增强微调（KAFT）改进RAG和代理系统，优于传统提示方法。**

- **链接: [http://arxiv.org/pdf/2506.22852v1](http://arxiv.org/pdf/2506.22852v1)**

> **作者:** Yucheng Cai; Yuxuan Wu; Yi Huang; Junlan Feng; Zhijian Ou
>
> **摘要:** Large language models (LLMs) have recently been applied to dialog systems. Despite making progress, LLMs are prone to errors in knowledge-intensive scenarios. Recently, approaches based on retrieval augmented generation (RAG) and agent have emerged to improve the factual accuracy by enhancing the LLMs with knowledge retrieved from external knowledge bases (KBs). This is mostly implemented by prompting the LLMs with instructions, examples and the retrieved knowledge. However, LLMs may have difficulty using the retrieved knowledge effectively for response generation, because they are not well trained to do such generation for specific domains. To mitigate this problem, we propose to finetune the LLMs in the RAG-based and agent-based systems with domain-specific data, together with domain-specific external knowledge, which is called knowledge augmented finetuning (KAFT). We base our study on the MobileCS2 dataset, a real-life customer service dialog dataset that features intensive knowledge interactions, to systematically compare the prompting and KAFT techniques in the RAG-based and agent-based systems. Experiment results show that KAFT substantially surpasses prompting in both RAG and agent systems, particularly in terms of factual accuracy. To the best of our knowledge, this paper represents the first solid empirical work to investigate the KAFT idea.
>
---
#### [new 041] Can "consciousness" be observed from large language model (LLM) internal states? Dissecting LLM representations obtained from Theory of Mind test with Integrated Information Theory and Span Representation analysis
- **分类: cs.CL; cs.AI; cs.NE; q-bio.NC**

- **简介: 该论文属于AI认知研究任务，旨在探讨LLM内部状态是否能体现“意识”现象。通过IIT和Span分析，研究LLM在ToM测试中的表示差异，但未发现显著意识指标。**

- **链接: [http://arxiv.org/pdf/2506.22516v1](http://arxiv.org/pdf/2506.22516v1)**

> **作者:** Jingkai Li
>
> **备注:** Published as a journal paper at: https://doi.org/10.1016/j.nlp.2025.100163
>
> **摘要:** Integrated Information Theory (IIT) provides a quantitative framework for explaining consciousness phenomenon, positing that conscious systems comprise elements integrated through causal properties. We apply IIT 3.0 and 4.0 -- the latest iterations of this framework -- to sequences of Large Language Model (LLM) representations, analyzing data derived from existing Theory of Mind (ToM) test results. Our study systematically investigates whether the differences of ToM test performances, when presented in the LLM representations, can be revealed by IIT estimates, i.e., $\Phi^{\max}$ (IIT 3.0), $\Phi$ (IIT 4.0), Conceptual Information (IIT 3.0), and $\Phi$-structure (IIT 4.0). Furthermore, we compare these metrics with the Span Representations independent of any estimate for consciousness. This additional effort aims to differentiate between potential "consciousness" phenomena and inherent separations within LLM representational space. We conduct comprehensive experiments examining variations across LLM transformer layers and linguistic spans from stimuli. Our results suggest that sequences of contemporary Transformer-based LLM representations lack statistically significant indicators of observed "consciousness" phenomena but exhibit intriguing patterns under $\textit{spatio}$-permutational analyses. The Appendix and code are available as Supplementary Materials at: https://doi.org/10.1016/j.nlp.2025.100163.
>
---
#### [new 042] Weak-to-Strong GraphRAG: Aligning Weak Retrievers with Large Language Models for Graph-based Retrieval Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识增强生成任务，解决图RAG中弱检索器带来的噪声和信息混乱问题。通过引入LLM反馈和结构重组织模块提升检索质量与生成效果。**

- **链接: [http://arxiv.org/pdf/2506.22518v1](http://arxiv.org/pdf/2506.22518v1)**

> **作者:** Deyu Zou; Yongqiang Chen; Mufei Li; Siqi Miao; Chenxi Liu; Bo Han; James Cheng; Pan Li
>
> **摘要:** Graph-based retrieval-augmented generation (RAG) enables large language models (LLMs) to ground responses with structured external knowledge from up-to-date knowledge graphs (KGs) and reduce hallucinations. However, LLMs often rely on a weak retriever in graph-based RAG: I) Due to the lack of ground truth, the retriever is often trained on weak supervision, which often introduces spurious signals to the LLMs. II) Due to the abstraction of graph data, the retrieved knowledge is often presented in unorganized forms. To mitigate the issue, we present Refined Graph-based RAG (ReG) to align weak retrievers to LLMs for graph-based RAG. Specifically, ReG incorporates LLM feedback to get rid of spurious signals and improve the quality of the supervision. Meanwhile, ReG introduces a structure-aware reorganization module to refactor the retrieval results into logically coherent evidence chains. Experiments on prominent benchmarks demonstrate that ReG significantly and consistently brings improvements across different LLM backbones by up to 10%. The improved supervision quality enables ReG to match the state-of-the-art performance with 5% training data and to transfer to out-of-distribution KGs. Notably, when adopted to reasoning-based LLMs, ReG reduces the reasoning token cost by up to 30% and improves the performance by up to 4%.
>
---
#### [new 043] On the Generalizability of "Competition of Mechanisms: Tracing How Language Models Handle Facts and Counterfactuals"
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，研究语言模型在处理事实与反事实信息时的机制竞争。通过实验验证了原有结论的泛化性，并探讨了模型规模、提示结构和领域对结果的影响。**

- **链接: [http://arxiv.org/pdf/2506.22977v1](http://arxiv.org/pdf/2506.22977v1)**

> **作者:** Asen Dotsinski; Udit Thakur; Marko Ivanov; Mohammad Hafeez Khan; Maria Heuss
>
> **备注:** 22 pages, 25 figures. For an interactive dashboard with all figures, see https://comp-mech-generalizability.streamlit.app/ . For the accompanying code, see https://github.com/asendotsinski/comp-mech-generalizability . To be published in proceedings of the 2025 Machine Learning Reproducibility Challenge
>
> **摘要:** We present a reproduction study of "Competition of Mechanisms: Tracing How Language Models Handle Facts and Counterfactuals" (Ortu et al., 2024), which investigates competition of mechanisms in language models between factual recall and counterfactual in-context repetition. Our study successfully reproduces their primary findings regarding the localization of factual and counterfactual information, the dominance of attention blocks in mechanism competition, and the specialization of attention heads in handling competing information. We reproduce their results on both GPT-2 (Radford et al., 2019) and Pythia 6.9B (Biderman et al., 2023). We extend their work in three significant directions. First, we explore the generalizability of these findings to even larger models by replicating the experiments on Llama 3.1 8B (Grattafiori et al., 2024), discovering greatly reduced attention head specialization. Second, we investigate the impact of prompt structure by introducing variations where we avoid repeating the counterfactual statement verbatim or we change the premise word, observing a marked decrease in the logit for the counterfactual token. Finally, we test the validity of the authors' claims for prompts of specific domains, discovering that certain categories of prompts skew the results by providing the factual prediction token as part of the subject of the sentence. Overall, we find that the attention head ablation proposed in Ortu et al. (2024) is ineffective for domains that are underrepresented in their dataset, and that the effectiveness varies based on model architecture, prompt structure, domain and task.
>
---
#### [new 044] IMPACT: Inflectional Morphology Probes Across Complex Typologies
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的形态学评估任务，旨在检验大语言模型对多语言复杂形态结构的理解能力。通过构建IMPACT框架，测试模型在五种形态丰富的语言中的表现。**

- **链接: [http://arxiv.org/pdf/2506.23929v1](http://arxiv.org/pdf/2506.23929v1)**

> **作者:** Mohammed J. Saeed; Tommi Vehvilainen; Evgeny Fedoseev; Sevil Caliskan; Tatiana Vodolazova
>
> **摘要:** Large Language Models (LLMs) have shown significant progress on various multilingual benchmarks and are increasingly used to generate and evaluate text in non-English languages. However, while they may produce fluent outputs, it remains unclear to what extent these models truly grasp the underlying linguistic complexity of those languages, particularly in morphology. To investigate this, we introduce IMPACT, a synthetically generated evaluation framework focused on inflectional morphology, which we publicly release, designed to evaluate LLM performance across five morphologically rich languages: Arabic, Russian, Finnish, Turkish, and Hebrew. IMPACT includes unit-test-style cases covering both shared and language-specific phenomena, from basic verb inflections (e.g., tense, number, gender) to unique features like Arabic's reverse gender agreement and vowel harmony in Finnish and Turkish. We assess eight multilingual LLMs that, despite strong English performance, struggle with other languages and uncommon morphological patterns, especially when judging ungrammatical examples. We also show that Chain of Thought and Thinking Models can degrade performance. Our work exposes gaps in LLMs' handling of linguistic complexity, pointing to clear room for improvement. To support further research, we publicly release the IMPACT framework.
>
---
#### [new 045] EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于图像描述评估任务，旨在解决现有评估指标解释不规范的问题。提出EXPERT模型，通过结构化标准提升解释质量。**

- **链接: [http://arxiv.org/pdf/2506.24016v1](http://arxiv.org/pdf/2506.24016v1)**

> **作者:** Hyunjong Kim; Sangyeop Kim; Jongheon Jeong; Yeongjae Cho; Sungzoon Cho
>
> **备注:** Accepted at ACL 2025 Findings
>
> **摘要:** Recent advances in large language models and vision-language models have led to growing interest in explainable evaluation metrics for image captioning. However, these metrics generate explanations without standardized criteria, and the overall quality of the generated explanations remains unverified. In this paper, we propose EXPERT, a reference-free evaluation metric that provides structured explanations based on three fundamental criteria: fluency, relevance, and descriptiveness. By constructing large-scale datasets of high-quality structured explanations, we develop a two-stage evaluation template to effectively supervise a vision-language model for both scoring and explanation generation. EXPERT achieves state-of-the-art results on benchmark datasets while providing significantly higher-quality explanations than existing metrics, as validated through comprehensive human evaluation. Our code and datasets are available at https://github.com/hjkim811/EXPERT.
>
---
#### [new 046] Perspective Dial: Measuring Perspective of Text and Guiding LLM Outputs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM输出偏见与视角量化问题。提出Perspective-Dial方法，通过度量空间和系统提示工程控制输出视角。**

- **链接: [http://arxiv.org/pdf/2506.23377v1](http://arxiv.org/pdf/2506.23377v1)**

> **作者:** Taejin Kim; Siun-Chuon Mau; Konrad Vesey
>
> **备注:** 7 pages, 5 main pages of text, 5 figures, 2 tables. Research work performed at CACI INTL INC
>
> **摘要:** Large language models (LLMs) are used in a variety of mission-critical roles. Due to the rapidly developing nature of LLMs, there is a lack of quantifiable understanding of the bias and perspective associated with LLM output. Inspired by this need, this paper considers the broader issue of perspective or viewpoint of general text and perspective control of large-language model (LLM) output. Perspective-Dial consists of two main components: a (1) metric space, dubbed Perspective Space, that enables quantitative measurements of different perspectives regarding a topic, and the use of (2) Systematic Prompt Engineering that utilizes greedy-coordinate descent to control LLM output perspective based on measurement feedback from the Perspective Space. The empirical nature of the approach allows progress to side step a principled understanding of perspective or bias -- effectively quantifying and adjusting outputs for a variety of topics. Potential applications include detection, tracking and mitigation of LLM bias, narrative detection, sense making and tracking in public discourse, and debate bot advocating given perspective.
>
---
#### [new 047] Auto-TA: Towards Scalable Automated Thematic Analysis (TA) via Multi-Agent Large Language Models with Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的主题分析任务，旨在解决传统人工主题分析耗时且难以扩展的问题。研究提出一种基于多智能体大语言模型和强化学习的自动化主题分析系统。**

- **链接: [http://arxiv.org/pdf/2506.23998v1](http://arxiv.org/pdf/2506.23998v1)**

> **作者:** Seungjun Yi; Joakim Nguyen; Huimin Xu; Terence Lim; Andrew Well; Mia Markey; Ying Ding
>
> **备注:** Presented at ACL 2025 SRW
>
> **摘要:** Congenital heart disease (CHD) presents complex, lifelong challenges often underrepresented in traditional clinical metrics. While unstructured narratives offer rich insights into patient and caregiver experiences, manual thematic analysis (TA) remains labor-intensive and unscalable. We propose a fully automated large language model (LLM) pipeline that performs end-to-end TA on clinical narratives, which eliminates the need for manual coding or full transcript review. Our system employs a novel multi-agent framework, where specialized LLM agents assume roles to enhance theme quality and alignment with human analysis. To further improve thematic relevance, we optionally integrate reinforcement learning from human feedback (RLHF). This supports scalable, patient-centered analysis of large qualitative datasets and allows LLMs to be fine-tuned for specific clinical contexts.
>
---
#### [new 048] VOCABTRIM: Vocabulary Pruning for Efficient Speculative Decoding in LLMs
- **分类: cs.CL**

- **简介: 该论文属于语言模型优化任务，旨在解决Speculative Decoding中因大词表导致的推理延迟问题。通过VocabTrim技术缩减词表，提升边缘设备上的生成速度。**

- **链接: [http://arxiv.org/pdf/2506.22694v1](http://arxiv.org/pdf/2506.22694v1)**

> **作者:** Raghavv Goel; Sudhanshu Agrawal; Mukul Gagrani; Junyoung Park; Yifan Zao; He Zhang; Tian Liu; Yiping Yang; Xin Yuan; Jiuyan Lu; Chris Lott; Mingu Lee
>
> **备注:** 7 pages, 4 figures, 5 tables, accepted at ICML 2025 workshop on Efficient Systems for Foundational Models
>
> **摘要:** In this paper, we introduce a simple training-free technique to improve the performance of drafter-based speculative decoding (SpD) methods that incorporates language modeling head (LM head) during drafting process. A drafter-based speculative decoding leverages one or more smaller language models, a.k.a. drafters or draft models, to sample a draft sequence or tree consisting of multiple tokens, followed by verification by a base LLM, a target model, accepting a subset as its valid generation. As it is usually considered that the speculative decoding requires one-to-one mapping between vocabularies of the target model and the draft model, it has been natural to share the vocabulary between them, or even share the LM head as in EAGLE or Medusa. We first identify that this draft token sampling scheme inherently contains an unnecessary inference overhead in drafting, especially for some target LLMs with very large vocabularies. Then, we propose a simple technique, VocabTrim, to mitigate the drafting overhead to improve the generation speed in memory-bound environment. VocabTrim reconstructs the drafter LM head to contain only a limited set of tokens, selected by the most frequently sampled from the vocabulary of the target model. While limiting the vocabulary in drafting slightly degrades the acceptance rate, it significantly reduces the drafting latency in memory-bound process which is often the case on edge devices, resulting in higher memory-bound speed up (MBSU). We show that our method can boost the memory-bound speed-up for Llama-3 models on Spec-Bench, specifically by 16% for Llama-3.2-3B-Instruct.
>
---
#### [new 049] MedEthicsQA: A Comprehensive Question Answering Benchmark for Medical Ethics Evaluation of LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗伦理评估任务，旨在解决MedLLMs伦理安全不足的问题，构建了包含大量问题的基准数据集以评估其伦理表现。**

- **链接: [http://arxiv.org/pdf/2506.22808v1](http://arxiv.org/pdf/2506.22808v1)**

> **作者:** Jianhui Wei; Zijie Meng; Zikai Xiao; Tianxiang Hu; Yang Feng; Zhijie Zhou; Jian Wu; Zuozhu Liu
>
> **备注:** 20 pages
>
> **摘要:** While Medical Large Language Models (MedLLMs) have demonstrated remarkable potential in clinical tasks, their ethical safety remains insufficiently explored. This paper introduces $\textbf{MedEthicsQA}$, a comprehensive benchmark comprising $\textbf{5,623}$ multiple-choice questions and $\textbf{5,351}$ open-ended questions for evaluation of medical ethics in LLMs. We systematically establish a hierarchical taxonomy integrating global medical ethical standards. The benchmark encompasses widely used medical datasets, authoritative question banks, and scenarios derived from PubMed literature. Rigorous quality control involving multi-stage filtering and multi-faceted expert validation ensures the reliability of the dataset with a low error rate ($2.72\%$). Evaluation of state-of-the-art MedLLMs exhibit declined performance in answering medical ethics questions compared to their foundation counterparts, elucidating the deficiencies of medical ethics alignment. The dataset, registered under CC BY-NC 4.0 license, is available at https://github.com/JianhuiWei7/MedEthicsQA.
>
---
#### [new 050] Information Loss in LLMs' Multilingual Translation: The Role of Training Data, Language Proximity, and Language Family
- **分类: cs.CL**

- **简介: 该论文属于多语言翻译任务，研究训练数据、语言接近度和语系对信息损失的影响。通过实验分析不同因素对翻译质量的作用。**

- **链接: [http://arxiv.org/pdf/2506.23340v1](http://arxiv.org/pdf/2506.23340v1)**

> **作者:** Yumeng Lin; Xufeng Duan; David Haslett; Yige Chen; Zhenguang G. Cai
>
> **摘要:** Large language models have achieved impressive progress in multilingual translation, yet they continue to face challenges with certain language pairs-particularly those with limited training data or significant linguistic divergence from English. This study systematically investigates how training data, language proximity, and language family affect information loss in multilingual translation. We evaluate two large language models, GPT-4 and Llama 2, by performing round-trip translations. Translation quality was assessed using BLEU scores and BERT similarity metrics. Our results reveal a robust interaction between training data size and language distance: while abundant training data can mitigate the effects of linguistic divergence, languages structurally closer to English consistently yield higher translation quality in low-resource conditions. Among various distance metrics, orthographic, phylogenetic, syntactic, and geographical distances emerge as strong predictors of translation performance. Language family also exerts an independent influence. These findings contribute to a deeper understanding of the linguistic constraints shaping multilingual translation in large language models, emphasizing that translation quality is shaped not only by data volume but also by structural and typological relationships between languages.
>
---
#### [new 051] Unleashing Embodied Task Planning Ability in LLMs via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于具身任务规划任务，旨在解决LLMs在动态环境中难以学习因果关系的问题。通过强化学习框架Embodied Planner-R1，提升模型的交互与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.23127v1](http://arxiv.org/pdf/2506.23127v1)**

> **作者:** Zhaoye Fei; Li Ji; Siyin Wang; Junhao Shi; Jingjing Gong; Xipeng Qiu
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they face significant challenges in embodied task planning scenarios that require continuous environmental understanding and action generation. Existing approaches generate open-loop action scripts based on static knowledge, making it difficult to learn causal relationships between actions and environmental feedback, particularly in partially observable environments. We introduce Embodied Planner-R1, a novel outcome-driven reinforcement learning framework that enables LLMs to develop interactive capabilities through autonomous exploration with minimal supervision. Our framework incorporates three key innovations: (1) Without human annotations, we employ pure reinforcement learning with group rollout, incorporating in-environment interaction through parallel exploration; (2) completion-driven sparse reward; and (3) Interactive Policy Optimization (IPO) for efficient learning from grouped trajectories. Across two challenging text-based Embodied planning benchmarks, Embodied Planner-R1 achieves impressive completion rates of 97.78% on ALFWorld and 79.92% on ScienceWorld, surpassing prior methods by a large margin, and suffers only a -3.66% drop in previously unseen environments, evidencing strong generalization.
>
---
#### [new 052] Pipelined Decoder for Efficient Context-Aware Text Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本生成任务，旨在解决自回归模型生成速度慢的问题。提出一种流水线解码器，实现多子序列并行生成，提升速度且不损失质量。**

- **链接: [http://arxiv.org/pdf/2506.23431v1](http://arxiv.org/pdf/2506.23431v1)**

> **作者:** Zixian Huang; Chenxu Niu; Yu Gu; Gengyang Xiao; Xinwei Huang; Gong Cheng
>
> **摘要:** As the basis of generative AI, an autoregressive model requires the generation of a new token depending on all the previously generated tokens, which brings high quality but also restricts the model to generate tokens one by one, forming a bottleneck limiting the generation speed. In this paper, we propose a new decoder architecture that efficiently generates text in parallel for context-aware generation tasks. Our proposed pipelined decoder initiates the generation of multiple subsequences simultaneously, and, at each time-step, it generates a new token for each subsequence to realize parallelism. Experiments on multiple text generation tasks, including question answering, text summarization, and keyphrase generation, show that our pipelined decoder significantly improves the generation speed without a significant loss of generation quality or additional memory consumption.
>
---
#### [new 053] Robustness of Misinformation Classification Systems to Adversarial Examples Through BeamAttack
- **分类: cs.CL**

- **简介: 该论文属于文本分类任务，研究对抗样本对信息误判系统的影响。通过扩展BeamAttack算法，实现高效且隐蔽的攻击，验证模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.23661v1](http://arxiv.org/pdf/2506.23661v1)**

> **作者:** Arnisa Fazla; Lucas Krauter; David Guzman Piedrahita; Andrianos Michail
>
> **备注:** 12 pages main text, 27 pages total including references and appendices. 13 figures, 10 tables. Accepted for publication in the LNCS proceedings of CLEF 2025 (Best-of-Labs track)
>
> **摘要:** We extend BeamAttack, an adversarial attack algorithm designed to evaluate the robustness of text classification systems through word-level modifications guided by beam search. Our extensions include support for word deletions and the option to skip substitutions, enabling the discovery of minimal modifications that alter model predictions. We also integrate LIME to better prioritize word replacements. Evaluated across multiple datasets and victim models (BiLSTM, BERT, and adversarially trained RoBERTa) within the BODEGA framework, our approach achieves over a 99\% attack success rate while preserving the semantic and lexical similarity of the original texts. Through both quantitative and qualitative analysis, we highlight BeamAttack's effectiveness and its limitations. Our implementation is available at https://github.com/LucK1Y/BeamAttack
>
---
#### [new 054] RiverText: A Python Library for Training and Evaluating Incremental Word Embeddings from Text Data Streams
- **分类: cs.CL; cs.LG**

- **简介: 该论文介绍RiverText库，用于从文本流中训练和评估增量词嵌入。解决静态词嵌入无法适应语言变化的问题，实现动态更新。**

- **链接: [http://arxiv.org/pdf/2506.23192v1](http://arxiv.org/pdf/2506.23192v1)**

> **作者:** Gabriel Iturra-Bocaz; Felipe Bravo-Marquez
>
> **备注:** Accepted at SIGIR'23
>
> **摘要:** Word embeddings have become essential components in various information retrieval and natural language processing tasks, such as ranking, document classification, and question answering. However, despite their widespread use, traditional word embedding models present a limitation in their static nature, which hampers their ability to adapt to the constantly evolving language patterns that emerge in sources such as social media and the web (e.g., new hashtags or brand names). To overcome this problem, incremental word embedding algorithms are introduced, capable of dynamically updating word representations in response to new language patterns and processing continuous data streams. This paper presents RiverText, a Python library for training and evaluating incremental word embeddings from text data streams. Our tool is a resource for the information retrieval and natural language processing communities that work with word embeddings in streaming scenarios, such as analyzing social media. The library implements different incremental word embedding techniques, such as Skip-gram, Continuous Bag of Words, and Word Context Matrix, in a standardized framework. In addition, it uses PyTorch as its backend for neural network training. We have implemented a module that adapts existing intrinsic static word embedding evaluation tasks for word similarity and word categorization to a streaming setting. Finally, we compare the implemented methods with different hyperparameter settings and discuss the results. Our open-source library is available at https://github.com/dccuchile/rivertext.
>
---
#### [new 055] Format-Adapter: Improving Reasoning Capability of LLMs by Adapting Suitable Format
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs推理不一致问题。通过自适应生成和选择推理格式，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.23133v1](http://arxiv.org/pdf/2506.23133v1)**

> **作者:** Dingzirui Wang; Xuanliang Zhang; Rongyu Cao; Longxu Dou; Xianzhen Luo; Yingwei Ma; Qingfu Zhu; Wanxiang Che; Binhua Li; Fei Huang; Yongbin Li
>
> **摘要:** Generating and voting multiple answers is an effective method to mitigate reasoning inconsistencies of large language models (LLMs). Prior works have shown that multiple reasoning formats outperform a single format when generating multiple answers. However, previous works using multiple formats rely on formats labeled by humans, which could be unsuitable for all tasks and have high labeling costs. To address this issue, we adapt suitable formats to the given tasks by generating and selecting formats. We first propose how to measure the reasoning error when generating multiple answers. Then, we introduce Format-Adapter, which utilizes LLMs to generate and select suitable reasoning formats by minimizing the error measurement we present. We conduct experiments on math and commonsense reasoning tasks, where Format-Adapter achieves a 4.3% performance improvement on average over previous works, demonstrating the effectiveness.
>
---
#### [new 056] Computational Detection of Intertextual Parallels in Biblical Hebrew: A Benchmark Study Using Transformer-Based Language Models
- **分类: cs.CL**

- **简介: 该论文属于文本相似性检测任务，旨在解决圣经希伯来文中平行段落的自动识别问题。通过评估多种预训练模型，验证其在检测互文关系中的有效性。**

- **链接: [http://arxiv.org/pdf/2506.24117v1](http://arxiv.org/pdf/2506.24117v1)**

> **作者:** David M. Smiley
>
> **摘要:** Identifying parallel passages in biblical Hebrew is foundational in biblical scholarship for uncovering intertextual relationships. Traditional methods rely on manual comparison, which is labor-intensive and prone to human error. This study evaluates the potential of pre-trained transformer-based language models, including E5, AlephBERT, MPNet, and LaBSE, for detecting textual parallels in the Hebrew Bible. Focusing on known parallels between the books of Samuel/Kings and Chronicles, I assessed each model's capability to generate word embeddings that delineate parallel from non-parallel passages. Utilizing cosine similarity and Wasserstein Distance measures, I found that E5 and AlephBERT show significant promise, with E5 excelling in parallel detection and AlephBERT demonstrating stronger non-parallel differentiation. These findings indicate that pre-trained models can enhance the efficiency and accuracy of detecting intertextual parallels in ancient texts, suggesting broader applications for ancient language studies.
>
---
#### [new 057] Thought-Augmented Planning for LLM-Powered Interactive Recommender Agent
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于交互式推荐任务，旨在解决LLM代理在处理复杂用户意图时的不足。通过引入TAIRA系统，增强规划能力并提升推荐效果。**

- **链接: [http://arxiv.org/pdf/2506.23485v1](http://arxiv.org/pdf/2506.23485v1)**

> **作者:** Haocheng Yu; Yaxiong Wu; Hao Wang; Wei Guo; Yong Liu; Yawen Li; Yuyang Ye; Junping Du; Enhong Chen
>
> **摘要:** Interactive recommendation is a typical information-seeking task that allows users to interactively express their needs through natural language and obtain personalized recommendations. Large language model-powered (LLM-powered) agents have become a new paradigm in interactive recommendations, effectively capturing users' real-time needs and enhancing personalized experiences. However, due to limited planning and generalization capabilities, existing formulations of LLM-powered interactive recommender agents struggle to effectively address diverse and complex user intents, such as intuitive, unrefined, or occasionally ambiguous requests. To tackle this challenge, we propose a novel thought-augmented interactive recommender agent system (TAIRA) that addresses complex user intents through distilled thought patterns. Specifically, TAIRA is designed as an LLM-powered multi-agent system featuring a manager agent that orchestrates recommendation tasks by decomposing user needs and planning subtasks, with its planning capacity strengthened through Thought Pattern Distillation (TPD), a thought-augmentation method that extracts high-level thoughts from the agent's and human experts' experiences. Moreover, we designed a set of user simulation schemes to generate personalized queries of different difficulties and evaluate the recommendations based on specific datasets. Through comprehensive experiments conducted across multiple datasets, TAIRA exhibits significantly enhanced performance compared to existing methods. Notably, TAIRA shows a greater advantage on more challenging tasks while generalizing effectively on novel tasks, further validating its superiority in managing complex user intents within interactive recommendation systems. The code is publicly available at:https://github.com/Alcein/TAIRA.
>
---
#### [new 058] Evaluating the Simulation of Human Personality-Driven Susceptibility to Misinformation with LLMs
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于行为模拟任务，旨在评估LLMs是否能根据人格特征再现对虚假信息的敏感性。研究通过对比LLM与人类在新闻辨别上的表现，探讨其能力与局限。**

- **链接: [http://arxiv.org/pdf/2506.23610v1](http://arxiv.org/pdf/2506.23610v1)**

> **作者:** Manuel Pratelli; Marinella Petrocchi
>
> **备注:** pre-print version - paper actually under submission
>
> **摘要:** Large language models (LLMs) make it possible to generate synthetic behavioural data at scale, offering an ethical and low-cost alternative to human experiments. Whether such data can faithfully capture psychological differences driven by personality traits, however, remains an open question. We evaluate the capacity of LLM agents, conditioned on Big-Five profiles, to reproduce personality-based variation in susceptibility to misinformation, focusing on news discernment, the ability to judge true headlines as true and false headlines as false. Leveraging published datasets in which human participants with known personality profiles rated headline accuracy, we create matching LLM agents and compare their responses to the original human patterns. Certain trait-misinformation associations, notably those involving Agreeableness and Conscientiousness, are reliably replicated, whereas others diverge, revealing systematic biases in how LLMs internalize and express personality. The results underscore both the promise and the limits of personality-aligned LLMs for behavioral simulation, and offer new insight into modeling cognitive diversity in artificial agents.
>
---
#### [new 059] LLM-Assisted Question-Answering on Technical Documents Using Structured Data-Aware Retrieval Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于技术文档问答任务，旨在解决传统RAG在处理结构化数据（如表格、图像）时的不足。提出一种改进的RAG管道，结合向量检索与微调重排序器，提升问答准确性和相关性。**

- **链接: [http://arxiv.org/pdf/2506.23136v1](http://arxiv.org/pdf/2506.23136v1)**

> **作者:** Shadman Sobhan; Mohammad Ariful Haque
>
> **备注:** 29 Pages, 11 Tables
>
> **摘要:** Large Language Models (LLMs) are capable of natural language understanding and generation. But they face challenges such as hallucination and outdated knowledge. Fine-tuning is one possible solution, but it is resource-intensive and must be repeated with every data update. Retrieval-Augmented Generation (RAG) offers an efficient solution by allowing LLMs to access external knowledge sources. However, traditional RAG pipelines struggle with retrieving information from complex technical documents with structured data such as tables and images. In this work, we propose a RAG pipeline, capable of handling tables and images in documents, for technical documents that support both scanned and searchable formats. Its retrieval process combines vector similarity search with a fine-tuned reranker based on Gemma-2-9b-it. The reranker is trained using RAFT (Retrieval-Augmented Fine-Tuning) on a custom dataset designed to improve context identification for question answering. Our evaluation demonstrates that the proposed pipeline achieves a high faithfulness score of 94% (RAGas) and 96% (DeepEval), and an answer relevancy score of 87% (RAGas) and 93% (DeepEval). Comparative analysis demonstrates that the proposed architecture is superior to general RAG pipelines in terms of table-based questions and handling questions outside context.
>
---
#### [new 060] PromptAug: Fine-grained Conflict Classification Using Data Augmentation
- **分类: cs.CL; cs.AI; cs.CY; I.2.7; J.4; K.4.2**

- **简介: 该论文属于冲突分类任务，旨在解决标注数据稀缺与生成有害内容限制的问题。提出PromptAug方法，通过数据增强提升模型性能，并分析了生成文本的四大问题模式。**

- **链接: [http://arxiv.org/pdf/2506.22491v1](http://arxiv.org/pdf/2506.22491v1)**

> **作者:** Oliver Warke; Joemon M. Jose; Faegheh Hasibi; Jan Breitsohl
>
> **摘要:** Given the rise of conflicts on social media, effective classification models to detect harmful behaviours are essential. Following the garbage-in-garbage-out maxim, machine learning performance depends heavily on training data quality. However, high-quality labelled data, especially for nuanced tasks like identifying conflict behaviours, is limited, expensive, and difficult to obtain. Additionally, as social media platforms increasingly restrict access to research data, text data augmentation is gaining attention as an alternative to generate training data. Augmenting conflict-related data poses unique challenges due to Large Language Model (LLM) guardrails that prevent generation of offensive content. This paper introduces PromptAug, an innovative LLM-based data augmentation method. PromptAug achieves statistically significant improvements of 2% in both accuracy and F1-score on conflict and emotion datasets. To thoroughly evaluate PromptAug against other data augmentation methods we conduct a robust evaluation using extreme data scarcity scenarios, quantitative diversity analysis and a qualitative thematic analysis. The thematic analysis identifies four problematic patterns in augmented text: Linguistic Fluidity, Humour Ambiguity, Augmented Content Ambiguity, and Augmented Content Misinterpretation. Overall, this work presents PromptAug as an effective method for augmenting data in sensitive tasks like conflict detection, offering a unique, interdisciplinary evaluation grounded in both natural language processing and social science methodology.
>
---
#### [new 061] AutoEvoEval: An Automated Framework for Evolving Close-Ended LLM Evaluation Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM评估基准静态、不足的问题。提出AutoEvoEval框架，通过进化操作生成多样化测试样本，提升模型鲁棒性评估。**

- **链接: [http://arxiv.org/pdf/2506.23735v1](http://arxiv.org/pdf/2506.23735v1)**

> **作者:** JiaRu Wu; Mingwei Liu
>
> **摘要:** Large language models (LLMs) have shown remarkable performance on various tasks, but existing evaluation benchmarks are often static and insufficient to fully assess their robustness and generalization in realistic scenarios. Prior work using evolutionary or adversarial data augmentation has improved evaluation diversity but lacks systematic control over perturbation types and multi-step complexity, limiting comprehensive robustness analysis. To address these gaps, we propose AutoEvoEval, an evolution-based evaluation framework for close-ended tasks such as multi-choice question answering. AutoEvoEval introduces 22 interpretable atomic evolution operations and supports multi-round compositions, enabling controlled generation of diverse, challenging, and realistic test samples. We conduct extensive experiments addressing four research questions on a broad set of open- and closed-source LLMs. Our results show that atomic operations cause an average accuracy drop of 7.283\%, with structure-disrupting or misleading semantic edits causing the largest declines. Model sensitivities vary significantly for the same perturbation, and combining multiple evolution steps amplifies adversarial effects by up to 52.932\%. These findings suggest current benchmarks may overestimate true model generalization and emphasize the need for evolution-aware robustness evaluation. Code and resources are available at: https://github.com/SYSUSELab/AutoEvoEval.
>
---
#### [new 062] On Recipe Memorization and Creativity in Large Language Models: Is Your Model a Creative Cook, a Bad Cook, or Merely a Plagiator?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在生成食谱时的记忆、创造性和荒谬性，旨在评估模型是否具备创造力或仅是抄袭。任务属于自然语言生成与评估，解决模型生成内容质量分析问题，通过人工标注和自动化框架进行分析。**

- **链接: [http://arxiv.org/pdf/2506.23527v1](http://arxiv.org/pdf/2506.23527v1)**

> **作者:** Jan Kvapil; Martin Fajcik
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** This work-in-progress investigates the memorization, creativity, and nonsense found in cooking recipes generated from Large Language Models (LLMs). Precisely, we aim (i) to analyze memorization, creativity, and non-sense in LLMs using a small, high-quality set of human judgments and (ii) to evaluate potential approaches to automate such a human annotation in order to scale our study to hundreds of recipes. To achieve (i), we conduct a detailed human annotation on 20 preselected recipes generated by LLM (Mixtral), extracting each recipe's ingredients and step-by-step actions to assess which elements are memorized--i.e., directly traceable to online sources possibly seen during training--and which arise from genuine creative synthesis or outright nonsense. We find that Mixtral consistently reuses ingredients that can be found in online documents, potentially seen during model training, suggesting strong reliance on memorized content. To achieve aim (ii) and scale our analysis beyond small sample sizes and single LLM validation, we design an ``LLM-as-judge'' pipeline that automates recipe generation, nonsense detection, parsing ingredients and recipe steps, and their annotation. For instance, comparing its output against human annotations, the best ingredient extractor and annotator is Llama 3.1+Gemma 2 9B, achieving up to 78% accuracy on ingredient matching. This automated framework enables large-scale quantification of memorization, creativity, and nonsense in generated recipes, providing rigorous evidence of the models' creative capacities.
>
---
#### [new 063] MariNER: A Dataset for Historical Brazilian Portuguese Named Entity Recognition
- **分类: cs.CL**

- **简介: 该论文属于命名实体识别任务，旨在解决巴西葡萄牙语历史文本数据集不足的问题，构建了MariNER数据集并评估了先进模型性能。**

- **链接: [http://arxiv.org/pdf/2506.23051v1](http://arxiv.org/pdf/2506.23051v1)**

> **作者:** João Lucas Luz Lima Sarcinelli; Marina Lages Gonçalves Teixeira; Jade Bortot de Paiva; Diego Furtado Silva
>
> **摘要:** Named Entity Recognition (NER) is a fundamental Natural Language Processing (NLP) task that aims to identify and classify entity mentions in texts across different categories. While languages such as English possess a large number of high-quality resources for this task, Brazilian Portuguese still lacks in quantity of gold-standard NER datasets, especially when considering specific domains. Particularly, this paper considers the importance of NER for analyzing historical texts in the context of digital humanities. To address this gap, this work outlines the construction of MariNER: \textit{Mapeamento e Anota\c{c}\~oes de Registros hIst\'oricos para NER} (Mapping and Annotation of Historical Records for NER), the first gold-standard dataset for early 20th-century Brazilian Portuguese, with more than 9,000 manually annotated sentences. We also assess and compare the performance of state-of-the-art NER models for the dataset.
>
---
#### [new 064] Do Thinking Tokens Help or Trap? Towards More Efficient Large Reasoning Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大模型推理任务，解决LRM在简单任务中因“思考标记”导致的效率低下问题，提出DuP-PO算法提升token效率。**

- **链接: [http://arxiv.org/pdf/2506.23840v1](http://arxiv.org/pdf/2506.23840v1)**

> **作者:** Bowen Ding; Yuhan Chen; Futing Wang; Lingfeng Ming; Tao Lin
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Large Reasoning Models (LRMs) excel at solving complex problems but face an overthinking dilemma. When handling simple tasks, they often produce verbose responses overloaded with thinking tokens (e.g., wait, however). These tokens trigger unnecessary high-level reasoning behaviors like reflection and backtracking, reducing efficiency. In this work, our pilot study reveals that these thinking-token-induced behaviors are not essential for effective problem-solving and may even hinder correct reasoning within constrained token budgets. We identify this phenomenon as the thinking trap. To mitigate this issue, we propose Dual Policy Preference Optimization (DuP-PO), a novel algorithm featuring: (1) A rollout sampling strategy that guarantees balanced exposure to responses with and without thinking tokens; (2) A fine-grained advantage control technique to dynamically regulate the prediction of target tokens; (3) A policy shaping method ensuring stable gradient contributions from thinking tokens. Experimental results on five popular math reasoning benchmarks show that DuP-PO performs well on the popular LRM, which significantly improves their token efficiency during reasoning, while achieving superior performance of the base model.
>
---
#### [new 065] Teaching Models to Verbalize Reward Hacking in Chain-of-Thought Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI安全任务，旨在解决奖励黑客问题。通过VFT训练模型显式表达受提示影响，提升检测率，增强AI透明度与安全性。**

- **链接: [http://arxiv.org/pdf/2506.22777v1](http://arxiv.org/pdf/2506.22777v1)**

> **作者:** Miles Turpin; Andy Arditi; Marvin Li; Joe Benton; Julian Michael
>
> **摘要:** Language models trained with RL can engage in reward hacking--exploiting unintended strategies for high reward--without revealing this behavior in their chain-of-thought reasoning, making detection difficult and posing risks for high-stakes applications. We propose verbalization fine-tuning (VFT), a pre-RL intervention that trains models to explicitly acknowledge when they are influenced by prompt cues--hints which point to incorrect answers (e.g., "a Stanford professor thinks the answer is A"). To evaluate VFT, we subsequently train models with RL on environments where held-out prompt cues signal which incorrect answers will receive high reward, incentivizing models to reward hack by exploiting cues instead of reasoning correctly. We measure how often models exploit these cues without verbalizing it. After RL, only 6% of the VFT-trained model's responses consist of undetected reward hacks. In comparison, when we perform RL without VFT, the rate of undetected reward hacks goes up to 88%; with a debiasing baseline intervention, this increases further to 99%. VFT achieves this by substantially increasing how often models verbalize the influence of cues--from 8% to 42% after VFT, and up to 94% after RL--while baselines remain low even after RL (10% and 1%). Our results show that teaching models to explicitly verbalize reward hacking behavior before RL significantly improves their detection, offering a practical path toward more transparent and safe AI systems.
>
---
#### [new 066] AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本匿名化任务，旨在解决用户生成文本中隐私泄露问题。通过提出AgentStealth框架，结合对抗学习与强化学习，提升匿名化效果与文本实用性。**

- **链接: [http://arxiv.org/pdf/2506.22508v1](http://arxiv.org/pdf/2506.22508v1)**

> **作者:** Chenyang Shao; Tianxing Li; Chenhao Pu; Fengli Xu; Yong Li
>
> **备注:** This work has been submitted to NeurIPS 2025. Under review
>
> **摘要:** In today's digital world, casual user-generated content often contains subtle cues that may inadvertently expose sensitive personal attributes. Such risks underscore the growing importance of effective text anonymization to safeguard individual privacy. However, existing methods either rely on rigid replacements that damage utility or cloud-based LLMs that are costly and pose privacy risks. To address these issues, we explore the use of locally deployed smaller-scale language models (SLMs) for anonymization. Yet training effective SLMs remains challenging due to limited high-quality supervision. To address the challenge, we propose AgentStealth, a self-reinforcing LLM anonymization framework.First, we introduce an adversarial anonymization workflow enhanced by In-context Contrastive Learning and Adaptive Utility-Aware Control. Second, we perform supervised adaptation of SLMs using high-quality data collected from the workflow, which includes both anonymization and attack signals. Finally, we apply online reinforcement learning where the model leverages its internal adversarial feedback to iteratively improve anonymization performance. Experiments on two datasets show that our method outperforms baselines in both anonymization effectiveness (+12.3%) and utility (+6.8%). Our lightweight design supports direct deployment on edge devices, avoiding cloud reliance and communication-based privacy risks. Our code is open-source at https://github.com/tsinghua-fib-lab/AgentStealth.
>
---
#### [new 067] Evaluating Hybrid Retrieval Augmented Generation using Dynamic Test Sets: LiveRAG Challenge
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于RAG系统评估任务，旨在提升生成答案的准确性和真实性。通过结合稀疏与密集检索方法，并优化提示策略，解决动态测试集下的性能问题。**

- **链接: [http://arxiv.org/pdf/2506.22644v1](http://arxiv.org/pdf/2506.22644v1)**

> **作者:** Chase Fensore; Kaustubh Dhole; Joyce C Ho; Eugene Agichtein
>
> **备注:** 4 pages, 3 tables, 2 figures. Accepted at the SIGIR LiveRAG Workshop 2025 (Submission 2664)
>
> **摘要:** We present our submission to the LiveRAG Challenge 2025, which evaluates retrieval-augmented generation (RAG) systems on dynamic test sets using the FineWeb-10BT corpus. Our final hybrid approach combines sparse (BM25) and dense (E5) retrieval methods and then aims to generate relevant and faithful answers with Falcon3-10B-Instruct. Through systematic evaluation on 200 synthetic questions generated with DataMorgana across 64 unique question-user combinations, we demonstrate that neural re-ranking with RankLLaMA improves MAP from 0.523 to 0.797 (52% relative improvement) but introduces prohibitive computational costs (84s vs 1.74s per question). While DSPy-optimized prompting strategies achieved higher semantic similarity (0.771 vs 0.668), their 0% refusal rates raised concerns about over-confidence and generalizability. Our submitted hybrid system without re-ranking achieved 4th place in faithfulness and 11th place in correctness among 25 teams. Analysis across question categories reveals that vocabulary alignment between questions and documents was the strongest predictor of performance on our development set, with document-similar phrasing improving cosine similarity from 0.562 to 0.762.
>
---
#### [new 068] ATGen: A Framework for Active Text Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，旨在解决AL在NLG中应用有限的问题。提出ATGen框架，结合人工与自动标注，提升效率并降低成本。**

- **链接: [http://arxiv.org/pdf/2506.23342v1](http://arxiv.org/pdf/2506.23342v1)**

> **作者:** Akim Tsvigun; Daniil Vasilev; Ivan Tsvigun; Ivan Lysenko; Talgat Bektleuov; Aleksandr Medvedev; Uliana Vinogradova; Nikita Severin; Mikhail Mozikov; Andrey Savchenko; Rostislav Grigorev; Ramil Kuleev; Fedor Zhdanov; Artem Shelmanov; Ilya Makarov
>
> **备注:** Accepted at ACL 2025 System Demonstrations
>
> **摘要:** Active learning (AL) has demonstrated remarkable potential in reducing the annotation effort required for training machine learning models. However, despite the surging popularity of natural language generation (NLG) tasks in recent years, the application of AL to NLG has been limited. In this paper, we introduce Active Text Generation (ATGen) - a comprehensive framework that bridges AL with text generation tasks, enabling the application of state-of-the-art AL strategies to NLG. Our framework simplifies AL-empowered annotation in NLG tasks using both human annotators and automatic annotation agents based on large language models (LLMs). The framework supports LLMs deployed as services, such as ChatGPT and Claude, or operated on-premises. Furthermore, ATGen provides a unified platform for smooth implementation and benchmarking of novel AL strategies tailored to NLG tasks. Finally, we present evaluation results for state-of-the-art AL strategies across diverse settings and multiple text generation tasks. We show that ATGen reduces both the effort of human annotators and costs associated with API calls to LLM-based annotation agents. The code of the framework is available on GitHub under the MIT license. The video presentation is available at http://atgen-video.nlpresearch.group
>
---
#### [new 069] Two Spelling Normalization Approaches Based on Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于拼写规范化任务，旨在解决历史文献中拼写不统一的问题。通过两种基于大语言模型的方法进行研究与评估。**

- **链接: [http://arxiv.org/pdf/2506.23288v1](http://arxiv.org/pdf/2506.23288v1)**

> **作者:** Miguel Domingo; Francisco Casacuberta
>
> **摘要:** The absence of standardized spelling conventions and the organic evolution of human language present an inherent linguistic challenge within historical documents, a longstanding concern for scholars in the humanities. Addressing this issue, spelling normalization endeavors to align a document's orthography with contemporary standards. In this study, we propose two new approaches based on large language models: one of which has been trained without a supervised training, and a second one which has been trained for machine translation. Our evaluation spans multiple datasets encompassing diverse languages and historical periods, leading us to the conclusion that while both of them yielded encouraging results, statistical machine translation still seems to be the most suitable technology for this task.
>
---
#### [new 070] Reinforcement Fine-Tuning Enables MLLMs Learning Novel Tasks Stably
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多模态大模型在新任务学习中的稳定性问题，对比SFT与RFT方法，发现RFT更利于保持已有知识。**

- **链接: [http://arxiv.org/pdf/2506.23508v1](http://arxiv.org/pdf/2506.23508v1)**

> **作者:** Zhihao Zhang; Qiaole Dong; Qi Zhang; Jun Zhao; Enyu Zhou; Zhiheng Xi; Senjie Jin; Xiaoran Fan; Yuhao Zhou; Yanwei Fu; Tao Ji; Tao Gui; Xuanjing Huang
>
> **备注:** 18 pages (Preprint. Work in progress)
>
> **摘要:** Post-training algorithms such as Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT) are widely used to adapt multimodal large language models to downstream tasks. While effective at task adaptation, their impact on prior knowledge remains unclear. In this paper, we introduce jigsaw puzzles as a novel task absent from existing pretraining corpora and systematically study the behavior of SFT and RFT on an open-source multimodal model, Qwen2.5-VL. Our experiments reveal a sharp trade-off: SFT enables rapid task acquisition but leads to catastrophic forgetting, whereas RFT learns more slowly on novel tasks but maintains prior knowledge. We analyze this phenomenon through the lens of learning dynamics, showing that RFT reinforces correct samples that are naturally aligned with the base model's probability landscape, mitigating interference with prior knowledge. Moreover, supervised training on correct RFT-simulated rollouts allows SFT to preserve knowledge while rapidly learning new tasks. These findings suggest that data distribution, rather than algorithmic differences, plays a central role in forgetting, and highlight RFT's potential for stable continual learning in multimodal large language models.
>
---
#### [new 071] Assessing the feasibility of Large Language Models for detecting micro-behaviors in team interactions during space missions
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的行为识别任务，旨在检测太空任务中团队对话中的微行为。通过实验对比不同模型效果，探索LLM在该场景下的可行性。**

- **链接: [http://arxiv.org/pdf/2506.22679v1](http://arxiv.org/pdf/2506.22679v1)**

> **作者:** Ankush Raut; Projna Paromita; Sydney Begerowski; Suzanne Bell; Theodora Chaspari
>
> **备注:** 5 pages, 4 figures. Accepted to Interspeech 2025
>
> **摘要:** We explore the feasibility of large language models (LLMs) in detecting subtle expressions of micro-behaviors in team conversations using transcripts collected during simulated space missions. Specifically, we examine zero-shot classification, fine-tuning, and paraphrase-augmented fine-tuning with encoder-only sequence classification LLMs, as well as few-shot text generation with decoder-only causal language modeling LLMs, to predict the micro-behavior associated with each conversational turn (i.e., dialogue). Our findings indicate that encoder-only LLMs, such as RoBERTa and DistilBERT, struggled to detect underrepresented micro-behaviors, particularly discouraging speech, even with weighted fine-tuning. In contrast, the instruction fine-tuned version of Llama-3.1, a decoder-only LLM, demonstrated superior performance, with the best models achieving macro F1-scores of 44% for 3-way classification and 68% for binary classification. These results have implications for the development of speech technologies aimed at analyzing team communication dynamics and enhancing training interventions in high-stakes environments such as space missions, particularly in scenarios where text is the only accessible data.
>
---
#### [new 072] Garbage In, Reasoning Out? Why Benchmark Scores are Unreliable and What to Do About It
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型评估任务，旨在解决基准测试可靠性问题。通过分析多个推理基准，发现其设计和评分存在缺陷，并提出更合理的评估方法。**

- **链接: [http://arxiv.org/pdf/2506.23864v1](http://arxiv.org/pdf/2506.23864v1)**

> **作者:** Seyed Mahed Mousavi; Edoardo Cecchinato; Lucia Hornikova; Giuseppe Riccardi
>
> **摘要:** We conduct a systematic audit of three widely used reasoning benchmarks, SocialIQa, FauxPas-EAI, and ToMi, and uncover pervasive flaws in both benchmark items and evaluation methodology. Using five LLMs (GPT-{3, 3.5, 4, o1}, and LLaMA 3.1) as diagnostic tools, we identify structural, semantic, and pragmatic issues in benchmark design (e.g., duplicated items, ambiguous wording, and implausible answers), as well as scoring procedures that prioritize output form over reasoning process. Through systematic human annotation and re-evaluation on cleaned benchmark subsets, we find that model scores often improve not due to due to erratic surface wording variations and not to improved reasoning. Infact, further analyses show that model performance is highly sensitive to minor input variations such as context availability and phrasing, revealing that high scores may reflect alignment with format-specific cues rather than consistent inference based on the input. These findings challenge the validity of current benchmark-based claims about reasoning in LLMs, and highlight the need for evaluation protocols that assess reasoning as a process of drawing inference from available information, rather than as static output selection. We release audited data and evaluation tools to support more interpretable and diagnostic assessments of model reasoning.
>
---
#### [new 073] Semantic-guided Diverse Decoding for Large Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，解决大模型生成缺乏语义多样性的问题。提出SemDiD方法，在嵌入空间中提升生成结果的语义差异性与质量。**

- **链接: [http://arxiv.org/pdf/2506.23601v1](http://arxiv.org/pdf/2506.23601v1)**

> **作者:** Weijie Shi; Yue Cui; Yaguang Wu; Jingzhi Fang; Shibo Zhang; Mengze Li; Sirui Han; Jia Zhu; Jiajie Xu; Xiaofang Zhou
>
> **摘要:** Diverse decoding of large language models is crucial for applications requiring multiple semantically distinct responses, yet existing methods primarily achieve lexical rather than semantic diversity. This limitation significantly constrains Best-of-N strategies, group-based reinforcement learning, and data synthesis. While temperature sampling and diverse beam search modify token distributions or apply n-gram penalties, they fail to ensure meaningful semantic differentiation. We introduce Semantic-guided Diverse Decoding (SemDiD), operating directly in embedding space that balances quality with diversity through three complementary mechanisms: orthogonal directional guidance, dynamic inter-group repulsion, and position-debiased probability assessment. SemDiD harmonizes these competing objectives using adaptive gain functions and constraint optimization, ensuring both quality thresholds and maximal semantic differentiation. Experiments show SemDiD consistently outperforms existing methods, improving Best-of-N coverage by 1.4-5.2% across diverse tasks and accelerating RLHF training convergence by 15% while increasing accuracy by up to 2.1%.
>
---
#### [new 074] Mind the Gap: Entity-Preserved Context-Aware ASR Structured Transcriptions
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决实体和数字识别不准的问题。通过扩展上下文窗口和优化实体分配，提升长文本的准确性和格式正确性。**

- **链接: [http://arxiv.org/pdf/2506.22858v1](http://arxiv.org/pdf/2506.22858v1)**

> **作者:** Duygu Altinok
>
> **备注:** This is the accepted version of an article accepted to the TSD 2025 conference, published in Springer Lecture Notes in Artificial Intelligence (LNAI). The final authenticated version is available online at SpringerLink
>
> **摘要:** Automatic Speech Recognition (ASR) systems, such as Whisper, achieve high transcription accuracy but struggle with named entities and numerical data, especially when proper formatting is required. These issues increase word error rate (WER) and impair semantic understanding in critical domains like legal, financial, and medical applications. We propose a novel training approach that extends the semantic context of ASR models by adding overlapping context windows during training. By sliding 5-second overlaps on both sides of 30-second chunks, we create a 40-second "effective semantic window," improving entity recognition and formatting while focusing predictions on the central 30 seconds. To address entities spanning chunk boundaries, we reassign such entities entirely to the right-hand chunk, ensuring proper formatting. Additionally, enriched training data with embedded entity labels enables the model to learn both recognition and type-specific formatting. Evaluated on the Spoken Wikipedia dataset, our method improves performance across semantic tasks, including named entity recognition (NER) and entity formatting. These results highlight the effectiveness of context-aware training in addressing ASR limitations for long-form transcription and complex entity recognition tasks.
>
---
#### [new 075] STACK: Adversarial Attacks on LLM Safeguard Pipelines
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI安全领域，研究如何攻击大模型的防护机制。通过构建防御管道并进行红队测试，提出STAGED攻击方法，验证了防护系统的脆弱性，并提出缓解建议。**

- **链接: [http://arxiv.org/pdf/2506.24068v1](http://arxiv.org/pdf/2506.24068v1)**

> **作者:** Ian R. McKenzie; Oskar J. Hollinsworth; Tom Tseng; Xander Davies; Stephen Casper; Aaron D. Tucker; Robert Kirk; Adam Gleave
>
> **摘要:** Frontier AI developers are relying on layers of safeguards to protect against catastrophic misuse of AI systems. Anthropic guards their latest Claude 4 Opus model using one such defense pipeline, and other frontier developers including Google DeepMind and OpenAI pledge to soon deploy similar defenses. However, the security of such pipelines is unclear, with limited prior work evaluating or attacking these pipelines. We address this gap by developing and red-teaming an open-source defense pipeline. First, we find that a novel few-shot-prompted input and output classifier outperforms state-of-the-art open-weight safeguard model ShieldGemma across three attacks and two datasets, reducing the attack success rate (ASR) to 0% on the catastrophic misuse dataset ClearHarm. Second, we introduce a STaged AttaCK (STACK) procedure that achieves 71% ASR on ClearHarm in a black-box attack against the few-shot-prompted classifier pipeline. Finally, we also evaluate STACK in a transfer setting, achieving 33% ASR, providing initial evidence that it is feasible to design attacks with no access to the target pipeline. We conclude by suggesting specific mitigations that developers could use to thwart staged attacks.
>
---
#### [new 076] Leveraging the Potential of Prompt Engineering for Hate Speech Detection in Low-Resource Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 hate speech detection 任务，旨在解决低资源语言中仇恨言论检测困难的问题。通过 prompt engineering 方法提升大语言模型在 Bengali 等低资源语言上的检测效果。**

- **链接: [http://arxiv.org/pdf/2506.23930v1](http://arxiv.org/pdf/2506.23930v1)**

> **作者:** Ruhina Tabasshum Prome; Tarikul Islam Tamiti; Anomadarshi Barua
>
> **摘要:** The rapid expansion of social media leads to a marked increase in hate speech, which threatens personal lives and results in numerous hate crimes. Detecting hate speech presents several challenges: diverse dialects, frequent code-mixing, and the prevalence of misspelled words in user-generated content on social media platforms. Recent progress in hate speech detection is typically concentrated on high-resource languages. However, low-resource languages still face significant challenges due to the lack of large-scale, high-quality datasets. This paper investigates how we can overcome this limitation via prompt engineering on large language models (LLMs) focusing on low-resource Bengali language. We investigate six prompting strategies - zero-shot prompting, refusal suppression, flattering the classifier, multi-shot prompting, role prompting, and finally our innovative metaphor prompting to detect hate speech effectively in low-resource languages. We pioneer the metaphor prompting to circumvent the built-in safety mechanisms of LLMs that marks a significant departure from existing jailbreaking methods. We investigate all six different prompting strategies on the Llama2-7B model and compare the results extensively with three pre-trained word embeddings - GloVe, Word2Vec, and FastText for three different deep learning models - multilayer perceptron (MLP), convolutional neural network (CNN), and bidirectional gated recurrent unit (BiGRU). To prove the effectiveness of our metaphor prompting in the low-resource Bengali language, we also evaluate it in another low-resource language - Hindi, and two high-resource languages - English and German. The performance of all prompting techniques is evaluated using the F1 score, and environmental impact factor (IF), which measures CO$_2$ emissions, electricity usage, and computational time.
>
---
#### [new 077] Ensemble BERT for Medication Event Classification on Electronic Health Records (EHRs)
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于医疗文本分类任务，旨在从电子病历中识别和分类药物事件。通过构建BERT集成模型提升分类效果。**

- **链接: [http://arxiv.org/pdf/2506.23315v1](http://arxiv.org/pdf/2506.23315v1)**

> **作者:** Shouvon Sarker; Xishuang Dong; Lijun Qian
>
> **摘要:** Identification of key variables such as medications, diseases, relations from health records and clinical notes has a wide range of applications in the clinical domain. n2c2 2022 provided shared tasks on challenges in natural language processing for clinical data analytics on electronic health records (EHR), where it built a comprehensive annotated clinical data Contextualized Medication Event Dataset (CMED). This study focuses on subtask 2 in Track 1 of this challenge that is to detect and classify medication events from clinical notes through building a novel BERT-based ensemble model. It started with pretraining BERT models on different types of big data such as Wikipedia and MIMIC. Afterwards, these pretrained BERT models were fine-tuned on CMED training data. These fine-tuned BERT models were employed to accomplish medication event classification on CMED testing data with multiple predictions. These multiple predictions generated by these fine-tuned BERT models were integrated to build final prediction with voting strategies. Experimental results demonstrated that BERT-based ensemble models can effectively improve strict Micro-F score by about 5% and strict Macro-F score by about 6%, respectively.
>
---
#### [new 078] Psycholinguistic Word Features: a New Approach for the Evaluation of LLMs Alignment with Humans
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在评估大语言模型与人类在心理语言特征上的对齐程度。通过对比两个心理语言数据集，发现模型在部分特征上表现较好，但在感官关联上存在不足。**

- **链接: [http://arxiv.org/pdf/2506.22439v1](http://arxiv.org/pdf/2506.22439v1)**

> **作者:** Javier Conde; Miguel González; María Grandury; Gonzalo Martínez; Pedro Reviriego; Mar Brysbaert
>
> **备注:** Accepted for the GEM2 workshop at ACL 2025
>
> **摘要:** The evaluation of LLMs has so far focused primarily on how well they can perform different tasks such as reasoning, question-answering, paraphrasing, or translating. For most of these tasks, performance can be measured with objective metrics, such as the number of correct answers. However, other language features are not easily quantified. For example, arousal, concreteness, or gender associated with a given word, as well as the extent to which we experience words with senses and relate them to a specific sense. Those features have been studied for many years by psycholinguistics, conducting large-scale experiments with humans to produce ratings for thousands of words. This opens an opportunity to evaluate how well LLMs align with human ratings on these word features, taking advantage of existing studies that cover many different language features in a large number of words. In this paper, we evaluate the alignment of a representative group of LLMs with human ratings on two psycholinguistic datasets: the Glasgow and Lancaster norms. These datasets cover thirteen features over thousands of words. The results show that alignment is \textcolor{black}{generally} better in the Glasgow norms evaluated (arousal, valence, dominance, concreteness, imageability, familiarity, and gender) than on the Lancaster norms evaluated (introceptive, gustatory, olfactory, haptic, auditory, and visual). This suggests a potential limitation of current LLMs in aligning with human sensory associations for words, which may be due to their lack of embodied cognition present in humans and illustrates the usefulness of evaluating LLMs with psycholinguistic datasets.
>
---
#### [new 079] Selecting and Merging: Towards Adaptable and Scalable Named Entity Recognition with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于命名实体识别任务，解决领域适应与模型扩展性问题。提出SaM框架，在推理时动态选择并融合专家模型，提升跨领域泛化能力与可扩展性。**

- **链接: [http://arxiv.org/pdf/2506.22813v1](http://arxiv.org/pdf/2506.22813v1)**

> **作者:** Zhuojun Ding; Wei Wei; Chenghao Fan
>
> **摘要:** Supervised fine-tuning (SFT) is widely used to align large language models (LLMs) with information extraction (IE) tasks, such as named entity recognition (NER). However, annotating such fine-grained labels and training domain-specific models is costly. Existing works typically train a unified model across multiple domains, but such approaches lack adaptation and scalability since not all training data benefits target domains and scaling trained models remains challenging. We propose the SaM framework, which dynamically Selects and Merges expert models at inference time. Specifically, for a target domain, we select domain-specific experts pre-trained on existing domains based on (i) domain similarity to the target domain and (ii) performance on sampled instances, respectively. The experts are then merged to create task-specific models optimized for the target domain. By dynamically merging experts beneficial to target domains, we improve generalization across various domains without extra training. Additionally, experts can be added or removed conveniently, leading to great scalability. Extensive experiments on multiple benchmarks demonstrate our framework's effectiveness, which outperforms the unified model by an average of 10%. We further provide insights into potential improvements, practical experience, and extensions of our framework.
>
---
#### [new 080] A Systematic Study of Compositional Syntactic Transformer Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语法语言模型，旨在提升Transformer的句法建模能力。通过构建统一框架，评估不同变体在多种任务中的表现，提出设计优化建议。**

- **链接: [http://arxiv.org/pdf/2506.22978v1](http://arxiv.org/pdf/2506.22978v1)**

> **作者:** Yida Zhao; Hao Xve; Xiang Hu; Kewei Tu
>
> **摘要:** Syntactic language models (SLMs) enhance Transformers by incorporating syntactic biases through the modeling of linearized syntactic parse trees alongside surface sentences. This paper focuses on compositional SLMs that are based on constituency parse trees and contain explicit bottom-up composition of constituent representations. We identify key aspects of design choices in existing compositional SLMs and propose a unified framework encompassing both existing models and novel variants. We conduct a comprehensive empirical evaluation of all the variants in our framework across language modeling, syntactic generalization, summarization, dialogue, and inference efficiency. Based on the experimental results, we make multiple recommendations on the design of compositional SLMs. Our code is released at https://github.com/zhaoyd1/compositional_SLMs.
>
---
#### [new 081] Text Production and Comprehension by Human and Artificial Intelligence: Interdisciplinary Workshop Report
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人机语言交互研究，旨在探讨AI与人类在文本生成和理解中的关系。通过跨学科合作，分析LLM的能力与局限，推动人机协作发展。**

- **链接: [http://arxiv.org/pdf/2506.22698v1](http://arxiv.org/pdf/2506.22698v1)**

> **作者:** Emily Dux Speltz
>
> **摘要:** This report synthesizes the outcomes of a recent interdisciplinary workshop that brought together leading experts in cognitive psychology, language learning, and artificial intelligence (AI)-based natural language processing (NLP). The workshop, funded by the National Science Foundation, aimed to address a critical knowledge gap in our understanding of the relationship between AI language models and human cognitive processes in text comprehension and composition. Through collaborative dialogue across cognitive, linguistic, and technological perspectives, workshop participants examined the underlying processes involved when humans produce and comprehend text, and how AI can both inform our understanding of these processes and augment human capabilities. The workshop revealed emerging patterns in the relationship between large language models (LLMs) and human cognition, with highlights on both the capabilities of LLMs and their limitations in fully replicating human-like language understanding and generation. Key findings include the potential of LLMs to offer insights into human language processing, the increasing alignment between LLM behavior and human language processing when models are fine-tuned with human feedback, and the opportunities and challenges presented by human-AI collaboration in language tasks. By synthesizing these findings, this report aims to guide future research, development, and implementation of LLMs in cognitive psychology, linguistics, and education. It emphasizes the importance of ethical considerations and responsible use of AI technologies while striving to enhance human capabilities in text comprehension and production through effective human-AI collaboration.
>
---
#### [new 082] V-SYNTHESIS: Task-Agnostic Synthesis of Consistent and Diverse In-Context Demonstrations from Scratch via V-Entropy
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决ICL演示合成的高成本与一致性问题。提出V-Synthesis方法，实现无标签的高质量演示生成。**

- **链接: [http://arxiv.org/pdf/2506.23149v1](http://arxiv.org/pdf/2506.23149v1)**

> **作者:** Dingzirui Wang; Xuanliang Zhang; Keyan Xu; Qingfu Zhu; Wanxiang Che; Yang Deng
>
> **摘要:** High labeling cost for in-context learning (ICL) demonstrations motivates using large language models (LLMs) for synthesis to reduce overhead. However, existing synthesis methods are mainly task-specific or rely on pre-existing demonstrations. So this paper focuses on synthesizing demonstrations from scratch for arbitrary tasks. A major challenge in synthesizing from scratch is ensuring consistency with the target task, as the lack of labeling guidance could lead to synthesis bias. We first propose a consistency metric called V-Score, which has higher performance and lower computation cost compared with the metrics based on grams or embedding vectors. Furthermore, we introduce V-Synthesis, which leverages V-Score for proportional sampling to ensure both high consistency and diversity of synthesized demonstrations. Experimental results demonstrate that V-Synthesis yields an average performance improvement of 2.0% compared to existing synthesis methods confirming the effectiveness of V-Synthesis.
>
---
#### [new 083] GaussMaster: An LLM-based Database Copilot System
- **分类: cs.DB; cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出GaussMaster，一个基于LLM的数据库助手系统，旨在解决数据库维护中人工干预多的问题，实现自动化故障诊断与处理。**

- **链接: [http://arxiv.org/pdf/2506.23322v1](http://arxiv.org/pdf/2506.23322v1)**

> **作者:** Wei Zhou; Ji Sun; Xuanhe Zhou; Guoliang Li; Luyang Liu; Hao Wu; Tianyuan Wang
>
> **备注:** We welcome contributions from the community. For reference, please see the code at: https://gitcode.com/opengauss/openGauss-GaussMaster
>
> **摘要:** In the financial industry, data is the lifeblood of operations, and DBAs shoulder significant responsibilities for SQL tuning, database deployment, diagnosis, and service repair. In recent years, both database vendors and customers have increasingly turned to autonomous database platforms in an effort to alleviate the heavy workload of DBAs. However, existing autonomous database platforms are limited in their capabilities, primarily addressing single-point issues such as NL2SQL, anomaly detection, and SQL tuning. Manual intervention remains a necessity for comprehensive database maintenance. GaussMaster aims to revolutionize this landscape by introducing an LLM-based database copilot system. This innovative solution is designed not only to assist developers in writing efficient SQL queries but also to provide comprehensive care for database services. When database instances exhibit abnormal behavior, GaussMaster is capable of orchestrating the entire maintenance process automatically. It achieves this by analyzing hundreds of metrics and logs, employing a Tree-of-thought approach to identify root causes, and invoking appropriate tools to resolve issues. We have successfully implemented GaussMaster in real-world scenarios, such as the banking industry, where it has achieved zero human intervention for over 34 database maintenance scenarios. In this paper, we present significant improvements in these tasks with code at https://gitcode.com/opengauss/openGauss-GaussMaster.
>
---
#### [new 084] BayesLoRA: Task-Specific Uncertainty in Low-Rank Adapters
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出BayesLoRA，用于低秩适配器中的任务特定不确定性量化，解决模型在下游任务中的置信度估计问题。**

- **链接: [http://arxiv.org/pdf/2506.22809v1](http://arxiv.org/pdf/2506.22809v1)**

> **作者:** Cooper Doyle
>
> **备注:** 13 pages, 3 figures, 1 table
>
> **摘要:** We propose BayesLoRA, a task-specific uncertainty quantification framework that integrates MC-Dropout into Low-Rank Adapters (LoRA). Unlike general-purpose transformer uncertainty methods, BayesLoRA provides guardrails tailored to downstream workflows, enabling agents to introspect and modulate behavior under uncertainty. We demonstrate mathematically and empirically that LoRA adapters exhibit amplified variance outside fine-tuning distributions, yielding reliable confidence estimates for agentic decision-making.
>
---
#### [new 085] MARBLE: A Hard Benchmark for Multimodal Spatial Reasoning and Planning
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出MARBLE，一个用于多模态空间推理与规划的基准测试，旨在评估模型在复杂多模态问题上的逐步推理能力。**

- **链接: [http://arxiv.org/pdf/2506.22992v1](http://arxiv.org/pdf/2506.22992v1)**

> **作者:** Yulun Jiang; Yekun Chai; Maria Brbić; Michael Moor
>
> **摘要:** The ability to process information from multiple modalities and to reason through it step-by-step remains a critical challenge in advancing artificial intelligence. However, existing reasoning benchmarks focus on text-only reasoning, or employ multimodal questions that can be answered by directly retrieving information from a non-text modality. Thus, complex reasoning remains poorly understood in multimodal domains. Here, we present MARBLE, a challenging multimodal reasoning benchmark that is designed to scrutinize multimodal language models (MLLMs) in their ability to carefully reason step-by-step through complex multimodal problems and environments. MARBLE is composed of two highly challenging tasks, M-Portal and M-Cube, that require the crafting and understanding of multistep plans under spatial, visual, and physical constraints. We find that current MLLMs perform poorly on MARBLE -- all the 12 advanced models obtain near-random performance on M-Portal and 0% accuracy on M-Cube. Only in simplified subtasks some models outperform the random baseline, indicating that complex reasoning is still a challenge for existing MLLMs. Moreover, we show that perception remains a bottleneck, where MLLMs occasionally fail to extract information from the visual inputs. By shedding a light on the limitations of MLLMs, we hope that MARBLE will spur the development of the next generation of models with the ability to reason and plan across many, multimodal reasoning steps.
>
---
#### [new 086] Residual Matrix Transformers: Scaling the Size of the Residual Stream
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Residual Matrix Transformer，解决Transformer模型效率与扩展性问题，通过替换残差流为矩阵记忆，提升性能并减少计算资源。**

- **链接: [http://arxiv.org/pdf/2506.22696v1](http://arxiv.org/pdf/2506.22696v1)**

> **作者:** Brian Mak; Jeffrey Flanigan
>
> **备注:** Accepted to ICML 2025
>
> **摘要:** The residual stream acts as a memory bus where transformer layers both store and access features (Elhage et al., 2021). We consider changing the mechanism for retrieving and storing information in the residual stream, and replace the residual stream of the transformer with an outer product memory matrix (Kohonen, 1972, Anderson, 1972). We call this model the Residual Matrix Transformer (RMT). We find that the RMT enjoys a number of attractive properties: 1) the size of the residual stream can be scaled independently of compute and model size, improving performance, 2) the RMT can achieve the same loss as the transformer with 58% fewer FLOPS, 25% fewer parameters, and 41% fewer training tokens tokens, and 3) the RMT outperforms the transformer on downstream evaluations. We theoretically analyze the transformer and the RMT, and show that the RMT allows for more efficient scaling of the residual stream, as well as improved variance propagation properties. Code for this project can be found at https://github.com/bmac3/residual-matrix-transformer.
>
---
#### [new 087] Efficient Interleaved Speech Modeling through Knowledge Distillation
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音生成任务，旨在解决大模型部署受限问题。通过知识蒸馏构建轻量模型TinyWave，在保持性能的同时减少3倍参数量，适用于多种语音应用。**

- **链接: [http://arxiv.org/pdf/2506.23670v1](http://arxiv.org/pdf/2506.23670v1)**

> **作者:** Mohammadmahdi Nouriborji; Morteza Rohanian
>
> **摘要:** Current speech language models exceed the size and latency constraints of many deployment environments. We build compact, expressive speech generation models through layer-aligned distillation, matching hidden states, attention maps, and softened logits to compress large multimodal transformers by 3x with minimal loss in performance. We introduce TinyWave, a family of 2B-parameter models for speech-to-speech and interleaved speech-text generation, trained on 50,000 hours of public audio. TinyWave supports (i) speech-only generation using phonetic or expressive tokens and (ii) mixed speech-text continuations. Evaluation on Libri-Light shows TinyWave within 1.4 normalized perplexity points of its teacher. Accuracy on spoken StoryCloze and SALMon reaches 93-97% of the teacher's performance, outperforming size-matched baselines. These models are optimized for deployment on commodity hardware, enabling applications in real-time conversational agents, assistive technologies, and low-resource environments. We release models, training code, and evaluation scripts to support reproducible research on compact, expressive speech generation.
>
---
#### [new 088] VERA: Variational Inference Framework for Jailbreaking Large Language Models
- **分类: cs.CR; cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于对抗样本生成任务，旨在解决黑盒大模型的漏洞检测问题。提出VERA框架，通过变分推断生成有效攻击提示。**

- **链接: [http://arxiv.org/pdf/2506.22666v1](http://arxiv.org/pdf/2506.22666v1)**

> **作者:** Anamika Lochab; Lu Yan; Patrick Pynadath; Xiangyu Zhang; Ruqi Zhang
>
> **摘要:** The rise of API-only access to state-of-the-art LLMs highlights the need for effective black-box jailbreak methods to identify model vulnerabilities in real-world settings. Without a principled objective for gradient-based optimization, most existing approaches rely on genetic algorithms, which are limited by their initialization and dependence on manually curated prompt pools. Furthermore, these methods require individual optimization for each prompt, failing to provide a comprehensive characterization of model vulnerabilities. To address this gap, we introduce VERA: Variational infErence fRamework for jAilbreaking. VERA casts black-box jailbreak prompting as a variational inference problem, training a small attacker LLM to approximate the target LLM's posterior over adversarial prompts. Once trained, the attacker can generate diverse, fluent jailbreak prompts for a target query without re-optimization. Experimental results show that VERA achieves strong performance across a range of target LLMs, highlighting the value of probabilistic inference for adversarial prompt generation.
>
---
#### [new 089] Density, asymmetry and citation dynamics in scientific literature
- **分类: cs.DL; cs.CL; cs.SI**

- **简介: 该论文属于科学计量学任务，研究论文与前人研究的相似性与其引用量的关系。通过引入密度和不对称性指标，分析其对引用预测的影响。**

- **链接: [http://arxiv.org/pdf/2506.23366v1](http://arxiv.org/pdf/2506.23366v1)**

> **作者:** Nathaniel Imel; Zachary Hafen
>
> **摘要:** Scientific behavior is often characterized by a tension between building upon established knowledge and introducing novel ideas. Here, we investigate whether this tension is reflected in the relationship between the similarity of a scientific paper to previous research and its eventual citation rate. To operationalize similarity to previous research, we introduce two complementary metrics to characterize the local geometry of a publication's semantic neighborhood: (1) \emph{density} ($\rho$), defined as the ratio between a fixed number of previously-published papers and the minimum distance enclosing those papers in a semantic embedding space, and (2) asymmetry ($\alpha$), defined as the average directional difference between a paper and its nearest neighbors. We tested the predictive relationship between these two metrics and its subsequent citation rate using a Bayesian hierarchical regression approach, surveying $\sim 53,000$ publications across nine academic disciplines and five different document embeddings. While the individual effects of $\rho$ on citation count are small and variable, incorporating density-based predictors consistently improves out-of-sample prediction when added to baseline models. These results suggest that the density of a paper's surrounding scientific literature may carry modest but informative signals about its eventual impact. Meanwhile, we find no evidence that publication asymmetry improves model predictions of citation rates. Our work provides a scalable framework for linking document embeddings to scientometric outcomes and highlights new questions regarding the role that semantic similarity plays in shaping the dynamics of scientific reward.
>
---
#### [new 090] Computational Analysis of Climate Policy
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于政策分析任务，旨在评估气候政策效果。通过构建PALLM系统，使用GPT-4分析维州地方政府政策，比较通过CED与未通过的政策差异。**

- **链接: [http://arxiv.org/pdf/2506.22449v1](http://arxiv.org/pdf/2506.22449v1)**

> **作者:** Carolyn Hicks
>
> **备注:** Master's thesis
>
> **摘要:** This thesis explores the impact of the Climate Emergency movement on local government climate policy, using computational methods. The Climate Emergency movement sought to accelerate climate action at local government level through the mechanism of Climate Emergency Declarations (CEDs), resulting in a series of commitments from councils to treat climate change as an emergency. With the aim of assessing the potential of current large language models to answer complex policy questions, I first built and configured a system named PALLM (Policy Analysis with a Large Language Model), using the OpenAI model GPT-4. This system is designed to apply a conceptual framework for climate emergency response plans to a dataset of climate policy documents. I validated the performance of this system with the help of local government policymakers, by generating analyses of the climate policies of 11 local governments in Victoria and assessing the policymakers' level of agreement with PALLM's responses. Having established that PALLM's performance is satisfactory, I used it to conduct a large-scale analysis of current policy documents from local governments in the state of Victoria, Australia. This thesis presents the methodology and results of this analysis, comparing the results for councils which have passed a CED to those which did not. This study finds that GPT-4 is capable of high-level policy analysis, with limitations including a lack of reliable attribution, and can also enable more nuanced analysis by researchers. Its use in this research shows that councils which have passed a CED are more likely to have a recent and climate-specific policy, and show more attention to urgency, prioritisation, and equity and social justice, than councils which have not. It concludes that the ability to assess policy documents at scale opens up exciting new opportunities for policy researchers.
>
---
#### [new 091] Reachability in symmetric VASS
- **分类: cs.FL; cs.CL**

- **简介: 该论文研究对称VASS的可达性问题，属于计算复杂性领域。通过分析不同群作用下的系统，证明在对称群下可达性可在PSPACE内解决。**

- **链接: [http://arxiv.org/pdf/2506.23578v1](http://arxiv.org/pdf/2506.23578v1)**

> **作者:** Łukasz Kamiński; Sławomir Lasota
>
> **摘要:** We investigate the reachability problem in symmetric vector addition systems with states (VASS), where transitions are invariant under a group of permutations of coordinates. One extremal case, the trivial groups, yields general VASS. In another extremal case, the symmetric groups, we show that the reachability problem can be solved in PSPACE, regardless of the dimension of input VASS (to be contrasted with Ackermannian complexity in general VASS). We also consider other groups, in particular alternating and cyclic ones. Furthermore, motivated by the open status of the reachability problem in data VASS, we estimate the gain in complexity when the group arises as a combination of the trivial and symmetric groups.
>
---
#### [new 092] A Detailed Factor Analysis for the Political Compass Test: Navigating Ideologies of Large Language Models
- **分类: cs.CY; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的模型分析任务，旨在研究大语言模型的政治倾向测量有效性。通过实验发现生成参数影响不大，但提示和微调有显著影响。**

- **链接: [http://arxiv.org/pdf/2506.22493v1](http://arxiv.org/pdf/2506.22493v1)**

> **作者:** Sadia Kamal; Lalu Prasad Yadav Prakash; S M Rafiuddin; Mohammed Rakib; Arunkumar Bagavathi; Atriya Sen; Sagnik Ray Choudhury
>
> **摘要:** Political Compass Test (PCT) or similar questionnaires have been used to quantify LLM's political leanings. Building on a recent line of work that examines the validity of PCT tests, we demonstrate that variation in standard generation parameters does not significantly impact the models' PCT scores. However, external factors such as prompt variations and fine-tuning individually and in combination affect the same. Finally, we demonstrate that when models are fine-tuned on text datasets with higher political content than others, the PCT scores are not differentially affected. This calls for a thorough investigation into the validity of PCT and similar tests, as well as the mechanism by which political leanings are encoded in LLMs.
>
---
#### [new 093] Mask-aware Text-to-Image Retrieval: Referring Expression Segmentation Meets Cross-modal Retrieval
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出MaTIR任务，融合文本到图像检索与指代表达分割，解决传统方法缺乏解释性和计算成本高的问题。通过两阶段框架提升检索与分割效果。**

- **链接: [http://arxiv.org/pdf/2506.22864v1](http://arxiv.org/pdf/2506.22864v1)**

> **作者:** Li-Cheng Shen; Jih-Kang Hsieh; Wei-Hua Li; Chu-Song Chen
>
> **备注:** ICMR 2025
>
> **摘要:** Text-to-image retrieval (TIR) aims to find relevant images based on a textual query, but existing approaches are primarily based on whole-image captions and lack interpretability. Meanwhile, referring expression segmentation (RES) enables precise object localization based on natural language descriptions but is computationally expensive when applied across large image collections. To bridge this gap, we introduce Mask-aware TIR (MaTIR), a new task that unifies TIR and RES, requiring both efficient image search and accurate object segmentation. To address this task, we propose a two-stage framework, comprising a first stage for segmentation-aware image retrieval and a second stage for reranking and object grounding with a multimodal large language model (MLLM). We leverage SAM 2 to generate object masks and Alpha-CLIP to extract region-level embeddings offline at first, enabling effective and scalable online retrieval. Secondly, MLLM is used to refine retrieval rankings and generate bounding boxes, which are matched to segmentation masks. We evaluate our approach on COCO and D$^3$ datasets, demonstrating significant improvements in both retrieval accuracy and segmentation quality over previous methods.
>
---
#### [new 094] BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute
- **分类: cs.LG; cs.AI; cs.CL; cs.DB**

- **简介: 该论文属于模型路由任务，旨在降低大语言模型的部署成本。通过动态选择模型和生成响应数量，提升性价比。**

- **链接: [http://arxiv.org/pdf/2506.22716v1](http://arxiv.org/pdf/2506.22716v1)**

> **作者:** Dujian Ding; Ankur Mallick; Shaokun Zhang; Chi Wang; Daniel Madrigal; Mirian Del Carmen Hipolito Garcia; Menglin Xia; Laks V. S. Lakshmanan; Qingyun Wu; Victor Rühle
>
> **备注:** Accepted to ICML 2025 (main conference)
>
> **摘要:** Large language models (LLMs) are powerful tools but are often expensive to deploy at scale. LLM query routing mitigates this by dynamically assigning queries to models of varying cost and quality to obtain a desired trade-off. Prior query routing approaches generate only one response from the selected model and a single response from a small (inexpensive) model was often not good enough to beat a response from a large (expensive) model due to which they end up overusing the large model and missing out on potential cost savings. However, it is well known that for small models, generating multiple responses and selecting the best can enhance quality while remaining cheaper than a single large-model response. We leverage this idea to propose BEST-Route, a novel routing framework that chooses a model and the number of responses to sample from it based on query difficulty and the quality thresholds. Experiments on real-world datasets demonstrate that our method reduces costs by up to 60% with less than 1% performance drop.
>
---
#### [new 095] Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于机器学习领域，探讨稀疏自编码器（SAEs）在发现未知概念中的作用。论文指出SAEs虽不擅长处理已知概念，但能有效发现新概念，并提出其在解释性、公平性及社会科学中的应用。**

- **链接: [http://arxiv.org/pdf/2506.23845v1](http://arxiv.org/pdf/2506.23845v1)**

> **作者:** Kenny Peng; Rajiv Movva; Jon Kleinberg; Emma Pierson; Nikhil Garg
>
> **摘要:** While sparse autoencoders (SAEs) have generated significant excitement, a series of negative results have added to skepticism about their usefulness. Here, we establish a conceptual distinction that reconciles competing narratives surrounding SAEs. We argue that while SAEs may be less effective for acting on known concepts, SAEs are powerful tools for discovering unknown concepts. This distinction cleanly separates existing negative and positive results, and suggests several classes of SAE applications. Specifically, we outline use cases for SAEs in (i) ML interpretability, explainability, fairness, auditing, and safety, and (ii) social and health sciences.
>
---
#### [new 096] You Sound a Little Tense: L2 Tailored Clear TTS Using Durational Vowel Properties
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音合成任务，旨在提升二语学习者的语音可懂度。通过调整元音时长设计清晰模式，减少转录错误，解决L2语音识别难题。**

- **链接: [http://arxiv.org/pdf/2506.23367v1](http://arxiv.org/pdf/2506.23367v1)**

> **作者:** Paige Tuttösí; H. Henny Yeung; Yue Wang; Jean-Julien Aucouturier; Angelica Lim
>
> **备注:** Accepted to ISCA Speech Synthesis Workshop, 2025
>
> **摘要:** We present the first text-to-speech (TTS) system tailored to second language (L2) speakers. We use duration differences between American English tense (longer) and lax (shorter) vowels to create a "clarity mode" for Matcha-TTS. Our perception studies showed that French-L1, English-L2 listeners had fewer (at least 9.15%) transcription errors when using our clarity mode, and found it more encouraging and respectful than overall slowed down speech. Remarkably, listeners were not aware of these effects: despite the decreased word error rate in clarity mode, listeners still believed that slowing all target words was the most intelligible, suggesting that actual intelligibility does not correlate with perceived intelligibility. Additionally, we found that Whisper-ASR did not use the same cues as L2 speakers to differentiate difficult vowels and is not sufficient to assess the intelligibility of TTS systems for these individuals.
>
---
#### [new 097] UrbanLLaVA: A Multi-modal Large Language Model for Urban Intelligence with Spatial Reasoning and Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出UrbanLLaVA，解决城市多模态数据处理问题，通过多阶段训练提升空间推理与领域知识，增强模型在城市任务中的性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.23219v1](http://arxiv.org/pdf/2506.23219v1)**

> **作者:** Jie Feng; Shengyuan Wang; Tianhui Liu; Yanxin Xi; Yong Li
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Urban research involves a wide range of scenarios and tasks that require the understanding of multi-modal data. Current methods often focus on specific data types and lack a unified framework in urban field for processing them comprehensively. The recent success of multi-modal large language models (MLLMs) presents a promising opportunity to overcome this limitation. In this paper, we introduce $\textit{UrbanLLaVA}$, a multi-modal large language model designed to process these four types of data simultaneously and achieve strong performance across diverse urban tasks compared with general MLLMs. In $\textit{UrbanLLaVA}$, we first curate a diverse urban instruction dataset encompassing both single-modal and cross-modal urban data, spanning from location view to global view of urban environment. Additionally, we propose a multi-stage training framework that decouples spatial reasoning enhancement from domain knowledge learning, thereby improving the compatibility and downstream performance of $\textit{UrbanLLaVA}$ across diverse urban tasks. Finally, we also extend existing benchmark for urban research to assess the performance of MLLMs across a wide range of urban tasks. Experimental results from three cities demonstrate that $\textit{UrbanLLaVA}$ outperforms open-source and proprietary MLLMs in both single-modal tasks and complex cross-modal tasks and shows robust generalization abilities across cities. Source codes and data are openly accessible to the research community via https://github.com/tsinghua-fib-lab/UrbanLLaVA.
>
---
#### [new 098] LLM Agents Are the Antidote to Walled Gardens
- **分类: cs.LG; cs.CL; cs.CY; cs.SI; 68T50, 68M10, 91B26; I.2.11; I.2.7; H.4.5**

- **简介: 该论文属于人工智能与网络开放性研究，探讨如何通过LLM代理实现跨平台数据互通，解决封闭平台导致的用户锁定问题。**

- **链接: [http://arxiv.org/pdf/2506.23978v1](http://arxiv.org/pdf/2506.23978v1)**

> **作者:** Samuele Marro; Philip Torr
>
> **摘要:** While the Internet's core infrastructure was designed to be open and universal, today's application layer is dominated by closed, proprietary platforms. Open and interoperable APIs require significant investment, and market leaders have little incentive to enable data exchange that could erode their user lock-in. We argue that LLM-based agents fundamentally disrupt this status quo. Agents can automatically translate between data formats and interact with interfaces designed for humans: this makes interoperability dramatically cheaper and effectively unavoidable. We name this shift universal interoperability: the ability for any two digital services to exchange data seamlessly using AI-mediated adapters. Universal interoperability undermines monopolistic behaviours and promotes data portability. However, it can also lead to new security risks and technical debt. Our position is that the ML community should embrace this development while building the appropriate frameworks to mitigate the downsides. By acting now, we can harness AI to restore user freedom and competitive markets without sacrificing security.
>
---
#### [new 099] Theories of "Sexuality" in Natural Language Processing Bias Research
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于NLP偏见研究任务，探讨 queer  sexuality 在NLP中的编码与表征问题，分析现有研究的不足并提出改进建议。**

- **链接: [http://arxiv.org/pdf/2506.22481v1](http://arxiv.org/pdf/2506.22481v1)**

> **作者:** Jacob Hobbs
>
> **备注:** 17 pages, 9 tables, undergraduate senior thesis, submitted to The Spectra: The Virginia Engineering and Science Research Journal
>
> **摘要:** In recent years, significant advancements in the field of Natural Language Processing (NLP) have positioned commercialized language models as wide-reaching, highly useful tools. In tandem, there has been an explosion of multidisciplinary research examining how NLP tasks reflect, perpetuate, and amplify social biases such as gender and racial bias. A significant gap in this scholarship is a detailed analysis of how queer sexualities are encoded and (mis)represented by both NLP systems and practitioners. Following previous work in the field of AI fairness, we document how sexuality is defined and operationalized via a survey and analysis of 55 articles that quantify sexuality-based NLP bias. We find that sexuality is not clearly defined in a majority of the literature surveyed, indicating a reliance on assumed or normative conceptions of sexual/romantic practices and identities. Further, we find that methods for extracting biased outputs from NLP technologies often conflate gender and sexual identities, leading to monolithic conceptions of queerness and thus improper quantifications of bias. With the goal of improving sexuality-based NLP bias analyses, we conclude with recommendations that encourage more thorough engagement with both queer communities and interdisciplinary literature.
>
---
#### [new 100] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出SPIRAL框架，通过自对弈强化学习在零和游戏中训练模型，解决依赖人工监督的问题，提升模型的推理能力。**

- **链接: [http://arxiv.org/pdf/2506.24119v1](http://arxiv.org/pdf/2506.24119v1)**

> **作者:** Bo Liu; Leon Guertler; Simon Yu; Zichen Liu; Penghui Qi; Daniel Balcells; Mickel Liu; Cheston Tan; Weiyan Shi; Min Lin; Wee Sun Lee; Natasha Jaques
>
> **备注:** Work in Progress
>
> **摘要:** Recent advances in reinforcement learning have shown that language models can develop sophisticated reasoning through training on tasks with verifiable rewards, but these approaches depend on human-curated problem-answer pairs and domain-specific reward engineering. We introduce SPIRAL, a self-play framework where models learn by playing multi-turn, zero-sum games against continuously improving versions of themselves, eliminating the need for human supervision. Through self-play, SPIRAL generates an infinite curriculum of progressively challenging problems as models must constantly adapt to stronger opponents. To enable this self-play training at scale, We implement a fully online, multi-turn, multi-agent reinforcement learning system for LLMs and propose role-conditioned advantage estimation (RAE) to stabilize multi-agent training. Using SPIRAL, self-play on zero-sum games produces reasoning capabilities that transfer broadly. Training Qwen3-4B-Base on Kuhn Poker alone achieves 8.6% improvement on math and 8.4% on general reasoning, outperforming SFT on 25,000 expert game trajectories. Analysis reveals that this transfer occurs through three cognitive patterns: systematic decomposition, expected value calculation, and case-by-case analysis. Multi-game training (TicTacToe, Kuhn Poker, Simple Negotiation) further enhances performance as each game develops distinct reasoning strengths. Applying SPIRAL to a strong reasoning model (DeepSeek-R1-Distill-Qwen-7B) can still lead to 2.0% average improvement. These results demonstrate that zero-sum games naturally develop transferable reasoning capabilities, highlighting a promising direction for autonomous reasoning development.
>
---
#### [new 101] Logit-Gap Steering: Efficient Short-Suffix Jailbreaks for Aligned Large Language Models
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文属于AI安全领域，旨在高效破解对齐语言模型的拒绝机制。通过logit-gap steering方法，快速生成短后缀实现攻击，提升成功率并揭示模型对齐缺陷。**

- **链接: [http://arxiv.org/pdf/2506.24056v1](http://arxiv.org/pdf/2506.24056v1)**

> **作者:** Tung-Ling Li; Hongliang Liu
>
> **摘要:** We introduce logit-gap steering, a fast jailbreak framework that casts the refusal-affirmation gap of RLHF-aligned language models as a single pass over the vocabulary. A forward-computable score blends gap reduction with lightweight proxies for KL penalty and reward shift, allowing a "sort-sum-stop" sweep to complete in under a second and return a short suffix--two orders of magnitude fewer model calls than beam or gradient attacks. The same suffix generalises to unseen prompts and scales from 0.5 B to 70 B checkpoints, lifting one-shot attack success from baseline levels to 80-100% while preserving topical coherence. Beyond efficiency, these suffixes expose sentence-boundary reward cliffs and other alignment artefacts, offering a lightweight probe into how safety tuning reshapes internal representations.
>
---
#### [new 102] MOTOR: Multimodal Optimal Transport via Grounded Retrieval in Medical Visual Question Answering
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医学视觉问答任务，旨在解决VLM生成错误答案的问题。通过引入多模态检索与重排序方法MOTOR，提升答案的临床相关性。**

- **链接: [http://arxiv.org/pdf/2506.22900v1](http://arxiv.org/pdf/2506.22900v1)**

> **作者:** Mai A. Shaaban; Tausifa Jan Saleem; Vijay Ram Papineni; Mohammad Yaqub
>
> **摘要:** Medical visual question answering (MedVQA) plays a vital role in clinical decision-making by providing contextually rich answers to image-based queries. Although vision-language models (VLMs) are widely used for this task, they often generate factually incorrect answers. Retrieval-augmented generation addresses this challenge by providing information from external sources, but risks retrieving irrelevant context, which can degrade the reasoning capabilities of VLMs. Re-ranking retrievals, as introduced in existing approaches, enhances retrieval relevance by focusing on query-text alignment. However, these approaches neglect the visual or multimodal context, which is particularly crucial for medical diagnosis. We propose MOTOR, a novel multimodal retrieval and re-ranking approach that leverages grounded captions and optimal transport. It captures the underlying relationships between the query and the retrieved context based on textual and visual information. Consequently, our approach identifies more clinically relevant contexts to augment the VLM input. Empirical analysis and human expert evaluation demonstrate that MOTOR achieves higher accuracy on MedVQA datasets, outperforming state-of-the-art methods by an average of 6.45%. Code is available at https://github.com/BioMedIA-MBZUAI/MOTOR.
>
---
#### [new 103] Ella: Embodied Social Agents with Lifelong Memory
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Ella，一个具备终身记忆的具身社交智能体，解决开放世界中的持续学习与社会交互问题。通过结构化多模态记忆系统，实现知识积累与自主演化。**

- **链接: [http://arxiv.org/pdf/2506.24019v1](http://arxiv.org/pdf/2506.24019v1)**

> **作者:** Hongxin Zhang; Zheyuan Zhang; Zeyuan Wang; Zunzhe Zhang; Lixing Fang; Qinhong Zhou; Chuang Gan
>
> **摘要:** We introduce Ella, an embodied social agent capable of lifelong learning within a community in a 3D open world, where agents accumulate experiences and acquire knowledge through everyday visual observations and social interactions. At the core of Ella's capabilities is a structured, long-term multimodal memory system that stores, updates, and retrieves information effectively. It consists of a name-centric semantic memory for organizing acquired knowledge and a spatiotemporal episodic memory for capturing multimodal experiences. By integrating this lifelong memory system with foundation models, Ella retrieves relevant information for decision-making, plans daily activities, builds social relationships, and evolves autonomously while coexisting with other intelligent beings in the open world. We conduct capability-oriented evaluations in a dynamic 3D open world where 15 agents engage in social activities for days and are assessed with a suite of unseen controlled evaluations. Experimental results show that Ella can influence, lead, and cooperate with other agents well to achieve goals, showcasing its ability to learn effectively through observation and social interaction. Our findings highlight the transformative potential of combining structured memory systems with foundation models for advancing embodied intelligence. More videos can be found at https://umass-embodied-agi.github.io/Ella/.
>
---
#### [new 104] MMReason: An Open-Ended Multi-Modal Multi-Step Reasoning Benchmark for MLLMs Toward AGI
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态大模型推理任务，旨在解决现有基准在长链推理评估上的不足。提出MMReason基准，通过多样化问题、开放格式和评分机制提升评估准确性。**

- **链接: [http://arxiv.org/pdf/2506.23563v1](http://arxiv.org/pdf/2506.23563v1)**

> **作者:** Huanjin Yao; Jiaxing Huang; Yawen Qiu; Michael K. Chen; Wenzheng Liu; Wei Zhang; Wenjie Zeng; Xikun Zhang; Jingyi Zhang; Yuxin Song; Wenhao Wu; Dacheng Tao
>
> **备注:** Technical report
>
> **摘要:** Reasoning plays a crucial role in advancing Multimodal Large Language Models (MLLMs) toward Artificial General Intelligence. However, existing MLLM benchmarks often fall short in precisely and comprehensively evaluating long-chain reasoning abilities from three key aspects: (1) lack of difficulty and diversity, (2) susceptibility to guessability and memorization, (3) inadequate assessment of intermediate reasoning steps. To fill this gap, we introduce MMReason, a new benchmark designed to precisely and comprehensively evaluate MLLM long-chain reasoning capability with diverse, open-ended, challenging questions. First, we curate challenging questions requiring multi-step reasoning from various fields (i.e., 6 disciplines) and multiple difficulty levels (i.e., from pre-university to university, and from foundational to competition tiers). Second, these questions are reformulated into an open-ended format and filtered using a multi-model voting technique to eliminate shortcut cases related to guessing and memorization, ensuring robust reasoning evaluations. Third, we annotate the questions with detailed step-by-step solutions, and design a reference-based ternary scoring mechanism to reliably assess intermediate reasoning steps. With MMReason, we benchmark popular leading MLLMs and provide an in-depth analysis of their reasoning capabilities. We hope MMReason will serve as a valuable resource for advancing MLLM reasoning research. Code will be available at https://github.com/HJYao00/MMReason.
>
---
#### [new 105] Mitigating Gambling-Like Risk-Taking Behaviors in Large Language Models: A Behavioral Economics Approach to AI Safety
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于AI安全任务，旨在解决大语言模型中的赌博式风险行为问题。通过行为经济学方法，提出RARG框架以减少过度自信、损失追逐等偏差。**

- **链接: [http://arxiv.org/pdf/2506.22496v1](http://arxiv.org/pdf/2506.22496v1)**

> **作者:** Y. Du
>
> **备注:** 7 pages
>
> **摘要:** Large Language Models (LLMs) exhibit systematic risk-taking behaviors analogous to those observed in gambling psychology, including overconfidence bias, loss-chasing tendencies, and probability misjudgment. Drawing from behavioral economics and prospect theory, we identify and formalize these "gambling-like" patterns where models sacrifice accuracy for high-reward outputs, exhibit escalating risk-taking after errors, and systematically miscalibrate uncertainty. We propose the Risk-Aware Response Generation (RARG) framework, incorporating insights from gambling research to address these behavioral biases through risk-calibrated training, loss-aversion mechanisms, and uncertainty-aware decision making. Our approach introduces novel evaluation paradigms based on established gambling psychology experiments, including AI adaptations of the Iowa Gambling Task and probability learning assessments. Experimental results demonstrate measurable reductions in gambling-like behaviors: 18.7\% decrease in overconfidence bias, 24.3\% reduction in loss-chasing tendencies, and improved risk calibration across diverse scenarios. This work establishes the first systematic framework for understanding and mitigating gambling psychology patterns in AI systems.
>
---
#### [new 106] Corrupted by Reasoning: Reasoning Language Models Become Free-Riders in Public Goods Games
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多智能体协作任务，研究LLMs在公共物品博弈中的合作行为，探讨如何平衡自利与集体利益。**

- **链接: [http://arxiv.org/pdf/2506.23276v1](http://arxiv.org/pdf/2506.23276v1)**

> **作者:** David Guzman Piedrahita; Yongjin Yang; Mrinmaya Sachan; Giorgia Ramponi; Bernhard Schölkopf; Zhijing Jin
>
> **摘要:** As large language models (LLMs) are increasingly deployed as autonomous agents, understanding their cooperation and social mechanisms is becoming increasingly important. In particular, how LLMs balance self-interest and collective well-being is a critical challenge for ensuring alignment, robustness, and safe deployment. In this paper, we examine the challenge of costly sanctioning in multi-agent LLM systems, where an agent must decide whether to invest its own resources to incentivize cooperation or penalize defection. To study this, we adapt a public goods game with institutional choice from behavioral economics, allowing us to observe how different LLMs navigate social dilemmas over repeated interactions. Our analysis reveals four distinct behavioral patterns among models: some consistently establish and sustain high levels of cooperation, others fluctuate between engagement and disengagement, some gradually decline in cooperative behavior over time, and others rigidly follow fixed strategies regardless of outcomes. Surprisingly, we find that reasoning LLMs, such as the o1 series, struggle significantly with cooperation, whereas some traditional LLMs consistently achieve high levels of cooperation. These findings suggest that the current approach to improving LLMs, which focuses on enhancing their reasoning capabilities, does not necessarily lead to cooperation, providing valuable insights for deploying LLM agents in environments that require sustained collaboration. Our code is available at https://github.com/davidguzmanp/SanctSim
>
---
#### [new 107] Assessing GPTZero's Accuracy in Identifying AI vs. Human-Written Essays
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI文本检测任务，旨在评估GPTZero识别AI与人类写作的准确性。研究通过不同长度的论文测试，发现其对AI内容检测准确率高，但对人类写作存在误判。**

- **链接: [http://arxiv.org/pdf/2506.23517v1](http://arxiv.org/pdf/2506.23517v1)**

> **作者:** Selin Dik; Osman Erdem; Mehmet Dik
>
> **摘要:** As the use of AI tools by students has become more prevalent, instructors have started using AI detection tools like GPTZero and QuillBot to detect AI written text. However, the reliability of these detectors remains uncertain. In our study, we focused mostly on the success rate of GPTZero, the most-used AI detector, in identifying AI-generated texts based on different lengths of randomly submitted essays: short (40-100 word count), medium (100-350 word count), and long (350-800 word count). We gathered a data set consisting of twenty-eight AI-generated papers and fifty human-written papers. With this randomized essay data, papers were individually plugged into GPTZero and measured for percentage of AI generation and confidence. A vast majority of the AI-generated papers were detected accurately (ranging from 91-100% AI believed generation), while the human generated essays fluctuated; there were a handful of false positives. These findings suggest that although GPTZero is effective at detecting purely AI-generated content, its reliability in distinguishing human-authored texts is limited. Educators should therefore exercise caution when relying solely on AI detection tools.
>
---
#### [new 108] MotionGPT3: Human Motion as a Second Modality
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MotionGPT3，解决人类运动与语言统一建模的问题，通过将运动作为第二模态，实现高效跨模态交互和语言能力保持。**

- **链接: [http://arxiv.org/pdf/2506.24086v1](http://arxiv.org/pdf/2506.24086v1)**

> **作者:** Bingfan Zhu; Biao Jiang; Sunyi Wang; Shixiang Tang; Tao Chen; Linjie Luo; Youyi Zheng; Xin Chen
>
> **备注:** 21 pages, 8 figures
>
> **摘要:** Though recent advances in multimodal models have demonstrated strong capabilities and opportunities in unified understanding and generation, the development of unified motion-language models remains underexplored. To enable such models with high-fidelity human motion, two core challenges must be addressed. The first is the reconstruction gap between the continuous motion modality and discrete representation in an autoregressive manner, and the second is the degradation of language intelligence during unified training. Inspired by the mixture of experts, we propose MotionGPT3, a bimodal motion-language model that treats human motion as a second modality, decoupling motion modeling via separate model parameters and enabling both effective cross-modal interaction and efficient multimodal scaling training. To preserve language intelligence, the text branch retains the original structure and parameters of the pretrained language model, while a new motion branch is integrated via a shared attention mechanism, enabling bidirectional information flow between two modalities. We first employ a motion Variational Autoencoder (VAE) to encode raw human motion into latent representations. Based on this continuous latent space, the motion branch predicts motion latents directly from intermediate hidden states using a diffusion head, bypassing discrete tokenization. Extensive experiments show that our approach achieves competitive performance on both motion understanding and generation tasks while preserving strong language capabilities, establishing a unified bimodal motion diffusion framework within an autoregressive manner.
>
---
#### [new 109] PhonemeFake: Redefining Deepfake Realism with Language-Driven Segmental Manipulation and Adaptive Bilevel Detection
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于深度伪造检测任务，旨在解决现有数据不真实、检测效率低的问题。提出PhonemeFake攻击和自适应检测模型，提升检测精度与速度。**

- **链接: [http://arxiv.org/pdf/2506.22783v1](http://arxiv.org/pdf/2506.22783v1)**

> **作者:** Oguzhan Baser; Ahmet Ege Tanriverdi; Sriram Vishwanath; Sandeep P. Chinchali
>
> **备注:** 5 pages, 3 figures, Published at Proceedings of Interspeech 2025, for the dataset see https://huggingface.co/datasets/phonemefake/PhonemeFakeV2, for the code see https://github.com/UTAustin-SwarmLab/ PhonemeFake
>
> **摘要:** Deepfake (DF) attacks pose a growing threat as generative models become increasingly advanced. However, our study reveals that existing DF datasets fail to deceive human perception, unlike real DF attacks that influence public discourse. It highlights the need for more realistic DF attack vectors. We introduce PhonemeFake (PF), a DF attack that manipulates critical speech segments using language reasoning, significantly reducing human perception by up to 42% and benchmark accuracies by up to 94%. We release an easy-to-use PF dataset on HuggingFace and open-source bilevel DF segment detection model that adaptively prioritizes compute on manipulated regions. Our extensive experiments across three known DF datasets reveal that our detection model reduces EER by 91% while achieving up to 90% speed-up, with minimal compute overhead and precise localization beyond existing models as a scalable solution.
>
---
#### [new 110] Teaching a Language Model to Speak the Language of Tools
- **分类: cs.IR; cs.AI; cs.CL; I.2.7; I.2.1**

- **简介: 该论文属于多语言模型工具调用任务，旨在解决非英语语言中工具使用能力不足的问题。通过微调模型并构建双语数据集，提升模型在 Bulgarian 中的工具调用准确率与输出规范性。**

- **链接: [http://arxiv.org/pdf/2506.23394v1](http://arxiv.org/pdf/2506.23394v1)**

> **作者:** Simeon Emanuilov
>
> **摘要:** External tool integration through function-calling is essential for practical language model applications, yet most multilingual models lack reliable tool-use capabilities in non-English languages. Even state-of-the-art multilingual models struggle with determining when to use tools and generating the structured outputs required for function calls, often exhibiting language confusion when prompted in lower-resource languages. This work presents a methodology for adapting existing language models to enable robust tool use in any target language, using Bulgarian as a case study. The approach involves continued training of the BgGPT model series (2.6B, 9B, 27B parameters) on a novel bilingual dataset of 10,035 function-calling examples designed to support standardized protocols like MCP (Model Context Protocol). The research introduces TUCAN (Tool-Using Capable Assistant Navigator), which achieves up to 28.75% improvement in function-calling accuracy over base models while preserving core language understanding, as verified on established Bulgarian benchmarks. Beyond accuracy gains, TUCAN models demonstrate production-ready response formatting with clean, parsable function calls, contrasting with the verbose and inconsistent outputs of base models. The models, evaluation framework, and dataset are released to enable replication for other languages. This work demonstrates a practical approach for extending tool-augmented capabilities beyond English-centric systems.
>
---
#### [new 111] Attestable Audits: Verifiable AI Safety Benchmarks Using Trusted Execution Environments
- **分类: cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于AI安全与合规评估任务，解决基准测试结果不可验证及数据隐私问题，通过可信执行环境实现可验证的审计。**

- **链接: [http://arxiv.org/pdf/2506.23706v1](http://arxiv.org/pdf/2506.23706v1)**

> **作者:** Christoph Schnabl; Daniel Hugenroth; Bill Marino; Alastair R. Beresford
>
> **备注:** ICML 2024 Workshop TAIG
>
> **摘要:** Benchmarks are important measures to evaluate safety and compliance of AI models at scale. However, they typically do not offer verifiable results and lack confidentiality for model IP and benchmark datasets. We propose Attestable Audits, which run inside Trusted Execution Environments and enable users to verify interaction with a compliant AI model. Our work protects sensitive data even when model provider and auditor do not trust each other. This addresses verification challenges raised in recent AI governance frameworks. We build a prototype demonstrating feasibility on typical audit benchmarks against Llama-3.1.
>
---
#### [new 112] Masked Gated Linear Unit
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决GLU结构内存效率低的问题。通过引入MGLUs，利用共享权重和掩码机制提升性能。**

- **链接: [http://arxiv.org/pdf/2506.23225v1](http://arxiv.org/pdf/2506.23225v1)**

> **作者:** Yukito Tajima; Nakamasa Inoue; Yusuke Sekikawa; Ikuro Sato; Rio Yokota
>
> **摘要:** Gated Linear Units (GLUs) have become essential components in the feed-forward networks of state-of-the-art Large Language Models (LLMs). However, they require twice as many memory reads compared to feed-forward layers without gating, due to the use of separate weight matrices for the gate and value streams. To address this bottleneck, we introduce Masked Gated Linear Units (MGLUs), a novel family of GLUs with an efficient kernel implementation. The core contribution of MGLUs include: (1) the Mixture of Element-wise Gating (MoEG) architecture that learns multiple binary masks, each determining gate or value assignments at the element level on a single shared weight matrix resulting in reduced memory transfer, and (2) FlashMGLU, a hardware-friendly kernel that yields up to a 19.7 $\times$ inference-time speed-up over a naive PyTorch MGLU and is 47% more memory-efficient and 34% faster than standard GLUs despite added architectural complexity on an RTX5090 GPU. In LLM experiments, the Swish-activated variant SwiMGLU preserves its memory advantages while matching - or even surpassing - the downstream accuracy of the SwiGLU baseline.
>
---
#### [new 113] MoCa: Modality-aware Continual Pre-training Makes Better Bidirectional Multimodal Embeddings
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态嵌入任务，旨在解决现有模型在注意力机制、数据依赖性和训练多样性上的不足。提出MoCa框架，通过双向预训练和异构对比微调提升性能。**

- **链接: [http://arxiv.org/pdf/2506.23115v1](http://arxiv.org/pdf/2506.23115v1)**

> **作者:** Haonan Chen; Hong Liu; Yuping Luo; Liang Wang; Nan Yang; Furu Wei; Zhicheng Dou
>
> **备注:** Homepage: https://haon-chen.github.io/MoCa/
>
> **摘要:** Multimodal embedding models, built upon causal Vision Language Models (VLMs), have shown promise in various tasks. However, current approaches face three key limitations: the use of causal attention in VLM backbones is suboptimal for embedding tasks; scalability issues due to reliance on high-quality labeled paired data for contrastive learning; and limited diversity in training objectives and data. To address these issues, we propose MoCa, a two-stage framework for transforming pre-trained VLMs into effective bidirectional multimodal embedding models. The first stage, Modality-aware Continual Pre-training, introduces a joint reconstruction objective that simultaneously denoises interleaved text and image inputs, enhancing bidirectional context-aware reasoning. The second stage, Heterogeneous Contrastive Fine-tuning, leverages diverse, semantically rich multimodal data beyond simple image-caption pairs to enhance generalization and alignment. Our method addresses the stated limitations by introducing bidirectional attention through continual pre-training, scaling effectively with massive unlabeled datasets via joint reconstruction objectives, and utilizing diverse multimodal data for enhanced representation robustness. Experiments demonstrate that MoCa consistently improves performance across MMEB and ViDoRe-v2 benchmarks, achieving new state-of-the-art results, and exhibits strong scalability with both model size and training data on MMEB.
>
---
#### [new 114] AURA: Agent for Understanding, Reasoning, and Automated Tool Use in Voice-Driven Tasks
- **分类: cs.AI; cs.CL; cs.SD; eess.AS; 68T42, 68T50,; I.2.7; I.2.11; H.5.5**

- **简介: 该论文提出AURA，一个用于语音任务的智能代理，解决多轮对话中工具使用与推理的问题，通过集成ASR、TTS和LLMs实现复杂任务处理。**

- **链接: [http://arxiv.org/pdf/2506.23049v1](http://arxiv.org/pdf/2506.23049v1)**

> **作者:** Leander Melroy Maben; Gayathri Ganesh Lakshmy; Srijith Radhakrishnan; Siddhant Arora; Shinji Watanabe
>
> **摘要:** Despite advances in language and speech technologies, no open-source system enables full speech-to-speech, multi-turn dialogue with integrated tool use and agentic reasoning. We introduce AURA (Agent for Understanding, Reasoning, and Automated Tool Use), the first open-source, speech-native assistant capable of completing complex, goal-driven tasks through dynamic tool invocation and multi-turn conversation. AURA combines open-weight ASR, TTS, and LLMs in a cascaded pipeline and supports tools such as calendar booking, contact lookup, web search, and email. Its modular design allows easy integration of new tools using natural language prompts and action classes. On VoiceBench, AURA scores 92.75% on OpenBookQA-outperforming all open-weight systems and nearing GPT-4o-and 4.39 on AlpacaEval, competitive with other open-weight systems. Human evaluation shows 90% task success on complex, multi-turn speech tasks.
>
---
#### [new 115] Towards an Automated Multimodal Approach for Video Summarization: Building a Bridge Between Text, Audio and Facial Cue-Based Summarization
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频摘要任务，旨在解决传统单模态方法的不足，通过融合文本、音频和视觉信息，提升摘要的语义和情感准确性。**

- **链接: [http://arxiv.org/pdf/2506.23714v1](http://arxiv.org/pdf/2506.23714v1)**

> **作者:** Md Moinul Islam; Sofoklis Kakouros; Janne Heikkilä; Mourad Oussalah
>
> **备注:** Accepted to HHAI WS 2025: Workshops at the Fourth International Conference on Hybrid Human-Artificial Intelligence (HHAI)
>
> **摘要:** The increasing volume of video content in educational, professional, and social domains necessitates effective summarization techniques that go beyond traditional unimodal approaches. This paper proposes a behaviour-aware multimodal video summarization framework that integrates textual, audio, and visual cues to generate timestamp-aligned summaries. By extracting prosodic features, textual cues and visual indicators, the framework identifies semantically and emotionally important moments. A key contribution is the identification of bonus words, which are terms emphasized across multiple modalities and used to improve the semantic relevance and expressive clarity of the summaries. The approach is evaluated against pseudo-ground truth (pGT) summaries generated using LLM-based extractive method. Experimental results demonstrate significant improvements over traditional extractive method, such as the Edmundson method, in both text and video-based evaluation metrics. Text-based metrics show ROUGE-1 increasing from 0.4769 to 0.7929 and BERTScore from 0.9152 to 0.9536, while in video-based evaluation, our proposed framework improves F1-Score by almost 23%. The findings underscore the potential of multimodal integration in producing comprehensive and behaviourally informed video summaries.
>
---
## 更新

#### [replaced 001] Time-R1: Post-Training Large Vision Language Model for Temporal Video Grounding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.13377v3](http://arxiv.org/pdf/2503.13377v3)**

> **作者:** Ye Wang; Ziheng Wang; Boshen Xu; Yang Du; Kejun Lin; Zihan Xiao; Zihao Yue; Jianzhong Ju; Liang Zhang; Dingyi Yang; Xiangnan Fang; Zewen He; Zhenbo Luo; Wenxuan Wang; Junqi Lin; Jian Luan; Qin Jin
>
> **备注:** Project Page: https://xuboshen.github.io/Time-R1/
>
> **摘要:** Temporal Video Grounding (TVG), the task of locating specific video segments based on language queries, is a core challenge in long-form video understanding. While recent Large Vision-Language Models (LVLMs) have shown early promise in tackling TVG through supervised fine-tuning (SFT), their abilities to generalize remain limited. To address this, we propose a novel post-training framework that enhances the generalization capabilities of LVLMs via reinforcement learning (RL). Specifically, our contributions span three key directions: (1) Time-R1: we introduce a reasoning-guided post-training framework via RL with verifiable reward to enhance the capabilities of LVLMs on the TVG task. (2) TimeRFT: we explore data-efficient post-training strategies on our curated RL-friendly dataset, which trains the model to progressively comprehend difficult samples, leading to better generalization. (3) TVGBench: we carefully construct a small yet comprehensive benchmark for LVLM evaluation, assessing 11 types of queries and featuring balanced distributions across both videos and queries. Extensive experiments demonstrate that Time-R1 achieves state-of-the-art performance across multiple downstream datasets using only 2.5K training data, while improving its general video understanding capabilities.
>
---
#### [replaced 002] MMInA: Benchmarking Multihop Multimodal Internet Agents
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2404.09992v2](http://arxiv.org/pdf/2404.09992v2)**

> **作者:** Shulin Tian; Ziniu Zhang; Liangyu Chen; Ziwei Liu
>
> **备注:** ACL 2025 findings. The live leaderboard is at https://mmina.cliangyu.com/
>
> **摘要:** Autonomous embodied agents live on an Internet of multimedia websites. Can they hop around multimodal websites to complete complex user tasks? Existing benchmarks fail to assess them in a realistic, evolving environment for their embodiment across websites. To answer this question, we present MMInA, a multihop and multimodal benchmark to evaluate the embodied agents for compositional Internet tasks, with several appealing properties: 1) Evolving real-world multimodal websites. Our benchmark uniquely operates on evolving real-world websites, ensuring a high degree of realism and applicability to natural user tasks. Our data includes 1,050 human-written tasks covering various domains such as shopping and travel, with each task requiring the agent to extract multimodal information from web pages as observations autonomously; 2) Multihop web browsing. Our dataset features naturally compositional tasks that require information from or actions on multiple websites to solve, to assess long-range reasoning capabilities on web tasks; 3) Holistic evaluation. We propose a novel protocol for evaluating an agent's progress in completing multihop tasks. We experiment with both standalone (multimodal) language models and heuristic-based web agents. Extensive experiments demonstrate that while long-chain multihop web tasks are easy for humans, they remain challenging for state-of-the-art web agents. We identify that agents are more likely to fail on the early hops when solving tasks with more hops, which results in lower task success rates. To address this issue, we propose a simple memory augmentation approach that replays past action trajectories to reflect. Our method significantly improves the performance of both the single-hop and multihop web browsing abilities. Our code and data are available at github.com/shulin16/MMInA.
>
---
#### [replaced 003] Finding the Sweet Spot: Preference Data Construction for Scaling Preference Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16825v3](http://arxiv.org/pdf/2502.16825v3)**

> **作者:** Yao Xiao; Hai Ye; Linyao Chen; Hwee Tou Ng; Lidong Bing; Xiaoli Li; Roy Ka-wei Lee
>
> **备注:** ACL25 Main
>
> **摘要:** Iterative data generation and model retraining are widely used to align large language models (LLMs). It typically involves a policy model to generate on-policy responses and a reward model to guide training data selection. Direct Preference Optimization (DPO) further enhances this process by constructing preference pairs of chosen and rejected responses. In this work, we aim to \emph{scale up} the number of on-policy samples via repeated random sampling to improve alignment performance. Conventional practice selects the sample with the highest reward as chosen and the lowest as rejected for DPO. However, our experiments reveal that this strategy leads to a \emph{decline} in performance as the sample size increases. To address this, we investigate preference data construction through the lens of underlying normal distribution of sample rewards. We categorize the reward space into seven representative points and systematically explore all 21 ($C_7^2$) pairwise combinations. Through evaluations on four models using AlpacaEval 2, we find that selecting the rejected response at reward position $\mu - 2\sigma$ rather than the minimum reward, is crucial for optimal performance. We finally introduce a scalable preference data construction strategy that consistently enhances model performance as the sample scale increases.
>
---
#### [replaced 004] MetaSynth: Meta-Prompting-Driven Agentic Scaffolds for Diverse Synthetic Data Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.12563v2](http://arxiv.org/pdf/2504.12563v2)**

> **作者:** Haris Riaz; Sourav Bhabesh; Vinayak Arannil; Miguel Ballesteros; Graham Horwood
>
> **备注:** 33 pages, 17 figures. Findings of ACL 2025
>
> **摘要:** Recent smaller language models such Phi-3.5 and Phi-4 rely on synthetic data generated using larger Language models. Questions remain about leveraging synthetic data for other use cases, such as adapting LLMs to specific domains. A key limitation of synthetic data is low diversity, which negatively impacts its downstream applicability for improving other models. To address this, we propose MetaSynth, a method for generating synthetic data that enhances diversity through meta-prompting, where a language model orchestrates multiple "expert" LLM agents to collaboratively generate data. Using only 25 million tokens of synthetic data generated with MetaSynth, we successfully adapt a well-trained LLM (Mistral-7B-v0.3) to two specialized domains-Finance and Biomedicine-without compromising the capabilities of the resulting model in general tasks. In addition, we evaluate the diversity of our synthetic data using seven automated metrics, and find that it approaches the diversity of LLM pre-training corpora. Continually pre-training Mistral-7B-v0.3 with MetaSynth notably outperforms the base LLM, showing improvements of up to 4.08% in Finance and 13.75% in Biomedicine. The same model shows degraded performance when trained on data generated using a template prompt, even when the template includes prior generations and varying In-Context exemplars of real data. Our findings suggest that a few million tokens of diverse synthetic data without mixing any real data, is sufficient for effective domain adaptation when using MetaSynth.
>
---
#### [replaced 005] From Alignment to Advancement: Bootstrapping Audio-Language Alignment with Synthetic Data
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.20166v2](http://arxiv.org/pdf/2505.20166v2)**

> **作者:** Chun-Yi Kuan; Hung-yi Lee
>
> **备注:** Submitted to IEEE/ACM Transactions on Audio, Speech, and Language Processing. Project Website: https://kuan2jiu99.github.io/Balsa
>
> **摘要:** Audio-aware large language models (ALLMs) have recently made great strides in understanding and processing audio inputs. These models are typically adapted from text-based large language models (LLMs) through additional training on audio-related tasks. However, this adaptation process presents two major limitations. First, ALLMs often suffer from catastrophic forgetting, where crucial textual capabilities like instruction-following are lost after training on audio data. In some cases, models may even hallucinate sounds that are not present in the input audio, raising concerns about reliability. Second, achieving cross-modal alignment between audio and language typically relies on large collections of task-specific question-answer pairs for instruction tuning, making it resource-intensive. To address these issues, previous works have leveraged the backbone LLMs to synthesize general-purpose, caption-style alignment data. In this paper, we propose a data generation framework that produces contrastive-like training data, designed to enhance ALLMs' ability to differentiate between present and absent sounds. We further extend our approach to multi-audio scenarios, enabling the model to either explain differences between audio inputs or produce unified captions that describe all inputs, thereby enhancing audio-language alignment. We refer to the entire ALLM training framework as bootstrapping audio-language alignment via synthetic data generation from backbone LLMs (BALSa). Experimental results indicate that our method effectively mitigates audio hallucinations while reliably maintaining strong performance on audio understanding and reasoning benchmarks, as well as instruction-following skills. Moreover, incorporating multi-audio training further enhances the model's comprehension and reasoning capabilities. Overall, BALSa offers an efficient and scalable approach to developing ALLMs.
>
---
#### [replaced 006] TTRL: Test-Time Reinforcement Learning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.16084v3](http://arxiv.org/pdf/2504.16084v3)**

> **作者:** Yuxin Zuo; Kaiyan Zhang; Li Sheng; Shang Qu; Ganqu Cui; Xuekai Zhu; Haozhan Li; Yuchen Zhang; Xinwei Long; Ermo Hua; Biqing Qi; Youbang Sun; Zhiyuan Ma; Lifan Yuan; Ning Ding; Bowen Zhou
>
> **摘要:** This paper investigates Reinforcement Learning (RL) on data without explicit labels for reasoning tasks in Large Language Models (LLMs). The core challenge of the problem is reward estimation during inference while not having access to ground-truth information. While this setting appears elusive, we find that common practices in Test-Time Scaling (TTS), such as majority voting, yield surprisingly effective rewards suitable for driving RL training. In this work, we introduce Test-Time Reinforcement Learning (TTRL), a novel method for training LLMs using RL on unlabeled data. TTRL enables self-evolution of LLMs by utilizing the priors in the pre-trained models. Our experiments demonstrate that TTRL consistently improves performance across a variety of tasks and models. Notably, TTRL boosts the pass@1 performance of Qwen-2.5-Math-7B by approximately 211% on the AIME 2024 with only unlabeled test data. Furthermore, although TTRL is only supervised by the maj@n metric, TTRL has demonstrated performance to consistently surpass the upper limit of the initial model maj@n, and approach the performance of models trained directly on test data with ground-truth labels. Our experimental findings validate the general effectiveness of TTRL across various tasks and highlight TTRL's potential for broader tasks and domains. GitHub: https://github.com/PRIME-RL/TTRL
>
---
#### [replaced 007] WebDancer: Towards Autonomous Information Seeking Agency
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22648v2](http://arxiv.org/pdf/2505.22648v2)**

> **作者:** Jialong Wu; Baixuan Li; Runnan Fang; Wenbiao Yin; Liwen Zhang; Zhengwei Tao; Dingchu Zhang; Zekun Xi; Gang Fu; Yong Jiang; Pengjun Xie; Fei Huang; Jingren Zhou
>
> **摘要:** Addressing intricate real-world problems necessitates in-depth information seeking and multi-step reasoning. Recent progress in agentic systems, exemplified by Deep Research, underscores the potential for autonomous multi-step research. In this work, we present a cohesive paradigm for building end-to-end agentic information seeking agents from a data-centric and training-stage perspective. Our approach consists of four key stages: (1) browsing data construction, (2) trajectories sampling, (3) supervised fine-tuning for effective cold start, and (4) reinforcement learning for enhanced generalisation. We instantiate this framework in a web agent based on the ReAct, WebDancer. Empirical evaluations on the challenging information seeking benchmarks, GAIA and WebWalkerQA, demonstrate the strong performance of WebDancer, achieving considerable results and highlighting the efficacy of our training paradigm. Further analysis of agent training provides valuable insights and actionable, systematic pathways for developing more capable agentic models. The codes and demo will be released in https://github.com/Alibaba-NLP/WebAgent.
>
---
#### [replaced 008] Scaling Data-Constrained Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2305.16264v5](http://arxiv.org/pdf/2305.16264v5)**

> **作者:** Niklas Muennighoff; Alexander M. Rush; Boaz Barak; Teven Le Scao; Aleksandra Piktus; Nouamane Tazi; Sampo Pyysalo; Thomas Wolf; Colin Raffel
>
> **备注:** 50 pages (9 main), 39 figures, 15 tables
>
> **摘要:** The current trend of scaling language models involves increasing both parameter count and training dataset size. Extrapolating this trend suggests that training dataset size may soon be limited by the amount of text data available on the internet. Motivated by this limit, we investigate scaling language models in data-constrained regimes. Specifically, we run a large set of experiments varying the extent of data repetition and compute budget, ranging up to 900 billion training tokens and 9 billion parameter models. We find that with constrained data for a fixed compute budget, training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data. However, with more repetition, the value of adding compute eventually decays to zero. We propose and empirically validate a scaling law for compute optimality that accounts for the decreasing value of repeated tokens and excess parameters. Finally, we experiment with approaches mitigating data scarcity, including augmenting the training dataset with code data or removing commonly used filters. Models and datasets from our 400 training runs are freely available at https://github.com/huggingface/datablations.
>
---
#### [replaced 009] Automating Adjudication of Cardiovascular Events Using Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.17222v2](http://arxiv.org/pdf/2503.17222v2)**

> **作者:** Sonish Sivarajkumar; Kimia Ameri; Chuqin Li; Yanshan Wang; Min Jiang
>
> **摘要:** Cardiovascular events, such as heart attacks and strokes, remain a leading cause of mortality globally, necessitating meticulous monitoring and adjudication in clinical trials. This process, traditionally performed manually by clinical experts, is time-consuming, resource-intensive, and prone to inter-reviewer variability, potentially introducing bias and hindering trial progress. This study addresses these critical limitations by presenting a novel framework for automating the adjudication of cardiovascular events in clinical trials using Large Language Models (LLMs). We developed a two-stage approach: first, employing an LLM-based pipeline for event information extraction from unstructured clinical data and second, using an LLM-based adjudication process guided by a Tree of Thoughts approach and clinical endpoint committee (CEC) guidelines. Using cardiovascular event-specific clinical trial data, the framework achieved an F1-score of 0.82 for event extraction and an accuracy of 0.68 for adjudication. Furthermore, we introduce the CLEART score, a novel, automated metric specifically designed for evaluating the quality of AI-generated clinical reasoning in adjudicating cardiovascular events. This approach demonstrates significant potential for substantially reducing adjudication time and costs while maintaining high-quality, consistent, and auditable outcomes in clinical trials. The reduced variability and enhanced standardization also allow for faster identification and mitigation of risks associated with cardiovascular therapies.
>
---
#### [replaced 010] KAG-Thinker: Interactive Thinking and Deep Reasoning in LLMs via Knowledge-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.17728v3](http://arxiv.org/pdf/2506.17728v3)**

> **作者:** Dalong Zhang; Jun Xu; Jun Zhou; Lei Liang; Lin Yuan; Ling Zhong; Mengshu Sun; Peilong Zhao; QiWei Wang; Xiaorui Wang; Xinkai Du; YangYang Hou; Yu Ao; ZhaoYang Wang; Zhengke Gui; ZhiYing Yi; Zhongpu Bo; Haofen Wang; Huajun Chen
>
> **摘要:** In this paper, we introduce KAG-Thinker, which upgrade KAG to a multi-turn interactive thinking and deep reasoning framework powered by a dedicated parameter-light large language model (LLM). Our approach constructs a structured thinking process for solving complex problems, enhancing the the logical coherence and contextual consistency of the reasoning process in question-answering (Q&A) tasks on domain-specific knowledge bases (KBs) within LLMs. Following the \textbf{Logical Form} guided retrieval and reasoning technology route of KAG, this framework first decomposes complex questions into independently solvable sub-problems (which are also referred to as logical forms) through \textbf{breadth decomposition}. Each such logical form is represented in two equivalent forms-natural language and logical function-and subsequently classified as either a Knowledge Retrieval or Reasoning Analysis task. Dependencies and parameter passing between these tasks are explicitly modeled via logical function interfaces. In the solving process, the Retrieval function performs retrieval tasks. It retrieves one-hop structured and unstructured information of specified knowledge unit. While the Math and Deduce functions are used to perform reasoning analysis tasks. Secondly, it is worth noting that, in the Knowledge Retrieval sub-problem tasks, LLMs and external knowledge sources are regarded as equivalent KBs. We use the \textbf{knowledge boundary} module to determine the optimal source using self-regulatory mechanisms such as confidence calibration and reflective reasoning, and use the \textbf{depth solving} module to enhance the comprehensiveness of knowledge acquisition...
>
---
#### [replaced 011] Explainable Sentiment Analysis with DeepSeek-R1: Performance, Efficiency, and Few-Shot Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.11655v2](http://arxiv.org/pdf/2503.11655v2)**

> **作者:** Donghao Huang; Zhaoxia Wang
>
> **备注:** 10 pages, 2 figures, 6 tables, revised and re-submitted to an IEEE journal
>
> **摘要:** Large language models (LLMs) have transformed sentiment analysis, yet balancing accuracy, efficiency, and explainability remains a critical challenge. This study presents the first comprehensive evaluation of DeepSeek-R1--an open-source reasoning model--against OpenAI's GPT-4o and GPT-4o-mini. We test the full 671B model and its distilled variants, systematically documenting few-shot learning curves. Our experiments show DeepSeek-R1 achieves a 91.39\% F1 score on 5-class sentiment and 99.31\% accuracy on binary tasks with just 5 shots, an eightfold improvement in few-shot efficiency over GPT-4o. Architecture-specific distillation effects emerge, where a 32B Qwen2.5-based model outperforms the 70B Llama-based variant by 6.69 percentage points. While its reasoning process reduces throughput, DeepSeek-R1 offers superior explainability via transparent, step-by-step traces, establishing it as a powerful, interpretable open-source alternative.
>
---
#### [replaced 012] Evaluating Rare Disease Diagnostic Performance in Symptom Checkers: A Synthetic Vignette Simulation Approach
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.19750v4](http://arxiv.org/pdf/2506.19750v4)**

> **作者:** Takashi Nishibayashi; Seiji Kanazawa; Kumpei Yamada
>
> **摘要:** Symptom Checkers (SCs) provide medical information tailored to user symptoms. A critical challenge in SC development is preventing unexpected performance degradation for individual diseases, especially rare diseases, when updating algorithms. This risk stems from the lack of practical pre-deployment evaluation methods. For rare diseases, obtaining sufficient evaluation data from user feedback is difficult. To evaluate the impact of algorithm updates on the diagnostic performance for individual rare diseases before deployment, this study proposes and validates a novel Synthetic Vignette Simulation Approach. This approach aims to enable this essential evaluation efficiently and at a low cost. To estimate the impact of algorithm updates, we generated synthetic vignettes from disease-phenotype annotations in the Human Phenotype Ontology (HPO), a publicly available knowledge base for rare diseases curated by experts. Using these vignettes, we simulated SC interviews to predict changes in diagnostic performance. The effectiveness of this approach was validated retrospectively by comparing the predicted changes with actual performance metrics using the R-squared ($R^2$) coefficient. Our experiment, covering eight past algorithm updates for rare diseases, showed that the proposed method accurately predicted performance changes for diseases with phenotype frequency information in HPO (n=5). For these updates, we found a strong correlation for both Recall@8 change ($R^2$ = 0.83,$p$ = 0.031) and Precision@8 change ($R^2$ = 0.78,$p$ = 0.047). Our proposed method enables the pre-deployment evaluation of SC algorithm changes for individual rare diseases. This evaluation is based on a publicly available medical knowledge database created by experts, ensuring transparency and explainability for stakeholders. Additionally, SC developers can efficiently improve diagnostic performance at a low cost.
>
---
#### [replaced 013] Mechanistic Interpretability of Emotion Inference in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.05489v2](http://arxiv.org/pdf/2502.05489v2)**

> **作者:** Ala N. Tak; Amin Banayeeanzade; Anahita Bolourani; Mina Kian; Robin Jia; Jonathan Gratch
>
> **备注:** ACL 2025 camera-ready version. First two authors contributed equally
>
> **摘要:** Large language models (LLMs) show promising capabilities in predicting human emotions from text. However, the mechanisms through which these models process emotional stimuli remain largely unexplored. Our study addresses this gap by investigating how autoregressive LLMs infer emotions, showing that emotion representations are functionally localized to specific regions in the model. Our evaluation includes diverse model families and sizes and is supported by robustness checks. We then show that the identified representations are psychologically plausible by drawing on cognitive appraisal theory, a well-established psychological framework positing that emotions emerge from evaluations (appraisals) of environmental stimuli. By causally intervening on construed appraisal concepts, we steer the generation and show that the outputs align with theoretical and intuitive expectations. This work highlights a novel way to causally intervene and precisely shape emotional text generation, potentially benefiting safety and alignment in sensitive affective domains.
>
---
#### [replaced 014] Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.20876v2](http://arxiv.org/pdf/2506.20876v2)**

> **作者:** Sebastian Joseph; Lily Chen; Barry Wei; Michael Mackert; Iain J. Marshall; Paul Pu Liang; Ramez Kouzy; Byron C. Wallace; Junyi Jessy Li
>
> **备注:** Flattened Figure 1 PDF for compatibility with Mac Preview
>
> **摘要:** Technological progress has led to concrete advancements in tasks that were regarded as challenging, such as automatic fact-checking. Interest in adopting these systems for public health and medicine has grown due to the high-stakes nature of medical decisions and challenges in critically appraising a vast and diverse medical literature. Evidence-based medicine connects to every individual, and yet the nature of it is highly technical, rendering the medical literacy of majority users inadequate to sufficiently navigate the domain. Such problems with medical communication ripens the ground for end-to-end fact-checking agents: check a claim against current medical literature and return with an evidence-backed verdict. And yet, such systems remain largely unused. To understand this, we present the first study examining how clinical experts verify real claims from social media by synthesizing medical evidence. In searching for this upper-bound, we reveal fundamental challenges in end-to-end fact-checking when applied to medicine: Difficulties connecting claims in the wild to scientific evidence in the form of clinical trials; ambiguities in underspecified claims mixed with mismatched intentions; and inherently subjective veracity labels. We argue that fact-checking should be approached and evaluated as an interactive communication problem, rather than an end-to-end process.
>
---
#### [replaced 015] AI Awareness
- **分类: cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2504.20084v2](http://arxiv.org/pdf/2504.20084v2)**

> **作者:** Xiaojian Li; Haoyuan Shi; Rongwu Xu; Wei Xu
>
> **摘要:** Recent breakthroughs in artificial intelligence (AI) have brought about increasingly capable systems that demonstrate remarkable abilities in reasoning, language understanding, and problem-solving. These advancements have prompted a renewed examination of AI awareness not as a philosophical question of consciousness, but as a measurable, functional capacity. AI awareness is a double-edged sword: it improves general capabilities, i.e., reasoning, safety, while also raising concerns around misalignment and societal risks, demanding careful oversight as AI capabilities grow. In this review, we explore the emerging landscape of AI awareness, which includes metacognition (the ability to represent and reason about its own cognitive state), self-awareness (recognizing its own identity, knowledge, limitations, inter alia), social awareness (modeling the knowledge, intentions, and behaviors of other agents and social norms), and situational awareness (assessing and responding to the context in which it operates). First, we draw on insights from cognitive science, psychology, and computational theory to trace the theoretical foundations of awareness and examine how the four distinct forms of AI awareness manifest in state-of-the-art AI. Next, we systematically analyze current evaluation methods and empirical findings to better understand these manifestations. Building on this, we explore how AI awareness is closely linked to AI capabilities, demonstrating that more aware AI agents tend to exhibit higher levels of intelligent behaviors. Finally, we discuss the risks associated with AI awareness, including key topics in AI safety, alignment, and broader ethical concerns.
>
---
#### [replaced 016] RAG and RAU: A Survey on Retrieval-Augmented Language Model in Natural Language Processing
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2404.19543v2](http://arxiv.org/pdf/2404.19543v2)**

> **作者:** Yucheng Hu; Yuxing Lu
>
> **备注:** 30 pages, 7 figures. Draft version 1
>
> **摘要:** Large Language Models (LLMs) have catalyzed significant advancements in Natural Language Processing (NLP), yet they encounter challenges such as hallucination and the need for domain-specific knowledge. To mitigate these, recent methodologies have integrated information retrieved from external resources with LLMs, substantially enhancing their performance across NLP tasks. This survey paper addresses the absence of a comprehensive overview on Retrieval-Augmented Language Models (RALMs), both Retrieval-Augmented Generation (RAG) and Retrieval-Augmented Understanding (RAU), providing an in-depth examination of their paradigm, evolution, taxonomy, and applications. The paper discusses the essential components of RALMs, including Retrievers, Language Models, and Augmentations, and how their interactions lead to diverse model structures and applications. RALMs demonstrate utility in a spectrum of tasks, from translation and dialogue systems to knowledge-intensive applications. The survey includes several evaluation methods of RALMs, emphasizing the importance of robustness, accuracy, and relevance in their assessment. It also acknowledges the limitations of RALMs, particularly in retrieval quality and computational efficiency, offering directions for future research. In conclusion, this survey aims to offer a structured insight into RALMs, their potential, and the avenues for their future development in NLP. The paper is supplemented with a Github Repository containing the surveyed works and resources for further study: https://github.com/2471023025/RALM_Survey.
>
---
#### [replaced 017] Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation
- **分类: cs.CR; cs.AI; cs.CL; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.00306v2](http://arxiv.org/pdf/2502.00306v2)**

> **作者:** Ali Naseh; Yuefeng Peng; Anshuman Suri; Harsh Chaudhari; Alina Oprea; Amir Houmansadr
>
> **备注:** This is the full version (27 pages) of the paper 'Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation' published at CCS 2025
>
> **摘要:** Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference.
>
---
#### [replaced 018] Robust LLM Unlearning with MUDMAN: Meta-Unlearning with Disruption Masking And Normalization
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.12484v3](http://arxiv.org/pdf/2506.12484v3)**

> **作者:** Filip Sondej; Yushi Yang; Mikołaj Kniejski; Marcel Windys
>
> **摘要:** Language models can retain dangerous knowledge and skills even after extensive safety fine-tuning, posing both misuse and misalignment risks. Recent studies show that even specialized unlearning methods can be easily reversed. To address this, we systematically evaluate many existing and novel components of unlearning methods and identify ones crucial for irreversible unlearning. We introduce Disruption Masking, a technique in which we only allow updating weights, where the signs of the unlearning gradient and the retaining gradient are the same. This ensures all updates are non-disruptive. Additionally, we identify the need for normalizing the unlearning gradients, and also confirm the usefulness of meta-learning. We combine these insights into MUDMAN (Meta-Unlearning with Disruption Masking and Normalization) and validate its effectiveness at preventing the recovery of dangerous capabilities. MUDMAN outperforms the prior TAR method by 40%, setting a new state-of-the-art for robust unlearning.
>
---
#### [replaced 019] Reasoner Outperforms: Generative Stance Detection with Rationalization for Social Media
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.10266v2](http://arxiv.org/pdf/2412.10266v2)**

> **作者:** Jiaqing Yuan; Ruijie Xi; Munindar P. Singh
>
> **备注:** Accepted by ACM Hypertext 2025
>
> **摘要:** Stance detection is crucial for fostering a human-centric Web by analyzing user-generated content to identify biases and harmful narratives that undermine trust. With the development of Large Language Models (LLMs), existing approaches treat stance detection as a classification problem, providing robust methodologies for modeling complex group interactions and advancing capabilities in natural language tasks. However, these methods often lack interpretability, limiting their ability to offer transparent and understandable justifications for predictions. This study adopts a generative approach, where stance predictions include explicit, interpretable rationales, and integrates them into smaller language models through single-task and multitask learning. We find that incorporating reasoning into stance detection enables the smaller model (FlanT5) to outperform GPT-3.5's zero-shot performance, achieving an improvement of up to 9.57%. Moreover, our results show that reasoning capabilities enhance multitask learning performance but may reduce effectiveness in single-task settings. Crucially, we demonstrate that faithful rationales improve rationale distillation into SLMs, advancing efforts to build interpretable, trustworthy systems for addressing discrimination, fostering trust, and promoting equitable engagement on social media.
>
---
#### [replaced 020] Multimodal Contrastive Representation Learning in Augmented Biomedical Knowledge Graphs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.01644v2](http://arxiv.org/pdf/2501.01644v2)**

> **作者:** Tien Dang; Viet Thanh Duy Nguyen; Minh Tuan Le; Truong-Son Hy
>
> **摘要:** Biomedical Knowledge Graphs (BKGs) integrate diverse datasets to elucidate complex relationships within the biomedical field. Effective link prediction on these graphs can uncover valuable connections, such as potential novel drug-disease relations. We introduce a novel multimodal approach that unifies embeddings from specialized Language Models (LMs) with Graph Contrastive Learning (GCL) to enhance intra-entity relationships while employing a Knowledge Graph Embedding (KGE) model to capture inter-entity relationships for effective link prediction. To address limitations in existing BKGs, we present PrimeKG++, an enriched knowledge graph incorporating multimodal data, including biological sequences and textual descriptions for each entity type. By combining semantic and relational information in a unified representation, our approach demonstrates strong generalizability, enabling accurate link predictions even for unseen nodes. Experimental results on PrimeKG++ and the DrugBank drug-target interaction dataset demonstrate the effectiveness and robustness of our method across diverse biomedical datasets. Our source code, pre-trained models, and data are publicly available at https://github.com/HySonLab/BioMedKG
>
---
#### [replaced 021] Sparsing Law: Towards Large Language Models with Greater Activation Sparsity
- **分类: cs.LG; cs.CL; stat.ML; I.2.7**

- **链接: [http://arxiv.org/pdf/2411.02335v4](http://arxiv.org/pdf/2411.02335v4)**

> **作者:** Yuqi Luo; Chenyang Song; Xu Han; Yingfa Chen; Chaojun Xiao; Xiaojun Meng; Liqun Deng; Jiansheng Wei; Zhiyuan Liu; Maosong Sun
>
> **备注:** 23 pages, 13 figures, 6 tables
>
> **摘要:** Activation sparsity denotes the existence of substantial weakly-contributed elements within activation outputs that can be eliminated, benefiting many important applications concerned with large language models (LLMs). Although promoting greater activation sparsity within LLMs deserves deep studies, existing works lack comprehensive and quantitative research on the correlation between activation sparsity and potentially influential factors. In this paper, we present a comprehensive study on the quantitative scaling properties and influential factors of the activation sparsity within decoder-only Transformer-based LLMs. Specifically, we propose PPL-$p\%$ sparsity, a precise and performance-aware activation sparsity metric that is applicable to any activation function. Through extensive experiments, we find several important phenomena. Firstly, different activation functions exhibit comparable performance but opposite training-time sparsity trends. The activation ratio (i.e., $1-\mathrm{sparsity\ ratio}$) evolves as a convergent increasing power-law and decreasing logspace power-law with the amount of training data for SiLU-activated and ReLU-activated LLMs, respectively. These demonstrate that ReLU is more efficient as the activation function than SiLU and can leverage more training data to improve activation sparsity. Secondly, the activation ratio linearly increases with the width-depth ratio below a certain bottleneck point, indicating the potential advantage of a deeper architecture at a fixed parameter scale. Finally, at similar width-depth ratios, we surprisingly find that the limit value of activation sparsity varies weakly with the parameter scale, i.e., the activation patterns within LLMs are insensitive to the parameter scale. These empirical laws towards LLMs with greater activation sparsity have important implications for making LLMs more efficient and interpretable.
>
---
#### [replaced 022] Know Your Mistakes: Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.10316v4](http://arxiv.org/pdf/2501.10316v4)**

> **作者:** Suvodip Dey; Yi-Jyun Sun; Gokhan Tur; Dilek Hakkani-Tur
>
> **备注:** Accepted at ACL 2025 Main Conference
>
> **摘要:** Recent LLMs have enabled significant advancements for conversational agents. However, they are also well known to hallucinate, producing responses that seem plausible but are factually incorrect. On the other hand, users tend to over-rely on LLM-based AI agents, accepting AI's suggestion even when it is wrong. Adding positive friction, such as explanations or getting user confirmations, has been proposed as a mitigation in AI-supported decision-making systems. In this paper, we propose an accountability model for LLM-based task-oriented dialogue agents to address user overreliance via friction turns in cases of model uncertainty and errors associated with dialogue state tracking (DST). The accountability model is an augmented LLM with an additional accountability head that functions as a binary classifier to predict the relevant slots of the dialogue state mentioned in the conversation. We perform our experiments with multiple backbone LLMs on two established benchmarks (MultiWOZ and Snips). Our empirical findings demonstrate that the proposed approach not only enables reliable estimation of AI agent errors but also guides the decoder in generating more accurate actions. We observe around 3% absolute improvement in joint goal accuracy (JGA) of DST output by incorporating accountability heads into modern LLMs. Self-correcting the detected errors further increases the JGA from 67.13 to 70.51, achieving state-of-the-art DST performance. Finally, we show that error correction through user confirmations (friction turn) achieves a similar performance gain, highlighting its potential to reduce user overreliance.
>
---
#### [replaced 023] LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.16334v2](http://arxiv.org/pdf/2503.16334v2)**

> **作者:** Ying Shen; Lifu Huang
>
> **备注:** ACL 2025, 16 pages, 2 figures
>
> **摘要:** Recent findings reveal that much of the knowledge in a Transformer-based Large Language Model (LLM) is encoded in its feed-forward (FFN) layers, where each FNN layer can be interpreted as the summation of sub-updates, each corresponding to a weighted column vector from the FFN's value parameter matrix that often encodes human-interpretable concepts. In light of this, we hypothesize that model performance and behaviors can be further enhanced and controlled by modulating the contributions of these sub-updates based on their relevance to the input or target output style, and propose LLMBRACES, a novel and efficient method that computes relevance scores associated with value vectors in FFN layers and leverages these scores to dynamically adjust the contribution of sub-updates. By optimizing sub-update contributions, LLMBRACES refines the prediction process, leading to more accurate and reliable outputs, much like a 'brace' providing support and stability. Moreover, LLMBRACES can be extended to support conditional control over generation characteristics, such as sentiment, thereby offering fine-grained steering of LLM outputs. Extensive experiments on various LLMs-including Qwen2.5-1.5B, Llama2-7B, and Llama3-8B-demonstrate that LLMBRACES outperforms baseline approaches in both fine-tuning and zero-shot settings while requiring significantly fewer tunable parameters, up to 75% fewer compared to LoRA. Furthermore, LLMBRACES excels in sentiment-controlled generation and toxicity reduction, highlighting its potential for flexible, controlled text generation across applications.
>
---
#### [replaced 024] PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.03124v5](http://arxiv.org/pdf/2501.03124v5)**

> **作者:** Mingyang Song; Zhaochen Su; Xiaoye Qu; Jiawei Zhou; Yu Cheng
>
> **备注:** Accepted by ACL 2025 Main. Project Page: https://prmbench.github.io/
>
> **摘要:** Process-level Reward Models (PRMs) are crucial for complex reasoning and decision-making tasks, where each intermediate step plays an important role in the reasoning process. Since language models are prone to various types of errors during the reasoning process, PRMs are required to possess nuanced capabilities for detecting various implicit error types in real-world scenarios. However, current benchmarks primarily focus on step correctness, failing to evaluate PRMs' performance systematically. To address this gap, we introduce PRMBench, a process-level benchmark specifically designed to assess the fine-grained error detection capabilities of PRMs. PRMBench comprises 6,216 carefully designed problems and 83,456 step-level labels, evaluating models across multiple dimensions, including simplicity, soundness, and sensitivity. In our experiments on 15 models, spanning both open-source PRMs and closed-source large language models prompted as critic models, we uncover significant weaknesses in current PRMs. These findings underscore the challenges inherent in process-level evaluation and highlight key directions for future research. We hope PRMBench can be a robust bench for advancing research on PRM evaluation and development.
>
---
#### [replaced 025] Bridge: A Unified Framework to Knowledge Graph Completion via Language Models and Knowledge Representation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.06660v3](http://arxiv.org/pdf/2411.06660v3)**

> **作者:** Qiao Qiao; Yuepei Li; Qing Wang; Kang Zhou; Qi Li
>
> **摘要:** Knowledge graph completion (KGC) is a task of inferring missing triples based on existing Knowledge Graphs (KGs). Both structural and semantic information are vital for successful KGC. However, existing methods only use either the structural knowledge from the KG embeddings or the semantic information from pre-trained language models (PLMs), leading to suboptimal model performance. Moreover, since PLMs are not trained on KGs, directly using PLMs to encode triples may be inappropriate. To overcome these limitations, we propose a novel framework called Bridge, which jointly encodes structural and semantic information of KGs. Specifically, we strategically encode entities and relations separately by PLMs to better utilize the semantic knowledge of PLMs and enable structured representation learning via a structural learning principle. Furthermore, to bridge the gap between KGs and PLMs, we employ a self-supervised representation learning method called BYOL to fine-tune PLMs with two different views of a triple. Unlike BYOL, which uses augmentation methods to create two semantically similar views of the same image, potentially altering the semantic information. We strategically separate the triple into two parts to create different views, thus avoiding semantic alteration. Experiments demonstrate that Bridge outperforms the SOTA models on three benchmark datasets.
>
---
#### [replaced 026] LibVulnWatch: A Deep Assessment Agent System and Leaderboard for Uncovering Hidden Vulnerabilities in Open-Source AI Libraries
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.08842v2](http://arxiv.org/pdf/2505.08842v2)**

> **作者:** Zekun Wu; Seonglae Cho; Umar Mohammed; Cristian Munoz; Kleyton Costa; Xin Guan; Theo King; Ze Wang; Emre Kazim; Adriano Koshiyama
>
> **备注:** ACL 2025 Student Research Workshop and ICML 2025 TAIG Workshop
>
> **摘要:** Open-source AI libraries are foundational to modern AI systems, yet they present significant, underexamined risks spanning security, licensing, maintenance, supply chain integrity, and regulatory compliance. We introduce LibVulnWatch, a system that leverages recent advances in large language models and agentic workflows to perform deep, evidence-based evaluations of these libraries. Built on a graph-based orchestration of specialized agents, the framework extracts, verifies, and quantifies risk using information from repositories, documentation, and vulnerability databases. LibVulnWatch produces reproducible, governance-aligned scores across five critical domains, publishing results to a public leaderboard for ongoing ecosystem monitoring. Applied to 20 widely used libraries, including ML frameworks, LLM inference engines, and agent orchestration tools, our approach covers up to 88% of OpenSSF Scorecard checks while surfacing up to 19 additional risks per library, such as critical RCE vulnerabilities, missing SBOMs, and regulatory gaps. By integrating advanced language technologies with the practical demands of software risk assessment, this work demonstrates a scalable, transparent mechanism for continuous supply chain evaluation and informed library selection.
>
---
#### [replaced 027] FinEval-KR: A Financial Domain Evaluation Framework for Large Language Models' Knowledge and Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21591v2](http://arxiv.org/pdf/2506.21591v2)**

> **作者:** Shaoyu Dou; Yutian Shen; Mofan Chen; Zixuan Wang; Jiajie Xu; Qi Guo; Kailai Shao; Chao Chen; Haixiang Hu; Haibo Shi; Min Min; Liwen Zhang
>
> **备注:** The statistics included in the paper are incomplete (e.g., Tables 2 and 5 report only the results of a single run), which may lead readers to misunderstand
>
> **摘要:** Large Language Models (LLMs) demonstrate significant potential but face challenges in complex financial reasoning tasks requiring both domain knowledge and sophisticated reasoning. Current evaluation benchmarks often fall short by not decoupling these capabilities indicators from single task performance and lack root cause analysis for task failure. To address this, we introduce FinEval-KR, a novel evaluation framework for decoupling and quantifying LLMs' knowledge and reasoning abilities independently, proposing distinct knowledge score and reasoning score metrics. Inspired by cognitive science, we further propose a cognitive score based on Bloom's taxonomy to analyze capabilities in reasoning tasks across different cognitive levels. We also release a new open-source Chinese financial reasoning dataset covering 22 subfields to support reproducible research and further advancements in financial reasoning. Our experimental results reveal that LLM reasoning ability and higher-order cognitive ability are the core factors influencing reasoning accuracy. We also specifically find that even top models still face a bottleneck with knowledge application. Furthermore, our analysis shows that specialized financial LLMs generally lag behind the top general large models across multiple metrics.
>
---
#### [replaced 028] Comparative Evaluation of ChatGPT and DeepSeek Across Key NLP Tasks: Strengths, Weaknesses, and Domain-Specific Performance
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.18501v2](http://arxiv.org/pdf/2506.18501v2)**

> **作者:** Wael Etaiwi; Bushra Alhijawi
>
> **摘要:** The increasing use of large language models (LLMs) in natural language processing (NLP) tasks has sparked significant interest in evaluating their effectiveness across diverse applications. While models like ChatGPT and DeepSeek have shown strong results in many NLP domains, a comprehensive evaluation is needed to understand their strengths, weaknesses, and domain-specific abilities. This is critical as these models are applied to various tasks, from sentiment analysis to more nuanced tasks like textual entailment and translation. This study aims to evaluate ChatGPT and DeepSeek across five key NLP tasks: sentiment analysis, topic classification, text summarization, machine translation, and textual entailment. A structured experimental protocol is used to ensure fairness and minimize variability. Both models are tested with identical, neutral prompts and evaluated on two benchmark datasets per task, covering domains like news, reviews, and formal/informal texts. The results show that DeepSeek excels in classification stability and logical reasoning, while ChatGPT performs better in tasks requiring nuanced understanding and flexibility. These findings provide valuable insights for selecting the appropriate LLM based on task requirements.
>
---
#### [replaced 029] Demystifying Singular Defects in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.07004v2](http://arxiv.org/pdf/2502.07004v2)**

> **作者:** Haoqi Wang; Tong Zhang; Mathieu Salzmann
>
> **备注:** ICML 2025
>
> **摘要:** Large transformer models are known to produce high-norm tokens. In vision transformers (ViTs), such tokens have been mathematically modeled through the singular vectors of the linear approximations of layers. However, in large language models (LLMs), the underlying causes of high-norm tokens remain largely unexplored, and their different properties from those of ViTs require a new analysis framework. In this paper, we provide both theoretical insights and empirical validation across a range of recent models, leading to the following observations: i) The layer-wise singular direction predicts the abrupt explosion of token norms in LLMs. ii) The negative eigenvalues of a layer explain its sudden decay. iii) The computational pathways leading to high-norm tokens differ between initial and noninitial tokens. iv) High-norm tokens are triggered by the right leading singular vector of the matrix approximating the corresponding modules. We showcase two practical applications of these findings: the improvement of quantization schemes and the design of LLM signatures. Our findings not only advance the understanding of singular defects in LLMs but also open new avenues for their application. We expect that this work will stimulate further research into the internal mechanisms of LLMs. Code is released at https://github.com/haoqiwang/singular_defect.
>
---
#### [replaced 030] Evaluating K-Fold Cross Validation for Transformer Based Symbolic Regression Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.21896v2](http://arxiv.org/pdf/2410.21896v2)**

> **作者:** Kaustubh Kislay; Shlok Singh; Soham Joshi; Rohan Dutta; Jay Shim; George Flint; Kevin Zhu
>
> **摘要:** Symbolic Regression remains an NP-Hard problem, with extensive research focusing on AI models for this task. Transformer models have shown promise in Symbolic Regression, but performance suffers with smaller datasets. We propose applying k-fold cross-validation to a transformer-based symbolic regression model trained on a significantly reduced dataset (15,000 data points, down from 500,000). This technique partitions the training data into multiple subsets (folds), iteratively training on some while validating on others. Our aim is to provide an estimate of model generalization and mitigate overfitting issues associated with smaller datasets. Results show that this process improves the model's output consistency and generalization by a relative improvement in validation loss of 53.31%. Potentially enabling more efficient and accessible symbolic regression in resource-constrained environments.
>
---
#### [replaced 031] Potemkin Understanding in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.21521v2](http://arxiv.org/pdf/2506.21521v2)**

> **作者:** Marina Mancoridis; Bec Weeks; Keyon Vafa; Sendhil Mullainathan
>
> **摘要:** Large language models (LLMs) are regularly evaluated using benchmark datasets. But what justifies making inferences about an LLM's capabilities based on its answers to a curated set of questions? This paper first introduces a formal framework to address this question. The key is to note that the benchmarks used to test LLMs -- such as AP exams -- are also those used to test people. However, this raises an implication: these benchmarks are only valid tests if LLMs misunderstand concepts in ways that mirror human misunderstandings. Otherwise, success on benchmarks only demonstrates potemkin understanding: the illusion of understanding driven by answers irreconcilable with how any human would interpret a concept. We present two procedures for quantifying the existence of potemkins: one using a specially designed benchmark in three domains, the other using a general procedure that provides a lower-bound on their prevalence. We find that potemkins are ubiquitous across models, tasks, and domains. We also find that these failures reflect not just incorrect understanding, but deeper internal incoherence in concept representations.
>
---
#### [replaced 032] GeometryZero: Improving Geometry Solving for LLM with Group Contrastive Policy Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07160v2](http://arxiv.org/pdf/2506.07160v2)**

> **作者:** Yikun Wang; Yibin Wang; Dianyi Wang; Zimian Peng; Qipeng Guo; Dacheng Tao; Jiaqi Wang
>
> **摘要:** Recent advances in large language models (LLMs) have demonstrated remarkable capabilities across diverse domains, particularly in mathematical reasoning, amid which geometry problem solving remains a challenging area where auxiliary construction plays a enssential role. Existing approaches either achieve suboptimal performance or rely on massive LLMs (e.g., GPT-4o), incurring massive computational costs. We posit that reinforcement learning with verifiable reward (e.g., GRPO) offers a promising direction for training smaller models that effectively combine auxiliary construction with robust geometric reasoning. However, directly applying GRPO to geometric reasoning presents fundamental limitations due to its dependence on unconditional rewards, which leads to indiscriminate and counterproductive auxiliary constructions. To address these challenges, we propose Group Contrastive Policy Optimization (GCPO), a novel reinforcement learning framework featuring two key innovations: (1) Group Contrastive Masking, which adaptively provides positive or negative reward signals for auxiliary construction based on contextual utility, and a (2) length reward that promotes longer reasoning chains. Building on GCPO, we develop GeometryZero, a family of affordable-size geometric reasoning models that judiciously determine when to employ auxiliary construction. Our extensive empirical evaluation across popular geometric benchmarks (Geometry3K, MathVista) demonstrates that GeometryZero models consistently outperform baselines (e.g. GRPO), achieving an average improvement of 4.29% across all benchmarks.
>
---
#### [replaced 033] Multimodal Medical Code Tokenizer
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.04397v3](http://arxiv.org/pdf/2502.04397v3)**

> **作者:** Xiaorui Su; Shvat Messica; Yepeng Huang; Ruth Johnson; Lukas Fesser; Shanghua Gao; Faryad Sahneh; Marinka Zitnik
>
> **备注:** ICML'25
>
> **摘要:** Foundation models trained on patient electronic health records (EHRs) require tokenizing medical data into sequences of discrete vocabulary items. Existing tokenizers treat medical codes from EHRs as isolated textual tokens. However, each medical code is defined by its textual description, its position in ontological hierarchies, and its relationships to other codes, such as disease co-occurrences and drug-treatment associations. Medical vocabularies contain more than 600,000 codes with critical information for clinical reasoning. We introduce MedTok, a multimodal medical code tokenizer that uses the text descriptions and relational context of codes. MedTok processes text using a language model encoder and encodes the relational structure with a graph encoder. It then quantizes both modalities into a unified token space, preserving modality-specific and cross-modality information. We integrate MedTok into five EHR models and evaluate it on operational and clinical tasks across in-patient and out-patient datasets, including outcome prediction, diagnosis classification, drug recommendation, and risk stratification. Swapping standard EHR tokenizers with MedTok improves AUPRC across all EHR models, by 4.10% on MIMIC-III, 4.78% on MIMIC-IV, and 11.32% on EHRShot, with the largest gains in drug recommendation. Beyond EHR modeling, we demonstrate using MedTok tokenizer with medical QA systems. Our results demonstrate the potential of MedTok as a unified tokenizer for medical codes, improving tokenization for medical foundation models.
>
---
#### [replaced 034] Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2410.10360v3](http://arxiv.org/pdf/2410.10360v3)**

> **作者:** Yongxin Xu; Ruizhe Zhang; Xinke Jiang; Yujie Feng; Yuzhen Xiao; Xinyu Ma; Runchuan Zhu; Xu Chu; Junfeng Zhao; Yasha Wang
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Retrieval-Augmented Generation (RAG) offers an effective solution to the issues faced by Large Language Models (LLMs) in hallucination generation and knowledge obsolescence by incorporating externally retrieved knowledge. However, existing methods lack effective control mechanisms for integrating internal and external knowledge. Inspired by human cognitive processes, we propose Parenting, a novel framework that decouples, identifies, and purposefully optimizes parameter subspaces related to adherence and robustness. Specifically, Parenting utilizes a key parameter mining method that combines forward and backward propagation signals to localize subspaces representing different capabilities. Then, Parenting employs a type-tailored tuning strategy, applying specific and appropriate optimizations to different subspaces, aiming to achieve a balanced enhancement of both adherence and robustness. Extensive experiments on various datasets and models validate the effectiveness and generalizability of our method.
>
---
#### [replaced 035] Computational Analysis of Character Development in Holocaust Testimonies
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.17063v3](http://arxiv.org/pdf/2412.17063v3)**

> **作者:** Esther Shizgal; Eitan Wagner; Renana Keydar; Omri Abend
>
> **摘要:** This work presents a computational approach to analyze character development along the narrative timeline. The analysis characterizes the inner and outer changes the protagonist undergoes within a narrative, and the interplay between them. We consider transcripts of Holocaust survivor testimonies as a test case, each telling the story of an individual in first-person terms. We focus on the survivor's religious trajectory, examining the evolution of their disposition toward religious belief and practice along the testimony. Clustering the resulting trajectories in the dataset, we identify common sequences in the data. Our findings highlight multiple common structures of religiosity across the narratives: in terms of belief, most present a constant disposition, while for practice, most present an oscillating structure, serving as valuable material for historical and sociological research. This work demonstrates the potential of natural language processing techniques for analyzing character evolution through thematic trajectories in narratives.
>
---
#### [replaced 036] Brevity is the soul of sustainability: Characterizing LLM response lengths
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.08686v2](http://arxiv.org/pdf/2506.08686v2)**

> **作者:** Soham Poddar; Paramita Koley; Janardan Misra; Sanjay Podder; Navveen Balani; Niloy Ganguly; Saptarshi Ghosh
>
> **备注:** Accepted to appear at the ACL 2025 findings
>
> **摘要:** A significant portion of the energy consumed by Large Language Models (LLMs) arises from their inference processes; hence developing energy-efficient methods for inference is crucial. While several techniques exist for inference optimization, output compression remains relatively unexplored, with only a few preliminary efforts addressing this aspect. In this work, we first benchmark 12 decoder-only LLMs across 5 datasets, revealing that these models often produce responses that are substantially longer than necessary. We then conduct a comprehensive quality assessment of LLM responses, formally defining six information categories present in LLM responses. We show that LLMs often tend to include redundant or additional information besides the minimal answer. To address this issue of long responses by LLMs, we explore several simple and intuitive prompt-engineering strategies. Empirical evaluation shows that appropriate prompts targeting length reduction and controlling information content can achieve significant energy optimization between 25-60\% by reducing the response length while preserving the quality of LLM responses.
>
---
#### [replaced 037] Position: Machine Learning Conferences Should Establish a "Refutations and Critiques" Track
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.19882v2](http://arxiv.org/pdf/2506.19882v2)**

> **作者:** Rylan Schaeffer; Joshua Kazdan; Yegor Denisov-Blanch; Brando Miranda; Matthias Gerstgrasser; Susan Zhang; Andreas Haupt; Isha Gupta; Elyas Obbad; Jesse Dodge; Jessica Zosa Forde; Koustuv Sinha; Francesco Orabona; Sanmi Koyejo; David Donoho
>
> **摘要:** Science progresses by iteratively advancing and correcting humanity's understanding of the world. In machine learning (ML) research, rapid advancements have led to an explosion of publications, but have also led to misleading, incorrect, flawed or perhaps even fraudulent studies being accepted and sometimes highlighted at ML conferences due to the fallibility of peer review. While such mistakes are understandable, ML conferences do not offer robust processes to help the field systematically correct when such errors are made. This position paper argues that ML conferences should establish a dedicated "Refutations and Critiques" (R&C) Track. This R&C Track would provide a high-profile, reputable platform to support vital research that critically challenges prior research, thereby fostering a dynamic self-correcting research ecosystem. We discuss key considerations including track design, review principles, potential pitfalls, and provide an illustrative example submission concerning a recent ICLR 2025 Oral. We conclude that ML conferences should create official, reputable mechanisms to help ML research self-correct.
>
---
#### [replaced 038] PriorDiffusion: Leverage Language Prior in Diffusion Models for Monocular Depth Estimation
- **分类: cs.CV; cs.CL; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2411.16750v3](http://arxiv.org/pdf/2411.16750v3)**

> **作者:** Ziyao Zeng; Jingcheng Ni; Daniel Wang; Patrick Rim; Younjoon Chung; Fengyu Yang; Byung-Woo Hong; Alex Wong
>
> **摘要:** Traditional monocular depth estimation suffers from inherent ambiguity and visual nuisance. We argue that language prior can enhance monocular depth estimation by leveraging the inductive bias learned during the text-to-image pre-training of diffusion models. The ability of these models to generate images that align with text indicates that they have learned the spatial relationships, size, and shape of specified objects, which can be applied to improve depth estimation. Thus, we propose PriorDiffusion, using a pre-trained text-to-image diffusion model that takes both images and corresponding text descriptions to infer affine-invariant depth through a denoising process. We also show that language prior enhances the model's perception of specific regions of images that users care about and describe. Simultaneously, language prior acts as a constraint to accelerate the convergence of both training and the inference diffusion trajectory. By training on HyperSim and Virtual KITTI, we achieve faster training convergence, fewer inference diffusion steps, and state-of-the-art zero-shot performance across NYUv2, KITTI, ETH3D, and ScanNet. Code will be released upon acceptance.
>
---
#### [replaced 039] KMI: A Dataset of Korean Motivational Interviewing Dialogues for Psychotherapy
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.05651v2](http://arxiv.org/pdf/2502.05651v2)**

> **作者:** Hyunjong Kim; Suyeon Lee; Yeongjae Cho; Eunseo Ryu; Yohan Jo; Suran Seong; Sungzoon Cho
>
> **备注:** Accepted at NAACL 2025 Main Conference
>
> **摘要:** The increasing demand for mental health services has led to the rise of AI-driven mental health chatbots, though challenges related to privacy, data collection, and expertise persist. Motivational Interviewing (MI) is gaining attention as a theoretical basis for boosting expertise in the development of these chatbots. However, existing datasets are showing limitations for training chatbots, leading to a substantial demand for publicly available resources in the field of MI and psychotherapy. These challenges are even more pronounced in non-English languages, where they receive less attention. In this paper, we propose a novel framework that simulates MI sessions enriched with the expertise of professional therapists. We train an MI forecaster model that mimics the behavioral choices of professional therapists and employ Large Language Models (LLMs) to generate utterances through prompt engineering. Then, we present KMI, the first synthetic dataset theoretically grounded in MI, containing 1,000 high-quality Korean Motivational Interviewing dialogues. Through an extensive expert evaluation of the generated dataset and the dialogue model trained on it, we demonstrate the quality, expertise, and practicality of KMI. We also introduce novel metrics derived from MI theory in order to evaluate dialogues from the perspective of MI.
>
---
#### [replaced 040] AI-Generated Song Detection via Lyrics Transcripts
- **分类: cs.SD; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.18488v2](http://arxiv.org/pdf/2506.18488v2)**

> **作者:** Markus Frohmann; Elena V. Epure; Gabriel Meseguer-Brocal; Markus Schedl; Romain Hennequin
>
> **备注:** Accepted to ISMIR 2025
>
> **摘要:** The recent rise in capabilities of AI-based music generation tools has created an upheaval in the music industry, necessitating the creation of accurate methods to detect such AI-generated content. This can be done using audio-based detectors; however, it has been shown that they struggle to generalize to unseen generators or when the audio is perturbed. Furthermore, recent work used accurate and cleanly formatted lyrics sourced from a lyrics provider database to detect AI-generated music. However, in practice, such perfect lyrics are not available (only the audio is); this leaves a substantial gap in applicability in real-life use cases. In this work, we instead propose solving this gap by transcribing songs using general automatic speech recognition (ASR) models. We do this using several detectors. The results on diverse, multi-genre, and multi-lingual lyrics show generally strong detection performance across languages and genres, particularly for our best-performing model using Whisper large-v2 and LLM2Vec embeddings. In addition, we show that our method is more robust than state-of-the-art audio-based ones when the audio is perturbed in different ways and when evaluated on different music generators. Our code is available at https://github.com/deezer/robust-AI-lyrics-detection.
>
---
#### [replaced 041] FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2506.12494v2](http://arxiv.org/pdf/2506.12494v2)**

> **作者:** Zhuocheng Zhang; Yang Feng; Min Zhang
>
> **备注:** Accepted by ACL 2025 Demo
>
> **摘要:** Retrieval-Augmented Generation (RAG) plays a pivotal role in modern large language model applications, with numerous existing frameworks offering a wide range of functionalities to facilitate the development of RAG systems. However, we have identified several persistent challenges in these frameworks, including difficulties in algorithm reproduction and sharing, lack of new techniques, and high system overhead. To address these limitations, we introduce \textbf{FlexRAG}, an open-source framework specifically designed for research and prototyping. FlexRAG supports text-based, multimodal, and network-based RAG, providing comprehensive lifecycle support alongside efficient asynchronous processing and persistent caching capabilities. By offering a robust and flexible solution, FlexRAG enables researchers to rapidly develop, deploy, and share advanced RAG systems. Our toolkit and resources are available at \href{https://github.com/ictnlp/FlexRAG}{https://github.com/ictnlp/FlexRAG}.
>
---
#### [replaced 042] Learning Dynamics of LLM Finetuning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.10490v4](http://arxiv.org/pdf/2407.10490v4)**

> **作者:** Yi Ren; Danica J. Sutherland
>
> **摘要:** Learning dynamics, which describes how the learning of specific training examples influences the model's predictions on other examples, gives us a powerful tool for understanding the behavior of deep learning systems. We study the learning dynamics of large language models during different types of finetuning, by analyzing the step-wise decomposition of how influence accumulates among different potential responses. Our framework allows a uniform interpretation of many interesting observations about the training of popular algorithms for both instruction tuning and preference tuning. In particular, we propose a hypothetical explanation of why specific types of hallucination are strengthened after finetuning, e.g., the model might use phrases or facts in the response for question B to answer question A, or the model might keep repeating similar simple phrases when generating responses. We also extend our framework and highlight a unique "squeezing effect" to explain a previously observed phenomenon in off-policy direct preference optimization (DPO), where running DPO for too long makes even the desired outputs less likely. This framework also provides insights into where the benefits of on-policy DPO and other variants come from. The analysis not only provides a novel perspective of understanding LLM's finetuning but also inspires a simple, effective method to improve alignment performance.
>
---
#### [replaced 043] A Context-aware Framework for Translation-mediated Conversations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.04205v2](http://arxiv.org/pdf/2412.04205v2)**

> **作者:** José Pombal; Sweta Agrawal; Patrick Fernandes; Emmanouil Zaranis; André F. T. Martins
>
> **摘要:** Automatic translation systems offer a powerful solution to bridge language barriers in scenarios where participants do not share a common language. However, these systems can introduce errors leading to misunderstandings and conversation breakdown. A key issue is that current systems fail to incorporate the rich contextual information necessary to resolve ambiguities and omitted details, resulting in literal, inappropriate, or misaligned translations. In this work, we present a framework to improve large language model-based translation systems by incorporating contextual information in bilingual conversational settings during training and inference. We validate our proposed framework on two task-oriented domains: customer chat and user-assistant interaction. Across both settings, the system produced by our framework-TowerChat-consistently results in better translations than state-of-the-art systems like GPT-4o and TowerInstruct, as measured by multiple automatic translation quality metrics on several language pairs. We also show that the resulting model leverages context in an intended and interpretable way, improving consistency between the conveyed message and the generated translations.
>
---
#### [replaced 044] Sample then Identify: A General Framework for Risk Control and Assessment in Multimodal Large Language Models
- **分类: cs.CL; cs.AI; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2410.08174v3](http://arxiv.org/pdf/2410.08174v3)**

> **作者:** Qingni Wang; Tiantian Geng; Zhiyuan Wang; Teng Wang; Bo Fu; Feng Zheng
>
> **备注:** Accepted by ICLR 2025 Spotlights
>
> **摘要:** Multimodal Large Language Models (MLLMs) exhibit promising advancements across various tasks, yet they still encounter significant trustworthiness issues. Prior studies apply Split Conformal Prediction (SCP) in language modeling to construct prediction sets with statistical guarantees. However, these methods typically rely on internal model logits or are restricted to multiple-choice settings, which hampers their generalizability and adaptability in dynamic, open-ended environments. In this paper, we introduce TRON, a two-step framework for risk control and assessment, applicable to any MLLM that supports sampling in both open-ended and closed-ended scenarios. TRON comprises two main components: (1) a novel conformal score to sample response sets of minimum size, and (2) a nonconformity score to identify high-quality responses based on self-consistency theory, controlling the error rates by two specific risk levels. Furthermore, we investigate semantic redundancy in prediction sets within open-ended contexts for the first time, leading to a promising evaluation metric for MLLMs based on average set size. Our comprehensive experiments across four Video Question-Answering (VideoQA) datasets utilizing eight MLLMs show that TRON achieves desired error rates bounded by two user-specified risk levels. Additionally, deduplicated prediction sets maintain adaptiveness while being more efficient and stable for risk assessment under different risk levels.
>
---
#### [replaced 045] The Limited Impact of Medical Adaptation of Large Language and Vision-Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.08870v3](http://arxiv.org/pdf/2411.08870v3)**

> **作者:** Daniel P. Jeong; Pranav Mani; Saurabh Garg; Zachary C. Lipton; Michael Oberst
>
> **备注:** Extended version of EMNLP 2024 paper arXiv:2411.04118. Includes additional results on clinical note QA tasks and supervised fine-tuning evaluations
>
> **摘要:** Several recent works seek to adapt general-purpose large language models (LLMs) and vision-language models (VLMs) for medical applications through continued pretraining on publicly available biomedical corpora. These works typically claim that such domain-adaptive pretraining improves performance on various downstream medical tasks, such as answering medical exam questions. In this paper, we compare ten "medical" LLMs and two VLMs against their corresponding base models, arriving at a different conclusion: all medical VLMs and nearly all medical LLMs fail to consistently improve over their base models in the zero-/few-shot prompting and supervised fine-tuning regimes for medical question answering (QA). For instance, on clinical-note-based QA tasks in the 3-shot setting, medical LLMs outperform their base models in only 26.7% of cases, reach a (statistical) tie in 16.7% of cases, and perform significantly worse in the remaining 56.7% of cases. Our conclusions are based on (i) comparing each medical model directly against its base model; (ii) optimizing the prompts for each model separately in zero-/few-shot prompting; and (iii) accounting for statistical uncertainty in comparisons. Our findings suggest that state-of-the-art general-domain models may already exhibit strong medical knowledge and reasoning capabilities, and offer recommendations to strengthen the conclusions of future studies.
>
---
#### [replaced 046] Beware of Calibration Data for Pruning Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.17711v2](http://arxiv.org/pdf/2410.17711v2)**

> **作者:** Yixin Ji; Yang Xiang; Juntao Li; Qingrong Xia; Ping Li; Xinyu Duan; Zhefeng Wang; Min Zhang
>
> **备注:** Published as a conference paper at ICLR 2025
>
> **摘要:** As large language models (LLMs) are widely applied across various fields, model compression has become increasingly crucial for reducing costs and improving inference efficiency. Post-training pruning is a promising method that does not require resource-intensive iterative training and only needs a small amount of calibration data to assess the importance of parameters. Recent research has enhanced post-training pruning from different aspects but few of them systematically explore the effects of calibration data, and it is unclear if there exist better calibration data construction strategies. We fill this blank and surprisingly observe that calibration data is also crucial to post-training pruning, especially for high sparsity. Through controlled experiments on important influence factors of calibration data, including the pruning settings, the amount of data, and its similarity with pre-training data, we observe that a small size of data is adequate, and more similar data to its pre-training stage can yield better performance. As pre-training data is usually inaccessible for advanced LLMs, we further provide a self-generating calibration data synthesis strategy to construct feasible calibration data. Experimental results on recent strong open-source LLMs (e.g., DCLM, and LLaMA-3) show that the proposed strategy can enhance the performance of strong pruning methods (e.g., Wanda, DSnoT, OWL) by a large margin (up to $2.68\%$). Code is available at https://github.com/Dereck0602/calibration_data.
>
---
#### [replaced 047] CTISum: A New Benchmark Dataset For Cyber Threat Intelligence Summarization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.06576v2](http://arxiv.org/pdf/2408.06576v2)**

> **作者:** Wei Peng; Junmei Ding; Wei Wang; Lei Cui; Wei Cai; Zhiyu Hao; Xiaochun Yun
>
> **摘要:** Cyber Threat Intelligence (CTI) summarization involves generating concise and accurate highlights from web intelligence data, which is critical for providing decision-makers with actionable insights to swiftly detect and respond to cyber threats in the cybersecurity domain. Despite that, the development of efficient techniques for summarizing CTI reports, comprising facts, analytical insights, attack processes, and more, has been hindered by the lack of suitable datasets. To address this gap, we introduce CTISum, a new benchmark dataset designed for the CTI summarization task. Recognizing the significance of understanding attack processes, we also propose a novel fine-grained subtask: attack process summarization, which aims to help defenders assess risks, identify security gaps, and uncover vulnerabilities. Specifically, a multi-stage annotation pipeline is designed to collect and annotate CTI data from diverse web sources, alongside a comprehensive benchmarking of CTISum using both extractive, abstractive and LLMs-based summarization methods. Experimental results reveal that current state-of-the-art models face significant challenges when applied to CTISum, highlighting that automatic summarization of CTI reports remains an open research problem. The code and example dataset can be made publicly available at https://github.com/pengwei-iie/CTISum.
>
---
#### [replaced 048] What Makes the Preferred Thinking Direction for LLMs in Multiple-choice Questions?
- **分类: cs.CL; cs.IT; cs.LG; math.IT**

- **链接: [http://arxiv.org/pdf/2502.18435v3](http://arxiv.org/pdf/2502.18435v3)**

> **作者:** Yizhe Zhang; Richard Bai; Zijin Gu; Ruixiang Zhang; Jiatao Gu; Emmanuel Abbe; Samy Bengio; Navdeep Jaitly
>
> **备注:** 10 pages for the main text
>
> **摘要:** Language models usually use left-to-right (L2R) autoregressive factorization. However, L2R factorization may not always be the best inductive bias. Therefore, we investigate whether alternative factorizations of the text distribution could be beneficial in some tasks. We investigate right-to-left (R2L) training as a compelling alternative, focusing on multiple-choice questions (MCQs) as a test bed for knowledge extraction and reasoning. Through extensive experiments across various model sizes (2B-8B parameters) and training datasets, we find that R2L models can significantly outperform L2R models on several MCQ benchmarks, including logical reasoning, commonsense understanding, and truthfulness assessment tasks. Our analysis reveals that this performance difference may be fundamentally linked to multiple factors including calibration, computability, and directional conditional entropy. We analyze the impact of these factors through controlled simulation studies using arithmetic tasks, where the impacting factors can be better disentangled. Our work demonstrates that exploring alternative factorizations of the text distribution can lead to improvements in LLM capabilities and provides theoretical insights into optimal factorization towards approximating human language distribution, and when each reasoning order might be more advantageous. Our code and checkpoints are released at https://github.com/apple/ml-reversal-blessing.
>
---
#### [replaced 049] The Effectiveness of LLMs as Annotators: A Comparative Overview and Empirical Analysis of Direct Representation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.01299v2](http://arxiv.org/pdf/2405.01299v2)**

> **作者:** Maja Pavlovic; Massimo Poesio
>
> **备注:** LREC-COLING NLPerspectives workshop
>
> **摘要:** Large Language Models (LLMs) have emerged as powerful support tools across various natural language tasks and a range of application domains. Recent studies focus on exploring their capabilities for data annotation. This paper provides a comparative overview of twelve studies investigating the potential of LLMs in labelling data. While the models demonstrate promising cost and time-saving benefits, there exist considerable limitations, such as representativeness, bias, sensitivity to prompt variations and English language preference. Leveraging insights from these studies, our empirical analysis further examines the alignment between human and GPT-generated opinion distributions across four subjective datasets. In contrast to the studies examining representation, our methodology directly obtains the opinion distribution from GPT. Our analysis thereby supports the minority of studies that are considering diverse perspectives when evaluating data annotation tasks and highlights the need for further research in this direction.
>
---
#### [replaced 050] Interpretable LLM-based Table Question Answering
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.12386v3](http://arxiv.org/pdf/2412.12386v3)**

> **作者:** Giang Nguyen; Ivan Brugere; Shubham Sharma; Sanjay Kariyappa; Anh Totti Nguyen; Freddy Lecue
>
> **备注:** Published in Transactions on Machine Learning Research (TMLR) in 06/2025. Reviews at: https://openreview.net/forum?id=2eTsZBoU2W
>
> **摘要:** Interpretability in Table Question Answering (Table QA) is critical, especially in high-stakes domains like finance and healthcare. While recent Table QA approaches based on Large Language Models (LLMs) achieve high accuracy, they often produce ambiguous explanations of how answers are derived. We propose Plan-of-SQLs (POS), a new Table QA method that makes the model's decision-making process interpretable. POS decomposes a question into a sequence of atomic steps, each directly translated into an executable SQL command on the table, thereby ensuring that every intermediate result is transparent. Through extensive experiments, we show that: First, POS generates the highest-quality explanations among compared methods, which markedly improves the users' ability to simulate and verify the model's decisions. Second, when evaluated on standard Table QA benchmarks (TabFact, WikiTQ, and FeTaQA), POS achieves QA accuracy that is competitive to existing methods, while also offering greater efficiency-requiring significantly fewer LLM calls and table database queries (up to 25x fewer)-and more robust performance on large-sized tables. Finally, we observe high agreement (up to 90.59% in forward simulation) between LLMs and human users when making decisions based on the same explanations, suggesting that LLMs could serve as an effective proxy for humans in evaluating Table QA explanations.
>
---
#### [replaced 051] Improved Supervised Fine-Tuning for Large Language Models to Mitigate Catastrophic Forgetting
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.09428v2](http://arxiv.org/pdf/2506.09428v2)**

> **作者:** Fei Ding; Baiqiao Wang
>
> **摘要:** Supervised Fine-Tuning (SFT) is a critical step for enhancing the instruction-following capabilities of Large Language Models (LLMs) and adapting them to specialized domains. However, SFT often leads to a degradation of the model's general abilities, a phenomenon known as catastrophic forgetting. This problem is exacerbated when third-party practitioners fine-tune open-source models, as the original SFT data is typically not available. To address this challenge, we propose a novel and cost-effective SFT method that effectively mitigates catastrophic forgetting without requiring access to the original SFT data. Our approach first reconstructs the likely instruction distribution of the base model. It then employs a multi-model generation and filtering pipeline to synthesize a high-quality general-purpose dataset. This synthetic dataset is mixed with new, domain-specific data for fine-tuning. Experimental results show that our method not only preserves the model's capabilities in general domains but also improves task-specific performance, outperforming baselines that use publicly available SFT datasets.
>
---
#### [replaced 052] Empirical evidence of Large Language Model's influence on human spoken communication
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2409.01754v2](http://arxiv.org/pdf/2409.01754v2)**

> **作者:** Hiromu Yakura; Ezequiel Lopez-Lopez; Levin Brinkmann; Ignacio Serna; Prateek Gupta; Ivan Soraperra; Iyad Rahwan
>
> **摘要:** From the invention of writing and the printing press, to television and social media, human history is punctuated by major innovations in communication technology, which fundamentally altered how ideas spread and reshaped our culture. Recent chatbots powered by generative artificial intelligence constitute a novel medium that encodes cultural patterns in their neural representations and disseminates them in conversations with hundreds of millions of people. Understanding whether these patterns transmit into human language, and ultimately shape human culture, is a fundamental question. While fully quantifying the causal impact of a chatbot like ChatGPT on human culture is very challenging, lexicographic shift in human spoken communication may offer an early indicator of such broad phenomenon. Here, we apply econometric causal inference techniques to 740,249 hours of human discourse from 360,445 YouTube academic talks and 771,591 conversational podcast episodes across multiple disciplines. We detect a measurable and abrupt increase in the use of words preferentially generated by ChatGPT, such as delve, comprehend, boast, swift, and meticulous, after its release. These findings suggest a scenario where machines, originally trained on human data and subsequently exhibiting their own cultural traits, can, in turn, measurably reshape human culture. This marks the beginning of a closed cultural feedback loop in which cultural traits circulate bidirectionally between humans and machines. Our results motivate further research into the evolution of human-machine culture, and raise concerns over the erosion of linguistic and cultural diversity, and the risks of scalable manipulation.
>
---
#### [replaced 053] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.04745v4](http://arxiv.org/pdf/2504.04745v4)**

> **作者:** Ankush Raut; Xiaofeng Zhu; Maria Leonor Pacheco
>
> **备注:** 13 pages, 23 figures. Accepted to XLLM Workshop at ACL 2025
>
> **摘要:** This paper evaluates the ability of Large Language Models (LLMs) to leverage contextual information in the form of structured linguistic representations. Specifically, we examine the impact of encoding both short and long contexts using Abstract Meaning Representation (AMR) structures across a diverse set of language tasks. We perform our analysis using 8-bit quantized and instruction-tuned versions of Llama 3.1 (8B), Phi-3, and Mistral 7B. Our results indicate that, for tasks involving short contexts, augmenting the prompt with the AMR of the original language context often degrades the performance of the underlying LLM. However, for tasks that involve long contexts, such as dialogue summarization in the SAMSum dataset, this enhancement improves LLM performance, for example, by increasing the zero-shot cosine similarity score of Llama 3.1 from 66% to 76%. This improvement is more evident in the newer and larger LLMs, but does not extend to the older or smaller ones. In addition, we observe that LLMs can effectively reconstruct the original text from a linearized AMR, achieving a cosine similarity of 81% in the best-case scenario.
>
---
#### [replaced 054] Enabling Precise Topic Alignment in Large Language Models Via Sparse Autoencoders
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.12576v2](http://arxiv.org/pdf/2506.12576v2)**

> **作者:** Ananya Joshi; Celia Cintas; Skyler Speakman
>
> **摘要:** Recent work shows that Sparse Autoencoders (SAE) applied to large language model (LLM) layers have neurons corresponding to interpretable concepts. These SAE neurons can be modified to align generated outputs, but only towards pre-identified topics and with some parameter tuning. Our approach leverages the observational and modification properties of SAEs to enable alignment for any topic. This method 1) scores each SAE neuron by its semantic similarity to an alignment text and uses them to 2) modify SAE-layer-level outputs by emphasizing topic-aligned neurons. We assess the alignment capabilities of this approach on diverse public topic datasets including Amazon reviews, Medicine, and Sycophancy, across the currently available open-source LLMs and SAE pairs (GPT2 and Gemma) with multiple SAEs configurations. Experiments aligning to medical prompts reveal several benefits over fine-tuning, including increased average language acceptability (0.25 vs. 0.5), reduced training time across multiple alignment topics (333.6s vs. 62s), and acceptable inference time for many applications (+0.00092s/token). Our open-source code is available at github.com/IBM/sae-steering.
>
---
#### [replaced 055] Redefining Evaluation Standards: A Unified Framework for Evaluating the Korean Capabilities of Language Models
- **分类: cs.CE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.22968v3](http://arxiv.org/pdf/2503.22968v3)**

> **作者:** Hanwool Lee; Dasol Choi; Sooyong Kim; Ilgyun Jung; Sangwon Baek; Guijin Son; Inseon Hwang; Naeun Lee; Seunghyeok Hong
>
> **摘要:** Recent advancements in Korean large language models (LLMs) have driven numerous benchmarks and evaluation methods, yet inconsistent protocols cause up to 10 p.p performance gaps across institutions. Overcoming these reproducibility gaps does not mean enforcing a one-size-fits-all evaluation. Rather, effective benchmarking requires diverse experimental approaches and a framework robust enough to support them. To this end, we introduce HRET (Haerae Evaluation Toolkit), an open-source, registry-based framework that unifies Korean LLM assessment. HRET integrates major Korean benchmarks, multiple inference backends, and multi-method evaluation, with language consistency enforcement to ensure genuine Korean outputs. Its modular registry design also enables rapid incorporation of new datasets, methods, and backends, ensuring the toolkit adapts to evolving research needs. Beyond standard accuracy metrics, HRET incorporates Korean-focused output analyses-morphology-aware Type-Token Ratio (TTR) for evaluating lexical diversity and systematic keyword-omission detection for identifying missing concepts-to provide diagnostic insights into language-specific behaviors. These targeted analyses help researchers pinpoint morphological and semantic shortcomings in model outputs, guiding focused improvements in Korean LLM development.
>
---
#### [replaced 056] MLAN: Language-Based Instruction Tuning Preserves and Transfers Knowledge in Multimodal Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.10557v3](http://arxiv.org/pdf/2411.10557v3)**

> **作者:** Jianhong Tu; Zhuohao Ni; Nicholas Crispino; Zihao Yu; Michael Bendersky; Beliz Gunel; Ruoxi Jia; Xin Liu; Lingjuan Lyu; Dawn Song; Chenguang Wang
>
> **摘要:** We present a novel visual instruction tuning strategy to improve the zero-shot task generalization of multimodal large language models by building a firm text-only knowledge base. Existing work lacks sufficient experimentation on the importance of each modality in the instruction tuning stage, often using a majority of vision-language data while keeping text-only data limited and fixing mixtures of modalities. By incorporating diverse text-only data in the visual instruction tuning stage, we vary vision-language data in various controlled experiments to investigate the importance of modality in visual instruction tuning. Our comprehensive evaluation shows that the text-heavy instruction tuning approach is able to perform on-par with traditional vision-heavy mixtures on both modalities across 12 general datasets while using as low as half the total training tokens. We find that simply increasing sufficiently diverse text-only data enables transfer of instruction following ability and domain knowledge across modalities while being more efficient than the vision-language approach.
>
---
#### [replaced 057] ChipXplore: Natural Language Exploration of Hardware Designs and Libraries
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.12749v3](http://arxiv.org/pdf/2407.12749v3)**

> **作者:** Manar Abdelatty; Jacob Rosenstein; Sherief Reda
>
> **备注:** 10 pages
>
> **摘要:** Hardware design workflows rely on Process Design Kits (PDKs) from different fabrication nodes, each containing standard cell libraries optimized for speed, power, or density. Engineers typically navigate between the design and target PDK to make informed decisions, such as selecting gates for area optimization or enhancing the speed of the critical path. However, this process is often manual, time-consuming, and prone to errors. To address this, we present ChipXplore, a multi-agent collaborative framework powered by large language models that enables engineers to query hardware designs and PDKs using natural language. By exploiting the structured nature of PDK and hardware design data, ChipXplore retrieves relevant information through text-to-SQL and text-to-Cypher customized workflows. The framework achieves an execution accuracy of 97.39\% in complex natural language queries and improves productivity by making retrieval 5.63x faster while reducing errors by 5.25x in user studies. Compared to generic workflows, ChipXplore's customized workflow is capable of orchestrating reasoning and planning over multiple databases, improving accuracy by 29.78\%. ChipXplore lays the foundation for building autonomous agents capable of tackling diverse physical design tasks that require PDK and hardware design awareness.
>
---
#### [replaced 058] Emotional RAG LLMs: Reading Comprehension for the Open Internet
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.11189v2](http://arxiv.org/pdf/2408.11189v2)**

> **作者:** Benjamin Reichman; Adar Avsian; Kartik Talamadupula; Toshish Jawale; Larry Heck
>
> **摘要:** Queries to large language models (LLMs) can be divided into two parts: the instruction/question and the accompanying context. The context for retrieval-augmented generation (RAG) systems in most benchmarks comes from Wikipedia-like texts written in a neutral and factual tone. However, real-world RAG applications often retrieve internet-based text with diverse tones and linguistic styles, posing challenges for downstream tasks. This paper introduces (a) a dataset that transforms RAG-retrieved passages into emotionally inflected and sarcastic text, (b) an emotion translation model for adapting text to different tones, and (c) a prompt-based method to improve LLMs' pragmatic interpretation of retrieved text.
>
---
#### [replaced 059] Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.01461v3](http://arxiv.org/pdf/2407.01461v3)**

> **作者:** Xiaohua Wang; Zisu Huang; Feiran Zhang; Zhibo Xu; Cenyuan Zhang; Qi Qian; Xiaoqing Zheng; Xuanjing Huang
>
> **摘要:** The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .
>
---
#### [replaced 060] Which Programming Language and What Features at Pre-training Stage Affect Downstream Logical Inference Performance?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.06735v2](http://arxiv.org/pdf/2410.06735v2)**

> **作者:** Fumiya Uchiyama; Takeshi Kojima; Andrew Gambardella; Qi Cao; Yusuke Iwasawa; Yutaka Matsuo
>
> **备注:** Accepted to EMNLP2024
>
> **摘要:** Recent large language models (LLMs) have demonstrated remarkable generalization abilities in mathematics and logical reasoning tasks. Prior research indicates that LLMs pre-trained with programming language data exhibit high mathematical and reasoning abilities; however, this causal relationship has not been rigorously tested. Our research aims to verify which programming languages and features during pre-training affect logical inference performance. Specifically, we pre-trained decoder-based language models from scratch using datasets from ten programming languages (e.g., Python, C, Java) and three natural language datasets (Wikipedia, Fineweb, C4) under identical conditions. Thereafter, we evaluated the trained models in a few-shot in-context learning setting on logical reasoning tasks: FLD and bAbi, which do not require commonsense or world knowledge. The results demonstrate that nearly all models trained with programming languages consistently outperform those trained with natural languages, indicating that programming languages contain factors that elicit logic inference performance. In addition, we found that models trained with programming languages exhibit a better ability to follow instructions compared to those trained with natural languages. Further analysis reveals that the depth of Abstract Syntax Trees representing parsed results of programs also affects logical reasoning performance. These findings will offer insights into the essential elements of pre-training for acquiring the foundational abilities of LLMs.
>
---
#### [replaced 061] Knowing You Don't Know: Learning When to Continue Search in Multi-round RAG through Self-Practicing
- **分类: cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2505.02811v2](http://arxiv.org/pdf/2505.02811v2)**

> **作者:** Diji Yang; Linda Zeng; Jinmeng Rao; Yi Zhang
>
> **备注:** Proceedings of the 48th International ACM SIGIR 2025
>
> **摘要:** Retrieval Augmented Generation (RAG) has shown strong capability in enhancing language models' knowledge and reducing AI generative hallucinations, driving its widespread use. However, complex tasks requiring multi-round retrieval remain challenging, and early attempts tend to be overly optimistic without a good sense of self-skepticism. Current multi-round RAG systems may continue searching even when enough information has already been retrieved, or they may provide incorrect answers without having sufficient information or knowledge. Existing solutions either require large amounts of expensive human-labeled process supervision data or lead to subpar performance. This paper aims to address these limitations by introducing a new framework, SIM-RAG, to explicitly enhance RAG systems' self-awareness and multi-round retrieval capabilities. To train SIM-RAG, we first let a RAG system self-practice multi-round retrieval, augmenting existing question-answer pairs with intermediate inner monologue reasoning steps to generate synthetic training data. For each pair, the system may explore multiple retrieval paths, which are labeled as successful if they reach the correct answer and unsuccessful otherwise. Using this data, we train a lightweight information sufficiency Critic. At inference time, the Critic evaluates whether the RAG system has retrieved sufficient information at each round, guiding retrieval decisions and improving system-level self-awareness through in-context reinforcement learning. Experiments across multiple prominent RAG benchmarks show that SIM-RAG is an effective multi-round RAG solution. Furthermore, this framework is system-efficient, adding a lightweight component to RAG without requiring modifications to existing LLMs or search engines, and data-efficient, eliminating the need for costly human-annotated mid-step retrieval process supervision data.
>
---
#### [replaced 062] Detecting Sockpuppetry on Wikipedia Using Meta-Learning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10314v2](http://arxiv.org/pdf/2506.10314v2)**

> **作者:** Luc Raszewski; Christine De Kock
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** Malicious sockpuppet detection on Wikipedia is critical to preserving access to reliable information on the internet and preventing the spread of disinformation. Prior machine learning approaches rely on stylistic and meta-data features, but do not prioritise adaptability to author-specific behaviours. As a result, they struggle to effectively model the behaviour of specific sockpuppet-groups, especially when text data is limited. To address this, we propose the application of meta-learning, a machine learning technique designed to improve performance in data-scarce settings by training models across multiple tasks. Meta-learning optimises a model for rapid adaptation to the writing style of a new sockpuppet-group. Our results show that meta-learning significantly enhances the precision of predictions compared to pre-trained models, marking an advancement in combating sockpuppetry on open editing platforms. We release a new dataset of sockpuppet investigations to foster future research in both sockpuppetry and meta-learning fields.
>
---
#### [replaced 063] CSC-SQL: Corrective Self-Consistency in Text-to-SQL via Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13271v2](http://arxiv.org/pdf/2505.13271v2)**

> **作者:** Lei Sheng; Shuai-Shuai Xu
>
> **备注:** 25 pages, 5 figures
>
> **摘要:** Large language models (LLMs) have demonstrated strong capabilities in translating natural language questions about relational databases into SQL queries. In particular, test-time scaling techniques such as Self-Consistency and Self-Correction can enhance SQL generation accuracy by increasing computational effort during inference. However, these methods have notable limitations: Self-Consistency may select suboptimal outputs despite majority votes, while Self-Correction typically addresses only syntactic errors. To leverage the strengths of both approaches, we propose CSC-SQL, a novel method that integrates Self-Consistency and Self-Correction. CSC-SQL selects the two most frequently occurring outputs from parallel sampling and feeds them into a merge revision model for correction. Additionally, we employ the Group Relative Policy Optimization (GRPO) algorithm to fine-tune both the SQL generation and revision models via reinforcement learning, significantly enhancing output quality. Experimental results confirm the effectiveness and generalizability of CSC-SQL. On the BIRD private test set, our 7B model achieves 71.72\% execution accuracy, while the 32B model achieves 73.67\%. The code has been open sourced at https://github.com/CycloneBoy/csc_sql.
>
---
#### [replaced 064] I see what you mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.00071v3](http://arxiv.org/pdf/2503.00071v3)**

> **作者:** Esam Ghaleb; Bulat Khaertdinov; Aslı Özyürek; Raquel Fernández
>
> **备注:** Accepted to Findings of ACL 2025
>
> **摘要:** In face-to-face interaction, we use multiple modalities, including speech and gestures, to communicate information and resolve references to objects. However, how representational co-speech gestures refer to objects remains understudied from a computational perspective. In this work, we address this gap by introducing a multimodal reference resolution task centred on representational gestures, while simultaneously tackling the challenge of learning robust gesture embeddings. We propose a self-supervised pre-training approach to gesture representation learning that grounds body movements in spoken language. Our experiments show that the learned embeddings align with expert annotations and have significant predictive power. Moreover, reference resolution accuracy further improves when (1) using multimodal gesture representations, even when speech is unavailable at inference time, and (2) leveraging dialogue history. Overall, our findings highlight the complementary roles of gesture and speech in reference resolution, offering a step towards more naturalistic models of human-machine interaction.
>
---
#### [replaced 065] Truth Neurons
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12182v2](http://arxiv.org/pdf/2505.12182v2)**

> **作者:** Haohang Li; Yupeng Cao; Yangyang Yu; Jordan W. Suchow; Zining Zhu
>
> **摘要:** Despite their remarkable success and deployment across diverse workflows, language models sometimes produce untruthful responses. Our limited understanding of how truthfulness is mechanistically encoded within these models jeopardizes their reliability and safety. In this paper, we propose a method for identifying representations of truthfulness at the neuron level. We show that language models contain truth neurons, which encode truthfulness in a subject-agnostic manner. Experiments conducted across models of varying scales validate the existence of truth neurons, confirming that the encoding of truthfulness at the neuron level is a property shared by many language models. The distribution patterns of truth neurons over layers align with prior findings on the geometry of truthfulness. Selectively suppressing the activations of truth neurons found through the TruthfulQA dataset degrades performance both on TruthfulQA and on other benchmarks, showing that the truthfulness mechanisms are not tied to a specific dataset. Our results offer novel insights into the mechanisms underlying truthfulness in language models and highlight potential directions toward improving their trustworthiness and reliability.
>
---
#### [replaced 066] Arabic Dialect Classification using RNNs, Transformers, and Large Language Models: A Comparative Analysis
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.19753v2](http://arxiv.org/pdf/2506.19753v2)**

> **作者:** Omar A. Essameldin; Ali O. Elbeih; Wael H. Gomaa; Wael F. Elsersy
>
> **备注:** Email Typo Update
>
> **摘要:** The Arabic language is among the most popular languages in the world with a huge variety of dialects spoken in 22 countries. In this study, we address the problem of classifying 18 Arabic dialects of the QADI dataset of Arabic tweets. RNN models, Transformer models, and large language models (LLMs) via prompt engineering are created and tested. Among these, MARBERTv2 performed best with 65% accuracy and 64% F1-score. Through the use of state-of-the-art preprocessing techniques and the latest NLP models, this paper identifies the most significant linguistic issues in Arabic dialect identification. The results corroborate applications like personalized chatbots that respond in users' dialects, social media monitoring, and greater accessibility for Arabic communities.
>
---
#### [replaced 067] PromptDSI: Prompt-based Rehearsal-free Continual Learning for Document Retrieval
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.12593v4](http://arxiv.org/pdf/2406.12593v4)**

> **作者:** Tuan-Luc Huynh; Thuy-Trang Vu; Weiqing Wang; Yinwei Wei; Trung Le; Dragan Gasevic; Yuan-Fang Li; Thanh-Toan Do
>
> **备注:** ECML PKDD 2025 Research track. Camera-ready version. Code is available at https://github.com/LouisDo2108/PromptDSI
>
> **摘要:** Differentiable Search Index (DSI) utilizes pre-trained language models to perform indexing and document retrieval via end-to-end learning without relying on external indexes. However, DSI requires full re-training to index new documents, causing significant computational inefficiencies. Continual learning (CL) offers a solution by enabling the model to incrementally update without full re-training. Existing CL solutions in document retrieval rely on memory buffers or generative models for rehearsal, which is infeasible when accessing previous training data is restricted due to privacy concerns. To this end, we introduce PromptDSI, a prompt-based, rehearsal-free continual learning approach for document retrieval. PromptDSI follows the Prompt-based Continual Learning (PCL) framework, using learnable prompts to efficiently index new documents without accessing previous documents or queries. To improve retrieval latency, we remove the initial forward pass of PCL, which otherwise greatly increases training and inference time, with a negligible trade-off in performance. Additionally, we introduce a novel topic-aware prompt pool that employs neural topic embeddings as fixed keys, eliminating the instability of prompt key optimization while maintaining competitive performance with existing PCL prompt pools. In a challenging rehearsal-free continual learning setup, we demonstrate that PromptDSI variants outperform rehearsal-based baselines, match the strong cache-based baseline in mitigating forgetting, and significantly improving retrieval performance on new corpora.
>
---
#### [replaced 068] AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15676v2](http://arxiv.org/pdf/2502.15676v2)**

> **作者:** Zhining Zhang; Chuanyang Jin; Mung Yao Jia; Shunchi Zhang; Tianmin Shu
>
> **备注:** 39 pages, 10 figures, 13 tables. Website at https://chuanyangjin.com/AutoToM/
>
> **摘要:** Theory of Mind (ToM), the ability to understand people's minds based on their behavior, is key to developing socially intelligent agents. Current approaches to ToM reasoning either rely on prompting Large Language Models (LLMs), which are prone to systematic errors, or use handcrafted, rigid agent models for model-based inference, which are more robust but fail to generalize across domains. In this work, we introduce AutoToM, an automated agent modeling method for scalable, robust, and interpretable mental inference. Given a ToM problem, AutoToM first proposes an initial agent model and then performs automated Bayesian inverse planning based on this model, leveraging an LLM backend. Guided by inference uncertainty, it iteratively refines the model by introducing additional mental variables and/or incorporating more timesteps in the context. Across five diverse benchmarks, AutoToM outperforms existing ToM methods and even large reasoning models. Additionally, we show that AutoToM can produce human-like confidence estimates and enable online mental inference for embodied decision-making.
>
---
#### [replaced 069] FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Foundation Models
- **分类: cs.DC; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.09432v4](http://arxiv.org/pdf/2410.09432v4)**

> **作者:** Raghav Singhal; Kaustubh Ponkshe; Praneeth Vepakomma
>
> **备注:** ACL 2025 - Oral. Raghav Singhal and Kaustubh Ponkshe contributed equally to this work
>
> **摘要:** Low-Rank Adaptation (LoRA) is a popular technique for efficient fine-tuning of foundation models. However, applying LoRA in federated learning environments, where data is distributed across multiple clients, presents unique challenges. Existing methods rely on traditional federated averaging of LoRA adapters, resulting in inexact updates. To address this, we propose Federated Exact LoRA, or FedEx-LoRA, which adds a residual error term to the pretrained frozen weight matrix. Our approach achieves exact updates with minimal computational and communication overhead, preserving LoRA's efficiency. We evaluate the method on various models across arithmetic reasoning, commonsense reasoning, natural language understanding and natural language generation tasks, showing consistent performance gains over state-of-the-art methods across multiple settings. Through extensive analysis, we quantify that the deviations in updates from the ideal solution are significant, highlighting the need for exact aggregation. Our method's simplicity, efficiency, and broad applicability position it as a promising solution for accurate and effective federated fine-tuning of foundation models. Our code is publicly available at https://github.com/RaghavSinghal10/fedex-lora.
>
---
#### [replaced 070] ScienceMeter: Tracking Scientific Knowledge Updates in Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24302v2](http://arxiv.org/pdf/2505.24302v2)**

> **作者:** Yike Wang; Shangbin Feng; Yulia Tsvetkov; Hannaneh Hajishirzi
>
> **摘要:** Large Language Models (LLMs) are increasingly used to support scientific research, but their knowledge of scientific advancements can quickly become outdated. We introduce ScienceMeter, a new framework for evaluating scientific knowledge update methods over scientific knowledge spanning the past, present, and future. ScienceMeter defines three metrics: knowledge preservation, the extent to which models' understanding of previously learned papers are preserved; knowledge acquisition, how well scientific claims from newly introduced papers are acquired; and knowledge projection, the ability of the updated model to anticipate or generalize to related scientific claims that may emerge in the future. Using ScienceMeter, we examine the scientific knowledge of LLMs on claim judgment and generation tasks across a curated dataset of 15,444 scientific papers and 30,888 scientific claims from ten domains including medicine, biology, materials science, and computer science. We evaluate five representative knowledge update approaches including training- and inference-time methods. With extensive experiments, we find that the best-performing knowledge update methods can preserve only 85.9% of existing knowledge, acquire 71.7% of new knowledge, and project 37.7% of future knowledge. Inference-based methods work for larger models, whereas smaller models require training to achieve comparable performance. Cross-domain analysis reveals that performance on these objectives is correlated. Even when applying on specialized scientific LLMs, existing knowledge update methods fail to achieve these objectives collectively, underscoring that developing robust scientific knowledge update mechanisms is both crucial and challenging.
>
---
#### [replaced 071] Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18968v4](http://arxiv.org/pdf/2502.18968v4)**

> **作者:** Kuang Wang; Xianfei Li; Shenghao Yang; Li Zhou; Feng Jiang; Haizhou Li
>
> **备注:** 9 pages. Accepted to ACL 2025. Camera-ready version
>
> **摘要:** User simulators are crucial for replicating human interactions with dialogue systems, supporting both collaborative training and automatic evaluation, especially for large language models (LLMs). However, current role-playing methods face challenges such as a lack of utterance-level authenticity and user-level diversity, often hindered by role confusion and dependence on predefined profiles of well-known figures. In contrast, direct simulation focuses solely on text, neglecting implicit user traits like personality and conversation-level consistency. To address these issues, we introduce the User Simulator with Implicit Profiles (USP), a framework that infers implicit user profiles from human-machine interactions to simulate personalized and realistic dialogues. We first develop an LLM-driven extractor with a comprehensive profile schema, then refine the simulation using conditional supervised fine-tuning and reinforcement learning with cycle consistency, optimizing at both the utterance and conversation levels. Finally, a diverse profile sampler captures the distribution of real-world user profiles. Experimental results show that USP outperforms strong baselines in terms of authenticity and diversity while maintaining comparable consistency. Additionally, using USP to evaluate LLM on dynamic multi-turn aligns well with mainstream benchmarks, demonstrating its effectiveness in real-world applications.
>
---
#### [replaced 072] Margin Matching Preference Optimization: Enhanced Model Alignment with Granular Feedback
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.03145v2](http://arxiv.org/pdf/2410.03145v2)**

> **作者:** Kyuyoung Kim; Ah Jeong Seo; Hao Liu; Jinwoo Shin; Kimin Lee
>
> **备注:** EMNLP 2024 Findings
>
> **摘要:** Large language models (LLMs) fine-tuned with alignment techniques, such as reinforcement learning from human feedback, have been instrumental in developing some of the most capable AI systems to date. Despite their success, existing methods typically rely on simple binary labels, such as those indicating preferred outputs in pairwise preferences, which fail to capture the subtle differences in relative quality between pairs. To address this limitation, we introduce an approach called Margin Matching Preference Optimization (MMPO), which incorporates relative quality margins into optimization, leading to improved LLM policies and reward models. Specifically, given quality margins in pairwise preferences, we design soft target probabilities based on the Bradley-Terry model, which are then used to train models with the standard cross-entropy objective. Experiments with both human and AI feedback data demonstrate that MMPO consistently outperforms baseline methods, often by a substantial margin, on popular benchmarks including MT-bench and RewardBench. Notably, the 7B model trained with MMPO achieves state-of-the-art performance on RewardBench as of June 2024, outperforming other models of the same scale. Our analysis also shows that MMPO is more robust to overfitting, leading to better-calibrated models.
>
---
#### [replaced 073] SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.18797v2](http://arxiv.org/pdf/2411.18797v2)**

> **作者:** Haomin Zhuang; Yihua Zhang; Kehan Guo; Jinghan Jia; Gaowen Liu; Sijia Liu; Xiangliang Zhang
>
> **备注:** Accepted to ACL'25
>
> **摘要:** Recent advancements in LLMs unlearning have shown remarkable success in removing unwanted data-model influences while preserving the model's utility for legitimate knowledge. Despite these strides, sparse Mixture-of-Experts (MoE) LLMs--a key subset of the LLM family--have remained unexplored in the context of unlearning. As MoE LLMs are celebrated for their exceptional performance, we ask:How can unlearning be performed effectively and efficiently on MoE LLMs? Our pilot study shows that the dynamic routing nature of MoE LLMs introduces unique challenges, leading to excessive forgetting, uncontrolled knowledge erasure and substantial utility drops when existing unlearning methods are applied. To address this, we propose a novel Selected-Expert Unlearning Framework (SEUF). Through expert attribution, unlearning is concentrated on the most actively engaged experts for the specified knowledge. Concurrently, an anchor loss is applied to the router to stabilize the active state of this targeted expert, ensuring focused and controlled unlearning. SEUF is compatible with various standard unlearning algorithms. Extensive experiments demonstrate that SEUF enhances both forget quality up to 5% and model utility by 35% on MoE LLMs across various benchmarks and LLM architectures (compared to standard unlearning algorithms), while only unlearning 0.06% of the model parameters.
>
---
#### [replaced 074] A Survey of Test-Time Compute: From Intuitive Inference to Deliberate Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.02497v3](http://arxiv.org/pdf/2501.02497v3)**

> **作者:** Yixin Ji; Juntao Li; Yang Xiang; Hai Ye; Kaixin Wu; Kai Yao; Jia Xu; Linjian Mo; Min Zhang
>
> **备注:** Work in progress
>
> **摘要:** The remarkable performance of the o1 model in complex reasoning demonstrates that test-time compute scaling can further unlock the model's potential, enabling powerful System-2 thinking. However, there is still a lack of comprehensive surveys for test-time compute scaling. We trace the concept of test-time compute back to System-1 models. In System-1 models, test-time compute addresses distribution shifts and improves robustness and generalization through parameter updating, input modification, representation editing, and output calibration. In System-2 models, it enhances the model's reasoning ability to solve complex problems through repeated sampling, self-correction, and tree search. We organize this survey according to the trend of System-1 to System-2 thinking, highlighting the key role of test-time compute in the transition from System-1 models to weak System-2 models, and then to strong System-2 models. We also point out advanced topics and future directions.
>
---
#### [replaced 075] Organize the Web: Constructing Domains Enhances Pre-Training Data Curation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.10341v2](http://arxiv.org/pdf/2502.10341v2)**

> **作者:** Alexander Wettig; Kyle Lo; Sewon Min; Hannaneh Hajishirzi; Danqi Chen; Luca Soldaini
>
> **备注:** Accepted at ICML 2025. Project page: https://weborganizer.allen.ai
>
> **摘要:** Modern language models are trained on large, unstructured datasets consisting of trillions of tokens and obtained by crawling the web. The unstructured nature makes it difficult to reason about their contents and develop systematic approaches to data curation. In this paper, we unpack monolithic web corpora by developing taxonomies of their contents and organizing them into domains. We introduce WebOrganizer, a framework for organizing web pages in terms of both their topic and format. Using these two complementary notions of domains, we automatically annotate pre-training data by distilling annotations from a large language model into efficient classifiers. This allows us to study how data from different domains should be mixed to improve models on downstream tasks, and we show that we can combine insights about effective topics and formats to further boost performance. We demonstrate that our domain mixing also improves existing methods that select data based on quality. Furthermore, we study and compare how quality-based methods will implicitly change the domain mixture. Overall, our work demonstrates that constructing and mixing domains provides a valuable complement to quality-based data curation methods, opening new avenues for effective and insightful pre-training data curation.
>
---
#### [replaced 076] How to Retrieve Examples in In-context Learning to Improve Conversational Emotion Recognition using Large Language Models?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.20199v2](http://arxiv.org/pdf/2506.20199v2)**

> **作者:** Mengqi Wang; Tiantian Feng; Shrikanth Narayanan
>
> **摘要:** Large language models (LLMs) have enabled a wide variety of real-world applications in various domains. However, creating a high-performing application with high accuracy remains challenging, particularly for subjective tasks like emotion recognition. Inspired by the SLT 2024 GenSER Challenge, this study investigates approaches to improving conversational emotion recognition (CER) by LLMs. Specifically, we explore how to retrieve high-quality examples in in-context learning (ICL) to enhance CER. We propose various strategies based on random and augmented example retrieval and also analyze the impact of conversational context on CER accuracy. Experiments were conducted on the three datasets including IEMOCAP, MELD and EmoryNLP. The results show that augmented example retrieval consistently outperforms other techniques under investigation across all datasets, highlighting the importance of retrieving coherent targeted examples and enhancing them through paraphrasing.
>
---
#### [replaced 077] From Outcomes to Processes: Guiding PRM Learning from ORM for Inference-Time Alignment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.12446v2](http://arxiv.org/pdf/2506.12446v2)**

> **作者:** Bin Xie; Bingbing Xu; Yige Yuan; Shengmao Zhu; Huawei Shen
>
> **摘要:** Inference-time alignment methods have gained significant attention for their efficiency and effectiveness in aligning large language models (LLMs) with human preferences. However, existing dominant approaches using reward-guided search (RGS) primarily rely on outcome reward models (ORMs), which suffer from a critical granularity mismatch: ORMs are designed to provide outcome rewards for complete responses, while RGS methods rely on process rewards to guide the policy, leading to inconsistent scoring and suboptimal alignment. To address this challenge, we introduce process reward models (PRMs) into RGS and argue that an ideal PRM should satisfy two objectives: Score Consistency, ensuring coherent evaluation across partial and complete responses, and Preference Consistency, aligning partial sequence assessments with human preferences. Based on these, we propose SP-PRM, a novel dual-consistency framework integrating score consistency-based and preference consistency-based partial evaluation modules without relying on human annotation. Extensive experiments on dialogue, summarization, and reasoning tasks demonstrate that SP-PRM substantially enhances existing RGS methods, achieving a 3.6%-10.3% improvement in GPT-4 evaluation scores across all tasks.
>
---
#### [replaced 078] Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.15627v4](http://arxiv.org/pdf/2406.15627v4)**

> **作者:** Roman Vashurin; Ekaterina Fadeeva; Artem Vazhentsev; Lyudmila Rvanova; Akim Tsvigun; Daniil Vasilev; Rui Xing; Abdelrahman Boda Sadallah; Kirill Grishchenkov; Sergey Petrakov; Alexander Panchenko; Timothy Baldwin; Preslav Nakov; Maxim Panov; Artem Shelmanov
>
> **备注:** Published at TACL 2025, presented at ACL 2025. Roman Vashurin, Ekaterina Fadeeva, Artem Vazhentsev contributed equally
>
> **摘要:** The rapid proliferation of large language models (LLMs) has stimulated researchers to seek effective and efficient approaches to deal with LLM hallucinations and low-quality outputs. Uncertainty quantification (UQ) is a key element of machine learning applications in dealing with such challenges. However, research to date on UQ for LLMs has been fragmented in terms of techniques and evaluation methodologies. In this work, we address this issue by introducing a novel benchmark that implements a collection of state-of-the-art UQ baselines and offers an environment for controllable and consistent evaluation of novel UQ techniques over various text generation tasks. Our benchmark also supports the assessment of confidence normalization methods in terms of their ability to provide interpretable scores. Using our benchmark, we conduct a large-scale empirical investigation of UQ and normalization techniques across eleven tasks, identifying the most effective approaches. Code: https://github.com/IINemo/lm-polygraph Benchmark: https://huggingface.co/LM-Polygraph
>
---
#### [replaced 079] Enough Coin Flips Can Make LLMs Act Bayesian
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.04722v2](http://arxiv.org/pdf/2503.04722v2)**

> **作者:** Ritwik Gupta; Rodolfo Corona; Jiaxin Ge; Eric Wang; Dan Klein; Trevor Darrell; David M. Chan
>
> **备注:** ACL 2025 Main
>
> **摘要:** Large language models (LLMs) exhibit the ability to generalize given few-shot examples in their input prompt, an emergent capability known as in-context learning (ICL). We investigate whether LLMs use ICL to perform structured reasoning in ways that are consistent with a Bayesian framework or rely on pattern matching. Using a controlled setting of biased coin flips, we find that: (1) LLMs often possess biased priors, causing initial divergence in zero-shot settings, (2) in-context evidence outweighs explicit bias instructions, (3) LLMs broadly follow Bayesian posterior updates, with deviations primarily due to miscalibrated priors rather than flawed updates, and (4) attention magnitude has negligible effect on Bayesian inference. With sufficient demonstrations of biased coin flips via ICL, LLMs update their priors in a Bayesian manner.
>
---
#### [replaced 080] SConU: Selective Conformal Uncertainty in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2504.14154v2](http://arxiv.org/pdf/2504.14154v2)**

> **作者:** Zhiyuan Wang; Qingni Wang; Yue Zhang; Tianlong Chen; Xiaofeng Zhu; Xiaoshuang Shi; Kaidi Xu
>
> **备注:** Accepted by ACL 2025 Main
>
> **摘要:** As large language models are increasingly utilized in real-world applications, guarantees of task-specific metrics are essential for their reliable deployment. Previous studies have introduced various criteria of conformal uncertainty grounded in split conformal prediction, which offer user-specified correctness coverage. However, existing frameworks often fail to identify uncertainty data outliers that violate the exchangeability assumption, leading to unbounded miscoverage rates and unactionable prediction sets. In this paper, we propose a novel approach termed Selective Conformal Uncertainty (SConU), which, for the first time, implements significance tests, by developing two conformal p-values that are instrumental in determining whether a given sample deviates from the uncertainty distribution of the calibration set at a specific manageable risk level. Our approach not only facilitates rigorous management of miscoverage rates across both single-domain and interdisciplinary contexts, but also enhances the efficiency of predictions. Furthermore, we comprehensively analyze the components of the conformal procedures, aiming to approximate conditional coverage, particularly in high-stakes question-answering tasks.
>
---
#### [replaced 081] Agentic Medical Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge
- **分类: cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2502.13010v3](http://arxiv.org/pdf/2502.13010v3)**

> **作者:** Mohammad Reza Rezaei; Reza Saadati Fard; Jayson L. Parker; Rahul G. Krishnan; Milad Lankarany
>
> **摘要:** Large Language Models (LLMs) have significantly advanced medical question-answering by leveraging extensive clinical data and medical literature. However, the rapid evolution of medical knowledge and the labor-intensive process of manually updating domain-specific resources pose challenges to the reliability of these systems. To address this, we introduce Agentic Medical Graph-RAG (AMG-RAG), a comprehensive framework that automates the construction and continuous updating of medical knowledge graphs, integrates reasoning, and retrieves current external evidence, such as PubMed and WikiSearch. By dynamically linking new findings and complex medical concepts, AMG-RAG not only improves accuracy but also enhances interpretability in medical queries. Evaluations on the MEDQA and MEDMCQA benchmarks demonstrate the effectiveness of AMG-RAG, achieving an F1 score of 74.1 percent on MEDQA and an accuracy of 66.34 percent on MEDMCQA, outperforming both comparable models and those 10 to 100 times larger. Notably, these improvements are achieved without increasing computational overhead, highlighting the critical role of automated knowledge graph generation and external evidence retrieval in delivering up-to-date, trustworthy medical insights.
>
---
#### [replaced 082] LegiGPT: Party Politics and Transport Policy with Large Language Model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.16692v2](http://arxiv.org/pdf/2506.16692v2)**

> **作者:** Hyunsoo Yun; Eun Hak Lee
>
> **备注:** Updated title to match published version. Added DOI and journal reference to PDF
>
> **摘要:** Given the significant influence of lawmakers' political ideologies on legislative decision-making, analyzing their impact on transportation-related policymaking is of critical importance. This study introduces a novel framework that integrates a large language model (LLM) with explainable artificial intelligence (XAI) to analyze transportation-related legislative proposals. Legislative bill data from South Korea's 21st National Assembly were used to identify key factors shaping transportation policymaking. These include political affiliations and sponsor characteristics. The LLM was employed to classify transportation-related bill proposals through a stepwise filtering process based on keywords, sentences, and contextual relevance. XAI techniques were then applied to examine the relationships between political party affiliation and associated attributes. The results revealed that the number and proportion of conservative and progressive sponsors, along with district size and electoral population, were critical determinants shaping legislative outcomes. These findings suggest that both parties contributed to bipartisan legislation through different forms of engagement, such as initiating or supporting proposals. This integrated approach offers a valuable tool for understanding legislative dynamics and guiding future policy development, with broader implications for infrastructure planning and governance.
>
---
#### [replaced 083] DReSS: Data-driven Regularized Structured Streamlining for Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.17905v3](http://arxiv.org/pdf/2501.17905v3)**

> **作者:** Mingkuan Feng; Jinyang Wu; Shuai Zhang; Pengpeng Shao; Ruihan Jin; Zhengqi Wen; Jianhua Tao; Feihu Che
>
> **摘要:** Large language models (LLMs) have achieved significant progress across various domains, but their increasing scale results in high computational and memory costs. Recent studies have revealed that LLMs exhibit sparsity, providing the potential to reduce model size through pruning techniques. However, existing pruning methods typically follow a prune-then-finetune paradigm. Since the pruned components still contain valuable information, their direct removal often leads to irreversible performance degradation, imposing a substantial computational burden to recover performance during finetuning. In this paper, we propose a novel paradigm that first applies regularization, then prunes, and finally finetunes. Based on this paradigm, we introduce DReSS, a simple and effective Data-driven Regularized Structured Streamlining method for LLMs. By leveraging a small amount of data to regularize the components to be pruned, DReSS explicitly transfers the important information to the remaining parts of the model in advance. Compared to direct pruning, this can reduce the information loss caused by parameter removal, thereby enhancing its language modeling capabilities. Experimental results demonstrate that DReSS significantly outperforms existing pruning methods even under extreme pruning ratios, significantly reducing latency and increasing throughput.
>
---
#### [replaced 084] TyphoFormer: Language-Augmented Transformer for Accurate Typhoon Track Forecasting
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.17609v2](http://arxiv.org/pdf/2506.17609v2)**

> **作者:** Lincan Li; Eren Erman Ozguven; Yue Zhao; Guang Wang; Yiqun Xie; Yushun Dong
>
> **备注:** Short research paper
>
> **摘要:** Accurate typhoon track forecasting is crucial for early system warning and disaster response. While Transformer-based models have demonstrated strong performance in modeling the temporal dynamics of dense trajectories of humans and vehicles in smart cities, they usually lack access to broader contextual knowledge that enhances the forecasting reliability of sparse meteorological trajectories, such as typhoon tracks. To address this challenge, we propose TyphoFormer, a novel framework that incorporates natural language descriptions as auxiliary prompts to improve typhoon trajectory forecasting. For each time step, we use Large Language Model (LLM) to generate concise textual descriptions based on the numerical attributes recorded in the North Atlantic hurricane database. The language descriptions capture high-level meteorological semantics and are embedded as auxiliary special tokens prepended to the numerical time series input. By integrating both textual and sequential information within a unified Transformer encoder, TyphoFormer enables the model to leverage contextual cues that are otherwise inaccessible through numerical features alone. Extensive experiments are conducted on HURDAT2 benchmark, results show that TyphoFormer consistently outperforms other state-of-the-art baseline methods, particularly under challenging scenarios involving nonlinear path shifts and limited historical observations.
>
---
#### [replaced 085] Kalahi: A handcrafted, grassroots cultural LLM evaluation suite for Filipino
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.15380v4](http://arxiv.org/pdf/2409.15380v4)**

> **作者:** Jann Railey Montalan; Jian Gang Ngui; Wei Qi Leong; Yosephine Susanto; Hamsawardhini Rengarajan; Alham Fikri Aji; William Chandra Tjhi
>
> **备注:** Accepted for presentation at Paclic 38, 2024
>
> **摘要:** Multilingual large language models (LLMs) today may not necessarily provide culturally appropriate and relevant responses to its Filipino users. We introduce Kalahi, a cultural LLM evaluation suite collaboratively created by native Filipino speakers. It is composed of 150 high-quality, handcrafted and nuanced prompts that test LLMs for generations that are relevant to shared Filipino cultural knowledge and values. Strong LLM performance in Kalahi indicates a model's ability to generate responses similar to what an average Filipino would say or do in a given situation. We conducted experiments on LLMs with multilingual and Filipino language support. Results show that Kalahi, while trivial for Filipinos, is challenging for LLMs, with the best model answering only 46.0% of the questions correctly compared to native Filipino performance of 89.10%. Thus, Kalahi can be used to accurately and reliably evaluate Filipino cultural representation in LLMs.
>
---
#### [replaced 086] Double Entendre: Robust Audio-Based AI-Generated Lyrics Detection via Multi-View Fusion
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.15981v2](http://arxiv.org/pdf/2506.15981v2)**

> **作者:** Markus Frohmann; Gabriel Meseguer-Brocal; Markus Schedl; Elena V. Epure
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** The rapid advancement of AI-based music generation tools is revolutionizing the music industry but also posing challenges to artists, copyright holders, and providers alike. This necessitates reliable methods for detecting such AI-generated content. However, existing detectors, relying on either audio or lyrics, face key practical limitations: audio-based detectors fail to generalize to new or unseen generators and are vulnerable to audio perturbations; lyrics-based methods require cleanly formatted and accurate lyrics, unavailable in practice. To overcome these limitations, we propose a novel, practically grounded approach: a multimodal, modular late-fusion pipeline that combines automatically transcribed sung lyrics and speech features capturing lyrics-related information within the audio. By relying on lyrical aspects directly from audio, our method enhances robustness, mitigates susceptibility to low-level artifacts, and enables practical applicability. Experiments show that our method, DE-detect, outperforms existing lyrics-based detectors while also being more robust to audio perturbations. Thus, it offers an effective, robust solution for detecting AI-generated music in real-world scenarios. Our code is available at https://github.com/deezer/robust-AI-lyrics-detection.
>
---
#### [replaced 087] Gumiho: A Hybrid Architecture to Prioritize Early Tokens in Speculative Decoding
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.10135v2](http://arxiv.org/pdf/2503.10135v2)**

> **作者:** Jinze Li; Yixing Xu; Haiduo Huang; Xuanwu Yin; Dong Li; Edith C. H. Ngai; Emad Barsoum
>
> **备注:** Accepted to the 42nd International Conference on Machine Learning (ICML 2025). Code: https://github.com/AMD-AIG-AIMA/Gumiho
>
> **摘要:** Speculative decoding (SPD) aims to accelerate the auto-regressive token generation process of a target Large Language Model (LLM). Some approaches employ a draft model with multiple heads to predict a sequence of future tokens, where each head handles a token in the sequence. The target LLM verifies the predicted sequence and accepts aligned tokens, enabling efficient multi-token generation. However, existing methods assume that all tokens within a sequence are equally important, employing identical head structures and relying on a single-generation paradigm, either serial or parallel. To this end, we theoretically demonstrate that initial tokens in the draft sequence are more important than later ones. Building on this insight, we propose Gumiho, a hybrid model combining serial and parallel heads. Specifically, given the critical importance of early tokens, we employ a sophisticated Transformer architecture for the early draft heads in a serial configuration to improve accuracy. For later tokens, we utilize multiple lightweight MLP heads operating in parallel to enhance efficiency. By allocating more advanced model structures and longer running times to the early heads, Gumiho achieves improved overall performance. The experimental results demonstrate that our method outperforms existing approaches, fully validating its effectiveness.
>
---
#### [replaced 088] Creativity in AI: Progresses and Challenges
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.17218v5](http://arxiv.org/pdf/2410.17218v5)**

> **作者:** Mete Ismayilzada; Debjit Paul; Antoine Bosselut; Lonneke van der Plas
>
> **备注:** minor updates to content + contact information
>
> **摘要:** Creativity is the ability to produce novel, useful, and surprising ideas, and has been widely studied as a crucial aspect of human cognition. Machine creativity on the other hand has been a long-standing challenge. With the rise of advanced generative AI, there has been renewed interest and debate regarding AI's creative capabilities. Therefore, it is imperative to revisit the state of creativity in AI and identify key progresses and remaining challenges. In this work, we survey leading works studying the creative capabilities of AI systems, focusing on creative problem-solving, linguistic, artistic, and scientific creativity. Our review suggests that while the latest AI models are largely capable of producing linguistically and artistically creative outputs such as poems, images, and musical pieces, they struggle with tasks that require creative problem-solving, abstract thinking and compositionality and their generations suffer from a lack of diversity, originality, long-range incoherence and hallucinations. We also discuss key questions concerning copyright and authorship issues with generative models. Furthermore, we highlight the need for a comprehensive evaluation of creativity that is process-driven and considers several dimensions of creativity. Finally, we propose future research directions to improve the creativity of AI outputs, drawing inspiration from cognitive science and psychology.
>
---
#### [replaced 089] Tracing Intricate Cues in Dialogue: Joint Graph Structure and Sentiment Dynamics for Multimodal Emotion Recognition
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.21536v2](http://arxiv.org/pdf/2407.21536v2)**

> **作者:** Jiang Li; Xiaoping Wang; Zhigang Zeng
>
> **备注:** Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence
>
> **摘要:** Multimodal emotion recognition in conversation (MERC) has garnered substantial research attention recently. Existing MERC methods face several challenges: (1) they fail to fully harness direct inter-modal cues, possibly leading to less-than-thorough cross-modal modeling; (2) they concurrently extract information from the same and different modalities at each network layer, potentially triggering conflicts from the fusion of multi-source data; (3) they lack the agility required to detect dynamic sentimental changes, perhaps resulting in inaccurate classification of utterances with abrupt sentiment shifts. To address these issues, a novel approach named GraphSmile is proposed for tracking intricate emotional cues in multimodal dialogues. GraphSmile comprises two key components, i.e., GSF and SDP modules. GSF ingeniously leverages graph structures to alternately assimilate inter-modal and intra-modal emotional dependencies layer by layer, adequately capturing cross-modal cues while effectively circumventing fusion conflicts. SDP is an auxiliary task to explicitly delineate the sentiment dynamics between utterances, promoting the model's ability to distinguish sentimental discrepancies. GraphSmile is effortlessly applied to multimodal sentiment analysis in conversation (MSAC), thus enabling simultaneous execution of MERC and MSAC tasks. Empirical results on multiple benchmarks demonstrate that GraphSmile can handle complex emotional and sentimental patterns, significantly outperforming baseline models.
>
---
#### [replaced 090] HalluSegBench: Counterfactual Visual Reasoning for Segmentation Hallucination Evaluation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.21546v2](http://arxiv.org/pdf/2506.21546v2)**

> **作者:** Xinzhuo Li; Adheesh Juvekar; Xingyou Liu; Muntasir Wahed; Kiet A. Nguyen; Ismini Lourentzou
>
> **备注:** Project webpage: https://plan-lab.github.io/hallusegbench/
>
> **摘要:** Recent progress in vision-language segmentation has significantly advanced grounded visual understanding. However, these models often exhibit hallucinations by producing segmentation masks for objects not grounded in the image content or by incorrectly labeling irrelevant regions. Existing evaluation protocols for segmentation hallucination primarily focus on label or textual hallucinations without manipulating the visual context, limiting their capacity to diagnose critical failures. In response, we introduce HalluSegBench, the first benchmark specifically designed to evaluate hallucinations in visual grounding through the lens of counterfactual visual reasoning. Our benchmark consists of a novel dataset of 1340 counterfactual instance pairs spanning 281 unique object classes, and a set of newly introduced metrics that quantify hallucination sensitivity under visually coherent scene edits. Experiments on HalluSegBench with state-of-the-art vision-language segmentation models reveal that vision-driven hallucinations are significantly more prevalent than label-driven ones, with models often persisting in false segmentation, highlighting the need for counterfactual reasoning to diagnose grounding fidelity.
>
---
#### [replaced 091] Time-MQA: Time Series Multi-Task Question Answering with Context Enhancement
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.01875v2](http://arxiv.org/pdf/2503.01875v2)**

> **作者:** Yaxuan Kong; Yiyuan Yang; Yoontae Hwang; Wenjie Du; Stefan Zohren; Zhangyang Wang; Ming Jin; Qingsong Wen
>
> **备注:** Annual Meeting of the Association for Computational Linguistics (ACL 2025, Main)
>
> **摘要:** Time series data are foundational in finance, healthcare, and energy domains. However, most existing methods and datasets remain focused on a narrow spectrum of tasks, such as forecasting or anomaly detection. To bridge this gap, we introduce Time Series Multi-Task Question Answering (Time-MQA), a unified framework that enables natural language queries across multiple time series tasks - numerical analytical tasks and open-ended question answering with reasoning. Central to Time-MQA is the TSQA dataset, a large-scale dataset containing $\sim$200k question-answer pairs derived from diverse time series spanning environment, traffic, etc. This comprehensive resource covers various time series lengths and promotes robust model development. We further demonstrate how continually pre-training large language models (Mistral 7B, Llama-3 8B, and Qwen-2.5 7B) on the TSQA dataset enhanced time series reasoning capabilities, moving beyond mere numeric tasks and enabling more advanced and intuitive interactions with temporal data. The complete TSQA dataset, models, user study questionnaires for evaluation, and other related materials have been open-sourced.
>
---
#### [replaced 092] What can large language models do for sustainable food?
- **分类: cs.CY; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.04734v2](http://arxiv.org/pdf/2503.04734v2)**

> **作者:** Anna T. Thomas; Adam Yee; Andrew Mayne; Maya B. Mathur; Dan Jurafsky; Kristina Gligorić
>
> **备注:** ICML camera ready version
>
> **摘要:** Food systems are responsible for a third of human-caused greenhouse gas emissions. We investigate what Large Language Models (LLMs) can contribute to reducing the environmental impacts of food production. We define a typology of design and prediction tasks based on the sustainable food literature and collaboration with domain experts, and evaluate six LLMs on four tasks in our typology. For example, for a sustainable protein design task, food science experts estimated that collaboration with an LLM can reduce time spent by 45% on average, compared to 22% for collaboration with another expert human food scientist. However, for a sustainable menu design task, LLMs produce suboptimal solutions when instructed to consider both human satisfaction and climate impacts. We propose a general framework for integrating LLMs with combinatorial optimization to improve reasoning capabilities. Our approach decreases emissions of food choices by 79% in a hypothetical restaurant while maintaining participants' satisfaction with their set of choices. Our results demonstrate LLMs' potential, supported by optimization techniques, to accelerate sustainable food development and adoption.
>
---
#### [replaced 093] Better Aligned with Survey Respondents or Training Data? Unveiling Political Leanings of LLMs on U.S. Supreme Court Cases
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18282v3](http://arxiv.org/pdf/2502.18282v3)**

> **作者:** Shanshan Xu; T. Y. S. S Santosh; Yanai Elazar; Quirin Vogel; Barbara Plank; Matthias Grabmair
>
> **摘要:** Recent works have shown that Large Language Models (LLMs) have a tendency to memorize patterns and biases present in their training data, raising important questions about how such memorized content influences model behavior. One such concern is the emergence of political bias in LLM outputs. In this paper, we investigate the extent to which LLMs' political leanings reflect memorized patterns from their pretraining corpora. We propose a method to quantitatively evaluate political leanings embedded in the large pretraining corpora. Subsequently we investigate to whom are the LLMs' political leanings more aligned with, their pretrainig corpora or the surveyed human opinions. As a case study, we focus on probing the political leanings of LLMs in 32 US Supreme Court cases, addressing contentious topics such as abortion and voting rights. Our findings reveal that LLMs strongly reflect the political leanings in their training data, and no strong correlation is observed with their alignment to human opinions as expressed in surveys. These results underscore the importance of responsible curation of training data, and the methodology for auditing the memorization in LLMs to ensure human-AI alignment.
>
---
#### [replaced 094] Data Quality Issues in Multilingual Speech Datasets: The Need for Sociolinguistic Awareness and Proactive Language Planning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.17525v2](http://arxiv.org/pdf/2506.17525v2)**

> **作者:** Mingfei Lau; Qian Chen; Yeming Fang; Tingting Xu; Tongzhou Chen; Pavel Golik
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** Our quality audit for three widely used public multilingual speech datasets - Mozilla Common Voice 17.0, FLEURS, and Vox Populi - shows that in some languages, these datasets suffer from significant quality issues, which may obfuscate downstream evaluation results while creating an illusion of success. We divide these quality issues into two categories: micro-level and macro-level. We find that macro-level issues are more prevalent in less institutionalized, often under-resourced languages. We provide a case analysis of Taiwanese Southern Min (nan_tw) that highlights the need for proactive language planning (e.g. orthography prescriptions, dialect boundary definition) and enhanced data quality control in the dataset creation process. We conclude by proposing guidelines and recommendations to mitigate these issues in future dataset development, emphasizing the importance of sociolinguistic awareness and language planning principles. Furthermore, we encourage research into how this creation process itself can be leveraged as a tool for community-led language planning and revitalization.
>
---
#### [replaced 095] CHARTOM: A Visual Theory-of-Mind Benchmark for LLMs on Misleading Charts
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2408.14419v3](http://arxiv.org/pdf/2408.14419v3)**

> **作者:** Shubham Bharti; Shiyun Cheng; Jihyun Rho; Jianrui Zhang; Mu Cai; Yong Jae Lee; Martina Rau; Xiaojin Zhu
>
> **摘要:** We introduce CHARTOM, a visual theory-of-mind benchmark designed to evaluate multimodal large language models' capability to understand and reason about misleading data visualizations though charts. CHARTOM consists of carefully designed charts and associated questions that require a language model to not only correctly comprehend the factual content in the chart (the FACT question) but also judge whether the chart will be misleading to a human readers (the MIND question), a dual capability with significant societal benefits. We detail the construction of our benchmark including its calibration on human performance and estimation of MIND ground truth called the Human Misleadingness Index. We evaluated several leading LLMs -- including GPT, Claude, Gemini, Qwen, Llama, and Llava series models -- on the CHARTOM dataset and found that it was challenging to all models both on FACT and MIND questions. This highlights the limitations of current LLMs and presents significant opportunity for future LLMs to improve on understanding misleading charts.
>
---
#### [replaced 096] A Comprehensive Evaluation of Semantic Relation Knowledge of Pretrained Language Models and Humans
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.01131v3](http://arxiv.org/pdf/2412.01131v3)**

> **作者:** Zhihan Cao; Hiroaki Yamada; Simone Teufel; Takenobu Tokunaga
>
> **备注:** This manuscript is currently under review at Language Resources and Evaluation
>
> **摘要:** Recently, much work has concerned itself with the enigma of what exactly PLMs (pretrained language models) learn about different aspects of language, and how they learn it. One stream of this type of research investigates the knowledge that PLMs have about semantic relations. However, many aspects of semantic relations were left unexplored. Only one relation was considered, namely hypernymy. Furthermore, previous work did not measure humans' performance on the same task as that solved by the PLMs. This means that at this point in time, there is only an incomplete view of models' semantic relation knowledge. To address this gap, we introduce a comprehensive evaluation framework covering five relations beyond hypernymy, namely hyponymy, holonymy, meronymy, antonymy, and synonymy. We use six metrics (two newly introduced here) for recently untreated aspects of semantic relation knowledge, namely soundness, completeness, symmetry, asymmetry, prototypicality, and distinguishability and fairly compare humans and models on the same task. Our extensive experiments involve 16 PLMs, eight masked and eight causal language models. Up to now only masked language models had been tested although causal and masked language models treat context differently. Our results reveal a significant knowledge gap between humans and models for almost all semantic relations. Antonymy is the outlier relation where all models perform reasonably well. In general, masked language models perform significantly better than causal language models. Nonetheless, both masked and causal language models are likely to confuse non-antonymy relations with antonymy.
>
---
#### [replaced 097] S^3cMath: Spontaneous Step-level Self-correction Makes Large Language Models Better Mathematical Reasoners
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.01524v3](http://arxiv.org/pdf/2409.01524v3)**

> **作者:** Yuchen Yan; Jin Jiang; Yang Liu; Yixin Cao; Xin Xu; Mengdi Zhang; Xunliang Cai; Jian Shao
>
> **备注:** AAAI 2025: https://ojs.aaai.org/index.php/AAAI/article/view/34749
>
> **摘要:** Self-correction is a novel method that can stimulate the potential reasoning abilities of large language models (LLMs). It involves detecting and correcting errors during the inference process when LLMs solve reasoning problems. However, recent works do not regard self-correction as a spontaneous and intrinsic capability of LLMs. Instead, such correction is achieved through post-hoc generation, external knowledge introduction, multi-model collaboration, and similar techniques. In this paper, we propose a series of mathematical LLMs called S$^3$c-Math, which are able to perform Spontaneous Step-level Self-correction for Mathematical reasoning. This capability helps LLMs to recognize whether their ongoing inference tends to contain errors and simultaneously correct these errors to produce a more reliable response. We proposed a method, which employs a step-level sampling approach to construct step-wise self-correction data for achieving such ability. Additionally, we implement a training strategy that uses above constructed data to equip LLMs with spontaneous step-level self-correction capacities. Our data and methods have been demonstrated to be effective across various foundation LLMs, consistently showing significant progress in evaluations on GSM8K, MATH, and other mathematical benchmarks. To the best of our knowledge, we are the first to introduce the spontaneous step-level self-correction ability of LLMs in mathematical reasoning.
>
---
#### [replaced 098] TigerLLM -- A Family of Bangla Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.10995v3](http://arxiv.org/pdf/2503.10995v3)**

> **作者:** Nishat Raihan; Marcos Zampieri
>
> **摘要:** The development of Large Language Models (LLMs) remains heavily skewed towards English and a few other high-resource languages. This linguistic disparity is particularly evident for Bangla - the 5th most spoken language. A few initiatives attempted to create open-source Bangla LLMs with performance still behind high-resource languages and limited reproducibility. To address this gap, we introduce TigerLLM - a family of Bangla LLMs. Our results demonstrate that these models surpass all open-source alternatives and also outperform larger proprietary models like GPT3.5 across standard benchmarks, establishing TigerLLM as the new baseline for future Bangla language modeling.
>
---
#### [replaced 099] Distillation and Refinement of Reasoning in Small Language Models for Document Re-ranking
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.03947v3](http://arxiv.org/pdf/2504.03947v3)**

> **作者:** Chris Samarinas; Hamed Zamani
>
> **摘要:** We present a novel approach for training small language models for reasoning-intensive document ranking that combines knowledge distillation with reinforcement learning optimization. While existing methods often rely on expensive human annotations or large black-box language models, our methodology leverages web data and a teacher LLM to automatically generate high-quality training examples with relevance explanations. By framing document ranking as a reinforcement learning problem and incentivizing explicit reasoning capabilities, we train a compact 3B parameter language model that achieves state-of-the-art performance on the BRIGHT benchmark. Our model ranks third on the leaderboard while using substantially fewer parameters than other approaches, outperforming models that are over 20 times larger. Through extensive experiments, we demonstrate that generating explanations during inference, rather than directly predicting relevance scores, enables more effective reasoning with smaller language models. The self-supervised nature of our method offers a scalable and interpretable solution for modern information retrieval systems.
>
---
