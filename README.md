# GladiaAgent

## 1. 简介 (Introduction)

**GladiaAgent** (亦称为 Predictive Dialog Agent - PDA) 是一个先进的对话式人工智能框架，其核心设计理念源于认知科学中的**预测编码 (Predictive Coding)** 理论。本项目旨在构建一个不仅仅能流畅对话，更能理解、学习并动态适应对话上下文的智能体。它致力于模拟类人对话的某些方面，例如基于经验的预期、对新奇信息的敏感度以及持续的知识积累。

与传统的基于固定脚本或简单检索的聊天机器人不同，GladiaAgent 的目标是：
*   **深度理解与推理：** 通过整合预测编码机制和大型语言模型 (LLM) 的能力，实现更深层次的语义理解和初步的推理。
*   **动态知识演化：** 智能体在交互过程中动态构建和更新其核心知识库，优先记忆那些具有“信息增益”或“预测意外”的内容。
*   **上下文感知交互：** 能够有效利用对话历史、已形成的知识以及当前对话的细微差别，生成高度相关且连贯的响应。

本项目探索了将认知理论与现代人工智能技术（如深度学习、NLP和LLM）相结合的可能性，以期创造出更智能、更自适应的对话体验。

## 2. 核心特性 (Core Features)

GladiaAgent 具备一系列使其在对话智能体领域独树一帜的特性：

*   🧠 **基于预测编码的智能决策:**
    *   核心的预测编码模块 (`PredictiveCodingAgent`) 评估输入信息（如从对话中提取的三元组）的“新奇性”或“重要性”。
    *   系统根据预测误差（衡量输入与智能体内部模型的匹配程度）来决定是否将新知识整合到其长期记忆中，从而实现更高效的学习。

*   📚 **动态演化的核心知识库:**
    *   在对话过程中，智能体从用户输入中提取关键信息（知识三元组）。
    *   这些信息经过评估后，有选择地存储在其核心知识库 (`knowledge_base_vectors`) 中，形成一个不断增长和优化的知识图谱。
    *   该知识库为智能体提供了事实依据和上下文参考。

*   💬 **多轮对话连贯性与深度记忆:**
    *   通过 `DialogHistoryBuffer` 精心管理对话历史。
    *   结合动态知识库的检索，确保在长时间的多轮对话中保持上下文连贯性和话题焦点。

*   💡 **上下文感知的响应生成:**
    *   `PredictiveDialogAgent (PDA)` 模块在生成响应前，会整合当前用户输入、完整的对话历史、从核心知识库中检索到的相关记忆，以及一个概念上的“思维链”。
    *   这种丰富的上下文被构建成精确的提示，指导强大的 DeepSeek 大型语言模型生成高度相关、信息丰富且符合当前对话场景的回答。

*   🖥️ **双模式交互界面:**
    *   提供直观的**图形用户界面 (GUI)** (`gladia_gui.py`)，方便用户进行交互和观察智能体的行为。
    *   同时支持传统的**命令行界面 (CLI)**，适用于开发者调试或在无图形环境下运行。

*   🔗 **模块化与可扩展设计:**
    *   系统采用模块化架构，主要功能（如预测编码、对话管理、知识集成、LLM接口）被封装在独立的组件中。
    *   这种设计便于未来的功能扩展、模块替换或针对特定应用的定制。

*   🛠️ **集成 DeepSeek LLM:**
    *   利用先进的 DeepSeek 大型语言模型进行复杂自然语言理解（如从非结构化文本中提取结构化的知识三元组）和高质量自然语言生成（对话响应）。

*   💾 **记忆持久化:**
    *   智能体的核心知识库和部分模型状态（如 `seRNN`）可以被保存到磁盘，并在下次启动时加载，实现了学习成果的持久化。
## 3. 技术架构 (Technical Architecture)

GladiaAgent 的架构设计旨在将认知启发的功能模块与强大的深度学习模型相结合，形成一个协同工作的智能对话系统。

### 3.1 整体概念流程 (Conceptual Flow)

GladiaAgent 的核心数据流和模块交互可以概括如下：

```mermaid
graph TD
    A[User Input (GUI/CLI)] --> B(Text Embedding Module);
    B --> UserInputEmbedded[User Input Embedded];

    UserInputEmbedded --> PDA[PredictiveDialogAgent PDA];
    A --> TripleExtractPath{Parallel Path for Knowledge};
    TripleExtractPath --> D[Triple Extraction (LLM)];
    D --> F[Triple Text];
    F --> G(Text Embedding Module);
    G --> H[Triple Embeddings];
    H --> I[seRNN Module];
    I --> J[Enhanced Triple Embeddings];
    J --> K[PredictiveCodingAgent PC];
    K -- Prediction Loss > Threshold --> L[Store in Core Knowledge Base];
    K -- Prediction Loss <= Threshold --> M(Discard/Ignore Triple);

    PDA --> N(Query Embedding Generation);
    N --> O[Query Core Knowledge Base];
    L --> O;

    O -- Retrieved Knowledge --> PDA;
    P[Dialog History] --> PDA;

    PDA -- Constructed Prompt --> Q[DeepSeek LLM];
    Q -- LLM Response Stream --> R(Format & Display Response);
    R --> A;

    subgraph CoreCognitiveLoop [Core Cognitive Loop / Knowledge Storage]
        D; F; G; H; I; J; K; L; M;
    end

    subgraph DialogManagement [Dialog Management & Generation]
        PDA; N; O; P; Q; R;
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style Q fill:#lightgrey,stroke:#333,stroke-width:2px
    style L fill:#ccffcc,stroke:#333,stroke-width:2px
    style K fill:#lightblue,stroke:#333,stroke-width:2px
    style I fill:#lightblue,stroke:#333,stroke-width:2px
图例与说明:

矩形框 (e.g., User Input): 表示数据、状态或外部实体。
圆角矩形框 (e.g., Text Embedding Module): 表示处理模块或组件。
菱形框 (e.g., Input Text): 表示数据分支点或中间数据形态。
箭头 (-->): 表示数据流或控制流方向。
Subgraph (e.g., Core Cognitive Loop): 表示一组功能上相关的模块。
流程解读:

用户输入 (User Input) 进入系统，首先由 文本嵌入模块 (Text Embedding Module) 处理，生成初步的文本表示 (UserInputEmbedded)。
该嵌入表示一方面直接送入 PredictiveDialogAgent (PDA) 用于即时对话处理；另一方面，原始用户输入会并行触发知识提取路径 (Parallel Path for Knowledge)：
原始文本送入 三元组提取模块 (Triple Extraction (LLM))，利用大型语言模型从文本中抽取出结构化的三元组知识 (Triple Text)。
提取出的三元组文本再次经过 文本嵌入模块 (Text Embedding Module) 转换为向量形式 (Triple Embeddings)。
这些三元组嵌入被送入 seRNN 模块 (seRNN Module) 进行深度处理和特征增强，得到 Enhanced Triple Embeddings。
核心认知循环 (Core Cognitive Loop / Knowledge Storage):
增强后的三元组嵌入由 PredictiveCodingAgent (PC) 进行评估。PC Agent 计算其预测损失。
如果预测损失高于预设阈值，表明该三元组具有新奇性或重要价值，其向量表示将被存入核心知识库 (Store in Core Knowledge Base)。
如果预测损失低于或等于阈值，则该三元组被认为信息量不足或已存在相似知识，将被忽略 (Discard/Ignore Triple)。
对话管理与生成 (Dialog Management & Generation):
PredictiveDialogAgent (PDA) 是对话管理的核心。它接收用户输入的嵌入表示，并生成用于查询的嵌入 (Query Embedding Generation)。
利用此查询嵌入，PDA 从核心知识库 (Core Knowledge Base) 中检索相关历史知识。对话历史 (Dialog History) 也作为上下文输入到 PDA。
PDA 整合所有这些信息（当前输入、检索到的知识、对话历史），构建一个全面的上下文提示 (Constructed Prompt)。
该提示被发送给 DeepSeek LLM。
LLM 生成的响应以流式 (LLM Response Stream) 返回，经过格式化后展示给用户 (Format & Display Response)，并反馈到新的用户输入循环中。
3.2 主要模块说明 (Key Modules)
以下是构成 GladiaAgent 核心功能的关键模块：

IntegratedSystem (integrated_system.py)
集成系统

作用: 作为系统的中央协调器，IntegratedSystem 负责初始化所有其他核心模块、适配器和配置。它定义并管理主要的处理流水线，控制数据如何在不同模块间流动。
功能:
加载和保存核心知识库及 seRNN 模型状态。
提供知识库查询接口。
调用 LLM 进行三元组提取。
管理将新知识存入知识库的决策过程（基于 PC 模块的输出，通过 store_triple_with_pc 方法）。
将用户对话请求路由到 PredictiveDialogAgent。
PredictiveDialogAgent (PDA) (PDA.py)

作用: 对话管理的核心引擎，负责处理用户与智能体的直接交互。
功能:
维护对话历史 (DialogHistoryBuffer) 和概念上的“思维链”。
为用户输入生成嵌入（通过调用 IntegratedSystem 的嵌入功能），并利用这些嵌入从核心知识库中检索最相关的上下文信息。
智能构建包含当前输入、对话历史、检索到的知识以及其他上下文信号（如名义上的预测误差）的系统提示。在当前版本的 PDA.py 中，“预测误差” (current_prediction_error) 主要作为提示的一部分，其动态更新机制可能需要进一步开发或明确；它旨在概念上指导LLM当前对话的“意外程度”。
通过异步客户端与配置的 DeepSeek 大型语言模型进行通信，以生成自然、连贯且上下文恰当的对话响应。
PredictiveCodingAgent (PC) (PredictiveCoding.py)

作用: 实现预测编码理论的核心计算逻辑。该模块的主要职责是评估输入数据相对于智能体内部模型的新奇性或意外性。
功能:
包含一个编码器（将输入，如来自 seRNN 的384维嵌入，编码到较低维度）和多个连续时间神经元 (ContinuousTimeNeuron) 及解码器。
通过其 forward 方法处理输入，计算预测损失 (prediction loss) 和重构损失 (reconstruction loss)。
IntegratedSystem 利用此模块计算出的 prediction_loss 来判断一个新提取的三元组（在通过 seRNN 处理后）是否足够“令人惊讶”或“信息丰富”，从而决定是否将其存储到核心知识库中。
seRNN (Spatially-Embedded Recurrent Neural Network) (seRNN.py)


作用: 一个增强型的循环神经网络，它在传统 RNN 的基础上加入了空间嵌入的概念和预测编码层，旨在更好地捕捉序列数据中的复杂依赖关系和结构信息。
功能:
处理输入的时间序列数据（例如，由三元组文本转换而来的嵌入向量序列）。
其内部包含 PredictiveCodingLayer 用于计算预测误差，SelectiveGatingMechanism 以及 SpatialEmbeddingLayer 来处理空间位置信息。
输出经过处理的序列表示，这些表示随后可能被 PredictiveCodingAgent 用作输入，以进行更高级的“新奇性”评估。
文本嵌入 (Text Embedding)  文本嵌入 （Text Embedding）

技术: 主要使用 sentence-transformers 库中的预训练模型（例如 paraphrase-MiniLM-L6-v2）。
作用: 将所有文本数据（包括用户的原始输入、从文本中提取的三元组的主谓宾部分等）转换为固定维度（本项目中为384维）的密集向量表示。这些嵌入是后续所有深度学习模块处理的基础。
大语言模型 (LLM - DeepSeek Integration)

技术: 通过 openai Python 库与 DeepSeek API 进行交互。
作用:
知识提取: 在 IntegratedSystem 中，LLM 被用于从用户提供的非结构化文本中精确提取结构化的知识三元组（主体-关系-客体）。
对话生成: 在 PredictiveDialogAgent 中，LLM 根据精心构建的上下文提示生成最终的用户回复，确保对话的自然流畅和信息丰富度。

3.3 辅助与实验性模块 （Supporting & Experimental Modules）
这些模块虽然存在于代码库中，但在当前核心运行流程中的作用可能不是主要的。

TN (Tensor Network 张量网络)

Adapters (适配器)

## 4.启动方式

1.在github上克隆项目地址
  git clone [项目地址]

2.
pip install -r requirements.txt
run main.py
