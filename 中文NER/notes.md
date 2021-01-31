# 中文NER
命名实体识别（NER）是信息提取的基本任务。

目前英文NER的最先进水平是使用LSTM-CRF模型结合集成了字符信息的单词表达式。

中文 NER 的一种直接方法是先进行单词细分（易产生分词错误），然后进行单词序列标记。

从目前来看，对于中文NER来说，基于字的方法比基于词的方法效果好。

## ACL2018《Chinese NER Using Lattice LSTM》
[源代码和数据](https://github.com/jiesutd/LatticeLSTM)

### 大体思想：网格结构的LSTM-CRF模型
+ 不仅编码输入的字序列，而且对所有与词典匹配的词也进行编码
+ 通过网格结构的LSTM将句子中与词典匹配的词表示出来，从而将潜在词信息集成到基于字的LSTM-CRF中
+ 使用网格结构的LSTM来自动控制信息从句子的开头流向末尾
+ 使用门控递归单元来选择句子中最相关的字和词
+ 有向无环图

### 优点
不会出现分词错误。

### 缺点
结构较为复杂，速度慢

难以迁移到其他神经网络模型

### 结果
用多个不同领域的数据集进行测试，结果比仅基于字的LSTM和仅基于词的LSTM的效果都要好，并有较强的鲁棒性。

### 模型
+ 主网络结构为LSTM-CRF
+ BIOES标注模式
  
  `LabelSet = {O, B-PER, I-PER, E-PER, S-PER, B-LOC, I-LOC, E-LOC, S-LOC, B-ORG, I-ORG, E-ORG, S-ORG}`
  + B，即Begin，表示开始
  + I，即Intermediate，表示中间
  + E，即End，表示结尾
  + S，即Single，表示单个字符
  + O，即Other，表示其他，用于标记无关字符
  + PER代表人名， LOC代表位置， ORG代表组织
+ 基于字的模型
  
  将字序列输入LSTM-CRF模型

  字 -> 字符向量 -> 双向LSTM -> CRF -> 序列标注
  + 字符向量 + 双字词组向量
  + 字符向量 + 分词标注向量
  
  分词标注向量采用BMES标注方式进行处理。其中B表示Begin即识别出边界，M表示Middle即识别出实体中间名，E表示End即实体名识别介绍，S表示Single表示独立成词
+ 基于词的模型
  
  词 -> 词向量 -> 双向LSTM

  拼接所得的两组隐层状态即为词的表示
  **集成字的表示**
  词向量 + 字符表示

  字符表示：

  + 用双向LSTM学习词中的每一个字
  + 用LSTM分别从正、反方向学习词中的每一个字再集成在一起
  + 用标准CNN来获得词中的字符序列表示
+ 网格模型
  
  可以看作基于字的模型 + 基于词的单元 + 控制信息流向的门

  + 参数：输入向量，输出隐层向量，单元向量，门向量
  + 基本递归LSTM函数（11）

### 实验
#### 实验配置
+ 数据集分割
+ 词向量化
+ 超参数设置
#### 实验结果
P - 精确率

R - 召回率

F1 - P 和 R 的调和平均，当 F1 较高时，模型的性能越好

## ACL2020《Simplify the Usage of Lexicon in Chinese NER》
 提出了一个更简单而有效的将词信息加入字符表示的方法：将每个字符匹配的所有词加入基于字符的NER模型。
 ### 优点
 + 避免设计复杂的序列模型结构
 + 对于任意神经元NER模型来说，只需要对字符表示层做细微的调整即可引入词典信息
 + 速度快，性能好
 + 可以用在预训练模型中（如BERT）
### 结果
在四个中文NER数据集上进行测试，用单层双向 LSTM 实现序列模型层时，此方法可以在速度和表现两方面较最先进水平均有所提升。
### 方法（SoftLexiocon）
字符 -> 稠密的向量 -> 将SoftLexicon特征加入字符表示 -> 序列建模层 -> CRF层

#### 字符表示层
每个字符用一个稠密的向量表示

+ 字符 + 双字词语

#### 合并字典信息
+ 扩展的Softword（ExSoftword）
  
  用一个五维向量保留所有可能的分词结果

  缺点：
  + 不能得到预训练的单词矢量表示
  + 损失了一些匹配信息，不能分辨哪一个是要储存的正确结果
+ SoftLexicon
  + 对匹配的词进行分类（BMES）
  + 压缩词语集为固定维数的向量
    压缩词语集：
    + 平均池化
    + 加权算法（词语出现频率）
  + 和字符表示相结合
    结合四个词语集的表示为固定维数的向量 -> 加入每个字符的表示（连接）

#### 序列建模层
对字符之间的依赖关系进行建模，如使用双向LSTM，CNN，transformer

#### 标注推断层
CRF

### 实验
大多数设置与 Lattice-LSTM 相同，包括预训练的词向量、测试数据集、比较基准、评估度量等

## ACL2020 《Chinese NER Using Flat-Lattice Transformer》

RNN 和 CNN 难以对长距离依赖建模，但这在 NER 中很有用

### 背景
介绍 Transformer 机制

只讨论 Transformer 编码部分，由 self-attention 和 feedforward network(FFN) 组成

每个子层之后是残差网络和层归一化

FFN 是在位置方面无线性转换的多层感知机

### 模型

+ 将 Lattice 转化为 Flat 结构
  通过字典得到字的网状结构后，将其扁平化
  首先用首尾相同的标记来构建字符序列，然后用其他标记（词）和他们的首、尾来构建 skip-path 
  
+ 短语相关位置编码

  是四种距离的非线性变换

将字符表示输入输出层，然后输入CRF

### 实验

+   实验设置

    用四个数据集：Ontonotes 4.0、MSRA、Resume、Weibo来评估模型

    使用双向LSTM-CRF和TENER作为基准模型（TENER是一个为了NER使用相对位置编码而不使用外部信息的Transformer)

    FLAT模型只用了一层Transformer编码器

+   总体表现

    在四个中文NER模型数据集上表现均优于基准模型和其他基于字典的模型

    但在小数据集上的提升不太明显，可能是因为Tranformer的特性

+   全连接结构的优点

    作者认为self-attention机制相较于Lattice LSTM的优点是：

    +   所有字符可以直接和它相关的词交互
    +   长距离依赖可以被很好地建模

    实现过程中，因为此模型只有一层，因此去掉了字符到自匹配的词和标记之间距离超过10的attention，然而结果会变差。因此利用自匹配词语信息对于中文NER很重要。

+   FLAT的效率

    此模型没有递归模块，可以充分利用GPU，因此运行效率较高

    因为此模型较为简单，所以批并行的效果更好

+   FLAT是如何带来改善的
    +   利用字典信息，预训练的词向量让实体分类更有效
    +   使用新的位置编码来更准确地定位实体

+   与BERT的兼容性

    对比BERT+FLAT和BERT（+CRF），对大数据集来说FLAT+BERT结果好得多，对小数据集来说FLAT+BERT结果只好一点点

### 相关工作

+   Lexicon-based NER
+   Lattice-based Transformer

## COLING2020 《Porous Lattice Transformer Encoder for Chinese NER》

### 特点

+   提出了一个新奇的用于中文NER的网格状transformer编码器，可以进行批处理且可以捕获字符和匹配的单词之间的依赖
+   通过一种多孔机制修改lattice-aware attention分布，增强捕获有用的局部上下文的能力
+   在四个数据集上进行测试，比基准方法快11.4倍，并且表现更好
+   此模型可以容易地集成到BERT中，二者结合表现更好

### 模型

+   Lattice输入层

    +   将语义信息和位置信息集成到字符表示中

+   多孔网格状transformer编码

    +   通过用中心共享结构取代全连接拓扑来增强相邻元素之间的联系，从而学习到稀疏attention系数

        +   Lattice-aware Self-Attention(LASA)

    +   用multi-head attention来从不同表示子空间捕获信息

        +   Porous Multi-Head Attention(PMHA)

            为了维持self-attention捕获长距离依赖的能力且增强捕获短范围依赖的能力，将transformer结构从全连接拓扑结构改为中心共享结构

+   BiGRU-CRF解码器

+   训练

### 实验

#### 实验设置

+   数据集同上
+   比较
    +   Lattice LSTM: 集成词典信息到字符表示，经过门控循环单元，避免分词错误
    +   LR-CNN：通过rethinking机制集成词典信息
    +   BERT-Tagger：利用BERT最后一层输出因为字符级可以丰富上下文表示从而进行句子标注
    +   PLTE[BERT]/LR-CNN[BERT]/Lattice LATM[BERT]：这三种方法用预训练的BERT表示来替代字符表示层，用softmax层来进行句子标记
+   超参数设置

## EMNLP2020 《Entity Enhanced BERT Pre-training for Chinese NER》

### 特点

+   半监督学习
+   使用预训练的语言模型（LM）
+   不适用预训练的实体向量，动态结合具体文档实体
+   使用一种明确的方式将实体编码到transformer结构

### 方法

总体结构：多任务学习的transformer结构，通过扩展标准transformer来集成实体级信息

三个输入组件：

+   无名称的语言模型：类似BERT没有下一个句子输入的任务
+   实体分类：增强预训练
+   NER：序列标注器

方法：

+   新词发现

    采用一种无监督方法来自动发现候选实体，计算连续字符之间的共同信息（MI）和左右交叉熵，并将这三个值相加作为可能实体的校验分数

+   字符实体transformer

    基于以基本BERT为基础的transformer结构构建模型，为了利用抽取出的实体将基本transformer扩展为由一堆多头字符实体self-attention块组成的字符实体transformer

    +   基本transformer
    +   字符实体匹配：用最大实体匹配算法来获得相应的实体标记序列，且用包含字符的最长的实体的索引来标记字符
    +   字符实体self-attention

+   masked语言模型（MLM）任务

    将输入的字符的一部分替换为[MASK]标记

+   实体分类任务

    为了增强字符和相关实体之间的联系，提出了实体分类任务，可以预测当前字符属于哪个具体实体

+   NER任务

    NER输出层是一个线性分类器

+   训练过程

    用预训练的BERT模型进行初始化，其他参数随机初始化

    过程中先用所有原始文本预训练一个语言模型来得到实体增强的模型参数然后用NER任务微调参数

    +   预训练
    +   微调

## ACL2019 《A Neural Multi-digraph Model for Chinese NER with Gazetteers》

地名词典对于中文NER很有用，但是现有方法通常依赖于手工决定选择策略，这通常导致效果不是最优的

### 特点
基于GNN，结合带有地名词典信息的多有向图结构，从而自动学习如何将多个地名词典信息集成到NER中

### 模型
包含多有向图，适合的GGNN嵌入层和一个BiLSTM-CRF层
+ 多有向图：建模文本和NE地名词典信息
+ GGNN结构：特征表示空间
+ BiLSTM-CRF：预测最终输出