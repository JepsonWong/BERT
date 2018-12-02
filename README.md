
## PyTorch ipython版本（针对自动翻译）

### make\_model()函数

参数定义：

src\_vocab：源文单词数目；tag\_vocab：译文单词数目；d\_model：所有模型支层的输出维度；d\_ff：前馈网络隐层的神经元个数；h：MultiHead层并行的attention个数；dropout：Dropout层参数（前馈神经网络输出层和MultiHeadAttention层接一个Dropout层、Embedding层后也接Dropout层）。

基本结构定义：

包括MultiHeadAttention层、PositionwiseFeedFoward层、PositionalEncoding层、两个Embedding层（一个针对源文、一个针对译文）的定义。

网络结构定义，即EncoderDecoder结构：

* Encoder定义了多层EncoderLayer，论文中为6层，整体最后会接一个norm层。EncoderLayer：MultiHeadAttention层 + SublayerConnection层 + PositionwiseFeedFoward层 + SublayerConnection层（resnet + norm）。
* Decoder定义了多层DecoderLayer，论文中为6层，整体最后会接一个norm层。DecoderLayer：MultiHeadAttention层 + SublayerConnection层 + MultiHeadAttention层 + SublayerConnection层 + PositionwiseFeedFoward层 + SublayerConnection层。
* InputEmbedding层 + PositionalEncoding层，注意：Embedding层进行了改进，除以根号下d\_model。
* OutputEmbedding层 + PositionalEncoding层，注意：Embedding层进行了改进，除以根号下d\_model。
* Generator层：Linear层 + Softmax层，输出翻译为某个词的概率。

### SimpleLossCompute类

loss和优化器定义：

* generator：网络结构中的Generator层计算预测结果。
* criterion：结合了Label Smoothing方法，用Label Smoothing类中的接口计算loss。
* opt：NpamOpt类的对象，根据梯度来更新权重。优化器，参数model\_size、factor和warmup和step配合控制学习率的变化。optimizer表示传入的优化器，一般使用Adam优化器。

### run\_epoch()函数

参数定义：

data\_iter：定义数据生成方式；model：定义的模型对象；loss\_compute：定义的loss和优化器对象。

基本功能：

读取batch数据，通过model forward得出结果，通过loss\_compute计算出loss并且进行优化；然后输出loss。

## PyTorch版本

### 要点

### 一些mask

Encoder阶段：non\_pad\_mask表示对输入句子进行mask，padding的数据不参与计算。self\_atten\_mask表示在attention阶段进行mask，对padding的数据不参与attention计算。

Decoder阶段：non\_pad\_mask表示对输入句子进行mask，padding的数据不参与计算。self\_atten\_mask表示在第一个attention阶段进行mask，对padding的数据以及在此位置之后的单词（避免将来的词影响当前的预测结果）不参与attention计算。dec\_enc\_atten\_mask表示在第二个attention阶段进行mask，对超出源句子长度的encoder的结果不参与attention。。

### Beam Search

这是其中比较晦涩难懂的一部分。

Beam.py：

```
self.next_ys = [torch.full((size,), Constants.PAD, dtype=torch.long, device=device)]
self.next_ys[0][0] = Constants.BOS
因为刚开始只begin都是Constants.BOS，所以只把第一个设为Constants.BOS，其他的设为padding表示不会采用其他的预测结果。

advance函数也会判断len(self.prev_ks)来采取不同措施。
```

数据结构如下

* self.size表示search的范围，保存前size个得分高的序列。
* self.\_done表示该序列是否解码完成。
* self.scores保存当前情况下前size个高的得分。
* self.all\_scores保存所有的得分。用list结构保存。
* self.prev\_ks保存当前的序列来自前一序列的哪个seach得出的结果。
* self.next\_ys保存每一时刻前k个得分高的单词id。

函数如下

* advance函数：传入self.size * word的二维数据。然后跟当前的得分相加。通过view展开为一维，选出top(size)个得分高的预测结果。将结果保存至all\_scores中，更新self.scores。得出当前top(size)结果所处的前面哪一个search，存入self.prev\_ks，将当前top(size)结果所预测的size个单词id，存入self.next\_ys中。判断当前得分最高的序列的预测结果是否是EOS，如果是说明该序列预测完成，更新self.\_done。
* get\_current\_state函数：得出当前最优的size个序列。通过调用get\_tentative\_hypothesis函数完成，这个函数每次调用一次get\_hypothesis函数获取第k个得分高的序列。

Translator.py：

数据结构如下

* inst\_dec\_beams：为每一个源序列创建一个beam结构用来beam search。
* active\_inst\_idx\_list：目前还没有预测完成的源序列id组成的list。
* inst\_idx\_to\_position\_map：目前还没有预测完成的源序列的id和它在list中的位置组成的dict。

**一些关键思考**

思考一下为什么要有inst\_dec\_beams，显而易见，每一个源序列都要beam search，但有些源序列已经预测完了就不需要继续predict乃至beam search了。所以需要active\_inst\_idx\_list来记录当前未完成的源序列id（这个是针对batch size大小数据的id），需要inst\_idx\_to\_position\_map（dict结构，元素为batch size大小数据的id: 当前剩余预测序列在此时src\_seq、src\_encoder中的id）中的value来记录当前该删除哪些src\_seq和src\_encoder。

函数如下

* translate\_batch表示对一个batch的序列进行翻译。
* beam\_decode\_step：为源序列预测每一时刻的结果。传入数据为inst\_dec\_beams（**预测过程中不会变化**）、len\_dec\_seq（当前预测的时刻，每次加1）、src\_seq, src\_enc, inst\_idx\_to\_position\_map（这三个参数每一时刻都会变，通过collate\_active\_info进行更新）。首先构造dec\_seq和dec\_pos；然后利用decoder预测，获取最后时刻的预测的结果返回；通过collect\_active\_inst\_idx\_list更新active\_inst\_idx\_list，这个阶段调用beam类的advance函数进行beam search结果更新。
* collate\_active\_info：每一时刻之后更新src\_seq, src\_enc, inst\_idx\_to\_position\_map，生成新的。

### 前期准备

数据下载、利用Moses的tokenizer.perl脚本处理数据，分词过程（**将标点符号也分开**）、利用preprocess.py构造.pt后缀的数据（只处理了训练集和验证集）（处理后的数据包括训练集和验证集的数据转化为数字集合，生成源语言和翻译语言的词典）。

数据处理过程如下：

1.preprocess.py将训练集和验证集转化为.pt后缀的数据，这个数据结构为：

```
    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}
```

2.train.py中通过torch.load函数读取刚刚保存的数据，然后利用读取的数据分别对train和valid构造TranslationDataset对象，然后利用TranslationDataset对象构造DataLoader对象。然后我们就可以用DataLoader对象进行迭代。

构造DataLoader对象加入了collate\_fn参数，可以利用这个函数对读出的数据进行自定义的一些操作。

### train.py 训练过程

构造Transformer类，ScheduledOptim优化器类，然后调用train函数进行训练。

cal\_performance函数：调用cal\_loss函数计算logloss，计算n\_correct来统计预测正确的单词数量。

### translate.py 验证过程

构造Translator类，Translator内部构造了Transformer类，调用translate\_batch函数来处理每batch的数据。

处理一个batch的数据在encoder阶段完成后，decoder阶段运行一次仅仅预测此时刻的翻译结果（如果目标句子最大长度为100，就要运行100次encoder；每次encoder传入当前得到的所有目标词的预测结果和此时的mask，mask的目的是把该预测词之后的词屏蔽掉）。

预测阶段采用**beam search**的方法，同时计算loss也采用了**label smoothing**的方法。
