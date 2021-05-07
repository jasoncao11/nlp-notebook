# Sequence to Sequence Model with Attention

## 结构：

![s2s4](../images/seq2seq4.png)

- Encoder 部分是一个单层双向RNN，输入为 src=[batch size, src len]，经过 Embedding 转换为 [batch size, src len, emb dim]，经过 encoder RNN，输出为所有时刻的隐状态 outputs=[batch size, src len, enc hid dim * 2] + 最后一个时刻的隐状态 hidden=[1 * 2, batch size, enc hid dim]，hidden 经过线性变换为 [batch size, dec hid dim]。

- Attention 部分接收解码器上一个时刻的隐状态 hidden=[batch size, dec hid dim] + 编码器所有时刻的隐状态 encoder_outputs=[batch size, src len, enc hid dim * 2]，输出 attention=[batch size, src len]。

- Decoder 是一个单层单向RNN，每个时刻接收 inputs=[batch size, 1] + 解码器上一个时刻的隐状态 hidden + 编码器所有时刻的隐状态 encoder_outputs，inputs 经过 embedding 转换为 embedded=[batch size, 1, emb dim]，attention 与 encoder_outputs 做点积运算得到 encoder_outputs 的加权平均 weighted，将 weighted 和 embedded 拼接起来与 hidden 一起传入 decoder RNN，得到 output=[batch size, 1, dec hid dim] 和 hidden=[1, batch size, dec hid dim]，output 和 weighted 以及 embedded三者拼接起来经过线性变换后得到该时刻的输出 [batch size, output dim]，hidden 则继续作为解码器下一个时刻的输入。   

- Seq2seq:
![s2s2](../images/seq2seq2.png)
![s2s3](../images/seq2seq3.png)

## Reference
- https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
- https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
