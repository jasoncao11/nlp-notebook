# BILSTM_CRF_NER

A PyTorch implementation of batch support bidirectional LSTM-CRF.

## Usage

Training/Test data should be formatted as shown in data directory.

### To train:

```
python train.py

output:
Epoch[0] - Loss:60.69867333498868
Epoch[1] - Loss:21.227553984131475
Epoch[2] - Loss:17.732702394928594
Epoch[3] - Loss:15.213178186705619
Epoch[4] - Loss:13.035220449621027
Epoch[5] - Loss:11.137125511362095
Epoch[6] - Loss:9.53712296245074
Epoch[7] - Loss:8.237270446738812
Epoch[8] - Loss:7.199581524338385
Epoch[9] - Loss:6.367835540964146
Epoch[10] - Loss:5.689121142782346
Epoch[11] - Loss:5.1327749864019525
Epoch[12] - Loss:4.665782478120592
Epoch[13] - Loss:4.268250306447347
Epoch[14] - Loss:3.925915930006239
Epoch[15] - Loss:3.6282048442146997
Epoch[16] - Loss:3.37015049385302
Epoch[17] - Loss:3.135349421790152
Epoch[18] - Loss:2.9285232069516423
Epoch[19] - Loss:2.7387238813169077
Epoch[20] - Loss:2.569127255015903
Epoch[21] - Loss:2.414230562219716
Epoch[22] - Loss:2.267430623372396
Epoch[23] - Loss:2.140453951527374
Epoch[24] - Loss:2.03003915632614
Epoch[25] - Loss:1.916411446802544
Epoch[26] - Loss:1.807385913651399
Epoch[27] - Loss:1.7090163899190498
Epoch[28] - Loss:1.6186452279187211
Epoch[29] - Loss:1.5372241922099181
Epoch[30] - Loss:1.457019238760977
Epoch[31] - Loss:1.37889370111504
Epoch[32] - Loss:1.3118280876766553
Epoch[33] - Loss:1.245032553720956
Epoch[34] - Loss:1.1838741880474668
Epoch[35] - Loss:1.1235163392442646
Epoch[36] - Loss:1.0764056715098294
Epoch[37] - Loss:1.018409713350161
Epoch[38] - Loss:0.9707665172490206
Epoch[39] - Loss:0.9277125916095695
Epoch[40] - Loss:0.9040766641347096
Epoch[41] - Loss:0.8419274163968635
Epoch[42] - Loss:0.8000989911532161
Epoch[43] - Loss:0.7613855573264036
Epoch[44] - Loss:0.723888613057859
Epoch[45] - Loss:0.6883248433922277
Epoch[46] - Loss:0.6520613675767725
Epoch[47] - Loss:0.6224016646544138
Epoch[48] - Loss:0.5939752086244449
Epoch[49] - Loss:0.5859614264483404
Epoch[50] - Loss:0.5854936961573783
Epoch[51] - Loss:0.5437964011322368
Epoch[52] - Loss:0.5057372107650294
Epoch[53] - Loss:0.4745464436333589
Epoch[54] - Loss:0.4507301395589655
Epoch[55] - Loss:0.42770399318801033
Epoch[56] - Loss:0.40824193394545355
Epoch[57] - Loss:0.3878540631496545
Epoch[58] - Loss:0.37071554739065843
Epoch[59] - Loss:0.3587678922246201
Epoch[60] - Loss:0.34116181520500566
Epoch[61] - Loss:0.3352175425700467
Epoch[62] - Loss:0.33029678945589547
Epoch[63] - Loss:0.34969438717822837
Epoch[64] - Loss:0.3591618009588935
Epoch[65] - Loss:0.3054806779731404
Epoch[66] - Loss:0.2747708572582765
Epoch[67] - Loss:0.2560846045462772
Epoch[68] - Loss:0.23897433341151536
Epoch[69] - Loss:0.22707724526072992
Epoch[70] - Loss:0.2139989531250915
Epoch[71] - Loss:0.20445856361678152
Epoch[72] - Loss:0.19547549988886323
Epoch[73] - Loss:0.194664974029016
Epoch[74] - Loss:0.1823046929908521
Epoch[75] - Loss:0.17688595383155226
Epoch[76] - Loss:0.18661629709631505
Epoch[77] - Loss:0.19261965637255196
Epoch[78] - Loss:0.22982747536717038
Epoch[79] - Loss:0.21963398203705298
Epoch[80] - Loss:0.18280247508576422
Epoch[81] - Loss:0.15916750105944547
Epoch[82] - Loss:0.1504186765864642
Epoch[83] - Loss:0.15260322226418388
Epoch[84] - Loss:0.16250605203888632
Epoch[85] - Loss:0.14170616029789954
Epoch[86] - Loss:0.12656304383217687
Epoch[87] - Loss:0.1235978768178911
Epoch[88] - Loss:0.12098929143012171
Epoch[89] - Loss:0.11045296422459862
Epoch[90] - Loss:0.09936632930931419
Epoch[91] - Loss:0.09173031234078938
Epoch[92] - Loss:0.09205582173484744
Epoch[93] - Loss:0.08524826876442841
Epoch[94] - Loss:0.08130430050119004
Epoch[95] - Loss:0.08147973232347556
Epoch[96] - Loss:0.08311017719332618
Epoch[97] - Loss:0.08646738645855827
Epoch[98] - Loss:0.09522969230557933
Epoch[99] - Loss:0.09701536212003592
Training costs:1282.8690948486328 seconds
```

### To evaluate:

```
python eval.py

output:

......

Sent: 这是第一位访华的希腊大船主，两周时间，他饱览中国的山水名胜，接触了各界人士，参观港口、码头和造船厂，探索与中国合作的途径。
NER: [['华', 'LOC'], ['希腊', 'LOC'], ['中国', 'LOC'], ['中国', 'LOC']]
Predicted NER: [['华', 'LOC'], ['希腊', 'LOC'], ['中国', 'LOC'], ['中国', 'LOC']]
---------------

Sent: 在希腊，人们常称他为“小奥纳西斯”。
NER: [['希腊', 'LOC'], ['奥纳西斯', 'PER']]
Predicted NER: [['希腊', 'LOC'], ['奥纳西斯', 'PER']]
---------------

Sent: 其实，他与那位娶了美国总统肯尼迪遗孀杰奎琳·肯尼迪的希腊船王奥纳西斯毫无血缘关系。
NER: [['美国', 'LOC'], ['肯尼迪', 'PER'], ['杰奎琳·肯尼迪', 'PER'], ['希腊', 'LOC'], ['奥纳西斯', 'PER']]
Predicted NER: [['美国', 'LOC'], ['肯尼迪', 'PER'], ['杰奎琳·肯尼迪', 'PER'], ['希腊', 'LOC'], ['奥纳西斯', 'PER']]
---------------

Sent: 这么称呼他，是因为他们有相似的经历：胆识过人，白手起家，历经一二十年创建了庞大的船队，成为政界、商界的风云人物。
NER: []
Predicted NER: []

......

gold_num = 6180
predict_num = 5835
correct_num = 4710
precision = 0.8071979434447301
recall = 0.7621359223300971
f1-score = 0.7840199750312109
```
### Others

1. settings.py is where the parameters are defined.

2. load_data.py implements a data generator for producing the tensor of batch size.

3. pytorch_tutorial_vec.py is the vectorized version of the [pytorch BiLSTM-CRF tutorial](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html), modified based on [https://zhuanlan.zhihu.com/p/97676647](https://zhuanlan.zhihu.com/p/97676647).
