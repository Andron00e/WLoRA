# Llama 3.1 8B Experiments

## GLUE MNLI

**Parameters**
- batch size=16
- learning rate=3e-4
- max steps=128
- scheduler=cosine
- warmup steps=10
- r=8

<img src="llama3_figures/mnli_loss.jpg" alt="MNLI Loss" width="500"/> <img src="llama3_figures/mnli_acc.jpg" alt="MNLI Accuracy" width="500"/>

## GLUE SST-2

**Parameters**
- batch size=16
- learning rate=8e-5
- max steps=128
- scheduler=cosine
- warmup steps=10
- r=8

<img src="llama3_figures/sst2_loss.png" alt="SST-2 Loss" width="500"/> <img src="llama3_figures/sst2_acc.png" alt="SST-2 Accuracy" width="500"/>

## GLUE QNLI

**Parameters**
- batch size=8
- learning=rate 5e-5
- max steps=56
- scheduler=cosine
- warmup steps=10
- r=8

<img src="llama3_figures/qnli_loss.png" alt="QNLI-2 Loss" width="500"/> <img src="llama3_figures/qnli_acc.png" alt="QNLI Accuracy" width="500"/>

## GLUE RTE

**Parameters**
- batch size=32
- learning rate=8e-5
- max steps=256
- scheduler=cosine
- warmup steps=10
- r=8

<img src="llama3_figures/rte_loss.png" alt="RTE Loss" width="500"/> <img src="llama3_figures/rte_acc.png" alt="RTE Accuracy" width="500"/>

## GLUE MRPC

**Parameters**
- batch size=8 (for lora=32)
- learning rate=8e-5
- max steps=251
- scheduler=cosine
- warmup steps=10
- r=8

<img src="llama3_figures/mrpc_loss.png" alt="MRPC Loss" width="500"/> <img src="llama3_figures/mrpc_f1.png" alt="MRPC F1-Score" width="500"/>
