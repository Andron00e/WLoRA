# Llama 3.1 8B and Google T5 3B Experiments

## GLUE MNLI (model: LLama 3.1 8B)

**Parameters**
- batch size=16
- learning rate=3e-4
- max steps=128
- scheduler=cosine
- warmup steps=10
- r=8

<img src="llama3_figures/mnli_loss.jpg" alt="MNLI Loss" width="500"/> <img src="llama3_figures/mnli_acc.jpg" alt="MNLI Accuracy" width="500"/>

## X-SUM (model: T5 3B) 

**Parameters**
- batch size=8
- learning rate=8e-4
- max steps=128
- scheduler=cosine
- warmup steps=10
- r=8

<img src="llama3_figures/xsum_loss.png" alt="X-SUM Loss" width="500"/> <img src="llama3_figures/xsum_rouge1.png" alt="X-SUM Rouge 1" width="500"/>

## GLUE SST-2 (model: LLama 3.1 8B)

**Parameters**
- batch size=16
- learning rate=8e-5
- max steps=128
- scheduler=cosine
- warmup steps=10
- r=8

<img src="llama3_figures/sst2_loss.png" alt="SST-2 Loss" width="500"/> <img src="llama3_figures/sst2_acc.png" alt="SST-2 Accuracy" width="500"/>

## GLUE QNLI (model: LLama 3.1 8B)

**Parameters**
- batch size=8
- learning=rate 5e-5
- max steps=56
- scheduler=cosine
- warmup steps=10
- r=8

<img src="llama3_figures/qnli_loss.png" alt="QNLI-2 Loss" width="500"/> <img src="llama3_figures/qnli_acc.png" alt="QNLI Accuracy" width="500"/>

## GLUE RTE (model: LLama 3.1 8B)

**Parameters**
- batch size=32
- learning rate=8e-5
- max steps=256
- scheduler=cosine
- warmup steps=10
- r=8

<img src="llama3_figures/rte_loss.png" alt="RTE Loss" width="500"/> <img src="llama3_figures/rte_acc.png" alt="RTE Accuracy" width="500"/>

## GLUE MRPC (model: LLama 3.1 8B)

**Parameters**
- batch size=8 (for lora=32)
- learning rate=8e-5
- max steps=251
- scheduler=cosine
- warmup steps=10
- r=8

<img src="llama3_figures/mrpc_loss.png" alt="MRPC Loss" width="500"/> <img src="llama3_figures/mrpc_f1.png" alt="MRPC F1-Score" width="500"/>
