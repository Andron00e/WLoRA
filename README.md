# Llama 3.1 8B Experiments

## GLUE MNLI

**Parameters**
- batch size 16
- learning rate 3e-4
- max steps 128
- scheduler cosine
- warmup steps 10
- r 8

<img src="llama3_figures/mnli_loss.jpg" alt="MNLI Loss" width="500"/> <img src="llama3_figures/mnli_acc.jpg" alt="MNLI Accuracy" width="500"/>

## GLUE SST-2

**Parameters**
- batch size 16
- learning rate 8e-5
- max steps 128
- scheduler cosine
- warmup steps 10

<img src="llama3_figures/sst2_loss.png" alt="SST-2 Loss" width="500"/> <img src="llama3_figures/sst2_acc.png" alt="SST-2 Accuracy" width="500"/>
