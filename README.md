# Improving QA Generalization by Concurrent Modeling of Multiple Biases

Please use the following citation:

```
@misc{wu2020improving,
      title={Improving QA Generalization by Concurrent Modeling of Multiple Biases}, 
      author={Mingzhu Wu and Nafise Sadat Moosavi and Andreas Rücklé and Iryna Gurevych},
      year={2020},
      eprint={2010.03338},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url = {https://arxiv.org/abs/2010.03338}
}
```

> **Abstract:** Existing NLP datasets contain various biases that models can easily exploit to achieve high performances on the corresponding evaluation sets. However, focusing on dataset-specific biases limits their ability to learn more generalizable knowledge about the task from more general data patterns. In this paper, we investigate the impact of debiasing methods for improving generalization and propose a general framework for improving the performance on both in-domain and unseen out-of-domain datasets by concurrent modeling of multiple biases in the training data. 
Our framework weights each example based on the biases it contains and the strength of those biases in the training data. It then uses these weights in the training objective so that the model relies less on examples with high bias weights.
We extensively evaluate our framework on extractive question answering with training data from various domains with multiple biases of different strengths. We perform the evaluations in two different settings, in which the model is trained on a single domain or multiple domains simultaneously, and show its effectiveness in both settings compared to state-of-the-art debiasing methods. 

Contact person: Mingzhu Wu, mwu@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## Install requirements 
**Setup a Python virtual environment**
```
cd baseline
virtualenv venv-my --python=python3.6 or python3 -m venv venv-my
source venv-my/bin/activate
```

**Install the requirements:**
```
pip install -r requirements.txt
```

## Running the experiments
**Train baseline/teacher models**
* Example - single domain teacher model on NaturalQuestions:
```
python run_qa.py --output_dir Models/NaturalQuestionsShort --config_file MRQA_BERTbase.json -o "{'iterator': {'type': 'distill_iterator'}, 'validation_iterator': {'type': 'distill_iterator'}, 'dataset_reader': {'sample_size': -1}, 'validation_dataset_reader': {'sample_size': -1}, 'train_data_path': 'https://mrqa.s3.us-east-2.amazonaws.com/data/train/NaturalQuestionsShort.jsonl.gz', 'validation_data_path': 'https://mrqa.s3.us-east-2.amazonaws.com/data/dev/NaturalQuestionsShort.jsonl.gz', 'trainer': {'cuda_device': [CUDE DEVICEID or -1 for CPU], 'num_epochs': 2, 'optimizer': {'type': 'bert_adam', 'lr': 3e-05, 'warmup': 0.1,'t_total': 40000}}}" --do_train
```

* Example - multi-domain baseline model on five datasets:
```
python run_qa.py --output_dir Models/mt-base --config_file MRQA_BERTbase.json -o "{'iterator': {'type': 'distill_iterator'}, 'validation_iterator': {'type': 'distill_iterator'}, 'train_data_path': 'https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/HotpotQA.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/TriviaQA-web.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz','validation_data_path': 'https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz', 'trainer': {'cuda_device': [CUDE DEVICEID or -1 for CPU], 'num_epochs': 2, 'optimizer': {'type': 'bert_adam', 'lr': 3e-05, 'warmup': 0.1,'t_total': 150000}}}" --do_train
```

**Train debiased models**

* Example - single domain Mb_WL method on NaturalQuestions:
```
python run_qa.py --output_dir Models/debiased-WL-NaturalQuestionsShort --config_file MRQA_BERTbase.json -o "{'iterator': {'type': 'distill_iterator'}, 'validation_iterator': {'type': 'distill_iterator'}, 'dataset_reader': {'sample_size': -1}, 'validation_dataset_reader': {'sample_size': -1}, 'train_data_path': 'https://mrqa.s3.us-east-2.amazonaws.com/data/train/NaturalQuestionsShort.jsonl.gz', 'validation_data_path': 'https://mrqa.s3.us-east-2.amazonaws.com/data/dev/NaturalQuestionsShort.jsonl.gz', 'teacher_path': 'Models/NaturalQuestions', 'trainer': {'cuda_device': [CUDE DEVICEID or -1 for CPU], 'num_epochs': 2, 'optimizer': {'type': 'bert_adam', 'lr': 3e-05, 'warmup': 0.1,'t_total': 40000}}}" --do_train --bias_type combine
```

* Example - single domain Mb_CR method on NaturalQuestions:
```
python run_qa.py --output_dir Models/debiased-CR-NaturalQuestionsShort --config_file MRQA_BERTbase.json -o "{'iterator': {'type': 'distill_iterator'}, 'validation_iterator': {'type': 'distill_iterator'}, 'dataset_reader': {'sample_size': -1}, 'validation_dataset_reader': {'sample_size': -1}, 'train_data_path': 'https://mrqa.s3.us-east-2.amazonaws.com/data/train/NaturalQuestionsShort.jsonl.gz', 'validation_data_path': 'https://mrqa.s3.us-east-2.amazonaws.com/data/dev/NaturalQuestionsShort.jsonl.gz', 'teacher_path': 'Models/NaturalQuestions', 'trainer': {'cuda_device': [CUDE DEVICEID or -1 for CPU], 'num_epochs': 2, 'optimizer': {'type': 'bert_adam', 'lr': 3e-05, 'warmup': 0.1,'t_total': 40000}}}" --do_train --bias_type combine --method CR
```

* Example - multi-domain  Mb_CR method on five dataset:
```
python run_qa.py --output_dir Models/mt-base --config_file MRQA_BERTbase.json -o "{'iterator': {'type': 'distill_iterator'}, 'validation_iterator': {'type': 'distill_iterator'}, 'train_data_path': 'https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/HotpotQA.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/TriviaQA-web.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz','validation_data_path': 'https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz,https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz', 'teacher_path': 'Models/SQuAD,Models/HotpotQA,Models/TriviaQA-web,Models/NewsQA,Models/NaturalQuestionsShort', 'trainer': {'cuda_device': [CUDE DEVICEID or -1 for CPU], 'num_epochs': 2, 'optimizer': {'type': 'bert_adam', 'lr': 3e-05, 'warmup': 0.1,'t_total': 150000}}}" --do_train --bias_type combine --method CR
```
Replace bias type *combine* with *wh-word, emptyqst, lexical or bidaf* for debiasing on each of the single bias.


**Making predictions** 
* Example for predicting NaturalQuestions dev using a debiased model:

```
python predict.py Models/debiased-WL-NaturalQuestionsShort https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz pred-NaturalQuestionsShort.json
```

**Evaluate**
```
python ../eval_qa.py https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz pred-NaturalQuestionsShort.json
```


## Acknowledgement
The code in this repository is build on the repository of [MRQA-Shared-Task-2019](https://github.com/mrqa/MRQA-Shared-Task-2019) with 
minor changes and additional scripts for knowledge distillation and debiasing. 
Please refer to the original page for more details.

