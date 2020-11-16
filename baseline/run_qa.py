import argparse
from typing import Any, Union, Dict, Iterable, List, Optional, Tuple
import mrqa_allennlp
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.file_utils import cached_path
from allennlp.common import Params
from allennlp.models.model import Model
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.optimizers import Optimizer
from allennlp.training.util import create_serialization_dir, evaluate
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.optimization import WarmupLinearSchedule
import copy
import numpy as np
import gzip
import json
import logging
import math
import os
import random
import torch
import time
import datetime
from allennlp.common.tqdm import Tqdm
from typing import Iterable
import torch.nn.functional as F
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


def get_pred_per_bias(bias, dataset_name):
    # bias weight of the bias model on the training data is pre-calculated and saved in the json file.
    with open("../bias-weights/"+bias+"/"+dataset_name+".json", "r") as fb:
        return json.load(fb)
    
def get_bias_weight(bias, batch, dataset_name):
    informative = informative_dict[dataset_name]
    pred_dict = all_pred_dict[dataset_name]

    bias_weights = []
    for instance in batch["metadata"]:
        qid = instance["question_id"]
        
        if bias == "combine":
            bias_weights.append([min(pred[qid][1]*informative[bs][0] for bs, pred in pred_dict.items()),\
                                 min(pred[qid][2]*informative[bs][0] for bs, pred in pred_dict.items())])
        else:
            bias_weights.append([pred_dict[bias][qid][1], pred_dict[bias][qid][2]])

    return torch.FloatTensor(bias_weights)


def probability_scaling(stu_logits, bias, tea_logits):
    tea_probs = F.softmax(tea_logits, dim=-1)

    weights = (1 - bias).unsqueeze(1).expand_as(tea_probs)
    weights = weights.to(tea_probs.device)
    exp_tea_probs = tea_probs ** weights
    norm_teacher_probs = exp_tea_probs / exp_tea_probs.sum(1).unsqueeze(1).expand_as(exp_tea_probs)

    loss_fct = nn.KLDivLoss(reduction='batchmean')
    loss = loss_fct(F.log_softmax(stu_logits / args.temperature, dim=-1), norm_teacher_probs / args.temperature)

    return loss


def loss_reweighting(stu_logits, bias, tea_logits):
    weights = (1 - bias).unsqueeze(1).expand_as(tea_logits)
    weights = weights.to(tea_logits.device)

    loss_fct = nn.KLDivLoss(reduction='none')
    loss = loss_fct(F.log_softmax(stu_logits / args.temperature, dim=-1), F.softmax(tea_logits / args.temperature, dim=-1))

    # weighted loss over all examples in one batch, then divide by batch size.
    return (weights * loss).sum() / weights.size()[0]


def train(params, datasets, student, val_iterator, cuda_device=-1, teachers=None):
    """ Train the model """
    tb_writer = SummaryWriter()
    t_total = params.params.get('trainer').get('optimizer').get("t_total")

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # BertAdam already has a scheduler
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, t_total=t_total)
    # parameters = [[n, p] for n, p in student.named_parameters() if p.requires_grad]
    # optimizer = Optimizer.from_params(parameters, params.get("trainer").pop("optimizer"))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(list(datasets["train"])))
    logger.info("  Num Epochs = %d", num_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", params.get("iterator").get("batch_size"))
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    total_training_loss = 0.0
    logging_loss = 0.0
    best_f1 = 0.0
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(student.vocab)

    for epoch in range(num_epochs):
        tr_loss = 0.0
        student.zero_grad()
        batches_this_epoch = 0
        epoch_start_time = time.time()
        logger.info("Training")
        logger.info("Epoch %d/%d", epoch, num_epochs - 1)

        # Get tqdm for the training batches
        train_generator = iterator(datasets["train"], num_epochs=1, shuffle=True)
        num_training_batches = math.ceil(iterator.get_num_batches(datasets["train"]))
        train_generator_tqdm = Tqdm.tqdm(train_generator, total=num_training_batches)

        for batch_group in train_generator_tqdm:
            batches_this_epoch += 1
            set_name = batch_group["metadata"][0]["dataset"]
            if teachers != {}:
                assert set_name in teachers.keys()
            teacher = teachers.get(set_name, None)

            student.train()
            if teacher is not None:
                teacher.eval()

            batch_group = nn_util.move_to_device(batch_group, cuda_device)
            output_dict = student(**batch_group)
            start_logits_stu = output_dict["span_start_logits"]
            end_logits_stu = output_dict["span_end_logits"]
            loss = output_dict["loss"]  # pure student loss, gold loss
            # Distillation loss
            if teacher is not None:
                with torch.no_grad():
                    teacher_output_dict = teacher(**batch_group)
                start_logits_tea = teacher_output_dict["span_start_logits"]
                end_logits_tea = teacher_output_dict["span_end_logits"]
                assert start_logits_stu.size() == start_logits_tea.size()
                assert end_logits_stu.size() == end_logits_tea.size()

                bias_weights = get_bias_weight(args.bias_type, batch_group, set_name)

                # confidence  regularization method
                if args.method == "CR":
                    loss_start = probability_scaling(start_logits_stu, bias_weights[:, 0], start_logits_tea)
                    loss_end = probability_scaling(end_logits_stu, bias_weights[:, 1], end_logits_tea)
                else:
                    # the WL method
                    loss_start = loss_reweighting(start_logits_stu, bias_weights[:, 0], start_logits_tea)
                    loss_end = loss_reweighting(end_logits_stu, bias_weights[:, 1], end_logits_tea)
                loss = loss_start + loss_end

            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            student.zero_grad()
            global_step += 1
            metrics = training_util.get_metrics(student, tr_loss, batches_this_epoch)
            description = training_util.description_from_metrics(metrics) + "\n"
            train_generator_tqdm.set_description(description, refresh=False)
        training_util.get_metrics(student, tr_loss, batches_this_epoch, reset=True)

        # evaluate on the validation dataset
        with torch.no_grad():
            logging.info("validation student")
            metrics = evaluate(student, datasets["validation"], val_iterator, cuda_device, batch_weight_key="")
            current_f1 = metrics["f1"]
            for key, value in metrics.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                tb_writer.add_scalar("lr", optimizer.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
            logging_loss = tr_loss
            logger.info("{'Epoch %d/%d, student exact_match': %s, 'f1': %s}", epoch, num_epochs-1, metrics["EM"], metrics["f1"])

        # Save model checkpoint
        model_path = os.path.join(args.output_dir, "model_state_epoch_{}.th".format(epoch))
        best_path = os.path.join(args.output_dir, "best.th")
        torch.save(student.state_dict(), model_path)
        if current_f1 > best_f1:
            torch.save(student.state_dict(), best_path)
            best_f1 = current_f1
        logger.info("Saving model checkpoint to %s", args.output_dir)

        epoch_elapsed_time = time.time() - epoch_start_time
        logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))
        total_training_loss += tr_loss
        student.get_metrics(reset=True)

    tb_writer.close()
    return global_step, total_training_loss / global_step


def load_teacher_model(teacher_path=None, device=-1):
    models = {}
    if teacher_path is not None:
        for tea_path in teacher_path.split(","):
            # teacher path is something like "Models/HotpotQA,Models/SQuAD"
            tea_name = tea_path.split("/")[-1]
            config = Params.from_file(os.path.join(tea_path, CONFIG_NAME))
            vocab_tea = Vocabulary.from_files(os.path.join(tea_path, "vocabulary"))
            model = Model.from_params(vocab=vocab_tea, params=config.get("model"))

            tea_model = copy.deepcopy(model)
            model_state = torch.load(os.path.join(tea_path, "best.th"), map_location=nn_util.device_mapping(cuda_device))
            tea_model.load_state_dict(model_state)
            logger.info("Load teacher model from %s", tea_path)

            # freeze the parameters of teacher model
            for p in tea_model.parameters():
                p.requires_grad = False

            if device >= 0:
                tea_model.to(device=device)
            models[tea_name] = tea_model

    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--config_file", default="", type=str, required=True,
                        help="The config file for the experiment.")
    parser.add_argument('-o', '--overrides', type=str, default="",
                        help='a JSON structure used to override the experiment configuration')
    parser.add_argument('--model_archive', default=None, type=str,
                        help='path to the saved model archive from training on the original data')

    # Distillation parameters
    parser.add_argument('--alpha_ce', default=0.5, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument('--alpha_squad', default=0.5, type=float,
                        help="True SQuAD loss linear weight. Only for distillation.")
    parser.add_argument('--temperature', default=2.0, type=float,
                        help="Distillation temperature. Only for distillation.")
    ## Other parameters
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--overwrite_output_dir', type=bool, default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--recover', type=bool, default=False,
                        help="whether to recover from exsiting model or not")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")

    # debiasing parameters
    parser.add_argument('--bias_type', default=None, type=str,
                        help='The type of bias that will be debiased')
    parser.add_argument('--method', default=None, type=str,
                        help='The method used for debiasing.')

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    # Set seed
    set_seed(args)

    params = Params.from_file(args.config_file, args.overrides)
    cuda_device = params.params.get('trainer').get('cuda_device', -1)

    create_serialization_dir(params, args.output_dir, recover=False, force=True)
    params.to_file(os.path.join(args.output_dir, CONFIG_NAME))

    teacher_path = params.pop("teacher_path", None)
    # support multi dataset training.
    all_datasets = training_util.datasets_from_params(params)

    # student model initialized from a pretrained QA model
    if args.model_archive is not None:
        student_model = load_archive(args.model_archive).model
    else:
        datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))
        if args.recover and os.path.exists(os.path.join(args.output_dir, "vocabulary")):
            vocab = Vocabulary.from_files(os.path.join(args.output_dir, "vocabulary"))
            params.pop("vocabulary", {})
        else:
            vocab = Vocabulary.from_params(
                params.pop("vocabulary", {}),
                (instance for key, dataset in all_datasets.items()
                 for instance in dataset
                 if key in datasets_for_vocab_creation)
            )

            vocab.save_to_files(os.path.join(args.output_dir, "vocabulary"))
        student_model = Model.from_params(vocab=vocab, params=params.pop("model"))

    val_iterator = DataIterator.from_params(params.pop("validation_iterator"))
    val_iterator.index_with(student_model.vocab)

    teacher_models = load_teacher_model(teacher_path, cuda_device)

    if cuda_device >= 0:
        student_model.to(device=cuda_device)

    logger.info("Training/evaluation parameters %s", params.params)

    # Training
    if args.do_train:
        # load scaling weight of each dataset based on each bias type, only need to load it once
        with open("../bias-weights/dataset-bias-weight.json", "r") as f_w:
            informative_dict = json.load(f_w)
        all_pred_dict = {}
        for dataset_name in teacher_models.keys():
            all_pred_dict[dataset_name] = {}
            if args.bias_type == "combine":
                for bs in ["wh-word", "emptyqst", "lexical", "bidaf"]:
                    all_pred_dict[dataset_name][bs] = get_pred_per_bias(bs, dataset_name)
            else:
                all_pred_dict[dataset_name][args.bias_type] = get_pred_per_bias(args.bias_type, dataset_name)

        num_epochs = params.params.get('trainer').get('num_epochs')
        global_step, tr_loss = train(params, all_datasets, student_model, val_iterator, cuda_device, teachers=teacher_models)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Now tar up results
        archive_model(args.output_dir, weights="model_state_epoch_{}.th".format(num_epochs - 1))

    if args.do_eval:
        logging.info("evaluate student model")
        evaluate(student_model, all_datasets["validation"], val_iterator, cuda_device, batch_weight_key="")
