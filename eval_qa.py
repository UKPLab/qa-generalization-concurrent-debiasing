"""Evaluation script adapted from the MRQA official evaluation script.
Usage:
    python eval_qa.py dataset_file.jsonl.gz prediction_file.json prediction.csv
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import string
import re
import json
import gzip
import sys
from collections import Counter
from allennlp.common.file_utils import cached_path
import pandas as pd


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def read_predictions(prediction_file):
    with open(prediction_file) as f:
        predictions = json.load(f)
    return predictions


def read_answers(gold_file):
    answers = {}
    with gzip.open(gold_file, 'rb') as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            if i == 0 and 'header' in example:
                continue
            for qa in example['qas']:
                answers[qa['qid']] = [qa['answers'],qa['question'], example['context'] ]
    return answers


def evaluate(answers, predictions, dataset_file, skip_no_answer=False):
    f1 = exact_match = total = 0
    values = []
    for qid, example in answers.items():
        ground_truths = example[0]
        if qid not in predictions:
            if not skip_no_answer:
                message = 'Unanswered question %s will receive score 0.' % qid
                # added by mingzhu 
                predictions[qid] = ["unanswerable", 0.0, 0.0]
                print(message)
                total += 1
            continue
        total += 1
        # prediction = predictions[qid]
        prediction = predictions[qid][0]
        truth = ";".join(ground_truths)
        context = example[2]
        question = example[1]
        em = str(metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths))
        f_score = str(metric_max_over_ground_truths(
            f1_score, prediction, ground_truths))
        # modify to add start and end bias weight in the log file, by Mingzhu 20200116 begin
        #values.append([qid, question, prediction, truth, em, f_score, context])
        if em == "False":
            start_bias, end_bias = 0.0, 0.0
            predictions[qid][1], predictions[qid][2] = 0.0, 0.0
        else:
            start_bias, end_bias = predictions[qid][1], predictions[qid][2]
        values.append([qid, question, prediction, truth, em, f_score, start_bias, end_bias, context])
        # end
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)


    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    # modify to add start and end bias weight in the log file, mingzhu 20200116
    # columns = ["qid", "question", "prediction", "gold", "em", "f_score", "context"]
    columns = ["qid", "question", "prediction", "gold", "em", "f_score", "start bias weight", "end bias weight", "context"]
    df = pd.DataFrame(values)
    df.columns = columns
    df.to_csv(dataset_file, sep="\t")
    return {'exact_match': exact_match, 'f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation for MRQA Workshop Shared Task')
    parser.add_argument('dataset_file', type=str, help='Dataset File')
    parser.add_argument('prediction_file', type=str, help='Prediction File')
    parser.add_argument('prediction_log_file', type=str, help='Prediction Log File')
    parser.add_argument('--skip-no-answer', action='store_true')
    args = parser.parse_args()

    answers = read_answers(cached_path(args.dataset_file))
    predictions = read_predictions(cached_path(args.prediction_file))
    metrics = evaluate(answers, predictions, args.prediction_log_file, args.skip_no_answer)

    with open(cached_path(args.prediction_file), "w") as f:
        json.dump(predictions, f)

    print(json.dumps(metrics))
