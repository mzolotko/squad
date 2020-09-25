"""Preprocess SQuaD for BERT

Usage:
    > source activate squad
    > python setup.py

Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

Author:
    based on Chris Chute (chute@stanford.edu)
"""

import numpy as np
import os
import spacy
import ujson as json
import urllib.request

from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile
import torch

from util import strip_last_ones

import time

from transformers import BertTokenizer


def download_url(url, output_path, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)


def url_to_data_path(url):
    return os.path.join('./data/', url.split('/')[-1])


def download(args):
    downloads = [
        # Can add other downloads here (e.g., other word vectors)
        ('GloVe word vectors', args.glove_url),
    ]

    for name, url in downloads:
        output_path = url_to_data_path(url)
        if not os.path.exists(output_path):
            print(f'Downloading {name}...')
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith('.zip'):
            extracted_path = output_path.replace('.zip', '')
            if not os.path.exists(extracted_path):
                print(f'Unzipping {name}...')
                with ZipFile(output_path, 'r') as zip_fh:
                    zip_fh.extractall(extracted_path)

    print('Downloading spacy language model...')
    run(['python', '-m', 'spacy', 'download', 'en'])

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

def word_tokenize_bert(sent1, sent2, pad_limit):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    return tokenizer, tokenizer(text=sent1, text_pair=sent2, padding='max_length', max_length=pad_limit) # dict ('input_ids': List[list], token_type_ids:  List[list], attention_mask:  List[list])

def convert_idx(text, tokens):
    prev_position = 0
    spans = []
    for token in tokens:
        if token.startswith('##'):  # for Bert wordpiece tokens ## is just signal that this is
            token = token[2:]       # a second, third etc. piece of the word. Strip them for correct span calculation
        current_position = text.find(token, prev_position)

        # if the token is not found, assume that this is [UNK] and that the corresponding text has length of 1
        # in such case, if this token is part of the answer, it will for sure be part of y expressed in terms of
        # token (rather than symbol) indices. The drawback that some subsequent irrelevant [UNK]s can also become
        # part of y. We assume this will hardly ever be the case
        if current_position < 0:
            if token != '[UNK]':
                print(f"Token {token} cannot be found")
            #raise Exception()
            current_position = prev_position
            spans.append((current_position, current_position + 1))
            current_position += 1
            prev_position = current_position
        else:
            spans.append((current_position, current_position + len(token)))
            current_position += len(token)
            prev_position = current_position
    return spans


def process_file(filename, data_type, args):

    para_limit = args.para_limit
    ques_limit = args.ques_limit
    print(f"Pre-processing {data_type} examples...")
    examples_spacy = []
    contexts = []
    questions = []
    all_answers = []
    uuids = []
    answers_start_end = []
    eval_examples = {}
    total = -1
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens_spacy = word_tokenize(context)
                spans_spacy = convert_idx(context, context_tokens_spacy)
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')

                    y1s, y2s = [], []
                    answer_texts = []
                    answer_bounds = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_bounds.append((answer_start, answer_end))
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans_spacy):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {'context_tokens_spacy': context_tokens_spacy,
                               "y1s_spacy": y1s,
                               "y2s_spacy": y2s,
                               "id": total}
                    answers_start_end.append(answer_bounds) # list (number of sent) of list (number of answers) of tuples (dim == 2, start and end)
                    examples_spacy.append(example)
                    contexts.append(context)
                    questions.append(ques)
                    all_answers.append(answer_texts)
                    uuids.append(qa["id"])

        tokenizer, tokenizer_result = word_tokenize_bert(sent1=questions, sent2=contexts, pad_limit=para_limit+ques_limit)
        tokens_bert = tokenizer_result['input_ids']
        token_type_ids = tokenizer_result['token_type_ids']
        attention_mask = tokenizer_result['attention_mask']

        ys_bert = []
        for i, _ in enumerate(questions):
            tokens_sentence = torch.tensor(tokens_bert[i])
            mask = strip_last_ones(torch.tensor(token_type_ids[i]).unsqueeze(dim=0)).bool()
            spans = convert_idx(contexts[i], tokenizer.convert_ids_to_tokens(torch.masked_select(tokens_sentence, mask)))
            ys_bert_sentence = []
            shift = int(np.argwhere(np.array(token_type_ids[i]) == 1).ravel()[0]) # first index where token_type_ids == 1

            for answer in answers_start_end[i]:
                answer_span = []
                answer_start = answer[0]
                answer_end = answer[1]
                for idx, span in enumerate(spans):
                    if not (answer_end <= span[0] or answer_start >= span[1]):
                        answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]


                y1 = y1 + shift
                y2 = y2 + shift
                ys_bert_sentence.append((y1, y2))
            eval_examples[str(i)] = {"context": contexts[i],
                                                 "question": questions[i],
                                                 "spans": spans,
                                                 "answers": all_answers[i],
                                                 "shift": shift,
                                                 "uuid": uuids[i]}
            ys_bert.append(ys_bert_sentence)

        print(f"{len(questions)} len of questions")
        print(f"{len(examples_spacy)} questions in total (spacy)")
        print(f"{len(tokens_bert)} questions in total (bert)")
        assert len(examples_spacy) == len(tokens_bert)

    return examples_spacy,  eval_examples, ys_bert, tokens_bert, token_type_ids, attention_mask


def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None, num_vectors=None):
    print(f"Pre-processing {data_type} vectors...")
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=num_vectors):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print(f"{len(embedding_dict)} / {len(filtered_elements)} tokens have corresponding {data_type} embedding vector")
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print(f"{len(filtered_elements)} tokens have corresponding {data_type} embedding vector")

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def convert_to_features(args, data, word2idx_dict, char2idx_dict, is_test):
    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    char_limit = args.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs


def is_answerable(n, list_of_ex):
    return bool(list_of_ex[n])


def build_features(args, examples, ys_bert, tokens_bert, token_type_ids, attention_mask, data_type, out_file, is_test=False):
    '''
    examples: list of dict( context_tokens_spacy, tokens_bert: list[para_limit + ques_limit], token_type_ids: list[para_limit + ques_limit],
                            attention_mask: list[para_limit + ques_limit], y1s, y2s, id)
    '''

    print('lengths of examples and ys_bert', len(examples), len(ys_bert))
    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit

    def drop_example(n, ex, is_test_=False):
        if is_test_:
            drop = False
        else:
            drop = len(tokens_bert[n]) > (para_limit + ques_limit) or \
                   (is_answerable(n, ys_bert) and
                    ys_bert[n][0][1] - ys_bert[n][0][0] > ans_limit)  # just 1st answer is taken into account

        return drop

    print(f"Converting {data_type} examples to indices...")
    total = 0
    total_ = 0
    meta = {}
    to_drop_lst = []
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total_ += 1
        if drop_example(n, example, is_test):
            #print(example)
            to_drop_lst.append(n)
            continue
        total += 1

        if is_answerable(n, ys_bert):
            start, end = ys_bert[n][-1][0], ys_bert[n][-1][1]
        else:
            start, end = 0, 0

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])
    print('len of tokens_bert:', len(tokens_bert) )

    for i in list(range(len(tokens_bert)))[::-1]:  # traversing in reverse order
        if len(tokens_bert[i]) > (para_limit + ques_limit) or \
        (is_answerable(i, ys_bert) and
         ys_bert[i][0][1] - ys_bert[i][0][0] > ans_limit):
            tokens_bert.pop(i)
            token_type_ids.pop(i)
            attention_mask.pop(i)

    assert len(tokens_bert) == len(y1s)


    np.savez(out_file,
             idxs=np.array(tokens_bert),
             token_type_ids=np.array(token_type_ids),
             attention_mask=np.array(attention_mask),
             y1s=np.array(y1s),
             y2s=np.array(y2s),
             ids=np.array(ids))
    print(f'files saved, {time.time() - tim:.2f} s. elapsed')
    print(f"Built {total} / {total_} instances of features in total")
    meta["total"] = total
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def pre_process(args):
    tim = time.time()
    #print('started pre_process')
    # Process training set and use it to decide on the word/character vocabularies
    train_examples, train_eval, train_ys_bert, train_tokens_bert, train_token_type_ids, train_attention_mask = process_file(args.train_file, "train", args)

    # Process dev and test sets
    dev_examples, dev_eval, dev_ys_bert, dev_tokens_bert, dev_token_type_ids, dev_attention_mask = process_file(args.dev_file, "dev", args)
    #print(f'exited process_file, {time.time() - tim:.2f} s. elapsed')
    build_features(args, train_examples, train_ys_bert, train_tokens_bert, train_token_type_ids, train_attention_mask,  "train", args.train_record_file)
    dev_meta = build_features(args, dev_examples, dev_ys_bert, dev_tokens_bert, dev_token_type_ids, dev_attention_mask, "dev", args.dev_record_file)
    #print(f'exited build_features, {time.time() - tim:.2f} s. elapsed')
    ###################
    if args.include_test_examples:
        test_examples,  test_eval, test_ys_bert, test_tokens_bert, test_token_type_ids, test_attention_mask = process_file(args.test_file, "test", args)
        save(args.test_eval_file, test_eval, message="test eval")
        test_meta = build_features(args, test_examples, test_ys_bert, test_tokens_bert, test_token_type_ids, test_attention_mask,  "test",
                                   args.test_record_file, is_test=True)
        save(args.test_meta_file, test_meta, message="test meta")

    save(args.train_eval_file, train_eval, message="train eval")
    save(args.dev_eval_file, dev_eval, message="dev eval")
    print(f'exited saving dev_eval, {time.time() - tim:.2f} s. elapsed')
    save(args.dev_meta_file, dev_meta, message="dev meta")
    print(f'exited saving dev_meta_file, {time.time() - tim:.2f} s. elapsed')


if __name__ == '__main__':
    # Get command-line args
    tim = time.time()
    print('started main')
    args_ = get_setup_args()

    # Download resources
    #download(args_)

    # Import spacy language model
    nlp = spacy.blank("en")

    # Preprocess dataset
    args_.train_file = url_to_data_path(args_.train_url)
    args_.dev_file = url_to_data_path(args_.dev_url)

    ############
    if args_.include_test_examples:
        args_.test_file = url_to_data_path(args_.test_url)

    print(f'entering pre_process, {time.time() - tim:.2f} s. elapsed')
    pre_process(args_)
    print(f'exit pre_process, {time.time() - tim:.2f} s. elapsed')
