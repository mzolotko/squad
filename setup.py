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


#def process_file(filename, data_type, word_counter, char_counter):
def process_file(filename, data_type, args):
    #tim = time.time()
    #print('entered process_file')
    para_limit = args.para_limit
    ques_limit = args.ques_limit
    #char_limit = args.char_limit
    print(f"Pre-processing {data_type} examples...")
    examples_spacy = []
    #contexts_all = []
    contexts = []
    #questions_all = []
    questions = []
    ##########################################
    all_answers = []
    uuids = []

    ##########################################

    answers_start_end = []
    eval_examples = {}
    #examples_bert_ref = {}
    ###############################
    #total = 0
    total = -1
    ###############################
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens_spacy = word_tokenize(context)
                #context_tokens_bert = word_tokenize_bert(context, para_limit)['input_ids']  #'input_ids': List (it's just one sentence)
                #context_token_type_ids = word_tokenize_bert(context, para_limit)['token_type_ids']  # token_type_ids:  List
                #context_attention_mask = word_tokenize_bert(context, para_limit)['attention_mask']  # token_type_ids:  List
                #context_chars = [list(token) for token in context_tokens]
                spans_spacy = convert_idx(context, context_tokens_spacy)
                #spans_bert = convert_idx(context, context_tokens_spacy)
                #for token in context_tokens:
                #    word_counter[token] += len(para["qas"])
                #    for char in token:
                #           char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')

                    #ques_tokens_bert = word_tokenize_bert(ques, ques_limit)['input_ids']  #'input_ids': List[list], token_type_ids:  List[list], attention_mask:  List[list])
                    #ques_token_type_ids = word_tokenize_bert(ques, ques_limit)['token_type_ids']  # token_type_ids:  List[list]
                    #ques_attention_mask = word_tokenize_bert(ques, ques_limit)['attention_mask']  # token_type_ids:  List[list]

                    #tokenizer_result = word_tokenize_bert(sent1=ques, sent2=context, pad_limit=para_limit+ques_limit)
                    #tokens_bert = tokenizer_result['input_ids']
                    #token_type_ids = tokenizer_result['token_type_ids']
                    #attention_mask = tokenizer_result['attention_mask']




                    #ques_tokens = word_tokenize(ques)
                    #ques_chars = [list(token) for token in ques_tokens]
                    #for token in ques_tokens:
                    #    word_counter[token] += 1
                    #    for char in token:
                    #        char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    #answer_starts = []
                    #answer_ends = []
                    answer_bounds = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        #answer_starts.append(answer_start)
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
                    example = {#"context_tokens": context_tokens,
                               'context_tokens_spacy': context_tokens_spacy,
                               #'context_tokens_bert': context_tokens_bert,
                               #'context_token_type_ids': context_token_type_ids,
                               #'context_attention_mask': context_attention_mask,
                               #'tokens_bert': tokens_bert,
                               #'token_type_ids': token_type_ids,
                               #'attention_mask': attention_mask,
                               #"context_chars": context_chars,
                               #"ques_tokens": ques_tokens,
                               #"ques_chars": ques_chars,
                               "y1s_spacy": y1s,
                               "y2s_spacy": y2s,
                               "id": total}
                    #answers_dict = {'answer_start': answer_starts,
                    #                    'answer_end': answer_ends}
                    answers_start_end.append(answer_bounds) # list (number of sent) of list (number of answers) of tuples (dim == 2, start and end)
                    examples_spacy.append(example)
                    contexts.append(context)
                    questions.append(ques)
                    #########################################################      
                    #eval_examples[str(total)] = {"context": context,
                    #                             "question": ques,
                    #                             "spans": spans_spacy,
                    #                             "answers": answer_texts,
                    #                             "uuid": qa["id"]}
                    #print(f'next example finished, {time.time() - tim:.2f} s. elapsed')
                    all_answers.append(answer_texts)
                    uuids.append(qa["id"])

                    #########################################################

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
	    #############################################
            shift = int(np.argwhere(np.array(token_type_ids[i]) == 1).ravel()[0]) # first index where token_type_ids == 1
	    #############################################
            for answer in answers_start_end[i]:

                answer_span = []
                answer_start = answer[0]
                answer_end = answer[1]

                for idx, span in enumerate(spans):
                    if not (answer_end <= span[0] or answer_start >= span[1]):
                        answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
		###########################################################
                #shift = np.argwhere(np.array(token_type_ids[i]) == 1).ravel()[0] # first index where token_type_ids == 1
		################################################################
                #print(f'answer {i}: y1 and y2 before shift: {y1}, {y2}' )
		###########################################
                y1 = y1 + shift
                y2 = y2 + shift
		##########################################
                #print(f'after shift: {y1}, {y2}' )
                ys_bert_sentence.append((y1, y2))
            eval_examples[str(i)] = {"context": contexts[i],
                                                 "question": questions[i],
                                                 "spans": spans,
                                                 "answers": all_answers[i],
                                                 "shift": shift,
                                                 "uuid": uuids[i]}

	    #if ys_bert_sentence != None:  # will be false if there are no answers: e.g. in the test set
            ys_bert.append(ys_bert_sentence)
            #examples_bert_ref[str(i)] = {"spans_bert": spans, "ys_bert": ys_bert_sentence}


        print(f"{len(questions)} len of questions")
        print(f"{len(examples_spacy)} questions in total (spacy)")
        print(f"{len(tokens_bert)} questions in total (bert)")
        assert len(examples_spacy) == len(tokens_bert)


    #return examples_spacy, examples_bert_ref,  eval_examples, ys_bert, tokens_bert, token_type_ids, attention_mask
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
    #return len(example['y2s']) > 0 and len(example['y1s']) > 0
    #print(n, len(list_of_ex))
    return bool(list_of_ex[n])


#def build_features(args, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
def build_features(args, examples, ys_bert, tokens_bert, token_type_ids, attention_mask, data_type, out_file, is_test=False):
    '''
    examples: list of dict( context_tokens_spacy, tokens_bert: list[para_limit + ques_limit], token_type_ids: list[para_limit + ques_limit],
                            attention_mask: list[para_limit + ques_limit], y1s, y2s, id)
    '''
    #tim = time.time()
    #print('entered build_features')

    #print('examples[0]')
    #print(examples[0])
    #print('examples[1]')
    #print(examples[1])
    #print('tokens_bert[0]')
    #print(tokens_bert[0])
    #print('tokens_bert[1]')
    #print(tokens_bert[1])
    print('lengths of examples and ys_bert', len(examples), len(ys_bert))
    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit
    #char_limit = args.char_limit

    def drop_example(n, ex, is_test_=False):
        if is_test_:
            drop = False
        else:
            #drop = len(ex["context_tokens_bert"]) > para_limit or \
            #       len(ex["ques_tokens_bert"]) > ques_limit or \
            # drop = len(ex["tokens_bert"]) > (para_limit + ques_limit) or \
            drop = len(tokens_bert[n]) > (para_limit + ques_limit) or \
                   (is_answerable(n, ys_bert) and
                    #ex["y2s"][0] - ex["y1s"][0] > ans_limit)
                    ys_bert[n][0][1] - ys_bert[n][0][0] > ans_limit)  # just 1st answer is taken into account
                    # indices: number of sentence, number of answer, index of start/end

        return drop

    print(f"Converting {data_type} examples to indices...")
    total = 0
    total_ = 0
    meta = {}
    to_drop_lst = []
    #context_idxs = []
    #context_token_type_ids = []
    #context_attention_mask = []
    #context_char_idxs = []
    #idxs = []
    #token_type_ids = []
    #attention_mask = []
    #ques_char_idxs = []
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

        #def _get_word(word):
        #    for each in (word, word.lower(), word.capitalize(), word.upper()):
        #        if each in word2idx_dict:
        #            return word2idx_dict[each]
        #    return 1

        #def _get_char(char):
        #    if char in char2idx_dict:
        #        return char2idx_dict[char]
        #    return 1

        #context_idx = np.zeros([para_limit], dtype=np.int32)
        #context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        #ques_idx = np.zeros([ques_limit], dtype=np.int32)
        #ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        #for i, token in enumerate(example["context_tokens"]):
        #    context_idx[i] = _get_word(token)
        #context_idxs.append(context_idx)
        #context_idxs.append(example['context_tokens_bert'])
        #context_token_type_ids.append(example['context_token_type_ids'])
        #context_attention_mask.append(example['context_attention_mask'])



        #for i, token in enumerate(example["ques_tokens"]):
        #    ques_idx[i] = _get_word(token)
        #ques_idxs.append(ques_idx)
        #idxs.append(example['tokens_bert'])
        #token_type_ids.append(example['token_type_ids'])
        #attention_mask.append(example['attention_mask'])

        #for i, token in enumerate(example["context_chars"]):
        #    for j, char in enumerate(token):
        #        if j == char_limit:
        #            break
        #        context_char_idx[i, j] = _get_char(char)
        #context_char_idxs.append(context_char_idx)

        #for i, token in enumerate(example["ques_chars"]):
        #    for j, char in enumerate(token):
        #        if j == char_limit:
        #            break
        #        ques_char_idx[i, j] = _get_char(char)
        #ques_char_idxs.append(ques_char_idx)

        if is_answerable(n, ys_bert):
            #start, end = example["y1s"][-1], example["y2s"][-1]
            start, end = ys_bert[n][-1][0], ys_bert[n][-1][1]
        else:
            start, end = 0, 0

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])
        #print(f'next example finished, {time.time() - tim:.2f} s. elapsed')
    #print('shape of context_idxs: ', np.array(context_idxs).shape)
    #print('shape of context_token_type_ids: ', np.array(context_token_type_ids).shape)
    #print('shape of context_attention_mask: ', np.array(context_attention_mask).shape)
    #print('shape of idxs: ', np.array(idxs).shape)
    #print('shape of token_type_ids: ', np.array(token_type_ids).shape)
    #print('shape of attention_mask: ', np.array(attention_mask).shape)

    print('len of tokens_bert:', len(tokens_bert) )
    #for i, ex in enumerate(tokens_bert):
    #     print(f'len of tokens_bert[{i}]: {len(tokens_bert[i])}',  )


    #for i, record in enumerate(tokens_bert):
    for i in list(range(len(tokens_bert)))[::-1]:  # traversing in reverse order
        if len(tokens_bert[i]) > (para_limit + ques_limit) or \
        (is_answerable(i, ys_bert) and
         #examples[i]["y2s"][0] - examples[i]["y1s"][0] > ans_limit):
         ys_bert[i][0][1] - ys_bert[i][0][0] > ans_limit):
            tokens_bert.pop(i)
            token_type_ids.pop(i)
            attention_mask.pop(i)



    #tokens_bert = np.array(tokens_bert, dtype=object)
    #token_type_ids = np.array(token_type_ids, dtype=object)
    #attention_mask = np.array(attention_mask, dtype=object)
    #tokens_bert = np.delete(tokens_bert, to_drop_lst)
    #token_type_ids = np.delete(token_type_ids, to_drop_lst)
    #attention_mask = np.delete(attention_mask, to_drop_lst)
    #print(len(tokens_bert) )
    #print(len(y1s) )
    assert len(tokens_bert) == len(y1s)


    np.savez(out_file,
             #context_idxs=np.array(context_idxs),
             #context_token_type_ids=np.array(context_token_type_ids),
             #context_attention_mask=np.array(context_attention_mask),
             #context_char_idxs=np.array(context_char_idxs),
             idxs=np.array(tokens_bert),
             token_type_ids=np.array(token_type_ids),
             attention_mask=np.array(attention_mask),
             #idx=tokens_bert,
             #token_type_ids=token_type_ids,
             #attention_mask=attention_mask,
             #ques_char_idxs=np.array(ques_char_idxs),
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
    #word_counter, char_counter = Counter(), Counter()
    #train_examples, train_eval = process_file(args.train_file, "train", word_counter, char_counter)
    #train_examples, train_ex_bert, train_eval, train_ys_bert, train_tokens_bert, train_token_type_ids, train_attention_mask = process_file(args.train_file, "train", args)
    train_examples, train_eval, train_ys_bert, train_tokens_bert, train_token_type_ids, train_attention_mask = process_file(args.train_file, "train", args)
    #word_emb_mat, word2idx_dict = get_embedding(
    #  word_counter, 'word', emb_file=args.glove_file, vec_size=args.glove_dim, num_vectors=args.glove_num_vecs)
    #char_emb_mat, char2idx_dict = get_embedding(
    #    char_counter, 'char', emb_file=None, vec_size=args.char_dim)

    # Process dev and test sets
    #dev_examples, dev_eval = process_file(args.dev_file, "dev", word_counter, char_counter)
    #dev_examples, dev_ex_bert, dev_eval, dev_ys_bert, dev_tokens_bert, dev_token_type_ids, dev_attention_mask = process_file(args.dev_file, "dev", args)
    dev_examples, dev_eval, dev_ys_bert, dev_tokens_bert, dev_token_type_ids, dev_attention_mask = process_file(args.dev_file, "dev", args)
    #print(f'exited process_file, {time.time() - tim:.2f} s. elapsed')
    build_features(args, train_examples, train_ys_bert, train_tokens_bert, train_token_type_ids, train_attention_mask,  "train", args.train_record_file)
    #dev_meta = build_features(args, dev_examples, "dev", args.dev_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(args, dev_examples, dev_ys_bert, dev_tokens_bert, dev_token_type_ids, dev_attention_mask, "dev", args.dev_record_file)
    #print(f'exited build_features, {time.time() - tim:.2f} s. elapsed')
    ###################
    if args.include_test_examples:
        #test_examples, test_eval = process_file(args.test_file, "test", word_counter, char_counter)
        #test_examples, test_ex_bert,  test_eval, test_ys_bert, test_tokens_bert, test_token_type_ids, test_attention_mask = process_file(args.test_file, "test", args)
        test_examples,  test_eval, test_ys_bert, test_tokens_bert, test_token_type_ids, test_attention_mask = process_file(args.test_file, "test", args)
        save(args.test_eval_file, test_eval, message="test eval")
        test_meta = build_features(args, test_examples, test_ys_bert, test_tokens_bert, test_token_type_ids, test_attention_mask,  "test",
                                   args.test_record_file, is_test=True)
        #test_meta = build_features(args, test_examples, "test",
        #                           args.test_record_file, word2idx_dict, char2idx_dict, is_test=True)
        save(args.test_meta_file, test_meta, message="test meta")

    #save(args.word_emb_file, word_emb_mat, message="word embedding")
    #save(args.char_emb_file, char_emb_mat, message="char embedding")
    save(args.train_eval_file, train_eval, message="train eval")
    #save(args.train_ref_file, train_ex_bert, message="train ref")
    save(args.dev_eval_file, dev_eval, message="dev eval")
    #save(args.dev_ref_file, dev_ex_bert, message="dev ref")
    print(f'exited saving dev_eval, {time.time() - tim:.2f} s. elapsed')
    #save(args.word2idx_file, word2idx_dict, message="word dictionary")
    #save(args.char2idx_file, char2idx_dict, message="char dictionary")
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


    #glove_dir = url_to_data_path(args_.glove_url.replace('.zip', ''))
    #glove_ext = f'.txt' if glove_dir.endswith('d') else f'.{args_.glove_dim}d.txt'
    #args_.glove_file = os.path.join(glove_dir, os.path.basename(glove_dir) + glove_ext)
    print(f'entering pre_process, {time.time() - tim:.2f} s. elapsed')
    pre_process(args_)
    print(f'exit pre_process, {time.time() - tim:.2f} s. elapsed')
