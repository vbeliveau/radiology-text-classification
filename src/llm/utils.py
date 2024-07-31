#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:09:01 2024

@author: vbeliveau
"""

import re
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    # BitsAndBytesConfig,
    pipeline,
)

root_dir = '/proc_data1/bd5/nlp'
default_batch_size = 128


def get_model_info(model, root_dir):

    if model == "biomistral":
        # model_id = "BioMistral/BioMistral-7B"
        model_id = f"{root_dir}/.cache/huggingface/hub/models--BioMistral--BioMistral-7B/snapshots/9a11e1ffa817c211cbb52ee1fb312dc6b61b40a5"
        model_str = "BioMistral-7B"

    if model == "llama3-8B":
        model_id = f"{root_dir}/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/7129260dd854a80eb10ace5f61c20324b472b31c"
        model_str = "Meta-Llama-3-70B-Instruct"

    if model == "llama3-70B":
        model_id = f"{root_dir}/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/7129260dd854a80eb10ace5f61c20324b472b31c"
        model_str = "Meta-Llama-3-70B-Instruct"

    if model == "meditron":
        model_id = "epfl-llm/meditron-70b"
        model_str = "meditron-70b"

    if model == "munin":
        model_id = "RJuro/munin-neuralbeagle-7b"
        model_str = "munin-neuralbeagle-7b"

    return model_id, model_str


def load_data(project_name, data_dir=f"{root_dir}/data/preproc", val=False):

    df = pd.read_csv(f"{data_dir}/{project_name}.csv")
    # Remove nans.. should not be happening at this stage
    df = df[[isinstance(text, str) for text in df["text"]]]

    # Convert labels to numerical categories
    factors = pd.factorize(df['label'])
    category_names = list(factors[1])
    categories_summary = "Categories: "

    n_factors = []
    for n_factor, factor in enumerate(factors[1]):
        n_factors += [sum([factor_ == n_factor for factor_ in factors[0]])]
        categories_summary += f" {factor} (n={n_factors[-1]})"
    print(categories_summary)
    df['label'] = factors[0]
    df["label_text"] = [factors[1][n] for n in factors[0]]
    n_classes = len(factors[1])

    def get_min_labels(labels):
        n_labels = [
            len([label_ for label_ in labels if label_ == label])
            for label in labels
        ]
        return min(n_labels)

    # Create train/test datasets
    train_test_dataset = Dataset.from_pandas(
        df).train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_dataset["train"]
    eval_dataset = train_test_dataset["test"]

    # Further split data into train/validation
    if val:
        df_train = pd.DataFrame({
            "text": train_dataset["text"],
            "label": train_dataset["label"],
            "label_text": train_dataset["label_text"]
        })
        train_val_dataset = Dataset.from_pandas(
            df_train).train_test_split(test_size=0.2, seed=42)
        train_dataset = train_val_dataset["train"]
        val_dataset = train_val_dataset["test"]

    # Count unique classes and get their weight
    classes_count = [len([label_ for label_ in train_dataset["label"] if label_ == label])
                     for label in list(set(train_dataset["label"]))]
    print(f"Classes count in training dataset: {classes_count}")
    classes_weight = Tensor([1/count for count in classes_count])

    out_dict = {
        "category_names": category_names,
        "classes_count": classes_count,
        "classes_weight": classes_weight,
        "eval_dataset": eval_dataset,
        "n_classes": n_classes,
        "train_dataset": train_dataset,
    }

    if val:
        out_dict.update({"val_dataset": val_dataset})

    return out_dict


def load_pipeline(model_id):

    print("Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if model_id.startswith("llama"):  # chat pipeline
        pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    else:

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
        )

        pipe = pipeline(
            "text-generation",
            device_map="auto",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
        )

    return pipe


def biomistral_prompt(query_text, labels, samples):
    prompt = "<s>[INST]You are an experienced radiologist that help users extract infromation from radiology reports. Your task is to categorize texts in the following categories:\n\n"
    prompt += "\n".join(labels)
    prompt += '\n\nYou will only respond with the category. Do not include the word "Category". Do not provide explanations or notes.'
    if samples is not None:
        for ni in range(len(samples)):
            text = samples.iloc[ni]["text"].replace("\n", " ")
            category = samples.iloc[ni]["label_text"]
            if ni == 0:
                prompt += f"\nCategorize the text: {text}\n[/INST]{category}</s>"
            else:
                prompt += f"[INST]\nCategorize the text: {text}\n[/INST]{category}</s>"
    else:
        prompt += "[/INST]"
    prompt += f"[INST]\nCategorize the text: {query_text}\n[/INST]"
    return prompt


def meditron_prompt(query_text, labels, samples):

    prompt = "<|im_start|> system\nYou are an experienced radiologist that help users extract infromation from radiology reports. Your task is to categorize texts in the following categories:\n\n"
    prompt += "\n".join(labels)
    prompt += '\n\nYou will only respond with the category. Do not include the word "Category". Do not provide explanations or notes.<|im_end|>'
    if samples is not None:
        for ni in range(len(samples)):
            text = samples.iloc[ni]["text"].replace("\n", " ")
            category = samples.iloc[ni]["label_text"]
            prompt += f"\n<|im_start|> question\nCategorize the text: {text}\n<|im_end|>\n<|im_start|> answer\n{category}<|im_end|>"
    prompt += f"\n<|im_start|> question\nCategorize the text: {text}\n<|im_end|>\n<|im_start|> answer\n"
    return prompt

    # if args.shots > 0:
    #     prompt = prompt[:-1]
    # if "orca" in args.checkpoint_name:
    #     system_msg = "You are an AI assistant who helps people find information."
    #     return f"<|im_start|> system\n{system_msg}<|im_end|>\n<|im_start|> question\n{prompt}<|im_end|>\n<|im_start|> answer\n"
    # elif "medical" in args.checkpoint_name:
    #     system_msg = "You are a helpful, respectful and honest assistant." + \
    #         "Always answer as helpfully as possible, while being safe." + \
    #         "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content." + \
    #         "Please ensure that your responses are socially unbiased and positive in nature.\n\n" + \
    #         "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct." + \
    #         "If you don't know the answer to a question, please don't share false information."""
    #     return f"<|im_start|> system\n{system_msg}<|im_end|>\n <|im_start|> user\n{prompt}<|im_end|>\n <|im_start|> assistant\n"
    # elif np.any([x in args.checkpoint_name for x in ["medmcqa", "medqa", "pubmedqa"]]):
    #     return f"<|im_start|>question\n{prompt}<|im_end|>\n<|im_start|>answer\n"
    # elif "med42" in args.checkpoint_name:
    #     if "Question:" in prompt:
    #         question = prompt.split("Question:")[1].strip()
    #     else:
    #         question = prompt
    #     return f'''\n<|system|>: You are a helpful medical assistant created by M42 Health in the UAE.\n<|prompter|>:{question}\n<|assistant|>:'''
    # else:
    #     return prompt


def munin_prompt(query_text, labels, samples):
    prompt = "You are an experienced radiologist that help users extract infromation from radiology reports. Categorize the text in <<<>>> into one of the following predefined categories:\n\n"
    prompt += "\n".join(labels)
    prompt += '\n\nYou will only respond with the category. Do not include the word "Category". Do not provide explanations or notes.'
    if samples is not None:
        prompt += "\n\n####\nHere are some examples:\n\n"
        for ni, sample in samples.iterrows():
            text = sample["text"].replace("\n", " ")
            category = sample["label_text"]
            prompt += f"Inquiry: {text}\nCategory: {category}\n"
        prompt += "###\n\n"
    prompt += f"<<<\nInquiry: {query_text}\n>>>"
    return prompt


def munin_danish_prompt(query_text, labels, samples):
    prompt = "Du er en erfaren radiolog, der hjælper brugere med at udtrække informationer fra radiologirapporter. Kategoriser teksten i <<<>>> i en af ​​følgende foruddefinerede kategorier:\n\n"
    prompt += "\n".join(labels)
    prompt += '\n\nDu svarer kun med kategorien. Medtag ikke ordet "Kategori". Giv ikke forklaringer eller noter.'
    if samples is not None:
        prompt += "\n\n####\nHer er nogle eksempler:\n\n"
        for ni, sample in samples.iterrows():
            text = sample["text"].replace("\n", " ")
            category = sample["label_text"]
            prompt += f"Forespørgsel: {text}\Kategori: {category}\n"
        prompt += "###\n\n"
    prompt += f"<<<\nForespørgsel: {query_text}\n>>>"
    return prompt


def llama_instruct_prompt(query_text, labels, samples):

    query = "You are an experienced radiologist that help users extract infromation from radiology reports. Your task is to categorize texts in the following categories:\n\n"
    query += "\n".join(labels)
    query += '\n\nYou will only respond with the category. Do not include the word "Category". Do not provide explanations or notes.'

    messages = [
        {
            "role": "system",
            "content": query
        }
    ]

    if samples is not None:
        for ni, sample in samples.iterrows():
            text = sample["text"].replace("\n", " ")
            messages += [{
                "role": "user",
                "content": f"Categorize the text: {text}"
            },
                {
                "role": "assistant",
                "content": sample["label_text"]
            }
            ]

    messages += [{
        "role": "user",
        "content": f"Categorize the text: {query_text}"
    }]

    return messages


def few_shots_predict_biomistral(
    pipe,
    query_text,
    df_train,
    labels,
    max_retry=1,
    n_samples=2,
    seed=42,
    valid_labels=None,
    verbose=False,
):

    for n_try in range(max_retry):

        if n_try > 0:
            print(f"Retry {n_try}...")

        if n_samples > 0:
            samples = df_train.groupby('label').apply(
                lambda x: x.sample(n_samples, random_state=seed + n_try),
                include_groups=False,
            )
        else:
            samples = None

        prompt = biomistral_prompt(query_text, labels, samples)

        if verbose:
            print(prompt)

        sequences = pipe(
            prompt,
            do_sample=True,
            max_new_tokens=10,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
        pred_label = sequences[0]["generated_text"][len(prompt):]

        status = True
        if valid_labels is not None and pred_label not in valid_labels:
            pred_label = f"INVALID LABEL: {pred_label}"
            status = False
        else:
            break

        if verbose:
            print(pred_label)

    return pred_label, status


def few_shots_predict_munin(
    pipe,
    query_text,
    df_train,
    labels,
    max_retry=1,
    n_samples=2,
    seed=42,
    valid_labels=None,
    verbose=False,
):

    for n_try in range(max_retry):

        if n_try > 0:
            print(f"Retry {n_try}...")

        if n_samples > 0:
            samples = df_train.groupby('label').apply(
                lambda x: x.sample(n_samples, random_state=seed + n_try),
                include_groups=False,
            )
        else:
            samples = None

        prompt = munin_prompt(query_text, labels, samples)

        if verbose:
            print(prompt)

        sequences = pipe(
            prompt,
            do_sample=True,
            max_new_tokens=10,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
        pred = sequences[0]["generated_text"]

        re_match = re.search("(?<=>>>\nCategory: ).*?(?=\n)", pred)

        # Handle output
        status = True

        if re_match is None:
            pred_label = f"INVALID OUTPUT MATCH: {sequences[0]['generated_text']}"
            status = False
        else:

            pred_label = re_match.group()

            if valid_labels is not None and pred_label not in valid_labels:
                pred_label = f"INVALID LABEL: {sequences[0]['generated_text']}"
                status = False
            else:
                break

        if verbose:
            print(pred_label)

    return pred_label, status


def few_shots_predict_llama(
    pipe,
    query_text,
    df_train,
    labels,
    max_retry=1,
    n_samples=2,
    seed=42,
    valid_labels=None,
    verbose=False,
):

    for n_try in range(max_retry):

        if n_try > 0:
            print(f"Retry {n_try}...")

        if n_samples > 0:
            samples = df_train.groupby('label').apply(
                lambda x: x.sample(n_samples, random_state=seed + n_try),
                include_groups=False,
            )
        else:
            samples = None

        messages = llama_instruct_prompt(query_text, labels, samples)

        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipe(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        pred_label = outputs[0]["generated_text"][-1]["content"]

        status = True
        if valid_labels is not None and pred_label not in valid_labels:
            pred_label = f"INVALID LABEL: {pred_label}"
            status = False
        else:
            break

        if verbose:
            print(pred_label)

    return pred_label, status


def predict_samples(
    pipe,
    df_train,
    df_test,
    labels,
    max_retry=1,
    model_id=None,
    n_samples=1,
    valid_labels=None,
    verbose=False
):

    pred_list = []
    status_list = []

    for n_sample, sample in tqdm(df_test.iterrows(), total=len(df_test)):

        if model_id == "biomistral":
            pred, status = few_shots_predict_biomistral(
                pipe,
                sample["text"],
                df_train,
                labels,
                max_retry=max_retry,
                n_samples=n_samples,
                seed=n_sample,
                valid_labels=valid_labels,
                verbose=verbose,
            )

        if model_id == "munin":
            pred, status = few_shots_predict_munin(
                pipe,
                sample["text"],
                df_train,
                labels,
                max_retry=max_retry,
                n_samples=n_samples,
                seed=n_sample,
                valid_labels=valid_labels,
                verbose=verbose,
            )

        if model_id.startswith("llama"):
            pred, status = few_shots_predict_llama(
                pipe,
                sample["text"],
                df_train,
                labels,
                max_retry=max_retry,
                n_samples=n_samples,
                seed=n_sample,
                valid_labels=valid_labels,
                verbose=verbose,
            )

        pred_list += [pred]
        status_list += [status]

        if verbose:
            print(
                f"True category: {sample['label_text']}, Predicted category: {pred}")

    return pred_list, status_list


def eval_few_shots(true_labels, pred_labels, labels):

    # Plot confusion matrix
    y_true = [[n for n in range(len(labels)) if label ==
               labels[n]][0] for label in true_labels]

    y_pred = [[n for n in range(len(labels)) if label ==
               labels[n]][0] for label in pred_labels]

    metrics = compute_metrics(y_true, y_pred)
    print(metrics)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    return metrics


def compute_metrics(y_true, y_pred):
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_score_macro": f1_score(y_true, y_pred, average="macro"),
    }
