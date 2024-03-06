#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:15:05 2023

@author: vbeliveau
"""

import os

from collections import OrderedDict
from typing import Optional, Tuple, Union

import webbrowser
import torch
import torch.nn as nn

from torch import Tensor

from captum.attr import (
    LayerConductance,
    LayerIntegratedGradients,
    IntegratedGradients,
    InterpretableEmbeddingBase,
    TokenReferenceBase,
    visualization,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)

from transformers.models.distilbert.modeling_distilbert import DistilBertModel, Embeddings
from transformers.modeling_outputs import BaseModelOutput

from datasets import load_from_disk
from setfit import SetFitModel
from sentence_transformers.models import Transformer


root_dir = '/proc_data1/bd5/nlp'
os.chdir(root_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"


def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


# model_path = 'models/BD-WP5 - Axial T2-trainable-head'
project = "FCD"
model_path = f"models/setfit/{project}-trainable-head/final_checkpoint"
dataset = load_from_disk(f"models/setfit/{project}-trainable-head/dataset.hf")
model = SetFitModel.from_pretrained(model_path).to(device)


class TransformerWrapper(Transformer):

    def __init__(self, model: Transformer):
        # super().__init__()  # Transformer init requires model_path, skipping
        self.__dict__ = model.__dict__.copy()    # just a shallow copy

    def forward(self, word_embeddings):
        # https://github.com/UKPLab/sentence-transformers/blob/179b659621c680371394d507683b25ba7faa0dd8/sentence_transformers/models/Transformer.py#L68
        """Returns token_embeddings, cls_token"""
        # trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        # if 'token_type_ids' in features:
        #     trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(
            inputs_embeds=word_embeddings, return_dict=False)
        output_tokens = output_states[0]

        attention_mask = torch.ones(word_embeddings.size()[:-1], device=device)
        features = {'token_embeddings': output_tokens,
                    'attention_mask': attention_mask}
        # features.update({'token_embeddings': output_tokens, 'attention_mask': attention_mask})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features


class DistilBertModelWrapper(DistilBertModel):

    def __init__(self, model: Transformer):
        # super().__init__()  # Transformer init requires model_path, skipping
        self.__dict__ = model.__dict__.copy()    # just a shallow copy

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(
                input_shape, device=device)  # (bs, seq_length)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        # embeddings = self.embeddings(input_ids, inputs_embeds)  # (bs, seq_length, dim)
        embeddings = self.embeddings(inputs_embeds)  # (bs, seq_length, dim)

        return self.transformer(
            x=embeddings,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class EmbeddingsWrapper(Embeddings):
    def __init__(self, embeddings: Embeddings):
        self.__dict__ = embeddings.__dict__.copy()    # just a shallow copy

    # def forward(self, input_ids: torch.Tensor, input_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            input_ids (torch.Tensor):
                torch.tensor(bs, max_seq_length) The token ids to embed.
            input_embeds (*optional*, torch.Tensor):
                The pre-computed word embeddings. Can only be passed if the input ids are `None`.


        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        # if input_ids is not None:
        #     input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)

        seq_length = input_embeds.size(1)

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            raise ValueError('Not implemented.')
            # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(
            position_ids)  # (bs, max_seq_length, dim)

        # (bs, max_seq_length, dim)
        embeddings = input_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


transformer_wrapper = TransformerWrapper(model.model_body._modules['0'])
transformer_wrapper.auto_model = DistilBertModelWrapper(
    transformer_wrapper.auto_model)
transformer_wrapper.auto_model.embeddings = EmbeddingsWrapper(
    transformer_wrapper.auto_model.embeddings)

modules = OrderedDict({
    'transformer': TransformerWrapper(model.model_body._modules['0'])
})

modules.update(
    OrderedDict(
        {
            key: value
            for key, value in model.model_body._modules.items()
            if key != "0"
        }
    )
)

modules.update({'model_head': model.model_head})

seq_model = nn.Sequential(modules)


def predict_wrapper(embeddings: Tensor):
    probs = seq_model(embeddings)['probs']
    return probs


# ig = IntegratedGradients(predict_wrapper, seq_model.transformer.auto_model.embeddings)
lig = LayerIntegratedGradients(
    predict_wrapper, seq_model.transformer.auto_model.embeddings)


def sentence_embeddings(model: nn.Sequential, sentence: str):

    device = model.transformer.auto_model.device
    tokenize = model.transformer.tokenize
    word_embeddings = model.transformer.auto_model.embeddings.word_embeddings

    if not isinstance(sentence, list):
        sentence = [sentence]

    with torch.no_grad():

        # https://github.com/UKPLab/sentence-transformers/blob/179b659621c680371394d507683b25ba7faa0dd8/sentence_transformers/SentenceTransformer.py#L161
        input_ids = tokenize(sentence)['input_ids'].to(device)

        # https://github.com/huggingface/transformers/blob/eec0d84e6a
        embeddings = word_embeddings(input_ids)
        embeddings.requires_grad = True

    return embeddings


def interpret_sentence(model: nn.Sequential, sentence: str, true_label: int, attr_label: int):

    model.eval()
    model.zero_grad()

    tokenizer = model.transformer.tokenizer
    # word_embeddings = model.transformer.auto_model.embeddings.word_embeddings

    with torch.no_grad():

        embeddings = sentence_embeddings(model, sentence)
        pred_vec = predict_wrapper(embeddings).detach().cpu().numpy().squeeze()
        pred_label = pred_vec.argmax()
        pred_prob = pred_vec[pred_label]
        # ref_embeddings = \
        #         word_embeddings(torch.tensor(
        #             tokenizer.pad_token_id)[None, None, ...].to(device))
        # ref_embeddings.requires_grad = True

    # compute attributions and approximation delta using integrated gradients
    attributions_ig, delta = lig.attribute(
        embeddings,
        target=attr_label,
        # baselines=ref_embeddings,
        n_steps=500,
        internal_batch_size=40,
        return_convergence_delta=True,
    )

    delta = delta.detach().cpu().numpy()
    print('pred: ', pred_label, '(', '%.2f' %
          pred_prob, ')', ', delta: ', abs(delta))

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence)['input_ids'])
    add_attributions_to_visualizer(
        attributions_ig, tokens, pred_prob, pred_label, true_label, attr_label, delta, vis_data_records_ig)


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def add_attributions_to_visualizer(attributions, tokens, pred_prob, pred_label, true_label, attr_label, delta, vis_data_records):

    attributions = summarize_attributions(attributions)

    # storing couple samples in an array for visualization purposes
    r"""
    A data record for storing attribution relevant information

    word_attributions
    pred_prob
    pred_class
    true_class
    attr_class
    attr_score
    raw_input_ids
    convergence_score
    
    """

    print(tokens)

    tokens = [token[2:] if token.startswith("##") else token
              for token in tokens]

    vis_data_records.append(
        visualization.VisualizationDataRecord(
            attributions,
            pred_prob,
            pred_label,
            true_label,
            attr_label,
            attributions.sum(),
            tokens[:len(attributions)],
            delta
        ))

# %%


# accumalate couple samples in this array for visualization purposes
vis_data_records_ig = []

# for ind in range(len(dataset['test'])):
for ind in range(5):
    sentence = dataset['test'][ind]['text']
    label = dataset['test'][ind]['label']
    interpret_sentence(seq_model, sentence=sentence,
                       true_label=label, attr_label=1)

html_obj = visualization.visualize_text(vis_data_records_ig)

with open('test.html', 'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored HTML file on the default browser
url = 'test.html'
webbrowser.open(url, new=2)

# %% View sentences with errors

# accumalate couple samples in this array for visualization purposes
vis_data_records_ig = []
seq_model.eval()

# for ind in range(len(dataset['test'])):
for ind in range(len(dataset['test'])):
    sentence = dataset['test'][ind]['text']
    label = dataset['test'][ind]['label']
    with torch.no_grad():
        embeddings = sentence_embeddings(seq_model, sentence)
        pred_label = int(predict_wrapper(embeddings).argmax().detach().cpu())
        if pred_label != label:
            interpret_sentence(seq_model, sentence=sentence,
                               true_label=label, attr_label=1)

html_obj = visualization.visualize_text(vis_data_records_ig)

with open('test.html', 'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored HTML file on the default browser
url = 'test.html'
webbrowser.open(url, new=2)
