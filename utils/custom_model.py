import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, RobertaModel


class NLPModel(nn.Module):

    def __init__(self, args):
        super(NLPModel, self).__init__()
        self.args = args

        self.text_model_config = AutoConfig.from_pretrained(args.text_model_name_or_path, num_labels=args.num_labels)
        self.text_model = RobertaModel.from_pretrained(args.text_model_name_or_path,
                                                       config=self.text_model_config)
        self.text_classifier = RobertaClassificationHead(self.text_model_config)

    def forward(self, batch):
        text_output = self.text_model(input_ids=batch['input_ids'],
                                      attention_mask=batch['attention_mask'],
                                      token_type_ids=batch['token_type_ids'] if self.args.token_type_ids else None)

        # text last_hidden_state.shape : (BS, SEQ_LEN, HIDDEN)
        text_last_hidden_state = text_output.last_hidden_state
        x = self.text_classifier(text_last_hidden_state)

        return x


class ImageModel(nn.Module):

    def __init__(self, args):
        super(ImageModel, self).__init__()
        self.args = args

        self.image_model = timm.create_model(args.image_model_name_or_path, pretrained=True,
                                             num_classes=args.num_labels)

    def forward(self, batch):
        x = self.image_model(batch['image'])

        return x


class MultiModalModel(nn.Module):

    def __init__(self, args):
        super(MultiModalModel, self).__init__()
        self.args = args

        self.text_model_config = AutoConfig.from_pretrained(args.text_model_name_or_path, num_labels=args.num_labels)
        self.text_model = RobertaModel.from_pretrained(args.text_model_name_or_path,
                                                       config=self.text_model_config)
        self.text_classifier = RobertaClassificationHead(self.text_model_config)

        self.image_model = timm.create_model(args.image_model_name_or_path, pretrained=True,
                                             num_classes=args.num_labels)

        self.classifier = nn.Linear(args.num_labels * 2, args.num_labels)

    def forward(self, batch):
        text_output = self.text_model(input_ids=batch['input_ids'],
                                      attention_mask=batch['attention_mask'],
                                      token_type_ids=batch['token_type_ids'] if self.args.token_type_ids else None)

        # text last_hidden_state.shape : (BS, SEQ_LEN, HIDDEN)
        text_last_hidden_state = text_output.last_hidden_state
        text_logits = self.text_classifier(text_last_hidden_state)

        image_logits = self.image_model(batch['image'])

        x = torch.cat((text_logits, image_logits), dim=1)
        x = self.classifier(F.relu(x))
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
