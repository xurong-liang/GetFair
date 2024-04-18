import torch
import torch.nn as nn
import numpy as np
from APG_module import APG_MLP
from transformers import AdamW
from transformers import BertForSequenceClassification, \
                        RobertaForSequenceClassification,\
                        DistilBertForSequenceClassification
from parameter_config import EMB_MODEL, MODEL_DEVICE_IDS, EMB_MODEL_PATH, EMB_CONFIG_PATH, HIDDEN_DIM, TOPIC_EMBEDDINGS_SHAPE

class EmbeddingClass(torch.nn.Module):
    """
    Embedding model class: using specified base model.
    """
    def __init__(self):
        super(EmbeddingClass, self).__init__()
        if EMB_MODEL == "roberta":
            self.model = RobertaForSequenceClassification.from_pretrained(EMB_MODEL_PATH, num_labels = 1)
        elif EMB_MODEL == "gpt2":
            self.model = DistilBertForSequenceClassification.from_pretrained(EMB_MODEL_PATH, num_labels = 1)
            self.pre_classifier = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        else:
            if EMB_CONFIG_PATH is not None:
                self.model = BertForSequenceClassification.from_pretrained(EMB_MODEL_PATH, config=EMB_CONFIG_PATH, num_labels = 1)
            else:
                self.model = BertForSequenceClassification.from_pretrained(EMB_MODEL_PATH, num_labels = 1)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, inputs):
        if EMB_MODEL == "gpt2":
            #GPT2
            x = self.model.distilbert(inputs)
            hidden_state = x[0]
            pooled_output = hidden_state[:, 0]
            pooled_output = self.pre_classifier(pooled_output)
            x = nn.ReLU()(pooled_output)
        else:
            if EMB_MODEL == "roberta":
                x = self.model.roberta(inputs)
            else:
                x = self.model.bert(inputs)
            x = x[1]

        emb = self.dropout(x)
        return emb
    
class MLP(nn.Module):
    """
    Simple MLP module.
    """
    def __init__(self, input_size, layers_data, binary=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.binary = binary
        self.dropout = nn.Dropout(p=0.3)
        for size, activation in layers_data:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size  # For the next layer
            if activation is not None:
                self.layers.append(activation)
                self.layers.append(self.dropout)

        if self.binary:
            self.output = nn.Linear(input_size, 1)
    
    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)

        if self.binary:
            input_data = self.output(input_data)

        return input_data
    
class Hypernetwork(nn.Module):
    """
    Hypernet module for generating model parameters.
    """
    def __init__(self, input_dim, num_weights_to_generate):
        super().__init__()
        self.l1 = nn.Linear(input_dim, 128)
        self.activation = nn.ReLU()
        self.l2 = nn.Linear(128, num_weights_to_generate)
    
    def forward(self, input_data):
        x = self.activation(self.l1(input_data))
        x = self.l2(x)
        return x
    
def initialise_models(num_target_groups, filter_layer_shapes, discriminator_layers, hate_classifier_layers, device, hypernet_device,\
                      hypernet_layers, emb_checkpoint = None, rank_k = 5):
    """
    Initialise models:
    (1) Embedding model
    (2) Hypernet
    (3) Target Group classifiers
    (4) Hate speech classifier
    """
    num_weights_to_generate = sum(np.prod(tuples[1:]) for layer in filter_layer_shapes for tuples in layer)

    embedding_model = EmbeddingClass()
    if emb_checkpoint is not None:
        embedding_model.load_state_dict(emb_checkpoint['embedding_model_state_dict'])

    embedding_model= nn.DataParallel(embedding_model, device_ids = MODEL_DEVICE_IDS[0], output_device=device)
    embedding_model.zero_grad()
    embedding_model.to(device)

    hypernet = APG_MLP(TOPIC_EMBEDDINGS_SHAPE, hypernet_layers, output_dim = num_weights_to_generate, rank_k = rank_k) # Or use custom Hypernet
    hypernet = nn.DataParallel(hypernet, device_ids = MODEL_DEVICE_IDS[1], output_device = hypernet_device)

    hypernet.zero_grad()
    hypernet.to(hypernet_device)

    group_classifiers = [MLP(filter_layer_shapes[0][0][-1], [(i, nn.Tanh()) for i in discriminator_layers], binary = True) \
                    for _ in range(num_target_groups)]
    group_classifiers = [nn.DataParallel(model, device_ids = MODEL_DEVICE_IDS[2], output_device=device) for model in group_classifiers]

    [model.zero_grad() for model in group_classifiers]
    [model.to(device) for model in group_classifiers]

    hate_classifier = MLP(filter_layer_shapes[0][0][-1], [(i, nn.Tanh()) for i in hate_classifier_layers], binary = True)
    if emb_checkpoint is not None:
        hate_classifier.load_state_dict(emb_checkpoint['hate_classifer_state_dict'])

    hate_classifier = nn.DataParallel(hate_classifier, device_ids = MODEL_DEVICE_IDS[3], output_device=device)

    hate_classifier.zero_grad()
    hate_classifier.to(device)

    return embedding_model, hypernet, group_classifiers, hate_classifier

def initialise_optimizers(embedding_model, hypernet, group_classifiers, hate_classifier, lr = [2e-5, 2e-5, 2e-5, 2e-5], emb_checkpoint = None):
    """
    Initialise optimizers for:
    (1) Embedding model
    (2) Hypernet
    (3) Group classifiers
    (4) Hate speech classifier
    """
    # Add or remove weight decay to layers other than the following list
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    def get_params(param_list):
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_list if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_list if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return optimizer_grouped_parameters

    param_optimizer = list(hypernet.named_parameters())
    optimizer_grouped_parameters = get_params(param_optimizer)
    group_emb_optimizer = AdamW(optimizer_grouped_parameters, lr = lr[0], correct_bias=True)

    group_optimizer = [AdamW(params = model.parameters(), lr = lr[1], correct_bias=True) for model in group_classifiers]

    param_optimizer = list(hate_classifier.named_parameters()) + list(hypernet.named_parameters())
    optimizer_grouped_parameters = get_params(param_optimizer)
    hate_speech_optimizer = AdamW(optimizer_grouped_parameters, lr = lr[2], correct_bias = True)

    param_optimizer = list(embedding_model.named_parameters()) +  list(hate_classifier.named_parameters())
    optimizer_grouped_parameters = get_params(param_optimizer)
    embedding_update_optimizer = AdamW(optimizer_grouped_parameters, lr = lr[3], correct_bias = True)
    
    # Load model checkpoints if availiable
    if emb_checkpoint is not None:
        embedding_update_optimizer.load_state_dict(emb_checkpoint['optimizer_state_dict'])
    
    return embedding_update_optimizer, group_emb_optimizer, group_optimizer, hate_speech_optimizer