"""
Helper functions for loading and preprocessing data, evaluation accuracy metrics, embedding aggregation,
and model component validation.

Credits goes to https://www.kaggle.com/code/yuval6967/toxic-train-bert-base-pytorch for preprocessing codes
"""
import torch 
import random
import logging
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
from parameter_config import IDENTITY_COLUMNS, FILENAME, TOPIC_EMBEDDINGS_PATH, TARGET, TEXT_FIELD

warnings.filterwarnings(action='once')
logging.basicConfig(filename=FILENAME, level=logging.DEBUG, format='')

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

def binarize_target(data, target):
    return (data[target] >= 0.5).astype(float)

def load_data(train_df, identity_columns, y_columns, batch_size):
    """
    Split the given dataset into train, test, val split using attribute-level threshold of 0.15.
    """
    # Keep entries that has any associated target identity attributes
    train_df = train_df[(train_df[identity_columns].fillna(0) >= 0.5).astype(int).sum(axis = 1) > 0]
    logging.debug('%s records after filtering ', len(train_df))
    logging.debug("hate_distribution = %s\n", np.unique((train_df[TARGET] >= 0.5).astype(int), return_counts = True))

    train_df['binary_target']  = (train_df[TARGET] >= 0.5).astype(float)
    # Optional: Convert label to binary
    train_df[y_columns] = (train_df[y_columns] >= 0.5).astype(float)
    train_df = train_df.drop_duplicates(subset = ["comment_text"])

    THRESHOLD = 0.15
    target_hate_distribution = [train_df[train_df[i] == 1]["binary_target"].value_counts()[1] for i in identity_columns]
    sample_count = (np.array(target_hate_distribution) * THRESHOLD).astype(int)

    test_df = sample_balanced_dataset(train_df, train_df.keys(), sample_count)
    train_df = train_df.drop(test_df.index, axis=0)
    val_df = sample_balanced_dataset(train_df, train_df.keys(), sample_count)
    train_df = train_df.drop(val_df.index, axis=0)

    if len(train_df) % batch_size != 0:
        remove_n = len(train_df) % batch_size
        drop_indices = np.random.choice(train_df.index, remove_n, replace=False)
        train_df = train_df.drop(drop_indices)

    return train_df, test_df, val_df

def get_topic_embeddings(topic_embedding_path, identity_columns):
    """
    Retrieve identity topic embeddings using pre-trained GloVe dict. 
    """
    glove_vocab_path, glove_embs_path = topic_embedding_path
    glove_vocab = np.load(glove_vocab_path)
    glove_embs = np.load(glove_embs_path)

    topic_embeddings = []
    for topic in identity_columns:
        words = topic.lower().split("_")
        embeddings = [glove_embs[np.where(glove_vocab == word)[0][0]] for word in words if word != "or" and word != "target"]
        topic_embeddings.append(torch.mean(torch.tensor(np.array(embeddings)).to(torch.float32), axis = 0))

    topic_embeddings = torch.stack(topic_embeddings)
 
    del glove_embs
    del glove_vocab
    return topic_embeddings

def get_hypernet_params(weights_pred, layer_shapes, num_target_groups):
    """
    Split the given weights list into corresponding Hypernet parameters for the defined layers. 
    -> Here we consider only a single layer
    """
    idx, params = 0, []
    layers = layer_shapes.copy()
    for shape in layers:
        shape = list(shape)
        shape[0] = num_target_groups
        shape = tuple(shape)
        offset = np.prod(shape[1:])
        params.append(weights_pred[:, idx: idx + offset].reshape(shape))
        idx += offset

    return params

def filter_mlp(x, params, strategy):
    """
    A simple MLP layer that implements either "early" or "late" fusion strategy.
    """
    batch_size = x.size(0)

    w, b = params
    if strategy == "late_fusion":
        w  = w.unsqueeze(0).expand(batch_size, -1, -1)
        b  = b.unsqueeze(0).expand(batch_size, -1)

    x = torch.bmm(x.unsqueeze(1), w).squeeze(1) + b
    x = torch.nn.Tanh()(x)
    x = torch.nn.Dropout(0.2)(x)
    return x

def get_filter_embedding(strategy, emb_outputs, params, y_batch, num_target_groups, device):
    """
    Obtain the filtered embedding using either "early" or "late" filtering strategy. 
    - Parameter Ensemble ("early") filtering: aggregate the individual topic-specific weights and biases before passing 
    into a single MLP layer for output.
    
    - Combinatorial Embedding ("late") filtering: aggregate the individual topic-specific embeddings obtained from 
    individual topic MLP filters.
    """
    averaged_embedding = []
    if strategy == "late_fusion":
        filtered_embeddings = [filter_mlp(emb_outputs.to(device), \
                                          [params[0][i].to(device), params[1][i].to(device)], strategy).to(device) \
                               for i in range(num_target_groups)]
        filtered_embeddings = torch.stack(filtered_embeddings) \
                                .reshape(-1, num_target_groups, len(params[1][0])).to(device)
        indicators = torch.div(y_batch, y_batch.sum(axis = 1).unsqueeze(1)) #[:, np.newaxis])
        indicators = indicators.to(device)
        averaged_embedding = torch.bmm(indicators.unsqueeze(1).float(), \
                                       torch.tensor(filtered_embeddings).float()).squeeze(1)
        """
        # Equivalent to: 

        zipped_embeddings = list(zip(*filtered_embeddings))
        for i in range(len(zipped_embeddings)):
            sample_target_embeddings = list(zipped_embeddings[i]) # single sample
            target = (y_batch[:, 1:][i, :] >= 0.5).int()
            avg = average_masked_embeddings(sample_target_embeddings, target.to(device).long())
            averaged_embedding.append(avg)
        averaged_embedding = torch.stack(averaged_embedding)
        """

    elif strategy == "early_fusion":
        batch_size = emb_outputs.size(0)
        param_shape = params[0][1].size(0)
        indicators = torch.div(y_batch, y_batch.sum(axis = 1).unsqueeze(1)).float()

        # Reshape the indicator matrix to have dimensions (batch_size, num_topics, param_shape [, param_shape])
        weight_indicators = (indicators.unsqueeze(2).unsqueeze(3)) * \
            torch.ones((batch_size, num_target_groups, param_shape, param_shape), device=device).float()
        bias_indicators = (indicators.unsqueeze(2)) * \
            torch.ones((batch_size, num_target_groups, param_shape), device=device).float()
        
        averaged_weight = torch.sum(torch.mul(weight_indicators , params[0].to(device).float()), axis = 1)
        averaged_bias = torch.sum(torch.mul(bias_indicators , params[1].to(device).float()), axis = 1)
        averaged_embedding = filter_mlp(emb_outputs, [averaged_weight, averaged_bias], strategy)
        """
        for i in range(len(y_batch)):
            sample = emb_outputs[i]
            target = (y_batch[:, 1:][i, :] >= 0.5).int()
            averaged_weight = average_masked_embeddings(params[0], target.to(device).long())
            averaged_bias = average_masked_embeddings(params[1], target.to(device).long())
            averaged_embedding.append(filter_mlp(sample.unsqueeze(0), [averaged_weight,averaged_bias]))
        averaged_embedding = torch.stack(averaged_embedding).squeeze(1)
        """
    return averaged_embedding

def sample_balanced_dataset(train_df, columns, sample_count):
    """
    Sample for a subset of data from both the negative (neutral texts) and positive (hateful texts) 
    according to a given list of sample counts for each target-identity group.
    """
    train_copy = train_df.copy()
    train_copy[IDENTITY_COLUMNS + ["binary_target"]] = (train_copy[IDENTITY_COLUMNS + ["binary_target"]]\
                                                        .fillna(0) >= 0.5).astype(int)
    pos_samples = train_copy[train_copy["binary_target"] == 1].copy()
    neg_samples = train_copy[train_copy["binary_target"] == 0].copy()

    current_samples = np.zeros(len(IDENTITY_COLUMNS))
    count = 0
    test_df = pd.DataFrame(columns = columns)
    while True:
        for i in IDENTITY_COLUMNS:
            ind = IDENTITY_COLUMNS.index(i)
            if current_samples[ind] < sample_count[ind]:
                while True:
                    sample = pos_samples[pos_samples[i] == 1].sample(random_state=count)
                    if (current_samples + sample[IDENTITY_COLUMNS].to_numpy() <= sample_count).all():
                        current_samples += sample[IDENTITY_COLUMNS].to_numpy()[0]
                        test_df  = pd.concat([test_df, sample])
                        pos_samples.drop(sample.index, inplace = True)
                        break
                    count += 1
        if (current_samples == sample_count).all():
            break

    # Sample Neg
    current_samples = np.zeros(len(IDENTITY_COLUMNS))
    count = 0
    while True:
        for i in IDENTITY_COLUMNS:
            ind = IDENTITY_COLUMNS.index(i)
            if current_samples[ind] < sample_count[ind]:
                while True:
                    sample = neg_samples[neg_samples[i] == 1].sample(random_state=count)
                    if (current_samples + sample[IDENTITY_COLUMNS].to_numpy() <= sample_count).all():
                        current_samples += sample[IDENTITY_COLUMNS].to_numpy()[0]
                        test_df  = pd.concat([test_df, sample])
                        neg_samples.drop(sample.index, inplace = True)
                        break
                    count += 1
        if (current_samples == sample_count).all():
            break

    return test_df

def convert_lines_onfly(example, max_seq_length,tokenizer,min_seq=0):
    """
    Credit goes to https://www.kaggle.com/code/yuval6967/toxic-train-bert-base-pytorch
    """
    max_seq_length -=2
    results =np.zeros((example.shape[0],max_seq_length+2),dtype=np.int_)
    longest=min_seq
    for i,text in enumerate(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>longest:
            longest = len(tokens_a) 
        if len(tokens_a)>max_seq_length:
        # if too long, randomly erase tokens
            a = np.arange(len(tokens_a))
            np.random.shuffle(a)
            tokens_a = list(np.array(tokens_a)[np.sort(a[:max_seq_length])])
        results[i] = np.array(tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * \
                                                                (max_seq_length - len(tokens_a)))
    results=results[:,:min(max_seq_length,longest)+2]
    return np.array(results)

def preprocess(data,hint=None,do_lower=True):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'+"\n\t"
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' '+p+' ')
        return text
    
    def intonation(text,hint,do_lower):
        new_text=[]
        for token in text.split():
            if do_lower:
                new_text.append(token.lower())
            else:
                new_text.append(token)
            if hint and token.isupper() and len(token)>1:
                new_text.append(hint)
        return " ".join(new_text) 
    data = data.astype(str).apply(lambda x: clean_special_chars(intonation(x,hint,do_lower), punct))
    return data

def average_masked_embeddings(embeddings, labels):
    """
    Aggregate the individual topic embeddings associated with a given sample
    """
    # Get embedding if topic is relevant
    masked_embeddings = torch.stack([num for num, bit in zip(embeddings, labels) if bit == 1])
    # Average the masked embeddings
    masked_embeddings = torch.mean(masked_embeddings, axis = 0)
    return masked_embeddings

def calculate_weights(data, y_columns):
    """
    Calculate the inverse frequency weights for each of the target-identity groups and hatefulness class label.
    """
    weights_tensor = data.copy()
    for i in y_columns:
        inv_ratio = 1 / ((weights_tensor[i] >= 0.5).astype(float).value_counts() / len(weights_tensor))
        weights_tensor[i] = (weights_tensor[i] >= 0.5).astype(float).replace({0: inv_ratio[0], 1: inv_ratio[1]})
    
    weights_tensor = torch.tensor(weights_tensor[y_columns].to_numpy()).float()
    return weights_tensor

def FPR(pred, ground_truth):
    """
    False positive rate
    """
    return len(pred[(pred == 1) & (ground_truth == 0)]) / len(ground_truth[ground_truth == 0])

def nFPED(pred, ground_truth, targets, num_target_groups):
    """
    Average FPR absolute distance
    """
    logging.debug("FPR dist = %s\n", np.abs([FPR(pred, ground_truth) - \
                   FPR(pred[np.array(targets[i]).astype(bool)], ground_truth[np.array(targets[i]).astype(bool)]) for i in range(num_target_groups)]))
    
    return np.sum(np.abs([FPR(pred, ground_truth) - \
                   FPR(pred[np.array(targets[i]).astype(bool)], ground_truth[np.array(targets[i]).astype(bool)]) \
                    for i in range(num_target_groups)])) / num_target_groups

def FNR(pred, ground_truth):
    """
    False negative rate
    """
    return len(pred[(pred == 0) & (ground_truth == 1)]) / len(ground_truth[ground_truth == 1])

def nFNED(pred, ground_truth, targets, num_target_groups):
    """
    Average FNR absolute distance
    """
    logging.debug("FNR dist = %s\n", np.abs([FNR(pred, ground_truth) - \
                   FNR(pred[np.array(targets[i]).astype(bool)], ground_truth[np.array(targets[i]).astype(bool)]) for i in range(num_target_groups)]))
    
    return np.sum(np.abs([FNR(pred, ground_truth) - \
                   FNR(pred[np.array(targets[i]).astype(bool)], ground_truth[np.array(targets[i]).astype(bool)]) \
                    for i in range(num_target_groups)])) / num_target_groups

def embedding_validation(embedding_model, hate_speech_classifier, tokenizer, test_df, batch_size, \
                         device, identity_columns):
    """
    Validate the embedding model: 
    - Overall F1-score & Acc, 
    - Target-wise hatefulness Acc, F1, Recall, Precision, Avg F1 Distance
    - nFPED. nFNED
    """
    num_target_groups = len(identity_columns)

    sentences = preprocess(test_df[TEXT_FIELD].astype(str).fillna("DUMMY_VALUE")).values
    val_label = test_df[TARGET].values
    test_df=test_df.fillna(0)
    X = sentences
    sorted_label = val_label

    target_labels = test_df[identity_columns].fillna(0).values
    sorted_target_labels = target_labels

    x_test = torch.arange(len(X))
    test = torch.utils.data.TensorDataset(x_test)
    batchs = batch_size
    mx = 320

    for param in embedding_model.parameters():
        param.requires_grad=False
    for param in hate_speech_classifier.parameters():
        param.requires_grad=False

    _=embedding_model.to(device)
    _=hate_speech_classifier.to(device)
    torch.cuda.empty_cache()

    test_loader = torch.utils.data.DataLoader(test, batch_size=batchs, shuffle=False)
    embedding_model.eval()
    hate_speech_classifier.eval()

    tk0 = tqdm(test_loader,leave=False)
    tranct=0
    y_label = np.zeros((len(X)))
    preds = np.zeros((len(X)))
    test_preds = np.zeros((len(X)))
    val_targets = [[] for x in range(num_target_groups)]
    for i,(ind_batch, ) in enumerate(tk0):
        x_batch=torch.tensor(convert_lines_onfly(X[ind_batch.numpy()], mx,tokenizer))
        output_embeddings = embedding_model(x_batch.to(device))
        y_pred = hate_speech_classifier(output_embeddings)

        y_label[i*batch_size:(i+1)*batch_size] = sorted_label[ind_batch.numpy()]
        preds[i*batch_size:(i+1)*batch_size]=torch.sigmoid(y_pred).detach().cpu().squeeze().numpy()
        test_preds[i*batch_size:(i+1)*batch_size] = y_pred.cpu().squeeze().numpy()

        for j in range(num_target_groups):
            val_targets[j].append(sorted_target_labels[ind_batch.numpy()][:, j])

        tranct=tranct+batchs*(x_batch.shape[1]==mx)
        tk0.set_postfix(trunct=tranct,gpu_memory=torch.cuda.memory_allocated() // 1024 ** 2,batch_len=x_batch.shape[1])

    y_preds = preds
    y_labels = (y_label >= 0.5).astype(int)
    cts_pred_hate = y_preds.astype(float)
    y_preditioncs = (y_preds >= 0.5).astype(int)

    for i in range(num_target_groups):
        val_targets[i] = np.concatenate(val_targets[i]).ravel().astype(int)

    val_f1 = f1_score(y_labels, y_preditioncs, zero_division=0)
    hate_target_f1 = [f1_score(y_preditioncs[np.array(val_targets[i]).astype(bool)], \
                                       y_labels[np.array(val_targets[i]).astype(bool)], zero_division=0) \
                                        for i in range(num_target_groups)]
    
    val_binary_auc = roc_auc_score(y_labels, y_preditioncs)
    val_auc = roc_auc_score(y_labels, cts_pred_hate)
    val_nFPED = nFPED(y_preditioncs, y_labels, val_targets, num_target_groups)
    val_nFNED = nFNED(y_preditioncs, y_labels, val_targets, num_target_groups)
    harmonic_fairness = (2 * val_nFPED * val_nFNED) / (val_nFPED + val_nFNED)

    logging.debug("F1 = %s", val_f1)
    logging.debug("Acc = %s", accuracy_score(y_labels, y_preditioncs))
    logging.debug("Binary rocauc: %s", val_binary_auc)
    logging.debug("rocauc: %s", val_auc)
    logging.debug("Recall: %s", recall_score(y_preditioncs, y_labels, zero_division=0))
    logging.debug("Precision: %s", precision_score(y_preditioncs, y_labels, zero_division=0))
    logging.debug("nFPED: %s", nFPED(y_preditioncs, y_labels, val_targets, num_target_groups))
    logging.debug("nFNED: %s\n", nFNED(y_preditioncs, y_labels, val_targets, num_target_groups))
    logging.debug(f"Harmonic Fairness: {harmonic_fairness}")

    return (val_f1, harmonic_fairness)

def group_valid(embedding_model, hypernet, group_classifiers, test_df, num_target_groups, \
                device, hypernet_device, tokenizer, batch_size, topic_embs, filter_layer_shapes, strategy):
    """
    Validate the individual target-group classifers (Essentially the same as the training code): 
    - Acc, F1, Recall & Precision
    """
    val_targets = [[] for x in range(num_target_groups)]
    val_preds = [[] for x in range(num_target_groups)]

    sentences = preprocess(test_df[TEXT_FIELD].astype(str).fillna("DUMMY_VALUE")).values
    val_label = test_df[IDENTITY_COLUMNS].fillna(0).values
    test_df=test_df.fillna(0)

    X = sentences
    sorted_label = val_label
    x_test = torch.arange(len(X))
    test = torch.utils.data.TensorDataset(x_test)
    batchs = batch_size
    mx = 320
    test_loader = torch.utils.data.DataLoader(test, batch_size = batchs, shuffle = False)

    for param in embedding_model.parameters():
        param.requires_grad=False
    for param in hypernet.parameters():
        param.requires_grad=False
    hypernet.eval()

    torch.cuda.empty_cache()

    _=embedding_model.to(device)
    embedding_model.eval()

    tk0 = tqdm(test_loader,leave=False)
    tranct=0
    nb_tr_steps=0

    with torch.no_grad():
        for ind,(ind_batch, ) in enumerate(tk0):
            x_batch=torch.tensor(convert_lines_onfly(X[ind_batch.numpy()], mx,tokenizer))
            emb_outputs = embedding_model(x_batch.to(device))
            y_batch = torch.tensor(sorted_label[ind_batch.numpy()]).to(device)
            weights_pred = hypernet(topic_embs).to(hypernet_device) # generate weights for all topics
            params = get_hypernet_params(weights_pred, filter_layer_shapes[0], num_target_groups)

            # Get masked + averaged embeddings for hatespeech classifier
            averaged_embedding = get_filter_embedding(strategy, emb_outputs, params, y_batch, \
                                                      num_target_groups, device).to(device)

            for i in range(num_target_groups):
                group_classifiers[i].eval()
                pred_group_labels = group_classifiers[i](averaged_embedding).to(device)
                val_preds[i].append((pred_group_labels).detach().cpu().squeeze().numpy())
                val_targets[i].append((sorted_label[:,i][ind_batch.numpy()] >= 0.5).astype(int))

            tranct=tranct+batchs*(x_batch.shape[1]==mx)
            tk0.set_postfix(trunct=tranct,gpu_memory=torch.cuda.memory_allocated() // 1024 ** 2, \
                            batch_len=x_batch.shape[1])

            nb_tr_steps += 1

    for i in range(num_target_groups):
        val_preds[i] = np.concatenate(val_preds[i]).ravel()
        val_preds[i] = torch.sigmoid(torch.tensor(val_preds[i])).numpy()
        val_preds[i] = (val_preds[i] >= 0.5).astype(int)
        val_targets[i] = np.concatenate(val_targets[i]).ravel().astype(int)

    logging.debug("F1 = %s", [f1_score(val_targets[i], val_preds[i], zero_division=0) for i in range(num_target_groups)])
    logging.debug("Acc = %s", [accuracy_score(val_targets[i], val_preds[i]) for i in range(num_target_groups)])
    logging.debug("Recall_score = %s", [recall_score(val_targets[i], val_preds[i], zero_division=0) \
                                        for i in range(num_target_groups)])
    logging.debug("Pcision_score = %s\n", [precision_score(val_targets[i], val_preds[i], zero_division=0) \
                                           for i in range(num_target_groups)])

def hate_valid(embedding_model, hypernet, hate_speech_classifier, group_classifiers, num_target_groups, test_df, \
               tokenizer, device, hypernet_device, batch_size, filter_layer_shapes, strategy, topic_embs, test_identities):
    """
    Validate the hate classifier (Essentially the same as the training code): 
    - Overall F1-score & Acc
    - Target-wise hate classiciation: Acc, F1, Recall, Precision, Avg F1 Distance
    - nFPED. nFNED
    - Target-group classiciation: Acc, F1, Recall, Precision
    """
    y_columns=[TARGET] + test_identities
    num_test_groups = len(test_identities)
    sentences = preprocess(test_df[TEXT_FIELD].astype(str).fillna("DUMMY_VALUE")).values
    test_df = test_df.fillna(0)
    weights_tensor = calculate_weights(test_df, y_columns)

    X = sentences
    x_test = torch.arange(len(X))
    test_df[y_columns] = (test_df[y_columns] >= 0.5).astype(float)
    y = test_df[y_columns].values
    y = torch.tensor(y,dtype=torch.float32)

    test = torch.utils.data.TensorDataset(x_test, torch.tensor(y,dtype=torch.float32), weights_tensor)
    batchs = batch_size
    mx = 320

    torch.cuda.empty_cache()
    test_loader = torch.utils.data.DataLoader(test, batch_size=batchs, shuffle=False)
    tk0 = tqdm(test_loader,leave=False)
    
    # Turn off training mode
    for param in embedding_model.parameters():
        param.requires_grad=False
    for param in hate_speech_classifier.parameters():
        param.requires_grad=False
    for i in range(num_target_groups):
        for param in group_classifiers[i].parameters():
            param.requires_grad=False
        group_classifiers[i].eval()

    for param in hypernet.parameters():
        param.requires_grad=False
    hypernet.eval()
    embedding_model.eval()
    hate_speech_classifier.eval()

    val_pred_hate = []
    val_hate_label = []
    val_targets = [[] for x in range(num_test_groups)]

    topic_embs = get_topic_embeddings(TOPIC_EMBEDDINGS_PATH, test_identities).to(hypernet_device)
    with torch.no_grad():
        for ind,(ind_batch, y_batch, w_batch) in enumerate(tk0):
            y_batch = y_batch.to(device)
            x_batch=torch.tensor(convert_lines_onfly(X[ind_batch.numpy()], mx,tokenizer))
            emb_outputs = embedding_model(x_batch.to(device))

            weights_pred = hypernet(topic_embs).to(hypernet_device) # generate weights for all topics -> linear
            params = get_hypernet_params(weights_pred, filter_layer_shapes[0], num_test_groups)
            
            val_averaged_embedding = get_filter_embedding(strategy, emb_outputs, params, y_batch[:, 1:], \
                                                          num_test_groups, device)
            val_hs_logits = hate_speech_classifier(val_averaged_embedding).to(device)
            val_pred_hate.append((val_hs_logits).detach().cpu().numpy())
            val_hate_label.append((y_batch[:, 0].detach().cpu().numpy() >= 0.5).astype(int))

            for i in range(num_test_groups):
                val_targets[i].append((y_batch[:, 1:][:, i].detach().cpu().numpy() >= 0.5).astype(int))

    val_pred_hate = np.concatenate(val_pred_hate).ravel()
    val_pred_hate = torch.sigmoid(torch.tensor(val_pred_hate)).numpy()
    cts_pred_hate = val_pred_hate.astype(float)
    val_pred_hate = (val_pred_hate >= 0.5).astype(int)
    val_hate_label = np.concatenate(val_hate_label).ravel().astype(int)
    
    for i in range(num_test_groups):
        val_targets[i] = np.concatenate(val_targets[i]).ravel().astype(int)

    acc = accuracy_score(val_pred_hate, val_hate_label)
    val_f1 = f1_score(val_pred_hate, val_hate_label, zero_division=0)
    val_binary_auc = roc_auc_score(val_hate_label, val_pred_hate)
    val_auc = roc_auc_score(val_hate_label, cts_pred_hate)

    val_nFPED = nFPED(val_pred_hate, val_hate_label, val_targets, num_test_groups)
    val_nFNED = nFNED(val_pred_hate, val_hate_label, val_targets, num_test_groups)
    recall = recall_score(val_pred_hate, val_hate_label, zero_division=0)
    precision = precision_score(val_pred_hate, val_hate_label, zero_division=0)
    harmonic_fairness = (2 * val_nFPED * val_nFNED) / (val_nFPED + val_nFNED)

    logging.debug("Hate Accuracy: %s", acc)
    logging.debug("Hate F1: %s", val_f1)
    logging.debug("Hate Recall: %s", recall)
    logging.debug("Hate Precision: %s", precision)
    logging.debug("nFPED: %s", val_nFPED)
    logging.debug("nFNED: %s\n", val_nFNED)
    logging.debug(f"Harmonic Fairness: {harmonic_fairness}")

    return (acc, val_f1, val_nFPED, val_nFNED, harmonic_fairness, recall, precision, val_auc, val_binary_auc)