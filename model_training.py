"""
Adversarial model training: (1) Embedding model, (2) target-identity group classifiers, (3) hate speech classifier.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.metrics import f1_score, accuracy_score
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, RobertaTokenizer, AutoTokenizer

from utils import preprocess, convert_lines_onfly, embedding_validation, \
                hate_valid, get_hypernet_params, get_topic_embeddings, get_filter_embedding, \
                calculate_weights, binarize_target, set_seed

from parameter_config import MAX_SEQUENCE_LENGTH, SEED, BATCH_SIZE, IDENTITY_COLUMNS, Y_COLUMNS, LOAD_EMB, \
                    DISCRIMINATOR_LAYERS, HATE_CLASSIFIER_LAYERS, EMBEDDING_EPOCH, TOPIC_EMBEDDINGS_PATH, STRATEGY, LEARNING_RATE, \
                    FILTER_LAYER_SHAPE, NUM_TARGET_GROUPS, FILENAME, GC_K, HC_L, MODULE_EPOCH, SAVE_EMBEDDING_MODEL, EMB_STOP_COUNT, \
                    SAVE_HATESPEECH_MODEL, DATA_PATHS, TARGET, TEXT_FIELD, TEST_UNSEEN_IDENTITY, EARLY_STOPPING_COUNT, \
                    HYPERNERT_LAYERS, GC_WARM_UP, FINE_TUNED_BERT, HYPERPARAMETERS, EMB_MODEL, DEVICE, HYPERNET_DEVICE
from models import *

warnings.filterwarnings(action='once')
torch.autograd.set_detect_anomaly(True)
logging.basicConfig(filename=FILENAME, level=logging.DEBUG, format='')

device=torch.device(DEVICE)
hypernet_device=torch.device(HYPERNET_DEVICE)

def train_emb(X, train, train_df, val_df, embedding_model, hate_classifier, \
          embedding_update_optimizer, tokenizer, scheduler, batch_size, device, embedding_epoch = 1):
    """
    Finetune Embedding model
    """
    logging.debug("===== Train Embedding Model =====\n")
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    
    best_f1 = 0
    stop_count = 0
    tq = tqdm(range(embedding_epoch))
    for epoch in tq:
        logging.debug(f"===== Epoch {epoch} =====\n")
        for param in embedding_model.parameters():
            param.requires_grad=True
        for param in hate_classifier.parameters():
            param.requires_grad=True
        embedding_model.train()
        hate_classifier.train()

        torch.cuda.empty_cache()

        avg_loss = 0.
        avg_accuracy = 0.
        avg_f1 = 0.
        embedding_update_optimizer.zero_grad()
        
        preds = np.zeros((len(X)))
        y_label = np.zeros((len(X)))
        tk0 = tqdm(train_loader, total = len(train_loader),leave = False)
        for i,(ind_batch, y_batch, w_batch) in enumerate(tk0):
            ind_batch.requires_grad = False
            x_batch = torch.tensor(convert_lines_onfly(X[ind_batch.numpy()], MAX_SEQUENCE_LENGTH, tokenizer))
            emb_outputs = embedding_model(x_batch.to(device))
            y_pred = hate_classifier(emb_outputs)

            loss =  F.binary_cross_entropy_with_logits(y_pred.flatten(), y_batch[:, 0].to(device), weight = w_batch[:, 0].to(device))

            y_label[i*batch_size:(i+1)*batch_size] = np.array(y_batch[:, 0])
            preds[i*batch_size:(i+1)*batch_size] = torch.sigmoid(y_pred).detach().cpu().squeeze().numpy()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(embedding_model.parameters()) + list(hate_classifier.parameters()), 1.0)
            embedding_update_optimizer.step()
            embedding_update_optimizer.zero_grad()
            scheduler.step()
            
            lossf = loss.item()
            acc = torch.mean(((torch.sigmoid(y_pred) >= 0.5) == (y_batch[:,0] >= 0.5).to(device)).to(torch.float32)).item()
            f1 = f1_score((np.array(y_batch[:, 0]) >= 0.5).astype(float), (torch.sigmoid(y_pred) >= 0.5).float().cpu().detach().numpy(), average = 'micro')
            tk0.set_postfix(loss = lossf, accuracy = acc, f1 = f1)
            avg_loss += loss.item() / len(train_loader)
            avg_accuracy += acc / len(train_loader)
            avg_f1 += f1 / len(train_loader)

        tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy, avg_f1 = avg_f1)

        # Model Validation
        logging.debug("Train F1 = %s", f1_score((y_label >= 0.5).astype(int), (preds >= 0.5).astype(int)))
        logging.debug("Train Acc = %s\n", accuracy_score((y_label >= 0.5).astype(int), (preds >= 0.5).astype(int)))
        logging.debug("=========== Embedding Train validation ===========")
        _ = embedding_validation(embedding_model, hate_classifier, tokenizer, train_df, batch_size, device, IDENTITY_COLUMNS)
        
        logging.debug("=========== Embedding Val validation ===========")
        logging.debug("Loss: %s", avg_loss)
        val_f1, hf = embedding_validation(embedding_model, hate_classifier, tokenizer, val_df, batch_size, device, IDENTITY_COLUMNS)

        # Save best model
        if val_f1 > best_f1:
            logging.debug("Saving emb model:%s %s", val_f1, hf)
            torch.save({
                'epoch': embedding_epoch,
                'embedding_model_state_dict': embedding_model.module.state_dict(),
                'hate_classifer_state_dict': hate_classifier.module.state_dict(),
                'optimizer_state_dict': embedding_update_optimizer.state_dict(),
                }, SAVE_EMBEDDING_MODEL)
            best_f1 = val_f1
        elif val_f1 < best_f1:
            stop_count += 1
            """Early stopping"""
            if stop_count == EMB_STOP_COUNT:
                break

    return embedding_model, hate_classifier, embedding_update_optimizer

def group_component(k, X, train, hate_classifier, hypernet, group_classifiers, tokenizer, embedding_model, topic_embs, group_optimizer,\
                    batch_size, device, hypernet_device, filter_layer_shapes, strategy, num_target_groups):
    """
    Train target-group classifiers
    """
    logging.debug("===== Train Group Classifier =====\n")

    tq = tqdm(range(k))
    for iteration in tq:
        logging.debug(f"===== Iteration {iteration} =====\n")
        torch.cuda.empty_cache()
        train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)

        for param in hate_classifier.parameters():
            param.requires_grad = False

        for param in hypernet.parameters():
            param.requires_grad = False

        for i in range(num_target_groups):
            for param in group_classifiers[i].parameters():
                param.requires_grad=True
            group_classifiers[i].train()

        tk0 = tqdm(train_loader, total = len(train_loader), leave = False)
        for ind, (ind_batch, y_batch, w_batch) in enumerate(tk0):
            y_batch, w_batch =  y_batch.to(device), w_batch.to(device)
            ind_batch.requires_grad = False
            x_batch=torch.tensor(convert_lines_onfly(X[ind_batch.numpy()], MAX_SEQUENCE_LENGTH, tokenizer))
            emb_outputs = embedding_model(x_batch.to(device))

            # Generate weights for all topics
            weights_pred = hypernet(topic_embs).to(hypernet_device)
            params = get_hypernet_params(weights_pred, filter_layer_shapes[0], num_target_groups)

            averaged_embedding = get_filter_embedding(strategy, emb_outputs, params, y_batch[:, 1:], num_target_groups, device).to(device)
            dis_losses = []
            for i in range(num_target_groups):
                pred_group_labels = group_classifiers[i](averaged_embedding)
                dis_loss = F.binary_cross_entropy_with_logits(pred_group_labels.flatten(), y_batch[:, 1:][:, i].float(), \
                                                              weight = w_batch[:, 1:][:, i])
                dis_losses.append(dis_loss)

                group_optimizer[i].zero_grad()
                dis_loss.backward(retain_graph = True)
                group_optimizer[i].step()

            var_dis_losses = Variable(torch.tensor(dis_losses).data, requires_grad = True)
            tk0.set_postfix(loss = np.round(var_dis_losses.mean().item(),4), group_loss = [np.round(loss.item(), 4) for loss in dis_losses])

        logging.debug("Losses Last:%s, %s\n", np.round(var_dis_losses.mean().item(),5), [np.round(loss.item(), 5) for loss in dis_losses])

    logging.debug("=========== Completed GC Training ===========\n")    
    return hate_classifier, hypernet, group_classifiers, group_optimizer

def hate_component(L, X, train, hate_classifier, hypernet, group_classifiers, tokenizer, embedding_model, topic_embs, hate_speech_optimizer, \
                   pair_count, pair_topic_sim, val_df, test_df, batch_size, device, hypernet_device, filter_layer_shapes, strategy, \
                    num_target_groups, group_weights, best_fairness, best_val_f1, stop_count, param_config = None):
    """
    Train Hate speech classifier:
    1. hate_loss: Hate Speech classification loss
    2. group_loss: Target group discriminator loss
    3. kld_loss: KLDiv loss (Imitation Learning)
    4. l_Reg: Semantic Gap Alignment (Semantic similarity Regulariser)
    """
    logging.debug("===== Train Hate Classifier =====\n")

    tq = tqdm(range(L))
    if param_config:
        w1 = param_config["lambda"]
        w2 = param_config["gamma"]
        w3 = param_config["mu"]

    early_stop = False
    for iteration in tq:
        logging.debug(f"===== Iteration {iteration} =====\n")
        torch.cuda.empty_cache()

        hypernet.train()
        for param in hate_classifier.parameters():
            param.requires_grad = True

        for param in hypernet.parameters():
            param.requires_grad = True
            
        for i in range(num_target_groups):
            group_classifiers[i].train(False)
            for param in group_classifiers[i].parameters():
                param.requires_grad = False

        train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
        hate_speech_optimizer.zero_grad()
        
        avg_hs_loss = 0
        avg_group_loss = 0
        avg_kld_loss = 0
        avg_l_reg = 0
        avg_loss_hate_speech = 0
        count=0

        tk0 = tqdm(train_loader, total = len(train_loader), leave = False)
        for ind,(ind_batch, y_batch, w_batch) in enumerate(tk0):
            y_batch, w_batch =  y_batch.to(device), w_batch.to(device)
            ind_batch.requires_grad = False
            x_batch=torch.tensor(convert_lines_onfly(X[ind_batch.numpy()], MAX_SEQUENCE_LENGTH, tokenizer))
            emb_outputs = embedding_model(x_batch.to(device))
            
            # generate weights for all topics
            weights_pred = hypernet(topic_embs).to(hypernet_device)
            params = get_hypernet_params(weights_pred, filter_layer_shapes[0], num_target_groups)

            # Get masked + averaged embeddings for hatespeech classifier
            averaged_embedding = get_filter_embedding(strategy, emb_outputs, params, y_batch[:, 1:], num_target_groups, device).to(device)

            # Caluclate Hate Classifier loss
            hs_logits = hate_classifier(averaged_embedding)
            hs_loss = F.binary_cross_entropy_with_logits(hs_logits.flatten(), y_batch[:, 0], weight=w_batch[:, 0])

            # Group classifier loss
            pred_target_labels = [group_classifiers[i](averaged_embedding) \
                                    for i in range(num_target_groups)]
            target_losses = torch.stack([F.binary_cross_entropy_with_logits(pred_target_labels[i].flatten(), \
                                    y_batch[:, 1:][:, i].float(), weight = w_batch[:, 1:][:, i])#.detach() \
                                        for i in range(num_target_groups)])
            
            # Calculate average target group loss
            weighted_loss = torch.mul(target_losses, group_weights)
            group_loss = torch.mean(weighted_loss)

            # Calculate KLDiv loss for S and S~ (Imitation Learning)
            s_logits = torch.sigmoid(hate_classifier(emb_outputs))
            ln_s_logits = torch.log(torch.cat((1 - s_logits, s_logits), dim = 1))
            hs_logits = torch.sigmoid(hs_logits)
            hs_pred = torch.cat((1 - hs_logits, hs_logits), dim = 1)
            kld_loss = F.kl_div(ln_s_logits, hs_pred, reduction = 'batchmean') 

            # Semantic similarity Regulariser
            l_reg = torch.Tensor([0.0]).to(torch.device(hypernet_device))
            for index, pair in enumerate(pair_count):
                i, j = pair
                topic_sim = pair_topic_sim[index]
                w_t1, b_t1 = params[0][i], params[1][i]
                w_t2, b_t2 = params[0][j], params[1][j]
                params_t1 = torch.cat((w_t1, b_t1.unsqueeze(0)), dim = 0).flatten()
                params_t2 = torch.cat((w_t2, b_t2.unsqueeze(0)), dim = 0).flatten()
                param_sim = torch.sum(nn.CosineSimilarity(dim = 0)(params_t1, params_t2))
                l_reg = l_reg + torch.square(topic_sim - param_sim)
            l_reg = l_reg.to(torch.device(device))

            # Update hate classifier
            # print(hs_loss, group_loss, kld_loss, l_reg)
            loss_hate_speech = (hs_loss - w1 * group_loss + w2 * kld_loss + w3 * l_reg)
            loss_hate_speech.backward()

            parameters = list(hate_classifier.parameters()) + list(hypernet.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            hate_speech_optimizer.step()
            hate_speech_optimizer.zero_grad()

            acc = accuracy_score((hs_logits.flatten().float().cpu().detach().numpy() >= 0.5).astype(float), \
                                 (y_batch[:,0].float().cpu().detach().numpy() >= 0.5).astype(float))
            f1 = f1_score((hs_logits.flatten().float().cpu().detach().numpy() >= 0.5).astype(float), \
                          (y_batch[:,0].float().cpu().detach().numpy() >= 0.5).astype(float), zero_division= 0)
            
            tk0.set_postfix(hs_loss = hs_loss.item(), group_loss = w1 * group_loss.item(), kld_loss = w2 * kld_loss.item(), \
                            l_reg = w3 * l_reg.item(), final_loss = loss_hate_speech.item(), f1 = f1, acc = acc)
            
            avg_hs_loss += hs_loss.item()
            avg_group_loss += w1 * group_loss.item()
            avg_kld_loss += w2 * kld_loss.item()
            avg_l_reg += w3 * l_reg.item()
            avg_loss_hate_speech += loss_hate_speech.item()
            count += 1

        logging.debug("============== Train Loss ==============")
        logging.debug("avg Losses hs:%s", np.round(avg_hs_loss / count, 5))
        logging.debug("avg Losses gp:%s", np.round(avg_group_loss / count, 5))
        logging.debug("avg Losses kld:%s", np.round(avg_kld_loss / count, 5))
        logging.debug("avg Losses l_reg:%s", np.round(avg_l_reg / count, 5))
        logging.debug("avg Losses Last:%s\n", np.round(avg_loss_hate_speech / count, 5))
        logging.debug("LAST Losses Last:%s\n", np.round(loss_hate_speech.item(),5))

        logging.debug("============== Hate Val validation ==============")
        val_res = hate_valid(embedding_model, hypernet, hate_classifier, group_classifiers, num_target_groups, val_df, tokenizer, \
                   device, hypernet_device, batch_size, filter_layer_shapes, strategy, topic_embs, IDENTITY_COLUMNS)
        val_acc, val_f1, val_nFPED, val_nFNED, harmonic_fairness, val_recall, val_precision, val_auc, val_binary_auc = val_res

        # harmonic_fairness = (2 * val_nFPED * val_nFNED) / (val_nFPED + val_nFNED)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            stop_count = 0

        if harmonic_fairness < best_fairness:
            best_fairness = harmonic_fairness
            stop_count = 0
            
            logging.debug("============== Valdidating on Best fairness - Hate Test Unseen ==============")
            test_acc, test_f1, test_nFPED, test_nFNED, best_test_fairness, test_recall, test_precision, test_auc, test_binary_auc = hate_valid(embedding_model, hypernet, hate_classifier, group_classifiers, num_target_groups, test_df, tokenizer, \
                   device, hypernet_device, batch_size, filter_layer_shapes, strategy, topic_embs, TEST_UNSEEN_IDENTITY)
            logging.debug("Test_acc: %s, Test_f1: %s, Test_nFPED: %s, Test_nFNED: %s, Test_Fairness: %s, Test_recall: %s, Test_precision: %s, Test_auc: %s", \
                              test_acc, test_f1, test_nFPED, test_nFNED, best_test_fairness, test_recall, test_precision, test_auc)
        else:
            # Early stopping condition
            if (val_f1 - best_val_f1 < -0.05) & (stop_count >= EARLY_STOPPING_COUNT):
                logging.debug("============== Valdidating Early stopping - Hate Test Unseen ==============")
                test_acc, test_f1, test_nFPED, test_nFNED, test_fairness, test_recall, test_precision, test_auc, test_binary_auc = \
                    hate_valid(embedding_model, hypernet, hate_classifier, group_classifiers, num_target_groups, test_df, tokenizer, \
                   device, hypernet_device, batch_size, filter_layer_shapes, strategy, topic_embs, TEST_UNSEEN_IDENTITY)
                logging.debug("Test_acc: %s, Test_f1: %s, Test_nFPED: %s, Test_nFNED: %s, Test_Fairness: %s, Test_recall: %s, Test_precision: %s, Test_auc: %s", \
                              test_acc, test_f1, test_nFPED, test_nFNED, test_fairness, test_recall, test_precision, test_auc)
                
                logging.debug("Early Stopping")
                early_stop = True
                break
            
            stop_count += 1
    
    logging.debug("============== Hate Test Unseen ==============")
    hate_valid(embedding_model, hypernet, hate_classifier, group_classifiers, num_target_groups, test_df, tokenizer, \
                   device, hypernet_device, batch_size, filter_layer_shapes, strategy, topic_embs, TEST_UNSEEN_IDENTITY)
    
    logging.debug("=========== Completed HC Training ===========\n")
    
    return hate_classifier, hypernet, group_classifiers, hate_speech_optimizer, best_fairness, best_val_f1, stop_count, early_stop, val_res

def train_getfair_module(X, train, train_df, val_df, test_df, embedding_model, hypernet, hate_classifier, group_classifiers, \
                            group_emb_optimizer, group_optimizer, hate_speech_optimizer, num_target_groups, tokenizer, \
                                batch_size, device, hypernet_device, filter_layer_shapes, topic_embedding_path, group_weights, \
                                    best_fairness, best_val_f1, stop_count, k = 0, L = 1, strategy = "early fusion", param_config = None):
    """
    Training the above components adversarially: 
    1. Hate speech classification training: Maximise loss of target group classifiers, Minimise loss of Hate speech classifier
    2. Target discriminator training: Minimise loss of target group classifiers
    """
    early_stop = False
    for param in embedding_model.parameters():
        param.requires_grad = False # freeze base bert
    embedding_model.train(False)
    hate_classifier.train()
    hypernet.train()
    for i in range(num_target_groups):
        group_classifiers[i].train()
    
    topic_embs = get_topic_embeddings(topic_embedding_path, IDENTITY_COLUMNS).to(torch.device(hypernet_device))
    cossim = nn.CosineSimilarity(dim = 0)
    pair_count = [(i, j) for i in range(len(topic_embs)) for j in range(i + 1, len(topic_embs))]
    pair_topic_sim = [cossim(topic_embs[i], topic_embs[j]) for i, j in pair_count]

    hate_classifier, hypernet, group_classifiers, hate_speech_optimizer, \
        best_fairness, best_val_f1, stop_count, early_stop, last_res = \
            hate_component(L, X, train, hate_classifier, hypernet, group_classifiers, tokenizer, embedding_model, 
                            topic_embs, hate_speech_optimizer, pair_count, pair_topic_sim, val_df, test_df, \
                            batch_size, device, hypernet_device, filter_layer_shapes, strategy, num_target_groups, \
                            group_weights, best_fairness, best_val_f1, stop_count, param_config = param_config)
    if not early_stop:
        torch.cuda.empty_cache()
        hate_classifier, hypernet, group_classifiers, group_optimizer = \
            group_component(k, X, train, hate_classifier, hypernet, group_classifiers, tokenizer, embedding_model, \
                            topic_embs, group_optimizer, batch_size, device, \
                            hypernet_device, filter_layer_shapes, strategy, num_target_groups)
    
    return embedding_model, hypernet, group_classifiers, hate_classifier, \
            group_emb_optimizer, group_optimizer, hate_speech_optimizer, \
                best_fairness, best_val_f1, stop_count, early_stop, last_res

def main():
    set_seed(SEED)

    if EMB_MODEL == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
    elif EMB_MODEL == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    else:
        tokenizer = AutoTokenizer.from_pretrained('./bert_uncased_L_12_H_256',local_files_only=True)

    # Read from existing files & preprocess data (binarize target, prepare dataloader)
    train_df = pd.read_csv(DATA_PATHS["train"])
    val_df = pd.read_csv(DATA_PATHS["val"])
    test_df = pd.read_csv(DATA_PATHS["test"])

    train_val_df = train_df.copy()
    train_df['binary_target']  = binarize_target(train_df, TARGET)
    test_df['binary_target']  = binarize_target(test_df, TARGET)
    val_df['binary_target']  = binarize_target(val_df, TARGET)
    sentences = sentences = preprocess(train_df[TEXT_FIELD].astype(str).fillna("DUMMY_VALUE")).values 
    train_df = train_df.fillna(0)
    train_df = train_df.drop([TEXT_FIELD],axis=1)
    
    X = sentences
    y = train_df[Y_COLUMNS].values
    y = torch.tensor(y,dtype=torch.float32)
    # Inverse freq weights -> all samples have same set of weights (#samp, #y_columns)
    weights_tensor = calculate_weights(train_df, Y_COLUMNS)
    group_weights = torch.exp(torch.tensor([y[:, 1:][:, i].float().sum() / len(y) for i in range(NUM_TARGET_GROUPS)])).to(device)
    train = torch.utils.data.TensorDataset(torch.arange(len(X)), torch.tensor(y,dtype=torch.float32), weights_tensor)

    """""""""""""""""""""""""""""
    Start Training
    """""""""""""""""""""""""""""   
    # Load checkpoint
    if LOAD_EMB:
        emb_checkpoint = torch.load(FINE_TUNED_BERT)
    else:
        emb_checkpoint = None
    
    logging.debug("\nUsing: %s\n", STRATEGY)
    logging.debug("Epoch Status: %s\n", (EMBEDDING_EPOCH, GC_K, HC_L, GC_WARM_UP, MODULE_EPOCH, EARLY_STOPPING_COUNT))
    
    embedding_model, hypernet, group_classifiers, hate_classifier = \
        initialise_models(NUM_TARGET_GROUPS, FILTER_LAYER_SHAPE, DISCRIMINATOR_LAYERS, HATE_CLASSIFIER_LAYERS, \
                        device, hypernet_device, HYPERNERT_LAYERS, emb_checkpoint=emb_checkpoint, rank_k = HYPERPARAMETERS["rank_k"])
        
    embedding_update_optimizer, group_emb_optimizer, group_optimizer, hate_speech_optimizer = \
        initialise_optimizers(embedding_model, hypernet, group_classifiers, hate_classifier, lr = LEARNING_RATE, emb_checkpoint = emb_checkpoint)

    num_train_optimization_steps = int(EMBEDDING_EPOCH*len(train)/BATCH_SIZE)
    num_warmup_steps = 0.05 * num_train_optimization_steps
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(embedding_update_optimizer, num_warmup_steps=num_warmup_steps, \
                                                num_training_steps=num_train_optimization_steps)

    if not LOAD_EMB:
        embedding_model, hate_classifier, embedding_update_optimizer = \
            train_emb(X, train, train_val_df, val_df, embedding_model, hate_classifier, embedding_update_optimizer, tokenizer, \
                        scheduler, BATCH_SIZE, device, embedding_epoch=EMBEDDING_EPOCH)
        logging.debug("===== Embedding Test Validation =====")
        embedding_validation(embedding_model, hate_classifier, tokenizer, test_df, BATCH_SIZE, device, TEST_UNSEEN_IDENTITY)

        # Load checkpoint
        emb_checkpoint = torch.load(FINE_TUNED_BERT)
            
        logging.debug("\nUsing: %s\n", STRATEGY)        
        embedding_model, hypernet, group_classifiers, hate_classifier = \
            initialise_models(NUM_TARGET_GROUPS, FILTER_LAYER_SHAPE, DISCRIMINATOR_LAYERS, HATE_CLASSIFIER_LAYERS, \
                            device, hypernet_device, HYPERNERT_LAYERS, emb_checkpoint=emb_checkpoint)
            
        embedding_update_optimizer, group_emb_optimizer, group_optimizer, hate_speech_optimizer = \
            initialise_optimizers(embedding_model, hypernet, group_classifiers, hate_classifier, emb_checkpoint = emb_checkpoint)
        
    logging.debug("\nUsing Saved Model\n")
    logging.debug("===== Embedding Val Validation =====")
    embedding_validation(embedding_model, hate_classifier, tokenizer, val_df, BATCH_SIZE, device, IDENTITY_COLUMNS)
    logging.debug("===== Embedding Test Unseen =====")
    embedding_validation(embedding_model, hate_classifier, tokenizer, test_df, BATCH_SIZE, device, TEST_UNSEEN_IDENTITY)

    # Optional warm-up of the target-group classifiers
    for param in embedding_model.parameters():
        param.requires_grad = False # freeze base bert
    embedding_model.train(False)
    hate_classifier.train()
    hypernet.train()
    for i in range(NUM_TARGET_GROUPS):
        group_classifiers[i].train()
    
    logging.debug(f"===== Warm Up group classifiers =====\n")
    topic_embs = get_topic_embeddings(TOPIC_EMBEDDINGS_PATH, IDENTITY_COLUMNS).to(torch.device(hypernet_device))

    hate_classifier, hypernet, group_classifiers, group_optimizer = group_component(GC_WARM_UP, X, train, hate_classifier, hypernet, \
                                                                                        group_classifiers, tokenizer, embedding_model, \
                                                                                        topic_embs, group_optimizer,\
                                                                                        BATCH_SIZE, device, hypernet_device, FILTER_LAYER_SHAPE, \
                                                                                            STRATEGY, NUM_TARGET_GROUPS)
    
    # Train GETFAIR iteratively
    best_fairness = np.inf
    best_val_f1 = 0
    stop_count = 0
    early_stop = False
    for i in range(MODULE_EPOCH):
        if not early_stop:
            logging.debug(f"===== Train FairEmbed Epoch {i} =====\n")
            embedding_model, hypernet, group_classifiers, hate_classifier, \
                group_emb_optimizer, group_optimizer, hate_speech_optimizer, \
                    best_fairness, best_val_f1, stop_count, early_stop, last_res = \
                    train_getfair_module(X, train, train_val_df, val_df, test_df, embedding_model, hypernet, hate_classifier, \
                                        group_classifiers, group_emb_optimizer, group_optimizer, hate_speech_optimizer, NUM_TARGET_GROUPS, \
                                            tokenizer, BATCH_SIZE, device, hypernet_device, FILTER_LAYER_SHAPE, TOPIC_EMBEDDINGS_PATH, group_weights, \
                                            best_fairness, best_val_f1, stop_count, k = GC_K, L = HC_L, strategy = STRATEGY, param_config = HYPERPARAMETERS)

    if not early_stop:
        val_results = last_res
        test_acc, test_f1, test_nFPED, test_nFNED, test_fairness, test_recall, test_precision, test_auc, _ = \
            hate_valid(embedding_model, hypernet, hate_classifier, group_classifiers, NUM_TARGET_GROUPS, test_df, tokenizer, \
            device, hypernet_device, BATCH_SIZE, FILTER_LAYER_SHAPE, STRATEGY, topic_embs, TEST_UNSEEN_IDENTITY)
        
        logging.debug("STOP_Test_acc: %s, STOP_Test_f1: %s, STOP_Test_nFPED: %s, STOP_TEST_nFNED: %s, STOP_Test_Fairness: %s, STOP_Test_recall: %s, STOP_Test_precision: %s, STOP_Test_auc: %s", \
                              test_acc, test_f1, test_nFPED, test_nFNED, test_fairness, test_recall, test_precision, test_auc)

        torch.save({
            'iteration': "END",
            'embedding_model_state_dict': embedding_model.module.state_dict(),
            'hypernet_state_dict': hypernet.module.state_dict(),
            'hate_classifer_state_dict': hate_classifier.module.state_dict(),
            'hate_speech_optimizer_state_dict': hate_speech_optimizer.state_dict(),
            }, "./STOP_" + SAVE_HATESPEECH_MODEL)
        
    logging.debug(f"===== Complete ALL Training ====")

if __name__ == "__main__":
    main()

