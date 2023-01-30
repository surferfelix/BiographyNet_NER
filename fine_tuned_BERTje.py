"""This model is a finetuned version of the BERTje model from wietsedv for named entity recognition on biographical texts
It should be able to detect PER and LOC ner labels

The finetuning code was largely adapted from the following notebook: https://github.com/cltl/ma-ml4nlp-labs/blob/main/code/bert4ner/bert_finetuner.ipynb"""
import random, time, os
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import logging, sys
from transformers import BertTokenizer
import pandas as pd
# Our code behind the scenes!
import BERTje_utils as utils
import argparse

class Read_Data_as_df():
    '''Will read the path to the data and process it as a pd dataframe'''
    def __init__(self, data):
        self.data = data
    
    def process(self): # TODO Need to get it to 
        d = pd.read_csv(self.data, delimiter = "\t", encoding = 'unicode_escape', names = ['token', 'gold'])
        # print(d.head())

class FineTune_On_Dataframe():
    """Finetunes pre-trained BERTje on dataframe data"""
    
    def __init__(self, train_path: str, eval_path: str, epochs=12, batch_size=8, model_path = 'GroNLP/bert-base-dutch-cased'):
        '''We initialize all hyperparameters here'''
        # TODO Maybe move these variables to a main, since it would be cleaner
        print(f'Model set to {model_path}')
        self.epochs = epochs
        self.model_name = model_path
        self.gpu_run_ix = 0
        self.seed_val = 1234500 # For reproducability
        self.seq_max_len = 256
        self.print_info_every = 10
        self.gradient_clip = 1.0
        self.learning_rate = 1e-5
        self.batch_size = batch_size
        self.train_data_path = train_path
        self.dev_data_path = eval_path
        self.save_model_dir = "saved_models"
        self.LABELS_FILENAME = f"{self.save_model_dir}/label2index.json"
        self.LOSS_TRN_FILENAME = f"{self.save_model_dir}/Losses_Train_{self.epochs}.json"
        self.LOSS_DEV_FILENAME = f"{self.save_model_dir}/Losses_Dev_{self.epochs}.json"
        self.PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index # -100
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)


        # Initialize Random seeds and validate if there's a GPU available...
        self.device, self.USE_CUDA = utils.get_torch_device(self.gpu_run_ix)
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)

        console_hdlr = logging.StreamHandler(sys.stdout)
        file_hdlr = logging.FileHandler(filename=f"{self.save_model_dir}/BERT_TokenClassifier_train_{self.epochs}.log")
        logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
        logging.info("Start Logging")

    def Load_Datasets(self):
        tokenizer = BertTokenizer.from_pretrained(self.model_name, do_basic_tokenize=False)

        # Load Train Dataset
        train_data, train_labels, train_label2index = utils.read_conll(self.train_data_path, has_labels=True)
        train_inputs, train_masks, train_labels, seq_lengths = utils.data_to_tensors(train_data, 
                                                                                    tokenizer, 
                                                                                    max_len=self.seq_max_len, 
                                                                                    labels=train_labels, 
                                                                                    label2index=train_label2index,
                                                                                    pad_token_label_id=self.PAD_TOKEN_LABEL_ID)
        utils.save_label_dict(train_label2index, filename=self.LABELS_FILENAME)
        index2label = {v: k for k, v in train_label2index.items()}

    
        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        # Load Dev Dataset
        dev_data, dev_labels, _ = utils.read_conll(self.dev_data_path, has_labels=True)
        dev_inputs, dev_masks, dev_labels, dev_lens = utils.data_to_tensors(dev_data, 
                                                                            tokenizer, 
                                                                            max_len=self.seq_max_len, 
                                                                            labels=dev_labels, 
                                                                            label2index=train_label2index,
                                                                            pad_token_label_id=self.PAD_TOKEN_LABEL_ID)

        # Create the DataLoader for our Development set.
        dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels, dev_lens)
        dev_sampler = RandomSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=self.batch_size)
        return train_label2index, index2label, train_dataloader, dev_dataloader, tokenizer

    def Initialize_Model_Components(self, train_label2index, index2label, train_dataloader):
        model = BertForTokenClassification.from_pretrained(self.model_name, num_labels=len(train_label2index))
        model.config.finetuning_task = 'token-classification'
        model.config.id2label = index2label
        model.config.label2id = train_label2index
        if self.USE_CUDA: 
            model.cuda()

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * self.epochs

        # Create optimizer and the learning rate scheduler.
        optimizer = AdamW(model.parameters(), lr=self.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
        return model, optimizer, scheduler

    def Fine_Tune(self, model, train_dataloader, dev_dataloader, optimizer, scheduler, tokenizer, index2label):
        loss_trn_values, loss_dev_values = [], []

        for epoch_i in range(1, self.epochs+1):
            # Perform one full pass over the training set.
            logging.info("")
            logging.info('======== Epoch {:} / {:} ========'.format(epoch_i, self.epochs))
            logging.info('Training...')

            t0 = time.time()
            total_loss = 0
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]
                total_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)

                # Update parameters
                optimizer.step()
                scheduler.step()

                # Progress update
                if step % self.print_info_every == 0 and step != 0:
                    # Calculate elapsed time in minutes.
                    elapsed = utils.format_time(time.time() - t0)
                    # Report progress.
                    logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Loss: {}.'.format(step, len(train_dataloader),
                                                                                                    elapsed, loss.item()))

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)

            # Store the loss value for plotting the learning curve.
            loss_trn_values.append(avg_train_loss)

            logging.info("")
            logging.info("  Average training loss: {0:.4f}".format(avg_train_loss))
            logging.info("  Training Epoch took: {:}".format(utils.format_time(time.time() - t0)))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on our validation set.
            t0 = time.time()
            # This will write the results to a file
            results = utils.evaluate_bert_model(dev_dataloader, self.batch_size, model, tokenizer, index2label, self.PAD_TOKEN_LABEL_ID, prefix="Validation Set")
            loss_dev_values.append(results['loss'])
            logging.info("  Validation Loss: {0:.2f}".format(results['loss']))
            logging.info("Macro avg:  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['macro avg']['precision']*100, results['macro avg']['recall']*100, results['macro avg']['f1-score']*100))
            logging.info("Weighted avg:  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['weighted avg']['precision']*100, results['weighted avg']['recall']*100, results['weighted avg']['f1-score']*100))
            logging.info("  Validation took: {:}".format(utils.format_time(time.time() - t0)))

            # Save Checkpoint for this Epoch
            utils.save_model(f"{self.save_model_dir}/EPOCH_{epoch_i}", {"args":[]}, model, tokenizer)


        utils.save_losses(loss_trn_values, filename=self.LOSS_TRN_FILENAME)
        utils.save_losses(loss_dev_values, filename=self.LOSS_DEV_FILENAME)
        logging.info("")
        logging.info("Training complete!")


def main(train_path:str, eval_path: str, epochs: int, batch_size: int, model_path = str):
    """Executes the Fine Tuning process"""
    a = FineTune_On_Dataframe(train_path, eval_path, epochs, batch_size, model_path) # We initialize our hyperparameters
    train_label2index, index2label, train_dataloader, dev_dataloader, tokenizer = a.Load_Datasets() # We load our dev and train data
    model, optimizer, scheduler = a.Initialize_Model_Components(train_label2index, index2label, train_dataloader) # We initialize the components
    a.Fine_Tune(model, train_dataloader, dev_dataloader, optimizer, scheduler, tokenizer, index2label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type = str, help = 'The path to use for finetuning')
    parser.add_argument("eval_path", type = str, help = 'The path on which each epoch is evaluated')
    parser.add_argument("-e","--epochs", type = int, help = 'The amount of iterations to train over the whole training data on', nargs = '?', const = 1, default = 8)
    parser.add_argument("-b","--batch_size", type = int, help = 'The size of each batch that will be taken into account when finetuning BERTje', nargs = '?', const = 1, default = 4)
    parser.add_argument("-p","--model_path", type = str, help = 'Optional argument for if you want to finetune another BERT model', default = 'GroNLP/bert-base-dutch-cased', required = False)
    args = parser.parse_args()
    main(args.train_path, args.eval_path, args.epochs, args.batch_size, args.model_path)
    
    
