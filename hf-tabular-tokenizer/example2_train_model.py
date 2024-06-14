############################################################################
##
## Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
##
## NVIDIA Sample Code
##
## Please refer to the NVIDIA end user license agreement (EULA) associated
## with this source code for terms and conditions that govern your use of
## this software. Any use, reproduction, disclosure, or distribution of
## this software and related documentation outside the terms of the EULA
## is strictly prohibited.
##
###########################################################################

from argparse import ArgumentParser
from collections import namedtuple
import math
from pathlib import Path
import platform

from datasets import load_dataset
import torch
from transformers import (DataCollatorForLanguageModeling, EarlyStoppingCallback, GPTJConfig, GPTJForCausalLM,
                          TrainingArguments, Trainer)

from coder.hf_tabular_tokenizer import HFTabularTokenizer
from utils.logger import create_logger

logger = create_logger(__name__)


def tokenize_function(text: str):
    """
    Function used by tokenized_datasets to create custom tabular tokenized dataset.
    Args:
        text: string to be tokenized by tabular_tokenizer

    Returns:
        encoded string
    """
    return tabular_tokenizer.encode(text)


def tokenized_datasets(example: dict) -> dict:
    """
    takes a list of token_ids and creates a item for causal LM.
    """
    token_ids = tokenize_function(example['text'][0])

    return {'input_ids': [token_ids],
            'attention_mask': [[1 for _ in range(len(token_ids))]],
            'labels': [token_ids],
            }


if __name__ == '__main__':
    parser = ArgumentParser(description='Example training of a transformers model with the Tabular Tokenizer.')
    parser.add_argument('--vocab-file', type=str, default='example_vocab.pkl', help='path to the vocab.pkl file')
    parser.add_argument('--seqlen', type=int, default=512, help='Sequence length of the tokenized documents and model')
    parser.add_argument('--train', type=str, default='train.json', help='path to train.json')
    parser.add_argument('--valid', type=str, default='valid.json', help='path to valid.json')
    parser.add_argument('--test', type=str, default='test.json', help='path to test.json')

    args = parser.parse_args()
    # some simple checks
    assert Path(args.vocab_file).exists()
    assert args.seqlen > 0
    assert Path(args.train).exists() and Path(args.valid).exists() and Path(args.test).exists()

    logger.info(f'Instantiating the HF Tabular Tokenizer')
    Constants = namedtuple('constants', ['END_OF_TEXT', 'NEW_LINE', 'DELIMITER'])
    CONSTS = Constants(END_OF_TEXT='<|endoftext|>', NEW_LINE='\n', DELIMITER=',')
    tabular_tokenizer = HFTabularTokenizer(args.vocab_file,
                                           special_tokens=[CONSTS.NEW_LINE, CONSTS.END_OF_TEXT],
                                           delimiter=CONSTS.DELIMITER
                                           )
    TOKENS_PER_ROW = sum(tabular_tokenizer.code_column.sizes) + 1

    logger.info('loadind the datasets')
    ds = load_dataset('json', data_files={'train': args.train,
                                          'valid': args.valid,
                                          'test': args.test},
                      num_proc=3)
    logger.info('Tokenizing the loaded dataset with the tabular tokenizer...')
    mapped_ds = ds.map(tokenized_datasets, batch_size=1, batched=True, remove_columns=["text"], num_proc=24)

    # model config
    logger.info('Training a very small model as an example. Goal is NOT to make the most accurate model...')
    model_config = GPTJConfig(vocab_size=tabular_tokenizer.vocab_size,
                              n_positions=args.seqlen,
                              n_layer=2,
                              n_head=1,  # num attn heads
                              n_embd=64,  # hidden size
                              # n_inner=256,  # intermediate size
                              bos_token_id=tabular_tokenizer.bos_id,
                              eos_token_id=tabular_tokenizer.eos_id,
                              )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPTJForCausalLM(model_config).to(device)

    training_args = TrainingArguments(
            output_dir="./experiments",
            optim='adamw_hf' if device == 'cpu' else 'adamw_torch_fused',
            learning_rate=1e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=1,
            num_train_epochs=1,
            bf16=False if device == 'cpu' else True,
            bf16_full_eval=False if device == 'cpu' else True,
            logging_first_step=True,
            logging_steps=10,
            report_to=['tensorboard'],
            load_best_model_at_end=True,
            save_strategy='steps',
            save_steps=10,
            evaluation_strategy='steps',
            eval_steps=10,
            save_total_limit=3,
            lr_scheduler_type='cosine',
            gradient_checkpointing=True,
            use_mps_device=True if device == 'cpu' else False,  # i.e. for appMacOS
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tabular_tokenizer, mlm=False, )
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=mapped_ds["train"],
            eval_dataset=mapped_ds["valid"],
            tokenizer=tabular_tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.03)]
    )

    trainer.train()


    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    if platform.system() == 'Darwin':  # MacOS test before training on a big cluster, training with Trainer on MacOS may have happened on an MPS device
        device = 'cpu'
        model = model.to(device)
    inputs = torch.tensor(mapped_ds['test']['input_ids'], dtype=torch.int32, device=device)[:, :-TOKENS_PER_ROW]
    logger.info('Example Conditional Generation of data: \n')
    generated_tokens = model.generate(inputs, max_new_tokens=TOKENS_PER_ROW)[0][-TOKENS_PER_ROW:].to('cpu').numpy().tolist()
    logger.info(generated_tokens)

    logger.info('Success!')

    try:
        logger.info("Try decoding. This won't work if the model hasn't been trained well enough")
        logger.info(tabular_tokenizer.decode(generated_tokens))
    except ValueError as exception:
        logger.error(exception)
        logger.error('Model needs to be trained longer or with better parameters.')
        logger.error(f"Here we show decoding the test set instead: {tabular_tokenizer.decode(mapped_ds['test']['input_ids'])[0]}")

    logger.info('Complete.')
