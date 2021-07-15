#
# Work based on prevoius research that can be found on my personal repository.
#
# AI-MuG: https://github.com/ehcalabres/ai-mug
#

# +-------------------+
# | Utility imports   |
# +-------------------+

import regex as re
import numpy as np

# +----------------------------+
# | Transformers (HF) imports  |
# +----------------------------+

import argparse

import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline

class GPT2MuGModel:
    def __init__(self, config) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['checkpoint'])
        self.model = AutoModelForCausalLM.from_pretrained(config['checkpoint'])

    def __call__(self, start_sequence='X:1\n', n_examples=2, max_length=300):
        music_compositor = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, config={'max_length': max_length})

        new_composition = music_compositor(start_sequence, max_length=max_length, num_return_sequences=n_examples)

        print("Compositions generated:")
        for element in new_composition:
            print("----------------------")
            print(element['generated_text'])
            print("----------------------")
    
    def train(self):

        print('Importing, preprocessing and creating dataset...')
        train_dataset = TextDataset(
            file_path=self.config['dataset']['train'],
            tokenizer=self.tokenizer,
            block_size=128
        )

        test_dataset = TextDataset(
            file_path=self.config['dataset']['test'],
            tokenizer=self.tokenizer,
            block_size=128
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        training_arguments = TrainingArguments(
            output_dir=self.config['save_directory'],
            overwrite_output_dir=True,
            num_train_epochs=self.config['num_train_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            eval_steps = 400,
            save_steps=800,
            warmup_steps=500,
            prediction_loss_only=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_arguments,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )

        print('Model training...')
        trainer.train()

        if self.config['save']:
            print('Saving model...')
            trainer.save_model(self.config['save_directory'])

        print('Training process completed succesfuly.')


def main():

    parser = argparse.ArgumentParser(description="Example of training a HF model with accelerator")
    parser.add_argument("--fp16", type=bool, default=False, help="If passed, will use FP16 training.")
    parser.add_argument("--cpu", type=bool, default=False, help="If passed, will train on the CPU.")
    parser.add_argument("--save", type=bool, default=False, help="If passed, the model will be saved after the training process.")
    parser.add_argument("--directory", type=str, default="./model_saved", help="Path to directory where model will be saved.")
    parser.add_argument("--inference", type=bool, default=False, help="If passed, will try to generate a new sample with the text provided with the parameter --inference-init (default='X:1')")
    parser.add_argument("--inference-text", type=str, default="X:1", help="Initial text from where the model will start generating.")

    args = parser.parse_args()

    config = {
        'checkpoint': 'distilgpt2', 
        'dataset': {
            'train': '../data/abc_dataset/irish_music - original.abc',
            'test': '../data/abc_dataset/irish_music_test.abc'},
        'lr': 5e-5, 
        'num_epochs': 10, 
        'seed': 42, 
        'batch_size': 4
    }

    model = GPT2MuGModel(config)

    if not args.inference:
        model.train()
    else:
        model(start_sequence=args.inference_text)

if __name__ == '__main__':
    transformers.logging.set_verbosity_warning()
    datasets.logging.set_verbosity_warning()
    main()
