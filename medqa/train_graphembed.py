import os
import json
import copy
import torch
import logging
import pickle
import transformers
import pandas as pd
from random import shuffle
from transformers import Trainer, LlamaConfig, LlamaForCausalLM
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model
from typing import Dict, Optional, Sequence, List

def to_pickle(df, f):
    with open(f, 'wb') as fname:
        pickle.dump(df, fname)

def open_pickle(f):
    with open(f, 'rb') as file:
        data = pickle.load(file)
    return data

SEED = 42
transformers.set_seed(SEED)
IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    is_lora: Optional[bool] = field(default=False)
    lora_rank: Optional[int] = field(default=16)
    # target_modules: Optional[List[str]] = field(default=)
    lora_alpha: Optional[int] = field(default=32)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    code_embeds_path: str = field(default=None, metadata={"help": "Path to the pre-trained knowledge embeddings."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    use_cache : bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    gradient_clipping : float = field(
        default=None
    )

def jsonl_load(data_path):
    """Load a .jsonl file into a dictionary."""
    filepaths = [os.path.join(data_path, filename) for filename in os.listdir(data_path) if filename.endswith('.jsonl')]    
    src_dict_ls = []
    for filepath in filepaths:
        lang = os.path.basename(filepath).split(".")[0]
        with open(filepath, "r", encoding='utf-8') as f:
            for line in f:
                src_dict = json.loads(line)
                src_dict["lang"] = lang
                src_dict_ls.append(src_dict)
                        
    res_dict_ls = []
    for src_dict in src_dict_ls:
        lang = src_dict["lang"]
        question = src_dict["question"]
        options = ""
        for key in src_dict["options"].keys():
            content = src_dict["options"][key]
            options += f"{key}. {content} "
        if isinstance(src_dict["answer_idx"], str):
            answer_id = src_dict["answer_idx"]
        elif isinstance(src_dict["answer_idx"], list):
            answer_id = ",".join(src_dict["answer_idx"])

        rationale = src_dict["rationale"]
        data_with_rationale = {
            "instruction" : f"You're a {lang} doctor, kindly address the medical queries according to the patient's account in {lang}. Let’s solve this step-by-step. You should first give the reason in {lang} for your choice. Then you should give the right answer index of the question.",
            "input":f"###Question: {question} Which of the following is the best treatment for this patient? ###Options: {options}",
            "output":f"###Rationale: {rationale}\n###Answer: OPTION {answer_id} IS CORRECT."
        }    
        res_dict_ls.append(data_with_rationale)
        
        data_without_rationale = {
            "instruction" : f"You're a {lang} doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly.",
            "input":f"###Question: {question} Which of the following is the best treatment for this patient? ###Options: {options}",
            "output":f"###Answer: OPTION {answer_id} IS CORRECT."
        }    
        res_dict_ls.append(data_without_rationale) 
               
    # shuffle the data for training
    shuffle(res_dict_ls)                
    return res_dict_ls

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, annot_list=None):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        if annot_list is not None:
            logging.warning("Using annot_list as list_data_dict...")
            # add annotated data
            list_data_dict = list(annot_list)
            shuffle(list_data_dict)
        else:
            logging.warning("Using data_path as list_data_dict...")
            # add data from jsonl files
            list_data_dict = jsonl_load(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, annot_list = None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, annot_list=annot_list)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

class CustomModel(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        # initialize variables for knowledge tokens
        self.code_embeds = torch.nn.Parameter(torch.empty(0, 128))
        
        # projector layer to line up dims between emb and LLM
        self.proj_layer =  torch.nn.Linear(128, 4096, bias=True)
        self.start_idx = 0

        self.base_vocab_size = config.vocab_size

    # set the knowledge token embeddings
    def setup(self, code_embeds):
        self.code_embeds = torch.nn.Parameter(torch.tensor(code_embeds, dtype=torch.float32))
        emb = self.get_input_embeddings()
        with torch.no_grad():
            self.start_idx = len(emb.weight.data) - len(self.code_embeds)
        
    # def forward(self, input_ids, labels=None, attention_mask=None, *args, **kwargs):
                
    #     emb = self.get_input_embeddings()
    #     emb.requires_grad_()
    #     lin_emb = self.proj_layer(self.code_embeds.cuda())

    #     embed_list = []
    #     for id in input_ids[0].cuda():
    #         # knowledge tokens
    #         if id >= self.base_vocab_size:
    #             shift_id = (id - self.base_vocab_size)
    #             embed_list.append(lin_emb[shift_id])
    #         # regular tokens
    #         else:
    #             embed_list.append(emb(torch.tensor(id)))
        
    #     # append knowledge tokens to text input
    #     input_embeds = torch.stack(tuple(embed_list))

    #     # perform next token prediction                          
    #     return super().forward(inputs_embeds=input_embeds.unsqueeze(dim=0),
    #                             labels=labels,
    #                             attention_mask=attention_mask)    

    def forward(self, input_ids, labels=None, attention_mask=None, *args, **kwargs):
        # Move input_ids to the same device as the model
        input_ids = input_ids.to(self.device)

        standard_embeddings = self.get_input_embeddings()

        projected_code_embeds = self.proj_layer(self.code_embeds)

        # Create a mask for knowledge tokens (where id >= base_vocab_size)
        # This mask will be used to select which embeddings to use for each input ID.
        is_knowledge_token = (input_ids >= self.base_vocab_size)

        inputs_embeds = standard_embeddings(input_ids)

        # Calculate the shifted IDs for knowledge tokens
        # We need to ensure that the indices for projected_cui_embeds are correct.
        # For each knowledge token in input_ids, its true index into projected_cui_embeds
        # is (original_id - self.base_vocab_size).
        # We only apply this where is_knowledge_token is True.
        
        # Get the actual knowledge token IDs that are present in input_ids
        knowledge_token_ids_in_batch = input_ids[is_knowledge_token]

        if knowledge_token_ids_in_batch.numel() > 0:
            # Shift these IDs to become indices for projected_cui_embeds
            shifted_knowledge_indices = knowledge_token_ids_in_batch - self.base_vocab_size
            
            # Use advanced indexing to replace the embeddings efficiently
            # inputs_embeds[is_knowledge_token] selects the positions where knowledge tokens are
            # projected_cui_embeds[shifted_knowledge_indices] selects the corresponding projected embeddings
            inputs_embeds[is_knowledge_token] = projected_code_embeds[shifted_knowledge_indices]


        # Perform next token prediction using the modified input embeddings
        return super().forward(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
        )

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
    config.architectures = ["CustomModel"]
    model = CustomModel(config=config).from_pretrained(model_args.model_name_or_path)

    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     trust_remote_code=True,
    # )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    if model_args.is_lora:
        
        config = LoraConfig(
            r = model_args.lora_rank,
            lora_alpha = model_args.lora_alpha,
            target_modules = ["q_proj", "v_proj", "proj_layer"],
            lora_dropout = 0.05,
            bias = 'none',
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, config)

        print(model.print_trainable_parameters())
    
    # pre-trained knowledge embeds
    code_embeds = open_pickle(data_args.code_embeds_path)

    node_df = pd.read_csv("/work/work_fran/bkg/data/logml_2025/connected_node_logml_df.csv", sep='\t')
    keep_codes = set(node_df.loc[node_df['ntype'] == 'ICD10CM']['node_id'].str.split(':', expand=True)[0].tolist())

    # annotated qa's
    annot_list = open_pickle("/work/work_fran/bkg/data/medqa_annot.pkl")
    
    # add new tokens and resize
    code_tokens = [('<%s>' % x) for x in list(keep_codes)]
    tokenizer.add_tokens(code_tokens, special_tokens=True)
    model.setup(code_embeds)
            
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, annot_list=annot_list)
    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
