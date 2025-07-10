import os
import json
import torch
import argparse
from tqdm import tqdm
from typing import Sequence
from peft import AutoPeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
import pandas as pd

def to_pickle(df, f):
    with open(f, 'wb') as fname:
        pickle.dump(df, fname)

def open_pickle(f):
    with open(f, 'rb') as file:
        data = pickle.load(file)
    return data

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument('--is_with_rationale', type=bool, default=False)
    parser.add_argument('--is_lora', type=bool, default=False)
    parser.add_argument('--code_embeds_path', type=str, default=None)
    args = parser.parse_args()
    return args


def inference_on_one(input_str: Sequence[str], model, tokenizer) -> str:
    model_inputs = tokenizer(
      input_str,
      return_tensors='pt',
      padding=True,
    )

    topk_output = model.generate(
        model_inputs.input_ids.cuda(),
        max_new_tokens=1000,
        top_k=50
    )
    output_str = tokenizer.batch_decode(topk_output)  # a list containing just one str

    return output_str[0]

def read_jsonl(filepath: str, is_with_rationale):
    """Load a .jsonl file into a dictionary."""
    src_dict_ls = []
    lang = os.path.basename(filepath).split(".")[0]
    with open(filepath, "r") as f:
        for line in f:
            src_dict = json.loads(line)
            src_dict["lang"] = lang
            src_dict_ls.append(src_dict)
            
    res_dict_ls = []
    for src_dict in src_dict_ls:
        question = src_dict["question"]
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
        if is_with_rationale:
            tmp = {
                "instruction" : f"You're a {lang} doctor, kindly address the medical queries according to the patient's account in {lang}. Let’s solve this step-by-step.  You should first give the reason in {lang} for your choice. Then you should give the right answer index of the question.",
                "input":f"###Question: {question} Which of the following is the best treatment for this patient? ###Options: {options}",
                "output":f"{answer_id}",
                "rationale":f"{rationale}"
            }    
        else:
            tmp = {
                "instruction" : f"You're a {lang} doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly.",
                "input":f"###Question: {question} Which of the following is the best treatment for this patient? ###Options: {options}",
                "output":f"{answer_id}",
                "rationale":f"{rationale}"
            }    
        res_dict_ls.append(tmp)
        
    return res_dict_ls


def prepare_data(data_list: Sequence[dict]) -> Sequence[dict]:
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    for _idx in tqdm(range(len(data_list))):
        data_entry = data_list[_idx]
        data_list[_idx]['sample_id'] = _idx

        data_list[_idx]['pmc_input'] = prompt_input.format_map(data_entry) if data_entry.get("input", "") != "" else prompt_no_input.format_map(data_entry)
        data_list[_idx]['pmc_output'] = data_entry['output']
        data_list[_idx]['rationale'] = data_entry['rationale']
    return data_list

def inference(test_filepath, model, tokenizer, save_dir, is_with_rationale):
    data_list = read_jsonl(test_filepath, is_with_rationale)
    data_list = prepare_data(data_list)
    answers = []
    for _idx in tqdm(range(len(data_list))):
        data_entry = data_list[_idx]
        sample_id = data_entry['sample_id']
        input_str = [
            data_entry['pmc_input']
        ]
        output_str = inference_on_one(input_str, model, tokenizer)
        response = output_str.split("### Response:")[1].strip()
        answers.append((response.replace("\n", ""), data_entry['pmc_output'], data_entry['rationale']))
        
    with open(save_dir + "/" +test_filepath.split("/")[1]+"_res.txt", "w", encoding="utf-8") as fp:
        for response, target, rationale in answers:
            fp.write(f"{response}[SPLIT]{target}[SPLIT]{rationale}\n")
            
def validate():
    args = parse_args()
    filepaths = [os.path.join(args.data_path, filename) for filename in os.listdir(args.data_path) if filename.endswith('.jsonl')]  

    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    config.architectures = ["CustomModel"]
    model = CustomModel(config=config).from_pretrained(args.model_name_or_path)
    
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=2048,
        use_fast=False,
        trust_remote_code=True
    )

    # pre-trained knowledge embeds
    code_embeds = open_pickle(args.code_embeds_path)

    node_df = pd.read_csv("/work/work_fran/bkg/data/logml_2025/connected_node_logml_df.csv", sep='\t')
    keep_codes = set(node_df.loc[node_df['ntype'] == 'ICD10CM']['node_id'].str.split(':', expand=True)[0].tolist())
    
    # add new tokens and resize
    code_tokens = [('<%s>' % x) for x in list(keep_codes)]
    tokenizer.add_tokens(code_tokens, special_tokens=True)
    model.setup(code_embeds)

    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)
        
    for idx, filepath in enumerate(filepaths):
        inference(filepath, model, tokenizer, args.save_dir, args.is_with_rationale)
        print(idx, "\t", filepath, " has done!")
    
if __name__ == '__main__':
    validate()
