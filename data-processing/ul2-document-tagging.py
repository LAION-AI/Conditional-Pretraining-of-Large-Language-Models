#!/usr/bin/env python

"""This code is a work in progress. The default is to use flan-ul2
however this same code is compatible with any of the flan-T5 models
available on the huggingface hub. Use whatever is the largest model
that your hardware can support. Having good labels is important for
good quality conditional pretraining examples."""

import sys
import time
import math
import re
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

inputfile = sys.argv[1]
device_id = int(sys.argv[2])
outfile = inputfile + ".out"

device_map = {
    'shared': device_id,
    'lm_head': device_id,
    'encoder': device_id,
    'decoder': device_id,
    'decoder.final_layer_norm': device_id,
    'decoder.dropout': device_id
}

tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-ul2",
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
)

with open(inputfile) as my_file:
    data = my_file.read()

paragraphs = data.split("")

excluded_list = {"label", "tag", "tags", "chapter", "labels", "story", "stories", "category", "movie", "movies"}

def Sort_Tuple(tup: List[Tuple]) -> List[Tuple]:
    tup.sort(key=lambda x: x[1], reverse=True)
    return tup

def ask_UL2(input_text: List[str]) -> List[Tuple[str, float]]:
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        input_ids=inputs["input_ids"].to(device_id),
        attention_mask=inputs["attention_mask"].to(device_id),
        do_sample=True,
        top_p=0.95,
        eos_token_id=1,
        max_length=tokenizer.model_max_length,
        bos_token_id=0,
        temperature=0.9,
        return_dict_in_generate=True,
        output_scores=True,
    )

    output_tuple = []
    probs = torch.stack(outputs.scores, dim=1).softmax(-1)
    for z, i in enumerate(outputs.sequences):
        out_text = tokenizer.decode(i, skip_special_tokens=True)
        logprobs = 0
        counter = 0
        if 1 in i[1:]:
            for k in i[1:]:
                word_prob = (round(probs[z][counter][k.item()].item(), 2)) + 0.001
                logprobs += math.log(word_prob)
                counter += 1
            out_tuple = (out_text, round((logprobs / (counter + 1)), 2))
            output_tuple.append(out_tuple)
        else:
            pass

    return output_tuple

def generate_topic(paragraph: str) -> List[Tuple[str, float]]:
    results = []
    length = int(len(paragraph) / 2)
    length2 = int(length / 2)

    para1 = paragraph[0:length]
    para2 = paragraph[(length - length2):(2 * length - length2)]
    para3 = paragraph[length:(2 * length)]

    toklens = []
    input_1 = (
        "Task: Generate a theme or category label for the provided document.\nDocument:\n" + para1 + "\nTheme: "
    )
        toklens.append(len(tokenizer.encode(input_1)))
    input_2 = (
        "Task: Generate a theme or category label for the provided document.\nDocument:\n" + para2 + "\nTheme: "
    )
    toklens.append(len(tokenizer.encode(input_2)))
    input_3 = (
        "Task: Generate a theme or category label for the provided document.\nDocument:\n" + para3 + "\nTheme: "
    )
    toklens.append(len(tokenizer.encode(input_3)))

    minlen = min(toklens)
    if minlen > 800:
        minlen = 800

    input_1f = tokenizer.decode(tokenizer.encode(input_1)[0:minlen])
    input_2f = tokenizer.decode(tokenizer.encode(input_2)[0:minlen])
    input_3f = tokenizer.decode(tokenizer.encode(input_3)[0:minlen])

    inputs = []
    for k in range(0, 12):
        inputs.append(input_1f)
    for k in range(0, 8):
        inputs.append(input_2f)
    for k in range(0, 4):
        inputs.append(input_3f)

    result_tuple = ask_UL2(inputs)
    for result in result_tuple:
        if result[1] > -5:
            if len(re.findall(r"\d{2}", result[0])) > 0 or len(re.findall(r'[A-Z]{4}', result[0])) > 0 or len(re.findall(r"\W", result[0].replace(" ","").replace(")","").replace("(","").replace(".","").replace("'",""))) > 0:
                pass
            else:
                results.append(result)

    if len(results) < 1:
        results.append(("no_topic", -50))
        return results
    else:
        sorted_results = Sort_Tuple(results)
        return sorted_results[0:16]

def dedup_list(toplist: List[Tuple[str, float]]) -> List[str]:
    temptops = set()
    for topic in toplist:
        if topic[1] > -6:
            item = topic[0].lower().replace(": ", ":").replace("label:", "").replace("category:", "")
            if len(item.split(" ")) > 3:
                pass
            else:
                temptops.add(item)
    return list(temptops)

def order_topics(combined_topics_lists: List[List[str]]) -> List[str]:
    strings_list = []
    for i in combined_topics_lists:
        for k in i:
            if k in excluded_list or len(k) < 3:
                pass
            else:
                strings_list.append(k)
    
    frequency_dict = {}
    for string in strings_list:
        if string in frequency_dict:
            frequency_dict[string] += 1
        else:
            frequency_dict[string] = 1

    sorted_strings = sorted(frequency_dict.keys(), key=lambda x: frequency_dict[x], reverse=True)

    return sorted_strings

start_time = time.perf_counter()
questions_dict = {}
uniq_id = 100000
for paragraph in paragraphs:
    start_time = time.perf_counter()
    if paragraph[0] == "\n":
        paragraph = paragraph[1:]
        if paragraph[0] == "\n":
            paragraph = paragraph[1:]
    print("-" * 8)
    print(paragraph[0:40])
    topic1a = dedup_list(generate_topic(paragraph))
    ordered_tops = order_topics([topic1a])
        newstring = "^-^-^-^-^-^-^\n[ "
    for topic in ordered_tops:
        newstring += topic + ", "
    finalstring = newstring[:-2] + "]"
    print(finalstring, file=open(outfile, "a"))
    print(paragraph, file=open(outfile, "a"))
    stop_time = time.perf_counter()
    generation_time = stop_time - start_time
    print("Generation time:", generation_time)


