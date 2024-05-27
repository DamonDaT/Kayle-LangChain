from dotenv import load_dotenv

import torch
import transformers

from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_core.prompts import PromptTemplate

from langchain_huggingface.llms import HuggingFacePipeline

# Initialize environment variables from .env file
load_dotenv(verbose=True)

"""
    CASE 1:
        Use HF Transformers to call Llama2 (Llama-2-7b-chat-hf) model.
"""

# LLM dir
llm_dir = r'/home/dateng/model/huggingface/meta-llama/Llama-2-7b-chat-hf'

# LLM
llm = AutoModelForCausalLM.from_pretrained(llm_dir, device_map='auto', torch_dtype=torch.float16)
# Tokenizer
llm_tokenizer = AutoTokenizer.from_pretrained(llm_dir)

# Tokenize
input_ids = llm_tokenizer('简单介绍下马斯克的时间拳法', return_tensors="pt").to('cuda')
# Generate
outputs = llm.generate(input_ids['input_ids'], max_length=1000)
# Decode
result_1 = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result_1)

"""
    CASE 2:
        Use LangChain and HF transformers pipeline to call Llama2 (Llama-2-7b-chat-hf) model.
"""

# LLM dir
llm_dir = r'/home/dateng/model/huggingface/meta-llama/Llama-2-7b-chat-hf'

# Crate HF pipeline
llm_pipeline = transformers.pipeline(
    'text-generation',
    model=llm_dir,

    torch_dtype=torch.float16,
    truncation=True,
    max_length=1000,
    device_map='auto'
)

# Create llm base on HuggingFacePipeline
llm_hf_pipeline = HuggingFacePipeline(pipeline=llm_pipeline, model_kwargs={'temperature': 0.8})

# Prompt template
prompt_template = PromptTemplate(template='介绍下龙珠里的{input}', input_variables=['input'])

# LLM Chain
llm_chain = prompt_template | llm_hf_pipeline
result_2 = llm_chain.invoke({'input': '卡卡罗特'})

print(result_2)
