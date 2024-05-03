# Langchain Libraries
from g4f import Provider, models
from langchain.llms.base import LLM

from langchain_g4f import G4FLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain_community.callbacks import get_openai_callback
from langchain.text_splitter import TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import StrOutputParser

# ------------------------------------------------------------
    # General ChatGPT function that's required for all the Call-type Prompts 
def chatgpt_function(prompt, transcript):
    model_kwargs={"seed":235, "top_p":0.01}
    llm: LLM = G4FLLM(
        model=models.gpt_35_turbo,
        provider=Provider.Aichatos,
    )
    template = """
    {prompt}

    Call Transcript: ```{text}```
    """
    prompt_main = PromptTemplate(
        input_variables=["prompt", "text"],
        template=template,)

    with get_openai_callback() as cb:
#             llm_chain = LLMChain(llm=llm, prompt=prompt_main)
        output_parser = StrOutputParser()
        llm_chain = prompt_main | llm | output_parser
        all_text = str(template) + str(prompt) + str(transcript)
        threshold = (llm.get_num_tokens(text=all_text) + tokens)
#             print("Total Tokens:",threshold)
        if int(threshold) <= 4000:
            chatgpt_output = llm_chain.invoke({"prompt":prompt, "text":transcript})
        else:
            transcript_ = token_limiter(transcript)
            chatgpt_output = llm_chain.invoke({"prompt":prompt, "text":transcript_})
    return chatgpt_output

# -------------------------------------------------------
# Function to get refined summary if Transcript is long
def token_limiter(transcript):
    text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=200)
    texts = text_splitter.split_text(transcript)
    docs = [Document(page_content=text) for text in texts]

    question_prompt_template = """
    I'm providing you a call transcript refined summary enclosed in triple backticks. summarize it furter.

    Call Transcript: ```{text}```

    Provide me a summary transcript. do not add add any title/ heading like summary or anything else. just give summary text.
    """

    question_prompt = PromptTemplate(
        template=question_prompt_template, input_variables=["text"]
    )

    refine_prompt_template = """
    Write a summary of the following text enclosed in triple backticks (```).

    ```{text}```

    """

    refine_prompt = PromptTemplate(
        template=refine_prompt_template, input_variables=["text"]
    )

    llm: LLM = G4FLLM(
        model=models.gpt_35_turbo,
        provider=Provider.Aichatos,
    )
    refine_chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
    )

    summary_refine = refine_chain.invoke({"input_documents": docs})
    return summary_refine['output_text']

if __name__ == "__main__":
    summary = token_limiter('./datas.pdf')
    print(summary)