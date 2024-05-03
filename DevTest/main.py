from g4f import Provider, models
from langchain_g4f import G4FLLM
from langchain.llms.base import LLM
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.chains.question_answering import load_qa_chain

def main(path, ch_size:int = 3000, ch_over:int = 200):
    
    text_splitter = TokenTextSplitter(chunk_size=ch_size, chunk_overlap=ch_over)
    texts = text_splitter.split_text(path)
    print(texts)
    docs = [Document(page_content=text) for text in texts]
    
    llm: LLM = G4FLLM(
        model=models.gpt_35_turbo,
        provider=Provider.Aichatos,
    )
    
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    
    query = input("Entrez votre question ? ")
    
    input_data = {
            'input_documents': docs,
            'question': query,
        }

    # Now, pass the 'input_data' dictionary to the 'invoke' method
    reponse = chain.invoke(input=input_data)['output_text']
    print(reponse)

if __name__ == "__main__":
    main('../datas.pdf')