import g4f
from g4f import Provider, models
from langchain.llms.base import LLM
from typing import List, Any
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_community.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain

class CustomLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt:str, stop:List[str] = None, **kwargs:Any) -> str:
        out = g4f.ChatCompletion.create(
            model=models.gpt_35_turbo,
            messages=[{"role": "user", "content": prompt}],
            provider=Provider.Aichatos
        )
        if stop:
            stop_indexes = (out.find(s) for s in stop if s in out)
            min_stop = min(stop_indexes, default=-1)
            if min_stop > -1:
                out = out[:min_stop]
        return out
    
    @classmethod
    def promptSystem(self):
        template = """
            Tu dois répondre aux questions dont les réponses ne sont pas inclue dans les informations fournies.
            Si la question n'a pas de réponse, tu peux le faire comprendre aux utilisateur\n
            
            {context}\n
            
            user_question : {question}
            
            Answer : 
            """
        return PromptTemplate(template=template, input_variables=['context', 'question'])



class Conversation():
    
    @classmethod
    def getDatabase(cls, chunk, embedding):
        """Retourne la notre base de données qui sera les connaissances du SQR

        Args:
            chunk (_type_): La liste contenant le texte segmenter
            embedding (_type_): La représentation numérique des mots

        Returns:
            _type_: Une base de connaissance utilisable pour notre SQR
        """        
        return faiss.FAISS.from_texts(chunk, embedding=embedding)
    
    @classmethod
    def loadQA(cls, llm):
        return load_qa_chain(llm, chain_type="stuff")
    
    @classmethod
    def getReponse(cls, chain: BaseCombineDocumentsChain, data:faiss.FAISS, query:str):
        """Cette méthode donne la réponse à une question

        Args:
            llm (_type_): Le modèle LLM à utiliser
            data (faiss.FAISS): La base de connaissance
            query (str): La requête ou la question

        Returns:
            _type_: La réponse à la question
        """        
        docs = data.similarity_search(query)
        input_data = {
            'input_documents': docs,
            'question': query,
        }
        return chain.invoke(input=input_data)