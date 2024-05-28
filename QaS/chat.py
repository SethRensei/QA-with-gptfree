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
            provider=Provider.OpenaiChat
        )
        if stop:
            stop_indexes = (out.find(s) for s in stop if s in out)
            min_stop = min(stop_indexes, default=-1)
            if min_stop > -1:
                out = out[:min_stop]
        return out


class Conversation():
    
    @classmethod
    def promptSystem(self):
        template = """
        Vous es Seth Rensei, un assistant routier. Vous devriez répondre à la question de la fin en utilisant le contexte suivant.
        Si la réponse n'est pas inclu dans le contexte, dites simplement que vous ne connaissez pas.
        N'essayez pas d'inventer de réponse.

        Context : {context}\n

        Question : {question}\n

        """

        return PromptTemplate(input_variables=['context', 'question'], template=template)

    @classmethod
    def createDBVector(cls, chunks, embedding, path:str):
        """Crée et enregistre la base de données vectorielles

        Args:
            chunk (_type_): La liste contenant le texte segmenter
            embedding (_type_): La représentation vectorielles (nombres) des mots
            path (str): Le chemin pour enregistrer la base de données

        Returns:
            _type_: Une base de connaissance utilisable pour notre SQR
        """        
        knowlegde_base = faiss.FAISS.from_documents(chunks, embedding=embedding)
        knowlegde_base.save_local(path)
        
        return knowlegde_base
    
    @classmethod
    def getDatabase(cls, embeddings, path:str):
        """_summary_

        Args:
            embeddings (_type_): La représentation vectorielles (nombres) des mots
            path (str): Le chemin vers le dossier de la base de connaissances

        Returns:
            _type_: _description_
        """        
        loaded_vectors = faiss.FAISS.load_local(folder_path=path, embeddings=embeddings, allow_dangerous_deserialization=True)
        return loaded_vectors.as_retriever()