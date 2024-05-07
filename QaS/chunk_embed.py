import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

class Chunk():
    """Une classe à deux fonctions principales, l'extraction du texte dans un pdf et segmentation du texte
    """    
    @classmethod
    def getDocs(cls, path:str):
        if os.path.exists(path):
            loader = PyPDFLoader(file_path='../datas.pdf')
            docs = loader.load()
            
            return docs
        
        return None
    
    @classmethod
    def createChunks(cls, docs, ch_size:int = 1000, ch_over:int = 200):
        """Crée les chunks qui sont des segments de texte qui représentent des unités sémantiques plus grandes que les mots individuels mais plus petites que les phrases complètes.

        Args:
            docs (List[Document]): Le document à segmenter
            chunck_size (int): La taille des caractères pour chaque chunk
            ch_over (int): Chevauchement entre les chunks en termes de nombre de caractères

        Returns:
            chunks (List[str]): Une liste contenant plusieurs segments du texte
        """        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=ch_size, 
            chunk_overlap= ch_over, 
            separators=["\n\n","\n"," ",""]) 
        chunks = text_splitter.split_documents(documents= docs)
        return chunks

class Embedding():
    """Cette classe traite sur la représentation vectorielle des mots.\n
    Ce qui revient à dire qu'un mot sera vu comme une donnée numérique
    """
        
    @classmethod
    def getEmbedding(cls, model):
        """Une méthode qui permet d'obtenir un embedding que l'on poura utiliser partout
        """
        
        model_kwargs = {'device': 'cpu'}
        embeddings = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs=model_kwargs,
        )
        
        return embeddings