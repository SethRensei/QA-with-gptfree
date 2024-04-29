import os

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

class Chunk():
    """Une classe à deux fonctions principales, l'extraction du texte dans un pdf et segmentation du texte
    """    
    @classmethod
    def getText(cls, path:str):
        if os.path.exists(path):
            pdf = PdfReader(path)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            
            return text
        
        return ""
    
    @classmethod
    def createChunks(cls, text:str, ch_size:int = 1000, ch_over:int = 200):
        """Crée les chunks qui sont des segments de texte qui représentent des unités sémantiques plus grandes que les mots individuels mais plus petites que les phrases complètes.

        Args:
            text (str): Le text à segmenter
            chunck_size (int): La taille des caractères pour chaque chunk
            ch_over (int): Chevauchement entre les chunks en termes de nombre de caractères

        Returns:
            chunks (List[str]): Une liste contenant plusieurs segments du texte
        """        
        text_split = CharacterTextSplitter(
            separator='\n',
            chunk_size=ch_size,
            chunk_overlap=ch_over,
            length_function=len
        )
        chunks = text_split.split_text(text)
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