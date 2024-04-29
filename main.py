from QaS import Chunk, Embedding
from Voice import Voice

voice = Voice()

if __name__ == "__main__":
    
    text = Chunk.getText('./datas.pdf')
    chunks = Chunk.createChunks(text=text, ch_size=2000)
    emb = Embedding.getEmbedding()
    
    # result est un tableau 
    result = emb.embed_query('Le texte à vectoriser')
    
    print(result)