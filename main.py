from Voice import Voice
from QaS import Embedding, Chunk, CustomLLM, Conversation

if __name__ == '__main__':

    text = Chunk.getText('./datas.pdf')
    chunks = Chunk.createChunks(text)
    emb = Embedding.getEmbedding("./LocalModel")

    knowledge_base = Conversation.getDatabase(chunk=chunks, embedding=emb)

    voice = Voice()
    llm = CustomLLM()
    chain = Conversation.loadQA(llm=llm)
    
    voice.say('Salut, je suis votre assistant. Comment puis-je vous aider ?')
    user_query = input('Vous : ')

    while user_query.lower() != 'sortir':
        response = Conversation.getReponse(chain=chain, data=knowledge_base, query=user_query)
        voice.say(response["output_text"])
        # print(f'Bot : {response}')
        user_query = input('Vous : ')