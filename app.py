from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from Voice import Voice
from QaS import Conversation, CustomLLM, Embedding

emb = Embedding.getEmbedding('./LocalModel')
retriever = Conversation.getDatabase(embeddings=emb, path='./vectors')

llm = CustomLLM()
prompt = Conversation.promptSystem()
parser = StrOutputParser()

result = RunnableParallel(context=retriever, question=RunnablePassthrough())
chain = result | prompt | llm | parser

voice = Voice()
query = input('Votre question ? ')
while query.lower() != 'sortir' :
    voice.say(f"{chain.invoke(query).replace('*', '')}\n")
    query = input('Votre question ? ')