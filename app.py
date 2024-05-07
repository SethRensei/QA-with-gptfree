from QaS import Conversation, CustomLLM, Embedding
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

emb = Embedding.getEmbedding('./LocalModel')
retriever = Conversation.getDatabase(embeddings=emb, path='./vectors')

prompt = Conversation.promptSystem()
parser = StrOutputParser()

result = RunnableParallel(context=retriever, question=RunnablePassthrough())
llm = CustomLLM()
chain = result | prompt | llm | parser

query = input('Votre question ? ')
while query.lower() != 'sortir' :
    print(f"{chain.invoke(query)}\n")
    query = input('Votre question ? ')