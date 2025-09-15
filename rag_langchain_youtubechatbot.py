from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


#video_id = "Gfr50f6ZBvo" # only the ID, not full URL.
video_id = input("Enter YouTube video ID: ")  # User can provide their own input
try:
    # If you don’t care which language, this returns the “best” one
    ytt_api = YouTubeTranscriptApi()
    ytt_api.fetch(video_id)
    fetched_transcript = ytt_api.fetch(video_id, languages=["en"])

    # Flatten it to plain text
    #transcript = " ".join(chunk["text"] for chunk in transcript_list)
    #print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")


transcript = " ".join(snippet.text for snippet in fetched_transcript)

# Step 1b - Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")
new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)


# Step 2 - Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
#retriever.invoke('What is deepmind')


# Step 3 - Augmentation
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

#Question
#question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
question          = input("Ask Question: ")
retrieved_docs    = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt = prompt.invoke({"context": context_text, "question": question})

# Step 4 - Generation
answer = llm.invoke(final_prompt)
print(answer.content)