from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv  # to load our Api..
import requests

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-V4-Pro',
                          task='text-generation',
                          huggingfacehub_api_token="*************")

model = ChatHuggingFace(llm=llm)


# Step 1...
def extract_video_id(url):
  if 'v=' in url:
    return url.split('v=')[1].split('&')[0]
  elif 'youtu.be/' in url:
    return url.split('youtu.be/')[-1]
  else:
    return None

# Step 2...
# Create a function to fetch transcript from the video
def fetch_transcript(video_id):
      try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id)

        # 🔹 1. Try manual English subtitles
        try:
            transcript = transcript_list.find_transcript(['en'])
        
        # 🔹 2. If not available → try auto-generated English
        except:
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
            
            # 🔹 3. If still not → take ANY available (manual or auto)
            except:
                transcript = None
                for t in transcript_list:
                    transcript = t
                    break

        # 🔹 4. Translate to English if needed
        try:
            if transcript.language_code != 'en':
                transcript = transcript.translate('en')
        except:
            pass  # translation not always available

        # 🔹 5. Fetch text
        fetched = transcript.fetch()

        text = " ".join(chunk.text for chunk in fetched)

        return text

      except Exception as e:
        print("Error:", e)
        return None


# Step 3...
# create a text splitter and vector store function...
def create_vector_store(text):
  # 1st...spit text into chunks...
  splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 100)  
  chunks = splitter.create_documents([text]) # create chunks each chunk has 800 character...try different for better perfromance...

  # 2nd...Embedding...
  embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2") # model for embedding (Open source model)...

 # 3...vector store...
  vector_store = FAISS.from_documents(chunks,embedding) # use FAISS vector_store...
  return vector_store

# 4...Generate result..
def get_answer(query, retriever):
    docs = retriever.invoke(query)
    context = " ".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {query}
    """

    result = model.invoke(prompt)

    return result.content

