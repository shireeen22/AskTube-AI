import streamlit as st
from rag_pipeline import *

st.title('AskTube AI🔍🤖')
st.header('🎥Welcome to our App')
st.subheader("Just ask your question related to this video")

# ------taking input from user------

video_url = st.text_input('Enter Youtube video URL')
if video_url:
  video_id = extract_video_id(video_url) # it will return video_id from url...
  if not video_id:
    st.error('Invalid URL❌')
  else:
    
    st.write("⏳ Processing video...")
    text = fetch_transcript(video_id)

    # If subtitles are absent for the video...
    if text is None:
      st.error('No subtitles for this video🚫')
    else:
      vector_store = create_vector_store(text)

      # use retriever...
      retriever = vector_store.as_retriever(search_type='similarity', 
                                search_kwargs={'k':3})
      
      st.success('Video processed successfully!✅Ask question👇')
      # taking query from user...
      query = st.text_input('Enter your query')
      if query:
        answer = get_answer(query,retriever)

        st.write("📌 Answer:")
        st.write(answer)
   