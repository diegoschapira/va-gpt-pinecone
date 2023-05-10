import streamlit as st
import openai
import pinecone
import pandas as pd

##Pinecone Auth
PINECONE_KEY = ''
PINECONE_INDEX_NAME = ''
PINECONE_ENV = ''

##Azure OpenAI Auth
AZURE_OPENAI_VERSION = ''
AZURE_OPENAI_ENDPOINT = '' 
AZURE_OPENAI_KEY = '' 

#OpenAi Settings
embed_model = "TextEmbeddingAda002"
completion_model = "Davinci003"
openai.api_type = "azure"
openai.api_version = AZURE_OPENAI_VERSION
openai.api_base = AZURE_OPENAI_ENDPOINT 
openai.api_key = AZURE_OPENAI_KEY 

## Connect to Pinecone
pinecone.init(
    api_key=PINECONE_KEY,
    environment=PINECONE_ENV  # may be different, check at app.pinecone.io
)
index = pinecone.Index(PINECONE_INDEX_NAME)

limit = 3750

st.set_page_config(page_title='Genie Help!',page_icon=":dart:")

header = st.container()
body = st.container()

with header:
    ##st.image('Oral-B-logo.png',width=200)
    st.title('''Hi I'm Genie! How can I help?''')
    
with body:
    
    with st.form(key="my_form"):
        query = st.text_input('Ask me a question including as much detail as possible (for example: Where I can find replacement head for Series 8 Type 5795 ?)')
                
        style = st.radio('Select a style of writing:',('Direct Answer','Email Message','Summary'))
        if style == 'Dierct Answer':
            instruction = '''Answer the question by extracting information from Context below. If information is not in Context answer "I'm unable to answer the question". Do not generate responses that don't use the information in Context.'''
        elif style == 'Email Message':
            instruction = '''You are a Braun chat assistant. Your goal is to build brand trust and make consumers feel valued. Write a personalized response to the consumer question by extracting information from Context. If information is not in Context answer "I'm unable to answer the question". Do not generate responses that don't use the information in Context or if information in Context is not related to the Question.'''
        elif style == 'Summary':
            instruction = '''Extract main issues from consumer question'''
        else:
            instruction = '''Answer the question by extracting information from the context below. If information is not in context answer "I don't know"'''

        submitted = st.form_submit_button("Ask")
    
    ###
    def retrieve(query):
        res_embds = openai.Embedding.create(
            input=[query],
            engine=embed_model
        )

        # get relevant contexts
        xq = res_embds['data'][0]['embedding']
        pine_res = index.query(xq, top_k=2, include_metadata=True)
        contexts = [x['metadata']['Context'] for x in pine_res['matches']]
        article_title = [x['metadata']['Question'] for x in pine_res['matches']]
        article_num = [x['metadata']['Article Number'] for x in pine_res['matches']]
        similarity = [x['score'] for x in pine_res['matches']]
        s = {"article number":article_num,
            "article title":article_title,
            "similarity score":similarity}
        df_sources = pd.DataFrame(s).astype({'article number': 'int'})
                

        # build our prompt with the retrieved contexts included
        prompt_start = (
            f"{instruction}.\n\n"+
            "Context:\n"
        )
        prompt_end = (
            f"\n\nConsumer Question: {query}\nResponse:"
        )
        # append contexts until hitting limit
        for i in range(1, len(contexts)):
            if len("\n\n---\n\n".join(contexts[:i])) >= limit:
                prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(contexts[:i-1]) +
                    prompt_end
                )
                break
            elif i == len(contexts)-1:
                prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(contexts) +
                    prompt_end
                )
        return prompt, df_sources

    def complete(prompt):
        # query text-davinci-003
        res = openai.Completion.create(
            engine=completion_model,
            prompt=prompt,
            temperature=0,
            max_tokens=600,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return res['choices'][0]['text'].strip()
    
        
    ###
    if submitted:    
        prompt_with_contexts = retrieve(query)[0]
        sources = retrieve(query)[1]
        answer = complete(prompt_with_contexts)
        st.subheader('Answer:')
        st.write(answer)
        st.subheader("The answer was sourced from these articles:")
        st.dataframe(sources)
        st.write(prompt_with_contexts.replace(instruction+".","").replace("Context:",""))
