from tiktoken import get_encoding, encoding_for_model
from weaviate_interface import WeaviateClient, WhereFilter
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_data)
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import sys
import json
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)
##############
# START CODE #
##############
data_path = './data/impact_theory_data.json'
## RETRIEVER
api_key = os.environ['WEAVIATE_API_KEY']
url = os.environ['WEAVIATE_ENDPOINT']
client = WeaviateClient(api_key, url)
available_classes = sorted(client.show_classes())
logger.info(available_classes)

## RERANKER
reranker = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')

## LLM 
model_ids = ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613']
model_name = model_ids[1]
llm = GPT_Turbo(model=model_name, api_key=os.environ['OPENAI_API_KEY'])

## ENCODING
encoding = encoding_for_model(model_name)

## DATA and CACHE
data_path = 'data/impact_theory_data.json'

## INDEX NAME

##############
#  END CODE  #
##############
data = load_data(data_path)
#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

def main():
    with st.sidebar:
        filter_guest_checkbox = st.checkbox('Filter Guest')
        guest_input = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')
        guest_filter = None
        if filter_guest_checkbox:
            guest_filter = WhereFilter(path=['guest'], operator='Equal', valueText=guest_input).todict()

        alpha_input = st.slider('Alpha for Hybrid Search', 0.00, 1.00, 0.30, step=0.05)
        retrieval_limit = st.slider('Hybrid Search Retrieval Results', 10, 300, 10, step=10)
        reranker_topk = st.slider('Reranker Top K', 1, 5, 3, step=1)
        temperature_input = st.slider('Temperature of LLM', 0.0, 2.0, 0.10, step=0.10)
        class_name = st.selectbox('Class Name:', options=available_classes, index=None, placeholder='Select Class Name')

    if class_name == 'Ada_data_256':
        client = WeaviateClient(api_key, url, model_name_or_path='text-embedding-ada-002', openai_api_key=os.environ['OPENAI_API_KEY'])
    elif class_name == 'Impact_theory_minilm_256':
        client = WeaviateClient(api_key, url)
    else:
        client = WeaviateClient(api_key, url, model_name_or_path='./models/finetuned-all-MiniLM-L6-v2-300')

    client.display_properties.append('summary')

    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

        if query:
            ##############
            # START CODE #
            ##############
                
            # make hybrid call to weaviate
            hybrid_response = client.hybrid_search(query,
                                                   class_name,
                                                   alpha=alpha_input,
                                                   display_properties=client.display_properties,
                                                   where_filter=guest_filter,
                                                   limit=retrieval_limit)
            # rerank results
            ranked_response = reranker.rerank(hybrid_response,
                                              query,
                                              apply_sigmoid=True,
                                              top_k=reranker_topk)
            
            
            # validate token count is below threshold
            token_threshold = 8000 if model_name == model_ids[0] else 3500
            valid_response = validate_token_threshold(ranked_response, 
                                                       question_answering_prompt_series, 
                                                       query=query,
                                                       tokenizer= encoding,# variable from ENCODING,
                                                       token_threshold=token_threshold, 
                                                       verbose=True)
            ##############
            #  END CODE  #
            ##############
            make_llm_call=True
            
            # # prep for streaming response
            st.subheader("Response from Impact Theory (context)")
            with st.spinner('Generating Response...'):
                st.markdown("----")
                #creates container for LLM response
                chat_container, response_box = [], st.empty()
                
                # execute chat call to LLM
                ##############
                # START CODE #
                ##############
                # generate LLM prompt
                prompt = generate_prompt_series(query=query, results=valid_response)

                if make_llm_call:


                ##############
                #  END CODE  #
                ##############
                    try:
                        #   inserts chat stream from LLM
                        for resp in llm.get_chat_completion(prompt=prompt,
                                                            temperature=temperature_input,
                                                            max_tokens=350,
                                                            show_response=True,
                                                            stream=True):
                            
                            with response_box:
                                content = resp.choices[0].delta.content
                                if content:
                                    chat_container.append(content)
                                    result = "".join(chat_container).strip()
                                    st.write(f'{result}')
                    except Exception as e:
                        logger.info('Making request with smaller context')
                        valid_response = validate_token_threshold(ranked_response,
                                                                  question_answering_prompt_series,
                                                                  query=query,
                                                                  tokenizer=encoding,
                                                                  token_threshold=3500,
                                                                  verbose=True)
                        
                        prompt = generate_prompt_series(query=query, results=valid_response)
                        for resp in llm.get_chat_completion(prompt=prompt,
                                                            temperature=temperature_input,
                                                            max_tokens=350,
                                                            show_response=True,
                                                            stream=True):
                            try:
                                with response_box:
                                    content = resp.choice[0].delta.content
                                    if content:
                                        chat_container.append(content)
                                        result = "".join(chat_container).strip()
                                        st.write(f'{result}')
                            except Exception as e:
                                print(e)

            ##############
            # START CODE #
            ##############
            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                col1, col2 = st.columns([7, 3], gap='large')
                image = hit['thumbnail_url']# get thumbnail_url
                episode_url = hit['episode_url']# get episode_url
                title = hit['title']# get title
                show_length = hit['length']# get length
                time_string = convert_seconds(show_length)# convert show_length to readable time string
            ##############
            #  END CODE  #
            ##############
                with col1:
                    st.write( search_result(  i=i, 
                                                url=episode_url,
                                                guest=hit['guest'],
                                                title=title,
                                                content=hit['content'], 
                                                length=time_string),
                            unsafe_allow_html=True)
                    st.write('\n\n')
                with col2:
                    # st.write(f"<a href={episode_url} <img src={image} width='200'></a>", 
                    #             unsafe_allow_html=True)
                    st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)

if __name__ == '__main__':
    main()