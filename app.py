import streamlit as st

# NLP Pkgs
import spacy
from textblob import TextBlob
from gensim.summarization import summarize


# Sumy Pkgs
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Summary Fun
def sumy_summarizer(docx,per_fit):
    parser = PlaintextParser.from_string(docx,Tokenizer('english'))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,per_fit)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)

    tokens = [token.text for token in docx]
    allData = ['"Tokens":{},\n"Lemma":{}'.format(token.text,token.lemma_) for token in docx]
    return allData



def entity_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)

    entities = [(entity.text,entity.label_) for entity in docx.ents]
    return entities




# Pkgs


def main():
    """ NLP App with Streamlit """

    st.sidebar.subheader("About the Summarizer")
    st.sidebar.text("NLPiffy App with Streamlit")
    st.sidebar.info("Developer [Info](https://www.linkedin.com/in/sonu-bhadana/)")

    st.title("NLPiffy with Streamlit")
    st.subheader("Natural Language Processing on the Go :")

    message = st.text_area("Enter text !","Type Here..." , height=300)
    # import pyperclip as pc
    # message = pc.paste()


    # Tokenization
    if st.checkbox("Show Token and Lemma"):
        st.subheader("Tokenize Your Text")
        if st.button("Analyze"):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)


    # Named Entity
    if st.checkbox("Show Named Entites"):
        st.subheader("Extract Entities from your Text")
        if st.button("Extract"):
            nlp_result = entity_analyzer(message)
            st.json(nlp_result)

    # Sentiment Analysis
    if st.checkbox("Show Sentiment Analysis"):
        st.subheader("Sentiment of your Text")
        if st.button("Analyze"):
            blob = TextBlob(message)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)

    # Text Summarization
    if st.checkbox("Show Text Summarization"):
        st.subheader("Summarize your Text")
        summary_options = st.selectbox("choose your summarizer",("gensim","sumy"))
        if(summary_options == "sumy"):
            percentage = st.slider("Chose the summary percentage : (Recommended equal or above 20%)", min_value=10, max_value=50,step=5)
            per_fit = percentage/5
        if st.button("Summarize"):

            if(summary_options == 'gensim'):
                st.text("Using Gensim")
                summary_result = summarize(message)

            elif(summary_options == 'sumy'):
                st.text("Using Sumy")
                summary_result = sumy_summarizer(message,per_fit)

            else:
                st.warning("Using Default Summarizer")
                st.text("Using Gensim")
                summary_result = summarize(message)

            st.success(summary_result)

if __name__ == '__main__':
    main()
