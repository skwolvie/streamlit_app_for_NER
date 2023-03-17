#Necessary imports
import streamlit as st
import nltk_download_utils
import pandas as pd
from matplotlib import pyplot as plt
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
from gensim.summarization.summarizer import summarize 

import spacy
nlp = spacy.load("en_core_web_sm")


from collections import Counter




#Headings for Web Application
st.title("Natural Language Processing Web Application Example")
st.subheader("What type of NLP service would you like to use?")

#Picking what NLP task you want to do
option = st.selectbox('NLP Service',('Sentiment Analysis', 'Entity Extraction', 'Text Summarization')) #option is stored in this variable

#Textbox for text user is entering
st.subheader("Enter the text you'd like to analyze.")
text = st.text_input('Enter text') #text is stored in this variable

#Display results of the NLP task
st.header("Results")

#Function to take in dictionary of entities, type of entity, and returns specific entities of specific type
def entRecognizer(entDict, typeEnt):
    entList = [ent for ent in entDict if entDict[ent] == typeEnt]
    return entList


#Sentiment Analysis
if option == 'Sentiment Analysis':

    #Creating graph for sentiment across each sentence in the text inputted
    sents = sent_tokenize(text)
    entireText = TextBlob(text)
    sentScores = []
    for sent in sents:
        text = TextBlob(sent)
        score = text.sentiment[0]
        sentScores.append(score)

    #Plotting sentiment scores per sentence in line graph
    st.subheader('Sentiment Scores per Sentence Graph')
    st.line_chart(sentScores)

    #Polarity and Subjectivity of the entire text inputted
    sentimentTotal = entireText.sentiment
    st.write("The sentiment average polarity and subjectivity scores of the overall text below.")
    st.write(sentimentTotal)

#Named Entity Recognition
elif option == 'Entity Extraction':

    #Getting Entity and type of Entity
    entities = []
    entityLabels = []
    doc = nlp(text)
    for ent in doc.ents:
        entities.append(ent.text)
        entityLabels.append(ent.label_)
    entDict = dict(zip(entities, entityLabels)) #Creating dictionary with entity and entity types

    #Using function to create lists of entities of each type
    entOrg = entRecognizer(entDict, "ORG")
    entCardinal = entRecognizer(entDict, "CARDINAL")
    entPerson = entRecognizer(entDict, "PERSON")
    entDate = entRecognizer(entDict, "DATE")
    entGPE = entRecognizer(entDict, "GPE")

    #Displaying entities of each type
    st.write("Organization Entities: " + str(entOrg))
    st.write("Cardinal Entities: " + str(entCardinal))
    st.write("Personal Entities: " + str(entPerson))
    st.write("Date Entities: " + str(entDate))
    st.write("GPE Entities: " + str(entGPE))

#Text Summarization
else:
    summWords = summarize(text)
    st.subheader("Summary")
    st.write(summWords)



st.text("")
st.text("")
st.text("")
st.markdown("***")
#---------------------------------#
# About
expander_bar = st.expander("About")
expander_bar.markdown("""
* **Python libraries:** spacy, gensim, matplotlib, pandas, nltk, textblob
* **Credit:** This Mini Streamlit APP is adapted from the Medium article [Streamlit for NLP Projects](https://towardsdatascience.com/building-web-applications-with-streamlit-for-nlp-projects-cdc1cf0b38db).

""")