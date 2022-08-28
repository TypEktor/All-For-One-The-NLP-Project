# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 18:55:04 2022

@author: gioek
"""

import re
import deEmojifyer as deEmo
import Contractions as co
from tqdm import tqdm
from cucco import Cucco
from bs4 import BeautifulSoup
import unidecode
from autocorrect import Speller
import nltk
from nltk.stem import SnowballStemmer
import warnings
warnings.filterwarnings("ignore")


def TwitterCleaner(reviews):
    """ 
    This function will remove all common tags that can be found in twitter texts. This function removes 5 categories
    that can be found mostly in tweets
        1. RT @..
        2. &..
        3. http..
        4. ...com
        5. @..
        
    arguments:
        reviews: reviews + their label of type dataframe
                    
    return:
        reviews: cleaned reviews + their label of type dataframe
         
    """
    # Removing RT
    remove_rt = lambda x: re.sub('RT @\w+ '," ",x)
    reviews["review"] = reviews['review'].map(remove_rt)
    # Removing &amp; which is equal to &
    reviews['review'] = reviews['review'].replace(r'&amp;', '', regex=True)
    # Removing sites
    reviews['review'] = reviews['review'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
    # Remove all the occurrences of text that ends with .com
    reviews['review'] = reviews['review'].replace(r"\ [A-Za-z]*\.com", '', regex=True)
    # Removing @...
    reviews['review'] = reviews['review'].replace(r'@\S+', '', regex=True)
    
    return reviews
       
def strip_html_tags(reviews):
    """ 
    This function will remove all the occurrences of html tags from the reviews
    
    arguments:
        reviews: reviews + their label of type dataframe 
                    
    return:
        reviews: cleaned reviews
         
    """
    # Initiating BeautifulSoup object soup.
    soup = BeautifulSoup(reviews, "html.parser")
    # Get all the reviews other than html tags.
    stripped_reviews = soup.get_text(separator=" ")
    return stripped_reviews 
    
def remove_newlines_tabs(reviews):
    """
    This function will remove all the occurrences of newlines, tabs, and combinations like: \\n, \\
    
    arguments:
       reviews: reviews + their label of type dataframe
                    
    return:
        reviews: cleaned reviews
    
    """
    
    # Replacing all the occurrences of \n,\\n,\t,\\ with a space.
    Formatted_reviews = reviews.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com')
    return Formatted_reviews

def accented_characters_removal(reviews):
    # this is a docstring
    """
    The function will remove accented characters from the reviews contained within the Dataset.
       
    arguments:
        reviews: reviews + their label of type dataframe
                    
    return:
        reviews: cleaned reviews + their label of type dataframe
        
    Example:
    Input : Málaga, àéêöhello
    Output : Malaga, aeeohello    
        
    """
    # Remove accented characters from reviews using unidecode.
    # Unidecode() - It takes unicode data & tries to represent it to ASCII characters. 
    reviews = unidecode.unidecode(reviews)
    return reviews

def remove_whitespace(reviews):
    """ This function will remove extra whitespaces from the reviews
    arguments:
        reviews: reviews + their label of type dataframe
                    
    return:
        reviews: cleaned reviews
        
    """
    pattern = re.compile(r'\s+') 
    Without_whitespace = re.sub(pattern, ' ', reviews)
    # There are some instances where there is no space after '?' & ')', 
    # So I am replacing these with one space so that It will not consider two words as one token.
    reviews = Without_whitespace.replace('?', ' ? ').replace(')', ') ')
    
    return reviews


def spelling_correction(reviews):
    ''' 
    This function will attempt to correct spellings.
    
    arguments:
         reviews: reviews + their label of type dataframe
         
    return:
        reviews: cleaned reviews
            
    '''
    # Check for spellings in English language
    spell = Speller(lang='en')
    Corrected_reviews = spell(reviews)
    return Corrected_reviews


def lemmatization(reviews):
    """This function converts word to their root words without explicitely cut down as done in stemming.
    
    arguments:
         reviews: reviews + their label of type dataframe
         
    return:
        reviews: reviews after lemmatisation
            
   """
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    # Converting words to their root forms
    lemma = [lemmatizer.lemmatize(w,'v') for w in w_tokenizer.tokenize(reviews)]
    return lemma

def stemming(reviews):
    """This function applies stemming, a set of general rules is used to identify the
    end bits of words that can be chopped off to leave the core root of the word
    
    arguments:
         reviews: reviews + their label of type dataframe
         
    return:
        reviews: reviews after stemming
   """
    sbEng = SnowballStemmer('english')
    reviews = [sbEng.stem(word) for word in reviews.split(' ')]
    return reviews

# All In One Cleaner
def AIO_cleaner(reviews, lower_case = False, tweets = False,
                             punctuation = False, emojis = False, stopwords = False,
                             accented = False, spelling = False, lem_stem = 'None'):
    
    """ This function attempts to apply a plethora of different cleaning techniques that are considered
    important when dealing with text data. The function gives choice to the user to select which method to use.
    The user can:
        • Decide whether he wants to lower case every single word
        • Decide whether he wants to delete symbols and text found it Tweets
        • Decide whether he wants to remove punctuations
        • Decide whether he wants to remove emojis
        • Decide whether he wants to remove stopwords
        • Decide whether he wants to change accented characters
        • Decide whether he wants to attempt to fix spelling mistakes
        • Decide whether he wants to use lemmatisation or stemming
        
    The function does a few more things, without input from the user:
        • Drops dublicates
        • Drops empty reviews
        • Replaces Contractions
        • Strips HTML Tags
        • Removes newlines and tabs
        • Removes Numbers
        • Removes whitespace
    
    arguments:
        reviews: reviews + their label of type dataframe
        
        lower_case: True if the user wants everything to be lowercase
        
        tweets: This is suggested if your corpus contains tweets. However, it is ideal
                if you want to clean sites (http, www). If true it removes RT, &, sites, @..
                
        punctuation: If True it removes all the puncuations from the sentense
        
        emojis: If True it removes all the emojis from the text
            
        stopwords: If True is removes all stopwords
        
        accented: If True it change all accented characters
         
        spelling: If True it fixes all spelling mistakes (Proceed with Caution)
          
        lem_stem: If lemmatization is given as input it will apply lemmatization, if stemming is given
        it will apply stemming (SnowballStemmer). Default is None.
        
    return:
        reviews: A clean dataframe that contains the clean reviews and their labels
                
    """
    tqdm.pandas()
    # Drops dublicates
    reviews = reviews.drop_duplicates()
    # Drops empty reviews
    reviews = reviews[reviews['review'].notnull()]
    
        
    if tweets:
        reviews = TwitterCleaner(reviews)
        print("• Tweet Symbols Removal (Complete) \n ------------------------------------")
        
    # Replace Contractions
    reviews["review"] = reviews['review'].map(lambda x: co.expandContractions(x))
    print("• Contractions Replacement (Complete) \n ------------------------------------")
    
    
    # Choose between setting everything lower case or not
    if lower_case:
        reviews['review'] = reviews['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
        print("• Lower Case (Complete) \n ------------------------------------")
    else:
        print('something with entity recognition')
        
    
    # Replace Contractions - Stage 2
    reviews["review"] = reviews['review'].map(lambda x: co.expandContractions(x))
    print("• Contractions Replacement (Complete) \n ------------------------------------")
    
    # Remove Accented Characters
    if accented:
        reviews["review"] = reviews['review'].map(lambda x: accented_characters_removal(x))
        print("• Accented Characters Removal (Complete) \n ------------------------------------")
    
    if punctuation:
        reviews['review'] = reviews['review'].str.replace('[^\w\s]','')
        print("• Punctuation Removal (Complete) \n ------------------------------------")
        
    if emojis:
        reviews['review'] = reviews['review'].progress_apply(deEmo.deEmojify)
        print("\n• Emojis Removal (Complete) \n ------------------------------------")
        
    if stopwords:
        norm = Cucco()
        norms = ['remove_stop_words']
        reviews["review"] = reviews['review'].map(lambda x: norm.normalize(x, norms))
        print("• Stopwords Removal (Complete) \n ------------------------------------")


    # Strip HTML Tags
    reviews['review'] = reviews['review'].apply(lambda x: strip_html_tags(x))
    # Remove newlines and tabs
    reviews['review'] = reviews['review'].apply(lambda x: remove_newlines_tabs(x))
    # Remove <br> and </br>
    # reviews['review'] = reviews['review'].replace('<br>', '', regex=True).replace('</br>', '', regex=True)
    # Remove Numbers
    reviews['review'] = reviews['review'].replace('[0-9]+', '', regex=True)

    
    # Spell Correction
    if spelling:
        reviews['review'] = reviews['review'].progress_apply(lambda x: spelling_correction(x))
        print("\n• Spell Correction  (Complete) \n ------------------------------------")
    
    
    if lem_stem == 'lemmatization':
        reviews['review'] = reviews['review'].progress_apply(lambda x: lemmatization(x))
        print("\n• Lemmatization  (Complete) \n ------------------------------------")
    if lem_stem == 'stemming':
        # https://www.nltk.org/_modules/nltk/stem/snowball.html
        reviews['review'] = reviews['review'].progress_apply(lambda x: ' '.join(stemming(x)))
        print("\n• Stemming  (Complete) \n ------------------------------------")
        
        
    # Remove whitespace
    reviews['review'] = reviews['review'].apply(lambda x: remove_whitespace(x))
    print("• HTML Tags, Newlines/Tabs, Numbers, Whitespace Removal  (Complete) \n ------------------------------------")
    
        
    
    # Drops dublicates again
    reviews = reviews.drop_duplicates()
    # Drops empty reviews
    reviews = reviews[reviews['review'].notnull()]
    filter = reviews["review"] != ""
    reviews = reviews[filter]
    
    # Trim
    reviews['review'] = reviews['review'].apply(lambda x: x.lstrip())
    
    print('Review Mean Value:' , reviews['review'].apply(lambda x: len(x.split(" "))).mean())
    
    return reviews
    


