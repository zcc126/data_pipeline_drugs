import os
import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
from collections import Counter
from nltk.corpus import stopwords

map_month_dict=dict(zip(['JANUARY', 'FEBRUARY', 'MARTCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER'],
                        ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']))
stop_words = [i.upper() for i in set(stopwords.words('english'))]

def pipeline(df, split_cols, remove_char_cols, col, time_col, merge_col, replace_col, map_month_dict, criterion_merge, criterion_replace, stop_words, source_name):
    
    split_text(df=df,
               cols=split_cols)
    
    remove_character(df=df,
                     cols=remove_char_cols)
    
    drop_useless_row (df=df,
                      col=col)
    
    alignment_time(df=df,
                   time_col=time_col,
                   map_month_dict=map_month_dict)
    
    df=merge_row_similarity(df=df,
                            col=merge_col,
                            criterion=criterion_merge)
    
    replace_similar_values(df=df,
                       col=replace_col,
                       criterion=criterion_replace)
    
    df_word=word_dict(df=df,
                   col=col,
                   stop_words=stop_words)
    
    df_word2=merge_word(df=df_word,
                       source_name=source_name)
    
    outdir = './cleaned_data'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    file_name=os.path.join(outdir, source_name + '.csv')
    result_name=os.path.join(outdir, 'drugs_in_'+ source_name +'.csv')
    
    df.to_csv(file_name)
    df_word2.to_csv(result_name)
    
    return df, df_word2


def get_df():
    cwd=os.getcwd()  # get current path
    df_dict={}
    for csv_name in [i for i in os.listdir(cwd) if i.endswith('.csv')]:
        df_name=csv_name.split('.')[1]
        df_path=os.path.join(cwd, csv_name)
        df_dict[df_name]=pd.read_csv(df_path)
    return df_dict

def alignment_text(x):
    x=unicodedata.normalize('NFD', x)
    return (
        (x.encode('ascii', errors='ignore')
         .decode('utf-8')
        ).upper()
    ).rstrip()

def split_text(df, cols):
    for col in cols:
        df[col]=df[col].apply(lambda x: str(x))
        df[col]=df[col].apply(alignment_text)
        df[col]=df[col].apply(lambda x: x.split(' '))

def remove_character(df, cols):
    for col in cols:
        df[col]=df[col].apply(lambda x: [str(i) for i in x])
        df[col]=df[col].apply(lambda x: [("").join(re.findall('[\w+]',i)) for i in x])
        df[col]=df[col].apply(lambda x: (" ").join(x))

def drop_useless_row (df, col):
    index_drop=df[df[col].isin([np.NaN, '', None, ' ', ['']])].index
    df.drop(index=index_drop.values, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
def alignment_month(x, map_dict):
    month_str=('').join(re.findall('[A-Za-z]', x))
    if len(month_str) >  0:
        month_num=pd.Series([month_str.upper()]).map(map_dict).values[0]
        x=re.sub(month_str, month_num, x)
    return x

def get_ymd(x):
    time_set=re.findall('\s*(\d+)\s*', x)                                           # output=['2020', '01', '01']
    time_set_len=[len(i) for i in time_set]                                         # output=[4, 2, 2]                                    
    time_set_loc=dict((length, index) for index,length in enumerate(time_set_len))  # output={4:0, 2:2}
    
    # get year
    year=time_set[time_set_loc[4]]
    
    # get month
    month=time_set[1]
    if len(month)==1:
        month='0'+month
    
    # get day
    day=time_set[2-time_set_loc[4]]
    if len(day)==1:
        day='0'+day
    
    x=('/').join([day, month, year])
    return x

def alignment_time(df, time_col, map_month_dict):
    df[time_col]=df[time_col].apply(lambda x: alignment_month(x, map_month_dict))
    df[time_col]=df[time_col].apply(lambda x: get_ymd(x))

    
def merge_row_similarity(df, col, criterion):
    doc=list(df[col].values)   # ['XXXX', 'XXXX', 'XXX',...]

    # identify the rows who have similar content by TfidfVectorizer
    vect = TfidfVectorizer(min_df=1, stop_words="english")
    tfidf = vect.fit_transform(doc)
    pairwise_similarity = tfidf * tfidf.T
    
    df_similarity=pd.DataFrame(pairwise_similarity.toarray())
    df_similarity['count']=(df_similarity>criterion).replace(False, np.NaN).count(axis=1)
    df_similarity['s_index']=df_similarity.apply((lambda x: sorted([count for count, value in enumerate(x) if value >criterion and value <1.0001 and x['count']>1])),axis=1)
    df_similarity['s_index']=df_similarity['s_index'].astype(str) # trun [4,5] into string for groupby function
    index_group=[] # save all similar rows' index
    
    for i in df_similarity.groupby('s_index'):
        index_id=re.findall('.(\d+).',i[0])      # output: ['4', '5']
        index_id=list(map(int, index_id))        # output: [4, 5]
        if len(index_id)>0:
            index_group.append(index_id)         # output: [[4,5], [x,y,z],...]
    
    # replace the multiple rows by only one new merged row
    if len(index_group)>0:
        df_new=pd.DataFrame(columns=df.columns)
        for index_s in index_group:
            df_new_s=pd.DataFrame(columns=df.columns)
            df_old_s=df[df.index.isin(index_s)]
            for i in df_old_s.columns.values:
                values=list(df_old_s[i].values)
                values=sum([], values)
                values=list(filter(lambda x: x not in [np.NaN, None, '', ' ', 'NAN', 'None', 'nan'], values)) # remove useless information
                values=list(set(values))
                if len(values)==1:
                    df_new_s[i]=values
                else:
                    df_new_s[i]=[values]
            df_new=pd.concat([df_new_s,df_new], axis=0, ignore_index=True, copy=False)
        df.drop(index=sum(index_group,[]), inplace=True)
        df=pd.concat([df,df_new], axis=0, ignore_index=True)
    return df

def replace_similar_values(df, col, criterion):
    
    # count element in the column
    count=Counter(list(df[col].values))
    count=dict(count)                                              
    count=pd.DataFrame([list(count.keys()), list(count.values())]).T
    count.columns=['name', 'count']
    count.sort_values(by='count', inplace=True, ascending=False)
    
    # we compare the similarity with unique values
    unique_items=list(set(df[col].values))
    df_sim=pd.DataFrame(index=unique_items, columns=unique_items)

    for i in unique_items:
        for j in unique_items:
            df_sim.loc[i,j]= SequenceMatcher(None, i, j).ratio()
            
    df_sim['count']=(df_sim>criterion).replace(False, np.NaN).count(axis=1)
    df_sim['s_index']=df_sim.apply((lambda x: sorted([count for count, value in enumerate(x) if value >criterion and value <1.0001 and x['count']>1])),axis=1)
    
    # replace the values in the original df
    df_sim['s_index']=df_sim['s_index'].astype(str)
    for i in df_sim.groupby('s_index'):
        index_id=re.findall('.(\d+).',i[0])
        index_id=list(map(int, index_id))
        if len(index_id)>0:
            index_id=df_sim.iloc[index_id].index.values
            top_name=count[count.name.isin(index_id)]['name'][0]
            df[col].replace(index_id, top_name, inplace=True)
    return df

def word_dict (df, col, stop_words):
    
    # turn "hello world" into ['hello', 'word']
    df['word_split']=df[col].apply(lambda x: x.split(' '))
    
    # remove stop words from string for each row
    df['word_split']=df['word_split'].apply(lambda x: list(set(x).difference(set(stop_words))))
    
    # remove digital word like '2' from string
    df['word_split']=df['word_split'].apply(lambda x: [i for i in x if not i.isdigit()])
    
    # join word, date, journal
    df['combinaison']=df.apply(lambda x: [[i, x['date'], x['journal'], x['id']] for i in x['word_split']], axis=1)
    
    df_word=pd.DataFrame(sum(df['combinaison'].values.tolist(),[]))
    df_word.columns=['word', 'date', 'journal', 'reference']
    df_word=df_word.drop_duplicates(ignore_index=True)
    
    df.drop(columns=['word_split', 'combinaison'], inplace=True)
    return df_word


# if one word has been quoted several times for several journals
def merge_word (df, source_name):
    
    df_new=pd.DataFrame(columns=df.columns) # new df to save the merged row
    index_drop=[]                           # row to be dropped in df
    
    for i in df.groupby('word'):
        if len(i[1])>1:
           
            new_row=pd.DataFrame(columns=i[1].columns)       
            new_row['word']=[i[0]]
            new_row['date']=[list(i[1].date.values)]
            new_row['journal']=[list(i[1].journal.values)]
            new_row['reference']=[list(i[1].reference.values)]
            
            df_new=pd.concat([df_new, new_row], ignore_index=True)
            index_drop.append(list(i[1].index.values))
            
    index_drop=sum(index_drop,[])        
    df.drop(index=index_drop, inplace=True)
    df=pd.concat([df,df_new],axis=0, ignore_index=True)
    df['source']=source_name
    return df

def search_drugs (drugs_list, df_sources, source_name):
    
    df=df_sources[df_sources.word.isin(drugs_list)]
                           
    outdir = './results'
    if not os.path.exists(outdir):
        os.mkdir(outdir)                      
    result_name=os.path.join(outdir, 'search_drugs_in'+ source_name + '.csv')
    df.to_csv(result_name)
    print(df)
    

if __name__=='__main__':
    
    df_set=get_df()
    df_clinical=df_set['clinical_trials'].copy()
    df_pubmed=df_set['pubmed'].copy()
    df_drugs=df_set['drugs'].copy()
    drugs_list=df_drugs.drug.values
    
    df_clinical, df_clinical_word=pipeline(df=df_clinical,
                                       split_cols=['scientific_title', 'journal', 'id'],
                                       remove_char_cols=['scientific_title', 'journal', 'id'],
                                       col='scientific_title',
                                       time_col='date',
                                       merge_col='scientific_title',
                                       replace_col='journal',
                                       map_month_dict=map_month_dict,
                                       criterion_merge=0.9,
                                       criterion_replace=0.9,
                                       stop_words=stop_words,
                                       source_name='clinical')
    
    search_drugs (drugs_list=drugs_list,
                  df_sources=df_clinical_word,
                  source_name='clinical')
    
    df_pubmed, df_pubmed_word=pipeline(df=df_pubmed,
                                   split_cols=['title', 'journal', 'id'],
                                   remove_char_cols=['title', 'journal', 'id'],
                                   col='title',
                                   time_col='date',
                                   merge_col='title',
                                   replace_col='journal',
                                   map_month_dict=map_month_dict,
                                   criterion_merge=0.9,
                                   criterion_replace=0.9,
                                   stop_words=stop_words,
                                   source_name='pubmed')
    
    search_drugs (drugs_list=drugs_list,
                  df_sources=df_pubmed_word,
                  source_name='pubmed')