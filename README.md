# Question-Answering-system-using-Cosine-similarity
In this project we will find the best possible answer to the multiple choice question using tfidf embeddings and cosine similarity.


#### Part 1 - Model:

  + We will build the term document matrix using data.txt.
  + Treat each sentence as single document
  + Term document matrix will contain tf-idf values.


#### Part 2 - Query:

  + We will be using test.jsonl file to obtain the questions. 
  + Each line contains 1 question in json format along with 4 options. 
  + To form a query, combine each question with it’s option, ie, (Q​i​ + O​i​).
  
#### Evaluation :

    + For each query, we find the *cosine similarity* between the query and each of the documents and use the similarity 
    score of the most matching document as the score of your system for the query. 
    +  We repeat this for all the options and our answer for a particular question should be the (Q​i + O​i​) combination 
    with the maximum score. 
    
