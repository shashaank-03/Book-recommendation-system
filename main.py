from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
@app.route('/survey')
def survey():
    return render_template("i1.html")
@app.route('/i2')
def showI2():
    return render_template("i2.html")
@app.route('/i3')
def showI3():
    return render_template("i3.html")
@app.route('/a')
def showA():
    return render_template("a.html")

@app.route("/recommend",methods=["GET","POST"])
def recommend():
    books=pd.read_csv('Books.csv',low_memory=False,encoding='latin-1')
    books=books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication','Image-URL-L','Publisher']]
    books.rename(columns={'Book-Title':'title','Book-Author':'author','Year-Of-Publication':'year','Image-URL-L':'img','Publication':'publication'},inplace=True)
    books['title']=books['title'].str.lower()
    # print(books.head(2))
    users=pd.read_csv('Users.csv',low_memory=False,encoding='latin-1')
    users.rename(columns={'User-ID':'user-id','Location':'location','Age':'age'},inplace=True)
    # print(users.head(2))
    ratings=pd.read_csv('Ratings.csv',low_memory=False,encoding='latin-1')
    ratings.rename(columns={'User-ID':'user-id','Book-Rating':'rating'},inplace=True)
    # print(ratings.head(2))
    x=ratings['user-id'].value_counts()>200
    y=x[x].index
    ratings=ratings[ratings['user-id'].isin(y)]
    rated_books=ratings.merge(books,on='ISBN')
    # print(rated_books.head(2))
    no_rating=rated_books.groupby('title')['rating'].count().reset_index()
    no_rating.rename(columns={'rating':'rating_count'},inplace='True')

    final_ratings=rated_books.merge(no_rating,on='title')
    final_ratings=final_ratings[final_ratings['rating_count']>=50]
    final_ratings.drop_duplicates(['user-id','title'],inplace=True)
    # print(final_ratings.head(2))
    book_pivot=final_ratings.pivot_table(columns='user-id',index='title',values='rating')
    book_pivot.fillna(0,inplace=True)
    #print(book_pivot.head()) 
    book_sparse=csr_matrix(book_pivot)
    model=NearestNeighbors(metric='cosine',algorithm='brute')
    model.fit(book_sparse)
    if request.method=='POST':
        title=request.form['book'].lower()
        #print(book_pivot.head(2))
        suggest=[]
        k=0
        try:
            k=1
            id=np.where(book_pivot.index==title)[0][0]
            distances, suggestions=model.kneighbors(book_pivot.iloc[id, :].values.reshape(1,-1), n_neighbors=8)
            # print(suggestions[0])
            book=[]
            for i in suggestions[0]:
                book.append(book_pivot.index[i])
            ids=[]
            for name in book:
                j=np.where(final_ratings['title']==name)[0][0]
                ids.append(j)
            urls=[]
            for i in ids: 
                url=final_ratings.iloc[i]['img']
                urls.append(url)
            


            
            # url=final_ratings.iloc[i]['img']
                
            suggest=urls
        except:
            
            k=0
            top10=final_ratings.sort_values('rating_count',ascending=False)
            top10.drop_duplicates(['title'],inplace=True)
            x=top10.head(10)['img']
            for i in x:
                suggest.append(i)
            
        
    return render_template("result.html",suggest=suggest,k=k,title=title.title())



if __name__=="__main__":
    app.run(debug=True)