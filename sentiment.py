from flask import Flask,request,render_template
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app=Flask(__name__)



model=tensorflow.keras.models.load_model('lstm_model.h5')



def prediction(x):
    ps=PorterStemmer()
    corpus=[]
    rev=re.sub('[^a-zA-Z]',' ',x)
    rev=rev.lower()
    rev=rev.split()
    rev=[ps.stem(word) for word in rev if not word in stopwords.words('english')]
    rev=' '.join(rev)
    corpus.append(rev)
    y=corpus[0]
    oh=[one_hot(y,10000)]
    ohp=pad_sequences(oh,padding='pre',maxlen=15)
    arr=np.array(ohp)
    pred=model.predict(arr)
    predm=np.argmax(pred,axis=1)
    output=""
    if predm[0]==0:
     output="Negative"
    elif predm[0]==1:
     output="Neutral"
    else:
     output="Positive"
    return output


@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        tweet=request.form['tweet']
        pred=prediction(tweet)
        return render_template('index.html',output='Sentiment of tweet is :{}'.format(pred))

if __name__=='__main__':
   app.run(debug=True)

