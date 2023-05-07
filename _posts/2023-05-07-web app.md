##`ufo-model.pkl` 파일을 만들기 위한 코드 작성

### 데이터 정리

**데이터 적재**


```python
import pandas as pd
import numpy as np
```


```python
ufos = pd.read_csv('https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/3-Web-App/1-Web-App/data/ufos.csv')
```


```python
ufos.head()
```





  <div id="df-2bc6d12a-a39a-4c65-8480-98d9ab54621e">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>city</th>
      <th>state</th>
      <th>country</th>
      <th>shape</th>
      <th>duration (seconds)</th>
      <th>duration (hours/min)</th>
      <th>comments</th>
      <th>date posted</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/10/1949 20:30</td>
      <td>san marcos</td>
      <td>tx</td>
      <td>us</td>
      <td>cylinder</td>
      <td>2700.0</td>
      <td>45 minutes</td>
      <td>This event took place in early fall around 194...</td>
      <td>4/27/2004</td>
      <td>29.883056</td>
      <td>-97.941111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/10/1949 21:00</td>
      <td>lackland afb</td>
      <td>tx</td>
      <td>NaN</td>
      <td>light</td>
      <td>7200.0</td>
      <td>1-2 hrs</td>
      <td>1949 Lackland AFB&amp;#44 TX.  Lights racing acros...</td>
      <td>12/16/2005</td>
      <td>29.384210</td>
      <td>-98.581082</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/10/1955 17:00</td>
      <td>chester (uk/england)</td>
      <td>NaN</td>
      <td>gb</td>
      <td>circle</td>
      <td>20.0</td>
      <td>20 seconds</td>
      <td>Green/Orange circular disc over Chester&amp;#44 En...</td>
      <td>1/21/2008</td>
      <td>53.200000</td>
      <td>-2.916667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/10/1956 21:00</td>
      <td>edna</td>
      <td>tx</td>
      <td>us</td>
      <td>circle</td>
      <td>20.0</td>
      <td>1/2 hour</td>
      <td>My older brother and twin sister were leaving ...</td>
      <td>1/17/2004</td>
      <td>28.978333</td>
      <td>-96.645833</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/10/1960 20:00</td>
      <td>kaneohe</td>
      <td>hi</td>
      <td>us</td>
      <td>light</td>
      <td>900.0</td>
      <td>15 minutes</td>
      <td>AS a Marine 1st Lt. flying an FJ4B fighter/att...</td>
      <td>1/22/2004</td>
      <td>21.418056</td>
      <td>-157.803611</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2bc6d12a-a39a-4c65-8480-98d9ab54621e')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-2bc6d12a-a39a-4c65-8480-98d9ab54621e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2bc6d12a-a39a-4c65-8480-98d9ab54621e');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




**새로운 데이터 프레임 생성**


```python
ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})

ufos.Country.unique()
```




    array(['us', nan, 'gb', 'ca', 'au', 'de'], dtype=object)




```python
ufos.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 80332 entries, 0 to 80331
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   Seconds    80332 non-null  float64
     1   Country    70662 non-null  object 
     2   Latitude   80332 non-null  float64
     3   Longitude  80332 non-null  float64
    dtypes: float64(3), object(1)
    memory usage: 2.5+ MB


**결측치 제거 및 데이터 축소**


```python
ufos.dropna(inplace=True)

ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

ufos.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 25863 entries, 2 to 80330
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   Seconds    25863 non-null  float64
     1   Country    25863 non-null  object 
     2   Latitude   25863 non-null  float64
     3   Longitude  25863 non-null  float64
    dtypes: float64(3), object(1)
    memory usage: 1010.3+ KB


**Scikit-learn의 라이브러리를 사용하여 Country 열의 범주형 특성 값을 수치형 특성 값으로 변환**


```python
from sklearn.preprocessing import LabelEncoder

ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

ufos.head()
```





  <div id="df-49d76135-6fb1-44eb-a68c-36bf3f080ba5">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Seconds</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>20.0</td>
      <td>3</td>
      <td>53.200000</td>
      <td>-2.916667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.0</td>
      <td>4</td>
      <td>28.978333</td>
      <td>-96.645833</td>
    </tr>
    <tr>
      <th>14</th>
      <td>30.0</td>
      <td>4</td>
      <td>35.823889</td>
      <td>-80.253611</td>
    </tr>
    <tr>
      <th>23</th>
      <td>60.0</td>
      <td>4</td>
      <td>45.582778</td>
      <td>-122.352222</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3.0</td>
      <td>3</td>
      <td>51.783333</td>
      <td>-0.783333</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-49d76135-6fb1-44eb-a68c-36bf3f080ba5')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-49d76135-6fb1-44eb-a68c-36bf3f080ba5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-49d76135-6fb1-44eb-a68c-36bf3f080ba5');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




###모델 빌드

**로지스틱 회귀**


```python
from sklearn.model_selection import train_test_split

Selected_features = ['Seconds','Latitude','Longitude']

X = ufos[Selected_features].values
y = ufos['Country']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```


```python
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        41
               1       0.85      0.46      0.60       250
               2       1.00      1.00      1.00         8
               3       1.00      1.00      1.00       131
               4       0.97      1.00      0.98      4743
    
        accuracy                           0.97      5173
       macro avg       0.96      0.89      0.92      5173
    weighted avg       0.97      0.97      0.97      5173
    
    Predicted labels:  [4 4 4 ... 3 4 4]
    Accuracy:  0.9698434177459888


###모델 `pickle`


```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

    [3]




---



## pkl 파일 저장

**위의 코드를 `ufo-model.pkl` 파일로 저장**

### 파일 작성

[https://github.com/microsoft/ML-For-Beginners/blob/main/3-Web-App/1-Web-App/README.md](https://github.com/microsoft/ML-For-Beginners/blob/main/3-Web-App/1-Web-App/README.md)

위의 코드 및 아래의 파일들은 위의 사이트를 참고하여 만들었다.

## styles.css 파일 작성

```
body {
	width: 100%;
	height: 100%;
	font-family: 'Helvetica';
	background: black;
	color: #fff;
	text-align: center;
	letter-spacing: 1.4px;
	font-size: 30px;
}

input {
	min-width: 150px;
}

.grid {
	width: 300px;
	border: 1px solid #2d2d2d;
	display: grid;
	justify-content: center;
	margin: 20px auto;
}

.box {
	color: #fff;
	background: #2d2d2d;
	padding: 12px;
	display: inline-block;
}
```

**위의 코드를 `style.css` 파일로 저장**

## index.html 파일 작성

```
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>🛸 UFO Appearance Prediction! 👽</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}"> 
</head>

<body>
 <div class="grid">

  <div class="box">

  <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>

    <form action="{{ url_for('predict')}}"method="post">
    	<input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
      <input type="text" name="latitude" placeholder="Latitude" required="required" />
		  <input type="text" name="longitude" placeholder="Longitude" required="required" />
      <button type="submit" class="btn">Predict country where the UFO is seen</button>
    </form>

  
   <p>{{ prediction_text }}</p>

 </div>
</div>

</body>
</html>
```
**위의 코드를 `index.html` 파일로 저장**

## app.py 파일 작성

```
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("./ufo-model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )


if __name__ == "__main__":
    app.run(debug=True)
```

**위의코드를 `app.py` 파일로 저장**

## requirements.txt 파일 작성

```
scikit-learn
pandas
numpy
flask
```

**위의 코드를 `requirements.txt` 파일로 저장**


##실행 과정

빈 디렉터리 생성

빈 디렉터리에 **`app.py`, `requirements.txt`, `ufo-model.pkl`**  파일 저장 

빈 디렉터리에 2개의 하위 디렉터리 **`static`, `templates`** 생성

**`static`** 디렉터리에 하위 디렉터리 **`css`** 생성 후 **`css`** 디렉터리에 **`style.css`** 파일 저장 

**`templates`** 디렉터리에 **`index.html`** 파일 저장

명령 프롬프트를 사용하여 가상 환경을 생성하였고 생성한 가상 환경을 통해 웹앱을 구현하였다.

requirements.txt 파일은 파이썬 가상 환경에서 ufo-model.pkl 파일을 구현하기 위한 파이썬 라이브러리 설치 파일이다.

![구현.PNG](data:image/png;base64,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)

![실행.PNG](data:image/png;base64,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ACzJCAAAAEsyAgAAwJKMAAAAsCQjAAAALMkIAAAASzICAADAkowAAACwJCMAAAAsyQgAAABLMgIAAMCSjAAAALAkIwAAACzJCAAAAEsyAgAAwJKMAAAAsCQjAAAALMkIAAAASzICAADAkowAAACwJCMAAAAsyQgAAABLMgIAAMCSjAAAALAkIwAAACzJCAAAAEsyAgAAwJKMAAAAsCQjAAAALMkIAAAASzICAADAkowAAACwJCMAAAAsyQgAAABLMgIAAMCSjAAAALAkIwAAACzJCAAAAEsyAgAAwJKMAAAAsCQjAAAALMkIAAAASzICAADAkowAAACwJCMAAAAsyQgAAABLMgIAAMCSjAAAALAkIwAAACzJCAAAAEsyAgAAwJKMAAAAsCQjAAAALMkIAAAASzICAADAkowAAACwJCMAAAAsyQgAAABLMgIAAMCSjAAAALAkIwAAACzJCAAAAEsyAgAAwJKMAAAAsCQjAAAALMkIAAAASzICAADAkowAAACwJCMAAAAsyQgAAABLMgIAAMCSjAAAALAkIwAAACzJCAAAAEsyAgAAwJKMAAAAsCQjAAAALMkIAAAASzICAADAkowAAACwJCMAAAAsyQgAAABLMgIAAMCSjAAAALAkIwAAACzJCAAAAEsyAgAAwJKMAAAAsCQjAAAALMkIAAAASzICAADAkowAAACwJCMAAAAsyQgAAABLMgIAAMCSjAAAALAkIwAAACzJCAAAAEsyAgAAwJKMAAAAsCQjAAAALMkIAAAASzICAADAkowAAACwJCMAAAAsyQgAAABLMgIAAMCSjAAAALAkIwAAACzJCAAAAEsyAgAAwJKMAAAAsCQjAAAALMkIAAAASzICAADAkowAAACwJCMAAAAsyQgAAABLMgIAAMCSjAAAALAkIwAAAIz46+9/AC2XwBas3rW8AAAAAElFTkSuQmCC)

위의 사진은 styles.css 파일을 적용하며 구현한 화면이다.

![html 코드 실행 화면.PNG](data:image/png;base64,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)

위의 사진은 styles.css 파일 없이 구현된 화면이다
