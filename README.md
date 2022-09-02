# Ispit

1.09. - Izdvajanje najbitnijih fragmenata koda i anki
2.09. - Provezbavanje svesaka koje ce doci i anki

Sa roka
```python
# Prebroj vrednosti
category_counts = data['category'].value_counts()
basic_counts = category_counts['Basic']
luxury_counts = category_counts['Luxury']

# Barplot sa vrednostima
plt.bar(x=[0, 1], height =[basic_counts, luxury_counts])
plt.xticks(ticks=[0,1], labels=["basic", "luxury"])
plt.show()

# Ispis procenata
plt.pie(x=has_pool_counts, labels=["no pool", "has pool"], autopct='%.4f')
plt.plot()


# Proširiti skup atributa dodajući mu četvrtu dimenziju koja odgovara monohromatskim slikama (eng. **channel last** notacija). Prikazati dimenziju matrice atributa nakon ove transformacije. 
data.shape # (2000, 64, 64)
data = data[...,np.newaxis]
data.shape # (2000, 64, 64, 1)

# y_test = # niz vektora verovatnoca
np.argmax(y_test, axis=1)
# inverz za keras.utils.to_categorical

history.history['loss']
history.history['val_loss']

history.history['accuracy']
history.history['val_accuracy'']
#Prikazati grafik promena funkcije gubitka i preciznosti u toku treniranja
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title('Loss')
plt.plot(np.arange(0, epochs), history.history['loss'], label='train')
plt.plot(np.arange(0, epochs), history.history['val_loss'], label='val')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.plot(np.arange(0, epochs), history.history['accuracy'], label='train')
plt.plot(np.arange(0, epochs), history.history['val_accuracy'], label='val')
plt.legend(loc='best')

plt.show()


# Konvertovati kolonu kategorije u numericke vrednosti
data['Species'] = data['Species'].astype('category').cat.codes

```

## Provezbati
05 klasifikacija -> 02 Logisticka regresija
10 mreze -> sve
13 kmeans
14 PCA

Na kraju:
05 klasifikacija -> 03, 04 sms poruke

1. Linearna regresija/klasifikacija
2. Neuronska mreza klasifikacija/regresija
3. PCA/kmeans

U jun2 je bilo
1. Linearna regresija
2. Mreza klasifikacija
3. PCA

Sep1 ce verovatno biti:
1. Linearna klasifikacija
2. Mreza regresija
3. KMeans

Prvo bi trebalo da naucim:
KMeans
Linearna klasifikacija
Mreza regresija
PCA
Linearna regresija
Mreza klasifikacija

# Vazna funkcije

## Obrada podataka

```python
data = datasets.load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=["MEDV"])
X.info()
X.describe()
X.hist(figsize = [15,15])
X.corr() # matrica korelacije

y = data['Outcome']
X = data.drop(columns=['Outcome'])

# Standardizacija
# X_train i X_test skup standardizuje se istim scaler objektom treniranim nad X_train podacima. y_train i y_test se ne standardiziuju, ostaju isti
# Jos neki skaleri: RobustScaler, MinMaxScaler, Normalizer, MinAbsScaler, MaxAbsScaler
# 1. Biblioteka
scaler = preprocessing.StandardScaler()
scaler.fit(X_train) # scaler.fit(X_train.reshape(-1, 1))
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
scaler.mean_
scaler.var_
# 2. Rucno
X_means = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train = (X_train-X_means)/X_std
X_test = (X_test-X_means)/X_std


np.bincount(y_train) # prepbroj instance

x = np.linspace(1, 5, N).reshape(N, 1)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.4, random_state=7, stratify=y)

# Podela na train, test, i validacioni
X_train_and_validation, X_test, y_train_and_validation, y_test = model_selection.train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y )
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X_train_and_validation, y_train_and_validation, train_size = 0.8, random_state = 42, stratify = y_train_and_validation )
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)

plt.plot(x, y, 'o')
plt.plot(x, f(b0, b1, x))
plt.show()


# subplot
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
...
plt.subplot(1, 3, 2)
plt.scatter(x_train_correct.flatten(), y_train, color='red', marker = 'v')
...
plt.show() # samo na kraju jednom



# izravatni slike

X_train = X_train.reshape(X_train.shape[0], image_width*image_height).astype('float32')
X_test = X_test.reshape(X_test.shape[0], image_size*image_size)
X_train /= 255. # svodjenje na 0,1
X_test /= 255. # svodjenje na 0,1
```

## Pravljenje modela


```python
# Cuvanje modela
import pickle
with open(model_filename, 'wb') as pickle_file:
    pickle.dump(model, pickle_file)

with open(model_filename, 'rb') as pickle_file:
    model_revived = pickle.load(pickle_file)
```


## Evaluacija


### R^2 score
Koeficijent determinacije $R^2$ izračunava udeo varijanse ciljne promenljive koji je objašnjen naučenim modelom (kako varira vrednosti ciljne promenljive kada variraju vrednosti atributa).
   
$R^2 = 1 - \frac{\sum_i{(y_i-\hat{y_i})^2}}{\sum_i{(y_i - \bar{y})^2}}$

Vrednosti koeficijenta determinacije su iz intervala ${(-\infty, 1]}$ i vrednosti bliže jedinici označavaju bolje modele. 

```python
def r_squared(y_true, y_predicted):
    u = ((y_true - y_predicted)**2).sum()
    y_mean = y_true.mean(); 
    v = ((y_true - y_mean)**2).sum()
    return 1 - u/v

r2 = metrics.r2_score(y_test, y_predicted)
r2
```

### Srednje kvadratna greska
$MSE = \frac{1}{n} \sum_i{(y_i-\hat{y_i})^2}$
```python
def mean_squared_error(y_true, y_predicted):
    return ((y_true - y_predicted)**2).mean()

mse = metrics.mean_squared_error(y_test, y_predicted)
mse
```

### Unarksna validacija
```python
accuracy_scores = model_selection.cross_val_score(pipeline|model, X, y, cv=5, scoring='accuracy')

```
### Metrics

```python
metrics.accuracy_score(y_test, y_test_predicted)
metrics.precision_score(y_test, y_test_predicted)
metrics.recall_score(y_test, y_test_predicted)
metrics.f1_score(y_test, y_test_predicted)
metrics.classification_report(y_test, y_test_predicted)
metrics.confusion_matrix(y_test, y_test_predicted)
metrics.roc_auc_score(y_test, y_predicted)
metrics.plot_roc_curve(logistic_regression, X_test, y_test)
```

Verovatnoce za svaku instancu:
```python
y_probabilities_predicted = model.predict_proba(X_test)

y_log_probabilities_predicted = model.predict_log_proba(X_test)
```

# Casovi

## 01

## 02

## 03 Linearna regresija 1
### 01 Pdosetnik verovatnoca i statistika
### 02 O linearnoj regresiji
```python
def f(b0, b1, x):
    return b0 + b1*x

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
model.coef_ # koeficijenti uz Beta
model.intercept_ # pomeraj


model.predict([[1.2], [1.3], [1.4]])
data = np.array([1,2,3,4,5,6,7])
# reshape(-1,1); -1 transponuje, 1 stavlja svaki element u zagradu
model.predict(data.reshape(-1,1))
model.score(x_test, y_test)
```

### 03 Algebarsi i optimizaconi pristup lienarnoj regresiji
Mur-Penrouzovo inverz
$P=X(X^TX)^{-1}X^T$
* matrica je simetrična
* matrica je idempotentna tj. važi PP = P

**Dodati jos jednu kolonu na pocetak sa svim jedinicama**
```
M = np.vstack([np.ones(train_size), x_train.ravel()]).transpose()
coef_pseudo = np.linalg.pinv(M).dot(y_train)
print('Koeficijenti modela: ', coef_pseudo[0][0], coef_pseudo[1][0])
```

**Ne moze da se primeni kod lose uslovljenih matrica**

$cond(A) = ||A||\cdot||A||^{-1}$

Provera uslovljenosti: `np.linalg.cond(A)`

`Ax=c` resava se
```
solution, _, _, _ = np.linalg.lstsq(A, c, rcond=None)
```

**Korelisani podaci => velika uslovljenost matrice**

TODO(mspasic): Onda ide ceo pristup lineranoj regresije gradijentnim spustom. Ovo uci ako ostane vremena.

## 04 Linearna regresija 2

```python
# Ispisati feature i coef
model.coef_.ravel() # razvija u 1D niz
for feature, coef in zip(data.feature_names, model.coef_.ravel()):
    print (feature, coef)
```

### + 01-Linearna regresija - primer primene nad realnim podacima.ipynb
### + 02-Standardizacija.ipynb
Sve upisano u obradu podataka

```python
x_train_correct = scaler.transform(x_train.reshape(-1, 1))
x_test_correct = scaler.transform(x_test.reshape(-1, 1))
plt.scatter(x_train_correct.flatten(), y_train, color='red', marker = 'v', label='train')
plt.scatter(x_test_correct.flatten(), y_test, color='blue', marker = 'o', label='test')
plt.legend(loc='best')
plt.show()

```
### 03-Linearna regresija sa tezinama.ipynb

```python
# Dovoljno je ovo
logistic_regression_with_weights = linear_model.LogisticRegression(class_weight='balanced')

# Rucno
from sklearn import utils
class_labels = np.unique(y_train)
number_of_classes = class_labels.size 
utils.class_weight.compute_class_weight('balanced', class_labels, y_train)
weights = y_train.size /(number_of_classes * np.bincount(y_train))
weights
```
**Linearna regesija samo sa tezinama koje se prosledjuju funkciji fit pri treniranju modela**  
Zadatak ce vec reci kako se racuna tezina koja se prosleduje. 

**Heteroskedastičnost i homoskedastičnost**

Heteroskedasticnost: Varijansa var(y|X) zavisi od X
Homoskedasticnost: Varijansa var(y|X) konstantna za X

Obicna linearna regresija ne radi, mora da se koristi tezinska linerana regresija.

**Tezine: reciprocna vrednost varijansa**

Prilikom poziva metoda <code>fit()</code> težine pojedinačnih instanci zadaju se parametrom <code>sample_weight</code>. U našem primeru instancama kod kojih je mala varijansa ciljne promenljive treba pridružiti veće težine kako bi model bio sigurniji, a instancama kod kojih je varijansa velika manje težine. To možemo postići uzimanjem reciprocne vrednosti varijanse $\omega_i = \frac{1}{\sigma_i^2}$ za vrednosti težina.

Šum biramo iz normalne raspodele $N(0, 1+\frac{1}{2}y^2)$, što znači da će za veće apsolutne vrednosti ciljne promenljive šum biti veći.

```python
x = np.random.normal(0, 90, N).reshape(N, 1)
noise = np.random.normal(0, 1 + 0.5*(3 - 2*x)**2)
y = 3 - 2*x + noise
improved_model = linear_model.LinearRegression()
def f(x):
    return 3 - 2*x

### BEGIN GLAVNO!
weights = 1 / (1 + 0.5*f(x)**2) # Zato sto je 1 + 0.5y^2 varijansa normalne raspodele suma
improved_model.fit(x, y, sample_weight=weights.ravel())
### END GLAVNO! proslediti tezine

# Grafik obicnih rezidula
plt.scatter(y, y - model.predict(x))
plt.title('Grafik reziduala')
plt.show()

# Grafik otezanih rezidula
plt.scatter(y, weights * (y - improved_model.predict(x)))
plt.title('Grafik otežanih reziduala')
plt.show()

```


### 04-Linearna regresija - analiza atributa.ipynb
Korelacija atributa:
```python
corrcoefs = []
for feature in data.feature_names:
    corrcoef = np.corrcoef(X[feature].values, y)[0, 1]
    corrcoefs.append(corrcoef)
    print(feature, corrcoef)

correfs = pd.DataFrame(corref, columns=['feature', 'corref'])
correfs.sort_values(inplace=True, by=['corref'])
number_of_features = correfs.shape[0]
correfs.plot(kind='barh')
plt.title('korelacija')
plt.yticks(np.arange(number_of_features), list(correfs['feature'].values))

# izvdjanje onih vecih od nula
correfs[correfs['corref'] > 0] 

# Funkcija pandas
X.corr()

# Primena filtera na matricu korelacije
def correlation_strength(value, threshold=0.5):
    if value < threshold:
        return 0
    return value
X.corr().applymap(correlation_strength)
```

Rekurizvna elminiacija atributa: `feature_selection.RFE`

```python
rfe = feature_selection.RFE(model, n_features_to_select=10, verbose=1)
rfe.fit(X_train_scaled, y_train)
rfe.support_
keep_features = feature_names[rfe.support_]
discarded_features = feature_names[~rfe.support_]
rfe.ranking_
X_train_rfe = rfe.transform(X_train_scaled)
X_test_rfe = rfe.transform(X_test_scaled)
model.fit(X_train_rfe, y_train)
model.score(X_test_rfe, y_test)
```


## 05 Klasifikacija
### 01-Klasifikacija.ipynb
Za evaluaciju klasifikacionih modela najčešće se koriste tačnost (engl. accuracy), preciznost (engl. precision), odziv (engl. recall) i F1 mera. Ove mere će zbog jednostavnosti biti definisane u terminima binarne klasifikacije. Za slučaj višeklasne klasifikacije ove mere se izračunavaju za svaku od klasa pojedinačno po principu one-vs-rest.  

`Tačnost` predstavlja ocenu ukupnog broja uspešno klasifikovanih instanci i izračunava se po formuli: $$Acc = \frac{TP + TN}{TP + FN + FP + TN}$$
Nije dobra mera za nebalansirane klase.

`Preciznost` predstavlja ocenu broja pozitivno klasifikovanih instanci i izračunava se po formuli: $$P = \frac{TP}{TP + FP}$$

`Odziv` predstavlja ocenu broja prepoznatih pozitivnih instanci i računa se po formuli: $$R = \frac{TP}{TP + FN}$$

 Preciznost i odziv se gotovo uvek razmatraju zajedno:
- visoka preciznost a nizak odziv modela ukazuje da model retko klasifikuje instance kao pozitivne, i samim tim još manje pravi grešku da neku instancu koja je negativna proglasi za pozitivnu, ili
- visok odziv a niska preciznost modela ukazuje da model često klasifikuje instance kao pozitivne, i time retko za neku instancu koja je pozitivna predviđa negativnu klasu.

`F1 mera` kombinuje preciznost i odziv i predstavlja njihovu harmonijsku sredinu: 

$$F1 = 2\frac{Prec \cdot Rec}{Prec + Rec}$$



### 02-Logistička regresija.ipynb

Priakz vaznosti atributa:

```python
N = len(data.feature_names)
values = model.coef_[0]
plt.figure(figsize=(10, 5))
plt.bar(np.arange(0, N), values)
plt.xticks(np.arange(0, N), data.feature_names, rotation='vertical')
plt.show()
```

Izdvajanje instanci:
```python
TP
TP_mask = (y_test == 1) & (y_test_predicted == 1)
FP_mask = (y_test == 0) & (y_test_predicted == 1)
TN_mask = (y_test == 0) & (y_test_predicted == 0)
FN_mask = (y_test == 1) & (y_test_predicted == 0)

TP_indexes = np.where(TP_mask == True)

# sve instance TP
X_test[TP_indexes] 


# Instance sortirane po verovatnoci da su pozitivne
y_probabilities_predicted = model.predict_proba(X_test)
y_probabilities_predicted[0:10]
model_confidence = []
for p in y_probabilities_predicted:
    model_confidence.append(np.abs(p[0] - p[1]))
model_confidence = np.array(model_confidence)
model_confidence.argsort() 
```

### 03-Rad sa tekstom.ipynb
### 04-Klasifikacija poruka.ipynb

## 06-nebalansirani-skupovi-viseklasna-klasifikacija
### 01-Nebalansirani skupovi podataka.ipynb

Postoji nekoliko klasa za pripremu nebalnasiranog skupa podataka.
Ponasaju se kao StandardScaler.
Podaci se puste kroz njih i onda se koriste za treniranje i ocenu...
Logisticka regresija sa tezinama

```python

smote = imblearn.over_sampling.SMOTE(random_state=0, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


ratio = 1/5
ros = imblearn.over_sampling.RandomOverSampler(random_state=0, sampling_strategy=ratio)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
y_predicted_oversampled = logistic_regression_oversampled.predict(X_test)
metrics.confusion_matrix(y_test, y_predicted_oversampled)
metrics.roc_auc_score(y_test, y_predicted_oversampled)

from sklearn import utils
class_labels = np.unique(y_train)
number_of_classes = class_labels.size 
utils.class_weight.compute_class_weight('balanced', class_labels, y_train)
weights = y_train.size /(number_of_classes * np.bincount(y_train))
```
### 02-Višeklasna klasifikacija.ipynb

Pretvaranje obicnog klasifikatora u viseklasni klasifikator
```python
ovr_classifier = multiclass.OneVsRestClassifier(svm.LinearSVC())
ovo_classifier = multiclass.OneVsOneClassifier(svm.LinearSVC())
```
## 07-kerneli

### 01-Kernel trik.ipynb
Nema nista bitno

### 02-Poredjenje linearnog i kernelizovanog SVMa.ipynb
```python
linear_svm = svm.LinearSVC(loss='hinge', C=1.0)
linear_svm.fit(X_train, y_train)

kernelized_svm = svm.SVC(kernel='rbf', gamma=1, C=1.0)
kernelized_svm.fit(X_train, y_train)

print("Broj potpornih vektora: ", kernelized_svm.support_vectors_.shape[0])
print("Broj potpornih vektora po klasama: ", format(kernelized_svm.n_support_))
print("Broj instanci u skupu za treniranje: ", X_train.shape[0])

for (X1_i, X2_i) in zip(X1, X2):
    plt.scatter(X1_i, X2_i, c = ['blue' if f(X1_i, X2_i)==-1 else 'orange'])
plt.scatter(kernelized_svm.support_vectors_[:, 0], kernelized_svm.support_vectors_[:, 1], c = ['red'])
```
### 03-Kernelizovani SVM podešavanje hiperparametara.ipynb
Za sve kombinacije trenieramo model
Cuvamo najbolje parametre
```python
X_train_and_validation, X_test, y_train_and_validation, y_test = model_selection.train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y )
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X_train_and_validation, y_train_and_validation, train_size = 0.8, random_state = 42, stratify = y_train_and_validation )
for C in Cs:
    for gamma in gammas:
        model = svm.SVC(kernel='rbf', gamma=gamma, C = C)
        model.fit(X_train, y_train)
        f1_score = metrics.f1_score(y_validation, model.predict(X_validation))
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_C = C
            best_gamma = gamma
```
### 04-Nadaraja-Votson regresija.ipynb
Nema nista korisno.

### 05-Ocena gustine raspodele.ipynb
Nema nista korisno.
## 08-stabla-odlucivanja
### 00-Pipelines.ipynb

```python
linreg_pipeline = pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression(C=1/2))
#ili
linreg_pipeline = pipeline.Pipeline(steps=[('scaler', preprocessing.StandardScaler()), ('linreg', linear_model.LogisticRegression())])
linreg_pipeline.set_params(linreg__C=2)
linreg_pipeline['linreg'].coef_

linreg_pipeline.fit(X_train, y_train)
linreg_pipeline.score(X_test, y_test)
```
### 03-Stabla odlučivanja i slučajne šume.ipynb
```python
model = tree.DecisionTreeClassifier(criterion='gini', max_features=0.9, max_depth=3, random_state=7)
# dalje kao i svaki drugi klasifikator

# stampanje stabla
plt.figure(figsize=(20, 10))
tree.plot_tree(model, fontsize=12, feature_names=list(X.columns), filled=True, rounded=True, class_names=['0','1'])

# Vaznost featurea
plt.barh(list(X.columns), model.feature_importances_)
plt.show()

# sume
from sklearn import ensemble
model_forest = ensemble.RandomForestClassifier(n_estimators=20, max_depth=3, random_state=7)
plt.barh(list(X.columns), model_forest.feature_importances_)
plt.show()
```
### 04-Prosta agregacija (engl. bagging).ipynb

```python
model_tree = ensemble.BaggingClassifier(tree.DecisionTreeClassifier(random_state=42), n_estimators=500, max_samples=100, bootstrap=True, random_state=42)
model_knn = ensemble.BaggingClassifier(neighbors.KNeighborsClassifier(n_neighbors=5, metric='cosine'), n_estimators=100, max_samples=0.9, bootstrap=False, random_state=42)
```
### 05-Pojačavanje (engl. boosting).ipynb
```python
model_adaboost= ensemble.AdaBoostRegressor(base_estimator=tree.DecisionTreeRegressor(max_depth=3), n_estimators=100, random_state=7)
plt.plot(range(0, model_adaboost.n_estimators), model_adaboost.estimator_errors_)
plt.show()
plt.barh(data.feature_names, model_adaboost.feature_importances_, color='orange')
plt.show()

model_xgboost = xgboost.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)
plt.figure(figsize=(10, 5))
ax = plt.subplot(1, 1, 1)
xgboost.plot_importance(model_xgboost, ax= ax, xlabel='Tezina atributa', ylabel=None, title='Vaznost atributa').set_yticklabels(data.feature_names)

plt.figure(figsize=(20, 15), dpi=180)
ax = plt.subplot(1, 1, 1)
xgboost.plot_tree(model_xgboost, num_trees=99, ax=ax, rankdir='LR')
plt.show()

# XGBoost modeli se mogu dodatno ubrzati radom sa specifičnim matricama koje se predstavljaju strukturom `DMatrix`. Za njihovo kreiranje se može koristiti `DMatrix` omotač.
train_data = xgboost.DMatrix(data=X_train, label=y_train, feature_names=data.feature_names)
test_data = xgboost.DMatrix(data=X_test, label=y_test, feature_names=data.feature_names)

#Praksa je da se uz ovako pripremljene podatke umesto funkcije `fit` koristi funkcija `train`, a da se za evaluaciju koristi funkcija `eval`. Objedinjeni parametri ovih funkcija zapisani u formi rečnika se prosleđuju funkciji `train`.
xboost_model_dm = xgboost.train({
    'objective':'reg:squarederror', 
    'n_estimators':100, 
    'max_depth':3, 
    'eval_metric':['rmse']}, train_data)
xboost_model_dm.eval(test_data, name='test')
```
## 09-evaluacija-regularizacija
### 01-Selekcija i evaluacija modela.ipynb
```python
# Bez hiperparametara
X = data.data
y = data.target
kf = model_selection.KFold(n_splits=10)
accuracy_scores = model_selection.cross_val_score(svm.SVC(), X, y, scoring='accuracy', cv=kf)


svc_pipeline =  pipeline.make_pipeline(preprocessing.StandardScaler(), svm.SVC())
accuracy_scores_with_pipeline = model_selection.cross_val_score(svc_pipeline, X, y, scoring='accuracy', cv=kf)
accuracy_scores_with_pipeline

accuracy_scores_with_cv = model_selection.cross_val_score(svc_pipeline, X, y, scoring='accuracy', cv=10)

# Sa hiperparametrima

```
### 02-Implementacije funkcija za evaluaciju.ipynb
### 03-Regularizacija.ipynb
### 04-Regularizacija.ipynb

## 10-mreze
### 01-Keras-binarna klasifikacija.ipynb

```python
# Pravljenje
model = Sequential()
model.add(InputLayer(input_shape=(8,)))
model.add(Dense(units=15, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=n_clases, activation='sigmoid'))
n_clases je broj klasa
Ako je binarna klasifikacija onda se stavi sigomid funkcija i jedan neuron, ako je viseklasna, onda units bude broj klasa
# Za klasifikaciju loss funkcija je binary_crossentropy
#Treniranje
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=NUMBER_OF_EPOCHS, batch_size=BATCH_SIZE, verbose=0, validation=0.1)



plt.title('Treniranje mreže - funkcija gubitka')
plt.xlabel('Epoha')
plt.ylabel('Funkcija gubitka')
plt.plot(history.epoch, history.history['loss'])
plt.show()

plt.title('Treniranje mreže - tačnost modela')
plt.xlabel('Epoha')
plt.ylabel('Tačnost')
plt.plot(history.epoch, history.history['accuracy'])
plt.show()

# Evaluacija
test_scores = model.evaluate(X_test, y_test)
train_scores = model.evaluate(X_train, y_train)

y_predicted = model.predict(X_test)
metrics.r2_score(y_predicted, y_test)

# Unakrsna validacija
classifier = KerasClassifier(build_fn=baseline_model, nb_epoch=50, batch_size=32, verbose=0)
pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', classifier)])
accuracy_scores = model_selection.cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
accuracy_scores
np.average(accuracy_scores)
```


### 02-Keras regresija.ipynb

```python
# Arhitektura modela
from tensorflow.keras import utils
utils.plot_model(model, to_file='model_keras_regresija.png', show_shapes=True)

# Cuvanje modela na disku
from tensorflow.keras.models import load_model
model.save('models/boston_housing.hdf5')
model_revived = load_model('models/boston_housing.hdf5', compile=False)


# Rano zaustavljenje
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(X_train, y_train, epochs=NUMBER_OF_EPOCHS, batch_size=BATCH_SIZE, verbose=0, validation_split=0.1, \
                    callbacks=[early_stopping_callback])
```

### 03-Keras-višeklasna klasifikacija.ipynb

```python
# U kategoricke podatke
y_train = keras.utils.to_categorical(y_train, number_of_classes)
y_test = keras.utils.to_categorical(y_test, number_of_classes)
model = Sequential([
    InputLayer(input_shape=(image_size*image_size,)),
    Dense(units=128, activation='relu'), 
    Dense(units=64, activation='relu'),
    Dense(units=number_of_classes, activation='softmax')
])
model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Jedan način da se doda regularizacija modelu je uz korišćenje L2 regularizacije na nivou slojeva mreže.
regularizer = regularizers.l2(l=0.01)
model_with_regularization = Sequential([
    InputLayer(input_shape=(image_size*image_size,)),
    Dense(units=128, activation='relu', kernel_regularizer=regularizer), 
    Dense(units=64, activation='relu', kernel_regularizer=regularizer),
    Dense(units=number_of_classes, activation='softmax')
])
```
## 11-CNN
### 01-MNIST CNN klasifikacija.ipynb
### 02-Covnet1D klasifikacija teksta.ipynb

## 12-autoenkoder-GAN
### 02-Autoenkoderi-dodatak.ipynb
### 02-Autoenkoderi.ipynb
### 03-GAN.ipynb

## 13 RNN-Kmeans
### 01-RNN uvod.ipynb
### 02-RNN analiza sentimenata.ipynb
### 03-RNN vremenske serije.ipynb
### 04-k-means.ipynb
### 05-k-means i kompresija slike.ipynb

## 14 PCA
### 03-Analiza glavnih komponenti.ipynb

1. Rucno  
1.1. 

2. Biblioteka
2.1. preprocessing.StandardScaler
2.2. decomposition.PCA

```python
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

plt.xlabel('Prva glavna komponenta')
plt.ylabel('Druga glavna komponenta')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np.where(y==0, 'red', 'blue'))
plt.show()

# medjsuboni odnos komponenti
pca.components_

# udeo varijanse koji objasnjava svaka komponenta
pca.explained_variance_ratio_

# odredjivanje polaznih atributa
pca.inverse_transform([10,7])

# matrica kovaijrance
pca.get_covariance()

# Grafikon komulativne sume objasnjenje varijanse
plt.bar(np.arange(pca.n_components_), pca.explained_variance_ratio_)
plt.bar(np.arange(pca.n_components_), np.cumsum(pca.explained_variance_ratio_), fill=False)
plt.xticks(np.arange(pca.n_components_))
plt.show()

```
### 04-Analiza glavnih komponenti i klasifikacija.ipynb

```python



```

# Linearna regresija




# 13 KMeans

## Vazni atributi
```
kmeans.labels_ # labele svake tacke
kmeans.clusters_centers_ # centri klasterovanja

X = np.array([[0, 0], [12, 3]])
X_predicted = kmeans.predict(X)
plt.scatter(X[:, 0], X[:,1], c=X_predicted)

```

## Vazne funkcije
```

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
labels_predicted = kmeans.predict(data)

plt.scatter(data[:,0], data[:,1], c=kmeans.labels_)
for cx, cy in kmeans.cluster_centers_:
    plt.scatter(cx, cy, marker='x')


result = kmeans.fit(X)
result.score(X) # negativna vrednost funkcije cilja

Ks = range(1,10)
kms = [KMeans(n_clusters=i, random_state=7) for i in Ks]
score = [km.fit(X).score(X) for km in kms]

metrics.silhouette_samples(X, y_predicted) # ocena za svaku instancu pojedinacno
metrics.silhouette_score(X, y_predicted, random_state=7) # ocena za sve klastere


# Kompresovanje slike
n_colors = 64
pixel_sample = 10000
image_2D_sample = shuffle(image_2D, random_state=0)[:pixel_sample] # 10000 nasumicnih piksela
kmeans = KMeans(n_clusters=n_colors, random_state=0)
kmeans.fit(image_2D_sample)
```
# Pomocne funkcije

```

x = np.linspace(1, 5, N).reshape(N, 1)

from sklearn import metrics
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, n_features=2, centers=4, random_state=6) # pravi klastere
plt.scatter(X[:, 0], X[:,1], c=y)

plt.plot(X, Y) # grafik funkcije 

# Normalizacija piksela
image = image.astype(np.float64) / 255 
image_width, image_height, image_depth = image.shape

# preoblikovanje u vektor
image_2D = np.reshape(image, (image_width*image_height, image_depth))

```