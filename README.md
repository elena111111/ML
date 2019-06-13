# Генетический алгоритм - отбор признаков для дерева решений  

Создан класс *Individ* с методами:  

```python

__init__(self, n_feat, clf, train_data_X, train_data_Y, test_data_X, test_data_Y) - зависит от количества признаков, классификатора, данных для обучения и тестирования
get_fitness(self) - значение качества для данного индивида


not_equal(self, other) - сравнение


# отношение <=
def __le__(self, other):
 
	return (self.get_fitness() <= other.get_fitness()) and self.not_equal(other)

  

# скрещивание индивидов
  
def crossover(self, other):

	result = self

	for i in range(self._n_feat):

		if random.random() > 0.5:

			result._bin_feat[i] = other._bin_feat[i]
    
	return result  



# мутация с определенной вероятностью    
def mutation(self, probability):

	result = self

	for i in range(self._n_feat):
		if random.random() < probability:
                	result._bin_feat[i] = 1 - result._bin_feat[i]
        return result

```




Класс *Population*, основные методы:

```python

 __init__(self, n_pop, clf, train_data_X, train_data_Y, test_data_X, test_data_Y)
_sort(self) - сортировка индивидов по увеличению ключа - критерия качества (fitness)


selection(self) - отбор n_pop лучших индивидов


get_best_ind(self) - получить лучшего по критерию качества индивида


get_next_generation(self, prob) - получить следующее поколение при помощи операций скрещивания и мутации (с вероятностью prob), убрать повторы

```




Функция генетического алгоритма *GenAlg*:

```python

'''

clf - классификатор
n - численность популяции

T - максимальное число итераций

prob - вероятность мутации

d - максимальная разность между поколениями 

возвращает объект типа Individ, лучший по качеству в своей популяции
 (и для истории вернем лучших индивидов в разных поколениях + оценки)
'''

def GenAlg(clf, train_data_X, train_data_Y, test_data_X, test_data_Y, n, T, prob, d):

	p = Population(n, clf, train_data_X, train_data_Y, test_data_X, test_data_Y)
    	bestIndivid = p.get_best_ind()
    	bestIndivids, scores = [], [] #для статистики
    	bestNumPop = 0
    	for t in range(T):
        	p.selection()
        	if p.get_best_ind().get_fitness() < bestIndivid.get_fitness():
            		bestIndivid = p.get_best_ind()
            		bestNumPop = t
        	bestIndivids.append(bestIndivid.get_feat())
        	scores.append(bestIndivid.get_fitness())
        	#условие выхода
        	if t - bestNumPop >= d:
            		return bestIndivid, bestIndivids, scores
        	p.get_next_generation(prob)
    		return bestIndivid, bestIndivids, scores
```



**Пример работы: Генетический алгоритм - отбор признаков для дерева решений**  
Начальные значения
```python 
clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=17, min_samples_leaf=6)
ds = datasets.load_wine() # и перемешаем точки

задать k
train_data_X, train_data_Y = ds.data[:k, :], ds.target[:k]
test_data_X, test_data_Y = ds.data[k:, :], ds.target[k:]
n = 100 #размер популяции
prob = 0.2 #вероятность мутации
T = 200 #макс итер
d = 50 #допуст разница между поколениями
```

запуск: 
```python
bestInd, res, scores = GenAlg(clf_tree, train_data_X, train_data_Y, test_data_X, test_data_Y, n, T, prob, d) 
```

Процесс работы генетического алгоритма:  
по ox - признаки (лучшие в поколении)
, по oy - номера поколений


![alt text](https://github.com/elena111111/ML/blob/master/GenAlg.png)

Мы получили следующее дерево решений: 
![alt text](https://github.com/elena111111/ML/blob/master/DecisionTree.png)

И так работает классифицируются точки по отобранным признакам с помощью дерева решений:
![alt text](https://github.com/elena111111/ML/blob/master/DecisionTreeClassifier.png)


# LSA - латентно-семантический анализ

1. Имея множество документов, составим частотную матрицу индексируемых слов: 
строки соответствуют индексированным словам, а столбцы — документам. 
В каждой ячейке матрицы указано какое количество раз слово встречается в соответствующем документе или
вероятность встречаемости слова в документе. 

2. Сингулярное разложение полученной матрицы (выделяет ключевые составляющие матрицы, позволяя игнорировать шумы):
<a href="https://www.codecogs.com/eqnedit.php?latex=M&space;=&space;U&space;*&space;W&space;*&space;V^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?M&space;=&space;U&space;*&space;W&space;*&space;V^T" title="M = U * W * V^T" /></a>
, где <a href="https://www.codecogs.com/eqnedit.php?latex=U,&space;V^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?U,&space;V^T" title="U, V^T" /></a>
 - ортогональные матрицы, 
<a href="https://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W" title="W" /></a>
 - диагональная матрица.

Для визуализации возьмем двумерное сингулярное разложение.

**Пример (отображение документов на плоскость):**  

```python
# входные данные:  
ds = datasets.fetch_20newsgroups(subset='train', categories=categories, shuffle=False, random_state=42)
data = ds.data

# построение матрицы встречаемости слов в документах
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))  #неинформативные слова не рассматриваем
dtm = vectorizer.fit_transform(data)

# уменьшение размерности с использованием усеченного SVD
lsa = TruncatedSVD(2)   
dtm_lsa = lsa.fit_transform(dtm)

# отображаем 0 (по OX) и 1 стоблец (по OY) из dtm_lsa, для интереса оставим метки классов

# полученную матрицу dtm_lsa используем как обучающую выборку для алгортима кластреризации

dtm_lsa = np.asarray(dtm_lsa)

algorithm = cluster.SpectralClustering(
        n_clusters=len(categories), affinity='cosine', eigen_tol=0.0001)
y_pred = algorithm.fit_predict(dtm_lsa) # цвета новых классов (не соответствуют цветам старых классов)

# посмотрим, насколько похожи результаты
```

![alt text](https://github.com/elena111111/ML/blob/master/SpectralClustering.png)


# Информационный критерий Акаике (AIC)
