# Генетический алгоритм

  
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
![alt text](https://github.com/elena111111/ML/blob/master/DecisionTree.pdf)

И так работает классифицируются точки по отобранным признакам с помощью дерева решений:
![alt text](https://github.com/elena111111/ML/blob/master/DecisionTreeClassifier.pdf)