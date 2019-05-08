# Генетический алгоритм

Рассматривается простой пример генетического алгоритма, работающий с бинарными признаками и минимизирующий их сумму.

Создан класс *Individ* с методами: 
```python
__init__(self, n_feat) - зависит от количества признаков и генерирует их (возможна передача минимизируемой функции - критерия качества)

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
            result._features[i] = other._features[i]
    return result

# мутация с определенной вероятностью
def mutation(self, probability):
    result = self
    for i in range(self._n_feat):
        if random.random() > probability:
            result._features[i] = 1 - result._features[i]
    return result
```

Класс *Population*, основные методы:
```python
__init__(self, n_pop, n_feat) - зависит от размера популяции и количества признаков (возможна передача минимизируемой функции - критерия качества)

_sort(self) - сортировка индивидов по увеличению ключа - критерия качества (fitness)

selection(self) - отбор n_pop лучших индивидов

get_best_ind(self) - получить лучшего по критерию качества индивида

get_next_generation(self, prob) - получить следующее поколение при помощи операций скрещивания и мутации (с вероятностью prob), убрать повторы
```

Функция генетического алгоритма *GenAlg*:
```python
'''
n - численность популяции
m - количество признаков
T - максимальное число итераций
prob - вероятность мутации
d - максимальна разность между поколениями
возвращает объект типа Individ, лучший по качеству в своей популяции
'''
def GenAlg(n, m, T, prob, d):
    p = Population(n, m)
    bestIndivid = p.get_best_ind()
    bestNumPop = 0
    for t in range(T):
        p.selection()
        p.print()
        if p.get_best_ind().get_fitness() < bestIndivid.get_fitness():
            bestIndivid = p.get_best_ind()
            bestNumPop = t
        bestIndivids.append(bestIndivid.get_feat())
        if t - bestNumPop >= d:
            return bestIndivid
        p.get_next_generation(prob)
    return bestIndivid
```

**Пример работы:**
по ox - признаки (лучшие в поколении)
по oy - номера поколений

![alt text](https://github.com/elena111111/ML/blob/master/GenAlg.png)