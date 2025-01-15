# В данном репозитории реализованы две надстройки над библиотекой pykan:
# 1. class *KAN_sourse.KAN_es* : 
## Сеть Колмогорова-Арнольда (КАН) с остановкой по валидационному набору.
Для использования данного алгоритма используйте класс KAN_es и его метод train_es() из файла KAN_sourse. [es означает "early stopping"].

Предварительно следует установить библиотеку pykan, прописав в терминале: pip install pykan.

В ноутбуке "Spectroscopy_FL_4_ions/Spectroscopy.ipynb" показано использование KAN_es на примере решения задачи обратной спектроскопии. Для корректной работы ноутбука неоюходимо скачать исходные данные по ссылке [https://disk.yandex.ru/d/07uIsXOGteY3dg] и поместить их в папку "Spectroscopy_FL_4_ions" данного проекта.

![pic1_1](/Pictures/pic_1_1.png)

![pic1_2](/Pictures/pic_1_2.png)

Ссылка на примеры реализации КАН с помощью библиотеки pykan: https://kindxiaoming.github.io/pykan/index.html

Ссылка на git-репозиторий с pykan: https://github.com/KindXiaoming/pykan/tree/master


# 2. class *KAN_sourse.KANRegressor* :
## Сеть Колмогорова-Арнольда (КАН), обернутый в классы библиотеки *sci-kit learn* (*RegressorMixin*, *BaseEstimator*). 
Это делает возможным включение его в общий пайплайн *sci-kit learn*: прменение *cross_validate*, *grid_searche*, использование внутри *pipline*, и т.д..

В ноутбуке "Housing_tasks/california-housing-dataset.ipynb" *KANRegressor* применяется к тестовой задаче регрессии и сравнивается с референсными методами. Демонстрируется совместимость *KANRegressor* с методами и классами библиотеки *sci-kit learn*.

Источник ноутбука: https://www.kaggle.com/code/olanrewajurasheed/california-housing-dataset

Поскольку *sci-kit learn* для представления векторных данных использует *pytorch*, а *pykan*  - *tensorflow*, внутри *KANRegressor* приходится конвертировать эти типы данных друг в друга. В результате чего *KANRegressor* может оказаться неэффективным в работе по памяти и времени, особенно на больших датасетах.

Так, в задаче california-housing-dataset присутствует ~20k примеров (размерность входного пространства: 8); кросс-валидация с 10 фолдами *KANRegressor(hidden_layer_sizes=[3,])* на лэптопе с 11th Gen Intel Core i5 занимает 55 минут.

![pic2_1](/Pictures/pic_2_1.png)

![pic2_2](/Pictures/pic_2_2.png)

![pic2_2](/Pictures/pic_2_3.png)

![pic2_2](/Pictures/pic_2_4.png)

![pic2_2](/Pictures/pic_2_5.png)