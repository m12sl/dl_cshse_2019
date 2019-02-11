# Полезные ссылки по питону

Гайд по python/numpy http://cs231n.github.io/python-numpy-tutorial/
Хорошие практики по pytorch: https://pytorch.org/docs/stable/notes/cuda.html

Все сложные вещи стоит писать в отдельных скриптах. 

Для отладки без принтов есть интерактивные отладчики: pdb/ipdb https://docs.python.org/3/library/pdb.html https://hasil-sharma.github.io/2017/python-ipdb/

Для профилирования потребления памяти есть специальный инструмент: https://pypi.org/project/memory-profiler/


Когда вы работаете с pytorch-тензорами будьте осторожны насчет утечек памяти:


```

log = []
for x, y in dataloader:
    output = model(x)
    loss = F.nll_loss(output, y) # << это тензор, со всеми предками для автоматического дифференцирования
    log.append(loss)  # << постоянный расход памяти
```
Используйте `loss.item()` чтобы получить только скаляр.


## Отладка моделек в pytorch

Вы можете нарисовать граф с помощью пакета pytorchviz и убедиться, что код отражает ваши намерения:

https://github.com/szagoruyko/pytorchviz/blob/master/examples.ipynb
