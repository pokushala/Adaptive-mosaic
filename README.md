# Мозайка

## Зависимости

* [ski-image](https://scikit-image.org/)
* [numpy](https://docs.scipy.org/doc/numpy/index.html)
* [scipy](https://www.scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [Pillow](https://pillow.readthedocs.io/en/stable/index.html)

## Данные с иконками 

Для тестирования работы набор иконок можно скачать [здесь](https://www.kaggle.com/danhendrycks/icons50).

## Пример запуска
```python
python .\mosaic.py --image путь_к_изображению --tiles-dir путь_к_директории_с_иконками
```
 Структура директории с иконками может быть любой. Используется рекурсивное сканирование и рассматриваются только файлы с раширением `.jpg`, `.png`.