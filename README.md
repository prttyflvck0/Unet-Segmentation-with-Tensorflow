# Unet-Segmentation-with-Tensorflow
This is a repository for building facades segmentation using CMP facade dataset and unet architecture of neural network with Tensorflow as a background.
Dataset: https://cmp.felk.cvut.cz/~tylecr1/facade/

Современные технологии искусственного интеллекта (ИИ) применяются в различных сферах, включая строительство, проектирование, эксплуатацию и продажу квартир. Некоторые примеры применения ИИ включают проектирование и моделирование зданий, оптимизацию производственных процессов, управление энергопотреблением зданий, прогнозирование цен на недвижимость и рекомендательные системы для покупателей недвижимости.

Сегментация изображений фасадов зданий и подсчет количества окон являются задачами компьютерного зрения, для решения которых применяются различные методы и подходы.

Одним из наиболее распространенных методов является метод с использованием сверточных нейронных сетей (CNN). В этом методе используются слои свертки и субдискретизации, которые позволяют выделять признаки изображения и создавать более абстрактные представления. Для задачи сегментации фасадов зданий, CNN может быть обучена на размеченном наборе данных, где каждый пиксель на изображении помечен как часть фасада или не является частью фасада.

Для подсчета количества окон на фасаде здания можно использовать различные методы, в том числе методы, основанные на алгоритмах сегментации изображений. Например, можно использовать CNN для сегментации фасада здания, а затем использовать дополнительные алгоритмы для выделения оконных отверстий на сегментированном изображении. Другой подход может заключаться в использовании методов детектирования объектов, которые могут выделять оконные рамы и подсчитывать их количество на изображении.

В целом, выбор метода для сегментации изображений фасадов зданий и подсчета количества окон зависит от конкретных требований и задач. Однако, использование сверточных нейронных сетей является наиболее распространенным и эффективным подходом для обработки изображений.

Примеры изображений, истинной маски сегментации и маски сегментации, созданной ИИ.

![cmp_b0271](https://user-images.githubusercontent.com/115422808/230939091-c822cb37-3cf5-4e62-a753-b50f170837ff.png)
![cmp_b0148](https://user-images.githubusercontent.com/115422808/230939345-a9a01789-3d1f-408c-8814-92406b87cf90.png)

Улучшить результаты можно путем увеличения количества данных, использования более качественной разметки данных, например вручную, с помощью веб-сервисов разметки.
