# Tarea 3 - Paralelización de KNN

## Ejercicio 1 (4pts)

- Paralelice `knn_digits_sec.py`, siguiendo la estructura del ejemplo
discutido en clase (`knn_hpc_class_vis.py`). Es decir, debe incluir las
directivas de comunicaci´on (`comm.bcast`, `comm.scatter`,
`comm.gather`). (4pts)

El algoritmo `knn_digits_1.py` cuenta con la implementación de un KNN en paralelo.

```bash
mpirun -n 4 python knn_digits.py
```

output: 
```bash
Accuracy: 0.9833
Execution time (parallel): 0.2590 sec
```

## Ejercicio 2 (3pts)

 - El código debe obtener los tiempos de ejecución, cómputo y
comunicación, así como la precisión del modelo (accuracy).
Realice las pruebas en Khipu. Incremente tanto procesos (p) como
datos (n). Ya que los datos importados (dígitos) tienen un tamaño
constante, se recomienda multiplicar la data para medir escalabilidad
o usar `from sklearn.datasets import make_classification` para generar
data variada

Corremos el sigiente script: ``knn_digits_2.sh`` con el objetivo de crear el archivo `knn_benchmark_results.csv`

finalmente corremos `generate_chart_2.py` para generar los  siguientes gráficos donde podemos ver a detalle los resultados de estos  experimentos.

#### Análisis de Datos Sintéticos
![Análisis de Datos Sintéticos](ejercicio_2/images/synthetic_data_analysis.png)

#### Análisis de Datos Reales
![Análisis de Datos Reales](ejercicio_2/images/real_data_analysis.png)
