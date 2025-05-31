# Tarea 3 - Paralelización de KNN

## Ejercicio 1 (4pts)

- Paralelice `knn_digits_sec.py`, siguiendo la estructura del ejemplo
discutido en clase (`knn_hpc_class_vis.py`). Es decir, debe incluir las
directivas de comunicaci´on (`comm.bcast`, `comm.scatter`,
`comm.gather`). (4pts)

El algoritmo `knn_digits.py` cuenta con la implementación de un KNN en paralelo.

```bash
mpirun -n 4 python knn_digits.py
```

```bash














