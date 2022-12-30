# lidi-floyd-warshall

## Estrategia
En lugar de tener bucles separados para la fase 2-3 y la fase 4, se realiza un único for (con único iterador "w"), que se divide en las 3 fases mencionadas.

* Fase 2: primeras "r" iteraciones.
* Fase 3: siguientes "r" iteraciones.
* Fase 4: restantes "r x r" iteraciones.

## Resultados frente a la versión 3
❌ Los GFLOPS resultaron bajos, produciendo incluso un retroceso en la mejora dada desde opt_5 (peor resultado que esta versión).

El posible problema es que muchos hilos ingresarán primero a fase 4 en lugar de las fases 2-3, produciendo esperas que no se dan en la versión 3.
