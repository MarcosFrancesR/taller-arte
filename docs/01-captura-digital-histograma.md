# Sesión 1: Captura Digital y el Histograma

A diferencia de la fotografía analógica, donde la respuesta química de la película es logarítmica, los **sensores digitales** son una familia de dispositivos que tienen un comportamiento lineal. Una implicación crítica de este hecho es que la capacidad del sensor para registrar información no se distribuye de forma equitativa.

En un sensor digital, el número de niveles de gris disponibles para codificar la luz se duplica con cada paso de exposición ($EV$). Esto significa que la mitad de los niveles discretos de cuantificación se encuentran en el paso más brillante del rango dinámico.

Desde el punto de vista del compromiso (_trade-off_) entre **ruido** (_noise_) y **detalle**, si exponemos "hacia la izquierda" (subexponiendo), el sensor registrará los datos en zonas donde la relación señal-ruido es baja, provocando la aparición de grano digital. Por el contrario, si exponemos "hacia la derecha" (**ETTR**), maximizamos dicha relación, minimizando la varianza del error de cuantificación.

## Modelos de Color y Espacios Vectoriales

Una familia característica de modelos de representación son los espacios de color, que asumen que cualquier color visible puede expresarse como una combinación lineal de componentes primarios.

El modelo **RGB** predice un color resultante como una combinación ponderada de tres vectores base (Rojo, Verde y Azul), con la siguiente forma:

$$
\mathbf{C} = w_r \mathbf{R} + w_g \mathbf{G} + w_b \mathbf{B}
$$

Donde $\mathbf{C}$ es el color resultante, y los pesos $w_i \in [0, 255]$ (en sistemas de 8 bits) determinan la intensidad de cada canal. Desde el punto de vista geométrico, este modelo define un **cubo cromático** dentro de un espacio euclídeo tridimensional.



### El Histograma como Distribución de Probabilidad

En el procesado de imagen, el histograma no es más que una representación de la función de masa de probabilidad (PMF) de los niveles de intensidad en la imagen. Si definimos $n_k$ como el número de píxeles con intensidad $r_k$, la probabilidad de ocurrencia de un nivel de intensidad $r_k$ es:

$$
P(r_k) = \frac{n_k}{N}
$$

Donde $N$ es el número total de píxeles de la imagen. 

- **Imágenes de bajo contraste**: Los valores de $P(r_k)$ están concentrados en un rango estrecho del eje $x$.
- **Imágenes sobreexpuestas**: La distribución muestra un sesgo hacia $r_k \rightarrow 255$ (el límite superior), lo que conocemos como "clipping".

#### Ecualización del Histograma

La técnica de ecualización busca encontrar una función de transformación $s = T(r)$ que produzca un histograma con una distribución uniforme. Esto mejora el contraste global de la imagen. La función de transformación ideal se basa en la función de distribución acumulada (CDF):

$$
s_k = T(r_k) = \sum_{j=0}^{k} P(r_j) = \sum_{j=0}^{k} \frac{n_j}{N}
$$

Esta propiedad es fundamental en el post-procesado, ya que permite redistribuir la energía lumínica de la escena para aprovechar todo el rango dinámico disponible.



## Limitaciones de los Sensores Lineales

Como hemos visto, los sensores captan la luz de forma lineal. Sin embargo, el ojo humano tiene una respuesta **no lineal** (gamma). Esto genera un conflicto entre la captura técnica y la percepción estética.

### El problema del "Clipping" o saturación

Si la intensidad de la luz en una zona de la escena supera la capacidad de pozo del fotodiodo (_well capacity_), el sensor se satura. En este punto, la derivada de la función de respuesta se hace cero:

$$
\frac{d f(x)}{dx} = 0 \quad \text{para} \quad x > \text{umbral de saturación}
$$

En la figura siguiente podemos observar cómo la información se pierde irremediablemente cuando el histograma "choca" contra el borde derecho.

Figure: Ejemplo de histograma con pérdida de información en altas luces {#fig-clipping}

![](images/t1_histogram_clipping.png)

> **Nota importante:** A diferencia de la recuperación de sombras, donde podemos aplicar algoritmos de reducción de ruido (como filtros gaussianos), la información "quemada" (blancos puros) no puede recuperarse mediante cálculo matemático, ya que todos los valores originales de la escena se mapean al mismo valor máximo $255$.

## Soluciones al Rango Dinámico Limitado

Para capturar escenas donde el contraste supera la capacidad del sensor (datos no linealmente representables en un solo disparo), podemos optar por:

- **Ingeniería de Captura (HDR)**: Realizar un _bracketing_ de exposición (tomar varias muestras con distintos $EV$) y fusionarlas mediante un operador de mapeo de tonos (_Tone Mapping_).
- **Filtros de Densidad Neutra (GND)**: Aplicar una máscara física degradada sobre la lente para reducir la intensidad lumínica de las altas luces antes de que la luz llegue al sensor.
- **Post-procesado No Paramétrico**: Utilizar herramientas como las **Máscaras de Luminosidad**, que permiten seleccionar píxeles basándose en su valor de brillo de forma granular sin asumir una curva de ajuste fija.



En la próxima sesión estudiaremos la **Teoría del Color Avanzada** y cómo el espacio **Lab** nos permite separar la luminancia de la crominancia para un control artístico total.