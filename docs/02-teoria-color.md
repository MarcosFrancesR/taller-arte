# Sesión 2: Teoría del Color y Espacios No Lineales

En la sesión anterior vimos cómo los sensores capturan la luz de forma lineal. Sin embargo, para que una obra de arte digital sea percibida correctamente por el ojo humano, debemos aplicar transformaciones matemáticas que compensen nuestra percepción logarítmica de la luminosidad.

## El Espacio de Color $L*a*b*$

A diferencia del modelo RGB (basado en hardware), el espacio **CIE L*a*b*** es un modelo de apariencia cromática basado en la percepción humana. Se divide en tres ejes ortogonales:

* **$L^*$ (Luminosidad):** Representa el brillo, desde el negro ($0$) hasta el blanco ($100$).
* **$a^*$:** El eje cromático que va del verde (negativo) al rojo (positivo).
* **$b^*$:** El eje cromático que va del azul (negativo) al amarillo (positivo).

La ventaja artística de este modelo es que permite editar la iluminación de una imagen sin alterar sus colores, y viceversa.

### Diferencia de Color ($\Delta E$)

Para medir qué tan diferentes son dos colores en una composición, utilizamos la distancia euclídea en este espacio:

$$
\Delta E = \sqrt{(L_2^* - L_1^*)^2 + (a_2^* - a_1^*)^2 + (b_2^* - b_1^*)^2}
$$

> [!TIP]
> Un valor de $\Delta E < 1.0$ es imperceptible para el ojo humano medio, mientras que un valor $> 5.0$ indica una diferencia de color claramente visible.

---

## Corrección Gamma y Percepción

Como el ojo es más sensible a las variaciones en las sombras que en las altas luces, las imágenes se almacenan aplicando una **curva Gamma** ($\gamma$). La relación entre la intensidad de salida ($I_{out}$) y la de entrada ($I_{in}$) sigue esta potencia:

$$
I_{out} = I_{in}^{\gamma}
$$

En la mayoría de monitores estándar, se utiliza un valor de $\gamma \approx 2.2$. 

### Comparativa de Modelos

| Modelo | Uso Principal | Ventaja |
| :--- | :--- | :--- |
| **RGB** | Pantallas y Sensores | Simplicidad técnica |
| **CMYK** | Impresión física | Mezcla sustractiva real |
| **Lab** | Retoque profesional | Independencia de luminancia |
| **HSV** | Selección de color | Intuitivo para artistas |

