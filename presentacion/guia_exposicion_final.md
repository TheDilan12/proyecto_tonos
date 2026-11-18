# Guía de exposición — Entrega final

Presentación: `entrega_final_tonos_color.pdf` (21 diapositivas, ~15 minutos).
Ritmo sugerido: 40–45 segundos por diapositiva y 1 minuto en las de E11 y resultados (12, 13 y 14).

Reparto (bloques continuos, unos 3,5–4 minutos por persona; cada diapositiva muestra arriba a la derecha quién expone):

| Expositor | Diapositivas | Tema | Entrega la palabra con… |
|---|---|---|---|
| **Dilan** | 1–5 | Portada, retroalimentación, negocio, del discreto al continuo y factores del dominio | «…por eso, además de la luz, revisamos cómo se relaciona el color con los metadatos.» |
| **Juan** | 6–10 | Metadatos, segmentación, objetivo y multicolor, fotos de cliente simuladas y los once enfoques | «…ahora David explica cómo entrenamos las redes y la idea central, E11.» |
| **David** | 11–15 | Entrenamiento, E11 (medir en vez de adivinar), resultados, análisis crítico y criterios | «…Isabel muestra qué tan robusto es y cómo lo usaría el área comercial.» |
| **Isabel** | 16–21 | Robustez, explicabilidad, desagregación, demostración, conclusiones y cierre | Abre la ronda de preguntas. |

En las preguntas, responde quien expuso el tema; las de modelos y métricas las puede apoyar David.

---

## Diapositiva por diapositiva

### Bloque de Dilan (diapositivas 1–5)

**1 · Portada. (Dilan)**
- Qué dice: el proyecto estima el color de una prenda a partir de una foto y lo compara contra el catálogo de una empresa de colorantes textiles.
- Orden de la charla: retroalimentación, preparación, modelos, evaluación y conclusiones.

**2 · Seis observaciones, seis cambios. (Dilan)**
- Qué dice: en la Entrega 1 el profesor hizo seis observaciones. Cada tarjeta es una observación y lo que cambiamos.
- Idea clave: la salida ya no es una de 46 clases sino un color continuo, y el modelo no necesita máscara al predecir.

**3 · Flujo del área comercial. (Dilan)**
- Qué dice: llega una foto de un cliente, el modelo estima el color en CIELAB y se buscan las referencias del catálogo más cercanas por ΔE2000.
- Idea clave: el sistema sugiere y el laboratorio hace el ajuste final.

**4 · Del discreto al continuo. (Dilan)**
- Qué dice:
  - Cada etiqueta de tono es una región del espacio CIELAB, no un punto.
  - Usamos la etiqueta para dos cosas: calcular un prototipo por tono (la mediana) y descartar prendas cuyo color medido no coincide con su etiqueta. Se descarta el 36 % del entrenamiento.
- Idea clave: así se pasa de etiquetas discretas a un color continuo.

**5 · Elementos del dominio. (Dilan)**
- Qué dice:
  - La misma camiseta medida bajo luz cálida cambia ΔE 27,9: es otro color.
  - El catálogo tiene luz neutra (solo el 5 % de sus fotos tiene una dominante de color), pero un cliente no.
  - Los nueve factores están en la tabla del notebook 03.

### Bloque de Juan (diapositivas 6–10)

**6 · Metadatos. (Juan)**
- Qué dice:
  - El color se asocia con el género y el tipo de prenda. Por ejemplo, el rosado es mucho más frecuente en mujeres y niñas, y el 82 % de los jeans es azul.
  - Kruskal-Wallis confirma que la claridad y la saturación cambian por género y por uso.
- Idea clave: por eso, en la evaluación revisamos el error por grupo, para comprobar que el modelo no aprendió el tipo de prenda en lugar del color.

**7 · Segmentación. (Juan)**
- Qué dice:
  - La máscara por umbral de la Entrega 1 tenía un 45 % de píxeles que no eran la prenda.
  - El modelo de segmentación de prendas aísla la prenda correcta según su subcategoría.
  - El balance de blancos usa el fondo blanco como referencia.
- Idea clave: la máscara solo se usa para construir el color objetivo; al predecir no se usa.

**8 · Objetivo y multicolor. (Juan)**
- Qué dice: el color objetivo es el color dominante de la prenda, el de mayor participación según k-means con 3 colores.
- Regla multicolor: una prenda es multicolor si su segundo color ocupa al menos el 25 % y está a ΔE ≥ 20 del primero.
  - Coincide bien con la etiqueta "Multi" (AUC 0,80).
  - Para estas prendas se entrega la paleta completa.

**9 · Fotos de cliente simuladas. (Juan)**
- Qué dice: se tomaron las mismas prendas de prueba y se les cambió el fondo, la luz (de 3.000 a 7.500 K), el brillo y la compresión JPEG.
- Idea clave: el color objetivo no cambia, así que el modelo debe recuperar el color bajo luz neutra.
- Si preguntan: cuando la segmentación pierde parte de la prenda, el fondo simulado la tapa un poco. Es una limitación conocida.

**10 · Once enfoques. (Juan)**
- Qué dice: recorrimos el curso, de la visión clásica al fine-tuning.
- E10 es la versión mejorada de E9, con la red completa y una cabeza de tonos. E11 es la red híbrida que mide el color sobre los píxeles.
- Idea clave: de E3 en adelante ninguno usa máscara manual al predecir.

### Bloque de David (diapositivas 11–15)

**11 · Costo y entrenamiento. (David)**
- E8–E10: pérdida Huber sobre el Lab más tareas auxiliares de grupo y tono. El aumento con fondo y luz simulados mantiene fijo el color objetivo, y así enseña constancia de color.
- E11 se entrena directamente con ΔE2000, la métrica del negocio, que implementamos diferenciable. Además aprende su mapa de atención y la corrección de la luz.
- Dato práctico: en la GPU RTX 4060 cada época tarda 22 s (E8/E9), 31 s (E10) y 47 s (E11).

**12 · Medir el color en vez de adivinarlo (E11), la idea central. (David)**
- Qué dice: una red normal resume la imagen en un vector y "adivina" el color. E11 hace lo que haría un colorista:
  1. señala qué píxeles son el color dominante de la prenda (mapa de atención);
  2. estima la luz de la foto (tres ganancias por canal);
  3. mide el color sobre esos píxeles ya corregidos.
- Por qué: con la máscara exacta el color sale con ΔE 0,6. Con la misma máscara a 1/4 de resolución sale 3,8, que es justo el error de E10.
- Supervisión gratis: la máscara del color dominante y la luz simulada, que conocemos.

**13 · Resultados (diapositiva central). (David)**
- Qué dice: en fotos de cliente, la Entrega 1 queda en 19,5, E9 en 4,5, E10 en 3,7 y **E11 en 2,2**. El 86 % de las prendas queda con ΔE < 5.
- Si preguntan por el 0,7 de E11 en catálogo: ahí casi replica la medición, porque el objetivo se mide sobre esos mismos píxeles. La cifra que importa es la de las fotos de cliente.

**14 · Cada mejora salió de medir dónde estaba el error (análisis crítico). (David)**
- Qué dice: dos diagnósticos y dos soluciones.
  - **Nombre:** con el color perfecto, el método por prototipo solo acierta 75 % del grupo y 60 % del tono. Solución: la red predice el nombre (E10).
  - **Color:** el error de E10 coincidía con medir con una máscara de baja resolución. Solución: medir a resolución completa (E11).
- Contexto: el piso físico de la simulación es ΔE ≈ 1,6 y E11 queda a 0,6 de él.
- Mención: usamos un segundo modelo (Claude Fable) como revisor del análisis de errores.

**15 · Criterios de éxito. (David)**
- Qué dice:
  - E11 cumple el F1 de grupo (0,83), el tono top-1 (80 %), el top-3 (97 %) y ahora también la **meta comercial**: una referencia equivalente aparece en el top-3 el 88 % de las veces.
  - El grupo queda en 86 %, cerca de la meta de 90 %.
- Alcance: se mide en prendas coherentes de un color. En todo el test, E11 da ΔE 2,7, 70 % de grupo y 60 % de tono.

### Bloque de Isabel (diapositivas 16–21)

**16 · Robustez. (Isabel)**
- Qué dice: con luz cálida, E1 sube a ΔE 25 y E8 a 15; E9 queda en 4,9, E10 en 4,0 y E11 en 2,4. Con cualquier fondo, E11 queda en 2,2–2,3.

**17 · Explicabilidad. (Isabel)**
- Qué dice: E8 mira el fondo; E9 y E10 miran una mancha difusa sobre la prenda.
- Idea clave: el mapa de atención de E11 sigue el contorno exacto del pantalón o de la camiseta y evita el estampado, la piel y el fondo. Además, ese mapa es literalmente lo que usa para medir el color.

**18 · Desagregación. (Isabel)**
- Qué dice: hombres 2,4 y mujeres 2,9. Unisex llega a 5,7 porque hay pocos datos, y las prendas multicolor a 6,0 (para ellas se entrega la paleta).
- E11 ya no tiene sesgo de claridad.

**19 · Demostración. (Isabel)**
- Qué dice: de la foto a tres referencias del catálogo; en el jean, la primera queda a ΔE 1,1.
- Idea clave: señalar la camiseta rosada. El color de E11 es correcto, pero la paleta de la segmentación tomó el fondo.

**20 · Conclusiones y limitaciones. (Isabel)**
- Qué dice: leer los cuatro puntos de cada columna.
- Énfasis: los datos de cliente y el catálogo son simulados.

**21 · Cierre. (Isabel)**
- Qué dice: tres cifras para cerrar: ΔE 2,2, 88 % con una referencia equivalente en el top-3 y 80 % de acierto en el tono. Luego el siguiente paso y la ronda de preguntas.

---

## Preguntas probables y respuestas cortas

- **¿Por qué no clasificar en 46 clases?**
  - El cliente pide cualquier color y las clases se solapan mucho.
  - Con un color continuo se puede comparar contra cualquier catálogo.
- **¿Cómo se pasa de etiqueta discreta a Lab continuo?**
  - El objetivo es el color medido en la prenda segmentada.
  - La etiqueta solo sirve para calcular el prototipo de cada tono y para filtrar prendas incoherentes.
- **¿Y si la foto real no tiene fondo blanco?**
  - Para eso se entrenaron E9, E10 y E11 con fondos y luces simulados; E11 además estima explícitamente la luz de la foto.
  - Al predecir no necesita ni máscara ni fondo blanco.
- **¿Qué pasa con las prendas de varios colores?**
  - Se detectan con la regla multicolor (AUC 0,80) y se entrega la paleta para que el asesor elija.
- **¿No es trampa cambiar cómo se mide el nombre?**
  - Las metas son las mismas, y el notebook 05 muestra las dos columnas: el método anterior y el nuevo.
  - Mostramos que con el método anterior ni un color perfecto llegaba a la meta (75 % de grupo).
  - Ahora el nombre lo predice la red, que es justo lo que usaría el área comercial.
- **¿Cómo puede el tono top-1 de E10 (80 %) superar el techo (60 %)?**
  - El techo es del método por prototipo, que usa un único color representativo por tono.
  - La cabeza de tonos aprende de la imagen cómo nombra el catálogo cada color (por ejemplo, cuándo un azul oscuro se llama Navy Blue).
- **¿E11 no es trampa, si aprende la máscara del objetivo?**
  - No. Al predecir no recibe ninguna máscara: la deduce de la foto del cliente, con fondo y luz que nunca vio.
  - La máscara solo se usa como supervisión durante el entrenamiento, igual que las etiquetas en cualquier modelo supervisado.
- **¿Se puede bajar más el ΔE?**
  - Con esta simulación, el piso con luz y máscara perfectas es ΔE ≈ 1,6, y E11 está en 2,2.
  - Lo que más ayudaría en la práctica es pedir al cliente una hoja blanca en la foto como referencia de luz.
- **¿Por qué la meta comercial se mide con referencias "equivalentes"?**
  - En el catálogo simulado la segunda referencia está a solo ΔE 1,1 de la primera, y eso es indistinguible a la vista.
  - Aun con la referencia exacta, E11 llega a 80 %.
  - Hay que medirla de nuevo con el catálogo real de la empresa.
- **¿Por qué sigue habiendo errores de nombre?**
  - El 36 % de las etiquetas no coincide con el color medido.
  - Tonos vecinos como Navy Blue y Blue quedan a pocos ΔE de distancia.
- **¿Las fotos simuladas representan la realidad?**
  - Solo en parte. Es la principal limitación.
  - El paso siguiente es validar con requerimientos reales y, idealmente, con medición de espectrofotómetro.
- **¿Por qué gray-world empeora el resultado?**
  - Supone que el promedio de la escena es gris.
  - En una prenda de un solo color grande, confunde el color de la prenda con el de la luz.

---

## Glosario de términos nuevos (complementa la guía de la Entrega 1)

| Término | Significado |
|---|---|
| **Objetivo continuo** | El valor a predecir es un color (L*, a*, b*) y no una clase. |
| **Prototipo de tono** | La mediana Lab de todas las prendas de un tono. Representa a ese nombre de color. |
| **Coherencia de etiqueta** | Una prenda es coherente si el prototipo de su etiqueta está entre los 5 más cercanos a su color medido. |
| **Human parsing / SegFormer** | Red que etiqueta cada píxel como piel, cabello, camisa, pantalón, etc. |
| **Von Kries** | Corrección de la luz que divide cada canal por el color de una referencia blanca (aquí, el fondo). |
| **Gray-world** | Corrección de la luz que supone que el promedio de la escena es gris. |
| **Constancia de color** | Capacidad de reconocer el mismo color bajo luces distintas, como hace el ojo humano. |
| **Temperatura de color (K)** | Tono de la luz: 3.000 K es cálida y amarilla, 6.500 K es luz de día neutra, 7.500 K es fría y azulada. |
| **k-means en Lab** | Agrupa los píxeles de la prenda en k colores; el grupo más grande es el dominante. |
| **Paleta** | Los k colores de la prenda con su participación (porcentaje). |
| **AUC** | Área bajo la curva ROC: 1 es detección perfecta y 0,5 es azar. |
| **V de Cramér** | Fuerza de asociación entre dos variables categóricas, de 0 a 1. |
| **Residuo estandarizado** | Cuánto se aleja una celda de una tabla de lo esperado si no hubiera relación. Valores mayores a 2 en absoluto son significativos. |
| **Kruskal-Wallis** | Prueba que compara medianas entre varios grupos sin suponer normalidad. |
| **Ridge** | Regresión lineal con penalización que evita el sobreajuste. |
| **Características congeladas** | Salida de una red preentrenada usada sin reentrenarla. |
| **DINOv2** | Vision Transformer entrenado sin etiquetas (auto-supervisado). |
| **Zero-shot** | Clasificar sin entrenar, comparando la imagen con textos ("a photo of a red garment"). |
| **Techo de una métrica** | Lo máximo que puede lograr cualquier modelo con esa forma de medir (aquí: evaluar el color perfecto). |
| **Cabeza de clasificación** | Capa final de la red que predice una clase (grupo o tono) a partir de las mismas características que predicen el color. |
| **Red híbrida (E11)** | Red que no regresa el color: predice dónde está y qué luz hay, y lo mide sobre los píxeles. |
| **Mapa de atención** | Peso por píxel (suma 1) que indica qué píxeles se usan para medir el color. |
| **Ganancias de canal** | Factores por R, G y B que corrigen el color de la luz (Von Kries). |
| **Pérdida diferenciable** | Función de error de la que se puede calcular la derivada, para entrenar la red con ella. |
| **Piso de ruido** | Error mínimo que tendría incluso un modelo perfecto, por el ruido de los datos o de la medición. |
| **TTA (test-time augmentation)** | Promediar la predicción de la imagen original y de su reflejo. |
| **Fine-tuning** | Reentrenar parte de una red preentrenada para la tarea propia. |
| **Huber (Smooth L1)** | Pérdida cuadrática para errores pequeños y lineal para errores grandes; es robusta a valores atípicos. |
| **Label smoothing** | Suaviza las etiquetas de clasificación para que la red no se sobreconfíe. |
| **AdamW / One-cycle** | Optimizador con decaimiento de pesos, y una tasa de aprendizaje que sube y luego baja durante el entrenamiento. |
| **ΔE76 / ΔE2000** | Distancia entre colores: ΔE76 es la distancia euclidiana en Lab; ΔE2000 corrige para imitar la percepción humana. Por encima de ~2 la diferencia es perceptible; por encima de 10 es otro color. |
| **hit@k** | Porcentaje de casos en que la referencia correcta del catálogo está entre las k sugeridas. |
| **Grad-CAM** | Mapa de calor de las regiones de la imagen que más influyen en la predicción. |
| **Metamerismo** | Dos colores que coinciden bajo una luz y difieren bajo otra. |
