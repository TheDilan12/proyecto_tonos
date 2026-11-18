# Guía de exposición — Entrega 1

**Proyecto:** Identificación automática de tonos de color: de la foto de una prenda al catálogo de colorantes textiles
**Duración:** 15 minutos de exposición + 5 de preguntas · 16 diapositivas (≈ 1 minuto por diapositiva)
**Material:** [entrega1_tonos_color.pdf](entrega1_tonos_color.pdf) · notebook [01_comprension_negocio_y_datos.ipynb](../notebooks/01_comprension_negocio_y_datos.ipynb)

Los términos en **negrita** dentro del guion se explican en el [glosario](#glosario) al final.

---

## Reparto sugerido

| Bloque | Diapositivas | Tiempo | Expositor (sugerido) |
|---|---|---|---|
| Negocio | 1–4 | 4 min | Dilan Monsalve Monsalve |
| Datos I: estructura, desbalance y color | 5–8 | 3 min 30 s | Juan David Trujillo Velez |
| Datos II: tonos vecinos, etiquetas, sesgo y aumento | 9–12 | 4 min | David Botero Londoño |
| Estrategia y cierre | 13–16 | 3 min 30 s | Isabel Jurado Jaramillo |

**Mensaje central** (si solo recuerdan una frase): *«Los clientes piden colores con fotos de prendas. Entrenamos un modelo que reconoce el tono de una prenda para que el área comercial encuentre más rápido la referencia del catálogo de colorantes. Como el color es la etiqueta, hay que medirlo solo sobre la prenda, no se puede alterar al aumentar datos y conviene predecirlo en dos niveles (46 tonos y 13 grupos).»*

---

## Guion por diapositiva

### 1 · Portada (≈ 30 s)
- Presentar al equipo y el tema: reconocer el tono de una prenda en una foto para llevarlo al **catálogo de colorantes**.
- Decir que se sigue **CRISP-DM** y que esta entrega cubre las dos primeras etapas: **comprensión del negocio** y **comprensión de los datos**.
- Adelantar la estructura: negocio → datos (EDA) → estrategia.

### 2 · Los requerimientos de color llegan como fotos de prendas (≈ 1 min 15 s)
- Contexto: una empresa que fabrica **colorantes para el sector textil**. Sus clientes (**tintorerías**, confeccionistas, marcas) le hacen **requerimientos** al área comercial, y muchos llegan como una foto: «quiero este color».
- El asesor comercial tiene que traducir esa foto a una **referencia del catálogo** de colorantes.
- Tres dolores:
  - **Interpretación subjetiva:** el tono se estima a ojo («un azul petróleo») y cada asesor lo nombra distinto.
  - **Búsqueda manual:** con ese nombre se recorre el catálogo, y el resultado depende de la experiencia del asesor.
  - **Reprocesos:** si el punto de partida está lejos, el laboratorio repite **pruebas de tinturado** y la respuesta al cliente se demora.

### 3 · Propuesta: de la foto al catálogo de colorantes (≈ 1 min 15 s)
- Explicar las dos filas del diagrama:
  - **Entrenamiento (este proyecto):** con *Fashion Product Images* (44.072 fotos de prendas con 46 tonos) el modelo aprende a reconocer el color de una prenda en una foto.
  - **Uso con cada requerimiento nuevo:** foto del cliente → **segmentar** la prenda → el modelo da el tono (**top-3**) y su color **CIELAB** → se compara por **ΔE2000** contra el catálogo de colorantes → el comercial recibe las referencias más cercanas.
- Por qué este dataset sirve: son fotos reales de prendas con nombres comerciales de tono, el mismo tipo de imagen que mandan los clientes.
- Remarcar el alcance: el laboratorio sigue validando con pruebas físicas. El modelo da un **mejor punto de partida**, no reemplaza la **igualación de color**.
- Requisito para el despliegue: el catálogo con las **coordenadas Lab** de cada referencia, que aporta la empresa.

### 4 · Objetivos y criterios de éxito (≈ 1 min 15 s)
- Objetivo de negocio: que el área comercial responda más rápido, con menos reprocesos y de forma consistente los requerimientos que llegan como foto.
- Tarea de visión: **clasificación multiclase** de 46 tonos con **salida jerárquica** de 13 grupos básicos (los 11 colores de **Berlin & Kay** + Metálico + Multi), más el color Lab de la prenda para cruzarlo con el catálogo.
- Leer la tabla de metas:
  - ≥ 90 % de **accuracy** y ≥ 0,80 de **F1-macro** en los 13 grupos.
  - ≥ 70 % **top-1** y ≥ 90 % **top-3** en los 46 tonos.
  - Un **ΔE2000** medio del error menor que el de la **línea base**.
  - ≥ 80 % de las sugerencias top-3 aceptadas por el asesor comercial.
  - Un tiempo de respuesta al requerimiento menor que el actual (primero hay que medir cuánto tarda hoy).
- Explicar por qué el ΔE: confundir Navy Blue con Blue es menos grave que confundirlo con Red. En el negocio, un error pequeño todavía lleva a una referencia útil del catálogo, y la accuracy no distingue esos dos errores.

### 5 · Fashion Product Images (≈ 1 min)
- Dataset público de Kaggle (Param Aggarwal). Usamos el espejo de Hugging Face en versión *small*.
- Cifras: 44.072 imágenes, 46 tonos agrupados en 13, 60×80 píxeles en el 99,9 % de las imágenes, 0 corruptas y 430 en escala de grises (casi todas Black, White o Silver, así que es coherente).
- Además de la imagen trae metadata (género, categoría, tipo de artículo, nombre comercial), que usamos para buscar sesgos y validar etiquetas.
- Formato **parquet** con la imagen **JPEG** embebida. La versión de alta resolución (~25 GB) queda como opción para el modelado.

### 6 · Desbalance extremo de cola larga (≈ 1 min)
- Mostrar la gráfica en escala logarítmica: cada barra está pintada con el color mediano de su tono.
- 1.940 veces más imágenes en la clase mayor (Black, 9.699) que en la menor (Lime Green, 5).
- El 60 % de los datos está en 5 tonos, y 16 tonos tienen menos de 100 imágenes.
- Consecuencias: usar F1-macro como métrica, una función de costo ponderada o **focal loss**, y **oversampling** de las clases raras.

### 7 · El producto no es toda la foto (≈ 1 min)
- El 98 % de las fotos tiene fondo blanco y el 67 % de los píxeles es casi blanco. Si se promediara la imagen completa, todo saldría «blanco».
- Para medir el color del producto construimos una **máscara de primer plano**: un píxel es producto si su **ΔE2000** contra el color del borde supera 6, y luego se limpia con una **apertura morfológica**.
- Señalar en la figura dos problemas:
  1. Una prenda blanca sobre fondo blanco se pierde.
  2. En ropa casi siempre aparece el modelo, y su piel, cabello y pantalón contaminan el color.
- Conclusión: hace falta **segmentar** el producto.

### 8 · Los tonos en un espacio de color perceptual (≈ 1 min)
- Explicar **CIELAB**: L\* es la luminosidad, y a\* y b\* el eje verde–rojo y el eje azul–amarillo. Se elige porque la distancia en este espacio se parece a la diferencia que percibe una persona, cosa que no pasa en RGB.
- Gráfica izquierda: cada punto es un producto pintado con su color. Forman un **continuo** sin fronteras limpias.
- Gráfica derecha: la mediana de cada tono. Los **acromáticos** (negro, blanco, gris, plata) se amontonan en el centro y solo la luminosidad los separa.

### 9 · Muchos tonos son casi el mismo color (≈ 1 min)
- La matriz muestra el ΔE2000 entre los 46 tonos: un tono más claro en la matriz es más parecido.
- Pares casi idénticos: Mushroom Brown · Khaki (0,8), Off White · Cream (2,1), Maroon · Burgundy (3,3). Un ΔE de 1–2 apenas se percibe.
- En 45 de 46 tonos la **variabilidad intra-clase** (cuánto varían las fotos de un mismo tono) supera la distancia al tono vecino.
- Por eso se propone la salida jerárquica 46 → 13: el grupo básico es confiable aunque el tono fino sea ambiguo.

### 10 · Etiquetas buenas, pero con ruido (≈ 1 min)
- **Auditoría de etiquetas:** el nombre comercial suele traer el color («Turtle Check Men *Navy Blue* Shirt»). La etiqueta coincide con ese nombre en el 93,5 % de los casos (94,9 % a nivel de grupo).
- Con el hash **MD5** encontramos 52 fotos idénticas publicadas con tonos distintos (105 registros en total). Es **ruido de etiqueta** seguro.
- Los desacuerdos típicos son productos bicolor (Black con detalles rojos) y tonos vecinos (Navy Blue ↔ Blue).
- Implicación: usar **label smoothing** y no esperar el 100 % en tonos finos.

### 11 · El tono delata el tipo de producto (≈ 1 min)
- Los tonos metálicos están casi solo en accesorios: relojes (497), aretes (324), tacones (230) y gafas (179).
- **Riesgo de atajo (*shortcut learning*):** el modelo puede aprender «reloj → Silver» sin mirar el color.
- Mitigación: evaluar por categoría, usar **Grad-CAM** para ver dónde mira el modelo y compararlo contra un modelo que solo ve el color.

### 12 · El color es la etiqueta: nada de ColorJitter (≈ 1 min)
- Es un punto clave y diferenciador del proyecto.
- En clasificación normal se usa **data augmentation** fotométrico (**ColorJitter**: cambiar tono, brillo y saturación). Aquí eso **cambia la clase**: el tono +0,08 da ΔE 15 y el brillo ×1,4 da ΔE 12,5, es decir, otro color.
- Solo se permiten transformaciones **geométricas** (flip, rotación, recorte y escala), que dejan el color casi igual (ΔE ≤ 1).

### 13 · División de datos y línea base (≈ 1 min)
- **Split** 70/15/15 **estratificado** por tono y **agrupado** por duplicados, para que una misma foto no quede en train y en test (**data leakage**). Los 46 tonos aparecen en los tres conjuntos.
- **Línea base:** un **k-NN** con 6 descriptores de color del producto logra 57 % de accuracy en 46 tonos y 64 % en 13 grupos. El F1-macro de 0,20 muestra que falla en las clases raras.
- La **matriz de confusión** muestra que Blanco, Gris y Metálico se confunden.
- Mensaje: el color promedio no basta, y esta línea base es la cifra que debemos superar.

### 14 · Segmentar y comparar cuatro familias de modelos (≈ 1 min 15 s)
- Paso previo: segmentar el producto con **SAM** o un segmentador de prendas.
- Cuatro familias de modelos, alineadas con el curso:
  1. **Clásico** (clase 1): histograma de color a\*b\* y color dominante por **k-means**, más un **SVM** o k-NN.
  2. **Transfer learning** (clases 2–3): **ResNet-50**, **ConvNeXt-T** y **ViT-B** preentrenados. Primero se entrena la cabeza y luego se hace **fine-tuning** completo.
  3. **Modelo fundacional** (sesión 4): **CLIP** en **zero-shot** y después **linear probe**.
  4. **Destilación de conocimiento** (opcional): pasar lo aprendido a un **MobileNet** liviano para la herramienta del área comercial.

### 15 · Entrenamiento y evaluación (≈ 1 min)
Recorrer la tabla, que responde lo que pide el enunciado:
- **Clases:** 46 tonos con una cabeza auxiliar de 13 grupos; Multi se trata aparte.
- **Función de costo:** cross-entropy ponderada o focal loss, label smoothing guiado por ΔE y una pérdida auxiliar del grupo.
- **Preparación:** segmentación, **resize** a 224 con **padding** blanco y **normalización** con la media del dataset (≈ 0,84 por el fondo blanco, muy distinta de **ImageNet**).
- **Aumento de datos:** solo geométrico, más oversampling.
- **Datos adicionales:** la versión de alta resolución si hace falta, y fusionar los tonos con menos de 30 ejemplos.
- **Evaluación:** F1-macro, accuracy, top-3, ΔE medio del error, matriz de confusión y Grad-CAM.
- **Entorno:** Google Colab con GPU T4 o GPU local, con PyTorch, timm y open_clip y semilla fija.

### 16 · En resumen (≈ 30 s)
Cerrar con tres cifras:
- **1.940×** de desbalance, que define la métrica y el costo.
- **45 de 46** tonos se solapan con su vecino, lo que justifica la salida jerárquica.
- **57 %** de la línea base: hay que segmentar y aprender más que el color promedio.

Luego abrir preguntas.

---

## Preguntas probables y cómo responderlas

| Pregunta | Respuesta corta |
|---|---|
| ¿Por qué no usan la versión de alta resolución? | Para el EDA, 60×80 basta: el color no necesita detalle fino y procesar las 44 mil imágenes toma 3 minutos. La de alta resolución (~25 GB) queda disponible si el modelado lo requiere. |
| ¿Por qué CIELAB y no RGB o HSV? | En CIELAB la distancia euclidiana se parece a la diferencia percibida, y ΔE2000 la corrige aún más. En RGB, dos colores a la misma distancia numérica pueden verse muy distintos o casi iguales. HSV separa el matiz, pero no es perceptualmente uniforme. |
| ¿Cómo eligieron los 13 grupos? | Con los 11 términos básicos de color de Berlin & Kay (presentes en muchas lenguas), más Metálico (plata, oro, bronce: no son tonos cromáticos) y Multi (varios colores). Está en `core/config.py`. |
| Si 45 de 46 tonos se solapan, ¿tiene sentido clasificar 46? | Sí, como salida fina con top-3 y ΔE: al negocio le sirve recibir tres sugerencias cercanas. La decisión confiable se toma a nivel de grupo. |
| ¿Qué harán con la clase Multi? | Excluirla del entrenamiento del tono principal o tratarla como clase aparte, porque no es un color. |
| ¿Cómo evitan que el modelo aprenda la categoría? | Midiendo las métricas por categoría, revisando con Grad-CAM que el modelo mire el producto y comparando contra el modelo que solo usa color. |
| ¿Por qué el k-NN es tan bajo? | Porque usa solo 6 números de color de una máscara que todavía incluye piel y otras prendas, y porque los tonos se solapan. Sirve como piso a superar. |
| ¿Qué tan confiable es la heurística de piel? | Es aproximada: también marca productos beige, café o naranja. La usamos solo como indicio de que hay personas en las fotos, no como cifra exacta. |
| ¿Cómo se definieron las metas? | Son metas iniciales propuestas por el equipo. Se ajustarán después de ver los primeros modelos. |
| ¿Por qué entrenar con fotos de moda si el negocio es de colorantes? | Porque los requerimientos llegan justamente como fotos de prendas. El dataset de moda tiene ese mismo tipo de imagen, con el tono etiquetado en 44 mil ejemplos, algo que la empresa no tiene hoy. El modelo aprende a reconocer el color de la prenda, y el catálogo traduce ese color a referencias de colorante. |
| ¿Una foto sirve para igualar un color textil? | No para la igualación final, que depende de la iluminación, la cámara y el sustrato, y la sigue haciendo el laboratorio con mediciones físicas. La foto sirve para acotar rápido la búsqueda en el catálogo y dar un mejor punto de partida. |
| ¿Cómo se conecta el modelo con el catálogo? | Cada referencia del catálogo tiene un color medido en Lab. El modelo estima el Lab de la prenda segmentada y se calcula el ΔE2000 contra cada referencia; las de menor ΔE son las sugerencias. Esa es la etapa de despliegue (opcional en CRISP-DM). |
| ¿Qué pasa si la foto del cliente está mal iluminada? | Es un riesgo real. Se propone corregir el balance de blancos (constancia de color) antes de medir y mostrar el top-3, no una sola respuesta. Se puede pedir al cliente una foto con una referencia blanca o una carta de color. |

---

## Glosario

### Metodología y conceptos generales
- **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*): metodología estándar para proyectos de datos en seis etapas: comprensión del negocio, comprensión de los datos, preparación, modelado, evaluación y despliegue.
- **Comprensión del negocio:** definir el problema, los objetivos y cómo se medirá el éxito desde el punto de vista de la organización.
- **Comprensión de los datos / EDA** (*Exploratory Data Analysis*): explorar el dataset para entender su estructura, calidad, distribución y sesgos antes de modelar.
- **Taxonomía:** el sistema de categorías que se usa para nombrar algo; aquí, la lista de 46 nombres de tono del dataset.

- **Clasificación multiclase:** el modelo asigna a cada imagen exactamente una clase entre varias posibles (46 tonos).
- **Salida jerárquica:** el modelo predice dos niveles a la vez, uno fino (46 tonos) y uno grueso (13 grupos), y cada tono pertenece a un solo grupo.
- **Berlin & Kay:** estudio de 1969 que identificó 11 términos básicos de color comunes a muchas lenguas: negro, blanco, gris, rojo, verde, azul, amarillo, naranja, café, rosado y morado.
- **Acromático:** color sin matiz (negro, blanco, gris); se diferencian solo por su luminosidad.
- **Línea base (*baseline*):** un modelo simple que sirve de referencia; cualquier modelo propuesto debe superarlo.
- **Shortcut learning (aprendizaje por atajo):** cuando el modelo aprende una pista fácil pero incorrecta (el tipo de producto) en vez de la característica real (el color).

### Negocio de colorantes textiles
- **Colorante:** sustancia que se fija en la fibra textil y le da color. Una empresa de colorantes vende referencias (cada una con un tono y unas propiedades) a quienes tiñen telas o prendas.
- **Catálogo de colorantes:** listado de las referencias que ofrece la empresa, con su tono y, normalmente, su medición de color en el laboratorio.
- **Referencia:** cada producto del catálogo (un colorante o una mezcla con un tono definido).
- **Coordenadas Lab de una referencia:** el color de la referencia medido en CIELAB (con un espectrofotómetro sobre una muestra teñida). Permite compararla por ΔE con el color estimado de la foto.
- **Requerimiento comercial:** la solicitud de un cliente al área comercial («necesito este color»), que puede llegar con una foto, una muestra física o un código de color.
- **Tintorería:** empresa que tiñe telas o prendas; es cliente típico de una empresa de colorantes.
- **Prueba de tinturado:** teñir una muestra en el laboratorio para comprobar si el color obtenido coincide con el pedido.
- **Igualación de color (*color matching*):** proceso de laboratorio para encontrar la receta de colorantes que reproduce un color objetivo. El modelo no lo reemplaza: le da un mejor punto de partida.
- **Espectrofotómetro:** instrumento que mide cómo refleja la luz una muestra en cada longitud de onda; de ahí se calculan sus coordenadas Lab. Es la medición de referencia en la industria, mucho más exacta que una foto.
- **Despliegue:** sexta etapa de CRISP-DM (opcional en este curso): poner el modelo en uso real, aquí como una herramienta para el área comercial.

### Datos y formatos
- **Dataset:** conjunto de datos; aquí, 44.072 fotos con su metadata.
- **Metadata:** información adicional de cada imagen (género, categoría, tipo de artículo, temporada, nombre comercial).
- **Parquet:** formato de archivo en columnas, comprimido y eficiente para tablas grandes; aquí guarda tanto la metadata como la imagen.
- **JPEG:** formato de imagen con compresión con pérdida.
- **Espejo (*mirror*):** copia del dataset original alojada en otro sitio (Hugging Face en vez de Kaggle).
- **Modo de color RGB / L:** RGB tiene tres canales (rojo, verde, azul); L es una imagen de un solo canal en escala de grises.
- **Relación de aspecto:** alto dividido entre ancho (80/60 = 1,33).
- **Desbalance de clases:** algunas clases tienen muchísimos más ejemplos que otras.
- **Cola larga (*long tail*):** distribución con pocas clases muy grandes y muchas clases pequeñas.
- **Escala logarítmica:** eje donde cada marca multiplica por 10 (10, 100, 1.000), útil para mostrar a la vez valores muy grandes y muy pequeños.

### Color
- **Espacio de color:** forma de representar un color con números (RGB, HSV, CIELAB…).
- **CIELAB (L\*a\*b\*):** espacio de color diseñado para que las distancias numéricas se parezcan a las diferencias que percibe una persona. **L\*** es la luminosidad (0 = negro, 100 = blanco), **a\*** el eje verde (−) a rojo (+) y **b\*** el eje azul (−) a amarillo (+).
- **Croma (C\*ab):** qué tan saturado o «vivo» es un color; es la distancia al centro del plano a\*b\*. Un gris tiene croma ≈ 0.
- **ΔE2000 (Delta E CIEDE2000):** fórmula estándar para medir la diferencia entre dos colores en CIELAB. Como referencia: ΔE < 1 no se nota; 1–2 apenas se percibe; 2–10 se nota; > 10 ya parece otro color.
- **Color mediano:** la mediana de L\*, a\* y b\* de los píxeles del producto. Es más robusta que el promedio frente a píxeles extremos (brillos, sombras).
- **Centroide de clase:** el color mediano de todas las imágenes de un tono; representa a ese tono.
- **YCrCb:** espacio de color que separa luminancia (Y) de crominancia (Cr, Cb). En visión clásica se usa para detectar piel, porque la piel humana cae en un rango estrecho de Cr y Cb.
- **Constancia de color:** la capacidad de reconocer el mismo color bajo distinta iluminación. Es el motivo por el que los cambios de brillo complican el problema.

### Procesamiento de imágenes (visión clásica)
- **Máscara de primer plano:** imagen binaria (blanco/negro) que marca qué píxeles pertenecen al producto y cuáles al fondo.
- **Umbralización:** decidir píxel por píxel con una regla de corte; aquí, «es producto si ΔE contra el fondo > 6».
- **Operaciones morfológicas:** operaciones sobre imágenes binarias con un elemento estructurante (aquí un disco de radio 1). La **apertura** (erosión seguida de dilatación) elimina puntos sueltos y ruido pequeño sin encoger los objetos grandes.
- **Segmentación:** dividir la imagen en regiones, por ejemplo separar el producto del fondo y de la persona.
- **Histograma de color:** conteo de cuántos píxeles caen en cada rango de color o intensidad.
- **Varianza del Laplaciano:** medida de nitidez. El Laplaciano resalta bordes; si su varianza es baja, la imagen tiene pocos bordes y probablemente está borrosa.
- **Resize:** cambiar el tamaño de la imagen (de 60×80 a 224×224 para los modelos preentrenados).
- **Padding:** rellenar los bordes (aquí con blanco) para llevar la imagen a un cuadrado sin deformarla.
- **Normalización:** restar la media y dividir por la desviación estándar de cada canal para que los valores de entrada queden en un rango estable para la red.

### Calidad de datos
- **Hash MD5:** «huella digital» de un archivo. Dos archivos con el mismo MD5 son idénticos byte a byte, así que sirve para hallar duplicados exactos.
- **dHash (hash perceptual por diferencias):** huella que resume la estructura visual de la imagen (si cada píxel es más claro que su vecino). Dos imágenes casi iguales, aunque tengan distinta compresión, producen el mismo dHash. Usamos una cuadrícula de 16×16.
- **Casi-duplicados:** imágenes visualmente iguales aunque no idénticas byte a byte.
- **Ruido de etiqueta:** etiquetas incorrectas o inconsistentes en el dataset.
- **Auditoría de etiquetas:** verificar las etiquetas contra otra fuente; aquí, el color mencionado en el nombre comercial.
- **Outlier / caso límite:** ejemplo atípico (máscara vacía, imagen borrosa, fondo oscuro).

### División y evaluación
- **Train / validation / test:** conjuntos de entrenamiento (para aprender), validación (para ajustar hiperparámetros y elegir modelo) y prueba (para medir el resultado final sin sesgo).
- **Split estratificado:** la división mantiene la proporción de cada clase en los tres conjuntos.
- **Split agrupado (StratifiedGroupKFold):** las imágenes de un mismo grupo (duplicados) quedan todas en el mismo conjunto.
- **Data leakage (fuga de datos):** que información del conjunto de prueba se filtre al entrenamiento (por ejemplo, la misma foto en train y en test), lo que infla las métricas.
- **Accuracy (exactitud):** porcentaje de predicciones correctas. Engaña con clases desbalanceadas: predecir siempre «Black» ya da un 22 %.
- **Precisión y recall:** la precisión es, de lo que el modelo dijo «Rojo», cuánto era rojo; el recall es, de todo lo rojo, cuánto encontró.
- **F1-score:** media armónica de precisión y recall.
- **F1-macro:** promedio del F1 de cada clase con el mismo peso. Una clase con 5 imágenes cuenta igual que una con 9.000, así que castiga fallar en las clases raras.
- **Top-1 / Top-3 accuracy:** top-1 exige que la primera predicción sea la correcta; top-3 acepta que la correcta esté entre las tres más probables.
- **Matriz de confusión:** tabla de clase real contra clase predicha; la diagonal son los aciertos y lo de fuera muestra qué se confunde con qué.
- **k-NN (k vecinos más cercanos):** clasificador que asigna a una imagen la clase más común entre sus k ejemplos más parecidos del entrenamiento (aquí k = 25, ponderado por distancia).
- **SVM (máquina de vectores de soporte):** clasificador clásico que busca la frontera con mayor margen entre clases.
- **k-means:** algoritmo que agrupa los píxeles en k grupos de color; el grupo más grande da el color dominante.
- **Variabilidad intra-clase:** cuánto varían entre sí las imágenes de una misma clase.
- **Separabilidad entre clases:** qué tan lejos están las clases entre sí comparado con su variabilidad interna.

### Aprendizaje profundo y modelos
- **CNN (red neuronal convolucional):** red que aprende filtros sobre la imagen. Su sesgo inductivo (localidad, equivarianza a traslaciones) la hace eficiente con imágenes.
- **ResNet-50:** CNN de 50 capas con conexiones residuales (atajos que permiten entrenar redes profundas).
- **ConvNeXt-T:** CNN moderna (2022) que incorpora ideas de los Transformers; la «T» es la versión *tiny*.
- **ViT-B (Vision Transformer, Base):** divide la imagen en parches y los procesa con *self-attention*, como un Transformer de texto.
- **MobileNet:** CNN liviana diseñada para correr en dispositivos con pocos recursos.
- **Preentrenado:** modelo ya entrenado en un dataset grande (como ImageNet) que se reutiliza.
- **ImageNet:** dataset de ~1,2 millones de imágenes y 1.000 clases, usado para preentrenar modelos. Sus medias de normalización (0,485 / 0,456 / 0,406) son el estándar.
- **Transfer learning (transferencia de aprendizaje):** reutilizar un modelo preentrenado y adaptarlo a una tarea nueva.
- **Cabeza (*head*):** la última capa del modelo, la que produce las probabilidades de cada clase. Se reemplaza para la tarea nueva.
- **Fine-tuning:** reentrenar también las capas del modelo preentrenado, con una tasa de aprendizaje baja.
- **Modelo fundacional:** modelo enorme entrenado con datos masivos que sirve para muchas tareas sin reentrenarse (CLIP, SAM).
- **CLIP:** modelo de OpenAI que aprendió a emparejar imágenes y textos, de modo que permite clasificar comparando la imagen con frases.
- **Zero-shot:** clasificar sin entrenar con ejemplos de las clases, solo describiéndolas con texto («a photo of a navy blue product»).
- **Linear probe:** congelar el modelo fundacional y entrenar solo una capa lineal encima de sus representaciones.
- **SAM (Segment Anything Model):** modelo fundacional de Meta que segmenta cualquier objeto en una imagen.
- **Destilación de conocimiento:** entrenar un modelo pequeño (alumno) para imitar las predicciones de uno grande (maestro).
- **Grad-CAM:** técnica de explicabilidad que muestra con un mapa de calor qué zonas de la imagen usó la red para decidir.
- **Semilla fija:** valor inicial del generador aleatorio, para que los experimentos se puedan repetir con los mismos resultados.

### Función de costo y entrenamiento
- **Función de costo (*loss*):** medida del error que el modelo minimiza durante el entrenamiento.
- **Cross-entropy (entropía cruzada):** la pérdida estándar de clasificación; penaliza asignar baja probabilidad a la clase correcta.
- **Cross-entropy ponderada:** da más peso al error en las clases raras para compensar el desbalance.
- **Focal loss:** variante que reduce el peso de los ejemplos fáciles y se concentra en los difíciles.
- **Label smoothing:** en vez de exigir 100 % de probabilidad a la clase correcta, reparte un poco entre las demás. Reduce el sobreajuste y tolera el ruido de etiqueta. «Guiado por ΔE» quiere decir que ese poco se reparte entre los tonos más parecidos.
- **Pérdida auxiliar:** un segundo término de error (aquí, el del grupo básico) que se suma al principal para guiar el aprendizaje.
- **Oversampling:** repetir más veces los ejemplos de las clases raras durante el entrenamiento.
- **Data augmentation (aumento de datos):** generar variaciones de las imágenes de entrenamiento para que el modelo generalice mejor.
- **Transformaciones geométricas:** cambian la posición o forma (flip, rotación, recorte, escala, traslación) sin alterar el color.
- **Transformaciones fotométricas:** cambian el color o la luz (tono, saturación, brillo, contraste).
- **ColorJitter:** transformación de torchvision que altera al azar el brillo, el contraste, la saturación y el tono.
- **Mixup:** mezclar dos imágenes y sus etiquetas. Aquí solo se permitiría entre imágenes del mismo tono.

### Herramientas
- **PyTorch / torchvision:** librería de deep learning y su módulo de visión (modelos y transformaciones).
- **timm:** colección de modelos de visión preentrenados para PyTorch.
- **open_clip:** implementación abierta de CLIP.
- **scikit-learn / scikit-image:** librerías de machine learning clásico y de procesamiento de imágenes.
- **Google Colab:** entorno de notebooks en la nube con GPU gratuita (por ejemplo, la T4).
- **Hugging Face / Kaggle:** plataformas donde se publican datasets y modelos.

---

## Consejos para la exposición
- Ensayar con cronómetro: el riesgo típico es pasarse en las diapositivas 2 y 13.
- En cada gráfica, decir primero **qué muestra** («cada barra es un tono…») y después **la conclusión**.
- Remarcar en cada hallazgo **qué decisión de modelado provoca**: es lo que pide la Entrega 1 («propuesta clara de la estrategia»).
- Tener abierto el notebook por si piden ver un análisis en detalle; las secciones siguen el mismo orden del checklist.
