# ComfyUI-BubbleText 💬

[English](README.md) | **Español**

Nodos para ComfyUI que ponen **texto limpio y legible dentro de los globos de diálogo** generados por modelos como Anima.

Los modelos de imagen dibujan bien los globos, pero con frases largas se equivocan en las letras ("REALUNIATY", palabras repetidas…). Estos nodos dejan que el modelo dibuje el globo y después **borran lo que escribió la IA y escriben tu texto** con una fuente real, centrado y al tamaño máximo que quepa.

## Nodos

### 💬 Globo de texto (`SpeechBubblePrompt`)
Funciona como un LoRA para el prompt: va entre tu prompt y el `CLIP Text Encode`.

- Con **ON** y un texto escrito, añade al final del prompt la petición del globo:
  `speech bubble, english text, a single large white speech bubble with a thick black outline and black text at the top of the image, the speech bubble says "tu texto"`
- Con **OFF** o el texto vacío, deja el prompt igual.
- Sale el prompt modificado y una configuración `globo` que se conecta a **Escribir en globo**.

### 💬 Escribir en globo (`SpeechBubbleRender`)
Va entre `VAE Decode` y el nodo de guardar. No tiene controles: usa la configuración de **Globo de texto**.

1. Busca los globos: blancos con contorno (también fino), blancos sin contorno si tienen letras dentro y, como respaldo, globos negros con letras dentro. Si hay globos unidos (por ejemplo, dos conectados por la colita), los separa.
2. Borra las letras inventadas por el modelo.
3. Escribe tu texto con la fuente elegida, partiendo las líneas y ajustando el tamaño para que quepa.

Si no encuentra ningún globo, deja la imagen tal cual.

### 💬 Texto en globo (auto) (`SpeechBubbleTextAuto`)
Versión todo en uno que solo escribe sobre la imagen, sin tocar el prompt.

## Opciones

| Opción | Qué hace |
|---|---|
| `activado` | ON/OFF. Apagado, no toca ni el prompt ni la imagen. |
| `texto` | Lo que va en el globo. **Una línea en blanco** separa textos para varios globos (en orden de lectura). Acepta emojis 😄 |
| `fuente` | Fuentes de cómic de Windows (Comic Sans, Impact, Arial Black…) y cualquier `.ttf`/`.otf` que pongas en la carpeta `fonts/`. |
| `mayusculas` | Escribe todo en mayúsculas, como en los cómics. |
| `tamano_maximo` | Tamaño máximo de letra. El nodo usa el mayor que quepa. |
| `orden_lectura` | Izquierda → derecha, o derecha → izquierda (manga). |
| `borrar_texto_ia` | Borra las letras que dibujó el modelo antes de escribir. |
| `color_texto` | `auto` (negro en globos claros, blanco en oscuros) o un color `#RRGGBB`. |
| `margen` | Espacio entre el texto y el borde del globo. |
| `umbral_blanco` | Qué tan blanco debe ser el globo. Bájalo si no detecta globos algo grises. |
| `estilo_prompt` | (Solo en Globo de texto) `natural (Anima)` pide el globo con frases; `tags (Illustrious / NoobAI)` usa tags estilo Danbooru, que los modelos basados en SDXL siguen mejor. |

## Emojis

Los caracteres que no tenga la fuente elegida se dibujan con **Segoe UI Emoji** (Windows), a color. Los emojis no se envían al prompt: el modelo los dibuja mal y no los necesita.
Los emojis compuestos (familias, tonos de piel, algunas banderas) salen por separado si tu Pillow no tiene `raqm`.

## Workflows de ejemplo

En `example_workflows/` (también aparecen en **Plantillas → ComfyUI-BubbleText** dentro de ComfyUI):

- **Anima - Globo de texto (basico)**: solo nodos de ComfyUI + estos nodos.
- **Anima - Globo de texto (LoRA Manager)**: con `Lora Loader`, `TriggerWord Toggle` y `Save Image` de [ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager). Las trigger words se ponen delante de tu prompt.

Usan Anima (`anima_baseV10.safetensors`, `qwen_3_06b_base.safetensors`, `qwen_image_vae.safetensors`). Cambia los modelos por los tuyos.

## Instalación

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/arturor1990/ComfyUI-BubbleText.git
```

Reinicia ComfyUI. Las dependencias (`numpy`, `scipy`, `Pillow`, `fonttools`) ya vienen con casi cualquier instalación de ComfyUI; si falta alguna:

```bash
pip install -r requirements.txt
```

## Consejos

- Frases cortas dan globos más grandes y letras más grandes.
- Si usas un LoRA Turbo con **CFG 1**, el prompt negativo no tiene efecto, así que no sirve para evitar globos de colores.
- Si el modelo reparte la frase en varios globos, tu texto se reparte entre ellos en orden de lectura, así ninguno se queda con letras de la IA.
- Las recetas de LoRA Manager y los metadatos de la imagen guardan el prompt **con** la parte del globo ya inyectada. Si reutilizas ese prompt, quita esa parte o apaga el nodo; si no, se añade dos veces.
- Con LoRA Manager instalado, **Save Recipe** usa la imagen con tu texto ya escrito (no la salida del VAE Decode con las letras de la IA).
