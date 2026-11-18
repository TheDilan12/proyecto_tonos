"""Une los slides del artifact en un HTML autónomo (1920x1080 por página) listo para imprimir a PDF."""
import base64
import json
import re
import sys
from pathlib import Path

origen = Path(sys.argv[1])          # carpeta con project/deck.json y project/slides/
figuras = Path(sys.argv[2])         # carpeta figuras/ del proyecto
destino = Path(sys.argv[3])         # archivo .html de salida

BLOBS = {
    "23438363b721393fb5d6088299dd4901": "04_distribucion_tonos.png",
    "e8975ccdefbc307b0fd9081d769cce8f": "06_mascaras.png",
    "a16d44a8720f2428ffe004c1197cf411": "06_plano_ab.png",
    "f49904548bbee996c4fdd893ddb9dd8a": "11_matriz_delta_e.png",
    "6def968c383d0b8ec744bdab9c13e2f8": "02_duplicados_conflicto.png",
    "2627a92b0869146002778f44bc133c32": "04_tono_vs_categoria.png",
    "6c40223d6f5eba26c96a0b841af2b03f": "08_aumento_datos.png",
    "1ab525dc5b24339e94370eead25cf10e": "11_confusion_baseline.png",
    "e402d0916b22a8618ccafdc656022ecf": "03_muestra_aleatoria.png",
    # deck de resultados
    "b015af0f7ad5e1bf8362733a832617c0": "p2_segmentacion.png",
    "9855f624b9562844a46ae169685c547e": "p2_regiones_tono.png",
    "d19d94d39ae8bed5da33bcd983262705": "p3_metadatos_color.png",
    "6821078ca3a99cba91039f5eeb40ed97": "p3_iluminantes.png",
    "142205f75b5eb17e47b372058de02667": "p2_paletas.png",
    "29c6ac5135d0898599980ab20fcf2f59": "p2_contexto_real.png",
    "217ed2c1385a97691b99960ad8c9704e": "p4_curvas.png",
    "44419209e1929254efcfec3082be0c8f": "p5_comparacion_modelos.png",
    "f9cfe083bf739c85828766556fa41b60": "p5_robustez_factores.png",
    "3e6dbec3fe3e292acc3bee83ee268564": "p5_gradcam.png",
    "f56cab7e1657f347cc5d2e0dcc5cc298": "p5_desagregacion.png",
    "038b65190c4b5a3ea8fe0d7120ca542d": "p5_demo_comercial.png",
    "95bdda006deec6c2952325da063f9ac9": "p3_factores_dominio.png",
    # versión con E10
    "93d437b633f666a45f56016204373e88": "p5_comparacion_modelos.png",
    "d64c35c39ed2f8844c870aa63283299a": "p5_gradcam.png",
    "353b90293fcb4da62df3b540d4f2169d": "p5_robustez_factores.png",
    "6dbed06556dafa763ea6880273f95a26": "p5_desagregacion.png",
    "9a74ebd1fe692fd73af8ece4b93ae661": "p5_demo_comercial.png",
    "054c50b71179a83677294114ca812890": "p4_curvas.png",
    # versión con E11
    "48ba886ec0b7603a5664942c7da5371b": "p5_comparacion_modelos.png",
    "3d566e530db7bf0552b03909722f0492": "p5_gradcam.png",
    "c2165e382b7ce43dfe0d23dcf63746de": "p5_robustez_factores.png",
    "e6f0baaa128a9c9f7e2d20363ba16eca": "p5_desagregacion.png",
    "e425044200c046bb47bb07ace081c3a3": "p5_demo_comercial.png",
    "bd43fc7da53e315e396bbfdab92bb678": "p4_curvas.png",
}

deck = json.loads((origen / "project" / "deck.json").read_text(encoding="utf-8"))
hrefs = sorted({f["href"] for f in deck["faces"].values() if "href" in f})


def embeber(m):
    archivo = figuras / BLOBS[m.group(1)]
    return 'src="data:image/png;base64,' + base64.b64encode(archivo.read_bytes()).decode() + '"'


def flecha(m):
    """<x-shape kind="arrow-right"> del formato Slides -> SVG equivalente (el navegador no lo conoce)."""
    w, h, color = m.group(1), m.group(2), m.group(3)
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 10 6" preserveAspectRatio="none" style="flex:none">'
            f'<polygon points="0,1.8 6,1.8 6,0 10,3 6,6 6,4.2 0,4.2" fill="{color}"/></svg>')


def conector(m):
    """<x-connector> (flecha entre pasos de un flujo) -> SVG."""
    return ('<svg width="66" height="24" viewBox="0 0 66 24" style="flex:none">'
            '<line x1="6" y1="12" x2="52" y2="12" stroke="#6A7180" stroke-width="3"/>'
            '<polygon points="50,5 62,12 50,19" fill="#6A7180"/></svg>')


# CSS compartido del deck (formato v5); los decks v4 usan estilos en línea
css_deck = ""
for ruta in deck.get("format", {}).get("shared", []):
    css_deck += (origen / ruta).read_text(encoding="utf-8")

secciones = []
for sid in deck["order"]:
    html = (origen / "project" / "slides" / f"{sid}.html").read_text(encoding="utf-8")
    html = re.sub(r"<aside>.*?</aside>", "", html, flags=re.S)
    html = re.sub(r'src="/_blob/([0-9a-f]{32})"', embeber, html)
    html = re.sub(r'<x-shape kind="arrow-right" style="width:(\d+)px;height:(\d+)px;background:(#[0-9A-Fa-f]+)"></x-shape>',
                  flecha, html)
    html = re.sub(r"<x-connector[^>]*></x-connector>", conector, html)
    secciones.append(html)

links = "\n".join(f'<link rel="stylesheet" href="{h}">' for h in hrefs)
pagina = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>{deck['title']}</title>
{links}
<style>
@page {{ size: 1920px 1080px; margin: 0 }}
* {{ box-sizing: border-box; margin: 0; padding: 0 }}
html, body {{ background: #888 }}
section {{ position: relative; width: 1920px; height: 1080px; overflow: hidden; page-break-after: always; break-after: page }}
h1, h2, h3, p {{ font-weight: inherit }}
table {{ border-collapse: collapse; width: 100% }}
th, td {{ padding: 0.35em 0.6em; border-bottom: 1px solid #D8D3C8; text-align: left }}
th {{ font-weight: 700 }}
img {{ display: block }}
</style>
<style>{css_deck}
* {{ animation: none !important }}
</style></head><body>
{chr(10).join(secciones)}
</body></html>"""
destino.write_text(pagina, encoding="utf-8")
print("ok", destino, len(secciones), "slides")
