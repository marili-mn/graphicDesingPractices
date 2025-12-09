import qrcode
from PIL import Image, ImageDraw, ImageOps

# === Configuración ===
URL = "https://biolink.site/juntospormocoreta"
LOGO_PATH = r"C:\Users\Administrador\Desktop\qrJob\juntosxmocoretaLogo.jpg"
OUTPUT_PATH = "qrJobRAW2.jpg"
DPI = 300
CM_TO_INCH = 2.54

# Calcular dimensiones en píxeles
def cm_to_pixels(cm, dpi=DPI):
    inches = cm / CM_TO_INCH
    return round(inches * dpi)

# Tamaño final (50x50 cm) con un margen mínimo
FINAL_SIZE = cm_to_pixels(50)
QR_MARGIN_CM = 1  # Margen de 1 cm en cada lado
QR_SIZE = cm_to_pixels(50 - 2 * QR_MARGIN_CM) # QR de 48x48 cm
LOGO_SIZE = cm_to_pixels(10) # Logo de 10x10 cm para mejor visibilidad

# === Crear QR optimizado ===
qr = qrcode.QRCode(
    version=None,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    border=4,
    box_size=10  # Valor base que ajustaremos después
)
qr.add_data(URL)
qr.make(fit=True)

# Calcular tamaño real del QR
modules_count = qr.modules_count
box_size = max(1, QR_SIZE // (modules_count + 2 * qr.border))

# Regenerar QR con el tamaño correcto
qr = qrcode.QRCode(
    version=qr.version,
    error_correction=qr.error_correction,
    box_size=box_size,
    border=qr.border
)
qr.add_data(URL)
qr_img = qr.make_image(
    fill_color="black",
    back_color="white"
).convert('RGB')

# === Procesar logo JPG a color ===
# logo = Image.open(LOGO_PATH).convert('RGB')
# 
# # Escalar manteniendo relación de aspecto
# logo_ratio = min(LOGO_SIZE / logo.width, LOGO_SIZE / logo.height)
# new_size = (
#     max(1, int(logo.width * logo_ratio)),
#     max(1, int(logo.height * logo_ratio))
# )
# logo = logo.resize(new_size, Image.LANCZOS)
# 
# # Crear máscara cuadrada para el logo
# mask = Image.new('L', new_size, 0)
# draw = ImageDraw.Draw(mask)
# draw.rectangle([0, 0, new_size[0], new_size[1]], fill=255)
# 
# # === Insertar logo en el centro del QR ===
# # Calcula las coordenadas para centrar el logo.
# position = (
#     (qr_img.width - new_size[0]) // 2,
#     (qr_img.height - new_size[1]) // 2
# )
# 
# # Se crea un fondo blanco del tamaño del logo para tapar el QR que hay detrás.
# bg = Image.new('RGB', new_size, (255, 255, 255))
# # Pega el fondo blanco en la imagen del QR.
# qr_img.paste(bg, position, mask)
# 
# # Pega el logo sobre el fondo blanco.
# qr_img.paste(logo, position, mask)

# === Crear lienzo final 50x50cm ===
canvas = Image.new('RGB', (FINAL_SIZE, FINAL_SIZE), (255, 255, 255))
margin = (FINAL_SIZE - QR_SIZE) // 2
canvas.paste(qr_img, (margin, margin))

# Añadir borde profesional
border_width = cm_to_pixels(0.5)  # 5mm
draw = ImageDraw.Draw(canvas)
draw.rectangle(
    [0, 0, FINAL_SIZE-1, FINAL_SIZE-1],
    outline="black",
    width=border_width
)

# === Guardar en máxima calidad ===
canvas.save(
    OUTPUT_PATH,
    dpi=(DPI, DPI),
    quality=100,
    subsampling=0,
    optimize=True
)

print(f"QR generado: {OUTPUT_PATH}")
print(f"Tamaño físico: 50x50 cm ({FINAL_SIZE}x{FINAL_SIZE} px)")
print(f"Logo tamaño: {LOGO_SIZE}px ({LOGO_SIZE/DPI*CM_TO_INCH:.1f}cm)")