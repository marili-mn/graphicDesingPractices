import qrcode
from PIL import Image, ImageDraw

# === Configuración ===
url = "https://marili-mn.github.io/frontend-portfolio/"
logo_path = r"C:\Users\Administrador\Desktop\anrufenLichCast\frontend-portfolio\assets\cardVectors\imgCardLogos\dragon-qr.png"
output_path = "evoke_RAW_QR.png"

# === Crear QR con margen reducido (2 módulos) ===
qr = qrcode.QRCode(
    version=5,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=25,
    border=2  # Mínimo funcional sin romper escaneabilidad
)
qr.add_data(url)
qr.make(fit=True)

# === Imagen QR ===
qr_img = qr.make_image(fill_color="black", back_color="white").convert('RGBA')
qr_width, qr_height = qr_img.size
module_count = qr.modules_count
module_size = qr_width // module_count

# === Preparar logo ===
logo = Image.open(logo_path).convert('RGBA')
logo_modules = 10
logo_size = logo_modules * module_size
logo = logo.resize((logo_size, logo_size), Image.LANCZOS)

# === Fondo blanco detrás del logo ===
logo_x = (qr_width - logo.width) // 2
logo_y = (qr_height - logo.height) // 2

draw = ImageDraw.Draw(qr_img)
draw.rounded_rectangle(
    [logo_x, logo_y, logo_x + logo.width, logo_y + logo.height],
    radius=module_size * 1.5,
    fill=(255, 255, 255, 255)
)

# === Pegar logo ===
qr_img.paste(logo, (logo_x, logo_y), logo)

# === Crear lienzo con marco exterior ===
marco_grosor = 8  # Grosor del borde externo en píxeles
marco_color = (0, 0, 0, 255)  # Negro sólido

canvas_width = qr_width + 2 * marco_grosor
canvas_height = qr_height + 2 * marco_grosor
canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))

# Dibujar el marco (rectángulo negro)
draw_canvas = ImageDraw.Draw(canvas)
draw_canvas.rectangle(
    [0, 0, canvas_width - 1, canvas_height - 1],
    outline=marco_color,
    width=marco_grosor
)

# Pegar el QR centrado dentro del marco
canvas.paste(qr_img, (marco_grosor, marco_grosor), qr_img)

# === Guardar resultado final ===
canvas.save(output_path, quality=100)
print(f"✅ QR compacto con marco guardado como: {output_path}")
