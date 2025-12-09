import PyPDF2
import os

def merge_pdfs_in_current_folder(output_name="allpdfsJoins.pdf"):
    # Obtener la ruta de la carpeta actual
    folder_path = os.getcwd()
    pdf_writer = PyPDF2.PdfWriter()
    
    # Obtener todos los archivos PDF en la carpeta actual
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    pdf_files.sort()  # Ordenar los archivos alfabéticamente

    # Verificar que existan archivos PDF
    if not pdf_files:
        print("No se encontraron archivos PDF en la carpeta actual.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        
        # Agregar cada página de cada archivo PDF
        for page in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page])

    # Guardar el archivo combinado en la carpeta actual
    output_path = os.path.join(folder_path, output_name)
    with open(output_path, 'wb') as output_pdf:
        pdf_writer.write(output_pdf)

    print(f"Se han unido los archivos en '{output_path}' exitosamente.")

if __name__ == "__main__":
    merge_pdfs_in_current_folder()
