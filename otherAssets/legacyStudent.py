import json

class Colorum:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Student:
    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = password
        self.courses = []
        self.progress = {}
        self.links = {}

    def enroll(self, course):
        self.courses.append(course)
        self.progress[course.title] = 0
        self.links[course.title] = course.links[:]
        print(f"{Colorum.OKGREEN}Te has inscrito en el curso '{course.title}'. ¡Comienza tu viaje de aprendizaje!{Colorum.ENDC}")

    def view_progress(self):
        print(f"\n{Colorum.HEADER}Tu progreso en los cursos:{Colorum.ENDC}")
        if not self.progress:
            print(f"{Colorum.WARNING}No estás inscrito en ningún curso.{Colorum.ENDC}")
        for course, progress in self.progress.items():
            print(f" - Curso: {course}, Progreso: {progress}%")

    def advance_progress(self, course_title):
        if course_title in self.progress:
            if self.links[course_title]:
                print(f"{Colorum.WARNING}Quedan enlaces de lectura para avanzar en el curso '{course_title}'. Completa la lectura primero.{Colorum.ENDC}")
            else:
                self.progress[course_title] += 10  # Simulación de progreso
                print(f"{Colorum.OKBLUE}Avanzaste en el curso '{course_title}'. Tu progreso ahora es {self.progress[course_title]}%.{Colorum.ENDC}")
        else:
            print(f"{Colorum.FAIL}No estás inscrito en este curso. Primero inscríbete.{Colorum.ENDC}")

    def complete_link(self, course_title):
        if course_title in self.links and self.links[course_title]:
            completed_link = self.links[course_title].pop(0)
            print(f"{Colorum.OKBLUE}Lectura completada: {completed_link}.{Colorum.ENDC}")
            if not self.links[course_title]:  # Si no quedan enlaces
                print(f"{Colorum.OKGREEN}¡Has completado todas las lecturas! Ahora puedes avanzar en el curso.{Colorum.ENDC}")
        else:
            print(f"{Colorum.FAIL}No hay enlaces pendientes para este curso o no estás inscrito en él.{Colorum.ENDC}")

    def obtain_certificate(self, course_title):
        if self.progress.get(course_title, 0) >= 100:
            print(f"{Colorum.OKGREEN}¡Felicidades! Has completado el curso '{course_title}'. Aquí está tu certificado.{Colorum.ENDC}")
        else:
            print(f"{Colorum.WARNING}Aún no has completado el curso '{course_title}'. Continúa para obtener el certificado.{Colorum.ENDC}")

    def save_progress(self):
        with open('student_data.json', 'w') as f:
            json.dump(self.__dict__, f)
        print(f"{Colorum.OKGREEN}Progreso guardado exitosamente.{Colorum.ENDC}")

    def load_progress(self):
        try:
            with open('student_data.json', 'r') as f:
                data = json.load(f)
                self.__dict__.update(data)
            print(f"{Colorum.OKGREEN}Progreso cargado exitosamente.{Colorum.ENDC}")
        except FileNotFoundError:
            print(f"{Colorum.WARNING}No se encontró archivo de progreso. Comenzando nuevo registro.{Colorum.ENDC}")

class Course:
    def __init__(self, title, description, price, links):
        self.title = title
        self.description = description
        self.price = price
        self.links = links

class RecommendationEngine:
    def recommend_courses(self, student, available_courses):
        print(f"\n{Colorum.HEADER}Recomendaciones personalizadas para {student.name}:{Colorum.ENDC}")
        for course in available_courses:
            print(f" - Curso: {course.title}, Descripción: {course.description}")

# Crear cursos de ejemplo con enlaces de lectura
course_python = Course("Fundamentals of Python", "Learn the fundamentals of Python programming", 100, 
                       ["https://python.org/doc1", "https://python.org/doc2", "https://python.org/doc3"])
course_js = Course("Advanced JavaScript", "Deep dive into advanced JavaScript techniques", 150, 
                   ["https://jsdoc.org/doc1", "https://jsdoc.org/doc2"])
course_data_science = Course("Data Science", "Learn techniques for data analysis", 200, 
                              ["https://datascience.com/doc1"])

available_courses = [course_python, course_js, course_data_science]

# Crear un estudiante de ejemplo
student = Student("Shauna Vayne", "lorem@example.com", "LoremPassword123")

# Cargar progreso del estudiante
student.load_progress()

# Crear el motor de recomendaciones
recommendation_engine = RecommendationEngine()

def show_menu():
    print(f"\n{Colorum.HEADER}--- Plataforma EdTech ---{Colorum.ENDC}")
    print("1. Ver Cursos Disponibles")
    print("2. Inscribirse en un Curso")
    print("3. Ver Progreso en Cursos")
    print("4. Marcar Enlace como Completado")
    print("5. Avanzar en un Curso")
    print("6. Obtener Certificado")
    print("7. Recibir Recomendaciones")
    print("8. Guardar Progreso")
    print("9. Salir")

def execute_option(option):
    match option:
        case "1":
            print(f"\n{Colorum.OKBLUE}Cursos Disponibles:{Colorum.ENDC}")
            for i, course in enumerate(available_courses, start=1):
                print(f"{i}. Curso: {course.title} - {course.description} (Precio: {course.price} monedas)")
        case "2":
            print(f"\n{Colorum.OKBLUE}Selecciona un curso para inscribirte:{Colorum.ENDC}")
            for i, course in enumerate(available_courses, start=1):
                print(f"{i}. Curso: {course.title}")
            selection = int(input("Introduce el número del curso: ")) - 1
            student.enroll(available_courses[selection])
        case "3":
            student.view_progress()
        case "4":
            course_title = input(f"\n{Colorum.OKBLUE}Introduce el título del curso para marcar un enlace como completado: {Colorum.ENDC}")
            student.complete_link(course_title)
        case "5":
            course_title = input(f"\n{Colorum.OKBLUE}Introduce el título del curso para avanzar: {Colorum.ENDC}")
            student.advance_progress(course_title)
        case "6":
            course_title = input(f"\n{Colorum.OKBLUE}Introduce el título del curso para obtener el certificado: {Colorum.ENDC}")
            student.obtain_certificate(course_title)
        case "7":
            recommendation_engine.recommend_courses(student, available_courses)
        case "8":
            student.save_progress()
        case "9":
            print(f"{Colorum.OKGREEN}¡Gracias por usar la plataforma EdTech! ¡Sigue aprendiendo y alcanzando nuevas metas!{Colorum.ENDC}")
            return False
        case _:
            print(f"{Colorum.FAIL}Opción inválida. Selecciona un número entre 1 y 9.{Colorum.ENDC}")
    return True

def start_platform():
    running = True
    while running:
        show_menu()
        option = input(f"\n{Colorum.OKBLUE}Elige tu opción: {Colorum.ENDC}")
        running = execute_option(option)

# Iniciar la plataforma interactiva
start_platform()
