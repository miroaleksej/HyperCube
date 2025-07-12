# update_3.py
import re
import os

def update_quantum_hypercube(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # ===== 1. Добавляем фрактальные гиперструктуры =====
    fractal_code = r"""
    def fractal_partition(self, depth=3, mutation_rate=0.05):
        \"\"\"
        Рекурсивное создание самоподобных гиперструктур
        :param depth: глубина рекурсии (уровни вложенности)
        :param mutation_rate: вероятность мутации физических законов
        :return: список дочерних гиперкубов
        \"\"\"
        if depth <= 0:
            return []
            
        print(f"Создание фрактальных структур (уровень {4-depth})...")
        child_cubes = []
        
        # Генерация октантов
        octants = self._split_into_octants()
        
        for i, octant_dims in enumerate(octants):
            # Создание дочернего гиперкуба
            child = QuantumHypercube(
                dimensions=octant_dims,
                resolution=self.resolution//2,
                quantum_correction=self.quantum_correction,
                hbar=self.hbar * 0.5  # Масштабирование ħ
            )
            
            # Мутация физического закона
            if np.random.rand() < mutation_rate:
                mutated_law = self._mutate_expression(self.law_expression)
                child.define_physical_law(mutated_law)
                mutation_info = " [мутировавший закон]"
            else:
                child.define_physical_law(self.law_expression)
                mutation_info = ""
            
            # Рекурсивное создание
            grandchildren = child.fractal_partition(depth-1, mutation_rate*0.7)
            
            child_cubes.append({
                'hypercube': child,
                'children': grandchildren,
                'origin_octant': i
            })
            
            print(f"Создан октант #{i+1}{mutation_info}")
        
        return child_cubes

    def _split_into_octants(self):
        \"\"\"Разбиение пространства на 2^d подобластей\"\"\"
        import itertools
        dim_count = len(self.dim_names)
        octants = []
        
        # Генерируем все комбинации битовых масок
        for bitmask in itertools.product([0,1], repeat=dim_count):
            new_dims = {}
            for i, dim in enumerate(self.dim_names):
                low, high = self.dimensions[dim]
                mid = (low + high) / 2
                # Выбираем половину пространства по битовой маске
                new_dims[dim] = (low, mid) if bitmask[i] == 0 else (mid, high)
            octants.append(new_dims)
            
        return octants

    def _mutate_expression(self, expression, intensity=0.1):
        \"\"\"Мутация физического закона через символьные преобразования\"\"\"
        from sympy import sympify, Add, Mul, sin, cos
        import random
        
        # Разрешенные мутации
        mutations = [
            lambda expr: expr * random.uniform(0.9, 1.1),  # Масштабирование
            lambda expr: expr + random.uniform(-0.1, 0.1),  # Сдвиг
            lambda expr: Mul(expr, sin(random.choice(self.dim_names)) if random.random() > 0.5 else expr,
            lambda expr: Add(expr, cos(random.choice(self.dim_names)) if random.random() > 0.5 else expr
        ]
        
        # Применяем мутации с вероятностью intensity
        expr = sympify(expression)
        for _ in range(int(1/intensity)):
            if random.random() < intensity:
                expr = random.choice(mutations)(expr)
                
        return str(expr).replace(' ', '')

    def fractal_query(self, point, depth=3):
        \"\"\"Запрос значения в фрактальной структуре\"\"\"
        # Определяем принадлежность к октанту
        octant_idx = self._find_octant(point)
        if octant_idx is None or depth <= 0:
            return self.query(point)
            
        # Рекурсивный запрос в дочерней структуре
        child = self.child_cubes[octant_idx]['hypercube']
        return child.fractal_query(point, depth-1)

    def _find_octant(self, point):
        \"\"\"Определение индекса октанта для точки\"\"\"
        octants = self._split_into_octants()
        for i, dims in enumerate(octants):
            in_octant = True
            for dim, (low, high) in dims.items():
                if not (low <= point[self.dim_names.index(dim)] <= high):
                    in_octant = False
                    break
            if in_octant:
                return i
        return None

    def visualize_fractal(self, projection_dims, depth=3, resolution=512):
        \"\"\"Визуализация фрактальной структуры\"\"\"
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Рекурсивное построение границ
        def draw_bounds(hypercube, current_depth):
            if current_depth > depth:
                return
                
            # Получаем границы для проекции
            dim1, dim2 = projection_dims
            x_min, x_max = hypercube.dimensions[dim1]
            y_min, y_max = hypercube.dimensions[dim2]
            
            # Рисуем прямоугольник
            rect = plt.Rectangle(
                (x_min, y_min), 
                x_max - x_min, 
                y_max - y_min,
                fill=False,
                edgecolor='blue',
                linewidth=1.5 - 0.3*current_depth,
                alpha=1.0 - 0.2*current_depth
            )
            ax.add_patch(rect)
            
            # Рекурсивно для дочерних структур
            if hasattr(hypercube, 'child_cubes'):
                for child_data in hypercube.child_cubes:
                    draw_bounds(child_data['hypercube'], current_depth+1)
        
        # Начинаем с текущего гиперкуба
        draw_bounds(self, 0)
        
        # Настройка графика
        ax.set_title(f"Фрактальная структура (глубина {depth})")
        ax.set_xlabel(projection_dims[0])
        ax.set_ylabel(projection_dims[1])
        ax.grid(True)
        
        # Сохранение в PNG
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf
    """
    
    # Вставляем новый код перед последней скобкой класса
    class_end = content.rfind('}')
    if class_end != -1:
        content = content[:class_end] + fractal_code + '\n' + content[class_end:]
    
    # Добавляем атрибут child_cubes в __init__
    init_pattern = r"def __init__\(self.*?\):"
    init_insert = r"""
        self.child_cubes = None  # Дочерние фрактальные структуры"""
    
    content = re.sub(
        init_pattern, 
        lambda m: m.group(0) + init_insert, 
        content
    )
    
    # Обновляем метод __repr__
    repr_pattern = r"f\"QuantumHypercube\(dimensions={len\(self.dimensions\)},"
    new_repr = r"f\"QuantumHypercube(dimensions={len(self.dimensions)}," \
               r" fractal={bool(self.child_cubes)},"""
    
    content = re.sub(repr_pattern, new_repr, content)
    
    # ===== 2. Обновляем зависимости =====
    import_pattern = r"import (numpy as np|re)"
    new_imports = r"""import numpy as np
import itertools
import random
import io
"""
    
    content = re.sub(import_pattern, new_imports, content, count=1)
    
    # Сохраняем обновленный файл
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Файл {filename} успешно обновлен фрактальными функциями!")

def update_hypercube_shell(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # ===== 1. Добавляем новые команды =====
    # Обновляем автодополнение
    completer_pattern = r"completer=WordCompleter\(\["
    new_completer = r"""completer=WordCompleter([
                'create', 'define_law', 'build', 'query', 'fractal_query', 
                'project', 'visualize_3d', 'visualize_fractal', 
                'fractalize', 'evolve', 'save', 'load', 'optimize', 
                'discover_laws', 'solve_schrodinger', 'set_units', 
                'set_symmetries', 'topology', 'exit', 'help', 'status'"""
    
    content = re.sub(completer_pattern, new_completer, content)
    
    # Добавляем обработчики команд
    command_pattern = r"elif command == 'topology':\n\s+self\.compute_topology_info\(args\)"
    new_commands = r"""
                elif command == 'fractalize':
                    self.create_fractal(args)
                    
                elif command == 'fractal_query':
                    self.fractal_query_point(args)
                    
                elif command == 'visualize_fractal':
                    self.visualize_fractal_structure(args)
                    
                elif command == 'evolve':
                    self.evolve_fractal(args)"""
    
    content = re.sub(command_pattern, command_pattern + new_commands, content)
    
    # ===== 2. Добавляем методы оболочки =====
    shell_methods = r"""
    def create_fractal(self, args):
        \"\"\"Создание фрактальной структуры гиперкуба\"\"\"
        if self.cube is None:
            print("Ошибка: гиперкуб не инициализирован")
            return
            
        depth = 3
        mutation_rate = 0.05
        
        if args:
            try:
                depth = int(args[0])
                if len(args) > 1: mutation_rate = float(args[1])
            except ValueError:
                print("Ошибка: неверные параметры")
                return
                
        print(f"Создание фрактальной структуры (глубина={depth}, мутации={mutation_rate})...")
        self.cube.child_cubes = self.cube.fractal_partition(depth, mutation_rate)
        print(f"Создано {sum(len(c['children']) for c in self.cube.child_cubes)} дочерних структур")
        
    def fractal_query_point(self, args):
        \"\"\"Запрос значения во фрактальной структуре\"\"\"
        if self.cube is None or not self.cube.child_cubes:
            print("Ошибка: фрактальная структура не создана")
            return
            
        if len(args) < 1:
            print("Ошибка: требуется точка (координаты через запятую)")
            return
            
        coords = self.parse_coordinates(args[0])
        if coords is None:
            return
            
        depth = 3
        if len(args) > 1:
            depth = int(args[1])
            
        value = self.cube.fractal_query(coords, depth)
        print(f"Фрактальное значение (глубина {depth}): {value:.6f}")
        
    def visualize_fractal_structure(self, args):
        \"\"\"Визуализация фрактальной структуры\"\"\"
        if self.cube is None or not self.cube.child_cubes:
            print("Ошибка: фрактальная структура не создана")
            return
            
        if len(args) < 2:
            print("Ошибка: требуется два измерения для проекции (например, x y)")
            return
            
        dim1, dim2 = args[0], args[1]
        depth = 3
        if len(args) > 2:
            depth = int(args[2])
            
        print(f"Визуализация фрактальной структуры ({dim1} vs {dim2}, глубина={depth})")
        buf = self.cube.visualize_fractal([dim1, dim2], depth)
        img = Image.open(buf)
        img.show()
        
    def evolve_fractal(self, args):
        \"\"\"Эволюция фрактальной структуры\"\"\"
        if self.cube is None or not self.cube.child_cubes:
            print("Ошибка: фрактальная структура не создана")
            return
            
        mutation_rate = 0.1
        if args:
            try:
                mutation_rate = float(args[0])
            except ValueError:
                print("Ошибка: вероятность мутации должна быть числом")
                return
                
        print(f"Эволюция фрактала (мутации={mutation_rate})...")
        self._recursive_evolve(self.cube.child_cubes, mutation_rate)
        print("Эволюция завершена!")
        
    def _recursive_evolve(self, child_cubes, mutation_rate):
        \"\"\"Рекурсивная эволюция дочерних структур\"\"\"
        for child_data in child_cubes:
            child = child_data['hypercube']
            # Мутация закона
            mutated_law = child._mutate_expression(child.law_expression, mutation_rate)
            child.define_physical_law(mutated_law)
            
            # Рекурсивная обработка детей
            if child_data['children']:
                self._recursive_evolve(child_data['children'], mutation_rate * 0.7)
    """
    
    # Вставляем новые методы в класс оболочки
    class_end = re.search(r"class QuantumHypercubeShell:.*?def", content, re.DOTALL)
    if class_end:
        insert_pos = class_end.end() - 3
        content = content[:insert_pos] + shell_methods + content[insert_pos:]
    
    # ===== 3. Обновляем справочную информацию =====
    help_pattern = r"topology \[riemann\|algebraic\] - Вычислить топологические характеристики"
    new_help = r"""topology [riemann|algebraic] - Вычислить топологические характеристики
  fractalize [глубина] [мутации] - Создать фрактальную структуру
  fractal_query <точка> [глубина] - Запрос во фрактале
  visualize_fractal <dim1> <dim2> [глубина] - Визуализация фрактала
  evolve [мутации] - Эволюция фрактальной структуры"""
    
    content = re.sub(help_pattern, new_help, content)
    
    # ===== 4. Добавляем необходимые импорты =====
    import_pattern = r"import numpy as np"
    new_imports = r"""import numpy as np
from PIL import Image
import io
"""
    
    content = re.sub(import_pattern, new_imports, content)
    
    # Сохраняем обновленный файл
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Файл {filename} успешно обновлен для работы с фракталами!")

def create_fractal_demo():
    """Создание демонстрационного скрипта для фрактальных структур"""
    demo_code = '''# fractal_demo.py
from quantum_hypercube import QuantumHypercube

# Создаем базовый гиперкуб
dims = {'x': (-10, 10), 'y': (-10, 10), 'z': (-5, 5)}
cube = QuantumHypercube(dims, resolution=64)
cube.define_physical_law("sin(x)*cos(y) + 0.1*z")

# Создаем фрактальную структуру
print("Создание фрактальной вселенной...")
fractal_structure = cube.fractal_partition(depth=4, mutation_rate=0.1)

# Визуализация
print("Визуализация фрактала...")
buf = cube.visualize_fractal(['x', 'y'], depth=4)
with open('fractal_universe.png', 'wb') as f:
    f.write(buf.getbuffer())

print("Демонстрация завершена! Результат сохранен в fractal_universe.png")
'''
    with open("fractal_demo.py", "w") as f:
        f.write(demo_code)
    print("Демонстрационный скрипт создан: fractal_demo.py")

if __name__ == "__main__":
    # Обновляем основной файл
    update_quantum_hypercube("quantum_hypercube.py")
    
    # Обновляем оболочку
    update_hypercube_shell("quantum_hypercube_shell.py")
    
    # Создаем демо-скрипт
    create_fractal_demo()
    
    print("\nОбновление фрактальных структур завершено!")
    print("Новые возможности:")
    print("1. fractalize - создание самоподобных гиперструктур")
    print("2. fractal_query - запросы в иерархической структуре")
    print("3. visualize_fractal - визуализация фрактальной геометрии")
    print("4. evolve - эволюция физических законов в дочерних структурах")
    print("\nДля тестирования запустите: python fractal_demo.py")