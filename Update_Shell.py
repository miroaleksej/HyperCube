# Добавляем в update.py новую функцию для обновления оболочки
def update_hypercube_shell(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # ===== 1. Добавляем новые команды =====
    # Добавляем команды в автодополнение
    completer_pattern = r"completer=WordCompleter\(\["
    new_completer = r"""completer=WordCompleter([
                'create', 'define_law', 'build', 'query', 'quantum_query',
                'project', 'visualize_3d', 'save', 'load', 'optimize', 
                'discover_laws', 'solve_schrodinger', 'set_units', 
                'set_symmetries', 'topology', 'exit', 'help', 'status'"""
    
    content = re.sub(completer_pattern, new_completer, content)
    
    # ===== 2. Добавляем обработку новых команд =====
    # Шаблон для вставки новых обработчиков команд
    command_pattern = r"elif command == 'optimize':\n\s+self\.optimize_params\(args\)"
    
    # Новые обработчики команд
    new_commands = r"""
                elif command == 'quantum_query':
                    self.quantum_query_point(args)
                    
                elif command == 'solve_schrodinger':
                    self.solve_schrodinger_equation(args)
                    
                elif command == 'discover_laws':
                    self.discover_physical_laws(args)
                    
                elif command == 'set_units':
                    self.set_units(args)
                    
                elif command == 'set_symmetries':
                    self.set_symmetries(args)
                    
                elif command == 'topology':
                    self.compute_topology_info(args)"""
    
    content = re.sub(command_pattern, command_pattern + new_commands, content)
    
    # ===== 3. Добавляем новые методы в класс оболочки =====
    new_methods = r"""
    def quantum_query_point(self, args):
        \"\"\"Квантовый запрос значения в суперпозиции\"\"\"
        if self.cube is None:
            print("Ошибка: гиперкуб не инициализирован")
            return
            
        if not args:
            print("Ошибка: требуется точка (координаты через запятую)")
            return
            
        coords = self.parse_coordinates(args[0])
        if coords is None:
            return
            
        if len(coords) != len(self.cube.dimensions):
            print(f"Ошибка: требуется {len(self.cube.dimensions)} координат")
            return
            
        try:
            # Параметры неопределенности
            uncertainty = 0.1
            samples = 20
            
            if len(args) > 1:
                uncertainty = float(args[1])
            if len(args) > 2:
                samples = int(args[2])
                
            values = self.cube.quantum_query(coords, uncertainty, samples)
            print(f"Квантовые значения в точке:")
            for i, val in enumerate(values):
                print(f"  Состояние {i+1}: {val:.6f}")
            print(f"Среднее: {np.mean(values):.6f} ± {np.std(values):.6f}")
        except Exception as e:
            print(f"Ошибка квантового запроса: {str(e)}")

    def solve_schrodinger_equation(self, args):
        \"\"\"Решить уравнение Шредингера для текущего гиперкуба\"\"\"
        if self.cube is None:
            print("Ошибка: гиперкуб не инициализирован")
            return
            
        if not args:
            print("Ошибка: требуется выражение потенциала")
            return
            
        potential_expr = " ".join(args)
        mass = 1.0
        hbar = 1.0
        points = 1000
        
        # Парсинг дополнительных параметров
        if '-m' in args:
            idx = args.index('-m')
            mass = float(args[idx+1])
        if '-ħ' in args:
            idx = args.index('-ħ')
            hbar = float(args[idx+1])
        if '-p' in args:
            idx = args.index('-p')
            points = int(args[idx+1])
            
        print(f"Решение уравнения Шредингера с потенциалом: {potential_expr}")
        result = self.cube.solve_schrodinger(potential_expr, mass, hbar, points)
        
        # Визуализация результатов
        plt.figure(figsize=(12, 8))
        plt.plot(result['x'], result['potential'], 'k-', lw=2, label='Потенциал')
        
        # Нормировка волновых функций для визуализации
        for i in range(min(3, len(result['energies']))):
            psi = result['wavefunctions'][:, i].real
            psi = psi / np.max(np.abs(psi)) * 0.1 * np.ptp(result['potential'])
            plt.plot(result['x'], result['energies'][i] + psi, 
                    label=f'Ψ_{i+1} (E={result["energies"][i]:.3f})')
        
        plt.xlabel('Координата')
        plt.ylabel('Энергия')
        plt.title('Решение уравнения Шредингера')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print(f"Найденные уровни энергии:")
        for i, E in enumerate(result['energies']):
            print(f"  Уровень {i+1}: {E:.6f}")

    def discover_physical_laws(self, args):
        \"\"\"Поиск новых физических законов\"\"\"
        if self.cube is None:
            print("Ошибка: гиперкуб не инициализирован")
            return
            
        # Параметры по умолчанию
        n_samples = 5000
        pop_size = 10000
        generations = 20
        
        # Парсинг аргументов
        if args:
            try:
                n_samples = int(args[0])
                if len(args) > 1: pop_size = int(args[1])
                if len(args) > 2: generations = int(args[2])
            except ValueError:
                print("Ошибка: параметры должны быть целыми числами")
                return
                
        print(f"Поиск физических законов (samples={n_samples}, pop={pop_size}, gen={generations})...")
        laws = self.cube.discover_physical_laws(
            n_samples=n_samples,
            population_size=pop_size,
            generations=generations
        )
        
        # Вывод результатов
        print("\nНайденные физические законы:")
        for i, law in enumerate(laws):
            print(f"\nЗакон #{i+1} (Точность: {law['fitness']:.4f})")
            print(f"Исходное: {law['expression']}")
            print(f"Упрощенное: {law['simplified']}")
        
        # Сохранение законов
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"discovered_laws_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write("Найденные физические законы:\n")
            for i, law in enumerate(laws):
                f.write(f"\nЗакон #{i+1} (Точность: {law['fitness']:.4f})\n")
                f.write(f"Исходное: {law['expression']}\n")
                f.write(f"Упрощенное: {law['simplified']}\n")
        
        print(f"\nЗаконы сохранены в {filename}")

    def set_units(self, args):
        \"\"\"Установка единиц измерения для физических величин\"\"\"
        if self.cube is None:
            print("Ошибка: гиперкуб не инициализирован")
            return
            
        if len(args) != len(self.cube.dimensions):
            print(f"Ошибка: требуется {len(self.cube.dimensions)} единиц измерения")
            return
            
        units = {}
        for i, dim in enumerate(self.cube.dim_names):
            units[dim] = args[i]
            
        self.cube.units = units
        print("Единицы измерения установлены:")
        for dim, unit in units.items():
            print(f"  {dim}: {unit}")

    def set_symmetries(self, args):
        \"\"\"Установка симметрий для физической системы\"\"\"
        if self.cube is None:
            print("Ошибка: гиперкуб не инициализирован")
            return
            
        symmetries = []
        for symmetry in args:
            try:
                # Синтаксис: "x->-x,y->y"
                parts = symmetry.split(',')
                transform = {}
                for part in parts:
                    var, expr = part.split('->')
                    transform[var.strip()] = expr.strip()
                symmetries.append(transform)
            except:
                print(f"Ошибка формата симметрии: {symmetry}")
                print("Используйте формат: 'x->-x,y->y'")
                return
                
        self.cube.symmetries = symmetries
        print("Симметрии установлены:")
        for i, sym in enumerate(symmetries):
            print(f"  Симметрия {i+1}: {sym}")

    def compute_topology_info(self, args):
        \"\"\"Вычисление топологических характеристик\"\"\"
        if self.cube is None:
            print("Ошибка: гиперкуб не инициализирован")
            return
            
        method = 'riemannian' if 'riemann' in args else 'algebraic'
        print(f"Вычисление топологии ({method} метод)...")
        topology = self.cube.compute_topology(method)
        
        print("\nТопологические характеристики:")
        print(f"  Характеристика Эйлера: {topology.get('euler_characteristic', 'N/A')}")
        print(f"  Скалярная кривизна: {topology.get('scalar_curvature', 'N/A'):.4f}")
        print(f"  Числа Бетти: {topology.get('betti_numbers', [])}")
        
        if 'persistence_diagram' in topology:
            print("\nПерсистентная диаграмма:")
            for dim, points in topology['persistence_diagram'].items():
                print(f"  Размерность {dim}: {len(points)} особенностей")
    """
    
    # Вставляем новые методы в класс оболочки
    class_end = re.search(r"class QuantumHypercubeShell:.*?def", content, re.DOTALL)
    if class_end:
        insert_pos = class_end.end() - 3  # Перед последним методом
        content = content[:insert_pos] + new_methods + content[insert_pos:]
    
    # ===== 4. Обновляем справочную информацию =====
    help_pattern = r"optimize <целевое_значение> - Оптимизировать параметры"
    new_help = r"""optimize <целевое_значение> - Оптимизировать параметры
  quantum_query <точка> [неопределенность] [образцы] - Квантовый запрос
  solve_schrodinger <потенциал> [-m масса] [-ħ hbar] [-p точки] - Решить УШ
  discover_laws [samples] [pop] [gen] - Поиск новых физических законов
  set_units <единицы...> - Установить единицы измерения
  set_symmetries <симметрии> - Установить симметрии системы
  topology [riemann|algebraic] - Вычислить топологические характеристики"""
    
    content = re.sub(help_pattern, new_help, content)
    
    # ===== 5. Добавляем необходимые импорты =====
    import_pattern = r"import numpy as np"
    new_imports = r"""import numpy as np
import matplotlib.pyplot as plt
import time
"""
    
    content = re.sub(import_pattern, new_imports, content)
    
    # Сохраняем обновленный файл
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Файл {filename} успешно обновлен!")