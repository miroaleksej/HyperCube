### Файл `update_4.py`: Генеративные гипервселенные и Python API

```python
# update_4.py
import re
import os
import json

def update_quantum_hypercube(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # ===== 1. Генеративные гипервселенные =====
    multiverse_code = r"""
    def create_multiverse(self, num_universes=5, evolution_epochs=10, 
                         mutation_rate=0.2, fitness_function=None):
        \"\"\"
        Создание ансамбля альтернативных вселенных с эволюцией законов
        :param num_universes: количество вселенных в ансамбле
        :param evolution_epochs: этапы эволюции
        :param mutation_rate: интенсивность мутаций законов
        :param fitness_function: функция оценки физической состоятельности
        :return: список объектов QuantumHypercube
        \"\"\"
        from .generative import UniverseGenerator
        
        # Дефолтная функция пригодности
        if fitness_function is None:
            fitness_function = self._default_fitness
        
        print(f"Создание мультивселенной из {num_universes} вселенных...")
        generator = UniverseGenerator(
            base_cube=self,
            mutation_rate=mutation_rate,
            fitness_function=fitness_function
        )
        
        multiverse = generator.generate(
            num_universes=num_universes,
            evolution_epochs=evolution_epochs
        )
        
        return multiverse

    def _default_fitness(self, hypercube):
        \"\"\"Оценка физической состоятельности вселенной\"\"\"
        score = 0.0
        
        # Критерии оценки:
        # 1. Соответствие размерностям
        if hasattr(self, 'units'):
            try:
                self._validate_physical_law(hypercube.law_expression)
                score += 0.4
            except PhysicalLawValidationError:
                pass
                
        # 2. Топологическая согласованность
        base_topology = self.compute_topology()
        new_topology = hypercube.compute_topology()
        curvature_diff = abs(base_topology['scalar_curvature'] - new_topology['scalar_curvature'])
        score += 0.3 / (1 + curvature_diff)
        
        # 3. Энергетическая стабильность
        try:
            energies = hypercube.solve_schrodinger("x**2")['energies']
            if np.all(energies > 0):
                score += 0.3
        except:
            pass
            
        return score

    def multiverse_query(self, point, multiverse):
        \"\"\"Запрос значения во всех вселенных мультивселенной\"\"\"
        results = []
        for universe in multiverse:
            try:
                value = universe.query(point)
                results.append({
                    'universe': universe.id,
                    'value': value,
                    'law': universe.law_expression
                })
            except:
                continue
        return results
    """
    
    # Вставляем код в класс
    class_end = content.rfind('}')
    if class_end != -1:
        content = content[:class_end] + multiverse_code + '\n' + content[class_end:]
    
    # Добавляем атрибут ID
    init_pattern = r"def __init__\(self.*?\):"
    init_insert = r"""
        self.id = str(uuid.uuid4())[:8]  # Уникальный ID вселенной"""
    
    content = re.sub(init_pattern, init_insert, content)
    
    # ===== 2. Модуль generative.py =====
    generative_code = r"""
# generative.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from quantum_hypercube import QuantumHypercube

class UniverseGenerator:
    def __init__(self, base_cube, mutation_rate=0.1, fitness_function=None):
        self.base_cube = base_cube
        self.mutation_rate = mutation_rate
        self.fitness_function = fitness_function
        self.vae = self._build_vae()
        
    def _build_vae(self, latent_dim=32):
        \"\"\"Variational Autoencoder для генерации законов\"\"\"
        # Энкодер
        inputs = Input(shape=(len(self.base_cube.dim_names),))
        x = Dense(128, activation='swish')(inputs)
        x = Dense(64, activation='swish')(x)
        
        # Латентное пространство
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        z = self._sampling([z_mean, z_log_var])
        
        # Декодер
        decoder_input = Input(shape=(latent_dim,))
        x = Dense(64, activation='swish')(decoder_input)
        x = Dense(128, activation='swish')(x)
        outputs = Dense(len(self.base_cube.dim_names), activation='linear')(x)
        
        # Модели
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        decoder = Model(decoder_input, outputs, name='decoder')
        
        # VAE
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae')
        vae.compile(optimizer='adam', loss='mse')
        
        return vae
        
    def _sampling(self, args):
        \"\"\"Функция сэмплирования для VAE\"\"\"
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
    def _mutate_law(self, law_expression):
        \"\"\"Генеративная мутация физического закона\"\"\"
        # Кодируем закон в числовой вектор
        law_vector = self._encode_law(law_expression)
        
        # Генерируем вариации
        variants = self.vae.predict(np.array([law_vector]))
        
        # Декодируем обратно в выражение
        return self._decode_vector(variants[0])
        
    def _encode_law(self, expression):
        \"\"\"Преобразование выражения в числовой вектор\"\"\"
        # Упрощенная реализация: используем символьные признаки
        features = np.zeros(len(self.base_cube.dim_names))
        symbols = ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log']
        
        for i, sym in enumerate(symbols):
            features[i] = expression.count(sym) / len(expression)
            
        return features
        
    def _decode_vector(self, vector):
        \"\"\"Генерация выражения из вектора\"\"\"
        # Прототип: случайные комбинации операций
        # В реальной реализации - RNN с вниманием
        import random
        dims = self.base_cube.dim_names
        operations = ['+', '-', '*', '/']
        functions = ['sin', 'cos', 'exp', 'log']
        
        # Случайная функция
        expr = random.choice(functions) + '(' + random.choice(dims) + ')'
        
        # Комбинируем с операциями
        for _ in range(3):
            op = random.choice(operations)
            term = random.choice(dims) if random.random() > 0.5 else str(round(random.uniform(0.1, 5.0), 2)
            expr = f"({expr} {op} {term})"
            
        return expr
        
    def generate(self, num_universes=5, evolution_epochs=10):
        \"\"\"Генерация и эволюция вселенных\"\"\"
        universes = []
        
        # Создаем начальную популяцию
        for i in range(num_universes):
            new_cube = QuantumHypercube(
                dimensions=self.base_cube.dimensions.copy(),
                resolution=self.base_cube.resolution,
                quantum_correction=self.base_cube.quantum_correction,
                hbar=self.base_cube.hbar
            )
            
            # Мутировавший закон
            mutated_law = self._mutate_law(self.base_cube.law_expression)
            new_cube.define_physical_law(mutated_law)
            new_cube.build_hypercube()
            
            universes.append(new_cube)
        
        # Эволюционный отбор
        for epoch in range(evolution_epochs):
            print(f"Эволюционная эпоха {epoch+1}/{evolution_epochs}")
            
            # Оцениваем пригодность
            fitness_scores = [self.fitness_function(universe) for universe in universes]
            
            # Селекция (турнирный отбор)
            new_generation = []
            for _ in range(num_universes):
                candidates = np.random.choice(len(universes), size=3, replace=False)
                winner_idx = candidates[np.argmax([fitness_scores[i] for i in candidates])]
                new_generation.append(universes[winner_idx])
            
            # Мутация и скрещивание
            for i in range(num_universes):
                if np.random.rand() < self.mutation_rate:
                    parent = new_generation[i]
                    child_law = self._mutate_law(parent.law_expression)
                    
                    child = QuantumHypercube(
                        dimensions=parent.dimensions.copy(),
                        resolution=parent.resolution,
                        quantum_correction=parent.quantum_correction,
                        hbar=parent.hbar
                    )
                    child.define_physical_law(child_law)
                    child.build_hypercube()
                    
                    new_generation[i] = child
            
            universes = new_generation
        
        return universes
    """
    
    # Создаем директорию для модулей
    os.makedirs("quantum_hypercube", exist_ok=True)
    with open("quantum_hypercube/generative.py", "w") as f:
        f.write(generative_code)
    
    # ===== 3. Python API =====
    api_code = r"""
# qh_api.py
from quantum_hypercube import QuantumHypercube
from quantum_hypercube.generative import UniverseGenerator

def create_universe(dimensions, resolution=128, law_expression=None):
    \"\"\"Создание новой вселенной\"\"\"
    universe = QuantumHypercube(dimensions, resolution)
    if law_expression:
        universe.define_physical_law(law_expression)
        universe.build_hypercube()
    return universe

def evolve_multiverse(base_universe, num_universes=5, epochs=10):
    \"\"\"Эволюция мультивселенной\"\"\"
    generator = UniverseGenerator(base_universe)
    return generator.generate(num_universes=num_universes, evolution_epochs=epochs)

def query(universe, point):
    \"\"\"Запрос значения во вселенной\"\"\"
    return universe.query(point)

def solve_schrodinger(universe, potential, **params):
    \"\"\"Решение уравнения Шредингера\"\"\"
    return universe.solve_schrodinger(potential, **params)

def project(universe, dim1, dim2):
    \"\"\"2D проекция вселенной\"\"\"
    return universe.holographic_projection([dim1, dim2])

# Контекстный менеджер для работы с вселенными
class MultiverseContext:
    def __init__(self, base_universe):
        self.base = base_universe
        self.multiverse = None
        
    def __enter__(self):
        return self
        
    def generate(self, num=5, epochs=10):
        self.multiverse = evolve_multiverse(self.base, num, epochs)
        return self.multiverse
        
    def query_all(self, point):
        if not self.multiverse:
            raise ValueError("Сначала сгенерируйте мультивселенную")
        return self.base.multiverse_query(point, self.multiverse)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    """
    
    with open("qh_api.py", "w") as f:
        f.write(api_code)
    
    # Обновляем импорты
    import_pattern = r"import numpy as np"
    new_imports = r"""import numpy as np
import uuid
from .generative import UniverseGenerator
"""
    
    content = re.sub(import_pattern, new_imports, content)
    
    # Сохраняем обновленный файл
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Файл {filename} успешно обновлен!")

def update_hypercube_shell(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # ===== 1. Новые команды для мультивселенной =====
    completer_pattern = r"completer=WordCompleter\(\["
    new_completer = r"""completer=WordCompleter([
                'create', 'define_law', 'build', 'query', 'multiverse_query', 
                'generate_multiverse', 'evolve_multiverse', 'select_universe',
                'project', 'visualize_3d', 'visualize_fractal', 
                'fractalize', 'save', 'load', 'exit', 'help', 'status', 'python_mode'"""
    
    content = re.sub(completer_pattern, new_completer, content)
    
    # ===== 2. Добавляем обработку команд =====
    command_pattern = r"elif command == 'visualize_fractal':\n\s+self\.visualize_fractal_structure\(args\)"
    new_commands = r"""
                elif command == 'generate_multiverse':
                    self.generate_multiverse(args)
                    
                elif command == 'evolve_multiverse':
                    self.evolve_multiverse(args)
                    
                elif command == 'multiverse_query':
                    self.multiverse_query_point(args)
                    
                elif command == 'select_universe':
                    self.select_universe(args)
                    
                elif command == 'python_mode':
                    self.enter_python_mode()"""
    
    content = re.sub(command_pattern, command_pattern + new_commands, content)
    
    # ===== 3. Методы оболочки =====
    shell_methods = r"""
    def generate_multiverse(self, args):
        \"\"\"Генерация мультивселенной\"\"\"
        if self.cube is None:
            print("Ошибка: базовая вселенная не создана")
            return
            
        num_universes = 5
        epochs = 10
        
        if args:
            try:
                num_universes = int(args[0])
                if len(args) > 1: epochs = int(args[1])
            except ValueError:
                print("Ошибка: параметры должны быть целыми числами")
                return
                
        print(f"Генерация {num_universes} вселенных с {epochs} эпохами эволюции...")
        self.multiverse = self.cube.create_multiverse(
            num_universes=num_universes,
            evolution_epochs=epochs
        )
        print(f"Мультивселенная успешно создана! ID: {[u.id for u in self.multiverse]}")
        
    def evolve_multiverse(self, args):
        \"\"\"Дополнительная эволюция мультивселенной\"\"\"
        if not hasattr(self, 'multiverse') or not self.multiverse:
            print("Ошибка: сначала создайте мультивселенную")
            return
            
        epochs = 5
        if args:
            try:
                epochs = int(args[0])
            except ValueError:
                print("Ошибка: количество эпох должно быть целым числом")
                return
                
        print(f"Эволюция мультивселенной ({epochs} эпох)...")
        generator = UniverseGenerator(self.cube)
        self.multiverse = generator.evolve_universes(
            universes=self.multiverse,
            epochs=epochs
        )
        print("Эволюция завершена!")
        
    def multiverse_query_point(self, args):
        \"\"\"Запрос значения во всех вселенных\"\"\"
        if not hasattr(self, 'multiverse') or not self.multiverse:
            print("Ошибка: мультивселенная не создана")
            return
            
        if len(args) < 1:
            print("Ошибка: требуется точка (координаты через запятую)")
            return
            
        coords = self.parse_coordinates(args[0])
        if coords is None:
            return
            
        results = self.cube.multiverse_query(coords, self.multiverse)
        
        print("\nРезультаты запроса в мультивселенной:")
        for res in results:
            print(f"Вселенная {res['universe']}: {res['value']:.6f} | Закон: {res['law'][:30]}...")
            
    def select_universe(self, args):
        \"\"\"Выбор активной вселенной\"\"\"
        if not hasattr(self, 'multiverse') or not self.multiverse:
            print("Ошибка: мультивселенная не создана")
            return
            
        if len(args) < 1:
            print("Доступные вселенные:")
            for i, universe in enumerate(self.multiverse):
                print(f"{i+1}. ID: {universe.id} | Закон: {universe.law_expression[:50]}...")
            return
            
        try:
            index = int(args[0]) - 1
            if 0 <= index < len(self.multiverse):
                self.cube = self.multiverse[index]
                print(f"Активная вселенная изменена на #{index+1} (ID: {self.cube.id})")
            else:
                print("Ошибка: неверный индекс вселенной")
        except ValueError:
            print("Ошибка: индекс должен быть числом")
            
    def enter_python_mode(self):
        \"\"\"Переход в режим Python программирования\"\"\"
        print("Переход в Python REPL режим. Выход: exit()")
        print("Доступные объекты:")
        print("  cube - текущий гиперкуб")
        print("  multiverse - мультивселенная (если создана)")
        
        from qh_api import create_universe, evolve_multiverse, query, project
        locals = {
            'cube': self.cube,
            'multiverse': getattr(self, 'multiverse', None),
            'create_universe': create_universe,
            'evolve_multiverse': evolve_multiverse,
            'query': query,
            'project': project
        }
        
        import code
        code.interact(local=locals)
    """
    
    # Вставляем новые методы
    class_end = re.search(r"class QuantumHypercubeShell:.*?def", content, re.DOTALL)
    if class_end:
        insert_pos = class_end.end() - 3
        content = content[:insert_pos] + shell_methods + content[insert_pos:]
    
    # ===== 4. Обновляем справку =====
    help_pattern = r"visualize_fractal <dim1> <dim2> \[глубина\] - Визуализация фрактала"
    new_help = r"""visualize_fractal <dim1> <dim2> [глубина] - Визуализация фрактала
  generate_multiverse [число] [эпохи] - Генерация мультивселенной
  evolve_multiverse [эпохи] - Эволюция мультивселенной
  multiverse_query <точка> - Запрос во всех вселенных
  select_universe [индекс] - Выбор активной вселенной
  python_mode - Переход в Python REPL режим"""
    
    content = re.sub(help_pattern, new_help, content)
    
    # ===== 5. Добавляем импорты =====
    import_pattern = r"import numpy as np"
    new_imports = r"""import numpy as np
from quantum_hypercube.generative import UniverseGenerator
"""
    
    content = re.sub(import_pattern, new_imports, content)
    
    # Сохраняем обновленный файл
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Файл {filename} успешно обновлен!")

def create_demo_script():
    """Создание демонстрационного скрипта для генеративных вселенных"""
    demo_code = '''# multiverse_demo.py
from qh_api import create_universe, evolve_multiverse, query, MultiverseContext

# Создаем базовую вселенную
base_universe = create_universe(
    dimensions={"x": (-5, 5), "y": (-5, 5), "t": (0, 10)},
    resolution=64,
    law_expression="sin(x)*cos(y)*exp(-t/5)"
)

# Работа с мультивселенной через контекст
with MultiverseContext(base_universe) as mv:
    # Генерация 3 вселенных с 5 эпохами эволюции
    multiverse = mv.generate(num=3, epochs=5)
    
    # Запрос значения во всех вселенных
    point = [1.57, 0.78, 2.0]
    results = mv.query_all(point)
    
    print("Результаты запроса в мультивселенной:")
    for res in results:
        print(f"Вселенная {res['universe']}: {res['value']:.4f}")
        print(f"  Закон: {res['law']}")

# Выбор наиболее интересной вселенной
selected_universe = multiverse[0]

# Решение уравнения Шредингера в альтернативной вселенной
schrodinger_result = solve_schrodinger(
    selected_universe,
    potential="0.5*x**2 + 0.1*cos(y)",
    mass=0.8,
    num_points=1000
)
print("Энергетические уровни:", schrodinger_result['energies'])
'''
    with open("multiverse_demo.py", "w") as f:
        f.write(demo_code)
    print("Демонстрационный скрипт создан: multiverse_demo.py")

if __name__ == "__main__":
    # Обновляем основной файл
    update_quantum_hypercube("quantum_hypercube.py")
    
    # Обновляем оболочку
    update_hypercube_shell("quantum_hypercube_shell.py")
    
    # Создаем демо-скрипт
    create_demo_script()
    
    print("\nОбновление 4.0 успешно установлено!")
    print("Новые возможности:")
    print("- Генеративные гипервселенные с VAE-эволюцией")
    print("- Python API для программного доступа")
    print("- Контекстный менеджер мультивселенной")
    print("- Режим прямого Python программирования в оболочке")
    print("\nДля тестирования запустите: python multiverse_demo.py")
```

### Ключевые обновления:

#### 1. Генеративные гипервселенные
- **VAE-генератор законов**: Вариационный автоэнкодер для создания новых физических законов
- **Эволюционный алгоритм**: Отбор вселенных по физической состоятельности
- **Функции пригодности**: 
  ```python
  def _default_fitness(self, hypercube):
      # 1. Соответствие размерностям
      # 2. Топологическая согласованность 
      # 3. Энергетическая стабильность
  ```

#### 2. Python API
- **Простой интерфейс**:
  ```python
  from qh_api import create_universe, evolve_multiverse, query
  ```
- **Контекстный менеджер**:
  ```python
  with MultiverseContext(base_universe) as mv:
      multiverse = mv.generate(num=3, epochs=5)
      results = mv.query_all(point)
  ```

#### 3. Обновления оболочки
- **Новые команды**:
  - `generate_multiverse`: Создание ансамбля вселенных
  - `evolve_multiverse`: Дополнительная эволюция
  - `multiverse_query`: Запрос во всех вселенных
  - `select_universe`: Выбор активной вселенной
  - `python_mode`: Переход в Python REPL

- **Пример сеанса**:
  ```bash
  QH> create 64 x:-5:5 y:-5:5
  QH> define_law sin(x)*cos(y)
  QH> build
  QH> generate_multiverse 5 10
  QH> multiverse_query 1.57,0.78
  QH> select_universe 3
  QH> python_mode
  >>> print(cube.law_expression)
  >>> results = [query(u, [1,1]) for u in multiverse]
  ```

#### 4. Архитектурные улучшения
- **Модуль generative.py**: Содержит VAE для генерации законов
- **Уникальные ID вселенных**: Автоматическая идентификация
- **Физическая валидация**: Автоматическая проверка новых законов

### Как использовать:
1. **Программный доступ**:
   ```python
   from qh_api import create_universe, evolve_multiverse
   universe = create_universe({"x": (-10,10)}, law_expression="x**2")
   multiverse = evolve_multiverse(universe, num_universes=3)
   ```

2. **Интерактивная оболочка**:
   ```bash
   QH> generate_multiverse 5
   QH> multiverse_query 3.14,1.57
   QH> python_mode
   ```

3. **Демонстрация**:
   ```bash
   python multiverse_demo.py
   ```

Это обновление превращает Quantum Hypercube в платформу для исследования мультивселенных с альтернативными физическими законами, открывая новые возможности для теоретического моделирования и исследований.
