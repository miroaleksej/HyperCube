# quantum_horizon_explorer.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from quantum_hypercube import QuantumHypercube
import time

class QuantumHorizonExplorer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(bottom=0.4)
        
        # Инициализация гиперкуба нового поколения
        self.dimensions = {'x': (-5, 5), 'y': (-5, 5), 't': (0, 10)}
        self.cube = QuantumHypercube(self.dimensions, resolution=64)
        self.current_law = "sin(x)*cos(y)*exp(-t/5)"
        self.cube.define_physical_law(self.current_law)
        
        # Настройка интерфейса
        self.setup_controls()
        self.update_visualization()
        
        # Параметры исследования
        self.horizon_level = 1.0
        self.entanglement_factor = 0.7
        self.discovery_log = []
        
        plt.show()
    
    def setup_controls(self):
        """Создание интерактивных элементов управления"""
        # Слайдеры для параметров
        ax_slider = plt.axes([0.25, 0.25, 0.65, 0.03])
        self.slider = Slider(
            ax_slider, 'Горизонт', 0.1, 5.0, 
            valinit=self.horizon_level, valstep=0.1
        )
        self.slider.on_changed(self.update_horizon)
        
        # Кнопки открытий
        ax_quantum = plt.axes([0.25, 0.15, 0.2, 0.05])
        self.btn_quantum = Button(ax_quantum, 'Квантовый скачок')
        self.btn_quantum.on_clicked(self.quantum_leap)
        
        ax_fractal = plt.axes([0.45, 0.15, 0.2, 0.05])
        self.btn_fractal = Button(ax_fractal, 'Фрактальный переход')
        self.btn_fractal.on_clicked(self.fractal_transition)
        
        ax_multiverse = plt.axes([0.65, 0.15, 0.2, 0.05])
        self.btn_multiverse = Button(ax_multiverse, 'Мультивселенная')
        self.btn_multiverse.on_clicked(self.multiverse_jump)
        
        # Поле вывода логов
        self.log_text = self.fig.text(
            0.05, 0.05, 
            "Инициализация Квантового Исследователя...\n",
            fontsize=10, 
            verticalalignment='bottom'
        )
    
    def update_visualization(self, t=2.5):
        """Обновление научной визуализации"""
        self.ax.clear()
        
        # Генерация данных
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        
        # Вычисление значений с учетом времени
        values = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                values[i, j] = self.cube.query([X[i, j], Y[i, j], t])
        
        # Голографическая проекция
        im = self.ax.imshow(
            values, 
            extent=[-5, 5, -5, 5],
            origin='lower', 
            cmap='plasma',
            alpha=0.8 + 0.2 * np.sin(time.time())
        )
        
        # Квантовые эффекты
        quantum_effect = self.horizon_level * np.random.randn(*values.shape)
        contour = self.ax.contour(
            X, Y, values + quantum_effect, 
            levels=10, colors='white', linewidths=0.7
        )
        
        # Настройка графика
        self.ax.set_title(
            f"Квантовый Горизонт v4.1\nЗакон: {self.current_law}",
            fontsize=14,
            color='cyan'
        )
        self.ax.set_xlabel('Измерение X', fontsize=12)
        self.ax.set_ylabel('Измерение Y', fontsize=12)
        
        # Эффекты горизонта
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.fig.canvas.draw_idle()
    
    def update_horizon(self, val):
        """Обновление уровня горизонта исследований"""
        self.horizon_level = val
        self.add_log(f"Горизонт расширен до уровня {val:.1f}")
        self.update_visualization()
    
    def quantum_leap(self, event):
        """Квантовый скачок в новое состояние"""
        new_law = self.generate_quantum_law()
        self.cube.define_physical_law(new_law)
        self.current_law = new_law
        
        self.add_log(f"КВАНТОВЫЙ СКАЧОК!\nНовый закон: {new_law}")
        self.update_visualization()
    
    def fractal_transition(self, event):
        """Переход во фрактальную реальность"""
        self.entanglement_factor = min(1.0, self.entanglement_factor + 0.1)
        self.add_log(f"ФРАКТАЛЬНЫЙ ПЕРЕХОД\nУровень запутанности: {self.entanglement_factor:.1f}")
        
        # Спецэффект
        for _ in range(10):
            self.horizon_level += 0.1
            self.slider.set_val(self.horizon_level)
            plt.pause(0.05)
        
        self.update_visualization()
    
    def multiverse_jump(self, event):
        """Прыжок в параллельную вселенную"""
        self.add_log("АКТИВАЦИЯ МУЛЬТИВСЕЛЕННОЙ")
        
        # Эффект перехода
        for i in range(5):
            self.ax.set_facecolor(plt.cm.viridis(i/5))
            self.fig.canvas.draw_idle()
            plt.pause(0.2)
        
        # Создание новой реальности
        new_dims = {dim: (self.dimensions[dim][0] * np.random.uniform(0.8, 1.2), 
                     self.dimensions[dim][1] * np.random.uniform(0.8, 1.2)) 
                    for dim in self.dimensions}
        
        self.dimensions = new_dims
        self.cube = QuantumHypercube(self.dimensions, resolution=64)
        self.quantum_leap(event)
        
        self.add_log("ПЕРЕХОД В ПАРАЛЛЕЛЬНУЮ РЕАЛЬНОСТЬ ЗАВЕРШЁН!")
    
    def generate_quantum_law(self):
        """Генерация нового физического закона"""
        components = [
            'sin', 'cos', 'exp', 'log', 'sqrt',
            'x', 'y', 't', 'x*y', 'x**2', 'y**2', 't**2',
            'x+y', 'x-y', 'x*t', 'y*t', 'pi'
        ]
        
        law = ""
        for _ in range(3 + int(self.horizon_level)):
            component = np.random.choice(components)
            operation = np.random.choice(['*', '+', '-', '/'])
            
            if law and not law.endswith('('):
                law += operation
                
            if component in ['sin', 'cos', 'exp', 'log', 'sqrt']:
                arg = np.random.choice(['x', 'y', 't', f'({np.random.choice(components)})'])
                law += f"{component}({arg})"
            else:
                law += component
        
        # Упрощение выражения
        if law.startswith('*') or law.startswith('/'):
            law = law[1:]
        
        return law[:80]  # Ограничение длины
    
    def add_log(self, message):
        """Добавление записи в журнал открытий"""
        timestamp = time.strftime("%H:%M:%S")
        self.discovery_log.append(f"[{timestamp}] {message}")
        
        # Отображаем последние 4 сообщения
        log_text = "\n".join(self.discovery_log[-4:])
        self.log_text.set_text(log_text)

if __name__ == "__main__":
    print("Активация Квантового Исследователя...")
    print("Готов к покорению новых горизонтов! 🌌")
    explorer = QuantumHorizonExplorer()
