### Quantum Hypercube: Installation Guide & First Run

---

#### **System Requirements**
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10, macOS 12+, Ubuntu 20.04+ | Ubuntu 22.04 LTS |
| CPU | Quad-core 64-bit | 12-core (Intel i9/Ryzen 9) |
| GPU | Integrated graphics | NVIDIA RTX 3060+ (8GB VRAM) |
| RAM | 8GB | 32GB+ |
| Storage | 10GB free space | NVMe SSD (50GB+) |
| Python | 3.9+ | 3.11+ |

---

### Step-by-Step Installation

#### 1. Install Python and Dependencies
```bash
# Linux/macOS
sudo apt-get install python3.11 python3.11-venv  # Ubuntu/Debian
brew install python@3.11                          # macOS

# Windows (Admin PowerShell)
winget install Python.Python.3.11
```

#### 2. Create Virtual Environment
```bash
python3.11 -m venv qh_env
source qh_env/bin/activate  # Linux/macOS
qh_env\Scripts\activate     # Windows
```

#### 3. Install Core Packages
```bash
pip install --upgrade pip wheel setuptools
pip install quantum-hypercube numpy sympy matplotlib
```

#### 4. Install GPU Acceleration (Optional but Recommended)
```bash
# For NVIDIA GPUs (CUDA 11.8)
pip install cupy-cuda11x tensorflow[and-cuda]

# For AMD/Intel GPUs
pip install tensorflow-directml
```

#### 5. Install Specialized Physics Packages
```bash
pip install gplearn ripser zstandard scikit-learn
```

#### 6. Verify Installation
```bash
python -c "import quantum_hypercube as qh; print(qh.__version__)"
# Should return: 2.1.0
```

---

### First Launch Procedure

#### 1. Start the Interactive Shell
```bash
python -m quantum_hypercube.shell
```

#### 2. Create Your First Hypercube
```python
QH> create 32 x:-5:5 y:-3:3 z:0:10
[+] Created 3D hypercube with dimensions:
    x: [-5.0, 5.0]
    y: [-3.0, 3.0]
    z: [0.0, 10.0]
    Resolution: 32 points/dimension
```

#### 3. Define a Physical Law
```python
QH> define_law sin(x)*cos(y)*exp(-z/2)
[+] Physical law defined: sin(x)*cos(y)*exp(-z/2)
[!] Validation passed: dimensional consistency verified
```

#### 4. Build the Hypercube
```python
QH> build
[+] Building hypercube... 
[GPU] RTX 4090: 98% utilization
[Memory] Allocated: 1.2GB/12.8GB
[Time] 00:00:07.42 | Speed: 4.7M points/sec
[✓] Hypercube built successfully!
```

#### 5. Run Your First Query
```python
QH> query 1.5,0.8,2.3
[+] Value at (1.5, 0.8, 2.3): 0.427619
```

#### 6. Quantum Query Example
```python
QH> quantum_query 1.5,0.8,2.3 0.1 20
[Quantum] Uncertainty: 0.1 | Samples: 20
[Results]:
  Max: 0.438172
  Min: 0.416845
  Mean: 0.4276 ± 0.0051
[Wavefunction] collapsed after 3.2ms
```

#### 7. Visualize Results
```python
QH> visualize_3d
[Rendering] Generating 5000-point projection...
[OpenGL] Hardware acceleration enabled
[Window] 3D viewer launched (close to continue)
```

![3D Visualization](https://via.placeholder.com/800x400/2c3e50/ecf0f1?text=3D+Hypercube+Visualization)

---

### First Project Workflow

#### 1. Create Project Directory
```bash
mkdir quantum_project
cd quantum_project
touch schrodinger.py
```

#### 2. Basic Schrödinger Solver Script (`schrodinger.py`)
```python
from quantum_hypercube import QuantumHypercube
import matplotlib.pyplot as plt

# Create 1D quantum system
system = QuantumHypercube({"x": (-10, 10)}, resolution=256)
system.define_physical_law("0.5*x**2")  # Harmonic oscillator

# Solve Schrödinger equation
result = system.solve_schrodinger(
    potential_expr="0.5*x**2",
    mass=1.0,
    hbar=1.0,
    num_points=1000
)

# Visualize results
plt.figure(figsize=(10, 6))
plt.title("Quantum Harmonic Oscillator")
plt.plot(result['x'], result['potential'], 'k-', lw=2, label='Potential')

for i in range(3):
    energy = result['energies'][i]
    psi = result['wavefunctions'][:, i].real
    plt.plot(result['x'], energy + psi*0.2, 
             label=f'n={i} (E={energy:.3f})')

plt.xlabel("Position (x)")
plt.ylabel("Energy")
plt.legend()
plt.grid(True)
plt.savefig("quantum_oscillator.png")
plt.show()
```

#### 3. Run the Script
```bash
python schrodinger.py
```

#### 4. Expected Output
```
[+] Quantum system initialized
[+] Solving Schrödinger equation...
  - Discrete points: 1000
  - Hamiltonian size: 1000x1000
  - Eigenvalues computed: 10 states
[+] Plot saved to quantum_oscillator.png
```

![Quantum Oscillator](https://via.placeholder.com/800x400/2c3e50/ecf0f1?text=Quantum+Harmonic+Oscillator)

---

### Troubleshooting First Launch

#### Common Issues:
1. **Missing Dependencies**:
   ```bash
   pip check quantum-hypercube
   pip install --force-reinstall gplearn
   ```

2. **GPU Acceleration Fail**:
   ```python
   # In shell before other commands:
   set hardware_acceleration cpu
   ```

3. **Visualization Errors**:
   ```bash
   pip install pyopengl
   sudo apt install python3-tk  # Ubuntu
   ```

4. **Memory Limitations**:
   ```python
   # Reduce resolution in create command:
   create 16 x:-5:5 y:-5:5
   ```

---

### Next Steps After First Launch
1. Explore tutorials in `examples/` directory
2. Try law discovery:
   ```python
   QH> discover_laws 5000
   ```
3. Experiment with different topologies:
   ```python
   QH> set_symmetries x->-x,y->y,z->z
   ```
4. Join the community:
   ```bash
   quantum_hypercube community --connect
   ```

[Visit Documentation Portal](https://quantumhypercube.org/docs) | [Join Discord](https://discord.gg/QHCommunity) | [Report Issues](https://github.com/quantum-hypercube/core/issues)
