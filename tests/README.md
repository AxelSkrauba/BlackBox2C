# Tests - BlackBox2C

## Estructura de Tests

```
tests/
├── __init__.py              # Inicialización del paquete de tests
├── test_analysis.py         # Tests de análisis de features (19 tests)
├── test_codegen.py          # Tests de generación de código (10 tests)
├── test_config.py           # Tests de configuración (7 tests)
├── test_converter.py        # Tests de integración (14 tests)
├── test_exporters.py        # Tests de exportadores multi-formato (40 tests)
├── test_optimizer.py        # Tests de optimización (9 tests)
├── test_regression.py       # Tests de regresión (16 tests)
├── test_reproducibility.py  # Tests de reproducibilidad (6 tests)
├── test_surrogate.py        # Tests de extracción de surrogate (7 tests)
└── README.md               # Este archivo
```

## Estadísticas

- **Total de tests**: 128
- **Estado**: 100% pasando ✅
- **Cobertura**: ~90%
- **Tiempo de ejecución**: ~1.5s

## Ejecutar Tests

### Instalar pytest

```bash
pip install pytest pytest-cov
```

### Ejecutar todos los tests

```bash
# Desde el directorio raíz del proyecto
pytest tests/

# Con verbose
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=blackbox2c --cov-report=html
```

### Ejecutar tests específicos

```bash
# Un archivo específico
pytest tests/test_config.py

# Una clase específica
pytest tests/test_config.py::TestConversionConfig

# Un test específico
pytest tests/test_config.py::TestConversionConfig::test_default_config
```

## Cobertura de Tests

### Módulos Cubiertos

- **test_config.py**: Configuración y validación de parámetros
  - Valores por defecto
  - Valores personalizados
  - Validación de parámetros inválidos
  - Ajuste automático por presupuesto de memoria

- **test_surrogate.py**: Extracción de modelos surrogate
  - Inicialización
  - Extracción desde Random Forest
  - Extracción desde SVM
  - Cálculo de fidelidad
  - Generación de muestras de frontera

- **test_optimizer.py**: Optimización de reglas
  - Niveles de optimización (low, medium, high)
  - Poda de ramas redundantes
  - Fusión de hojas similares
  - Análisis de complejidad
  - Importancia de features

- **test_codegen.py**: Generación de código C
  - Generación básica
  - Punto flotante vs punto fijo
  - Diferentes precisiones (8, 16, 32 bits)
  - Estimación de tamaño
  - Nombres personalizados

- **test_converter.py**: Integración end-to-end
  - Conversión de diferentes modelos
  - Validación de entrada
  - Manejo de errores
  - Recolección de métricas

- **test_reproducibility.py**: Tests de reproducibilidad (software regression)
  - Fidelidad consistente
  - Tamaño de código razonable
  - Reproducibilidad con mismo random_state
  - Diferentes configuraciones

## Métricas de Éxito

### Criterios de Aceptación

✅ **Todos los tests pasan**: 100% de tests exitosos  
✅ **Cobertura >80%**: Al menos 80% del código cubierto  
✅ **Sin warnings**: No hay warnings de pytest  
✅ **Tiempo razonable**: Suite completa <30 segundos  

### Resultados Esperados

```
tests/test_config.py ............          [ 15%]
tests/test_surrogate.py .........          [ 30%]
tests/test_optimizer.py ..........         [ 45%]
tests/test_codegen.py ............         [ 60%]
tests/test_converter.py ..............     [ 80%]
tests/test_regression.py .......           [100%]

============ 60 passed in 15.23s ============
```

## Agregar Nuevos Tests

### Template para nuevo test

```python
"""
Tests for [module_name].
"""

import pytest
import numpy as np
from blackbox2c import [imports]


class Test[ClassName]:
    """Test [ClassName] class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Setup code
        return data
    
    def test_[feature_name](self, sample_data):
        """Test [specific feature]."""
        # Arrange
        # Act
        # Assert
        assert condition
```

### Mejores Prácticas

1. **Nombres descriptivos**: `test_convert_random_forest_with_high_fidelity`
2. **Un concepto por test**: Cada test valida una cosa específica
3. **Fixtures para setup**: Reutilizar código de preparación
4. **Assertions claras**: Mensajes de error informativos
5. **Independencia**: Tests no dependen entre sí

## Tests Pendientes (Futuro)

### Fase 2: Tests Avanzados

- [ ] Tests de performance/benchmarking
- [ ] Tests de integración con hardware real
- [ ] Tests de compilación de código C generado
- [ ] Tests de precisión numérica (punto fijo vs flotante)
- [ ] Tests de casos extremos (edge cases)

### Fase 3: Tests de Nuevos Features

- [ ] Tests de regresión (cuando se implemente)
- [ ] Tests de exportación a múltiples formatos
- [ ] Tests de cuantización avanzada
- [ ] Tests de optimizaciones específicas por arquitectura

## Debugging Tests

### Test falla

```bash
# Ejecutar con más detalle
pytest tests/test_config.py::test_invalid_max_depth -vv

# Detener en el primer fallo
pytest tests/ -x

# Mostrar print statements
pytest tests/ -s
```

### Ver cobertura

```bash
# Generar reporte HTML
pytest tests/ --cov=blackbox2c --cov-report=html

# Abrir en navegador
# htmlcov/index.html
```

## CI/CD Integration

### GitHub Actions (futuro)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=blackbox2c
```

## Mantenimiento

### Actualizar tests cuando

- Se agrega un nuevo feature
- Se encuentra un bug (agregar test de regresión)
- Se modifica la API pública
- Se optimiza código existente

### Revisar tests cuando

- Tests fallan después de cambios
- Cobertura disminuye
- Tests se vuelven lentos (>1 minuto)

---

**Última actualización**: 2025-10-09  
**Cobertura actual**: ~85% (estimado)  
**Tests totales**: 60+
