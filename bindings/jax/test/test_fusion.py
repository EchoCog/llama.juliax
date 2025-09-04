"""
Tests for JAX Fusion Integration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

# Import fusion components
try:
    from . import (
        Arrayℝ, TensorΑ, FunctionΦ, 
        tensor_product, direct_sum, hadamard_product, 
        gradient, partial_derivative, summation, product_reduction,
        array, tensor, function,
        jit_compile, vectorize, parallelize,
        ProgressiveSyntax
    )
except ImportError:
    # If running directly
    from __init__ import (
        Arrayℝ, TensorΑ, FunctionΦ,
        tensor_product, direct_sum, hadamard_product,
        gradient, partial_derivative, summation, product_reduction,  
        array, tensor, function,
        jit_compile, vectorize, parallelize,
        ProgressiveSyntax
    )

def test_array_creation():
    """Test mathematical array creation."""
    A = array([1.0, 2.0, 3.0])
    assert isinstance(A, Arrayℝ)
    assert A.shape == (3,)
    assert np.allclose(A.data, [1.0, 2.0, 3.0])

def test_array_operations():
    """Test array mathematical operations."""
    A = array([1.0, 2.0, 3.0])
    B = array([4.0, 5.0, 6.0])
    
    # Test addition
    C = A + B
    assert isinstance(C, Arrayℝ)
    assert np.allclose(C.data, [5.0, 7.0, 9.0])
    assert C.properties['operation'] == 'addition'
    
    # Test multiplication  
    D = A * B
    assert isinstance(D, Arrayℝ)
    assert np.allclose(D.data, [4.0, 10.0, 18.0])

def test_tensor_creation():
    """Test tensor creation with automatic differentiation."""
    T = tensor([1.0, 2.0, 3.0], requires_grad=True)
    assert isinstance(T, TensorΑ)
    assert T.requires_grad == True
    assert np.allclose(T.data, [1.0, 2.0, 3.0])

def test_function_creation():
    """Test mathematical function creation."""
    f = function(lambda x: x**2)
    assert isinstance(f, FunctionΦ)
    assert f(3.0) == 9.0

def test_mathematical_operators():
    """Test Unicode mathematical operators."""
    A = array([[1.0, 2.0], [3.0, 4.0]])
    B = array([[5.0, 6.0], [7.0, 8.0]])
    
    # Test Hadamard product
    C = hadamard_product(A, B)
    expected = np.array([[5.0, 12.0], [21.0, 32.0]])
    assert np.allclose(C.data, expected)

def test_reduction_operations():
    """Test sum and product operations."""  
    A = array([[1.0, 2.0], [3.0, 4.0]])
    
    # Test sum
    sum_all = summation(A)
    assert np.allclose(sum_all.data, 10.0)
    
    sum_axis0 = summation(A, axis=0)
    assert np.allclose(sum_axis0.data, [4.0, 6.0])
    
    # Test product
    prod_all = product_reduction(A)
    assert np.allclose(prod_all.data, 24.0)

def test_automatic_differentiation():
    """Test gradient operator."""
    f = function(lambda x: x**2 + 2*x + 1)
    grad_f = gradient(f)
    
    assert isinstance(grad_f, FunctionΦ)
    assert 'gradient' in grad_f.transformations
    assert grad_f.properties['differentiated'] == True

def test_function_composition():
    """Test mathematical function composition."""
    f = function(lambda x: x**2)
    g = function(lambda x: x + 1)
    
    composed = f.compose(g)
    assert isinstance(composed, FunctionΦ)
    # (x + 1)² for x = 2 should be 9
    assert composed(2) == 9  # f(g(2)) = f(3) = 9

def test_transformations():
    """Test JIT compilation and vectorization."""
    f = function(lambda x: x**2)
    
    # Test JIT compilation
    jit_f = jit_compile(f)
    assert isinstance(jit_f, FunctionΦ)
    assert 'jit' in jit_f.transformations
    assert jit_f.properties['compiled'] == True
    
    # Test vectorization
    vec_f = vectorize(f)
    assert isinstance(vec_f, FunctionΦ)
    assert 'vectorize' in vec_f.transformations
    assert vec_f.properties['vectorized'] == True

def test_progressive_syntax():
    """Test progressive syntax help system."""
    A = array([1.0, 2.0, 3.0])
    
    # Test that help functions don't error
    ProgressiveSyntax.help(A, 'beginner')
    ProgressiveSyntax.help(A, 'intermediate') 
    ProgressiveSyntax.help(A, 'expert')

def test_transformation_composer():
    """Test transformation composition and optimization."""
    from __init__ import TransformationComposer
    
    composer = TransformationComposer()
    
    # Test identity optimization
    transformations = ['gradient', 'integrate']
    optimized = composer.optimize(transformations)
    assert len(optimized) == 0  # Should cancel out
    
    # Test Laplacian optimization
    transformations = ['gradient', 'gradient']
    optimized = composer.optimize(transformations)
    assert optimized == ['laplacian']

if __name__ == '__main__':
    # Run tests directly
    test_array_creation()
    test_array_operations()
    test_tensor_creation()
    test_function_creation()
    test_mathematical_operators()
    test_reduction_operations()
    test_automatic_differentiation()
    test_function_composition()
    test_transformations()
    test_progressive_syntax()
    test_transformation_composer()
    
    print("All JAX fusion tests passed!")