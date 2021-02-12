# TensorFlow 2.0
<br/>

#### Installing TensorFlow:

To install the CPU version of TensorFlow:

```
pip install tensorflow
```

To install the CUDA enables GPU version of TensorFlow:

```
pip install tensorflow-gpu
```

#### Importing TensorFlow:

```python
%tensorflow_version 2.x # Only required while running on a Ipython notebook.
import tensorflow as tf
print(tf.version) # Checking the version of TensorFlow.
```

#### Tensors

<div style="text-align: justify">
Tensors are a multi-dimensional arrays with a uniform type (called a **dtype**). A tensor is a generalization of vectors and matrices to potentially higher dimensions. Tensors are a fundamental aspect of TensorFlow. They are the main objects that are passed around manipulated throughout the program. Tensorflow programs work by building a graph of Tensor objects that details how tensors are related. Each tensor has a data-type and a shape associated to it. Just like vectors and matrices, tensors can also have operations like addition, subtraction, dot-product, cross-product and others.
</div>

#### Creating Tensors using TensorFlow:

To create a tensor, you defined the value of the tensor and the data-type.
```python
string = tf.Variable("this is a string", tf.string) 
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)
```

<br/>

#### Data-types for Tensors Supported by TensorFlow:

<ul>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#float16"><code translate="no" dir="ltr">tf.float16</code></a>: 16-bit half-precision floating-point.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#float32"><code translate="no" dir="ltr">tf.float32</code></a>: 32-bit single-precision floating-point.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#float64"><code translate="no" dir="ltr">tf.float64</code></a>: 64-bit double-precision floating-point.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#bfloat16"><code translate="no" dir="ltr">tf.bfloat16</code></a>: 16-bit truncated floating-point.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#complex64"><code translate="no" dir="ltr">tf.complex64</code></a>: 64-bit single-precision complex.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#complex128"><code translate="no" dir="ltr">tf.complex128</code></a>: 128-bit double-precision complex.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#int8"><code translate="no" dir="ltr">tf.int8</code></a>: 8-bit signed integer.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#uint8"><code translate="no" dir="ltr">tf.uint8</code></a>: 8-bit unsigned integer.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#uint16"><code translate="no" dir="ltr">tf.uint16</code></a>: 16-bit unsigned integer.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#uint32"><code translate="no" dir="ltr">tf.uint32</code></a>: 32-bit unsigned integer.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#uint64"><code translate="no" dir="ltr">tf.uint64</code></a>: 64-bit unsigned integer.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#int16"><code translate="no" dir="ltr">tf.int16</code></a>: 16-bit signed integer.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#int32"><code translate="no" dir="ltr">tf.int32</code></a>: 32-bit signed integer.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#int64"><code translate="no" dir="ltr">tf.int64</code></a>: 64-bit signed integer.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#bool"><code translate="no" dir="ltr">tf.bool</code></a>: Boolean.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#string"><code translate="no" dir="ltr">tf.string</code></a>: String.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#qint8"><code translate="no" dir="ltr">tf.qint8</code></a>: Quantized 8-bit signed integer.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#quint8"><code translate="no" dir="ltr">tf.quint8</code></a>: Quantized 8-bit unsigned integer.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#qint16"><code translate="no" dir="ltr">tf.qint16</code></a>: Quantized 16-bit signed integer.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#quint16"><code translate="no" dir="ltr">tf.quint16</code></a>: Quantized 16-bit unsigned integer.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#qint32"><code translate="no" dir="ltr">tf.qint32</code></a>: Quantized 32-bit signed integer.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#resource"><code translate="no" dir="ltr">tf.resource</code></a>: Handle to a mutable resource.</li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf#variant"><code translate="no" dir="ltr">tf.variant</code></a>: Values of arbitrary types.</li>
</ul>

#### Rank / Degree of Tensors:
<div style="text-align: justify">
Rank or the degree of tensors is the number of dimensions involved in the tensor. The above created tensor is of rank 0, which is also known as a scalar. Defining a tensor of higher dimensions.

```python
rank1_tensor = tf.Variable(["Test"], tf.string) 
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)
print(tf.rank(rank2_tensor)) # Checking the rank of a tensor.
```
The rank of a tensor is directly related to the level of nested lists. The rank of the variable rank1_tensor is 1 as the deepest level of nesting is 1 and rank2_tensor is 2 as the deepest level of nesting 2.

</div>

#### Shape of a Tensor:
<div style="text-align: justify">
Shape of the tensor is the number of elements that exist in each dimenstion of the tensor.

```python
rank2_tensor.shape
```
	

</div>