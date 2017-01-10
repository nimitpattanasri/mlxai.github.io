---
layout: post
title: A Modular Approach to Implementing Fully-Connected Neural Networks
---

I write this post to clarify non-trivial issues in implementing forward and backward layers of fully-connected neural networks. The code is short and seems intuitive. However, I would like to elaborate on finding partial derivative w.r.t. the bias, that is, clarifying the expression `db = np.sum(dout, axis=0)` for the uninitiated.

To become a backpropagation ninja, I recommend succinct study materials from [Stanford CS231n](http://vision.stanford.edu/teaching/cs231n/syllabus.html) [here](http://cs231n.github.io/optimization-2/) and [here](http://cs231n.github.io/neural-networks-case-study/).

The materials will give you intuitive understanding about how to compute gradient with a backpropagation technique. This technique uses a chain rule and enjoys considering only the output and its inputs of single neurons, avoiding analyzing gradient of a single monolithic function in one shot.

The exercise [FullyConnectedNets.ipynb](http://vision.stanford.edu/teaching/cs231n/winter1516_assignment2.zip) provided with the materials will introduce you to a modular layer design, and then use those layers to implement fully-connected networks of arbitrary depth.

> ... For each layer we will implement a `forward` and a `backward` function. The `forward` function will receive inputs, weights, and other parameters and will return both an output and a `cache` object storing data needed for the backward pass, like this:
{% highlight python %}
def layer_forward(x, w):
  """ Receive inputs x and weights w """
  # Do some computations ...
  z = # ... some intermediate value
  # Do some more computations ...
  out = # the output
   
  cache = (x, w, z, out) # Values we need to compute gradients
   
  return out, cache
{% endhighlight %}

> The backward pass will receive upstream derivatives and the `cache` object, and will return gradients with respect to the inputs and weights, like this:

{% highlight python %}
def layer_backward(dout, cache):
  """
  Receive derivative of loss with respect to outputs and cache,
  and compute derivative with respect to inputs.
  """
  # Unpack cache values
  x, w, z, out = cache
  
  # Use values in cache to compute derivatives
  dx = # Derivative of loss with respect to x
  dw = # Derivative of loss with respect to w
  
  return dx, dw
{% endhighlight %}

> After implementing a bunch of layers this way, we will be able to easily combine them to build classifiers with different architectures.

Below is a sample implementation of `layer_forward` and `layer_backward`.

{% highlight python %}
def layer_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.
  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.
  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  out = x.reshape(x.shape[0], w.shape[0]).dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache

def layer_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dx = dout.dot(w.T).reshape(x.shape)
  dw = x.reshape(x.shape[0], w.shape[0]).T.dot(dout)
  db = np.sum(dout, axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db
{% endhighlight %}

It is not difficult to derive gradients `dx` and `dw` and implement them in Python. The gradient `db`, however, requires careful thought. The expression `db = np.sum(dout, axis=0)` is not easy to understand. 

First, you need to understand what really happens inside this simple equation $$out = xw + b$$. One way to understand this is by analyzing dimensions of the matrices. That is, $$\underset{(N \times M)}{out} = \underset{(N \times D)}x\underset{(D \times M)}w + \underset{(1 \times M)}b$$.

Do you notice incompatibility of $$b$$'s dimension and $$out$$'s? Think of it for a second and read on! 

What happens in the expression is broadcast of $$b$$ from $$1 \times M$$ to $$N \times M$$. Intuitively, there is only one set of bias of size $$M$$ which shares across all examples $$x$$. In fact, there is an *invisible input* for the bias term. That is, $$\underset{(N \times M)}{out} = \underset{(N \times D)}x\underset{(D \times M)}w + \underset{(N \times 1)}{ix}\underset{(1 \times M)}b$$ where $$ix$$ contains all 1's and stays there just for the sake of facilitating later calculation. (As the term suggests, bias does not depend on the input.)

That's it for `layer_forward`. Now, let's proceed to `layer_backward`.

Considering $$\underset{(N \times M)}{out} = \underset{(N \times D)}x\underset{(D \times M)}w + \underset{(N \times 1)}{ix}\underset{(1 \times M)}b$$, we can derive $$db$$ (in short, for $$\frac{dLoss}{db}$$) using a chain rule with $$\underset{(1 \times M)}{db} = \underset{(1 \times N)}{ix^\mathsf{T}} \underset{(N \times M)}{dout}$$. Recall that $$ix$$ is all 1's; therefore, the expression just sums $$dout$$ along axis = 0, that is, `db = np.sum(dout, axis=0)`.

Not really trivial, is it?

###### Lessons Learned
* Understanding NumPy's broadcast is a must.
* Analyzing dimensions of matrices might come in handy.

###### Further Reading
* [Array Broadcasting in NumPy](http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc)
* [http://cs231n.github.io/linear-classify/](http://cs231n.github.io/linear-classify/)
* [http://cs231n.github.io/optimization-2/](http://cs231n.github.io/optimization-2/)
* [http://cs231n.github.io/neural-networks-case-study/](http://cs231n.github.io/neural-networks-case-study/)





<script type="text/javascript" async
  src="//cdn.mathjax.org/mathjax/latest/MathJax.js">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    extensions: ["tex2jax.js"],
    jax: ["input/TeX", "output/HTML-CSS"],
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
      processEscapes: true
    },
    "HTML-CSS": { availableFonts: ["TeX"] },
    menuSettings: { zoom: "Click" }
  });
</script>
