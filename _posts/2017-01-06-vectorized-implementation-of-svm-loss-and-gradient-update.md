---
layout: post
title: Vectorized Implementation of SVM Loss and Gradient Update
---

SVM multiclass classification computes scores, based on learnable weights, for each class and predicts one with the maximum score. Gradient descent is a common technique used to find optimal weights.

###### Loss function

Quality of weights is often expressed by a loss function, our unhappiness with classification result, and we want its value to be as small as possible. To minimize the loss, we have to define a loss function and find their partial derivatives with respect to the weights to update them iteratively.

SVM loss (a.k.a. hinge loss) function can be defined as:

$$
\begin{equation}
L_i = \sum_{j\neq y_i} \left[ \max(0, x_iw_j - x_iw_{y_i} + \Delta) \right] \tag{1}
\end{equation}
$$

where

* $$i$$ iterates over all N examples,
* $$j$$ iterates over all C classes,
* $$L_i$$ is loss for classifying a single example $$x_i$$ (row vector), 
* $$w_j$$ is the weights (column vector) for computing the score of class $$j$$, 
* $$y_i$$ is the index of the correct class of $$x_i$$, and
* $$\Delta$$ is a margin parameter

Intuitively, SVM wants score, $$x_iw_{y_i}$$, of the correct class, $$y_i$$, to be greater than any other classes, $$x_iw_j$$, by at least $$\Delta$$ such that the loss becomes zero (clamped with the max operation). 

###### Analytic gradient

Gradient of the loss function for a single example can be written in full detail as:

$$
\nabla_{w} L_i 
  =
  \begin{bmatrix}
    \frac{dL_i}{dw_1} & \frac{dL_i}{dw_2} & \cdots & \frac{dL_i}{dw_C} 
  \end{bmatrix}
  = 
  \begin{bmatrix}
    \frac{dL_i}{dw_{11}} & \frac{dL_i}{dw_{21}} & \cdots & \frac{dL_i}{dw_{y_i1}} & \cdots & \frac{dL_i}{dw_{C1}} \\
    \vdots & \ddots \\
    \frac{dL_i}{dw_{1D}} & \frac{dL_i}{dw_{2D}} & \cdots & \frac{dL_i}{dw_{y_iD}} & \cdots & \frac{dL_i}{dw_{CD}} 
  \end{bmatrix}
$$

First, let's find a sub-gradient $$\frac{dL_i}{dw_{11}}$$ by considering all the terms in equation (1):

$$
\begin{align*}
L_i = &\max(0, x_{i1}w_{11} + x_{i2}w_{12} \ldots + x_{iD}w_{1D} - x_{i1}w_{y_i1} - x_{i2}w_{y_i2} \ldots - x_{iD}w_{y_iD} + \Delta) + \\
 &\max(0, x_{i1}w_{21} + x_{i2}w_{22} \ldots + x_{iD}w_{2D} - x_{i1}w_{y_i1} - x_{i2}w_{y_i2} \ldots - x_{iD}w_{y_iD} + \Delta) + \\
&\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
&\max(0, x_{i1}w_{C1} + x_{i2}w_{C2} \ldots + x_{iD}w_{CD} - x_{i1}w_{y_i1} - x_{i2}w_{y_i2} \ldots - x_{iD}w_{y_iD} + \Delta)
\end{align*}
$$

For a general case, if $$(x_iw_1 - x_iw_{y_i} + \Delta) > 0$$:

$$
\begin{equation}
\frac{dL_i}{dw_{11}} = x_{i1}
\end{equation}
$$

Equivalently, using an indicator function:

$$
\begin{equation}
\frac{dL_i}{dw_{11}} = \mathbb{1}(x_iw_1 - x_iw_{y_i} + \Delta > 0) x_{i1}
\end{equation}
$$

Similarly,

$$
\begin{equation}
\frac{dL_i}{dw_{12}} = \mathbb{1}(x_iw_1 - x_iw_{y_i} + \Delta > 0) x_{i2} \\
\frac{dL_i}{dw_{13}} = \mathbb{1}(x_iw_1 - x_iw_{y_i} + \Delta > 0) x_{i3} \\
\vdots \\
\frac{dL_i}{dw_{1D}} = \mathbb{1}(x_iw_1 - x_iw_{y_i} + \Delta > 0) x_{iD}
\end{equation}
$$

Hence, 

$$
\begin{align*}
\frac{dL_i}{dw_{j}} &= \mathbb{1}(x_iw_j - x_iw_{y_i} + \Delta > 0)
  \begin{bmatrix}
  x_{i1} \\
  x_{i2} \\
  \vdots \\
  x_{iD}
  \end{bmatrix}
\\
&= \mathbb{1}(x_iw_j - x_iw_{y_i} + \Delta > 0) x_i \tag{2}
\end{align*}
$$

For a special case where $$j=y_i$$,

$$
\begin{equation}
\frac{dL_i}{dw_{y_{i1}}} = -(\ldots) x_{i1}
\end{equation}
$$

The coefficent of $$x_{i1}$$ is the number of classes that meet the desire margin. Mathematically speaking, $$\sum_{j\neq y_i} \mathbb{1}(x_iw_j - x_iw_{y_i} + \Delta > 0)$$. Hence,

$$
\begin{align*}
\frac{dL_i}{dw_{y_i}} &= - \sum_{j\neq y_i} \mathbb{1}(x_iw_j - x_iw_{y_i} + \Delta > 0)
  \begin{bmatrix}
  x_{i1} \\
  x_{i2} \\
  \vdots \\
  x_{iD}
  \end{bmatrix}
\\
&= - \sum_{j\neq y_i} \mathbb{1}(x_iw_j - x_iw_{y_i} + \Delta > 0) x_i \tag{3}
\end{align*}
$$

Equipped with equations (1), (2) and (3), we have enough information to implement a loss function and gradient update.

The IPython Notebook [svm.ipynb](http://vision.stanford.edu/teaching/cs231n/winter1516_assignment1.zip) from [Stanford CS231n](http://vision.stanford.edu/teaching/cs231n/syllabus.html) is a great starting point to understand implementation of the SVM classifier. The exercise asks us to implement both non-vectorized and vectorized versions of loss function and gradient update.

###### Non-vectorized implementation

Looking at the terms in equation (1) suggests us to compute scores of all classes with $$x_iW$$, given an example $$x_i$$. The loss function can be implemented with two for-loops. The inner loop collects loss of all classes of a single example and the outer loop collects it across all examples.

We compute analytic gradient
$$
\nabla_{w} L_i 
  =
  \begin{bmatrix}
    \frac{dL_i}{dw_1} & \frac{dL_i}{dw_2} & \cdots & \frac{dL_i}{dw_C} 
  \end{bmatrix}
$$
one element at a time in the inner loop. Considering equation (2), we compute the gradient w.r.t. weights of class $$j$$ with `dW[:,j] += X[i,:]`. Note that we use `+=` here as we have to collect the gradient across all classes $$j$$ and across all examples. Considering equation (3), we compute the gradient of class $y_i$ with `dW[:,y[i]] -= X[i,:]`. Unlike the previous case, this single class $$y_i$$ requires us to count the number of classes that satisfy the margin condition; hence, the use of `-=`.

{% highlight python %}
def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i,:].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 
      if margin > 0:
        loss += margin
        dW[:,y[i]] -= X[i,:] 
        dW[:,j] += X[i,:] 

  # Averaging over all examples
  loss /= num_train
  dW /= num_train

  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  
  return loss, dW
{% endhighlight %}
<center><em><sup>cs231n/classifiers/linear_svm.py</sup></em></center>

---

###### Vectorized implementation

Instead of computing scores for each example, $$x_iW$$, we can compute them all at once with full matrix multiplication, $$XW$$. To compute the loss, this score matrix has to be subtracted row-wise by scores of correct classes and then added with $$\Delta$$. Because the loss equation sums over all $$j$$ except $$y_i$$, we have to set the $$y_i$$ component to zero. The trick to select correct-class scores across all examples is to use an array indexing technique together with NumPy's `arange`. The idea of computing loss is illustrated below.

[![Illustration of computing loss with fully vectorized operations]({{ site.url }}/assets/article_images/2017-01-06-vectorized-implementation-of-svm-loss-and-gradient-update/1.jpg)]({{ site.url }}/assets/article_images/2017-01-06-vectorized-implementation-of-svm-loss-and-gradient-update/1.jpg){:target="_blank"}

Computing gradient in fully vectorized form is more complicated, but, fortunately, intermediate result of loss computation can be reused. When we compute the loss, we produce a matrix, called `margins` (see the code below). This is almost exactly what we need to compute gradient in equation (2). The indicator function in (2) suggests us to "binarize" this matrix with `binary[margins > 0] = 1`. According to equation (3), we need to update the binarized matrix by summing across each column with `row_sum = np.sum(binary, axis=1)`, taking the negative values, and assigning them to $$y_i$$ components with `binary[np.arange(num_train), y] = -row_sum.T`. Finally, by looking at the last component of both equations, we multiply the binarized matrix with $$X^T$$. (The gradient matrix is of shape DxC; the only way to produce this is $$X^T binary$$) 

Below is how vectorized computation flows. 

[![Illustration of computing gradient of SVM loss with fully vectorized operations]({{ site.url }}/assets/article_images/2017-01-06-vectorized-implementation-of-svm-loss-and-gradient-update/2.jpg)]({{ site.url }}/assets/article_images/2017-01-06-vectorized-implementation-of-svm-loss-and-gradient-update/2.jpg){:target="_blank"}

For those who are still unconvinced, see alternative explanation below.

[![Step-by-step explanation]({{ site.url }}/assets/article_images/2017-01-06-vectorized-implementation-of-svm-loss-and-gradient-update/3.jpg)]({{ site.url }}/assets/article_images/2017-01-06-vectorized-implementation-of-svm-loss-and-gradient-update/3.jpg){:target="_blank"}

{% highlight python %}
def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  #############################################################################
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  yi_scores = scores[np.arange(scores.shape[0]),y] # http://stackoverflow.com/a/23435843/459241 
  margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1)
  margins[np.arange(num_train),y] = 0
  loss = np.mean(np.sum(margins, axis=1))
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  binary = margins
  binary[margins > 0] = 1
  row_sum = np.sum(binary, axis=1)
  binary[np.arange(num_train), y] = -row_sum.T
  dW = np.dot(X.T, binary)

  # Average
  dW /= num_train

  # Regularize
  dW += reg*W

  return loss, dW
{% endhighlight %}
<center><em><sup>cs231n/classifiers/linear_svm.py</sup></em></center>

---

###### Further Reading
* [http://cs231n.github.io/linear-classify/](http://cs231n.github.io/linear-classify/)
* [http://cs231n.github.io/optimization-1/](http://cs231n.github.io/optimization-1/)
* [http://cs231n.github.io/assignments2016/assignment1/](http://cs231n.github.io/assignments2016/assignment1/)



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
