---
layout: post
title: Implementing a Softmax Classifier with Vectorized Operations
comments: true
---

Implementing a Softmax classifier is almost similar to [SVM one]({{ site.url }}/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html), except using a different loss function. A Softmax classifier optimizes a cross-entropy loss that has the form:

$$
\begin{align*}
L_i &= -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right) \\ 
&= -f_{y_i} + \log\sum_j e^{f_j} \tag{1} \\
\end{align*}
$$

where 

* $$f_j(z) = \frac{e^{z_j}}{\sum_k e^{z_k}}$$ is a Softmax function,
* $$L_i$$ is loss for classifying a single example $$x_i$$,
* $$y_i$$ is the index of the correct class of $$x_i$$, and
* $$f_j$$ is the score for predicting class $$j$$, computed by $$x_iw_j$$

Equation (1) makes sense as the loss will be minimum when score of predicting the correct class ($$f_{y_i}$$) is maximized and accumulated scores for predicting other classess ($$f_j$$) is minimized.

It is not difficult to derive analytic gradient of this loss function as follows:

$$
\begin{align*}
\frac{dL_i}{dw_j} = \frac{e^{f_{j}}}{\sum_j e^{f_j}} x_i \tag{2}
\end{align*}
$$

and

$$
\begin{align*}
\frac{dL_i}{dw_{y_i}} = ( \frac{e^{f_{y_i}}}{\sum_j e^{f_j}} - 1 ) x_i \tag{3}
\end{align*}
$$

###### Vectorized computation of loss function

The term $$f_j = x_iw_j$$ in equation (1) suggests us to vectorize computation with a matrix multiplication, $$f = XW$$. To avoid [numerical stability](http://cs231n.github.io/linear-classify/#softmax), we have to update the scores matrix with $$-\max f_j$$.

###### Vectorized computation of gradient

Looking at the terms $$\frac{e^{f}}{\sum_j e^{f_j}}$$ in equations (2) and (3), we can vectorize computation by multiplying $$X$$ with $$W$$, as in the case of loss computation. To get the term $$\frac{e^{f}}{\sum_j e^{f_j}} x_i$$, $$XW$$ has to multiply with $$X$$ *somehow*. Keep in mind that the gradient matrix $$\frac{dL_i}{dW}$$ shares the shape with $$W$$, that is, $$D\times C$$. Vectorization becomes clear once we analyze the shape of these matrices. In this case, the only way to produce $$\frac{dL_i}{dW}$$ of shape $$D \times C$$ is $$\underset{D\times N}X^T(\underset{N\times D}X \underset{D\times C}W)$$.

The IPython Notebook [softmax.ipynb](http://vision.stanford.edu/teaching/cs231n/winter1516_assignment1.zip) from [Stanford CS231n](http://vision.stanford.edu/teaching/cs231n/syllabus.html) is a great starting point to understand implementation of a Softmax classifier. The exercise asks us to implement both non-vectorized and vectorized versions of loss function and gradient update. Below is a sample of vectorized implementation.

{% highlight python %}
def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]

  f = X.dot(W)
  f -= np.matrix(np.max(f, axis=1)).T
    
  term1 = -f[np.arange(N), y]
  sum_j = np.sum(np.exp(f), axis=1)
  term2 = np.log(sum_j)
  loss = term1 + term2
  loss /= N 
  loss += 0.5 * reg * np.sum(W * W)
  
  coef = np.exp(f) / np.matrix(sum_j).T
  coef[np.arange(N),y] -= 1
  dW = X.T.dot(coef)
  dW /= N
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
{% endhighlight %}
<center><em><sup>cs231n/classifiers/softmax.py</sup></em></center>

<br/>

###### Further Reading
* [http://cs231n.github.io/linear-classify/#softmax](http://cs231n.github.io/linear-classify/#softmax)





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
