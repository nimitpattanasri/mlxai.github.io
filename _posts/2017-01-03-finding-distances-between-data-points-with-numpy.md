---
layout: post
title: Finding distances between data points with NumPy
---

Finding distances between training and test data is essential to a k-Nearest Neighbor (kNN) classifier. The IPython Notebook [knn.ipynb](http://vision.stanford.edu/teaching/cs231n/winter1516_assignment1.zip) from Stanford CS231n will walk us through implementing the kNN classifier for classifying images data.

> The goal of this exercise is to wrap our head around vectorized array operations with NumPy.

First, let’s warm up with finding L2 distances by implementing two for-loops. The code at line 21, from innermost to outermost, first takes the difference element-wise between two data points, square them element-wise, sum across all elements, and then take the square root.

{% highlight python linenos=table %}
def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.
    Input:
    X - An num_test x dimension array where each row is a test point.
    Output:
    dists - A num_test x num_train array where dists[i, j] is the distance
            between the ith test point and the jth training point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]               #
        #####################################################################
        dists[i,j] = np.sqrt(np.sum(np.square(X[i,:]-self.X_train[j,:])))
    return dists
{% endhighlight %}
<center><em><sup>cs231n/classifiers/k_nearest_neighbor.py</sup></em></center>

---

Second, the exercise challenges us to implement with one for-loop. Instead of finding one distance at a time, find ones between each test data point and all training data points. The trick is boiled down into one broadcast at line 15.

{% highlight python linenos=table %}
def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.
    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      dists[i,:] = np.sqrt(np.sum(np.square(self.X_train - X[i,:]), axis=1))
    return dists
{% endhighlight %}
<center><em><sup>cs231n/classifiers/k_nearest_neighbor.py</sup></em></center>

---

Last, to implement the function without using loops is a fun exercise. One may be tempted to use two gigantic broadcasts as explained in [here](http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc). This is an elegant but inefficient solution.

The exercise instead gives a hint of how to vectorize these operations efficiently with one matrix multiplication and two broadcast sums. The trick is to use a quadratic form.

[![A quadratic form with some rearrangement suggesting two broadcast sums and one matrix multiplication]({{ site.url }}/assets/article_images/2017-01-03-finding-distances-between-data-points-with-numpy/1.jpg)]({{ site.url }}/assets/article_images/2017-01-03-finding-distances-between-data-points-with-numpy/1.jpg){:target="_blank"}

[![An illustration of fully vectorized implementation]({{ site.url }}/assets/article_images/2017-01-03-finding-distances-between-data-points-with-numpy/2.jpg)]({{ site.url }}/assets/article_images/2017-01-03-finding-distances-between-data-points-with-numpy/2.jpg){:target="_blank"}

{% highlight python linenos=table %}
def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.
    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################

    # Don't do this gigantic broadcasts. Too long to execute...
    # dists = np.sqrt(np.sum(np.square(X[:,np.newaxis,:] - self.X_train[np.newaxis,:,:]), axis=2))
    
    M = np.dot(X, self.X_train.T)
    te = np.square(X).sum(axis=1)
    tr = np.square(self.X_train).sum(axis=1)
    dists = np.sqrt(-2*M + np.matrix(tr) + np.matrix(te).T)

    return dists
{% endhighlight %}
<center><i><sup>cs231n/classifiers/k_nearest_neighbor.py</sup></i></center>
