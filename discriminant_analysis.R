

Comp_priors <- function(train_labels,K) {
  #' Compute the priors of each class label 
  #' 
  #' @param train_labels a vector of labels with length equal to n
  #' @param K the number of classes in the response
  #' 
  #' @return a probability vector of length K
  
  pi_vec <- rep(0, K)
  
  #####################################################################
  #  TODO                                                             #
  #####################################################################
  for (k in 0:K-1) {
    pi_vec[k+1] = sum(train_labels==k)/length(train_labels)
  }
  
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  
  return(pi_vec)
}
  


Comp_cond_means <- function(train_data, train_labels, K) {
  #' Compute the conditional means of each class 
  #' 
  #' @param train_data a n by p matrix containing p features of n training points
  #' @param train_labels a vector of labels with length equal to n
  #' @param K the number of levels in the response
  #' 
  #' @return a p by K matrix, each column represents the conditional mean given
  #'   each class.
  
  p <- ncol(train_data)
  mean_mat <- matrix(0, p, K)
  
  #####################################################################
  #  TODO                                                             #
  #####################################################################
  k=0
  while (k < K) {
    k_index = which(train_labels==k)
    k_mat = matrix(0,nrow = length(k_index), ncol = p)
    for (i in 1:length(k_index)) {
      k_mat[i,] = train_data[k_index[i],]
    }
    means = colMeans(k_mat)
    mean_mat[,k+1] = means
    k=k+1
  }
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  
  return(mean_mat)
}



Comp_cond_covs <- function(train_data, train_labels, K, method = "LDA") {
  #' Compute the conditional covariance matrix of each class
  #' 
  #' @param train_data a n by p matrix containing p features of n training points
  #' @param train_labels a vector of labels with length equal to n
  #' @param K the number of levels in the response
  #' @param method one of the methods in "LDA", "QDA" and "NB"
  #' 
  #' @return 
  #'  if \code{method} is "QDA", return an array with dimension (p, p, K),
  #'    containing p by p covariance matrices of each class;
  #'  else if \code{method} is "NB", return a p by K matrix containing the 
  #'    diagonal covariance entries of each class; 
  #'  else return a p by p covariance matrix.
  
  p <- ncol(train_data)
  
  
  #####################################################################
  #  TODO                                                             #
  #####################################################################
  
  cov_arr <- NA
  
  k=0
  LDA_mat = matrix(0,nrow = p,ncol = p)
  QDA_list = list()
  NB_mat=matrix(NA,nrow=p,ncol=K)
  while (k < K) {
    k_index = which(train_labels==k)
    n_k = length(k_index)
    k_mat = matrix(0,nrow = length(k_index), ncol = p)
    for (i in 1:length(k_index)) {
      k_mat[i,] = train_data[k_index[i],]
    }
    means = colMeans(k_mat)
    diff_mat = k_mat-rep(means,each=nrow(k_mat))
    class_mat = t(diff_mat)%*%diff_mat
    #LDA
    LDA_mat = LDA_mat+class_mat
    #QDA
    class_mat = matrix(0,p,p)
    for (i in 1:n_k) {
      class_mat = class_mat + as.matrix(k_mat[i,]-means)%*%(k_mat[i,]-means)
    }
    QDA_list[[k+1]] = class_mat/as.numeric(n_k-1)
    #NB
    NB_mat[,k+1] = diag(class_mat/(n_k-1))
    k=k+1
  }
  LDA_mat = LDA_mat/(nrow(train_data)-K)
  if (method == "QDA") {
    cov_arr = QDA_list
  } else if (method == "NB") {
    cov_arr = NB_mat
  } else {
    cov_arr = LDA_mat
  }
  
  return(cov_arr)
  
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
}




Predict_posterior <- function(test_data, priors, means, covs, method = "LDA") {
  
  #' Predict the posterior probabilities of each class 
  #'
  #' @param test_data a n_test by p feature matrix 
  #' @param priors a vector of prior probabilities with length equal to K
  #' @param means a p by K matrix containing conditional means given each class
  #' @param covs covariance matrices of each class, depending on \code{method}
  #' @param method one of the methods in "LDA", "QDA" and "NB"
  #'   
  #' @return a n_test by K matrix: each row contains the posterior probabilities 
  #'   of each class.
  
  n_test <- nrow(test_data)
  K <- length(priors)
  
  posteriors <- matrix(0, n_test, K) 
  
  #####################################################################
  #  TODO                                                             #
  #####################################################################

  sum = rep(0,nrow(test_data))
  p = ncol(test_data)
  post_mat = matrix(0,nrow(test_data),K)
  for (i in 1:nrow(test_data)) {
    k = 0
    x_i = as.matrix(test_data[i,])
    while (k<K) { 
      if (method == "LDA") {
        cov_k = covs
      } else if (method == "QDA") {
        cov_k = (covs[[k+1]])
      } else {
        cov_k = diag(covs[,k+1])
      }
      mu_k = means[,k+1]
      diff_k = as.matrix(x_i - mu_k)
      prior_k = priors[k+1]
      cov_k = as.matrix(cov_k)
      f_k = ((2*pi)^(-p/2)) * (det(cov_k)^(-1/2)) * exp((-1/2)*t(diff_k)%*%solve(cov_k)%*%diff_k)
      sum[i]=sum[i]+prior_k*f_k
      post_mat[i,k+1] = prior_k*f_k
      k=k+1
    }
    post_mat[i,] = post_mat[i,]/as.numeric(sum[i])
  }
  posteriors = post_mat
  
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  
  return(posteriors)
}


Predict_labels <- function(posteriors) {
  
  #' Predict labels based on the posterior probabilities over K classes
  #' 
  #' @param posteriors A n by K posterior probabilities
  #' 
  #' @return A vector of predicted labels with length equal to n
  
  n_test <- nrow(posteriors)
  pred_labels <- rep(NA, n_test)
  
  #####################################################################
  #  TODO                                                             #
  #####################################################################
  for (i in 1:n_test) {
    pred_labels[i] = which.max(posteriors[i,])
  }
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  
  return(pred_labels)
}




