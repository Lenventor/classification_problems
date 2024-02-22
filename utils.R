# Functions to load the data and to print digits

library(readr)


Load_data <- function(data_dir) {
  # Function to read the data
  raw_data <- data.matrix(read_csv(data_dir, col_names = FALSE))
  x_data <- raw_data[,-1]
  y_data <- raw_data[,1]
  return(list(x = x_data, y = y_data))
}





Plot_digits <- function(digits_index, digits_data){
  # Function to plot the specified data points
  # 
  # @param digits_index: a vector of indices of the digits to be visualized
  # @param digits_data: the matrix with each row representing the features 
  #   for one digit
  
  val <- par(no.readonly=TRUE) # record the current setup of par 
  
  par(mfrow=c(2, 5), mar=c(2, 1, 2, 0.5))
  for (i in digits_index){ 
    m <- matrix(digits_data[i,], nrow=8, byrow=T) 
    plot(as.raster(m), asp=NA) 
  }
  
  par(val)  #reset the original graphics parameters
}