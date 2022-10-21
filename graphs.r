library(graphics)
library(grid)
library(ggplot2)

data <- read.delim("/Users/kevindu/Desktop/Employment/multiagent_snake/multiagent_snake/features.txt", sep='\t', header = FALSE)

nums <- data[1,]

N = length(nums)

NUM_AVG = 50

consec_avg <- function(lst, N){
  avgs <- integer(length(lst) / N)
  for(i in 0:(length(lst) / N - 1)){
    x = 0
    count = 0
    for(j in 1:N){
      num = as.numeric(lst[i*N + j])
      if(num >= 0){
        x = x + num
        count = count + 1
      }
    }
    avgs[i+1] = x / count
  }
  avgs
}

consec_sd <- function(lst, N){
  ans <- integer(length(lst) / N)
  consec = integer(N)
  for(i in 0:(length(lst) / N - 1)){
    count = 0
    for(j in 1:N){
      num = as.numeric(lst[i*N + j])
      if(num >= 0){
        consec[count+1] = num
        count = count + 1
      }
    }
    ans[i+1] = sd(consec[1:count]) / sqrt(count) * 2
  }
  ans
}

M = floor(N / NUM_AVG)
x = 1:M * NUM_AVG

par(las=1, mar = c(2.1, 2.5, 0.1, 0.1))

shade_error <- function(values, color){
  avg_values = consec_avg(values, NUM_AVG)
  sd_values = consec_sd(values, NUM_AVG)
  polygon(c(x, rev(x)), c(avg_values + sd_values, rev(avg_values - sd_values)), col=color, border = NA)
}

plot_with_error <- function(values, range, label, color, shade, add_values = NA, add_color = NA, add_shade = NA, include_axis = TRUE){
  plot(c(), c(),
       xlim=c(1,N), ylim=range,
       xlab="", ylab="",
       xaxt="n", yaxt="n")
  
  shade_error(values, col=shade)
  if(!is.na(add_values)){
    shade_error(add_values, add_shade)
  }
  lines(x, consec_avg(values, NUM_AVG), col=color)
  if(!is.na(add_values)){
    lines(x, consec_avg(add_values, NUM_AVG), col=add_color)
  }
  axis(2, mgp=c(3, .5, 0))
  if(include_axis){
    axis(1, mgp=c(3, .3, 0))
  }
  title(ylab=label, line=1.7, cex.lab=1)
  title(xlab="Game Number", line=1.2, cex.lab=1)
}

pink = rgb(255, 200, 200, max = 255, alpha = 125)
light_blue = rgb(200, 200, 255, max = 255, alpha = 125)
grey = rgb(200, 200, 200, max = 255, alpha = 125)

#Individual plots

plot_with_error(nums, c(50,100), "Average Score", "red", pink)
lines(c(0, N), c(59.09, 59.09), col="black", lty=2)
lines(c(0, N), c(100, 100), col="gray30", lty=3)
legend(2500, 85, legend=c("AlphaZero", "Naive tree search", "Maximum score"),
       col=c("red", "black", "gray30"), lty=1:3, cex=0.8)

plot_with_error(nums == 100, c(0,1), "Win Rate", "blue", light_blue)
# plot_with_error(data[2,], c(0.06, 0.16), "Perimeter States", "black", grey)
# 
# plot_with_error(data[3,], c(0.7,1), "Apple and Tail Visibility", "black", grey,
#                 add_values = data[4,], add_color = "red", add_shade = pink)
# legend(4000, 0.8, legend=c("Apple visibility", "Tail visibility"),
#        col=c("black", "red"), lty = c(1,1), cex=0.8)
# 
# plot_with_error(data[5,], c(4.7,6.5), "L1 and Dynamic Distance", "black", grey,
#                 add_values = data[6,], add_color = "red", add_shade = pink)
# 
# legend(3500, 6.3, legend=c("L1 distance", "Dynamic distance"),
#        col=c("black", "red"), lty = c(1,1), cex=0.8)
# 
# plot_with_error(data[7,], c(1.6, 2.4), "Connected Components", "black", grey)
# plot_with_error(data[10,], c(0.16,0.25), "Turns", "black", grey)


# Stacked plots



# plot_with_error(nums, c(50,100), "Average Score", "red", pink)
# lines(c(0, N), c(59.09, 59.09), col="black", lty=2)
# lines(c(0, N), c(100, 100), col="gray30", lty=3)
# legend(2500, 85, legend=c("AlphaZero", "Naive tree search", "Maximum score"),
#        col=c("red", "black", "gray30"), lty=1:3, cex=0.8)
# 
# plot_with_error(nums == 100, c(0,1), "Win Rate", "blue", light_blue)
# 
# xpar = 2.5
# rmargin = 0.2
# 
# layout(matrix(1:5, ncol = 1), widths = 1, heights = c(1,1,1,1,1.3), respect = FALSE)
# par(mar = c(0, xpar, rmargin, rmargin))
# plot_with_error(data[2,], c(0.057, 0.16), "Perimeter States", "black", grey, include_axis = FALSE)
# 
# par(mar = c(0, xpar, 0, rmargin))
# plot_with_error(data[3,], c(0.67,1.02), "Apple/Tail Visibility", "black", grey,
#                 add_values = data[4,], add_color = "red", add_shade = pink, include_axis = FALSE)
# legend(4400, 0.82, legend=c("Apple visibility", "Tail visibility"),
#        col=c("black", "red"), lty = c(1,1), cex=0.8)
# 
# par(mar = c(0, xpar, 0, rmargin))
# plot_with_error(data[5,], c(4.7,6.55), "L1/Dynamic Distance", "black", grey,
#                 add_values = data[6,], add_color = "red", add_shade = pink, include_axis = FALSE)
# 
# legend(4400, 6.5, legend=c("L1 distance", "Dynamic distance"),
#        col=c("black", "red"), lty = c(1,1), cex=0.8)
# 
# par(mar = c(0, xpar, 0, rmargin))
# plot_with_error(data[7,], c(1.6, 2.4), "Connected\nComponents", "black", grey, include_axis = FALSE)
# 
# par(mar = c(2.5, xpar, 0, rmargin))
# plot_with_error(data[10,], c(0.16,0.25), "Turns", "black", grey)
# 
