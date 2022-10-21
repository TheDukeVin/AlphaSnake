# Multi-agent Snake
This project aims to apply tree-based search algorithms to the game of multi-agent snake to find approximate Nash Equilibria. Our current implementation uses a variation of the AlphaZero algorithm to achieve state-of-the-art performance in single-agent Snake.

To run, use the following command to compile all files:

```
g++ -O2 -std=c++11 -pthread main.cpp data.cpp convNet.cpp environment.cpp InputLayerCode.cpp trainer.cpp
```

and execute `./a.out`. The current version uses multithreading and can run on larger systems to accelerate the training. 

The output of the training is logged in details.out, which contains data about all training games generated during the execution. To visualize this data, open visual.html, enter the file details.out into the prompt, and enter the game number.

Below is a demonstration of the algorithm's progress over 6000 training games.