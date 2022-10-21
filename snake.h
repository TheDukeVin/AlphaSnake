//
//  snake.h
//  snake
//
//  Created by Kevin Du on 1/18/22.
//

#include <iostream>
#include <fstream>
#include <thread>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <string>
#include <vector>
#include <list>
#include <cassert>
#include <mutex>
#include <condition_variable>

#ifndef snake_h
#define snake_h

using namespace std;

//environment details

#define boardx 10
#define boardy 10
#define maxTime 1200

#define numAgentActions 4
#define numChanceActions (boardx*boardy)
#define maxNumActions (boardx*boardy)

//training deatils

#define maxNorm 100
#define batchSize 100
#define numBatches 30

//#define scoreNorm 10
#define queueSize 10000

#define numGames 25000
#define numPaths 200

#define maxStates (maxTime*2*numPaths)
#define evalPeriod 100
#define numEvalGames 100
#define evalZscore 2

#define discountFactor 0.98

// Passing Value or Full

#define PASS_VALUE 0
#define PASS_FULL 1

// Multithreading

#define NUM_THREADS 50

const string outAddress = "snake_conv.txt";

double squ(double x);

int max(int x, int y);

double min(double x, double y);

double abs(double x);

// For the network
double randWeight(double startingParameterRange);

double nonlinear(double x);

// If f = nonlinear, then this function is (f' \circ f^{-1}).
double dinvnonlinear(double x);

// Input to the network

struct networkInput{
    int snake[boardx][boardy];
    int pos[3][2]; // head, tail, and apple positions
    //double param[3]; // timer, score, and actionType. score and timer are normalized.
};

const int symDir[8][2] = {
    { 1,0},
    { 1,3},
    { 1,2},
    { 1,1},
    {-1,1},
    {-1,2},
    {-1,3},
    {-1,0}
};

class Layer{
public:
    ifstream* netIn;
    ofstream* netOut;
    int numParams, numWeights, numBias;
    double* params;
    double* weights;
    double* bias;
    double* Dparams;
    double* Dweights;
    double* Dbias;
    
    double* inputs;
    double* outputs;
    double* Dinputs;
    double* Doutputs;
    
    virtual void pass(){};
    virtual void backProp(bool increment = false){};
    virtual void accumulateGradient(){};
    void setupParams();
    void randomize(double startingParameterRange);
    void resetGradient();
    void updateParameters(double mult, double momentum);
    void save();
    void readNet();
    
    virtual ~Layer(){}
};

class ConvLayer : public Layer{
public:
    int inputDepth, inputHeight, inputWidth;
    int outputDepth, outputHeight, outputWidth;
    int convHeight, convWidth;
    int shiftr, shiftc;
    int w1, w2, w3;
    /*
    double weights[maxDepth][maxDepth][maxConvSize][maxConvSize]; // accessed in inputl, outputl, r, c.
    double bias[maxDepth];
    double Dweights[maxDepth][maxDepth][maxConvSize][maxConvSize];
    double Dbias[maxDepth];*/
    
    ConvLayer(int inD, int inH, int inW, int outD, int outH, int outW, int convH, int convW);
    
    virtual void pass();
    virtual void backProp(bool increment = false);
    virtual void accumulateGradient();
    
    virtual ~ConvLayer(){
        delete[] params;
        delete[] Dparams;
    }
};

class PoolLayer : public Layer{
public:
    int inputDepth, inputHeight, inputWidth;
    int outputDepth, outputHeight, outputWidth;
    int* maxIndices;
    
    PoolLayer(int inD, int inH, int inW, int outD, int outH, int outW);
    
    virtual void pass();
    virtual void backProp(bool increment = false);
    
    virtual ~PoolLayer(){
        delete[] maxIndices;
    }
};

class DenseLayer : public Layer{
public:
    int inputSize, outputSize;
    
    DenseLayer(int inSize, int outSize);
    
    virtual void pass();
    virtual void backProp(bool increment = false);
    virtual void accumulateGradient();
    
    virtual ~DenseLayer(){
        delete[] params;
        delete[] Dparams;
    }
};

// Input layer tailored to snake environment

class InputLayer : public Layer{
public:
    int outputDepth, outputHeight, outputWidth;
    int convHeight, convWidth;
    int shiftr, shiftc;
    int posShiftr, posShiftc;
    int w1, w2, w3;
    
    networkInput* env;
    
    double* snakeWeights; // accessed in cellType, outputl, r, c.
    double* posWeights;
    //double* paramWeights;
    
    double* DsnakeWeights;
    double* DposWeights;
    //double* DparamWeights;
    
    InputLayer(int outD, int outH, int outW, int convH, int convW, networkInput* input);
    
    virtual void pass();
    virtual void accumulateGradient();
    
    virtual ~InputLayer(){
        delete[] params;
        delete[] Dparams;
    }
};

class OutputLayer : public Layer{
public:
    int inputSize, outputSize;
    
    OutputLayer(int inSize, int outputSize);
    
    virtual void pass();
    virtual void backProp(bool increment = false);
    virtual void accumulateGradient();
    
    virtual ~OutputLayer(){
        delete[] params;
        delete[] Dparams;
    }
};

class Branch{
public:
    int numLayers;
    
    // For network initiation
    int prevDepth, prevHeight, prevWidth;
    networkInput* input;
    vector<Layer*> layerHold;
    Layer** layers;
    
    double* prevActivation;
    double* prevDbias;
    double* output;
    double* Doutput;
    
    void initEnvironmentInput(int depth, int height, int width, int convHeight, int convWidth);
    void addConvLayer(int depth, int height, int width, int convHeight, int convWidth);
    void addPoolLayer(int depth, int height, int width);
    void addFullyConnectedLayer(int numNodes);
    void addOutputLayer(int numNodes);
    void setup();
};

class Agent{
public:
    networkInput* input;
    int numLayers;
    
    Layer** layers; // keep an array of pointers, since derived classes need to be accessed by reference.
    Branch commonBranch;
    Branch policyBranch;
    Branch valueBranch;
    
    double policyOutput[numAgentActions];
    double valueOutput;
    bool validAction[numAgentActions]; // MUST BE FILLED IN for evaluation to work.
    double policyExpected[numAgentActions];
    double valueExpected;
    
    // For file I/O
    ifstream* netIn;
    ofstream* netOut;
    
    Agent(){
        input = new networkInput;
        commonBranch.input = input;
    }
    
    // For network usage and training
    void copyParam(Agent& a);
    void setupCommonBranch();
    void setup();
    void resetGradient();
    void randomize(double startingParameterRange);
    
    void pass(int mode); // Inputs are PASS_VALUE or PASS_FULL
    void backProp(int mode);
    void updateParameters(double mult, double momentum);
    
    void save(string fileName);
    void readNet(string fileName);
};

// Environment things

const int numActions[2] = {numAgentActions, numChanceActions};

const int dir[4][2] = {{0,1}, {1,0}, {0,-1}, {-1,0}};

class Environment{
private:
    double score;
public:
    int timer;
    int actionType; // 0 = action state, 1 = reaction state.
    
    int snakeSize;
    int headx,heady;
    int tailx, taily;
    int applex,appley;
    int snake[boardx][boardy]; // -1 = not snake. 0 to 3 = snake unit pointing to next unit. 4 = head.
    
    void initialize();
    
    bool isEndState();
    bool validAction(int actionIndex); // returns whether the action is valid.
    bool validAgentAction(int d);
    bool validChanceAction(int pos);
    void makeAction(int actionIndex);
    void setAction(Environment* currState, int actionIndex);
    void inputSymmetric(Agent& net, int t);
    void copyEnv(Environment* e);
    void print();// optional function for debugging
    void log();// optional function for debugging
    
    double getReward();
    
    void agentAction(int actionIndex);
    void chanceAction(int actionIndex);

    double features(int featureType);
    double features2(Environment* nextState, int featureType);
    double distance(int featureType);

    void BFS(int* dist, int sourcex, int sourcey, bool fill = true);
    int lastAction();
    
private:
    double getScore();
};

// Data things

class Data{
public:
    Environment e;
    double expectedValue;
    double expectedPolicy[numAgentActions];
    
    Data(){}
    Data(Environment* givenEnv, double givenExpected);
    void trainAgent(Agent& a);
};

class DataQueue{
public:
    Data* queue[queueSize];
    int gameLengths[queueSize];
    int index;
    int currSize, numFilled;
    double learnRate, momentum;
    
    DataQueue();
    void enqueue(Data* d, int gameLength);
    void trainAgent(Agent& a);
    vector<int> readGames(); // returns the maximum score out of the games read.
};

// Trainer

class Trainer{
public:
    
    double actionTemperature = 2;
    double explorationConstant = 0.5;
    
    Agent a;

    int output_gameLength;
    Data* output_game;
    
    string gameLog;
    string valueLog;
    
    Trainer(){
        for(int i=0; i<maxStates; i++){
            outcomes[i] = NULL;
        }
    }
    
    //Storage for the tree:
    int* outcomes[maxStates];
    int subtreeSize[maxStates];
    double sumScore[maxStates];
    Environment roots[maxTime*2];
    int rootIndices[maxTime*2];
    double values[maxStates];
    double policy[maxStates][numAgentActions];
    
    // Implementing the tree search
    int index;
    int rootIndex, rootState;
    
    // For executing a training iteration:
    double actionProbs[numAgentActions];
    
    void initializeNode(Environment& env, int currNode);
    void trainTree();
    
    int path[maxStates];
    double rewards[maxTime*2];
    int times[maxTime*2];
    
    void expandPath();
    void printTree();
    void computeActionProbs();
    int optActionProbs();
    int sampleActionProbs();
    int getRandomChanceAction(Environment* e);
};

#endif /* snake_h */
