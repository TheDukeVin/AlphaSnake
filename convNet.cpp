//
//  convNet_code.cpp
//  convNet_code
//
//  Created by Kevin Du on 1/17/22.
//


#include "snake.h"

double squ(double x){
    return x * x;
}

int max(int x, int y){
    if(x < y) return y;
    return x;
}

double max(double x, double y){
    if(x < y) return y;
    return x;
}

double min(double x, double y){
    if(x < y) return x;
    return y;
}


double abs(double x){
    if(x < 0) return -x;
    return x;
}

double randWeight(double startingParameterRange){
    return (((double)rand() / RAND_MAX)*2-1) * startingParameterRange;
}

double nonlinear(double x){
    if(x>0) return x;
    return x*0.1;
}

double dinvnonlinear(double x){
    if(x>0) return 1;
    return 0.1;
}

// Layer

void Layer::setupParams(){
    numParams = numWeights + numBias;
    //params = new double[numParams];
    params = (double*) malloc(numParams * sizeof(double));
    assert(params != NULL);
    weights = params;
    bias = params + numWeights;
    //Dparams = new double[numParams];
    Dparams = (double*) malloc(numParams * sizeof(double));
    assert(Dparams != NULL);
    Dweights = Dparams;
    Dbias = Dparams + numWeights;
}

void Layer::randomize(double startingParameterRange){
    for(int i=0; i<numParams; i++){
        params[i] = randWeight(startingParameterRange);
    }
}

void Layer::resetGradient(){
    for(int i=0; i<numParams; i++){
        Dparams[i] = 0;
    }
}

void Layer::updateParameters(double mult, double momentum){
    for(int i=0; i<numParams; i++){
        params[i] -= Dparams[i] * mult;
        Dparams[i] *= momentum;
    }
    // Regularize
    double sum = 0;
    for(int i=0; i<numParams; i++){
        sum += squ(params[i]);
    }
    //cout<<"Param sum: "<<sum<<'\n';
    /*
    if(sum <= maxNorm) return;
    for(int i=0; i<numParams; i++){
        params[i] *= sqrt(maxNorm / sum);
    }*/
}

void Layer::save(){
    for(int i=0; i<numParams; i++){
        (*netOut)<<params[i]<<' ';
    }
    (*netOut)<<"\n\n";
}

void Layer::readNet(){
    for(int i=0; i<numParams; i++){
        (*netIn)>>params[i];
    }
}


// ConvLayer

ConvLayer::ConvLayer(int inD, int inH, int inW, int outD, int outH, int outW, int convH, int convW){
    inputDepth = inD;
    inputHeight = inH;
    inputWidth = inW;
    outputDepth = outD;
    outputHeight = outH;
    outputWidth = outW;
    convHeight = convH;
    convWidth = convW;
    
    shiftr = (inputHeight - outputHeight - convHeight + 1) / 2;
    shiftc = (inputWidth - outputWidth - convWidth + 1) / 2;
    w1 = outputDepth * convHeight * convWidth;
    w2 = convHeight * convWidth;
    w3 = convWidth;
    
    numWeights = inputDepth * outputDepth * convHeight * convWidth;
    numBias = outputDepth;
    this->setupParams();
}

void ConvLayer::pass(){
    double sum;
    for(int j=0; j<outputDepth; j++){
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                sum = bias[j];
                for(int r=0; r<convHeight; r++){
                    for(int c=0; c<convWidth; c++){
                        int inputr = x + r + shiftr;
                        int inputc = y + c + shiftc;
                        if(inputr >= 0 && inputr < inputHeight && inputc >= 0 && inputc < inputWidth){
                            for(int i=0; i<inputDepth; i++){
                                sum += inputs[i*inputHeight*inputWidth + inputr*inputWidth + inputc] * weights[i*w1 + j*w2 + r*w3 + c];
                            }
                        }
                    }
                }
                outputs[j*outputHeight*outputWidth + x*outputWidth + y] = nonlinear(sum);
            }
        }
    }
}

void ConvLayer::backProp(bool increment){
    for(int i=0; i<inputDepth*inputHeight*inputWidth; i++){
        Dinputs[i] = 0;
    }
    for(int j=0; j<outputDepth; j++){
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                for(int r=0; r<convHeight; r++){
                    for(int c=0; c<convWidth; c++){
                        int inputr = x + r + shiftr;
                        int inputc = y + c + shiftc;
                        if(inputr >= 0 && inputr < inputHeight && inputc >= 0 && inputc < inputWidth){
                            for(int i=0; i<inputDepth; i++){
                                Dinputs[i*inputHeight*inputWidth + inputr*inputWidth + inputc] += weights[i*w1 + j*w2 + r*w3 + c] * Doutputs[j*outputHeight*outputWidth + x*outputWidth + y];
                            }
                        }
                    }
                }
            }
        }
    }
    for(int i=0; i<inputDepth*inputHeight*inputWidth; i++){
        Dinputs[i] *= dinvnonlinear(inputs[i]);
    }
}

void ConvLayer::accumulateGradient(){
    for(int j=0; j<outputDepth; j++){
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                double Dout = Doutputs[j*outputHeight*outputWidth + x*outputWidth + y];
                Dbias[j] += Dout;
                for(int r=0; r<convHeight; r++){
                    for(int c=0; c<convWidth; c++){
                        int inputr = x + r + shiftr;
                        int inputc = y + c + shiftc;
                        if(inputr >= 0 && inputr < inputHeight && inputc >= 0 && inputc < inputWidth){
                            for(int i=0; i<inputDepth; i++){
                                Dweights[i*w1 + j*w2 + r*w3 + c] += inputs[i*inputHeight*inputWidth + inputr*inputWidth + inputc] * Dout;
                            }
                        }
                    }
                }
            }
        }
    }
}


// PoolLayer

PoolLayer::PoolLayer(int inD, int inH, int inW, int outD, int outH, int outW){
    inputDepth = inD;
    inputHeight = inH;
    inputWidth = inW;
    outputDepth = outD;
    outputHeight = outH;
    outputWidth = outW;
    
    numWeights = 0;
    numBias = 0;
    numParams = 0;
    
    maxIndices = new int[outD * outH * outW];
}

void PoolLayer::pass(){
    double maxVal,candVal;
    int maxIndex;
    int index;
    for(int j=0; j<outputDepth; j++){
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                maxVal = -100000;
                maxIndex = -1;
                for(int r=0; r<2; r++){
                    for(int c=0; c<2; c++){
                        index = j*inputHeight*inputWidth + (2*x+r)*inputWidth + (2*y+c);
                        candVal = inputs[index];
                        if(maxVal < candVal){
                            maxVal = candVal;
                            maxIndex = index;
                        }
                    }
                }
                index = j*outputHeight*outputWidth + x*outputWidth + y;
                outputs[index] = maxVal;
                maxIndices[index] = maxIndex;
            }
        }
    }
}

void PoolLayer::backProp(bool increment){
    for(int i=0; i<inputDepth*inputHeight*inputWidth; i++){
        Dinputs[i] = 0;
    }
    for(int i=0; i<outputDepth*outputHeight*outputWidth; i++){
        Dinputs[maxIndices[i]] = Doutputs[i];
    }
}


// DenseLayer

DenseLayer::DenseLayer(int inSize, int outSize){
    inputSize = inSize;
    outputSize = outSize;
    
    numWeights = inputSize * outputSize;
    numBias = outputSize;
    this->setupParams();
}

void DenseLayer::pass(){
    double sum;
    for(int i=0; i<outputSize; i++){
        sum = bias[i];
        for(int j=0; j<inputSize; j++){
            sum += weights[j*outputSize + i] * inputs[j];
        }
        outputs[i] = nonlinear(sum);
    }
}

void DenseLayer::backProp(bool increment){
    double sum;
    for(int i=0; i<inputSize; i++){
        sum = 0;
        for(int j=0; j<outputSize; j++){
            sum += weights[i*outputSize + j] * Doutputs[j];
        }
        if(increment){
            Dinputs[i] += sum * dinvnonlinear(inputs[i]);
        }
        else{
            Dinputs[i] = sum * dinvnonlinear(inputs[i]);
        }
    }
}

void DenseLayer::accumulateGradient(){
    for(int i=0; i<outputSize; i++){
        Dbias[i] += Doutputs[i];
        for(int j=0; j<inputSize; j++){
            Dweights[j*outputSize + i] += Doutputs[i] * inputs[j];
        }
    }
}

// Output Layer

OutputLayer::OutputLayer(int inSize, int outSize){
    inputSize = inSize;
    outputSize = outSize;
    
    numWeights = inputSize * outputSize;
    numBias = outputSize;
    this->setupParams();
}

void OutputLayer::pass(){
    double sum;
    for(int i=0; i<outputSize; i++){
        sum = bias[i];
        for(int j=0; j<inputSize; j++){
            sum += weights[j*outputSize + i] * inputs[j];
        }
        outputs[i] = sum;
    }
}

void OutputLayer::backProp(bool increment){
    double sum;
    for(int i=0; i<inputSize; i++){
        sum = 0;
        for(int j=0; j<outputSize; j++){
            sum += weights[i*outputSize + j] * Doutputs[j];
        }
        Dinputs[i] = sum * dinvnonlinear(inputs[i]);
    }
}

void OutputLayer::accumulateGradient(){
    for(int i=0; i<outputSize; i++){
        Dbias[i] += Doutputs[i];
        for(int j=0; j<inputSize; j++){
            Dweights[j*outputSize + i] += Doutputs[i] * inputs[j];
        }
    }
}


// Branch

void Branch::initEnvironmentInput(int depth, int height, int width, int convHeight, int convWidth){
    Layer* layer = new InputLayer(depth, height, width, convHeight, convWidth, input);
    
    layerHold.push_back(layer);
    prevDepth = depth;
    prevHeight = height;
    prevWidth = width;
    
    int numOutputs = depth * height * width;
    prevActivation = new double[numOutputs];
    prevDbias = new double[numOutputs];
    layer->outputs = prevActivation;
    layer->Doutputs = prevDbias;
}

void Branch::addConvLayer(int depth, int height, int width, int convHeight, int convWidth){
    Layer* layer = new ConvLayer(prevDepth, prevHeight, prevWidth, depth, height, width, convHeight, convWidth);
    
    layerHold.push_back(layer);
    prevDepth = depth;
    prevHeight = height;
    prevWidth = width;
    
    layer->inputs = prevActivation;
    layer->Dinputs = prevDbias;
    int numOutputs = depth * height * width;
    prevActivation = new double[numOutputs];
    prevDbias = new double[numOutputs];
    layer->outputs = prevActivation;
    layer->Doutputs = prevDbias;
    output = layer->outputs;
    Doutput = layer->Doutputs;
}

void Branch::addPoolLayer(int depth, int height, int width){
    Layer* layer = new PoolLayer(prevDepth, prevHeight, prevWidth, depth, height, width);
    
    layerHold.push_back(layer);
    prevDepth = depth;
    prevHeight = height;
    prevWidth = width;
    
    layer->inputs = prevActivation;
    layer->Dinputs = prevDbias;
    int numOutputs = depth * height * width;
    prevActivation = new double[numOutputs];
    prevDbias = new double[numOutputs];
    layer->outputs = prevActivation;
    layer->Doutputs = prevDbias;
    output = layer->outputs;
    Doutput = layer->Doutputs;
}

void Branch::addFullyConnectedLayer(int numNodes){
    Layer* layer = new DenseLayer(prevDepth * prevHeight * prevWidth, numNodes);
    
    layerHold.push_back(layer);
    prevDepth = numNodes;
    prevHeight = 1;
    prevWidth = 1;
    
    layer->inputs = prevActivation;
    layer->Dinputs = prevDbias;
    prevActivation = new double[numNodes];
    prevDbias = new double[numNodes];
    layer->outputs = prevActivation;
    layer->Doutputs = prevDbias;
    output = layer->outputs;
    Doutput = layer->Doutputs;
}

void Branch::addOutputLayer(int numNodes){
    Layer* layer = new OutputLayer(prevDepth * prevHeight * prevWidth, numNodes);
    
    layerHold.push_back(layer);
    
    layer->inputs = prevActivation;
    layer->Dinputs = prevDbias;
    prevActivation = new double[numNodes];
    prevDbias = new double[numNodes];
    layer->outputs = prevActivation;
    layer->Doutputs = prevDbias;
    output = layer->outputs;
    Doutput = layer->Doutputs;
}

void Branch::setup(){
    for(int i=0; i<numLayers; i++){
        layers[i] = layerHold[i];
    }
}

// ConvNet

void Agent::setupCommonBranch(){
    int numCommonOutput = commonBranch.prevDepth * commonBranch.prevHeight * commonBranch.prevWidth;
    policyBranch.prevDepth = numCommonOutput;
    policyBranch.prevHeight = 1;
    policyBranch.prevWidth = 1;
    policyBranch.prevActivation = commonBranch.output;
    policyBranch.prevDbias = commonBranch.Doutput;
    
    valueBranch.prevDepth = numCommonOutput;
    valueBranch.prevHeight = 1;
    valueBranch.prevWidth = 1;
    valueBranch.prevActivation = commonBranch.output;
    valueBranch.prevDbias = commonBranch.Doutput;
}

void Agent::setup(){
    commonBranch.numLayers = commonBranch.layerHold.size();
    policyBranch.numLayers = policyBranch.layerHold.size();
    valueBranch.numLayers = valueBranch.layerHold.size();
    numLayers = commonBranch.numLayers + policyBranch.numLayers + valueBranch.numLayers;
    layers = new Layer*[numLayers];
    commonBranch.layers = layers;
    policyBranch.layers = layers + commonBranch.numLayers;
    valueBranch.layers = layers + commonBranch.numLayers + policyBranch.numLayers;
    commonBranch.setup();
    policyBranch.setup();
    valueBranch.setup();
    resetGradient();
}


void Agent::randomize(double startingParameterRange){
    for(int l=0; l<numLayers; l++){
        layers[l]->randomize(startingParameterRange);
    }
}

void Agent::pass(int mode){
    for(int i=0; i<commonBranch.numLayers; i++){
        commonBranch.layers[i]->pass();
    }
    for(int i=0; i<valueBranch.numLayers; i++){
        valueBranch.layers[i]->pass();
    }
    valueOutput = valueBranch.output[0];
    assert(abs(valueOutput) < 1000);
    if(mode == PASS_VALUE){
        return;
    }
    
    for(int i=0; i<policyBranch.numLayers; i++){
        policyBranch.layers[i]->pass();
    }
    double sum = 0;
    for(int i=0; i<numAgentActions; i++){
        if(validAction[i]){
            sum += exp(policyBranch.output[i]);
            assert(abs(policyBranch.output[i]) < 1000);
        }
    }
    if(sum == 0){
        return;
    }
    for(int i=0; i<numAgentActions; i++){
        if(validAction[i]){
            policyOutput[i] = exp(policyBranch.output[i]) / sum;
        }
        else{
            policyOutput[i] = -1;
        }
    }
}

void Agent::resetGradient(){
    for(int l=0; l<numLayers; l++){
        layers[l]->resetGradient();
    }
}

void Agent::backProp(int mode){
    pass(mode);
    valueBranch.Doutput[0] = 2 * (valueOutput - valueExpected);
    for(int i=valueBranch.numLayers-1; i>=0; i--){
        valueBranch.layers[i]->accumulateGradient();
        valueBranch.layers[i]->backProp();
    }
    
    if(mode == PASS_FULL){
        double sum = 0;
        for(int i=0; i<numAgentActions; i++){
            if(validAction[i]){
                policyBranch.Doutput[i] = policyOutput[i] - policyExpected[i];
                sum += policyExpected[i];
            }
            else{
                policyBranch.Doutput[i] = 0;
            }
        }
        if(abs(sum - 1) > 0.000001){
            cout<<"Incorrect sum: "<<sum<<'\n';
            assert(false);
        }
        for(int i=policyBranch.numLayers-1; i>0; i--){
            policyBranch.layers[i]->accumulateGradient();
            policyBranch.layers[i]->backProp();
        }
        policyBranch.layers[0]->accumulateGradient();
        policyBranch.layers[0]->backProp(true);
    }
    for(int i=commonBranch.numLayers-1; i>0; i--){
        commonBranch.layers[i]->accumulateGradient();
        commonBranch.layers[i]->backProp();
    }
    commonBranch.layers[0]->accumulateGradient();
}

void Agent::updateParameters(double mult, double momentum){
    for(int l=0; l<numLayers; l++){
        layers[l]->updateParameters(mult, momentum);
    }
}

void Agent::save(string fileName){
    netOut = new ofstream(fileName);
    for(int l=0; l<numLayers; l++){
        layers[l]->netOut = netOut;
    }
    for(int l=0; l<numLayers; l++){
        layers[l]->save();
    }
    netOut->close();
}

void Agent::readNet(string fileName){
    netIn = new ifstream(fileName);
    for(int l=0; l<numLayers; l++){
        layers[l]->netIn = netIn;
    }
    for(int l=0; l<numLayers; l++){
        layers[l]->readNet();
    }
    netIn->close();
}

void Agent::copyParam(Agent& a){
    for(int l=0; l<numLayers; l++){
        for(int i=0; i<layers[l]->numParams; i++){
            layers[l]->params[i] = a.layers[l]->params[i];
        }
    }
}
