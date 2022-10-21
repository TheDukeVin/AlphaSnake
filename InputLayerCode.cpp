//
//  InputLayerCode.cpp
//  InputLayerCode
//
//  Created by Kevin Du on 1/22/22.
//

#include "snake.h"

InputLayer::InputLayer(int outD, int outH, int outW, int convH, int convW, networkInput* input){
    outputDepth = outD;
    outputHeight = outH;
    outputWidth = outW;
    convHeight = convH;
    convWidth = convW;
    env = input;
    shiftr = (boardx - outputHeight - convHeight + 1) / 2;
    shiftc = (boardy - outputWidth - convWidth + 1) / 2;
    posShiftr = (1 - convHeight) / 2;
    posShiftc = (1 - convWidth) / 2;
    w1 = outputDepth * convHeight * convWidth;
    w2 = convHeight * convWidth;
    w3 = convWidth;
    
    int numSnakeWeights = 4 * outputDepth * convHeight * convWidth;
    int numPosWeights = 3 * outputDepth * convHeight * convWidth;
    //int numParamWeights = 3 * outputDepth;
    numWeights = numSnakeWeights + numPosWeights; // + numParamWeights;
    numBias = outputDepth;
    this->setupParams();
    snakeWeights = weights;
    posWeights = weights + numSnakeWeights;
    //paramWeights = weights + (numSnakeWeights + numPosWeights);
    DsnakeWeights = Dweights;
    DposWeights = Dweights + numSnakeWeights;
    //DparamWeights = Dweights + (numSnakeWeights + numPosWeights);
}

void InputLayer::pass(){
    //double inc;
    for(int j=0; j<outputDepth; j++){
        /*
        inc = 0;
        for(int i=0; i<3; i++){
            inc += env->param[i] * paramWeights[i*outputDepth + j];
        }*/
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                outputs[j*outputHeight*outputWidth + x*outputWidth + y] = bias[j]; // + inc;
            }
        }
    }
    for(int i=0; i<3; i++){
        for(int r=0; r<convHeight; r++){
            for(int c=0; c<convWidth; c++){
                int outputr = env->pos[i][0] + r + posShiftr;
                int outputc = env->pos[i][1] + c + posShiftc;
                if(outputr >= 0 && outputr < outputHeight && outputc >= 0 && outputc < outputWidth){
                    for(int j=0; j<outputDepth; j++){
                        outputs[j*outputHeight*outputWidth + outputr*outputWidth + outputc] += posWeights[i*w1 + j*w2 + r*w3 + c];
                    }
                }
            }
        }
    }
    double output;
    for(int j=0; j<outputDepth; j++){
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                output = outputs[j*outputHeight*outputWidth + x*outputWidth + y];
                for(int r=0; r<convHeight; r++){
                    for(int c=0; c<convWidth; c++){
                        int inputr = x + r + shiftr;
                        int inputc = y + c + shiftc;
                        if(inputr >= 0 && inputr < boardx && inputc >= 0 && inputc < boardy){
                            int input = env->snake[inputr][inputc];
                            if(input != -1){
                                output += snakeWeights[input*w1 + j*w2 + r*w3 + c];
                            }
                        }
                    }
                }
                outputs[j*outputHeight*outputWidth + x*outputWidth + y] = nonlinear(output);
            }
        }
    }
}

void InputLayer::accumulateGradient(){
    double sum;
    for(int j=0; j<outputDepth; j++){
        sum = 0;
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                sum += Doutputs[j*outputHeight*outputWidth + x*outputWidth + y];
            }
        }
        /*
        for(int i=0; i<3; i++){
            DparamWeights[i*outputDepth + j] += sum * env->param[i];
        }
         */
        Dbias[j] += sum;
    }
    for(int i=0; i<3; i++){
        for(int r=0; r<convHeight; r++){
            for(int c=0; c<convWidth; c++){
                int outputr = env->pos[i][0] + r + posShiftr;
                int outputc = env->pos[i][1] + c + posShiftc;
                if(outputr >= 0 && outputr < outputHeight && outputc >= 0 && outputc < outputWidth){
                    for(int j=0; j<outputDepth; j++){
                        DposWeights[i*w1 + j*w2 + r*w3 + c] += Doutputs[j*outputHeight*outputWidth + outputr*outputWidth + outputc];
                    }
                }
            }
        }
    }
    double Doutput;
    for(int j=0; j<outputDepth; j++){
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                Doutput = Doutputs[j*outputHeight*outputWidth + x*outputWidth + y];
                for(int r=0; r<convHeight; r++){
                    for(int c=0; c<convWidth; c++){
                        int inputr = x + r + shiftr;
                        int inputc = y + c + shiftc;
                        if(inputr >= 0 && inputr < boardx && inputc >= 0 && inputc < boardy){
                            int input = env->snake[inputr][inputc];
                            if(input != -1){
                                DsnakeWeights[input*w1 + j*w2 + r*w3 + c] += Doutput;
                            }
                        }
                    }
                }
            }
        }
    }
}
