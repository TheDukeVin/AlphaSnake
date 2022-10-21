//
//  data.cpp
//  data
//
//  Created by Kevin Du on 2/2/22.
//

#include "snake.h"

Data::Data(Environment* givenEnv, double givenExpected){
    e.copyEnv(givenEnv);
    expectedValue = givenExpected;
}

void Data::trainAgent(Agent& a){
    int symID = rand()%8;
    e.inputSymmetric(a, symID);
    a.valueExpected = expectedValue;
    if(e.actionType == 0 && !e.isEndState()){
        for(int i=0; i<numAgentActions; i++){
            a.policyExpected[(symDir[symID][0]*i + symDir[symID][1] + 4) % 4] = expectedPolicy[i];
        }
        a.backProp(PASS_FULL);
    }
    else{
        a.backProp(PASS_VALUE);
    }
    assert(abs(a.valueOutput) < 1000);
}

DataQueue::DataQueue(){
    index = 0;
    numFilled = 0;
    for(int i=0; i<queueSize; i++){
        queue[i] = NULL;
    }
}

void DataQueue::enqueue(Data* d, int gameLength){
    assert(index < currSize);
    if(queue[index] != NULL){
        delete queue[index];
    }
    queue[index] = d;
    gameLengths[index] = gameLength;
    numFilled = max(index + 1, numFilled);
    index++;
    if(index == currSize){
        index = 0;
    }
}

void DataQueue::trainAgent(Agent& a){
    int i,j;
    for(i=0; i<numBatches; i++){
        for(j=0; j<batchSize; j++){
            int gameIndex = rand() % numFilled;
            queue[gameIndex][rand() % gameLengths[gameIndex]].trainAgent(a);
        }
        a.updateParameters(learnRate / batchSize, momentum);
    }
}

vector<int> DataQueue::readGames(){
    ifstream fin("games.in");
    vector<int> scores;
    while(true){
        string hold;
        if(!(fin>>hold)){
            break;
        }
        if(hold == "Game"){
            fin>>hold>>hold;
            continue;
        }
        int input = stoi(hold);
        
        vector<Environment> envs;
        Environment initialEnv;
        initialEnv.initialize();
        initialEnv.chanceAction(input);
        envs.push_back(initialEnv);
        
        for(int i=1; true; i++){
            fin>>input;
            assert(envs[i-1].validAction(input));
            Environment new_env;
            new_env.setAction(&envs[i-1], input);
            envs.push_back(new_env);
            if(new_env.isEndState()){
                break;
            }
        }
        int gameLength = envs.size();
        
        Data* game = new Data[gameLength];
        
        double value = envs[gameLength-1].getReward();
        for(int i=gameLength-1; i>=0; i--){
            game[i].expectedValue = value;
            game[i].e = envs[i];
            if(i > 0){
                value = envs[i-1].getReward() + value * pow(discountFactor, envs[i].timer - envs[i-1].timer);
            }
        }
        enqueue(game, gameLength);
        //maxScore = max(maxScore, game[gameLength - 1].e.getScore());
        cout<<game[gameLength - 1].e.snakeSize<<',';
        scores.push_back(game[gameLength - 1].e.snakeSize);
        
        for(int i=0; i<gameLength; i++){
            double value;
            fin>>value;
            assert(abs(value - game[i].expectedValue) < 0.001);
        }
        for(int i=0; i<6*gameLength; i++){
            double value;
            fin>>value;
        }
        for(int i=0; i<gameLength; i++){
            double sum = 0;
            double policy[numAgentActions];
            for(int j=0; j<numAgentActions; j++){
                fin>>policy[j];
                if(policy[j] > 0){
                    sum += policy[j];
                }
            }
            if(sum > 0){
                assert(abs(sum - 1) < 0.01);
            }
            assert((game[i].e.isEndState() || game[i].e.actionType == 1) == (sum == 0));
            /*
            if(game[i].e.isEndState() != (sum == 0)){
                cout<<'\n'<<i<<'\n';
                game[i].e.log();
                cout<<"End state: " << game[i].e.isEndState();
                assert(false);
            }*/
            for(int j=0; j<numAgentActions; j++){
                if(policy[j] < 0){
                    game[i].expectedPolicy[j] = -1;
                }
                else{
                    game[i].expectedPolicy[j] = policy[j] / sum;
                }
                if(game[i].e.actionType == 0 && !game[i].e.isEndState()){
                    assert((game[i].e.validAction(j)) == (game[i].expectedPolicy[j] != -1));
                }
            }
        }
    }
    cout<<"\n\n";
    return scores;
}
