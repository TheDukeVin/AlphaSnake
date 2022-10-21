//
//  trainer.cpp
//  trainer
//
//  Created by Kevin Du on 1/18/22.
//

#include "snake.h"



void Trainer::initializeNode(Environment& env, int currNode){
    if(outcomes[currNode] != NULL){
        delete outcomes[currNode];
    }
    int numOutcomes = numActions[env.actionType];
    outcomes[currNode] = new int[numOutcomes];
    for(int i=0; i<numOutcomes; i++){
        if(!env.validAction(i)){
            outcomes[currNode][i] = -2;
        }
        else{
            outcomes[currNode][i] = -1;
        }
    }
    subtreeSize[currNode] = 0;
    sumScore[currNode] = 0;
    
    // Evaluate the network at the current node.
/*
    // Use learned features

    values[currNode] = 3 + env.getReward()
                         + ((env.features(2) - 1) * 4) 
                         - (env.features(3) * 0.1) 
                         + (1 - env.features(5));

    if(env.actionType == 0){
        int numValidActions = 0;
        for(int d=0; d<numAgentActions; d++){
            numValidActions ++;
        }
        for(int d=0; d<numAgentActions; d++){
            if(outcomes[currNode][d] != -2){
                policy[currNode][d] = 1.0 / numValidActions;
            }
            else{
                policy[currNode][d] = -1;
            }
        }
    }
    */
    
    int symID = rand()%8;
    env.inputSymmetric(a, symID);
    if(env.actionType == 0){
        a.pass(PASS_FULL);
    }
    else{
        a.pass(PASS_VALUE);
    }
    values[currNode] = a.valueOutput;
    if(env.actionType == 0){
        for(int d=0; d<numAgentActions; d++){
            policy[currNode][d] = a.policyOutput[(symDir[symID][0]*d + symDir[symID][1] + 4) % 4];
            if(outcomes[currNode][d] != -2){
                assert(policy[currNode][d] != -1);
            }
        }
    }
}

void Trainer::trainTree(){
    double search_values[maxTime*2];
    double search_policies[maxTime*2][numAgentActions];
    for(int i=0; i<maxTime*2; i++){
        for(int j=0; j<numAgentActions; j++){
            search_policies[i][j] = -1;
        }
    }
    for(int i=0; i<maxStates; i++){
        for(int j=0; j<numAgentActions; j++){
            policy[i][j] = -1;
        }
    }
    
    roots[0].initialize();
    rootIndex = 0;
    //ofstream fout(valueLog, ios::app);
    //fout<<(roots[0].applex * boardy + roots[0].appley)<<' ';
    initializeNode(roots[0], 0);
    index = 1;
    
    int chosenAction;
    for(rootState=0; rootState<maxTime*2; rootState++){
        rootIndices[rootState] = rootIndex;
        if(roots[rootState].actionType == 0){
            for(int j=0; j<numPaths; j++){
                expandPath();
            }
            computeActionProbs();
            for(int j=0; j<numAgentActions; j++){
                search_policies[rootState][j] = actionProbs[j];
            }
            chosenAction = sampleActionProbs();
        }
        else{
            chosenAction = getRandomChanceAction(&roots[rootState]);
            if(outcomes[rootIndex][chosenAction] == -1){
                outcomes[rootIndex][chosenAction] = index;
                Environment env;
                env.setAction(&roots[rootState], chosenAction);
                initializeNode(env, index);
                index++;
            }
        }
        if(subtreeSize[rootIndex] != 0){
            search_values[rootState] = sumScore[rootIndex] / subtreeSize[rootIndex];
        }
        else{
            search_values[rootState] = 0;
        }
        roots[rootState+1].setAction(&roots[rootState], chosenAction);
        rootIndex = outcomes[rootIndex][chosenAction];
        //fout<<chosenAction<<' ';
        if(roots[rootState+1].isEndState()){
            break;
        }
    }
    //fout<<"\n";

    int numStates = rootState + 2;
    Data* game = new Data[numStates];

    double value = roots[numStates-1].getReward();
    for(int i=numStates-1; i>=0; i--){
        game[i] = Data(&roots[i], value);
        if(i > 0){
            value = roots[i-1].getReward() + value * pow(discountFactor, roots[i].timer - roots[i-1].timer);
        }
    }
    for(int i=0; i<numStates; i++){
        for(int j=0; j<numAgentActions; j++){
            game[i].expectedPolicy[j] = search_policies[i][j];
        }
    }
    //dq->enqueue(game, numStates);
    output_gameLength = numStates;
    output_game = game;
/*
    for(int i=0; i<numStates; i++){
        fout<<game[i].expectedValue<<' ';
    }
    fout<<"\n";
    
    for(int i=0; i<numStates; i++){
        fout<<values[rootIndices[i]]<<' ';
    }
    fout<<"\n";
    
    search_values[numStates - 1] = game[numStates - 1].e.getReward();
    for(int i=0; i<numStates; i++){
        fout<<search_values[i];
        if(i != numStates-1) fout<<' ';
    }
    fout<<"\n";
    
    for(int i=0; i<numStates; i++){
        for(int j=0; j<numAgentActions; j++){
            fout<<policy[rootIndices[i]][j]<<' ';
        }
    }
    fout<<"\n";
    
    for(int i=0; i<numStates; i++){
        for(int j=0; j<numAgentActions; j++){
            fout<<search_policies[i][j]<<' ';
        }
    }
    fout<<"\n";

    fout.close();

    return &roots[numStates-1];*/
}

void Trainer::expandPath(){
    int currNode = rootIndex;
    int nextNode,nextAction;
    int count = 0;
    int currType;
    int maxIndex;
    double maxVal,candVal;
    int i;
    Environment env;
    env.copyEnv(&roots[rootState]);

    for(int i=0; i<2*maxTime; i++){
        times[i] = -1;
    }
    
    while(currNode != -1 && !env.isEndState()){
        path[count] = currNode;
        rewards[count] = env.getReward();
        times[count] = env.timer;
        count++;
        currType = env.actionType;
        maxVal = -1000000;
        maxIndex = -1;
        for(i=0; i<numActions[currType]; i++){
            nextNode = outcomes[currNode][i];
            if(nextNode == -2){
                continue;
            }
            if(currType == 0){
                assert(policy[currNode][i] != -1);
                double Qval;
                int size = 0;
                if(nextNode == -1){
                    if(subtreeSize[currNode] == 0){
                        Qval = 0;
                    }
                    else{
                        Qval = sumScore[currNode] / subtreeSize[currNode];
                    }
                }
                else{
                    Qval = sumScore[nextNode] / subtreeSize[nextNode];
                    size = subtreeSize[nextNode];
                }
                candVal = Qval + explorationConstant * policy[currNode][i] * sqrt(subtreeSize[currNode] + 1) / (size + 1);
                //candVal = sumScore[nextNode] / subtreeSize[nextNode] + explorationConstant * log(subtreeSize[currNode]) / sqrt(subtreeSize[nextNode]);
            }
            if(currType == 1){
                if(nextNode == -1){
                    candVal = (double) rand() / RAND_MAX + 1;
                }
                else{
                    candVal = (double)rand() / RAND_MAX - subtreeSize[nextNode];
                }
            }
            if(candVal > maxVal){
                maxVal = candVal;
                maxIndex = i;
            }
        }
        assert(maxIndex != -1);
        nextAction = maxIndex;
        currNode = outcomes[currNode][maxIndex];
        env.makeAction(maxIndex);
    }
    double newVal;
    if(currNode == -1){
        outcomes[path[count-1]][nextAction] = index;
        initializeNode(env, index);
        
        newVal = values[index];
        
        path[count] = index;
        times[count] = env.timer;
        index++;
        count++;
    }
    else{
        newVal = env.getReward();
        path[count] = currNode;
        times[count] = env.timer;
        count++;
    }
    double value = newVal;
    for(i=count-1; i>=0; i--){
        subtreeSize[path[i]]++;
        sumScore[path[i]] += value;
        assert(times[i] >= 0);
        if(i > 0){
            value = rewards[i-1] + value * pow(discountFactor, times[i] - times[i-1]);
        }
    }
}

void Trainer::printTree(){
    /*
    ofstream fout(outAddress, ios::app);
    for(int i=0; i<index; i++){
        fout<<"State "<<i<<'\n';
        states[i].print();
        fout<<"Outcomes: ";
        for(int j=0; j<numActions[states[i].actionType]; j++){
            fout<<outcomes[i][j];
        }
        fout<<'\n';
        fout<<"Size: "<<size[i]<<'\n';
        fout<<"Sum score: "<<sumScore[i]<<'\n';
        fout<<'\n';
    }
    fout.close();*/
}

void Trainer::computeActionProbs(){
    int i;
    int nextIndex;
    double sum = 0;
    for(i=0; i<numAgentActions; i++){
        nextIndex = outcomes[rootIndex][i];
        if(nextIndex != -2){
            actionProbs[i] = pow(subtreeSize[nextIndex], actionTemperature);
            sum += actionProbs[i];
        }
        else{
            actionProbs[i] = -1;
        }
    }
    for(int i=0; i<numAgentActions; i++){
        if(actionProbs[i] != -1){
            actionProbs[i] /= sum;
        }
    }
}

int Trainer::optActionProbs(){
    int i;
    int maxIndex = 0;
    for(i=1; i<numAgentActions; i++){
        if(actionProbs[i] > actionProbs[maxIndex]){
            maxIndex = i;
        }
    }
    return maxIndex;
}

int Trainer::sampleActionProbs(){
    int i;
    double parsum = 0;
    double randReal = (double)rand() / RAND_MAX;
    
    int actionIndex = -1;
    for(i=0; i<numAgentActions; i++){
        if(actionProbs[i] == -1){
            continue;
        }
        parsum += actionProbs[i];
        if(randReal <= parsum){
            actionIndex = i;
            break;
        }
    }
    return actionIndex;
}

int Trainer::getRandomChanceAction(Environment* e){
    int i;
    int possibleActions[numChanceActions];
    int numPossibleActions = 0;
    for(i=0; i<numChanceActions; i++){
        if(e->validAction(i)){
            possibleActions[numPossibleActions] = i;
            numPossibleActions++;
        }
    }
    return possibleActions[rand() % numPossibleActions];
}

