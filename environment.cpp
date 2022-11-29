//
//  environment.cpp
//  environment
//
//  Created by Kevin Du on 1/18/22.
//

#include "snake.h"

void Environment::initialize(){
    timer = 0;
    score = 0;
    actionType = 0;
    snakeSize = 2;
    headx = boardx/2;
    heady = 2;
    tailx = headx;
    taily = 1;
    
    int i,j;
    for(i=0; i<boardx; i++){
        for(j=0; j<boardy; j++){
           snake[i][j] = -1;
        }
    }
    snake[headx][heady] = 4;
    snake[tailx][taily] = 0;
    
    while(true){
        applex = rand()%boardx;
        appley = rand()%boardy;
        if(snake[applex][appley] == -1){
            break;
        }
    }
}

bool Environment::isEndState(){
    if(timer == maxTime){
        return true;
    }
    int newx,newy;
    for(int d=0; d<4; d++){
        newx = headx + dir[d][0];
        newy = heady + dir[d][1];
        if(newx == -1 || newx == boardx){
            continue;
        }
        if(newy == -1 || newy == boardy){
            continue;
        }
        if(snake[newx][newy] == -1){
            return false;
        }
    }
    return true;
}

double Environment::getScore(){
    if(snakeSize == boardx * boardy) return score + 10 + (maxTime - timer) * 0.2;
    return score;
}

bool Environment::validAction(int actionIndex){ // returns whether the action is valid.
    if(actionType == 0){
        return validAgentAction(actionIndex);
    }
    else{
        return validChanceAction(actionIndex);
    }
}

bool Environment::validAgentAction(int d){
    int newHeadx = headx + dir[d][0];
    int newHeady = heady + dir[d][1];
    if(newHeadx == -1 || newHeadx == boardx){
        return false;
    }
    if(newHeady == -1 || newHeady == boardy){
        return false;
    }
    return snake[newHeadx][newHeady] == -1;
}

bool Environment::validChanceAction(int pos){
    int newApplex = pos / boardy;
    int newAppley = pos % boardy;
    return snake[newApplex][newAppley] == -1;
}

void Environment::makeAction(int actionIndex){
    if(actionType == 0){
        agentAction(actionIndex);
    }
    else{
        chanceAction(actionIndex);
    }
    
    // Unfold path
    while(actionType == 0 && !isEndState()){
        int nextAction = -1;
        for(int i=0; i<numAgentActions; i++){
            if(validAction(i)){
                if(nextAction == -1){
                    nextAction = i;
                }
                else{
                    return;
                }
            }
        }
        agentAction(nextAction);
    }
}

void Environment::setAction(Environment* currState, int actionIndex){
    copyEnv(currState);
    makeAction(actionIndex);
}

void Environment::agentAction(int actionIndex){
    timer++;
    
    int newHeadx = headx + dir[actionIndex][0];
    int newHeady = heady + dir[actionIndex][1];
    snake[headx][heady] = actionIndex;
    headx = newHeadx;
    heady = newHeady;
    snake[newHeadx][newHeady] = 4;
    
    if(headx == applex && heady == appley){
        score += 1;
        //score += 1 - timer*0.5/maxTime;
        snakeSize++;
        actionType = 1;
    }
    else{
        int tailDir = snake[tailx][taily];
        snake[tailx][taily] = -1;
        tailx += dir[tailDir][0];
        taily += dir[tailDir][1];
    }
}

void Environment::chanceAction(int actionIndex){
    applex = actionIndex / boardy;
    appley = actionIndex % boardy;
    actionType = 0;
}

void Environment::inputSymmetric(Agent& net, int t){
    networkInput* a = net.input;
    int m = boardx-1;
    int sym[8][2][3] = {
        {{ 1, 0, 0},{ 0, 1, 0}},
        {{ 0,-1, m},{ 1, 0, 0}},
        {{-1, 0, m},{ 0,-1, m}},
        {{ 0, 1, 0},{-1, 0, m}},
        {{ 0, 1, 0},{ 1, 0, 0}},
        {{ 1, 0, 0},{ 0,-1, m}},
        {{ 0,-1, m},{-1, 0, m}},
        {{-1, 0, m},{ 0, 1, 0}}
    };
    /*
    a->param[0] = (double) timer / maxTime;
    a->param[1] = score / scoreNorm;
    a->param[2] = actionType;*/
    a->pos[0][0] = sym[t][0][0]*headx + sym[t][0][1]*heady + sym[t][0][2];
    a->pos[0][1] = sym[t][1][0]*headx + sym[t][1][1]*heady + sym[t][1][2];
    a->pos[1][0] = sym[t][0][0]*tailx + sym[t][0][1]*taily + sym[t][0][2];
    a->pos[1][1] = sym[t][1][0]*tailx + sym[t][1][1]*taily + sym[t][1][2];
    a->pos[2][0] = sym[t][0][0]*applex + sym[t][0][1]*appley + sym[t][0][2];
    a->pos[2][1] = sym[t][1][0]*applex + sym[t][1][1]*appley + sym[t][1][2];
    
    int i,j,x,y;
    for(i=0; i<boardx; i++){
        for(j=0; j<boardy; j++){
            x = sym[t][0][0]*i + sym[t][0][1]*j + sym[t][0][2];
            y = sym[t][1][0]*i + sym[t][1][1]*j + sym[t][1][2];
            if(snake[i][j] == -1 || snake[i][j] == 4){
                a->snake[x][y] = -1;
            }
            else{
                a->snake[x][y] = (symDir[t][0]*snake[i][j] + symDir[t][1] + 4) % 4;
            }
        }
    }
    
    // FILL IN VALID ACTIONS FOR NETWORK
    if(actionType == 0){
        for(int i=0; i<4; i++){
            net.validAction[(symDir[t][0]*i + symDir[t][1] + 4) % 4] = validAction(i);
        }
    }
}

void Environment::copyEnv(Environment* e){
    timer = e->timer;
    score = e->score;
    actionType = e->actionType;
    snakeSize = e->snakeSize;
    headx = e->headx;
    heady = e->heady;
    tailx = e->tailx;
    taily = e->taily;
    applex = e->applex;
    appley = e->appley;
    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            snake[i][j] = e->snake[i][j];
        }
    }
}

void Environment::print(){ // optional function for debugging
    ofstream fout(outAddress, ios::app);
    fout<<"Timer: "<<timer<<'\n';
    fout<<"Score: "<<score<<'\n';
    fout<<"Action type: "<<actionType<<'\n';
    fout<<"Snake size: "<<snakeSize<<'\n';
    int i,j;
    for(i=0; i<boardx; i++){
        for(j=0; j<boardy; j++){
            if(i == applex && j == appley){
                fout<<'A'<<' ';
            }
            else{
                if(snake[i][j] == -1){
                    fout<<". ";
                }
                else{
                    fout<<snake[i][j]<<' ';
                }
            }
        }
        fout<<'\n';
    }
    fout.close();
}

void Environment::log(){ // optional function for debugging
    cout<<"Timer: "<<timer<<'\n';
    cout<<"Score: "<<score<<'\n';
    cout<<"Action type: "<<actionType<<'\n';
    cout<<"Snake size: "<<snakeSize<<'\n';
    int i,j;
    for(i=0; i<boardx; i++){
        for(j=0; j<boardy; j++){
            if(i == applex && j == appley){
                cout<<'A'<<' ';
            }
            else{
                if(snake[i][j] == -1){
                    cout<<". ";
                }
                else{
                    cout<<snake[i][j]<<' ';
                }
            }
        }
        cout<<'\n';
    }
}

double Environment::getReward(){
    if(snakeSize == boardx * boardy) return 20;
    if(timer == maxTime) return 0;
    if(isEndState()) return -20;
    if(actionType == 1) return 1;
    return 0;
}

void Environment::BFS(int* dist, int sourcex, int sourcey, bool fill){
    if(fill){
        for(int i=0; i<boardx * boardy; i++){
            dist[i] = -1;
        }
    }
    list<int> queue;
    queue.push_back(sourcex * boardy + sourcey);
    dist[sourcex * boardy + sourcey] = 0;
    int numVisibleNodes = 0;
    while(!queue.empty()){
        int currNode = queue.front();
        queue.pop_front();
        numVisibleNodes++;
        int currx = currNode / boardy;
        int curry = currNode % boardy;
        for(int i=0; i<4; i++){
            int nextx = currx + dir[i][0];
            int nexty = curry + dir[i][1];
            if(nextx != -1 && nextx != boardx && nexty != -1 && nexty != boardy && 
            (snake[nextx][nexty] == -1 || (nextx == tailx && nexty == taily) ) && dist[nextx * boardy + nexty] == -1){
                dist[nextx * boardy + nexty] = dist[currNode] + 1;
                queue.push_back(nextx * boardy + nexty);
            }
        }
    }
}

double Environment::features(int featureType){
    if(featureType == 0){
        int currx = headx;
        int curry = heady;
        for(int r=0; r<10; r++){
            if(currx != 0 && currx != boardx-1 && curry != 0 && curry != boardy-1){
                return 0;
            }
            int nextx, nexty;
            int i;
            for(i=0; i<4; i++){
                nextx = currx + dir[i][0];
                nexty = curry + dir[i][1];
                if(nextx == -1 || nextx == boardx){
                    continue;
                }
                if(nexty == -1 || nexty == boardy){
                    continue;
                }
                if(snake[nextx][nexty] == (i + 2) % 4){
                    break;
                }
            }
            if(currx == tailx && curry == taily){
                return 0;
            }
            assert(i < 4);
            currx = nextx;
            curry = nexty;
        }
        return 1;
    }
    else if(featureType == 1){
        int dist[boardx * boardy];
        BFS(dist, headx, heady);
        if(dist[applex * boardy + appley] != -1){
            return 1;
        }
        return 0;
    }
    else if(featureType == 2){
        int dist[boardx * boardy];
        BFS(dist, headx, heady);
        if(dist[tailx * boardy + taily] != -1){
            return 1;
        }
        return 0;
    }
    else if(featureType == 3){
        return distance(0);
    }
    else if(featureType == 4){
        return distance(1);
    }
    else if(featureType == 5){
        int dist[boardx * boardy];
        for(int i=0; i<boardx * boardy; i++){
            dist[i] = -1;
        }
        int numConn = 0;
        for(int i=0; i<boardx; i++){
            for(int j=0; j<boardy; j++){
                if(dist[i * boardy + j] == -1 && snake[i][j] == -1){
                    numConn++;
                    BFS(dist, i, j, false);
                }
            }
        }
        return numConn;
    }
    // Not very impactful
    /*
    if(actionType == 0){
        return -1;
    }
    int numInvalidActions = 0;
    int invalidAction;
    for(int i=0; i<4; i++){
        int nextx = headx + dir[i][0];
        int nexty = heady + dir[i][1];
        if(nextx == -1 || nextx == boardx || nexty == -1 || nexty == boardy){
            numInvalidActions++;
            invalidAction = i;
        }
    }
    if(numInvalidActions != 1){
        return -1;
    }
    int prevx = headx - dir[invalidAction][0];
    int prevy = heady - dir[invalidAction][1];
    if(snake[prevx][prevy] == invalidAction){
        return 1;
    }
    return 0;
    */
    assert(false);
    return -1;
}

double Environment::distance(int featureType){
    if(actionType == 1){
        return -1;
    }
    if(featureType == 0){
        return abs(headx - applex) + abs(heady - appley);
    }
    else if(featureType == 1){
        Environment env;
        env.copyEnv(this);
        int dist[boardx * boardy];
        env.BFS(dist, applex, appley);
        bool possible = false;
        for(int d=0; d<4; d++){
            int nextx = env.headx + dir[d][0];
            int nexty = env.heady + dir[d][1];
            if(env.validAgentAction(d) && dist[nextx * boardy + nexty] != -1){
                possible = true;
                break;
            }
        }
        if(!possible){
            return -1;
        }
        for(int i=0; true; i++){
            if(env.actionType == 1){
                if(i < distance(0)){
                    cout<<i<<' '<<distance(0)<<'\n';
                    log();
                }
                assert(i >= distance(0));
                return i;
            }
            /*
            cout<<"LOG:\n";
            for(int i=0; i<10; i++){
                for(int j=0; j<10; j++){
                    cout<<dist[i * boardy + j]<<'\t';
                }
                cout<<'\n';
            }
            env.log();
            */
            env.BFS(dist, applex, appley);
            int minIndex = -1;
            int minDist = 1000;
            for(int d=0; d<4; d++){
                int nextx = env.headx + dir[d][0];
                int nexty = env.heady + dir[d][1];
                if(env.validAgentAction(d) && dist[nextx * boardy + nexty] < minDist && dist[nextx * boardy + nexty] != -1){
                    minDist = dist[nextx * boardy + nexty];
                    minIndex = d;
                }
            }
            assert(minIndex != -1);
            assert(env.validAgentAction(minIndex));
            //cout<<"ACTION: "<<minIndex<<'\n';
            //env.log();
            env.agentAction(minIndex);
            
        }
    }
    assert(false);
    return -1;
}

int Environment::lastAction(){
    int nextx, nexty;
    int i;
    for(i=0; i<4; i++){
        nextx = headx + dir[i][0];
        nexty = heady + dir[i][1];
        if(nextx == -1 || nextx == boardx){
            continue;
        }
        if(nexty == -1 || nexty == boardy){
            continue;
        }
        if(snake[nextx][nexty] == (i + 2) % 4){
            break;
        }
    }
    return i;
}

double Environment::features2(Environment* nextState, int featureType){
    if(featureType < 2){
        if(actionType == 1){
            return -1;
        }
        if(nextState->distance(featureType) < distance(featureType)){
            return 1;
        }
        return 0;
    }
    else if(featureType == 2){
        if((lastAction() - nextState->lastAction()) % 2 == 1){
            return 1;
        }
        return 0;
    }
    assert(false);
    return -1;
}
