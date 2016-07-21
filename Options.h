#ifndef _PARSER_OPTIONS_
#define _PARSER_OPTIONS_

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "N3L.h"

using namespace std;

class Options {
public:

  int wordCutOff;
  int featCutOff;
  int charCutOff;
  int tagCutOff;
  dtype initRange;
  int maxIter;
  int batchSize;
  dtype adaEps;
  dtype adaAlpha;
  dtype regParameter;
  dtype dropProb;

  int linearHiddenSize;
  int hiddenSize;
  int rnnHiddenSize;
  int wordEmbSize;
  int wordcontext;
  bool wordEmbFineTune;
  int tagEmbSize;
  bool tagEmbFineTune;
  int charEmbSize;
  int charcontext;
  bool charEmbFineTune;
  int charhiddenSize;

  int verboseIter;
  bool saveIntermediate;
  bool train;
  int maxInstance;
  vector<string> testFiles;
  string outBest;
  bool seg;
  int relu;
  int atomLayers;
  int rnnLayers;

  Options() {
    wordCutOff = 0;
    featCutOff = 0;
    charCutOff = 0;
    tagCutOff = 0;
    initRange = 0.01;
    maxIter = 1000;
    batchSize = 1;
    adaEps = 1e-6;
    adaAlpha = 0.01;
    regParameter = 1e-8;
    dropProb = 0.0;

    linearHiddenSize = 30;
    hiddenSize = 200;
    rnnHiddenSize = 300;
    wordEmbSize = 50;
    wordcontext = 2;
    wordEmbFineTune = true;
    tagEmbSize = 50;
    tagEmbFineTune = true;
    charEmbSize = 50;
    charcontext = 2;
    charEmbFineTune = true;
    charhiddenSize = 50;

    verboseIter = 100;
    saveIntermediate = true;
    train = false;
    maxInstance = -1;
    testFiles.clear();
    outBest = "";
    relu = 0;
    seg = false;
    atomLayers = 1;
    rnnLayers = 1;

  }

  virtual ~Options() {

  }

  void setOptions(const vector<string> &vecOption) {
    int i = 0;
    for (; i < vecOption.size(); ++i) {
      pair<string, string> pr;
      string2pair(vecOption[i], pr, '=');
      if (pr.first == "wordCutOff")
        wordCutOff = atoi(pr.second.c_str());
      if (pr.first == "featCutOff")
        featCutOff = atoi(pr.second.c_str());
      if (pr.first == "charCutOff")
        charCutOff = atoi(pr.second.c_str());
      if (pr.first == "tagCutOff")
        tagCutOff = atoi(pr.second.c_str());        
      if (pr.first == "initRange")
        initRange = atof(pr.second.c_str());
      if (pr.first == "maxIter")
        maxIter = atoi(pr.second.c_str());
      if (pr.first == "batchSize")
        batchSize = atoi(pr.second.c_str());
      if (pr.first == "adaEps")
        adaEps = atof(pr.second.c_str());
      if (pr.first == "adaAlpha")
        adaAlpha = atof(pr.second.c_str());
      if (pr.first == "regParameter")
        regParameter = atof(pr.second.c_str());
      if (pr.first == "dropProb")
        dropProb = atof(pr.second.c_str());

      if (pr.first == "linearHiddenSize")
        linearHiddenSize = atoi(pr.second.c_str());
      if (pr.first == "hiddenSize")
        hiddenSize = atoi(pr.second.c_str());
      if (pr.first == "rnnHiddenSize")
        rnnHiddenSize = atoi(pr.second.c_str());
      if (pr.first == "wordcontext")
        wordcontext = atoi(pr.second.c_str());
      if (pr.first == "wordEmbSize")
        wordEmbSize = atoi(pr.second.c_str());
      if (pr.first == "wordEmbFineTune")
        wordEmbFineTune = (pr.second == "true") ? true : false;
      if (pr.first == "tagEmbSize")
        tagEmbSize = atoi(pr.second.c_str());
      if (pr.first == "tagEmbFineTune")
        tagEmbFineTune = (pr.second == "true") ? true : false;        	
      if (pr.first == "charcontext")
        charcontext = atoi(pr.second.c_str());
      if (pr.first == "charEmbSize")
        charEmbSize = atoi(pr.second.c_str());
      if (pr.first == "charEmbFineTune")
        charEmbFineTune = (pr.second == "true") ? true : false;
      if (pr.first == "charhiddenSize")
        charhiddenSize = atoi(pr.second.c_str());
        
      if (pr.first == "verboseIter")
        verboseIter = atoi(pr.second.c_str());
      if (pr.first == "train")
        train = (pr.second == "true") ? true : false;
      if (pr.first == "saveIntermediate")
        saveIntermediate = (pr.second == "true") ? true : false;
      if (pr.first == "maxInstance")
        maxInstance = atoi(pr.second.c_str());
      if (pr.first == "testFile")
        testFiles.push_back(pr.second);
      if (pr.first == "outBest")
        outBest = pr.second;
      if (pr.first == "relu")
        relu = atoi(pr.second.c_str());
      if (pr.first == "seg")
        seg = (pr.second == "true") ? true : false;
      if (pr.first == "atomLayers")
        atomLayers = atoi(pr.second.c_str());
      if (pr.first == "rnnLayers")
        rnnLayers = atoi(pr.second.c_str());

    }
  }

  void showOptions() {
    std::cout << "wordCutOff = " << wordCutOff << std::endl;
    std::cout << "featCutOff = " << featCutOff << std::endl;
    std::cout << "charCutOff = " << charCutOff << std::endl;
    std::cout << "tagCutOff = " << tagCutOff << std::endl;
    std::cout << "initRange = " << initRange << std::endl;
    std::cout << "maxIter = " << maxIter << std::endl;
    std::cout << "batchSize = " << batchSize << std::endl;
    std::cout << "adaEps = " << adaEps << std::endl;
    std::cout << "adaAlpha = " << adaAlpha << std::endl;
    std::cout << "regParameter = " << regParameter << std::endl;
    std::cout << "dropProb = " << dropProb << std::endl;

    std::cout << "linearHiddenSize = " << linearHiddenSize << std::endl;
    std::cout << "hiddenSize = " << hiddenSize << std::endl;
    std::cout << "rnnHiddenSize = " << rnnHiddenSize << std::endl;
    std::cout << "wordEmbSize = " << wordEmbSize << std::endl;
    std::cout << "wordcontext = " << wordcontext << std::endl;
    std::cout << "wordEmbFineTune = " << wordEmbFineTune << std::endl;
    std::cout << "tagEmbSize = " << tagEmbSize << std::endl;
    std::cout << "tagEmbFineTune = " << tagEmbFineTune << std::endl;
    std::cout << "charEmbSize = " << charEmbSize << std::endl;
    std::cout << "charcontext = " << charcontext << std::endl;
    std::cout << "charEmbFineTune = " << charEmbFineTune << std::endl;
    std::cout << "charhiddenSize = " << charhiddenSize << std::endl;

    std::cout << "verboseIter = " << verboseIter << std::endl;
    std::cout << "saveItermediate = " << saveIntermediate << std::endl;
    std::cout << "train = " << train << std::endl;
    std::cout << "maxInstance = " << maxInstance << std::endl;

    for (int idx = 0; idx < testFiles.size(); idx++) {
      std::cout << "testFile = " << testFiles[idx] << std::endl;
    }      

    std::cout << "outBest = " << outBest << std::endl;
    std::cout << "relu = " << relu << std::endl;
    std::cout << "seg = " << seg << std::endl;
    std::cout << "atomLayers = " << atomLayers << std::endl;
    std::cout << "rnnLayers = " << rnnLayers << std::endl;
  }

  void writeOption(const string& outputModelFile) {
    ofstream outf;
    outf.open(outputModelFile.c_str());
    outf << "BEGIN_OPTION:" << std::endl;
    outf << "wordCutOff = " << wordCutOff << std::endl;
    outf << "featCutOff = " << featCutOff << std::endl;
    outf << "charCutOff = " << charCutOff << std::endl;
    outf << "tagCutOff = " << tagCutOff << std::endl;
    outf << "initRange = " << initRange << std::endl;
    outf << "maxIter = " << maxIter << std::endl;
    outf << "batchSize = " << batchSize << std::endl;
    outf << "adaEps = " << adaEps << std::endl;
    outf << "adaAlpha = " << adaAlpha << std::endl;
    outf << "regParameter = " << regParameter << std::endl;
    outf << "dropProb = " << dropProb << std::endl;

    outf << "linearHiddenSize = " << linearHiddenSize << std::endl;
    outf << "hiddenSize = " << hiddenSize << std::endl;
    outf << "rnnHiddenSize = " << rnnHiddenSize << std::endl;
    outf << "wordEmbSize = " << wordEmbSize << std::endl;
    outf << "wordcontext = " << wordcontext << std::endl;
    outf << "wordEmbFineTune = " << wordEmbFineTune << std::endl;
    outf << "tagEmbSize = " << tagEmbSize << std::endl;
    outf << "tagEmbFineTune = " << tagEmbFineTune << std::endl;
    outf << "charEmbSize = " << charEmbSize << std::endl;
    outf << "charcontext = " << charcontext << std::endl;
    outf << "charEmbFineTune = " << charEmbFineTune << std::endl;
    outf << "charhiddenSize = " << charhiddenSize << std::endl;

    outf << "verboseIter = " << verboseIter << std::endl;
    outf << "saveItermediate = " << saveIntermediate << std::endl;
    outf << "train = " << train << std::endl;
    outf << "maxInstance = " << maxInstance << std::endl;
    for (int idx = 0; idx < testFiles.size(); idx++) {
      outf << "testFile = " << testFiles[idx] << std::endl;
    }
    outf << "outBest = " << outBest << std::endl;
    outf << "relu = " << relu << std::endl;
    outf << "seg = " << seg << std::endl;
    outf << "atomLayers = " << atomLayers << std::endl;
    outf << "rnnLayers = " << rnnLayers << std::endl; 
    outf << "END_OPTION!" << std::endl;  
    outf << std::endl; 
    outf.close();
  }

  void load(const std::string& infile) {
    ifstream inf;
    inf.open(infile.c_str());
    vector<string> vecLine;
    while (1) {
      string strLine;
      if (!my_getline(inf, strLine)) {
        break;
      }
      if (strLine == "END_OPTION!") {
        cout << "Finished loading option file!" << endl;
        break;
      }
      if (strLine.empty())
        continue;
      vecLine.push_back(strLine);
    }
    inf.close();
    setOptions(vecLine);
  }


  void writeModel(LStream &outf) {


    WriteVector(outf, testFiles);
    WriteString(outf, outBest);

    WriteBinary(outf, wordCutOff);
    WriteBinary(outf, featCutOff);
    WriteBinary(outf, charCutOff);
    WriteBinary(outf, tagCutOff);
    WriteBinary(outf, initRange);
    WriteBinary(outf, maxIter);
    WriteBinary(outf, batchSize);
    WriteBinary(outf, adaEps);
    WriteBinary(outf, adaAlpha);
    WriteBinary(outf, regParameter);
    WriteBinary(outf, dropProb);
    WriteBinary(outf, linearHiddenSize);
    WriteBinary(outf, hiddenSize);
    WriteBinary(outf, rnnHiddenSize);
    WriteBinary(outf, wordEmbSize);
    WriteBinary(outf, wordcontext);
    WriteBinary(outf, wordEmbFineTune);
    WriteBinary(outf, tagEmbSize);
    WriteBinary(outf, tagEmbFineTune);
    WriteBinary(outf, charEmbSize);
    WriteBinary(outf, charcontext);
    WriteBinary(outf, charEmbFineTune);
    WriteBinary(outf, charhiddenSize);
    WriteBinary(outf, verboseIter);
    WriteBinary(outf, saveIntermediate);
    WriteBinary(outf, train);
    WriteBinary(outf, maxInstance);
    WriteBinary(outf, seg);
    WriteBinary(outf, relu);
    WriteBinary(outf, atomLayers);
    WriteBinary(outf, rnnLayers);

  }

  void loadModel(LStream &inf) {
    ReadVector(inf, testFiles);
    ReadString(inf, outBest);

    ReadBinary(inf, wordCutOff);
    ReadBinary(inf, featCutOff);
    ReadBinary(inf, charCutOff);
    ReadBinary(inf, tagCutOff);
    ReadBinary(inf, initRange);
    ReadBinary(inf, maxIter);
    ReadBinary(inf, batchSize);
    ReadBinary(inf, adaEps);
    ReadBinary(inf, adaAlpha);
    ReadBinary(inf, regParameter);
    ReadBinary(inf, dropProb);
    ReadBinary(inf, linearHiddenSize);
    ReadBinary(inf, hiddenSize);
    ReadBinary(inf, rnnHiddenSize);
    ReadBinary(inf, wordEmbSize);
    ReadBinary(inf, wordcontext);
    ReadBinary(inf, wordEmbFineTune);
    ReadBinary(inf, tagEmbSize);
    ReadBinary(inf, tagEmbFineTune);
    ReadBinary(inf, charEmbSize);
    ReadBinary(inf, charcontext);
    ReadBinary(inf, charEmbFineTune);
    ReadBinary(inf, charhiddenSize);
    ReadBinary(inf, verboseIter);
    ReadBinary(inf, saveIntermediate);
    ReadBinary(inf, train);
    ReadBinary(inf, maxInstance);
    ReadBinary(inf, seg);
    ReadBinary(inf, relu);
    ReadBinary(inf, atomLayers);
    ReadBinary(inf, rnnLayers);
  }


};

#endif

