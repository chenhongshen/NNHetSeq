/*
 * Labeler.h
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#ifndef SRC_NNCRF_H_
#define SRC_NNCRF_H_


#include "N3L.h"
#include "basic/STACK_LSTMCRFMMClassifier.h"
#include "Options.h"
#include "Instance.h"
#include "Example.h"


#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Labeler {

public:
  std::string nullkey;
  std::string unknownkey;
  std::string seperateKey;

public:
  Alphabet m_featAlphabet;
  Alphabet m_labelAlphabet;

  Alphabet m_wordAlphabet;
  Alphabet m_charAlphabet;

  NRVec<Alphabet> m_tagAlphabets;

  Alphabet m_base_featAlphabet;
  Alphabet m_base_labelAlphabet;

  Alphabet m_base_wordAlphabet;
  Alphabet m_base_charAlphabet;

  NRVec<Alphabet> m_base_tagAlphabets;
public:
  Options m_options;
  Options m_base_options;

  Pipe m_pipe;

#if USE_CUDA==1
  STACK_LSTMCRFMMClassifier<gpu> m_classifier;
#else
  STACK_LSTMCRFMMClassifier<cpu> m_classifier;
#endif

public:
  void readWordEmbeddings(const string& inFile, NRMat<dtype>& wordEmb);

public:
  Labeler();
  virtual ~Labeler();

public:

  int createAlphabet(const vector<Instance>& vecInsts);

  int addTestWordAlpha(const vector<Instance>& vecInsts);
  int addTestCharAlpha(const vector<Instance>& vecInsts);
  int addTestTagAlpha(const vector<Instance>& vecInsts);

  void extractLinearFeatures(vector<string>& features, const Instance* pInstance, int idx);
  void extractFeature(Feature& feat, const Instance* pInstance, int idx);

  void convert2Example(const Instance* pInstance, Example& exam);
  void initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams);

  void base_extractFeature(Feature& feat, const Instance* pInstance, int idx);
  void base_convert2Example(const Instance* pInstance, Example& exam);
  void base_initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams);
public:
  void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile, const string& wordEmbFile, const string& charEmbFile, const string& baseModelFile);
  int predict(const vector<Feature>& features, const vector<Feature>& base_features, vector<string>& outputs, const vector<string>& words);
  void test(const string& testFile, const string& outputFile, const string& modelFile);

  void writeModelFile(const string& outputModelFile);
  void loadModelFile(const string& inputModelFile);
  void loadBaseModelFile(const string& inputModelFile);

};

#endif /* SRC_NNCRF_H_ */
