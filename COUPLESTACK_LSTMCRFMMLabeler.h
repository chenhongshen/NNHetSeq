/*
 * Labeler.h
 *
 *  Created on: May 4, 2016
 *      Author: hongshen chen based on mszhang
 */

#ifndef SRC_NNCRF_H_
#define SRC_NNCRF_H_


#include "N3L.h"
#include "basic/COUPLESTACK_LSTMCRFMMClassifier.h"
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
  Alphabet m_labelAlphabet_c1;
  Alphabet m_labelAlphabet_c2;

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
  COUPLESTACK_LSTMCRFMMClassifier<gpu> m_classifier;
#else
  COUPLESTACK_LSTMCRFMMClassifier<cpu> m_classifier;
#endif

public:
  void readWordEmbeddings(const string& inFile, NRMat<dtype>& wordEmb);

public:
  Labeler();
  virtual ~Labeler();

public:

  int createAlphabet(const vector<Instance>& c1_vecInsts,const vector<Instance>& c2_vecInsts);

  int addTestWordAlpha(const vector<Instance>& vecInsts);
  int addTestCharAlpha(const vector<Instance>& vecInsts);
  int addTestTagAlpha(const vector<Instance>& vecInsts);

  void extractLinearFeatures(vector<string>& features, const Instance* pInstance, int idx);
  void extractFeature(Feature& feat, const Instance* pInstance, int idx);

  void convert2Example(const Instance* pInstance, Example& exam, const int corpus_type);
  void initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams, const int corpus_type);

  void random_select_train(vector<Example>& c1_Exams,vector<Example>& c2_Exams,
vector<Example>& c1_base_Exams,vector<Example>& c2_base_Exams,
vector<pair<int,vector<Example> > >&type_Exams,
vector<pair<int,vector<Example> > >&base_type_Exams);

  void base_extractFeature(Feature& feat, const Instance* pInstance, int idx);
  void base_convert2Example(const Instance* pInstance, Example& exam, const int corpus_type);
  void base_initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams, const int corpus_type);
public:
  void train(const string& c1_trainFile, const string& c1_devFile, const string& c1_testFile,
    const string& c2_trainFile, const string& c2_devFile, const string& c2_testFile,
    const string& modelFile, const string& optionFile, const string& wordEmbFile, const string& charEmbFile, const string& baseModelFile);
  int predict(const vector<Feature>& features, const vector<Feature>& base_features, vector<string>& outputs, const vector<string>& words, const int corpus_type);
  void test(const string& testFile, const string& outputFile, const string& modelFile, const int corpus_type);

  void writeModelFile(const string& outputModelFile);
  void loadModelFile(const string& inputModelFile);
  void loadBaseModelFile(const string& inputModelFile);

};

#endif /* SRC_NNCRF_H_ */
