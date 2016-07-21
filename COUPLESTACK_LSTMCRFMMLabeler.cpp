/*
 * Labeler.cpp
 *
 *  Created on: May 4, 2016
 *      Author: hongshen chen based on mszhang
 */

#include "COUPLESTACK_LSTMCRFMMLabeler.h"

#include "Argument_helper.h"

Labeler::Labeler() {
  // TODO Auto-generated constructor stub
  nullkey = "-null-";
  unknownkey = "-unknown-";
  seperateKey = "#";

}

Labeler::~Labeler() {
  // TODO Auto-generated destructor stub
  m_classifier.release();
}

int Labeler::createAlphabet(const vector<Instance>& c1_vecInsts,const vector<Instance>& c2_vecInsts) {
  cout << "Creating Alphabet..." << endl;

  int numInstance;
  hash_map<string, int> feature_stat;
  hash_map<string, int> word_stat;
  hash_map<string, int> char_stat;
  vector<hash_map<string, int> > tag_stat;


  // tag num
  int c1_tagNum=c1_vecInsts[0].tagfeatures[0].size();
  int c2_tagNum=c2_vecInsts[0].tagfeatures[0].size();
  int tagNum = c1_tagNum>c2_tagNum?c1_tagNum:c2_tagNum;
  tag_stat.resize(tagNum);
  m_tagAlphabets.resize(tagNum);

  m_labelAlphabet_c1.clear();
  for (numInstance = 0; numInstance < c1_vecInsts.size(); numInstance++) {
    const Instance *pInstance = &c1_vecInsts[numInstance];

    const vector<string> &words = pInstance->words;
    const vector<string> &labels = pInstance->labels;
    const vector<vector<string> > &sparsefeatures = pInstance->sparsefeatures;
    const vector<vector<string> > &charfeatures = pInstance->charfeatures;

    // tag features and check tag numbers
    const vector<vector<string> > &tagfeatures = pInstance->tagfeatures;
    for (int iter_tag = 0; iter_tag < tagfeatures.size(); iter_tag++) {
      assert(tagNum == tagfeatures[iter_tag].size());
    }

    vector<string> features;
    int curInstSize = labels.size();
    int labelId;
    for (int i = 0; i < curInstSize; ++i) {
      labelId = m_labelAlphabet_c1.from_string(labels[i]);

      string curword = normalize_to_lowerwithdigit(words[i]);
      word_stat[curword]++;
      for (int j = 0; j < charfeatures[i].size(); j++)
        char_stat[charfeatures[i][j]]++;
      // tag stat increase
      for (int j = 0; j < tagfeatures[i].size(); j++)
        tag_stat[j][tagfeatures[i][j]]++;
      for (int j = 0; j < sparsefeatures[i].size(); j++)
        feature_stat[sparsefeatures[i][j]]++;
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  cout << "corpus1 num: " << numInstance << " " << endl;
  m_labelAlphabet_c2.clear();

  for (numInstance = 0; numInstance < c2_vecInsts.size(); numInstance++) {
    const Instance *pInstance = &c2_vecInsts[numInstance];

    const vector<string> &words = pInstance->words;
    const vector<string> &labels = pInstance->labels;
    const vector<vector<string> > &sparsefeatures = pInstance->sparsefeatures;
    const vector<vector<string> > &charfeatures = pInstance->charfeatures;

    // tag features and check tag numbers
    const vector<vector<string> > &tagfeatures = pInstance->tagfeatures;
    for (int iter_tag = 0; iter_tag < tagfeatures.size(); iter_tag++) {
      assert(tagNum == tagfeatures[iter_tag].size());
    }

    vector<string> features;
    int curInstSize = labels.size();
    int labelId;
    for (int i = 0; i < curInstSize; ++i) {
      labelId = m_labelAlphabet_c2.from_string(labels[i]);

      string curword = normalize_to_lowerwithdigit(words[i]);
      word_stat[curword]++;
      for (int j = 0; j < charfeatures[i].size(); j++)
        char_stat[charfeatures[i][j]]++;
      // tag stat increase
      for (int j = 0; j < tagfeatures[i].size(); j++)
        tag_stat[j][tagfeatures[i][j]]++;
      for (int j = 0; j < sparsefeatures[i].size(); j++)
        feature_stat[sparsefeatures[i][j]]++;
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }
  cout << endl << "corpus2 num: " << numInstance << " " << endl;

  cout << "Corpus1 Label num: " << m_labelAlphabet_c1.size() << endl;
  cout << "Corpus2 Label num: " << m_labelAlphabet_c2.size() << endl;
  cout << "Total word num: " << word_stat.size() << endl;
  cout << "Total char num: " << char_stat.size() << endl;
  cout << "Total feature num: " << feature_stat.size() << endl;
  // tag print information
  cout << "tag num = " << tagNum << endl;
  for (int iter_tag = 0; iter_tag < tagNum; iter_tag++) {
    cout << "Total tag " << iter_tag << " num: " << tag_stat[iter_tag].size() << endl;
  }

  m_featAlphabet.clear();
  m_wordAlphabet.clear();
  m_wordAlphabet.from_string(nullkey);
  m_wordAlphabet.from_string(unknownkey);
  m_charAlphabet.clear();
  m_charAlphabet.from_string(nullkey);
  m_charAlphabet.from_string(unknownkey);
  //tag apheabet init
  for (int i = 0; i < tagNum; i++) {
    m_tagAlphabets[i].clear();
    m_tagAlphabets[i].from_string(nullkey);
    m_tagAlphabets[i].from_string(unknownkey);
  }

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = feature_stat.begin(); feat_iter != feature_stat.end(); feat_iter++) {
    if (feat_iter->second > m_options.featCutOff) {
      m_featAlphabet.from_string(feat_iter->first);
    }
  }

  for (feat_iter = word_stat.begin(); feat_iter != word_stat.end(); feat_iter++) {
    if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
      m_wordAlphabet.from_string(feat_iter->first);
    }
  }

  for (feat_iter = char_stat.begin(); feat_iter != char_stat.end(); feat_iter++) {
    if (!m_options.charEmbFineTune || feat_iter->second > m_options.charCutOff) {
      m_charAlphabet.from_string(feat_iter->first);
    }
  }

  // tag cut off, default tagCutOff is zero
  for (int i = 0; i < tagNum; i++) {
    for (feat_iter = tag_stat[i].begin(); feat_iter != tag_stat[i].end(); feat_iter++) {
      if (!m_options.tagEmbFineTune || feat_iter->second > m_options.tagCutOff) {
        m_tagAlphabets[i].from_string(feat_iter->first);
      }
    }
  }

  cout << "Remain feature num: " << m_featAlphabet.size() << endl;
  cout << "Remain words num: " << m_wordAlphabet.size() << endl;
  cout << "Remain char num: " << m_charAlphabet.size() << endl;
  // tag Remain num print
  for (int i = 0; i < tagNum; i++) {
    cout << "Remain tag " << i << " num: " << m_tagAlphabets[i].size() << endl;
  }

  m_labelAlphabet_c1.set_fixed_flag(true);
  m_labelAlphabet_c2.set_fixed_flag(true);
  m_featAlphabet.set_fixed_flag(true);
  m_wordAlphabet.set_fixed_flag(true);
  m_charAlphabet.set_fixed_flag(true);
  // tag Alphabet fixed  
  for (int iter_tag = 0; iter_tag < tagNum; iter_tag++) {
    m_tagAlphabets[iter_tag].set_fixed_flag(true);
  }

  return 0;
}

int Labeler::addTestWordAlpha(const vector<Instance>& vecInsts) {
  cout << "Adding word Alphabet..." << endl;

  int numInstance;
  hash_map<string, int> word_stat;
  m_wordAlphabet.set_fixed_flag(false);

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    const vector<string> &words = pInstance->words;

    int curInstSize = words.size();
    for (int i = 0; i < curInstSize; ++i) {
      string curword = normalize_to_lowerwithdigit(words[i]);
      word_stat[curword]++;
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = word_stat.begin(); feat_iter != word_stat.end(); feat_iter++) {
    if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
      m_wordAlphabet.from_string(feat_iter->first);
    }
  }

  m_wordAlphabet.set_fixed_flag(true);

  return 0;
}

int Labeler::addTestCharAlpha(const vector<Instance>& vecInsts) {
  cout << "Adding char Alphabet..." << endl;

  int numInstance;
  hash_map<string, int> char_stat;
  m_charAlphabet.set_fixed_flag(false);

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    const vector<vector<string> > &charfeatures = pInstance->charfeatures;

    int curInstSize = charfeatures.size();
    for (int i = 0; i < curInstSize; ++i) {
      for (int j = 1; j < charfeatures[i].size(); j++)
        char_stat[charfeatures[i][j]]++;
    }
    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = char_stat.begin(); feat_iter != char_stat.end(); feat_iter++) {
    if (!m_options.charEmbFineTune || feat_iter->second > m_options.charCutOff) {
      m_charAlphabet.from_string(feat_iter->first);
    }
  }

  m_charAlphabet.set_fixed_flag(true);

  return 0;
}

// tag AddTestTagAlpha
int Labeler::addTestTagAlpha(const vector<Instance>& vecInsts) {
  cout << "Adding tag Alphabet..." << endl;

  int numInstance;
  int tagNum = vecInsts[0].tagfeatures[0].size();
  vector<hash_map<string, int> > tag_stat(tagNum);
  for (int i = 0; i < tagNum; i++) {
    m_tagAlphabets[i].set_fixed_flag(false);
  }

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    //const vector<vector<string> > &charfeatures = pInstance->charfeatures;
    const vector<vector<string> > &tagfeatures = pInstance->tagfeatures;
    for (int iter_tag = 0; iter_tag < tagfeatures.size(); iter_tag++) {
      assert(tagNum == tagfeatures[iter_tag].size());
    }

    int curInstSize = tagfeatures.size();
    for (int i = 0; i < curInstSize; ++i) {
      for (int j = 1; j < tagfeatures[i].size(); j++)
        tag_stat[j][tagfeatures[i][j]]++;
    }
    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  hash_map<string, int>::iterator feat_iter;
  for (int i = 0; i < tagNum; i++) {
    for (feat_iter = tag_stat[i].begin(); feat_iter != tag_stat[i].end(); feat_iter++) {
      if (!m_options.tagEmbFineTune || feat_iter->second > m_options.tagCutOff) {
        m_tagAlphabets[i].from_string(feat_iter->first);
      }
    }
  }

  for (int i = 0; i < tagNum; i++) {
    m_tagAlphabets[i].set_fixed_flag(true);
  }

  return tagNum;
}

void Labeler::extractFeature(Feature& feat, const Instance* pInstance, int idx) {
  feat.clear();

  const vector<string>& words = pInstance->words;
  int sentsize = words.size();
  string curWord = idx >= 0 && idx < sentsize ? normalize_to_lowerwithdigit(words[idx]) : nullkey;

  // word features
  int unknownId = m_wordAlphabet.from_string(unknownkey);

  int curWordId = m_wordAlphabet.from_string(curWord);
  if (curWordId >= 0)
    feat.words.push_back(curWordId);
  else
    feat.words.push_back(unknownId);

  // tag features
  const vector<vector<string> > &tagfeatures = pInstance->tagfeatures;
  int tagNum = tagfeatures[idx].size();
  for (int i = 0; i < tagNum; i++) {
    unknownId = m_tagAlphabets[i].from_string(unknownkey);
    int curTagId = m_tagAlphabets[i].from_string(tagfeatures[idx][i]);
    if (curTagId >= 0)
      feat.tags.push_back(curTagId);
    else
      feat.tags.push_back(unknownId);
  }

  // char features
  unknownId = m_charAlphabet.from_string(unknownkey);

  const vector<vector<string> > &charfeatures = pInstance->charfeatures;

  const vector<string>& cur_chars = charfeatures[idx];
  int cur_char_size = cur_chars.size();

  // actually we support a max window of m_options.charcontext = 2
  for (int i = 0; i < cur_char_size; i++) {
    string curChar = cur_chars[i];

    int curCharId = m_charAlphabet.from_string(curChar);
    if (curCharId >= 0)
      feat.chars.push_back(curCharId);
    else
      feat.chars.push_back(unknownId);
  }

  int nullkeyId = m_charAlphabet.from_string(nullkey);
  if (feat.chars.empty()) {
    feat.chars.push_back(nullkeyId);
  }

  const vector<string>& linear_features = pInstance->sparsefeatures[idx];
  for (int i = 0; i < linear_features.size(); i++) {
    int curFeatId = m_featAlphabet.from_string(linear_features[i]);
    if (curFeatId >= 0)
      feat.linear_features.push_back(curFeatId);
  }

}

void Labeler::convert2Example(const Instance* pInstance, Example& exam, const int corpus_type) {
  exam.clear();
  Alphabet& m_labelAlphabet=(corpus_type==1?m_labelAlphabet_c1:m_labelAlphabet_c2);
  const vector<string> &labels = pInstance->labels;
  int curInstSize = labels.size();
  for (int i = 0; i < curInstSize; ++i) {
    string orcale = labels[i];

    int numLabel1s = m_labelAlphabet.size();
    vector<int> curlabels, curlabel2s;
    for (int j = 0; j < numLabel1s; ++j) {
      string str = m_labelAlphabet.from_id(j);
      if (str.compare(orcale) == 0)
        curlabels.push_back(1);
      else
        curlabels.push_back(0);
    }

    exam.m_labels.push_back(curlabels);
    Feature feat;
    extractFeature(feat, pInstance, i);
    exam.m_features.push_back(feat);
  }
}

void Labeler::initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams, const int corpus_type) {
  int numInstance;
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];
    Example curExam;
    convert2Example(pInstance, curExam, corpus_type);
    vecExams.push_back(curExam);

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  cout << numInstance << " " << endl;
}

void Labeler::base_extractFeature(Feature& feat, const Instance* pInstance, int idx) {
  feat.clear();

  const vector<string>& words = pInstance->words;
  int sentsize = words.size();
  string curWord = idx >= 0 && idx < sentsize ? normalize_to_lowerwithdigit(words[idx]) : nullkey;

  // word features
  int unknownId = m_base_wordAlphabet.from_string(unknownkey);

  int curWordId = m_base_wordAlphabet.from_string(curWord);
  if (curWordId >= 0)
    feat.words.push_back(curWordId);
  else
    feat.words.push_back(unknownId);

  // tag features
  const vector<vector<string> > &tagfeatures = pInstance->tagfeatures;
  int tagNum = tagfeatures[idx].size();
  for (int i = 0; i < tagNum; i++) {
    unknownId = m_base_tagAlphabets[i].from_string(unknownkey);
    int curTagId = m_base_tagAlphabets[i].from_string(tagfeatures[idx][i]);
    if (curTagId >= 0)
      feat.tags.push_back(curTagId);
    else
      feat.tags.push_back(unknownId);
  }

  // char features
  unknownId = m_base_charAlphabet.from_string(unknownkey);

  const vector<vector<string> > &charfeatures = pInstance->charfeatures;

  const vector<string>& cur_chars = charfeatures[idx];
  int cur_char_size = cur_chars.size();

  // actually we support a max window of m_options.charcontext = 2
  for (int i = 0; i < cur_char_size; i++) {
    string curChar = cur_chars[i];

    int curCharId = m_base_charAlphabet.from_string(curChar);
    if (curCharId >= 0)
      feat.chars.push_back(curCharId);
    else
      feat.chars.push_back(unknownId);
  }

  int nullkeyId = m_base_charAlphabet.from_string(nullkey);
  if (feat.chars.empty()) {
    feat.chars.push_back(nullkeyId);
  }

  const vector<string>& linear_features = pInstance->sparsefeatures[idx];
  for (int i = 0; i < linear_features.size(); i++) {
    int curFeatId = m_base_featAlphabet.from_string(linear_features[i]);
    if (curFeatId >= 0)
      feat.linear_features.push_back(curFeatId);
  }

}
void Labeler::base_convert2Example(const Instance* pInstance, Example& exam, int corpus_type) {
  exam.clear();
  Alphabet& m_base_labelAlphabet=(corpus_type==1?m_labelAlphabet_c1:m_labelAlphabet_c2);
  const vector<string> &labels = pInstance->labels;
  int curInstSize = labels.size();
  for (int i = 0; i < curInstSize; ++i) {
    string orcale = labels[i];

    int numLabel1s = m_base_labelAlphabet.size();
    vector<int> curlabels, curlabel2s;
    for (int j = 0; j < numLabel1s; ++j) {
      string str = m_base_labelAlphabet.from_id(j);
      if (str.compare(orcale) == 0)
        curlabels.push_back(1);
      else
        curlabels.push_back(0);
    }

    exam.m_labels.push_back(curlabels);
    Feature feat;
    base_extractFeature(feat, pInstance, i);
    exam.m_features.push_back(feat);
  }
}

void Labeler::base_initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams, int corpus_type) {
  int numInstance;
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];
    Example curExam;
    base_convert2Example(pInstance, curExam, corpus_type);
    vecExams.push_back(curExam);

    if ((numInstance + 1) % m_base_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_base_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_base_options.maxInstance > 0 && numInstance == m_base_options.maxInstance)
      break;
  }

  cout << numInstance << " " << endl;
}
void Labeler::random_select_train(vector<Example>& c1_Exams,vector<Example>& c2_Exams,
vector<Example>& c1_base_Exams,vector<Example>& c2_base_Exams,
vector<pair<int,vector<Example> > >&type_Exams,
vector<pair<int,vector<Example> > >&base_type_Exams){
  std::vector<int> c1_indexes, c2_indexes;
  int c1_inputSize = c1_Exams.size();
  for (int i = 0; i < c1_inputSize; ++i)
    c1_indexes.push_back(i);
  int c2_inputSize = c2_Exams.size();
  for (int i = 0; i < c2_inputSize; ++i)
    c2_indexes.push_back(i);
  int partnum=c1_inputSize<c2_inputSize?c1_inputSize:c2_inputSize;
  random_shuffle(c1_indexes.begin(), c1_indexes.end());
  random_shuffle(c2_indexes.begin(), c2_indexes.end());
  vector<Example> cur_batch_exams,cur_batch_base_exams;
  for (int i=0; i<partnum; ++i){
    cur_batch_exams.push_back(c1_Exams[c1_indexes[i]]);
    cur_batch_base_exams.push_back(c1_base_Exams[c1_indexes[i]]);
    if (0==(i+1)%m_options.batchSize){
      type_Exams.push_back(make_pair(1,cur_batch_exams));
      base_type_Exams.push_back(make_pair(1,cur_batch_base_exams));
      cur_batch_exams.clear();
      cur_batch_base_exams.clear();
    }
  }
  if (!cur_batch_exams.empty()){
    type_Exams.push_back(make_pair(1,cur_batch_exams));
    base_type_Exams.push_back(make_pair(1,cur_batch_base_exams));
    cur_batch_exams.clear();
    cur_batch_base_exams.clear();
  }

  for (int i=0; i<partnum; ++i){
    cur_batch_exams.push_back(c2_Exams[c2_indexes[i]]);
    cur_batch_base_exams.push_back(c2_base_Exams[c2_indexes[i]]);
    if (0==(i+1)%m_options.batchSize){
      type_Exams.push_back(make_pair(2,cur_batch_exams));
      base_type_Exams.push_back(make_pair(2,cur_batch_base_exams));
      cur_batch_exams.clear();
      cur_batch_base_exams.clear();
    }
  }
  if (!cur_batch_exams.empty()){
    type_Exams.push_back(make_pair(2,cur_batch_exams));
    base_type_Exams.push_back(make_pair(2,cur_batch_base_exams));
    cur_batch_exams.clear();
    cur_batch_base_exams.clear();
  }
}
void Labeler::train(const string& c1_trainFile, const string& c1_devFile, const string& c1_testFile,
    const string& c2_trainFile, const string& c2_devFile, const string& c2_testFile,
    const string& modelFile, const string& optionFile,
    const string& wordEmbFile, const string& charEmbFile, const string& basemodelFile) {
  loadBaseModelFile(basemodelFile);
  if (optionFile != "")
    m_options.load(optionFile);
  m_options.showOptions();
  vector<Instance> c1_trainInsts, c1_devInsts, c1_testInsts, c2_trainInsts, c2_devInsts, c2_testInsts;
  static vector<Instance> decodeInstResults;
  static Instance curDecodeInst;
  bool c1_bCurIterBetter = false, c2_bCurIterBetter = false;

  m_pipe.readInstances(c1_trainFile, c1_trainInsts, m_options.maxInstance);
  if (c1_devFile != "")
    m_pipe.readInstances(c1_devFile, c1_devInsts, m_options.maxInstance);
  if (c1_testFile != "")
    m_pipe.readInstances(c1_testFile, c1_testInsts, m_options.maxInstance);
  m_pipe.readInstances(c2_trainFile, c2_trainInsts, m_options.maxInstance);
  if (c2_devFile != "")
    m_pipe.readInstances(c2_devFile, c2_devInsts, m_options.maxInstance);
  if (c2_testFile != "")
    m_pipe.readInstances(c2_testFile, c2_testInsts, m_options.maxInstance);

  //Ensure that each file in m_options.testFiles exists!
  vector<vector<Instance> > otherInsts(m_options.testFiles.size());
  for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
    m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_options.maxInstance);
  }

  //std::cout << "Training example number: " << trainInsts.size() << std::endl;
  //std::cout << "Dev example number: " << trainInsts.size() << std::endl;
  //std::cout << "Test example number: " << trainInsts.size() << std::endl;

  createAlphabet(c1_trainInsts,c2_trainInsts);

  if (!m_options.wordEmbFineTune) {
    addTestWordAlpha(c1_devInsts);
    addTestWordAlpha(c1_testInsts);
    addTestWordAlpha(c2_devInsts);
    addTestWordAlpha(c2_testInsts);
    for (int idx = 0; idx < otherInsts.size(); idx++) {
      addTestWordAlpha(otherInsts[idx]);
    }
    cout << "Remain words num: " << m_wordAlphabet.size() << endl;
  }

  NRMat<dtype> wordEmb;
  if (wordEmbFile != "") {
    readWordEmbeddings(wordEmbFile, wordEmb);
  } else {
    wordEmb.resize(m_wordAlphabet.size(), m_options.wordEmbSize);
    wordEmb.randu(1000);
  }

  NRMat<dtype> charEmb;
  if (charEmbFile != "") {
    readWordEmbeddings(charEmbFile, charEmb);
  } else {
    charEmb.resize(m_charAlphabet.size(), m_options.charEmbSize);
    charEmb.randu(1001);
  }

  NRVec<NRMat<dtype> > tagEmbs(m_tagAlphabets.size());
  for (int idx = 0; idx < tagEmbs.size(); idx++) {
    tagEmbs[idx].resize(m_tagAlphabets[idx].size(), m_options.tagEmbSize);
    tagEmbs[idx].randu(1002 + idx);
  }

  
  m_classifier.init(wordEmb, m_options.wordcontext, charEmb, m_options.charcontext, tagEmbs,
    m_labelAlphabet_c1.size(), m_labelAlphabet_c2.size(), m_options.charhiddenSize, m_options.rnnHiddenSize, m_options.hiddenSize);
  m_classifier.setWordEmbFinetune(m_options.wordEmbFineTune);
  m_classifier.setTagEmbFinetune(m_options.tagEmbFineTune);
  m_classifier.setDropValue(m_options.dropProb);

  vector<Example> c1_trainExamples, c1_devExamples, c1_testExamples, c2_trainExamples, c2_devExamples, c2_testExamples;
  initialExamples(c1_trainInsts, c1_trainExamples, 1);
  initialExamples(c1_devInsts, c1_devExamples, 1);
  initialExamples(c1_testInsts, c1_testExamples, 1);
  initialExamples(c2_trainInsts, c2_trainExamples, 2);
  initialExamples(c2_devInsts, c2_devExamples, 2);
  initialExamples(c2_testInsts, c2_testExamples, 2);

//  vector<int> otherInstNums(otherInsts.size());
//  vector<vector<Example> > otherExamples(otherInsts.size());
//  for (int idx = 0; idx < otherInsts.size(); idx++) {
//    initialExamples(otherInsts[idx], otherExamples[idx]);
//    otherInstNums[idx] = otherExamples[idx].size();
//  }

  vector<Example> c1_base_trainExamples, c1_base_devExamples, c1_base_testExamples,c2_base_trainExamples, c2_base_devExamples, c2_base_testExamples;
  //vector<int> c1_base_otherInstNums(c1_otherInsts.size());
  //vector<int> c2_base_otherInstNums(c2_otherInsts.size());
  //vector<vector<Example> > c1_base_otherExamples(c1_otherInsts.size());
  //vector<vector<Example> > c2_base_otherExamples(c2_otherInsts.size());

  base_initialExamples(c1_trainInsts, c1_base_trainExamples, 1);
  base_initialExamples(c1_devInsts, c1_base_devExamples, 1);
  base_initialExamples(c1_testInsts, c1_base_testExamples, 1);
  base_initialExamples(c2_trainInsts, c2_base_trainExamples, 2);
  base_initialExamples(c2_devInsts, c2_base_devExamples, 2);
  base_initialExamples(c2_testInsts, c2_base_testExamples, 2);
  assert(c1_trainExamples.size()==c1_base_trainExamples.size());
  assert(c1_devExamples.size()==c1_base_devExamples.size());
  assert(c1_testExamples.size()==c1_base_testExamples.size());
  assert(c2_trainExamples.size()==c2_base_trainExamples.size());
  assert(c2_devExamples.size()==c2_base_devExamples.size());
  assert(c2_testExamples.size()==c2_base_testExamples.size());
  //for (int idx = 0; idx < otherInsts.size(); idx++) {
  //  base_initialExamples(otherInsts[idx], base_otherExamples[idx]);
  //  base_otherInstNums[idx] = base_otherExamples[idx].size();
  //}

  dtype c1_bestDIS = 0,c2_bestDIS = 0;

  srand(0);
  static Metric eval;
  static Metric c1_metric_dev, c1_metric_test;
  int c1_devNum = c1_devExamples.size(), c1_testNum = c1_testExamples.size();
  static Metric c2_metric_dev, c2_metric_test;
  int c2_devNum = c2_devExamples.size(), c2_testNum = c2_testExamples.size();

  //static vector<Example> subExamples;
  //static vector<Example> base_subExamples;
  //int devNum = devExamples.size(), testNum = testExamples.size();
  for (int iter = 0; iter < m_options.maxIter; ++iter) {
    std::cout << "##### Iteration " << iter << std::endl;
    vector<pair<int,vector<Example> > > type_trainExamples;
    vector<pair<int,vector<Example> > > base_type_trainExamples;
    random_select_train(c1_trainExamples,c2_trainExamples,
      c1_base_trainExamples,c2_base_trainExamples,
      type_trainExamples,base_type_trainExamples);
    assert(type_trainExamples.size()==base_type_trainExamples.size());
    int inputSize = type_trainExamples.size();
    std::vector<int> indexes;
    for (int i = 0; i < inputSize; ++i)
      indexes.push_back(i);
    random_shuffle(indexes.begin(), indexes.end());
    eval.reset();
    for (int updateIter = 0; updateIter < inputSize; updateIter++) {
      pair<int,vector<Example> >& sub_type_exmas=type_trainExamples[indexes[updateIter]];
      pair<int,vector<Example> >& base_sub_type_exmas=base_type_trainExamples[indexes[updateIter]];
      //subExamples.clear();base_subExamples.clear();
      int cur_type=sub_type_exmas.first; int base_cur_type=base_sub_type_exmas.first;
      assert(cur_type==base_cur_type);
      vector<Example>& subExamples=sub_type_exmas.second;
      vector<Example>& base_subExamples=base_sub_type_exmas.second;
      int curUpdateIter = iter * inputSize + updateIter;
      dtype cost = m_classifier.process(subExamples, base_subExamples, curUpdateIter, cur_type);
      eval.overall_label_count += m_classifier._eval.overall_label_count;
      eval.correct_label_count += m_classifier._eval.correct_label_count;

      if ((curUpdateIter + 1) % m_options.verboseIter == 0) {
        //m_classifier.checkgrads(subExamples, base_subExamples, curUpdateIter+1);

        std::cout << "current: " << updateIter + 1 << ", total block: " << inputSize << std::endl;
        std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
      }
      m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps, cur_type);

    }

    if (c1_devNum > 0 && c2_devNum > 0) {
      c1_bCurIterBetter = false;c2_bCurIterBetter = false;
      if (!m_options.outBest.empty())
        decodeInstResults.clear();
      c1_metric_dev.reset();
      for (int idx = 0; idx < c1_devExamples.size(); idx++) {
        vector<string> result_labels;
        predict(c1_devExamples[idx].m_features, c1_base_devExamples[idx].m_features, result_labels, c1_devInsts[idx].words, 1);

        if (m_options.seg)
          c1_devInsts[idx].SegEvaluate(result_labels, c1_metric_dev);
        else
          c1_devInsts[idx].Evaluate(result_labels, c1_metric_dev);

        if (!m_options.outBest.empty()) {
          curDecodeInst.copyValuesFrom(c1_devInsts[idx]);
          curDecodeInst.assignLabel(result_labels);
          decodeInstResults.push_back(curDecodeInst);
        }
      }

      std::cout<<"corpu1 dev:"<<std::endl;
      c1_metric_dev.print();

      if (!m_options.outBest.empty() && c1_metric_dev.getAccuracy() > c1_bestDIS) {
        m_pipe.outputAllInstances(c1_devFile + m_options.outBest, decodeInstResults);
        c1_bCurIterBetter = true;
      }

      if (c1_testNum > 0) {
        if (!m_options.outBest.empty())
          decodeInstResults.clear();
        c1_metric_test.reset();
        for (int idx = 0; idx < c1_testExamples.size(); idx++) {
          vector<string> result_labels;
          predict(c1_testExamples[idx].m_features, c1_base_testExamples[idx].m_features, result_labels, c1_testInsts[idx].words, 1);

          if (m_options.seg)
            c1_testInsts[idx].SegEvaluate(result_labels, c1_metric_test);
          else
            c1_testInsts[idx].Evaluate(result_labels, c1_metric_test);

          if (c1_bCurIterBetter && !m_options.outBest.empty()) {
            curDecodeInst.copyValuesFrom(c1_testInsts[idx]);
            curDecodeInst.assignLabel(result_labels);
            decodeInstResults.push_back(curDecodeInst);
          }
        }
        std::cout << "corpus1 test:" << std::endl;
        c1_metric_test.print();

        if (!m_options.outBest.empty() && c1_bCurIterBetter) {
          m_pipe.outputAllInstances(c1_testFile + m_options.outBest, decodeInstResults);
        }
      }
      if (c2_devNum > 0) {
        c2_bCurIterBetter = false;
        if (!m_options.outBest.empty())
          decodeInstResults.clear();
        c2_metric_dev.reset();
        for (int idx = 0; idx < c2_devExamples.size(); idx++) {
          vector<string> result_labels;
          predict(c2_devExamples[idx].m_features, c2_base_devExamples[idx].m_features, result_labels, c2_devInsts[idx].words, 2);

          if (m_options.seg)
            c2_devInsts[idx].SegEvaluate(result_labels, c2_metric_dev);
          else
            c2_devInsts[idx].Evaluate(result_labels, c2_metric_dev);

          if (!m_options.outBest.empty()) {
            curDecodeInst.copyValuesFrom(c2_devInsts[idx]);
            curDecodeInst.assignLabel(result_labels);
            decodeInstResults.push_back(curDecodeInst);
          }
        }
      }
      std::cout<<"corpu2 dev:"<<std::endl;
      c2_metric_dev.print();

      if (!m_options.outBest.empty() && c2_metric_dev.getAccuracy() > c2_bestDIS) {
        m_pipe.outputAllInstances(c2_devFile + m_options.outBest, decodeInstResults);
        c2_bCurIterBetter = true;
      }
      if (c2_testNum > 0) {
        if (!m_options.outBest.empty())
          decodeInstResults.clear();
        c2_metric_test.reset();
        for (int idx = 0; idx < c2_testExamples.size(); idx++) {
          vector<string> result_labels;
          predict(c2_testExamples[idx].m_features, c2_base_testExamples[idx].m_features, result_labels, c2_testInsts[idx].words, 2);

          if (m_options.seg)
            c2_testInsts[idx].SegEvaluate(result_labels, c2_metric_test);
          else
            c2_testInsts[idx].Evaluate(result_labels, c2_metric_test);

          if (c2_bCurIterBetter && !m_options.outBest.empty()) {
            curDecodeInst.copyValuesFrom(c2_testInsts[idx]);
            curDecodeInst.assignLabel(result_labels);
            decodeInstResults.push_back(curDecodeInst);
          }
        }
        std::cout << "corpus2 test:" << std::endl;
        c2_metric_test.print();

        if (!m_options.outBest.empty() && c2_bCurIterBetter) {
          m_pipe.outputAllInstances(c2_testFile + m_options.outBest, decodeInstResults);
        }
      }

      if (m_options.saveIntermediate && c1_metric_dev.getAccuracy() > c1_bestDIS) {
        std::cout << "C1 Exceeds best previous performance of " << c1_bestDIS << ". Saving model file.." << std::endl;
        c1_bestDIS = c1_metric_dev.getAccuracy();
        writeModelFile(modelFile+".type1");
      }
      if (m_options.saveIntermediate && c2_metric_dev.getAccuracy() > c2_bestDIS) {
        std::cout << "C2 Exceeds best previous performance of " << c2_bestDIS << ". Saving model file.." << std::endl;
        c2_bestDIS = c2_metric_dev.getAccuracy();
        writeModelFile(modelFile+".type2");
      }
    }
    // Clear gradients
  }
}

int Labeler::predict(const vector<Feature>& features, const vector<Feature>& base_features, vector<string>& outputs, const vector<string>& words, const int corpus_type) {
  assert(features.size() == words.size());
  assert(base_features.size() == words.size());
  vector<int> labelIdx, label2Idx;
  m_classifier.predict(features, base_features, labelIdx, corpus_type);
  outputs.clear();
  Alphabet& _labelAlphabet=(1==corpus_type?m_labelAlphabet_c1:m_labelAlphabet_c2);

  for (int idx = 0; idx < words.size(); idx++) {
    string label = _labelAlphabet.from_id(labelIdx[idx]);
    outputs.push_back(label);
  }

  return 0;
}

void Labeler::test(const string& testFile, const string& outputFile, const string& modelFile, const int corpus_type) {
  loadModelFile(modelFile);
  vector<Instance> testInsts;
  m_pipe.readInstances(testFile, testInsts);

  vector<Example> testExamples;
  initialExamples(testInsts, testExamples, corpus_type);
  vector<Example> base_testExamples;
  base_initialExamples(testInsts, base_testExamples, corpus_type);

  int testNum = testExamples.size();
  vector<Instance> testInstResults;
  Metric metric_test;
  metric_test.reset();
  for (int idx = 0; idx < testExamples.size(); idx++) {
    vector<string> result_labels;
    predict(testExamples[idx].m_features, base_testExamples[idx].m_features, result_labels, testInsts[idx].words, corpus_type);
    if (m_options.seg) {
      testInsts[idx].SegEvaluate(result_labels, metric_test);
    }
    else{
      testInsts[idx].Evaluate(result_labels, metric_test);
    }

    Instance curResultInst;
    curResultInst.copyValuesFrom(testInsts[idx]);
    curResultInst.assignLabel(result_labels);
    testInstResults.push_back(curResultInst);
  }
  
  std::cout << "test:" << std::endl;
  metric_test.print();
  clock_t start, end;

  m_pipe.outputAllInstances(outputFile, testInstResults);


}

void Labeler::readWordEmbeddings(const string& inFile, NRMat<dtype>& wordEmb) {
  static ifstream inf;
  if (inf.is_open()) {
    inf.close();
    inf.clear();
  }
  inf.open(inFile.c_str());

  static string strLine, curWord;
  static int wordId;

  //find the first line, decide the wordDim;
  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (!strLine.empty())
      break;
  }

  int unknownId = m_wordAlphabet.from_string(unknownkey);

  static vector<string> vecInfo;
  split_bychar(strLine, vecInfo, ' ');
  int wordDim = vecInfo.size() - 1;

  std::cout << "word embedding dim is " << wordDim << std::endl;
  m_options.wordEmbSize = wordDim;

  wordEmb.resize(m_wordAlphabet.size(), wordDim);
  wordEmb = 0.0;
  curWord = normalize_to_lowerwithdigit(vecInfo[0]);
  wordId = m_wordAlphabet.from_string(curWord);
  hash_set<int> indexers;
  dtype sum[wordDim];
  int count = 0;
  bool bHasUnknown = false;
  if (wordId >= 0) {
    count++;
    if (unknownId == wordId)
      bHasUnknown = true;
    indexers.insert(wordId);
    for (int idx = 0; idx < wordDim; idx++) {
      dtype curValue = atof(vecInfo[idx + 1].c_str());
      sum[idx] = curValue;
      wordEmb[wordId][idx] = curValue;
    }

  } else {
    for (int idx = 0; idx < wordDim; idx++) {
      sum[idx] = 0.0;
    }
  }

  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (strLine.empty())
      continue;
    split_bychar(strLine, vecInfo, ' ');
    if (vecInfo.size() != wordDim + 1) {
      std::cout << "error embedding file" << std::endl;
    }
    curWord = normalize_to_lowerwithdigit(vecInfo[0]);
    wordId = m_wordAlphabet.from_string(curWord);
    if (wordId >= 0) {
      count++;
      if (unknownId == wordId)
        bHasUnknown = true;
      indexers.insert(wordId);

      for (int idx = 0; idx < wordDim; idx++) {
        dtype curValue = atof(vecInfo[idx + 1].c_str());
        sum[idx] += curValue;
        wordEmb[wordId][idx] += curValue;
      }
    }

  }

  if (!bHasUnknown) {
    for (int idx = 0; idx < wordDim; idx++) {
      wordEmb[unknownId][idx] = sum[idx] / count;
    }
    count++;
    std::cout << unknownkey << " not found, using averaged value to initialize." << std::endl;
  }

  int oovWords = 0;
  int totalWords = 0;
  for (int id = 0; id < m_wordAlphabet.size(); id++) {
    if (indexers.find(id) == indexers.end()) {
      oovWords++;
      for (int idx = 0; idx < wordDim; idx++) {
        wordEmb[id][idx] = wordEmb[unknownId][idx];
      }
    }
    totalWords++;
  }

  std::cout << "OOV num is " << oovWords << ", total num is " << m_wordAlphabet.size() << ", embedding oov ratio is " << oovWords * 1.0 / m_wordAlphabet.size()
      << std::endl;

}
void Labeler::loadBaseModelFile(const string& inputModelFile) {
  std::cout << "Start load basement model from file: " << inputModelFile << std::endl;
  LStream inf(inputModelFile, "rb");
  m_base_options.loadModel(inf);
  m_base_options.showOptions();
  m_base_wordAlphabet.loadModel(inf);
  m_base_charAlphabet.loadModel(inf);
  m_base_labelAlphabet.loadModel(inf);
  m_base_featAlphabet.loadModel(inf);
  m_classifier.loadBaseModel(inf);

  int base_tagAlphabets_size;
  ReadBinary(inf, base_tagAlphabets_size);
  m_base_tagAlphabets.resize(base_tagAlphabets_size);
  for (int idx = 0; idx < base_tagAlphabets_size; idx++) {
    m_base_tagAlphabets[idx].loadModel(inf);
  }

  ReadString(inf, nullkey);
  ReadString(inf, unknownkey);
  ReadString(inf, seperateKey);
  std::cout << "Basement Model has been loaded from file: " << inputModelFile << std::endl;
}

void Labeler::loadModelFile(const string& inputModelFile) {
  std::cout << "Start load model from file: " << inputModelFile << std::endl;

  LStream inf(inputModelFile, "rb");
  m_options.loadModel(inf);
  // m_options.showOptions();
  m_wordAlphabet.loadModel(inf);
  m_charAlphabet.loadModel(inf);
  m_labelAlphabet_c1.loadModel(inf);
  m_labelAlphabet_c2.loadModel(inf);
  m_featAlphabet.loadModel(inf);
  m_classifier.loadModel(inf);

  int m_tagAlphabets_size;
  ReadBinary(inf, m_tagAlphabets_size);
  m_tagAlphabets.resize(m_tagAlphabets_size);
  for (int idx = 0; idx < m_tagAlphabets_size; idx++) {
    m_tagAlphabets[idx].loadModel(inf);
  }

  ReadString(inf, nullkey);
  ReadString(inf, unknownkey);
  ReadString(inf, seperateKey);
  std::cout << "Model has been loaded from file: " << inputModelFile << std::endl;


}

void Labeler::writeModelFile(const string & outputModelFile) {
  std::cout << "Start write model to file: " << outputModelFile << std::endl;
  LStream outf(outputModelFile, "w+");
  m_options.writeModel(outf);
  m_wordAlphabet.writeModel(outf);
  m_charAlphabet.writeModel(outf);
  m_labelAlphabet_c1.writeModel(outf);
  m_labelAlphabet_c2.writeModel(outf);
  m_featAlphabet.writeModel(outf);
  m_classifier.writeModel(outf);

  int m_tagAlphabets_size = m_tagAlphabets.size();
  WriteBinary(outf, m_tagAlphabets_size);
  for (int idx = 0; idx < m_tagAlphabets_size; idx++) {
    m_tagAlphabets[idx].writeModel(outf);
  }

  WriteString(outf, nullkey);
  WriteString(outf, unknownkey);
  WriteString(outf, seperateKey);

  std::cout << "Model has been written in file: " << outputModelFile << std::endl;

}


int main(int argc, char* argv[]) {
#if USE_CUDA==1
  InitTensorEngine();
#else
  InitTensorEngine<cpu>();
#endif

  std::string c1_trainFile = "", c1_devFile = "", c1_testFile = "";
  std::string c2_trainFile = "", c2_devFile = "", c2_testFile = "";
  std::string modelFile = "";std::string basemodelFile = "";
  std::string wordEmbFile = "", charEmbFile = "", optionFile = "";
  std::string outputFile = "";

  bool bTrain = false;
  dsr::Argument_helper ah;

  ah.new_flag("l", "learn", "train or test", bTrain);
  ah.new_named_string("c1_train", "trainCorpus_type1", "named_string", "training corpus of type1 to train a model, must when training", c1_trainFile);
  ah.new_named_string("c1_dev", "devCorpus_type1", "named_string", "development corpus of type1 to train a model, optional when training", c1_devFile);
  ah.new_named_string("c1_test", "testCorpus_type1", "named_string",
      "testing corpus of type1 to train a model or input file to test a model, optional when training and must when testing", c1_testFile);
  ah.new_named_string("c2_train", "trainCorpus_type2", "named_string", "training corpus of type2 to train a model, must when training", c2_trainFile);
  ah.new_named_string("c2_dev", "devCorpus_type2", "named_string", "development corpus of type2 to train a model, optional when training", c2_devFile);
  ah.new_named_string("c2_test", "testCorpus_type2", "named_string",
      "testing corpus of type2 to train a model or input file to test a model, optional when training and must when testing", c2_testFile);
  ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
  ah.new_named_string("basemodel", "basemodelFile", "named_string", "basemodel file, must when training and testing", basemodelFile);
  ah.new_named_string("word", "wordEmbFile", "named_string", "pretrained word embedding file to train a model, optional when training", wordEmbFile);
  ah.new_named_string("char", "charEmbFile", "named_string", "pretrained char embedding file to train a model, optional when training", charEmbFile);
  ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
  ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);

  ah.process(argc, argv);

  Labeler tagger;
  if (bTrain) {
    tagger.train(c1_trainFile, c1_devFile, c1_testFile,
        c2_trainFile, c2_devFile, c2_testFile, modelFile, optionFile, wordEmbFile, charEmbFile, basemodelFile);
  } else {
    tagger.test(c1_testFile, outputFile, modelFile, 1);
  }

  //test(argv);
  //ah.write_values(std::cout);
#if USE_CUDA==1
  ShutdownTensorEngine();
#else
  ShutdownTensorEngine<cpu>();
#endif
}
