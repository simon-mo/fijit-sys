#ifndef ALIGN_MAP_H
#define ALIGN_MAP_H

#include <vector>
#include <map>
#include <string>

using namespace std;

constexpr int NoopNode = -1;
using PaddedNodeSeq = vector<int>;
using AlignSolution = map<string, PaddedNodeSeq>;

class AlignmentDB {
public:
    AlignmentDB(string path);
    AlignSolution get_align(vector<string> models);
private:
    string path_;
};

#endif /* ALIGN_MAP_H */
