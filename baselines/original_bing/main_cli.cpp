// Command-line entry point for the authors' BING code (torrvision/Objectness).
// Replaces Src/main.cpp so the dataset folder and settings come from the command
// line instead of a hard-coded path. Everything else is the unmodified upstream code.
//
//   BING_linux <VOC-style dataset dir ending in '/'> [numPerSz=130] [base=2] [W=8] [NSS=2]
//
// Trains stage I + II on ImageSets/Main/train.txt, predicts proposals for test.txt,
// writes Results/BBoxesB<base>W<W>MAXBGR/<id>.txt and prints the average time per image.
#include "kyheader.h"
#include "Objectness.h"
#include "ValStructVec.h"
#include "CmShow.h"

int main(int argc, char* argv[])
{
    if (argc < 2) {
        printf("usage: %s <dataset dir/> [numPerSz] [base] [W] [NSS]\n", argv[0]);
        return 1;
    }
    string dir = argv[1];
    int numPerSz = argc > 2 ? atoi(argv[2]) : 130;
    double base = argc > 3 ? atof(argv[3]) : 2;
    int W = argc > 4 ? atoi(argv[4]) : 8;
    int NSS = argc > 5 ? atoi(argv[5]) : 2;

    srand(0);
    DataSetVOC voc(dir);
    voc.loadAnnotations();
    printf("Dataset: `%s' with %d training and %d testing\n", _S(voc.wkDir), voc.trainNum, voc.testNum);
    printf("Base = %g, W = %d, NSS = %d, perSz = %d\n", base, W, NSS, numPerSz);

    Objectness objNess(voc, base, W, NSS);
    vector<vector<Vec4i>> boxesTests;
    objNess.getObjBndBoxesForTestsFast(boxesTests, numPerSz);
    return 0;
}
