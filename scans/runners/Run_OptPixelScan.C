// Run_OptPixelScan.C
// Runner macro: compiles/loads ParameterScan_OptPixel_Lib.C and runs the scan.

#include <TROOT.h>
#include <vector>
#include <string>

// --- FORWARD DECLARATION ---
// This tells ROOT "this function exists" so it doesn't complain before loading the lib.
void RunOptPixelScan(
    double r1_min_cm, double r1_max_cm, double rN_min_cm, double rN_max_cm,
    double step_cm, double minSpacing_cm, int samplesPerConfig,
    const std::vector<int>& layerCounts, double Bfield_T, unsigned int rngSeed,
    const std::string& csvOutPath, const std::string& geomDirPath, bool doLoadAll
);

void Run_OptPixelScan() {
  double r1_min_cm = 1.0, r1_max_cm = 5.0;
  double rN_min_cm = 10.0, rN_max_cm = 20.0;
  double step_cm   = 0.2;

  double minSpacing_cm = 0.5;
  int samplesPerConfig = 50;

  std::vector<int> layerCounts = {3, 4, 5, 6};

  double Bfield_T = 2.0;
  unsigned int rngSeed = 12345;

  std::string csvOut  = "results/csvs/optpixel_scan.csv";
  std::string geomDir = "results/geometries/optpixel";

  bool doLoadAll = true;

  RunOptPixelScan(
    r1_min_cm, r1_max_cm,
    rN_min_cm, rN_max_cm,
    step_cm,
    minSpacing_cm,
    samplesPerConfig,
    layerCounts,
    Bfield_T,
    rngSeed,
    csvOut,
    geomDir,
    doLoadAll
  );
}