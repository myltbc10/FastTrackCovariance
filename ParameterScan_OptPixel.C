// ParameterScan_OptPixel.C
// Grid over (r1, rN), Monte Carlo sample intermediate radii for each N (3..6),
// generate geometry files, evaluate sigma(d0) at 1 GeV and sigma(pT)/pT at 100 GeV.
//
// Run:
//   root -l
//   .L ParameterScan_OptPixel.C+
//   ParameterScan_OptPixel();
//
// Output:
//   results/optpixel_scan.csv
//   geometry_files/optpixel/*.txt

#include <TROOT.h>
#include <TSystem.h>
#include <TRandom3.h>
#include <TString.h>
#include <TMath.h>

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <algorithm>

// Pull in your project headers directly (since you have them locally)
#include "geometry_scripts/SolGeom.h"
#include "trkcovariance_scripts/SolTrack.h"

static std::string JoinRadii(const std::vector<double>& r_cm) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);
  for (size_t i = 0; i < r_cm.size(); i++) {
    if (i) oss << ";";
    oss << r_cm[i];
  }
  return oss.str();
}

// Sample N radii between r1 and rN in cm with min spacing (cm).
// Uses random gap weights so constraints are always satisfied.
static std::vector<double> SampleRadiiCM(double r1_cm, double rN_cm, int N,
                                         double minSpacing_cm, TRandom3& rng) {
  std::vector<double> r;
  if (N < 2) return r;

  const double span = rN_cm - r1_cm;
  const double minTotal = minSpacing_cm * (N - 1);
  if (span < minTotal) return r;

  const double extra = span - minTotal;

  // Draw positive weights w_i ~ Exp(1), normalize
  std::vector<double> w(N - 1, 0.0);
  double wsum = 0.0;
  for (int i = 0; i < N - 1; i++) {
    w[i] = -std::log(std::max(1e-12, rng.Uniform()));
    wsum += w[i];
  }
  if (wsum <= 0) return r;

  std::vector<double> gaps(N - 1, minSpacing_cm);
  for (int i = 0; i < N - 1; i++) gaps[i] += extra * (w[i] / wsum);

  r.resize(N);
  r[0] = r1_cm;
  for (int i = 1; i < N; i++) r[i] = r[i - 1] + gaps[i - 1];

  // Force exact endpoint
  r.back() = rN_cm;

  // Safety check
  for (int i = 1; i < N; i++) {
    if ((r[i] - r[i - 1]) + 1e-9 < minSpacing_cm) { r.clear(); return r; }
    if (!(r[i] > r[i - 1])) { r.clear(); return r; }
  }
  return r;
}

// Write a geometry file compatible with SolGeom::GeoRead parsing.
//
// Format per line:
// tyLay label xMin xMax rPos thLay rlLay nmLay stLayU stLayL sgLayU sgLayL flLay
//
// IMPORTANT: SolGeom uses METERS.
static bool WriteOptPixelGeometryFile(const TString& outPath,
                                      const std::vector<double>& radii_cm) {
  std::ofstream f(outPath.Data());
  if (!f.is_open()) return false;

  // Project specs:
  // half-length: 25 cm
  // hit resolution: 10 um
  // material: 1% X0 per layer
  const double halfLen_m = 0.25;            // 25 cm
  const double rlLay_m   = 9.370e-2;        // Si X0 ~ 9.37 cm in meters
  const double thLay_m   = 0.01 * rlLay_m;  // 1% X0 thickness
  const double sgU_m     = 10e-6;           // 10 um
  const double sgL_m     = 10e-6;           // 10 um

  const int    nmLay     = 2;               // measure (u,l)
  const double stU       = 0.0;
  const double stL       = TMath::Pi() / 2.0;
  const int    flLay     = 1;               // measurement layer
  const int    tyLay     = 1;               // barrel cylinder
  const char*  label     = "VTXBARREL";

  f << std::setprecision(10);

  for (double r_cm : radii_cm) {
    const double r_m = r_cm * 1e-2;
    f << tyLay << " " << label << " "
      << (-halfLen_m) << " " << (halfLen_m) << " "
      << r_m << " "
      << thLay_m << " " << rlLay_m << " "
      << nmLay << " "
      << stU << " " << stL << " "
      << sgU_m << " " << sgL_m << " "
      << flLay
      << "\n";
  }

  f.close();
  return true;
}

// Evaluate objectives for one geometry file.
// - d0@1GeV in microns from SolTrack::s_D() (meters -> um)
// - (sigma(pt)/pt)@100GeV in percent from SolTrack::s_pt() (dimensionless -> %)
static bool EvaluateGeometryObjectives(const TString& geomFile,
                                       double Bfield_T,
                                       double& out_d0_1GeV_um,
                                       double& out_pt_100GeV_percent) {
  SolGeom* G = new SolGeom((char*)geomFile.Data(), Bfield_T);

  Double_t x[3] = {0.0, 0.0, 0.0};

  // Central tracks => theta = pi/2 => pz = 0
  auto fill_p = [&](double pt, Double_t p[3]) {
    p[0] = pt;   // px
    p[1] = 0.0;  // py
    p[2] = 0.0;  // pz
  };

  // pt = 1 GeV
  {
    Double_t p[3];
    fill_p(1.0, p);
    SolTrack* tr = new SolTrack(x, p, G);
    tr->CovCalc(kTRUE, kTRUE);
    out_d0_1GeV_um = 1e6 * tr->s_D(); // m -> um
    delete tr;
  }

  // pt = 100 GeV
  {
    Double_t p[3];
    fill_p(100.0, p);
    SolTrack* tr = new SolTrack(x, p, G);
    tr->CovCalc(kTRUE, kTRUE);
    out_pt_100GeV_percent = 100.0 * tr->s_pt(); // -> %
    delete tr;
  }

  delete G;

  return std::isfinite(out_d0_1GeV_um) && std::isfinite(out_pt_100GeV_percent);
}

void ParameterScan_OptPixel() {
  // Keep your existing workflow available (harmless if already loaded)
  gROOT->ProcessLine(".L LoadAll.c");
  gROOT->ProcessLine("LoadAll(\"\")");

  gSystem->mkdir("results", kTRUE);
  gSystem->mkdir("geometry_files/optpixel", kTRUE);

  // Scan settings (cm)
  const double r1_min = 1.0, r1_max = 5.0;
  const double rN_min = 10.0, rN_max = 20.0;
  const double step   = 0.2;

  const double minSpacing = 0.5;  // cm
  const int samplesPerN   = 50;   // 50 Monte Carlo samples per N per (r1,rN)

  const double Bfield_T = 2.0;

  TRandom3 rng(12345);

  std::ofstream out("results/optpixel_scan.csv");
  out << "r1_cm,rN_cm,N,radii_cm,d0_1GeV_um,pt_100GeV_percent,geom_file\n";
  out << std::fixed << std::setprecision(6);

  long long nTried = 0, nLogged = 0;

  double best_d0 = 1e99; TString best_d0_file = "";
  double best_pt = 1e99; TString best_pt_file = "";

  for (double r1 = r1_min; r1 <= r1_max + 1e-9; r1 += step) {
    for (double rN = rN_min; rN <= rN_max + 1e-9; rN += step) {

      // If even N=3 can't fit, skip fast
      if ((rN - r1) < minSpacing * (3 - 1)) continue;

      for (int N = 3; N <= 6; N++) {

        // Skip infeasible N for this (r1, rN)
        if ((rN - r1) < minSpacing * (N - 1)) continue;

        for (int s = 0; s < samplesPerN; s++) {
          nTried++;

          auto radii_cm = SampleRadiiCM(r1, rN, N, minSpacing, rng);
          if (radii_cm.empty()) continue;

          // Build safe filename WITHOUT destroying ".txt"
          TString base;
          base.Form("OptPix_r1_%0.1f_rN_%0.1f_N%d_s%03d", r1, rN, N, s);
          base.ReplaceAll(".", "p");
          const TString geomPath = TString("geometry_files/optpixel/") + base + ".txt";

          if (!WriteOptPixelGeometryFile(geomPath, radii_cm)) {
            std::cerr << "Failed to write: " << geomPath << "\n";
            continue;
          }

          double d0_um = NAN, pt_pct = NAN;
          if (!EvaluateGeometryObjectives(geomPath, Bfield_T, d0_um, pt_pct)) {
            std::cerr << "Failed to eval: " << geomPath << "\n";
            continue;
          }

          out << r1 << "," << rN << "," << N << ",\""
              << JoinRadii(radii_cm) << "\"," << d0_um << "," << pt_pct << ","
              << geomPath.Data() << "\n";

          nLogged++;

          bool keep = false;

          if (d0_um < best_d0) {
            best_d0 = d0_um;
            best_d0_file = geomPath;
            keep = true;
          }

          if (pt_pct < best_pt) {
            best_pt = pt_pct;
            best_pt_file = geomPath;
            keep = true;
          }

          if (!keep) {
            gSystem->Unlink(geomPath);
          }


          if (d0_um < best_d0) { best_d0 = d0_um; best_d0_file = geomPath; }
          if (pt_pct < best_pt) { best_pt = pt_pct; best_pt_file = geomPath; }

          if (nLogged % 1000 == 0) {
            std::cout << "[Progress] logged=" << nLogged
                      << " tried=" << nTried
                      << " last r1=" << r1 << " rN=" << rN
                      << " N=" << N
                      << " d0=" << d0_um << "um"
                      << " pt=" << pt_pct << "%\n";
          }
        }
      }
    }
  }

  out.close();

  std::cout << "\n=== Scan complete ===\n";
  std::cout << "Tried:  " << nTried  << "\n";
  std::cout << "Logged: " << nLogged << "\n";
  std::cout << "CSV:    results/optpixel_scan.csv\n";
  std::cout << "Best d0@1GeV:   " << best_d0 << " um  @ " << best_d0_file << "\n";
  std::cout << "Best pt@100GeV: " << best_pt << " %   @ " << best_pt_file << "\n";
}
