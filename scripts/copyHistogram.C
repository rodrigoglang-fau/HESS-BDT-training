void copyHistogram(const char* f1, const char* f2, const char* f3, const char* f4) {

TFile *myfile1 = new TFile(f1,"READ");
TH1D *h = (TH1D*) myfile1->Get(f2);
TFile *myfile2 = new TFile(f3,"recreate");
h->SetName(f4);
h->Write();
gSystem->Exit(0);

return;

}
