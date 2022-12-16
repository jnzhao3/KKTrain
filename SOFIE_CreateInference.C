//
//  parse the model to produce the inference function and data.  Note
//  the output file name must end in .hxx, as the OutputGenerated
//  function assumes that when creating the associated .dat file
//  based on $ROOTSYS/tutorials/tmva/TMVA_SOFIE_Keras_HiggsModel.C
//  Note this needs to be run with root 6.26/10 or later to produce
//  ordered data and code
//
using namespace TMVA::Experimental;

void SOFIE_CreateInference(const char* modelFile="TrainBkg.h5"){

  //Parsing the saved Keras .h5 file into RModel object
  SOFIE::RModel model = SOFIE::PyKeras::Parse(modelFile);
  //    model.Initialize(1); // overwrite batch size.  Wait for root 6.28

  //Generating inference code
  model.Generate();
  TString modelHeaderFile = modelFile;
  modelHeaderFile.ReplaceAll(".h5",".hxx");
  model.OutputGenerated(std::string(modelHeaderFile));
  cout << "Saved inference function as " << modelHeaderFile << endl;

}
