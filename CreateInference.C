/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro parses a Keras .h5 file
/// into RModel object and further generating the .hxx header files for inference.
///
/// \macro_code
/// \macro_output
/// \author Sanjiban Sengupta

using namespace TMVA::Experimental;

void CreateInference(const char* kname){
    string modelname = string(kname) + string(".h5");
    string infername = string(kname) + string(".hxx");
    //Parsing the saved Keras .h5 file into RModel object
    SOFIE::RModel model = SOFIE::PyKeras::Parse(modelname);

    //Generating inference code
    model.Generate();
    model.OutputGenerated(infername);
}
