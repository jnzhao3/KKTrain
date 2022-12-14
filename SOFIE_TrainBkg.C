/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro provides a simple example for the parsing of Keras .h5 file
/// into RModel object and further generating the .hxx header files for inference.
///
/// \macro_code
/// \macro_output
/// \author Sanjiban Sengupta

using namespace TMVA::Experimental;

void SOFIE_TrainBkg(){

    //Parsing the saved Keras .h5 file into RModel object
    SOFIE::RModel model = SOFIE::PyKeras::Parse("TrainBkg.h5");
//    model.Initialize(1); // overwrite batch size

    //Generating inference code
    model.Generate();
//    cout << "Saving model as TrainBkg.cc" << endl;
//    model.OutputGenerated("TrainBkg.cc");
    model.OutputGenerated("TrainBkg.hxx");

   //Printing required input tensors
   std::cout<<"\n\n";
   model.PrintRequiredInputTensors();

   //Printing initialized tensors (weights)
   std::cout<<"\n\n";
   model.PrintInitializedTensors();

   //Printing intermediate tensors
   std::cout<<"\n\n";
   model.PrintIntermediateTensors();

   //Checking if tensor already exist in model
   std::cout<<"\n\nTensor \"dense2bias0\" already exist: "<<std::boolalpha<<model.CheckIfTensorAlreadyExist("dense2bias0")<<"\n\n";
   std::vector<size_t> tensorShape = model.GetTensorShape("dense2bias0");
   std::cout<<"Shape of tensor \"dense2bias0\": ";
   for(auto& it:tensorShape){
       std::cout<<it<<",";
   }
   std::cout<<"\n\nData type of tensor \"dense2bias0\": ";
   SOFIE::ETensorType tensorType = model.GetTensorType("dense2bias0");
   std::cout<<SOFIE::ConvertTypeToString(tensorType);

}
