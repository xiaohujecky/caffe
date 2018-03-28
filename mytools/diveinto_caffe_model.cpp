#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include "caffe/proto/caffe.pb.h"
#include "caffe/caffe.hpp"

using namespace std;
using namespace caffe;

int main(int argc, char* argv[]) 
{ 

 caffe::NetParameter net; 

 //fstream input("lenet_iter_10000.caffemodel", ios::in | ios::binary); 
 // Following code reading protofile at a limitted size.
 //fstream input(argv[1], ios::in | ios::binary);  
 //if (!net.ParseFromIstream(&input)) 
 if (!ReadProtoFromBinaryFile(argv[1], &net))
 { 
   cerr << "Failed to parse address book." << endl; 
   return -1; 
 } 
 LOG(INFO) << "name : " << net.mutable_name();
 LOG(INFO) << "Layer(repeated) size : " << net.layer_size();
 //printf("Repeated Size = %d\n", msg.layer_size());
 LOG(INFO) << "Input size : " << net.input_size();
 LOG(INFO) << "Input dim : " << net.mutable_input();

 ::google::protobuf::RepeatedPtrField< LayerParameter >* layer = net.mutable_layer();
 ::google::protobuf::RepeatedPtrField< LayerParameter >::iterator it = layer->begin();
 int layer_index = 0;
 for (; it != layer->end(); ++it){ 
     cout << "\nLayer : " << layer_index << ", -------------------------------------" << endl;
     ::google::protobuf::RepeatedPtrField< ::std::string > *bottom = it->mutable_bottom();
     ::google::protobuf::RepeatedPtrField< ::std::string >::iterator b_it = bottom->begin();
     for (; b_it != bottom->end(); ++b_it){
        LOG(INFO) << "bottom : " << *b_it;

     }
     ::google::protobuf::RepeatedPtrField< ::std::string > *top = it->mutable_top();
     ::google::protobuf::RepeatedPtrField< ::std::string >::iterator t_it = top->begin();
     for (; t_it != top->end(); ++t_it)
        LOG(INFO) << "top : " << *t_it; 
     ::google::protobuf::RepeatedPtrField< ::caffe::BlobProto > *blob_proto = it->mutable_blobs();
     ::google::protobuf::RepeatedPtrField< ::caffe::BlobProto >::iterator blob_it = blob_proto->begin();
     cout << "Blobs size : " << it->mutable_blobs()->size() << endl;
     for (; blob_it != blob_proto->end(); ++blob_it){
        //LOG(INFO) << "blob dim : " << blob_it->mutable_shape()->dim_size(); 
        cout << "Blob dim [num channels height width](size) : " << blob_it->num() <<" "
            <<blob_it->channels()<<" "<<blob_it->height()<<" "<<blob_it->width()<<" "<<blob_it->mutable_shape()->dim_size()<<endl; 
        ::caffe::Blob<float> blob;
        blob.FromProto(*blob_it, true);
        LOG(INFO) << "blob shape_ : " << blob.shape_string();
        LOG(INFO) << "blob count_ : " << blob.count();
        float *p_data = blob.mutable_cpu_data();
        float *p_diff = blob.mutable_cpu_diff();
        cout << "cpu data : " << endl; 
        int p_data_idx=0;
        for (auto p=0; p < blob.count(); p++){
           cout << *p_data++ << " ";  
           //if (++p_data_idx > blob.width()*blob.height()*blob.channels()){
           if (++p_data_idx > blob.width()){
                 p_data_idx=0;
                 cout<<endl;
                 break;
           }
        }
        //cout << "cpu diff : " << endl; 
        //for (auto p=0; p < blob.count(); p++){
        //   cout << *p_diff++ << " ";  
        //   if (++p_data_idx > 20){
        //         p_data_idx=0;
        //         cout<<endl;
        //   }
        //}
     }
   cout << "name : " << it->name() << endl;
   cout << "type : " << it->type() << endl;
   cout << "bottom size : " << it->bottom_size() << endl;
   cout << "top size : " << it->top_size() << endl;
   cout << "filler max weight : " << it->convolution_param().weight_filler().max() << endl;

   layer_index ++;
 } 

 return 0;
}
